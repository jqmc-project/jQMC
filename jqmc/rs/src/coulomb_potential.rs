// Copyright (C) 2024- Kosuke Nakano
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//! 1:1 Rust port of the *molecular* (non-PBC) bare Coulomb value functions.
//!
//! Mirrors `jqmc/coulomb_potential.py`:
//!   - `compute_bare_coulomb_potential_el_el`          (py:2878)
//!   - `compute_bare_coulomb_potential_ion_ion`        (py:2934)
//!   - `compute_bare_coulomb_potential_el_ion_element_wise` (py:2698)
//!   - `compute_bare_coulomb_potential`                (py:2658, molecular branch)
//!
//! These are autograd-free, RNG-free pure value functions.
//! The PyO3 boundary here takes the already-extracted `R_carts` /
//! `_effective_charges` arrays rather than a full `CoulombPotentialData` struct;
//! data-struct marshalling through `from_py`. The math below is the closed form
//! the JAX path reduces to for
//! `pbc_flag == False`.

use crate::data::JastrowData;
use crate::determinant::{compute_geminal_all_elements, det_sign_lnabs, GeminalData, OrbData};
use numpy::ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// ECP (pseudopotential) tables, split local / non-local exactly as the jqmc
/// `Coulomb_potential_data` derived arrays (`_*_local_part` / `_*_non_local_part`)
/// that the `compute_ecp_*` kernels consume. Owned; borrowed by value.
pub struct EcpData {
    pub nuc_l: Vec<i64>,
    pub e_l: Vec<f64>,
    pub c_l: Vec<f64>,
    pub p_l: Vec<f64>,
    pub nuc_n: Vec<i64>,
    pub am: Vec<i64>,
    pub e_n: Vec<f64>,
    pub c_n: Vec<f64>,
    pub p_n: Vec<f64>,
}

impl EcpData {
    /// Read the split ECP arrays from a `Coulomb_potential_data`.
    pub fn from_py(cp: &Bound<'_, PyAny>) -> PyResult<Self> {
        let l =
            |a: &str| -> PyResult<Vec<f64>> { cp.getattr(a)?.call_method0("tolist")?.extract() };
        Ok(EcpData {
            nuc_l: cp.getattr("_nucleus_index_local_part")?.extract()?,
            e_l: l("_exponents_local_part")?,
            c_l: l("_coefficients_local_part")?,
            p_l: l("_powers_local_part")?,
            nuc_n: cp.getattr("_nucleus_index_non_local_part")?.extract()?,
            am: cp.getattr("_ang_mom_non_local_part")?.extract()?,
            e_n: l("_exponents_non_local_part")?,
            c_n: l("_coefficients_non_local_part")?,
            p_n: l("_powers_non_local_part")?,
        })
    }
}

/// Mirrors `jqmc.coulomb_potential.Coulomb_potential_data`,
/// owning what the value / e_L kernels need: the effective charges, the ECP
/// split tables ([`EcpData`], when `ecp_flag`), the periodic Ewald tables
/// (nested [`crate::ewald::EwaldData`], when PBC), and the cell / min-image
/// metadata. Built once per run via [`CoulombPotentialData::from_py`], the
/// one parse of these fields that the native driver and the lib.rs wrappers share.
pub struct CoulombPotentialData {
    pub positions: Array2<f64>,
    pub effective_charges: Vec<f64>,
    pub pbc: bool,
    pub cell: Array2<f64>,
    pub n_max: i32,
    pub ecp: Option<EcpData>,
    pub ewald: Option<crate::ewald::EwaldData>,
}

impl CoulombPotentialData {
    pub fn from_py(cp: &Bound<'_, PyAny>) -> PyResult<Self> {
        let sd = cp.getattr("structure_data")?;
        let np = cp.py().import("numpy")?;
        let as_arr2 = |o: Bound<'_, PyAny>| -> PyResult<Array2<f64>> {
            Ok(np
                .call_method1("asarray", (o,))?
                .extract::<numpy::PyReadonlyArray2<f64>>()?
                .as_array()
                .to_owned())
        };
        let pbc: bool = sd.getattr("pbc_flag")?.extract()?;
        let n_max: i32 = if pbc {
            sd.getattr("_n_max")?.extract()?
        } else {
            0
        };
        let cell = if pbc {
            as_arr2(sd.getattr("cell")?)?
        } else {
            Array2::<f64>::zeros((3, 3))
        };
        let ecp = if cp.getattr("ecp_flag")?.extract::<bool>()? {
            Some(EcpData::from_py(cp)?)
        } else {
            None
        };
        Ok(CoulombPotentialData {
            positions: as_arr2(sd.getattr("positions")?)?,
            effective_charges: cp
                .getattr("_effective_charges")?
                .call_method0("tolist")?
                .extract()?,
            pbc,
            cell,
            n_max,
            ecp,
            ewald: crate::ewald::EwaldData::from_py(cp)?,
        })
    }
}

/// Legendre polynomial `P_l(x)` via the standard recurrence (l small: s/p/d/f).
#[inline]
pub(crate) fn legendre(l: i64, x: f64) -> f64 {
    match l {
        0 => 1.0,
        1 => x,
        _ => {
            let (mut p0, mut p1) = (1.0_f64, x);
            for n in 1..l {
                let nn = n as f64;
                let p2 = ((2.0 * nn + 1.0) * x * p1 - nn * p0) / (nn + 1.0);
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Tabulated Legendre derivative `P_l'(x)` for `l = 0..6`, a polynomial valid on the
/// closed interval `[-1, 1]` (NaN outside `0..=6`, like the XLA
/// `jqmc.derivatives.ecp_force._legendre_derivative_tablated`).
///
/// The recurrence form `l (P_{l-1} - x P_l) / (1 - x^2)` is 0/0 at `x = +-1`, which is
/// reached whenever an electron lies exactly on a quadrature direction of the current
/// mesh rotation (e.g. right after a GFMC T-move / DLT-move onto a mesh point), and it
/// loses digits by cancellation next to it for `l >= 2`.
#[inline]
pub(crate) fn legendre_deriv(l: i64, x: f64) -> f64 {
    match l {
        0 => 0.0,
        1 => 1.0,
        2 => 3.0 * x,
        3 => 0.5 * (15.0 * x * x - 3.0),
        4 => 0.5 * (35.0 * x * x * x - 15.0 * x),
        5 => 0.125 * (315.0 * x.powi(4) - 210.0 * x * x + 15.0),
        6 => 0.125 * (693.0 * x.powi(5) - 630.0 * x.powi(3) + 105.0 * x),
        _ => f64::NAN,
    }
}

/// Full Jastrow value `J = J1 + J2 + J3 + J_DTN` evaluated in the `jastrow_ratio` zone.
///
/// Every caller is a full-recompute reference / fallback route that forms the ratio
/// `exp(J(R') - J(R))` from two of these values (ECP mesh references and fallbacks,
/// discretized-kinetic references, mesh force / strain fallbacks, debug oracles). The
/// precision policy assigns that ratio to `jastrow_ratio` (float64 in both modes), so
/// each component, its coefficients and the sum are evaluated in that zone, never in
/// `jastrow_eval` (float32 in mixed), whose rounding of `|J|` the difference would keep
/// (issue #781). In full precision this equals the `jastrow_eval` evaluation bit
/// for bit.
#[inline]
pub(crate) fn jastrow_value_for_ratio(
    j: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    const ZONE: &str = "jastrow_ratio";
    let mut v = 0.0;
    let f32_arithmetic = j.precision_policy.is_f32(ZONE);
    let add = |acc: f64, x: f64| -> f64 {
        if f32_arithmetic {
            ((acc as f32) + (x as f32)) as f64
        } else {
            acc + x
        }
    };
    if let Some(ob) = &j.one_body {
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_one_body_in_zone(ob, ZONE, r_up, r_dn),
        );
    }
    if let Some(tb) = &j.two_body {
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_two_body_in_zone(tb, ZONE, r_up, r_dn),
        );
    }
    if let Some((orb, jm)) = j3 {
        // Home-cell-wrap for the periodic J3 orbital value, matching jqmc
        // `compute_AOs`: the batch pGTO tiling underflows for a
        // walker roamed out of the home cell -> a wrong J3 base/mesh ratio in the
        // ECP non-local force. Bit-exact identity in-cell; `None` for molecular /
        // PW / B-spline.
        let ruw = orb.wrap_electrons(r_up).unwrap_or_else(|| r_up.to_owned());
        let rdw = orb.wrap_electrons(r_dn).unwrap_or_else(|| r_dn.to_owned());
        let au = orb.compute(&ruw.view());
        let ad = orb.compute(&rdw.view());
        v = add(
            v,
            crate::jastrow_factor::compute_jastrow_three_body_value_with_policy(
                jm,
                &au.view(),
                &ad.view(),
                f32_arithmetic,
            ),
        );
    }
    if let Some(dtn) = &j.dtn {
        v = add(
            v,
            crate::dtn::compute_dtn_value_in_zone(dtn, ZONE, r_up, r_dn),
        );
    }
    v
}

/// `r^q` for the ECP radial factor `r^(power-2)`: the ccECP/BFD tables use
/// integer powers, so `q` is mostly 0 or 1. `pow` (< 1 ULP) returns an exactly
/// representable result exactly, hence `pow(r, 0) == 1` and `pow(r, 1) == r`
/// bit for bit; skipping the libm call for those is value-identical.
#[inline]
/// J1/J2 ratios of the ECP quadrature mesh of one electron (empty when the
/// caller needs the determinant ratio only), via
/// [`crate::jastrow_factor::jastrow_1b2b_ratio_mesh`], or via the FAST periodic
/// twin when the caller (cutoff mode only) built one.
#[allow(clippy::too_many_arguments)]
fn j12_mesh_ratios(
    jastrow: &JastrowData,
    j12_fast_prep: Option<&crate::jastrow_factor::J12FastPeriodic>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    i: usize,
    mesh: &ArrayView2<f64>,
    det_only: bool,
    nv: usize,
) -> Vec<f64> {
    if det_only {
        return Vec::new();
    }
    let mut o = vec![0.0_f64; nv];
    crate::timed!(
        "ecp.j12",
        match j12_fast_prep {
            Some(prep) => prep.ratio_mesh(r_up, r_dn, is_up, i, mesh, &mut o),
            None => crate::jastrow_factor::jastrow_1b2b_ratio_mesh(
                jastrow, r_up, r_dn, is_up, i, mesh, &mut o
            ),
        }
    );
    o
}

pub(crate) fn ecp_pow(r: f64, q: f64) -> f64 {
    if q == 0.0 {
        1.0
    } else if q == 1.0 {
        r
    } else {
        r.powf(q)
    }
}

/// Local ECP energy `compute_ecp_local_parts_all_pairs` (coulomb_potential.py),
/// molecular branch. For each local term `t` (nucleus `i_atom[t]`, exponent,
/// coefficient, power) and each electron at distance `r` from that nucleus:
/// `V_l = coef * r^(power-2) * exp(-exponent * r^2)`. Summed over terms x electrons.
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_local_parts_all_pairs(
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> f64 {
    let inv = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    // |R - r_e|^2 (Wigner-Seitz minimum image under PBC).
    let dist2 = |r: &ArrayView2<f64>, el: usize, cx: f64, cy: f64, cz: f64| -> f64 {
        let d0 = [cx - r[[el, 0]], cy - r[[el, 1]], cz - r[[el, 2]]];
        let d = match (cell_nmax, &inv) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image(d0, cell, iv, n),
            _ => d0,
        };
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let mut v = 0.0_f64;
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let (cx, cy, cz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        let (e, c, p) = (exponents[t], coefficients[t], powers[t]);
        for el in 0..r_up.nrows() {
            let r2 = dist2(r_up, el, cx, cy, cz);
            v += c * ecp_pow(r2.sqrt(), p - 2.0) * (-e * r2).exp();
        }
        for el in 0..r_dn.nrows() {
            let r2 = dist2(r_dn, el, cx, cy, cz);
            v += c * ecp_pow(r2.sqrt(), p - 2.0) * (-e * r2).exp();
        }
    }
    v
}

/// Local-ECP contribution of one electron: `Sum_t coef_t * r^(power_t-2) *
/// exp(-exp_t r^2)` over the local channel terms, with the same Wigner-Seitz
/// minimum image as [`compute_ecp_local_parts_all_pairs`]. Two GFMC callers:
/// the stateful chain caches this scalar for every electron and refreshes only
/// the row selected by the hop, and the incremental projection interval carries
/// per-electron local-ECP
/// rows summed electron-major (vs the reference's channel-major total), an fp
/// reordering bounded by the interval twin test.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ecp_local_one(
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    r: [f64; 3],
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> f64 {
    let inv = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    let mut v = 0.0_f64;
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let d0 = [
            positions[[ia, 0]] - r[0],
            positions[[ia, 1]] - r[1],
            positions[[ia, 2]] - r[2],
        ];
        let d = match (cell_nmax, &inv) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image(d0, cell, iv, n),
            _ => d0,
        };
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        v += coefficients[t] * ecp_pow(r2.sqrt(), powers[t] - 2.0) * (-exponents[t] * r2).exp();
    }
    v
}

/// Euclidean distance between row `i` of `a` and row `j` of `b`.
///
/// Matches `jnp.linalg.norm(r_i - r_j)`. The subtraction is done in f64,
/// mirroring the Python "reconstruct `r_i - r_j` in fp64" note that guards
/// against catastrophic cancellation.
#[inline]
fn row_distance(a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize) -> f64 {
    let mut acc = 0.0_f64;
    for k in 0..a.ncols() {
        let d = a[[i, k]] - b[[j, k]];
        acc += d * d;
    }
    acc.sqrt()
}

/// Fast-path gate for the electron-fixed w-dot mesh contraction (cutoff mode
/// only): pure MO-projected GTO orbital with the flat shell tables. PW /
/// B-spline / bare-AO fall back to the batched orb-mesh path.
pub(crate) fn ecp_wdot_ok(orb: &OrbData, cutoff: bool) -> bool {
    cutoff
        && orb.pw.is_none()
        && orb.bspline.is_none()
        && matches!(
            orb.aos.kind,
            crate::atomic_orbital::AoKind::Sphe | crate::atomic_orbital::AoKind::Cart
        )
        && !orb.aos.shells.is_empty()
        && orb.aos.num_ao > 0
        && orb.mo_coefficients.is_some()
}

/// J3 analogue of [`ecp_wdot_ok`]. Unlike the determinant path, a J3 orbital
/// may be either a raw AO basis or an MO projection; both can be folded into a
/// raw-AO contraction vector before entering the quadrature loop.
pub(crate) fn ecp_j3_wdot_ok(orb: &OrbData, cutoff: bool) -> bool {
    cutoff
        && orb.pw.is_none()
        && orb.bspline.is_none()
        && matches!(
            orb.aos.kind,
            crate::atomic_orbital::AoKind::Sphe | crate::atomic_orbital::AoKind::Cart
        )
        && !orb.aos.shells.is_empty()
        && orb.aos.num_ao > 0
}

/// Fold the J3 single-move exponent into one raw-AO weight row per electron.
/// From `j3_ratio`, symmetry gives
///
/// `Delta J3 = (AO_new - AO_old[:,i]) .
///   (j1 + cross_other + rowsum(W_same) - W_same[:,i])`.
///
/// The parenthesized vector is fixed for an electron throughout one local
/// energy evaluation. Project it through C once (when the J3 basis is MO) and
/// the ECP mesh needs only `AO_raw(mesh).weight - base`, avoiding both the full
/// AO mesh and the dense J3 ratio matvec at every quadrature point.
pub(crate) fn ecp_j3_wdot_all(
    orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    state: &crate::jastrow_factor::J3RatioState,
    is_up: bool,
) -> (Array2<f64>, Vec<f64>) {
    let n_orb = j_matrix.nrows();
    let (aos, same, cross_same, cross_other) = if is_up {
        (&state.aos_up, &state.w_up, &state.cross_up, &state.cross_dn)
    } else {
        (&state.aos_dn, &state.w_dn, &state.cross_dn, &state.cross_up)
    };
    let ne = aos.ncols();
    let mut q = Array2::<f64>::zeros((ne, n_orb));
    let mut base = vec![0.0_f64; ne];
    // `cross_same` is maintained as rowsum(W_same) by the streaming J3 state.
    // Keep the orbital-major traversal: W_same is row-major, and its contiguous
    // reads matter more here than the strided writes to the much smaller q table.
    for a in 0..n_orb {
        let common = j_matrix[[a, n_orb]] + cross_other[a] + cross_same[a];
        for i in 0..ne {
            let qa = common - same[[a, i]];
            q[[i, a]] = qa;
            base[i] += aos[[a, i]] * qa;
        }
    }
    match &orb.mo_coefficients {
        Some(c) => {
            debug_assert_eq!(c.nrows(), n_orb);
            (crate::determinant::matmul(&q.view(), &c.view()), base)
        }
        None => {
            debug_assert_eq!(orb.aos.num_ao, n_orb);
            (q, base)
        }
    }
}

/// ALL-electron contraction vectors for the ECP-nl mesh det ratios, UP moves
/// (fast cutoff path). Folding the electron-fixed factors first turns each mesh
/// det ratio into one raw-AO dot: `dr_v = 1 + AO_raw(mesh_v).W[i,:] - c_base[i]`,
/// with `T = [paired_dn | lambda_unpaired].G^-1` (orb_num, N), `W = T^T.C_up`
/// (N, num_ao) and `c_base[i] = ao_up[:,i].T[:,i]`. Batched over all electrons as
/// TWO faer GEMMs so `C` is streamed from DRAM once per e_L (the per-electron
/// GEMV form re-read the 1.4MB `C^T` per electron -- memory-bound). Same bilinear
/// form as `compute_v_up_move_batch` + the `v.G^-1` loop, fp-reordered, so the
/// caller gates it to the cutoff (fast, floor-compared) mode.
pub(crate) fn ecp_wdot_all_up(
    geminal: &GeminalData,
    st: &crate::determinant::DetRatioState,
    ginv: &ArrayView2<f64>,
) -> (Array2<f64>, Vec<f64>) {
    use crate::determinant::matmul;
    let orb_num_up = st.ao_up.nrows();
    let n_dn_p = st.paired_dn.ncols();
    let n_unp = geminal.n_unpaired();
    let n_tot = n_dn_p + n_unp;
    debug_assert_eq!(n_tot, ginv.nrows());
    // M = [paired_dn | lambda_unpaired]  (orb_num_up, N) -- the same augmented
    // matrix the kinetic's `mg` uses.
    let mut m = Array2::<f64>::zeros((orb_num_up, n_tot));
    // SD stores the paired table compactly with only `orb_num_dn` rows; its
    // excess-up occupied rows belong exclusively to the unpaired block.
    for o in 0..st.paired_dn.nrows() {
        for j in 0..n_dn_p {
            m[[o, j]] = st.paired_dn[[o, j]];
        }
    }
    for o in 0..orb_num_up {
        for j in 0..n_unp {
            m[[o, n_dn_p + j]] = geminal.lambda_value(o, geminal.orb_num_dn + j);
        }
    }
    let t_all = matmul(&m.view(), ginv); // (orb_num_up, N)
    let n_e = t_all.ncols();
    let mut c_base = vec![0.0_f64; n_e];
    for o in 0..orb_num_up {
        for (i, cb) in c_base.iter_mut().enumerate() {
            *cb += st.ao_up[[o, i]] * t_all[[o, i]];
        }
    }
    let c = geminal
        .orb_up
        .mo_coefficients
        .as_ref()
        .expect("ecp_wdot_all_up: MO required");
    let w_t = matmul(&t_all.t(), &c.view()); // (N, num_ao), row i contiguous
    (w_t, c_base)
}

/// As [`ecp_wdot_all_up`] for DN moves:
/// `dr_v = 1 + AO_raw(mesh_v).W[i,:] - c_base[i]`, `T = G^-1.paired_up_lambda`
/// (N, orb_num_dn), `W = T.C_dn` (N, num_ao), `c_base[i] = ao_dn[:,i].T[i,:]`.
pub(crate) fn ecp_wdot_all_dn(
    geminal: &GeminalData,
    st: &crate::determinant::DetRatioState,
    ginv: &ArrayView2<f64>,
) -> (Array2<f64>, Vec<f64>) {
    use crate::determinant::matmul;
    let t_all = matmul(ginv, &st.paired_up_lambda.view()); // (N, orb_num_dn)
    let n_e = st.ao_dn.ncols(); // dn ELECTRON count (<= N rows of t_all)
    let orb_num_dn = t_all.ncols();
    let mut c_base = vec![0.0_f64; n_e];
    for (i, cb) in c_base.iter_mut().enumerate() {
        let mut s = 0.0;
        for o in 0..orb_num_dn {
            s += st.ao_dn[[o, i]] * t_all[[i, o]];
        }
        *cb = s;
    }
    let c = geminal
        .orb_dn
        .mo_coefficients
        .as_ref()
        .expect("ecp_wdot_all_dn: MO required");
    let w_t = matmul(&t_all.view(), &c.view()); // (N, num_ao)
    (w_t, c_base)
}

/// MO-space form of [`ecp_wdot_all_up`] for a projection-scoped candidate
/// table. Candidate MO values are cached across all hops in the interval, so
/// folding through the large MO->AO coefficient matrix on every hop would be
/// redundant. The returned rows contract directly with cached MO candidates.
pub(crate) fn det_mowdot_all_up(
    geminal: &GeminalData,
    st: &crate::determinant::DetRatioState,
    ginv: &ArrayView2<f64>,
) -> (Array2<f64>, Vec<f64>) {
    use crate::determinant::matmul;
    let orb_num_up = st.ao_up.nrows();
    let n_dn_p = st.paired_dn.ncols();
    let n_unp = geminal.n_unpaired();
    let n_tot = n_dn_p + n_unp;
    let mut m = Array2::<f64>::zeros((orb_num_up, n_tot));
    for o in 0..st.paired_dn.nrows() {
        for j in 0..n_dn_p {
            m[[o, j]] = st.paired_dn[[o, j]];
        }
    }
    for o in 0..orb_num_up {
        for j in 0..n_unp {
            m[[o, n_dn_p + j]] = geminal.lambda_value(o, geminal.orb_num_dn + j);
        }
    }
    let t_all = matmul(&m.view(), ginv);
    let n_e = t_all.ncols();
    let mut c_base = vec![0.0_f64; n_e];
    for o in 0..orb_num_up {
        for (i, cb) in c_base.iter_mut().enumerate() {
            *cb += st.ao_up[[o, i]] * t_all[[o, i]];
        }
    }
    (t_all.t().as_standard_layout().to_owned(), c_base)
}

/// MO-space down-spin counterpart of [`det_mowdot_all_up`].
pub(crate) fn det_mowdot_all_dn(
    _geminal: &GeminalData,
    st: &crate::determinant::DetRatioState,
    ginv: &ArrayView2<f64>,
) -> (Array2<f64>, Vec<f64>) {
    use crate::determinant::matmul;
    let t_all = matmul(ginv, &st.paired_up_lambda.view());
    let n_e = st.ao_dn.ncols();
    let orb_num_dn = t_all.ncols();
    let mut c_base = vec![0.0_f64; n_e];
    for (i, cb) in c_base.iter_mut().enumerate() {
        let mut s = 0.0;
        for o in 0..orb_num_dn {
            s += st.ao_dn[[o, i]] * t_all[[i, o]];
        }
        *cb = s;
    }
    (t_all, c_base)
}

/// FAST-mode radial cut of the ECP non-local quadrature, in Ha (mirrors
/// `jqmc._setting.FAST_ECP_NONLOCAL_RADIAL_CUT`; pinned by
/// `tests::fast_ecp_nonlocal_radial_cut_pinned_to_setting`).
///
/// On the FAST route only (`cutoff` with the Sherman-Morrison inverse and the
/// determinant ratio state, i.e. the native production chunk), an (electron,
/// non-local term) pair whose radial factor `|v_t(r)| = |c r^(p-2) exp(-e r^2)|`
/// is below this bound skips the whole 12-point quadrature (mesh build, det
/// w-dot, J1/J2, J3 and DTN mesh ratios). This evaluates exactly a modified
/// Hamiltonian H' whose semilocal radial functions are truncated where
/// `|v_t| < EPS`, so pointwise `|Delta v_t| <= EPS`; with the rotation-averaged
/// quadrature equal to the angular projector (norm <= 1), `||H - H'|| <=
/// N_e * n_t * EPS` (n_t = non-local terms per nucleus) and every energy
/// expectation (VMC, fixed-node) moves by at most that. Water 20 (160 e, one
/// non-local term per O): <= 1.6e-10 Ha. The per-sample local energy change is
/// NOT bounded by EPS (the mesh psi ratio is unbounded near nodes). The exact
/// (non-FAST) route never cuts.
pub(crate) const FAST_ECP_NONLOCAL_RADIAL_CUT: f64 = 1.0e-12;

/// Non-local ECP energy `compute_ecp_non_local_parts_all_pairs` (coulomb_potential.py),
/// molecular. For each non-local term (nucleus, ang-mom `l`, radial V_l) and each
/// electron, integrate the single-electron wavefunction ratio over a quadrature
/// sphere weighted by `(2l+1)*P_l(costheta)*w_v`. `grid_points` are pre-rotated by RT.
///
/// On the FAST route (`cutoff` and `geminal_inv` both set) pairs with
/// `|v_t| < FAST_ECP_NONLOCAL_RADIAL_CUT` are dropped (see that constant); the
/// kept / dropped pair counts go to the route-predicate counters 16 / 17.
/// Every other route evaluates every pair, bit-identically to before the cut.
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_nonlocal_parts_all_pairs(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    det_only: bool,
    nn: usize,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    geminal_inv: Option<&ArrayView2<f64>>,
    cutoff: bool,
    det_state_in: Option<&crate::determinant::DetRatioState>,
    j3_state_in: Option<&crate::jastrow_factor::J3RatioState>,
) -> f64 {
    let radial_cut = if cutoff && geminal_inv.is_some() {
        FAST_ECP_NONLOCAL_RADIAL_CUT
    } else {
        0.0
    };
    let (v, kept, dropped) = compute_ecp_nonlocal_parts_all_pairs_radial_cut(
        geminal,
        jastrow,
        r_up,
        r_dn,
        positions,
        nucleus_index,
        ang_mom,
        exponents,
        coefficients,
        powers,
        weights,
        grid_points,
        det_only,
        nn,
        j3,
        cell_nmax,
        geminal_inv,
        cutoff,
        det_state_in,
        j3_state_in,
        radial_cut,
    );
    if radial_cut > 0.0 {
        crate::gfmc::record_ecp_radial_cut(kept, dropped);
    }
    v
}

/// [`compute_ecp_nonlocal_parts_all_pairs`] with an explicit radial cut
/// `radial_cut >= 0` (Ha), returning `(value, kept_pairs, dropped_pairs)`.
///
/// The cut applies only on the Sherman-Morrison route (`geminal_inv` set): an
/// (electron, term) pair with `|v_t(r)| < radial_cut` is skipped before its
/// quadrature mesh is built. `radial_cut = 0` drops nothing (`|v| < 0` is never
/// true, NaN is kept). The pair counts are 0 when no cut applies. Exposed so the
/// tests can compare the cut and the uncut FAST route on identical input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ecp_nonlocal_parts_all_pairs_radial_cut(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    det_only: bool,
    nn: usize,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    geminal_inv: Option<&ArrayView2<f64>>,
    // Opt-in per-AO distance cutoff on the mesh orbital eval (dropped AO values < 1e-12;
    // production e_L only -- the bit-exact test path passes `false`).
    cutoff: bool,
    // Caller-supplied streaming det-ratio state from the MCMC sweep: `Some`
    // skips the per-e_L
    // `init_det_ratio_state` re-eval (2 full orbital evals + 2 lambda GEMMs).
    // The sweep state is single-convention (seeded/streamed with
    // `compute_single`); the fast chunk passes it for MOLECULAR leaves only
    // because single and batch evaluation are identical there. The reference
    // path passes `None` and reinitializes the batch per local-energy call.
    det_state_in: Option<&crate::determinant::DetRatioState>,
    // Caller-supplied streaming J3-ratio state: `Some` skips the per-e_L
    // `init_j3_ratio_state` rebuild (2 dense n_orb^2 x Ne faer GEMMs). The MCMC sweep
    // seeds/streams J3 via the SAME `orb.compute` batch path this init uses, so on
    // MOLECULAR leaves the aos columns are bit-equal and the streamed w tables differ
    // only at the fp floor (fast chunk = statistically guarded). `None` = per-e_L rebuild.
    j3_state_in: Option<&crate::jastrow_factor::J3RatioState>,
    // Radial cut in Ha (see `FAST_ECP_NONLOCAL_RADIAL_CUT`); applied on the SM route only.
    radial_cut: f64,
) -> (f64, u64, u64) {
    assert!(
        radial_cut >= 0.0,
        "ECP radial cut must be non-negative: {radial_cut}"
    );
    // SM route only: the exact full-recompute branch (`geminal_inv = None`) never cuts.
    let radial_cut = if geminal_inv.is_some() {
        radial_cut
    } else {
        0.0
    };
    let mut n_kept = 0_u64;
    let mut n_dropped = 0_u64;
    // Fast-update path (Sherman-Morrison / matrix-determinant lemma): when the
    // base geminal inverse G^{-1} is supplied, each mesh-point det ratio
    // det(G_mesh)/det(G_base) = 1 + v*G^{-1}[:,i] (up move, row i changed) or
    // 1 + G^{-1}[j,:]*u (dn move, col j changed) -- O(N) instead of an O(N^3)
    // full geminal rebuild + LU. Mirrors jqmc's compute_local_energy_fast reuse
    // of the streaming inverse. `None` keeps the exact full-recompute path.
    // Home-cell-wrapped electron coordinates for the SM-path base AO evaluations
    // only, replicating jqmc's `compute_AOs` internal wrap: the base
    // determinant/J3 orbital matrices are the full-image value (consistent with the
    // reused inverse = inv(G(full-image))) even for a far-wandered electron, WHILE
    // the nucleus geometry (`rel`, mesh points), the Jastrow ratio and the Legendre
    // angle keep the RAW coordinate (jqmc's non-invariant reference: its nonlocal
    // Jastrow ratio differs raw vs wrapped by ~0.06 Ha). No-op for molecular / PW.
    let r_up_ao = crate::atomic_orbital::wrap_r(&geminal.orb_up.aos, r_up);
    let r_dn_ao = crate::atomic_orbital::wrap_r(&geminal.orb_dn.aos, r_dn);
    let r_up_ao_v = r_up_ao.view();
    let r_dn_ao_v = r_dn_ao.view();
    // Re-init only when the caller did not supply a state (bit-exact paths).
    let det_state_owned = if det_state_in.is_none() {
        geminal_inv
            .map(|_| crate::determinant::init_det_ratio_state(geminal, &r_up_ao_v, &r_dn_ao_v))
    } else {
        None
    };
    let det_state: Option<&crate::determinant::DetRatioState> =
        det_state_in.or(det_state_owned.as_ref());
    let inv_cell = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    // rel = R - r_e, Wigner-Seitz minimum-imaged under PBC.
    let rel_vec = |rx0: f64, ry0: f64, rz0: f64| -> [f64; 3] {
        match (cell_nmax, &inv_cell) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image([rx0, ry0, rz0], cell, iv, n),
            _ => [rx0, ry0, rz0],
        }
    };
    // Per-electron nearest-NN nucleus filter (production uses NN=1): an electron
    // only projects onto its `nn` nearest ECP nuclei. `nn == 0` => all-pairs.
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |rx: f64, ry: f64, rz: f64| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d0 = [
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ];
                let dd = match (cell_nmax, &inv_cell) {
                    (Some((cell, n)), Some(iv)) => crate::ewald::min_image(d0, cell, iv, n),
                    _ => d0,
                };
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let nv = weights.len();
    // Base det sign/ln|det|: consumed ONLY by the exact full-recompute branch
    // (`geminal_inv = None`). The SM path never reads them -- skip the full
    // geminal rebuild + LU there (it would be dead work).
    let (sd, ld) = if geminal_inv.is_some() {
        (0.0, 0.0)
    } else {
        det_sign_lnabs(&compute_geminal_all_elements(geminal, r_up, r_dn).view())
    };
    // Base log-Jastrow for the per-mesh ratio. Only needed for the exact
    // full-recompute path (geminal_inv = None); the SM path uses incremental
    // single-move ratios that are relative to the base directly.
    let jd = if det_only || geminal_inv.is_some() {
        0.0
    } else {
        jastrow_value_for_ratio(jastrow, j3, r_up, r_dn)
    };
    // Fast single-move Jastrow-ratio state (mirrors the MCMC sweep): when the SM
    // det path is active, replace the O(n_orb*N) per-mesh full jastrow_value_for_ratio
    // recompute (the e_L hot spot) with O(n_orb+N) rank-1 ratios.
    // Reuse the sweep's advanced J3 state when supplied (fast + molecular),
    // else rebuild fresh (bit-exact path / periodic). `j3_owned` holds the fresh build
    // only when we actually rebuild; `j3_ratio_state` is the borrowed handle used below.
    let j3_owned = if j3_state_in.is_none() && !det_only && geminal_inv.is_some() {
        j3.map(|(orb, jm)| {
            // J3 base orbitals wrap internally too (the J3 mesh orbitals below use
            // the wrapped mesh), so the J3 ratio stays full-image-consistent.
            let au = orb.compute(&r_up_ao_v);
            let ad = orb.compute(&r_dn_ao_v);
            crate::timed!(
                "ecp.j3_setup",
                crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    jm,
                    &au.view(),
                    &ad.view(),
                    jastrow.precision_policy.is_f32("jastrow_ratio"),
                )
            )
        })
    } else {
        None
    };
    let j3_ratio_state: Option<&crate::jastrow_factor::J3RatioState> =
        j3_state_in.or(j3_owned.as_ref());
    let mut v_total = 0.0_f64;
    // Per-electron nearest-ECP-nucleus set, computed ONCE, not per channel inside
    // the loop (that would be n_channels x redundant and the dominant nonlocal cost).
    let nearest_up: Vec<Vec<usize>> = (0..r_up.nrows())
        .map(|i| nearest_set(r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]))
        .collect();
    let nearest_dn: Vec<Vec<usize>> = (0..r_dn.nrows())
        .map(|i| nearest_set(r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]))
        .collect();

    // FAST periodic J1/J2 mesh ratios (fastmath sincos/exp; floor-compared), cutoff
    // mode only: the bit-exact path (`cutoff = false`) keeps the libm routine.
    let j12_fast_prep = if cutoff && !det_only {
        crate::jastrow_factor::J12FastPeriodic::new(jastrow)
    } else {
        None
    };
    // Fast-cutoff w-dot tables, ALL electrons at once (two faer GEMMs per spin,
    // C read from DRAM once per e_L) -- see `ecp_wdot_all_up`. Built only when the
    // SM path is active and the orbital qualifies; `None` keeps the batched
    // orb-mesh path (bit-exact mode).
    // Each eligible decision (inverse + ratio state present) is recorded in the
    // route-predicate counters (observability only; see `gfmc::record_ecp_wdot`).
    let wdot_up_all = match (geminal_inv, &det_state) {
        (Some(ginv), Some(st)) => {
            let engaged = ecp_wdot_ok(&geminal.orb_up, cutoff);
            crate::gfmc::record_ecp_wdot(engaged);
            engaged.then(|| crate::timed!("ecp.wbuild", ecp_wdot_all_up(geminal, st, ginv)))
        }
        _ => None,
    };
    let wdot_dn_all = match (geminal_inv, &det_state) {
        (Some(ginv), Some(st)) => {
            let engaged = ecp_wdot_ok(&geminal.orb_dn, cutoff);
            crate::gfmc::record_ecp_wdot(engaged);
            engaged.then(|| crate::timed!("ecp.wbuild", ecp_wdot_all_dn(geminal, st, ginv)))
        }
        _ => None,
    };
    let j3_wdot_up_all = match (j3, j3_ratio_state) {
        (Some((orb, jm)), Some(st)) => {
            let engaged = ecp_j3_wdot_ok(orb, cutoff);
            crate::gfmc::record_ecp_j3_wdot(engaged);
            engaged.then(|| ecp_j3_wdot_all(orb, jm, st, true))
        }
        _ => None,
    };
    let j3_wdot_dn_all = match (j3, j3_ratio_state) {
        (Some((orb, jm)), Some(st)) => {
            let engaged = ecp_j3_wdot_ok(orb, cutoff);
            crate::gfmc::record_ecp_j3_wdot(engaged);
            engaged.then(|| ecp_j3_wdot_all(orb, jm, st, false))
        }
        _ => None,
    };
    // Reused J3 ratio scratch: `aos_new`, `aos_new-aos_old` and the spin-length
    // matvec output are allocated once per local-energy call, not per quadrature
    // point (96k ratios in a 4-water/500-step MCMC). All shapes are fixed for
    // this local-energy call.
    let j3_norb = j3.map(|(_, jm)| jm.nrows()).unwrap_or(0);
    let mut j3_col = vec![0.0_f64; j3_norb];
    let mut j3_aos_p = vec![0.0_f64; j3_norb];
    let mut j3_v = vec![0.0_f64; r_up.nrows().max(r_dn.nrows())];

    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let (cx, cy, cz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        let l = ang_mom[t];
        let (e, c, p) = (exponents[t], coefficients[t], powers[t]);

        // up electrons
        for i in 0..r_up.nrows() {
            if !nearest_up[i].contains(&ia) {
                continue;
            }
            let rel = rel_vec(cx - r_up[[i, 0]], cy - r_up[[i, 1]], cz - r_up[[i, 2]]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            let v_l = c * ecp_pow(nrm, p - 2.0) * (-e * nrm * nrm).exp();
            if radial_cut > 0.0 {
                if v_l.abs() < radial_cut {
                    n_dropped += 1;
                    continue;
                }
                n_kept += 1;
            }
            let mut acc = 0.0_f64;
            // All nv mesh points for this (electron, channel) + cos angles, built
            // once so the orbital eval / det-ratio contraction batch over them.
            let mut mesh_pts = numpy::ndarray::Array2::<f64>::zeros((nv, 3));
            let mut cosv = vec![0.0_f64; nv];
            for v in 0..nv {
                let (dx, dy, dz) = (
                    grid_points[[v, 0]],
                    grid_points[[v, 1]],
                    grid_points[[v, 2]],
                );
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                // cos theta = (-rel/nrm)*(delta/|delta|); mesh = r_e + rel + |rel|*delta.
                cosv[v] = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                mesh_pts[[v, 0]] = r_up[[i, 0]] + rx + nrm * dx;
                mesh_pts[[v, 1]] = r_up[[i, 1]] + ry + nrm * dy;
                mesh_pts[[v, 2]] = r_up[[i, 2]] + rz + nrm * dz;
            }
            // The quadrature mesh
            // lies on a sphere of radius |rel| around the (min-image) nucleus, so a
            // point can fall OUTSIDE the home cell even when the electron is inside
            // it. The image-tiled batch AO (`orb.compute`, enumerated relative to
            // the home cell) then collapses for that point -> a wrong det/J3 ratio
            // and a ~1 Ha ECP-nl divergence from jqmc (which home-cell-wraps every
            // AO eval). Wrap the mesh points into the home cell for the AO evals
            // ONLY; the Legendre angle (`cosv`) and the min-image Jastrow / DTN
            // ratios keep the RAW geometry (translation-invariant). No-op for
            // molecular / pure PW / B-spline.
            let mesh_pts_w = crate::atomic_orbital::wrap_r(&geminal.orb_up.aos, &mesh_pts.view());
            match (geminal_inv, &det_state) {
                (Some(ginv), Some(st)) => {
                    let j3_wdot = j3_wdot_up_all
                        .as_ref()
                        .map(|(wt, base)| (wt.row(i), base[i]));
                    let j3_wdrs = j3_wdot.as_ref().map(|(w, _)| {
                        let mut drs = vec![0.0_f64; nv];
                        crate::timed!("ecp.j3_wdot", {
                            let ws = w.as_slice().expect("J3 W row contiguous");
                            let orb = j3.unwrap().0;
                            match orb.aos.kind {
                                crate::atomic_orbital::AoKind::Sphe => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_sphe(
                                        &orb.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                                crate::atomic_orbital::AoKind::Cart => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_cart(
                                        &orb.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                            }
                        });
                        drs
                    });
                    let j3_mesh = if j3_wdot.is_none() {
                        crate::timed!(
                            "ecp.j3_compute",
                            j3.map(|(orb, _)| orb.compute(&mesh_pts_w.view()))
                        )
                    } else {
                        None
                    };
                    // Fast cutoff path: electron-fixed w-dot contraction (dr from a
                    // single O(n_surviving_ao) AO dot per mesh point; the MO mesh is
                    // never formed). Same bilinear form as the batched path below,
                    // contracted in the cheap order -- fp-reordered, so gated to the
                    // cutoff (fast, floor-compared) mode only.
                    let wdot = wdot_up_all.as_ref().map(|(wt, cb)| (wt.row(i), cb[i]));
                    // One shell sweep contracts ALL nv mesh points (8-lane exp).
                    let wdrs = wdot.as_ref().map(|(w, _)| {
                        let mut drs = vec![0.0_f64; nv];
                        crate::timed!("ecp.wdot", {
                            let ws = w.as_slice().expect("W row contiguous");
                            match geminal.orb_up.aos.kind {
                                crate::atomic_orbital::AoKind::Sphe => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_sphe(
                                        &geminal.orb_up.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                                crate::atomic_orbital::AoKind::Cart => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_cart(
                                        &geminal.orb_up.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                            }
                        });
                        drs
                    });
                    // Batched SM path: one orb.compute + one v-GEMM over all nv pts
                    // (up move: row i changed; det ratio = 1 + v*G^-^1[:,i]).
                    let orb_vmesh = if wdot.is_none() {
                        let orb_mesh = crate::timed!(
                            "ecp.orb_compute",
                            if cutoff {
                                geminal.orb_up.compute_mo_cutoff(&mesh_pts_w.view())
                            } else {
                                geminal.orb_up.compute(&mesh_pts_w.view())
                            }
                        );
                        Some(crate::timed!(
                            "ecp.sm_vgemm",
                            crate::determinant::compute_v_up_move_batch(
                                geminal,
                                st,
                                i,
                                &orb_mesh.view()
                            )
                        ))
                    } else {
                        None
                    };
                    // DTN mesh ratios for all nv points at once (old-side + fixed
                    // r_kI table hoisted out of the per-mesh loop).
                    let dtn_mesh: Option<Vec<f64>> = if det_only {
                        None
                    } else {
                        jastrow.dtn.as_ref().map(|dtn| {
                            let mut o = vec![0.0_f64; nv];
                            crate::timed!(
                                "ecp.dtn_ratio",
                                crate::dtn::compute_dtn_ratio_mesh(
                                    dtn,
                                    r_up,
                                    r_dn,
                                    true,
                                    i,
                                    &mesh_pts.view(),
                                    &mut o
                                )
                            );
                            o
                        })
                    };
                    // J1/J2 ratios for all nv points, old-side terms hoisted
                    // (bitwise identical to the per-point routine).
                    let j12_mesh = j12_mesh_ratios(
                        jastrow,
                        j12_fast_prep.as_ref(),
                        r_up,
                        r_dn,
                        true,
                        i,
                        &mesh_pts.view(),
                        det_only,
                        nv,
                    );
                    for v in 0..nv {
                        let dr = match (&wdot, &orb_vmesh) {
                            (Some((_, c_base)), _) => 1.0 + wdrs.as_ref().unwrap()[v] - c_base,
                            (None, Some(vmesh)) => {
                                let mut d = 1.0_f64;
                                for a in 0..vmesh.nrows() {
                                    d += vmesh[[a, v]] * ginv[[a, i]];
                                }
                                d
                            }
                            (None, None) => unreachable!(),
                        };
                        let jfac = if det_only {
                            1.0
                        } else {
                            let mut jr = j12_mesh[v];
                            if let (Some((_, jm)), Some(st3), Some(jm_mesh)) =
                                (j3, j3_ratio_state, &j3_mesh)
                            {
                                for a in 0..jm_mesh.nrows() {
                                    j3_col[a] = jm_mesh[[a, v]];
                                }
                                let j3r = crate::timed!(
                                    "ecp.j3_ratio",
                                    crate::jastrow_factor::j3_ratio_with_scratch(
                                        jm,
                                        st3,
                                        true,
                                        i,
                                        &j3_col,
                                        &mut j3_aos_p,
                                        &mut j3_v,
                                    )
                                );
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow, jr, j3r,
                                );
                            } else if let (Some((_, base)), Some(drs)) = (&j3_wdot, &j3_wdrs) {
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow,
                                    jr,
                                    (drs[v] - base).exp(),
                                );
                            }
                            if let Some(dr) = &dtn_mesh {
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow, jr, dr[v],
                                );
                            }
                            jr
                        };
                        acc += ((2 * l + 1) as f64) * legendre(l, cosv[v]) * weights[v] * jfac * dr;
                    }
                }
                _ => {
                    // Full-recompute reference (geminal_inv = None). The moved
                    // electron uses the home-cell-wrapped mesh point (see above) so
                    // its AO does not collapse when the sphere leaves the cell.
                    let mut mesh = r_up.to_owned();
                    for v in 0..nv {
                        mesh[[i, 0]] = mesh_pts_w[[v, 0]];
                        mesh[[i, 1]] = mesh_pts_w[[v, 1]];
                        mesh[[i, 2]] = mesh_pts_w[[v, 2]];
                        let (sn, ln) = det_sign_lnabs(
                            &compute_geminal_all_elements(geminal, &mesh.view(), r_dn).view(),
                        );
                        let det_ratio = sn * sd * (ln - ld).exp();
                        let jfac = if det_only {
                            1.0
                        } else {
                            (jastrow_value_for_ratio(jastrow, j3, &mesh.view(), r_dn) - jd).exp()
                        };
                        acc += ((2 * l + 1) as f64)
                            * legendre(l, cosv[v])
                            * weights[v]
                            * jfac
                            * det_ratio;
                    }
                }
            }
            v_total += v_l * acc;
        }

        // dn electrons
        for i in 0..r_dn.nrows() {
            if !nearest_dn[i].contains(&ia) {
                continue;
            }
            let rel = rel_vec(cx - r_dn[[i, 0]], cy - r_dn[[i, 1]], cz - r_dn[[i, 2]]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            let v_l = c * ecp_pow(nrm, p - 2.0) * (-e * nrm * nrm).exp();
            if radial_cut > 0.0 {
                if v_l.abs() < radial_cut {
                    n_dropped += 1;
                    continue;
                }
                n_kept += 1;
            }
            let mut acc = 0.0_f64;
            let mut mesh_pts = numpy::ndarray::Array2::<f64>::zeros((nv, 3));
            let mut cosv = vec![0.0_f64; nv];
            for v in 0..nv {
                let (dx, dy, dz) = (
                    grid_points[[v, 0]],
                    grid_points[[v, 1]],
                    grid_points[[v, 2]],
                );
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                cosv[v] = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                mesh_pts[[v, 0]] = r_dn[[i, 0]] + rx + nrm * dx;
                mesh_pts[[v, 1]] = r_dn[[i, 1]] + ry + nrm * dy;
                mesh_pts[[v, 2]] = r_dn[[i, 2]] + rz + nrm * dz;
            }
            // Periodicity guarantor (see the up-electron block): wrap the mesh
            // points into the home cell for the AO evals only.
            let mesh_pts_w = crate::atomic_orbital::wrap_r(&geminal.orb_dn.aos, &mesh_pts.view());
            match (geminal_inv, &det_state) {
                (Some(ginv), Some(st)) => {
                    let j3_wdot = j3_wdot_dn_all
                        .as_ref()
                        .map(|(wt, base)| (wt.row(i), base[i]));
                    let j3_wdrs = j3_wdot.as_ref().map(|(w, _)| {
                        let mut drs = vec![0.0_f64; nv];
                        crate::timed!("ecp.j3_wdot", {
                            let ws = w.as_slice().expect("J3 W row contiguous");
                            let orb = j3.unwrap().0;
                            match orb.aos.kind {
                                crate::atomic_orbital::AoKind::Sphe => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_sphe(
                                        &orb.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                                crate::atomic_orbital::AoKind::Cart => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_cart(
                                        &orb.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                            }
                        });
                        drs
                    });
                    let j3_mesh = if j3_wdot.is_none() {
                        crate::timed!(
                            "ecp.j3_compute",
                            j3.map(|(orb, _)| orb.compute(&mesh_pts_w.view()))
                        )
                    } else {
                        None
                    };
                    // Fast cutoff path (see the up-electron block): electron-fixed
                    // w-dot contraction, fp-reordered so cutoff-mode only.
                    let wdot = wdot_dn_all.as_ref().map(|(wt, cb)| (wt.row(i), cb[i]));
                    // One shell sweep contracts ALL nv mesh points (8-lane exp).
                    let wdrs = wdot.as_ref().map(|(w, _)| {
                        let mut drs = vec![0.0_f64; nv];
                        crate::timed!("ecp.wdot", {
                            let ws = w.as_slice().expect("W row contiguous");
                            match geminal.orb_dn.aos.kind {
                                crate::atomic_orbital::AoKind::Sphe => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_sphe(
                                        &geminal.orb_dn.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                                crate::atomic_orbital::AoKind::Cart => {
                                    crate::atomic_orbital::dot_aos_multi_cutoff_cart(
                                        &geminal.orb_dn.aos,
                                        &mesh_pts_w.view(),
                                        ws,
                                        &mut drs,
                                    )
                                }
                            }
                        });
                        drs
                    });
                    // dn move: col i changed; det ratio = 1 + G^-^1[i,:]*u.
                    let orb_umesh = if wdot.is_none() {
                        let orb_mesh = crate::timed!(
                            "ecp.orb_compute",
                            if cutoff {
                                geminal.orb_dn.compute_mo_cutoff(&mesh_pts_w.view())
                            } else {
                                geminal.orb_dn.compute(&mesh_pts_w.view())
                            }
                        );
                        Some(crate::timed!(
                            "ecp.sm_vgemm",
                            crate::determinant::compute_u_dn_move_batch(
                                geminal,
                                st,
                                i,
                                &orb_mesh.view()
                            )
                        ))
                    } else {
                        None
                    };
                    let dtn_mesh: Option<Vec<f64>> = if det_only {
                        None
                    } else {
                        jastrow.dtn.as_ref().map(|dtn| {
                            let mut o = vec![0.0_f64; nv];
                            crate::timed!(
                                "ecp.dtn_ratio",
                                crate::dtn::compute_dtn_ratio_mesh(
                                    dtn,
                                    r_up,
                                    r_dn,
                                    false,
                                    i,
                                    &mesh_pts.view(),
                                    &mut o
                                )
                            );
                            o
                        })
                    };
                    let j12_mesh = j12_mesh_ratios(
                        jastrow,
                        j12_fast_prep.as_ref(),
                        r_up,
                        r_dn,
                        false,
                        i,
                        &mesh_pts.view(),
                        det_only,
                        nv,
                    );
                    for v in 0..nv {
                        let dr = match (&wdot, &orb_umesh) {
                            (Some((_, c_base)), _) => 1.0 + wdrs.as_ref().unwrap()[v] - c_base,
                            (None, Some(umesh)) => {
                                let mut d = 1.0_f64;
                                for a in 0..umesh.nrows() {
                                    d += ginv[[i, a]] * umesh[[a, v]];
                                }
                                d
                            }
                            (None, None) => unreachable!(),
                        };
                        let jfac = if det_only {
                            1.0
                        } else {
                            let mut jr = j12_mesh[v];
                            if let (Some((_, jm)), Some(st3), Some(jm_mesh)) =
                                (j3, j3_ratio_state, &j3_mesh)
                            {
                                for a in 0..jm_mesh.nrows() {
                                    j3_col[a] = jm_mesh[[a, v]];
                                }
                                let j3r = crate::timed!(
                                    "ecp.j3_ratio",
                                    crate::jastrow_factor::j3_ratio_with_scratch(
                                        jm,
                                        st3,
                                        false,
                                        i,
                                        &j3_col,
                                        &mut j3_aos_p,
                                        &mut j3_v,
                                    )
                                );
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow, jr, j3r,
                                );
                            } else if let (Some((_, base)), Some(drs)) = (&j3_wdot, &j3_wdrs) {
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow,
                                    jr,
                                    (drs[v] - base).exp(),
                                );
                            }
                            if let Some(dr) = &dtn_mesh {
                                jr = crate::jastrow_factor::multiply_ratio_components(
                                    jastrow, jr, dr[v],
                                );
                            }
                            jr
                        };
                        acc += ((2 * l + 1) as f64) * legendre(l, cosv[v]) * weights[v] * jfac * dr;
                    }
                }
                _ => {
                    let mut mesh = r_dn.to_owned();
                    for v in 0..nv {
                        mesh[[i, 0]] = mesh_pts_w[[v, 0]];
                        mesh[[i, 1]] = mesh_pts_w[[v, 1]];
                        mesh[[i, 2]] = mesh_pts_w[[v, 2]];
                        let (sn, ln) = det_sign_lnabs(
                            &compute_geminal_all_elements(geminal, r_up, &mesh.view()).view(),
                        );
                        let det_ratio = sn * sd * (ln - ld).exp();
                        let jfac = if det_only {
                            1.0
                        } else {
                            (jastrow_value_for_ratio(jastrow, j3, r_up, &mesh.view()) - jd).exp()
                        };
                        acc += ((2 * l + 1) as f64)
                            * legendre(l, cosv[v])
                            * weights[v]
                            * jfac
                            * det_ratio;
                    }
                }
            }
            v_total += v_l * acc;
        }
    }
    (v_total, n_kept, n_dropped)
}

/// Non-local ECP quadrature mesh for the Casula moves (tmove & dltmove): one
/// candidate configuration per `(spin, electron i, nearest nucleus ia,
/// quadrature point v)`. Mirrors jqmc `compute_ecp_non_local_parts_nearest_neighbors`
/// but returns per-candidate data (the move mesh) rather than only the summed energy.
///
/// Each returned tuple is `(is_up, elec_index, new_pos, det_amp, jastrow_ratio)`
/// where the moved electron `i` hops to `new_pos = r_e + rel + |rel|*Omega_v`
/// (Omega_v = grid point rotated by `rt`), `det_amp = det_ratio * Sum_channels-at-ia
/// [V_l*(2l+1)*P_l(costheta_v)*w_v]` (DETERMINANT-only), and `jastrow_ratio = exp(DeltaJ)`.
/// tmove uses `V = det_amp*jastrow_ratio`; dltmove keeps them split (det-only SP,
/// off-diagonal `min(det_amp,0)*jastrow_ratio`). Full-recompute; molecular path.
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_nonlocal_mesh(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nn: usize,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> Vec<(bool, usize, [f64; 3], f64, f64)> {
    let nv = weights.len();
    // rel = R_ia - r_e, Wigner-Seitz minimum-imaged under PBC (mesh point = r_e + rel + |rel|*Omega).
    let inv_cell = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    let mic = |v: [f64; 3]| -> [f64; 3] {
        match (cell_nmax, &inv_cell) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image(v, cell, iv, n),
            _ => v,
        }
    };
    // Rotate the quadrature grid by rt (jqmc: grid_points @ RT). Identity when rt=I.
    let gp: Vec<[f64; 3]> = (0..nv)
        .map(|v| {
            let mut o = [0.0_f64; 3];
            for c in 0..3 {
                for k in 0..3 {
                    o[c] += grid_points[[v, k]] * rt[k][c];
                }
            }
            o
        })
        .collect();
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |rx: f64, ry: f64, rz: f64| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let dd = mic([
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ]);
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let (sd, ld) = det_sign_lnabs(&compute_geminal_all_elements(geminal, r_up, r_dn).view());
    let jd = jastrow_value_for_ratio(jastrow, j3, r_up, r_dn);
    let mut out: Vec<(bool, usize, [f64; 3], f64, f64)> = Vec::new();

    // Closure shared by both spins: `is_up` selects the block; `r` is the moving
    // spin's coords, `mesh_full`/`other` build the full config for the det/J ratio.
    let mut do_spin = |is_up: bool, r: &ArrayView2<f64>| {
        for i in 0..r.nrows() {
            let near = nearest_set(r[[i, 0]], r[[i, 1]], r[[i, 2]]);
            for &ia in &near {
                let rel = mic([
                    positions[[ia, 0]] - r[[i, 0]],
                    positions[[ia, 1]] - r[[i, 1]],
                    positions[[ia, 2]] - r[[i, 2]],
                ]);
                let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
                let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
                for v in 0..nv {
                    let (dx, dy, dz) = (gp[v][0], gp[v][1], gp[v][2]);
                    let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                    let cosv = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                    let new_pos = [
                        r[[i, 0]] + rx + nrm * dx,
                        r[[i, 1]] + ry + nrm * dy,
                        r[[i, 2]] + rz + nrm * dz,
                    ];
                    // Sum over channels at this nucleus of V_l*(2l+1)*P_l*w.
                    let mut ang = 0.0_f64;
                    for t in 0..nucleus_index.len() {
                        if nucleus_index[t] as usize != ia {
                            continue;
                        }
                        let l = ang_mom[t];
                        let v_l = coefficients[t]
                            * ecp_pow(nrm, powers[t] - 2.0)
                            * (-exponents[t] * nrm * nrm).exp();
                        ang += v_l * ((2 * l + 1) as f64) * legendre(l, cosv) * weights[v];
                    }
                    // wf_ratio = exp(DeltaJ) * det_new/det_base for the single-electron hop.
                    let (sn, ln, jnew) = if is_up {
                        let mut mesh = r.to_owned();
                        mesh[[i, 0]] = new_pos[0];
                        mesh[[i, 1]] = new_pos[1];
                        mesh[[i, 2]] = new_pos[2];
                        let (sn, ln) = det_sign_lnabs(
                            &compute_geminal_all_elements(geminal, &mesh.view(), r_dn).view(),
                        );
                        (
                            sn,
                            ln,
                            jastrow_value_for_ratio(jastrow, j3, &mesh.view(), r_dn),
                        )
                    } else {
                        let mut mesh = r.to_owned();
                        mesh[[i, 0]] = new_pos[0];
                        mesh[[i, 1]] = new_pos[1];
                        mesh[[i, 2]] = new_pos[2];
                        let (sn, ln) = det_sign_lnabs(
                            &compute_geminal_all_elements(geminal, r_up, &mesh.view()).view(),
                        );
                        (
                            sn,
                            ln,
                            jastrow_value_for_ratio(jastrow, j3, r_up, &mesh.view()),
                        )
                    };
                    let det_ratio = sn * sd * (ln - ld).exp();
                    let jr = (jnew - jd).exp();
                    out.push((is_up, i, new_pos, ang * det_ratio, jr));
                }
            }
        }
    };
    do_spin(true, r_up);
    do_spin(false, r_dn);
    out
}

/// Electron-electron Coulomb energy: `sum_{i<j} 1/|r_i - r_j|`.
///
/// All electrons carry charge `-1`, so every pair product `Z_i Z_j = +1`
/// (matches the Python `np.full(..., -1.0)` + `Z_i * Z_j` construction).
pub fn compute_bare_coulomb_potential_el_el(
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    let n_up = r_up_carts.nrows();
    let n_dn = r_dn_carts.nrows();
    let n = n_up + n_dn;
    let ncols = if n_up > 0 {
        r_up_carts.ncols()
    } else {
        r_dn_carts.ncols()
    };

    // Component `k` of row `t` of the stacked [up; dn] electron block.
    let coord = |t: usize, k: usize| -> f64 {
        if t < n_up {
            r_up_carts[[t, k]]
        } else {
            r_dn_carts[[t - n_up, k]]
        }
    };

    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0_f64;
            for k in 0..ncols {
                let d = coord(i, k) - coord(j, k);
                d2 += d * d;
            }
            acc += 1.0 / d2.sqrt();
        }
    }
    acc
}

/// Ion-ion Coulomb energy: `sum_{i<j} Z_i Z_j / |R_i - R_j|`.
pub fn compute_bare_coulomb_potential_ion_ion(r_carts: &ArrayView2<f64>, charges: &[f64]) -> f64 {
    let n = r_carts.nrows();
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            acc += charges[i] * charges[j] / row_distance(r_carts, i, r_carts, j);
        }
    }
    acc
}

/// Element-wise electron-ion interaction for one spin block.
///
/// For each electron `e` (charge `-1`): `sum_I Z_I * (-1) / |R_I - r_e|`.
/// Returns a vector of length `r_carts_e.nrows()` (one entry per electron),
/// matching the Python `jnp.sum(..., axis=1)` over the ion axis.
fn el_ion_block(
    r_ion_carts: &ArrayView2<f64>,
    ion_charges: &[f64],
    r_elec_carts: &ArrayView2<f64>,
) -> Vec<f64> {
    let n_ion = r_ion_carts.nrows();
    let n_el = r_elec_carts.nrows();
    let mut out = vec![0.0_f64; n_el];
    for (e, out_e) in out[..n_el].iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (i, ion_charges_i) in ion_charges[..n_ion].iter().enumerate() {
            // electron charge is -1.0
            acc += -*ion_charges_i / row_distance(r_ion_carts, i, r_elec_carts, e);
        }
        *out_e = acc;
    }
    out
}

/// Element-wise electron-ion interactions for up- and down-spin electrons.
pub fn compute_bare_coulomb_potential_el_ion_element_wise(
    r_carts: &ArrayView2<f64>,
    charges: &[f64],
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    (
        el_ion_block(r_carts, charges, r_up_carts),
        el_ion_block(r_carts, charges, r_dn_carts),
    )
}

/// Composite molecular bare Coulomb energy (ion-ion + electron-ion + el-el).
///
/// Mirrors the `pbc_flag == False` branch of `compute_bare_coulomb_potential`.
pub fn compute_bare_coulomb_potential(
    r_carts: &ArrayView2<f64>,
    charges: &[f64],
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    let ii = compute_bare_coulomb_potential_ion_ion(r_carts, charges);
    let (up, dn) = compute_bare_coulomb_potential_el_ion_element_wise(
        r_carts, charges, r_up_carts, r_dn_carts,
    );
    let ei: f64 = up.iter().sum::<f64>() + dn.iter().sum::<f64>();
    let ee = compute_bare_coulomb_potential_el_el(r_up_carts, r_dn_carts);
    ii + ei + ee
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn ecp_pow_matches_powf_bitwise() {
        // Exact-result shortcuts of `ecp_pow` equal libm `powf` bit for bit.
        let mut x = 0.0_f64;
        for i in 0..20000 {
            x = x * 1.37 + 0.011 * (i % 97) as f64 + 1e-300;
            if x > 1e3 {
                x *= 1e-6;
            }
            for q in [-1.0, 0.0, 1.0, 2.0, 0.5] {
                assert_eq!(ecp_pow(x, q).to_bits(), x.powf(q).to_bits(), "x={x} q={q}");
            }
        }
        for q in [0.0, 1.0] {
            assert_eq!(ecp_pow(0.0, q).to_bits(), 0.0_f64.powf(q).to_bits());
        }
    }

    /// `P_l'` vs the independent identity `P_l' = sum_{k = l-1, l-3, ...} (2k+1) P_k`
    /// (finite on the closed interval) and the endpoint values `P_l'(+-1) = (+-1)^(l+1) l(l+1)/2`,
    /// including `x = +-1` where the recurrence form `l(P_{l-1}-xP_l)/(1-x^2)` is 0/0.
    #[test]
    fn legendre_deriv_closed_form_including_endpoints() {
        for l in 0..=6_i64 {
            for &x in &[-1.0, -0.999_999, -0.5, 0.0, 0.3, 0.999_999, 1.0] {
                let mut oracle = 0.0;
                let mut k = l - 1;
                while k >= 0 {
                    oracle += ((2 * k + 1) as f64) * legendre(k, x);
                    k -= 2;
                }
                let got = legendre_deriv(l, x);
                assert!(got.is_finite(), "l={l} x={x}: {got}");
                assert!(
                    (got - oracle).abs()
                        <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE * (1.0 + oracle.abs()),
                    "l={l} x={x}: {got} vs {oracle}"
                );
            }
            let end = (l * (l + 1)) as f64 / 2.0;
            assert_eq!(legendre_deriv(l, 1.0), end);
            let sgn = if l % 2 == 1 { 1.0 } else { -1.0 };
            assert_eq!(legendre_deriv(l, -1.0), sgn * end);
        }
        assert!(legendre_deriv(7, 0.5).is_nan());
        assert!(legendre_deriv(-1, 0.5).is_nan());
    }

    #[test]
    fn el_el_two_electrons_unit_distance() {
        let up = array![[0.0, 0.0, 0.0]];
        let dn = array![[1.0, 0.0, 0.0]];
        let v = compute_bare_coulomb_potential_el_el(&up.view(), &dn.view());
        assert!((v - 1.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }

    #[test]
    fn ion_ion_h2() {
        // two +1 charges at distance 2 -> 1/2
        let r = array![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]];
        let z = [1.0, 1.0];
        let v = compute_bare_coulomb_potential_ion_ion(&r.view(), &z);
        assert!((v - 0.5).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }

    #[test]
    fn fast_ecp_nonlocal_radial_cut_pinned_to_setting() {
        // The Rust radial cut is the central `jqmc._setting` value, not a copy that can drift.
        assert_eq!(
            FAST_ECP_NONLOCAL_RADIAL_CUT.to_bits(),
            crate::central_setting_f64("FAST_ECP_NONLOCAL_RADIAL_CUT").to_bits()
        );
    }

    /// Two ECP nuclei (s, p, d shells each) for the radial-cut tests.
    fn radial_cut_test_aos(
        kind: crate::atomic_orbital::AoKind,
        atoms: &[[f64; 3]],
    ) -> crate::atomic_orbital::AosGtoData {
        use crate::atomic_orbital::{AoKind, AosGtoData};
        let shells: [(i64, &[f64], &[f64]); 3] = [
            (0, &[5.0, 1.2, 0.3], &[0.3, 0.5, 0.4]),
            (1, &[1.5, 0.4], &[0.6, 0.5]),
            (2, &[0.8], &[1.0]),
        ];
        let mut d = AosGtoData::empty();
        d.kind = kind;
        let mut centers = Vec::new();
        let mut k = 0_usize;
        for c in atoms {
            for &(l, z, cf) in &shells {
                let comps: Vec<(i64, [i64; 3])> = match kind {
                    AoKind::Cart => {
                        let mut v = Vec::new();
                        for nx in (0..=l).rev() {
                            for ny in (0..=(l - nx)).rev() {
                                v.push((0, [nx, ny, l - nx - ny]));
                            }
                        }
                        v
                    }
                    AoKind::Sphe => (-l..=l).map(|m| (m, [0, 0, 0])).collect(),
                };
                for (m, p) in comps {
                    centers.extend_from_slice(c);
                    d.angular_momentums.push(l);
                    d.magnetic.push(m);
                    d.poly_x.push(p[0]);
                    d.poly_y.push(p[1]);
                    d.poly_z.push(p[2]);
                    d.kernel_to_home.push(k);
                    for (zz, cc) in z.iter().zip(cf.iter()) {
                        d.exponents.push(*zz);
                        d.coefficients.push(*cc);
                        d.orbital_indices.push(k);
                    }
                    k += 1;
                }
            }
        }
        d.num_ao = k;
        d.n_image = 1;
        d.num_ao_kernel = k;
        d.molecular_sphe = kind == AoKind::Sphe;
        d.precision_policy = crate::PrecisionPolicy::full();
        d.centers = Array2::from_shape_vec((k, 3), centers).unwrap();
        d.finalize()
    }

    /// FAST radial cut (`compute_ecp_nonlocal_parts_all_pairs_radial_cut`): the
    /// dropped (electron, term) pairs are exactly those with `|v_t(r)| < EPS`, and
    /// the FAST value with the cut equals the FAST value without it minus exactly
    /// the dropped pairs' contributions, each recomputed here independently from
    /// full determinants and Jastrow values (reduction-order tolerance). Checked
    /// at the production EPS and at a data-derived EPS that drops half the pairs
    /// (so the dropped sum is far above the tolerance); both drop and keep pairs.
    /// The exact route (`geminal_inv = None`) never cuts and stays bit-identical.
    #[test]
    fn fast_ecp_radial_cut_drops_exactly_sub_bound_pairs() {
        use crate::atomic_orbital::AoKind;
        use crate::data::JastrowTwoBodyData;
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let atoms = [[0.0, 0.0, 0.0], [5.0, 0.3, -0.2]];
        let positions = Array2::from_shape_vec((2, 3), atoms.concat()).unwrap();
        // Non-local terms per nucleus: (l, exponent, coefficient, power) -- a ccECP-like
        // tight s channel (O: 85.9 exp(-13.66 r^2)), a p channel and an r^-1 s channel.
        let terms: [(i64, f64, f64, f64); 3] = [
            (0, 13.66, 85.9, 2.0),
            (1, 6.0, -12.0, 2.0),
            (0, 3.0, 4.0, 1.0),
        ];
        let (mut nuc, mut am, mut ex, mut co, mut pw) = (vec![], vec![], vec![], vec![], vec![]);
        for ia in 0..atoms.len() {
            for &(l, e, c, p) in &terms {
                nuc.push(ia as i64);
                am.push(l);
                ex.push(e);
                co.push(c);
                pw.push(p);
            }
        }
        // Electron distances to the nearest nucleus span 0.3..2.7 bohr.
        let r_up = array![
            [0.3, 0.1, -0.2],
            [1.6, 0.5, 0.4],
            [4.6, 0.2, 0.1],
            [5.0, 2.4, -1.0]
        ];
        let r_dn = array![[0.0, -0.9, 0.6], [2.4, 0.1, 0.0], [5.5, 0.3, 0.2]];
        // 12-point icosahedral mesh, generically rotated.
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut verts = Vec::new();
        for s1 in [-1.0, 1.0] {
            for s2 in [-1.0, 1.0] {
                verts.push([0.0, s1, s2 * phi]);
                verts.push([s1, s2 * phi, 0.0]);
                verts.push([s2 * phi, 0.0, s1]);
            }
        }
        let (ca, sa, cb, sb) = (0.7_f64.cos(), 0.7_f64.sin(), 1.3_f64.cos(), 1.3_f64.sin());
        let rot = [
            [ca, -sa * cb, sa * sb],
            [sa, ca * cb, -ca * sb],
            [0.0, sb, cb],
        ];
        let nv = verts.len();
        let mut grid = Array2::<f64>::zeros((nv, 3));
        for (v, p) in verts.iter().enumerate() {
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            for c in 0..3 {
                grid[[v, c]] = (0..3).map(|k| p[k] / n * rot[k][c]).sum::<f64>();
            }
        }
        let weights = vec![1.0 / nv as f64; nv];
        let mut seed = 0x2545f4914f6cdd1d_u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for (kind, label) in [(AoKind::Sphe, "sphe"), (AoKind::Cart, "cart")] {
            let aos = radial_cut_test_aos(kind, &atoms);
            let na = aos.num_ao;
            let mo_up = Array2::<f64>::from_shape_fn((r_up.nrows(), na), |_| rnd());
            let mo_dn = Array2::<f64>::from_shape_fn((r_dn.nrows(), na), |_| rnd());
            let gem = GeminalData {
                orb_up: OrbData::new_mo(aos.clone(), mo_up),
                orb_dn: OrbData::new_mo(aos, mo_dn),
                lambda: None,
                orb_num_up: r_up.nrows(),
                orb_num_dn: r_dn.nrows(),
            };
            let jas = JastrowData {
                precision_policy: crate::PrecisionPolicy::full(),
                one_body: None,
                two_body: Some(JastrowTwoBodyData {
                    precision_policy: crate::PrecisionPolicy::full(),
                    jastrow_2b_param: 1.0,
                    jastrow_2b_type: "pade".to_string(),
                    pbc_flag: false,
                    cell: None,
                }),
                three_body: None,
                three_body_complex_only: false,
                dtn: None,
            };
            let g = compute_geminal_all_elements(&gem, &r_up.view(), &r_dn.view());
            let ginv = crate::determinant::inverse(&g.view());
            let (s0, l0) = det_sign_lnabs(&g.view());
            let j0 = jastrow_value_for_ratio(&jas, None, &r_up.view(), &r_dn.view());
            // Independent per-pair data: |v_t| and the full-recompute contribution v_t * acc.
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for is_up in [true, false] {
                let r = if is_up { &r_up } else { &r_dn };
                for i in 0..r.nrows() {
                    let d2 = |ia: usize| {
                        (0..3)
                            .map(|k| (atoms[ia][k] - r[[i, k]]).powi(2))
                            .sum::<f64>()
                    };
                    let ia = if d2(0) <= d2(1) { 0 } else { 1 };
                    for t in 0..nuc.len() {
                        if nuc[t] as usize != ia {
                            continue;
                        }
                        let rel: Vec<f64> = (0..3).map(|k| atoms[ia][k] - r[[i, k]]).collect();
                        let nrm = d2(ia).sqrt();
                        let v_t = co[t] * nrm.powf(pw[t] - 2.0) * (-ex[t] * nrm * nrm).exp();
                        let mut acc = 0.0;
                        for v in 0..nv {
                            let dv: Vec<f64> = (0..3).map(|k| grid[[v, k]]).collect();
                            let dn = dv.iter().map(|x| x * x).sum::<f64>().sqrt();
                            let cos = -(0..3).map(|k| rel[k] * dv[k]).sum::<f64>() / (nrm * dn);
                            let p_l = if am[t] == 0 { 1.0 } else { cos };
                            let mut moved = r.clone();
                            for k in 0..3 {
                                moved[[i, k]] = r[[i, k]] + rel[k] + nrm * dv[k];
                            }
                            let (gu, gd) = if is_up {
                                (moved.view(), r_dn.view())
                            } else {
                                (r_up.view(), moved.view())
                            };
                            let (s1, l1) = det_sign_lnabs(
                                &compute_geminal_all_elements(&gem, &gu, &gd).view(),
                            );
                            let jr = (jastrow_value_for_ratio(&jas, None, &gu, &gd) - j0).exp();
                            acc += ((2 * am[t] + 1) as f64)
                                * p_l
                                * weights[v]
                                * jr
                                * s1
                                * s0
                                * (l1 - l0).exp();
                        }
                        pairs.push((v_t.abs(), v_t * acc));
                    }
                }
            }
            let ginv_v = ginv.view();
            // `sm`: FAST Sherman-Morrison route (mesh cutoff on); otherwise the exact route.
            let run = |sm: bool, cut: f64| {
                compute_ecp_nonlocal_parts_all_pairs_radial_cut(
                    &gem,
                    &jas,
                    &r_up.view(),
                    &r_dn.view(),
                    &positions.view(),
                    &nuc,
                    &am,
                    &ex,
                    &co,
                    &pw,
                    &weights,
                    &grid.view(),
                    false,
                    1,
                    None,
                    None,
                    if sm { Some(&ginv_v) } else { None },
                    true,
                    None,
                    None,
                    cut,
                )
            };
            let (full, k0, d0) = run(true, 0.0);
            assert_eq!((k0, d0), (0, 0), "no cut, no pair count");
            // Data-derived EPS strictly between the two middle |v_t| values.
            let mut mags: Vec<f64> = pairs.iter().map(|p| p.0).collect();
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = (mags[mags.len() / 2 - 1], mags[mags.len() / 2]);
            assert!(lo < hi, "fixture needs distinct middle |v_t|");
            for eps in [FAST_ECP_NONLOCAL_RADIAL_CUT, (lo * hi).sqrt()] {
                let (cut_val, kept, dropped) = run(true, eps);
                let want_dropped = pairs.iter().filter(|p| p.0 < eps).count() as u64;
                assert_eq!(dropped, want_dropped, "{label} eps={eps}: dropped pairs");
                assert_eq!(
                    kept,
                    pairs.len() as u64 - want_dropped,
                    "{label} eps={eps}: kept pairs"
                );
                assert!(
                    dropped > 0 && kept > 0,
                    "{label} eps={eps}: vacuous ({kept} kept, {dropped} dropped)"
                );
                let dropped_sum: f64 = pairs.iter().filter(|p| p.0 < eps).map(|p| p.1).sum();
                let tol = atol + rtol * full.abs();
                assert!(
                    (cut_val - (full - dropped_sum)).abs() <= tol,
                    "{label} eps={eps}: cut {cut_val:e} vs full {full:e} - dropped {dropped_sum:e}"
                );
                if eps > FAST_ECP_NONLOCAL_RADIAL_CUT {
                    assert!(
                        dropped_sum.abs() > tol,
                        "{label}: dropped sum {dropped_sum:e} below tolerance"
                    );
                }
                // The exact route ignores the cut: bit-identical, no pair count.
                let (ex_cut, ke, de) = run(false, eps);
                let (ex_full, _, _) = run(false, 0.0);
                assert_eq!((ke, de), (0, 0));
                assert_eq!(
                    ex_cut.to_bits(),
                    ex_full.to_bits(),
                    "{label}: exact route changed by the cut"
                );
            }
        }
    }

    #[test]
    fn el_ion_sign_is_attractive() {
        let r = array![[0.0, 0.0, 0.0]];
        let z = [1.0];
        let up = array![[1.0, 0.0, 0.0]];
        let dn = array![[0.0, 2.0, 0.0]];
        let (u, d) = compute_bare_coulomb_potential_el_ion_element_wise(
            &r.view(),
            &z,
            &up.view(),
            &dn.view(),
        );
        assert!((u[0] - (-1.0)).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
        assert!((d[0] - (-0.5)).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }
}
