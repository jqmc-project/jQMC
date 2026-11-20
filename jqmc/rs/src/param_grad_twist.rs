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

//! Complex-twist (k != Gamma) Jastrow-parameter `de_L/dc = dT/dc` kernels.
//! At a twist the orbitals are complex but the Jastrow J(r) is real
//! (real electron distances, real params), so for a real Jastrow parameter `c`:
//!
//!   dT/dc = -1/2 Sum grad^2 f_c - Sum (gD + gJ) . grad f_c ,   f_c = dJ/dc,
//!
//! where the ONLY complex piece is the determinant drift `gD` (complex geminal).
//! The final variational estimate is real, but an individual walker's local energy
//! and `de_L/dc` are complex. The optimizer source contract therefore requires the
//! full complex reduction; its legacy diagnostic projection is performed in Python.
//!
//! The Jastrow field (grad_f, lap_f) and gJ are basis/geminal-INDEPENDENT, so the
//! same field builders (`param_grad_el::{j1_field,j2_field}`, the DTN reduction, the
//! J3 machinery) are reused verbatim. For an ECP twist the mesh `dV_ecp_nl/dc`
//! adds on via the same complex-mesh O_k identity without an early real projection.
//!
//! 1:1 with the verified real paths (`param_grad_el`, `param_grad_dtn`,
//! `param_grad_ecp`); the complex kinetic reduction mirrors
//! `complex_det::compute_kinetic_withj_complex`.

use numpy::ndarray::{Array2, ArrayView2, Axis};
use numpy::Complex64;

use crate::complex_det::{
    grads_and_laplacian_ln_det_complex, j3_grads_lap_complex, ComplexGeminal, ComplexOrb,
};
use crate::complex_ecp::compute_ecp_nonlocal_mesh_complex;
use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part;
use crate::param_grad::{grad_j1_param, grad_j2_param, j1_param_move_delta};
use crate::param_grad_dtn::{dtn_kinetic_reduce, DtnMovedOk, DtnOkData};
use crate::param_grad_el::{j1_field, j2_field};

/// Combined REAL cotangent drift `gJ + Re(gD)` for the twisted wavefunction, per
/// electron (`(n_up,3)`, `(n_dn,3)`). `gD` = complex determinant drift (real part
/// only -- e_L is real); `gJ` = real Jastrow drift (J1/J2/DTN via the real kernel +
/// the complex-Bloch J3 drift `j3_grads_lap_complex`). Feeds the shared reductions.
pub(crate) fn twist_drift(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (gd_up_c, gd_dn_c, _, _) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
    let (gj_up, gj_dn) = twist_gj(Some(jastrow), j3c, j3m, r_up, r_dn);
    let re = |g: &Array2<f64>, gc: &Array2<Complex64>| -> Array2<f64> {
        let mut out = g.clone();
        for i in 0..out.nrows() {
            for d in 0..3 {
                out[[i, d]] += gc[[i, d]].re;
            }
        }
        out
    };
    (re(&gj_up, &gd_up_c), re(&gj_dn, &gd_dn_c))
}

/// Full complex cotangent drift `gJ + gD`, retained for exact per-walker optimizer
/// sources. `gJ` is real and is lifted to complex only at this assembly seam.
fn twist_drift_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<Complex64>, Array2<Complex64>) {
    let (mut gd_up, mut gd_dn, _, _) = grads_and_laplacian_ln_det_complex(gem, r_up, r_dn);
    let (gj_up, gj_dn) = twist_gj(Some(jastrow), j3c, j3m, r_up, r_dn);
    for i in 0..gd_up.nrows() {
        for d in 0..3 {
            gd_up[[i, d]].re += gj_up[[i, d]];
        }
    }
    for i in 0..gd_dn.nrows() {
        for d in 0..3 {
            gd_dn[[i, d]].re += gj_dn[[i, d]];
        }
    }
    (gd_up, gd_dn)
}

fn reduce_field_complex(
    grad_f: &[[f64; 3]],
    lap_f: &[f64],
    drift_up: &Array2<Complex64>,
    drift_dn: &Array2<Complex64>,
    n_up: usize,
) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for e in 0..grad_f.len() {
        acc.re -= 0.5 * lap_f[e];
        for d in 0..3 {
            let drift = if e < n_up {
                drift_up[[e, d]]
            } else {
                drift_dn[[e - n_up, d]]
            };
            acc -= drift * grad_f[e][d];
        }
    }
    acc
}

/// REAL Jastrow drift `gJ` alone (no determinant part), per electron. `J(r)` at a
/// twist is real (real distances, real params) even though the orbitals are complex,
/// so J1/J2/DTN come from the shared real kernel and only the J3 orbital is the
/// `Re`-projected complex Bloch AO. `jastrow = None` (jnone) => zero drift.
pub fn twist_gj(
    jastrow: Option<&JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (mut gj_up, mut gj_dn) = match jastrow {
        Some(j) => {
            let (gu, gd, _, _) = compute_grads_and_laplacian_jastrow_part(j, r_up, r_dn);
            (gu, gd)
        }
        None => (
            Array2::<f64>::zeros((r_up.nrows(), 3)),
            Array2::<f64>::zeros((r_dn.nrows(), 3)),
        ),
    };
    if let (Some(orb), Some(jm)) = (j3c, j3m) {
        let (g3u, g3d, _, _) = j3_grads_lap_complex(orb, jm, r_up, r_dn);
        gj_up = gj_up + g3u;
        gj_dn = gj_dn + g3d;
    }
    (gj_up, gj_dn)
}

/// AGPn twist `de_L/d(lambda_diag, lambda_unpaired)` WITH Jastrow: the paired-block
/// DIAGONAL + unpaired columns of the complex dense-`lambda` kinetic gradient (the
/// chain rule for `lambda_matrix = hstack([diag(lambda_diag), lambda_unpaired])`).
///
/// The with-Jastrow twin of `complex_det::grad_t_lambda_diag_complex` (which is the
/// `gj = 0` jnone case): `J` does not depend on lambda, so the Jastrow enters only
/// through the drift cross term `-(gD + gJ) . d gD/dlambda` that `twist_gj` supplies.
/// The whole chain stays HOLOMORPHIC and `Re` is taken LAST, on the reduced leaves --
/// `lambda_diag` is a REAL variational DOF, so `de_L/d lambda_diag = Re(dT/dlambda)`.
pub fn grad_t_lambda_diag_twist(
    gem: &ComplexGeminal,
    jastrow: Option<&JastrowData>,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (numpy::ndarray::Array1<f64>, Array2<f64>) {
    let (gj_up, gj_dn) = twist_gj(jastrow, j3c, j3m, r_up, r_dn);
    let dense = crate::complex_det::grad_t_lambda_complex_gj(gem, r_up, r_dn, &gj_up, &gj_dn);
    let m_dn = gem.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = numpy::ndarray::Array1::from_iter((0..n).map(|k| dense[[k, k]].re));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).mapv(|z| z.re);
    (diag, unpaired)
}

/// Exact per-walker `de_L/dJ1_param = dT/dJ1` (complex scalar) for a twist.
pub fn grad_t_j1_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j1: &JastrowOneBodyData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Complex64 {
    let (grad_f, lap_f) = j1_field(j1, r_up, r_dn);
    let (du, dd) = twist_drift_complex(gem, jastrow, j3c, j3m, r_up, r_dn);
    reduce_field_complex(&grad_f, &lap_f, &du, &dd, r_up.nrows())
}

/// Exact per-walker `de_L/dJ2_param = dT/dJ2` (complex scalar) for a twist.
pub fn grad_t_j2_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j2: &JastrowTwoBodyData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Complex64 {
    let (grad_f, lap_f) = j2_field(j2, r_up, r_dn);
    let (du, dd) = twist_drift_complex(gem, jastrow, j3c, j3m, r_up, r_dn);
    reduce_field_complex(&grad_f, &lap_f, &du, &dd, r_up.nrows())
}

/// Exact per-walker `de_L/d{alpha,beta,gamma}_indep = dT/dc` for a twist.
pub fn grad_t_dtn_twist(
    d: &DtnOkData,
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> crate::param_grad_dtn::DtnGrads<Complex64> {
    let (du_re, dd_re) = twist_drift(gem, jastrow, j3c, j3m, r_up, r_dn);
    let (du_c, dd_c) = twist_drift_complex(gem, jastrow, j3c, j3m, r_up, r_dn);
    let du_im = du_c.mapv(|z| z.im);
    let dd_im = dd_c.mapv(|z| z.im);
    let zero_up = Array2::<f64>::zeros(du_re.dim());
    let zero_dn = Array2::<f64>::zeros(dd_re.dim());

    // The shared real reducer includes the Laplacian term. Its imaginary part has
    // no Laplacian contribution, so subtract the zero-drift evaluation from the
    // imaginary-drift evaluation. This keeps the large DTN field builder single-owned.
    let real = dtn_kinetic_reduce(d, r_up, r_dn, &du_re, &dd_re);
    let imag_with_lap = dtn_kinetic_reduce(d, r_up, r_dn, &du_im, &dd_im);
    let lap_only = dtn_kinetic_reduce(d, r_up, r_dn, &zero_up, &zero_dn);
    let combine = |r: Option<Array2<f64>>,
                   i: Option<Array2<f64>>,
                   l: Option<Array2<f64>>|
     -> Option<Array2<Complex64>> {
        match (r, i, l) {
            (Some(r), Some(i), Some(l)) => Some(Array2::from_shape_fn(r.dim(), |idx| {
                Complex64::new(r[idx], i[idx] - l[idx])
            })),
            (None, None, None) => None,
            _ => panic!("inconsistent DTN channel publication"),
        }
    };
    (
        combine(real.0, imag_with_lap.0, lap_only.0),
        combine(real.1, imag_with_lap.1, lap_only.1),
        combine(real.2, imag_with_lap.2, lap_only.2),
    )
}

/// `O_k = dJ3/dj_matrix` on a COMPLEX-Bloch J3 orbital leaf (twist), shape
/// `(orb, orb+1)`. The complex twin of `param_grad::grad_j3_jmatrix`.
///
/// J3 is a REAL correlation factor built on `A = Re(orbital)` (jqmc:
/// `jnp.real(compute_orb(orb_data, r))`), so once `A` is in hand the c-axis reduction is
/// the SAME basis-free core the real path uses. `ComplexOrb::value` supplies `A` for every
/// twist J3 leaf: single complex pGTO, MO-wrapped composite pGTO(+)PW / pGTO(+)B-spline,
/// and the pure complex plane-wave / B-spline leaves. Re is taken AFTER the MO
/// contraction (`Re(C)Re(AO) != Re(C*AO)`); the gto block is home-cell wrapped, matching
/// `grad_j3_jmatrix`'s `wrap_electrons` (exact -- `dJ3/dj_matrix` is lattice-shift
/// invariant -- and required for the ECP mesh points, which roam off the home cell).
pub fn grad_j3_jmatrix_complex(
    j3c: &ComplexOrb,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    let a_up = re(&j3c.value(r_up));
    let a_dn = re(&j3c.value(r_dn));
    crate::param_grad::j3_jmatrix_from_orb_values(&a_up, &a_dn)
}

/// `de_L/dj_matrix = Re(dT/dj_matrix)` (J3), shape `(orb, orb+1)`, for a twist. 1:1
/// with `param_grad_el::grad_t_j3` but every J3 AO quantity is `Re(complex Bloch sphe
/// AO)` and the cotangent drift is the twist `gJ + Re(gD)`.
pub fn grad_t_j3_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: &ComplexOrb,
    j3m: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let re = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    // ORBITAL-rank value/grad/laplacian with `Re` LAST (jqmc: `map(jnp.real,
    // compute_orb_vgl(orb_data, r))`). `ComplexOrb::value_grad_lap` covers every J3 leaf --
    // single complex pGTO, MO-wrapped composite pGTO(+)PW / pGTO(+)B-spline, and the PURE
    // complex plane-wave / B-spline leaves -- and applies the MO projection before the
    // projection to Re, since Re(C)Re(AO) != Re(C*AO).
    let vgl_re = |r: &ArrayView2<f64>| {
        let (v, g, l) = j3c.value_grad_lap(r);
        (re(&v), [re(&g[0]), re(&g[1]), re(&g[2])], re(&l))
    };
    let (a_up, ga_up, la_up) = vgl_re(r_up);
    let (a_dn, ga_dn, la_dn) = vgl_re(r_dn);
    let orb = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();

    // Combined drift gJ + Re(gD) (the J3 drift itself is part of gJ via j3c/j3m).
    let (du, dd) = twist_drift(gem, jastrow, Some(j3c), Some(j3m), r_up, r_dn);
    // h = -(gJ + gD)
    let h_up = -&du;
    let h_dn = -&dd;
    let hga = |h: &Array2<f64>, ga: &[Array2<f64>; 3], n: usize| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((orb, n));
        for a in 0..orb {
            for k in 0..n {
                let mut s = 0.0;
                for d in 0..3 {
                    s += h[[k, d]] * ga[d][[a, k]];
                }
                out[[a, k]] = s;
            }
        }
        out
    };
    let hga_up = hga(&h_up, &ga_up, n_up);
    let hga_dn = hga(&h_dn, &ga_dn, n_dn);

    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    // strict-upper ones (mirror param_grad_el::grad_t_j3's `k2`)
    let k2 = |n: usize| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                m[[i, j]] = 1.0;
            }
        }
        m
    };
    let k2u = k2(n_up);
    let k2d = k2(n_dn);
    let block = |cu: &Array2<f64>, cd: &Array2<f64>| -> Array2<f64> {
        let uu = cu.dot(&k2u.dot(&a_up.t())) + a_up.dot(&k2u).dot(&cu.t());
        let dd = cd.dot(&k2d.dot(&a_dn.t())) + a_dn.dot(&k2d).dot(&cd.t());
        let cus = cu.sum_axis(Axis(1));
        let cds = cd.sum_axis(Axis(1));
        let mut ud = Array2::<f64>::zeros((orb, orb));
        for i in 0..orb {
            for j in 0..orb {
                ud[[i, j]] = cus[i] * s_dn[j] + s_up[i] * cds[j];
            }
        }
        uu + dd + ud
    };
    let m_grad = block(&hga_up, &hga_dn) - 0.5 * block(&la_up, &la_dn);
    let mut j1col = hga_up.sum_axis(Axis(1)) + hga_dn.sum_axis(Axis(1));
    j1col = j1col - 0.5 * (la_up.sum_axis(Axis(1)) + la_dn.sum_axis(Axis(1)));

    let mut out = Array2::<f64>::zeros((orb, orb + 1));
    out.slice_mut(numpy::ndarray::s![.., 0..orb])
        .assign(&m_grad);
    for i in 0..orb {
        out[[i, orb]] = j1col[i];
    }
    out
}

// ============================================================================
// dV_ecp(non-local)/dc for the JASTROW / DTN blocks at a twist. The complex mesh
// (`compute_ecp_nonlocal_mesh_complex`) gives per-candidate `(is_up, i, new_pos,
// det_amp, jr)`; the c-independent weight is `v_grid = det_amp * jr` (COMPLEX). For a
// real Jastrow param the ratio's c-dependence is exp(J(r')-J(r)) so the O_k identity
//   dV/dc = Sum_g v_grid_g * O_k(r'_g) - (Sum_g v_grid_g) * O_k(r)
// holds with O_k = dJ/dc REAL; e_L is real => dE/dc = Re(dV/dc) =
//   Sum_g Re(v_grid_g) O_k(r'_g) - Re(sumV) O_k(r).
// ============================================================================

/// Nearest-neighbour ECP count matching the production non-local e_L (NN_default=1).
const NN_DEFAULT: usize = 1;

#[inline]
fn moved(base: &ArrayView2<f64>, idx: usize, pos: [f64; 3]) -> Array2<f64> {
    let mut m = base.to_owned();
    m[[idx, 0]] = pos[0];
    m[[idx, 1]] = pos[1];
    m[[idx, 2]] = pos[2];
    m
}

/// The per-candidate complex ECP mesh `(is_up, i, new_pos, det_amp, jr)` shared by every
/// twist parameter-gradient kernel.
///
/// Production rides the SAME rank-1 determinant / incremental-Jastrow candidate mesh
/// the native complex e_L uses (`gfmc::compute_ecp_nonlocal_mesh_complex_fast` over a
/// one-shot `MeshFastStateC`): every candidate is a single-electron move from one base,
/// so a full complex geminal LU plus a full Jastrow value per candidate (the reference
/// mesh, O(N^3 + N^2 + N orb n_img) per candidate) is pure redundancy. The reference
/// mesh stays the fallback for a `mesh_fast_state_complex` gate miss and the oracle
/// under the test-only `JQMC_ECP_PARAM_FORCE_REFERENCE` hook.
#[allow(clippy::too_many_arguments)]
fn twist_param_mesh(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> Vec<(bool, usize, [f64; 3], Complex64, f64)> {
    let rt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if !crate::gfmc::ecp_param_force_reference() {
        if let Some(mut st) = crate::timed!(
            "pd_twist.ecp_fast_state",
            crate::gfmc::mesh_fast_state_complex(gem, jastrow, j3c, j3m, r_up, r_dn)
        ) {
            st.enable_candidate_tables();
            crate::gfmc::record_ecp_param_mesh(None);
            return crate::timed!(
                "pd_twist.ecp_fast_mesh",
                crate::gfmc::compute_ecp_nonlocal_mesh_complex_fast(
                    gem,
                    jastrow,
                    j3c,
                    j3m,
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
                    &rt,
                    NN_DEFAULT,
                    cell.map(|c| (c, nmax)),
                    &st,
                )
            );
        }
    }
    let cands = compute_ecp_nonlocal_mesh_complex(
        gem,
        jastrow,
        j3c,
        j3m,
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
        &rt,
        NN_DEFAULT,
        cell,
        nmax,
        None,
    );
    crate::gfmc::record_ecp_param_mesh(Some(cands.len()));
    cands
}

/// Complex ECP mesh (per-candidate `(is_up, i, new_pos, Re(v_grid))`) plus `Re(sumV)`.
#[allow(clippy::too_many_arguments)]
fn one_move_mesh_re(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> (Vec<crate::gfmc::MeshCandidate>, f64) {
    let cands = twist_param_mesh(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    let mut out = Vec::with_capacity(cands.len());
    let mut sum_v = 0.0_f64;
    for (is_up, i, pos, det_amp, jr) in cands {
        let vg = (det_amp * jr).re;
        sum_v += vg;
        out.push((is_up, i, pos, vg));
    }
    (out, sum_v)
}

/// Exact complex ECP derivative for a scalar real Jastrow `O_k`.
fn ecp_reduce_scalar_complex<F>(
    mesh: &[crate::gfmc::MeshCandidateC],
    sum_v: Complex64,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ok: F,
) -> Complex64
where
    F: Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> f64,
{
    let ok_base = ok(r_up, r_dn);
    let mut acc = Complex64::new(0.0, 0.0);
    for &(is_up, i, pos, vg) in mesh {
        let o = if is_up {
            let m = moved(r_up, i, pos);
            ok(&m.view(), r_dn)
        } else {
            let m = moved(r_dn, i, pos);
            ok(r_up, &m.view())
        };
        acc += vg * o;
    }
    acc - sum_v * ok_base
}

/// AGPn `dV_ecp_nl/d(lambda_diag, lambda_unpaired)` at a twist -- the natural-orbital
/// twin of `param_grad_ecp::grad_ecp_lambda`.
///
/// Same weighted mesh O_k identity as the scalar reduce above, retaining the
/// complex `O_k = dln det/dlambda` and complex mesh weights until multiplication:
///   `d Re(V)/dc = Re[Sum_g W_g O_k(r'_g) - O_k(r) Sum_g W_g]`.
/// Real parameter storage does not make O_k real: projecting either factor
/// first loses the imaginary cross term, including on paired occupations.
/// This is the STREAMED (per-mesh-point O_k) form, which is what the XLA reference
/// takes for an AGPn geminal (`one_move_mesh.fast_supported_c` returns False whenever the
/// geminal carries a `lambda_diag` leaf, so the collapsed dense-lambda GEMM path is
/// never the reference here).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_lambda_diag_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> (numpy::ndarray::Array1<f64>, Array2<f64>) {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        crate::complex_det::grad_ln_det_lambda_complex_holo(gem, ru, rd)
    };
    let base = ok(r_up, r_dn);
    let m_dn = gem.orb_num_dn;
    let mut d = numpy::ndarray::Array1::<f64>::zeros(m_dn.min(base.nrows()));
    let mut u = Array2::<f64>::zeros((base.nrows(), base.ncols() - m_dn));
    for &(is_up, i, pos, vg) in &mesh {
        let shifted = if is_up {
            let m = moved(r_up, i, pos);
            ok(&m.view(), r_dn)
        } else {
            let m = moved(r_dn, i, pos);
            ok(r_up, &m.view())
        };
        // Contract directly from the dense complex cotangent, with no per-point
        // real dense copy and no paired/unpaired temporary arrays.
        for (k, value) in d.iter_mut().enumerate() {
            *value += (vg * shifted[[k, k]]).re;
        }
        for ((a, b), value) in u.indexed_iter_mut() {
            *value += (vg * shifted[[a, m_dn + b]]).re;
        }
    }
    for (k, value) in d.iter_mut().enumerate() {
        *value -= (sum_v * base[[k, k]]).re;
    }
    for ((a, b), value) in u.indexed_iter_mut() {
        *value -= (sum_v * base[[a, m_dn + b]]).re;
    }
    (d, u)
}

/// `dV_ecp_nl/dJ1_param` (scalar) at a twist.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j1_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j1: &JastrowOneBodyData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> Complex64 {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    if crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
        return ecp_reduce_scalar_complex(&mesh, sum_v, r_up, r_dn, |ru, rd| {
            grad_j1_param(j1, ru, rd)
        });
    }
    crate::gfmc::record_ecp_param_ok(None);
    mesh.iter()
        .fold(Complex64::new(0.0, 0.0), |acc, &(is_up, i, pos, vg)| {
            acc + vg * j1_param_move_delta(j1, r_up, r_dn, is_up, i, pos)
        })
}

/// `dV_ecp_nl/dJ2_param` (scalar) at a twist.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j2_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j2: &JastrowTwoBodyData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> Complex64 {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    if crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
        return ecp_reduce_scalar_complex(&mesh, sum_v, r_up, r_dn, |ru, rd| {
            grad_j2_param(j2, ru, rd)
        });
    }
    crate::gfmc::record_ecp_param_ok(None);
    crate::timed!(
        "pd_twist.ecp_j2_reduce",
        crate::param_grad_ecp::mesh_reduce_j2_incremental(j2, r_up, r_dn, mesh.iter().copied())
    )
}

/// Complex ECP mesh KEEPING the complex per-candidate weight `v_grid = det_amp*jr`
/// (and `sumV`). Needed wherever the O_k the identity multiplies is itself COMPLEX --
/// i.e. the geminal-BASIS (radial) O_k, which is the holomorphic `dln det/dradial`.
/// Re-projecting either factor first would drop the `Im(v_grid)*Im(O_k)` cross term
/// (`Re(A)Re(B) != Re(A*B)`); `one_move_mesh_re` above is correct only because the Jastrow
/// O_k it serves is real.
#[allow(clippy::too_many_arguments)]
fn one_move_mesh_cplx(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> (Vec<crate::gfmc::MeshCandidateC>, Complex64) {
    let cands = twist_param_mesh(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    let mut out = Vec::with_capacity(cands.len());
    let mut sum_v = Complex64::new(0.0, 0.0);
    for (is_up, i, pos, det_amp, jr) in cands {
        let vg = det_amp * jr;
        sum_v += vg;
        out.push((is_up, i, pos, vg));
    }
    (out, sum_v)
}

/// Dense AGP `dV_ecp(non-local)/d(lambda_matrix)` at a twist.
///
/// Both the mesh weight and the dense-lambda log-derivative are holomorphic complex
/// quantities, so their product must remain complex until the optimizer convention
/// projects it.  This is the dense counterpart of `grad_ecp_lambda_diag_twist`; the
/// latter projects the products onto real natural-orbital parameter outputs.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_lambda_twist(
    gem: &ComplexGeminal,
    lam: &ArrayView2<Complex64>,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> Array2<Complex64> {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        crate::complex_det::grad_ln_det_lambda_complex_holo_clam(gem, lam, ru, rd)
    };
    let base = ok(r_up, r_dn);
    let mut out = Array2::<Complex64>::zeros(base.dim());
    for &(is_up, i, pos, vg) in &mesh {
        let moved_ok = if is_up {
            let moved_up = moved(r_up, i, pos);
            ok(&moved_up.view(), r_dn)
        } else {
            let moved_dn = moved(r_dn, i, pos);
            ok(r_up, &moved_dn.view())
        };
        out.scaled_add(vg, &moved_ok);
    }
    out.scaled_add(-sum_v, &base);
    out
}

/// Exact per-walker `dV_ecp_nl/dradial.{exp,coeff}` (geminal basis) at a twist:
/// `dE/dc = Sum_g v_grid_g O_k^holo(r'_g) - sumV * O_k^holo(r)` with
/// `O_k^holo = dln det/dradial` the HOLOMORPHIC complex-Bloch radial O_k
/// (`param_grad_basis::grad_radial_complex`). 1:1 the real `param_grad_ecp::grad_ecp_radial`,
/// and 1:1 jqmc's `basis.grad_ln_det_wrt_basis(..., holomorphic=True)` ECP seam.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_radial_twist(
    gem: &ComplexGeminal,
    lam: Option<&ArrayView2<Complex64>>,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
    a2s_up: &[usize],
    n_shell_up: usize,
    a2s_dn: &[usize],
    n_shell_dn: usize,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> crate::param_grad_basis::RadialGrads<Complex64> {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        crate::param_grad_basis::grad_radial_complex(
            gem, lam, a2s_up, n_shell_up, a2s_dn, n_shell_dn, ru, rd,
        )
    };
    let base = ok(r_up, r_dn);
    let zero = Complex64::new(0.0, 0.0);
    let mut up = (vec![zero; base.0 .0.len()], vec![zero; base.0 .1.len()]);
    let mut dn = (vec![zero; base.1 .0.len()], vec![zero; base.1 .1.len()]);
    let axpy = |dst: &mut (Vec<Complex64>, Vec<Complex64>),
                src: &(Vec<Complex64>, Vec<Complex64>),
                s: Complex64| {
        for (d, v) in dst.0.iter_mut().zip(src.0.iter()) {
            *d += v * s;
        }
        for (d, v) in dst.1.iter_mut().zip(src.1.iter()) {
            *d += v * s;
        }
    };
    for &(is_up, i, pos, vg) in &mesh {
        let o = if is_up {
            let m = moved(r_up, i, pos);
            ok(&m.view(), r_dn)
        } else {
            let m = moved(r_dn, i, pos);
            ok(r_up, &m.view())
        };
        axpy(&mut up, &o.0, vg);
        axpy(&mut dn, &o.1, vg);
    }
    axpy(&mut up, &base.0, -sum_v);
    axpy(&mut dn, &base.1, -sum_v);
    (up, dn)
}

/// `dV_ecp_nl/dj_matrix` (J3), shape `(orb, orb+1)`, at a twist.
///
/// Same complex-mesh O_k identity as the scalar J1/J2 twins, with the ARRAY-valued
/// `O_k = dJ3/dj_matrix`: `dV/dc = Sum_g Re(v_grid_g) O_k(r'_g) - Re(sumV) O_k(r)`.
/// Re is taken on the candidate weight `det_amp * jr` as a PRODUCT (inside `one_move_mesh_re`),
/// never factor-by-factor, and `O_k` is real by construction.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j3_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: &ComplexOrb,
    j3m: &ArrayView2<f64>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> Array2<f64> {
    let (mesh, sum_v) = one_move_mesh_re(
        gem,
        jastrow,
        Some(j3c),
        Some(j3m),
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
        cell,
        nmax,
    );
    if !crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(None);
        // `A = Re(orbital)` exactly as `grad_j3_jmatrix_complex` (ComplexOrb wraps and
        // restores the Bloch phase per column, so a candidate column equals the moved
        // electron's column), then the exact linear one-column increment.
        let re = |r: &ArrayView2<f64>| j3c.value(r).mapv(|x| x.re);
        let a_up = re(r_up);
        let a_dn = re(r_dn);
        let pts = crate::param_grad::mesh_positions(mesh.iter().map(|m| m.2));
        let a_cand = re(&pts.view());
        let moves: Vec<(bool, usize, f64)> = mesh.iter().map(|m| (m.0, m.1, m.3)).collect();
        let (d_up, d_dn) = crate::param_grad::j3_mesh_column_changes(&a_up, &a_dn, &a_cand, &moves);
        return crate::timed!(
            "pd_twist.ecp_j3_increment",
            crate::param_grad::j3_jmatrix_mesh_increment(&a_up, &a_dn, &d_up, &d_dn)
        );
    }
    crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| grad_j3_jmatrix_complex(j3c, ru, rd);
    let base = ok(r_up, r_dn);
    let mut acc = Array2::<f64>::zeros(base.dim());
    for &(is_up, i, pos, vg) in &mesh {
        let o = if is_up {
            let m = moved(r_up, i, pos);
            ok(&m.view(), r_dn)
        } else {
            let m = moved(r_dn, i, pos);
            ok(r_up, &m.view())
        };
        acc += &(o * vg);
    }
    acc - &(base * sum_v)
}

/// `dV_ecp_nl/d(J3 radial.{exp,coeff})` at a twist.  The J3 radial O_k is
/// real (`J3` consumes `Re(orbital)`), so this follows the same real-mesh
/// identity as [`grad_ecp_j3_twist`] while the O_k kernel itself retains the
/// complex Bloch MO/image chain until its final projection.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j3_radial_twist(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: &ComplexOrb,
    j3m: &ArrayView2<f64>,
    a2s: &[usize],
    n_shell: usize,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> (Vec<f64>, Vec<f64>) {
    let (mesh, sum_v) = one_move_mesh_re(
        gem,
        jastrow,
        Some(j3c),
        Some(j3m),
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
        cell,
        nmax,
    );
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        crate::param_grad_basis::grad_j3_radial_complex(j3c, j3m, a2s, n_shell, ru, rd)
    };
    let base = ok(r_up, r_dn);
    let mut out = (vec![0.0; base.0.len()], vec![0.0; base.1.len()]);
    let axpy = |dst: &mut (Vec<f64>, Vec<f64>), src: &(Vec<f64>, Vec<f64>), scale: f64| {
        for (d, v) in dst.0.iter_mut().zip(src.0.iter()) {
            *d += scale * v;
        }
        for (d, v) in dst.1.iter_mut().zip(src.1.iter()) {
            *d += scale * v;
        }
    };
    for &(is_up, i, pos, vg) in &mesh {
        let moved_ok = if is_up {
            let moved_up = moved(r_up, i, pos);
            ok(&moved_up.view(), r_dn)
        } else {
            let moved_dn = moved(r_dn, i, pos);
            ok(r_up, &moved_dn.view())
        };
        axpy(&mut out, &moved_ok, vg);
    }
    axpy(&mut out, &base, -sum_v);
    out
}

/// `dV_ecp_nl/d{alpha,beta,gamma}_indep` (DTN u/chi/f) at a twist.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_dtn_twist(
    d: &DtnOkData,
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j3m: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
) -> crate::param_grad_dtn::DtnGrads<Complex64> {
    let (mesh, sum_v) = one_move_mesh_cplx(
        gem,
        jastrow,
        j3c,
        j3m,
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
        cell,
        nmax,
    );
    // O_k at each mesh point moves one electron: swap its terms instead of recomputing.
    let mut ok = DtnMovedOk::new(d, r_up, r_dn);
    let (a0, b0, g0) = ok.base_grads();
    let mut a = a0.as_ref().map(|x| Array2::<Complex64>::zeros(x.dim()));
    let mut b = b0.as_ref().map(|x| Array2::<Complex64>::zeros(x.dim()));
    let mut g = g0.as_ref().map(|x| Array2::<Complex64>::zeros(x.dim()));
    for &(is_up, i, pos, vg) in &mesh {
        let (am, bm, gm) = ok.moved_grads(is_up, i, pos);
        if let (Some(acc), Some(v)) = (a.as_mut(), am.as_ref()) {
            *acc = &*acc + &v.mapv(|x| vg * x);
        }
        if let (Some(acc), Some(v)) = (b.as_mut(), bm.as_ref()) {
            *acc = &*acc + &v.mapv(|x| vg * x);
        }
        if let (Some(acc), Some(v)) = (g.as_mut(), gm.as_ref()) {
            *acc = &*acc + &v.mapv(|x| vg * x);
        }
    }
    if let (Some(acc), Some(v)) = (a.as_mut(), a0.as_ref()) {
        *acc = &*acc - &v.mapv(|x| sum_v * x);
    }
    if let (Some(acc), Some(v)) = (b.as_mut(), b0.as_ref()) {
        *acc = &*acc - &v.mapv(|x| sum_v * x);
    }
    if let (Some(acc), Some(v)) = (g.as_mut(), g0.as_ref()) {
        *acc = &*acc - &v.mapv(|x| sum_v * x);
    }
    (a, b, g)
}
