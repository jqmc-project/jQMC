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

//! GFMC (LRDMC) force and stress observables for the novar=0 measurement split.
//! Hellmann--Feynman derivatives use the continuous measurement local energy;
//! lattice clipping and S-prime membership belong only to propagation. Pulay
//! terms remain pure wave-function derivatives. This mirrors the production
//! and debug routes in `jqmc_gfmc.py`.

use crate::complex_det::det_ratio_complex;
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ao_cart_deriv::{ao_grad, pgto_grad, pgto_grad_screened, AoScreen, DerivScreens};
use crate::atomic_orbital::AosGtoData;
use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{compute_geminal_all_elements, det_sign_lnabs};
use crate::determinant::{GeminalData, OrbData};
use crate::ewald::{inv3, min_image};
use crate::forces::{
    grad_j1_wrt_r, grad_j1_wrt_r_pbc, grad_j3_wrt_r, grad_t_wrt_r_w, grad_v_wrt_r,
};
use crate::forces_ecp::leg_and_deriv;
use crate::forces_pbc::{bspline_det_egrad_side, pgto_det_force_egrad};
use crate::param_grad_el::{
    det_grad_field, det_grad_field_gi, gem_forward, gem_forward_gi, geminal_reverse_w,
    jastrow_grads, jastrow_grads_recomputed_channels,
};
use crate::stress::{tfull_grad_wrt_r_w, vbare_grad_wrt_r};

/// A cart-pgto force leaf: `(cart_aos, c_cart (orbital->cart-AO), home_atom per cart-AO)`.
pub type PgtoLeaf = (AosGtoData, Array2<f64>, Vec<usize>);

/// Periodic SWCT geometry `(cell, inverse cell, image translations)`.
type SwctPeriodic = (Array2<f64>, [[f64; 3]; 3], Vec<[f64; 3]>);
/// One det mesh candidate `(det_ratio, wg (M_up,N_up), bar_dn (M_dn,N_dn), det_drift)`.
type DetCandidate = (f64, Array2<f64>, Array2<f64>, [f64; 3]);
/// Fused det candidates `(det_ratio, det drift row)` plus the group data.
type FusedDetCandidates = (Vec<(f64, [f64; 3])>, DetStressFusedGroup);
/// Fresh `(folded, kernel)` gradient tables of one moving-side leaf.
type FreshStrainTables = ([Array2<f64>; 3], [Array2<f64>; 3]);
/// One Jastrow mesh candidate `(jr, pulay (n_atom,3), gju (n_up,3), gjd (n_dn,3))`.
type JasCandidate = (f64, Array2<f64>, Array2<f64>, Array2<f64>);
/// Precontracted `(cm_base_up, cm_base_dn, base_du, base_dd)`.
type BaseBarsDrift = (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>);
/// `(J1 Pulay (n_atom,3), J1 drift, J2 pair grads per up / dn partner)`.
type J12Terms = (Array2<f64>, [f64; 3], Vec<[f64; 3]>, Vec<[f64; 3]>);

/// Periodic J3 context: the OrbData (for periodic value/drift via jastrow_grads/
/// jastrow_value_for_ratio) + the cart-pgto basis (cart_j3, T_mol, per-sphe-orb home atom,
/// j_matrix) for the periodic J3 Pulay + electron-axis kinetic kernels.
pub struct J3Pbc<'a> {
    pub orb: &'a OrbData,
    pub cart_j3: &'a AosGtoData,
    pub t_mol: ArrayView2<'a, f64>,
    pub jm: ArrayView2<'a, f64>,
    pub home: &'a [usize],
    pub pw: Option<crate::forces_pbc::J3Extra<'a>>,
    pub gto_first: bool,
}

/// Periodic J3 context for the STRESS kernels (adds `kernel` = per-cart-AO kernel
/// atom, needed by the field-strain leaves) alongside the [`J3Pbc`] force fields.
pub struct J3StressPbc<'a> {
    pub orb: &'a OrbData,
    pub cart_j3: &'a AosGtoData,
    pub t_mol: ArrayView2<'a, f64>,
    pub jm: ArrayView2<'a, f64>,
    pub home: &'a [usize],
    pub kernel: &'a [usize],
    pub pw: Option<crate::forces_pbc::J3Extra<'a>>,
    pub gto_first: bool,
}

/// dln|Psi|/dR (Pulay, `(n_atom,3)`) at one config: det + J1 + J3 (DTN unsupported here --
/// guard at the wrapper).
#[allow(clippy::too_many_arguments)]
pub fn lnpsi_pulay_r(
    gem: &GeminalData,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
) -> Array2<f64> {
    lnpsi_pulay_r_gi(
        gem, j1, j3_orb, j3_jm, dtn, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, None,
    )
}

/// `lnpsi_pulay_r` reusing a caller-supplied BASE geminal inverse for the determinant
/// Pulay (`grad_lndet_wrt_r_gi`); the Jastrow (J1/J3/DTN) Pulay parts are inverse-free.
/// `None` == fresh LU (the mesh-config re-evals). Used by the VMC ECP force so the
/// base `O_R` reuses the sweep Sherman-Morrison inverse.
#[allow(clippy::too_many_arguments)]
pub fn lnpsi_pulay_r_gi(
    gem: &GeminalData,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    let mut d_r = crate::timed!(
        "f.pp_det",
        crate::forces::grad_lndet_wrt_r_gi(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, g_inv)
    );
    if let Some(j1) = j1 {
        d_r += &grad_j1_wrt_r(j1, r_up, r_dn);
    }
    if let (Some(orb), Some(jm), Some(nuc)) = (j3_orb, j3_jm, nuc_j3) {
        d_r += &crate::timed!("f.pp_j3", grad_j3_wrt_r(orb, jm, r_up, r_dn, nuc, n_atom));
    }
    if let Some(d) = dtn {
        d_r += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, r_up, r_dn);
    }
    d_r
}

/// gradln|Psi| drift per electron at one config: `(N_up,3), (N_dn,3)` (det + J1/J2/DTN + J3).
pub fn lnpsi_drift(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    lnpsi_drift_gi(gem, jastrow, j3_orb, j3_jm, r_up, r_dn, None)
}

/// Determinant drift plus the Jastrow channels that must be RECOMPUTED per
/// configuration. A force mesh pairs this with
/// `jastrow_drift_move_delta` for the channels that expose a rank-one delta;
/// between them every channel is counted exactly once
/// without exposing channel-specific branching to callers.
#[allow(dead_code)]
fn lnpsi_drift_recomputed_channels(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
    let (gj_up, gj_dn) = jastrow_grads_recomputed_channels(jastrow, j3_orb, j3_jm, r_up, r_dn);
    (gd_up + &gj_up, gd_dn + &gj_dn)
}

/// `lnpsi_drift` reusing a caller-supplied BASE geminal inverse for the determinant
/// drift (`det_grad_field_gi`); the Jastrow drift is inverse-free. `None` == fresh LU.
pub fn lnpsi_drift_gi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let (gd_up, gd_dn) = crate::param_grad_el::det_grad_field_gi(gem, r_up, r_dn, g_inv);
    let (gj_up, gj_dn) = jastrow_grads(jastrow, j3_orb, j3_jm, r_up, r_dn);
    (gd_up + &gj_up, gd_dn + &gj_dn)
}

/// GFMC Pulay-force building block: `(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)`.
/// 1:1 `_grad_ln_psi_rR` (jqmc_gfmc.py, identical to the MCMC interface -- ln|Psi| is
/// LRDMC-independent).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_ln_psi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    gfmc_grad_ln_psi_gi(
        gem, jastrow, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, None,
    )
}

/// `gfmc_grad_ln_psi` reusing a caller-supplied BASE geminal inverse (= the sweep SM
/// inverse) for the determinant Pulay + drift (`lnpsi_pulay_r_gi` / `lnpsi_drift_gi`);
/// the Jastrow (J1/J2/J3/DTN) pieces are inverse-free. `None` == fresh LU (the current
/// `gfmc_grad_ln_psi` semantics). Used by the native MCMC Pulay force so the base
/// `O_R`/`O_r` reuse the sweep inverse (near-singular-stable, no fresh LU -> no NaN;
/// avoiding a second near-singular solve). The J3 path shares the
/// AO-basis `grad_j3_wrt_r`/`jastrow_grads` kernels, which already handle an MO-wrapped
/// production J3 (verified bit-exact vs the JAX Pulay oracle) -- no `as_raw_ao` needed.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_ln_psi_gi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    n_atom: usize,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let dtn = jastrow.and_then(|j| j.dtn.as_ref());
    // Lower an MO-wrapped J3 to its AO-equivalent so the Pulay `grad_j3_wrt_r`
    // (`dln|Psi|/dR`) is exact for a production `mos_data` J3 -- `grad_j3_wrt_r` mixes
    // MO-value coeffs with raw-AO gradients indexed by MO index, silently wrong when
    // `num_mo != num_ao`. Raw-AO / `None` J3 -> a no-op clone (bit-identical). The
    // drift (`jastrow_grads`) is MO-correct on its own; the AO-equivalent J3 keeps it so.
    let _j3_ao: Option<(OrbData, Array2<f64>)> =
        crate::timed!("f.pp_lower_j3", lower_j3_to_ao(j3_orb, j3_jm));
    let _j3_ao_view = _j3_ao.as_ref().map(|(_, m)| m.view());
    // `lower_j3_to_ao` returns `Some` iff a J3 is present, so the else arm is only the
    // no-J3 case -> `(None, None)` (mirrors `vmc_diag_grads`; avoids a lifetime clash
    // between the borrowed-view arm and the original-param arm).
    let (j3_orb, j3_jm): (Option<&OrbData>, Option<&ArrayView2<f64>>) =
        match (&_j3_ao, &_j3_ao_view) {
            (Some((o, _)), Some(v)) => (Some(o), Some(v)),
            _ => (None, None),
        };
    let d_r = lnpsi_pulay_r_gi(
        gem, j1, j3_orb, j3_jm, dtn, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, g_inv,
    );
    let (du, dd) = lnpsi_drift_gi(gem, jastrow, j3_orb, j3_jm, r_up, r_dn, g_inv);
    (d_r, du, dd)
}

/// ECP local force building block (molecular): `dV_local/dr_e (n_e,3)` and
/// `dV_local/dR (n_atom,3)`. `nuc_l/exps/coefs/powers` are the PRE-SPLIT local
/// channel arrays (`_*_local_part`). V_local = Sum c*r^(p-2)*exp(-e r^2),
/// r=|R_I-r_e|; d/dr_e = Sum vlp*u_hat (u_hat=(r_e-R_I)/r), d/dR_I = -that.
fn ecp_local_force(
    positions: &ArrayView2<f64>,
    nuc_l: &[i64],
    exps: &[f64],
    coefs: &[f64],
    powers: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let mut d_re = Array2::<f64>::zeros((n_e, 3));
    let mut d_r = Array2::<f64>::zeros((n_atom, 3));
    for t in 0..nuc_l.len() {
        let ia = nuc_l[t] as usize;
        let (e, c, p) = (exps[t], coefs[t], powers[t]);
        let (cx, cy, cz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        for el in 0..n_e {
            let re = elec(el);
            let rel = [re[0] - cx, re[1] - cy, re[2] - cz]; // r_e - R_I
            let r = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let vlp =
                c * ((p - 2.0) * r.powf(p - 3.0) - 2.0 * e * r.powf(p - 1.0)) * (-e * r * r).exp();
            let u = [rel[0] / r, rel[1] / r, rel[2] / r];
            for d in 0..3 {
                d_re[[el, d]] += vlp * u[d];
                d_r[[ia, d]] -= vlp * u[d];
            }
        }
    }
    (d_re, d_r)
}

/// `de_L^t/d(R, r_up, r_dn)` for the ECP molecular LRDMC diagonal local energy
/// (tmove/dltmove). Diagonal via `gfmc_diag_grads(ecp=true)` + ECP local force +
/// ECP non-local mesh force. 1:1 jax.grad of `_compute_V_elements_t` (ECP branch).
/// Non-local: mirrors `compute_ecp_nonlocal_mesh` (molecular, cell=None); per
/// candidate `V = ang*ratio`, tmove uses the full det*Jastrow ratio, dltmove
/// splits on the det-only amplitude (SP: det-only ratio; FN: full). The mesh
/// electron `new_pos = R_I + nrm*g_hat` chains through r_i (via nrm) and R_I.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ecp(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> GfmcElGrads {
    let n_up = r_up.nrows();

    // (1) diagonal e_L derivative (ECP semantics: ei-channel = bare, no el-ion swap)
    let (mut d_r, mut d_re) = gfmc_diag_grads(
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        nuc_up,
        nuc_dn,
        nuc_j3,
        true,
    );

    // (2) ECP local force
    let (dl_re, dl_r) = ecp_local_force(positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn);
    d_re += &dl_re;
    d_r += &dl_r;

    // (3) ECP non-local mesh force
    let (nl_re, nl_r, _grid_stress) = ecp_nonlocal_force_mol(
        gem,
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        r_up,
        r_dn,
        rt,
        nuc_up,
        nuc_dn,
        nuc_j3,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        None,
        FastScreen::default(),
        None,
    );
    d_re += &nl_re;
    d_r += &nl_r;

    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// SWCT weights `omega_{alphai} = kappa_alpha/Sum_beta kappa_beta`, `kappa_alpha = 1/|r_i - R_alpha|^4` -- `(n_atom, N_e)`.
/// 1:1 `swct.py::evaluate_swct_omega`.
pub fn swct_omega(positions: &ArrayView2<f64>, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    swct_omega_geometry(&SwctGeometry::new(positions.to_owned(), None), r_carts)
}

/// `Sum_i grad_{r_i} omega_{alphai}` per atom, `(n_atom, 3)`. 1:1 `swct.py::evaluate_swct_domega`.
/// gradkappa_alpha = -4*(r-R_alpha)/d_alpha^6 ; gradomega_alpha = (gradkappa_alpha*S - kappa_alpha*gradS)/S^2.
pub fn swct_domega(positions: &ArrayView2<f64>, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    swct_domega_geometry(&SwctGeometry::new(positions.to_owned(), None), r_carts)
}

/// Immutable SWCT geometry, prepared once per batch. Image translations are
/// lexicographic, exactly like the streaming scan in `swct.py`. Do not use the
/// general Ewald `min_image` helper here: its half-integer rounding convention
/// differs from JAX's ties-to-even wrap at Wigner-Seitz boundaries.
pub struct SwctGeometry {
    positions: Array2<f64>,
    periodic: Option<SwctPeriodic>,
}

impl SwctGeometry {
    pub fn new(positions: Array2<f64>, cell_nmax: Option<(Array2<f64>, i32)>) -> Self {
        let periodic = cell_nmax.map(|(cell, nmax)| {
            let inv = inv3(&cell.view());
            let mut translations = Vec::new();
            for a in -nmax..=nmax {
                for b in -nmax..=nmax {
                    for c in -nmax..=nmax {
                        translations.push(std::array::from_fn(|k| {
                            a as f64 * cell[[0, k]]
                                + b as f64 * cell[[1, k]]
                                + c as f64 * cell[[2, k]]
                        }));
                    }
                }
            }
            (cell, inv, translations)
        });
        Self {
            positions,
            periodic,
        }
    }

    pub fn n_atom(&self) -> usize {
        self.positions.nrows()
    }

    fn displacement(&self, electron: [f64; 3], atom: usize) -> [f64; 3] {
        let diff: [f64; 3] = std::array::from_fn(|k| electron[k] - self.positions[[atom, k]]);
        let Some((cell, inv, translations)) = &self.periodic else {
            return diff;
        };
        let frac: [f64; 3] = std::array::from_fn(|k| {
            let f = diff[0] * inv[0][k] + diff[1] * inv[1][k] + diff[2] * inv[2][k];
            f - f.round_ties_even()
        });
        let wrapped: [f64; 3] = std::array::from_fn(|k| {
            frac[0] * cell[[0, k]] + frac[1] * cell[[1, k]] + frac[2] * cell[[2, k]]
        });
        let mut best = wrapped;
        let mut best_d2 = f64::INFINITY;
        for translation in translations {
            let candidate = std::array::from_fn(|k| wrapped[k] + translation[k]);
            let d2 = candidate[0] * candidate[0]
                + candidate[1] * candidate[1]
                + candidate[2] * candidate[2];
            if d2 < best_d2 {
                best = candidate;
                best_d2 = d2;
            }
        }
        best
    }
}

struct SwctScratch {
    kappa: Vec<f64>,
    grad_kappa: Vec<[f64; 3]>,
}

impl SwctScratch {
    fn new(n_atom: usize) -> Self {
        Self {
            kappa: vec![0.0; n_atom],
            grad_kappa: vec![[0.0; 3]; n_atom],
        }
    }

    /// Geometry is reconstructed in f64 before its use-site arithmetic. The
    /// `swct` zone is f64 in BOTH supported precision modes.
    fn fill(&mut self, geometry: &SwctGeometry, electron: [f64; 3]) -> (f64, [f64; 3]) {
        let mut sum = 0.0;
        let mut grad_sum = [0.0; 3];
        for a in 0..geometry.n_atom() {
            let diff = geometry.displacement(electron, a);
            let d2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            let kappa = 1.0 / (d2 * d2);
            self.kappa[a] = kappa;
            sum += kappa;
            for c in 0..3 {
                self.grad_kappa[a][c] = -4.0 * diff[c] * (kappa / d2);
                grad_sum[c] += self.grad_kappa[a][c];
            }
        }
        (sum, grad_sum)
    }
}

pub fn swct_omega_geometry(geometry: &SwctGeometry, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    let n_atom = geometry.n_atom();
    let n_e = r_carts.nrows();
    let mut out = Array2::<f64>::zeros((n_atom, n_e));
    let mut scratch = SwctScratch::new(n_atom);
    for i in 0..n_e {
        let (sum, _) = scratch.fill(geometry, std::array::from_fn(|c| r_carts[[i, c]]));
        for a in 0..n_atom {
            out[[a, i]] = scratch.kappa[a] / sum;
        }
    }
    out
}

pub fn swct_domega_geometry(geometry: &SwctGeometry, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    let n_atom = geometry.n_atom();
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    let mut scratch = SwctScratch::new(n_atom);
    for i in 0..r_carts.nrows() {
        let (sum, grad_sum) = scratch.fill(geometry, std::array::from_fn(|c| r_carts[[i, c]]));
        for a in 0..n_atom {
            for c in 0..3 {
                out[[a, c]] +=
                    (scratch.grad_kappa[a][c] * sum - scratch.kappa[a] * grad_sum[c]) / (sum * sum);
            }
        }
    }
    out
}

/// Stream each electron directly into contracted fields and domega, retaining
/// only O(n_atom) scratch per walker, never (batch,n_atom,n_e) weights.
/// Complex fields are contracted without conjugation or real projection.
pub fn swct_corrections<T>(
    geometry: &SwctGeometry,
    r_carts: &ArrayView2<f64>,
    fields: &ArrayView2<T>,
) -> (Array2<T>, Array2<f64>)
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
{
    let n_atom = geometry.n_atom();
    let n_field = fields.ncols();
    let mut dot = Array2::from_elem((n_atom, n_field), T::default());
    let mut domega = Array2::<f64>::zeros((n_atom, 3));
    let mut scratch = SwctScratch::new(n_atom);
    for i in 0..r_carts.nrows() {
        let (sum, grad_sum) = scratch.fill(geometry, std::array::from_fn(|c| r_carts[[i, c]]));
        for a in 0..n_atom {
            let omega = scratch.kappa[a] / sum;
            for c in 0..n_field {
                dot[[a, c]] += fields[[i, c]] * omega;
            }
            for c in 0..3 {
                domega[[a, c]] +=
                    (scratch.grad_kappa[a][c] * sum - scratch.kappa[a] * grad_sum[c]) / (sum * sum);
            }
        }
    }
    (dot, domega)
}

#[cfg(test)]
mod swct_tests {
    use super::*;
    use numpy::ndarray::array;
    use numpy::Complex64;

    #[test]
    fn swct_streaming_retains_complex_fields_and_empty_spin() {
        let geometry = SwctGeometry::new(array![[0.0, 0.0, 0.0]], None);
        let r = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let fields = array![
            [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 1.0)],
            [Complex64::new(4.0, -1.0), Complex64::new(2.0, 3.0)],
        ];
        let (dot, domega) = swct_corrections(&geometry, &r.view(), &fields.view());
        assert_eq!(
            dot,
            array![[Complex64::new(5.0, 1.0), Complex64::new(-1.0, 4.0)]]
        );
        assert_eq!(domega, Array2::<f64>::zeros((1, 3)));
        let (dot, domega) = swct_corrections(
            &geometry,
            &Array2::<f64>::zeros((0, 3)).view(),
            &Array2::<Complex64>::zeros((0, 6)).view(),
        );
        assert_eq!(dot, Array2::<Complex64>::zeros((1, 6)));
        assert_eq!(domega, Array2::<f64>::zeros((1, 3)));
    }

    #[test]
    fn swct_periodic_images_preserve_ties_and_lattice_shifts() {
        let positions = array![[0.0, 0.0, 0.0]];
        let cell = array![[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let geometry = SwctGeometry::new(positions.clone(), Some((cell, 1)));
        assert_eq!(geometry.displacement([1.0, 1.0, 0.0], 0), [0.0, -1.0, 0.0]);
        assert_eq!(geometry.displacement([2.0, -5.0, 2.0], 0), [0.0, -1.0, 0.0]);
        let cubic = array![[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let geometry = SwctGeometry::new(positions.clone(), Some((cubic.clone(), 1)));
        for x in [-1.0, 1.0, 3.0] {
            assert_eq!(geometry.displacement([x, 0.0, 0.0], 0), [-1.0, 0.0, 0.0]);
        }
        // With no neighbor images the wrap itself exposes ties-to-even.
        let geometry = SwctGeometry::new(positions, Some((cubic, 0)));
        assert_eq!(geometry.displacement([1.0, 0.0, 0.0], 0), [1.0, 0.0, 0.0]);
        assert_eq!(geometry.displacement([3.0, 0.0, 0.0], 0), [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn swct_streaming_matches_dense_molecular_contraction() {
        let positions = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let r = array![[0.5, 0.25, 1.0], [1.5, 0.5, 0.25], [0.25, 0.75, 0.5]];
        let fields = Array2::from_shape_fn((3, 6), |(i, c)| (i * 6 + c) as f64 - 5.0);
        let geometry = SwctGeometry::new(positions.clone(), None);
        let (dot, domega) = swct_corrections(&geometry, &r.view(), &fields.view());
        let omega = swct_omega(&positions.view(), &r.view());
        let mut dense = Array2::<f64>::zeros((2, 6));
        for a in 0..2 {
            for c in 0..6 {
                for i in 0..3 {
                    dense[[a, c]] += omega[[a, i]] * fields[[i, c]];
                }
            }
        }
        assert_eq!(dot, dense);
        assert_eq!(domega, swct_domega(&positions.view(), &r.view()));
    }
}

/// Result of the LRDMC HF-force derivative: `de_L^t/dR (n_atom,3)`,
/// `de_L^t/dr_up (N_up,3)`, `de_L^t/dr_dn (N_dn,3)`.
pub struct GfmcElGrads {
    pub d_r: Array2<f64>,
    pub d_rup: Array2<f64>,
    pub d_rdn: Array2<f64>,
}

/// `de_L^t/d(R, r_up, r_dn)` for the ALL-ELECTRON molecular LRDMC diagonal local
/// energy (1:1 jax.grad of `_compute_V_elements_t`, AE branch, `jnp.real` wrap
/// trivial on the real path). See the module doc for the S' reduction.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ae(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let (d_r, d_re) = gfmc_diag_grads(
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        nuc_up,
        nuc_dn,
        nuc_j3,
        false,
    );
    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// Diagonal LRDMC local-energy position-gradient core, shared by the AE and ECP
/// force. Returns `(de_L^diag/dR (n_atom,3), de_L^diag/dr (n_e,3))`. `ecp=true`
/// selects the ZV `max` channel = BARE el-ion (parcutg=1), so the el-ion swap
/// term (AE-only) vanishes and S' is selected on `ei_bare > zv`. All the other
/// S' pieces (masked continuum kinetic, strained mesh) are basis/potential-
/// agnostic and apply identically. The ECP local + non-local force is added by
/// the caller on top of this diagonal.
#[allow(clippy::too_many_arguments)]
fn gfmc_diag_grads(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    ecp: bool,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = positions.nrows();
    let j3 = match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => Some((o, m)),
        _ => None,
    };
    let j1 = jastrow_full.one_body.as_ref();
    let j2 = jastrow_full.two_body.as_ref();
    let has_j = j1.is_some() || j2.is_some() || jastrow_full.dtn.is_some() || j3.is_some();
    let jas_opt: Option<&JastrowData> = if has_j { Some(jastrow_full) } else { None };

    let _ = (alat, rt, ecp);

    let dtn = jastrow_full.dtn.as_ref();
    // Full drift cotangent cot = gradlnPsi = jastrow_grads(incl DTN) + det_grad_field, used by
    // the DTN kinetic-force side (the det/J1/J2/J3 sides build their own cot internally,
    // which also includes the DTN drift, so det*DTN |grad|^2 cross terms are already captured).
    let cot = if dtn.is_some() {
        let (gj_up, gj_dn) = jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn);
        let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
        let mut c = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for d in 0..3 {
                c[[e, d]] = if e < n_up {
                    gj_up[[e, d]] + gd_up[[e, d]]
                } else {
                    let l = e - n_up;
                    gj_dn[[l, d]] + gd_dn[[l, d]]
                };
            }
        }
        Some(c)
    } else {
        None
    };

    // ---- smooth base: d(V_ee + V_ii + Sum ei + T_cont)/dtheta ----
    let mut d_r = grad_t_wrt_r_w(
        gem, jas_opt, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom, None,
    );
    d_r += &grad_v_wrt_r(positions, z_eff, r_up, r_dn);
    let (tu, td) = tfull_grad_wrt_r_w(gem, jas_opt, j1, j2, j3_orb, j3_jm, r_up, r_dn, None);
    let (vu, vd) = vbare_grad_wrt_r(positions, z_eff, r_up, r_dn);
    let mut d_re = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            d_re[[e, c]] = if e < n_up {
                tu[[e, c]] + vu[[e, c]]
            } else {
                let l = e - n_up;
                td[[l, c]] + vd[[l, c]]
            };
        }
    }
    // DTN kinetic-force channel (smooth base): dT_DTN/dR and dT_DTN/dr with cot=gradlnPsi.
    if let (Some(d), Some(cot)) = (dtn, &cot) {
        d_r += &crate::forces_dtn::grad_t_dtn_wrt_r_atom(d, cot, r_up, r_dn);
        d_re += &crate::forces_dtn::grad_t_dtn_wrt_r(d, cot, r_up, r_dn);
    }

    (d_r, d_re)
}

// ============================================================================
// VMC (MCMC) HF force -- molecular AE + ECP. The VMC continuous local energy
// gradient de_L/d(R, r) the MCMC driver's `_grad_e_L_rR` emits. It is EXACTLY the
// `gfmc_diag_grads` SMOOTH BASE (d(V_ee+V_ii+Sum ei_bare + T_cont)/dtheta + DTN
// kinetic base) WITHOUT the LRDMC S' discretization terms (masked kinetic, el-ion
// swap, lattice-mesh) -- those are the alat-regularization corrections, absent from
// the continuous VMC e_L. So gfmc(LRDMC) != VMC once any electron kinetic sign-
// flips, producing large relative divergence on J12/J123/JDTN, which is why this
// is a distinct kernel, not a call into gfmc_grad_e_l_ae. Reuses the sweep SM
// geminal inverse `g_inv`, so the kinetic dT/dR + dT/dr +
// DTN-cotangent stay near-singular-stable (fresh LU NaNs; the AS-regularized SM
// inverse is stable). ECP local/non-local force (shared VMC/LRDMC -- the ECP
// potential is NOT alat-discretized) is added by the py caller on top of the diag.
// ============================================================================

/// Lower an MO-wrapped J3 orbital to its AO-equivalent: raw-AO orbital + AO-space
/// j_matrix `M_ao = C^T M_mo C`, `j1_ao = C^T j1_mo` (last column). The J3
/// value/grad/lap are invariant under this change of basis, and `R` enters only via
/// the (R-independent-`C`) AO centers, so the AO-basis J3 force kernels (`j3_side`
/// raw AOs, `tfull` J3, kinetic drift, AND `grad_j3_wrt_r` Pulay) stay exact AND
/// consistent. Production J3 (`orb_data=mos_data`) is MO-wrapped: `grad_j3_wrt_r`
/// mixes MO-value coeffs (`compute`) with raw-AO gradients (`ao_grad`) indexed by MO
/// index, so `num_mo != num_ao` silently corrupts the Pulay `dln|Psi|/dR` (the drift
/// path `jastrow_grads` is MO-correct on its own). `as_raw_ao` is a no-op clone for a
/// raw-AO J3, so `None`/raw-AO callers are bit-identical. Returns an OWNED
/// `(orbital, j_matrix)` so downstream borrows share one local lifetime.
pub fn lower_j3_to_ao(
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
) -> Option<(OrbData, Array2<f64>)> {
    match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => {
            let jm_ao = match &o.mo_coefficients {
                Some(c) => {
                    let num_mo = c.nrows();
                    let num_ao = c.ncols();
                    let m_mo = m.slice(numpy::ndarray::s![.., 0..num_mo]);
                    let j1_mo = m.column(num_mo);
                    let m_ao = c.t().dot(&m_mo).dot(c);
                    let j1_ao = c.t().dot(&j1_mo);
                    let mut jm = Array2::<f64>::zeros((num_ao, num_ao + 1));
                    jm.slice_mut(numpy::ndarray::s![.., 0..num_ao])
                        .assign(&m_ao);
                    jm.column_mut(num_ao).assign(&j1_ao);
                    jm
                }
                None => m.to_owned(),
            };
            Some((o.as_raw_ao(), jm_ao))
        }
        _ => None,
    }
}

/// VMC diagonal local-energy position gradient `(de_L/dR (n_atom,3), de_L/dr (n_e,3))`
/// = the `gfmc_diag_grads` smooth base (dT + dV_bare + DTN kinetic), reusing `g_inv`.
/// `Some(g_inv)` == the sweep SM inverse (observable path); `None` == fresh LU.
#[allow(clippy::too_many_arguments)]
pub fn vmc_diag_grads(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let out = vmc_diag_grads_with_forward(
        gem,
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        z_eff,
        r_up,
        r_dn,
        nuc_up,
        nuc_dn,
        nuc_j3,
        g_inv,
        FastScreen::default(),
        VmcShare::default(),
        None,
    );
    (out.d_r, out.d_re)
}

/// What [`vmc_diag_grads_with_forward`] also hands back from its kinetic forward.
#[derive(Clone, Copy, Default, Debug)]
pub struct VmcShare {
    /// The Pulay `(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)` of
    /// `gfmc_grad_ln_psi_gi` at the same configuration and inverse.
    pub pulay: bool,
    /// The base fields of the ECP mesh force ([`MeshShared`]).
    pub mesh: bool,
}

/// Output of [`vmc_diag_grads_with_forward`].
pub struct VmcDiagOut {
    pub d_r: Array2<f64>,
    pub d_re: Array2<f64>,
    pub pulay: Option<(Array2<f64>, Array2<f64>, Array2<f64>)>,
    pub mesh: Option<MeshShared>,
}

/// Variant of [`vmc_diag_grads`] whose determinant forward fields use
/// `forward_gem`, while AO-centre derivative contractions use the Cartesian
/// `gem`.  This distinction matters in mixed precision because rounding a
/// spherical AO value before its MO projection is not equivalent to first
/// lowering the AO to Cartesian components and then rounding those components.
#[allow(clippy::too_many_arguments)]
pub fn vmc_diag_grads_with_forward(
    gem: &GeminalData,
    forward_gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    g_inv: Option<&ArrayView2<f64>>,
    screen: FastScreen,
    share: VmcShare,
    det_tables: Option<crate::param_grad_el::MoDerivTables>,
) -> VmcDiagOut {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = positions.nrows();
    // The J3 the ECP mesh receives is the caller's leaf: share its tables only
    // when lowering is the identity (raw-AO J3).
    let j3_raw_ao = j3_orb.is_none_or(|o| o.mo_coefficients.is_none());
    // Lower an MO-wrapped J3 to its AO-equivalent (see `lower_j3_to_ao`) so the
    // AO-basis J3 force kernels stay exact for a production `mos_data` J3.
    let _j3_owned: Option<(OrbData, Array2<f64>)> =
        crate::timed!("f.hf_lower_j3", lower_j3_to_ao(j3_orb, j3_jm));
    let _j3_jm_view = _j3_owned.as_ref().map(|(_, m)| m.view());
    let (j3_orb, j3_jm): (Option<&OrbData>, Option<&ArrayView2<f64>>) =
        match (&_j3_owned, &_j3_jm_view) {
            (Some((o, _)), Some(v)) => (Some(o), Some(v)),
            _ => (None, None),
        };
    let j3 = match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => Some((o, m)),
        _ => None,
    };
    let j1 = jastrow_full.one_body.as_ref();
    let j2 = jastrow_full.two_body.as_ref();
    let dtn = jastrow_full.dtn.as_ref();
    let has_j = j1.is_some() || j2.is_some() || dtn.is_some() || j3.is_some();
    let jas_opt: Option<&JastrowData> = if has_j { Some(jastrow_full) } else { None };

    // Shared kinetic fields (jastrow_grads / geminal_reverse / det-grad-field /
    // fused det & J3 AO derivative bundles) computed ONCE and threaded through both
    // the dR (grad_t) and dr (tfull) kernels below (w = None => full dT).
    let kin = crate::timed!(
        "f.hf_kin_shared",
        crate::forces::KinShared::build_with_forward(
            gem,
            forward_gem,
            jas_opt,
            j3_orb,
            j3_jm,
            r_up,
            r_dn,
            None,
            g_inv,
            screen,
            Some(crate::forces::KinFuse {
                nuc_up,
                nuc_dn,
                nuc_j3,
                n_atom,
            }),
            det_tables,
        )
    );
    // DTN cotangent cot = gradlnPsi = jastrow_grads(incl DTN) + det_grad_field, with
    // the det field on the reused SM inverse (mirrors gfmc_diag_grads' cot); both
    // fields are the ones KinShared already evaluated.
    let cot = if dtn.is_some() {
        let mut c = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for d in 0..3 {
                c[[e, d]] = if e < n_up {
                    kin.gj_up[[e, d]] + kin.gd_up[[e, d]]
                } else {
                    let l = e - n_up;
                    kin.gj_dn[[l, d]] + kin.gd_dn[[l, d]]
                };
            }
        }
        Some(c)
    } else {
        None
    };
    // dR: dT/dR (with Jastrow) + dV_bare/dR
    let mut d_r = crate::timed!(
        "f.hf_dR_kin",
        crate::forces::grad_t_wrt_r_w_gi_shared(
            gem,
            jas_opt,
            j1,
            j3_orb,
            j3_jm,
            r_up,
            r_dn,
            nuc_up,
            nuc_dn,
            nuc_j3,
            n_atom,
            None,
            g_inv,
            Some(&kin),
        )
    );
    d_r += &grad_v_wrt_r(positions, z_eff, r_up, r_dn);
    // dr: dT/dr (with Jastrow) + dV_bare/dr
    let (tu, td) = crate::timed!(
        "f.hf_dr_kin",
        crate::stress::tfull_grad_wrt_r_w_gi_shared(
            gem,
            jas_opt,
            j1,
            j2,
            j3_orb,
            j3_jm,
            r_up,
            r_dn,
            None,
            g_inv,
            Some(&kin)
        )
    );
    let (vu, vd) = vbare_grad_wrt_r(positions, z_eff, r_up, r_dn);
    let mut d_re = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            d_re[[e, c]] = if e < n_up {
                tu[[e, c]] + vu[[e, c]]
            } else {
                let l = e - n_up;
                td[[l, c]] + vd[[l, c]]
            };
        }
    }
    // DTN kinetic-force channel: dT_DTN/dR and dT_DTN/dr with cot=gradlnPsi.
    if let (Some(d), Some(cot)) = (dtn, &cot) {
        d_r += &crate::forces_dtn::grad_t_dtn_wrt_r_atom(d, cot, r_up, r_dn);
        d_re += &crate::forces_dtn::grad_t_dtn_wrt_r(d, cot, r_up, r_dn);
    }
    // The determinant AO gradient tables at (r_up, r_dn) are handed back for the
    // ECP mesh base Pulay (the same `gem` leaves; see `ecp_nonlocal_force_mol`).
    if !(share.pulay || share.mesh) || g_inv.is_none() {
        return VmcDiagOut {
            d_r,
            d_re,
            pulay: None,
            mesh: None,
        };
    }
    // ---- fields shared with the Pulay force and the ECP mesh base ----
    // (g_inv = Some: the forward's `ou`/`od` are the standalone value tables.)
    let _t_sh = crate::prof::on().then(std::time::Instant::now);
    let crate::forces::KinShared {
        gj_up: kin_gj_up,
        gj_dn: kin_gj_dn,
        bundle_up,
        bundle_dn,
        det_grad_up,
        det_grad_dn,
        j3: kin_j3,
        fwd,
        fused,
        ..
    } = kin;
    let ga_up = crate::forces::take_grad(det_grad_up, bundle_up);
    let ga_dn = crate::forces::take_grad(det_grad_dn, bundle_dn);
    // Determinant Pulay cotangents C^T WG, C^T GQ[:N_dn]^T: ONE value contraction
    // with the AO gradients gives the Pulay (-Sum_k) and the exact drift (Sum_c).
    let (cm_up, cm_dn) = match fused.and_then(|fz| fz.pulay_bars) {
        Some(bars) => bars,
        None => (
            crate::forces::pull_to_ao(&gem.orb_up, &fwd.wg),
            crate::forces::pull_to_ao(
                &gem.orb_dn,
                &fwd.gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(),
            ),
        ),
    };
    let (s_up, det_du) = crate::forces::val_contract(&cm_up, &ga_up);
    let (s_dn, det_dd) = crate::forces::val_contract(&cm_dn, &ga_dn);
    // J3 value tables (standalone `compute`, as `grad_j3_wrt_r` / the mesh base),
    // AO gradients (the kinetic bundle's exact value derivative) and the kernel
    // coefficient `dJ3/dA` (the Pulay cotangent, and the drift's: dJ3/dr = coef.gradA).
    let j3_tab = match (j3_orb, kin_j3) {
        (Some(orb), Some(sj)) => Some(MeshJ3Tables {
            a_up: orb.compute(r_up),
            a_dn: orb.compute(r_dn),
            ga_up: sj.bundle_up.grad,
            ga_dn: sj.bundle_dn.grad,
        }),
        _ => None,
    };
    let j3_coef = match (&j3_tab, j3_jm) {
        (Some(t), Some(jm)) => Some(crate::forces::j3_coef(jm, &t.a_up, &t.a_dn)),
        _ => None,
    };
    // Exact (unscreened) Jastrow drift `jastrow_grads`. The FAST build screened only
    // its J3 part, so without that screen the build's field IS it; with it, the
    // J1/J2/DTN part plus `coef.gradA` (the fp64 `jastrow_grads` J3 kernel's
    // contraction; its f32 zone keeps the kernel itself).
    let (gj_up, gj_dn) = if !has_j {
        (Array2::zeros((n_up, 3)), Array2::zeros((n_dn, 3)))
    } else if screen.j3 && j3.is_some() {
        match (&j3_tab, &j3_coef, j3_orb) {
            (Some(t), Some((cu, cd)), Some(orb))
                if !orb.precision_policy.is_f32("jastrow_grad_lap") =>
            {
                let (pu, pd, _, _) =
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(
                        jastrow_full,
                        r_up,
                        r_dn,
                    );
                let (_, eu) = crate::forces::val_contract(cu, &t.ga_up);
                let (_, ed) = crate::forces::val_contract(cd, &t.ga_dn);
                (pu + eu, pd + ed)
            }
            _ => crate::timed!(
                "f.sh_jas_grads",
                jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn)
            ),
        }
    } else {
        (kin_gj_up, kin_gj_dn)
    };
    let pulay = (share.pulay && std::ptr::eq(gem, forward_gem)).then(|| {
        // 1:1 `gfmc_grad_ln_psi_gi`: det + J1 + J3 + DTN Pulay; det + Jastrow drift.
        let mut p = Array2::<f64>::zeros((n_atom, 3));
        crate::forces::scatter_rows_neg(&s_up, nuc_up, &mut p);
        crate::forces::scatter_rows_neg(&s_dn, nuc_dn, &mut p);
        if let Some(j1) = j1 {
            p += &grad_j1_wrt_r(j1, r_up, r_dn);
        }
        if let (Some(t), Some((coef_up, coef_dn)), Some(nuc)) = (&j3_tab, &j3_coef, nuc_j3) {
            let (su, _) = crate::forces::val_contract(coef_up, &t.ga_up);
            let (sd, _) = crate::forces::val_contract(coef_dn, &t.ga_dn);
            let mut pj = Array2::<f64>::zeros((n_atom, 3));
            crate::forces::scatter_rows_neg(&su, nuc, &mut pj);
            crate::forces::scatter_rows_neg(&sd, nuc, &mut pj);
            p += &pj;
        }
        if let Some(d) = dtn {
            p += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, r_up, r_dn);
        }
        (p, &det_du + &gj_up, &det_dd + &gj_dn)
    });
    let mesh = share.mesh.then(|| MeshShared {
        det: MeshDetShared {
            fwd: MeshFwd::from(fwd),
            ga_up,
            ga_dn,
            cm_up,
            cm_dn,
            du: det_du,
            dd: det_dd,
        },
        jas: (has_j && j3_raw_ao).then(|| MeshJasShared {
            gj_up,
            gj_dn,
            j3: j3_tab,
        }),
    });
    if let Some(t) = _t_sh {
        crate::prof::add("f.hf_share", t.elapsed().as_nanos());
    }
    VmcDiagOut {
        d_r,
        d_re,
        pulay,
        mesh,
    }
}

/// VMC molecular AE HF force `(de_L/dR, de_L/dr_up, de_L/dr_dn)` = `vmc_diag_grads`
/// split by spin. ECP adds its local/non-local force in the py caller.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_e_l_ae(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    g_inv: Option<&ArrayView2<f64>>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let (d_r, d_re) = vmc_diag_grads(
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        z_eff,
        r_up,
        r_dn,
        nuc_up,
        nuc_dn,
        nuc_j3,
        g_inv,
    );
    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

// ============================================================================
// PBC (Ewald, Gamma-point) GFMC HF force -- jnone ECP. Periodic analog of
// gfmc_grad_e_l_ecp: periodic det kinetic (dT/dR, dT/dr), Ewald Coulomb force,
// periodic Pulay/drift for the S' mesh terms + ECP nonlocal periodic mesh force.
// jnone => no Jastrow anywhere (drift = det only, ratio = det only, jr = 1).
// ============================================================================

/// Periodic det Pulay `dln|detG|/dR` `(n_atom,3)` at one config (lndet cotangents
/// WG / G_i_n_vQ -> pgto_det_force_side, bar_lap=0).
#[allow(
    clippy::too_many_arguments,
    reason = "per-spin leaves, cell and configuration inputs"
)]
fn pbc_lndet_pulay(
    gem: &GeminalData,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    pbc_lndet_pulay_drift_sel(
        gem, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv, true, false,
    )
    .0
    .unwrap()
}

/// Periodic det Pulay `(n_atom,3)` AND electron-axis drift (`(N_up,3)`, `(N_dn,3)`) at one
/// config from ONE geminal forward and one fused pgto-gradient sweep per leaf (the two
/// channels contract the same cotangents against the same tables). Each output is
/// bit-identical to [`pbc_lndet_pulay`] / [`pbc_lndet_drift`].
#[allow(clippy::too_many_arguments)]
fn pbc_lndet_pulay_drift(
    gem: &GeminalData,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (p, d) = pbc_lndet_pulay_drift_sel(
        gem, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv, true, true,
    );
    let (du, dd) = d.unwrap();
    (p.unwrap(), du, dd)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pbc_lndet_pulay_drift_sel(
    gem: &GeminalData,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
    want_pulay: bool,
    want_drift: bool,
) -> (Option<Array2<f64>>, Option<(Array2<f64>, Array2<f64>)>) {
    let (bar_up, bar_dn) = gem_value_bars_pbc(gem, r_up, r_dn, g_inv);
    let mut pulay = want_pulay.then(|| Array2::<f64>::zeros((n_atom, 3)));
    let mut e_up = want_drift.then(|| Array2::<f64>::zeros((r_up.nrows(), 3)));
    let mut e_dn = want_drift.then(|| Array2::<f64>::zeros((r_dn.nrows(), 3)));
    // Value cotangent only (zero Laplacian / gradient cotangents): gradient tables suffice.
    for (aos, c, home) in up_leaves {
        pgto_det_force_egrad(
            aos,
            c,
            &bar_up,
            None,
            None,
            r_up,
            cell,
            home,
            pulay.as_mut(),
            e_up.as_mut(),
            fast_screens(&gem.orb_up, DerivScreens::FAST),
        );
    }
    for (aos, c, home) in dn_leaves {
        pgto_det_force_egrad(
            aos,
            c,
            &bar_dn,
            None,
            None,
            r_dn,
            cell,
            home,
            pulay.as_mut(),
            e_dn.as_mut(),
            fast_screens(&gem.orb_dn, DerivScreens::FAST),
        );
    }
    let drift = match (e_up, e_dn) {
        (Some(mut e_up), Some(mut e_dn)) => {
            let zu = Array2::<f64>::zeros(bar_up.raw_dim());
            let zd = Array2::<f64>::zeros(bar_dn.raw_dim());
            // B-spline geminal child (fixed cell-fractional grid; electron-axis drift nonzero).
            if let Some(bs) = &gem.orb_up.bspline {
                let bmo = gem.orb_up.bspline_mo_block();
                bspline_det_egrad_side(bs, bmo.as_ref(), &bar_up, &zu, None, r_up, &mut e_up);
            }
            if let Some(bs) = &gem.orb_dn.bspline {
                let bmo = gem.orb_dn.bspline_mo_block();
                bspline_det_egrad_side(bs, bmo.as_ref(), &bar_dn, &zd, None, r_dn, &mut e_dn);
            }
            // Composite plane-wave child (R-independent; electron-axis drift dvalue/dr = -sin(theta)*G).
            let zg_up = [zu.clone(), zu.clone(), zu.clone()];
            let zg_dn = [zd.clone(), zd.clone(), zd.clone()];
            pw_det_egrad_side(&gem.orb_up, &bar_up, &zu, &zg_up, r_up, &mut e_up);
            pw_det_egrad_side(&gem.orb_dn, &bar_dn, &zd, &zg_dn, r_dn, &mut e_dn);
            Some((e_up, e_dn))
        }
        _ => None,
    };
    (pulay, drift)
}

/// Periodic Pulay `dln|Psi|/dR` `(n_atom,3)` at one config: lndet + J1 (Williamson).
/// J2 is R-independent (e-e); J3 periodic is a follow-up (guarded at the caller).
#[allow(clippy::too_many_arguments)]
fn pbc_lnpsi_pulay(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    let det = pbc_lndet_pulay(gem, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv);
    pbc_lnpsi_pulay_from_det(det, jas, j3, cell, n_atom, r_up, r_dn)
}

/// [`pbc_lnpsi_pulay`] given its determinant part `det` (adds J1 / J3 / DTN).
fn pbc_lnpsi_pulay_from_det(
    det: Array2<f64>,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let mut out = det;
    if let Some(j1) = &jas.one_body {
        out += &grad_j1_wrt_r_pbc(j1, r_up, r_dn);
    }
    if let Some(c) = j3 {
        out += &crate::forces_pbc::grad_j3_wrt_r_pbc(
            c.cart_j3,
            &c.t_mol.to_owned(),
            &c.jm,
            r_up,
            r_dn,
            c.home,
            n_atom,
            cell,
            c.pw,
            c.gto_first,
        );
    }
    if let Some(d) = &jas.dtn {
        out += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, r_up, r_dn);
    }
    out
}

/// Periodic determinant electron-axis drift `dln|detG|/dr` (`(N_up,3)`, `(N_dn,3)`) in the
/// SAME full-image cart-leaf convention as the dR Pulay channel (`pbc_lndet_pulay` ->
/// `pgto_det_force_side`) -- the electron-axis mirror, built via `pgto_det_egrad_side`
/// (+ B-spline / PW composite children) with ORBITAL-only cotangents (`bar_orb = WG` up /
/// `GQ^T` dn, `bar_lap = 0`, `bar_gou = 0`).
///
/// The `det_grad_field` route evaluates the orbital grad through `orb_value_grad_lap`,
/// whose batch (`compute_aos_grad_lap`) and single-sphe paths apply the dynamic LEPSBAS
/// per-image cutoff that the cart force leaves (`pgto_deriv`, full static image tiling) do
/// NOT. That single-vs-batch (and cart-vs-sphe) image gap -- `~exp(-LEPSBAS) ~ 1e-9`,
/// Gaussian-curvature-amplified to ~1e-8 -- would break the per-config translation sum rule
/// `sum_I dln|detG|/dR_I = -sum_i dln|detG|/dr_i`. Sharing the cart-leaf `pgto_deriv` on BOTH
/// force channels closes it to fp64 for cart AND sphe. Matches jqmc's JAX convention (static
/// image expansion, no dynamic cutoff), so the native-vs-JAX force parity also tightens.
fn pbc_lndet_drift(
    gem: &GeminalData,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    pbc_lndet_pulay_drift_sel(
        gem, up_leaves, dn_leaves, cell, 0, r_up, r_dn, g_inv, false, true,
    )
    .1
    .unwrap()
}

/// Periodic drift `dln|Psi|/dr` per electron (det + Jastrow incl J3, periodic-aware).
#[allow(clippy::too_many_arguments)]
fn pbc_drift(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let (gd_up, gd_dn) = pbc_lndet_drift(gem, up_leaves, dn_leaves, cell, r_up, r_dn, g_inv);
    pbc_drift_from_det(gd_up, gd_dn, jas, j3, cell, r_up, r_dn)
}

/// [`pbc_drift`] given its determinant part (adds the J1/J2/DTN + periodic J3 drift).
fn pbc_drift_from_det(
    gd_up: Array2<f64>,
    gd_dn: Array2<f64>,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    cell: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    // J1/J2/DTN drift via the shared kernel (J3 passed as None); the periodic J3 drift is
    // computed separately below in the full-image cart-leaf convention so it matches the dR
    // Pulay channel (`grad_j3_wrt_r_pbc`) and the per-config J3 translation sum rule closes.
    // The batch J3 drift (`compute_grads_and_laplacian_jastrow_three_body`) applies the
    // LEPSBAS per-image cutoff the cart J3 leaf does not (~1e-9 gap, curvature-amplified).
    let (mut gj_up, mut gj_dn) = jastrow_grads(Some(jas), None, None, r_up, r_dn);
    if let Some(c) = j3 {
        let (j3u, j3d) = crate::forces_pbc::grad_j3_wrt_r_e_pbc(
            c.cart_j3,
            &c.t_mol.to_owned(),
            &c.jm,
            r_up,
            r_dn,
            cell,
            c.pw,
            c.gto_first,
        );
        gj_up = gj_up + j3u;
        gj_dn = gj_dn + j3d;
    }
    (gd_up + &gj_up, gd_dn + &gj_dn)
}

/// Derivative of the PBC real-Gamma novar=0 measurement local energy.
/// Includes continuous determinant/Jastrow kinetic, Ewald, ECP local and
/// non-local quadrature contributions; propagation S-prime terms are absent.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ecp_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let n_atom = positions.nrows();
    let cell = &ew.cell;
    let inv = inv3(cell);
    let _ = alat;

    // Pure PW/B-spline J3 has no atom-centered J3Pbc context. Its borrowed
    // batch leaf still supplies kinetic cross terms and ECP ratio derivatives.
    // ---- smooth base: periodic det+J kinetic + Ewald ee/ei/ii force ----
    // One base geminal forward shared by the kinetic and the ECP mesh force.
    let fwd = gem_forward_pbc(gem, r_up, r_dn, None);
    let (mut d_r, mut d_re) = crate::timed!(
        "f.kin_withj",
        pbc_kinetic_withj_grads(
            gem,
            jas,
            j3,
            up_leaves,
            dn_leaves,
            cell,
            n_atom,
            r_up,
            r_dn,
            None,
            None,
            batch_j3,
            Some(&fwd),
        )
    );
    // e-e force (d/dr): Ewald.
    let dee = crate::forces_ewald::grad_eee_wrt_r(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        ew.n_max,
    );
    d_re += &dee;
    // ei (d/dr + d/dR), ii (d/dR)
    let (dei_r, dei_ion) = crate::forces_ewald::grad_eei(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    d_re += &dei_r;
    d_r += &dei_ion;
    let dii = crate::forces_ewald::grad_eii_wrt_r(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    d_r += &dii;

    // ---- ECP local periodic force (min-image) ----
    ecp_local_force_pbc(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, cell, &inv, ew.n_max, r_up, r_dn, &mut d_r,
        &mut d_re,
    );

    // ---- ECP non-local periodic mesh force ----
    ecp_nonlocal_force_pbc(
        gem,
        jas,
        j3,
        up_leaves,
        dn_leaves,
        positions,
        r_up,
        r_dn,
        rt,
        cell,
        &inv,
        ew.n_max,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        None,
        batch_j3,
        Some(&fwd),
        &mut d_r,
        &mut d_re,
    );

    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// Compatibility debug entry for [`gfmc_grad_e_l_ecp_pbc`].
/// Both routes share the novar=0 measurement kernel.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ecp_pbc_debug(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    gfmc_grad_e_l_ecp_pbc(
        gem,
        jas,
        j3,
        up_leaves,
        dn_leaves,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        batch_j3,
    )
}

// ============================================================================
// VMC (MCMC) HF force for periodic real-Gamma Ewald, AE and ECP. The VMC
// continuous local-energy gradient de_L/d(R,r) that the MCMC
// driver's `_grad_e_L_rR` emits: EXACTLY the `gfmc_grad_e_l_ecp_pbc` SMOOTH BASE
// (periodic det+J1/J2/J3/DTN kinetic + Ewald ee/ei/ii + ECP local + ECP non-local
// continuum) WITHOUT the LRDMC S' discretization terms (masked kinetic + lattice
// mesh) -- those are the alat-regularization corrections, absent from the continuous
// VMC e_L because LRDMC and VMC differ once an electron kinetic term changes sign.
// Reuses the sweep SM `g_inv` for the kinetic + ECP-nl base config (the periodic e_L
// avoiding a fresh LU that can become NaN near a nucleus. ECP non-local is the NN=1
// continuum (tmove full ratio, dltmove=false), 1:1 the native periodic e_L.
// ============================================================================

/// VMC periodic HF force `(de_L/dR, de_L/dr_up, de_L/dr_dn)`. `ecp=false` => bare
/// Ewald AE (no ECP local/non-local). Reuses `g_inv` (the sweep SM inverse).
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_e_l_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nn: usize,
    g_inv: Option<&ArrayView2<f64>>,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let n_atom = positions.nrows();
    let cell = &ew.cell;
    let inv = inv3(cell);
    // For far-image robustness, the determinant is home-cell-wrapped inside
    // the geminal-value kernels (`gem_forward` / `compute_geminal_all_elements`, jqmc
    // `compute_AOs` convention); the J3 drift/value AO wraps INSIDE
    // `compute_grads_and_laplacian_jastrow_three_body` / `jastrow_value_for_ratio`, and the DTN force
    // kernels MIN-IMAGE their displacements (`forces_dtn::dtn_disp`, matching the DTN value/
    // drift kernels + jqmc `_dtn_scalar_dist_with_derivs`). J1/J2 (Williamson, wrap-invariant),
    // Ewald and ECP geometry (min-image) keep RAW positions here -- jqmc's reference is the
    // raw walker. Validated == JAX at scattered configs several lattice vectors out (~1e-7).
    // ---- smooth base: periodic det+J kinetic (reuse g_inv) + Ewald ee/ei/ii ----
    // One base geminal forward shared by the kinetic and the ECP mesh force.
    let fwd = gem_forward_pbc(gem, r_up, r_dn, g_inv);
    let (mut d_r, mut d_re) = crate::timed!(
        "f.kin_withj",
        pbc_kinetic_withj_grads(
            gem,
            jas,
            j3,
            up_leaves,
            dn_leaves,
            cell,
            n_atom,
            r_up,
            r_dn,
            None,
            g_inv,
            batch_j3,
            Some(&fwd),
        )
    );
    let dee = crate::forces_ewald::grad_eee_wrt_r(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        ew.n_max,
    );
    d_re += &dee;
    let (dei_r, dei_ion) = crate::forces_ewald::grad_eei(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    d_re += &dei_r;
    d_r += &dei_ion;
    let dii = crate::forces_ewald::grad_eii_wrt_r(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    d_r += &dii;
    // ---- ECP local + non-local (continuum: tmove full ratio, dltmove=false) ----
    if ecp {
        ecp_local_force_pbc(
            positions, nuc_l, e_l_p, c_l_p, p_l_p, cell, &inv, ew.n_max, r_up, r_dn, &mut d_r,
            &mut d_re,
        );
        crate::timed!(
            "f.ecp_nl",
            ecp_nonlocal_force_pbc(
                gem,
                jas,
                j3,
                up_leaves,
                dn_leaves,
                positions,
                r_up,
                r_dn,
                rt,
                cell,
                &inv,
                ew.n_max,
                nuc_n,
                am_n,
                e_n,
                c_n,
                p_n,
                weights,
                grid_points,
                nn,
                false,
                g_inv,
                batch_j3,
                Some(&fwd),
                &mut d_r,
                &mut d_re,
            )
        );
    }
    // NO S' block (VMC continuous).
    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// VMC periodic Pulay force `(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)` = the JAX
/// `compute_grad_ln_Psi` (periodic det Pulay + J1/J3/DTN nuclear force; drift = det+J).
/// Pure WF derivative (LRDMC-independent -- no S'), reusing the sweep SM `g_inv`.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_ln_psi_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> GfmcElGrads {
    // Far-image robustness lives inside the kernels: `gem_forward` and
    // `det_grad_field` wrap the determinant electron (jqmc `compute_AOs` convention), the J3
    // drift wraps inside `compute_grads_and_laplacian_jastrow_three_body`, and the DTN chi/f
    // nuclear force min-images (`forces_dtn::dtn_disp`, matching jqmc) -- no position wrap here.
    // The determinant Pulay and drift share one geminal forward and one AO sweep.
    let (det_p, det_du, det_dd) = crate::timed!(
        "pp.lndet",
        pbc_lndet_pulay_drift(gem, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv)
    );
    let d_r = pbc_lnpsi_pulay_from_det(det_p, jas, j3, cell, n_atom, r_up, r_dn);
    let (du, dd) = pbc_drift_from_det(det_du, det_dd, jas, j3, cell, r_up, r_dn);
    GfmcElGrads {
        d_r,
        d_rup: du,
        d_rdn: dd,
    }
}

/// ECP local periodic force (min-image): `dV_local/dr_e` and `dV_local/dR`.
#[allow(clippy::too_many_arguments)]
fn ecp_local_force_pbc(
    positions: &ArrayView2<f64>,
    nuc_l: &[i64],
    exps: &[f64],
    coefs: &[f64],
    powers: &[f64],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    d_r: &mut Array2<f64>,
    d_re: &mut Array2<f64>,
) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    for t in 0..nuc_l.len() {
        let ia = nuc_l[t] as usize;
        let (e, c, p) = (exps[t], coefs[t], powers[t]);
        for el in 0..n_e {
            let re = elec(el);
            // min-image displacement r_e - R_I (d(min-image)/dr = id a.e.)
            let d0 = [
                re[0] - positions[[ia, 0]],
                re[1] - positions[[ia, 1]],
                re[2] - positions[[ia, 2]],
            ];
            let rel = min_image(d0, cell, inv, n_max);
            let r = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let vlp =
                c * ((p - 2.0) * r.powf(p - 3.0) - 2.0 * e * r.powf(p - 1.0)) * (-e * r * r).exp();
            let u = [rel[0] / r, rel[1] / r, rel[2] / r];
            for dd in 0..3 {
                d_re[[el, dd]] += vlp * u[dd];
                d_r[[ia, dd]] -= vlp * u[dd];
            }
        }
    }
}

/// Sherman-Morrison rank-1 determinant update for one moved electron (periodic
/// geminal; generic over the real-Gamma `f64` and the complex-twist `Complex64` scalar,
/// where the algebra is the same with plain transposes, no conjugation). Given the base forward fields (`w`,`q`,`ginv`,`lam_p`, and the
/// base MO value blocks `ou`/`od`) and the moved electron's MO value column
/// `mo_val` (M,1), returns `(det_ratio, wg = W'G'^{-1} (M_up,N), gq = G'^{-1}Q' (N,M_dn))`
/// at the moved config -- the same projected bars a fresh `gem_forward_gi` would emit, but
/// via an O(N^2) rank-1 hop rooted at the shared (AS-regularized) base inverse instead of an
/// O(N^3) LU. Shared verbatim by the mesh FORCE (`MeshDetFastPbc`, complex `MeshDetFastC`)
/// and STRESS (`MeshStressFastPbc`) fast paths (identical algebra).
#[allow(clippy::too_many_arguments)]
fn sm_rank1_bars<T>(
    spin_up: bool,
    li: usize,
    mo_val: &Array2<T>,
    ou: &Array2<T>,
    od: &Array2<T>,
    w: &Array2<T>,
    q: &Array2<T>,
    ginv: &Array2<T>,
    lam_p: Option<&Array2<T>>,
    n: usize,
) -> (T, Array2<T>, Array2<T>)
where
    T: numpy::ndarray::LinalgScalar + PartialEq + std::ops::AddAssign + std::ops::SubAssign,
{
    let (zero, one) = (T::zero(), T::one());
    let mut ginv_new = ginv.clone();
    let ratio;
    if spin_up {
        let delta: Vec<T> = (0..mo_val.nrows())
            .map(|a| mo_val[[a, 0]] - ou[[a, li]])
            .collect();
        let mut vv = vec![zero; n];
        for j in 0..n {
            let mut s = zero;
            for a in 0..w.nrows() {
                s += w[[a, j]] * delta[a];
            }
            vv[j] = s;
        }
        let ainv_u: Vec<T> = (0..n).map(|i| ginv[[i, li]]).collect();
        let mut rr = one;
        for j in 0..n {
            rr += vv[j] * ainv_u[j];
        }
        ratio = rr;
        let mut vt_ainv = vec![zero; n];
        for i in 0..n {
            let vi = vv[i];
            if vi != zero {
                for j in 0..n {
                    vt_ainv[j] += vi * ginv[[i, j]];
                }
            }
        }
        for i in 0..n {
            let ai = ainv_u[i] / ratio;
            if ai == zero {
                continue;
            }
            for j in 0..n {
                ginv_new[[i, j]] -= ai * vt_ainv[j];
            }
        }
    } else {
        let delta: Vec<T> = (0..mo_val.nrows())
            .map(|b| mo_val[[b, 0]] - od[[b, li]])
            .collect();
        let mut u = vec![zero; n];
        for i in 0..n {
            let mut s = zero;
            for b in 0..q.ncols() {
                s += q[[i, b]] * delta[b];
            }
            u[i] = s;
        }
        let mut ainv_u = vec![zero; n];
        for i in 0..n {
            let mut s = zero;
            for j in 0..n {
                s += ginv[[i, j]] * u[j];
            }
            ainv_u[i] = s;
        }
        ratio = one + ainv_u[li];
        let vt_ainv: Vec<T> = (0..n).map(|j| ginv[[li, j]]).collect();
        for i in 0..n {
            let ai = ainv_u[i] / ratio;
            if ai == zero {
                continue;
            }
            for j in 0..n {
                ginv_new[[i, j]] -= ai * vt_ainv[j];
            }
        }
    }
    let (wg, gq);
    if spin_up {
        let mut q2 = q.clone();
        let m_dn = q.ncols();
        for b in 0..m_dn {
            let mut s = zero;
            for a in 0..mo_val.nrows() {
                let lp = lam_p.map_or(if a == b { one } else { zero }, |l| l[[a, b]]);
                s += mo_val[[a, 0]] * lp;
            }
            q2[[li, b]] = s;
        }
        wg = w.dot(&ginv_new);
        gq = ginv_new.dot(&q2);
    } else {
        let mut w2 = w.clone();
        for a in 0..w.nrows() {
            let mut s = zero;
            for b in 0..q.ncols() {
                let lp = lam_p.map_or(if a == b { one } else { zero }, |l| l[[a, b]]);
                s += lp * mo_val[[b, 0]];
            }
            w2[[a, li]] = s;
        }
        wg = w2.dot(&ginv_new);
        gq = ginv_new.dot(q);
    }
    (ratio, wg, gq)
}

/// The linear part of the [`sm_rank1_bars`] bar deltas. For a hop of electron `li` with
/// moved orbital change `delta` the candidate bars are `base + L(delta) / ratio`, with `L`
/// linear (the moved column of the new inverse is the old one over the ratio, which cancels
/// the quadratic term), so a weighted sum of candidate deltas `Sum_v c_v d_v` is `L(dmo)`
/// for `dmo = Sum_v (c_v / ratio_v) delta_v`. Returns `(L_wg (M_up,N_up), L_bar_dn
/// (M_dn,N_dn))`. Row move (up): `t = ginv^T W^T dmo`, `dq = Lam^T dmo - q^T t`,
/// `L_wg = -wg[:,li] (x) t`, `L_gq = ginv[:,li] (x) dq`. Column move (dn): `s = ginv q dmo`,
/// `dw = Lam dmo - W s`, `L_wg = dw (x) ginv[li,:]`, `L_gq = -s (x) gq[li,:]`.
#[allow(clippy::too_many_arguments)]
fn sm_linear_bars<T>(
    spin_up: bool,
    li: usize,
    dmo: &numpy::ndarray::Array1<T>,
    w: &Array2<T>,
    q: &Array2<T>,
    ginv: &Array2<T>,
    lam_p: Option<&Array2<T>>,
    n_dn: usize,
) -> (Array2<T>, Array2<T>)
where
    T: numpy::ndarray::LinalgScalar + std::ops::AddAssign + std::ops::Neg<Output = T>,
{
    use numpy::ndarray::Array1;
    let n = ginv.nrows();
    let lp = |a: usize, b: usize| -> T {
        lam_p.map_or(if a == b { T::one() } else { T::zero() }, |l| l[[a, b]])
    };
    let m_up = w.nrows();
    let m_dn = q.ncols();
    let mut acc_wg = Array2::<T>::zeros((m_up, n));
    let mut acc_gq = Array2::<T>::zeros((n, m_dn));
    if spin_up {
        let t = ginv.t().dot(&w.t().dot(dmo));
        let mut dq = Array1::<T>::zeros(m_dn);
        for b in 0..m_dn {
            let mut s = T::zero();
            for a in 0..m_up {
                s += dmo[a] * lp(a, b);
            }
            dq[b] = s;
        }
        let dq = dq - q.t().dot(&t);
        let wg_li = w.dot(&ginv.column(li));
        for a in 0..m_up {
            for j in 0..n {
                acc_wg[[a, j]] = -wg_li[a] * t[j];
            }
        }
        for i in 0..n {
            for b in 0..m_dn {
                acc_gq[[i, b]] = ginv[[i, li]] * dq[b];
            }
        }
    } else {
        let s_v = ginv.dot(&q.dot(dmo));
        let mut dw = Array1::<T>::zeros(m_up);
        for a in 0..m_up {
            let mut acc = T::zero();
            for b in 0..m_dn {
                acc += lp(a, b) * dmo[b];
            }
            dw[a] = acc;
        }
        let dw = dw - w.dot(&s_v);
        let gq_li = ginv.row(li).dot(q);
        for a in 0..m_up {
            for j in 0..n {
                acc_wg[[a, j]] = dw[a] * ginv[[li, j]];
            }
        }
        for i in 0..n {
            for b in 0..m_dn {
                acc_gq[[i, b]] = -s_v[i] * gq_li[b];
            }
        }
    }
    let acc_dn = acc_gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned();
    (acc_wg, acc_dn)
}

/// The geminal forward of the force / stress kernels at the base configuration. When the
/// periodic FAST chunk e_L applies its runtime AO screen to both spins ([`fast_screens`]), the
/// forward is built from the e_L's own screened tables instead of the exact AO sweeps: the
/// orbital values of the FAST sweep (`mcmc::fast_orb_matrix`, the values the determinant state
/// and the swept inverse `g_inv` carry) and the FAST folded kinetic gradients / Laplacians
/// (`compute_aos_single_value_grad_lap_fast_cut_folded_into`, the e_L kinetic tables), so the
/// derivative kernels contract the reverse of the screened energy. Without a supplied inverse
/// under an fp32 `ao_eval` zone, G is built from that kernel's own value output instead (the
/// fused convention, #758). Otherwise the exact [`gem_forward_gi`].
pub(crate) fn gem_forward_pbc(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> crate::param_grad_el::GemFwd {
    let fast = fast_screens(&gem.orb_up, DerivScreens::FAST) != DerivScreens::EXACT
        && fast_screens(&gem.orb_dn, DerivScreens::FAST) != DerivScreens::EXACT;
    if !fast {
        return gem_forward_gi(gem, r_up, r_dn, g_inv);
    }
    let tables = |orb: &OrbData, r: &ArrayView2<f64>| {
        let rw = orb.wrap_electrons(r).unwrap_or_else(|| r.to_owned());
        let na = orb.aos.num_ao;
        let ne = rw.nrows();
        // Value table owning G and its inverse. With a supplied running inverse (or an fp64
        // `ao_eval` zone, where the sweep values are the screened e_L values) it is the FAST
        // sweep's own value table. Without one under an fp32 `ao_eval` zone, the sweep values
        // are zone-local fp32 arithmetic while the derivative tables below are fp64, so G, its
        // fresh inverse and those derivatives would describe two different geminal matrices
        // (#758, as `gem_forward_gi` and #681): take the value output of the same FAST kernel
        // instead, rounded to `ao_eval` at the completed AO value (the fused convention).
        let fused_value = g_inv.is_none() && orb.precision_policy.is_f32("ao_eval");
        let ov = (!fused_value).then(|| crate::mcmc::fast_orb_matrix(orb, &rw.view()));
        let mut vt = fused_value.then(|| Array2::<f64>::zeros((na, ne)));
        let mut scratch = Vec::new();
        let (mut v, mut gx, mut gy, mut gz, mut lap) = (
            vec![0.0; na],
            vec![0.0; na],
            vec![0.0; na],
            vec![0.0; na],
            vec![0.0; na],
        );
        // (num_ao, 4 * ne): [grad_x | grad_y | grad_z | lap] for one MO GEMM.
        let mut stacked = Array2::<f64>::zeros((na, 4 * ne));
        for e in 0..ne {
            crate::atomic_orbital::compute_aos_single_value_grad_lap_fast_cut_folded_into(
                &orb.aos,
                [rw[[e, 0]], rw[[e, 1]], rw[[e, 2]]],
                &mut scratch,
                &mut v,
                &mut gx,
                &mut gy,
                &mut gz,
                &mut lap,
            );
            for a in 0..na {
                stacked[[a, e]] = gx[a];
                stacked[[a, ne + e]] = gy[a];
                stacked[[a, 2 * ne + e]] = gz[a];
                stacked[[a, 3 * ne + e]] = lap[a];
            }
            if let Some(vt) = vt.as_mut() {
                crate::determinant::round_to_ao_eval(&mut v);
                for a in 0..na {
                    vt[[a, e]] = v[a];
                }
            }
        }
        let c = orb
            .mo_coefficients
            .as_ref()
            .expect("FAST periodic forward needs MO");
        let y = c.dot(&stacked);
        let sl = |k: usize| {
            y.slice(numpy::ndarray::s![.., k * ne..(k + 1) * ne])
                .to_owned()
        };
        let ov = match (ov, vt) {
            (Some(ov), _) => ov,
            (None, Some(vt)) => c.dot(&vt),
            (None, None) => unreachable!("one FAST value table is always formed"),
        };
        (ov, [sl(0), sl(1), sl(2)], sl(3))
    };
    let (ou, gou, lap_up) = tables(&gem.orb_up, r_up);
    let (od, godn, lap_dn) = tables(&gem.orb_dn, r_dn);
    let m_dn = gem.orb_num_dn;
    let n_dn = r_dn.nrows();
    let n_up = ou.ncols();
    let w = gem.w_from_dn(&od.view(), n_up);
    let ginv = match g_inv {
        Some(gi) => gi.to_owned(),
        None => crate::determinant::inverse(&ou.t().dot(&w).view()),
    };
    let q = gem.up_t_times_paired(&ou.view());
    let wg = w.dot(&ginv);
    let gq = ginv.dot(&q);
    crate::param_grad_el::GemFwd {
        ou_fused: ou.clone(),
        od_fused: od.clone(),
        ou,
        od,
        lap_up,
        lap_dn,
        gou,
        godn,
        w_fused: w.clone(),
        w,
        ginv,
        q_fused: q.clone(),
        q,
        wg,
        gq,
        lam_p: gem
            .lambda
            .as_ref()
            .map(|lam| lam.slice(numpy::ndarray::s![.., 0..m_dn]).to_owned()),
        n_dn,
        m_dn,
    }
}

/// The value-only determinant bars `(W G^-1 (M_up,N_up), ((G^-1 Q)[:n_dn])^T (M_dn,N_dn))` of
/// [`gem_forward_pbc`] (the Pulay / drift contractions need no gradients or Laplacians): on the
/// FAST screened orbital values when [`fast_screens`] applies to both spins, else from the
/// exact [`gem_forward_gi`].
fn gem_value_bars_pbc(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let fast = fast_screens(&gem.orb_up, DerivScreens::FAST) != DerivScreens::EXACT
        && fast_screens(&gem.orb_dn, DerivScreens::FAST) != DerivScreens::EXACT;
    if !fast {
        let f = gem_forward_gi(gem, r_up, r_dn, g_inv);
        let bar_dn = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
        return (f.wg, bar_dn);
    }
    let vals = |orb: &OrbData, r: &ArrayView2<f64>| {
        let rw = orb.wrap_electrons(r).unwrap_or_else(|| r.to_owned());
        if g_inv.is_some() || !orb.precision_policy.is_f32("ao_eval") {
            return crate::mcmc::fast_orb_matrix(orb, &rw.view());
        }
        // No supplied inverse under an fp32 `ao_eval` zone: the value output of the FAST
        // derivative kernels, rounded at the completed AO value, as in [`gem_forward_pbc`].
        let (na, ne) = (orb.aos.num_ao, rw.nrows());
        let mut vt = Array2::<f64>::zeros((na, ne));
        let (mut scratch, mut v) = (Vec::new(), vec![0.0; na]);
        for e in 0..ne {
            crate::atomic_orbital::compute_aos_single_value_fast_cut_folded_into(
                &orb.aos,
                [rw[[e, 0]], rw[[e, 1]], rw[[e, 2]]],
                &mut scratch,
                &mut v,
            );
            crate::determinant::round_to_ao_eval(&mut v);
            for a in 0..na {
                vt[[a, e]] = v[a];
            }
        }
        orb.mo_coefficients
            .as_ref()
            .expect("FAST periodic forward needs MO")
            .dot(&vt)
    };
    let ou = vals(&gem.orb_up, r_up);
    let od = vals(&gem.orb_dn, r_dn);
    let n_up = ou.ncols();
    let n_dn = r_dn.nrows();
    let w = gem.w_from_dn(&od.view(), n_up);
    let ginv = match g_inv {
        Some(gi) => gi.to_owned(),
        None => crate::determinant::inverse(&ou.t().dot(&w).view()),
    };
    let q = gem.up_t_times_paired(&ou.view());
    let gq = ginv.dot(&q);
    (
        w.dot(&ginv),
        gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(),
    )
}

/// The e_L screens a periodic determinant derivative table follows for orbital `o`: `scr`
/// when the native FAST chunk e_L evaluates this orbital with its runtime AO screen (the
/// `periodic_fast_gto` gate of the chunk: periodic MO-GTO, no PW / B-spline child, either
/// `ao_eval` precision) on a Cartesian leaf, whose cart pgto force leaves carry exactly the
/// e_L's shell table; [`DerivScreens::EXACT`] otherwise (molecular, composite, B-spline, and
/// spherical bases, whose e_L screen keys on the spherical shells rather than on the
/// cart-lowered force leaves). Under an fp32 `ao_eval` zone the FAST orbital values are the
/// screened fp32 sweep values (`mcmc::fast_orb_matrix`), the derivative tables stay fp64.
pub(crate) fn fast_screens(o: &OrbData, scr: DerivScreens) -> DerivScreens {
    let fast = o.aos.n_image > 1
        && o.pw.is_none()
        && o.bspline.is_none()
        && o.mo_coefficients.is_some()
        && !o.aos.shells.is_empty()
        && o.aos.kind == crate::atomic_orbital::AoKind::Cart;
    if fast {
        scr
    } else {
        DerivScreens::EXACT
    }
}

/// Periodic (real-Gamma) analogue of [`MeshDetFast`]: cached base
/// determinant fields for the ECP-nonlocal mesh single-electron hops. The periodic
/// determinant Pulay/drift contract the FOLDED cart-pgto gradient (`pgto_grad`, one
/// per force leaf) against `wg`/`gq` -- exactly the shared object the rank-1 SM
/// inverse update produces cheaply. Only the moved electron's pgto-grad column is
/// recomputed; the base inverse is Sherman-Morrison'd (rank-1). Reproduces
/// `pbc_lndet_pulay` (`pgto_det_force_side`) + `pbc_lndet_drift` (`pgto_det_egrad_side`)
/// at the moved config, rooted at the shared base inverse. Pure sphe-pgto scope
/// (composite PW / B-spline children -> the slow fallback at the caller); the mesh
/// caller guards. The det-only (LRDMC SP) branch keeps the full-recompute path.
struct MeshDetFastPbc<'a> {
    orb_up: &'a OrbData,
    orb_dn: &'a OrbData,
    up_leaves: &'a [PgtoLeaf],
    dn_leaves: &'a [PgtoLeaf],
    ou: Array2<f64>,              // (M_up, N_up) periodic MO value (wrapped, LEPSBAS)
    od: Array2<f64>,              // (M_dn, N_dn)
    ga_up: Vec<[Array2<f64>; 3]>, // per up-leaf folded pgto grad (num_cart_ao, N_up)
    ga_dn: Vec<[Array2<f64>; 3]>,
    gou: [Array2<f64>; 3], // (M_up, N_up) MO-level grad (incl PW child; composite drift)
    godn: [Array2<f64>; 3], // (M_dn, N_dn)
    w: Array2<f64>,
    ginv: Array2<f64>,
    q: Array2<f64>,
    lam_p: Option<Array2<f64>>,
    n_up: usize,
    n_dn: usize,
    n_atom: usize,
    // Composite pgto(+)pw: the PW child has NO nuclear dependence (det Pulay from the pgto
    // leaves alone; PW -> 0) but DOES contribute the electron-axis drift -- so a composite
    // candidate reconstructs the drift at MO level (`gou`/`godn`, which include the PW block)
    // instead of the pgto-leaf-only `leaf_drift`. Pure-pgto keeps the leaf drift path.
    is_composite: bool,
    /// The base `ou`/`od`, `G` and its inverse come from the fused value/grad/lap value
    /// table (no supplied running inverse; [`gem_forward_pbc`] -> `gem_forward_gi`). The
    /// moved electron's candidate column must then come from that table's value kernel
    /// too, or the rank-1 det ratio `G'/G` mixes two value tables (#758). A supplied
    /// running inverse keeps the standalone value-only tables it is maintained from.
    fused_base: bool,
    /// Pure pgto: base bars `(w ginv, (ginv q)[:n_dn]^T)` and the base det Pulay / drift
    /// (`base_pulay_drift`), the anchor of the per-candidate rank-1 deltas.
    base_bars: Option<(Array2<f64>, Array2<f64>)>,
    base_det: Option<(Array2<f64>, Array2<f64>, Array2<f64>)>,
    /// Pure pgto: per-leaf shell geometry of the fused moved-electron mesh contraction
    /// ([`crate::mesh_vgs`]); `None` keeps the table sweep.
    geom_up: Option<Vec<crate::mesh_vgs::ShellGeom>>,
    geom_dn: Option<Vec<crate::mesh_vgs::ShellGeom>>,
}

/// Group-level data of the fused pure-pgto candidates ([`MeshDetFastPbc::candidates`]):
/// the moved electron's folded AO value table per moving-side leaf `(n_cao, nv)`, from
/// which the weighted bar deltas of the whole group are formed once
/// ([`MeshDetFastPbc::fused_group_bars`]).
struct DetForceGroup {
    spin_up: bool,
    li: usize,
    vtab: Vec<Array2<f64>>,
}

/// One determinant force candidate of [`MeshDetFastPbc::candidates`]: det ratio, Pulay
/// `(n_atom,3)` and drift (`(N_up,3)`, `(N_dn,3)`). Pure pgto: `pulay` / `du` / `dd` are the
/// base fields plus the moved electron's column (its fresh-minus-base Pulay column and its
/// exact drift row); the rest of the candidate is LINEAR in the bar deltas `d_bars`, which
/// the caller sums with the candidate weights and folds in once per group
/// ([`MeshDetFastPbc::group_delta`]). Composite: exact full fields, `d_bars = None`.
struct DetForceCand {
    ratio: f64,
    pulay: Array2<f64>,
    du: Array2<f64>,
    dd: Array2<f64>,
    d_bars: Option<(Array2<f64>, Array2<f64>)>,
}

impl<'a> MeshDetFastPbc<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gem: &'a GeminalData,
        up_leaves: &'a [PgtoLeaf],
        dn_leaves: &'a [PgtoLeaf],
        cell: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        n_atom: usize,
        g_inv: Option<&ArrayView2<f64>>,
        fwd: Option<&crate::param_grad_el::GemFwd>,
    ) -> Option<Self> {
        // sphe-pgto OR composite pgto(+)pw (the PW child is drift-only, handled at MO level).
        // B-spline is still full-recompute (no analytic pgto-leaf + the spline drift needs a
        // dedicated column update -- out of this fast path).
        if gem.orb_up.bspline.is_some() || gem.orb_dn.bspline.is_some() {
            return None;
        }
        let is_composite = gem.orb_up.pw.is_some() || gem.orb_dn.pw.is_some();
        let f = match fwd {
            Some(f) => f.clone(),
            None => gem_forward_pbc(gem, r_up, r_dn, g_inv),
        };
        let su = fast_screens(&gem.orb_up, DerivScreens::FAST).value;
        let sd = fast_screens(&gem.orb_dn, DerivScreens::FAST).value;
        let ga_up = up_leaves
            .iter()
            .map(|(aos, _, _)| pgto_grad_screened(aos, r_up, cell, su))
            .collect();
        let ga_dn = dn_leaves
            .iter()
            .map(|(aos, _, _)| pgto_grad_screened(aos, r_dn, cell, sd))
            .collect();
        let mut m = MeshDetFastPbc {
            orb_up: &gem.orb_up,
            orb_dn: &gem.orb_dn,
            up_leaves,
            dn_leaves,
            ou: f.ou,
            od: f.od,
            ga_up,
            ga_dn,
            gou: f.gou,
            godn: f.godn,
            w: f.w,
            ginv: f.ginv,
            q: f.q,
            lam_p: f.lam_p,
            n_up: r_up.nrows(),
            n_dn: f.n_dn,
            n_atom,
            is_composite,
            fused_base: g_inv.is_none(),
            base_bars: None,
            base_det: None,
            geom_up: None,
            geom_dn: None,
        };
        if !is_composite {
            let wg = m.w.dot(&m.ginv);
            let bar_dn = m
                .ginv
                .dot(&m.q)
                .slice(numpy::ndarray::s![0..m.n_dn, ..])
                .t()
                .to_owned();
            m.base_det = Some(m.pulay_drift_of(&wg, &bar_dn));
            m.base_bars = Some((wg, bar_dn));
            let geoms = |leaves: &[PgtoLeaf]| -> Option<Vec<crate::mesh_vgs::ShellGeom>> {
                leaves
                    .iter()
                    .map(|(aos, _, home)| crate::mesh_vgs::ShellGeom::new(aos, home, None))
                    .collect()
            };
            if std::env::var("JQMC_RS_MESH_TABLES").is_err() {
                m.geom_up = geoms(up_leaves);
                m.geom_dn = geoms(dn_leaves);
            }
        }
        Some(m)
    }

    /// Pure-pgto candidates from ONE fused value + gradient contraction of the moved
    /// electron over the mesh ([`crate::mesh_vgs::mesh_vgs`], the e_L's mesh loop order),
    /// instead of folded AO value / gradient tables. The moved electron's candidate bar
    /// column is its base column over the det ratio (`ginv' e_li = ginv e_li / ratio` for a
    /// row move, `e_li^T ginv' = e_li^T ginv / ratio` for a column move), so with the
    /// home-rank weights `w = C^T bar_base[:, li]` the contraction gives the det ratio
    /// (`val`), the moved drift row (`grad / ratio`) and the moved Pulay column
    /// (`atom / ratio`). The other bar columns are linear in `delta / ratio`, so the caller
    /// sums `coeff / ratio` over the group and [`Self::fused_group_bars`] forms the weighted
    /// bar deltas once.
    fn candidates_fused(
        &self,
        spin_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        cell: &ArrayView2<f64>,
    ) -> Option<(Vec<DetForceCand>, DetForceGroup)> {
        let geoms = if spin_up {
            self.geom_up.as_ref()?
        } else {
            self.geom_dn.as_ref()?
        };
        let (b_wg, b_bdn) = self.base_bars.as_ref()?;
        let (b_pulay, b_du, b_dd) = self.base_det.as_ref()?;
        let nv = new_positions.len();
        let orb = if spin_up { self.orb_up } else { self.orb_dn };
        let (leaves, ga) = if spin_up {
            (self.up_leaves, &self.ga_up)
        } else {
            (self.dn_leaves, &self.ga_dn)
        };
        let bar = if spin_up { b_wg } else { b_bdn };
        let barcol = bar.column(li);
        let inv = inv3(cell);
        let rmat = Array2::from_shape_vec(
            (nv, 3),
            new_positions
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        )
        .unwrap();
        let rw = crate::mcmc::wrap_rows(&rmat.view(), cell, &inv);
        let pts: Vec<[f64; 3]> = (0..nv)
            .map(|v| [rw[[v, 0]], rw[[v, 1]], rw[[v, 2]]])
            .collect();
        let screen = fast_screens(orb, DerivScreens::MESH).value;
        if screen == AoScreen::None {
            return None;
        }
        let mut out = crate::mesh_vgs::MeshVgs::new(nv, Some(self.n_atom), false, None);
        let mut vtab = Vec::with_capacity(leaves.len());
        // base moved-column Pulay `B[a] = Sum_{c in a} w_c grad phi_c(r_li)` (FAST tables).
        let mut base_col = Array2::<f64>::zeros((self.n_atom, 3));
        for (i, ((aos, c_cart, home), geom)) in leaves.iter().zip(geoms.iter()).enumerate() {
            let w = c_cart.t().dot(&barcol); // (n_cao,)
            let mut o =
                crate::mesh_vgs::MeshVgs::new(nv, Some(self.n_atom), false, Some(aos.num_ao));
            crate::mesh_vgs::mesh_vgs(
                aos,
                &pts,
                crate::mesh_vgs::Weights::Shared(w.as_slice().unwrap()),
                screen,
                Some(geom),
                &mut o,
            );
            for v in 0..nv {
                out.val[v] += o.val[v];
                for d in 0..3 {
                    out.grad[v][d] += o.grad[v][d];
                }
            }
            for (a, b) in out.atom.iter_mut().zip(o.atom.iter()) {
                for d in 0..3 {
                    a[d] += b[d];
                }
            }
            for c in 0..w.len() {
                for d in 0..3 {
                    base_col[[home[c], d]] += w[c] * ga[i][d][[c, li]];
                }
            }
            vtab.push(Array2::from_shape_vec((aos.num_ao, nv), o.vtab).unwrap());
        }
        let cands = (0..nv)
            .map(|v| {
                let ratio = out.val[v];
                let inv_r = 1.0 / ratio;
                let mut pulay = b_pulay.clone();
                for a in 0..self.n_atom {
                    for d in 0..3 {
                        pulay[[a, d]] -=
                            (out.atom[v * self.n_atom + a][d] - base_col[[a, d]]) * inv_r;
                    }
                }
                let mut du = b_du.clone();
                let mut dd = b_dd.clone();
                let mov_d = if spin_up { &mut du } else { &mut dd };
                for d in 0..3 {
                    mov_d[[li, d]] = out.grad[v][d] * inv_r;
                }
                DetForceCand {
                    ratio,
                    pulay,
                    du,
                    dd,
                    d_bars: None,
                }
            })
            .collect();
        Some((cands, DetForceGroup { spin_up, li, vtab }))
    }

    /// Weighted bar deltas `(Sum_v beta_v d_wg_v, Sum_v beta_v d_bar_dn_v)` of a fused
    /// group for `beta_v = coeff_v` (the candidate weight), given `betas[v] = coeff_v /
    /// det_ratio_v`: the Sherman-Morrison bar deltas of [`sm_rank1_bars`] are
    /// `L(delta_v) / ratio_v` with `L` linear in the moved orbital change `delta_v`, so the
    /// sum is `L(D)`, `D = Sum_v betas[v] delta_v = C . (vtab betas) - (Sum betas) o[:, li]`.
    fn fused_group_bars(&self, grp: &DetForceGroup, betas: &[f64]) -> (Array2<f64>, Array2<f64>) {
        let leaves = if grp.spin_up {
            self.up_leaves
        } else {
            self.dn_leaves
        };
        let base_o = if grp.spin_up { &self.ou } else { &self.od };
        let bsum: f64 = betas.iter().sum();
        let beta = numpy::ndarray::ArrayView1::from(betas);
        let mut dmo = base_o.column(grp.li).mapv(|x| -bsum * x);
        for ((_, c_cart, _), vt) in leaves.iter().zip(grp.vtab.iter()) {
            dmo += &c_cart.dot(&vt.dot(&beta));
        }
        sm_linear_bars(
            grp.spin_up,
            grp.li,
            &dmo,
            &self.w,
            &self.q,
            &self.ginv,
            self.lam_p.as_ref(),
            self.n_dn,
        )
    }

    /// Det Pulay and drift of bars `(wg, bar_dn)` contracted with the BASE pgto grad tables.
    fn pulay_drift_of(
        &self,
        wg: &Array2<f64>,
        bar_dn: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut pulay = Array2::<f64>::zeros((self.n_atom, 3));
        let mut du = Array2::<f64>::zeros((self.n_up, 3));
        let mut dd = Array2::<f64>::zeros((self.n_dn, 3));
        for (i, leaf) in self.up_leaves.iter().enumerate() {
            self.leaf_pulay(leaf, wg, &self.ga_up[i], &mut pulay);
        }
        for (i, leaf) in self.dn_leaves.iter().enumerate() {
            self.leaf_pulay(leaf, bar_dn, &self.ga_dn[i], &mut pulay);
        }
        for (i, leaf) in self.up_leaves.iter().enumerate() {
            self.leaf_drift(leaf, wg, &self.ga_up[i], &mut du);
        }
        for (i, leaf) in self.dn_leaves.iter().enumerate() {
            self.leaf_drift(leaf, bar_dn, &self.ga_dn[i], &mut dd);
        }
        (pulay, du, dd)
    }

    /// The bar-delta part of a group's weighted candidates: `(pulay, du, dd)` of the summed
    /// weighted bar deltas `(acc_up, acc_dn)` on the base tables. The moving side's row `li`
    /// is not part of it (each candidate carries its exact moved row); the caller skips it.
    fn group_delta(
        &self,
        acc_up: &Array2<f64>,
        acc_dn: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        self.pulay_drift_of(acc_up, acc_dn)
    }

    /// Pulay (`-` scatter by home atom) / drift (`+` per electron) contraction of
    /// one cart-pgto leaf, given `bar (m, N)` and its folded pgto grad `ga`.
    fn leaf_pulay(
        &self,
        leaf: &PgtoLeaf,
        bar: &Array2<f64>,
        ga: &[Array2<f64>; 3],
        out: &mut Array2<f64>,
    ) {
        let (_, c_cart, home) = leaf;
        let cm = c_cart.t().dot(bar); // (num_cart_ao, N)
        let n = cm.ncols();
        for c in 0..cm.nrows() {
            let ia = home[c];
            for d in 0..3 {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += cm[[c, k]] * ga[d][[c, k]];
                }
                out[[ia, d]] -= acc;
            }
        }
    }

    fn leaf_drift(
        &self,
        leaf: &PgtoLeaf,
        bar: &Array2<f64>,
        ga: &[Array2<f64>; 3],
        out: &mut Array2<f64>,
    ) {
        let (_, c_cart, _) = leaf;
        let cm = c_cart.t().dot(bar);
        let n = cm.ncols();
        for k in 0..n {
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..cm.nrows() {
                    acc += cm[[c, k]] * ga[d][[c, k]];
                }
                out[[k, d]] += acc;
            }
        }
    }

    /// Base-config determinant Pulay `dln|detG|/dR (n_atom,3)` and drift (`(N_up,3)`,
    /// `(N_dn,3)`) from the cached base fields -- bit-identical to `pbc_lndet_pulay` /
    /// `pbc_lndet_drift` at the same `g_inv` (same forward, same pgto-grad tables, same
    /// contraction order), without their two extra geminal forwards and AO sweeps.
    /// Pure pgto only (`None` for a composite, whose drift has a PW block).
    fn base_pulay_drift(&self) -> Option<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        // `None` for a composite (its drift has a PW block); cached at construction.
        self.base_det.clone()
    }

    /// The `nv` mesh candidates of one `(spin_up, local index li)` group, electron `li`
    /// moved to each of `new_positions`. Per candidate
    /// `(det_ratio, det_pulay (n_atom,3), det_drift_up, det_drift_dn)`. The moved
    /// electron's MO values and pgto-grad columns are evaluated for the whole group in
    /// one batched call each (the periodic AO sweep over all image-tiled primitives has a
    /// large fixed cost per call), then each candidate does its rank-1 hop.
    fn candidates(
        &self,
        spin_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        cell: &ArrayView2<f64>,
    ) -> Vec<DetForceCand> {
        let nv = new_positions.len();
        let orb = if spin_up { self.orb_up } else { self.orb_dn };
        let rmat = Array2::from_shape_vec(
            (nv, 3),
            new_positions
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        )
        .unwrap();
        // moved electron MO values (wrapped, LEPSBAS -- matches gem_forward's base ou).
        let rmw = orb
            .wrap_electrons(&rmat.view())
            .unwrap_or_else(|| rmat.clone());
        // Pure pgto: the moved orbital value under the e_L's ECP-mesh screen (the same
        // shell + primitive `pz_val` cut as `dot_aos_multi_cutoff_cart`), so the candidate
        // ratio and its gradient columns below are the value and derivative of ONE
        // screened AO set. A composite keeps the orbital evaluator (its PW block).
        let moved_leaves: &[PgtoLeaf] = if spin_up {
            self.up_leaves
        } else {
            self.dn_leaves
        };
        let mesh_scr = fast_screens(orb, DerivScreens::MESH).value;
        // Moved-electron folded pgto grad columns (raw positions; the pgto kernels wrap
        // themselves) and, under the mesh screen, the moved orbital values from the SAME
        // sweep: `orb = Sum_leaves C_cart . AO_cart`.
        let mut mo_screened: Option<Array2<f64>> = None;
        let moved_cols: Vec<[Array2<f64>; 3]> = moved_leaves
            .iter()
            .map(|(aos, c, _)| {
                if mesh_scr == AoScreen::None {
                    pgto_grad(aos, &rmat.view(), cell)
                } else {
                    let dv = crate::ao_cart_deriv::pgto_derivs_screened(
                        aos,
                        &rmat.view(),
                        cell,
                        true,
                        false,
                        false,
                        DerivScreens::MESH,
                    );
                    let v = c.dot(dv.val.as_ref().unwrap());
                    mo_screened = Some(match mo_screened.take() {
                        None => v,
                        Some(a) => a + &v,
                    });
                    dv.grad
                }
            })
            .collect();
        // Composite: full composite MO value and grad of the moved electron at the
        // HOME-WRAPPED positions (see the drift branch below); the value is the fused table.
        let (mo_fused, mg_all) = if self.is_composite {
            let (mv, mgx, mgy, mgz, _) = crate::determinant::orb_value_grad_lap(orb, &rmw.view());
            (Some(mv), Some([mgx, mgy, mgz]))
        } else {
            (None, None)
        };
        // The moved orbital column comes from the value table of the base `G`: the fused
        // value kernel when the base is fused (`fused_base`), else the standalone value-only
        // kernel of the running inverse. Full precision, pure pgto: the same fp64 AO kernel
        // and MO projection either way (bitwise).
        let mo_all = match (mo_screened, mo_fused) {
            (Some(m), _) => m,
            (None, Some(mv)) if self.fused_base => mv,
            (None, _) if self.fused_base => {
                let av = crate::atomic_orbital::compute_aos_fused_value(&orb.aos, &rmw.view());
                match &orb.mo_coefficients {
                    None => av,
                    Some(c) => crate::determinant::matmul(&c.view(), &av.view()),
                }
            }
            (None, _) => orb.compute(&rmw.view()),
        }; // (M, nv)
        (0..nv)
            .map(|v| {
                let mo_val = mo_all.slice(numpy::ndarray::s![.., v..v + 1]).to_owned();
                if self.base_det.is_some() {
                    self.candidate_rank1(spin_up, li, v, &mo_val, &moved_cols)
                } else {
                    let (ratio, pulay, du, dd) =
                        self.candidate_one(spin_up, li, v, &mo_val, &moved_cols, mg_all.as_ref());
                    DetForceCand {
                        ratio,
                        pulay,
                        du,
                        dd,
                        d_bars: None,
                    }
                }
            })
            .collect()
    }

    /// Pure-pgto candidate `v` of [`Self::candidates`] (see [`DetForceCand`]): the rank-1 bars,
    /// the base Pulay / drift plus the moved electron's column, and the bar deltas.
    fn candidate_rank1(
        &self,
        spin_up: bool,
        li: usize,
        v: usize,
        mo_val: &Array2<f64>,
        moved_cols: &[[Array2<f64>; 3]],
    ) -> DetForceCand {
        let (ratio, wg, gq) = sm_rank1_bars(
            spin_up,
            li,
            mo_val,
            &self.ou,
            &self.od,
            &self.w,
            &self.q,
            &self.ginv,
            self.lam_p.as_ref(),
            self.n_up,
        );
        let bar_dn = gq
            .slice(numpy::ndarray::s![0..self.n_dn, ..])
            .t()
            .to_owned();
        let (b_pulay, b_du, b_dd) = self.base_det.as_ref().unwrap();
        let (b_wg, b_bdn) = self.base_bars.as_ref().unwrap();
        let mut pulay = b_pulay.clone();
        let mut du = b_du.clone();
        let mut dd = b_dd.clone();
        let (mov_leaves, mov_ga, mov_bar) = if spin_up {
            (self.up_leaves, &self.ga_up, &wg)
        } else {
            (self.dn_leaves, &self.ga_dn, &bar_dn)
        };
        // moved column `li`: exact drift row, and the fresh-minus-base Pulay column.
        let barcol = mov_bar.column(li);
        let mut row = [0.0_f64; 3];
        for (i, (_, c_cart, home)) in mov_leaves.iter().enumerate() {
            let cm_li = c_cart.t().dot(&barcol); // (num_cart_ao,)
            for d in 0..3 {
                let fresh = moved_cols[i][d].column(v);
                let base = mov_ga[i][d].column(li);
                for c in 0..cm_li.len() {
                    row[d] += cm_li[c] * fresh[c];
                    pulay[[home[c], d]] -= cm_li[c] * (fresh[c] - base[c]);
                }
            }
        }
        let mov_d = if spin_up { &mut du } else { &mut dd };
        for d in 0..3 {
            mov_d[[li, d]] = row[d];
        }
        DetForceCand {
            ratio,
            pulay,
            du,
            dd,
            d_bars: Some((&wg - b_wg, &bar_dn - b_bdn)),
        }
    }

    /// One candidate of [`Self::candidates`] (column `v` of the batched moved-electron tables).
    fn candidate_one(
        &self,
        spin_up: bool,
        li: usize,
        v: usize,
        mo_val: &Array2<f64>,
        moved_cols: &[[Array2<f64>; 3]],
        mg_all: Option<&[Array2<f64>; 3]>,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        // rank-1 Sherman-Morrison inverse update + det ratio + projected bars (identical
        // algebra to mol; shared with the mesh-stress fast path).
        let (ratio, wg, gq) = sm_rank1_bars(
            spin_up,
            li,
            mo_val,
            &self.ou,
            &self.od,
            &self.w,
            &self.q,
            &self.ginv,
            self.lam_p.as_ref(),
            self.n_up,
        );
        let bar_dn = gq
            .slice(numpy::ndarray::s![0..self.n_dn, ..])
            .t()
            .to_owned(); // (M_dn, N_dn)

        // patched ga (moved column only) for the moved spin.
        let patch = |base: &[[Array2<f64>; 3]]| -> Vec<[Array2<f64>; 3]> {
            base.iter()
                .enumerate()
                .map(|(leaf_i, g)| {
                    let mut g2 = g.clone();
                    for d in 0..3 {
                        for c in 0..g2[d].nrows() {
                            g2[d][[c, li]] = moved_cols[leaf_i][d][[c, v]];
                        }
                    }
                    g2
                })
                .collect()
        };
        let ga_up_c: Vec<[Array2<f64>; 3]>;
        let ga_dn_c: Vec<[Array2<f64>; 3]>;
        let (ga_up_ref, ga_dn_ref) = if spin_up {
            ga_up_c = patch(&self.ga_up);
            (ga_up_c.as_slice(), self.ga_dn.as_slice())
        } else {
            ga_dn_c = patch(&self.ga_dn);
            (self.ga_up.as_slice(), ga_dn_c.as_slice())
        };

        // Pulay (nuclear) contraction over the cart-pgto leaves only -- the PW child has no
        // nuclear dependence, so it contributes ZERO to the det atomic force.
        let mut pulay = Array2::<f64>::zeros((self.n_atom, 3));
        let mut du = Array2::<f64>::zeros((self.n_up, 3));
        let mut dd = Array2::<f64>::zeros((self.n_dn, 3));
        for (i, leaf) in self.up_leaves.iter().enumerate() {
            self.leaf_pulay(leaf, &wg, &ga_up_ref[i], &mut pulay);
        }
        for (i, leaf) in self.dn_leaves.iter().enumerate() {
            self.leaf_pulay(leaf, &bar_dn, &ga_dn_ref[i], &mut pulay);
        }
        if let Some(mg) = mg_all {
            // Drift (electron axis) at MO level: `gou`/`godn` include the PW block (its grad is
            // nonzero), so the pgto-leaf drift alone would miss the PW contribution. Moved
            // column = full composite MO grad at the HOME-WRAPPED new_pos, the same
            // convention `gem_forward_gi` uses for the base `gou`/`godn` (it wraps before
            // `orb_value_grad_lap`) and the moved `mo_val` above uses. The raw position
            // left the static pgto image tiling for a walker outside the home cell, so the
            // moved column lost its pgto block and the ECP mesh force on the ECP atom was
            // wrong while the walker's e_L stayed exact.
            if spin_up {
                let mut gou_c = self.gou.clone();
                for a in 0..gou_c[0].nrows() {
                    for d in 0..3 {
                        gou_c[d][[a, li]] = mg[d][[a, v]];
                    }
                }
                du = drift_up(&gou_c, &wg, self.n_up);
                dd = drift_dn(&self.godn, &gq, self.n_dn);
            } else {
                let mut godn_c = self.godn.clone();
                for b in 0..godn_c[0].nrows() {
                    for d in 0..3 {
                        godn_c[d][[b, li]] = mg[d][[b, v]];
                    }
                }
                du = drift_up(&self.gou, &wg, self.n_up);
                dd = drift_dn(&godn_c, &gq, self.n_dn);
            }
        } else {
            for (i, leaf) in self.up_leaves.iter().enumerate() {
                self.leaf_drift(leaf, &wg, &ga_up_ref[i], &mut du);
            }
            for (i, leaf) in self.dn_leaves.iter().enumerate() {
                self.leaf_drift(leaf, &bar_dn, &ga_dn_ref[i], &mut dd);
            }
        }
        (ratio, pulay, du, dd)
    }
}

/// Strain-STRESS analogue of [`MeshDetFastPbc`]: the ECP-nl mesh-STRESS loop's per-candidate
/// det STRAIN Pulay (3x3) + moved-electron det drift, rooted at the shared (AS-regularized)
/// base geminal inverse via the same rank-1 Sherman-Morrison hop. Each candidate replaces the
/// THREE fresh O(N^3) geminal forwards the full-recompute path does per mesh point
/// (`compute_geminal_all_elements` det ratio + `spp` det Pulay + `det_grad_field` drift) with
/// one O(N^2) SM update + one moved-electron orbital eval.
///
/// Decomposition per candidate:
///  * det ratio, projected bars `wg`/`bar_dn` -> `sm_rank1_bars` (IDENTICAL to the force hop);
///  * det STRAIN Pulay (3x3) = `pgto_det_stress_side` contracting the rank-1 bars over the
///    pgto leaves (the caller adds the config-cheap Jastrow strains). PW / B-spline det VALUE
///    strain is affine (frac) invariant -> 0, so those leaves are simply absent;
///  * moved-electron det drift (3) = the MO-level `det_grad_field` row for the hopped electron
///    (`sum_a gou_moved[:,li]*wg[:,li]`, incl the PW / B-spline block).
///
/// Coverage: pgto, composite pgto(+)pw AND pure-B-spline. Unlike the force fast path, the
/// stress needs NO pgto-leaf force grads (bspline Pulay is 0 and the drift is MO-level), so
/// B-spline IS covered here where [`MeshDetFastPbc`] falls back.
struct MeshStressFastPbc<'a> {
    orb_up: &'a OrbData,
    orb_dn: &'a OrbData,
    ou: Array2<f64>,            // (M_up, N_up) base periodic MO value (wrapped)
    od: Array2<f64>,            // (M_dn, N_dn)
    w: Array2<f64>,             // (M_up, N_up)
    ginv: Array2<f64>,          // (N_up, N_up) base inverse (reused sweep SM inverse)
    q: Array2<f64>,             // (N_up, M_dn)
    lam_p: Option<Array2<f64>>, // dense only; SD identity pairing is implicit
    n_up: usize,
    n_dn: usize,
}

impl<'a> MeshStressFastPbc<'a> {
    fn new(
        gem: &'a GeminalData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        g_inv: Option<&ArrayView2<f64>>,
        fwd: Option<&crate::param_grad_el::GemFwd>,
    ) -> Option<Self> {
        // pgto / composite pgto(+)pw / pure-B-spline all supported (SM ratio via orb value,
        // MO-level drift via orb_value_grad_lap, strain Pulay via the pgto leaves alone).
        let f = match fwd {
            Some(f) => f.clone(),
            None => gem_forward_pbc(gem, r_up, r_dn, g_inv),
        };
        Some(MeshStressFastPbc {
            orb_up: &gem.orb_up,
            orb_dn: &gem.orb_dn,
            ou: f.ou,
            od: f.od,
            w: f.w,
            ginv: f.ginv,
            q: f.q,
            lam_p: f.lam_p,
            n_up: r_up.nrows(),
            n_dn: f.n_dn,
        })
    }

    /// The mesh candidates of one `(spin_up, local index li)` group, electron `li` moved to
    /// each of `new_positions`. Per candidate `(det_ratio, wg (M_up,N_up), bar_dn (M_dn,N_dn),
    /// det_drift [3])`. The bars feed `pgto_det_stress_side` at the caller (moved config);
    /// `det_drift` is the det-only electron-axis drift of the moved electron (add the Jastrow
    /// drift at the caller). The moved electron's MO value/grad columns of the whole group
    /// come from ONE batched `orb_value_grad_lap` call.
    fn candidates(
        &self,
        spin_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        // The moved orbital values and gradients `(M, nv)` when the caller already has them
        // (mesh-screened, from its strain-table sweep); `None` evaluates them here.
        pre: Option<(Array2<f64>, [Array2<f64>; 3])>,
    ) -> Vec<DetCandidate> {
        let nv = new_positions.len();
        let orb = if spin_up { self.orb_up } else { self.orb_dn };
        let rmat = Array2::from_shape_vec(
            (nv, 3),
            new_positions
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        )
        .unwrap();
        // Wrap exactly as gem_forward_gi's base convention (identity for composite/B-spline),
        // so the moved MO value/grad column matches a fresh forward at the moved config.
        let rmw = orb
            .wrap_electrons(&rmat.view())
            .unwrap_or_else(|| rmat.clone());
        // Pure pgto under the e_L's ECP-mesh screen: the caller's strain-table sweep supplies the
        // value and gradient (`pre`); otherwise the orbital evaluator.
        let (mo_all, mg) = if let Some(p) = pre {
            p
        } else {
            let (mo_all, mgx, mgy, mgz, _) =
                crate::determinant::orb_value_grad_lap(orb, &rmw.view());
            (mo_all, [mgx, mgy, mgz])
        };
        (0..nv)
            .map(|v| {
                let mo_val = mo_all.slice(numpy::ndarray::s![.., v..v + 1]).to_owned();
                let (ratio, wg, gq) = sm_rank1_bars(
                    spin_up,
                    li,
                    &mo_val,
                    &self.ou,
                    &self.od,
                    &self.w,
                    &self.q,
                    &self.ginv,
                    self.lam_p.as_ref(),
                    self.n_up,
                );
                let bar_dn = gq
                    .slice(numpy::ndarray::s![0..self.n_dn, ..])
                    .t()
                    .to_owned();
                // moved-electron det drift (MO level, 1:1 det_grad_field row `li`).
                let mut drift = [0.0_f64; 3];
                if spin_up {
                    // gD_up[li,d] = sum_a gou_moved[d][a,li]*wg[a,li]
                    for d in 0..3 {
                        let mut s = 0.0;
                        for a in 0..mg[d].nrows() {
                            s += mg[d][[a, v]] * wg[[a, li]];
                        }
                        drift[d] = s;
                    }
                } else {
                    // gD_dn[li,d] = sum_b godn_moved[d][b,li]*gq[li,b]
                    for d in 0..3 {
                        let mut s = 0.0;
                        for b in 0..mg[d].nrows() {
                            s += mg[d][[b, v]] * gq[[li, b]];
                        }
                        drift[d] = s;
                    }
                }
                (ratio, wg, bar_dn, drift)
            })
            .collect()
    }
}

/// Cached base pgto strain-AO tables of ONE leaf (folded grad `gf`, kernel grad `gk`,
/// wrapped positions `rw`) -- re-used for every mesh candidate whose electrons did not move.
struct DetStrainTables {
    gf: [Array2<f64>; 3], // (num_cart_ao, N) folded gradient
    gk: [Array2<f64>; 3], // (num_ao_kernel, N) kernel gradient
    rw: Array2<f64>,      // (N, 3) wrapped positions
}

/// STRAIN-Pulay analogue of the per-candidate `pgto_det_stress_side` sweep, wrapping
/// [`MeshStressFastPbc`] (its SM rank-1 bars source). The periodic determinant
/// strain Pulay per mesh candidate does not re-evaluate a full
/// periodic AO-gradient sweep (`pgto_grad` + the kernel-rank AO gradient over EVERY electron) at the
/// moved config, because only ONE electron (column `li`) moves per candidate.
///
/// The strain Pulay is BILINEAR in (bars, strain-AO-tables) via [`field_stress`]:
/// `field_stress(C^T bar, gf, gk, rw)` is LINEAR in `bar` AND column-separable over the
/// electron index `k` (term1 sums `(mu,k)`, term2 sums `(kap,k)`, each `k` independent).
/// Decompose per candidate as (base bars + delta bars) x (base tables + li-column swap):
/// ```text
/// S_side(cand_bar, cand_tables)
///   = FS_side(base_bar, base_tables)                    [BASE x BASE, hoisted once/step]
///   + FS_side(cand_bar - base_bar, base_tables)         [delta bars x base tables]
///   + [moving side ONLY] FScol_li(cand_bar[:,li], FRESH tables)
///                      - FScol_li(cand_bar[:,li], base tables)   [li-column table swap]
/// ```
/// The moved electron lives on the MOVING side (up iff `is_up`); its tables' column `li` is
/// re-evaluated FRESH (single-electron `pgto_grad`/kernel-rank AO gradient, batched over the nv mesh
/// points), while every other column reuses the cached base tables. The NON-moving side's
/// electrons never move -> its tables are entirely base, so only the SM-updated (delta) bars
/// enter. `cand_bar` = the [`MeshStressFastPbc::candidate`] SM rank-1 bars; the base bars are
/// `w.dot(ginv)` / `(ginv.dot(q))[:n_dn].T` (the same base gem forward the SM hop roots at, so
/// base + SM-delta == candidate bars exactly). `base_strain` reconstructs the exact base
/// kernel so mesh = base + exact delta (fp reassociation only).
///
/// Coverage matches [`MeshStressFastPbc`] (pgto / composite pgto(+)pw / pure-B-spline): the
/// strain Pulay only sums the pgto `leaves_up`/`leaves_dn` -- PW value strain is affine (frac)
/// invariant (absent leaf) and B-spline has no pgto leaf (empty -> zero base strain and no
/// delta), so all three are covered with no extra gate.
struct MeshDetStressPbc<'a> {
    fast: MeshStressFastPbc<'a>,
    leaves_up: &'a [crate::stress_pbc::StressLeaf],
    leaves_dn: &'a [crate::stress_pbc::StressLeaf],
    tab_up: Vec<DetStrainTables>,
    tab_dn: Vec<DetStrainTables>,
    base_wg: Array2<f64>,     // (M_up, N_up) base up bars
    base_bar_dn: Array2<f64>, // (M_dn, N_dn) base dn bars
    base_strain: Array2<f64>, // (3,3) base det strain Pulay (both sides)
    cell: Array2<f64>,
    positions: Array2<f64>,
    /// Fused mode ([`Self::candidates_fused`]): per-leaf shell geometry (home atoms +
    /// strain offsets) of both spins, and the wrapped base electrons. The base tables
    /// `tab_*` are then not built and `base_strain` is zero: the determinant strain enters
    /// only as the delta of the candidates (the base part cancels against the base Pulay).
    fused: Option<(
        Vec<crate::mesh_vgs::ShellGeom>,
        Vec<crate::mesh_vgs::ShellGeom>,
    )>,
    rw_up: Array2<f64>,
    rw_dn: Array2<f64>,
}

/// Group data of [`MeshDetStressPbc::candidates_fused`]: the moved electron's folded value
/// table per moving-side leaf `(n_cao, nv)` and the fresh li-column strain response per
/// candidate, `FS(C^T bar_base[:,li], fresh column v)` summed over the leaves.
struct DetStressFusedGroup {
    is_up: bool,
    li: usize,
    vtab: Vec<Array2<f64>>,
    strain: Vec<[[f64; 3]; 3]>,
}

impl<'a> MeshDetStressPbc<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gem: &'a GeminalData,
        leaves_up: &'a [crate::stress_pbc::StressLeaf],
        leaves_dn: &'a [crate::stress_pbc::StressLeaf],
        cell: &ArrayView2<f64>,
        positions: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        g_inv: Option<&ArrayView2<f64>>,
        fwd: Option<&crate::param_grad_el::GemFwd>,
    ) -> Option<Self> {
        let fast = MeshStressFastPbc::new(gem, r_up, r_dn, g_inv, fwd)?;
        let inv = inv3(cell);
        let rw_up = crate::mcmc::wrap_rows(r_up, cell, &inv);
        let rw_dn = crate::mcmc::wrap_rows(r_dn, cell, &inv);
        // Fused mode: pure pgto under the FAST e_L screens on both spins.
        let pure = gem.orb_up.pw.is_none()
            && gem.orb_dn.pw.is_none()
            && gem.orb_up.bspline.is_none()
            && gem.orb_dn.bspline.is_none()
            && fast_screens(&gem.orb_up, DerivScreens::MESH).value != AoScreen::None
            && fast_screens(&gem.orb_dn, DerivScreens::MESH).value != AoScreen::None
            && !leaves_up.is_empty()
            && !leaves_dn.is_empty()
            && std::env::var("JQMC_RS_MESH_TABLES").is_err();
        let geoms =
            |leaves: &[crate::stress_pbc::StressLeaf]| -> Option<Vec<crate::mesh_vgs::ShellGeom>> {
                leaves
                    .iter()
                    .map(|(aos, _, home, ker)| {
                        crate::mesh_vgs::ShellGeom::new(aos, home, Some((ker, positions)))
                    })
                    .collect()
            };
        let fused = if pure {
            geoms(leaves_up).zip(geoms(leaves_dn))
        } else {
            None
        };
        if let Some(fz) = fused {
            let base_wg = fast.w.dot(&fast.ginv);
            let base_gq = fast.ginv.dot(&fast.q);
            let base_bar_dn = base_gq
                .slice(numpy::ndarray::s![0..fast.n_dn, ..])
                .t()
                .to_owned();
            return Some(MeshDetStressPbc {
                fast,
                leaves_up,
                leaves_dn,
                tab_up: Vec::new(),
                tab_dn: Vec::new(),
                base_wg,
                base_bar_dn,
                base_strain: Array2::<f64>::zeros((3, 3)),
                cell: cell.to_owned(),
                positions: positions.to_owned(),
                fused: Some(fz),
                rw_up,
                rw_dn,
            });
        }
        // one fused sweep per leaf: kernel-rank grad `gk` + its fold `gf` (== pgto_grad),
        // under the FAST e_L value screen (the base orbital the energy uses).
        let tables = |aos: &AosGtoData, rw: &Array2<f64>, scr: DerivScreens| {
            let t = crate::ao_cart_deriv::pgto_strain_tables(aos, &rw.view(), false, false, scr);
            DetStrainTables {
                gf: t.gf,
                gk: t.gk,
                rw: rw.clone(),
            }
        };
        let tab_up: Vec<DetStrainTables> = leaves_up
            .iter()
            .map(|(aos, _, _, _)| {
                tables(aos, &rw_up, fast_screens(&gem.orb_up, DerivScreens::FAST))
            })
            .collect();
        let tab_dn: Vec<DetStrainTables> = leaves_dn
            .iter()
            .map(|(aos, _, _, _)| {
                tables(aos, &rw_dn, fast_screens(&gem.orb_dn, DerivScreens::FAST))
            })
            .collect();
        // base bars = the base gem-forward reverse (same g_inv-reused inverse the SM hop roots
        // at), so base + SM-delta == candidate bars exactly.
        let base_wg = fast.w.dot(&fast.ginv);
        let base_gq = fast.ginv.dot(&fast.q);
        let base_bar_dn = base_gq
            .slice(numpy::ndarray::s![0..fast.n_dn, ..])
            .t()
            .to_owned();
        // base det strain Pulay (both sides): the exact `pgto_det_stress_side` kernel over the
        // cached base tables (once per step).
        let mut base_strain = Array2::<f64>::zeros((3, 3));
        for ((aos, c, home, ker), tab) in leaves_up.iter().zip(tab_up.iter()) {
            let cm = c.t().dot(&base_wg);
            crate::stress_pbc::field_stress(
                aos,
                &cm,
                &tab.gf,
                &tab.gk,
                &tab.rw,
                home,
                ker,
                positions,
                &mut base_strain,
            );
        }
        for ((aos, c, home, ker), tab) in leaves_dn.iter().zip(tab_dn.iter()) {
            let cm = c.t().dot(&base_bar_dn);
            crate::stress_pbc::field_stress(
                aos,
                &cm,
                &tab.gf,
                &tab.gk,
                &tab.rw,
                home,
                ker,
                positions,
                &mut base_strain,
            );
        }
        Some(MeshDetStressPbc {
            fast,
            leaves_up,
            leaves_dn,
            tab_up,
            tab_dn,
            base_wg,
            base_bar_dn,
            base_strain,
            cell: cell.to_owned(),
            positions: positions.to_owned(),
            fused: None,
            rw_up,
            rw_dn,
        })
    }

    /// Fused-mode candidates of one group: per candidate `(det_ratio, det drift row)` from ONE
    /// fused value + gradient + strain contraction of the moved electron over the mesh
    /// ([`crate::mesh_vgs`]) with the home-rank weights `w = C^T bar_base[:, li]` (the moved
    /// candidate bar column is the base column over the det ratio, see
    /// [`MeshDetFastPbc::candidates_fused`]), plus the group data for the caller's
    /// weighted sums ([`Self::fused_finish`]).
    fn candidates_fused(
        &self,
        is_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
    ) -> Option<FusedDetCandidates> {
        let (gu, gd) = self.fused.as_ref()?;
        let nv = new_positions.len();
        let cell = self.cell.view();
        let inv = inv3(&cell);
        let rmat = Array2::from_shape_fn((nv, 3), |(v, c)| new_positions[v][c]);
        let rw = crate::mcmc::wrap_rows(&rmat.view(), &cell, &inv);
        let pts: Vec<[f64; 3]> = (0..nv)
            .map(|v| [rw[[v, 0]], rw[[v, 1]], rw[[v, 2]]])
            .collect();
        let (leaves, geoms, orb, bar) = if is_up {
            (self.leaves_up, gu, self.fast.orb_up, &self.base_wg)
        } else {
            (self.leaves_dn, gd, self.fast.orb_dn, &self.base_bar_dn)
        };
        let screen = fast_screens(orb, DerivScreens::MESH).value;
        let barcol = bar.column(li);
        let mut val = vec![0.0_f64; nv];
        let mut grad = vec![[0.0_f64; 3]; nv];
        let mut strain = vec![[[0.0_f64; 3]; 3]; nv];
        let mut vtab = Vec::with_capacity(leaves.len());
        for ((aos, c, _, _), geom) in leaves.iter().zip(geoms.iter()) {
            let w = c.t().dot(&barcol);
            let mut o = crate::mesh_vgs::MeshVgs::new(nv, None, true, Some(aos.num_ao));
            crate::mesh_vgs::mesh_vgs(
                aos,
                &pts,
                crate::mesh_vgs::Weights::Shared(w.as_slice().unwrap()),
                screen,
                Some(geom),
                &mut o,
            );
            for v in 0..nv {
                val[v] += o.val[v];
                for a in 0..3 {
                    grad[v][a] += o.grad[v][a];
                    for (s, &os) in strain[v][a].iter_mut().zip(&o.strain[v][a]) {
                        *s += os;
                    }
                }
            }
            vtab.push(Array2::from_shape_vec((aos.num_ao, nv), o.vtab).unwrap());
        }
        let cands = (0..nv)
            .map(|v| {
                let r = val[v];
                (r, [grad[v][0] / r, grad[v][1] / r, grad[v][2] / r])
            })
            .collect();
        Some((
            cands,
            DetStressFusedGroup {
                is_up,
                li,
                vtab,
                strain,
            },
        ))
    }

    /// Fold one fused group into the step accumulators for candidate weights `betas[v] =
    /// coeff_v / det_ratio_v` (`coeff_v` = the candidate's mesh weight): returns the fresh
    /// li-column strain `Sum_v betas[v] strain_v` (3x3), adds the weighted bar deltas
    /// `L(dmo)` ([`sm_linear_bars`]) to `acc`, and `Sum_v betas[v]` to the moved electron's
    /// base-column weight `colw` (the `- FS(bar[:,li], base column li)` swap term, folded into
    /// [`Self::fused_finish`]).
    fn fused_group(
        &self,
        grp: &DetStressFusedGroup,
        betas: &[f64],
        acc: &mut (Array2<f64>, Array2<f64>),
        colw: &mut (Vec<f64>, Vec<f64>),
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((3, 3));
        for (v, st) in grp.strain.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    out[[a, b]] += betas[v] * st[a][b];
                }
            }
        }
        let bsum: f64 = betas.iter().sum();
        let (leaves, base_o) = if grp.is_up {
            (self.leaves_up, &self.fast.ou)
        } else {
            (self.leaves_dn, &self.fast.od)
        };
        let beta = numpy::ndarray::ArrayView1::from(betas);
        let mut dmo = base_o.column(grp.li).mapv(|x| -bsum * x);
        for ((_, c, _, _), vt) in leaves.iter().zip(grp.vtab.iter()) {
            dmo += &c.dot(&vt.dot(&beta));
        }
        let (aw, ad) = sm_linear_bars(
            grp.is_up,
            grp.li,
            &dmo,
            &self.fast.w,
            &self.fast.q,
            &self.fast.ginv,
            self.fast.lam_p.as_ref(),
            self.fast.n_dn,
        );
        acc.0 += &aw;
        acc.1 += &ad;
        if grp.is_up {
            colw.0[grp.li] += bsum;
        } else {
            colw.1[grp.li] += bsum;
        }
        out
    }

    /// The base-table part of the fused mode, once per step: `Sum_k FS(C^T (acc[:,k] -
    /// colw[k] bar_base[:,k]), base column k)` over both spins (the weighted delta bars on the
    /// base tables minus the swapped-out base columns), as fused contractions at the base
    /// electrons under the FAST value screen.
    fn fused_finish(
        &self,
        acc: &(Array2<f64>, Array2<f64>),
        colw: &(Vec<f64>, Vec<f64>),
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((3, 3));
        let positions = self.positions.view();
        let cell = self.cell.view();
        for (leaves, a, cw, bar, rw, orb) in [
            (
                self.leaves_up,
                &acc.0,
                &colw.0,
                &self.base_wg,
                &self.rw_up,
                self.fast.orb_up,
            ),
            (
                self.leaves_dn,
                &acc.1,
                &colw.1,
                &self.base_bar_dn,
                &self.rw_dn,
                self.fast.orb_dn,
            ),
        ] {
            let mut wb = a.clone();
            for (k, cw_k) in cw[..wb.ncols()].iter().enumerate() {
                if *cw_k != 0.0 {
                    wb.column_mut(k).scaled_add(-*cw_k, &bar.column(k));
                }
            }
            for (aos, c, home, ker) in leaves {
                crate::stress_pbc::pgto_det_stress_side_screened(
                    aos,
                    c,
                    &wb,
                    &rw.view(),
                    &cell,
                    home,
                    ker,
                    &positions,
                    fast_screens(orb, DerivScreens::FAST),
                    &mut out,
                );
            }
        }
        out
    }

    /// Batched det ratio + moved-electron det drift + det STRAIN Pulay pieces for the nv mesh
    /// candidates of one `(electron li, atom)` group. The candidate strain Pulay is
    /// `base_strain + swap_li + FS(d_up, base up tables) + FS(d_dn, base dn tables)` (the exact
    /// `pgto_det_stress_side` sum at the moved config, reassociated), with `swap_li` the moved
    /// electron's li-column table swap `FS(bar[:,li], fresh column) - FS(bar[:,li], base column)`
    /// and `d_up`/`d_dn` the candidate minus base bars. Every piece is LINEAR in the candidate
    /// bars, so the caller sums its weighted `d_up`/`d_dn` over the step (folded in by ONE
    /// [`Self::delta_strain`] call) and its weighted moved-bar columns `bar_li` over the group
    /// (folded in by ONE [`Self::group_swap_strain`] call on the returned [`DetStressGroup`]),
    /// instead of field sweeps per candidate. Only the SINGLE moved electron's AO strain tables
    /// are re-evaluated (batched over the nv points).
    fn candidates(
        &self,
        is_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
    ) -> (Vec<DetStressCand>, DetStressGroup) {
        let nv = new_positions.len();
        let cell = self.cell.view();
        let inv = inv3(&cell);
        let mut rmat = Array2::<f64>::zeros((nv, 3));
        for (v, p) in new_positions.iter().enumerate() {
            rmat[[v, 0]] = p[0];
            rmat[[v, 1]] = p[1];
            rmat[[v, 2]] = p[2];
        }
        let rw_new = crate::mcmc::wrap_rows(&rmat.view(), &cell, &inv);
        // moving side (the moved electron's side): FRESH single-electron AO strain tables at
        // ALL nv candidate positions in ONE batched AO sweep, per moving-side leaf.
        let mov_leaves = if is_up {
            self.leaves_up
        } else {
            self.leaves_dn
        };
        // under the e_L's ECP-mesh value screen (the moved orbital the energy uses).
        let mov_orb = if is_up {
            self.fast.orb_up
        } else {
            self.fast.orb_dn
        };
        let mesh_scr = fast_screens(mov_orb, DerivScreens::MESH);
        let screened = mesh_scr.value != AoScreen::None;
        // Under the mesh screen the same sweep also gives the moved orbital value and
        // gradient (`orb = Sum_leaves C_cart . AO_cart`) for the rank-1 candidates.
        let mut pre: Option<(Array2<f64>, [Array2<f64>; 3])> = None;
        let fresh: Vec<FreshStrainTables> = mov_leaves
            .iter()
            .map(|(aos, c, _, _)| {
                let t = if screened {
                    crate::ao_cart_deriv::pgto_strain_tables_with_value(
                        aos,
                        &rw_new.view(),
                        mesh_scr,
                    )
                } else {
                    crate::ao_cart_deriv::pgto_strain_tables(
                        aos,
                        &rw_new.view(),
                        false,
                        false,
                        mesh_scr,
                    )
                };
                if screened {
                    let v = c.dot(t.vf.as_ref().unwrap());
                    let g = [c.dot(&t.gf[0]), c.dot(&t.gf[1]), c.dot(&t.gf[2])];
                    pre = Some(match pre.take() {
                        None => (v, g),
                        Some((a, [g0, g1, g2])) => (a + &v, [g0 + &g[0], g1 + &g[1], g2 + &g[2]]),
                    });
                }
                (t.gf, t.gk)
            })
            .collect();
        let cands = self.fast.candidates(is_up, li, new_positions, pre);
        let out = cands
            .into_iter()
            .map(|(ratio, wg, bar_dn, drift)| {
                let cand_mov = if is_up { &wg } else { &bar_dn };
                DetStressCand {
                    ratio,
                    drift,
                    bar_li: cand_mov.column(li).to_owned(),
                    d_up: &wg - &self.base_wg,
                    d_dn: &bar_dn - &self.base_bar_dn,
                }
            })
            .collect();
        (
            out,
            DetStressGroup {
                is_up,
                li,
                fresh,
                rw_new,
            },
        )
    }

    /// The li-column table swap of one group for the weighted moved-bar columns `g_new`
    /// `(M, nv)` (column v = `w_v * bar_v[:,li]`, the candidate's mesh weight times its moved
    /// bar column): `Sum_leaves FS(C^T g_new, fresh tables) - FS(C^T Sum_v g_new[:,v], base
    /// column li)` (3x3), i.e. `Sum_v w_v * swap_li(v)`.
    fn group_swap_strain(&self, grp: &DetStressGroup, g_new: &Array2<f64>) -> Array2<f64> {
        use numpy::ndarray::s;
        let positions = self.positions.view();
        let (mov_leaves, mov_tab) = if grp.is_up {
            (self.leaves_up, &self.tab_up)
        } else {
            (self.leaves_dn, &self.tab_dn)
        };
        let li = grp.li;
        let g_li = g_new
            .sum_axis(numpy::ndarray::Axis(1))
            .insert_axis(numpy::ndarray::Axis(1));
        let col =
            |a: &Array2<f64>, j: usize| -> Array2<f64> { a.slice(s![.., j..j + 1]).to_owned() };
        let mut out = Array2::<f64>::zeros((3, 3));
        for ((aos, c, home, ker), (tab, (gf_f, gk_f))) in
            mov_leaves.iter().zip(mov_tab.iter().zip(grp.fresh.iter()))
        {
            // + fresh moved-electron tables at every candidate position
            let cm_new = c.t().dot(g_new);
            crate::stress_pbc::field_stress(
                aos,
                &cm_new,
                gf_f,
                gk_f,
                &grp.rw_new,
                home,
                ker,
                &positions,
                &mut out,
            );
            // - base li column (folded into base_strain + delta with the BASE AO tables)
            let gf_bc = [
                col(&tab.gf[0], li),
                col(&tab.gf[1], li),
                col(&tab.gf[2], li),
            ];
            let gk_bc = [
                col(&tab.gk[0], li),
                col(&tab.gk[1], li),
                col(&tab.gk[2], li),
            ];
            let rw_bc = tab.rw.slice(s![li..li + 1, ..]).to_owned();
            let neg_cm = c.t().dot(&g_li).mapv(|x| -x);
            crate::stress_pbc::field_stress(
                aos, &neg_cm, &gf_bc, &gk_bc, &rw_bc, home, ker, &positions, &mut out,
            );
        }
        out
    }

    /// Zero accumulators for the weighted delta bars `(acc_up (M_up,N_up), acc_dn (M_dn,N_dn))`.
    fn delta_accumulators(&self) -> (Array2<f64>, Array2<f64>) {
        (
            Array2::<f64>::zeros(self.base_wg.raw_dim()),
            Array2::<f64>::zeros(self.base_bar_dn.raw_dim()),
        )
    }

    /// `Sum_leaves FS(C^T acc, base tables)` over both spins (3x3): the delta-bar part of the
    /// candidate strain Pulay for an accumulated (weighted) delta bar.
    fn delta_strain(&self, acc_up: &Array2<f64>, acc_dn: &Array2<f64>) -> Array2<f64> {
        let positions = self.positions.view();
        let mut out = Array2::<f64>::zeros((3, 3));
        for ((aos, c, home, ker), tab) in self.leaves_up.iter().zip(self.tab_up.iter()) {
            let cm = c.t().dot(acc_up);
            crate::stress_pbc::field_stress(
                aos, &cm, &tab.gf, &tab.gk, &tab.rw, home, ker, &positions, &mut out,
            );
        }
        for ((aos, c, home, ker), tab) in self.leaves_dn.iter().zip(self.tab_dn.iter()) {
            let cm = c.t().dot(acc_dn);
            crate::stress_pbc::field_stress(
                aos, &cm, &tab.gf, &tab.gk, &tab.rw, home, ker, &positions, &mut out,
            );
        }
        out
    }
}

/// One determinant mesh-stress candidate of [`MeshDetStressPbc::candidates`].
struct DetStressCand {
    ratio: f64,
    drift: [f64; 3],
    /// the moved electron's candidate bar column `(M,)` (moving side).
    bar_li: numpy::ndarray::Array1<f64>,
    /// candidate minus base bars (up `(M_up,N_up)`, dn `(M_dn,N_dn)`).
    d_up: Array2<f64>,
    d_dn: Array2<f64>,
}

/// The moved electron's fresh strain tables of one mesh group (see
/// [`MeshDetStressPbc::group_swap_strain`]).
struct DetStressGroup {
    is_up: bool,
    li: usize,
    /// per moving-side leaf: fresh folded / kernel gradient tables over the nv positions.
    fresh: Vec<FreshStrainTables>,
    /// wrapped candidate positions `(nv, 3)`.
    rw_new: Array2<f64>,
}

/// ECP non-local periodic mesh force (min-image). Mirrors compute_ecp_nonlocal_mesh
/// (cell != None) + the molecular gfmc ECP nonlocal force, with periodic Pulay/drift.
/// tmove -> full det*J ratio; dltmove -> split on det-only amplitude (SP: det-only
/// ratio + det-only mesh grads; FN: full). jnone (jas empty) => jr=1, tmove==dltmove.
#[allow(clippy::too_many_arguments)]
fn ecp_nonlocal_force_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    g_inv: Option<&ArrayView2<f64>>,
    // PURE PW/B-spline J3 leaf (home None -> not a cart `j3`). It enters the reflected-config VALUE
    // ratio (`j3ref` below) and the electron drift (`pbc_drift` result), but carries NO nuclear
    // Pulay (dAO/dR = 0), so `pbc_lnpsi_pulay` is untouched. The real twin of the complex
    // `gfmc_ecp_nonlocal_force_complex` `batch_j3` channel.
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
    fwd: Option<&crate::param_grad_el::GemFwd>,
    d_r: &mut Array2<f64>,
    d_re: &mut Array2<f64>,
) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    // Add the PURE leaf J3 electron drift onto a J3-less base drift (`pbc_drift` sees only the cart
    // `j3`, None for these leaves). A no-op when `batch_j3` is None. dR-Pulay stays 0 (dAO/dR = 0).
    let add_batch_drift = |du: Array2<f64>,
                           dd: Array2<f64>,
                           ru: &ArrayView2<f64>,
                           rd: &ArrayView2<f64>|
     -> (Array2<f64>, Array2<f64>) {
        match batch_j3 {
            None => (du, dd),
            Some((orb, jm)) => {
                let (j3u, j3d, _, _) =
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_three_body(
                        orb, jm, ru, rd,
                    );
                (du + &j3u, dd + &j3d)
            }
        }
    };
    let elec = |gi: usize| -> [f64; 3] {
        if gi < n_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let nv = weights.len();
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
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    inv,
                    n_max,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    // `compute_geminal_all_elements` home-cell-wraps the electron INTERNALLY (jqmc `compute_AOs`
    // convention), so a walker roamed far from the home cell does not underflow the det ratio.
    // Cache base determinant fields; each candidate is a single-electron rank-1 hop.
    // `None` (composite PW / B-spline) => fall back to the per-candidate full recompute.
    let mesh_fast = MeshDetFastPbc::new(
        gem, up_leaves, dn_leaves, cell, r_up, r_dn, n_atom, g_inv, fwd,
    );
    // Base log-det anchor of the full-recompute det ratio (only without the rank-1 cache).
    let (sd, ld) = if mesh_fast.is_none() {
        det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view())
    } else {
        (1.0, 0.0)
    };
    // The reflected-config VALUE ratio's J3 comes from EITHER a cart-lowerable `j3` OR a PURE
    // PW/B-spline `batch_j3` leaf; both feed `jastrow_value_for_ratio` identically as `(&OrbData, &j_matrix)`.
    let (j3o, j3m): (Option<&OrbData>, Option<Array2<f64>>) = match (j3, batch_j3) {
        (Some(c), _) => (Some(c.orb), Some(c.jm.to_owned())),
        (None, Some((o, m))) => (Some(o), Some(m.to_owned())),
        (None, None) => (None, None),
    };
    let j3m_v = j3m.as_ref().map(|m| m.view());
    let j3ref = j3m_v.as_ref().map(|m| (j3o.unwrap(), m));
    // Cache base Jastrow fields (value / J3 coupling / Pulay /
    // drift anchors); each candidate's periodic Jastrow value/Pulay/drift becomes a
    // one-electron delta instead of a full `jastrow_value_for_ratio` + `grad_j3_wrt_r_pbc` +
    // `grad_j3_wrt_r_e_pbc` recompute. `None` (DTN / MO-J3 / composite / B-spline) or
    // `JQMC_RS_ECPF_SLOW=1` -> the full-recompute reference path below.
    let has_j = jas.one_body.is_some()
        || jas.two_body.is_some()
        || jas.dtn.is_some()
        || j3.is_some()
        || batch_j3.is_some();
    // A pure PW/B-spline `batch_j3` leaf (`j3 == None`) cannot use the one-electron Jastrow path:
    // `MeshJasFastPbc::new` is passed the cart `j3` (None here) and would silently drop the batch
    // J3 from its candidate value/Pulay/drift. Force the full-recompute reference path (which threads
    // the batch J3 through `j3ref` / `add_batch_drift`) whenever a batch leaf is active.
    let mesh_jas = if has_j && batch_j3.is_none() && std::env::var("JQMC_RS_ECPF_SLOW").is_err() {
        crate::timed!(
            "ecpf.jas_base_pbc",
            MeshJasFastPbc::new(jas, j3, cell, r_up, r_dn, n_atom)
        )
    } else {
        None
    };
    // Base jastrow value: ONLY the fallback per-candidate `exp(jnew - jd)` ratio needs
    // it (the O(Ne^2*nao^2) periodic J3 bilinear); the one-electron path emits the
    // ratio directly, so never pay it when the cache covers this configuration.
    let jd = if mesh_jas.is_some() {
        0.0
    } else {
        crate::coulomb_potential::jastrow_value_for_ratio(jas, j3ref, r_up, r_dn)
    };
    // base ln|Psi| gradients: full (det+J) and det-only (for the dltmove SP branch).
    // The BASE config reuses `g_inv` (VMC observable, near-singular-stable); the
    // moved-config mesh re-evals below stay fresh (a shifted electron -> new geminal).
    // The pure-pgto determinant part comes from the mesh cache's base fields (same
    // forward + pgto-grad tables, bit-identical), so only the Jastrow parts are added here.
    let (base_pulay_f, (base_du_f, base_dd_f)) =
        match mesh_fast.as_ref().and_then(|m| m.base_pulay_drift()) {
            // The one-electron Jastrow cache holds the same base Jastrow Pulay / drift
            // (`base_jp` / `base_gj*`: the J1 + J3 + DTN Pulay and the J1/J2/DTN + cart-J3
            // drift kernels of `pbc_lnpsi_pulay_from_det` / `pbc_drift_from_det`).
            Some((dp, du, dd)) => match mesh_jas.as_ref() {
                Some(mj) => (dp + &mj.base_jp, (du + &mj.base_gju, dd + &mj.base_gjd)),
                None => (
                    pbc_lnpsi_pulay_from_det(dp, jas, j3, cell, n_atom, r_up, r_dn),
                    pbc_drift_from_det(du, dd, jas, j3, cell, r_up, r_dn),
                ),
            },
            None => (
                pbc_lnpsi_pulay(
                    gem, jas, j3, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv,
                ),
                pbc_drift(gem, jas, j3, up_leaves, dn_leaves, cell, r_up, r_dn, g_inv),
            ),
        };
    let (base_du_f, base_dd_f) = add_batch_drift(base_du_f, base_dd_f, r_up, r_dn);
    let (base_pulay_d, base_du_d, base_dd_d) = if dltmove {
        // det-only base: the rank-1 mesh's cached base fields when available.
        match mesh_fast.as_ref().and_then(|m| m.base_pulay_drift()) {
            Some(b) => b,
            None => {
                let p = pbc_lndet_pulay(gem, up_leaves, dn_leaves, cell, n_atom, r_up, r_dn, g_inv);
                let (du, dd) = crate::param_grad_el::det_grad_field_gi(gem, r_up, r_dn, g_inv);
                (p, du, dd)
            }
        }
    } else {
        (
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
        )
    };
    for gi in 0..n_e {
        let is_up = gi < n_up;
        let re = elec(gi);
        for &ia in &nearest_set(re) {
            let rel0 = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let rel = min_image(rel0, cell, inv, n_max);
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let uhat = [rel[0] / nrm, rel[1] / nrm, rel[2] / nrm];
            let li = if is_up { gi } else { gi - n_up };
            // ---- batched one-electron periodic Jastrow fields for all nv mesh points
            // (value ratio + J1+J3 Pulay + full Jastrow drift, base + delta) ----
            let new_positions: Vec<[f64; 3]> = (0..nv)
                .map(|v| {
                    let g = gp[v];
                    [
                        re[0] + rel[0] + nrm * g[0],
                        re[1] + rel[1] + nrm * g[1],
                        re[2] + rel[2] + nrm * g[2],
                    ]
                })
                .collect();
            let jas_res: Option<Vec<JasCandidate>> = mesh_jas.as_ref().map(|mj| {
                crate::timed!(
                    "ecpf.jas_fast_pbc",
                    mj.candidates(is_up, li, &new_positions, r_up, r_dn)
                )
            });
            // batched rank-1 determinant ratio + Pulay/drift of the whole group: the fused
            // mesh contraction for pure pgto, else the table sweep.
            let (det_res, det_grp) = match mesh_fast.as_ref() {
                Some(f) => crate::timed!(
                    "ecpf.det_cand_pbc",
                    match f.candidates_fused(is_up, li, &new_positions, cell) {
                        Some((c, g)) => (Some(c), Some(g)),
                        None => (Some(f.candidates(is_up, li, &new_positions, cell)), None),
                    }
                ),
                None => (None, None),
            };
            let mut betas = vec![0.0_f64; nv];
            // weighted bar deltas of the group's rank-1 candidates (see `DetForceCand`).
            let mut grp_acc: Option<(Array2<f64>, Array2<f64>)> = None;
            for v in 0..nv {
                let g = gp[v];
                let dnv = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let cosv = (-rel[0] * g[0] - rel[1] * g[1] - rel[2] * g[2]) / (nrm * dnv);
                let new_pos = new_positions[v];
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (p, e, c) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-e * nrm * nrm).exp();
                    let v_l = c * nrm.powf(p - 2.0) * expf;
                    let v_lp =
                        c * ((p - 2.0) * nrm.powf(p - 3.0) - 2.0 * e * nrm.powf(p - 1.0)) * expf;
                    let (pl, dpl) = leg_and_deriv(l, cosv);
                    let w = ((2 * l + 1) as f64) * weights[v];
                    ang += v_l * w * pl;
                    aprime += v_lp * w * pl;
                    acos += v_l * w * dpl;
                }
                let mut dang_dri = [0.0_f64; 3];
                for d in 0..3 {
                    dang_dri[d] =
                        aprime * (-uhat[d]) + acos * (g[d] / (nrm * dnv) + cosv * uhat[d] / nrm);
                }
                // Moved config -- only the fallback (full-recompute) branches consume it.
                let mk_mu_md = || {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if is_up {
                        mu[[gi, 0]] = new_pos[0];
                        mu[[gi, 1]] = new_pos[1];
                        mu[[gi, 2]] = new_pos[2];
                    } else {
                        let k = gi - n_up;
                        md[[k, 0]] = new_pos[0];
                        md[[k, 1]] = new_pos[1];
                        md[[k, 2]] = new_pos[2];
                    }
                    (mu, md)
                };
                // Single-electron rank-1 determinant ratio plus Pulay/drift (fast
                // path); `None` (composite) => per-candidate full recompute.
                // Use one-electron periodic Jastrow fields
                // (value ratio + Pulay/drift, base + delta) when `jas_res` is `Some`.
                let jas_v = jas_res.as_ref().map(|jr| &jr[v]);
                let fast = det_res.as_ref().map(|dr| &dr[v]);
                let det_ratio = match fast {
                    Some(fc) => fc.ratio,
                    None => {
                        let (mu, md) = mk_mu_md();
                        let (sn, ln) = det_sign_lnabs(
                            &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                        );
                        sn * sd * (ln - ld).exp()
                    }
                };
                let jr = match jas_v {
                    Some((jr, _, _, _)) => *jr,
                    None => {
                        let (mu, md) = mk_mu_md();
                        (crate::coulomb_potential::jastrow_value_for_ratio(
                            jas,
                            j3ref,
                            &mu.view(),
                            &md.view(),
                        ) - jd)
                            .exp()
                    }
                };
                let det_amp = ang * det_ratio;
                // tmove -> full ratio; dltmove -> det-only (SP, det_amp>=0) / full (FN)
                let use_full = !dltmove || det_amp < 0.0;
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                let coeff = ang * ratio;
                if let (Some(fc), Some(_)) = (fast, det_grp.as_ref()) {
                    betas[v] = coeff / fc.ratio;
                }
                if let Some((d_wg, d_bdn)) = fast.and_then(|fc| fc.d_bars.as_ref()) {
                    match grp_acc.as_mut() {
                        Some((au, ad)) => {
                            au.scaled_add(coeff, d_wg);
                            ad.scaled_add(coeff, d_bdn);
                        }
                        None => grp_acc = Some((d_wg * coeff, d_bdn * coeff)),
                    }
                }
                // mesh ln|Psi| gradients matching the ratio's Jastrow inclusion
                let (pm, dmu, dmd, base_pulay, base_du, base_dd) = if use_full {
                    match fast.map(|fc| (fc.ratio, &fc.pulay, &fc.du, &fc.dd)) {
                        Some((_, pm_det, dmu_det, dmd_det)) => {
                            // Determinant (rank-1) plus Jastrow: the one-electron update (base + delta)
                            // when covered, else the full periodic recompute.
                            if let Some((_, pj, gu, gd)) = jas_v {
                                let pm = pm_det + pj;
                                let dmu = dmu_det + gu;
                                let dmd = dmd_det + gd;
                                (pm, dmu, dmd, &base_pulay_f, &base_du_f, &base_dd_f)
                            } else {
                                let (mu, md) = mk_mu_md();
                                let mut pm = pm_det.clone();
                                if let Some(j1) = &jas.one_body {
                                    pm += &grad_j1_wrt_r_pbc(j1, &mu.view(), &md.view());
                                }
                                if let Some(c) = j3 {
                                    pm += &crate::forces_pbc::grad_j3_wrt_r_pbc(
                                        c.cart_j3,
                                        &c.t_mol.to_owned(),
                                        &c.jm,
                                        &mu.view(),
                                        &md.view(),
                                        c.home,
                                        n_atom,
                                        cell,
                                        c.pw,
                                        c.gto_first,
                                    );
                                }
                                if let Some(d) = &jas.dtn {
                                    pm += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(
                                        d,
                                        &mu.view(),
                                        &md.view(),
                                    );
                                }
                                let (mut gju, mut gjd) =
                                    jastrow_grads(Some(jas), None, None, &mu.view(), &md.view());
                                if let Some(c) = j3 {
                                    let (j3u, j3d) = crate::forces_pbc::grad_j3_wrt_r_e_pbc(
                                        c.cart_j3,
                                        &c.t_mol.to_owned(),
                                        &c.jm,
                                        &mu.view(),
                                        &md.view(),
                                        cell,
                                        c.pw,
                                        c.gto_first,
                                    );
                                    gju = gju + j3u;
                                    gjd = gjd + j3d;
                                }
                                let dmu = dmu_det + &gju;
                                let dmd = dmd_det + &gjd;
                                // A PURE PW/B-spline `batch_j3` leaf (j3 == None here) still contributes
                                // its electron drift (dR-Pulay is 0); mesh_jas was disabled so this
                                // full-Jastrow-recompute branch owns the batch add.
                                let (dmu, dmd) = add_batch_drift(dmu, dmd, &mu.view(), &md.view());
                                (pm, dmu, dmd, &base_pulay_f, &base_du_f, &base_dd_f)
                            }
                        }
                        None => {
                            let (mu, md) = mk_mu_md();
                            let pm = pbc_lnpsi_pulay(
                                gem,
                                jas,
                                j3,
                                up_leaves,
                                dn_leaves,
                                cell,
                                n_atom,
                                &mu.view(),
                                &md.view(),
                                None,
                            );
                            let (dmu, dmd) = pbc_drift(
                                gem,
                                jas,
                                j3,
                                up_leaves,
                                dn_leaves,
                                cell,
                                &mu.view(),
                                &md.view(),
                                None,
                            );
                            let (dmu, dmd) = add_batch_drift(dmu, dmd, &mu.view(), &md.view());
                            (pm, dmu, dmd, &base_pulay_f, &base_du_f, &base_dd_f)
                        }
                    }
                } else if let Some(fc) = fast.filter(|fc| fc.d_bars.is_some() || det_grp.is_some())
                {
                    // det-only (SP) candidate: the same rank-1 determinant fields.
                    (
                        fc.pulay.clone(),
                        fc.du.clone(),
                        fc.dd.clone(),
                        &base_pulay_d,
                        &base_du_d,
                        &base_dd_d,
                    )
                } else {
                    let (mu, md) = mk_mu_md();
                    let pm = pbc_lndet_pulay(
                        gem,
                        up_leaves,
                        dn_leaves,
                        cell,
                        n_atom,
                        &mu.view(),
                        &md.view(),
                        None,
                    );
                    let (dmu, dmd) = det_grad_field(gem, &mu.view(), &md.view());
                    (pm, dmu, dmd, &base_pulay_d, &base_du_d, &base_dd_d)
                };
                let drift_mesh = |ee: usize, c: usize| -> f64 {
                    if ee < n_up {
                        dmu[[ee, c]]
                    } else {
                        dmd[[ee - n_up, c]]
                    }
                };
                let drift_base = |ee: usize, c: usize| -> f64 {
                    if ee < n_up {
                        base_du[[ee, c]]
                    } else {
                        base_dd[[ee - n_up, c]]
                    }
                };
                for d in 0..3 {
                    d_re[[gi, d]] += ratio * dang_dri[d];
                    d_r[[ia, d]] += ratio * (-dang_dri[d]);
                }
                let dm_dot_g: f64 = (0..3).map(|c| drift_mesh(gi, c) * g[c]).sum();
                for ee in 0..n_e {
                    if ee == gi {
                        for d in 0..3 {
                            d_re[[gi, d]] += coeff * (-dm_dot_g * uhat[d] - drift_base(gi, d));
                        }
                    } else {
                        for d in 0..3 {
                            d_re[[ee, d]] += coeff * (drift_mesh(ee, d) - drift_base(ee, d));
                        }
                    }
                }
                for ib in 0..n_atom {
                    for c in 0..3 {
                        let mut val = pm[[ib, c]] - base_pulay[[ib, c]];
                        if ib == ia {
                            val += drift_mesh(gi, c) + dm_dot_g * uhat[c];
                        }
                        d_r[[ib, c]] += coeff * val;
                    }
                }
            }
            // The bar-delta part of the group's rank-1 candidates (linear in the weighted bar
            // deltas): Pulay into `d_r`, drift into every electron row except the moved one,
            // which each candidate carried exactly.
            if let (Some(f), Some(g)) = (mesh_fast.as_ref(), det_grp.as_ref()) {
                grp_acc = Some(f.fused_group_bars(g, &betas));
            }
            if let (Some(f), Some((au, ad))) = (mesh_fast.as_ref(), grp_acc.as_ref()) {
                let (p, gu, gd) = f.group_delta(au, ad);
                *d_r += &p;
                for ee in 0..n_e {
                    if ee == gi {
                        continue;
                    }
                    for d in 0..3 {
                        d_re[[ee, d]] += if ee < n_up {
                            gu[[ee, d]]
                        } else {
                            gd[[ee - n_up, d]]
                        };
                    }
                }
            }
        }
    }
}

// ============================================================================
// PBC withJ periodic kinetic force (det + J1 Williamson + J2 Williamson),
// electron-axis dT/dr AND atom-axis dT/dR, per-electron weighted (S' mask).
// J3 handled separately. jnone reduces to pbc_kinetic_det_grads (jas empty).
// ============================================================================

/// `(d(Sum_e w_e*kc_e)/dR (n_atom,3), d/dr (n_e,3))` for det + J1 + J2 (periodic
/// Williamson). `w=None` => full dT/dtheta. cot = jastrow_grads(J1/J2/DTN)+det_grad_field.
#[allow(clippy::too_many_arguments)]
/// PW kinetic ELECTRON-force side for a composite determinant: contract the det
/// kinetic cotangents (bar_orb/bar_lap/bar_gou, on the MO rank) with the PW
/// electron-derivatives `dvalue/dr_d = -s*G_d`, `dlap/dr_d = g2*s*G_d`,
/// `dgrad_c/dr_d = -G_c*c*G_d` (phi = cos(G*r-phase)). No-op when orb has no PW.
fn pw_det_egrad_side(
    orb: &OrbData,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: &[Array2<f64>; 3],
    r: &ArrayView2<f64>,
    out: &mut Array2<f64>,
) {
    let pw = match &orb.pw {
        Some(p) => p,
        None => return,
    };
    let c_pw_owned = orb.pw_coeff_block();
    let c_pw = c_pw_owned.view();
    // MO-rank cotangents -> PW-rank
    let cm_orb = c_pw.t().dot(bar_orb);
    let cm_lap = c_pw.t().dot(bar_lap);
    let cm_gou = [
        c_pw.t().dot(&bar_gou[0]),
        c_pw.t().dot(&bar_gou[1]),
        c_pw.t().dot(&bar_gou[2]),
    ];
    let n_e = r.nrows();
    for e in 0..n_e {
        let (re0, re1, re2) = (r[[e, 0]], r[[e, 1]], r[[e, 2]]);
        for k in 0..pw.num_ao {
            let (gk0, gk1, gk2) = (pw.gx[k], pw.gy[k], pw.gz[k]);
            let theta = gk0 * re0 + gk1 * re1 + gk2 * re2 - pw.phase[k];
            let (s, cth) = theta.sin_cos();
            let g2 = pw.g2[k];
            let orb_k = cm_orb[[k, e]];
            let lap_k = cm_lap[[k, e]];
            let gou_dot =
                cm_gou[0][[k, e]] * gk0 + cm_gou[1][[k, e]] * gk1 + cm_gou[2][[k, e]] * gk2;
            let gk = [gk0, gk1, gk2];
            for d in 0..3 {
                out[[e, d]] +=
                    (-s * gk[d]) * orb_k + (g2 * s * gk[d]) * lap_k - gou_dot * cth * gk[d];
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "explicit wavefunction, cell and configuration inputs"
)]
pub(crate) fn pbc_kinetic_withj_grads(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    up_leaves: &[PgtoLeaf],
    dn_leaves: &[PgtoLeaf],
    cell: &ArrayView2<f64>,
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
    // The base geminal forward when the caller already has it ([`gem_forward_pbc`]).
    fwd: Option<&crate::param_grad_el::GemFwd>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let fwd_own;
    let fwd = match fwd {
        Some(f) => f,
        None => {
            fwd_own = gem_forward_pbc(gem, r_up, r_dn, g_inv);
            &fwd_own
        }
    };
    // The J3 drift must ride the FULL drift (det cotangents + `cotm`) so its cross terms with the
    // J1/J2/det kinetic force are captured -- for a cart J3 via `c.orb`, for a PURE PW/B-spline J3
    // via `batch_j3` (its cart force/kinetic self-term is served by `j3_leaf_egrad_side` below,
    // NOT the cart `j3_pbc_egrad_side`).
    // det cotangents with the full Jastrow drift incl J3 (bar_gou != 0 for withJ)
    let (gj_up, gj_dn) = match (j3, batch_j3) {
        (Some(c), _) => jastrow_grads(Some(jas), Some(c.orb), Some(&c.jm), r_up, r_dn),
        (None, Some((o, m))) => jastrow_grads(Some(jas), Some(o), Some(m), r_up, r_dn),
        (None, None) => jastrow_grads(Some(jas), None, None, r_up, r_dn),
    };
    let (w_up, w_dn) = match w {
        None => (None, None),
        Some(w) => (Some(w[0..n_up].to_vec()), Some(w[n_up..].to_vec())),
    };
    // `g_inv`: `Some` reuses the caller's running sweep inverse and remains
    // stable near singular configurations; `None` selects a fresh LU for LRDMC.
    let b = crate::param_grad_el::geminal_reverse_from_fwd(
        fwd,
        &gj_up,
        &gj_dn,
        w_up.as_deref(),
        w_dn.as_deref(),
    );
    let mut d_r = Array2::<f64>::zeros((n_atom, 3));
    let mut e_up = Array2::<f64>::zeros((n_up, 3));
    let mut e_dn = Array2::<f64>::zeros((r_dn.nrows(), 3));
    // Nuclear (`d_r`) and electron-axis (`e_*`) sides of each leaf share ONE fused
    // derivative sweep (grad + Hessian + grad-of-Laplacian).
    for (aos, c, home) in up_leaves {
        pgto_det_force_egrad(
            aos,
            c,
            &b.bar_orb_up,
            Some(&b.bar_lap_up),
            Some(&b.bar_gou),
            r_up,
            cell,
            home,
            Some(&mut d_r),
            Some(&mut e_up),
            fast_screens(&gem.orb_up, DerivScreens::FAST),
        );
    }
    for (aos, c, home) in dn_leaves {
        pgto_det_force_egrad(
            aos,
            c,
            &b.bar_orb_dn,
            Some(&b.bar_lap_dn),
            Some(&b.bar_godn),
            r_dn,
            cell,
            home,
            Some(&mut d_r),
            Some(&mut e_dn),
            fast_screens(&gem.orb_dn, DerivScreens::FAST),
        );
    }
    // B-spline geminal: the nuclear det side is 0 (dAO/dR = 0, empty pgto leaves), but the
    // electron-axis kinetic force is nonzero (the electron moves through the fixed grid).
    // Project through the B-spline MO SUB-block (composite-aware; see pbc_kinetic_det_grads).
    if let Some(bs) = &gem.orb_up.bspline {
        let bmo = gem.orb_up.bspline_mo_block();
        bspline_det_egrad_side(
            bs,
            bmo.as_ref(),
            &b.bar_orb_up,
            &b.bar_lap_up,
            Some(&b.bar_gou),
            r_up,
            &mut e_up,
        );
    }
    if let Some(bs) = &gem.orb_dn.bspline {
        let bmo = gem.orb_dn.bspline_mo_block();
        bspline_det_egrad_side(
            bs,
            bmo.as_ref(),
            &b.bar_orb_dn,
            &b.bar_lap_dn,
            Some(&b.bar_godn),
            r_dn,
            &mut e_dn,
        );
    }
    // PW kinetic ELECTRON force for a composite (pgto(+)pw) determinant. The PW is
    // R-independent (no atom force / Pulay), but the grad^2/grad terms depend on the
    // electron position -> an electron-axis kinetic force via `pw_det_egrad`.
    pw_det_egrad_side(
        &gem.orb_up,
        &b.bar_orb_up,
        &b.bar_lap_up,
        &b.bar_gou,
        r_up,
        &mut e_up,
    );
    pw_det_egrad_side(
        &gem.orb_dn,
        &b.bar_orb_dn,
        &b.bar_lap_dn,
        &b.bar_godn,
        r_dn,
        &mut e_dn,
    );
    let mut d_re = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for d in 0..3 {
            d_re[[e, d]] = if e < n_up {
                e_up[[e, d]]
            } else {
                e_dn[[e - n_up, d]]
            };
        }
    }
    // cot = gJ + gD (full drift), per electron; the Jastrow-side kinetic force (J1/J2/J3/DTN,
    // both axes, S'-weighted) is a real reduction of this drift -> factored into the shared
    // `pbc_jastrow_kinetic_grads` (reused by the complex-twist kinetic, cot = gJ + Re(gD)).
    let (gd_up, gd_dn) = crate::param_grad_el::det_grad_field_from_fwd(fwd);
    let mut cotm = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            cotm[[e, c]] = if e < n_up {
                gj_up[[e, c]] + gd_up[[e, c]]
            } else {
                let l = e - n_up;
                gj_dn[[l, c]] + gd_dn[[l, c]]
            };
        }
    }
    let (jr, jre) = pbc_jastrow_kinetic_grads(jas, j3, &cotm, r_up, r_dn, cell, n_atom, w);
    d_r += &jr;
    d_re += &jre;
    // PURE PW/B-spline J3 kinetic SELF force (electron axis; the cart `pbc_jastrow_kinetic_grads`
    // J3 block is inactive here since `j3 == None`). The nuclear (dR) self side is 0 (dAO/dR = 0);
    // the J3 dR contribution already rode `cotm` into the J1/J2/det kinetic force above.
    if let Some((orb, jm)) = batch_j3 {
        j3_leaf_egrad_side(orb, jm, r_up, r_dn, &cotm, w, &mut d_re);
    }
    (d_r, d_re)
}

/// J3 kinetic ELECTRON force `dT_J3/dr` (accumulate into `dt (Ne,3)`) for a PURE PW / B-spline
/// J3 leaf -- the leaf-owning analog of the cart [`crate::forces_pbc::j3_pbc_egrad_side`].
///
/// The J3 bilinear coefficient cotangents (`bcoef`/`bgo`/`blo`, and the `bar_a` M-contraction +
/// cross-spin `bar_s`) are LEAF-INDEPENDENT: they are built from the J3 orbital VALUE/GRAD/LAP
/// (`orb_value_grad_lap`, the same evaluator the determinant kinetic path and the J3 drift use)
/// and the full drift `cot = gJ + gD`. The leaf's second-derivative fields (grad AO,
/// grad(grad AO)=Hessian, grad(grad^2 AO)=gradlap) are supplied by the SAME
/// [`bspline_det_egrad_side`] / [`pw_det_egrad_side`] the determinant kinetic force uses -- the
/// final contraction is identical (`bar_orb.gradAO + bar_lap.grad(grad^2AO) + bar_gou.grad(gradAO)`),
/// so no new second-derivative math is written; `bar_orb = bar_a + bar_s`, `bar_lap = blo`,
/// `bar_gou = bgo`. The NUCLEAR (dR) side is 0 for these atom-center-free leaves (dAO/dR = 0);
/// the J3 dR contribution rides the J1/J2/det cross terms via the full drift `cot` in the caller.
/// `w` = S' mask (None => 1). NB: the electron coordinate is home-cell WRAPPED (matching
/// `compute_grads_and_laplacian_jastrow_three_body`) so `orb_value_grad_lap` and the leaf
/// egrad fields see the SAME in-cell coordinate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn j3_leaf_egrad_side(
    orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cot: &Array2<f64>,
    w: Option<&[f64]>,
    dt: &mut Array2<f64>,
) {
    use numpy::ndarray::{Array1, Axis};
    let ruw = orb.wrap_electrons(r_up).unwrap_or_else(|| r_up.to_owned());
    let rdw = orb.wrap_electrons(r_dn).unwrap_or_else(|| r_dn.to_owned());
    let (ru, rd) = (ruw.view(), rdw.view());
    let (a_up, gxu, gyu, gzu, lo_u) = crate::determinant::orb_value_grad_lap(orb, &ru);
    let (a_dn, gxd, gyd, gzd, lo_d) = crate::determinant::orb_value_grad_lap(orb, &rd);
    let go_u = [gxu, gyu, gzu];
    let go_d = [gxd, gyd, gzd];
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = crate::forces_pbc::k_strict_lower(n_up);
    let kd = crate::forces_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let mk = |a: &Array2<f64>, k: &Array2<f64>, cross: &Array1<f64>| -> Array2<f64> {
        let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
        let nc = a.ncols();
        for p in 0..orb_n {
            for c in 0..nc {
                g[[p, c]] += j1col[p] + cross[p];
            }
        }
        g
    };
    let coef_up = mk(&a_up, &ku, &m_s_dn);
    let coef_dn = mk(&a_dn, &kd, &mt_s_up);
    let side = |coef: &Array2<f64>,
                go: &[Array2<f64>; 3],
                lo: &Array2<f64>,
                off: usize|
     -> (Array2<f64>, [Array2<f64>; 3], Array2<f64>) {
        let nc = coef.ncols();
        let mut bcoef = Array2::<f64>::zeros((orb_n, nc));
        let mut bgo = [
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
            Array2::<f64>::zeros((orb_n, nc)),
        ];
        let mut blo = Array2::<f64>::zeros((orb_n, nc));
        for a in 0..orb_n {
            for k in 0..nc {
                let wk = w.map_or(1.0, |w| w[off + k]);
                let mut acc = 0.0;
                for d in 0..3 {
                    acc += cot[[off + k, d]] * go[d][[a, k]];
                    bgo[d][[a, k]] = -wk * cot[[off + k, d]] * coef[[a, k]];
                }
                bcoef[[a, k]] = wk * (-0.5 * lo[[a, k]] - acc);
                blo[[a, k]] = -0.5 * wk * coef[[a, k]];
            }
        }
        (bcoef, bgo, blo)
    };
    let (bcoef_u, bgo_u, blo_u) = side(&coef_up, &go_u, &lo_u, 0);
    let (bcoef_d, bgo_d, blo_d) = side(&coef_dn, &go_d, &lo_d, n_up);
    // bar_a (value-cotangent, M-contracted) + cross-spin row-sum bar_s, both AT ORBITAL RANK.
    let mut bar_orb_u = m.t().dot(&bcoef_u).dot(&ku.t()) + m.dot(&bcoef_u).dot(&ku);
    let mut bar_orb_d = m.t().dot(&bcoef_d).dot(&kd.t()) + m.dot(&bcoef_d).dot(&kd);
    let bar_s_dn = m.t().dot(&bcoef_u.sum_axis(Axis(1))); // up bcoef -> dn electrons
    let bar_s_up = m.dot(&bcoef_d.sum_axis(Axis(1))); // dn bcoef -> up electrons
    for a in 0..orb_n {
        for k in 0..n_up {
            bar_orb_u[[a, k]] += bar_s_up[a];
        }
        for k in 0..n_dn {
            bar_orb_d[[a, k]] += bar_s_dn[a];
        }
    }
    // Delegate the grad / Hessian / gradlap field contraction to the SAME leaf egrad kernels the
    // determinant kinetic force uses (bar_orb=grad, bar_lap=gradlap, bar_gou=Hessian).
    let mut e_up = Array2::<f64>::zeros((n_up, 3));
    let mut e_dn = Array2::<f64>::zeros((n_dn, 3));
    if let Some(bs) = &orb.bspline {
        let bmo = orb.bspline_mo_block();
        bspline_det_egrad_side(
            bs,
            bmo.as_ref(),
            &bar_orb_u,
            &blo_u,
            Some(&bgo_u),
            &ru,
            &mut e_up,
        );
        bspline_det_egrad_side(
            bs,
            bmo.as_ref(),
            &bar_orb_d,
            &blo_d,
            Some(&bgo_d),
            &rd,
            &mut e_dn,
        );
    }
    if orb.pw.is_some() {
        pw_det_egrad_side(orb, &bar_orb_u, &blo_u, &bgo_u, &ru, &mut e_up);
        pw_det_egrad_side(orb, &bar_orb_d, &blo_d, &bgo_d, &rd, &mut e_dn);
    }
    for e in 0..n_up {
        for d in 0..3 {
            dt[[e, d]] += e_up[[e, d]];
        }
    }
    for e in 0..n_dn {
        for d in 0..3 {
            dt[[n_up + e, d]] += e_dn[[e, d]];
        }
    }
}

/// J3 kinetic STRAIN `dT_J3/deps` (3x3, accumulate into `out`) for a PURE PW / B-spline J3 leaf --
/// the leaf-owning analog of the cart [`crate::stress_pbc::j3_pbc_kstress_w`].
///
/// The J3 bilinear cotangents (`bgo` = gou/gradient field cotangent, `blo` = lap field cotangent)
/// are LEAF-INDEPENDENT: `coef = dJ3/dA` from the orbital VALUE (`orb.compute`) and the drift
/// `bg = -(gJ+gD)`, exactly as `j3_pbc_kstress_w`. The **value** field-strain (`bar_a`) is 0 for
/// these leaves (the PW/B-spline AO value is affine-strain-invariant -- fractional coords fixed),
/// so ONLY the lap + gou metric responses survive; those are supplied by the SAME
/// [`crate::stress_pbc::bspline_kstress_side`] / [`crate::stress_pbc::pw_lap_stress`]+
/// [`crate::stress_pbc::pw_gou_stress`] the determinant kinetic stress uses (no new strain math).
/// `w` = S' mask (None => 1). Electron coords home-cell WRAPPED (value convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn j3_leaf_kstress_side(
    orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    bg_up: &Array2<f64>,
    bg_dn: &Array2<f64>,
    w: Option<&[f64]>,
    out: &mut Array2<f64>,
) {
    use numpy::ndarray::{Array1, Axis};
    let ruw = orb.wrap_electrons(r_up).unwrap_or_else(|| r_up.to_owned());
    let rdw = orb.wrap_electrons(r_dn).unwrap_or_else(|| r_dn.to_owned());
    let (ru, rd) = (ruw.view(), rdw.view());
    let a_up = orb.compute(&ru);
    let a_dn = orb.compute(&rd);
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = crate::forces_pbc::k_strict_lower(n_up);
    let kd = crate::forces_pbc::k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let coef = |a: &Array2<f64>, k: &Array2<f64>, cross: &Array1<f64>| -> Array2<f64> {
        let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
        let nc = a.ncols();
        for p in 0..orb_n {
            for c in 0..nc {
                g[[p, c]] += j1col[p] + cross[p];
            }
        }
        g
    };
    let coef_up = coef(&a_up, &ku, &m_s_dn);
    let coef_dn = coef(&a_dn, &kd, &mt_s_up);
    // Kinetic strain cotangents (bar_a value term = 0 for affine-invariant PW/B-spline value):
    // bgo = w * bg * coef (gou field), blo = -0.5 * w * coef (lap field).
    let side =
        |coef: &Array2<f64>, bg: &Array2<f64>, off: usize| -> ([Array2<f64>; 3], Array2<f64>) {
            let nc = coef.ncols();
            let mut bgo = [
                Array2::<f64>::zeros((orb_n, nc)),
                Array2::<f64>::zeros((orb_n, nc)),
                Array2::<f64>::zeros((orb_n, nc)),
            ];
            let mut blo = Array2::<f64>::zeros((orb_n, nc));
            for a in 0..orb_n {
                for k in 0..nc {
                    let wk = w.map_or(1.0, |v| v[off + k]);
                    for d in 0..3 {
                        bgo[d][[a, k]] = wk * bg[[k, d]] * coef[[a, k]];
                    }
                    blo[[a, k]] = wk * (-0.5) * coef[[a, k]];
                }
            }
            (bgo, blo)
        };
    let (bgo_u, blo_u) = side(&coef_up, bg_up, 0);
    let (bgo_d, blo_d) = side(&coef_dn, bg_dn, n_up);
    if orb.bspline.is_some() {
        crate::stress_pbc::bspline_kstress_side(orb, &blo_u, Some(&bgo_u), &ru, out);
        crate::stress_pbc::bspline_kstress_side(orb, &blo_d, Some(&bgo_d), &rd, out);
    }
    if let Some(pw) = &orb.pw {
        let c_pw = orb.pw_coeff_block();
        let cpw = c_pw.view();
        crate::stress_pbc::pw_lap_stress(pw, &cpw.t().dot(&blo_u), &ru, out);
        crate::stress_pbc::pw_gou_stress(
            pw,
            &[
                cpw.t().dot(&bgo_u[0]),
                cpw.t().dot(&bgo_u[1]),
                cpw.t().dot(&bgo_u[2]),
            ],
            &ru,
            out,
        );
        crate::stress_pbc::pw_lap_stress(pw, &cpw.t().dot(&blo_d), &rd, out);
        crate::stress_pbc::pw_gou_stress(
            pw,
            &[
                cpw.t().dot(&bgo_d[0]),
                cpw.t().dot(&bgo_d[1]),
                cpw.t().dot(&bgo_d[2]),
            ],
            &rd,
            out,
        );
    }
}

/// Jastrow-side periodic kinetic force `(d(Sum_e w_e*kc^J_e)/dR (n_atom,3), d/dr (n_e,3))` from a
/// precomputed FULL drift cotangent `cotm = gJ + gD` (real; the complex-twist path passes
/// `gJ + Re(gD)`). Covers J1/J2 (Williamson) + J3 (periodic sphe pgto) + DTN (u/chi/f, min-image),
/// per-electron S'-weighted (`w=None` => full). Factored out of `pbc_kinetic_withj_grads` so the
/// real (real gD) and complex-twist (Re complex gD) kinetic forces share ONE Jastrow reduction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pbc_jastrow_kinetic_grads(
    jas: &JastrowData,
    j3: Option<&J3Pbc>,
    cotm: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    n_atom: usize,
    w: Option<&[f64]>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let wt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let cot = |e: usize, c: usize| -> f64 { cotm[[e, c]] };
    let mut d_r = Array2::<f64>::zeros((n_atom, 3));
    let mut d_re = Array2::<f64>::zeros((n_e, 3));

    // ---- J1 (e-N): dt[e] = -0.5 w_e DLAP1 - w_e cot_e*Hess1 ; d_r[ia] = -dt[e] ----
    // Metric = the J1 data's OWN convention (Williamson iff it carries a cell): jqmc keys
    // each Jastrow term's metric on ITS OWN structure_data.pbc_flag, never the geminal's.
    if let Some(j1) = &jas.one_body {
        let cm_j1 = j1
            .structure_data
            .cell
            .as_ref()
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let r_atoms = j1.structure_data.positions.view();
        let p = j1.jastrow_1b_param;
        let is_exp = j1.jastrow_1b_type == "exp";
        for e in 0..n_e {
            let re = elec(e);
            let ce = [cot(e, 0), cot(e, 1), cot(e, 2)];
            for ia in 0..n_atom {
                let z = j1.structure_data.atomic_numbers[ia] - j1.core_electrons[ia];
                let cc = (2.0 * z).powf(0.25);
                let pref = if is_exp {
                    -(2.0 * z).powf(0.75)
                } else {
                    -2.0 * z
                };
                let disp = [
                    re[0] - r_atoms[[ia, 0]],
                    re[1] - r_atoms[[ia, 1]],
                    re[2] - r_atoms[[ia, 2]],
                ];
                let (d, gd, hd, lap_d, glap_d) =
                    crate::forces_pbc::jastrow_dist_derivs(disp, cm_j1.as_ref());
                let (f1, f2, f3) = j_radial(is_exp, p * cc, d, pref * cc / 2.0, pref, cc);
                let (dlap, hess_c) = j_lap_hess_contract(f1, f2, f3, &gd, &hd, lap_d, &glap_d, &ce);
                for dd in 0..3 {
                    let dte = -0.5 * wt(e) * dlap[dd] - wt(e) * hess_c[dd];
                    d_re[[e, dd]] += dte;
                    d_r[[ia, dd]] -= dte;
                }
            }
        }
    }

    // ---- J2 (e-e): pair (e<k), dt[e] += val, dt[k] -= val ----
    // Metric = the J2 data's OWN convention (Williamson iff j2.cell is Some; a cell-less
    // J2 inside a periodic determinant is Euclidean -- matching j2_pair / jqmc's
    // compute_Jastrow_two_body, whose PBC branch keys on the J2's structure_data).
    if let Some(j2) = &jas.two_body {
        let cm_j2 = j2
            .cell
            .as_ref()
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let p = j2.jastrow_2b_param;
        let is_exp = j2.jastrow_2b_type == "exp";
        for e in 0..n_e {
            let re = elec(e);
            for k in (e + 1)..n_e {
                let rk = elec(k);
                let disp = [re[0] - rk[0], re[1] - rk[1], re[2] - rk[2]];
                let (d, gd, hd, lap_d, glap_d) =
                    crate::forces_pbc::jastrow_dist_derivs(disp, cm_j2.as_ref());
                let (h1, h2, h3) = j2_radial(is_exp, p, d);
                // DLAP2 = d(grad^2J2)/dr_e ; Hess2 = d(grad_e J2)/dr_e
                let q = gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2];
                let mut hd_gd = [0.0; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        hd_gd[i] += hd[i][j] * gd[j];
                    }
                }
                let mut dlap2 = [0.0; 3];
                for dd in 0..3 {
                    dlap2[dd] = h3 * q * gd[dd]
                        + h2 * 2.0 * hd_gd[dd]
                        + h2 * lap_d * gd[dd]
                        + h1 * glap_d[dd];
                }
                let ce = [cot(e, 0), cot(e, 1), cot(e, 2)];
                let ck = [cot(k, 0), cot(k, 1), cot(k, 2)];
                // Hess2[a][b] = h2 gd_a gd_b + h1 hd_ab ; (cot*Hess2)[d] = Sum_a cot_a Hess2[a][d]
                let hess2_dot = |cotv: &[f64; 3], dd: usize| -> f64 {
                    let mut s = 0.0;
                    for a in 0..3 {
                        s += cotv[a] * (h2 * gd[a] * gd[dd] + h1 * hd[a][dd]);
                    }
                    s
                };
                for dd in 0..3 {
                    let val = -0.5 * (wt(e) + wt(k)) * dlap2[dd] - wt(e) * hess2_dot(&ce, dd)
                        + wt(k) * hess2_dot(&ck, dd);
                    d_re[[e, dd]] += val;
                    d_re[[k, dd]] -= val;
                }
            }
        }
    }

    // ---- J3 (e-e-n, periodic sphe pgto basis) ----
    if let Some(c) = j3 {
        // bar_gj = -cot per spin for the d/dR side; j3_pbc_side_w applies the mask w internally.
        let mut bg_up = Array2::<f64>::zeros((n_up, 3));
        let mut bg_dn = Array2::<f64>::zeros((r_dn.nrows(), 3));
        for e in 0..n_e {
            for d in 0..3 {
                let v = cot(e, d);
                if e < n_up {
                    bg_up[[e, d]] = -v;
                } else {
                    bg_dn[[e - n_up, d]] = -v;
                }
            }
        }
        crate::forces_pbc::j3_pbc_side_w(
            c.cart_j3,
            &c.t_mol.to_owned(),
            &c.jm,
            r_up,
            r_dn,
            &bg_up,
            &bg_dn,
            cell,
            c.home,
            w,
            c.pw,
            c.gto_first,
            &mut d_r,
        );
        crate::forces_pbc::j3_pbc_egrad_side(
            c.cart_j3,
            &c.t_mol.to_owned(),
            &c.jm,
            r_up,
            r_dn,
            cotm,
            w,
            cell,
            c.pw,
            c.gto_first,
            &mut d_re,
        );
    }

    // ---- DTN (periodic: min-image displacement in the force kernels, cot = gradlnPsi) ----
    if let Some(d) = &jas.dtn {
        d_r += &crate::forces_dtn::grad_t_dtn_wrt_r_atom_w(d, cotm, r_up, r_dn, w);
        d_re += &crate::forces_dtn::grad_t_dtn_wrt_r_w(d, cotm, r_up, r_dn, w);
    }
    (d_r, d_re)
}

/// J1 radial derivatives f'(d), f''(d), f'''(d) for the one-body kernel.
/// `a = p*c`, prefactors `pref*c/2` etc. (exp) or the pade form.
#[inline]
fn j_radial(is_exp: bool, a: f64, d: f64, _f1pref: f64, pref: f64, c: f64) -> (f64, f64, f64) {
    if is_exp {
        let ee = (-a * d).exp();
        (
            pref * c / 2.0 * ee,
            -pref * (a * c) / 2.0 * ee,
            pref * (a * a * c) / 2.0 * ee,
        )
    } else {
        let u = 1.0 + a * d;
        (
            pref / (2.0 * u * u),
            -pref * a / u.powi(3),
            3.0 * pref * a * a / u.powi(4),
        )
    }
}

/// J2 radial derivatives h1(d),h2(d),h3(d) (two-body pade/exp).
#[inline]
fn j2_radial(is_exp: bool, p: f64, d: f64) -> (f64, f64, f64) {
    if is_exp {
        let ee = (-p * d).exp();
        (0.5 * ee, -0.5 * p * ee, 0.5 * p * p * ee)
    } else {
        let u = 1.0 + p * d;
        (1.0 / (2.0 * u * u), -p / u.powi(3), 3.0 * p * p / u.powi(4))
    }
}

/// Shared J1 lap-gradient + Hess*cot contraction (Williamson geometry).
/// Returns `(DLAP[d], (cot*Hess)[d])` with `Hess = f2 gd(x)gd + f1 Hd`,
/// `DLAP = f3 q gd + 2 f2 (Hd*gd) + f2 lap_d gd + f1 glap_d`.
#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "scalar inputs of an inlined hot contraction helper"
)]
fn j_lap_hess_contract(
    f1: f64,
    f2: f64,
    f3: f64,
    gd: &[f64; 3],
    hd: &[[f64; 3]; 3],
    lap_d: f64,
    glap_d: &[f64; 3],
    cot: &[f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let q = gd[0] * gd[0] + gd[1] * gd[1] + gd[2] * gd[2];
    let mut hd_gd = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            hd_gd[i] += hd[i][j] * gd[j];
        }
    }
    let mut dlap = [0.0; 3];
    let mut hess_c = [0.0; 3];
    for dd in 0..3 {
        dlap[dd] = f3 * q * gd[dd] + 2.0 * f2 * hd_gd[dd] + f2 * lap_d * gd[dd] + f1 * glap_d[dd];
        let mut s = 0.0;
        for a in 0..3 {
            s += cot[a] * (f2 * gd[a] * gd[dd] + f1 * hd[a][dd]);
        }
        hess_c[dd] = s;
    }
    (dlap, hess_c)
}

// ============================================================================
// GFMC (LRDMC) HF stress of the novar=0 continuous measurement energy.
// Propagation-only S-prime and lattice-clipping derivatives are absent.
// ============================================================================

/// `de_L^t/deps (3x3)` for the molecular AE jnone LRDMC diagonal local energy.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ae(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let j3 = match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => Some((o, m)),
        _ => None,
    };
    let j1 = jastrow_full.one_body.as_ref();
    let j2 = jastrow_full.two_body.as_ref();
    let dtn = jastrow_full.dtn.as_ref();
    let has_j = j1.is_some() || j2.is_some() || jastrow_full.dtn.is_some() || j3.is_some();
    let jas_opt: Option<&JastrowData> = if has_j { Some(jastrow_full) } else { None };

    // DTN kinetic strain is not part of the shared determinant/J1/J2/J3 stress
    // reverse. Build its full electron cotangent explicitly, as in the molecular
    // ECP+DTN stress path.
    let cot = dtn.map(|_| {
        let (gj_up, gj_dn) = jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn);
        let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
        let mut out = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                out[[e, c]] = if e < n_up {
                    gj_up[[e, c]] + gd_up[[e, c]]
                } else {
                    gj_dn[[e - n_up, c]] + gd_dn[[e - n_up, c]]
                };
            }
        }
        out
    });

    let _ = (alat, rt);

    // continuum VMC HF stress (d(T_cont + V_bare)/deps)
    let mut sigma = crate::stress::grad_el_wrt_strain_withj(
        gem, jas_opt, j1, j2, j3_orb, j3_jm, positions, z_eff, r_up, r_dn, nuc_up, nuc_dn, nuc_j3,
    );
    if let (Some(d), Some(cot)) = (dtn, cot.as_ref()) {
        let d_r = crate::forces_dtn::grad_t_dtn_wrt_r_atom(d, cot, r_up, r_dn);
        let d_re = crate::forces_dtn::grad_t_dtn_wrt_r(d, cot, r_up, r_dn);
        sigma += &crate::stress::virial(&d_re, &d_r, r_up, r_dn, positions);
    }
    sigma
}

/// Cached base-config DETERMINANT fields for the molecular ECP-nonlocal mesh
/// single-electron hops. Each mesh candidate moves one electron
/// `gi -> r'`, so the candidate geminal `G'` is a RANK-1 modification of the base
/// `G` (only that electron's orbital value/grad column changes). Rebuilding the
/// FULL system per candidate (fresh LU + `orb_value_grad_lap` for
/// all `N_e` electrons inside `lnpsi_pulay_r`/`lnpsi_drift`/`compute_geminal`, plus
/// a per-atom Pulay contraction) costs 96.9% of the HF force of a 20-water cluster. Here we cache the
/// base fields ONCE and, per candidate:
///   1. recompute only the moved electron's orbital column (one AO value+grad eval);
///   2. Sherman-Morrison the shared base inverse (rank-1) -> `G'^-1`, `det ratio`;
///   3. contract the (cached, single-column-patched) AO-grad fields against the
///      updated inverse for the determinant Pulay (`dln|detG|/dR`) + drift
///      (`dln|detG|/dr`).
///      This reproduces `grad_lndet_wrt_r_gi` + `det_grad_field_gi` at the moved config
///      but rooted at the shared base inverse (rank-1 SM) instead of a fresh per-candidate
///      LU -- algebraically identical, agreeing to O(cond*eps) ~1e-12 (the SM update at a
///      near-singular base uses the caller's AS-regularized base inverse, so it is the
///      STABLE choice, matching the reuse contract everywhere else). The Jastrow
///      Pulay/drift/value pieces stay full-recompute (inverse-free, cheap, bit-identical).
///      Molecular pure-gto scope (sphe arrives sphe->cart MO-wrapped: `c_up=Some(c2s)`,
///      `aos=cart` -- handled by the MO branch).
///      MO-project a raw-AO column stack `(num_ao, ncol)` by `C (num_mo, num_ao)` (or
///      pass through when `c` is `None`, i.e. a raw-AO orbital).
#[cfg(test)]
#[inline]
fn mo_project(c: Option<&Array2<f64>>, ao: &Array2<f64>) -> Array2<f64> {
    match c {
        None => ao.clone(),
        Some(c) => c.dot(ao),
    }
}

/// Value-level determinant base fields of the molecular ECP rank-1 path (the
/// moved electron's ratio and the Sherman-Morrison bars need only these).
pub struct MeshFwd {
    ou: Array2<f64>,
    od: Array2<f64>,
    w: Array2<f64>,
    ginv: Array2<f64>,
    q: Array2<f64>,
    wg: Array2<f64>,
    gq: Array2<f64>,
    lam_p: Option<Array2<f64>>,
    n_dn: usize,
}

impl From<crate::param_grad_el::GemFwd> for MeshFwd {
    /// The value-level part of a kinetic forward: `ou`/`od` are the standalone
    /// value tables when the forward reuses a running inverse (`g_inv = Some`),
    /// i.e. the same tables [`gem_forward_mesh_gi`] evaluates.
    fn from(f: crate::param_grad_el::GemFwd) -> Self {
        MeshFwd {
            ou: f.ou,
            od: f.od,
            w: f.w,
            ginv: f.ginv,
            q: f.q,
            wg: f.wg,
            gq: f.gq,
            lam_p: f.lam_p,
            n_dn: f.n_dn,
        }
    }
}

/// Determinant base fields for the molecular ECP rank-1 path, mirroring jqmc's
/// `one_move_mesh.det_base`.  With a supplied running inverse the base keeps the
/// standalone value-only tables that inverse is maintained from.  Without one,
/// G and its fresh inverse come from the fused value/grad/lap value table, the
/// same table the kinetic forward (`gem_forward_gi`) inverts: the base drift
/// contracts G^{-1} with the fp64 AO gradient tables (`MeshBase`), and in mixed
/// precision the standalone value kernel (fp32 arithmetic) would describe a
/// different geminal matrix (jqmc `determinant._orbital_value_grad_lap_tables`,
/// #681).  Both value tables are fp32-rounded in mixed; only the arithmetic that
/// produced them differs.  No MO gradient table is formed.
fn gem_forward_mesh_gi(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> MeshFwd {
    let fused_value = |orb: &OrbData, r: &ArrayView2<f64>| -> Array2<f64> {
        // Molecular pure-gto leaves only (asserted by `MeshDetFast::new_with_forward`).
        let av = crate::atomic_orbital::compute_aos_fused_value(&orb.aos, r);
        match &orb.mo_coefficients {
            None => av,
            Some(c) => crate::determinant::matmul(&c.view(), &av.view()),
        }
    };
    let (ou, od) = match g_inv {
        Some(_) => (gem.orb_up.compute(r_up), gem.orb_dn.compute(r_dn)),
        None => (
            fused_value(&gem.orb_up, r_up),
            fused_value(&gem.orb_dn, r_dn),
        ),
    };
    let n_up = ou.ncols();
    let n_dn = r_dn.nrows();
    let m_dn = gem.orb_num_dn;
    let w = gem.w_from_dn(&od.view(), n_up);
    let ginv = match g_inv {
        Some(gi) => gi.to_owned(),
        None => crate::determinant::inverse(&ou.t().dot(&w).view()),
    };
    let q = gem.up_t_times_paired(&ou.view());
    let wg = w.dot(&ginv);
    let gq = ginv.dot(&q);
    MeshFwd {
        ou,
        od,
        w,
        ginv,
        q,
        wg,
        gq,
        lam_p: gem
            .lambda
            .as_ref()
            .map(|lam| lam.slice(numpy::ndarray::s![.., 0..m_dn]).to_owned()),
        n_dn,
    }
}

/// Pull an M-level coefficient `bar (M, N)` back to AO level: `C^T bar (num_ao, N)`
/// (or pass through for a raw-AO orbital). Mirrors `forces::pull_to_ao`.
#[inline]
fn pull_ao(c: Option<&Array2<f64>>, bar: &Array2<f64>) -> Array2<f64> {
    match c {
        None => bar.clone(),
        Some(c) => c.t().dot(bar),
    }
}

/// Pulay orbital-side contraction `s[d,c] = -Sum_k (C^T bar)[c,k]*ga[d][c,k]`,
/// scattered into `out` by per-AO home atom. 1:1 `forces::orbital_side` (value-grad).
#[cfg(test)]
fn pulay_side_c(
    c: Option<&Array2<f64>>,
    bar: &Array2<f64>,
    ga: &[Array2<f64>; 3],
    nuc: &[usize],
    out: &mut Array2<f64>,
) {
    let cm = pull_ao(c, bar); // (num_ao, N)
    let num_ao = cm.nrows();
    let n = cm.ncols();
    for c in 0..num_ao {
        let ia = nuc[c];
        for d in 0..3 {
            let mut acc = 0.0;
            for k in 0..n {
                acc += cm[[c, k]] * ga[d][[c, k]];
            }
            out[[ia, d]] -= acc;
        }
    }
}

/// Numeric mesh-candidate state (no orbital references), so the rank-1 candidate
/// algebra can be unit-tested in isolation against the reference recompute. A mesh
/// candidate is a single-electron hop: only the moved electron's orbital column of
/// `G` changes, so `G^-1` updates by Sherman-Morrison and every determinant-derivative
/// contraction (drift `wg`/`gq`, Pulay `C^T bar`) is a BASE value (precomputed once)
/// plus a rank-1 correction -- no full-size GEMM per candidate.
struct MeshBase<'a> {
    c_up: Option<&'a Array2<f64>>, // MO coeffs (num_mo, num_ao) or None (raw-AO)
    c_dn: Option<&'a Array2<f64>>,
    nuc_up: &'a [usize], // per-AO home atom (num_ao_up)
    nuc_dn: &'a [usize],
    ou: Array2<f64>,            // (M_up, N_up) orb value
    od: Array2<f64>,            // (M_dn, N_dn)
    ga_up: [Array2<f64>; 3],    // (num_ao_up, N_up) raw AO grad (Pulay + drift)
    ga_dn: [Array2<f64>; 3],    // (num_ao_dn, N_dn)
    w: Array2<f64>,             // (M_up, N_up)  = [lam_p*od | lam_unp]
    ginv: Array2<f64>,          // (N_up, N_up)
    q: Array2<f64>,             // (N_up, M_dn)  = ou^T lam_p
    lam_p: Option<Array2<f64>>, // dense only; SD identity pairing is implicit
    n_up: usize,
    n_dn: usize,
    n_atom: usize,
    // --- base contractions (constant across the mesh candidates at this config) ---
    wg_base: Array2<f64>, // (M_up, N_up) = W ginv
    gq_base: Array2<f64>, // (N_up, M_dn) = ginv q
    base_du: Array2<f64>, // (N_up, 3) base determinant drift
    base_dd: Array2<f64>, // (N_dn, 3)
    // --- moved-electron tables (see `moved_tables`) ---
    cmt_up: Array2<f64>, // (N_up, num_ao_up) = (C_up^T wg_base)^T, contiguous rows
    cmt_dn: Array2<f64>, // (n_dn, num_ao_dn)
    pself_up: Array2<f64>, // (N_up, n_atom*3) per-electron base Pulay
    pself_dn: Array2<f64>, // (n_dn, n_atom*3)
}

/// Moved-electron tables of a [`MeshBase`]: the base bars as contiguous rows
/// `cm^T` and each electron's own base Pulay term
/// `pself[k, 3 I + d] = Sum_{c in I} cm[c, k] ga[d][c, k]`, so a group's moved
/// column enters [`MeshBase::accumulate`] without strided column reads.
fn moved_tables(
    cm: &Array2<f64>,
    ga: &[Array2<f64>; 3],
    nuc: &[usize],
    n_atom: usize,
) -> (Array2<f64>, Array2<f64>) {
    let (num_ao, n) = cm.dim();
    let mut pself = Array2::<f64>::zeros((n, 3 * n_atom));
    for c in 0..num_ao {
        let ia = nuc[c];
        for k in 0..n {
            let w = cm[[c, k]];
            for d in 0..3 {
                pself[[k, 3 * ia + d]] += w * ga[d][[c, k]];
            }
        }
    }
    (cm.t().as_standard_layout().to_owned(), pself)
}

/// Weighted mesh-candidate sums of one configuration (see [`MeshBase::accumulate`]):
/// the summed rank-1 bar changes and the per-group moved-column / self-row terms.
struct MeshDetAcc {
    /// `dbar` summed per moved electron (column): `(M_up, N_up)` / `(M_dn, N_dn)`.
    dbar_up: Array2<f64>,
    dbar_dn: Array2<f64>,
    /// Forward-AO-level `Sum_v beta_v ao_v` and `Sum_v beta_v` per moved electron
    /// (`MeshDetFast::accumulate`); their MO image enters `dbar` in `MeshDetFast::finish`.
    abar_up: Option<Array2<f64>>,
    abar_dn: Option<Array2<f64>>,
    b_up: Vec<f64>,
    b_dn: Vec<f64>,
    pulay: Array2<f64>, // (n_atom, 3)
}

impl<'a> MeshBase<'a> {
    /// Derivative-leaf AO count of one spin (the rows of `cmt_up` / `cmt_dn`).
    fn num_ao(&self, spin_up: bool) -> usize {
        if spin_up {
            self.cmt_up.ncols()
        } else {
            self.cmt_dn.ncols()
        }
    }

    #[inline]
    fn lp_rows(&self) -> usize {
        self.w.nrows()
    }

    #[inline]
    fn lp_cols(&self) -> usize {
        self.q.ncols()
    }

    #[cfg(test)]
    #[inline]
    fn lp(&self, a: usize, b: usize) -> f64 {
        self.lam_p
            .as_ref()
            .map_or(if a == b { 1.0 } else { 0.0 }, |l| l[[a, b]])
    }

    /// `pre = Some((cm_base_up, cm_base_dn, base_du, base_dd))`: the AO-level
    /// base bars and the base drift, when the caller already contracted them
    /// (`vmc_diag_grads_with_forward`); `None` forms them here.
    #[allow(clippy::too_many_arguments)]
    fn new(
        f: MeshFwd,
        ga_up: [Array2<f64>; 3],
        ga_dn: [Array2<f64>; 3],
        c_up: Option<&'a Array2<f64>>,
        c_dn: Option<&'a Array2<f64>>,
        nuc_up: &'a [usize],
        nuc_dn: &'a [usize],
        n_up: usize,
        n_atom: usize,
        pre: Option<BaseBarsDrift>,
    ) -> Self {
        let n_dn = f.n_dn;
        let wg_base = f.wg; // W ginv (precomputed by gem_forward_gi)
        let gq_base = f.gq; // ginv q
        let (cm_base_up, cm_base_dn, base_du, base_dd) = match pre {
            Some(p) => p,
            None => {
                let mdn_base = gq_base
                    .slice(numpy::ndarray::s![0..n_dn, ..])
                    .t()
                    .to_owned(); // (M_dn, n_dn)
                let cm_up = pull_ao(c_up, &wg_base);
                let cm_dn = pull_ao(c_dn, &mdn_base);
                // drift Sum_a gou WG = Sum_c (C^T WG) gradAO
                let (_, du) = crate::forces::val_contract(&cm_up, &ga_up);
                let (_, dd) = crate::forces::val_contract(&cm_dn, &ga_dn);
                (cm_up, cm_dn, du, dd)
            }
        };
        let (cmt_up, pself_up) = moved_tables(&cm_base_up, &ga_up, nuc_up, n_atom);
        let (cmt_dn, pself_dn) = moved_tables(&cm_base_dn, &ga_dn, nuc_dn, n_atom);
        MeshBase {
            cmt_up,
            cmt_dn,
            pself_up,
            pself_dn,
            c_up,
            c_dn,
            nuc_up,
            nuc_dn,
            ou: f.ou,
            od: f.od,
            ga_up,
            ga_dn,
            w: f.w,
            ginv: f.ginv,
            q: f.q,
            lam_p: f.lam_p,
            n_up,
            n_dn,
            n_atom,
            wg_base,
            gq_base,
            base_du,
            base_dd,
        }
    }

    /// Base determinant drift `(dln|detG|/dr_up (N_up,3), dln|detG|/dr_dn (N_dn,3))`.
    fn base_drift(&self) -> (Array2<f64>, Array2<f64>) {
        (self.base_du.clone(), self.base_dd.clone())
    }

    /// Left vector `u` of one mesh hop of electron (`spin_up`, `li`): the candidate
    /// determinant ratio is `r = 1 + u.(mo_new - mo_old)` and the moved electron's
    /// candidate drift is `u.mo_grad_new / r`. Up hop: `u = (W G^-1)[:,li]`
    /// (`M_up`); down hop: `u = (G^-1 q)[li,:]` (`M_dn`).
    fn moved_coef(&self, spin_up: bool, li: usize) -> Vec<f64> {
        if spin_up {
            self.wg_base.column(li).to_vec()
        } else {
            self.gq_base.row(li).to_vec()
        }
    }

    /// The moved electron's base MO column (`ou[:,li]` or `od[:,li]`).
    fn moved_old(&self, spin_up: bool, li: usize) -> Vec<f64> {
        if spin_up {
            self.ou.column(li).to_vec()
        } else {
            self.od.column(li).to_vec()
        }
    }

    fn new_acc(&self) -> MeshDetAcc {
        MeshDetAcc {
            dbar_up: Array2::zeros(self.ou.raw_dim()),
            dbar_dn: Array2::zeros(self.od.raw_dim()),
            abar_up: None,
            abar_dn: None,
            b_up: vec![0.0; self.n_up],
            b_dn: vec![0.0; self.n_dn],
            pulay: Array2::zeros((self.n_atom, 3)),
        }
    }

    /// The pairing block `lam_p (M_up, M_dn)`; the SD identity pairing as a matrix.
    fn lp_mat(&self) -> Array2<f64> {
        match &self.lam_p {
            Some(l) => l.clone(),
            None => Array2::from_shape_fn((self.lp_rows(), self.lp_cols()), |(a, b)| {
                if a == b {
                    1.0
                } else {
                    0.0
                }
            }),
        }
    }

    /// Add one `(electron, atom)` candidate group, weighted, into `acc`.
    ///
    /// Each candidate `v` is a one-column hop of electron (`spin_up`, `li`), so its
    /// geminal inverse is the Sherman-Morrison update `G_v^-1 = G^-1 - G^-1 e u_v^T G^-1 / r_v`.
    /// The ratio-scaled update `r_v (G_v^-1 - G^-1)` is LINEAR in the column change
    /// `delta_v = mo_v - mo_old` (`r_v = 1 + u.delta_v` is linear too), and so are the
    /// determinant drift/Pulay contractions built from it. Therefore
    /// `Sum_v beta_v r_v (F_v - F_base)` for every drift/Pulay field `F` needs ONE
    /// aggregated column change `dbar = Sum_v beta_v delta_v` per group:
    ///
    ///  * up hop:   `r (WG' - WG) = -wl (x) z`,           `z = WG^T delta`, `wl = WG[:,li]`;
    ///    `r (GQ' - GQ) = G^-1[:,li] (x) (lam_p^T delta - q^T z)`.
    ///  * down hop: `r (WG' - WG) = (lam_p delta - W a) (x) G^-1[li,:]`, `a = GQ delta`;
    ///    `r (GQ' - GQ) = -a (x) GQ[li,:]`.
    ///
    /// These rank-1 bar changes are linear in `dbar`, so only `dbar` is summed per
    /// moved electron; [`Self::rank1_sums`] forms them for all groups at once and
    /// [`Self::finish`] contracts them against the AO gradient tables once. The moved
    /// electron's own AO-gradient column enters only the Pulay term, as
    /// `-cm_base[:,li] (ga1bar - b ga[:,li])` (`r cm'[:,li] = cm_base[:,li]`), with
    /// `mo_bar = Sum_v beta_v mo_v`, `b = Sum_v beta_v`, `ga1bar = Sum_v beta_v ga1_v`.
    /// The moved electron's drift row is excluded (the caller treats it per
    /// candidate), so its self term is cancelled here.
    #[cfg(test)]
    fn accumulate(
        &self,
        spin_up: bool,
        li: usize,
        mo_bar: &[f64],
        b: f64,
        ga1bar: &[Vec<f64>; 3],
        acc: &mut MeshDetAcc,
    ) {
        // The rank-1 bar changes are linear in `dbar`: sum `dbar` per moved electron
        // here and form them for all groups at once in `finish`.
        let (dbar, old) = if spin_up {
            (&mut acc.dbar_up, &self.ou)
        } else {
            (&mut acc.dbar_dn, &self.od)
        };
        for a in 0..dbar.nrows() {
            dbar[[a, li]] += mo_bar[a] - b * old[[a, li]];
        }
        if spin_up {
            self.moved_pulay(
                &self.cmt_up,
                &self.pself_up,
                self.nuc_up,
                li,
                b,
                ga1bar,
                acc,
            );
        } else {
            self.moved_pulay(
                &self.cmt_dn,
                &self.pself_dn,
                self.nuc_dn,
                li,
                b,
                ga1bar,
                acc,
            );
        }
    }

    /// The summed rank-1 bar changes `(dWG (M_up, N_up), dGQ (N_up, M_dn))` and the
    /// moved-row drift terms of every group, from the per-electron `dbar` sums:
    ///  * up:  `Z = WG^T Dbar_up`, `dWG -= WG Z^T`, `dGQ += G^-1 (lam_p^T Dbar_up - q^T Z)^T`,
    ///    `du[k] += Z[k,k] base_du[k]`;
    ///  * dn:  `A = GQ Dbar_dn`, `dWG += (lam_p Dbar_dn - W A) G^-1[:N_dn]`,
    ///    `dGQ -= A GQ[:N_dn]`, `dd[l] += A[l,l] base_dd[l]`.
    #[allow(clippy::type_complexity)]
    fn rank1_sums(
        &self,
        dbar_up: &Array2<f64>,
        dbar_dn: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let n_dn = self.n_dn;
        let lp = self.lp_mat();
        let z = self.wg_base.t().dot(dbar_up); // (N_up, N_up)
        let a = self.gq_base.dot(dbar_dn); // (N_up, N_dn)
        let mut dwg = -self.wg_base.dot(&z.t());
        dwg += &(lp.dot(dbar_dn) - self.w.dot(&a))
            .dot(&self.ginv.slice(numpy::ndarray::s![0..n_dn, ..]));
        let mut dgq = self
            .ginv
            .dot(&(lp.t().dot(dbar_up) - self.q.t().dot(&z)).t());
        dgq -= &a.dot(&self.gq_base.slice(numpy::ndarray::s![0..n_dn, ..]));
        let du = Array2::from_shape_fn((self.n_up, 3), |(k, d)| z[[k, k]] * self.base_du[[k, d]]);
        let dd = Array2::from_shape_fn((n_dn, 3), |(l, d)| a[[l, l]] * self.base_dd[[l, d]]);
        (dwg, dgq, du, dd)
    }

    /// The moved electron's own AO-gradient column in the Pulay:
    /// `-Sum_c cm[c,li] (ga1bar[d][c] - b ga[d][c,li])`, scattered by atom.
    #[allow(clippy::too_many_arguments)]
    fn moved_pulay(
        &self,
        cmt: &Array2<f64>,
        pself: &Array2<f64>,
        nuc: &[usize],
        li: usize,
        b: f64,
        ga1bar: &[Vec<f64>; 3],
        acc: &mut MeshDetAcc,
    ) {
        let row = cmt.row(li);
        for (d, ga1bar_d) in ga1bar[..3].iter().enumerate() {
            for (c, &w) in row.iter().enumerate() {
                acc.pulay[[nuc[c], d]] -= w * ga1bar_d[c];
            }
        }
        for ia in 0..self.n_atom {
            for d in 0..3 {
                acc.pulay[[ia, d]] += b * pself[[li, 3 * ia + d]];
            }
        }
    }

    /// Contract the accumulated bar changes once:
    /// `(Sum beta r dPulay (n_atom,3), Sum beta r d drift_up (N_up,3), ... drift_dn (N_dn,3))`.
    fn finish(&self, acc: MeshDetAcc) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let MeshDetAcc {
            dbar_up,
            dbar_dn,
            mut pulay,
            ..
        } = acc;
        let (dwg, dgq, du, dd) = self.rank1_sums(&dbar_up, &dbar_dn);
        // One AO-level contraction per spin serves the Pulay (`-Sum_k`, scattered
        // by atom) and the drift (`Sum_c`, = `Sum_a gou dWG`).
        let cm_u = pull_ao(self.c_up, &dwg);
        let (s_u, e_u) = crate::forces::val_contract(&cm_u, &self.ga_up);
        crate::forces::scatter_rows_neg(&s_u, self.nuc_up, &mut pulay);
        let mdn = dgq
            .slice(numpy::ndarray::s![0..self.n_dn, ..])
            .t()
            .to_owned();
        let cm_d = pull_ao(self.c_dn, &mdn);
        let (s_d, e_d) = crate::forces::val_contract(&cm_d, &self.ga_dn);
        crate::forces::scatter_rows_neg(&s_d, self.nuc_dn, &mut pulay);
        (pulay, e_u + du, e_d + dd)
    }
    /// Reference (full-recompute) candidate: identical math to the pre-rank-1 path.
    /// Kept for the parity unit test only (`candidate_cols` must match to ~1e-12).
    #[cfg(test)]
    fn candidate_cols_ref(
        &self,
        spin_up: bool,
        li: usize,
        mo_val: &Array2<f64>,
        _mo_g: &[Array2<f64>; 3],
        ga1: &[Array2<f64>; 3],
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = self.n_up;
        let mut ginv_new = self.ginv.clone();
        let ratio;
        if spin_up {
            let delta: Vec<f64> = (0..mo_val.nrows())
                .map(|a| mo_val[[a, 0]] - self.ou[[a, li]])
                .collect();
            let mut v = vec![0.0_f64; n];
            for (j, v_j) in v[..n].iter_mut().enumerate() {
                let mut s = 0.0;
                for (a, delta_a) in delta[..self.w.nrows()].iter().enumerate() {
                    s += self.w[[a, j]] * *delta_a;
                }
                *v_j = s;
            }
            let ainv_u: Vec<f64> = (0..n).map(|i| self.ginv[[i, li]]).collect();
            let mut r = 1.0_f64;
            for j in 0..n {
                r += v[j] * ainv_u[j];
            }
            ratio = r;
            let mut vt_ainv = vec![0.0_f64; n];
            for (i, v_i) in v[..n].iter().enumerate() {
                let vi = *v_i;
                if vi != 0.0 {
                    for (j, vt_ainv_j) in vt_ainv[..n].iter_mut().enumerate() {
                        *vt_ainv_j += vi * self.ginv[[i, j]];
                    }
                }
            }
            for i in 0..n {
                let ai = ainv_u[i] / ratio;
                if ai == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ginv_new[[i, j]] -= ai * vt_ainv[j];
                }
            }
        } else {
            let delta: Vec<f64> = (0..mo_val.nrows())
                .map(|b| mo_val[[b, 0]] - self.od[[b, li]])
                .collect();
            let mut u = vec![0.0_f64; n];
            for (i, u_i) in u[..n].iter_mut().enumerate() {
                let mut s = 0.0;
                for (b, delta_b) in delta[..self.q.ncols()].iter().enumerate() {
                    s += self.q[[i, b]] * *delta_b;
                }
                *u_i = s;
            }
            let mut ainv_u = vec![0.0_f64; n];
            for (i, ainv_u_i) in ainv_u[..n].iter_mut().enumerate() {
                let mut s = 0.0;
                for (j, u_j) in u[..n].iter().enumerate() {
                    s += self.ginv[[i, j]] * *u_j;
                }
                *ainv_u_i = s;
            }
            ratio = 1.0 + ainv_u[li];
            let vt_ainv: Vec<f64> = (0..n).map(|j| self.ginv[[li, j]]).collect();
            for i in 0..n {
                let ai = ainv_u[i] / ratio;
                if ai == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ginv_new[[i, j]] -= ai * vt_ainv[j];
                }
            }
        }
        // MO gradient tables of the candidate configuration: the MO projection of
        // its AO gradient tables (the moved electron's column replaced by `ga1`).
        let (wg, gq, ga_up_c, ga_dn_c);
        if spin_up {
            let mut gau2 = self.ga_up.clone();
            let mut ou2 = self.ou.clone();
            for a in 0..ou2.nrows() {
                ou2[[a, li]] = mo_val[[a, 0]];
            }
            for cc in 0..gau2[0].nrows() {
                for d in 0..3 {
                    gau2[d][[cc, li]] = ga1[d][[cc, 0]];
                }
            }
            let mut q2 = self.q.clone();
            for b in 0..self.lp_cols() {
                let mut s = 0.0;
                for a in 0..self.lp_rows() {
                    s += ou2[[a, li]] * self.lp(a, b);
                }
                q2[[li, b]] = s;
            }
            wg = self.w.dot(&ginv_new);
            gq = ginv_new.dot(&q2);
            ga_up_c = gau2;
            ga_dn_c = self.ga_dn.clone();
        } else {
            let mut gad2 = self.ga_dn.clone();
            let mut od2 = self.od.clone();
            for b in 0..od2.nrows() {
                od2[[b, li]] = mo_val[[b, 0]];
            }
            for cc in 0..gad2[0].nrows() {
                for d in 0..3 {
                    gad2[d][[cc, li]] = ga1[d][[cc, 0]];
                }
            }
            let mut w2 = self.w.clone();
            for a in 0..self.lp_rows() {
                let mut s = 0.0;
                for b in 0..self.lp_cols() {
                    s += self.lp(a, b) * od2[[b, li]];
                }
                w2[[a, li]] = s;
            }
            wg = w2.dot(&ginv_new);
            gq = ginv_new.dot(&self.q);
            ga_dn_c = gad2;
            ga_up_c = self.ga_up.clone();
        }
        let proj = |c: Option<&Array2<f64>>, ga: &[Array2<f64>; 3]| -> [Array2<f64>; 3] {
            [
                mo_project(c, &ga[0]),
                mo_project(c, &ga[1]),
                mo_project(c, &ga[2]),
            ]
        };
        let gou_c = proj(self.c_up, &ga_up_c);
        let godn_c = proj(self.c_dn, &ga_dn_c);
        let du = drift_up(&gou_c, &wg, self.n_up);
        let dd = drift_dn(&godn_c, &gq, self.n_dn);
        let mut pulay = Array2::<f64>::zeros((self.n_atom, 3));
        pulay_side_c(self.c_up, &wg, &ga_up_c, self.nuc_up, &mut pulay);
        let mdn = gq
            .slice(numpy::ndarray::s![0..self.n_dn, ..])
            .t()
            .to_owned();
        pulay_side_c(self.c_dn, &mdn, &ga_dn_c, self.nuc_dn, &mut pulay);
        (ratio, pulay, du, dd)
    }
}

/// Mesh-fast determinant cache = numeric `MeshBase` + the orbital references needed
/// to evaluate the moved electron's AO/MO column per candidate.
/// Leaves the FAST mesh value screen applies to in the force
/// ([`crate::ao_cart_deriv::ao_val_grad_mesh_screened`]).
fn mol_cart_screen_leaf(aos: &crate::atomic_orbital::AosGtoData) -> bool {
    aos.kind == crate::atomic_orbital::AoKind::Cart && aos.n_image == 1 && !aos.shells.is_empty()
}

/// Determinant base of [`MeshDetFast`] taken from the VMC force's kinetic forward
/// at the same configuration and inverse (see `vmc_diag_grads_with_forward`):
/// the value-level forward, the AO gradient tables of `gem`, the AO-level base
/// bars `C^T WG`, `C^T GQ[:N_dn]^T` and the base determinant drift.
pub struct MeshDetShared {
    pub fwd: MeshFwd,
    pub ga_up: [Array2<f64>; 3],
    pub ga_dn: [Array2<f64>; 3],
    pub cm_up: Array2<f64>,
    pub cm_dn: Array2<f64>,
    pub du: Array2<f64>,
    pub dd: Array2<f64>,
}

/// Jastrow base of [`MeshJasFast`] taken from the VMC force: the exact base
/// `jastrow_grads` drift and, for a raw-AO J3, its value / AO gradient tables.
pub struct MeshJasShared {
    pub gj_up: Array2<f64>,
    pub gj_dn: Array2<f64>,
    pub j3: Option<MeshJ3Tables>,
}

/// J3 AO tables at the base configuration: values (`OrbData::compute`) and
/// gradients, `(nao, N)` per spin.
pub struct MeshJ3Tables {
    pub a_up: Array2<f64>,
    pub a_dn: Array2<f64>,
    pub ga_up: [Array2<f64>; 3],
    pub ga_dn: [Array2<f64>; 3],
}

/// Base fields an ECP mesh force can take from the VMC force step.
pub struct MeshShared {
    pub det: MeshDetShared,
    pub jas: Option<MeshJasShared>,
}

struct MeshDetFast<'a> {
    value_orb_up: &'a OrbData,
    value_orb_dn: &'a OrbData,
    deriv_orb_up: &'a OrbData,
    deriv_orb_dn: &'a OrbData,
    base: MeshBase<'a>,
    /// `C_fwd^T u` for every moved electron (columns): the forward-AO image of
    /// [`MeshBase::moved_coef`], so a candidate's ratio and moved drift are AO-level
    /// dots (no per-candidate MO projection). `None` = the forward orbitals are the
    /// derivative orbitals, reuse the base bars `base.cmt_up/dn`.
    cm_fwd: Option<(Array2<f64>, Array2<f64>)>,
}

/// The mesh candidates of one `(electron, atom)` group, determinant part.
struct MeshDetGroup {
    spin_up: bool,
    li: usize,
    /// Per candidate determinant ratio `r_v`.
    ratio: Vec<f64>,
    /// Per candidate determinant drift of the moved electron.
    moved: Vec<[f64; 3]>,
    /// Screened: the AO indices of the rows of `ao` / `ga1` (every other AO is zero
    /// at every candidate); `None`: all rows.
    rows: Option<Vec<usize>>,
    ao: Array2<f64>,       // forward AO values (rows, nv)
    ga1: [Array2<f64>; 3], // derivative AO gradients (rows, nv)
}

impl<'a> MeshDetFast<'a> {
    /// Build a molecular mesh cache with determinant values/drifts evaluated in
    /// `forward_gem`'s stored AO representation, while Pulay AO derivatives are
    /// evaluated on the Cartesian-lowered `gem` leaves. `g_inv` is the shared base
    /// inverse (`Some` = reuse the sweep SM inverse; `None` = fresh LU here).
    /// Molecular pure-gto only (panics otherwise -- the mesh mol caller guarantees it).
    #[allow(clippy::too_many_arguments)]
    fn new_with_forward(
        gem: &'a GeminalData,
        forward_gem: &'a GeminalData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        nuc_up: &'a [usize],
        nuc_dn: &'a [usize],
        n_atom: usize,
        g_inv: Option<&ArrayView2<f64>>,
        shared: Option<MeshDetShared>,
    ) -> Self {
        assert!(
            gem.orb_up.pw.is_none()
                && gem.orb_up.bspline.is_none()
                && gem.orb_dn.pw.is_none()
                && gem.orb_dn.bspline.is_none()
                && forward_gem.orb_up.pw.is_none()
                && forward_gem.orb_up.bspline.is_none()
                && forward_gem.orb_dn.pw.is_none()
                && forward_gem.orb_dn.bspline.is_none(),
            "MeshDetFast: molecular pure-gto orbitals required"
        );
        let (f, ga_up, ga_dn, pre) = match shared {
            Some(sh) => (
                sh.fwd,
                sh.ga_up,
                sh.ga_dn,
                Some((sh.cm_up, sh.cm_dn, sh.du, sh.dd)),
            ),
            None => {
                let f = crate::timed!(
                    "db.forward",
                    gem_forward_mesh_gi(forward_gem, r_up, r_dn, g_inv)
                );
                // Raw AO gradients (Pulay scatter is at AO level, per-AO home atom). The
                // base config is un-wrapped for molecular (n_image==1), matching gem_forward.
                let (ga_up, ga_dn) = crate::timed!(
                    "db.ao_grad",
                    (
                        ao_grad(&gem.orb_up.aos, r_up),
                        ao_grad(&gem.orb_dn.aos, r_dn)
                    )
                );
                (f, ga_up, ga_dn, None)
            }
        };
        let _t_base = crate::prof::on().then(std::time::Instant::now);
        let base = MeshBase::new(
            f,
            ga_up,
            ga_dn,
            gem.orb_up.mo_coefficients.as_ref(),
            gem.orb_dn.mo_coefficients.as_ref(),
            nuc_up,
            nuc_dn,
            r_up.nrows(),
            n_atom,
            pre,
        );
        if let Some(t) = _t_base {
            crate::prof::add("db.mesh_base", t.elapsed().as_nanos());
        }
        let cm_fwd = (!std::ptr::eq(gem, forward_gem)).then(|| {
            let mdn = base
                .gq_base
                .slice(numpy::ndarray::s![0..base.n_dn, ..])
                .t()
                .to_owned();
            (
                pull_ao(forward_gem.orb_up.mo_coefficients.as_ref(), &base.wg_base),
                pull_ao(forward_gem.orb_dn.mo_coefficients.as_ref(), &mdn),
            )
        });
        MeshDetFast {
            value_orb_up: &forward_gem.orb_up,
            value_orb_dn: &forward_gem.orb_dn,
            deriv_orb_up: &gem.orb_up,
            deriv_orb_dn: &gem.orb_dn,
            base,
            cm_fwd,
        }
    }

    fn base_drift(&self) -> (Array2<f64>, Array2<f64>) {
        self.base.base_drift()
    }

    /// Evaluate the `nv` mesh candidates of one `(electron, atom)` group: ONE AO
    /// value/gradient sweep for all points, then per candidate the ratio
    /// `r_v = 1 + u.(mo_v - mo_old)` and the moved electron's drift `u.grad mo_v / r_v`
    /// as AO-level dots with `C_fwd^T u` (O(num_ao) each).
    ///
    /// `screen`: the caller's e_L uses the FAST mesh value screen; for identical
    /// forward/derivative molecular Cartesian leaves the mesh AO values and
    /// gradients are then taken on that keep set ([`FastScreen`]).
    fn prepare(
        &self,
        spin_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        screen: bool,
        prim_cut: bool,
    ) -> MeshDetGroup {
        let nv = new_positions.len();
        let (value_orb, deriv_orb) = if spin_up {
            (self.value_orb_up, self.deriv_orb_up)
        } else {
            (self.value_orb_dn, self.deriv_orb_dn)
        };
        let mut r = Array2::<f64>::zeros((nv, 3));
        for (v, p) in new_positions.iter().enumerate() {
            r[[v, 0]] = p[0];
            r[[v, 1]] = p[1];
            r[[v, 2]] = p[2];
        }
        // Values and electron gradients are determinant-forward fields and must
        // therefore retain the original spherical/cartesian AO representation.
        // Raw AO gradients used for the nuclear Pulay scatter remain Cartesian.
        let same = std::ptr::eq(value_orb, deriv_orb);
        // Screened: only the rows of shells kept at some candidate (the others are
        // exact zeros) are stored; `rows` maps them to AO indices.
        let (rows, ao, ga1) = crate::timed!(
            "dc.ao_eval",
            if screen && same && mol_cart_screen_leaf(&value_orb.aos) {
                let t = crate::ao_cart_deriv::ao_val_grad_mesh_screened(
                    &value_orb.aos,
                    &r.view(),
                    prim_cut,
                );
                (Some(t.rows), t.val, t.grad)
            } else {
                (
                    None,
                    value_orb.eval_aos(&r.view()), // (num_ao, nv)
                    crate::ao_cart_deriv::ao_grad_fused(&deriv_orb.aos, &r.view()),
                )
            }
        );
        // Forward == derivative (Cartesian) leaves: one gradient sweep serves both.
        let gao_own;
        let gao = if same {
            &ga1
        } else {
            let (gx, gy, gz) = crate::atomic_orbital::compute_aos_grad(&value_orb.aos, &r.view());
            gao_own = [gx, gy, gz];
            &gao_own
        };
        let p: Vec<f64> = match (&self.cm_fwd, spin_up) {
            (Some((up, _)), true) => up.column(li).to_vec(),
            (Some((_, dn)), false) => dn.column(li).to_vec(),
            (None, true) => self.base.cmt_up.row(li).to_vec(),
            (None, false) => self.base.cmt_dn.row(li).to_vec(),
        };
        let u = self.base.moved_coef(spin_up, li);
        let u_old: f64 = u
            .iter()
            .zip(self.base.moved_old(spin_up, li))
            .map(|(a, b)| a * b)
            .sum();
        let mut ratio = Vec::with_capacity(nv);
        let mut moved = Vec::with_capacity(nv);
        // AO-major sweep over the contiguous (num_ao, nv) tables; per candidate the
        // sums still run over c in ascending order.
        let mut sv = vec![0.0_f64; nv];
        let mut gv = vec![[0.0_f64; 3]; nv];
        {
            let (ao_c, g0, g1, g2) = (
                ao.as_standard_layout(),
                gao[0].as_standard_layout(),
                gao[1].as_standard_layout(),
                gao[2].as_standard_layout(),
            );
            let (ao_s, gs) = (
                ao_c.as_slice().unwrap(),
                [
                    g0.as_slice().unwrap(),
                    g1.as_slice().unwrap(),
                    g2.as_slice().unwrap(),
                ],
            );
            for k in 0..ao.nrows() {
                let (pc, o) = (p[rows.as_ref().map_or(k, |r| r[k])], k * nv);
                for v in 0..nv {
                    sv[v] += pc * ao_s[o + v];
                    for d in 0..3 {
                        gv[v][d] += pc * gs[d][o + v];
                    }
                }
            }
        }
        for v in 0..nv {
            let (s, g) = (sv[v], gv[v]);
            let rv = 1.0 + (s - u_old);
            ratio.push(rv);
            moved.push([g[0] / rv, g[1] / rv, g[2] / rv]);
        }
        MeshDetGroup {
            spin_up,
            li,
            ratio,
            moved,
            rows,
            ao,
            ga1,
        }
    }

    fn new_acc(&self) -> MeshDetAcc {
        self.base.new_acc()
    }

    /// Add `Sum_v beta_v r_v (F_v - F_base)` of one group to `acc`
    /// (see [`MeshBase::accumulate`]).
    /// The forward AO values enter `dbar` only through their MO image, which is
    /// linear: sum `Sum_v beta_v ao_v` per moved electron at AO level and project
    /// all electrons at once in [`Self::finish`].
    fn accumulate(&self, g: &MeshDetGroup, beta: &[f64], acc: &mut MeshDetAcc) {
        let beta_v = ArrayView1::from(beta);
        let b: f64 = beta.iter().sum();
        // Screened groups store only their live rows: scatter them (the forward
        // and derivative leaves coincide there).
        let full = |x: Array1<f64>| -> Vec<f64> {
            match &g.rows {
                None => x.to_vec(),
                Some(rows) => {
                    let mut f = vec![0.0_f64; self.base.num_ao(g.spin_up)];
                    for (k, &c) in rows.iter().enumerate() {
                        f[c] = x[k];
                    }
                    f
                }
            }
        };
        let abar = Array1::from(full(g.ao.dot(&beta_v)));
        let (slot, bs, n) = if g.spin_up {
            (&mut acc.abar_up, &mut acc.b_up, self.base.n_up)
        } else {
            (&mut acc.abar_dn, &mut acc.b_dn, self.base.n_dn)
        };
        let tab = slot.get_or_insert_with(|| Array2::zeros((abar.len(), n)));
        tab.column_mut(g.li).scaled_add(1.0, &abar);
        bs[g.li] += b;
        let ga1bar: [Vec<f64>; 3] = std::array::from_fn(|d| full(g.ga1[d].dot(&beta_v)));
        if g.spin_up {
            self.base.moved_pulay(
                &self.base.cmt_up,
                &self.base.pself_up,
                self.base.nuc_up,
                g.li,
                b,
                &ga1bar,
                acc,
            );
        } else {
            self.base.moved_pulay(
                &self.base.cmt_dn,
                &self.base.pself_dn,
                self.base.nuc_dn,
                g.li,
                b,
                &ga1bar,
                acc,
            );
        }
    }

    fn finish(&self, mut acc: MeshDetAcc) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // dbar[:, k] = C_fwd (Sum beta ao)[:, k] - (Sum beta)_k mo_old[:, k]
        let fold = |abar: Option<Array2<f64>>,
                    bs: &[f64],
                    orb: &OrbData,
                    old: &Array2<f64>,
                    dbar: &mut Array2<f64>| {
            if let Some(ab) = abar {
                let mo = match orb.mo_coefficients.as_ref() {
                    None => ab,
                    Some(c) => c.dot(&ab),
                };
                for a in 0..dbar.nrows() {
                    for k in 0..dbar.ncols() {
                        dbar[[a, k]] += mo[[a, k]] - bs[k] * old[[a, k]];
                    }
                }
            }
        };
        fold(
            acc.abar_up.take(),
            &acc.b_up,
            self.value_orb_up,
            &self.base.ou,
            &mut acc.dbar_up,
        );
        fold(
            acc.abar_dn.take(),
            &acc.b_dn,
            self.value_orb_dn,
            &self.base.od,
            &mut acc.dbar_dn,
        );
        self.base.finish(acc)
    }
}

/// determinant drift up `gD_up[k,d] = Sum_a gou[d][a,k]*wg[a,k]` (`(N_up,3)`).
#[inline]
fn drift_up<T: numpy::ndarray::LinalgScalar + std::ops::AddAssign>(
    gou: &[Array2<T>; 3],
    wg: &Array2<T>,
    n_up: usize,
) -> Array2<T> {
    let m_up = wg.nrows();
    let mut out = Array2::<T>::zeros((n_up, 3));
    for d in 0..3 {
        for k in 0..n_up {
            let mut s = T::zero();
            for a in 0..m_up {
                s += gou[d][[a, k]] * wg[[a, k]];
            }
            out[[k, d]] = s;
        }
    }
    out
}

/// determinant drift dn `gD_dn[l,d] = Sum_b godn[d][b,l]*gq[l,b]` (`(N_dn,3)`).
#[inline]
fn drift_dn<T: numpy::ndarray::LinalgScalar + std::ops::AddAssign>(
    godn: &[Array2<T>; 3],
    gq: &Array2<T>,
    n_dn: usize,
) -> Array2<T> {
    let m_dn = godn[0].nrows();
    let mut out = Array2::<T>::zeros((n_dn, 3));
    for d in 0..3 {
        for l in 0..n_dn {
            let mut s = T::zero();
            for b in 0..m_dn {
                s += godn[d][[b, l]] * gq[[l, b]];
            }
            out[[l, d]] = s;
        }
    }
    out
}

/// One-electron-move Jastrow mesh fields for the molecular ECP non-local FORCE
/// candidate loop (the Jastrow analogue of [`MeshDetFast`]). Per-candidate
/// FULL recomputes (`jastrow_value_for_ratio` + `grad_j1_wrt_r` + `grad_j3_wrt_r`
/// + `jastrow_grads`) cost O(nao^2*Ne) dense-J3 GEMMs + full AO-derivative sweeps per
///   mesh point, and the loop visits `n_e x NN x Nv` candidates per force eval -- the
///   water20 dense-J3 per-step force pathology (>600 s/step vs 3.6 s JAX). A mesh
///   candidate hops ONE electron, so every Jastrow field is its base value plus a
///   one-column update:
///
///  * value ratio: the one-electron updates (`jastrow_1b2b_ratio` plus
///    `j3_ratio` on a shared `J3RatioState` -- the same coupling-block symmetry
///    used by the e_L mesh).
///  * J3 Pulay + drift: under `M == M^T` the kernel coefficient collapses to
///    `coeff[:,k] = j1 + M.(s_tot - a_k)` (serves `grad_j3_wrt_r` AND the J3 drift
///    of `jastrow_grads`). Moving electron g leaves `coeff[:,g]` INVARIANT
///    (`s_tot - a_g` is unchanged) and shifts every other column by the single
///    matvec `w_delta = M.(a_new - a_old)` (`j3_matvec`, AVX-512). The Pulay /
///    drift deltas are then O(nao) / O(nao*Ne) contractions against the cached
///    base AO-gradient tables -- no per-candidate GEMM, no per-candidate full-AO
///    derivative sweep.
///  * J1/J2: diagonal/pair kernels -- O(Natom) / O(Ne) deltas.
///
/// The mesh fields are anchored to the SAME base kernels the full recompute used
/// (`mesh = base_kernel_value + exact table delta`), so the candidate difference
/// terms (`mesh - base`, the only thing the force consumes) match a full
/// recompute to fp-association noise -- the dispatch/parity suites compare at
/// `get_tolerance_min` zones. DTN IS covered (exact one-electron u/chi/f move delta,
/// `compute_dtn_ratio_mesh` + the move-delta drift/Pulay kernels below). NOT covered
/// (constructor returns `None` -> the caller keeps the full-recompute path): an
/// MO-wrapped J3 leaf, a periodic-wrap J3 leaf, or any periodic J1/J2 cell (this is
/// the MOLECULAR kernel).
struct MeshJasFast<'a, 'v> {
    jdata: &'a JastrowData,
    j3: Option<MeshJasFastJ3<'a, 'v>>,
    /// One atom-centric DTN neighbor state shared by every mesh candidate.
    /// `None` is the dense oracle/fallback route (`cap` absent or `cap >= Ne`).
    dtn_state: Option<crate::dtn::DtnRatioState>,
    n_up: usize,
    n_dn: usize,
    n_atom: usize,
    /// Full Jastrow drift at the base config (`jastrow_grads`, J1+J2+J3).
    base_gju: Array2<f64>,
    base_gjd: Array2<f64>,
    /// Under an fp32 `jastrow_grad_lap` zone: the base-configuration term cache of the fp32 DTN
    /// gradient, built on the first candidate, so each candidate's DTN drift delta re-sums only
    /// the terms its move touches (bit-identical to [`crate::dtn::dtn_drift_move_delta`]).
    dtn_terms: std::sync::OnceLock<crate::dtn::DtnGradTermsF32>,
}

/// The mesh candidates of one `(electron, atom)` group, Jastrow part.
struct MeshJasGroup {
    is_up: bool,
    li: usize,
    /// Per candidate Jastrow ratio.
    jr: Vec<f64>,
    /// Per candidate full Jastrow drift of the moved electron.
    moved: Vec<[f64; 3]>,
    /// Per candidate J1/J2/DTN deltas `(pulay (n_atom,3), drift_up, drift_dn)`.
    extra: Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>,
    j3: Option<MeshJasGroupJ3>,
}

struct MeshJasGroupJ3 {
    delta: Array2<f64>,       // (nao, nv) a_new - a_old[:,li]
    ga_new: [Array2<f64>; 3], // (nao, nv)
    coeff_li: Vec<f64>,       // (nao) invariant coefficient column of the moved electron
}

/// Weighted Jastrow mesh-candidate sums of one configuration.
struct MeshJasAcc {
    pulay: Array2<f64>,
    du: Array2<f64>,
    dd: Array2<f64>,
    /// J3: `dbar = Sum_v w_v delta_v` summed per moved electron (columns),
    /// `(nao, N_up)` / `(nao, N_dn)`; the coupling products are formed in `finish`.
    dbar_j3: Option<(Array2<f64>, Array2<f64>)>,
}

struct MeshJasFastJ3<'a, 'v> {
    orb: &'a OrbData,
    jm: ArrayView2<'v, f64>,
    state: crate::jastrow_factor::J3RatioState,
    /// Contiguous copy of the square coupling block (the batched `M.Delta` gemm operand).
    j3c: Array2<f64>,
    ga_up: [Array2<f64>; 3],
    ga_dn: [Array2<f64>; 3],
    /// `gsum[d][c] = Sum_k ga_up[d][c,k] + Sum_k ga_dn[d][c,k]` (per-AO total).
    gsum: [Vec<f64>; 3],
    /// `c0 = j1_col + M.s_up + M.s_dn` -- `coeff[:,k] = c0 - W[:,k]`.
    c0: Vec<f64>,
    nuc: &'a [usize],
}

impl<'a, 'v> MeshJasFast<'a, 'v> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        jdata: &'a JastrowData,
        j3_orb: Option<&'a OrbData>,
        j3_jm: Option<&ArrayView2<'v, f64>>,
        nuc_j3: Option<&'a [usize]>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        n_atom: usize,
        shared: Option<MeshJasShared>,
    ) -> Option<Self> {
        let (shared_gj, mut shared_j3) = match shared {
            Some(sh) => (Some((sh.gj_up, sh.gj_dn)), sh.j3),
            None => (None, None),
        };
        // Coverage gates (see the type doc): molecular J1/J2, DTN (u/chi/f move-delta),
        // and a raw-AO unwrapped J3 (the force callers pass the cart-lowered raw-AO leaf).
        if let Some(j1) = &jdata.one_body {
            if j1.structure_data.cell.is_some() {
                return None;
            }
        }
        if let Some(j2) = &jdata.two_body {
            if j2.cell.is_some() {
                return None;
            }
        }
        let j3 = match (j3_orb, j3_jm) {
            (Some(orb), Some(jm)) => {
                if orb.mo_coefficients.is_some()
                    || orb.wrap_electrons(r_up).is_some()
                    || jm.as_slice().is_none()
                {
                    return None;
                }
                let nuc = nuc_j3?;
                let tables = shared_j3.take();
                let (a_up, a_dn, ga_in) = match tables {
                    Some(t) => (t.a_up, t.a_dn, Some((t.ga_up, t.ga_dn))),
                    None => (
                        crate::timed!("jb.ao_val", orb.compute(r_up)), // raw-AO values (nao, N_up)
                        crate::timed!("jb.ao_val", orb.compute(r_dn)),
                        None,
                    ),
                };
                let state = crate::timed!(
                    "jb.j3_state",
                    crate::jastrow_factor::init_j3_ratio_state_with_policy(
                        jm,
                        &a_up.view(),
                        &a_dn.view(),
                        jdata.precision_policy.is_f32("jastrow_ratio"),
                    )
                );
                let (ga_up, ga_dn) = match ga_in {
                    Some(g) => g,
                    None => (
                        crate::timed!("jb.ao_grad", ao_grad(&orb.aos, r_up)),
                        crate::timed!("jb.ao_grad", ao_grad(&orb.aos, r_dn)),
                    ),
                };
                let nao = a_up.nrows();
                let mut gsum: [Vec<f64>; 3] = [vec![0.0; nao], vec![0.0; nao], vec![0.0; nao]];
                for d in 0..3 {
                    for c in 0..nao {
                        let mut s = 0.0;
                        for k in 0..ga_up[d].ncols() {
                            s += ga_up[d][[c, k]];
                        }
                        for k in 0..ga_dn[d].ncols() {
                            s += ga_dn[d][[c, k]];
                        }
                        gsum[d][c] = s;
                    }
                }
                let n_orb = jm.nrows();
                let j1col: Vec<f64> = (0..n_orb).map(|o| jm[[o, n_orb]]).collect();
                let c0: Vec<f64> = (0..n_orb)
                    .map(|o| j1col[o] + state.cross_up[o] + state.cross_dn[o])
                    .collect();
                let j3c = jm.slice(numpy::ndarray::s![.., 0..n_orb]).to_owned();
                Some(MeshJasFastJ3 {
                    orb,
                    jm: *jm,
                    state,
                    j3c,
                    ga_up,
                    ga_dn,
                    gsum,
                    c0,
                    nuc,
                })
            }
            _ => None,
        };
        let dtn_state = jdata.dtn.as_ref().and_then(|d| {
            d.precision_policy
                .is_f32("jastrow_ratio")
                .then(|| crate::dtn::init_dtn_ratio_state(d, r_up, r_dn))
        });
        // Base drift anchor (moved electron's candidate drift = base + exact delta).
        let has_j = jdata.one_body.is_some()
            || jdata.two_body.is_some()
            || jdata.dtn.is_some()
            || j3_orb.is_some();
        let jas_opt: Option<&JastrowData> = if has_j { Some(jdata) } else { None };
        // Every channel including DTN comes from the one generic entry point --
        // the base anchor has no reason to single a channel out.
        let (base_gju, base_gjd) = match shared_gj {
            Some(g) => g,
            None => crate::timed!(
                "jb.jas_grads",
                jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn)
            ),
        };
        Some(MeshJasFast {
            jdata,
            j3,
            dtn_state,
            n_up: r_up.nrows(),
            n_dn: r_dn.nrows(),
            n_atom,
            base_gju,
            base_gjd,
            dtn_terms: std::sync::OnceLock::new(),
        })
    }

    /// Evaluate the `nv` mesh candidates of one `(electron, atom)` group: electron
    /// (`is_up`, local `li`) hops `r_old -> new_positions[v]`. Per candidate this
    /// gives the Jastrow ratio, the moved electron's full Jastrow drift, and the
    /// J1/J2/DTN Pulay/drift deltas (O(Natom)/O(Ne)). The J3 Pulay/drift deltas are
    /// linear in the AO column change and are summed per group in [`Self::accumulate`].
    #[allow(
        clippy::too_many_arguments,
        reason = "move, configuration and screen inputs"
    )]
    fn prepare(
        &self,
        is_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        screen: bool,
        prim_cut: bool,
    ) -> MeshJasGroup {
        let nv = new_positions.len();
        let mut jr = Vec::with_capacity(nv);
        let mut extra = Vec::with_capacity(nv);
        let r_old = if is_up {
            [r_up[[li, 0]], r_up[[li, 1]], r_up[[li, 2]]]
        } else {
            [r_dn[[li, 0]], r_dn[[li, 1]], r_dn[[li, 2]]]
        };
        let old = self.j12_terms(r_old, r_up, r_dn);
        for &new_pos in new_positions {
            let (r, p, gu, gd) = self.candidate_j12(is_up, li, new_pos, r_up, r_dn, &old);
            jr.push(r);
            extra.push((p, gu, gd));
        }
        let mut own = vec![[0.0_f64; 3]; nv];

        // ---- J3: ratio exp(delta.coeff_li) and the moved electron's own drift ----
        let j3 = self.j3.as_ref().map(|j3| {
            let n_orb = j3.jm.nrows();
            let mut rmat = Array2::<f64>::zeros((nv, 3));
            for (v, p) in new_positions.iter().enumerate() {
                rmat[[v, 0]] = p[0];
                rmat[[v, 1]] = p[1];
                rmat[[v, 2]] = p[2];
            }
            // `screen`: the caller's e_L contracts the J3 mesh AOs with the FAST value
            // screen (`ecp_j3_wdot_all`, log-ratio `delta.coeff_li` in fp64); take the
            // values, gradients and ratio on that same screened model.
            let screened = screen && mol_cart_screen_leaf(&j3.orb.aos);
            let (a_new, ga_new) = if screened {
                // `delta = a_new - a_old` is dense, so are its tables.
                crate::ao_cart_deriv::ao_val_grad_mesh_screened(&j3.orb.aos, &rmat.view(), prim_cut)
                    .into_dense(n_orb)
            } else {
                (
                    j3.orb.compute(&rmat.view()), // (nao, nv)
                    crate::ao_cart_deriv::ao_grad_fused(&j3.orb.aos, &rmat.view()), // [3](nao, nv)
                )
            };
            let (aos_old, w_same, ga_same) = if is_up {
                (&j3.state.aos_up, &j3.state.w_up, &j3.ga_up)
            } else {
                (&j3.state.aos_dn, &j3.state.w_dn, &j3.ga_dn)
            };
            // Moving electron li leaves its own coefficient column invariant:
            // coeff_li = j1 + M.(s_tot - a_li), and the log J3 ratio is delta.coeff_li.
            let coeff_li: Vec<f64> = (0..n_orb).map(|c| j3.c0[c] - w_same[[c, li]]).collect();
            let mut delta = a_new.clone();
            for c in 0..n_orb {
                let a_old = aos_old[[c, li]];
                for v in 0..nv {
                    delta[[c, v]] -= a_old;
                }
            }
            // AO-major sweep: log-ratio `delta.coeff_li` and the moved electron's own
            // drift `coeff_li.(ga_new - ga_old)`, per candidate summed over c ascending.
            let mut tv = vec![0.0_f64; nv];
            {
                let (dl, g0, g1, g2) = (
                    delta.as_standard_layout(),
                    ga_new[0].as_standard_layout(),
                    ga_new[1].as_standard_layout(),
                    ga_new[2].as_standard_layout(),
                );
                let (dl, gn) = (
                    dl.as_slice().unwrap(),
                    [
                        g0.as_slice().unwrap(),
                        g1.as_slice().unwrap(),
                        g2.as_slice().unwrap(),
                    ],
                );
                for c in 0..n_orb {
                    let (k, o) = (coeff_li[c], c * nv);
                    let go = [
                        ga_same[0][[c, li]],
                        ga_same[1][[c, li]],
                        ga_same[2][[c, li]],
                    ];
                    for v in 0..nv {
                        tv[v] += k * dl[o + v];
                        for d in 0..3 {
                            own[v][d] += k * (gn[d][o + v] - go[d]);
                        }
                    }
                }
            }
            for v in 0..nv {
                let ratio = if j3.state.f32_arithmetic && !screened {
                    let aos_new_col: Vec<f64> = (0..n_orb).map(|c| a_new[[c, v]]).collect();
                    crate::jastrow_factor::j3_ratio(&j3.jm, &j3.state, is_up, li, &aos_new_col)
                } else {
                    tv[v].exp()
                };
                jr[v] = crate::jastrow_factor::multiply_ratio_components(self.jdata, jr[v], ratio);
            }
            MeshJasGroupJ3 {
                delta,
                ga_new,
                coeff_li,
            }
        });

        // ---- DTN deltas (u/chi/f); anchored to the same base kernels the loop's
        // base pulay/drift use, so mesh = base + exact one-electron delta. ----
        if let Some(dd) = &self.jdata.dtn {
            let mut ratios = vec![0.0_f64; nv];
            let mesh = Array2::from_shape_vec(
                (nv, 3),
                new_positions
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect::<Vec<f64>>(),
            )
            .unwrap();
            match &self.dtn_state {
                Some(state) => crate::dtn::compute_dtn_ratio_mesh_cached(
                    dd,
                    state,
                    r_up,
                    r_dn,
                    is_up,
                    li,
                    &mesh.view(),
                    &mut ratios,
                ),
                None => crate::dtn::compute_dtn_ratio_mesh(
                    dd,
                    r_up,
                    r_dn,
                    is_up,
                    li,
                    &mesh.view(),
                    &mut ratios,
                ),
            }
            for (v, &new_pos) in new_positions.iter().enumerate() {
                let (pulay, gju, gjd) = &mut extra[v];
                jr[v] =
                    crate::jastrow_factor::multiply_ratio_components(self.jdata, jr[v], ratios[v]);
                *pulay += &crate::forces_dtn::grad_j_dtn_wrt_r_atom_move_delta(
                    dd, r_up, r_dn, is_up, li, new_pos,
                );
                let (du, dd2) = if dd.precision_policy.is_f32("jastrow_grad_lap") {
                    self.dtn_terms
                        .get_or_init(|| crate::dtn::DtnGradTermsF32::new(dd, r_up, r_dn))
                        .move_delta(dd, r_up, r_dn, is_up, li, new_pos)
                } else {
                    crate::dtn::dtn_drift_move_delta(dd, r_up, r_dn, is_up, li, new_pos)
                };
                *gju += &du;
                *gjd += &dd2;
            }
        }

        let base_row = if is_up {
            self.base_gju.row(li)
        } else {
            self.base_gjd.row(li)
        };
        let moved = (0..nv)
            .map(|v| {
                let (_, gju, gjd) = &extra[v];
                let dr = if is_up { gju.row(li) } else { gjd.row(li) };
                [
                    base_row[0] + dr[0] + own[v][0],
                    base_row[1] + dr[1] + own[v][1],
                    base_row[2] + dr[2] + own[v][2],
                ]
            })
            .collect();
        MeshJasGroup {
            is_up,
            li,
            jr,
            moved,
            extra,
            j3,
        }
    }

    fn new_acc(&self) -> MeshJasAcc {
        MeshJasAcc {
            pulay: Array2::zeros((self.n_atom, 3)),
            du: Array2::zeros((self.n_up, 3)),
            dd: Array2::zeros((self.n_dn, 3)),
            dbar_j3: self.j3.as_ref().map(|j3| {
                let nao = j3.jm.nrows();
                (
                    Array2::zeros((nao, self.n_up)),
                    Array2::zeros((nao, self.n_dn)),
                )
            }),
        }
    }

    /// Add `Sum_v w_v (F_v - F_base)` of one group to `acc` for the Jastrow Pulay
    /// and drift fields, the moved electron's drift row excluded. J3: the deltas are
    /// linear in `delta_v = a_new_v - a_old` (the shift of every other coefficient
    /// column is `wd_v = M.delta_v`), so only `Sum_v w_v delta_v` is kept, per moved
    /// electron; [`Self::finish`] applies the coupling block once per spin.
    fn accumulate(&self, g: &MeshJasGroup, w: &[f64], acc: &mut MeshJasAcc) {
        let li = g.li;
        for (v, (p, gu, gd)) in g.extra.iter().enumerate() {
            let wv = w[v];
            if wv == 0.0 {
                continue;
            }
            acc.pulay.scaled_add(wv, p);
            for n in 0..self.n_up {
                if !(g.is_up && n == li) {
                    for d in 0..3 {
                        acc.du[[n, d]] += wv * gu[[n, d]];
                    }
                }
            }
            for n in 0..self.n_dn {
                if !(!g.is_up && n == li) {
                    for d in 0..3 {
                        acc.dd[[n, d]] += wv * gd[[n, d]];
                    }
                }
            }
        }
        if let (Some(j3), Some(gj3), Some((db_up, db_dn))) = (&self.j3, &g.j3, acc.dbar_j3.as_mut())
        {
            let w_v = ArrayView1::from(w);
            let b: f64 = w.iter().sum();
            let dbar = gj3.delta.dot(&w_v); // (nao)
            let db = if g.is_up { db_up } else { db_dn };
            db.column_mut(li).scaled_add(1.0, &dbar);
            // The moved electron's own coefficient column (invariant) times its AO
            // gradient change; the `M.dbar` terms are formed in `finish`.
            let ga_same = if g.is_up { &j3.ga_up } else { &j3.ga_dn };
            for (d, ga_same_d) in ga_same[..3].iter().enumerate() {
                let gnb = gj3.ga_new[d].dot(&w_v);
                for c in 0..gnb.len() {
                    acc.pulay[[j3.nuc[c], d]] -=
                        gj3.coeff_li[c] * (gnb[c] - b * ga_same_d[[c, li]]);
                }
            }
        }
    }

    /// `(Sum w dPulay (n_atom,3), Sum w d drift_up (N_up,3), Sum w d drift_dn (N_dn,3))`.
    fn finish(&self, acc: MeshJasAcc) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let MeshJasAcc {
            mut pulay,
            mut du,
            mut dd,
            dbar_j3,
        } = acc;
        if let (Some(j3), Some((db_up, db_dn))) = (&self.j3, dbar_j3) {
            // Per group `wd = M.dbar` shifts every other coefficient column; with
            // `W = M Dbar` (one product per spin):
            //   Pulay[nuc c] -= Sum_k W[c,k] (gsum[c] - ga[c,k]),
            //   moved drift[k] -= Sum_c W[c,k] ga[c,k],
            //   and every electron's drift shifts by Sum_c wd_sum[c] ga[c,n].
            let mut wd = vec![0.0_f64; j3.jm.nrows()];
            for (db, ga, dr) in [(&db_up, &j3.ga_up, &mut du), (&db_dn, &j3.ga_dn, &mut dd)] {
                let wm = j3.j3c.dot(db);
                let (sw, ew) = crate::forces::val_contract(&wm, ga);
                for c in 0..wm.nrows() {
                    let rs: f64 = wm.row(c).sum();
                    wd[c] += rs;
                    for d in 0..3 {
                        pulay[[j3.nuc[c], d]] -= rs * j3.gsum[d][c] - sw[[c, d]];
                    }
                }
                *dr -= &ew;
            }
            // Every electron's J3 drift shifts by Sum_c wd[c] ga[d][c,n].
            let wd = ArrayView1::from(&wd[..]);
            for d in 0..3 {
                let su = j3.ga_up[d].t().dot(&wd);
                let sd = j3.ga_dn[d].t().dot(&wd);
                for n in 0..self.n_up {
                    du[[n, d]] += su[n];
                }
                for n in 0..self.n_dn {
                    dd[[n, d]] += sd[n];
                }
            }
        }
        (pulay, du, dd)
    }
    /// J1 Pulay row / drift and J2 pair gradients at ONE electron position `p`:
    /// `(J1 Pulay (n_atom,3), J1 drift [3], J2 pair grads g(r_k - p) per up / dn
    /// partner)`. A mesh group's candidate deltas are `terms(new) - terms(old)`;
    /// the old-position terms are evaluated once per group.
    fn j12_terms(&self, p: [f64; 3], r_up: &ArrayView2<f64>, r_dn: &ArrayView2<f64>) -> J12Terms {
        let mut pulay = Array2::<f64>::zeros((self.n_atom, 3));
        let mut dg = [0.0_f64; 3];
        // ---- J1 (diagonal in the electron) ----
        if let Some(j1) = &self.jdata.one_body {
            let pos = &j1.structure_data.positions;
            let zn = &j1.structure_data.atomic_numbers;
            let ce = &j1.core_electrons;
            let a = j1.jastrow_1b_param;
            let is_exp = j1.jastrow_1b_type == "exp";
            for ia in 0..self.n_atom {
                let z = zn[ia] - ce[ia];
                let c = (2.0 * z).powf(0.25);
                let pref = if is_exp {
                    -(2.0 * z).powf(0.75)
                } else {
                    -2.0 * z
                };
                // Pulay row (diff = R - r, unit vector), 1:1 grad_j1_wrt_r.
                let dx = pos[[ia, 0]] - p[0];
                let dy = pos[[ia, 1]] - p[1];
                let dz = pos[[ia, 2]] - p[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let dterm_dr = if is_exp {
                    pref * (c / 2.0) * (-a * c * dist).exp()
                } else {
                    pref / (2.0 * (1.0 + a * c * dist).powi(2))
                };
                let inv = 1.0 / dist;
                pulay[[ia, 0]] += dterm_dr * dx * inv;
                pulay[[ia, 1]] += dterm_dr * dy * inv;
                pulay[[ia, 2]] += dterm_dr * dz * inv;
                // Drift row (grad wrt the electron = -unit direction), 1:1
                // j1_grad_lap_one_spin: fprime(r_safe) * (r-R)/r_safe with the
                // SAME EPS_SAFE_DISTANCE clamp (exact anchor match).
                let r_safe = dist.max(crate::jastrow_factor::EPS_SAFE_DISTANCE);
                let inv_safe = 1.0 / r_safe;
                let fprime = if is_exp {
                    -(2.0 * z).powf(0.75) * (c / 2.0) * (-a * c * r_safe).exp()
                } else {
                    -z / ((1.0 + a * c * r_safe) * (1.0 + a * c * r_safe))
                };
                dg[0] += fprime * (-dx) * inv_safe;
                dg[1] += fprime * (-dy) * inv_safe;
                dg[2] += fprime * (-dz) * inv_safe;
            }
        }
        // ---- J2 pair gradients g(r_k - p) for every partner ----
        let (mut gup, mut gdn) = (Vec::new(), Vec::new());
        if let Some(j2) = &self.jdata.two_body {
            let a = j2.jastrow_2b_param;
            let is_exp = j2.jastrow_2b_type == "exp";
            let pair = |partners: &ArrayView2<f64>, out: &mut Vec<[f64; 3]>| {
                out.extend((0..partners.nrows()).map(|k| {
                    crate::jastrow_factor::j2_pair(
                        is_exp,
                        a,
                        partners[[k, 0]] - p[0],
                        partners[[k, 1]] - p[1],
                        partners[[k, 2]] - p[2],
                        None,
                    )
                    .0
                }));
            };
            pair(r_up, &mut gup);
            pair(r_dn, &mut gdn);
        }
        (pulay, dg, gup, gdn)
    }

    /// The J1/J2 part of one mesh candidate: its ratio and the Pulay/drift DELTAS
    /// from the base (pair/diagonal kernels -- O(Natom)/O(Ne)). J3/DTN are layered on
    /// by [`Self::prepare`]. `old` = [`Self::j12_terms`] at the moved electron's base
    /// position.
    #[allow(clippy::type_complexity)]
    fn candidate_j12(
        &self,
        is_up: bool,
        li: usize,
        new_pos: [f64; 3],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        old: &(Array2<f64>, [f64; 3], Vec<[f64; 3]>, Vec<[f64; 3]>),
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        let jr =
            crate::jastrow_factor::jastrow_1b2b_ratio(self.jdata, r_up, r_dn, is_up, li, new_pos);
        let (np, ndg, ngu, ngd) = self.j12_terms(new_pos, r_up, r_dn);
        let (op, odg, ogu, ogd) = old;
        let pulay = np - op;
        let mut gju = Array2::<f64>::zeros((self.n_up, 3));
        let mut gjd = Array2::<f64>::zeros((self.n_dn, 3));
        let mut dself = [ndg[0] - odg[0], ndg[1] - odg[1], ndg[2] - odg[2]];
        if self.jdata.two_body.is_some() {
            // partner k's drift term from the moved electron: g(r_k - r_moved);
            // the moved electron's own term: -g (antisymmetric pair gradient).
            let mut upd = |gn: &[[f64; 3]],
                           go: &[[f64; 3]],
                           out: &mut Array2<f64>,
                           exclude: Option<usize>| {
                for k in 0..gn.len() {
                    if exclude == Some(k) {
                        continue;
                    }
                    for d in 0..3 {
                        let dlt = gn[k][d] - go[k][d];
                        out[[k, d]] += dlt;
                        dself[d] -= dlt;
                    }
                }
            };
            if is_up {
                upd(&ngu, ogu, &mut gju, Some(li));
                upd(&ngd, ogd, &mut gjd, None);
            } else {
                upd(&ngd, ogd, &mut gjd, Some(li));
                upd(&ngu, ogu, &mut gju, None);
            }
        }
        let own = if is_up { &mut gju } else { &mut gjd };
        for d in 0..3 {
            own[[li, d]] += dself[d];
        }
        (jr, pulay, gju, gjd)
    }
}

/// PERIODIC one-electron-move Jastrow mesh fields for the ECP non-local FORCE
/// candidate loop -- the periodic analogue of [`MeshJasFast`].
/// A per-candidate FULL recompute in [`ecp_nonlocal_force_pbc`]
/// (`jastrow_value_for_ratio` + `jastrow_grads` + `grad_j3_wrt_r_pbc` + `grad_j3_wrt_r_e_pbc`)
/// pays the O(nao^2*Ne) J3 bilinear + a full periodic pGTO derivative sweep over every
/// electron PER mesh candidate (~700 s/step on a periodic pGTO cell with J1+J2+J3).
/// A mesh candidate hops ONE
/// electron, so each field is base + one-column update, EXACTLY the [`MeshJasFast`]
/// pattern -- with two periodic-specific twists:
///
///  * TWO J3 AO conventions. The periodic J3 VALUE routes through
///    `orb.wrap_electrons` + `orb.compute` (the LEPSBAS per-image-cutoff path
///    `jastrow_value_for_ratio` uses), while the J3 Pulay + electron-drift route through the
///    cart-lowered full-image `t_mol.pgto_deriv` / `t_mol.pgto_grad` leaf
///    (`grad_j3_wrt_r_pbc` / `grad_j3_wrt_r_e_pbc`). The two disagree at the ~1e-9
///    single-vs-batch image level, so the ratio anchors on a VALUE [`J3RatioState`]
///    (wrapped `orb.compute`) and the Pulay/drift on a SEPARATE PGTO
///    [`J3RatioState`] + pGTO gradient tables. Both reuse the SAME symmetric coupling
///    block (`assert_j3_block_symmetric`): `coeff[:,k] = c0 - w[:,k]`,
///    `c0 = j1col + cross_up + cross_dn`, the identity `M.a.(K+K^T)=M.(s-a_k)` that
///    used by the molecular path -- byte-for-byte the same contraction math, only the
///    AO tables and the home-atom scatter (`home` vs `nuc`) differ.
///  * J1/J2 keep the OWN-cell Williamson smooth metric: the value ratio reuses the
///    periodic-aware [`jastrow_1b2b_ratio`], and the Pulay/drift deltas mirror
///    `grad_j1_wrt_r_pbc` / `j1_grad_lap_one_spin` / `j2_pair` with the SAME
///    `smooth_dist` + `EPS_SAFE_DISTANCE` clamp (mesh = base + exact table delta).
///
/// DTN IS covered (cell-aware min-image u/chi/f move delta -- see the constructor
/// doc). `None` (constructor) -> the caller keeps the full-recompute path: an
/// MO-wrapped / composite (PW) / B-spline J3 leaf (the delta cannot serve the pGTO
/// cart-lowering exactly). `JQMC_RS_ECPF_SLOW=1` forces the reference path at the caller.
struct MeshJasFastPbc<'a> {
    jdata: &'a JastrowData,
    /// Own-cell Williamson metric for the J1 nuclear terms (`None` = Euclidean).
    ci_j1: Option<crate::jastrow_factor::CellInfo>,
    /// Own-cell Williamson metric for the J2 pair terms (`None` = Euclidean).
    ci_j2: Option<crate::jastrow_factor::CellInfo>,
    j3: Option<MeshJasFastPbcJ3<'a>>,
    n_up: usize,
    n_dn: usize,
    n_atom: usize,
    base_jp: Array2<f64>,
    base_gju: Array2<f64>,
    base_gjd: Array2<f64>,
    /// Under an fp32 `jastrow_grad_lap` zone: the base-configuration term cache of the fp32 DTN
    /// gradient, built on the first candidate, so each candidate's DTN drift delta re-sums only
    /// the terms its move touches (bit-identical to [`crate::dtn::dtn_drift_move_delta`]).
    dtn_terms: std::sync::OnceLock<crate::dtn::DtnGradTermsF32>,
}

struct MeshJasFastPbcJ3<'a> {
    orb: &'a OrbData,
    cart_j3: &'a AosGtoData,
    t_mol: Array2<f64>,
    home: &'a [usize],
    cell: Array2<f64>,
    n_orb: usize,
    /// J3 coefficients retained for the zone-aware rank-1 ratio leaf.
    jm: Array2<f64>,
    /// VALUE-side ratio state (wrapped `orb.compute` AOs -- the `jastrow_value_for_ratio` path).
    val_state: crate::jastrow_factor::J3RatioState,
    /// PGTO-side state (cart-lowered full-image AOs -- the Pulay/drift coefficient path).
    pg_state: crate::jastrow_factor::J3RatioState,
    /// Contiguous copy of the symmetric coupling block (the batched `M.Delta` operand).
    j3c: Array2<f64>,
    /// The J1-like last column of `j_matrix`.
    j1col: Vec<f64>,
    /// PGTO AO gradients at the base config (`t_mol.pgto_grad`).
    ga_up: [Array2<f64>; 3],
    ga_dn: [Array2<f64>; 3],
    /// `gsum[d][c] = Sum_k ga_up[d][c,k] + Sum_k ga_dn[d][c,k]` (per-AO total).
    gsum: [Vec<f64>; 3],
    /// `c0 = j1col + pg.cross_up + pg.cross_dn` -- `coeff[:,k] = c0 - w_pg[:,k]`.
    c0: Vec<f64>,
    /// Shell geometry of `cart_j3` for the fused moved-electron mesh contraction
    /// ([`crate::mesh_vgs`]); `None` keeps the table sweep.
    geom: Option<crate::mesh_vgs::ShellGeom>,
}

/// Home atom of every Cartesian AO of a cart-lowered J3 leaf, from the orbital home atoms
/// `home` through the lowering `t_mol (n_orb, n_cao)`; `None` when a Cartesian AO feeds
/// orbitals of two atoms. AOs that feed no orbital take the atom of their shell's first
/// used AO (their weight is zero).
fn j3_cart_home(cart: &AosGtoData, t_mol: &Array2<f64>, home: &[usize]) -> Option<Vec<usize>> {
    let n_cao = cart.num_ao;
    let mut h: Vec<Option<usize>> = vec![None; n_cao];
    for k in 0..n_cao {
        for c in 0..t_mol.nrows() {
            if t_mol[[c, k]] != 0.0 {
                match h[k] {
                    None => h[k] = Some(home[c]),
                    Some(a) if a != home[c] => return None,
                    _ => {}
                }
            }
        }
    }
    for sh in &cart.shells {
        let fill = sh.aos.iter().find_map(|&kap| h[cart.kernel_to_home[kap]]);
        for &kap in &sh.aos {
            let k = cart.kernel_to_home[kap];
            if h[k].is_none() {
                h[k] = fill;
            }
        }
    }
    Some(h.into_iter().map(|a| a.unwrap_or(0)).collect())
}

impl<'a> MeshJasFastPbc<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        jdata: &'a JastrowData,
        j3ctx: Option<&'a J3Pbc<'_>>,
        cell: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        n_atom: usize,
    ) -> Option<Self> {
        // Coverage gates (see the type doc): DTN (u/chi/f move-delta, cell-aware min-image),
        // and a pure-pGTO periodic J3 leaf (the cart-lowering `grad_j3_wrt_r_pbc` covers;
        // MO/PW/B-spline keep the full path).
        let j3 = match j3ctx {
            Some(c) => {
                if c.orb.mo_coefficients.is_some() || c.orb.pw.is_some() || c.orb.bspline.is_some()
                {
                    return None;
                }
                let jm = c.jm.view();
                let n_orb = jm.nrows();
                let t_mol = c.t_mol.to_owned();
                let cell_o = cell.to_owned();
                // VALUE-side AOs: the wrapped `orb.compute` path (bit-identical to `jastrow_value_for_ratio`).
                let ruw = c
                    .orb
                    .wrap_electrons(r_up)
                    .unwrap_or_else(|| r_up.to_owned());
                let rdw = c
                    .orb
                    .wrap_electrons(r_dn)
                    .unwrap_or_else(|| r_dn.to_owned());
                let av_up = c.orb.compute(&ruw.view());
                let av_dn = c.orb.compute(&rdw.view());
                let val_state = crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    &jm,
                    &av_up.view(),
                    &av_dn.view(),
                    jdata.precision_policy.is_f32("jastrow_ratio"),
                );
                // PGTO-side AOs + gradients: the cart-lowered full-image leaf.
                // value + gradient of the cart-lowered leaf from one fused sweep per spin.
                let dvu =
                    crate::ao_cart_deriv::pgto_derivs(c.cart_j3, r_up, cell, true, false, false);
                let dvd =
                    crate::ao_cart_deriv::pgto_derivs(c.cart_j3, r_dn, cell, true, false, false);
                let ap_up = t_mol.dot(dvu.val.as_ref().unwrap());
                let ap_dn = t_mol.dot(dvd.val.as_ref().unwrap());
                let pg_state = crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    &jm,
                    &ap_up.view(),
                    &ap_dn.view(),
                    jdata.precision_policy.is_f32("jastrow_ratio"),
                );
                let (gpu, gpd) = (&dvu.grad, &dvd.grad);
                let ga_up = [t_mol.dot(&gpu[0]), t_mol.dot(&gpu[1]), t_mol.dot(&gpu[2])];
                let ga_dn = [t_mol.dot(&gpd[0]), t_mol.dot(&gpd[1]), t_mol.dot(&gpd[2])];
                let mut gsum: [Vec<f64>; 3] =
                    [vec![0.0; n_orb], vec![0.0; n_orb], vec![0.0; n_orb]];
                for d in 0..3 {
                    for cc in 0..n_orb {
                        let mut s = 0.0;
                        for k in 0..ga_up[d].ncols() {
                            s += ga_up[d][[cc, k]];
                        }
                        for k in 0..ga_dn[d].ncols() {
                            s += ga_dn[d][[cc, k]];
                        }
                        gsum[d][cc] = s;
                    }
                }
                let j1col: Vec<f64> = (0..n_orb).map(|o| jm[[o, n_orb]]).collect();
                let c0: Vec<f64> = (0..n_orb)
                    .map(|o| j1col[o] + pg_state.cross_up[o] + pg_state.cross_dn[o])
                    .collect();
                let j3c = jm.slice(numpy::ndarray::s![.., 0..n_orb]).to_owned();
                let geom = if std::env::var("JQMC_RS_MESH_TABLES").is_err() {
                    j3_cart_home(c.cart_j3, &t_mol, c.home)
                        .and_then(|h| crate::mesh_vgs::ShellGeom::new(c.cart_j3, &h, None))
                } else {
                    None
                };
                Some(MeshJasFastPbcJ3 {
                    orb: c.orb,
                    cart_j3: c.cart_j3,
                    t_mol,
                    home: c.home,
                    cell: cell_o,
                    n_orb,
                    jm: jm.to_owned(),
                    val_state,
                    pg_state,
                    j3c,
                    j1col,
                    ga_up,
                    ga_dn,
                    gsum,
                    c0,
                    geom,
                })
            }
            None => None,
        };
        // Base anchors: the SAME periodic kernels the full-recompute path anchors on.
        let ci_j1 = jdata
            .one_body
            .as_ref()
            .and_then(|j1| j1.structure_data.cell.as_ref())
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let ci_j2 = jdata
            .two_body
            .as_ref()
            .and_then(|j2| j2.cell.as_ref())
            .map(|c| crate::jastrow_factor::CellInfo::new(&c.view()));
        let mut base_jp = Array2::<f64>::zeros((n_atom, 3));
        if let Some(j1) = &jdata.one_body {
            base_jp += &grad_j1_wrt_r_pbc(j1, r_up, r_dn);
        }
        if let Some(c) = j3ctx {
            base_jp += &crate::forces_pbc::grad_j3_wrt_r_pbc(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                r_up,
                r_dn,
                c.home,
                n_atom,
                cell,
                c.pw,
                c.gto_first,
            );
        }
        if let Some(d) = &jdata.dtn {
            base_jp += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, r_up, r_dn);
        }
        let has_j = jdata.one_body.is_some()
            || jdata.two_body.is_some()
            || jdata.dtn.is_some()
            || j3ctx.is_some();
        let jas_opt: Option<&JastrowData> = if has_j { Some(jdata) } else { None };
        let (mut base_gju, mut base_gjd) = jastrow_grads(jas_opt, None, None, r_up, r_dn);
        if let Some(c) = j3ctx {
            let (j3u, j3d) = crate::forces_pbc::grad_j3_wrt_r_e_pbc(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                r_up,
                r_dn,
                cell,
                c.pw,
                c.gto_first,
            );
            base_gju = base_gju + j3u;
            base_gjd = base_gjd + j3d;
        }
        Some(MeshJasFastPbc {
            jdata,
            ci_j1,
            ci_j2,
            j3,
            n_up: r_up.nrows(),
            n_dn: r_dn.nrows(),
            n_atom,
            base_jp,
            base_gju,
            base_gjd,
            dtn_terms: std::sync::OnceLock::new(),
        })
    }

    /// BATCHED mesh candidates of one `(electron, atom)` group -- 1:1 the semantics of
    /// [`MeshJasFast::candidates`] (returns `(jr, pulay (n_atom,3), gju (n_up,3),
    /// gjd (n_dn,3))` per candidate) but on the periodic value/pGTO split above.
    fn candidates(
        &self,
        is_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> Vec<JasCandidate> {
        let nv = new_positions.len();
        let mut out: Vec<JasCandidate> = new_positions
            .iter()
            .map(|&new_pos| self.candidate_j12(is_up, li, new_pos, r_up, r_dn))
            .collect();

        if let Some(j3) = &self.j3 {
            let n_orb = j3.n_orb;
            let cell = j3.cell.view();
            let mut rmat = Array2::<f64>::zeros((nv, 3));
            for (v, p) in new_positions.iter().enumerate() {
                rmat[[v, 0]] = p[0];
                rmat[[v, 1]] = p[1];
                rmat[[v, 2]] = p[2];
            }
            // ---- VALUE ratio (wrapped `orb.compute` AOs) ----
            let ruw = j3
                .orb
                .wrap_electrons(&rmat.view())
                .unwrap_or_else(|| rmat.clone());
            let av_new = j3.orb.compute(&ruw.view()); // (n_orb, nv)
            let av_old = if is_up {
                &j3.val_state.aos_up
            } else {
                &j3.val_state.aos_dn
            };
            let wv_same = if is_up {
                &j3.val_state.w_up
            } else {
                &j3.val_state.w_dn
            };
            let cross_v = if is_up {
                &j3.val_state.cross_dn
            } else {
                &j3.val_state.cross_up
            };
            let mut dv = av_new.clone();
            for c in 0..n_orb {
                let a_old = av_old[[c, li]];
                for v in 0..nv {
                    dv[[c, v]] -= a_old;
                }
            }
            let vmv = dv.t().dot(wv_same); // (nv, N_same)
                                           // ---- PGTO Pulay/drift (cart-lowered full-image AOs + gradients) ----
            let wp_same_li = if is_up {
                j3.pg_state.w_up.column(li).to_owned()
            } else {
                j3.pg_state.w_dn.column(li).to_owned()
            };
            // coeff_li = c0 - w_pg[:, li]: the moved electron's (invariant) AO coefficient.
            let coeff_li: Vec<f64> = (0..n_orb).map(|c| j3.c0[c] - wp_same_li[c]).collect();
            // The moved-electron AO gradients enter only through `coeff_li . ga_new` (total
            // and per home atom): with `geom`, ONE fused unscreened contraction gives those
            // sums and the folded value table; otherwise the exact table sweep.
            let (ap_new, fused_g, ga_new) = match &j3.geom {
                Some(geom) => {
                    let inv = inv3(&cell);
                    let rw = crate::mcmc::wrap_rows(&rmat.view(), &cell, &inv);
                    let pts: Vec<[f64; 3]> = (0..nv)
                        .map(|v| [rw[[v, 0]], rw[[v, 1]], rw[[v, 2]]])
                        .collect();
                    let wc = j3
                        .t_mol
                        .t()
                        .dot(&numpy::ndarray::ArrayView1::from(&coeff_li[..]));
                    let mut o = crate::mesh_vgs::MeshVgs::new(
                        nv,
                        Some(self.n_atom),
                        false,
                        Some(j3.cart_j3.num_ao),
                    );
                    crate::mesh_vgs::mesh_vgs(
                        j3.cart_j3,
                        &pts,
                        crate::mesh_vgs::Weights::Shared(wc.as_slice().unwrap()),
                        AoScreen::None,
                        Some(geom),
                        &mut o,
                    );
                    let vt = Array2::from_shape_vec(
                        (j3.cart_j3.num_ao, nv),
                        std::mem::take(&mut o.vtab),
                    )
                    .unwrap();
                    (j3.t_mol.dot(&vt), Some(o), None)
                }
                None => {
                    let dvn = crate::ao_cart_deriv::pgto_derivs(
                        j3.cart_j3,
                        &rmat.view(),
                        &cell,
                        true,
                        false,
                        false,
                    );
                    let ap_new = j3.t_mol.dot(dvn.val.as_ref().unwrap());
                    let gpn = &dvn.grad;
                    let ga_new = [
                        j3.t_mol.dot(&gpn[0]),
                        j3.t_mol.dot(&gpn[1]),
                        j3.t_mol.dot(&gpn[2]),
                    ]; // [3](n_orb, nv)
                    (ap_new, None, Some(ga_new))
                }
            };
            let ap_old = if is_up {
                &j3.pg_state.aos_up
            } else {
                &j3.pg_state.aos_dn
            };
            let wp_same = if is_up {
                &j3.pg_state.w_up
            } else {
                &j3.pg_state.w_dn
            };
            let ga_same = if is_up { &j3.ga_up } else { &j3.ga_dn };
            let mut dp = ap_new.clone();
            for c in 0..n_orb {
                let a_old = ap_old[[c, li]];
                for v in 0..nv {
                    dp[[c, v]] -= a_old;
                }
            }
            let wd = j3.j3c.dot(&dp); // (n_orb, nv)
            let su = [
                wd.t().dot(&j3.ga_up[0]),
                wd.t().dot(&j3.ga_up[1]),
                wd.t().dot(&j3.ga_up[2]),
            ];
            let sd = [
                wd.t().dot(&j3.ga_dn[0]),
                wd.t().dot(&j3.ga_dn[1]),
                wd.t().dot(&j3.ga_dn[2]),
            ];

            for v in 0..nv {
                let (jr, pulay, gju, gjd) = &mut out[v];
                // VALUE ratio: term1 = j1^T.dv ; term23 = Sum_{m != li} vmv ; term4 = cross_v^T.dv
                let ratio = if j3.val_state.f32_arithmetic {
                    let aos_new_col: Vec<f64> = (0..n_orb).map(|c| av_new[[c, v]]).collect();
                    crate::jastrow_factor::j3_ratio(
                        &j3.jm.view(),
                        &j3.val_state,
                        is_up,
                        li,
                        &aos_new_col,
                    )
                } else {
                    let mut term1 = 0.0;
                    let mut term4 = 0.0;
                    for c in 0..n_orb {
                        term1 += j3.j1col[c] * dv[[c, v]];
                        term4 += cross_v[c] * dv[[c, v]];
                    }
                    let mut term23 = 0.0;
                    for m in 0..vmv.ncols() {
                        if m != li {
                            term23 += vmv[[v, m]];
                        }
                    }
                    (term1 + term23 + term4).exp()
                };
                *jr = crate::jastrow_factor::multiply_ratio_components(self.jdata, *jr, ratio);
                // Pulay delta (PGTO coefficient), scatter -dP by the per-AO home atom.
                for d in 0..3 {
                    for c in 0..n_orb {
                        let ga_old_cli = ga_same[d][[c, li]];
                        let coeff_li = j3.c0[c] - wp_same[[c, li]];
                        let new_part = match &ga_new {
                            Some(g) => coeff_li * g[d][[c, v]],
                            None => 0.0,
                        };
                        let dpv = wd[[c, v]] * (j3.gsum[d][c] - ga_old_cli) + new_part
                            - coeff_li * ga_old_cli;
                        pulay[[j3.home[c], d]] -= dpv;
                    }
                    if let Some(o) = &fused_g {
                        for a in 0..self.n_atom {
                            pulay[[a, d]] -= o.atom[v * self.n_atom + a][d];
                        }
                    }
                }
                // Drift deltas (PGTO coefficient): batched wd-shift for every electron,
                // moved electron keeps its (invariant) coeff column but swaps its AO grad.
                for d in 0..3 {
                    for n in 0..self.n_up {
                        gju[[n, d]] += su[d][[v, n]];
                    }
                    for n in 0..self.n_dn {
                        gjd[[n, d]] += sd[d][[v, n]];
                    }
                    let mut s_own = 0.0;
                    let mut s_wd = 0.0;
                    for c in 0..n_orb {
                        let coeff_li = j3.c0[c] - wp_same[[c, li]];
                        let new_part = match &ga_new {
                            Some(g) => g[d][[c, v]],
                            None => 0.0,
                        };
                        s_own += coeff_li * (new_part - ga_same[d][[c, li]]);
                        s_wd += wd[[c, v]] * ga_same[d][[c, li]];
                    }
                    if let Some(o) = &fused_g {
                        s_own += o.grad[v][d];
                    }
                    if is_up {
                        gju[[li, d]] += s_own - s_wd;
                    } else {
                        gjd[[li, d]] += s_own - s_wd;
                    }
                }
            }
        }

        // ---- DTN deltas (u/chi/f; cell-aware min-image); mesh = base + one-electron delta. ----
        if let Some(dd) = &self.jdata.dtn {
            let mut ratios = vec![0.0_f64; nv];
            let mesh = Array2::from_shape_vec(
                (nv, 3),
                new_positions
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect::<Vec<f64>>(),
            )
            .unwrap();
            crate::dtn::compute_dtn_ratio_mesh(
                dd,
                r_up,
                r_dn,
                is_up,
                li,
                &mesh.view(),
                &mut ratios,
            );
            for (v, &new_pos) in new_positions.iter().enumerate() {
                let (jr, pulay, gju, gjd) = &mut out[v];
                *jr = crate::jastrow_factor::multiply_ratio_components(self.jdata, *jr, ratios[v]);
                *pulay += &crate::forces_dtn::grad_j_dtn_wrt_r_atom_move_delta(
                    dd, r_up, r_dn, is_up, li, new_pos,
                );
                let (du, dd2) = if dd.precision_policy.is_f32("jastrow_grad_lap") {
                    self.dtn_terms
                        .get_or_init(|| crate::dtn::DtnGradTermsF32::new(dd, r_up, r_dn))
                        .move_delta(dd, r_up, r_dn, is_up, li, new_pos)
                } else {
                    crate::dtn::dtn_drift_move_delta(dd, r_up, r_dn, is_up, li, new_pos)
                };
                *gju += &du;
                *gjd += &dd2;
            }
        }
        out
    }

    /// The periodic J1/J2 part of one mesh candidate (own-cell Williamson smooth
    /// metric). J3 is layered on by [`Self::candidates`].
    fn candidate_j12(
        &self,
        is_up: bool,
        li: usize,
        new_pos: [f64; 3],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        let jr =
            crate::jastrow_factor::jastrow_1b2b_ratio(self.jdata, r_up, r_dn, is_up, li, new_pos);
        let mut pulay = self.base_jp.clone();
        let mut gju = self.base_gju.clone();
        let mut gjd = self.base_gjd.clone();
        let r_old = if is_up {
            [r_up[[li, 0]], r_up[[li, 1]], r_up[[li, 2]]]
        } else {
            [r_dn[[li, 0]], r_dn[[li, 1]], r_dn[[li, 2]]]
        };

        // ---- J1 deltas (diagonal in the electron; own-cell smooth metric) ----
        if let Some(j1) = &self.jdata.one_body {
            let pos = &j1.structure_data.positions;
            let zn = &j1.structure_data.atomic_numbers;
            let ce = &j1.core_electrons;
            let a = j1.jastrow_1b_param;
            let is_exp = j1.jastrow_1b_type == "exp";
            let (mut dgx, mut dgy, mut dgz) = (0.0_f64, 0.0_f64, 0.0_f64);
            for ia in 0..self.n_atom {
                let z = zn[ia] - ce[ia];
                let c = (2.0 * z).powf(0.25);
                let pref = if is_exp {
                    -(2.0 * z).powf(0.75)
                } else {
                    -2.0 * z
                };
                let mut term = |p: [f64; 3], sign: f64| {
                    // disp = r - R ; smooth grad ge = dd/dr_e (Euclidean fallback if no cell).
                    let dx = p[0] - pos[[ia, 0]];
                    let dy = p[1] - pos[[ia, 1]];
                    let dz = p[2] - pos[[ia, 2]];
                    let (dist, ge) = match &self.ci_j1 {
                        Some(ci) => {
                            let (d, g, _, _) = crate::jastrow_factor::smooth_dist([dx, dy, dz], ci);
                            (d, g)
                        }
                        None => {
                            let d = (dx * dx + dy * dy + dz * dz)
                                .sqrt()
                                .max(crate::jastrow_factor::EPS_SAFE_DISTANCE);
                            (d, [dx / d, dy / d, dz / d])
                        }
                    };
                    // Radial kernel derivative (Pulay `grad_j1_wrt_r_pbc` == drift `fprime`).
                    let dterm_dr = if is_exp {
                        pref * (c / 2.0) * (-a * c * dist).exp()
                    } else {
                        pref / (2.0 * (1.0 + a * c * dist).powi(2))
                    };
                    // Pulay row: dJ1/dR = dterm_dr*(-ge) (1:1 grad_j1_wrt_r_pbc).
                    pulay[[ia, 0]] += sign * dterm_dr * (-ge[0]);
                    pulay[[ia, 1]] += sign * dterm_dr * (-ge[1]);
                    pulay[[ia, 2]] += sign * dterm_dr * (-ge[2]);
                    // Drift row: dJ1/dr = dterm_dr*ge (1:1 j1_grad_lap_one_spin).
                    dgx += sign * dterm_dr * ge[0];
                    dgy += sign * dterm_dr * ge[1];
                    dgz += sign * dterm_dr * ge[2];
                };
                term(new_pos, 1.0);
                term(r_old, -1.0);
            }
            if is_up {
                gju[[li, 0]] += dgx;
                gju[[li, 1]] += dgy;
                gju[[li, 2]] += dgz;
            } else {
                gjd[[li, 0]] += dgx;
                gjd[[li, 1]] += dgy;
                gjd[[li, 2]] += dgz;
            }
        }

        // ---- J2 drift deltas (pair kernel; only pairs touching the moved electron) ----
        if let Some(j2) = &self.jdata.two_body {
            let a = j2.jastrow_2b_param;
            let is_exp = j2.jastrow_2b_type == "exp";
            let ci = self.ci_j2.as_ref();
            let mut dself = [0.0_f64; 3];
            let mut upd = |partners: &ArrayView2<f64>,
                           out: &mut Array2<f64>,
                           exclude: Option<usize>| {
                for k in 0..partners.nrows() {
                    if exclude == Some(k) {
                        continue;
                    }
                    let (pk0, pk1, pk2) = (partners[[k, 0]], partners[[k, 1]], partners[[k, 2]]);
                    let (gn, _) = crate::jastrow_factor::j2_pair(
                        is_exp,
                        a,
                        pk0 - new_pos[0],
                        pk1 - new_pos[1],
                        pk2 - new_pos[2],
                        ci,
                    );
                    let (go, _) = crate::jastrow_factor::j2_pair(
                        is_exp,
                        a,
                        pk0 - r_old[0],
                        pk1 - r_old[1],
                        pk2 - r_old[2],
                        ci,
                    );
                    for d in 0..3 {
                        out[[k, d]] += gn[d] - go[d];
                        dself[d] -= gn[d] - go[d]; // antisymmetric pair gradient
                    }
                }
            };
            if is_up {
                upd(r_up, &mut gju, Some(li));
                upd(r_dn, &mut gjd, None);
                for d in 0..3 {
                    gju[[li, d]] += dself[d];
                }
            } else {
                upd(r_dn, &mut gjd, Some(li));
                upd(r_up, &mut gju, None);
                for d in 0..3 {
                    gjd[[li, d]] += dself[d];
                }
            }
        }

        (jr, pulay, gju, gjd)
    }
}

/// PERIODIC one-electron-move Jastrow mesh fields for the ECP non-local STRESS candidate
/// loop (`vmc_grad_e_l_eps_pbc`) -- the periodic-stress analogue of [`MeshJasFastPbc`]
/// path. The stress mesh loop needs, per candidate, the value ratio (the
/// det-ratio weight), the moved electron's Jastrow DRIFT row, and the 3x3 value-STRAIN
/// Pulay (J1/J2/J3 cell-strain). A FULL recompute of the J3 pieces
/// (`jastrow_value_for_ratio` + `jastrow_grads` + `j3_pbc_value_strain`) per candidate is a full
/// periodic pGTO sweep over EVERY electron. A candidate hops ONE electron, so:
///  * the J3 value ratio reuses the value-convention [`J3RatioState`] rank-1 update (exactly
///    as [`MeshJasFastPbc`]);
///  * the moved electron's J3 batch DRIFT reduces to `(c0_val - w_val[:,li])` contracted
///    with the FRESH single-electron value-grad AOs -- byte-matching the batch
///    `compute_grads_and_laplacian_jastrow_three_body` convention the stress (and its JAX
///    oracle) use, NOT the cart full-image force-drift convention;
///  * the J3 VALUE strain reuses the SAME symmetric-block coef `coef[:,c] = c0_pg - w_pg[:,c]`
///    (the identity `M.a.(K+K^T) = M.(s - a_k)`): moving electron `li` shifts every OTHER
///    column's coef by the UNIFORM `u = M.(a_new - a_old)` and leaves `li`'s coef invariant,
///    so the moved-config strain = `base + FIELD_STRESS(u broadcast, BASE AO tables) +
///    (li column's FRESH tables - li column's BASE tables)`. The strain AO tables
///    (`pgto_grad`/kernel-rank AO gradient) are re-evaluated ONLY for the moved electron.
///
/// J1/J2 drift + value strain stay on the cheap full kernels (no AO sweep; `jastrow_grads`
/// J1/J2 + `jastrow_strain`). DTN / MO-J3 / composite-PW /
/// B-spline J3 -> `None` (caller keeps the full recompute), matching [`MeshJasFastPbc`];
/// `JQMC_RS_ECPF_SLOW=1` forces the reference path at the caller.
struct MeshJasStressPbc<'a> {
    mj: MeshJasFastPbc<'a>,
    positions: Array2<f64>,
    cell: Array2<f64>,
    j3s: Option<MeshJasStressJ3>,
    /// Base DTN value strain (3x3) at the base config; per candidate the moved-config
    /// DTN value strain = `base + dtn_value_strain_move_delta` (u/chi/f). `None` = no DTN.
    dtn_base_strain: Option<Array2<f64>>,
    /// Base J1/J2 value strain (3x3); per candidate `base + jastrow_strain_move_delta_batch`.
    j12_base_strain: Array2<f64>,
    /// Base J1/J2/DTN electron gradients (`jastrow_grads` without J3), `(n_up,3)`, `(n_dn,3)`:
    /// the moved row of a candidate is this row plus the one-electron J1/J2 and DTN deltas.
    base_g12_up: Array2<f64>,
    base_g12_dn: Array2<f64>,
}

/// Cached base strain tables (pGTO convention) + batch-drift base coeff for the stress J3.
struct MeshJasStressJ3 {
    /// batch-drift coeff base `c0_val = j1col + val_cross_up + val_cross_dn` (len n_orb).
    c0_val: Vec<f64>,
    /// base pGTO strain AO tables (folded grad `gf`, kernel grad `gk`, wrapped positions).
    gf_up: [Array2<f64>; 3],
    gk_up: [Array2<f64>; 3],
    rw_up: Array2<f64>,
    gf_dn: [Array2<f64>; 3],
    gk_dn: [Array2<f64>; 3],
    rw_dn: Array2<f64>,
    /// base J3 value strain (3x3) at the base config.
    base_strain: Array2<f64>,
    /// per-CART-AO home / kernel atom maps (the [`field_stress`] scatter ranks; the FORCE
    /// ctx `home` is per-sphe-orb and must NOT feed the strain leaves).
    home_cart: Vec<usize>,
    kernel: Vec<usize>,
    /// kernel-row image translations `t[kap] = centers[kap] - pos[kernel[kap]]`.
    tvec: Vec<[f64; 3]>,
    /// Per-step COLUMN-SUMMED base strain response, per spin: `bcast[mu][3c+d]` = the
    /// [`field_stress`] output of a column-UNIFORM cotangent `cm[mu,:] = e_mu` over ALL of
    /// that spin's base columns (fold term `Sum_k gf[c][mu,k]*(rw[k,d]-pos[home,d])` MINUS
    /// the kernel term `Sum_{kap:home=mu}(Sum_k gk[c][kap,k])*t[kap,d]`). The per-candidate
    /// broadcast delta `u (x) all-columns` then contracts in O(num_cart_ao*9) instead of a
    /// per-candidate O(num_ao_kernel*Ne*9) field sweep.
    bcast_up: Array2<f64>,
    bcast_dn: Array2<f64>,
    /// Shell geometry (home atoms + strain offsets) of `cart_j3` for the fused moved-electron
    /// mesh contraction ([`crate::mesh_vgs`]); `None` keeps the table sweep.
    geom: Option<crate::mesh_vgs::ShellGeom>,
}

/// Column-summed base strain-response tensor (see `MeshJasStressJ3::bcast_up`).
fn strain_bcast(
    cart: &AosGtoData,
    gf: &[Array2<f64>; 3],
    gk: &[Array2<f64>; 3],
    rw: &Array2<f64>,
    home_cart: &[usize],
    tvec: &[[f64; 3]],
    positions: &ArrayView2<f64>,
) -> Array2<f64> {
    let num_cart = cart.num_ao;
    let n = rw.nrows();
    let mut out = Array2::<f64>::zeros((num_cart, 9));
    for mu in 0..num_cart {
        let ra = home_cart[mu];
        for c in 0..3 {
            let (mut s_rw, mut s_g) = ([0.0_f64; 3], 0.0_f64);
            for k in 0..n {
                let g = gf[c][[mu, k]];
                s_g += g;
                for d in 0..3 {
                    s_rw[d] += g * rw[[k, d]];
                }
            }
            for d in 0..3 {
                out[[mu, 3 * c + d]] = s_rw[d] - s_g * positions[[ra, d]];
            }
        }
    }
    for kap in 0..cart.num_ao_kernel {
        let mu = cart.kernel_to_home[kap];
        let t = tvec[kap];
        for c in 0..3 {
            let mut s = 0.0;
            for k in 0..n {
                s += gk[c][[kap, k]];
            }
            for d in 0..3 {
                out[[mu, 3 * c + d]] -= s * t[d];
            }
        }
    }
    out
}

impl<'a> MeshJasStressPbc<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        jdata: &'a JastrowData,
        j3_force: Option<&'a J3Pbc<'_>>,
        j3ctx: Option<&J3StressPbc<'_>>,
        cell: &ArrayView2<f64>,
        positions: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        n_atom: usize,
    ) -> Option<Self> {
        // `j3_force` is the caller-owned [`J3Pbc`] view of `j3ctx` (drops `kernel`), built
        // long-lived so the shared value/pGTO rank-1 state can borrow it (identical gates).
        let mj = MeshJasFastPbc::new(jdata, j3_force, cell, r_up, r_dn, n_atom)?;
        // Strain caches only when a covered (pure-pGTO) J3 leaf is present.
        let j3s = match (j3ctx, mj.j3.as_ref()) {
            (Some(c), Some(j3)) => {
                let n_orb = j3.n_orb;
                let inv = inv3(cell);
                let c0_val: Vec<f64> = (0..n_orb)
                    .map(|o| j3.j1col[o] + j3.val_state.cross_up[o] + j3.val_state.cross_dn[o])
                    .collect();
                let rw_up = crate::mcmc::wrap_rows(r_up, cell, &inv);
                let tu = crate::ao_cart_deriv::pgto_strain_tables(
                    c.cart_j3,
                    &rw_up.view(),
                    false,
                    false,
                    DerivScreens::EXACT,
                );
                let (gf_up, gk_up) = (tu.gf, tu.gk);
                let rw_dn = crate::mcmc::wrap_rows(r_dn, cell, &inv);
                let td = crate::ao_cart_deriv::pgto_strain_tables(
                    c.cart_j3,
                    &rw_dn.view(),
                    false,
                    false,
                    DerivScreens::EXACT,
                );
                let (gf_dn, gk_dn) = (td.gf, td.gk);
                let mut base_strain = Array2::<f64>::zeros((3, 3));
                crate::stress_pbc::j3_pbc_value_strain(
                    c.cart_j3,
                    &c.t_mol.to_owned(),
                    &c.jm,
                    r_up,
                    r_dn,
                    cell,
                    c.home,
                    c.kernel,
                    positions,
                    c.pw,
                    c.gto_first,
                    &mut base_strain,
                );
                let home_cart = c.home.to_vec();
                let kernel = c.kernel.to_vec();
                let tvec: Vec<[f64; 3]> = (0..c.cart_j3.num_ao_kernel)
                    .map(|kap| {
                        let at = kernel[kap];
                        [
                            c.cart_j3.centers[[kap, 0]] - positions[[at, 0]],
                            c.cart_j3.centers[[kap, 1]] - positions[[at, 1]],
                            c.cart_j3.centers[[kap, 2]] - positions[[at, 2]],
                        ]
                    })
                    .collect();
                let bcast_up = strain_bcast(
                    c.cart_j3, &gf_up, &gk_up, &rw_up, &home_cart, &tvec, positions,
                );
                let bcast_dn = strain_bcast(
                    c.cart_j3, &gf_dn, &gk_dn, &rw_dn, &home_cart, &tvec, positions,
                );
                let geom = if std::env::var("JQMC_RS_MESH_TABLES").is_err() {
                    crate::mesh_vgs::ShellGeom::new(
                        c.cart_j3,
                        &home_cart,
                        Some((&kernel, positions)),
                    )
                } else {
                    None
                };
                Some(MeshJasStressJ3 {
                    geom,
                    c0_val,
                    gf_up,
                    gk_up,
                    rw_up,
                    gf_dn,
                    gk_dn,
                    rw_dn,
                    base_strain,
                    home_cart,
                    kernel,
                    tvec,
                    bcast_up,
                    bcast_dn,
                })
            }
            _ => None,
        };
        let dtn_base_strain = jdata
            .dtn
            .as_ref()
            .map(|d| crate::stress_pbc::dtn_value_strain(d, r_up, r_dn));
        let mut j12_base_strain = Array2::<f64>::zeros((3, 3));
        crate::stress_pbc::jastrow_strain(
            jdata.one_body.as_ref(),
            jdata.two_body.as_ref(),
            r_up,
            r_dn,
            &mut j12_base_strain,
        );
        let (base_g12_up, base_g12_dn) = jastrow_grads(Some(jdata), None, None, r_up, r_dn);
        Some(MeshJasStressPbc {
            mj,
            positions: positions.to_owned(),
            cell: cell.to_owned(),
            j3s,
            dtn_base_strain,
            j12_base_strain,
            base_g12_up,
            base_g12_dn,
        })
    }

    /// BATCHED stress mesh candidates of one `(electron, atom)` group. Returns per candidate
    /// `(jr, jdrift_li, j_strain)`: `jr` is the full J1/J2/DTN/J3 value ratio, `jdrift_li` the
    /// moved electron's Jastrow drift row (J1/J2/DTN as `jastrow_grads`, plus the J3 batch
    /// drift), `j_strain` the moved-config Jastrow value strain (J1/J2 + DTN + J3). Every piece
    /// is base + one-electron delta; the caller adds the determinant parts.
    fn stress_candidates(
        &self,
        is_up: bool,
        li: usize,
        new_positions: &[[f64; 3]],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> Vec<(f64, [f64; 3], Array2<f64>)> {
        let nv = new_positions.len();
        // J1/J2: value ratio, moved drift row (base + the one-electron delta of
        // `MeshJasFastPbc::candidate_j12`) and value strain (base + delta).
        let j12_strain = crate::stress_pbc::jastrow_strain_move_delta_batch(
            self.mj.jdata.one_body.as_ref(),
            self.mj.jdata.two_body.as_ref(),
            r_up,
            r_dn,
            is_up,
            li,
            new_positions,
        );
        let (base_g, mj_g) = if is_up {
            (&self.base_g12_up, &self.mj.base_gju)
        } else {
            (&self.base_g12_dn, &self.mj.base_gjd)
        };
        let mut out: Vec<(f64, [f64; 3], Array2<f64>)> = new_positions
            .iter()
            .zip(j12_strain)
            .map(|(&np, ds)| {
                let (jr, _, gju, gjd) = self.mj.candidate_j12(is_up, li, np, r_up, r_dn);
                let g = if is_up { &gju } else { &gjd };
                let mut drift = [0.0; 3];
                for d in 0..3 {
                    drift[d] = base_g[[li, d]] + (g[[li, d]] - mj_g[[li, d]]);
                }
                (jr, drift, &self.j12_base_strain + &ds)
            })
            .collect();

        // ---- DTN (independent of J3): fold the value ratio into `jr` and set the
        // moved-config DTN value strain (base + one-electron delta) into the strain
        // accumulator. The moved-electron DTN DRIFT row is supplied by the caller's
        // per-candidate full jastrow-drift kernel (which includes DTN), so nothing is
        // added to `jdrift` here (that stays J3-only). ----
        if let (Some(dd), Some(base_strain)) =
            (self.mj.jdata.dtn.as_ref(), self.dtn_base_strain.as_ref())
        {
            let mut ratios = vec![0.0_f64; nv];
            let mesh = Array2::from_shape_vec(
                (nv, 3),
                new_positions
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect::<Vec<f64>>(),
            )
            .unwrap();
            crate::dtn::compute_dtn_ratio_mesh(
                dd,
                r_up,
                r_dn,
                is_up,
                li,
                &mesh.view(),
                &mut ratios,
            );
            let g_idx = if is_up { li } else { r_up.nrows() + li };
            // Moved electron's DTN gradient row: the fp64 drift row, or under an fp32
            // `jastrow_grad_lap` zone the matching row of the fp32 analytic gradient leaf
            // (`dtn_grad_row_f32`, bit-identical to that row of the full fp32 kernel); the old
            // row once per group.
            let f32_row = dd.precision_policy.is_f32("jastrow_grad_lap");
            let drift_row = |pos: [f64; 3]| {
                if f32_row {
                    crate::dtn::dtn_grad_row_f32(dd, r_up, r_dn, g_idx, pos)
                } else {
                    crate::stress_pbc::dtn_drift_row(dd, r_up, r_dn, g_idx, pos)
                }
            };
            let row_old = {
                let p = if is_up { r_up.row(li) } else { r_dn.row(li) };
                drift_row([p[0], p[1], p[2]])
            };
            let deltas = crate::stress_pbc::dtn_value_strain_move_delta_batch(
                dd,
                r_up,
                r_dn,
                is_up,
                li,
                new_positions,
            );
            for (v, delta) in deltas.iter().enumerate() {
                out[v].0 = crate::jastrow_factor::multiply_ratio_components(
                    self.mj.jdata,
                    out[v].0,
                    ratios[v],
                );
                out[v].2 += &(base_strain + delta);
                // moved row of the DTN drift delta: the moved electron's DTN gradient at the
                // candidate minus at its configuration position (the old part once per group).
                let new = drift_row(new_positions[v]);
                for d in 0..3 {
                    out[v].1[d] += new[d] - row_old[d];
                }
            }
        }

        let (j3, s3) = match (self.mj.j3.as_ref(), self.j3s.as_ref()) {
            (Some(j3), Some(s3)) => (j3, s3),
            _ => return out,
        };
        use numpy::ndarray::s;
        let cell = self.cell.view();
        let inv = inv3(&cell);
        let n_orb = j3.n_orb;
        let cart = j3.cart_j3;
        // per-CART-AO scatter maps (field_stress ranks) -- NOT the per-sphe force `home`.
        let home = s3.home_cart.as_slice();
        let kernel = &s3.kernel;
        let positions = self.positions.view();
        let num_cart_ao = cart.num_ao;
        let mut rmat = Array2::<f64>::zeros((nv, 3));
        for (v, p) in new_positions.iter().enumerate() {
            rmat[[v, 0]] = p[0];
            rmat[[v, 1]] = p[1];
            rmat[[v, 2]] = p[2];
        }
        // ---- value-convention AOs (value + grad) of the moved electron ----
        let ruw = j3
            .orb
            .wrap_electrons(&rmat.view())
            .unwrap_or_else(|| rmat.clone());
        let (av_new, gvx, gvy, gvz, _lap) =
            crate::determinant::orb_value_grad_lap(j3.orb, &ruw.view());
        let av_old = if is_up {
            &j3.val_state.aos_up
        } else {
            &j3.val_state.aos_dn
        };
        let wv_same = if is_up {
            &j3.val_state.w_up
        } else {
            &j3.val_state.w_dn
        };
        let cross_v = if is_up {
            &j3.val_state.cross_dn
        } else {
            &j3.val_state.cross_up
        };
        let mut dv = av_new.clone();
        for o in 0..n_orb {
            let a_old = av_old[[o, li]];
            for v in 0..nv {
                dv[[o, v]] -= a_old;
            }
        }
        let vmv = dv.t().dot(wv_same); // (nv, N_same)
                                       // batch-drift coeff (base, li column): coef_drift[o] = c0_val[o] - w_val_same[o,li]
        let coef_drift: Vec<f64> = (0..n_orb)
            .map(|o| s3.c0_val[o] - wv_same[[o, li]])
            .collect();
        // li column coef (pGTO, base): coef_li[o] = c0_pg[o] - w_pg_same[o,li]; -> cart rank.
        let w_pg_same = if is_up {
            &j3.pg_state.w_up
        } else {
            &j3.pg_state.w_dn
        };
        let coef_li: Vec<f64> = (0..n_orb).map(|o| j3.c0[o] - w_pg_same[[o, li]]).collect();
        let cm_li = j3
            .t_mol
            .t()
            .dot(&Array2::from_shape_vec((n_orb, 1), coef_li).unwrap());
        let rw_new = crate::mcmc::wrap_rows(&rmat.view(), &cell, &inv);
        // With `geom`, ONE fused unscreened contraction of the moved electron gives the folded
        // pGTO value table and the fresh li-column strain response `FS(cm_li, fresh col v)`;
        // otherwise the value sweep plus the kernel-rank strain tables.
        let fused = s3.geom.as_ref().map(|geom| {
            let pts: Vec<[f64; 3]> = (0..nv)
                .map(|v| [rw_new[[v, 0]], rw_new[[v, 1]], rw_new[[v, 2]]])
                .collect();
            let mut o = crate::mesh_vgs::MeshVgs::new(nv, None, true, Some(num_cart_ao));
            crate::mesh_vgs::mesh_vgs(
                cart,
                &pts,
                crate::mesh_vgs::Weights::Shared(cm_li.as_slice().unwrap()),
                AoScreen::None,
                Some(geom),
                &mut o,
            );
            o
        });
        // ---- pGTO-convention value AO of the moved electron: strain `u = M.(a_new - a_old)` ----
        let ap_new = match &fused {
            Some(o) => j3
                .t_mol
                .dot(&Array2::from_shape_vec((num_cart_ao, nv), o.vtab.clone()).unwrap()),
            None => j3.t_mol.dot(&crate::ao_cart_deriv::pgto_deriv(
                j3.cart_j3,
                &rmat.view(),
                &cell,
                (0, 0, 0),
            )),
        };
        let ap_old = if is_up {
            &j3.pg_state.aos_up
        } else {
            &j3.pg_state.aos_dn
        };
        let mut dp = ap_new.clone();
        for o in 0..n_orb {
            let a_old = ap_old[[o, li]];
            for v in 0..nv {
                dp[[o, v]] -= a_old;
            }
        }
        let wd = j3.j3c.dot(&dp); // (n_orb, nv), column v == u
                                  // moved-electron pGTO strain AO tables (fresh; the SINGLE moved electron over nv cands).
        let fresh_tabs = if fused.is_none() {
            let tn = crate::ao_cart_deriv::pgto_strain_tables(
                j3.cart_j3,
                &rw_new.view(),
                false,
                false,
                DerivScreens::EXACT,
            );
            Some((tn.gf, tn.gk))
        } else {
            None
        };
        let (gf_same, gk_same, rw_same, bcast_same) = if is_up {
            (&s3.gf_up, &s3.gk_up, &s3.rw_up, &s3.bcast_up)
        } else {
            (&s3.gf_dn, &s3.gk_dn, &s3.rw_dn, &s3.bcast_dn)
        };
        let bcast_other = if is_up { &s3.bcast_dn } else { &s3.bcast_up };
        // fixed li-column base strain tables (single column) for the (B) subtraction.
        let col =
            |a: &Array2<f64>, j: usize| -> Array2<f64> { a.slice(s![.., j..j + 1]).to_owned() };
        let gf_base_li = [
            col(&gf_same[0], li),
            col(&gf_same[1], li),
            col(&gf_same[2], li),
        ];
        let gk_base_li = [
            col(&gk_same[0], li),
            col(&gk_same[1], li),
            col(&gk_same[2], li),
        ];
        let rw_base_li = rw_same.slice(s![li..li + 1, ..]).to_owned();
        let mut sub_li = Array2::<f64>::zeros((3, 3));
        crate::stress_pbc::field_stress(
            cart,
            &cm_li,
            &gf_base_li,
            &gk_base_li,
            &rw_base_li,
            home,
            kernel,
            &positions,
            &mut sub_li,
        );
        // Per-GROUP combined broadcast tensor `comb[mu][3c+d]` = same-spin bcast MINUS the
        // li column's response PLUS the other-spin bcast: the exact strain response of a
        // column-uniform `u` over every base column except li (same spin) and every base
        // column of the other spin. Per candidate the delta is then `Sum_mu u[mu]*comb[mu]`
        // -- O(num_cart_ao*9) instead of a per-candidate O(num_ao_kernel*Ne*9) field sweep.
        let mut comb = bcast_same + bcast_other;
        for mu in 0..num_cart_ao {
            let ra = home[mu];
            for c in 0..3 {
                let g = gf_same[c][[mu, li]];
                for d in 0..3 {
                    comb[[mu, 3 * c + d]] -= g * (rw_same[[li, d]] - positions[[ra, d]]);
                }
            }
        }
        for kap in 0..cart.num_ao_kernel {
            let mu = cart.kernel_to_home[kap];
            let t = s3.tvec[kap];
            for c in 0..3 {
                let g = gk_same[c][[kap, li]];
                if g == 0.0 {
                    continue;
                }
                for d in 0..3 {
                    comb[[mu, 3 * c + d]] += g * t[d];
                }
            }
        }

        for v in 0..nv {
            let (jr, drift, strain) = &mut out[v];
            // ---- J3 VALUE ratio factor ----
            let ratio = if j3.val_state.f32_arithmetic {
                let aos_new_col: Vec<f64> = (0..n_orb).map(|o| av_new[[o, v]]).collect();
                crate::jastrow_factor::j3_ratio(
                    &j3.jm.view(),
                    &j3.val_state,
                    is_up,
                    li,
                    &aos_new_col,
                )
            } else {
                let mut term1 = 0.0;
                let mut term4 = 0.0;
                for o in 0..n_orb {
                    term1 += j3.j1col[o] * dv[[o, v]];
                    term4 += cross_v[o] * dv[[o, v]];
                }
                let mut term23 = 0.0;
                for m in 0..vmv.ncols() {
                    if m != li {
                        term23 += vmv[[v, m]];
                    }
                }
                (term1 + term23 + term4).exp()
            };
            *jr = crate::jastrow_factor::multiply_ratio_components(self.mj.jdata, *jr, ratio);
            // ---- J3 batch DRIFT of the moved electron ----
            for (d, g) in [&gvx, &gvy, &gvz].iter().enumerate() {
                let mut sdr = 0.0;
                for o in 0..n_orb {
                    sdr += coef_drift[o] * g[[o, v]];
                }
                drift[d] += sdr;
            }
            // ---- J3 VALUE strain = base + delta ----
            let mut sm = s3.base_strain.clone();
            let u_cart = j3.t_mol.t().dot(&wd.slice(s![.., v]).to_owned()); // (num_cart_ao)
                                                                            // (A)+(C) uniform broadcast `u` over every non-li base column, via the
                                                                            // per-group combined tensor (see `comb` above).
            for mu in 0..num_cart_ao {
                let u = u_cart[mu];
                if u == 0.0 {
                    continue;
                }
                for c in 0..3 {
                    for d in 0..3 {
                        sm[[c, d]] += u * comb[[mu, 3 * c + d]];
                    }
                }
            }
            // (B) li column: FRESH tables - BASE tables (coef `cm_li` invariant under the hop).
            match (&fused, &fresh_tabs) {
                (Some(o), _) => {
                    for c in 0..3 {
                        for d in 0..3 {
                            sm[[c, d]] += o.strain[v][c][d];
                        }
                    }
                }
                (None, Some((gf_new, gk_new))) => {
                    let gf_new_li = [col(&gf_new[0], v), col(&gf_new[1], v), col(&gf_new[2], v)];
                    let gk_new_li = [col(&gk_new[0], v), col(&gk_new[1], v), col(&gk_new[2], v)];
                    let rw_new_li = rw_new.slice(s![v..v + 1, ..]).to_owned();
                    crate::stress_pbc::field_stress(
                        cart, &cm_li, &gf_new_li, &gk_new_li, &rw_new_li, home, kernel, &positions,
                        &mut sm,
                    );
                }
                (None, None) => unreachable!(),
            }
            sm -= &sub_li;
            // `+=` (not `=`): the DTN block above may have already deposited the DTN value
            // strain into the accumulator; the J3 moved-config strain adds on top.
            *strain += &sm;
        }
        out
    }
}

/// The CALLER's local energy is the FAST-chunk e_L, whose AO tables are screened
/// (molecular Cartesian leaves): the ECP non-local mesh contractions use the FAST
/// value screen of `dot_aos_multi_cutoff_cart` (shell kept iff
/// `pz_val*r^2 <= LEPSBAS`, primitive kept iff `prim_val[q]*r^2 <= LEPSBAS`), and
/// the kinetic uses the FAST derivative screen (`pz_deriv*r^2 <= LEPSBAS`,
/// `compute_aos_grad_lap_fast_cut`). When set, the force takes the corresponding
/// AO quantities on the SAME keep sets
/// ([`crate::ao_cart_deriv::ao_val_grad_mesh_screened`],
/// [`crate::ao_cart_deriv::ao_deriv_bundle_fast_cut`]), i.e. it differentiates the
/// screened terms that e_L evaluates. Default (both off): exact unscreened AOs (the
/// bit-exact / binding routes). The GFMC e_L (`gfmc::MeshFastState::force_screen`)
/// screens its ECP mesh AOs per shell only (`mesh_shell_only`).
#[derive(Clone, Copy, Default, Debug)]
pub struct FastScreen {
    /// Determinant AOs (molecular Cartesian geminal, forward == derivative leaves).
    pub det: bool,
    /// J3 AOs (molecular Cartesian J3 leaf).
    pub j3: bool,
    /// The e_L's FAST ECP radial cut: (electron, channel) pairs with
    /// `|v_l(|r - R|)| < FAST_ECP_NONLOCAL_RADIAL_CUT` are dropped, here too.
    pub ecp_radial: bool,
    /// The e_L's ECP mesh value screen keeps every primitive of a kept shell (the
    /// GFMC molecular candidate batch) instead of also dropping primitives with
    /// `prim_val[q]*r^2 > LEPSBAS` (the VMC FAST `dot_aos_multi_cutoff_cart`).
    pub mesh_shell_only: bool,
}

/// Molecular ECP non-local mesh force `(d_re (n_e,3), d_r (n_atom,3))` = d(Sum V_nl)/d(r,R)
/// on the NN mesh (tmove/dltmove). Extracted from gfmc_grad_e_l_ecp; reused by the ECP stress
/// (its virial IS the ECP non-local stress, since V_nl is a function of r,R only -- fixed grid).
#[allow(clippy::too_many_arguments)]
pub fn ecp_nonlocal_force_mol(
    gem: &GeminalData,
    forward_gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    // BASE-config geminal inverse reuse (VMC observable; `None` == fresh LU). Mesh
    // re-evaluations always use their own fresh inverse.
    g_inv: Option<&ArrayView2<f64>>,
    screen: FastScreen,
    // Base fields the caller already holds at (r_up, r_dn) with the same `g_inv`
    // (`vmc_diag_grads_with_forward`); `None` evaluates them here.
    shared: Option<MeshShared>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (shared_det, shared_jas) = match shared {
        Some(sh) => (Some(sh.det), sh.jas),
        None => (None, None),
    };
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let j3 = match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => Some((o, m)),
        _ => None,
    };
    let j1 = jastrow_full.one_body.as_ref();
    let dtn = jastrow_full.dtn.as_ref();
    let has_j = j1.is_some() || jastrow_full.two_body.is_some() || dtn.is_some() || j3.is_some();
    let jas_opt: Option<&JastrowData> = if has_j { Some(jastrow_full) } else { None };
    let mut d_r = Array2::<f64>::zeros((n_atom, 3));
    let mut d_re = Array2::<f64>::zeros((n_e, 3));
    // grid virial `Sum (dV_nl/dgp_v)(x)gp_v` of the ECP quadrature directions. Diagnostic only:
    // no stress consumer adds it, because the strained quadrature keeps its
    // directions fixed; carrying them as RT@F.T moves the points off the ECP sphere.
    let mut grid_stress = Array2::<f64>::zeros((3, 3));
    let elec = |gi: usize| -> [f64; 3] {
        if gi < n_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    // rotated grid (grid_points @ RT); molecular has no min-image.
    let nv = weights.len();
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
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let dd = [
                    re[0] - positions[[ia, 0]],
                    re[1] - positions[[ia, 1]],
                    re[2] - positions[[ia, 2]],
                ];
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    // Cache base determinant fields; each mesh candidate is a single-electron
    // rank-1 hop (Sherman-Morrison), so we NEVER rebuild the full system per candidate.
    let mesh_fast = crate::timed!(
        "ecpf.det_base",
        MeshDetFast::new_with_forward(
            gem,
            forward_gem,
            r_up,
            r_dn,
            nuc_up,
            nuc_dn,
            n_atom,
            g_inv,
            shared_det
        )
    );
    // Cache base Jastrow fields (J3 coupling tables + drift anchors); each
    // candidate's Jastrow value/Pulay/drift is a one-column update instead of a
    // full recompute (`None` -> uncovered leaf (MO-J3/periodic) -> full-recompute path).
    // `JQMC_RS_ECPF_SLOW=1` forces the full-recompute reference path for A/B debugging
    // for the one-electron update; instrumentation only, never set in production).
    let mesh_jas = if has_j && std::env::var("JQMC_RS_ECPF_SLOW").is_err() {
        crate::timed!(
            "ecpf.jas_base",
            MeshJasFast::new(
                jastrow_full,
                j3_orb,
                j3_jm,
                nuc_j3,
                r_up,
                r_dn,
                n_atom,
                shared_jas
            )
        )
    } else {
        None
    };
    // Full-recompute Jastrow fallback (has_j but no one-electron update): base value,
    // Pulay and drift, ONLY computed when used -- the base value is the
    // O(Ne^2*nao^2) scalar J3 bilinear (1.8 s on the water20 dense-J3 cell).
    let jas_fallback = has_j && mesh_jas.is_none();
    let jas_pulay_full = |mu: &ArrayView2<f64>, md: &ArrayView2<f64>| -> Array2<f64> {
        let mut p = Array2::<f64>::zeros((n_atom, 3));
        if let Some(j1) = j1 {
            p += &crate::timed!("ecpf.j1_pulay", grad_j1_wrt_r(j1, mu, md));
        }
        if let (Some(o), Some(m), Some(nc)) = (j3_orb, j3_jm, nuc_j3) {
            p += &crate::timed!("ecpf.j3_pulay", grad_j3_wrt_r(o, m, mu, md, nc, n_atom));
        }
        if let Some(d) = dtn {
            p += &crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, mu, md);
        }
        p
    };
    let (jd, base_jp_fb) = if jas_fallback {
        (
            crate::timed!(
                "ecpf.base_jval",
                crate::coulomb_potential::jastrow_value_for_ratio(jastrow_full, j3, r_up, r_dn)
            ),
            jas_pulay_full(r_up, r_dn),
        )
    } else {
        (0.0, Array2::zeros((0, 0)))
    };
    // base ln|Psi| drifts (full = det+Jastrow; det = det only). Computed once.
    let (base_det_du, base_det_dd) = crate::timed!("ecpf.base_drift_det", mesh_fast.base_drift());
    let (base_ju, base_jd) = if let Some(mj) = &mesh_jas {
        // the identical `jastrow_grads` call MeshJasFast::new anchors on
        (mj.base_gju.clone(), mj.base_gjd.clone())
    } else if has_j {
        crate::timed!(
            "ecpf.base_jgrads",
            jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn)
        )
    } else {
        (Array2::zeros((n_up, 3)), Array2::zeros((n_e - n_up, 3)))
    };
    let base_du_f = &base_det_du + &base_ju;
    let base_dd_f = &base_det_dd + &base_jd;
    let row = |up: &Array2<f64>, dn: &Array2<f64>, ee: usize| -> [f64; 3] {
        let (m, k) = if ee < n_up { (up, ee) } else { (dn, ee - n_up) };
        [m[[k, 0]], m[[k, 1]], m[[k, 2]]]
    };

    // Every candidate's `coeff_v * (F_v - F_base)` (F = det/Jastrow Pulay and the
    // drift of every electron but the moved one) is summed in these accumulators:
    // the determinant part per group from ONE aggregated column change, the
    // contraction against the big AO-gradient tables once per configuration.
    let mut det_acc = mesh_fast.new_acc();
    let mut jas_acc = mesh_jas.as_ref().map(|mj| mj.new_acc());
    let mut fb_pulay = Array2::<f64>::zeros((n_atom, 3));
    let mut fb_du = Array2::<f64>::zeros((n_up, 3));
    let mut fb_dd = Array2::<f64>::zeros((n_e - n_up, 3));
    let mut beta = vec![0.0_f64; nv];
    let mut wj = vec![0.0_f64; nv];

    for gi in 0..n_e {
        let is_up = gi < n_up;
        let re = elec(gi);
        for &ia in &nearest_set(re) {
            let rel = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let uhat = [rel[0] / nrm, rel[1] / nrm, rel[2] / nrm]; // electron->atom
            let li = if is_up { gi } else { gi - n_up };
            // The e_L's radial cut depends on |r_e - R_ia| only: when it drops every
            // channel of this atom, all of the group's candidates have ang = 0 and
            // contribute nothing (the e_L skips the pair too), so skip the group.
            if screen.ecp_radial
                && (0..nuc_n.len()).all(|t| {
                    nuc_n[t] as usize != ia
                        || (c_n[t]
                            * crate::coulomb_potential::ecp_pow(nrm, p_n[t] - 2.0)
                            * (-e_n[t] * nrm * nrm).exp())
                        .abs()
                            < crate::coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT
                })
            {
                continue;
            }
            // ---- pass 1: candidate geometry (all nv mesh points of this (gi, ia)) ----
            let new_positions: Vec<[f64; 3]> = (0..nv)
                .map(|v| {
                    let g = gp[v];
                    [
                        re[0] + rel[0] + nrm * g[0],
                        re[1] + rel[1] + nrm * g[1],
                        re[2] + rel[2] + nrm * g[2],
                    ]
                })
                .collect();
            // ---- per-candidate ratios + moved-electron drifts (one AO sweep each) ----
            let det_g = crate::timed!(
                "ecpf.det_cand",
                mesh_fast.prepare(
                    is_up,
                    li,
                    &new_positions,
                    screen.det,
                    !screen.mesh_shell_only
                )
            );
            let jas_g = mesh_jas.as_ref().map(|mj| {
                crate::timed!(
                    "ecpf.jas_fast",
                    mj.prepare(
                        is_up,
                        li,
                        &new_positions,
                        r_up,
                        r_dn,
                        screen.j3,
                        !screen.mesh_shell_only
                    )
                )
            });
            // ---- pass 2: per-candidate angular channel + moved-electron force terms ----
            for v in 0..nv {
                let __cand_t = crate::prof::on().then(std::time::Instant::now);
                let g = gp[v];
                let dnv = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let cosv = (-rel[0] * g[0] - rel[1] * g[1] - rel[2] * g[2]) / (nrm * dnv);
                let new_pos = new_positions[v];
                // angular sum + its nrm/cosv derivatives over channels at this atom
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (p, e, c) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-e * nrm * nrm).exp();
                    let v_l = c * nrm.powf(p - 2.0) * expf;
                    if screen.ecp_radial {
                        // the e_L's own drop test (`compute_ecp_nonlocal_parts_all_pairs_radial_cut`)
                        let v_el = c
                            * crate::coulomb_potential::ecp_pow(nrm, p - 2.0)
                            * (-e * nrm * nrm).exp();
                        if v_el.abs() < crate::coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT {
                            continue;
                        }
                    }
                    let v_lp =
                        c * ((p - 2.0) * nrm.powf(p - 3.0) - 2.0 * e * nrm.powf(p - 1.0)) * expf;
                    let (pl, dpl) = leg_and_deriv(l, cosv);
                    let w = ((2 * l + 1) as f64) * weights[v];
                    ang += v_l * w * pl;
                    aprime += v_lp * w * pl;
                    acos += v_l * w * dpl;
                }
                // dang/dr_i = aprime*(-u_hat) + acos*(g/(nrm*dnv) + cosv*u_hat/nrm); dang/dR_ia = -that.
                let mut dang_dri = [0.0_f64; 3];
                for d in 0..3 {
                    dang_dri[d] =
                        aprime * (-uhat[d]) + acos * (g[d] / (nrm * dnv) + cosv * uhat[d] / nrm);
                }
                // moved config (fallback Jastrow full recompute only)
                let moved_cfg = || {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if is_up {
                        mu[[gi, 0]] = new_pos[0];
                        mu[[gi, 1]] = new_pos[1];
                        mu[[gi, 2]] = new_pos[2];
                    } else {
                        let k = gi - n_up;
                        md[[k, 0]] = new_pos[0];
                        md[[k, 1]] = new_pos[1];
                        md[[k, 2]] = new_pos[2];
                    }
                    (mu, md)
                };
                let det_ratio = det_g.ratio[v];
                let jr = match &jas_g {
                    Some(jg) => jg.jr[v],
                    None if jas_fallback => {
                        let (mu, md) = moved_cfg();
                        let jnew = crate::timed!(
                            "ecpf.jval",
                            crate::coulomb_potential::jastrow_value_for_ratio(
                                jastrow_full,
                                j3,
                                &mu.view(),
                                &md.view()
                            )
                        );
                        (jnew - jd).exp()
                    }
                    None => 1.0,
                };
                let det_amp = ang * det_ratio;
                // branch: tmove -> full ratio; dltmove -> det-only (SP, det_amp>=0) / full (FN)
                let use_full = !dltmove || det_amp < 0.0;
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                let coeff = ang * ratio; // = V for this candidate
                beta[v] = if use_full { ang * jr } else { ang }; // coeff = beta * det_ratio
                wj[v] = if use_full { coeff } else { 0.0 };

                // ln|Psi| drift of the moved electron at the candidate, and its base.
                let mut dm = det_g.moved[v];
                let base_gi = if use_full {
                    row(&base_du_f, &base_dd_f, gi)
                } else {
                    row(&base_det_du, &base_det_dd, gi)
                };
                if use_full {
                    if let Some(jg) = &jas_g {
                        for (d, dm_d) in dm.iter_mut().enumerate() {
                            *dm_d += jg.moved[v][d];
                        }
                    } else if jas_fallback {
                        let (mu, md) = moved_cfg();
                        let pj = jas_pulay_full(&mu.view(), &md.view());
                        let (gju, gjd) = crate::timed!(
                            "ecpf.jgrads",
                            jastrow_grads(jas_opt, j3_orb, j3_jm, &mu.view(), &md.view())
                        );
                        fb_pulay.scaled_add(coeff, &(pj - &base_jp_fb));
                        fb_du.scaled_add(coeff, &(&gju - &base_ju));
                        fb_dd.scaled_add(coeff, &(&gjd - &base_jd));
                        let jm = row(&gju, &gjd, gi);
                        let jb = row(&base_ju, &base_jd, gi);
                        for d in 0..3 {
                            dm[d] += jm[d];
                            // the moved row is handled per candidate below
                            if is_up {
                                fb_du[[li, d]] -= coeff * (jm[d] - jb[d]);
                            } else {
                                fb_dd[[li, d]] -= coeff * (jm[d] - jb[d]);
                            }
                        }
                    }
                }
                // (a) ang-part: dang/dtheta * ratio
                for d in 0..3 {
                    d_re[[gi, d]] += ratio * dang_dri[d];
                    d_r[[ia, d]] += ratio * (-dang_dri[d]);
                }
                // (b) ratio-part, moved electron: dlnPsi_mesh/dr_i = -(drift_mesh[i]*g)*u_hat,
                // and R_ia also moves new_pos: + drift_mesh[i] + (drift_mesh[i]*g)*u_hat.
                // Every other electron/atom term is in the accumulators.
                let dm_dot_g: f64 = (0..3).map(|c| dm[c] * g[c]).sum();
                for d in 0..3 {
                    d_re[[gi, d]] += coeff * (-dm_dot_g * uhat[d] - base_gi[d]);
                    d_r[[ia, d]] += coeff * (dm[d] + dm_dot_g * uhat[d]);
                }
                // (c) grid virial (diagnostic, not a stress term): dV_nl/dgp_v =
                // acos*(dcosv/dgp)*ratio + coeff*nrm*drift_mesh[gi] (new_pos = R + nrm*gp).
                // Accumulate Sum (dV_nl/dgp_c)*gp_d.
                for a in 0..3 {
                    let dcosv_dg = -rel[a] / (nrm * dnv) - cosv * g[a] / (dnv * dnv);
                    let dv_dga = acos * dcosv_dg * ratio + coeff * nrm * dm[a];
                    for b in 0..3 {
                        grid_stress[[a, b]] += dv_dga * g[b];
                    }
                }
                if let Some(t) = __cand_t {
                    crate::prof::add("ecpf.cand_total", t.elapsed().as_nanos());
                }
            }
            crate::timed!(
                "ecpf.det_acc",
                mesh_fast.accumulate(&det_g, &beta, &mut det_acc)
            );
            if let (Some(mj), Some(jg), Some(acc)) = (&mesh_jas, &jas_g, jas_acc.as_mut()) {
                crate::timed!("ecpf.jas_acc", mj.accumulate(jg, &wj, acc));
            }
        }
    }
    // ---- other electrons / all atoms: Sum_v coeff_v (F_v - F_base) ----
    let mut add = |(p, du, dd): (Array2<f64>, Array2<f64>, Array2<f64>)| {
        d_r += &p;
        let mut up = d_re.slice_mut(numpy::ndarray::s![0..n_up, ..]);
        up += &du;
        let mut dn = d_re.slice_mut(numpy::ndarray::s![n_up.., ..]);
        dn += &dd;
    };
    add(crate::timed!("ecpf.det_finish", mesh_fast.finish(det_acc)));
    if let (Some(mj), Some(acc)) = (&mesh_jas, jas_acc) {
        add(mj.finish(acc));
    }
    if jas_fallback {
        add((fb_pulay, fb_du, fb_dd));
    }
    (d_re, d_r, grid_stress)
}

/// `de_L^t/deps (3x3)` for the molecular ECP jnone LRDMC diagonal e_L (tmove/dltmove).
/// Continuum ECP stress = bare+kinetic (grad_el_wrt_strain_jnone) + virial(ECP local
/// force) + virial(NN non-local force ecp_nonlocal_force_mol). The NN non-local V_nl is
/// a function of (r,R) only (fixed grid) so its stress = its force virial. + S' pattern
/// (masked kinetic virial; swap=0 for ECP; strained-mesh sigma_PP).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_jnone(
    gem: &GeminalData,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> Array2<f64> {
    let jnone = JastrowData {
        precision_policy: gem.orb_up.precision_policy,
        one_body: None,
        two_body: None,
        three_body: None,
        three_body_complex_only: false,
        dtn: None,
    };

    let _ = alat;

    let mut sigma =
        crate::stress::grad_el_wrt_strain_jnone(gem, positions, z_eff, r_up, r_dn, nuc_up, nuc_dn);
    let (dl_re, dl_r) = ecp_local_force(positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn);
    let (nl_re, nl_r, _grid_stress) = ecp_nonlocal_force_mol(
        gem,
        gem,
        &jnone,
        None,
        None,
        positions,
        r_up,
        r_dn,
        rt,
        nuc_up,
        nuc_dn,
        None,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        None,
        FastScreen::default(),
        None,
    );
    let ecp_re = &dl_re + &nl_re;
    let ecp_r = &dl_r + &nl_r;
    sigma += &crate::stress::virial(&ecp_re, &ecp_r, r_up, r_dn, positions);
    sigma
}

/// Molecular ECP GFMC stress with the complete real Jastrow family
/// (J1/J2/J3/DTN), for tmove or dltmove.  This is the with-Jastrow counterpart
/// of [`gfmc_stress_ecp_jnone`]: ECP S' membership uses the bare electron-ion
/// channel, the smooth kinetic and masked kinetic carry the full Jastrow drift,
/// shifted-config Pulay stress includes every active Jastrow leaf, and the ECP
/// non-local mesh contributes its position virial (quadrature directions held fixed
/// under the strain; no grid-stretch term).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_withj(
    gem: &GeminalData,
    jastrow_full: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let j3 = match (j3_orb, j3_jm) {
        (Some(o), Some(m)) => Some((o, m)),
        _ => None,
    };
    let j1 = jastrow_full.one_body.as_ref();
    let j2 = jastrow_full.two_body.as_ref();
    let dtn = jastrow_full.dtn.as_ref();
    let has_j = j1.is_some() || j2.is_some() || dtn.is_some() || j3.is_some();
    let jas_opt = if has_j { Some(jastrow_full) } else { None };

    let _ = alat;
    // Full drift for the explicit DTN kinetic force.  The other kinetic
    // families obtain the same cotangent inside their shared reverse kernels.
    let cot = dtn.map(|_| {
        let (gj_up, gj_dn) = jastrow_grads(jas_opt, j3_orb, j3_jm, r_up, r_dn);
        let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
        let mut out = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                out[[e, c]] = if e < n_up {
                    gj_up[[e, c]] + gd_up[[e, c]]
                } else {
                    gj_dn[[e - n_up, c]] + gd_dn[[e - n_up, c]]
                };
            }
        }
        out
    });

    let mut sigma = crate::stress::grad_el_wrt_strain_withj(
        gem, jas_opt, j1, j2, j3_orb, j3_jm, positions, z_eff, r_up, r_dn, nuc_up, nuc_dn, nuc_j3,
    );
    if let (Some(d), Some(cot)) = (dtn, cot.as_ref()) {
        let d_r = crate::forces_dtn::grad_t_dtn_wrt_r_atom(d, cot, r_up, r_dn);
        let d_re = crate::forces_dtn::grad_t_dtn_wrt_r(d, cot, r_up, r_dn);
        sigma += &crate::stress::virial(&d_re, &d_r, r_up, r_dn, positions);
    }

    let (dl_re, dl_r) = ecp_local_force(positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn);
    let (nl_re, nl_r, _grid_stress) = ecp_nonlocal_force_mol(
        gem,
        gem,
        jastrow_full,
        j3_orb,
        j3_jm,
        positions,
        r_up,
        r_dn,
        rt,
        nuc_up,
        nuc_dn,
        nuc_j3,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        None,
        FastScreen::default(),
        None,
    );
    sigma += &crate::stress::virial(&(&dl_re + &nl_re), &(&dl_r + &nl_r), r_up, r_dn, positions);
    sigma
}

/// `de_L^t/deps (3x3)` for the PBC (Ewald, Gamma) ECP jnone LRDMC diagonal e_L (tmove).
/// Continuum = det kinetic (pgto_kstress) + Ewald ee/ei/ii strain + ECP local
/// strain + NN ECP non-local stress (fixed quadrature directions). - w-masked PBC kinetic stress
/// + strained-mesh sigma_PP (pgto_det_stress). `leaves_*` are the (cart_aos, c_cart,
///   home, kernel_atom) stress leaves. stress_PP is the existing grad_lnpsi_pbc_strain.
#[allow(clippy::too_many_arguments)]
fn gfmc_stress_ecp_pbc_jnone_route(
    gem: &GeminalData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    debug: bool,
) -> Array2<f64> {
    use crate::stress_pbc::pgto_kstress_side_screened;
    let n_up = r_up.nrows();
    let cell = &ew.cell;

    // det kinetic stress via pgto_kstress_side (geminal_reverse_w cotangents; w=None => full)
    let kin_stress = |w: Option<&[f64]>| -> Array2<f64> {
        let zju = Array2::<f64>::zeros((n_up, 3));
        let zjd = Array2::<f64>::zeros((r_dn.nrows(), 3));
        let (wu, wd) = match w {
            None => (None, None),
            Some(w) => (Some(w[0..n_up].to_vec()), Some(w[n_up..].to_vec())),
        };
        let b = geminal_reverse_w(
            gem,
            r_up,
            r_dn,
            &zju,
            &zjd,
            wu.as_deref(),
            wd.as_deref(),
            None,
        );
        let mut o = Array2::<f64>::zeros((3, 3));
        for (aos, c, home, ker) in leaves_up {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_up,
                &b.bar_lap_up,
                None,
                r_up,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_up, DerivScreens::FAST),
                &mut o,
            );
        }
        for (aos, c, home, ker) in leaves_dn {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_dn,
                &b.bar_lap_dn,
                None,
                r_dn,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_dn, DerivScreens::FAST),
                &mut o,
            );
        }
        // PW kinetic stress (composite pgto(+)pw determinant): lap only (jnone has
        // no gou cross; PW det value Pulay is strain-invariant -> sigma_PP needs none).
        let pw_kin =
            |orb: &OrbData, bar_lap: &Array2<f64>, r: &ArrayView2<f64>, o: &mut Array2<f64>| {
                if let Some(pw) = &orb.pw {
                    let c_pw_owned = orb.pw_coeff_block();
                    let c_pw = c_pw_owned.view();
                    crate::stress_pbc::pw_lap_stress(pw, &c_pw.t().dot(bar_lap), r, o);
                }
            };
        pw_kin(&gem.orb_up, &b.bar_lap_up, r_up, &mut o);
        pw_kin(&gem.orb_dn, &b.bar_lap_dn, r_dn, &mut o);
        // B-spline kinetic stress (pure B-spline or pgto(+)B-spline composite).
        // Its AO value is invariant at fixed fractional coordinates, so there is
        // no determinant-value Pulay term; the Laplacian metric response is the
        // complete jnone contribution.  `bspline_kstress_side` selects only the
        // B-spline MO block from the full composite-rank reverse cotangent.
        if gem.orb_up.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(&gem.orb_up, &b.bar_lap_up, None, r_up, &mut o);
        }
        if gem.orb_dn.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(&gem.orb_dn, &b.bar_lap_dn, None, r_dn, &mut o);
        }
        o
    };

    // ---- continuum ----
    let mut sigma = kin_stress(None);
    sigma += &crate::forces_ewald::grad_eee_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eei_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eii_strain(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    sigma += &crate::forces_ecp::grad_vlocal_strain(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell, ew.n_max,
    );
    // The quadrature mesh is defined in its reference orientation and rotated by
    // the per-measurement RT before both the value and derivative evaluations.
    // Passing the raw mesh here agrees only for RT=I and leaves the jnone stress
    // inconsistent with the production ECP value for a general rotation.
    let mut rotated_grid_points = Array2::<f64>::zeros(grid_points.raw_dim());
    for v in 0..grid_points.nrows() {
        for c in 0..3 {
            for k in 0..3 {
                rotated_grid_points[[v, c]] += grid_points[[v, k]] * rt[k][c];
            }
        }
    }
    sigma += &crate::stress_pbc::grad_v_ecp_nonlocal_strain_gfmc(
        gem,
        leaves_up,
        leaves_dn,
        positions,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        &rotated_grid_points.view(),
        r_up,
        r_dn,
        cell,
        ew.n_max,
        nn,
    );
    let _ = (alat, debug);
    sigma
}

#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_jnone(
    gem: &GeminalData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_jnone_route(
        gem,
        leaves_up,
        leaves_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        false,
    )
}

/// Frozen full-recompute S' oracle for [`gfmc_stress_ecp_pbc_jnone`].
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_jnone_debug(
    gem: &GeminalData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_jnone_route(
        gem,
        leaves_up,
        leaves_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        true,
    )
}

/// `de_L^t/deps (3x3)` for the PBC (Ewald, Gamma) ECP **withJ (J1/J2)** LRDMC diagonal
/// e_L (tmove/dltmove). Generalises [`gfmc_stress_ecp_pbc_jnone`]: the continuum
/// kinetic becomes the full withJ periodic kinetic stress (det gou Hessian + J1/J2
/// smooth-distance kinetic strain), the ECP non-local mesh ratio carries `exp(DeltaJ)`
/// with full (det+J) mesh Pulay/drift, and the strained-mesh sigma_PP + w-masked kinetic
/// pick up the Jastrow. J3/DTN are guarded at the caller (no periodic J3 value-strain).
#[allow(clippy::too_many_arguments)]
fn gfmc_stress_ecp_pbc_withj_route(
    gem: &GeminalData,
    jas: &JastrowData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    j3: Option<&J3StressPbc>,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
    debug: bool,
) -> Array2<f64> {
    use crate::stress_pbc::{
        dtn_value_strain, j3_pbc_value_strain, jastrow_strain, pgto_det_stress_side_screened,
        pgto_kstress_side_screened,
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let cell = &ew.cell;
    let inv = inv3(cell);
    let nmax = ew.n_max;
    let (j3o, j3m) = match (j3, batch_j3) {
        (Some(c), _) => (Some(c.orb), Some(c.jm.view())),
        (None, Some((orb, jm))) => (Some(orb), Some(jm.view())),
        (None, None) => (None, None),
    };
    let j3ke = j3o.zip(j3m.as_ref());
    // Pure grid-orbital J3 is affine-value invariant, but its kinetic metric
    // response and full mesh-ratio drift remain live under strain.

    // full ln|Psi| strain-Pulay (det + J1/J2 cell-strain) at a config
    let spp_full = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> Array2<f64> {
        let f = gem_forward(gem, ru, rd);
        let bu = f.wg.clone();
        let bd = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
        let mut o = Array2::<f64>::zeros((3, 3));
        for (aos, c, home, ker) in leaves_up {
            pgto_det_stress_side_screened(
                aos,
                c,
                &bu,
                ru,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_up, DerivScreens::FAST),
                &mut o,
            );
        }
        for (aos, c, home, ker) in leaves_dn {
            pgto_det_stress_side_screened(
                aos,
                c,
                &bd,
                rd,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_dn, DerivScreens::FAST),
                &mut o,
            );
        }
        jastrow_strain(j1, j2, ru, rd, &mut o);
        if let Some(d) = dtn {
            o += &dtn_value_strain(d, ru, rd);
        }
        if let Some(c) = j3 {
            j3_pbc_value_strain(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                ru,
                rd,
                cell,
                c.home,
                c.kernel,
                positions,
                c.pw,
                c.gto_first,
                &mut o,
            );
        }
        o
    };
    // det-only strain-Pulay (dltmove SP branch)
    let spp_det = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> Array2<f64> {
        let f = gem_forward(gem, ru, rd);
        let bu = f.wg.clone();
        let bd = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
        let mut o = Array2::<f64>::zeros((3, 3));
        for (aos, c, home, ker) in leaves_up {
            pgto_det_stress_side_screened(
                aos,
                c,
                &bu,
                ru,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_up, DerivScreens::FAST),
                &mut o,
            );
        }
        for (aos, c, home, ker) in leaves_dn {
            pgto_det_stress_side_screened(
                aos,
                c,
                &bd,
                rd,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_dn, DerivScreens::FAST),
                &mut o,
            );
        }
        o
    };
    // full drift (det + J1/J2/J3/DTN) per electron at a config
    let drift_full = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> (Array2<f64>, Array2<f64>) {
        let (gd_up, gd_dn) = det_grad_field(gem, ru, rd);
        let (gj_up, gj_dn) = jastrow_grads(Some(jas), j3o, j3m.as_ref(), ru, rd);
        (gd_up + &gj_up, gd_dn + &gj_dn)
    };

    // full withJ periodic kinetic stress, per-electron weighted (w=None => all)
    // One base geminal forward (fresh LU; the FAST e_L tables when the e_L screens) for the
    // kinetic reverse, the det drift and the ECP mesh hops.
    let fwd = gem_forward_pbc(gem, r_up, r_dn, None);
    let kin_stress = |w: Option<&[f64]>| -> Array2<f64> {
        let (gj_up, gj_dn) = jastrow_grads(Some(jas), j3o, j3m.as_ref(), r_up, r_dn);
        let (wu, wd) = match w {
            None => (None, None),
            Some(w) => (Some(w[0..n_up].to_vec()), Some(w[n_up..].to_vec())),
        };
        let b = crate::param_grad_el::geminal_reverse_from_fwd(
            &fwd,
            &gj_up,
            &gj_dn,
            wu.as_deref(),
            wd.as_deref(),
        );
        let mut o = Array2::<f64>::zeros((3, 3));
        for (aos, c, home, ker) in leaves_up {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_up,
                &b.bar_lap_up,
                Some(&b.bar_gou),
                r_up,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_up, DerivScreens::FAST),
                &mut o,
            );
        }
        for (aos, c, home, ker) in leaves_dn {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_dn,
                &b.bar_lap_dn,
                Some(&b.bar_godn),
                r_dn,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_dn, DerivScreens::FAST),
                &mut o,
            );
        }
        // PW kinetic stress for a composite (pgto(+)pw) determinant: lap (|G|^2) +
        // gou (kappa<->kappa). The PW det VALUE Pulay is strain-invariant (frac-invariant),
        // so sigma_PP needs no PW -- only the kinetic does. Mirrors py_grad_t_pbc_strain.
        let pw_kin = |orb: &OrbData,
                      bar_lap: &Array2<f64>,
                      bar_gou: &[Array2<f64>; 3],
                      r: &ArrayView2<f64>,
                      o: &mut Array2<f64>| {
            if let Some(pw) = &orb.pw {
                let c_pw_owned = orb.pw_coeff_block();
                let c_pw = c_pw_owned.view();
                let cm_lap_pw = c_pw.t().dot(bar_lap);
                crate::stress_pbc::pw_lap_stress(pw, &cm_lap_pw, r, o);
                let cm_gou_pw = [
                    c_pw.t().dot(&bar_gou[0]),
                    c_pw.t().dot(&bar_gou[1]),
                    c_pw.t().dot(&bar_gou[2]),
                ];
                crate::stress_pbc::pw_gou_stress(pw, &cm_gou_pw, r, o);
            }
        };
        pw_kin(&gem.orb_up, &b.bar_lap_up, &b.bar_gou, r_up, &mut o);
        pw_kin(&gem.orb_dn, &b.bar_lap_dn, &b.bar_godn, r_dn, &mut o);
        // B-spline kinetic stress (pure B-spline or pgto(+)B-spline composite):
        // Laplacian metric response plus the determinant/Jastrow drift cross.
        // The helper slices the B-spline MO block out of the full composite
        // cotangent, so this is additive with the pGTO side above, not a fallback.
        if gem.orb_up.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(
                &gem.orb_up,
                &b.bar_lap_up,
                Some(&b.bar_gou),
                r_up,
                &mut o,
            );
        }
        if gem.orb_dn.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(
                &gem.orb_dn,
                &b.bar_lap_dn,
                Some(&b.bar_godn),
                r_dn,
                &mut o,
            );
        }
        // J1/J2/DTN cell-straining kinetic strain:
        // cot = gJ_e + gD_e (full drift, UNWEIGHTED);
        // the per-electron S' mask `w` is applied INSIDE the *_w kernels (weighting
        // the shared J2/u laplacian by (w_i+w_j)/2 -- cot-weighting alone would miss it).
        let (gd_up, gd_dn) = crate::param_grad_el::det_grad_field_from_fwd(&fwd);
        let mut cot = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                let (gj, gd) = if e < n_up {
                    (gj_up[[e, c]], gd_up[[e, c]])
                } else {
                    (gj_dn[[e - n_up, c]], gd_dn[[e - n_up, c]])
                };
                cot[[e, c]] = gj + gd;
            }
        }
        crate::stress_pbc::jastrow_kstress_w(j1, j2, &cot, r_up, r_dn, w, &mut o);
        if let Some(d) = dtn {
            crate::stress_pbc::dtn_kstress_w(d, &cot, r_up, r_dn, w, &mut o);
        }
        if let Some(c) = j3 {
            let bg_up = -&cot.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = -&cot.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            crate::stress_pbc::j3_pbc_kstress_w(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                r_up,
                r_dn,
                &bg_up,
                &bg_dn,
                w,
                cell,
                c.home,
                c.kernel,
                positions,
                c.pw,
                c.gto_first,
                &mut o,
            );
        }
        if let Some((orb, jm)) = batch_j3 {
            let bg_up = -&cot.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = -&cot.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            j3_leaf_kstress_side(orb, jm, r_up, r_dn, &bg_up, &bg_dn, w, &mut o);
        }
        o
    };

    // ---- continuum ----
    let mut sigma = crate::timed!("gs.kin", kin_stress(None));
    sigma += &crate::forces_ewald::grad_eee_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        nmax,
    );
    sigma += &crate::forces_ewald::grad_eei_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eii_strain(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    sigma += &crate::forces_ecp::grad_vlocal_strain(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell, nmax,
    );

    // ---- ECP non-local stress (withJ mesh ratio; fixed quadrature directions) ----
    if !debug {
        // Production: the shared rank-1 mesh (fresh base LU, as below), the VMC stress core
        // with this interval's rotated quadrature and the dltmove split.
        let gp: Vec<[f64; 3]> = (0..weights.len())
            .map(|v| {
                let mut out = [0.0_f64; 3];
                for c in 0..3 {
                    for k in 0..3 {
                        out[c] += grid_points[[v, k]] * rt[k][c];
                    }
                }
                out
            })
            .collect();
        sigma += &crate::timed!(
            "gs.ecp_nl",
            ecp_nonlocal_stress_mesh_pbc(
                gem,
                jas,
                j3,
                leaves_up,
                leaves_dn,
                positions,
                r_up,
                r_dn,
                cell,
                nmax,
                nuc_n,
                am_n,
                e_n,
                c_n,
                p_n,
                weights,
                &gp,
                nn,
                dltmove,
                None,
                true,
                batch_j3,
                Some(&fwd),
            )
        );
        let _ = alat;
        return sigma;
    }
    // Frozen full-recompute oracle (`gfmc_stress_ecp_pbc_withj_debug`).
    let jd0 = crate::coulomb_potential::jastrow_value_for_ratio(jas, j3ke, r_up, r_dn);
    let (s0, l0) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    let base_pulay_f = spp_full(r_up, r_dn);
    let base_pulay_d = spp_det(r_up, r_dn);
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    &inv,
                    nmax,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let nv = weights.len();
    let gp: Vec<[f64; 3]> = (0..nv)
        .map(|v| {
            let mut out = [0.0_f64; 3];
            for c in 0..3 {
                for k in 0..3 {
                    out[c] += grid_points[[v, k]] * rt[k][c];
                }
            }
            out
        })
        .collect();
    for gi in 0..n_e {
        let is_up = gi < n_up;
        let re = if is_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        };
        for &ia in &nearest_set(re) {
            let rel0 = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let rel = min_image(rel0, cell, &inv, nmax);
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            for v in 0..nv {
                let vg = gp[v];
                let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                // accumulate ECP angular factors over l-terms on atom ia
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (pe, be, ce) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-be * nrm * nrm).exp();
                    let vl = ce * nrm.powf(pe - 2.0) * expf;
                    let vlp = ce
                        * ((pe - 2.0) * nrm.powf(pe - 3.0) - 2.0 * be * nrm.powf(pe - 1.0))
                        * expf;
                    let (pl, dpl) = leg_and_deriv(l, cth);
                    let wv = ((2 * l + 1) as f64) * weights[v];
                    ang += vl * wv * pl;
                    aprime += vlp * wv * pl;
                    acos += vl * wv * dpl;
                }
                let rp = [
                    re[0] + rel[0] + nrm * vg[0],
                    re[1] + rel[1] + nrm * vg[1],
                    re[2] + rel[2] + nrm * vg[2],
                ];
                let (mu, md) = {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if is_up {
                        mu[[gi, 0]] = rp[0];
                        mu[[gi, 1]] = rp[1];
                        mu[[gi, 2]] = rp[2];
                    } else {
                        let k = gi - n_up;
                        md[[k, 0]] = rp[0];
                        md[[k, 1]] = rp[1];
                        md[[k, 2]] = rp[2];
                    }
                    (mu, md)
                };
                let (sm, lm) = det_sign_lnabs(
                    &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                );
                let det_ratio = sm * s0 * (lm - l0).exp();
                let det_amp = ang * det_ratio;
                let use_full = !dltmove || det_amp < 0.0;
                let jr = (crate::coulomb_potential::jastrow_value_for_ratio(
                    jas,
                    j3ke,
                    &mu.view(),
                    &md.view(),
                ) - jd0)
                    .exp();
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                // mesh Pulay + drift matching the ratio's Jastrow inclusion
                let (pulay_m, dmu, dmd, pulay_base) = if use_full {
                    let (du, dd) = drift_full(&mu.view(), &md.view());
                    (spp_full(&mu.view(), &md.view()), du, dd, &base_pulay_f)
                } else {
                    let (du, dd) = det_grad_field(gem, &mu.view(), &md.view());
                    (spp_det(&mu.view(), &md.view()), du, dd, &base_pulay_d)
                };
                let drift = if is_up {
                    [dmu[[gi, 0]], dmu[[gi, 1]], dmu[[gi, 2]]]
                } else {
                    let k = gi - n_up;
                    [dmd[[k, 0]], dmd[[k, 1]], dmd[[k, 2]]]
                };
                let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                for c in 0..3 {
                    for d in 0..3 {
                        let dcth =
                            -(vg[c] * rel[d]) / (nrm * vgn) - cth * rel[c] * rel[d] / (nrm * nrm);
                        let da = aprime * rel[c] * rel[d] / nrm + acos * dcth;
                        let mesh_wf = pulay_m[[c, d]] - drift[c] * nrm * vg[d]
                            + (rel[c] * rel[d] / nrm) * drift_vg;
                        let dln_ratio = mesh_wf - pulay_base[[c, d]];
                        sigma[[c, d]] += ratio * (da + ang * dln_ratio);
                    }
                }
            }
        }
    }

    let _ = (alat, debug);
    sigma
}

#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_withj(
    gem: &GeminalData,
    jas: &JastrowData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    j3: Option<&J3StressPbc>,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_withj_route(
        gem,
        jas,
        leaves_up,
        leaves_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        j1,
        j2,
        dtn,
        j3,
        batch_j3,
        false,
    )
}

/// Frozen full-recompute S' oracle for [`gfmc_stress_ecp_pbc_withj`].
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_withj_debug(
    gem: &GeminalData,
    jas: &JastrowData,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    dtn: Option<&crate::dtn::JastrowDtnData>,
    j3: Option<&J3StressPbc>,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_withj_route(
        gem,
        jas,
        leaves_up,
        leaves_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        j1,
        j2,
        dtn,
        j3,
        batch_j3,
        true,
    )
}

// ============================================================================
// VMC (MCMC) periodic real-Gamma stress. The
// continuous VMC de_L/deps (HF stress) / dln|Psi|/deps (Pulay stress) the driver's
// `_grad_e_L_eps` / `_grad_ln_psi_eps` emit: EXACTLY the `gfmc_stress_ecp_pbc_withj`
// SMOOTH BASE (withJ periodic kinetic stress + Ewald ee/ei/ii strain + ECP local
// strain + ECP non-local mesh stress with the quadrature DIRECTIONS HELD FIXED) WITHOUT the
// LRDMC S' discretization. No RT-strain "grid-stretch" term `dV/dvg_c * vg_d` is
// added: it would move the strained quadrature points off the sphere of the
// strained electron-nucleus distance and bias the non-local ECP stress, while the
// rotation average of the fixed-direction derivative is the exact strain derivative of
// the angular integral. The GFMC and VMC cores both
// match jqmc's `_apply_strain`: every periodic Jastrow metric, including J2
// (e-e), deforms with the physical lattice. The sweep SM `g_inv`
// is reused for the base-config det kinetic + Pulay + drift
// (near-singular-stable, no fresh LU, as in the e_L).
// ============================================================================

/// VMC periodic Pulay stress `dln|Psi|/deps` (3x3): det Pulay stress (reuse `g_inv`) +
/// J1/J2/DTN/J3 cell-strain. The PW/B-spline det
/// VALUE Pulay is affine-invariant (frac-invariant) so it contributes nothing (only pgto
/// leaves are in `leaves_*`). 1:1 the production `pbc_stress.stress_lnPsi_pbc`.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_ln_psi_eps_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3StressPbc>,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    use crate::stress_pbc::{
        dtn_value_strain, j3_pbc_value_strain, jastrow_strain, pgto_det_stress_side_screened,
    };
    let (bu, bd) = gem_value_bars_pbc(gem, r_up, r_dn, g_inv);
    let mut o = Array2::<f64>::zeros((3, 3));
    for (aos, c, home, ker) in leaves_up {
        pgto_det_stress_side_screened(
            aos,
            c,
            &bu,
            r_up,
            cell,
            home,
            ker,
            positions,
            fast_screens(&gem.orb_up, DerivScreens::FAST),
            &mut o,
        );
    }
    for (aos, c, home, ker) in leaves_dn {
        pgto_det_stress_side_screened(
            aos,
            c,
            &bd,
            r_dn,
            cell,
            home,
            ker,
            positions,
            fast_screens(&gem.orb_dn, DerivScreens::FAST),
            &mut o,
        );
    }
    jastrow_strain(
        jas.one_body.as_ref(),
        jas.two_body.as_ref(),
        r_up,
        r_dn,
        &mut o,
    );
    if let Some(d) = jas.dtn.as_ref() {
        o += &dtn_value_strain(d, r_up, r_dn);
    }
    if let Some(c) = j3 {
        j3_pbc_value_strain(
            c.cart_j3,
            &c.t_mol.to_owned(),
            &c.jm,
            r_up,
            r_dn,
            cell,
            c.home,
            c.kernel,
            positions,
            c.pw,
            c.gto_first,
            &mut o,
        );
    }
    o
}

/// VMC periodic HF stress `de_L/deps` (3x3): withJ periodic kinetic stress + Ewald +
/// ECP local + ECP non-local mesh stress (fixed quadrature directions), NO S'. `ecp=false` =>
/// bare-Ewald AE. Reuses the sweep SM `g_inv` for the BASE config (kinetic + Pulay/drift);
/// the ECP-nl reflected mesh configs use a fresh LU (a different, non-near-singular config,
/// as the GFMC core does). J1/J2/J3/DTN cells strain.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_e_l_eps_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3StressPbc>,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    g_inv: Option<&ArrayView2<f64>>,
    fast_mesh: bool,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
) -> Array2<f64> {
    use crate::stress_pbc::{jastrow_kstress_w, pgto_kstress_side_screened};
    let cell = &ew.cell;
    let nmax = ew.n_max;
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    // The J3 drift rides the FULL drift (kinetic cotangents) so its cross terms with the
    // J1/J2/det kinetic strain are captured -- cart J3 via `c.orb`, PURE PW/B-spline via `batch_j3`
    // (its J3 kinetic SELF-strain is served by `j3_leaf_kstress_side` inside `kin_stress`, not the
    // cart `j3_pbc_kstress_w`).
    let (j3o, j3m) = match (j3, batch_j3) {
        (Some(c), _) => (Some(c.orb), Some(c.jm.to_owned())),
        (None, Some((o, mm))) => (Some(o), Some(mm.to_owned())),
        (None, None) => (None, None),
    };
    let j3m_v = j3m.as_ref().map(|m| m.view());
    let j1 = jas.one_body.as_ref();
    let j2 = jas.two_body.as_ref();
    let dtn = jas.dtn.as_ref();

    // One base geminal forward for the kinetic reverse, the det drift and the ECP mesh hops.
    let fwd = gem_forward_pbc(gem, r_up, r_dn, g_inv);
    // full withJ periodic kinetic stress (base config only; reuse g_inv).
    let kin_stress = || -> Array2<f64> {
        let (gj_up, gj_dn) = jastrow_grads(Some(jas), j3o, j3m_v.as_ref(), r_up, r_dn);
        let b = crate::param_grad_el::geminal_reverse_from_fwd(&fwd, &gj_up, &gj_dn, None, None);
        let mut o = Array2::<f64>::zeros((3, 3));
        for (aos, c, home, ker) in leaves_up {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_up,
                &b.bar_lap_up,
                Some(&b.bar_gou),
                r_up,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_up, DerivScreens::FAST),
                &mut o,
            );
        }
        for (aos, c, home, ker) in leaves_dn {
            pgto_kstress_side_screened(
                aos,
                c,
                &b.bar_orb_dn,
                &b.bar_lap_dn,
                Some(&b.bar_godn),
                r_dn,
                cell,
                home,
                ker,
                positions,
                fast_screens(&gem.orb_dn, DerivScreens::FAST),
                &mut o,
            );
        }
        // PW kinetic stress for a composite (pgto(+)pw) determinant.
        let pw_kin = |orb: &OrbData,
                      bar_lap: &Array2<f64>,
                      bar_gou: &[Array2<f64>; 3],
                      r: &ArrayView2<f64>,
                      o: &mut Array2<f64>| {
            if let Some(pw) = &orb.pw {
                let c_pw_owned = orb.pw_coeff_block();
                let c_pw = c_pw_owned.view();
                let cm_lap_pw = c_pw.t().dot(bar_lap);
                crate::stress_pbc::pw_lap_stress(pw, &cm_lap_pw, r, o);
                let cm_gou_pw = [
                    c_pw.t().dot(&bar_gou[0]),
                    c_pw.t().dot(&bar_gou[1]),
                    c_pw.t().dot(&bar_gou[2]),
                ];
                crate::stress_pbc::pw_gou_stress(pw, &cm_gou_pw, r, o);
            }
        };
        pw_kin(&gem.orb_up, &b.bar_lap_up, &b.bar_gou, r_up, &mut o);
        pw_kin(&gem.orb_dn, &b.bar_lap_dn, &b.bar_godn, r_dn, &mut o);
        // B-spline det kinetic stress (pure-B-spline determinant): the affine-invariant metric
        // response, sharing the SAME g_inv-reused kinetic reverse cotangents as the pgto/pw sides
        // (`bar_lap`/`bar_gou`). The B-spline VALUE Pulay is frac-invariant (0), so nothing is
        // added to the Pulay stress; only this kinetic piece survives.
        if gem.orb_up.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(
                &gem.orb_up,
                &b.bar_lap_up,
                Some(&b.bar_gou),
                r_up,
                &mut o,
            );
        }
        if gem.orb_dn.bspline.is_some() {
            crate::stress_pbc::bspline_kstress_side(
                &gem.orb_dn,
                &b.bar_lap_dn,
                Some(&b.bar_godn),
                r_dn,
                &mut o,
            );
        }
        // J1/J2/DTN kinetic strain (cell-straining).
        let (gd_up, gd_dn) = crate::param_grad_el::det_grad_field_from_fwd(&fwd);
        let mut cot = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                let (gj, gd) = if e < n_up {
                    (gj_up[[e, c]], gd_up[[e, c]])
                } else {
                    (gj_dn[[e - n_up, c]], gd_dn[[e - n_up, c]])
                };
                cot[[e, c]] = gj + gd;
            }
        }
        jastrow_kstress_w(j1, j2, &cot, r_up, r_dn, None, &mut o);
        if let Some(d) = dtn {
            crate::stress_pbc::dtn_kstress_w(d, &cot, r_up, r_dn, None, &mut o);
        }
        if let Some(c) = j3 {
            let bg_up = -&cot.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = -&cot.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            crate::stress_pbc::j3_pbc_kstress_w(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                r_up,
                r_dn,
                &bg_up,
                &bg_dn,
                None,
                cell,
                c.home,
                c.kernel,
                positions,
                c.pw,
                c.gto_first,
                &mut o,
            );
        }
        // PURE PW/B-spline J3 kinetic SELF-strain (the cart `j3_pbc_kstress_w` above is inactive
        // for a batch leaf since `j3 == None`); `bg = -(gJ+gD) = -cot`.
        if let Some((orb, jm)) = batch_j3 {
            let bg_up = -&cot.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
            let bg_dn = -&cot.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
            j3_leaf_kstress_side(orb, jm, r_up, r_dn, &bg_up, &bg_dn, None, &mut o);
        }
        o
    };

    // ---- continuum (NO S') ----
    let mut sigma = crate::timed!("s.kin", kin_stress());
    sigma += &crate::forces_ewald::grad_eee_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        nmax,
    );
    sigma += &crate::forces_ewald::grad_eei_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eii_strain(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    if !ecp {
        return sigma;
    }
    sigma += &crate::forces_ecp::grad_vlocal_strain(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell, nmax,
    );

    // ---- ECP non-local stress (withJ mesh ratio, fixed quadrature directions; tmove full ratio) ----
    let grid: Vec<[f64; 3]> = (0..weights.len())
        .map(|v| {
            [
                grid_points[[v, 0]],
                grid_points[[v, 1]],
                grid_points[[v, 2]],
            ]
        })
        .collect();
    sigma += &ecp_nonlocal_stress_mesh_pbc(
        gem,
        jas,
        j3,
        leaves_up,
        leaves_dn,
        positions,
        r_up,
        r_dn,
        cell,
        nmax,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        &grid,
        nn,
        false,
        g_inv,
        fast_mesh,
        batch_j3,
        Some(&fwd),
    );
    sigma
}

/// Periodic real-Gamma ECP non-local mesh STRESS `dV_nl/deps` (3x3) with Jastrow at one
/// configuration -- the mesh part shared by the VMC (`vmc_grad_e_l_eps_pbc`) and GFMC
/// (`gfmc_stress_ecp_pbc_withj`) HF stresses. `grid` = the quadrature directions (already
/// rotated by the caller's `rt`), held fixed under the strain. `dltmove`: split each candidate
/// on its det-only amplitude (SP: det-only ratio + det-only mesh fields; FN: full), else the
/// tmove full ratio. `g_inv` = the base-config inverse to reuse (`None` = fresh LU). The mesh
/// candidates are single-electron rank-1 hops off the cached base fields
/// ([`MeshDetStressPbc`], [`MeshJasStressPbc`]); `fast_mesh = false` keeps the per-candidate
/// full recompute (the comparison oracle).
#[allow(clippy::too_many_arguments)]
fn ecp_nonlocal_stress_mesh_pbc(
    gem: &GeminalData,
    jas: &JastrowData,
    j3: Option<&J3StressPbc>,
    leaves_up: &[crate::stress_pbc::StressLeaf],
    leaves_dn: &[crate::stress_pbc::StressLeaf],
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    nmax: i32,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid: &[[f64; 3]],
    nn: usize,
    dltmove: bool,
    g_inv: Option<&ArrayView2<f64>>,
    fast_mesh: bool,
    batch_j3: Option<(&OrbData, &ArrayView2<f64>)>,
    // The base geminal forward at `g_inv` when the caller already has it.
    fwd: Option<&crate::param_grad_el::GemFwd>,
) -> Array2<f64> {
    use crate::stress_pbc::{
        dtn_value_strain, j3_pbc_value_strain, jastrow_strain, pgto_det_stress_side_screened,
    };
    let inv = inv3(cell);
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let (j3o, j3m) = match (j3, batch_j3) {
        (Some(c), _) => (Some(c.orb), Some(c.jm.to_owned())),
        (None, Some((o, mm))) => (Some(o), Some(mm.to_owned())),
        (None, None) => (None, None),
    };
    let j3m_v = j3m.as_ref().map(|m| m.view());
    // The reflected-config VALUE ratio's J3 (`jd0` + per-candidate `jr`) comes from EITHER a
    // cart-lowerable `j3` OR a PURE PW/B-spline `batch_j3` leaf; both feed `jastrow_value_for_ratio`
    // identically as `(&OrbData, &j_matrix)`. (The batch leaf's VALUE STRAIN-Pulay is 0,
    // affine-invariant, so `spp_full` correctly excludes it -- only the value RATIO carries it.)
    let j3val_ref = j3m_v.as_ref().map(|m| (j3o.unwrap(), m));
    let j1 = jas.one_body.as_ref();
    let j2 = jas.two_body.as_ref();
    let dtn = jas.dtn.as_ref();
    // det-only strain-Pulay at a config; `gi` reuses the base inverse (None at mesh configs).
    let spp_det =
        |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>, gi: Option<&ArrayView2<f64>>| -> Array2<f64> {
            let f = gem_forward_gi(gem, ru, rd, gi);
            let bu = f.wg.clone();
            let bd = f.gq.slice(numpy::ndarray::s![0..f.n_dn, ..]).t().to_owned();
            let mut o = Array2::<f64>::zeros((3, 3));
            for (aos, c, home, ker) in leaves_up {
                pgto_det_stress_side_screened(
                    aos,
                    c,
                    &bu,
                    ru,
                    cell,
                    home,
                    ker,
                    positions,
                    fast_screens(&gem.orb_up, DerivScreens::FAST),
                    &mut o,
                );
            }
            for (aos, c, home, ker) in leaves_dn {
                pgto_det_stress_side_screened(
                    aos,
                    c,
                    &bd,
                    rd,
                    cell,
                    home,
                    ker,
                    positions,
                    fast_screens(&gem.orb_dn, DerivScreens::FAST),
                    &mut o,
                );
            }
            o
        };
    // Jastrow (J1/J2/DTN/J3) value strain at a config, accumulated into `o`.
    let jas_strain = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>, o: &mut Array2<f64>| {
        jastrow_strain(j1, j2, ru, rd, o);
        if let Some(d) = dtn {
            *o += &dtn_value_strain(d, ru, rd);
        }
        if let Some(c) = j3 {
            j3_pbc_value_strain(
                c.cart_j3,
                &c.t_mol.to_owned(),
                &c.jm,
                ru,
                rd,
                cell,
                c.home,
                c.kernel,
                positions,
                c.pw,
                c.gto_first,
                o,
            );
        }
    };
    // full ln|Psi| strain-Pulay (det + J1/J2/DTN/J3 cell-strain) at a config.
    let spp_full =
        |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>, gi: Option<&ArrayView2<f64>>| -> Array2<f64> {
            let mut o = spp_det(ru, rd, gi);
            jas_strain(ru, rd, &mut o);
            o
        };
    let jas_value = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| -> f64 {
        crate::coulomb_potential::jastrow_value_for_ratio(jas, j3val_ref, ru, rd)
    };

    // Cache base geminal fields; each mesh candidate is a single-electron rank-1 hop
    // (SM det ratio + rank-1 det strain Pulay + MO-level moved-electron drift) instead of three
    // fresh O(N^3) forwards. The determinant strain Pulay itself becomes base + delta
    // (cached base AO strain tables; only the moved electron's column is re-evaluated), instead
    // of a full per-candidate periodic AO-gradient sweep. `fast_mesh=false` keeps the
    // full-recompute path (the comparison oracle). Covers pgto / composite / pure-B-spline.
    let mesh_det_stress = if fast_mesh {
        MeshDetStressPbc::new(
            gem, leaves_up, leaves_dn, cell, positions, r_up, r_dn, g_inv, fwd,
        )
    } else {
        None
    };
    // Cache base Jastrow strain/drift/value fields; each candidate's
    // periodic Jastrow value ratio + moved-electron J3 drift + J1/J2/J3 value strain becomes a
    // one-electron delta instead of a full `jastrow_value_for_ratio` + `jastrow_grads` +
    // `j3_pbc_value_strain` recompute. `None` (MO-J3 / composite-PW / B-spline J3) or
    // `JQMC_RS_ECPF_SLOW=1` -> the full-recompute reference path below. Gated on `fast_mesh`
    // (the `fast_mesh=false` A/B oracle keeps the full recompute for both det AND Jastrow).
    // Long-lived [`J3Pbc`] bridge (drops the stress-only `kernel`) so the shared value/pGTO
    // rank-1 state inside `MeshJasStressPbc` can borrow it for the whole mesh loop.
    let j3_force_bridge = j3.map(|c| J3Pbc {
        orb: c.orb,
        cart_j3: c.cart_j3,
        t_mol: c.t_mol,
        jm: c.jm,
        home: c.home,
        pw: c.pw,
        gto_first: c.gto_first,
    });
    // A pure PW/B-spline `batch_j3` leaf (`j3 == None`) cannot use the one-electron Jastrow path:
    // `MeshJasStressPbc::new` is passed the cart `j3` (None here) and would silently drop the batch
    // J3 from its candidate value ratio / drift delta / value-strain delta (exposed whenever J1/J2/DTN
    // are also present, which would otherwise keep the cached path active). Force the full-recompute reference
    // path -- which threads the batch J3 through `j3val_ref` / the drift -- for any batch leaf.
    let mesh_jas_stress = if fast_mesh
        && batch_j3.is_none()
        && (j1.is_some() || j2.is_some() || dtn.is_some() || j3.is_some())
        && std::env::var("JQMC_RS_ECPF_SLOW").is_err()
    {
        MeshJasStressPbc::new(
            jas,
            j3_force_bridge.as_ref(),
            j3,
            cell,
            positions,
            r_up,
            r_dn,
            positions.nrows(),
        )
    } else {
        None
    };
    // Fused determinant mode: the base det strain cancels between the candidates and the
    // base Pulay, so only the Jastrow strain is evaluated at the base configuration.
    let det_fused = mesh_det_stress.as_ref().is_some_and(|m| m.fused.is_some());
    // The base value / log-det anchors are only read by the full-recompute fallbacks.
    let jd0 = if mesh_det_stress.is_none() || mesh_jas_stress.is_none() {
        jas_value(r_up, r_dn)
    } else {
        0.0
    };
    let (s0, l0) = if mesh_det_stress.is_none() {
        det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view())
    } else {
        (1.0, 0.0)
    };
    let base_pulay_f = if det_fused {
        let mut o = Array2::<f64>::zeros((3, 3));
        jas_strain(r_up, r_dn, &mut o);
        o
    } else {
        spp_full(r_up, r_dn, g_inv)
    };
    let base_pulay_d = if dltmove && !det_fused {
        spp_det(r_up, r_dn, g_inv)
    } else {
        Array2::<f64>::zeros((3, 3))
    };
    // Weighted delta-bar accumulators of the determinant strain Pulay (see
    // `MeshDetStressPbc::candidates`), folded in once after the loop.
    let mut det_acc = mesh_det_stress
        .as_ref()
        .filter(|_| !det_fused)
        .map(|m| m.delta_accumulators());
    let mut fused_acc = mesh_det_stress.as_ref().filter(|_| det_fused).map(|m| {
        (
            m.delta_accumulators(),
            (vec![0.0_f64; n_up], vec![0.0_f64; n_e - n_up]),
        )
    });
    let base_strain = mesh_det_stress
        .as_ref()
        .map_or_else(|| Array2::<f64>::zeros((3, 3)), |m| m.base_strain.clone());
    let mut sigma = Array2::<f64>::zeros((3, 3));
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    &inv,
                    nmax,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let nv = weights.len();
    for gi in 0..n_e {
        let is_up = gi < n_up;
        let re = if is_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        };
        for &ia in &nearest_set(re) {
            let rel0 = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let rel = min_image(rel0, cell, &inv, nmax);
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let li = if is_up { gi } else { gi - n_up };
            let new_positions: Vec<[f64; 3]> = (0..nv)
                .map(|v| {
                    let vg = grid[v];
                    [
                        re[0] + rel[0] + nrm * vg[0],
                        re[1] + rel[1] + nrm * vg[1],
                        re[2] + rel[2] + nrm * vg[2],
                    ]
                })
                .collect();
            // batched one-electron periodic Jastrow fields for all nv mesh points (value
            // ratio + moved-electron J3 drift + J1/J2/J3 value strain, base + delta).
            let mjs_res = mesh_jas_stress.as_ref().map(|m| {
                crate::timed!(
                    "eps.jas_fast",
                    m.stress_candidates(is_up, li, &new_positions, r_up, r_dn)
                )
            });
            // batched one-electron det ratio + drift + STRAIN Pulay for all nv mesh points
            // (base + delta; only the moved electron's AO strain tables are re-evaluated).
            let mut fgrp: Option<DetStressFusedGroup> = None;
            let det_grp = mesh_det_stress.as_ref().map(|m| {
                crate::timed!("eps.det_fast", {
                    match m.candidates_fused(is_up, li, &new_positions) {
                        Some((c, g)) => {
                            fgrp = Some(g);
                            let cands = c
                                .into_iter()
                                .map(|(ratio, drift)| DetStressCand {
                                    ratio,
                                    drift,
                                    bar_li: numpy::ndarray::Array1::zeros(0),
                                    d_up: Array2::zeros((0, 0)),
                                    d_dn: Array2::zeros((0, 0)),
                                })
                                .collect();
                            (cands, None)
                        }
                        None => {
                            let (c, g) = m.candidates(is_up, li, &new_positions);
                            (c, Some(g))
                        }
                    }
                })
            });
            let det_res = det_grp.as_ref().map(|(c, _)| c);
            let mut betas = vec![0.0_f64; nv];
            // weighted moved-bar columns of the group (the li-column table swap, see
            // `MeshDetStressPbc::group_swap_strain`).
            let mut g_new = det_res.map(|c| Array2::<f64>::zeros((c[0].bar_li.len(), nv)));
            for v in 0..nv {
                let vg = grid[v];
                let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (pe, be, ce) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-be * nrm * nrm).exp();
                    let vl = ce * nrm.powf(pe - 2.0) * expf;
                    let vlp = ce
                        * ((pe - 2.0) * nrm.powf(pe - 3.0) - 2.0 * be * nrm.powf(pe - 1.0))
                        * expf;
                    let (pl, dpl) = leg_and_deriv(l, cth);
                    let wv = ((2 * l + 1) as f64) * weights[v];
                    ang += vl * wv * pl;
                    aprime += vlp * wv * pl;
                    acos += vl * wv * dpl;
                }
                let rp = new_positions[v];
                let (mu, md) = {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if is_up {
                        mu[[gi, 0]] = rp[0];
                        mu[[gi, 1]] = rp[1];
                        mu[[gi, 2]] = rp[2];
                    } else {
                        let k = gi - n_up;
                        md[[k, 0]] = rp[0];
                        md[[k, 1]] = rp[1];
                        md[[k, 2]] = rp[2];
                    }
                    (mu, md)
                };
                let moved_row = |gu: &Array2<f64>, gd: &Array2<f64>| -> [f64; 3] {
                    let mut o = [0.0; 3];
                    for (cc, oc) in o.iter_mut().enumerate() {
                        *oc = if is_up {
                            gu[[gi, cc]]
                        } else {
                            gd[[gi - n_up, cc]]
                        };
                    }
                    o
                };
                // (ratio, mesh strain Pulay, moved-electron drift, base strain Pulay); the
                // fast path's delta-bar det strain is folded in after the loop (`det_acc`).
                let (ratio, pulay_m, drift, pulay_base) = match &det_res {
                    Some(dres) => {
                        let dc = &dres[v];
                        let det_amp = ang * dc.ratio;
                        let use_full = !dltmove || det_amp < 0.0;
                        let (ratio, pm, drift, base) = if !use_full {
                            // SP (dltmove): det-only ratio, strain Pulay and drift.
                            (dc.ratio, base_strain.clone(), dc.drift, &base_pulay_d)
                        } else {
                            let mut pm = base_strain.clone();
                            let mut drift = dc.drift;
                            // Jastrow value ratio + strain + moved-electron drift: the
                            // one-electron update (base + delta) when supported, otherwise the
                            // full periodic recompute.
                            let jr = if let Some((jr, jdrift, jstrain)) =
                                mjs_res.as_ref().map(|r| &r[v])
                            {
                                // Jastrow value strain and moved drift row: base + one-electron delta.
                                pm += jstrain;
                                for cc in 0..3 {
                                    drift[cc] += jdrift[cc];
                                }
                                *jr
                            } else {
                                jas_strain(&mu.view(), &md.view(), &mut pm);
                                let (gj_up, gj_dn) = jastrow_grads(
                                    Some(jas),
                                    j3o,
                                    j3m_v.as_ref(),
                                    &mu.view(),
                                    &md.view(),
                                );
                                let jrow = moved_row(&gj_up, &gj_dn);
                                for cc in 0..3 {
                                    drift[cc] += jrow[cc];
                                }
                                (jas_value(&mu.view(), &md.view()) - jd0).exp()
                            };
                            (dc.ratio * jr, pm, drift, &base_pulay_f)
                        };
                        // delta-bar part of the candidate det strain, weight ratio*ang.
                        betas[v] = ratio * ang / dc.ratio;
                        if let Some((acc_up, acc_dn)) = det_acc.as_mut() {
                            acc_up.scaled_add(ratio * ang, &dc.d_up);
                            acc_dn.scaled_add(ratio * ang, &dc.d_dn);
                        }
                        if let Some(g) = g_new.as_mut() {
                            g.column_mut(v).scaled_add(ratio * ang, &dc.bar_li);
                        }
                        (ratio, pm, drift, base)
                    }
                    None => {
                        // full-recompute (fresh inverse per candidate) -- the comparison oracle.
                        let (sm, lm) = det_sign_lnabs(
                            &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                        );
                        let det_ratio = sm * s0 * (lm - l0).exp();
                        let det_amp = ang * det_ratio;
                        let use_full = !dltmove || det_amp < 0.0;
                        if use_full {
                            let pulay_m = spp_full(&mu.view(), &md.view(), None);
                            let (gd_up, gd_dn) =
                                det_grad_field_gi(gem, &mu.view(), &md.view(), None);
                            let (gj_up, gj_dn) = jastrow_grads(
                                Some(jas),
                                j3o,
                                j3m_v.as_ref(),
                                &mu.view(),
                                &md.view(),
                            );
                            let drift = moved_row(&(gd_up + &gj_up), &(gd_dn + &gj_dn));
                            let jr = (jas_value(&mu.view(), &md.view()) - jd0).exp();
                            (det_ratio * jr, pulay_m, drift, &base_pulay_f)
                        } else {
                            let pulay_m = spp_det(&mu.view(), &md.view(), None);
                            let (gd_up, gd_dn) =
                                det_grad_field_gi(gem, &mu.view(), &md.view(), None);
                            (det_ratio, pulay_m, moved_row(&gd_up, &gd_dn), &base_pulay_d)
                        }
                    }
                };
                let drift_vg = drift[0] * vg[0] + drift[1] * vg[1] + drift[2] * vg[2];
                for c in 0..3 {
                    for d in 0..3 {
                        let dcth =
                            -(vg[c] * rel[d]) / (nrm * vgn) - cth * rel[c] * rel[d] / (nrm * nrm);
                        let da = aprime * rel[c] * rel[d] / nrm + acos * dcth;
                        let mesh_wf = pulay_m[[c, d]] - drift[c] * nrm * vg[d]
                            + (rel[c] * rel[d] / nrm) * drift_vg;
                        let dln_ratio = mesh_wf - pulay_base[[c, d]];
                        sigma[[c, d]] += ratio * (da + ang * dln_ratio);
                    }
                }
            }
            if let (Some(m), Some(g), Some((acc, colw))) =
                (mesh_det_stress.as_ref(), fgrp.as_ref(), fused_acc.as_mut())
            {
                sigma += &m.fused_group(g, &betas, acc, colw);
            }
            if let (Some(m), Some((_, Some(grp))), Some(g)) =
                (mesh_det_stress.as_ref(), det_grp.as_ref(), g_new.as_ref())
            {
                sigma += &m.group_swap_strain(grp, g);
            }
        }
    }
    if let (Some(m), Some((acc_up, acc_dn))) = (mesh_det_stress.as_ref(), det_acc.as_ref()) {
        sigma += &m.delta_strain(acc_up, acc_dn);
    }
    if let (Some(m), Some((acc, colw))) = (mesh_det_stress.as_ref(), fused_acc.as_ref()) {
        sigma += &m.fused_finish(acc, colw);
    }
    sigma
}

// ============================================================================
// COMPLEX-TWIST (k != 0) GFMC/LRDMC HF force + stress (ECP, periodic).
// The intermediate mesh S' / geminal reverse are COMPLEX (holomorphic in the
// complex Bloch orbitals); the observable HF force/stress is REAL (Re(d e_L^t/dX)).
// Mirrors the real `gfmc_grad_e_l_ecp_pbc` / `gfmc_stress_ecp_pbc_*` structure,
// swapping the determinant side to the complex kernels and Re-projecting at the
// boundary. jnone + j12 (J1/J2) + DTN + j123ao (complex-sphe Bloch J3, via the
// complex J3 electron-axis kinetic force + masked/S'-weighted complex J3 kinetic
// reverse). GFMC_t and GFMC_n share this reduction (the differentiated e_L^t is identical).
// ============================================================================
use crate::complex_det::{compute_geminal_complex, det_phase_lnabs, ComplexGeminal, ComplexOrb};
use numpy::Complex64;

/// Complex-twist J3 context: the complex Bloch-sphe J3 leaf orbital + real j_matrix + the
/// per-sphe-orb `home` atom map (Pulay/force scatter) and per-kernel-AO `kernel` atom map
/// (field-strain leaves). The J3 value/drift/force/stress all use `A = Re(complex sphe AO)`.
pub struct J3PbcComplex<'a> {
    pub orb: &'a ComplexOrb,
    pub jm: ArrayView2<'a, f64>,
    pub home: &'a [usize],
    pub kernel: &'a [usize],
}

/// Complex det `(grad, lap)` fields at a config: reuse the caller-supplied complex
/// geminal inverse (`gi = Some`, the MCMC sweep's running Sherman-Morrison inverse --
/// near-singular-stable, no fresh LU) or rebuild a fresh one (`None`, mesh/candidate
/// configs). Bit-identical to `grads_and_laplacian_ln_det_complex` when `gi` is `None`.
#[allow(clippy::type_complexity)]
fn det_grads_lap_c(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
) {
    match gi {
        Some(v) => {
            crate::complex_det::grads_and_laplacian_ln_det_complex_with_inv(gem, r_up, r_dn, v)
        }
        None => crate::complex_det::grads_and_laplacian_ln_det_complex(gem, r_up, r_dn),
    }
}

/// Holomorphic (un-Re'd) full drift `d ln Psi / d r` per electron `(n_e,3)` at a twist:
/// complex determinant drift `gD` + real Jastrow drift `gJ` (J1/J2/DTN). For the complex
/// mesh S' term, whose hop weight `t` is complex, so `Re(t * d lnPsi/dr)` needs the full
/// complex drift (`Re(t)Re(g) - Im(t)Im(g)`).
fn holo_drift_c(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> Array2<Complex64> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let (gd_up, gd_dn, _, _) = det_grads_lap_c(gem, r_up, r_dn, gi);
    let (gj_up, gj_dn) = jas_drift_c(jas, j3, r_up, r_dn);
    let mut out = Array2::<Complex64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            out[[e, c]] = if e < n_up {
                gd_up[[e, c]] + Complex64::new(gj_up[[e, c]], 0.0)
            } else {
                let l = e - n_up;
                gd_dn[[l, c]] + Complex64::new(gj_dn[[l, c]], 0.0)
            };
        }
    }
    out
}

/// Real Jastrow drift `(gJ_up, gJ_dn)` at a twist (J1/J2/DTN + complex-sphe J3): the
/// Jastrow half of [`holo_drift_c`].
fn jas_drift_c(
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (mut gj_up, mut gj_dn) = jastrow_grads(Some(jas), None, None, r_up, r_dn);
    if let Some(c) = j3 {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(c.orb, &c.jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    (gj_up, gj_dn)
}

/// Add the real Jastrow nuclear force (J1 Williamson + J3 value + DTN) into the holomorphic
/// Pulay `out` `(n_atom,3)`: the Jastrow half of [`holo_pulay_c`].
fn add_jas_pulay_c(
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    n_atom: usize,
    out: &mut Array2<Complex64>,
) {
    if let Some(j1) = &jas.one_body {
        let jf = grad_j1_wrt_r_pbc(j1, r_up, r_dn);
        for i in 0..n_atom {
            for c in 0..3 {
                out[[i, c]] += Complex64::new(jf[[i, c]], 0.0);
            }
        }
    }
    // J3 VALUE Pulay `dJ3/dR` (the J3 AO basis is atom-centered -> nonzero nuclear force).
    if let Some(c) = j3 {
        let jf =
            crate::complex_stress::j3_value_force_complex(c.orb, &c.jm, r_up, r_dn, c.home, n_atom);
        for i in 0..n_atom {
            for cc in 0..3 {
                out[[i, cc]] += Complex64::new(jf[[i, cc]], 0.0);
            }
        }
    }
    if let Some(d) = &jas.dtn {
        let jf = crate::forces_dtn::grad_j_dtn_wrt_r_atom(d, r_up, r_dn);
        for i in 0..n_atom {
            for c in 0..3 {
                out[[i, c]] += Complex64::new(jf[[i, c]], 0.0);
            }
        }
    }
}

/// Holomorphic (un-Re'd) full Pulay force `d ln Psi / d R` `(n_atom,3)` at a twist:
/// complex determinant Pulay + real Jastrow nuclear force (J1 Williamson + DTN;
/// J2/J3 e-e carry no nuclear force). `det_only=true` drops the Jastrow (dltmove SP branch).
#[allow(
    clippy::too_many_arguments,
    reason = "explicit wavefunction, cell and configuration inputs"
)]
fn holo_pulay_c(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    det_only: bool,
    gi: Option<&ArrayView2<Complex64>>,
) -> Array2<Complex64> {
    let mut out = crate::complex_det::pbc_det_pulay_force_complex(
        gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, gi,
    );
    if !det_only {
        add_jas_pulay_c(jas, j3, r_up, r_dn, n_atom, &mut out);
    }
    out
}

/// Complex determinant + Jastrow(J1/J2/DTN) continuum kinetic force (BOTH axes), per-electron
/// S'-weighted, Re-projected: `(d_r (n_atom,3), d_re (n_e,3))`. Det side via
/// `complex_det_kinetic_grads`; Jastrow side via the shared `pbc_jastrow_kinetic_grads` fed
/// `cot = gJ + Re(gD)`. `w=None` => full dT/dtheta.
#[allow(clippy::too_many_arguments)]
fn complex_kinetic_force(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    n_atom: usize,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    w: Option<&[f64]>,
    gi: Option<&ArrayView2<Complex64>>,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
    // The base complex geminal forward at `gi` when the caller already has it.
    fwd: Option<&crate::complex_det::ComplexGemFwd>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    // The det-side bar_gou seed (and cot) needs the FULL Jastrow drift incl the complex J3 -- either
    // a cart-lowerable J3 (`j3`, home Some) OR a PURE PW/B-spline complex J3 leaf (`batch_j3`, home
    // None: its dR-Pulay is 0, so it never rides the cart `j3` path).
    let (mut gj_up, mut gj_dn) = jastrow_grads(Some(jas), None, None, r_up, r_dn);
    if let Some(c) = j3 {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(c.orb, &c.jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    if let Some((orb, jm)) = batch_j3 {
        let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(orb, jm, r_up, r_dn);
        gj_up += &g3u;
        gj_dn += &g3d;
    }
    let (wu, wd) = match w {
        None => (None, None),
        Some(w) => (Some(w[0..n_up].to_vec()), Some(w[n_up..].to_vec())),
    };
    let (mut d_r, mut d_re) = crate::complex_det::complex_det_kinetic_grads(
        gem,
        r_up,
        r_dn,
        &gj_up,
        &gj_dn,
        wu.as_deref(),
        wd.as_deref(),
        nuc_up,
        nuc_dn,
        n_atom,
        gi,
        fwd,
    );
    // cot = gJ + Re(gD)
    let (gd_up, gd_dn) = match fwd {
        Some(f) => (
            drift_up(&f.gou, &f.wg, n_up),
            drift_dn(&f.godn, &f.gq, f.n_dn),
        ),
        None => {
            let (gu, gd, _, _) = det_grads_lap_c(gem, r_up, r_dn, gi);
            (gu, gd)
        }
    };
    let mut cotm = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for c in 0..3 {
            cotm[[e, c]] = if e < n_up {
                gj_up[[e, c]] + gd_up[[e, c]].re
            } else {
                let l = e - n_up;
                gj_dn[[l, c]] + gd_dn[[l, c]].re
            };
        }
    }
    // J1/J2/DTN Jastrow kinetic (shared real reduction; j3=None -- the complex J3 kinetic is
    // the complex-Bloch-sphe reverse below, not the real cart three-body).
    let (jr, jre) = pbc_jastrow_kinetic_grads(jas, None, &cotm, r_up, r_dn, cell, n_atom, w);
    d_r += &jr;
    d_re += &jre;
    // J3 (complex Bloch sphe): masked kinetic reverse on BOTH axes. Atom axis via
    // `j3_side_complex` (bar_gj = -cot), electron axis via `j3_egrad_side_complex` (cot).
    if let Some(c) = j3 {
        let bar_gj = |e: usize, d: usize| -> f64 { -cotm[[e, d]] };
        crate::complex_det::j3_side_complex(c.orb, &c.jm, r_up, r_dn, &bar_gj, c.home, w, &mut d_r);
        // A PURE (MO-wrapped) plane-wave / B-spline complex J3 leaf has an empty GTO block, so its
        // electron-axis kinetic SELF-force is served by the leaf-owning `j3_leaf_egrad_side_complex`
        // (the complex twin of the real `j3_leaf_egrad_side`); the atom-centered GTO path
        // `j3_egrad_side_complex` covers only `j3.aos`. (The dR self side is 0 for both -- dAO/dR=0.)
        if c.orb.aos.num_ao == 0 && (c.orb.bspline.is_some() || c.orb.pw.is_some()) {
            crate::complex_det::j3_leaf_egrad_side_complex(
                c.orb, &c.jm, r_up, r_dn, &cotm, w, &mut d_re,
            );
        } else {
            crate::complex_det::j3_egrad_side_complex(
                c.orb, &c.jm, r_up, r_dn, &cotm, w, &mut d_re,
            );
        }
    }
    // PURE PW/B-spline complex J3 leaf kinetic SELF electron-force (home None -> no cart dR side;
    // its dR contribution already rode `cotm` into the J1/J2/det kinetic force above). The complex
    // twin of the real `full_kinetic_grads` `batch_j3` electron-axis leaf force.
    if let Some((orb, jm)) = batch_j3 {
        crate::complex_det::j3_leaf_egrad_side_complex(orb, jm, r_up, r_dn, &cotm, w, &mut d_re);
    }
    (d_r, d_re)
}

/// Complex-twist analogue of [`MeshDetFastPbc`]: cached base determinant fields for the
/// ECP non-local FORCE mesh, where every candidate hops ONE electron. The base forward
/// (`complex_gem_forward`, rooted at the caller's swept inverse `gi` when given) is built
/// once; each candidate evaluates only the moved electron's orbital value/gradient row and
/// updates the inverse by the rank-1 [`sm_rank1_bars`] (complex, plain transposes). It
/// returns the det ratio, the holomorphic det drift of every electron (base orbital
/// gradients, moved row fresh) and the det Pulay (`pbc_det_pulay_force_complex` on the
/// rank-1 bars, base AO gradients with the moved column swapped). The Pulay runs over the
/// atom-centred GTO block only: a pure B-spline / plane-wave orbital has dAO/dR = 0 and
/// contributes nothing, so it is skipped. A tiled orbital set (its own field VJP) keeps the
/// per-candidate full recompute at the caller (`new` -> `None`).
struct MeshDetFastC<'a> {
    orb_up: &'a ComplexOrb,
    orb_dn: &'a ComplexOrb,
    f_own: Option<crate::complex_det::ComplexGemFwd>,
    f_ref: Option<&'a crate::complex_det::ComplexGemFwd>,
    /// Base GTO-block AO gradients `(num_gto, N)`; `None` when the spin has no GTO block
    /// (and in the grouped mode, which never forms them).
    ga_up: Option<[Array2<Complex64>; 3]>,
    ga_dn: Option<[Array2<Complex64>; 3]>,
    nuc_up: &'a [usize],
    nuc_dn: &'a [usize],
    n_up: usize,
    n_atom: usize,
    /// Grouped mode ([`Self::group`], both spins covered by
    /// [`crate::complex_stress::moved_group_ok`]): home-atom shell geometry of each spin's gto
    /// block (`None` for a spin without one).
    grouped: bool,
    geom_up: Option<crate::mesh_vgs::ShellGeom>,
    geom_dn: Option<crate::mesh_vgs::ShellGeom>,
}

/// One `(electron, atom)` group of grouped complex det force candidates
/// ([`MeshDetFastC::group`]): per candidate the det ratio, the moved electron's det drift row and
/// its moved-column det Pulay `(n_atom,3)` (fresh minus base column, over the ratio), plus the
/// moved electron's AO value table for the group's weighted bar deltas ([`MeshDetFastC::fold`]).
struct DetForceGroupC {
    spin_up: bool,
    li: usize,
    ratio: Vec<Complex64>,
    drow: Vec<[Complex64; 3]>,
    pcol: Vec<Array2<Complex64>>,
    aotab: Array2<Complex64>,
}

impl<'a> MeshDetFastC<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gem: &'a ComplexGeminal,
        nuc_up: &'a [usize],
        nuc_dn: &'a [usize],
        n_atom: usize,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        gi: Option<&ArrayView2<Complex64>>,
        fwd: Option<&'a crate::complex_det::ComplexGemFwd>,
    ) -> Option<Self> {
        if gem.orb_up.tiled.is_some() || gem.orb_dn.tiled.is_some() {
            return None;
        }
        let geom =
            |orb: &ComplexOrb, nuc: &[usize]| -> Option<Option<crate::mesh_vgs::ShellGeom>> {
                if orb.aos.num_ao == 0 {
                    return Some(None);
                }
                crate::mesh_vgs::ShellGeom::new(&orb.aos, nuc, None).map(Some)
            };
        let ok = |orb: &ComplexOrb| crate::complex_stress::moved_group_ok(orb);
        let (mut grouped, mut geom_up, mut geom_dn) = (false, None, None);
        if ok(&gem.orb_up) && ok(&gem.orb_dn) && std::env::var("JQMC_RS_ECPF_SLOW").is_err() {
            if let (Some(gu), Some(gd)) = (geom(&gem.orb_up, nuc_up), geom(&gem.orb_dn, nuc_dn)) {
                grouped = true;
                geom_up = gu;
                geom_dn = gd;
            }
        }
        let ga = |orb: &ComplexOrb, r: &ArrayView2<f64>| {
            (!grouped && orb.aos.num_ao > 0).then(|| orb.ao_grad(r))
        };
        let f_own = match fwd {
            Some(_) => None,
            None => Some(crate::complex_det::complex_gem_forward(gem, r_up, r_dn, gi)),
        };
        Some(MeshDetFastC {
            orb_up: &gem.orb_up,
            orb_dn: &gem.orb_dn,
            f_own,
            f_ref: fwd,
            ga_up: ga(&gem.orb_up, r_up),
            ga_dn: ga(&gem.orb_dn, r_dn),
            nuc_up,
            nuc_dn,
            n_up: r_up.nrows(),
            n_atom,
            grouped,
            geom_up,
            geom_dn,
        })
    }

    fn fwd(&self) -> &crate::complex_det::ComplexGemFwd {
        match self.f_ref {
            Some(f) => f,
            None => self.f_own.as_ref().unwrap(),
        }
    }

    /// Base det drift `(n_e,3)` from the forward (`gou . W G^-1`, `godn . G^-1 Q`).
    fn base_drift(&self) -> Array2<Complex64> {
        let f = self.fwd();
        let du = drift_up(&f.gou, &f.wg, self.n_up);
        let dd = drift_dn(&f.godn, &f.gq, f.n_dn);
        numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[du.view(), dd.view()]).unwrap()
    }

    /// Grouped mode: the `nv` candidates of electron `li` (`spin_up`) moved to `pts`, from ONE
    /// fused contraction of the moved electron over the mesh with the base bar column
    /// `col = bar_base[:, li]` ([`crate::complex_stress::moved_group_complex`], the e_L mesh value
    /// screen): the candidate bar column is the base column over the det ratio, so the
    /// contracted value is the det ratio, the contracted gradient over it the moved det drift
    /// row and the atom-resolved one (minus the base column at `r_li`, unscreened as the base
    /// tables) over it the moved-column det Pulay. The other bar columns are linear in the
    /// moved orbital change over the ratio ([`Self::fold`]).
    fn group(
        &self,
        spin_up: bool,
        li: usize,
        pts: &[[f64; 3]],
        r_li: [f64; 3],
    ) -> Option<DetForceGroupC> {
        if !self.grouped {
            return None;
        }
        let f = self.fwd();
        let (orb, geom) = if spin_up {
            (self.orb_up, self.geom_up.as_ref())
        } else {
            (self.orb_dn, self.geom_dn.as_ref())
        };
        let col: Array1<Complex64> = if spin_up {
            f.wg.column(li).to_owned()
        } else {
            f.gq.row(li).to_owned()
        };
        let na = if orb.aos.num_ao > 0 { self.n_atom } else { 0 };
        let mv = crate::complex_stress::moved_group_complex(
            orb,
            pts,
            &col,
            geom,
            na,
            false,
            AoScreen::MeshValue,
        )?;
        let base = if na > 0 {
            Some(crate::complex_stress::moved_group_complex(
                orb,
                &[r_li],
                &col,
                geom,
                na,
                false,
                AoScreen::None,
            )?)
        } else {
            None
        };
        let nv = pts.len();
        let mut ratio = Vec::with_capacity(nv);
        let mut drow = Vec::with_capacity(nv);
        let mut pcol = Vec::with_capacity(nv);
        for v in 0..nv {
            let r = mv.val[v];
            ratio.push(r);
            drow.push([mv.grad[v][0] / r, mv.grad[v][1] / r, mv.grad[v][2] / r]);
            let mut p = Array2::<Complex64>::zeros((self.n_atom, 3));
            if let Some(b) = &base {
                for a in 0..na {
                    for d in 0..3 {
                        p[[a, d]] = -(mv.atom[v * na + a][d] - b.atom[a][d]) / r;
                    }
                }
            }
            pcol.push(p);
        }
        Some(DetForceGroupC {
            spin_up,
            li,
            ratio,
            drow,
            pcol,
            aotab: mv.aotab,
        })
    }

    /// Fold a group's weighted bar deltas `Sum_v c_v (bar_v - bar_base)` for `s[v] = c_v /
    /// det_ratio_v` into the step accumulators: `L(dmo)` of the complex [`sm_linear_bars`] with
    /// `dmo = C (aotab s) - (Sum s) mo_base[:, li]` into `acc` (both spins), and its moved
    /// column `li` also into `colfix` (the moved electron's drift row is carried per candidate).
    fn fold(
        &self,
        g: &DetForceGroupC,
        s: &[Complex64],
        acc: &mut (Array2<Complex64>, Array2<Complex64>),
        colfix: &mut (Array2<Complex64>, Array2<Complex64>),
    ) {
        let f = self.fwd();
        let orb = if g.spin_up { self.orb_up } else { self.orb_dn };
        let ssum: Complex64 = s.iter().sum();
        let a = g.aotab.dot(&ArrayView1::from(s));
        let mo_new = match &orb.mo {
            Some(c) => c.dot(&a),
            None => a,
        };
        let base_o = if g.spin_up {
            f.ou.column(g.li)
        } else {
            f.od.column(g.li)
        };
        let dmo = &mo_new - &base_o.mapv(|x| x * ssum);
        let (lw, ld) = sm_linear_bars(
            g.spin_up,
            g.li,
            &dmo,
            &f.w,
            &f.q,
            &f.ginv,
            f.lam_p.as_ref(),
            f.n_dn,
        );
        acc.0 += &lw;
        acc.1 += &ld;
        if g.spin_up {
            let mut c = colfix.0.column_mut(g.li);
            c += &lw.column(g.li);
        } else {
            let mut c = colfix.1.column_mut(g.li);
            c += &ld.column(g.li);
        }
    }

    fn new_acc(&self) -> (Array2<Complex64>, Array2<Complex64>) {
        let f = self.fwd();
        (
            Array2::zeros(f.wg.raw_dim()),
            Array2::zeros((f.gq.ncols(), f.n_dn)),
        )
    }

    /// The bar-delta part of the whole step: the det Pulay `(n_atom,3)` (real part) of `acc` on
    /// the base gto tables (one fused contraction per spin at the base electrons) and the det
    /// drift `(n_e,3)` of `acc - colfix` on the base orbital gradients.
    fn finish(
        &self,
        acc: &(Array2<Complex64>, Array2<Complex64>),
        colfix: &(Array2<Complex64>, Array2<Complex64>),
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<Complex64>) {
        let f = self.fwd();
        let mut pulay = Array2::<f64>::zeros((self.n_atom, 3));
        for (orb, bar, geom, r) in [
            (self.orb_up, &acc.0, self.geom_up.as_ref(), r_up.view()),
            (self.orb_dn, &acc.1, self.geom_dn.as_ref(), r_dn.view()),
        ] {
            let Some(geom) = geom else { continue };
            let cm = match &orb.mo {
                Some(c) => c.t().dot(bar),
                None => bar.clone(),
            };
            let g0 = orb.gto_ao_offset();
            let cm = cm
                .slice(numpy::ndarray::s![g0..g0 + orb.aos.num_ao, ..])
                .to_owned();
            let (_, atom) = crate::complex_stress::fused_complex_kin_force(
                orb,
                &r,
                &cm,
                None,
                None,
                geom,
                self.n_atom,
            )
            .expect("grouped complex det force: gto block covered");
            pulay -= &atom;
        }
        let du = drift_up(&f.gou, &(&acc.0 - &colfix.0), self.n_up);
        let dd = drift_dn(&f.godn, &(&acc.1 - &colfix.1).t().to_owned(), f.n_dn);
        let drift =
            numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[du.view(), dd.view()]).unwrap();
        (pulay, drift)
    }

    /// One spin side of the det Pulay `-Sum_k cm[a,k] * gradAO[a,k]` scattered to `nuc[a]`
    /// (`cm` = GTO rows of `MO^T bar`), with column `li` taken from `moved` when given.
    fn pulay_side(
        orb: &ComplexOrb,
        bar: &Array2<Complex64>,
        ga: &[Array2<Complex64>; 3],
        nuc: &[usize],
        moved: Option<(usize, &[Array2<Complex64>; 3])>,
        out: &mut Array2<Complex64>,
    ) {
        let cm = crate::complex_det::gto_slice_cm(
            orb,
            match &orb.mo {
                Some(c) => c.t().dot(bar),
                None => bar.clone(),
            },
        );
        for a in 0..cm.nrows() {
            let ia = nuc[a];
            for c in 0..3 {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..cm.ncols() {
                    let g = match moved {
                        Some((li, col)) if k == li => col[c][[a, 0]],
                        _ => ga[c][[a, k]],
                    };
                    acc += cm[[a, k]] * g;
                }
                out[[ia, c]] -= acc;
            }
        }
    }

    /// One mesh candidate (`spin_up`, local index `li`, moved to `new_pos`):
    /// `(det_ratio, det_pulay (n_atom,3), det_drift (n_e,3))`, holomorphic.
    fn candidate(
        &self,
        spin_up: bool,
        li: usize,
        new_pos: [f64; 3],
    ) -> (Complex64, Array2<Complex64>, Array2<Complex64>) {
        let f = self.fwd();
        let (n_up, n_dn) = (self.n_up, f.n_dn);
        let orb = if spin_up { self.orb_up } else { self.orb_dn };
        let r1 = Array2::from_shape_vec((1, 3), new_pos.to_vec()).unwrap();
        let (mv, mg, _) = orb.value_grad_lap(&r1.view());
        let (ratio, wg, gq) = sm_rank1_bars(
            spin_up,
            li,
            &mv,
            &f.ou,
            &f.od,
            &f.w,
            &f.q,
            &f.ginv,
            f.lam_p.as_ref(),
            n_up,
        );
        // Drift on the base orbital gradients; the moved electron's row uses its fresh one.
        let mut du = drift_up(&f.gou, &wg, n_up);
        let mut dd = drift_dn(&f.godn, &gq, n_dn);
        for d in 0..3 {
            let mut acc = Complex64::new(0.0, 0.0);
            for a in 0..mg[d].nrows() {
                acc += mg[d][[a, 0]] * if spin_up { wg[[a, li]] } else { gq[[li, a]] };
            }
            if spin_up {
                du[[li, d]] = acc;
            } else {
                dd[[li, d]] = acc;
            }
        }
        let mut drift = Array2::<Complex64>::zeros((n_up + n_dn, 3));
        drift.slice_mut(numpy::ndarray::s![0..n_up, ..]).assign(&du);
        drift.slice_mut(numpy::ndarray::s![n_up.., ..]).assign(&dd);
        let mut pulay = Array2::<Complex64>::zeros((self.n_atom, 3));
        let moved_ga = (orb.aos.num_ao > 0).then(|| orb.ao_grad(&r1.view()));
        let moved = moved_ga.as_ref().map(|g| (li, g));
        if let Some(ga) = &self.ga_up {
            Self::pulay_side(
                self.orb_up,
                &wg,
                ga,
                self.nuc_up,
                moved.filter(|_| spin_up),
                &mut pulay,
            );
        }
        if let Some(ga) = &self.ga_dn {
            let bar_dn = gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned();
            Self::pulay_side(
                self.orb_dn,
                &bar_dn,
                ga,
                self.nuc_dn,
                moved.filter(|_| !spin_up),
                &mut pulay,
            );
        }
        (ratio, pulay, drift)
    }
}

/// The real Jastrow side of the complex mesh force candidates: the one-electron
/// [`MeshJasFastPbc`] when it covers the Jastrow, else the fields recomputed at the moved
/// config with the `jastrow_ratio`-zone single-move ratio of `ratio_state` (issue #757).
struct MeshJasC<'a, 'j, 'b, 'd> {
    jas: &'a JastrowData,
    j3: Option<&'a J3PbcComplex<'j>>,
    batch_j3: Option<(&'a crate::complex_det::ComplexOrb, &'a ArrayView2<'b, f64>)>,
    mesh: Option<&'a MeshJasFastPbc<'d>>,
    j3c: Option<&'a ComplexOrb>,
    /// Built whenever `mesh` is `None` and a Jastrow is present.
    ratio_state: Option<&'a crate::gfmc::ComplexJasRatioState>,
    with_j: bool,
}

impl MeshJasC<'_, '_, '_, '_> {
    /// `(Jastrow Pulay (n_atom,3), Jastrow drift (n_e,3))` at a full config (J1/J2/DTN + J3 +
    /// the pure-leaf J3 drift).
    fn fields(
        &self,
        ru: &ArrayView2<f64>,
        rd: &ArrayView2<f64>,
        n_atom: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut pc = Array2::<Complex64>::zeros((n_atom, 3));
        add_jas_pulay_c(self.jas, self.j3, ru, rd, n_atom, &mut pc);
        let (mut gu, mut gd) = jas_drift_c(self.jas, self.j3, ru, rd);
        if let Some((orb, jm)) = self.batch_j3 {
            let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(orb, jm, ru, rd);
            gu += &g3u;
            gd += &g3d;
        }
        let gj =
            numpy::ndarray::concatenate(numpy::ndarray::Axis(0), &[gu.view(), gd.view()]).unwrap();
        (pc.mapv(|x| x.re), gj)
    }

    /// `exp(J(r') - J(r))` for moving electron `li` (spin `is_up`) of the base `(ru, rd)` to
    /// `pos`, in the `jastrow_ratio` zone.
    #[allow(clippy::too_many_arguments)]
    fn ratio(
        &self,
        ru: &ArrayView2<f64>,
        rd: &ArrayView2<f64>,
        is_up: bool,
        li: usize,
        pos: [f64; 3],
    ) -> f64 {
        crate::gfmc::bump_route_counter_by(&crate::gfmc::GFMC_CPLX_ECPF_GROUPED_JAS_RATIOS, 1);
        self.ratio_state
            .expect("complex mesh-force Jastrow ratio state: built whenever the one-electron mesh is absent")
            .ratio(self.jas, self.j3c, ru, rd, is_up, li, pos)
    }
}

/// Grouped complex-twist ECP non-local mesh force (the [`MeshDetFastC`] grouped mode): the same
/// estimator as the per-candidate loop of [`gfmc_ecp_nonlocal_force_complex`], with the det
/// Pulay / drift of every candidate split into its moved electron's exact column / row
/// ([`MeshDetFastC::group`]) and the part LINEAR in the weighted bar deltas, which is summed over
/// the whole mesh ([`MeshDetFastC::fold`]) and contracted once on the base tables
/// ([`MeshDetFastC::finish`]). The base det Pulay cancels between the candidates and the base
/// (`Sum_v c_v (P_v - P_0)`), so only the base det drift row of each moved electron is needed.
#[allow(clippy::too_many_arguments)]
fn ecp_nl_force_complex_grouped(
    md: &MeshDetFastC,
    jc: &MeshJasC,
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gp: &[[f64; 3]],
    uniq: &[usize],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    nn: usize,
    dltmove: bool,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let nv = gp.len();
    let z = Complex64::new(0.0, 0.0);
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.to_vec();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    inv,
                    n_max,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let det_base = md.base_drift();
    // Base Jastrow Pulay / drift (the det base Pulay cancels, see above).
    let (pj0, gj0) = match jc.mesh {
        Some(m) => (
            m.base_jp.clone(),
            numpy::ndarray::concatenate(
                numpy::ndarray::Axis(0),
                &[m.base_gju.view(), m.base_gjd.view()],
            )
            .unwrap(),
        ),
        None if jc.with_j => jc.fields(r_up, r_dn, n_atom),
        None => (Array2::zeros((n_atom, 3)), Array2::zeros((n_e, 3))),
    };
    let mut acc = md.new_acc();
    let mut colfix = md.new_acc();
    let mut d_r = Array2::<Complex64>::zeros((n_atom, 3));
    let mut d_re = Array2::<Complex64>::zeros((n_e, 3));
    for gi in 0..n_e {
        let is_up = gi < n_up;
        let li = if is_up { gi } else { gi - n_up };
        let re = elec(gi);
        for &ia in &nearest_set(re) {
            let rel = min_image(
                [
                    positions[[ia, 0]] - re[0],
                    positions[[ia, 1]] - re[1],
                    positions[[ia, 2]] - re[2],
                ],
                cell,
                inv,
                n_max,
            );
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let uhat = [rel[0] / nrm, rel[1] / nrm, rel[2] / nrm];
            let new_positions: Vec<[f64; 3]> = gp
                .iter()
                .map(|g| {
                    [
                        re[0] + rel[0] + nrm * g[0],
                        re[1] + rel[1] + nrm * g[1],
                        re[2] + rel[2] + nrm * g[2],
                    ]
                })
                .collect();
            let grp = md
                .group(is_up, li, &new_positions, re)
                .expect("grouped complex det force: leaf covered");
            let jas_res = jc
                .mesh
                .map(|m| m.candidates(is_up, li, &new_positions, r_up, r_dn));
            let mut s = vec![z; nv];
            for v in 0..nv {
                let g = gp[v];
                let dnv = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let cosv = (-rel[0] * g[0] - rel[1] * g[1] - rel[2] * g[2]) / (nrm * dnv);
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (p, e, c) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-e * nrm * nrm).exp();
                    let v_l = c * nrm.powf(p - 2.0) * expf;
                    let v_lp =
                        c * ((p - 2.0) * nrm.powf(p - 3.0) - 2.0 * e * nrm.powf(p - 1.0)) * expf;
                    let (pl, dpl) = leg_and_deriv(l, cosv);
                    let w = ((2 * l + 1) as f64) * weights[v];
                    ang += v_l * w * pl;
                    aprime += v_lp * w * pl;
                    acos += v_l * w * dpl;
                }
                let mut dang_dri = [0.0_f64; 3];
                for d in 0..3 {
                    dang_dri[d] =
                        aprime * (-uhat[d]) + acos * (g[d] / (nrm * dnv) + cosv * uhat[d] / nrm);
                }
                let det_ratio = grp.ratio[v];
                // Jastrow ratio and fields at the moved config.
                let jfull: Option<(f64, Array2<f64>, Array2<f64>)> = match &jas_res {
                    Some(r) => {
                        let (jr, pj, gju, gjd) = &r[v];
                        let gj = numpy::ndarray::concatenate(
                            numpy::ndarray::Axis(0),
                            &[gju.view(), gjd.view()],
                        )
                        .unwrap();
                        Some((*jr, pj.clone(), gj))
                    }
                    None if jc.with_j => {
                        let (mut mu, mut md2) = (r_up.to_owned(), r_dn.to_owned());
                        let np = new_positions[v];
                        if is_up {
                            for c in 0..3 {
                                mu[[li, c]] = np[c];
                            }
                        } else {
                            for c in 0..3 {
                                md2[[li, c]] = np[c];
                            }
                        }
                        let jr = jc.ratio(r_up, r_dn, is_up, li, np);
                        let (pj, gj) = jc.fields(&mu.view(), &md2.view(), n_atom);
                        Some((jr, pj, gj))
                    }
                    None => None,
                };
                let jr = jfull.as_ref().map_or(1.0, |j| j.0);
                let det_amp = det_ratio * ang;
                let use_full = !dltmove || det_amp.re < 0.0;
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                let coeff = ratio * ang;
                s[v] = coeff / det_ratio;
                let jon = if use_full { jfull.as_ref() } else { None };
                for d in 0..3 {
                    d_re[[gi, d]] += ratio * dang_dri[d];
                    d_r[[ia, d]] += ratio * (-dang_dri[d]);
                }
                // moved electron's full drift row and its base
                let mut drow = grp.drow[v];
                let mut brow = [det_base[[gi, 0]], det_base[[gi, 1]], det_base[[gi, 2]]];
                if let Some((_, _, gj)) = jon {
                    for c in 0..3 {
                        drow[c] += Complex64::new(gj[[gi, c]], 0.0);
                        brow[c] += Complex64::new(gj0[[gi, c]], 0.0);
                    }
                }
                let dm_dot_g: Complex64 = (0..3).map(|c| drow[c] * g[c]).sum();
                for d in 0..3 {
                    d_re[[gi, d]] += coeff * (-dm_dot_g * uhat[d] - brow[d]);
                }
                if let Some((_, _, gj)) = jon {
                    for ee in 0..n_e {
                        if ee == gi {
                            continue;
                        }
                        for d in 0..3 {
                            d_re[[ee, d]] += coeff * (gj[[ee, d]] - gj0[[ee, d]]);
                        }
                    }
                }
                let pcol = &grp.pcol[v];
                for ib in 0..n_atom {
                    for c in 0..3 {
                        let mut val = pcol[[ib, c]];
                        if let Some((_, pj, _)) = jon {
                            val += pj[[ib, c]] - pj0[[ib, c]];
                        }
                        if ib == ia {
                            val += drow[c] + dm_dot_g * uhat[c];
                        }
                        d_r[[ib, c]] += coeff * val;
                    }
                }
            }
            md.fold(&grp, &s, &mut acc, &mut colfix);
        }
    }
    let (p_lin, d_lin) = md.finish(&acc, &colfix, r_up, r_dn);
    let mut d_r = d_r.mapv(|x| x.re);
    d_r += &p_lin;
    let mut d_re_r = d_re.mapv(|x| x.re);
    d_re_r += &d_lin.mapv(|x| x.re);
    (d_r, d_re_r)
}

/// Complex-twist ECP non-local periodic mesh force (BOTH axes, Re-projected):
/// `(d_r (n_atom,3), d_re (n_e,3))`. Mirror of the real `ecp_nonlocal_force_pbc` with the
/// complex determinant ratio + holomorphic Pulay/drift; grid rotated by RT (LRDMC mesh).
/// jnone/j12/DTN + complex J3 (via holo_pulay_c/holo_drift_c). tmove (use_full always); dltmove det-only SP branch.
#[allow(clippy::too_many_arguments)]
fn gfmc_ecp_nonlocal_force_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    // Optional base-config complex inverse. `None` builds one locally; mesh
    // candidates always rebuild fresh in this low-level ECP helper.
    gi: Option<&ArrayView2<Complex64>>,
    // PURE PW/B-spline complex J3 leaf (home None -> not a cart `j3`). It enters the reflected-config
    // VALUE ratio (`jval`) and the electron drift (`holo_drift_c` result), but carries NO nuclear
    // Pulay (dAO/dR = 0), so `holo_pulay_c` is untouched.
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
    // The base complex geminal forward at `gi` when the caller already has it.
    fwd: Option<&crate::complex_det::ComplexGemFwd>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let j1 = jas.one_body.as_ref();
    let j2 = jas.two_body.as_ref();
    let dtn = jas.dtn.as_ref();
    let j3c: Option<&ComplexOrb> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.orb),
        (None, Some((o, _))) => Some(o),
        (None, None) => None,
    };
    let j3m_owned: Option<Array2<f64>> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.jm.to_owned()),
        (None, Some((_, m))) => Some(m.to_owned()),
        (None, None) => None,
    };
    let j3m = j3m_owned.as_ref().map(|a| a.view());
    let with_j =
        j1.is_some() || j2.is_some() || dtn.is_some() || j3.is_some() || batch_j3.is_some();
    // Add the PURE PW/B-spline leaf J3 electron drift onto a holo-drift (`holo_drift_c` sees only
    // the cart `j3`, None for these leaves). A no-op when `batch_j3` is None.
    let add_batch_drift = |mut du: Array2<Complex64>,
                           ru: &ArrayView2<f64>,
                           rd: &ArrayView2<f64>|
     -> Array2<Complex64> {
        if let Some((orb, jm)) = batch_j3 {
            let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(orb, jm, ru, rd);
            let nu = ru.nrows();
            for e in 0..du.nrows() {
                for c in 0..3 {
                    du[[e, c]] += Complex64::new(
                        if e < nu {
                            g3u[[e, c]]
                        } else {
                            g3d[[e - nu, c]]
                        },
                        0.0,
                    );
                }
            }
        }
        du
    };
    let elec = |gi: usize| -> [f64; 3] {
        if gi < n_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let nv = weights.len();
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
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    inv,
                    n_max,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    // One-electron mesh fields (#574): every candidate hops ONE electron, so the det ratio,
    // drift and Pulay come from the rank-1 [`MeshDetFastC`] and the real Jastrow ones from
    // [`MeshJasFastPbc`] (J1/J2/DTN; the Jastrow is real at a twist). `None` (tiled orbitals;
    // a complex J3 or pure PW/B-spline J3 leaf for the Jastrow) -> the per-candidate full
    // recompute below.
    let mesh_det = MeshDetFastC::new(gem, nuc_up, nuc_dn, n_atom, r_up, r_dn, gi, fwd);
    let mesh_jas = if with_j && j3.is_none() && batch_j3.is_none() {
        MeshJasFastPbc::new(jas, None, cell, r_up, r_dn, n_atom)
    } else {
        None
    };
    // Without the one-electron Jastrow mesh (a J3 leaf), the candidate Jastrow ratio is the
    // `jastrow_ratio`-zone single-move ratio, as in the twist ECP energy / parameter meshes; the
    // difference of two whole `jastrow_value_complex` values rounds in `jastrow_eval` (float32
    // in mixed), issue #757.
    let jas_ratio_state = (with_j && mesh_jas.is_none())
        .then(|| crate::gfmc::ComplexJasRatioState::new(jas, j3c, j3m.as_ref(), r_up, r_dn));
    if let Some(md) = mesh_det.as_ref().filter(|m| m.grouped) {
        let jctx = MeshJasC {
            jas,
            j3,
            batch_j3,
            mesh: mesh_jas.as_ref(),
            j3c,
            ratio_state: jas_ratio_state.as_ref(),
            with_j,
        };
        return ecp_nl_force_complex_grouped(
            md, &jctx, positions, r_up, r_dn, &gp, &uniq, cell, inv, n_max, nuc_n, am_n, e_n, c_n,
            p_n, weights, nn, dltmove,
        );
    }
    let det0 = if mesh_det.is_none() {
        det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn))
    } else {
        (Complex64::new(1.0, 0.0), 0.0)
    };
    let base_pulay_f = holo_pulay_c(gem, jas, j3, r_up, r_dn, nuc_up, nuc_dn, n_atom, false, gi);
    let base_du_f = add_batch_drift(holo_drift_c(gem, jas, j3, r_up, r_dn, gi), r_up, r_dn);
    let base_pulay_d = if dltmove {
        holo_pulay_c(gem, jas, j3, r_up, r_dn, nuc_up, nuc_dn, n_atom, true, gi)
    } else {
        Array2::zeros((0, 0))
    };
    let base_du_d = if dltmove {
        let (gu, gd, _, _) = det_grads_lap_c(gem, r_up, r_dn, gi);
        let mut o = Array2::<Complex64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                o[[e, c]] = if e < n_up {
                    gu[[e, c]]
                } else {
                    gd[[e - n_up, c]]
                };
            }
        }
        o
    } else {
        Array2::zeros((0, 0))
    };
    let mut d_r = Array2::<Complex64>::zeros((n_atom, 3));
    let mut d_re = Array2::<Complex64>::zeros((n_e, 3));
    for gi in 0..n_e {
        let is_up = gi < n_up;
        let li = if is_up { gi } else { gi - n_up };
        let re = elec(gi);
        for &ia in &nearest_set(re) {
            let rel0 = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let rel = min_image(rel0, cell, inv, n_max);
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let uhat = [rel[0] / nrm, rel[1] / nrm, rel[2] / nrm];
            let new_positions: Vec<[f64; 3]> = gp
                .iter()
                .map(|g| {
                    [
                        re[0] + rel[0] + nrm * g[0],
                        re[1] + rel[1] + nrm * g[1],
                        re[2] + rel[2] + nrm * g[2],
                    ]
                })
                .collect();
            let jas_res = mesh_jas
                .as_ref()
                .map(|mj| mj.candidates(is_up, li, &new_positions, r_up, r_dn));
            for v in 0..nv {
                let g = gp[v];
                let dnv = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                let cosv = (-rel[0] * g[0] - rel[1] * g[1] - rel[2] * g[2]) / (nrm * dnv);
                let new_pos = new_positions[v];
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (p, e, c) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-e * nrm * nrm).exp();
                    let v_l = c * nrm.powf(p - 2.0) * expf;
                    let v_lp =
                        c * ((p - 2.0) * nrm.powf(p - 3.0) - 2.0 * e * nrm.powf(p - 1.0)) * expf;
                    let (pl, dpl) = leg_and_deriv(l, cosv);
                    let w = ((2 * l + 1) as f64) * weights[v];
                    ang += v_l * w * pl;
                    aprime += v_lp * w * pl;
                    acos += v_l * w * dpl;
                }
                let mut dang_dri = [0.0_f64; 3];
                for d in 0..3 {
                    dang_dri[d] =
                        aprime * (-uhat[d]) + acos * (g[d] / (nrm * dnv) + cosv * uhat[d] / nrm);
                }
                let (mu, md) = {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if is_up {
                        mu[[li, 0]] = new_pos[0];
                        mu[[li, 1]] = new_pos[1];
                        mu[[li, 2]] = new_pos[2];
                    } else {
                        md[[li, 0]] = new_pos[0];
                        md[[li, 1]] = new_pos[1];
                        md[[li, 2]] = new_pos[2];
                    }
                    (mu, md)
                };
                let fast = mesh_det.as_ref().map(|m| m.candidate(is_up, li, new_pos));
                let det_ratio = match &fast {
                    Some((r, _, _)) => *r,
                    None => det_ratio_complex(
                        det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), &md.view())),
                        det0,
                    ),
                };
                let jas_v = jas_res.as_ref().map(|r| &r[v]);
                let jr = match (jas_v, jas_ratio_state.as_ref()) {
                    (Some((jr, _, _, _)), _) => *jr,
                    (None, Some(st)) => st.ratio(jas, j3c, r_up, r_dn, is_up, li, new_pos),
                    (None, None) => 1.0,
                };
                let det_amp = det_ratio * ang;
                let use_full = !dltmove || det_amp.re < 0.0;
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                let coeff = ratio * ang;
                let (pm, dmesh, base_pulay, base_du) = match (fast, use_full) {
                    // Rank-1 det fields plus the Jastrow ones (one-electron when covered).
                    (Some((_, mut pm, mut dmesh)), true) => {
                        match jas_v {
                            Some((_, pj, gju, gjd)) => {
                                pm += &pj.mapv(|x| Complex64::new(x, 0.0));
                                for e in 0..n_e {
                                    for c in 0..3 {
                                        let x = if e < n_up {
                                            gju[[e, c]]
                                        } else {
                                            gjd[[e - n_up, c]]
                                        };
                                        dmesh[[e, c]] += Complex64::new(x, 0.0);
                                    }
                                }
                            }
                            None => {
                                add_jas_pulay_c(jas, j3, &mu.view(), &md.view(), n_atom, &mut pm);
                                let (gju, gjd) = jas_drift_c(jas, j3, &mu.view(), &md.view());
                                for e in 0..n_e {
                                    for c in 0..3 {
                                        let x = if e < n_up {
                                            gju[[e, c]]
                                        } else {
                                            gjd[[e - n_up, c]]
                                        };
                                        dmesh[[e, c]] += Complex64::new(x, 0.0);
                                    }
                                }
                                dmesh = add_batch_drift(dmesh, &mu.view(), &md.view());
                            }
                        }
                        (pm, dmesh, &base_pulay_f, &base_du_f)
                    }
                    // dltmove SP branch: the det-only fields are the rank-1 ones as they are.
                    (Some((_, pm, dmesh)), false) => (pm, dmesh, &base_pulay_d, &base_du_d),
                    (None, true) => (
                        holo_pulay_c(
                            gem,
                            jas,
                            j3,
                            &mu.view(),
                            &md.view(),
                            nuc_up,
                            nuc_dn,
                            n_atom,
                            false,
                            None,
                        ),
                        add_batch_drift(
                            holo_drift_c(gem, jas, j3, &mu.view(), &md.view(), None),
                            &mu.view(),
                            &md.view(),
                        ),
                        &base_pulay_f,
                        &base_du_f,
                    ),
                    (None, false) => {
                        let (gu, gd, _, _) = crate::complex_det::grads_and_laplacian_ln_det_complex(
                            gem,
                            &mu.view(),
                            &md.view(),
                        );
                        let mut dm = Array2::<Complex64>::zeros((n_e, 3));
                        for e in 0..n_e {
                            for c in 0..3 {
                                dm[[e, c]] = if e < n_up {
                                    gu[[e, c]]
                                } else {
                                    gd[[e - n_up, c]]
                                };
                            }
                        }
                        (
                            holo_pulay_c(
                                gem,
                                jas,
                                j3,
                                &mu.view(),
                                &md.view(),
                                nuc_up,
                                nuc_dn,
                                n_atom,
                                true,
                                None,
                            ),
                            dm,
                            &base_pulay_d,
                            &base_du_d,
                        )
                    }
                };
                for d in 0..3 {
                    d_re[[gi, d]] += ratio * dang_dri[d];
                    d_r[[ia, d]] += ratio * (-dang_dri[d]);
                }
                let dm_dot_g: Complex64 = (0..3).map(|c| dmesh[[gi, c]] * g[c]).sum();
                for ee in 0..n_e {
                    if ee == gi {
                        for d in 0..3 {
                            d_re[[gi, d]] += coeff * (-dm_dot_g * uhat[d] - base_du[[gi, d]]);
                        }
                    } else {
                        for d in 0..3 {
                            d_re[[ee, d]] += coeff * (dmesh[[ee, d]] - base_du[[ee, d]]);
                        }
                    }
                }
                for ib in 0..n_atom {
                    for c in 0..3 {
                        let mut val = pm[[ib, c]] - base_pulay[[ib, c]];
                        if ib == ia {
                            val += dmesh[[gi, c]] + dm_dot_g * uhat[c];
                        }
                        d_r[[ib, c]] += coeff * val;
                    }
                }
            }
        }
    }
    (d_r.mapv(|x| x.re), d_re.mapv(|x| x.re))
}

/// Complex-twist GFMC/LRDMC HF force `d e_L^t/d(R, r_up, r_dn)` (real, Re-projected) for the
/// periodic AE/ECP twisted geminal. jnone + j12 (J1/J2) + DTN + j123ao (complex-sphe Bloch J3).
/// Mirrors `gfmc_grad_e_l_ecp_pbc`.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ecp_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let n_atom = positions.nrows();
    let cell = &ew.cell;
    let inv = inv3(cell);

    // One base inverse per HF-force fold feeds the smooth kinetic and ECP
    // non-local base fields below.
    let gi_base = crate::timed!(
        "f.c.ginv",
        crate::complex_det::inverse_complex(&crate::complex_det::compute_geminal_complex(
            gem, r_up, r_dn
        ),)
    );
    let fwd0 = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, Some(&gi_base.view()));
    // ---- continuum kinetic (det complex + J1/J2/J3/DTN), both axes ----
    let (mut d_r, mut d_re) = crate::timed!(
        "f.c.kin_base",
        complex_kinetic_force(
            gem,
            jas,
            j3,
            r_up,
            r_dn,
            cell,
            n_atom,
            nuc_up,
            nuc_dn,
            None,
            Some(&gi_base.view()),
            batch_j3,
            Some(&fwd0),
        )
    );
    // ---- ee force: bare Ewald ee (real) ----
    let dee = crate::forces_ewald::grad_eee_wrt_r(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        ew.n_max,
    );
    d_re += &dee;
    let (dei_r, dei_ion) = crate::forces_ewald::grad_eei(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    d_re += &dei_r;
    d_r += &dei_ion;
    let dii = crate::forces_ewald::grad_eii_wrt_r(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    d_r += &dii;
    // ---- ECP local periodic force (real, min-image) ----
    ecp_local_force_pbc(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, cell, &inv, ew.n_max, r_up, r_dn, &mut d_r,
        &mut d_re,
    );
    // ---- ECP non-local periodic mesh force (complex, both axes) ----
    let (nl_r, nl_re) = crate::timed!(
        "f.c.ecp_nl",
        gfmc_ecp_nonlocal_force_complex(
            gem,
            jas,
            j3,
            nuc_up,
            nuc_dn,
            positions,
            r_up,
            r_dn,
            rt,
            cell,
            &inv,
            ew.n_max,
            nuc_n,
            am_n,
            e_n,
            c_n,
            p_n,
            weights,
            grid_points,
            nn,
            dltmove,
            // Base fields use the once-built local inverse and forward.
            Some(&gi_base.view()),
            batch_j3,
            Some(&fwd0),
        )
    );
    d_r += &nl_r;
    d_re += &nl_re;

    let _ = (alat, ecp);

    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// Compatibility debug entry for [`gfmc_grad_e_l_ecp_pbc_complex`].
/// Both routes share the novar=0 measurement kernel.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_grad_e_l_ecp_pbc_complex_debug(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    gfmc_grad_e_l_ecp_pbc_complex(
        gem,
        jas,
        j3,
        nuc_up,
        nuc_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        ecp,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        batch_j3,
    )
}

// ============================================================================
// VMC (MCMC) HF + Pulay force for complex-twist (k != Gamma) ECP. This mirrors
// `vmc_grad_e_l_pbc` for the twisted complex geminal:
// the `gfmc_grad_e_l_ecp_pbc_complex` SMOOTH BASE (complex continuum kinetic +
// Ewald ee/ei/ii + ECP local + ECP non-local, each already Re-projected at the
// boundary -- the Re(A)Re(B) != Re(A*B) trap is handled INSIDE the complex kernels)
// WITHOUT the S' block. Like the real path, the VMC complex cores REUSE the sweep's running
// complex Sherman-Morrison inverse (`gi = Some`) for every base-config determinant field --
// a fresh per-config `inverse_complex` NaNs at a near-singular geminal, the AS-regularized
// swept inverse does not. `gi = None` rebuilds fresh (the GFMC callers, which carry no swept
// inverse, and the comparison oracle).
// ============================================================================

/// VMC complex-twist HF force `(de_L/dR, de_L/dr_up, de_L/dr_dn)` (real, Re-projected).
/// `ecp=false` => bare Ewald AE (no ECP local/non-local). NN=1 continuum (dltmove=false).
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_e_l_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nn: usize,
    gi: Option<&ArrayView2<Complex64>>,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let cell = &ew.cell;
    let inv = inv3(cell);
    // Far-image robustness lives INSIDE the complex geminal-value kernels (the batch complex AO
    // wraps via WrapCtx); DTN/Ewald/ECP keep RAW positions (see `vmc_grad_e_l_pbc`).
    // ---- continuum kinetic (complex det + J1/J2/J3/DTN), both axes, Re-projected ----
    // One base forward shared by the kinetic reverse and the ECP non-local mesh.
    let fwd0 = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, gi);
    let (mut d_r, mut d_re) = complex_kinetic_force(
        gem,
        jas,
        j3,
        r_up,
        r_dn,
        cell,
        n_atom,
        nuc_up,
        nuc_dn,
        None,
        gi,
        batch_j3,
        Some(&fwd0),
    );
    // ---- ee force: bare Ewald ee (RAW positions) ----
    let dee = crate::forces_ewald::grad_eee_wrt_r(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        ew.n_max,
    );
    d_re += &dee;
    let (dei_r, dei_ion) = crate::forces_ewald::grad_eei(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    d_re += &dei_r;
    d_r += &dei_ion;
    let dii = crate::forces_ewald::grad_eii_wrt_r(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    d_r += &dii;
    // ---- ECP local (real) + non-local (complex ratio, Re-projected inside) ----
    if ecp {
        ecp_local_force_pbc(
            positions, nuc_l, e_l_p, c_l_p, p_l_p, cell, &inv, ew.n_max, r_up, r_dn, &mut d_r,
            &mut d_re,
        );
        let (nl_r, nl_re) = gfmc_ecp_nonlocal_force_complex(
            gem,
            jas,
            j3,
            nuc_up,
            nuc_dn,
            positions,
            r_up,
            r_dn,
            rt,
            cell,
            &inv,
            ew.n_max,
            nuc_n,
            am_n,
            e_n,
            c_n,
            p_n,
            weights,
            grid_points,
            nn,
            false,
            gi,
            batch_j3,
            Some(&fwd0),
        );
        d_r += &nl_r;
        d_re += &nl_re;
    }
    // NO S' block (VMC continuous).
    let _ = n_e;
    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// VMC complex-twist Pulay force `(dln|Psi|/dR, dln|Psi|/dr_up, dln|Psi|/dr_dn)` =
/// Re(holomorphic det Pulay/drift) + real Jastrow (1:1 JAX `compute_grad_ln_Psi`
/// `take_real=True` / `Re(grad ln Psi)`). Each output is a SINGLE complex quantity
/// Re-projected at the very end (no two-complex-factor cross term here). No S'.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_ln_psi_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    n_atom: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> GfmcElGrads {
    let n_up = r_up.nrows();
    // Far-image robustness lives INSIDE the complex geminal-value kernels (batch complex AO
    // wraps via WrapCtx); no position wrap here (DTN uses RAW Euclidean, matching jqmc).
    let pulay = holo_pulay_c(gem, jas, j3, r_up, r_dn, nuc_up, nuc_dn, n_atom, false, gi);
    let drift = holo_drift_c(gem, jas, j3, r_up, r_dn, gi);
    let d_r = pulay.mapv(|x| x.re);
    let d_re = drift.mapv(|x| x.re);
    let d_rup = d_re.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let d_rdn = d_re.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    GfmcElGrads { d_r, d_rup, d_rdn }
}

/// Complex-twist holomorphic ln|Psi| STRAIN (Pulay stress, un-Re'd) `(3,3)` complex: complex
/// determinant Pulay stress + real Jastrow value strain (J1/J2 + DTN). For the complex GFMC
/// stress mesh S' term / ECP-nl mesh (whose hop weight is complex, so Re-project at the boundary).
#[allow(
    clippy::too_many_arguments,
    reason = "explicit wavefunction, cell and configuration inputs"
)]
fn holo_spp_c(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    det_only: bool,
    gi: Option<&ArrayView2<Complex64>>,
    // The DTN value strain at this config when the caller already has it (the mesh
    // candidates: base + one-electron move delta); `None` evaluates it here.
    dtn_strain: Option<&Array2<f64>>,
) -> Array2<Complex64> {
    let mut out = crate::complex_stress::pbc_det_pulay_stress_complex(
        gem, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, gi,
    );
    if !det_only {
        let jo = jas_strain_c(jas, j3, r_up, r_dn, positions, dtn_strain);
        for c in 0..3 {
            for d in 0..3 {
                out[[c, d]] += Complex64::new(jo[[c, d]], 0.0);
            }
        }
    }
    out
}

/// The Jastrow part of [`holo_spp_c`]: J1/J2/DTN/J3 value strain (real 3x3) via the
/// cell-straining kernels. The twist strain oracle (`_e_L_t_ecp_strained` ->
/// `ham._apply_strain`) deforms every periodic Jastrow metric with the physical lattice.
fn jas_strain_c(
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    dtn_strain: Option<&Array2<f64>>,
) -> Array2<f64> {
    {
        let mut jo = Array2::<f64>::zeros((3, 3));
        crate::stress_pbc::jastrow_strain(
            jas.one_body.as_ref(),
            jas.two_body.as_ref(),
            r_up,
            r_dn,
            &mut jo,
        );
        if let Some(d) = &jas.dtn {
            match dtn_strain {
                Some(sd) => jo += sd,
                None => jo += &crate::stress_pbc::dtn_value_strain(d, r_up, r_dn),
            }
        }
        if let Some(c) = j3 {
            jo += &crate::complex_stress::j3_value_strain_complex(
                c.orb, &c.jm, r_up, r_dn, positions, c.home, c.kernel,
            );
        }
        jo
    }
}

/// Complex-twist ECP non-local periodic mesh STRESS `dV_ecp_nl/deps` (real 3x3), NN-filtered,
/// geometry strain with fixed quadrature directions, Re-projected. Mirror of the real
/// `gfmc_stress_ecp_pbc_withj` ECP-nl inline loop with the complex determinant ratio + holomorphic
/// Pulay stress / drift. jnone/j12/DTN/complex J3.
///
/// `dltmove` (DLA) carries the SAME split as the force twin `gfmc_ecp_nonlocal_force_complex` and
/// the real stress twin `gfmc_stress_ecp_pbc_withj`: the fixed-node-clamped hop is selected on the
/// DETERMINANT-ONLY amplitude `Re(det_ratio * ang)`, and the SP (non-accepted, `det_amp >= 0`)
/// candidates drop the Jastrow from BOTH the hop weight (`ratio = det_ratio`, no `jr`) and the
/// `d ln ratio / d eps` chain (det-only strain-Pulay `holo_spp_c(det_only=true)` + det-only drift,
/// each against the matching det-only BASE). `tmove` keeps `use_full` always. Without the split a
/// `dltmove` run silently returns the t-move stress for the DLA Hamiltonian.
///
/// Fixed-phase convention: every factor here stays HOLOMORPHIC (complex `det_ratio`, complex
/// strain-Pulay / drift) and `Re` is taken ONCE at the very end (`sigma.mapv(|x| x.re)`), because
/// `Re(A)*Re(B) != Re(A*B)`. The `use_full` PREDICATE is the one deliberate real projection: it is
/// a branch on the fixed-node sign of the discretised hop amplitude, matching the XLA
/// `jnp.real(...)` clamp, not a factor in the product.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gfmc_ecp_nonlocal_strain_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    // Optional base-config complex inverse. `None` builds one locally.
    gi: Option<&ArrayView2<Complex64>>,
    fast_mesh: bool,
    // PURE PW/B-spline complex J3 leaf: enters the reflected VALUE ratio (`jval`) + electron drift
    // (`holo_drift_c`); its VALUE strain-Pulay is 0 (affine-invariant), so `holo_spp_c` is untouched.
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
    // The base complex geminal forward at `gi` when the caller already has it.
    base_fwd: Option<&crate::complex_det::ComplexGemFwd>,
    // The base Jastrow value strain (`jas_strain_c` at the base config) when the caller has it.
    base_jas: Option<&Array2<f64>>,
    // Base det bars `(W G^-1, (G^-1 Q)[:n_dn]^T)` whose det VALUE strain (real part) the caller
    // also wants (the Pulay stress): evaluated in the same final base-table pass, returned second.
    pp_bars: Option<(&Array2<Complex64>, &Array2<Complex64>)>,
) -> (Array2<f64>, Option<Array2<f64>>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let j1 = jas.one_body.as_ref();
    let j2 = jas.two_body.as_ref();
    let dtn = jas.dtn.as_ref();
    let j3c: Option<&ComplexOrb> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.orb),
        (None, Some((o, _))) => Some(o),
        (None, None) => None,
    };
    let j3m_owned: Option<Array2<f64>> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.jm.to_owned()),
        (None, Some((_, m))) => Some(m.to_owned()),
        (None, None) => None,
    };
    let j3m = j3m_owned.as_ref().map(|a| a.view());
    let with_j =
        j1.is_some() || j2.is_some() || dtn.is_some() || j3.is_some() || batch_j3.is_some();
    // Add the PURE leaf J3 electron drift onto a holo-drift (`holo_drift_c` sees only cart `j3`).
    let add_batch_drift = |mut du: Array2<Complex64>,
                           ru: &ArrayView2<f64>,
                           rd: &ArrayView2<f64>|
     -> Array2<Complex64> {
        if let Some((orb, jm)) = batch_j3 {
            let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(orb, jm, ru, rd);
            let nu = ru.nrows();
            for e in 0..du.nrows() {
                for c in 0..3 {
                    du[[e, c]] += Complex64::new(
                        if e < nu {
                            g3u[[e, c]]
                        } else {
                            g3d[[e - nu, c]]
                        },
                        0.0,
                    );
                }
            }
        }
        du
    };
    let elec = |gi: usize| -> [f64; 3] {
        if gi < n_up {
            [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
        } else {
            let k = gi - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let nv = weights.len();
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
        let mut u: Vec<usize> = nuc_n.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |re: [f64; 3]| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut dd: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let d = min_image(
                    [
                        re[0] - positions[[ia, 0]],
                        re[1] - positions[[ia, 1]],
                        re[2] - positions[[ia, 2]],
                    ],
                    cell,
                    inv,
                    n_max,
                );
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ia)
            })
            .collect();
        dd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dd.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    // Base det anchor of the full-recompute ratio (the rank-1 path below never reads it).
    let det0 = if fast_mesh && gi.is_some() {
        (Complex64::new(1.0, 0.0), 0.0)
    } else {
        det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn))
    };
    // det-only per-electron drift at a config: the complex determinant grad field alone (no
    // Jastrow), stacked up|dn into one (n_e,3) block so the mesh loop indexes it like
    // `holo_drift_c`. `g` = that config's complex inverse when one is available (base `gi`, or
    // the rank-1-updated candidate inverse under `fast_mesh`), else `None` = fresh.
    let det_only_drift = |ru: &ArrayView2<f64>,
                          rd: &ArrayView2<f64>,
                          g: Option<&ArrayView2<Complex64>>|
     -> Array2<Complex64> {
        let (gu, gd, _, _) = det_grads_lap_c(gem, ru, rd, g);
        let mut o = Array2::<Complex64>::zeros((n_e, 3));
        for e in 0..n_e {
            for c in 0..3 {
                o[[e, c]] = if e < n_up {
                    gu[[e, c]]
                } else {
                    gd[[e - n_up, c]]
                };
            }
        }
        o
    };
    // With `fast_mesh=true` and a base complex inverse, every reflected mesh
    // candidate is a SINGLE-electron hop, i.e. a rank-1 update of the complex geminal, so the
    // candidate det ratio AND the candidate inverse follow in closed form from the base inverse
    // (Sherman-Morrison, O(N^2)) instead of a fresh `compute_geminal_complex` + `det_phase_lnabs` +
    // two `inverse_complex` O(N^3) rebuilds inside the candidate Pulay-stress / drift. The
    // streaming state (`DetRatioStateC`) is built ONCE for the whole mesh loop.
    // `fast_mesh=false` keeps the full per-candidate recompute -- the comparison oracle, and the
    // only path GFMC uses (it has no swept base inverse).
    let sm_base: Option<(crate::complex_det::DetRatioStateC, Array2<Complex64>)> =
        match (fast_mesh, gi) {
            (true, Some(g0)) => Some((
                crate::complex_det::init_det_ratio_state_complex(gem, r_up, r_dn),
                g0.to_owned(),
            )),
            _ => None,
        };
    // Rank-1 path: the base forward's `W` / `Q` give the moved electron's det drift directly
    // from the candidate inverse (`gD_up[li] = gou_new . (W G'^-1)[:, li]`,
    // `gD_dn[li] = godn_new . (G'^-1 Q)[li, :]`; an up move leaves `W`, a dn move `Q`,
    // unchanged), and the DTN value strain is base + one-electron move delta -- instead of a
    // full determinant/Jastrow drift and a full DTN strain sweep per candidate.
    let base_wq = sm_base.as_ref().map(|(_, g0)| match base_fwd {
        Some(f) => (f.w.clone(), f.q.clone()),
        None => {
            let f = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, Some(&g0.view()));
            (f.w, f.q)
        }
    });
    // Real J1/J2/DTN Jastrow without a J3 leaf (rank-1 path): every candidate's value ratio,
    // moved drift row and value strain from one-electron deltas (`MeshJasStressPbc`, the real
    // stress twin), instead of a full Jastrow value / gradient / strain per candidate.
    let mesh_jas = match (&sm_base, j3c) {
        (Some(_), None) if with_j && std::env::var("JQMC_RS_ECPF_SLOW").is_err() => {
            MeshJasStressPbc::new(
                jas,
                None,
                None,
                cell,
                positions,
                r_up,
                r_dn,
                positions.nrows(),
            )
        }
        _ => None,
    };
    // Without the one-electron Jastrow mesh (a J3 leaf, or no swept base inverse), the
    // candidate Jastrow ratio is the `jastrow_ratio`-zone single-move ratio, as in the twist
    // ECP energy / parameter meshes; the difference of two whole `jastrow_value_complex`
    // values rounds in `jastrow_eval` (float32 in mixed), issue #757.
    let jas_ratio_state = (with_j && mesh_jas.is_none())
        .then(|| crate::gfmc::ComplexJasRatioState::new(jas, j3c, j3m.as_ref(), r_up, r_dn));
    let dtn_base_strain = match (&sm_base, dtn, &mesh_jas) {
        (Some(_), Some(d), None) => Some(crate::stress_pbc::dtn_value_strain(d, r_up, r_dn)),
        _ => None,
    };
    // Rank-1 path: the candidate det strain is accumulated as weighted FULL bars (`det_acc`);
    // subtracting the total weight times the base bars there (`tot_w`, before the final call)
    // removes the base det strain exactly, so the base strain Pulay carries only the Jastrow
    // part (and the det-only base is zero) instead of two base det strain sweeps.
    let rank1 = sm_base.is_some() && base_wq.is_some();
    // Grouped rank-1 mode: every candidate's det ratio, moved drift row and moved-column strain
    // from ONE fused pass of the moved electron per group
    // ([`crate::complex_stress::moved_group_complex`]), and the other bar columns as the
    // LINEAR bar deltas of the group ([`sm_linear_bars`]) instead of a rank-1 inverse and two
    // bar products per candidate.
    let geom_s = |orb: &ComplexOrb,
                  home: &[usize],
                  kern: &[usize]|
     -> Option<Option<crate::mesh_vgs::ShellGeom>> {
        if orb.aos.num_ao == 0 {
            return Some(None);
        }
        crate::mesh_vgs::ShellGeom::new(&orb.aos, home, Some((kern, positions))).map(Some)
    };
    let grouped_geoms = if rank1
        && crate::complex_stress::moved_group_ok(&gem.orb_up)
        && crate::complex_stress::moved_group_ok(&gem.orb_dn)
        && std::env::var("JQMC_RS_ECPF_SLOW").is_err()
    {
        match (
            geom_s(&gem.orb_up, home_up, kernel_up),
            geom_s(&gem.orb_dn, home_dn, kernel_dn),
        ) {
            (Some(gu), Some(gd)) => Some((gu, gd)),
            _ => None,
        }
    } else {
        None
    };
    let base_mo = grouped_geoms.as_ref().map(|_| match base_fwd {
        Some(fw) => (fw.ou.clone(), fw.od.clone(), fw.lam_p.clone()),
        None => {
            let fw = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, gi);
            (fw.ou, fw.od, fw.lam_p)
        }
    });
    let base_pulay_f = if rank1 {
        match base_jas {
            Some(b) => b.mapv(|x| Complex64::new(x, 0.0)),
            None => {
                jas_strain_c(jas, j3, r_up, r_dn, positions, None).mapv(|x| Complex64::new(x, 0.0))
            }
        }
    } else {
        holo_spp_c(
            gem, jas, j3, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, false, gi,
            None,
        )
    };
    // det-only strain-Pulay + det-only drift BASE (dltmove SP branch only; `tmove` never reads
    // them). Both take the SAME base SM inverse `gi` as the full base -- `holo_spp_c` /
    // `det_grads_lap_c` thread it into the complex det field either way, so the det-only branch
    // is NOT demoted to the NaN-prone fresh-LU path.
    let base_pulay_d = if dltmove && rank1 {
        Array2::<Complex64>::zeros((3, 3))
    } else if dltmove {
        holo_spp_c(
            gem, jas, j3, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, true, gi,
            None,
        )
    } else {
        Array2::zeros((0, 0))
    };
    let mut tot_w = Complex64::new(0.0, 0.0);
    let mut sigma = Array2::<Complex64>::zeros((3, 3));
    // Weighted candidate-bar accumulators of the rank-1 det strain (all columns, per spin).
    let mut det_acc = base_wq.as_ref().map(|(w0, q0)| {
        (
            Array2::<Complex64>::zeros(w0.raw_dim()),
            Array2::<Complex64>::zeros((q0.ncols(), r_dn.nrows())),
        )
    });
    for gi_e in 0..n_e {
        let re = elec(gi_e);
        let is_up_e = gi_e < n_up;
        let li_e = if is_up_e { gi_e } else { gi_e - n_up };
        let (orb_e, home_e, kern_e) = if is_up_e {
            (&gem.orb_up, home_up, kernel_up)
        } else {
            (&gem.orb_dn, home_dn, kernel_dn)
        };
        for &ia in &nearest_set(re) {
            let rel0 = [
                positions[[ia, 0]] - re[0],
                positions[[ia, 1]] - re[1],
                positions[[ia, 2]] - re[2],
            ];
            let rel = min_image(rel0, cell, inv, n_max);
            let nrm = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            // Per-group moved-column accumulators: weighted candidate bar columns at the new
            // positions, their sum at the base position (the li-column table swap).
            let pts: Vec<[f64; 3]> = gp
                .iter()
                .map(|vg| {
                    [
                        re[0] + rel[0] + nrm * vg[0],
                        re[1] + rel[1] + nrm * vg[1],
                        re[2] + rel[2] + nrm * vg[2],
                    ]
                })
                .collect();
            // Moved-electron orbital values (the value convention of the SM base) and gradients
            // of the whole group, one batched evaluation each (rank-1 path).
            let grp_mv = match (&grouped_geoms, &sm_base, &base_wq) {
                (Some((gu, gd)), Some((_, g0)), Some((w0, q0))) => {
                    let col = if is_up_e {
                        w0.dot(&g0.column(li_e))
                    } else {
                        g0.row(li_e).dot(q0)
                    };
                    let geom = if is_up_e { gu.as_ref() } else { gd.as_ref() };
                    Some(
                        crate::complex_stress::moved_group_complex(
                            orb_e,
                            &pts,
                            &col,
                            geom,
                            0,
                            true,
                            AoScreen::MeshValue,
                        )
                        .expect("grouped complex mesh strain: leaf covered"),
                    )
                }
                _ => None,
            };
            let grp_orb = sm_base.as_ref().filter(|_| grp_mv.is_none()).map(|_| {
                let rmat = Array2::from_shape_vec(
                    (pts.len(), 3),
                    pts.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
                )
                .unwrap();
                let vals = crate::complex_det::complex_candidate_batch(orb_e, &pts);
                // Only the moved bar column contracts the gradients (the drift row): the
                // contracted form when the leaf is covered, else the gradient tables.
                let contracted = match (&sm_base, &base_wq) {
                    (Some((_, g0)), Some((w0, q0))) => {
                        let col = if is_up_e {
                            w0.dot(&g0.column(li_e))
                        } else {
                            g0.row(li_e).dot(q0)
                        };
                        crate::complex_stress::contracted_grad_complex(orb_e, &rmat.view(), &col)
                    }
                    _ => None,
                };
                let grads = match contracted {
                    Some(_) => None,
                    None => Some(orb_e.value_grad_lap(&rmat.view()).1),
                };
                (vals, grads, contracted)
            });
            // One-electron Jastrow fields of the whole group (rank-1 path, no J3 leaf).
            let grp_jas = mesh_jas
                .as_ref()
                .map(|m| m.stress_candidates(is_up_e, li_e, &pts, r_up, r_dn));
            // One-electron DTN value-strain deltas of the whole group (rank-1 path only).
            let grp_dtn = dtn_base_strain.as_ref().map(|_| {
                crate::stress_pbc::dtn_value_strain_move_delta_batch(
                    dtn.unwrap(),
                    r_up,
                    r_dn,
                    is_up_e,
                    li_e,
                    &pts,
                )
            });
            let mut grp_acc = det_acc.as_ref().map(|(au, ad)| {
                let m = if is_up_e { au.nrows() } else { ad.nrows() };
                (
                    Array2::<Complex64>::zeros((m, nv)),
                    Array2::<Complex64>::zeros((m, 1)),
                    Array2::<f64>::zeros((nv, 3)),
                )
            });
            // The moved candidate bar column is the base column over the det ratio, so the
            // group's weighted moved columns are `s_v * base_col` (`s_v = wgt_v / det_ratio_v`).
            let grp_col = match (&sm_base, &base_wq) {
                (Some((_, g0)), Some((w0, q0))) => Some(if is_up_e {
                    w0.dot(&g0.column(li_e))
                } else {
                    g0.row(li_e).dot(q0)
                }),
                _ => None,
            };
            let mut grp_s = vec![Complex64::new(0.0, 0.0); nv];
            for v in 0..nv {
                let vg = gp[v];
                let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                let cth = -(rel[0] * vg[0] + rel[1] * vg[1] + rel[2] * vg[2]) / (nrm * vgn);
                let (mut ang, mut aprime, mut acos) = (0.0_f64, 0.0_f64, 0.0_f64);
                for t in 0..nuc_n.len() {
                    if nuc_n[t] as usize != ia {
                        continue;
                    }
                    let l = am_n[t];
                    let (pe, be, ce) = (p_n[t], e_n[t], c_n[t]);
                    let expf = (-be * nrm * nrm).exp();
                    let vl = ce * nrm.powf(pe - 2.0) * expf;
                    let vlp = ce
                        * ((pe - 2.0) * nrm.powf(pe - 3.0) - 2.0 * be * nrm.powf(pe - 1.0))
                        * expf;
                    let (pl, dpl) = leg_and_deriv(l, cth);
                    let wv = ((2 * l + 1) as f64) * weights[v];
                    ang += vl * wv * pl;
                    aprime += vlp * wv * pl;
                    acos += vl * wv * dpl;
                }
                let rp = [
                    re[0] + rel[0] + nrm * vg[0],
                    re[1] + rel[1] + nrm * vg[1],
                    re[2] + rel[2] + nrm * vg[2],
                ];
                let (mu, md) = {
                    let (mut mu, mut md) = (r_up.to_owned(), r_dn.to_owned());
                    if gi_e < n_up {
                        mu[[gi_e, 0]] = rp[0];
                        mu[[gi_e, 1]] = rp[1];
                        mu[[gi_e, 2]] = rp[2];
                    } else {
                        let k = gi_e - n_up;
                        md[[k, 0]] = rp[0];
                        md[[k, 1]] = rp[1];
                        md[[k, 2]] = rp[2];
                    }
                    (mu, md)
                };
                let (det_ratio, cand_inv, ao_moved) = match &sm_base {
                    Some(_) if grp_mv.is_some() => (grp_mv.as_ref().unwrap().val[v], None, None),
                    Some((st, g0)) => {
                        let n = g0.nrows();
                        let is_up = gi_e < n_up;
                        let li = if is_up { gi_e } else { gi_e - n_up };
                        let ao_new: Vec<Complex64> = grp_orb.as_ref().unwrap().0.row(v).to_vec();
                        let (u, v): (Vec<Complex64>, Vec<Complex64>) = if is_up {
                            let v =
                                crate::complex_det::compute_v_up_move_complex(gem, st, li, &ao_new);
                            let mut u = vec![Complex64::new(0.0, 0.0); n];
                            u[li] = Complex64::new(1.0, 0.0);
                            (u, v)
                        } else {
                            let u = crate::complex_det::compute_u_dn_move_complex(st, li, &ao_new);
                            let mut v = vec![Complex64::new(0.0, 0.0); n];
                            v[li] = Complex64::new(1.0, 0.0);
                            (u, v)
                        };
                        let mut ainv_u = vec![Complex64::new(0.0, 0.0); n];
                        let mut vt_ainv = vec![Complex64::new(0.0, 0.0); n];
                        let r = crate::complex_det::sm_ratio_complex(
                            &g0.view(),
                            &u,
                            &v,
                            &mut ainv_u,
                            &mut vt_ainv,
                        );
                        // G'^-1 = G^-1 - (G^-1 u)(v^T G^-1)/ratio (exact rank-1 inverse update).
                        let mut gc = g0.clone();
                        for i in 0..n {
                            let ai = ainv_u[i] / r;
                            for j in 0..n {
                                let cur = gc[[i, j]];
                                gc[[i, j]] = cur - ai * vt_ainv[j];
                            }
                        }
                        (r, Some(gc), Some(ao_new))
                    }
                    None => (
                        det_ratio_complex(
                            det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), &md.view())),
                            det0,
                        ),
                        None,
                        None,
                    ),
                };
                let cand_gi = cand_inv.as_ref().map(|m| m.view());
                let jr = match (grp_jas.as_ref(), jas_ratio_state.as_ref()) {
                    (Some(g), _) => g[v].0,
                    (None, Some(st)) => st.ratio(jas, j3c, r_up, r_dn, is_up_e, li_e, rp),
                    (None, None) => 1.0,
                };
                // tmove -> full det*J ratio; dltmove -> det-only (SP, det_amp >= 0) / full (FN).
                let det_amp = det_ratio * ang;
                let use_full = !dltmove || det_amp.re < 0.0;
                let ratio = if use_full { det_ratio * jr } else { det_ratio };
                // Mesh strain-Pulay + drift + BASE must match the ratio's Jastrow inclusion, or the
                // SP candidates would mix a det-only weight with a det+J `d ln ratio / d eps`.
                // BOTH branches take the candidate inverse `cand_gi` (the rank-1 update of the base
                // inverse under `fast_mesh`, `None` under the full-recompute oracle): the det-only
                // variant differs from the full one only by DROPPING the Jastrow strain / drift,
                // never by which complex det inverse it uses, so the rank-1 mesh path serves it unchanged.
                let (pulay_m, drift, base_pulay) = match (&cand_inv, &base_wq) {
                    (Some(_), Some(_)) | (None, Some(_))
                        if cand_inv.is_some() || grp_mv.is_some() =>
                    {
                        let (w0, q0) = base_wq.as_ref().unwrap();
                        // Rank-1 candidate: moved-electron det drift from the candidate inverse.
                        let is_up = gi_e < n_up;
                        let li = if is_up { gi_e } else { gi_e - n_up };
                        let mut drift = [Complex64::new(0.0, 0.0); 3];
                        if let Some(mv) = &grp_mv {
                            // moved bar column = base column / det ratio
                            for (d, drift_d) in drift.iter_mut().enumerate() {
                                *drift_d = mv.grad[v][d] / det_ratio;
                            }
                        } else if let Some(cg) = &grp_orb.as_ref().unwrap().2 {
                            // moved bar column = base column / det ratio (rank-1 inverse update)
                            for d in 0..3 {
                                drift[d] = cg[v][d] / det_ratio;
                            }
                        } else {
                            let gc = cand_inv.as_ref().unwrap();
                            let gg = grp_orb.as_ref().unwrap().1.as_ref().unwrap();
                            let g_new = [
                                gg[0].slice(numpy::ndarray::s![.., v..v + 1]),
                                gg[1].slice(numpy::ndarray::s![.., v..v + 1]),
                                gg[2].slice(numpy::ndarray::s![.., v..v + 1]),
                            ];
                            if is_up {
                                // (W G'^-1)[:, li]
                                let col: Vec<Complex64> = (0..w0.nrows())
                                    .map(|a| {
                                        (0..w0.ncols()).map(|j| w0[[a, j]] * gc[[j, li]]).sum()
                                    })
                                    .collect();
                                for d in 0..3 {
                                    drift[d] =
                                        (0..col.len()).map(|a| g_new[d][[a, 0]] * col[a]).sum();
                                }
                            } else {
                                // (G'^-1 Q)[li, :]
                                let row: Vec<Complex64> = (0..q0.ncols())
                                    .map(|b| {
                                        (0..q0.nrows()).map(|j| gc[[li, j]] * q0[[j, b]]).sum()
                                    })
                                    .collect();
                                for d in 0..3 {
                                    drift[d] =
                                        (0..row.len()).map(|b| g_new[d][[b, 0]] * row[b]).sum();
                                }
                            }
                        }
                        let dtn_moved = dtn_base_strain
                            .as_ref()
                            .zip(grp_dtn.as_ref())
                            .map(|(b, dl)| b + &dl[v]);
                        // Jastrow strain only; the candidate det strain is LINEAR in the candidate
                        // bars and column-separable, so its weighted sum is accumulated below and
                        // evaluated once per group (moved column) / once per call (all columns).
                        let pm = if use_full {
                            match grp_jas.as_ref() {
                                Some(g) => g[v].2.mapv(|x| Complex64::new(x, 0.0)),
                                None => jas_strain_c(
                                    jas,
                                    j3,
                                    &mu.view(),
                                    &md.view(),
                                    positions,
                                    dtn_moved.as_ref(),
                                )
                                .mapv(|x| Complex64::new(x, 0.0)),
                            }
                        } else {
                            Array2::<Complex64>::zeros((3, 3))
                        };
                        if grp_mv.is_some() {
                            grp_s[v] = ratio * Complex64::new(ang, 0.0) / det_ratio;
                        } else {
                            let gc = cand_inv.as_ref().unwrap();
                            let ao_new = ao_moved.as_ref().unwrap();
                            let aon =
                                Array2::from_shape_vec((ao_new.len(), 1), ao_new.clone()).unwrap();
                            let n_dn = r_dn.nrows();
                            let (wg, bar_dn) = if is_up {
                                // up move: Q row li <- new paired orbital row, W unchanged
                                let qrow = gem.up_t_times_paired(&aon);
                                let mut q2 = q0.clone();
                                q2.row_mut(li).assign(&qrow.row(0));
                                let gq = gc.dot(&q2);
                                (
                                    w0.dot(gc),
                                    gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(),
                                )
                            } else {
                                // dn move: W column li <- new paired orbital column, Q unchanged
                                let wcol = gem.paired_times(&aon);
                                let mut w2 = w0.clone();
                                w2.column_mut(li).assign(&wcol.column(0));
                                let gq = gc.dot(q0);
                                (
                                    w2.dot(gc),
                                    gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(),
                                )
                            };
                            let wgt = ratio * Complex64::new(ang, 0.0);
                            grp_s[v] = wgt / det_ratio;
                            tot_w += wgt;
                            let (acc_up, acc_dn) = det_acc.as_mut().unwrap();
                            acc_up.scaled_add(wgt, &wg);
                            acc_dn.scaled_add(wgt, &bar_dn);
                            let bmov = if is_up { &wg } else { &bar_dn };
                            let (g_new, g_li, g_pos) = grp_acc.as_mut().unwrap();
                            for a in 0..bmov.nrows() {
                                let x = wgt * bmov[[a, li]];
                                g_new[[a, v]] = x;
                                g_li[[a, 0]] += x;
                            }
                            for c in 0..3 {
                                g_pos[[v, c]] = rp[c];
                            }
                        }
                        let full_grp_jas = if use_full { grp_jas.as_ref() } else { None };
                        if let Some(gj) = full_grp_jas {
                            let jd = gj[v].1;
                            for c in 0..3 {
                                drift[c] += Complex64::new(jd[c], 0.0);
                            }
                            (pm, drift, &base_pulay_f)
                        } else if use_full {
                            let (gj_up, gj_dn) = jas_drift_c(jas, j3, &mu.view(), &md.view());
                            let (bu, bd) = match batch_j3 {
                                Some((o, m)) => {
                                    let (g3u, g3d, _, _) = crate::complex_det::j3_grads_lap_complex(
                                        o,
                                        m,
                                        &mu.view(),
                                        &md.view(),
                                    );
                                    (Some(g3u), Some(g3d))
                                }
                                None => (None, None),
                            };
                            for c in 0..3 {
                                let (gj, gb) = if is_up {
                                    (gj_up[[li, c]], bu.as_ref().map_or(0.0, |g| g[[li, c]]))
                                } else {
                                    (gj_dn[[li, c]], bd.as_ref().map_or(0.0, |g| g[[li, c]]))
                                };
                                drift[c] += Complex64::new(gj, 0.0);
                                drift[c] += Complex64::new(gb, 0.0);
                            }
                            (pm, drift, &base_pulay_f)
                        } else {
                            (pm, drift, &base_pulay_d)
                        }
                    }
                    _ => {
                        // Full per-candidate recompute (the comparison oracle).
                        let spp = |det_only: bool| {
                            holo_spp_c(
                                gem,
                                jas,
                                j3,
                                &mu.view(),
                                &md.view(),
                                positions,
                                home_up,
                                kernel_up,
                                home_dn,
                                kernel_dn,
                                det_only,
                                cand_gi.as_ref(),
                                None,
                            )
                        };
                        let (pulay_m, dcfg, base_pulay) = if use_full {
                            (
                                spp(false),
                                add_batch_drift(
                                    holo_drift_c(
                                        gem,
                                        jas,
                                        j3,
                                        &mu.view(),
                                        &md.view(),
                                        cand_gi.as_ref(),
                                    ),
                                    &mu.view(),
                                    &md.view(),
                                ),
                                &base_pulay_f,
                            )
                        } else {
                            (
                                spp(true),
                                det_only_drift(&mu.view(), &md.view(), cand_gi.as_ref()),
                                &base_pulay_d,
                            )
                        };
                        let drift = [dcfg[[gi_e, 0]], dcfg[[gi_e, 1]], dcfg[[gi_e, 2]]];
                        (pulay_m, drift, base_pulay)
                    }
                };
                let drift_vg: Complex64 = (0..3).map(|c| drift[c] * vg[c]).sum();
                for c in 0..3 {
                    for d in 0..3 {
                        let dcth =
                            -(vg[c] * rel[d]) / (nrm * vgn) - cth * rel[c] * rel[d] / (nrm * nrm);
                        let da = aprime * rel[c] * rel[d] / nrm + acos * dcth;
                        let mesh_wf = pulay_m[[c, d]] - drift[c] * (nrm * vg[d])
                            + (rel[c] * rel[d] / nrm) * drift_vg;
                        let dln_ratio = mesh_wf - base_pulay[[c, d]];
                        sigma[[c, d]] += ratio
                            * (Complex64::new(da, 0.0) + Complex64::new(ang, 0.0) * dln_ratio);
                    }
                }
            }
            if let (Some(mv), Some((ou0, od0, lam_p)), Some((_, g0)), Some((w0, q0)), Some(col)) = (
                grp_mv.as_ref(),
                base_mo.as_ref(),
                sm_base.as_ref(),
                base_wq.as_ref(),
                grp_col.as_ref(),
            ) {
                // moved column: + the fresh fields at the candidate positions; the other columns
                // (and the base moved column) as the weighted bar deltas on the base tables.
                for (v, st) in mv.strain.iter().enumerate() {
                    for c in 0..3 {
                        for d in 0..3 {
                            sigma[[c, d]] += grp_s[v] * st[c][d];
                        }
                    }
                }
                let ssum: Complex64 = grp_s.iter().sum();
                let a = mv.aotab.dot(&ArrayView1::from(&grp_s[..]));
                let mo_new = match &orb_e.mo {
                    Some(c) => c.dot(&a),
                    None => a,
                };
                let base_o = if is_up_e {
                    ou0.column(li_e)
                } else {
                    od0.column(li_e)
                };
                let dmo = &mo_new - &base_o.mapv(|x| x * ssum);
                let (lw, ld) = sm_linear_bars(
                    is_up_e,
                    li_e,
                    &dmo,
                    w0,
                    q0,
                    g0,
                    lam_p.as_ref(),
                    r_dn.nrows(),
                );
                let (acc_up, acc_dn) = det_acc.as_mut().unwrap();
                *acc_up += &lw;
                *acc_dn += &ld;
                // the swapped-out base moved column: - Sum_v s_v * base column
                let acc = if is_up_e { acc_up } else { acc_dn };
                acc.column_mut(li_e).scaled_add(-ssum, col);
                continue;
            }
            let fused_swap = match (grp_acc.as_ref(), grp_col.as_ref()) {
                (Some((_, _, g_pos)), Some(col)) => {
                    crate::complex_stress::det_strain_side_complex_col(
                        orb_e,
                        &g_pos.view(),
                        col,
                        &grp_s,
                        home_e,
                        kern_e,
                        positions,
                    )
                }
                _ => None,
            };
            if let Some(sw) = fused_swap {
                // moved column: + fresh fields at the candidate positions (fused); - the base
                // fields at li fold into the all-column base call below (column li of `acc`).
                sigma += &sw;
                let (_, g_li, _) = grp_acc.as_ref().unwrap();
                let (acc_up, acc_dn) = det_acc.as_mut().unwrap();
                let acc = if is_up_e { acc_up } else { acc_dn };
                for a in 0..acc.nrows() {
                    acc[[a, li_e]] -= g_li[[a, 0]];
                }
            } else if let Some((g_new, g_li, g_pos)) = grp_acc.as_ref() {
                // moved column: + fresh fields at the candidate positions - base fields at li
                crate::complex_stress::det_strain_side_complex(
                    orb_e,
                    &g_pos.view(),
                    g_new,
                    home_e,
                    kern_e,
                    positions,
                    &mut sigma,
                );
                let r_li = if is_up_e {
                    r_up.slice(numpy::ndarray::s![li_e..li_e + 1, ..])
                } else {
                    r_dn.slice(numpy::ndarray::s![li_e..li_e + 1, ..])
                };
                let mut sub = Array2::<Complex64>::zeros((3, 3));
                crate::complex_stress::det_strain_side_complex(
                    orb_e, &r_li, g_li, home_e, kern_e, positions, &mut sub,
                );
                sigma -= &sub;
            }
        }
    }
    if let (true, Some((acc_up, acc_dn)), Some((_, g0)), Some((w0, q0))) = (
        rank1 && grouped_geoms.is_none(),
        det_acc.as_mut(),
        sm_base.as_ref(),
        base_wq.as_ref(),
    ) {
        let n_dn = r_dn.nrows();
        acc_up.scaled_add(-tot_w, &w0.dot(g0));
        let bdn = g0
            .dot(q0)
            .slice(numpy::ndarray::s![0..n_dn, ..])
            .t()
            .to_owned();
        acc_dn.scaled_add(-tot_w, &bdn);
    }
    let mut sigma = sigma.mapv(|x| x.re);
    let mut pp = None;
    match (det_acc.as_ref(), pp_bars) {
        (Some((acc_up, acc_dn)), Some((pu, pd))) => {
            // every column at the base positions, once for the whole mesh, together with the
            // caller's base bars (real parts only are read)
            let [hu, ppu] = crate::complex_stress::det_strain_side_complex_re_pair(
                &gem.orb_up,
                r_up,
                [acc_up, pu],
                home_up,
                kernel_up,
                positions,
            );
            let [hd, ppd] = crate::complex_stress::det_strain_side_complex_re_pair(
                &gem.orb_dn,
                r_dn,
                [acc_dn, pd],
                home_dn,
                kernel_dn,
                positions,
            );
            sigma += &hu;
            sigma += &hd;
            pp = Some(ppu + &ppd);
        }
        (Some((acc_up, acc_dn)), None) => {
            let mut acc = Array2::<Complex64>::zeros((3, 3));
            crate::complex_stress::det_strain_side_complex(
                &gem.orb_up,
                r_up,
                acc_up,
                home_up,
                kernel_up,
                positions,
                &mut acc,
            );
            crate::complex_stress::det_strain_side_complex(
                &gem.orb_dn,
                r_dn,
                acc_dn,
                home_dn,
                kernel_dn,
                positions,
                &mut acc,
            );
            sigma += &acc.mapv(|x| x.re);
        }
        (None, Some((pu, pd))) => {
            let mut acc = Array2::<Complex64>::zeros((3, 3));
            crate::complex_stress::det_strain_side_complex(
                &gem.orb_up,
                r_up,
                pu,
                home_up,
                kernel_up,
                positions,
                &mut acc,
            );
            crate::complex_stress::det_strain_side_complex(
                &gem.orb_dn,
                r_dn,
                pd,
                home_dn,
                kernel_dn,
                positions,
                &mut acc,
            );
            pp = Some(acc.mapv(|x| x.re));
        }
        (None, None) => {}
    }
    (sigma, pp)
}

/// Body of [`gfmc_stress_ecp_pbc_complex`] and [`gfmc_stress_ecp_pbc_complex_with_pp`].
#[allow(clippy::too_many_arguments)]
fn gfmc_stress_ecp_pbc_complex_core(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
    // `Some(gi_pp)`: also return the Pulay stress `dln|Psi|/deps` (as
    // `vmc_grad_ln_psi_eps_pbc_complex` at the inverse `gi_pp`, `None` = this kernel's fresh
    // base inverse) from the same base forward and final base-table pass.
    want_pp: Option<Option<&ArrayView2<Complex64>>>,
) -> (Array2<f64>, Option<Array2<f64>>) {
    let cell = &ew.cell;
    let inv = inv3(cell);
    let nmax = ew.n_max;
    // A cart-lowerable J3 (`j3`, home Some) OR a PURE PW/B-spline complex J3 leaf (`batch_j3`, home
    // None): the latter feeds `complex_kinetic_stress_w` with j3c/j3m Some but home/kernel None, so
    // its kinetic SELF-strain routes through the leaf-owning `j3_leaf_kstress_side_complex`. The
    // j_matrix is owned locally (a small copy, once) to reconcile the `j3` / `batch_j3` view
    // lifetimes (`ArrayView` is lifetime-invariant).
    let j3c = match (j3, batch_j3) {
        (Some(c), _) => Some(c.orb),
        (None, Some((orb, _))) => Some(orb),
        (None, None) => None,
    };
    let j3m_owned: Option<Array2<f64>> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.jm.to_owned()),
        (None, Some((_, jm))) => Some(jm.to_owned()),
        (None, None) => None,
    };
    let (j3_home, j3_kernel) = match j3 {
        Some(c) => (Some(c.home), Some(c.kernel)),
        None => (None, None),
    };
    let j3m_v = j3m_owned.as_ref().map(|a| a.view());
    let j3m = j3m_v.as_ref();
    // One fresh base inverse and one base forward, shared by the kinetic and the non-local
    // rank-1 mesh path.
    let gi_base = crate::complex_det::inverse_complex(
        &crate::complex_det::compute_geminal_complex(gem, r_up, r_dn),
    );
    let fwd0 = crate::complex_det::complex_gem_forward(gem, r_up, r_dn, Some(&gi_base.view()));
    // ---- continuum ----
    let mut sigma = crate::complex_stress::complex_kinetic_stress_w(
        gem,
        jas,
        r_up,
        r_dn,
        cell,
        positions,
        home_up,
        kernel_up,
        home_dn,
        kernel_dn,
        j3c,
        j3m,
        j3_home,
        j3_kernel,
        None,
        Some(&gi_base.view()),
        Some(&fwd0),
    );
    // ---- ee strain: bare Ewald ee ----
    sigma += &crate::forces_ewald::grad_eee_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        nmax,
    );
    sigma += &crate::forces_ewald::grad_eei_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eii_strain(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    sigma += &crate::forces_ecp::grad_vlocal_strain(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell, nmax,
    );
    // The base Jastrow value strain, shared by the mesh base and the Pulay stress.
    let base_jas = want_pp.map(|_| jas_strain_c(jas, j3, r_up, r_dn, positions, None));
    // Pulay det bars: the shared forward's orbitals with the requested inverse.
    let pp_bars = want_pp.map(|g| {
        let n_dn = r_dn.nrows();
        match g {
            None => (
                fwd0.wg.clone(),
                fwd0.gq
                    .slice(numpy::ndarray::s![0..n_dn, ..])
                    .t()
                    .to_owned(),
            ),
            Some(g) => {
                let gq = g.dot(&fwd0.q);
                (
                    fwd0.w.dot(g),
                    gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(),
                )
            }
        }
    });
    let (nl, pp_det) = gfmc_ecp_nonlocal_strain_complex(
        gem,
        jas,
        j3,
        home_up,
        kernel_up,
        home_dn,
        kernel_dn,
        positions,
        r_up,
        r_dn,
        rt,
        cell,
        &inv,
        nmax,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        // This HF stress kernel builds one base `inverse_complex` per fold call
        // (O(N^3) once), which turns on the rank-1
        // mesh path (`sm_base`): every reflected candidate then uses Sherman-Morrison
        // O(N^2) algebra instead of a fresh `compute_geminal_complex` + `det_phase_lnabs` +
        // two `inverse_complex` rebuilds in the complex-twist GFMC full-recompute path.
        Some(&gi_base.view()),
        true,
        batch_j3,
        Some(&fwd0),
        base_jas.as_ref(),
        pp_bars.as_ref().map(|(a, b)| (a, b)),
    );
    sigma += &nl;
    let pp = match (pp_det, base_jas) {
        (Some(d), Some(j)) => Some(d + &j),
        _ => None,
    };

    let _ = (alat, ecp);
    (sigma, pp)
}

/// Complex-twist GFMC/LRDMC HF stress `d e_L^t/deps` (real 3x3) for the periodic AE/ECP twisted
/// geminal. jnone + j12 (J1/J2) + DTN + j123ao (complex-sphe Bloch J3). Mirrors `gfmc_stress_ecp_pbc_withj`.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_complex_core(
        gem,
        jas,
        j3,
        home_up,
        kernel_up,
        home_dn,
        kernel_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        ecp,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        batch_j3,
        None,
    )
    .0
}

/// [`gfmc_stress_ecp_pbc_complex`] together with the Pulay stress `dln|Psi|/deps` at the inverse
/// `gi_pp` (`None` = the HF kernel's fresh base inverse), sharing the base forward, the base
/// Jastrow strain and the final base-table pass: `(HF stress, Pulay stress)`.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_complex_with_pp(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
    gi_pp: Option<&ArrayView2<Complex64>>,
) -> (Array2<f64>, Array2<f64>) {
    let (hf, pp) = gfmc_stress_ecp_pbc_complex_core(
        gem,
        jas,
        j3,
        home_up,
        kernel_up,
        home_dn,
        kernel_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        ecp,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        batch_j3,
        Some(gi_pp),
    );
    (hf, pp.expect("Pulay stress requested"))
}

/// Compatibility debug entry for [`gfmc_stress_ecp_pbc_complex`].
/// Both routes share the novar=0 measurement kernel.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_stress_ecp_pbc_complex_debug(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> Array2<f64> {
    gfmc_stress_ecp_pbc_complex(
        gem,
        jas,
        j3,
        home_up,
        kernel_up,
        home_dn,
        kernel_dn,
        positions,
        z_eff,
        r_up,
        r_dn,
        alat,
        rt,
        ew,
        ecp,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am_n,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        batch_j3,
    )
}

/// VMC (MCMC) COMPLEX-twist Pulay stress `dln|Psi|/deps` (real 3x3, Re-projected): the
/// SMOOTH BASE of `gfmc_stress_ecp_pbc_complex`'s Pulay = `holo_spp_c` (complex det Pulay
/// stress + real J1/J2/DTN/J3 value strain), no S'. 1:1 the production
/// `pbc_stress.stress_lnPsi_pbc(take_real=True)` at a twist. `gi = Some` reuses the sweep's
/// running complex SM inverse for the det field (`None` -> fresh, the oracle).
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_ln_psi_eps_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gi: Option<&ArrayView2<Complex64>>,
) -> Array2<f64> {
    let _ = cell;
    holo_spp_c(
        gem, jas, j3, r_up, r_dn, positions, home_up, kernel_up, home_dn, kernel_dn, false, gi,
        None,
    )
    .mapv(|x| x.re)
}

/// VMC (MCMC) COMPLEX-twist HF stress `de_L/deps` (real 3x3, Re-projected): the SMOOTH BASE
/// of `gfmc_stress_ecp_pbc_complex` (complex withJ kinetic stress +
/// Ewald ee/ei/ii strain + ECP local strain + complex ECP non-local mesh stress with
/// fixed quadrature directions), WITHOUT the S' block. `ecp=false` => bare-Ewald AE.
#[allow(clippy::too_many_arguments)]
pub fn vmc_grad_e_l_eps_pbc_complex(
    gem: &ComplexGeminal,
    jas: &JastrowData,
    j3: Option<&J3PbcComplex>,
    home_up: &[usize],
    kernel_up: &[usize],
    home_dn: &[usize],
    kernel_dn: &[usize],
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ew: &crate::ewald::EwaldRef,
    ecp: bool,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am_n: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nn: usize,
    gi: Option<&ArrayView2<Complex64>>,
    fast_mesh: bool,
    batch_j3: Option<(&crate::complex_det::ComplexOrb, &ArrayView2<f64>)>,
) -> Array2<f64> {
    let cell = &ew.cell;
    let inv = inv3(cell);
    let nmax = ew.n_max;
    // A cart-lowerable J3 (`j3`, home Some) OR a PURE PW/B-spline complex J3 leaf (`batch_j3`, home
    // None): the latter feeds `complex_kinetic_stress_w` with j3c/j3m Some but home/kernel None, so
    // its kinetic SELF-strain routes through the leaf-owning `j3_leaf_kstress_side_complex`. The
    // j_matrix is owned locally (a small copy, once) to reconcile the `j3` / `batch_j3` view
    // lifetimes (`ArrayView` is lifetime-invariant).
    let j3c = match (j3, batch_j3) {
        (Some(c), _) => Some(c.orb),
        (None, Some((orb, _))) => Some(orb),
        (None, None) => None,
    };
    let j3m_owned: Option<Array2<f64>> = match (j3, batch_j3) {
        (Some(c), _) => Some(c.jm.to_owned()),
        (None, Some((_, jm))) => Some(jm.to_owned()),
        (None, None) => None,
    };
    let (j3_home, j3_kernel) = match j3 {
        Some(c) => (Some(c.home), Some(c.kernel)),
        None => (None, None),
    };
    let j3m_v = j3m_owned.as_ref().map(|a| a.view());
    let j3m = j3m_v.as_ref();
    // One base forward at `gi`, shared by the kinetic and the non-local rank-1 mesh path.
    let fwd0 = gi.map(|g| crate::complex_det::complex_gem_forward(gem, r_up, r_dn, Some(g)));
    let mut sigma = crate::timed!(
        "cs.kin",
        crate::complex_stress::complex_kinetic_stress_w(
            gem,
            jas,
            r_up,
            r_dn,
            cell,
            positions,
            home_up,
            kernel_up,
            home_dn,
            kernel_dn,
            j3c,
            j3m,
            j3_home,
            j3_kernel,
            None,
            gi,
            fwd0.as_ref(),
        )
    );
    // ---- ee strain: bare Ewald ee ----
    sigma += &crate::forces_ewald::grad_eee_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        cell,
        nmax,
    );
    sigma += &crate::forces_ewald::grad_eei_strain(
        r_up,
        r_dn,
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
        ew.n_max,
    );
    sigma += &crate::forces_ewald::grad_eii_strain(
        ew.eta,
        &ew.real_images,
        &ew.g_vectors,
        ew.g_factors,
        z_eff,
        positions,
        cell,
    );
    if !ecp {
        return sigma;
    }
    sigma += &crate::forces_ecp::grad_vlocal_strain(
        positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell, nmax,
    );
    sigma += &crate::timed!(
        "cs.ecp_nl",
        gfmc_ecp_nonlocal_strain_complex(
            gem,
            jas,
            j3,
            home_up,
            kernel_up,
            home_dn,
            kernel_dn,
            positions,
            r_up,
            r_dn,
            rt,
            cell,
            &inv,
            nmax,
            // VMC HF stress is the t-move (non-DLA) base -> `dltmove=false`.
            nuc_n,
            am_n,
            e_n,
            c_n,
            p_n,
            weights,
            grid_points,
            nn,
            false,
            gi,
            fast_mesh,
            batch_j3,
            fwd0.as_ref(),
            None,
            None,
        )
        .0
    );
    sigma
}

#[cfg(test)]
mod mesh_rank1_tests {
    use super::*;
    use numpy::ndarray::Array2;

    // Tiny deterministic LCG in (-1,1) so the parity test needs no rand dep.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn mat(&mut self, r: usize, c: usize) -> Array2<f64> {
            let mut m = Array2::<f64>::zeros((r, c));
            for i in 0..r {
                for j in 0..c {
                    m[[i, j]] = self.next();
                }
            }
            m
        }
    }

    fn inv(a: &Array2<f64>) -> Array2<f64> {
        crate::determinant::inverse(&a.view())
    }

    // Build a self-contained MeshBase from random fields (owns c_up/c_dn via the
    // caller's storage). Dims: M_up, N (=n_up), M_dn, n_dn, num_ao_up, num_ao_dn.
    #[allow(clippy::too_many_arguments)]
    fn make_base<'a>(
        rng: &mut Lcg,
        m_up: usize,
        n: usize,
        m_dn: usize,
        n_dn: usize,
        c_up: &'a Array2<f64>,
        c_dn: &'a Array2<f64>,
        nuc_up: &'a [usize],
        nuc_dn: &'a [usize],
        n_atom: usize,
    ) -> MeshBase<'a> {
        let ou = rng.mat(m_up, n);
        let od = rng.mat(m_dn, n_dn);
        let ga_up = [
            rng.mat(c_up.ncols(), n),
            rng.mat(c_up.ncols(), n),
            rng.mat(c_up.ncols(), n),
        ];
        let ga_dn = [
            rng.mat(c_dn.ncols(), n_dn),
            rng.mat(c_dn.ncols(), n_dn),
            rng.mat(c_dn.ncols(), n_dn),
        ];
        let lam_p = rng.mat(m_up, m_dn);
        // Physical constraints the rank-1 algebra relies on: q = ou^T lam_p (all rows),
        // and w[:, 0..n_dn] = lam_p od (paired columns; the unpaired tail is free).
        let q = ou.t().dot(&lam_p); // (n, m_dn)
        let lpod = lam_p.dot(&od); // (m_up, n_dn)
        let mut w = rng.mat(m_up, n);
        for a in 0..m_up {
            for l in 0..n_dn {
                w[[a, l]] = lpod[[a, l]];
            }
        }
        // diagonally dominant -> invertible ginv source.
        let mut g = rng.mat(n, n);
        for i in 0..n {
            g[[i, i]] += (n as f64) + 3.0;
        }
        let ginv = inv(&g);

        let wg_base = w.dot(&ginv);
        let gq_base = ginv.dot(&q);
        let mdn_base = gq_base
            .slice(numpy::ndarray::s![0..n_dn, ..])
            .t()
            .to_owned();
        let cm_base_up = pull_ao(Some(c_up), &wg_base);
        let cm_base_dn = pull_ao(Some(c_dn), &mdn_base);
        // MO gradient tables = MO projection of the AO gradient tables.
        let gou = [
            c_up.dot(&ga_up[0]),
            c_up.dot(&ga_up[1]),
            c_up.dot(&ga_up[2]),
        ];
        let godn = [
            c_dn.dot(&ga_dn[0]),
            c_dn.dot(&ga_dn[1]),
            c_dn.dot(&ga_dn[2]),
        ];
        let base_du = drift_up(&gou, &wg_base, n);
        let base_dd = drift_dn(&godn, &gq_base, n_dn);
        let (cmt_up, pself_up) = moved_tables(&cm_base_up, &ga_up, nuc_up, n_atom);
        let (cmt_dn, pself_dn) = moved_tables(&cm_base_dn, &ga_dn, nuc_dn, n_atom);

        MeshBase {
            c_up: Some(c_up),
            c_dn: Some(c_dn),
            nuc_up,
            nuc_dn,
            ou,
            od,
            ga_up,
            ga_dn,
            w,
            ginv,
            q,
            lam_p: Some(lam_p),
            n_up: n,
            n_dn,
            n_atom,
            wg_base,
            gq_base,
            cmt_up,
            cmt_dn,
            pself_up,
            pself_dn,
            base_du,
            base_dd,
        }
    }

    fn assert_close(tag: &str, a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert_eq!(a.dim(), b.dim(), "{tag} shape");
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            let s = 1.0_f64.max(x.abs()).max(y.abs());
            assert!(d <= tol * s, "{tag}: {x} vs {y} (abs {d})");
        }
    }

    /// The aggregated group sums must reproduce the per-candidate FULL-RECOMPUTE
    /// reference (fresh `G'^-1`, full W/q/drift/Pulay rebuild): per candidate the
    /// ratio and the moved electron's drift, and `Sum_v beta_v r_v (F_v - F_base)`
    /// for the Pulay and every other electron's drift. Generic random columns plus a
    /// group whose first candidate barely moves the column.
    #[test]
    fn mesh_group_aggregate_matches_reference() {
        let (m_up, n, m_dn, n_dn) = (6usize, 4usize, 5usize, 3usize);
        let (na_up, na_dn, n_atom) = (7usize, 8usize, 2usize);
        let mut rng = Lcg(0x0fed_cba9_8765_4321);
        let c_up = rng.mat(m_up, na_up);
        let c_dn = rng.mat(m_dn, na_dn);
        let nuc_up: Vec<usize> = (0..na_up).map(|i| i % n_atom).collect();
        let nuc_dn: Vec<usize> = (0..na_dn).map(|i| (i + 1) % n_atom).collect();
        let base = make_base(
            &mut rng, m_up, n, m_dn, n_dn, &c_up, &c_dn, &nuc_up, &nuc_dn, n_atom,
        );
        // base fields recomputed from scratch (independent of the cached anchors).
        let mut p_base = Array2::<f64>::zeros((n_atom, 3));
        pulay_side_c(
            Some(&c_up),
            &base.w.dot(&base.ginv),
            &base.ga_up,
            &nuc_up,
            &mut p_base,
        );
        let gq = base.ginv.dot(&base.q);
        let mdn = gq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned();
        pulay_side_c(Some(&c_dn), &mdn, &base.ga_dn, &nuc_dn, &mut p_base);
        let gou_b = [
            c_up.dot(&base.ga_up[0]),
            c_up.dot(&base.ga_up[1]),
            c_up.dot(&base.ga_up[2]),
        ];
        let godn_b = [
            c_dn.dot(&base.ga_dn[0]),
            c_dn.dot(&base.ga_dn[1]),
            c_dn.dot(&base.ga_dn[2]),
        ];
        let u_base = drift_up(&gou_b, &base.w.dot(&base.ginv), n);
        let d_base = drift_dn(&godn_b, &gq, n_dn);

        let nv = 5usize;
        let tol = crate::setting_consts::RTOL_RS_UNIT_RANK1_FORCE;
        // Every group also goes into ONE accumulator (both spins, two groups per
        // moved electron), which must reproduce the summed references.
        let mut all_acc = base.new_acc();
        let mut all_ref = (
            Array2::<f64>::zeros((n_atom, 3)),
            Array2::<f64>::zeros((n, 3)),
            Array2::<f64>::zeros((n_dn, 3)),
        );
        for near in [false, true] {
            for (spin_up, m, na, nloc) in [(true, m_up, na_up, n), (false, m_dn, na_dn, n_dn)] {
                for li in 0..nloc {
                    let old = base.moved_old(spin_up, li);
                    let mut mo_vals = rng.mat(m, nv);
                    if near {
                        for a in 0..m {
                            mo_vals[[a, 0]] = old[a] + 1e-9 * rng.next();
                        }
                    }
                    let ga1s = [rng.mat(na, nv), rng.mat(na, nv), rng.mat(na, nv)];
                    // MO gradients of the candidates = MO projection of their AO gradients.
                    let cm = if spin_up { &c_up } else { &c_dn };
                    let mo_gs = [cm.dot(&ga1s[0]), cm.dot(&ga1s[1]), cm.dot(&ga1s[2])];
                    let beta: Vec<f64> = (0..nv).map(|_| rng.next()).collect();

                    // reference: per-candidate full recompute, then the weighted sum.
                    let mut e_p = Array2::<f64>::zeros((n_atom, 3));
                    let mut e_u = Array2::<f64>::zeros((n, 3));
                    let mut e_d = Array2::<f64>::zeros((n_dn, 3));
                    let mut mv_ref = Array2::<f64>::zeros((nv, 3));
                    let mut mv_fast = Array2::<f64>::zeros((nv, 3));
                    let u = base.moved_coef(spin_up, li);
                    let u_old: f64 = u.iter().zip(&old).map(|(a, b)| a * b).sum();
                    for v in 0..nv {
                        let col = |mat: &Array2<f64>| -> Array2<f64> {
                            mat.slice(numpy::ndarray::s![.., v..v + 1]).to_owned()
                        };
                        let mg1 = [col(&mo_gs[0]), col(&mo_gs[1]), col(&mo_gs[2])];
                        let ga1 = [col(&ga1s[0]), col(&ga1s[1]), col(&ga1s[2])];
                        let (r0, p0, u0, d0) =
                            base.candidate_cols_ref(spin_up, li, &col(&mo_vals), &mg1, &ga1);
                        // fast per-candidate fields
                        let s: f64 = (0..m).map(|a| u[a] * mo_vals[[a, v]]).sum();
                        let r1 = 1.0 + (s - u_old);
                        assert!(
                            (r0 - r1).abs()
                                <= crate::setting_consts::ATOL_PYTHON_FP64_LOOSE
                                    * r0.abs().max(1.0),
                            "ratio spin_up={spin_up} li={li} v={v}: {r0} vs {r1}"
                        );
                        let moved_ref = if spin_up {
                            u0.row(li).to_owned()
                        } else {
                            d0.row(li).to_owned()
                        };
                        for d in 0..3 {
                            let gd: f64 = (0..m).map(|a| u[a] * mo_gs[d][[a, v]]).sum();
                            mv_ref[[v, d]] = moved_ref[d];
                            mv_fast[[v, d]] = gd / r1;
                        }
                        let w = beta[v] * r0;
                        e_p.scaled_add(w, &(&p0 - &p_base));
                        e_u.scaled_add(w, &(&u0 - &u_base));
                        e_d.scaled_add(w, &(&d0 - &d_base));
                    }
                    assert_close("moved drift", &mv_ref, &mv_fast, tol);
                    let own = if spin_up { &mut e_u } else { &mut e_d };
                    own.row_mut(li).fill(0.0);

                    let bv = ArrayView1::from(&beta[..]);
                    let mo_bar = mo_vals.dot(&bv);
                    let ga1bar = [
                        ga1s[0].dot(&bv).to_vec(),
                        ga1s[1].dot(&bv).to_vec(),
                        ga1s[2].dot(&bv).to_vec(),
                    ];
                    let mut acc = base.new_acc();
                    base.accumulate(
                        spin_up,
                        li,
                        mo_bar.as_slice().unwrap(),
                        beta.iter().sum(),
                        &ga1bar,
                        &mut acc,
                    );
                    let (p1, u1, d1) = base.finish(acc);
                    assert_close("agg pulay", &e_p, &p1, tol);
                    assert_close("agg drift_up", &e_u, &u1, tol);
                    assert_close("agg drift_dn", &e_d, &d1, tol);
                    base.accumulate(
                        spin_up,
                        li,
                        mo_bar.as_slice().unwrap(),
                        beta.iter().sum(),
                        &ga1bar,
                        &mut all_acc,
                    );
                    all_ref.0 += &e_p;
                    all_ref.1 += &e_u;
                    all_ref.2 += &e_d;
                }
            }
        }
        let (p1, u1, d1) = base.finish(all_acc);
        assert_close("all pulay", &all_ref.0, &p1, tol);
        assert_close("all drift_up", &all_ref.1, &u1, tol);
        assert_close("all drift_dn", &all_ref.2, &d1, tol);
    }
}
