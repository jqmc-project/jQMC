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

//! Analytical strain derivatives (stress), molecular affine part. Strain eps
//! deforms positions affinely (r'=(I+eps)r), so the stress is the virial
//! contraction `dQ/deps_cd = Sum_e (dQ/dr_e)_c*r_e,d + Sum_I (dQ/dR_I)_c*R_I,d`.
//! Cartesian molecular scope. Autograd-free; reuses the nuclear-force machinery.

use numpy::ndarray::{Array2, ArrayView2, Axis};

use crate::ao_cart_deriv::{ao_deriv_bundle, ao_grad, ao_gradlap, AoDerivBundle};
use crate::atomic_orbital::compute_aos;
use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{GeminalData, OrbData};
use crate::forces::{
    grad_j1_wrt_r, grad_j3_wrt_r, grad_lndet_wrt_r, grad_t_wrt_r, grad_v_wrt_r, pull_to_ao,
    KinShared, KinSharedJ3,
};
use crate::param_grad_el::{det_grad_field, geminal_reverse, jastrow_grads};

/// `dlnPsi/deps` (Pulay stress, 3x3).
#[allow(clippy::too_many_arguments)]
pub fn grad_lnpsi_wrt_strain(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = positions.nrows();

    // electron drift dlnPsi/dr = gD + gJ
    let (gd_up, gd_dn) = det_grad_field(gem, r_up, r_dn);
    let (gj_up, gj_dn) = jastrow_grads(jastrow, j3_orb, j3_jm, r_up, r_dn);
    let drift = |e: usize, c: usize| -> f64 {
        if e < n_up {
            gd_up[[e, c]] + gj_up[[e, c]]
        } else {
            let l = e - n_up;
            gd_dn[[l, c]] + gj_dn[[l, c]]
        }
    };
    let rcoord = |e: usize, d: usize| -> f64 {
        if e < n_up {
            r_up[[e, d]]
        } else {
            r_dn[[e - n_up, d]]
        }
    };

    // dlnPsi/dR: determinant plus J1 and J3.
    let mut d_r = grad_lndet_wrt_r(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom);
    if let Some(j1) = j1 {
        d_r += &grad_j1_wrt_r(j1, r_up, r_dn);
    }
    if let (Some(orb), Some(jm), Some(nuc)) = (j3_orb, j3_jm, nuc_j3) {
        d_r += &grad_j3_wrt_r(orb, jm, r_up, r_dn, nuc, n_atom);
    }

    // stress[c,d] = Sum_e drift[e,c]*r[e,d] + Sum_I dR[I,c]*R[I,d]
    let mut stress = Array2::<f64>::zeros((3, 3));
    for c in 0..3 {
        for d in 0..3 {
            let mut acc = 0.0;
            for e in 0..n_e {
                acc += drift(e, c) * rcoord(e, d);
            }
            for ia in 0..n_atom {
                acc += d_r[[ia, c]] * positions[[ia, d]];
            }
            stress[[c, d]] = acc;
        }
    }
    stress
}

// ============================================================================
// de_L/deps (HF stress), molecular affine, all-electron, without Jastrow.
// ============================================================================

/// `dT_det/dr_e` per electron `(N,3)` per spin: same trterm cotangents as the
/// kinetic force, contracted per-electron with orbital r-derivatives, `+` sign.
pub(crate) fn tdet_grad_wrt_r(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let zu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let zd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    let b = geminal_reverse(gem, r_up, r_dn, &zu, &zd); // gJ=0 -> jnone cotangents
    let side = |orb: &OrbData,
                r: &ArrayView2<f64>,
                bar_orb: &Array2<f64>,
                bar_lap: &Array2<f64>|
     -> Array2<f64> {
        let cm_orb = pull_to_ao(orb, bar_orb);
        let cm_lap = pull_to_ao(orb, bar_lap);
        let ga = ao_grad(&orb.aos, r);
        let gl = ao_gradlap(&orb.aos, r);
        let num_ao = cm_orb.nrows();
        let n = cm_orb.ncols();
        let mut out = Array2::<f64>::zeros((n, 3));
        for k in 0..n {
            for d in 0..3 {
                let mut acc = 0.0;
                for c in 0..num_ao {
                    acc += cm_orb[[c, k]] * ga[d][[c, k]] + cm_lap[[c, k]] * gl[d][[c, k]];
                }
                out[[k, d]] = acc;
            }
        }
        out
    };
    (
        side(&gem.orb_up, r_up, &b.bar_orb_up, &b.bar_lap_up),
        side(&gem.orb_dn, r_dn, &b.bar_orb_dn, &b.bar_lap_dn),
    )
}

/// `dV_bare/dr_e` per electron `(N,3)` per spin: `Sum_{e'!=e}(r_e-r_e')/|*|^3 -
/// Sum_I Z_I(r_e-R_I)/|*|^3`.
pub(crate) fn vbare_grad_wrt_r(
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
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
    let mut g = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        let re = elec(e);
        // e-e repulsion
        for ep in 0..n_e {
            if ep == e {
                continue;
            }
            let r2 = elec(ep);
            let dx = re[0] - r2[0];
            let dy = re[1] - r2[1];
            let dz = re[2] - r2[2];
            let inv3 = (dx * dx + dy * dy + dz * dz).powf(-1.5);
            // d(1/r_ee')/dr_e = -(r_e-r_e')/|*|^3
            g[[e, 0]] -= dx * inv3;
            g[[e, 1]] -= dy * inv3;
            g[[e, 2]] -= dz * inv3;
        }
        // e-N: d(-Z/r_eI)/dr_e = +Z (r_e-R_I)/|*|^3
        for ia in 0..n_atom {
            let dx = re[0] - positions[[ia, 0]];
            let dy = re[1] - positions[[ia, 1]];
            let dz = re[2] - positions[[ia, 2]];
            let inv3 = (dx * dx + dy * dy + dz * dz).powf(-1.5);
            g[[e, 0]] += z_eff[ia] * dx * inv3;
            g[[e, 1]] += z_eff[ia] * dy * inv3;
            g[[e, 2]] += z_eff[ia] * dz * inv3;
        }
    }
    let gu = g.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let gd = g.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    (gu, gd)
}

/// `de_L/deps` (HF stress, 3x3), all-electron molecular without Jastrow.
#[allow(clippy::too_many_arguments)]
pub fn grad_el_wrt_strain_jnone(
    gem: &GeminalData,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let n_atom = positions.nrows();

    // de_L/dr_e = dT_det/dr + dV_bare/dr
    let (tu, td) = tdet_grad_wrt_r(gem, r_up, r_dn);
    let (vu, vd) = vbare_grad_wrt_r(positions, z_eff, r_up, r_dn);
    let der_r = |e: usize, c: usize| -> f64 {
        if e < n_up {
            tu[[e, c]] + vu[[e, c]]
        } else {
            let l = e - n_up;
            td[[l, c]] + vd[[l, c]]
        }
    };
    let rcoord = |e: usize, d: usize| -> f64 {
        if e < n_up {
            r_up[[e, d]]
        } else {
            r_dn[[e - n_up, d]]
        }
    };
    // de_L/dR_I = dT_det/dR + dV/dR
    let d_r = {
        let t = grad_t_wrt_r(
            gem, None, None, None, None, r_up, r_dn, nuc_up, nuc_dn, None, n_atom,
        );
        let v = grad_v_wrt_r(positions, z_eff, r_up, r_dn);
        t + &v
    };

    let mut stress = Array2::<f64>::zeros((3, 3));
    for c in 0..3 {
        for d in 0..3 {
            let mut acc = 0.0;
            for e in 0..n_e {
                acc += der_r(e, c) * rcoord(e, d);
            }
            for ia in 0..n_atom {
                acc += d_r[[ia, c]] * positions[[ia, d]];
            }
            stress[[c, d]] = acc;
        }
    }
    stress
}

/// `de_L/deps` (HF stress, 3x3) for an ECP system (jnone). Adds the ECP local +
/// non-local contributions to the bare-Coulomb virial. `der_r_ecp` = `dV_ecp/dr`
/// (local + non-local) per electron `(Ne,3)`; `d_r_ecp` = `dV_ecp/dR` (local +
/// non-local) per atom `(n_atom,3)`.
#[allow(clippy::too_many_arguments)]
pub fn grad_el_wrt_strain_ecp(
    gem: &GeminalData,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    der_r_ecp: &Array2<f64>,
    d_r_ecp: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let (tu, td) = tdet_grad_wrt_r(gem, r_up, r_dn);
    let (vu, vd) = vbare_grad_wrt_r(positions, z_eff, r_up, r_dn);
    let der_r = |e: usize, c: usize| -> f64 {
        let base = if e < n_up {
            tu[[e, c]] + vu[[e, c]]
        } else {
            let l = e - n_up;
            td[[l, c]] + vd[[l, c]]
        };
        base + der_r_ecp[[e, c]]
    };
    let rcoord = |e: usize, d: usize| -> f64 {
        if e < n_up {
            r_up[[e, d]]
        } else {
            r_dn[[e - n_up, d]]
        }
    };
    let d_r = {
        let t = grad_t_wrt_r(
            gem, None, None, None, None, r_up, r_dn, nuc_up, nuc_dn, None, n_atom,
        );
        let v = grad_v_wrt_r(positions, z_eff, r_up, r_dn);
        t + &v + d_r_ecp
    };
    let mut stress = Array2::<f64>::zeros((3, 3));
    for c in 0..3 {
        for d in 0..3 {
            let mut acc = 0.0;
            for e in 0..n_e {
                acc += der_r(e, c) * rcoord(e, d);
            }
            for ia in 0..n_atom {
                acc += d_r[[ia, c]] * positions[[ia, d]];
            }
            stress[[c, d]] = acc;
        }
    }
    stress
}

// ============================================================================
// Molecular de_L/deps with J1/J2/J3 Jastrow terms.
// Needs dT/dr WITH Jastrow (kinetic_egrad.grad_T_wrt_r): orbital gou + J1 + J2 (e-e,
// R-independent so absent from the force) + J3, all the per-electron `+grad_r` direction.
// ============================================================================

fn k_strict_lower(n: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..i {
            m[[i, j]] = 1.0;
        }
    }
    m
}

/// Strain virial `Sum_e der_r[e,c]*r[e,d] + Sum_I d_r[I,c]*R[I,d]` (3x3). The stress
/// is linear in the per-electron/per-atom energy gradients, so extra contributions
/// (e.g. DTN) compose additively onto a base stress.
pub fn virial(
    der_r: &Array2<f64>,
    d_r: &Array2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    let rcoord = |e: usize, d: usize| -> f64 {
        if e < n_up {
            r_up[[e, d]]
        } else {
            r_dn[[e - n_up, d]]
        }
    };
    let mut s = Array2::<f64>::zeros((3, 3));
    for c in 0..3 {
        for d in 0..3 {
            let mut acc = 0.0;
            for e in 0..n_e {
                acc += der_r[[e, c]] * rcoord(e, d);
            }
            for ia in 0..n_atom {
                acc += d_r[[ia, c]] * positions[[ia, d]];
            }
            s[[c, d]] = acc;
        }
    }
    s
}

/// Orbital-side `dT/dr_e` per electron `(N,3)`: value+lap+gou cotangents contracted
/// per-electron with `+grad_r AO` (no scatter). gou (Hessian) is the with-Jastrow term.
pub(crate) fn orb_grad_wrt_r(
    orb: &OrbData,
    r: &ArrayView2<f64>,
    bar_orb: &Array2<f64>,
    bar_lap: &Array2<f64>,
    bar_gou: &[Array2<f64>; 3],
    precomp: Option<&AoDerivBundle>,
) -> Array2<f64> {
    let cm_orb = pull_to_ao(orb, bar_orb);
    let cm_lap = pull_to_ao(orb, bar_lap);
    let cm_gou = [
        pull_to_ao(orb, &bar_gou[0]),
        pull_to_ao(orb, &bar_gou[1]),
        pull_to_ao(orb, &bar_gou[2]),
    ];
    let owned = if precomp.is_none() {
        Some(ao_deriv_bundle(&orb.aos, r))
    } else {
        None
    };
    let bundle = precomp.unwrap_or_else(|| owned.as_ref().unwrap());
    let ga = &bundle.grad;
    let gl = &bundle.gradlap;
    let h = &bundle.hessian;
    let num_ao = cm_orb.nrows();
    let n = cm_orb.ncols();
    let mut out = Array2::<f64>::zeros((n, 3));
    for k in 0..n {
        for d in 0..3 {
            let mut acc = 0.0;
            for c in 0..num_ao {
                acc += cm_orb[[c, k]] * ga[d][[c, k]] + cm_lap[[c, k]] * gl[d][[c, k]];
                for e in 0..3 {
                    acc += cm_gou[e][[c, k]] * h[d][e][[c, k]];
                }
            }
            out[[k, d]] = acc;
        }
    }
    out
}

/// `dT/dr` per electron WITH Jastrow (J1/J2/J3), `(N_up,3),(N_dn,3)`. 1:1
/// kinetic_egrad.grad_T_wrt_r. `cot = gJ + gD` is the kinetic gradient cotangent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tfull_grad_wrt_r(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    tfull_grad_wrt_r_w(gem, jastrow, j1, j2, j3_orb, j3_jm, r_up, r_dn, None)
}

/// `tfull_grad_wrt_r` with per-electron kinetic-element weights `w` (global index):
/// `d(Sum_k w_k*kc_k)/dr_e` for all e -- the masked electron-axis kinetic derivative
/// (GFMC ZV branch corrections). `None` = all-1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tfull_grad_wrt_r_w(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
) -> (Array2<f64>, Array2<f64>) {
    tfull_grad_wrt_r_w_gi(gem, jastrow, j1, j2, j3_orb, j3_jm, r_up, r_dn, w, None)
}

/// `tfull_grad_wrt_r_w` reusing a caller-supplied geminal inverse (`gem_forward_gi`):
/// the MCMC observable VMC drift dT/dr reuses the sweep SM inverse. `None` == fresh.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tfull_grad_wrt_r_w_gi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    tfull_grad_wrt_r_w_gi_shared(
        gem, jastrow, j1, j2, j3_orb, j3_jm, r_up, r_dn, w, g_inv, None,
    )
}

/// `tfull_grad_wrt_r_w_gi` reusing the shared per-step kinetic fields (`KinShared`),
/// so `jastrow_grads` / `geminal_reverse` / det-grad-field / the det & J3 AO bundles
/// are computed once and shared with the dR (`grad_t`) kernel. `shared = None`
/// builds locally (still fused). A `shared` build is valid only for `w = None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tfull_grad_wrt_r_w_gi_shared(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    w: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
    shared: Option<&KinShared>,
) -> (Array2<f64>, Array2<f64>) {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let wt = |e: usize| -> f64 { w.map_or(1.0, |w| w[e]) };
    let local;
    let sh: &KinShared = match shared {
        Some(s) => s,
        None => {
            local = KinShared::build(gem, jastrow, j3_orb, j3_jm, r_up, r_dn, w, g_inv);
            &local
        }
    };
    let (gj_up, gj_dn) = (&sh.gj_up, &sh.gj_dn);
    let b = &sh.b;
    // orbital side (det, + J3 when `fz.j3`), contracted in the KinShared build if fused
    let (ou, od) = match &sh.fused {
        Some(fz) => (fz.d_re_up.clone(), fz.d_re_dn.clone()),
        None => (
            orb_grad_wrt_r(
                &gem.orb_up,
                r_up,
                &b.bar_orb_up,
                &b.bar_lap_up,
                &b.bar_gou,
                sh.bundle_up.as_ref(),
            ),
            orb_grad_wrt_r(
                &gem.orb_dn,
                r_dn,
                &b.bar_orb_dn,
                &b.bar_lap_dn,
                &b.bar_godn,
                sh.bundle_dn.as_ref(),
            ),
        ),
    };
    // dT accumulated over all electrons (N_e, 3)
    let mut dt = Array2::<f64>::zeros((n_e, 3));
    for k in 0..n_up {
        for d in 0..3 {
            dt[[k, d]] = ou[[k, d]];
        }
    }
    for k in 0..(n_e - n_up) {
        for d in 0..3 {
            dt[[n_up + k, d]] = od[[k, d]];
        }
    }
    // cot = gJ + gD (Ne,3)
    let (gd_up, gd_dn) = (&sh.gd_up, &sh.gd_dn);
    let mut cot = Array2::<f64>::zeros((n_e, 3));
    for e in 0..n_e {
        for d in 0..3 {
            cot[[e, d]] = if e < n_up {
                gj_up[[e, d]] + gd_up[[e, d]]
            } else {
                let l = e - n_up;
                gj_dn[[l, d]] + gd_dn[[l, d]]
            };
        }
    }
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    // ---- J2 (e-e): pair (k,e); dgJ[k]/dr_e and dgJ[e]/dr_e (self) ----
    if let Some(j2) = j2 {
        let p = j2.jastrow_2b_param;
        let is_exp = j2.jastrow_2b_type == "exp";
        for e in 0..n_e {
            for k in 0..n_e {
                if k == e {
                    continue;
                }
                let rk = elec(k);
                let re = elec(e);
                let dvec = [rk[0] - re[0], rk[1] - re[1], rk[2] - re[2]];
                let rr = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                let u = [dvec[0] / rr, dvec[1] / rr, dvec[2] / rr];
                let (h1, h2, h3) = if is_exp {
                    let ee = (-p * rr).exp();
                    (0.5 * ee, -0.5 * p * ee, 0.5 * p * p * ee)
                } else {
                    let uu = 1.0 + p * rr;
                    (
                        1.0 / (2.0 * uu * uu),
                        -p / uu.powi(3),
                        3.0 * p * p / uu.powi(4),
                    )
                };
                // Hess = h2 u_hat(x)u_hat + (h1/r)(I - u_hat(x)u_hat)
                let mut hess = [[0.0; 3]; 3];
                for a in 0..3 {
                    for bb in 0..3 {
                        let id = if a == bb { 1.0 } else { 0.0 };
                        hess[a][bb] = h2 * u[a] * u[bb] + (h1 / rr) * (id - u[a] * u[bb]);
                    }
                }
                let lp = h3 + 2.0 * h2 / rr - 2.0 * h1 / (rr * rr);
                // (k!=e): dgJ[k]=-Hess, dlap[k]=-Lp u_hat  ->  -0.5(-Lp u_hat) - cot[k]*(-Hess)
                // (k==e self): dgJ[e]=+Hess, dlap[e]=-Lp u_hat  ->  -0.5(-Lp u_hat) - cot[e]*Hess
                // element k's kc weighted wt(k); element e's own (self) weighted wt(e)
                for d in 0..3 {
                    let hk: f64 = (0..3).map(|a| cot[[k, a]] * hess[a][d]).sum();
                    let he: f64 = (0..3).map(|a| cot[[e, a]] * hess[a][d]).sum();
                    dt[[e, d]] += wt(k) * (-0.5 * (-lp * u[d]) - (-hk)); // k!=e
                    dt[[e, d]] += wt(e) * (-0.5 * (-lp * u[d]) - he); // self
                }
            }
        }
    }
    // ---- J1 (e-N, diagonal in e) ----
    if let Some(j1) = j1 {
        let r_atoms = j1.structure_data.positions.view();
        let n_atom = j1.structure_data.n_atoms();
        let p = j1.jastrow_1b_param;
        let is_exp = j1.jastrow_1b_type == "exp";
        for e in 0..n_e {
            let re = elec(e);
            for ia in 0..n_atom {
                let z = j1.structure_data.atomic_numbers[ia] - j1.core_electrons[ia];
                let c = (2.0 * z).powf(0.25);
                let pref = if is_exp {
                    -(2.0 * z).powf(0.75)
                } else {
                    -2.0 * z
                };
                let dvec = [
                    re[0] - r_atoms[[ia, 0]],
                    re[1] - r_atoms[[ia, 1]],
                    re[2] - r_atoms[[ia, 2]],
                ];
                let dist = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                let u = [dvec[0] / dist, dvec[1] / dist, dvec[2] / dist];
                let (f1, f2, f3) = if is_exp {
                    let a = p * c;
                    let ee = (-a * dist).exp();
                    (
                        pref * c / 2.0 * ee,
                        -pref * (p * c * c) / 2.0 * ee,
                        pref * (p * p * c * c * c) / 2.0 * ee,
                    )
                } else {
                    let bb = p * c;
                    let uu = 1.0 + bb * dist;
                    (
                        pref / (2.0 * uu * uu),
                        -pref * bb / uu.powi(3),
                        3.0 * pref * bb * bb / uu.powi(4),
                    )
                };
                let lp = f3 + 2.0 * f2 / dist - 2.0 * f1 / (dist * dist);
                // Hess_I = f2 u_hat(x)u_hat + (f1/dist)(I - u_hat(x)u_hat); dlap += Lp u_hat (element e -> wt(e))
                for d in 0..3 {
                    let mut hcd = 0.0;
                    for a in 0..3 {
                        let id = if a == d { 1.0 } else { 0.0 };
                        let hess_ad = f2 * u[a] * u[d] + (f1 / dist) * (id - u[a] * u[d]);
                        hcd += cot[[e, a]] * hess_ad;
                    }
                    dt[[e, d]] += wt(e) * (-0.5 * (lp * u[d]) - hcd);
                }
            }
        }
    }
    // ---- J3 (e-e-n, orbital basis) ----
    let j3_fused = sh.fused.as_ref().is_some_and(|fz| fz.j3);
    if let (Some(orb), Some(jm), false) = (j3_orb, j3_jm, j3_fused) {
        crate::timed!(
            "f.dr_j3side",
            j3_grad_wrt_r(orb, jm, r_up, r_dn, &cot, w, &mut dt, sh.j3.as_ref())
        );
    }
    let dtu = dt.slice(numpy::ndarray::s![0..n_up, ..]).to_owned();
    let dtd = dt.slice(numpy::ndarray::s![n_up.., ..]).to_owned();
    (dtu, dtd)
}

/// J3 `dT_J3/dr` per electron: VJP of (gJ3,lapJ3) forward contracted per-electron,
/// plus the cross-spin `s=SumA` terms. 1:1 kinetic_egrad._j3_side_r. `cot=gJ+gD`.
#[allow(clippy::too_many_arguments)]
fn j3_grad_wrt_r(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cot: &Array2<f64>,
    w: Option<&[f64]>,
    dt: &mut Array2<f64>,
    precomp: Option<&KinSharedJ3>,
) {
    let aos = &j3_orb.aos;
    let owned: Option<(Array2<f64>, Array2<f64>, AoDerivBundle, AoDerivBundle)> =
        if precomp.is_none() {
            Some((
                compute_aos(aos, r_up),
                compute_aos(aos, r_dn),
                ao_deriv_bundle(aos, r_up),
                ao_deriv_bundle(aos, r_dn),
            ))
        } else {
            None
        };
    let (a_up, a_dn, bu, bd) = match precomp {
        Some(j3) => (&j3.a_up, &j3.a_dn, &j3.bundle_up, &j3.bundle_dn),
        None => {
            let o = owned.as_ref().unwrap();
            (&o.0, &o.1, &o.2, &o.3)
        }
    };
    let go_u = &bu.grad;
    let go_d = &bd.grad;
    let lo_u = &bu.laplacian;
    let lo_d = &bd.laplacian;
    let hu = &bu.hessian;
    let hd = &bd.hessian;
    let glu = &bu.gradlap;
    let gld = &bd.gradlap;
    let orb_n = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();
    let ku = k_strict_lower(n_up);
    let kd = k_strict_lower(n_dn);
    let m = j_matrix.slice(numpy::ndarray::s![.., 0..orb_n]).to_owned();
    let j1col = j_matrix.column(orb_n).to_owned();
    let s_up = a_up.sum_axis(Axis(1));
    let s_dn = a_dn.sum_axis(Axis(1));
    let m_s_dn = m.dot(&s_dn);
    let mt_s_up = m.t().dot(&s_up);
    let mk =
        |a: &Array2<f64>, k: &Array2<f64>, cross: &numpy::ndarray::Array1<f64>| -> Array2<f64> {
            let mut g = m.dot(a).dot(k) + m.t().dot(a).dot(&k.t());
            let nc = a.ncols();
            for p in 0..orb_n {
                for c in 0..nc {
                    g[[p, c]] += j1col[p] + cross[p];
                }
            }
            g
        };
    let coef_up = mk(a_up, &ku, &m_s_dn);
    let coef_dn = mk(a_dn, &kd, &mt_s_up);
    // bar_coefA[a,k] = -0.5 lO - Sum_d cot[k,d] gO[d,a,k]; bgo[d][a,k]=-cot[k,d] coef; blo=-0.5 coef
    // (all per-column k = per-electron kc_k seeds -> weighted by w[off+k])
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
    let (bcoef_u, bgo_u, blo_u) = side(&coef_up, go_u, lo_u, 0);
    let (bcoef_d, bgo_d, blo_d) = side(&coef_dn, go_d, lo_d, n_up);
    // per-electron de = bar_A_self*gO + bar_gO*H + bar_lO*GL  (bar_A_self = self part only)
    let de_side = |bcoef: &Array2<f64>,
                   bgo: &[Array2<f64>; 3],
                   blo: &Array2<f64>,
                   go: &[Array2<f64>; 3],
                   h: &[[Array2<f64>; 3]; 3],
                   gl: &[Array2<f64>; 3],
                   k: &Array2<f64>,
                   off: usize,
                   dt: &mut Array2<f64>| {
        let nc = bcoef.ncols();
        let bar_a = m.t().dot(bcoef).dot(&k.t()) + m.dot(bcoef).dot(k); // self only
        for e in 0..nc {
            for c in 0..3 {
                let mut acc = 0.0;
                for a in 0..orb_n {
                    acc += bar_a[[a, e]] * go[c][[a, e]];
                    for d in 0..3 {
                        acc += bgo[d][[a, e]] * h[c][d][[a, e]];
                    }
                    acc += blo[[a, e]] * gl[c][[a, e]];
                }
                dt[[off + e, c]] += acc;
            }
        }
    };
    de_side(&bcoef_u, &bgo_u, &blo_u, go_u, hu, glu, &ku, 0, dt);
    de_side(&bcoef_d, &bgo_d, &blo_d, go_d, hd, gld, &kd, n_up, dt);
    // cross-spin s terms: s_dn enters coef_up via M, s_up enters coef_dn via MT
    let bar_s_dn = m.t().dot(&bcoef_u.sum_axis(Axis(1))); // (orb,)
    let bar_s_up = m.dot(&bcoef_d.sum_axis(Axis(1)));
    for e in 0..n_dn {
        for c in 0..3 {
            let mut acc = 0.0;
            for a in 0..orb_n {
                acc += bar_s_dn[a] * go_d[c][[a, e]];
            }
            dt[[n_up + e, c]] += acc;
        }
    }
    for e in 0..n_up {
        for c in 0..3 {
            let mut acc = 0.0;
            for a in 0..orb_n {
                acc += bar_s_up[a] * go_u[c][[a, e]];
            }
            dt[[e, c]] += acc;
        }
    }
}

/// `de_L/deps` (HF stress, 3x3) with J1/J2/J3 and bare Coulomb. Virial of
/// (dT/dr with Jastrow + dV_bare/dr) and
/// (dT/dR withJ + dV/dR).
#[allow(clippy::too_many_arguments)]
pub fn grad_el_wrt_strain_withj(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j1: Option<&JastrowOneBodyData>,
    j2: Option<&JastrowTwoBodyData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    z_eff: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    nuc_j3: Option<&[usize]>,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let n_atom = positions.nrows();
    // de_L/dr = dT/dr(withJ) + dV_bare/dr
    let (tu, td) = tfull_grad_wrt_r(gem, jastrow, j1, j2, j3_orb, j3_jm, r_up, r_dn);
    let (vu, vd) = vbare_grad_wrt_r(positions, z_eff, r_up, r_dn);
    let der_r = |e: usize, c: usize| -> f64 {
        if e < n_up {
            tu[[e, c]] + vu[[e, c]]
        } else {
            let l = e - n_up;
            td[[l, c]] + vd[[l, c]]
        }
    };
    let rcoord = |e: usize, d: usize| -> f64 {
        if e < n_up {
            r_up[[e, d]]
        } else {
            r_dn[[e - n_up, d]]
        }
    };
    // de_L/dR = dT/dR(withJ) + dV/dR
    let d_r = {
        let t = grad_t_wrt_r(
            gem, jastrow, j1, j3_orb, j3_jm, r_up, r_dn, nuc_up, nuc_dn, nuc_j3, n_atom,
        );
        let v = grad_v_wrt_r(positions, z_eff, r_up, r_dn);
        t + &v
    };
    let mut stress = Array2::<f64>::zeros((3, 3));
    for c in 0..3 {
        for d in 0..3 {
            let mut acc = 0.0;
            for e in 0..n_e {
                acc += der_r(e, c) * rcoord(e, d);
            }
            for ia in 0..n_atom {
                acc += d_r[[ia, c]] * positions[[ia, d]];
            }
            stress[[c, d]] = acc;
        }
    }
    stress
}
