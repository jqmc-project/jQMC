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

//! `de_L/dc = dT/dc` because `dV/dc = 0` exactly. This is a hand-written VJP of
//! the analytical kinetic forward; it is autograd-free and supports molecular
//! (Euclidean), real-Gamma systems.
//!
//! Reduction for Jastrow params: `dT/dc = -1/2Sumgrad^2f_c - Sum(gJ+gD)*gradf_c`,
//! `f_c=dJ/dc`, `gJ=gradJ`, `gD=gradln|detG|`. Geminal params: reverse pass through
//! the geminal matrix algebra (`_geminal_reverse`).

use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{
    inverse, orb_value_grad_lap, orb_value_grad_lap_cut, GeminalData, OrbData,
};
use crate::jastrow_factor::{
    compute_grads_and_laplacian_jastrow_part, compute_grads_and_laplacian_jastrow_three_body,
    smooth_dist, CellInfo,
};

/// Periodic-aware distance field for the Jastrow kinetic dT/dc reduction: returns
/// `(d, grad_r d [3], |grad_r d|^2, grad^2_r d)` for a displacement `disp = r_a - r_b`.
/// `None` cell => Euclidean (`grad = u_hat`, `|grad|^2 = 1`, `grad^2 = 2/d`); `Some` =>
/// Williamson smooth periodic distance (matches the J1/J2 value + periodic-force paths).
#[inline]
fn dist_field(disp: [f64; 3], ci: Option<&CellInfo>) -> (f64, [f64; 3], f64, f64) {
    match ci {
        Some(ci) => smooth_dist(disp, ci),
        None => {
            let d = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
            let inv = 1.0 / d;
            (
                d,
                [disp[0] * inv, disp[1] * inv, disp[2] * inv],
                1.0,
                2.0 * inv,
            )
        }
    }
}

/// Shared geminal forward pieces from orbital value/grad/lap.
#[derive(Clone)]
pub(crate) struct GemFwd {
    pub ou: Array2<f64>,        // (M_up, N_up) value table owning G / running G^-1
    pub od: Array2<f64>,        // (M_dn, N_dn) value table owning G / running G^-1
    pub ou_fused: Array2<f64>,  // (M_up, N_up) kinetic value/grad/lap table
    pub od_fused: Array2<f64>,  // (M_dn, N_dn) kinetic value/grad/lap table
    pub lap_up: Array2<f64>,    // (M_up, N_up)
    pub lap_dn: Array2<f64>,    // (M_dn, N_dn)
    pub gou: [Array2<f64>; 3],  // (M_up, N_up) per axis
    pub godn: [Array2<f64>; 3], // (M_dn, N_dn)
    pub w: Array2<f64>,         // inverse-route W (M_up, N_up)
    pub w_fused: Array2<f64>,   // kinetic direct-route W (M_up, N_up)
    pub ginv: Array2<f64>,      // (N_up, N_up)
    pub q: Array2<f64>,         // inverse-route Q (N_up, M_dn)
    pub q_fused: Array2<f64>,   // kinetic direct-route Q (N_up, M_dn)
    pub wg: Array2<f64>,        // (M_up, N_up)
    pub gq: Array2<f64>,        // (N_up, M_dn)
    pub lam_p: Option<Array2<f64>>, // dense AGP/AGPn only; SD paired selector is implicit
    pub n_dn: usize,
    pub m_dn: usize,
}

pub(crate) fn gem_forward(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> GemFwd {
    gem_forward_gi(gem, r_up, r_dn, None)
}

/// `gem_forward` reusing a caller-supplied geminal inverse `g_inv == G^{-1}`.
/// `Some(g_inv)` skips the fresh LU; the MCMC
/// observable HF-force / drift path threads the Sherman-Morrison running inverse
/// the sweep emits, matching the e_L port's
/// `compute_grads_and_laplacian_ln_det_with_inv`. The fresh LU can NaN at a
/// near-singular geminal whereas the AS-regularized SM inverse is stable
/// (jqmc `compute_local_energy_fast` docstring). `None` = fresh LU (every existing
/// caller: value/param-grad/stress/LRDMC re-eval paths stay bit-identical). This
/// is the shared inverse-reuse plumbing used by force, stress, and parameter
/// gradient observables.
pub(crate) fn gem_forward_gi(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> GemFwd {
    gem_forward_gi_cut(gem, r_up, r_dn, g_inv, false)
}

/// [`gem_forward_gi`] with the FAST derivative-screened molecular AO grad/Laplacian
/// when `fast_cut` (see `orb_value_grad_lap_cut`): the MO gradient/Laplacian
/// tables of the FAST e_L kinetic.
pub(crate) fn gem_forward_gi_cut(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
    fast_cut: bool,
) -> GemFwd {
    gem_forward_gi_cut_tables(gem, r_up, r_dn, g_inv, fast_cut, None)
}

/// MO derivative tables `([x, y, z, lap] up, [x, y, z, lap] dn)`, each `(M, N)`.
pub(crate) type MoDerivTables<'a> = ([&'a Array2<f64>; 4], [&'a Array2<f64>; 4]);

/// [`gem_forward_gi_cut`] whose MO gradient / Laplacian tables are supplied by
/// the caller (`tables = Some`): the FAST chunk's carried kinetic tables
/// (`DetGradLapCache::mo_tables`, the same screened tables at the same positions),
/// so only the kinetic value table is evaluated here. Molecular MO leaves only
/// (no wrap, no PW / B-spline child); `None` evaluates every table.
pub(crate) fn gem_forward_gi_cut_tables(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
    fast_cut: bool,
    tables: Option<MoDerivTables>,
) -> GemFwd {
    // Home-cell-wrap for the geminal-VALUE observable path (jqmc `compute_AOs`): the batch
    // `orb_value_grad_lap` uses the un-wrapped `compute_aos`, so a far-roamed periodic walker
    // would underflow the static pGTO image tiling to a 0 geminal and NaN the drift/kinetic
    // fields. `None` (molecular / PW / B-spline) is the identity (near-atom bit-identical).
    let ruw = gem
        .orb_up
        .wrap_electrons(r_up)
        .unwrap_or_else(|| r_up.to_owned());
    let rdw = gem
        .orb_dn
        .wrap_electrons(r_dn)
        .unwrap_or_else(|| r_dn.to_owned());
    let (r_up, r_dn) = (ruw.view(), rdw.view());
    let fields = |orb: &OrbData, r: &ArrayView2<f64>, t: Option<[&Array2<f64>; 4]>| match t {
        None => orb_value_grad_lap_cut(orb, r, fast_cut),
        Some([x, y, z, l]) => {
            // the value table of `orb_value_grad_lap_cut` (fused value kernel, MO projection)
            debug_assert!(orb.pw.is_none() && orb.bspline.is_none());
            let av = crate::atomic_orbital::compute_aos_fused_value(&orb.aos, r);
            let c = orb
                .mo_coefficients
                .as_ref()
                .expect("carried tables are MO tables");
            (
                crate::determinant::matmul(&c.view(), &av.view()),
                x.clone(),
                y.clone(),
                z.clone(),
                l.clone(),
            )
        }
    };
    let (t_up, t_dn) = match tables {
        Some((u, d)) => (Some(u), Some(d)),
        None => (None, None),
    };
    let (ou_fused, gxu, gyu, gzu, lap_up) = fields(&gem.orb_up, &r_up, t_up);
    let (od_fused, gxd, gyd, gzd, lap_dn) = fields(&gem.orb_dn, &r_dn, t_dn);
    // A running MCMC/GFMC inverse is maintained from the standalone value-only
    // determinant-ratio tables. The kinetic forward, however, uses the values
    // emitted by the fused value/grad/lap kernel. Mixed precision makes these
    // two value tables numerically distinct, so retain both instead of silently
    // treating them as one primal. The fresh route owns only the fused primal
    // and therefore needs no separate value-only AO evaluation.
    let (ou, od) = if g_inv.is_some() {
        (gem.orb_up.compute(&r_up), gem.orb_dn.compute(&r_dn))
    } else {
        (ou_fused.clone(), od_fused.clone())
    };
    let m_dn = gem.orb_num_dn;
    let n_dn = r_dn.nrows();
    let n_up = ou.ncols();
    let w = gem.w_from_dn(&od.view(), n_up);
    let w_fused = gem.w_from_dn(&od_fused.view(), n_up);
    // G^{-1}: reuse the sweep SM inverse when supplied (near-singular-stable),
    // else a fresh LU of G = O_up^T W (the value/param-grad convention).
    let ginv = match g_inv {
        Some(gi) => gi.to_owned(),
        None => inverse(&ou.t().dot(&w).view()),
    };
    let q = gem.up_t_times_paired(&ou.view()); // inverse route (N_up, M_dn)
    let q_fused = gem.up_t_times_paired(&ou_fused.view()); // direct kinetic route
    let wg = w.dot(&ginv); // standalone determinant-ratio route (M_up, N_up)
    let gq = ginv.dot(&q); // standalone determinant-ratio route (N_up, M_dn)
    GemFwd {
        ou,
        od,
        ou_fused,
        od_fused,
        lap_up,
        lap_dn,
        gou: [gxu, gyu, gzu],
        godn: [gxd, gyd, gzd],
        w,
        w_fused,
        ginv,
        q,
        q_fused,
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

/// `gD` = per-electron `gradln|detG|` fields, `(N_up,3)` and `(N_dn,3)`.
pub fn det_grad_field(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    det_grad_field_gi(gem, r_up, r_dn, None)
}

/// `det_grad_field` reusing a caller-supplied geminal inverse (see `gem_forward_gi`).
pub(crate) fn det_grad_field_gi(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    det_grad_field_from_fwd(&gem_forward_gi(gem, r_up, r_dn, g_inv))
}

/// [`det_grad_field_gi`] on an already evaluated [`gem_forward_gi`] forward.
pub(crate) fn det_grad_field_from_fwd(f: &GemFwd) -> (Array2<f64>, Array2<f64>) {
    let n_up = f.ou.ncols();
    let m_up = f.ou.nrows();
    let n_dn = f.n_dn;
    let m_dn = f.m_dn;
    // gD_up[k,d] = Sum_a gou[d][a,k]*WG[a,k]
    let mut gd_up = Array2::<f64>::zeros((n_up, 3));
    for d in 0..3 {
        for k in 0..n_up {
            let mut s = 0.0;
            for a in 0..m_up {
                s += f.gou[d][[a, k]] * f.wg[[a, k]];
            }
            gd_up[[k, d]] = s;
        }
    }
    // gD_dn[l,d] = Sum_b godn[d][b,l]*GQ[l,b]  (GQ rows 0..N_dn)
    let mut gd_dn = Array2::<f64>::zeros((n_dn, 3));
    for d in 0..3 {
        for l in 0..n_dn {
            let mut s = 0.0;
            for b in 0..m_dn {
                s += f.godn[d][[b, l]] * f.gq[[l, b]];
            }
            gd_dn[[l, d]] = s;
        }
    }
    (gd_up, gd_dn)
}

/// Cotangent bundle from `_geminal_reverse` for the kinetic derivative with Jastrow.
pub(crate) struct GemReverse {
    pub bar_orb_up: Array2<f64>,
    pub bar_orb_dn: Array2<f64>,
    pub bar_orb_up_inverse: Array2<f64>,
    pub bar_orb_dn_inverse: Array2<f64>,
    pub bar_orb_up_fused: Array2<f64>,
    pub bar_orb_dn_fused: Array2<f64>,
    pub bar_lap_up: Array2<f64>,
    pub bar_lap_dn: Array2<f64>,
    pub bar_gou: [Array2<f64>; 3],
    pub bar_godn: [Array2<f64>; 3],
    pub bar_lam_p: Array2<f64>,
    pub bar_lam_unp: Array2<f64>,
}

/// Reverse pass of `T` (with Jastrow grads `gJ`) -> cotangents. With `gJ=0` this
/// reduces to the jnone det-kinetic VJP.
pub(crate) fn geminal_reverse(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
) -> GemReverse {
    geminal_reverse_w(gem, r_up, r_dn, gj_up, gj_dn, None, None, None)
}

/// `geminal_reverse` with per-electron seed weights `w_up`/`w_dn`: the cotangents
/// are those of `Sum_k w_k*kc_k` (det part) instead of the full `T = Sum_k kc_k`.
/// All seeds are per-electron indexed (columns k for up, rows l for dn), so
/// weighting the seed columns/rows gives the masked-kinetic VJP exactly
/// (GFMC ZV branch corrections need `Sum_{kinS'} kc_k` derivatives). `None` = all-1.
#[allow(
    clippy::too_many_arguments,
    reason = "per-spin drift and weight inputs of the geminal reverse pass"
)]
pub(crate) fn geminal_reverse_w(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    w_up: Option<&[f64]>,
    w_dn: Option<&[f64]>,
    g_inv: Option<&ArrayView2<f64>>,
) -> GemReverse {
    let f = gem_forward_gi(gem, r_up, r_dn, g_inv);
    geminal_reverse_from_fwd(&f, gj_up, gj_dn, w_up, w_dn)
}

/// [`geminal_reverse_w`] on an already evaluated [`gem_forward_gi`] forward.
pub(crate) fn geminal_reverse_from_fwd(
    f: &GemFwd,
    gj_up: &Array2<f64>,
    gj_dn: &Array2<f64>,
    w_up: Option<&[f64]>,
    w_dn: Option<&[f64]>,
) -> GemReverse {
    let wg_fused = f.w_fused.dot(&f.ginv);
    let gq_fused = f.ginv.dot(&f.q_fused);
    let n_up = f.ou.ncols();
    let m_up = f.ou.nrows();
    let n_dn = f.n_dn;
    let m_dn = f.m_dn;
    let wu = |k: usize| -> f64 { w_up.map_or(1.0, |w| w[k]) };
    let wd = |l: usize| -> f64 { w_dn.map_or(1.0, |w| w[l]) };

    // seeds
    // bar_lap_up = -0.5*w_k*WG (M_up, N_up)
    let mut bar_lap_up = &wg_fused * -0.5;
    if w_up.is_some() {
        for a in 0..m_up {
            for k in 0..n_up {
                bar_lap_up[[a, k]] *= wu(k);
            }
        }
    }
    // bar_lap_dn = -0.5 * w_l * GQ[:N_dn]T
    let mut bar_lap_dn = Array2::<f64>::zeros((m_dn, n_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            bar_lap_dn[[b, l]] = -0.5 * wd(l) * gq_fused[[l, b]];
        }
    }
    // bar_WG[a,k] = w_k*(-0.5*lap_up[a,k] - Sum_d gJ_up[k,d]*gou[d][a,k])
    let mut bar_wg = Array2::<f64>::zeros((m_up, n_up));
    for a in 0..m_up {
        for k in 0..n_up {
            let mut s = -0.5 * f.lap_up[[a, k]];
            for d in 0..3 {
                s -= gj_up[[k, d]] * f.gou[d][[a, k]];
            }
            bar_wg[[a, k]] = wu(k) * s;
        }
    }
    // bar_gou[d][a,k] = -w_k*gJ_up[k,d]*WG[a,k]
    let mut bar_gou = [
        Array2::<f64>::zeros((m_up, n_up)),
        Array2::<f64>::zeros((m_up, n_up)),
        Array2::<f64>::zeros((m_up, n_up)),
    ];
    for d in 0..3 {
        for a in 0..m_up {
            for k in 0..n_up {
                bar_gou[d][[a, k]] = -wu(k) * gj_up[[k, d]] * wg_fused[[a, k]];
            }
        }
    }
    // bar_GQ rows (N_dn, M_dn): [l,b] = w_l*(-0.5*lap_dn[b,l] - Sum_d gJ_dn[l,d]*godn[d][b,l])
    let mut bar_gq = Array2::<f64>::zeros((n_up, m_dn));
    for l in 0..n_dn {
        for b in 0..m_dn {
            let mut s = -0.5 * f.lap_dn[[b, l]];
            for d in 0..3 {
                s -= gj_dn[[l, d]] * f.godn[d][[b, l]];
            }
            bar_gq[[l, b]] = wd(l) * s;
        }
    }
    // bar_godn[d][b,l] = -w_l*gJ_dn[l,d]*GQ[l,b]
    let mut bar_godn = [
        Array2::<f64>::zeros((m_dn, n_dn)),
        Array2::<f64>::zeros((m_dn, n_dn)),
        Array2::<f64>::zeros((m_dn, n_dn)),
    ];
    for d in 0..3 {
        for b in 0..m_dn {
            for l in 0..n_dn {
                bar_godn[d][[b, l]] = -wd(l) * gj_dn[[l, d]] * gq_fused[[l, b]];
            }
        }
    }

    // back-propagate
    let bar_ginv = f.w_fused.t().dot(&bar_wg) + bar_gq.dot(&f.q_fused.t()); // (N_up,N_up)
    let bar_w_direct = bar_wg.dot(&f.ginv.t()); // (M_up,N_up)
    let bar_q_direct = f.ginv.t().dot(&bar_gq); // (N_up,M_dn)
    let bar_g = f.ginv.t().dot(&bar_ginv).dot(&f.ginv.t()) * -1.0; // -GinvT*bar_Ginv*GinvT
    let bar_w_inverse = f.ou.dot(&bar_g); // (M_up,N_up)
    let bar_q_term = if let Some(lam_p) = &f.lam_p {
        lam_p.dot(&bar_q_direct.t())
    } else {
        let mut out = Array2::<f64>::zeros((m_up, n_up));
        out.slice_mut(numpy::ndarray::s![0..m_dn, ..])
            .assign(&bar_q_direct.t());
        out
    };
    let bar_orb_up_inverse = f.w.dot(&bar_g.t()); // (M_up,N_up)
    let bar_orb_up_fused = bar_q_term;
    let bar_w_inverse_left = bar_w_inverse
        .slice(numpy::ndarray::s![.., 0..n_dn])
        .to_owned(); // (M_up,N_dn)
    let bar_w_direct_left = bar_w_direct
        .slice(numpy::ndarray::s![.., 0..n_dn])
        .to_owned(); // (M_up,N_dn)
    let bar_orb_dn_inverse = if let Some(lam_p) = &f.lam_p {
        lam_p.t().dot(&bar_w_inverse_left)
    } else {
        bar_w_inverse_left
            .slice(numpy::ndarray::s![0..m_dn, ..])
            .to_owned()
    };
    let bar_orb_dn_fused = if let Some(lam_p) = &f.lam_p {
        lam_p.t().dot(&bar_w_direct_left)
    } else {
        bar_w_direct_left
            .slice(numpy::ndarray::s![0..m_dn, ..])
            .to_owned()
    };
    let (bar_lam_p, bar_lam_unp) = if f.lam_p.is_some() {
        (
            bar_w_direct_left.dot(&f.od_fused.t())
                + f.ou_fused.dot(&bar_q_direct)
                + bar_w_inverse_left.dot(&f.od.t()),
            bar_w_direct
                .slice(numpy::ndarray::s![.., n_dn..])
                .to_owned()
                + bar_w_inverse
                    .slice(numpy::ndarray::s![.., n_dn..])
                    .to_owned(),
        )
    } else {
        (
            Array2::<f64>::zeros((m_up, m_dn)),
            Array2::<f64>::zeros((m_up, n_up - n_dn)),
        )
    };

    GemReverse {
        bar_orb_up: &bar_orb_up_inverse + &bar_orb_up_fused,
        bar_orb_dn: &bar_orb_dn_inverse + &bar_orb_dn_fused,
        bar_orb_up_inverse,
        bar_orb_dn_inverse,
        bar_orb_up_fused,
        bar_orb_dn_fused,
        bar_lap_up,
        bar_lap_dn,
        bar_gou,
        bar_godn,
        bar_lam_p,
        bar_lam_unp,
    }
}

pub(crate) fn jastrow_grads(
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    jastrow_grads_cut(jastrow, j3_orb, j3_jm, r_up, r_dn, false)
}

/// [`jastrow_grads`] whose J3 part is the FAST derivative-screened kernel
/// (`compute_grads_and_laplacian_jastrow_three_body_fast`, the J3 drift of the
/// FAST e_L kinetic) when `fast_j3`.
pub(crate) fn jastrow_grads_cut(
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    fast_j3: bool,
) -> (Array2<f64>, Array2<f64>) {
    // J1/J2/DTN part (J3 is a separate kernel, added below).
    let (mut gu, mut gd) = match jastrow {
        None => (
            Array2::zeros((r_up.nrows(), 3)),
            Array2::zeros((r_dn.nrows(), 3)),
        ),
        Some(jd) => {
            let (gu, gd, _lu, _ld) = compute_grads_and_laplacian_jastrow_part(jd, r_up, r_dn);
            (gu, gd)
        }
    };
    if let (Some(orb), Some(jm)) = (j3_orb, j3_jm) {
        let (j3u, j3d, _, _) = if fast_j3 {
            crate::jastrow_factor::compute_grads_and_laplacian_jastrow_three_body_fast(
                orb, jm, r_up, r_dn,
            )
        } else {
            compute_grads_and_laplacian_jastrow_three_body(orb, jm, r_up, r_dn)
        };
        gu = gu + j3u;
        gd = gd + j3d;
    }
    (gu, gd)
}

/// Jastrow drift from the channels that must be RECOMPUTED per configuration.
///
/// The split is by CAPABILITY, not by channel identity: a channel appears here
/// when it has no rank-one move delta, and in
/// [`jastrow_drift_move_delta`] when it has one. A mesh caller therefore states
/// the move and never names a channel; this function decides which channels
/// contribute to each side.
///
/// Callers that want the whole Jastrow drift at one configuration want
/// [`jastrow_grads`] instead.
pub(crate) fn jastrow_grads_recomputed_channels(
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut gu = Array2::<f64>::zeros((r_up.nrows(), 3));
    let mut gd = Array2::<f64>::zeros((r_dn.nrows(), 3));
    if let Some(jd) = jastrow {
        if let Some(ob) = &jd.one_body {
            let (g_up, _) = crate::jastrow_factor::j1_grad_lap_one_spin(ob, r_up);
            let (g_dn, _) = crate::jastrow_factor::j1_grad_lap_one_spin(ob, r_dn);
            gu += &g_up;
            gd += &g_dn;
        }
        if let Some(tb) = &jd.two_body {
            let (g_up, g_dn, _, _) = crate::jastrow_factor::j2_grad_lap(tb, r_up, r_dn);
            gu += &g_up;
            gd += &g_dn;
        }
    }
    if let (Some(orb), Some(jm)) = (j3_orb, j3_jm) {
        let (j3u, j3d, _, _) = compute_grads_and_laplacian_jastrow_three_body(orb, jm, r_up, r_dn);
        gu += &j3u;
        gd += &j3d;
    }
    (gu, gd)
}

/// Jastrow drift CHANGE from the channels that expose a rank-one move delta,
/// for the one-electron move `(is_up, li) -> new_pos` off the base configuration.
///
/// Twin of [`jastrow_grads_recomputed_channels`]: between them the two cover
/// every channel exactly once, so `recomputed(moved) - recomputed(base) +
/// move_delta(base -> moved)` is the whole Jastrow drift change. `None` means no
/// active channel offers a delta, i.e. the recompute difference is already
/// complete.
#[allow(dead_code)]
pub(crate) fn jastrow_drift_move_delta(
    jastrow: Option<&JastrowData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_pos: [f64; 3],
) -> Option<(Array2<f64>, Array2<f64>)> {
    let dtn = jastrow?.dtn.as_ref()?;
    Some(crate::dtn::dtn_drift_move_delta(
        dtn, r_up, r_dn, is_up, li, new_pos,
    ))
}

/// Nuclear-Pulay counterpart of [`jastrow_drift_move_delta`]: the `d/dR` change
/// contributed by the rank-one channels for the same one-electron move.
#[allow(dead_code)]
pub(crate) fn jastrow_pulay_move_delta(
    jastrow: Option<&JastrowData>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    li: usize,
    new_pos: [f64; 3],
) -> Option<Array2<f64>> {
    let dtn = jastrow?.dtn.as_ref()?;
    Some(crate::forces_dtn::grad_j_dtn_wrt_r_atom_move_delta(
        dtn, r_up, r_dn, is_up, li, new_pos,
    ))
}

/// Home-cell wrap of both spin coordinate blocks for the AO-evaluation half of a
/// `dT/dp` observable -- the same "periodicity guarantor" contract as
/// [`crate::determinant::compute_grads_and_laplacian_ln_det`] and
/// `param_grad_ecp::collect_mesh`. jqmc wraps every electron coordinate before an
/// AO eval, because the truncated periodic image sum is enumerated relative to the
/// home cell: a RAW coordinate one lattice vector out falls beyond the static
/// tiling, so the AO (and hence the geminal built from it) collapses. Wrapping at
/// the entry -- not inside the shared low-level AO evaluations -- keeps the
/// per-move MCMC value path identical to JAX's accept/reject calculation; these `dT/dp`
/// kernels are never on the Metropolis path. No-op for molecular / pure-PW leaves
/// (`wrap_cell == None`).
///
/// CRITICAL: the wrapped coordinates feed ONLY the AO stack (determinant + J3
/// orbital). The Jastrow-metric fields (J1 / J2 / DTN electron distances) MUST keep
/// the RAW coordinate: jqmc evaluates them with the Jastrow's OWN structure_data,
/// which for many wavefunctions carries `pbc_flag == False` -> a molecular
/// (Euclidean) J2 that is NOT wrap-invariant. Wrapping such a J2 (each electron
/// independently into `[0,1)`) splits a nearby pair across the cell and diverges
/// from XLA, which never wraps the Jastrow input. Periodic (smooth / min-image)
/// J1/J2/DTN are wrap-invariant, so RAW is exact for them too. See `kinetic_drift`
/// / `jastrow_grads_split`, which route raw vs wrapped per channel.
pub(crate) fn wrap_both(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    (
        crate::atomic_orbital::wrap_r(&gem.orb_up.aos, r_up),
        crate::atomic_orbital::wrap_r(&gem.orb_dn.aos, r_dn),
    )
}

/// Jastrow drift `gradln|Psi_J|` (the geminal_reverse seed) with the wrap policy
/// split: J1/J2/DTN on the RAW coordinate (molecular / min-image metric -- matches
/// jqmc, which wraps only inside compute_AOs), J3 on the WRAPPED coordinate (its
/// orbital is an AO stack subject to the image-sum truncation). Molecular / pure-PW
/// leaves make raw == wrapped, so this reduces to `jastrow_grads` there.
#[allow(clippy::too_many_arguments)]
fn jastrow_grads_split(
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rw_up: &ArrayView2<f64>,
    rw_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (mut gu, mut gd) = match jastrow {
        None => (
            Array2::zeros((r_up.nrows(), 3)),
            Array2::zeros((r_dn.nrows(), 3)),
        ),
        Some(jd) => {
            let (gu, gd, _lu, _ld) = compute_grads_and_laplacian_jastrow_part(jd, r_up, r_dn);
            (gu, gd)
        }
    };
    if let (Some(orb), Some(jm)) = (j3_orb, j3_jm) {
        let (j3u, j3d, _, _) =
            compute_grads_and_laplacian_jastrow_three_body(orb, jm, rw_up, rw_dn);
        gu = gu + j3u;
        gd = gd + j3d;
    }
    (gu, gd)
}

/// Full kinetic drift `gradln|Psi| = gJ + gD` under the wrap policy split: the
/// Jastrow metric (J1/J2/DTN) on RAW, J3 orbital + determinant AO on WRAPPED.
/// Consumed by the Jastrow `dT/dp` reductions (`grad_t_j1/j2/j3`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kinetic_drift(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    rw_up: &ArrayView2<f64>,
    rw_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (gj_up, gj_dn) = jastrow_grads_split(jastrow, j3_orb, j3_jm, r_up, r_dn, rw_up, rw_dn);
    let (gd_up, gd_dn) = det_grad_field(gem, rw_up, rw_dn);
    (&gj_up + &gd_up, &gj_dn + &gd_dn)
}

/// `dT/dlambda` (with optional Jastrow); `gJ=0` when `jastrow=None` -> jnone.
pub fn grad_t_lambda(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rw_up, rw_dn) = (&rw_up.view(), &rw_dn.view());
    // Jastrow metric on RAW (`r_up`/`r_dn`), J3 orbital on WRAPPED; the geminal
    // reverse (AO stack) runs on WRAPPED.
    let (gj_up, gj_dn) = jastrow_grads_split(jastrow, j3_orb, j3_jm, r_up, r_dn, rw_up, rw_dn);
    let b = geminal_reverse(gem, rw_up, rw_dn, &gj_up, &gj_dn);
    let m_up = b.bar_lam_p.nrows();
    let ncols = gem
        .lambda
        .as_ref()
        .expect("grad_t_lambda is undefined for occupied-only SD")
        .ncols();
    let mut out = Array2::<f64>::zeros((m_up, ncols));
    out.slice_mut(numpy::ndarray::s![.., 0..b.bar_lam_p.ncols()])
        .assign(&b.bar_lam_p);
    out.slice_mut(numpy::ndarray::s![.., b.bar_lam_p.ncols()..])
        .assign(&b.bar_lam_unp);
    out
}

/// AGPn dE_L on the natural-orbital leaves: `dE_L/d lambda_diag` = paired-block
/// DIAGONAL of the dense-`lambda` kinetic gradient, `dE_L/d lambda_unpaired` = its
/// unpaired columns (same chain rule as `grad_ln_det_lambda_diag`; real path).
pub fn grad_t_lambda_diag(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    // Closed-shell full-occupation AGPn factorization:
    // det(Ou^T diag(lambda) Od) = det(Ou^T) prod(lambda) det(Od).
    // Lambda changes only a configuration-independent normalization, hence
    // dE_L/dlambda is exactly zero.  Pin the invariant before the reverse
    // pass so ill-conditioned PW/B-spline matrices cannot amplify the
    // algebraic cancellation into platform-dependent noise.
    if gem.n_unpaired() == 0 && gem.orb_num_up() == r_up.nrows() && gem.orb_num_dn == r_dn.nrows() {
        return (
            Array1::zeros(gem.orb_num_dn),
            Array2::zeros((gem.orb_num_up(), 0)),
        );
    }
    let dense = grad_t_lambda(gem, jastrow, j3_orb, j3_jm, r_up, r_dn);
    let m_dn = gem.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = Array1::from_iter((0..n).map(|k| dense[[k, k]]));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    (diag, unpaired)
}

/// `dT/dC` (up, dn mo_coefficients) with optional Jastrow.
pub fn grad_t_mo(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    grad_t_mo_gi(gem, jastrow, j3_orb, j3_jm, r_up, r_dn, None)
}

/// `dT/dC` reusing a caller-supplied MCMC geminal inverse when available.
///
/// Production parameter derivatives must use the same Sherman-Morrison inverse
/// carried by the sweep as the XLA route. `None` builds a fresh LU inverse,
/// for standalone/free-kernel callers.
pub fn grad_t_mo_gi(
    gem: &GeminalData,
    jastrow: Option<&JastrowData>,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rw_up, rw_dn) = (&rw_up.view(), &rw_dn.view());
    // Jastrow metric on RAW, J3 orbital + geminal AO stack on WRAPPED.
    let (gj_up, gj_dn) = jastrow_grads_split(jastrow, j3_orb, j3_jm, r_up, r_dn, rw_up, rw_dn);
    let b = geminal_reverse_w(gem, rw_up, rw_dn, &gj_up, &gj_dn, None, None, g_inv);
    let (r_up, r_dn) = (rw_up, rw_dn);
    let to_c = |orb: &OrbData,
                r: &ArrayView2<f64>,
                bar_orb_inverse: &Array2<f64>,
                bar_orb_fused: &Array2<f64>,
                bar_lap: &Array2<f64>,
                bar_g: &[Array2<f64>; 3]|
     -> Array2<f64> {
        // Full AO stack (gto + composite PW child, or pure PW / B-spline), matching
        // the `mo_coefficients` column order, so `dT/dC` covers every AO -- not just
        // the GTO block (the composite/plane-wave gap).
        let (av, gx, gy, gz, lap) = crate::determinant::aos_stack_value_grad_lap(orb, r);
        let mut gc = bar_orb_fused.dot(&av.t()) + bar_lap.dot(&lap.t());
        gc = if g_inv.is_some() {
            gc + bar_orb_inverse.dot(&orb.eval_aos(r).t())
        } else {
            gc + bar_orb_inverse.dot(&av.t())
        };
        gc = gc + bar_g[0].dot(&gx.t()) + bar_g[1].dot(&gy.t()) + bar_g[2].dot(&gz.t());
        gc
    };
    let mut gc_up = to_c(
        &gem.orb_up,
        r_up,
        &b.bar_orb_up_inverse,
        &b.bar_orb_up_fused,
        &b.bar_lap_up,
        &b.bar_gou,
    );
    let mut gc_dn = to_c(
        &gem.orb_dn,
        r_dn,
        &b.bar_orb_dn_inverse,
        &b.bar_orb_dn_fused,
        &b.bar_lap_dn,
        &b.bar_godn,
    );
    // If occupied SD spans the complete one-particle basis, C is an invertible
    // configuration-independent change of basis.  It multiplies Psi by det(C)
    // but cannot change local energy, so dE_L/dC is exactly zero.  Returning the
    // invariant directly avoids condition-number-amplified cancellation noise.
    if gem.is_sd() {
        if gc_up.nrows() == gc_up.ncols() {
            gc_up.fill(0.0);
        }
        if gc_dn.nrows() == gc_dn.ncols() {
            gc_dn.fill(0.0);
        }
    }
    (gc_up, gc_dn)
}

/// J1 per-electron field `f_c = dJ1/dp`: returns `(grad_f [n_e][3], lap_f [n_e])`.
/// Basis/geminal-INDEPENDENT (electron-nucleus distances + params only), so the same
/// field serves the molecular, real-Gamma, AND complex-twist kinetic reductions.
pub(crate) fn j1_field(
    j1: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let elec = |g: usize| -> [f64; 3] {
        if g < n_up {
            [r_up[[g, 0]], r_up[[g, 1]], r_up[[g, 2]]]
        } else {
            let k = g - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let r_atoms = j1.structure_data.positions.view();
    let n_atom = j1.structure_data.n_atoms();
    let p = j1.jastrow_1b_param;
    let is_exp = j1.jastrow_1b_type == "exp";
    let ci = j1
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let (mut grad_f, mut lap_f) = (vec![[0.0_f64; 3]; n_e], vec![0.0_f64; n_e]);
    for e in 0..n_e {
        let re = elec(e);
        for at in 0..n_atom {
            let z = j1.structure_data.atomic_numbers[at] - j1.core_electrons[at];
            let c = (2.0 * z).powf(0.25);
            let pref = if is_exp {
                -(2.0 * z).powf(0.75)
            } else {
                -2.0 * z
            };
            let a = p * c;
            let disp = [
                re[0] - r_atoms[[at, 0]],
                re[1] - r_atoms[[at, 1]],
                re[2] - r_atoms[[at, 2]],
            ];
            let (dist, gd, q, lap_d) = dist_field(disp, ci.as_ref());
            let (dterm_p, dterm_pp) = if is_exp {
                let ee = (-a * dist).exp();
                (
                    -pref * (c * c / 2.0) * dist * ee,
                    -pref * (c * c / 2.0) * ee * (1.0 - a * dist),
                )
            } else {
                let aa = -pref * c / 2.0;
                let u = 1.0 + a * dist;
                (
                    2.0 * aa * dist / u.powi(3),
                    2.0 * aa * (1.0 - 2.0 * a * dist) / u.powi(4),
                )
            };
            grad_f[e][0] += dterm_p * gd[0];
            grad_f[e][1] += dterm_p * gd[1];
            grad_f[e][2] += dterm_p * gd[2];
            lap_f[e] += dterm_pp * q + dterm_p * lap_d;
        }
    }
    (grad_f, lap_f)
}

/// `dT/dJ1_param` (scalar).
pub fn grad_t_j1(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j1: &JastrowOneBodyData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rw_up, rw_dn) = (&rw_up.view(), &rw_dn.view());
    // dJ1/dp field on RAW (Jastrow metric); drift splits raw vs wrapped internally.
    let (grad_f, lap_f) = j1_field(j1, r_up, r_dn);
    let (drift_up, drift_dn) =
        kinetic_drift(gem, Some(jastrow), j3_orb, j3_jm, r_up, r_dn, rw_up, rw_dn);
    reduce_field(&grad_f, &lap_f, &drift_up, &drift_dn, r_up.nrows())
}

/// J2 per-electron field `f_c = dJ2/dp`: returns `(grad_f [n_e][3], lap_f [n_e])`.
/// Distance-only (basis/geminal-INDEPENDENT) -> shared by molecular / Gamma / twist.
pub(crate) fn j2_field(
    j2: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n_e = n_up + n_dn;
    let elec = |g: usize| -> [f64; 3] {
        if g < n_up {
            [r_up[[g, 0]], r_up[[g, 1]], r_up[[g, 2]]]
        } else {
            let k = g - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let p = j2.jastrow_2b_param;
    // g(r) = d(u)/d(param) is the per-pair field derivative; gp = d g/dr and
    // gpp = d^2 g/dr^2 are its radial derivatives. Pade and exp J2 have distinct
    // forms (see jqmc compute_Jastrow_two_body): the exp branch must NOT reuse
    // the pade radial derivatives.
    let is_exp = match j2.jastrow_2b_type.as_str() {
        "exp" => true,
        "pade" => false,
        other => panic!("Unknown jastrow_2b_type: {other}"),
    };
    let ci = j2.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let (mut grad_f, mut lap_f) = (vec![[0.0_f64; 3]; n_e], vec![0.0_f64; n_e]);
    for k in 0..n_e {
        let rk = elec(k);
        for j in 0..n_e {
            if j == k {
                continue;
            }
            let rj = elec(j);
            let disp = [rk[0] - rj[0], rk[1] - rj[1], rk[2] - rj[2]];
            let (dist, gd, q, lap_d) = dist_field(disp, ci.as_ref());
            // u(r) = (1/(2p))*(1 - exp(-p r))  [exp]  or  r/(2(1+p r))  [pade];
            // g(r) = du/dp; here gp/gpp are the radial 1st/2nd derivatives of g.
            let (gp, gpp) = if is_exp {
                let e = (-p * dist).exp();
                (-0.5 * dist * e, 0.5 * (p * dist - 1.0) * e)
            } else {
                let u = 1.0 + p * dist;
                (-dist / u.powi(3), (2.0 * p * dist - 1.0) / u.powi(4))
            };
            grad_f[k][0] += gp * gd[0];
            grad_f[k][1] += gp * gd[1];
            grad_f[k][2] += gp * gd[2];
            lap_f[k] += gpp * q + gp * lap_d;
        }
    }
    (grad_f, lap_f)
}

/// `dT/dJ2_param` (scalar).
pub fn grad_t_j2(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j2: &JastrowTwoBodyData,
    j3_orb: Option<&OrbData>,
    j3_jm: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rw_up, rw_dn) = (&rw_up.view(), &rw_dn.view());
    // dJ2/dp field on RAW (Jastrow metric -- a molecular J2 is NOT wrap-invariant);
    // drift splits raw (J1/J2/DTN) vs wrapped (J3 orbital + determinant) internally.
    let (grad_f, lap_f) = j2_field(j2, r_up, r_dn);
    let (drift_up, drift_dn) =
        kinetic_drift(gem, Some(jastrow), j3_orb, j3_jm, r_up, r_dn, rw_up, rw_dn);
    reduce_field(&grad_f, &lap_f, &drift_up, &drift_dn, r_up.nrows())
}

/// Pure reduction `-1/2Sumgrad^2f - Sum(drift)*gradf` given the combined per-electron
/// cotangent drift `(gJ+gD)` (`drift_up (n_up,3)`, `drift_dn (n_dn,3)`). Basis-agnostic:
/// the real path feeds `gJ + Re(gD_real)`, the complex-twist path feeds `gJ + Re(gD_complex)`.
pub(crate) fn reduce_field(
    grad_f: &[[f64; 3]],
    lap_f: &[f64],
    drift_up: &Array2<f64>,
    drift_dn: &Array2<f64>,
    n_up: usize,
) -> f64 {
    let n_e = grad_f.len();
    let mut acc = 0.0_f64;
    for e in 0..n_e {
        acc += -0.5 * lap_f[e];
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

/// `dT/dj_matrix` (J3), shape `(orb, orb+1)`.
pub fn grad_t_j3(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3_orb: &OrbData,
    j3_jm: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let (rw_up, rw_dn) = wrap_both(gem, r_up, r_dn);
    let (rw_up, rw_dn) = (&rw_up.view(), &rw_dn.view());
    // dJ3/dj_matrix field is an AO (orbital) stack -> WRAPPED. The kinetic drift
    // routes J1/J2/DTN on RAW and J3 orbital + determinant on WRAPPED.
    let (drift_up, drift_dn) = kinetic_drift(
        gem,
        Some(jastrow),
        Some(j3_orb),
        Some(j3_jm),
        r_up,
        r_dn,
        rw_up,
        rw_dn,
    );
    let (r_up, r_dn) = (rw_up, rw_dn);
    let (a_up, gxu, gyu, gzu, la_up) = orb_value_grad_lap(j3_orb, r_up);
    let (a_dn, gxd, gyd, gzd, la_dn) = orb_value_grad_lap(j3_orb, r_dn);
    let ga_up = [gxu, gyu, gzu];
    let ga_dn = [gxd, gyd, gzd];
    let orb = a_up.nrows();
    let n_up = a_up.ncols();
    let n_dn = a_dn.ncols();

    // h = -(gJ+gD)
    let h_up = -&drift_up; // (N_up,3)
    let h_dn = -&drift_dn;
    // hGA[a,k] = Sum_d h[k,d]*GA[d][a,k]
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

    let s_up = a_up.sum_axis(Axis(1)); // (orb,)
    let s_dn = a_dn.sum_axis(Axis(1));
    // strict-upper ones
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
    // block(coeff_up (orb,N_up), coeff_dn (orb,N_dn)) -> (orb,orb)
    let block = |cu: &Array2<f64>, cd: &Array2<f64>| -> Array2<f64> {
        let uu = cu.dot(&k2u.dot(&a_up.t())) + a_up.dot(&k2u).dot(&cu.t());
        let dd = cd.dot(&k2d.dot(&a_dn.t())) + a_dn.dot(&k2d).dot(&cd.t());
        let cus = cu.sum_axis(Axis(1)); // (orb,)
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
    // j1col grad
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
