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

//! `dV_ecp(non-local)/dc` via the O_k(r') identity. The non-local ECP energy is
//! `V = Sum_g V_grid_g*(Psi(r'_g)/Psi(r))` with c-independent weights/projector, so
//! `dV/dc = Sum_g V_grid_g*O_k(r'_g) - V*O_k(r)`, `O_k=dlnPsi/dc` for geminal
//! parameters because the Jastrow does not depend on lambda/C. This path is
//! autograd-free and reuses the value-gradient kernels.

use numpy::ndarray::{Array1, Array2, ArrayView2};

use crate::coulomb_potential::legendre;
use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{compute_geminal_all_elements, inverse, GeminalData, OrbData};
use crate::param_grad::{
    grad_j1_param, grad_j2_param, grad_j3_jmatrix, grad_ln_det_lambda_from_orb,
    grad_ln_det_mo_from_ao, j1_param_move_delta, j2_param_row, j2_partners,
    j3_jmatrix_mesh_increment, j3_mesh_column_changes, mesh_positions,
};
use crate::param_grad_basis;
use crate::param_grad_dtn::{DtnMovedOk, DtnOkData};

/// Per-electron nearest-ECP-nucleus count used by the production non-local e_L
/// (`jqmc._setting.NN_default`); the `dV_ecp_nl/dc` mesh must match that same set so
/// its derivative equals `compute_grad_e_L` (which differentiates the NN e_L).
pub(crate) const NN_DEFAULT: usize = 1;

/// One mesh point: its non-local weight `V_grid` and the single-electron
/// displacement (which spin, which electron, new Cartesian position).
pub(crate) struct MeshPoint {
    pub v_grid: f64,
    pub is_up: bool,
    pub idx: usize,
    pub pos: [f64; 3],
}

/// Enumerate the non-local ECP mesh (full-recompute path, 1:1 with
/// `compute_ecp_nonlocal_parts_all_pairs`'s `geminal_inv=None` branch),
/// collecting per-grid weights and displaced configs. Returns `(mesh, sumV)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_mesh(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    // Periodic min-image of the electron->nucleus displacement (None => molecular).
    // 1:1 with the value path `compute_ecp_nonlocal_parts_all_pairs` `rel_vec`.
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    // Per-electron nearest-`nn`-ECP-nucleus filter (production e_L uses NN_default=1,
    // 1:1 with `compute_ecp_non_local_parts_nearest_neighbors`). `0` => all-pairs.
    nn: usize,
) -> (Vec<MeshPoint>, f64) {
    let nv = weights.len();
    let mut mesh = Vec::new();
    let mut sum_v = 0.0_f64;
    let inv_cell = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    // ---- Fast one-electron-delta base (mirrors the value path
    // `compute_ecp_nonlocal_parts_all_pairs` SM branch, `geminal_inv=Some`): each
    // mesh point moves ONE electron, so the det ratio is a rank-1 Sherman-Morrison
    // update of a SINGLE base `G^{-1}` (no per-point O(N^3) geminal rebuild + LU),
    // and the Jastrow factor is the incremental single-move ratio (no per-point
    // full `jastrow_value_for_ratio` recompute). Byte-consistent with the native e_L
    // (both go through the same SM det ratio + `jastrow_1b2b_ratio`/`j3_ratio`/DTN
    // mesh ratio), so `v_grid`/`sum_v` differ only at the fp floor (<~1e-13) from a
    // full per-point recompute. Home-cell-wrap the base + mesh AO evals
    // (identity for molecular / PW), keeping the RAW geometry for the Legendre
    // angle and the min-image Jastrow ratios -- exactly the value path's split. ----
    let r_up_ao = crate::atomic_orbital::wrap_r(&geminal.orb_up.aos, r_up);
    let r_dn_ao = crate::atomic_orbital::wrap_r(&geminal.orb_dn.aos, r_dn);
    let r_up_ao_v = r_up_ao.view();
    let r_dn_ao_v = r_dn_ao.view();
    let det_state = crate::determinant::init_det_ratio_state(geminal, &r_up_ao_v, &r_dn_ao_v);
    let ginv = crate::determinant::inverse(
        &compute_geminal_all_elements(geminal, &r_up_ao_v, &r_dn_ao_v).view(),
    );
    // Fast single-move Jastrow-ratio state for the J3 channel (built on the wrapped
    // base orbitals, matching the wrapped mesh orbitals used per point below).
    let j3_ratio_state = j3.map(|(orb, jm)| {
        let au = orb.compute(&r_up_ao_v);
        let ad = orb.compute(&r_dn_ao_v);
        (
            jm,
            crate::jastrow_factor::init_j3_ratio_state_with_policy(
                jm,
                &au.view(),
                &ad.view(),
                jastrow.precision_policy.is_f32("jastrow_ratio"),
            ),
        )
    });
    // rel = R - r_e, Wigner-Seitz minimum-imaged under PBC (else raw Euclidean).
    let rel_vec = |rx0: f64, ry0: f64, rz0: f64| -> [f64; 3] {
        match (cell_nmax, &inv_cell) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image([rx0, ry0, rz0], cell, iv, n),
            _ => [rx0, ry0, rz0],
        }
    };
    // Per-electron nearest-nucleus set (min-image aware), matching the value path.
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
    let nearest_up: Vec<Vec<usize>> = (0..r_up.nrows())
        .map(|i| nearest_set(r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]))
        .collect();
    let nearest_dn: Vec<Vec<usize>> = (0..r_dn.nrows())
        .map(|i| nearest_set(r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]))
        .collect();

    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let (cx, cy, cz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        let l = ang_mom[t];
        let (e, c, p) = (exponents[t], coefficients[t], powers[t]);

        for &is_up in &[true, false] {
            let r_e = if is_up { r_up.view() } else { r_dn.view() };
            let orb = if is_up {
                &geminal.orb_up
            } else {
                &geminal.orb_dn
            };
            let nearest = if is_up { &nearest_up } else { &nearest_dn };
            let n_e = r_e.nrows();
            for i in 0..n_e {
                if !nearest[i].contains(&ia) {
                    continue;
                }
                let rel = rel_vec(cx - r_e[[i, 0]], cy - r_e[[i, 1]], cz - r_e[[i, 2]]);
                let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
                let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
                let v_l = c * nrm.powf(p - 2.0) * (-e * nrm * nrm).exp();
                // All nv mesh points (raw) + cos angles built once so the orbital
                // eval / det-ratio contraction batch over them (value-path layout).
                let mut mesh_pts = Array2::<f64>::zeros((nv, 3));
                let mut cosv = vec![0.0_f64; nv];
                let mut poss = vec![[0.0_f64; 3]; nv];
                for v in 0..nv {
                    let (dx, dy, dz) = (
                        grid_points[[v, 0]],
                        grid_points[[v, 1]],
                        grid_points[[v, 2]],
                    );
                    let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                    cosv[v] = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                    let pos = [
                        r_e[[i, 0]] + rx + nrm * dx,
                        r_e[[i, 1]] + ry + nrm * dy,
                        r_e[[i, 2]] + rz + nrm * dz,
                    ];
                    poss[v] = pos;
                    mesh_pts[[v, 0]] = pos[0];
                    mesh_pts[[v, 1]] = pos[1];
                    mesh_pts[[v, 2]] = pos[2];
                }
                // Home-cell-wrap the mesh points for the AO evals ONLY.
                let mesh_pts_w = crate::atomic_orbital::wrap_r(&orb.aos, &mesh_pts.view());
                let orb_mesh = orb.compute(&mesh_pts_w.view());
                // Batched rank-1 factors: up move => row i changed, ratio =
                // 1 + v*G^{-1}[:,i]; dn move => col i changed, ratio = 1 + G^{-1}[i,:]*u.
                let vmesh = if is_up {
                    crate::determinant::compute_v_up_move_batch(
                        geminal,
                        &det_state,
                        i,
                        &orb_mesh.view(),
                    )
                } else {
                    crate::determinant::compute_u_dn_move_batch(
                        geminal,
                        &det_state,
                        i,
                        &orb_mesh.view(),
                    )
                };
                let j3_mesh = j3.map(|(orb3, _)| orb3.compute(&mesh_pts_w.view()));
                let dtn_mesh: Option<Vec<f64>> = jastrow.dtn.as_ref().map(|dtn| {
                    let mut o = vec![0.0_f64; nv];
                    crate::dtn::compute_dtn_ratio_mesh(
                        dtn,
                        r_up,
                        r_dn,
                        is_up,
                        i,
                        &mesh_pts.view(),
                        &mut o,
                    );
                    o
                });
                for v in 0..nv {
                    let mut dr = 1.0_f64;
                    if is_up {
                        for a in 0..vmesh.nrows() {
                            dr += vmesh[[a, v]] * ginv[[a, i]];
                        }
                    } else {
                        for a in 0..vmesh.nrows() {
                            dr += ginv[[i, a]] * vmesh[[a, v]];
                        }
                    }
                    let mut jfac = crate::jastrow_factor::jastrow_1b2b_ratio(
                        jastrow, r_up, r_dn, is_up, i, poss[v],
                    );
                    if let (Some((jm, st3)), Some(jm_mesh)) = (&j3_ratio_state, &j3_mesh) {
                        let col: Vec<f64> = (0..jm_mesh.nrows()).map(|a| jm_mesh[[a, v]]).collect();
                        jfac *= crate::jastrow_factor::j3_ratio(jm, st3, is_up, i, &col);
                    }
                    if let Some(dr_dtn) = &dtn_mesh {
                        jfac *= dr_dtn[v];
                    }
                    let vg =
                        v_l * ((2 * l + 1) as f64) * legendre(l, cosv[v]) * weights[v] * jfac * dr;
                    sum_v += vg;
                    mesh.push(MeshPoint {
                        v_grid: vg,
                        is_up,
                        idx: i,
                        pos: poss[v],
                    });
                }
            }
        }
    }
    (mesh, sum_v)
}

#[inline]
fn with_moved(base: &ArrayView2<f64>, idx: usize, pos: [f64; 3]) -> Array2<f64> {
    let mut m = base.to_owned();
    m[[idx, 0]] = pos[0];
    m[[idx, 1]] = pos[1];
    m[[idx, 2]] = pos[2];
    m
}

/// Rank-1 outer product `a b^T` -> (a.len(), b.len()). Used once per moved
/// electron to fold the Sherman-Morrison reduced vectors into the accumulator.
#[inline]
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let (na, nb) = (a.len(), b.len());
    let mut o = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        let ai = a[i];
        if ai == 0.0 {
            continue;
        }
        for j in 0..nb {
            o[[i, j]] = ai * b[j];
        }
    }
    o
}

/// Near-singular fallback threshold on the single-move det ratio `d = det_new/det_base`.
/// The Sherman-Morrison-incremental O_k divides the rank-1 correction by `d`; below
/// `sqrt(eps)` (the classical half-mantissa rank-1-update conditioning bound) the moved
/// geminal is within fp resolution of singular, so we defer THAT candidate to the fresh
/// per-point rebuild (the pre-optimization path). Derived from `f64::EPSILON`, not a
/// tuned/measured constant. In practice ECP mesh moves land near the ECP shell (well away
/// from geminal nodes) so `d = O(1)` and the guard never fires on physical fixtures.
#[inline]
fn sm_singular_guard() -> f64 {
    f64::EPSILON.sqrt()
}

/// Upper fallback threshold on `|d|`, from the base geminal's conditioning (both the
/// dlambda and the dC SM reduces). The SM increment is built from the SINGLE base
/// inverse, whose fp64 error is ~`eps * kappa(G_base)` relative; the rank-1 fold
/// scales it by the per-electron base-inverse column (size ~|d| when the move lifts a
/// near-singular base, e.g. a near-coincident same-spin pair it breaks up), so the
/// increment's forward error is ~`eps * kappa * |d|`. The fresh per-point rebuild
/// carries only the problem's own `eps * kappa` floor. Measured on near-coincident
/// SD configurations (dC reduce, fp64): kappa=6.3e6 with one |d|=3.7e3 candidate gave
/// a 7.2e-6 relative SM-vs-rebuild gap (model 5.1e-6); 10 of 12 such configurations
/// were 3e-5..4e-4 off, while 1-ulp changes of the MO coefficients moved the rebuild
/// by < 1e-8 and the SM result by up to 5e-4. Holding the error to the same
/// half-mantissa budget as the lower guard (`eps * kappa * |d| <= sqrt(eps)`) gives
/// `|d| <= 1 / (sqrt(eps) * kappa_1)`, with the O(N^2) 1-norm estimate
/// `kappa_1 = ||G||_1 ||G^{-1}||_1 >= 1` from the matrices already formed. Above it
/// THAT candidate takes the fresh per-point rebuild; a non-finite estimate (exactly
/// singular base) sends every candidate there, as the non-finite `d` already did.
/// Derived from `f64::EPSILON`, not tuned. A well-conditioned base with ECP moves
/// (`|d| = O(1)`) stays far below it, so the hot path is unchanged.
fn sm_large_ratio_guard(g: &Array2<f64>, ginv: &Array2<f64>) -> f64 {
    let norm1 = |m: &Array2<f64>| {
        m.columns()
            .into_iter()
            .map(|c| c.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    };
    let kappa = norm1(g) * norm1(ginv);
    if kappa.is_finite() && kappa > 0.0 {
        1.0 / (f64::EPSILON.sqrt() * kappa)
    } else {
        0.0
    }
}

/// True when mesh candidate `d` may use the SM increment; otherwise it takes the
/// fresh per-point rebuild. `lower` is [`sm_effective_guard`] (`+inf` under the
/// test-only force-rebuild hook), `upper` is [`sm_large_ratio_guard`] of the base.
#[inline]
fn sm_ratio_ok(dj: f64, lower: f64, upper: f64) -> bool {
    dj.is_finite() && dj.abs() >= lower && dj.abs() <= upper
}

/// Effective guard, honoring the test-only `JQMC_ECP_OK_FORCE_REBUILD` hook: when
/// set, the guard is `+inf` so EVERY mesh candidate takes the fresh per-point rebuild
/// branch (the exact pre-optimization O_k reduce). Lets an A/B test compare the SM-
/// incremental fast path against the per-point reference on identical inputs -- incl.
/// a near-singular probe -- with no numeric drift beyond fp association. Read once per
/// param-grad call (never per mesh point), so it is off the hot path.
#[inline]
fn sm_effective_guard() -> f64 {
    if std::env::var_os("JQMC_ECP_OK_FORCE_REBUILD").is_some() {
        f64::INFINITY
    } else {
        sm_singular_guard()
    }
}

/// `dV_ecp(non-local)/dlambda`, shape == `lambda`.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_lambda(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> Array2<f64> {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    // Sherman-Morrison incremental O_k(r') mesh reduction.
    // Each mesh point moves ONE electron, so G shifts by a rank-1 update and
    //   O_k(r')  =  O_k_base  +  rank1(mp)
    // where rank1(mp) comes from the SM-updated inverse OFF THE SINGLE BASE `G^{-1}`
    // (reused per candidate -- never a fresh per-point LU, so it inherits the
    // near-singular stability of `collect_mesh`'s det ratio; the known
    // fresh-LU NaN mode is avoided). The value-path identity is
    //   dV/dlambda = Sum_mp v_grid_mp * O_k(r'_mp) - O_k_base * Sum_v,
    // and since Sum_mp v_grid_mp == Sum_v the O_k_base term cancels exactly, leaving
    //   dV/dlambda = Sum_mp v_grid_mp * rank1(mp).
    // Algebra (up-move electron i; A=o_up^T, B=o_dn, D=[lam_p*B|lam_u], P=G^{-1}A):
    //   G_new = G + e_i (D^T da)^T,  da = orb_mesh_i - o_up[:,i],  v = D^T da,
    //   d = 1 + v.G^{-1}[:,i],  z = P^T v,  c = (da - z)/d,
    //   h = [B * G^{-1}[:n_dn,i] | G^{-1}[n_dn:,i]]  (per-electron),
    //   rank1 = outer(c, h)  ==>  contribution = outer(Sum_mp v_grid c_mp, h).
    // Dn-move (col i of B changes; E=A*lam_p, w=E*db, gc2=G^{-1}w):
    //   d = 1 + gc2[i],  h2 = [db - B*gc2[:n_dn] | -gc2[n_dn:]],  c2 = P[i,:] (fixed),
    //   contribution = outer(P[i,:], Sum_mp v_grid/d * h2_mp).
    // Per-candidate cost = a few (M x k) GEMMs; NO per-point O(N^3) inverse.
    let _ = sum_v;
    // Home-cell wrap for the AO evals, 1:1 `collect_mesh` (which wraps its own base
    // + mesh AO evals) and `compute_grads_and_laplacian_ln_det`. jqmc wraps every
    // electron coordinate before an AO eval; the truncated image sum is enumerated
    // relative to the home cell, so a RAW coordinate outside it falls beyond the
    // static tiling and the AO collapses. Identity for molecular / pure PW.
    let r_up_w = crate::atomic_orbital::wrap_r(&geminal.orb_up.aos, r_up);
    let r_dn_w = crate::atomic_orbital::wrap_r(&geminal.orb_dn.aos, r_dn);
    // EXACT (un-screened) AO eval: this output is indexed BY lambda/AO, so a
    // LEPSBAS-screened row would land as an exact 0.0 where XLA has a small non-zero.
    let o_up0 = geminal.orb_up.compute_exact(&r_up_w.view()); // (M_up, N_up)
    let o_dn0 = geminal.orb_dn.compute_exact(&r_dn_w.view()); // (M_dn, N_dn)
    let ok_base = grad_ln_det_lambda_from_orb(geminal, &o_up0.view(), &o_dn0.view());
    let mut out = Array2::<f64>::zeros(ok_base.dim());

    let m_dn = geminal.orb_num_dn;
    let n_dn = o_dn0.ncols();
    let n_up = o_up0.ncols();
    let m_up = o_up0.nrows();
    let lambda = geminal
        .lambda
        .as_ref()
        .expect("grad_ecp_lambda is undefined for occupied-only SD");
    let ncols = lambda.ncols();
    let n_unp = ncols - m_dn;
    let lam = lambda.view();
    let lam_p = lam.slice(numpy::ndarray::s![.., 0..m_dn]).to_owned(); // (M_up, M_dn)
    let lam_u = lam.slice(numpy::ndarray::s![.., m_dn..]).to_owned(); // (M_up, n_unp)
    let a = o_up0.t().to_owned(); // A (N_up, M_up)
    let b = o_dn0.clone(); // B (M_dn, N_dn)
                           // D = [lam_p*B | lam_u]  (M_up, N_up)
    let mut d_mat = Array2::<f64>::zeros((m_up, n_up));
    d_mat
        .slice_mut(numpy::ndarray::s![.., 0..n_dn])
        .assign(&lam_p.dot(&b));
    d_mat
        .slice_mut(numpy::ndarray::s![.., n_dn..])
        .assign(&lam_u);
    let g = a.dot(&d_mat); // G (N_up, N_up)
    let ginv = inverse(&g.view());
    let p_base = ginv.dot(&a); // P = G^{-1} A (N_up, M_up)
    let e_mat = a.dot(&lam_p); // E = A lam_p (N_up, M_dn)
    let guard = sm_effective_guard();
    let upper = sm_large_ratio_guard(&g, &ginv);

    let n_up_e = r_up.nrows();
    let n_dn_e = r_dn.nrows();
    let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_up_e + n_dn_e];
    let mut pts: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_up_e + n_dn_e];
    for mp in &mesh {
        let e = if mp.is_up { mp.idx } else { n_up_e + mp.idx };
        buckets[e].push((pts[e].len(), mp.v_grid));
        pts[e].push(mp.pos);
    }
    for e in 0..(n_up_e + n_dn_e) {
        if pts[e].is_empty() {
            continue;
        }
        let is_up = e < n_up_e;
        let idx = if is_up { e } else { e - n_up_e };
        let orb = if is_up {
            &geminal.orb_up
        } else {
            &geminal.orb_dn
        };
        let k = pts[e].len();
        let mut ptmat = Array2::<f64>::zeros((k, 3));
        for (j, p) in pts[e].iter().enumerate() {
            ptmat[[j, 0]] = p[0];
            ptmat[[j, 1]] = p[1];
            ptmat[[j, 2]] = p[2];
        }
        // `MeshPoint::pos` is the RAW mesh point (the min-image Jastrow/DTN ratios and
        // the Legendre angle need it raw); wrap it for the AO eval only, exactly as
        // `collect_mesh` does with `mesh_pts_w`.
        let ptmat_w = crate::atomic_orbital::wrap_r(&orb.aos, &ptmat.view());
        let orb_mesh = orb.compute_exact(&ptmat_w.view()); // (M_x, k)
        let mut vgc = vec![0.0_f64; k];
        for &(j, vg) in &buckets[e] {
            vgc[j] = vg;
        }
        if is_up {
            // delta = orb_mesh - o_up0[:,idx]  (M_up, k)
            let mut delta = orb_mesh.clone();
            for al in 0..m_up {
                let old = o_up0[[al, idx]];
                for x in 0..k {
                    delta[[al, x]] -= old;
                }
            }
            let v = d_mat.t().dot(&delta); // (N_up, k)
            let z = p_base.t().dot(&v); // (M_up, k)
            let g_col = ginv.column(idx).to_owned(); // (N_up)
                                                     // h = [B*g_col[:n_dn] | g_col[n_dn:]]  (ncols), per-electron.
            let mut h = Array1::<f64>::zeros(ncols);
            let bp = b.dot(&g_col.slice(numpy::ndarray::s![0..n_dn])); // (M_dn)
            for t in 0..m_dn {
                h[t] = bp[t];
            }
            for u in 0..n_unp {
                h[m_dn + u] = g_col[n_dn + u];
            }
            let mut c_e = Array1::<f64>::zeros(m_up);
            let mut scratch: Option<Array2<f64>> = None;
            let (mut n_sm, mut n_rb) = (0_u64, 0_u64);
            for x in 0..k {
                let mut dj = 1.0_f64;
                for al in 0..n_up {
                    dj += g_col[al] * v[[al, x]];
                }
                if sm_ratio_ok(dj, guard, upper) {
                    n_sm += 1;
                    let inv = vgc[x] / dj;
                    for al in 0..m_up {
                        c_e[al] += inv * (delta[[al, x]] - z[[al, x]]);
                    }
                } else {
                    n_rb += 1;
                    let sc = scratch.get_or_insert_with(|| o_up0.clone());
                    sc.column_mut(idx).assign(&orb_mesh.column(x));
                    let full = grad_ln_det_lambda_from_orb(geminal, &sc.view(), &o_dn0.view());
                    out.scaled_add(vgc[x], &(&full - &ok_base));
                    sc.column_mut(idx).assign(&o_up0.column(idx));
                }
            }
            crate::gfmc::record_ecp_param_sm_ratio(n_sm, n_rb);
            out += &outer(&c_e, &h);
        } else {
            // delta = orb_mesh - o_dn0[:,idx]  (M_dn, k)
            let mut delta = orb_mesh.clone();
            for bd in 0..m_dn {
                let old = o_dn0[[bd, idx]];
                for x in 0..k {
                    delta[[bd, x]] -= old;
                }
            }
            let w = e_mat.dot(&delta); // (N_up, k)
            let g_col2 = ginv.dot(&w); // (N_up, k)
            let y2 = b.dot(&g_col2.slice(numpy::ndarray::s![0..n_dn, ..])); // (M_dn, k)
            let c2 = p_base.row(idx).to_owned(); // P[idx,:] (M_up), per-electron
            let mut h_e = Array1::<f64>::zeros(ncols);
            let mut scratch: Option<Array2<f64>> = None;
            let (mut n_sm, mut n_rb) = (0_u64, 0_u64);
            for x in 0..k {
                let dj = 1.0 + g_col2[[idx, x]];
                if sm_ratio_ok(dj, guard, upper) {
                    n_sm += 1;
                    let inv = vgc[x] / dj;
                    for t in 0..m_dn {
                        h_e[t] += inv * (delta[[t, x]] - y2[[t, x]]);
                    }
                    for u in 0..n_unp {
                        h_e[m_dn + u] += inv * (-g_col2[[n_dn + u, x]]);
                    }
                } else {
                    n_rb += 1;
                    let sc = scratch.get_or_insert_with(|| o_dn0.clone());
                    sc.column_mut(idx).assign(&orb_mesh.column(x));
                    let full = grad_ln_det_lambda_from_orb(geminal, &o_up0.view(), &sc.view());
                    out.scaled_add(vgc[x], &(&full - &ok_base));
                    sc.column_mut(idx).assign(&o_dn0.column(idx));
                }
            }
            crate::gfmc::record_ecp_param_sm_ratio(n_sm, n_rb);
            out += &outer(&c2, &h_e);
        }
    }
    out
}

/// AGPn `dV_ecp(non-local)/d(lambda_diag, lambda_unpaired)` (real path).
///
/// `lambda_matrix = hstack([diag(lambda_diag), lambda_unpaired])` is LINEAR in the
/// natural-orbital leaves, and so is the whole weighted mesh reduce, so the leaf
/// gradient is just the chain rule applied to the dense-`lambda` result: the
/// paired-block DIAGONAL (the off-diagonal paired entries are structural zeros) plus
/// the unpaired columns. Same mapping as `param_grad::grad_ln_det_lambda_diag` /
/// `param_grad_el::grad_t_lambda_diag`.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_lambda_diag(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> (Array1<f64>, Array2<f64>) {
    let dense = grad_ecp_lambda(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
    );
    let m_dn = geminal.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = Array1::from_iter((0..n).map(|k| dense[[k, k]]));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    (diag, unpaired)
}

/// `dV_ecp(non-local)/dC` (up, dn mo_coefficients).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_mo(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> (Array2<f64>, Array2<f64>) {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    // ---- Sherman-Morrison-incremental dV_ecp/dC (see `grad_ecp_lambda` for the
    // shared derivation + cancellation). Here O_k = dln|det|/dC with orb = C*AO, so
    // a one-electron move changes ONE AO column -> ONE orb column -> rank-1 G shift.
    // Base fields mirror `grad_ln_det_mo_from_ao` (raw AO). W=[lam_p*orb_dn|lam_u],
    // Q=orb_up^T lam_p, GOU=W G^{-1}, GQ=G^{-1} Q. Up-move electron i:
    //   dao = ao_mesh_i - a_up[:,i],  do = C_up dao,  v = W^T do,  gc = G^{-1}[:,i],
    //   d = 1 + gc.v,  s = a_up (G^{-T} v),  dq = lam_p^T do,  t = GQ^T v,
    //   dC_up += outer(GOU[:,i], Sum v_grid (dao - s)/d),
    //   dC_dn += outer(Sum v_grid (dq - t)/d, a_dn G^{-1}[:n_dn,i]).
    // Dn-move electron i (do = C_dn dao, w = Q do, gc2 = G^{-1} w):
    //   d = 1 + gc2[i],  Wg = W gc2,  dw = lam_p do,  y2 = a_dn gc2[:n_dn],
    //   dC_up += outer(Sum v_grid (dw - Wg)/d, a_up G^{-1}[i,:]),
    //   dC_dn += outer(GQ[i,:], Sum v_grid (dao - y2)/d).
    // No per-point O(N^3) inverse; candidates with |d| < sqrt(eps) or
    // |d| > 1/(sqrt(eps) kappa_1(G)) defer to the fresh rebuild (`sm_ratio_ok`).
    let _ = sum_v;
    // Home-cell wrap for the AO evals -- see `grad_ecp_lambda` above (same contract:
    // raw coords fall outside the static image tiling and collapse the AO).
    let r_up_w = crate::atomic_orbital::wrap_r(&geminal.orb_up.aos, r_up);
    let r_dn_w = crate::atomic_orbital::wrap_r(&geminal.orb_dn.aos, r_dn);
    // EXACT (un-screened) AO eval -- see `grad_ecp_lambda` (output indexed BY AO).
    let a_up0 = geminal.orb_up.eval_aos_exact(&r_up_w.view()); // (num_ao_up, N_up)
    let a_dn0 = geminal.orb_dn.eval_aos_exact(&r_dn_w.view()); // (num_ao_dn, N_dn)
    let (bu, bd) = grad_ln_det_mo_from_ao(geminal, &a_up0.view(), &a_dn0.view());
    let mut au = Array2::<f64>::zeros(bu.dim());
    let mut ad = Array2::<f64>::zeros(bd.dim());

    let c_up = geminal
        .orb_up
        .mo_coefficients
        .as_ref()
        .expect("grad_ecp_mo: up spin has no mo_coefficients");
    let c_dn = geminal
        .orb_dn
        .mo_coefficients
        .as_ref()
        .expect("grad_ecp_mo: dn spin has no mo_coefficients");
    let orb_up0 = c_up.dot(&a_up0); // (m_up, N_up)
    let orb_dn0 = c_dn.dot(&a_dn0); // (m_dn, N_dn)
    let m_up = orb_up0.nrows();
    let m_dn = geminal.orb_num_dn;
    let n_dn = a_dn0.ncols();
    let n_up = a_up0.ncols();
    let num_ao_up = a_up0.nrows();
    let num_ao_dn = a_dn0.nrows();
    // SD has no stored selector: build the geminal products directly from the
    // occupied MO rows.  Dense AGP/AGPn continues through the same helpers.
    let w = geminal.w_from_dn(&orb_dn0.view(), n_up);
    let g = orb_up0.t().dot(&w);
    let ginv = inverse(&g.view());
    let q = geminal.up_t_times_paired(&orb_up0.view()); // (N_up, m_dn)
    let gou = w.dot(&ginv); // (m_up, N_up)
    let gq = ginv.dot(&q); // (N_up, m_dn)
    let guard = sm_effective_guard();
    let upper = sm_large_ratio_guard(&g, &ginv);

    let n_up_e = r_up.nrows();
    let n_dn_e = r_dn.nrows();
    let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_up_e + n_dn_e];
    let mut pts: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_up_e + n_dn_e];
    for mp in &mesh {
        let e = if mp.is_up { mp.idx } else { n_up_e + mp.idx };
        buckets[e].push((pts[e].len(), mp.v_grid));
        pts[e].push(mp.pos);
    }
    for e in 0..(n_up_e + n_dn_e) {
        if pts[e].is_empty() {
            continue;
        }
        let is_up = e < n_up_e;
        let idx = if is_up { e } else { e - n_up_e };
        let orb = if is_up {
            &geminal.orb_up
        } else {
            &geminal.orb_dn
        };
        let k = pts[e].len();
        let mut ptmat = Array2::<f64>::zeros((k, 3));
        for (j, p) in pts[e].iter().enumerate() {
            ptmat[[j, 0]] = p[0];
            ptmat[[j, 1]] = p[1];
            ptmat[[j, 2]] = p[2];
        }
        // RAW `MeshPoint::pos` wrapped for the AO eval only -- see `grad_ecp_lambda`.
        let ptmat_w = crate::atomic_orbital::wrap_r(&orb.aos, &ptmat.view());
        let ao_mesh = orb.eval_aos_exact(&ptmat_w.view()); // (num_ao, k)
        let mut vgc = vec![0.0_f64; k];
        for &(j, vg) in &buckets[e] {
            vgc[j] = vg;
        }
        if is_up {
            // dao = ao_mesh - a_up0[:,idx]  (num_ao_up, k)
            let mut dao = ao_mesh.clone();
            for r in 0..num_ao_up {
                let old = a_up0[[r, idx]];
                for x in 0..k {
                    dao[[r, x]] -= old;
                }
            }
            let d_o = c_up.dot(&dao); // (m_up, k)
            let v = w.t().dot(&d_o); // (N_up, k)
            let g_row = ginv.t().dot(&v); // (N_up, k)
            let s = a_up0.dot(&g_row); // (num_ao_up, k)
            let dq = geminal.paired_t_times(&d_o.view()); // (m_dn, k)
            let t = gq.t().dot(&v); // (m_dn, k)
            let g_col = ginv.column(idx).to_owned(); // (N_up)
            let gou_i = gou.column(idx).to_owned(); // (m_up), per-electron
            let mut r_vec = Array1::<f64>::zeros(num_ao_dn); // a_dn0 @ g_col[:n_dn]
            for jj in 0..num_ao_dn {
                let mut su = 0.0;
                for nn in 0..n_dn {
                    su += a_dn0[[jj, nn]] * g_col[nn];
                }
                r_vec[jj] = su;
            }
            let mut cu = Array1::<f64>::zeros(num_ao_up);
            let mut cd = Array1::<f64>::zeros(m_dn);
            let mut sc_up: Option<Array2<f64>> = None;
            let (mut n_sm, mut n_rb) = (0_u64, 0_u64);
            for x in 0..k {
                let mut dj = 1.0_f64;
                for al in 0..n_up {
                    dj += g_col[al] * v[[al, x]];
                }
                if sm_ratio_ok(dj, guard, upper) {
                    n_sm += 1;
                    let inv = vgc[x] / dj;
                    for r in 0..num_ao_up {
                        cu[r] += inv * (dao[[r, x]] - s[[r, x]]);
                    }
                    for bcol in 0..m_dn {
                        cd[bcol] += inv * (dq[[bcol, x]] - t[[bcol, x]]);
                    }
                } else {
                    n_rb += 1;
                    let sc = sc_up.get_or_insert_with(|| a_up0.clone());
                    sc.column_mut(idx).assign(&ao_mesh.column(x));
                    let (gu, gd) = grad_ln_det_mo_from_ao(geminal, &sc.view(), &a_dn0.view());
                    au.scaled_add(vgc[x], &(&gu - &bu));
                    ad.scaled_add(vgc[x], &(&gd - &bd));
                    sc.column_mut(idx).assign(&a_up0.column(idx));
                }
            }
            crate::gfmc::record_ecp_param_sm_ratio(n_sm, n_rb);
            au += &outer(&gou_i, &cu);
            ad += &outer(&cd, &r_vec);
        } else {
            // dao = ao_mesh - a_dn0[:,idx]  (num_ao_dn, k)
            let mut dao = ao_mesh.clone();
            for r in 0..num_ao_dn {
                let old = a_dn0[[r, idx]];
                for x in 0..k {
                    dao[[r, x]] -= old;
                }
            }
            let d_o = c_dn.dot(&dao); // (m_dn, k)
            let ww = q.dot(&d_o); // (N_up, k)
            let g_col2 = ginv.dot(&ww); // (N_up, k)
            let wg = w.dot(&g_col2); // (m_up, k)
            let dw = geminal.paired_times(&d_o.view()); // (m_up, k)
            let y2 = a_dn0.dot(&g_col2.slice(numpy::ndarray::s![0..n_dn, ..])); // (num_ao_dn, k)
            let g_row2 = ginv.row(idx).to_owned(); // (N_up)
            let tq = gq.row(idx).to_owned(); // (m_dn), per-electron
            let mut ru2 = Array1::<f64>::zeros(num_ao_up); // a_up0 @ g_row2
            for jj in 0..num_ao_up {
                let mut su = 0.0;
                for nn in 0..n_up {
                    su += a_up0[[jj, nn]] * g_row2[nn];
                }
                ru2[jj] = su;
            }
            let mut cu2 = Array1::<f64>::zeros(m_up);
            let mut cd2 = Array1::<f64>::zeros(num_ao_dn);
            let mut sc_dn: Option<Array2<f64>> = None;
            let (mut n_sm, mut n_rb) = (0_u64, 0_u64);
            for x in 0..k {
                let dj = 1.0 + g_col2[[idx, x]];
                if sm_ratio_ok(dj, guard, upper) {
                    n_sm += 1;
                    let inv = vgc[x] / dj;
                    for al in 0..m_up {
                        cu2[al] += inv * (dw[[al, x]] - wg[[al, x]]);
                    }
                    for r in 0..num_ao_dn {
                        cd2[r] += inv * (dao[[r, x]] - y2[[r, x]]);
                    }
                } else {
                    n_rb += 1;
                    let sc = sc_dn.get_or_insert_with(|| a_dn0.clone());
                    sc.column_mut(idx).assign(&ao_mesh.column(x));
                    let (gu, gd) = grad_ln_det_mo_from_ao(geminal, &a_up0.view(), &sc.view());
                    au.scaled_add(vgc[x], &(&gu - &bu));
                    ad.scaled_add(vgc[x], &(&gd - &bd));
                    sc.column_mut(idx).assign(&a_dn0.column(idx));
                }
            }
            crate::gfmc::record_ecp_param_sm_ratio(n_sm, n_rb);
            au += &outer(&cu2, &ru2);
            ad += &outer(&tq, &cd2);
        }
    }
    // A square occupied-SD coefficient matrix is an invertible, configuration-
    // independent basis change.  The determinant factor cancels from every
    // non-local ECP ratio, so this derivative is exactly zero.
    if geminal.is_sd() {
        if au.nrows() == au.ncols() {
            au.fill(0.0);
        }
        if ad.nrows() == ad.ncols() {
            ad.fill(0.0);
        }
    }
    (au, ad)
}

// ============================================================================
// dV_ecp(non-local)/dc for the JASTROW / DTN / BASIS param blocks (same O_k(r')
// mesh identity as the geminal case, but with the Jastrow/DTN/basis O_k kernel).
// For a Jastrow/DTN param the wavefunction factor is exp(J), so
// O_k = dlnPsi/dc = dJ/dc (the VALUE gradient); for a basis (radial) param the
// determinant carries it, so O_k = dln|det|/d(radial). All autograd-free, reusing
// `collect_mesh` plus the existing value-gradient kernels evaluated at
// each moved-electron mesh configuration. Molecular (Euclidean mesh) scope.
// ============================================================================

/// Scalar `dV_ecp(non-local)/dp` for a single-parameter Jastrow O_k `ok(r_up, r_dn)`.
#[allow(clippy::too_many_arguments)]
fn mesh_reduce_scalar<F>(
    mesh: &[MeshPoint],
    sum_v: f64,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ok: F,
) -> f64
where
    F: Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> f64,
{
    let ok_base = ok(r_up, r_dn);
    let mut acc = 0.0_f64;
    for mp in mesh {
        let o = if mp.is_up {
            let m = with_moved(r_up, mp.idx, mp.pos);
            ok(&m.view(), r_dn)
        } else {
            let m = with_moved(r_dn, mp.idx, mp.pos);
            ok(r_up, &m.view())
        };
        acc += o * mp.v_grid;
    }
    acc - ok_base * sum_v
}

/// Array `dV_ecp(non-local)/dc` for a matrix-valued Jastrow O_k `ok(r_up, r_dn)`.
fn mesh_reduce_array<F>(
    mesh: &[MeshPoint],
    sum_v: f64,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    ok: F,
) -> Array2<f64>
where
    F: Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64>,
{
    let ok_base = ok(r_up, r_dn);
    let mut acc = Array2::<f64>::zeros(ok_base.dim());
    for mp in mesh {
        let o = if mp.is_up {
            let m = with_moved(r_up, mp.idx, mp.pos);
            ok(&m.view(), r_dn)
        } else {
            let m = with_moved(r_dn, mp.idx, mp.pos);
            ok(r_up, &m.view())
        };
        acc += &(o * mp.v_grid);
    }
    acc - &(ok_base * sum_v)
}

/// `dV_ecp(non-local)/dJ1_param` (scalar).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j1(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    j1: &JastrowOneBodyData,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> f64 {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    if crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
        return mesh_reduce_scalar(&mesh, sum_v, r_up, r_dn, |ru, rd| grad_j1_param(j1, ru, rd));
    }
    crate::gfmc::record_ecp_param_ok(None);
    mesh.iter()
        .map(|mp| mp.v_grid * j1_param_move_delta(j1, r_up, r_dn, mp.is_up, mp.idx, mp.pos))
        .sum()
}

/// `dV_ecp(non-local)/dJ2_param` (scalar).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j2(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    j2: &JastrowTwoBodyData,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> f64 {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    if crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
        return mesh_reduce_scalar(&mesh, sum_v, r_up, r_dn, |ru, rd| grad_j2_param(j2, ru, rd));
    }
    crate::gfmc::record_ecp_param_ok(None);
    mesh_reduce_j2_incremental(
        j2,
        r_up,
        r_dn,
        mesh.iter().map(|mp| (mp.is_up, mp.idx, mp.pos, mp.v_grid)),
    )
}

/// Exact incremental J2 mesh reduce `Sum_g v_g [O(r'_g) - O(r)]`, generic over the
/// candidate weight type (real path `f64`, twist `Complex64`). The mesh is grouped by
/// moved electron, so the `(N - 1)` partner table is rebuilt only when it changes.
pub(crate) fn mesh_reduce_j2_incremental<T, I>(
    j2: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    mesh: I,
) -> T
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>,
    I: Iterator<Item = (bool, usize, [f64; 3], T)>,
{
    let mut acc = T::default();
    // (moved electron, its partner table, its base-position row)
    let mut cached: Option<((bool, usize), Array2<f64>, f64)> = None;
    for (is_up, i, pos, vg) in mesh {
        let fresh = !matches!(&cached, Some((key, _, _)) if *key == (is_up, i));
        if fresh {
            let old = if is_up {
                [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
            } else {
                [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
            };
            let partners = j2_partners(r_up, r_dn, is_up, i);
            let base_row = j2_param_row(j2, &partners.view(), old);
            cached = Some(((is_up, i), partners, base_row));
        }
        let (_, partners, base_row) = cached.as_ref().expect("partner cache");
        acc = acc + vg * (j2_param_row(j2, &partners.view(), pos) - *base_row);
    }
    acc
}

/// `dV_ecp(non-local)/dj_matrix` (J3), shape `(orb, orb+1)`.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j3(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3_orb: &OrbData,
    j3_jm: &ArrayView2<f64>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> Array2<f64> {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        Some((j3_orb, j3_jm)),
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
        cell_nmax,
        NN_DEFAULT,
    );
    if crate::gfmc::ecp_param_force_reference() {
        crate::gfmc::record_ecp_param_ok(Some(mesh.len() + 1));
        return mesh_reduce_array(&mesh, sum_v, r_up, r_dn, |ru, rd| {
            grad_j3_jmatrix(j3_orb, ru, rd)
        });
    }
    crate::gfmc::record_ecp_param_ok(None);
    // Same home-cell wrap + orbital evaluation as `grad_j3_jmatrix` (base electrons and
    // every candidate column, evaluated in ONE batched call), then the exact linear
    // one-column increment.
    let orb_vals = |r: &ArrayView2<f64>| -> Array2<f64> {
        let w = j3_orb.wrap_electrons(r).unwrap_or_else(|| r.to_owned());
        j3_orb.compute(&w.view())
    };
    let a_up = orb_vals(r_up);
    let a_dn = orb_vals(r_dn);
    let pts = mesh_positions(mesh.iter().map(|mp| mp.pos));
    let a_cand = orb_vals(&pts.view());
    let moves: Vec<(bool, usize, f64)> = mesh
        .iter()
        .map(|mp| (mp.is_up, mp.idx, mp.v_grid))
        .collect();
    let (d_up, d_dn) = j3_mesh_column_changes(&a_up, &a_dn, &a_cand, &moves);
    j3_jmatrix_mesh_increment(&a_up, &a_dn, &d_up, &d_dn)
}

/// `dV_ecp(non-local)/d{alpha,beta,gamma}_indep` (DTN u/chi/f), each `None` if the
/// channel is disabled. O_k = the DTN VALUE param-gradient `compute_dtn_param_grads`.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_dtn(
    d: &DtnOkData,
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> crate::param_grad_dtn::DtnGrads {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    // O_k at each mesh point moves one electron: swap its terms instead of recomputing.
    let mut ok = DtnMovedOk::new(d, r_up, r_dn);
    let (a0, b0, g0) = ok.base_grads();
    let mut a = a0.as_ref().map(|x| Array2::<f64>::zeros(x.dim()));
    let mut b = b0.as_ref().map(|x| Array2::<f64>::zeros(x.dim()));
    let mut g = g0.as_ref().map(|x| Array2::<f64>::zeros(x.dim()));
    for mp in &mesh {
        let (am, bm, gm) = ok.moved_grads(mp.is_up, mp.idx, mp.pos);
        if let (Some(acc), Some(v)) = (a.as_mut(), am.as_ref()) {
            *acc = &*acc + &(v * mp.v_grid);
        }
        if let (Some(acc), Some(v)) = (b.as_mut(), bm.as_ref()) {
            *acc = &*acc + &(v * mp.v_grid);
        }
        if let (Some(acc), Some(v)) = (g.as_mut(), gm.as_ref()) {
            *acc = &*acc + &(v * mp.v_grid);
        }
    }
    if let (Some(acc), Some(v)) = (a.as_mut(), a0.as_ref()) {
        *acc = &*acc - &(v * sum_v);
    }
    if let (Some(acc), Some(v)) = (b.as_mut(), b0.as_ref()) {
        *acc = &*acc - &(v * sum_v);
    }
    if let (Some(acc), Some(v)) = (g.as_mut(), g0.as_ref()) {
        *acc = &*acc - &(v * sum_v);
    }
    (a, b, g)
}

/// `dV_ecp(non-local)/d(radial)` (geminal basis exp/coeff) via the same mesh
/// identity with O_k = `grad_radial` (dln|det|/d(radial)). Returns per-spin
/// `((dexp, dcoeff), (dexp, dcoeff))`. Cart-GTO only (as `grad_radial`).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_radial(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> crate::param_grad_basis::RadialGrads {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        j3,
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
        cell_nmax,
        NN_DEFAULT,
    );
    let cell = cell_nmax.map(|(c, _)| c);
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        param_grad_basis::grad_radial(
            geminal, a2s_up, n_shell_up, a2s_dn, n_shell_dn, ru, rd, cell,
        )
    };
    let base = ok(r_up, r_dn);
    let mut up = (vec![0.0; base.0 .0.len()], vec![0.0; base.0 .1.len()]);
    let mut dn = (vec![0.0; base.1 .0.len()], vec![0.0; base.1 .1.len()]);
    let axpy = |dst: &mut (Vec<f64>, Vec<f64>), src: &(Vec<f64>, Vec<f64>), s: f64| {
        for (d, v) in dst.0.iter_mut().zip(src.0.iter()) {
            *d += v * s;
        }
        for (d, v) in dst.1.iter_mut().zip(src.1.iter()) {
            *d += v * s;
        }
    };
    for mp in &mesh {
        let o = if mp.is_up {
            let m = with_moved(r_up, mp.idx, mp.pos);
            ok(&m.view(), r_dn)
        } else {
            let m = with_moved(r_dn, mp.idx, mp.pos);
            ok(r_up, &m.view())
        };
        axpy(&mut up, &o.0, mp.v_grid);
        axpy(&mut dn, &o.1, mp.v_grid);
    }
    axpy(&mut up, &base.0, -sum_v);
    axpy(&mut dn, &base.1, -sum_v);
    (up, dn)
}

/// `dV_ecp(non-local)/d(J3 orbital radial)` (`j3_basis_exp` / `j3_basis_coeff`) via
/// the same mesh identity with `O_k = param_grad_basis::grad_j3_radial`
/// (`d ln|Psi| / d(J3 radial)`). Returns `(dexp, dcoeff)` at the J3 leaf's shell rank.
/// 1:1 `one_move_mesh._j3_has_opt_basis` branch (which assembles the same weighted sum in
/// closed form via the moved-column cotangent; here it is the direct per-config
/// reduce the other Rust `grad_ecp_*` families use).
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_j3_radial(
    geminal: &GeminalData,
    jastrow: &JastrowData,
    j3_orb: &OrbData,
    j3_jm: &ArrayView2<f64>,
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
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> (Vec<f64>, Vec<f64>) {
    let (mesh, sum_v) = collect_mesh(
        geminal,
        jastrow,
        Some((j3_orb, j3_jm)),
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
        cell_nmax,
        NN_DEFAULT,
    );
    let ok = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>| {
        param_grad_basis::grad_j3_radial(j3_orb, j3_jm, a2s, n_shell, ru, rd)
    };
    let base = ok(r_up, r_dn);
    let mut acc = (vec![0.0; base.0.len()], vec![0.0; base.1.len()]);
    let axpy = |dst: &mut (Vec<f64>, Vec<f64>), src: &(Vec<f64>, Vec<f64>), s: f64| {
        for (d, v) in dst.0.iter_mut().zip(src.0.iter()) {
            *d += v * s;
        }
        for (d, v) in dst.1.iter_mut().zip(src.1.iter()) {
            *d += v * s;
        }
    };
    for mp in &mesh {
        let o = if mp.is_up {
            let m = with_moved(r_up, mp.idx, mp.pos);
            ok(&m.view(), r_dn)
        } else {
            let m = with_moved(r_dn, mp.idx, mp.pos);
            ok(r_up, &m.view())
        };
        axpy(&mut acc, &o, mp.v_grid);
    }
    axpy(&mut acc, &base, -sum_v);
    acc
}
