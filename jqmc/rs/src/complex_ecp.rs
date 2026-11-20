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

//! Complex Bloch ECP non-local energy (PBC twist k!=Gamma). The det ratio
//! `Psi(r')/Psi(r) = det(G_mesh)/det(G_base)` is COMPLEX at a twist (the only complex
//! piece of the potential -- Ewald + ECP-local are real distance functions). jnone
//! full-recompute reference, mirror of `coulomb_potential::compute_ecp_nonlocal_parts_all_pairs`
//! with the real det ratio replaced by a complex one. PBC min-image (Wigner-Seitz).

use crate::complex_det::det_ratio_complex;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::Complex64;

use crate::complex_det::{compute_geminal_complex, det_phase_lnabs, ComplexGeminal, ComplexOrb};
use crate::coulomb_potential::legendre;
use crate::ewald::{inv3, min_image};

/// Complex ECP non-local energy `Sum_t V_l(r_eI)*(2l+1)*P_l(costheta)*w_g*ratio` (jnone,
/// all-pairs). `cell`/`nmax` enable the PBC Wigner-Seitz min image.
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_nonlocal_complex(
    gem: &ComplexGeminal,
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
    compute_ecp_nonlocal_withj_complex(
        gem,
        None,
        None,
        None,
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
        0,
    )
}

/// Complex ECP non-local energy WITH Jastrow: the det ratio is multiplied by the
/// Jastrow ratio `exp(J(r')-J(r))` (J1/J2 Williamson + J3 = Re(complex sphe AO) + DTN).
/// Pass `jastrow=j3c=None` for the jnone path. Full-recompute determinant (matches the
/// jnone reference); the Jastrow ratio is the `jastrow_ratio`-zone single-move factor
/// ([`crate::gfmc::ComplexJasRatioState`], issue #757).
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_nonlocal_withj_complex(
    gem: &ComplexGeminal,
    jastrow: Option<&crate::data::JastrowData>,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
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
    // Nearest-ECP-nucleus cutoff per electron (production uses NN=1, matching jqmc's
    // `compute_ecp_non_local_parts_nearest_neighbors`); `nn == 0` => all-pairs (the
    // per-piece kernel tests pass 0, comparing to jqmc with NN=natom).
    nn: usize,
) -> Complex64 {
    // Mesh Jastrow ratio in the `jastrow_ratio` zone (float64 in both modes): the
    // single-move factor of the fast mesh state. The difference of two whole
    // `jastrow_value_complex` values would round in `jastrow_eval`
    // (float32 in mixed), issue #757.
    let jas_ratio = jastrow
        .filter(|jd| {
            jd.one_body.is_some() || jd.two_body.is_some() || jd.dtn.is_some() || j3c.is_some()
        })
        .map(|jd| {
            (
                jd,
                crate::gfmc::ComplexJasRatioState::new(jd, j3c, j_matrix, r_up, r_dn),
            )
        });
    let jratio = |is_up: bool, i: usize, pos: [f64; 3]| -> f64 {
        match &jas_ratio {
            Some((jd, st)) => st.ratio(jd, j3c, r_up, r_dn, is_up, i, pos),
            None => 1.0,
        }
    };
    let nv = weights.len();
    let inv_cell = cell.map(inv3);
    let rel_vec = |x: f64, y: f64, z: f64| -> [f64; 3] {
        match (cell, &inv_cell) {
            (Some(c), Some(iv)) => min_image([x, y, z], c, iv, nmax),
            _ => [x, y, z],
        }
    };
    // Per-electron nearest-NN ECP-nucleus filter (production NN=1), 1:1 the real
    // `compute_ecp_nonlocal_parts_all_pairs`. `nn == 0` (or nn >= natom) => all-pairs.
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
                let dd = rel_vec(
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                );
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
    // base det (complex)
    let g_base = compute_geminal_complex(gem, r_up, r_dn);
    let det_base = det_phase_lnabs(&g_base);
    // For the mesh ratio we move one electron of the up (or dn) leaf and rebuild G.
    let ou_base = gem.orb_up.value(r_up);
    let od_base = gem.orb_dn.value(r_dn);
    let mut v_total = Complex64::new(0.0, 0.0);
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let (cx, cy, cz) = (positions[[ia, 0]], positions[[ia, 1]], positions[[ia, 2]]);
        let l = ang_mom[t];
        let (e, c, p) = (exponents[t], coefficients[t], powers[t]);
        // up electrons: moving electron i changes column i of ou -> G = ou'T*W.
        for i in 0..r_up.nrows() {
            if !nearest_up[i].contains(&ia) {
                continue;
            }
            let rel = rel_vec(cx - r_up[[i, 0]], cy - r_up[[i, 1]], cz - r_up[[i, 2]]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            let v_l = c * crate::coulomb_potential::ecp_pow(nrm, p - 2.0) * (-e * nrm * nrm).exp();
            let w = gem.w_from_dn(&od_base, r_up.nrows());
            let mut acc = Complex64::new(0.0, 0.0);
            for v in 0..nv {
                let (dx, dy, dz) = (
                    grid_points[[v, 0]],
                    grid_points[[v, 1]],
                    grid_points[[v, 2]],
                );
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                let cth = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                let mp = [[
                    r_up[[i, 0]] + rx + nrm * dx,
                    r_up[[i, 1]] + ry + nrm * dy,
                    r_up[[i, 2]] + rz + nrm * dz,
                ]];
                let mp_arr = Array2::from_shape_vec((1, 3), mp[0].to_vec()).unwrap();
                let col = gem.orb_up.value(&mp_arr.view()); // (orb_up, 1)
                                                            // G_mesh = ou_with_col_i_replaced T * W
                let mut ou_m = ou_base.clone();
                for a in 0..ou_m.nrows() {
                    ou_m[[a, i]] = col[[a, 0]];
                }
                let g_mesh = ou_m.t().dot(&w);
                let ratio =
                    det_ratio_complex(det_phase_lnabs(&g_mesh), det_base) * jratio(true, i, mp[0]);
                // Continuum VMC keeps the complex matrix element. Fixed-phase
                // LRDMC uses the separate mesh API and projects each candidate
                // before its sign split in gfmc.rs.
                acc += Complex64::new(((2 * l + 1) as f64) * legendre(l, cth) * weights[v], 0.0)
                    * ratio;
            }
            v_total += Complex64::new(v_l, 0.0) * acc;
        }
        // dn electrons: moving electron i changes column i of od -> W changes col i.
        for i in 0..r_dn.nrows() {
            if !nearest_dn[i].contains(&ia) {
                continue;
            }
            let rel = rel_vec(cx - r_dn[[i, 0]], cy - r_dn[[i, 1]], cz - r_dn[[i, 2]]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            let v_l = c * crate::coulomb_potential::ecp_pow(nrm, p - 2.0) * (-e * nrm * nrm).exp();
            let mut acc = Complex64::new(0.0, 0.0);
            for v in 0..nv {
                let (dx, dy, dz) = (
                    grid_points[[v, 0]],
                    grid_points[[v, 1]],
                    grid_points[[v, 2]],
                );
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                let cth = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                let mp = [
                    r_dn[[i, 0]] + rx + nrm * dx,
                    r_dn[[i, 1]] + ry + nrm * dy,
                    r_dn[[i, 2]] + rz + nrm * dz,
                ];
                let mp_arr = Array2::from_shape_vec((1, 3), mp.to_vec()).unwrap();
                let col = gem.orb_dn.value(&mp_arr.view()); // (orb_dn, 1)
                let mut od_m = od_base.clone();
                for a in 0..od_m.nrows() {
                    od_m[[a, i]] = col[[a, 0]];
                }
                let w_m = gem.w_from_dn(&od_m, r_up.nrows());
                let g_mesh = ou_base.t().dot(&w_m);
                let ratio =
                    det_ratio_complex(det_phase_lnabs(&g_mesh), det_base) * jratio(false, i, mp);
                // The continuum potential is complex; fixed-phase projection
                // belongs to the LRDMC mesh consumer, not this VMC sum.
                acc += Complex64::new(((2 * l + 1) as f64) * legendre(l, cth) * weights[v], 0.0)
                    * ratio;
            }
            v_total += Complex64::new(v_l, 0.0) * acc;
        }
    }
    v_total
}

/// Rank-1 twin of [`compute_ecp_nonlocal_withj_complex`] for the MCMC local energy.
///
/// The reference rebuilds the full complex geminal, its LU determinant and the full
/// Jastrow value for every quadrature point (and again for every channel term), which is
/// `O(N^3)` per point. Every candidate is a single-electron move from the same base, so
/// the determinant ratio is the Sherman-Morrison rank-1 ratio on the base inverse and the
/// Jastrow factor is the incremental single-move ratio: the [`crate::gfmc::MeshFastStateC`]
/// candidate-table mesh kernel the twist GFMC already uses. Like the reference, the
/// continuum VMC sum keeps the complex matrix element `V * Psi'/Psi` (no per-candidate
/// Re projection; `ang` and `jr` are real, so each term is `det_amp * jr`). Fixed-phase
/// LRDMC projects per candidate at its own consumption site in `gfmc.rs`. Returns `None`
/// on the fast-state coverage-gate miss; the caller then keeps the reference path.
#[allow(clippy::too_many_arguments)]
pub fn compute_ecp_nonlocal_withj_complex_fast(
    gem: &ComplexGeminal,
    jastrow: &crate::data::JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
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
    nn: usize,
    mesh_wdot: bool,
    // FAST chunk only: the walker's carried determinant base (the sweep's orbital
    // tables + its running Sherman-Morrison inverse, the inverse the kinetic term
    // already uses). `Some` skips the per-e_L orbital re-evaluation, `G` rebuild and
    // re-inversion (`mesh_fast_state_complex_from`); `None` seeds a fresh base.
    det_seed: Option<(&crate::complex_det::DetRatioStateC, &ArrayView2<Complex64>)>,
) -> Option<Complex64> {
    crate::gfmc::record_ecp_cplx_seed(det_seed.is_some());
    let mut st = crate::timed!(
        "eL_c.ecp_nl_state",
        match det_seed {
            Some((det, ginv)) => {
                crate::gfmc::mesh_fast_state_complex_from(
                    gem,
                    jastrow,
                    j3c,
                    j_matrix,
                    r_up,
                    r_dn,
                    det.clone(),
                    ginv.to_owned(),
                )
            }
            None => crate::gfmc::mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, r_up, r_dn),
        }
    )?;
    st.enable_candidate_tables();
    // FAST chunk chain: AO-space mesh det-ratio contraction (the state is one-shot,
    // so no candidate table would be reused across hops).
    if mesh_wdot {
        st.enable_ecp_wdot();
    }
    // `grid_points` already carries the caller's rotation; the mesh kernel's own
    // rotation is the identity.
    let rt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mesh = crate::timed!(
        "eL_c.ecp_nl_mesh",
        crate::gfmc::compute_ecp_nonlocal_mesh_complex_fast(
            gem,
            jastrow,
            j3c,
            j_matrix,
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
            nn,
            cell.map(|c| (c, nmax)),
            &st,
        )
    );
    let v: Complex64 = mesh.iter().map(|(_, _, _, amp, jr)| *amp * *jr).sum();
    Some(v)
}

/// Complex Bloch ECP non-local **per-candidate mesh** for the twist GFMC move (analog of
/// `coulomb_potential::compute_ecp_nonlocal_mesh` but COMPLEX det ratio). One candidate per
/// (spin, electron i, nearest nucleus ia, quadrature v): returns `(is_up, i, new_pos,
/// det_amp, jr)` with `det_amp = det_phase_lnabs(mesh)/det_phase_lnabs(base) * Sum_channels-at-ia
/// [V_l*(2l+1)*P_l(costheta)*w_v]` (complex) and `jr = exp(DeltaJ)` (real). PBC min-image; rt-rotated grid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ecp_nonlocal_mesh_complex(
    gem: &ComplexGeminal,
    jastrow: &crate::data::JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
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
    cell: Option<&ArrayView2<f64>>,
    nmax: i32,
    // FAST chain mode: `(jastrow_data, shared complex Sherman-Morrison base state)`.
    // `Some` replaces the per-candidate FULL complex geminal + LU + full Jastrow value
    // with a rank-1 complex det ratio times the incremental (real) single-move Jastrow
    // ratio -- the SAME one-electron move class the kinetic mesh uses, so ONE state
    // serves both meshes. `None` keeps the byte-exact reference path.
    fast: Option<(&crate::data::JastrowData, &crate::gfmc::MeshFastStateC)>,
) -> Vec<(bool, usize, [f64; 3], Complex64, f64)> {
    let nv = weights.len();
    let inv_cell = cell.map(inv3);
    let mic = |v: [f64; 3]| -> [f64; 3] {
        match (cell, &inv_cell) {
            (Some(c), Some(iv)) => min_image(v, c, iv, nmax),
            _ => v,
        }
    };
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
    // Reference branch (`fast == None`): the Jastrow factor is the `jastrow_ratio`-zone
    // single-move ratio, the same factor the FAST branch uses; the difference of two whole
    // `jastrow_value_complex` values would round in `jastrow_eval` (float32
    // in mixed), issue #757.
    let with_j = jastrow.one_body.is_some()
        || jastrow.two_body.is_some()
        || j3c.is_some()
        || jastrow.dtn.is_some();
    let jas_ratio = (fast.is_none() && with_j)
        .then(|| crate::gfmc::ComplexJasRatioState::new(jastrow, j3c, j_matrix, r_up, r_dn));
    let jr_ref = |is_up: bool, i: usize, pos: [f64; 3]| -> f64 {
        match &jas_ratio {
            Some(st) => st.ratio(jastrow, j3c, r_up, r_dn, is_up, i, pos),
            None => 1.0,
        }
    };
    let det_base = det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn));
    let ou_base = gem.orb_up.value(r_up);
    let od_base = gem.orb_dn.value(r_dn);
    let w_base = gem.w_from_dn(&od_base, r_up.nrows());
    // Sum over channels at nucleus ia of V_l*(2l+1)*P_l(costheta)*w_v (real).
    let ang_at = |ia: usize, nrm: f64, cth: f64, v: usize| -> f64 {
        let mut a = 0.0_f64;
        for t in 0..nucleus_index.len() {
            if nucleus_index[t] as usize != ia {
                continue;
            }
            let l = ang_mom[t];
            let v_l = coefficients[t]
                * crate::coulomb_potential::ecp_pow(nrm, powers[t] - 2.0)
                * (-exponents[t] * nrm * nrm).exp();
            a += v_l * ((2 * l + 1) as f64) * legendre(l, cth) * weights[v];
        }
        a
    };
    let mut out: Vec<(bool, usize, [f64; 3], Complex64, f64)> = Vec::new();
    // up electrons: moving electron i replaces column i of ou -> G = ou'T*W_base.
    for i in 0..r_up.nrows() {
        let near = nearest_set(r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]);
        for &ia in &near {
            let rel = mic([
                positions[[ia, 0]] - r_up[[i, 0]],
                positions[[ia, 1]] - r_up[[i, 1]],
                positions[[ia, 2]] - r_up[[i, 2]],
            ]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            for (v, gp_v) in gp[..nv].iter().enumerate() {
                let (dx, dy, dz) = (gp_v[0], gp_v[1], gp_v[2]);
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                let cth = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                let new_pos = [
                    r_up[[i, 0]] + rx + nrm * dx,
                    r_up[[i, 1]] + ry + nrm * dy,
                    r_up[[i, 2]] + rz + nrm * dz,
                ];
                let (det_ratio, jr) = match fast {
                    // FAST: rank-1 complex det ratio (holomorphic -- no Re anywhere) +
                    // incremental REAL Jastrow ratio.
                    Some((jd, st)) => (
                        st.det_ratio(gem, true, i, new_pos),
                        st.jas_ratio(jd, j3c, j_matrix, r_up, r_dn, true, i, new_pos),
                    ),
                    None => {
                        let mp_arr = Array2::from_shape_vec((1, 3), new_pos.to_vec()).unwrap();
                        let col = gem.orb_up.value(&mp_arr.view());
                        let mut ou_m = ou_base.clone();
                        for a in 0..ou_m.nrows() {
                            ou_m[[a, i]] = col[[a, 0]];
                        }
                        let dr =
                            det_ratio_complex(det_phase_lnabs(&ou_m.t().dot(&w_base)), det_base);
                        (dr, jr_ref(true, i, new_pos))
                    }
                };
                out.push((
                    true,
                    i,
                    new_pos,
                    Complex64::new(ang_at(ia, nrm, cth, v), 0.0) * det_ratio,
                    jr,
                ));
            }
        }
    }
    // dn electrons: moving electron i replaces column i of od -> W changes.
    for i in 0..r_dn.nrows() {
        let near = nearest_set(r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]);
        for &ia in &near {
            let rel = mic([
                positions[[ia, 0]] - r_dn[[i, 0]],
                positions[[ia, 1]] - r_dn[[i, 1]],
                positions[[ia, 2]] - r_dn[[i, 2]],
            ]);
            let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
            let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
            for (v, gp_v) in gp[..nv].iter().enumerate() {
                let (dx, dy, dz) = (gp_v[0], gp_v[1], gp_v[2]);
                let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                let cth = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                let new_pos = [
                    r_dn[[i, 0]] + rx + nrm * dx,
                    r_dn[[i, 1]] + ry + nrm * dy,
                    r_dn[[i, 2]] + rz + nrm * dz,
                ];
                let (det_ratio, jr) = match fast {
                    Some((jd, st)) => (
                        st.det_ratio(gem, false, i, new_pos),
                        st.jas_ratio(jd, j3c, j_matrix, r_up, r_dn, false, i, new_pos),
                    ),
                    None => {
                        let mp_arr = Array2::from_shape_vec((1, 3), new_pos.to_vec()).unwrap();
                        let col = gem.orb_dn.value(&mp_arr.view());
                        let mut od_m = od_base.clone();
                        for a in 0..od_m.nrows() {
                            od_m[[a, i]] = col[[a, 0]];
                        }
                        let w_m = gem.w_from_dn(&od_m, r_up.nrows());
                        let dr =
                            det_ratio_complex(det_phase_lnabs(&ou_base.t().dot(&w_m)), det_base);
                        (dr, jr_ref(false, i, new_pos))
                    }
                };
                out.push((
                    false,
                    i,
                    new_pos,
                    Complex64::new(ang_at(ia, nrm, cth, v), 0.0) * det_ratio,
                    jr,
                ));
            }
        }
    }
    out
}
