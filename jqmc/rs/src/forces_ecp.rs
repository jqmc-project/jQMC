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

//! ECP non-local nuclear force `dV_ecp/dR` without a Jastrow.
//! Two parts: (a) wavefunction-ratio (O_R identity, reuses determinant Pulay over the
//! quadrature mesh) + (b) explicit projector/mesh geometry (V_l', Legendre',
//! dcostheta, mesh motion). CART molecular (geminal via sphe->cart at the wrapper).

use numpy::ndarray::{Array2, ArrayView2};

use crate::coulomb_potential::{legendre, legendre_deriv};
use crate::data::JastrowData;
use crate::determinant::{compute_geminal_all_elements, det_sign_lnabs, GeminalData, OrbData};
use crate::forces::grad_lndet_wrt_r;
use crate::param_grad_ecp::collect_mesh;
use crate::param_grad_el::det_grad_field;

fn with_moved(base: &ArrayView2<f64>, idx: usize, pos: [f64; 3]) -> Array2<f64> {
    let mut m = base.to_owned();
    m[[idx, 0]] = pos[0];
    m[[idx, 1]] = pos[1];
    m[[idx, 2]] = pos[2];
    m
}

/// Unique ECP nuclei (sorted) appearing in `nucleus_index` -- the per-electron
/// nearest-nucleus candidate set for the NN=1 non-local filter.
fn uniq_nuclei(nucleus_index: &[i64]) -> Vec<usize> {
    let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
    u.sort_unstable();
    u.dedup();
    u
}

/// Nearest ECP nucleus (molecular Euclidean) to electron position `re` among
/// `uniq`. The explicit projector / mesh-motion / drift terms must sum over the
/// SAME per-electron nearest atom as the wf-ratio mesh (`collect_mesh`, NN=1):
/// the production non-local e_L is nearest-neighbour (`NN_default = 1`), so an
/// all-atoms sum over-counts the (exponentially small but nonzero) contribution
/// of non-nearest atoms and diverges from `compute_grad_e_L` for close-packed
/// molecules (H2's second atom is ~0; water's O/H neighbours are not). Ties
/// resolve to the smallest index (strict `<`, `uniq` sorted) -- 1:1 `collect_mesh`.
fn nearest_ecp_nuc(re: [f64; 3], positions: &ArrayView2<f64>, uniq: &[usize]) -> usize {
    let mut best = uniq[0];
    let mut bestd = f64::INFINITY;
    for &ia in uniq {
        let dx = re[0] - positions[[ia, 0]];
        let dy = re[1] - positions[[ia, 1]];
        let dz = re[2] - positions[[ia, 2]];
        let d = dx * dx + dy * dy + dz * dz;
        if d < bestd {
            bestd = d;
            best = ia;
        }
    }
    best
}

/// Legendre `P_l(x)` and `P_l'(x)` (tabulated derivative polynomial, finite at `x = +-1`).
pub(crate) fn leg_and_deriv(l: i64, x: f64) -> (f64, f64) {
    (legendre(l, x), legendre_deriv(l, x))
}

/// `dV_ecp(non-local)/dR` (n_atom,3), jnone. wf-ratio + explicit parts.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_force(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
) -> Array2<f64> {
    grad_ecp_force_gi(
        gem,
        jastrow,
        j3,
        r_up,
        r_dn,
        nuc_up,
        nuc_dn,
        positions,
        nucleus_index,
        ang_mom,
        exponents,
        coefficients,
        powers,
        weights,
        grid_points,
        None,
    )
}

/// `grad_ecp_force` reusing a caller-supplied BASE geminal inverse (`gem_forward_gi`):
/// the MCMC observable VMC ECP force reuses the sweep SM inverse for the base-config
/// `O_R(r)` (near-singular-stable). Mesh re-evals at shifted configs keep their own
/// fresh inverse (jqmc `mesh-config re-evals use their own inverse`). `None` == fresh.
#[allow(clippy::too_many_arguments)]
pub fn grad_ecp_force_gi(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    nuc_up: &[usize],
    nuc_dn: &[usize],
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    let n_atom = positions.nrows();
    let n_up = r_up.nrows();

    // ===== Part (a): wavefunction-ratio = Sum_g V_grid*O_R(r'_g) - sumV*O_R(r) =====
    let (mesh, sum_v) = collect_mesh(
        gem,
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
        None,
        crate::param_grad_ecp::NN_DEFAULT, // molecular ECP force: NN=1, 1:1 the non-local e_L
    );
    let or_base =
        crate::forces::grad_lndet_wrt_r_gi(gem, r_up, r_dn, nuc_up, nuc_dn, n_atom, g_inv);
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    for mp in &mesh {
        let or = if mp.is_up {
            let m = with_moved(r_up, mp.idx, mp.pos);
            grad_lndet_wrt_r(gem, &m.view(), r_dn, nuc_up, nuc_dn, n_atom)
        } else {
            let m = with_moved(r_dn, mp.idx, mp.pos);
            grad_lndet_wrt_r(gem, r_up, &m.view(), nuc_up, nuc_dn, n_atom)
        };
        out += &(or * mp.v_grid);
    }
    out -= &(or_base * sum_v);

    // ===== Part (b): explicit projector / mesh geometry =====
    let (s_base, l_base) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    let nv = weights.len();
    let uniq = uniq_nuclei(nucleus_index); // NN=1 candidate set (1:1 collect_mesh part (a))
                                           // group (atom, l) -> channel contraction terms (coef, pow, exp)
    let elec = |spin: usize, ei: usize| -> [f64; 3] {
        if spin == 0 {
            [r_up[[ei, 0]], r_up[[ei, 1]], r_up[[ei, 2]]]
        } else {
            [r_dn[[ei, 0]], r_dn[[ei, 1]], r_dn[[ei, 2]]]
        }
    };

    for i_atom in 0..n_atom {
        // distinct l-channels at this atom
        let mut ls: Vec<i64> = nucleus_index
            .iter()
            .zip(ang_mom.iter())
            .filter(|(&ia, _)| ia as usize == i_atom)
            .map(|(_, &l)| l)
            .collect();
        ls.sort_unstable();
        ls.dedup();
        let (cx, cy, cz) = (
            positions[[i_atom, 0]],
            positions[[i_atom, 1]],
            positions[[i_atom, 2]],
        );
        for l in ls {
            // channel terms
            let terms: Vec<(f64, f64, f64)> = (0..nucleus_index.len())
                .filter(|&t| nucleus_index[t] as usize == i_atom && ang_mom[t] == l)
                .map(|t| (coefficients[t], powers[t], exponents[t]))
                .collect();
            for spin in 0..2 {
                let nr = if spin == 0 {
                    r_up.nrows()
                } else {
                    r_dn.nrows()
                };
                for ei in 0..nr {
                    let e = if spin == 0 { ei } else { n_up + ei };
                    let re = elec(spin, ei);
                    // NN=1: this electron's projector geometry lives only at its nearest atom.
                    if nearest_ecp_nuc(re, positions, &uniq) != i_atom {
                        continue;
                    }
                    // rel = R_I - r_e ; r_eI = |rel| ; ueI = (r_e - R_I)/r_eI = -rel/r_eI
                    let rel = [cx - re[0], cy - re[1], cz - re[2]];
                    let r_ei = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                    let uei = [-rel[0] / r_ei, -rel[1] / r_ei, -rel[2] / r_ei];
                    let mut vl = 0.0_f64;
                    let mut vlp = 0.0_f64;
                    for &(a, n, b) in &terms {
                        let e2 = (-b * r_ei * r_ei).exp();
                        vl += a * r_ei.powf(n) * e2;
                        vlp += a
                            * ((n - 2.0) * r_ei.powf(n - 3.0) - 2.0 * b * r_ei.powf(n - 1.0))
                            * e2;
                    }
                    vl *= r_ei.powf(-2.0);
                    for v in 0..nv {
                        let vg = [
                            grid_points[[v, 0]],
                            grid_points[[v, 1]],
                            grid_points[[v, 2]],
                        ];
                        let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                        let vhat = [vg[0] / vgn, vg[1] / vgn, vg[2] / vgn];
                        let cth = uei[0] * vhat[0] + uei[1] * vhat[1] + uei[2] * vhat[2];
                        let (pl, dpl) = leg_and_deriv(l, cth);
                        // mesh config: electron e -> R_I + r_eI*v_g
                        let rprime = [cx + r_ei * vg[0], cy + r_ei * vg[1], cz + r_ei * vg[2]];
                        let (mu, md) = if spin == 0 {
                            (with_moved(r_up, ei, rprime), r_dn.to_owned())
                        } else {
                            (r_up.to_owned(), with_moved(r_dn, ei, rprime))
                        };
                        let (sm, lm) = det_sign_lnabs(
                            &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                        );
                        let ratio = sm * s_base * (lm - l_base).exp();
                        let vgrid = vl * ((2 * l + 1) as f64) * pl * weights[v] * ratio;
                        // O = gradlnPsi field at mesh config, electron e
                        let (gdu, gdd) = det_grad_field(gem, &mu.view(), &md.view());
                        let oe = if e < n_up {
                            [gdu[[e, 0]], gdu[[e, 1]], gdu[[e, 2]]]
                        } else {
                            let k = e - n_up;
                            [gdd[[k, 0]], gdd[[k, 1]], gdd[[k, 2]]]
                        };
                        let dcth = [
                            (vhat[0] - cth * uei[0]) / r_ei,
                            (vhat[1] - cth * uei[1]) / r_ei,
                            (vhat[2] - cth * uei[2]) / r_ei,
                        ];
                        let pref = ((2 * l + 1) as f64) * weights[v] * ratio;
                        let v_dot_o = vg[0] * oe[0] + vg[1] * oe[1] + vg[2] * oe[2];
                        for d in 0..3 {
                            // explicit dc_t/dR_I = -(V_l' P_l u_hat + V_l P_l' dcostheta)
                            out[[i_atom, d]] -= pref * (vlp * pl * uei[d] + vl * dpl * dcth[d]);
                            // mesh-motion: dr'/dR = I - v_g(x)u_hat
                            out[[i_atom, d]] += vgrid * (oe[d] - uei[d] * v_dot_o);
                        }
                    }
                }
            }
        }
    }
    out
}

/// `dV_ecp(local)/deps` (3x3) for a PERIODIC ECP system: min-image distance virial
/// `Sum_t Sum_e V_t'(r_eI)*d(x)d/r_eI`, d = min-image(r_e-R_I). Iterates the dedicated
/// `_*_local_part` term arrays (1:1 `compute_ecp_local_parts_all_pairs`):
/// `V_t(r)=c*r^{p-2}*e^{-e r^2}`, `V_t'(r)=c[(p-2)r^{p-3}-2e r^{p-1}]e^{-e r^2}`.
#[allow(clippy::too_many_arguments)]
pub fn grad_vlocal_strain(
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    nmax: i32,
) -> Array2<f64> {
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let inv = crate::ewald::inv3(cell);
    let elec = |e: usize| -> [f64; 3] {
        if e < n_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            let k = e - n_up;
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        }
    };
    let mut out = Array2::<f64>::zeros((3, 3));
    for t in 0..nucleus_index.len() {
        let ia = nucleus_index[t] as usize;
        let (b, c, n) = (exponents[t], coefficients[t], powers[t]);
        for e in 0..n_e {
            let re = elec(e);
            let d0 = [
                re[0] - positions[[ia, 0]],
                re[1] - positions[[ia, 1]],
                re[2] - positions[[ia, 2]],
            ];
            let d = crate::ewald::min_image(d0, cell, &inv, nmax);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let vp =
                c * ((n - 2.0) * r.powf(n - 3.0) - 2.0 * b * r.powf(n - 1.0)) * (-b * r * r).exp();
            for cc in 0..3 {
                for dd in 0..3 {
                    out[[cc, dd]] += vp * d[cc] * d[dd] / r;
                }
            }
        }
    }
    out
}

/// `dV_ecp(local)/dr_e` per electron (Ne,3): full e-N radial local channel,
/// `+V_l'(r_eI)*u_hat_eI` (electron axis, `+` sign).
#[allow(clippy::too_many_arguments)]
pub fn grad_vlocal_wrt_r(
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    max_ang: &[i64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_atom = positions.nrows();
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
    let mut out = Array2::<f64>::zeros((n_e, 3));
    for i_atom in 0..n_atom {
        let lloc = max_ang[i_atom];
        let terms: Vec<(f64, f64, f64)> = (0..nucleus_index.len())
            .filter(|&t| nucleus_index[t] as usize == i_atom && ang_mom[t] == lloc)
            .map(|t| (coefficients[t], powers[t], exponents[t]))
            .collect();
        let (cx, cy, cz) = (
            positions[[i_atom, 0]],
            positions[[i_atom, 1]],
            positions[[i_atom, 2]],
        );
        for e in 0..n_e {
            let re = elec(e);
            let rel = [cx - re[0], cy - re[1], cz - re[2]];
            let r = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let u = [-rel[0] / r, -rel[1] / r, -rel[2] / r];
            let mut vlp = 0.0_f64;
            for &(a, n, b) in &terms {
                vlp += a
                    * ((n - 2.0) * r.powf(n - 3.0) - 2.0 * b * r.powf(n - 1.0))
                    * (-b * r * r).exp();
            }
            for d in 0..3 {
                out[[e, d]] += vlp * u[d];
            }
        }
    }
    out
}

/// `dV_ecp(non-local)/dr_e` per electron (Ne,3), jnone. Mesh-numerator (electron
/// field O at each mesh config, with the moved-electron Jacobian `v_g(x)u_hat`) +
/// explicit projector/weight geometry (moved electron only) - denominator
/// (`sumV*O_base`).
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_wrt_r(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    max_ang: &[i64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
) -> Array2<f64> {
    grad_v_ecp_wrt_r_gi(
        gem,
        r_up,
        r_dn,
        positions,
        nucleus_index,
        ang_mom,
        exponents,
        coefficients,
        powers,
        max_ang,
        weights,
        grid_points,
        None,
    )
}

/// `grad_v_ecp_wrt_r` reusing a caller-supplied BASE geminal inverse (`gem_forward_gi`):
/// MCMC observable VMC ECP drift dV_ecp/dr reuses the sweep SM inverse for the
/// base-config `O` field; mesh re-evals stay fresh. `None` == fresh.
#[allow(clippy::too_many_arguments)]
pub fn grad_v_ecp_wrt_r_gi(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    max_ang: &[i64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    g_inv: Option<&ArrayView2<f64>>,
) -> Array2<f64> {
    let n_atom = positions.nrows();
    let n_up = r_up.nrows();
    let n_e = n_up + r_dn.nrows();
    let nv = weights.len();
    // O field (gradln det) accessor -> (Ne,3) from per-spin det_grad_field result
    let o_full = |gu: &Array2<f64>, gd: &Array2<f64>| -> Array2<f64> {
        let mut o = Array2::<f64>::zeros((n_e, 3));
        for e in 0..n_e {
            for d in 0..3 {
                o[[e, d]] = if e < n_up {
                    gu[[e, d]]
                } else {
                    gd[[e - n_up, d]]
                };
            }
        }
        o
    };
    let (gu0, gd0) = crate::param_grad_el::det_grad_field_gi(gem, r_up, r_dn, g_inv);
    let o_base = o_full(&gu0, &gd0);
    let (s_base, l_base) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    // NN=1 candidate set: atoms carrying a non-local channel (max_ang > 0), 1:1 the
    // non-local drift e_L (nearest-neighbour). The local drift stays all-pairs.
    let uniq: Vec<usize> = uniq_nuclei(nucleus_index)
        .into_iter()
        .filter(|&ia| max_ang[ia] > 0)
        .collect();
    let elec = |spin: usize, ei: usize| -> [f64; 3] {
        if spin == 0 {
            [r_up[[ei, 0]], r_up[[ei, 1]], r_up[[ei, 2]]]
        } else {
            [r_dn[[ei, 0]], r_dn[[ei, 1]], r_dn[[ei, 2]]]
        }
    };
    let mut dv = Array2::<f64>::zeros((n_e, 3));
    for i_atom in 0..n_atom {
        let (cx, cy, cz) = (
            positions[[i_atom, 0]],
            positions[[i_atom, 1]],
            positions[[i_atom, 2]],
        );
        // non-local channels: ang_mom in 0..max_ang_mom_plus_1
        for l in 0..max_ang[i_atom] {
            let terms: Vec<(f64, f64, f64)> = (0..nucleus_index.len())
                .filter(|&t| nucleus_index[t] as usize == i_atom && ang_mom[t] == l)
                .map(|t| (coefficients[t], powers[t], exponents[t]))
                .collect();
            if terms.is_empty() {
                continue;
            }
            for spin in 0..2 {
                let nr = if spin == 0 {
                    r_up.nrows()
                } else {
                    r_dn.nrows()
                };
                for ei in 0..nr {
                    let e = if spin == 0 { ei } else { n_up + ei };
                    let re = elec(spin, ei);
                    // NN=1: this electron's non-local drift lives only at its nearest atom.
                    if nearest_ecp_nuc(re, positions, &uniq) != i_atom {
                        continue;
                    }
                    let rel = [cx - re[0], cy - re[1], cz - re[2]];
                    let r_ei = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
                    let uei = [-rel[0] / r_ei, -rel[1] / r_ei, -rel[2] / r_ei];
                    let mut vl = 0.0_f64;
                    let mut vlp = 0.0_f64;
                    for &(a, n, b) in &terms {
                        let e2 = (-b * r_ei * r_ei).exp();
                        vl += a * r_ei.powf(n) * e2;
                        vlp += a
                            * ((n - 2.0) * r_ei.powf(n - 3.0) - 2.0 * b * r_ei.powf(n - 1.0))
                            * e2;
                    }
                    vl *= r_ei.powf(-2.0);
                    for v in 0..nv {
                        let vg = [
                            grid_points[[v, 0]],
                            grid_points[[v, 1]],
                            grid_points[[v, 2]],
                        ];
                        let vgn = (vg[0] * vg[0] + vg[1] * vg[1] + vg[2] * vg[2]).sqrt();
                        let vhat = [vg[0] / vgn, vg[1] / vgn, vg[2] / vgn];
                        let cth = uei[0] * vhat[0] + uei[1] * vhat[1] + uei[2] * vhat[2];
                        let (pl, dpl) = leg_and_deriv(l, cth);
                        // mesh config: electron e -> R_I + r_eI*v_g
                        let rprime = [cx + r_ei * vg[0], cy + r_ei * vg[1], cz + r_ei * vg[2]];
                        let (mu, md) = if spin == 0 {
                            (with_moved(r_up, ei, rprime), r_dn.to_owned())
                        } else {
                            (r_up.to_owned(), with_moved(r_dn, ei, rprime))
                        };
                        let (sm, lm) = det_sign_lnabs(
                            &compute_geminal_all_elements(gem, &mu.view(), &md.view()).view(),
                        );
                        let ratio = sm * s_base * (lm - l_base).exp();
                        let vgrid = vl * ((2 * l + 1) as f64) * pl * weights[v] * ratio;
                        let (gum, gdm) = det_grad_field(gem, &mu.view(), &md.view());
                        let o_mesh = o_full(&gum, &gdm);
                        // mesh-numerator: i!=e -> Vgrid*O_mesh[i]; i==e -> Vgrid*ueI*(v_g*O_mesh[e])
                        let v_dot_oe = vg[0] * o_mesh[[e, 0]]
                            + vg[1] * o_mesh[[e, 1]]
                            + vg[2] * o_mesh[[e, 2]];
                        for i in 0..n_e {
                            if i == e {
                                for d in 0..3 {
                                    dv[[e, d]] += vgrid * uei[d] * v_dot_oe;
                                }
                            } else {
                                for d in 0..3 {
                                    dv[[i, d]] += vgrid * o_mesh[[i, d]];
                                }
                            }
                        }
                        // explicit (moved electron only): dcostheta/dr_e = (v-costheta*u_hat)/r_eI
                        let dcth = [
                            (vhat[0] - cth * uei[0]) / r_ei,
                            (vhat[1] - cth * uei[1]) / r_ei,
                            (vhat[2] - cth * uei[2]) / r_ei,
                        ];
                        let pref = ((2 * l + 1) as f64) * weights[v] * ratio;
                        for d in 0..3 {
                            dv[[e, d]] += pref * (vlp * pl * uei[d] + vl * dpl * dcth[d]);
                        }
                        // denominator: -Vgrid*O_base (all electrons)
                        for i in 0..n_e {
                            for d in 0..3 {
                                dv[[i, d]] -= vgrid * o_base[[i, d]];
                            }
                        }
                    }
                }
            }
        }
    }
    dv
}

/// `dV_ecp(local)/dR` (n_atom,3): translation identity `d/dR_I=-d/dr_e` over the
/// short-range local channel (`ang == max_ang_mom_plus_1`).
#[allow(clippy::too_many_arguments)]
pub fn grad_vlocal_force(
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    max_ang: &[i64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let n_atom = positions.nrows();
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
    let mut out = Array2::<f64>::zeros((n_atom, 3));
    for i_atom in 0..n_atom {
        let lloc = max_ang[i_atom];
        let terms: Vec<(f64, f64, f64)> = (0..nucleus_index.len())
            .filter(|&t| nucleus_index[t] as usize == i_atom && ang_mom[t] == lloc)
            .map(|t| (coefficients[t], powers[t], exponents[t]))
            .collect();
        let (cx, cy, cz) = (
            positions[[i_atom, 0]],
            positions[[i_atom, 1]],
            positions[[i_atom, 2]],
        );
        for e in 0..n_e {
            let re = elec(e);
            let rel = [cx - re[0], cy - re[1], cz - re[2]];
            let r = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
            let u = [-rel[0] / r, -rel[1] / r, -rel[2] / r]; // (r_e - R_I)/r
            let mut vlp = 0.0_f64;
            for &(a, n, b) in &terms {
                vlp += a
                    * ((n - 2.0) * r.powf(n - 3.0) - 2.0 * b * r.powf(n - 1.0))
                    * (-b * r * r).exp();
            }
            for d in 0..3 {
                out[[i_atom, d]] += -vlp * u[d];
            }
        }
    }
    out
}
