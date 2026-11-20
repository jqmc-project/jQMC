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

//! Closed-form parameter gradients `O_k = dlnPsi/dc` (the SR / linear-method log
//! derivatives).
//!
//! Each function differentiates the EXACT jqmc forward kernel (the `_auto` =
//! `jax.grad(forward)` oracle) in closed form -- no autodiff. This module covers
//! first derivatives of lnPsi: geminal `lambda` / `mo_coefficients`, Jastrow
//! J1 / J2 / J3 `j_matrix`. DTN and basis-param live in `param_grad_dtn` /
//! `param_grad_basis` (separate, need extra `from_py` data).
//!
//! Conventions: real Gamma-point and un-conjugated O_k.

use numpy::ndarray::{Array1, Array2, ArrayView2};

use crate::data::{JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{inverse, GeminalData, OrbData};
use crate::jastrow_factor::{smooth_dist, CellInfo};

#[inline]
fn eucl(a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize) -> f64 {
    let dx = a[[i, 0]] - b[[j, 0]];
    let dy = a[[i, 1]] - b[[j, 1]];
    let dz = a[[i, 2]] - b[[j, 2]];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// `q(y) = 1 - (1 + y) exp(-y)` without cancellation at small `|y|`.
///
/// The exp-form Jastrow parameter derivative is `-q(y) / (2 a^2)`: `q` is
/// O(y^2) while both of its terms are O(y), so `-(1 - e) + y e` loses every
/// digit at `y ~ 2^-26` (and even an `exp_m1` rewriting keeps ~eps/y relative
/// error). Below `|y| = 1/4` the Taylor series
/// `sum_{n>=2} (-1)^n (n-1) y^n / n!` through `y^13` (first omitted term
/// < 3e-17 relative); above it `(1 - e) - y e`, `e = exp(-y)`, which has no
/// significant cancellation there (`q(1/4) = 0.026`). Mirrors the XLA
/// `derivatives.assemble._one_minus_one_plus_x_exp_neg_x`.
#[inline]
fn one_minus_one_plus_y_exp_neg(y: f64) -> f64 {
    if y.abs() < 0.25 {
        // Horner in y of c_n = (-1)^n (n-1)/n!, n = 13 down to 2, times y^2.
        const C: [f64; 12] = [
            1.0 / 2.0,
            -2.0 / 6.0,
            3.0 / 24.0,
            -4.0 / 120.0,
            5.0 / 720.0,
            -6.0 / 5040.0,
            7.0 / 40320.0,
            -8.0 / 362880.0,
            9.0 / 3628800.0,
            -10.0 / 39916800.0,
            11.0 / 479001600.0,
            -12.0 / 6227020800.0,
        ];
        let mut poly = C[11];
        for k in (0..11).rev() {
            poly = poly * y + C[k];
        }
        y * y * poly
    } else {
        let e = (-y).exp();
        (1.0 - e) - y * e
    }
}

/// `O_k = dJ1/dp` (scalar).
/// Differentiates `compute_jastrow_one_body` wrt `jastrow_1b_param`.
pub fn grad_j1_param(
    data: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let a = data.jastrow_1b_param;
    let r_atoms = data.structure_data.positions.view();
    let n_atoms = data.structure_data.n_atoms();
    let is_exp = match data.jastrow_1b_type.as_str() {
        "exp" => true,
        "pade" => false,
        other => panic!("Unknown jastrow_1b_type: {other}"),
    };
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));

    // d(prefac*kernel)/da; prefac is param-independent.
    let dcontrib = |r_en: f64, z_eff: f64| -> f64 {
        let coeff = (2.0 * z_eff).powf(0.25);
        let prefac = if is_exp {
            -(2.0 * z_eff).powf(0.75)
        } else {
            -2.0 * z_eff
        };
        let dkernel = if is_exp {
            // -(1 - e)/(2a^2) + (c r) e/(2a) = -q(a c r)/(2a^2).
            -one_minus_one_plus_y_exp_neg(a * coeff * r_en) / (2.0 * a * a)
        } else {
            let den = 1.0 + a * coeff * r_en;
            -coeff * r_en * r_en / (2.0 * den * den)
        };
        prefac * dkernel
    };

    let spin_sum = |elec: &ArrayView2<f64>| -> f64 {
        let mut s = 0.0_f64;
        for e in 0..elec.nrows() {
            for at in 0..n_atoms {
                let z_eff = data.structure_data.atomic_numbers[at] - data.core_electrons[at];
                let r_en = match &ci {
                    Some(ci) => {
                        smooth_dist(
                            [
                                elec[[e, 0]] - r_atoms[[at, 0]],
                                elec[[e, 1]] - r_atoms[[at, 1]],
                                elec[[e, 2]] - r_atoms[[at, 2]],
                            ],
                            ci,
                        )
                        .0
                    }
                    None => eucl(elec, e, &r_atoms, at),
                };
                s += dcontrib(r_en, z_eff);
            }
        }
        s
    };
    spin_sum(r_up) + spin_sum(r_dn)
}

/// `O_k = dJ2/dp` (scalar).
pub fn grad_j2_param(
    data: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f64 {
    let a = data.jastrow_2b_param;
    let is_exp = match data.jastrow_2b_type.as_str() {
        "exp" => true,
        "pade" => false,
        other => panic!("Unknown jastrow_2b_type: {other}"),
    };
    let ci = data.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let df = |r_ij: f64| -> f64 {
        if is_exp {
            // -(1 - e)/(2a^2) + r e/(2a) = -q(a r)/(2a^2).
            -one_minus_one_plus_y_exp_neg(a * r_ij) / (2.0 * a * a)
        } else {
            let den = 1.0 + a * r_ij;
            -r_ij * r_ij / (2.0 * den * den)
        }
    };
    let dist = |x: &ArrayView2<f64>, i: usize, y: &ArrayView2<f64>, j: usize| -> f64 {
        match &ci {
            Some(ci) => {
                smooth_dist(
                    [
                        x[[i, 0]] - y[[j, 0]],
                        x[[i, 1]] - y[[j, 1]],
                        x[[i, 2]] - y[[j, 2]],
                    ],
                    ci,
                )
                .0
            }
            None => eucl(x, i, y, j),
        }
    };

    let mut acc = 0.0_f64;
    // anti-parallel upxdn
    for i in 0..r_up.nrows() {
        for j in 0..r_dn.nrows() {
            acc += df(dist(r_up, i, r_dn, j));
        }
    }
    // parallel within each spin i<j
    let parallel = |elec: &ArrayView2<f64>| -> f64 {
        let n = elec.nrows();
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                s += df(dist(elec, i, elec, j));
            }
        }
        s
    };
    acc + parallel(r_up) + parallel(r_dn)
}

/// `O_k = dJ3/dj_matrix`, shape `(orb_num, orb_num+1)`.
/// `j_matrix = [M | j1col]`, M symmetric
/// orbital-orbital, K = strict-lower-triangular ones. J3 reads `Re(orb)`
/// (no-op for the real Gamma path).
pub fn grad_j3_jmatrix(
    j3_orb: &OrbData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    // Home-cell-wrap the J3-orbital AO eval (jqmc `compute_AOs`): the truncated
    // periodic image sum is enumerated relative to the home cell, so a far-roamed
    // electron (a whole lattice vector out -- e.g. the ECP mesh points fed by
    // `grad_ecp_j3`, or a roamed MCMC walker) would fall beyond the static tiling and
    // collapse the orbital. `dJ3/dj_matrix` is lattice-shift invariant, so wrapping
    // the input is exact. No-op for molecular / pure-PW leaves (`wrap_cell == None`).
    let ruw = j3_orb
        .wrap_electrons(r_up)
        .unwrap_or_else(|| r_up.to_owned());
    let rdw = j3_orb
        .wrap_electrons(r_dn)
        .unwrap_or_else(|| r_dn.to_owned());
    let (r_up, r_dn) = (&ruw.view(), &rdw.view());
    let a_up = j3_orb.compute(r_up); // (orb_num, N_up)
    let a_dn = j3_orb.compute(r_dn); // (orb_num, N_dn)
    j3_jmatrix_from_orb_values(&a_up, &a_dn)
}

/// The basis-free core of [`grad_j3_jmatrix`]: `dJ3/dj_matrix` from the J3 ORBITAL
/// values alone (`(orb_num, N_up)` / `(orb_num, N_dn)`).
///
/// Every J3 leaf -- single-leaf GTO/pGTO, MO-wrapped composite / plane-wave /
/// B-spline, real or complex-Bloch -- reduces to this once its orbital values are in
/// hand, because `J3 = Sum_{i<j} A[.,i] M A[.,j] + Sum_i j1col . A[.,i]` depends on the
/// basis ONLY through `A`. The complex twin (`param_grad_twist::grad_j3_jmatrix_complex`)
/// feeds `Re(A)` here, matching jqmc's `jnp.real(compute_orb(...))` projection.
pub fn j3_jmatrix_from_orb_values(a_up: &Array2<f64>, a_dn: &Array2<f64>) -> Array2<f64> {
    let orb_num = a_up.nrows();

    // sum_K = A*KT*AT, K=strict-lower ones -> (sum_K)[p,q] = Sum_{i<j} A[p,i] A[q,j].
    let sum_k = |a: &Array2<f64>| -> Array2<f64> {
        let n = a.ncols();
        // c[p,j] = Sum_{i<j} A[p,i]  (prefix sums over electron axis)
        let mut c = Array2::<f64>::zeros((orb_num, n));
        for p in 0..orb_num {
            let mut pref = 0.0_f64;
            for j in 0..n {
                c[[p, j]] = pref;
                pref += a[[p, j]];
            }
        }
        // (A*KT*AT)[p,q] = Sum_j c[p,j]*A[q,j]
        c.dot(&a.t())
    };

    let mut grad_m = sum_k(a_up);
    grad_m += &sum_k(a_dn);
    // outer(s_up, s_dn)
    let s_up = a_up.sum_axis(numpy::ndarray::Axis(1)); // (orb_num,)
    let s_dn = a_dn.sum_axis(numpy::ndarray::Axis(1));
    for p in 0..orb_num {
        for q in 0..orb_num {
            grad_m[[p, q]] += s_up[p] * s_dn[q];
        }
    }
    // hstack([grad_M, (s_up + s_dn)[:,None]])
    let mut out = Array2::<f64>::zeros((orb_num, orb_num + 1));
    out.slice_mut(numpy::ndarray::s![.., 0..orb_num])
        .assign(&grad_m);
    for p in 0..orb_num {
        out[[p, orb_num]] = s_up[p] + s_dn[p];
    }
    out
}

// ---- Exact one-electron O_k increments for the ECP mesh reduce ----
// Every ECP non-local mesh candidate moves ONE electron, so the mesh identity
// `Sum_g v_g O_k(r'_g) - (Sum_g v_g) O_k(r) = Sum_g v_g [O_k(r'_g) - O_k(r)]` only
// needs the change of O_k under a single-electron move. For the additive J1 (one
// term per electron) and pairwise J2 (one row of pair terms per electron) that change
// is the moved electron's own term(s) at the new minus the old position, evaluated by
// the SAME closed-form kernels on a one-electron slice; for J3 `j_matrix` it is
// exactly linear in the moved orbital column (see `j3_jmatrix_mesh_increment`).
// Cost per candidate drops from O(N) / O(N^2) / O(N orb^2 + N orb n_img) to
// O(n_atoms) / O(N) / O(orb n_img).

#[inline]
fn row1(p: [f64; 3]) -> Array2<f64> {
    Array2::from_shape_vec((1, 3), p.to_vec()).expect("1x3")
}

#[inline]
fn row_of(r: &ArrayView2<f64>, i: usize) -> [f64; 3] {
    [r[[i, 0]], r[[i, 1]], r[[i, 2]]]
}

/// `O_k^{J1}(r with electron (is_up, i) at pos) - O_k^{J1}(r)` (exact).
pub(crate) fn j1_param_move_delta(
    data: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    i: usize,
    pos: [f64; 3],
) -> f64 {
    let empty = Array2::<f64>::zeros((0, 3));
    let old = if is_up {
        row_of(r_up, i)
    } else {
        row_of(r_dn, i)
    };
    let one = |p: [f64; 3]| grad_j1_param(data, &row1(p).view(), &empty.view());
    one(pos) - one(old)
}

/// All electrons except `(is_up, i)` stacked `(N - 1, 3)` -- the partners of the
/// moved electron in every J2 pair (J2 uses the same kernel for both spin pairings).
pub(crate) fn j2_partners(
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    i: usize,
) -> Array2<f64> {
    let n = r_up.nrows() + r_dn.nrows() - 1;
    let mut out = Array2::<f64>::zeros((n, 3));
    let mut k = 0;
    for spin_up in [true, false] {
        let n_s = if spin_up { r_up.nrows() } else { r_dn.nrows() };
        for e in 0..n_s {
            if spin_up == is_up && e == i {
                continue;
            }
            for d in 0..3 {
                out[[k, d]] = if spin_up { r_up[[e, d]] } else { r_dn[[e, d]] };
            }
            k += 1;
        }
    }
    out
}

/// The moved electron's own J2 O_k row `Sum_j dJ2/dp(|pos - r_j|)` over `partners`
/// ([`j2_partners`]); `O_k^{J2}` changes under a one-electron move by
/// `j2_param_row(new) - j2_param_row(old)` (exact).
pub(crate) fn j2_param_row(
    data: &JastrowTwoBodyData,
    partners: &ArrayView2<f64>,
    pos: [f64; 3],
) -> f64 {
    grad_j2_param(data, &row1(pos).view(), partners)
}

/// Exact `Sum_g v_g [O_k(r'_g) - O_k(r)]` for `O_k = dJ3/dj_matrix`, shape
/// `(orb, orb + 1)`, from the J3 orbital values `a_up` / `a_dn` and the per-electron
/// aggregated column changes `d_up[:, i] = Sum_{g moves up i} v_g (a(r'_g) - a_up[:, i])`
/// (likewise `d_dn`).
///
/// [`j3_jmatrix_from_orb_values`] is `[M | s]` with `M = SK(A_up) + SK(A_dn) +
/// s_up s_dn^T`, `SK(A)[p,q] = Sum_{i<j} A[p,i] A[q,j]`, `s = s_up + s_dn`. Replacing
/// up column `i` by `a + delta` changes `SK(A_up)` by `delta R_i^T + L_i delta^T`
/// (`L_i` / `R_i` = sums of the columns left / right of `i`; no `delta delta^T` term
/// because `i != j`), `s_up s_dn^T` by `delta s_dn^T` and `s` by `delta`. Every change is
/// LINEAR in `delta`, so the weighted sum over a move's candidates is the same update
/// applied to their weighted sum `D_i`. Down moves mirror it with `s_up D_i^T` on the
/// left factor.
pub(crate) fn j3_jmatrix_mesh_increment(
    a_up: &Array2<f64>,
    a_dn: &Array2<f64>,
    d_up: &Array2<f64>,
    d_dn: &Array2<f64>,
) -> Array2<f64> {
    let orb = a_up.nrows();
    let s_up = a_up.sum_axis(numpy::ndarray::Axis(1));
    let s_dn = a_dn.sum_axis(numpy::ndarray::Axis(1));
    // Columns L_i (strict prefix) and R_i (strict suffix) of the electron axis, with the
    // opposite-spin column sum folded into R (up moves) or L (down moves).
    let lr =
        |a: &Array2<f64>, add: &Array1<f64>, add_to_right: bool| -> (Array2<f64>, Array2<f64>) {
            let n = a.ncols();
            let tot = a.sum_axis(numpy::ndarray::Axis(1));
            let mut l = Array2::<f64>::zeros((orb, n));
            let mut r = Array2::<f64>::zeros((orb, n));
            let mut pref = Array1::<f64>::zeros(orb);
            for i in 0..n {
                for p in 0..orb {
                    let left = pref[p];
                    let right = tot[p] - left - a[[p, i]];
                    if add_to_right {
                        l[[p, i]] = left;
                        r[[p, i]] = right + add[p];
                    } else {
                        l[[p, i]] = left + add[p];
                        r[[p, i]] = right;
                    }
                    pref[p] += a[[p, i]];
                }
            }
            (l, r)
        };
    let (l_up, r_up) = lr(a_up, &s_dn, true);
    let (l_dn, r_dn) = lr(a_dn, &s_up, false);
    let m = d_up.dot(&r_up.t()) + l_up.dot(&d_up.t()) + d_dn.dot(&r_dn.t()) + l_dn.dot(&d_dn.t());
    let col = d_up.sum_axis(numpy::ndarray::Axis(1)) + d_dn.sum_axis(numpy::ndarray::Axis(1));
    let mut out = Array2::<f64>::zeros((orb, orb + 1));
    out.slice_mut(numpy::ndarray::s![.., 0..orb]).assign(&m);
    for p in 0..orb {
        out[[p, orb]] = col[p];
    }
    out
}

/// Aggregate the per-electron weighted column changes `(d_up, d_dn)` consumed by
/// [`j3_jmatrix_mesh_increment`]: `a_cand[:, g]` is the J3 orbital column at candidate
/// `g`'s position and `moves[g] = (is_up, i, v_g)`.
pub(crate) fn j3_mesh_column_changes(
    a_up: &Array2<f64>,
    a_dn: &Array2<f64>,
    a_cand: &Array2<f64>,
    moves: &[(bool, usize, f64)],
) -> (Array2<f64>, Array2<f64>) {
    let orb = a_up.nrows();
    let mut d_up = Array2::<f64>::zeros((orb, a_up.ncols()));
    let mut d_dn = Array2::<f64>::zeros((orb, a_dn.ncols()));
    for (g, &(is_up, i, vg)) in moves.iter().enumerate() {
        let (d, a) = if is_up {
            (&mut d_up, a_up)
        } else {
            (&mut d_dn, a_dn)
        };
        for p in 0..orb {
            d[[p, i]] += vg * (a_cand[[p, g]] - a[[p, i]]);
        }
    }
    (d_up, d_dn)
}

/// Candidate positions `(n_cand, 3)` of a mesh given as `(is_up, i, pos, v)` moves.
pub(crate) fn mesh_positions(pos: impl ExactSizeIterator<Item = [f64; 3]>) -> Array2<f64> {
    let n = pos.len();
    let mut out = Array2::<f64>::zeros((n, 3));
    for (g, p) in pos.enumerate() {
        out[[g, 0]] = p[0];
        out[[g, 1]] = p[1];
        out[[g, 2]] = p[2];
    }
    out
}

/// `O_k = dln|det G|/dlambda` (AGP `lambda_matrix`), shape == `lambda.shape`.
/// Uses the determinant inverse contraction for `dln|det G|/dlambda`.
pub fn grad_ln_det_lambda(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let orb_up = gem.orb_up.compute(r_up); // (M_up, N_up)
    let orb_dn = gem.orb_dn.compute(r_dn); // (M_dn, N_dn)
    grad_ln_det_lambda_from_orb(gem, &orb_up.view(), &orb_dn.view())
}

/// Orbital-core of [`grad_ln_det_lambda`]: everything after the (per-electron
/// independent) `orb.compute`, given the already-evaluated orbital matrices
/// `orb_up` (M_up, N_up) and `orb_dn` (M_dn, N_dn). Byte-identical to
/// `grad_ln_det_lambda` when fed `orb.compute(r)`; the ECP-nonlocal `dV/dlambda`
/// mesh reduce reuses the base orbital matrix for the un-moved spin/columns and
/// re-evaluates only the single moved column.
pub(crate) fn grad_ln_det_lambda_from_orb(
    gem: &GeminalData,
    orb_up: &ArrayView2<f64>,
    orb_dn: &ArrayView2<f64>,
) -> Array2<f64> {
    let m_dn = gem.orb_num_dn;
    let n_dn = orb_dn.ncols();

    let a = orb_up.t().to_owned(); // A = orb_upT : (N_up, M_up)
    let b = orb_dn.to_owned(); // B : (M_dn, N_dn)

    let lam = gem
        .lambda
        .as_ref()
        .expect("grad_ln_det_lambda is undefined for occupied-only SD")
        .view();
    let lam_p = lam.slice(numpy::ndarray::s![.., 0..m_dn]); // (M_up, M_dn)
    let lam_u = lam.slice(numpy::ndarray::s![.., m_dn..]); // (M_up, n_unpaired)

    // G = [A*lambda_p*B | A*lambda_u]  (N_up, N_up)
    let g_paired = a.dot(&lam_p).dot(&b); // (N_up, N_dn)
    let g_unpaired = a.dot(&lam_u); // (N_up, n_unpaired)
    let n_up = a.nrows();
    let mut g = Array2::<f64>::zeros((n_up, n_up));
    g.slice_mut(numpy::ndarray::s![.., 0..n_dn])
        .assign(&g_paired);
    g.slice_mut(numpy::ndarray::s![.., n_dn..])
        .assign(&g_unpaired);

    let g_inv = inverse(&g.view());
    let p = g_inv.dot(&a); // P = G^-^1*A : (N_up, M_up)

    // grad_paired = P[:N_dn]T * BT : (M_up, M_dn)
    let p_top = p.slice(numpy::ndarray::s![0..n_dn, ..]); // (N_dn, M_up)
    let grad_paired = p_top.t().dot(&b.t()); // (M_up, M_dn)
    let grad_unpaired = p.slice(numpy::ndarray::s![n_dn.., ..]).t().to_owned(); // (M_up, n_unpaired)

    let m_up = grad_paired.nrows();
    let mut out = Array2::<f64>::zeros((m_up, lam.ncols()));
    out.slice_mut(numpy::ndarray::s![.., 0..m_dn])
        .assign(&grad_paired);
    out.slice_mut(numpy::ndarray::s![.., m_dn..])
        .assign(&grad_unpaired);
    out
}

/// AGPn `O_k` on the natural-orbital leaves. The AGPn `lambda_matrix` is
/// `hstack([diag(lambda_diag), lambda_unpaired])`, so by the chain rule
/// `dln|det|/d lambda_diag[k]` = the paired-block DIAGONAL of the dense-`lambda`
/// gradient (`lambda_matrix[k,k] = lambda_diag[k]`, off-diagonal paired entries are
/// structural zeros), and `dln|det|/d lambda_unpaired` = the unpaired columns.
/// 1:1 with `derivatives.geminal.dln_det_dlambda_bare`. Real Gamma path.
pub fn grad_ln_det_lambda_diag(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let dense = grad_ln_det_lambda(gem, r_up, r_dn);
    let m_dn = gem.orb_num_dn;
    let n = m_dn.min(dense.nrows());
    let diag = Array1::from_iter((0..n).map(|k| dense[[k, k]]));
    let unpaired = dense.slice(numpy::ndarray::s![.., m_dn..]).to_owned();
    (diag, unpaired)
}

/// `O_k = dln|det G|/dC` (`mo_coefficients`) for up and down spins. Requires MO orbitals (SD/AGPn);
/// panics if a spin has no `mo_coefficients`.
pub fn grad_ln_det_mo(
    gem: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let a_up = gem.orb_up.eval_aos(r_up); // (num_ao_up, N_up)
    let a_dn = gem.orb_dn.eval_aos(r_dn); // (num_ao_dn, N_dn)
    grad_ln_det_mo_from_ao(gem, &a_up.view(), &a_dn.view())
}

/// AO-core of [`grad_ln_det_mo`]: everything after the (per-electron-independent)
/// `eval_aos`, given the already-evaluated AO matrices `a_up` (num_ao_up, N_up) and
/// `a_dn` (num_ao_dn, N_dn). Byte-identical to `grad_ln_det_mo` when fed
/// `orb.eval_aos(r)`. The ECP-nonlocal `dV/dC` mesh reduce reuses the base AO for
/// the un-moved spin and columns and only re-evaluates the single moved column,
/// so the per-mesh-point cost drops to small GEMMs (no redundant full AO sweep).
pub(crate) fn grad_ln_det_mo_from_ao(
    gem: &GeminalData,
    a_up: &ArrayView2<f64>,
    a_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let n_dn = a_dn.ncols();

    let c_up = gem
        .orb_up
        .mo_coefficients
        .as_ref()
        .expect("grad_ln_det_mo: up spin has no mo_coefficients");
    let c_dn = gem
        .orb_dn
        .mo_coefficients
        .as_ref()
        .expect("grad_ln_det_mo: dn spin has no mo_coefficients");
    let orb_up = c_up.dot(a_up); // (m_up, N_up)
    let orb_dn = c_dn.dot(a_dn); // (m_dn, N_dn)

    let n_up = orb_up.ncols();
    let w = gem.w_from_dn(&orb_dn.view(), n_up);
    let g = orb_up.t().dot(&w);
    let g_inv = inverse(&g.view());

    // W = [lambda_p*orb_dn | lambda_u]  (m_up, N_up);  grad_orb_up = W*G^-^1
    let grad_orb_up = w.dot(&g_inv); // (m_up, N_up)
    let grad_c_up = grad_orb_up.dot(&a_up.t()); // (m_up, num_ao_up)

    // Q = orb_upT*lambda_p (N_up, m_dn); grad_orb_dn = (G^-^1*Q)[:N_dn]T
    let q = gem.up_t_times_paired(&orb_up.view()); // (N_up, m_dn)
    let giq = g_inv.dot(&q); // (N_up, m_dn)
    let grad_orb_dn = giq.slice(numpy::ndarray::s![0..n_dn, ..]).t().to_owned(); // (m_dn, N_dn)
    let grad_c_dn = grad_orb_dn.dot(&a_dn.t()); // (m_dn, num_ao_dn)

    (grad_c_up, grad_c_dn)
}
