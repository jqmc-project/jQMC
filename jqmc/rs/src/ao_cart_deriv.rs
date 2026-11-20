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

//! Autograd-free arbitrary-order derivative kernel for Cartesian-GTO AOs. Each contracted cart GTO
//! factorizes per axis, so any mixed derivative `d^ax d^ay d^az` is a product of
//! 1D Hermite-Gaussian derivatives `P_k(t;n,Z)*e^{-Zt^2}`. Foundation for forces
//! used by nuclear-force kernels: gradient, Laplacian, Hessian, and
//! gradient-of-Laplacian (third order).

use numpy::ndarray::{Array2, ArrayView2};

use crate::atomic_orbital::AosGtoData;

#[inline]
fn powi_safe(t: f64, m: i64) -> f64 {
    if m < 0 {
        0.0
    } else {
        powi_inline(t, m as u32)
    }
}

/// `t.powi(m)` for `m >= 0`, inlined: the same square-and-multiply sequence as the
/// `__powidf2` runtime routine `f64::powi` lowers to for a run-time exponent (so the result
/// is bit-identical), without the out-of-line call that dominated the periodic AO
/// derivative sweeps (up to 12 calls per primitive/electron pair).
#[inline]
pub(crate) fn powi_inline(t: f64, m: u32) -> f64 {
    let (mut a, mut pow, mut mul) = (t, m, 1.0_f64);
    loop {
        if pow & 1 != 0 {
            mul *= a;
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        a *= a;
    }
    mul
}

/// 1D Hermite-Gaussian derivative polynomial `P_k(t; n, Z)` (no `e^{-Zt^2}`),
/// closed form for k <= 3 (`P_{k+1}=P_k'-2Z t P_k`).
#[inline]
fn p1d(t: f64, n: i64, z: f64, k: u32) -> f64 {
    let nf = n as f64;
    match k {
        0 => powi_safe(t, n),
        1 => nf * powi_safe(t, n - 1) - 2.0 * z * powi_safe(t, n + 1),
        2 => {
            nf * (nf - 1.0) * powi_safe(t, n - 2) - 2.0 * z * (2.0 * nf + 1.0) * powi_safe(t, n)
                + 4.0 * z * z * powi_safe(t, n + 2)
        }
        3 => {
            nf * (nf - 1.0) * (nf - 2.0) * powi_safe(t, n - 3)
                - 6.0 * z * nf * nf * powi_safe(t, n - 1)
                + 12.0 * z * z * (nf + 1.0) * powi_safe(t, n + 1)
                - 8.0 * z * z * z * powi_safe(t, n + 3)
        }
        _ => panic!("p1d: order {k} > 3 unsupported"),
    }
}

/// Cartesian-GTO AO derivative of order `(ax,ay,az)`, shape `(num_ao, N)`.
/// Molecular cart leaves only (`num_ao_kernel == num_ao`).
pub fn ao_deriv(aos: &AosGtoData, r: &ArrayView2<f64>, order: (u32, u32, u32)) -> Array2<f64> {
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let mut out = Array2::<f64>::zeros((num_ao, n_e));
    for p in 0..aos.exponents.len() {
        let z = aos.exponents[p];
        let nc = aos.normalizations[p] * aos.coefficients[p];
        let ao = aos.orbital_indices[p];
        let nx = aos.poly_x[ao];
        let ny = aos.poly_y[ao];
        let nz = aos.poly_z[ao];
        let (cx, cy, cz) = (
            aos.centers[[ao, 0]],
            aos.centers[[ao, 1]],
            aos.centers[[ao, 2]],
        );
        for e in 0..n_e {
            let x = r[[e, 0]] - cx;
            let y = r[[e, 1]] - cy;
            let zc = r[[e, 2]] - cz;
            let r2 = x * x + y * y + zc * zc;
            let rad = nc * (-z * r2).exp();
            out[[ao, e]] +=
                rad * p1d(x, nx, z, order.0) * p1d(y, ny, z, order.1) * p1d(zc, nz, z, order.2);
        }
    }
    out
}

/// Fused molecular cart-GTO derivative bundle: `grad`, symmetric `hessian`,
/// `gradlap` (grad of Laplacian) and `laplacian`, all produced in ONE sweep over
/// the primitive table. Each order shares the single `exp` and the 1D Hermite
/// polynomials `p1d(.,.,.,0..=3)` per (primitive, electron) instead of the 18
/// independent [`ao_deriv`] passes (each re-evaluating `exp`). The per-order
/// accumulation order and the final elementwise combinations reproduce
/// [`ao_grad`]/[`ao_hessian`]/[`ao_gradlap`]/[`ao_laplacian`] BIT-FOR-BIT (same
/// primitive-outer/electron-inner loop, same product association, same sums).
pub struct AoDerivBundle {
    pub grad: [Array2<f64>; 3],
    pub gradlap: [Array2<f64>; 3],
    pub hessian: [[Array2<f64>; 3]; 3],
    pub laplacian: Array2<f64>,
}

/// Build an [`AoDerivBundle`] for molecular cart leaves (`num_ao_kernel == num_ao`).
pub fn ao_deriv_bundle(aos: &AosGtoData, r: &ArrayView2<f64>) -> AoDerivBundle {
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let z0 = || Array2::<f64>::zeros((num_ao, n_e));
    // grad
    let (mut a100, mut a010, mut a001) = (z0(), z0(), z0());
    // hessian distinct orders (200/020/002 also feed the Laplacian)
    let (mut a200, mut a020, mut a002) = (z0(), z0(), z0());
    let (mut a110, mut a101, mut a011) = (z0(), z0(), z0());
    // gradlap distinct orders
    let (mut a300, mut a030, mut a003) = (z0(), z0(), z0());
    let (mut a120, mut a102) = (z0(), z0());
    let (mut a210, mut a012) = (z0(), z0());
    let (mut a201, mut a021) = (z0(), z0());
    for p in 0..aos.exponents.len() {
        let z = aos.exponents[p];
        let nc = aos.normalizations[p] * aos.coefficients[p];
        let ao = aos.orbital_indices[p];
        let nx = aos.poly_x[ao];
        let ny = aos.poly_y[ao];
        let nz = aos.poly_z[ao];
        let (cx, cy, cz) = (
            aos.centers[[ao, 0]],
            aos.centers[[ao, 1]],
            aos.centers[[ao, 2]],
        );
        for e in 0..n_e {
            let x = r[[e, 0]] - cx;
            let y = r[[e, 1]] - cy;
            let zc = r[[e, 2]] - cz;
            let r2 = x * x + y * y + zc * zc;
            let rad = nc * (-z * r2).exp();
            let (px0, px1, px2, px3) = (
                p1d(x, nx, z, 0),
                p1d(x, nx, z, 1),
                p1d(x, nx, z, 2),
                p1d(x, nx, z, 3),
            );
            let (py0, py1, py2, py3) = (
                p1d(y, ny, z, 0),
                p1d(y, ny, z, 1),
                p1d(y, ny, z, 2),
                p1d(y, ny, z, 3),
            );
            let (pz0, pz1, pz2, pz3) = (
                p1d(zc, nz, z, 0),
                p1d(zc, nz, z, 1),
                p1d(zc, nz, z, 2),
                p1d(zc, nz, z, 3),
            );
            // grad: rad*px1*py0*pz0 etc. -- identical association to ao_deriv.
            a100[[ao, e]] += rad * px1 * py0 * pz0;
            a010[[ao, e]] += rad * px0 * py1 * pz0;
            a001[[ao, e]] += rad * px0 * py0 * pz1;
            // hessian
            a200[[ao, e]] += rad * px2 * py0 * pz0;
            a020[[ao, e]] += rad * px0 * py2 * pz0;
            a002[[ao, e]] += rad * px0 * py0 * pz2;
            a110[[ao, e]] += rad * px1 * py1 * pz0;
            a101[[ao, e]] += rad * px1 * py0 * pz1;
            a011[[ao, e]] += rad * px0 * py1 * pz1;
            // gradlap
            a300[[ao, e]] += rad * px3 * py0 * pz0;
            a030[[ao, e]] += rad * px0 * py3 * pz0;
            a003[[ao, e]] += rad * px0 * py0 * pz3;
            a120[[ao, e]] += rad * px1 * py2 * pz0;
            a102[[ao, e]] += rad * px1 * py0 * pz2;
            a210[[ao, e]] += rad * px2 * py1 * pz0;
            a012[[ao, e]] += rad * px0 * py1 * pz2;
            a201[[ao, e]] += rad * px2 * py0 * pz1;
            a021[[ao, e]] += rad * px0 * py2 * pz1;
        }
    }
    // laplacian = a200 + a020 + a002 (matches ao_laplacian's += order)
    let mut laplacian = a200.clone();
    laplacian += &a020;
    laplacian += &a002;
    // gradlap components (match ao_gradlap's += order)
    let mut gx = a300;
    gx += &a120;
    gx += &a102;
    let mut gy = a210;
    gy += &a030;
    gy += &a012;
    let mut gz = a201;
    gz += &a021;
    gz += &a003;
    AoDerivBundle {
        grad: [a100, a010, a001],
        gradlap: [gx, gy, gz],
        hessian: [
            [a200, a110.clone(), a101.clone()],
            [a110, a020, a011.clone()],
            [a101, a011, a002],
        ],
        laplacian,
    }
}

/// FAST-mode [`ao_deriv_bundle`] (molecular Cartesian leaves): the derivative
/// bundle of the FAST e_L kinetic, which contracts unscreened AO values (the
/// determinant / J3 ratio states) with AO gradients and Laplacians on the FAST
/// derivative screen (`pz_deriv * r^2 <= LEPSBAS`, [`crate::atomic_orbital::Shell::pz_deriv`],
/// `compute_aos_grad_lap_fast_cut`).
///
/// * `grad` is the derivative of the (unscreened) values: never screened.
/// * `hessian`, `gradlap` (and `laplacian`) are the derivatives of the screened
///   gradients / Laplacians: exact zeros for the (shell, electron) pairs the
///   FAST derivative screen drops, the same pairs the FAST kinetic drops.
/// * Evaluation: shell-major. The radial `exp` is taken once per (shell
///   primitive, electron) and the 1D Hermite factors `P_k(t; n, Z)` once per
///   (axis, n <= l), from a power table, instead of per AO with `powi`.
///   Kept entries equal [`ao_deriv_bundle`] to fp rounding.
///
/// Un-finalized leaves (no shells) fall back to the exact bundle.
pub fn ao_deriv_bundle_fast_cut(aos: &AosGtoData, r: &ArrayView2<f64>) -> AoDerivBundle {
    if aos.shells.is_empty() || aos.n_image != 1 {
        return ao_deriv_bundle(aos, r);
    }
    let lepsbas = crate::atomic_orbital::LEPSBAS;
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let z0 = || Array2::<f64>::zeros((num_ao, n_e));
    let (mut a100, mut a010, mut a001) = (z0(), z0(), z0());
    let (mut a200, mut a020, mut a002) = (z0(), z0(), z0());
    let (mut a110, mut a101, mut a011) = (z0(), z0(), z0());
    let (mut a300, mut a030, mut a003) = (z0(), z0(), z0());
    let (mut a120, mut a102) = (z0(), z0());
    let (mut a210, mut a012) = (z0(), z0());
    let (mut a201, mut a021) = (z0(), z0());
    let lmax = aos
        .shells
        .iter()
        .map(|s| s.l.max(0) as usize)
        .max()
        .unwrap_or(0);
    let stride = 4 * (lmax + 1); // tab[(a * n_e + e) * stride + 4n + k] = P_k(t_a(e); n, Z)
    let mut tab = vec![0.0_f64; 3 * n_e * stride];
    let mut pw = vec![0.0_f64; lmax + 4];
    let mut t = vec![[0.0_f64; 3]; n_e];
    let mut r2 = vec![0.0_f64; n_e];
    let mut alive = vec![false; n_e];
    let mut ex = vec![0.0_f64; n_e];
    for sh in &aos.shells {
        let l = sh.l.max(0) as usize;
        let [cx, cy, cz] = sh.center;
        for e in 0..n_e {
            t[e] = [r[[e, 0]] - cx, r[[e, 1]] - cy, r[[e, 2]] - cz];
            r2[e] = t[e][0] * t[e][0] + t[e][1] * t[e][1] + t[e][2] * t[e][2];
            alive[e] = sh.pz_deriv * r2[e] <= lepsbas;
        }
        for (q, &z) in sh.prim_z.iter().enumerate() {
            for e in 0..n_e {
                ex[e] = (-z * r2[e]).exp();
                for (a, &t_ea) in t[e].iter().enumerate() {
                    pw[0] = 1.0;
                    for m in 1..l + 4 {
                        pw[m] = pw[m - 1] * t_ea;
                    }
                    let p = |m: usize, back: usize| if m < back { 0.0 } else { pw[m - back] };
                    let base = (a * n_e + e) * stride;
                    for n in 0..=l {
                        let nf = n as f64;
                        let tb = &mut tab[base + 4 * n..base + 4 * n + 4];
                        tb[0] = pw[n];
                        tb[1] = nf * p(n, 1) - 2.0 * z * pw[n + 1];
                        tb[2] = nf * (nf - 1.0) * p(n, 2) - 2.0 * z * (2.0 * nf + 1.0) * pw[n]
                            + 4.0 * z * z * pw[n + 2];
                        tb[3] = nf * (nf - 1.0) * (nf - 2.0) * p(n, 3)
                            - 6.0 * z * nf * nf * p(n, 1)
                            + 12.0 * z * z * (nf + 1.0) * pw[n + 1]
                            - 8.0 * z * z * z * pw[n + 3];
                    }
                }
            }
            for &ao in &sh.aos {
                let nc = aos.flat_nc[aos.ao_prim_offset[ao] + q];
                let (nx, ny, nz) = (
                    4 * aos.poly_x[ao] as usize,
                    4 * aos.poly_y[ao] as usize,
                    4 * aos.poly_z[ao] as usize,
                );
                for e in 0..n_e {
                    let rad = nc * ex[e];
                    let xs = &tab[e * stride + nx..e * stride + nx + 4];
                    let ys = &tab[(n_e + e) * stride + ny..(n_e + e) * stride + ny + 4];
                    let zs = &tab[(2 * n_e + e) * stride + nz..(2 * n_e + e) * stride + nz + 4];
                    a100[[ao, e]] += rad * xs[1] * ys[0] * zs[0];
                    a010[[ao, e]] += rad * xs[0] * ys[1] * zs[0];
                    a001[[ao, e]] += rad * xs[0] * ys[0] * zs[1];
                    if !alive[e] {
                        continue;
                    }
                    a200[[ao, e]] += rad * xs[2] * ys[0] * zs[0];
                    a020[[ao, e]] += rad * xs[0] * ys[2] * zs[0];
                    a002[[ao, e]] += rad * xs[0] * ys[0] * zs[2];
                    a110[[ao, e]] += rad * xs[1] * ys[1] * zs[0];
                    a101[[ao, e]] += rad * xs[1] * ys[0] * zs[1];
                    a011[[ao, e]] += rad * xs[0] * ys[1] * zs[1];
                    a300[[ao, e]] += rad * xs[3] * ys[0] * zs[0];
                    a030[[ao, e]] += rad * xs[0] * ys[3] * zs[0];
                    a003[[ao, e]] += rad * xs[0] * ys[0] * zs[3];
                    a120[[ao, e]] += rad * xs[1] * ys[2] * zs[0];
                    a102[[ao, e]] += rad * xs[1] * ys[0] * zs[2];
                    a210[[ao, e]] += rad * xs[2] * ys[1] * zs[0];
                    a012[[ao, e]] += rad * xs[0] * ys[1] * zs[2];
                    a201[[ao, e]] += rad * xs[2] * ys[0] * zs[1];
                    a021[[ao, e]] += rad * xs[0] * ys[2] * zs[1];
                }
            }
        }
    }
    let mut laplacian = a200.clone();
    laplacian += &a020;
    laplacian += &a002;
    let mut gx = a300;
    gx += &a120;
    gx += &a102;
    let mut gy = a210;
    gy += &a030;
    gy += &a012;
    let mut gz = a201;
    gz += &a021;
    gz += &a003;
    AoDerivBundle {
        grad: [a100, a010, a001],
        gradlap: [gx, gy, gz],
        hessian: [
            [a200, a110.clone(), a101.clone()],
            [a110, a020, a011.clone()],
            [a101, a011, a002],
        ],
        laplacian,
    }
}

/// `exp(-x)` is exactly `0.0` in f64 for `x > 745.14` (below half the smallest subnormal);
/// a primitive/electron pair past this argument contributes exactly nothing.
const EXP_UNDERFLOW_ARG: f64 = 746.0;

/// Runtime AO screen of the native FAST e_L, applied to a derivative table so the table
/// is the exact derivative of the screened AO the energy uses (a screened-out shell image
/// contributes zero to the value AND to every derivative of it). The criteria are the
/// e_L's own, keyed by `NATIVE_AO_SCREEN_EXPONENT` (`LEPSBAS`) and the per-shell exponents
/// built from `FAST_AO_SCREEN_ABS_BOUND`:
/// * [`AoScreen::ShellValue`]: `Shell::pz_val * r^2 > LEPSBAS` drops the shell image --
///   the periodic FAST orbital value of the sweep (`compute_aos_single_fold_cutoff_into`);
/// * [`AoScreen::MeshValue`]: the shell screen plus the per-primitive
///   `Shell::prim_val[q] * r^2 > LEPSBAS` -- the ECP non-local mesh value
///   (`dot_aos_multi_cutoff_cart`);
/// * [`AoScreen::Deriv`]: `Shell::pz_deriv * r^2 > LEPSBAS` -- the FAST kinetic AO
///   gradient / Laplacian (`compute_aos_single_value_grad_lap_fast_cut_folded_into`).
///
/// Only the shell-major sweep (finalized leaves) applies a screen; a leaf without the
/// shell table is evaluated unscreened, as by the e_L.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AoScreen {
    None,
    ShellValue,
    MeshValue,
    Deriv,
}

/// Several kernel-rank cart-GTO derivative orders (total order <= 3), output at the
/// image-tiled kernel rank `(num_ao_kernel, N)` (for pgto leaves; the centers in
/// `aos.centers` already carry `R_I + T_n`), in ONE sweep over the primitive table, with
/// one [`AoScreen`] per requested order. Each primitive/electron pair pays the radial
/// `exp` once for all `orders`, instead of once per order (the image-tiled periodic table
/// has `n_image` times the molecular primitive count, so these repeated sweeps dominated
/// the periodic force/stress kernels); the shell-major sweep agrees with the
/// per-primitive sweep up to floating-point reassociation.
pub(crate) fn ao_deriv_kernel_multi_screened(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    orders: &[(u32, u32, u32)],
    screens: &[AoScreen],
    // `true`: accumulate each kernel AO straight into its home row (`kernel_to_home`), i.e.
    // return the image-folded `(num_ao, N)` tables (the fold of the kernel-rank output, up to
    // floating-point reassociation) without materialising the kernel rank.
    fold: bool,
) -> Vec<Array2<f64>> {
    use crate::atomic_orbital::LEPSBAS;
    debug_assert_eq!(orders.len(), screens.len());
    let n_e = r.nrows();
    let fold = fold && !aos.kernel_to_home.is_empty();
    let n_k = if fold { aos.num_ao } else { aos.num_ao_kernel };
    let out_row = |ao: usize| if fold { aos.kernel_to_home[ao] } else { ao };
    let mut outs: Vec<Vec<f64>> = orders.iter().map(|_| vec![0.0; n_k * n_e]).collect();
    let kx = orders.iter().map(|o| o.0).max().unwrap_or(0);
    let ky = orders.iter().map(|o| o.1).max().unwrap_or(0);
    let kz = orders.iter().map(|o| o.2).max().unwrap_or(0);
    let rr: Vec<[f64; 3]> = (0..n_e)
        .map(|e| [r[[e, 0]], r[[e, 1]], r[[e, 2]]])
        .collect();
    let (mut px, mut py, mut pz) = ([0.0_f64; 4], [0.0_f64; 4], [0.0_f64; 4]);
    if aos.shells.is_empty() {
        // Per-primitive sweep (leaves built without the shell table).
        for p in 0..aos.exponents.len() {
            let z = aos.exponents[p];
            let nc = aos.normalizations[p] * aos.coefficients[p];
            let ao = aos.orbital_indices[p]; // kernel-AO
            let nx = aos.poly_x[ao];
            let ny = aos.poly_y[ao];
            let nz = aos.poly_z[ao];
            let (cx, cy, cz) = (
                aos.centers[[ao, 0]],
                aos.centers[[ao, 1]],
                aos.centers[[ao, 2]],
            );
            let row = out_row(ao) * n_e;
            for (e, re) in rr.iter().enumerate() {
                let x = re[0] - cx;
                let y = re[1] - cy;
                let zc = re[2] - cz;
                let r2 = x * x + y * y + zc * zc;
                if z * r2 > EXP_UNDERFLOW_ARG {
                    // `exp(-z r2)` is exactly 0.0: the pair adds exactly nothing to any order.
                    continue;
                }
                let rad = nc * (-z * r2).exp();
                for k in 0..=kx {
                    px[k as usize] = p1d(x, nx, z, k);
                }
                for k in 0..=ky {
                    py[k as usize] = p1d(y, ny, z, k);
                }
                for k in 0..=kz {
                    pz[k as usize] = p1d(zc, nz, z, k);
                }
                for (o, out) in orders.iter().zip(outs.iter_mut()) {
                    out[row + e] += rad * px[o.0 as usize] * py[o.1 as usize] * pz[o.2 as usize];
                }
            }
        }
    } else {
        // Shell-major sweep. Every AO of a shell has the same center and the same primitive
        // exponents in the same order (the shell key), so `exp(-z r^2)` is evaluated once
        // per (shell, primitive, electron). A derivative of order (i, j, k) of
        // `x^a y^b z^c exp(-z r^2)` is `exp(-z r^2)` times a polynomial of degree i + j + k in
        // the exponent (`hermite_kappa`), so each AO forms the radial moments
        // `Sum_q N_q c_q z_q^m exp(-z_q r^2)` (m <= 3) once and assembles every order from them.
        let mut ez: Vec<f64> = Vec::new();
        let mut on = vec![true; orders.len()];
        // Per (shell image, electron): the 1D Hermite coefficients of every axis power `n` of the
        // shell (each AO reads its axis tables from these instead of rebuilding them).
        const NPOW: usize = 8;
        let mut hcx = [[[0.0_f64; 4]; 4]; NPOW];
        let mut hcy = [[[0.0_f64; 4]; 4]; NPOW];
        let mut hcz = [[[0.0_f64; 4]; 4]; NPOW];
        let n_poly = aos
            .poly_x
            .iter()
            .chain(aos.poly_y.iter())
            .chain(aos.poly_z.iter())
            .copied()
            .max()
            .unwrap_or(0)
            .max(0);
        let kap: Vec<[[f64; 4]; 4]> = (0..=n_poly.max(NPOW as i64 - 1))
            .map(hermite_kappa)
            .collect();
        let mmax = orders
            .iter()
            .map(|o| (o.0 + o.1 + o.2) as usize)
            .max()
            .unwrap_or(0);
        assert!(
            mmax <= 3,
            "ao_deriv_kernel_multi_screened: total order {mmax} > 3 unsupported"
        );
        // The per-primitive mesh screen enters the shared radial moments, so it applies to every
        // order of a call or to none (callers pass `DerivScreens::MESH` for all orders).
        let any_mesh = screens.contains(&AoScreen::MeshValue);
        let all_mesh = screens.iter().all(|&sc| sc == AoScreen::MeshValue);
        assert!(
            !any_mesh || all_mesh,
            "AoScreen::MeshValue must cover every order of a call"
        );
        for sh in &aos.shells {
            let [cx, cy, cz] = sh.center;
            let nq = sh.prim_z.len();
            ez.resize(nq, 0.0);
            let lsh = sh
                .aos
                .iter()
                .map(|&ao| aos.poly_x[ao].max(aos.poly_y[ao]).max(aos.poly_z[ao]))
                .max()
                .unwrap_or(0)
                .max(0) as usize;
            assert!(
                lsh < NPOW,
                "ao_deriv_kernel_multi_screened: axis power {lsh} >= {NPOW} unsupported"
            );
            for (e, re) in rr.iter().enumerate() {
                let x = re[0] - cx;
                let y = re[1] - cy;
                let zc = re[2] - cz;
                let r2 = x * x + y * y + zc * zc;
                // e_L screen flags of this (shell image, electron).
                let val_alive = sh.pz_val * r2 <= LEPSBAS;
                let der_alive = sh.pz_deriv * r2 <= LEPSBAS;
                for (o, sc) in screens.iter().enumerate() {
                    on[o] = match sc {
                        AoScreen::None => true,
                        AoScreen::ShellValue | AoScreen::MeshValue => val_alive,
                        AoScreen::Deriv => der_alive,
                    };
                }
                if !on.iter().any(|&b| b) {
                    continue;
                }
                // Every order on the mesh screen: a primitive past its `prim_val` cut adds
                // nothing to any order, so it is skipped before its exp.
                let mut alive = false;
                for (q, ez_q) in ez[..nq].iter_mut().enumerate() {
                    let z = sh.prim_z[q];
                    let prim_dead =
                        all_mesh && sh.prim_val.get(q).is_some_and(|&pv| pv * r2 > LEPSBAS);
                    if prim_dead || z * r2 > EXP_UNDERFLOW_ARG {
                        *ez_q = f64::NAN; // marker: exactly-zero pair, skipped below
                    } else {
                        *ez_q = crate::fastmath::exp_neg(-z * r2);
                        alive = true;
                    }
                }
                if !alive {
                    continue;
                }
                // powers t^p, p <= lsh + 3, and the Hermite tables of every n <= lsh.
                let ptab = |t: f64| -> [f64; NPOW + 3] {
                    let mut p = [0.0_f64; NPOW + 3];
                    let mut v = 1.0_f64;
                    for slot in p.iter_mut().take(lsh + 4) {
                        *slot = v;
                        v *= t;
                    }
                    p
                };
                let (px, py, pz) = (ptab(x), ptab(y), ptab(zc));
                let fill = |pt: &[f64; NPOW + 3], kmax: u32, hc: &mut [[[f64; 4]; 4]; NPOW]| {
                    for (n, h) in hc.iter_mut().enumerate().take(lsh + 1) {
                        let kap_n = &kap[n];
                        for k in 0..=kmax as usize {
                            for m in 0..=k {
                                let pw = n as i64 - k as i64 + 2 * m as i64;
                                h[k][m] = if pw < 0 {
                                    0.0
                                } else {
                                    kap_n[k][m] * pt[pw as usize]
                                };
                            }
                        }
                    }
                };
                fill(&px, kx, &mut hcx);
                fill(&py, ky, &mut hcy);
                fill(&pz, kz, &mut hcz);
                for &ao in &sh.aos {
                    let (nx, ny, nz) = (aos.poly_x[ao], aos.poly_y[ao], aos.poly_z[ao]);
                    // Radial moments `S[m] = Sum_q N_q c_q z_q^m exp(-z_q r^2)` over the live
                    // primitives: every requested derivative is a polynomial of degree <= 3 in
                    // the exponent, so the primitive sum is formed once for all orders.
                    let off = aos.ao_prim_offset[ao];
                    let mut sm = [0.0_f64; 4];
                    for (q, ez_q) in ez[..nq].iter().enumerate() {
                        if ez_q.is_nan() {
                            continue;
                        }
                        let z = sh.prim_z[q];
                        let mut t = aos.flat_nc[off + q] * *ez_q;
                        for v in sm.iter_mut().take(mmax + 1) {
                            *v += t;
                            t *= z;
                        }
                    }
                    let (hx, hy, hz) = (&hcx[nx as usize], &hcy[ny as usize], &hcz[nz as usize]);
                    let row = out_row(ao) * n_e;
                    for (oi, (o, out)) in orders.iter().zip(outs.iter_mut()).enumerate() {
                        if !on[oi] {
                            continue;
                        }
                        let (ox, oy, oz) = (o.0 as usize, o.1 as usize, o.2 as usize);
                        let mut acc = 0.0;
                        for mx in 0..=ox {
                            for my in 0..=oy {
                                let cxy = hx[ox][mx] * hy[oy][my];
                                for mz in 0..=oz {
                                    acc += cxy * hz[oz][mz] * sm[mx + my + mz];
                                }
                            }
                        }
                        out[row + e] += acc;
                    }
                }
            }
        }
    }
    outs.into_iter()
        .map(|v| Array2::from_shape_vec((n_k, n_e), v).unwrap())
        .collect()
}

/// `kappa(n,k,m)` of `d^k/dt^k [t^n exp(-z t^2)] = exp(-z t^2) Sum_m kappa(n,k,m) z^m
/// t^(n-k+2m)` for `k, m <= 3`: `kappa(n,0,0) = 1`, `kappa(n,k+1,m) = (n-k+2m) kappa(n,k,m) -
/// 2 kappa(n,k,m-1)` ([`p1d`] collects the same terms at a given `z`).
fn hermite_kappa(n: i64) -> [[f64; 4]; 4] {
    let mut kap = [[0.0_f64; 4]; 4];
    kap[0][0] = 1.0;
    for k in 0..3 {
        for m in 0..=k + 1 {
            let keep = if m <= k {
                kap[k][m] * (n - k as i64 + 2 * m as i64) as f64
            } else {
                0.0
            };
            let raise = if m >= 1 { -2.0 * kap[k][m - 1] } else { 0.0 };
            kap[k + 1][m] = keep + raise;
        }
    }
    kap
}

/// Fold a kernel-rank `(num_ao_kernel, N)` table onto the home rank `(num_ao, N)`
/// via `kernel_to_home` (real Gamma: Bloch phase 1).
fn fold_kernel(aos: &AosGtoData, raw: &Array2<f64>) -> Array2<f64> {
    let n_e = raw.ncols();
    let mut out = Array2::<f64>::zeros((aos.num_ao, n_e));
    for k in 0..aos.num_ao_kernel {
        let home = aos.kernel_to_home[k];
        for e in 0..n_e {
            out[[home, e]] += raw[[k, e]];
        }
    }
    out
}

/// Periodic (pgto) cart-GTO derivative `(num_ao, N)` for a REAL Gamma-point leaf:
/// wrap electrons into the home cell, evaluate the kernel-rank derivative on the
/// image-tiled AOs, then fold the image axis via `kernel_to_home` (Bloch phase =
/// 1 at Gamma). Complex/twisted leaves are NOT supported (Rust is all-f64).
pub fn pgto_deriv(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    order: (u32, u32, u32),
) -> Array2<f64> {
    pgto_deriv_multi(aos, r, cell, &[order]).pop().unwrap()
}

/// [`pgto_deriv`] for several orders from one fused kernel sweep (each output
/// bit-identical to the single-order call).
fn pgto_deriv_multi(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    orders: &[(u32, u32, u32)],
) -> Vec<Array2<f64>> {
    let none = vec![AoScreen::None; orders.len()];
    pgto_deriv_multi_screened(aos, r, cell, orders, &none)
}

fn pgto_deriv_multi_screened(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    orders: &[(u32, u32, u32)],
    screens: &[AoScreen],
) -> Vec<Array2<f64>> {
    let inv = crate::ewald::inv3(cell);
    let r_w = crate::mcmc::wrap_rows(r, cell, &inv);
    ao_deriv_kernel_multi_screened(aos, &r_w.view(), orders, screens, true)
}

/// Which e_L screen a derivative table follows: `value` for the AO value and its first
/// derivatives (the derivative of the screened value), `high` for the second and third
/// derivatives (the derivatives of the screened gradient / Laplacian).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DerivScreens {
    pub value: AoScreen,
    pub high: AoScreen,
}

impl DerivScreens {
    /// Unscreened static full-image tiling (the exact / XLA convention).
    pub(crate) const EXACT: DerivScreens = DerivScreens {
        value: AoScreen::None,
        high: AoScreen::None,
    };
    /// The FAST periodic e_L wavefunction: shell-screened orbital value
    /// (`pz_val`), and kinetic gradient / Laplacian screened with `pz_deriv`.
    pub(crate) const FAST: DerivScreens = DerivScreens {
        value: AoScreen::ShellValue,
        high: AoScreen::Deriv,
    };
    /// The ECP non-local mesh orbital value (shell + per-primitive `pz_val` screens).
    pub(crate) const MESH: DerivScreens = DerivScreens {
        value: AoScreen::MeshValue,
        high: AoScreen::MeshValue,
    };
}

/// [`pgto_grad`] under one [`AoScreen`].
pub(crate) fn pgto_grad_screened(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    screen: AoScreen,
) -> [Array2<f64>; 3] {
    take3(&mut pgto_deriv_multi_screened(aos, r, cell, &GRAD_ORDERS, &[screen; 3]).into_iter())
}

const GRAD_ORDERS: [(u32, u32, u32); 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];
/// Hessian orders `00,11,22,01,02,12`.
const HESS_ORDERS: [(u32, u32, u32); 6] = [
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
];
/// grad-of-Laplacian orders, `[x: 300,120,102 | y: 210,030,012 | z: 201,021,003]`.
const GRADLAP_ORDERS: [(u32, u32, u32); 9] = [
    (3, 0, 0),
    (1, 2, 0),
    (1, 0, 2),
    (2, 1, 0),
    (0, 3, 0),
    (0, 1, 2),
    (2, 0, 1),
    (0, 2, 1),
    (0, 0, 3),
];

fn take3(v: &mut std::vec::IntoIter<Array2<f64>>) -> [Array2<f64>; 3] {
    [v.next().unwrap(), v.next().unwrap(), v.next().unwrap()]
}

/// `[H00,H11,H22,H01,H02,H12]` -> symmetric `H[i][j]`.
fn hess_from6(v: &mut std::vec::IntoIter<Array2<f64>>) -> [[Array2<f64>; 3]; 3] {
    let (h00, h11, h22) = (v.next().unwrap(), v.next().unwrap(), v.next().unwrap());
    let (h01, h02, h12) = (v.next().unwrap(), v.next().unwrap(), v.next().unwrap());
    [
        [h00, h01.clone(), h02.clone()],
        [h01, h11, h12.clone()],
        [h02, h12, h22],
    ]
}

/// The nine third-order tables -> `[dx grad^2, dy grad^2, dz grad^2]` (the `+=` order is
/// that of the per-order sums).
fn gradlap_from9(v: &mut std::vec::IntoIter<Array2<f64>>) -> [Array2<f64>; 3] {
    let mut gx = v.next().unwrap();
    gx += &v.next().unwrap();
    gx += &v.next().unwrap();
    let mut gy = v.next().unwrap();
    gy += &v.next().unwrap();
    gy += &v.next().unwrap();
    let mut gz = v.next().unwrap();
    gz += &v.next().unwrap();
    gz += &v.next().unwrap();
    [gx, gy, gz]
}

/// Periodic `gradAO` `[dx,dy,dz]`, each `(num_ao, N)` (real Gamma pgto).
pub fn pgto_grad(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> [Array2<f64>; 3] {
    take3(&mut pgto_deriv_multi(aos, r, cell, &GRAD_ORDERS).into_iter())
}

/// Periodic `grad^2AO`, `(num_ao, N)` (real Gamma pgto).
pub fn pgto_laplacian(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> Array2<f64> {
    let mut v = pgto_deriv_multi(aos, r, cell, &HESS_ORDERS[0..3]).into_iter();
    let mut o = v.next().unwrap();
    o += &v.next().unwrap();
    o += &v.next().unwrap();
    o
}

/// Periodic `grad(grad^2AO)` `[dxgrad^2,dygrad^2,dzgrad^2]`, each `(num_ao, N)` (real Gamma pgto).
pub fn pgto_gradlap(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> [Array2<f64>; 3] {
    gradlap_from9(&mut pgto_deriv_multi(aos, r, cell, &GRADLAP_ORDERS).into_iter())
}

/// Periodic symmetric AO Hessian `H[i][j]`, each `(num_ao, N)` (real Gamma pgto).
pub fn pgto_hessian(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
) -> [[Array2<f64>; 3]; 3] {
    hess_from6(&mut pgto_deriv_multi(aos, r, cell, &HESS_ORDERS).into_iter())
}

/// Periodic (folded, real Gamma) cart-pgto derivative tables of one leaf at one
/// configuration -- gradient, plus on request value, Hessian + Laplacian and
/// gradient-of-Laplacian -- from ONE fused kernel sweep. Every field is bit-identical
/// to the matching stand-alone call ([`pgto_deriv`] `(0,0,0)`, [`pgto_grad`],
/// [`pgto_hessian`], [`pgto_laplacian`], [`pgto_gradlap`]); a kernel that needs
/// several of them at the same electrons pays one sweep instead of up to 22.
pub struct PgtoDerivs {
    pub val: Option<Array2<f64>>,
    pub grad: [Array2<f64>; 3],
    pub hess: Option<[[Array2<f64>; 3]; 3]>,
    pub lap: Option<Array2<f64>>,
    pub gradlap: Option<[Array2<f64>; 3]>,
}

/// Build [`PgtoDerivs`]: `val` adds the value, `second` the Hessian and Laplacian,
/// `third` the gradient-of-Laplacian.
pub fn pgto_derivs(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    val: bool,
    second: bool,
    third: bool,
) -> PgtoDerivs {
    pgto_derivs_screened(aos, r, cell, val, second, third, DerivScreens::EXACT)
}

/// [`pgto_derivs`] under the e_L screens `scr` (see [`DerivScreens`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pgto_derivs_screened(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    val: bool,
    second: bool,
    third: bool,
    scr: DerivScreens,
) -> PgtoDerivs {
    let mut orders: Vec<(u32, u32, u32)> = Vec::with_capacity(19);
    let mut screens: Vec<AoScreen> = Vec::with_capacity(19);
    if val {
        orders.push((0, 0, 0));
        screens.push(scr.value);
    }
    orders.extend_from_slice(&GRAD_ORDERS);
    screens.extend([scr.value; 3]);
    if second {
        orders.extend_from_slice(&HESS_ORDERS);
        screens.extend([scr.high; 6]);
    }
    if third {
        orders.extend_from_slice(&GRADLAP_ORDERS);
        screens.extend([scr.high; 9]);
    }
    let mut v = pgto_deriv_multi_screened(aos, r, cell, &orders, &screens).into_iter();
    let val = if val { v.next() } else { None };
    let grad = take3(&mut v);
    let hess = if second {
        Some(hess_from6(&mut v))
    } else {
        None
    };
    let lap = hess.as_ref().map(|h| {
        let mut lap = h[0][0].clone();
        lap += &h[1][1];
        lap += &h[2][2];
        lap
    });
    let gradlap = if third {
        Some(gradlap_from9(&mut v))
    } else {
        None
    };
    PgtoDerivs {
        val,
        grad,
        hess,
        lap,
        gradlap,
    }
}

/// Kernel-rank (image-tiled, unfolded) strain-stress tables of one pgto leaf at the
/// WRAPPED electrons `r_w` together with their folds, from one fused sweep:
/// gradient (`gk`/`gf`), and on request Hessian (`hk`/`hf`) and gradient-of-Laplacian
/// (`glk`/`glf`). `field_stress` consumes both ranks. Each `*f` fold is bit-identical
/// to [`pgto_grad`] / [`pgto_hessian`] / [`pgto_gradlap`] at the un-wrapped electrons
/// (those wrap internally with the same `wrap_rows`), each `*k` to the unscreened
/// kernel-rank tables of [`ao_deriv_kernel_multi_screened`].
pub struct PgtoStrainTables {
    /// folded value (only from [`pgto_strain_tables_with_value`]).
    pub vf: Option<Array2<f64>>,
    pub gk: [Array2<f64>; 3],
    pub gf: [Array2<f64>; 3],
    pub hk: Option<[[Array2<f64>; 3]; 3]>,
    pub hf: Option<[[Array2<f64>; 3]; 3]>,
    pub glk: Option<[Array2<f64>; 3]>,
    pub glf: Option<[Array2<f64>; 3]>,
}

/// Build [`PgtoStrainTables`] at the wrapped electrons `r_w`.
pub(crate) fn pgto_strain_tables(
    aos: &AosGtoData,
    r_w: &ArrayView2<f64>,
    second: bool,
    third: bool,
    scr: DerivScreens,
) -> PgtoStrainTables {
    strain_tables_impl(aos, r_w, false, second, third, scr)
}

/// [`pgto_strain_tables`] (gradient only) plus the folded AO value from the same sweep.
pub(crate) fn pgto_strain_tables_with_value(
    aos: &AosGtoData,
    r_w: &ArrayView2<f64>,
    scr: DerivScreens,
) -> PgtoStrainTables {
    strain_tables_impl(aos, r_w, true, false, false, scr)
}

fn strain_tables_impl(
    aos: &AosGtoData,
    r_w: &ArrayView2<f64>,
    val: bool,
    second: bool,
    third: bool,
    scr: DerivScreens,
) -> PgtoStrainTables {
    let mut orders: Vec<(u32, u32, u32)> = Vec::with_capacity(19);
    let mut screens: Vec<AoScreen> = Vec::with_capacity(19);
    if val {
        orders.push((0, 0, 0));
        screens.push(scr.value);
    }
    orders.extend_from_slice(&GRAD_ORDERS);
    screens.extend([scr.value; 3]);
    if second {
        orders.extend_from_slice(&HESS_ORDERS);
        screens.extend([scr.high; 6]);
    }
    if third {
        orders.extend_from_slice(&GRADLAP_ORDERS);
        screens.extend([scr.high; 9]);
    }
    let raw = ao_deriv_kernel_multi_screened(aos, r_w, &orders, &screens, false);
    let folded: Vec<Array2<f64>> = raw.iter().map(|k| fold_kernel(aos, k)).collect();
    let mut rk = raw.into_iter();
    let mut rf = folded.into_iter();
    let vf = if val {
        rk.next();
        rf.next()
    } else {
        None
    };
    let (gk, gf) = (take3(&mut rk), take3(&mut rf));
    let (hk, hf) = if second {
        (Some(hess_from6(&mut rk)), Some(hess_from6(&mut rf)))
    } else {
        (None, None)
    };
    let (glk, glf) = if third {
        (Some(gradlap_from9(&mut rk)), Some(gradlap_from9(&mut rf)))
    } else {
        (None, None)
    };
    PgtoStrainTables {
        vf,
        gk,
        gf,
        hk,
        hf,
        glk,
        glf,
    }
}

/// `gradAO`: `[dx, dy, dz]`, each `(num_ao, N)`.
pub fn ao_grad(aos: &AosGtoData, r: &ArrayView2<f64>) -> [Array2<f64>; 3] {
    [
        ao_deriv(aos, r, (1, 0, 0)),
        ao_deriv(aos, r, (0, 1, 0)),
        ao_deriv(aos, r, (0, 0, 1)),
    ]
}

/// Fused `gradAO` `[dx, dy, dz]` -- one primitive sweep sharing the single `exp`
/// and the 1D Hermite polynomials across the three axes. Bit-for-bit identical to
/// [`ao_grad`] (same primitive-outer/electron-inner accumulation, same product
/// association) but pays the radial `exp` once per (primitive, electron) instead
/// of three times. Molecular cart leaves only.
pub fn ao_grad_fused(aos: &AosGtoData, r: &ArrayView2<f64>) -> [Array2<f64>; 3] {
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let mut dx = Array2::<f64>::zeros((num_ao, n_e));
    let mut dy = Array2::<f64>::zeros((num_ao, n_e));
    let mut dz = Array2::<f64>::zeros((num_ao, n_e));
    for p in 0..aos.exponents.len() {
        let z = aos.exponents[p];
        let nc = aos.normalizations[p] * aos.coefficients[p];
        let ao = aos.orbital_indices[p];
        let nx = aos.poly_x[ao];
        let ny = aos.poly_y[ao];
        let nz = aos.poly_z[ao];
        let (cx, cy, cz) = (
            aos.centers[[ao, 0]],
            aos.centers[[ao, 1]],
            aos.centers[[ao, 2]],
        );
        for e in 0..n_e {
            let x = r[[e, 0]] - cx;
            let y = r[[e, 1]] - cy;
            let zc = r[[e, 2]] - cz;
            let r2 = x * x + y * y + zc * zc;
            let rad = nc * (-z * r2).exp();
            let (px0, px1) = (p1d(x, nx, z, 0), p1d(x, nx, z, 1));
            let (py0, py1) = (p1d(y, ny, z, 0), p1d(y, ny, z, 1));
            let (pz0, pz1) = (p1d(zc, nz, z, 0), p1d(zc, nz, z, 1));
            dx[[ao, e]] += rad * px1 * py0 * pz0;
            dy[[ao, e]] += rad * px0 * py1 * pz0;
            dz[[ao, e]] += rad * px0 * py0 * pz1;
        }
    }
    [dx, dy, dz]
}

/// AO values and gradients ([`MeshAoTables`]: only the rows of shells kept at some
/// point are stored) on the FAST ECP-mesh VALUE screen of [`crate::atomic_orbital::dot_aos_multi_cutoff_cart`]:
/// at point `r_e`, shell `s` contributes only if `pz_val * r^2 <= LEPSBAS`, and
/// inside a kept shell primitive `q` only if `prim_val[q] * r^2 <= LEPSBAS`
/// ([`crate::atomic_orbital::Shell::prim_val`]); every other entry is an exact zero.
/// `prim_cut = false` keeps every primitive of a kept shell: the shell-only
/// `pz_val` screen of the GFMC molecular candidate batch
/// (`gfmc::molecular_aos_candidate_batch_t`), whose e_L the GFMC force differentiates.
/// The values are the screened AO model whose contraction the FAST e_L evaluates
/// at the ECP mesh points, and the gradients are its exact analytic derivative
/// wherever the keep set is locally constant. Kept entries equal the
/// [`ao_grad_fused`] values to fp rounding (power tables instead of `powi`, two
/// radial sums per AO instead of per-primitive products, and the FAST e_L's
/// `fastmath::exp_neg` instead of libm `exp`). Molecular Cartesian leaves only.
pub fn ao_val_grad_mesh_screened(
    aos: &AosGtoData,
    r: &ArrayView2<f64>,
    prim_cut: bool,
) -> MeshAoTables {
    debug_assert!(aos.n_image == 1 && aos.kind == crate::atomic_orbital::AoKind::Cart);
    let lepsbas = crate::atomic_orbital::LEPSBAS;
    let n_e = r.nrows();
    let num_ao = aos.num_ao;
    let lmax = aos
        .shells
        .iter()
        .map(|s| s.l.max(0) as usize)
        .max()
        .unwrap_or(0);
    let pmax = aos.shells.iter().map(|s| s.prim_z.len()).max().unwrap_or(0);
    let alive = |shell: &crate::atomic_orbital::Shell, e: usize| {
        let [cx, cy, cz] = shell.center;
        let t = [r[[e, 0]] - cx, r[[e, 1]] - cy, r[[e, 2]] - cz];
        (shell.pz_val * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2])).partial_cmp(&lepsbas)
            != Some(std::cmp::Ordering::Greater)
    };
    // Rows: the AOs of the shells kept at some point, ascending; `row_of[ao]`.
    let mut row_of = vec![usize::MAX; num_ao];
    for shell in &aos.shells {
        if (0..n_e).any(|e| alive(shell, e)) {
            for &ao in &shell.aos {
                row_of[ao] = 0;
            }
        }
    }
    let rows: Vec<usize> = (0..num_ao).filter(|&ao| row_of[ao] == 0).collect();
    for (k, &ao) in rows.iter().enumerate() {
        row_of[ao] = k;
    }
    let n_rows = rows.len();
    let (mut val, mut dx, mut dy, mut dz) = (
        vec![0.0_f64; n_rows * n_e],
        vec![0.0_f64; n_rows * n_e],
        vec![0.0_f64; n_rows * n_e],
        vec![0.0_f64; n_rows * n_e],
    );
    // d/dt (t^n e^{-Z r^2}) = (n t^(n-1) - 2 Z t^(n+1)) e^{-Z r^2}, so per AO
    // the value and gradient need only the two radial sums
    // R0 = Sum_q nc_q e_q and R1 = Sum_q nc_q Z_q e_q over the kept primitives,
    // and the monomials pw[a][m] = t_a^m (m <= l+1).
    let mut pw = [
        vec![0.0_f64; lmax + 2],
        vec![0.0_f64; lmax + 2],
        vec![0.0_f64; lmax + 2],
    ];
    // ex[q] = e_q, zex[q] = Z_q e_q; both 0 for a screened primitive.
    let mut ex = vec![0.0_f64; pmax];
    let mut zex = vec![0.0_f64; pmax];
    for shell in &aos.shells {
        let l = shell.l.max(0) as usize;
        let np = shell.prim_z.len();
        let [cx, cy, cz] = shell.center;
        for e in 0..n_e {
            let t = [r[[e, 0]] - cx, r[[e, 1]] - cy, r[[e, 2]] - cz];
            let r2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
            if shell.pz_val * r2 > lepsbas {
                continue;
            }
            for a in 0..3 {
                pw[a][0] = 1.0;
                for m in 1..l + 2 {
                    pw[a][m] = pw[a][m - 1] * t[a];
                }
            }
            for (q, &z) in shell.prim_z.iter().enumerate() {
                if prim_cut && shell.prim_val[q] * r2 > lepsbas {
                    ex[q] = 0.0;
                    zex[q] = 0.0;
                    continue;
                }
                // the FAST e_L mesh kernel's exp (`atomic_orbital::exp_pd_into` core)
                ex[q] = crate::fastmath::exp_neg(-z * r2);
                zex[q] = z * ex[q];
            }
            for &ao in &shell.aos {
                let nc = &aos.flat_nc[aos.ao_prim_offset[ao]..aos.ao_prim_offset[ao] + np];
                let (mut r0, mut r1) = (0.0_f64, 0.0_f64);
                for q in 0..np {
                    r0 += nc[q] * ex[q];
                    r1 += nc[q] * zex[q];
                }
                let n = [
                    aos.poly_x[ao] as usize,
                    aos.poly_y[ao] as usize,
                    aos.poly_z[ao] as usize,
                ];
                let p0 = [pw[0][n[0]], pw[1][n[1]], pw[2][n[2]]];
                // dP_a/dt_a with the radial factor: n t^(n-1) R0 - 2 t^(n+1) R1
                let d = |a: usize| {
                    let lower = if n[a] == 0 {
                        0.0
                    } else {
                        n[a] as f64 * pw[a][n[a] - 1]
                    };
                    lower * r0 - 2.0 * pw[a][n[a] + 1] * r1
                };
                let i = row_of[ao] * n_e + e;
                val[i] = r0 * p0[0] * p0[1] * p0[2];
                dx[i] = d(0) * p0[1] * p0[2];
                dy[i] = p0[0] * d(1) * p0[2];
                dz[i] = p0[0] * p0[1] * d(2);
            }
        }
    }
    let to2 = |v: Vec<f64>| Array2::from_shape_vec((n_rows, n_e), v).unwrap();
    let (val, dx, dy, dz) = (to2(val), to2(dx), to2(dy), to2(dz));
    MeshAoTables {
        rows,
        val,
        grad: [dx, dy, dz],
    }
}

/// AO values and gradients at a set of points on the rows `rows` (ascending AO
/// indices), each `(rows.len(), N)`; every AO not in `rows` is an exact zero at
/// every point ([`ao_val_grad_mesh_screened`]).
pub struct MeshAoTables {
    pub rows: Vec<usize>,
    pub val: Array2<f64>,
    pub grad: [Array2<f64>; 3],
}

impl MeshAoTables {
    /// The `(num_ao, N)` tables with the zero rows filled in.
    pub fn into_dense(self, num_ao: usize) -> (Array2<f64>, [Array2<f64>; 3]) {
        if self.rows.len() == num_ao {
            return (self.val, self.grad);
        }
        let n = self.val.ncols();
        let fill = |t: &Array2<f64>| {
            let mut o = Array2::<f64>::zeros((num_ao, n));
            for (k, &ao) in self.rows.iter().enumerate() {
                o.row_mut(ao).assign(&t.row(k));
            }
            o
        };
        (
            fill(&self.val),
            [
                fill(&self.grad[0]),
                fill(&self.grad[1]),
                fill(&self.grad[2]),
            ],
        )
    }
}

/// `grad^2AO`, `(num_ao, N)`.
pub fn ao_laplacian(aos: &AosGtoData, r: &ArrayView2<f64>) -> Array2<f64> {
    let mut o = ao_deriv(aos, r, (2, 0, 0));
    o += &ao_deriv(aos, r, (0, 2, 0));
    o += &ao_deriv(aos, r, (0, 0, 2));
    o
}

/// `grad(grad^2AO)`: `[dxgrad^2, dygrad^2, dzgrad^2]`, each `(num_ao, N)` (3rd order, kinetic force).
pub fn ao_gradlap(aos: &AosGtoData, r: &ArrayView2<f64>) -> [Array2<f64>; 3] {
    let mut gx = ao_deriv(aos, r, (3, 0, 0));
    gx += &ao_deriv(aos, r, (1, 2, 0));
    gx += &ao_deriv(aos, r, (1, 0, 2));
    let mut gy = ao_deriv(aos, r, (2, 1, 0));
    gy += &ao_deriv(aos, r, (0, 3, 0));
    gy += &ao_deriv(aos, r, (0, 1, 2));
    let mut gz = ao_deriv(aos, r, (2, 0, 1));
    gz += &ao_deriv(aos, r, (0, 2, 1));
    gz += &ao_deriv(aos, r, (0, 0, 3));
    [gx, gy, gz]
}

/// Symmetric AO Hessian `H[i][j]`, each `(num_ao, N)`.
pub fn ao_hessian(aos: &AosGtoData, r: &ArrayView2<f64>) -> [[Array2<f64>; 3]; 3] {
    let h00 = ao_deriv(aos, r, (2, 0, 0));
    let h11 = ao_deriv(aos, r, (0, 2, 0));
    let h22 = ao_deriv(aos, r, (0, 0, 2));
    let h01 = ao_deriv(aos, r, (1, 1, 0));
    let h02 = ao_deriv(aos, r, (1, 0, 1));
    let h12 = ao_deriv(aos, r, (0, 1, 1));
    [
        [h00, h01.clone(), h02.clone()],
        [h01, h11, h12.clone()],
        [h02, h12, h22],
    ]
}

#[cfg(test)]
mod mesh_screen_tests {
    use super::*;
    use crate::atomic_orbital::{dot_aos_multi_cutoff_cart, AoKind, LEPSBAS};

    /// The screened mesh AO model is the one the FAST e_L contracts, and its
    /// gradient is that model's derivative. Independent reference: the FAST e_L
    /// kernel itself (`dot_aos_multi_cutoff_cart`, a separate shell/SoA
    /// implementation) -- its `w`-contraction must equal `w . val`, and its
    /// centred finite difference must equal `w . grad` wherever the keep set is
    /// the same at both stencil points. Points span radii where tight shells and
    /// primitives switch off, so both kept and dropped entries are exercised.
    #[test]
    fn mesh_screened_val_grad_matches_fast_mesh_contraction() {
        let d = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 1, 0.0, true);
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let fd_atol = crate::central_setting_f64("ATOL_FD_VS_ANALYTIC_TIGHT");
        let fd_rtol = crate::central_setting_f64("RTOL_FD_VS_ANALYTIC_TIGHT");
        let h = 1.0e-5; // finite-difference input, not a tolerance
        let w: Vec<f64> = (0..d.num_ao)
            .map(|c| ((c * 37 % 11) as f64 - 5.0) / 7.0)
            .collect();
        // 12 points per radius on a fixed direction set around atom 0.
        let dirs: [[f64; 3]; 12] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.6, 0.8, 0.0],
            [0.3, -0.4, 0.866],
            [-0.5, -0.5, std::f64::consts::FRAC_1_SQRT_2],
            [0.8, 0.36, -0.48],
            [-0.2, 0.9, 0.3873],
            [0.0, -0.8, -0.6],
            [0.6, 0.0, -0.8],
            [-0.9, 0.3, -0.3162],
            [0.48, 0.64, 0.6],
        ];
        let c0 = [0.1, -0.2, 0.3];
        let keep = |p: [f64; 3]| -> Vec<bool> {
            let mut k = Vec::new();
            for sh in &d.shells {
                let r2 = (p[0] - sh.center[0]).powi(2)
                    + (p[1] - sh.center[1]).powi(2)
                    + (p[2] - sh.center[2]).powi(2);
                let alive = sh.pz_val * r2 <= LEPSBAS;
                for q in 0..sh.prim_z.len() {
                    k.push(alive && sh.prim_val[q] * r2 <= LEPSBAS);
                }
            }
            k
        };
        let (mut n_dropped, mut n_kept, mut n_fd) = (0usize, 0usize, 0usize);
        for &rad in &[0.15, 0.4, 0.8, 1.2, 1.7, 2.5, 3.5, 5.0, 7.0] {
            let nv = dirs.len();
            let mut pts = Array2::<f64>::zeros((nv, 3));
            for (v, u) in dirs.iter().enumerate() {
                let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                for k in 0..3 {
                    pts[[v, k]] = c0[k] + rad * u[k] / n;
                }
            }
            let (val, grad) = ao_val_grad_mesh_screened(&d, &pts.view(), true).into_dense(d.num_ao);
            let mut fast = vec![0.0_f64; nv];
            dot_aos_multi_cutoff_cart(&d, &pts.view(), &w, &mut fast);
            for v in 0..nv {
                let p = [pts[[v, 0]], pts[[v, 1]], pts[[v, 2]]];
                for k in keep(p) {
                    if k {
                        n_kept += 1;
                    } else {
                        n_dropped += 1;
                    }
                }
                let mine: f64 = (0..d.num_ao).map(|c| w[c] * val[[c, v]]).sum();
                assert!(
                    (mine - fast[v]).abs() <= atol + rtol * fast[v].abs(),
                    "value r={rad} v={v}: {mine} vs FAST {}",
                    fast[v]
                );
                for dd in 0..3 {
                    let (mut pp, mut pm) = (pts.clone(), pts.clone());
                    pp[[v, dd]] += h;
                    pm[[v, dd]] -= h;
                    let (prow, mrow) = (
                        [pp[[v, 0]], pp[[v, 1]], pp[[v, 2]]],
                        [pm[[v, 0]], pm[[v, 1]], pm[[v, 2]]],
                    );
                    if keep(prow) != keep(p) || keep(mrow) != keep(p) {
                        continue; // keep set changes inside the stencil
                    }
                    let (mut fp, mut fm) = (vec![0.0_f64; nv], vec![0.0_f64; nv]);
                    dot_aos_multi_cutoff_cart(&d, &pp.view(), &w, &mut fp);
                    dot_aos_multi_cutoff_cart(&d, &pm.view(), &w, &mut fm);
                    let fd = (fp[v] - fm[v]) / (2.0 * h);
                    let an: f64 = (0..d.num_ao).map(|c| w[c] * grad[dd][[c, v]]).sum();
                    assert!(
                        (fd - an).abs() <= fd_atol + fd_rtol * an.abs(),
                        "grad r={rad} v={v} d={dd}: FD {fd} vs {an}"
                    );
                    n_fd += 1;
                }
            }
        }
        assert!(
            n_dropped > 0 && n_kept > 0 && n_fd > 0,
            "screen not exercised: {n_dropped} {n_kept} {n_fd}"
        );
    }

    /// The FAST bundle is the derivative bundle of the FAST kinetic: its `grad`
    /// is the unscreened value derivative (reference: the exact `ao_grad_fused`),
    /// its Laplacian is the FAST kinetic's own screened Laplacian, and its Hessian
    /// / grad-Laplacian are the derivatives of the FAST kinetic's screened
    /// gradients / Laplacian. Independent reference: the FAST kinetic kernel
    /// `compute_aos_grad_lap_fast_cut` (shell-sweep implementation) and its
    /// centred finite difference wherever the keep set is the same at both
    /// stencil points.
    #[test]
    fn fast_cut_bundle_is_derivative_of_fast_kinetic_tables() {
        use crate::atomic_orbital::compute_aos_grad_lap_fast_cut;
        let d = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 1, 0.0, true);
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let fd_atol = crate::central_setting_f64("ATOL_FD_VS_ANALYTIC_TIGHT");
        let fd_rtol = crate::central_setting_f64("RTOL_FD_VS_ANALYTIC_TIGHT");
        let h = 1.0e-5; // finite-difference input, not a tolerance
        let close = |a: f64, b: f64, at: f64, rt: f64| (a - b).abs() <= at + rt * b.abs();
        let alive = |p: [f64; 3]| -> Vec<bool> {
            d.shells
                .iter()
                .map(|sh| {
                    let r2 = (p[0] - sh.center[0]).powi(2)
                        + (p[1] - sh.center[1]).powi(2)
                        + (p[2] - sh.center[2]).powi(2);
                    sh.pz_deriv * r2 <= LEPSBAS
                })
                .collect()
        };
        let mut pts_v = Vec::new();
        for &rad in &[0.2, 0.6, 1.1, 1.6, 2.3, 3.2, 4.5, 6.5] {
            for u in [[0.6, 0.8, 0.0], [0.0, -0.6, 0.8], [0.48, 0.6, -0.64]] {
                pts_v.push([0.1 + rad * u[0], -0.2 + rad * u[1], 0.3 + rad * u[2]]);
            }
        }
        let n = pts_v.len();
        let pts = Array2::from_shape_vec((n, 3), pts_v.concat()).unwrap();
        let b = ao_deriv_bundle_fast_cut(&d, &pts.view());
        let (_, _, _, lap) = compute_aos_grad_lap_fast_cut(&d, &pts.view());
        let g_ref = ao_grad_fused(&d, &pts.view());
        let (mut n_dead, mut n_fd) = (0usize, 0usize);
        for e in 0..n {
            let p = [pts[[e, 0]], pts[[e, 1]], pts[[e, 2]]];
            n_dead += alive(p).iter().filter(|&&a| !a).count();
            for c in 0..d.num_ao {
                for (k, g_ref_k) in g_ref.iter().enumerate() {
                    assert!(
                        close(b.grad[k][[c, e]], g_ref_k[[c, e]], atol, rtol),
                        "grad ao={c} e={e} d={k}"
                    );
                }
                assert!(
                    close(b.laplacian[[c, e]], lap[[c, e]], atol, rtol),
                    "lap ao={c} e={e}"
                );
            }
            for k in 0..3 {
                let (mut pp, mut pm) = (pts.clone(), pts.clone());
                pp[[e, k]] += h;
                pm[[e, k]] -= h;
                let prow = [pp[[e, 0]], pp[[e, 1]], pp[[e, 2]]];
                let mrow = [pm[[e, 0]], pm[[e, 1]], pm[[e, 2]]];
                if alive(prow) != alive(p) || alive(mrow) != alive(p) {
                    continue;
                }
                let (pgx, pgy, pgz, plap) = compute_aos_grad_lap_fast_cut(&d, &pp.view());
                let (mgx, mgy, mgz, mlap) = compute_aos_grad_lap_fast_cut(&d, &pm.view());
                let (pg, mg) = ([pgx, pgy, pgz], [mgx, mgy, mgz]);
                for c in 0..d.num_ao {
                    for j in 0..3 {
                        let fd = (pg[j][[c, e]] - mg[j][[c, e]]) / (2.0 * h);
                        assert!(
                            close(fd, b.hessian[j][k][[c, e]], fd_atol, fd_rtol),
                            "hess ao={c} e={e} {j}{k}: {fd}"
                        );
                    }
                    let fd = (plap[[c, e]] - mlap[[c, e]]) / (2.0 * h);
                    assert!(
                        close(fd, b.gradlap[k][[c, e]], fd_atol, fd_rtol),
                        "gradlap ao={c} e={e} d={k}: {fd}"
                    );
                }
                n_fd += 1;
            }
        }
        assert!(
            n_dead > 0 && n_fd > 0,
            "screen not exercised: {n_dead} {n_fd}"
        );
    }
}
