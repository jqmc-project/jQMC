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

//! Complex Bloch-pGTO AO evaluation (PBC twist k!=Gamma). The real Gaussian kernel is
//! UNCHANGED; complex enters via two unit-modulus phases (jqmc `_ComplexBlochPGTOMixin`):
//!   1. image-sum phase `e^{ik*T_n}` per kernel-AO (`_image_phase_factors`), applied
//!      as the weight in the image fold (instead of the real path's plain sum);
//!   2. wrap-restore phase `e^{2pii k_frac*floor(r*inv_cell)}` (jqmc's convention
//!      to fix the phase of its *wrapped* finite image sum).
//!      jqmc wraps `r` into the home cell, sums the truncated images centred there,
//!      then re-applies `restore`. The Rust kernel (like the real-Gamma path) evaluates at
//!      the UNWRAPPED `r` against the static image-tiled centres, so the Bloch sum
//!      `Sum_n e^{ik*T_n} chi(r-T_n)` is already the true amplitude at `r` -- **no restore
//!      needed** (verified: with-restore double-counts by `e^{ik*T_0}`). So
//!      `AO_k(r)[home,e] = Sum_{kappa->home} phase[kappa]*kernel_real[kappa,e]`.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::Complex64;
use pyo3::prelude::*;

use crate::atomic_orbital::{
    compute_aos_gradlap_kernel_sphe, compute_aos_hessian_kernel_sphe,
    compute_aos_kernel_with_policy, compute_aos_vgl_kernel, AosGtoData,
};

/// Periodic wrap+restore context for the complex pgto leaf (jqmc
/// `_ComplexBlochPGTOMixin`). For |r|>~half-cell the static image-tiled kernel
/// misses images; jqmc wraps r into the home cell (so the truncated image sum
/// suffices) and re-applies `restore = e^{2pii k_frac*floor(r*inv_cell)}`.
pub struct WrapCtx {
    pub cell: [[f64; 3]; 3],     // rows = lattice vectors a_i
    pub inv_cell: [[f64; 3]; 3], // r_frac = r * inv_cell
    pub k_frac: [f64; 3],
}

impl WrapCtx {
    /// Wrap `r` per-electron into the home cell -> (`r_wrapped`, `restore[e]`).
    /// `n0 = floor(r*inv_cell)`, `r_wrapped = r - n0*cell`, `restore = e^{2pii n0*k_frac}`.
    pub fn wrap(&self, r: &ArrayView2<f64>) -> (Array2<f64>, Vec<Complex64>) {
        let n = r.nrows();
        let mut rw = Array2::<f64>::zeros((n, 3));
        let mut restore = vec![Complex64::new(1.0, 0.0); n];
        for e in 0..n {
            let mut n0 = [0.0_f64; 3];
            for (j, n0_j) in n0.iter_mut().enumerate() {
                let frac = r[[e, 0]] * self.inv_cell[0][j]
                    + r[[e, 1]] * self.inv_cell[1][j]
                    + r[[e, 2]] * self.inv_cell[2][j];
                *n0_j = frac.floor();
            }
            for c in 0..3 {
                rw[[e, c]] = r[[e, c]]
                    - (n0[0] * self.cell[0][c] + n0[1] * self.cell[1][c] + n0[2] * self.cell[2][c]);
            }
            let ph = 2.0
                * std::f64::consts::PI
                * (n0[0] * self.k_frac[0] + n0[1] * self.k_frac[1] + n0[2] * self.k_frac[2]);
            restore[e] = Complex64::new(0.0, ph).exp();
        }
        (rw, restore)
    }

    /// Single-point form of [`Self::wrap`]: the same per-electron arithmetic in the
    /// same order, so `wrap_point(r[e]) == (wrap(r).0[e], wrap(r).1[e])` bitwise.
    pub fn wrap_point(&self, p: [f64; 3]) -> ([f64; 3], Complex64) {
        let mut n0 = [0.0_f64; 3];
        for (j, n0_j) in n0.iter_mut().enumerate() {
            let frac = p[0] * self.inv_cell[0][j]
                + p[1] * self.inv_cell[1][j]
                + p[2] * self.inv_cell[2][j];
            *n0_j = frac.floor();
        }
        let mut rw = [0.0_f64; 3];
        for c in 0..3 {
            rw[c] = p[c]
                - (n0[0] * self.cell[0][c] + n0[1] * self.cell[1][c] + n0[2] * self.cell[2][c]);
        }
        let ph = 2.0
            * std::f64::consts::PI
            * (n0[0] * self.k_frac[0] + n0[1] * self.k_frac[1] + n0[2] * self.k_frac[2]);
        (rw, Complex64::new(0.0, ph).exp())
    }
}

/// Multiply each electron-column `e` of `a` (m,N) by `restore[e]` in place.
pub fn apply_restore(a: &mut Array2<Complex64>, restore: &[Complex64]) {
    let (m, n) = (a.nrows(), a.ncols());
    for i in 0..m {
        for e in 0..n {
            a[[i, e]] *= restore[e];
        }
    }
}

pub fn apply_restore_f32(a: &mut Array2<Complex64>, restore: &[Complex64]) {
    let (m, n) = (a.nrows(), a.ncols());
    for i in 0..m {
        for e in 0..n {
            let ar = a[[i, e]].re as f32;
            let ai = a[[i, e]].im as f32;
            let br = restore[e].re as f32;
            let bi = restore[e].im as f32;
            a[[i, e]] = Complex64::new((ar * br - ai * bi) as f64, (ar * bi + ai * br) as f64);
        }
    }
}

/// Complex plane-wave leaf of a pgto(+)pw composite at a twist k!=Gamma: `AO_k(r) =
/// e^{i*gk_k*r}`, `gk = G + k` (`_gk_vectors_jnp`, twist already folded in). Unlike
/// the real cos/sin PW, this is a pure complex exponential -- no `cell` wrap (the
/// `e^{ik*r}` Bloch phase must survive). grad `i*gk*AO`, lap `-|gk|^2*AO`,
/// Hessian `-gk_a gk_b*AO`.
pub struct ComplexPwData {
    pub gk: Vec<[f64; 3]>, // (num_ao,)
    pub num_ao: usize,
    /// REAL-Gamma (k = 0) child marker. jqmc's real PW basis (`AOs_plane_wave_data`) is a
    /// per-AO REAL cos/sin -- value `cos(G.r - is_sin*pi/2)` (`_phase_is_sin_jnp`), NOT the
    /// twist complex exponential `e^{i gk.r}`. When `Some(is_sin)`, the kernel evaluates the
    /// real cos/sin form (imaginary part identically zero) so the complex composite reduces
    /// to jqmc's real Gamma AO block. `None` = genuine twist -> pure complex exponential.
    pub phase_is_sin: Option<Vec<f64>>,
    /// Construction-time arithmetic policy (see `AosGtoData::precision_policy`).
    pub(crate) precision_policy: crate::PrecisionPolicy,
}

/// PW value at phase `theta = g.r`: real cos/sin (Gamma) or complex exponential (twist).
#[inline]
fn pw_value(phase_is_sin: &Option<Vec<f64>>, k: usize, theta: f64) -> Complex64 {
    match phase_is_sin {
        None => Complex64::new(0.0, theta).exp(),
        Some(s) => Complex64::new((theta - s[k] * std::f64::consts::FRAC_PI_2).cos(), 0.0),
    }
}

/// d(value)/d theta at phase `theta`: complex `i*e^{i theta}` (twist) or real
/// `-sin(theta - is_sin*pi/2)` (Gamma). grad_c = this * g_c, lap = -|g|^2 * value.
#[inline]
fn pw_dtheta(phase_is_sin: &Option<Vec<f64>>, k: usize, theta: f64, val: Complex64) -> Complex64 {
    match phase_is_sin {
        None => Complex64::new(0.0, 1.0) * val,
        Some(s) => Complex64::new(-(theta - s[k] * std::f64::consts::FRAC_PI_2).sin(), 0.0),
    }
}

impl ComplexPwData {
    pub fn from_py(pw_py: &Bound<'_, PyAny>) -> PyResult<Self> {
        let np = pw_py.py().import("numpy")?;
        // Twist child (`AOs_plane_wave_complex_data`) exposes `_gk_vectors_jnp` (gk = G + k)
        // and a complex-exponential basis. A REAL-Gamma child (`AOs_plane_wave_data`, k = 0)
        // lacks it, exposing only `_g_vectors_jnp` (bare G) and a REAL cos/sin basis selected
        // per AO by `_phase_is_sin_jnp`. Detect the Gamma child so the complex kernel evaluates
        // jqmc's real cos/sin form (imaginary parts vanish) -- a legitimate k = 0 cross-check.
        let (gk_attr, phase_is_sin) = match pw_py.getattr("_gk_vectors_jnp") {
            Ok(a) => (a, None),
            Err(_) => {
                let g = pw_py.getattr("_g_vectors_jnp")?;
                let s: numpy::PyReadonlyArray1<f64> = np
                    .call_method1("asarray", (pw_py.getattr("_phase_is_sin_jnp")?,))?
                    .call_method1("astype", (np.getattr("float64")?,))?
                    .extract()?;
                (g, Some(s.as_array().to_vec()))
            }
        };
        let gkv: numpy::PyReadonlyArray2<f64> =
            np.call_method1("asarray", (gk_attr,))?.extract()?;
        let g = gkv.as_array();
        let num_ao = g.nrows();
        let gk = (0..num_ao)
            .map(|k| [g[[k, 0]], g[[k, 1]], g[[k, 2]]])
            .collect();
        Ok(ComplexPwData {
            gk,
            num_ao,
            phase_is_sin,
            precision_policy: crate::construction_precision_policy(),
        })
    }
}

/// Complex PW values `(num_ao, N)`.
pub fn compute_pw_complex(data: &ComplexPwData, r: &ArrayView2<f64>) -> Array2<Complex64> {
    compute_pw_complex_with_policy(data, r, data.precision_policy.is_f32("ao_eval"))
}

pub fn compute_pw_complex_with_policy(
    data: &ComplexPwData,
    r: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> Array2<Complex64> {
    let n = r.nrows();
    let mut out = Array2::<Complex64>::zeros((data.num_ao, n));
    let tau = std::f64::consts::TAU;
    if !f32_arithmetic && data.phase_is_sin.is_none() {
        // fp64 Bloch plane waves e^{i theta}: vectorized sin/cos over the points
        // (`sincos_pd_into`, <= 2e-15 absolute), as the real PW path's `cos_pd_into`,
        // instead of one scalar complex `exp` per (PW, point).
        let (xs, ys, zs): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            (0..n).map(|e| r[[e, 0]]).collect(),
            (0..n).map(|e| r[[e, 1]]).collect(),
            (0..n).map(|e| r[[e, 2]]).collect(),
        );
        let mut th = vec![0.0_f64; n];
        let mut sn = vec![0.0_f64; n];
        let mut cs = vec![0.0_f64; n];
        for k in 0..data.num_ao {
            let g = data.gk[k];
            for e in 0..n {
                th[e] = g[0] * xs[e] + g[1] * ys[e] + g[2] * zs[e];
            }
            crate::atomic_orbital::sincos_pd_into(&th, &mut sn, &mut cs);
            let mut row = out.row_mut(k);
            for e in 0..n {
                row[e] = Complex64::new(cs[e], sn[e]);
            }
        }
        return out;
    }
    if f32_arithmetic {
        // `gk*r` grows with the distance from the home cell, and an f32 product
        // loses ~|r||gk|*6e-8 rad. Form it in f64 and reduce it modulo 2 pi before
        // the f32 cast (jqmc `AOs_plane_wave_complex_data._value_in_zone`); the fp32
        // sin/cos is the vectorizable `fastmath::sincos_f32`
        // (IMPL-RS-F32-AO-EVAL-SIMD-001).
        let mut th = vec![0.0_f32; n];
        for k in 0..data.num_ao {
            let g = data.gk[k];
            for e in 0..n {
                let theta64 = g[0] * r[[e, 0]] + g[1] * r[[e, 1]] + g[2] * r[[e, 2]];
                th[e] = (theta64 - tau * (theta64 / tau).round()) as f32;
            }
            let mut row = out.row_mut(k);
            match &data.phase_is_sin {
                None => {
                    for e in 0..n {
                        let (sn, cs) = crate::fastmath::sincos_f32(th[e]);
                        row[e] = Complex64::new(cs as f64, sn as f64);
                    }
                }
                Some(s) => {
                    let shift = s[k] as f32 * std::f32::consts::FRAC_PI_2;
                    for e in 0..n {
                        row[e] = Complex64::new(
                            crate::fastmath::sincos_f32(th[e] - shift).1 as f64,
                            0.0,
                        );
                    }
                }
            }
        }
        return out;
    }
    for k in 0..data.num_ao {
        let g = data.gk[k];
        for e in 0..n {
            {
                let theta = g[0] * r[[e, 0]] + g[1] * r[[e, 1]] + g[2] * r[[e, 2]];
                out[[k, e]] = pw_value(&data.phase_is_sin, k, theta);
            }
        }
    }
    out
}

/// Complex PW value + grad `[dx,dy,dz]` + Laplacian, each `(num_ao, N)`.
#[allow(clippy::type_complexity)]
pub fn compute_pw_vgl_complex(
    data: &ComplexPwData,
    r: &ArrayView2<f64>,
) -> (Array2<Complex64>, [Array2<Complex64>; 3], Array2<Complex64>) {
    let n = r.nrows();
    let mk = || Array2::<Complex64>::zeros((data.num_ao, n));
    let (mut val, mut gx, mut gy, mut gz, mut lap) = (mk(), mk(), mk(), mk(), mk());
    let value_is_f32 = data.precision_policy.is_f32("ao_eval");
    for k in 0..data.num_ao {
        let g = data.gk[k];
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        for e in 0..n {
            let theta = g[0] * r[[e, 0]] + g[1] * r[[e, 1]] + g[2] * r[[e, 2]];
            let v = pw_value(&data.phase_is_sin, k, theta);
            val[[k, e]] = if value_is_f32 {
                Complex64::new((v.re as f32) as f64, (v.im as f32) as f64)
            } else {
                v
            };
            let dt = pw_dtheta(&data.phase_is_sin, k, theta, v);
            gx[[k, e]] = dt * g[0];
            gy[[k, e]] = dt * g[1];
            gz[[k, e]] = dt * g[2];
            lap[[k, e]] = -g2 * v;
        }
    }
    (val, [gx, gy, gz], lap)
}

/// Complex PW Hessian `grad_agrad_b AO = -gk_a gk_b*value` (3x3) each `(num_ao, N)`.
/// `value = e^{i gk*r}` (twist) or `cos(G.r - is_sin*pi/2)` (Gamma); `d^2/d theta^2 = -value`
/// holds for both, so the Hessian is `-g_a g_b * value` in either case.
pub fn compute_pw_hessian_complex(
    data: &ComplexPwData,
    r: &ArrayView2<f64>,
) -> [[Array2<Complex64>; 3]; 3] {
    let n = r.nrows();
    let mk = || Array2::<Complex64>::zeros((data.num_ao, n));
    let mut h = [[mk(), mk(), mk()], [mk(), mk(), mk()], [mk(), mk(), mk()]];
    for k in 0..data.num_ao {
        let g = data.gk[k];
        for e in 0..n {
            let theta = g[0] * r[[e, 0]] + g[1] * r[[e, 1]] + g[2] * r[[e, 2]];
            let v = pw_value(&data.phase_is_sin, k, theta);
            for a in 0..3 {
                for b in 0..3 {
                    h[a][b][[k, e]] = -g[a] * g[b] * v;
                }
            }
        }
    }
    h
}

/// Bloch-weighted image fold of a real kernel-rank array `(num_ao_kernel, N)` ->
/// complex `(num_ao, N)`: `Sum_{kappa->home} e^{ik*T_kappa}*kernel[kappa,e]`.
pub fn fold_complex(
    data: &AosGtoData,
    kernel: &Array2<f64>,
    image_phase: &[Complex64],
) -> Array2<Complex64> {
    let n = kernel.ncols();
    let mut out = Array2::<Complex64>::zeros((data.num_ao, n));
    for kap in 0..data.num_ao_kernel {
        let home = data.kernel_to_home[kap];
        let ph = image_phase[kap];
        for e in 0..n {
            out[[home, e]] += ph * kernel[[kap, e]];
        }
    }
    out
}

fn fold_complex_f32(
    data: &AosGtoData,
    kernel: &Array2<f64>,
    image_phase: &[Complex64],
) -> Array2<Complex64> {
    // Contiguous-row form of the per-element fold: the same `re += pr*v`,
    // `im += pi*v` accumulation in kernel-AO order, hence the same bits.
    let n = kernel.ncols();
    let mut re = vec![0.0_f32; data.num_ao * n];
    let mut im = vec![0.0_f32; data.num_ao * n];
    for (kap, image_phase_kap) in image_phase[..data.num_ao_kernel].iter().enumerate() {
        let home = data.kernel_to_home[kap];
        let pr = image_phase_kap.re as f32;
        let pi = image_phase_kap.im as f32;
        let krow = kernel.row(kap);
        let (rr, ii) = (
            &mut re[home * n..(home + 1) * n],
            &mut im[home * n..(home + 1) * n],
        );
        for e in 0..n {
            let v = krow[e] as f32;
            rr[e] += pr * v;
            ii[e] += pi * v;
        }
    }
    let mut out = Vec::with_capacity(data.num_ao * n);
    for i in 0..data.num_ao * n {
        out.push(Complex64::new(re[i] as f64, im[i] as f64));
    }
    Array2::from_shape_vec((data.num_ao, n), out).expect("fold shape")
}

/// Complex Bloch AO values `(num_ao, N)`. `image_phase` len `num_ao_kernel`.
/// Evaluated at unwrapped `r` (image-tiled) -> no restore phase. Matches jqmc
/// `AOs_*_pgto_complex_data._compute_AOs` to the dynamic-AO-cutoff tolerance.
pub fn compute_aos_complex(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
) -> Array2<Complex64> {
    compute_aos_complex_with_policy(
        data,
        image_phase,
        r,
        data.precision_policy.is_f32(data.value_zone),
    )
}

pub fn compute_aos_complex_with_policy(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> Array2<Complex64> {
    let kernel = crate::timed!(
        "cao.kernel",
        compute_aos_kernel_with_policy(data, r, f32_arithmetic)
    );
    crate::timed!(
        "cao.fold",
        if f32_arithmetic {
            fold_complex_f32(data, &kernel, image_phase)
        } else {
            fold_complex(data, &kernel, image_phase)
        }
    )
}

/// Complex Bloch `grad(grad^2AO)` `[dx,dy,dz]` each `(num_ao, N)`, SPHE leaf. Real sphe
/// gradlap kernel (closed form) Bloch-folded with `e^{ik*T_n}`.
pub fn compute_aos_gradlap_complex(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
) -> [Array2<Complex64>; 3] {
    let gl = compute_aos_gradlap_kernel_sphe(data, r, false);
    [
        fold_complex(data, &gl[0], image_phase),
        fold_complex(data, &gl[1], image_phase),
        fold_complex(data, &gl[2], image_phase),
    ]
}

/// Complex Bloch AO Hessian `[[dadb]]` (3x3) each `(num_ao, N)`, SPHE leaf. Real sphe
/// Hessian kernel (closed form) Bloch-folded with `e^{ik*T_n}`. For the complex withJ
/// gou cross term (d(gradAO)/d{R,eps} = -/+gradgradAO).
pub fn compute_aos_hessian_complex(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
) -> [[Array2<Complex64>; 3]; 3] {
    let h = compute_aos_hessian_kernel_sphe(data, r, false);
    [
        [
            fold_complex(data, &h[0][0], image_phase),
            fold_complex(data, &h[0][1], image_phase),
            fold_complex(data, &h[0][2], image_phase),
        ],
        [
            fold_complex(data, &h[1][0], image_phase),
            fold_complex(data, &h[1][1], image_phase),
            fold_complex(data, &h[1][2], image_phase),
        ],
        [
            fold_complex(data, &h[2][0], image_phase),
            fold_complex(data, &h[2][1], image_phase),
            fold_complex(data, &h[2][2], image_phase),
        ],
    ]
}

/// Complex Bloch AO value + grad `[dx,dy,dz]` + Laplacian, each `(num_ao, N)`.
/// Real kernel-rank vgl (sphe+cart) Bloch-folded with `e^{ik*T_n}`. Unwrapped r
/// -> no restore. 1:1 jqmc complex `_compute_AOs{,_grad,_laplacian}`.
#[allow(clippy::type_complexity)]
pub fn compute_aos_vgl_complex(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
) {
    let (val, gx, gy, gz, lap) = compute_aos_vgl_kernel(data, r, false);
    (
        fold_complex(data, &val, image_phase),
        fold_complex(data, &gx, image_phase),
        fold_complex(data, &gy, image_phase),
        fold_complex(data, &gz, image_phase),
        fold_complex(data, &lap, image_phase),
    )
}

#[allow(clippy::type_complexity)]
pub fn compute_aos_vgl_complex_with_policy(
    data: &AosGtoData,
    image_phase: &[Complex64],
    r: &ArrayView2<f64>,
    value_f32: bool,
) -> (
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
    Array2<Complex64>,
) {
    let (val, gx, gy, gz, lap) = compute_aos_vgl_kernel(data, r, false);
    let v = if value_f32 {
        let rounded = val.mapv(|x| (x as f32) as f64);
        fold_complex_f32(data, &rounded, image_phase)
    } else {
        fold_complex(data, &val, image_phase)
    };
    (
        v,
        fold_complex(data, &gx, image_phase),
        fold_complex(data, &gy, image_phase),
        fold_complex(data, &gz, image_phase),
        fold_complex(data, &lap, image_phase),
    )
}

#[cfg(test)]
mod fold_f32_tests {
    use super::*;

    #[test]
    fn fold_complex_f32_rows_is_bitwise_per_element() {
        // The contiguous-row fp32 Bloch fold
        // must reproduce the per-element scatter accumulation written out below bit for bit.
        // (Signed zeros in the kernel exercise the +0/-0 accumulation.)
        for n_image in [1, 3, 6] {
            let d = crate::atomic_orbital::tests::vgl_test_data_l(
                crate::atomic_orbital::AoKind::Cart,
                n_image,
                9.0,
                false,
            );
            let n = 4;
            let phase: Vec<Complex64> = (0..d.num_ao_kernel)
                .map(|k| Complex64::new(0.0, 0.37 * k as f64 - 0.5).exp())
                .collect();
            let kernel =
                Array2::from_shape_fn((d.num_ao_kernel, n), |(k, e)| match (k * 3 + e) % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => ((k * 5 + e * 11) % 17) as f64 * 0.013 - 0.1,
                });
            let got = fold_complex_f32(&d, &kernel, &phase);
            let mut re = vec![0.0_f32; d.num_ao * n];
            let mut im = vec![0.0_f32; d.num_ao * n];
            for k in 0..d.num_ao_kernel {
                let h = d.kernel_to_home[k];
                for e in 0..n {
                    let v = kernel[[k, e]] as f32;
                    re[h * n + e] += phase[k].re as f32 * v;
                    im[h * n + e] += phase[k].im as f32 * v;
                }
            }
            for i in 0..d.num_ao {
                for e in 0..n {
                    assert_eq!(
                        got[[i, e]].re.to_bits(),
                        (re[i * n + e] as f64).to_bits(),
                        "re n_image={n_image}"
                    );
                    assert_eq!(
                        got[[i, e]].im.to_bits(),
                        (im[i * n + e] as f64).to_bits(),
                        "im n_image={n_image}"
                    );
                }
            }
        }
    }
}
