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

//! Branch-free, auto-vectorizable fp64 kernels for wide-SIMD, long-FP-latency
//! cores (A64FX): plain counted loops without calls, divisions shared per element.
//!
//! `erfc` here is `exp(-z^2) * P(u)`, `u` a linear map of `t = K/(K+z)`, with `P` a
//! Chebyshev fit of erfcx on z in [0, ZMAX] (tools/gen_erfc.py, degree 26). Relative
//! error versus `libm::erfc` is below 5e-15; above ZMAX (`erfc < 4e-20`) it is 0.

const TMIN: f64 = 0.315_789_473_684_210_5;
const K: f64 = 3.0;
const C: [f64; 27] = [
    0.312_035_241_513_324_4,
    0.367_131_031_226_527_6,
    0.205_493_799_661_698_87,
    0.085_619_389_134_162_99,
    0.025_283_677_310_025_916,
    0.004_490_129_648_068_742,
    0.000_108_976_546_897_508_12,
    -0.000_148_561_999_978_381_01,
    -0.000_019_943_387_716_917_53,
    5.445_675_439_703_194e-6,
    1.135_842_034_768_632_2e-6,
    -2.871_414_517_451_345e-7,
    -5.420_969_371_492_743e-8,
    1.974_450_978_734_928_3e-8,
    1.733_899_277_821_542_6e-9,
    -1.424_881_417_717_095_7e-9,
    5.804_236_089_624_264_5e-11,
    8.854_354_331_029_151e-11,
    -1.787_933_354_598_612e-11,
    -3.284_607_098_225_162e-12,
    1.933_086_530_183_455_2e-12,
    -1.366_563_415_077_738_8e-13,
    -1.211_264_124_385_673_3e-13,
    3.557_820_090_642_348_3e-14,
    1.927_327_882_604_076_5e-15,
    -2.496_179_279_875_237_7e-15,
    2.901_204_897_214_130_6e-16,
];
const ZMAX: f64 = 6.5;

/// `exp(x)` for `x <= 0`, degree-13 Taylor in Estrin form (shallow FMA tree: the
/// Horner chain costs one full FP latency per coefficient on A64FX).
///
/// Below [`EXP_NEG_MIN_ARG`] (just above `ln(f64::MIN_POSITIVE)`) the result is 0,
/// like libm's underflow (libm returns subnormals in between, below 2.3e-308), so a
/// Gaussian far from its centre is exactly zero and a vanishing determinant stays
/// detectable. NaN propagates.
#[inline(always)]
pub fn exp_neg(x_in: f64) -> f64 {
    const SHIFTER: f64 = 6_755_399_441_055_744.0;
    let x = if x_in > EXP_NEG_MIN_ARG {
        x_in
    } else {
        EXP_NEG_MIN_ARG
    };
    let t = x.mul_add(std::f64::consts::LOG2_E, SHIFTER);
    let kf = t - SHIFTER;
    let r = (-kf).mul_add(
        5.497_923_018_708_371e-14,
        (-kf).mul_add(6.931_471_805_598_903e-1, x),
    );
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let p01 = r + 1.0;
    let p23 = (1.0 / 6.0_f64).mul_add(r, 0.5);
    let p45 = (1.0 / 120.0_f64).mul_add(r, 1.0 / 24.0);
    let p67 = (1.0 / 5040.0_f64).mul_add(r, 1.0 / 720.0);
    let p89 = (1.0 / 362_880.0_f64).mul_add(r, 1.0 / 40_320.0);
    let pab = (1.0 / 39_916_800.0_f64).mul_add(r, 1.0 / 3_628_800.0);
    let pcd = (1.0 / 6_227_020_800.0_f64).mul_add(r, 1.0 / 479_001_600.0);
    let p03 = p23.mul_add(r2, p01);
    let p47 = p67.mul_add(r2, p45);
    let p8b = pab.mul_add(r2, p89);
    let p07 = p47.mul_add(r4, p03);
    let p8d = pcd.mul_add(r4, p8b);
    let p = p8d.mul_add(r8, p07);
    // `t = SHIFTER + k` exactly (|k| <= 1022 stays in SHIFTER's binade, whose ulp
    // is 1), so the integer k is the difference of the bit patterns.
    let k = t.to_bits() as i64 - SHIFTER.to_bits() as i64;
    let v = p * f64::from_bits(((k + 1023) << 52) as u64);
    if x_in < EXP_NEG_MIN_ARG {
        0.0
    } else if x_in.is_nan() {
        x_in
    } else {
        v
    }
}

/// Smallest argument [`exp_neg`] evaluates; `exp(-708) = 3.3e-308` is still normal.
pub const EXP_NEG_MIN_ARG: f64 = -708.0;

/// Scalar fp64 sin/cos core (~1 ULP): Cody-Waite reduction by pi/2 (2-part,
/// FMA-compensated; exact for |x| <~ 1e6, well beyond the PW theta range)
/// into `r` in [-pi/4, pi/4], cephes minimax sin/cos polynomials, branchless
/// quadrant swap+sign from `q mod 4` (compiles to selects, so the callers'
/// loops auto-vectorize). This is the SINGLE fast sin/cos implementation (plane-wave AO hot
/// paths and the FAST periodic J1/J2 smooth distance).
#[inline(always)]
pub fn sincos(x: f64) -> (f64, f64) {
    const INVPIO2: f64 = std::f64::consts::FRAC_2_PI;
    const PIO2_1: f64 = 1.570_796_326_734_125_6; // pi/2 high (33 bits)
    const PIO2_1T: f64 = 6.077_100_506_506_192e-11; // pi/2 tail
                                                    // cephes sin minimax (coeff of z^0..z^5 in the bracket, high->low order):
    const S0: f64 = 1.589_623_015_765_465_6e-10;
    const S1: f64 = -2.505_074_776_285_780_7e-8;
    const S2: f64 = 2.755_731_362_138_572_2e-6;
    const S3: f64 = -1.984_126_982_958_954e-4;
    const S4: f64 = 8.333_333_333_322_118e-3;
    const S5: f64 = -1.666_666_666_666_663e-1;
    // cephes cos minimax:
    const C0: f64 = -1.135_853_652_138_768_2e-11;
    const C1: f64 = 2.087_570_084_197_473e-9;
    const C2: f64 = -2.755_731_417_929_674e-7;
    const C3: f64 = 2.480_158_728_885_170_4e-5;
    const C4: f64 = -1.388_888_888_887_305_6e-3;
    const C5: f64 = 4.166_666_666_666_659_5e-2;
    // Round-to-nearest-even via the 1.5*2^52 shifter (see [`exp_neg`]):
    // keeps the reduction free of the non-vectorizable f64::round lowering and
    // matches AVX-512 vrndscale mode 0 (round to nearest even). |theta| stays far below
    // 2^51, so the shifter trick is exact.
    const SHIFTER: f64 = 6_755_399_441_055_744.0;
    let fq = x.mul_add(INVPIO2, SHIFTER) - SHIFTER; // round(x*2/pi)
    let mut r = (-fq).mul_add(PIO2_1, x);
    r = (-fq).mul_add(PIO2_1T, r);
    let z = r * r;
    // sin poly: rsin = r + r*z*(S5 + z*(S4 + ... + z*S0))
    let mut ps = S0.mul_add(z, S1);
    ps = ps.mul_add(z, S2);
    ps = ps.mul_add(z, S3);
    ps = ps.mul_add(z, S4);
    ps = ps.mul_add(z, S5);
    let rsin = (ps * z).mul_add(r, r);
    // cos poly: rcos = 1 - 0.5*z + z^2*(C5 + z*(C4 + ... + z*C0))
    let mut pc = C0.mul_add(z, C1);
    pc = pc.mul_add(z, C2);
    pc = pc.mul_add(z, C3);
    pc = pc.mul_add(z, C4);
    pc = pc.mul_add(z, C5);
    let rcos = pc.mul_add(z * z, (-0.5_f64).mul_add(z, 1.0));
    // Branchless quadrant selection from q mod 4.
    let q = fq as i64;
    let swap = q & 1 != 0;
    let s_val = if swap { rcos } else { rsin };
    let c_val = if swap { rsin } else { rcos };
    let s_val = if q & 2 != 0 { -s_val } else { s_val };
    let c_val = if (q + 1) & 2 != 0 { -c_val } else { c_val };
    (s_val, c_val)
}

/// fp32 `exp(x)` for `x <= 88` (~1 ULP; cephes `expf` minimax): the vectorizable
/// fp32 counterpart of [`exp_neg`] for the mixed `ao_eval` zone, which XLA evaluates
/// in fp32 with its own (non-libm) polynomial. Round-to-nearest of `x*log2e` via
/// the 1.5*2^23 shifter, 2-part Cody-Waite `ln2`, degree-6 polynomial, direct
/// binary32 `2^k`. Below -87.3 (smallest normal) the result is flushed to +0.
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    const SHIFTER: f32 = 12_582_912.0; // 1.5 * 2^23
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let xc = x.clamp(-87.0, 88.0);
    let kf = xc.mul_add(std::f32::consts::LOG2_E, SHIFTER) - SHIFTER;
    let r = (-kf).mul_add(LN2_LO, (-kf).mul_add(LN2_HI, xc));
    let mut p = 1.987_569_1e-4_f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_5e-1);
    p = p.mul_add(r, 5e-1);
    let y = (p * r).mul_add(r, r) + 1.0;
    // `as` saturates (NaN -> 0); kf is integral with |kf| <= 127 otherwise.
    let k = kf as i32;
    let v = y * f32::from_bits(((k + 127) << 23) as u32);
    if x < -87.0 {
        // exp(-87) = 1.6e-38: the remaining tail is at or below the smallest
        // normal float, i.e. negligible against any finite AO coefficient.
        0.0
    } else {
        v
    }
}

/// `dst[i] = exp_f32(src[i])` (counted loop; auto-vectorized, 16 lanes on AVX-512).
#[inline]
pub fn exp_f32_into(src: &[f32], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let (src, dst) = (&src[..n], &mut dst[..n]);
    for i in 0..n {
        dst[i] = exp_f32(src[i]);
    }
}

/// fp32 sin/cos (~1-2 ULP for |x| <~ 1e4; cephes `sinf`/`cosf` minimax on
/// [-pi/4, pi/4]): 3-part Cody-Waite reduction by pi/2 and the branchless
/// quadrant selection of [`sincos`]. The vectorizable fp32 plane-wave kernel of
/// the mixed `ao_eval` zone.
#[inline(always)]
pub fn sincos_f32(x: f32) -> (f32, f32) {
    const SHIFTER: f32 = 12_582_912.0;
    const INVPIO2: f32 = std::f32::consts::FRAC_2_PI;
    const P1: f32 = 1.570_312_5;
    const P2: f32 = 4.837_513e-4;
    const P3: f32 = 7.549_79e-8;
    let fq = x.mul_add(INVPIO2, SHIFTER) - SHIFTER;
    let mut r = (-fq).mul_add(P1, x);
    r = (-fq).mul_add(P2, r);
    r = (-fq).mul_add(P3, r);
    let z = r * r;
    let ps = (-1.951_529_6e-4_f32)
        .mul_add(z, 8.332_161e-3)
        .mul_add(z, -1.666_665_5e-1);
    let rsin = (ps * z).mul_add(r, r);
    let pc = 2.443_315_7e-5_f32
        .mul_add(z, -1.388_731_6e-3)
        .mul_add(z, 4.166_664_6e-2);
    let rcos = pc.mul_add(z * z, (-0.5_f32).mul_add(z, 1.0));
    let q = fq as i32;
    let swap = q & 1 != 0;
    let s_val = if swap { rcos } else { rsin };
    let c_val = if swap { rsin } else { rcos };
    let s_val = if q & 2 != 0 { -s_val } else { s_val };
    let c_val = if (q + 1) & 2 != 0 { -c_val } else { c_val };
    (s_val, c_val)
}

/// `erfc(z)` for `0 <= z < ZMAX`.
#[inline(always)]
fn erfc_core(z: f64) -> f64 {
    let t = K / (K + z);
    let u = 2.0 * (t - TMIN) / (1.0 - TMIN) - 1.0;
    let mut p = C[C.len() - 1];
    let mut i = C.len() - 1;
    while i > 0 {
        i -= 1;
        p = p.mul_add(u, C[i]);
    }
    p * exp_neg(-z * z)
}

/// Scalar `erfc` for tests.
pub fn erfc_scalar(z: f64) -> f64 {
    if z < ZMAX {
        erfc_core(z)
    } else {
        0.0
    }
}

/// Division guard of the masked `d = 0` term (`jqmc._setting.EPS_machine_precision`;
/// the guarded value is discarded by the mask).
const EPS_MACHINE_PRECISION: f64 = 1.0e-300;

/// `Sum_n erfc(eta d_n)/d_n` with `d_n = |mic + image_n|`. `alat = None` masks the
/// `d = 0` term; `Some(a)` floors every distance at `a` (the discretized potential).
/// `work` is scratch of the image-list length; `ix/iy/iz` are the image columns and
/// `inorm` their norms `|T_n|`, sorted ascending (see `ewald::image_columns`).
///
/// Every image with `|T_n| >= ZMAX/eta + |mic|` has `d_n >= |T_n| - |mic| >= ZMAX/eta`,
/// where the masked kernel below contributes exactly 0; the sum therefore runs only
/// over the sorted prefix below that radius (plus a 1e-9 bohr rounding margin). The
/// skipped terms are exact zeros, so this is the full masked sum up to summation
/// order.
#[allow(clippy::too_many_arguments)]
pub fn ewald_image_sum(
    mic: [f64; 3],
    ix: &[f64],
    iy: &[f64],
    iz: &[f64],
    inorm: &[f64],
    eta: f64,
    alat: Option<f64>,
    work: &mut [f64],
) -> f64 {
    debug_assert_eq!(inorm.len(), ix.len());
    debug_assert!(inorm.windows(2).all(|w| w[0] <= w[1]));
    let n = if eta > 0.0 {
        let mnorm = (mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2]).sqrt();
        let cut = ZMAX / eta + mnorm + 1.0e-9;
        inorm.partition_point(|&t| t < cut)
    } else {
        ix.len()
    };
    let (ix, iy, iz, work) = (&ix[..n], &iy[..n], &iz[..n], &mut work[..n]);
    match alat {
        None => {
            for i in 0..n {
                let (vx, vy, vz) = (mic[0] + ix[i], mic[1] + iy[i], mic[2] + iz[i]);
                let d2 = vx * vx + vy * vy + vz * vz;
                let d = (if d2 > EPS_MACHINE_PRECISION {
                    d2
                } else {
                    EPS_MACHINE_PRECISION
                })
                .sqrt();
                let z = eta * d;
                let zc = if z < ZMAX { z } else { ZMAX };
                let v = erfc_core(zc) / d;
                work[i] = if d2 > 0.0 && z < ZMAX { v } else { 0.0 };
            }
        }
        Some(a) => {
            for i in 0..n {
                let (vx, vy, vz) = (mic[0] + ix[i], mic[1] + iy[i], mic[2] + iz[i]);
                let d2 = vx * vx + vy * vy + vz * vz;
                let d0 = (d2 + crate::jastrow_factor::EPS_SAFE_DISTANCE).sqrt();
                let d = if d0 > a { d0 } else { a };
                let z = eta * d;
                let zc = if z < ZMAX { z } else { ZMAX };
                let v = erfc_core(zc) / d;
                work[i] = if z < ZMAX { v } else { 0.0 };
            }
        }
    }
    sum_lanes(work)
}

/// Derivative form of [`ewald_image_sum`] for the Ewald forces and strain:
/// `grad = Sum_n g(d_n) v_n` and, with `strain`, `vv = Sum_n g(d_n) v_n (x) v_n`
/// (upper triangle `xx, xy, xz, yy, yz, zz`), where `v_n = mic + image_n`,
/// `g(d) = (1/d) d/dd [erfc(eta d)/d] = -(2 eta/sqrt(pi)) exp(-eta^2 d^2)/d^2
/// - erfc(eta d)/d^3`. Images with `d_n < skip` are left out (the self term).
/// The same sorted-prefix cut as [`ewald_image_sum`]: the kernel masks every term
/// with `eta d >= ZMAX` to exactly 0, and every image past `ZMAX/eta + |mic|` is
/// such a term, so the cut drops only exact zeros. `erfc` and the Gaussian share
///   one exponential.
#[allow(clippy::too_many_arguments)]
pub fn ewald_image_grad_sum(
    mic: [f64; 3],
    ix: &[f64],
    iy: &[f64],
    iz: &[f64],
    inorm: &[f64],
    eta: f64,
    skip: f64,
    strain: bool,
    work: &mut [f64],
) -> ([f64; 3], [f64; 6]) {
    debug_assert_eq!(inorm.len(), ix.len());
    debug_assert!(inorm.windows(2).all(|w| w[0] <= w[1]));
    let mnorm = (mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2]).sqrt();
    let cut = ZMAX / eta + mnorm + 1.0e-9;
    let n = inorm.partition_point(|&t| t < cut);
    let (ix, iy, iz, work) = (&ix[..n], &iy[..n], &iz[..n], &mut work[..n]);
    let two_eta_sqrtpi = 2.0 * eta / std::f64::consts::PI.sqrt();
    for i in 0..n {
        let (vx, vy, vz) = (mic[0] + ix[i], mic[1] + iy[i], mic[2] + iz[i]);
        let d2 = vx * vx + vy * vy + vz * vz;
        let d2s = if d2 > EPS_MACHINE_PRECISION {
            d2
        } else {
            EPS_MACHINE_PRECISION
        };
        let d = d2s.sqrt();
        let z = eta * d;
        let zc = if z < ZMAX { z } else { ZMAX };
        let t = K / (K + zc);
        let u = 2.0 * (t - TMIN) / (1.0 - TMIN) - 1.0;
        let mut p = C[C.len() - 1];
        let mut k = C.len() - 1;
        while k > 0 {
            k -= 1;
            p = p.mul_add(u, C[k]);
        }
        let e = exp_neg(-zc * zc);
        let g = -(two_eta_sqrtpi * e + p * e / d) / d2s;
        work[i] = if d >= skip && z < ZMAX { g } else { 0.0 };
    }
    let mut gr = [0.0_f64; 3];
    let mut vv = [0.0_f64; 6];
    for i in 0..n {
        let g = work[i];
        let (vx, vy, vz) = (mic[0] + ix[i], mic[1] + iy[i], mic[2] + iz[i]);
        gr[0] += g * vx;
        gr[1] += g * vy;
        gr[2] += g * vz;
        if strain {
            vv[0] += g * vx * vx;
            vv[1] += g * vx * vy;
            vv[2] += g * vx * vz;
            vv[3] += g * vy * vy;
            vv[4] += g * vy * vz;
            vv[5] += g * vz * vz;
        }
    }
    (gr, vv)
}

/// Sum with 8 independent accumulators (an in-order sum is one FP latency per
/// element, and SVE `fadda` is no better).
#[inline]
pub fn sum_lanes(x: &[f64]) -> f64 {
    let mut acc = [0.0_f64; 8];
    let mut it = x.chunks_exact(8);
    for c in &mut it {
        for l in 0..8 {
            acc[l] += c[l];
        }
    }
    let mut s = 0.0;
    for r in it.remainder() {
        s += r;
    }
    for a in acc {
        s += a;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn erfc_and_exp_accuracy() {
        let rtol_erfc = crate::central_setting_f64("RTOL_FASTMATH_ERFC_VS_LIBM");
        let rtol_exp = crate::central_setting_f64("RTOL_FASTMATH_EXP_VS_LIBM");
        let mut z = 1e-6_f64;
        while z < 6.5 {
            let (a, b) = (libm::erfc(z), erfc_scalar(z));
            assert!(((a - b) / a).abs() < rtol_erfc, "erfc z={z} {a} {b}");
            let (a, b) = ((-z * z).exp(), exp_neg(-z * z));
            assert!(((a - b) / a).abs() < rtol_exp, "exp z={z} {a} {b}");
            z += 3.3e-4;
        }
        // Underflow and NaN follow libm: exactly 0 far below the normal range, NaN stays NaN.
        let mut x = -700.0_f64;
        while x >= EXP_NEG_MIN_ARG {
            let (a, b) = (x.exp(), exp_neg(x));
            assert!(((a - b) / a).abs() < rtol_exp, "exp x={x} {a} {b}");
            x -= 0.37;
        }
        for x in [-745.2_f64, -800.0, -1.0e4, f64::NEG_INFINITY] {
            assert_eq!(x.exp(), 0.0);
            assert_eq!(exp_neg(x), 0.0, "exp_neg({x}) must underflow to 0");
        }
        assert!(exp_neg(f64::NAN).is_nan());
    }
}

#[cfg(test)]
mod tests_f32 {
    use super::*;
    #[test]
    fn exp_f32_and_sincos_f32_match_libm_to_fp32_ulps() {
        // Relative to libm's fp32 functions, in units of fp32 eps.
        let rtol = crate::central_setting_f64("RTOL_FASTMATH_F32_VS_LIBM");
        let mut x = -87.0_f32;
        while x <= 0.0 {
            let (a, b) = (x.exp() as f64, exp_f32(x) as f64);
            assert!(((a - b) / a).abs() <= rtol, "exp x={x} {a} {b}");
            x += 1.7e-3;
        }
        assert_eq!(exp_f32(-200.0), 0.0);
        assert_eq!(exp_f32(0.0), 1.0);
        let atol = crate::central_setting_f64("ATOL_FASTMATH_F32_SINCOS_VS_LIBM");
        let mut t = -400.0_f32;
        while t <= 400.0 {
            let (s, c) = sincos_f32(t);
            assert!(
                ((t.sin() - s) as f64).abs() <= atol,
                "sin t={t} {} {s}",
                t.sin()
            );
            assert!(
                ((t.cos() - c) as f64).abs() <= atol,
                "cos t={t} {} {c}",
                t.cos()
            );
            t += 0.0137;
        }
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    #[test]
    fn image_sum_matches_libm() {
        let atol = crate::central_setting_f64("ATOL_FASTMATH_EWALD_IMAGE_SUM_VS_LIBM");
        let ix = [0.0, -6.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        let iy = [0.0, 0.0, 0.0, -6.0, 6.0, 0.0, 0.0];
        let iz = [0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 6.0];
        let inorm = [0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0];
        for eta in [0.1, 0.4, 0.9, 1.5] {
            for mic in [[0.3, -1.2, 2.2], [0.0, 0.0, 0.0], [3.0, 3.0, -3.0]] {
                let mut w = [0.0; 7];
                let s = ewald_image_sum(mic, &ix, &iy, &iz, &inorm, eta, None, &mut w);
                let mut r = 0.0;
                for n in 0..7 {
                    let (a, b, c) = (mic[0] + ix[n], mic[1] + iy[n], mic[2] + iz[n]);
                    let d2: f64 = a * a + b * b + c * c;
                    if d2 > 0.0 {
                        let d = d2.sqrt();
                        r += libm::erfc(eta * d) / d;
                    }
                }
                assert!((s - r).abs() < atol, "eta={eta} mic={mic:?} {s} {r}");
            }
        }
    }
}
