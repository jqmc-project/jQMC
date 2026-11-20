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

//! Range-safe complex division (issue #164).
//!
//! `num_complex`'s `a / b` is `(a * conj(b)) / |b|^2` with
//! `|b|^2 = b.re^2 + b.im^2` formed unscaled, so it underflows to zero once
//! `|b| < ~1.5e-154` (`d / d = 0 / 0 = NaN`, `1 / d = inf`) and overflows once
//! `|b| > ~1.3e154`, although the exact quotient is representable. A complex
//! geminal whose LU pivots or determinants lie outside that window (e.g. MO
//! coefficients uniformly scaled by 1e-85, `G ~ 1e-173`) then gave a NaN/inf
//! "inverse" and NaN determinant ratios in the native complex routes.
//!
//! [`div_range_safe`] brings both operands to a max-component magnitude in
//! `[0.5, 1)` by exact powers of two, forms the quotient with the same
//! `num_complex` arithmetic and restores the exponent difference by exact
//! power-of-two steps. Same design (and name) as the helper of PR #138 in
//! `complex_det.rs`; it lives in its own module so the two merge without a
//! duplicate definition.

use numpy::Complex64;

/// Power-of-two scale `2**-e` with `x = m 2**e`, `0.5 <= m < 1`
/// (`numpy.frexp`), the exponent clipped to `[-1021, 1021]`; `1` for a zero or
/// non-finite `x`.
fn pow2_scale(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return 1.0;
    }
    let (y, shift) = if x < f64::MIN_POSITIVE {
        (x * 2f64.powi(54), 54)
    } else {
        (x, 0)
    };
    let biased = ((y.to_bits() >> 52) & 0x7ff) as i32;
    let exponent = (biased - 1022 - shift).clamp(-1021, 1021);
    f64::from_bits(((1023 - exponent) as u64) << 52)
}

/// Binary exponent of a normal power of two (every `pow2_scale` value).
fn pow2_exponent(scale: f64) -> i32 {
    ((scale.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `x * 2**e` without an out-of-range intermediate: one exact power-of-two
/// multiplication when `e` is a normal exponent, else two same-sign
/// power-of-two steps (each intermediate lies between `|x|` and the result).
fn mul_pow2(x: Complex64, e: i32) -> Complex64 {
    let pow2 = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    if (-1022..=1023).contains(&e) {
        x * pow2(e)
    } else {
        let half = e / 2;
        x * pow2(half) * pow2(e - half)
    }
}

/// Range-safe complex division `a / b`.
///
/// Finite whenever the exact quotient is representable. While the unscaled
/// `num_complex` operation keeps its intermediates (`a.re b.re`, `|b|^2`, ...)
/// in the normal range, every step here is an exact power-of-two rescaling of
/// it, so the result is bit-identical to `a / b`. A zero numerator over a
/// finite non-zero `b` gives the same signed zero as `a / b` in range (and not
/// `0 / 0 = NaN` for a tiny `b`). A zero or non-finite divisor, or a
/// non-finite numerator, falls back to `a / b` (same IEEE `inf` / `NaN`
/// semantics).
pub fn div_range_safe(a: Complex64, b: Complex64) -> Complex64 {
    let a_max = a.re.abs().max(a.im.abs());
    let b_max = b.re.abs().max(b.im.abs());
    if !(a_max.is_finite() && b_max.is_finite() && b_max > 0.0) {
        return a / b;
    }
    let sb = pow2_scale(b_max);
    if a_max == 0.0 {
        return a / (b * sb);
    }
    let sa = pow2_scale(a_max);
    let q = (a * sa) / (b * sb);
    // a / b = (a sa / (b sb)) * sb / sa.
    mul_pow2(q, pow2_exponent(sb) - pow2_exponent(sa))
}
