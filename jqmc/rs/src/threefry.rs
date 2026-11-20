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
//
// Third-party material in this file:
//
// * The Threefry-2x32 implementation and the `split` / `random_bits` /
//   `randint` / `uniform` transforms are derived from JAX
//   (https://github.com/jax-ml/jax), `jax/_src/prng.py` and
//   `jax/_src/random.py` of JAX 0.7.2:
//
//     Copyright 2018 The JAX Authors.
//     Copyright 2021 The JAX Authors.
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.
//
//   They were translated from Python to Rust and modified for jQMC. The
//   Apache-2.0 text is in LICENSES/Apache-2.0.txt. The rest of this file is
//   under the BSD-3-Clause licence above.
//
// * The coefficients and regions of `erfinv` are those of XLA's `ErfInv32`
//   and `ErfInv64` (https://github.com/openxla/xla,
//   `xla/hlo/builder/lib/math.cc`), which implement the approximation of
//   M. Giles, "Approximating the erfinv function", GPU Computing Gems, Jade
//   Edition (2011):
//
//     Copyright 2018 The OpenXLA Authors.
//
//     Licensed under the Apache License, Version 2.0; see the notice above.
//     The Apache-2.0 text is in LICENSES/Apache-2.0.txt.
//
//   They were translated to Rust and modified for jQMC: `erfinv` evaluates in
//   f64, keeps the single-precision central and intermediate regions, and uses
//   the double-precision final region for `w >= 16` instead of extrapolating
//   the single-precision fit.

//! Bit-exact port of JAX's Threefry-2x32 counter-based PRNG and the
//! `split` / `random_bits` / `randint` / `uniform` transforms
//! (`jax/_src/prng.py`, `jax/_src/random.py`) for native MCMC moves.
//!
//! JAX 0.7.2 default config is `threefry_partitionable = True` -> the
//! "foldlike / partitionable" code paths, which are what this mirrors:
//!   - split(key, n):    row i = threefry_2x32(key, [hi32(i), lo32(i)])
//!   - random_bits 32:   bits1 ^ bits2
//!   - random_bits 64:   (bits1 << 32) | bits2
//!     For counts i < 2^32 (always true here) hi32(i)=0, lo32(i)=i.
//!
//! All arithmetic is u32/u64 wrapping to match XLA exactly.
//!
//! A bit-exact `normal` is deliberately NOT ported: JAX's `normal = sqrt(2)*erfinv(u)`
//! goes through XLA's `erf_inv` polynomial, which is fragile to match to the
//! last ULP. Empirical comparison showed that the XLA ErfInv64
//! algorithm itself is reproducible in pure Rust, but its `w = -Log1p(-x*x)`
//! step calls XLA CPU's OWN vectorized `log1p`/`log` (an Eigen/LLVM polynomial),
//! which is NOT bit-identical to the system libm used by Rust/numpy: measured
//! ~12% of points differ by 1 ULP for `log1p` (and ~1% even for plain `log`),
//! and `jax.log1p != kahan(jax.log)` -- so XLA's log1p is its own op and is
//! backend-dependent (CPU vs GPU differ). A 1-ULP proposal perturbation flips a
//! downstream Metropolis accept and blows the chain apart, so an "approximate"
//! native normal is NOT chain-identical. The `bitexact` chain mode therefore keeps
//! the Gaussian proposal as injected JAX-generated normals (`_draw_injected_normals`), while every
//! discrete decision (electron/axis selection, Metropolis accept) is bit-exact
//! native.
//!
//! The MCMC per-measurement ECP-mesh RT rotation (`_generate_rotation_matrix`) is
//! likewise NOT natively bit-exact, for the same XLA-fp reason: the Haar map uses
//! `asin` plus `cos`/`sin`, and JAX contracts the uniform scaling and matrix expressions
//! (`cg*sa*sb - ca*sg`, ...) into fused multiply-adds, which round differently than
//! Rust's separate mul/add. FMA contraction and transcendental lowering are backend/LLVM-flag
//! dependent, so the MCMC RT, like the normals, is injected from JAX per measurement
//! through `_jit_vmap_generate_RTs`. GFMC/LRDMC instead samples the same Haar law
//! natively; its contract is numerical XLA/Rust parity, not bit identity.

const PARITY: u32 = 0x1BD1_1BDA;
const ROT0: [u32; 4] = [13, 15, 26, 6];
const ROT1: [u32; 4] = [17, 29, 16, 24];

#[inline(always)]
fn rotl(x: u32, d: u32) -> u32 {
    x.rotate_left(d)
}

#[inline(always)]
fn round(mut x0: u32, mut x1: u32, r: u32) -> (u32, u32) {
    x0 = x0.wrapping_add(x1);
    x1 = rotl(x1, r);
    x1 ^= x0;
    (x0, x1)
}

/// Core Threefry-2x32 hash. `key` and `count` are pairs of u32.
/// 1:1 with `_threefry2x32_lowering(use_rolled_loops=False)`.
pub fn threefry_2x32(key: [u32; 2], count: [u32; 2]) -> [u32; 2] {
    let ks = [key[0], key[1], key[0] ^ key[1] ^ PARITY];
    let mut x0 = count[0].wrapping_add(ks[0]);
    let mut x1 = count[1].wrapping_add(ks[1]);

    // The 5 key-injection blocks alternate ROT0/ROT1 and add a rotating
    // (ks, ks+counter) pair after each block.
    let blocks: [([u32; 4], usize, usize, u32); 5] = [
        (ROT0, 1, 2, 1),
        (ROT1, 2, 0, 2),
        (ROT0, 0, 1, 3),
        (ROT1, 1, 2, 4),
        (ROT0, 2, 0, 5),
    ];
    for (rots, a, b, ctr) in blocks {
        for r in rots {
            let (n0, n1) = round(x0, x1, r);
            x0 = n0;
            x1 = n1;
        }
        x0 = x0.wrapping_add(ks[a]);
        x1 = x1.wrapping_add(ks[b]).wrapping_add(ctr);
    }
    [x0, x1]
}

/// `jax.random.split(key, n)` (partitionable path).
pub fn split(key: [u32; 2], n: usize) -> Vec<[u32; 2]> {
    (0..n).map(|i| threefry_2x32(key, [0, i as u32])).collect()
}

/// `jax.random.fold_in(key, data)` = threefry_2x32(key, threefry_seed(data)),
/// and threefry_seed(u32 data) = [0, data].
pub fn fold_in(key: [u32; 2], data: u32) -> [u32; 2] {
    threefry_2x32(key, [0, data])
}

/// `random_bits(key, 32, (n,))` (partitionable): `bits1 ^ bits2`.
pub fn random_bits_32(key: [u32; 2], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let b = threefry_2x32(key, [0, i as u32]);
            b[0] ^ b[1]
        })
        .collect()
}

/// `random_bits(key, 64, (n,))` (partitionable): `(bits1 << 32) | bits2`.
pub fn random_bits_64(key: [u32; 2], n: usize) -> Vec<u64> {
    (0..n)
        .map(|i| {
            let b = threefry_2x32(key, [0, i as u32]);
            ((b[0] as u64) << 32) | (b[1] as u64)
        })
        .collect()
}

/// `jax.random.uniform(key, (n,), float64, minval, maxval)`.
/// Mantissa-fill of an exponent-1 double, then shift/scale; matches `_uniform`.
pub fn uniform_f64(key: [u32; 2], n: usize, minval: f64, maxval: f64) -> Vec<f64> {
    random_bits_64(key, n)
        .into_iter()
        .map(|bits| {
            // float_bits = (bits >> (64-52)) | bits_of(1.0)
            let float_bits = (bits >> 12) | 0x3FF0_0000_0000_0000u64;
            let u = f64::from_bits(float_bits) - 1.0; // in [0, 1)
            (u * (maxval - minval) + minval).max(minval)
        })
        .collect()
}

/// `jax.random.randint(key, (n,), minval, maxval, int32)`. Bit-exact with
/// `_randint` (nbits=32): split, two 32-bit draws, biased-modulus reduction.
pub fn randint_i32(key: [u32; 2], n: usize, minval: i32, maxval: i32) -> Vec<i32> {
    let parts = split(key, 2);
    let (k1, k2) = (parts[0], parts[1]);
    let higher = random_bits_32(k1, n);
    let lower = random_bits_32(k2, n);

    // span = maxval - minval as u32 (assume maxval > minval, in-range int32).
    let mut span = (maxval as i64 - minval as i64) as u32;
    if maxval <= minval {
        span = 1;
    }
    // multiplier = (2^16 % span)^2 % span
    let m0 = (1u32 << 16) % span;
    let multiplier = m0.wrapping_mul(m0) % span;

    higher
        .iter()
        .zip(lower.iter())
        .map(|(&hi, &lo)| {
            // random_offset = ((hi % span) * mult + (lo % span)) % span, u32 wrapping
            let ro = (hi % span).wrapping_mul(multiplier).wrapping_add(lo % span) % span;
            (minval as i64 + ro as i64) as i32
        })
        .collect()
}

/// `jax.random.randint(key, (n,), minval, maxval)` with the **default int
/// dtype** = int64 under `jax_enable_x64` (nbits=64). This is what the jqmc MCMC
/// body uses (`randint(subkey, shape=(), minval=0, maxval=...)` with no dtype).
/// Same algorithm as `randint_i32` but with 64-bit draws and a 2^32 multiplier.
pub fn randint_i64(key: [u32; 2], n: usize, minval: i64, maxval: i64) -> Vec<i64> {
    let parts = split(key, 2);
    let higher = random_bits_64(parts[0], n);
    let lower = random_bits_64(parts[1], n);

    let mut span = (maxval - minval) as u64;
    if maxval <= minval {
        span = 1;
    }
    // multiplier = (2^32 % span)^2 % span
    let m0 = (1u64 << 32) % span;
    let multiplier = m0.wrapping_mul(m0) % span;

    higher
        .iter()
        .zip(lower.iter())
        .map(|(&hi, &lo)| {
            let ro = (hi % span).wrapping_mul(multiplier).wrapping_add(lo % span) % span;
            minval + ro as i64
        })
        .collect()
}

// ===========================================================================
// NATIVE Gaussian proposal + ECP rotation (PRODUCTION "fast" chain mode).
//
// The `bitexact` chain mode injects JAX-generated normals / RTs (the erfinv +
// FMA-contraction ULP traps documented at the top of this file). The `fast`
// mode instead generates BOTH natively from the SAME bit-exact threefry
// `split`/`uniform` stream above (so the key chain -- and hence the discrete
// electron/axis/Metropolis draws the sweep reproduces -- stays aligned with the
// per-step path), applying a NATIVE uniform->normal transform. That transform
// (erfinv below) is deliberately NOT XLA-bit-identical: `fast` mode is a
// production DEFAULT checked pointwise against an independent inverse function
// on specified inputs, not pinned to the JAX chain. Deterministic driver parity
// injects identical proposal inputs through the test-only seam.
// ===========================================================================

/// `nextafter(-1.0, 0.0)` = the `minval` JAX's `_normal_real` feeds `uniform`
/// (open lower bound so `erfinv` never hits `-inf`). Bits: sign|exp=1022|mant=1s.
pub const NORMAL_LO: f64 = f64::from_bits(0xBFEF_FFFF_FFFF_FFFF);

/// Native inverse error function using Giles' piecewise polynomials in f64.
/// Keep the original central/intermediate arithmetic; use the double-precision
/// final region for w >= 16 instead of extrapolating the single-precision fit
/// to fp64 endpoints. Coefficients/region: Giles, "Approximating the erfinv
/// function", https://people.maths.ox.ac.uk/~gilesm/codes/erfinv/ (DP_1).
/// Numerical inverse-function agreement, not XLA bit identity, is the contract.
pub fn erfinv(x: f64) -> f64 {
    // Clamp into the open interval so the endpoints stay finite.
    let x = x.clamp(-0.999_999_999_999_999_9, 0.999_999_999_999_999_9);
    let mut w = -((1.0 - x) * (1.0 + x)).ln();
    let mut p;
    if w < 5.0 {
        w -= 2.5;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 2.185_808_7e-04 + p * w;
        p = -1.253_725_03e-03 + p * w;
        p = -4.177_681_64e-03 + p * w;
        p = 2.466_407_27e-01 + p * w;
        p = 1.501_409_41e+00 + p * w;
    } else if w < 16.0 {
        w = w.sqrt() - 3.0;
        p = -2.002_142_57e-04;
        p = 1.009_505_58e-04 + p * w;
        p = 1.349_343_22e-03 + p * w;
        p = -3.673_428_44e-03 + p * w;
        p = 5.739_507_73e-03 + p * w;
        p = -7.622_461_3e-03 + p * w;
        p = 9.438_870_47e-03 + p * w;
        p = 1.001_674_06e+00 + p * w;
        p = 2.832_976_82e+00 + p * w;
    } else {
        w = w.sqrt() - 5.0;
        p = -2.710_992_061_643_857_3e-11;
        p = -2.555_641_816_996_525e-10 + p * w;
        p = 1.507_657_269_350_054_8e-09 + p * w;
        p = -3.789_465_440_126_737e-09 + p * w;
        p = 7.615_701_208_078_34e-09 + p * w;
        p = -1.496_002_662_714_924e-08 + p * w;
        p = 2.914_795_345_090_108e-08 + p * w;
        p = -6.771_199_775_845_234e-08 + p * w;
        p = 2.290_048_222_802_665_5e-07 + p * w;
        p = -9.929_827_294_231_7e-07 + p * w;
        p = 4.526_062_597_223_154e-06 + p * w;
        p = -1.968_177_810_553_167e-05 + p * w;
        p = 7.599_527_703_001_776e-05 + p * w;
        p = -2.150_301_193_004_447_7e-04 + p * w;
        p = -1.387_193_183_362_312_3e-04 + p * w;
        p = 1.010_300_464_864_534_4 + p * w;
        p = 4.849_906_401_408_584 + p * w;
    }
    p * x
}

/// One native `N(0,1)` draw from a threefry subkey, mirroring JAX's
/// `normal(subkey, ()) = sqrt(2) * erfinv(uniform(subkey, (), lo, 1))` but with
/// the native `erfinv`. The `uniform` half is the bit-exact `uniform_f64`, so
/// the ONLY non-XLA-exact step is the numerically checked erfinv approximation.
pub fn native_normal(subkey: [u32; 2]) -> f64 {
    let u = uniform_f64(subkey, 1, NORMAL_LO, 1.0)[0];
    std::f64::consts::SQRT_2 * erfinv(u)
}

/// Native Gaussian-proposal normals for one sweep step of `num_moves` moves,
/// derived from the SAME threefry key chain the native sweep threads: each move
/// consumes 6 `split`s (electron up/dn choice, up index, dn index, **Gaussian
/// proposal**, axis, Metropolis-b), the 4th subkey being the proposal. Only the
/// proposal subkey is turned into a normal; the other five advance the key so
/// `key_after` == the sweep's `final_key` for the same start key (the advance is
/// outcome-independent). Returns `(normals[num_moves], key_after)`.
pub fn native_step_normals(key0: [u32; 2], num_moves: usize) -> (Vec<f64>, [u32; 2]) {
    let mut key = key0;
    let mut out = Vec::with_capacity(num_moves);
    for _ in 0..num_moves {
        // three discrete-choice splits (values unused here; only advance key)
        for _ in 0..3 {
            key = split(key, 2)[0];
        }
        // 4th split = the Gaussian-proposal subkey
        let s = split(key, 2);
        key = s[0];
        out.push(native_normal(s[1]));
        // axis + Metropolis-b splits
        for _ in 0..2 {
            key = split(key, 2)[0];
        }
    }
    (out, key)
}

/// Native Haar random rotation matrix, mirroring jqmc's `_generate_rotation_matrix`:
/// `_, subkey = split(key)`; draw alpha/gamma uniform modulo 2pi and sin(beta)
/// uniform on [-1,1], then build `Rz(gamma) Ry(beta) Rx(alpha)` and return
/// `R.T`. The uniform draw is bit-exact; the
/// cos/sin + matrix assembly are native libm (statistically a valid random
/// rotation, not XLA-bit-identical -- consistent with `fast` mode).
pub fn native_rotation_matrix(key: [u32; 2]) -> [[f64; 3]; 3] {
    let subkey = split(key, 2)[1];
    let u = uniform_f64(subkey, 3, 0.0, 1.0);
    let alpha = std::f64::consts::TAU * u[0];
    let beta = (2.0 * u[1] - 1.0).asin();
    let gamma = std::f64::consts::TAU * u[2];
    let (ca, sa) = (alpha.cos(), alpha.sin());
    let (cb, sb) = (beta.cos(), beta.sin());
    let (cg, sg) = (gamma.cos(), gamma.sin());
    // R rows exactly as _generate_rotation_matrix (before the .T).
    let r = [
        [cb * cg, cg * sa * sb - ca * sg, sa * sg + ca * cg * sb],
        [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
        [-sb, cb * sa, ca * cb],
    ];
    // return R.T : rt[i][j] = r[j][i]
    let mut rt = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rt[i][j] = r[j][i];
        }
    }
    rt
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known vectors captured from JAX 0.7.2.
    #[test]
    fn threefry_known_vectors() {
        assert_eq!(threefry_2x32([0, 0], [0, 0]), [1797259609, 2579123966]);
        assert_eq!(
            threefry_2x32([0x13198a2e, 0x03707344], [0x243f6a88, 0x85a308d3]),
            [3297917596, 1212020640]
        );
    }

    #[test]
    fn split_known_vector() {
        let s = split([0, 0], 2);
        assert_eq!(s[0], [1797259609, 2579123966]);
        assert_eq!(s[1], [928981903, 3453687069]);
    }

    // SciPy special.erfinv references at exact fp64 inputs. Neighbors straddle
    // the two region boundaries; powers of two reach nextafter(1, 0). Both
    // signs are checked directly because forward erf saturates in this tail.
    #[test]
    fn erfinv_known_values() {
        // These point checks keep their own budgets; the table below adds
        // coverage and does not relax them.
        assert!(erfinv(0.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
        assert!(
            (erfinv(0.5) - 0.476_936_276_204).abs()
                < crate::setting_consts::ATOL_RS_ERFINV_LEGACY_CORE
        );
        assert!(
            (erfinv(-0.5) + 0.476_936_276_204).abs()
                < crate::setting_consts::ATOL_RS_ERFINV_LEGACY_CORE
        );
        assert!(
            (erfinv(0.9) - 1.163_087_153_676).abs()
                < crate::setting_consts::ATOL_RS_ERFINV_LEGACY_CORE
        );
        assert!(
            (erfinv(0.99) - 1.821_386_367_775).abs()
                < crate::setting_consts::ATOL_RS_ERFINV_LEGACY_TAIL
        );
        assert!(
            (erfinv(-0.99) + 1.821_386_367_775).abs()
                < crate::setting_consts::ATOL_RS_ERFINV_LEGACY_TAIL
        );
        let atol = crate::central_setting_f64("atol_consistency");
        let rtol = crate::central_setting_f64("rtol_consistency");
        let cases: [(f64, f64); 16] = [
            (0.0, 0.0),
            (0.5, 0.476_936_276_204_469_9),
            (0.9, 1.163_087_153_676_674_3),
            (0.99, 1.821_386_367_718_449_6),
            (f64::from_bits(0x3fef_e45a_cf14_4b66), 2.072_794_083_941_318),
            (
                f64::from_bits(0x3fef_e45a_cf14_4b67),
                2.072_794_083_941_325_5,
            ),
            (
                f64::from_bits(0x3fef_e45a_cf14_4b68),
                2.072_794_083_941_332_6,
            ),
            (f64::from_bits(0x3fef_ffff_e1ca_a436), 3.839_783_581_154_581),
            (
                f64::from_bits(0x3fef_ffff_e1ca_a437),
                3.839_783_581_403_561_5,
            ),
            (f64::from_bits(0x3fef_ffff_e1ca_a438), 3.839_783_581_652_542),
            (
                f64::from_bits(0x3fef_ffff_e000_0000),
                3.832_506_856_900_710_4,
            ),
            (f64::from_bits(0x3fef_ffff_ff80_0000), 4.328_028_275_800_946),
            (f64::from_bits(0x3fef_ffff_ffff_e000), 5.051_254_085_249_39),
            (f64::from_bits(0x3fef_ffff_ffff_ffe0), 5.564_773_326_996_244),
            (f64::from_bits(0x3fef_ffff_ffff_fffe), 5.805_018_683_193_453),
            (
                f64::from_bits(0x3fef_ffff_ffff_ffff),
                5.863_584_748_755_167_6,
            ),
        ];
        for (input, reference) in cases {
            for sign in [-1.0, 1.0] {
                let actual = erfinv(sign * input);
                assert!(
                    (actual - sign * reference).abs() <= atol + rtol * reference.abs(),
                    "inverse value mismatch: input={}, actual={actual}, reference={}",
                    sign * input,
                    sign * reference,
                );
            }
        }
    }

    fn assert_native_normal_inverse(normal: impl Fn([u32; 2]) -> f64) {
        let atol = crate::central_setting_f64("atol_consistency");
        let rtol = crate::central_setting_f64("rtol_consistency");
        // One million subkeys split from a fixed key. libm's
        // forward erf, not native's inverse polynomial, verifies each mapping.
        // The Python GDD anchor independently verifies split/uniform with JAX
        // and output values with SciPy on this key and the original Python key.
        for subkey in split([1234567, 890123], 1_000_000) {
            let input = uniform_f64(subkey, 1, NORMAL_LO, 1.0)[0];
            let value = normal(subkey);
            assert!(value.is_finite(), "Gaussian result must be finite");
            let recovered = libm::erf(value / std::f64::consts::SQRT_2);
            assert!(
                (recovered - input).abs() <= atol + rtol * input.abs(),
                "Gaussian inverse-transform mismatch: input={input}, value={value}"
            );
        }
    }

    #[test]
    fn native_normal_inverse_transform() {
        assert_native_normal_inverse(native_normal);
    }

    #[test]
    #[should_panic(expected = "Gaussian inverse-transform mismatch")]
    fn native_normal_rejects_sign_flip() {
        assert_native_normal_inverse(|key| -native_normal(key));
    }

    #[test]
    #[should_panic(expected = "Gaussian inverse-transform mismatch")]
    fn native_normal_rejects_scale_error() {
        assert_native_normal_inverse(|key| 2.0 * native_normal(key));
    }

    #[test]
    #[should_panic(expected = "Gaussian inverse-transform mismatch")]
    fn native_normal_rejects_untransformed_uniform() {
        assert_native_normal_inverse(|key| uniform_f64(key, 1, NORMAL_LO, 1.0)[0]);
    }
}
