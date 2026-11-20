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

//! 1:1 Rust port of the *molecular* (kernel-rank) GTO AO value evaluation,
//! mirroring `_compute_AOs_cart_debug` (atomic_orbital.py) and
//! `_compute_AOs_sphe_debug` (atomic_orbital.py), plus the real solid
//! harmonic `_compute_S_l_m_debug` (atomic_orbital.py).
//!
//! The molecular path has `_num_ao_kernel == num_ao` and no image folding;
//! periodic pGTO leaves additionally expand images and apply Bloch phases.
//! These are pure value functions without autodiff.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// `(value, grad_x, grad_y, grad_z, laplacian)` orbital tables, each `(num_orb, N)`
/// for the default `Array2` storage.
pub(crate) type ValueGradLap<A = Array2<f64>> = (A, A, A, A, A);

/// Angular kind of a GTO basis.
#[derive(Clone, Copy, PartialEq)]
pub enum AoKind {
    Cart,
    Sphe,
}

/// Radial screening exponent, `exp(-LEPSBAS) ~= 1e-9`. Every screen tests
/// `pz * r^2 > LEPSBAS` with a per-shell effective exponent `pz`; what that
/// guarantees depends on the exponent:
///
/// PERIODIC image screens (`pz_min*r^2 > LEPSBAS -> skip`): the shell's slowest
/// primitive has decayed by `exp(-LEPSBAS)` RELATIVE TO ITS OWN PEAK factor
/// `|norm*coef|`. This is not an absolute bound on the dropped AO value: the
/// dropped value is `~ |norm*coef| * r^l * exp(-LEPSBAS)`, and `norm` grows as
/// `zeta^(3/4 + l/2)` for tight primitives.
///
/// FAST-mode molecular VALUE screens (ECP mesh AO contraction
/// `dot_aos_*_cutoff_*`, per-move orbital `compute_mo_single_cutoff*_into`,
/// GFMC molecular candidate batch) use the per-shell `pz_val` instead
/// ([`Shell::pz_val`]), chosen so that every dropped AO VALUE of the shell is
/// below [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in ABSOLUTE AO units (issue #100;
/// [`value_cut_exponent`]).
///
/// FAST-mode DERIVATIVE screens (`derivative_alive`, see [`Shell::pz_deriv`])
/// use the per-shell `pz_deriv` instead, chosen so that every dropped gradient
/// component and Laplacian of every AO of the shell is below
/// [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in ABSOLUTE AO units (issue #87,
/// [`derivative_cut_exponent`]).
pub(crate) use crate::setting_consts::NATIVE_AO_SCREEN_EXPONENT as LEPSBAS; // = -ln(1e-9)

/// Absolute bound on every AO value (FAST value screens, [`Shell::pz_val`]) and
/// every AO gradient component / Laplacian (FAST derivative screens,
/// [`Shell::pz_deriv`]) that a FAST-mode molecular screen may drop, in absolute
/// AO units (normalization and contraction coefficients included).
///
/// The bound is 1e-12, not `exp(-LEPSBAS) ~= 1e-9` (issue #100).
/// A per-AO bound is amplified in the local energy by the number of dropped AOs
/// per electron, the MO coefficients and the inverse Slater matrix: with a 1e-9
/// bound for both screens, a linear (H2O)8 chain (ccECP cc-pVDZ) kept a FAST
/// e_L error of median 4e-7 Ha per walker; with 1e-12 it is ~1e-9.
pub(crate) use crate::setting_consts::FAST_AO_SCREEN_ABS_BOUND as FAST_SCREEN_ABS_BOUND;

/// Kernel-rank GTO AO data (1:1 with `AOs_{cart,sphe}_gto_data` fields used by
/// the value evaluator).
#[derive(Clone)]
pub struct AosGtoData {
    pub kind: AoKind,
    pub num_ao_kernel: usize,
    /// Molecular AO count. For molecular leaves `num_ao == num_ao_kernel`;
    /// for pGTO leaves `num_ao_kernel == n_image * num_ao` and the kernel
    /// output is image-folded back to `num_ao` rows.
    pub num_ao: usize,
    /// Number of lattice images summed (1 for molecular).
    pub n_image: usize,
    /// Boundary metadata for spherical prefactors, independent of precision policy.
    pub molecular_sphe: bool,
    /// Arithmetic policy captured when this leaf was built: the
    /// owning `NativeObsCtx` policy inside `build_obs_ctx`, or the explicitly
    /// configured standalone policy for a free `jqmc_rs` kernel.  Every value
    /// kernel reads its `value_zone` from HERE, never from the process-wide
    /// standalone setting, so a context's routes cannot depend on whether
    /// `configure_native_precision` was ever called.
    pub(crate) precision_policy: crate::PrecisionPolicy,
    /// AO-owned value zone, including guarded complex Bloch GTO values.
    pub(crate) value_zone: &'static str,
    /// AO centers, shape `(num_ao_kernel, 3)` (= `_atomic_center_carts_np`).
    pub centers: Array2<f64>,
    pub angular_momentums: Vec<i64>,
    /// Per-primitive AO index (`orbital_indices`), length `num_prim`.
    pub orbital_indices: Vec<usize>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    /// Per-AO multiplier from a shared radial coefficient to the effective
    /// component coefficient. An empty vector denotes the legacy unit scale.
    pub ao_coefficient_scales: Vec<f64>,
    // Cartesian-only: polynomial orders per AO.
    pub poly_x: Vec<i64>,
    pub poly_y: Vec<i64>,
    pub poly_z: Vec<i64>,
    // Spherical-only: magnetic quantum number per AO.
    pub magnetic: Vec<i64>,
    // --- Precomputed once (electron-independent), filled by `finalize()`. ---
    /// Per-primitive normalization `N_p` (length = num primitives).
    pub normalizations: Vec<f64>,
    /// Primitive indices grouped per AO (length = num_ao_kernel).
    pub prim_groups: Vec<Vec<usize>>,
    /// Precomputed solid-harmonic angular evaluator per AO (sphe only; empty for cart).
    /// Removes the binom/factorial recompute from the per-electron hot loop.
    pub sphe_angular: Vec<SpheAngular>,
    /// Precomputed angular value+grad tables per AO (sphe only; empty for cart).
    /// The grad/lap sibling of `sphe_angular`: hoists the binom/factorial/trig
    /// recompute out of `s_l_m_value_and_grad` (bit-identical replay).
    pub sphe_ang_vg: Vec<SpheAngVG>,
    // --- Tight flat per-AO primitive layout (contiguous, no indirection) ---
    /// `norm_p * coef_p`, reordered so each AO's primitives are contiguous.
    pub flat_nc: Vec<f64>,
    /// Exponents, reordered to match `flat_nc`.
    pub flat_exp: Vec<f64>,
    /// Per-AO primitive range start; length `num_ao_kernel + 1`.
    pub ao_prim_offset: Vec<usize>,
    /// Shell groups (sphe only): AOs sharing center+l+radial -> exp computed
    /// once per shell and reused across the 2l+1 angular components.
    pub shells: Vec<Shell>,
    // --- Flat shell-primitive layout for the vectorized (AVX-512) exp path ---
    /// Shell centers, `3*nshell` contiguous.
    pub sh_center: Vec<f64>,
    /// Prim range per shell into `sh_neg_pz`/`sh_pnc`; length `nshell+1`.
    pub sh_off: Vec<u32>,
    /// `-Z_p` per shell primitive (shell-contiguous).
    pub sh_neg_pz: Vec<f64>,
    /// `norm*coef` per shell primitive (shell-contiguous).
    pub sh_pnc: Vec<f64>,
    /// Per-primitive FAST value-screen exponent (shell-contiguous, aligned with
    /// `sh_neg_pz`), [`Shell::prim_val`].
    pub sh_pzprim: Vec<f64>,
    /// Smallest exponent per shell (length nshell) -- the per-image cutoff key.
    pub sh_pzmin: Vec<f64>,
    /// FAST value-screen exponent per shell (length nshell), [`Shell::pz_val`].
    pub sh_pzval: Vec<f64>,
    /// AO range per shell into `sh_ao_kernel`/`sh_ao_home`; length nshell+1.
    pub sh_ao_off: Vec<u32>,
    /// Kernel AO index per (shell,ao) -- for the `sphe_angular` lookup.
    pub sh_ao_kernel: Vec<u32>,
    /// Image-folded home AO index per (shell,ao), from [`AosGtoData::kernel_to_home`].
    pub sh_ao_home: Vec<u32>,
    /// kernel-AO -> home-AO fold map, length `num_ao_kernel`. Derived from the
    /// per-shell ragged layout; identity for molecular
    /// (`num_ao_kernel == num_ao`). A uniform `kernel_ao % num_ao` would be valid
    /// only for `n_image x num_ao` tiling.
    pub kernel_to_home: Vec<usize>,
    /// Inverse of `kernel_to_home` in CSR form, built in `finalize`: the kernel AOs
    /// of home AO `h` are `home_kernel[home_kernel_off[h]..home_kernel_off[h + 1]]`,
    /// in ascending kernel order (the term order of the fp32 adjacent-tree image
    /// fold). Empty for un-finalized data.
    pub(crate) home_kernel_off: Vec<usize>,
    pub(crate) home_kernel: Vec<usize>,
    // --- Flat solid-harmonic angular tables (sphe), per kernel AO. Contiguous
    //     so the per-move AO eval streams instead of chasing per-AO heap Vecs. ---
    /// `|m|` per AO.
    pub ang_mabs: Vec<i32>,
    /// m-coeff range per AO into `ang_mc`; length `num_ao_kernel+1`.
    pub ang_mc_off: Vec<u32>,
    /// `mc[p]` flat (the x^p*y^(|m|-p) coefficients).
    pub ang_mc: Vec<f64>,
    /// lambda-term range per AO into `ang_lam_*`; length `num_ao_kernel+1`.
    pub ang_lam_off: Vec<u32>,
    /// lambda coefficient / r^2-power / z-power, flat.
    pub ang_lam_c: Vec<f64>,
    pub ang_lam_k: Vec<i32>,
    pub ang_lam_ez: Vec<i32>,
    // --- Periodicity guarantor (real-Gamma pGTO). ---
    /// Home-cell lattice vectors (rows), `Some` only for periodic leaves
    /// (`structure_data.pbc_flag`, whatever the retained image count). jqmc wraps every electron coordinate into the home cell
    /// (`_wrap_into_cell_if_pbc`) BEFORE evaluating the AO, because its truncated
    /// image sum is enumerated relative to the home cell; a raw coordinate several
    /// lattice vectors outside falls beyond the static image tiling and the sum
    /// misses the dominant images (grad/lap/kinetic collapse). The static image
    /// centres already carry the SAME `T_0` the wrap removes, so wrapping restores
    /// bit-agreement with jqmc at ALL configs. `None` for molecular (no-op).
    pub wrap_cell: Option<[[f64; 3]; 3]>,
    /// `inv(wrap_cell)`; `r_frac = r @ wrap_inv`. Meaningful only when `wrap_cell`
    /// is `Some`.
    pub wrap_inv: [[f64; 3]; 3],
    /// Cartesian shell families (images/atoms of one radial shell type), built
    /// in `finalize`; empty for sphe. See [`CartFamily`].
    pub cart_fam: Vec<CartFamily>,
    /// Per-primitive Cartesian fp64 normalization
    /// `primitive_normalization_cart_fused_f64(p, ao(p))`, built in `finalize`
    /// (empty for sphe / un-finalized data, which then compute it on the fly).
    /// The molecular `ao_eval` kernel rounds it once to fp32; the periodic one
    /// keeps it at storage precision (issue #718).
    pub(crate) cart_norm_fused_f64: Vec<f64>,
}

/// Cartesian shells that share `l`, exponents, per-component `norm*coef`,
/// polynomial orders and value-screen exponent, differing only by center (the
/// periodic images and equivalent atoms of one basis shell). Kernels loop over
/// the members, which gives long uniform loops instead of one short loop per
/// shell.
#[derive(Clone, Default)]
pub struct CartFamily {
    pub prim_z: Vec<f64>,
    /// `norm*coef`, component-major: `nc[j * nprim + q]`.
    pub nc: Vec<f64>,
    /// Polynomial orders per component.
    pub poly: Vec<[usize; 3]>,
    pub pz_val: f64,
    /// Per-primitive value-screen exponents ([`Shell::prim_val`]; equal for all
    /// members, which share exponents and coefficients).
    pub prim_val: Vec<f64>,
    /// Member centers (SoA).
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    pub cz: Vec<f64>,
    /// Home AO of component `j` of member `m`: `home[j * n_member + m]`.
    pub home: Vec<u32>,
}

/// A shell of spherical AOs sharing the same center and radial part. The radial
/// sum `Sum (norm*coef)*exp(-Z*r^2)` is identical for all AOs in the shell, so it is
/// evaluated once and multiplied by each AO's angular part.
#[derive(Clone)]
pub struct Shell {
    pub center: [f64; 3],
    pub l: i64,
    pub prim_z: Vec<f64>,
    pub prim_c: Vec<f64>,
    pub prim_nc: Vec<f64>, // norm * coef per shell primitive
    pub aos: Vec<usize>,   // kernel-rank AO indices in this shell
    /// Smallest exponent in the shell (slowest-decaying primitive). Used for the
    /// per-image radial cutoff `pz_min*r^2 > LEPSBAS => shell ~= 0` so far periodic
    /// images are skipped per-shell (TurboRVB-style adaptive image truncation;
    /// XLA can't do this -- its static shape forces a uniform image count).
    pub pz_min: f64,
    /// Effective exponent of the FAST-mode derivative screen
    /// (`derivative_alive = pz_deriv * r^2 <= LEPSBAS`). Chosen once per shell by
    /// [`derivative_cut_exponent`] so that, wherever the screen drops a shell, every
    /// gradient component and the Laplacian of every AO of the shell are below
    /// [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in absolute AO units (normalization and contraction
    /// coefficients included). Always `<= pz_min`: the derivative cut radius is
    /// never smaller than the plain `pz_min` one. Equals `pz_min` when that cut
    /// already meets the bound (diffuse / weakly normalized shells) and
    /// `l <= 39`; for larger `l` the cut is never inside the Gaussian-polynomial
    /// turning point `r^2 = (l+2) / (2 pz_min)`.
    pub pz_deriv: f64,
    /// Effective exponent of the FAST-mode VALUE screens
    /// (`pz_val * r^2 > LEPSBAS -> shell skipped`; the ECP mesh AO contraction,
    /// the per-move orbital evaluation and the GFMC molecular candidate batch).
    /// Chosen once per shell by [`value_cut_exponent`] so that, wherever a value
    /// screen drops the shell, the value of every AO of the shell is below
    /// [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in absolute AO units (normalization and
    /// contraction coefficients included; issue #100). Always `<= pz_min`, i.e.
    /// never a tighter cut than the plain `pz_min` one; `0.0` (screen disabled)
    /// for a non-finite / non-positive exponent.
    pub pz_val: f64,
    /// Per-primitive FAST value-screen exponents (one per `prim_z`), from
    /// [`prim_value_cut_exponents`]: inside a kept shell, primitive `q` is dropped
    /// at a point where `prim_val[q] * r^2 > LEPSBAS`. The dropped primitives of
    /// one AO add up to less than [`FAST_SCREEN_ABS_BOUND`] in absolute AO units,
    /// so each AO value still changes by less than the bound. Used only by the ECP
    /// mesh contractions `dot_aos_multi_cutoff_{sphe,cart}`.
    pub prim_val: Vec<f64>,
}

/// Per-primitive FAST value-screen exponents of one shell (see [`Shell::prim_val`]).
///
/// With the notation of [`value_cut_exponent`], primitive `q` contributes at most
/// `Pm r^l A_q exp(-zeta_q r^2)` to any AO of the shell. Each exponent is the
/// [`value_cut_exponent`] of that single term with its coefficient bound raised
/// to `nq * A_q` (`nq` = primitives in the shell), so every primitive dropped by
/// `prim_val[q] * r^2 > LEPSBAS` contributes less than
/// `FAST_SCREEN_ABS_BOUND / nq` and all dropped primitives of an AO together
/// less than [`FAST_SCREEN_ABS_BOUND`]. Each value is `<= zeta_q`; `0.0` (never
/// screen) for a non-finite / non-positive exponent or coefficient.
pub(crate) fn prim_value_cut_exponents(
    kind: AoKind,
    l: i64,
    prim_z: &[f64],
    amax: &[f64],
) -> Vec<f64> {
    let nq = prim_z.len() as f64;
    prim_z
        .iter()
        .zip(amax)
        .map(|(&z, &a)| value_cut_exponent(kind, l, &[z], &[a * nq], z))
        .collect()
}

/// Effective exponent `pz_deriv` of the FAST-mode derivative screen of one shell.
///
/// A shell of angular momentum `l` evaluates AOs `phi = P(x,y,z) g(r^2)` with
/// `g = sum_q a_q exp(-zeta_q r^2)` (`a_q` = that AO's `norm*coef`) and a
/// homogeneous degree-`l` angular polynomial `P` (Cartesian monomial, or
/// `sqrt((2l+1)/4pi)` times the Racah-normalized real solid harmonic). With
/// `r.grad P = l P`:
///   grad phi = g grad P - 2 r_vec P sum_q zeta_q a_q e_q
///   lap  phi = g lap P  + P sum_q a_q e_q (4 zeta_q^2 r^2 - (4l+6) zeta_q)
/// so, with `A_q = max over the shell's AOs of |a_q|` (mixed-sign contractions
/// bounded term by term),
///   |d phi / dx_i| <= sum_q A_q e_q [G r^(l-1) + 2 zeta_q Pm r^(l+1)]
///   |lap phi|      <= sum_q A_q e_q [L r^(l-2) + (4 zeta_q^2 r^2 + (4l+6) zeta_q) Pm r^l]
/// where `|P| <= Pm r^l`, `|grad P| <= G r^(l-1)`, `|lap P| <= L r^(l-2)`:
/// Cartesian `Pm = 1, G = l, L = l(l-1)` (`|x_i| <= r`); spherical
/// `Pm = s, G = s sqrt(l(2l+1)), L = 0` with `s = sqrt((2l+1)/4pi)` (from
/// `sum_m S_lm^2 = r^(2l)` and `sum_m |grad S_lm|^2 = l(2l+1) r^(2l-2)` for the
/// Racah normalization; `S_lm` is harmonic).
///
/// Returns `LEPSBAS / r2_cut`, where `r2_cut` is (to bisection accuracy,
/// rounded outward) the smallest `r^2 >= r2_mono` at which both bounds are
/// below [`FAST_SCREEN_ABS_BOUND`], with
///   r2_mono = max(LEPSBAS, (l+2)/2) / zeta_min.
/// Every term of either bound is a non-negative constant times
/// `(r^2)^p exp(-zeta_q r^2)` with `p <= (l+2)/2`, which decreases
/// monotonically for `r^2 >= p / zeta_q`; past `r2_mono` every term, and hence
/// the bound, is therefore non-increasing, so the first `r^2 >= r2_mono` below
/// the threshold is a valid cut for all larger distances. The `(l+2)/2` term
/// is what keeps this valid for any accepted angular momentum: for `l <= 39`
/// it is below `LEPSBAS` and `r2_mono` is the plain `LEPSBAS / pz_min` cut, while
/// for larger `l` the search starts past every Gaussian-polynomial turning
/// point instead (the bound can dip below the threshold at `LEPSBAS / pz_min`
/// and rise above it farther out; PR #97, `l = 82`). Everything inside
/// `r2_mono` keeps its derivatives.
pub(crate) fn derivative_cut_exponent(
    kind: AoKind,
    l: i64,
    prim_z: &[f64],
    amax: &[f64],
    pz_min: f64,
) -> f64 {
    if !(pz_min.is_finite() && pz_min > 0.0) {
        return pz_min;
    }
    let lf = l.max(0) as f64;
    let (pm, gm, lm) = match kind {
        AoKind::Cart => (1.0, lf, lf * (lf - 1.0)),
        AoKind::Sphe => {
            let s = ((2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI)).sqrt();
            (s, s * (lf * (2.0 * lf + 1.0)).sqrt(), 0.0)
        }
    };
    let li = l.max(0) as i32;
    let eps = FAST_SCREEN_ABS_BOUND;
    let bound = |r2: f64| -> f64 {
        let r = r2.sqrt();
        let rl = r.powi(li);
        let rlm1 = if gm > 0.0 { r.powi(li - 1) } else { 0.0 };
        let rlm2 = if lm > 0.0 { r.powi(li - 2) } else { 0.0 };
        let (mut bg, mut bl) = (0.0_f64, 0.0_f64);
        for (q, &ze) in prim_z.iter().enumerate() {
            let w = amax[q] * (-ze * r2).exp();
            bg += w * (gm * rlm1 + 2.0 * ze * pm * rl * r);
            bl += w * (lm * rlm2 + (4.0 * ze * ze * r2 + (4.0 * lf + 6.0) * ze) * pm * rl);
        }
        bg.max(bl)
    };
    let below = |r2: f64| -> bool { bound(r2) < eps };
    // Smallest exponent actually present (pz_min by construction; recomputed so
    // the monotonicity start never depends on the caller's bookkeeping).
    let zeta_min = prim_z.iter().copied().fold(pz_min, f64::min);
    if !(zeta_min.is_finite() && zeta_min > 0.0) {
        // Non-positive / non-finite exponent: keep every derivative (never screen).
        return 0.0;
    }
    // Largest power of r^2 in the bound is (l+2)/2 (the 4 zeta^2 r^2 * r^l
    // Laplacian term); start past every term's turning point, i.e. at
    // r2_mono = max(LEPSBAS, (l+2)/2) / zeta_min, expressed as the effective
    // exponent LEPSBAS / r2_mono (exactly zeta_min for l <= 39).
    let p_max = 0.5 * (lf + 2.0);
    first_cut_exponent_past(monotone_start_exponent(zeta_min, p_max), below)
}

/// Effective exponent of the monotonicity start `r2_mono = max(LEPSBAS, p_max) /
/// zeta_min` of a screen bound whose terms are non-negative constants times
/// `(r^2)^p exp(-zeta_q r^2)` with `p <= p_max` (expressed as `LEPSBAS /
/// r2_mono`): exactly `zeta_min` (the plain `pz_min` cut) when `p_max <=
/// LEPSBAS`, else rounded outward (slightly smaller exponent = slightly larger
/// `r^2`). Shared by [`derivative_cut_exponent`] and [`value_cut_exponent`].
fn monotone_start_exponent(zeta_min: f64, p_max: f64) -> f64 {
    if p_max <= LEPSBAS {
        zeta_min
    } else {
        zeta_min * (LEPSBAS / p_max) * (1.0 - 8.0 * f64::EPSILON)
    }
}

/// Doubling + bisection for the smallest `r^2 >= LEPSBAS / start_exp` at which
/// `below(r^2)` holds (rounded outward to bisection accuracy), returned as the
/// effective exponent `LEPSBAS / r^2`. The caller guarantees the bound is
/// non-increasing past `LEPSBAS / start_exp`. Returns `start_exp` when the bound
/// already holds there, and `0.0` (never screen) when it never does (non-finite
/// coefficients).
fn first_cut_exponent_past(start_exp: f64, below: impl Fn(f64) -> bool) -> f64 {
    let mut lo = LEPSBAS / start_exp;
    if below(lo) {
        return start_exp;
    }
    let mut hi = 2.0 * lo;
    let mut grow = 0;
    while !below(hi) {
        lo = hi;
        hi *= 2.0;
        grow += 1;
        if grow > 200 {
            // Non-finite coefficients: keep every term (never screen).
            return 0.0;
        }
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if below(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    LEPSBAS / hi
}

/// Effective exponent `pz_val` of the FAST-mode VALUE screens of one shell
/// (issue #100; the value analogue of [`derivative_cut_exponent`]).
///
/// With the notation of [`derivative_cut_exponent`] (`phi = P g`,
/// `g = sum_q a_q exp(-zeta_q r^2)`, `|P| <= Pm r^l`, `A_q = max over the
/// shell's AOs of |a_q|`; Cartesian `Pm = 1`, spherical
/// `Pm = sqrt((2l+1)/4pi)`), every AO of the shell obeys
///   |phi| <= Pm r^l sum_q A_q exp(-zeta_q r^2).
/// Each term is a non-negative constant times `(r^2)^(l/2) exp(-zeta_q r^2)`,
/// non-increasing for `r^2 >= (l/2) / zeta_q`. Returns `LEPSBAS / r2_cut`, where
/// `r2_cut` is (to bisection accuracy, rounded outward) the smallest
/// `r^2 >= r2_mono = max(LEPSBAS, l/2) / zeta_min` at which the bound is below
/// [`FAST_SCREEN_ABS_BOUND`]; past `r2_mono` the bound is non-increasing, so
/// every AO value a screen `pz_val * r^2 > LEPSBAS` drops is below
/// [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in absolute AO units, for any accepted
/// angular momentum (the start depends only on `l` and `zeta_min`, not on the
/// threshold, so it holds for any threshold). The plain value screen
/// `pz_min * r^2 > LEPSBAS` is only relative to the shell's peak factor
/// `|norm*coef|` and drops AO values up to ~1e-6 for tight high-l shells
/// (Cu ccECP cc-pVTZ g: 1.4e-6; O ccECP cc-pVQZ g: 7.3e-7).
/// Never tighter than that plain cut (`pz_val <= pz_min`); a non-finite /
/// non-positive exponent (or non-finite coefficients) disables the screen
/// (`0.0`).
pub(crate) fn value_cut_exponent(
    kind: AoKind,
    l: i64,
    prim_z: &[f64],
    amax: &[f64],
    pz_min: f64,
) -> f64 {
    let zeta_min = prim_z.iter().copied().fold(pz_min, f64::min);
    if !(pz_min.is_finite() && pz_min > 0.0 && zeta_min.is_finite() && zeta_min > 0.0) {
        // Non-positive / non-finite exponent: never screen.
        return 0.0;
    }
    let lf = l.max(0) as f64;
    let pm = match kind {
        AoKind::Cart => 1.0,
        AoKind::Sphe => ((2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI)).sqrt(),
    };
    let li = l.max(0) as i32;
    let eps = FAST_SCREEN_ABS_BOUND;
    let below = |r2: f64| -> bool {
        let rl = r2.sqrt().powi(li);
        let mut b = 0.0_f64;
        for (q, &ze) in prim_z.iter().enumerate() {
            b += amax[q] * (-ze * r2).exp();
        }
        pm * rl * b < eps
    };
    first_cut_exponent_past(monotone_start_exponent(zeta_min, 0.5 * lf), below)
}

/// Precomputed real solid harmonic (x the `sqrt((2l+1)/4pi)` AO prefactor),
/// evaluated as `(Sum_k lam_k * r^2^k * z^ez) * (Sum_p mc_p * x^p * y^(|m|-p))`.
/// All binom/factorial work is done once in [`SpheAngular::new`].
#[derive(Clone)]
pub struct SpheAngular {
    m_abs: i64,
    /// `mc[p] = binom(|m|,p) * (cos|sin)((|m|-p)*pi/2)`; sin for m<0.
    m_coeffs: Vec<f64>,
    /// `(lam_k, k, ez)` for the z/r part (all prefactors folded into lam_k).
    lambda_terms: Vec<LambdaTerm>,
}

/// One `(lam_k, k, ez)` term of [`SpheAngular`]'s z/r part.
type LambdaTerm = (f64, i32, i32);

impl SpheAngular {
    fn new(l: i64, m: i64) -> Self {
        let m_abs = m.abs();
        let mut m_coeffs = Vec::with_capacity((m_abs + 1) as usize);
        for p in 0..=m_abs {
            let ang = if m >= 0 {
                (((m_abs - p) as f64) * std::f64::consts::FRAC_PI_2).cos()
            } else {
                (((m_abs - p) as f64) * std::f64::consts::FRAC_PI_2).sin()
            };
            m_coeffs.push(binom(m_abs, p) * ang);
        }
        // Fold the Lambda prefactor AND the AO-level sqrt((2l+1)/4pi) into every lam_k.
        let prefac = ((2 - if m_abs == 0 { 1 } else { 0 }) as f64 * factorial(l - m_abs)
            / factorial(l + m_abs))
        .sqrt()
            * ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
        let kmax = (l - m_abs) / 2;
        let mut lambda_terms = Vec::new();
        for k in 0..=kmax {
            let lam = (-1.0_f64).powi(k as i32)
                * 2.0_f64.powi(-(l as i32))
                * binom(l, k)
                * binom(2 * l - 2 * k, l)
                * factorial(l - 2 * k)
                / factorial(l - 2 * k - m_abs);
            let ez = (l - 2 * k - m_abs) as i32;
            lambda_terms.push((prefac * lam, k as i32, ez));
        }
        SpheAngular {
            m_abs,
            m_coeffs,
            lambda_terms,
        }
    }

    /// Expose the precomputed angular tables for the flat per-AO layout.
    fn parts(&self) -> (i64, &[f64], &[LambdaTerm]) {
        (self.m_abs, &self.m_coeffs, &self.lambda_terms)
    }

    /// Fast angular evaluation `= sqrt((2l+1)/4pi) * S_{l,m}(x,y,z)`. `r2 = x^2+y^2+z^2`.
    #[inline]
    pub fn eval(&self, x: f64, y: f64, z: f64, r2: f64) -> f64 {
        let mut m_val = 0.0_f64;
        for (p, &c) in self.m_coeffs.iter().enumerate() {
            m_val += c * x.powi(p as i32) * y.powi((self.m_abs - p as i64) as i32);
        }
        let mut lam = 0.0_f64;
        for &(c, k, ez) in &self.lambda_terms {
            lam += c * r2.powi(k) * z.powi(ez);
        }
        lam * m_val
    }

    /// Same polynomial as [`Self::eval`], with every operand and accumulation
    /// performed in f32. Inputs are coordinate differences already formed in
    /// walker/storage precision, matching the Python ao_eval boundary.
    #[inline]
    fn eval_f32(&self, x: f32, y: f32, z: f32, r2: f32) -> f32 {
        let mut m_val = 0.0_f32;
        for (p, &c) in self.m_coeffs.iter().enumerate() {
            m_val += (c as f32) * x.powi(p as i32) * y.powi((self.m_abs - p as i64) as i32);
        }
        let mut lam = 0.0_f32;
        for &(c, k, ez) in &self.lambda_terms {
            lam += (c as f32) * r2.powi(k) * z.powi(ez);
        }
        lam * m_val
    }
}

impl AosGtoData {
    pub(crate) fn coefficient_scale(&self, ao: usize) -> f64 {
        self.ao_coefficient_scales.get(ao).copied().unwrap_or(1.0)
    }

    /// Zero-AO placeholder, used as the `OrbData.aos` field when the orbital is a
    /// B-spline leaf (which carries no GTO primitives). Never evaluated -- the
    /// value/grad/lap paths branch on `OrbData.bspline` before touching `aos`.
    pub fn empty() -> Self {
        AosGtoData {
            kind: AoKind::Cart,
            num_ao_kernel: 0,
            num_ao: 0,
            n_image: 1,
            molecular_sphe: false,
            precision_policy: crate::construction_precision_policy(),
            value_zone: "ao_eval",
            centers: Array2::<f64>::zeros((0, 3)),
            angular_momentums: Vec::new(),
            orbital_indices: Vec::new(),
            exponents: Vec::new(),
            coefficients: Vec::new(),
            ao_coefficient_scales: Vec::new(),
            poly_x: Vec::new(),
            poly_y: Vec::new(),
            poly_z: Vec::new(),
            magnetic: Vec::new(),
            normalizations: Vec::new(),
            prim_groups: Vec::new(),
            sphe_angular: Vec::new(),
            sphe_ang_vg: Vec::new(),
            flat_nc: Vec::new(),
            flat_exp: Vec::new(),
            ao_prim_offset: vec![0],
            shells: Vec::new(),
            sh_center: Vec::new(),
            sh_off: vec![0],
            sh_neg_pz: Vec::new(),
            sh_pnc: Vec::new(),
            sh_pzprim: Vec::new(),
            sh_pzmin: Vec::new(),
            sh_pzval: Vec::new(),
            sh_ao_off: vec![0],
            sh_ao_kernel: Vec::new(),
            sh_ao_home: Vec::new(),
            kernel_to_home: Vec::new(),
            ang_mabs: Vec::new(),
            ang_mc_off: vec![0],
            ang_mc: Vec::new(),
            ang_lam_off: vec![0],
            ang_lam_c: Vec::new(),
            ang_lam_k: Vec::new(),
            ang_lam_ez: Vec::new(),
            wrap_cell: None,
            wrap_inv: [[0.0; 3]; 3],
            cart_fam: Vec::new(),
            cart_norm_fused_f64: Vec::new(),
            home_kernel_off: Vec::new(),
            home_kernel: Vec::new(),
        }
    }

    /// Precompute the electron-independent per-primitive normalizations and the
    /// per-AO primitive groups. MUST be called after constructing the struct
    /// (done by `from_py`); hoists this work out of the per-move hot loop.
    pub fn finalize(mut self) -> Self {
        self.normalizations = primitive_normalizations(&self);
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); self.num_ao_kernel];
        for (prim, &ao) in self.orbital_indices.iter().enumerate() {
            groups[ao].push(prim);
        }
        self.prim_groups = groups;
        if self.kind == AoKind::Sphe {
            self.sphe_angular = (0..self.num_ao_kernel)
                .map(|ao| SpheAngular::new(self.angular_momentums[ao], self.magnetic[ao]))
                .collect();
            self.sphe_ang_vg = (0..self.num_ao_kernel)
                .map(|ao| SpheAngVG::new(self.angular_momentums[ao], self.magnetic[ao]))
                .collect();
            // Flat per-AO angular tables (contiguous, cache-friendly hot loop).
            let nk = self.num_ao_kernel;
            let mut ang_mabs = Vec::with_capacity(nk);
            let mut ang_mc_off = Vec::with_capacity(nk + 1);
            let mut ang_mc = Vec::new();
            let mut ang_lam_off = Vec::with_capacity(nk + 1);
            let (mut ang_lam_c, mut ang_lam_k, mut ang_lam_ez) =
                (Vec::new(), Vec::new(), Vec::new());
            ang_mc_off.push(0);
            ang_lam_off.push(0);
            for ao in 0..nk {
                let (mabs, mc, lam) = self.sphe_angular[ao].parts();
                ang_mabs.push(mabs as i32);
                ang_mc.extend_from_slice(mc);
                ang_mc_off.push(ang_mc.len() as u32);
                for &(c, k, ez) in lam {
                    ang_lam_c.push(c);
                    ang_lam_k.push(k);
                    ang_lam_ez.push(ez);
                }
                ang_lam_off.push(ang_lam_c.len() as u32);
            }
            self.ang_mabs = ang_mabs;
            self.ang_mc_off = ang_mc_off;
            self.ang_mc = ang_mc;
            self.ang_lam_off = ang_lam_off;
            self.ang_lam_c = ang_lam_c;
            self.ang_lam_k = ang_lam_k;
            self.ang_lam_ez = ang_lam_ez;
        }
        // Tight flat per-AO primitive layout.
        let mut flat_nc = Vec::with_capacity(self.exponents.len());
        let mut flat_exp = Vec::with_capacity(self.exponents.len());
        let mut ao_prim_offset = Vec::with_capacity(self.num_ao_kernel + 1);
        ao_prim_offset.push(0);
        for ao in 0..self.num_ao_kernel {
            for &p in &self.prim_groups[ao] {
                flat_nc.push(self.normalizations[p] * self.coefficients[p]);
                flat_exp.push(self.exponents[p]);
            }
            ao_prim_offset.push(flat_nc.len());
        }
        self.flat_nc = flat_nc;
        self.flat_exp = flat_exp;
        self.ao_prim_offset = ao_prim_offset;

        // Shell grouping: group AOs by (center, l, primitive signature).
        // For sphe the per-primitive normalization depends on (l, Z) only, so all
        // AOs in a shell share the identical radial sum -> exp once per shell.
        // Cart normalizations also depend on (nx, ny, nz), so its radial sum is
        // still evaluated per AO.  It nevertheless needs this table for one
        // distance/cutoff decision per shell in
        // `compute_mo_single_cutoff_cart_into`, so shells are built for cart too;
        // without them that cart fast path would be unreachable.
        {
            use std::collections::HashMap;
            /// `(center x/y/z bits, l, primitive (exponent, coefficient) bits)`.
            type ShellKey = (u64, u64, u64, i64, Vec<(u64, u64)>);
            let mut shell_of: HashMap<ShellKey, usize> = HashMap::new();
            let mut shells: Vec<Shell> = Vec::new();
            for ao in 0..self.num_ao_kernel {
                let cx = self.centers[[ao, 0]];
                let cy = self.centers[[ao, 1]];
                let cz = self.centers[[ao, 2]];
                let l = self.angular_momentums[ao];
                let sig: Vec<(u64, u64)> = self.prim_groups[ao]
                    .iter()
                    .map(|&p| (self.exponents[p].to_bits(), self.coefficients[p].to_bits()))
                    .collect();
                let key = (cx.to_bits(), cy.to_bits(), cz.to_bits(), l, sig);
                let sidx = *shell_of.entry(key).or_insert_with(|| {
                    let prim_z: Vec<f64> = self.prim_groups[ao]
                        .iter()
                        .map(|&p| self.exponents[p])
                        .collect();
                    let prim_c: Vec<f64> = self.prim_groups[ao]
                        .iter()
                        .map(|&p| self.coefficients[p])
                        .collect();
                    let prim_nc: Vec<f64> = self.prim_groups[ao]
                        .iter()
                        .map(|&p| self.normalizations[p] * self.coefficients[p])
                        .collect();
                    let pz_min = prim_z.iter().cloned().fold(f64::INFINITY, f64::min);
                    shells.push(Shell {
                        center: [cx, cy, cz],
                        l,
                        prim_z,
                        prim_c,
                        prim_nc,
                        aos: Vec::new(),
                        pz_min,
                        pz_deriv: pz_min,
                        pz_val: pz_min,
                        prim_val: Vec::new(),
                    });
                    shells.len() - 1
                });
                shells[sidx].aos.push(ao);
            }
            // AO-magnitude-aware derivative cut per shell (issue #87). The shell
            // signature fixes the primitive order, so primitive `q` of every AO of
            // the shell sits at `ao_prim_offset[ao] + q`; Cartesian components of
            // one shell carry different normalizations, hence the max over AOs.
            for sh in shells.iter_mut() {
                let mut amax = vec![0.0_f64; sh.prim_z.len()];
                for &ao in &sh.aos {
                    for (q, a) in amax.iter_mut().enumerate() {
                        let v = self.flat_nc[self.ao_prim_offset[ao] + q].abs();
                        if !matches!(
                            v.partial_cmp(a),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ) {
                            *a = v;
                        }
                    }
                }
                sh.pz_deriv =
                    derivative_cut_exponent(self.kind, sh.l, &sh.prim_z, &amax, sh.pz_min);
                // Same `amax`, value-only bound (issue #100).
                sh.pz_val = value_cut_exponent(self.kind, sh.l, &sh.prim_z, &amax, sh.pz_min);
                sh.prim_val = prim_value_cut_exponents(self.kind, sh.l, &sh.prim_z, &amax);
            }
            // Fully flat per-shell layout (contiguous; cache-friendly hot loop).
            let mut sh_center = Vec::with_capacity(shells.len() * 3);
            let mut sh_off: Vec<u32> = Vec::with_capacity(shells.len() + 1);
            let mut sh_neg_pz: Vec<f64> = Vec::new();
            let mut sh_pnc: Vec<f64> = Vec::new();
            let mut sh_pzprim: Vec<f64> = Vec::new();
            let mut sh_pzmin: Vec<f64> = Vec::with_capacity(shells.len());
            let mut sh_pzval: Vec<f64> = Vec::with_capacity(shells.len());
            let mut sh_ao_off: Vec<u32> = Vec::with_capacity(shells.len() + 1);
            let mut sh_ao_kernel: Vec<u32> = Vec::new();
            let mut sh_ao_home: Vec<u32> = Vec::new();
            sh_off.push(0);
            sh_ao_off.push(0);
            for sh in &shells {
                sh_center.extend_from_slice(&sh.center);
                sh_pzmin.push(sh.pz_min);
                sh_pzval.push(sh.pz_val);
                for i in 0..sh.prim_z.len() {
                    sh_neg_pz.push(-sh.prim_z[i]);
                    sh_pnc.push(sh.prim_nc[i]);
                    sh_pzprim.push(sh.prim_val[i]);
                }
                sh_off.push(sh_neg_pz.len() as u32);
                for &ao in &sh.aos {
                    sh_ao_kernel.push(ao as u32);
                    sh_ao_home.push(self.kernel_to_home[ao] as u32);
                }
                sh_ao_off.push(sh_ao_kernel.len() as u32);
            }
            self.sh_center = sh_center;
            self.sh_off = sh_off;
            self.sh_neg_pz = sh_neg_pz;
            self.sh_pnc = sh_pnc;
            self.sh_pzprim = sh_pzprim;
            self.sh_pzmin = sh_pzmin;
            self.sh_pzval = sh_pzval;
            self.sh_ao_off = sh_ao_off;
            self.sh_ao_kernel = sh_ao_kernel;
            self.sh_ao_home = sh_ao_home;
            self.shells = shells;
        }
        // Families pay off only when they are long (periodic images): A64FX Si
        // (~270 members/family) wdot 0.93 -> 0.67 ms, but open water (1-2
        // members) 0.038 -> 0.142 ms. Keep them only above a mean of
        // CART_FAM_MIN_MEMBERS members.
        self.cart_norm_fused_f64 = if self.kind == AoKind::Cart {
            let mut v = vec![0.0_f64; self.exponents.len()];
            for ao in 0..self.num_ao_kernel {
                for &p in &self.prim_groups[ao] {
                    v[p] = primitive_normalization_cart_fused_f64(&self, p, ao);
                }
            }
            v
        } else {
            Vec::new()
        };
        {
            let mut off = vec![0_usize; self.num_ao + 1];
            for &h in &self.kernel_to_home {
                off[h + 1] += 1;
            }
            for h in 0..self.num_ao {
                off[h + 1] += off[h];
            }
            let mut fill = off.clone();
            let mut ker = vec![0_usize; self.kernel_to_home.len()];
            for (k, &h) in self.kernel_to_home.iter().enumerate() {
                ker[fill[h]] = k;
                fill[h] += 1;
            }
            self.home_kernel_off = off;
            self.home_kernel = ker;
        }
        self.cart_fam = if self.kind == AoKind::Cart {
            let fams = build_cart_families(&self);
            if !fams.is_empty() && self.shells.len() >= CART_FAM_MIN_MEMBERS * fams.len() {
                fams
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        self
    }

    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Distinguish cart vs sphe by which angular metadata the dataclass carries.
        let is_cart =
            obj.hasattr("polynominal_order_x")? && !obj.getattr("polynominal_order_x")?.is_none();
        let kind = if is_cart { AoKind::Cart } else { AoKind::Sphe };

        let num_ao_kernel: usize = obj.getattr("_num_ao_kernel")?.extract()?;
        let num_ao: usize = obj.getattr("num_ao")?.extract()?;
        // `_n_image` exists on both molecular (==1) and pGTO leaves.
        let n_image: usize = obj.getattr("_n_image")?.extract()?;
        // Periodicity is a property of the structure, not of the image count: a
        // periodic leaf whose build-time truncation keeps only the home image has
        // `n_image == 1` but is still periodic (jqmc wraps it in
        // `_wrap_into_cell_if_pbc` exactly like a multi-image leaf).
        let periodic = n_image > 1 || leaf_pbc_flag(obj)?;
        // A one-image periodic leaf is not molecular. Store the actual boundary
        // independently of the policy used now: explicit-policy consumers may
        // reuse this model later.
        let molecular_sphe = kind == AoKind::Sphe && !periodic;
        // Per-kernel-AO centers.  Read the JAX property through numpy so the
        // periodic `image_index @ cell` contraction is the same one used by the
        // XLA AO kernel.  The numpy property uses a separate BLAS reduction and
        // can differ by an fp64 ulp; after `r - R` is cast to fp32 at a cell seam
        // that becomes a several-ulp AO discrepancy.
        let np = obj.py().import("numpy")?;
        let centers_arr: numpy::PyReadonlyArray2<f64> = np
            .call_method1("asarray", (obj.getattr("_atomic_center_carts_jnp")?,))?
            .extract()?;
        let centers = centers_arr.as_array().to_owned();

        // kernel-AO -> home-AO fold (per-shell RAGGED layout). Present only on
        // periodic pGTO leaves; molecular folds to identity (num_ao_kernel == num_ao).
        let kernel_to_home: Vec<usize> = if obj.hasattr("_ragged_kernel_to_home")? {
            extract_usize_vec(&obj.getattr("_ragged_kernel_to_home")?)?
        } else {
            (0..num_ao_kernel).collect()
        };
        let angular_momentums: Vec<i64> = obj.getattr("angular_momentums")?.extract()?;
        let orbital_indices: Vec<usize> = obj.getattr("orbital_indices")?.extract()?;

        // The canonical radial basis is the per-shell-primitive
        // `AO_radial_shell_data`, with
        // `ao_prim_to_shell_prim` mapping each full per-AO-primitive to its shell-prim.
        // Expand to the full per-AO-primitive arrays ONCE here (the kernel entry), so
        // the value kernels read contiguous `Vec<f64>` (no per-call gather).
        // `aos.exponents`/`aos.coefficients` are @property doing exactly this take.
        let radial = obj.getattr("radial")?;
        let radial_exponents: Vec<f64> = pyarray_or_seq_f64(&radial.getattr("exponents")?)?;
        let radial_coefficients: Vec<f64> = pyarray_or_seq_f64(&radial.getattr("coefficients")?)?;
        let ao_prim_to_shell_prim: Vec<usize> =
            extract_usize_vec(&obj.getattr("ao_prim_to_shell_prim")?)?;
        let exponents: Vec<f64> = ao_prim_to_shell_prim
            .iter()
            .map(|&s| radial_exponents[s])
            .collect();
        let mut coefficients: Vec<f64> = ao_prim_to_shell_prim
            .iter()
            .map(|&s| radial_coefficients[s])
            .collect();
        let ao_coefficient_scales: Vec<f64> = if obj.hasattr("ao_coefficient_scales")? {
            pyarray_or_seq_f64(&obj.getattr("ao_coefficient_scales")?)?
        } else {
            Vec::new()
        };
        if !ao_coefficient_scales.is_empty() {
            if ao_coefficient_scales.len() != num_ao_kernel
                || ao_coefficient_scales.iter().any(|scale| !scale.is_finite())
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "GTO coefficient scales must contain one finite value per kernel AO",
                ));
            }
            for (coefficient, &ao) in coefficients.iter_mut().zip(&orbital_indices) {
                *coefficient *= ao_coefficient_scales[ao];
            }
        }

        // Periodicity guarantor (real-Gamma pGTO): read the home cell so the eval
        // entry points can wrap electron coordinates into it, matching jqmc's
        // `_wrap_into_cell_if_pbc`. Every periodic leaf carries a meaningful
        // `structure_data._cell_jnp`, including a one-image periodic leaf (its single
        // home image only covers home-cell coordinates); molecular leaves stay `None`.
        let (wrap_cell, wrap_inv) = if periodic {
            let sd = obj.getattr("structure_data")?;
            let cell_a: numpy::PyReadonlyArray2<f64> = np
                .call_method1("asarray", (sd.getattr("_cell_jnp")?,))?
                .extract()?;
            let c = cell_a.as_array();
            let cell = [
                [c[[0, 0]], c[[0, 1]], c[[0, 2]]],
                [c[[1, 0]], c[[1, 1]], c[[1, 2]]],
                [c[[2, 0]], c[[2, 1]], c[[2, 2]]],
            ];
            let inv = crate::complex_det::inv_3x3(&cell);
            (Some(cell), inv)
        } else {
            (None, [[0.0; 3]; 3])
        };

        let (poly_x, poly_y, poly_z, magnetic) = if is_cart {
            (
                obj.getattr("polynominal_order_x")?.extract()?,
                obj.getattr("polynominal_order_y")?.extract()?,
                obj.getattr("polynominal_order_z")?.extract()?,
                Vec::new(),
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                obj.getattr("magnetic_quantum_numbers")?.extract()?,
            )
        };

        Ok(AosGtoData {
            kind,
            num_ao_kernel,
            num_ao,
            n_image,
            molecular_sphe,
            precision_policy: crate::construction_precision_policy(),
            value_zone: if obj.hasattr("_image_phase_factors")? {
                "ao_bloch_eval"
            } else {
                "ao_eval"
            },
            centers,
            kernel_to_home,
            angular_momentums,
            orbital_indices,
            exponents,
            coefficients,
            ao_coefficient_scales,
            poly_x,
            poly_y,
            poly_z,
            magnetic,
            normalizations: Vec::new(),
            prim_groups: Vec::new(),
            sphe_angular: Vec::new(),
            sphe_ang_vg: Vec::new(),
            flat_nc: Vec::new(),
            flat_exp: Vec::new(),
            ao_prim_offset: Vec::new(),
            shells: Vec::new(),
            sh_center: Vec::new(),
            sh_off: Vec::new(),
            sh_neg_pz: Vec::new(),
            sh_pnc: Vec::new(),
            sh_pzprim: Vec::new(),
            sh_pzmin: Vec::new(),
            sh_pzval: Vec::new(),
            sh_ao_off: Vec::new(),
            sh_ao_kernel: Vec::new(),
            sh_ao_home: Vec::new(),
            ang_mabs: Vec::new(),
            ang_mc_off: Vec::new(),
            ang_mc: Vec::new(),
            ang_lam_off: Vec::new(),
            ang_lam_c: Vec::new(),
            ang_lam_k: Vec::new(),
            ang_lam_ez: Vec::new(),
            wrap_cell,
            wrap_inv,
            cart_fam: Vec::new(),
            cart_norm_fused_f64: Vec::new(),
            home_kernel_off: Vec::new(),
            home_kernel: Vec::new(),
        }
        .finalize())
    }
}

/// Read a numpy f64 array (or any sequence of floats) into a Vec<f64>.
fn pyarray_or_seq_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        Ok(arr.as_array().to_vec())
    } else {
        obj.extract::<Vec<f64>>()
    }
}

/// Read an index array (tuple of ints, or numpy int32/int64 array) into Vec<usize>.
fn extract_usize_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(a) = obj.extract::<numpy::PyReadonlyArray1<i64>>() {
        return Ok(a.as_array().iter().map(|&x| x as usize).collect());
    }
    if let Ok(a) = obj.extract::<numpy::PyReadonlyArray1<i32>>() {
        return Ok(a.as_array().iter().map(|&x| x as usize).collect());
    }
    let v: Vec<i64> = obj.extract()?;
    Ok(v.into_iter().map(|x| x as usize).collect())
}

/// Per-primitive normalization `N_p` (electron-independent), 1:1 with the
/// `N_n` factor in the value/grad/lap kernels. Hoisted out of the electron loop.
fn primitive_normalizations(data: &AosGtoData) -> Vec<f64> {
    let mut norms = vec![0.0_f64; data.exponents.len()];
    for (p, &ao) in data.orbital_indices.iter().enumerate() {
        let l = data.angular_momentums[ao];
        let z = data.exponents[p];
        norms[p] = match data.kind {
            AoKind::Cart => {
                let nx = data.poly_x[ao];
                let ny = data.poly_y[ao];
                let nz = data.poly_z[ao];
                ((2.0 * z / std::f64::consts::PI).powf(1.5)
                    * (8.0 * z).powi(l as i32)
                    * factorial(nx)
                    * factorial(ny)
                    * factorial(nz)
                    / (factorial(2 * nx) * factorial(2 * ny) * factorial(2 * nz)))
                .sqrt()
            }
            AoKind::Sphe => ((2.0_f64.powi((2 * l + 3) as i32)
                * factorial(l + 1)
                * (2.0 * z).powf(l as f64 + 1.5))
                / (factorial(2 * l + 2) * std::f64::consts::PI.sqrt()))
            .sqrt(),
        };
    }
    norms
}

#[inline]
fn factorial(n: i64) -> f64 {
    let mut acc = 1.0_f64;
    let mut k = 2_i64;
    while k <= n {
        acc *= k as f64;
        k += 1;
    }
    acc
}

#[inline]
fn binom(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// Real regular solid harmonic `S_{l,m}` value only.
fn compute_s_l_m(l: i64, m: i64, x: f64, y: f64, z: f64, r_norm: f64) -> f64 {
    s_l_m_value_and_grad(l, m, x, y, z, r_norm).0
}

/// Real regular solid harmonic `S_{l,m}` (= r^l * real spherical harmonic),
/// 1:1 with `_compute_S_l_m_debug`, together with its analytic Cartesian
/// gradient `(dS/dx, dS/dy, dS/dz)`. `(x, y, z) = r - R`, `r_norm = |r - R|`.
///
/// `S = Lambda(r, z) * M(x, y)`, `M = A_m (m>=0)` or `B_m (m<0)`.
/// Derivatives use `G = (dLambda/dr)/r` (no 1/r singularity since the lowest
/// `dLambda/dr` term is O(r)) and the explicit z-derivative `Hz = dLambda/dz|_r`.
fn s_l_m_value_and_grad(
    l: i64,
    m: i64,
    x: f64,
    y: f64,
    z: f64,
    r_norm: f64,
) -> (f64, f64, f64, f64) {
    let m_abs = m.abs();
    let trig = |q: i64| -> f64 {
        // cos/sin of (q * pi/2); used by A_m / B_m respectively.
        q as f64 * std::f64::consts::FRAC_PI_2
    };

    // M(x,y) and its x,y partials. For m>=0 use cos (A_m), else sin (B_m).
    let (mut mm, mut mx, mut my) = (0.0_f64, 0.0_f64, 0.0_f64);
    for p in 0..=m_abs {
        let b = binom(m_abs, p);
        let ang = if m >= 0 {
            trig(m_abs - p).cos()
        } else {
            trig(m_abs - p).sin()
        };
        let xp = x.powi(p as i32);
        let yq = y.powi((m_abs - p) as i32);
        mm += b * xp * yq * ang;
        if p >= 1 {
            mx += b * (p as f64) * x.powi((p - 1) as i32) * yq * ang;
        }
        if m_abs - p >= 1 {
            my += b * xp * ((m_abs - p) as f64) * y.powi((m_abs - p - 1) as i32) * ang;
        }
    }

    let lambda_lm = |k: i64| -> f64 {
        (-1.0_f64).powi(k as i32)
            * 2.0_f64.powi(-(l as i32))
            * binom(l, k)
            * binom(2 * l - 2 * k, l)
            * factorial(l - 2 * k)
            / factorial(l - 2 * k - m_abs)
    };
    let prefac = ((2 - if m_abs == 0 { 1 } else { 0 }) as f64 * factorial(l - m_abs)
        / factorial(l + m_abs))
    .sqrt();
    let kmax = (l - m_abs) / 2;

    let mut lambda = 0.0_f64; // Lambda(r,z)
    let mut g = 0.0_f64; // (dLambda/dr)/r
    let mut hz = 0.0_f64; // dLambda/dz|_r
    for k in 0..=kmax {
        let lam = lambda_lm(k);
        let e = l - 2 * k - m_abs; // z exponent
        lambda += lam * r_norm.powi((2 * k) as i32) * z.powi(e as i32);
        if k >= 1 {
            g += lam * (2 * k) as f64 * r_norm.powi((2 * k - 2) as i32) * z.powi(e as i32);
        }
        if e >= 1 {
            hz += lam * r_norm.powi((2 * k) as i32) * (e as f64) * z.powi((e - 1) as i32);
        }
    }
    lambda *= prefac;
    g *= prefac;
    hz *= prefac;

    let s = lambda * mm;
    let dsdx = mm * x * g + lambda * mx;
    let dsdy = mm * y * g + lambda * my;
    let dsdz = mm * (z * g + hz);
    (s, dsdx, dsdy, dsdz)
}

/// Precomputed constant tables for [`s_l_m_value_and_grad`] (per sphe AO):
/// the binomials, `cos/sin(q*pi/2)` factors, `lambda_lm(k)` coefficients and the
/// normalization `prefac` are all electron-INDEPENDENT, yet the scalar path
/// recomputed them (factorial loops + libm cos/sin) per (AO, electron) -- the
/// dominant cost of the batch AO grad/lap sweep. [`SpheAngVG::eval`] replays the
/// EXACT same runtime expression tree as `s_l_m_value_and_grad` on these
/// precomputed scalars (which are produced by the same expressions on the same
/// integer inputs), so it is bit-identical to the scalar path.
#[derive(Clone)]
pub struct SpheAngVG {
    l: i64,
    m_abs: i64,
    /// `binom(m_abs, p)`, p = 0..=m_abs.
    b: Vec<f64>,
    /// `cos((m_abs-p)*pi/2)` (m>=0) or `sin(...)` (m<0), p = 0..=m_abs.
    ang: Vec<f64>,
    /// `lambda_lm(k)`, k = 0..=(l-m_abs)/2.
    lam: Vec<f64>,
    /// `sqrt((2-delta_m0) * (l-|m|)! / (l+|m|)!)`.
    prefac: f64,
}

impl SpheAngVG {
    fn new(l: i64, m: i64) -> Self {
        let m_abs = m.abs();
        let trig = |q: i64| -> f64 { q as f64 * std::f64::consts::FRAC_PI_2 };
        let mut b = Vec::with_capacity((m_abs + 1) as usize);
        let mut ang = Vec::with_capacity((m_abs + 1) as usize);
        for p in 0..=m_abs {
            b.push(binom(m_abs, p));
            ang.push(if m >= 0 {
                trig(m_abs - p).cos()
            } else {
                trig(m_abs - p).sin()
            });
        }
        let lambda_lm = |k: i64| -> f64 {
            (-1.0_f64).powi(k as i32)
                * 2.0_f64.powi(-(l as i32))
                * binom(l, k)
                * binom(2 * l - 2 * k, l)
                * factorial(l - 2 * k)
                / factorial(l - 2 * k - m_abs)
        };
        let prefac = ((2 - if m_abs == 0 { 1 } else { 0 }) as f64 * factorial(l - m_abs)
            / factorial(l + m_abs))
        .sqrt();
        let kmax = (l - m_abs) / 2;
        let lam = (0..=kmax).map(lambda_lm).collect();
        SpheAngVG {
            l,
            m_abs,
            b,
            ang,
            lam,
            prefac,
        }
    }

    /// Bit-identical replay of [`s_l_m_value_and_grad`]'s body on the
    /// precomputed tables. Returns `(S, dS/dx, dS/dy, dS/dz)`.
    #[inline]
    pub fn eval(&self, x: f64, y: f64, z: f64, r_norm: f64) -> (f64, f64, f64, f64) {
        let m_abs = self.m_abs;
        let (mut mm, mut mx, mut my) = (0.0_f64, 0.0_f64, 0.0_f64);
        for p in 0..=m_abs {
            let b = self.b[p as usize];
            let ang = self.ang[p as usize];
            let xp = x.powi(p as i32);
            let yq = y.powi((m_abs - p) as i32);
            mm += b * xp * yq * ang;
            if p >= 1 {
                mx += b * (p as f64) * x.powi((p - 1) as i32) * yq * ang;
            }
            if m_abs - p >= 1 {
                my += b * xp * ((m_abs - p) as f64) * y.powi((m_abs - p - 1) as i32) * ang;
            }
        }
        let kmax = (self.l - m_abs) / 2;
        let mut lambda = 0.0_f64;
        let mut g = 0.0_f64;
        let mut hz = 0.0_f64;
        for k in 0..=kmax {
            let lam = self.lam[k as usize];
            let e = self.l - 2 * k - m_abs;
            lambda += lam * r_norm.powi((2 * k) as i32) * z.powi(e as i32);
            if k >= 1 {
                g += lam * (2 * k) as f64 * r_norm.powi((2 * k - 2) as i32) * z.powi(e as i32);
            }
            if e >= 1 {
                hz += lam * r_norm.powi((2 * k) as i32) * (e as f64) * z.powi((e - 1) as i32);
            }
        }
        lambda *= self.prefac;
        g *= self.prefac;
        hz *= self.prefac;
        let s = lambda * mm;
        let dsdx = mm * x * g + lambda * mx;
        let dsdy = mm * y * g + lambda * my;
        let dsdz = mm * (z * g + hz);
        (s, dsdx, dsdy, dsdz)
    }
}

/// Angular part `P = sqrt((2l+1)/4pi) * S_{l,m}` and its Cartesian gradient
/// `(dP/dx, dP/dy, dP/dz)` for a spherical AO leaf (harmonic, so `grad^2 P = 0`).
/// Mirrors the `AoKind::Sphe` branch of [`ao_value_grad_lap`]; used by the radial
/// param-grad kinetic reduction (`param_grad_basis::grad_radial_kinetic_side`).
pub fn sphe_angular_value_grad(
    data: &AosGtoData,
    ao: usize,
    x: f64,
    y: f64,
    z: f64,
    r2: f64,
) -> (f64, f64, f64, f64) {
    let l = data.angular_momentums[ao];
    let nlm = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
    // Precomputed-table replay of `s_l_m_value_and_grad` (bit-identical -- see
    // `SpheAngVG`; hoists the per-eval binom/factorial/trig). Fallback for
    // hand-built structs whose `finalize()` was skipped.
    let (s, gx, gy, gz) = if let Some(tab) = data.sphe_ang_vg.get(ao) {
        tab.eval(x, y, z, r2.sqrt())
    } else {
        s_l_m_value_and_grad(l, data.magnetic[ao], x, y, z, r2.sqrt())
    };
    (nlm * s, nlm * gx, nlm * gy, nlm * gz)
}

/// Per-(AO, electron) value, gradient and laplacian bundle.
pub struct AoVGL {
    pub value: f64,
    pub grad: [f64; 3],
    pub laplacian: f64,
}

/// Evaluate value + analytic gradient + laplacian for AO `ao` at one electron.
///
/// Radial moments: `NR = Sum N_p c_p e^{-Z r^2}`, `ZNR = Sum Z*NR_p`, `Z2NR = Sum Z^2*NR_p`.
/// `grad_a = NR*d_a P - 2*diff_a*ZNR*P`;
/// `lap = NR*grad^2P - 4*ZNR*(r*gradP) + (4r^2*Z2NR - 6*ZNR)*P`.
fn ao_value_grad_lap(
    data: &AosGtoData,
    norms: &[f64],
    ao: usize,
    prims: &[usize],
    x: f64,
    y: f64,
    z: f64,
) -> AoVGL {
    let l = data.angular_momentums[ao];
    let r2 = x * x + y * y + z * z;
    let r_norm = r2.sqrt();

    let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &p in prims {
        let z_exp = data.exponents[p];
        let coeff = data.coefficients[p];
        let pref = norms[p] * coeff * (-z_exp * r2).exp();
        nr += pref;
        znr += z_exp * pref;
        z2nr += z_exp * z_exp * pref;
    }

    // Angular part P and its derivatives.
    let (pp, dpx, dpy, dpz, lap_p);
    match data.kind {
        AoKind::Cart => {
            let nx = data.poly_x[ao];
            let ny = data.poly_y[ao];
            let nz = data.poly_z[ao];
            let powm = |v: f64, n: i64| -> f64 {
                if n < 0 {
                    0.0
                } else {
                    v.powi(n as i32)
                }
            };
            let xx = powm(x, nx);
            let yy = powm(y, ny);
            let zz = powm(z, nz);
            pp = xx * yy * zz;
            dpx = (nx as f64) * powm(x, nx - 1) * yy * zz;
            dpy = xx * (ny as f64) * powm(y, ny - 1) * zz;
            dpz = xx * yy * (nz as f64) * powm(z, nz - 1);
            lap_p = (nx as f64) * ((nx - 1) as f64) * powm(x, nx - 2) * yy * zz
                + xx * (ny as f64) * ((ny - 1) as f64) * powm(y, ny - 2) * zz
                + xx * yy * (nz as f64) * ((nz - 1) as f64) * powm(z, nz - 2);
        }
        AoKind::Sphe => {
            let nlm = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
            // Precomputed-table replay (bit-identical to `s_l_m_value_and_grad`,
            // see `SpheAngVG`); scalar fallback for un-finalized test structs.
            let (s, gx, gy, gz) = if let Some(tab) = data.sphe_ang_vg.get(ao) {
                tab.eval(x, y, z, r_norm)
            } else {
                s_l_m_value_and_grad(l, data.magnetic[ao], x, y, z, r_norm)
            };
            pp = nlm * s;
            dpx = nlm * gx;
            dpy = nlm * gy;
            dpz = nlm * gz;
            // Regular solid harmonics are harmonic polynomials: grad^2(angular) = 0.
            lap_p = 0.0;
        }
    }

    let value = nr * pp;
    let grad = [
        nr * dpx - 2.0 * x * znr * pp,
        nr * dpy - 2.0 * y * znr * pp,
        nr * dpz - 2.0 * z * znr * pp,
    ];
    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
    let laplacian = nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
    AoVGL {
        value,
        grad,
        laplacian,
    }
}

/// `structure_data.pbc_flag` of a GTO leaf (`false` when the leaf carries no
/// structure). Shared by the real and complex native leaves so both decide
/// "periodic" exactly as jqmc's `_wrap_into_cell_if_pbc` does.
pub(crate) fn leaf_pbc_flag(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match obj.getattr("structure_data") {
        Ok(sd) if !sd.is_none() => sd.getattr("pbc_flag")?.extract::<bool>(),
        _ => Ok(false),
    }
}

/// Wrap one Cartesian position into the home cell (jqmc `_wrap_into_cell`):
/// `frac = pos @ inv_cell`, `frac -= floor(frac)`, `pos_w = frac @ cell`. No-op
/// (returns `pos`) for molecular leaves (`wrap_cell == None`). Idempotent, so a
/// point already in the home cell is unchanged.
#[inline]
pub(crate) fn wrap_pos(data: &AosGtoData, pos: [f64; 3]) -> [f64; 3] {
    match data.wrap_cell {
        None => pos,
        Some(cell) => {
            let inv = &data.wrap_inv;
            let mut out = [0.0_f64; 3];
            for c in 0..3 {
                let mut frac = pos[0] * inv[0][c] + pos[1] * inv[1][c] + pos[2] * inv[2][c];
                frac -= frac.floor();
                // frac @ cell (column c contribution accumulated below)
                out[0] += frac * cell[c][0];
                out[1] += frac * cell[c][1];
                out[2] += frac * cell[c][2];
            }
            out
        }
    }
}

/// Wrap a batch of Cartesian positions `(n_elec, 3)` into the home cell, or copy
/// them through unchanged for molecular leaves. Mirrors jqmc's
/// `_wrap_into_cell_if_pbc` applied at every AO eval. Returns an owned array so
/// the eval entry points can rebind `r_carts` to a local (the wrapped positions
/// live only for the call); the `(n_elec, 3)` copy is negligible vs the AO sweep.
pub(crate) fn wrap_r(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    if data.wrap_cell.is_none() {
        return r_carts.to_owned();
    }
    let n = r_carts.nrows();
    let mut rw = Array2::<f64>::zeros((n, 3));
    for e in 0..n {
        let w = wrap_pos(data, [r_carts[[e, 0]], r_carts[[e, 1]], r_carts[[e, 2]]]);
        rw[[e, 0]] = w[0];
        rw[[e, 1]] = w[1];
        rw[[e, 2]] = w[2];
    }
    rw
}

/// Evaluate AO values at `r_carts` (shape `(n_elec, 3)`).
/// Returns `(num_ao_kernel, n_elec)` -- matching jqmc's `(num_ao, n_elec)` for
/// the molecular path.
pub fn compute_aos(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    // No home-cell wrap here: this batch VALUE path is shared with the MCMC
    // sweep's geminal init (`OrbData::eval_aos`), which must match JAX's
    // accept/reject calculation. The far-image wrap is applied once at the
    // OBSERVABLE boundary (`orb_value_grad_lap`) instead, covering value + grad
    // + lap consistently without perturbing the sweep.
    // XLA fixes the periodic image set at build time and evaluates every retained
    // image. Default Rust entry points follow that exact convention; approximate
    // cutoffs are confined to explicitly named fast/cutoff kernels.
    compute_aos_with_policy(data, r_carts, data.precision_policy.is_f32(data.value_zone))
}

pub(crate) fn compute_aos_with_policy(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> Array2<f64> {
    if f32_arithmetic {
        fold_image_axis_f32(data, compute_aos_kernel_screened_f32(data, r_carts, false))
    } else {
        fold_image_axis(data, compute_aos_kernel_screened(data, r_carts, false))
    }
}

/// AO value emitted by jqmc's fused value/grad/laplacian API.  In mixed
/// precision its shared analytic body is evaluated in the fp64
/// `ao_grad_lap` zone and only the completed AO value is rounded to fp32.
/// This is intentionally distinct from [`compute_aos_with_policy`], whose
/// standalone value kernel performs its zone-local arithmetic in fp32.
pub(crate) fn compute_aos_fused_value(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    if data.precision_policy.is_f32(data.value_zone) {
        fold_image_axis_f32(
            data,
            compute_aos_kernel_fused_value_f32(data, r_carts, false),
        )
    } else {
        fold_image_axis(data, compute_aos_kernel_screened(data, r_carts, false))
    }
}

/// EXACT (un-screened) sibling of [`compute_aos`]: evaluates EVERY kernel AO of the
/// static image tiling at every electron, with no runtime `pz_min*r^2 > LEPSBAS`
/// distance screen. This is the JAX/XLA convention -- jqmc truncates the periodic
/// image set ONCE at BUILD time (`AO_IMAGE_TRUNCATION_ATOL`, a cell-to-cell
/// closest-approach radius per (alpha, L)) and then evaluates whatever survived
/// exactly, for every electron. The runtime screen is therefore NOT "a subset of
/// what jqmc already drops": it drops
/// (kernel-AO, electron) pairs that jqmc KEEPS, whenever this particular electron
/// happens to be far from an image the cell as a whole is close to. Measured cost of
/// the difference on `pbc_composite_jnone` dE_L/dC: ONE component where the screened
/// path returns exactly 0.0 while XLA returns 1.06e-7.
/// Observable / parity kernels must use THIS; the screened [`compute_aos`] stays on
/// the MCMC per-move sweep, whose accept/reject contract is a separate question.
pub fn compute_aos_exact(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    let wrapped = wrap_r(data, r_carts);
    compute_aos_with_policy(
        data,
        &wrapped.view(),
        data.precision_policy.is_f32(data.value_zone),
    )
}

/// Pre-fold per-kernel-AO values `(num_ao_kernel, N)` -- the body of
/// [`compute_aos`] without the image fold. Used by the complex Bloch path
/// (which folds with the per-image phase `e^{ik*T_n}` instead of a plain sum).
///
/// UNSCREENED (`screen == false`): the complex Bloch fold is an OBSERVABLE /
/// param-grad path (e_L, O_k, dE_L/dc, force, stress) whose output is indexed BY
/// AO, so a runtime `min_exp*r^2 > LEPSBAS` screen would land as an EXACT 0.0 where
/// XLA has a small non-zero -- the same defect fixed for the real observable path
/// by [`compute_aos_exact`]. Measured with the
/// screen ON the complex Bloch AO VALUE differed 9.4e-9 from XLA (fold order itself
/// is bit-identical, 2.8e-17), which amplified through the geminal inverse to a
/// 1.9e-7 O_k floor and a 1.5e-7 ECP dE_L/dc floor. The complex
/// MCMC sweep's hot loop uses the single-position kernels (`compute_aos_single*`,
/// their own truncation), not this batch fold, so exact evaluation here does not
/// perturb accept/reject.
#[cfg(test)]
pub(crate) fn compute_aos_kernel(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    compute_aos_kernel_with_policy(data, r_carts, data.precision_policy.is_f32(data.value_zone))
}

pub(crate) fn compute_aos_kernel_with_policy(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> Array2<f64> {
    if f32_arithmetic {
        compute_aos_kernel_screened_f32(data, r_carts, false)
    } else {
        compute_aos_kernel_screened(data, r_carts, false)
    }
}

fn compute_aos_kernel_fused_value_f32(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> Array2<f64> {
    if data.kind == AoKind::Cart {
        // XLA's fused Cartesian value/grad/lap kernel owns its shared radial
        // and polynomial body in the fp64 `ao_grad_lap` zone, then casts the
        // completed per-AO value to `ao_eval` immediately before the value is
        // consumed.  The Rust force path likewise shares the cart-lowered AO
        // leaf with fp64 derivatives, so putting the f32 boundary inside the
        // primitive algebra changes the forward cotangents by several ulps.
        // Keep the boundary at the same completed-value site as XLA.
        let n_elec = r_carts.nrows();
        let mut out = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
        let periodic = screen && data.n_image > 1;
        // Electron-independent `N_p * c_p` (formed once per AO, not per point) and
        // one reused term buffer for the adjacent-tree primitive sum.
        let mut nc: Vec<f64> = Vec::new();
        let mut terms: Vec<f64> = Vec::new();
        for ao in 0..data.num_ao_kernel {
            let prims = &data.prim_groups[ao];
            nc.clear();
            nc.extend(prims.iter().map(|&p| {
                primitive_normalization_cart_fused_f64(data, p, ao) * data.coefficients[p]
            }));
            let min_exp = if periodic {
                prims
                    .iter()
                    .map(|&p| data.exponents[p])
                    .fold(f64::INFINITY, f64::min)
            } else {
                0.0
            };
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
                let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
                let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
                let r2 = x * x + y * y + z * z;
                if periodic && min_exp * r2 > LEPSBAS {
                    continue;
                }
                terms.clear();
                terms.extend(
                    prims
                        .iter()
                        .zip(&nc)
                        .map(|(&p, &ncp)| ncp * (-data.exponents[p] * r2).exp()),
                );
                let radial = reduce_sum_f64_in_place(&mut terms);
                let pow = |base: f64, exponent: i64| -> f64 {
                    let mut value = 1.0_f64;
                    for _ in 0..exponent {
                        value *= base;
                    }
                    value
                };
                let angular =
                    (pow(x, data.poly_x[ao]) * pow(y, data.poly_y[ao])) * pow(z, data.poly_z[ao]);
                out[[ao, e]] = f64::from((radial * angular) as f32);
            }
        }
        return out;
    }

    // The spherical fused kernel likewise owns its common radial/angular body
    // in ao_grad_lap (fp64) and rounds only the completed value output.
    compute_aos_kernel_screened(data, r_carts, screen).mapv(|v| f64::from(v as f32))
}

/// Point-major fp32 scratch of [`compute_aos_kernel_screened_f32`].
struct F32Points {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    r2: Vec<f32>,
    arg: Vec<f32>,
    ev: Vec<f32>,
    rad: Vec<f32>,
    dead: Vec<bool>,
}

impl F32Points {
    fn new(n: usize) -> Self {
        F32Points {
            x: vec![0.0; n],
            y: vec![0.0; n],
            z: vec![0.0; n],
            r2: vec![0.0; n],
            arg: vec![0.0; n],
            ev: vec![0.0; n],
            rad: vec![0.0; n],
            dead: vec![false; n],
        }
    }

    /// `r - c` formed in fp64, then cast (the zone-local operands), `r2` in fp32,
    /// and the periodic image screen `min_exp * r2 > LEPSBAS` in fp32.
    fn load(&mut self, r: &[Vec<f64>; 3], c: [f64; 3], screen_exp: Option<f32>) {
        let [rx, ry, rz] = r;
        for e in 0..self.x.len() {
            let x = (rx[e] - c[0]) as f32;
            let y = (ry[e] - c[1]) as f32;
            let z = (rz[e] - c[2]) as f32;
            self.x[e] = x;
            self.y[e] = y;
            self.z[e] = z;
            self.r2[e] = x * x + y * y + z * z;
        }
        match screen_exp {
            Some(m) => {
                for e in 0..self.x.len() {
                    self.dead[e] = m * self.r2[e] > LEPSBAS as f32;
                }
            }
            None => self.dead.fill(false),
        }
    }

    /// `ev[e] = exp(neg_z * r2[e])` over all points (one vector stream).
    fn exp_scaled(&mut self, neg_z: f32) {
        for e in 0..self.r2.len() {
            self.arg[e] = neg_z * self.r2[e];
        }
        crate::fastmath::exp_f32_into(&self.arg, &mut self.ev);
    }

    /// `ev[e] = exp(fl32(-z * r2[e]))` with the storage-precision exponent `z`:
    /// the product is formed in f64 and rounded ONCE to the f32 argument, as the
    /// periodic XLA kernels do with `(-Z_storage * r^2).astype(ao_eval)`.
    fn exp_scaled_storage(&mut self, z: f64) {
        for e in 0..self.r2.len() {
            self.arg[e] = (-z * f64::from(self.r2[e])) as f32;
        }
        crate::fastmath::exp_f32_into(&self.arg, &mut self.ev);
    }
}

/// Completed fp32 AO value `radial * angular` (the undefined `0 * non-finite`
/// tail product of a range-validated basis is replaced by 0).
#[inline(always)]
fn f32_ao_value(radial: f32, angular: f32) -> f64 {
    if radial == 0.0 && !angular.is_finite() {
        0.0
    } else {
        (radial * angular) as f64
    }
}

/// fp32 `ao_eval` kernel of the mixed zone (IMPL-RS-F32-AO-EVAL-SIMD-001): XLA's
/// zone-local fp32 AO value arithmetic, with every product, cast and reduction in
/// the operand order of a scalar per-point loop. The loops run over the points of a
/// shell (or AO), so the only transcendental -- fp32 `exp`, the vectorizable
/// [`crate::fastmath::exp_f32`] (XLA likewise uses its own polynomial, not libm)
/// -- streams over all points at once, and each exponential of a Cartesian shell
/// is shared by its components.
fn compute_aos_kernel_screened_f32(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> Array2<f64> {
    let n_elec = r_carts.nrows();
    let mut out = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    let periodic = screen && data.n_image > 1;
    let mut g = F32Points::new(n_elec);
    let rc: [Vec<f64>; 3] = [
        r_carts.column(0).to_vec(),
        r_carts.column(1).to_vec(),
        r_carts.column(2).to_vec(),
    ];

    if data.kind == AoKind::Sphe && !data.shells.is_empty() {
        for shell in &data.shells {
            // Molecular XLA forms the electron-independent N*N_lm*c in storage
            // precision, then rounds once to the AO zone. Reuse the already
            // hoisted f64 primitive normalization.
            let molecular = data.molecular_sphe;
            let nlm64 = (((2 * shell.l + 1) as f64) / (4.0 * std::f64::consts::PI)).sqrt();
            let prims = &data.prim_groups[shell.aos[0]];
            g.load(&rc, shell.center, periodic.then_some(shell.pz_min as f32));
            g.rad.fill(0.0);
            for (i, &p) in prims.iter().enumerate() {
                if molecular {
                    let pn = ((data.normalizations[p] * nlm64) * data.coefficients[p]) as f32;
                    g.exp_scaled(-(shell.prim_z[i] as f32));
                    for e in 0..n_elec {
                        g.rad[e] = pn.mul_add(g.ev[e], g.rad[e]);
                    }
                } else {
                    // Periodic XLA (`AOs_sphe_gto_data._compute_AOs`, non-guarded
                    // branch) keeps the whole electron-independent radial factor at
                    // storage precision -- N_n N_lm, c and Z; only r^2 and exp of the
                    // once-rounded exponent are f32, and N_n N_lm (c exp(-Z r^2)) is
                    // rounded ONCE to f32 (issue #145).
                    let z = shell.prim_z[i];
                    for e in 0..n_elec {
                        g.arg[e] = (-z * f64::from(g.r2[e])) as f32;
                    }
                    crate::fastmath::exp_f32_into(&g.arg, &mut g.ev);
                    let (c, nl) = (shell.prim_c[i], data.normalizations[p] * nlm64);
                    for e in 0..n_elec {
                        g.rad[e] += (nl * (c * f64::from(g.ev[e]))) as f32;
                    }
                }
            }
            for &ao in &shell.aos {
                let m = data.magnetic[ao];
                let mut row = out.row_mut(ao);
                for e in 0..n_elec {
                    if !g.dead[e] {
                        let angular = sphe_s_xla_f32(shell.l, m, g.x[e], g.y[e], g.z[e], g.r2[e]);
                        row[e] = f32_ao_value(g.rad[e], angular);
                    }
                }
            }
        }
        return out;
    }

    // Periodic XLA (`AOs_cart_gto_data._compute_AOs`, `pbc_flag` branch) keeps
    // the whole electron-independent radial factor at storage precision -- N, c
    // and Z; only r^2 and exp of the once-rounded exponent are f32, and
    // N (c exp(-Z r^2)) is rounded ONCE to f32 (issue #718, the convention of
    // the periodic spherical branch above, issue #145). Molecular Cartesian
    // leaves keep the f32 N, c and Z of the molecular XLA kernel.
    let cart_storage_radial = data.kind == AoKind::Cart && data.wrap_cell.is_some();

    if data.kind == AoKind::Cart && cart_f32_shells_tile(data) {
        let mut evs = Vec::<f32>::new();
        let (mut ax, mut ay, mut az) = (
            vec![0.0_f32; n_elec],
            vec![0.0_f32; n_elec],
            vec![0.0_f32; n_elec],
        );
        for shell in &data.shells {
            let nq = shell.prim_z.len();
            let min_exp = shell
                .prim_z
                .iter()
                .map(|&z| z as f32)
                .fold(f32::INFINITY, f32::min);
            g.load(&rc, shell.center, periodic.then_some(min_exp));
            evs.resize(nq * n_elec, 0.0);
            for q in 0..nq {
                if cart_storage_radial {
                    g.exp_scaled_storage(shell.prim_z[q]);
                } else {
                    g.exp_scaled(-(shell.prim_z[q] as f32));
                }
                evs[q * n_elec..(q + 1) * n_elec].copy_from_slice(&g.ev);
            }
            // The XLA CPU Cartesian bucket path casts AO centers to the zone dtype
            // before reconstructing its polynomial coordinates (the radial path
            // above intentionally casts after r-R).
            let c = shell.center;
            let c32 = [
                (c[0] as f32) as f64,
                (c[1] as f32) as f64,
                (c[2] as f32) as f64,
            ];
            for e in 0..n_elec {
                ax[e] = (rc[0][e] - c32[0]) as f32;
                ay[e] = (rc[1][e] - c32[1]) as f32;
                az[e] = (rc[2][e] - c32[2]) as f32;
            }
            for &ao in &shell.aos {
                g.rad.fill(0.0);
                for (q, &p) in data.prim_groups[ao].iter().enumerate() {
                    let ev = &evs[q * n_elec..(q + 1) * n_elec];
                    if cart_storage_radial {
                        let (pn, c) = (cart_norm_fused_f64_of(data, p, ao), data.coefficients[p]);
                        for (e, ev_e) in ev[..n_elec].iter().enumerate() {
                            g.rad[e] += (pn * (c * f64::from(*ev_e))) as f32;
                        }
                        continue;
                    }
                    let pn = cart_norm_f32_of(data, p, ao);
                    let cf = data.coefficients[p] as f32;
                    for (e, ev_e) in ev[..n_elec].iter().enumerate() {
                        g.rad[e] = pn.mul_add(cf * *ev_e, g.rad[e]);
                    }
                }
                let (px, py, pz) = (
                    data.poly_x[ao] as i32,
                    data.poly_y[ao] as i32,
                    data.poly_z[ao] as i32,
                );
                let mut row = out.row_mut(ao);
                for e in 0..n_elec {
                    if !g.dead[e] {
                        let angular = ax[e].powi(px) * ay[e].powi(py) * az[e].powi(pz);
                        row[e] = f32_ao_value(g.rad[e], angular);
                    }
                }
            }
        }
        return out;
    }

    for ao in 0..data.num_ao_kernel {
        let prims = &data.prim_groups[ao];
        let min_exp = prims
            .iter()
            .map(|&p| data.exponents[p] as f32)
            .fold(f32::INFINITY, f32::min);
        let c = [
            data.centers[[ao, 0]],
            data.centers[[ao, 1]],
            data.centers[[ao, 2]],
        ];
        g.load(&rc, c, periodic.then_some(min_exp));
        // Match the CPU XLA fusion of the dense primitive reduction:
        // `R = c*exp(...)`, followed by `radial = fma(N, R, radial)`.
        g.rad.fill(0.0);
        for &p in prims {
            if cart_storage_radial {
                g.exp_scaled_storage(data.exponents[p]);
                let (pn, c) = (cart_norm_fused_f64_of(data, p, ao), data.coefficients[p]);
                for e in 0..n_elec {
                    g.rad[e] += (pn * (c * f64::from(g.ev[e]))) as f32;
                }
                continue;
            }
            // Static AO normalization is hoisted exactly as XLA constant-folds it.
            let pn = cart_norm_f32_of(data, p, ao);
            let cf = data.coefficients[p] as f32;
            g.exp_scaled(-(data.exponents[p] as f32));
            for e in 0..n_elec {
                g.rad[e] = pn.mul_add(cf * g.ev[e], g.rad[e]);
            }
        }
        let c32 = [
            (c[0] as f32) as f64,
            (c[1] as f32) as f64,
            (c[2] as f32) as f64,
        ];
        let mut row = out.row_mut(ao);
        for e in 0..n_elec {
            if g.dead[e] {
                continue;
            }
            let angular = match data.kind {
                AoKind::Cart => {
                    let ax = (r_carts[[e, 0]] - c32[0]) as f32;
                    let ay = (r_carts[[e, 1]] - c32[1]) as f32;
                    let az = (r_carts[[e, 2]] - c32[2]) as f32;
                    ax.powi(data.poly_x[ao] as i32)
                        * ay.powi(data.poly_y[ao] as i32)
                        * az.powi(data.poly_z[ao] as i32)
                }
                AoKind::Sphe => data.sphe_angular[ao].eval_f32(g.x[e], g.y[e], g.z[e], g.r2[e]),
            };
            row[e] = f32_ao_value(g.rad[e], angular);
        }
    }
    out
}

/// Whether the Cartesian shells tile the kernel AOs exactly (each AO once, with
/// its shell's center bits and exponent bits in the same order), so the shared
/// per-shell exponentials of [`compute_aos_kernel_screened_f32`] are the per-AO
/// ones.
fn cart_f32_shells_tile(data: &AosGtoData) -> bool {
    if data.shells.is_empty() {
        return false;
    }
    let mut seen = vec![false; data.num_ao_kernel];
    for shell in &data.shells {
        for &ao in &shell.aos {
            if ao >= data.num_ao_kernel || seen[ao] {
                return false;
            }
            seen[ao] = true;
            let prims = &data.prim_groups[ao];
            if prims.len() != shell.prim_z.len()
                || (0..3).any(|c| data.centers[[ao, c]].to_bits() != shell.center[c].to_bits())
                || prims
                    .iter()
                    .zip(&shell.prim_z)
                    .any(|(&p, &z)| data.exponents[p].to_bits() != z.to_bits())
            {
                return false;
            }
        }
    }
    seen.iter().all(|&s| s)
}

/// Cartesian primitive normalization for the molecular `ao_eval` zone (mixed:
/// f32). This mirrors `AOs_cart_gto_data._compute_AOs`: the electron-independent
/// normalization is formed from the f64 storage values with the same
/// expression as the fused kernel and rounded once to f32. Building it from
/// f32 pieces overflowed `(8 Z)^l` or underflowed the factorial ratio although
/// the normalization itself is representable in f32 (issue #111).
#[inline(always)]
fn cart_norm_f32_of(data: &AosGtoData, p: usize, ao: usize) -> f32 {
    cart_norm_fused_f64_of(data, p, ao) as f32
}

/// Cached [`primitive_normalization_cart_fused_f64`] (bit-identical: the same
/// value formed once in `finalize`).
#[inline(always)]
fn cart_norm_fused_f64_of(data: &AosGtoData, p: usize, ao: usize) -> f64 {
    match data.cart_norm_fused_f64.get(p) {
        Some(&v) if data.cart_norm_fused_f64.len() == data.exponents.len() => v,
        _ => primitive_normalization_cart_fused_f64(data, p, ao),
    }
}

/// Cartesian normalization expression used by XLA's fused value/grad/lap
/// kernel.  In particular, `(8 Z)^l` is a static left-to-right multiply
/// chain, not the dynamic integer-power lowering used by the standalone AO
/// value kernel.
#[inline]
fn primitive_normalization_cart_fused_f64(data: &AosGtoData, p: usize, ao: usize) -> f64 {
    let nx = data.poly_x[ao];
    let ny = data.poly_y[ao];
    let nz = data.poly_z[ao];
    let factorial_ratio = (factorial(nx) * factorial(ny) * factorial(nz))
        / (factorial(2 * nx) * factorial(2 * ny) * factorial(2 * nz));
    let z = data.exponents[p];
    let base = 8.0 * z;
    let mut z_l = 1.0_f64;
    for _ in 0..data.angular_momentums[ao] {
        z_l *= base;
    }
    let n_z = (2.0 * z / std::f64::consts::PI).powf(1.5) * z_l;
    (n_z * factorial_ratio).sqrt()
}

/// Value-only regular solid harmonics with the same explicit expression tree
/// as `_compute_S_l_m`.  The XLA value kernel uses these named polynomials,
/// whereas Rust's generic factorial expansion is only algebraically equal and
/// differs by fp32 ulps after reassociation.
#[inline]
fn sphe_s_xla_f32(l: i64, m: i64, x: f32, y: f32, z: f32, r2: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let rn = if r2 > 0.0 { r2.sqrt() } else { 0.0 };
    let lnorm = |ll: i64| (4.0_f32 * pi / (2 * ll + 1) as f32).sqrt();
    // XLA's CPU lowering contracts the multiply-subtract polynomials below
    // into FMA instructions.  `mul_add` makes that rounding point explicit;
    // spelling these as separate `a*b-c` operations is one fp32 ULP away for
    // common f-shell coordinates and is amplified by the ECP mesh ratio.
    let sq_diff = |a: f32, b: f32| (-b).mul_add(b, a * a);
    let scaled_sq_diff = |scale: f32, a: f32, b2: f32| scale.mul_add(a * a, -b2);
    match (l, m) {
        (0, 0) => lnorm(0) * 0.5_f32 * (1.0_f32 / pi).sqrt() * rn.powf(0.0),
        (1, -1) => lnorm(1) * (3.0_f32 / (4.0_f32 * pi)).sqrt() * y,
        (1, 0) => lnorm(1) * (3.0_f32 / (4.0_f32 * pi)).sqrt() * z,
        (1, 1) => lnorm(1) * (3.0_f32 / (4.0_f32 * pi)).sqrt() * x,
        (2, -2) => lnorm(2) * 0.5_f32 * (15.0_f32 / pi).sqrt() * x * y,
        (2, -1) => lnorm(2) * 0.5_f32 * (15.0_f32 / pi).sqrt() * y * z,
        (2, 0) => lnorm(2) * 0.25_f32 * (5.0_f32 / pi).sqrt() * scaled_sq_diff(3.0, z, r2),
        (2, 1) => lnorm(2) * 0.5_f32 * (15.0_f32 / pi).sqrt() * x * z,
        (2, 2) => lnorm(2) * 0.25_f32 * (15.0_f32 / pi).sqrt() * sq_diff(x, y),
        (3, -3) => {
            lnorm(3)
                * 0.25_f32
                * (35.0_f32 / (2.0_f32 * pi)).sqrt()
                * y
                * scaled_sq_diff(3.0, x, y * y)
        }
        (3, -2) => lnorm(3) * 0.5_f32 * (105.0_f32 / pi).sqrt() * x * y * z,
        (3, -1) => {
            lnorm(3)
                * 0.25_f32
                * (21.0_f32 / (2.0_f32 * pi)).sqrt()
                * y
                * scaled_sq_diff(5.0, z, r2)
        }
        (3, 0) => {
            lnorm(3) * 0.25_f32 * (7.0_f32 / pi).sqrt() * z * 5.0_f32.mul_add(z * z, -3.0_f32 * r2)
        }
        (3, 1) => {
            lnorm(3)
                * 0.25_f32
                * (21.0_f32 / (2.0_f32 * pi)).sqrt()
                * x
                * scaled_sq_diff(5.0, z, r2)
        }
        (3, 2) => lnorm(3) * 0.25_f32 * (105.0_f32 / pi).sqrt() * sq_diff(x, y) * z,
        (3, 3) => {
            lnorm(3)
                * 0.25_f32
                * (35.0_f32 / (2.0_f32 * pi)).sqrt()
                * x
                * (-3.0_f32).mul_add(y * y, x * x)
        }
        (4, -4) => lnorm(4) * 0.75_f32 * (35.0_f32 / pi).sqrt() * x * y * (x.powi(2) - y.powi(2)),
        (4, -3) => {
            lnorm(4)
                * 0.75_f32
                * (35.0_f32 / (2.0_f32 * pi)).sqrt()
                * y
                * z
                * (3.0_f32 * x.powi(2) - y.powi(2))
        }
        (4, -2) => {
            lnorm(4) * 0.75_f32 * (5.0_f32 / pi).sqrt() * x * y * (7.0_f32 * z.powi(2) - rn.powi(2))
        }
        (4, -1) => {
            lnorm(4)
                * 0.75_f32
                * (5.0_f32 / (2.0_f32 * pi)).sqrt()
                * y
                * (7.0_f32 * z.powi(3) - 3.0_f32 * z * rn.powi(2))
        }
        (4, 0) => {
            lnorm(4)
                * (3.0_f32 / 16.0_f32)
                * (1.0_f32 / pi).sqrt()
                * (35.0_f32 * z.powi(4) - 30.0_f32 * z.powi(2) * rn.powi(2) + 3.0_f32 * rn.powi(4))
        }
        (4, 1) => {
            lnorm(4)
                * 0.75_f32
                * (5.0_f32 / (2.0_f32 * pi)).sqrt()
                * x
                * (7.0_f32 * z.powi(3) - 3.0_f32 * z * rn.powi(2))
        }
        (4, 2) => {
            lnorm(4)
                * (3.0_f32 / 8.0_f32)
                * (5.0_f32 / pi).sqrt()
                * (x.powi(2) - y.powi(2))
                * (7.0_f32 * z.powi(2) - rn.powi(2))
        }
        (4, 3) => {
            lnorm(4)
                * (0.75_f32
                    * (35.0_f32 / (2.0_f32 * pi)).sqrt()
                    * x
                    * z
                    * (x.powi(2) - 3.0_f32 * y.powi(2)))
        }
        (4, 4) => {
            lnorm(4)
                * (3.0_f32 / 16.0_f32)
                * (35.0_f32 / pi).sqrt()
                * (x.powi(2) * (x.powi(2) - 3.0_f32 * y.powi(2))
                    - y.powi(2) * (3.0_f32 * x.powi(2) - y.powi(2)))
        }
        _ => compute_s_l_m(l, m, x as f64, y as f64, z as f64, rn as f64) as f32,
    }
}

/// [`compute_aos_kernel`] with the runtime LEPSBAS distance screen switchable.
/// `screen == false` reproduces the XLA convention exactly (see [`compute_aos_exact`]).
pub(crate) fn compute_aos_kernel_screened(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> Array2<f64> {
    let n_elec = r_carts.nrows();
    let mut out = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));

    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    // Dynamic per-position image cutoff (TurboRVB epsbas; same as the move-loop
    // compute_aos_single path): for periodic AOs skip the kernel-AO whose most
    // diffuse primitive has already decayed below exp(-LEPSBAS) at this electron.
    // The e_L batch otherwise evaluated the FULL static kernel for every electron
    // (no skip) -- the periodic e_L hot spot, and inconsistent with the move loop.
    // `screen == false` (the exact/observable convention) disables the runtime skip
    // entirely -- every kernel AO of the static tiling is evaluated, as XLA does.
    let periodic = screen && data.n_image > 1;

    // Sphe shell-reuse (mirrors compute_aos_single): AOs sharing (center, l,
    // radial signature) share the radial sum `Sum norm*coef*exp(-Z r^2)`, so it
    // is computed ONCE per (shell, electron) and reused across the shell's 2l+1
    // angular components -- vs the per-AO loop below which recomputed the same
    // radial exp for every AO in the shell (the (2l+1)x e_L waste). Bit-identical:
    // the shell prim (z, norm*coef) order is the SAME `prim_groups[ao]` order, the
    // radial is the same sequential sum, and the shell `pz_min` equals the per-AO
    // `min_exp` so the periodic cutoff decision is unchanged.
    if data.kind == AoKind::Sphe && !data.shells.is_empty() && kernel_sphe_soa_ok(data) {
        kernel_sphe_soa(data, r_carts, periodic, &mut out);
        return out;
    }
    if data.kind == AoKind::Sphe && !data.shells.is_empty() {
        for shell in &data.shells {
            let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
            let pz = &shell.prim_z;
            let pnc = &shell.prim_nc;
            let pz_min = shell.pz_min;
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - cx;
                let y = r_carts[[e, 1]] - cy;
                let z = r_carts[[e, 2]] - cz;
                let r2 = x * x + y * y + z * z;
                if periodic && pz_min * r2 > LEPSBAS {
                    continue; // negligible image: AO rows stay 0
                }
                let mut radial_norm = 0.0_f64;
                for i in 0..pz.len() {
                    radial_norm += pnc[i] * exp_neg_fast_scalar(-pz[i] * r2);
                }
                for &ao in &shell.aos {
                    out[[ao, e]] = radial_norm * data.sphe_angular[ao].eval(x, y, z, r2);
                }
            }
        }
        return out;
    }

    // Cartesian counterpart of the shell-reuse path above.  Radial
    // normalizations remain AO-specific, but all Cartesian components in a
    // shell share the same exponent list and geometry, so each exponential is
    // evaluated once per (shell, electron) and fanned out to the components.
    if data.kind == AoKind::Cart && !data.shells.is_empty() && kernel_cart_soa_ok(data) {
        kernel_cart_soa(data, r_carts, periodic, &mut out);
        return out;
    }
    if data.kind == AoKind::Cart && !data.shells.is_empty() {
        for shell in &data.shells {
            let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - cx;
                let y = r_carts[[e, 1]] - cy;
                let z = r_carts[[e, 2]] - cz;
                let r2 = x * x + y * y + z * z;
                if periodic && shell.pz_min * r2 > LEPSBAS {
                    continue;
                }
                for q in 0..shell.prim_z.len() {
                    let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                    for &ao in &shell.aos {
                        let p = data.ao_prim_offset[ao] + q;
                        out[[ao, e]] += data.flat_nc[p] * ev;
                    }
                }
                for &ao in &shell.aos {
                    out[[ao, e]] *= x.powi(data.poly_x[ao] as i32)
                        * y.powi(data.poly_y[ao] as i32)
                        * z.powi(data.poly_z[ao] as i32);
                }
            }
        }
        return out;
    }

    for ao in 0..data.num_ao_kernel {
        let cx = data.centers[[ao, 0]];
        let cy = data.centers[[ao, 1]];
        let cz = data.centers[[ao, 2]];
        let prims = &groups[ao];
        let min_exp = if periodic {
            prims
                .iter()
                .map(|&p| data.exponents[p])
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };

        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - cx;
            let y = r_carts[[e, 1]] - cy;
            let z = r_carts[[e, 2]] - cz;
            let r2 = x * x + y * y + z * z;
            if periodic && min_exp * r2 > LEPSBAS {
                continue; // negligible image: out stays 0 (folded as 0)
            }
            let r_norm = r2.sqrt();

            // radial * normalization, summed over primitives
            let mut radial_norm = 0.0_f64;
            for &p in prims {
                let z_exp = data.exponents[p];
                let coeff = data.coefficients[p];
                radial_norm += norms[p] * coeff * exp_neg_fast_scalar(-z_exp * r2);
            }

            // angular part
            let angular = match data.kind {
                AoKind::Cart => {
                    let nx = data.poly_x[ao];
                    let ny = data.poly_y[ao];
                    let nz = data.poly_z[ao];
                    x.powi(nx as i32) * y.powi(ny as i32) * z.powi(nz as i32)
                }
                AoKind::Sphe => {
                    // Precomputed angular: no binom/factorial in the hot loop.
                    let _ = r_norm;
                    data.sphe_angular[ao].eval(x, y, z, r2)
                }
            };

            out[[ao, e]] = radial_norm * angular;
        }
    }
    out
}

/// `x^n` by the same binary exponentiation as `f64::powi` (compiler-rt `__powidf2`:
/// `r = 1; loop { if n odd: r *= x; n /= 2; if n == 0 break; x *= x }`), so a table
/// built from it reproduces `powi` bit for bit; `n` is loop-invariant at the call sites,
/// so the loops over points still vectorize.
#[inline(always)]
fn powi_exact(mut x: f64, mut n: u32) -> f64 {
    let mut r = 1.0_f64;
    loop {
        if n & 1 == 1 {
            r *= x;
        }
        n >>= 1;
        if n == 0 {
            break;
        }
        x *= x;
    }
    r
}

/// The SoA kernel needs the flat angular tables and 0 <= l <= 7 everywhere.
fn kernel_sphe_soa_ok(data: &AosGtoData) -> bool {
    data.ang_mc_off.len() == data.num_ao_kernel + 1
        && data.angular_momentums.iter().all(|&l| (0..=7).contains(&l))
}

/// Point-major (SoA) form of the spherical shell loop of [`compute_aos_kernel_screened`]:
/// per shell and chunk of points, the distances, the radial sum and every AO's angular
/// polynomial run as loops over points. Same terms in the same order as
/// `radial * SpheAngular::eval` (power tables reproduce `powi`), hence bitwise identical
/// kernel values; screened-out periodic images stay 0.
fn kernel_sphe_soa(data: &AosGtoData, r: &ArrayView2<f64>, periodic: bool, out: &mut Array2<f64>) {
    const CH: usize = 64;
    let n = r.nrows();
    let (mut dx, mut dy, mut dz, mut r2) =
        ([0.0_f64; CH], [0.0_f64; CH], [0.0_f64; CH], [0.0_f64; CH]);
    let mut rad = [0.0_f64; CH];
    let mut alive = [true; CH];
    let mut xp = [[0.0_f64; CH]; 8];
    let mut yp = [[0.0_f64; CH]; 8];
    let mut zp = [[0.0_f64; CH]; 8];
    let mut r2p = [[0.0_f64; CH]; 4];
    let (mut mv, mut lv) = ([0.0_f64; CH], [0.0_f64; CH]);
    for shell in &data.shells {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        let l = data.angular_momentums[shell.aos[0]] as usize;
        let mut e0 = 0;
        while e0 < n {
            let m = (n - e0).min(CH);
            let mut any = false;
            for i in 0..m {
                let e = e0 + i;
                dx[i] = r[[e, 0]] - cx;
                dy[i] = r[[e, 1]] - cy;
                dz[i] = r[[e, 2]] - cz;
                r2[i] = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
                alive[i] = !(periodic && shell.pz_min * r2[i] > LEPSBAS);
                any |= alive[i];
            }
            if !any {
                e0 += m;
                continue;
            }
            rad[..m].fill(0.0);
            for (q, &zq) in shell.prim_z.iter().enumerate() {
                let c = shell.prim_nc[q];
                for i in 0..m {
                    rad[i] += c * exp_neg_fast_scalar(-zq * r2[i]);
                }
            }
            for k in 0..=l {
                for i in 0..m {
                    xp[k][i] = powi_exact(dx[i], k as u32);
                    yp[k][i] = powi_exact(dy[i], k as u32);
                    zp[k][i] = powi_exact(dz[i], k as u32);
                }
            }
            for (k, r2p_k) in r2p[..l / 2 + 1].iter_mut().enumerate() {
                for i in 0..m {
                    r2p_k[i] = powi_exact(r2[i], k as u32);
                }
            }
            for &ao in &shell.aos {
                let mabs = data.ang_mabs[ao] as usize;
                mv[..m].fill(0.0);
                let ms = data.ang_mc_off[ao] as usize;
                for (p, t) in (ms..data.ang_mc_off[ao + 1] as usize).enumerate() {
                    let c = data.ang_mc[t];
                    let (a, b) = (&xp[p], &yp[mabs - p]);
                    for i in 0..m {
                        mv[i] += c * a[i] * b[i];
                    }
                }
                lv[..m].fill(0.0);
                for t in data.ang_lam_off[ao] as usize..data.ang_lam_off[ao + 1] as usize {
                    let c = data.ang_lam_c[t];
                    let (a, b) = (
                        &r2p[data.ang_lam_k[t] as usize],
                        &zp[data.ang_lam_ez[t] as usize],
                    );
                    for i in 0..m {
                        lv[i] += c * a[i] * b[i];
                    }
                }
                let mut row = out.row_mut(ao);
                for i in 0..m {
                    if alive[i] {
                        row[e0 + i] = rad[i] * (lv[i] * mv[i]);
                    }
                }
            }
            e0 += m;
        }
    }
}

/// Point-major (SoA) form of the Cartesian shell loop of [`compute_aos_kernel_screened`]:
/// same `Sum_q nc * exp` accumulation per AO and the same `(x^nx * y^ny) * z^nz` product
/// (powers via [`powi_exact`]), hence bitwise identical kernel values; screened-out
/// periodic images stay 0. Requires shells of at most 16 components and non-negative orders.
fn kernel_cart_soa(data: &AosGtoData, r: &ArrayView2<f64>, periodic: bool, out: &mut Array2<f64>) {
    const CH: usize = 64;
    const NA: usize = 16;
    let n = r.nrows();
    let (mut dx, mut dy, mut dz, mut r2) =
        ([0.0_f64; CH], [0.0_f64; CH], [0.0_f64; CH], [0.0_f64; CH]);
    let mut alive = [true; CH];
    let mut ev = [0.0_f64; CH];
    let mut acc = [[0.0_f64; CH]; NA];
    for shell in &data.shells {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        let na = shell.aos.len();
        let mut e0 = 0;
        while e0 < n {
            let m = (n - e0).min(CH);
            let mut any = false;
            for i in 0..m {
                let e = e0 + i;
                dx[i] = r[[e, 0]] - cx;
                dy[i] = r[[e, 1]] - cy;
                dz[i] = r[[e, 2]] - cz;
                r2[i] = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
                alive[i] = !(periodic && shell.pz_min * r2[i] > LEPSBAS);
                any |= alive[i];
            }
            if !any {
                e0 += m;
                continue;
            }
            for row in acc.iter_mut().take(na) {
                row[..m].fill(0.0);
            }
            for (q, &zq) in shell.prim_z.iter().enumerate() {
                for i in 0..m {
                    ev[i] = exp_neg_fast_scalar(-zq * r2[i]);
                }
                for (j, &ao) in shell.aos.iter().enumerate() {
                    let c = data.flat_nc[data.ao_prim_offset[ao] + q];
                    let row = &mut acc[j];
                    for i in 0..m {
                        row[i] += c * ev[i];
                    }
                }
            }
            for (j, &ao) in shell.aos.iter().enumerate() {
                let (nx, ny, nz) = (
                    data.poly_x[ao] as u32,
                    data.poly_y[ao] as u32,
                    data.poly_z[ao] as u32,
                );
                let a = &acc[j];
                let mut row = out.row_mut(ao);
                for i in 0..m {
                    if alive[i] {
                        row[e0 + i] = a[i]
                            * (powi_exact(dx[i], nx)
                                * powi_exact(dy[i], ny)
                                * powi_exact(dz[i], nz));
                    }
                }
            }
            e0 += m;
        }
    }
}

fn kernel_cart_soa_ok(data: &AosGtoData) -> bool {
    data.shells.iter().all(|s| s.aos.len() <= 16)
        && data
            .poly_x
            .iter()
            .chain(&data.poly_y)
            .chain(&data.poly_z)
            .all(|&p| (0..=31).contains(&p))
}

/// Fast single-position AO value evaluation (the per-move hot path).
/// Returns a `Vec<f64>` of length `num_ao` (image-folded). Uses RAW SLICES and
/// no `Array2` allocation, eliminating the ndarray bounds-check + 2D-index
/// overhead that dominated `orb_eval` (~85% of the move) in the batch path.
pub fn compute_aos_single(data: &AosGtoData, pos: [f64; 3]) -> Vec<f64> {
    if data.precision_policy.is_f32(data.value_zone) {
        return compute_aos_single_f32(data, pos);
    }
    // No home-cell wrap here. This is a per-move sweep value path; the native
    // sweep reproduces JAX's accept/reject calculation, and wrapping
    // perturbs the Metropolis ratio enough to flip a decision (walker drift ->
    // force_HF error on the pgto+bspline composite). The far-image wrap
    // is applied on the BATCH grad/lap OBSERVABLE paths only (sweep uses VALUE, not
    // grad/lap). See wrap_pos / wrap_r docs.
    let nk = data.num_ao_kernel;
    let mut out = vec![0.0_f64; nk];

    // Fast sphe path: shell-reuse (exp once per shell, reused across 2l+1 AOs).
    if data.kind == AoKind::Sphe && !data.shells.is_empty() {
        for shell in &data.shells {
            let dx = pos[0] - shell.center[0];
            let dy = pos[1] - shell.center[1];
            let dz = pos[2] - shell.center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let mut rad = 0.0_f64;
            let pz = &shell.prim_z;
            let pnc = &shell.prim_nc;
            for i in 0..pz.len() {
                rad += pnc[i] * exp_neg_fast_scalar(-pz[i] * r2);
            }
            for &ao in &shell.aos {
                out[ao] = rad * data.sphe_angular[ao].eval(dx, dy, dz, r2);
            }
        }
        return fold_single(data, out);
    }

    // Cartesian shell reuse.  Cartesian primitive normalizations depend on
    // (nx, ny, nz), so the radial sums differ between angular components, but
    // the expensive exp(-zeta*r^2) values do not.  Evaluate each primitive exp
    // once per shell and accumulate it into every Cartesian component.  The
    // primitive order within each AO is unchanged, preserving the scalar
    // accumulation result while removing up to (l+1)(l+2)/2 duplicate exp calls.
    if data.kind == AoKind::Cart && !data.shells.is_empty() {
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            for &ao in &shell.aos {
                out[ao] = 0.0;
            }
            for q in 0..shell.prim_z.len() {
                let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                for &ao in &shell.aos {
                    let p = data.ao_prim_offset[ao] + q;
                    out[ao] += data.flat_nc[p] * ev;
                }
            }
            for &ao in &shell.aos {
                out[ao] *= x.powi(data.poly_x[ao] as i32)
                    * y.powi(data.poly_y[ao] as i32)
                    * z.powi(data.poly_z[ao] as i32);
            }
        }
        return fold_single(data, out);
    }

    let centers = data
        .centers
        .as_slice()
        .expect("centers contiguous (C-order)");
    let flat_nc: &[f64] = &data.flat_nc;
    let flat_exp: &[f64] = &data.flat_exp;
    let off: &[usize] = &data.ao_prim_offset;

    for ao in 0..nk {
        let cx = centers[ao * 3];
        let cy = centers[ao * 3 + 1];
        let cz = centers[ao * 3 + 2];
        let x = pos[0] - cx;
        let y = pos[1] - cy;
        let z = pos[2] - cz;
        let r2 = x * x + y * y + z * z;

        // contiguous primitive range for this AO (no per-prim indirection)
        let mut radial = 0.0_f64;
        let (s, e) = (off[ao], off[ao + 1]);
        for (&c, &a) in flat_nc[s..e].iter().zip(&flat_exp[s..e]) {
            radial += c * exp_neg_fast_scalar(-a * r2);
        }

        let ang = match data.kind {
            AoKind::Sphe => data.sphe_angular[ao].eval(x, y, z, r2),
            AoKind::Cart => {
                x.powi(data.poly_x[ao] as i32)
                    * y.powi(data.poly_y[ao] as i32)
                    * z.powi(data.poly_z[ao] as i32)
            }
        };
        out[ao] = radial * ang;
    }

    fold_single(data, out)
}

pub(crate) fn compute_aos_single_with_policy(
    data: &AosGtoData,
    pos: [f64; 3],
    f32_arithmetic: bool,
) -> Vec<f64> {
    if f32_arithmetic {
        compute_aos_single_f32(data, pos)
    } else {
        // Avoid consulting the standalone policy on a context-owned path.
        let out = compute_aos_kernel_screened(
            data,
            &Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap().view(),
            false,
        );
        fold_single(data, out.column(0).to_vec())
    }
}

fn compute_aos_single_f32(data: &AosGtoData, pos: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0_f64; data.num_ao];
    compute_aos_single_f32_into(data, pos, false, &mut out);
    out
}

/// Single-position fp32 `ao_eval` value kernel into `out` (length `num_ao`,
/// image-folded), without the per-call batch scratch of
/// [`compute_aos_kernel_screened_f32`].
///
/// `screen == false`: every value is bit-identical to the one-point batch
/// `fold_image_axis_f32(compute_aos_kernel_screened_f32(data, pos, false))`, i.e.
/// to [`compute_aos_with_policy`]`(.., true)`: the same zone-local operands, casts,
/// fused multiply-adds and products per point, and the same adjacent-tree fp32
/// image fold over the kernel AOs of each home AO in ascending order.
///
/// `screen == true` is the FAST per-move variant: a shell with
/// `pz_val * r^2 > LEPSBAS` (r^2 formed in fp64, as in the fp64 FAST kernels) is
/// skipped, so its kernel AOs contribute 0. Every value of a skipped shell is below
/// `FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units ([`Shell::pz_val`], issue
/// #100); the kept shells keep the `screen == false` arithmetic.
pub(crate) fn compute_aos_single_f32_into(
    data: &AosGtoData,
    pos: [f64; 3],
    screen: bool,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), data.num_ao);
    if data.shells.is_empty() {
        // Un-finalized data (`finalize` builds the shells and the home-kernel
        // CSR together): the batch kernel itself (its per-AO fallback).
        let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
        let folded = fold_image_axis_f32(
            data,
            compute_aos_kernel_screened_f32(data, &r.view(), false),
        );
        for (dst, src) in out.iter_mut().zip(folded.column(0)) {
            *dst = *src;
        }
        return;
    }
    if data.n_image == 1 {
        single_kernel_f32_into(data, pos, screen, out);
        return;
    }
    let mut ker = vec![0.0_f64; data.num_ao_kernel];
    single_kernel_f32_into(data, pos, screen, &mut ker);
    let mut buf = Vec::<f32>::new();
    for (h, dst) in out.iter_mut().enumerate() {
        let ks = &data.home_kernel[data.home_kernel_off[h]..data.home_kernel_off[h + 1]];
        if ks.is_empty() {
            *dst = 0.0;
            continue;
        }
        buf.clear();
        buf.extend(ks.iter().map(|&k| ker[k] as f32));
        // `reduce_sum_f32` / `fold_image_axis_f32`: adjacent binary tree,
        // pair i <- (2i, 2i+1), odd tail carried.
        let mut len = buf.len();
        while len > 1 {
            let pairs = len / 2;
            for i in 0..pairs {
                buf[i] = buf[2 * i] + buf[2 * i + 1];
            }
            if len % 2 == 1 {
                buf[pairs] = buf[len - 1];
            }
            len = pairs + len % 2;
        }
        *dst = f64::from(buf[0]);
    }
}

/// Kernel-rank (pre-fold) body of [`compute_aos_single_f32_into`] for finalized
/// data (`shells` non-empty). `ker` has `num_ao_kernel` entries.
fn single_kernel_f32_into(data: &AosGtoData, pos: [f64; 3], screen: bool, ker: &mut [f64]) {
    ker.fill(0.0);
    let skip = |shell: &Shell| -> bool {
        if !screen {
            return false;
        }
        let dx = pos[0] - shell.center[0];
        let dy = pos[1] - shell.center[1];
        let dz = pos[2] - shell.center[2];
        shell.pz_val * (dx * dx + dy * dy + dz * dz) > LEPSBAS
    };
    match data.kind {
        AoKind::Sphe => {
            for shell in &data.shells {
                if skip(shell) {
                    continue;
                }
                let c = shell.center;
                let x = (pos[0] - c[0]) as f32;
                let y = (pos[1] - c[1]) as f32;
                let z = (pos[2] - c[2]) as f32;
                let r2 = x * x + y * y + z * z;
                let nlm64 = (((2 * shell.l + 1) as f64) / (4.0 * std::f64::consts::PI)).sqrt();
                let prims = &data.prim_groups[shell.aos[0]];
                let mut rad = 0.0_f32;
                for (i, &p) in prims.iter().enumerate() {
                    if data.molecular_sphe {
                        let pn = ((data.normalizations[p] * nlm64) * data.coefficients[p]) as f32;
                        let ev = crate::fastmath::exp_f32(-(shell.prim_z[i] as f32) * r2);
                        rad = pn.mul_add(ev, rad);
                    } else {
                        let ev =
                            crate::fastmath::exp_f32((-shell.prim_z[i] * f64::from(r2)) as f32);
                        let (cf, nl) = (shell.prim_c[i], data.normalizations[p] * nlm64);
                        rad += (nl * (cf * f64::from(ev))) as f32;
                    }
                }
                for &ao in &shell.aos {
                    let angular = sphe_s_xla_f32(shell.l, data.magnetic[ao], x, y, z, r2);
                    ker[ao] = f32_ao_value(rad, angular);
                }
            }
        }
        AoKind::Cart => {
            // Finalized Cartesian shells tile the kernel AOs (one shell per
            // (center, l, exponent/coefficient) signature), which is the
            // `cart_f32_shells_tile` condition of the batch shell-reuse path.
            let cart_storage_radial = data.wrap_cell.is_some();
            for shell in &data.shells {
                if skip(shell) {
                    continue;
                }
                let c = shell.center;
                let x = (pos[0] - c[0]) as f32;
                let y = (pos[1] - c[1]) as f32;
                let z = (pos[2] - c[2]) as f32;
                let r2 = x * x + y * y + z * z;
                // The radial sums live in `ker` as exact f32 values until the product.
                // Periodic leaves keep N, c and Z at storage precision and round
                // N (c exp(-Z r^2)) once, as the batch kernel (issue #718).
                for (q, &zq) in shell.prim_z.iter().enumerate() {
                    if cart_storage_radial {
                        let ev = crate::fastmath::exp_f32((-zq * f64::from(r2)) as f32);
                        for &ao in &shell.aos {
                            let p = data.prim_groups[ao][q];
                            let (pn, c) =
                                (cart_norm_fused_f64_of(data, p, ao), data.coefficients[p]);
                            ker[ao] = f64::from(ker[ao] as f32 + (pn * (c * f64::from(ev))) as f32);
                        }
                        continue;
                    }
                    let ev = crate::fastmath::exp_f32(-(zq as f32) * r2);
                    for &ao in &shell.aos {
                        let p = data.prim_groups[ao][q];
                        let pn = cart_norm_f32_of(data, p, ao);
                        let cf = data.coefficients[p] as f32;
                        ker[ao] = f64::from(pn.mul_add(cf * ev, ker[ao] as f32));
                    }
                }
                let c32 = [
                    (c[0] as f32) as f64,
                    (c[1] as f32) as f64,
                    (c[2] as f32) as f64,
                ];
                let ax = (pos[0] - c32[0]) as f32;
                let ay = (pos[1] - c32[1]) as f32;
                let az = (pos[2] - c32[2]) as f32;
                for &ao in &shell.aos {
                    let (px, py, pz) = (
                        data.poly_x[ao] as i32,
                        data.poly_y[ao] as i32,
                        data.poly_z[ao] as i32,
                    );
                    let angular = ax.powi(px) * ay.powi(py) * az.powi(pz);
                    ker[ao] = f32_ao_value(ker[ao] as f32, angular);
                }
            }
        }
    }
}

/// Vectorized fp64 `exp` over a flat buffer (`dst[i] = exp(src[i])`).
/// Single portable implementation: an 8-wide chunked loop over [`exp_neg_fast_scalar`],
/// auto-vectorized by the compiler under `target-cpu=native` (round / FMA /
/// integer scale construction all map to vector instructions). The fused
/// per-shell path inlines the same scalar core, so every exp route in the
/// crate shares ONE set of numerics. Arguments are AO primitive exponents
/// (`-Z*r^2 <= 0`); tails below -700 are clamped inside the core, so no
/// overflow handling is needed.
pub fn exp_pd_into(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len().min(dst.len());
    let (src, dst) = (&src[..n], &mut dst[..n]);
    for i in 0..n {
        dst[i] = exp_neg_fast_scalar(src[i]);
    }
}

/// Scalar exp core for non-positive AO arguments: range reduction
/// `x = k*ln2 + r` (k = round(x*log2e), `|r| <= ln2/2`) + degree-9 Taylor
/// (Horner/FMA) + direct binary64 `2^k` construction. This is the SINGLE exp
/// implementation -- both the batched [`exp_pd_into`] loop and the fused
/// per-shell single-electron path inline it (no argument/exp scratch traffic,
/// no libm call per shell primitive). AO arguments are non-positive; tails
/// below -700 are physically negligible and are clamped so the normal
/// binary64 scale construction is always valid.
#[inline(always)]
fn exp_neg_fast_scalar(x: f64) -> f64 {
    crate::fastmath::exp_neg(x)
}

/// Vectorized fp64 `cos` over a flat buffer (`dst[i] = cos(src[i])`, ~1 ULP).
/// The plane-wave AO value hot path evaluates cos over the (num_pw x n_elec)
/// argument matrix. Portable loop over [`sincos_core`]; auto-vectorized under
/// `target-cpu=native`.
#[inline]
pub fn cos_pd_into(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    let n8 = src.len() & !7;
    for (sc, dc) in src[..n8].chunks_exact(8).zip(dst[..n8].chunks_exact_mut(8)) {
        for l in 0..8 {
            dc[l] = sincos_core(sc[l]).1;
        }
    }
    for i in n8..src.len() {
        dst[i] = sincos_core(src[i]).1;
    }
}

/// In-place `cos` (`buf[i] = cos(buf[i])`); the plane-wave value path writes
/// theta into the output buffer (via the `G @ r^T` gemm) then cos-es it in
/// place.
#[inline]
pub fn cos_pd_inplace(buf: &mut [f64]) {
    let n8 = buf.len() & !7;
    for bc in buf[..n8].chunks_exact_mut(8) {
        for bc_l in bc[..8].iter_mut() {
            *bc_l = sincos_core(*bc_l).1;
        }
    }
    for v in buf[n8..].iter_mut() {
        *v = sincos_core(*v).1;
    }
}

/// Vectorized fp64 `sin_cos` (`sin_dst[i], cos_dst[i] = src[i].sin_cos()`,
/// ~1 ULP). Used by the plane-wave value+grad+lap e_L kernel. Portable loop
/// over [`sincos_core`].
#[inline]
pub fn sincos_pd_into(src: &[f64], sin_dst: &mut [f64], cos_dst: &mut [f64]) {
    debug_assert_eq!(src.len(), sin_dst.len());
    debug_assert_eq!(src.len(), cos_dst.len());
    let n8 = src.len() & !7;
    for ((sc, sd), cd) in src[..n8]
        .chunks_exact(8)
        .zip(sin_dst[..n8].chunks_exact_mut(8))
        .zip(cos_dst[..n8].chunks_exact_mut(8))
    {
        for l in 0..8 {
            let (s, c) = sincos_core(sc[l]);
            sd[l] = s;
            cd[l] = c;
        }
    }
    for i in n8..src.len() {
        let (s, c) = sincos_core(src[i]);
        sin_dst[i] = s;
        cos_dst[i] = c;
    }
}

/// Scalar fp64 sin/cos core (~1 ULP); the single implementation lives in
/// [`crate::fastmath::sincos`] (shared with the FAST periodic J1/J2 distance).
#[inline(always)]
fn sincos_core(x: f64) -> (f64, f64) {
    crate::fastmath::sincos(x)
}

/// No-alloc vectorized single-position AO eval (sphe molecular). Fills `out`
/// (`num_ao`). `arg`/`expv` are scratch of length >= total shell primitives,
/// `sh_geo` of length >= nshell. The 2060 primitive `exp`s are batched into one
/// AVX-512 pass (vs scalar libm exp per primitive). Retained for reference;
/// see [`exp_pd_into`] note -- the fused scalar path is faster in the hot loop.
#[allow(dead_code)]
pub fn compute_aos_single_fast(
    data: &AosGtoData,
    pos: [f64; 3],
    arg: &mut [f64],
    expv: &mut [f64],
    sh_geo: &mut [[f64; 4]],
    out: &mut [f64],
) {
    let nshell = data.shells.len();
    // pass 1: per-shell geometry + fill exp arguments (-Z*r^2).
    for (s, sh_geo_s) in sh_geo[..nshell].iter_mut().enumerate() {
        let dx = pos[0] - data.sh_center[s * 3];
        let dy = pos[1] - data.sh_center[s * 3 + 1];
        let dz = pos[2] - data.sh_center[s * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;
        *sh_geo_s = [dx, dy, dz, r2];
        let (a, b) = (data.sh_off[s] as usize, data.sh_off[s + 1] as usize);
        for (p, arg_p) in arg[..b].iter_mut().enumerate().skip(a) {
            *arg_p = data.sh_neg_pz[p] * r2;
        }
    }
    let total = data.sh_neg_pz.len();
    // pass 2: one vectorized exp over all primitives.
    exp_pd_into(&arg[..total], &mut expv[..total]);
    // pass 3: per-shell radial sum x each AO's angular part.
    for (s, sh_geo_s) in sh_geo[..nshell].iter().enumerate() {
        let (a, b) = (data.sh_off[s] as usize, data.sh_off[s + 1] as usize);
        let mut rad = 0.0_f64;
        for (p, expv_p) in expv[..b].iter().enumerate().skip(a) {
            rad += data.sh_pnc[p] * *expv_p;
        }
        let [dx, dy, dz, r2] = *sh_geo_s;
        for &ao in &data.shells[s].aos {
            out[ao] = rad * data.sphe_angular[ao].eval(dx, dy, dz, r2);
        }
    }
}

/// Cartesian molecular sibling of [`compute_aos_single_fast`]. All shell
/// primitive exponentials are gathered into caller-owned scratch and evaluated
/// in one AVX2/AVX-512 pass; AO-specific normalized coefficients are applied in
/// the reduction because Cartesian component normalizations differ.
pub fn compute_aos_single_cart_fast(
    data: &AosGtoData,
    pos: [f64; 3],
    arg: &mut [f64],
    expv: &mut [f64],
    sh_geo: &mut [[f64; 4]],
    out: &mut [f64],
) {
    debug_assert!(data.kind == AoKind::Cart);
    debug_assert_eq!(data.n_image, 1);
    let nshell = data.shells.len();
    for (s, sh_geo_s) in sh_geo[..nshell].iter_mut().enumerate() {
        let shell = &data.shells[s];
        let x = pos[0] - shell.center[0];
        let y = pos[1] - shell.center[1];
        let z = pos[2] - shell.center[2];
        let r2 = x * x + y * y + z * z;
        *sh_geo_s = [x, y, z, r2];
        let base = data.sh_off[s] as usize;
        for q in 0..shell.prim_z.len() {
            arg[base + q] = -shell.prim_z[q] * r2;
        }
    }
    let total = data.sh_neg_pz.len();
    exp_pd_into(&arg[..total], &mut expv[..total]);
    for (s, sh_geo_s) in sh_geo[..nshell].iter().enumerate() {
        let shell = &data.shells[s];
        let [x, y, z, _] = *sh_geo_s;
        let base = data.sh_off[s] as usize;
        for &ao in &shell.aos {
            let mut radial = 0.0_f64;
            for q in 0..shell.prim_z.len() {
                radial += data.flat_nc[data.ao_prim_offset[ao] + q] * expv[base + q];
            }
            out[ao] = radial
                * x.powi(data.poly_x[ao] as i32)
                * y.powi(data.poly_y[ao] as i32)
                * z.powi(data.poly_z[ao] as i32);
        }
    }
}

/// No-alloc periodic single-position AO eval (static image tiling, no truncation).
/// Accumulates DIRECTLY into the image-folded `out` (length `num_ao`), skipping
/// the `n_image*num_ao` kernel buffer and the separate fold pass. Sphe only.
/// No runtime distance screen: every image shell of the static tiling is
/// evaluated (the XLA convention), only accumulated straight into the home rank.
pub fn compute_aos_single_fold_into(data: &AosGtoData, pos: [f64; 3], out: &mut [f64]) {
    debug_assert_eq!(out.len(), data.num_ao);
    // NB: no home-cell wrap. This is the per-move SWEEP value path and follows
    // the same static image tiling as XLA.
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let nshell = data.sh_pzmin.len();
    let cen = &data.sh_center; // 3*nshell, contiguous
    let off = &data.sh_off; // primitive ranges
    let neg_pz = &data.sh_neg_pz; // already -Z
    let pnc = &data.sh_pnc;
    let ao_off = &data.sh_ao_off;
    let ao_ker = &data.sh_ao_kernel;
    let ao_home = &data.sh_ao_home;
    let ang = &data.sphe_angular;
    for s in 0..nshell {
        let dx = pos[0] - cen[s * 3];
        let dy = pos[1] - cen[s * 3 + 1];
        let dz = pos[2] - cen[s * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let mut rad = 0.0_f64;
        let (ps, pe) = (off[s] as usize, off[s + 1] as usize);
        for p in ps..pe {
            rad += pnc[p] * exp_neg_fast_scalar(neg_pz[p] * r2);
        }
        let (as_, ae) = (ao_off[s] as usize, ao_off[s + 1] as usize);
        for j in as_..ae {
            let ker = ao_ker[j] as usize;
            out[ao_home[j] as usize] += rad * ang[ker].eval(dx, dy, dz, r2);
        }
    }
}

/// FAST-mode periodic sibling of [`compute_aos_single_fold_into`]: the same
/// image-folded single-position AO value, but every (image) shell with
/// `pz_val*r^2 > LEPSBAS` is skipped -- every AO value it would add is below
/// `FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units ([`Shell::pz_val`], the
/// value screen of [`compute_mo_single_cutoff_into`] / [`dot_aos_single_cutoff_sphe`],
/// issue #100). Kept shells use exactly the arithmetic of the exact kernel
/// (Sphe: flat `sh_*` radial + `sphe_angular`; Cart: per-AO `flat_nc` radial and
/// `powi` monomial as [`compute_mo_single_cutoff_cart_into`]), accumulated in the
/// same shell-ascending order into the home rank. Used by the FAST energy
/// chunk's periodic sweep only; the exact sweep keeps [`compute_aos_single_fold_into`].
pub fn compute_aos_single_fold_cutoff_into(data: &AosGtoData, pos: [f64; 3], out: &mut [f64]) {
    debug_assert_eq!(out.len(), data.num_ao);
    debug_assert!(!data.shells.is_empty());
    for x in out.iter_mut() {
        *x = 0.0;
    }
    match data.kind {
        AoKind::Sphe => {
            let nshell = data.sh_pzval.len();
            let cen = &data.sh_center;
            let pzval = &data.sh_pzval;
            let off = &data.sh_off;
            let neg_pz = &data.sh_neg_pz;
            let pnc = &data.sh_pnc;
            let ao_off = &data.sh_ao_off;
            let ao_ker = &data.sh_ao_kernel;
            let ao_home = &data.sh_ao_home;
            let ang = &data.sphe_angular;
            for s in 0..nshell {
                let dx = pos[0] - cen[s * 3];
                let dy = pos[1] - cen[s * 3 + 1];
                let dz = pos[2] - cen[s * 3 + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if pzval[s] * r2 > LEPSBAS {
                    continue; // every AO of the shell < FAST_SCREEN_ABS_BOUND (absolute)
                }
                let mut rad = 0.0_f64;
                let (ps, pe) = (off[s] as usize, off[s + 1] as usize);
                for p in ps..pe {
                    rad += pnc[p] * exp_neg_fast_scalar(neg_pz[p] * r2);
                }
                let (as_, ae) = (ao_off[s] as usize, ao_off[s + 1] as usize);
                for j in as_..ae {
                    let ker = ao_ker[j] as usize;
                    out[ao_home[j] as usize] += rad * ang[ker].eval(dx, dy, dz, r2);
                }
            }
        }
        AoKind::Cart => {
            for shell in &data.shells {
                let dx = pos[0] - shell.center[0];
                let dy = pos[1] - shell.center[1];
                let dz = pos[2] - shell.center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if shell.pz_val * r2 > LEPSBAS {
                    continue; // whole shell < FAST_SCREEN_ABS_BOUND (absolute)
                }
                for &ao in &shell.aos {
                    let (s, e) = (data.ao_prim_offset[ao], data.ao_prim_offset[ao + 1]);
                    let mut rad = 0.0_f64;
                    for p in s..e {
                        rad += data.flat_nc[p] * exp_neg_fast_scalar(-data.flat_exp[p] * r2);
                    }
                    let val = rad
                        * dx.powi(data.poly_x[ao] as i32)
                        * dy.powi(data.poly_y[ao] as i32)
                        * dz.powi(data.poly_z[ao] as i32);
                    out[data.kernel_to_home[ao]] += val;
                }
            }
        }
    }
}

/// Weighted single-point AO contraction with the per-shell distance cutoff:
/// returns `Sum_ao w[ao] * AO_ao(pos)` (`w` length `num_ao`, home rank) without
/// materialising the AO or MO vector. Shell-cutoff semantics identical to
/// [`compute_mo_single_cutoff_into`] (`pz_val*r^2 > LEPSBAS` -> shell dropped;
/// every dropped AO value `< FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units, see
/// [`Shell::pz_val`]); image folding via `sh_ao_home` (identity for molecular;
/// on a periodic leaf each image shell is screened by the same per-shell bound). Sphe only (flat `sh_*` layout).
/// The ECP-nl fast path contracts each mesh point against a precomputed
/// electron-fixed weight vector, turning the per-point cost from
/// O(n_surv*num_mo) into O(n_surv).
pub fn dot_aos_single_cutoff_sphe(data: &AosGtoData, pos: [f64; 3], w: &[f64]) -> f64 {
    debug_assert_eq!(w.len(), data.num_ao);
    let nshell = data.sh_pzval.len();
    let cen = &data.sh_center;
    let pzval = &data.sh_pzval;
    let off = &data.sh_off;
    let neg_pz = &data.sh_neg_pz;
    let pnc = &data.sh_pnc;
    let ao_off = &data.sh_ao_off;
    let ao_ker = &data.sh_ao_kernel;
    let ao_home = &data.sh_ao_home;
    let mut acc = 0.0_f64;
    for s in 0..nshell {
        let dx = pos[0] - cen[s * 3];
        let dy = pos[1] - cen[s * 3 + 1];
        let dz = pos[2] - cen[s * 3 + 2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if pzval[s] * r2 > LEPSBAS {
            continue; // every AO of the shell < FAST_SCREEN_ABS_BOUND (absolute)
        }
        let mut rad = 0.0_f64;
        let (ps, pe) = (off[s] as usize, off[s + 1] as usize);
        for p in ps..pe {
            rad += pnc[p] * exp_neg_fast_scalar(neg_pz[p] * r2);
        }
        let (as_, ae) = (ao_off[s] as usize, ao_off[s + 1] as usize);
        for j in as_..ae {
            let ker = ao_ker[j] as usize;
            acc += rad * eval_angular_flat(data, ker, dx, dy, dz, r2) * w[ao_home[j] as usize];
        }
    }
    acc
}

/// Multi-point sibling of [`dot_aos_single_cutoff_sphe`]: contracts ALL `nv`
/// quadrature points of one ECP (electron, channel) mesh against the same weight
/// vector in ONE shell sweep -- `out[v] = Sum_ao w[ao]*AO_ao(pts[v])`. The shell
/// metadata (center, prim range, AO range) is loaded once per shell instead of
/// once per (shell, point), and the radial `exp` runs 8-lane (`exp_pd_into`,
/// AVX-512 where available) across the points: the mesh dot is exp-bound. Same
/// per-(shell, point) `pz_val*r2 > LEPSBAS` cutoff ([`Shell::pz_val`]: every
/// dropped AO value `< FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units). Cutoff/fast mode only (the
/// vector exp is a <=1-ULP polynomial, not libm-identical -- floor-compared path).
pub fn dot_aos_multi_cutoff_sphe(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    if pts.nrows() <= SPHE_SOA_POINTS {
        debug_assert_eq!(w.len(), data.num_ao);
        debug_assert_eq!(out.len(), pts.nrows());
        out.fill(0.0);
        dot_aos_multi_cutoff_sphe_soa(data, pts, w, out);
    } else {
        dot_aos_multi_cutoff_sphe_generic(data, pts, w, out);
    }
}

/// Generic (any mesh size) form of [`dot_aos_multi_cutoff_sphe`].
fn dot_aos_multi_cutoff_sphe_generic(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(w.len(), data.num_ao);
    let nv = pts.nrows();
    debug_assert_eq!(out.len(), nv);
    for x in out.iter_mut() {
        *x = 0.0;
    }
    let nshell = data.sh_pzval.len();
    let cen = &data.sh_center;
    let pzval = &data.sh_pzval;
    let off = &data.sh_off;
    let neg_pz = &data.sh_neg_pz;
    let pnc = &data.sh_pnc;
    let ao_off = &data.sh_ao_off;
    let ao_ker = &data.sh_ao_kernel;
    let ao_home = &data.sh_ao_home;
    // Per-point scratch. `args`/`expv` are padded to a multiple of 8 so
    // `exp_pd_into`'s AVX-512 lane runs even for nv = 4/6/12 (pad args stay 0;
    // exp(0)=1 in dead pad lanes is never read).
    let nv8 = nv.div_ceil(8) * 8;
    let mut dx = vec![0.0_f64; nv];
    let mut dy = vec![0.0_f64; nv];
    let mut dz = vec![0.0_f64; nv];
    let mut r2 = vec![0.0_f64; nv];
    let mut alive = vec![false; nv];
    let mut rad = vec![0.0_f64; nv];
    let mut args = vec![0.0_f64; nv8];
    let mut expv = vec![0.0_f64; nv8];
    for s in 0..nshell {
        let (cx, cy, cz) = (cen[s * 3], cen[s * 3 + 1], cen[s * 3 + 2]);
        let pzv = pzval[s];
        let mut any = false;
        for v in 0..nv {
            let ddx = pts[[v, 0]] - cx;
            let ddy = pts[[v, 1]] - cy;
            let ddz = pts[[v, 2]] - cz;
            let rr = ddx * ddx + ddy * ddy + ddz * ddz;
            dx[v] = ddx;
            dy[v] = ddy;
            dz[v] = ddz;
            r2[v] = rr;
            let a = pzv * rr <= LEPSBAS;
            alive[v] = a;
            any |= a;
        }
        if !any {
            continue;
        }
        for r in rad.iter_mut() {
            *r = 0.0;
        }
        let (ps, pe) = (off[s] as usize, off[s + 1] as usize);
        let r2min = r2
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        for p in ps..pe {
            // Primitive value screen (Shell::prim_val), as the SoA form.
            let pzq = data.sh_pzprim[p];
            if pzq * r2min > LEPSBAS {
                continue;
            }
            let npz = neg_pz[p];
            for v in 0..nv {
                args[v] = npz * r2[v];
            }
            exp_pd_into(&args, &mut expv);
            let c = pnc[p];
            for v in 0..nv {
                let e = if pzq * r2[v] <= LEPSBAS { expv[v] } else { 0.0 };
                rad[v] += c * e;
            }
        }
        let (as_, ae) = (ao_off[s] as usize, ao_off[s + 1] as usize);
        // Shared power tables per (shell, point): the shell's 2l+1 angular
        // components read the SAME x^p, y^q, z^e, r2^k monomials -- computed once
        // here instead of per-AO `powi` chains (the mesh dot is angular-bound).
        // Max exponent = the shell's l (all AOs in a shell share l).
        let l_sh = data.angular_momentums[ao_ker[as_] as usize].max(0) as usize;
        if l_sh > 7 {
            // Exotic high-l shell: per-AO fallback (table size bound).
            for j in as_..ae {
                let ker = ao_ker[j] as usize;
                let wa = w[ao_home[j] as usize];
                if wa == 0.0 {
                    continue;
                }
                for v in 0..nv {
                    if alive[v] {
                        out[v] +=
                            rad[v] * eval_angular_flat(data, ker, dx[v], dy[v], dz[v], r2[v]) * wa;
                    }
                }
            }
            continue;
        }
        for v in 0..nv {
            if !alive[v] {
                continue;
            }
            let mut xp = [1.0_f64; 8];
            let mut yp = [1.0_f64; 8];
            let mut zp = [1.0_f64; 8];
            let mut r2p = [1.0_f64; 5];
            for k in 1..=l_sh.min(7) {
                xp[k] = xp[k - 1] * dx[v];
                yp[k] = yp[k - 1] * dy[v];
                zp[k] = zp[k - 1] * dz[v];
            }
            for k in 1..=(l_sh / 2).min(4) {
                r2p[k] = r2p[k - 1] * r2[v];
            }
            let rv = rad[v];
            let mut acc_v = 0.0_f64;
            for j in as_..ae {
                let ker = ao_ker[j] as usize;
                let wa = w[ao_home[j] as usize];
                if wa == 0.0 {
                    continue;
                }
                let mabs = data.ang_mabs[ker];
                let (ms, me) = (
                    data.ang_mc_off[ker] as usize,
                    data.ang_mc_off[ker + 1] as usize,
                );
                let mut m_val = 0.0_f64;
                for (p, t) in (ms..me).enumerate() {
                    m_val += data.ang_mc[t] * xp[p] * yp[(mabs as usize) - p];
                }
                let (ls, le) = (
                    data.ang_lam_off[ker] as usize,
                    data.ang_lam_off[ker + 1] as usize,
                );
                let mut lam = 0.0_f64;
                for t in ls..le {
                    lam += data.ang_lam_c[t]
                        * r2p[data.ang_lam_k[t] as usize]
                        * zp[data.ang_lam_ez[t] as usize];
                }
                acc_v += lam * m_val * wa;
            }
            out[v] += rv * acc_v;
        }
    }
}

/// Outward pad (relative and in bohr) of [`mesh_bounding_sphere`]'s radius: far
/// above the few-ulp rounding of the centroid and distances at bohr-scale
/// coordinates, so the sphere always contains every mesh point and the screen
/// stays conservative. A geometric safety margin, not a comparison tolerance.
const MESH_BOUND_PAD: f64 = 1.0e-12;

/// Bounding sphere `(centre, radius)` of a mesh-point set (centroid and the largest
/// distance to it). By the triangle inequality every point is at least
/// `|c - centre| - radius` from a shell centre `c`, so a shell whose value screen
/// fails at that distance is dead at every point and can be skipped before the
/// per-point distance loop, with exactly the same result.
#[inline]
fn mesh_bounding_sphere(px: &[f64], py: &[f64], pz: &[f64]) -> ([f64; 3], f64) {
    let n = px.len() as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for v in 0..px.len() {
        mx += px[v];
        my += py[v];
        mz += pz[v];
    }
    let (mx, my, mz) = (mx / n, my / n, mz / n);
    let mut r2 = 0.0_f64;
    for v in 0..px.len() {
        let d = (px[v] - mx).powi(2) + (py[v] - my).powi(2) + (pz[v] - mz).powi(2);
        r2 = r2.max(d);
    }
    // Small relative pad so rounding in the centroid can never drop a live shell.
    (
        [mx, my, mz],
        r2.sqrt() * (1.0 + MESH_BOUND_PAD) + MESH_BOUND_PAD,
    )
}

/// `true` when the value screen `pz_val * r^2 > LEPSBAS` holds for every point of the
/// mesh bounded by `(m, rad)` (see [`mesh_bounding_sphere`]).
#[inline]
fn shell_dead_on_mesh(c: [f64; 3], pz_val: f64, m: [f64; 3], rad: f64) -> bool {
    let d = ((c[0] - m[0]).powi(2) + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt() - rad;
    d > 0.0 && pz_val * d * d > LEPSBAS
}

const SPHE_SOA_POINTS: usize = 32;

/// Point-major (SoA) form of [`dot_aos_multi_cutoff_sphe`] for meshes of at most
/// [`SPHE_SOA_POINTS`] points and shells with l <= 7: every inner loop runs over the
/// mesh points with fixed-size stack scratch, no per-call allocation and no per-point
/// branches. Per point it adds the same terms in the same order as the generic form
/// (AO by AO within a shell, then `out[v] += rad[v] * acc[v]`), so results are bitwise
/// identical; points outside the shell's value screen add `0.0`. Shells with l > 7 use
/// the generic per-AO angular evaluation.
fn dot_aos_multi_cutoff_sphe_soa(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    const NP: usize = SPHE_SOA_POINTS;
    let nv = pts.nrows();
    let (mut px, mut py, mut pz) = ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    for v in 0..nv {
        px[v] = pts[[v, 0]];
        py[v] = pts[[v, 1]];
        pz[v] = pts[[v, 2]];
    }
    let mut xp = [[1.0_f64; NP]; 8];
    let mut yp = [[1.0_f64; NP]; 8];
    let mut zp = [[1.0_f64; NP]; 8];
    let mut r2p = [[1.0_f64; NP]; 5];
    let mut alive = [false; NP];
    let mut rad = [0.0_f64; NP];
    let mut acc = [0.0_f64; NP];
    let mut mv = [0.0_f64; NP];
    let mut lv = [0.0_f64; NP];
    let mut expv = [0.0_f64; NP];
    let out = &mut out[..nv];
    let nshell = data.sh_pzval.len();
    let (bm, brad) = mesh_bounding_sphere(&px[..nv], &py[..nv], &pz[..nv]);
    for s in 0..nshell {
        let (cx, cy, cz) = (
            data.sh_center[s * 3],
            data.sh_center[s * 3 + 1],
            data.sh_center[s * 3 + 2],
        );
        let pzv = data.sh_pzval[s];
        if shell_dead_on_mesh([cx, cy, cz], pzv, bm, brad) {
            continue;
        }
        let mut any = false;
        for v in 0..nv {
            let (dx, dy, dz) = (px[v] - cx, py[v] - cy, pz[v] - cz);
            let rr = dx * dx + dy * dy + dz * dz;
            xp[1][v] = dx;
            yp[1][v] = dy;
            zp[1][v] = dz;
            r2p[1][v] = rr;
            let a = pzv * rr <= LEPSBAS;
            alive[v] = a;
            any |= a;
        }
        if !any {
            continue;
        }
        let (ps, pe) = (data.sh_off[s] as usize, data.sh_off[s + 1] as usize);
        rad[..nv].fill(0.0);
        let mut r2min = f64::INFINITY;
        for &r2 in &r2p[1][..nv] {
            r2min = if r2 < r2min { r2 } else { r2min };
        }
        for p in ps..pe {
            // Primitive value screen (Shell::prim_val): dead at every point -> skip;
            // otherwise points outside it add exactly 0.
            let pzq = data.sh_pzprim[p];
            if pzq * r2min > LEPSBAS {
                continue;
            }
            let npz = data.sh_neg_pz[p];
            for v in 0..nv {
                let e = exp_neg_fast_scalar(npz * r2p[1][v]);
                expv[v] = if pzq * r2p[1][v] <= LEPSBAS { e } else { 0.0 };
            }
            let c = data.sh_pnc[p];
            for v in 0..nv {
                rad[v] += c * expv[v];
            }
        }
        let (as_, ae) = (data.sh_ao_off[s] as usize, data.sh_ao_off[s + 1] as usize);
        let l_sh = data.angular_momentums[data.sh_ao_kernel[as_] as usize].max(0) as usize;
        if l_sh > 7 {
            // Exotic high-l shell: per-AO generic angular evaluation (as the generic form).
            for j in as_..ae {
                let ker = data.sh_ao_kernel[j] as usize;
                let wa = w[data.sh_ao_home[j] as usize];
                if wa == 0.0 {
                    continue;
                }
                for v in 0..nv {
                    if alive[v] {
                        out[v] += rad[v]
                            * eval_angular_flat(data, ker, xp[1][v], yp[1][v], zp[1][v], r2p[1][v])
                            * wa;
                    }
                }
            }
            continue;
        }
        for k in 2..=l_sh {
            for v in 0..nv {
                xp[k][v] = xp[k - 1][v] * xp[1][v];
                yp[k][v] = yp[k - 1][v] * yp[1][v];
                zp[k][v] = zp[k - 1][v] * zp[1][v];
            }
        }
        for k in 2..=(l_sh / 2) {
            let (lo, hi) = r2p.split_at_mut(k);
            for ((out, &prev), &r2) in hi[0][..nv]
                .iter_mut()
                .zip(&lo[k - 1][..nv])
                .zip(&lo[1][..nv])
            {
                *out = prev * r2;
            }
        }
        acc[..nv].fill(0.0);
        for j in as_..ae {
            let ker = data.sh_ao_kernel[j] as usize;
            let wa = w[data.sh_ao_home[j] as usize];
            if wa == 0.0 {
                continue;
            }
            let mabs = data.ang_mabs[ker] as usize;
            mv[..nv].fill(0.0);
            let ms = data.ang_mc_off[ker] as usize;
            let me = data.ang_mc_off[ker + 1] as usize;
            for (p, t) in (ms..me).enumerate() {
                let c = data.ang_mc[t];
                let (a, b) = (&xp[p], &yp[mabs - p]);
                for v in 0..nv {
                    mv[v] += c * a[v] * b[v];
                }
            }
            lv[..nv].fill(0.0);
            for t in data.ang_lam_off[ker] as usize..data.ang_lam_off[ker + 1] as usize {
                let c = data.ang_lam_c[t];
                let (a, b) = (
                    &r2p[data.ang_lam_k[t] as usize],
                    &zp[data.ang_lam_ez[t] as usize],
                );
                for v in 0..nv {
                    lv[v] += c * a[v] * b[v];
                }
            }
            for v in 0..nv {
                acc[v] += lv[v] * mv[v] * wa;
            }
        }
        for v in 0..nv {
            out[v] += if alive[v] { rad[v] * acc[v] } else { 0.0 };
        }
    }
}

/// Cartesian sibling of [`dot_aos_multi_cutoff_sphe`] (same `pz_val` value
/// screen, [`Shell::pz_val`]).  The ECP determinant
/// ratio needs only `AO(mesh).w`, not the full AO and MO matrices.  Contract the
/// Cartesian components into `w` inside each shell, sharing every radial
/// exponential across the shell's angular components and all later projection
/// stages.
pub fn dot_aos_multi_cutoff_cart(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    debug_assert!(matches!(data.kind, AoKind::Cart));
    debug_assert_eq!(w.len(), data.num_ao);
    debug_assert_eq!(out.len(), pts.nrows());
    out.fill(0.0);

    const STACK_POINTS: usize = 32;
    const STACK_SHELL_AOS: usize = 16;
    let nv = pts.nrows();
    let nv4 = nv.div_ceil(4) * 4;
    let max_shell_aos = data.shells.iter().map(|s| s.aos.len()).max().unwrap_or(0);

    // ECP meshes and ordinary Cartesian shells are tiny (6/12 points and at
    // most the 10 components of an f shell in the production water inputs).
    // This contraction is called once per (electron, channel), so heap scratch
    // here meant O(10^5) allocator operations per MCMC run. Keep the common
    // shape entirely on the thread stack; retain a heap fallback for exotic
    // quadratures/high-l bases.
    if !data.cart_fam.is_empty() && max_shell_aos <= 16 && cart_fam_enabled() {
        dot_aos_multi_cutoff_cart_fam(data, pts, w, out);
    } else if nv <= FAST_POINTS
        && max_shell_aos <= STACK_SHELL_AOS
        && data
            .poly_x
            .iter()
            .chain(&data.poly_y)
            .chain(&data.poly_z)
            .all(|&n| n <= 7)
    {
        dot_aos_multi_cutoff_cart_soa(data, pts, w, out);
    } else if nv4 <= STACK_POINTS && max_shell_aos <= STACK_SHELL_AOS {
        let mut r2 = [0.0_f64; STACK_POINTS];
        let mut alive = [false; STACK_POINTS];
        let mut angular_w = [0.0_f64; STACK_POINTS * STACK_SHELL_AOS];
        let mut args = [0.0_f64; STACK_POINTS];
        let mut expv = [0.0_f64; STACK_POINTS];
        dot_aos_multi_cutoff_cart_impl(
            data,
            pts,
            w,
            out,
            &mut r2[..nv],
            &mut alive[..nv],
            &mut angular_w[..nv * STACK_SHELL_AOS],
            STACK_SHELL_AOS,
            &mut args[..nv4],
            &mut expv[..nv4],
        );
    } else {
        let mut r2 = vec![0.0_f64; nv];
        let mut alive = vec![false; nv];
        let mut angular_w = vec![0.0_f64; nv * max_shell_aos];
        let mut args = vec![0.0_f64; nv4];
        let mut expv = vec![0.0_f64; nv4];
        dot_aos_multi_cutoff_cart_impl(
            data,
            pts,
            w,
            out,
            &mut r2,
            &mut alive,
            &mut angular_w,
            max_shell_aos,
            &mut args,
            &mut expv,
        );
    }
}

const FAST_POINTS: usize = 32;

/// Structure-of-arrays form of [`dot_aos_multi_cutoff_cart_impl`]: every inner loop
/// runs over the mesh points without branches or per-point table copies. A shell
/// dead at all points costs only the distance loop; points dead in a live shell get
/// weight 0, which adds exactly nothing.
const CART_FAM_MIN_MEMBERS: usize = 16;

fn build_cart_families(d: &AosGtoData) -> Vec<CartFamily> {
    use std::collections::HashMap;
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut fams: Vec<CartFamily> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (si, sh) in d.shells.iter().enumerate() {
        let np = sh.prim_z.len();
        let mut key: Vec<u64> = vec![np as u64, sh.aos.len() as u64, sh.pz_val.to_bits()];
        key.extend(sh.prim_z.iter().map(|z| z.to_bits()));
        for &ao in &sh.aos {
            if d.poly_x[ao] < 0 || d.poly_y[ao] < 0 || d.poly_z[ao] < 0 {
                return Vec::new();
            }
            key.extend([
                d.poly_x[ao] as u64,
                d.poly_y[ao] as u64,
                d.poly_z[ao] as u64,
            ]);
            let off = d.ao_prim_offset[ao];
            key.extend(d.flat_nc[off..off + np].iter().map(|c| c.to_bits()));
        }
        let fi = *index.entry(key).or_insert_with(|| {
            let mut nc = Vec::with_capacity(np * sh.aos.len());
            let mut poly = Vec::with_capacity(sh.aos.len());
            for &ao in &sh.aos {
                let off = d.ao_prim_offset[ao];
                nc.extend_from_slice(&d.flat_nc[off..off + np]);
                poly.push([
                    d.poly_x[ao] as usize,
                    d.poly_y[ao] as usize,
                    d.poly_z[ao] as usize,
                ]);
            }
            fams.push(CartFamily {
                prim_z: sh.prim_z.clone(),
                nc,
                poly,
                pz_val: sh.pz_val,
                prim_val: sh.prim_val.clone(),
                ..Default::default()
            });
            members.push(Vec::new());
            fams.len() - 1
        });
        members[fi].push(si);
    }
    for (f, mem) in fams.iter_mut().zip(&members) {
        let nm = mem.len();
        let ncomp = f.poly.len();
        f.home = vec![0; ncomp * nm];
        for (m, &si) in mem.iter().enumerate() {
            let sh = &d.shells[si];
            f.cx.push(sh.center[0]);
            f.cy.push(sh.center[1]);
            f.cz.push(sh.center[2]);
            for (j, &ao) in sh.aos.iter().enumerate() {
                f.home[j * nm + m] = d.kernel_to_home[ao] as u32;
            }
        }
    }
    fams
}

/// Whether the Cartesian ECP contraction uses the family kernel. Default: on for
/// aarch64 (A64FX: Si wdot 0.93 -> 0.67 ms/call), off elsewhere (x86: 0.13 -> 0.16,
/// slower). `JQMC_RS_CART_FAM=1|0` overrides.
fn cart_fam_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("JQMC_RS_CART_FAM").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => cfg!(target_arch = "aarch64"),
    })
}

/// `d^n` for a loop-invariant small `n` (vectorizes over the caller's lanes).
#[inline(always)]
fn ipow(d: f64, n: usize) -> f64 {
    let mut p = 1.0;
    for _ in 0..n {
        p *= d;
    }
    p
}

/// Family form of [`dot_aos_multi_cutoff_cart_soa`]: for each mesh point, every
/// family is swept over its members in fixed-size chunks (lanes = members), so
/// the exp, polynomial and contraction loops are long and branch-free. Members
/// outside the value screen get weight 0 (adds exactly nothing). Same terms as
/// the shell form; only the summation order differs.
fn dot_aos_multi_cutoff_cart_fam(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    const CH: usize = 64;
    const NA: usize = 16;
    let nv = pts.nrows();
    let (mut dx, mut dy, mut dz) = ([0.0_f64; CH], [0.0_f64; CH], [0.0_f64; CH]);
    let mut r2 = [0.0_f64; CH];
    let mut ev = [0.0_f64; CH];
    let mut acc = [0.0_f64; CH];
    let mut rad = [[0.0_f64; CH]; NA];
    for v in 0..nv {
        let (x, y, z) = (pts[[v, 0]], pts[[v, 1]], pts[[v, 2]]);
        acc.fill(0.0);
        for f in &data.cart_fam {
            let nm = f.cx.len();
            let np = f.prim_z.len();
            let ncomp = f.poly.len();
            let pzv = f.pz_val;
            let mut m0 = 0;
            while m0 < nm {
                let n = (nm - m0).min(CH);
                let (cx, cy, cz) = (&f.cx[m0..m0 + n], &f.cy[m0..m0 + n], &f.cz[m0..m0 + n]);
                for i in 0..n {
                    dx[i] = x - cx[i];
                    dy[i] = y - cy[i];
                    dz[i] = z - cz[i];
                    r2[i] = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
                }
                for row in rad.iter_mut().take(ncomp) {
                    row[..n].fill(0.0);
                }
                for q in 0..np {
                    let zq = f.prim_z[q];
                    let pzq = f.prim_val[q];
                    for i in 0..n {
                        let e = crate::fastmath::exp_neg(-zq * r2[i]);
                        // Primitive value screen (Shell::prim_val): adds exactly 0.
                        ev[i] = if pzq * r2[i] <= LEPSBAS { e } else { 0.0 };
                    }
                    for (j, rad_j) in rad[..ncomp].iter_mut().enumerate() {
                        let c = f.nc[j * np + q];
                        let row = rad_j;
                        for i in 0..n {
                            row[i] += c * ev[i];
                        }
                    }
                }
                for (j, rad_j) in rad[..ncomp].iter().enumerate() {
                    let [nx, ny, nz] = f.poly[j];
                    let home = &f.home[j * nm + m0..j * nm + m0 + n];
                    let row = rad_j;
                    for i in 0..n {
                        let live = if pzv * r2[i] <= LEPSBAS { 1.0 } else { 0.0 };
                        let ang = ipow(dx[i], nx) * ipow(dy[i], ny) * ipow(dz[i], nz);
                        acc[i] += row[i] * ang * (w[home[i] as usize] * live);
                    }
                }
                m0 += n;
            }
        }
        out[v] = crate::fastmath::sum_lanes(&acc);
    }
}

fn dot_aos_multi_cutoff_cart_soa(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
) {
    const NP: usize = FAST_POINTS;
    const NA: usize = 16;
    let nv = pts.nrows();
    assert!(nv <= NP);
    let (mut px, mut py, mut pz) = ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    for v in 0..nv {
        px[v] = pts[[v, 0]];
        py[v] = pts[[v, 1]];
        pz[v] = pts[[v, 2]];
    }
    let mut xp = [[1.0_f64; NP]; 8];
    let mut yp = [[1.0_f64; NP]; 8];
    let mut zp = [[1.0_f64; NP]; 8];
    let mut r2 = [0.0_f64; NP];
    let mut mask = [0.0_f64; NP];
    let mut ang = [[0.0_f64; NP]; NA];
    let mut args = [0.0_f64; NP];
    let mut expv = [0.0_f64; NP];
    let mut contracted = [0.0_f64; NP];
    let out = &mut out[..nv];
    let (bm, brad) = mesh_bounding_sphere(&px[..nv], &py[..nv], &pz[..nv]);

    for shell in &data.shells {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        if shell_dead_on_mesh([cx, cy, cz], shell.pz_val, bm, brad) {
            continue;
        }
        let mut r2min = f64::INFINITY;
        for v in 0..nv {
            let (dx, dy, dz) = (px[v] - cx, py[v] - cy, pz[v] - cz);
            let rr = dx * dx + dy * dy + dz * dz;
            xp[1][v] = dx;
            yp[1][v] = dy;
            zp[1][v] = dz;
            r2[v] = rr;
            r2min = if rr < r2min { rr } else { r2min };
        }
        if shell.pz_val * r2min > LEPSBAS {
            continue;
        }
        for v in 0..nv {
            mask[v] = if shell.pz_val * r2[v] <= LEPSBAS {
                1.0
            } else {
                0.0
            };
        }
        let l = shell
            .aos
            .first()
            .map(|&ao| data.angular_momentums[ao].max(0) as usize)
            .unwrap_or(0)
            .min(7);
        for k in 2..=l {
            for v in 0..nv {
                xp[k][v] = xp[k - 1][v] * xp[1][v];
                yp[k][v] = yp[k - 1][v] * yp[1][v];
                zp[k][v] = zp[k - 1][v] * zp[1][v];
            }
        }
        for (j, &ao) in shell.aos.iter().enumerate() {
            let wj = w[data.kernel_to_home[ao]];
            let (ax, ay, az) = (
                &xp[data.poly_x[ao] as usize],
                &yp[data.poly_y[ao] as usize],
                &zp[data.poly_z[ao] as usize],
            );
            let row = &mut ang[j];
            for v in 0..nv {
                row[v] = ax[v] * ay[v] * az[v] * wj * mask[v];
            }
        }
        for q in 0..shell.prim_z.len() {
            // Primitive value screen (Shell::prim_val): dead at every point -> skip;
            // otherwise points outside it add exactly 0.
            let pzq = shell.prim_val[q];
            if pzq * r2min > LEPSBAS {
                continue;
            }
            let z = shell.prim_z[q];
            for v in 0..nv {
                args[v] = -z * r2[v];
            }
            exp_pd_into(&args[..nv], &mut expv[..nv]);
            for v in 0..nv {
                expv[v] = if pzq * r2[v] <= LEPSBAS { expv[v] } else { 0.0 };
            }
            contracted[..nv].fill(0.0);
            for (j, &ao) in shell.aos.iter().enumerate() {
                let c = data.flat_nc[data.ao_prim_offset[ao] + q];
                let row = &ang[j];
                for v in 0..nv {
                    contracted[v] += c * row[v];
                }
            }
            for v in 0..nv {
                out[v] += expv[v] * contracted[v];
            }
        }
    }
}

/// `true` when [`dot_aos_multi_cutoff_bloch`] covers this leaf: a non-empty shell
/// table and, for Cartesian leaves, shells of at most 16 components with monomial
/// orders `<= 7` (the stack tables of the point-major kernel). Spherical shells of
/// any `l` are covered (`l > 7` takes the per-AO angular evaluation).
pub(crate) fn bloch_mesh_dot_ok(data: &AosGtoData) -> bool {
    if data.shells.is_empty() || data.num_ao == 0 {
        return false;
    }
    match data.kind {
        AoKind::Sphe => !data.sh_pzval.is_empty(),
        AoKind::Cart => {
            data.shells.iter().all(|s| s.aos.len() <= 16)
                && data
                    .poly_x
                    .iter()
                    .chain(&data.poly_y)
                    .chain(&data.poly_z)
                    .all(|&n| (0..=7).contains(&n))
        }
    }
}

/// Bloch-weighted complex sibling of [`dot_aos_multi_cutoff_sphe`] /
/// [`dot_aos_multi_cutoff_cart`] for the complex twist leaf, whose AO is the
/// phase-weighted image fold `AO_mu(r) = Sum_{kappa -> mu} phase[kappa] *
/// kernel_kappa(r)` of the REAL Gaussian kernel (`complex_ao::fold_complex`).
/// Returns `out[v] = Sum_kappa (w[home(kappa)] * phase[kappa]) * kernel_kappa(pts[v])`
/// as split real/imaginary parts, without forming the AO or MO mesh. The complex
/// weight of a kernel AO is formed only inside a live shell. Same FAST value
/// screens as the real contractions: the per-(shell, point) `pz_val` screen and
/// the per-primitive `prim_val` screen, each dropping AO values below
/// `FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units ([`Shell::pz_val`],
/// [`Shell::prim_val`]); the fast `exp` of the real kernels. FAST route only
/// (floor-compared). Caller-gated by [`bloch_mesh_dot_ok`]. `pts` are the
/// (home-cell wrapped) evaluation points; any count (blocks of 32 points, each
/// point's terms independent of the other points of its block).
pub(crate) fn dot_aos_multi_cutoff_bloch(
    data: &AosGtoData,
    pts: &[[f64; 3]],
    w_re: &[f64],
    w_im: &[f64],
    phase: &[numpy::Complex64],
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    debug_assert!(bloch_mesh_dot_ok(data));
    debug_assert_eq!(w_re.len(), data.num_ao);
    debug_assert_eq!(w_im.len(), data.num_ao);
    debug_assert_eq!(phase.len(), data.num_ao_kernel);
    debug_assert_eq!(out_re.len(), pts.len());
    debug_assert_eq!(out_im.len(), pts.len());
    const NP: usize = FAST_POINTS;
    let mut v0 = 0;
    while v0 < pts.len() {
        let n = (pts.len() - v0).min(NP);
        let (blk, ore, oim) = (
            &pts[v0..v0 + n],
            &mut out_re[v0..v0 + n],
            &mut out_im[v0..v0 + n],
        );
        match data.kind {
            AoKind::Sphe => dot_bloch_sphe_soa(data, blk, w_re, w_im, phase, ore, oim),
            AoKind::Cart => dot_bloch_cart_soa(data, blk, w_re, w_im, phase, ore, oim),
        }
        v0 += n;
    }
}

/// Complex weight `w[home] * phase[ker]` of one kernel AO.
#[inline(always)]
fn bloch_weight(
    w_re: &[f64],
    w_im: &[f64],
    phase: &[numpy::Complex64],
    home: usize,
    ker: usize,
) -> (f64, f64) {
    let (a, b) = (w_re[home], w_im[home]);
    let p = phase[ker];
    (a * p.re - b * p.im, a * p.im + b * p.re)
}

/// Spherical point-major block of [`dot_aos_multi_cutoff_bloch`] (at most 32
/// points): the shell, primitive and angular arithmetic of
/// [`dot_aos_multi_cutoff_sphe_soa`] with a complex per-kernel-AO weight.
fn dot_bloch_sphe_soa(
    data: &AosGtoData,
    pts: &[[f64; 3]],
    w_re: &[f64],
    w_im: &[f64],
    phase: &[numpy::Complex64],
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    const NP: usize = FAST_POINTS;
    let nv = pts.len();
    debug_assert!(nv <= NP);
    let (mut px, mut py, mut pz) = ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    for v in 0..nv {
        px[v] = pts[v][0];
        py[v] = pts[v][1];
        pz[v] = pts[v][2];
    }
    let mut xp = [[1.0_f64; NP]; 8];
    let mut yp = [[1.0_f64; NP]; 8];
    let mut zp = [[1.0_f64; NP]; 8];
    let mut r2p = [[1.0_f64; NP]; 5];
    let mut alive = [false; NP];
    let mut rad = [0.0_f64; NP];
    let mut acc_re = [0.0_f64; NP];
    let mut acc_im = [0.0_f64; NP];
    let mut mv = [0.0_f64; NP];
    let mut lv = [0.0_f64; NP];
    let mut expv = [0.0_f64; NP];
    out_re.fill(0.0);
    out_im.fill(0.0);
    let nshell = data.sh_pzval.len();
    let (bm, brad) = mesh_bounding_sphere(&px[..nv], &py[..nv], &pz[..nv]);
    for s in 0..nshell {
        let (cx, cy, cz) = (
            data.sh_center[s * 3],
            data.sh_center[s * 3 + 1],
            data.sh_center[s * 3 + 2],
        );
        let pzv = data.sh_pzval[s];
        if shell_dead_on_mesh([cx, cy, cz], pzv, bm, brad) {
            continue;
        }
        let mut any = false;
        for v in 0..nv {
            let (dx, dy, dz) = (px[v] - cx, py[v] - cy, pz[v] - cz);
            let rr = dx * dx + dy * dy + dz * dz;
            xp[1][v] = dx;
            yp[1][v] = dy;
            zp[1][v] = dz;
            r2p[1][v] = rr;
            let a = pzv * rr <= LEPSBAS;
            alive[v] = a;
            any |= a;
        }
        if !any {
            continue;
        }
        let (ps, pe) = (data.sh_off[s] as usize, data.sh_off[s + 1] as usize);
        rad[..nv].fill(0.0);
        let mut r2min = f64::INFINITY;
        for &r2 in &r2p[1][..nv] {
            r2min = if r2 < r2min { r2 } else { r2min };
        }
        for p in ps..pe {
            let pzq = data.sh_pzprim[p];
            if pzq * r2min > LEPSBAS {
                continue;
            }
            let npz = data.sh_neg_pz[p];
            for v in 0..nv {
                let e = exp_neg_fast_scalar(npz * r2p[1][v]);
                expv[v] = if pzq * r2p[1][v] <= LEPSBAS { e } else { 0.0 };
            }
            let c = data.sh_pnc[p];
            for v in 0..nv {
                rad[v] += c * expv[v];
            }
        }
        let (as_, ae) = (data.sh_ao_off[s] as usize, data.sh_ao_off[s + 1] as usize);
        let l_sh = data.angular_momentums[data.sh_ao_kernel[as_] as usize].max(0) as usize;
        if l_sh > 7 {
            for j in as_..ae {
                let ker = data.sh_ao_kernel[j] as usize;
                let (wr, wi) = bloch_weight(w_re, w_im, phase, data.sh_ao_home[j] as usize, ker);
                if wr == 0.0 && wi == 0.0 {
                    continue;
                }
                for v in 0..nv {
                    if alive[v] {
                        let a = rad[v]
                            * eval_angular_flat(data, ker, xp[1][v], yp[1][v], zp[1][v], r2p[1][v]);
                        out_re[v] += a * wr;
                        out_im[v] += a * wi;
                    }
                }
            }
            continue;
        }
        for k in 2..=l_sh {
            for v in 0..nv {
                xp[k][v] = xp[k - 1][v] * xp[1][v];
                yp[k][v] = yp[k - 1][v] * yp[1][v];
                zp[k][v] = zp[k - 1][v] * zp[1][v];
            }
        }
        for k in 2..=(l_sh / 2) {
            let (lo, hi) = r2p.split_at_mut(k);
            for ((out, &prev), &r2) in hi[0][..nv]
                .iter_mut()
                .zip(&lo[k - 1][..nv])
                .zip(&lo[1][..nv])
            {
                *out = prev * r2;
            }
        }
        acc_re[..nv].fill(0.0);
        acc_im[..nv].fill(0.0);
        for j in as_..ae {
            let ker = data.sh_ao_kernel[j] as usize;
            let (wr, wi) = bloch_weight(w_re, w_im, phase, data.sh_ao_home[j] as usize, ker);
            if wr == 0.0 && wi == 0.0 {
                continue;
            }
            let mabs = data.ang_mabs[ker] as usize;
            mv[..nv].fill(0.0);
            let ms = data.ang_mc_off[ker] as usize;
            let me = data.ang_mc_off[ker + 1] as usize;
            for (p, t) in (ms..me).enumerate() {
                let c = data.ang_mc[t];
                let (a, b) = (&xp[p], &yp[mabs - p]);
                for v in 0..nv {
                    mv[v] += c * a[v] * b[v];
                }
            }
            lv[..nv].fill(0.0);
            for t in data.ang_lam_off[ker] as usize..data.ang_lam_off[ker + 1] as usize {
                let c = data.ang_lam_c[t];
                let (a, b) = (
                    &r2p[data.ang_lam_k[t] as usize],
                    &zp[data.ang_lam_ez[t] as usize],
                );
                for v in 0..nv {
                    lv[v] += c * a[v] * b[v];
                }
            }
            for v in 0..nv {
                let ang = lv[v] * mv[v];
                acc_re[v] += ang * wr;
                acc_im[v] += ang * wi;
            }
        }
        for v in 0..nv {
            if alive[v] {
                out_re[v] += rad[v] * acc_re[v];
                out_im[v] += rad[v] * acc_im[v];
            }
        }
    }
}

/// Cartesian point-major block of [`dot_aos_multi_cutoff_bloch`] (at most 32
/// points): the shell, primitive and monomial arithmetic of
/// [`dot_aos_multi_cutoff_cart_soa`] with a complex per-kernel-AO weight.
fn dot_bloch_cart_soa(
    data: &AosGtoData,
    pts: &[[f64; 3]],
    w_re: &[f64],
    w_im: &[f64],
    phase: &[numpy::Complex64],
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    const NP: usize = FAST_POINTS;
    const NA: usize = 16;
    let nv = pts.len();
    debug_assert!(nv <= NP);
    let (mut px, mut py, mut pz) = ([0.0_f64; NP], [0.0_f64; NP], [0.0_f64; NP]);
    for v in 0..nv {
        px[v] = pts[v][0];
        py[v] = pts[v][1];
        pz[v] = pts[v][2];
    }
    let mut xp = [[1.0_f64; NP]; 8];
    let mut yp = [[1.0_f64; NP]; 8];
    let mut zp = [[1.0_f64; NP]; 8];
    let mut r2 = [0.0_f64; NP];
    let mut mask = [0.0_f64; NP];
    let mut ang_re = [[0.0_f64; NP]; NA];
    let mut ang_im = [[0.0_f64; NP]; NA];
    let mut args = [0.0_f64; NP];
    let mut expv = [0.0_f64; NP];
    let mut con_re = [0.0_f64; NP];
    let mut con_im = [0.0_f64; NP];
    out_re.fill(0.0);
    out_im.fill(0.0);
    let (bm, brad) = mesh_bounding_sphere(&px[..nv], &py[..nv], &pz[..nv]);
    for shell in &data.shells {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        if shell_dead_on_mesh([cx, cy, cz], shell.pz_val, bm, brad) {
            continue;
        }
        let mut r2min = f64::INFINITY;
        for v in 0..nv {
            let (dx, dy, dz) = (px[v] - cx, py[v] - cy, pz[v] - cz);
            let rr = dx * dx + dy * dy + dz * dz;
            xp[1][v] = dx;
            yp[1][v] = dy;
            zp[1][v] = dz;
            r2[v] = rr;
            r2min = if rr < r2min { rr } else { r2min };
        }
        if shell.pz_val * r2min > LEPSBAS {
            continue;
        }
        for v in 0..nv {
            mask[v] = if shell.pz_val * r2[v] <= LEPSBAS {
                1.0
            } else {
                0.0
            };
        }
        let l = shell
            .aos
            .first()
            .map(|&ao| data.angular_momentums[ao].max(0) as usize)
            .unwrap_or(0)
            .min(7);
        for k in 2..=l {
            for v in 0..nv {
                xp[k][v] = xp[k - 1][v] * xp[1][v];
                yp[k][v] = yp[k - 1][v] * yp[1][v];
                zp[k][v] = zp[k - 1][v] * zp[1][v];
            }
        }
        for (j, &ao) in shell.aos.iter().enumerate() {
            let (wr, wi) = bloch_weight(w_re, w_im, phase, data.kernel_to_home[ao], ao);
            let (ax, ay, az) = (
                &xp[data.poly_x[ao] as usize],
                &yp[data.poly_y[ao] as usize],
                &zp[data.poly_z[ao] as usize],
            );
            for v in 0..nv {
                let m = ax[v] * ay[v] * az[v] * mask[v];
                ang_re[j][v] = m * wr;
                ang_im[j][v] = m * wi;
            }
        }
        for q in 0..shell.prim_z.len() {
            let pzq = shell.prim_val[q];
            if pzq * r2min > LEPSBAS {
                continue;
            }
            let z = shell.prim_z[q];
            for v in 0..nv {
                args[v] = -z * r2[v];
            }
            exp_pd_into(&args[..nv], &mut expv[..nv]);
            for v in 0..nv {
                expv[v] = if pzq * r2[v] <= LEPSBAS { expv[v] } else { 0.0 };
            }
            con_re[..nv].fill(0.0);
            con_im[..nv].fill(0.0);
            for (j, &ao) in shell.aos.iter().enumerate() {
                let c = data.flat_nc[data.ao_prim_offset[ao] + q];
                let (rr, ri) = (&ang_re[j], &ang_im[j]);
                for v in 0..nv {
                    con_re[v] += c * rr[v];
                    con_im[v] += c * ri[v];
                }
            }
            for v in 0..nv {
                out_re[v] += expv[v] * con_re[v];
                out_im[v] += expv[v] * con_im[v];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dot_aos_multi_cutoff_cart_impl(
    data: &AosGtoData,
    pts: &ArrayView2<f64>,
    w: &[f64],
    out: &mut [f64],
    r2: &mut [f64],
    alive: &mut [bool],
    angular_w: &mut [f64],
    angular_stride: usize,
    args: &mut [f64],
    expv: &mut [f64],
) {
    let nv = pts.nrows();

    for shell in &data.shells {
        let l = shell
            .aos
            .first()
            .map(|&ao| data.angular_momentums[ao].max(0) as usize)
            .unwrap_or(0);
        let mut any_alive = false;
        for v in 0..nv {
            let dx = pts[[v, 0]] - shell.center[0];
            let dy = pts[[v, 1]] - shell.center[1];
            let dz = pts[[v, 2]] - shell.center[2];
            let rr = dx * dx + dy * dy + dz * dz;
            r2[v] = rr;
            alive[v] = shell.pz_val * rr <= LEPSBAS;
            any_alive |= alive[v];
            if !alive[v] {
                continue;
            }

            // Angular factor times the electron-specific contraction weight is
            // independent of the primitive exponent.
            let (xp, yp, zp) = cart_power_tables(dx, dy, dz, l);
            for (j, &ao) in shell.aos.iter().enumerate() {
                let angular = cart_angular_from_tables(data, ao, dx, dy, dz, &xp, &yp, &zp);
                angular_w[v * angular_stride + j] = angular * w[data.kernel_to_home[ao]];
            }
        }
        if !any_alive {
            continue;
        }

        for q in 0..shell.prim_z.len() {
            let pzq = shell.prim_val[q];
            for v in 0..nv {
                args[v] = -shell.prim_z[q] * r2[v];
            }
            exp_pd_into(args, expv);
            for v in 0..nv {
                // Shell and primitive value screens (Shell::prim_val).
                if !alive[v] || pzq * r2[v] > LEPSBAS {
                    continue;
                }
                let mut contracted = 0.0_f64;
                for (j, &ao) in shell.aos.iter().enumerate() {
                    let p = data.ao_prim_offset[ao] + q;
                    contracted += data.flat_nc[p] * angular_w[v * angular_stride + j];
                }
                out[v] += expv[v] * contracted;
            }
        }
    }
}

/// Fused molecular MO projection with a per-shell distance cutoff: skips shells
/// with `pz_val*r^2 > LEPSBAS`, where every AO value of the shell is below
/// [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in absolute AO units ([`Shell::pz_val`], issue #100;
/// ~=74% of AOs for a 20-water cluster electron near one molecule), and for each
/// surviving AO accumulates `mo_out += ao_val * C_T[ao,:]` (sparse AXPY over the
/// contiguous transposed MO coefficients). This is the molecular analogue of the
/// periodic image cutoff -- native can prune AOs per-move; XLA's static shapes
/// must evaluate (and read C for) all of them. `c_t` is `(num_ao, num_mo)`
/// row-contiguous. Sphe molecular only.
pub fn compute_mo_single_cutoff_into(
    data: &AosGtoData,
    c_t: &[f64],
    num_mo: usize,
    pos: [f64; 3],
    mo_out: &mut [f64],
) {
    for x in mo_out.iter_mut() {
        *x = 0.0;
    }
    for shell in &data.shells {
        let dx = pos[0] - shell.center[0];
        let dy = pos[1] - shell.center[1];
        let dz = pos[2] - shell.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if shell.pz_val * r2 > LEPSBAS {
            continue; // every AO of the shell < FAST_SCREEN_ABS_BOUND = 1e-12 (absolute)
        }
        let mut rad = 0.0_f64;
        for i in 0..shell.prim_z.len() {
            rad += shell.prim_nc[i] * exp_neg_fast_scalar(-shell.prim_z[i] * r2);
        }
        for &ao in &shell.aos {
            let val = rad * data.sphe_angular[ao].eval(dx, dy, dz, r2);
            let row = &c_t[ao * num_mo..ao * num_mo + num_mo];
            for m in 0..num_mo {
                mo_out[m] += val * row[m];
            }
        }
    }
}

/// Cartesian analogue of [`compute_mo_single_cutoff_into`]. Cart AOs carry their
/// angular normalization per-AO (dxx != dxy), so -- unlike the sphe path -- the
/// radial can't be shared across a shell; we loop per-AO. Per shell: skip if
/// `pz_val*r^2 > LEPSBAS` (every AO value of the shell below `FAST_SCREEN_ABS_BOUND` in
/// absolute AO units, [`Shell::pz_val`]), else per AO accumulate
/// `mo_out += (Sum_p nc_p e^{-z_p r^2})*monomial * C_T[ao,:]`. Molecular only
/// (`num_ao_kernel == num_ao`). Bit-identical to the full eval up to the dropped
/// `< FAST_SCREEN_ABS_BOUND = 1e-12` (absolute) AO values.
pub fn compute_mo_single_cutoff_cart_into(
    data: &AosGtoData,
    c_t: &[f64],
    num_mo: usize,
    pos: [f64; 3],
    mo_out: &mut [f64],
) {
    for x in mo_out.iter_mut() {
        *x = 0.0;
    }
    // Shell-based: one distance + `pz_val` cutoff per shell (from the `shells`
    // table) rejects far shells cheaply -- no per-AO min-exp scan and one skip
    // decision per shell (not per AO). Radial stays per-AO (cart AOs carry a
    // per-AO angular normalization in `flat_nc`), read from the AO's own flat
    // primitive range so it is bit-identical to the full eval.
    for shell in &data.shells {
        let dx = pos[0] - shell.center[0];
        let dy = pos[1] - shell.center[1];
        let dz = pos[2] - shell.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if shell.pz_val * r2 > LEPSBAS {
            continue; // whole shell < FAST_SCREEN_ABS_BOUND = 1e-12 (absolute)
        }
        for &ao in &shell.aos {
            let (s, e) = (data.ao_prim_offset[ao], data.ao_prim_offset[ao + 1]);
            let mut rad = 0.0_f64;
            for p in s..e {
                rad += data.flat_nc[p] * exp_neg_fast_scalar(-data.flat_exp[p] * r2);
            }
            let (px, py, pz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
            let val = rad * dx.powi(px as i32) * dy.powi(py as i32) * dz.powi(pz as i32);
            let row = &c_t[ao * num_mo..ao * num_mo + num_mo];
            for m in 0..num_mo {
                mo_out[m] += val * row[m];
            }
        }
    }
}

/// No-alloc single-position AO eval into a caller-provided buffer (`out` length
/// `num_ao`). Molecular only (`n_image == 1`); the hot per-move path.
/// Flat per-AO solid-harmonic angular eval (reads the contiguous `ang_*` tables
/// instead of chasing the per-AO `SpheAngular` heap Vecs -- the cache-friendly
/// hot-loop form that kills the in-sweep AO-eval eviction penalty).
#[inline]
pub fn eval_angular_flat(data: &AosGtoData, ao: usize, x: f64, y: f64, z: f64, r2: f64) -> f64 {
    let mabs = data.ang_mabs[ao];
    let (ms, me) = (
        data.ang_mc_off[ao] as usize,
        data.ang_mc_off[ao + 1] as usize,
    );
    let mut m_val = 0.0_f64;
    for (t, p) in (ms..me).zip(0_i32..) {
        m_val += data.ang_mc[t] * x.powi(p) * y.powi(mabs - p);
    }
    let (ls, le) = (
        data.ang_lam_off[ao] as usize,
        data.ang_lam_off[ao + 1] as usize,
    );
    let mut lam = 0.0_f64;
    for t in ls..le {
        lam += data.ang_lam_c[t] * r2.powi(data.ang_lam_k[t]) * z.powi(data.ang_lam_ez[t]);
    }
    lam * m_val
}

#[inline]
fn cart_power_tables(x: f64, y: f64, z: f64, l: usize) -> ([f64; 8], [f64; 8], [f64; 8]) {
    let mut xp = [1.0_f64; 8];
    let mut yp = [1.0_f64; 8];
    let mut zp = [1.0_f64; 8];
    for k in 1..=l.min(7) {
        xp[k] = xp[k - 1] * x;
        yp[k] = yp[k - 1] * y;
        zp[k] = zp[k - 1] * z;
    }
    (xp, yp, zp)
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "scalar inputs of an inlined hot angular helper"
)]
fn cart_angular_from_tables(
    data: &AosGtoData,
    ao: usize,
    x: f64,
    y: f64,
    z: f64,
    xp: &[f64; 8],
    yp: &[f64; 8],
    zp: &[f64; 8],
) -> f64 {
    let (nx, ny, nz) = (
        data.poly_x[ao] as usize,
        data.poly_y[ao] as usize,
        data.poly_z[ao] as usize,
    );
    if nx <= 7 && ny <= 7 && nz <= 7 {
        xp[nx] * yp[ny] * zp[nz]
    } else {
        x.powi(nx as i32) * y.powi(ny as i32) * z.powi(nz as i32)
    }
}

pub fn compute_aos_single_into(data: &AosGtoData, pos: [f64; 3], out: &mut [f64]) {
    debug_assert_eq!(data.n_image, 1);
    debug_assert_eq!(out.len(), data.num_ao_kernel);
    if data.kind == AoKind::Sphe && !data.sh_pzmin.is_empty() {
        // Fully flat: shell centers/prims + per-AO angular all from contiguous
        // arrays (no scattered Vec<Shell>/Vec<SpheAngular> heap chasing).
        let nshell = data.sh_pzmin.len();
        let cen = &data.sh_center;
        let off = &data.sh_off;
        let neg_pz = &data.sh_neg_pz;
        let pnc = &data.sh_pnc;
        let ao_off = &data.sh_ao_off;
        let ao_ker = &data.sh_ao_kernel;
        for s in 0..nshell {
            let dx = pos[0] - cen[s * 3];
            let dy = pos[1] - cen[s * 3 + 1];
            let dz = pos[2] - cen[s * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let mut rad = 0.0_f64;
            for p in off[s] as usize..off[s + 1] as usize {
                rad += pnc[p] * exp_neg_fast_scalar(neg_pz[p] * r2);
            }
            for ao_ker_j in ao_ker[..ao_off[s + 1] as usize]
                .iter()
                .skip(ao_off[s] as usize)
            {
                let ao = *ao_ker_j as usize; // = home AO (molecular)
                out[ao] = rad * eval_angular_flat(data, ao, dx, dy, dz, r2);
            }
        }
        return;
    }
    if data.kind == AoKind::Cart && !data.shells.is_empty() {
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            for &ao in &shell.aos {
                out[ao] = 0.0;
            }
            for q in 0..shell.prim_z.len() {
                let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                for &ao in &shell.aos {
                    let p = data.ao_prim_offset[ao] + q;
                    out[ao] += data.flat_nc[p] * ev;
                }
            }
            for &ao in &shell.aos {
                out[ao] *= x.powi(data.poly_x[ao] as i32)
                    * y.powi(data.poly_y[ao] as i32)
                    * z.powi(data.poly_z[ao] as i32);
            }
        }
        return;
    }
    // cart / fallback flat path
    let centers = data.centers.as_slice().expect("centers contiguous");
    let flat_nc: &[f64] = &data.flat_nc;
    let flat_exp: &[f64] = &data.flat_exp;
    let off: &[usize] = &data.ao_prim_offset;
    for ao in 0..data.num_ao_kernel {
        let x = pos[0] - centers[ao * 3];
        let y = pos[1] - centers[ao * 3 + 1];
        let z = pos[2] - centers[ao * 3 + 2];
        let r2 = x * x + y * y + z * z;
        let mut radial = 0.0_f64;
        let (s, e) = (off[ao], off[ao + 1]);
        for (&c, &a) in flat_nc[s..e].iter().zip(&flat_exp[s..e]) {
            radial += c * exp_neg_fast_scalar(-a * r2);
        }
        let ang = match data.kind {
            AoKind::Sphe => data.sphe_angular[ao].eval(x, y, z, r2),
            AoKind::Cart => {
                x.powi(data.poly_x[ao] as i32)
                    * y.powi(data.poly_y[ao] as i32)
                    * z.powi(data.poly_z[ao] as i32)
            }
        };
        out[ao] = radial * ang;
    }
}

/// Production-fast Cartesian molecular single-position AO evaluation.  This is
/// deliberately separate from `compute_aos_single_into`: bitexact/parity callers
/// retain libm `exp`, while the statistically guarded fast MCMC chain opts into
/// the ~1e-11 relative-error scalar polynomial.
pub fn compute_aos_single_cart_fast_exp_into(data: &AosGtoData, pos: [f64; 3], out: &mut [f64]) {
    debug_assert_eq!(data.n_image, 1);
    debug_assert!(data.kind == AoKind::Cart);
    debug_assert_eq!(out.len(), data.num_ao_kernel);
    for shell in &data.shells {
        let x = pos[0] - shell.center[0];
        let y = pos[1] - shell.center[1];
        let z = pos[2] - shell.center[2];
        let r2 = x * x + y * y + z * z;
        for &ao in &shell.aos {
            out[ao] = 0.0;
        }
        for q in 0..shell.prim_z.len() {
            let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
            for &ao in &shell.aos {
                let p = data.ao_prim_offset[ao] + q;
                out[ao] += data.flat_nc[p] * ev;
            }
        }
        for &ao in &shell.aos {
            out[ao] *= x.powi(data.poly_x[ao] as i32)
                * y.powi(data.poly_y[ao] as i32)
                * z.powi(data.poly_z[ao] as i32);
        }
    }
}

/// Fold a kernel-rank single-electron AO vector to `num_ao` (no-op molecular).
#[inline]
fn fold_single(data: &AosGtoData, out: Vec<f64>) -> Vec<f64> {
    if data.n_image == 1 {
        out
    } else {
        // RAGGED fold: scatter each kernel-AO into its home AO (see fold_image_axis).
        let mut folded = vec![0.0_f64; data.num_ao];
        for k in 0..data.num_ao_kernel {
            folded[data.kernel_to_home[k]] += out[k];
        }
        folded
    }
}

fn fold_single_f32(data: &AosGtoData, out: Vec<f64>) -> Vec<f64> {
    if data.n_image == 1 {
        out
    } else {
        let mut terms = vec![Vec::<f32>::new(); data.num_ao];
        for k in 0..data.num_ao_kernel {
            terms[data.kernel_to_home[k]].push(out[k] as f32);
        }
        terms
            .into_iter()
            .map(|values| f64::from(reduce_sum_f32(values)))
            .collect()
    }
}

/// Collapse the kernel-rank image axis: `(n_image*num_ao, n_elec) -> (num_ao, n_elec)`,
/// summing over images. 1:1 with `_fold_image_axis` (image-major layout:
/// `reshape(n_image, num_ao, -1).sum(axis=0)`). No-op when `n_image == 1`.
fn fold_image_axis(data: &AosGtoData, kernel: Array2<f64>) -> Array2<f64> {
    if data.n_image == 1 {
        return kernel;
    }
    // Ragged fold: each kernel AO scatters into its home AO according to the
    // per-shell image layout. This subsumes a uniform
    // `img*num_ao` tiling (which is just identity-by-modulo when num_ao_kernel ==
    // n_image*num_ao).
    let n_elec = kernel.ncols();
    let mut out = Array2::<f64>::zeros((data.num_ao, n_elec));
    for k in 0..data.num_ao_kernel {
        let home = data.kernel_to_home[k];
        for e in 0..n_elec {
            out[[home, e]] += kernel[[k, e]];
        }
    }
    out
}

fn fold_image_axis_f32(data: &AosGtoData, kernel: Array2<f64>) -> Array2<f64> {
    if data.n_image == 1 {
        return kernel;
    }
    // Point-vectorized form of `reduce_sum_f32` per (home AO, electron): the
    // terms of a home AO are its kernel AOs in ascending order, reduced by the
    // same adjacent binary tree (pair i <- 2i + 2i+1, odd tail carried), one
    // contiguous row of electrons at a time -- the same additions in the same
    // order, hence the same bits.
    let n = kernel.ncols();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); data.num_ao];
    for k in 0..data.num_ao_kernel {
        groups[data.kernel_to_home[k]].push(k);
    }
    let mut out = Array2::<f64>::zeros((data.num_ao, n));
    let mut buf = Vec::<f32>::new();
    for (home, ks) in groups.iter().enumerate() {
        if ks.is_empty() {
            continue;
        }
        buf.resize(ks.len() * n, 0.0);
        for (j, &k) in ks.iter().enumerate() {
            let row = kernel.row(k);
            for e in 0..n {
                buf[j * n + e] = row[e] as f32;
            }
        }
        let mut len = ks.len();
        while len > 1 {
            let pairs = len / 2;
            for i in 0..pairs {
                for e in 0..n {
                    buf[i * n + e] = buf[2 * i * n + e] + buf[(2 * i + 1) * n + e];
                }
            }
            if len % 2 == 1 {
                buf.copy_within((len - 1) * n..len * n, pairs * n);
            }
            len = pairs + len % 2;
        }
        let mut orow = out.row_mut(home);
        for e in 0..n {
            orow[e] = f64::from(buf[e]);
        }
    }
    out
}

/// XLA lowers `reduce_sum` as an adjacent binary tree.  Reproduce that
/// reassociation for fp32 primitive contraction and periodic image folds; a
/// sequential scatter accumulation differs by several ulps on large slabs.
#[inline]
fn reduce_sum_f32(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    while values.len() > 1 {
        let old_len = values.len();
        let pairs = old_len / 2;
        for i in 0..pairs {
            values[i] = values[2 * i] + values[2 * i + 1];
        }
        if old_len % 2 == 1 {
            values[pairs] = values[old_len - 1];
        }
        values.truncate(pairs + old_len % 2);
    }
    values[0]
}

#[inline]
/// XLA's adjacent-tree `reduce_sum` (as [`reduce_sum_f32`]) in fp64 on a
/// caller-owned buffer; `values` is clobbered.
fn reduce_sum_f64_in_place(values: &mut [f64]) -> f64 {
    let mut len = values.len();
    if len == 0 {
        return 0.0;
    }
    while len > 1 {
        let pairs = len / 2;
        for i in 0..pairs {
            values[i] = values[2 * i] + values[2 * i + 1];
        }
        if len % 2 == 1 {
            values[pairs] = values[len - 1];
        }
        len = pairs + len % 2;
    }
    values[0]
}

/// Analytic AO gradient. Returns `(grad_x, grad_y, grad_z)`, each
/// `(num_ao_kernel, n_elec)`, mirroring `compute_AOs_grad`.
/// Solid harmonic `S_lm` value + grad `[dx,dy,dz]` + Hessian `[[d^2]]` (6 unique).
/// `S = Lambda(u=r^2,z)*M(x,y)`; dLambda via u/z building blocks, dM via x/y. For the sphe AO
/// Hessian (complex withJ gou term).
#[allow(clippy::type_complexity)]
fn s_l_m_value_grad_hess(
    l: i64,
    m: i64,
    x: f64,
    y: f64,
    z: f64,
    r_norm: f64,
) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let m_abs = m.abs();
    let trig = |q: i64| -> f64 { q as f64 * std::f64::consts::FRAC_PI_2 };
    // M(x,y) + 1st + 2nd partials
    let (mut mm, mut mx, mut my, mut mxx, mut mxy, mut myy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in 0..=m_abs {
        let b = binom(m_abs, p);
        let ang = if m >= 0 {
            trig(m_abs - p).cos()
        } else {
            trig(m_abs - p).sin()
        };
        let q = m_abs - p;
        let xp = x.powi(p as i32);
        let yq = y.powi(q as i32);
        mm += b * xp * yq * ang;
        if p >= 1 {
            mx += b * (p as f64) * x.powi((p - 1) as i32) * yq * ang;
        }
        if q >= 1 {
            my += b * xp * (q as f64) * y.powi((q - 1) as i32) * ang;
        }
        if p >= 2 {
            mxx += b * (p as f64) * ((p - 1) as f64) * x.powi((p - 2) as i32) * yq * ang;
        }
        if p >= 1 && q >= 1 {
            mxy +=
                b * (p as f64) * (q as f64) * x.powi((p - 1) as i32) * y.powi((q - 1) as i32) * ang;
        }
        if q >= 2 {
            myy += b * xp * (q as f64) * ((q - 1) as f64) * y.powi((q - 2) as i32) * ang;
        }
    }
    let lambda_lm = |k: i64| -> f64 {
        (-1.0_f64).powi(k as i32)
            * 2.0_f64.powi(-(l as i32))
            * binom(l, k)
            * binom(2 * l - 2 * k, l)
            * factorial(l - 2 * k)
            / factorial(l - 2 * k - m_abs)
    };
    let prefac = ((2 - if m_abs == 0 { 1 } else { 0 }) as f64 * factorial(l - m_abs)
        / factorial(l + m_abs))
    .sqrt();
    let kmax = (l - m_abs) / 2;
    let u = r_norm * r_norm;
    // Lambda building blocks (d wrt u and z)
    let (mut la, mut lu, mut luu, mut lz, mut luz, mut lzz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let upow = |k: i64| -> f64 {
        if k < 0 {
            0.0
        } else {
            u.powi(k as i32)
        }
    };
    let zpow = |e: i64| -> f64 {
        if e < 0 {
            0.0
        } else {
            z.powi(e as i32)
        }
    };
    for k in 0..=kmax {
        let lam = lambda_lm(k) * prefac;
        let e = l - 2 * k - m_abs;
        let (kf, ef) = (k as f64, e as f64);
        la += lam * upow(k) * zpow(e);
        if k >= 1 {
            lu += lam * kf * upow(k - 1) * zpow(e);
        }
        if k >= 2 {
            luu += lam * kf * (kf - 1.0) * upow(k - 2) * zpow(e);
        }
        if e >= 1 {
            lz += lam * upow(k) * ef * zpow(e - 1);
        }
        if k >= 1 && e >= 1 {
            luz += lam * kf * upow(k - 1) * ef * zpow(e - 1);
        }
        if e >= 2 {
            lzz += lam * upow(k) * ef * (ef - 1.0) * zpow(e - 2);
        }
    }
    // gradLambda
    let dl = [2.0 * x * lu, 2.0 * y * lu, lz + 2.0 * z * lu];
    // Hessian of Lambda (full)
    let mut hl = [[0.0; 3]; 3];
    hl[0][0] = 2.0 * lu + 4.0 * x * x * luu;
    hl[1][1] = 2.0 * lu + 4.0 * y * y * luu;
    hl[2][2] = lzz + 4.0 * z * luz + 2.0 * lu + 4.0 * z * z * luu;
    hl[0][1] = 4.0 * x * y * luu;
    hl[1][0] = hl[0][1];
    hl[0][2] = 2.0 * x * luz + 4.0 * x * z * luu;
    hl[2][0] = hl[0][2];
    hl[1][2] = 2.0 * y * luz + 4.0 * y * z * luu;
    hl[2][1] = hl[1][2];
    let dm = [mx, my, 0.0];
    let mut hm = [[0.0; 3]; 3];
    hm[0][0] = mxx;
    hm[1][1] = myy;
    hm[0][1] = mxy;
    hm[1][0] = mxy; // others 0 (M -| z)
    let s = la * mm;
    let grad = [
        dl[0] * mm + la * dm[0],
        dl[1] * mm + la * dm[1],
        dl[2] * mm + la * dm[2],
    ];
    let mut hess = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            hess[a][b] = hl[a][b] * mm + dl[a] * dm[b] + dl[b] * dm[a] + la * hm[a][b];
        }
    }
    (s, grad, hess)
}

/// Pre-fold per-kernel-AO `gradgradAO` `[[dadb]]` (6 unique -> 3x3) each `(num_ao_kernel,
/// N)`, SPHE. `grad_agrad_b AO = 4 r_a r_b s2 Y - 2delta_ab s1 Y - 2 s1(r_a d_b Y + r_b d_a Y)
/// + f d_ad_b Y`, d_ad_b Y = nlm*S_ab (`s_l_m_value_grad_hess`). For complex withJ gou.
/// Cart monomial `M = x^px y^py z^pz` value.
#[inline]
fn cmono(x: f64, y: f64, z: f64, a: i64, b: i64, c: i64) -> f64 {
    let p = |v: f64, e: i64| -> f64 {
        if e < 0 {
            0.0
        } else {
            v.powi(e as i32)
        }
    };
    p(x, a) * p(y, b) * p(z, c)
}

/// Cart monomial Laplacian `grad^2M`.
#[inline]
fn cmono_lap(x: f64, y: f64, z: f64, px: i64, py: i64, pz: i64) -> f64 {
    (px * (px - 1)) as f64 * cmono(x, y, z, px - 2, py, pz)
        + (py * (py - 1)) as f64 * cmono(x, y, z, px, py - 2, pz)
        + (pz * (pz - 1)) as f64 * cmono(x, y, z, px, py, pz - 2)
}

/// Cart monomial value, gradient, Hessian `(M, dM, d^2M)` -- the angular factor for
/// a cart AO (`AO = f(r^2)*M`), replacing the sphe harmonic `Y` in the field-strain.
fn cart_mono_vgh(
    px: i64,
    py: i64,
    pz: i64,
    x: f64,
    y: f64,
    z: f64,
) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let (fx, fy, fz) = (px as f64, py as f64, pz as f64);
    let val = cmono(x, y, z, px, py, pz);
    let grad = [
        fx * cmono(x, y, z, px - 1, py, pz),
        fy * cmono(x, y, z, px, py - 1, pz),
        fz * cmono(x, y, z, px, py, pz - 1),
    ];
    let mut hess = [[0.0_f64; 3]; 3];
    hess[0][0] = fx * (fx - 1.0) * cmono(x, y, z, px - 2, py, pz);
    hess[1][1] = fy * (fy - 1.0) * cmono(x, y, z, px, py - 2, pz);
    hess[2][2] = fz * (fz - 1.0) * cmono(x, y, z, px, py, pz - 2);
    hess[0][1] = fx * fy * cmono(x, y, z, px - 1, py - 1, pz);
    hess[0][2] = fx * fz * cmono(x, y, z, px - 1, py, pz - 1);
    hess[1][2] = fy * fz * cmono(x, y, z, px, py - 1, pz - 1);
    hess[1][0] = hess[0][1];
    hess[2][0] = hess[0][2];
    hess[2][1] = hess[1][2];
    (val, grad, hess)
}

pub(crate) fn compute_aos_hessian_kernel_sphe(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> [[Array2<f64>; 3]; 3] {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let m = data.num_ao_kernel;
    let mk = || Array2::<f64>::zeros((m, n_elec));
    let mut h = [[mk(), mk(), mk()], [mk(), mk(), mk()], [mk(), mk(), mk()]];
    let periodic = screen && data.n_image > 1;
    let is_cart = data.kind == AoKind::Cart;
    for ao in 0..m {
        let l = data.angular_momentums[ao];
        let nlm = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
        let min_exp = if periodic {
            groups[ao]
                .iter()
                .map(|&p| data.exponents[p])
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        for e in 0..n_elec {
            let rc = [
                r_carts[[e, 0]] - data.centers[[ao, 0]],
                r_carts[[e, 1]] - data.centers[[ao, 1]],
                r_carts[[e, 2]] - data.centers[[ao, 2]],
            ];
            let r2 = rc[0] * rc[0] + rc[1] * rc[1] + rc[2] * rc[2];
            if periodic && min_exp * r2 > LEPSBAS {
                continue;
            }
            let (mut f, mut s1, mut s2) = (0.0, 0.0, 0.0);
            for &p in &groups[ao] {
                let zexp = data.exponents[p];
                let a = norms[p] * data.coefficients[p] * exp_neg_fast_scalar(-zexp * r2);
                f += a;
                s1 += zexp * a;
                s2 += zexp * zexp * a;
            }
            // angular value/grad/hess: cart monomial `M` OR sphe harmonic `nlm*Y`.
            let (yval, dy, yab_m) = if is_cart {
                cart_mono_vgh(
                    data.poly_x[ao],
                    data.poly_y[ao],
                    data.poly_z[ao],
                    rc[0],
                    rc[1],
                    rc[2],
                )
            } else {
                let (sval, sgrad, shess) =
                    s_l_m_value_grad_hess(l, data.magnetic[ao], rc[0], rc[1], rc[2], r2.sqrt());
                (
                    nlm * sval,
                    [nlm * sgrad[0], nlm * sgrad[1], nlm * sgrad[2]],
                    [
                        [nlm * shess[0][0], nlm * shess[0][1], nlm * shess[0][2]],
                        [nlm * shess[1][0], nlm * shess[1][1], nlm * shess[1][2]],
                        [nlm * shess[2][0], nlm * shess[2][1], nlm * shess[2][2]],
                    ],
                )
            };
            for a in 0..3 {
                for b in 0..3 {
                    let dab = if a == b { 1.0 } else { 0.0 };
                    h[a][b][[ao, e]] = 4.0 * rc[a] * rc[b] * s2 * yval
                        - 2.0 * dab * s1 * yval
                        - 2.0 * s1 * (rc[a] * dy[b] + rc[b] * dy[a])
                        + f * yab_m[a][b];
                }
            }
        }
    }
    h
}

/// Pre-fold per-kernel-AO `grad(grad^2AO)` `[dx,dy,dz]` each `(num_ao_kernel, N)`, SPHE
/// only. Closed form: a sphe AO = f(r^2)*Y (Y harmonic, degree l) => grad^2AO = g(r^2)*Y,
/// g = 4r^2f''+(6+4l)f' => `grad_c(grad^2AO) = 2 r_c (dg/dr^2) Y + g d_c Y`. Radial sums
/// a_p=norm*coef: f=Suma e^{-Zr^2}, s1=SumZa, s2=SumZ^2a, s3=SumZ^3a; g=4r^2s2-(6+4l)s1;
/// dg/dr^2=(10+4l)s2-4r^2s3. (cart not supported here -- complex leaves are sphe.)
pub(crate) fn compute_aos_gradlap_kernel_sphe(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> [Array2<f64>; 3] {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let m = data.num_ao_kernel;
    let mut gl = [
        Array2::<f64>::zeros((m, n_elec)),
        Array2::<f64>::zeros((m, n_elec)),
        Array2::<f64>::zeros((m, n_elec)),
    ];
    let periodic = screen && data.n_image > 1;
    let is_cart = data.kind == AoKind::Cart;
    for ao in 0..m {
        let l = data.angular_momentums[ao];
        let nlm = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
        let min_exp = if periodic {
            groups[ao]
                .iter()
                .map(|&p| data.exponents[p])
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            let r2 = x * x + y * y + z * z;
            if periodic && min_exp * r2 > LEPSBAS {
                continue;
            }
            let (mut s0, mut s1, mut s2, mut s3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for &p in &groups[ao] {
                let z_exp = data.exponents[p];
                let a = norms[p] * data.coefficients[p] * exp_neg_fast_scalar(-z_exp * r2);
                s0 += a;
                s1 += z_exp * a;
                s2 += z_exp * z_exp * a;
                s3 += z_exp * z_exp * z_exp * a;
            }
            let rc = [x, y, z];
            if is_cart {
                // AO = f*M (M cart monomial, NOT harmonic -> grad^2M != 0).
                // grad^2AO = g_c*M + f*grad^2M,  g_c = 4r^2s2-(6+4n)s1,  n = px+py+pz.
                // grad_d(grad^2AO) = 2r_d*dg_c*M + g_c*d_dM - 2r_d*s1*grad^2M + f*grad^2(d_dM).
                let (px, py, pz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                let n = (px + py + pz) as f64;
                let mval = cmono(x, y, z, px, py, pz);
                let gm = [
                    px as f64 * cmono(x, y, z, px - 1, py, pz),
                    py as f64 * cmono(x, y, z, px, py - 1, pz),
                    pz as f64 * cmono(x, y, z, px, py, pz - 1),
                ];
                let lap_m = cmono_lap(x, y, z, px, py, pz);
                // grad^2(d_d M) = p_d * grad^2(monomial with p_d reduced by 1)
                let lap_dm = [
                    px as f64 * cmono_lap(x, y, z, px - 1, py, pz),
                    py as f64 * cmono_lap(x, y, z, px, py - 1, pz),
                    pz as f64 * cmono_lap(x, y, z, px, py, pz - 1),
                ];
                let g_c = 4.0 * r2 * s2 - (6.0 + 4.0 * n) * s1;
                let dg_c = (10.0 + 4.0 * n) * s2 - 4.0 * r2 * s3;
                for c in 0..3 {
                    gl[c][[ao, e]] = 2.0 * rc[c] * dg_c * mval + g_c * gm[c]
                        - 2.0 * rc[c] * s1 * lap_m
                        + s0 * lap_dm[c];
                }
            } else {
                let (s, sgx, sgy, sgz) =
                    s_l_m_value_and_grad(l, data.magnetic[ao], x, y, z, r2.sqrt());
                let yval = nlm * s;
                let dy = [nlm * sgx, nlm * sgy, nlm * sgz];
                let lf = l as f64;
                let g = 4.0 * r2 * s2 - (6.0 + 4.0 * lf) * s1;
                let dg = (10.0 + 4.0 * lf) * s2 - 4.0 * r2 * s3;
                for c in 0..3 {
                    gl[c][[ao, e]] = 2.0 * rc[c] * dg * yval + g * dy[c];
                }
            }
        }
    }
    gl
}

/// Pre-fold per-kernel-AO value + grad + Laplacian `(num_ao_kernel, N)` each
/// (`val, gx, gy, gz, lap`). The body of the grad/lap path without the image
/// fold -- for the complex Bloch path (phase-weighted fold). sphe + cart.
#[allow(clippy::type_complexity)]
pub(crate) fn compute_aos_vgl_kernel(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
    screen: bool,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let m = data.num_ao_kernel;
    let mut val = Array2::<f64>::zeros((m, n_elec));
    let mut gx = Array2::<f64>::zeros((m, n_elec));
    let mut gy = Array2::<f64>::zeros((m, n_elec));
    let mut gz = Array2::<f64>::zeros((m, n_elec));
    let mut lap = Array2::<f64>::zeros((m, n_elec));
    let periodic = screen && data.n_image > 1;
    for ao in 0..m {
        let min_exp = if periodic {
            groups[ao]
                .iter()
                .map(|&p| data.exponents[p])
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            if periodic && min_exp * (x * x + y * y + z * z) > LEPSBAS {
                continue;
            }
            let vgl = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z);
            val[[ao, e]] = vgl.value;
            gx[[ao, e]] = vgl.grad[0];
            gy[[ao, e]] = vgl.grad[1];
            gz[[ao, e]] = vgl.grad[2];
            lap[[ao, e]] = vgl.laplacian;
        }
    }
    (val, gx, gy, gz, lap)
}

pub fn compute_aos_grad(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let mut gx = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    let mut gy = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    let mut gz = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    for ao in 0..data.num_ao_kernel {
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            let vgl = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z);
            gx[[ao, e]] = vgl.grad[0];
            gy[[ao, e]] = vgl.grad[1];
            gz[[ao, e]] = vgl.grad[2];
        }
    }
    (
        fold_image_axis(data, gx),
        fold_image_axis(data, gy),
        fold_image_axis(data, gz),
    )
}

/// EXACT (un-screened) sibling of [`compute_aos_grad`]: no runtime `LEPSBAS`
/// distance skip on periodic leaves -- every kernel AO of the static image tiling
/// is differentiated at every point, matching jqmc's `compute_AOs_grad` (the XLA
/// convention). Observable / density-response paths (e.g. the MPC nuclear force at
/// the density grid nodes) MUST use this: a screened grad lands an EXACT 0.0 where
/// XLA has a small non-zero, matching the value-kernel issue described above.
pub fn compute_aos_grad_exact(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let mut gx = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    let mut gy = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    let mut gz = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    for ao in 0..data.num_ao_kernel {
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            let vgl = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z);
            gx[[ao, e]] = vgl.grad[0];
            gy[[ao, e]] = vgl.grad[1];
            gz[[ao, e]] = vgl.grad[2];
        }
    }
    (
        fold_image_axis(data, gx),
        fold_image_axis(data, gy),
        fold_image_axis(data, gz),
    )
}

/// Fused AO gradient + laplacian in ONE pass. Returns `(gx, gy, gz, lap)`, each
/// `(num_ao, n_elec)`. Replaces separate `compute_aos_grad` + `compute_aos_laplacian`
/// (which each swept the fused per-AO `ao_value_grad_lap` and discarded 3 of its 4
/// outputs) -- halving the AO grad/lap radial-exp work in the kinetic hot path.
/// Numerically identical to calling the two separately (same `ao_value_grad_lap`).
pub fn compute_aos_grad_lap(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let m = data.num_ao_kernel;
    let mut gx = Array2::<f64>::zeros((m, n_elec));
    let mut gy = Array2::<f64>::zeros((m, n_elec));
    let mut gz = Array2::<f64>::zeros((m, n_elec));
    let mut lap = Array2::<f64>::zeros((m, n_elec));
    // EXACT observable/parity path: NO runtime LEPSBAS distance screen. jqmc truncates
    // the periodic image set once at BUILD time and then evaluates every surviving
    // kernel AO exactly for every electron; a runtime per-(AO, electron) skip drops
    // terms XLA keeps. The opt-in approximation
    // lives in `compute_aos_grad_lap_fast_cut` (production FAST chunk only).
    // Sphe shell-reuse (the kinetic-grad^2 sibling of the value shell-reuse in
    // `compute_aos_kernel`): the radial moments `nr=Sum a`, `znr=Sum Z a`,
    // `z2nr=Sum Z^2 a` (a = norm*coef*exp(-Z r^2)) are shared by all AOs in the
    // shell, so they are summed ONCE per (shell, electron) and reused across the
    // shell's angular components -- vs the per-AO `ao_value_grad_lap` which
    // re-summed the same three radial exps for every AO in the shell. Sphe
    // harmonics are harmonic (grad^2 P = 0), so the per-AO combination below is
    // the EXACT `ao_value_grad_lap` sphe branch (same nr/znr/z2nr sequential sum,
    // same `s_l_m_value_and_grad` angular via `sphe_angular_value_grad`, same
    // `pz_min == min_exp` cutoff) -> bit-identical.
    if data.kind == AoKind::Sphe && !data.shells.is_empty() {
        for shell in &data.shells {
            let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
            let pz = &shell.prim_z;
            let pnc = &shell.prim_nc;
            let pz_min = shell.pz_min;
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - cx;
                let y = r_carts[[e, 1]] - cy;
                let z = r_carts[[e, 2]] - cz;
                let r2 = x * x + y * y + z * z;
                let _ = pz_min; // no runtime screen on the exact path (see above)
                let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
                for i in 0..pz.len() {
                    let a = pnc[i] * exp_neg_fast_scalar(-pz[i] * r2);
                    nr += a;
                    znr += pz[i] * a;
                    z2nr += pz[i] * pz[i] * a;
                }
                let radial_lap_c = 4.0 * r2 * z2nr - 6.0 * znr;
                for &ao in &shell.aos {
                    let (pp, dpx, dpy, dpz) = sphe_angular_value_grad(data, ao, x, y, z, r2);
                    gx[[ao, e]] = nr * dpx - 2.0 * x * znr * pp;
                    gy[[ao, e]] = nr * dpy - 2.0 * y * znr * pp;
                    gz[[ao, e]] = nr * dpz - 2.0 * z * znr * pp;
                    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                    lap[[ao, e]] = -4.0 * znr * r_dot_grad_p + radial_lap_c * pp;
                }
            }
        }
        return (
            fold_image_axis(data, gx),
            fold_image_axis(data, gy),
            fold_image_axis(data, gz),
            fold_image_axis(data, lap),
        );
    }
    // Cart shell reuse: the AOs of a shell share the center and the primitive exponents
    // (the shell key), so `exp(-Z r^2)` is evaluated once per (shell, primitive, electron)
    // instead of once per AO. Each AO keeps its own `norm*coef` and its own sequential
    // radial sums, and the angular part is the `ao_value_grad_lap` cart branch (powers via
    // the bit-identical inlined `powi`), so every output is bit-identical to the per-AO loop
    // below.
    if data.kind == AoKind::Cart && !data.shells.is_empty() {
        let powm = |v: f64, n: i64| -> f64 {
            if n < 0 {
                0.0
            } else {
                crate::ao_cart_deriv::powi_inline(v, n as u32)
            }
        };
        let mut ez: Vec<f64> = Vec::new();
        for shell in &data.shells {
            let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
            let pz = &shell.prim_z;
            ez.resize(pz.len(), 0.0);
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - cx;
                let y = r_carts[[e, 1]] - cy;
                let z = r_carts[[e, 2]] - cz;
                let r2 = x * x + y * y + z * z;
                for (q, zq) in pz.iter().enumerate() {
                    ez[q] = (-zq * r2).exp();
                }
                for &ao in &shell.aos {
                    let off = data.ao_prim_offset[ao];
                    let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for (q, &z_exp) in pz.iter().enumerate() {
                        let pref = data.flat_nc[off + q] * ez[q];
                        nr += pref;
                        znr += z_exp * pref;
                        z2nr += z_exp * z_exp * pref;
                    }
                    let (nx, ny, nz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                    let xx = powm(x, nx);
                    let yy = powm(y, ny);
                    let zz = powm(z, nz);
                    let pp = xx * yy * zz;
                    let dpx = (nx as f64) * powm(x, nx - 1) * yy * zz;
                    let dpy = xx * (ny as f64) * powm(y, ny - 1) * zz;
                    let dpz = xx * yy * (nz as f64) * powm(z, nz - 1);
                    let lap_p = (nx as f64) * ((nx - 1) as f64) * powm(x, nx - 2) * yy * zz
                        + xx * (ny as f64) * ((ny - 1) as f64) * powm(y, ny - 2) * zz
                        + xx * yy * (nz as f64) * ((nz - 1) as f64) * powm(z, nz - 2);
                    gx[[ao, e]] = nr * dpx - 2.0 * x * znr * pp;
                    gy[[ao, e]] = nr * dpy - 2.0 * y * znr * pp;
                    gz[[ao, e]] = nr * dpz - 2.0 * z * znr * pp;
                    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                    lap[[ao, e]] =
                        nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
                }
            }
        }
        return (
            fold_image_axis(data, gx),
            fold_image_axis(data, gy),
            fold_image_axis(data, gz),
            fold_image_axis(data, lap),
        );
    }
    for ao in 0..m {
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            let vgl = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z);
            gx[[ao, e]] = vgl.grad[0];
            gy[[ao, e]] = vgl.grad[1];
            gz[[ao, e]] = vgl.grad[2];
            lap[[ao, e]] = vgl.laplacian;
        }
    }
    (
        fold_image_axis(data, gx),
        fold_image_axis(data, gy),
        fold_image_axis(data, gz),
        fold_image_axis(data, lap),
    )
}

/// FAST-mode molecular batch AO grad+lap: [`compute_aos_grad_lap`]'s
/// shell-reuse sweep with a distance cutoff ALSO applied on molecular leaves
/// (`pz_deriv*r2 > LEPSBAS` -> the shell's grad/lap rows stay 0 for that
/// electron). `pz_deriv` ([`Shell::pz_deriv`], [`derivative_cut_exponent`])
/// bounds every dropped gradient component and Laplacian of every AO of the
/// shell by [`FAST_SCREEN_ABS_BOUND`] `= 1e-12` in ABSOLUTE AO units. (A plain
/// `pz_min*r2 > LEPSBAS` cut is only relative to the shell's peak factor
/// `|norm*coef|`; for uncontracted tight primitives the dropped Laplacian grows
/// as ~zeta^1.75 and biases the FAST e_L upward, issue #87.) The exact batch
/// never cuts molecular (bit-parity with JAX); this variant serves the
/// production FAST chunk kinetic and J3 grad/lap only (floor-compared), where it
/// skips the far (shell, electron) pairs of a spatially extended cluster. Falls
/// back to the exact path for un-finalized leaves.
pub fn compute_aos_grad_lap_fast_cut(
    data: &AosGtoData,
    r_carts: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    if data.shells.is_empty() {
        return compute_aos_grad_lap(data, r_carts);
    }
    let n_elec = r_carts.nrows();
    let m = data.num_ao_kernel;
    let mut gx = Array2::<f64>::zeros((m, n_elec));
    let mut gy = Array2::<f64>::zeros((m, n_elec));
    let mut gz = Array2::<f64>::zeros((m, n_elec));
    let mut lap = Array2::<f64>::zeros((m, n_elec));

    // Cartesian shell counterpart: primitive exponents and geometry are shared
    // by all angular components, while normalized coefficients are AO-specific.
    // Evaluate each exp once and fan it into per-component radial moments. The
    // electron axis is contiguous scratch, so exp(-zeta*r^2) runs through the
    // AVX2/AVX-512 vector kernel instead of scalar libm for every electron.
    if data.kind == AoKind::Cart {
        let powm = |v: f64, tab: &[f64; 8], n: i64| -> f64 {
            if n < 0 {
                0.0
            } else if n < 8 {
                tab[n as usize]
            } else {
                v.powi(n as i32)
            }
        };
        let ne4 = n_elec.div_ceil(4) * 4;
        let mut xv = vec![0.0_f64; n_elec];
        let mut yv = vec![0.0_f64; n_elec];
        let mut zv = vec![0.0_f64; n_elec];
        let mut r2v = vec![0.0_f64; n_elec];
        let mut alive = vec![false; n_elec];
        let mut args = vec![0.0_f64; ne4];
        let mut expv = vec![0.0_f64; ne4];
        let mut nr = vec![0.0_f64; n_elec];
        let mut znr = vec![0.0_f64; n_elec];
        let mut z2nr = vec![0.0_f64; n_elec];
        let mut radial_scale = vec![0.0_f64; data.num_ao_kernel];
        for shell in &data.shells {
            if let Some(q) = shell.prim_nc.iter().position(|&x| x != 0.0) {
                let denom = shell.prim_nc[q];
                for &ao in &shell.aos {
                    radial_scale[ao] = data.flat_nc[data.ao_prim_offset[ao] + q] / denom;
                }
            }
        }
        for shell in &data.shells {
            let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
            for e in 0..n_elec {
                let x = r_carts[[e, 0]] - cx;
                let y = r_carts[[e, 1]] - cy;
                let z = r_carts[[e, 2]] - cz;
                let r2 = x * x + y * y + z * z;
                xv[e] = x;
                yv[e] = y;
                zv[e] = z;
                r2v[e] = r2;
                alive[e] = shell.pz_deriv * r2 <= LEPSBAS;
            }
            nr.fill(0.0);
            znr.fill(0.0);
            z2nr.fill(0.0);
            for q in 0..shell.prim_z.len() {
                let ze = shell.prim_z[q];
                for e in 0..n_elec {
                    args[e] = -ze * r2v[e];
                }
                exp_pd_into(&args, &mut expv);
                for e in 0..n_elec {
                    if !alive[e] {
                        continue;
                    }
                    let a = shell.prim_nc[q] * expv[e];
                    nr[e] += a;
                    znr[e] += ze * a;
                    z2nr[e] += ze * ze * a;
                }
            }
            for e in 0..n_elec {
                if !alive[e] {
                    continue;
                }
                let (x, y, z, r2) = (xv[e], yv[e], zv[e], r2v[e]);
                let (xp, yp, zp) = cart_power_tables(x, y, z, 7);
                for &ao in &shell.aos {
                    let scale = radial_scale[ao];
                    let nr = nr[e] * scale;
                    let znr = znr[e] * scale;
                    let z2nr = z2nr[e] * scale;
                    let (nx, ny, nz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                    let xx = powm(x, &xp, nx);
                    let yy = powm(y, &yp, ny);
                    let zz = powm(z, &zp, nz);
                    let pp = xx * yy * zz;
                    let dpx = nx as f64 * powm(x, &xp, nx - 1) * yy * zz;
                    let dpy = xx * ny as f64 * powm(y, &yp, ny - 1) * zz;
                    let dpz = xx * yy * nz as f64 * powm(z, &zp, nz - 1);
                    let lap_p = nx as f64 * (nx - 1) as f64 * powm(x, &xp, nx - 2) * yy * zz
                        + xx * ny as f64 * (ny - 1) as f64 * powm(y, &yp, ny - 2) * zz
                        + xx * yy * nz as f64 * (nz - 1) as f64 * powm(z, &zp, nz - 2);
                    gx[[ao, e]] = nr * dpx - 2.0 * x * znr * pp;
                    gy[[ao, e]] = nr * dpy - 2.0 * y * znr * pp;
                    gz[[ao, e]] = nr * dpz - 2.0 * z * znr * pp;
                    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                    lap[[ao, e]] =
                        nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
                }
            }
        }
        return (
            fold_image_axis(data, gx),
            fold_image_axis(data, gy),
            fold_image_axis(data, gz),
            fold_image_axis(data, lap),
        );
    }

    if data.kind != AoKind::Sphe {
        return compute_aos_grad_lap(data, r_carts);
    }
    for shell in &data.shells {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        let pz = &shell.prim_z;
        let pnc = &shell.prim_nc;
        let pz_deriv = shell.pz_deriv;
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - cx;
            let y = r_carts[[e, 1]] - cy;
            let z = r_carts[[e, 2]] - cz;
            let r2 = x * x + y * y + z * z;
            // FAST-mode derivative cutoff: applied for molecular AND periodic alike.
            if pz_deriv * r2 > LEPSBAS {
                continue;
            }
            let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
            for i in 0..pz.len() {
                let a = pnc[i] * exp_neg_fast_scalar(-pz[i] * r2);
                nr += a;
                znr += pz[i] * a;
                z2nr += pz[i] * pz[i] * a;
            }
            let radial_lap_c = 4.0 * r2 * z2nr - 6.0 * znr;
            for &ao in &shell.aos {
                let (pp, dpx, dpy, dpz) = sphe_angular_value_grad(data, ao, x, y, z, r2);
                gx[[ao, e]] = nr * dpx - 2.0 * x * znr * pp;
                gy[[ao, e]] = nr * dpy - 2.0 * y * znr * pp;
                gz[[ao, e]] = nr * dpz - 2.0 * z * znr * pp;
                let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                lap[[ao, e]] = -4.0 * znr * r_dot_grad_p + radial_lap_c * pp;
            }
        }
    }
    (
        fold_image_axis(data, gx),
        fold_image_axis(data, gy),
        fold_image_axis(data, gz),
        fold_image_axis(data, lap),
    )
}

/// Allocation-free single-position sibling of [`compute_aos_grad_lap_fast_cut`].
///
/// The GFMC chain updates exactly one moved-electron derivative column after a
/// hop. Sending that one point through the batch kernel allocates four matrices
/// and, on Cartesian bases, pads the one-point exponential stream to four lanes.
/// This shell-reuse kernel evaluates each primitive exponential once and writes
/// the four AO derivative components directly into caller-owned buffers.
pub fn compute_aos_single_value_grad_lap_fast_cut_into(
    data: &AosGtoData,
    pos: [f64; 3],
    value: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    let na = data.num_ao_kernel;
    debug_assert_eq!(data.n_image, 1);
    debug_assert_eq!(value.len(), na);
    debug_assert_eq!(gx.len(), na);
    debug_assert_eq!(gy.len(), na);
    debug_assert_eq!(gz.len(), na);
    debug_assert_eq!(lap.len(), na);

    if data.shells.is_empty() {
        let points = Array2::from_shape_vec((1, 3), pos.to_vec()).expect("one derivative point");
        let value_owned = compute_aos_single(data, pos);
        let (x, y, z, l) = compute_aos_grad_lap(data, &points.view());
        for a in 0..na {
            value[a] = value_owned[a];
            gx[a] = x[[a, 0]];
            gy[a] = y[[a, 0]];
            gz[a] = z[[a, 0]];
            lap[a] = l[[a, 0]];
        }
        return;
    }

    if data.kind == AoKind::Cart {
        let powm = |v: f64, tab: &[f64; 8], n: i64| -> f64 {
            if n < 0 {
                0.0
            } else if n < 8 {
                tab[n as usize]
            } else {
                v.powi(n as i32)
            }
        };
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            for &ao in &shell.aos {
                value[ao] = 0.0;
                gx[ao] = 0.0;
                gy[ao] = 0.0;
                gz[ao] = 0.0;
                lap[ao] = 0.0;
            }
            let derivative_alive = shell.pz_deriv * r2 <= LEPSBAS;
            // Temporarily use gx/gy/gz as the radial moments
            // Sum(c exp), Sum(zeta c exp), Sum(zeta^2 c exp).
            for q in 0..shell.prim_z.len() {
                let ze = shell.prim_z[q];
                let ev = exp_neg_fast_scalar(-ze * r2);
                for &ao in &shell.aos {
                    let p = data.ao_prim_offset[ao] + q;
                    let term = data.flat_nc[p] * ev;
                    value[ao] += term;
                    gx[ao] += term;
                    gy[ao] += ze * term;
                    gz[ao] += ze * ze * term;
                }
            }
            let (xp, yp, zp) = cart_power_tables(x, y, z, 7);
            for &ao in &shell.aos {
                let nr = gx[ao];
                let znr = gy[ao];
                let z2nr = gz[ao];
                let (nx, ny, nz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                let xx = powm(x, &xp, nx);
                let yy = powm(y, &yp, ny);
                let zz = powm(z, &zp, nz);
                let pp = xx * yy * zz;
                value[ao] *= pp;
                if !derivative_alive {
                    gx[ao] = 0.0;
                    gy[ao] = 0.0;
                    gz[ao] = 0.0;
                    lap[ao] = 0.0;
                    continue;
                }
                let dpx = nx as f64 * powm(x, &xp, nx - 1) * yy * zz;
                let dpy = xx * ny as f64 * powm(y, &yp, ny - 1) * zz;
                let dpz = xx * yy * nz as f64 * powm(z, &zp, nz - 1);
                let lap_p = nx as f64 * (nx - 1) as f64 * powm(x, &xp, nx - 2) * yy * zz
                    + xx * ny as f64 * (ny - 1) as f64 * powm(y, &yp, ny - 2) * zz
                    + xx * yy * nz as f64 * (nz - 1) as f64 * powm(z, &zp, nz - 2);
                gx[ao] = nr * dpx - 2.0 * x * znr * pp;
                gy[ao] = nr * dpy - 2.0 * y * znr * pp;
                gz[ao] = nr * dpz - 2.0 * z * znr * pp;
                let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                lap[ao] =
                    nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
            }
        }
        return;
    }

    if data.kind == AoKind::Sphe {
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            for &ao in &shell.aos {
                value[ao] = 0.0;
                gx[ao] = 0.0;
                gy[ao] = 0.0;
                gz[ao] = 0.0;
                lap[ao] = 0.0;
            }
            let derivative_alive = shell.pz_deriv * r2 <= LEPSBAS;
            let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
            for q in 0..shell.prim_z.len() {
                let ze = shell.prim_z[q];
                let term = shell.prim_nc[q] * exp_neg_fast_scalar(-ze * r2);
                nr += term;
                znr += ze * term;
                z2nr += ze * ze * term;
            }
            let radial_lap_c = 4.0 * r2 * z2nr - 6.0 * znr;
            for &ao in &shell.aos {
                let (pp, dpx, dpy, dpz) = sphe_angular_value_grad(data, ao, x, y, z, r2);
                value[ao] = nr * pp;
                if !derivative_alive {
                    continue;
                }
                gx[ao] = nr * dpx - 2.0 * x * znr * pp;
                gy[ao] = nr * dpy - 2.0 * y * znr * pp;
                gz[ao] = nr * dpz - 2.0 * z * znr * pp;
                let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                lap[ao] = -4.0 * znr * r_dot_grad_p + radial_lap_c * pp;
            }
        }
        return;
    }

    unreachable!("AO kind is Cartesian or spherical")
}

/// Image-folded, FAST-mode AO VALUE at ONE position into a caller buffer of
/// HOME rank (`num_ao`) -- the periodic-capable value sibling of gfmc's
/// molecular candidate batch (which asserts `n_image == 1` and can therefore
/// ASSIGN through `kernel_to_home`; here several kernel AOs fold onto one home
/// AO, so the fold ACCUMULATES).
///
/// Contract (home-cell wrap): under PBC (`n_image > 1`) `pos` MUST already be
/// wrapped into the home cell by the caller. The static image tiling is
/// enumerated relative to the home cell (see [`AosGtoData::wrap_cell`]); a raw
/// coordinate several lattice vectors out falls beyond the tiling and the
/// image sum silently loses its dominant terms. Wrapping is NOT done here so
/// the caller controls exactly one wrap site per hop (candidates are wrapped
/// once, then every kernel sees the same bits).
///
/// Screen: the TurboRVB-style per-(shell, point) radial screen
/// `shell.pz_min * r2 > LEPSBAS -> skip` is applied ONLY when periodic
/// (`n_image > 1`), matching [`compute_aos_kernel_screened`]'s
/// `periodic = screen && n_image > 1` rule -- so for a molecular leaf this
/// value is bit-identical to the unscreened molecular candidate convention.
///
/// Floating-point conventions:
/// - kernel phase: shell-major, radial contraction in the per-AO `flat_nc`
///   primitive order (the [`compute_aos_single_value_grad_lap_fast_cut_into`]
///   convention) -- on Cart the value here is bit-identical to that kernel's
///   value; on Sphe the angular part uses the value-only
///   [`eval_angular_flat`] replay (the molecular candidate-batch convention),
///   which may differ from the derivative kernel's `sphe_angular_value_grad`
///   replay at the ULP level, exactly as in the existing molecular chain.
/// - fold phase: kernel-AO ASCENDING order, identical to
///   [`fold_single`] / [`fold_image_axis`].
///   `kernel_scratch` is resized to `num_ao_kernel` (allocation-free in steady
///   state). Scalar `exp` (an AVX `exp_pd_into` batch variant would slot into the
///   kernel phase, as in gfmc's molecular candidate batch).
// Used by the GFMC fixed-mesh candidate and derivative routes.
pub(crate) fn compute_aos_single_value_fast_cut_folded_into(
    data: &AosGtoData,
    pos: [f64; 3],
    kernel_scratch: &mut Vec<f64>,
    out: &mut [f64],
) {
    debug_assert!(!data.shells.is_empty());
    debug_assert_eq!(out.len(), data.num_ao);
    let nk = data.num_ao_kernel;
    kernel_scratch.clear();
    kernel_scratch.resize(nk, 0.0);
    let kv = kernel_scratch.as_mut_slice();
    let periodic = data.n_image > 1;
    match data.kind {
        AoKind::Cart => {
            let powm = |v: f64, tab: &[f64; 8], n: i64| -> f64 {
                if n < 0 {
                    0.0
                } else if n < 8 {
                    tab[n as usize]
                } else {
                    v.powi(n as i32)
                }
            };
            for shell in &data.shells {
                let x = pos[0] - shell.center[0];
                let y = pos[1] - shell.center[1];
                let z = pos[2] - shell.center[2];
                let r2 = x * x + y * y + z * z;
                if periodic && shell.pz_min * r2 > LEPSBAS {
                    continue; // negligible image: home rows keep the other images' sum
                }
                for q in 0..shell.prim_z.len() {
                    let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                    for &ao in &shell.aos {
                        let p = data.ao_prim_offset[ao] + q;
                        kv[ao] += data.flat_nc[p] * ev;
                    }
                }
                let (xp, yp, zp) = cart_power_tables(x, y, z, 7);
                for &ao in &shell.aos {
                    kv[ao] *= powm(x, &xp, data.poly_x[ao])
                        * powm(y, &yp, data.poly_y[ao])
                        * powm(z, &zp, data.poly_z[ao]);
                }
            }
        }
        AoKind::Sphe => {
            for shell in &data.shells {
                let x = pos[0] - shell.center[0];
                let y = pos[1] - shell.center[1];
                let z = pos[2] - shell.center[2];
                let r2 = x * x + y * y + z * z;
                if periodic && shell.pz_min * r2 > LEPSBAS {
                    continue;
                }
                let mut rad = 0.0_f64;
                for q in 0..shell.prim_z.len() {
                    rad += shell.prim_nc[q] * exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                }
                for &ao in &shell.aos {
                    kv[ao] = rad * eval_angular_flat(data, ao, x, y, z, r2);
                }
            }
        }
    }
    if data.n_image == 1 {
        out.copy_from_slice(kv);
        return;
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for k in 0..nk {
        out[data.kernel_to_home[k]] += kv[k];
    }
}

/// Candidate-batch VALUE rows `(npoint, num_ao)` (row-major, home rank) for a
/// periodic pGTO leaf -- the `n_image > 1`-capable counterpart of gfmc's
/// `molecular_aos_candidate_batch`. Thin loop over
/// [`compute_aos_single_value_fast_cut_folded_into`]; see that kernel's doc for
/// the wrap contract (positions pre-wrapped under PBC), the periodic-only
/// LEPSBAS screen and the kernel-AO-ascending fold order.
// Used by the GFMC fixed-mesh candidate and derivative routes.
pub(crate) fn compute_aos_candidate_batch_folded(
    data: &AosGtoData,
    positions: &[[f64; 3]],
) -> Array2<f64> {
    // All candidate points of a call in one point-vectorized pass of the
    // screened shell kernel (`kernel_*_soa`: the same per-shell image screen,
    // flat radial moments and fast exp as the single-point kernel), then the
    // ascending kernel->home fold. Equal to the per-point
    // `compute_aos_single_value_fast_cut_folded_into` up to the multiplication
    // order of Cartesian powers >= 4 (an ulp class inside the FAST screen bound).
    let n = positions.len();
    let mut r = Array2::<f64>::zeros((n, 3));
    for (v, p) in positions.iter().enumerate() {
        r[[v, 0]] = p[0];
        r[[v, 1]] = p[1];
        r[[v, 2]] = p[2];
    }
    let folded = fold_image_axis(data, compute_aos_kernel_screened(data, &r.view(), true));
    folded.t().as_standard_layout().into_owned()
}

/// Image-FOLDED sibling of [`compute_aos_single_value_grad_lap_fast_cut_into`]:
/// value/grad(x,y,z)/laplacian at ONE position into caller buffers of HOME rank
/// (`num_ao` each), accumulating over periodic images. The kernel-rank original
/// asserts `n_image == 1` and writes `num_ao_kernel` buffers with no fold.
///
/// Contract (home-cell wrap): under PBC `pos` MUST already be wrapped into the
/// home cell by the caller (same contract as
/// [`compute_aos_single_value_fast_cut_folded_into`]).
///
/// fp-consistency with the batch init: evaluation is TWO-PHASE -- the kernel
/// phase fills kernel-rank scratch shell-by-shell with EXACTLY the kernel-rank
/// sibling's math (per-AO `flat_nc` radial moments, `cart_power_tables`
/// monomials / `sphe_angular_value_grad` angular replay, `derivative_alive`
/// screen), then the fold phase scatters `kernel -> home` in kernel-AO
/// ASCENDING order, bit-matching [`fold_image_axis`] as used by the batch
/// [`compute_aos_grad_lap_fast_cut`]. So a fresh batch-initialized derivative
/// table and this single-point update agree on the per-home-AO image-term
/// ORDER; the remaining batch-vs-single differences are the ones the molecular
/// chain already tolerates (radial `flat_nc` form here vs the batch's
/// `radial_scale` factorization on Cart, and `exp_pd_into` vs libm `exp` --
/// both ULP class).
///
/// Screens:
/// - VALUE: periodic-only shell skip (`n_image > 1 && pz_min*r2 > LEPSBAS`),
///   matching the folded candidate value batch it shares tables with. For
///   `n_image == 1` this makes the whole kernel bit-identical to the
///   kernel-rank sibling (unscreened value, identity fold).
/// - GRAD/LAP: the FAST-mode `derivative_alive = pz_deriv*r2 <= LEPSBAS`
///   zeroing (dropped derivatives `< FAST_SCREEN_ABS_BOUND = 1e-12` in absolute AO units, see
///   [`Shell::pz_deriv`]), applied for molecular AND periodic alike -- the
///   [`compute_aos_grad_lap_fast_cut`] rule. A periodic image whose value is
///   screened but whose derivatives are not keeps its derivative terms.
///
/// `kernel_scratch` is resized to `5 * num_ao_kernel` (allocation-free in
/// steady state). This scalar implementation does not use AVX; an
/// `exp_pd_into`-based vector radial pass would fit the per-shell primitive loop.
// Used by the GFMC fixed-mesh candidate and derivative routes.
#[allow(
    clippy::too_many_arguments,
    reason = "caller-owned value/grad/lap output buffers"
)]
pub(crate) fn compute_aos_single_value_grad_lap_fast_cut_folded_into(
    data: &AosGtoData,
    pos: [f64; 3],
    kernel_scratch: &mut Vec<f64>,
    value: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    let na = data.num_ao;
    let nk = data.num_ao_kernel;
    debug_assert!(!data.shells.is_empty());
    debug_assert_eq!(value.len(), na);
    debug_assert_eq!(gx.len(), na);
    debug_assert_eq!(gy.len(), na);
    debug_assert_eq!(gz.len(), na);
    debug_assert_eq!(lap.len(), na);
    kernel_scratch.clear();
    kernel_scratch.resize(5 * nk, 0.0);
    let (kv, rest) = kernel_scratch.split_at_mut(nk);
    let (kgx, rest) = rest.split_at_mut(nk);
    let (kgy, rest) = rest.split_at_mut(nk);
    let (kgz, klap) = rest.split_at_mut(nk);
    let periodic = data.n_image > 1;

    if data.kind == AoKind::Cart {
        let powm = |v: f64, tab: &[f64; 8], n: i64| -> f64 {
            if n < 0 {
                0.0
            } else if n < 8 {
                tab[n as usize]
            } else {
                v.powi(n as i32)
            }
        };
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            // Periodic image screen on the VALUE (pz_min, shared with the folded
            // candidate value batch); the derivatives keep their own
            // AO-magnitude-aware `pz_deriv` cut, as in the batch
            // `compute_aos_grad_lap_fast_cut`. `pz_deriv <= pz_min`, so an image
            // whose derivatives are screened has its value screened too.
            let derivative_alive = shell.pz_deriv * r2 <= LEPSBAS;
            if periodic && !derivative_alive {
                continue; // negligible image (value AND derivatives)
            }
            let value_alive = !periodic || shell.pz_min * r2 <= LEPSBAS;
            // Temporarily use kgx/kgy/kgz as the radial moments
            // Sum(c exp), Sum(zeta c exp), Sum(zeta^2 c exp) -- exactly the
            // kernel-rank sibling's buffer reuse.
            for q in 0..shell.prim_z.len() {
                let ze = shell.prim_z[q];
                let ev = exp_neg_fast_scalar(-ze * r2);
                for &ao in &shell.aos {
                    let p = data.ao_prim_offset[ao] + q;
                    let term = data.flat_nc[p] * ev;
                    kv[ao] += term;
                    kgx[ao] += term;
                    kgy[ao] += ze * term;
                    kgz[ao] += ze * ze * term;
                }
            }
            let (xp, yp, zp) = cart_power_tables(x, y, z, 7);
            for &ao in &shell.aos {
                let nr = kgx[ao];
                let znr = kgy[ao];
                let z2nr = kgz[ao];
                let (nx, ny, nz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                let xx = powm(x, &xp, nx);
                let yy = powm(y, &yp, ny);
                let zz = powm(z, &zp, nz);
                let pp = xx * yy * zz;
                kv[ao] = if value_alive { kv[ao] * pp } else { 0.0 };
                if !derivative_alive {
                    kgx[ao] = 0.0;
                    kgy[ao] = 0.0;
                    kgz[ao] = 0.0;
                    continue; // klap[ao] untouched: still 0
                }
                let dpx = nx as f64 * powm(x, &xp, nx - 1) * yy * zz;
                let dpy = xx * ny as f64 * powm(y, &yp, ny - 1) * zz;
                let dpz = xx * yy * nz as f64 * powm(z, &zp, nz - 1);
                let lap_p = nx as f64 * (nx - 1) as f64 * powm(x, &xp, nx - 2) * yy * zz
                    + xx * ny as f64 * (ny - 1) as f64 * powm(y, &yp, ny - 2) * zz
                    + xx * yy * nz as f64 * (nz - 1) as f64 * powm(z, &zp, nz - 2);
                kgx[ao] = nr * dpx - 2.0 * x * znr * pp;
                kgy[ao] = nr * dpy - 2.0 * y * znr * pp;
                kgz[ao] = nr * dpz - 2.0 * z * znr * pp;
                let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                klap[ao] =
                    nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
            }
        }
    } else {
        for shell in &data.shells {
            let x = pos[0] - shell.center[0];
            let y = pos[1] - shell.center[1];
            let z = pos[2] - shell.center[2];
            let r2 = x * x + y * y + z * z;
            // Value / derivative screens as in the Cartesian branch above.
            let derivative_alive = shell.pz_deriv * r2 <= LEPSBAS;
            if periodic && !derivative_alive {
                continue;
            }
            let value_alive = !periodic || shell.pz_min * r2 <= LEPSBAS;
            let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
            for q in 0..shell.prim_z.len() {
                let ze = shell.prim_z[q];
                let term = shell.prim_nc[q] * exp_neg_fast_scalar(-ze * r2);
                nr += term;
                znr += ze * term;
                z2nr += ze * ze * term;
            }
            let radial_lap_c = 4.0 * r2 * z2nr - 6.0 * znr;
            for &ao in &shell.aos {
                let (pp, dpx, dpy, dpz) = sphe_angular_value_grad(data, ao, x, y, z, r2);
                if value_alive {
                    kv[ao] = nr * pp;
                }
                if !derivative_alive {
                    continue;
                }
                kgx[ao] = nr * dpx - 2.0 * x * znr * pp;
                kgy[ao] = nr * dpy - 2.0 * y * znr * pp;
                kgz[ao] = nr * dpz - 2.0 * z * znr * pp;
                let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                klap[ao] = -4.0 * znr * r_dot_grad_p + radial_lap_c * pp;
            }
        }
    }

    // Fold phase: kernel -> home in kernel-AO ASCENDING order (bit-matching
    // fold_image_axis / fold_single). Identity copy for the molecular rank.
    if data.n_image == 1 {
        value.copy_from_slice(kv);
        gx.copy_from_slice(kgx);
        gy.copy_from_slice(kgy);
        gz.copy_from_slice(kgz);
        lap.copy_from_slice(klap);
        return;
    }
    for buf in [&mut *value, &mut *gx, &mut *gy, &mut *gz, &mut *lap] {
        for o in buf.iter_mut() {
            *o = 0.0;
        }
    }
    for k in 0..nk {
        let h = data.kernel_to_home[k];
        value[h] += kv[k];
        gx[h] += kgx[k];
        gy[h] += kgy[k];
        gz[h] += kgz[k];
        lap[h] += klap[k];
    }
}

/// Single-position AO VALUE + gradient + Laplacian, `(num_ao,)` each
/// (image-folded), in the same exact full-image convention as
/// [`compute_aos_single`] (the per-move VALUE path the MCMC sweep seeds/streams
/// its Sherman-Morrison `geminal_inv` with -- `orb_matrix_via_single`). Unlike the
/// BATCH [`compute_aos_grad_lap`]. The value + grad + laplacian all come from the SAME per-AO
/// `ao_value_grad_lap` call over `num_ao_kernel`, so they are mutually
/// self-consistent (a sphe value taken from `compute_aos_single`'s shell-reuse
/// path instead would use a different iteration and mismatch the grad). This is
/// the value+grad/laplacian sibling of `orb_matrix_via_single`: the observable
/// kinetic grad^2 G must use the SAME AO convention as the reused SM inverse, else
/// the ~1e-9 single-vs-batch mismatch is amplified by the large Gaussian curvature
/// into a ~0.06 Ha kinetic error. This function does not home-cell-wrap.
pub fn compute_aos_single_value_grad_lap(
    data: &AosGtoData,
    pos: [f64; 3],
) -> ValueGradLap<Vec<f64>> {
    compute_aos_single_value_grad_lap_with_policy(data, pos, false)
}

/// [`compute_aos_single_value_grad_lap`] with the fused `ao_eval` value cast of
/// jqmc's `compute_AOs_value_grad_lap`: when `value_f32`, each
/// kernel-image value is rounded to fp32 and the image fold of the VALUE runs
/// in fp32 (`_fold_image_axis(val, "ao_eval")` preserves the value dtype);
/// gradients and Laplacians stay fp64.
pub(crate) fn compute_aos_single_value_grad_lap_with_policy(
    data: &AosGtoData,
    pos: [f64; 3],
    value_f32: bool,
) -> ValueGradLap<Vec<f64>> {
    let nk = data.num_ao_kernel;
    let mut val = vec![0.0_f64; nk];
    let mut gx = vec![0.0_f64; nk];
    let mut gy = vec![0.0_f64; nk];
    let mut gz = vec![0.0_f64; nk];
    let mut lap = vec![0.0_f64; nk];
    if data.shells.is_empty() {
        single_vgl_per_ao(data, pos, &mut val, &mut gx, &mut gy, &mut gz, &mut lap);
    } else {
        single_vgl_per_shell(data, pos, &mut val, &mut gx, &mut gy, &mut gz, &mut lap);
    }
    let val = if value_f32 {
        fold_single_f32(data, val)
    } else {
        fold_single(data, val)
    };
    (
        val,
        fold_single(data, gx),
        fold_single(data, gy),
        fold_single(data, gz),
        fold_single(data, lap),
    )
}

/// Exponent argument beyond which glibc `exp` returns exactly 0 in fp64; a
/// shell with `pz_min * r^2` above it contributes exactly zero (no screening
/// error), so the full-image convention is unchanged.
const EXP_UNDERFLOW_ARG: f64 = 745.2;

/// Shell-grouped form of [`single_vgl_per_ao`]: one distance and one `exp` per
/// primitive per shell (shared by every AO of the shell, whose primitive order
/// is the shell's), power tables instead of `powi`, and exactly-zero far images
/// skipped. Same formulas and primitive summation order per AO.
fn single_vgl_per_shell(
    data: &AosGtoData,
    pos: [f64; 3],
    val: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    let mut ev: Vec<f64> = Vec::with_capacity(32);
    let nlm_of = |l: i64| ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
    for shell in &data.shells {
        let x = pos[0] - shell.center[0];
        let y = pos[1] - shell.center[1];
        let z = pos[2] - shell.center[2];
        let r2 = x * x + y * y + z * z;
        if shell.pz_min * r2 > EXP_UNDERFLOW_ARG {
            continue; // outputs already zero
        }
        ev.clear();
        ev.extend(shell.prim_z.iter().map(|&ze| exp_neg_fast_scalar(-ze * r2)));
        match data.kind {
            AoKind::Cart => {
                let (xp, yp, zp) = cart_power_tables(x, y, z, 7);
                let powm = |v: f64, tab: &[f64; 8], n: i64| -> f64 {
                    if n < 0 {
                        0.0
                    } else if n < 8 {
                        tab[n as usize]
                    } else {
                        v.powi(n as i32)
                    }
                };
                for &ao in &shell.aos {
                    let off = data.ao_prim_offset[ao];
                    let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for (q, &ze) in shell.prim_z.iter().enumerate() {
                        let pref = data.flat_nc[off + q] * ev[q];
                        nr += pref;
                        znr += ze * pref;
                        z2nr += ze * ze * pref;
                    }
                    let (nx, ny, nz) = (data.poly_x[ao], data.poly_y[ao], data.poly_z[ao]);
                    let xx = powm(x, &xp, nx);
                    let yy = powm(y, &yp, ny);
                    let zz = powm(z, &zp, nz);
                    let pp = xx * yy * zz;
                    let dpx = nx as f64 * powm(x, &xp, nx - 1) * yy * zz;
                    let dpy = xx * ny as f64 * powm(y, &yp, ny - 1) * zz;
                    let dpz = xx * yy * nz as f64 * powm(z, &zp, nz - 1);
                    let lap_p = nx as f64 * (nx - 1) as f64 * powm(x, &xp, nx - 2) * yy * zz
                        + xx * ny as f64 * (ny - 1) as f64 * powm(y, &yp, ny - 2) * zz
                        + xx * yy * nz as f64 * (nz - 1) as f64 * powm(z, &zp, nz - 2);
                    val[ao] = nr * pp;
                    gx[ao] = nr * dpx - 2.0 * x * znr * pp;
                    gy[ao] = nr * dpy - 2.0 * y * znr * pp;
                    gz[ao] = nr * dpz - 2.0 * z * znr * pp;
                    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                    lap[ao] =
                        nr * lap_p - 4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
                }
            }
            AoKind::Sphe => {
                let r_norm = r2.sqrt();
                for &ao in &shell.aos {
                    let off = data.ao_prim_offset[ao];
                    let (mut nr, mut znr, mut z2nr) = (0.0_f64, 0.0_f64, 0.0_f64);
                    for (q, &ze) in shell.prim_z.iter().enumerate() {
                        let pref = data.flat_nc[off + q] * ev[q];
                        nr += pref;
                        znr += ze * pref;
                        z2nr += ze * ze * pref;
                    }
                    let nlm = nlm_of(data.angular_momentums[ao]);
                    let (s, sx, sy, sz) = if let Some(tab) = data.sphe_ang_vg.get(ao) {
                        tab.eval(x, y, z, r_norm)
                    } else {
                        s_l_m_value_and_grad(
                            data.angular_momentums[ao],
                            data.magnetic[ao],
                            x,
                            y,
                            z,
                            r_norm,
                        )
                    };
                    let (pp, dpx, dpy, dpz) = (nlm * s, nlm * sx, nlm * sy, nlm * sz);
                    val[ao] = nr * pp;
                    gx[ao] = nr * dpx - 2.0 * x * znr * pp;
                    gy[ao] = nr * dpy - 2.0 * y * znr * pp;
                    gz[ao] = nr * dpz - 2.0 * z * znr * pp;
                    let r_dot_grad_p = x * dpx + y * dpy + z * dpz;
                    lap[ao] = -4.0 * znr * r_dot_grad_p + (4.0 * r2 * z2nr - 6.0 * znr) * pp;
                }
            }
        }
    }
}

/// Reference per-AO evaluation (one `ao_value_grad_lap` per kernel AO).
fn single_vgl_per_ao(
    data: &AosGtoData,
    pos: [f64; 3],
    val: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    for ao in 0..data.num_ao_kernel {
        let x = pos[0] - data.centers[[ao, 0]];
        let y = pos[1] - data.centers[[ao, 1]];
        let z = pos[2] - data.centers[[ao, 2]];
        let vgl = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z);
        val[ao] = vgl.value;
        gx[ao] = vgl.grad[0];
        gy[ao] = vgl.grad[1];
        gz[ao] = vgl.grad[2];
        lap[ao] = vgl.laplacian;
    }
}

/// Analytic AO laplacian. Returns `(num_ao_kernel, n_elec)`,
/// mirroring `compute_AOs_laplacian`.
pub fn compute_aos_laplacian(data: &AosGtoData, r_carts: &ArrayView2<f64>) -> Array2<f64> {
    let n_elec = r_carts.nrows();
    let groups = &data.prim_groups;
    let norms = &data.normalizations;
    let mut lap = Array2::<f64>::zeros((data.num_ao_kernel, n_elec));
    // EXACT observable/parity path: NO runtime LEPSBAS distance screen. jqmc truncates
    // the periodic image set once at BUILD time and then evaluates every surviving
    // kernel AO exactly for every electron; a runtime per-(AO, electron) skip drops
    // terms XLA keeps. The opt-in approximation
    // lives in `compute_aos_grad_lap_fast_cut` (production FAST chunk only).
    for ao in 0..data.num_ao_kernel {
        for e in 0..n_elec {
            let x = r_carts[[e, 0]] - data.centers[[ao, 0]];
            let y = r_carts[[e, 1]] - data.centers[[ao, 1]];
            let z = r_carts[[e, 2]] - data.centers[[ao, 2]];
            lap[[ao, e]] = ao_value_grad_lap(data, norms, ao, &groups[ao], x, y, z).laplacian;
        }
    }
    fold_image_axis(data, lap)
}

// ============================================================================
// Plane-wave AO basis (`AOs_plane_wave_data`) -- one child of a pgto(+)pw composite.
// Real Gamma-point basis: phi_k(r) = cos(G_k*r - phase_k), phase_k = is_sin_k*pi/2
// (cos-basis -> cos(G*r), sin-basis -> sin(G*r)). G_k = m_k @ (2pi*inv(cell)T),
// precomputed Cartesian. grad = -sin(G*r-phase)*G, lap = -|G|^2*value. In exact
// arithmetic cos/sin are wrap-invariant, but under mixed precision evaluating an
// unwrapped coordinate changes the f32 argument reduction. jqmc wraps in fp64
// before entering the ao_eval zone, so the Rust f32 path must do the same.
// ============================================================================
#[derive(Clone)]
pub struct AosPwData {
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    pub gz: Vec<f64>,
    pub phase: Vec<f64>, // is_sin * pi/2
    pub g2: Vec<f64>,    // |G|^2
    pub num_ao: usize,
    pub g_mat: Array2<f64>, // (num_ao, 3) Cartesian G rows (for the theta = G @ r^T GEMM)
    pub g_mat_f32: Array2<f32>,
    pub phase_f32: Vec<f32>,
    pub wrap_cell: [[f64; 3]; 3],
    pub wrap_inv: [[f64; 3]; 3],
    /// Construction-time arithmetic policy (see `AosGtoData::precision_policy`).
    pub(crate) precision_policy: crate::PrecisionPolicy,
}

impl AosPwData {
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // `g_vectors` is a @property: Cartesian G = m @ (2pi*inv(cell)T), (num_pw,3).
        let g: numpy::PyReadonlyArray2<f64> = obj.getattr("g_vectors")?.extract()?;
        let g = g.as_array();
        let num_ao = g.nrows();
        let is_sin: numpy::PyReadonlyArray1<f64> = obj.getattr("phase_is_sin")?.extract()?;
        let is_sin = is_sin.as_slice()?;
        let cell_a: numpy::PyReadonlyArray2<f64> =
            obj.getattr("structure_data")?.getattr("cell")?.extract()?;
        let cell_view = cell_a.as_array();
        let wrap_cell = [
            [cell_view[[0, 0]], cell_view[[0, 1]], cell_view[[0, 2]]],
            [cell_view[[1, 0]], cell_view[[1, 1]], cell_view[[1, 2]]],
            [cell_view[[2, 0]], cell_view[[2, 1]], cell_view[[2, 2]]],
        ];
        let wrap_inv = crate::complex_det::inv_3x3(&wrap_cell);
        let half_pi = std::f64::consts::FRAC_PI_2;
        let mut d = AosPwData {
            gx: Vec::with_capacity(num_ao),
            gy: Vec::with_capacity(num_ao),
            gz: Vec::with_capacity(num_ao),
            phase: Vec::with_capacity(num_ao),
            g2: Vec::with_capacity(num_ao),
            num_ao,
            g_mat: g.to_owned(),
            g_mat_f32: g.mapv(|value| value as f32),
            phase_f32: Vec::with_capacity(num_ao),
            wrap_cell,
            wrap_inv,
            precision_policy: crate::construction_precision_policy(),
        };
        for k in 0..num_ao {
            let (a, b, c) = (g[[k, 0]], g[[k, 1]], g[[k, 2]]);
            d.gx.push(a);
            d.gy.push(b);
            d.gz.push(c);
            d.phase.push(is_sin[k] * half_pi);
            d.phase_f32.push((is_sin[k] * half_pi) as f32);
            d.g2.push(a * a + b * b + c * c);
        }
        Ok(d)
    }
}

/// Flatten electron coords into contiguous columns for SIMD-over-electrons.
fn flat_r(r: &ArrayView2<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = r.nrows();
    let mut rx = vec![0.0; n];
    let mut ry = vec![0.0; n];
    let mut rz = vec![0.0; n];
    for e in 0..n {
        rx[e] = r[[e, 0]];
        ry[e] = r[[e, 1]];
        rz[e] = r[[e, 2]];
    }
    (rx, ry, rz)
}

/// PW values `(num_ao, n_elec)`: `out[k,e] = cos(G_k*r_e - phase_k)`.
pub fn compute_pw(data: &AosPwData, r: &ArrayView2<f64>) -> Array2<f64> {
    if data.precision_policy.is_f32("ao_eval") {
        return compute_pw_f32(data, r);
    }
    let n_elec = r.nrows();
    let (rx, ry, rz) = flat_r(r);
    // theta = G @ r^T (BLAS gemv/gemm over the num_ao axis), minus the sin-phase,
    // then one 8-lane cos pass over the whole (num_ao, n_elec) buffer.
    let mut rmat = Array2::<f64>::zeros((3, n_elec));
    {
        let rr = rmat.as_slice_mut().unwrap();
        rr[0..n_elec].copy_from_slice(&rx);
        rr[n_elec..2 * n_elec].copy_from_slice(&ry);
        rr[2 * n_elec..3 * n_elec].copy_from_slice(&rz);
    }
    let mut out = data.g_mat.dot(&rmat); // (num_ao, n_elec) = theta (pre-phase)
    let od = out.as_slice_mut().unwrap();
    for k in 0..data.num_ao {
        let phk = data.phase[k];
        if phk != 0.0 {
            let row = &mut od[k * n_elec..k * n_elec + n_elec];
            for v in row.iter_mut() {
                *v -= phk;
            }
        }
    }
    cos_pd_inplace(od); // in-place 8-lane cos (~1 ULP)
    out
}

pub(crate) fn compute_pw_with_policy(
    data: &AosPwData,
    r: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> Array2<f64> {
    if f32_arithmetic {
        compute_pw_f32(data, r)
    } else {
        // Inline fp64 scalar form to keep context execution independent of the
        // standalone session policy.
        let mut out = Array2::<f64>::zeros((data.num_ao, r.nrows()));
        for k in 0..data.num_ao {
            for e in 0..r.nrows() {
                out[[k, e]] =
                    (data.gx[k] * r[[e, 0]] + data.gy[k] * r[[e, 1]] + data.gz[k] * r[[e, 2]]
                        - data.phase[k])
                        .cos();
            }
        }
        out
    }
}

fn compute_pw_f32(data: &AosPwData, r: &ArrayView2<f64>) -> Array2<f64> {
    let wrapped: Vec<[f32; 3]> = (0..r.nrows())
        .map(|e| {
            let p = wrap_pw_position(data, [r[[e, 0]], r[[e, 1]], r[[e, 2]]]);
            [p[0] as f32, p[1] as f32, p[2] as f32]
        })
        .collect();
    let mut rmat = Array2::<f32>::zeros((3, r.nrows()));
    for e in 0..r.nrows() {
        rmat[[0, e]] = wrapped[e][0];
        rmat[[1, e]] = wrapped[e][1];
        rmat[[2, e]] = wrapped[e][2];
    }
    // Match XLA's `jnp.dot(rc, g.T)` reduction rather than a scalar three-term
    // expression. The two are algebraically identical, but their f32 FMA /
    // reassociation differs by several ulps and is amplified by determinant
    // inverses in parameter derivatives.
    let theta = data.g_mat_f32.dot(&rmat);
    let mut out = Array2::<f64>::zeros((data.num_ao, r.nrows()));
    // fp32 cos of the zone-local argument with the vectorizable
    // `fastmath::sincos_f32` (IMPL-RS-F32-AO-EVAL-SIMD-001).
    let n = r.nrows();
    let mut arg = vec![0.0_f32; n];
    for k in 0..data.num_ao {
        let ph = data.phase_f32[k];
        for e in 0..n {
            arg[e] = theta[[k, e]] - ph;
        }
        let mut row = out.row_mut(k);
        for e in 0..n {
            row[e] = crate::fastmath::sincos_f32(arg[e]).1 as f64;
        }
    }
    out
}

#[inline]
fn wrap_pw_position(data: &AosPwData, r: [f64; 3]) -> [f64; 3] {
    let mut frac = [0.0_f64; 3];
    for (k, frac_k) in frac.iter_mut().enumerate() {
        let s =
            r[0] * data.wrap_inv[0][k] + r[1] * data.wrap_inv[1][k] + r[2] * data.wrap_inv[2][k];
        *frac_k = s - s.floor();
    }
    let mut out = [0.0_f64; 3];
    for (k, out_k) in out.iter_mut().enumerate() {
        *out_k = frac[0] * data.wrap_cell[0][k]
            + frac[1] * data.wrap_cell[1][k]
            + frac[2] * data.wrap_cell[2][k];
    }
    out
}

/// PW values at one position into `out[0..num_ao]` (the per-move PW hot path:
/// ~10^4 cos/move; libm scalar cos).
pub fn compute_pw_single_into(data: &AosPwData, pos: [f64; 3], out: &mut [f64]) {
    for (k, out_k) in out[..data.num_ao].iter_mut().enumerate() {
        *out_k =
            (data.gx[k] * pos[0] + data.gy[k] * pos[1] + data.gz[k] * pos[2] - data.phase[k]).cos();
    }
}

/// PW values at one position, `(num_ao,)`.
pub fn compute_pw_single(data: &AosPwData, pos: [f64; 3]) -> Vec<f64> {
    let mut out = vec![0.0_f64; data.num_ao];
    compute_pw_single_into(data, pos, &mut out);
    out
}

/// Fused PW value + grad + laplacian, each `(num_ao, n_elec)`.
/// `val=cos(theta')`, `grad=-sin(theta')*G`, `lap=-|G|^2*val`, with `theta'=G*r-phase`.
pub fn compute_pw_value_grad_lap(data: &AosPwData, r: &ArrayView2<f64>) -> ValueGradLap {
    let n_elec = r.nrows();
    let mut wrapped = Array2::<f64>::zeros((n_elec, 3));
    for e in 0..n_elec {
        let p = wrap_pw_position(data, [r[[e, 0]], r[[e, 1]], r[[e, 2]]]);
        wrapped[[e, 0]] = p[0];
        wrapped[[e, 1]] = p[1];
        wrapped[[e, 2]] = p[2];
    }
    let (rx, ry, rz) = flat_r(&wrapped.view());
    let m = data.num_ao;
    let mut gx = Array2::<f64>::zeros((m, n_elec));
    let mut gy = Array2::<f64>::zeros((m, n_elec));
    let mut gz = Array2::<f64>::zeros((m, n_elec));
    let mut lap = Array2::<f64>::zeros((m, n_elec));
    // theta = G @ r^T (one BLAS gemm), then per-row 8-lane sin_cos.
    let mut rmat = Array2::<f64>::zeros((3, n_elec));
    {
        let rr = rmat.as_slice_mut().unwrap();
        rr[0..n_elec].copy_from_slice(&rx);
        rr[n_elec..2 * n_elec].copy_from_slice(&ry);
        rr[2 * n_elec..3 * n_elec].copy_from_slice(&rz);
    }
    let mut val = data.g_mat.dot(&rmat); // (m, n_elec) = theta (pre-phase); becomes val=cos in place
    let vd = val.as_slice_mut().unwrap();
    let gxd = gx.as_slice_mut().unwrap();
    let gyd = gy.as_slice_mut().unwrap();
    let gzd = gz.as_slice_mut().unwrap();
    let lapd = lap.as_slice_mut().unwrap();
    let mut theta = vec![0.0_f64; n_elec];
    let mut sbuf = vec![0.0_f64; n_elec];
    let mut cbuf = vec![0.0_f64; n_elec];
    let value_is_f32 = data.precision_policy.is_f32("ao_eval");
    for k in 0..m {
        let (gxk, gyk, gzk, phk, g2k) = (
            data.gx[k],
            data.gy[k],
            data.gz[k],
            data.phase[k],
            data.g2[k],
        );
        let base = k * n_elec;
        for e in 0..n_elec {
            theta[e] = vd[base + e] - phk;
        }
        sincos_pd_into(&theta, &mut sbuf, &mut cbuf); // 8-lane sin_cos (~1 ULP)
        for e in 0..n_elec {
            let (s, c) = (sbuf[e], cbuf[e]);
            // XLA's fused kernel owns the shared trigonometric body in the
            // fp64 ao_grad_lap zone, then rounds only the completed value to
            // ao_eval. Gradients and laplacian keep the original fp64 `c`.
            vd[base + e] = if value_is_f32 { (c as f32) as f64 } else { c };
            gxd[base + e] = -s * gxk;
            gyd[base + e] = -s * gyk;
            gzd[base + e] = -s * gzk;
            lapd[base + e] = -g2k * c;
        }
    }
    (val, gx, gy, gz, lap)
}

/// Single-position sibling of [`compute_pw_value_grad_lap`]: PW value + grad +
/// laplacian at one position into caller buffers (each `num_ao`).
/// `val = cos(theta')`, `grad = -sin(theta')*G`, `lap = -|G|^2*val`, with
/// `theta' = G*pos - phase` in the exact [`compute_pw_single_into`] term order,
/// so the VALUE here bit-matches the per-move PW value path. No internal wrap
/// (also matching `compute_pw_single_into`): under PBC pass the pre-wrapped
/// candidate position -- the batch sibling wraps internally, so batch/single
/// agree for home-cell positions (the cos/sin are wrap-invariant only in exact
/// math). Scalar libm sin/cos; the batch's 8-lane `sincos_pd` (~1 ULP) is where
/// a vector variant would slot in. This kernel is f64-only and has no `ao_eval` f32
/// value rounding (the batch applies it under the mixed policy).
// Used by the GFMC fixed-mesh candidate and derivative routes.
pub(crate) fn compute_pw_single_value_grad_lap_into(
    data: &AosPwData,
    pos: [f64; 3],
    value: &mut [f64],
    gx: &mut [f64],
    gy: &mut [f64],
    gz: &mut [f64],
    lap: &mut [f64],
) {
    debug_assert_eq!(value.len(), data.num_ao);
    debug_assert_eq!(gx.len(), data.num_ao);
    debug_assert_eq!(gy.len(), data.num_ao);
    debug_assert_eq!(gz.len(), data.num_ao);
    debug_assert_eq!(lap.len(), data.num_ao);
    for k in 0..data.num_ao {
        let theta = data.gx[k] * pos[0] + data.gy[k] * pos[1] + data.gz[k] * pos[2] - data.phase[k];
        let c = theta.cos();
        let s = theta.sin();
        value[k] = c;
        gx[k] = -s * data.gx[k];
        gy[k] = -s * data.gy[k];
        gz[k] = -s * data.gz[k];
        lap[k] = -data.g2[k] * c;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn s_orbital_value() {
        // l=0 s GTO, single primitive coeff=1 exp=1, at distance 0 -> norm only.
        let data = AosGtoData {
            kind: AoKind::Sphe,
            num_ao_kernel: 1,
            num_ao: 1,
            n_image: 1,
            molecular_sphe: true,
            precision_policy: crate::PrecisionPolicy::full(),
            value_zone: "ao_eval",
            centers: Array2::zeros((1, 3)),
            angular_momentums: vec![0],
            orbital_indices: vec![0],
            kernel_to_home: vec![0],
            exponents: vec![1.0],
            coefficients: vec![1.0],
            ao_coefficient_scales: Vec::new(),
            poly_x: vec![],
            poly_y: vec![],
            poly_z: vec![],
            magnetic: vec![0],
            normalizations: vec![],
            prim_groups: vec![],
            sphe_angular: vec![],
            sphe_ang_vg: vec![],
            flat_nc: vec![],
            flat_exp: vec![],
            ao_prim_offset: vec![],
            shells: vec![],
            sh_center: vec![],
            sh_off: vec![],
            sh_neg_pz: vec![],
            sh_pnc: vec![],
            sh_pzprim: vec![],
            sh_pzmin: vec![],
            sh_pzval: vec![],
            sh_ao_off: vec![],
            sh_ao_kernel: vec![],
            sh_ao_home: vec![],
            ang_mabs: vec![],
            ang_mc_off: vec![],
            ang_mc: vec![],
            ang_lam_off: vec![],
            ang_lam_c: vec![],
            ang_lam_k: vec![],
            ang_lam_ez: vec![],
            wrap_cell: None,
            wrap_inv: [[0.0; 3]; 3],
            cart_fam: Vec::new(),
            cart_norm_fused_f64: Vec::new(),
            home_kernel_off: Vec::new(),
            home_kernel: Vec::new(),
        }
        .finalize();
        let r = array![[0.0, 0.0, 0.0]];
        let v = compute_aos(&data, &r.view());
        // N = sqrt(2^3 * 1! * 2^1.5 / (2! * sqrt(pi))) ; angular = sqrt(1/4pi)*S_00, S_00=1
        let n = ((2.0_f64.powi(3) * 1.0 * 2.0_f64.powf(1.5)) / (2.0 * std::f64::consts::PI.sqrt()))
            .sqrt();
        let ang = (1.0_f64 / (4.0 * std::f64::consts::PI)).sqrt();
        assert!(
            (v[[0, 0]] - n * ang).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE,
            "{}",
            v[[0, 0]]
        );
    }

    /// Two atoms (s, p, d shells, contracted) x `n_image` images along x with
    /// spacing `a`; far images exercise the exact-underflow shell skip.
    fn vgl_test_data(kind: AoKind, n_image: usize, a: f64) -> AosGtoData {
        vgl_test_data_l(kind, n_image, a, false)
    }

    pub(crate) fn vgl_test_data_l(
        kind: AoKind,
        n_image: usize,
        a: f64,
        high_l: bool,
    ) -> AosGtoData {
        let atoms = [[0.1, -0.2, 0.3], [1.4, 0.9, -0.7]];
        let mut shells: Vec<(i64, &[f64], &[f64])> = vec![
            (0, &[13.0, 2.0, 0.3], &[0.2, 0.5, 0.4]),
            (1, &[3.1, 0.45], &[0.6, 0.5]),
            (2, &[0.8], &[1.0]),
            (0, &[60.0, 0.12], &[0.3, 0.8]),
        ];
        if high_l {
            shells.extend([(3, &[1.1][..], &[1.0][..]), (4, &[0.9][..], &[1.0][..])]);
            if kind == AoKind::Sphe {
                shells.push((5, &[0.7][..], &[1.0][..])); // 21 Cartesian components: Sphe only
            }
        }
        let mut d = AosGtoData::empty();
        d.kind = kind;
        let mut home_aos: Vec<_> = Vec::new();
        for c in atoms {
            for &(l, z, cf) in &shells {
                match kind {
                    AoKind::Cart => {
                        for nx in (0..=l).rev() {
                            for ny in (0..=(l - nx)).rev() {
                                home_aos.push((c, l, 0, [nx, ny, l - nx - ny], z, cf));
                            }
                        }
                    }
                    AoKind::Sphe => {
                        for m in -l..=l {
                            home_aos.push((c, l, m, [0, 0, 0], z, cf));
                        }
                    }
                }
            }
        }
        let na = home_aos.len();
        let mut centers = Vec::new();
        for img in 0..n_image {
            for (h, &(c, l, m, p, z, cf)) in home_aos.iter().enumerate() {
                let k = centers.len() / 3;
                centers.extend_from_slice(&[c[0] + a * img as f64, c[1], c[2]]);
                d.angular_momentums.push(l);
                d.magnetic.push(m);
                d.poly_x.push(p[0]);
                d.poly_y.push(p[1]);
                d.poly_z.push(p[2]);
                d.kernel_to_home.push(h);
                for (zz, cc) in z.iter().zip(cf.iter()) {
                    d.exponents.push(*zz);
                    d.coefficients.push(*cc);
                    d.orbital_indices.push(k);
                }
            }
        }
        d.num_ao = na;
        d.n_image = n_image;
        d.num_ao_kernel = na * n_image;
        d.molecular_sphe = kind == AoKind::Sphe && n_image == 1;
        d.precision_policy = crate::PrecisionPolicy::full();
        d.centers = Array2::from_shape_vec((na * n_image, 3), centers).unwrap();
        d.finalize()
    }

    #[test]
    fn fold_image_axis_f32_rows_is_bitwise_per_element_tree() {
        // The point-vectorized image fold must reproduce the per-(AO, electron)
        // adjacent-tree reduction bit for bit, signed zeros included.
        for n_image in [2, 5, 6, 7] {
            let d = vgl_test_data(AoKind::Cart, n_image, 9.0);
            let n = 5;
            let kernel = Array2::from_shape_fn((d.num_ao_kernel, n), |(k, e)| {
                let t = ((k * 7 + e * 13) % 29) as f64;
                if (k + e) % 11 == 0 {
                    -0.0
                } else {
                    (t - 14.0) * 0.371 * 10f64.powi((k % 9) as i32 - 4)
                }
            });
            let got = fold_image_axis_f32(&d, kernel.clone());
            for home in 0..d.num_ao {
                for e in 0..n {
                    let terms: Vec<f32> = (0..d.num_ao_kernel)
                        .filter(|&k| d.kernel_to_home[k] == home)
                        .map(|k| kernel[[k, e]] as f32)
                        .collect();
                    let want = f64::from(reduce_sum_f32(terms));
                    assert_eq!(
                        got[[home, e]].to_bits(),
                        want.to_bits(),
                        "n_image={n_image} home={home} e={e}"
                    );
                }
            }
        }
    }

    #[test]
    fn cart_f32_shell_reuse_is_bitwise_per_ao() {
        // The mixed Cartesian value kernel with the shared per-shell exp must
        // reproduce the per-AO f32 loop (shells removed) bit for bit.
        // `periodic` marks the leaf periodic (`wrap_cell`), which selects the
        // storage-precision radial factor of issue #718.
        for (n_image, a, periodic) in [
            (1, 0.0, false),
            (6, 9.0, false),
            (6, 20.0, false),
            (6, 9.0, true),
            (6, 20.0, true),
        ] {
            for high_l in [false, true] {
                let mut d = vgl_test_data_l(AoKind::Cart, n_image, a, high_l);
                if periodic {
                    d.wrap_cell = Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]);
                }
                assert!(!d.shells.is_empty());
                let mut per_ao = d.clone();
                per_ao.shells.clear();
                let pts = array![
                    [0.0, 0.0, 0.0],
                    [0.7, -0.3, 0.2],
                    [2.5, 1.1, -1.9],
                    [0.1, -0.2, 0.3],
                    [9.4, 0.8, -0.6],
                    [31.0, -4.0, 2.0]
                ];
                for screen in [false, true] {
                    let s = compute_aos_kernel_screened_f32(&d, &pts.view(), screen);
                    let p = compute_aos_kernel_screened_f32(&per_ao, &pts.view(), screen);
                    assert_eq!(s.shape(), p.shape());
                    for (x, y) in s.iter().zip(p.iter()) {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "n_image={n_image} periodic={periodic} high_l={high_l} screen={screen}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn single_f32_value_is_bitwise_one_point_batch() {
        // The single-position fp32 ao_eval kernel (per-move MCMC route) must give
        // the one-point batch value (kernel + adjacent-tree image fold) bit for bit,
        // molecular and periodic, Sphe and Cart, low and high l.
        let pts = [
            [0.0, 0.0, 0.0],
            [0.7, -0.3, 0.2],
            [2.5, 1.1, -1.9],
            [0.1, -0.2, 0.3],
            [9.4, 0.8, -0.6],
            [31.0, -4.0, 2.0],
        ];
        for kind in [AoKind::Cart, AoKind::Sphe] {
            for (n_image, a, periodic) in [
                (1, 0.0, false),
                (6, 9.0, false),
                (6, 20.0, false),
                (6, 9.0, true),
                (6, 20.0, true),
            ] {
                for high_l in [false, true] {
                    let mut d = vgl_test_data_l(kind, n_image, a, high_l);
                    if periodic {
                        // Periodic leaf (`wrap_cell`): the Cartesian storage-precision
                        // radial factor of issue #718.
                        d.wrap_cell = Some([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]);
                    }
                    for pos in pts {
                        let r = Array2::from_shape_vec((1, 3), pos.to_vec()).unwrap();
                        let want = fold_image_axis_f32(
                            &d,
                            compute_aos_kernel_screened_f32(&d, &r.view(), false),
                        );
                        let mut got = vec![f64::NAN; d.num_ao];
                        compute_aos_single_f32_into(&d, pos, false, &mut got);
                        for (h, (g, w)) in got.iter().zip(want.column(0)).enumerate() {
                            assert_eq!(
                                g.to_bits(),
                                w.to_bits(),
                                "cart={} n_image={n_image} periodic={periodic} high_l={high_l} pos={pos:?} ao={h}: {g} vs {w}", kind == AoKind::Cart
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn single_f32_fast_screen_drops_only_bounded_values() {
        // FAST variant (molecular): every AO is either the unscreened fp32 value
        // bit for bit, or a dropped value below FAST_SCREEN_ABS_BOUND.
        for kind in [AoKind::Cart, AoKind::Sphe] {
            for high_l in [false, true] {
                let d = vgl_test_data_l(kind, 1, 0.0, high_l);
                let mut dropped = 0;
                for pos in [
                    [0.7, -0.3, 0.2],
                    [4.0, 1.1, -1.9],
                    [7.5, 0.8, -0.6],
                    [12.0, -4.0, 2.0],
                ] {
                    let mut exact = vec![0.0; d.num_ao];
                    let mut fast = vec![0.0; d.num_ao];
                    compute_aos_single_f32_into(&d, pos, false, &mut exact);
                    compute_aos_single_f32_into(&d, pos, true, &mut fast);
                    for (h, (f, e)) in fast.iter().zip(&exact).enumerate() {
                        if f.to_bits() != e.to_bits() {
                            dropped += 1;
                            assert!(
                                *f == 0.0 && e.abs() < FAST_SCREEN_ABS_BOUND,
                                "cart={} high_l={high_l} pos={pos:?} ao={h}: fast {f} exact {e}",
                                kind == AoKind::Cart
                            );
                        }
                    }
                }
                assert!(
                    dropped > 0,
                    "cart={} high_l={high_l}: the screen never dropped a non-zero value",
                    kind == AoKind::Cart
                );
            }
        }
    }

    #[test]
    fn single_vgl_shell_matches_per_ao() {
        // Same arithmetic, shell-grouped vs per-AO: reduction-order class.
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        for kind in [AoKind::Cart, AoKind::Sphe] {
            for (n_image, a) in [(1, 0.0), (6, 9.0), (6, 20.0)] {
                let d = vgl_test_data(kind, n_image, a);
                assert!(!d.shells.is_empty());
                let nk = d.num_ao_kernel;
                for pos in [
                    [0.0, 0.0, 0.0],
                    [0.7, -0.3, 0.2],
                    [2.5, 1.1, -1.9],
                    [0.1, -0.2, 0.3],
                ] {
                    let mut r: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; nk]).collect();
                    let mut s: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; nk]).collect();
                    {
                        let [v, x, y, z, l] = &mut r[..] else {
                            unreachable!()
                        };
                        single_vgl_per_ao(&d, pos, v, x, y, z, l);
                    }
                    {
                        let [v, x, y, z, l] = &mut s[..] else {
                            unreachable!()
                        };
                        single_vgl_per_shell(&d, pos, v, x, y, z, l);
                    }
                    for c in 0..5 {
                        for k in 0..nk {
                            let (a0, b0) = (r[c][k], s[c][k]);
                            assert!(
                                (a0 - b0).abs() <= atol + rtol * a0.abs(),
                                "cart={} img={n_image} comp={c} ao={k}: {a0} vs {b0}",
                                kind == AoKind::Cart
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wdot_family_matches_shell_kernel() {
        // Same terms, family vs shell summation order: reduction-order class.
        let atol = crate::central_setting_f64("ATOL_REDUCTION_ORDER");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        for (n_image, a) in [(1, 0.0), (6, 9.0), (6, 20.0)] {
            let mut d = vgl_test_data(AoKind::Cart, n_image, a);
            d.cart_fam = build_cart_families(&d);
            assert!(!d.cart_fam.is_empty());
            let nfam_members: usize = d.cart_fam.iter().map(|f| f.cx.len()).sum();
            assert_eq!(nfam_members, d.shells.len());
            let w: Vec<f64> = (0..d.num_ao)
                .map(|k| ((k * 37 % 11) as f64 - 5.0) * 0.13)
                .collect();
            let mut pts = Array2::<f64>::zeros((12, 3));
            for v in 0..12 {
                let t = v as f64 * 0.7;
                pts[[v, 0]] = 0.4 + 2.0 * t.cos() + 3.0 * (v % 3) as f64;
                pts[[v, 1]] = -0.3 + 1.5 * t.sin();
                pts[[v, 2]] = 0.2 + 0.4 * t;
            }
            let (mut o1, mut o2) = (vec![0.0; 12], vec![0.0; 12]);
            dot_aos_multi_cutoff_cart_soa(&d, &pts.view(), &w, &mut o1);
            dot_aos_multi_cutoff_cart_fam(&d, &pts.view(), &w, &mut o2);
            for v in 0..12 {
                assert!(
                    (o1[v] - o2[v]).abs() <= atol + rtol * o1[v].abs(),
                    "img={n_image} v={v}: {} vs {}",
                    o1[v],
                    o2[v]
                );
            }
        }
    }

    #[test]
    fn kernel_sphe_soa_matches_shell_eval_bitwise() {
        for (n_image, a) in [(1, 0.0), (6, 9.0)] {
            let d = vgl_test_data_l(AoKind::Sphe, n_image, a, true);
            assert!(kernel_sphe_soa_ok(&d));
            let n = 150;
            let mut r = Array2::<f64>::zeros((n, 3));
            for e in 0..n {
                let t = e as f64 * 0.37;
                r[[e, 0]] = 0.5 + 2.5 * t.cos() + (e % 7) as f64;
                r[[e, 1]] = -0.2 + 1.7 * t.sin();
                r[[e, 2]] = 0.3 + 0.05 * e as f64;
            }
            for periodic in [false, true] {
                let mut got = Array2::<f64>::zeros((d.num_ao_kernel, n));
                kernel_sphe_soa(&d, &r.view(), periodic, &mut got);
                for shell in &d.shells {
                    for e in 0..n {
                        let (x, y, z) = (
                            r[[e, 0]] - shell.center[0],
                            r[[e, 1]] - shell.center[1],
                            r[[e, 2]] - shell.center[2],
                        );
                        let r2 = x * x + y * y + z * z;
                        let dead = periodic && shell.pz_min * r2 > LEPSBAS;
                        let mut rad = 0.0_f64;
                        for i in 0..shell.prim_z.len() {
                            rad += shell.prim_nc[i] * exp_neg_fast_scalar(-shell.prim_z[i] * r2);
                        }
                        for &ao in &shell.aos {
                            let want = if dead {
                                0.0
                            } else {
                                rad * d.sphe_angular[ao].eval(x, y, z, r2)
                            };
                            assert_eq!(
                                want.to_bits(),
                                got[[ao, e]].to_bits(),
                                "ao={ao} e={e} l={}",
                                d.angular_momentums[ao]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn single_fold_cutoff_drops_only_sub_bound_values() {
        // FAST periodic sweep value: kept shells bit-equal to the unscreened
        // shell arithmetic, every dropped AO value below FAST_SCREEN_ABS_BOUND.
        for kind in [AoKind::Sphe, AoKind::Cart] {
            for (n_image, a) in [(1, 0.0), (6, 9.0)] {
                let d = vgl_test_data_l(kind, n_image, a, true);
                let mut got = vec![0.0; d.num_ao];
                let mut exact_fold = vec![0.0; d.num_ao];
                let (mut n_drop, mut n_keep) = (0usize, 0usize);
                for e in 0..200 {
                    let t = e as f64 * 0.37;
                    let pos = [
                        0.5 + 4.0 * t.cos() + (e % 11) as f64 * 3.0,
                        -0.2 + 2.7 * t.sin(),
                        0.3 + 0.03 * e as f64,
                    ];
                    compute_aos_single_fold_cutoff_into(&d, pos, &mut got);
                    let mut want = vec![0.0; d.num_ao];
                    for shell in &d.shells {
                        let (x, y, z) = (
                            pos[0] - shell.center[0],
                            pos[1] - shell.center[1],
                            pos[2] - shell.center[2],
                        );
                        let r2 = x * x + y * y + z * z;
                        let dead = shell.pz_val * r2 > LEPSBAS;
                        for &ao in &shell.aos {
                            let v = match kind {
                                AoKind::Sphe => {
                                    let mut rad = 0.0_f64;
                                    for i in 0..shell.prim_z.len() {
                                        rad += shell.prim_nc[i]
                                            * exp_neg_fast_scalar(-shell.prim_z[i] * r2);
                                    }
                                    rad * d.sphe_angular[ao].eval(x, y, z, r2)
                                }
                                AoKind::Cart => {
                                    let (s, t) = (d.ao_prim_offset[ao], d.ao_prim_offset[ao + 1]);
                                    let mut rad = 0.0_f64;
                                    for p in s..t {
                                        rad +=
                                            d.flat_nc[p] * exp_neg_fast_scalar(-d.flat_exp[p] * r2);
                                    }
                                    rad * x.powi(d.poly_x[ao] as i32)
                                        * y.powi(d.poly_y[ao] as i32)
                                        * z.powi(d.poly_z[ao] as i32)
                                }
                            };
                            if dead {
                                n_drop += 1;
                                assert!(
                                    v.abs() < FAST_SCREEN_ABS_BOUND,
                                    "dropped |{v}| >= bound (sphe {}, ao {ao})",
                                    kind == AoKind::Sphe
                                );
                            } else {
                                n_keep += 1;
                                want[d.kernel_to_home[ao]] += v;
                            }
                        }
                    }
                    for h in 0..d.num_ao {
                        assert_eq!(
                            want[h].to_bits(),
                            got[h].to_bits(),
                            "sphe {} n_image {n_image} home {h}",
                            kind == AoKind::Sphe
                        );
                    }
                    if kind == AoKind::Sphe && n_image > 1 {
                        compute_aos_single_fold_into(&d, pos, &mut exact_fold);
                        for h in 0..d.num_ao {
                            // At most n_image kernel AOs fold into one home AO, each
                            // dropped below the bound; the kept part differs by the
                            // fast exp / summation order (reduction-order class).
                            let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
                            assert!(
                                (exact_fold[h] - got[h]).abs()
                                    <= n_image as f64 * FAST_SCREEN_ABS_BOUND
                                        + rtol * exact_fold[h].abs()
                            );
                        }
                    }
                }
                assert!(n_keep > 0);
                if n_image > 1 {
                    assert!(n_drop > 0, "test geometry must exercise the screen");
                }
            }
        }
    }

    #[test]
    fn mesh_dot_prim_screen_changes_each_ao_below_bound() {
        // Kept part: fast exp vs libm and summation order (reduction-order class).
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        // ECP mesh contraction with the shell + primitive value screens: with a
        // unit weight vector the output is one screened (image-folded) AO value,
        // which differs from the exact unscreened AO by less than
        // FAST_SCREEN_ABS_BOUND per kernel AO. Non-vacuity: primitives of kept
        // shells are dropped somewhere.
        for kind in [AoKind::Sphe, AoKind::Cart] {
            for (n_image, a) in [(1, 0.0), (6, 9.0)] {
                let d = vgl_test_data_l(kind, n_image, a, true);
                let mut n_prim_drop = 0usize;
                for m in 0..40 {
                    let t = m as f64 * 0.53;
                    let (cx, cy, cz) = (
                        0.2 + 3.5 * t.cos() + (m % 5) as f64 * 1.7,
                        -0.4 + 2.2 * t.sin(),
                        0.1 + 0.06 * m as f64,
                    );
                    let rad = 0.3 + 0.05 * (m % 9) as f64;
                    let mut pts = Array2::<f64>::zeros((12, 3));
                    for v in 0..12 {
                        let (th, ph) = (0.4 + 0.23 * v as f64, 1.1 * v as f64);
                        pts[[v, 0]] = cx + rad * th.sin() * ph.cos();
                        pts[[v, 1]] = cy + rad * th.sin() * ph.sin();
                        pts[[v, 2]] = cz + rad * th.cos();
                    }
                    let exact = compute_aos(&d, &pts.view());
                    for shell in &d.shells {
                        for v in 0..12 {
                            let r2: f64 = (0..3)
                                .map(|k| (pts[[v, k]] - shell.center[k]).powi(2))
                                .sum();
                            if shell.pz_val * r2 <= LEPSBAS {
                                n_prim_drop +=
                                    shell.prim_val.iter().filter(|&&p| p * r2 > LEPSBAS).count();
                            }
                        }
                    }
                    let mut out = vec![0.0; 12];
                    for h in 0..d.num_ao {
                        let mut w = vec![0.0; d.num_ao];
                        w[h] = 1.0;
                        match kind {
                            AoKind::Sphe => {
                                dot_aos_multi_cutoff_sphe(&d, &pts.view(), &w, &mut out)
                            }
                            AoKind::Cart => {
                                dot_aos_multi_cutoff_cart(&d, &pts.view(), &w, &mut out)
                            }
                        }
                        for v in 0..12 {
                            let want = exact[[h, v]]; // compute_aos is image-folded
                            let tol = n_image as f64 * FAST_SCREEN_ABS_BOUND + rtol * want.abs();
                            assert!(
                                (out[v] - want).abs() <= tol,
                                "sphe {} n_image {n_image} mesh {m} home {h} v {v}: {} vs {want}",
                                kind == AoKind::Sphe,
                                out[v]
                            );
                        }
                    }
                }
                assert!(
                    n_prim_drop > 0,
                    "test geometry must exercise the primitive screen"
                );
            }
        }
    }

    #[test]
    fn bloch_mesh_dot_matches_exact_folded_kernel() {
        // The complex Bloch mesh dot drops, per (image) kernel AO, only values
        // below FAST_AO_SCREEN_ABS_BOUND (pz_val shell + prim_val primitive
        // screens); the kept part differs from the exact libm kernel by the fast
        // exp and the summation order (reduction-order class, relative to the
        // absolute-term sum). Reference: exact unscreened kernel, Bloch-folded
        // with the same phases, contracted with the same complex weights.
        let bound = crate::central_setting_f64("FAST_AO_SCREEN_ABS_BOUND");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        for kind in [AoKind::Sphe, AoKind::Cart] {
            for (n_image, a) in [(1, 0.0), (6, 9.0)] {
                let d = vgl_test_data_l(kind, n_image, a, true);
                assert!(bloch_mesh_dot_ok(&d));
                let phase: Vec<numpy::Complex64> = (0..d.num_ao_kernel)
                    .map(|k| numpy::Complex64::new(0.0, 0.37 * k as f64 + 0.11).exp())
                    .collect();
                let w_re: Vec<f64> = (0..d.num_ao)
                    .map(|k| ((k * 37 % 11) as f64 - 5.0) * 0.13)
                    .collect();
                let w_im: Vec<f64> = (0..d.num_ao)
                    .map(|k| ((k * 23 % 7) as f64 - 3.0) * 0.21)
                    .collect();
                let (mut n_dropped, mut n_live) = (0usize, 0usize);
                for (m, nv) in [(0usize, 12usize), (1, 40), (2, 12), (3, 26)] {
                    let t = m as f64 * 0.53;
                    let (cx, cy, cz) = (
                        0.2 + 3.5 * t.cos() + (m % 5) as f64 * 1.7,
                        -0.4 + 2.2 * t.sin(),
                        0.1 + 0.06 * m as f64,
                    );
                    let rad = 0.3 + 0.4 * m as f64;
                    let pts: Vec<[f64; 3]> = (0..nv)
                        .map(|v| {
                            let (th, ph) = (0.4 + 0.23 * v as f64, 1.1 * v as f64);
                            [
                                cx + rad * th.sin() * ph.cos(),
                                cy + rad * th.sin() * ph.sin(),
                                cz + rad * th.cos(),
                            ]
                        })
                        .collect();
                    let pa = Array2::from_shape_fn((nv, 3), |(v, k)| pts[v][k]);
                    let kern = compute_aos_kernel(&d, &pa.view());
                    let (mut o_re, mut o_im) = (vec![0.0; nv], vec![0.0; nv]);
                    dot_aos_multi_cutoff_bloch(
                        &d, &pts, &w_re, &w_im, &phase, &mut o_re, &mut o_im,
                    );
                    for v in 0..nv {
                        let (mut r_re, mut r_im, mut wsum, mut scale) = (0.0, 0.0, 0.0, 0.0);
                        for k in 0..d.num_ao_kernel {
                            let h = d.kernel_to_home[k];
                            let wc = numpy::Complex64::new(w_re[h], w_im[h]) * phase[k];
                            let term = wc * kern[[k, v]];
                            r_re += term.re;
                            r_im += term.im;
                            wsum += wc.norm();
                            scale += term.norm();
                        }
                        let tol = bound * wsum + rtol * scale;
                        assert!(
                            (o_re[v] - r_re).abs() <= tol && (o_im[v] - r_im).abs() <= tol,
                            "{} n_image {n_image} mesh {m} v {v}: ({}, {}) vs ({r_re}, {r_im}) tol {tol}",
                            kind == AoKind::Sphe,
                            o_re[v],
                            o_im[v]
                        );
                        if scale > 1.0e-6 {
                            n_live += 1;
                        }
                        for shell in &d.shells {
                            let r2: f64 =
                                (0..3).map(|c| (pts[v][c] - shell.center[c]).powi(2)).sum();
                            if shell.pz_val * r2 > LEPSBAS
                                || shell.prim_val.iter().any(|&p| p * r2 > LEPSBAS)
                            {
                                n_dropped += 1;
                            }
                        }
                    }
                }
                // Non-vacuity: the screens drop terms and the contraction is not ~0.
                assert!(n_dropped > 0, "geometry must exercise the value screens");
                assert!(
                    n_live >= 45,
                    "contraction must be non-negligible at most of the 90 points ({n_live})"
                );
            }
        }
    }

    #[test]
    fn kernel_cart_soa_matches_shell_loop_bitwise() {
        for (n_image, a) in [(1, 0.0), (6, 9.0)] {
            let d = vgl_test_data_l(AoKind::Cart, n_image, a, true);
            assert!(kernel_cart_soa_ok(&d));
            let n = 150;
            let mut r = Array2::<f64>::zeros((n, 3));
            for e in 0..n {
                let t = e as f64 * 0.37;
                r[[e, 0]] = 0.5 + 2.5 * t.cos() + (e % 7) as f64;
                r[[e, 1]] = -0.2 + 1.7 * t.sin();
                r[[e, 2]] = 0.3 + 0.05 * e as f64;
            }
            for periodic in [false, true] {
                let mut got = Array2::<f64>::zeros((d.num_ao_kernel, n));
                kernel_cart_soa(&d, &r.view(), periodic, &mut got);
                let mut want = Array2::<f64>::zeros((d.num_ao_kernel, n));
                for shell in &d.shells {
                    for e in 0..n {
                        let (x, y, z) = (
                            r[[e, 0]] - shell.center[0],
                            r[[e, 1]] - shell.center[1],
                            r[[e, 2]] - shell.center[2],
                        );
                        let r2 = x * x + y * y + z * z;
                        if periodic && shell.pz_min * r2 > LEPSBAS {
                            continue;
                        }
                        for q in 0..shell.prim_z.len() {
                            let ev = exp_neg_fast_scalar(-shell.prim_z[q] * r2);
                            for &ao in &shell.aos {
                                want[[ao, e]] += d.flat_nc[d.ao_prim_offset[ao] + q] * ev;
                            }
                        }
                        for &ao in &shell.aos {
                            want[[ao, e]] *= x.powi(d.poly_x[ao] as i32)
                                * y.powi(d.poly_y[ao] as i32)
                                * z.powi(d.poly_z[ao] as i32);
                        }
                    }
                }
                for k in 0..d.num_ao_kernel {
                    for e in 0..n {
                        assert_eq!(
                            want[[k, e]].to_bits(),
                            got[[k, e]].to_bits(),
                            "ao={k} e={e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wdot_sphe_soa_matches_generic_bitwise() {
        for (n_image, a) in [(1, 0.0), (6, 9.0), (6, 20.0)] {
            let d = vgl_test_data(AoKind::Sphe, n_image, a);
            let w: Vec<f64> = (0..d.num_ao)
                .map(|k| ((k * 37 % 11) as f64 - 5.0) * 0.13)
                .collect();
            for nv in [6usize, 12, 32] {
                let mut pts = Array2::<f64>::zeros((nv, 3));
                for v in 0..nv {
                    let t = v as f64 * 0.7;
                    pts[[v, 0]] = 0.4 + 2.0 * t.cos() + 3.0 * (v % 3) as f64;
                    pts[[v, 1]] = -0.3 + 1.5 * t.sin();
                    pts[[v, 2]] = 0.2 + 0.4 * t;
                }
                let (mut o1, mut o2) = (vec![0.0; nv], vec![7.0; nv]);
                dot_aos_multi_cutoff_sphe_generic(&d, &pts.view(), &w, &mut o1);
                dot_aos_multi_cutoff_sphe(&d, &pts.view(), &w, &mut o2);
                for v in 0..nv {
                    assert_eq!(
                        o1[v].to_bits(),
                        o2[v].to_bits(),
                        "img={n_image} nv={nv} v={v}: {} vs {}",
                        o1[v],
                        o2[v]
                    );
                }
            }
        }
    }

    #[test]
    fn solid_harmonic_pz() {
        // l=1, m=0 -> S_10 = z
        let v = compute_s_l_m(1, 0, 0.3, -0.4, 0.5, (0.3f64.powi(2) + 0.16 + 0.25).sqrt());
        assert!(
            (v - 0.5).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE,
            "{v}"
        );
    }
}

#[cfg(test)]
mod simd_trig_tests {
    use super::{cos_pd_into, sincos_pd_into};

    #[test]
    fn cos_and_sincos_match_libm() {
        // Cover small + large args (PW theta can reach ~200) and non-multiple-of-8 tails.
        let mut src = Vec::new();
        let mut x = -210.0_f64;
        while x < 210.0 {
            src.push(x);
            x += 0.013_37;
        }
        // a few exact-ish quadrant boundaries
        for k in -5..=5 {
            src.push(k as f64 * std::f64::consts::FRAC_PI_2);
        }
        let n = src.len();
        let mut c = vec![0.0; n];
        let mut s = vec![0.0; n];
        let mut c2 = vec![0.0; n];
        cos_pd_into(&src, &mut c);
        sincos_pd_into(&src, &mut s, &mut c2);
        for i in 0..n {
            let (rs, rc) = src[i].sin_cos();
            assert!(
                (c[i] - rc).abs() <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE,
                "cos {} : {} vs {}",
                src[i],
                c[i],
                rc
            );
            assert!(
                (c2[i] - rc).abs() <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE,
                "sincos.c {} : {} vs {}",
                src[i],
                c2[i],
                rc
            );
            assert!(
                (s[i] - rs).abs() <= crate::setting_consts::ATOL_RS_UNIT_NEAR_MACHINE,
                "sincos.s {} : {} vs {}",
                src[i],
                s[i],
                rs
            );
        }
    }
}
