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

//! 1:1 Rust port of the autograd-free Jastrow *value* functions
//! (`jqmc/jastrow_factor.py`), molecular path.
//!
//!   - `compute_Jastrow_one_body`  (py:287)  -> [`compute_jastrow_one_body`]
//!   - `compute_Jastrow_two_body`  (py:917)  -> [`compute_jastrow_two_body`]
//!   - `compute_Jastrow_three_body`(py:1361) -> [`compute_jastrow_three_body_value`]
//!
//! NN-J3 and DTN-NN are excluded; this module contains the analytic polynomial
//! J1/J2/J3 kernels. The J3 evaluator here
//! takes the already-evaluated AO/MO matrices (`aos_up`, `aos_dn`) -- porting
//! the AO/MO kernel itself belongs to the atomic-orbital module. The bilinear J3
//! algebra over those matrices is implemented here.

use crate::data::{JastrowData, JastrowOneBodyData, JastrowTwoBodyData};
use crate::determinant::{matmul, orb_value_grad_lap, OrbData};
use numpy::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

/// `1 - exp(-y)` (`-expm1(-y)`) without the small-`y` subtraction cancellation
/// of the exp-form J1/J2 kernels. `exp_m1` is used only for `y < ln 2` (kept
/// out of line: at physical separations that branch is rare); for `y >= ln 2`,
/// `exp(-y) <= 1/2`, so `1 - exp(-y)` has no cancellation and the common path
/// keeps the cheaper `exp` (bit-identical to a plain `1 - exp(-y)` there). XLA
/// evaluates the same function as `-jnp.expm1(-y)` (`jqmc/jastrow_factor.py`).
#[inline]
fn one_minus_exp_neg(y: f64) -> f64 {
    if y < std::f64::consts::LN_2 {
        neg_expm1_neg_cold(y)
    } else {
        1.0 - (-y).exp()
    }
}

#[cold]
#[inline(never)]
fn neg_expm1_neg_cold(y: f64) -> f64 {
    -(-y).exp_m1()
}

/// f32 counterpart of [`one_minus_exp_neg`] for the mixed-precision kernels.
///
/// No `exp_m1` call, so the f32 loops keep their cost: below `|y| = 1/4` the
/// degree-6 Taylor series `y (1 - y/2 (1 - y/3 (... (1 - y/6))))` (first
/// omitted term `y^6 / 7! < 5e-8` relative, below half an f32 ulp), above it
/// `1 - exp(-y)` (a few ulp at most, since `exp(-1/4) < 0.78`).
#[inline]
fn one_minus_exp_neg_f32(y: f32) -> f32 {
    if y.abs() < 0.25 {
        y * (1.0
            - y * (1.0 / 2.0)
                * (1.0
                    - y * (1.0 / 3.0)
                        * (1.0
                            - y * (1.0 / 4.0) * (1.0 - y * (1.0 / 5.0) * (1.0 - y * (1.0 / 6.0))))))
    } else {
        1.0 - (-y).exp()
    }
}

/// Multiply independently evaluated Jastrow ratio components at the
/// `jastrow_ratio` zone boundary.  The Python/XLA route combines J1/J2, J3 and
/// DTN after converting each component to the zone dtype, so mixed precision
/// requires this product itself to be rounded in f32 as well.
#[inline]
pub(crate) fn multiply_ratio_components(data: &JastrowData, a: f64, b: f64) -> f64 {
    if data.precision_policy.is_f32("jastrow_ratio") {
        ((a as f32) * (b as f32)) as f64
    } else {
        a * b
    }
}

/// `compute_grads_and_laplacian_Jastrow_three_body` (jastrow_factor.py),
/// molecular analytic. Per-electron `gradJ3` `(N,3)` and `grad^2J3` `(N,)`.
/// `g[o,n] = j1[o] + Sum_{m>n} (J3*A)[o,m] + Sum_{m<n} (J3T*A)[o,m] + Sum_m (J3*A_other)[o,m]`,
/// then `gradJ3[n] = Sum_o g[o,n]*gradorb[o,n]`, `grad^2J3[n] = Sum_o g[o,n]*grad^2orb[o,n]`.
///
/// Periodicity contract matching jqmc `compute_AOs`:
/// the J3 orbital is a batch pGTO whose static image tiling is enumerated relative
/// to the home cell, so a walker roamed a few lattice vectors out underflows the
/// AO -> the J3 drift/laplacian silently collapse (measured ~1e-2 in the far-config
/// periodic j123 HF force through `jastrow_grads`). Home-cell-wrap the electron
/// coordinates HERE (kernel-level, one convention for every consumer: e_L kinetic,
/// LRDMC drift/kinetic, VMC force cotangents, dT/dc). grad/lap are translation-
/// invariant; the wrap is bit-exact identity for in-cell electrons and a no-op
/// (`None`) for molecular / PW / B-spline leaves.
pub fn compute_grads_and_laplacian_jastrow_three_body(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    compute_grads_and_laplacian_jastrow_three_body_impl(j3_orb, j_matrix, r_up, r_dn, false)
}

/// Production-fast J3 kinetic sibling: molecular GTO leaves use Cartesian /
/// spherical shell-reused AO grad+lap with the LEPSBAS distance floor and SIMD
/// exponentials. Exact/parity/GFMC callers retain the public function above.
pub fn compute_grads_and_laplacian_jastrow_three_body_fast(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    compute_grads_and_laplacian_jastrow_three_body_impl(j3_orb, j_matrix, r_up, r_dn, true)
}

pub struct J3GradLapCache {
    // Electron-major derivative rows. The ratio state keeps J*A in its native
    // (orbital, electron) layout, but each kinetic contraction consumes all
    // orbitals of one electron at once.
    up_x: Array2<f64>,
    up_y: Array2<f64>,
    up_z: Array2<f64>,
    up_lap: Array2<f64>,
    dn_x: Array2<f64>,
    dn_y: Array2<f64>,
    dn_z: Array2<f64>,
    dn_lap: Array2<f64>,
    /// Final per-electron contractions.  TurboRVB keeps these as the last four
    /// `tabpip` columns; keeping the same level of state avoids re-reading both
    /// the q table and all derivative tables on every LRDMC hop.
    output: Option<J3GradLapOutput>,
    /// Orbital-major `(n_orb, ne)` mirror of the eight derivative tables,
    /// built lazily by [`advance_j3_grad_lap_cache_with_q_delta`] and kept in
    /// step by every real row writer. It lets the per-hop advance run the
    /// SAME per-electron sequential sums vectorised across electrons
    /// (IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001); `None` whenever a writer that
    /// does not maintain it touched the tables.
    shadow: Option<J3GradShadow>,
}

/// Orbital-major copies `[x, y, z, lap]` per spin of a [`J3GradLapCache`].
struct J3GradShadow {
    up: [Array2<f64>; 4],
    dn: [Array2<f64>; 4],
}

impl J3GradShadow {
    fn build(cache: &J3GradLapCache) -> Self {
        let t = |m: &Array2<f64>| m.t().as_standard_layout().to_owned();
        J3GradShadow {
            up: [
                t(&cache.up_x),
                t(&cache.up_y),
                t(&cache.up_z),
                t(&cache.up_lap),
            ],
            dn: [
                t(&cache.dn_x),
                t(&cache.dn_y),
                t(&cache.dn_z),
                t(&cache.dn_lap),
            ],
        }
    }

    /// Mirror one electron's freshly written derivative rows (`cols[k]` is the
    /// `(n_orb, 1)` column `[x, y, z, lap][k]`).
    fn set_electron(&mut self, is_up: bool, electron: usize, cols: [&Array2<f64>; 4]) {
        let tables = if is_up { &mut self.up } else { &mut self.dn };
        for (dst, src) in tables.iter_mut().zip(cols) {
            for a in 0..dst.nrows() {
                dst[[a, electron]] = src[[a, 0]];
            }
        }
    }
}

struct J3GradLapOutput {
    grad_up: Array2<f64>,
    grad_dn: Array2<f64>,
    lap_up: Array1<f64>,
    lap_dn: Array1<f64>,
}

/// Derivative half of TurboRVB's Jastrow table. Orbital values and `J3*A`
/// already live in [`J3RatioState`]; only derivative AO columns are additional.
pub fn init_j3_grad_lap_cache(
    j3_orb: &OrbData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> J3GradLapCache {
    let eval =
        |r: &ArrayView2<f64>| crate::atomic_orbital::compute_aos_grad_lap_fast_cut(&j3_orb.aos, r);
    let (up_x, up_y, up_z, up_lap) = eval(r_up);
    let (dn_x, dn_y, dn_z, dn_lap) = eval(r_dn);
    J3GradLapCache {
        up_x: up_x.t().as_standard_layout().to_owned(),
        up_y: up_y.t().as_standard_layout().to_owned(),
        up_z: up_z.t().as_standard_layout().to_owned(),
        up_lap: up_lap.t().as_standard_layout().to_owned(),
        dn_x: dn_x.t().as_standard_layout().to_owned(),
        dn_y: dn_y.t().as_standard_layout().to_owned(),
        dn_z: dn_z.t().as_standard_layout().to_owned(),
        dn_lap: dn_lap.t().as_standard_layout().to_owned(),
        output: None,
        shadow: None,
    }
}

pub fn update_j3_grad_lap_cache(
    j3_orb: &OrbData,
    cache: &mut J3GradLapCache,
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) {
    let point = Array2::from_shape_vec((1, 3), pos.to_vec()).expect("one J3 derivative point");
    let (gx, gy, gz, lap) =
        crate::atomic_orbital::compute_aos_grad_lap_fast_cut(&j3_orb.aos, &point.view());
    let assign = |dst: &mut Array2<f64>, src: &Array2<f64>| {
        for a in 0..dst.ncols() {
            dst[[electron, a]] = src[[a, 0]];
        }
    };
    if is_up {
        assign(&mut cache.up_x, &gx);
        assign(&mut cache.up_y, &gy);
        assign(&mut cache.up_z, &gz);
        assign(&mut cache.up_lap, &lap);
    } else {
        assign(&mut cache.dn_x, &gx);
        assign(&mut cache.dn_y, &gy);
        assign(&mut cache.dn_z, &gz);
        assign(&mut cache.dn_lap, &lap);
    }
    if let Some(sh) = cache.shadow.as_mut() {
        sh.set_electron(is_up, electron, [&gx, &gy, &gz, &lap]);
    }
}

pub fn contract_j3_grad_lap_cache(
    j_matrix: &ArrayView2<f64>,
    state: &J3RatioState,
    cache: &J3GradLapCache,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_orb = j_matrix.nrows();
    let contract_spin = |is_up: bool| {
        let (w_same, cross_same, cross_other, gx, gy, gz, lap) = if is_up {
            (
                &state.w_up,
                &state.cross_up,
                &state.cross_dn,
                &cache.up_x,
                &cache.up_y,
                &cache.up_z,
                &cache.up_lap,
            )
        } else {
            (
                &state.w_dn,
                &state.cross_dn,
                &state.cross_up,
                &cache.dn_x,
                &cache.dn_y,
                &cache.dn_z,
                &cache.dn_lap,
            )
        };
        let ne = w_same.ncols();
        let mut grad = Array2::<f64>::zeros((ne, 3));
        let mut lp = Array1::<f64>::zeros(ne);
        let mut q = vec![0.0_f64; n_orb];
        for e in 0..ne {
            for o in 0..n_orb {
                q[o] = j_matrix[[o, n_orb]] + cross_other[o] + cross_same[o] - w_same[[o, e]];
            }
            let qr = numpy::ndarray::ArrayView1::from(&q);
            grad[[e, 0]] = qr.dot(&gx.row(e));
            grad[[e, 1]] = qr.dot(&gy.row(e));
            grad[[e, 2]] = qr.dot(&gz.row(e));
            lp[e] = qr.dot(&lap.row(e));
        }
        (grad, lp)
    };
    let (grad_up, lap_up) = contract_spin(true);
    let (grad_dn, lap_dn) = contract_spin(false);
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Contract J3 derivatives with the per-electron q rows already constructed
/// for the fixed-mesh J3 ratios.  The mesh ratio and continuum kinetic terms
/// use the identical
/// `j1 + cross_other + cross_same - W_same[:,electron]` vector, so rebuilding
/// it in both consumers is redundant.
pub fn contract_j3_grad_lap_cache_with_q(
    cache: &mut J3GradLapCache,
    q_up: &ArrayView2<f64>,
    q_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    if let Some(out) = cache.output.as_ref() {
        return (
            out.grad_up.clone(),
            out.grad_dn.clone(),
            out.lap_up.clone(),
            out.lap_dn.clone(),
        );
    }
    let contract_spin = |q: &ArrayView2<f64>,
                         gx: &Array2<f64>,
                         gy: &Array2<f64>,
                         gz: &Array2<f64>,
                         lap: &Array2<f64>| {
        debug_assert_eq!(q.raw_dim(), gx.raw_dim());
        let ne = q.nrows();
        let mut grad = Array2::<f64>::zeros((ne, 3));
        let mut lp = Array1::<f64>::zeros(ne);
        for e in 0..ne {
            let qr = q.row(e);
            grad[[e, 0]] = qr.dot(&gx.row(e));
            grad[[e, 1]] = qr.dot(&gy.row(e));
            grad[[e, 2]] = qr.dot(&gz.row(e));
            lp[e] = qr.dot(&lap.row(e));
        }
        (grad, lp)
    };
    let (grad_up, lap_up) =
        contract_spin(q_up, &cache.up_x, &cache.up_y, &cache.up_z, &cache.up_lap);
    let (grad_dn, lap_dn) =
        contract_spin(q_dn, &cache.dn_x, &cache.dn_y, &cache.dn_z, &cache.dn_lap);
    cache.output = Some(J3GradLapOutput {
        grad_up: grad_up.clone(),
        grad_dn: grad_dn.clone(),
        lap_up: lap_up.clone(),
        lap_dn: lap_dn.clone(),
    });
    (grad_up, grad_dn, lap_up, lap_dn)
}

/// Unchanged-electron half of [`advance_j3_grad_lap_cache_with_q_delta`]:
/// every electron except (`is_up`, `electron`) adds `sum_a T[i, a] * d[a]` to
/// each of its four final contractions.
///
/// Exactness (IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001): each sum starts at 0.0
/// and accumulates in ascending `a` with a separately rounded product -- the
/// order of a per-row scalar loop over the electron-major tables. The
/// orbital-major mirror runs those same per-electron chains side by side, so
/// every sum is bit-identical while the loop vectorises across electrons.
pub(crate) fn advance_j3_grad_outputs_unchanged(
    cache: &mut J3GradLapCache,
    d: &[f64],
    is_up: bool,
    electron: usize,
) {
    let advance_unchanged = |moved_spin: bool,
                             tables: &[Array2<f64>; 4],
                             grad_out: &mut Array2<f64>,
                             lap_out: &mut Array1<f64>| {
        let ne = tables[0].ncols();
        let flat: [&[f64]; 4] = std::array::from_fn(|c| {
            tables[c]
                .as_slice()
                .expect("J3 derivative mirror contiguous")
        });
        let mut acc = vec![0.0_f64; 4 * ne];
        for a in 0..d.len() {
            let da = d[a];
            for c in 0..4 {
                let row = &flat[c][a * ne..(a + 1) * ne];
                for (s, &t) in acc[c * ne..(c + 1) * ne].iter_mut().zip(row) {
                    *s += t * da;
                }
            }
        }
        for i in 0..ne {
            if moved_spin && i == electron {
                continue;
            }
            grad_out[[i, 0]] += acc[i];
            grad_out[[i, 1]] += acc[ne + i];
            grad_out[[i, 2]] += acc[2 * ne + i];
            lap_out[i] += acc[3 * ne + i];
        }
    };
    if cache.shadow.is_none() {
        cache.shadow = Some(J3GradShadow::build(cache));
    }
    let sh = cache
        .shadow
        .as_ref()
        .expect("J3 derivative mirror built above");
    let out = cache
        .output
        .as_mut()
        .expect("J3 derivative output initialized before interval hop");
    advance_unchanged(is_up, &sh.up, &mut out.grad_up, &mut out.lap_up);
    advance_unchanged(!is_up, &sh.dn, &mut out.grad_dn, &mut out.lap_dn);
}

/// Advance TurboRVB-style final J3 derivative columns after one accepted hop.
///
/// For every unchanged electron the q row receives the same
/// `d = J3 * (AO_new - AO_old)`, hence each final derivative column receives
/// one dot product of its existing derivative row with `d`.  The moved
/// electron's q row is invariant; only its derivative AO row is replaced and
/// its four final contractions are rebuilt.  This is exactly the algebra used
/// by the full contraction, with O(N*AO) derivative reads but no O(N*AO) q
/// reads and a single hot common vector.
#[allow(clippy::too_many_arguments)]
pub fn advance_j3_grad_lap_cache_with_q_delta(
    j3_orb: &OrbData,
    cache: &mut J3GradLapCache,
    q_up: &ArrayView2<f64>,
    q_dn: &ArrayView2<f64>,
    d: &[f64],
    is_up: bool,
    electron: usize,
    pos: [f64; 3],
) {
    crate::timed!(
        "gfmc.adv_j3_grad_adv",
        advance_j3_grad_outputs_unchanged(cache, d, is_up, electron)
    );

    crate::timed!(
        "gfmc.adv_j3_grad_ao",
        update_j3_grad_lap_cache(j3_orb, cache, is_up, electron, pos)
    );
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut z = 0.0_f64;
    let mut l = 0.0_f64;
    if is_up {
        let qr = q_up.row(electron);
        let xr = cache.up_x.row(electron);
        let yr = cache.up_y.row(electron);
        let zr = cache.up_z.row(electron);
        let lr = cache.up_lap.row(electron);
        for a in 0..d.len() {
            let qa = qr[a];
            x += xr[a] * qa;
            y += yr[a] * qa;
            z += zr[a] * qa;
            l += lr[a] * qa;
        }
        let out = cache.output.as_mut().expect("J3 derivative output live");
        out.grad_up[[electron, 0]] = x;
        out.grad_up[[electron, 1]] = y;
        out.grad_up[[electron, 2]] = z;
        out.lap_up[electron] = l;
    } else {
        let qr = q_dn.row(electron);
        let xr = cache.dn_x.row(electron);
        let yr = cache.dn_y.row(electron);
        let zr = cache.dn_z.row(electron);
        let lr = cache.dn_lap.row(electron);
        for a in 0..d.len() {
            let qa = qr[a];
            x += xr[a] * qa;
            y += yr[a] * qa;
            z += zr[a] * qa;
            l += lr[a] * qa;
        }
        let out = cache.output.as_mut().expect("J3 derivative output live");
        out.grad_dn[[electron, 0]] = x;
        out.grad_dn[[electron, 1]] = y;
        out.grad_dn[[electron, 2]] = z;
        out.lap_dn[electron] = l;
    }
}

fn compute_grads_and_laplacian_jastrow_three_body_impl(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    fast_cut: bool,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    if j3_orb.precision_policy.is_f32("jastrow_grad_lap") {
        return compute_grads_and_laplacian_jastrow_three_body_f32(j3_orb, j_matrix, r_up, r_dn);
    }
    let ruw = j3_orb
        .wrap_electrons(r_up)
        .unwrap_or_else(|| r_up.to_owned());
    let rdw = j3_orb
        .wrap_electrons(r_dn)
        .unwrap_or_else(|| r_dn.to_owned());
    let (r_up, r_dn) = (&ruw.view(), &rdw.view());
    let eval = |r: &ArrayView2<f64>| {
        if fast_cut
            && j3_orb.pw.is_none()
            && j3_orb.bspline.is_none()
            && !j3_orb.aos.shells.is_empty()
        {
            let av = crate::atomic_orbital::compute_aos(&j3_orb.aos, r);
            let (gx, gy, gz, lap) =
                crate::atomic_orbital::compute_aos_grad_lap_fast_cut(&j3_orb.aos, r);
            match &j3_orb.mo_coefficients {
                None => (av, gx, gy, gz, lap),
                Some(c) => (
                    matmul(&c.view(), &av.view()),
                    matmul(&c.view(), &gx.view()),
                    matmul(&c.view(), &gy.view()),
                    matmul(&c.view(), &gz.view()),
                    matmul(&c.view(), &lap.view()),
                ),
            }
        } else {
            orb_value_grad_lap(j3_orb, r)
        }
    };
    let (au, aux, auy, auz, aul) = eval(r_up); // (n_orb, n_up)
    let (ad, adx, ady, adz, adl) = eval(r_dn); // (n_orb, n_dn)
    let n_orb = au.nrows();
    let n_up = au.ncols();
    let n_dn = ad.ncols();
    let j1: Vec<f64> = (0..n_orb).map(|o| j_matrix[[o, n_orb]]).collect();
    let j3 = j_matrix.slice(s![.., 0..n_orb]); // (n_orb, n_orb)

    // The J3 coupling block is symmetric (bosonic Jastrow;
    // enforced at native-ctx build by `assert_j3_block_symmetric`, and by jqmc's
    // `Jastrow_three_body_data.sanity_check`). Hence J3^T*A == J3*A, so the single
    // product feeds BOTH the suffix (J3*A, m>n) and the prefix (J3^T*A, m<n)
    // triangular sums, so no scalar `mat_t_mul` triple loop (a kin.j3 hot spot) is
    // needed -- a mathematical identity, not an approximation.
    let a_up = matmul(&j3, &au.view()); // J3*A_up == J3^T*A_up   (n_orb, n_up)
    let a_dn = matmul(&j3, &ad.view()); // J3*A_dn == J3^T*A_dn   (n_orb, n_dn)

    // g[o,n] with the triangular suffix/prefix sums + cross-spin row-sum. Under
    // symmetry the prefix and suffix operands coincide (both `a`).
    let build_g = |a: &Array2<f64>,
                   cross: &Array2<f64>, // the other spin's J3*A (row-summed)
                   ne: usize,
                   ne_cross: usize|
     -> Array2<f64> {
        let mut g = Array2::<f64>::zeros((n_orb, ne));
        for o in 0..n_orb {
            let rs_cross: f64 = (0..ne_cross).map(|m| cross[[o, m]]).sum();
            // suffix[n] = Sum_{m>n} a[o,m]
            let mut suffix = vec![0.0_f64; ne];
            let mut acc = 0.0;
            for n in (0..ne).rev() {
                suffix[n] = acc;
                acc += a[[o, n]];
            }
            // prefix accumulate Sum_{m<n} a[o,m]  (== J3^T*A prefix by symmetry)
            let mut pref = 0.0;
            for n in 0..ne {
                g[[o, n]] = j1[o] + suffix[n] + pref + rs_cross;
                pref += a[[o, n]];
            }
        }
        g
    };
    let g_up = build_g(&a_up, &a_dn, n_up, n_dn);
    let g_dn = build_g(&a_dn, &a_up, n_dn, n_up);

    let contract = |g: &Array2<f64>,
                    gx: &Array2<f64>,
                    gy: &Array2<f64>,
                    gz: &Array2<f64>,
                    lap: &Array2<f64>,
                    ne: usize|
     -> (Array2<f64>, Array1<f64>) {
        let mut grad = Array2::<f64>::zeros((ne, 3));
        let mut lp = Array1::<f64>::zeros(ne);
        for n in 0..ne {
            let (mut x, mut y, mut z, mut l) = (0.0, 0.0, 0.0, 0.0);
            for o in 0..n_orb {
                let gv = g[[o, n]];
                x += gv * gx[[o, n]];
                y += gv * gy[[o, n]];
                z += gv * gz[[o, n]];
                l += gv * lap[[o, n]];
            }
            grad[[n, 0]] = x;
            grad[[n, 1]] = y;
            grad[[n, 2]] = z;
            lp[n] = l;
        }
        (grad, lp)
    };
    let (grad_up, lap_up) = contract(&g_up, &aux, &auy, &auz, &aul, n_up);
    let (grad_dn, lap_dn) = contract(&g_dn, &adx, &ady, &adz, &adl, n_dn);
    (grad_up, grad_dn, lap_up, lap_dn)
}

fn compute_grads_and_laplacian_jastrow_three_body_f32(
    j3_orb: &OrbData,
    j_matrix: &ArrayView2<f64>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let ruw = j3_orb
        .wrap_electrons(r_up)
        .unwrap_or_else(|| r_up.to_owned());
    let rdw = j3_orb
        .wrap_electrons(r_dn)
        .unwrap_or_else(|| r_dn.to_owned());
    let (au, aux, auy, auz, aul) = orb_value_grad_lap(j3_orb, &ruw.view());
    let (ad, adx, ady, adz, adl) = orb_value_grad_lap(j3_orb, &rdw.view());
    let (norb, nu, nd) = (au.nrows(), au.ncols(), ad.ncols());
    let mul = |aos: &Array2<f64>| -> Array2<f32> {
        // out[a, e] = sum_b (J[a,b] as f32) * (A[b,e] as f32), b ascending, fp32;
        // A is cast once and each row update runs over contiguous electrons (the
        // same per-element operation sequence as an indexed triple loop).
        let ne = aos.ncols();
        let a32 = aos.mapv(|v| v as f32).as_standard_layout().to_owned();
        let a32 = a32.as_slice().expect("fp32 AO table contiguous");
        let mut out = Array2::<f32>::zeros((norb, ne));
        for a in 0..norb {
            let mut orow = out.row_mut(a);
            let orow = orow.as_slice_mut().expect("fp32 J*A row contiguous");
            for b in 0..norb {
                let jab = j_matrix[[a, b]] as f32;
                for (o, &v) in orow.iter_mut().zip(&a32[b * ne..(b + 1) * ne]) {
                    *o += jab * v;
                }
            }
        }
        out
    };
    let wu = mul(&au);
    let wd = mul(&ad);
    let build_g = |w: &Array2<f32>, cross: &Array2<f32>| -> Array2<f32> {
        let ne = w.ncols();
        let mut g = Array2::<f32>::zeros((norb, ne));
        for o in 0..norb {
            let mut cross_sum = 0.0_f32;
            for e in 0..cross.ncols() {
                cross_sum += cross[[o, e]];
            }
            let mut suffix = vec![0.0_f32; ne];
            let mut s = 0.0_f32;
            for e in (0..ne).rev() {
                suffix[e] = s;
                s += w[[o, e]];
            }
            let mut prefix = 0.0_f32;
            for e in 0..ne {
                g[[o, e]] = j_matrix[[o, norb]] as f32 + suffix[e] + prefix + cross_sum;
                prefix += w[[o, e]];
            }
        }
        g
    };
    let gu = build_g(&wu, &wd);
    let gd = build_g(&wd, &wu);
    let contract =
        |g: &Array2<f32>, gx: &Array2<f64>, gy: &Array2<f64>, gz: &Array2<f64>, l: &Array2<f64>| {
            let mut grad = Array2::<f64>::zeros((g.ncols(), 3));
            let mut lap = Array1::<f64>::zeros(g.ncols());
            for e in 0..g.ncols() {
                // XLA promotes g(fp32) * AO-gradient(fp64) to fp64.  The AO
                // Laplacian is explicitly cast into jastrow_grad_lap, so only the
                // Laplacian contraction remains fp32.
                let (mut x, mut y, mut z) = (0.0_f64, 0.0_f64, 0.0_f64);
                let mut lp = 0.0_f32;
                for o in 0..norb {
                    let q = g[[o, e]];
                    x += q as f64 * gx[[o, e]];
                    y += q as f64 * gy[[o, e]];
                    z += q as f64 * gz[[o, e]];
                    lp += q * l[[o, e]] as f32;
                }
                grad[[e, 0]] = x;
                grad[[e, 1]] = y;
                grad[[e, 2]] = z;
                lap[e] = lp as f64;
            }
            (grad, lap)
        };
    let (gup, lup) = contract(&gu, &aux, &auy, &auz, &aul);
    let (gdn, ldn) = contract(&gd, &adx, &ady, &adz, &adl);
    let _ = (nu, nd);
    (gup, gdn, lup, ldn)
}

/// J3-advance matvec: `d_j[b] = Sum_c j3[b,c]*delta[c]` (row dot per row of the
/// dense nxn j_matrix, row-major, read once). By symmetry, the coupling
/// block is symmetric, so `j3^T*delta == j3*delta` -- the advance's second output
/// (`d_jt`) is bit-equal to `d_j`, so a SINGLE matvec
/// drives both the column and the cross-rowsum update (halves the accept work).
/// The dense matvec is the J3 hot spot (n_orb^2 per accept); single portable
/// row-dot with 8 independent FMA accumulators, auto-vectorized under
/// `target-cpu=native`.
pub(crate) fn j3_matvec(jm: &[f64], delta: &[f64], n: usize, ncol: usize, d_j: &mut [f64]) {
    // See the SLP-vectorization rule above dot_product_fast in gfmc.rs:
    // single chunks_exact loop + flat 16-lane accumulator + sequential fold.
    let n16 = n & !15;
    for b in 0..n {
        let row = &jm[b * ncol..b * ncol + n];
        let mut a = [0.0_f64; 16];
        for (rc, dc) in row[..n16]
            .chunks_exact(16)
            .zip(delta[..n16].chunks_exact(16))
        {
            for l in 0..16 {
                a[l] = rc[l].mul_add(dc[l], a[l]);
            }
        }
        let mut s = 0.0;
        for a_l in &a {
            s += *a_l;
        }
        for c in n16..n {
            s += row[c] * delta[c];
        }
        d_j[b] = s;
    }
}

/// J3-ratio accumulate: `v[m] += Sum_a aos_p[a]*W[a,m]`, reading rows of the
/// (n_orbxn) W matrix once. Because W == WT, a second
/// accumulate (`p` over WT) would equal `v` -- one AXPY suffices; the caller
/// forms both the m>k (suffix) and m<k (prefix) sums from this single `v`.
/// Single portable FMA AXPY over the contiguous n-wide rows, auto-vectorized
/// under `target-cpu=native`. Runs every move.
fn j3_ratio_v(w: &[f64], aos_p: &[f64], n_orb: usize, n: usize, v: &mut [f64]) {
    for a in 0..n_orb {
        let apa = aos_p[a];
        if apa == 0.0 {
            continue;
        }
        let wr = &w[a * n..a * n + n];
        for (vm, &wm) in v[..n].iter_mut().zip(wr.iter()) {
            *vm = apa.mul_add(wm, *vm);
        }
    }
}

#[inline]
fn distance(a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize) -> f64 {
    let mut acc = 0.0_f64;
    for k in 0..a.ncols() {
        let d = a[[i, k]] - b[[j, k]];
        acc += d * d;
    }
    acc.sqrt()
}

#[inline]
fn distance_f32(a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize) -> f32 {
    let mut acc = 0.0_f32;
    for k in 0..a.ncols() {
        let d = (a[[i, k]] - b[[j, k]]) as f32;
        acc += d * d;
    }
    acc.sqrt()
}

#[inline]
fn dist_point_row(p: [f64; 3], b: &ArrayView2<f64>, j: usize) -> f64 {
    let dx = p[0] - b[[j, 0]];
    let dy = p[1] - b[[j, 1]];
    let dz = p[2] - b[[j, 2]];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// One-body J1 contribution of a single electron at `r` (sum over atoms).
/// Spin-independent (electron charge -1 for both spins).
fn j1_single_electron(data: &JastrowOneBodyData, r: [f64; 3]) -> f64 {
    let a = data.jastrow_1b_param;
    let is_exp = data.jastrow_1b_type == "exp";
    let r_atoms = data.structure_data.positions.view();
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let mut s = 0.0_f64;
    for at in 0..data.structure_data.n_atoms() {
        let z_eff = data.structure_data.atomic_numbers[at] - data.core_electrons[at];
        let r_en = match &ci {
            Some(ci) => {
                smooth_dist(
                    [
                        r[0] - r_atoms[[at, 0]],
                        r[1] - r_atoms[[at, 1]],
                        r[2] - r_atoms[[at, 2]],
                    ],
                    ci,
                )
                .0
            }
            None => dist_point_row(r, &r_atoms, at),
        };
        let coeff = (2.0 * z_eff).powf(0.25);
        let prefac = if is_exp {
            -(2.0 * z_eff).powf(0.75)
        } else {
            -2.0 * z_eff
        };
        let kernel = if is_exp {
            1.0 / (2.0 * a) * one_minus_exp_neg(a * coeff * r_en)
        } else {
            r_en / (2.0 * (1.0 + a * coeff * r_en))
        };
        s += prefac * kernel;
    }
    s
}

/// [`j1_single_electron`] with its per-call constants (cell geometry, the
/// per-atom `powf` factors, the form tag) built once; [`J1Prep::eval`] repeats
/// the same arithmetic in the same order, so values are bitwise identical.
struct J1Prep<'a> {
    data: &'a JastrowOneBodyData,
    ci: Option<CellInfo>,
    is_exp: bool,
    coeff: Vec<f64>,
    prefac: Vec<f64>,
}

impl<'a> J1Prep<'a> {
    fn new(data: &'a JastrowOneBodyData) -> Self {
        let is_exp = data.jastrow_1b_type == "exp";
        let n = data.structure_data.n_atoms();
        let (mut coeff, mut prefac) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for at in 0..n {
            let z_eff = data.structure_data.atomic_numbers[at] - data.core_electrons[at];
            coeff.push((2.0 * z_eff).powf(0.25));
            prefac.push(if is_exp {
                -(2.0 * z_eff).powf(0.75)
            } else {
                -2.0 * z_eff
            });
        }
        let ci = data
            .structure_data
            .cell
            .as_ref()
            .map(|c| CellInfo::new(&c.view()));
        J1Prep {
            data,
            ci,
            is_exp,
            coeff,
            prefac,
        }
    }

    #[inline]
    fn eval(&self, r: [f64; 3]) -> f64 {
        let a = self.data.jastrow_1b_param;
        let r_atoms = self.data.structure_data.positions.view();
        let mut s = 0.0_f64;
        for at in 0..self.coeff.len() {
            let r_en = match &self.ci {
                Some(ci) => {
                    smooth_dist(
                        [
                            r[0] - r_atoms[[at, 0]],
                            r[1] - r_atoms[[at, 1]],
                            r[2] - r_atoms[[at, 2]],
                        ],
                        ci,
                    )
                    .0
                }
                None => dist_point_row(r, &r_atoms, at),
            };
            let coeff = self.coeff[at];
            let kernel = if self.is_exp {
                1.0 / (2.0 * a) * one_minus_exp_neg(a * coeff * r_en)
            } else {
                r_en / (2.0 * (1.0 + a * coeff * r_en))
            };
            s += self.prefac[at] * kernel;
        }
        s
    }
}

/// [`j2_from_dist`] with the parameter and form tag passed in (same arithmetic).
#[inline(always)]
fn j2_from_dist_kind(a: f64, is_exp: bool, dist: f64) -> f64 {
    if is_exp {
        1.0 / (2.0 * a) * one_minus_exp_neg(a * dist)
    } else {
        dist / 2.0 * (1.0 + a * dist).powi(-1)
    }
}

/// Two-body J2 pair value as a function of distance.
#[inline]
fn j2_from_dist(data: &JastrowTwoBodyData, dist: f64) -> f64 {
    let a = data.jastrow_2b_param;
    if data.jastrow_2b_type == "exp" {
        1.0 / (2.0 * a) * one_minus_exp_neg(a * dist)
    } else {
        dist / 2.0 * (1.0 + a * dist).powi(-1)
    }
}

pub(crate) use crate::setting_consts::EPS_safe_distance as EPS_SAFE_DISTANCE;

/// Precomputed full-Gram geometry for conventional J1/J2. The DTN branch keeps
/// its separate Euclidean minimum-image law and only uses `cell`, `inv` and
/// `mic_half_widths`.
/// Mixed derivative factors are formed AFTER the cell/inverse casts, matching
/// the XLA zone boundary, not by casting already contracted fp64 Gram entries.
pub(crate) struct CellInfo {
    pub(crate) inv: [[f64; 3]; 3],
    pub(crate) cell: [[f64; 3]; 3],
    pub(crate) gram: [[f64; 3]; 3],
    pub(crate) inverse_gram: [[f64; 3]; 3],
    inv_f32: [[f32; 3]; 3],
    cell_f32: [[f32; 3]; 3],
    gram_f32: [[f32; 3]; 3],
    // Diagonal, upper and lower triangles respectively hold the coefficients
    // of z_i, z_i*z_j and h_i*h_j in the trace polynomial.
    trace_coeff: [[f64; 3]; 3],
    trace_coeff_f32: [[f32; 3]; 3],
    // DTN minimum-image shell `(N_a, N_b, N_c)` searched after the
    // parallelepiped wrap in [`min_image_disp`] (issue #68). Same rule as
    // `jqmc.structure._minimum_image_shell_half_widths(cell)`: with
    // `rho = min_i |b_i| / 2` (>= the Wigner-Seitz inscribed radius) and the
    // reciprocal columns `g_k = inv[:, k]`, `N_k = ceil(rho |g_k| + 1/2) - 1`,
    // and `N_k = 0` for an axis exactly orthogonal to the other two. Every
    // displacement whose minimum image is shorter than `rho` then has
    // `|n_k| <= N_k` relative to the wrap. `[0, 0, 0]` (orthorhombic) is the
    // plain wrap.
    pub(crate) mic_half_widths: [i64; 3],
}

impl CellInfo {
    #[inline(always)]
    pub(crate) fn new(cell: &ArrayView2<f64>) -> Self {
        let inv = crate::ewald::inv3(cell);
        let cm = [
            [cell[[0, 0]], cell[[0, 1]], cell[[0, 2]]],
            [cell[[1, 0]], cell[[1, 1]], cell[[1, 2]]],
            [cell[[2, 0]], cell[[2, 1]], cell[[2, 2]]],
        ];
        Self::from_parts(cm, inv)
    }

    // Expose unused precision/derivative fields to dead-code elimination in
    // value-only callers; this changes no arithmetic in the retained fields.
    #[inline(always)]
    pub(crate) fn from_parts(cell: [[f64; 3]; 3], inv: [[f64; 3]; 3]) -> Self {
        let cell_f32 = cell.map(|row| row.map(|value| value as f32));
        let inv_f32 = inv.map(|row| row.map(|value| value as f32));
        let mut gram = [[0.0; 3]; 3];
        let mut inverse_gram = [[0.0; 3]; 3];
        let mut gram_f32 = [[0.0; 3]; 3];
        let mut inverse_gram_f32 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in i..3 {
                for k in 0..3 {
                    gram[i][j] += cell[i][k] * cell[j][k];
                    inverse_gram[i][j] += inv[k][i] * inv[k][j];
                    gram_f32[i][j] += cell_f32[i][k] * cell_f32[j][k];
                    inverse_gram_f32[i][j] += inv_f32[k][i] * inv_f32[k][j];
                }
                // Both Gram matrices are symmetric. Mirror each dot product
                // without changing its k-ordered accumulation in either dtype.
                gram[j][i] = gram[i][j];
                inverse_gram[j][i] = inverse_gram[i][j];
                gram_f32[j][i] = gram_f32[i][j];
                inverse_gram_f32[j][i] = inverse_gram_f32[i][j];
            }
        }
        let mut trace_coeff = [[0.0; 3]; 3];
        let mut trace_coeff_f32 = [[0.0; 3]; 3];
        for i in 0..3 {
            trace_coeff[i][i] = 2.0 * gram[i][i] * inverse_gram[i][i] - 4.0;
            trace_coeff_f32[i][i] = 2.0 * gram_f32[i][i] * inverse_gram_f32[i][i] - 4.0;
            for j in (i + 1)..3 {
                trace_coeff[i][j] = 8.0 * gram[i][j] * inverse_gram[i][j];
                trace_coeff[j][i] = 4.0 * gram[i][j] * (inverse_gram[i][i] + inverse_gram[j][j]);
                trace_coeff_f32[i][j] = 8.0 * gram_f32[i][j] * inverse_gram_f32[i][j];
                trace_coeff_f32[j][i] =
                    4.0 * gram_f32[i][j] * (inverse_gram_f32[i][i] + inverse_gram_f32[j][j]);
            }
        }
        let mic_half_widths = dtn_mic_half_widths(&gram, &inverse_gram);
        Self {
            inv,
            cell,
            gram,
            inverse_gram,
            inv_f32,
            cell_f32,
            gram_f32,
            trace_coeff,
            trace_coeff_f32,
            mic_half_widths,
        }
    }
}

/// DTN minimum-image **Euclidean** distance + Cartesian derivatives for `x = r_a -
/// r_b` (hard `(L-r)^C` cutoff needs the true min-image distance, NOT
/// the Williamson smooth metric -- jqmc `_dtn_scalar_dist_with_derivs`). Inside the cutoff (`L < R_inscribed`) the image index is locally
/// constant, so the derivatives collapse to the plain Euclidean forms on the
/// min-image displacement `u_min`: returns `(d, u_min/d [3], |gradd|^2=1, 2/d)`.
#[inline]
pub(crate) fn min_image_dist(x: [f64; 3], ci: &CellInfo) -> (f64, [f64; 3], f64, f64) {
    let u = min_image_disp(x, ci);
    let d = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        .sqrt()
        .max(EPS_SAFE_DISTANCE);
    (d, [u[0] / d, u[1] / d, u[2] / d], 1.0, 2.0 / d)
}

/// DTN minimum-image shell half-widths from the cell Gram matrix
/// (`gram[i][j] = b_i . b_j`) and the reciprocal Gram matrix
/// (`inverse_gram[k][k] = |g_k|^2`); see `CellInfo::mic_half_widths`.
fn dtn_mic_half_widths(gram: &[[f64; 3]; 3], inverse_gram: &[[f64; 3]; 3]) -> [i64; 3] {
    let shortest_row_sq = gram[0][0].min(gram[1][1]).min(gram[2][2]);
    let rho = 0.5 * shortest_row_sq.sqrt();
    let mut out = [0_i64; 3];
    for k in 0..3 {
        let orthogonal = (0..3).all(|i| i == k || gram[k][i] == 0.0);
        if !orthogonal {
            let bound = (rho * inverse_gram[k][k].sqrt() + 0.5).ceil() as i64 - 1;
            out[k] = bound.max(0);
        }
    }
    out
}

/// Min-image DISPLACEMENT (the DTN min-image convention shared with
/// [`min_image_dist`] and jqmc `_dtn_minimum_image`): first the
/// parallelepiped wrap `u_w = x - round(x@inv_cell)@cell`, then the shortest
/// of `u_w` and `u_w + n@cell` over the static shell `|n_k| <= N_k`
/// (`CellInfo::mic_half_widths`). The wrap alone is exact only below half the
/// smallest interplanar spacing of the given basis; with the shell the result
/// is the true minimum image for every displacement whose minimum image is
/// shorter than the Wigner-Seitz inscribed radius, the bound on every DTN
/// cutoff reach (issue #68). Beyond that the result may be a longer image,
/// which still lies outside every cutoff. Ties keep the wrapped `u_w`
/// (strict `<`). Bit-exact identity (`u_min == x`, the subtracted lattice
/// shift is exactly 0) whenever the rounded image index is 0 and no shell
/// candidate is strictly shorter -- i.e. for every in-cell displacement shorter
/// than the cutoffs -- so the DTN force kernels built on it stay bit-identical
/// at near configs. An orthorhombic cell (`N = 0`) takes the plain wrap.
#[inline]
pub(crate) fn min_image_disp(x: [f64; 3], ci: &CellInfo) -> [f64; 3] {
    // u_frac = x*inv_cell ; u_min = x - round(u_frac)*cell
    let mut uf = [0.0_f64; 3];
    for (k, uf_k) in uf.iter_mut().enumerate() {
        *uf_k = x[0] * ci.inv[0][k] + x[1] * ci.inv[1][k] + x[2] * ci.inv[2][k];
    }
    let n = [uf[0].round(), uf[1].round(), uf[2].round()];
    let mut u = [0.0_f64; 3];
    for c in 0..3 {
        u[c] = x[c] - (n[0] * ci.cell[0][c] + n[1] * ci.cell[1][c] + n[2] * ci.cell[2][c]);
    }
    let [ha, hb, hc] = ci.mic_half_widths;
    if ha == 0 && hb == 0 && hc == 0 {
        return u;
    }
    let mut best = u;
    let mut best_d2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    for na in -ha..=ha {
        for nb in -hb..=hb {
            for nc in -hc..=hc {
                if na == 0 && nb == 0 && nc == 0 {
                    continue;
                }
                let (fa, fb, fc) = (na as f64, nb as f64, nc as f64);
                let mut cand = [0.0_f64; 3];
                for c in 0..3 {
                    cand[c] = u[c] + (fa * ci.cell[0][c] + fb * ci.cell[1][c] + fc * ci.cell[2][c]);
                }
                let d2 = cand[0] * cand[0] + cand[1] * cand[1] + cand[2] * cand[2];
                if d2 < best_d2 {
                    best = cand;
                    best_d2 = d2;
                }
            }
        }
    }
    best
}

/// Positive periodic pair geometry of [`positive_geometry_kernel`], generic over the
/// precision route (`PositiveGeometry` f64, `PositiveGeometryF32` f32). The value-only
/// f32 route reads only `square` / `numerator` / `trace`.
pub(crate) struct PositiveGeometryOf<T> {
    pub(crate) square: T,
    pub(crate) numerator: [T; 3],
    pub(crate) trace: T,
    pub(crate) cart: [T; 3],
    pub(crate) z: [T; 3],
    pub(crate) h: [T; 3],
}
pub(crate) type PositiveGeometry = PositiveGeometryOf<f64>;
pub(crate) type PositiveGeometryF32 = PositiveGeometryOf<f32>;

// Both precision routes use the same closed algebra, specialized at compile
// time. No per-pair inversion or context allocation occurs in this helper.
macro_rules! positive_geometry_kernel {
    ($name:ident, $state:ident, $real:ty, $pi:expr) => {
        // Value-only callers must erase derivative fields before entering the
        // pair loop; outlining this larger shared kernel computes them anyway.
        #[inline(always)]
        fn $name(
            x: [$real; 3],
            cell: &[[$real; 3]; 3],
            inv: &[[$real; 3]; 3],
            gram: &[[$real; 3]; 3],
            trace_coeff: &[[$real; 3]; 3],
        ) -> $state {
            let pi: $real = $pi;
            let image: [$real; 3] = std::array::from_fn(|i| {
                (x[0] * inv[0][i] + x[1] * inv[1][i] + x[2] * inv[2][i]).round_ties_even()
            });
            let local: [$real; 3] = std::array::from_fn(|k| {
                x[k] - (image[0] * cell[0][k] + image[1] * cell[1][k] + image[2] * cell[2][k])
            });
            let mut z = [0.0; 3];
            let mut h = [0.0; 3];
            let mut theta = [0.0; 3];
            for i in 0..3 {
                theta[i] =
                    pi * (local[0] * inv[0][i] + local[1] * inv[1][i] + local[2] * inv[2][i]);
                let (sn, cs) = theta[i].sin_cos();
                z[i] = sn * sn;
                h[i] = sn * cs;
            }
            // Only near coalescence does the linear Cartesian reconstruction
            // need a cancellation-free residual. Away from it the direct
            // positive form avoids three unnecessary scalar residual branches.
            let cart: [$real; 3] = if theta
                .iter()
                .all(|value| value * value < <$real>::EPSILON.sqrt().sqrt())
            {
                let correction: [$real; 3] = theta.map(|value| {
                    let square = value * value;
                    value
                        * square
                        * (-2.0 / 3.0
                            + square
                                * (2.0 / 15.0 + square * (-4.0 / 315.0 + square * (2.0 / 2835.0))))
                        / pi
                });
                std::array::from_fn(|k| {
                    local[k]
                        + correction[0] * cell[0][k]
                        + correction[1] * cell[1][k]
                        + correction[2] * cell[2][k]
                })
            } else {
                std::array::from_fn(|k| {
                    (h[0] * cell[0][k] + h[1] * cell[1][k] + h[2] * cell[2][k]) / pi
                })
            };
            let off_h: [$real; 3] = std::array::from_fn(|i| {
                (0..3).filter(|&j| i != j).map(|j| gram[i][j] * h[j]).sum()
            });
            let numerator = std::array::from_fn(|k| {
                cart[k]
                    - (2.0 / pi)
                        * (z[0] * off_h[0] * inv[k][0]
                            + z[1] * off_h[1] * inv[k][1]
                            + z[2] * off_h[2] * inv[k][2])
            });
            let mut square = cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2];
            let mut trace = 3.0;
            for i in 0..3 {
                square += gram[i][i] * z[i] * z[i] / (pi * pi);
                trace += trace_coeff[i][i] * z[i];
                for j in (i + 1)..3 {
                    trace += trace_coeff[i][j] * z[i] * z[j] - trace_coeff[j][i] * h[i] * h[j];
                }
            }
            $state {
                square,
                numerator,
                trace,
                cart,
                z,
                h,
            }
        }
    };
}
positive_geometry_kernel!(
    positive_geometry_f64,
    PositiveGeometry,
    f64,
    std::f64::consts::PI
);
positive_geometry_kernel!(
    positive_geometry_f32,
    PositiveGeometryF32,
    f32,
    std::f32::consts::PI
);

#[inline(always)]
pub(crate) fn positive_geometry(x: [f64; 3], ci: &CellInfo) -> PositiveGeometry {
    positive_geometry_f64(x, &ci.cell, &ci.inv, &ci.gram, &ci.trace_coeff)
}

/// Positive full-Gram periodic distance with the existing safe-distance clamp.
#[inline(always)]
pub(crate) fn smooth_dist(x: [f64; 3], ci: &CellInfo) -> (f64, [f64; 3], f64, f64) {
    let raw = positive_geometry(x, ci);
    let d = raw.square.sqrt().max(EPS_SAFE_DISTANCE);
    let grad = raw.numerator.map(|value| value / d);
    let q = grad.iter().map(|value| value * value).sum::<f64>();
    (d, grad, q, (raw.trace - q) / d)
}

#[inline(always)]
fn smooth_dist_value_f32(x: [f64; 3], ci: &CellInfo) -> f32 {
    // Reconstruct value geometry in fp64 before crossing jastrow_eval.
    positive_geometry(x, ci)
        .square
        .sqrt()
        .max(EPS_SAFE_DISTANCE) as f32
}

#[inline(always)]
fn smooth_dist_f32(x: [f64; 3], ci: &CellInfo) -> (f32, [f32; 3], f32, f32) {
    let raw = positive_geometry_f32(
        x.map(|v| v as f32),
        &ci.cell_f32,
        &ci.inv_f32,
        &ci.gram_f32,
        &ci.trace_coeff_f32,
    );
    let d = raw.square.sqrt().max(EPS_SAFE_DISTANCE as f32);
    let grad = raw.numerator.map(|value| value / d);
    let q = grad.iter().map(|value| value * value).sum::<f32>();
    (d, grad, q, (raw.trace - q) / d)
}

/// Closed Hessian and gradient-of-Laplacian of the positive law. The caller
/// chooses the existing denominator convention; force/stress use unclamped
/// geometry while Jastrow kernels retain EPS_SAFE_DISTANCE.
#[inline]
#[allow(clippy::type_complexity)]
pub(crate) fn positive_hessian(
    raw: &PositiveGeometry,
    ci: &CellInfo,
    d: f64,
) -> ([f64; 3], f64, f64, [[f64; 3]; 3], [f64; 3]) {
    let g = raw.numerator.map(|value| value / d);
    let q = g.iter().map(|value| value * value).sum::<f64>();
    let lap = (raw.trace - q) / d;
    let off_h: [f64; 3] = std::array::from_fn(|i| {
        (0..3)
            .filter(|&j| i != j)
            .map(|j| ci.gram[i][j] * raw.h[j])
            .sum()
    });
    let weights: [f64; 3] = std::array::from_fn(|i| {
        2.0 * ci.gram[i][i] * (1.0 - 2.0 * raw.z[i]) * raw.z[i] - 4.0 * raw.h[i] * off_h[i]
    });
    let mut transform = [[0.0; 3]; 3];
    for (i, transform_i) in transform.iter_mut().enumerate() {
        for (j, t_ij) in transform_i.iter_mut().enumerate() {
            *t_ij = if i == j { 1.0 } else { 0.0 };
            for k in 0..3 {
                *t_ij -= 2.0 * ci.inv[i][k] * raw.z[k] * ci.cell[k][j];
            }
        }
    }
    let mut hess = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut half = 0.0;
            for k in 0..3 {
                half +=
                    transform[i][k] * transform[j][k] + ci.inv[i][k] * weights[k] * ci.inv[j][k];
            }
            hess[i][j] = (half - g[i] * g[j]) / d;
        }
    }
    let trace_theta: [f64; 3] = std::array::from_fn(|i| {
        let mut value = 2.0 * (2.0 * ci.gram[i][i] * ci.inverse_gram[i][i] - 4.0) * raw.h[i];
        for j in 0..3 {
            if i != j {
                value += 16.0 * raw.h[i] * raw.z[j] * ci.gram[i][j] * ci.inverse_gram[i][j]
                    - 4.0
                        * (1.0 - 2.0 * raw.z[i])
                        * raw.h[j]
                        * ci.gram[i][j]
                        * (ci.inverse_gram[i][i] + ci.inverse_gram[j][j]);
            }
        }
        value
    });
    let gradlap = std::array::from_fn(|i| {
        let trace_grad =
            std::f64::consts::PI * (0..3).map(|k| trace_theta[k] * ci.inv[i][k]).sum::<f64>();
        let grad_q = 2.0 * (0..3).map(|j| hess[i][j] * g[j]).sum::<f64>();
        (trace_grad - grad_q - lap * g[i]) / d
    });
    (g, q, lap, hess, gradlap)
}

#[inline]
#[allow(clippy::type_complexity)]
pub(crate) fn smooth_dist_hess(
    x: [f64; 3],
    ci: &CellInfo,
) -> (f64, [f64; 3], f64, f64, [[f64; 3]; 3], [f64; 3]) {
    let raw = positive_geometry(x, ci);
    let d = raw.square.sqrt().max(EPS_SAFE_DISTANCE);
    let (g, q, lap, hess, gradlap) = positive_hessian(&raw, ci, d);
    (d, g, q, lap, hess, gradlap)
}

/// One-body Jastrow grad/lap per spin (`compute_grads_and_laplacian_Jastrow_one_body`,
/// molecular). Returns `(grad (n,3), lap (n,))`.
pub(crate) fn j1_grad_lap_one_spin(
    data: &JastrowOneBodyData,
    r: &ArrayView2<f64>,
) -> (Array2<f64>, Array1<f64>) {
    if data.precision_policy.is_f32("jastrow_grad_lap") {
        return j1_grad_lap_one_spin_f32(data, r);
    }
    let n = r.nrows();
    let pos = &data.structure_data.positions;
    let zn = &data.structure_data.atomic_numbers;
    let ce = &data.core_electrons;
    let n_atom = pos.nrows();
    let a = data.jastrow_1b_param;
    let is_exp = data.jastrow_1b_type == "exp";
    let eps = EPS_SAFE_DISTANCE;
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let radial: Vec<_> = (0..n_atom)
        .map(|i| {
            let z_eff = zn[i] - ce[i];
            (
                (2.0 * z_eff).powf(0.25),
                if is_exp {
                    (2.0 * z_eff).powf(0.75)
                } else {
                    2.0 * z_eff
                },
            )
        })
        .collect();
    let mut grad = Array2::<f64>::zeros((n, 3));
    let mut lap = Array1::<f64>::zeros(n);
    for e in 0..n {
        let (mut gx, mut gy, mut gz, mut lp) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n_atom {
            let (c, aa) = radial[i];
            let dx = r[[e, 0]] - pos[[i, 0]];
            let dy = r[[e, 1]] - pos[[i, 1]];
            let dz = r[[e, 2]] - pos[[i, 2]];
            // Distance + its grad/lap geometry (Euclidean or Williamson smooth).
            let (d, gvec, gnorm2, lap_d) = match &ci {
                Some(ci) => smooth_dist([dx, dy, dz], ci),
                None => {
                    let r_safe = (dx * dx + dy * dy + dz * dz).sqrt().max(eps);
                    (
                        r_safe,
                        [dx / r_safe, dy / r_safe, dz / r_safe],
                        1.0,
                        2.0 / r_safe,
                    )
                }
            };
            // Radial kernel derivatives u'(d), u''(d).
            let (fprime, fsecond);
            if is_exp {
                let ex = (-a * c * d).exp();
                fprime = -aa * (c / 2.0) * ex;
                fsecond = aa * (a * c * c / 2.0) * ex;
            } else {
                let denom = 1.0 + a * c * d;
                fprime = -aa / (2.0 * denom * denom);
                fsecond = aa * a * c / (denom * denom * denom);
            }
            gx += fprime * gvec[0];
            gy += fprime * gvec[1];
            gz += fprime * gvec[2];
            lp += fsecond * gnorm2 + fprime * lap_d;
        }
        grad[[e, 0]] = gx;
        grad[[e, 1]] = gy;
        grad[[e, 2]] = gz;
        lap[e] = lp;
    }
    (grad, lap)
}

fn j1_grad_lap_one_spin_f32(
    data: &JastrowOneBodyData,
    r: &ArrayView2<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let mut grad = Array2::<f64>::zeros((r.nrows(), 3));
    let mut lap = Array1::<f64>::zeros(r.nrows());
    let pos = &data.structure_data.positions;
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let a = data.jastrow_1b_param as f32;
    let is_exp = data.jastrow_1b_type == "exp";
    let radial: Vec<_> = (0..pos.nrows())
        .map(|i| {
            let z = (data.structure_data.atomic_numbers[i] - data.core_electrons[i]) as f32;
            (
                (2.0 * z).powf(0.25),
                if is_exp {
                    (2.0 * z).powf(0.75)
                } else {
                    2.0 * z
                },
            )
        })
        .collect();
    for e in 0..r.nrows() {
        let (mut gx, mut gy, mut gz, mut lp) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for i in 0..pos.nrows() {
            // Form r - R from the fp64 walker and nuclear coordinates, then
            // downcast (XLA `compute_grads_and_laplacian_Jastrow_one_body`):
            // rounding the absolute coordinates to f32 first shifts a small
            // e-N separation by up to half an f32 ulp of |R|.
            let diff64 = [
                r[[e, 0]] - pos[[i, 0]],
                r[[e, 1]] - pos[[i, 1]],
                r[[e, 2]] - pos[[i, 2]],
            ];
            let diff = [diff64[0] as f32, diff64[1] as f32, diff64[2] as f32];
            let (d, g, q, ld) = match &ci {
                Some(ci) => smooth_dist_f32(diff64, ci),
                None => {
                    let d = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        .sqrt()
                        .max(EPS_SAFE_DISTANCE as f32);
                    (d, [diff[0] / d, diff[1] / d, diff[2] / d], 1.0, 2.0 / d)
                }
            };
            let (c, aa) = radial[i];
            let (fp, fs) = if is_exp {
                let ex = (-a * c * d).exp();
                (-aa * (c / 2.0) * ex, aa * (a * c * c / 2.0) * ex)
            } else {
                let den = 1.0 + a * c * d;
                (-aa / (2.0 * den * den), aa * a * c / (den * den * den))
            };
            if ci.is_none() {
                // Match the molecular XLA expression tree exactly:
                // (fp * diff) / r and fs + (2 * fp) / r.
                gx += (fp * diff[0]) / d;
                gy += (fp * diff[1]) / d;
                gz += (fp * diff[2]) / d;
                lp += fs + (2.0 * fp) / d;
            } else {
                gx += fp * g[0];
                gy += fp * g[1];
                gz += fp * g[2];
                lp += fs * q + fp * ld;
            }
        }
        grad[[e, 0]] = gx as f64;
        grad[[e, 1]] = gy as f64;
        grad[[e, 2]] = gz as f64;
        lap[e] = lp as f64;
    }
    (grad, lap)
}

/// Two-body pair gradient vector + laplacian for displacement `(dx,dy,dz)`
/// (pade/exp). Molecular when `ci` is `None`; Williamson smooth distance when
/// periodic. Returns `(grad_i J [3], grad^2_i J)` for the kernel `u`:
/// `gradJ = u'(d)*gradd`, `grad^2J = u''(d)*|gradd|^2 + u'(d)*grad^2d`.
#[inline]
pub(crate) fn j2_pair(
    is_exp: bool,
    a: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    ci: Option<&CellInfo>,
) -> ([f64; 3], f64) {
    let (d, gvec, gnorm2, lap_d) = match ci {
        Some(ci) => smooth_dist([dx, dy, dz], ci),
        None => {
            let r = (dx * dx + dy * dy + dz * dz).sqrt().max(EPS_SAFE_DISTANCE);
            (r, [dx / r, dy / r, dz / r], 1.0, 2.0 / r)
        }
    };
    let (up, upp);
    if is_exp {
        let ex = (-a * d).exp();
        up = 0.5 * ex;
        upp = -(a / 2.0) * ex;
    } else {
        let denom = 1.0 + a * d;
        up = 0.5 / (denom * denom);
        upp = -a / (denom * denom * denom);
    }
    (
        [up * gvec[0], up * gvec[1], up * gvec[2]],
        upp * gnorm2 + up * lap_d,
    )
}

/// Two-body Jastrow grad/lap (`compute_grads_and_laplacian_Jastrow_two_body`,
/// molecular). Returns `(grad_up,grad_dn,lap_up,lap_dn)`.
pub(crate) fn j2_grad_lap(
    data: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    if data.precision_policy.is_f32("jastrow_grad_lap") {
        return j2_grad_lap_f32(data, r_up, r_dn);
    }
    let nu = r_up.nrows();
    let nd = r_dn.nrows();
    let a = data.jastrow_2b_param;
    let is_exp = data.jastrow_2b_type == "exp";
    let ci = data.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let ci_ref = ci.as_ref();
    let mut gu = Array2::<f64>::zeros((nu, 3));
    let mut gd = Array2::<f64>::zeros((nd, 3));
    let mut lu = Array1::<f64>::zeros(nu);
    let mut ld = Array1::<f64>::zeros(nd);
    for i in 0..nu {
        // Periodic distance is even: evaluate each same-spin pair once.
        // Contributions still reach each electron in ascending partner order.
        for j in (if ci.is_some() { i + 1 } else { 0 })..nu {
            if i == j {
                continue;
            }
            let dx = r_up[[i, 0]] - r_up[[j, 0]];
            let dy = r_up[[i, 1]] - r_up[[j, 1]];
            let dz = r_up[[i, 2]] - r_up[[j, 2]];
            let (g, lp) = j2_pair(is_exp, a, dx, dy, dz, ci_ref);
            gu[[i, 0]] += g[0];
            gu[[i, 1]] += g[1];
            gu[[i, 2]] += g[2];
            lu[i] += lp;
            if ci.is_some() {
                gu[[j, 0]] -= g[0];
                gu[[j, 1]] -= g[1];
                gu[[j, 2]] -= g[2];
                lu[j] += lp;
            }
        }
    }
    for i in 0..nd {
        for j in (if ci.is_some() { i + 1 } else { 0 })..nd {
            if i == j {
                continue;
            }
            let dx = r_dn[[i, 0]] - r_dn[[j, 0]];
            let dy = r_dn[[i, 1]] - r_dn[[j, 1]];
            let dz = r_dn[[i, 2]] - r_dn[[j, 2]];
            let (g, lp) = j2_pair(is_exp, a, dx, dy, dz, ci_ref);
            gd[[i, 0]] += g[0];
            gd[[i, 1]] += g[1];
            gd[[i, 2]] += g[2];
            ld[i] += lp;
            if ci.is_some() {
                gd[[j, 0]] -= g[0];
                gd[[j, 1]] -= g[1];
                gd[[j, 2]] -= g[2];
                ld[j] += lp;
            }
        }
    }
    for i in 0..nu {
        for j in 0..nd {
            let dx = r_up[[i, 0]] - r_dn[[j, 0]];
            let dy = r_up[[i, 1]] - r_dn[[j, 1]];
            let dz = r_up[[i, 2]] - r_dn[[j, 2]];
            let (g, lp) = j2_pair(is_exp, a, dx, dy, dz, ci_ref);
            gu[[i, 0]] += g[0];
            gu[[i, 1]] += g[1];
            gu[[i, 2]] += g[2];
            gd[[j, 0]] -= g[0];
            gd[[j, 1]] -= g[1];
            gd[[j, 2]] -= g[2];
            lu[i] += lp;
            ld[j] += lp;
        }
    }
    (gu, gd, lu, ld)
}

fn j2_pair_f32(is_exp: bool, a: f32, diff64: [f64; 3], ci: Option<&CellInfo>) -> ([f32; 3], f32) {
    let diff = [diff64[0] as f32, diff64[1] as f32, diff64[2] as f32];
    let (d, g, q, ld) = match ci {
        Some(ci) => smooth_dist_f32(diff64, ci),
        None => {
            let d = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                .sqrt()
                .max(EPS_SAFE_DISTANCE as f32);
            (d, [diff[0] / d, diff[1] / d, diff[2] / d], 1.0, 2.0 / d)
        }
    };
    let (up, upp) = if is_exp {
        let ex = (-a * d).exp();
        (0.5 * ex, -(a / 2.0) * ex)
    } else {
        let den = 1.0 + a * d;
        (0.5 / (den * den), -a / (den * den * den))
    };
    if ci.is_none() {
        let coeff = up / d;
        (
            [coeff * diff[0], coeff * diff[1], coeff * diff[2]],
            upp + (2.0 * up) / d,
        )
    } else {
        ([up * g[0], up * g[1], up * g[2]], upp * q + up * ld)
    }
}

fn j2_grad_lap_f32(
    data: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let (nu, nd) = (r_up.nrows(), r_dn.nrows());
    let mut gu = Array2::<f32>::zeros((nu, 3));
    let mut gd = Array2::<f32>::zeros((nd, 3));
    let mut lu = Array1::<f32>::zeros(nu);
    let mut ld = Array1::<f32>::zeros(nd);
    let ci = data.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let a = data.jastrow_2b_param as f32;
    let exp = data.jastrow_2b_type == "exp";
    // Pair displacement formed in fp64 and downcast afterwards, as the XLA
    // `compute_grads_and_laplacian_Jastrow_two_body` does.
    let diff = |x: &ArrayView2<f64>, i: usize, y: &ArrayView2<f64>, j: usize| {
        [
            x[[i, 0]] - y[[j, 0]],
            x[[i, 1]] - y[[j, 1]],
            x[[i, 2]] - y[[j, 2]],
        ]
    };
    for i in 0..nu {
        for j in (if ci.is_some() { i + 1 } else { 0 })..nu {
            if i == j {
                continue;
            }
            let (g, l) = j2_pair_f32(exp, a, diff(r_up, i, r_up, j), ci.as_ref());
            for c in 0..3 {
                gu[[i, c]] += g[c];
                if ci.is_some() {
                    gu[[j, c]] -= g[c];
                }
            }
            lu[i] += l;
            if ci.is_some() {
                lu[j] += l;
            }
        }
    }
    for i in 0..nd {
        for j in (if ci.is_some() { i + 1 } else { 0 })..nd {
            if i == j {
                continue;
            }
            let (g, l) = j2_pair_f32(exp, a, diff(r_dn, i, r_dn, j), ci.as_ref());
            for c in 0..3 {
                gd[[i, c]] += g[c];
                if ci.is_some() {
                    gd[[j, c]] -= g[c];
                }
            }
            ld[i] += l;
            if ci.is_some() {
                ld[j] += l;
            }
        }
    }
    for i in 0..nu {
        for j in 0..nd {
            let (g, l) = j2_pair_f32(exp, a, diff(r_up, i, r_dn, j), ci.as_ref());
            for c in 0..3 {
                gu[[i, c]] += g[c];
                gd[[j, c]] -= g[c];
            }
            lu[i] += l;
            ld[j] += l;
        }
    }
    (
        gu.mapv(f64::from),
        gd.mapv(f64::from),
        lu.mapv(f64::from),
        ld.mapv(f64::from),
    )
}

/// `compute_grads_and_laplacian_Jastrow_part` (J1+J2; molecular analytic).
/// J3, NN-J3, and DTN are evaluated by their dedicated kernels.
pub fn compute_grads_and_laplacian_jastrow_part(
    jdata: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let f32_arithmetic = jdata.precision_policy.is_f32("jastrow_grad_lap");
    let (mut gu, mut gd, mut lu, mut ld) = compute_grads_and_laplacian_j12_part(jdata, r_up, r_dn);
    let add2 = |a: Array2<f64>, b: Array2<f64>| {
        if f32_arithmetic {
            (a.mapv(|x| x as f32) + b.mapv(|x| x as f32)).mapv(f64::from)
        } else {
            a + b
        }
    };
    let add1 = |a: Array1<f64>, b: Array1<f64>| {
        if f32_arithmetic {
            (a.mapv(|x| x as f32) + b.mapv(|x| x as f32)).mapv(f64::from)
        } else {
            a + b
        }
    };
    if let Some(dtn) = &jdata.dtn {
        let (dgu, dgd, dlu, dld) = crate::dtn::compute_grads_and_laplacian_dtn(dtn, r_up, r_dn);
        gu = add2(gu, dgu);
        gd = add2(gd, dgd);
        lu = add1(lu, Array1::from_vec(dlu));
        ld = add1(ld, Array1::from_vec(dld));
    }
    (gu, gd, lu, ld)
}

/// The J1+J2 legs of [`compute_grads_and_laplacian_jastrow_part`] (DTN
/// excluded), with the SAME per-leg accumulation chain -- the parent function
/// is exactly this plus the DTN add, so callers that stream the DTN grad/lap
/// separately (the complex GFMC fast chain's projection-scoped
/// `DtnGradLapCache`) reproduce the parent's fp sequence by adding the DTN
/// tables afterwards in the same order.
pub(crate) fn compute_grads_and_laplacian_j12_part(
    jdata: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let nu = r_up.nrows();
    let nd = r_dn.nrows();
    let mut gu = Array2::<f64>::zeros((nu, 3));
    let mut gd = Array2::<f64>::zeros((nd, 3));
    let mut lu = Array1::<f64>::zeros(nu);
    let mut ld = Array1::<f64>::zeros(nd);
    let f32_arithmetic = jdata.precision_policy.is_f32("jastrow_grad_lap");
    let add2 = |a: Array2<f64>, b: Array2<f64>| {
        if f32_arithmetic {
            (a.mapv(|x| x as f32) + b.mapv(|x| x as f32)).mapv(f64::from)
        } else {
            a + b
        }
    };
    let add1 = |a: Array1<f64>, b: Array1<f64>| {
        if f32_arithmetic {
            (a.mapv(|x| x as f32) + b.mapv(|x| x as f32)).mapv(f64::from)
        } else {
            a + b
        }
    };
    if let Some(ob) = &jdata.one_body {
        let (g_up, l_up) = j1_grad_lap_one_spin(ob, r_up);
        let (g_dn, l_dn) = j1_grad_lap_one_spin(ob, r_dn);
        gu = add2(gu, g_up);
        gd = add2(gd, g_dn);
        lu = add1(lu, l_up);
        ld = add1(ld, l_dn);
    }
    if let Some(tb) = &jdata.two_body {
        let (g_up, g_dn, l_up, l_dn) = j2_grad_lap(tb, r_up, r_dn);
        gu = add2(gu, g_up);
        gd = add2(gd, g_dn);
        lu = add1(lu, l_up);
        ld = add1(ld, l_dn);
    }
    (gu, gd, lu, ld)
}

/// One-body Jastrow J1 (molecular). Mirrors `compute_Jastrow_one_body`.
///
/// For each electron and atom: exponential `-A*(1-exp(-a*c*r))/(2*a)`
/// or Pade `-Z_eff*r/(1+a*c*r)`, with `A=(2*Z_eff)^{3/4}`,
/// `c=(2*Z_eff)^{1/4}`, `Z_eff=Z-core_electrons`.
pub fn compute_jastrow_one_body(
    data: &JastrowOneBodyData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    compute_jastrow_one_body_in_zone(data, "jastrow_eval", r_up_carts, r_dn_carts)
}

/// [`compute_jastrow_one_body`] in the precision of `zone`. The whole-value ratio
/// routes pass `"jastrow_ratio"` (issue #781).
pub(crate) fn compute_jastrow_one_body_in_zone(
    data: &JastrowOneBodyData,
    zone: &str,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    if data.precision_policy.is_f32(zone) {
        return compute_jastrow_one_body_f32(data, r_up_carts, r_dn_carts) as f64;
    }
    let a = data.jastrow_1b_param;
    let r_atoms = data.structure_data.positions.view();
    let n_atoms = data.structure_data.n_atoms();
    let is_exp = match data.jastrow_1b_type.as_str() {
        "exp" => true,
        "pade" => false,
        other => panic!("Unknown jastrow_1b_type: {other}"),
    };

    let radial: Vec<_> = (0..n_atoms)
        .map(|at| {
            let z_eff = data.structure_data.atomic_numbers[at] - data.core_electrons[at];
            (
                (2.0 * z_eff).powf(0.25),
                if is_exp {
                    -(2.0 * z_eff).powf(0.75)
                } else {
                    -2.0 * z_eff
                },
            )
        })
        .collect();
    let atom_contrib = |r_en: f64, (coeff, prefac): (f64, f64)| -> f64 {
        let kernel = if is_exp {
            1.0 / (2.0 * a) * one_minus_exp_neg(a * coeff * r_en)
        } else {
            r_en / (2.0 * (1.0 + a * coeff * r_en))
        };
        prefac * kernel
    };

    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let spin_sum = |elec: &ArrayView2<f64>| -> f64 {
        let mut s = 0.0_f64;
        for e in 0..elec.nrows() {
            for at in 0..n_atoms {
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
                    None => distance(elec, e, &r_atoms, at),
                };
                s += atom_contrib(r_en, radial[at]);
            }
        }
        s
    };
    spin_sum(r_up_carts) + spin_sum(r_dn_carts)
}

fn compute_jastrow_one_body_f32(
    data: &JastrowOneBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f32 {
    let a = data.jastrow_1b_param as f32;
    // Resolve the functional form once, not per electron-nucleus pair.
    let is_exp = data.jastrow_1b_type == "exp";
    let atoms = data.structure_data.positions.view();
    let radial: Vec<_> = (0..atoms.nrows())
        .map(|at| {
            let z_eff = (data.structure_data.atomic_numbers[at] - data.core_electrons[at]) as f32;
            (
                (2.0 * z_eff).powf(0.25),
                if is_exp {
                    -(2.0 * z_eff).powf(0.75)
                } else {
                    -2.0 * z_eff
                },
            )
        })
        .collect();
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let spin_sum = |r: &ArrayView2<f64>| -> f32 {
        // Match the existing XLA short-molecular reduction boundaries: first
        // atoms per electron, then electrons per spin. Flattening these sums
        // loses different fp32 bits in J(r') - J(r), amplified by exp(delta J)
        // in the full-recompute ECP path. Full/PBC/long reductions stay below.
        if ci.is_none() && atoms.nrows() <= 8 && r.nrows() <= 8 {
            // Eight is the XLA _sum_contributions short-axis boundary, not
            // a fixture size. Unroll that fixed tree without a reduction loop.
            let reduce = |v: [f32; 8], len: usize| -> f32 {
                match len {
                    0 => 0.0,
                    1 => v[0],
                    2 => v[0] + v[1],
                    3 | 4 => (v[0] + v[1]) + (v[2] + v[3]),
                    _ => ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])),
                }
            };
            let mut electrons = [0.0_f32; 8];
            for e in 0..r.nrows() {
                let mut terms = [0.0_f32; 8];
                for at in 0..atoms.nrows() {
                    let x = (r[[e, 0]] - atoms[[at, 0]]) as f32;
                    let y = (r[[e, 1]] - atoms[[at, 1]]) as f32;
                    let z = (r[[e, 2]] - atoms[[at, 2]]) as f32;
                    let dist = (x * x + y * y + z * z).sqrt();
                    let (coeff, prefac) = radial[at];
                    let kernel = if is_exp {
                        (1.0 / (2.0 * a)) * one_minus_exp_neg_f32(a * coeff * dist)
                    } else {
                        dist / (2.0 * (1.0 + a * coeff * dist))
                    };
                    terms[at] = prefac * kernel;
                }
                electrons[e] = reduce(terms, atoms.nrows());
            }
            return reduce(electrons, r.nrows());
        }
        let mut sum = 0.0_f32;
        for e in 0..r.nrows() {
            for at in 0..atoms.nrows() {
                // r - R in fp64, downcast afterwards (XLA `compute_Jastrow_one_body`).
                let diff = [
                    r[[e, 0]] - atoms[[at, 0]],
                    r[[e, 1]] - atoms[[at, 1]],
                    r[[e, 2]] - atoms[[at, 2]],
                ];
                let dist = match &ci {
                    Some(ci) => smooth_dist_value_f32(diff, ci),
                    None => {
                        let (x, y, z) = (diff[0] as f32, diff[1] as f32, diff[2] as f32);
                        (x * x + y * y + z * z).sqrt()
                    }
                };
                let (coeff, prefac) = radial[at];
                let kernel = if is_exp {
                    (1.0 / (2.0 * a)) * one_minus_exp_neg_f32(a * coeff * dist)
                } else {
                    dist / (2.0 * (1.0 + a * coeff * dist))
                };
                sum += prefac * kernel;
            }
        }
        sum
    };
    spin_sum(r_up) + spin_sum(r_dn)
}

/// Two-body Jastrow J2 (molecular). Mirrors `compute_Jastrow_two_body`.
///
/// `f(a, r_ee)`: pade `r/2 / (1 + a r)`, exp `1/(2a)(1 - exp(-a r))`.
/// Sums anti-parallel (all upxdn) + parallel up (i<j) + parallel dn (i<j).
pub fn compute_jastrow_two_body(
    data: &JastrowTwoBodyData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    compute_jastrow_two_body_in_zone(data, "jastrow_eval", r_up_carts, r_dn_carts)
}

/// [`compute_jastrow_two_body`] in the precision of `zone`. The whole-value ratio
/// routes pass `"jastrow_ratio"` (issue #781).
pub(crate) fn compute_jastrow_two_body_in_zone(
    data: &JastrowTwoBodyData,
    zone: &str,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
) -> f64 {
    if data.precision_policy.is_f32(zone) {
        return compute_jastrow_two_body_f32(data, r_up_carts, r_dn_carts) as f64;
    }
    let a = data.jastrow_2b_param;
    let is_exp = match data.jastrow_2b_type.as_str() {
        "exp" => true,
        "pade" => false,
        other => panic!("Unknown jastrow_2b_type: {other}"),
    };
    let f = |r_ij: f64| -> f64 {
        if is_exp {
            1.0 / (2.0 * a) * one_minus_exp_neg(a * r_ij)
        } else {
            r_ij / 2.0 * (1.0 + a * r_ij).powi(-1)
        }
    };

    let ci = data.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dist = |a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize| -> f64 {
        match &ci {
            Some(ci) => {
                smooth_dist(
                    [
                        a[[i, 0]] - b[[j, 0]],
                        a[[i, 1]] - b[[j, 1]],
                        a[[i, 2]] - b[[j, 2]],
                    ],
                    ci,
                )
                .0
            }
            None => distance(a, i, b, j),
        }
    };

    // anti-parallel: all up x dn pairs
    let mut acc = 0.0_f64;
    for i in 0..r_up_carts.nrows() {
        for j in 0..r_dn_carts.nrows() {
            acc += f(dist(r_up_carts, i, r_dn_carts, j));
        }
    }
    // parallel within each spin: i < j
    let parallel_sum = |elec: &ArrayView2<f64>| -> f64 {
        let n = elec.nrows();
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                s += f(dist(elec, i, elec, j));
            }
        }
        s
    };
    acc + parallel_sum(r_up_carts) + parallel_sum(r_dn_carts)
}

fn compute_jastrow_two_body_f32(
    data: &JastrowTwoBodyData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> f32 {
    let a = data.jastrow_2b_param as f32;
    let ci = data.cell.as_ref().map(|c| CellInfo::new(&c.view()));
    let dist = |x: &ArrayView2<f64>, i: usize, y: &ArrayView2<f64>, j: usize| -> f32 {
        match &ci {
            Some(ci) => smooth_dist_value_f32(
                [
                    x[[i, 0]] - y[[j, 0]],
                    x[[i, 1]] - y[[j, 1]],
                    x[[i, 2]] - y[[j, 2]],
                ],
                ci,
            ),
            None => distance_f32(x, i, y, j),
        }
    };
    // Resolve the functional form once, not per electron pair.
    let is_exp = data.jastrow_2b_type == "exp";
    let value = |d: f32| -> f32 {
        if is_exp {
            (1.0 / (2.0 * a)) * one_minus_exp_neg_f32(a * d)
        } else {
            (d / 2.0) * (1.0 + a * d).recip()
        }
    };
    let mut sum = 0.0_f32;
    for i in 0..r_up.nrows() {
        for j in 0..r_dn.nrows() {
            sum += value(dist(r_up, i, r_dn, j));
        }
    }
    let mut parallel = |spin: &ArrayView2<f64>| {
        for i in 0..spin.nrows() {
            for j in i + 1..spin.nrows() {
                sum += value(dist(spin, i, spin, j));
            }
        }
    };
    parallel(r_up);
    parallel(r_dn);
    sum
}

/// Three-body Jastrow J3 bilinear value. Mirrors `compute_Jastrow_three_body`.
///
/// `aos_*` have shape `(orb_num, n_elec)` (AO/MO value per electron, real).
/// `j_matrix` has shape `(orb_num, orb_num + 1)`: the square block
/// `j_matrix[:, :-1]` is the J3 coupling, the last column is the J1-like vector.
pub fn compute_jastrow_three_body_value(
    j_matrix: &ArrayView2<f64>,
    aos_up: &ArrayView2<f64>,
    aos_dn: &ArrayView2<f64>,
) -> f64 {
    compute_jastrow_three_body_value_with_policy(
        j_matrix,
        aos_up,
        aos_dn,
        crate::standalone_precision_policy().is_f32("jastrow_eval"),
    )
}

pub(crate) fn compute_jastrow_three_body_value_with_policy(
    j_matrix: &ArrayView2<f64>,
    aos_up: &ArrayView2<f64>,
    aos_dn: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> f64 {
    if f32_arithmetic {
        return compute_jastrow_three_body_value_f32(j_matrix, aos_up, aos_dn) as f64;
    }
    let orb_num = j_matrix.nrows();
    debug_assert_eq!(j_matrix.ncols(), orb_num + 1);
    let n_up = aos_up.ncols();
    let n_dn = aos_dn.ncols();

    // j1 vector = last column; j3 block = j_matrix[:, :-1].
    let j1_dot = |aos: &ArrayView2<f64>, i: usize| -> f64 {
        let mut s = 0.0;
        for al in 0..orb_num {
            s += j_matrix[[al, orb_num]] * aos[[al, i]];
        }
        s
    };
    // aos_i^T J3 aos_j over the square block.
    let bilinear = |a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize| -> f64 {
        let mut s = 0.0;
        for al in 0..orb_num {
            let ai = a[[al, i]];
            if ai == 0.0 {
                continue;
            }
            let mut row = 0.0;
            for bm in 0..orb_num {
                row += j_matrix[[al, bm]] * b[[bm, j]];
            }
            s += ai * row;
        }
        s
    };

    let mut j3 = 0.0_f64;
    // J1-like terms
    for i in 0..n_up {
        j3 += j1_dot(aos_up, i);
    }
    for i in 0..n_dn {
        j3 += j1_dot(aos_dn, i);
    }
    // parallel up (i<j), parallel dn (i<j)
    for i in 0..n_up {
        for j in (i + 1)..n_up {
            j3 += bilinear(aos_up, i, aos_up, j);
        }
    }
    for i in 0..n_dn {
        for j in (i + 1)..n_dn {
            j3 += bilinear(aos_dn, i, aos_dn, j);
        }
    }
    // up x dn (all pairs)
    for i in 0..n_up {
        for j in 0..n_dn {
            j3 += bilinear(aos_up, i, aos_dn, j);
        }
    }
    j3
}

fn compute_jastrow_three_body_value_f32(
    j_matrix: &ArrayView2<f64>,
    aos_up: &ArrayView2<f64>,
    aos_dn: &ArrayView2<f64>,
) -> f32 {
    let n = j_matrix.nrows();
    let j1_dot = |aos: &ArrayView2<f64>, i: usize| -> f32 {
        let mut s = 0.0_f32;
        for a in 0..n {
            s += j_matrix[[a, n]] as f32 * aos[[a, i]] as f32;
        }
        s
    };
    let bilinear = |a: &ArrayView2<f64>, i: usize, b: &ArrayView2<f64>, j: usize| -> f32 {
        let mut s = 0.0_f32;
        for x in 0..n {
            let mut row = 0.0_f32;
            for y in 0..n {
                row += j_matrix[[x, y]] as f32 * b[[y, j]] as f32;
            }
            s += a[[x, i]] as f32 * row;
        }
        s
    };
    let mut out = 0.0_f32;
    for i in 0..aos_up.ncols() {
        out += j1_dot(aos_up, i);
    }
    for i in 0..aos_dn.ncols() {
        out += j1_dot(aos_dn, i);
    }
    for i in 0..aos_up.ncols() {
        for j in i + 1..aos_up.ncols() {
            out += bilinear(aos_up, i, aos_up, j);
        }
    }
    for i in 0..aos_dn.ncols() {
        for j in i + 1..aos_dn.ncols() {
            out += bilinear(aos_dn, i, aos_dn, j);
        }
    }
    for i in 0..aos_up.ncols() {
        for j in 0..aos_dn.ncols() {
            out += bilinear(aos_up, i, aos_dn, j);
        }
    }
    out
}

/// J1*J2 Jastrow ratio `exp(J(r') - J(r))` for a single-electron move
/// (molecular path). Mirrors the J1+J2 blocks of
/// `_compute_ratio_Jastrow_part_rank1_update` (jastrow_factor.py).
///
/// `is_up` / `moved_index` select the moved electron; `new_pos` is its proposed
/// position. `r_up_carts`/`r_dn_carts` are the current (old) configuration.
pub fn jastrow_1b2b_ratio(
    data: &JastrowData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
    is_up: bool,
    moved_index: usize,
    new_pos: [f64; 3],
) -> f64 {
    if data.precision_policy.is_f32("jastrow_ratio") {
        return jastrow_1b2b_ratio_f32(data, r_up_carts, r_dn_carts, is_up, moved_index, new_pos)
            as f64;
    }
    jastrow_1b2b_delta(data, r_up_carts, r_dn_carts, is_up, moved_index, new_pos).exp()
}

/// Logarithm of [`jastrow_1b2b_ratio`], exposed to fused proposal kernels so
/// J1/J2 and J3 deltas can share a single exponential.
pub(crate) fn jastrow_1b2b_delta(
    data: &JastrowData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
    is_up: bool,
    moved_index: usize,
    new_pos: [f64; 3],
) -> f64 {
    // old position of the moved electron (branch -- avoids unifying the two
    // independently-borrowed array views into one lifetime).
    let r_old = if is_up {
        [
            r_up_carts[[moved_index, 0]],
            r_up_carts[[moved_index, 1]],
            r_up_carts[[moved_index, 2]],
        ]
    } else {
        [
            r_dn_carts[[moved_index, 0]],
            r_dn_carts[[moved_index, 1]],
            r_dn_carts[[moved_index, 2]],
        ]
    };

    // DeltaJ1: only the moved electron's electron-nucleus terms change.
    let mut delta = 0.0_f64;
    if let Some(j1) = &data.one_body {
        delta += j1_single_electron(j1, new_pos) - j1_single_electron(j1, r_old);
    }

    // DeltaJ2: pairs of the moved electron with every other electron (both spins).
    if let Some(j2) = &data.two_body {
        // PBC: the e-e distance must use the Williamson smooth periodic distance
        // (matching compute_Jastrow_two_body), else the ratio disagrees with the
        // full recompute -- the same fix as jqmc's _compute_ratio_Jastrow_part_*.
        let ci = j2.cell.as_ref().map(|c| CellInfo::new(&c.view()));
        let ci_ref = ci.as_ref();
        if is_up {
            // same-spin partners exclude self (self-pair J2(0) cancels)
            delta += j2_delta_block(j2, new_pos, r_old, r_up_carts, Some(moved_index), ci_ref);
            delta += j2_delta_block(j2, new_pos, r_old, r_dn_carts, None, ci_ref);
        } else {
            delta += j2_delta_block(j2, new_pos, r_old, r_dn_carts, Some(moved_index), ci_ref);
            delta += j2_delta_block(j2, new_pos, r_old, r_up_carts, None, ci_ref);
        }
    }

    delta
}

/// [`jastrow_1b2b_ratio`] for one moved electron at every row of `mesh`
/// (`out[v]` = the ratio at `mesh[v]`), bitwise identical to calling it per
/// point. The old-position terms (J1 at `r_old`, every J2 pair at `r_old`) do
/// not depend on the new position, so they are evaluated once here instead of
/// once per point; each point then adds the same `new - old` terms in the same
/// order. The fp32 `jastrow_ratio` policy uses the per-point routine.
pub fn jastrow_1b2b_ratio_mesh(
    data: &JastrowData,
    r_up_carts: &ArrayView2<f64>,
    r_dn_carts: &ArrayView2<f64>,
    is_up: bool,
    moved_index: usize,
    mesh: &ArrayView2<f64>,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), mesh.nrows());
    if data.precision_policy.is_f32("jastrow_ratio") {
        for (v, o) in out.iter_mut().enumerate() {
            let p = [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]];
            *o = jastrow_1b2b_ratio(data, r_up_carts, r_dn_carts, is_up, moved_index, p);
        }
        return;
    }
    if is_up {
        j12_ratio_mesh_impl(data, r_up_carts, r_dn_carts, moved_index, mesh, out);
    } else {
        j12_ratio_mesh_impl(data, r_dn_carts, r_up_carts, moved_index, mesh, out);
    }
}

/// Body of [`jastrow_1b2b_ratio_mesh`]: `same` holds the moved electron's spin.
fn j12_ratio_mesh_impl(
    data: &JastrowData,
    same: &ArrayView2<'_, f64>,
    other: &ArrayView2<'_, f64>,
    moved_index: usize,
    mesh: &ArrayView2<'_, f64>,
    out: &mut [f64],
) {
    let r_old = [
        same[[moved_index, 0]],
        same[[moved_index, 1]],
        same[[moved_index, 2]],
    ];
    // Per-call constants of `j1_single_electron` / `j2_from_dist` (cell
    // geometry, per-atom powf factors, form tag), built once per mesh.
    let j1p = data.one_body.as_ref().map(J1Prep::new);
    let j1_old = j1p.as_ref().map(|p| p.eval(r_old));
    let ci = data
        .two_body
        .as_ref()
        .and_then(|j2| j2.cell.as_ref().map(|c| CellInfo::new(&c.view())));
    let j2k = data
        .two_body
        .as_ref()
        .map(|j2| (j2.jastrow_2b_param, j2.jastrow_2b_type == "exp"));
    // Old-side J2 pair values, partner order as `j2_delta_block`.
    let (mut old_same, mut old_other) = (Vec::new(), Vec::new());
    if let Some((a, is_exp)) = j2k {
        for k in 0..same.nrows() {
            old_same.push(j2_from_dist_kind(
                a,
                is_exp,
                j2_pair_dist(r_old, same, k, ci.as_ref()),
            ));
        }
        for k in 0..other.nrows() {
            old_other.push(j2_from_dist_kind(
                a,
                is_exp,
                j2_pair_dist(r_old, other, k, ci.as_ref()),
            ));
        }
    }
    for (v, o) in out.iter_mut().enumerate() {
        let new_pos = [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]];
        let mut delta = 0.0_f64;
        if let (Some(p), Some(j1o)) = (&j1p, j1_old) {
            delta += p.eval(new_pos) - j1o;
        }
        if let Some((a, is_exp)) = j2k {
            let mut s = 0.0_f64;
            for (k, old_same_k) in old_same[..same.nrows()].iter().enumerate() {
                if k == moved_index {
                    continue;
                }
                s += j2_from_dist_kind(a, is_exp, j2_pair_dist(new_pos, same, k, ci.as_ref()))
                    - *old_same_k;
            }
            delta += s;
            let mut s = 0.0_f64;
            for (k, old_other_k) in old_other[..other.nrows()].iter().enumerate() {
                s += j2_from_dist_kind(a, is_exp, j2_pair_dist(new_pos, other, k, ci.as_ref()))
                    - *old_other_k;
            }
            delta += s;
        }
        *o = delta.exp();
    }
}

/// Distance of `p` to partner row `k` as [`j2_delta_block`] measures it.
#[inline]
fn j2_pair_dist(
    p: [f64; 3],
    partners: &ArrayView2<'_, f64>,
    k: usize,
    ci: Option<&CellInfo>,
) -> f64 {
    match ci {
        Some(ci) => {
            let q = [partners[[k, 0]], partners[[k, 1]], partners[[k, 2]]];
            smooth_dist([p[0] - q[0], p[1] - q[1], p[2] - q[2]], ci).0
        }
        None => dist_point_row(p, partners, k),
    }
}

// ============================================================================
// FAST periodic J1/J2 value route.
//
// The exact periodic J1/J2 ratio spends most of its time in three libm
// `sin_cos` calls per pair (`positive_geometry_kernel`). The FAST (native
// energy chunk, `cutoff` / `ao_cutoff`, floor-compared) callers instead build a
// [`J12FastPeriodic`] once and evaluate the pair and atom sums in blocks of
// `FAST_J12_LANES` partners with the branch-free `crate::fastmath::sincos`.
// Same closed algebra as `smooth_dist`(..).0 and `j2_from_dist_kind` /
// `J1Prep::eval`, with the same cancellation-safe exponential radial form.
// The trigonometric functions, the coalescence branch (a select here) and
// the summation order differ. Accuracy: `fast_periodic_j12_matches_exact`
// (central `ATOL_FASTMATH_PERIODIC_SMOOTH_DIST` /
// `RTOL_FASTMATH_PERIODIC_J12_RATIO`). The exact routes never come here.
// ============================================================================

/// Partners (or atoms) per auto-vectorized block of the FAST periodic route.
const FAST_J12_LANES: usize = 8;

/// FAST value-only twin of `smooth_dist(x, ci).0` (see the section note).
#[inline(always)]
fn smooth_dist_value_fast(x: [f64; 3], ci: &CellInfo) -> f64 {
    use std::f64::consts::PI;
    // Round-half-even via the 1.5*2^52 shifter (as `fastmath::sincos`), equal
    // to `round_ties_even` for the |x . inv| << 2^51 of electron coordinates.
    const SHIFTER: f64 = 6_755_399_441_055_744.0;
    let (cell, inv) = (&ci.cell, &ci.inv);
    let image: [f64; 3] = std::array::from_fn(|i| {
        ((x[0] * inv[0][i] + x[1] * inv[1][i] + x[2] * inv[2][i]) + SHIFTER) - SHIFTER
    });
    let local: [f64; 3] = std::array::from_fn(|k| {
        x[k] - (image[0] * cell[0][k] + image[1] * cell[1][k] + image[2] * cell[2][k])
    });
    let theta: [f64; 3] = std::array::from_fn(|i| {
        PI * (local[0] * inv[0][i] + local[1] * inv[1][i] + local[2] * inv[2][i])
    });
    let mut z = [0.0; 3];
    let mut h = [0.0; 3];
    for i in 0..3 {
        let (sn, cs) = crate::fastmath::sincos(theta[i]);
        z[i] = sn * sn;
        h[i] = sn * cs;
    }
    let small_bound = f64::EPSILON.sqrt().sqrt();
    let small = (theta[0] * theta[0] < small_bound)
        & (theta[1] * theta[1] < small_bound)
        & (theta[2] * theta[2] < small_bound);
    let correction: [f64; 3] = theta.map(|value| {
        let square = value * value;
        value
            * square
            * (-2.0 / 3.0
                + square * (2.0 / 15.0 + square * (-4.0 / 315.0 + square * (2.0 / 2835.0))))
            / PI
    });
    let mut square = 0.0;
    for k in 0..3 {
        let near = local[k]
            + correction[0] * cell[0][k]
            + correction[1] * cell[1][k]
            + correction[2] * cell[2][k];
        let far = (h[0] * cell[0][k] + h[1] * cell[1][k] + h[2] * cell[2][k]) / PI;
        let c = if small { near } else { far };
        square += c * c;
    }
    for (i, z_i) in z.iter().enumerate() {
        square += ci.gram[i][i] * *z_i * *z_i / (PI * PI);
    }
    let d = square.sqrt();
    if d > EPS_SAFE_DISTANCE {
        d
    } else {
        EPS_SAFE_DISTANCE
    }
}

/// Radial form shared by the J1 atom and J2 pair terms, without its
/// prefactor: `1 - exp(-b d)` (exp) or `d / (1 + b d)` (pade).
#[inline(always)]
fn j12_form_fast<const EXP: bool>(b: f64, d: f64) -> f64 {
    if EXP {
        one_minus_exp_neg(b * d)
    } else {
        d / (1.0 + b * d)
    }
}

/// Padded structure-of-arrays term list: term `l` contributes
/// `w[l] * form(b[l], dist(p - (x[l], y[l], z[l])))`; padding lanes have `w = 0`.
#[derive(Default)]
struct FastTermSoa {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
    b: Vec<f64>,
}

impl FastTermSoa {
    fn push(&mut self, q: [f64; 3], w: f64, b: f64) {
        self.x.push(q[0]);
        self.y.push(q[1]);
        self.z.push(q[2]);
        self.w.push(w);
        self.b.push(b);
    }

    fn pad(&mut self) {
        while !self.w.len().is_multiple_of(FAST_J12_LANES) {
            self.push([0.0; 3], 0.0, 0.0);
        }
    }

    /// `sum_l w[l] * form(b[l], dist(p - q_l))`.
    #[inline(always)]
    fn sum<const EXP: bool>(&self, p: [f64; 3], ci: &CellInfo) -> f64 {
        let mut acc = [0.0_f64; FAST_J12_LANES];
        for ((((xs, ys), zs), ws), bs) in self
            .x
            .chunks_exact(FAST_J12_LANES)
            .zip(self.y.chunks_exact(FAST_J12_LANES))
            .zip(self.z.chunks_exact(FAST_J12_LANES))
            .zip(self.w.chunks_exact(FAST_J12_LANES))
            .zip(self.b.chunks_exact(FAST_J12_LANES))
        {
            for l in 0..FAST_J12_LANES {
                let d = smooth_dist_value_fast([p[0] - xs[l], p[1] - ys[l], p[2] - zs[l]], ci);
                acc[l] += ws[l] * j12_form_fast::<EXP>(bs[l], d);
            }
        }
        acc.iter().sum()
    }

    fn sum_dyn(&self, exp_form: bool, p: [f64; 3], ci: &CellInfo) -> f64 {
        if exp_form {
            self.sum::<true>(p, ci)
        } else {
            self.sum::<false>(p, ci)
        }
    }
}

/// Per-sweep / per-e_L constants of the FAST periodic J1/J2 value route.
pub(crate) struct J12FastPeriodic {
    /// J1 atom terms (`w = prefac / (2a)` exp, `prefac / 2` pade; `b = a * coeff`).
    j1: Option<(CellInfo, bool, FastTermSoa)>,
    /// J2 cell, form tag, prefactor (`1 / (2a)` exp, `1 / 2` pade) and `b = a`.
    j2: Option<(CellInfo, bool, f64, f64)>,
}

impl J12FastPeriodic {
    /// `Some` only for an fp64 `jastrow_ratio` policy with at least one J1/J2
    /// term, every present term periodic. Callers construct it only on their
    /// FAST route; `None` keeps the exact routine.
    pub(crate) fn new(data: &JastrowData) -> Option<Self> {
        if data.precision_policy.is_f32("jastrow_ratio") {
            return None;
        }
        if data.one_body.is_none() && data.two_body.is_none() {
            return None;
        }
        let j1 = match &data.one_body {
            None => None,
            Some(ob) => {
                let cell = ob.structure_data.cell.as_ref()?;
                let ci = CellInfo::new(&cell.view());
                let is_exp = ob.jastrow_1b_type == "exp";
                let a = ob.jastrow_1b_param;
                let pos = &ob.structure_data.positions;
                let mut soa = FastTermSoa::default();
                for at in 0..ob.structure_data.n_atoms() {
                    let z_eff = ob.structure_data.atomic_numbers[at] - ob.core_electrons[at];
                    let coeff = (2.0 * z_eff).powf(0.25);
                    let (w, b) = if is_exp {
                        (-(2.0 * z_eff).powf(0.75) / (2.0 * a), a * coeff)
                    } else {
                        (-2.0 * z_eff / 2.0, a * coeff)
                    };
                    soa.push([pos[[at, 0]], pos[[at, 1]], pos[[at, 2]]], w, b);
                }
                soa.pad();
                Some((ci, is_exp, soa))
            }
        };
        let j2 = match &data.two_body {
            None => None,
            Some(tb) => {
                let cell = tb.cell.as_ref()?;
                let is_exp = tb.jastrow_2b_type == "exp";
                let a = tb.jastrow_2b_param;
                let w = if is_exp { 1.0 / (2.0 * a) } else { 0.5 };
                Some((CellInfo::new(&cell.view()), is_exp, w, a))
            }
        };
        Some(Self { j1, j2 })
    }

    #[inline]
    fn j1_value(&self, p: [f64; 3]) -> f64 {
        match &self.j1 {
            Some((ci, is_exp, soa)) => soa.sum_dyn(*is_exp, p, ci),
            None => 0.0,
        }
    }

    /// FAST twin of [`jastrow_1b2b_delta`] (same arguments).
    pub(crate) fn delta(
        &self,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        moved: usize,
        new_pos: [f64; 3],
    ) -> f64 {
        let (same, other) = if is_up {
            (r_up.view(), r_dn.view())
        } else {
            (r_dn.view(), r_up.view())
        };
        let r_old = [same[[moved, 0]], same[[moved, 1]], same[[moved, 2]]];
        let mut delta = 0.0;
        if self.j1.is_some() {
            delta += self.j1_value(new_pos) - self.j1_value(r_old);
        }
        if let Some((ci, is_exp, w, b)) = &self.j2 {
            let s = if *is_exp {
                j2_pair_delta_fast::<true>(ci, *b, new_pos, r_old, &same, Some(moved))
                    + j2_pair_delta_fast::<true>(ci, *b, new_pos, r_old, &other, None)
            } else {
                j2_pair_delta_fast::<false>(ci, *b, new_pos, r_old, &same, Some(moved))
                    + j2_pair_delta_fast::<false>(ci, *b, new_pos, r_old, &other, None)
            };
            delta += w * s;
        }
        delta
    }

    /// FAST twin of [`jastrow_1b2b_ratio_mesh`] (same arguments).
    pub(crate) fn ratio_mesh(
        &self,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        moved: usize,
        mesh: &ArrayView2<f64>,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), mesh.nrows());
        let (same, other) = if is_up {
            (r_up.view(), r_dn.view())
        } else {
            (r_dn.view(), r_up.view())
        };
        let r_old = [same[[moved, 0]], same[[moved, 1]], same[[moved, 2]]];
        // Every partner of the moved electron (both spins, self excluded),
        // padded once; its old-side sum is shared by all mesh points.
        let j2 = self.j2.as_ref().map(|(ci, is_exp, w, b)| {
            let mut soa = FastTermSoa::default();
            for k in 0..same.nrows() {
                if k != moved {
                    soa.push([same[[k, 0]], same[[k, 1]], same[[k, 2]]], 1.0, *b);
                }
            }
            for k in 0..other.nrows() {
                soa.push([other[[k, 0]], other[[k, 1]], other[[k, 2]]], 1.0, *b);
            }
            soa.pad();
            let old = soa.sum_dyn(*is_exp, r_old, ci);
            (ci, *is_exp, *w, soa, old)
        });
        let j1_old = self.j1_value(r_old);
        for (v, o) in out.iter_mut().enumerate() {
            let p = [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]];
            let mut delta = 0.0;
            if self.j1.is_some() {
                delta += self.j1_value(p) - j1_old;
            }
            if let Some((ci, is_exp, w, soa, old)) = &j2 {
                delta += w * (soa.sum_dyn(*is_exp, p, ci) - old);
            }
            *o = delta.exp();
        }
    }
}

/// `sum_k [form(b, dist(new - q_k)) - form(b, dist(old - q_k))]` over the rows
/// `q_k` of `partners` except `exclude`, gathered into stack blocks.
#[inline(always)]
fn j2_pair_delta_fast<const EXP: bool>(
    ci: &CellInfo,
    b: f64,
    new_pos: [f64; 3],
    r_old: [f64; 3],
    partners: &ArrayView2<f64>,
    exclude: Option<usize>,
) -> f64 {
    let n = partners.nrows();
    let mut acc = [0.0_f64; FAST_J12_LANES];
    let mut k0 = 0;
    while k0 < n {
        let mut q = [[0.0_f64; FAST_J12_LANES]; 3];
        let mut m = [0.0_f64; FAST_J12_LANES];
        for l in 0..FAST_J12_LANES.min(n - k0) {
            let k = k0 + l;
            for c in 0..3 {
                q[c][l] = partners[[k, c]];
            }
            m[l] = if exclude == Some(k) { 0.0 } else { 1.0 };
        }
        for l in 0..FAST_J12_LANES {
            let dn = smooth_dist_value_fast(
                [
                    new_pos[0] - q[0][l],
                    new_pos[1] - q[1][l],
                    new_pos[2] - q[2][l],
                ],
                ci,
            );
            let d_old = smooth_dist_value_fast(
                [r_old[0] - q[0][l], r_old[1] - q[1][l], r_old[2] - q[2][l]],
                ci,
            );
            acc[l] += m[l] * (j12_form_fast::<EXP>(b, dn) - j12_form_fast::<EXP>(b, d_old));
        }
        k0 += FAST_J12_LANES;
    }
    acc.iter().sum()
}

/// J1/J2 deltas for one LRDMC candidate group.  The projection-scoped tabpip
/// table owns reuse across hops; this leaf deliberately follows the common
/// scalar evaluator for every J1/J2 form and boundary condition.
pub(crate) fn jastrow_1b2b_deltas_into(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    moved: usize,
    new_pos: &[[f64; 3]],
    out: &mut [f64],
) {
    debug_assert_eq!(new_pos.len(), out.len());
    for (dst, &pos) in out.iter_mut().zip(new_pos) {
        *dst = if data.precision_policy.is_f32("jastrow_ratio") {
            f64::from(jastrow_1b2b_delta_f32(data, r_up, r_dn, is_up, moved, pos))
        } else {
            jastrow_1b2b_delta(data, r_up, r_dn, is_up, moved, pos)
        };
    }
}

/// Whether a projection candidate lattice has an active J1/J2 channel to
/// cache.  This is representation-, boundary-, form-, and
/// precision-independent.
#[inline]
pub(crate) fn j12_candidate_cache_ok(data: &JastrowData) -> bool {
    data.one_body.is_some() || data.two_body.is_some()
}

/// Projection-scoped generic J1/J2 derivative table. A GFMC hop changes only
/// pair terms involving the moved electron, so
/// all gradients and Laplacians can be advanced in O(N) rather than rebuilt by
/// the O(N^2) all-pairs kernel at every sub-step.
pub(crate) struct J12GradLapCache {
    gu: Array2<f64>,
    gd: Array2<f64>,
    lu: Array1<f64>,
    ld: Array1<f64>,
}

pub(crate) fn clone_j12_grad_lap_cache(
    cache: &J12GradLapCache,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    (
        cache.gu.clone(),
        cache.gd.clone(),
        cache.lu.clone(),
        cache.ld.clone(),
    )
}

/// Coverage gate for the generic J1/J2 grad/lap cache:
/// any J1/J2 term, molecular or periodic, exp or pade. The delta chain may
/// accumulate in a different order from a full mixed-precision rebuild, but
/// that must not disable the production cache.
pub(crate) fn j12_generic_cache_ok(data: &JastrowData) -> bool {
    data.one_body.is_some() || data.two_body.is_some()
}

/// Seed the [`J12GradLapCache`] from the J1+J2 legs only
/// ([`compute_grads_and_laplacian_j12_part`] -- DTN is streamed separately on
/// the projection chain), for every supported form and boundary condition.
pub(crate) fn init_j12_grad_lap_cache_generic(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> J12GradLapCache {
    debug_assert!(j12_generic_cache_ok(data));
    let (gu, gd, lu, ld) = compute_grads_and_laplacian_j12_part(data, r_up, r_dn);
    J12GradLapCache { gu, gd, lu, ld }
}

/// Per-hop advance of the generic J1/J2 cache.
/// - J1 is a per-electron atom sum, so only the moved electron's row changes:
///   the delta of two 1-row [`j1_grad_lap_one_spin`] evaluations (exact same
///   kernel as the seed's rows).
/// - J2 pair deltas run the seed's own [`j2_pair`] kernel with the fixture's
///   actual form (`exp`/`pade`) and cell (periodic Williamson smooth metric
///   when present). Partner rows use the moved-side pair evaluation (gradient
///   odd, Laplacian even in the separation -- the symmetry the full kernel's
///   two-sided loops evaluate from each side).
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_j12_grad_lap_cache_generic(
    data: &JastrowData,
    cache: &mut J12GradLapCache,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    moved_up: bool,
    moved: usize,
    old_pos: [f64; 3],
) {
    debug_assert!(j12_generic_cache_ok(data));
    let new_pos = if moved_up {
        [r_up[[moved, 0]], r_up[[moved, 1]], r_up[[moved, 2]]]
    } else {
        [r_dn[[moved, 0]], r_dn[[moved, 1]], r_dn[[moved, 2]]]
    };
    if let Some(ob) = &data.one_body {
        let row = |p: [f64; 3]| {
            let r1 = Array2::from_shape_vec((1, 3), p.to_vec()).expect("one J1 point");
            j1_grad_lap_one_spin(ob, &r1.view())
        };
        let (g_new, l_new) = row(new_pos);
        let (g_old, l_old) = row(old_pos);
        let (grad, lap) = if moved_up {
            (&mut cache.gu, &mut cache.lu)
        } else {
            (&mut cache.gd, &mut cache.ld)
        };
        for axis in 0..3 {
            grad[[moved, axis]] += g_new[[0, axis]] - g_old[[0, axis]];
        }
        lap[moved] += l_new[0] - l_old[0];
    }
    if let Some(j2) = &data.two_body {
        let a = j2.jastrow_2b_param;
        let is_exp = j2.jastrow_2b_type == "exp";
        let ci = j2.cell.as_ref().map(|c| CellInfo::new(&c.view()));
        let ci_ref = ci.as_ref();
        let mut moved_dg = [0.0_f64; 3];
        let mut moved_dl = 0.0_f64;
        let mut update_partners = |positions: &ArrayView2<f64>,
                                   same_spin: bool,
                                   grad: &mut Array2<f64>,
                                   lap: &mut Array1<f64>| {
            for j in 0..positions.nrows() {
                if same_spin && j == moved {
                    continue;
                }
                let p = [positions[[j, 0]], positions[[j, 1]], positions[[j, 2]]];
                let (g_old, l_old) = j2_pair(
                    is_exp,
                    a,
                    old_pos[0] - p[0],
                    old_pos[1] - p[1],
                    old_pos[2] - p[2],
                    ci_ref,
                );
                let (g_new, l_new) = j2_pair(
                    is_exp,
                    a,
                    new_pos[0] - p[0],
                    new_pos[1] - p[1],
                    new_pos[2] - p[2],
                    ci_ref,
                );
                for axis in 0..3 {
                    let dg = g_new[axis] - g_old[axis];
                    moved_dg[axis] += dg;
                    // The partner sees the opposite pair-gradient.
                    grad[[j, axis]] -= dg;
                }
                let dl = l_new - l_old;
                moved_dl += dl;
                lap[j] += dl;
            }
        };
        if moved_up {
            update_partners(r_up, true, &mut cache.gu, &mut cache.lu);
            update_partners(r_dn, false, &mut cache.gd, &mut cache.ld);
            for (axis, moved_dg_axis) in moved_dg.iter().enumerate() {
                cache.gu[[moved, axis]] += *moved_dg_axis;
            }
            cache.lu[moved] += moved_dl;
        } else {
            update_partners(r_dn, true, &mut cache.gd, &mut cache.ld);
            update_partners(r_up, false, &mut cache.gu, &mut cache.lu);
            for (axis, moved_dg_axis) in moved_dg.iter().enumerate() {
                cache.gd[[moved, axis]] += *moved_dg_axis;
            }
            cache.ld[moved] += moved_dl;
        }
    }
}

/// Assemble a [`J3GradLapCache`] from externally built electron-major
/// derivative tables (each `(ne, n_orb)`), for callers whose AO family lives
/// outside this module, such as the complex `Re(value_grad_lap)` chain in
/// `complex_det.rs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn j3_grad_lap_cache_from_tables(
    up_x: Array2<f64>,
    up_y: Array2<f64>,
    up_z: Array2<f64>,
    up_lap: Array2<f64>,
    dn_x: Array2<f64>,
    dn_y: Array2<f64>,
    dn_z: Array2<f64>,
    dn_lap: Array2<f64>,
) -> J3GradLapCache {
    J3GradLapCache {
        up_x,
        up_y,
        up_z,
        up_lap,
        dn_x,
        dn_y,
        dn_z,
        dn_lap,
        output: None,
        shadow: None,
    }
}

/// Replace one electron's derivative row from externally evaluated columns.
/// The real-orbital twin is [`update_j3_grad_lap_cache`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn j3_grad_lap_cache_set_row(
    cache: &mut J3GradLapCache,
    is_up: bool,
    electron: usize,
    gx: &[f64],
    gy: &[f64],
    gz: &[f64],
    lap: &[f64],
) {
    cache.shadow = None;
    let assign = |dst: &mut Array2<f64>, src: &[f64]| {
        debug_assert_eq!(dst.ncols(), src.len());
        for a in 0..dst.ncols() {
            dst[[electron, a]] = src[a];
        }
    };
    if is_up {
        assign(&mut cache.up_x, gx);
        assign(&mut cache.up_y, gy);
        assign(&mut cache.up_z, gz);
        assign(&mut cache.up_lap, lap);
    } else {
        assign(&mut cache.dn_x, gx);
        assign(&mut cache.dn_y, gy);
        assign(&mut cache.dn_z, gz);
        assign(&mut cache.dn_lap, lap);
    }
}

/// Build one spin block of a projection-scoped J1/J2 log-ratio table.
///
/// This is the Rust counterpart of XLA's independent
/// `J1_candidate_mesh_state` / `J2_candidate_mesh_state`.  The two logarithms
/// share one table here because both consumers ultimately need their sum.
/// `jastrow_1b2b_deltas_into` preserves the configured ratio precision.
pub(crate) fn fill_j12_candidate_deltas(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    candidates: &[[f64; 3]],
    rows_per_electron: usize,
    out: &mut [f64],
) {
    debug_assert!(j12_candidate_cache_ok(data));
    debug_assert_eq!(candidates.len(), out.len());
    let ne = if is_up { r_up.nrows() } else { r_dn.nrows() };
    debug_assert_eq!(candidates.len(), rows_per_electron * ne);
    for e in 0..ne {
        let off = rows_per_electron * e;
        jastrow_1b2b_deltas_into(
            data,
            r_up,
            r_dn,
            is_up,
            e,
            &candidates[off..off + rows_per_electron],
            &mut out[off..off + rows_per_electron],
        );
    }
}

#[inline]
fn j2_candidate_pair(data: &JastrowTwoBodyData, p: [f64; 3], q: [f64; 3]) -> f64 {
    let diff = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let distance = match &data.cell {
        Some(cell) => smooth_dist(diff, &CellInfo::new(&cell.view())).0,
        None => (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt(),
    };
    j2_from_dist(data, distance)
}

#[inline]
fn j2_candidate_pair_f32(data: &JastrowTwoBodyData, p: [f64; 3], q: [f64; 3]) -> f32 {
    let diff = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let distance = match &data.cell {
        Some(cell) => smooth_dist_value_f32(diff, &CellInfo::new(&cell.view())),
        None => {
            let (x, y, z) = (diff[0] as f32, diff[1] as f32, diff[2] as f32);
            (x * x + y * y + z * z).sqrt()
        }
    };
    j2_value_f32(data, distance)
}

#[allow(clippy::too_many_arguments)]
fn update_j12_candidate_spin_generic(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    coords: &ArrayView2<f64>,
    is_up: bool,
    moved_up: bool,
    moved: usize,
    old_pos: [f64; 3],
    new_pos: [f64; 3],
    candidates: &[[f64; 3]],
    rows_per_electron: usize,
    deltas: &mut [f64],
    live: Option<&[bool]>,
) {
    let f32_ratio = data.precision_policy.is_f32("jastrow_ratio");
    for e in 0..coords.nrows() {
        // Rows of a dead electron are never read; they are rebuilt (mover
        // branch below) when that electron moves.
        if live.is_some_and(|l| !l[e]) {
            continue;
        }
        let off = rows_per_electron * e;
        let row = &candidates[off..off + rows_per_electron];
        let dst = &mut deltas[off..off + rows_per_electron];
        if is_up == moved_up && e == moved {
            // The mover's base position and candidate coordinates both
            // changed.  Rebuilding exactly is O(K*N), versus O(K*N^2) for the
            // whole table, and prevents accumulated correction drift.
            jastrow_1b2b_deltas_into(data, r_up, r_dn, is_up, e, row, dst);
            continue;
        }
        let Some(j2) = &data.two_body else {
            // Pure J1 has no cross-electron coupling: all non-mover rows are
            // invariant.
            continue;
        };
        let base = [coords[[e, 0]], coords[[e, 1]], coords[[e, 2]]];
        if f32_ratio {
            let old_base = j2_candidate_pair_f32(j2, base, old_pos);
            let new_base = j2_candidate_pair_f32(j2, base, new_pos);
            for (delta, &candidate) in dst.iter_mut().zip(row) {
                // Match the XLA candidate-state correction ordering:
                // u(c,new)-u(base,new)-u(c,old)+u(base,old), rounded in the
                // jastrow_ratio zone before it is added to the cached log.
                let correction = j2_candidate_pair_f32(j2, candidate, new_pos)
                    - new_base
                    - j2_candidate_pair_f32(j2, candidate, old_pos)
                    + old_base;
                *delta = f64::from((*delta as f32) + correction);
            }
        } else {
            let old_base = j2_candidate_pair(j2, base, old_pos);
            let new_base = j2_candidate_pair(j2, base, new_pos);
            for (delta, &candidate) in dst.iter_mut().zip(row) {
                *delta += j2_candidate_pair(j2, candidate, new_pos)
                    - new_base
                    - j2_candidate_pair(j2, candidate, old_pos)
                    + old_base;
            }
        }
    }
}

/// Advance the generic J1/J2 candidate table after one accepted hop using the
/// same four-term low-rank identity as XLA.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_j12_candidate_deltas_after_hop(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    moved_up: bool,
    moved: usize,
    old_pos: [f64; 3],
    candidates_up: &[[f64; 3]],
    candidates_dn: &[[f64; 3]],
    rows_per_electron: usize,
    deltas_up: &mut [f64],
    deltas_dn: &mut [f64],
) {
    update_j12_candidate_deltas_after_hop_live(
        data,
        r_up,
        r_dn,
        moved_up,
        moved,
        old_pos,
        candidates_up,
        candidates_dn,
        rows_per_electron,
        deltas_up,
        deltas_dn,
        None,
    );
}

/// [`update_j12_candidate_deltas_after_hop`] restricted to the rows of LIVE
/// electrons (`live = Some((up, dn))`, per-spin electron masks; GFMC ECP
/// zero-amplitude groups, IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001). Live rows get
/// the identical update; dead rows are left untouched (never read, rebuilt
/// when their electron moves). `None` updates every row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_j12_candidate_deltas_after_hop_live(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    moved_up: bool,
    moved: usize,
    old_pos: [f64; 3],
    candidates_up: &[[f64; 3]],
    candidates_dn: &[[f64; 3]],
    rows_per_electron: usize,
    deltas_up: &mut [f64],
    deltas_dn: &mut [f64],
    live: Option<(&[bool], &[bool])>,
) {
    debug_assert!(j12_candidate_cache_ok(data));
    let new_pos = if moved_up {
        [r_up[[moved, 0]], r_up[[moved, 1]], r_up[[moved, 2]]]
    } else {
        [r_dn[[moved, 0]], r_dn[[moved, 1]], r_dn[[moved, 2]]]
    };
    update_j12_candidate_spin_generic(
        data,
        r_up,
        r_dn,
        r_up,
        true,
        moved_up,
        moved,
        old_pos,
        new_pos,
        candidates_up,
        rows_per_electron,
        deltas_up,
        live.map(|(up, _)| up),
    );
    update_j12_candidate_spin_generic(
        data,
        r_up,
        r_dn,
        r_dn,
        false,
        moved_up,
        moved,
        old_pos,
        new_pos,
        candidates_dn,
        rows_per_electron,
        deltas_dn,
        live.map(|(_, dn)| dn),
    );
}

fn j1_single_electron_f32(data: &JastrowOneBodyData, r: [f64; 3]) -> f32 {
    let atoms = data.structure_data.positions.view();
    let ci = data
        .structure_data
        .cell
        .as_ref()
        .map(|c| CellInfo::new(&c.view()));
    let a = data.jastrow_1b_param as f32;
    let mut sum = 0.0_f32;
    for at in 0..atoms.nrows() {
        // XLA materializes the static nuclear coordinates in jastrow_eval
        // before subtracting them from the fp64 walker position, then casts
        // the reconstructed difference back to the zone dtype.
        let diff = [
            r[0] - (atoms[[at, 0]] as f32) as f64,
            r[1] - (atoms[[at, 1]] as f32) as f64,
            r[2] - (atoms[[at, 2]] as f32) as f64,
        ];
        let d = match &ci {
            Some(ci) => smooth_dist_value_f32(diff, ci),
            None => {
                let (x, y, z) = (diff[0] as f32, diff[1] as f32, diff[2] as f32);
                (x * x + y * y + z * z).sqrt()
            }
        };
        let z = (data.structure_data.atomic_numbers[at] - data.core_electrons[at]) as f32;
        let c = (2.0 * z).powf(0.25);
        let aa = if data.jastrow_1b_type == "exp" {
            -(2.0 * z).powf(0.75)
        } else {
            -2.0 * z
        };
        let u = if data.jastrow_1b_type == "exp" {
            (1.0 / (2.0 * a)) * one_minus_exp_neg_f32(a * c * d)
        } else {
            d / (2.0 * (1.0 + a * c * d))
        };
        sum += aa * u;
    }
    sum
}

fn j2_value_f32(data: &JastrowTwoBodyData, d: f32) -> f32 {
    let a = data.jastrow_2b_param as f32;
    if data.jastrow_2b_type == "exp" {
        (1.0 / (2.0 * a)) * one_minus_exp_neg_f32(a * d)
    } else {
        (d / 2.0) * (1.0 + a * d).recip()
    }
}

fn jastrow_1b2b_ratio_f32(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    moved: usize,
    new_pos: [f64; 3],
) -> f32 {
    jastrow_1b2b_delta_f32(data, r_up, r_dn, is_up, moved, new_pos).exp()
}

/// Candidate-state logarithm in the configured fp32 ratio zone.  Keeping the
/// logarithm (instead of only the exponentiated one-shot ratio) lets the
/// projection table apply XLA's low-rank J2 correction after every hop.
fn jastrow_1b2b_delta_f32(
    data: &JastrowData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    moved: usize,
    new_pos: [f64; 3],
) -> f32 {
    let old = if is_up {
        [r_up[[moved, 0]], r_up[[moved, 1]], r_up[[moved, 2]]]
    } else {
        [r_dn[[moved, 0]], r_dn[[moved, 1]], r_dn[[moved, 2]]]
    };
    let mut delta = 0.0_f32;
    if let Some(j1) = &data.one_body {
        delta += j1_single_electron_f32(j1, new_pos) - j1_single_electron_f32(j1, old);
    }
    if let Some(j2) = &data.two_body {
        let ci = j2.cell.as_ref().map(|c| CellInfo::new(&c.view()));
        // Match XLA's rank-1 route: reduce the complete new and old partner
        // blocks independently, then subtract the two rounded fp32 sums.
        // Accumulating pairwise deltas changes cancellation by several ulps on
        // triclinic cells and does not match a full fp32 recomputation.
        let mut new_sum = 0.0_f32;
        let mut old_sum = 0.0_f32;
        let mut block = |partners: &ArrayView2<f64>, exclude: Option<usize>| {
            for k in 0..partners.nrows() {
                if exclude == Some(k) {
                    continue;
                }
                let p = [partners[[k, 0]], partners[[k, 1]], partners[[k, 2]]];
                let distance = |q: [f64; 3]| -> f32 {
                    let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                    match &ci {
                        Some(ci) => smooth_dist_value_f32(d, ci),
                        None => {
                            let (x, y, z) = (d[0] as f32, d[1] as f32, d[2] as f32);
                            (x * x + y * y + z * z).sqrt()
                        }
                    }
                };
                new_sum += j2_value_f32(j2, distance(new_pos));
                old_sum += j2_value_f32(j2, distance(old));
            }
        };
        if is_up {
            block(r_up, Some(moved));
            block(r_dn, None);
        } else {
            block(r_dn, Some(moved));
            block(r_up, None);
        }
        delta += new_sum - old_sum;
    }
    delta
}

/// DeltaJ2 contribution from one partner-spin block (free fn -> higher-ranked over
/// the `partners` borrow, so callers pass r_up/r_dn with independent lifetimes).
/// `ci = Some` => periodic (Williamson smooth distance); `None` => molecular.
fn j2_delta_block(
    j2: &JastrowTwoBodyData,
    new_pos: [f64; 3],
    r_old: [f64; 3],
    partners: &ArrayView2<f64>,
    exclude: Option<usize>,
    ci: Option<&CellInfo>,
) -> f64 {
    let mut s = 0.0_f64;
    for k in 0..partners.nrows() {
        if exclude == Some(k) {
            continue;
        }
        let (d_new, d_old) = match ci {
            Some(ci) => {
                let p = [partners[[k, 0]], partners[[k, 1]], partners[[k, 2]]];
                let dn = smooth_dist(
                    [new_pos[0] - p[0], new_pos[1] - p[1], new_pos[2] - p[2]],
                    ci,
                )
                .0;
                let do_ = smooth_dist([r_old[0] - p[0], r_old[1] - p[1], r_old[2] - p[2]], ci).0;
                (dn, do_)
            }
            None => (
                dist_point_row(new_pos, partners, k),
                dist_point_row(r_old, partners, k),
            ),
        };
        s += j2_from_dist(j2, d_new) - j2_from_dist(j2, d_old);
    }
    s
}

// ============================================================================
// J3 (three-body) rank-1 ratio and streaming state.
//
// This state is decoupled from AO data: the orbital matrices (aos_up/aos_dn) and the moved
// electron's new orbital column are supplied by the caller (the orbital eval
// is handled by the orbital module). This keeps the streaming algebra independent
// of the concrete AO representation.
// ============================================================================

/// Slim ratio-only J3 streaming state. `n_orb` = orbital dim.
///
/// The J3 coupling block is symmetric, so `j3^T*aos == j3*aos` and the state
/// caches ONE product per spin (no separate `j3^T*aos` table or second rowsum) --
/// `w_up` / `w_dn` serve as both the suffix (m>k) and prefix (m<k) operand, and each
/// cross vector is that table's rowsum.
pub struct J3RatioState {
    pub f32_arithmetic: bool,
    pub aos_up: Array2<f64>,   // (n_orb, N_up)
    pub aos_dn: Array2<f64>,   // (n_orb, N_dn)
    pub w_up: Array2<f64>,     // j3 * aos_up  (== j3^T * aos_up under symmetry)
    pub w_dn: Array2<f64>,     // j3 * aos_dn  (== j3^T * aos_dn)
    pub cross_up: Array1<f64>, // rowsum(w_up)  (up_cross_vec, consumed by dn moves)
    pub cross_dn: Array1<f64>, // rowsum(w_dn)  (dn_cross_vec, consumed by up moves)
}

/// The J3 coupling block `j_matrix[:, 0..n_orb]` must be symmetric
/// (bosonic Jastrow; jqmc's `Jastrow_three_body_data.sanity_check` asserts it too).
/// The streaming/grad kernels exploit `j3^T*x == j3*x` to halve the dense GEMMs +
/// matvecs, so a non-symmetric block would silently corrupt results -- strict-raise
/// at native-ctx build (once per run). `atol`/`rtol` are sourced from
/// `jqmc._setting.{atol,rtol}_consistency` (the designated fp64 STORAGE-symmetry
/// floor), never hardcoded here.
pub fn assert_j3_block_symmetric(
    j_matrix: &ArrayView2<f64>,
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    let n_orb = j_matrix.nrows();
    let mut worst = 0.0_f64;
    for a in 0..n_orb {
        for b in (a + 1)..n_orb {
            let jab = j_matrix[[a, b]];
            let jba = j_matrix[[b, a]];
            let d = (jab - jba).abs();
            // allclose semantics: |J-J^T| <= atol + rtol*|J^T|
            if d > atol + rtol * jba.abs() && d > worst {
                worst = d;
            }
        }
    }
    if worst > 0.0 {
        return Err(format!(
            "J3 coupling block j_matrix[:, 0..{n_orb}] must be symmetric (bosonic Jastrow); \
             max|J - J^T| = {worst:.3e} exceeds atol={atol:.1e}+rtol={rtol:.1e}. The native \
             streaming/ratio/grad kernels rely on j3^T*x == j3*x."
        ));
    }
    Ok(())
}

/// `out[a,m] = Sum_b j3[a,b] aos[b,m]`  (j3 * aos).
#[allow(dead_code)] // scalar reference for j3_mul_faer (kept for cross-checks)
fn j3_mul(j3: &ArrayView2<f64>, aos: &ArrayView2<f64>) -> Array2<f64> {
    let n_orb = j3.nrows();
    let n = aos.ncols();
    let mut out = Array2::<f64>::zeros((n_orb, n));
    for a in 0..n_orb {
        for b in 0..n_orb {
            let jab = j3[[a, b]];
            if jab == 0.0 {
                continue;
            }
            for m in 0..n {
                out[[a, m]] += jab * aos[[b, m]];
            }
        }
    }
    out
}

/// faer GEMM `out (n_orb, n) = j3 (n_orb,n_orb) * aos (n_orb, n)`. `j3c` is the
/// j3 block as a contiguous row-major slice (length n_orb^2). ndarray row-major
/// `(R,C)` reads as faer column-major `(C,R)` = its transpose, so reading the
/// row-major buffers column-major gives j3T / aosT and we compute the transposed
/// product `outT = aosT*j3T` straight into a row-major `out`. Replaces the dense
/// O(n_orb^2*n) triple loop with a BLAS-class micro-kernel. Mirrors `project_faer`.
fn j3_mul_faer(j3c: &[f64], aos: &ArrayView2<f64>, n_orb: usize) -> Array2<f64> {
    let n = aos.ncols();
    let aosc = aos.to_owned();
    let mut out = Array2::<f64>::zeros((n_orb, n));
    let j3_faer = faer::MatRef::from_column_major_slice(j3c, n_orb, n_orb); // = j3T
    let aos_faer = faer::MatRef::from_column_major_slice(aosc.as_slice().unwrap(), n, n_orb); // = aosT
    let mut res = faer::MatMut::from_column_major_slice_mut(out.as_slice_mut().unwrap(), n, n_orb); // = outT
    faer::linalg::matmul::matmul(
        res.as_mut(),
        faer::Accum::Replace,
        aos_faer,
        j3_faer,
        1.0,
        faer::Par::Seq,
    );
    out
}

fn row_sum(m: &Array2<f64>) -> Array1<f64> {
    let mut s = Array1::<f64>::zeros(m.nrows());
    for i in 0..m.nrows() {
        let mut acc = 0.0;
        for j in 0..m.ncols() {
            acc += m[[i, j]];
        }
        s[i] = acc;
    }
    s
}

fn j3_block<'a>(j_matrix: &'a ArrayView2<f64>) -> (ArrayView2<'a, f64>, ArrayView1<'a, f64>) {
    let n_orb = j_matrix.nrows();
    let j3 = j_matrix.slice(numpy::ndarray::s![.., 0..n_orb]);
    let j1 = j_matrix.slice(numpy::ndarray::s![.., n_orb]);
    (j3, j1)
}

/// `_init_jastrow_ratio_state`: build the cached tables from orbital matrices.
/// Symmetry permits two faer GEMMs (`j3*aos_up`, `j3*aos_dn`) instead of four;
/// the `j3^T*aos` tables are the same product by symmetry, and each cross vector
/// is the corresponding table's rowsum.
pub fn init_j3_ratio_state(
    j_matrix: &ArrayView2<f64>,
    aos_up: &ArrayView2<f64>,
    aos_dn: &ArrayView2<f64>,
) -> J3RatioState {
    init_j3_ratio_state_with_policy(
        j_matrix,
        aos_up,
        aos_dn,
        crate::standalone_precision_policy().is_f32("jastrow_ratio"),
    )
}

pub fn init_j3_ratio_state_with_policy(
    j_matrix: &ArrayView2<f64>,
    aos_up: &ArrayView2<f64>,
    aos_dn: &ArrayView2<f64>,
    f32_arithmetic: bool,
) -> J3RatioState {
    let (j3, _j1) = j3_block(j_matrix);
    // Materialize the j3 block contiguous ONCE (the slice has row stride n_orb+1);
    // both GEMMs reuse it via faer.
    let n_orb = j_matrix.nrows();
    let j3c = j3.to_owned();
    let j3cs = j3c.as_slice().expect("j3c contiguous");
    let (w_up, w_dn, cross_up, cross_dn) = if f32_arithmetic {
        let mul = |aos: &ArrayView2<f64>| {
            let mut out = Array2::<f32>::zeros((n_orb, aos.ncols()));
            for a in 0..n_orb {
                for b in 0..n_orb {
                    let jab = j3[[a, b]] as f32;
                    for e in 0..aos.ncols() {
                        out[[a, e]] += jab * aos[[b, e]] as f32;
                    }
                }
            }
            out.mapv(f64::from)
        };
        let wu = mul(aos_up);
        let wd = mul(aos_dn);
        let sum = |m: &Array2<f64>| {
            Array1::from_iter((0..m.nrows()).map(|i| {
                let mut s = 0.0_f32;
                for e in 0..m.ncols() {
                    s += m[[i, e]] as f32;
                }
                s as f64
            }))
        };
        let cu = sum(&wu);
        let cd = sum(&wd);
        (wu, wd, cu, cd)
    } else {
        let wu = j3_mul_faer(j3cs, aos_up, n_orb);
        let wd = j3_mul_faer(j3cs, aos_dn, n_orb);
        let cu = row_sum(&wu);
        let cd = row_sum(&wd);
        (wu, wd, cu, cd)
    };
    J3RatioState {
        f32_arithmetic,
        aos_up: if f32_arithmetic {
            aos_up.mapv(|x| (x as f32) as f64)
        } else {
            aos_up.to_owned()
        },
        aos_dn: if f32_arithmetic {
            aos_dn.mapv(|x| (x as f32) as f64)
        } else {
            aos_dn.to_owned()
        },
        w_up,
        w_dn,
        cross_up,
        cross_dn,
    }
}

/// J3 rank-1 ratio `exp(J3(r') - J3(r))` for a single move, using cached state.
/// `aos_new_col` = orbital values at the proposed position (n_orb,).
/// Because W == WT, the m>k (suffix) and m<k (prefix) contributions
/// are both drawn from a SINGLE accumulate `v` (one table read instead of two).
pub fn j3_ratio(
    j_matrix: &ArrayView2<f64>,
    state: &J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
) -> f64 {
    let n_orb = j_matrix.nrows();
    let n = if is_up {
        state.w_up.ncols()
    } else {
        state.w_dn.ncols()
    };
    let mut aos_p = vec![0.0_f64; n_orb];
    let mut v = vec![0.0_f64; n];
    j3_ratio_with_scratch(
        j_matrix,
        state,
        is_up,
        moved_index,
        aos_new_col,
        &mut aos_p,
        &mut v,
    )
}

/// Allocation-free sibling of [`j3_ratio`] for the MCMC and ECP inner loops.
/// `aos_p_scratch` needs `n_orb` entries and `v_scratch` the moved spin count.
pub(crate) fn j3_ratio_with_scratch(
    j_matrix: &ArrayView2<f64>,
    state: &J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
    aos_p_scratch: &mut [f64],
    v_scratch: &mut [f64],
) -> f64 {
    if state.f32_arithmetic {
        let n_orb = j_matrix.nrows();
        let old = if is_up { &state.aos_up } else { &state.aos_dn };
        let delta = &mut aos_p_scratch[..n_orb];
        for a in 0..n_orb {
            delta[a] = (aos_new_col[a] as f32 - old[[a, moved_index]] as f32) as f64;
        }
        let mut total = 0.0_f32;
        for a in 0..n_orb {
            total += j_matrix[[a, n_orb]] as f32 * delta[a] as f32;
        }
        let (w, cross) = if is_up {
            (&state.w_up, &state.cross_dn)
        } else {
            (&state.w_dn, &state.cross_up)
        };
        let v = &mut v_scratch[..w.ncols()];
        for m in 0..w.ncols() {
            if m == moved_index {
                continue;
            }
            let mut vm = 0.0_f32;
            for a in 0..n_orb {
                vm += delta[a] as f32 * w[[a, m]] as f32;
            }
            v[m] = vm as f64;
            total += vm;
        }
        for a in 0..n_orb {
            total += cross[a] as f32 * delta[a] as f32;
        }
        return total.exp() as f64;
    }
    let (_j3, j1) = j3_block(j_matrix);
    let n_orb = j_matrix.nrows();
    let aos_old = if is_up { &state.aos_up } else { &state.aos_dn };
    // aos_p = aos_new - aos_old[:, k]
    let aos_p = &mut aos_p_scratch[..n_orb];
    for a in 0..n_orb {
        aos_p[a] = aos_new_col[a] - aos_old[[a, moved_index]];
    }

    let mut term1 = 0.0;
    for a in 0..n_orb {
        term1 += j1[a] * aos_p[a];
    }

    let (w, cross): (&Array2<f64>, &Array1<f64>) = if is_up {
        (&state.w_up, &state.cross_dn) // dn_cross_vec for an up move
    } else {
        (&state.w_dn, &state.cross_up) // up_cross_vec for a dn move
    };
    let n = w.ncols();
    // V[m] = Sum_a aos_p[a]*W[a,m]. Accumulate over ROWS a (W rows are contiguous,
    // N-wide) to avoid the strided column walk. Under symmetry the prefix operand
    // P[m] = Sum_a aos_p[a]*WT[a,m] equals V[m], so both the m>k and m<k
    // partial sums read this one `v` -- i.e. Sum_{m != k} v[m].
    let v = &mut v_scratch[..n];
    v.fill(0.0);
    j3_ratio_v(w.as_slice().expect("W contiguous"), aos_p, n_orb, n, v);
    let mut term23 = 0.0;
    for (m, v_m) in v[..n].iter().enumerate() {
        if m != moved_index {
            term23 += *v_m;
        }
    }
    let mut term4 = 0.0;
    for a in 0..n_orb {
        term4 += cross[a] * aos_p[a];
    }
    (term1 + term23 + term4).exp()
}

/// Production-fast algebraic reduction of [`j3_ratio`]. Since `cross_up/down`
/// are already the row sums of `W_up/down`, the complete same-spin contribution
/// `Sum_{m != k} aos_p.W[:,m]` is one dot with
/// `cross_same - W[:,k]`. Together with J1 and the other-spin row sum this turns
/// an O(n_orb * n_elec) AXPY into one O(n_orb) dot per proposal. The sum is
/// reassociated, so bit-exact callers retain [`j3_ratio_with_scratch`].
pub(crate) fn j3_ratio_fast_linear(
    j_matrix: &ArrayView2<f64>,
    state: &J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
) -> f64 {
    let delta = j3_delta_fast_linear(j_matrix, state, is_up, moved_index, aos_new_col);
    if state.f32_arithmetic {
        (delta as f32).exp() as f64
    } else {
        delta.exp()
    }
}

/// Logarithm of [`j3_ratio_fast_linear`] for proposal kernels that fuse the
/// J1/J2 and J3 exponentials.
pub(crate) fn j3_delta_fast_linear(
    j_matrix: &ArrayView2<f64>,
    state: &J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
) -> f64 {
    let n_orb = j_matrix.nrows();
    let (aos_old, w_same, cross_same, cross_other) = if is_up {
        (&state.aos_up, &state.w_up, &state.cross_up, &state.cross_dn)
    } else {
        (&state.aos_dn, &state.w_dn, &state.cross_dn, &state.cross_up)
    };
    if state.f32_arithmetic {
        let mut delta = 0.0_f32;
        for a in 0..n_orb {
            let aos_p = aos_new_col[a] as f32 - aos_old[[a, moved_index]] as f32;
            let q = j_matrix[[a, n_orb]] as f32 + cross_other[a] as f32 + cross_same[a] as f32
                - w_same[[a, moved_index]] as f32;
            delta += aos_p * q;
        }
        delta as f64
    } else {
        let mut delta = 0.0_f64;
        for a in 0..n_orb {
            let aos_p = aos_new_col[a] - aos_old[[a, moved_index]];
            let q =
                j_matrix[[a, n_orb]] + cross_other[a] + cross_same[a] - w_same[[a, moved_index]];
            delta += aos_p * q;
        }
        delta
    }
}

/// `_advance_jastrow_ratio_state` after an accepted single-electron move.
/// A single matvec `d_j = j3*delta` (== `j3^T*delta`) updates
/// both the moved column of `w_{up,dn}` AND the moved spin's cross rowsum.
pub fn advance_j3_ratio_state(
    j_matrix: &ArrayView2<f64>,
    state: &mut J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
) {
    let _ = advance_j3_ratio_state_with_delta(j_matrix, state, is_up, moved_index, aos_new_col);
}

/// Advance [`J3RatioState`] and return the common q-vector increment
/// `d = J3 * (AO_new - AO_old)`.  Persistent LRDMC candidate tables use this
/// exact vector to update every unchanged electron group with one bulk
/// candidate-AO matvec, mirroring TurboRVB's `uptabpip` table advance.
pub(crate) fn advance_j3_ratio_state_with_delta(
    j_matrix: &ArrayView2<f64>,
    state: &mut J3RatioState,
    is_up: bool,
    moved_index: usize,
    aos_new_col: &[f64],
) -> Vec<f64> {
    if state.f32_arithmetic {
        let n = j_matrix.nrows();
        let aos = if is_up {
            &mut state.aos_up
        } else {
            &mut state.aos_dn
        };
        let mut delta = vec![0.0_f32; n];
        for a in 0..n {
            delta[a] = aos_new_col[a] as f32 - aos[[a, moved_index]] as f32;
            aos[[a, moved_index]] = aos_new_col[a] as f32 as f64;
        }
        let mut dj = vec![0.0_f32; n];
        for a in 0..n {
            for b in 0..n {
                dj[a] += j_matrix[[a, b]] as f32 * delta[b];
            }
        }
        let (w, cross) = if is_up {
            (&mut state.w_up, &mut state.cross_up)
        } else {
            (&mut state.w_dn, &mut state.cross_dn)
        };
        for a in 0..n {
            w[[a, moved_index]] = ((w[[a, moved_index]] as f32) + dj[a]) as f64;
            cross[a] = ((cross[a] as f32) + dj[a]) as f64;
        }
        return dj.into_iter().map(f64::from).collect();
    }
    let n_orb = j_matrix.nrows();
    let ncol = j_matrix.ncols();
    let jm = j_matrix.as_slice().expect("j_matrix contiguous");
    let aos = if is_up {
        &mut state.aos_up
    } else {
        &mut state.aos_dn
    };
    let delta: Vec<f64> = (0..n_orb)
        .map(|a| aos_new_col[a] - aos[[a, moved_index]])
        .collect();
    for a in 0..n_orb {
        aos[[a, moved_index]] = aos_new_col[a];
    }
    // d_j = j3*delta (row dots), the dense j_matrix read ONCE, AVX-512 (J3 hot spot).
    let mut d_j = vec![0.0_f64; n_orb];
    j3_matvec(jm, &delta, n_orb, ncol, &mut d_j);
    let (w, cross) = if is_up {
        (&mut state.w_up, &mut state.cross_up)
    } else {
        (&mut state.w_dn, &mut state.cross_dn)
    };
    for a in 0..n_orb {
        w[[a, moved_index]] += d_j[a];
        cross[a] += d_j[a];
    }
    d_j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::StructureData;
    use numpy::ndarray::{array, Array2};

    fn h_atom_structure() -> StructureData {
        StructureData {
            positions: Array2::zeros((1, 3)),
            atomic_numbers: vec![1.0],
            pbc_flag: false,
            cell: None,
        }
    }

    #[test]
    fn j2_pade_single_pair() {
        let data = JastrowTwoBodyData {
            precision_policy: crate::PrecisionPolicy::full(),
            jastrow_2b_param: 0.5,
            jastrow_2b_type: "pade".to_string(),
            pbc_flag: false,
            cell: None,
        };
        let up = array![[0.0, 0.0, 0.0]];
        let dn = array![[1.0, 0.0, 0.0]];
        // r=1 -> 1/2 * 1/(1+0.5) = 1/3
        let v = compute_jastrow_two_body(&data, &up.view(), &dn.view());
        assert!(
            (v - 1.0 / 3.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE,
            "{v}"
        );
    }

    #[test]
    fn j1_exp_sign_negative() {
        let data = JastrowOneBodyData {
            precision_policy: crate::PrecisionPolicy::full(),
            jastrow_1b_param: 1.0,
            jastrow_1b_type: "exp".to_string(),
            structure_data: h_atom_structure(),
            core_electrons: vec![0.0],
        };
        let up = array![[0.5, 0.0, 0.0]];
        let dn: Array2<f64> = Array2::zeros((0, 3));
        let v = compute_jastrow_one_body(&data, &up.view(), &dn.view());
        assert!(v < 0.0, "J1 attractive prefactor should be negative: {v}");
    }

    #[test]
    fn j12_ratio_mesh_matches_per_point_bitwise() {
        // Hoisted old-side terms: the mesh routine must equal the per-point
        // jastrow_1b2b_ratio bit for bit, open and periodic J2, both spins.
        let structure = |cell: Option<Array2<f64>>| StructureData {
            positions: array![[0.1, -0.2, 0.3], [1.3, 0.8, -0.6]],
            atomic_numbers: vec![8.0, 1.0],
            pbc_flag: cell.is_some(),
            cell,
        };
        let up = array![[0.5, 0.1, -0.2], [1.1, 0.7, 0.4], [-0.3, 0.9, 0.2]];
        let dn = array![[0.2, -0.6, 0.8], [1.7, 0.2, -0.9]];
        let mut mesh = Array2::<f64>::zeros((12, 3));
        for v in 0..12 {
            let t = 0.6 * v as f64;
            mesh[[v, 0]] = 0.3 + 0.8 * t.cos();
            mesh[[v, 1]] = -0.1 + 0.8 * t.sin();
            mesh[[v, 2]] = 0.2 + 0.1 * v as f64;
        }
        for cell in [
            None,
            Some(array![[5.0, 0.0, 0.0], [0.4, 4.5, 0.0], [0.0, 0.3, 6.0]]),
        ] {
            for (t1, t2) in [("exp", "pade"), ("pade", "exp")] {
                let data = JastrowData {
                    precision_policy: crate::PrecisionPolicy::full(),
                    one_body: Some(JastrowOneBodyData {
                        precision_policy: crate::PrecisionPolicy::full(),
                        jastrow_1b_param: 0.9,
                        jastrow_1b_type: t1.to_string(),
                        structure_data: structure(cell.clone()),
                        core_electrons: vec![2.0, 0.0],
                    }),
                    two_body: Some(JastrowTwoBodyData {
                        precision_policy: crate::PrecisionPolicy::full(),
                        jastrow_2b_param: 0.7,
                        jastrow_2b_type: t2.to_string(),
                        pbc_flag: cell.is_some(),
                        cell: cell.clone(),
                    }),
                    three_body: None,
                    three_body_complex_only: false,
                    dtn: None,
                };
                for (is_up, n) in [(true, up.nrows()), (false, dn.nrows())] {
                    for i in 0..n {
                        let mut got = vec![0.0; 12];
                        jastrow_1b2b_ratio_mesh(
                            &data,
                            &up.view(),
                            &dn.view(),
                            is_up,
                            i,
                            &mesh.view(),
                            &mut got,
                        );
                        for v in 0..12 {
                            let p = [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]];
                            let want =
                                jastrow_1b2b_ratio(&data, &up.view(), &dn.view(), is_up, i, p);
                            assert_eq!(
                                want.to_bits(),
                                got[v].to_bits(),
                                "up={is_up} i={i} v={v} pbc={}",
                                cell.is_some()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fast_periodic_j12_matches_exact() {
        // FAST periodic J1/J2 route (fastmath sincos) vs the exact libm
        // routines: distance, single-move delta, and ECP-mesh ratio, both J1/J2
        // forms, both spins, > FAST_J12_LANES partners (full + padded blocks),
        // electrons outside the home cell and a near-coalescence pair.
        let atol_d = crate::central_setting_f64("ATOL_FASTMATH_PERIODIC_SMOOTH_DIST");
        let rtol_r = crate::central_setting_f64("RTOL_FASTMATH_PERIODIC_J12_RATIO");
        let cell = array![[5.0, 0.0, 0.0], [0.4, 4.5, 0.0], [0.0, 0.3, 6.0]];
        let ci = CellInfo::new(&cell.view());
        // Deterministic quasi-random coordinates in [-1.2, 1.2) x the cell span.
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.4 - 1.2
        };
        let mut rand_pos = |scale: f64| {
            [
                scale * 5.0 * next(),
                scale * 4.5 * next(),
                scale * 6.0 * next(),
            ]
        };

        // (1) distance, including the coalescence (all |theta| small) branch.
        let (mut max_dd, mut n_small) = (0.0_f64, 0usize);
        for t in 0..4000 {
            let x = if t % 4 == 0 {
                rand_pos(1e-3)
            } else {
                rand_pos(1.0)
            };
            let exact = smooth_dist(x, &ci).0;
            let fast = smooth_dist_value_fast(x, &ci);
            if t % 4 == 0 {
                n_small += 1;
            }
            max_dd = max_dd.max((fast - exact).abs());
        }
        eprintln!("fast_periodic_j12: max |d_fast - d_exact| = {max_dd:e}");
        assert!(
            n_small > 0 && max_dd <= atol_d,
            "smooth distance: {max_dd:e} > {atol_d:e}"
        );

        // (2) ratios.
        let n_up = 7;
        let n_dn = 6;
        let mut up = Array2::<f64>::zeros((n_up, 3));
        let mut dn = Array2::<f64>::zeros((n_dn, 3));
        for i in 0..n_up {
            let p = rand_pos(1.0);
            for c in 0..3 {
                up[[i, c]] = p[c];
            }
        }
        for i in 0..n_dn {
            let p = rand_pos(1.0);
            for c in 0..3 {
                dn[[i, c]] = p[c];
            }
        }
        // Near-coalescence partner of up[0].
        for c in 0..3 {
            dn[[0, c]] = up[[0, c]] + 1e-4 * (c as f64 + 1.0);
        }
        let nv = 12;
        let structure = |cell: Option<Array2<f64>>| StructureData {
            positions: array![[0.1, -0.2, 0.3], [1.3, 0.8, -0.6], [-2.0, 1.9, 2.4]],
            atomic_numbers: vec![8.0, 1.0, 1.0],
            pbc_flag: cell.is_some(),
            cell,
        };
        let make = |cell: Option<Array2<f64>>,
                    t1: &str,
                    t2: &str,
                    [a1, a2]: [f64; 2],
                    policy: crate::PrecisionPolicy| JastrowData {
            precision_policy: policy,
            one_body: Some(JastrowOneBodyData {
                precision_policy: policy,
                jastrow_1b_param: a1,
                jastrow_1b_type: t1.to_string(),
                structure_data: structure(cell.clone()),
                core_electrons: vec![2.0, 0.0, 0.0],
            }),
            two_body: Some(JastrowTwoBodyData {
                precision_policy: policy,
                jastrow_2b_param: a2,
                jastrow_2b_type: t2.to_string(),
                pbc_flag: cell.is_some(),
                cell: cell.clone(),
            }),
            three_body: None,
            three_body_complex_only: false,
            dtn: None,
        };
        // Route scope: molecular J1/J2 never takes the FAST periodic twin.
        assert!(J12FastPeriodic::new(&make(
            None,
            "exp",
            "pade",
            [0.9, 0.7],
            crate::PrecisionPolicy::full()
        ))
        .is_none());
        let (mut max_rel, mut max_dev, mut n_diff_bits, mut n_cmp) =
            (0.0_f64, 0.0_f64, 0usize, 0usize);
        // Default mixed keeps jastrow_ratio in fp64 while the other
        // supported AO/Jastrow zones use fp32.
        let mixed = crate::PrecisionPolicy {
            dtypes: crate::PRECISION_ZONES.map(|zone| match zone {
                "ao_eval" | "jastrow_eval" | "jastrow_grad_lap" => crate::RealDType::F32,
                _ => crate::RealDType::F64,
            }),
        };
        for (mode, policy) in [("full", crate::PrecisionPolicy::full()), ("mixed", mixed)] {
            for (t1, t2, parameters) in [
                ("exp", "pade", [0.9, 0.7]),
                ("pade", "exp", [0.9, 0.7]),
                ("exp", "exp", [0.9, 0.7]),
                ("exp", "pade", [1.0e-14, 0.7]),
                ("pade", "exp", [0.9, 1.0e-14]),
            ] {
                let data = make(Some(cell.clone()), t1, t2, parameters, policy);
                let mut case_max = 0.0_f64;
                let prep =
                    J12FastPeriodic::new(&data).expect("periodic fp64 J1/J2 engages the FAST twin");
                for (is_up, n) in [(true, n_up), (false, n_dn)] {
                    for i in 0..n {
                        let centre = if is_up {
                            up.row(i).to_owned()
                        } else {
                            dn.row(i).to_owned()
                        };
                        let mut mesh = Array2::<f64>::zeros((nv, 3));
                        for v in 0..nv {
                            let t = 0.6 * v as f64;
                            mesh[[v, 0]] = centre[0] + 0.8 * t.cos();
                            mesh[[v, 1]] = centre[1] + 0.8 * t.sin();
                            mesh[[v, 2]] = centre[2] + 0.1 * v as f64 - 0.5;
                        }
                        let mut exact = vec![0.0; nv];
                        let mut fast = vec![0.0; nv];
                        jastrow_1b2b_ratio_mesh(
                            &data,
                            &up.view(),
                            &dn.view(),
                            is_up,
                            i,
                            &mesh.view(),
                            &mut exact,
                        );
                        prep.ratio_mesh(&up.view(), &dn.view(), is_up, i, &mesh.view(), &mut fast);
                        for v in 0..nv {
                            let p = [mesh[[v, 0]], mesh[[v, 1]], mesh[[v, 2]]];
                            let single = prep.delta(&up.view(), &dn.view(), is_up, i, p).exp();
                            for got in [fast[v], single] {
                                let rel = (got / exact[v] - 1.0).abs();
                                case_max = case_max.max(rel);
                                n_diff_bits += usize::from(got.to_bits() != exact[v].to_bits());
                                n_cmp += 1;
                            }
                            max_dev = max_dev.max((exact[v] - 1.0).abs());
                        }
                    }
                }
                eprintln!("fast_periodic_j12: {mode} {t1}/{t2} {parameters:?} relative error {case_max:e}");
                max_rel = max_rel.max(case_max);
            }
        }
        eprintln!("fast_periodic_j12: max |R_fast/R_exact - 1| = {max_rel:e} over {n_cmp} ratios ({n_diff_bits} not bitwise equal)");
        assert!(max_rel <= rtol_r, "ratio: {max_rel:e} > {rtol_r:e}");
        // Non-vacuity: the exact ratios are far from 1 at this tolerance (a
        // FAST route returning 1 would fail), and the FAST route is a distinct
        // arithmetic route (not silently the exact one).
        assert!(
            max_dev > rtol_r.sqrt(),
            "fixture ratios too close to 1: {max_dev:e}"
        );
        assert!(
            n_diff_bits > 0,
            "FAST route reproduced the exact route bit for bit"
        );
    }

    #[test]
    fn j3_j1_column_only() {
        // orb_num=1, j3 block zero, j1 column = 2.0; one up electron with ao=3.0
        let jm = array![[0.0, 2.0]];
        let aos_up = array![[3.0]]; // (orb=1, n_up=1)
        let aos_dn: Array2<f64> = Array2::zeros((1, 0));
        let v = compute_jastrow_three_body_value(&jm.view(), &aos_up.view(), &aos_dn.view());
        assert!(
            (v - 6.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE,
            "{v}"
        );
    }

    /// IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001: the orbital-major mirror advance
    /// of the final J3 derivative contractions reproduces an
    /// electron-major per-row scalar loop (`legacy` below) BIT FOR BIT, and the mirror kept in
    /// step by `J3GradShadow::set_electron` equals a fresh rebuild.
    #[test]
    fn j3_grad_outputs_unchanged_mirror_is_bitwise_legacy() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let (n_orb, n_up, n_dn) = (19_usize, 7_usize, 5_usize);
        let mut table = |ne: usize| Array2::from_shape_fn((ne, n_orb), |_| next());
        let tabs: Vec<Array2<f64>> = (0..8)
            .map(|k| table(if k < 4 { n_up } else { n_dn }))
            .collect();
        let make = || {
            let mut c = j3_grad_lap_cache_from_tables(
                tabs[0].clone(),
                tabs[1].clone(),
                tabs[2].clone(),
                tabs[3].clone(),
                tabs[4].clone(),
                tabs[5].clone(),
                tabs[6].clone(),
                tabs[7].clone(),
            );
            c.output = Some(J3GradLapOutput {
                grad_up: Array2::from_shape_fn((n_up, 3), |(i, k)| 0.1 * i as f64 - 0.3 * k as f64),
                grad_dn: Array2::from_shape_fn((n_dn, 3), |(i, k)| 0.2 * k as f64 - 0.1 * i as f64),
                lap_up: Array1::from_shape_fn(n_up, |i| 0.5 - 0.1 * i as f64),
                lap_dn: Array1::from_shape_fn(n_dn, |i| 0.05 * i as f64 - 0.4),
            });
            c
        };
        let legacy = |cache: &mut J3GradLapCache, d: &[f64], is_up: bool, electron: usize| {
            let out = cache.output.as_mut().unwrap();
            let spins = [
                (
                    is_up,
                    [&cache.up_x, &cache.up_y, &cache.up_z, &cache.up_lap],
                    &mut out.grad_up,
                    &mut out.lap_up,
                ),
                (
                    !is_up,
                    [&cache.dn_x, &cache.dn_y, &cache.dn_z, &cache.dn_lap],
                    &mut out.grad_dn,
                    &mut out.lap_dn,
                ),
            ];
            for (moved_spin, t, grad_out, lap_out) in spins {
                for i in 0..t[0].nrows() {
                    if moved_spin && i == electron {
                        continue;
                    }
                    let (mut dx, mut dy, mut dz, mut dl) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                    for a in 0..d.len() {
                        let da = d[a];
                        dx += t[0][[i, a]] * da;
                        dy += t[1][[i, a]] * da;
                        dz += t[2][[i, a]] * da;
                        dl += t[3][[i, a]] * da;
                    }
                    grad_out[[i, 0]] += dx;
                    grad_out[[i, 1]] += dy;
                    grad_out[[i, 2]] += dz;
                    lap_out[i] += dl;
                }
            }
        };
        let bits = |a: &Array2<f64>| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let bits1 = |a: &Array1<f64>| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut fast = make();
        let mut slow = make();
        for (hop, &(is_up, e)) in [
            (true, 3_usize),
            (false, 0),
            (false, 4),
            (true, 0),
            (true, 6),
        ]
        .iter()
        .enumerate()
        {
            let d: Vec<f64> = (0..n_orb)
                .map(|_| next() * 10f64.powi(hop as i32 - 2))
                .collect();
            advance_j3_grad_outputs_unchanged(&mut fast, &d, is_up, e);
            legacy(&mut slow, &d, is_up, e);
            let (fo, so) = (fast.output.as_ref().unwrap(), slow.output.as_ref().unwrap());
            assert_eq!(bits(&fo.grad_up), bits(&so.grad_up), "hop {hop}: grad_up");
            assert_eq!(bits(&fo.grad_dn), bits(&so.grad_dn), "hop {hop}: grad_dn");
            assert_eq!(bits1(&fo.lap_up), bits1(&so.lap_up), "hop {hop}: lap_up");
            assert_eq!(bits1(&fo.lap_dn), bits1(&so.lap_dn), "hop {hop}: lap_dn");
            // Row writer: replace the moved electron's rows in both layouts.
            let cols: Vec<Array2<f64>> = (0..4)
                .map(|_| Array2::from_shape_fn((n_orb, 1), |_| next()))
                .collect();
            for c in [&mut fast, &mut slow] {
                let rows = if is_up {
                    [&mut c.up_x, &mut c.up_y, &mut c.up_z, &mut c.up_lap]
                } else {
                    [&mut c.dn_x, &mut c.dn_y, &mut c.dn_z, &mut c.dn_lap]
                };
                for (dst, src) in rows.into_iter().zip(&cols) {
                    for a in 0..n_orb {
                        dst[[e, a]] = src[[a, 0]];
                    }
                }
            }
            fast.shadow
                .as_mut()
                .expect("mirror built by the first advance")
                .set_electron(is_up, e, [&cols[0], &cols[1], &cols[2], &cols[3]]);
            let fresh = J3GradShadow::build(&fast);
            let kept = fast.shadow.as_ref().unwrap();
            for k in 0..4 {
                assert_eq!(
                    bits(&kept.up[k]),
                    bits(&fresh.up[k]),
                    "hop {hop}: mirror up[{k}]"
                );
                assert_eq!(
                    bits(&kept.dn[k]),
                    bits(&fresh.dn[k]),
                    "hop {hop}: mirror dn[{k}]"
                );
            }
        }
    }
}
