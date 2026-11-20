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

//! MCMC single-electron move driver.
//!
//! Mirrors the proposal + Metropolis logic of the jqmc `body_fun`
//! (jqmc_mcmc.py), threading the **bit-exact threefry RNG** in the
//! exact split order jqmc uses. The wavefunction-ratio pieces (Det_T_ratio,
//! J_ratio, R_AS_ratio) are evaluated by the native determinant, Jastrow and
//! AS streaming kernels.
//!
//! Gaussian proposal displacement (`jax.random.normal`) is INJECTED by the
//! caller (`injected_normals`); its threefry subkey is still
//! consumed so the key sequence stays aligned with jqmc.

use crate::data::JastrowData;
use crate::determinant::{
    self, advance_det_ratio_state_from_orb, as_proposed, as_recompute,
    compute_u_dn_move_from_orb_into, compute_v_up_move_from_orb_into, init_det_ratio_state,
    sm_apply_inplace, sm_ratio_into, sm_update, DetRatioState, GeminalData, OrbData,
};
use crate::jastrow_factor::{
    self, advance_j3_ratio_state, j3_ratio, j3_ratio_fast_linear, j3_ratio_with_scratch,
    J3RatioState,
};
use crate::threefry;
use numpy::ndarray::{Array2, ArrayView2};

/// Periodic `(cell, inverse cell, image cutoff)` of a sweep; `None` = molecular.
type CellNmax<'a, 'v> = Option<(&'a ArrayView2<'v, f64>, &'a [[f64; 3]; 3], i32)>;

/// RNG state for a move, threaded like `jax_PRNG_key` in the jqmc body.
pub struct MoveRng {
    pub key: [u32; 2],
}

impl MoveRng {
    pub fn new(key: [u32; 2]) -> Self {
        MoveRng { key }
    }
    /// `jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)`:
    /// new key = split[0], returns subkey = split[1].
    pub fn split_off(&mut self) -> [u32; 2] {
        let s = threefry::split(self.key, 2);
        self.key = s[0];
        s[1]
    }
    /// jqmc body uses the default-int (int64 under x64) randint.
    pub fn randint(&mut self, minval: i64, maxval: i64) -> i64 {
        let sk = self.split_off();
        threefry::randint_i64(sk, 1, minval, maxval)[0]
    }
    pub fn uniform(&mut self) -> f64 {
        let sk = self.split_off();
        threefry::uniform_f64(sk, 1, 0.0, 1.0)[0]
    }
}

/// The RNG-driven choices of one move, in jqmc's exact draw order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveDraws {
    pub is_up: bool,
    pub selected_index: usize,
    pub axis: usize,
    pub metropolis_b: f64,
    pub final_key: [u32; 2],
}

/// Draw the RNG choices for one move (electron selection, axis, Metropolis b),
/// consuming the same threefry subkeys as the jqmc body in the same order.
///
/// Order (jqmc_mcmc.py): rand_num(total) -> up_index -> dn_index ->
/// [normal subkey, consumed] -> axis(3) -> metropolis b.
pub fn move_draws(key: [u32; 2], n_up: usize, n_dn: usize) -> MoveDraws {
    let total = (n_up + n_dn) as i64;
    let mut rng = MoveRng::new(key);

    let rand_num = rng.randint(0, total);
    let is_up = rand_num < n_up as i64;
    // jqmc uses randint(0, len(r_*_carts)); for a zero-electron spin the
    // randint(0,0) span collapses to 1 -> 0 (still consumes a subkey).
    let up_index = rng.randint(0, n_up as i64);
    let dn_index = rng.randint(0, n_dn as i64);
    let selected_index = if is_up { up_index } else { dn_index } as usize;

    // Gaussian proposal subkey is consumed (value injected separately).
    let _normal_subkey = rng.split_off();

    let axis = rng.randint(0, 3) as usize;
    let metropolis_b = rng.uniform();

    MoveDraws {
        is_up,
        selected_index,
        axis,
        metropolis_b,
        final_key: rng.key,
    }
}

/// Kato-cusp step-size scaling `f_l = 1/Z^2 * (1 + Z^2*r) / (1 + r)`
/// (jqmc_mcmc.py). `r` = min-image distance to the nearest nucleus.
#[inline]
pub fn kato_f_l(z: f64, norm_r_r: f64) -> f64 {
    1.0 / (z * z) * (1.0 + z * z * norm_r_r) / (1.0 + norm_r_r)
}

/// Forward/backward proposal ratio `T_ratio` (jqmc_mcmc.py).
#[inline]
pub fn t_ratio(f_l: f64, f_prime_l: f64, dist2: f64, dt: f64) -> f64 {
    (f_l / f_prime_l)
        * (-dist2
            * (1.0 / (2.0 * f_prime_l * f_prime_l * dt * dt) - 1.0 / (2.0 * f_l * f_l * dt * dt)))
            .exp()
}

/// Metropolis acceptance clamp mirroring XLA's `jnp.min(jnp.array([1.0, x]))`:
/// NaN PROPAGATES, so the caller's `b < acc` comparison is false and the move is
/// REJECTED, keeping the walker at its last valid configuration. Rust's bare
/// `f64::min` is IEEE minNum, which ABSORBS the NaN (`f64::NAN.min(1.0) == 1.0`)
/// and turned a degenerate ratio (determinant underflow / SM division by ~0) into
/// an unconditional accept. On `mol_ecp_sphe_j12`, XLA
/// accepted 0/10 such moves while Rust accepted 10/10, marching the walker to
/// ~300 Bohr. Every acceptance site must clamp through this helper.
#[inline]
pub fn accept_clamp(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else {
        x.min(1.0)
    }
}

/// Metropolis acceptance test: accept iff `b < min(1, R_ratio * T_ratio)`,
/// where `R_ratio = |R_AS_ratio * WF_ratio|^2` (jqmc_mcmc.py).
/// NaN ratio => reject (see [`accept_clamp`]).
#[inline]
pub fn metropolis_accept(r_ratio: f64, t_ratio_val: f64, b: f64) -> bool {
    let acc = accept_clamp(r_ratio * t_ratio_val);
    b < acc
}

/// Nearest atom index by Euclidean distance (molecular path). PBC min-image is
/// a follow-up (with PBC Jastrow). Returns `(index, distance)`.
pub fn nearest_atom_molecular(positions: &[[f64; 3]], r: [f64; 3]) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for (i, p) in positions.iter().enumerate() {
        let dx = r[0] - p[0];
        let dy = r[1] - p[1];
        let dz = r[2] - p[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    (best, best_d2.sqrt())
}

/// Nearest atom by Euclidean distance (molecular). Returns `(index, |r - R|)`.
fn nearest_atom(positions: &ArrayView2<f64>, r: [f64; 3]) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for i in 0..positions.nrows() {
        let dx = r[0] - positions[[i, 0]];
        let dy = r[1] - positions[[i, 1]];
        let dz = r[2] - positions[[i, 2]];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    (best, best_d2.sqrt())
}

/// Nearest atom by Wigner-Seitz **min-image** distance (PBC). Mirrors jqmc's
/// `_find_nearest_index_jnp` + `_get_min_dist_rel_R_cart_jnp` (both reduce each
/// per-atom displacement via `_minimum_image_displacement`). Returns
/// `(index, |min-image(r - R)|)`. Translation-invariant => correct on the
/// unwrapped walker positions jqmc carries.
pub fn nearest_atom_periodic(
    positions: &ArrayView2<f64>,
    r: [f64; 3],
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
    n_max: i32,
) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for i in 0..positions.nrows() {
        let mic = crate::ewald::min_image(
            [
                r[0] - positions[[i, 0]],
                r[1] - positions[[i, 1]],
                r[2] - positions[[i, 2]],
            ],
            cell,
            inv,
            n_max,
        );
        let d2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    (best, best_d2.sqrt())
}

/// Wrap a Cartesian point into the home parallelepiped `[0,1)^3` fractional
/// (jqmc `_wrap_into_cell`): `frac = r*inv; frac -= floor(frac); r = frac*cell`.
/// Used ONLY for the AO-eval coordinate (the periodic image-fold is centered on
/// the home cell); walker storage stays unwrapped, matching jqmc.
#[inline]
pub(crate) fn wrap_into_cell(r: [f64; 3], cell: &ArrayView2<f64>, inv: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut frac = [0.0_f64; 3];
    for k in 0..3 {
        let s = r[0] * inv[0][k] + r[1] * inv[1][k] + r[2] * inv[2][k];
        frac[k] = s - s.floor();
    }
    let mut out = [0.0_f64; 3];
    for k in 0..3 {
        out[k] = frac[0] * cell[[0, k]] + frac[1] * cell[[1, k]] + frac[2] * cell[[2, k]];
    }
    out
}

/// Row-wise [`wrap_into_cell`] over an `(N,3)` array (for the streaming-state
/// initial AO evals). Returns an owned wrapped copy.
pub(crate) fn wrap_rows(
    arr: &ArrayView2<f64>,
    cell: &ArrayView2<f64>,
    inv: &[[f64; 3]; 3],
) -> Array2<f64> {
    let mut out = arr.to_owned();
    for i in 0..arr.nrows() {
        let w = wrap_into_cell([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]], cell, inv);
        out[[i, 0]] = w[0];
        out[[i, 1]] = w[1];
        out[[i, 2]] = w[2];
    }
    out
}

/// Select Euclidean (molecular) or min-image (PBC) nearest atom.
#[inline]
fn nearest_atom_dispatch(
    positions: &ArrayView2<f64>,
    r: [f64; 3],
    cell_nmax: CellNmax<'_, '_>,
) -> (usize, f64) {
    match cell_nmax {
        Some((cell, inv, n_max)) => nearest_atom_periodic(positions, r, cell, inv, n_max),
        None => nearest_atom(positions, r),
    }
}

/// Result of one single-electron move (the `body_fun` outcome).
pub struct MoveResult {
    pub accepted: bool,
    pub is_up: bool,
    pub selected_index: usize,
    pub new_pos: [f64; 3],
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    pub geminal: Array2<f64>,
    pub geminal_inv: Array2<f64>,
    pub det_t_ratio: f64,
    pub r_ratio: f64,
    pub t_ratio: f64,
    pub acceptance_ratio: f64,
    pub metropolis_b: f64,
    pub final_key: [u32; 2],
}

/// One native single-electron MCMC move (molecular path), 1:1 with the jqmc
/// `body_fun` (jqmc_mcmc.py). Det streaming state is re-`init`ed
/// from positions each call (the loop will use `advance`); the Gaussian
/// displacement is INJECTED (`injected_normal`), its threefry subkey consumed.
#[allow(clippy::too_many_arguments)]
pub fn single_move(
    geminal_data: &GeminalData,
    jastrow_data: &JastrowData,
    // The J3 orbital and j_matrix are parsed by the caller from
    // ``jastrow_three_body_data`` exactly as ``mcmc_sweep`` receives them. ``None``
    // = no J3 leaf. Without these the Metropolis ratio would silently drop the J3
    // channel while the sweep computes it (a wrong accept/reject decision).
    j3_orb: Option<&OrbData>,
    j3_jmatrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    geminal: &ArrayView2<f64>,
    geminal_inv: &ArrayView2<f64>,
    dt: f64,
    epsilon_as: f64,
    key: [u32; 2],
    injected_normal: f64,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> MoveResult {
    let inv_cell = cell_nmax.map(|(cell, _)| crate::ewald::inv3(cell));
    let nearest_cn: CellNmax<'_, '_> = match (cell_nmax, &inv_cell) {
        (Some((cell, n_max)), Some(inv)) => Some((cell, inv, n_max)),
        _ => None,
    };
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let d = move_draws(key, n_up, n_dn);
    let is_up = d.is_up;
    let k = d.selected_index;

    // old position of the moved electron (branch to keep r_up/r_dn lifetimes distinct)
    let old_r = if is_up {
        [r_up[[k, 0]], r_up[[k, 1]], r_up[[k, 2]]]
    } else {
        [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
    };

    // proposal step size (Kato cusp) at old position
    let (nearest, norm_r_r) = nearest_atom_dispatch(positions, old_r, nearest_cn);
    let z = charges[nearest];
    let f_l = kato_f_l(z, norm_r_r);
    let sigma = f_l * dt;
    let g = injected_normal * sigma;
    let mut new_r = old_r;
    new_r[d.axis] += g;

    // backward step size at new position
    let (nearest2, norm_r_r_new) = nearest_atom_dispatch(positions, new_r, nearest_cn);
    let z2 = charges[nearest2];
    let f_prime_l = kato_f_l(z2, norm_r_r_new);
    let dist2 = {
        let dx = new_r[0] - old_r[0];
        let dy = new_r[1] - old_r[1];
        let dz = new_r[2] - old_r[2];
        dx * dx + dy * dy + dz * dz
    };
    let t_ratio_val = t_ratio(f_l, f_prime_l, dist2, dt);

    // AO-eval coordinates: home-cell-wrapped under PBC exactly as `mcmc_sweep`
    // does (the periodic AO image fold is home-cell centered); positions are
    // stored, and min-image / J1/J2 / DTN evaluated, on the UNWRAPPED
    // coordinates. Identity for molecular.
    let (r_up_ao, r_dn_ao, ao_pos) = match (cell_nmax, &inv_cell) {
        (Some((cell, _)), Some(inv)) => (
            wrap_rows(r_up, cell, inv),
            wrap_rows(r_dn, cell, inv),
            wrap_into_cell(new_r, cell, inv),
        ),
        _ => (r_up.to_owned(), r_dn.to_owned(), new_r),
    };
    let (r_up_ao, r_dn_ao) = (r_up_ao.view(), r_dn_ao.view());

    // det ratio: v (up-move row delta) / u (dn-move col delta) + one-hot
    let state = init_det_ratio_state(geminal_data, &r_up_ao, &r_dn_ao);
    let n = geminal.nrows();
    let (u, v): (Vec<f64>, Vec<f64>) = if is_up {
        let v = determinant::compute_v_up_move(geminal_data, &state, k, ao_pos);
        let mut u = vec![0.0; n];
        u[k] = 1.0;
        (u, v)
    } else {
        let u = determinant::compute_u_dn_move(geminal_data, &state, k, ao_pos);
        let mut v = vec![0.0; n];
        v[k] = 1.0;
        (u, v)
    };
    let (det_t_ratio, geminal_inv_new, geminal_new) =
        sm_update(geminal_inv, geminal, &u, &v, is_up, k);

    // AS regularization ratio with epsilon clamp. Issues #131/#134: skipped at
    // epsilon_as == 0 (as the sweep); the proposed state's moved line is rebuilt
    // from the new orbital values and its moved inverse column / row is the
    // exact X e_k / R (e_kT X / R); the carried geminal_new / geminal_inv_new are
    // unchanged.
    let r_as_ratio = if epsilon_as > 0.0 {
        let mut g_as = geminal_new.clone();
        let mut gi_as = geminal_inv_new.clone();
        let mut line = vec![0.0_f64; n];
        if is_up {
            let ao_new = geminal_data.orb_up.compute_single(new_r);
            determinant::compute_up_row_from_orb_into(geminal_data, &state, &ao_new, &mut line);
            let col_k = geminal_inv.column(k).to_owned();
            for t in 0..n {
                g_as[[k, t]] = line[t];
                gi_as[[t, k]] = col_k[t] / det_t_ratio;
            }
        } else {
            let ao_new = geminal_data.orb_dn.compute_single(new_r);
            determinant::compute_dn_col_from_orb_into(&state, &ao_new, &mut line);
            let row_k = geminal_inv.row(k).to_owned();
            for t in 0..n {
                g_as[[t, k]] = line[t];
                gi_as[[k, t]] = row_k[t] / det_t_ratio;
            }
        }
        let r_as_p = determinant::compute_as_regularization_factor(&g_as.view(), &gi_as.view());
        let r_as_o = determinant::compute_as_regularization_factor(geminal, geminal_inv);
        determinant::as_guiding_ratio(r_as_p, r_as_o, epsilon_as)
    } else {
        1.0
    };

    // Jastrow ratio: J1/J2 (molecular) x J3 x DTN. ``single_move``
    // must include EVERY Jastrow channel the sweep does, or its Metropolis ratio (hence
    // accept/reject) diverges from the XLA-verified ``mcmc_sweep`` on any J3/DTN leaf.
    let j12 = jastrow_factor::jastrow_1b2b_ratio(jastrow_data, r_up, r_dn, is_up, k, new_r);
    // J3 rank-1 ratio: build the streaming state at the input config (seed AOs via the
    // batch ``orb.compute`` path, exactly as ``mcmc_sweep``), then the moved electron's
    // AO column at new_r via the same ``eval_ao_into`` the sweep uses per move.
    let j3r = match (j3_orb, j3_jmatrix) {
        (Some(orb), Some(jm)) => {
            let aos_up = orb.compute(&r_up_ao);
            let aos_dn = orb.compute(&r_dn_ao);
            let st = jastrow_factor::init_j3_ratio_state_with_policy(
                jm,
                &aos_up.view(),
                &aos_dn.view(),
                jastrow_data.precision_policy.is_f32("jastrow_ratio"),
            );
            // J3 matrix is expressed at orbital rank.  For a composite raw-AO
            // orbital this is the full concatenated child rank, not the embedded
            // GTO child's ``aos.num_ao``.
            let na = jm.nrows();
            let mut j3_ao = vec![0.0_f64; na];
            eval_ao_into(orb, ao_pos, &mut j3_ao, false);
            jastrow_factor::j3_ratio(jm, &st, is_up, k, &j3_ao[..na])
        }
        _ => 1.0,
    };
    // DTN leg ratio: build the streaming state at the input config, then the single-move
    // ratio (same ``compute_dtn_ratio_streaming`` the sweep evaluates per move).
    let jdtn = match &jastrow_data.dtn {
        Some(dtn) => {
            let st = crate::dtn::init_dtn_ratio_state(dtn, r_up, r_dn);
            crate::dtn::compute_dtn_ratio_streaming(dtn, &st, r_up, r_dn, is_up, k, new_r)
        }
        None => 1.0,
    };
    let j_ratio = crate::jastrow_factor::multiply_ratio_components(
        jastrow_data,
        crate::jastrow_factor::multiply_ratio_components(jastrow_data, j12, j3r),
        jdtn,
    );

    // Metropolis acceptance
    let wf_ratio = j_ratio * det_t_ratio;
    let combined = r_as_ratio * wf_ratio;
    let r_ratio = combined * combined; // |.|^2 (real path)
    let acceptance_ratio = accept_clamp(r_ratio * t_ratio_val);
    let accepted = d.metropolis_b < acceptance_ratio;

    // build outcome
    let (mut out_up, mut out_dn) = (r_up.to_owned(), r_dn.to_owned());
    let (out_gem, out_inv) = if accepted {
        let tgt = if is_up { &mut out_up } else { &mut out_dn };
        tgt[[k, 0]] = new_r[0];
        tgt[[k, 1]] = new_r[1];
        tgt[[k, 2]] = new_r[2];
        (geminal_new, geminal_inv_new)
    } else {
        (geminal.to_owned(), geminal_inv.to_owned())
    };

    MoveResult {
        accepted,
        is_up,
        selected_index: k,
        new_pos: new_r,
        r_up: out_up,
        r_dn: out_dn,
        geminal: out_gem,
        geminal_inv: out_inv,
        det_t_ratio,
        r_ratio,
        t_ratio: t_ratio_val,
        acceptance_ratio,
        metropolis_b: d.metropolis_b,
        final_key: d.final_key,
    }
}

/// Result of a full `num_moves` sweep (one measurement block).
pub struct SweepResult {
    pub accepted: usize,
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    pub geminal: Array2<f64>,
    pub geminal_inv: Array2<f64>,
    /// Streaming det-ratio state at the final walker config (1:1 with jQMC's
    /// `Det_ratio_streaming_state`). Carried so that it can feed jQMC's
    /// `compute_local_energy_fast` (`_e_L_fast`), which consumes `geminal_inv` + this.
    pub det_state: crate::determinant::DetRatioState,
    /// Streaming J3-ratio state at the final walker config:
    /// the fast energy chunk hands this to the e_L's ECP-nl to SKIP its per-e_L
    /// `init_j3_ratio_state` rebuild (the sweep already advanced these tables). The
    /// sweep seeds/streams J3 via the SAME `orb.compute` batch path the ECP-nl init
    /// uses, so for MOLECULAR leaves (n_image==1) the final aos columns are bit-equal
    /// to a fresh eval; the w_up/w_dn tables differ only at the streaming-vs-faer
    /// fp floor -> fast-chunk (statistically-guarded) only. `None` when no J3.
    pub j3_state: Option<J3RatioState>,
    pub final_key: [u32; 2],
}

/// No-alloc AO eval into `ao` using the vectorized-exp fast path when this is a
/// sphe molecular basis; falls back otherwise.
#[inline]
fn eval_ao_into(orb: &OrbData, pos: [f64; 3], ao: &mut [f64], fast: bool) {
    // Raw molecular GTO orbital: avoid allocating a Vec for every proposed
    // electron move; this is exactly the same scalar AO kernel used by
    // `compute_single`. (The production `-j3 mo` J3 is MO-wrapped, not raw
    // GTO, and takes the `compute_single` path below.)
    if orb.pw.is_none() && orb.bspline.is_none() && orb.mo_coefficients.is_none() {
        if orb.aos.n_image == 1 {
            if fast && orb.aos.kind == crate::atomic_orbital::AoKind::Cart {
                crate::atomic_orbital::compute_aos_single_cart_fast_exp_into(&orb.aos, pos, ao);
            } else {
                crate::atomic_orbital::compute_aos_single_into(&orb.aos, pos, ao);
            }
        } else if orb.aos.kind == crate::atomic_orbital::AoKind::Sphe
            && !orb.aos.sh_pzmin.is_empty()
        {
            // Periodic pGTO stores image-expanded kernel AOs but the streaming
            // J3 state is home-AO rank.  The molecular single-AO kernel indexes
            // the expanded axis and therefore overruns this buffer in release
            // builds.  Fold image contributions directly into the home axis.
            crate::atomic_orbital::compute_aos_single_fold_into(
                &orb.aos,
                pos,
                &mut ao[..orb.aos.num_ao],
            );
        } else {
            let v = orb.compute_single(pos);
            ao[..v.len()].copy_from_slice(&v);
        }
        return;
    }
    // J3 consumes ORBITAL-rank values, not merely the embedded GTO child's AO
    // block.  ``compute_single`` is the single source for bare GTO, MO-wrapped,
    // and composite gto(+){pw,bspline} leaves, so the MO/composite projection is
    // always applied and the buffer holds the full orbital rank, not only
    // ``aos.num_ao``.
    let v = orb.compute_single(pos);
    ao[..v.len()].copy_from_slice(&v);
}

/// Orbital matrix `(M, N)` at the (home-cell wrapped) electrons `r` through the FAST chunk's
/// screened single-electron evaluator (`eval_orb_into` with `cutoff = true`), i.e. the orbital
/// values the FAST e_L's determinant is built from. Test support for the screened-e_L
/// reference (`_e_L_fast_screened`).
pub(crate) fn fast_orb_matrix(orb: &OrbData, r: &ArrayView2<f64>) -> Array2<f64> {
    let nmo = orb
        .mo_coefficients
        .as_ref()
        .map(|c| c.nrows())
        .unwrap_or(orb.ao_count());
    let mut ao = vec![0.0_f64; orb.ao_count().max(orb.aos.num_ao_kernel)];
    let mut mo = vec![0.0_f64; nmo];
    let mut out = Array2::<f64>::zeros((nmo, r.nrows()));
    for e in 0..r.nrows() {
        let n = eval_orb_into(
            orb,
            [r[[e, 0]], r[[e, 1]], r[[e, 2]]],
            &mut ao,
            &mut mo,
            true,
        );
        for a in 0..n {
            out[[a, e]] = mo[a];
        }
    }
    out
}

/// No-alloc orbital-column eval into reused buffers; returns the MO length.
#[inline]
fn eval_orb_into(
    orb: &OrbData,
    pos: [f64; 3],
    ao: &mut [f64],
    mo: &mut [f64],
    cutoff: bool,
) -> usize {
    let nmo = orb
        .mo_coefficients
        .as_ref()
        .map(|c| c.nrows())
        .unwrap_or(orb.ao_count());
    // Pure B-spline leaf (no gto block): single-position spline value into the AO
    // buffer, then MO project.
    if let Some(bs) = &orb.bspline {
        if orb.aos.num_ao == 0 {
            orb.compute_single_into(pos, &mut ao[..bs.num_ao], &mut mo[..nmo]);
            return nmo;
        }
    }
    // Composite (pgto(+)pw or pgto(+)bspline): the gto-only fast paths below would
    // drop the extra block, so route through compute_single, which concatenates
    // both children in child order before the MO projection.
    if orb.pw.is_some() || orb.bspline.is_some() {
        let v = orb.compute_single(pos);
        mo[..nmo].copy_from_slice(&v);
        return nmo;
    }
    // FAST per-move route of an fp32 `ao_eval` zone (molecular and periodic,
    // Sphe and Cart): the single-position fp32 kernel with the FAST `pz_val`
    // shell screen (dropped AO values < FAST_SCREEN_ABS_BOUND = 1e-12 absolute,
    // issue #100); kept shells are bit-identical to the policy-aware value of the
    // seed/cache rebuild (`compute_single`), so a carried G and its cached
    // columns stay one AO convention. fp64 MO projection (`mo_eval`).
    if cutoff && orb.precision_policy.is_f32("ao_eval") && !orb.aos.shells.is_empty() {
        let na = orb.aos.num_ao;
        crate::atomic_orbital::compute_aos_single_f32_into(&orb.aos, pos, true, &mut ao[..na]);
        match (&orb.mo_coefficients, &orb.mo_coefficients_t) {
            (None, _) => mo[..nmo].copy_from_slice(&ao[..na]),
            (Some(_), Some(ct))
                if orb.aos.n_image == 1 && ct.nrows() == na && ct.ncols() == nmo =>
            {
                // Sparse AXPY over the kept AOs (screened shells are exact zeros).
                let ct = ct.as_slice().expect("C_T contiguous");
                mo[..nmo].fill(0.0);
                for (a, &v) in ao[..na].iter().enumerate() {
                    if v != 0.0 {
                        let row = &ct[a * nmo..(a + 1) * nmo];
                        for (dst, &cm) in mo[..nmo].iter_mut().zip(row) {
                            *dst += v * cm;
                        }
                    }
                }
            }
            (Some(c), _) => crate::determinant::project_8acc(
                c.as_slice().expect("C contiguous"),
                &ao[..na],
                c.ncols(),
                c.nrows(),
                &mut mo[..nmo],
            ),
        }
        return nmo;
    }
    // Production-fast molecular Cartesian path: preserve the dense full-AO +
    // faer projection that wins at water sizes, changing only primitive exp.
    // Both shell fast paths below evaluate in f64. Mixed ao_eval must use the
    // policy-aware path shared by the seed/cache rebuild: otherwise each fast
    // energy-chunk boundary combines a carried f64 G with newly seeded f32
    // orbitals, and the next incremental updates preserve their absolute gap.
    if cutoff
        && !orb.precision_policy.is_f32("ao_eval")
        && orb.aos.n_image == 1
        && orb.aos.kind == crate::atomic_orbital::AoKind::Cart
        && !orb.aos.shells.is_empty()
    {
        crate::atomic_orbital::compute_aos_single_cart_fast_exp_into(
            &orb.aos,
            pos,
            &mut ao[..orb.aos.num_ao],
        );
        match &orb.mo_coefficients {
            None => mo[..nmo].copy_from_slice(&ao[..orb.aos.num_ao]),
            Some(c) => crate::determinant::project_faer(
                c.as_slice().expect("C contiguous"),
                &ao[..orb.aos.num_ao],
                c.ncols(),
                c.nrows(),
                &mut mo[..nmo],
            ),
        }
        return nmo;
    }
    // Molecular sphe MO fast path with per-shell distance cutoff + sparse AXPY MO
    // projection -- skips the ~74% of far AOs (and their C columns) for a large
    // open cluster. Every dropped AO value is < FAST_SCREEN_ABS_BOUND = 1e-12 in absolute
    // AO units (`Shell::pz_val`, issue #100); off by default (bit-exact).
    if cutoff
        && !orb.precision_policy.is_f32("ao_eval")
        && orb.aos.n_image == 1
        && orb.aos.kind == crate::atomic_orbital::AoKind::Sphe
        && !orb.aos.shells.is_empty()
    {
        if let Some(ct) = &orb.mo_coefficients_t {
            crate::atomic_orbital::compute_mo_single_cutoff_into(
                &orb.aos,
                ct.as_slice().expect("C_T contiguous"),
                nmo,
                pos,
                &mut mo[..nmo],
            );
            return nmo;
        }
    }
    // Periodic FAST path (Sphe and Cart): image-folded value with the same
    // absolute `pz_val` shell screen (dropped AO values < FAST_SCREEN_ABS_BOUND =
    // 1e-12, issue #100), then the dense MO projection of the exact periodic path.
    // FAST chunk only (`cutoff`); the exact sweep keeps the unscreened fold.
    if cutoff
        && !orb.precision_policy.is_f32("ao_eval")
        && orb.aos.n_image > 1
        && !orb.aos.shells.is_empty()
    {
        if let Some(c) = &orb.mo_coefficients {
            crate::atomic_orbital::compute_aos_single_fold_cutoff_into(
                &orb.aos,
                pos,
                &mut ao[..orb.aos.num_ao],
            );
            crate::determinant::project_8acc(
                c.as_slice().expect("C contiguous"),
                &ao[..orb.aos.num_ao],
                c.ncols(),
                c.nrows(),
                &mut mo[..nmo],
            );
            return nmo;
        }
    }
    if orb.aos.n_image == 1 {
        orb.compute_single_into(pos, &mut ao[..orb.aos.num_ao], &mut mo[..nmo]);
    } else if orb.aos.kind == crate::atomic_orbital::AoKind::Sphe && !orb.aos.shells.is_empty() {
        // Periodic sphe: no-alloc adaptive-image fold directly into ao, then project.
        crate::atomic_orbital::compute_aos_single_fold_into(
            &orb.aos,
            pos,
            &mut ao[..orb.aos.num_ao],
        );
        match &orb.mo_coefficients {
            None => mo[..nmo].copy_from_slice(&ao[..orb.aos.num_ao]),
            Some(c) => crate::determinant::project_8acc(
                c.as_slice().expect("C contiguous"),
                &ao[..orb.aos.num_ao],
                c.ncols(),
                c.nrows(),
                &mut mo[..nmo],
            ),
        }
    } else {
        let v = orb.compute_single(pos);
        mo[..nmo].copy_from_slice(&v);
    }
    nmo
}

/// Streaming det-ratio caches exactly as [`mcmc_sweep`] seeds them when no carried
/// state is supplied: orbital matrices via the per-move `compute_single` path at the
/// (already home-cell-wrapped, under PBC) positions.
fn seed_det_ratio_state(
    geminal_data: &GeminalData,
    r_up_ao: &ArrayView2<f64>,
    r_dn_ao: &ArrayView2<f64>,
) -> crate::determinant::DetRatioState {
    let ao_up0 = crate::determinant::orb_matrix_via_single(&geminal_data.orb_up, r_up_ao);
    let ao_dn0 = crate::determinant::orb_matrix_via_single(&geminal_data.orb_dn, r_dn_ao);
    crate::determinant::init_det_ratio_state_from_orb_mats(geminal_data, ao_up0, ao_dn0)
}

/// Geminal `G` implied by the caches [`mcmc_sweep`] seeds at `(r_up, r_dn)`.
///
/// The sweep advances the carried `G` by rank-1 DELTAS against its own cached
/// orbital columns, so any difference between the caller's seed `G` and these
/// caches is never corrected: it stays in `G` as a fixed absolute offset. Under a
/// mixed `ao_eval` (fp32) policy the JAX seed and this native evaluation round
/// differently (~3e-7 of max|G|); once an electron moves far out and its row
/// decays below that offset, its Sherman-Morrison ratios stop following
/// `|Psi|^2`. Seeding the driver's `G` from this function makes the
/// carried matrix and the streaming caches one consistent evaluation.
pub fn sweep_seed_geminal(
    geminal_data: &GeminalData,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> Array2<f64> {
    let (r_up_ao, r_dn_ao) = match cell_nmax {
        Some((cell, _)) => {
            let inv = crate::ewald::inv3(cell);
            (wrap_rows(r_up, cell, &inv), wrap_rows(r_dn, cell, &inv))
        }
        None => (r_up.to_owned(), r_dn.to_owned()),
    };
    let state = seed_det_ratio_state(geminal_data, &r_up_ao.view(), &r_dn_ao.view());
    crate::determinant::geminal_from_det_ratio_state(geminal_data, &state)
}

/// Native MCMC sweep: `num_moves` single-electron moves with **streaming-state
/// advance** (det ratio + J3), maintaining `geminal_inv` by rank-1 SM (no
/// re-inversion). 1:1 with the jqmc `fori_loop` over `body_fun`
/// (jqmc_mcmc.py). Molecular path; Gaussian displacements injected
/// (`injected_normals[mv]`).
#[allow(
    clippy::too_many_arguments,
    reason = "explicit wavefunction, sweep and RNG inputs of the native sweep"
)]
pub fn mcmc_sweep(
    geminal_data: &GeminalData,
    jastrow_data: &JastrowData,
    j3_orb: Option<&OrbData>,
    j3_jmatrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    geminal0: &ArrayView2<f64>,
    geminal_inv0: &ArrayView2<f64>,
    dt: f64,
    epsilon_as: f64,
    key: [u32; 2],
    injected_normals: &[f64],
    num_moves: usize,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    // Reuse the determinant streaming state from the previous sweep of the SAME
    // walker.  This skips rebuilding both spin AO matrices and the ratio state at
    // every measurement step.  The caller must pass a state matching exactly
    // `r_up0`/`r_dn0`; parity/per-step entry points deliberately pass `None`.
    det_state_in: Option<DetRatioState>,
    // J3 state carry: `Some` is the previous sweep's advanced J3-ratio
    // state for the SAME walker at exactly `r_up0`/`r_dn0` (the fast energy chunk
    // threads it across the steps of ONE `run_energy_chunk` call, skipping this
    // sweep's `init_j3_ratio_state` dense rebuild). The sweep advances the J3 aos
    // columns with the SAME single-point AO eval used per move, which on MOLECULAR
    // leaves (n_image==1; caller-gated, mirroring the det-state reuse gate) is
    // bit-equal to the batch init eval -- the carried state differs from a fresh
    // rebuild only at the streamed-GEMM fp floor (fast chain is statistically
    // guarded, not bit-exact). `None` = build fresh here (bitexact chunk /
    // per-step dispatch / first chunk step). Refresh policy: the carry NEVER
    // crosses a `run_energy_chunk` boundary -- the chunk IS the driver's
    // measurement interval (`mcmc_interval`), so every measurement window starts
    // from an exact fresh build; K = chunk length, derived from the driver's
    // cadence, no tuned constant.
    j3_state_in: Option<J3RatioState>,
    // Molecular sphe per-shell distance cutoff for the per-move det orbital eval
    // (fused `compute_mo_single_cutoff_into`: skip shells whose every AO value is
    // below FAST_SCREEN_ABS_BOUND = 1e-12 in absolute AO units, `Shell::pz_val` + sparse-AXPY MO
    // projection over the surviving AOs). Value-changing (drops nonzero AO values
    // < 1e-12; the same `pz_val` screen as the ECP-mesh AO cutoff, issue #100), so it is enabled ONLY on the
    // statistically-guarded FAST chunk -- the bitexact / per-step-dispatch parity
    // paths pass `false` and keep the full eval (byte-contract vs JAX preserved).
    ao_cutoff: bool,
) -> SweepResult {
    // Precompute inv(cell) once for the PBC min-image nearest-atom proposal and
    // the AO-eval home-cell wrap.
    let inv_cell = cell_nmax.map(|(cell, _)| crate::ewald::inv3(cell));
    let nearest_cn: CellNmax<'_, '_> = match (cell_nmax, &inv_cell) {
        (Some((cell, n_max)), Some(inv)) => Some((cell, inv, n_max)),
        _ => None,
    };
    // PBC: wrap the coordinate fed to AO eval (image-fold is home-cell centered).
    let wrap_ci: Option<(&ArrayView2<f64>, &[[f64; 3]; 3])> = match (cell_nmax, &inv_cell) {
        (Some((cell, _)), Some(inv)) => Some((cell, inv)),
        _ => None,
    };
    let n_up = r_up0.nrows();
    let n_dn = r_dn0.nrows();
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut geminal = geminal0.to_owned();
    let mut geminal_inv = geminal_inv0.to_owned();
    let n = geminal.nrows();

    // Streaming states are built from AOs at the INITIAL positions; under PBC
    // those evals must use wrapped coords (consistent with the per-move eval and
    // with the caller's G0, which jqmc builds via compute_geminal's internal wrap).
    let (r_up0_ao, r_dn0_ao) = match wrap_ci {
        Some((cell, inv)) => (wrap_rows(r_up0, cell, inv), wrap_rows(r_dn0, cell, inv)),
        None => (r_up0.to_owned(), r_dn0.to_owned()),
    };
    let (r_up0_ao_v, r_dn0_ao_v) = (r_up0_ao.view(), r_dn0_ao.view());

    // streaming states, built once. Seed the orbital matrices via the SAME
    // per-move `compute_single` path the sweep uses (not the batch `compute`):
    // on a periodic CART pGTO leaf the batch path applies the LEPSBAS far-image
    // truncation that `compute_single` (and JAX) does not, so a batch seed would
    // be ~1e-9 inconsistent with the passed-in JAX `geminal_inv0` and drift the
    // SM-streamed inverse over the sweep for a composite pGTO+B-spline leaf.
    let mut det_state = match det_state_in {
        Some(state) => state,
        None => seed_det_ratio_state(geminal_data, &r_up0_ao_v, &r_dn0_ao_v),
    };
    let mut j3_state: Option<J3RatioState> = match (j3_orb, j3_jmatrix) {
        // Reuse the carried state when supplied (fast chunk, molecular
        // J3 leaf; see the parameter doc) instead of the dense per-sweep rebuild.
        (Some(_), Some(_)) if j3_state_in.is_some() => j3_state_in,
        (Some(orb), Some(jm)) => {
            let aos_up = orb.compute(&r_up0_ao_v);
            let aos_dn = orb.compute(&r_dn0_ao_v);
            Some(crate::timed!(
                "j3.setup",
                crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    jm,
                    &aos_up.view(),
                    &aos_dn.view(),
                    jastrow_data.precision_policy.is_f32("jastrow_ratio"),
                )
            ))
        }
        _ => None,
    };
    // DTN leg streaming state (built once; moved electron's row refreshed on accept).
    let mut dtn_state = jastrow_data
        .dtn
        .as_ref()
        .map(|d| crate::dtn::init_dtn_ratio_state(d, r_up0, r_dn0));

    // --- reusable scratch buffers (no per-move allocation; keeps the constant
    //     C matrix L2-resident across moves) ---
    // AO buffer must hold the full composite (gto + pw); MO buffer must hold the
    // orbital MO count, which can exceed n (the geminal size) for a PW basis.
    let na_of = |o: &OrbData| o.ao_count();
    let nmo_of = |o: &OrbData| {
        o.mo_coefficients
            .as_ref()
            .map(|c| c.nrows())
            .unwrap_or(na_of(o))
    };
    let na_max = na_of(&geminal_data.orb_up).max(na_of(&geminal_data.orb_dn));
    let nmo_max = nmo_of(&geminal_data.orb_up)
        .max(nmo_of(&geminal_data.orb_dn))
        .max(n);
    let mut det_ao = vec![0.0_f64; na_max];
    let mut det_mo = vec![0.0_f64; nmo_max];
    let mut u_buf = vec![0.0_f64; n];
    let mut v_buf = vec![0.0_f64; n];
    let mut ainv_u = vec![0.0_f64; n];
    let mut vt_ainv = vec![0.0_f64; n];
    let mut as_line = vec![0.0_f64; n];
    let mut row_n2 = vec![0.0_f64; n];
    let mut col_n2 = vec![0.0_f64; n];
    // J3 ratio state is at ORBITAL rank.  A composite's output may be much larger
    // than its GTO child's ``aos.num_ao``; an MO wrapper may change the rank again.
    let j3_na = j3_orb.map(nmo_of).unwrap_or(0);
    let mut j3_ao = vec![0.0_f64; j3_na];
    let mut j3_aos_p = vec![0.0_f64; j3_na];
    let mut j3_v = vec![0.0_f64; n_up.max(n_dn)];
    // AS cache for the current (old) state -- recomputed only on accept.
    // epsilon_AS == 0 makes max(epsilon_AS, R_AS)/R_AS identically one for
    // the positive regularizer used here. Production benchmark inputs use
    // zero, so avoid the per-proposal O(N^2) AS formula and, more importantly,
    // the accepted-move full row/column norm rebuild.
    let use_as = epsilon_as > 0.0;
    let mut r_as_o = if use_as {
        as_recompute(
            &geminal.view(),
            &geminal_inv.view(),
            &mut row_n2,
            &mut col_n2,
        )
    } else {
        1.0
    };

    let mut cur_key = key;
    let mut accepted = 0usize;
    // --- TEMP instrumentation (PROFILE=1) ---
    let prof = std::env::var("JQMC_RS_PROFILE").is_ok();
    // `ao_cutoff` is caller-scoped (fast chunk = on, bitexact / per-step = off);
    // see the parameter doc-comment on the fn signature.
    let (mut t_orb, mut t_det, mut t_as, mut t_jas, mut t_adv) =
        (0u128, 0u128, 0u128, 0u128, 0u128);
    let mut t_rng = 0u128;
    let mut t_accept = 0u128;
    let (mut t_ao, mut t_mo) = (0u128, 0u128);
    use std::time::Instant;

    // FAST chunk only (`ao_cutoff`): periodic J1/J2 ratio through the fastmath
    // sincos/exp twin (floor-compared); the bitexact / per-step paths keep libm.
    let j12_fast_prep = if ao_cutoff {
        jastrow_factor::J12FastPeriodic::new(jastrow_data)
    } else {
        None
    };
    for injected_normals_mv in injected_normals[..num_moves].iter() {
        let ts_rng = if prof { Some(Instant::now()) } else { None };
        let d = move_draws(cur_key, n_up, n_dn);
        cur_key = d.final_key;
        let is_up = d.is_up;
        let k = d.selected_index;

        let old_r = if is_up {
            [r_up[[k, 0]], r_up[[k, 1]], r_up[[k, 2]]]
        } else {
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        };

        // proposal
        let (nearest, norm_r_r) = nearest_atom_dispatch(positions, old_r, nearest_cn);
        let f_l = kato_f_l(charges[nearest], norm_r_r);
        let g = *injected_normals_mv * (f_l * dt);
        let mut new_r = old_r;
        new_r[d.axis] += g;
        let (nearest2, norm2) = nearest_atom_dispatch(positions, new_r, nearest_cn);
        let f_prime = kato_f_l(charges[nearest2], norm2);
        let dist2 = {
            let dx = new_r[0] - old_r[0];
            let dy = new_r[1] - old_r[1];
            let dz = new_r[2] - old_r[2];
            dx * dx + dy * dy + dz * dz
        };
        let t_ratio_val = t_ratio(f_l, f_prime, dist2, dt);
        if let Some(t) = ts_rng {
            t_rng += t.elapsed().as_nanos();
        }

        // AO-eval coordinate: home-cell-wrapped under PBC (storage stays new_r,
        // unwrapped); identity for molecular. min-image/smooth use unwrapped new_r.
        let ao_pos = match wrap_ci {
            Some((cell, inv)) => wrap_into_cell(new_r, cell, inv),
            None => new_r,
        };

        // det ratio from the streaming state (no re-init). Orbital column at the
        // new position written ONCE into det_mo (reused by v/u + advance).
        let ts = if prof { Some(Instant::now()) } else { None };
        let orb = if is_up {
            &geminal_data.orb_up
        } else {
            &geminal_data.orb_dn
        };
        let nmo;
        // split AO vs MO for diagnosis (profiling only)
        let prof_split_c = if prof && !ao_cutoff && orb.bspline.is_none() && orb.aos.n_image == 1 {
            orb.mo_coefficients.as_ref()
        } else {
            None
        };
        if let Some(c) = prof_split_c {
            let na = orb.aos.num_ao;
            nmo = c.nrows();
            crate::atomic_orbital::compute_aos_single_into(&orb.aos, ao_pos, &mut det_ao[..na]);
            if let Some(t) = ts {
                t_ao += t.elapsed().as_nanos();
            }
            let ts2 = Instant::now();
            crate::determinant::project_faer(
                c.as_slice().unwrap(),
                &det_ao[..na],
                na,
                nmo,
                &mut det_mo[..nmo],
            );
            t_mo += ts2.elapsed().as_nanos();
        } else {
            nmo = eval_orb_into(orb, ao_pos, &mut det_ao, &mut det_mo, ao_cutoff);
            if let Some(t) = ts {
                t_orb += t.elapsed().as_nanos();
            }
        }
        let det_orb_new = &det_mo[..nmo];

        // v/u into reused buffers; u (up) or v (dn) is the e_k unit column.
        let ts = if prof { Some(Instant::now()) } else { None };
        if is_up {
            compute_v_up_move_from_orb_into(geminal_data, &det_state, k, det_orb_new, &mut v_buf);
            for x in u_buf.iter_mut() {
                *x = 0.0;
            }
            u_buf[k] = 1.0;
        } else {
            compute_u_dn_move_from_orb_into(&det_state, k, det_orb_new, &mut u_buf);
            for x in v_buf.iter_mut() {
                *x = 0.0;
            }
            v_buf[k] = 1.0;
        }
        // det ratio + rank-1 factors (no 80x80 copy); inverse/geminal updated
        // IN PLACE only on accept.
        let det_t = sm_ratio_into(
            &geminal_inv.view(),
            &u_buf,
            &v_buf,
            &mut ainv_u,
            &mut vt_ainv,
        );
        if let Some(t) = ts {
            t_det += t.elapsed().as_nanos();
        }

        // AS ratio: proposed via incremental rank-1 formula (no materialization);
        // old state cached, recomputed only on accept.
        let ts = if prof { Some(Instant::now()) } else { None };
        let r_as_ratio = if use_as {
            // Issue #134: the moved line is rebuilt from det_orb_new only when the
            // incremental G_k + v / cross norms lost digits.
            let r_as_p = as_proposed(
                &geminal.view(),
                &geminal_inv.view(),
                &row_n2,
                &col_n2,
                &u_buf,
                &v_buf,
                &ainv_u,
                &vt_ainv,
                det_t,
                is_up,
                k,
                &mut as_line,
                &mut |out: &mut [f64]| {
                    if is_up {
                        determinant::compute_up_row_from_orb_into(
                            geminal_data,
                            &det_state,
                            det_orb_new,
                            out,
                        )
                    } else {
                        determinant::compute_dn_col_from_orb_into(&det_state, det_orb_new, out)
                    }
                },
            );
            determinant::as_guiding_ratio(r_as_p, r_as_o, epsilon_as)
        } else {
            1.0
        };
        if let Some(t) = ts {
            t_as += t.elapsed().as_nanos();
        }

        // Jastrow J1/J2 + J3. J3 AO column at new_r into reused buffer.
        let ts = if prof { Some(Instant::now()) } else { None };
        let j12 = match &j12_fast_prep {
            Some(prep) => prep
                .delta(&r_up.view(), &r_dn.view(), is_up, k, new_r)
                .exp(),
            None => jastrow_factor::jastrow_1b2b_ratio(
                jastrow_data,
                &r_up.view(),
                &r_dn.view(),
                is_up,
                k,
                new_r,
            ),
        };
        let has_j3 = j3_orb.is_some() && j3_state.is_some();
        if has_j3 {
            eval_ao_into(j3_orb.unwrap(), ao_pos, &mut j3_ao, ao_cutoff);
        }
        let j3r = match (j3_jmatrix, &j3_state) {
            (Some(jm), Some(st)) if has_j3 && ao_cutoff => {
                j3_ratio_fast_linear(jm, st, is_up, k, &j3_ao[..j3_na])
            }
            (Some(jm), Some(st)) if has_j3 => {
                j3_ratio_with_scratch(jm, st, is_up, k, &j3_ao[..j3_na], &mut j3_aos_p, &mut j3_v)
            }
            _ => 1.0,
        };
        let jdtn = match (&jastrow_data.dtn, &dtn_state) {
            (Some(dtn), Some(st)) => crate::dtn::compute_dtn_ratio_streaming(
                dtn,
                st,
                &r_up.view(),
                &r_dn.view(),
                is_up,
                k,
                new_r,
            ),
            _ => 1.0,
        };
        let j_ratio = crate::jastrow_factor::multiply_ratio_components(
            jastrow_data,
            crate::jastrow_factor::multiply_ratio_components(jastrow_data, j12, j3r),
            jdtn,
        );
        if let Some(t) = ts {
            t_jas += t.elapsed().as_nanos();
        }

        let wf = j_ratio * det_t;
        let comb = r_as_ratio * wf;
        let r_ratio = comb * comb;
        let acc = accept_clamp(r_ratio * t_ratio_val);

        if d.metropolis_b < acc {
            let ts_accept = if prof { Some(Instant::now()) } else { None };
            accepted += 1;
            if is_up {
                r_up[[k, 0]] = new_r[0];
                r_up[[k, 1]] = new_r[1];
                r_up[[k, 2]] = new_r[2];
            } else {
                r_dn[[k, 0]] = new_r[0];
                r_dn[[k, 1]] = new_r[1];
                r_dn[[k, 2]] = new_r[2];
            }
            // rank-1 update of G / G^-^1 in place (the only O(N^2) write, ~1/3 moves)
            sm_apply_inplace(
                &mut geminal_inv,
                &mut geminal,
                &u_buf,
                &v_buf,
                &ainv_u,
                &vt_ainv,
                det_t,
                is_up,
                k,
            );
            // refresh AS caches (F, row/col norms, old-state R_AS) from new state
            if use_as {
                r_as_o = as_recompute(
                    &geminal.view(),
                    &geminal_inv.view(),
                    &mut row_n2,
                    &mut col_n2,
                );
            }
            let ts = if prof { Some(Instant::now()) } else { None };
            advance_det_ratio_state_from_orb(geminal_data, &mut det_state, is_up, k, det_orb_new);
            if let (Some(jm), Some(st)) = (j3_jmatrix, j3_state.as_mut()) {
                if has_j3 {
                    crate::timed!(
                        "j3.advance",
                        advance_j3_ratio_state(jm, st, is_up, k, &j3_ao[..j3_na])
                    );
                }
            }
            if let (Some(dtn), Some(st)) = (&jastrow_data.dtn, dtn_state.as_mut()) {
                crate::dtn::advance_dtn_ratio_state(dtn, st, &r_up.view(), &r_dn.view(), is_up, k);
            }
            if let Some(t) = ts {
                t_adv += t.elapsed().as_nanos();
            }
            if let Some(t) = ts_accept {
                t_accept += t.elapsed().as_nanos();
            }
        }
    }
    if prof {
        let n = num_moves as f64;
        eprintln!(
            "[PROFILE us/move] rng+prop={:.2} orb_eval={:.2}(AO={:.2} MO={:.2}) det_v/u+sm={:.2} as={:.2} jastrow={:.2} accept={:.2} advance={:.2}",
            t_rng as f64 / n / 1000.0,
            t_orb as f64 / n / 1000.0,
            t_ao as f64 / n / 1000.0,
            t_mo as f64 / n / 1000.0,
            t_det as f64 / n / 1000.0,
            t_as as f64 / n / 1000.0,
            t_jas as f64 / n / 1000.0,
            t_accept as f64 / n / 1000.0,
            t_adv as f64 / n / 1000.0,
        );
    }

    SweepResult {
        accepted,
        r_up,
        r_dn,
        geminal,
        geminal_inv,
        det_state,
        j3_state,
        final_key: cur_key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kato_and_tratio_basic() {
        // f_l with Z=1, r=0 -> 1.0
        assert!((kato_f_l(1.0, 0.0) - 1.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
        // T_ratio with f_l==f_prime_l -> exp(0)*1 = 1
        assert!(
            (t_ratio(0.5, 0.5, 0.3, 1.0) - 1.0).abs()
                < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE
        );
    }

    #[test]
    fn metropolis_threshold() {
        assert!(metropolis_accept(2.0, 1.0, 0.5)); // acc=min(1,2)=1 > 0.5
        assert!(!metropolis_accept(0.4, 1.0, 0.5)); // acc=0.4 < 0.5
    }

    #[test]
    fn metropolis_nan_ratio_rejects_like_xla() {
        // `jnp.min` propagates NaN and `b < NaN` is false,
        // so XLA REJECTS a degenerate (NaN) acceptance ratio. Bare f64::min
        // absorbed the NaN (acc == 1.0 -> unconditional accept). The clamp must
        // reject for EVERY b in [0,1), including b == 0.
        assert!(accept_clamp(f64::NAN).is_nan());
        assert!(!metropolis_accept(f64::NAN, 1.0, 0.0));
        assert!(!metropolis_accept(f64::NAN, 1.0, 0.999999));
        assert!(!metropolis_accept(1.0, f64::NAN, 0.0));
        // non-NaN behaviour unchanged by the clamp
        assert!((accept_clamp(2.0) - 1.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_TIGHT);
        assert!((accept_clamp(0.4) - 0.4).abs() < crate::setting_consts::ATOL_PYTHON_FP64_TIGHT);
    }

    #[test]
    fn nearest_atom_picks_closest() {
        let pos = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let (i, d) = nearest_atom_molecular(&pos, [4.0, 0.0, 0.0]);
        assert_eq!(i, 1);
        assert!((d - 1.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }
}

/// Complex sweep result. Mirrors `SweepResult` with complex geminal/inverse + the
/// complex streaming det-ratio state.
pub struct SweepResultC {
    pub accepted: usize,
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    pub geminal: Array2<numpy::Complex64>,
    pub geminal_inv: Array2<numpy::Complex64>,
    pub det_state: crate::complex_det::DetRatioStateC,
    pub final_key: [u32; 2],
    /// The sweep's accepted moves in acceptance order (`record_moves` only;
    /// empty otherwise). Consumers that advance per-move state outside the
    /// sweep (the FAST chunk's carried kinetic tables) replay them in order:
    /// rank-1 updates do not commute.
    pub accepted_moves: Vec<AcceptedMoveC>,
}

/// One accepted move of [`mcmc_sweep_complex_carry`]: spin, electron index and
/// the RAW (unwrapped) old and new positions.
#[derive(Clone, Copy, Debug)]
pub struct AcceptedMoveC {
    pub is_up: bool,
    pub index: usize,
    pub old_pos: [f64; 3],
    pub new_pos: [f64; 3],
}

/// Issue #131: exponent `e` with `x` in `[2^(e-1), 2^e)` for a finite
/// positive `x` (subnormals included).
fn as_pow2_exponent(x: f64) -> i32 {
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        let y = x * 2f64.powi(54);
        ((y.to_bits() >> 52) & 0x7ff) as i32 - 1022 - 54
    } else {
        biased - 1022
    }
}

/// `x * 2^e` by exact power-of-two steps that stay in the normal exponent range,
/// so no intermediate overflows or underflows unless the result does.
fn as_mul_pow2(mut x: numpy::Complex64, mut e: i32) -> numpy::Complex64 {
    let pow2 = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    while e > 1023 {
        x *= pow2(1023);
        e -= 1023;
    }
    while e < -1022 {
        x *= pow2(-1022);
        e += 1022;
    }
    x * pow2(e)
}

/// Issue #131: range-safe complex quotient `a / b` for the AS proposed
/// state. `num_complex` forms `|b|^2` and `a * conj(b)` unscaled, so a finite
/// quotient such as `ainv_u / R` with `|R| ~ 5e161` (or `|ainv_u| |R|` above
/// ~1.8e308) came out `NaN`. Both operands are first brought to a max-component
/// magnitude in `[0.5, 1)` by exact powers of two and the exponent difference is
/// restored afterwards. Where the unscaled division stays in the normal range
/// every step is an exact power-of-two rescaling of it, so the result is
/// bit-identical to `a / b`; a zero or non-finite operand falls back to `a / b`.
fn as_div_range_safe(a: numpy::Complex64, b: numpy::Complex64) -> numpy::Complex64 {
    let a_max = a.re.abs().max(a.im.abs());
    let b_max = b.re.abs().max(b.im.abs());
    if !(a_max.is_finite() && b_max.is_finite() && a_max > 0.0 && b_max > 0.0) {
        return a / b;
    }
    let (ea, eb) = (as_pow2_exponent(a_max), as_pow2_exponent(b_max));
    let q = as_mul_pow2(a, -ea) / as_mul_pow2(b, -eb);
    as_mul_pow2(q, ea - eb)
}

/// Native MCMC sweep for a COMPLEX Bloch twist geminal. Same scaffolding as
/// `mcmc_sweep` (threefry move draws, Kato drift proposal, T_ratio) but the determinant
/// ratio / inverse maintenance / AS regularization are complex; acceptance uses
/// `r_ratio = |R_AS*J*det_t|^2` (norm_sqr). Jastrow = J1/J2/J3/DTN; these factors
/// are real and geometry-only, so they are valid for complex systems.
/// AS proposed factor uses a materialized proposed G^-^1 (correctness-first; only when
/// epsilon_as>0 -- with epsilon_as=0 the AS ratio is identically 1).
#[allow(clippy::too_many_arguments)]
pub fn mcmc_sweep_complex(
    gem: &crate::complex_det::ComplexGeminal,
    jastrow_data: &JastrowData,
    j3c: Option<&crate::complex_det::ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    // Caller-supplied STREAMING state at `(r_up0, r_dn0)`: the complex geminal and its
    // inverse, threaded in exactly as the real `mcmc_sweep` threads `geminal0`/`geminal_inv0`.
    // The driver already carries them across steps (the previous sweep's emitted
    // `geminal`/`geminal_inv`), so rebuilding them here would both duplicate O(N^3) work and
    // DISCARD the Sherman-Morrison chain -- a fresh `inverse_complex` NaNs at a near-singular
    // geminal where the AS-regularized swept inverse stays finite.
    geminal0: &ArrayView2<numpy::Complex64>,
    geminal_inv0: &ArrayView2<numpy::Complex64>,
    dt: f64,
    epsilon_as: f64,
    key: [u32; 2],
    injected_normals: &[f64],
    num_moves: usize,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
) -> SweepResultC {
    mcmc_sweep_complex_carry(
        gem,
        jastrow_data,
        j3c,
        j3_jm,
        positions,
        charges,
        r_up0,
        r_dn0,
        geminal0,
        geminal_inv0,
        dt,
        epsilon_as,
        key,
        injected_normals,
        num_moves,
        cell_nmax,
        None,
        false,
    )
}

/// [`mcmc_sweep_complex`] with the FAST chunk's carry inputs:
/// - `det_state_in`: the previous sweep's advanced `DetRatioStateC` for the SAME
///   walker at exactly `(r_up0, r_dn0)` (the FAST complex chunk threads it across
///   the steps of ONE `run_energy_chunk` call, together with the carried `G` /
///   SM inverse). `None` builds it here with the batch `init_det_ratio_state_complex`
///   (the per-step dispatch, the native driver and the first chunk step). The
///   carried columns come from the same `orb_single_complex` evaluations the sweep
///   uses per move, so a carried state differs from a fresh build only at the
///   fp floor (FAST chain, never on a bit-exact path).
/// - `record_moves`: report the accepted moves in order (`SweepResultC::accepted_moves`).
///   Both off is the [`mcmc_sweep_complex`] entry, bit-identical to it.
#[allow(clippy::too_many_arguments)]
pub fn mcmc_sweep_complex_carry(
    gem: &crate::complex_det::ComplexGeminal,
    jastrow_data: &JastrowData,
    j3c: Option<&crate::complex_det::ComplexOrb>,
    j3_jm: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    geminal0: &ArrayView2<numpy::Complex64>,
    geminal_inv0: &ArrayView2<numpy::Complex64>,
    dt: f64,
    epsilon_as: f64,
    key: [u32; 2],
    injected_normals: &[f64],
    num_moves: usize,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    det_state_in: Option<crate::complex_det::DetRatioStateC>,
    record_moves: bool,
) -> SweepResultC {
    use crate::complex_det::{
        advance_det_ratio_state_complex, compute_as_regularization_factor_complex,
        compute_dn_col_complex, compute_u_dn_move_complex, compute_up_row_complex,
        compute_v_up_move_complex, init_det_ratio_state_complex, orb_single_complex,
        sm_apply_inplace_complex, sm_ratio_complex,
    };
    use numpy::Complex64;
    // Complex-Bloch J3: the J3 AO basis is A = Re(complex sphe AO); feed Re(*) of the
    // complex J3 orb evals into the EXISTING real j3-ratio streaming machinery.
    let re2 = |m: &Array2<Complex64>| m.mapv(|x| x.re);
    let re_col = |orb: &crate::complex_det::ComplexOrb, pos: [f64; 3]| -> Vec<f64> {
        orb_single_complex(orb, pos).iter().map(|x| x.re).collect()
    };
    let inv_cell = cell_nmax.map(|(cell, _)| crate::ewald::inv3(cell));
    let nearest_cn: CellNmax<'_, '_> = match (cell_nmax, &inv_cell) {
        (Some((cell, n_max)), Some(inv)) => Some((cell, inv, n_max)),
        _ => None,
    };
    let wrap_ci: Option<(&ArrayView2<f64>, &[[f64; 3]; 3])> = match (cell_nmax, &inv_cell) {
        (Some((cell, _)), Some(inv)) => Some((cell, inv)),
        _ => None,
    };
    // NOTE: the complex AO (`ComplexOrb`) does its OWN far-field wrap+restore internally
    // (WrapCtx), exactly like jqmc's `_ComplexBlochPGTOMixin`. So -- unlike the real sweep,
    // whose image-tiled AO needs a sweep-level `wrap_into_cell` -- here we pass UNWRAPPED
    // coords everywhere and let ComplexOrb wrap. Pre-wrapping would double-wrap (and apply
    // the restore phase at the wrong floor), diverging from jqmc once an electron leaves
    // the home cell. `wrap_ci` is unused for the complex path (kept for nearest-atom only).
    let _ = &wrap_ci;
    let n_up = r_up0.nrows();
    let n_dn = r_dn0.nrows();
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut geminal = geminal0.to_owned();
    let mut geminal_inv = geminal_inv0.to_owned();
    let n = geminal.nrows();
    let mut det_state = match det_state_in {
        Some(state) => state,
        None => crate::timed!(
            "sweep_c.det_init",
            init_det_ratio_state_complex(gem, r_up0, r_dn0)
        ),
    };
    let mut dtn_state = crate::timed!(
        "sweep_c.dtn_init",
        jastrow_data
            .dtn
            .as_ref()
            .map(|d| crate::dtn::init_dtn_ratio_state(d, r_up0, r_dn0))
    );
    // Complex J3 streaming state from Re(complex sphe AO) at the initial config.
    let mut j3_state: Option<J3RatioState> = crate::timed!(
        "sweep_c.j3_init",
        match (j3c, j3_jm) {
            (Some(orb), Some(jm)) => {
                let au = re2(&orb.value(r_up0));
                let ad = re2(&orb.value(r_dn0));
                Some(crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    jm,
                    &au.view(),
                    &ad.view(),
                    jastrow_data.precision_policy.is_f32("jastrow_ratio"),
                ))
            }
            _ => None,
        }
    );
    let mut accepted_moves: Vec<AcceptedMoveC> = Vec::new();

    let mut ainv_u = vec![Complex64::new(0.0, 0.0); n];
    let mut vt_ainv = vec![Complex64::new(0.0, 0.0); n];
    let mut r_as_o = compute_as_regularization_factor_complex(&geminal.view(), &geminal_inv.view());

    let mut cur_key = key;
    let mut accepted = 0usize;
    for injected_normals_mv in injected_normals[..num_moves].iter() {
        let d = move_draws(cur_key, n_up, n_dn);
        cur_key = d.final_key;
        let is_up = d.is_up;
        let k = d.selected_index;
        let old_r = if is_up {
            [r_up[[k, 0]], r_up[[k, 1]], r_up[[k, 2]]]
        } else {
            [r_dn[[k, 0]], r_dn[[k, 1]], r_dn[[k, 2]]]
        };
        let (nearest, norm_r_r) = nearest_atom_dispatch(positions, old_r, nearest_cn);
        let f_l = kato_f_l(charges[nearest], norm_r_r);
        let g = *injected_normals_mv * (f_l * dt);
        let mut new_r = old_r;
        new_r[d.axis] += g;
        let (nearest2, norm2) = nearest_atom_dispatch(positions, new_r, nearest_cn);
        let f_prime = kato_f_l(charges[nearest2], norm2);
        let dist2 = {
            let (dx, dy, dz) = (
                new_r[0] - old_r[0],
                new_r[1] - old_r[1],
                new_r[2] - old_r[2],
            );
            dx * dx + dy * dy + dz * dz
        };
        let t_ratio_val = t_ratio(f_l, f_prime, dist2, dt);
        // complex orbital column at the UNWRAPPED proposed position (ComplexOrb wraps
        // internally) + rank-1 v/u
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        let ao_new = orb_single_complex(orb, new_r);
        let (u, v): (Vec<Complex64>, Vec<Complex64>) = if is_up {
            let v = compute_v_up_move_complex(gem, &det_state, k, &ao_new);
            let mut u = vec![Complex64::new(0.0, 0.0); n];
            u[k] = Complex64::new(1.0, 0.0);
            (u, v)
        } else {
            let u = compute_u_dn_move_complex(&det_state, k, &ao_new);
            let mut v = vec![Complex64::new(0.0, 0.0); n];
            v[k] = Complex64::new(1.0, 0.0);
            (u, v)
        };
        let det_t = sm_ratio_complex(&geminal_inv.view(), &u, &v, &mut ainv_u, &mut vt_ainv);

        // AS ratio (materialized proposed G^-^1; only when regularization is on).
        let r_as_ratio = if epsilon_as > 0.0 {
            let mut gi_p = geminal_inv.clone();
            // Issue #131: range-safe quotients by det_t (|R| beyond ~1e154).
            for i in 0..n {
                let ai = as_div_range_safe(ainv_u[i], det_t);
                for j in 0..n {
                    let cur = gi_p[[i, j]];
                    gi_p[[i, j]] = cur - ai * vt_ainv[j];
                }
            }
            // Issues #131/#134: moved line of G_p rebuilt from the new orbital
            // values (G + delta cancels for a tiny ratio) and the moved inverse
            // column / row in its exact form X e_k / R (e_kT X / R).
            let mut g_p = geminal.clone();
            if is_up {
                let row = compute_up_row_complex(gem, &det_state, &ao_new);
                for j in 0..n {
                    g_p[[k, j]] = row[j];
                    gi_p[[j, k]] = as_div_range_safe(ainv_u[j], det_t);
                }
            } else {
                let col = compute_dn_col_complex(&det_state, &ao_new);
                for i in 0..n {
                    g_p[[i, k]] = col[i];
                    gi_p[[k, i]] = as_div_range_safe(vt_ainv[i], det_t);
                }
            }
            let r_as_p = compute_as_regularization_factor_complex(&g_p.view(), &gi_p.view());
            crate::determinant::as_guiding_ratio(r_as_p, r_as_o, epsilon_as)
        } else {
            1.0
        };

        // Jastrow ratios. J1/J2/DTN (real, geometry-only) + complex-Bloch J3 = Re(complex
        // sphe AO) fed into the real j3-ratio machinery at the UNWRAPPED proposed position.
        let j12 = crate::jastrow_factor::jastrow_1b2b_ratio(
            jastrow_data,
            &r_up.view(),
            &r_dn.view(),
            is_up,
            k,
            new_r,
        );
        let jdtn = match (&jastrow_data.dtn, &dtn_state) {
            (Some(dtn), Some(st)) => crate::dtn::compute_dtn_ratio_streaming(
                dtn,
                st,
                &r_up.view(),
                &r_dn.view(),
                is_up,
                k,
                new_r,
            ),
            _ => 1.0,
        };
        let mut j3_ao_re: Vec<f64> = Vec::new();
        let j3r = match (j3c, j3_jm, &j3_state) {
            (Some(orb), Some(jm), Some(st)) => {
                j3_ao_re = re_col(orb, new_r);
                j3_ratio(jm, st, is_up, k, &j3_ao_re)
            }
            _ => 1.0,
        };
        let j_ratio = crate::jastrow_factor::multiply_ratio_components(
            jastrow_data,
            crate::jastrow_factor::multiply_ratio_components(jastrow_data, j12, j3r),
            jdtn,
        );
        let wf = Complex64::new(j_ratio * r_as_ratio, 0.0) * det_t;
        let r_ratio = wf.norm_sqr();
        let acc = accept_clamp(r_ratio * t_ratio_val);

        if d.metropolis_b < acc {
            accepted += 1;
            if is_up {
                r_up[[k, 0]] = new_r[0];
                r_up[[k, 1]] = new_r[1];
                r_up[[k, 2]] = new_r[2];
            } else {
                r_dn[[k, 0]] = new_r[0];
                r_dn[[k, 1]] = new_r[1];
                r_dn[[k, 2]] = new_r[2];
            }
            sm_apply_inplace_complex(
                &mut geminal_inv,
                &mut geminal,
                &u,
                &v,
                &ainv_u,
                &vt_ainv,
                det_t,
                is_up,
                k,
            );
            r_as_o = compute_as_regularization_factor_complex(&geminal.view(), &geminal_inv.view());
            advance_det_ratio_state_complex(gem, &mut det_state, is_up, k, &ao_new);
            if let (Some(dtn), Some(st)) = (&jastrow_data.dtn, dtn_state.as_mut()) {
                crate::dtn::advance_dtn_ratio_state(dtn, st, &r_up.view(), &r_dn.view(), is_up, k);
            }
            if let (Some(jm), Some(st)) = (j3_jm, j3_state.as_mut()) {
                advance_j3_ratio_state(jm, st, is_up, k, &j3_ao_re);
            }
            if record_moves {
                accepted_moves.push(AcceptedMoveC {
                    is_up,
                    index: k,
                    old_pos: old_r,
                    new_pos: new_r,
                });
            }
        }
    }

    SweepResultC {
        accepted,
        r_up,
        r_dn,
        geminal,
        geminal_inv,
        det_state,
        final_key: cur_key,
        accepted_moves,
    }
}
