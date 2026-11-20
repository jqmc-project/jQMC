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

//! GFMC (lattice-regularized / continuum Green's-function Monte Carlo) kernels
//! and projection drivers, composed from the verified VMC pieces (move loop +
//! local energy).
//!
//! Ported 1:1 from `jqmc/jqmc_gfmc.py`, among others:
//!   - [`compute_g_l`]        <- `_compute_G_L`           (branching-weight accumulation)
//!   - [`reconfigure_indices`] <- the single-process reconfiguration step
//!
//! The walker propagation reuses the native VMC kernels. The importance-weight
//! bookkeeping (G_L) and the stochastic reconfiguration (branching) that turn a
//! set of weighted walkers into an unweighted resampled set are pure array
//! math -- no AO eval -- so they are testable independently of the AO kernels.

use crate::complex_det::det_ratio_complex;
use numpy::ndarray::{Array2, ArrayView2};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// One projection mesh candidate `(is_up, electron, candidate position, matrix element)`;
/// [`MeshCandidateC`] carries the complex-twist element.
pub(crate) type MeshCandidate = (bool, usize, [f64; 3], f64);
pub(crate) type MeshCandidateC = (bool, usize, [f64; 3], Complex64);
/// ECP mesh geometry `(positions_up, angular_up, positions_dn, angular_dn)`.
type EcpMeshGeometry<'a> = (&'a [[f64; 3]], &'a [f64], &'a [[f64; 3]], &'a [f64]);

// ---------------------------------------------------------------------------
// Route-engagement counters (observability only -- no effect on numerics).
// Exposed to Python as `jqmc_rs.gfmc_route_counters()` so per-fixture tests can
// PROVE which projection-scoped caches actually engaged instead of assuming.
// Counting is disabled in normal production runs. `gfmc_route_counters_reset`
// enables it for the explicit route-engagement test window. This avoids
// cache-line contention from global `fetch_add` operations in the 128-way
// candidate hot loops while preserving the existing diagnostic surface.
// ---------------------------------------------------------------------------

static GFMC_ROUTE_COUNTERS_ON: AtomicBool = AtomicBool::new(false);

#[inline]
fn bump_route_counter(counter: &AtomicU64) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Always-on (not gated by `GFMC_ROUTE_COUNTERS_ON`) witness that a
/// projection mesh took the full-value J3 slow fallback (no streaming J3 state
/// / unsupported candidate class). Production reads and clears it through
/// `jqmc_rs.gfmc_take_j3_slow_fallback_seen()` to log a once-per-run warning,
/// since that route silently costs O(10-100x) in LRDMC wall time. Set once per
/// mesh evaluation (a plain relaxed store; no read-modify-write contention).
static GFMC_J3_SLOW_FALLBACK_SEEN: AtomicBool = AtomicBool::new(false);

/// Count a J3 slow-fallback mesh evaluation (route counter, test window only)
/// and raise the always-on production witness [`GFMC_J3_SLOW_FALLBACK_SEEN`].
#[inline]
fn note_j3_slow_fallback(counter: &AtomicU64) {
    bump_route_counter(counter);
    if !GFMC_J3_SLOW_FALLBACK_SEEN.load(Ordering::Relaxed) {
        GFMC_J3_SLOW_FALLBACK_SEEN.store(true, Ordering::Relaxed);
    }
}

/// Return whether any projection mesh took the J3 slow fallback since the last
/// call, and clear the witness (see [`GFMC_J3_SLOW_FALLBACK_SEEN`]).
pub fn gfmc_take_j3_slow_fallback_seen() -> bool {
    GFMC_J3_SLOW_FALLBACK_SEEN.swap(false, Ordering::Relaxed)
}

/// Candidate-group consumptions of the streaming DTN mesh cache
/// (`compute_dtn_ratio_mesh_cached` calls through a [`MeshFastState`]).
static GFMC_DTN_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
/// Projection intervals whose carried state ALSO qualified for the molecular
/// determinant/J3 candidate + derivative fast tables (`fast_mol_kinetic_ok`).
static GFMC_DET_FAST_INTERVALS: AtomicU64 = AtomicU64::new(0);
/// Projection intervals whose carried state did NOT qualify for the molecular
/// fast tables (DTN / ECP-geometry / ECP-local / J1J2 caches always persist).
/// These intervals may still engage the per-basis candidate and derivative
/// routes; the per-class counters below report that engagement.
static GFMC_REDUCED_STATE_INTERVALS: AtomicU64 = AtomicU64::new(0);
// Per-basis candidate-table events. "Build" = a full candidate table
// was (re)built (interval rebuild OR a per-substep owned build when the mesh
// cache did not engage); "Refresh" = only the moved electron's rows were
// re-evaluated after a hop. One event per table (det-up / det-dn / J3 count
// separately, since the det and J3 bases are independent).
static GFMC_CAND_BUILD_MOL: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_REFRESH_MOL: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_BUILD_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_REFRESH_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_BUILD_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_REFRESH_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_BUILD_PGTOPW: AtomicU64 = AtomicU64::new(0);
static GFMC_CAND_REFRESH_PGTOPW: AtomicU64 = AtomicU64::new(0);
/// Fast-mesh evaluations in which at least one determinant spin fell back to
/// the slow (per-point `compute_single`) candidate path because its basis
/// class is unsupported (empty placeholder, simultaneous PW+B-spline leaves,
/// or a periodic leaf without a PBC cell). No silent fallback.
static GFMC_CAND_DET_SLOW: AtomicU64 = AtomicU64::new(0);
/// Fast-mesh evaluations with a J3 term that could not use a candidate table
/// (streaming J3 state absent -- MO-wrapped / non-contiguous `j_matrix` /
/// unsupported periodic representation -- or a basis class without a J3
/// candidate batch). These take the full-value or per-point J3 fallback.
static GFMC_J3_SLOW_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// Builds of the basis-generic determinant derivative cache
/// (`init_det_grad_lap_cache_folded`); the molecular twin is counted by
/// `det_fast_intervals` engagement as before.
static GFMC_DET_GRAD_GENERIC_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Molecular MO-space determinant candidate batches projected by the
/// lane-tiled kernel `lane_tiled_mo_projection_into` (row-major transposed MO
/// coefficients present; IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001).
static GFMC_CAND_MO_TILED: AtomicU64 = AtomicU64::new(0);
/// Those batches that took the dense per-(point, MO) `dot_product_fast` loop
/// (no row-major transposed coefficient copy on the leaf).
static GFMC_CAND_MO_DENSE: AtomicU64 = AtomicU64::new(0);
/// Real FAST ECP mesh (electron, nearest-nucleus) quadrature groups evaluated
/// (IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001) ...
static GFMC_ECP_LIVE_GROUPS: AtomicU64 = AtomicU64::new(0);
/// ... and groups of electrons whose every quadrature amplitude is exactly 0.0
/// (e.g. a zero-coefficient non-local channel, or full underflow), skipped.
static GFMC_ECP_ZERO_GROUPS: AtomicU64 = AtomicU64::new(0);
/// Real FAST GFMC ECP (electron, non-local term) pairs kept / dropped by the
/// radial cut `|v_t| < FAST_ECP_NONLOCAL_RADIAL_CUT`, counted per geometry
/// (re)build of an electron (IMPL-RS-GFMC-FAST-ECP-RADIAL-CUT-001).
static GFMC_ECP_CUT_KEPT_PAIRS: AtomicU64 = AtomicU64::new(0);
static GFMC_ECP_CUT_DROPPED_PAIRS: AtomicU64 = AtomicU64::new(0);
/// Unmoved DTN candidate-table rows skipped as provably far from the mover
/// (IMPL-RS-DTN-TABPIP-FAR-ROW-SKIP-001), counted in `dtn.rs`.
pub(crate) static GFMC_DTN_FAR_ROWS: AtomicU64 = AtomicU64::new(0);
/// Complex-twist grouped ECP mesh-force candidates whose Jastrow factor came from the
/// `jastrow_ratio`-zone fallback ratio (`gfmc_forces::MeshJasC::ratio`, issue #757).
pub(crate) static GFMC_CPLX_ECPF_GROUPED_JAS_RATIOS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn bump_route_counter_by(counter: &AtomicU64, n: u64) {
    if n > 0 && GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        counter.fetch_add(n, Ordering::Relaxed);
    }
}
// Real-path derivative-cache MISS counters. These make an unexpected full
// rebuild observable instead of indistinguishable from a cache hit in numerical
// parity tests. Supported fixtures expect zero misses in every precision mode.
/// Sub-steps that recomputed the determinant grad/Laplacian in full because
/// `fast_det_gradlap_ok` refused (an `Unsupported` orbital class: empty
/// placeholder, periodic tiling without a cell, or simultaneous PW+B-spline
/// leaves). The molecular RAW-AO AGP/AGPn builds the basis-generic cache ([13])
/// instead.
static GFMC_DET_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
/// Sub-steps that recomputed the J1/J2 grad/Laplacian in full because no
/// compatible cache state was available.
static GFMC_J12_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
/// Sub-steps that recomputed the J3 grad/Laplacian in full because a present
/// J3 term had no compatible streaming state.
static GFMC_J3_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
/// Sub-steps where a J1/J2 full rebuild also recomputed a present DTN
/// grad/Laplacian instead of using the carried DTN table.
static GFMC_DTN_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
// Complex-twist (k != Gamma) counters. Same build/refresh semantics as
// the real per-basis counters above, but for the COMPLEX candidate tables
// (`Array2<Complex64>` det tables + the real `Re(complex AO)` J3 table).
/// Complex projection intervals that carried a persistent [`MeshFastStateC`]
/// across all sub-steps, analogous to `det_fast_intervals` and
/// `reduced_state_intervals`).
static GFMC_CPLX_STATE_INTERVALS: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_BUILD_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_REFRESH_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_BUILD_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_REFRESH_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_BUILD_PGTOPW: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_CAND_REFRESH_PGTOPW: AtomicU64 = AtomicU64::new(0);
/// Complex fast-mesh evaluations in which at least one determinant spin fell
/// back to the per-point `orb_single_complex` candidate path (pure PW,
/// pGTO+B-spline composite, or empty placeholder -- out of scope for the
/// candidate tables). No silent fallback.
static GFMC_CPLX_CAND_DET_SLOW: AtomicU64 = AtomicU64::new(0);
/// Complex fast-mesh evaluations with a J3 term whose orb class has no
/// candidate batch (falls back to per-point `j3_ao_re_single`).
static GFMC_CPLX_J3_SLOW_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// Complex projection intervals where `mesh_fast_state_complex` declined to
/// build (non-contiguous `j_matrix` gate) -- the whole interval runs on the
/// byte-exact reference path.
static GFMC_CPLX_STATE_GATE_MISS: AtomicU64 = AtomicU64::new(0);
// Complex continuum-kinetic derivative-cache counters. "Build" means a
// full derivative table was seeded (one event per spin block, counted under
// that orbital's basis class); "Refresh" = the moved electron's column was
// re-evaluated after a hop (moved spin's class).
static GFMC_CPLX_DERIV_BUILD_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DERIV_REFRESH_PGTO: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DERIV_BUILD_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DERIV_REFRESH_BSPLINE: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DERIV_BUILD_PGTOPW: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DERIV_REFRESH_PGTOPW: AtomicU64 = AtomicU64::new(0);
/// Complex e_L evaluations that ran the per-sub-step
/// `grads_and_laplacian_ln_det_complex` full rebuild while a fast state
/// existed (unsupported det basis class). No silent fallback.
static GFMC_CPLX_DERIV_DET_SLOW: AtomicU64 = AtomicU64::new(0);
/// Complex-path DTN grad/lap cache events (basis-agnostic: DTN is a pure
/// distance function).
static GFMC_CPLX_DTN_GRAD_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_DTN_GRAD_REFRESHES: AtomicU64 = AtomicU64::new(0);
/// Complex e_L evaluations with a DTN term that unexpectedly kept the full
/// per-sub-step DTN grad/lap rebuild while a fast state existed.
static GFMC_CPLX_DTN_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
/// Candidate-group consumptions of the batched DTN mesh ratio kernel
/// (`compute_dtn_ratio_mesh_cached`) through a [`MeshFastStateC`] -- the
/// complex twin of `dtn_cache_hits` (index 0).
static GFMC_CPLX_DTN_MESH_HITS: AtomicU64 = AtomicU64::new(0);
// Complex-path J1/J2 and J3 grad/lap cache counters.
static GFMC_CPLX_J12_GRAD_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_J12_GRAD_REFRESHES: AtomicU64 = AtomicU64::new(0);
/// Complex e_L evaluations with a J1/J2 term that unexpectedly kept the full
/// rebuild while a fast state existed.
static GFMC_CPLX_J12_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_J3_GRAD_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_CPLX_J3_GRAD_REFRESHES: AtomicU64 = AtomicU64::new(0);
/// Complex e_L evaluations with a J3 term that kept the full
/// `j3_grads_lap_complex` rebuild (no carried J3 ratio state, or an
/// unsupported J3 orb class).
static GFMC_CPLX_J3_GRAD_SLOW: AtomicU64 = AtomicU64::new(0);
/// Full builds of the projection-scoped DTN candidate-ratio table (the Rust
/// analogue of TurboRVB `tabpip`, restricted to the fixed six-point kinetic mesh).
static GFMC_DTN_TABPIP_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Accepted-hop low-rank advances of that DTN candidate-ratio table.
static GFMC_DTN_TABPIP_ADVANCES: AtomicU64 = AtomicU64::new(0);
/// Candidate groups served directly from the maintained DTN ratio table.
static GFMC_DTN_TABPIP_HITS: AtomicU64 = AtomicU64::new(0);
/// J1/J2 candidate log-ratio table events, split by candidate lattice so the
/// kinetic and ECP obligations can be proved independently.
static GFMC_J12_TABPIP_KIN_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_KIN_ADVANCES: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_KIN_HITS: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_KIN_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_ECP_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_ECP_ADVANCES: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_ECP_HITS: AtomicU64 = AtomicU64::new(0);
static GFMC_J12_TABPIP_ECP_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// TurboRVB-style determinant winv lifecycle/consumer events (real+complex).
static GFMC_WINV_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_WINV_ADVANCES: AtomicU64 = AtomicU64::new(0);
static GFMC_WINV_KIN_HITS: AtomicU64 = AtomicU64::new(0);
static GFMC_WINV_ECP_HITS: AtomicU64 = AtomicU64::new(0);
static GFMC_WINV_FALLBACKS: AtomicU64 = AtomicU64::new(0);
/// DTN tabpip lifecycle/consumer events for the arbitrary-position ECP mesh.
/// Kept separate from the fixed kinetic lattice so both obligations are
/// independently assertable in fixture tests.
static GFMC_DTN_TABPIP_ECP_BUILDS: AtomicU64 = AtomicU64::new(0);
static GFMC_DTN_TABPIP_ECP_ADVANCES: AtomicU64 = AtomicU64::new(0);
static GFMC_DTN_TABPIP_ECP_HITS: AtomicU64 = AtomicU64::new(0);
/// Projection-carried geminal inverse consumed by the force/stress Pulay path.
static GFMC_FORCE_GINV_HANDOFFS: AtomicU64 = AtomicU64::new(0);
/// Production force fold-ins that rebuilt the geminal inverse.
static GFMC_FORCE_GINV_FALLBACKS: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Route-PREDICATE counters. Same observability-only
// contract, enable flag and measurement window as the GFMC route counters
// above (`gfmc_route_counters_reset` zeroes both families), for the Rust
// route predicates outside the GFMC projection that choose between a fast and
// a same-value slower route:
// `coulomb_potential::ecp_wdot_ok` / `ecp_j3_wdot_ok` (ECP non-local w-dot
// contraction vs the batched orb-mesh path) and the MCMC chunk's
// `det_carry_ok` / `j3_carry_ok` (carry the streaming ratio state across steps
// vs rebuild it), and `param_grad_ecp::sm_ratio_ok` (SM increment vs fresh
// per-point rebuild of an ECP parameter-gradient mesh candidate). Each predicate has an "engaged" and a "declined" counter so
// a test can prove the fast route ran AND that the decline did not fire.
// Kept as a separate append-only array so the GFMC array keeps its indices.
// ---------------------------------------------------------------------------

/// ECP non-local evaluations (one event per spin table) that built the
/// electron-fixed determinant w-dot table (`ecp_wdot_ok` true).
static RS_ECP_WDOT_BUILDS: AtomicU64 = AtomicU64::new(0);
/// ECP non-local evaluations (one event per spin) with an SM inverse and a
/// determinant ratio state whose `ecp_wdot_ok` declined (batched orb-mesh path).
static RS_ECP_WDOT_DECLINES: AtomicU64 = AtomicU64::new(0);
/// J3 twin of `RS_ECP_WDOT_BUILDS` (`ecp_j3_wdot_ok` true).
static RS_ECP_J3_WDOT_BUILDS: AtomicU64 = AtomicU64::new(0);
/// J3 twin of `RS_ECP_WDOT_DECLINES` (J3 present with a ratio state, declined).
static RS_ECP_J3_WDOT_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Real MCMC chunk steps that carried the determinant ratio state (`det_carry_ok`).
static RS_MCMC_DET_CARRY_STEPS: AtomicU64 = AtomicU64::new(0);
/// Real MCMC chunk steps whose `det_carry_ok` declined (state rebuilt per step).
static RS_MCMC_DET_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Real MCMC chunk steps with a J3 term that carried the J3 ratio state (`j3_carry_ok`).
static RS_MCMC_J3_CARRY_STEPS: AtomicU64 = AtomicU64::new(0);
/// Real MCMC chunk steps with a J3 term whose `j3_carry_ok` declined.
static RS_MCMC_J3_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);
/// ECP non-local parameter-gradient mesh candidates (`param_grad_ecp`
/// `grad_ecp_mo` / `grad_ecp_lambda` O_k reduce) that took the rank-1 SM
/// increment off the base inverse (`sm_ratio_ok` true).
static RS_ECP_PARAM_SM_INCREMENTS: AtomicU64 = AtomicU64::new(0);
/// Those mesh candidates whose `sm_ratio_ok` declined (non-finite ratio,
/// |d| below the lower guard, or above the conditioning-aware upper guard
/// 1/(sqrt(eps) kappa_1(G_base))): the fresh per-point rebuild ran instead.
static RS_ECP_PARAM_SM_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Accepted LRDMC interval hops whose carried geminal inverse passed
/// the per-hop probe-residual screen (`determinant::interval_ginv_carry_ok`
/// true, the Sherman-Morrison carry is kept). One event per hop per walker.
static RS_INTERVAL_GINV_CARRIES: AtomicU64 = AtomicU64::new(0);
/// Those hops whose screen declined: the residual exceeded
/// `SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`, so `G^-1` was rebuilt from scratch
/// (and, on the real path, the winv tables with it). The refresh count.
static RS_INTERVAL_GINV_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Complex-twist ECP parameter-gradient meshes (`param_grad_twist`
/// `one_move_mesh_{re,cplx}`, one event per kernel call) served by the rank-1
/// determinant / incremental-Jastrow candidate mesh (`MeshFastStateC`), the same
/// mesh the native complex e_L uses.
static RS_ECP_PARAM_FAST_MESHES: AtomicU64 = AtomicU64::new(0);
/// Mesh CANDIDATES evaluated on the full-recompute reference mesh
/// (`complex_ecp::compute_ecp_nonlocal_mesh_complex(.., None)`): each one costs a full
/// Jastrow value and a full complex geminal determinant. Non-zero only on a
/// `mesh_fast_state_complex` gate miss or under the test-only
/// `JQMC_ECP_PARAM_FORCE_REFERENCE` hook.
static RS_ECP_PARAM_REF_MESH_CANDIDATES: AtomicU64 = AtomicU64::new(0);
/// ECP parameter-gradient J1 / J2 / J3 `j_matrix` mesh reduces (one event per
/// reduce, real and twist) that applied the exact one-electron O_k increment.
static RS_ECP_PARAM_OK_INCREMENTAL: AtomicU64 = AtomicU64::new(0);
/// Full J1 / J2 / J3 `j_matrix` O_k evaluations (all electrons) inside those mesh
/// reduces: one per candidate plus the base under the reference reduce, zero on the
/// incremental route.
static RS_ECP_PARAM_OK_FULL_EVALS: AtomicU64 = AtomicU64::new(0);
/// FAST-route ECP non-local (electron, non-local term) pairs whose radial factor
/// `|v_t(r)| >= FAST_ECP_NONLOCAL_RADIAL_CUT` (quadrature evaluated).
static RS_ECP_RADIAL_CUT_KEPT: AtomicU64 = AtomicU64::new(0);
/// FAST-route ECP non-local pairs with `|v_t(r)| < FAST_ECP_NONLOCAL_RADIAL_CUT`
/// whose quadrature was skipped (`coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT`).
static RS_ECP_RADIAL_CUT_DROPPED: AtomicU64 = AtomicU64::new(0);

/// Complex-twist FAST chunk ECP non-local evaluations (one event per spin) that
/// contracted the mesh determinant ratio in AO space (`ComplexOrb::mesh_wdot_ok`
/// true: `ComplexOrb::mesh_wdot_into` against the electron-fixed `q C` weights).
static RS_ECP_CPLX_WDOT_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Those evaluations whose `mesh_wdot_ok` declined (orbital-rank candidate table
/// `complex_candidate_batch`, or the per-point ratio for an unsupported class).
static RS_ECP_CPLX_WDOT_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Complex FAST ECP non-local states (`compute_ecp_nonlocal_withj_complex_fast`)
/// built on the walker's CARRIED determinant base (sweep orbital tables + running
/// SM inverse; `mesh_fast_state_complex_from`) -- the FAST chunk e_L route.
static RS_ECP_CPLX_SEED_CARRIED: AtomicU64 = AtomicU64::new(0);
/// Those states seeded FRESH (orbital re-evaluation + `G` rebuild + inverse):
/// the per-step dispatch / native driver, which carry no base.
static RS_ECP_CPLX_SEED_FRESH: AtomicU64 = AtomicU64::new(0);
/// Complex-twist FAST chunk local energies whose kinetic term was contracted from the
/// chunk-carried derivative tables ([`ComplexChunkKinCarry`], `kin_carry_ok` true).
static RS_CPLX_KIN_CARRY_STEPS: AtomicU64 = AtomicU64::new(0);
/// Those local energies whose `kin_carry_ok` declined (unsupported determinant class,
/// or the test-only `JQMC_RS_KIN_COMPLEX_FRESH` hook): the fresh per-e_L kinetic ran.
static RS_CPLX_KIN_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Seeds of the carried complex kinetic tables: one per walker chunk, plus every
/// probe-declined re-seed (index 26).
static RS_CPLX_KIN_CARRY_SEEDS: AtomicU64 = AtomicU64::new(0);
/// Accepted sweep moves replayed into the carried complex kinetic tables.
static RS_CPLX_KIN_CARRY_ADVANCES: AtomicU64 = AtomicU64::new(0);
/// Re-seeds forced by the carried-inverse probe residual of the tables' own kinetic inverse
/// (`kin_ginv_carry_ok` declined).
static RS_CPLX_KIN_GINV_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);
/// Complex-twist FAST chunk steps whose sweep starts from the CARRIED determinant base
/// (geminal, SM inverse and `DetRatioStateC` threaded across steps, `det_base_carry_ok`).
static RS_CPLX_DET_BASE_CARRY_STEPS: AtomicU64 = AtomicU64::new(0);
/// Those steps that rebuilt the base per step (`compute_geminal_complex` +
/// `inverse_complex` + batch `init_det_ratio_state_complex`; test-only
/// `JQMC_RS_DET_COMPLEX_FRESH_BASE` hook).
static RS_CPLX_DET_BASE_CARRY_DECLINES: AtomicU64 = AtomicU64::new(0);

/// Test-only reference hook for the ECP parameter-gradient Jastrow reduces: when
/// `JQMC_ECP_PARAM_FORCE_REFERENCE` is set, the twist kernels evaluate every candidate
/// on the full-recompute reference mesh and the J1 / J2 / J3 `j_matrix` reduces
/// recompute the full O_k per candidate (no one-electron O_k increments), so an A/B test can
/// compare the fast route against it on identical inputs. Read once per kernel call.
#[inline]
pub(crate) fn ecp_param_force_reference() -> bool {
    std::env::var_os("JQMC_ECP_PARAM_FORCE_REFERENCE").is_some()
}

/// Record one twist ECP parameter-gradient mesh: `None` = fast mesh, `Some(n)` =
/// `n` candidates on the reference mesh.
#[inline]
pub(crate) fn record_ecp_param_mesh(reference_candidates: Option<usize>) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        match reference_candidates {
            None => RS_ECP_PARAM_FAST_MESHES.fetch_add(1, Ordering::Relaxed),
            Some(n) => RS_ECP_PARAM_REF_MESH_CANDIDATES.fetch_add(n as u64, Ordering::Relaxed),
        };
    }
}

/// Record one J1 / J2 / J3 ECP O_k mesh reduce: `None` = incremental, `Some(n)` =
/// `n` full O_k evaluations.
#[inline]
pub(crate) fn record_ecp_param_ok(full_evals: Option<usize>) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        match full_evals {
            None => RS_ECP_PARAM_OK_INCREMENTAL.fetch_add(1, Ordering::Relaxed),
            Some(n) => RS_ECP_PARAM_OK_FULL_EVALS.fetch_add(n as u64, Ordering::Relaxed),
        };
    }
}

/// Snapshot of the route-predicate counters, index order (append-only):
///  0 ecp_wdot_builds               1 ecp_wdot_declines
///  2 ecp_j3_wdot_builds            3 ecp_j3_wdot_declines
///  4 mcmc_det_carry_steps          5 mcmc_det_carry_declines
///  6 mcmc_j3_carry_steps           7 mcmc_j3_carry_declines
///  8 ecp_param_sm_increments       9 ecp_param_sm_declines
/// 10 interval_ginv_carries         11 interval_ginv_carry_declines (refreshes)
/// 12 ecp_param_fast_meshes        13 ecp_param_ref_mesh_candidates
/// 14 ecp_param_ok_incremental     15 ecp_param_ok_full_evals
/// 16 ecp_cplx_wdot_builds         17 ecp_cplx_wdot_declines
/// 18 ecp_radial_cut_kept_pairs    19 ecp_radial_cut_dropped_pairs
/// 20 ecp_cplx_seed_carried        21 ecp_cplx_seed_fresh
/// 22 cplx_kin_carry_steps         23 cplx_kin_carry_declines
/// 24 cplx_kin_carry_seeds         25 cplx_kin_carry_advances
/// 26 cplx_kin_ginv_carry_declines (probe re-seeds)
/// 27 cplx_det_base_carry_steps    28 cplx_det_base_carry_declines
pub fn route_predicate_counters() -> [u64; 29] {
    [
        RS_ECP_WDOT_BUILDS.load(Ordering::Relaxed),
        RS_ECP_WDOT_DECLINES.load(Ordering::Relaxed),
        RS_ECP_J3_WDOT_BUILDS.load(Ordering::Relaxed),
        RS_ECP_J3_WDOT_DECLINES.load(Ordering::Relaxed),
        RS_MCMC_DET_CARRY_STEPS.load(Ordering::Relaxed),
        RS_MCMC_DET_CARRY_DECLINES.load(Ordering::Relaxed),
        RS_MCMC_J3_CARRY_STEPS.load(Ordering::Relaxed),
        RS_MCMC_J3_CARRY_DECLINES.load(Ordering::Relaxed),
        RS_ECP_PARAM_SM_INCREMENTS.load(Ordering::Relaxed),
        RS_ECP_PARAM_SM_DECLINES.load(Ordering::Relaxed),
        RS_INTERVAL_GINV_CARRIES.load(Ordering::Relaxed),
        RS_INTERVAL_GINV_CARRY_DECLINES.load(Ordering::Relaxed),
        RS_ECP_PARAM_FAST_MESHES.load(Ordering::Relaxed),
        RS_ECP_PARAM_REF_MESH_CANDIDATES.load(Ordering::Relaxed),
        RS_ECP_PARAM_OK_INCREMENTAL.load(Ordering::Relaxed),
        RS_ECP_PARAM_OK_FULL_EVALS.load(Ordering::Relaxed),
        RS_ECP_CPLX_WDOT_BUILDS.load(Ordering::Relaxed),
        RS_ECP_CPLX_WDOT_DECLINES.load(Ordering::Relaxed),
        RS_ECP_RADIAL_CUT_KEPT.load(Ordering::Relaxed),
        RS_ECP_RADIAL_CUT_DROPPED.load(Ordering::Relaxed),
        RS_ECP_CPLX_SEED_CARRIED.load(Ordering::Relaxed),
        RS_ECP_CPLX_SEED_FRESH.load(Ordering::Relaxed),
        RS_CPLX_KIN_CARRY_STEPS.load(Ordering::Relaxed),
        RS_CPLX_KIN_CARRY_DECLINES.load(Ordering::Relaxed),
        RS_CPLX_KIN_CARRY_SEEDS.load(Ordering::Relaxed),
        RS_CPLX_KIN_CARRY_ADVANCES.load(Ordering::Relaxed),
        RS_CPLX_KIN_GINV_CARRY_DECLINES.load(Ordering::Relaxed),
        RS_CPLX_DET_BASE_CARRY_STEPS.load(Ordering::Relaxed),
        RS_CPLX_DET_BASE_CARRY_DECLINES.load(Ordering::Relaxed),
    ]
}

#[inline]
fn bump_predicate(engaged: bool, on: &AtomicU64, off: &AtomicU64) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        if engaged {
            on.fetch_add(1, Ordering::Relaxed);
        } else {
            off.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Record one `ecp_wdot_ok` decision (called only where the w-dot table is
/// eligible at all: SM inverse and determinant ratio state both present).
#[inline]
pub(crate) fn record_ecp_wdot(engaged: bool) {
    bump_predicate(engaged, &RS_ECP_WDOT_BUILDS, &RS_ECP_WDOT_DECLINES);
}

/// Record one complex `mesh_wdot_ok` decision (FAST chunk state with the w-dot
/// route enabled; one decision per spin per evaluation).
#[inline]
pub(crate) fn record_ecp_cplx_wdot(engaged: bool) {
    bump_predicate(
        engaged,
        &RS_ECP_CPLX_WDOT_BUILDS,
        &RS_ECP_CPLX_WDOT_DECLINES,
    );
}

/// Record one complex FAST ECP state seed: `carried` = built on the caller's
/// determinant base, otherwise seeded fresh.
#[inline]
pub(crate) fn record_ecp_cplx_seed(carried: bool) {
    bump_predicate(carried, &RS_ECP_CPLX_SEED_CARRIED, &RS_ECP_CPLX_SEED_FRESH);
}

/// Record one complex FAST chunk e_L's `kin_carry_ok` decision (carried kinetic
/// tables vs the fresh per-e_L kinetic).
#[inline]
pub(crate) fn record_cplx_kin_carry(engaged: bool) {
    bump_predicate(
        engaged,
        &RS_CPLX_KIN_CARRY_STEPS,
        &RS_CPLX_KIN_CARRY_DECLINES,
    );
}

/// Record one complex FAST chunk step's `det_base_carry_ok` decision.
#[inline]
pub(crate) fn record_cplx_det_base_carry(engaged: bool) {
    bump_predicate(
        engaged,
        &RS_CPLX_DET_BASE_CARRY_STEPS,
        &RS_CPLX_DET_BASE_CARRY_DECLINES,
    );
}

/// Record one `ecp_j3_wdot_ok` decision (J3 term and J3 ratio state present).
#[inline]
pub(crate) fn record_ecp_j3_wdot(engaged: bool) {
    bump_predicate(engaged, &RS_ECP_J3_WDOT_BUILDS, &RS_ECP_J3_WDOT_DECLINES);
}

/// Record one FAST ECP non-local evaluation's radial-cut outcome: `kept` pairs
/// evaluated, `dropped` pairs skipped (`|v_t| < FAST_ECP_NONLOCAL_RADIAL_CUT`).
#[inline]
pub(crate) fn record_ecp_radial_cut(kept: u64, dropped: u64) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        RS_ECP_RADIAL_CUT_KEPT.fetch_add(kept, Ordering::Relaxed);
        RS_ECP_RADIAL_CUT_DROPPED.fetch_add(dropped, Ordering::Relaxed);
    }
}

/// Record one real MCMC chunk step's `det_carry_ok` decision.
#[inline]
pub(crate) fn record_mcmc_det_carry(engaged: bool) {
    bump_predicate(
        engaged,
        &RS_MCMC_DET_CARRY_STEPS,
        &RS_MCMC_DET_CARRY_DECLINES,
    );
}

/// Record one real MCMC chunk step's `j3_carry_ok` decision (J3 present only).
#[inline]
pub(crate) fn record_mcmc_j3_carry(engaged: bool) {
    bump_predicate(engaged, &RS_MCMC_J3_CARRY_STEPS, &RS_MCMC_J3_CARRY_DECLINES);
}

/// Record the `sm_ratio_ok` decisions of one ECP non-local parameter-gradient
/// mesh loop (`param_grad_ecp`): `engaged` candidates took the SM increment,
/// `declined` took the fresh per-point rebuild. The loop tallies plain local
/// integers and calls this once afterwards, so the per-candidate cost is a
/// register increment and the enable flag is read once per loop.
#[inline]
pub(crate) fn record_ecp_param_sm_ratio(engaged: u64, declined: u64) {
    if GFMC_ROUTE_COUNTERS_ON.load(Ordering::Relaxed) {
        RS_ECP_PARAM_SM_INCREMENTS.fetch_add(engaged, Ordering::Relaxed);
        RS_ECP_PARAM_SM_DECLINES.fetch_add(declined, Ordering::Relaxed);
    }
}

/// Record one per-hop interval-inverse screen decision (`engaged` =
/// the carried `G^-1` was kept, otherwise it was refreshed).
#[inline]
pub(crate) fn record_interval_ginv_carry(engaged: bool) {
    bump_predicate(
        engaged,
        &RS_INTERVAL_GINV_CARRIES,
        &RS_INTERVAL_GINV_CARRY_DECLINES,
    );
}

/// Snapshot of the GFMC route counters, index order (append-only):
///  0 dtn_cache_hits
///  1 det_fast_intervals            (molecular fast-table intervals)
///  2 reduced_state_intervals      (non-molecular intervals)
///  3 cand_build_molecular          4 cand_refresh_molecular
///  5 cand_build_periodic_pgto      6 cand_refresh_periodic_pgto
///  7 cand_build_bspline            8 cand_refresh_bspline
///  9 cand_build_pgto_pw           10 cand_refresh_pgto_pw
/// 11 cand_det_slow_path           12 j3_slow_fallbacks
/// 13 det_grad_generic_builds
/// 14 cplx_state_intervals         (complex carried-state intervals)
/// 15 cplx_cand_build_pgto         16 cplx_cand_refresh_pgto
/// 17 cplx_cand_build_bspline      18 cplx_cand_refresh_bspline
/// 19 cplx_cand_build_pgto_pw      20 cplx_cand_refresh_pgto_pw
/// 21 cplx_cand_det_slow_path      22 cplx_j3_slow_fallbacks
/// 23 cplx_state_gate_miss
/// 24 cplx_deriv_build_pgto        25 cplx_deriv_refresh_pgto
/// 26 cplx_deriv_build_bspline     27 cplx_deriv_refresh_bspline
/// 28 cplx_deriv_build_pgto_pw     29 cplx_deriv_refresh_pgto_pw
/// 30 cplx_deriv_det_slow_path     (complex det derivative cache)
/// 31 cplx_dtn_grad_builds         32 cplx_dtn_grad_refreshes
/// 33 cplx_dtn_grad_slow_path      34 cplx_dtn_mesh_hits
/// 35 cplx_j12_grad_builds         36 cplx_j12_grad_refreshes
/// 37 cplx_j12_grad_slow_path      38 cplx_j3_grad_builds
/// 39 cplx_j3_grad_refreshes       40 cplx_j3_grad_slow_path
/// 41 det_grad_slow                42 j12_grad_slow
/// 43 j3_grad_slow                 44 dtn_grad_slow
/// 45 dtn_tabpip_builds            46 dtn_tabpip_advances
/// 47 dtn_tabpip_hits
/// 48 j12_tabpip_kin_builds         49 j12_tabpip_kin_advances
/// 50 j12_tabpip_kin_hits           51 j12_tabpip_kin_fallbacks
/// 52 j12_tabpip_ecp_builds         53 j12_tabpip_ecp_advances
/// 54 j12_tabpip_ecp_hits           55 j12_tabpip_ecp_fallbacks
/// 56 winv_builds                    57 winv_advances
/// 58 winv_kin_hits                  59 winv_ecp_hits
/// 60 winv_fallbacks
/// 61 dtn_tabpip_ecp_builds          62 dtn_tabpip_ecp_advances
/// 63 dtn_tabpip_ecp_hits
/// 64 force_ginv_handoffs             65 force_ginv_fallbacks
/// 66 cand_mo_tiled                   67 cand_mo_dense
/// 68 ecp_live_groups                 69 ecp_zero_groups
/// 70 ecp_cut_kept_pairs              71 ecp_cut_dropped_pairs
/// 72 dtn_far_rows
/// 73 cplx_ecpf_grouped_jas_ratios
pub fn gfmc_route_counters() -> [u64; 74] {
    [
        GFMC_DTN_CACHE_HITS.load(Ordering::Relaxed),
        GFMC_DET_FAST_INTERVALS.load(Ordering::Relaxed),
        GFMC_REDUCED_STATE_INTERVALS.load(Ordering::Relaxed),
        GFMC_CAND_BUILD_MOL.load(Ordering::Relaxed),
        GFMC_CAND_REFRESH_MOL.load(Ordering::Relaxed),
        GFMC_CAND_BUILD_PGTO.load(Ordering::Relaxed),
        GFMC_CAND_REFRESH_PGTO.load(Ordering::Relaxed),
        GFMC_CAND_BUILD_BSPLINE.load(Ordering::Relaxed),
        GFMC_CAND_REFRESH_BSPLINE.load(Ordering::Relaxed),
        GFMC_CAND_BUILD_PGTOPW.load(Ordering::Relaxed),
        GFMC_CAND_REFRESH_PGTOPW.load(Ordering::Relaxed),
        GFMC_CAND_DET_SLOW.load(Ordering::Relaxed),
        GFMC_J3_SLOW_FALLBACKS.load(Ordering::Relaxed),
        GFMC_DET_GRAD_GENERIC_BUILDS.load(Ordering::Relaxed),
        GFMC_CPLX_STATE_INTERVALS.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_BUILD_PGTO.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_REFRESH_PGTO.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_BUILD_BSPLINE.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_REFRESH_BSPLINE.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_BUILD_PGTOPW.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_REFRESH_PGTOPW.load(Ordering::Relaxed),
        GFMC_CPLX_CAND_DET_SLOW.load(Ordering::Relaxed),
        GFMC_CPLX_J3_SLOW_FALLBACKS.load(Ordering::Relaxed),
        GFMC_CPLX_STATE_GATE_MISS.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_BUILD_PGTO.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_REFRESH_PGTO.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_BUILD_BSPLINE.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_REFRESH_BSPLINE.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_BUILD_PGTOPW.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_REFRESH_PGTOPW.load(Ordering::Relaxed),
        GFMC_CPLX_DERIV_DET_SLOW.load(Ordering::Relaxed),
        GFMC_CPLX_DTN_GRAD_BUILDS.load(Ordering::Relaxed),
        GFMC_CPLX_DTN_GRAD_REFRESHES.load(Ordering::Relaxed),
        GFMC_CPLX_DTN_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_CPLX_DTN_MESH_HITS.load(Ordering::Relaxed),
        GFMC_CPLX_J12_GRAD_BUILDS.load(Ordering::Relaxed),
        GFMC_CPLX_J12_GRAD_REFRESHES.load(Ordering::Relaxed),
        GFMC_CPLX_J12_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_CPLX_J3_GRAD_BUILDS.load(Ordering::Relaxed),
        GFMC_CPLX_J3_GRAD_REFRESHES.load(Ordering::Relaxed),
        GFMC_CPLX_J3_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_DET_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_J12_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_J3_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_DTN_GRAD_SLOW.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_BUILDS.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_ADVANCES.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_HITS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_KIN_BUILDS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_KIN_ADVANCES.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_KIN_HITS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_KIN_FALLBACKS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_ECP_BUILDS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_ECP_ADVANCES.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_ECP_HITS.load(Ordering::Relaxed),
        GFMC_J12_TABPIP_ECP_FALLBACKS.load(Ordering::Relaxed),
        GFMC_WINV_BUILDS.load(Ordering::Relaxed),
        GFMC_WINV_ADVANCES.load(Ordering::Relaxed),
        GFMC_WINV_KIN_HITS.load(Ordering::Relaxed),
        GFMC_WINV_ECP_HITS.load(Ordering::Relaxed),
        GFMC_WINV_FALLBACKS.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_ECP_BUILDS.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_ECP_ADVANCES.load(Ordering::Relaxed),
        GFMC_DTN_TABPIP_ECP_HITS.load(Ordering::Relaxed),
        GFMC_FORCE_GINV_HANDOFFS.load(Ordering::Relaxed),
        GFMC_FORCE_GINV_FALLBACKS.load(Ordering::Relaxed),
        GFMC_CAND_MO_TILED.load(Ordering::Relaxed),
        GFMC_CAND_MO_DENSE.load(Ordering::Relaxed),
        GFMC_ECP_LIVE_GROUPS.load(Ordering::Relaxed),
        GFMC_ECP_ZERO_GROUPS.load(Ordering::Relaxed),
        GFMC_ECP_CUT_KEPT_PAIRS.load(Ordering::Relaxed),
        GFMC_ECP_CUT_DROPPED_PAIRS.load(Ordering::Relaxed),
        GFMC_DTN_FAR_ROWS.load(Ordering::Relaxed),
        GFMC_CPLX_ECPF_GROUPED_JAS_RATIOS.load(Ordering::Relaxed),
    ]
}

/// Zero the GFMC route counters and the route-predicate counters
/// (per-fixture measurement windows); also enables counting.
pub fn gfmc_route_counters_reset() {
    GFMC_ROUTE_COUNTERS_ON.store(true, Ordering::Relaxed);
    for c in [
        &GFMC_DTN_CACHE_HITS,
        &GFMC_DET_FAST_INTERVALS,
        &GFMC_REDUCED_STATE_INTERVALS,
        &GFMC_CAND_BUILD_MOL,
        &GFMC_CAND_REFRESH_MOL,
        &GFMC_CAND_BUILD_PGTO,
        &GFMC_CAND_REFRESH_PGTO,
        &GFMC_CAND_BUILD_BSPLINE,
        &GFMC_CAND_REFRESH_BSPLINE,
        &GFMC_CAND_BUILD_PGTOPW,
        &GFMC_CAND_REFRESH_PGTOPW,
        &GFMC_CAND_DET_SLOW,
        &GFMC_J3_SLOW_FALLBACKS,
        &GFMC_DET_GRAD_GENERIC_BUILDS,
        &GFMC_CPLX_STATE_INTERVALS,
        &GFMC_CPLX_CAND_BUILD_PGTO,
        &GFMC_CPLX_CAND_REFRESH_PGTO,
        &GFMC_CPLX_CAND_BUILD_BSPLINE,
        &GFMC_CPLX_CAND_REFRESH_BSPLINE,
        &GFMC_CPLX_CAND_BUILD_PGTOPW,
        &GFMC_CPLX_CAND_REFRESH_PGTOPW,
        &GFMC_CPLX_CAND_DET_SLOW,
        &GFMC_CPLX_J3_SLOW_FALLBACKS,
        &GFMC_CPLX_STATE_GATE_MISS,
        &GFMC_CPLX_DERIV_BUILD_PGTO,
        &GFMC_CPLX_DERIV_REFRESH_PGTO,
        &GFMC_CPLX_DERIV_BUILD_BSPLINE,
        &GFMC_CPLX_DERIV_REFRESH_BSPLINE,
        &GFMC_CPLX_DERIV_BUILD_PGTOPW,
        &GFMC_CPLX_DERIV_REFRESH_PGTOPW,
        &GFMC_CPLX_DERIV_DET_SLOW,
        &GFMC_CPLX_DTN_GRAD_BUILDS,
        &GFMC_CPLX_DTN_GRAD_REFRESHES,
        &GFMC_CPLX_DTN_GRAD_SLOW,
        &GFMC_CPLX_DTN_MESH_HITS,
        &GFMC_DET_GRAD_SLOW,
        &GFMC_J12_GRAD_SLOW,
        &GFMC_J3_GRAD_SLOW,
        &GFMC_DTN_GRAD_SLOW,
        &GFMC_CPLX_J12_GRAD_BUILDS,
        &GFMC_CPLX_J12_GRAD_REFRESHES,
        &GFMC_CPLX_J12_GRAD_SLOW,
        &GFMC_CPLX_J3_GRAD_BUILDS,
        &GFMC_CPLX_J3_GRAD_REFRESHES,
        &GFMC_CPLX_J3_GRAD_SLOW,
        &GFMC_DTN_TABPIP_BUILDS,
        &GFMC_DTN_TABPIP_ADVANCES,
        &GFMC_DTN_TABPIP_HITS,
        &GFMC_J12_TABPIP_KIN_BUILDS,
        &GFMC_J12_TABPIP_KIN_ADVANCES,
        &GFMC_J12_TABPIP_KIN_HITS,
        &GFMC_J12_TABPIP_KIN_FALLBACKS,
        &GFMC_J12_TABPIP_ECP_BUILDS,
        &GFMC_J12_TABPIP_ECP_ADVANCES,
        &GFMC_J12_TABPIP_ECP_HITS,
        &GFMC_J12_TABPIP_ECP_FALLBACKS,
        &GFMC_WINV_BUILDS,
        &GFMC_WINV_ADVANCES,
        &GFMC_WINV_KIN_HITS,
        &GFMC_WINV_ECP_HITS,
        &GFMC_WINV_FALLBACKS,
        &GFMC_DTN_TABPIP_ECP_BUILDS,
        &GFMC_DTN_TABPIP_ECP_ADVANCES,
        &GFMC_DTN_TABPIP_ECP_HITS,
        &GFMC_FORCE_GINV_HANDOFFS,
        &GFMC_FORCE_GINV_FALLBACKS,
        &GFMC_CAND_MO_TILED,
        &GFMC_CAND_MO_DENSE,
        &GFMC_ECP_LIVE_GROUPS,
        &GFMC_ECP_ZERO_GROUPS,
        &GFMC_ECP_CUT_KEPT_PAIRS,
        &GFMC_ECP_CUT_DROPPED_PAIRS,
        &GFMC_DTN_FAR_ROWS,
        &GFMC_CPLX_ECPF_GROUPED_JAS_RATIOS,
        &RS_ECP_WDOT_BUILDS,
        &RS_ECP_WDOT_DECLINES,
        &RS_ECP_J3_WDOT_BUILDS,
        &RS_ECP_J3_WDOT_DECLINES,
        &RS_MCMC_DET_CARRY_STEPS,
        &RS_MCMC_DET_CARRY_DECLINES,
        &RS_MCMC_J3_CARRY_STEPS,
        &RS_MCMC_J3_CARRY_DECLINES,
        &RS_ECP_PARAM_SM_INCREMENTS,
        &RS_ECP_PARAM_SM_DECLINES,
        &RS_INTERVAL_GINV_CARRIES,
        &RS_INTERVAL_GINV_CARRY_DECLINES,
        &RS_ECP_PARAM_FAST_MESHES,
        &RS_ECP_PARAM_REF_MESH_CANDIDATES,
        &RS_ECP_PARAM_OK_INCREMENTAL,
        &RS_ECP_PARAM_OK_FULL_EVALS,
        &RS_ECP_CPLX_WDOT_BUILDS,
        &RS_ECP_CPLX_WDOT_DECLINES,
        &RS_ECP_RADIAL_CUT_KEPT,
        &RS_ECP_RADIAL_CUT_DROPPED,
        &RS_ECP_CPLX_SEED_CARRIED,
        &RS_ECP_CPLX_SEED_FRESH,
        &RS_CPLX_KIN_CARRY_STEPS,
        &RS_CPLX_KIN_CARRY_DECLINES,
        &RS_CPLX_KIN_CARRY_SEEDS,
        &RS_CPLX_KIN_CARRY_ADVANCES,
        &RS_CPLX_KIN_GINV_CARRY_DECLINES,
        &RS_CPLX_DET_BASE_CARRY_STEPS,
        &RS_CPLX_DET_BASE_CARRY_DECLINES,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Record the projection-to-force state handoff once per produced force block.
/// This is observability only; it deliberately lives at the fold-in boundary
/// so force and stress sharing is measured independently of numeric kernels.
pub(crate) fn record_force_state_handoff(has_ginv: bool) {
    if has_ginv {
        bump_route_counter(&GFMC_FORCE_GINV_HANDOFFS);
    } else {
        bump_route_counter(&GFMC_FORCE_GINV_FALLBACKS);
    }
}

/// Accumulated GFMC weights `G_L` in binary log space (`_compute_G_L_split`). For `w_l` of
/// shape `(A, x)` returns `(mantissa, exponent)`, each `(A-k, x)`, with
/// `G_L[j, c] = Prod_{i=0}^{k-1} w_l[j+i, c] = mantissa[j, c] * 2^exponent[j, c]`. Every factor
/// is split by `frexp` and the running mantissa is renormalized after each multiplication, so
/// the magnitude lives in the exact integer exponent and nothing overflows for any window length
/// `k = num_gfmc_collect_steps`. Power-of-two scaling is exact, so the mantissa rounding is that
/// of the sequential product.
pub fn compute_g_l_split(w_l: &ArrayView2<f64>, k: usize) -> (Array2<f64>, Array2<i64>) {
    let (a, x) = (w_l.nrows(), w_l.ncols());
    let rows = a.saturating_sub(k);
    let mut mantissa = Array2::<f64>::ones((rows, x));
    let mut exponent = Array2::<i64>::zeros((rows, x));
    for j in 0..rows {
        for i in 0..k {
            for c in 0..x {
                let (factor_mantissa, factor_exponent) = libm::frexp(w_l[[j + i, c]]);
                let (m, renormalization) = libm::frexp(mantissa[[j, c]] * factor_mantissa);
                mantissa[[j, c]] = m;
                exponent[[j, c]] += i64::from(factor_exponent) + i64::from(renormalization);
            }
        }
    }
    (mantissa, exponent)
}

/// `mantissa * 2^exponent` (C `ldexp`); exponents beyond the i32 range give the same 0/inf.
fn ldexp_i64(mantissa: f64, exponent: i64) -> f64 {
    libm::scalbn(
        mantissa,
        exponent.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
    )
}

/// Unscaled accumulated GFMC weights `G_L` (`_compute_G_L`). For `w_l` of shape `(A, x)`
/// (A projection steps, x walkers), returns `(A-k, x)` with
/// `G_L[j, c] = Prod_{i=0}^{k-1} w_l[j+i, c]` -- the product of `k =
/// num_gfmc_collect_steps` consecutive branching weights (the discretized
/// Green's-function projector weight over the collection window). This raw product
/// leaves the f64 range once `k * tau * |E|` exceeds about 709; estimators use
/// [`compute_g_l_retained`].
pub fn compute_g_l(w_l: &ArrayView2<f64>, k: usize) -> Array2<f64> {
    let (mantissa, exponent) = compute_g_l_split(w_l, k);
    let mut out = Array2::<f64>::zeros(mantissa.raw_dim());
    for ((o, &m), &e) in out.iter_mut().zip(mantissa.iter()).zip(exponent.iter()) {
        *o = ldexp_i64(m, e);
    }
    out
}

/// Accumulated weights of the rows after `num_warmup_steps` on one common scale
/// (`_compute_retained_G_L`): the retained rows of [`compute_g_l_split`] are multiplied by the
/// single exact factor `2^-M`, `M` = the largest exponent among the nonzero finite retained
/// mantissas (0 if there is none). The factor cancels in every weighted mean and jackknife
/// ratio, so the estimates are those of the unscaled products while every weight stays finite.
pub fn compute_g_l_retained(
    w_l: &ArrayView2<f64>,
    k: usize,
    num_warmup_steps: usize,
) -> Array2<f64> {
    let (mantissa, exponent) = compute_g_l_split(w_l, k);
    let start = num_warmup_steps.min(mantissa.nrows());
    let rows = mantissa.nrows() - start;
    let x = mantissa.ncols();
    let mut shift: Option<i64> = None;
    for j in start..mantissa.nrows() {
        for c in 0..x {
            let m = mantissa[[j, c]];
            if m != 0.0 && m.is_finite() {
                let e = exponent[[j, c]];
                shift = Some(shift.map_or(e, |s| s.max(e)));
            }
        }
    }
    let shift = shift.unwrap_or(0);
    let mut out = Array2::<f64>::zeros((rows, x));
    for j in 0..rows {
        for c in 0..x {
            out[[j, c]] = ldexp_i64(mantissa[[start + j, c]], exponent[[start + j, c]] - shift);
        }
    }
    out
}

/// Mixed single-walker GFMC estimator `Sum_j e[j+k] G_L[j] / Sum_j G_L[j]` with the scaled
/// weights of [`compute_g_l_retained`] (`G_L[j]` = product of `w[j..j+k]`, paired with
/// `e[j+k]`). NaN when no window is complete or the weights sum to zero.
pub(crate) fn mixed_window_energy(e: &[f64], w: &[f64], k: usize) -> f64 {
    let n = e.len().min(w.len());
    let w_view = ArrayView2::from_shape((n, 1), &w[..n]).expect("contiguous (n, 1) weight view");
    let g_l = compute_g_l_retained(&w_view, k, 0);
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for j in 0..g_l.nrows() {
        num += e[j + k] * g_l[[j, 0]];
        den += g_l[[j, 0]];
    }
    if den != 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

/// Systematic stochastic reconfiguration (branching). Given walker `weights`
/// (length N) and one uniform draw `zeta` in `[0, 1)`, returns the N resampled
/// SOURCE indices. Single-process equivalent of the jqmc reconfiguration:
/// `cdf = cumsum(w / Sumw)`, `z_i = (i + zeta) / N`, `idx_i = searchsorted(cdf,
/// z_i, side='left')` (the first index with `cdf[idx] >= z_i`). Low-variance
/// (one random number for the whole population) and survivor-preserving.
pub fn reconfigure_indices(weights: &[f64], zeta: f64) -> Vec<usize> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let sum: f64 = weights.iter().sum();
    let mut cdf = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for i in 0..n {
        acc += weights[i] / sum;
        cdf[i] = acc;
    }
    let nf = n as f64;
    (0..n)
        .map(|i| {
            let z = (i as f64 + zeta) / nf;
            // searchsorted(..., side='left'): first idx with cdf[idx] >= z.
            let idx = cdf.partition_point(|&c| c < z);
            idx.min(n - 1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr2;

    #[test]
    fn g_l_window_product() {
        // w_l (4,1); k=2 -> rows = [w0*w1, w1*w2, w2*w3].
        let w = arr2(&[[2.0], [3.0], [4.0], [5.0]]);
        let g = compute_g_l(&w.view(), 2);
        assert_eq!(g.shape(), &[2, 1]); // A - k = 4 - 2 = 2
        assert!((g[[0, 0]] - 6.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
        assert!((g[[1, 0]] - 12.0).abs() < crate::setting_consts::ATOL_PYTHON_FP64_LOOSE);
    }

    #[test]
    fn reconfigure_uniform_is_identity_like() {
        // Equal weights + zeta=0.5 -> z_i=(i+0.5)/N lands one per cell -> idx i.
        let w = vec![1.0; 5];
        let idx = reconfigure_indices(&w, 0.5);
        assert_eq!(idx, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn reconfigure_concentrates_on_heavy_walker() {
        // Walker 1 carries ~all the weight -> every survivor maps to index 1.
        let w = vec![1e-12, 1.0, 1e-12];
        let idx = reconfigure_indices(&w, 0.3);
        assert!(idx.iter().all(|&i| i == 1), "{idx:?}");
    }

    // NaN-propagating real min/max, analogous to `mcmc::accept_clamp`.
    // Bare f64::min/max are IEEE minNum/maxNum and absorb the NaN.
    #[test]
    fn nan_min_max_propagate_nan_in_either_operand() {
        let n = f64::NAN;
        assert!(
            nan_min(n, 0.0).is_nan(),
            "NaN in the FIRST operand must propagate"
        );
        assert!(
            nan_min(0.0, n).is_nan(),
            "NaN in the SECOND operand must propagate"
        );
        assert!(nan_max(n, 0.0).is_nan());
        assert!(nan_max(0.0, n).is_nan());
        assert!(nan_min(n, n).is_nan());
        // The defect being fixed: the bare IEEE minNum/maxNum absorb it.
        assert_eq!(n.min(0.0), 0.0);
        assert_eq!(n.max(0.0), 0.0);
    }

    #[test]
    fn nan_min_max_identity_on_finite_inputs() {
        for (a, b) in [
            (-1.5, 0.0),
            (0.0, -1.5),
            (2.0, 3.0),
            (3.0, 2.0),
            (0.0, 0.0),
            (-0.0, 0.0),
        ] {
            assert_eq!(nan_min(a, b), a.min(b), "nan_min({a},{b})");
            assert_eq!(nan_max(a, b), a.max(b), "nan_max({a},{b})");
        }
        assert_eq!(nan_min(f64::INFINITY, 1.0), 1.0);
        assert_eq!(nan_max(f64::NEG_INFINITY, 1.0), 1.0);
    }

    // --- COMPLEX helpers: JAX's complex min/max/ge are lexicographic SELECTS, so they do
    // NOT propagate NaN -- the second operand wins. Values MEASURED against jnp (see the
    // c_lt doc comment): jnp.minimum(nan+0j,0)==0, jnp.maximum(0,nan+0j)==nan,
    // (nan+0j)>=0 is False, (0+nanj)>=0 is False.
    #[test]
    fn complex_helpers_match_jax_lexicographic_nan_semantics() {
        let z = Complex64::new(0.0, 0.0);
        let nan_re = Complex64::new(f64::NAN, 0.0);
        let nan_im = Complex64::new(0.0, f64::NAN);
        assert_eq!(c_min0(nan_re), z);
        assert_eq!(c_max0(nan_re), z);
        assert_eq!(c_min0(nan_im), z);
        assert_eq!(c_max0(nan_im), z);
        // jnp.maximum(a, b) picks a ONLY when strictly greater -> NaN yields b.
        let b = Complex64::new(-1.0, 0.0);
        assert_eq!(c_max(nan_re, b), b);
        assert!(c_max(b, nan_re).re.is_nan());
        assert!(!c_ge0(nan_re));
        assert!(!c_ge0(nan_im));
    }

    #[test]
    fn complex_helpers_identity_on_finite_inputs() {
        let z = Complex64::new(0.0, 0.0);
        assert_eq!(c_min0(Complex64::new(-1.0, 5.0)), Complex64::new(-1.0, 5.0));
        assert_eq!(c_min0(Complex64::new(1.0, -5.0)), z);
        assert_eq!(c_max0(Complex64::new(0.0, 1.0)), Complex64::new(0.0, 1.0));
        assert_eq!(c_max0(Complex64::new(0.0, -1.0)), z);
        assert_eq!(
            c_max(Complex64::new(2.0, 0.0), Complex64::new(1.0, 9.0)),
            Complex64::new(2.0, 0.0)
        );
        assert_eq!(
            c_max(Complex64::new(1.0, 9.0), Complex64::new(2.0, 0.0)),
            Complex64::new(2.0, 0.0)
        );
        // Ties on the real part fall through to the imaginary part (lexicographic).
        assert_eq!(
            c_max(Complex64::new(1.0, 2.0), Complex64::new(1.0, 3.0)),
            Complex64::new(1.0, 3.0)
        );
        assert!(c_ge0(Complex64::new(0.0, 1.0)));
        assert!(c_ge0(z));
        assert!(!c_ge0(Complex64::new(0.0, -1.0)));
        assert!(c_ge0(Complex64::new(1.0, -5.0)));
        assert!(!c_ge0(Complex64::new(-1.0, 5.0)));
    }
}

// ============================================================================
// LRDMC discretized kinetic kernel (the core new GFMC kernel).
// 1:1 wavefunction.py::compute_discretized_kinetic_energy: for each electron move
// it +/-alat along the 3 (rotated) axes -> 6*(N_up+N_dn) neighbor configs; the hopping
// amplitude is elements[c] = -1/(2 alat^2)*wf_ratio, wf_ratio = exp(DeltaJ)*det_c/det_base.
// Correctness-first (full recompute per neighbor). `rt` is the (rotated) axis matrix RT.
// ============================================================================

use crate::complex_det::{
    compute_geminal_complex, det_phase_lnabs, grads_and_laplacian_ln_det_complex,
    j3_grads_lap_complex, ComplexGeminal, ComplexOrb,
};
use crate::data::JastrowData;
use crate::determinant::{compute_geminal_all_elements, det_sign_lnabs, GeminalData, OrbData};
use numpy::Complex64;

/// NaN-PROPAGATING minimum, mirroring `jnp.minimum` on a REAL array (XLA `min_hlo` is
/// IEEE-754-2019 minimum: NaN in EITHER operand yields NaN).  Rust's bare `f64::min` is
/// IEEE minNum, which ABSORBS the NaN (`f64::NAN.min(0.0) == 0.0`) -- so a degenerate
/// configuration (determinant ratio NaN) that makes XLA surface a NaN walker instead
/// silently contributed ZERO to the Rust FN sum / escape time and the chain marched on
/// down a finite but different trajectory. This is the GFMC analogue of
/// [`crate::mcmc::accept_clamp`]. Every FN/SP split, ZV composition, and escape-time clamp
/// routes through these two.
#[inline]
pub fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// NaN-PROPAGATING maximum, mirroring `jnp.maximum` on a REAL array.  See [`nan_min`].
#[inline]
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// JAX COMPLEX ordering is lexicographic on `(re, im)`. The GFMC FN/SP clamps, sign-flip,
/// and ZV max/where all run on complex arrays with NO explicit Re (jqmc fixed-phase), so
/// these mirror `jnp.minimum(x,0)` / `jnp.maximum(x,0)` / `jnp.maximum(a,b)` / `x>=0` exactly.
///
/// Unlike the real case above, JAX's complex
/// min/max/ge do NOT propagate NaN.  `_minmax_complex_lowering` (jax/_src/lax/lax.py) is a
/// pure `select(pick_x, x, y)` on lexicographic comparisons, and every comparison against
/// NaN is FALSE, so the SECOND operand wins: MEASURED `jnp.minimum(nan+0j, 0) == 0` but
/// `jnp.minimum(0, nan+0j) == nan`, and `(nan+0j) >= 0` is `False`.  The helpers below
/// therefore reproduce that ASYMMETRY exactly rather than propagating -- matching XLA is
/// the contract, not NaN propagation per se.
#[inline]
fn c_lt(a: Complex64, b: Complex64) -> bool {
    a.re < b.re || (a.re == b.re && a.im < b.im)
}
/// `jnp.minimum(x, 0)`: picks `x` only when `x < 0` lexicographically -> NaN yields `0`.
#[inline]
fn c_min0(x: Complex64) -> Complex64 {
    if c_lt(x, Complex64::new(0.0, 0.0)) {
        x
    } else {
        Complex64::new(0.0, 0.0)
    }
}
/// `jnp.maximum(x, 0)`: picks `x` only when `x > 0` lexicographically -> NaN yields `0`.
#[inline]
fn c_max0(x: Complex64) -> Complex64 {
    if c_lt(Complex64::new(0.0, 0.0), x) {
        x
    } else {
        Complex64::new(0.0, 0.0)
    }
}
/// `jnp.maximum(a, b)`: picks the FIRST operand only when it is STRICTLY greater
/// lexicographically, else the second -- so a NaN in either operand yields `b`, exactly as
/// `select(pick_x, x, y)` does.  (`if c_lt(a, b) { b } else { a }` agrees on
/// every finite input but returns `a` on NaN, where XLA returns `b`.)
#[inline]
fn c_max(a: Complex64, b: Complex64) -> Complex64 {
    if c_lt(b, a) {
        a
    } else {
        b
    }
}
/// `x >= 0` on complex: lexicographic `ge`, all-false against NaN (so NaN -> `false`).
/// `!c_lt(x, 0)` would return `true` for NaN, which is the opposite of XLA.
#[inline]
fn c_ge0(x: Complex64) -> bool {
    if x.re == 0.0 {
        x.im >= 0.0
    } else {
        x.re > 0.0
    }
}

/// Euler-angle rotation `R = Rz(gamma)*Ry(beta)*Rx(alpha)` (jqmc `_generate_rotation_matrix_t`,
/// jqmc_gfmc.py). Returns **`RT`** (the `RT` fed to the kinetic/ECP mesh builders).
fn rotation_matrix_t(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
    let (ca, sa) = (alpha.cos(), alpha.sin());
    let (cb, sb) = (beta.cos(), beta.sin());
    let (cg, sg) = (gamma.cos(), gamma.sin());
    // R (jqmc closed form), then transpose.
    let r = [
        [cb * cg, cg * sa * sb - ca * sg, sa * sg + ca * cg * sb],
        [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
        [-sb, cb * sa, ca * cb],
    ];
    let mut rt = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rt[i][j] = r[j][i];
        }
    }
    rt
}

/// Haar SO(3) rotation sampled from a Threefry key without advancing the GFMC
/// key carry.  For Rz(gamma) Ry(beta) Rx(alpha), invariant measure is uniform
/// in alpha, gamma, and sin(beta).
fn sample_rt_from_key(key: [u32; 2], random_mesh: bool) -> [[f64; 3]; 3] {
    if !random_mesh {
        return RT_ID;
    }
    let u = crate::threefry::uniform_f64(key, 3, 0.0, 1.0);
    let alpha = std::f64::consts::TAU * u[0];
    let beta = (2.0 * u[1] - 1.0).asin();
    let gamma = std::f64::consts::TAU * u[2];
    rotation_matrix_t(alpha, beta, gamma)
}

/// Match Python `_split_lrdmc_rotation_keys`: the kinetic mesh uses the
/// rotation key itself, while ECP uses an independent child substream.
fn ecp_rotation_key(rotation_key: [u32; 2]) -> [u32; 2] {
    crate::threefry::split(rotation_key, 2)[1]
}

/// Legacy driver sampler with the same Haar law as `sample_rt_from_key`.
fn sample_rt(rng: &mut GfmcRng, random_mesh: bool) -> [[f64; 3]; 3] {
    if !random_mesh {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let alpha = std::f64::consts::TAU * rng.uniform();
    let beta = (2.0 * rng.uniform() - 1.0).asin();
    let gamma = std::f64::consts::TAU * rng.uniform();
    rotation_matrix_t(alpha, beta, gamma)
}

/// The 6 lattice shift vectors `+/-alat*rt[axis]` in x+,x-,y+,y-,z+,z- order.
pub(crate) fn lattice_shifts(alat: f64, rt: &[[f64; 3]; 3]) -> [[f64; 3]; 6] {
    let mut s = [[0.0_f64; 3]; 6];
    for axis in 0..3 {
        for c in 0..3 {
            s[2 * axis][c] = alat * rt[axis][c];
            s[2 * axis + 1][c] = -alat * rt[axis][c];
        }
    }
    s
}

/// Discretized-kinetic hopping amplitudes `elements_kinetic_part` (length
/// `6*(N_up+N_dn)`), up-configs first (electron-major, shift-minor), then dn.
#[allow(clippy::too_many_arguments)]
pub fn compute_discretized_kinetic_elements(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
) -> Vec<f64> {
    let shifts = lattice_shifts(alat, rt);
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let j_base = crate::coulomb_potential::jastrow_value_for_ratio(jastrow, j3, r_up, r_dn);
    let (s_x, l_x) = det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, r_dn).view());
    let inv_2a2 = -1.0 / (2.0 * alat * alat);
    let mut out = Vec::with_capacity(6 * (n_up + n_dn));
    // up configs: move up-electron e by shift s
    for e in 0..n_up {
        for shifts_s in &shifts {
            let mut mu = r_up.to_owned();
            for c in 0..3 {
                mu[[e, c]] += shifts_s[c];
            }
            let jp =
                crate::coulomb_potential::jastrow_value_for_ratio(jastrow, j3, &mu.view(), r_dn);
            let (sp, lp) =
                det_sign_lnabs(&compute_geminal_all_elements(gem, &mu.view(), r_dn).view());
            let wf_ratio = (jp - j_base).exp() * (sp * s_x) * (lp - l_x).exp();
            out.push(inv_2a2 * wf_ratio);
        }
    }
    // dn configs
    for e in 0..n_dn {
        for shifts_s in &shifts {
            let mut md = r_dn.to_owned();
            for c in 0..3 {
                md[[e, c]] += shifts_s[c];
            }
            let jp =
                crate::coulomb_potential::jastrow_value_for_ratio(jastrow, j3, r_up, &md.view());
            let (sp, lp) =
                det_sign_lnabs(&compute_geminal_all_elements(gem, r_up, &md.view()).view());
            let wf_ratio = (jp - j_base).exp() * (sp * s_x) * (lp - l_x).exp();
            out.push(inv_2a2 * wf_ratio);
        }
    }
    out
}

/// Shared per-configuration base state for the FAST (Sherman-Morrison rank-1) mesh
/// kernels: the streaming det-ratio state (single-path AO matrices), the base geminal
/// inverse, and the incremental Jastrow states (J3 rank-1 tables + DTN streaming rows).
/// Built ONCE per projection interval by the GFMC drivers (or once per `e_L`
/// evaluation on the stateless entry points) and consumed by BOTH one-electron-move
/// meshes of the LRDMC sub-step -- the 6N discretized-kinetic hop candidates AND the
/// ECP T-move quadrature mesh (same move class: each point displaces exactly one
/// electron).
///
/// Per-member gating: the state exists for every basis
/// and PBC. `det_state`/`ginv` init is basis-generic (`compute_single` covers
/// GTO/pGTO images/PW/B-spline, MO or raw AO) and the SM rank-1 advance is pure
/// algebra. `j3_state` is `None` for MO-wrapped / periodic image-tiled /
/// non-contiguous-`j_matrix` J3 -- consumers then fall back to the full-value J3
/// evaluation FOR J3 ONLY. The candidate AO tables and derivative caches engage
/// per [`CandidateBasisClass`]: molecular GTO keeps the
/// legacy bit-identical kernels, while periodic pGTO / B-spline / pGTO+PW dets
/// (and raw-AO periodic pGTO J3) drive the folded basis-generic kernels
/// on home-cell-wrapped candidate positions. Pure PW and pGTO+B-spline share
/// the generic composite evaluator; genuinely empty or malformed simultaneous
/// PW+B-spline leaves take the counted slow path. The basis-agnostic members
/// (`dtn_state`, `ecp_geometry_cache`, `ecp_local_cache`,
/// `j12_grad_cache`) persist and advance across the interval for all bases.
pub(crate) struct MeshFastState {
    det_state: crate::determinant::DetRatioState,
    ginv: Array2<f64>,
    j3_state: Option<crate::jastrow_factor::J3RatioState>,
    dtn_state: Option<crate::dtn::DtnRatioState>,
    /// All-electron x six-direction DTN log-ratio table, maintained across
    /// accepted hops while the kinetic rotation is fixed.
    dtn_candidate_table: std::cell::RefCell<Option<crate::dtn::DtnCandidateTable>>,
    /// DTN ratios for the arbitrary ECP quadrature points.  This is separate
    /// from the six-direction kinetic table because the two meshes coexist.
    ecp_dtn_candidate_table: std::cell::RefCell<Option<crate::dtn::DtnCandidateTable>>,
    ecp_dtn_dirty: std::cell::Cell<Option<HopChange>>,
    /// Home-cell wrap (AO-eval coordinate only): `Some((cell, inv))` under PBC.
    cell: Option<(Array2<f64>, [[f64; 3]; 3])>,
    /// Experimental TurboRVB-style candidate-mesh table. Enabled only when the
    /// random rotation is held fixed for one GFMC_n projection interval. The
    /// determinant/J3 derivative tables below do not depend on mesh rotation.
    reuse_fixed_mesh: bool,
    kinetic_cache: std::cell::RefCell<Option<KineticAoCache>>,
    kinetic_dirty: std::cell::Cell<Option<HopChange>>,
    ecp_cache: std::cell::RefCell<Option<EcpAoCache>>,
    ecp_geometry_cache: std::cell::RefCell<Option<EcpGeometryCache>>,
    ecp_dirty: std::cell::Cell<Option<HopChange>>,
    /// When the ECP mesh is evaluated before the kinetic mesh, molecular
    /// determinant candidates for the moved electron are evaluated as one
    /// `(6 + Nv)` batch.  The ECP rows are consumed immediately and these six
    /// kinetic rows are handed to the following kinetic-cache refresh.
    kinetic_det_prefetch: std::cell::RefCell<Option<DetCandidatePrefetch>>,
    det_grad_cache: std::cell::RefCell<Option<crate::determinant::DetGradLapCache>>,
    j3_grad_cache: std::cell::RefCell<Option<crate::jastrow_factor::J3GradLapCache>>,
    j12_grad_cache: std::cell::RefCell<Option<crate::jastrow_factor::J12GradLapCache>>,
    /// Projection-scoped, basis-agnostic, PBC-native DTN derivative table,
    /// shared with the same seed/advance implementation as the complex path.
    dtn_grad_cache: std::cell::RefCell<Option<crate::dtn::DtnGradLapCache>>,
    /// All-electron determinant contraction rows, built as cache-blocked GEMMs
    /// and shared by the kinetic and ECP candidate meshes until the next hop.
    det_wdot_cache: std::cell::RefCell<Option<DetWdotCache>>,
    /// J3 analogue of `det_wdot_cache`.  The production molecular J3 basis is
    /// raw AO, so every candidate becomes one row dot plus the J1/J2 delta.
    j3_wdot_cache: std::cell::RefCell<Option<J3WdotCache>>,
    /// `J3 * (AO_new - AO_old)` from the last accepted hop.  Both the kinetic
    /// and ECP persistent candidate tables consume (but do not take) this
    /// common increment when their independently cached meshes are refreshed.
    j3_pending_delta: std::cell::RefCell<Option<J3PendingDelta>>,
    ecp_local_cache: std::cell::RefCell<Option<EcpLocalCache>>,
    ecp_local_dirty: std::cell::Cell<Option<(bool, usize)>>,
    /// Skip electrons whose ECP quadrature amplitudes are all exactly 0.0
    /// (IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001). Read once per state from the
    /// test-only A/B hook `JQMC_RS_GFMC_ECP_KEEP_ZERO_GROUPS` (set = keep).
    ecp_skip_zero_groups: bool,
    /// FAST radial cut of the ECP quadrature, in Ha
    /// (IMPL-RS-GFMC-FAST-ECP-RADIAL-CUT-001): an (electron, non-local term)
    /// pair with `|v_t(r)| < ecp_radial_cut` contributes nothing (truncated
    /// Hamiltonian H'). `FAST_ECP_NONLOCAL_RADIAL_CUT`, or 0.0 (no cut) under
    /// the test-only A/B hook `JQMC_RS_GFMC_ECP_NO_RADIAL_CUT`.
    ecp_radial_cut: f64,
}

#[derive(Clone, Copy)]
struct HopChange {
    is_up: bool,
    electron: usize,
    old_pos: [f64; 3],
}

struct DetCandidatePrefetch {
    is_up: bool,
    electron: usize,
    kinetic: Array2<f64>,
}

struct DetWdotCache {
    up: Array2<f64>,
    up_base: Vec<f64>,
    dn: Array2<f64>,
    dn_base: Vec<f64>,
}

/// Rank-one advance of the real determinant ``winvup/winvdo`` tables.
///
/// ``up`` stores ``(C_up G^-1)^T`` and ``dn`` stores
/// ``G^-1 C_dn``.  The update is O(N*M), uses the same SM factors that update
/// ``ginv``, and performs no candidate-orbital evaluation.
#[allow(clippy::too_many_arguments)]
fn advance_det_wdot_cache(
    cache: &mut DetWdotCache,
    gem: &GeminalData,
    det_state: &crate::determinant::DetRatioState,
    is_up: bool,
    e: usize,
    det_col: &[f64],
    sm_u: &[f64],
    sm_x: &[f64],
    sm_y: &[f64],
    rho: f64,
) {
    let n = cache.up.nrows();
    let m_up = cache.up.ncols();
    let m_dn = cache.dn.ncols();
    debug_assert_eq!(sm_x.len(), n);
    debug_assert_eq!(sm_y.len(), n);

    if is_up {
        let pivot_up = cache.up.row(e).to_vec();
        for (i, sm_y_i) in sm_y[..n].iter().enumerate() {
            let scale = *sm_y_i / rho;
            for (a, pivot_up_a) in pivot_up[..m_up].iter().enumerate() {
                cache.up[[i, a]] -= scale * *pivot_up_a;
            }
        }
        let mut yd_minus_delta = vec![0.0_f64; m_dn];
        for a in 0..m_dn {
            let yd = (0..n)
                .map(|i| sm_y[i] * det_state.paired_up_lambda[[i, a]])
                .sum::<f64>();
            let new_coeff = if gem.is_sd() {
                det_col[a]
            } else {
                (0..det_col.len())
                    .map(|o| det_col[o] * gem.lambda_value(o, a))
                    .sum::<f64>()
            };
            let delta = new_coeff - det_state.paired_up_lambda[[e, a]];
            yd_minus_delta[a] = yd - delta;
        }
        for (i, sm_x_i) in sm_x[..n].iter().enumerate() {
            let scale = *sm_x_i / rho;
            for (a, yd_minus_delta_a) in yd_minus_delta[..m_dn].iter().enumerate() {
                cache.dn[[i, a]] -= scale * *yd_minus_delta_a;
            }
        }
    } else {
        let mut cup_x_minus_delta = vec![0.0_f64; m_up];
        for a in 0..m_up {
            let cup_x = (0..n).map(|i| cache.up[[i, a]] * sm_u[i]).sum::<f64>();
            let delta = if gem.is_sd() {
                if a < det_state.paired_dn.nrows() {
                    det_col[a] - det_state.paired_dn[[a, e]]
                } else {
                    0.0
                }
            } else {
                let new_coeff = (0..det_col.len())
                    .map(|b| gem.lambda_value(a, b) * det_col[b])
                    .sum::<f64>();
                new_coeff - det_state.paired_dn[[a, e]]
            };
            cup_x_minus_delta[a] = cup_x - delta;
        }
        for (i, sm_y_i) in sm_y[..n].iter().enumerate() {
            let scale = *sm_y_i / rho;
            for (a, cup_x_minus_delta_a) in cup_x_minus_delta[..m_up].iter().enumerate() {
                cache.up[[i, a]] -= scale * *cup_x_minus_delta_a;
            }
        }
        let pivot_dn = cache.dn.row(e).to_vec();
        for (i, sm_x_i) in sm_x[..n].iter().enumerate() {
            let scale = *sm_x_i / rho;
            for (a, pivot_dn_a) in pivot_dn[..m_dn].iter().enumerate() {
                cache.dn[[i, a]] -= scale * *pivot_dn_a;
            }
        }
    }

    // Preserve the existing affine convention exactly: ``ratio =
    // 1-base + q.orb(new)``. Re-evaluating these O(N*M) bases also absorbs
    // harmless roundoff drift in the maintained weights at the current point.
    for i in 0..n {
        let mut base = 0.0;
        // Iterate the always-full cache row (``m_up == ncols``); the moved
        // spin's ``det_col`` has that spin's length only, so it is read on
        // its branch alone (a down hop in an open shell has m_dn < m_up).
        for (a, w) in cache.up.row(i).iter().enumerate() {
            let value = if is_up && i == e {
                det_col[a]
            } else {
                det_state.ao_up[[a, i]]
            };
            base += *w * value;
        }
        cache.up_base[i] = base;
    }
    for i in 0..det_state.ao_dn.ncols() {
        let mut base = 0.0;
        for (a, w) in cache.dn.row(i).iter().enumerate() {
            let value = if !is_up && i == e {
                det_col[a]
            } else {
                det_state.ao_dn[[a, i]]
            };
            base += *w * value;
        }
        cache.dn_base[i] = base;
    }
}

struct J3WdotCache {
    up: Array2<f64>,
    up_base: Vec<f64>,
    dn: Array2<f64>,
    dn_base: Vec<f64>,
}

struct J3PendingDelta {
    d: Vec<f64>,
    /// Post-hop `AO_current^T*d`, shared by kinetic and ECP candidate-table
    /// advances. Kept spin-separated to match the ratio-state layout.
    current_up: Vec<f64>,
    current_dn: Vec<f64>,
}

fn j3_current_dots(state: &crate::jastrow_factor::J3RatioState, d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut up = vec![0.0_f64; state.aos_up.ncols()];
    let mut dn = vec![0.0_f64; state.aos_dn.ncols()];
    for (a, &da) in d.iter().enumerate() {
        let au = state.aos_up.row(a);
        let au = au.as_slice().expect("J3 up AO row contiguous");
        for (value, &ao) in up.iter_mut().zip(au) {
            *value += ao * da;
        }
        let ad = state.aos_dn.row(a);
        let ad = ad.as_slice().expect("J3 down AO row contiguous");
        for (value, &ao) in dn.iter_mut().zip(ad) {
            *value += ao * da;
        }
    }
    (up, dn)
}

/// Advance the all-electron J3 q rows instead of invalidating and rebuilding
/// them after every hop. For every unchanged electron q gains the common
/// `d=J3*delta_AO`; the moved electron's q is invariant. The positive
/// `AO_current.q` bases follow from the same shared current dots.
fn advance_j3_wdot_cache(
    cache: &mut J3WdotCache,
    state: &crate::jastrow_factor::J3RatioState,
    moved_is_up: bool,
    moved_e: usize,
    d: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(cache.up.ncols(), d.len());
    debug_assert_eq!(cache.dn.ncols(), d.len());
    let (current_up, current_dn) = j3_current_dots(state, d);

    for (e, current_up_e) in current_up[..cache.up.nrows()].iter().enumerate() {
        if moved_is_up && e == moved_e {
            continue;
        }
        let row = cache.up.row_mut(e);
        for (q, &da) in row.into_iter().zip(d) {
            *q += da;
        }
        cache.up_base[e] += *current_up_e;
    }
    for (e, current_dn_e) in current_dn[..cache.dn.nrows()].iter().enumerate() {
        if !moved_is_up && e == moved_e {
            continue;
        }
        let row = cache.dn.row_mut(e);
        for (q, &da) in row.into_iter().zip(d) {
            *q += da;
        }
        cache.dn_base[e] += *current_dn_e;
    }

    let (q, aos, base) = if moved_is_up {
        (&cache.up, &state.aos_up, &mut cache.up_base[moved_e])
    } else {
        (&cache.dn, &state.aos_dn, &mut cache.dn_base[moved_e])
    };
    *base = 0.0;
    for a in 0..d.len() {
        *base += aos[[a, moved_e]] * q[[moved_e, a]];
    }
    (current_up, current_dn)
}

struct PackedCandidateGroup {
    /// Ascending raw-AO indices represented by each packed row.
    columns: Vec<usize>,
    /// Row-major `rows x columns.len()` values.
    values: Vec<f64>,
    rows: usize,
}

struct EcpLocalCache {
    up: Vec<f64>,
    dn: Vec<f64>,
}

struct KineticAoCache {
    alat_bits: u64,
    rt_bits: [u64; 9],
    det_up: Array2<f64>,
    det_dn: Array2<f64>,
    j3: Option<Array2<f64>>,
    /// Ascending raw-AO columns alive in at least one of each electron's six
    /// kinetic candidates. Molecular fast-mode LEPSBAS screening makes this
    /// local even when the J3 coefficient matrix itself is dense.
    j3_active: Option<Vec<Vec<usize>>>,
    /// Physically packed sibling of `j3`, used by the persistent table advance
    /// so exact-zero molecular columns consume neither bandwidth nor SIMD
    /// lanes. One group per electron.
    j3_packed: Option<Vec<PackedCandidateGroup>>,
    /// J3 log ratio for every row of `j3`, maintained across accepted hops.
    j3_delta: Option<Vec<f64>>,
    j12_up: Option<Vec<f64>>,
    j12_dn: Option<Vec<f64>>,
}

struct EcpAoCache {
    rt_bits: [u64; 9],
    rows_per_electron: usize,
    det_up: Array2<f64>,
    det_dn: Array2<f64>,
    j3: Option<Array2<f64>>,
    /// Ascending raw-AO columns alive in each electron's ECP candidate group.
    /// The molecular GTO evaluator writes exact zero outside LEPSBAS, so the
    /// persistent tabpip advance need not stream the dense candidate matrix.
    j3_active: Option<Vec<Vec<usize>>>,
    /// Physically packed sibling of `j3`, one ECP candidate group/electron.
    j3_packed: Option<Vec<PackedCandidateGroup>>,
    /// J3 log ratio for every row of `j3`, maintained across accepted hops.
    j3_delta: Option<Vec<f64>>,
    j12_up: Option<Vec<f64>>,
    j12_dn: Option<Vec<f64>>,
}

struct EcpGeometryCache {
    rt_bits: [u64; 9],
    rows_per_electron: usize,
    positions_up: Vec<[f64; 3]>,
    positions_dn: Vec<[f64; 3]>,
    angular_up: Vec<f64>,
    angular_dn: Vec<f64>,
}

/// Whether the determinant leaves can use the MOLECULAR det/J3 candidate-table
/// and derivative-cache members of [`MeshFastState`] (the truly molecular-GTO
/// parts).  The rank-1 algebra and the derivative tables are independent of the
/// angular AO representation; both Cartesian and spherical GTOs are supported
/// by `compute_aos_grad_lap_fast_cut` and the candidate-table dispatcher below.
///
/// This gate does not decide whether a
/// projection-scoped state exists at all: the state is carried for EVERY basis
/// and PBC, and each cache member engages only when its own prerequisites hold
/// (DTN / ECP-geometry / ECP-local / J1J2 members are basis-agnostic).
#[inline]
fn fast_mol_kinetic_ok_cell(gem: &GeminalData, cell_present: bool) -> bool {
    !cell_present
        && gem.orb_up.mo_coefficients.is_some()
        && gem.orb_dn.mo_coefficients.is_some()
        && gem.orb_up.pw.is_none()
        && gem.orb_dn.pw.is_none()
        && gem.orb_up.bspline.is_none()
        && gem.orb_dn.bspline.is_none()
        && gem.orb_up.aos.n_image == 1
        && gem.orb_dn.aos.n_image == 1
        && !gem.orb_up.aos.shells.is_empty()
        && !gem.orb_dn.aos.shells.is_empty()
}

#[inline]
fn fast_mol_kinetic_ok(gem: &GeminalData, pbc: Option<&crate::ewald::EwaldRef>) -> bool {
    fast_mol_kinetic_ok_cell(gem, pbc.is_some())
}

#[inline]
fn molecular_candidate_ok(orb: &OrbData, require_raw_ao: bool) -> bool {
    orb.pw.is_none()
        && orb.bspline.is_none()
        && (!require_raw_ao || orb.mo_coefficients.is_none())
        && orb.aos.n_image == 1
        && !orb.aos.shells.is_empty()
}

/// Candidate-route capability class of one orbital block: which
/// per-basis candidate/derivative kernel family the projection-scoped mesh
/// caches may drive for this orbital. `MolecularGto` is the legacy route and
/// keeps the exact legacy kernels (bit-identical chain); the other supported
/// classes drive the folded kernels (`compute_aos_candidate_batch_folded`,
/// `AosBsplineData::candidate_batch_into`, `composite_candidate_batch_into`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CandidateBasisClass {
    /// Molecular GTO (no PBC cell, single image) -- legacy bit-identical route.
    MolecularGto,
    /// Periodic pGTO under a PBC cell (image-folded candidate kernels; a
    /// molecular-shaped leaf under a cell also lands here: the fold is the
    /// identity and the value screen stays off at `n_image == 1`).
    PeriodicPgto,
    /// Pure B-spline leaf (0-AO GTO placeholder).
    Bspline,
    /// Generic composite evaluator: pure PW, pGTO+PW, or pGTO+B-spline
    /// (``gto_first`` concat). The variant keeps the name `PgtoPw` for all
    /// three so route-counter indices remain append-only.
    PgtoPw,
    /// Slow path: an actually empty orbital placeholder, a periodic pGTO leaf
    /// without the cell needed by its pre-wrap contract, or a simultaneous
    /// PW+B-spline payload (not a valid production leaf). Never silent.
    Unsupported,
}

fn candidate_basis_class(
    orb: &OrbData,
    cell_present: bool,
    require_raw_ao: bool,
) -> CandidateBasisClass {
    use CandidateBasisClass::*;
    if require_raw_ao && orb.mo_coefficients.is_some() {
        return Unsupported;
    }
    match (&orb.pw, &orb.bspline) {
        (None, None) => {
            if orb.aos.shells.is_empty() || orb.aos.num_ao == 0 {
                Unsupported // Empty placeholder: never build an all-zero table.
            } else if !cell_present {
                if orb.aos.n_image == 1 {
                    MolecularGto
                } else {
                    // Periodic tiling but no cell to wrap candidates with:
                    // the folded kernels' pre-wrap contract cannot be met.
                    Unsupported
                }
            } else {
                PeriodicPgto
            }
        }
        (Some(_), None) => {
            if orb.aos.num_ao == 0 || orb.aos.n_image == 1 || cell_present {
                PgtoPw
            } else {
                Unsupported
            }
        }
        (None, Some(bs)) => {
            if orb.aos.num_ao == 0 && bs.num_ao > 0 {
                Bspline // wraps internally in fractional space
            } else if bs.num_ao > 0 && (orb.aos.n_image == 1 || cell_present) {
                PgtoPw // generic composite builder also handles pGTO+B-spline
            } else {
                Unsupported
            }
        }
        (Some(_), Some(_)) => Unsupported,
    }
}

/// Determinant-side capability for one spin. The production Python model uses
/// either MO/MO or raw-AO/raw-AO spin sectors; classifying each side here keeps
/// the native table width aligned with its actual orbital representation.
fn det_candidate_class(
    _gem: &GeminalData,
    orb: &OrbData,
    cell_present: bool,
) -> CandidateBasisClass {
    candidate_basis_class(orb, cell_present, false)
}

/// Column space a determinant candidate table is built in. Each spin is
/// classified from its own orbital leaf so the table width and maintained
/// winv-row width agree. Production Python inputs are matched MO/MO or
/// raw-AO/raw-AO pairs.
/// - `Mo`: rows in occupied-MO space, contracted against the SHARED
///   `det_wdot` rows (`candidate_wdot_into`);
/// - `RawAo`: rows in the orbital's own raw-AO space, consumed by
///   [`MeshFastState::det_ratios_from_candidate_aos_into`], which folds the
///   rank-1 weight `q` through the MO coefficients ITSELF and therefore
///   requires AO-width rows for MO-wrapped and raw-AO leaves alike.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CandidateSpace {
    Mo,
    RawAo,
}

fn det_candidate_space(orb: &OrbData) -> CandidateSpace {
    if orb.mo_coefficients.is_some() {
        CandidateSpace::Mo
    } else {
        CandidateSpace::RawAo
    }
}

/// J3 candidate capability: raw-AO molecular GTO (legacy) or raw-AO periodic
/// pGTO. Raw-AO B-spline and composite J3 candidate batches are unsupported and
/// fall back to the per-point full-value J3 path.
fn j3_candidate_class(orb: &OrbData, cell_present: bool) -> CandidateBasisClass {
    match candidate_basis_class(orb, cell_present, true) {
        c @ (CandidateBasisClass::MolecularGto | CandidateBasisClass::PeriodicPgto) => c,
        _ => CandidateBasisClass::Unsupported,
    }
}

/// Whether the basis-generic determinant derivative cache
/// (`init_det_grad_lap_cache_folded` / `update_det_grad_lap_cache_and_value_folded`)
/// may engage: both spins in a supported class and the geminal NOT on the
/// molecular MO fast route (`fast_mol_kinetic_ok`, which keeps the legacy
/// kernels). Besides the periodic pGTO / B-spline / pGTO+PW leaves this covers
/// the molecular RAW-AO geminal (AO-basis AGP / AGPn from `to_AGP` /
/// `convert_from_MOs_to_AOs`): the folded single-point kernel evaluates a
/// one-image molecular shell list without screening the value, and the
/// raw-AO derivative tables keep AO rank, so the same per-hop column refresh
/// replaces the per-sub-step full rebuild.
fn fast_generic_det_gradlap_cell(gem: &GeminalData, cell_present: bool) -> bool {
    use CandidateBasisClass::*;
    !fast_mol_kinetic_ok_cell(gem, cell_present)
        && matches!(
            candidate_basis_class(&gem.orb_up, cell_present, false),
            MolecularGto | PeriodicPgto | Bspline | PgtoPw
        )
        && matches!(
            candidate_basis_class(&gem.orb_dn, cell_present, false),
            MolecularGto | PeriodicPgto | Bspline | PgtoPw
        )
}

/// Driver-level gate for `MeshFastState::det_grad_lap`: the molecular fast
/// tables or the basis-generic derivative cache.
#[inline]
fn fast_det_gradlap_ok_cell(gem: &GeminalData, cell_present: bool) -> bool {
    fast_mol_kinetic_ok_cell(gem, cell_present) || fast_generic_det_gradlap_cell(gem, cell_present)
}

#[inline]
fn fast_det_gradlap_ok(gem: &GeminalData, pbc: Option<&crate::ewald::EwaldRef>) -> bool {
    fast_det_gradlap_ok_cell(gem, pbc.is_some())
}

#[derive(Clone, Copy)]
enum CandidateEvent {
    Build,
    Refresh,
}

fn note_candidate_event(class: CandidateBasisClass, event: CandidateEvent) {
    use CandidateBasisClass::*;
    let counter = match (class, event) {
        (MolecularGto, CandidateEvent::Build) => &GFMC_CAND_BUILD_MOL,
        (MolecularGto, CandidateEvent::Refresh) => &GFMC_CAND_REFRESH_MOL,
        (PeriodicPgto, CandidateEvent::Build) => &GFMC_CAND_BUILD_PGTO,
        (PeriodicPgto, CandidateEvent::Refresh) => &GFMC_CAND_REFRESH_PGTO,
        (Bspline, CandidateEvent::Build) => &GFMC_CAND_BUILD_BSPLINE,
        (Bspline, CandidateEvent::Refresh) => &GFMC_CAND_REFRESH_BSPLINE,
        (PgtoPw, CandidateEvent::Build) => &GFMC_CAND_BUILD_PGTOPW,
        (PgtoPw, CandidateEvent::Refresh) => &GFMC_CAND_REFRESH_PGTOPW,
        (Unsupported, _) => &GFMC_CAND_DET_SLOW,
    };
    bump_route_counter(counter);
}

/// Candidate-route capability class of one complex orbital leaf.
/// The complex batch evaluator (`ComplexOrb::value`) is a single polymorphic
/// entry, so unlike the real classifier there is no per-class kernel choice --
/// the class only decides supported-vs-unsupported and which counter proves
/// engagement. Pure PW and pGTO+B-spline use the same polymorphic evaluator as
/// pGTO+PW and are therefore supported here.
/// `MolecularGto` never occurs on the twist (the complex leaves are periodic
/// by construction; a Gamma-real child folds with unit phases and still lands
/// in `PeriodicPgto`).
fn complex_candidate_class(orb: &ComplexOrb) -> CandidateBasisClass {
    use CandidateBasisClass::*;
    match (&orb.pw, &orb.bspline) {
        (None, None) => {
            if orb.aos.shells.is_empty() || orb.aos.num_ao == 0 {
                Unsupported // Empty placeholder: never build an all-zero table.
            } else {
                PeriodicPgto
            }
        }
        (Some(_), None) => {
            PgtoPw // generic ComplexOrb::value covers pure PW and pGTO+PW
        }
        (None, Some(bs)) => {
            if orb.aos.num_ao == 0 && bs.num_ao > 0 {
                Bspline // two real splines + Bloch phase, wraps internally
            } else if bs.num_ao > 0 {
                PgtoPw // generic ComplexOrb::value covers pGTO+B-spline
            } else {
                Unsupported
            }
        }
        (Some(_), Some(_)) => Unsupported,
    }
}

fn note_candidate_event_c(class: CandidateBasisClass, event: CandidateEvent) {
    use CandidateBasisClass::*;
    let counter = match (class, event) {
        (PeriodicPgto, CandidateEvent::Build) => &GFMC_CPLX_CAND_BUILD_PGTO,
        (PeriodicPgto, CandidateEvent::Refresh) => &GFMC_CPLX_CAND_REFRESH_PGTO,
        (Bspline, CandidateEvent::Build) => &GFMC_CPLX_CAND_BUILD_BSPLINE,
        (Bspline, CandidateEvent::Refresh) => &GFMC_CPLX_CAND_REFRESH_BSPLINE,
        (PgtoPw, CandidateEvent::Build) => &GFMC_CPLX_CAND_BUILD_PGTOPW,
        (PgtoPw, CandidateEvent::Refresh) => &GFMC_CPLX_CAND_REFRESH_PGTOPW,
        // MolecularGto cannot occur on the twist (see `complex_candidate_class`);
        // count it as the slow path if it ever does rather than silently.
        (MolecularGto, _) | (Unsupported, _) => &GFMC_CPLX_CAND_DET_SLOW,
    };
    bump_route_counter(counter);
}

/// Complex derivative-cache event counter (build = table seed per spin
/// block, refresh = per-hop moved-electron column), keyed by the orbital's
/// basis class exactly like [`note_candidate_event_c`].
fn note_deriv_event_c(class: CandidateBasisClass, event: CandidateEvent) {
    use CandidateBasisClass::*;
    let counter = match (class, event) {
        (PeriodicPgto, CandidateEvent::Build) => &GFMC_CPLX_DERIV_BUILD_PGTO,
        (PeriodicPgto, CandidateEvent::Refresh) => &GFMC_CPLX_DERIV_REFRESH_PGTO,
        (Bspline, CandidateEvent::Build) => &GFMC_CPLX_DERIV_BUILD_BSPLINE,
        (Bspline, CandidateEvent::Refresh) => &GFMC_CPLX_DERIV_REFRESH_BSPLINE,
        (PgtoPw, CandidateEvent::Build) => &GFMC_CPLX_DERIV_BUILD_PGTOPW,
        (PgtoPw, CandidateEvent::Refresh) => &GFMC_CPLX_DERIV_REFRESH_PGTOPW,
        // The cache is gated on supported classes; count a leak rather than
        // silently misattributing it.
        (MolecularGto, _) | (Unsupported, _) => &GFMC_CPLX_DERIV_DET_SLOW,
    };
    bump_route_counter(counter);
}

/// Gate for the complex determinant derivative cache: both spins must use a
/// supported class (periodic pGTO cplx / B-spline cplx / pGTO+PW
/// cplx). The fused `value_grad_lap` seeding kernel is polymorphic over the
/// whole complex leaf taxonomy, but the out-of-scope classes (pure PW,
/// pGTO+B-spline) keep the counted per-sub-step rebuild, by the same scope
/// rule that keeps them off the candidate tables.
#[inline]
fn complex_det_gradlap_ok(gem: &ComplexGeminal) -> bool {
    complex_candidate_class(&gem.orb_up) != CandidateBasisClass::Unsupported
        && complex_candidate_class(&gem.orb_dn) != CandidateBasisClass::Unsupported
}

/// Gate for the streamed J1/J2/DTN grad/lap route of the complex local energy.
/// Accumulation-order differences are handled by backend-parity tolerance and
/// do not disable the cache in mixed precision.
#[inline]
fn complex_jaspart_gradlap_ok(jastrow: &JastrowData) -> bool {
    crate::jastrow_factor::j12_generic_cache_ok(jastrow) || jastrow.dtn.is_some()
}

/// Complex determinant candidate table for one spin: the batched
/// `ComplexOrb::value` chain (fold + wrap/restore + PW / B-spline children + MO
/// projection -- the SAME kernel family `orb_single_complex` drives one row at
/// a time, so seed and refresh share one fp chain and a table row matches the
/// per-point slow path's column). Candidate positions are passed RAW: the
/// complex leaf home-cell-wraps internally (`WrapCtx`), so unlike the real
/// routes there is no external wrap site to keep consistent.
fn complex_det_candidate_batch(
    orb: &ComplexOrb,
    class: CandidateBasisClass,
    positions: &[[f64; 3]],
    event: CandidateEvent,
) -> Array2<Complex64> {
    note_candidate_event_c(class, event);
    debug_assert!(class != CandidateBasisClass::Unsupported);
    crate::complex_det::complex_candidate_batch(orb, positions)
}

/// Real-valued `Re(complex orbital)` J3 candidate table: rows are the
/// exact `j3_ao_re_single` columns the per-point complex J3 ratio consumes, so
/// the rank-1 `j3_ratio` off a table row reproduces the per-point chain.
fn complex_j3_candidate_batch(
    j3: &ComplexOrb,
    class: CandidateBasisClass,
    positions: &[[f64; 3]],
    event: CandidateEvent,
) -> Array2<f64> {
    note_candidate_event_c(class, event);
    crate::complex_det::j3_re_candidate_batch(j3, positions)
}

/// Home-cell wrap of a candidate-position block for the AO-evaluation kernels.
/// The candidate kernels require pre-wrapped positions
/// under PBC, and there must be exactly ONE wrap site per (re)build/refresh so
/// every kernel sees the same bits. Borrow-through when there is no PBC cell
/// (the molecular chain sees the identical slice -- bit-identical). DTN and
/// J1/J2 consumers keep the RAW positions (MCMC streaming convention).
fn wrapped_candidate_positions<'a>(
    st: &MeshFastState,
    raw: &'a [[f64; 3]],
) -> std::borrow::Cow<'a, [[f64; 3]]> {
    match &st.cell {
        None => std::borrow::Cow::Borrowed(raw),
        Some(_) => std::borrow::Cow::Owned(raw.iter().map(|&p| st.wrap(p)).collect()),
    }
}

/// AO-layer candidate value batch for the non-molecular routes.
/// Positions must be PRE-WRAPPED under PBC (`wrapped_candidate_positions`).
fn basis_ao_candidate_batch(
    orb: &OrbData,
    class: CandidateBasisClass,
    positions: &[[f64; 3]],
) -> Array2<f64> {
    match class {
        CandidateBasisClass::PeriodicPgto => {
            crate::atomic_orbital::compute_aos_candidate_batch_folded(&orb.aos, positions)
        }
        CandidateBasisClass::Bspline => {
            let bs = orb
                .bspline
                .as_ref()
                .expect("B-spline candidate route requires the spline leaf");
            let mut out = Array2::<f64>::zeros((positions.len(), bs.num_ao));
            bs.candidate_batch_into(
                positions,
                out.as_slice_mut().expect("candidate batch contiguous"),
            );
            out
        }
        CandidateBasisClass::PgtoPw => {
            let n_tot = orb.ao_count();
            let mut out = Array2::<f64>::zeros((positions.len(), n_tot));
            crate::determinant::composite_candidate_batch_into(
                orb,
                positions,
                out.as_slice_mut().expect("candidate batch contiguous"),
            );
            out
        }
        CandidateBasisClass::MolecularGto | CandidateBasisClass::Unsupported => {
            unreachable!("callers dispatch molecular/unsupported before the per-basis batch")
        }
    }
}

/// Determinant candidate table for one spin, in the space `space` names.
/// Molecular MO keeps the legacy MO-space batch (bit-identical kernels and
/// call order); a molecular raw-AO leaf (raw-AO SD/AGP: no `mo_coefficients`)
/// keeps the same legacy AO batch WITHOUT the projection, exactly as
/// `MeshFastState::det_ratios` builds it. The non-molecular routes build the
/// per-basis AO batch and project through the SCALAR
/// `project_candidates_to_mos` only in `Mo` space.
///
/// Floating-point conventions, one per table:
/// - (a) periodic pGTO / composite-GTO VALUE rows use the SCREENED folded
///   batch (`LEPSBAS` image screen when `n_image > 1`): the fast-mode ~1e-9
///   accuracy class, accepted under the fast-chain tolerance contract (the
///   prod-vs-debug tiers are the detectors).
/// - (b) B-spline rows use the `value_single` bit-chain -- the SAME kernel
///   family seeds the table and refreshes hopped rows (never the batch
///   `value()` fp order).
/// - (c) MO projection is the deterministic scalar `project_candidates_to_mos`
///   (gfmc's AVX2 six-row `candidate_wdot` kernel is where a vector variant
///   can be added only after verifying its interaction with the refresh chain).
///   Determinant candidate rows enter the carried geminal, so they take
///   the same fused `ao_eval` value convention as its seed and hop columns (fp64
///   body, completed AO value rounded to fp32 before any MO projection). A no-op
///   in full precision. J3 candidate rows are NOT routed through here.
fn round_det_candidate_aos(orb: &OrbData, aos: &mut Array2<f64>) {
    if orb.precision_policy.is_f32("ao_eval") {
        crate::determinant::round_to_ao_eval(
            aos.as_slice_mut().expect("candidate AO batch contiguous"),
        );
    }
}

fn det_candidate_batch(
    orb: &OrbData,
    class: CandidateBasisClass,
    space: CandidateSpace,
    positions: &[[f64; 3]],
    event: CandidateEvent,
) -> Array2<f64> {
    note_candidate_event(class, event);
    // The MO projection is applied iff the CONSUMER is the MO-space `det_wdot`
    // contraction and this leaf is MO-wrapped. A raw-AO leaf cannot reach a
    // projection kernel; the routing decision is made here, so the molecular
    // batch holds no unconditional `expect`.
    let mo = match space {
        CandidateSpace::Mo => orb.mo_coefficients.as_ref(),
        CandidateSpace::RawAo => None,
    };
    match (class, mo) {
        (CandidateBasisClass::MolecularGto, Some(c)) => {
            molecular_mos_candidate_batch(&orb.aos, c, orb.mo_coefficients_t.as_ref(), positions)
        }
        (CandidateBasisClass::MolecularGto, None) => {
            let mut aos = molecular_aos_candidate_batch(&orb.aos, positions);
            round_det_candidate_aos(orb, &mut aos);
            aos
        }
        (CandidateBasisClass::Unsupported, _) => {
            unreachable!("unsupported det class must take the slow path")
        }
        (class, None) => {
            let mut aos = basis_ao_candidate_batch(orb, class, positions);
            round_det_candidate_aos(orb, &mut aos);
            aos
        }
        (class, Some(c)) => {
            let mut aos = crate::timed!(
                "cand.basis_ao",
                basis_ao_candidate_batch(orb, class, positions)
            );
            round_det_candidate_aos(orb, &mut aos);
            // Release-mode shape contract: a composite whose MO columns do not span the
            // concatenated AO stack must fail loudly, not project
            // garbage.
            assert_eq!(
                aos.ncols(),
                c.ncols(),
                "candidate AO batch width must match the MO coefficient columns"
            );
            crate::timed!(
                "cand.mo_proj",
                crate::determinant::project_candidates_to_mos(&aos.view(), &c.view())
            )
        }
    }
}

/// TurboRVB's `upnewwf` evaluates the moved electron's lattice and nonlocal
/// projector positions in one orbital batch.  Reproduce that data flow for
/// the molecular determinant cache: one AO exponential stream and one MO
/// coefficient traversal produce both the six kinetic rows and the ECP rows.
/// Other basis families retain their established candidate kernels.
#[allow(clippy::too_many_arguments)]
fn combined_molecular_det_refresh(
    gem: &GeminalData,
    is_up: bool,
    electron: usize,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    ecp_positions: &[[f64; 3]],
    class: CandidateBasisClass,
    space: CandidateSpace,
) -> Option<(Array2<f64>, Array2<f64>)> {
    if class != CandidateBasisClass::MolecularGto || space != CandidateSpace::Mo {
        return None;
    }
    let shifts = lattice_shifts(alat, rt);
    let mut positions = Vec::with_capacity(6 + ecp_positions.len());
    for shift in &shifts {
        let (x, y, z) = if is_up {
            (
                r_up[[electron, 0]],
                r_up[[electron, 1]],
                r_up[[electron, 2]],
            )
        } else {
            (
                r_dn[[electron, 0]],
                r_dn[[electron, 1]],
                r_dn[[electron, 2]],
            )
        };
        positions.push([x + shift[0], y + shift[1], z + shift[2]]);
    }
    positions.extend_from_slice(ecp_positions);
    let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
    let all = crate::timed!(
        "gfmc.det_candidate_combined",
        det_candidate_batch(orb, class, space, &positions, CandidateEvent::Refresh)
    );
    let kinetic = all.slice(numpy::ndarray::s![..6, ..]).to_owned();
    let ecp = all.slice(numpy::ndarray::s![6.., ..]).to_owned();
    Some((kinetic, ecp))
}

/// Raw-AO J3 candidate table: molecular legacy batch or the periodic folded
/// batch (convention (a) above; positions pre-wrapped by the caller).
fn j3_candidate_batch(
    orb: &OrbData,
    class: CandidateBasisClass,
    positions: &[[f64; 3]],
    event: CandidateEvent,
) -> Array2<f64> {
    note_candidate_event(class, event);
    match class {
        CandidateBasisClass::MolecularGto => molecular_aos_candidate_batch(&orb.aos, positions),
        CandidateBasisClass::PeriodicPgto => {
            crate::atomic_orbital::compute_aos_candidate_batch_folded(&orb.aos, positions)
        }
        _ => unreachable!("J3 candidate route is molecular or periodic pGTO raw-AO only"),
    }
}

/// Evaluate a spin-concatenated J3 candidate table only when the table is
/// actually being built.  The fixed-mesh hot path normally refreshes one
/// moved-electron group, so materializing `up || dn` on every hop is pure
/// allocator and memory-bandwidth traffic (46 KiB/hop for 06water).
fn j3_candidate_batch_two_spins(
    orb: &OrbData,
    class: CandidateBasisClass,
    up: &[[f64; 3]],
    dn: &[[f64; 3]],
    event: CandidateEvent,
) -> Array2<f64> {
    let mut positions = Vec::with_capacity(up.len() + dn.len());
    positions.extend_from_slice(up);
    positions.extend_from_slice(dn);
    j3_candidate_batch(orb, class, &positions, event)
}

/// Molecular spherical/Cartesian AO values for a candidate batch, row-major
/// `(npoint, nao)`. Primitive exponent arguments from every shell/point pair
/// are gathered so `exp_pd_into` evaluates them as one compact AVX stream. The
/// radial contraction is shared by every angular component in a shell; only
/// the final angular replay differs between Cart and Sphe.
#[derive(Default)]
struct CandidateScratch {
    /// Transposed AO batch `(num_ao, nvp)` row-major, zero where screened;
    /// `nvp` = point count padded to [`MO_POINT_BLOCK`].
    aot: Vec<f64>,
    /// Compact primitive exponent arguments / exponentials of the shells kept
    /// at some point (`nprim * nv` per shell), and each shell's offset into
    /// them (`usize::MAX` = screened at every point).
    args: Vec<f64>,
    exps: Vec<f64>,
    shell_off: Vec<usize>,
    /// Structure-of-arrays candidate coordinates.
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    /// Per-point displacement / squared distance / radial sum of one shell.
    dx: Vec<f64>,
    dy: Vec<f64>,
    dz: Vec<f64>,
    r2: Vec<f64>,
    radial: Vec<f64>,
    /// Cartesian monomials `[x, x^2, x^3, x^4, y, .., z^4]` of one shell (12
    /// rows of `nv`) followed by one row of ones (`x^0`).
    pw: Vec<f64>,
    /// Per-point FAST value-screen survival of one shell.
    keep: Vec<bool>,
    radial_scale: Vec<f64>,
    /// Per home AO column: belongs to a shell kept at some point of the batch.
    col_kept: Vec<bool>,
    /// One gathered AO row (dense fallback / leftover-MO replay).
    row: Vec<f64>,
}

/// Point block of the lane-tiled MO projection (`acc` = `MO_POINT_BLOCK` x 8
/// MOs, twelve 4-wide registers).
const MO_POINT_BLOCK: usize = 6;

thread_local! {
    static CANDIDATE_SCRATCH: std::cell::RefCell<CandidateScratch> =
        std::cell::RefCell::new(CandidateScratch::default());
}

// SLP-vectorization rule (verified by objdump on Sapphire Rapids): a reduction
// kernel auto-vectorizes to clean full-width FMA chains ONLY as a single
// chunks_exact loop over ONE flat accumulator array, folded by a plain
// sequential sum, with a plain scalar tail. Adding a narrower sub-tail loop
// that updates part of the same accumulator, a loop-carried index across the
// tails, or a pairwise fold makes LLVM's SLP emit 2-wide xmm FMAs behind
// full-width loads + vpermpd (~20x slower). Keep these bodies boring.
#[inline]
fn dot_product_fast(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    // 16 lanes -> 2 independent zmm chains (AVX-512) / 4 ymm chains (AVX2);
    // a single 8-lane chain is FMA-latency-bound at ~half throughput.
    let n16 = x.len() & !15;
    let mut a = [0.0_f64; 16];
    for (xc, yc) in x[..n16].chunks_exact(16).zip(y[..n16].chunks_exact(16)) {
        for l in 0..16 {
            a[l] = xc[l].mul_add(yc[l], a[l]);
        }
    }
    let mut s = 0.0;
    for a_l in &a {
        s += *a_l;
    }
    for i in n16..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// Contract a row-major candidate block with one pre-folded all-electron
/// determinant/J3 weight row. Determinant candidates use MO space while J3
/// candidates use its native raw-AO basis.
#[inline]
fn candidate_wdot_into(aos: &ArrayView2<f64>, weight: &[f64], base: f64, out: &mut [f64]) {
    debug_assert_eq!(aos.ncols(), weight.len());
    debug_assert_eq!(aos.nrows(), out.len());
    for (s, value) in out.iter_mut().enumerate() {
        let ao = aos.row(s);
        let ao = ao.as_slice().expect("candidate AO row contiguous");
        *value = base + dot_product_fast(weight, ao);
    }
}

/// Sparse-column sibling for a cached candidate group after molecular
/// LEPSBAS screening. `active` is ascending, so the retained multiply/add
/// order is the same as the scalar dense traversal with exact-zero columns
/// removed.
#[inline]
fn candidate_wdot_active_into(
    aos: &ArrayView2<f64>,
    weight: &[f64],
    active: &[usize],
    base: f64,
    out: &mut [f64],
) {
    debug_assert_eq!(aos.ncols(), weight.len());
    debug_assert_eq!(aos.nrows(), out.len());
    for (s, value) in out.iter_mut().enumerate() {
        let mut sum = base;
        for &a in active {
            sum += weight[a] * aos[[s, a]];
        }
        *value = sum;
    }
}

fn candidate_active_columns(aos: &ArrayView2<f64>, rows_per_group: usize) -> Vec<Vec<usize>> {
    debug_assert!(rows_per_group > 0);
    debug_assert_eq!(aos.nrows() % rows_per_group, 0);
    (0..aos.nrows() / rows_per_group)
        .map(|g| {
            let first = g * rows_per_group;
            (0..aos.ncols())
                .filter(|&a| (first..first + rows_per_group).any(|r| aos[[r, a]] != 0.0))
                .collect()
        })
        .collect()
}

fn pack_candidate_group(
    aos: &ArrayView2<f64>,
    first_row: usize,
    rows: usize,
    columns: Vec<usize>,
) -> PackedCandidateGroup {
    let mut values = Vec::with_capacity(rows * columns.len());
    for row in first_row..first_row + rows {
        for &column in &columns {
            values.push(aos[[row, column]]);
        }
    }
    PackedCandidateGroup {
        columns,
        values,
        rows,
    }
}

fn pack_candidate_groups(
    aos: &ArrayView2<f64>,
    rows_per_group: usize,
    active: Vec<Vec<usize>>,
) -> Vec<PackedCandidateGroup> {
    active
        .into_iter()
        .enumerate()
        .map(|(group, columns)| {
            pack_candidate_group(aos, group * rows_per_group, rows_per_group, columns)
        })
        .collect()
}

#[inline]
fn packed_candidate_wdot_into(
    group: &PackedCandidateGroup,
    full_weight: &[f64],
    base: f64,
    packed_weight: &mut Vec<f64>,
    out: &mut [f64],
) {
    debug_assert_eq!(group.rows, out.len());
    packed_weight.clear();
    packed_weight.extend(group.columns.iter().map(|&a| full_weight[a]));
    let width = group.columns.len();
    for (row, value) in out.iter_mut().enumerate() {
        let first = row * width;
        *value = base + dot_product_fast(&packed_weight[..], &group.values[first..first + width]);
    }
}

fn report_candidate_active_once(
    label: &str,
    active: &[Vec<usize>],
    total_columns: usize,
    once: &std::sync::atomic::AtomicBool,
) {
    use std::sync::atomic::Ordering;
    if std::env::var_os("JQMC_RS_DIAG_ACTIVE").is_none()
        || once.swap(true, Ordering::Relaxed)
        || active.is_empty()
    {
        return;
    }
    let min = active.iter().map(Vec::len).min().unwrap_or(0);
    let max = active.iter().map(Vec::len).max().unwrap_or(0);
    let sum: usize = active.iter().map(Vec::len).sum();
    eprintln!(
        "JQMC_RS_DIAG_ACTIVE {label}: groups={} columns={} active_min={} active_mean={:.3} active_max={}",
        active.len(),
        total_columns,
        min,
        sum as f64 / active.len() as f64,
        max,
    );
}

static KIN_J3_ACTIVE_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static ECP_J3_ACTIVE_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Build the persistent J3 log-ratio table for an all-electron candidate AO
/// matrix. Rows are grouped by electron, with all up-spin groups followed by
/// all down-spin groups. This initialization is paid once per fixed mesh; the
/// hop path below advances it with the common `J3*delta_AO` vector.
fn fill_j3_candidate_deltas(
    jm: &ArrayView2<f64>,
    state: &crate::jastrow_factor::J3RatioState,
    candidate_aos: &ArrayView2<f64>,
    n_up: usize,
    rows_per_electron: usize,
) -> Vec<f64> {
    let n_orb = jm.nrows();
    debug_assert_eq!(candidate_aos.ncols(), n_orb);
    debug_assert_eq!(candidate_aos.nrows() % rows_per_electron, 0);
    let n_e = candidate_aos.nrows() / rows_per_electron;
    let mut out = vec![0.0_f64; candidate_aos.nrows()];
    let mut q = vec![0.0_f64; n_orb];
    for global_e in 0..n_e {
        let (is_up, e) = if global_e < n_up {
            (true, global_e)
        } else {
            (false, global_e - n_up)
        };
        let (aos_old, w_same, cross_same, cross_other) = if is_up {
            (&state.aos_up, &state.w_up, &state.cross_up, &state.cross_dn)
        } else {
            (&state.aos_dn, &state.w_dn, &state.cross_dn, &state.cross_up)
        };
        let mut base = 0.0_f64;
        for a in 0..n_orb {
            let qa = jm[[a, n_orb]] + cross_other[a] + cross_same[a] - w_same[[a, e]];
            q[a] = qa;
            base -= aos_old[[a, e]] * qa;
        }
        let first = global_e * rows_per_electron;
        let rows = candidate_aos.slice(numpy::ndarray::s![first..first + rows_per_electron, ..]);
        candidate_wdot_into(&rows, &q, base, &mut out[first..first + rows_per_electron]);
    }
    out
}

/// Live-electron mask for [`live_rows_batch`]: row subsetting of the ECP
/// table builds is restricted to the molecular GTO kernels (row-independent by
/// construction); other classes build every row and only skip the ratio work.
fn mol_live(class: CandidateBasisClass, live: &[bool]) -> Option<&[bool]> {
    (class == CandidateBasisClass::MolecularGto).then_some(live)
}

/// Candidate rows of the LIVE electrons only, scattered into a full-height
/// table; rows of dead electrons stay exact zeros and are never read
/// (IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001). `live = None` evaluates every row.
/// Exact because every candidate-row kernel it is used with evaluates each
/// row independently of the other rows of its batch.
fn live_rows_batch(
    positions: &[[f64; 3]],
    rows_per_electron: usize,
    live: Option<&[bool]>,
    eval: impl Fn(&[[f64; 3]]) -> Array2<f64>,
) -> Array2<f64> {
    let live = match live {
        Some(l) if !l.iter().all(|&b| b) => l,
        _ => return eval(positions),
    };
    debug_assert_eq!(live.len() * rows_per_electron, positions.len());
    let mut sub = Vec::with_capacity(positions.len());
    for (e, &l) in live.iter().enumerate() {
        if l {
            sub.extend_from_slice(&positions[e * rows_per_electron..(e + 1) * rows_per_electron]);
        }
    }
    let part = eval(&sub);
    let mut full = Array2::<f64>::zeros((positions.len(), part.ncols()));
    let mut r = 0;
    for (e, &l) in live.iter().enumerate() {
        if l {
            full.slice_mut(numpy::ndarray::s![
                e * rows_per_electron..(e + 1) * rows_per_electron,
                ..
            ])
            .assign(&part.slice(numpy::ndarray::s![r..r + rows_per_electron, ..]));
            r += rows_per_electron;
        }
    }
    full
}

/// Exact accepted-hop advance of a persistent J3 candidate log-ratio table.
/// For every unchanged electron `i`, q_i gains the same vector
/// `d = J3*(AO_new-AO_old)`, so all candidate updates are one contiguous
/// matrix-vector product. Only the moved electron's newly centered candidate
/// group is rebuilt from the post-hop state.
#[allow(clippy::too_many_arguments)]
fn advance_j3_candidate_deltas(
    jm: &ArrayView2<f64>,
    state: &crate::jastrow_factor::J3RatioState,
    n_up: usize,
    rows_per_electron: usize,
    moved_is_up: bool,
    moved_e: usize,
    d: &[f64],
    current_up: &[f64],
    current_dn: &[f64],
    packed: &[PackedCandidateGroup],
    table: &mut [f64],
    live: Option<&[bool]>,
) {
    let n_orb = jm.nrows();
    debug_assert_eq!(d.len(), n_orb);
    // Rows of dead electrons (`live[global_e] == false`, ECP zero-amplitude
    // groups) are never read; they are rebuilt when their electron moves.
    let is_live = |global_e: usize| live.is_none_or(|l| l[global_e]);
    let n_e = table.len() / rows_per_electron;
    debug_assert_eq!(packed.len(), n_e);
    let moved_global = if moved_is_up { moved_e } else { n_up + moved_e };

    let n_dn = n_e - n_up;
    debug_assert_eq!(current_up.len(), n_up);
    debug_assert_eq!(current_dn.len(), n_dn);

    // Candidate rows are physically packed to the exact nonzero columns left
    // by LEPSBAS. In water20 this reduces both kinetic and ECP rows from 340
    // columns to about 72 while preserving ascending accumulation order.
    let mut packed_weight = Vec::<f64>::new();
    let mut dots = vec![0.0_f64; rows_per_electron];
    for global_e in 0..n_e {
        if global_e == moved_global || !is_live(global_e) {
            continue;
        }
        packed_candidate_wdot_into(&packed[global_e], d, 0.0, &mut packed_weight, &mut dots);
        let first = global_e * rows_per_electron;
        let current = if global_e < n_up {
            current_up[global_e]
        } else {
            current_dn[global_e - n_up]
        };
        for local_row in 0..rows_per_electron {
            table[first + local_row] += dots[local_row] - current;
        }
    }

    if !is_live(moved_global) {
        return;
    }
    // q for the moved electron is invariant under its own hop because the
    // cross-row increment and W[:,moved] increment cancel exactly.
    let (aos_old, w_same, cross_same, cross_other) = if moved_is_up {
        (&state.aos_up, &state.w_up, &state.cross_up, &state.cross_dn)
    } else {
        (&state.aos_dn, &state.w_dn, &state.cross_dn, &state.cross_up)
    };
    let mut q = vec![0.0_f64; n_orb];
    let mut base = 0.0_f64;
    for a in 0..n_orb {
        let qa = jm[[a, n_orb]] + cross_other[a] + cross_same[a] - w_same[[a, moved_e]];
        q[a] = qa;
        base -= aos_old[[a, moved_e]] * qa;
    }
    let first = moved_global * rows_per_electron;
    packed_candidate_wdot_into(
        &packed[moved_global],
        &q,
        base,
        &mut packed_weight,
        &mut table[first..first + rows_per_electron],
    );
}

pub(crate) fn molecular_aos_candidate_batch(
    data: &crate::atomic_orbital::AosGtoData,
    positions: &[[f64; 3]],
) -> Array2<f64> {
    CANDIDATE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        molecular_aos_candidate_batch_into(data, positions, &mut scratch)
    })
}

/// Row-major `(npoint, nao)` molecular candidate AO batch on caller-owned
/// scratch: [`molecular_aos_candidate_batch_t`] followed by a transpose of the
/// kept columns (screened columns stay exact zeros).
fn molecular_aos_candidate_batch_into(
    data: &crate::atomic_orbital::AosGtoData,
    positions: &[[f64; 3]],
    scratch: &mut CandidateScratch,
) -> Array2<f64> {
    let nv = positions.len();
    let na = data.num_ao;
    let nvp = molecular_aos_candidate_batch_t(data, positions, scratch);
    let mut out = Array2::<f64>::zeros((nv, na));
    let o = out.as_slice_mut().expect("candidate AO batch contiguous");
    for a in 0..na {
        if !scratch.col_kept[a] {
            continue;
        }
        let src = &scratch.aot[a * nvp..a * nvp + nv];
        for (v, &x) in src.iter().enumerate() {
            o[v * na + a] = x;
        }
    }
    out
}

/// Molecular spherical/Cartesian candidate AO values, TRANSPOSED: fills
/// `scratch.aot` as `(nao, nvp)` row-major (screened entries and the padded
/// points are exact zeros), marks `scratch.col_kept`, returns `nvp`.
///
/// Exactness (IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001): per shell the geometry,
/// the `pz_val*r2 > LEPSBAS` value screen, the primitive exponentials
/// (`exp_neg(-zeta*r2)`, the elementwise core of `exp_pd_into`), the
/// sequential radial sum, the Cartesian monomials `x, x*x, (x*x)*x, ..`, the
/// angular product `(xp*yp)*zp` and the value `(radial*scale)*angular` are the
/// per-(shell, point) expressions, evaluated elementwise across all
/// points of the batch (no reassociation, no FMA contraction), and screened
/// pairs leave exact zeros. Exponentials of fully screened shells are not
/// evaluated. Bit-identical to an all-pairs kernel (the test reference below).
fn molecular_aos_candidate_batch_t(
    data: &crate::atomic_orbital::AosGtoData,
    positions: &[[f64; 3]],
    scratch: &mut CandidateScratch,
) -> usize {
    debug_assert_eq!(data.n_image, 1);
    debug_assert!(!data.shells.is_empty());
    let nv = positions.len();
    let na = data.num_ao;
    let nvp = nv.div_ceil(MO_POINT_BLOCK) * MO_POINT_BLOCK;
    let CandidateScratch {
        aot,
        args,
        exps,
        shell_off,
        px,
        py,
        pz,
        dx,
        dy,
        dz,
        r2,
        radial,
        pw,
        keep,
        radial_scale,
        col_kept,
        ..
    } = scratch;
    aot.clear();
    aot.resize(na * nvp, 0.0);
    col_kept.clear();
    col_kept.resize(na, false);
    for buf in [
        &mut *px,
        &mut *py,
        &mut *pz,
        &mut *dx,
        &mut *dy,
        &mut *dz,
        &mut *r2,
        &mut *radial,
    ] {
        buf.clear();
        buf.resize(nv, 0.0);
    }
    keep.clear();
    keep.resize(nv, false);
    for (v, p) in positions.iter().enumerate() {
        px[v] = p[0];
        py[v] = p[1];
        pz[v] = p[2];
    }
    let (px, py, pz) = (&px[..nv], &py[..nv], &pz[..nv]);
    let (dx, dy, dz, r2, radial, keep) = (
        &mut dx[..nv],
        &mut dy[..nv],
        &mut dz[..nv],
        &mut r2[..nv],
        &mut radial[..nv],
        &mut keep[..nv],
    );
    // One shell's displacement, squared distance and FAST value screen
    // (`Shell::pz_val`: every dropped AO value is below
    // FAST_SCREEN_ABS_BOUND = 1e-12 in absolute AO units, issue #100); the
    // same expressions as a per-(shell, point) kernel.
    let geometry = |shell: &crate::atomic_orbital::Shell,
                    dx: &mut [f64],
                    dy: &mut [f64],
                    dz: &mut [f64],
                    r2: &mut [f64],
                    keep: &mut [bool]|
     -> bool {
        let (cx, cy, cz) = (shell.center[0], shell.center[1], shell.center[2]);
        let pz_val = shell.pz_val;
        let mut any = false;
        for v in 0..nv {
            let x = px[v] - cx;
            let y = py[v] - cy;
            let z = pz[v] - cz;
            let rr = x * x + y * y + z * z;
            dx[v] = x;
            dy[v] = y;
            dz[v] = z;
            r2[v] = rr;
            let k = (pz_val * rr).partial_cmp(&crate::atomic_orbital::LEPSBAS)
                != Some(std::cmp::Ordering::Greater);
            keep[v] = k;
            any |= k;
        }
        any
    };
    // Pass 1: exponent arguments of every shell kept at some point, one
    // compact stream for the vectorised `exp_pd_into`.
    args.clear();
    shell_off.clear();
    for shell in &data.shells {
        if geometry(shell, &mut *dx, &mut *dy, &mut *dz, &mut *r2, &mut *keep) {
            shell_off.push(args.len());
            for &zeta in &shell.prim_z {
                args.extend(r2.iter().map(|&rr| -zeta * rr));
            }
        } else {
            shell_off.push(usize::MAX);
        }
    }
    exps.clear();
    exps.resize(args.len(), 0.0);
    crate::atomic_orbital::exp_pd_into(args.as_slice(), exps.as_mut_slice());
    let is_cart = data.kind == crate::atomic_orbital::AoKind::Cart;
    if is_cart {
        // Cartesian components in one shell have identical primitive
        // exponents and contraction coefficients. Their normalization
        // differs by one primitive-independent component scale.
        radial_scale.clear();
        radial_scale.resize(data.num_ao, 0.0);
        for shell in &data.shells {
            let qref = shell.prim_nc.iter().position(|&x| x != 0.0);
            if let Some(q) = qref {
                let denom = shell.prim_nc[q];
                for &ao in &shell.aos {
                    radial_scale[ao] = data.flat_nc[data.ao_prim_offset[ao] + q] / denom;
                }
            }
        }
        pw.clear();
        pw.resize(13 * nv, 0.0);
        pw[12 * nv..].fill(1.0);
    }
    // Pass 2: radial sums and angular replay of the kept shells.
    for (s, shell) in data.shells.iter().enumerate() {
        let off = shell_off[s];
        if off == usize::MAX {
            continue;
        }
        geometry(shell, &mut *dx, &mut *dy, &mut *dz, &mut *r2, &mut *keep);
        radial.fill(0.0);
        for (q, &nc) in shell.prim_nc.iter().enumerate() {
            let e = &exps[off + q * nv..off + (q + 1) * nv];
            for v in 0..nv {
                radial[v] += nc * e[v];
            }
        }
        if is_cart {
            {
                let (xs, rest) = pw.split_at_mut(4 * nv);
                let (ys, rest) = rest.split_at_mut(4 * nv);
                let zs = &mut rest[..4 * nv];
                for (p, d) in [(xs, &*dx), (ys, &*dy), (zs, &*dz)] {
                    let (p1, rest) = p.split_at_mut(nv);
                    let (p2, rest) = rest.split_at_mut(nv);
                    let (p3, p4) = rest.split_at_mut(nv);
                    for v in 0..nv {
                        let x = d[v];
                        let x2 = x * x;
                        let x3 = x2 * x;
                        p1[v] = x;
                        p2[v] = x2;
                        p3[v] = x3;
                        p4[v] = x3 * x;
                    }
                }
            }
            // Row of x^k (k = 0 -> the ones row) inside `pw`.
            let pow_row = |axis: usize, k: usize| -> std::ops::Range<usize> {
                if k == 0 {
                    12 * nv..13 * nv
                } else {
                    (4 * axis + k - 1) * nv..(4 * axis + k) * nv
                }
            };
            for &ao in &shell.aos {
                let nx = data.poly_x[ao] as usize;
                let ny = data.poly_y[ao] as usize;
                let nz = data.poly_z[ao] as usize;
                let col = data.kernel_to_home[ao];
                col_kept[col] = true;
                let scale = radial_scale[ao];
                let dst = &mut aot[col * nvp..col * nvp + nv];
                if nx < 5 && ny < 5 && nz < 5 {
                    let xp = &pw[pow_row(0, nx)];
                    let yp = &pw[pow_row(1, ny)];
                    let zp = &pw[pow_row(2, nz)];
                    for v in 0..nv {
                        let angular = xp[v] * yp[v] * zp[v];
                        let val = (radial[v] * scale) * angular;
                        dst[v] = if keep[v] { val } else { 0.0 };
                    }
                } else {
                    for v in 0..nv {
                        if keep[v] {
                            let angular = dx[v].powi(nx as i32)
                                * dy[v].powi(ny as i32)
                                * dz[v].powi(nz as i32);
                            dst[v] = (radial[v] * scale) * angular;
                        }
                    }
                }
            }
        } else {
            // The normalized radial contraction is common to all 2l+1
            // spherical components; the flat angular tables are the
            // value-only replay of the single-position Sphe kernel.
            for &ao in &shell.aos {
                let col = data.kernel_to_home[ao];
                col_kept[col] = true;
                let dst = &mut aot[col * nvp..col * nvp + nv];
                for v in 0..nv {
                    if keep[v] {
                        dst[v] = radial[v]
                            * crate::atomic_orbital::eval_angular_flat(
                                data, ao, dx[v], dy[v], dz[v], r2[v],
                            );
                    }
                }
            }
        }
    }
    nvp
}

/// Projection-scoped determinant candidate table in MO space. The expensive
/// raw-AO -> MO projection is paid only for the initial fixed mesh and the six
/// rows of the electron moved by the previous hop. Subsequent candidate
/// contractions therefore have occupied-orbital width instead of raw-AO width.
///
/// The coefficients are a PARAMETER, not an `Option` read off the orbital: a
/// leaf without them has no MO space to project into and must never reach this
/// kernel (see [`det_candidate_space`]).
///
/// `coeff_t` is the orbital's transposed coefficient copy
/// (`OrbData::mo_coefficients_t`, `(num_ao, num_mo)` row-major). When present
/// the projection runs the lane-tiled kernel [`lane_tiled_mo_projection_into`];
/// otherwise the dense per-(point, MO) [`dot_product_fast`] loop. Both give
/// `out[s, m] = dot_product_fast(ao[s, :], C[m, :])` bit for bit.
fn molecular_mos_candidate_batch(
    orb_aos: &crate::atomic_orbital::AosGtoData,
    coeff: &Array2<f64>,
    coeff_t: Option<&Array2<f64>>,
    positions: &[[f64; 3]],
) -> Array2<f64> {
    CANDIDATE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let scratch = &mut *scratch;
        let nv = positions.len();
        let na = orb_aos.num_ao;
        let nm = coeff.nrows();
        debug_assert_eq!(na, coeff.ncols());
        let nvp = crate::timed!(
            "gfmc.det_candidate_ao",
            molecular_aos_candidate_batch_t(orb_aos, positions, scratch)
        );
        // Fused `ao_eval` value convention of the carried geminal.
        if orb_aos.precision_policy.is_f32("ao_eval") {
            crate::determinant::round_to_ao_eval(&mut scratch.aot);
        }
        let mut out = Array2::<f64>::zeros((nv, nm));
        let ct = coeff_t.and_then(|ct| {
            (ct.nrows() == na && ct.ncols() == nm)
                .then(|| ct.as_slice())
                .flatten()
        });
        crate::timed!("gfmc.det_candidate_mo", {
            let o = out.as_slice_mut().expect("candidate MO batch contiguous");
            match ct {
                Some(ct) => {
                    bump_route_counter(&GFMC_CAND_MO_TILED);
                    lane_tiled_mo_projection_into(scratch, nv, nvp, na, ct, coeff, o)
                }
                None => {
                    bump_route_counter(&GFMC_CAND_MO_DENSE);
                    for s in 0..nv {
                        gather_candidate_row(scratch, nvp, na, s);
                        for m in 0..nm {
                            let c = coeff.row(m);
                            let c = c.as_slice().expect("MO coefficient row contiguous");
                            o[s * nm + m] = dot_product_fast(&scratch.row, c);
                        }
                    }
                }
            }
        });
        out
    })
}

/// Gather point `s` of the transposed AO batch into `scratch.row` (length
/// `na`, row-major order).
fn gather_candidate_row(scratch: &mut CandidateScratch, nvp: usize, na: usize, s: usize) {
    let CandidateScratch { aot, row, .. } = scratch;
    row.clear();
    row.extend((0..na).map(|a| aot[a * nvp + s]));
}

/// Lane-tiled replay of the dense candidate MO projection
/// `out[s, m] = dot_product_fast(ao[s, :], C[m, :])`
/// (IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001).
///
/// `dot_product_fast` keeps 16 lane accumulators: column `a < n16`
/// (`n16 = na & !15`) is fused-multiply-added into lane `a % 16` in ascending
/// `a`; the lanes are folded into `0.0` in lane order; the scalar tail columns
/// `a >= n16` are then added with a separately rounded product in ascending
/// `a`. Each lane is an independent FMA chain over `k = a / 16`, i.e. one
/// small GEMM per lane with an in-order `k` loop, so this kernel computes lane
/// `l` for a `MO_POINT_BLOCK` x 8 tile of (point, MO) pairs in registers and
/// folds it into `out` in ascending `l` -- the same operations on every
/// (point, MO) pair, vectorised across MOs and blocked over points. Columns of
/// shells screened at every point of the batch hold exact zeros and are
/// skipped (`fma(0, c, acc) == acc` for finite `c`; only an all-underflow
/// `-0.0` accumulator could change sign). MOs beyond the last full group of 8
/// run [`dot_product_fast`] itself on the gathered row.
fn lane_tiled_mo_projection_into(
    scratch: &mut CandidateScratch,
    nv: usize,
    nvp: usize,
    na: usize,
    ct: &[f64],
    coeff: &Array2<f64>,
    out: &mut [f64],
) {
    const MB: usize = 8;
    const PB: usize = MO_POINT_BLOCK;
    let nm = coeff.nrows();
    debug_assert_eq!(ct.len(), na * nm);
    debug_assert_eq!(out.len(), nv * nm);
    debug_assert_eq!(nvp % PB, 0);
    let n16 = na & !15;
    let nm8 = nm & !(MB - 1);
    out.fill(0.0);
    {
        let aot = &scratch.aot;
        let col_kept = &scratch.col_kept;
        for l in 0..16.min(n16) {
            for m0 in (0..nm8).step_by(MB) {
                for s0 in (0..nvp).step_by(PB) {
                    let mut acc = [[0.0_f64; MB]; PB];
                    let mut a = l;
                    while a < n16 {
                        if col_kept[a] {
                            let crow: &[f64; MB] = ct[a * nm + m0..a * nm + m0 + MB]
                                .try_into()
                                .expect("MO coefficient tile");
                            let arow: &[f64; PB] = aot[a * nvp + s0..a * nvp + s0 + PB]
                                .try_into()
                                .expect("AO point tile");
                            for p in 0..PB {
                                let b = arow[p];
                                for j in 0..MB {
                                    acc[p][j] = b.mul_add(crow[j], acc[p][j]);
                                }
                            }
                        }
                        a += 16;
                    }
                    for p in 0..PB.min(nv.saturating_sub(s0)) {
                        let dst = &mut out[(s0 + p) * nm + m0..(s0 + p) * nm + m0 + MB];
                        for j in 0..MB {
                            dst[j] += acc[p][j];
                        }
                    }
                }
            }
        }
        // Tail columns, after the full lane fold, in ascending order.
        for a in n16..na {
            if !col_kept[a] {
                continue;
            }
            let crow = &ct[a * nm..a * nm + nm8];
            for s in 0..nv {
                let b = aot[a * nvp + s];
                let dst = &mut out[s * nm..s * nm + nm8];
                for (o, &c) in dst.iter_mut().zip(crow) {
                    *o += b * c;
                }
            }
        }
    }
    if nm8 < nm {
        for s in 0..nv {
            gather_candidate_row(scratch, nvp, na, s);
            for m in nm8..nm {
                let c = coeff.row(m);
                let c = c.as_slice().expect("MO coefficient row contiguous");
                out[s * nm + m] = dot_product_fast(&scratch.row, c);
            }
        }
    }
}

impl MeshFastState {
    #[inline]
    pub(crate) fn ginv(&self) -> &Array2<f64> {
        &self.ginv
    }

    /// The streaming determinant caches the running inverse was advanced with.
    #[inline]
    pub(crate) fn det_state(&self) -> &crate::determinant::DetRatioState {
        &self.det_state
    }

    /// The AO screens of this state's measurement e_L, as the force's
    /// [`crate::gfmc_forces::FastScreen`]: on the molecular MO route the
    /// continuum kinetic takes the FAST derivative-screened tables
    /// (`init_det_grad_lap_cache`, `init_j3_grad_lap_cache`), the ECP mesh AOs
    /// the per-shell value screen of the molecular candidate batch, and the
    /// ECP quadrature the radial cut `ecp_radial_cut`.
    pub(crate) fn force_screen(
        &self,
        gem: &GeminalData,
        j3: Option<&OrbData>,
    ) -> crate::gfmc_forces::FastScreen {
        // The force's screened kernels are the molecular Cartesian ones.
        let molecular = self.cell.is_none();
        let cart = |o: &OrbData| o.aos.kind == crate::atomic_orbital::AoKind::Cart;
        crate::gfmc_forces::FastScreen {
            det: molecular
                && fast_mol_kinetic_ok_cell(gem, false)
                && cart(&gem.orb_up)
                && cart(&gem.orb_dn),
            j3: molecular
                && self.j3_state.is_some()
                && j3.is_some_and(|o| {
                    cart(o) && j3_candidate_class(o, false) == CandidateBasisClass::MolecularGto
                }),
            ecp_radial: self.ecp_radial_cut > 0.0,
            mesh_shell_only: true,
        }
    }

    /// The carried determinant MO grad/Laplacian tables at the state's current
    /// positions (`None` before the first `det_grad_lap`). On the molecular MO
    /// route these are the screened kinetic tables of the measurement e_L.
    pub(crate) fn det_grad_cache(
        &self,
    ) -> std::cell::Ref<'_, Option<crate::determinant::DetGradLapCache>> {
        self.det_grad_cache.borrow()
    }

    #[inline]
    fn wrap(&self, pos: [f64; 3]) -> [f64; 3] {
        match &self.cell {
            Some((c, iv)) => crate::mcmc::wrap_into_cell(pos, &c.view(), iv),
            None => pos,
        }
    }

    fn ensure_det_wdot_cache(&self, gem: &GeminalData) {
        if self.det_wdot_cache.borrow().is_some() {
            return;
        }
        let ((up, up_base), (dn, dn_base)) = crate::timed!(
            "gfmc.det_wdot_build",
            (
                crate::coulomb_potential::det_mowdot_all_up(
                    gem,
                    &self.det_state,
                    &self.ginv.view(),
                ),
                crate::coulomb_potential::det_mowdot_all_dn(
                    gem,
                    &self.det_state,
                    &self.ginv.view(),
                ),
            )
        );
        *self.det_wdot_cache.borrow_mut() = Some(DetWdotCache {
            up,
            up_base,
            dn,
            dn_base,
        });
        bump_route_counter(&GFMC_WINV_BUILDS);
    }

    fn ensure_j3_wdot_cache(&self, j3: Option<(&OrbData, &ArrayView2<f64>)>) {
        if self.j3_wdot_cache.borrow().is_some() {
            return;
        }
        let (orb, jm) = match (j3, self.j3_state.as_ref()) {
            (Some(x), Some(st)) => (x.0, (x.1, st)),
            _ => return,
        };
        let ((up, up_base), (dn, dn_base)) = crate::timed!(
            "gfmc.j3_wdot_build",
            (
                crate::coulomb_potential::ecp_j3_wdot_all(orb, jm.0, jm.1, true),
                crate::coulomb_potential::ecp_j3_wdot_all(orb, jm.0, jm.1, false),
            )
        );
        *self.j3_wdot_cache.borrow_mut() = Some(J3WdotCache {
            up,
            up_base,
            dn,
            dn_base,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn ecp_local(
        &self,
        positions: &ArrayView2<f64>,
        nucleus_index: &[i64],
        exponents: &[f64],
        coefficients: &[f64],
        powers: &[f64],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    ) -> f64 {
        let eval = |r: &ArrayView2<f64>, e: usize| {
            crate::coulomb_potential::compute_ecp_local_one(
                positions,
                nucleus_index,
                exponents,
                coefficients,
                powers,
                [r[[e, 0]], r[[e, 1]], r[[e, 2]]],
                cell_nmax,
            )
        };
        let mut cache = self.ecp_local_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(EcpLocalCache {
                up: (0..r_up.nrows()).map(|e| eval(r_up, e)).collect(),
                dn: (0..r_dn.nrows()).map(|e| eval(r_dn, e)).collect(),
            });
            self.ecp_local_dirty.set(None);
        } else if let Some((is_up, e)) = self.ecp_local_dirty.take() {
            let cache = cache.as_mut().expect("local ECP cache initialized");
            if is_up {
                cache.up[e] = eval(r_up, e);
            } else {
                cache.dn[e] = eval(r_dn, e);
            }
        }
        let cache = cache.as_ref().expect("local ECP cache initialized");
        cache.up.iter().chain(cache.dn.iter()).copied().sum()
    }

    fn det_grad_lap(
        &self,
        gem: &GeminalData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        numpy::ndarray::Array1<f64>,
        numpy::ndarray::Array1<f64>,
    ) {
        // Named precondition at the gate boundary: callers MUST have checked
        // `fast_det_gradlap_ok`. The `else` arm below is the legacy molecular
        // route, whose kernels require MO coefficients; without this the
        // violation only surfaces as an opaque expect deep inside
        // `determinant.rs` (the exact failure mode the candidate-table route
        // had for raw-AO leaves).
        assert!(
            fast_det_gradlap_ok_cell(gem, self.cell.is_some()),
            "det_grad_lap requires the molecular MO or generic derivative route"
        );
        let mut cache = self.det_grad_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(if fast_generic_det_gradlap_cell(gem, self.cell.is_some()) {
                // Basis-generic derivative tables are seeded at the same
                // home-cell-wrapped coordinates as the determinant ratio state.
                // `crate::mcmc::wrap_rows` is the exact wrap that
                // `mesh_fast_state` used for `orb_matrix_via_single`.
                bump_route_counter(&GFMC_DET_GRAD_GENERIC_BUILDS);
                let (ruw, rdw) = match &self.cell {
                    Some((c, iv)) => (
                        crate::mcmc::wrap_rows(r_up, &c.view(), iv),
                        crate::mcmc::wrap_rows(r_dn, &c.view(), iv),
                    ),
                    None => (r_up.to_owned(), r_dn.to_owned()),
                };
                crate::determinant::init_det_grad_lap_cache_folded(gem, &ruw.view(), &rdw.view())
            } else {
                crate::determinant::init_det_grad_lap_cache(gem, r_up, r_dn)
            });
        }
        // The kinetic mesh immediately preceding this call builds the same
        // determinant contraction rows.  Share them with the continuum kinetic
        // term instead of repeating two small GEMMs for every hop.
        // `ensure_det_wdot_cache` builds the table whenever it is absent, so the
        // rows are always present here. The matrix-product contraction
        // (`contract_det_grad_lap_cache`, two GEMMs per call) is NOT a silent
        // fallback of this route; it stays callable only as the
        // from-scratch reference of `debug_det_grad_lap_chain`.
        self.ensure_det_wdot_cache(gem);
        let wdot = self.det_wdot_cache.borrow();
        let w = wdot
            .as_ref()
            .expect("ensure_det_wdot_cache always seeds the determinant contraction rows");
        crate::determinant::contract_det_grad_lap_cache_with_wdot(
            cache.as_ref().expect("det derivative cache initialized"),
            &w.up.view(),
            &w.dn.view(),
        )
    }

    fn j3_grad_lap(
        &self,
        orb: &OrbData,
        jm: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        numpy::ndarray::Array1<f64>,
        numpy::ndarray::Array1<f64>,
    ) {
        let mut cache = self.j3_grad_cache.borrow_mut();
        if cache.is_none() {
            // A periodic pGTO leaf evaluates a static AO image tiling
            // enumerated relative to the home cell
            // (`AosGtoData::wrap_cell`'s doc comment: an unwrapped coordinate
            // several lattice vectors outside "falls beyond the static image
            // tiling and the sum misses the dominant images -- grad/lap/kinetic
            // collapse"). `r_up`/`r_dn` here are the walker's RAW stored
            // positions (this method's own args), which are legitimately
            // outside the home cell under PBC. Wrap with the SAME
            // primitive `mesh_fast_state` used to build every other AO-eval
            // input (`ruw`/`rdw`), so the seed matches the advance's
            // `wrapped_newpos` convention.
            let (ruw, rdw) = match &self.cell {
                Some((c, iv)) => (
                    crate::mcmc::wrap_rows(r_up, &c.view(), iv),
                    crate::mcmc::wrap_rows(r_dn, &c.view(), iv),
                ),
                None => (r_up.to_owned(), r_dn.to_owned()),
            };
            *cache = Some(crate::jastrow_factor::init_j3_grad_lap_cache(
                orb,
                &ruw.view(),
                &rdw.view(),
            ));
        }
        self.ensure_j3_wdot_cache(Some((orb, jm)));
        let wdot = self.j3_wdot_cache.borrow();
        match wdot.as_ref() {
            Some(w) if orb.mo_coefficients.is_none() => {
                crate::jastrow_factor::contract_j3_grad_lap_cache_with_q(
                    cache.as_mut().expect("J3 derivative cache initialized"),
                    &w.up.view(),
                    &w.dn.view(),
                )
            }
            _ => crate::jastrow_factor::contract_j3_grad_lap_cache(
                jm,
                self.j3_state.as_ref().expect("J3 ratio state initialized"),
                cache.as_ref().expect("J3 derivative cache initialized"),
            ),
        }
    }

    fn j12_grad_lap(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        numpy::ndarray::Array1<f64>,
        numpy::ndarray::Array1<f64>,
    ) {
        let (gu, gd, lu, ld) = {
            let mut cache = self.j12_grad_cache.borrow_mut();
            if cache.is_none() {
                *cache = Some(crate::jastrow_factor::init_j12_grad_lap_cache_generic(
                    jastrow, r_up, r_dn,
                ));
            }
            crate::jastrow_factor::clone_j12_grad_lap_cache(
                cache.as_ref().expect("J2 derivative cache initialized"),
            )
        };
        // The cache holds the J1/J2 legs only. DTN is added here, in the parent
        // kernel's order (((J1)+J2)+DTN), because the incremental advance can
        // move only the J2 pair terms. Folding DTN into that seed would leave
        // it pinned to the interval-start configuration, so the DTN leg owns a
        // separate projection-scoped cache. The real and complex paths use the
        // same basis-agnostic, PBC-native
        // `dtn::{init,update,clone}_dtn_grad_lap_cache` operations.
        match &jastrow.dtn {
            Some(d) => {
                let (dgu, dgd, dlu, dld) = {
                    let mut cache = self.dtn_grad_cache.borrow_mut();
                    if cache.is_none() {
                        *cache = Some(crate::dtn::init_dtn_grad_lap_cache(d, r_up, r_dn));
                    }
                    crate::dtn::clone_dtn_grad_lap_cache(
                        cache.as_ref().expect("DTN derivative cache initialized"),
                    )
                };
                (
                    gu + dgu,
                    gd + dgd,
                    lu + numpy::ndarray::Array1::from_vec(dlu),
                    ld + numpy::ndarray::Array1::from_vec(dld),
                )
            }
            None => (gu, gd, lu, ld),
        }
    }

    /// Per-hop refresh rule (production; the debug full-recompute
    /// routes are untouched). After the hop's Sherman-Morrison update the
    /// carried `G^-1` is screened by the carried-inverse probe residual against the
    /// geminal the advanced streaming caches imply (O(N^2 + N M), no `G`
    /// formed). When it exceeds `SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`, the
    /// determinant base is rebuilt from scratch at the post-hop positions with
    /// the interval seed itself ([`seed_det_state_and_inverse`]: AO tables,
    /// `G`, `G^-1`), and the winv tables derived from the old inverse are
    /// dropped (rebuilt from the new `G^-1` by `ensure_det_wdot_cache` at the
    /// next consumer). A refreshed state therefore equals a fresh one. No
    /// other trigger.
    fn screen_interval_ginv(
        &mut self,
        gem: &GeminalData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) {
        let residual = crate::determinant::det_state_inverse_probe_residual(
            gem,
            &self.det_state,
            &self.ginv.view(),
        );
        let carry = crate::determinant::interval_ginv_carry_ok(residual);
        record_interval_ginv_carry(carry);
        if !carry {
            let (ruw, rdw) = match &self.cell {
                Some((c, iv)) => (
                    crate::mcmc::wrap_rows(r_up, &c.view(), iv),
                    crate::mcmc::wrap_rows(r_dn, &c.view(), iv),
                ),
                None => (r_up.to_owned(), r_dn.to_owned()),
            };
            let (det_state, ginv) = seed_det_state_and_inverse(gem, &ruw.view(), &rdw.view());
            self.det_state = det_state;
            self.ginv = ginv;
            *self.det_wdot_cache.borrow_mut() = None;
        }
    }

    /// Advance the cached determinant/J3/DTN state after the GFMC categorical
    /// hop. The determinant inverse receives the same rank-1 Sherman-Morrison
    /// update as the native MCMC sweep; rebuilding all-electron state on every
    /// fixed GFMC_n substep is therefore unnecessary.
    #[allow(clippy::too_many_arguments)]
    fn advance_after_hop(
        &mut self,
        gem: &GeminalData,
        jastrow: &JastrowData,
        j3: Option<(&OrbData, &ArrayView2<f64>)>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        old_pos: [f64; 3],
        newpos: [f64; 3],
    ) {
        let n = self.ginv.nrows();
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        let wrapped_newpos = self.wrap(newpos);
        let det_col = {
            let mut grad_cache = self.det_grad_cache.borrow_mut();
            match grad_cache.as_mut() {
                // The cache was created by `det_grad_lap` under the same pure
                // predicate, so seed and refresh always pick one kernel family
                // (molecular legacy or folded basis-generic kernels).
                Some(cache) => crate::timed!(
                    "gfmc.adv_det_vgl",
                    if fast_generic_det_gradlap_cell(gem, self.cell.is_some()) {
                        crate::determinant::update_det_grad_lap_cache_and_value_folded(
                            gem,
                            cache,
                            is_up,
                            e,
                            wrapped_newpos,
                        )
                    } else {
                        crate::determinant::update_det_grad_lap_cache_and_value(
                            gem,
                            cache,
                            is_up,
                            e,
                            wrapped_newpos,
                        )
                    }
                ),
                None => crate::timed!("gfmc.adv_det_orb", orb.compute_single_fused(wrapped_newpos)),
            }
        };
        crate::timed!("gfmc.adv_det_inverse", {
            let mut u = vec![0.0_f64; n];
            let mut v = vec![0.0_f64; n];
            if is_up {
                u[e] = 1.0;
                crate::determinant::compute_v_up_move_from_orb_into(
                    gem,
                    &self.det_state,
                    e,
                    &det_col,
                    &mut v,
                );
            } else {
                v[e] = 1.0;
                crate::determinant::compute_u_dn_move_from_orb_into(
                    &self.det_state,
                    e,
                    &det_col,
                    &mut u,
                );
            }
            let mut ainv_u = vec![0.0_f64; n];
            let mut vt_ainv = vec![0.0_f64; n];
            let ratio = crate::determinant::sm_ratio_into(
                &self.ginv.view(),
                &u,
                &v,
                &mut ainv_u,
                &mut vt_ainv,
            );
            if let Some(cache) = self.det_wdot_cache.borrow_mut().as_mut() {
                crate::timed!(
                    "gfmc.adv_det_winv",
                    advance_det_wdot_cache(
                        cache,
                        gem,
                        &self.det_state,
                        is_up,
                        e,
                        &det_col,
                        &u,
                        &ainv_u,
                        &vt_ainv,
                        ratio,
                    )
                );
                bump_route_counter(&GFMC_WINV_ADVANCES);
            }
            let inv = self.ginv.as_slice_mut().expect("G inverse contiguous");
            for i in 0..n {
                let ai = ainv_u[i] / ratio;
                for k in 0..n {
                    inv[i * n + k] -= ai * vt_ainv[k];
                }
            }
        });
        crate::timed!(
            "gfmc.adv_det_state",
            crate::determinant::advance_det_ratio_state_from_orb(
                gem,
                &mut self.det_state,
                is_up,
                e,
                &det_col,
            )
        );
        crate::timed!(
            "gfmc.adv_det_screen",
            self.screen_interval_ginv(gem, r_up, r_dn)
        );
        if let (Some((orb, jm)), Some(st)) = (j3, self.j3_state.as_mut()) {
            let col = crate::timed!("gfmc.adv_j3_orb", orb.compute_single(wrapped_newpos));
            let d = crate::timed!(
                "gfmc.adv_j3_state",
                crate::jastrow_factor::advance_j3_ratio_state_with_delta(jm, st, is_up, e, &col,)
            );
            let (current_up, current_dn) = if orb.mo_coefficients.is_none() {
                match self.j3_wdot_cache.borrow_mut().as_mut() {
                    Some(cache) => crate::timed!(
                        "gfmc.adv_j3_wdot",
                        advance_j3_wdot_cache(cache, st, is_up, e, &d)
                    ),
                    None => j3_current_dots(st, &d),
                }
            } else {
                // A MO-wrapped J3 cache stores projected q rows; the common
                // raw-basis increment cannot be added without projection.
                *self.j3_wdot_cache.borrow_mut() = None;
                j3_current_dots(st, &d)
            };
            if let Some(cache) = self.j3_grad_cache.borrow_mut().as_mut() {
                crate::timed!(
                    "gfmc.adv_j3_grad",
                    if orb.mo_coefficients.is_none() {
                        let wdot = self.j3_wdot_cache.borrow();
                        let w = wdot
                            .as_ref()
                            .expect("raw-AO J3 derivative route carries q rows");
                        crate::jastrow_factor::advance_j3_grad_lap_cache_with_q_delta(
                            orb,
                            cache,
                            &w.up.view(),
                            &w.dn.view(),
                            &d,
                            is_up,
                            e,
                            wrapped_newpos,
                        )
                    } else {
                        crate::jastrow_factor::update_j3_grad_lap_cache(
                            orb,
                            cache,
                            is_up,
                            e,
                            wrapped_newpos,
                        )
                    }
                );
            }
            *self.j3_pending_delta.borrow_mut() = Some(J3PendingDelta {
                d,
                current_up,
                current_dn,
            });
        }
        if let (Some(d), Some(st)) = (&jastrow.dtn, self.dtn_state.as_mut()) {
            crate::timed!(
                "gfmc.adv_dtn",
                crate::dtn::advance_dtn_ratio_state(d, st, r_up, r_dn, is_up, e)
            );
            if let Some(table) = self.dtn_candidate_table.borrow_mut().as_mut() {
                crate::timed!(
                    "gfmc.adv_dtn_tabpip",
                    crate::dtn::advance_dtn_candidate_table(
                        d, table, st, r_up, r_dn, is_up, e, old_pos,
                    )
                );
                bump_route_counter(&GFMC_DTN_TABPIP_ADVANCES);
            }
        }
        if let Some(cache) = self.j12_grad_cache.borrow_mut().as_mut() {
            crate::timed!(
                "gfmc.adv_j12_grad",
                crate::jastrow_factor::update_j12_grad_lap_cache_generic(
                    jastrow, cache, r_up, r_dn, is_up, e, old_pos,
                )
            );
        }
        if let Some(cache) = self.dtn_grad_cache.borrow_mut().as_mut() {
            let d = jastrow
                .dtn
                .as_ref()
                .expect("DTN derivative cache exists only with a DTN term");
            crate::timed!(
                "gfmc.adv_dtn_grad",
                crate::dtn::update_dtn_grad_lap_cache(d, cache, r_up, r_dn, is_up, e, old_pos)
            );
        }
        if self.reuse_fixed_mesh {
            let change = HopChange {
                is_up,
                electron: e,
                old_pos,
            };
            self.kinetic_dirty.set(Some(change));
            self.ecp_dirty.set(Some(change));
            self.ecp_dtn_dirty.set(Some(change));
        }
        self.ecp_local_dirty.set(Some((is_up, e)));
        // Persistent candidate J3 delta tables consume `j3_pending_delta`;
        // raw-AO q rows were advanced above and remain valid for continuum
        // derivatives and non-fixed candidate fallbacks.
    }

    fn ensure_kinetic_dtn_candidate_table(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        shifts: &[[f64; 3]],
    ) {
        if !self.reuse_fixed_mesh {
            return;
        }
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return,
        };
        let mut slot = self.dtn_candidate_table.borrow_mut();
        let rebuild = slot
            .as_ref()
            .is_none_or(|table| !table.matches_layout(r_up, r_dn, shifts));
        if rebuild {
            *slot = Some(crate::timed!(
                "gfmc.kin_dtn_tabpip_build",
                crate::dtn::init_dtn_candidate_table(d, state, r_up, r_dn, shifts)
            ));
            bump_route_counter(&GFMC_DTN_TABPIP_BUILDS);
        }
    }

    fn ensure_ecp_dtn_candidate_table(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        positions_up: &[[f64; 3]],
        positions_dn: &[[f64; 3]],
        rows_per_electron: usize,
    ) {
        if !self.reuse_fixed_mesh {
            return;
        }
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return,
        };
        let mut slot = self.ecp_dtn_candidate_table.borrow_mut();
        let rebuild = slot.as_ref().is_none_or(|table| {
            !table.matches_position_layout(
                r_up,
                r_dn,
                positions_up,
                positions_dn,
                rows_per_electron,
            )
        });
        if rebuild {
            *slot = Some(crate::timed!(
                "gfmc.ecp_dtn_tabpip_build",
                crate::dtn::init_dtn_candidate_table_from_positions(
                    d,
                    state,
                    r_up,
                    r_dn,
                    positions_up,
                    positions_dn,
                    rows_per_electron,
                )
            ));
            self.ecp_dtn_dirty.set(None);
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_BUILDS);
        } else if let Some(change) = self.ecp_dtn_dirty.take() {
            let positions = if change.is_up {
                positions_up
            } else {
                positions_dn
            };
            let offset = change.electron * rows_per_electron;
            crate::timed!(
                "gfmc.ecp_dtn_tabpip_advance",
                crate::dtn::advance_dtn_candidate_table_from_positions(
                    d,
                    slot.as_mut().expect("ECP DTN table initialized"),
                    state,
                    r_up,
                    r_dn,
                    change.is_up,
                    change.electron,
                    change.old_pos,
                    &positions[offset..offset + rows_per_electron],
                )
            );
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_ADVANCES);
        }
    }

    /// Prefer the maintained all-electron candidate table when this group is
    /// one row of the fixed kinetic lattice.  ECP and stateless calls retain
    /// the established geometry-cache mesh kernel.
    fn dtn_group_ratios(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
    ) -> Option<Vec<f64>> {
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return None,
        };
        if let Some(ratios) = self
            .dtn_candidate_table
            .borrow()
            .as_ref()
            .and_then(|table| table.ratios_for_group(d, is_up, e, newpos))
        {
            bump_route_counter(&GFMC_DTN_TABPIP_HITS);
            return Some(ratios);
        }
        if let Some(ratios) = self
            .ecp_dtn_candidate_table
            .borrow()
            .as_ref()
            .and_then(|table| table.ratios_for_group(d, is_up, e, newpos))
        {
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_HITS);
            return Some(ratios);
        }
        let points = Array2::from_shape_fn((newpos.len(), 3), |(row, axis)| newpos[row][axis]);
        let mut ratios = vec![0.0_f64; newpos.len()];
        crate::dtn::compute_dtn_ratio_mesh_cached(
            d,
            state,
            r_up,
            r_dn,
            is_up,
            e,
            &points.view(),
            &mut ratios,
        );
        bump_route_counter(&GFMC_DTN_CACHE_HITS);
        Some(ratios)
    }

    /// Combined single-move Jastrow ratio (J1/J2 + J3 + DTN) for moving electron
    /// (`is_up`, `e`) to `np` -- the exact factor set of the reference
    /// `exp(jastrow_value_for_ratio(new) - jastrow_value_for_ratio(base))`.
    #[allow(clippy::too_many_arguments)]
    fn jas_ratios(
        &self,
        jastrow: &JastrowData,
        j3: Option<(&OrbData, &ArrayView2<f64>)>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(newpos.len());
        let dtn_ratios = self.dtn_group_ratios(jastrow, r_up, r_dn, is_up, e, newpos);
        // A molecular J3 reaches this point as a raw GTO basis: the production
        // `-j3 mo` (MO-wrapped) J3 is lowered to its raw-AO equivalent at the
        // GFMC parse boundary (`lower_molecular_mo_j3`); an MO-wrapped
        // J3 that is NOT lowered has no streaming state and takes the slow
        // fallback below. Reuse one AO buffer for the whole candidate group. This removes one heap allocation
        // per mesh point while retaining the accurate small-water AO kernel.
        // A raw-AO periodic pGTO J3 takes the image-folded candidate
        // batch on home-cell-wrapped positions (same dispatch the mesh caches
        // use); other J3 classes with a live streaming state fall through to
        // the per-point `compute_single` branch below.
        let j3_fast = j3
            .filter(|_| self.j3_state.is_some())
            .and_then(
                |(orb, jm)| match j3_candidate_class(orb, self.cell.is_some()) {
                    class @ CandidateBasisClass::MolecularGto => Some((
                        jm,
                        j3_candidate_batch(orb, class, newpos, CandidateEvent::Build),
                    )),
                    class @ CandidateBasisClass::PeriodicPgto => {
                        let eval = wrapped_candidate_positions(self, newpos);
                        Some((
                            jm,
                            j3_candidate_batch(orb, class, &eval, CandidateEvent::Build),
                        ))
                    }
                    _ => None,
                },
            );
        let j3_fast = j3_fast;
        // Slow J3 fallback: J3 present but no streaming state (MO-wrapped,
        // periodic image-tiled, or non-contiguous `j_matrix` J3). The ratio is
        // the full-value difference `exp(J3(r') - J3(r))` with the reference
        // `jastrow_value_for_ratio` conventions (home-cell wrap for the AO eval,
        // both whole values in the `jastrow_ratio` zone that owns the ratio, never
        // `jastrow_eval`, issue #781); J1/J2 and DTN keep their cached fast
        // forms above/below.
        let j3_slow = match (j3, &self.j3_state) {
            (Some((orb, jm)), None) => {
                let f32_ratio = jastrow.precision_policy.is_f32("jastrow_ratio");
                let ruw = orb.wrap_electrons(r_up).unwrap_or_else(|| r_up.to_owned());
                let rdw = orb.wrap_electrons(r_dn).unwrap_or_else(|| r_dn.to_owned());
                let au = orb.compute(&ruw.view());
                let ad = orb.compute(&rdw.view());
                let base = crate::jastrow_factor::compute_jastrow_three_body_value_with_policy(
                    jm,
                    &au.view(),
                    &ad.view(),
                    f32_ratio,
                );
                Some((orb, jm, base, f32_ratio))
            }
            _ => None,
        };
        // Owned moved-spin coordinates for the slow J3 fallback (row `e` is
        // overwritten per candidate; all other rows stay pristine).
        let mut j3_slow_moved = j3_slow.as_ref().map(|_| {
            if is_up {
                r_up.to_owned()
            } else {
                r_dn.to_owned()
            }
        });
        for (candidate, &np) in newpos.iter().enumerate() {
            let mut jr =
                crate::jastrow_factor::jastrow_1b2b_ratio(jastrow, r_up, r_dn, is_up, e, np);
            if let (Some((jm, cols)), Some(st)) = (&j3_fast, &self.j3_state) {
                let col = cols.row(out.len());
                jr = crate::jastrow_factor::multiply_ratio_components(
                    jastrow,
                    jr,
                    crate::jastrow_factor::j3_ratio_fast_linear(
                        jm,
                        st,
                        is_up,
                        e,
                        col.as_slice().expect("candidate AO row contiguous"),
                    ),
                );
            } else if let (Some((orb, jm)), Some(st)) = (j3, &self.j3_state) {
                let col = orb.compute_single(self.wrap(np));
                let j3r = if self.cell.is_none() {
                    crate::jastrow_factor::j3_ratio_fast_linear(jm, st, is_up, e, &col)
                } else {
                    crate::jastrow_factor::j3_ratio(jm, st, is_up, e, &col)
                };
                jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, j3r);
            } else if let (Some((orb, jm, base, f32_ratio)), Some(rm)) =
                (&j3_slow, j3_slow_moved.as_mut())
            {
                rm[[e, 0]] = np[0];
                rm[[e, 1]] = np[1];
                rm[[e, 2]] = np[2];
                let val = {
                    // Reborrow BOTH parameter views locally so the moved-row
                    // view and the pristine-spin view share one lifetime
                    // (ArrayBase is invariant over ViewRepr, so mixing the
                    // caller's view lifetimes with a local one does not unify).
                    let rm_view = rm.view();
                    let ru_reb = r_up.view();
                    let rd_reb = r_dn.view();
                    let (ruv, rdv): (&ArrayView2<f64>, &ArrayView2<f64>) = if is_up {
                        (&rm_view, &rd_reb)
                    } else {
                        (&ru_reb, &rm_view)
                    };
                    let ruw = orb.wrap_electrons(ruv).unwrap_or_else(|| ruv.to_owned());
                    let rdw = orb.wrap_electrons(rdv).unwrap_or_else(|| rdv.to_owned());
                    let au = orb.compute(&ruw.view());
                    let ad = orb.compute(&rdw.view());
                    crate::jastrow_factor::compute_jastrow_three_body_value_with_policy(
                        jm,
                        &au.view(),
                        &ad.view(),
                        *f32_ratio,
                    )
                };
                let j3r = if jastrow.precision_policy.is_f32("jastrow_ratio") {
                    f64::from(((val - *base) as f32).exp())
                } else {
                    (val - *base).exp()
                };
                jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, j3r);
            }
            if let Some(ratios) = &dtn_ratios {
                jr = crate::jastrow_factor::multiply_ratio_components(
                    jastrow,
                    jr,
                    ratios[candidate],
                );
            }
            out.push(jr);
        }
        out
    }

    /// Consume a maintained J1/J2 log-ratio row when J3 is inactive, so a
    /// plain J1/J2 wavefunction reads the table it builds and advances (the J3
    /// candidate-AO branch is not the only consumer).
    #[allow(
        clippy::too_many_arguments,
        reason = "move, candidate and precomputed-ratio inputs"
    )]
    fn jas_ratios_from_precomputed_j12_into(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
        deltas: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(newpos.len(), deltas.len());
        debug_assert_eq!(newpos.len(), out.len());
        let dtn_ratios = self.dtn_group_ratios(jastrow, r_up, r_dn, is_up, e, newpos);
        let f32_ratio = jastrow.precision_policy.is_f32("jastrow_ratio");
        for s in 0..newpos.len() {
            let mut ratio = if f32_ratio {
                f64::from((deltas[s] as f32).exp())
            } else {
                deltas[s].exp()
            };
            if let Some(dtn) = &dtn_ratios {
                ratio = crate::jastrow_factor::multiply_ratio_components(jastrow, ratio, dtn[s]);
            }
            out[s] = ratio;
        }
    }

    /// Consume a persistent J3 log-ratio row, optionally alongside the
    /// persistent J1/J2 row. This is the final `tabpip`-style route: no q-vector
    /// construction, J3 wdot rebuild, or candidate-AO contraction occurs in
    /// the per-electron mesh loop.
    #[allow(clippy::too_many_arguments)]
    fn jas_ratios_from_precomputed_j3_into(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
        j12_deltas: Option<&[f64]>,
        j3_deltas: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(newpos.len(), j3_deltas.len());
        debug_assert_eq!(newpos.len(), out.len());
        match j12_deltas {
            Some(deltas) => out.copy_from_slice(deltas),
            None => crate::jastrow_factor::jastrow_1b2b_deltas_into(
                jastrow, r_up, r_dn, is_up, e, newpos, out,
            ),
        }
        let dtn_ratios = self.dtn_group_ratios(jastrow, r_up, r_dn, is_up, e, newpos);
        for s in 0..out.len() {
            let mut ratio = (out[s] + j3_deltas[s]).exp();
            if let Some(dtn) = &dtn_ratios {
                ratio = crate::jastrow_factor::multiply_ratio_components(jastrow, ratio, dtn[s]);
            }
            out[s] = ratio;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn jas_ratios_from_candidate_aos_into(
        &self,
        jastrow: &JastrowData,
        jm: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
        cols: &ArrayView2<f64>,
        q_scratch: &mut [f64],
        precomputed_j12: Option<&[f64]>,
        precomputed_j3: Option<(&[f64], f64)>,
        active_j3: Option<&[usize]>,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), newpos.len());
        // Call sites gate the candidate-AO J3 route on `j3_state.is_some()`
        // (see `j3_fast` in the kinetic/ECP fast meshes), so this expect holds.
        let st = self.j3_state.as_ref().expect("J3 state for candidate AOs");
        let n_orb = jm.nrows();
        debug_assert!(q_scratch.len() >= n_orb);
        let (aos_old, w_same, cross_same, cross_other) = if is_up {
            (&st.aos_up, &st.w_up, &st.cross_up, &st.cross_dn)
        } else {
            (&st.aos_dn, &st.w_dn, &st.cross_dn, &st.cross_up)
        };
        let dtn_ratios = self.dtn_group_ratios(jastrow, r_up, r_dn, is_up, e, newpos);
        let j3_base = match precomputed_j3 {
            Some((weight, old_dot)) => {
                debug_assert_eq!(weight.len(), n_orb);
                q_scratch[..n_orb].copy_from_slice(weight);
                -old_dot
            }
            None => {
                let mut base = 0.0;
                for a in 0..n_orb {
                    let q = jm[[a, n_orb]] + cross_other[a] + cross_same[a] - w_same[[a, e]];
                    q_scratch[a] = q;
                    base -= aos_old[[a, e]] * q;
                }
                base
            }
        };
        let f32_ratio = jastrow.precision_policy.is_f32("jastrow_ratio");
        if !f32_ratio {
            match precomputed_j12 {
                Some(deltas) => out.copy_from_slice(deltas),
                None => crate::jastrow_factor::jastrow_1b2b_deltas_into(
                    jastrow, r_up, r_dn, is_up, e, newpos, out,
                ),
            }
        }
        let mut j3_dots = [0.0_f64; 6];
        let batched_j3 = !f32_ratio && newpos.len() == 6;
        if batched_j3 {
            match active_j3 {
                Some(active) => candidate_wdot_active_into(
                    cols,
                    &q_scratch[..n_orb],
                    active,
                    j3_base,
                    &mut j3_dots,
                ),
                None => candidate_wdot_into(cols, &q_scratch[..n_orb], j3_base, &mut j3_dots),
            }
        }
        for (s, &np) in newpos.iter().enumerate() {
            if f32_ratio {
                let col = cols.row(s);
                let j12 = match precomputed_j12 {
                    Some(deltas) => f64::from((deltas[s] as f32).exp()),
                    None => {
                        crate::jastrow_factor::jastrow_1b2b_ratio(jastrow, r_up, r_dn, is_up, e, np)
                    }
                };
                let j3r = crate::jastrow_factor::j3_ratio_fast_linear(
                    jm,
                    st,
                    is_up,
                    e,
                    col.as_slice().expect("candidate AO row contiguous"),
                );
                let mut jr = crate::jastrow_factor::multiply_ratio_components(jastrow, j12, j3r);
                if let Some(ratios) = &dtn_ratios {
                    jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, ratios[s]);
                }
                out[s] = jr;
                continue;
            }
            let mut delta = out[s];
            if batched_j3 {
                delta += j3_dots[s];
            } else {
                let col = cols.row(s);
                delta += match active_j3 {
                    Some(active) => {
                        let mut dot = j3_base;
                        for &a in active {
                            dot += q_scratch[a] * col[a];
                        }
                        dot
                    }
                    None => {
                        let col = col.as_slice().expect("candidate AO row contiguous");
                        j3_base + dot_product_fast(&q_scratch[..n_orb], col)
                    }
                };
            }
            let mut jr = delta.exp();
            if let Some(ratios) = &dtn_ratios {
                jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, ratios[s]);
            }
            out[s] = jr;
        }
    }

    fn det_ratios_from_candidate_aos(
        &self,
        gem: &GeminalData,
        is_up: bool,
        e: usize,
        aos: &ArrayView2<f64>,
    ) -> Vec<f64> {
        let npts = aos.nrows();
        let n = self.ginv.nrows();
        let m = if is_up {
            self.det_state.ao_up.nrows()
        } else {
            self.det_state.ao_dn.nrows()
        };
        let mut col = vec![0.0_f64; m];
        let mut delta = vec![0.0_f64; n];
        let mut weight = vec![0.0_f64; aos.ncols()];
        let mut out = vec![0.0_f64; npts];
        self.det_ratios_from_candidate_aos_into(
            gem,
            is_up,
            e,
            aos,
            &mut col,
            &mut delta,
            &mut weight,
            &mut out,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn det_ratios_from_candidate_aos_into(
        &self,
        gem: &GeminalData,
        is_up: bool,
        e: usize,
        aos: &ArrayView2<f64>,
        col: &mut [f64],
        _delta: &mut [f64],
        weight: &mut [f64],
        out: &mut [f64],
    ) {
        let npts = aos.nrows();
        let n = self.ginv.nrows();
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        debug_assert_eq!(out.len(), npts);
        debug_assert!(
            col.len()
                >= if is_up {
                    self.det_state.ao_up.nrows()
                } else {
                    self.det_state.ao_dn.nrows()
                }
        );
        debug_assert!(weight.len() >= aos.ncols());

        // For a fixed base configuration and moved electron, the complete
        // determinant ratio is affine in the proposed orbital column:
        //   ratio = 1 + q . (MO(new) - MO(old)).
        // Fold q through the AO->MO coefficients once per six-point candidate
        // group. Each candidate then needs one nao dot product instead of four
        // nao dot products plus the paired-table and inverse contractions.
        let m = col.len();
        col.fill(0.0); // q in MO space
        if is_up {
            let n_dn = self.det_state.paired_dn.ncols();
            for (a, col_a) in col[..self.det_state.paired_dn.nrows()]
                .iter_mut()
                .enumerate()
            {
                let mut q = 0.0;
                for j in 0..n_dn {
                    q += self.det_state.paired_dn[[a, j]] * self.ginv[[j, e]];
                }
                *col_a = q;
            }
            for j in 0..gem.n_unpaired() {
                let gi = self.ginv[[n_dn + j, e]];
                for (a, q) in col.iter_mut().enumerate().take(m) {
                    *q += gem.lambda_value(a, gem.orb_num_dn + j) * gi;
                }
            }
        } else {
            for (a, col_a) in col[..m].iter_mut().enumerate() {
                let mut q = 0.0;
                for i in 0..n {
                    q += self.ginv[[e, i]] * self.det_state.paired_up_lambda[[i, a]];
                }
                *col_a = q;
            }
        }
        let old = if is_up {
            &self.det_state.ao_up
        } else {
            &self.det_state.ao_dn
        };
        let mut base = 1.0;
        for a in 0..m {
            base -= col[a] * old[[a, e]];
        }
        // The table MUST be in raw-AO space (`CandidateSpace::RawAo`): the fold
        // below turns `q` into AO-space weights, so an MO-width table would be
        // contracted against AO-space weights and return a plausible-looking
        // wrong ratio. Release-mode contract on both widths -- the mis-space
        // this assertion prevents that candidate-space routing defect.
        match &orb.mo_coefficients {
            Some(c) => {
                let na = c.ncols();
                assert_eq!(
                    aos.ncols(),
                    na,
                    "candidate table must be raw-AO width for the MO weight fold"
                );
                let cs = c.as_slice().expect("C contiguous");
                weight[..na].fill(0.0);
                for a in 0..m {
                    let qa = col[a];
                    let row = &cs[a * na..a * na + na];
                    for h in 0..na {
                        weight[h] += qa * row[h];
                    }
                }
            }
            None => {
                assert_eq!(
                    aos.ncols(),
                    m,
                    "raw-AO candidate table width must match the orbital rank"
                );
                weight[..m].copy_from_slice(&col[..m]);
            }
        }
        let nw = aos.ncols();
        for (s, out_s) in out[..npts].iter_mut().enumerate() {
            let ao = aos.row(s);
            let ao = ao.as_slice().expect("candidate AO row contiguous");
            let mut acc = [0.0_f64; 8];
            let n8 = nw & !7;
            let mut h = 0;
            while h < n8 {
                for k in 0..8 {
                    acc[k] += weight[h + k] * ao[h + k];
                }
                h += 8;
            }
            let mut ratio = base + acc.into_iter().sum::<f64>();
            for h in n8..nw {
                ratio += weight[h] * ao[h];
            }
            *out_s = ratio;
        }
    }

    /// Batched signed det ratios for `npts` candidate positions of ONE electron
    /// (`is_up`, `e`): orbital columns via `compute_single` (wrap applied), the paired
    /// contraction as one GEMM, then `1 + v G^-1 e_e` / `1 + e_e G^-1 u` per point.
    fn det_ratios(
        &self,
        gem: &GeminalData,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
    ) -> Vec<f64> {
        let npts = newpos.len();
        let n = self.ginv.nrows();
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        let m = if is_up {
            self.det_state.ao_up.nrows()
        } else {
            self.det_state.ao_dn.nrows()
        };
        if self.cell.is_none() && molecular_candidate_ok(orb, false) {
            // Reuse AO/MO scratch over the candidate group. Besides eliminating
            // per-point Vec allocation, while the small-water MO projection keeps
            // the low-overhead 8-accumulator kernel.
            let mut aos = molecular_aos_candidate_batch(&orb.aos, newpos);
            round_det_candidate_aos(orb, &mut aos);
            return self.det_ratios_from_candidate_aos(gem, is_up, e, &aos.view());
        }
        let mut orb_mesh = Array2::<f64>::zeros((m, npts));
        for (s, np) in newpos.iter().enumerate() {
            let col = orb.compute_single_fused(self.wrap(*np));
            for a in 0..m {
                orb_mesh[[a, s]] = col[a];
            }
        }
        let mut out = vec![0.0_f64; npts];
        if is_up {
            let vmat = crate::determinant::compute_v_up_move_batch(
                gem,
                &self.det_state,
                e,
                &orb_mesh.view(),
            );
            for s in 0..npts {
                let mut ratio = 1.0_f64;
                for i in 0..n {
                    ratio += vmat[[i, s]] * self.ginv[[i, e]];
                }
                out[s] = ratio;
            }
        } else {
            let umat = crate::determinant::compute_u_dn_move_batch(
                gem,
                &self.det_state,
                e,
                &orb_mesh.view(),
            );
            for s in 0..npts {
                let mut ratio = 1.0_f64;
                for i in 0..n {
                    ratio += self.ginv[[e, i]] * umat[[i, s]];
                }
                out[s] = ratio;
            }
        }
        out
    }
}

/// Build the [`MeshFastState`] for one configuration. Always succeeds: the
/// det-side streaming state (`compute_single` path) and the DTN/J1J2/ECP cache
/// slots are basis-agnostic and PBC-native. Only the streaming J3 ratio state
/// carries a coverage gate: MO-wrapped J3 or a non-contiguous `j_matrix` leave
/// `j3_state = None` and every consumer falls back to the full-value J3
/// evaluation FOR J3 ONLY (the reference `jastrow_value_for_ratio` convention), keeping
/// the remaining cached members live. The production molecular J3 is
/// MO-wrapped (`-j3 mo`), so the real-path GFMC parse lowers it to raw AO
/// (`lower_molecular_mo_j3`) before it reaches this gate; only a
/// periodic MO-wrapped J3 still arrives here MO-wrapped. Raw-AO periodic pGTO J3
/// under a PBC cell also carries a streaming state (`J3RatioState` is pure
/// linear algebra over home-rank AO columns; init/advance evaluate at the
/// ewald-cell-wrapped coordinate, the same wrap the det chain uses). Molecular
/// + PBC-Gamma J1/J2, DTN, and raw-AO molecular J3 use the same streaming
///   ratio kernels the production MCMC sweep uses.
///   Base built from scratch at home-cell-wrapped positions: single-path AO
///   matrices -> streaming det-ratio state -> consistent base geminal -> G^-1.
///   Single-path so every SM candidate column is bit-consistent with the stored
///   base column (exact det_new/det_base ratio). Shared by the interval seed and
///   the in-interval inverse refresh, so a refreshed state equals a fresh one.
fn seed_det_state_and_inverse(
    gem: &GeminalData,
    ruw: &ArrayView2<f64>,
    rdw: &ArrayView2<f64>,
) -> (crate::determinant::DetRatioState, Array2<f64>) {
    // The carried geminal uses the fused `ao_eval` value convention
    // (fp64 body, completed AO value rounded) for seed, hop and candidate rows;
    // the in-interval inverse refresh re-seeds through this same function, so a refreshed
    // state equals a fresh one bitwise and G keeps one value convention.
    let ao_up0 = crate::determinant::orb_matrix_via_single_fused(&gem.orb_up, ruw);
    let ao_dn0 = crate::determinant::orb_matrix_via_single_fused(&gem.orb_dn, rdw);
    let det_state = crate::determinant::init_det_ratio_state_from_orb_mats(gem, ao_up0, ao_dn0);
    let g = crate::determinant::geminal_from_det_ratio_state(gem, &det_state);
    let ginv = crate::determinant::inverse(&g.view());
    (det_state, ginv)
}

/// Whether a real-path J3 orbital is a MOLECULAR MO-wrapped GTO leaf
/// (`MOs_data` over a one-image molecular GTO basis, no plane-wave / B-spline
/// child, no wrap cell): the production `jqmc-tool convert-to -j3 mo` J3.
/// [`mesh_fast_state`] gates its streaming J3 ratio state (and with it the
/// kinetic/ECP candidate caches) on a raw-AO orbital, so such a J3 must be
/// lowered with [`lower_molecular_mo_j3`] before it reaches the GFMC kernels.
pub(crate) fn is_molecular_mo_j3(orb: &OrbData) -> bool {
    orb.mo_coefficients.is_some()
        && orb.pw.is_none()
        && orb.bspline.is_none()
        && orb.aos.wrap_cell.is_none()
        && orb.aos.n_image == 1
        && orb.aos.num_ao > 0
}

/// Lower a nonzero molecular MO-wrapped J3 to its exact raw-AO
/// equivalent (`M_ao = C^T M_mo C`, `j1_ao = C^T j1_mo`, raw-AO orbital) via
/// [`crate::gfmc_forces::lower_j3_to_ao`]. The J3 value, ratios, gradient and
/// Laplacian are invariant under this change of basis, and the raw-AO form is
/// the one [`mesh_fast_state`] streams (`J3RatioState` + candidate tables).
/// Without it every kinetic/ECP mesh evaluation took the full-value J3 slow
/// fallback (all electrons' AOs re-evaluated per candidate; ~40x slower LRDMC
/// on the example01 water trial). Any other J3 (raw AO, periodic, composite)
/// is returned unchanged: a periodic MO-wrapped J3 keeps its existing route.
pub(crate) fn lower_molecular_mo_j3(orb: OrbData, jm: Array2<f64>) -> (OrbData, Array2<f64>) {
    if !is_molecular_mo_j3(&orb) {
        return (orb, jm);
    }
    crate::gfmc_forces::lower_j3_to_ao(Some(&orb), Some(&jm.view()))
        .expect("lower_j3_to_ao returns Some for a present J3")
}

pub(crate) fn mesh_fast_state(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    pbc: Option<&crate::ewald::EwaldRef>,
) -> MeshFastState {
    let j3_state_ok = j3.is_some_and(|(orb, jm)| {
        orb.mo_coefficients.is_none()
            && jm.as_slice().is_some()
            && (orb.wrap_electrons(r_up).is_none()
                // Periodic raw-AO pGTO J3 streams too, provided the
                // ewald cell is present so the init coordinates below (`ruw` /
                // `rdw`) and every later advance/candidate wrap use one and
                // the same home-cell convention (`MeshFastState::wrap`).
                || (pbc.is_some()
                    && candidate_basis_class(orb, true, true)
                        == CandidateBasisClass::PeriodicPgto))
    });
    // PBC wrap plumbing: AO-eval coordinates are home-cell-wrapped (matching the batch
    // geminal's `wrap_electrons`); walker storage / Jastrow / DTN args stay unwrapped
    // (exactly the MCMC streaming convention).
    let cell = pbc.map(|p| (p.cell.to_owned(), crate::ewald::inv3(&p.cell)));
    let (ruw, rdw) = match &cell {
        Some((c, iv)) => (
            crate::mcmc::wrap_rows(r_up, &c.view(), iv),
            crate::mcmc::wrap_rows(r_dn, &c.view(), iv),
        ),
        None => (r_up.to_owned(), r_dn.to_owned()),
    };
    let (det_state, ginv) = seed_det_state_and_inverse(gem, &ruw.view(), &rdw.view());
    let j3_state = j3.filter(|_| j3_state_ok).map(|(orb, jm)| {
        let au = orb.compute(&ruw.view());
        let ad = orb.compute(&rdw.view());
        crate::jastrow_factor::init_j3_ratio_state_with_policy(
            jm,
            &au.view(),
            &ad.view(),
            jastrow.precision_policy.is_f32("jastrow_ratio"),
        )
    });
    let dtn_state = jastrow
        .dtn
        .as_ref()
        .map(|d| crate::dtn::init_dtn_ratio_state(d, r_up, r_dn));
    MeshFastState {
        det_state,
        ginv,
        j3_state,
        dtn_state,
        dtn_candidate_table: std::cell::RefCell::new(None),
        ecp_dtn_candidate_table: std::cell::RefCell::new(None),
        ecp_dtn_dirty: std::cell::Cell::new(None),
        cell,
        reuse_fixed_mesh: false,
        kinetic_cache: std::cell::RefCell::new(None),
        kinetic_dirty: std::cell::Cell::new(None),
        ecp_cache: std::cell::RefCell::new(None),
        ecp_geometry_cache: std::cell::RefCell::new(None),
        ecp_dirty: std::cell::Cell::new(None),
        kinetic_det_prefetch: std::cell::RefCell::new(None),
        det_grad_cache: std::cell::RefCell::new(None),
        j3_grad_cache: std::cell::RefCell::new(None),
        j12_grad_cache: std::cell::RefCell::new(None),
        dtn_grad_cache: std::cell::RefCell::new(None),
        det_wdot_cache: std::cell::RefCell::new(None),
        j3_wdot_cache: std::cell::RefCell::new(None),
        j3_pending_delta: std::cell::RefCell::new(None),
        ecp_local_cache: std::cell::RefCell::new(None),
        ecp_local_dirty: std::cell::Cell::new(None),
        ecp_skip_zero_groups: std::env::var_os("JQMC_RS_GFMC_ECP_KEEP_ZERO_GROUPS").is_none(),
        ecp_radial_cut: if std::env::var_os("JQMC_RS_GFMC_ECP_NO_RADIAL_CUT").is_some() {
            0.0
        } else {
            crate::coulomb_potential::FAST_ECP_NONLOCAL_RADIAL_CUT
        },
    }
}

/// FAST discretized-kinetic hopping amplitudes: the byte-for-byte identity is dropped
/// for a Sherman-Morrison rank-1 restructure. Each mesh point moves exactly ONE electron
/// by `+/- alat` along one (rotated) axis, so only that electron's geminal row/column
/// changes. Instead of re-evaluating the FULL geminal + LU + full Jastrow value per mesh
/// point (`compute_discretized_kinetic_elements`, the O(N^3)+O(N^2)-eval-per-point path
/// that XLA batches into one GEMM), this consumes the once-per-config [`MeshFastState`]
/// and forms each element from a rank-1 det ratio times the incremental single-move
/// Jastrow ratio, mirroring the MCMC sweep conventions. Same layout / sign as the
/// reference (up-major shift-minor then dn); `element = -1/(2a^2) * jr * det_ratio`.
/// Result differs from the reference only at the association-class fp floor -- FAST
/// chain mode only (statistically guarded); the bitexact GFMC_n chain keeps the
/// reference path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_discretized_kinetic_elements_fast(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    st: &MeshFastState,
) -> Vec<f64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let shifts = lattice_shifts(alat, rt);
    let inv_2a2 = -1.0 / (2.0 * alat * alat);
    let mut out = Vec::with_capacity(6 * (n_up + n_dn));
    let mut allpos = Vec::with_capacity(6 * (n_up + n_dn));
    for e in 0..n_up {
        for shift in &shifts {
            allpos.push([
                r_up[[e, 0]] + shift[0],
                r_up[[e, 1]] + shift[1],
                r_up[[e, 2]] + shift[2],
            ]);
        }
    }
    for e in 0..n_dn {
        for shift in &shifts {
            allpos.push([
                r_dn[[e, 0]] + shift[0],
                r_dn[[e, 1]] + shift[1],
                r_dn[[e, 2]] + shift[2],
            ]);
        }
    }
    let mut rt_bits = [0_u64; 9];
    for a in 0..3 {
        for b in 0..3 {
            rt_bits[3 * a + b] = rt[a][b].to_bits();
        }
    }
    st.ensure_kinetic_dtn_candidate_table(jastrow, r_up, r_dn, &shifts);
    let cell_present = st.cell.is_some();
    let det_up_class = det_candidate_class(gem, &gem.orb_up, cell_present);
    let det_dn_class = det_candidate_class(gem, &gem.orb_dn, cell_present);
    let det_up_fast = det_up_class != CandidateBasisClass::Unsupported;
    let det_dn_fast = det_dn_class != CandidateBasisClass::Unsupported;
    let det_up_space = det_candidate_space(&gem.orb_up);
    let det_dn_space = det_candidate_space(&gem.orb_dn);
    if !det_up_fast || !det_dn_fast {
        bump_route_counter(&GFMC_CAND_DET_SLOW);
    }
    let j3_class = j3.map(|(orb, _)| j3_candidate_class(orb, cell_present));
    let j3_fast = j3.is_none_or(|_| {
        st.j3_state.is_some() && j3_class != Some(CandidateBasisClass::Unsupported)
    });
    if j3.is_some() && !j3_fast {
        note_j3_slow_fallback(&GFMC_J3_SLOW_FALLBACKS);
    }
    // One wrap site per mesh evaluation: every candidate table is
    // seeded/refreshed from these home-cell coordinates; J1/J2 deltas and DTN
    // keep the raw `allpos`. Borrow-through (bit-identical) without a cell.
    let eval_all = wrapped_candidate_positions(st, &allpos);
    let eval_pos: &[[f64; 3]] = &eval_all;
    let j12_cache_ok = crate::jastrow_factor::j12_candidate_cache_ok(jastrow);
    let cache_ok = st.reuse_fixed_mesh && det_up_fast && det_dn_fast && j3_fast;
    let mut cache_guard = cache_ok.then(|| st.kinetic_cache.borrow_mut());
    if let Some(slot) = cache_guard.as_mut() {
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.alat_bits != alat.to_bits()
                || c.rt_bits != rt_bits
                || c.det_up.nrows() != 6 * n_up
                || c.det_dn.nrows() != 6 * n_dn
        });
        if rebuild {
            let det_up = crate::timed!(
                "gfmc.kin_det_ao_up",
                det_candidate_batch(
                    &gem.orb_up,
                    det_up_class,
                    det_up_space,
                    &eval_pos[..6 * n_up],
                    CandidateEvent::Build,
                )
            );
            let det_dn = crate::timed!(
                "gfmc.kin_det_ao_dn",
                det_candidate_batch(
                    &gem.orb_dn,
                    det_dn_class,
                    det_dn_space,
                    &eval_pos[6 * n_up..],
                    CandidateEvent::Build,
                )
            );
            let j3_cache = j3.map(|(orb, _)| {
                crate::timed!(
                    "gfmc.kin_j3_ao",
                    j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3"),
                        eval_pos,
                        CandidateEvent::Build,
                    )
                )
            });
            let j3_active = j3_cache
                .as_ref()
                .map(|aos| candidate_active_columns(&aos.view(), 6));
            if let (Some(aos), Some(active)) = (&j3_cache, &j3_active) {
                report_candidate_active_once(
                    "kinetic_j3",
                    active,
                    aos.ncols(),
                    &KIN_J3_ACTIVE_REPORTED,
                );
            }
            let j3_packed = match (&j3_cache, &j3_active) {
                (Some(aos), Some(active)) => {
                    Some(pack_candidate_groups(&aos.view(), 6, active.clone()))
                }
                _ => None,
            };
            let j3_delta = match (j3, st.j3_state.as_ref(), j3_cache.as_ref()) {
                (Some((_, jm)), Some(state), Some(aos)) if !state.f32_arithmetic => {
                    Some(crate::timed!(
                        "gfmc.kin_j3_tabpip_build",
                        fill_j3_candidate_deltas(jm, state, &aos.view(), n_up, 6)
                    ))
                }
                _ => None,
            };
            let (j12_up, j12_dn) = if j12_cache_ok {
                let mut up = vec![0.0; 6 * n_up];
                let mut dn = vec![0.0; 6 * n_dn];
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    true,
                    &allpos[..6 * n_up],
                    6,
                    &mut up,
                );
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    false,
                    &allpos[6 * n_up..],
                    6,
                    &mut dn,
                );
                (Some(up), Some(dn))
            } else {
                (None, None)
            };
            if j12_cache_ok {
                bump_route_counter(&GFMC_J12_TABPIP_KIN_BUILDS);
            }
            **slot = Some(KineticAoCache {
                alat_bits: alat.to_bits(),
                rt_bits,
                det_up,
                det_dn,
                j3: j3_cache,
                j3_active,
                j3_packed,
                j3_delta,
                j12_up,
                j12_dn,
            });
            st.kinetic_dirty.set(None);
        } else if let Some(change) = st.kinetic_dirty.take() {
            let is_up = change.is_up;
            let e = change.electron;
            let global_e = if is_up { e } else { n_up + e };
            // Refresh rows come from the SAME wrapped coordinates (and the same
            // kernel family) the table was seeded from -- seed/refresh fp
            // consistency within one cache.
            let pos = &eval_pos[6 * global_e..6 * global_e + 6];
            let cache = slot.as_mut().expect("kinetic cache initialized");
            let prefetched = st
                .kinetic_det_prefetch
                .borrow_mut()
                .take()
                .filter(|p| p.is_up == is_up && p.electron == e)
                .map(|p| p.kinetic);
            if is_up {
                let fresh = match prefetched {
                    Some(fresh) => fresh,
                    None => crate::timed!(
                        "gfmc.kin_det_ao_up",
                        det_candidate_batch(
                            &gem.orb_up,
                            det_up_class,
                            det_up_space,
                            pos,
                            CandidateEvent::Refresh
                        )
                    ),
                };
                cache
                    .det_up
                    .slice_mut(numpy::ndarray::s![6 * e..6 * e + 6, ..])
                    .assign(&fresh);
            } else {
                let fresh = match prefetched {
                    Some(fresh) => fresh,
                    None => crate::timed!(
                        "gfmc.kin_det_ao_dn",
                        det_candidate_batch(
                            &gem.orb_dn,
                            det_dn_class,
                            det_dn_space,
                            pos,
                            CandidateEvent::Refresh
                        )
                    ),
                };
                cache
                    .det_dn
                    .slice_mut(numpy::ndarray::s![6 * e..6 * e + 6, ..])
                    .assign(&fresh);
            }
            if let Some((orb, _)) = j3 {
                let fresh = crate::timed!(
                    "gfmc.kin_j3_ao",
                    j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3"),
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                if let Some(aos) = cache.j3.as_mut() {
                    aos.slice_mut(numpy::ndarray::s![6 * global_e..6 * global_e + 6, ..])
                        .assign(&fresh);
                }
                let columns = candidate_active_columns(&fresh.view(), 6)
                    .pop()
                    .expect("one refreshed kinetic J3 candidate group");
                if let Some(active) = cache.j3_active.as_mut() {
                    active[global_e] = columns.clone();
                }
                if let Some(packed) = cache.j3_packed.as_mut() {
                    packed[global_e] = pack_candidate_group(&fresh.view(), 0, 6, columns);
                }
            }
            if let (Some((_, jm)), Some(state), Some(pending)) = (
                j3,
                st.j3_state.as_ref(),
                st.j3_pending_delta.borrow().as_ref(),
            ) {
                if let (Some(packed), Some(table)) = (&cache.j3_packed, &mut cache.j3_delta) {
                    crate::timed!(
                        "gfmc.kin_j3_tabpip_advance",
                        advance_j3_candidate_deltas(
                            jm,
                            state,
                            n_up,
                            6,
                            change.is_up,
                            change.electron,
                            &pending.d,
                            &pending.current_up,
                            &pending.current_dn,
                            packed,
                            table,
                            None,
                        )
                    );
                }
            }
            if let (Some(up), Some(dn)) = (&mut cache.j12_up, &mut cache.j12_dn) {
                crate::timed!(
                    "gfmc.kin_j12_advance",
                    crate::jastrow_factor::update_j12_candidate_deltas_after_hop(
                        jastrow,
                        r_up,
                        r_dn,
                        change.is_up,
                        change.electron,
                        change.old_pos,
                        &allpos[..6 * n_up],
                        &allpos[6 * n_up..],
                        6,
                        up,
                        dn,
                    )
                );
                bump_route_counter(&GFMC_J12_TABPIP_KIN_ADVANCES);
            }
        }
    }

    let det_up_owned = if !cache_ok && det_up_fast {
        Some(crate::timed!(
            "gfmc.kin_det_ao_up",
            det_candidate_batch(
                &gem.orb_up,
                det_up_class,
                det_up_space,
                &eval_pos[..6 * n_up],
                CandidateEvent::Build,
            )
        ))
    } else {
        None
    };
    let det_dn_owned = if !cache_ok && det_dn_fast {
        Some(crate::timed!(
            "gfmc.kin_det_ao_dn",
            det_candidate_batch(
                &gem.orb_dn,
                det_dn_class,
                det_dn_space,
                &eval_pos[6 * n_up..],
                CandidateEvent::Build,
            )
        ))
    } else {
        None
    };
    let j3_owned = j3.and_then(|(orb, jm)| {
        if !cache_ok && j3_fast {
            Some((
                jm,
                crate::timed!(
                    "gfmc.kin_j3_ao",
                    j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3"),
                        eval_pos,
                        CandidateEvent::Build,
                    )
                ),
            ))
        } else {
            None
        }
    });
    let cached = cache_guard.as_ref().and_then(|slot| slot.as_ref());
    let det_up_aos = cached.map(|c| &c.det_up).or(det_up_owned.as_ref());
    let det_dn_aos = cached.map(|c| &c.det_dn).or(det_dn_owned.as_ref());
    let j12_up = cached.and_then(|c| c.j12_up.as_deref());
    let j12_dn = cached.and_then(|c| c.j12_dn.as_deref());
    let j3_aos = match (j3, cached) {
        (Some((_, jm)), Some(c)) => c.j3.as_ref().map(|aos| (jm, aos)),
        _ => j3_owned.as_ref().map(|(jm, aos)| (*jm, aos)),
    };
    let j3_delta = cached.and_then(|c| c.j3_delta.as_deref());
    let j3_active = cached.and_then(|c| c.j3_active.as_ref());
    if det_up_aos.is_some() && det_dn_aos.is_some() {
        st.ensure_det_wdot_cache(gem);
    }
    if j3_aos.is_some() && j3_delta.is_none() && !jastrow.precision_policy.is_f32("jastrow_ratio") {
        st.ensure_j3_wdot_cache(j3);
    }
    let det_wdot_guard = st.det_wdot_cache.borrow();
    let det_wdot = det_wdot_guard.as_ref();
    let j3_wdot_guard = st.j3_wdot_cache.borrow();
    let j3_wdot = j3_wdot_guard.as_ref();
    // Free closure over the moving spin's coords (higher-ranked borrow, same pattern as
    // the reference's `do_spin`): the two spin views have independent lifetimes.
    {
        let m_up = st.det_state.ao_up.nrows();
        let m_dn = st.det_state.ao_dn.nrows();
        let mut det_col = vec![0.0_f64; m_up.max(m_dn)];
        let mut det_delta = vec![0.0_f64; st.ginv.nrows()];
        let mut det_weight = vec![0.0_f64; gem.orb_up.ao_count().max(gem.orb_dn.ao_count())];
        let mut ratios = vec![0.0_f64; 6];
        let mut jas = vec![0.0_f64; 6];
        let mut j3_q = vec![0.0_f64; j3.map_or(0, |(orb, _)| orb.ao_count())];
        let mut do_spin = |is_up: bool, r: &ArrayView2<f64>| {
            for e in 0..r.nrows() {
                let global_e = if is_up { e } else { n_up + e };
                let newpos = &allpos[6 * global_e..6 * global_e + 6];
                crate::timed!(
                    "gfmc.kin_det_ratio",
                    match (if is_up { det_up_aos } else { det_dn_aos }, det_wdot) {
                        (Some(aos), Some(wdot)) => {
                            bump_route_counter(&GFMC_WINV_KIN_HITS);
                            let local = aos.slice(numpy::ndarray::s![6 * e..6 * e + 6, ..]);
                            let (weights, bases) = if is_up {
                                (&wdot.up, &wdot.up_base)
                            } else {
                                (&wdot.dn, &wdot.dn_base)
                            };
                            let row = weights.row(e);
                            candidate_wdot_into(
                                &local,
                                row.as_slice().expect("det weight row contiguous"),
                                1.0 - bases[e],
                                &mut ratios,
                            );
                        }
                        (Some(aos), None) => {
                            bump_route_counter(&GFMC_WINV_FALLBACKS);
                            let local = aos.slice(numpy::ndarray::s![6 * e..6 * e + 6, ..]);
                            let m = if is_up { m_up } else { m_dn };
                            st.det_ratios_from_candidate_aos_into(
                                gem,
                                is_up,
                                e,
                                &local,
                                &mut det_col[..m],
                                &mut det_delta,
                                &mut det_weight,
                                &mut ratios,
                            );
                        }
                        (None, _) => ratios = st.det_ratios(gem, is_up, e, newpos),
                    }
                );
                let j12 = if is_up { j12_up } else { j12_dn }.map(|v| &v[6 * e..6 * e + 6]);
                if j12_cache_ok {
                    if j12.is_some() && (j3_delta.is_some() || j3_aos.is_some() || j3.is_none()) {
                        bump_route_counter(&GFMC_J12_TABPIP_KIN_HITS);
                    } else {
                        bump_route_counter(&GFMC_J12_TABPIP_KIN_FALLBACKS);
                    }
                }
                crate::timed!(
                    "gfmc.kin_jas_ratio",
                    if let Some(table) = j3_delta {
                        st.jas_ratios_from_precomputed_j3_into(
                            jastrow,
                            r_up,
                            r_dn,
                            is_up,
                            e,
                            newpos,
                            j12,
                            &table[6 * global_e..6 * global_e + 6],
                            &mut jas,
                        );
                    } else {
                        match j3_aos {
                            Some((jm, aos)) => {
                                let local = aos
                                    .slice(numpy::ndarray::s![6 * global_e..6 * global_e + 6, ..]);
                                if let Some(wdot) = j3_wdot {
                                    let (weights, bases) = if is_up {
                                        (&wdot.up, &wdot.up_base)
                                    } else {
                                        (&wdot.dn, &wdot.dn_base)
                                    };
                                    let row = weights.row(e);
                                    st.jas_ratios_from_candidate_aos_into(
                                        jastrow,
                                        jm,
                                        r_up,
                                        r_dn,
                                        is_up,
                                        e,
                                        newpos,
                                        &local,
                                        &mut j3_q,
                                        j12,
                                        Some((
                                            row.as_slice().expect("J3 weight row contiguous"),
                                            bases[e],
                                        )),
                                        j3_active.map(|active| active[global_e].as_slice()),
                                        &mut jas,
                                    );
                                } else {
                                    st.jas_ratios_from_candidate_aos_into(
                                        jastrow, jm, r_up, r_dn, is_up, e, newpos, &local,
                                        &mut j3_q, j12, None, None, &mut jas,
                                    );
                                }
                            }
                            None if j3.is_none() && j12.is_some() => st
                                .jas_ratios_from_precomputed_j12_into(
                                    jastrow,
                                    r_up,
                                    r_dn,
                                    is_up,
                                    e,
                                    newpos,
                                    j12.expect("guarded J1/J2 candidate row"),
                                    &mut jas,
                                ),
                            None => jas = st.jas_ratios(jastrow, j3, r_up, r_dn, is_up, e, newpos),
                        }
                    }
                );
                for s in 0..6 {
                    out.push(inv_2a2 * jas[s] * ratios[s]);
                }
            }
        };
        crate::timed!("gfmc.kin_fold", {
            do_spin(true, r_up);
            do_spin(false, r_dn);
        });
    }
    out
}

/// FAST ECP T-move quadrature mesh: same candidate ORDER and per-candidate quantities as
/// [`crate::coulomb_potential::compute_ecp_nonlocal_mesh`] (`(is_up, i, new_pos,
/// ang * det_ratio, jastrow_ratio)`), but with the per-point FULL geminal + LU + full
/// Jastrow value replaced by the [`MeshFastState`] rank-1 machinery -- each quadrature
/// point moves exactly one electron, the identical move class as the kinetic mesh. The
/// `ang` (radial x Legendre x weight) factor is reproduced verbatim from the reference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ecp_nonlocal_mesh_fast(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    // ECP quadrature orientation (`grid_points @ rt`).
    rt: &[[f64; 3]; 3],
    // Independent six-point kinetic-lattice orientation.  The molecular
    // `(Nv + 6)` refresh batch uses this for its six kinetic candidates;
    // using `rt` here cross-wires the two caches when their streams differ.
    kinetic_rt: &[[f64; 3]; 3],
    nn: usize,
    dltmove: bool,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    st: &MeshFastState,
) -> (f64, f64, Vec<MeshCandidate>) {
    let nv = weights.len();
    let n_up = r_up.nrows();
    let inv_cell = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    let mic = |v: [f64; 3]| -> [f64; 3] {
        match (cell_nmax, &inv_cell) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image(v, cell, iv, n),
            _ => v,
        }
    };
    let mut rt_bits = [0_u64; 9];
    for a in 0..3 {
        for b in 0..3 {
            rt_bits[3 * a + b] = rt[a][b].to_bits();
        }
    }
    // Rotate the quadrature grid by rt (jqmc: grid_points @ RT). Identity when rt=I.
    let gp: Vec<[f64; 3]> = (0..nv)
        .map(|v| {
            let mut o = [0.0_f64; 3];
            for c in 0..3 {
                for k in 0..3 {
                    o[c] += grid_points[[v, k]] * rt[k][c];
                }
            }
            o
        })
        .collect();
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    // ECP table entries grouped per nucleus, in ascending table order (the
    // order the per-point channel sum below accumulates them).
    let terms_of: Vec<Vec<usize>> = {
        let n_nuc = uniq.last().map_or(0, |&m| m + 1);
        let mut by = vec![Vec::new(); n_nuc];
        for (t, &ia) in nucleus_index.iter().enumerate() {
            by[ia as usize].push(t);
        }
        by
    };
    let nearest_set = |rx: f64, ry: f64, rz: f64| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        if nn == 1 {
            // Stable-sort semantics of the general branch below: the FIRST
            // nucleus (ascending index) of minimal squared distance.
            let mut best: Option<(f64, usize)> = None;
            for &ia in &uniq {
                let dd = mic([
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ]);
                let d2 = dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2];
                best = match best {
                    Some((b, ib)) if d2.partial_cmp(&b).unwrap() != std::cmp::Ordering::Less => {
                        Some((b, ib))
                    }
                    _ => Some((d2, ia)),
                };
            }
            return best.map(|(_, ia)| vec![ia]).unwrap_or_default();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let dd = mic([
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ]);
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let radial_cut = st.ecp_radial_cut;
    let cut_counts = std::cell::Cell::new((0_u64, 0_u64));
    let nearest_count = if nn == 0 || nn >= uniq.len() {
        uniq.len()
    } else {
        nn
    };
    let rows_per_electron = nearest_count * nv;
    let build_electron =
        |r: &ArrayView2<f64>, i: usize, allpos: &mut Vec<[f64; 3]>, allangs: &mut Vec<f64>| {
            let near = nearest_set(r[[i, 0]], r[[i, 1]], r[[i, 2]]);
            for &ia in &near {
                let rel = mic([
                    positions[[ia, 0]] - r[[i, 0]],
                    positions[[ia, 1]] - r[[i, 1]],
                    positions[[ia, 2]] - r[[i, 2]],
                ]);
                let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
                let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
                // The radial channel factor v_l(|r - R|) is independent of
                // the quadrature point: evaluated once per (electron, nucleus)
                // with the identical expression, then replayed per point in
                // the same ascending-table order (bit-identical).
                let terms = &terms_of[ia];
                let mut v_ls = [0.0_f64; 8];
                let mut v_ls_heap = Vec::new();
                let v_l_of: &mut [f64] = if terms.len() <= v_ls.len() {
                    &mut v_ls[..terms.len()]
                } else {
                    v_ls_heap.resize(terms.len(), 0.0);
                    &mut v_ls_heap[..]
                };
                for (k, &t) in terms.iter().enumerate() {
                    let v_l = coefficients[t]
                        * crate::coulomb_potential::ecp_pow(nrm, powers[t] - 2.0)
                        * (-exponents[t] * nrm * nrm).exp();
                    // FAST radial cut (IMPL-RS-GFMC-FAST-ECP-RADIAL-CUT-001):
                    // the truncated Hamiltonian H' has v_t = 0 where
                    // |v_t| < EPS (the VMC FAST rule of
                    // `compute_ecp_nonlocal_parts_all_pairs`).
                    v_l_of[k] = if radial_cut > 0.0 && v_l.abs() < radial_cut {
                        cut_counts.set((cut_counts.get().0, cut_counts.get().1 + 1));
                        0.0
                    } else {
                        if radial_cut > 0.0 {
                            cut_counts.set((cut_counts.get().0 + 1, cut_counts.get().1));
                        }
                        v_l
                    };
                }
                for v in 0..nv {
                    let (dx, dy, dz) = (gp[v][0], gp[v][1], gp[v][2]);
                    let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                    let cosv = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                    let np = [
                        r[[i, 0]] + rx + nrm * dx,
                        r[[i, 1]] + ry + nrm * dy,
                        r[[i, 2]] + rz + nrm * dz,
                    ];
                    let mut ang = 0.0_f64;
                    for (k, &t) in terms.iter().enumerate() {
                        let l = ang_mom[t];
                        let v_l = v_l_of[k];
                        ang += v_l
                            * ((2 * l + 1) as f64)
                            * crate::coulomb_potential::legendre(l, cosv)
                            * weights[v];
                    }
                    allpos.push(np);
                    allangs.push(ang);
                }
            }
        };
    let build_spin = |r: &ArrayView2<f64>| {
        let mut allpos = Vec::with_capacity(r.nrows() * rows_per_electron);
        let mut allangs = Vec::with_capacity(r.nrows() * rows_per_electron);
        for i in 0..r.nrows() {
            build_electron(r, i, &mut allpos, &mut allangs);
        }
        (allpos, allangs)
    };
    let mut geometry_guard = st
        .reuse_fixed_mesh
        .then(|| st.ecp_geometry_cache.borrow_mut());
    let mut geometry_owned = None;
    if let Some(slot) = geometry_guard.as_mut() {
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.rt_bits != rt_bits
                || c.rows_per_electron != rows_per_electron
                || c.positions_up.len() != r_up.nrows() * rows_per_electron
                || c.positions_dn.len() != r_dn.nrows() * rows_per_electron
        });
        if rebuild {
            let ((positions_up, angular_up), (positions_dn, angular_dn)) =
                (build_spin(r_up), build_spin(r_dn));
            **slot = Some(EcpGeometryCache {
                rt_bits,
                rows_per_electron,
                positions_up,
                positions_dn,
                angular_up,
                angular_dn,
            });
        } else if let Some(change) = st.ecp_dirty.get() {
            let is_up = change.is_up;
            let e = change.electron;
            let mut positions_fresh = Vec::with_capacity(rows_per_electron);
            let mut angular_fresh = Vec::with_capacity(rows_per_electron);
            crate::timed!(
                "gfmc.ecp_geom_refresh",
                if is_up {
                    build_electron(r_up, e, &mut positions_fresh, &mut angular_fresh);
                } else {
                    build_electron(r_dn, e, &mut positions_fresh, &mut angular_fresh);
                }
            );
            debug_assert_eq!(positions_fresh.len(), rows_per_electron);
            let offset = e * rows_per_electron;
            let cache = slot.as_mut().expect("ECP geometry cache initialized");
            let (positions_cached, angular_cached) = if is_up {
                (&mut cache.positions_up, &mut cache.angular_up)
            } else {
                (&mut cache.positions_dn, &mut cache.angular_dn)
            };
            positions_cached[offset..offset + rows_per_electron].copy_from_slice(&positions_fresh);
            angular_cached[offset..offset + rows_per_electron].copy_from_slice(&angular_fresh);
        }
    } else {
        let (positions_up, angular_up) = build_spin(r_up);
        let (positions_dn, angular_dn) = build_spin(r_dn);
        geometry_owned = Some((positions_up, angular_up, positions_dn, angular_dn));
    }
    let (all_up, angular_up, all_dn, angular_dn): EcpMeshGeometry<'_> =
        if let Some(slot) = geometry_guard.as_ref() {
            let cache = slot.as_ref().expect("ECP geometry cache initialized");
            (
                &cache.positions_up,
                &cache.angular_up,
                &cache.positions_dn,
                &cache.angular_dn,
            )
        } else {
            let (positions_up, angular_up, positions_dn, angular_dn) = geometry_owned
                .as_ref()
                .expect("non-cached ECP geometry initialized");
            (positions_up, angular_up, positions_dn, angular_dn)
        };

    {
        let (kept, dropped) = cut_counts.take();
        bump_route_counter_by(&GFMC_ECP_CUT_KEPT_PAIRS, kept);
        bump_route_counter_by(&GFMC_ECP_CUT_DROPPED_PAIRS, dropped);
    }
    st.ensure_ecp_dtn_candidate_table(jastrow, r_up, r_dn, all_up, all_dn, rows_per_electron);

    // IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001: an electron whose every ECP
    // quadrature amplitude `ang` is exactly 0.0 (a zero-coefficient non-local
    // channel, such as the ccECP H placeholder, or full underflow) contributes
    // `a_m = 0 * ratio * jas = +-0` to FN, SP and the hop cdf, i.e. nothing:
    // adding +-0 to an accumulator that started at +0.0 changes no bit. Its
    // ratios are therefore not evaluated and its candidate-table rows are
    // neither built nor advanced (they are rebuilt when it moves, the only
    // event that can change its geometry before the next rotation rebuild).
    let live_of = |angs: &[f64], ne: usize| -> Vec<bool> {
        (0..ne)
            .map(|e| {
                !st.ecp_skip_zero_groups
                    || angs[e * rows_per_electron..(e + 1) * rows_per_electron]
                        .iter()
                        .any(|&x| x != 0.0)
            })
            .collect()
    };
    let live_up = live_of(angular_up, n_up);
    let live_dn = live_of(angular_dn, r_dn.nrows());
    let live_all: Vec<bool> = live_up.iter().chain(&live_dn).copied().collect();

    struct CandidateGroup<'a> {
        is_up: bool,
        electron: usize,
        spin_offset: usize,
        newpos: &'a [[f64; 3]],
        angs: &'a [f64],
    }
    let mut groups = Vec::<CandidateGroup>::with_capacity((n_up + r_dn.nrows()) * nearest_count);
    for electron in 0..n_up {
        for nearest in 0..nearest_count {
            let spin_offset = electron * rows_per_electron + nearest * nv;
            groups.push(CandidateGroup {
                is_up: true,
                electron,
                spin_offset,
                newpos: &all_up[spin_offset..spin_offset + nv],
                angs: &angular_up[spin_offset..spin_offset + nv],
            });
        }
    }
    for electron in 0..r_dn.nrows() {
        for nearest in 0..nearest_count {
            let spin_offset = electron * rows_per_electron + nearest * nv;
            groups.push(CandidateGroup {
                is_up: false,
                electron,
                spin_offset,
                newpos: &all_dn[spin_offset..spin_offset + nv],
                angs: &angular_dn[spin_offset..spin_offset + nv],
            });
        }
    }

    // AO exponentiation dominates this small molecular kernel. Evaluate every
    // ECP candidate of a spin in one AVX stream, instead of restarting a tiny
    // `nv`-point stream for every electron/nucleus pair.
    let cell_present = st.cell.is_some();
    let det_up_class = det_candidate_class(gem, &gem.orb_up, cell_present);
    let det_dn_class = det_candidate_class(gem, &gem.orb_dn, cell_present);
    let det_up_fast = det_up_class != CandidateBasisClass::Unsupported;
    let det_dn_fast = det_dn_class != CandidateBasisClass::Unsupported;
    let det_up_space = det_candidate_space(&gem.orb_up);
    let det_dn_space = det_candidate_space(&gem.orb_dn);
    if !det_up_fast || !det_dn_fast {
        bump_route_counter(&GFMC_CAND_DET_SLOW);
    }
    let j3_class = j3.map(|(orb, _)| j3_candidate_class(orb, cell_present));
    let j3_fast = j3.is_none_or(|_| {
        st.j3_state.is_some() && j3_class != Some(CandidateBasisClass::Unsupported)
    });
    if j3.is_some() && !j3_fast {
        note_j3_slow_fallback(&GFMC_J3_SLOW_FALLBACKS);
    }
    let j12_cache_ok = crate::jastrow_factor::j12_candidate_cache_ok(jastrow);
    let rows_up = rows_per_electron;
    let rows_dn = rows_per_electron;
    let cache_ok = st.reuse_fixed_mesh
        && det_up_fast
        && det_dn_fast
        && j3_fast
        && rows_up == rows_dn
        && rows_up * n_up == all_up.len()
        && rows_dn * r_dn.nrows() == all_dn.len();
    // One wrap site per mesh evaluation: candidate tables are
    // seeded/refreshed from home-cell coordinates; the geometry cache, J1/J2
    // deltas and DTN keep the raw quadrature positions.
    let eval_up_cow = wrapped_candidate_positions(st, all_up);
    let eval_dn_cow = wrapped_candidate_positions(st, all_dn);
    let eval_up: &[[f64; 3]] = &eval_up_cow;
    let eval_dn: &[[f64; 3]] = &eval_dn_cow;
    let mut cache_guard = cache_ok.then(|| st.ecp_cache.borrow_mut());
    if let Some(slot) = cache_guard.as_mut() {
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.rt_bits != rt_bits
                || c.rows_per_electron != rows_up
                || c.det_up.nrows() != all_up.len()
                || c.det_dn.nrows() != all_dn.len()
        });
        if rebuild {
            let det_up = crate::timed!(
                "gfmc.ecp_det_ao_up",
                live_rows_batch(eval_up, rows_up, mol_live(det_up_class, &live_up), |p| {
                    det_candidate_batch(
                        &gem.orb_up,
                        det_up_class,
                        det_up_space,
                        p,
                        CandidateEvent::Build,
                    )
                })
            );
            let det_dn = crate::timed!(
                "gfmc.ecp_det_ao_dn",
                live_rows_batch(eval_dn, rows_dn, mol_live(det_dn_class, &live_dn), |p| {
                    det_candidate_batch(
                        &gem.orb_dn,
                        det_dn_class,
                        det_dn_space,
                        p,
                        CandidateEvent::Build,
                    )
                })
            );
            let j3_cache = j3.map(|(orb, _)| {
                let class = j3_class.expect("J3 class computed alongside j3");
                crate::timed!("gfmc.ecp_j3_ao", {
                    let mut both = Vec::with_capacity(eval_up.len() + eval_dn.len());
                    both.extend_from_slice(eval_up);
                    both.extend_from_slice(eval_dn);
                    live_rows_batch(&both, rows_up, mol_live(class, &live_all), |p| {
                        j3_candidate_batch(orb, class, p, CandidateEvent::Build)
                    })
                })
            });
            let j3_active = j3_cache
                .as_ref()
                .map(|aos| candidate_active_columns(&aos.view(), rows_up));
            if let (Some(aos), Some(active)) = (&j3_cache, &j3_active) {
                report_candidate_active_once(
                    "ecp_j3",
                    active,
                    aos.ncols(),
                    &ECP_J3_ACTIVE_REPORTED,
                );
            }
            let j3_packed = match (&j3_cache, &j3_active) {
                (Some(aos), Some(active)) => {
                    Some(pack_candidate_groups(&aos.view(), rows_up, active.clone()))
                }
                _ => None,
            };
            let j3_delta = match (j3, st.j3_state.as_ref(), j3_cache.as_ref()) {
                (Some((_, jm)), Some(state), Some(aos)) if !state.f32_arithmetic => {
                    Some(crate::timed!(
                        "gfmc.ecp_j3_tabpip_build",
                        fill_j3_candidate_deltas(jm, state, &aos.view(), n_up, rows_up)
                    ))
                }
                _ => None,
            };
            let (j12_up, j12_dn) = if j12_cache_ok {
                let mut up = vec![0.0; all_up.len()];
                let mut dn = vec![0.0; all_dn.len()];
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow, r_up, r_dn, true, all_up, rows_up, &mut up,
                );
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow, r_up, r_dn, false, all_dn, rows_dn, &mut dn,
                );
                (Some(up), Some(dn))
            } else {
                (None, None)
            };
            if j12_cache_ok {
                bump_route_counter(&GFMC_J12_TABPIP_ECP_BUILDS);
            }
            **slot = Some(EcpAoCache {
                rt_bits,
                rows_per_electron: rows_up,
                det_up,
                det_dn,
                j3: j3_cache,
                j3_active,
                j3_packed,
                j3_delta,
                j12_up,
                j12_dn,
            });
            st.ecp_dirty.set(None);
        } else if let Some(change) = st.ecp_dirty.take() {
            let is_up = change.is_up;
            let e = change.electron;
            // Refresh rows come from the SAME wrapped coordinates (and the
            // same kernel family) the table was seeded from.
            let spin_pos = if is_up { eval_up } else { eval_dn };
            let off = rows_up * e;
            let pos = &spin_pos[off..off + rows_up];
            let live_e = if is_up { live_up[e] } else { live_dn[e] };
            let cache = slot.as_mut().expect("ECP cache initialized");
            let combined = combined_molecular_det_refresh(
                gem,
                is_up,
                e,
                r_up,
                r_dn,
                alat,
                kinetic_rt,
                if live_e { pos } else { &pos[..0] },
                if is_up { det_up_class } else { det_dn_class },
                if is_up { det_up_space } else { det_dn_space },
            );
            if !live_e {
                // Dead electron: its ECP rows are never read.
            } else if is_up {
                let fresh = match combined.as_ref() {
                    Some((_, ecp)) => ecp.to_owned(),
                    None => crate::timed!(
                        "gfmc.ecp_det_ao_up",
                        det_candidate_batch(
                            &gem.orb_up,
                            det_up_class,
                            det_up_space,
                            pos,
                            CandidateEvent::Refresh
                        )
                    ),
                };
                cache
                    .det_up
                    .slice_mut(numpy::ndarray::s![off..off + rows_up, ..])
                    .assign(&fresh);
            } else {
                let fresh = match combined.as_ref() {
                    Some((_, ecp)) => ecp.to_owned(),
                    None => crate::timed!(
                        "gfmc.ecp_det_ao_dn",
                        det_candidate_batch(
                            &gem.orb_dn,
                            det_dn_class,
                            det_dn_space,
                            pos,
                            CandidateEvent::Refresh
                        )
                    ),
                };
                cache
                    .det_dn
                    .slice_mut(numpy::ndarray::s![off..off + rows_up, ..])
                    .assign(&fresh);
            }
            if let Some((kinetic, _)) = combined {
                *st.kinetic_det_prefetch.borrow_mut() = Some(DetCandidatePrefetch {
                    is_up,
                    electron: e,
                    kinetic,
                });
            }
            if let (Some(_), false) = (j3, live_e) {
                let global_e = if is_up { e } else { n_up + e };
                if let Some(active) = cache.j3_active.as_mut() {
                    active[global_e] = Vec::new();
                }
                if let Some(packed) = cache.j3_packed.as_mut() {
                    packed[global_e] = PackedCandidateGroup {
                        columns: Vec::new(),
                        values: Vec::new(),
                        rows: rows_up,
                    };
                }
            } else if let Some((orb, _)) = j3 {
                let global_off = if is_up { off } else { all_up.len() + off };
                let fresh = crate::timed!(
                    "gfmc.ecp_j3_ao",
                    j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3"),
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                if let Some(aos) = cache.j3.as_mut() {
                    aos.slice_mut(numpy::ndarray::s![global_off..global_off + rows_up, ..])
                        .assign(&fresh);
                }
                let global_e = if is_up { e } else { n_up + e };
                let columns = candidate_active_columns(&fresh.view(), rows_up)
                    .pop()
                    .expect("one refreshed ECP J3 candidate group");
                if let Some(active) = cache.j3_active.as_mut() {
                    active[global_e] = columns.clone();
                }
                if let Some(packed) = cache.j3_packed.as_mut() {
                    packed[global_e] = pack_candidate_group(&fresh.view(), 0, rows_up, columns);
                }
            }
            if let (Some((_, jm)), Some(state), Some(pending)) = (
                j3,
                st.j3_state.as_ref(),
                st.j3_pending_delta.borrow().as_ref(),
            ) {
                if let (Some(packed), Some(table)) = (&cache.j3_packed, &mut cache.j3_delta) {
                    crate::timed!(
                        "gfmc.ecp_j3_tabpip_advance",
                        advance_j3_candidate_deltas(
                            jm,
                            state,
                            n_up,
                            rows_up,
                            change.is_up,
                            change.electron,
                            &pending.d,
                            &pending.current_up,
                            &pending.current_dn,
                            packed,
                            table,
                            Some(&live_all),
                        )
                    );
                }
            }
            if let (Some(up), Some(dn)) = (&mut cache.j12_up, &mut cache.j12_dn) {
                crate::timed!(
                    "gfmc.ecp_j12_advance",
                    crate::jastrow_factor::update_j12_candidate_deltas_after_hop_live(
                        jastrow,
                        r_up,
                        r_dn,
                        change.is_up,
                        change.electron,
                        change.old_pos,
                        all_up,
                        all_dn,
                        rows_up,
                        up,
                        dn,
                        Some((&live_up, &live_dn)),
                    )
                );
                bump_route_counter(&GFMC_J12_TABPIP_ECP_ADVANCES);
            }
        }
    }

    let det_up_owned = if !cache_ok && det_up_fast {
        Some(crate::timed!(
            "gfmc.ecp_det_ao_up",
            live_rows_batch(eval_up, rows_up, mol_live(det_up_class, &live_up), |p| {
                det_candidate_batch(
                    &gem.orb_up,
                    det_up_class,
                    det_up_space,
                    p,
                    CandidateEvent::Build,
                )
            })
        ))
    } else {
        None
    };
    let det_dn_owned = if !cache_ok && det_dn_fast {
        Some(crate::timed!(
            "gfmc.ecp_det_ao_dn",
            live_rows_batch(eval_dn, rows_dn, mol_live(det_dn_class, &live_dn), |p| {
                det_candidate_batch(
                    &gem.orb_dn,
                    det_dn_class,
                    det_dn_space,
                    p,
                    CandidateEvent::Build,
                )
            })
        ))
    } else {
        None
    };
    let j3_owned = j3.and_then(|(orb, jm)| {
        if !cache_ok && j3_fast {
            Some((
                jm,
                crate::timed!(
                    "gfmc.ecp_j3_ao",
                    j3_candidate_batch_two_spins(
                        orb,
                        j3_class.expect("J3 class computed alongside j3"),
                        eval_up,
                        eval_dn,
                        CandidateEvent::Build,
                    )
                ),
            ))
        } else {
            None
        }
    });
    let cached = cache_guard.as_ref().and_then(|slot| slot.as_ref());
    let det_up_aos = cached.map(|c| &c.det_up).or(det_up_owned.as_ref());
    let det_dn_aos = cached.map(|c| &c.det_dn).or(det_dn_owned.as_ref());
    let j12_up = cached.and_then(|c| c.j12_up.as_deref());
    let j12_dn = cached.and_then(|c| c.j12_dn.as_deref());
    let j3_aos = match (j3, cached) {
        (Some((_, jm)), Some(c)) => c.j3.as_ref().map(|aos| (jm, aos)),
        _ => j3_owned.as_ref().map(|(jm, aos)| (*jm, aos)),
    };
    let j3_delta = cached.and_then(|c| c.j3_delta.as_deref());

    if det_up_aos.is_some() && det_dn_aos.is_some() {
        st.ensure_det_wdot_cache(gem);
    }
    if j3_aos.is_some() && j3_delta.is_none() && !jastrow.precision_policy.is_f32("jastrow_ratio") {
        st.ensure_j3_wdot_cache(j3);
    }
    let det_wdot_guard = st.det_wdot_cache.borrow();
    let det_wdot = det_wdot_guard.as_ref();
    let j3_wdot_guard = st.j3_wdot_cache.borrow();
    let j3_wdot = j3_wdot_guard.as_ref();

    // Fold the ECP amplitudes directly into the production representation.
    // Returning `(det_amp, jastrow_ratio)` and immediately consuming it in
    // `gfmc_lrdmc_ecp_state` would allocate and write a second ~100 KiB
    // candidate vector on every hop for 06water.
    let mut fn_ecp = 0.0_f64;
    let mut sp_ecp = 0.0_f64;
    let mut out: Vec<MeshCandidate> = Vec::with_capacity(all_up.len() + all_dn.len());
    let m_up = st.det_state.ao_up.nrows();
    let m_dn = st.det_state.ao_dn.nrows();
    let mut det_col = vec![0.0_f64; m_up.max(m_dn)];
    let mut det_delta = vec![0.0_f64; st.ginv.nrows()];
    let mut det_weight = vec![0.0_f64; gem.orb_up.ao_count().max(gem.orb_dn.ao_count())];
    let mut ratios = vec![0.0_f64; nv];
    let mut jas = vec![0.0_f64; nv];
    let mut j3_q = vec![0.0_f64; j3.map_or(0, |(orb, _)| orb.ao_count())];
    let _fold_timer = crate::prof::on().then(std::time::Instant::now);
    for group in groups {
        let npts = group.newpos.len();
        let live = if group.is_up {
            live_up[group.electron]
        } else {
            live_dn[group.electron]
        };
        if !live {
            bump_route_counter(&GFMC_ECP_ZERO_GROUPS);
            for v in 0..npts {
                out.push((group.is_up, group.electron, group.newpos[v], 0.0));
            }
            continue;
        }
        bump_route_counter(&GFMC_ECP_LIVE_GROUPS);
        let det_aos = if group.is_up { det_up_aos } else { det_dn_aos };
        crate::timed!(
            "gfmc.ecp_det_ratio",
            match (det_aos, det_wdot) {
                (Some(aos), Some(wdot)) => {
                    bump_route_counter(&GFMC_WINV_ECP_HITS);
                    let local = aos.slice(numpy::ndarray::s![
                        group.spin_offset..group.spin_offset + npts,
                        ..
                    ]);
                    let (weights, bases) = if group.is_up {
                        (&wdot.up, &wdot.up_base)
                    } else {
                        (&wdot.dn, &wdot.dn_base)
                    };
                    let row = weights.row(group.electron);
                    candidate_wdot_into(
                        &local,
                        row.as_slice().expect("det weight row contiguous"),
                        1.0 - bases[group.electron],
                        &mut ratios[..npts],
                    );
                }
                (Some(aos), None) => {
                    bump_route_counter(&GFMC_WINV_FALLBACKS);
                    let local = aos.slice(numpy::ndarray::s![
                        group.spin_offset..group.spin_offset + npts,
                        ..
                    ]);
                    let m = if group.is_up { m_up } else { m_dn };
                    st.det_ratios_from_candidate_aos_into(
                        gem,
                        group.is_up,
                        group.electron,
                        &local,
                        &mut det_col[..m],
                        &mut det_delta,
                        &mut det_weight,
                        &mut ratios[..npts],
                    );
                }
                (None, _) => {
                    ratios = st.det_ratios(gem, group.is_up, group.electron, group.newpos)
                }
            }
        );
        let j12 = if group.is_up { j12_up } else { j12_dn }
            .map(|v| &v[group.spin_offset..group.spin_offset + npts]);
        if j12_cache_ok {
            if j12.is_some() && (j3_delta.is_some() || j3_aos.is_some() || j3.is_none()) {
                bump_route_counter(&GFMC_J12_TABPIP_ECP_HITS);
            } else {
                bump_route_counter(&GFMC_J12_TABPIP_ECP_FALLBACKS);
            }
        }
        crate::timed!(
            "gfmc.ecp_jas_ratio",
            if let Some(table) = j3_delta {
                let offset = if group.is_up {
                    group.spin_offset
                } else {
                    all_up.len() + group.spin_offset
                };
                st.jas_ratios_from_precomputed_j3_into(
                    jastrow,
                    r_up,
                    r_dn,
                    group.is_up,
                    group.electron,
                    group.newpos,
                    j12,
                    &table[offset..offset + npts],
                    &mut jas[..npts],
                );
            } else {
                match j3_aos {
                    Some((jm, aos)) => {
                        let offset = if group.is_up {
                            group.spin_offset
                        } else {
                            all_up.len() + group.spin_offset
                        };
                        let local = aos.slice(numpy::ndarray::s![offset..offset + npts, ..]);
                        if let Some(wdot) = j3_wdot {
                            let (weights, bases) = if group.is_up {
                                (&wdot.up, &wdot.up_base)
                            } else {
                                (&wdot.dn, &wdot.dn_base)
                            };
                            let row = weights.row(group.electron);
                            st.jas_ratios_from_candidate_aos_into(
                                jastrow,
                                jm,
                                r_up,
                                r_dn,
                                group.is_up,
                                group.electron,
                                group.newpos,
                                &local,
                                &mut j3_q,
                                j12,
                                Some((
                                    row.as_slice().expect("J3 weight row contiguous"),
                                    bases[group.electron],
                                )),
                                None,
                                &mut jas[..npts],
                            );
                        } else {
                            st.jas_ratios_from_candidate_aos_into(
                                jastrow,
                                jm,
                                r_up,
                                r_dn,
                                group.is_up,
                                group.electron,
                                group.newpos,
                                &local,
                                &mut j3_q,
                                j12,
                                None,
                                None,
                                &mut jas[..npts],
                            );
                        }
                    }
                    None if j3.is_none() && j12.is_some() => st
                        .jas_ratios_from_precomputed_j12_into(
                            jastrow,
                            r_up,
                            r_dn,
                            group.is_up,
                            group.electron,
                            group.newpos,
                            j12.expect("guarded J1/J2 ECP candidate row"),
                            &mut jas[..npts],
                        ),
                    None => {
                        jas = st.jas_ratios(
                            jastrow,
                            j3,
                            r_up,
                            r_dn,
                            group.is_up,
                            group.electron,
                            group.newpos,
                        )
                    }
                }
            }
        );
        for v in 0..npts {
            let det_amp = group.angs[v] * ratios[v];
            let a_m = if dltmove {
                sp_ecp += nan_max(det_amp, 0.0);
                nan_min(det_amp, 0.0) * jas[v]
            } else {
                let vfull = det_amp * jas[v];
                sp_ecp += nan_max(vfull, 0.0);
                nan_min(vfull, 0.0)
            };
            fn_ecp += a_m;
            out.push((group.is_up, group.electron, group.newpos[v], a_m));
        }
    }
    if let Some(t0) = _fold_timer {
        crate::prof::add("gfmc.ecp_fold", t0.elapsed().as_nanos());
    }
    (fn_ecp, sp_ecp, out)
}

/// LRDMC all-electron diagonal local energy `e_L` + off-diagonal rate `b_L`
/// (`non_diagonal_sum_hamiltonian`, the FN kinetic sum). 1:1 jqmc_gfmc._projection_t_core
/// (all-electron branch, no ECP): diag kinetic 3/a^2 N + ZV-regularized Coulomb + SP + FN.
/// Returns (e_L, b_L, elements) -- elements are the raw hopping amplitudes for the move.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_lrdmc_ae(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    pbc: Option<&crate::ewald::EwaldRef>,
    fast: bool,
) -> (f64, f64, f64, Vec<f64>) {
    let (e_l, e_l_meas, b_l, elements, _) = gfmc_lrdmc_ae_state(
        gem,
        jastrow,
        j3,
        positions,
        eff_charges,
        r_up,
        r_dn,
        alat,
        rt,
        pbc,
        fast,
        None,
        None,
    );
    (e_l, e_l_meas, b_l, elements)
}

#[allow(clippy::too_many_arguments)]
fn gfmc_lrdmc_ae_state(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    pbc: Option<&crate::ewald::EwaldRef>,
    fast: bool,
    state_in: Option<&MeshFastState>,
    pot: Option<&IntervalPotState>,
) -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n = n_up + n_dn;
    // FAST chain mode: Sherman-Morrison rank-1 kinetic elements off the shared base
    // state (per-member fall-back on a coverage-gate miss; the state itself always
    // exists in fast mode).
    let built_state = crate::timed!(
        "gfmc.base_state",
        if fast && state_in.is_none() {
            Some(mesh_fast_state(gem, jastrow, j3, r_up, r_dn, pbc))
        } else {
            None
        }
    );
    let fast_state = state_in.or(built_state.as_ref());
    let elements = crate::timed!(
        "gfmc.kin_mesh",
        match fast_state {
            Some(st) => compute_discretized_kinetic_elements_fast(
                gem, jastrow, j3, r_up, r_dn, alat, rt, st,
            ),
            None => compute_discretized_kinetic_elements(gem, jastrow, j3, r_up, r_dn, alat, rt),
        }
    );
    // FN/SP split + per-electron nondiag & sign-flip
    let mut fn_sum = 0.0;
    let mut sp = 0.0;
    for &el in &elements {
        fn_sum += nan_min(el, 0.0);
        sp += nan_max(el, 0.0);
    }
    let mut nondiag_kin = vec![0.0_f64; n]; // per electron: Sum_6 (el + 1/(2a^2))
    let mut sign_flip = vec![false; n];
    let q = 1.0 / (2.0 * alat * alat);
    for e in 0..n {
        for s in 0..6 {
            let el = elements[e * 6 + s];
            nondiag_kin[e] += el + q;
            if el >= 0.0 {
                sign_flip[e] = true;
            }
        }
    }
    // continuum per-electron kinetic KE_i = -1/2(grad^2lnPsi_i + |gradlnPsi_i|^2)
    let (gu, gd, lu, ld) = crate::timed!(
        "gfmc.det_gradlap",
        match fast_state {
            Some(st) if fast_det_gradlap_ok(gem, pbc) => st.det_grad_lap(gem, r_up, r_dn),
            _ => {
                bump_route_counter(&GFMC_DET_GRAD_SLOW);
                crate::determinant::compute_grads_and_laplacian_ln_det(gem, r_up, r_dn)
            }
        }
    );
    let (gju, gjd, lju, ljd) = crate::timed!(
        "gfmc.j12_gradlap",
        match fast_state {
            Some(st) if crate::jastrow_factor::j12_generic_cache_ok(jastrow) => {
                st.j12_grad_lap(jastrow, r_up, r_dn)
            }
            _ => {
                bump_route_counter(&GFMC_J12_GRAD_SLOW);
                // The full Jastrow rebuild also evaluates DTN, so record its
                // carried table as unused when no compatible incremental
                // J1/J2 state is available.
                if jastrow.dtn.is_some() {
                    bump_route_counter(&GFMC_DTN_GRAD_SLOW);
                }
                crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(jastrow, r_up, r_dn)
            }
        }
    );
    let (g3u, g3d, l3u, l3d) = crate::timed!(
        "gfmc.j3_gradlap",
        match j3 {
            Some((orb, jm)) => match fast_state {
                // The state existing does NOT imply the fast J3 path: the J3
                // streaming state is per-member gated (None for MO/periodic/
                // non-contiguous J3), so require it explicitly.
                Some(st) if st.j3_state.is_some() => st.j3_grad_lap(orb, jm, r_up, r_dn),
                _ => {
                    bump_route_counter(&GFMC_J3_GRAD_SLOW);
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_three_body(
                        orb, jm, r_up, r_dn,
                    )
                }
            },
            None => (
                Array2::zeros((n_up, 3)),
                Array2::zeros((n_dn, 3)),
                numpy::ndarray::Array1::zeros(n_up),
                numpy::ndarray::Array1::zeros(n_dn),
            ),
        }
    );
    let ke_cont = |i: usize, up: bool| -> f64 {
        let (g, gj, g3, l, lj, l3) = if up {
            (&gu, &gju, &g3u, &lu, &lju, &l3u)
        } else {
            (&gd, &gjd, &g3d, &ld, &ljd, &l3d)
        };
        let mut gnorm2 = 0.0;
        for c in 0..3 {
            let gc = g[[i, c]] + gj[[i, c]] + g3[[i, c]];
            gnorm2 += gc * gc;
        }
        -0.5 * (l[i] + lj[i] + l3[i] + gnorm2)
    };
    // Per-electron el-ion (bare, discretized): molecular -Z/|r-R| vs periodic Ewald ei.
    let (ei_bare, ei_disc): (Vec<f64>, Vec<f64>) = if let Some(ps) = pot {
        (
            ps.ei_bare.clone(),
            ps.ei_disc
                .clone()
                .expect("AE interval potential state carries discretized ei rows"),
        )
    } else {
        match pbc {
            None => {
                let mol = |re: [f64; 3]| -> (f64, f64) {
                    let (mut bare, mut disc) = (0.0, 0.0);
                    for ia in 0..positions.nrows() {
                        let (dx, dy, dz) = (
                            re[0] - positions[[ia, 0]],
                            re[1] - positions[[ia, 1]],
                            re[2] - positions[[ia, 2]],
                        );
                        let d = (dx * dx + dy * dy + dz * dz).sqrt();
                        bare += -eff_charges[ia] / d;
                        disc += -eff_charges[ia] / nan_max(d, alat);
                    }
                    (bare, disc)
                };
                let mut b = vec![0.0; n];
                let mut d = vec![0.0; n];
                for (spin_up, nr) in [(true, n_up), (false, n_dn)] {
                    for i in 0..nr {
                        let re = if spin_up {
                            [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
                        } else {
                            [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
                        };
                        let gi = if spin_up { i } else { n_up + i };
                        let (bb, dd) = mol(re);
                        b[gi] = bb;
                        d[gi] = dd;
                    }
                }
                (b, d)
            }
            Some(p) => {
                let bare = crate::ewald::compute_ewald_potential_ei_per_electron(
                    r_up,
                    r_dn,
                    p.eta,
                    &p.real_images,
                    &p.g_vectors,
                    p.g_factors,
                    &p.cell,
                    p.n_max,
                    eff_charges,
                    positions,
                    None,
                );
                let disc = crate::ewald::compute_ewald_potential_ei_per_electron(
                    r_up,
                    r_dn,
                    p.eta,
                    &p.real_images,
                    &p.g_vectors,
                    p.g_factors,
                    &p.cell,
                    p.n_max,
                    eff_charges,
                    positions,
                    Some(alat),
                );
                (bare, disc)
            }
        }
    };
    // ZV composition per electron (AE: ei channel of the sign-flip max = discretized).
    // `meas_ei_tc` accumulates the MEASUREMENT (novar=0) per-electron pieces:
    // bare el-ion + continuum kinetic, no lattice hop terms, no max clip.
    let mut coul_ei = 0.0;
    let mut meas_ei_tc = 0.0;
    let mut sprime_mask = vec![0.0_f64; n];
    for (spin_up, nr) in [(true, n_up), (false, n_dn)] {
        for i in 0..nr {
            let gi = if spin_up { i } else { n_up + i };
            let tc = ke_cont(i, spin_up);
            let zv = ei_bare[gi] + tc - nondiag_kin[gi];
            let opt = if sign_flip[gi] {
                nan_max(zv, ei_disc[gi])
            } else {
                zv
            };
            if sign_flip[gi] && ei_disc[gi] > zv {
                sprime_mask[gi] = 1.0;
            }
            coul_ei += opt;
            meas_ei_tc += ei_bare[gi] + tc;
        }
    }
    let (el_el, ion_ion) = if let Some(ps) = pot {
        let ee = match pbc {
            Some(p) => ps
                .ee
                .as_ref()
                .expect("PBC potential state carries Ewald ee")
                .energy(p.g_factors),
            None => ps
                .ee_molecular
                .expect("molecular potential state carries bare ee"),
        };
        (ee, ps.ion_ion)
    } else {
        match pbc {
            None => (
                crate::coulomb_potential::compute_bare_coulomb_potential_el_el(r_up, r_dn),
                crate::coulomb_potential::compute_bare_coulomb_potential_ion_ion(
                    positions,
                    eff_charges,
                ),
            ),
            Some(p) => (
                crate::ewald::compute_ewald_potential_ee(
                    r_up,
                    r_dn,
                    p.eta,
                    &p.real_images,
                    &p.g_vectors,
                    p.g_factors,
                    &p.cell,
                    p.n_max,
                    None,
                ),
                p.ii,
            ),
        }
    };
    let diag_kin = 3.0 / (alat * alat) * (n as f64);
    let e_l = diag_kin + (el_el + ion_ion + coul_ei) + sp + fn_sum;
    // Measurement local energy (novar=0 split, cf. TurboRVB enertrue): the
    // CONTINUUM local energy. Equals `e_l` on non-clipped configurations by
    // the zero-variance identity; `e_l` keeps feeding the propagation weight.
    let e_l_meas = el_el + ion_ion + meas_ei_tc;
    let b_l = fn_sum; // non_diagonal_sum_hamiltonian (AE, no ECP)
    (e_l, e_l_meas, b_l, elements, sprime_mask)
}

/// LRDMC/GFMC local energy for the ECP (Casula T-move) case. Adds to the AE kinetic
/// machinery: (a) the el-ion ZV max-arg uses the BARE point-charge el-ion (parcutg=1,
/// no alat truncation for ECP -- jqmc_gfmc.py), (b) the local ECP radial part, and
/// (c) the non-local tmove mesh (per-candidate V_nonlocal, FN/SP split feeding e_L and
/// the move). Returns (e_L, b_L, kinetic_elements, ecp_candidates) where each candidate
/// is `(is_up, elec_index, new_pos, V_nonlocal)`.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_lrdmc_ecp(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    pbc: Option<&crate::ewald::EwaldRef>,
    fast: bool,
) -> (f64, f64, f64, Vec<f64>, Vec<MeshCandidate>) {
    let (e_l, e_l_meas, b_l, elements, cands, _) = gfmc_lrdmc_ecp_state(
        gem,
        jastrow,
        j3,
        positions,
        eff_charges,
        r_up,
        r_dn,
        alat,
        rt,
        rt,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        pbc,
        fast,
        None,
        None,
    );
    (e_l, e_l_meas, b_l, elements, cands)
}

#[allow(clippy::too_many_arguments)]
fn gfmc_lrdmc_ecp_state(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    pbc: Option<&crate::ewald::EwaldRef>,
    fast: bool,
    state_in: Option<&MeshFastState>,
    pot: Option<&IntervalPotState>,
) -> IntervalEvalReal {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n = n_up + n_dn;
    // FAST chain mode: ONE shared Sherman-Morrison base state serves both
    // one-electron-move meshes of this sub-step (kinetic hops + ECP T-move quadrature).
    let built_state = crate::timed!(
        "gfmc.base_state",
        if fast && state_in.is_none() {
            Some(mesh_fast_state(gem, jastrow, j3, r_up, r_dn, pbc))
        } else {
            None
        }
    );
    let fast_state = state_in.or(built_state.as_ref());
    let cell_nmax: Option<(&ArrayView2<f64>, i32)> = pbc.map(|p| (&p.cell, p.n_max));
    // Populate the ECP cache before the kinetic cache so the moved electron's
    // `(Nv + 6)` determinant candidates can share one AO/MO batch.  The mesh
    // values themselves are folded into e_L in their original arithmetic
    // position below.
    let (fast_ecp, mesh) = crate::timed!(
        "gfmc.one_move_mesh",
        match fast_state {
            Some(st) => (
                Some(compute_ecp_nonlocal_mesh_fast(
                    gem,
                    jastrow,
                    j3,
                    r_up,
                    r_dn,
                    alat,
                    positions,
                    nuc_n,
                    am,
                    e_n,
                    c_n,
                    p_n,
                    weights,
                    grid_points,
                    ecp_rt,
                    kinetic_rt,
                    nn,
                    dltmove,
                    cell_nmax,
                    st,
                )),
                None,
            ),
            None => (
                None,
                Some(crate::coulomb_potential::compute_ecp_nonlocal_mesh(
                    gem,
                    jastrow,
                    r_up,
                    r_dn,
                    positions,
                    nuc_n,
                    am,
                    e_n,
                    c_n,
                    p_n,
                    weights,
                    grid_points,
                    ecp_rt,
                    nn,
                    j3,
                    cell_nmax,
                )),
            ),
        }
    );
    let elements = crate::timed!(
        "gfmc.kin_mesh",
        match fast_state {
            Some(st) => compute_discretized_kinetic_elements_fast(
                gem, jastrow, j3, r_up, r_dn, alat, kinetic_rt, st,
            ),
            None =>
                compute_discretized_kinetic_elements(gem, jastrow, j3, r_up, r_dn, alat, kinetic_rt),
        }
    );
    let mut fn_sum = 0.0;
    let mut sp = 0.0;
    for &el in &elements {
        fn_sum += nan_min(el, 0.0);
        sp += nan_max(el, 0.0);
    }
    let mut nondiag_kin = vec![0.0_f64; n];
    let mut sign_flip = vec![false; n];
    let q = 1.0 / (2.0 * alat * alat);
    for e in 0..n {
        for s in 0..6 {
            let el = elements[e * 6 + s];
            nondiag_kin[e] += el + q;
            if el >= 0.0 {
                sign_flip[e] = true;
            }
        }
    }
    let (gu, gd, lu, ld) = crate::timed!(
        "gfmc.det_gradlap",
        match fast_state {
            Some(st) if fast_det_gradlap_ok(gem, pbc) => st.det_grad_lap(gem, r_up, r_dn),
            _ => {
                bump_route_counter(&GFMC_DET_GRAD_SLOW);
                crate::determinant::compute_grads_and_laplacian_ln_det(gem, r_up, r_dn)
            }
        }
    );
    let (gju, gjd, lju, ljd) = crate::timed!(
        "gfmc.j12_gradlap",
        match fast_state {
            Some(st) if crate::jastrow_factor::j12_generic_cache_ok(jastrow) => {
                st.j12_grad_lap(jastrow, r_up, r_dn)
            }
            _ => {
                bump_route_counter(&GFMC_J12_GRAD_SLOW);
                // The full Jastrow rebuild also evaluates DTN, so record its
                // carried table as unused when no compatible incremental
                // J1/J2 state is available.
                if jastrow.dtn.is_some() {
                    bump_route_counter(&GFMC_DTN_GRAD_SLOW);
                }
                crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(jastrow, r_up, r_dn)
            }
        }
    );
    let (g3u, g3d, l3u, l3d) = crate::timed!(
        "gfmc.j3_gradlap",
        match j3 {
            Some((orb, jm)) => match fast_state {
                // The state existing does NOT imply the fast J3 path: the J3
                // streaming state is per-member gated (None for MO/periodic/
                // non-contiguous J3), so require it explicitly.
                Some(st) if st.j3_state.is_some() => st.j3_grad_lap(orb, jm, r_up, r_dn),
                _ => {
                    bump_route_counter(&GFMC_J3_GRAD_SLOW);
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_three_body(
                        orb, jm, r_up, r_dn,
                    )
                }
            },
            None => (
                Array2::zeros((n_up, 3)),
                Array2::zeros((n_dn, 3)),
                numpy::ndarray::Array1::zeros(n_up),
                numpy::ndarray::Array1::zeros(n_dn),
            ),
        }
    );
    let ke_cont = |i: usize, up: bool| -> f64 {
        let (g, gj, g3, l, lj, l3) = if up {
            (&gu, &gju, &g3u, &lu, &lju, &l3u)
        } else {
            (&gd, &gjd, &g3d, &ld, &ljd, &l3d)
        };
        let mut gnorm2 = 0.0;
        for c in 0..3 {
            let gc = g[[i, c]] + gj[[i, c]] + g3[[i, c]];
            gnorm2 += gc * gc;
        }
        -0.5 * (l[i] + lj[i] + l3[i] + gnorm2)
    };
    // el-ion PER ELECTRON (ECP: parcutg=1 => ZV max-arg = BARE, no alat floor). Molecular
    // -Z/|r-R| vs periodic Ewald per-electron ei (bare, same as AE but max-arg stays bare).
    let ei_bare_v: Vec<f64> = crate::timed!(
        "gfmc.coul_ei",
        if let Some(ps) = pot {
            ps.ei_bare.clone()
        } else {
            match pbc {
                None => {
                    let mut v = vec![0.0; n];
                    for (spin_up, nr) in [(true, n_up), (false, n_dn)] {
                        for i in 0..nr {
                            let re = if spin_up {
                                [r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]
                            } else {
                                [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
                            };
                            let mut bare = 0.0;
                            for ia in 0..positions.nrows() {
                                let (dx, dy, dz) = (
                                    re[0] - positions[[ia, 0]],
                                    re[1] - positions[[ia, 1]],
                                    re[2] - positions[[ia, 2]],
                                );
                                bare += -eff_charges[ia] / (dx * dx + dy * dy + dz * dz).sqrt();
                            }
                            v[if spin_up { i } else { n_up + i }] = bare;
                        }
                    }
                    v
                }
                Some(p) => crate::ewald::compute_ewald_potential_ei_per_electron(
                    r_up,
                    r_dn,
                    p.eta,
                    &p.real_images,
                    &p.g_vectors,
                    p.g_factors,
                    &p.cell,
                    p.n_max,
                    eff_charges,
                    positions,
                    None,
                ),
            }
        }
    );
    let mut coul_ei = 0.0;
    let mut sprime_mask = vec![0.0_f64; n];
    // `meas_ei_tc`: MEASUREMENT (novar=0) per-electron pieces -- bare el-ion +
    // continuum kinetic, no lattice hop terms, no max clip.
    let mut meas_ei_tc = 0.0;
    for (spin_up, nr) in [(true, n_up), (false, n_dn)] {
        for i in 0..nr {
            let gi = if spin_up { i } else { n_up + i };
            let bare = ei_bare_v[gi];
            let tc = ke_cont(i, spin_up);
            let zv = bare + tc - nondiag_kin[gi];
            let opt = if sign_flip[gi] { nan_max(zv, bare) } else { zv };
            if sign_flip[gi] && bare > zv {
                sprime_mask[gi] = 1.0;
            }
            coul_ei += opt;
            meas_ei_tc += bare + tc;
        }
    }
    let (el_el, ion_ion) = crate::timed!(
        "gfmc.coul_eeii",
        if let Some(ps) = pot {
            let ee = match pbc {
                Some(p) => ps
                    .ee
                    .as_ref()
                    .expect("PBC potential state carries Ewald ee")
                    .energy(p.g_factors),
                None => ps
                    .ee_molecular
                    .expect("molecular potential state carries bare ee"),
            };
            (ee, ps.ion_ion)
        } else {
            match pbc {
                None => (
                    crate::coulomb_potential::compute_bare_coulomb_potential_el_el(r_up, r_dn),
                    crate::coulomb_potential::compute_bare_coulomb_potential_ion_ion(
                        positions,
                        eff_charges,
                    ),
                ),
                Some(p) => (
                    crate::ewald::compute_ewald_potential_ee(
                        r_up,
                        r_dn,
                        p.eta,
                        &p.real_images,
                        &p.g_vectors,
                        p.g_factors,
                        &p.cell,
                        p.n_max,
                        None,
                    ),
                    p.ii,
                ),
            }
        }
    );
    let diag_kin = 3.0 / (alat * alat) * (n as f64);
    let ecp_local = crate::timed!(
        "gfmc.ecp_local",
        match pot.and_then(|ps| ps.ecp_local.as_ref()) {
            // Incremental interval state: per-electron local-ECP rows (electron-
            // major sum; an fp reordering of the reference channel-major total,
            // bounded by the interval twin obligation).
            Some(rows) => rows.iter().sum::<f64>(),
            None => match fast_state {
                Some(st) =>
                    st.ecp_local(positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell_nmax,),
                None => crate::coulomb_potential::compute_ecp_local_parts_all_pairs(
                    positions, nuc_l, e_l_p, c_l_p, p_l_p, r_up, r_dn, cell_nmax,
                ),
            },
        }
    );
    // Per-candidate FN move amplitude a_m (<=0) + SP diagonal fold. tmove uses the full
    // det*Jastrow ratio uniformly; dltmove (jqmc_gfmc.py) folds max(det_only,0) into
    // SP and multiplies only the FN off-diagonal by the separate Jastrow ratio.
    let (fn_ecp, sp_ecp, cands) = match fast_ecp {
        Some(folded) => folded,
        None => {
            let mesh = mesh.expect("legacy ECP mesh accompanies the non-fast route");
            let mut fn_ecp = 0.0;
            let mut sp_ecp = 0.0;
            let mut cands: Vec<MeshCandidate> = Vec::with_capacity(mesh.len());
            for (is_up, i, pos, det_amp, jr) in mesh {
                let a_m = if dltmove {
                    sp_ecp += nan_max(det_amp, 0.0);
                    nan_min(det_amp, 0.0) * jr
                } else {
                    let vfull = det_amp * jr;
                    sp_ecp += nan_max(vfull, 0.0);
                    nan_min(vfull, 0.0)
                };
                fn_ecp += a_m;
                cands.push((is_up, i, pos, a_m));
            }
            (fn_ecp, sp_ecp, cands)
        }
    };
    let e_l =
        diag_kin + (el_el + ion_ion + coul_ei) + ecp_local + (sp + sp_ecp) + (fn_sum + fn_ecp);
    // Measurement local energy (novar=0 split): continuum e_L; the ECP
    // non-local (t-move localized) part is part of the continuum local
    // energy (TurboRVB's vpotoff) and is kept as-is.
    let e_l_meas = el_el + ion_ion + meas_ei_tc + ecp_local + sp_ecp + fn_ecp;
    let b_l = fn_sum + fn_ecp;
    (e_l, e_l_meas, b_l, elements, cands, sprime_mask)
}

/// Complex Bloch-twist (k!=Gamma) all-electron LRDMC diagonal e_L, jnone (classical fixed-phase).
/// Everything stays COMPLEX (complex det ratios / continuum kinetic); only the caller Re's the
/// escape denominator + weight exponent + move probs (jqmc _projection_t_core: the FN/SP clamps,
/// sign-flip, and ZV max/where run on complex with JAX lexicographic ordering -- see c_min0/c_max).
/// Diagonal Coulomb = Ewald (real) + per-electron Ewald ei ZV. Returns (e_L, b_L, elements) complex.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_lrdmc_ae_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    pbc: &crate::ewald::EwaldRef,
    fast: bool,
) -> (Complex64, Complex64, Complex64, Vec<Complex64>) {
    let (e_l, e_l_meas, b_l, elements, _) = gfmc_lrdmc_ae_complex_state(
        gem,
        jastrow,
        j3c,
        j_matrix,
        positions,
        eff_charges,
        r_up,
        r_dn,
        alat,
        rt,
        pbc,
        fast,
        None,
        None,
    );
    (e_l, e_l_meas, b_l, elements)
}

#[allow(clippy::too_many_arguments)]
fn gfmc_lrdmc_ae_complex_state(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    pbc: &crate::ewald::EwaldRef,
    fast: bool,
    state_in: Option<&MeshFastStateC>,
    pot: Option<&IntervalPotState>,
) -> (Complex64, Complex64, Complex64, Vec<Complex64>, Vec<f64>) {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n = n_up + n_dn;
    let shifts = lattice_shifts(alat, rt);
    let inv_2a2 = Complex64::new(-1.0 / (2.0 * alat * alat), 0.0);
    // FAST chain mode: complex Sherman-Morrison rank-1 kinetic elements off the shared
    // base state (transparent fall-back to the reference on a coverage-gate miss).
    // Complex interval drivers pass their carried per-interval state in
    // `state_in` (mirroring the real `gfmc_lrdmc_ae_state`); the stateless oracle entry
    // still builds one per evaluation.
    let built_state = if fast && state_in.is_none() {
        mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, r_up, r_dn)
    } else {
        None
    };
    let fast_state = state_in.or(built_state.as_ref());
    // Jastrow is real: DTN factors out of the twist exactly like J1/J2/J3 (real exp(DeltaJ)).
    let j1 = jastrow.one_body.as_ref();
    let j2 = jastrow.two_body.as_ref();
    let dtn = jastrow.dtn.as_ref();
    let with_j = j1.is_some() || j2.is_some() || j3c.is_some() || dtn.is_some();
    // 1. complex discretized-kinetic amplitudes = -1/(2a^2)*det(mesh)/det(base)*exp(DeltaJ).
    // The base det / base Jastrow value are built ONLY on the reference branch: the FAST
    // branch's ratios are relative to the SM base state, so paying for a full geminal
    // det + a full Jastrow value here would be dead work on the production path.
    let elements: Vec<Complex64> = match fast_state {
        Some(st) => discretized_kinetic_elements_complex_fast(
            gem, jastrow, j3c, j_matrix, r_up, r_dn, alat, rt, st,
        ),
        None => {
            let det_base = det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn));
            // Hop Jastrow ratio in the `jastrow_ratio` zone (the single-move factor of the
            // FAST branch), not the difference of two whole `jastrow_value_complex`
            // values, which rounds in `jastrow_eval` (float32 in mixed), issue #757.
            let jas_ratio =
                with_j.then(|| ComplexJasRatioState::new(jastrow, j3c, j_matrix, r_up, r_dn));
            let mut elements: Vec<Complex64> = Vec::with_capacity(6 * n);
            for e in 0..n_up {
                for shifts_s in &shifts {
                    let mut mu = r_up.to_owned();
                    for c in 0..3 {
                        mu[[e, c]] += shifts_s[c];
                    }
                    let det_m = det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), r_dn));
                    let jr = jas_ratio.as_ref().map_or(1.0, |st| {
                        let pos = [mu[[e, 0]], mu[[e, 1]], mu[[e, 2]]];
                        st.ratio(jastrow, j3c, r_up, r_dn, true, e, pos)
                    });
                    elements.push(inv_2a2 * det_ratio_complex(det_m, det_base) * jr);
                }
            }
            for e in 0..n_dn {
                for shifts_s in &shifts {
                    let mut md = r_dn.to_owned();
                    for c in 0..3 {
                        md[[e, c]] += shifts_s[c];
                    }
                    let det_m = det_phase_lnabs(&compute_geminal_complex(gem, r_up, &md.view()));
                    let jr = jas_ratio.as_ref().map_or(1.0, |st| {
                        let pos = [md[[e, 0]], md[[e, 1]], md[[e, 2]]];
                        st.ratio(jastrow, j3c, r_up, r_dn, false, e, pos)
                    });
                    elements.push(inv_2a2 * det_ratio_complex(det_m, det_base) * jr);
                }
            }
            elements
        }
    };
    // 2. FN/SP split + per-electron nondiag & sign-flip (complex, lexicographic).
    let mut fn_sum = Complex64::new(0.0, 0.0);
    let mut sp = Complex64::new(0.0, 0.0);
    for &el in &elements {
        fn_sum += c_min0(el);
        sp += c_max0(el);
    }
    let q = Complex64::new(1.0 / (2.0 * alat * alat), 0.0);
    let mut nondiag = vec![Complex64::new(0.0, 0.0); n];
    let mut sign_flip = vec![false; n];
    for e in 0..n {
        for s in 0..6 {
            let el = elements[e * 6 + s];
            nondiag[e] += el + q;
            if c_ge0(el) {
                sign_flip[e] = true;
            }
        }
    }
    // 3. complex continuum KE per electron KE_i = -1/2(grad^2lnPsi_i + (gradlnPsi_i)^2) [v*v, not |v|^2].
    // gradlnPsi = gradln det (complex) + gradJ (real: J1/J2/DTN + Re'd complex J3); grad^2 likewise.
    // The carried state streams determinant and Jastrow derivative tables;
    // unsupported orbital classes keep their counted full rebuilds.
    let (gu, gd, lu, ld) = crate::timed!(
        "gfmc.det_gradlap_c",
        match fast_state {
            Some(st) if complex_det_gradlap_ok(gem) => st.det_grad_lap_c(gem, r_up, r_dn),
            _ => {
                if fast_state.is_some() {
                    bump_route_counter(&GFMC_CPLX_DERIV_DET_SLOW);
                }
                grads_and_laplacian_ln_det_complex(gem, r_up, r_dn)
            }
        }
    );
    let (gju, gjd, lju, ljd) = if with_j {
        crate::timed!(
            "gfmc.j12_gradlap_c",
            match fast_state {
                Some(st) if complex_jaspart_gradlap_ok(jastrow) =>
                    st.jas_part_grad_lap(jastrow, r_up, r_dn),
                _ => {
                    if fast_state.is_some() {
                        if jastrow.dtn.is_some() {
                            bump_route_counter(&GFMC_CPLX_DTN_GRAD_SLOW);
                        }
                        if jastrow.one_body.is_some() || jastrow.two_body.is_some() {
                            bump_route_counter(&GFMC_CPLX_J12_GRAD_SLOW);
                        }
                    }
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(
                        jastrow, r_up, r_dn,
                    )
                }
            }
        )
    } else {
        (
            Array2::zeros((n_up, 3)),
            Array2::zeros((n_dn, 3)),
            numpy::ndarray::Array1::zeros(n_up),
            numpy::ndarray::Array1::zeros(n_dn),
        )
    };
    let (g3u, g3d, l3u, l3d) = crate::timed!(
        "gfmc.j3_gradlap_c",
        match (j3c, j_matrix) {
            (Some(j3), Some(jm)) => match fast_state {
                // The streamed J3 derivative cache requires a carried J3 ratio
                // state (contiguous j_matrix) and a supported
                // complex J3 orb class; misses keep the counted full rebuild.
                Some(st)
                    if st.j3_state.is_some()
                        && complex_candidate_class(j3) != CandidateBasisClass::Unsupported =>
                    st.j3_grad_lap_c(j3, jm, r_up, r_dn),
                _ => {
                    if fast_state.is_some() {
                        bump_route_counter(&GFMC_CPLX_J3_GRAD_SLOW);
                    }
                    j3_grads_lap_complex(j3, jm, r_up, r_dn)
                }
            },
            _ => (
                Array2::zeros((n_up, 3)),
                Array2::zeros((n_dn, 3)),
                numpy::ndarray::Array1::zeros(n_up),
                numpy::ndarray::Array1::zeros(n_dn),
            ),
        }
    );
    let ke_cont = |i: usize, up: bool| -> Complex64 {
        let (g, l, gj, lj, g3, l3) = if up {
            (&gu, &lu, &gju, &lju, &g3u, &l3u)
        } else {
            (&gd, &ld, &gjd, &ljd, &g3d, &l3d)
        };
        let mut g2 = Complex64::new(0.0, 0.0);
        for c in 0..3 {
            let gc = g[[i, c]] + Complex64::new(gj[[i, c]] + g3[[i, c]], 0.0);
            g2 += gc * gc;
        }
        -0.5 * (l[i] + Complex64::new(lj[i] + l3[i], 0.0) + g2)
    };
    // 4. per-electron Ewald ei (bare + discretized) -- real. Carried interval
    // rows when the incremental potential state is present.
    let (ei_bare, ei_disc) = match pot {
        Some(ps) => (
            ps.ei_bare.clone(),
            ps.ei_disc
                .clone()
                .expect("AE interval potential state carries discretized ei rows"),
        ),
        None => (
            crate::ewald::compute_ewald_potential_ei_per_electron(
                r_up,
                r_dn,
                pbc.eta,
                &pbc.real_images,
                &pbc.g_vectors,
                pbc.g_factors,
                &pbc.cell,
                pbc.n_max,
                eff_charges,
                positions,
                None,
            ),
            crate::ewald::compute_ewald_potential_ei_per_electron(
                r_up,
                r_dn,
                pbc.eta,
                &pbc.real_images,
                &pbc.g_vectors,
                pbc.g_factors,
                &pbc.cell,
                pbc.n_max,
                eff_charges,
                positions,
                Some(alat),
            ),
        ),
    };
    // 5. complex ZV composition.
    let mut coul_ei = Complex64::new(0.0, 0.0);
    let mut sprime_mask = vec![0.0_f64; n];
    // `meas_ei_tc`: MEASUREMENT (novar=0) per-electron pieces (complex Tc).
    let mut meas_ei_tc = Complex64::new(0.0, 0.0);
    for (up, nr) in [(true, n_up), (false, n_dn)] {
        for i in 0..nr {
            let gi = if up { i } else { n_up + i };
            let tc = ke_cont(i, up);
            let zv = Complex64::new(ei_bare[gi], 0.0) + tc - nondiag[gi];
            let opt = if sign_flip[gi] {
                c_max(zv, Complex64::new(ei_disc[gi], 0.0))
            } else {
                zv
            };
            if sign_flip[gi] && ei_disc[gi] > zv.re {
                sprime_mask[gi] = 1.0;
            }
            coul_ei += opt;
            meas_ei_tc += Complex64::new(ei_bare[gi], 0.0) + tc;
        }
    }
    let ee = match pot {
        Some(ps) => ps
            .ee
            .as_ref()
            .expect("complex PBC potential state carries Ewald ee")
            .energy(pbc.g_factors),
        None => crate::ewald::compute_ewald_potential_ee(
            r_up,
            r_dn,
            pbc.eta,
            &pbc.real_images,
            &pbc.g_vectors,
            pbc.g_factors,
            &pbc.cell,
            pbc.n_max,
            None,
        ),
    };
    let diag_kin = 3.0 / (alat * alat) * (n as f64);
    let e_l = Complex64::new(diag_kin + ee + pbc.ii, 0.0) + coul_ei + sp + fn_sum;
    // Measurement local energy (novar=0 split): continuum e_L (complex).
    let e_l_meas = Complex64::new(ee + pbc.ii, 0.0) + meas_ei_tc;
    let b_l = fn_sum; // non_diagonal_sum_hamiltonian (complex; caller Re's for escape)
    (e_l, e_l_meas, b_l, elements, sprime_mask)
}

// ---------------------------------------------------------------------------
// COMPLEX-TWIST FAST (Sherman-Morrison rank-1) mesh state -- the complex analogue of
// `MeshFastState` / `mesh_fast_state`. Same contract: built ONCE per `e_L` evaluation
// and consumed by BOTH one-electron-move meshes of the LRDMC sub-step (the 6N
// discretized-kinetic hops AND the ECP T-move quadrature), replacing the per-point
// FULL complex geminal + LU + full Jastrow value with a rank-1 complex det ratio times
// the incremental (REAL) single-move Jastrow ratio.
//
// HOLOMORPHY: the det ratio stays COMPLEX end to end (`1 + v^T G^-1 u`, no Re anywhere);
// the Jastrow is real by construction on the twist (`jastrow_value_complex` returns f64 --
// J1/J2/DTN are geometry-only and J3 is the `Re(complex AO)` bilinear), so the element
// `-1/(2a^2) * det_ratio * jr` carries the same complex value the reference builds. Re is
// taken by the CALLER at the three fixed-phase decision points, never here.
// ---------------------------------------------------------------------------

/// Complex orbital matrix `(num_orb, N)` built COLUMN BY COLUMN through the same
/// single-position path the SM candidate columns use. The batch `ComplexOrb::value`
/// and the 1-row `orb_single_complex` are the same math but not the same fp
/// association, so anchoring the base on the single path makes every rank-1 ratio an
/// exact `det_new/det_base` for the stored base (mirrors the real
/// `orb_matrix_via_single`).
fn orb_matrix_via_single_complex(orb: &ComplexOrb, r: &ArrayView2<f64>) -> Array2<Complex64> {
    let n = r.nrows();
    if n == 0 {
        // No electron of this spin: the batch path already returns the correct
        // `(num_orb, 0)` shape and there is no column to rebuild.
        return orb.value(r);
    }
    let mut out: Option<Array2<Complex64>> = None;
    for i in 0..n {
        let col = crate::complex_det::orb_single_complex(orb, [r[[i, 0]], r[[i, 1]], r[[i, 2]]]);
        let m = col.len();
        let o = out.get_or_insert_with(|| Array2::<Complex64>::zeros((m, n)));
        for a in 0..m {
            o[[a, i]] = col[a];
        }
    }
    out.expect("n > 0")
}

/// Single-move complex-twist Jastrow ratio `exp(J(r') - J(r))` in the `jastrow_ratio`
/// zone: the J1/J2 delta (`jastrow_1b2b_ratio`, or a caller-supplied delta), the rank-1
/// J3 ratio over `Re(complex J3 orbital)` columns and the DTN streaming ratio. The body of
/// [`MeshFastStateC::jas_ratio`], shared with the Jastrow-only [`ComplexJasRatioState`].
#[allow(clippy::too_many_arguments)]
fn complex_single_move_jas_ratio(
    jastrow: &JastrowData,
    j3_state: Option<&crate::jastrow_factor::J3RatioState>,
    dtn_state: Option<&crate::dtn::DtnRatioState>,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    is_up: bool,
    e: usize,
    pos: [f64; 3],
    j12_delta: Option<f64>,
    j3_col: Option<&[f64]>,
    dtn_ratio: Option<f64>,
) -> f64 {
    let mut jr = match j12_delta {
        Some(delta) if jastrow.precision_policy.is_f32("jastrow_ratio") => {
            f64::from((delta as f32).exp())
        }
        Some(delta) => delta.exp(),
        None => crate::jastrow_factor::jastrow_1b2b_ratio(jastrow, r_up, r_dn, is_up, e, pos),
    };
    if let (Some(jm), Some(st)) = (j_matrix, j3_state) {
        let j3r = match j3_col {
            Some(col) => Some(crate::jastrow_factor::j3_ratio(jm, st, is_up, e, col)),
            None => j3c.map(|j3| {
                let col = crate::complex_det::j3_ao_re_single(j3, pos);
                crate::jastrow_factor::j3_ratio(jm, st, is_up, e, &col)
            }),
        };
        if let Some(j3r) = j3r {
            jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, j3r);
        }
    }
    if let (Some(d), Some(st)) = (&jastrow.dtn, dtn_state) {
        let dtn_r = match dtn_ratio {
            Some(r) => r,
            None => crate::dtn::compute_dtn_ratio_streaming(d, st, r_up, r_dn, is_up, e, pos),
        };
        jr = crate::jastrow_factor::multiply_ratio_components(jastrow, jr, dtn_r);
    }
    jr
}

/// Jastrow-only base state for the complex-twist single-move ratio (issue #757): the
/// J3 rank-1 tables over `Re(complex J3 orbital)` and the DTN streaming rows of
/// [`MeshFastStateC`], without its determinant base. The full-recompute twist ECP
/// kernels (mesh-force/strain fallbacks, reference meshes, reference discretized
/// kinetic hops) take their Jastrow factor from [`Self::ratio`], i.e. the same
/// `jastrow_ratio`-zone factor as the fast mesh state, instead of the difference of
/// two whole `jastrow_value_complex` values, which round in `jastrow_eval` (float32 in
/// mixed). `j_matrix` is copied to standard layout, so no coverage gate applies.
pub(crate) struct ComplexJasRatioState {
    j_matrix: Option<Array2<f64>>,
    j3_state: Option<crate::jastrow_factor::J3RatioState>,
    dtn_state: Option<crate::dtn::DtnRatioState>,
}

impl ComplexJasRatioState {
    pub(crate) fn new(
        jastrow: &JastrowData,
        j3c: Option<&ComplexOrb>,
        j_matrix: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> Self {
        let j_matrix = match j3c {
            Some(_) => j_matrix.map(|m| m.as_standard_layout().to_owned()),
            None => None,
        };
        let j3_state = match (j3c, j_matrix.as_ref()) {
            (Some(j3), Some(jm)) => {
                let au = crate::complex_det::j3_ao_re(j3, r_up);
                let ad = crate::complex_det::j3_ao_re(j3, r_dn);
                Some(crate::jastrow_factor::init_j3_ratio_state_with_policy(
                    &jm.view(),
                    &au.view(),
                    &ad.view(),
                    jastrow.precision_policy.is_f32("jastrow_ratio"),
                ))
            }
            _ => None,
        };
        let dtn_state = jastrow
            .dtn
            .as_ref()
            .map(|d| crate::dtn::init_dtn_ratio_state(d, r_up, r_dn));
        Self {
            j_matrix,
            j3_state,
            dtn_state,
        }
    }

    /// `exp(J(r') - J(r))` for moving electron `e` (spin `is_up`) of the base `(r_up, r_dn)`
    /// to `pos`; `jastrow` / `j3c` are the ones the state was built from.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ratio(
        &self,
        jastrow: &JastrowData,
        j3c: Option<&ComplexOrb>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        pos: [f64; 3],
    ) -> f64 {
        let jm = self.j_matrix.as_ref().map(|m| m.view());
        complex_single_move_jas_ratio(
            jastrow,
            self.j3_state.as_ref(),
            self.dtn_state.as_ref(),
            j3c,
            jm.as_ref(),
            r_up,
            r_dn,
            is_up,
            e,
            pos,
            None,
            None,
            None,
        )
    }
}

/// Shared per-configuration base state for the COMPLEX FAST mesh kernels: the complex
/// streaming det-ratio state, the complex base geminal inverse, and the incremental
/// (real) Jastrow states (J3 rank-1 tables over `Re(complex AO)` + DTN streaming rows).
///
/// Mirroring the real [`MeshFastState`], complex drivers build this once per
/// projection interval and advance it after every hop
/// (complex SM rank-1 on `ginv`; the down-spin `-k` leaf already stores the
/// conjugated up-spin coefficients, so the algebra is a plain transpose:
/// the dn channel's conjugation lives in its MO coefficients, exactly the convention
/// the seed `inverse_complex(ou^T W)` and `det_ratio`'s `1 + v^T G^-1 u` use; the
/// XLA twin documents the same inverse-seeding convention in jqmc_gfmc.py),
/// and reuse the candidate AO tables below across the interval's fixed mesh rotation
/// with a per-hop refresh of only the moved electron's rows. The ECP geometry and
/// local-channel caches are REAL and reuse the real path's cache structs verbatim.
pub(crate) struct MeshFastStateC {
    det_state: crate::complex_det::DetRatioStateC,
    ginv: Array2<Complex64>,
    det_winv: DetWdotCacheC,
    j3_state: Option<crate::jastrow_factor::J3RatioState>,
    dtn_state: Option<crate::dtn::DtnRatioState>,
    /// Real DTN candidate ratios are twist-independent and use the same
    /// tabpip-style state as the Gamma path.
    dtn_candidate_table: std::cell::RefCell<Option<crate::dtn::DtnCandidateTable>>,
    ecp_dtn_candidate_table: std::cell::RefCell<Option<crate::dtn::DtnCandidateTable>>,
    ecp_dtn_dirty: std::cell::Cell<Option<HopChange>>,
    /// TurboRVB-style candidate-table reuse, enabled only when the mesh rotation
    /// is held fixed for one projection interval (the interval drivers set it;
    /// the stateless `e_L` oracle entry points leave it off and rebuild owned
    /// tables per evaluation).
    reuse_fixed_mesh: bool,
    /// FAST ECP mesh AO-space w-dot ([`ComplexOrb::mesh_wdot_into`]) instead of the
    /// orbital-rank determinant candidate table. Set only by the one-shot FAST
    /// chunk e_L (`compute_ecp_nonlocal_withj_complex_fast` with `mesh_wdot`),
    /// whose state is never advanced, so no table survives to be reused; the
    /// interval drivers (tables reused across hops) and the per-step dispatch keep
    /// it off.
    ecp_wdot: bool,
    kinetic_cache: std::cell::RefCell<Option<KineticAoCacheC>>,
    kinetic_dirty: std::cell::Cell<Option<HopChange>>,
    ecp_cache: std::cell::RefCell<Option<EcpAoCacheC>>,
    ecp_geometry_cache: std::cell::RefCell<Option<EcpGeometryCache>>,
    ecp_dirty: std::cell::Cell<Option<HopChange>>,
    ecp_local_cache: std::cell::RefCell<Option<EcpLocalCache>>,
    ecp_local_dirty: std::cell::Cell<Option<(bool, usize)>>,
    /// Complex continuum-kinetic determinant derivative cache. Fused-family
    /// tables own their kinetic-quality `kin_ginv`, are lazily seeded by the
    /// first local-energy evaluation, and advance one column after each hop.
    det_grad_cache: std::cell::RefCell<Option<crate::complex_det::DetGradLapCacheC>>,
    /// DTN electron grad/lap cache. Its real-valued distance tables share the
    /// same seed and update machinery as the real path.
    dtn_grad_cache: std::cell::RefCell<Option<crate::dtn::DtnGradLapCache>>,
    /// Generic J1/J2 grad/lap cache. The real `J12GradLapCache` representation
    /// and its generalized updates also cover periodic and exponential forms.
    j12_grad_cache: std::cell::RefCell<Option<crate::jastrow_factor::J12GradLapCache>>,
    /// Complex J3 derivative cache with fused-family `Re(value_grad_lap)` rows
    /// and its own fp64 kinetic `J3RatioState`.
    j3_grad_cache: std::cell::RefCell<Option<crate::complex_det::J3GradLapCacheC>>,
}

struct DetWdotCacheC {
    up: Array2<Complex64>,
    up_base: Vec<Complex64>,
    dn: Array2<Complex64>,
    dn_base: Vec<Complex64>,
}

fn init_det_wdot_cache_c(
    gem: &ComplexGeminal,
    st: &crate::complex_det::DetRatioStateC,
    ginv: &ArrayView2<Complex64>,
) -> DetWdotCacheC {
    let zero = Complex64::new(0.0, 0.0);
    let n = ginv.nrows();
    let n_dn = st.ao_dn.ncols();
    let m_up = st.ao_up.nrows();
    let m_dn = st.ao_dn.nrows();
    let mut up = Array2::<Complex64>::zeros((n, m_up));
    let mut dn = Array2::<Complex64>::zeros((n, m_dn));
    for i in 0..n {
        for a in 0..m_up {
            let mut q = zero;
            if a < st.paired_dn.nrows() {
                for j in 0..n_dn {
                    q += st.paired_dn[[a, j]] * ginv[[j, i]];
                }
            }
            for j in 0..gem.n_unpaired() {
                q += gem.lambda_c_value(a, gem.orb_num_dn + j) * ginv[[n_dn + j, i]];
            }
            up[[i, a]] = q;
        }
        for a in 0..m_dn {
            let mut q = zero;
            for k in 0..n {
                q += ginv[[i, k]] * st.paired_up_lambda[[k, a]];
            }
            dn[[i, a]] = q;
        }
    }
    let mut up_base = vec![zero; n];
    for i in 0..n {
        for a in 0..m_up {
            up_base[i] += up[[i, a]] * st.ao_up[[a, i]];
        }
    }
    let mut dn_base = vec![zero; n_dn];
    for i in 0..n_dn {
        for a in 0..m_dn {
            dn_base[i] += dn[[i, a]] * st.ao_dn[[a, i]];
        }
    }
    DetWdotCacheC {
        up,
        up_base,
        dn,
        dn_base,
    }
}

#[allow(clippy::too_many_arguments)]
fn advance_det_wdot_cache_c(
    cache: &mut DetWdotCacheC,
    gem: &ComplexGeminal,
    st: &crate::complex_det::DetRatioStateC,
    is_up: bool,
    e: usize,
    det_col: &[Complex64],
    sm_u: &[Complex64],
    sm_x: &[Complex64],
    sm_y: &[Complex64],
    rho: Complex64,
) {
    let zero = Complex64::new(0.0, 0.0);
    let n = cache.up.nrows();
    let m_up = cache.up.ncols();
    let m_dn = cache.dn.ncols();
    if is_up {
        let pivot = cache.up.row(e).to_vec();
        for (i, sm_y_i) in sm_y[..n].iter().enumerate() {
            let scale = *sm_y_i / rho;
            for (a, pivot_a) in pivot[..m_up].iter().enumerate() {
                cache.up[[i, a]] -= scale * *pivot_a;
            }
        }
        let mut yd_minus_delta = vec![zero; m_dn];
        for a in 0..m_dn {
            let mut yd = zero;
            for (i, sm_y_i) in sm_y[..n].iter().enumerate() {
                yd += *sm_y_i * st.paired_up_lambda[[i, a]];
            }
            let new_coeff = if gem.is_sd() {
                det_col[a]
            } else {
                (0..det_col.len())
                    .map(|o| det_col[o] * gem.lambda_c_value(o, a))
                    .sum::<Complex64>()
            };
            yd_minus_delta[a] = yd - (new_coeff - st.paired_up_lambda[[e, a]]);
        }
        for (i, sm_x_i) in sm_x[..n].iter().enumerate() {
            let scale = *sm_x_i / rho;
            for (a, yd_minus_delta_a) in yd_minus_delta[..m_dn].iter().enumerate() {
                cache.dn[[i, a]] -= scale * *yd_minus_delta_a;
            }
        }
    } else {
        let mut cup_x_minus_delta = vec![zero; m_up];
        for a in 0..m_up {
            let mut cup_x = zero;
            for (i, sm_u_i) in sm_u[..n].iter().enumerate() {
                cup_x += cache.up[[i, a]] * *sm_u_i;
            }
            let delta = if gem.is_sd() {
                if a < st.paired_dn.nrows() {
                    det_col[a] - st.paired_dn[[a, e]]
                } else {
                    zero
                }
            } else {
                let new_coeff = (0..det_col.len())
                    .map(|b| gem.lambda_c_value(a, b) * det_col[b])
                    .sum::<Complex64>();
                new_coeff - st.paired_dn[[a, e]]
            };
            cup_x_minus_delta[a] = cup_x - delta;
        }
        for (i, sm_y_i) in sm_y[..n].iter().enumerate() {
            let scale = *sm_y_i / rho;
            for (a, cup_x_minus_delta_a) in cup_x_minus_delta[..m_up].iter().enumerate() {
                cache.up[[i, a]] -= scale * *cup_x_minus_delta_a;
            }
        }
        let pivot = cache.dn.row(e).to_vec();
        for (i, sm_x_i) in sm_x[..n].iter().enumerate() {
            let scale = *sm_x_i / rho;
            for (a, pivot_a) in pivot[..m_dn].iter().enumerate() {
                cache.dn[[i, a]] -= scale * *pivot_a;
            }
        }
    }
    for i in 0..n {
        let mut base = zero;
        // As in the real refresh: ``det_col`` is read on its branch only.
        for (a, w) in cache.up.row(i).iter().enumerate() {
            let value = if is_up && i == e {
                det_col[a]
            } else {
                st.ao_up[[a, i]]
            };
            base += *w * value;
        }
        cache.up_base[i] = base;
    }
    for i in 0..st.ao_dn.ncols() {
        let mut base = zero;
        for (a, w) in cache.dn.row(i).iter().enumerate() {
            let value = if !is_up && i == e {
                det_col[a]
            } else {
                st.ao_dn[[a, i]]
            };
            base += *w * value;
        }
        cache.dn_base[i] = base;
    }
}

/// Complex kinetic 6N-mesh candidate tables: complex determinant MO/orbital
/// rows + the real `Re(complex AO)` J3 rows + basis-independent J1/J2
/// candidate log ratios.  The latter use the same periodic/form/precision
/// generic low-rank update as the real path.
struct KineticAoCacheC {
    alat_bits: u64,
    rt_bits: [u64; 9],
    det_up: Array2<Complex64>,
    det_dn: Array2<Complex64>,
    j3: Option<Array2<f64>>,
    j12_up: Option<Vec<f64>>,
    j12_dn: Option<Vec<f64>>,
}

/// Complex ECP T-move quadrature candidate tables. The determinant tables are
/// `None` on the AO-space w-dot route (`MeshFastStateC::ecp_wdot`), which
/// contracts every mesh point directly; `rows_up` / `rows_dn` record the
/// candidate layout either way.
struct EcpAoCacheC {
    rt_bits: [u64; 9],
    rows_per_electron: usize,
    rows_up: usize,
    rows_dn: usize,
    det_up: Option<Array2<Complex64>>,
    det_dn: Option<Array2<Complex64>>,
    j3: Option<Array2<f64>>,
    j12_up: Option<Vec<f64>>,
    j12_dn: Option<Vec<f64>>,
}

impl MeshFastStateC {
    #[inline]
    pub(crate) fn ginv(&self) -> &Array2<Complex64> {
        &self.ginv
    }

    /// The streaming determinant caches the running inverse was advanced with.
    #[inline]
    pub(crate) fn det_state(&self) -> &crate::complex_det::DetRatioStateC {
        &self.det_state
    }

    /// Opt a one-shot state into the candidate-table kernels (batched DTN candidate
    /// table, cached AO rows) that the interval drivers use. The tables are built once
    /// for the configuration the state was seeded at.
    pub(crate) fn enable_candidate_tables(&mut self) {
        self.reuse_fixed_mesh = true;
    }

    /// Opt a one-shot FAST chunk state into the ECP mesh AO-space w-dot
    /// determinant ratio (see the `ecp_wdot` field).
    pub(crate) fn enable_ecp_wdot(&mut self) {
        self.ecp_wdot = true;
    }

    /// Combined single-move Jastrow ratio (J1/J2 + J3 + DTN), the exact factor set of
    /// the reference `exp(jastrow_value_complex(new) - jastrow_value_complex(base))`.
    /// REAL by construction (the twist Jastrow is real), so no Re-projection happens
    /// here -- there is no imaginary part to drop.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jas_ratio(
        &self,
        jastrow: &JastrowData,
        j3c: Option<&ComplexOrb>,
        j_matrix: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        pos: [f64; 3],
    ) -> f64 {
        self.jas_ratio_with_j3_col(
            jastrow, j3c, j_matrix, r_up, r_dn, is_up, e, pos, None, None, None,
        )
    }

    /// [`Self::jas_ratio`] with the J3 `Re(complex AO)` column optionally supplied
    /// from a candidate table row containing the exact `j3_ao_re_single` values, so
    /// the rank-1 `j3_ratio` chain is unchanged); `None` falls back to the
    /// per-point column evaluation. `dtn_ratio` optionally supplies the DTN move
    /// ratio from a per-electron batched `compute_dtn_ratio_mesh_cached` group,
    /// mirroring the real path's `jas_ratios`; `None` keeps the
    /// per-point streaming kernel.
    #[allow(clippy::too_many_arguments)]
    fn jas_ratio_with_j3_col(
        &self,
        jastrow: &JastrowData,
        j3c: Option<&ComplexOrb>,
        j_matrix: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        pos: [f64; 3],
        j12_delta: Option<f64>,
        j3_col: Option<&[f64]>,
        dtn_ratio: Option<f64>,
    ) -> f64 {
        complex_single_move_jas_ratio(
            jastrow,
            self.j3_state.as_ref(),
            self.dtn_state.as_ref(),
            j3c,
            j_matrix,
            r_up,
            r_dn,
            is_up,
            e,
            pos,
            j12_delta,
            j3_col,
            dtn_ratio,
        )
    }

    fn ensure_kinetic_dtn_candidate_table(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        shifts: &[[f64; 3]],
    ) {
        if !self.reuse_fixed_mesh {
            return;
        }
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return,
        };
        let mut slot = self.dtn_candidate_table.borrow_mut();
        let rebuild = slot
            .as_ref()
            .is_none_or(|table| !table.matches_layout(r_up, r_dn, shifts));
        if rebuild {
            *slot = Some(crate::timed!(
                "gfmc.kin_dtn_tabpip_build_c",
                crate::dtn::init_dtn_candidate_table(d, state, r_up, r_dn, shifts)
            ));
            bump_route_counter(&GFMC_DTN_TABPIP_BUILDS);
        }
    }

    fn ensure_ecp_dtn_candidate_table(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        positions_up: &[[f64; 3]],
        positions_dn: &[[f64; 3]],
        rows_per_electron: usize,
    ) {
        if !self.reuse_fixed_mesh {
            return;
        }
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return,
        };
        let mut slot = self.ecp_dtn_candidate_table.borrow_mut();
        let rebuild = slot.as_ref().is_none_or(|table| {
            !table.matches_position_layout(
                r_up,
                r_dn,
                positions_up,
                positions_dn,
                rows_per_electron,
            )
        });
        if rebuild {
            *slot = Some(crate::timed!(
                "gfmc.ecp_dtn_tabpip_build_c",
                crate::dtn::init_dtn_candidate_table_from_positions(
                    d,
                    state,
                    r_up,
                    r_dn,
                    positions_up,
                    positions_dn,
                    rows_per_electron,
                )
            ));
            self.ecp_dtn_dirty.set(None);
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_BUILDS);
        } else if let Some(change) = self.ecp_dtn_dirty.take() {
            let positions = if change.is_up {
                positions_up
            } else {
                positions_dn
            };
            let offset = change.electron * rows_per_electron;
            crate::timed!(
                "gfmc.ecp_dtn_tabpip_advance_c",
                crate::dtn::advance_dtn_candidate_table_from_positions(
                    d,
                    slot.as_mut().expect("complex ECP DTN table initialized"),
                    state,
                    r_up,
                    r_dn,
                    change.is_up,
                    change.electron,
                    change.old_pos,
                    &positions[offset..offset + rows_per_electron],
                )
            );
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_ADVANCES);
        }
    }

    /// Per-electron candidate-group DTN ratios off the carried streaming state
    /// via the batched mesh kernel, the complex twin of the real `jas_ratios`
    /// DTN branch: `None` when no DTN term or streaming state exists.
    /// fp class: in the mixed lane the batched kernel delegates to the exact
    /// per-point streaming reduction (identical results); in the fp64 lane it
    /// is the same J-difference evaluated in the mesh kernel's association --
    /// the documented fast-chain fp-floor class the real path already carries.
    fn dtn_group_ratios(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        newpos: &[[f64; 3]],
    ) -> Option<Vec<f64>> {
        let (d, state) = match (&jastrow.dtn, &self.dtn_state) {
            (Some(d), Some(state)) => (d, state),
            _ => return None,
        };
        if let Some(ratios) = self
            .dtn_candidate_table
            .borrow()
            .as_ref()
            .and_then(|table| table.ratios_for_group(d, is_up, e, newpos))
        {
            bump_route_counter(&GFMC_DTN_TABPIP_HITS);
            return Some(ratios);
        }
        if let Some(ratios) = self
            .ecp_dtn_candidate_table
            .borrow()
            .as_ref()
            .and_then(|table| table.ratios_for_group(d, is_up, e, newpos))
        {
            bump_route_counter(&GFMC_DTN_TABPIP_ECP_HITS);
            return Some(ratios);
        }
        let points = Array2::from_shape_fn((newpos.len(), 3), |(row, axis)| newpos[row][axis]);
        let mut ratios = vec![0.0_f64; newpos.len()];
        crate::dtn::compute_dtn_ratio_mesh_cached(
            d,
            state,
            r_up,
            r_dn,
            is_up,
            e,
            &points.view(),
            &mut ratios,
        );
        bump_route_counter(&GFMC_CPLX_DTN_MESH_HITS);
        Some(ratios)
    }

    /// Streamed complex continuum-kinetic det grad/lap: seed the fused-family
    /// derivative cache on first use (per-spin build events under the orbital's
    /// basis class), then contract the maintained tables against the cache's
    /// own kinetic-quality inverse. Caller-gated by [`complex_det_gradlap_ok`].
    fn det_grad_lap_c(
        &self,
        gem: &ComplexGeminal,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<Complex64>,
        Array2<Complex64>,
        numpy::ndarray::Array1<Complex64>,
        numpy::ndarray::Array1<Complex64>,
    ) {
        let mut cache = self.det_grad_cache.borrow_mut();
        if cache.is_none() {
            note_deriv_event_c(complex_candidate_class(&gem.orb_up), CandidateEvent::Build);
            note_deriv_event_c(complex_candidate_class(&gem.orb_dn), CandidateEvent::Build);
            *cache = Some(crate::timed!(
                "gfmc.det_gradlap_seed_c",
                crate::complex_det::init_det_grad_lap_cache_complex(gem, r_up, r_dn)
            ));
        }
        crate::complex_det::contract_det_grad_lap_cache_complex(
            gem,
            cache
                .as_ref()
                .expect("complex det derivative cache initialized"),
        )
    }

    /// Streamed J1/J2(+DTN) grad/lap for the complex e_L: the J1/J2 legs come
    /// from the projection-scoped generic [`crate::jastrow_factor::J12GradLapCache`]
    /// and the DTN leg from the projection-scoped
    /// [`crate::dtn::DtnGradLapCache`], added in
    /// `compute_grads_and_laplacian_jastrow_part`'s exact accumulation order
    /// (((J1)+J2)+DTN). In full precision the seed-time result matches the
    /// parent kernel; the DTN leg is in the fp64 `jastrow_grad_lap_carry` zone
    /// in both modes, and later sub-steps carry the delta-update accumulation
    /// order.
    fn jas_part_grad_lap(
        &self,
        jastrow: &JastrowData,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        numpy::ndarray::Array1<f64>,
        numpy::ndarray::Array1<f64>,
    ) {
        let (gu, gd, lu, ld) = if crate::jastrow_factor::j12_generic_cache_ok(jastrow) {
            let mut cache = self.j12_grad_cache.borrow_mut();
            if cache.is_none() {
                bump_route_counter(&GFMC_CPLX_J12_GRAD_BUILDS);
                *cache = Some(crate::timed!(
                    "gfmc.j12_gradlap_seed_c",
                    crate::jastrow_factor::init_j12_grad_lap_cache_generic(jastrow, r_up, r_dn)
                ));
            }
            crate::jastrow_factor::clone_j12_grad_lap_cache(
                cache.as_ref().expect("J12 derivative cache initialized"),
            )
        } else {
            // No J1/J2 term (DTN-only Jastrow): the J12 legs are the zero
            // tables the parent kernel starts from.
            crate::jastrow_factor::compute_grads_and_laplacian_j12_part(jastrow, r_up, r_dn)
        };
        let (gu, gd, lu, ld) = match &jastrow.dtn {
            Some(d) => {
                let mut cache = self.dtn_grad_cache.borrow_mut();
                if cache.is_none() {
                    bump_route_counter(&GFMC_CPLX_DTN_GRAD_BUILDS);
                    *cache = Some(crate::timed!(
                        "gfmc.dtn_gradlap_seed_c",
                        crate::dtn::init_dtn_grad_lap_cache(d, r_up, r_dn)
                    ));
                }
                let (dgu, dgd, dlu, dld) = crate::dtn::clone_dtn_grad_lap_cache(
                    cache.as_ref().expect("DTN derivative cache initialized"),
                );
                (
                    gu + dgu,
                    gd + dgd,
                    lu + numpy::ndarray::Array1::from_vec(dlu),
                    ld + numpy::ndarray::Array1::from_vec(dld),
                )
            }
            None => (gu, gd, lu, ld),
        };
        (gu, gd, lu, ld)
    }

    /// Streamed complex J3 grad/lap: seed the fused-family `Re(value_grad_lap)`
    /// derivative cache on first use, then contract with the real fast path's
    /// q-vector kernel against the cache's own fp64 kinetic `J3RatioState`.
    fn j3_grad_lap_c(
        &self,
        j3: &ComplexOrb,
        jm: &ArrayView2<f64>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        numpy::ndarray::Array1<f64>,
        numpy::ndarray::Array1<f64>,
    ) {
        let mut cache = self.j3_grad_cache.borrow_mut();
        if cache.is_none() {
            bump_route_counter(&GFMC_CPLX_J3_GRAD_BUILDS);
            *cache = Some(crate::timed!(
                "gfmc.j3_gradlap_seed_c",
                crate::complex_det::init_j3_grad_lap_cache_complex(j3, jm, r_up, r_dn)
            ));
        }
        crate::complex_det::contract_j3_grad_lap_cache_complex(
            jm,
            cache
                .as_ref()
                .expect("complex J3 derivative cache initialized"),
        )
    }

    /// Complex rank-1 det ratio `det(G')/det(G)` for moving electron (`is_up`, `e`) to
    /// `pos`. Fully holomorphic: no Re is taken on the ratio or on any factor of it.
    pub(crate) fn det_ratio(
        &self,
        gem: &ComplexGeminal,
        is_up: bool,
        e: usize,
        pos: [f64; 3],
    ) -> Complex64 {
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        let ao_new = crate::complex_det::orb_single_complex(orb, pos);
        let n = self.ginv.nrows();
        let mut ratio = Complex64::new(1.0, 0.0);
        if is_up {
            // u = e_e -> ratio = 1 + v^T G^-1 e_e = 1 + sum_i v_i G^-1[i, e]
            let v = crate::complex_det::compute_v_up_move_complex(gem, &self.det_state, e, &ao_new);
            for (i, v_i1) in v[..n].iter().enumerate() {
                ratio += *v_i1 * self.ginv[[i, e]];
            }
        } else {
            // v = e_e -> ratio = 1 + e_e^T G^-1 u = 1 + sum_i G^-1[e, i] u_i
            let u = crate::complex_det::compute_u_dn_move_complex(&self.det_state, e, &ao_new);
            for (i, u_i1) in u[..n].iter().enumerate() {
                ratio += self.ginv[[e, i]] * *u_i1;
            }
        }
        ratio
    }

    /// Batched complex determinant ratios for one candidate group from a candidate
    /// table (`aos` rows = orbital columns at the candidate positions, the exact
    /// `orb_single_complex` chain). Complex twin of the real
    /// `det_ratios_from_candidate_aos_into`: for a fixed base configuration and
    /// moved electron the complete ratio is AFFINE in the proposed orbital column,
    ///   ratio = base + q . orb(new),  base = 1 - q . orb(old),
    /// with q the `G^-1`-folded pairing row. This REASSOCIATES the per-point
    /// `1 + v^T G^-1 e_e` sum (fold `G^-1` into q first, then one orbital-width
    /// dot per candidate) -- the SAME documented reassociation the real twin
    /// carries, accepted at the fast-chain fp floor (statistically guarded).
    /// No conjugation here: the down-spin `-k` MO already stores the conjugated
    /// up-spin coefficients, so plain-transpose algebra matches term-for-term the
    /// `compute_v_up_move_complex` / `compute_u_dn_move_complex` factors.
    /// The tables are ORBITAL-rank (the MO projection already happened inside
    /// `ComplexOrb::value`), so unlike the real twin there is no AO-width weight
    /// fold and no candidate-space-width ambiguity.
    fn det_ratios_from_candidate_aos_c_into(
        &self,
        _gem: &ComplexGeminal,
        is_up: bool,
        e: usize,
        aos: &numpy::ndarray::ArrayView2<Complex64>,
        _q: &mut [Complex64],
        out: &mut [Complex64],
    ) {
        let npts = aos.nrows();
        debug_assert_eq!(out.len(), npts);
        let (weights, base) = if is_up {
            (&self.det_winv.up, self.det_winv.up_base[e])
        } else {
            (&self.det_winv.dn, self.det_winv.dn_base[e])
        };
        let m = weights.ncols();
        debug_assert_eq!(aos.ncols(), m);
        let q = weights.row(e);
        let q = q.as_slice().expect("complex w-inverse row contiguous");
        let affine_base = Complex64::new(1.0, 0.0) - base;
        // Four candidates per pass with independent accumulators (instruction-level
        // parallelism); each candidate's sum keeps the original order and the exact
        // `Complex64` multiply/add arithmetic, so every ratio is bit-identical to the
        // one-candidate-at-a-time loop.
        let Some(flat) = aos.as_slice() else {
            for (s, o) in out.iter_mut().enumerate() {
                let mut acc = affine_base;
                for (qa, v) in q.iter().zip(aos.row(s).iter()) {
                    acc += *qa * *v;
                }
                *o = acc;
            }
            return;
        };
        let row_of = |s: usize| -> &[Complex64] { &flat[s * m..s * m + m] };
        let mut s = 0;
        while s + 4 <= npts {
            let (r0, r1, r2, r3) = (row_of(s), row_of(s + 1), row_of(s + 2), row_of(s + 3));
            let mut a = [(affine_base.re, affine_base.im); 4];
            for k in 0..m {
                let (qr, qi) = (q[k].re, q[k].im);
                for (acc, v) in a.iter_mut().zip([r0[k], r1[k], r2[k], r3[k]]) {
                    acc.0 += qr * v.re - qi * v.im;
                    acc.1 += qr * v.im + qi * v.re;
                }
            }
            for (t, acc) in a.iter().enumerate() {
                out[s + t] = Complex64::new(acc.0, acc.1);
            }
            s += 4;
        }
        while s < npts {
            let row = row_of(s);
            let mut acc = affine_base;
            for (qa, v) in q.iter().zip(row) {
                acc += *qa * *v;
            }
            out[s] = acc;
            s += 1;
        }
    }

    /// Per-electron local-ECP cache (REAL -- byte-identical machinery to the real
    /// [`MeshFastState::ecp_local`]: the local channel is a radial distance
    /// function, untouched by the twist): full build on first use, then only the
    /// hopped electron's entry is re-evaluated.
    #[allow(clippy::too_many_arguments)]
    fn ecp_local(
        &self,
        positions: &ArrayView2<f64>,
        nucleus_index: &[i64],
        exponents: &[f64],
        coefficients: &[f64],
        powers: &[f64],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    ) -> f64 {
        let eval = |r: &ArrayView2<f64>, e: usize| {
            crate::coulomb_potential::compute_ecp_local_one(
                positions,
                nucleus_index,
                exponents,
                coefficients,
                powers,
                [r[[e, 0]], r[[e, 1]], r[[e, 2]]],
                cell_nmax,
            )
        };
        let mut cache = self.ecp_local_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(EcpLocalCache {
                up: (0..r_up.nrows()).map(|e| eval(r_up, e)).collect(),
                dn: (0..r_dn.nrows()).map(|e| eval(r_dn, e)).collect(),
            });
            self.ecp_local_dirty.set(None);
        } else if let Some((is_up, e)) = self.ecp_local_dirty.take() {
            let cache = cache.as_mut().expect("local ECP cache initialized");
            if is_up {
                cache.up[e] = eval(r_up, e);
            } else {
                cache.dn[e] = eval(r_dn, e);
            }
        }
        let cache = cache.as_ref().expect("local ECP cache initialized");
        cache.up.iter().chain(cache.dn.iter()).copied().sum()
    }

    /// Complex twin of [`MeshFastState::screen_interval_ginv`]. A
    /// refresh rebuilds the determinant base and `G^-1` from scratch at the
    /// post-hop positions with the interval seed itself
    /// ([`seed_det_state_and_inverse_complex`]) and re-seeds the winv tables
    /// (`det_winv`) from it. The determinant derivative cache carries a second,
    /// fused-family copy of `G^-1` (`kin_ginv`, with the contraction weights
    /// advanced by the same factors); it is dropped with the refresh and
    /// re-seeded from scratch at the current configuration by the next
    /// `det_grad_lap_c`, so every carried inverse of the interval is re-anchored
    /// by the one decision. Called before the derivative-cache advance of the
    /// hop, which then has nothing to advance.
    fn screen_interval_ginv(
        &mut self,
        gem: &ComplexGeminal,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) {
        let residual = crate::complex_det::det_state_inverse_probe_residual_complex(
            gem,
            &self.det_state,
            &self.ginv.view(),
        );
        let carry = crate::determinant::interval_ginv_carry_ok(residual);
        record_interval_ginv_carry(carry);
        if !carry {
            let (det_state, ginv) = seed_det_state_and_inverse_complex(gem, r_up, r_dn);
            self.det_state = det_state;
            self.ginv = ginv;
            self.det_winv = init_det_wdot_cache_c(gem, &self.det_state, &self.ginv.view());
            bump_route_counter(&GFMC_WINV_BUILDS);
            *self.det_grad_cache.borrow_mut() = None;
        }
    }

    /// Advance the cached complex determinant, J3, and DTN state after the GFMC
    /// categorical hop, mirroring [`MeshFastState::advance_after_hop`].
    /// The complex inverse receives the rank-1 Sherman-Morrison update with
    /// plain-transpose, conjugation-free algebra for the down-spin `-k` leaf,
    /// using exactly the factors [`Self::det_ratio`] contracts. The advanced
    /// `ginv` therefore stays the
    /// inverse of the geminal the streaming det state represents. `r_up`/`r_dn`
    /// are the POST-hop coordinates (the drivers update positions first, exactly
    /// like the real interval drivers).
    #[allow(clippy::too_many_arguments)]
    fn advance_after_hop(
        &mut self,
        gem: &ComplexGeminal,
        jastrow: &JastrowData,
        j3c: Option<&ComplexOrb>,
        j_matrix: Option<&ArrayView2<f64>>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        is_up: bool,
        e: usize,
        old_pos: [f64; 3],
        newpos: [f64; 3],
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let n = self.ginv.nrows();
        let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
        // The SAME single-position kernel family that seeded the base AO matrices
        // (`orb_matrix_via_single_complex`) -- seed/advance fp consistency.
        let ao_col = crate::timed!(
            "gfmc.adv_det_orb_c",
            crate::complex_det::orb_single_complex(orb, newpos)
        );
        crate::timed!("gfmc.adv_det_inverse_c", {
            let mut u = vec![zero; n];
            let mut v = vec![zero; n];
            if is_up {
                u[e] = Complex64::new(1.0, 0.0);
                let vv =
                    crate::complex_det::compute_v_up_move_complex(gem, &self.det_state, e, &ao_col);
                v.copy_from_slice(&vv);
            } else {
                v[e] = Complex64::new(1.0, 0.0);
                let uu = crate::complex_det::compute_u_dn_move_complex(&self.det_state, e, &ao_col);
                u.copy_from_slice(&uu);
            }
            let mut ainv_u = vec![zero; n];
            let mut vt_ainv = vec![zero; n];
            let ratio = crate::complex_det::sm_ratio_complex(
                &self.ginv.view(),
                &u,
                &v,
                &mut ainv_u,
                &mut vt_ainv,
            );
            crate::timed!(
                "gfmc.adv_det_winv_c",
                advance_det_wdot_cache_c(
                    &mut self.det_winv,
                    gem,
                    &self.det_state,
                    is_up,
                    e,
                    &ao_col,
                    &u,
                    &ainv_u,
                    &vt_ainv,
                    ratio,
                )
            );
            bump_route_counter(&GFMC_WINV_ADVANCES);
            for (i, ainv_u_i) in ainv_u[..n].iter().enumerate() {
                let ai = *ainv_u_i / ratio;
                for (k, vt_ainv_k) in vt_ainv[..n].iter().enumerate() {
                    let cur = self.ginv[[i, k]];
                    self.ginv[[i, k]] = cur - ai * *vt_ainv_k;
                }
            }
        });
        crate::timed!(
            "gfmc.adv_det_state_c",
            crate::complex_det::advance_det_ratio_state_complex(
                gem,
                &mut self.det_state,
                is_up,
                e,
                &ao_col,
            )
        );
        crate::timed!(
            "gfmc.adv_det_screen_c",
            self.screen_interval_ginv(gem, r_up, r_dn)
        );
        // Advance the fused-family derivative cache using its own fused
        // single-point column + rank-1 update on its own `kin_ginv`; the value
        // column above is a different fp family by the seeding policy and is
        // deliberately not shared).
        if let Some(cache) = self.det_grad_cache.borrow_mut().as_mut() {
            let orb = if is_up { &gem.orb_up } else { &gem.orb_dn };
            note_deriv_event_c(complex_candidate_class(orb), CandidateEvent::Refresh);
            crate::timed!(
                "gfmc.adv_det_grad_c",
                crate::complex_det::update_det_grad_lap_cache_complex(gem, cache, is_up, e, newpos)
            );
        }
        // Apply the DTN grad/lap moved-electron delta at post-move coordinates.
        if let Some(cache) = self.dtn_grad_cache.borrow_mut().as_mut() {
            let d = jastrow
                .dtn
                .as_ref()
                .expect("DTN derivative cache exists only with a DTN term");
            bump_route_counter(&GFMC_CPLX_DTN_GRAD_REFRESHES);
            crate::timed!(
                "gfmc.adv_dtn_grad_c",
                crate::dtn::update_dtn_grad_lap_cache(d, cache, r_up, r_dn, is_up, e, old_pos)
            );
        }
        // Advance the generic J1/J2 cache by the moved-electron pair delta.
        if let Some(cache) = self.j12_grad_cache.borrow_mut().as_mut() {
            bump_route_counter(&GFMC_CPLX_J12_GRAD_REFRESHES);
            crate::timed!(
                "gfmc.adv_j12_grad_c",
                crate::jastrow_factor::update_j12_grad_lap_cache_generic(
                    jastrow, cache, r_up, r_dn, is_up, e, old_pos,
                )
            );
        }
        // Refresh the complex J3 derivative row without sharing the ratio
        // chain's `j3_ao_re_single` column; the two use different fp families.
        if let (Some(j3), Some(jm)) = (j3c, j_matrix) {
            if let Some(cache) = self.j3_grad_cache.borrow_mut().as_mut() {
                bump_route_counter(&GFMC_CPLX_J3_GRAD_REFRESHES);
                crate::timed!(
                    "gfmc.adv_j3_grad_c",
                    crate::complex_det::update_j3_grad_lap_cache_complex(
                        j3, jm, cache, is_up, e, newpos,
                    )
                );
            }
        }
        if let (Some(j3), Some(jm), Some(st)) = (j3c, j_matrix, self.j3_state.as_mut()) {
            let col = crate::timed!(
                "gfmc.adv_j3_orb_c",
                crate::complex_det::j3_ao_re_single(j3, newpos)
            );
            crate::timed!(
                "gfmc.adv_j3_state_c",
                crate::jastrow_factor::advance_j3_ratio_state(jm, st, is_up, e, &col)
            );
        }
        if let (Some(d), Some(st)) = (&jastrow.dtn, self.dtn_state.as_mut()) {
            crate::timed!(
                "gfmc.adv_dtn_c",
                crate::dtn::advance_dtn_ratio_state(d, st, r_up, r_dn, is_up, e)
            );
            if let Some(table) = self.dtn_candidate_table.borrow_mut().as_mut() {
                crate::timed!(
                    "gfmc.adv_dtn_tabpip_c",
                    crate::dtn::advance_dtn_candidate_table(
                        d, table, st, r_up, r_dn, is_up, e, old_pos,
                    )
                );
                bump_route_counter(&GFMC_DTN_TABPIP_ADVANCES);
            }
        }
        if self.reuse_fixed_mesh {
            let change = HopChange {
                is_up,
                electron: e,
                old_pos,
            };
            self.kinetic_dirty.set(Some(change));
            self.ecp_dirty.set(Some(change));
            self.ecp_dtn_dirty.set(Some(change));
        }
        self.ecp_local_dirty.set(Some((is_up, e)));
    }
}

/// Build the [`MeshFastStateC`] for one configuration, or `None` on a coverage-gate miss
/// (caller then keeps the byte-exact reference path). The only gate mirrors the real
/// `mesh_fast_state`'s J3 gate that still applies on the twist: a non-contiguous
/// `j_matrix` (the rank-1 J3 tables need the contiguous coupling block). The complex AO
/// leaf does its OWN wrap/restore internally (`ComplexOrb::value`), so unlike the real
/// path there is no home-cell wrap to replicate here.
/// Complex twin of [`seed_det_state_and_inverse`]: the base built ONCE on the
/// SINGLE-position AO path so every SM candidate column is fp-consistent with
/// the stored base column (exact det_new/det_base ratio). Shared by the
/// interval seed and the in-interval inverse refresh.
fn seed_det_state_and_inverse_complex(
    gem: &ComplexGeminal,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> (crate::complex_det::DetRatioStateC, Array2<Complex64>) {
    let ao_up = orb_matrix_via_single_complex(&gem.orb_up, r_up);
    let ao_dn = orb_matrix_via_single_complex(&gem.orb_dn, r_dn);
    let paired_dn = if gem.is_sd() {
        ao_dn.clone()
    } else {
        gem.paired_times(&ao_dn)
    };
    let paired_up_lambda = gem.up_t_times_paired(&ao_up);
    let w = gem.w_from_dn(&ao_dn, ao_up.ncols());
    let g = ao_up.t().dot(&w);
    let ginv = crate::complex_det::inverse_complex(&g);
    let det_state = crate::complex_det::DetRatioStateC {
        ao_up,
        ao_dn,
        paired_dn,
        paired_up_lambda,
    };
    (det_state, ginv)
}

pub(crate) fn mesh_fast_state_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
) -> Option<MeshFastStateC> {
    if let Some(jm) = j_matrix {
        jm.as_slice()?;
    }
    let (det_state, ginv) = seed_det_state_and_inverse_complex(gem, r_up, r_dn);
    mesh_fast_state_complex_from(gem, jastrow, j3c, j_matrix, r_up, r_dn, det_state, ginv)
}

/// [`mesh_fast_state_complex`] on a CALLER-supplied determinant base: the orbital tables
/// and the running inverse the caller already carries for the same walker (the FAST
/// chunk passes the sweep's `DetRatioStateC` and its Sherman-Morrison inverse, the
/// same inverse its kinetic term consumes), instead of re-evaluating every orbital
/// column, re-forming `G` and re-inverting it. The two bases describe the same
/// determinant and differ only at the fp floor (batch vs single-position orbital
/// association; SM-carried vs fresh inverse), so this is a FAST-chain-only entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mesh_fast_state_complex_from(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    det_state: crate::complex_det::DetRatioStateC,
    ginv: Array2<Complex64>,
) -> Option<MeshFastStateC> {
    if let Some(jm) = j_matrix {
        jm.as_slice()?;
    }
    let det_winv = init_det_wdot_cache_c(gem, &det_state, &ginv.view());
    bump_route_counter(&GFMC_WINV_BUILDS);
    let j3_state = match (j3c, j_matrix) {
        (Some(j3), Some(jm)) => {
            let au = crate::complex_det::j3_ao_re(j3, r_up);
            let ad = crate::complex_det::j3_ao_re(j3, r_dn);
            Some(crate::jastrow_factor::init_j3_ratio_state_with_policy(
                jm,
                &au.view(),
                &ad.view(),
                jastrow.precision_policy.is_f32("jastrow_ratio"),
            ))
        }
        _ => None,
    };
    let dtn_state = jastrow
        .dtn
        .as_ref()
        .map(|d| crate::dtn::init_dtn_ratio_state(d, r_up, r_dn));
    Some(MeshFastStateC {
        det_state,
        ginv,
        det_winv,
        j3_state,
        dtn_state,
        dtn_candidate_table: std::cell::RefCell::new(None),
        ecp_dtn_candidate_table: std::cell::RefCell::new(None),
        ecp_dtn_dirty: std::cell::Cell::new(None),
        reuse_fixed_mesh: false,
        ecp_wdot: false,
        kinetic_cache: std::cell::RefCell::new(None),
        kinetic_dirty: std::cell::Cell::new(None),
        ecp_cache: std::cell::RefCell::new(None),
        ecp_geometry_cache: std::cell::RefCell::new(None),
        ecp_dirty: std::cell::Cell::new(None),
        ecp_local_cache: std::cell::RefCell::new(None),
        ecp_local_dirty: std::cell::Cell::new(None),
        det_grad_cache: std::cell::RefCell::new(None),
        dtn_grad_cache: std::cell::RefCell::new(None),
        j12_grad_cache: std::cell::RefCell::new(None),
        j3_grad_cache: std::cell::RefCell::new(None),
    })
}

// ============================================================================
// Complex-twist FAST energy chunk: carried kinetic derivative tables.
//
// Without a carry, the complex chunk (`run_energy_chunk(complex=True)`, native stream,
// energy only) rebuilds the kinetic term from scratch at every local energy: fused
// value/grad/lap of every orbital at every electron
// (`grads_and_laplacian_ln_det_complex_with_inv`) plus the full J1/J2/DTN and J3
// grad/lap. [`ComplexChunkKinCarry`] instead holds the SAME projection-scoped
// derivative caches the complex GFMC e_L streams (`MeshFastStateC::det_grad_lap_c`,
// `jas_part_grad_lap`, `j3_grad_lap_c`): seeded once per chunk at the chunk's start
// positions, advanced by replaying the sweep's accepted moves in order (one fused
// single-point column + rank-1 update per move), contracted per e_L. Same math,
// same seeding convention (every det table, its value state and its own `kin_ginv`
// come from the fused fp64 `value_grad_lap` family, never from the sweep's
// ao_eval-zone value chain). FAST chunk only: the per-step dispatch keeps the
// fresh kinetic, bit-exact.
//
// Coordinates: the det, J1/J2 and DTN legs take RAW positions (complex leaves
// home-cell-wrap internally; J1/J2/DTN are min-image), exactly like the fresh
// kinetic; the J3 leg takes home-cell-wrapped positions, as the fresh observable
// kinetic does (`kinetic_withj_complex_assemble(.., wrap_j3 = true)`).
//
// Re-anchoring: after each step's replay the tables' own `kin_ginv` is screened
// with the carried-inverse probe residual against the geminal its value state implies (the
// per-hop rule of `MeshFastStateC::screen_interval_ginv`); a residual above
// `SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD` (or NaN) re-seeds every table at the
// current positions. The carry never crosses a chunk boundary.
// ============================================================================

/// Jastrow legs of [`ComplexChunkKinCarry`].
enum CplxChunkJasLegs {
    /// J1/J2 (`j12_generic_cache_ok`) and DTN carried; a missing leg is the
    /// fresh J1/J2 part (no J1/J2 term: zero tables) / absent.
    Carried {
        // Boxed: the J1/J2 tables are the large leg of this per-chunk enum.
        j12: Option<Box<crate::jastrow_factor::J12GradLapCache>>,
        dtn: Option<crate::dtn::DtnGradLapCache>,
    },
    /// DTN under an fp32 `jastrow_grad_lap` zone: the fresh kernel adds the
    /// J1/J2 and DTN legs in that zone, and this chunk keeps that route (the
    /// DTN cache itself is in the fp64 `jastrow_grad_lap_carry` zone), so the
    /// whole Jastrow part stays fresh per e_L.
    Fresh,
}

/// Chunk-carried complex kinetic derivative tables (see the section comment).
pub(crate) struct ComplexChunkKinCarry {
    det: crate::complex_det::DetGradLapCacheC,
    jas: CplxChunkJasLegs,
    /// Complex J3 tables (`None`: no J3 term, or an unsupported J3 leaf class,
    /// whose J3 leg is evaluated fresh per e_L).
    j3: Option<crate::complex_det::J3GradLapCacheC>,
    /// RAW positions the tables describe; advanced move by move by the replay.
    r_up: Array2<f64>,
    r_dn: Array2<f64>,
    /// Moves applied since the last seed (the probe is skipped right after a seed).
    moves_since_seed: usize,
}

/// Coverage of [`ComplexChunkKinCarry`]: the complex det derivative cache's
/// orbital-class gate (`complex_det_gradlap_ok`, the complex GFMC gate).
#[inline]
pub(crate) fn complex_chunk_kin_carry_covers(gem: &ComplexGeminal) -> bool {
    complex_det_gradlap_ok(gem)
}

impl ComplexChunkKinCarry {
    /// Seed every table at `(r_up, r_dn)` (RAW). Caller-gated by
    /// [`complex_chunk_kin_carry_covers`].
    pub(crate) fn seed(
        gem: &ComplexGeminal,
        jastrow: &JastrowData,
        j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
    ) -> Self {
        bump_route_counter(&RS_CPLX_KIN_CARRY_SEEDS);
        let det = crate::complex_det::init_det_grad_lap_cache_complex(gem, r_up, r_dn);
        let dtn_f32 = jastrow
            .dtn
            .as_ref()
            .is_some_and(|d| d.precision_policy.is_f32("jastrow_grad_lap"));
        let jas = if dtn_f32 {
            CplxChunkJasLegs::Fresh
        } else {
            CplxChunkJasLegs::Carried {
                j12: crate::jastrow_factor::j12_generic_cache_ok(jastrow).then(|| {
                    Box::new(crate::jastrow_factor::init_j12_grad_lap_cache_generic(
                        jastrow, r_up, r_dn,
                    ))
                }),
                dtn: jastrow
                    .dtn
                    .as_ref()
                    .map(|d| crate::dtn::init_dtn_grad_lap_cache(d, r_up, r_dn)),
            }
        };
        let j3 = match j3 {
            Some((orb, jm)) if complex_candidate_class(orb) != CandidateBasisClass::Unsupported => {
                let rwu = crate::atomic_orbital::wrap_r(&orb.aos, r_up);
                let rwd = crate::atomic_orbital::wrap_r(&orb.aos, r_dn);
                Some(crate::complex_det::init_j3_grad_lap_cache_complex(
                    orb,
                    jm,
                    &rwu.view(),
                    &rwd.view(),
                ))
            }
            _ => None,
        };
        ComplexChunkKinCarry {
            det,
            jas,
            j3,
            r_up: r_up.to_owned(),
            r_dn: r_dn.to_owned(),
            moves_since_seed: 0,
        }
    }

    /// Apply one accepted move (in sweep order): the moved electron's fused
    /// det column + rank-1 `kin_ginv` update, the J1/J2 and DTN moved-electron
    /// deltas at the post-move configuration, and the J3 row at the wrapped
    /// new position.
    pub(crate) fn advance(
        &mut self,
        gem: &ComplexGeminal,
        jastrow: &JastrowData,
        j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
        mv: &crate::mcmc::AcceptedMoveC,
    ) {
        bump_route_counter(&RS_CPLX_KIN_CARRY_ADVANCES);
        let (is_up, e, new_pos) = (mv.is_up, mv.index, mv.new_pos);
        {
            let r = if is_up {
                &mut self.r_up
            } else {
                &mut self.r_dn
            };
            debug_assert_eq!([r[[e, 0]], r[[e, 1]], r[[e, 2]]], mv.old_pos);
            for c in 0..3 {
                r[[e, c]] = new_pos[c];
            }
        }
        crate::complex_det::update_det_grad_lap_cache_complex(
            gem,
            &mut self.det,
            is_up,
            e,
            new_pos,
        );
        if let CplxChunkJasLegs::Carried { j12, dtn } = &mut self.jas {
            if let Some(c) = j12.as_mut() {
                crate::jastrow_factor::update_j12_grad_lap_cache_generic(
                    jastrow,
                    c,
                    &self.r_up.view(),
                    &self.r_dn.view(),
                    is_up,
                    e,
                    mv.old_pos,
                );
            }
            if let (Some(c), Some(d)) = (dtn.as_mut(), jastrow.dtn.as_ref()) {
                crate::dtn::update_dtn_grad_lap_cache(
                    d,
                    c,
                    &self.r_up.view(),
                    &self.r_dn.view(),
                    is_up,
                    e,
                    mv.old_pos,
                );
            }
        }
        if let (Some(c), Some((orb, jm))) = (self.j3.as_mut(), j3) {
            crate::complex_det::update_j3_grad_lap_cache_complex(
                orb,
                jm,
                c,
                is_up,
                e,
                crate::atomic_orbital::wrap_pos(&orb.aos, new_pos),
            );
        }
        self.moves_since_seed += 1;
    }

    /// Per-hop residual screen of the tables' own kinetic inverse after a step's replay;
    /// a declined screen re-seeds every table at the current positions.
    pub(crate) fn screen(
        &mut self,
        gem: &ComplexGeminal,
        jastrow: &JastrowData,
        j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
    ) {
        if self.moves_since_seed == 0 {
            return;
        }
        let residual =
            crate::complex_det::det_grad_lap_cache_inverse_residual_complex(gem, &self.det);
        // NaN declines (unlike `interval_ginv_carry_ok`, the re-seed here is the
        // cheap direction: one fresh kinetic-cost build).
        let kin_ginv_carry_ok = residual.is_finite()
            && residual <= crate::determinant::SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD;
        if !kin_ginv_carry_ok {
            bump_route_counter(&RS_CPLX_KIN_GINV_CARRY_DECLINES);
            let (ru, rd) = (self.r_up.clone(), self.r_dn.clone());
            *self = Self::seed(gem, jastrow, j3, &ru.view(), &rd.view());
        }
    }

    /// RAW positions the tables describe (equal to the walker's after a replay).
    pub(crate) fn positions(&self) -> (&Array2<f64>, &Array2<f64>) {
        (&self.r_up, &self.r_dn)
    }

    /// Complex kinetic energy contracted from the carried tables: the det leg
    /// ([`crate::complex_det::contract_det_grad_lap_cache_complex`]), the J1/J2 +
    /// DTN legs in `compute_grads_and_laplacian_jastrow_part`'s order
    /// (((J1)+J2)+DTN), the J3 leg added after them, summed by
    /// [`crate::complex_det::kinetic_complex_from_fields`] -- the fresh kinetic's
    /// assembly.
    pub(crate) fn kinetic(
        &self,
        gem: &ComplexGeminal,
        jastrow: &JastrowData,
        j3: Option<(&ComplexOrb, &ArrayView2<f64>)>,
    ) -> Complex64 {
        let (ru, rd) = (self.r_up.view(), self.r_dn.view());
        let (gd_up, gd_dn, ld_up, ld_dn) =
            crate::complex_det::contract_det_grad_lap_cache_complex(gem, &self.det);
        let (mut gj_up, mut gj_dn, mut lj_up, mut lj_dn) = match &self.jas {
            CplxChunkJasLegs::Fresh => {
                crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(jastrow, &ru, &rd)
            }
            CplxChunkJasLegs::Carried { j12, dtn } => {
                let (gu, gd, lu, ld) = match j12 {
                    Some(c) => crate::jastrow_factor::clone_j12_grad_lap_cache(c),
                    None => crate::jastrow_factor::compute_grads_and_laplacian_j12_part(
                        jastrow, &ru, &rd,
                    ),
                };
                match dtn {
                    Some(c) => {
                        let (dgu, dgd, dlu, dld) = crate::dtn::clone_dtn_grad_lap_cache(c);
                        (
                            gu + dgu,
                            gd + dgd,
                            lu + numpy::ndarray::Array1::from_vec(dlu),
                            ld + numpy::ndarray::Array1::from_vec(dld),
                        )
                    }
                    None => (gu, gd, lu, ld),
                }
            }
        };
        if let Some((orb, jm)) = j3 {
            let (g3u, g3d, l3u, l3d) = match &self.j3 {
                Some(c) => crate::complex_det::contract_j3_grad_lap_cache_complex(jm, c),
                None => {
                    let rwu = crate::atomic_orbital::wrap_r(&orb.aos, &ru);
                    let rwd = crate::atomic_orbital::wrap_r(&orb.aos, &rd);
                    j3_grads_lap_complex(orb, jm, &rwu.view(), &rwd.view())
                }
            };
            gj_up = gj_up + g3u;
            gj_dn = gj_dn + g3d;
            lj_up = lj_up + l3u;
            lj_dn = lj_dn + l3d;
        }
        crate::complex_det::kinetic_complex_from_fields(
            &gd_up, &gd_dn, &ld_up, &ld_dn, &gj_up, &gj_dn, &lj_up, &lj_dn,
        )
    }
}

/// FAST complex discretized-kinetic hopping amplitudes: same layout / sign / value as the
/// reference loop inside [`gfmc_lrdmc_ae_complex`] (`element = -1/(2a^2) * det_ratio * jr`,
/// up-major shift-minor then dn), with the per-point full complex geminal + LU + full
/// Jastrow value replaced by the [`MeshFastStateC`] rank-1 machinery. Differs from the
/// reference only at the association-class fp floor -- FAST chain mode only.
///
/// When the interval driver holds the mesh rotation fixed,
/// (`reuse_fixed_mesh`), the complex det orbital rows and the real `Re(complex AO)` J3
/// rows for all 6N candidates live in [`KineticAoCacheC`] across the interval, with a
/// per-hop refresh of only the moved electron's six rows. Candidate positions are RAW
/// (unwrapped): the complex leaves home-cell-wrap internally (`WrapCtx` / B-spline
/// fractional wrap / PW Bloch phase), so the real path's single external wrap site
/// has no complex counterpart. J1/J2 and DTN take the raw positions
/// per point through the SAME streaming kernels the per-point path uses.
#[allow(clippy::too_many_arguments)]
fn discretized_kinetic_elements_complex_fast(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    st: &MeshFastStateC,
) -> Vec<Complex64> {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let shifts = lattice_shifts(alat, rt);
    let inv_2a2 = Complex64::new(-1.0 / (2.0 * alat * alat), 0.0);
    // Candidate layout: electron-major (global up 0..n_up then dn), shift-minor --
    // exactly the reference element order.
    let mut allpos = Vec::with_capacity(6 * (n_up + n_dn));
    for e in 0..n_up {
        for shift in &shifts {
            allpos.push([
                r_up[[e, 0]] + shift[0],
                r_up[[e, 1]] + shift[1],
                r_up[[e, 2]] + shift[2],
            ]);
        }
    }
    for e in 0..n_dn {
        for shift in &shifts {
            allpos.push([
                r_dn[[e, 0]] + shift[0],
                r_dn[[e, 1]] + shift[1],
                r_dn[[e, 2]] + shift[2],
            ]);
        }
    }
    st.ensure_kinetic_dtn_candidate_table(jastrow, r_up, r_dn, &shifts);
    let det_up_class = complex_candidate_class(&gem.orb_up);
    let det_dn_class = complex_candidate_class(&gem.orb_dn);
    let det_up_fast = det_up_class != CandidateBasisClass::Unsupported;
    let det_dn_fast = det_dn_class != CandidateBasisClass::Unsupported;
    if !det_up_fast || !det_dn_fast {
        bump_route_counter(&GFMC_CPLX_CAND_DET_SLOW);
    }
    // J3 table route: requires the streaming J3 state (present iff j3c + contiguous
    // j_matrix at build) and a supported J3 orb class for the batched Re(AO) rows.
    let j3_class = j3c.map(complex_candidate_class);
    let j3_fast = j3c.is_none()
        || (st.j3_state.is_some() && j3_class != Some(CandidateBasisClass::Unsupported));
    if j3c.is_some() && !j3_fast {
        note_j3_slow_fallback(&GFMC_CPLX_J3_SLOW_FALLBACKS);
    }
    let j12_cache_ok = crate::jastrow_factor::j12_candidate_cache_ok(jastrow);
    let cache_ok = st.reuse_fixed_mesh && det_up_fast && det_dn_fast && j3_fast;
    let mut cache_guard = cache_ok.then(|| st.kinetic_cache.borrow_mut());
    if let Some(slot) = cache_guard.as_mut() {
        let mut rt_bits = [0_u64; 9];
        for a in 0..3 {
            for b in 0..3 {
                rt_bits[3 * a + b] = rt[a][b].to_bits();
            }
        }
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.alat_bits != alat.to_bits()
                || c.rt_bits != rt_bits
                || c.det_up.nrows() != 6 * n_up
                || c.det_dn.nrows() != 6 * n_dn
        });
        if rebuild {
            let det_up = crate::timed!(
                "gfmc.kin_det_ao_up_c",
                complex_det_candidate_batch(
                    &gem.orb_up,
                    det_up_class,
                    &allpos[..6 * n_up],
                    CandidateEvent::Build,
                )
            );
            let det_dn = crate::timed!(
                "gfmc.kin_det_ao_dn_c",
                complex_det_candidate_batch(
                    &gem.orb_dn,
                    det_dn_class,
                    &allpos[6 * n_up..],
                    CandidateEvent::Build,
                )
            );
            let j3_cache = j3c.map(|orb| {
                crate::timed!(
                    "gfmc.kin_j3_ao_c",
                    complex_j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3c"),
                        &allpos,
                        CandidateEvent::Build,
                    )
                )
            });
            let (j12_up, j12_dn) = if j12_cache_ok {
                let mut up = vec![0.0; 6 * n_up];
                let mut dn = vec![0.0; 6 * n_dn];
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    true,
                    &allpos[..6 * n_up],
                    6,
                    &mut up,
                );
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    false,
                    &allpos[6 * n_up..],
                    6,
                    &mut dn,
                );
                bump_route_counter(&GFMC_J12_TABPIP_KIN_BUILDS);
                (Some(up), Some(dn))
            } else {
                (None, None)
            };
            **slot = Some(KineticAoCacheC {
                alat_bits: alat.to_bits(),
                rt_bits,
                det_up,
                det_dn,
                j3: j3_cache,
                j12_up,
                j12_dn,
            });
            st.kinetic_dirty.set(None);
        } else if let Some(change) = st.kinetic_dirty.take() {
            let is_up = change.is_up;
            let e = change.electron;
            let global_e = if is_up { e } else { n_up + e };
            // Refresh rows come from the SAME kernel family the table was seeded
            // from (one batched `ComplexOrb::value` call) -- seed/refresh fp
            // consistency within one cache.
            let pos = &allpos[6 * global_e..6 * global_e + 6];
            let cache = slot.as_mut().expect("complex kinetic cache initialized");
            if is_up {
                let fresh = crate::timed!(
                    "gfmc.kin_det_ao_up_c",
                    complex_det_candidate_batch(
                        &gem.orb_up,
                        det_up_class,
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                cache
                    .det_up
                    .slice_mut(numpy::ndarray::s![6 * e..6 * e + 6, ..])
                    .assign(&fresh);
            } else {
                let fresh = crate::timed!(
                    "gfmc.kin_det_ao_dn_c",
                    complex_det_candidate_batch(
                        &gem.orb_dn,
                        det_dn_class,
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                cache
                    .det_dn
                    .slice_mut(numpy::ndarray::s![6 * e..6 * e + 6, ..])
                    .assign(&fresh);
            }
            if let (Some(orb), Some(aos)) = (j3c, cache.j3.as_mut()) {
                let fresh = crate::timed!(
                    "gfmc.kin_j3_ao_c",
                    complex_j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3c"),
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                aos.slice_mut(numpy::ndarray::s![6 * global_e..6 * global_e + 6, ..])
                    .assign(&fresh);
            }
            if let (Some(up), Some(dn)) = (&mut cache.j12_up, &mut cache.j12_dn) {
                crate::jastrow_factor::update_j12_candidate_deltas_after_hop(
                    jastrow,
                    r_up,
                    r_dn,
                    change.is_up,
                    change.electron,
                    change.old_pos,
                    &allpos[..6 * n_up],
                    &allpos[6 * n_up..],
                    6,
                    up,
                    dn,
                );
                bump_route_counter(&GFMC_J12_TABPIP_KIN_ADVANCES);
            }
        }
    }
    // Owned per-evaluation tables when the interval cache did not engage (stateless
    // oracle entry points, or a supported class without `reuse_fixed_mesh`).
    let det_up_owned = if !cache_ok && det_up_fast {
        Some(crate::timed!(
            "gfmc.kin_det_ao_up_c",
            complex_det_candidate_batch(
                &gem.orb_up,
                det_up_class,
                &allpos[..6 * n_up],
                CandidateEvent::Build,
            )
        ))
    } else {
        None
    };
    let det_dn_owned = if !cache_ok && det_dn_fast {
        Some(crate::timed!(
            "gfmc.kin_det_ao_dn_c",
            complex_det_candidate_batch(
                &gem.orb_dn,
                det_dn_class,
                &allpos[6 * n_up..],
                CandidateEvent::Build,
            )
        ))
    } else {
        None
    };
    let j3_owned = j3c.and_then(|orb| {
        if !cache_ok && j3_fast && st.j3_state.is_some() {
            Some(crate::timed!(
                "gfmc.kin_j3_ao_c",
                complex_j3_candidate_batch(
                    orb,
                    j3_class.expect("J3 class computed alongside j3c"),
                    &allpos,
                    CandidateEvent::Build,
                )
            ))
        } else {
            None
        }
    });
    let cached = cache_guard.as_ref().and_then(|slot| slot.as_ref());
    let det_up_aos = cached.map(|c| &c.det_up).or(det_up_owned.as_ref());
    let det_dn_aos = cached.map(|c| &c.det_dn).or(det_dn_owned.as_ref());
    let j3_aos = match cached {
        Some(c) => c.j3.as_ref(),
        None => j3_owned.as_ref(),
    };
    let j12_up = cached.and_then(|c| c.j12_up.as_deref());
    let j12_dn = cached.and_then(|c| c.j12_dn.as_deref());
    let mut out = Vec::with_capacity(6 * (n_up + n_dn));
    let zero = Complex64::new(0.0, 0.0);
    let m_up = st.det_state.ao_up.nrows();
    let m_dn = st.det_state.ao_dn.nrows();
    let mut q_scratch = vec![zero; m_up.max(m_dn)];
    let mut ratios = vec![zero; 6];
    // Free closure over the moving spin's coords (higher-ranked borrow, the same pattern
    // the real reference's `do_spin` uses): the two spin views have independent lifetimes.
    {
        let mut do_spin = |is_up: bool, r: &ArrayView2<f64>| {
            for e in 0..r.nrows() {
                let global_e = if is_up { e } else { n_up + e };
                let group = &allpos[6 * global_e..6 * global_e + 6];
                crate::timed!(
                    "gfmc.kin_det_ratio_c",
                    match if is_up { det_up_aos } else { det_dn_aos } {
                        Some(aos) => {
                            bump_route_counter(&GFMC_WINV_KIN_HITS);
                            let local = aos.slice(numpy::ndarray::s![6 * e..6 * e + 6, ..]);
                            st.det_ratios_from_candidate_aos_c_into(
                                gem,
                                is_up,
                                e,
                                &local,
                                &mut q_scratch,
                                &mut ratios,
                            );
                        }
                        None => {
                            for (s, &pos) in group.iter().enumerate() {
                                ratios[s] = st.det_ratio(gem, is_up, e, pos);
                            }
                        }
                    }
                );
                // Use one batched DTN mesh-ratio evaluation per electron
                // group (the real path's convention) instead of six per-point
                // streaming reductions.
                let dtn_ratios = crate::timed!(
                    "gfmc.kin_dtn_ratio_c",
                    st.dtn_group_ratios(jastrow, r_up, r_dn, is_up, e, group)
                );
                let j12 = if is_up { j12_up } else { j12_dn }.map(|v| &v[6 * e..6 * e + 6]);
                if j12_cache_ok {
                    if j12.is_some() {
                        bump_route_counter(&GFMC_J12_TABPIP_KIN_HITS);
                    } else {
                        bump_route_counter(&GFMC_J12_TABPIP_KIN_FALLBACKS);
                    }
                }
                for (s, &pos) in group.iter().enumerate() {
                    let j3_row = j3_aos.map(|t| t.row(6 * global_e + s));
                    let j3_col = j3_row
                        .as_ref()
                        .map(|row| row.as_slice().expect("J3 candidate row contiguous"));
                    let jr = crate::timed!(
                        "gfmc.kin_jas_ratio_c",
                        st.jas_ratio_with_j3_col(
                            jastrow,
                            j3c,
                            j_matrix,
                            r_up,
                            r_dn,
                            is_up,
                            e,
                            pos,
                            j12.map(|row| row[s]),
                            j3_col,
                            dtn_ratios.as_ref().map(|v| v[s]),
                        )
                    );
                    out.push(inv_2a2 * ratios[s] * jr);
                }
            }
        };
        do_spin(true, r_up);
        do_spin(false, r_dn);
    }
    out
}

/// Fast complex ECP T-move quadrature mesh with the same candidate order and
/// per-candidate quantities as [`crate::complex_ecp::compute_ecp_nonlocal_mesh_complex`]
/// (`(is_up, i, new_pos, ang * det_ratio, jastrow_ratio)`; up electrons then dn,
/// per electron the nearest-nucleus set in distance order, quadrature-minor), but
/// with the per-point complex det ratio driven off the [`MeshFastStateC`] candidate
/// tables and the REAL quadrature geometry / angular factors cached across the
/// interval in the real path's [`EcpGeometryCache`] (the geometry is untouched by
/// the twist). The `ang` (radial x Legendre x weight) factor is reproduced verbatim
/// from the reference. NO Re anywhere here: the fixed-phase per-element Re
/// projection stays at the consumption site in [`gfmc_lrdmc_ecp_complex`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_ecp_nonlocal_mesh_complex_fast(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    nucleus_index: &[i64],
    ang_mom: &[i64],
    exponents: &[f64],
    coefficients: &[f64],
    powers: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    rt: &[[f64; 3]; 3],
    nn: usize,
    cell_nmax: Option<(&ArrayView2<f64>, i32)>,
    st: &MeshFastStateC,
) -> Vec<(bool, usize, [f64; 3], Complex64, f64)> {
    let nv = weights.len();
    let n_up = r_up.nrows();
    let inv_cell = cell_nmax.map(|(c, _)| crate::ewald::inv3(c));
    let mic = |v: [f64; 3]| -> [f64; 3] {
        match (cell_nmax, &inv_cell) {
            (Some((cell, n)), Some(iv)) => crate::ewald::min_image(v, cell, iv, n),
            _ => v,
        }
    };
    let mut rt_bits = [0_u64; 9];
    for a in 0..3 {
        for b in 0..3 {
            rt_bits[3 * a + b] = rt[a][b].to_bits();
        }
    }
    // Rotate the quadrature grid by rt (jqmc: grid_points @ RT). Identity when rt=I.
    let gp: Vec<[f64; 3]> = (0..nv)
        .map(|v| {
            let mut o = [0.0_f64; 3];
            for c in 0..3 {
                for k in 0..3 {
                    o[c] += grid_points[[v, k]] * rt[k][c];
                }
            }
            o
        })
        .collect();
    let uniq: Vec<usize> = {
        let mut u: Vec<usize> = nucleus_index.iter().map(|&x| x as usize).collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let nearest_set = |rx: f64, ry: f64, rz: f64| -> Vec<usize> {
        if nn == 0 || nn >= uniq.len() {
            return uniq.clone();
        }
        let mut d: Vec<(f64, usize)> = uniq
            .iter()
            .map(|&ia| {
                let dd = mic([
                    rx - positions[[ia, 0]],
                    ry - positions[[ia, 1]],
                    rz - positions[[ia, 2]],
                ]);
                (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2], ia)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.iter().take(nn).map(|&(_, ia)| ia).collect()
    };
    let nearest_count = if nn == 0 || nn >= uniq.len() {
        uniq.len()
    } else {
        nn
    };
    let rows_per_electron = nearest_count * nv;
    // Per-candidate geometry + angular factor, verbatim the reference conventions
    // (`rel = mic(position - r)`, `cosv`, `new_pos = r + rel + nrm * d`, `ang` =
    // the per-nucleus channel sum) -- identical math to the real fast mesh; the
    // twist never touches it.
    let build_electron =
        |r: &ArrayView2<f64>, i: usize, allpos: &mut Vec<[f64; 3]>, allangs: &mut Vec<f64>| {
            let near = nearest_set(r[[i, 0]], r[[i, 1]], r[[i, 2]]);
            for &ia in &near {
                let rel = mic([
                    positions[[ia, 0]] - r[[i, 0]],
                    positions[[ia, 1]] - r[[i, 1]],
                    positions[[ia, 2]] - r[[i, 2]],
                ]);
                let (rx, ry, rz) = (rel[0], rel[1], rel[2]);
                let nrm = (rx * rx + ry * ry + rz * rz).sqrt();
                for v in 0..nv {
                    let (dx, dy, dz) = (gp[v][0], gp[v][1], gp[v][2]);
                    let dnv = (dx * dx + dy * dy + dz * dz).sqrt();
                    let cosv = (-rx * dx - ry * dy - rz * dz) / (nrm * dnv);
                    let np = [
                        r[[i, 0]] + rx + nrm * dx,
                        r[[i, 1]] + ry + nrm * dy,
                        r[[i, 2]] + rz + nrm * dz,
                    ];
                    let mut ang = 0.0_f64;
                    for t in 0..nucleus_index.len() {
                        if nucleus_index[t] as usize != ia {
                            continue;
                        }
                        let l = ang_mom[t];
                        let v_l = coefficients[t]
                            * crate::coulomb_potential::ecp_pow(nrm, powers[t] - 2.0)
                            * (-exponents[t] * nrm * nrm).exp();
                        ang += v_l
                            * ((2 * l + 1) as f64)
                            * crate::coulomb_potential::legendre(l, cosv)
                            * weights[v];
                    }
                    allpos.push(np);
                    allangs.push(ang);
                }
            }
        };
    let build_spin = |r: &ArrayView2<f64>| {
        let mut allpos = Vec::with_capacity(r.nrows() * rows_per_electron);
        let mut allangs = Vec::with_capacity(r.nrows() * rows_per_electron);
        for i in 0..r.nrows() {
            build_electron(r, i, &mut allpos, &mut allangs);
        }
        (allpos, allangs)
    };
    let (all_up, angular_up, all_dn, angular_dn) = if st.reuse_fixed_mesh {
        let mut slot = st.ecp_geometry_cache.borrow_mut();
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.rt_bits != rt_bits
                || c.rows_per_electron != rows_per_electron
                || c.positions_up.len() != r_up.nrows() * rows_per_electron
                || c.positions_dn.len() != r_dn.nrows() * rows_per_electron
        });
        if rebuild {
            let ((positions_up, angular_up), (positions_dn, angular_dn)) =
                (build_spin(r_up), build_spin(r_dn));
            *slot = Some(EcpGeometryCache {
                rt_bits,
                rows_per_electron,
                positions_up,
                positions_dn,
                angular_up,
                angular_dn,
            });
        } else if let Some(change) = st.ecp_dirty.get() {
            let is_up = change.is_up;
            let e = change.electron;
            let mut positions_fresh = Vec::with_capacity(rows_per_electron);
            let mut angular_fresh = Vec::with_capacity(rows_per_electron);
            if is_up {
                build_electron(r_up, e, &mut positions_fresh, &mut angular_fresh);
            } else {
                build_electron(r_dn, e, &mut positions_fresh, &mut angular_fresh);
            }
            debug_assert_eq!(positions_fresh.len(), rows_per_electron);
            let offset = e * rows_per_electron;
            let cache = slot.as_mut().expect("ECP geometry cache initialized");
            let (positions_cached, angular_cached) = if is_up {
                (&mut cache.positions_up, &mut cache.angular_up)
            } else {
                (&mut cache.positions_dn, &mut cache.angular_dn)
            };
            positions_cached[offset..offset + rows_per_electron].copy_from_slice(&positions_fresh);
            angular_cached[offset..offset + rows_per_electron].copy_from_slice(&angular_fresh);
        }
        let cache = slot.as_ref().expect("ECP geometry cache initialized");
        (
            cache.positions_up.clone(),
            cache.angular_up.clone(),
            cache.positions_dn.clone(),
            cache.angular_dn.clone(),
        )
    } else {
        let (positions_up, angular_up) = build_spin(r_up);
        let (positions_dn, angular_dn) = build_spin(r_dn);
        (positions_up, angular_up, positions_dn, angular_dn)
    };

    st.ensure_ecp_dtn_candidate_table(jastrow, r_up, r_dn, &all_up, &all_dn, rows_per_electron);

    let det_up_class = complex_candidate_class(&gem.orb_up);
    let det_dn_class = complex_candidate_class(&gem.orb_dn);
    let det_up_fast = det_up_class != CandidateBasisClass::Unsupported;
    let det_dn_fast = det_dn_class != CandidateBasisClass::Unsupported;
    if !det_up_fast || !det_dn_fast {
        bump_route_counter(&GFMC_CPLX_CAND_DET_SLOW);
    }
    let j3_class = j3c.map(complex_candidate_class);
    let j3_fast = j3c.is_none()
        || (st.j3_state.is_some() && j3_class != Some(CandidateBasisClass::Unsupported));
    if j3c.is_some() && !j3_fast {
        note_j3_slow_fallback(&GFMC_CPLX_J3_SLOW_FALLBACKS);
    }
    // FAST AO-space w-dot determinant ratio (one-shot chunk e_L only): each spin
    // independently, recorded once per spin per evaluation where it is eligible.
    let det_up_wdot_ok = st.ecp_wdot && det_up_fast && gem.orb_up.mesh_wdot_ok();
    let det_dn_wdot_ok = st.ecp_wdot && det_dn_fast && gem.orb_dn.mesh_wdot_ok();
    if st.ecp_wdot {
        record_ecp_cplx_wdot(det_up_wdot_ok);
        record_ecp_cplx_wdot(det_dn_wdot_ok);
    }
    let j12_cache_ok = crate::jastrow_factor::j12_candidate_cache_ok(jastrow);
    let cache_ok = st.reuse_fixed_mesh
        && det_up_fast
        && det_dn_fast
        && j3_fast
        && rows_per_electron * n_up == all_up.len()
        && rows_per_electron * r_dn.nrows() == all_dn.len();
    let mut allpos = Vec::with_capacity(all_up.len() + all_dn.len());
    allpos.extend_from_slice(&all_up);
    allpos.extend_from_slice(&all_dn);
    let mut cache_guard = cache_ok.then(|| st.ecp_cache.borrow_mut());
    if let Some(slot) = cache_guard.as_mut() {
        let rebuild = slot.as_ref().is_none_or(|c| {
            c.rt_bits != rt_bits
                || c.rows_per_electron != rows_per_electron
                || c.rows_up != all_up.len()
                || c.rows_dn != all_dn.len()
                || c.det_up.is_some() == det_up_wdot_ok
                || c.det_dn.is_some() == det_dn_wdot_ok
        });
        if rebuild {
            let det_up = (!det_up_wdot_ok).then(|| {
                crate::timed!(
                    "gfmc.ecp_det_ao_up_c",
                    complex_det_candidate_batch(
                        &gem.orb_up,
                        det_up_class,
                        &all_up,
                        CandidateEvent::Build,
                    )
                )
            });
            let det_dn = (!det_dn_wdot_ok).then(|| {
                crate::timed!(
                    "gfmc.ecp_det_ao_dn_c",
                    complex_det_candidate_batch(
                        &gem.orb_dn,
                        det_dn_class,
                        &all_dn,
                        CandidateEvent::Build,
                    )
                )
            });
            let j3_cache = j3c.map(|orb| {
                crate::timed!(
                    "gfmc.ecp_j3_ao_c",
                    complex_j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3c"),
                        &allpos,
                        CandidateEvent::Build,
                    )
                )
            });
            let (j12_up, j12_dn) = if j12_cache_ok {
                let mut up = vec![0.0; all_up.len()];
                let mut dn = vec![0.0; all_dn.len()];
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    true,
                    &all_up,
                    rows_per_electron,
                    &mut up,
                );
                crate::jastrow_factor::fill_j12_candidate_deltas(
                    jastrow,
                    r_up,
                    r_dn,
                    false,
                    &all_dn,
                    rows_per_electron,
                    &mut dn,
                );
                bump_route_counter(&GFMC_J12_TABPIP_ECP_BUILDS);
                (Some(up), Some(dn))
            } else {
                (None, None)
            };
            **slot = Some(EcpAoCacheC {
                rt_bits,
                rows_per_electron,
                rows_up: all_up.len(),
                rows_dn: all_dn.len(),
                det_up,
                det_dn,
                j3: j3_cache,
                j12_up,
                j12_dn,
            });
            st.ecp_dirty.set(None);
        } else if let Some(change) = st.ecp_dirty.take() {
            let is_up = change.is_up;
            let e = change.electron;
            let spin_pos: &[[f64; 3]] = if is_up { &all_up } else { &all_dn };
            let off = rows_per_electron * e;
            let pos = &spin_pos[off..off + rows_per_electron];
            let cache = slot.as_mut().expect("complex ECP cache initialized");
            // The w-dot route carries no determinant table to refresh.
            if is_up {
                if let Some(det_up) = cache.det_up.as_mut() {
                    let fresh = crate::timed!(
                        "gfmc.ecp_det_ao_up_c",
                        complex_det_candidate_batch(
                            &gem.orb_up,
                            det_up_class,
                            pos,
                            CandidateEvent::Refresh,
                        )
                    );
                    det_up
                        .slice_mut(numpy::ndarray::s![off..off + rows_per_electron, ..])
                        .assign(&fresh);
                }
            } else if let Some(det_dn) = cache.det_dn.as_mut() {
                let fresh = crate::timed!(
                    "gfmc.ecp_det_ao_dn_c",
                    complex_det_candidate_batch(
                        &gem.orb_dn,
                        det_dn_class,
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                det_dn
                    .slice_mut(numpy::ndarray::s![off..off + rows_per_electron, ..])
                    .assign(&fresh);
            }
            if let (Some(orb), Some(aos)) = (j3c, cache.j3.as_mut()) {
                let global_off = if is_up { off } else { all_up.len() + off };
                let fresh = crate::timed!(
                    "gfmc.ecp_j3_ao_c",
                    complex_j3_candidate_batch(
                        orb,
                        j3_class.expect("J3 class computed alongside j3c"),
                        pos,
                        CandidateEvent::Refresh,
                    )
                );
                aos.slice_mut(numpy::ndarray::s![
                    global_off..global_off + rows_per_electron,
                    ..
                ])
                .assign(&fresh);
            }
            if let (Some(up), Some(dn)) = (&mut cache.j12_up, &mut cache.j12_dn) {
                crate::jastrow_factor::update_j12_candidate_deltas_after_hop(
                    jastrow,
                    r_up,
                    r_dn,
                    change.is_up,
                    change.electron,
                    change.old_pos,
                    &all_up,
                    &all_dn,
                    rows_per_electron,
                    up,
                    dn,
                );
                bump_route_counter(&GFMC_J12_TABPIP_ECP_ADVANCES);
            }
        }
    }
    let det_up_owned = if !cache_ok && det_up_fast && !det_up_wdot_ok {
        Some(crate::timed!(
            "gfmc.ecp_det_ao_up_c",
            complex_det_candidate_batch(&gem.orb_up, det_up_class, &all_up, CandidateEvent::Build)
        ))
    } else {
        None
    };
    let det_dn_owned = if !cache_ok && det_dn_fast && !det_dn_wdot_ok {
        Some(crate::timed!(
            "gfmc.ecp_det_ao_dn_c",
            complex_det_candidate_batch(&gem.orb_dn, det_dn_class, &all_dn, CandidateEvent::Build)
        ))
    } else {
        None
    };
    let j3_owned = j3c.and_then(|orb| {
        if !cache_ok && j3_fast && st.j3_state.is_some() {
            Some(crate::timed!(
                "gfmc.ecp_j3_ao_c",
                complex_j3_candidate_batch(
                    orb,
                    j3_class.expect("J3 class computed alongside j3c"),
                    &allpos,
                    CandidateEvent::Build,
                )
            ))
        } else {
            None
        }
    });
    let cached = cache_guard.as_ref().and_then(|slot| slot.as_ref());
    let det_up_aos = cached
        .and_then(|c| c.det_up.as_ref())
        .or(det_up_owned.as_ref());
    let det_dn_aos = cached
        .and_then(|c| c.det_dn.as_ref())
        .or(det_dn_owned.as_ref());
    // AO-space w-dot weights `q C` per spin (rows = moved electron), and the shared
    // contraction scratch.
    let wdot_up = det_up_wdot_ok.then(|| {
        crate::timed!(
            "gfmc.ecp_det_wbuild_c",
            crate::complex_det::mesh_wdot_weights(&gem.orb_up, &st.det_winv.up.view(), n_up)
        )
    });
    let wdot_dn = det_dn_wdot_ok.then(|| {
        crate::timed!(
            "gfmc.ecp_det_wbuild_c",
            crate::complex_det::mesh_wdot_weights(
                &gem.orb_dn,
                &st.det_winv.dn.view(),
                r_dn.nrows()
            )
        )
    });
    let mut wdot_scratch = crate::complex_det::MeshWdotScratch::default();
    let mut wdot_vals = vec![Complex64::new(0.0, 0.0); nv];
    let j3_aos = match cached {
        Some(c) => c.j3.as_ref(),
        None => j3_owned.as_ref(),
    };
    let j12_up = cached.and_then(|c| c.j12_up.as_deref());
    let j12_dn = cached.and_then(|c| c.j12_dn.as_deref());

    struct CandidateGroupC {
        is_up: bool,
        electron: usize,
        spin_offset: usize,
    }
    let mut groups = Vec::<CandidateGroupC>::with_capacity((n_up + r_dn.nrows()) * nearest_count);
    for (is_up, nr) in [(true, n_up), (false, r_dn.nrows())] {
        for electron in 0..nr {
            for nearest in 0..nearest_count {
                groups.push(CandidateGroupC {
                    is_up,
                    electron,
                    spin_offset: electron * rows_per_electron + nearest * nv,
                });
            }
        }
    }

    let zero = Complex64::new(0.0, 0.0);
    let m_up = st.det_state.ao_up.nrows();
    let m_dn = st.det_state.ao_dn.nrows();
    let mut q_scratch = vec![zero; m_up.max(m_dn)];
    let mut ratios = vec![zero; nv];
    let mut out: Vec<(bool, usize, [f64; 3], Complex64, f64)> = Vec::with_capacity(allpos.len());
    for group in groups {
        let (spin_pos, spin_angs): (&[[f64; 3]], &[f64]) = if group.is_up {
            (&all_up, &angular_up)
        } else {
            (&all_dn, &angular_dn)
        };
        let newpos = &spin_pos[group.spin_offset..group.spin_offset + nv];
        let angs = &spin_angs[group.spin_offset..group.spin_offset + nv];
        let (wdot, wdot_orb, wdot_base) = if group.is_up {
            (
                wdot_up.as_ref(),
                &gem.orb_up,
                st.det_winv.up_base.get(group.electron),
            )
        } else {
            (
                wdot_dn.as_ref(),
                &gem.orb_dn,
                st.det_winv.dn_base.get(group.electron),
            )
        };
        if let Some((w_re, w_im)) = wdot {
            // FAST AO-space w-dot: ratio = (1 - base) + (q C) . AO(new), the same
            // affine form as the table route below with the MO fold moved into the
            // electron-fixed weights.
            crate::timed!("gfmc.ecp_det_wdot_c", {
                let w_re = w_re.row(group.electron);
                let w_im = w_im.row(group.electron);
                wdot_orb.mesh_wdot_into(
                    w_re.as_slice().expect("w-dot weight row contiguous"),
                    w_im.as_slice().expect("w-dot weight row contiguous"),
                    newpos,
                    &mut wdot_scratch,
                    &mut wdot_vals,
                );
                let affine_base = Complex64::new(1.0, 0.0)
                    - *wdot_base.expect("w-dot base row for every moved electron");
                for (r, d) in ratios.iter_mut().zip(&wdot_vals) {
                    *r = affine_base + *d;
                }
            });
        } else {
            crate::timed!(
                "gfmc.ecp_det_ratio_c",
                match if group.is_up { det_up_aos } else { det_dn_aos } {
                    Some(aos) => {
                        bump_route_counter(&GFMC_WINV_ECP_HITS);
                        let local = aos.slice(numpy::ndarray::s![
                            group.spin_offset..group.spin_offset + nv,
                            ..
                        ]);
                        st.det_ratios_from_candidate_aos_c_into(
                            gem,
                            group.is_up,
                            group.electron,
                            &local,
                            &mut q_scratch,
                            &mut ratios,
                        );
                    }
                    None => {
                        for (v, &pos) in newpos.iter().enumerate() {
                            ratios[v] = st.det_ratio(gem, group.is_up, group.electron, pos);
                        }
                    }
                }
            );
        }
        // Use one batched DTN mesh-ratio evaluation per quadrature group
        // (the real path's convention) instead of nv per-point streaming
        // reductions.
        let dtn_ratios = crate::timed!(
            "gfmc.ecp_dtn_ratio_c",
            st.dtn_group_ratios(jastrow, r_up, r_dn, group.is_up, group.electron, newpos)
        );
        let j12 = if group.is_up { j12_up } else { j12_dn }
            .map(|row| &row[group.spin_offset..group.spin_offset + nv]);
        if j12_cache_ok {
            if j12.is_some() {
                bump_route_counter(&GFMC_J12_TABPIP_ECP_HITS);
            } else {
                bump_route_counter(&GFMC_J12_TABPIP_ECP_FALLBACKS);
            }
        }
        for (v, &pos) in newpos.iter().enumerate() {
            let global_row = if group.is_up {
                group.spin_offset + v
            } else {
                all_up.len() + group.spin_offset + v
            };
            let j3_row = j3_aos.map(|t| t.row(global_row));
            let j3_col = j3_row
                .as_ref()
                .map(|row| row.as_slice().expect("J3 candidate row contiguous"));
            let jr = crate::timed!(
                "gfmc.ecp_jas_ratio_c",
                st.jas_ratio_with_j3_col(
                    jastrow,
                    j3c,
                    j_matrix,
                    r_up,
                    r_dn,
                    group.is_up,
                    group.electron,
                    pos,
                    j12.map(|row| row[v]),
                    j3_col,
                    dtn_ratios.as_ref().map(|rs| rs[v]),
                )
            );
            out.push((
                group.is_up,
                group.electron,
                pos,
                Complex64::new(angs[v], 0.0) * ratios[v],
                jr,
            ));
        }
    }
    out
}

/// Complex Bloch-twist ECP LRDMC diagonal e_L (Casula tmove, fixed-phase). Adds to the complex
/// AE machinery: ZV el-ion max-arg = BARE Ewald ei (ECP parcutg=1), ECP-local (real, min-image),
/// and the complex per-candidate non-local mesh (FN/SP via lexicographic clamps). Returns
/// (e_L, b_L, kin_elements, ecp_cands) where each candidate is (is_up, i, new_pos, a_m) with a_m
/// the complex FN move amplitude (= c_min0(det_amp*jr)).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_lrdmc_ecp_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    rt: &[[f64; 3]; 3],
    pbc: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    fast: bool,
) -> (
    Complex64,
    Complex64,
    Complex64,
    Vec<Complex64>,
    Vec<MeshCandidateC>,
) {
    let (e_l, e_l_meas, b_l, elements, cands, _) = gfmc_lrdmc_ecp_complex_state(
        gem,
        jastrow,
        j3c,
        j_matrix,
        positions,
        eff_charges,
        r_up,
        r_dn,
        alat,
        rt,
        rt,
        pbc,
        nuc_l,
        e_l_p,
        c_l_p,
        p_l_p,
        nuc_n,
        am,
        e_n,
        c_n,
        p_n,
        weights,
        grid_points,
        nn,
        dltmove,
        fast,
        None,
        None,
    );
    (e_l, e_l_meas, b_l, elements, cands)
}

#[allow(clippy::too_many_arguments)]
fn gfmc_lrdmc_ecp_complex_state(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    pbc: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    fast: bool,
    state_in: Option<&MeshFastStateC>,
    pot: Option<&IntervalPotState>,
) -> IntervalEvalComplex {
    let n_up = r_up.nrows();
    let n_dn = r_dn.nrows();
    let n = n_up + n_dn;
    let shifts = lattice_shifts(alat, kinetic_rt);
    let inv_2a2 = Complex64::new(-1.0 / (2.0 * alat * alat), 0.0);
    let j1 = jastrow.one_body.as_ref();
    let j2 = jastrow.two_body.as_ref();
    let dtn = jastrow.dtn.as_ref();
    let with_j = j1.is_some() || j2.is_some() || j3c.is_some() || dtn.is_some();
    // Fast chains share one complex SM base state between both one-electron-move
    // meshes: the 6N kinetic hops and ECP T-move quadrature. Interval drivers pass
    // their carried per-interval state in `state_in`.
    let built_state = if fast && state_in.is_none() {
        mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, r_up, r_dn)
    } else {
        None
    };
    let fast_state = state_in.or(built_state.as_ref());
    // 1. complex discretized-kinetic amplitudes. Base det / base Jastrow value are built
    // ONLY on the reference branch (see the AE twin).
    let elements: Vec<Complex64> = match fast_state {
        Some(st) => discretized_kinetic_elements_complex_fast(
            gem, jastrow, j3c, j_matrix, r_up, r_dn, alat, kinetic_rt, st,
        ),
        None => {
            let det_base = det_phase_lnabs(&compute_geminal_complex(gem, r_up, r_dn));
            // Hop Jastrow ratio in the `jastrow_ratio` zone (the single-move factor of the
            // FAST branch), not the difference of two whole `jastrow_value_complex`
            // values, which rounds in `jastrow_eval` (float32 in mixed), issue #757.
            let jas_ratio =
                with_j.then(|| ComplexJasRatioState::new(jastrow, j3c, j_matrix, r_up, r_dn));
            let mut elements: Vec<Complex64> = Vec::with_capacity(6 * n);
            for e in 0..n_up {
                for shifts_s in &shifts {
                    let mut mu = r_up.to_owned();
                    for c in 0..3 {
                        mu[[e, c]] += shifts_s[c];
                    }
                    let det_m = det_phase_lnabs(&compute_geminal_complex(gem, &mu.view(), r_dn));
                    let jr = jas_ratio.as_ref().map_or(1.0, |st| {
                        let pos = [mu[[e, 0]], mu[[e, 1]], mu[[e, 2]]];
                        st.ratio(jastrow, j3c, r_up, r_dn, true, e, pos)
                    });
                    elements.push(inv_2a2 * det_ratio_complex(det_m, det_base) * jr);
                }
            }
            for e in 0..n_dn {
                for shifts_s in &shifts {
                    let mut md = r_dn.to_owned();
                    for c in 0..3 {
                        md[[e, c]] += shifts_s[c];
                    }
                    let det_m = det_phase_lnabs(&compute_geminal_complex(gem, r_up, &md.view()));
                    let jr = jas_ratio.as_ref().map_or(1.0, |st| {
                        let pos = [md[[e, 0]], md[[e, 1]], md[[e, 2]]];
                        st.ratio(jastrow, j3c, r_up, r_dn, false, e, pos)
                    });
                    elements.push(inv_2a2 * det_ratio_complex(det_m, det_base) * jr);
                }
            }
            elements
        }
    };
    // 2. FN/SP + nondiag + sign-flip (complex lexicographic).
    let mut fn_sum = Complex64::new(0.0, 0.0);
    let mut sp = Complex64::new(0.0, 0.0);
    for &el in &elements {
        fn_sum += c_min0(el);
        sp += c_max0(el);
    }
    let q = Complex64::new(1.0 / (2.0 * alat * alat), 0.0);
    let mut nondiag = vec![Complex64::new(0.0, 0.0); n];
    let mut sign_flip = vec![false; n];
    for e in 0..n {
        for s in 0..6 {
            let el = elements[e * 6 + s];
            nondiag[e] += el + q;
            if c_ge0(el) {
                sign_flip[e] = true;
            }
        }
    }
    // 3. complex continuum KE per electron (det + real J1/J2/DTN + Re'd complex J3).
    // Stream determinant and DTN grad/lap from the carried state.
    let (gu, gd, lu, ld) = crate::timed!(
        "gfmc.det_gradlap_c",
        match fast_state {
            Some(st) if complex_det_gradlap_ok(gem) => st.det_grad_lap_c(gem, r_up, r_dn),
            _ => {
                if fast_state.is_some() {
                    bump_route_counter(&GFMC_CPLX_DERIV_DET_SLOW);
                }
                grads_and_laplacian_ln_det_complex(gem, r_up, r_dn)
            }
        }
    );
    let (gju, gjd, lju, ljd) = if with_j {
        crate::timed!(
            "gfmc.j12_gradlap_c",
            match fast_state {
                Some(st) if complex_jaspart_gradlap_ok(jastrow) =>
                    st.jas_part_grad_lap(jastrow, r_up, r_dn),
                _ => {
                    if fast_state.is_some() {
                        if jastrow.dtn.is_some() {
                            bump_route_counter(&GFMC_CPLX_DTN_GRAD_SLOW);
                        }
                        if jastrow.one_body.is_some() || jastrow.two_body.is_some() {
                            bump_route_counter(&GFMC_CPLX_J12_GRAD_SLOW);
                        }
                    }
                    crate::jastrow_factor::compute_grads_and_laplacian_jastrow_part(
                        jastrow, r_up, r_dn,
                    )
                }
            }
        )
    } else {
        (
            Array2::zeros((n_up, 3)),
            Array2::zeros((n_dn, 3)),
            numpy::ndarray::Array1::zeros(n_up),
            numpy::ndarray::Array1::zeros(n_dn),
        )
    };
    let (g3u, g3d, l3u, l3d) = crate::timed!(
        "gfmc.j3_gradlap_c",
        match (j3c, j_matrix) {
            (Some(j3), Some(jm)) => match fast_state {
                // The streamed J3 derivative cache requires a carried J3 ratio
                // state (contiguous j_matrix) and a supported
                // complex J3 orb class; misses keep the counted full rebuild.
                Some(st)
                    if st.j3_state.is_some()
                        && complex_candidate_class(j3) != CandidateBasisClass::Unsupported =>
                    st.j3_grad_lap_c(j3, jm, r_up, r_dn),
                _ => {
                    if fast_state.is_some() {
                        bump_route_counter(&GFMC_CPLX_J3_GRAD_SLOW);
                    }
                    j3_grads_lap_complex(j3, jm, r_up, r_dn)
                }
            },
            _ => (
                Array2::zeros((n_up, 3)),
                Array2::zeros((n_dn, 3)),
                numpy::ndarray::Array1::zeros(n_up),
                numpy::ndarray::Array1::zeros(n_dn),
            ),
        }
    );
    let ke_cont = |i: usize, up: bool| -> Complex64 {
        let (g, l, gj, lj, g3, l3) = if up {
            (&gu, &lu, &gju, &lju, &g3u, &l3u)
        } else {
            (&gd, &ld, &gjd, &ljd, &g3d, &l3d)
        };
        let mut g2 = Complex64::new(0.0, 0.0);
        for c in 0..3 {
            let gc = g[[i, c]] + Complex64::new(gj[[i, c]] + g3[[i, c]], 0.0);
            g2 += gc * gc;
        }
        -0.5 * (l[i] + Complex64::new(lj[i] + l3[i], 0.0) + g2)
    };
    // 4. per-electron Ewald ei (ECP: ZV max-arg = BARE, no alat floor). Carried
    // interval rows when the incremental potential state is present.
    let ei_bare = match pot {
        Some(ps) => ps.ei_bare.clone(),
        None => crate::ewald::compute_ewald_potential_ei_per_electron(
            r_up,
            r_dn,
            pbc.eta,
            &pbc.real_images,
            &pbc.g_vectors,
            pbc.g_factors,
            &pbc.cell,
            pbc.n_max,
            eff_charges,
            positions,
            None,
        ),
    };
    let mut coul_ei = Complex64::new(0.0, 0.0);
    // `meas_ei_tc`: MEASUREMENT (novar=0) per-electron pieces (complex Tc).
    let mut meas_ei_tc = Complex64::new(0.0, 0.0);
    let mut sprime_mask = vec![0.0_f64; n];
    for (up, nr) in [(true, n_up), (false, n_dn)] {
        for i in 0..nr {
            let gi = if up { i } else { n_up + i };
            let bare = Complex64::new(ei_bare[gi], 0.0);
            let tc = ke_cont(i, up);
            let zv = bare + tc - nondiag[gi];
            let opt = if sign_flip[gi] { c_max(zv, bare) } else { zv };
            if sign_flip[gi] && bare.re > zv.re {
                sprime_mask[gi] = 1.0;
            }
            coul_ei += opt;
            meas_ei_tc += bare + tc;
        }
    }
    let ee = match pot {
        Some(ps) => ps
            .ee
            .as_ref()
            .expect("complex PBC potential state carries Ewald ee")
            .energy(pbc.g_factors),
        None => crate::ewald::compute_ewald_potential_ee(
            r_up,
            r_dn,
            pbc.eta,
            &pbc.real_images,
            &pbc.g_vectors,
            pbc.g_factors,
            &pbc.cell,
            pbc.n_max,
            None,
        ),
    };
    let diag_kin = 3.0 / (alat * alat) * (n as f64);
    // ECP local (real, min-image) + complex non-local mesh (tmove per-candidate).
    // With a state, the local channel streams through the per-electron
    // cache (only the hopped electron re-evaluates across the interval) and the
    // non-local mesh runs off the candidate tables + geometry cache; without a
    // state, both keep the byte-exact reference path.
    let cell_view = pbc.cell;
    let ecp_local = crate::timed!(
        "gfmc.ecp_local_c",
        match pot.and_then(|ps| ps.ecp_local.as_ref()) {
            // Incremental interval state: carried per-electron local-ECP rows
            // (electron-major fp reordering of the reference channel-major
            // total; twin-bounded).
            Some(rows) => rows.iter().sum::<f64>(),
            None => match fast_state {
                Some(st) => st.ecp_local(
                    positions,
                    nuc_l,
                    e_l_p,
                    c_l_p,
                    p_l_p,
                    r_up,
                    r_dn,
                    Some((&cell_view, pbc.n_max)),
                ),
                None => crate::coulomb_potential::compute_ecp_local_parts_all_pairs(
                    positions,
                    nuc_l,
                    e_l_p,
                    c_l_p,
                    p_l_p,
                    r_up,
                    r_dn,
                    Some((&cell_view, pbc.n_max)),
                ),
            },
        }
    );
    let mesh = crate::timed!(
        "gfmc.one_move_mesh_c",
        match fast_state {
            Some(st) => compute_ecp_nonlocal_mesh_complex_fast(
                gem,
                jastrow,
                j3c,
                j_matrix,
                r_up,
                r_dn,
                positions,
                nuc_n,
                am,
                e_n,
                c_n,
                p_n,
                weights,
                grid_points,
                ecp_rt,
                nn,
                Some((&cell_view, pbc.n_max)),
                st,
            ),
            None => crate::complex_ecp::compute_ecp_nonlocal_mesh_complex(
                gem,
                jastrow,
                j3c,
                j_matrix,
                r_up,
                r_dn,
                positions,
                nuc_n,
                am,
                e_n,
                c_n,
                p_n,
                weights,
                grid_points,
                ecp_rt,
                nn,
                Some(&cell_view),
                pbc.n_max,
                None,
            ),
        }
    );
    let mut fn_ecp = Complex64::new(0.0, 0.0);
    let mut sp_ecp = Complex64::new(0.0, 0.0);
    let mut cands: Vec<MeshCandidateC> = Vec::with_capacity(mesh.len());
    for (is_up, i, pos, det_amp, jr) in mesh {
        // Fixed-phase per-element real projection:
        // the effective Hamiltonian the walk samples has REAL matrix elements
        // Re[V * Psi'/Psi] PER CANDIDATE (the complex fixed-node), so the
        // sign split (c_min0/c_max0) and the b_L / S' sums act on the projected
        // real value -- mirroring XLA's project_real=True. The full complex
        // ratio would agree only by linearity at the Re-consuming points and would
        // leak the imaginary component into the stored e_L; the {Re V<0} selection
        // would carry Im into b_L on any complex fixture with Re(V)<0.
        let a_m = if dltmove {
            // DLA (dltmove, jqmc_gfmc.py, flag_determinant_only=True): the SP
            // diagonal folds max(Re(det_only), 0); the FN off-diagonal is
            // min(Re(det_only), 0) * Jastrow_ratio. The fixed-phase per-element real
            // projection is applied to the determinant-only ratio at the source
            // (before the sign split); the Jastrow ratio `jr` stays complex and its Re
            // is taken at the b_L / e_L consumption (matching XLA's `jnp.real` at
            // p_list / non_diagonal_sum, so Im(jr) enters as the dropped walker phase).
            let d = Complex64::new(det_amp.re, 0.0);
            sp_ecp += c_max0(d);
            c_min0(d) * jr
        } else {
            // tmove: full (det*Jastrow) fixed-phase ratio, uniform sign split.
            let v = Complex64::new((det_amp * jr).re, 0.0);
            sp_ecp += c_max0(v);
            c_min0(v)
        };
        fn_ecp += a_m;
        cands.push((is_up, i, pos, a_m));
    }
    let e_l = Complex64::new(diag_kin + ee + pbc.ii + ecp_local, 0.0)
        + coul_ei
        + (sp + sp_ecp)
        + (fn_sum + fn_ecp);
    // Measurement local energy (novar=0 split): continuum e_L (complex);
    // ECP non-local (fixed-phase localized) kept as-is.
    let e_l_meas = Complex64::new(ee + pbc.ii + ecp_local, 0.0) + meas_ei_tc + sp_ecp + fn_ecp;
    let b_l = fn_sum + fn_ecp;
    (e_l, e_l_meas, b_l, elements, cands, sprime_mask)
}

/// Single-walker complex Bloch-twist AE GFMC_t chain (jnone, fixed-phase). Reuses the verified
/// complex e_L: escape uses Re(b_L), weight uses exp(-tau*Re(e_L)), the FN move categorical uses
/// Re(min(elements,0)); the stored per-step e_L is COMPLEX (energy = Re of the mixed estimator).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ae_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    random_mesh: bool,
    pbc: &crate::ewald::EwaldRef,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_re: Vec<f64> = Vec::new();
    let mut w_hist: Vec<f64> = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut tau_left = tau;
        let mut w_l = 1.0_f64;
        let mut e_l_last = 0.0_f64;
        let mut interval_rt = None;
        while tau_left > 0.0 {
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, e_l_meas, b_l, elements) = gfmc_lrdmc_ae_complex(
                gem,
                jastrow,
                j3c,
                j_matrix,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                pbc,
                false,
            );
            // novar=0 split: measurement carry (complex fixed-phase).
            e_l_last = e_l_meas.re;
            let xi = rng.uniform();
            let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l.re); // Re(non_diag_sum)
            w_l *= (-tau_update * e_l.re).exp();
            tau_left -= tau_update;
            if tau_left > 0.0 {
                // FN move: real part of the complex FN amplitudes (fixed-phase categorical).
                let mut fsum = 0.0;
                for &el in &elements {
                    fsum += c_min0(el).re;
                }
                let u = rng.uniform();
                let mut acc = 0.0;
                let mut k = elements.len() - 1;
                for (idx, &el) in elements.iter().enumerate() {
                    acc += c_min0(el).re / fsum;
                    if u <= acc {
                        k = idx;
                        break;
                    }
                }
                let s = k % 6;
                if k < 6 * n_up {
                    let e = k / 6;
                    for c in 0..3 {
                        r_up[[e, c]] += shifts[s][c];
                    }
                } else {
                    let e = (k - 6 * n_up) / 6;
                    for c in 0..3 {
                        r_dn[[e, c]] += shifts[s][c];
                    }
                }
            }
        }
        if step >= n_warmup {
            e_re.push(e_l_last);
            w_hist.push(w_l);
        }
    }
    let energy = mixed_window_energy(&e_re, &w_hist, num_collect);
    (energy, e_re, w_hist)
}

/// Single-walker complex Bloch-twist ECP GFMC_t chain (Casula tmove, fixed-phase). Reuses the
/// verified complex ECP e_L: escape uses Re(b_L), weight exp(-tau*Re(e_L)); the FN move categorical
/// runs over Re(c_min0(elements)) ++ Re(c_min0(ecp candidates)). Stored e_L per step is Re(e_L).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ecp_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    random_mesh: bool,
    pbc: &crate::ewald::EwaldRef,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_re: Vec<f64> = Vec::new();
    let mut w_hist: Vec<f64> = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut tau_left = tau;
        let mut w_l = 1.0_f64;
        let mut e_l_last = 0.0_f64;
        let mut interval_rt = None;
        while tau_left > 0.0 {
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, e_l_meas, b_l, elements, cands) = gfmc_lrdmc_ecp_complex(
                gem,
                jastrow,
                j3c,
                j_matrix,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                pbc,
                nuc_l,
                e_l_p,
                c_l_p,
                p_l_p,
                nuc_n,
                am,
                e_n,
                c_n,
                p_n,
                weights,
                grid_points,
                nn,
                dltmove,
                false,
            );
            // novar=0 split: measurement carry (complex fixed-phase).
            e_l_last = e_l_meas.re;
            let xi = rng.uniform();
            let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l.re);
            w_l *= (-tau_update * e_l.re).exp();
            tau_left -= tau_update;
            if tau_left > 0.0 {
                // FN move over Re of the complex FN amplitudes (kinetic ++ ecp).
                let nk = elements.len();
                let mut fsum = 0.0;
                for &el in &elements {
                    fsum += c_min0(el).re;
                }
                for &(_, _, _, a_m) in &cands {
                    fsum += a_m.re;
                }
                let u = rng.uniform();
                let mut acc = 0.0;
                let mut chosen = nk + cands.len() - 1;
                'pick: {
                    for (idx, &el) in elements.iter().enumerate() {
                        acc += c_min0(el).re / fsum;
                        if u <= acc {
                            chosen = idx;
                            break 'pick;
                        }
                    }
                    for (m, &(_, _, _, a_m)) in cands.iter().enumerate() {
                        acc += a_m.re / fsum;
                        if u <= acc {
                            chosen = nk + m;
                            break 'pick;
                        }
                    }
                }
                if chosen < nk {
                    let s = chosen % 6;
                    if chosen < 6 * n_up {
                        let e = chosen / 6;
                        for c in 0..3 {
                            r_up[[e, c]] += shifts[s][c];
                        }
                    } else {
                        let e = (chosen - 6 * n_up) / 6;
                        for c in 0..3 {
                            r_dn[[e, c]] += shifts[s][c];
                        }
                    }
                } else {
                    let (is_up, e, new_pos, _) = cands[chosen - nk];
                    if is_up {
                        for c in 0..3 {
                            r_up[[e, c]] = new_pos[c];
                        }
                    } else {
                        for c in 0..3 {
                            r_dn[[e, c]] = new_pos[c];
                        }
                    }
                }
            }
        }
        if step >= n_warmup {
            e_re.push(e_l_last);
            w_hist.push(w_l);
        }
    }
    let energy = mixed_window_energy(&e_re, &w_hist, num_collect);
    (energy, e_re, w_hist)
}

/// Single-walker all-electron LRDMC/GFMC chain (the production nw=1 config). Per branching
/// step: a continuous-time projection of length `tau` (escape-time hops on the +/-alat lattice,
/// weight w_L *= exp(-tau_up*e_L)), then record (e_L_final, w_L). Energy = Sum e_L*G_L / Sum G_L
/// with the G_L window product over `num_collect` branching weights. 1:1 jqmc_gfmc._projection
/// (identity mesh rotation). Returns (energy, e_L per step, w_L per step).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ae(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    random_mesh: bool,
    pbc: Option<&crate::ewald::EwaldRef>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_hist = Vec::new();
    let mut w_hist = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut tau_left = tau;
        let mut w_l = 1.0_f64;
        let mut e_l_last = 0.0_f64;
        let mut interval_rt = None;
        while tau_left > 0.0 {
            // Draw a rotation at every step (the RNG schedule), but use the interval's
            // first rotation.
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, e_l_meas, b_l, elements) = gfmc_lrdmc_ae(
                gem,
                jastrow,
                j3,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                pbc,
                false,
            );
            // novar=0 split: measurement carry.
            e_l_last = e_l_meas;
            let xi = rng.uniform();
            // b_l < 0 => log(1-xi)/b_l > 0
            let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l);
            w_l *= (-tau_update * e_l).exp();
            tau_left -= tau_update;
            if tau_left > 0.0 {
                // categorical over FN amplitudes; apply the chosen single-electron hop
                let mut cdf: Vec<f64> = Vec::with_capacity(elements.len());
                let mut acc = 0.0;
                let mut fsum = 0.0;
                for &el in &elements {
                    fsum += nan_min(el, 0.0);
                }
                for &el in &elements {
                    acc += nan_min(el, 0.0) / fsum;
                    cdf.push(acc);
                }
                let u = rng.uniform();
                let k = cdf.partition_point(|&c| c < u).min(elements.len() - 1);
                let s = k % 6;
                if k < 6 * n_up {
                    let e = k / 6;
                    for c in 0..3 {
                        r_up[[e, c]] += shifts[s][c];
                    }
                } else {
                    let e = (k - 6 * n_up) / 6;
                    for c in 0..3 {
                        r_dn[[e, c]] += shifts[s][c];
                    }
                }
            }
        }
        if step >= n_warmup {
            e_hist.push(e_l_last);
            w_hist.push(w_l);
        }
    }
    // Mixed estimator matching jqmc GFMC_t: e_L=stored_e_L[k:], G_L=compute_g_l(stored_w_L,k),
    // energy = Sum e_L*G_L / Sum G_L. G_L[j]=Prod_{i<k} w[j+i] pairs with e[k+j].
    let energy = mixed_window_energy(&e_hist, &w_hist, num_collect);
    (energy, e_hist, w_hist)
}

/// Multi-walker (population) all-electron GFMC_t with stochastic reconfiguration
/// (branching). Each branching step: every walker is projected independently for
/// imaginary time `tau` (own escape-time chain), then the population is resampled by
/// systematic comb sampling on the branching weights (jqmc_gfmc.py, one shared
/// `zeta` per step) and positions are copied from the chosen walkers (weights reset).
/// Per step the WALKER-AVERAGED estimator is recorded: e_L = Sum w*e_L / Sum w, w = Sum w / nw.
/// Production is nw=1+MPI ([[cpu-benchmark-methodology]]); this is the single-process
/// population variant for completeness. Returns (energy, e_L per step, w per step).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ae_pop(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    num_walkers: usize,
    random_mesh: bool,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    // Independent per-walker starting configs: jitter the seed config with a per-walker
    // Gaussian so the population is not degenerate on step 0.
    let mut walkers: Vec<(Array2<f64>, Array2<f64>)> = (0..num_walkers)
        .map(|w| {
            let mut ru = r_up0.to_owned();
            let mut rd = r_dn0.to_owned();
            if w > 0 {
                for r in [&mut ru, &mut rd] {
                    for v in r.iter_mut() {
                        *v += 0.3 * rng.normal();
                    }
                }
            }
            (ru, rd)
        })
        .collect();
    let mut e_hist = Vec::new();
    let mut w_hist = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut e_l_w = vec![0.0_f64; num_walkers];
        let mut w_l_w = vec![0.0_f64; num_walkers];
        for (wi, (r_up, r_dn)) in walkers.iter_mut().enumerate() {
            let mut tau_left = tau;
            let mut w_l = 1.0_f64;
            let mut e_l_last = 0.0_f64;
            let mut interval_rt = None;
            while tau_left > 0.0 {
                let sampled_rt = sample_rt(&mut rng, random_mesh);
                let rt = *interval_rt.get_or_insert(sampled_rt);
                let shifts = lattice_shifts(alat, &rt);
                let (e_l, e_l_meas, b_l, elements) = gfmc_lrdmc_ae(
                    gem,
                    jastrow,
                    j3,
                    positions,
                    eff_charges,
                    &r_up.view(),
                    &r_dn.view(),
                    alat,
                    &rt,
                    None,
                    false,
                );
                // novar=0 split: measurement carry.
                e_l_last = e_l_meas;
                let xi = rng.uniform();
                let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l);
                w_l *= (-tau_update * e_l).exp();
                tau_left -= tau_update;
                if tau_left > 0.0 {
                    let mut fsum = 0.0;
                    for &el in &elements {
                        fsum += nan_min(el, 0.0);
                    }
                    let u = rng.uniform();
                    let mut acc = 0.0;
                    let mut k = elements.len() - 1;
                    for (idx, &el) in elements.iter().enumerate() {
                        acc += nan_min(el, 0.0) / fsum;
                        if u <= acc {
                            k = idx;
                            break;
                        }
                    }
                    let s = k % 6;
                    if k < 6 * n_up {
                        let e = k / 6;
                        for c in 0..3 {
                            r_up[[e, c]] += shifts[s][c];
                        }
                    } else {
                        let e = (k - 6 * n_up) / 6;
                        for c in 0..3 {
                            r_dn[[e, c]] += shifts[s][c];
                        }
                    }
                }
            }
            e_l_w[wi] = e_l_last;
            w_l_w[wi] = w_l;
        }
        // Walker-averaged estimator for this branching step.
        let wsum: f64 = w_l_w.iter().sum();
        if step >= n_warmup && wsum > 0.0 {
            let ew: f64 = e_l_w.iter().zip(&w_l_w).map(|(e, w)| e * w).sum::<f64>() / wsum;
            e_hist.push(ew);
            w_hist.push(wsum / num_walkers as f64);
        }
        // Stochastic reconfiguration: resample walker positions by comb sampling.
        let zeta = rng.uniform();
        let chosen = reconfigure_indices(&w_l_w, zeta);
        let picked: Vec<(Array2<f64>, Array2<f64>)> = chosen
            .iter()
            .map(|&src| (walkers[src].0.clone(), walkers[src].1.clone()))
            .collect();
        walkers = picked;
    }
    let energy = mixed_window_energy(&e_hist, &w_hist, num_collect);
    (energy, e_hist, w_hist)
}

/// Single-walker all-electron **GFMC_n** (`lrdmc-bra`): the fixed-step branching projector,
/// a DIFFERENT projector from GFMC_t. Per branching interval of `num_per_meas` discrete
/// lattice hops, the weight multiplies by `b_x = |b_L| / (H_LL - E_scf)` each step (H_LL =
/// diagonal_sum_hamiltonian = e_L - b_L; |b_L| = -Sum FN off-diagonal), NOT GFMC_t's
/// `exp(-tau*e_L)`. Move sampling / rotation mesh identical to GFMC_t. E_scf self-tunes to the
/// running mixed energy (jqmc's on-the-fly jackknife; jqmc_gfmc.py).
/// Returns (energy, e_L per interval, w per interval).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ae_n(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    num_per_meas: usize,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    e_scf0: f64,
    random_mesh: bool,
) -> (f64, Vec<f64>, Vec<f64>) {
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_scf = e_scf0;
    // Mixed estimator E = Sum e*G_L / Sum G_L over a history; used both for the final
    // energy (measured portion) and the on-the-fly E_scf (full history so far).
    let mixed = |e: &[f64], w: &[f64]| -> f64 { mixed_window_energy(e, w, num_collect) };
    let mut e_all = Vec::new();
    let mut w_all = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut w_l = 1.0_f64;
        let mut interval_rt = None;
        for _ in 0..num_per_meas {
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, _e_l_meas, b_l, elements) = gfmc_lrdmc_ae(
                gem,
                jastrow,
                j3,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                None,
                false,
            );
            let h_ll = e_l - b_l; // diagonal_sum_hamiltonian
            let b_x = (-b_l) / (h_ll - e_scf); // |b_L| / (H_LL - E_scf)
            w_l *= b_x;
            // FN categorical hop (same as GFMC_t).
            let mut fsum = 0.0;
            for &el in &elements {
                fsum += nan_min(el, 0.0);
            }
            let u = rng.uniform();
            let mut acc = 0.0;
            let mut k = elements.len() - 1;
            for (idx, &el) in elements.iter().enumerate() {
                acc += nan_min(el, 0.0) / fsum;
                if u <= acc {
                    k = idx;
                    break;
                }
            }
            let s = k % 6;
            if k < 6 * n_up {
                let e = k / 6;
                for c in 0..3 {
                    r_up[[e, c]] += shifts[s][c];
                }
            } else {
                let e = (k - 6 * n_up) / 6;
                for c in 0..3 {
                    r_dn[[e, c]] += shifts[s][c];
                }
            }
        }
        // Fresh e_L at the final interval positions with the interval rotation.
        let measurement_rt = interval_rt.unwrap_or(id);
        let (_, e_final, _, _) = gfmc_lrdmc_ae(
            gem,
            jastrow,
            j3,
            positions,
            eff_charges,
            &r_up.view(),
            &r_dn.view(),
            alat,
            &measurement_rt,
            None,
            false,
        );
        e_all.push(e_final);
        w_all.push(w_l);
        // On-the-fly E_scf -> running mixed energy (every 10 intervals to bound cost).
        if e_all.len() > num_collect && step % 10 == 0 {
            let e = mixed(&e_all, &w_all);
            if e.is_finite() {
                e_scf = e;
            }
        }
    }
    let energy = mixed(&e_all[n_warmup..], &w_all[n_warmup..]);
    (energy, e_all.split_off(n_warmup), w_all.split_off(n_warmup))
}

/// Single-walker ECP LRDMC/GFMC chain (Casula T-move). Mirrors [`gfmc_run_ae`] but the
/// escape-time projection uses the combined move mesh: 6*N lattice hops (kinetic) plus the
/// non-local tmove candidates. The FN move probabilities and b_L span both meshes.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ecp(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    tau: f64,
    alat: f64,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    random_mesh: bool,
    pbc: Option<&crate::ewald::EwaldRef>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_hist = Vec::new();
    let mut w_hist = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut tau_left = tau;
        let mut w_l = 1.0_f64;
        let mut e_l_last = 0.0_f64;
        let mut interval_rt = None;
        while tau_left > 0.0 {
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, e_l_meas, b_l, elements, cands) = gfmc_lrdmc_ecp(
                gem,
                jastrow,
                j3,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                nuc_l,
                e_l_p,
                c_l_p,
                p_l_p,
                nuc_n,
                am,
                e_n,
                c_n,
                p_n,
                weights,
                grid_points,
                nn,
                dltmove,
                pbc,
                false,
            );
            // novar=0 split: measurement carry.
            e_l_last = e_l_meas;
            let xi = rng.uniform();
            let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l);
            w_l *= (-tau_update * e_l).exp();
            tau_left -= tau_update;
            if tau_left > 0.0 {
                // FN amplitudes over the concatenated (kinetic ++ ecp) mesh.
                let nk = elements.len();
                let mut fsum = 0.0;
                for &el in &elements {
                    fsum += nan_min(el, 0.0);
                }
                for &(_, _, _, v) in &cands {
                    fsum += nan_min(v, 0.0);
                }
                let u = rng.uniform();
                let mut acc = 0.0;
                let mut chosen = nk + cands.len() - 1;
                'pick: {
                    for (idx, &el) in elements.iter().enumerate() {
                        acc += nan_min(el, 0.0) / fsum;
                        if u <= acc {
                            chosen = idx;
                            break 'pick;
                        }
                    }
                    for (m, &(_, _, _, v)) in cands.iter().enumerate() {
                        acc += nan_min(v, 0.0) / fsum;
                        if u <= acc {
                            chosen = nk + m;
                            break 'pick;
                        }
                    }
                }
                if chosen < nk {
                    let s = chosen % 6;
                    if chosen < 6 * n_up {
                        let e = chosen / 6;
                        for c in 0..3 {
                            r_up[[e, c]] += shifts[s][c];
                        }
                    } else {
                        let e = (chosen - 6 * n_up) / 6;
                        for c in 0..3 {
                            r_dn[[e, c]] += shifts[s][c];
                        }
                    }
                } else {
                    let (is_up, e, new_pos, _) = cands[chosen - nk];
                    if is_up {
                        for c in 0..3 {
                            r_up[[e, c]] = new_pos[c];
                        }
                    } else {
                        for c in 0..3 {
                            r_dn[[e, c]] = new_pos[c];
                        }
                    }
                }
            }
        }
        if step >= n_warmup {
            e_hist.push(e_l_last);
            w_hist.push(w_l);
        }
    }
    let energy = mixed_window_energy(&e_hist, &w_hist, num_collect);
    (energy, e_hist, w_hist)
}

/// Single-walker ECP **GFMC_n** (`lrdmc-bra`) chain: the fixed-step projector `w_L *= b_x =
/// |b_L|/(H_LL - E_scf)` (H_LL = e_L - b_L) over `num_per_meas` discrete hops, with the combined
/// kinetic+ECP move mesh (Casula tmove/dltmove). E_scf self-tunes to the running mixed energy.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_run_ecp_n(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    num_per_meas: usize,
    num_collect: usize,
    n_warmup: usize,
    n_measure: usize,
    seed: u64,
    e_scf0: f64,
    nuc_l: &[i64],
    e_l_p: &[f64],
    c_l_p: &[f64],
    p_l_p: &[f64],
    nuc_n: &[i64],
    am: &[i64],
    e_n: &[f64],
    c_n: &[f64],
    p_n: &[f64],
    weights: &[f64],
    grid_points: &ArrayView2<f64>,
    nn: usize,
    dltmove: bool,
    random_mesh: bool,
    pbc: Option<&crate::ewald::EwaldRef>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let n_up = r_up0.nrows();
    let mut rng = GfmcRng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut e_scf = e_scf0;
    let mixed = |e: &[f64], w: &[f64]| -> f64 { mixed_window_energy(e, w, num_collect) };
    let ecp_el = |ru: &ArrayView2<f64>, rd: &ArrayView2<f64>, rt: &[[f64; 3]; 3]| {
        gfmc_lrdmc_ecp(
            gem,
            jastrow,
            j3,
            positions,
            eff_charges,
            ru,
            rd,
            alat,
            rt,
            nuc_l,
            e_l_p,
            c_l_p,
            p_l_p,
            nuc_n,
            am,
            e_n,
            c_n,
            p_n,
            weights,
            grid_points,
            nn,
            dltmove,
            pbc,
            false,
        )
    };
    let mut e_all = Vec::new();
    let mut w_all = Vec::new();
    for step in 0..(n_warmup + n_measure) {
        let mut w_l = 1.0_f64;
        let mut interval_rt = None;
        for _ in 0..num_per_meas {
            let sampled_rt = sample_rt(&mut rng, random_mesh);
            let rt = *interval_rt.get_or_insert(sampled_rt);
            let shifts = lattice_shifts(alat, &rt);
            let (e_l, _, b_l, elements, cands) = ecp_el(&r_up.view(), &r_dn.view(), &rt);
            let h_ll = e_l - b_l;
            w_l *= (-b_l) / (h_ll - e_scf);
            // Combined kinetic ++ ECP FN move (same as gfmc_run_ecp).
            let nk = elements.len();
            let mut fsum = 0.0;
            for &el in &elements {
                fsum += nan_min(el, 0.0);
            }
            for &(_, _, _, v) in &cands {
                fsum += nan_min(v, 0.0);
            }
            let u = rng.uniform();
            let mut acc = 0.0;
            let mut chosen = nk + cands.len() - 1;
            'pick: {
                for (idx, &el) in elements.iter().enumerate() {
                    acc += nan_min(el, 0.0) / fsum;
                    if u <= acc {
                        chosen = idx;
                        break 'pick;
                    }
                }
                for (m, &(_, _, _, v)) in cands.iter().enumerate() {
                    acc += nan_min(v, 0.0) / fsum;
                    if u <= acc {
                        chosen = nk + m;
                        break 'pick;
                    }
                }
            }
            if chosen < nk {
                let s = chosen % 6;
                if chosen < 6 * n_up {
                    let e = chosen / 6;
                    for c in 0..3 {
                        r_up[[e, c]] += shifts[s][c];
                    }
                } else {
                    let e = (chosen - 6 * n_up) / 6;
                    for c in 0..3 {
                        r_dn[[e, c]] += shifts[s][c];
                    }
                }
            } else {
                let (is_up, e, new_pos, _) = cands[chosen - nk];
                if is_up {
                    for c in 0..3 {
                        r_up[[e, c]] = new_pos[c];
                    }
                } else {
                    for c in 0..3 {
                        r_dn[[e, c]] = new_pos[c];
                    }
                }
            }
        }
        let measurement_rt = interval_rt.unwrap_or(id);
        let (_, e_final, _, _, _) = ecp_el(&r_up.view(), &r_dn.view(), &measurement_rt);
        e_all.push(e_final);
        w_all.push(w_l);
        if e_all.len() > num_collect && step % 10 == 0 {
            let e = mixed(&e_all, &w_all);
            if e.is_finite() {
                e_scf = e;
            }
        }
    }
    let energy = mixed(&e_all[n_warmup..], &w_all[n_warmup..]);
    (energy, e_all.split_off(n_warmup), w_all.split_off(n_warmup))
}

/// SplitMix64 uniform RNG for the GFMC escape-time / move draws (matches native_driver::Rng).
struct GfmcRng(u64);
impl GfmcRng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard-normal draw (Box-Muller; used only to jitter the initial walker population).
    fn normal(&mut self) -> f64 {
        let u1 = self
            .uniform()
            .max(crate::setting_consts::EPS_machine_precision);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ============================================================================
// Production projection-interval drivers for the `jqmc_gfmc.py` whole-run
// native chunk). One call = ONE branching interval for ONE walker -- the exact
// GFMC analogue of `run_energy_chunk` for the MCMC step loop. Python keeps the
// MPI branching / reconfiguration (collective across ranks), the bin statistics
// and I/O; everything inside the interval (mesh rotation, LRDMC e_L via the SAME
// verified `gfmc_lrdmc_{ae,ecp}` kernels above, the branching weight, and the FN
// categorical hop) runs here.
//
// RNG contract (threefry, mirroring the jqmc drivers' key chains exactly):
//   * GFMC_n (`gfmc_interval_n`, jqmc `_projection_n`): the jqmc key scan still
//     does `key,rot = split(key); key,move = split(key)` per sub-step, while the
//     FIRST rotation is held for the whole projection interval. We replay that
//     chain bit-exactly (`threefry::split` is XLA-bit-exact), so
//     the hop uniform `uniform(move_key)` is bit-exact in BOTH chain modes and
//     the returned `final_key` is byte-equal to the JAX projection's. The ONLY
//     non-reproducible stream is the first mesh-rotation trig (XLA sin/cos + fused
//     uniform scale): BITEXACT mode therefore INJECTS the JAX-drawn interval `R^T`
//     (`injected_rts[0]`), FAST mode builds it natively from the same bit-exact
//     threefry angles + native trig. That is the single mode-divergence point.
//   * GFMC_t (`gfmc_interval_t`, jqmc `_projection_t_core`): FAST mode ONLY.
//     A bitexact contract is unattainable BY CONSTRUCTION: the sub-step COUNT
//     itself depends on the fp value of `tau_update = log(1-xi)/b_L` (XLA `log`
//     vs libm `ln`, plus the kernel-parity fp floor of `b_L`), so one ulp can
//     change the number of projection sub-steps and the whole chain identity.
//     The vmapped JAX while_loop also advances ALL walkers' key chains to the
//     rank-local batch maximum iteration count. The native batch wrappers replay
//     only these masked-step key splits after the per-walker intervals finish;
//     scalar intervals retain their own stopping count. The chain keeps
//     jqmc's split ORDER (rot / xi / move) so every uniform is still drawn from
//     the bit-exact threefry stream; interval parity is checked at the public boundary.
// ============================================================================

/// Borrowed ECP channel bundle for the interval drivers (the ctx owns the parse).
pub struct GfmcEcpParams<'a> {
    pub nuc_l: &'a [i64],
    pub e_l: &'a [f64],
    pub c_l: &'a [f64],
    pub p_l: &'a [f64],
    pub nuc_n: &'a [i64],
    pub am: &'a [i64],
    pub e_n: &'a [f64],
    pub c_n: &'a [f64],
    pub p_n: &'a [f64],
    pub weights: &'a [f64],
    pub grid_points: ArrayView2<'a, f64>,
    pub nn: usize,
    pub dltmove: bool,
}

/// Carried potential state for one projection interval. A projection hop moves exactly one
/// electron, so the diagonal potential pieces admit one-electron updates:
///
///   * `ee`   -- electron-electron Ewald (running real-space half-sum + S(G)),
///     advanced at O(N*n_img + n_g) per hop;
///   * `ei_bare` / `ei_disc` -- per-electron electron-ion Ewald rows (each row
///     depends ONLY on that electron; the moved row is recomputed
///     BIT-IDENTICALLY to the full kernel via
///     `ewald_ei_single_electron` with the interval-constant ion
///     structure factor);
///   * `ecp_local` -- per-electron local-ECP rows (ECP runs only), moved row
///     recomputed via `compute_ecp_local_one`.
///
/// The same state also carries molecular bare Coulomb: its total ee energy is
/// advanced from only the moved electron's old/new interaction rows, while its
/// ei row is recomputed exactly. The kinetic / ECP-nonlocal channels -- and therefore `b_L`,
/// the hop cdf, the escape-time substep count, and the walk itself -- are NOT
/// touched by this state: only the diagonal `e_L` (and hence `w_L`) can drift
/// at the fp-reordering level. The ONLY leaf that accumulates per-hop deltas is
/// the ee running state; the drift bound is `~eps_mach * sqrt(n_hops)` relative
/// (< 1e-11 even at 1e5 hops), pinned per interval by the incremental-vs-full
/// twin test at 1e-9 absolute -- no
/// periodic refresh is needed (and a refresh under XLA's vmapped `lax.cond`
/// would degenerate to per-hop full rebuilds, so BOTH backends omit it to
/// keep the algorithm identical).
pub(crate) struct IntervalPotState {
    ee: Option<crate::ewald::EwaldEeState>,
    /// Bare molecular electron-electron energy; mutually exclusive with `ee`.
    ee_molecular: Option<f64>,
    /// Constant ion-ion energy (Ewald under PBC, bare Coulomb for molecules).
    ion_ion: f64,
    ei_bare: Vec<f64>,
    /// Discretized (alat-floored) ei rows -- carried for the AE composition only.
    ei_disc: Option<Vec<f64>>,
    /// Interval-constant ion reciprocal structure factor halves.
    si_re: Vec<f64>,
    si_im: Vec<f64>,
    /// Per-electron local-ECP rows (ECP runs only).
    ecp_local: Option<Vec<f64>>,
    /// Interval-constant el-ion row tables (image columns, cell inverse, separable
    /// phase) for the moved row's recompute (PBC only).
    ei_row: Option<crate::ewald::EiRowKernel>,
    /// Liveness counter: number of applied one-electron updates (anti-vacuity
    /// witness surfaced as `n_pot_updates` on the interval output).
    n_updates: u64,
}

impl IntervalPotState {
    /// Full build from the current configuration (interval start). `ecp` selects
    /// the composition: `Some` carries per-electron local-ECP rows and skips the
    /// discretized ei rows (the ECP ZV max-arg is bare); `None` (AE) carries both
    /// ei variants and no local-ECP rows.
    fn build(
        positions: &ArrayView2<f64>,
        eff_charges: &[f64],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        alat: f64,
        ecp: Option<&GfmcEcpParams<'_>>,
        p: &crate::ewald::EwaldRef<'_>,
    ) -> Self {
        let ee = crate::timed!(
            "gfmc.pot_build",
            crate::ewald::EwaldEeState::build(
                r_up,
                r_dn,
                p.eta,
                &p.real_images,
                &p.g_vectors,
                &p.cell,
                p.n_max,
            )
        );
        let (si_re, si_im) =
            crate::ewald::compute_ion_structure_factor(&p.g_vectors, eff_charges, positions);
        let ei_bare = crate::ewald::compute_ewald_potential_ei_per_electron(
            r_up,
            r_dn,
            p.eta,
            &p.real_images,
            &p.g_vectors,
            p.g_factors,
            &p.cell,
            p.n_max,
            eff_charges,
            positions,
            None,
        );
        let ei_disc = match ecp {
            Some(_) => None,
            None => Some(crate::ewald::compute_ewald_potential_ei_per_electron(
                r_up,
                r_dn,
                p.eta,
                &p.real_images,
                &p.g_vectors,
                p.g_factors,
                &p.cell,
                p.n_max,
                eff_charges,
                positions,
                Some(alat),
            )),
        };
        let ecp_local = ecp.map(|e| {
            let n_up = r_up.nrows();
            let mut v = vec![0.0_f64; n_up + r_dn.nrows()];
            for (row, out) in v.iter_mut().enumerate() {
                let re = if row < n_up {
                    [r_up[[row, 0]], r_up[[row, 1]], r_up[[row, 2]]]
                } else {
                    let i = row - n_up;
                    [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
                };
                *out = crate::coulomb_potential::compute_ecp_local_one(
                    positions,
                    e.nuc_l,
                    e.e_l,
                    e.c_l,
                    e.p_l,
                    re,
                    Some((&p.cell, p.n_max)),
                );
            }
            v
        });
        Self {
            ee: Some(ee),
            ee_molecular: None,
            ion_ion: p.ii,
            ei_bare,
            ei_disc,
            si_re,
            si_im,
            ecp_local,
            ei_row: Some(crate::ewald::EiRowKernel::new(
                &p.real_images,
                &p.g_vectors,
                &p.cell,
            )),
            n_updates: 0,
        }
    }

    /// Molecular sibling of [`Self::build`]. It preserves the reference
    /// electron-major ei row arithmetic and carries only the scalar ee total;
    /// local ECP already has an exact moved-row cache in `MeshFastState`.
    fn build_molecular(
        positions: &ArrayView2<f64>,
        eff_charges: &[f64],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        alat: f64,
        ecp: Option<&GfmcEcpParams<'_>>,
    ) -> Self {
        let n_up = r_up.nrows();
        let n = n_up + r_dn.nrows();
        let mut ei_bare = vec![0.0_f64; n];
        let mut ei_disc = ecp.is_none().then(|| vec![0.0_f64; n]);
        for gi in 0..n {
            let re = if gi < n_up {
                [r_up[[gi, 0]], r_up[[gi, 1]], r_up[[gi, 2]]]
            } else {
                let i = gi - n_up;
                [r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]
            };
            let mut bare = 0.0_f64;
            let mut disc = 0.0_f64;
            for ia in 0..positions.nrows() {
                let dx = re[0] - positions[[ia, 0]];
                let dy = re[1] - positions[[ia, 1]];
                let dz = re[2] - positions[[ia, 2]];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                bare += -eff_charges[ia] / d;
                if ei_disc.is_some() {
                    disc += -eff_charges[ia] / nan_max(d, alat);
                }
            }
            ei_bare[gi] = bare;
            if let Some(rows) = ei_disc.as_mut() {
                rows[gi] = disc;
            }
        }
        Self {
            ee: None,
            ee_molecular: Some(
                crate::coulomb_potential::compute_bare_coulomb_potential_el_el(r_up, r_dn),
            ),
            ion_ion: crate::coulomb_potential::compute_bare_coulomb_potential_ion_ion(
                positions,
                eff_charges,
            ),
            ei_bare,
            ei_disc,
            si_re: Vec::new(),
            si_im: Vec::new(),
            ecp_local: None,
            ei_row: None,
            n_updates: 0,
        }
    }

    /// Apply the one-electron move (`is_up`, `e`: `oldpos` -> `newpos`) after the
    /// hop was applied to `r_up`/`r_dn` (POST-hop views). The ei / local-ECP rows
    /// are recomputed exactly (bit-identical to a full evaluation); only the ee
    /// running state accumulates fp-reordering drift, bounded by the twin test.
    #[allow(clippy::too_many_arguments)]
    fn advance_after_hop(
        &mut self,
        positions: &ArrayView2<f64>,
        eff_charges: &[f64],
        r_up: &ArrayView2<f64>,
        r_dn: &ArrayView2<f64>,
        alat: f64,
        ecp: Option<&GfmcEcpParams<'_>>,
        p: Option<&crate::ewald::EwaldRef<'_>>,
        is_up: bool,
        e: usize,
        oldpos: [f64; 3],
        newpos: [f64; 3],
    ) {
        let n_up = r_up.nrows();
        let n_dn = r_dn.nrows();
        self.n_updates += 1;
        let mut others: Vec<[f64; 3]> = Vec::with_capacity(n_up + n_dn - 1);
        for i in 0..n_up {
            if !(is_up && i == e) {
                others.push([r_up[[i, 0]], r_up[[i, 1]], r_up[[i, 2]]]);
            }
        }
        for i in 0..n_dn {
            if !(!is_up && i == e) {
                others.push([r_dn[[i, 0]], r_dn[[i, 1]], r_dn[[i, 2]]]);
            }
        }
        let gi = if is_up { e } else { n_up + e };
        if let Some(p) = p {
            crate::timed!("gfmc.pot_ee_adv", {
                self.ee
                    .as_mut()
                    .expect("PBC potential state carries Ewald ee")
                    .advance(
                        gi,
                        &others,
                        oldpos,
                        newpos,
                        p.eta,
                        &p.real_images,
                        &p.g_vectors,
                        &p.cell,
                        p.n_max,
                    );
            });
            let eff = eff_charges;
            crate::timed!("gfmc.pot_ei_adv", {
                let row = self
                    .ei_row
                    .as_mut()
                    .expect("PBC potential state carries the ei row kernel");
                self.ei_bare[gi] = crate::ewald::ewald_ei_single_electron_with(
                    row,
                    newpos,
                    p.eta,
                    p.g_factors,
                    &p.cell,
                    p.n_max,
                    eff,
                    positions,
                    &self.si_re,
                    &self.si_im,
                    None,
                );
                if let Some(d) = self.ei_disc.as_mut() {
                    d[gi] = crate::ewald::ewald_ei_single_electron_with(
                        row,
                        newpos,
                        p.eta,
                        p.g_factors,
                        &p.cell,
                        p.n_max,
                        eff,
                        positions,
                        &self.si_re,
                        &self.si_im,
                        Some(alat),
                    );
                }
            });
            crate::timed!("gfmc.pot_ecploc_adv", {
                if let (Some(ecp), Some(v)) = (ecp, self.ecp_local.as_mut()) {
                    v[gi] = crate::coulomb_potential::compute_ecp_local_one(
                        positions,
                        ecp.nuc_l,
                        ecp.e_l,
                        ecp.c_l,
                        ecp.p_l,
                        newpos,
                        Some((&p.cell, p.n_max)),
                    );
                }
            });
        } else {
            let mut old_row = 0.0_f64;
            let mut new_row = 0.0_f64;
            for r in &others {
                let odx = oldpos[0] - r[0];
                let ody = oldpos[1] - r[1];
                let odz = oldpos[2] - r[2];
                old_row += 1.0 / (odx * odx + ody * ody + odz * odz).sqrt();
                let ndx = newpos[0] - r[0];
                let ndy = newpos[1] - r[1];
                let ndz = newpos[2] - r[2];
                new_row += 1.0 / (ndx * ndx + ndy * ndy + ndz * ndz).sqrt();
            }
            *self
                .ee_molecular
                .as_mut()
                .expect("molecular potential state carries bare ee") += new_row - old_row;

            let mut bare = 0.0_f64;
            let mut disc = 0.0_f64;
            for ia in 0..positions.nrows() {
                let dx = newpos[0] - positions[[ia, 0]];
                let dy = newpos[1] - positions[[ia, 1]];
                let dz = newpos[2] - positions[[ia, 2]];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                bare += -eff_charges[ia] / d;
                if self.ei_disc.is_some() {
                    disc += -eff_charges[ia] / nan_max(d, alat);
                }
            }
            self.ei_bare[gi] = bare;
            if let Some(rows) = self.ei_disc.as_mut() {
                rows[gi] = disc;
            }
        }
    }
}

/// One projection interval's outputs (per walker).
pub struct GfmcIntervalOut {
    pub w_l: f64,
    pub e_l: f64,
    /// `V_diag` at the final config (GFMC_n weighting `w_L / (V_diag - E_scf)`); 0.0 for GFMC_t.
    pub v_diag: f64,
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    pub final_key: [u32; 2],
    pub n_substeps: usize,
    /// Mesh rotation `R^T` of the FINAL measurement sub-step (identity when no
    /// sub-step ran). The GFMC force/stress fold-in evaluates the LRDMC
    /// HF-force kernels on the returned config with THIS rotation -- exactly the
    /// `latest_RTs` the JAX projection carries into `grad_e_L_{t,n}_wrt_positions`.
    pub last_rt: [[f64; 3]; 3],
    /// Liveness witness of the incremental interval potential state: number of applied
    /// one-electron potential updates. 0 on BITEXACT/debug full-recompute
    /// intervals; >0 proves the molecular or PBC incremental path stayed live.
    pub n_pot_updates: u64,
    /// Projection-scoped tabpip/winv state advanced to the returned config.
    /// Force/stress uses only its carried inverse; S-prime membership remains
    /// internal to propagation.
    pub(crate) chain_state: Option<MeshFastState>,
}

const RT_ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// AE/ECP sub-step evaluation: (e_L, b_L, kinetic elements, ecp candidates).
/// Drives the SAME kernels the reference chains use (zero kernel duplication).
#[allow(clippy::too_many_arguments)]
fn interval_eval(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: Option<&crate::ewald::EwaldRef>,
    fast: bool,
    state: Option<&MeshFastState>,
    pot: Option<&IntervalPotState>,
) -> IntervalEvalReal {
    match ecp {
        Some(e) => gfmc_lrdmc_ecp_state(
            gem,
            jastrow,
            j3,
            positions,
            eff_charges,
            r_up,
            r_dn,
            alat,
            kinetic_rt,
            ecp_rt,
            e.nuc_l,
            e.e_l,
            e.c_l,
            e.p_l,
            e.nuc_n,
            e.am,
            e.e_n,
            e.c_n,
            e.p_n,
            e.weights,
            &e.grid_points,
            e.nn,
            e.dltmove,
            pbc,
            fast,
            state,
            pot,
        ),
        None => {
            let (e_l, e_l_meas, b_l, els, sprime_mask) = gfmc_lrdmc_ae_state(
                gem,
                jastrow,
                j3,
                positions,
                eff_charges,
                r_up,
                r_dn,
                alat,
                kinetic_rt,
                pbc,
                fast,
                state,
                pot,
            );
            (e_l, e_l_meas, b_l, els, Vec::new(), sprime_mask)
        }
    }
}

/// FN categorical hop, mirroring jqmc's `cdf = cumsum(p/p.sum()); k = searchsorted(cdf, u)`
/// over the concatenated (kinetic ++ ecp) FN mesh, with JAX's clamp-at-index semantics.
fn interval_hop(
    elements: &[f64],
    cands: &[MeshCandidate],
    u: f64,
    alat: f64,
    rt: &[[f64; 3]; 3],
    r_up: &mut Array2<f64>,
    r_dn: &mut Array2<f64>,
) -> (bool, usize, [f64; 3], [f64; 3]) {
    let n_up = r_up.nrows();
    let nk = elements.len();
    let total = nk + cands.len();
    let mut fsum = 0.0;
    for &el in elements {
        fsum += nan_min(el, 0.0);
    }
    for c in cands {
        fsum += nan_min(c.3, 0.0);
    }
    // `jnp.searchsorted(cdf, u)` (side='left'), then JAX indexing clamps len -> len-1.
    // Materialize the cdf and binary-search it exactly as the reference
    // `gfmc_run_*` chain does. A
    // linear scan `if acc >= u { k = idx; break; }` falls through to `k = total - 1`
    // when the whole cdf is NaN (every `>=` false), whereas `partition_point(|c| c < u)`
    // returns 0 there. MEASURED: `jnp.searchsorted(all-NaN cdf, u) == 0` (JAX's binary
    // search takes the `cdf[mid] < u` branch, which is false on NaN), so partition_point
    // is the form that matches XLA.
    let mut cdf: Vec<f64> = Vec::with_capacity(total);
    let mut acc = 0.0;
    for idx in 0..total {
        let amp = if idx < nk {
            elements[idx]
        } else {
            cands[idx - nk].3
        };
        acc += nan_min(amp, 0.0) / fsum;
        cdf.push(acc);
    }
    let k = cdf.partition_point(|&c| c < u).min(total - 1);
    if k < nk {
        let shifts = lattice_shifts(alat, rt);
        let s = k % 6;
        if k < 6 * n_up {
            let e = k / 6;
            let old = [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]];
            for c in 0..3 {
                r_up[[e, c]] += shifts[s][c];
            }
            (true, e, [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]], old)
        } else {
            let e = (k - 6 * n_up) / 6;
            let old = [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]];
            for c in 0..3 {
                r_dn[[e, c]] += shifts[s][c];
            }
            (false, e, [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]], old)
        }
    } else {
        let (is_up, e, pos, _) = cands[k - nk];
        let old = if is_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]]
        };
        if is_up {
            for c in 0..3 {
                r_up[[e, c]] = pos[c];
            }
        } else {
            for c in 0..3 {
                r_dn[[e, c]] = pos[c];
            }
        }
        (is_up, e, pos, old)
    }
}

/// GFMC_n (fixed-population LRDMC) projection interval: `num_per_meas` fixed sub-steps of
/// the discrete projector `w_L *= b_x = -b_L / (V_diag - E_scf)` + FN hop, then the final
/// e_L / V_diag measurement at the LAST sub-step's mesh rotation (jqmc GFMC_n recomputes
/// `_jit_vmap_V_elements_n(..., latest_RTs, ...)` at the post-hop positions).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_interval_n(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    e_scf: f64,
    num_per_meas: usize,
    key0: [u32; 2],
    injected_rts: Option<&numpy::ndarray::ArrayView3<f64>>,
    random_mesh: bool,
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: Option<&crate::ewald::EwaldRef>,
    full_recompute: bool,
) -> GfmcIntervalOut {
    let mut key = key0;
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut w_l = 1.0_f64;
    let mut last_kinetic_rt = RT_ID;
    let mut last_rt = RT_ID;
    let mut interval_rt: Option<[[f64; 3]; 3]> = None;
    let mut interval_ecp_rt: Option<[[f64; 3]; 3]> = None;
    // FAST chain mode uses the Sherman-Morrison rank-1 kinetic elements; the BITEXACT
    // mode (JAX-injected RT) keeps the byte-exact reference path so the hop cdf / w_L
    // chain reproduces the XLA interval-rotation walk (test_jqmc_gfmc pin).
    //
    // The projection-scoped state is carried for EVERY basis and PBC; each cache
    // member engages only when its own prerequisites hold. `fast_mol_kinetic_ok`
    // gates only the molecular det/J3 candidate and derivative tables; the per-basis
    // per-basis routes (periodic pGTO / B-spline / pGTO+PW dets, raw-AO periodic
    // pGTO J3) engage through `CandidateBasisClass` inside the mesh kernels and
    // are counted per class, while the basis-agnostic members (DTN, ECP
    // geometry, ECP local, J1/J2 tables) persist and are advanced incrementally
    // across the interval regardless.
    let fast = injected_rts.is_none();
    let mut chain_state = if fast {
        Some(mesh_fast_state(
            gem,
            jastrow,
            j3,
            &r_up.view(),
            &r_dn.view(),
            pbc,
        ))
    } else {
        None
    };
    if let Some(st) = chain_state.as_mut() {
        st.reuse_fixed_mesh = true;
        if fast_mol_kinetic_ok(gem, pbc) {
            bump_route_counter(&GFMC_DET_FAST_INTERVALS);
        } else {
            bump_route_counter(&GFMC_REDUCED_STATE_INTERVALS);
        }
    }
    // Incremental interval potential state for fast chains;
    // `full_recompute=true` is the debug/oracle twin route). The BITEXACT mode
    // keeps the byte-exact full recompute unconditionally.
    let mut pot = if fast && !full_recompute {
        Some(match pbc {
            Some(p) => IntervalPotState::build(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
                p,
            ),
            None => IntervalPotState::build_molecular(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
            ),
        })
    } else {
        None
    };
    for _ in 0..num_per_meas {
        // jqmc `_split_step_keys`: key,rot = split(key); key,move = split(key).
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let rot_key = s[1];
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let move_key = s[1];
        let rt = *interval_rt.get_or_insert_with(|| match injected_rts {
            Some(rts) => {
                // BITEXACT: consume the first JAX-drawn R^T for the interval.
                // Later injected rotations are accepted for the bridge/key
                // contract but not used.
                let mut m = [[0.0_f64; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        m[a][b] = rts[[0, a, b]];
                    }
                }
                m
            }
            None => sample_rt_from_key(rot_key, random_mesh),
        });
        let ecp_rt = *interval_ecp_rt
            .get_or_insert_with(|| sample_rt_from_key(ecp_rotation_key(rot_key), random_mesh));
        let (e_l, _, b_l, elements, cands, _) = interval_eval(
            gem,
            jastrow,
            j3,
            positions,
            eff_charges,
            &r_up.view(),
            &r_dn.view(),
            alat,
            &rt,
            &ecp_rt,
            ecp,
            pbc,
            fast,
            chain_state.as_ref(),
            pot.as_ref(),
        );
        // jqmc `_body_step_core`: b_x = 1/(V_diag - E_scf) * (-V_nondiag); V_diag = e_L - b_L.
        let b_x = 1.0 / ((e_l - b_l) - e_scf) * (-b_l);
        w_l *= b_x;
        // Hop uniform: bit-exact threefry replay of `jax.random.uniform(move_key)`.
        let u = crate::threefry::uniform_f64(move_key, 1, 0.0, 1.0)[0];
        let (is_up, e, newpos, oldpos) =
            interval_hop(&elements, &cands, u, alat, &rt, &mut r_up, &mut r_dn);
        if let Some(st) = chain_state.as_mut() {
            st.advance_after_hop(
                gem,
                jastrow,
                j3,
                &r_up.view(),
                &r_dn.view(),
                is_up,
                e,
                oldpos,
                newpos,
            );
        }
        if let Some(ps) = pot.as_mut() {
            ps.advance_after_hop(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
                pbc,
                is_up,
                e,
                oldpos,
                newpos,
            );
        }
        last_kinetic_rt = rt;
        last_rt = ecp_rt;
    }
    // Final-config measurement with the LAST sub-step's RT (identity when num_per_meas == 0,
    // matching the jqmc fori_loop's RT=eye carry init). The measurement is a FRESH
    // full evaluation (pot=None), which bounds the drift of the final e_L.
    let (e_fin_prop, e_fin_meas, b_fin, _, _, _) = interval_eval(
        gem,
        jastrow,
        j3,
        positions,
        eff_charges,
        &r_up.view(),
        &r_dn.view(),
        alat,
        &last_kinetic_rt,
        &last_rt,
        ecp,
        pbc,
        fast,
        chain_state.as_ref(),
        None,
    );
    GfmcIntervalOut {
        w_l,
        // novar=0 split: measurement e_L for the estimator numerator;
        // V_diag keeps the PROPAGATION composition for w/(V_diag - E_scf).
        e_l: e_fin_meas,
        v_diag: e_fin_prop - b_fin,
        r_up,
        r_dn,
        final_key: key,
        n_substeps: num_per_meas,
        last_rt,
        n_pot_updates: pot.map(|ps| ps.n_updates).unwrap_or(0),
        chain_state,
    }
}

/// GFMC_t (tau-discretized) projection interval, FAST chain mode only (see module note:
/// the escape-time sub-step count is fp-dependent, so no bitexact contract exists; the
/// pure-JAX path IS the bitexact/verification mode). Escape-time loop `tau_update =
/// min(tau_left, ln(1-xi)/b_L)`, `w_L *= exp(-tau_update * e_L)`, FN hop while tau remains.
/// Returns the LAST sub-step's e_L (jqmc `_projection_t` semantics: the e_L when tau ran
/// out, evaluated at the pre-hop positions) and the per-walker sub-step count
/// (`projection_counter`).
#[allow(clippy::too_many_arguments)]
pub fn gfmc_interval_t(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    tau: f64,
    key0: [u32; 2],
    random_mesh: bool,
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: Option<&crate::ewald::EwaldRef>,
    full_recompute: bool,
) -> GfmcIntervalOut {
    let mut key = key0;
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut w_l = 1.0_f64;
    let mut tau_left = tau;
    let mut e_l_last = 0.0_f64;
    let mut last_rt = RT_ID;
    let mut interval_rt: Option<[[f64; 3]; 3]> = None;
    let mut interval_ecp_rt: Option<[[f64; 3]; 3]> = None;
    let mut count = 0usize;
    // The determinant/J3 derivative tables depend only on the electronic
    // configuration, not on the random kinetic/ECP mesh rotation. Carry them
    // through the variable-length tau loop exactly as GFMC_n does. Candidate
    // mesh AO tables are reused too, because the rotation is fixed within
    // each escape-time projection interval.
    //
    // The state is carried for EVERY basis and PBC (per-member gating): the
    // molecular det/J3 tables engage only under `fast_mol_kinetic_ok`, the
    // per-basis candidate and derivative routes engage through
    // `CandidateBasisClass` inside the mesh kernels (counted per class), and
    // the basis-agnostic DTN / ECP-geometry / ECP-local / J1J2 caches persist
    // and advance incrementally across the interval for all bases.
    let mut chain_state = Some(crate::timed!(
        "gfmc.state_build",
        mesh_fast_state(gem, jastrow, j3, &r_up.view(), &r_dn.view(), pbc,)
    ));
    if let Some(st) = chain_state.as_mut() {
        st.reuse_fixed_mesh = true;
        if fast_mol_kinetic_ok(gem, pbc) {
            bump_route_counter(&GFMC_DET_FAST_INTERVALS);
        } else {
            bump_route_counter(&GFMC_REDUCED_STATE_INTERVALS);
        }
    }
    // Incremental interval potential state; the escape-time
    // chain is FAST-only by construction). `full_recompute=true` selects the
    // debug/oracle twin. NOTE: the potential enters ONLY the diagonal e_L (and
    // hence w_L) -- b_L and the hop cdf are untouched, so the sub-step count
    // and the walk are identical between the two routes.
    let mut pot = if !full_recompute {
        Some(match pbc {
            Some(p) => IntervalPotState::build(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
                p,
            ),
            None => IntervalPotState::build_molecular(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
            ),
        })
    } else {
        None
    };
    while tau_left > 0.0 {
        count += 1;
        // jqmc `_projection_t_core` split order: rotation / xi (escape) / move.
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let rot_key = s[1];
        let rt = *interval_rt.get_or_insert_with(|| sample_rt_from_key(rot_key, random_mesh));
        let ecp_rt = *interval_ecp_rt
            .get_or_insert_with(|| sample_rt_from_key(ecp_rotation_key(rot_key), random_mesh));
        // The measurement e_L (last sub-step, jqmc `_projection_t` semantics) is
        // evaluated at THIS config with THIS rotation; the force fold-in reuses it.
        last_rt = ecp_rt;
        let (e_l, e_l_meas, b_l, elements, cands, _sprime_mask) = crate::timed!(
            "gfmc.interval_eval",
            interval_eval(
                gem,
                jastrow,
                j3,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                &ecp_rt,
                ecp,
                pbc,
                true,
                chain_state.as_ref(),
                pot.as_ref(),
            )
        );
        // novar=0 split: the carried MEASUREMENT e_L; the weight below uses
        // the clipped propagation e_L.
        e_l_last = e_l_meas;
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let xi_key = s[1];
        let xi = crate::threefry::uniform_f64(xi_key, 1, 0.0, 1.0)[0];
        let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l);
        w_l *= (-tau_update * e_l).exp();
        tau_left -= tau_update;
        // The move uniform is drawn unconditionally per sub-step (jqmc order); the hop
        // itself only fires while projection time remains.
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let mv_key = s[1];
        let u = crate::threefry::uniform_f64(mv_key, 1, 0.0, 1.0)[0];
        if tau_left > 0.0 {
            let (is_up, e, newpos, oldpos) = crate::timed!(
                "gfmc.interval_hop",
                interval_hop(&elements, &cands, u, alat, &rt, &mut r_up, &mut r_dn)
            );
            if let Some(st) = chain_state.as_mut() {
                crate::timed!(
                    "gfmc.advance_hop",
                    st.advance_after_hop(
                        gem,
                        jastrow,
                        j3,
                        &r_up.view(),
                        &r_dn.view(),
                        is_up,
                        e,
                        oldpos,
                        newpos,
                    )
                );
            }
            if let Some(ps) = pot.as_mut() {
                ps.advance_after_hop(
                    positions,
                    eff_charges,
                    &r_up.view(),
                    &r_dn.view(),
                    alat,
                    ecp,
                    pbc,
                    is_up,
                    e,
                    oldpos,
                    newpos,
                );
            }
        }
    }
    GfmcIntervalOut {
        w_l,
        e_l: e_l_last,
        v_diag: 0.0,
        r_up,
        r_dn,
        final_key: key,
        n_substeps: count,
        last_rt,
        n_pot_updates: pot.map(|ps| ps.n_updates).unwrap_or(0),
        chain_state,
    }
}

// ============================================================================
// Complex-twist (k != Gamma) GFMC projection intervals for fast chains.
// Mirror `gfmc_interval_{n,t}` EXACTLY (same threefry split order -> the fast
// native chain draws every uniform from the same bit-exact stream the JAX complex
// projection does) but drive the verified complex e_L kernels
// (`gfmc_lrdmc_{ae,ecp}_complex`) with fixed-phase Re-projection at the three real
// decision points -- escape `Re(b_L)`, weight `exp(-tau*Re(e_L))` / `b_x` on
// `Re(e_L),Re(b_L)`, and the FN categorical over `Re(c_min0(.))` (the SAME
// reduction the reference `gfmc_run_{ae,ecp}_complex` chains use -> zero e_L kernel
// duplication). FAST-mode-only BY DESIGN:
//   * GFMC_t: the escape-time sub-step count is fp-dependent (`log(1-xi)/Re(b_L)`),
//     so no byte-for-byte bitexact contract can exist (the real GFMC_t rationale).
//   * GFMC_n: the complex e_L carries the twist kernel-parity fp floor (complex det
//     + Re-projection) which enters the hop cdf and w_L, so a bitexact CHAIN-identity
//     vs the XLA per-step path is not guaranteed -- exactly the periodic real-Gamma
//     GFMC_n rationale (already fast-only). Both are statistically guarded.
// Energy-only: complex force/stress stays on the per-step JAX dispatch (driver gate).
// ============================================================================

/// One complex-twist projection interval's outputs (per walker). `e_l` is the full
/// complex measurement (the driver Re's for the energy accumulation and stores the
/// complex e_L, mirroring the complex MCMC energy chunk); `w_l` / `v_diag` are the
/// Re-projected real branching quantities.
pub struct GfmcIntervalOutComplex {
    pub w_l: f64,
    pub e_l: Complex64,
    pub v_diag: f64,
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    pub final_key: [u32; 2],
    pub n_substeps: usize,
    /// The LAST sub-step's mesh rotation -- the rotation the measurement `e_l` was
    /// evaluated at, and therefore the one the complex force/stress fold-in reuses.
    pub last_rt: [[f64; 3]; 3],
    /// Incremental-potential liveness counter (see [`GfmcIntervalOut::n_pot_updates`]).
    pub n_pot_updates: u64,
    pub(crate) chain_state: Option<MeshFastStateC>,
}

/// Complex AE/ECP sub-step evaluation: (e_L, b_L, kinetic elements, ecp candidates).
/// Drives the same complex kernels as the reference chains. Drivers pass their
/// carried per-interval [`MeshFastStateC`] through `state`.
#[allow(clippy::too_many_arguments)]
fn interval_eval_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up: &ArrayView2<f64>,
    r_dn: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: &crate::ewald::EwaldRef,
    fast: bool,
    state: Option<&MeshFastStateC>,
    pot: Option<&IntervalPotState>,
) -> IntervalEvalComplex {
    match ecp {
        Some(e) => gfmc_lrdmc_ecp_complex_state(
            gem,
            jastrow,
            j3c,
            j_matrix,
            positions,
            eff_charges,
            r_up,
            r_dn,
            alat,
            kinetic_rt,
            ecp_rt,
            pbc,
            e.nuc_l,
            e.e_l,
            e.c_l,
            e.p_l,
            e.nuc_n,
            e.am,
            e.e_n,
            e.c_n,
            e.p_n,
            e.weights,
            &e.grid_points,
            e.nn,
            e.dltmove,
            fast,
            state,
            pot,
        ),
        None => {
            let (e_l, e_l_meas, b_l, els, sprime_mask) = gfmc_lrdmc_ae_complex_state(
                gem,
                jastrow,
                j3c,
                j_matrix,
                positions,
                eff_charges,
                r_up,
                r_dn,
                alat,
                kinetic_rt,
                pbc,
                fast,
                state,
                pot,
            );
            (e_l, e_l_meas, b_l, els, Vec::new(), sprime_mask)
        }
    }
}

/// Complex fixed-phase FN categorical hop: mirrors `interval_hop` but the amplitudes
/// are `Re(c_min0(.))` (the reduction the reference `gfmc_run_{ae,ecp}_complex` use).
/// Returns `(is_up, electron, new_pos, old_pos)` so the driver can advance
/// the carried [`MeshFastStateC`] after the hop (same contract as `interval_hop`).
fn interval_hop_complex(
    elements: &[Complex64],
    cands: &[MeshCandidateC],
    u: f64,
    alat: f64,
    rt: &[[f64; 3]; 3],
    r_up: &mut Array2<f64>,
    r_dn: &mut Array2<f64>,
) -> (bool, usize, [f64; 3], [f64; 3]) {
    let n_up = r_up.nrows();
    let nk = elements.len();
    let total = nk + cands.len();
    let mut fsum = 0.0;
    for &el in elements {
        fsum += c_min0(el).re;
    }
    for c in cands {
        fsum += c_min0(c.3).re;
    }
    // `jnp.searchsorted(cdf, u)` + JAX index clamp -- see the real `interval_hop`: the
    // materialised-cdf binary search is the form that matches XLA on an all-NaN cdf
    // (index 0); the linear `acc >= u` scan fell to `total - 1`.
    let mut cdf: Vec<f64> = Vec::with_capacity(total);
    let mut acc = 0.0;
    for idx in 0..total {
        let amp = if idx < nk {
            c_min0(elements[idx]).re
        } else {
            c_min0(cands[idx - nk].3).re
        };
        acc += amp / fsum;
        cdf.push(acc);
    }
    let k = cdf.partition_point(|&c| c < u).min(total - 1);
    if k < nk {
        let shifts = lattice_shifts(alat, rt);
        let s = k % 6;
        if k < 6 * n_up {
            let e = k / 6;
            let old = [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]];
            for c in 0..3 {
                r_up[[e, c]] += shifts[s][c];
            }
            (true, e, [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]], old)
        } else {
            let e = (k - 6 * n_up) / 6;
            let old = [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]];
            for c in 0..3 {
                r_dn[[e, c]] += shifts[s][c];
            }
            (false, e, [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]], old)
        }
    } else {
        let (is_up, e, pos, _) = cands[k - nk];
        let old = if is_up {
            [r_up[[e, 0]], r_up[[e, 1]], r_up[[e, 2]]]
        } else {
            [r_dn[[e, 0]], r_dn[[e, 1]], r_dn[[e, 2]]]
        };
        if is_up {
            for c in 0..3 {
                r_up[[e, c]] = pos[c];
            }
        } else {
            for c in 0..3 {
                r_dn[[e, c]] = pos[c];
            }
        }
        (is_up, e, pos, old)
    }
}

/// Complex-twist GFMC_n projection interval (fast chain only). Mirrors
/// `gfmc_interval_n` with Re-projected `b_x` and `V_diag`.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_interval_n_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    e_scf: f64,
    num_per_meas: usize,
    key0: [u32; 2],
    injected_rts: Option<&numpy::ndarray::ArrayView3<f64>>,
    random_mesh: bool,
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: &crate::ewald::EwaldRef,
    full_recompute: bool,
) -> GfmcIntervalOutComplex {
    let mut key = key0;
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut w_l = 1.0_f64;
    let mut last_kinetic_rt = RT_ID;
    let mut last_rt = RT_ID;
    let mut interval_rt: Option<[[f64; 3]; 3]> = None;
    let mut interval_ecp_rt: Option<[[f64; 3]; 3]> = None;
    // Same 2-mode divergence point as the REAL `gfmc_interval_n`: injected (JAX-drawn)
    // R^T => BITEXACT, which also keeps the byte-exact reference mesh kernels; no
    // injection => FAST (native trig off the same bit-exact threefry angles + the
    // complex Sherman-Morrison mesh state).
    //
    // Mirroring the real driver, the complex state is built once per projection
    // interval and advanced after every hop; the candidate
    // AO tables and the ECP geometry/local caches live across the interval's fixed
    // mesh rotation with per-hop refresh of the moved electron's rows. A build
    // decline (non-contiguous `j_matrix`) leaves the interval on the per-substep
    // path (counted, never silent).
    let fast = injected_rts.is_none();
    let mut chain_state = if fast {
        mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, &r_up.view(), &r_dn.view())
    } else {
        None
    };
    if let Some(st) = chain_state.as_mut() {
        st.reuse_fixed_mesh = true;
        bump_route_counter(&GFMC_CPLX_STATE_INTERVALS);
    } else if fast {
        bump_route_counter(&GFMC_CPLX_STATE_GATE_MISS);
    }
    // Incremental interval potential state. The Ewald and local-ECP
    // pieces of the complex-twist diagonal are REAL and identical to the real
    // interval's; the state enters only the diagonal e_L / w_L.
    let mut pot = if fast && !full_recompute {
        Some(IntervalPotState::build(
            positions,
            eff_charges,
            &r_up.view(),
            &r_dn.view(),
            alat,
            ecp,
            pbc,
        ))
    } else {
        None
    };
    for _ in 0..num_per_meas {
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let rot_key = s[1];
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let move_key = s[1];
        let rt = *interval_rt.get_or_insert_with(|| match injected_rts {
            Some(rts) => {
                let mut m = [[0.0_f64; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        m[a][b] = rts[[0, a, b]];
                    }
                }
                m
            }
            None => sample_rt_from_key(rot_key, random_mesh),
        });
        let ecp_rt = *interval_ecp_rt
            .get_or_insert_with(|| sample_rt_from_key(ecp_rotation_key(rot_key), random_mesh));
        let (e_l, _, b_l, elements, cands, _) = interval_eval_complex(
            gem,
            jastrow,
            j3c,
            j_matrix,
            positions,
            eff_charges,
            &r_up.view(),
            &r_dn.view(),
            alat,
            &rt,
            &ecp_rt,
            ecp,
            pbc,
            fast,
            chain_state.as_ref(),
            pot.as_ref(),
        );
        // Re-projected discrete projector (fixed-phase): V_diag = Re(e_L - b_L).
        let b_x = 1.0 / ((e_l.re - b_l.re) - e_scf) * (-b_l.re);
        w_l *= b_x;
        let u = crate::threefry::uniform_f64(move_key, 1, 0.0, 1.0)[0];
        let (is_up, e, newpos, oldpos) =
            interval_hop_complex(&elements, &cands, u, alat, &rt, &mut r_up, &mut r_dn);
        if let Some(st) = chain_state.as_mut() {
            st.advance_after_hop(
                gem,
                jastrow,
                j3c,
                j_matrix,
                &r_up.view(),
                &r_dn.view(),
                is_up,
                e,
                oldpos,
                newpos,
            );
        }
        if let Some(ps) = pot.as_mut() {
            ps.advance_after_hop(
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                ecp,
                Some(pbc),
                is_up,
                e,
                oldpos,
                newpos,
            );
        }
        last_kinetic_rt = rt;
        last_rt = ecp_rt;
    }
    // Final measurement = fresh full evaluation (pot=None), as in the real interval.
    let (e_fin_prop, e_fin_meas, b_fin, _, _, _) = interval_eval_complex(
        gem,
        jastrow,
        j3c,
        j_matrix,
        positions,
        eff_charges,
        &r_up.view(),
        &r_dn.view(),
        alat,
        &last_kinetic_rt,
        &last_rt,
        ecp,
        pbc,
        fast,
        chain_state.as_ref(),
        None,
    );
    GfmcIntervalOutComplex {
        w_l,
        // novar=0 split (complex twin): meas e_L; prop V_diag.
        e_l: e_fin_meas,
        v_diag: e_fin_prop.re - b_fin.re,
        r_up,
        r_dn,
        final_key: key,
        n_substeps: num_per_meas,
        last_rt,
        n_pot_updates: pot.map(|ps| ps.n_updates).unwrap_or(0),
        chain_state,
    }
}

/// Complex-twist GFMC_t projection interval (fast chain only). Mirrors
/// `gfmc_interval_t` with Re-projected escape / weight.
#[allow(clippy::too_many_arguments)]
pub fn gfmc_interval_t_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    tau: f64,
    key0: [u32; 2],
    random_mesh: bool,
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: &crate::ewald::EwaldRef,
    full_recompute: bool,
) -> GfmcIntervalOutComplex {
    let mut key = key0;
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let mut w_l = 1.0_f64;
    let mut tau_left = tau;
    let mut e_l_last = Complex64::new(0.0, 0.0);
    let mut last_rt = RT_ID;
    let mut interval_rt: Option<[[f64; 3]; 3]> = None;
    let mut interval_ecp_rt: Option<[[f64; 3]; 3]> = None;
    let mut count = 0usize;
    // Mirroring real `gfmc_interval_t`, carry one complex state per escape-time
    // projection interval and advance it after every hop;
    // candidate tables reusable because the rotation is fixed within the interval.
    let mut chain_state = crate::timed!(
        "gfmc.state_build_c",
        mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, &r_up.view(), &r_dn.view())
    );
    if let Some(st) = chain_state.as_mut() {
        st.reuse_fixed_mesh = true;
        bump_route_counter(&GFMC_CPLX_STATE_INTERVALS);
    } else {
        bump_route_counter(&GFMC_CPLX_STATE_GATE_MISS);
    }
    // The incremental interval potential state reuses the complex diagonal's
    // real Ewald and local-ECP pieces from the real interval.
    let mut pot = if !full_recompute {
        Some(IntervalPotState::build(
            positions,
            eff_charges,
            &r_up.view(),
            &r_dn.view(),
            alat,
            ecp,
            pbc,
        ))
    } else {
        None
    };
    while tau_left > 0.0 {
        count += 1;
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let rot_key = s[1];
        let rt = *interval_rt.get_or_insert_with(|| sample_rt_from_key(rot_key, random_mesh));
        let ecp_rt = *interval_ecp_rt
            .get_or_insert_with(|| sample_rt_from_key(ecp_rotation_key(rot_key), random_mesh));
        // The measurement e_L (last sub-step, jqmc `_projection_t` semantics) is evaluated
        // at THIS config with THIS rotation; the force fold-in reuses it.
        last_rt = ecp_rt;
        let (e_l, e_l_meas, b_l, elements, cands, _sprime_mask) = crate::timed!(
            "gfmc.interval_eval_c",
            interval_eval_complex(
                gem,
                jastrow,
                j3c,
                j_matrix,
                positions,
                eff_charges,
                &r_up.view(),
                &r_dn.view(),
                alat,
                &rt,
                &ecp_rt,
                ecp,
                pbc,
                true,
                chain_state.as_ref(),
                pot.as_ref(),
            )
        );
        // novar=0 split: carried MEASUREMENT e_L (complex).
        e_l_last = e_l_meas;
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let xi_key = s[1];
        let xi = crate::threefry::uniform_f64(xi_key, 1, 0.0, 1.0)[0];
        let tau_update = nan_min(tau_left, (1.0 - xi).ln() / b_l.re);
        w_l *= (-tau_update * e_l.re).exp();
        tau_left -= tau_update;
        let s = crate::threefry::split(key, 2);
        key = s[0];
        let mv_key = s[1];
        let u = crate::threefry::uniform_f64(mv_key, 1, 0.0, 1.0)[0];
        if tau_left > 0.0 {
            let (is_up, e, newpos, oldpos) = crate::timed!(
                "gfmc.interval_hop_c",
                interval_hop_complex(&elements, &cands, u, alat, &rt, &mut r_up, &mut r_dn)
            );
            if let Some(st) = chain_state.as_mut() {
                crate::timed!(
                    "gfmc.advance_hop_c",
                    st.advance_after_hop(
                        gem,
                        jastrow,
                        j3c,
                        j_matrix,
                        &r_up.view(),
                        &r_dn.view(),
                        is_up,
                        e,
                        oldpos,
                        newpos,
                    )
                );
            }
            if let Some(ps) = pot.as_mut() {
                ps.advance_after_hop(
                    positions,
                    eff_charges,
                    &r_up.view(),
                    &r_dn.view(),
                    alat,
                    ecp,
                    Some(pbc),
                    is_up,
                    e,
                    oldpos,
                    newpos,
                );
            }
        }
    }
    GfmcIntervalOutComplex {
        w_l,
        e_l: e_l_last,
        v_diag: 0.0,
        r_up,
        r_dn,
        final_key: key,
        n_substeps: count,
        last_rt,
        n_pot_updates: pot.map(|ps| ps.n_updates).unwrap_or(0),
        chain_state,
    }
}

/// Real per-electron determinant derivative block `(grad_up, grad_dn, lap_up, lap_dn)`.
pub type DetGradLapReal = (
    Array2<f64>,
    Array2<f64>,
    numpy::ndarray::Array1<f64>,
    numpy::ndarray::Array1<f64>,
);

/// Complex per-electron determinant derivative block.
pub type DetGradLapComplex = (
    Array2<Complex64>,
    Array2<Complex64>,
    numpy::ndarray::Array1<Complex64>,
    numpy::ndarray::Array1<Complex64>,
);

/// Apply one prescribed hop to the walker arrays; returns the old position.
fn probe_apply_hop(
    r_up: &mut Array2<f64>,
    r_dn: &mut Array2<f64>,
    is_up: bool,
    e: usize,
    newpos: [f64; 3],
) -> Result<[f64; 3], String> {
    let r = if is_up { r_up } else { r_dn };
    if e >= r.nrows() {
        return Err(format!(
            "hop electron index {e} out of range for {} {} electrons",
            r.nrows(),
            if is_up { "up" } else { "dn" }
        ));
    }
    let old = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
    for c in 0..3 {
        r[[e, c]] = newpos[c];
    }
    Ok(old)
}

/// Carried-versus-fresh probe for the REAL interval determinant derivative
/// caches.
///
/// Builds the PRODUCTION projection-scoped [`MeshFastState`] at `(r_up0,
/// r_dn0)`, evaluates the continuum-kinetic determinant grad/lap once per
/// sub-step exactly as `interval_eval` does, and advances the state with the
/// production `advance_after_hop` for every prescribed hop. At each requested
/// checkpoint (hop count, 0 = seed) it returns the carried result together with
/// a from-scratch evaluation: a NEW state built at the current configuration,
/// whose first evaluation seeds every table and builds the contraction weights
/// by the full matrix products. The third member of each row is the
/// matrix-product contraction (`contract_det_grad_lap_cache`) of the CARRIED
/// derivative cache with the carried `G^-1` and value state, which isolates the
/// carried contraction rows from the rest of the chain. Errors when the
/// configuration does not take the streamed route (`fast_det_gradlap_ok`), so
/// the probe cannot pass vacuously.
#[allow(clippy::too_many_arguments)]
pub fn debug_det_grad_lap_chain(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    pbc: Option<&crate::ewald::EwaldRef>,
    hops: &[(bool, usize, [f64; 3])],
    checkpoints: &[usize],
) -> Result<Vec<(usize, DetGradLapReal, DetGradLapReal, DetGradLapReal)>, String> {
    if !fast_det_gradlap_ok(gem, pbc) {
        return Err("geminal does not take the streamed determinant derivative route".into());
    }
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let fresh = |ru: &Array2<f64>, rd: &Array2<f64>| -> DetGradLapReal {
        let st = mesh_fast_state(gem, jastrow, j3, &ru.view(), &rd.view(), pbc);
        st.det_grad_lap(gem, &ru.view(), &rd.view())
    };
    let twin = |st: &MeshFastState| -> DetGradLapReal {
        let cache = st.det_grad_cache.borrow();
        crate::determinant::contract_det_grad_lap_cache(
            gem,
            &st.ginv.view(),
            &st.det_state,
            cache.as_ref().expect("det derivative cache seeded"),
        )
    };
    let mut st = mesh_fast_state(gem, jastrow, j3, &r_up.view(), &r_dn.view(), pbc);
    st.reuse_fixed_mesh = true;
    let mut out = Vec::new();
    let carried = st.det_grad_lap(gem, &r_up.view(), &r_dn.view());
    if checkpoints.contains(&0) {
        out.push((0, carried, fresh(&r_up, &r_dn), twin(&st)));
    }
    for (k, &(is_up, e, newpos)) in hops.iter().enumerate() {
        let old = probe_apply_hop(&mut r_up, &mut r_dn, is_up, e, newpos)?;
        st.advance_after_hop(
            gem,
            jastrow,
            j3,
            &r_up.view(),
            &r_dn.view(),
            is_up,
            e,
            old,
            newpos,
        );
        let carried = st.det_grad_lap(gem, &r_up.view(), &r_dn.view());
        if checkpoints.contains(&(k + 1)) {
            out.push((k + 1, carried, fresh(&r_up, &r_dn), twin(&st)));
        }
    }
    Ok(out)
}

/// Complex twin of [`debug_det_grad_lap_chain`] for [`MeshFastStateC`]: the
/// carried `det_grad_lap_c` result versus a fresh state built at the current
/// configuration, plus the matrix-product contraction of the carried cache
/// ([`crate::complex_det::contract_det_grad_lap_cache_complex_matmul`], the
/// carried `kin_ginv` and value state), which isolates the carried contraction
/// weights. Errors when the geminal does not take the streamed route
/// (`complex_det_gradlap_ok`) or the state build declines.
#[allow(clippy::too_many_arguments)]
pub fn debug_det_grad_lap_chain_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    hops: &[(bool, usize, [f64; 3])],
    checkpoints: &[usize],
) -> Result<
    Vec<(
        usize,
        DetGradLapComplex,
        DetGradLapComplex,
        DetGradLapComplex,
    )>,
    String,
> {
    if !complex_det_gradlap_ok(gem) {
        return Err(
            "complex geminal does not take the streamed determinant derivative route".into(),
        );
    }
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let fresh = |ru: &Array2<f64>, rd: &Array2<f64>| -> Result<DetGradLapComplex, String> {
        let st = mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, &ru.view(), &rd.view())
            .ok_or("complex chain state build declined")?;
        Ok(st.det_grad_lap_c(gem, &ru.view(), &rd.view()))
    };
    let twin = |st: &MeshFastStateC| -> DetGradLapComplex {
        let cache = st.det_grad_cache.borrow();
        crate::complex_det::contract_det_grad_lap_cache_complex_matmul(
            gem,
            cache.as_ref().expect("complex det derivative cache seeded"),
        )
    };
    let mut st = mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, &r_up.view(), &r_dn.view())
        .ok_or("complex chain state build declined")?;
    st.reuse_fixed_mesh = true;
    let mut out = Vec::new();
    let carried = st.det_grad_lap_c(gem, &r_up.view(), &r_dn.view());
    if checkpoints.contains(&0) {
        out.push((0, carried, fresh(&r_up, &r_dn)?, twin(&st)));
    }
    for (k, &(is_up, e, newpos)) in hops.iter().enumerate() {
        let old = probe_apply_hop(&mut r_up, &mut r_dn, is_up, e, newpos)?;
        st.advance_after_hop(
            gem,
            jastrow,
            j3c,
            j_matrix,
            &r_up.view(),
            &r_dn.view(),
            is_up,
            e,
            old,
            newpos,
        );
        let carried = st.det_grad_lap_c(gem, &r_up.view(), &r_dn.view());
        if checkpoints.contains(&(k + 1)) {
            out.push((k + 1, carried, fresh(&r_up, &r_dn)?, twin(&st)));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Long-chain mesh / inverse / DTN-table probe (test only).
// ---------------------------------------------------------------------------

/// One named array of a mesh-chain probe snapshot (row-major data + shape).
pub enum ProbeArray {
    Real(Vec<usize>, Vec<f64>),
    Complex(Vec<usize>, Vec<Complex64>),
}

/// Named arrays of one probe snapshot.
pub type ProbeSnapshot = Vec<(&'static str, ProbeArray)>;

/// One checkpoint of [`debug_mesh_chain`] / [`debug_mesh_chain_complex`].
pub struct MeshChainRow {
    pub hop: usize,
    pub r_up: Array2<f64>,
    pub r_dn: Array2<f64>,
    /// The carried production state after `hop` hops (and its evaluation).
    pub carried: ProbeSnapshot,
    /// A new state built at the same configuration (every table seeded fresh).
    pub fresh: ProbeSnapshot,
    /// A new state at the same configuration whose inverse is the CARRIED
    /// `G^-1` (contraction weights rebuilt from it): isolates the derived
    /// candidate state from the inverse chain.
    pub oracle: ProbeSnapshot,
}

/// Hop selection of the mesh-chain probe.
#[derive(Clone, Copy, Debug)]
pub enum ProbeHopPolicy {
    /// The production categorical FN hop with the prescribed uniform per hop.
    Categorical,
    /// The kinetic candidate of smallest `|element|` above `floor` whose
    /// electron differs from the previous hop's (the 1/rho amplification class
    /// of `test_lrdmc_mesh_chain_consistency.py`).
    Adversarial { floor: f64 },
}

fn probe_real1(v: &[f64]) -> ProbeArray {
    ProbeArray::Real(vec![v.len()], v.to_vec())
}

fn probe_real2(a: &Array2<f64>) -> ProbeArray {
    ProbeArray::Real(vec![a.nrows(), a.ncols()], a.iter().copied().collect())
}

fn probe_cplx1(v: &[Complex64]) -> ProbeArray {
    ProbeArray::Complex(vec![v.len()], v.to_vec())
}

fn probe_cplx2(a: &Array2<Complex64>) -> ProbeArray {
    ProbeArray::Complex(vec![a.nrows(), a.ncols()], a.iter().copied().collect())
}

/// The DTN members shared by the real and the complex interval states.
fn probe_dtn_members(
    out: &mut ProbeSnapshot,
    kinetic: &Option<crate::dtn::DtnCandidateTable>,
    ecp: &Option<crate::dtn::DtnCandidateTable>,
    grad: &Option<crate::dtn::DtnGradLapCache>,
) {
    for (prefix, table) in [("dtn_kin", kinetic), ("dtn_ecp", ecp)] {
        if let Some(t) = table.as_ref() {
            let (n_e, n_k, n_atom) = t.probe_dims();
            let (log_delta, cand_pos, cand_r_ei) = t.probe_arrays();
            let names: [&'static str; 3] = if prefix == "dtn_kin" {
                [
                    "dtn_kin_log_delta",
                    "dtn_kin_candidate_pos",
                    "dtn_kin_cand_r_ei",
                ]
            } else {
                [
                    "dtn_ecp_log_delta",
                    "dtn_ecp_candidate_pos",
                    "dtn_ecp_cand_r_ei",
                ]
            };
            out.push((names[0], ProbeArray::Real(vec![n_e, n_k], log_delta)));
            out.push((names[1], ProbeArray::Real(vec![n_e, n_k, 3], cand_pos)));
            out.push((
                names[2],
                ProbeArray::Real(vec![n_e, n_k, n_atom], cand_r_ei),
            ));
        }
    }
    if let Some(c) = grad.as_ref() {
        out.push(("dtn_grad_up", probe_real2(&c.gu)));
        out.push(("dtn_grad_dn", probe_real2(&c.gd)));
        out.push(("dtn_lap_up", probe_real1(&c.lu)));
        out.push(("dtn_lap_dn", probe_real1(&c.ld)));
    }
}

type IntervalEvalReal = (f64, f64, f64, Vec<f64>, Vec<MeshCandidate>, Vec<f64>);

type IntervalEvalComplex = (
    Complex64,
    Complex64,
    Complex64,
    Vec<Complex64>,
    Vec<MeshCandidateC>,
    Vec<f64>,
);

fn probe_eval_real(ev: &IntervalEvalReal, out: &mut ProbeSnapshot) {
    out.push(("e_L", probe_real1(&[ev.0])));
    out.push(("e_L_meas", probe_real1(&[ev.1])));
    out.push(("b_L", probe_real1(&[ev.2])));
    out.push(("kinetic_elements", probe_real1(&ev.3)));
    let vals: Vec<f64> = ev.4.iter().map(|c| c.3).collect();
    let pos: Vec<f64> = ev.4.iter().flat_map(|c| c.2).collect();
    let who: Vec<f64> =
        ev.4.iter()
            .map(|c| if c.0 { c.1 as f64 } else { -1.0 - c.1 as f64 })
            .collect();
    out.push(("ecp_values", probe_real1(&vals)));
    out.push(("ecp_positions", ProbeArray::Real(vec![ev.4.len(), 3], pos)));
    out.push(("ecp_electron", probe_real1(&who)));
}

fn probe_eval_complex(ev: &IntervalEvalComplex, out: &mut ProbeSnapshot) {
    out.push(("e_L", probe_cplx1(&[ev.0])));
    out.push(("e_L_meas", probe_cplx1(&[ev.1])));
    out.push(("b_L", probe_cplx1(&[ev.2])));
    out.push(("kinetic_elements", probe_cplx1(&ev.3)));
    let vals: Vec<Complex64> = ev.4.iter().map(|c| c.3).collect();
    let pos: Vec<f64> = ev.4.iter().flat_map(|c| c.2).collect();
    let who: Vec<f64> =
        ev.4.iter()
            .map(|c| if c.0 { c.1 as f64 } else { -1.0 - c.1 as f64 })
            .collect();
    out.push(("ecp_values", probe_cplx1(&vals)));
    out.push(("ecp_positions", ProbeArray::Real(vec![ev.4.len(), 3], pos)));
    out.push(("ecp_electron", probe_real1(&who)));
}

fn probe_snapshot_real(st: &MeshFastState, ev: &IntervalEvalReal) -> ProbeSnapshot {
    let mut out = Vec::new();
    probe_eval_real(ev, &mut out);
    out.push(("ginv", probe_real2(&st.ginv)));
    if let Some(w) = st.det_wdot_cache.borrow().as_ref() {
        out.push(("winv_up", probe_real2(&w.up)));
        out.push(("winv_up_base", probe_real1(&w.up_base)));
        out.push(("winv_dn", probe_real2(&w.dn)));
        out.push(("winv_dn_base", probe_real1(&w.dn_base)));
    }
    probe_dtn_members(
        &mut out,
        &st.dtn_candidate_table.borrow(),
        &st.ecp_dtn_candidate_table.borrow(),
        &st.dtn_grad_cache.borrow(),
    );
    out
}

fn probe_snapshot_complex(st: &MeshFastStateC, ev: &IntervalEvalComplex) -> ProbeSnapshot {
    let mut out = Vec::new();
    probe_eval_complex(ev, &mut out);
    out.push(("ginv", probe_cplx2(&st.ginv)));
    out.push(("winv_up", probe_cplx2(&st.det_winv.up)));
    out.push(("winv_up_base", probe_cplx1(&st.det_winv.up_base)));
    out.push(("winv_dn", probe_cplx2(&st.det_winv.dn)));
    out.push(("winv_dn_base", probe_cplx1(&st.det_winv.dn_base)));
    probe_dtn_members(
        &mut out,
        &st.dtn_candidate_table.borrow(),
        &st.ecp_dtn_candidate_table.borrow(),
        &st.dtn_grad_cache.borrow(),
    );
    out
}

/// Kinetic hop `k` of the fixed lattice (the kinetic branch of `interval_hop`).
fn probe_kinetic_hop(
    k: usize,
    alat: f64,
    rt: &[[f64; 3]; 3],
    r_up: &mut Array2<f64>,
    r_dn: &mut Array2<f64>,
) -> (bool, usize, [f64; 3], [f64; 3]) {
    let n_up = r_up.nrows();
    let shifts = lattice_shifts(alat, rt);
    let s = k % 6;
    let (is_up, e, r) = if k < 6 * n_up {
        (true, k / 6, r_up)
    } else {
        (false, (k - 6 * n_up) / 6, r_dn)
    };
    let old = [r[[e, 0]], r[[e, 1]], r[[e, 2]]];
    for c in 0..3 {
        r[[e, c]] += shifts[s][c];
    }
    (is_up, e, [r[[e, 0]], r[[e, 1]], r[[e, 2]]], old)
}

/// Index of the smallest-magnitude kinetic element above `floor` whose electron
/// differs from `last` (electron = `index / 6`).
fn probe_adversarial_pick(mags: &[f64], floor: f64, last: Option<usize>) -> Result<usize, String> {
    let mut best: Option<(usize, f64)> = None;
    for (k, &m) in mags.iter().enumerate() {
        if !(m.is_finite() && m > floor) || Some(k / 6) == last {
            continue;
        }
        if best.is_none_or(|(_, b)| m < b) {
            best = Some((k, m));
        }
    }
    best.map(|(k, _)| k)
        .ok_or_else(|| "adversarial hop: no finite kinetic element above the floor".to_string())
}

/// Long-chain probe of the REAL projection-scoped interval state.
///
/// Builds the production [`MeshFastState`] at `(r_up0, r_dn0)` with the fixed
/// kinetic / ECP rotations of one projection interval, and for every sub-step
/// runs the production `interval_eval` (kinetic mesh, ECP mesh, e_L), picks the
/// hop (`policy`) and applies the production `advance_after_hop`. At every hop
/// count in `checkpoints` it snapshots the carried state, a fresh state built at
/// the same configuration and a same-inverse oracle (see [`MeshChainRow`]); the
/// carried snapshot also holds the winv tables rebuilt from the carried
/// `(det_state, G^-1)` (`winv_rebuilt_*`). Not used by any driver.
#[allow(clippy::too_many_arguments)]
pub fn debug_mesh_chain(
    gem: &GeminalData,
    jastrow: &JastrowData,
    j3: Option<(&OrbData, &ArrayView2<f64>)>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: Option<&crate::ewald::EwaldRef>,
    policy: ProbeHopPolicy,
    uniforms: &[f64],
    n_hops: usize,
    checkpoints: &[usize],
) -> Result<Vec<MeshChainRow>, String> {
    if matches!(policy, ProbeHopPolicy::Categorical) && uniforms.len() < n_hops {
        return Err("categorical mesh-chain probe needs one uniform per hop".into());
    }
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let eval = |st: &MeshFastState, ru: &Array2<f64>, rd: &Array2<f64>| -> IntervalEvalReal {
        interval_eval(
            gem,
            jastrow,
            j3,
            positions,
            eff_charges,
            &ru.view(),
            &rd.view(),
            alat,
            kinetic_rt,
            ecp_rt,
            ecp,
            pbc,
            true,
            Some(st),
            None,
        )
    };
    let mut st = mesh_fast_state(gem, jastrow, j3, &r_up.view(), &r_dn.view(), pbc);
    st.reuse_fixed_mesh = true;
    let mut out = Vec::new();
    let mut last: Option<usize> = None;
    for k in 0..=n_hops {
        let ev = eval(&st, &r_up, &r_dn);
        if checkpoints.contains(&k) {
            let mut carried = probe_snapshot_real(&st, &ev);
            let (wu, wub) =
                crate::coulomb_potential::det_mowdot_all_up(gem, &st.det_state, &st.ginv.view());
            let (wd, wdb) =
                crate::coulomb_potential::det_mowdot_all_dn(gem, &st.det_state, &st.ginv.view());
            carried.push(("winv_rebuilt_up", probe_real2(&wu)));
            carried.push(("winv_rebuilt_up_base", probe_real1(&wub)));
            carried.push(("winv_rebuilt_dn", probe_real2(&wd)));
            carried.push(("winv_rebuilt_dn_base", probe_real1(&wdb)));
            let mut fresh_st = mesh_fast_state(gem, jastrow, j3, &r_up.view(), &r_dn.view(), pbc);
            fresh_st.reuse_fixed_mesh = true;
            let fev = eval(&fresh_st, &r_up, &r_dn);
            let fresh = probe_snapshot_real(&fresh_st, &fev);
            let mut oracle_st = mesh_fast_state(gem, jastrow, j3, &r_up.view(), &r_dn.view(), pbc);
            oracle_st.reuse_fixed_mesh = true;
            oracle_st.ginv = st.ginv.clone();
            let oev = eval(&oracle_st, &r_up, &r_dn);
            let oracle = probe_snapshot_real(&oracle_st, &oev);
            out.push(MeshChainRow {
                hop: k,
                r_up: r_up.clone(),
                r_dn: r_dn.clone(),
                carried,
                fresh,
                oracle,
            });
        }
        if k == n_hops {
            break;
        }
        let (is_up, e, newpos, oldpos) = match policy {
            ProbeHopPolicy::Categorical => interval_hop(
                &ev.3,
                &ev.4,
                // Checked above: one uniform per hop (only this policy reads them).
                *uniforms.get(k).expect("one uniform per hop"),
                alat,
                kinetic_rt,
                &mut r_up,
                &mut r_dn,
            ),
            ProbeHopPolicy::Adversarial { floor } => {
                let mags: Vec<f64> = ev.3.iter().map(|x| x.abs()).collect();
                let pick = probe_adversarial_pick(&mags, floor, last)?;
                last = Some(pick / 6);
                probe_kinetic_hop(pick, alat, kinetic_rt, &mut r_up, &mut r_dn)
            }
        };
        st.advance_after_hop(
            gem,
            jastrow,
            j3,
            &r_up.view(),
            &r_dn.view(),
            is_up,
            e,
            oldpos,
            newpos,
        );
    }
    Ok(out)
}

/// Complex twin of [`debug_mesh_chain`] for [`MeshFastStateC`] (periodic twist).
#[allow(clippy::too_many_arguments)]
pub fn debug_mesh_chain_complex(
    gem: &ComplexGeminal,
    jastrow: &JastrowData,
    j3c: Option<&ComplexOrb>,
    j_matrix: Option<&ArrayView2<f64>>,
    positions: &ArrayView2<f64>,
    eff_charges: &[f64],
    r_up0: &ArrayView2<f64>,
    r_dn0: &ArrayView2<f64>,
    alat: f64,
    kinetic_rt: &[[f64; 3]; 3],
    ecp_rt: &[[f64; 3]; 3],
    ecp: Option<&GfmcEcpParams<'_>>,
    pbc: &crate::ewald::EwaldRef,
    policy: ProbeHopPolicy,
    uniforms: &[f64],
    n_hops: usize,
    checkpoints: &[usize],
) -> Result<Vec<MeshChainRow>, String> {
    if matches!(policy, ProbeHopPolicy::Categorical) && uniforms.len() < n_hops {
        return Err("categorical mesh-chain probe needs one uniform per hop".into());
    }
    let mut r_up = r_up0.to_owned();
    let mut r_dn = r_dn0.to_owned();
    let eval = |st: &MeshFastStateC, ru: &Array2<f64>, rd: &Array2<f64>| -> IntervalEvalComplex {
        interval_eval_complex(
            gem,
            jastrow,
            j3c,
            j_matrix,
            positions,
            eff_charges,
            &ru.view(),
            &rd.view(),
            alat,
            kinetic_rt,
            ecp_rt,
            ecp,
            pbc,
            true,
            Some(st),
            None,
        )
    };
    let build = |ru: &Array2<f64>, rd: &Array2<f64>| -> Result<MeshFastStateC, String> {
        let mut st = mesh_fast_state_complex(gem, jastrow, j3c, j_matrix, &ru.view(), &rd.view())
            .ok_or("complex chain state build declined")?;
        st.reuse_fixed_mesh = true;
        Ok(st)
    };
    let mut st = build(&r_up, &r_dn)?;
    let mut out = Vec::new();
    let mut last: Option<usize> = None;
    for k in 0..=n_hops {
        let ev = eval(&st, &r_up, &r_dn);
        if checkpoints.contains(&k) {
            let mut carried = probe_snapshot_complex(&st, &ev);
            let rebuilt = init_det_wdot_cache_c(gem, &st.det_state, &st.ginv.view());
            carried.push(("winv_rebuilt_up", probe_cplx2(&rebuilt.up)));
            carried.push(("winv_rebuilt_up_base", probe_cplx1(&rebuilt.up_base)));
            carried.push(("winv_rebuilt_dn", probe_cplx2(&rebuilt.dn)));
            carried.push(("winv_rebuilt_dn_base", probe_cplx1(&rebuilt.dn_base)));
            let fresh_st = build(&r_up, &r_dn)?;
            let fev = eval(&fresh_st, &r_up, &r_dn);
            let fresh = probe_snapshot_complex(&fresh_st, &fev);
            let mut oracle_st = build(&r_up, &r_dn)?;
            oracle_st.ginv = st.ginv.clone();
            oracle_st.det_winv =
                init_det_wdot_cache_c(gem, &oracle_st.det_state, &oracle_st.ginv.view());
            let oev = eval(&oracle_st, &r_up, &r_dn);
            let oracle = probe_snapshot_complex(&oracle_st, &oev);
            out.push(MeshChainRow {
                hop: k,
                r_up: r_up.clone(),
                r_dn: r_dn.clone(),
                carried,
                fresh,
                oracle,
            });
        }
        if k == n_hops {
            break;
        }
        let (is_up, e, newpos, oldpos) = match policy {
            ProbeHopPolicy::Categorical => interval_hop_complex(
                &ev.3,
                &ev.4,
                // Checked above: one uniform per hop (only this policy reads them).
                *uniforms.get(k).expect("one uniform per hop"),
                alat,
                kinetic_rt,
                &mut r_up,
                &mut r_dn,
            ),
            ProbeHopPolicy::Adversarial { floor } => {
                let mags: Vec<f64> = ev.3.iter().map(|x| x.norm()).collect();
                let pick = probe_adversarial_pick(&mags, floor, last)?;
                last = Some(pick / 6);
                probe_kinetic_hop(pick, alat, kinetic_rt, &mut r_up, &mut r_dn)
            }
        };
        st.advance_after_hop(
            gem,
            jastrow,
            j3c,
            j_matrix,
            &r_up.view(),
            &r_dn.view(),
            is_up,
            e,
            oldpos,
            newpos,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod det_winv_chain_tests {
    //! Candidate-table advance unit test of the projection interval.
    //!
    //! The production interval inverse and winv tables are judged along long categorical and
    //! adversarial chains against fresh states and independent XLA anchors by
    //! ``tests/test_rs_kernel/test_lrdmc_mesh_chain_rust.py``.
    use super::*;
    use numpy::ndarray::Array2;

    /// Deterministic LCG (no external rng crates in the test layer).
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG; top 53 bits -> [0, 1).
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal_ish(&mut self) -> f64 {
            // Irwin--Hall(6) approximation: mean 0, ~unit variance; plenty for
            // building well-conditioned random tables.
            (0..6).map(|_| self.next_f64()).sum::<f64>() - 3.0
        }
    }

    #[test]
    fn j3_candidate_table_advance_matches_fresh_rebuild() {
        let n_orb = 7usize;
        let n_up = 3usize;
        let n_dn = 2usize;
        let rows = 6usize;
        let mut rng = Lcg(0x5eed_2026_0828);

        let mut jm = Array2::<f64>::zeros((n_orb, n_orb + 1));
        for a in 0..n_orb {
            jm[[a, n_orb]] = 0.2 * rng.normal_ish();
            for b in 0..=a {
                let value = 0.15 * rng.normal_ish();
                jm[[a, b]] = value;
                jm[[b, a]] = value;
            }
        }
        let aos_up = Array2::from_shape_fn((n_orb, n_up), |_| rng.normal_ish());
        let aos_dn = Array2::from_shape_fn((n_orb, n_dn), |_| rng.normal_ish());
        let mut state = crate::jastrow_factor::init_j3_ratio_state_with_policy(
            &jm.view(),
            &aos_up.view(),
            &aos_dn.view(),
            false,
        );
        let build_wdot = |state: &crate::jastrow_factor::J3RatioState| {
            let build_spin = |is_up: bool| {
                let (aos, same, cross_same, cross_other) = if is_up {
                    (&state.aos_up, &state.w_up, &state.cross_up, &state.cross_dn)
                } else {
                    (&state.aos_dn, &state.w_dn, &state.cross_dn, &state.cross_up)
                };
                let mut q = Array2::<f64>::zeros((aos.ncols(), n_orb));
                let mut base = vec![0.0_f64; aos.ncols()];
                for a in 0..n_orb {
                    let common = jm[[a, n_orb]] + cross_same[a] + cross_other[a];
                    for e in 0..aos.ncols() {
                        q[[e, a]] = common - same[[a, e]];
                        base[e] += aos[[a, e]] * q[[e, a]];
                    }
                }
                (q, base)
            };
            let (up, up_base) = build_spin(true);
            let (dn, dn_base) = build_spin(false);
            J3WdotCache {
                up,
                up_base,
                dn,
                dn_base,
            }
        };
        let mut wdot = build_wdot(&state);
        let mut candidates =
            Array2::from_shape_fn((rows * (n_up + n_dn), n_orb), |_| rng.normal_ish());
        let mut table =
            fill_j3_candidate_deltas(&jm.view(), &state, &candidates.view(), n_up, rows);

        let moved_is_up = true;
        let moved_e = 1usize;
        let new_col: Vec<f64> = (0..n_orb).map(|_| rng.normal_ish()).collect();
        let d = crate::jastrow_factor::advance_j3_ratio_state_with_delta(
            &jm.view(),
            &mut state,
            moved_is_up,
            moved_e,
            &new_col,
        );
        let (current_up, current_dn) =
            advance_j3_wdot_cache(&mut wdot, &state, moved_is_up, moved_e, &d);
        let wdot_reference = build_wdot(&state);
        let wdot_worst = wdot
            .up
            .iter()
            .chain(wdot.dn.iter())
            .chain(wdot.up_base.iter())
            .chain(wdot.dn_base.iter())
            .zip(
                wdot_reference
                    .up
                    .iter()
                    .chain(wdot_reference.dn.iter())
                    .chain(wdot_reference.up_base.iter())
                    .chain(wdot_reference.dn_base.iter()),
            )
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            wdot_worst < crate::setting_consts::ATOL_RS_UNIT_REDUCTION,
            "incremental J3 wdot differs from fresh rebuild by {wdot_worst:.3e}"
        );
        let moved_global = moved_e;
        for row in moved_global * rows..(moved_global + 1) * rows {
            for a in 0..n_orb {
                candidates[[row, a]] = rng.normal_ish();
            }
        }
        let active = candidate_active_columns(&candidates.view(), rows);
        let packed = pack_candidate_groups(&candidates.view(), rows, active);
        advance_j3_candidate_deltas(
            &jm.view(),
            &state,
            n_up,
            rows,
            moved_is_up,
            moved_e,
            &d,
            &current_up,
            &current_dn,
            &packed,
            &mut table,
            None,
        );
        let reference =
            fill_j3_candidate_deltas(&jm.view(), &state, &candidates.view(), n_up, rows);
        let worst = table
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < crate::setting_consts::ATOL_RS_UNIT_REDUCTION,
            "incremental J3 candidate table differs from fresh rebuild by {worst:.3e}"
        );
    }
}

#[cfg(test)]
mod mol_candidate_exact_tests {
    //! IMPL-RS-GFMC-MOL-CANDIDATE-EXACT-001: the screened-pair-only exp stream
    //! of the molecular candidate AO batch and the sparse lane-preserving MO
    //! projection are BIT-IDENTICAL to all-pairs / dense reference kernels.
    use super::*;
    use crate::atomic_orbital::AoKind;
    use numpy::ndarray::Array2;

    /// All-(shell, point) exp-stream reference kernel.
    fn legacy_molecular_aos_candidate_batch(
        data: &crate::atomic_orbital::AosGtoData,
        positions: &[[f64; 3]],
    ) -> Array2<f64> {
        let nv = positions.len();
        let mut out = Array2::<f64>::zeros((nv, data.num_ao));
        let np = data.sh_neg_pz.len();
        let mut args = vec![0.0; np * nv];
        let mut exps = vec![0.0; np * nv];
        for (s, shell) in data.shells.iter().enumerate() {
            let base = data.sh_off[s] as usize;
            for (v, pos) in positions.iter().enumerate() {
                let x = pos[0] - shell.center[0];
                let y = pos[1] - shell.center[1];
                let z = pos[2] - shell.center[2];
                let r2 = x * x + y * y + z * z;
                for (q, &zeta) in shell.prim_z.iter().enumerate() {
                    args[(base + q) * nv + v] = -zeta * r2;
                }
            }
        }
        crate::atomic_orbital::exp_pd_into(&args, &mut exps);
        let mut radial_scale = vec![0.0; data.num_ao];
        if data.kind == AoKind::Cart {
            for shell in &data.shells {
                if let Some(q) = shell.prim_nc.iter().position(|&x| x != 0.0) {
                    let denom = shell.prim_nc[q];
                    for &ao in &shell.aos {
                        radial_scale[ao] = data.flat_nc[data.ao_prim_offset[ao] + q] / denom;
                    }
                }
            }
        }
        for (s, shell) in data.shells.iter().enumerate() {
            let base = data.sh_off[s] as usize;
            for (v, pos) in positions.iter().enumerate() {
                let x = pos[0] - shell.center[0];
                let y = pos[1] - shell.center[1];
                let z = pos[2] - shell.center[2];
                let r2 = x * x + y * y + z * z;
                if shell.pz_val * r2 > crate::atomic_orbital::LEPSBAS {
                    continue;
                }
                let mut radial = 0.0_f64;
                for q in 0..shell.prim_z.len() {
                    radial += shell.prim_nc[q] * exps[(base + q) * nv + v];
                }
                match data.kind {
                    AoKind::Cart => {
                        let xp = [1.0, x, x * x, x * x * x, x * x * x * x];
                        let yp = [1.0, y, y * y, y * y * y, y * y * y * y];
                        let zp = [1.0, z, z * z, z * z * z, z * z * z * z];
                        for &ao in &shell.aos {
                            let nx = data.poly_x[ao] as usize;
                            let ny = data.poly_y[ao] as usize;
                            let nz = data.poly_z[ao] as usize;
                            let angular = if nx < xp.len() && ny < yp.len() && nz < zp.len() {
                                xp[nx] * yp[ny] * zp[nz]
                            } else {
                                x.powi(nx as i32) * y.powi(ny as i32) * z.powi(nz as i32)
                            };
                            out[[v, data.kernel_to_home[ao]]] =
                                (radial * radial_scale[ao]) * angular;
                        }
                    }
                    AoKind::Sphe => {
                        for &ao in &shell.aos {
                            out[[v, data.kernel_to_home[ao]]] = radial
                                * crate::atomic_orbital::eval_angular_flat(data, ao, x, y, z, r2);
                        }
                    }
                }
            }
        }
        out
    }

    /// Mixed near / mid / far points: some shells screened at some points,
    /// none at others, and an all-screened far point.
    fn points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3],
            [0.7, -0.3, 0.2],
            [2.5, 1.1, -1.9],
            [6.0, -1.0, 2.0],
            [9.5, 0.5, -0.5],
            [13.0, 2.0, 1.0],
            [-11.0, 4.0, 3.0],
            [40.0, 0.0, 0.0],
        ]
    }

    fn bits(a: &Array2<f64>) -> Vec<u64> {
        a.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn candidate_aos_bitwise_legacy_all_pairs() {
        for (kind, label) in [(AoKind::Cart, "cart"), (AoKind::Sphe, "sphe")] {
            for high_l in [false, true] {
                let d = crate::atomic_orbital::tests::vgl_test_data_l(kind, 1, 0.0, high_l);
                for pts in [points(), far_points(), vec![[0.3, 0.2, -0.1]]] {
                    let new = molecular_aos_candidate_batch(&d, &pts);
                    let old = legacy_molecular_aos_candidate_batch(&d, &pts);
                    assert_eq!(
                        bits(&new),
                        bits(&old),
                        "{label} high_l={high_l} n={}",
                        pts.len()
                    );
                }
                // Non-vacuity: kept and fully screened points both occur.
                let new = molecular_aos_candidate_batch(&d, &points());
                let zero_rows = new
                    .rows()
                    .into_iter()
                    .filter(|r| r.iter().all(|&x| x == 0.0))
                    .count();
                assert!(
                    zero_rows >= 1 && zero_rows < new.nrows(),
                    "{label}: {zero_rows} zero rows"
                );
            }
        }
    }

    /// The GFMC force's shell-only mesh screen
    /// (`ao_val_grad_mesh_screened(.., prim_cut = false)`) is the AO model of this
    /// candidate batch, the kernel the GFMC e_L contracts at the ECP mesh points,
    /// and its gradient is that model's derivative. Independent reference: the
    /// candidate batch itself and its centred finite difference wherever the
    /// shell keep set is the same at both stencil points. Radii span the points
    /// where tight shells switch off and where a kept shell holds a primitive the
    /// VMC primitive screen (`prim_val`) would drop, so the two screens differ.
    #[test]
    fn force_shell_screen_is_candidate_batch_model() {
        use crate::atomic_orbital::LEPSBAS;
        // Values: relative reduction order only (exact zeros stay exact), so the
        // primitives the VMC screen would drop (each below 1e-12 in absolute AO
        // units) are resolved; the test requires that they are.
        let atol = crate::central_setting_f64("ATOL_BIT_EXACT");
        let rtol = crate::central_setting_f64("RTOL_REDUCTION_ORDER");
        let fd_atol = crate::central_setting_f64("ATOL_FD_VS_ANALYTIC_TIGHT");
        let fd_rtol = crate::central_setting_f64("RTOL_FD_VS_ANALYTIC_TIGHT");
        let h = 1.0e-5; // finite-difference input, not a tolerance
        let close = |a: f64, b: f64, at: f64, rt: f64| (a - b).abs() <= at + rt * b.abs();
        let dirs = [
            [0.6, 0.8, 0.0],
            [0.0, -0.6, 0.8],
            [0.48, 0.6, -0.64],
            [-0.8, 0.0, -0.6],
        ];
        let mut prim_screen_resolved = false;
        for high_l in [false, true] {
            let d = crate::atomic_orbital::tests::vgl_test_data_l(AoKind::Cart, 1, 0.0, high_l);
            let alive = |p: [f64; 3]| -> Vec<bool> {
                d.shells
                    .iter()
                    .map(|sh| {
                        let r2 = (p[0] - sh.center[0]).powi(2)
                            + (p[1] - sh.center[1]).powi(2)
                            + (p[2] - sh.center[2]).powi(2);
                        sh.pz_val * r2 <= LEPSBAS
                    })
                    .collect()
            };
            let mut pts = Vec::new();
            for &rad in &[0.15, 0.4, 0.8, 1.2, 1.7, 2.5, 3.5, 5.0, 7.0, 9.5] {
                for u in dirs {
                    pts.push([0.1 + rad * u[0], -0.2 + rad * u[1], 0.3 + rad * u[2]]);
                }
            }
            // Just beyond the radius where the VMC screen drops a primitive of a
            // shell that stays kept: the dropped primitive is largest there.
            for sh in &d.shells {
                for &pv in &sh.prim_val {
                    if pv > sh.pz_val {
                        let rad = (LEPSBAS / pv).sqrt() * 1.001; // geometry, not a tolerance
                        let u = dirs[0];
                        pts.push([
                            sh.center[0] + rad * u[0],
                            sh.center[1] + rad * u[1],
                            sh.center[2] + rad * u[2],
                        ]);
                    }
                }
            }
            let n = pts.len();
            let r = Array2::from_shape_vec((n, 3), pts.concat()).unwrap();
            let (val, grad) = crate::ao_cart_deriv::ao_val_grad_mesh_screened(&d, &r.view(), false)
                .into_dense(d.num_ao);
            let batch = molecular_aos_candidate_batch(&d, &pts);
            // The VMC screen's tables differ from this model beyond the bound.
            let (val_vmc, _) = crate::ao_cart_deriv::ao_val_grad_mesh_screened(&d, &r.view(), true)
                .into_dense(d.num_ao);
            prim_screen_resolved |= (0..n)
                .any(|v| (0..d.num_ao).any(|a| !close(val_vmc[[a, v]], batch[[v, a]], atol, rtol)));
            let (mut n_prim_only, mut n_shell_dead, mut n_fd) = (0usize, 0usize, 0usize);
            for (v, &p) in pts.iter().enumerate() {
                for sh in &d.shells {
                    let r2 = (p[0] - sh.center[0]).powi(2)
                        + (p[1] - sh.center[1]).powi(2)
                        + (p[2] - sh.center[2]).powi(2);
                    if sh.pz_val * r2 > LEPSBAS {
                        n_shell_dead += 1;
                    } else if sh.prim_val.iter().any(|&z| z * r2 > LEPSBAS) {
                        n_prim_only += 1;
                    }
                }
                for a in 0..d.num_ao {
                    assert!(
                        close(val[[a, v]], batch[[v, a]], atol, rtol),
                        "high_l={high_l} value ao={a} v={v}: {} vs batch {}",
                        val[[a, v]],
                        batch[[v, a]]
                    );
                }
                for k in 0..3 {
                    let (mut pp, mut pm) = (p, p);
                    pp[k] += h;
                    pm[k] -= h;
                    if alive(pp) != alive(p) || alive(pm) != alive(p) {
                        continue; // keep set changes inside the stencil
                    }
                    let (bp, bm) = (
                        molecular_aos_candidate_batch(&d, &[pp]),
                        molecular_aos_candidate_batch(&d, &[pm]),
                    );
                    for a in 0..d.num_ao {
                        let fd = (bp[[0, a]] - bm[[0, a]]) / (2.0 * h);
                        assert!(
                            close(fd, grad[k][[a, v]], fd_atol, fd_rtol),
                            "high_l={high_l} grad ao={a} v={v} d={k}: FD {fd} vs {}",
                            grad[k][[a, v]]
                        );
                    }
                    n_fd += 1;
                }
            }
            assert!(
                n_prim_only > 0 && n_shell_dead > 0 && n_fd > 0,
                "screen not exercised: {n_prim_only} {n_shell_dead} {n_fd}"
            );
        }
        assert!(
            prim_screen_resolved,
            "the comparison does not resolve the primitive screen"
        );
    }

    /// Points beyond the cut radius of the tight shells (but not of the
    /// diffuse ones): some AO columns are screened at EVERY point.
    fn far_points() -> Vec<[f64; 3]> {
        vec![
            [9.5, 0.5, -0.5],
            [10.0, 3.0, 1.0],
            [-9.0, 4.0, 3.0],
            [40.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn candidate_mos_sparse_bitwise_dense() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut screened_columns_seen = false;
        for (kind, label) in [(AoKind::Cart, "cart"), (AoKind::Sphe, "sphe")] {
            for high_l in [false, true] {
                let d = crate::atomic_orbital::tests::vgl_test_data_l(kind, 1, 0.0, high_l);
                let na = d.num_ao;
                assert_ne!(na % 16, 0, "fixture should exercise the scalar tail");
                let mut mixed = points();
                for k in 0..23 {
                    let t = k as f64 * 0.37;
                    mixed.push([1.3 * t.sin(), 0.9 * t.cos() - 0.2, 0.4 * t]);
                }
                // MO widths straddle the 8-MO tile (leftover replay), point
                // counts straddle the point block.
                for &nm in &[1_usize, 5, 8, 16, 37, 300] {
                    let c = Array2::from_shape_fn((nm, na), |_| next());
                    let ct = c.t().as_standard_layout().to_owned();
                    for pts in [mixed.clone(), far_points(), vec![[0.3, 0.2, -0.1]]] {
                        let tiled = molecular_mos_candidate_batch(&d, &c, Some(&ct), &pts);
                        let dense = molecular_mos_candidate_batch(&d, &c, None, &pts);
                        let reference = {
                            let aos = legacy_molecular_aos_candidate_batch(&d, &pts);
                            Array2::from_shape_fn((pts.len(), nm), |(s, m)| {
                                dot_product_fast(
                                    aos.row(s).as_slice().unwrap(),
                                    c.row(m).as_slice().unwrap(),
                                )
                            })
                        };
                        let tag =
                            format!("{label} high_l={high_l} na={na} nm={nm} n={}", pts.len());
                        assert_eq!(bits(&tiled), bits(&reference), "tiled {tag}");
                        assert_eq!(bits(&dense), bits(&reference), "dense {tag}");
                        // A wrong-shape transpose must fall back to the dense loop.
                        let bad = Array2::<f64>::zeros((na + 1, nm));
                        let fallback = molecular_mos_candidate_batch(&d, &c, Some(&bad), &pts);
                        assert_eq!(bits(&fallback), bits(&reference), "fallback {tag}");
                    }
                }
                let mut scratch = CandidateScratch::default();
                let _ = molecular_aos_candidate_batch_t(&d, &far_points(), &mut scratch);
                screened_columns_seen |= scratch.col_kept.iter().any(|&k| !k);
            }
        }
        assert!(
            screened_columns_seen,
            "no AO column was screened at every point"
        );
    }
}
