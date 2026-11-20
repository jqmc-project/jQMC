"""QMC module (GFMC).

Precision Zones:
    This module owns no Precision Zone; it consumes the zones of the
    wavefunction and Hamiltonian modules it calls.

See :mod:`jqmc._precision` for details.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import groupby
from logging import getLogger
from typing import Any, Literal, Self, TypeAlias

import jax
import numpy as np
import numpy.typing as npt
from jax import grad, jit, lax, vmap
from jax import numpy as jnp
from jax import typing as jnpt
from mpi4py import MPI

from ._build_info import log_build_info_once
from ._diff_mask import DiffMask, apply_diff_mask
from ._driver_helpers import (
    _generate_init_electron_configurations,
    _check_mpi4py_jax_distribution_consistency,
    _init_external_control_file,
    _initialize_jax_distributed,
    _log_memory_usage,
    _read_external_stop_flag,
    _remove_external_control_file,
    _require,
    _raise_collective_nonfinite_observables,
    _require_lrdmc_bra_num_mcmc_per_measurement,
    _validate_num_mcmc_per_measurement,
    derive_rank_rng_streams,
)
from ._route_probe import route_site
from ._precision import get_tolerance_min, precision_policy_snapshot
from . import _setting as _setting_module
from ._setting import (
    ANALYTIC_GRAD_ENABLED,
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    GFMC_ON_THE_FLY_BIN_BLOCKS,
    GFMC_ON_THE_FLY_COLLECT_STEPS,
    MPC_CUTOFF_HA,
    Nv_default,
    flag_rust_on_CPU,
    get_eps,
    walker_map,
)
from .coulomb_potential import (
    Ecp_candidate_mesh_streaming_state,
    _advance_ecp_candidate_mesh_streaming_state,
    _init_ecp_candidate_mesh_streaming_state,
    advance_ewald_ee_interval_state,
    compute_coulomb_potential,
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
    compute_ewald_ee_from_interval_state,
    compute_ewald_ee_interval_state,
)
from .determinant import (
    Det_ratio_streaming_state,
    Geminal_data,
    _compute_u_dn_move_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _det_state_inverse_probe_residual,
    _geminal_from_det_ratio_state,
    _sherman_morrison_ratio_and_inverse,
    compute_geminal_all_elements,
    compute_geminal_dn_one_column_elements,
    compute_geminal_up_one_row_elements,
)
from .hamiltonians import Hamiltonian_data
from .derivatives import bridge
from .jqmc_mcmc import (
    _NV_BASE_MESH,
    _extract_active_positions_from_ham,
    _geminal_inv_batched,
    _geminal_pinv_batched,
    _geminal_pinv_from_matrix,
    _haar_rotation_matrix_from_uniform,
    _refresh_observable_geminal_inv,
    _inject_active_positions,
    _leaf_to_fp64,
    _native_swct_forces,
)
from .jastrow_factor import (
    J2_candidate_mesh_state,
    J3_candidate_mesh_state,
    J_DTN_candidate_mesh_state,
    Jastrow_three_body_streaming_state,
    _compute_ratio_Jastrow_part_rank1_update,
    compute_Jastrow_part_ratio_zone,
    structure_positions_grad_Jastrow,
)
from .mpc import build_mpc_data, compute_mpc_correction_ee
from .swct import evaluate_swct_domega, evaluate_swct_omega, evaluate_swct_omega_dot_batched
from .wavefunction import (
    Discretized_kinetic_mesh_streaming_state,
    Kinetic_streaming_state,
    Wavefunction_data,
    _advance_discretized_kinetic_mesh_streaming_state,
    _advance_kinetic_energy_all_elements_streaming_state,
    _init_discretized_kinetic_mesh_streaming_state,
    _init_kinetic_energy_all_elements_streaming_state,
    _kinetic_energy_from_streaming_state,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_discretized_kinetic_energy_from_streaming_state,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
)

# create new logger level for development
DEVEL_LEVEL = 5
logging.addLevelName(DEVEL_LEVEL, "DEVEL")


# a new method to create a new logger
def _loglevel_devel(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(DEVEL_LEVEL):
        self._log(DEVEL_LEVEL, message, args, **kwargs)


logging.Logger.devel = _loglevel_devel

# set logger
logger = getLogger("jqmc").getChild(__name__)

# Once-per-process latch for the native J3 slow-fallback warning.
_NATIVE_J3_SLOW_FALLBACK_WARNED = False


def _warn_native_j3_slow_fallback_once() -> bool:
    """Log ONE warning per run when a native projection mesh took the J3 slow fallback.

    The native GFMC kinetic/ECP meshes stream the three-body Jastrow through a
    rank-1 ratio state and persistent candidate tables. A J3 the state cannot
    cover (no streaming state, e.g. a periodic MO-wrapped J3) silently falls back
    to re-evaluating every electron's J3 orbitals per mesh point, which is
    correct but O(10-100x) slower. ``jqmc_rs.gfmc_take_j3_slow_fallback_seen``
    reads and clears the always-on Rust witness; the first time it is set this
    logs a warning, later hits stay silent.

    Returns:
        bool: Whether the witness was set since the previous call.
    """
    global _NATIVE_J3_SLOW_FALLBACK_WARNED
    import jqmc_rs

    seen = bool(jqmc_rs.gfmc_take_j3_slow_fallback_seen())
    if seen and not _NATIVE_J3_SLOW_FALLBACK_WARNED:
        _NATIVE_J3_SLOW_FALLBACK_WARNED = True
        logger.warning(
            "Native GFMC projection: the three-body Jastrow took the full-value slow fallback "
            "(no streaming J3 ratio state for this J3 orbital type). Results are correct but the "
            "LRDMC projection can be one to two orders of magnitude slower than with a "
            "streamed J3."
        )
    return seen


# Single-flag derivative backend for GFMC HF force/stress, mirroring jqmc_mcmc.
# ``True`` (default; see jqmc._setting.ANALYTIC_GRAD_ENABLED) routes the Hellmann--Feynman
# force/stress through the verified analytic continuum-measurement kernels
# ``jqmc.derivatives.gfmc``; ``False`` keeps the autograd (``jax.grad`` /
# ``jax.jacrev`` of the novar=0 measurement local energy) closures. The SAME flag drives both production drivers
# (GFMC_t / GFMC_n) AND the ``_GFMC_{t,n}_debug`` oracles, so cross-backend parity tests compare
# analytic-vs-analytic (or autograd-vs-autograd) at strict tolerance in BOTH precision modes.
_ANALYTIC_GRAD_ENABLED = ANALYTIC_GRAD_ENABLED

NonLocalMove: TypeAlias = Literal["tmove", "dltmove"]
GFMCIntervalPotState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
GFMCProjectionTResult: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    GFMCIntervalPotState,
]
GFMCArrayEstimate: TypeAlias = tuple[npt.NDArray, npt.NDArray]
GFMCStressEstimate: TypeAlias = GFMCArrayEstimate | tuple[npt.NDArray, npt.NDArray, float, float]
NativeGFMCChunk: TypeAlias = dict[str, Any]
ActivePositions: TypeAlias = dict[str, jax.Array]
GFMCPositionGradient: TypeAlias = tuple[ActivePositions, jax.Array, jax.Array]
GFMCProjectionTLoopState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    GFMCIntervalPotState,
]
GFMCProjectionTStreamingState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Kinetic_streaming_state,
    Discretized_kinetic_mesh_streaming_state,
    Ecp_candidate_mesh_streaming_state | None,
    jax.Array,
    jax.Array,
    jax.Array,
    GFMCIntervalPotState,
    jax.Array,
]
GFMCProjectionNResult: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Kinetic_streaming_state,
    Discretized_kinetic_mesh_streaming_state,
    jax.Array,
]
GFMCProjectionNLoopState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    GFMCIntervalPotState,
    jax.Array,
    jax.Array,
]
GFMCProjectionNStreamingState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Kinetic_streaming_state,
    Discretized_kinetic_mesh_streaming_state,
    Ecp_candidate_mesh_streaming_state | None,
    GFMCIntervalPotState,
    jax.Array,
    jax.Array,
    jax.Array,
]
GFMCProjectionNStepResult: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    GFMCIntervalPotState,
]


def _walker_gfmc_kernel(kernel: Callable[..., object], non_local_move: NonLocalMove) -> Callable[..., object]:
    """``jit`` of a per-walker GFMC measurement-derivative ``kernel`` over the walker axis.

    Same call convention as ``jit(vmap(kernel, in_axes=(None, 0, 0, 0, None, 0)))`` --
    ``(h, r_up, r_dn, RT, alat, geminal_inv)`` with the Hamiltonian and ``alat`` unbatched --
    but the walkers are mapped by ``jqmc._setting.walker_map`` within
    ``WALKER_STREAM_BUDGET_BYTES``.
    """

    def batched(
        h: Hamiltonian_data,
        ru: jax.Array,
        rd: jax.Array,
        rt: jax.Array,
        al: jax.Array,
        gi: jax.Array | None,
    ) -> object:
        return walker_map(
            lambda ru_, rd_, rt_, gi_: kernel(h, ru_, rd_, rt_, al, non_local_move=non_local_move, geminal_inv=gi_),
            ru,
            rd,
            rt,
            gi,
        )

    return jit(batched)


def _rust_cpu_backend_active() -> bool:
    """Resolve the shared Rust-on-CPU flag independently of chunk eligibility."""
    env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if env_rust is not None:
        flag_rust = env_rust.strip().lower() not in ("0", "false", "no", "off", "")
    else:
        flag_rust = flag_rust_on_CPU
    return flag_rust and jax.default_backend() == "cpu"


def _has_any_jastrow(hamiltonian_data: Hamiltonian_data) -> bool:
    """True when the wavefunction carries any Jastrow factor (J1/J2/J3/DTN)."""
    jd = hamiltonian_data.wavefunction_data.jastrow_data
    return any(
        getattr(jd, _n, None) is not None
        for _n in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data", "jastrow_dtn_data")
    )


def _analytic_hf_enabled_for(hamiltonian_data: Hamiltonian_data, non_local_move: NonLocalMove) -> bool:
    """Return whether the analytic HF force/stress backend is enabled.

    Analytic whenever the module flag is set.  For ``dltmove`` the driver forms a DIFFERENT ``e_L``
    than ``tmove`` (the fixed-node-clamped non-local hop uses the determinant-only amplitude and
    re-weights only the accepted part by the Jastrow ratio ``exp(J_mesh - J_ref)``), so
    ``e_L^dlt = e_L^t + delta`` with ``delta = sum_g max(V^det_g, 0)(1 - jr_g)``.  The ``delta``
    force/stress is served in CLOSED FORM by
    ``gfmc.grad_e_L_t_wrt_{positions,strain}(..., non_local_move="dltmove")`` for BOTH molecular AND
    PERIODIC-Gamma ECP (including periodic ``dltmove + Jastrow``:
    the ``delta`` FORCE reuses the min-image-aware molecular kernel verbatim, and the
    ``delta`` STRESS uses the periodic-Pulay-stress ``gfmc._dltmove_delta_strain_pbc``; verified vs
    the driver jacrev/grad oracle to 7e-15 / 1.1e-13 on lithium PBC ECP).  ``jnone + dltmove`` ==
    ``tmove`` (``jr = 1``, no correction).

    No intersection remains un-ported: complex-twist (k != 0) Hamiltonians, DTN included, are
    served by the analytic kernels, and ``gfmc._assert_scope`` documents that boundary without
    excluding any currently constructible Hamiltonian.
    """
    if not _ANALYTIC_GRAD_ENABLED:
        return False
    return True


# --------------------------------------------------------------------------- #
# GFMC_t projection kernels live at module scope for reuse by both drivers;
# they capture nothing from ``GFMC_t.run()``. Importable so the test oracles call PRODUCTION code instead of a
# transcription. MPC never enters the projection (physical = no): e_L / w_L are
# always the plain Ewald Hamiltonian.
# --------------------------------------------------------------------------- #


@jit
def _generate_rotation_matrix_t(alpha: jnpt.ArrayLike, beta: jnpt.ArrayLike, gamma: jnpt.ArrayLike) -> jax.Array:
    # Precompute all necessary cosines and sines
    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)

    # Combine the rotations directly
    R = jnp.array(
        [
            [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
            [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
            [-sin_b, cos_b * sin_a, cos_a * cos_b],
        ],
        dtype=jnp.float64,
    )
    return R


def _split_lrdmc_rotation_keys(rotation_key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return independent kinetic/ECP mesh keys without perturbing the walk.

    The kinetic lattice uses ``rotation_key`` itself. The non-local ECP
    quadrature uses a deterministic child stream. Neither key is threaded back
    into the GFMC carry, so move keys and all later projection keys are the
    same as with a single rotation key.
    """
    return rotation_key, jax.random.split(rotation_key, 2)[1]


def _sample_lrdmc_rotation_matrix(rotation_key: jax.Array, random_discretized_mesh: bool) -> jax.Array:
    """Sample one Haar-distributed LRDMC mesh rotation without advancing key."""
    if random_discretized_mesh:
        uniforms = jax.random.uniform(rotation_key, shape=(3,), minval=0.0, maxval=1.0)
        return _haar_rotation_matrix_from_uniform(uniforms)
    return jnp.eye(3, dtype=jnp.float64)


# --------------------------------------------------------------------------- #
# Incremental interval potential state. XLA mirror of the Rust
# ``gfmc.rs::IntervalPotState``: within one projection interval each hop moves
# exactly ONE electron, so the diagonal potential pieces admit one-electron
# updates -- ee via the carried structure-factor state, per-electron ei rows
# (bare + alat-discretized) and per-electron local-ECP rows via a single-row
# recompute of the moved electron. The kinetic / ECP-nonlocal channels (and
# therefore b_L, the hop cdf, the escape-time sub-step count, and the walk
# itself) are untouched: only the diagonal e_L / w_L can drift, at the
# fp-reordering level, pinned by IMPL-GFMC-INCREMENTAL-TWIN-001.
#
# PBC-only (molecular Coulomb is measured-negligible and stays byte-identical
# on the reference path). The state is ALWAYS present in the projection carry
# (fixed pytree shape); whether it is USED is a static trace-time decision by
# ``_gfmc_pot_active`` -- molecular systems and the debug full-recompute route
# (env JQMC_GFMC_POT_FULL_RECOMPUTE=1, the twin oracle) carry inert leaves.
# --------------------------------------------------------------------------- #

# Trace-time liveness witness (anti-vacuity): incremented whenever a projection
# body is TRACED with the incremental potential branch active. A parity test
# that claims to exercise the incremental route asserts this advanced.
_GFMC_POT_TRACE_WITNESS: int = 0


def _gfmc_incremental_pot_enabled() -> bool:
    """Static (trace-time) switch for the incremental interval potential.

    Default ON. ``JQMC_GFMC_POT_FULL_RECOMPUTE=1`` selects the debug/oracle
    full-recompute twin route (IMPL-GFMC-INCREMENTAL-TWIN-001) -- identical
    walk, per-sub-step recomputed diagonal potentials.
    """
    env = os.environ.get("JQMC_GFMC_POT_FULL_RECOMPUTE")
    if env is not None and env.strip().lower() not in ("0", "false", "no", "off", ""):
        return False
    return True


def _gfmc_pot_active(hamiltonian_data: Hamiltonian_data) -> bool:
    """True when the incremental interval potential drives this Hamiltonian."""
    return hamiltonian_data.coulomb_potential_data.ewald_data is not None and _gfmc_incremental_pot_enabled()


def _gfmc_build_interval_pot_state(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    alat: float,
) -> GFMCIntervalPotState:
    """Build the interval potential carry for ONE walker (vmap-compatible).

    Returns the fixed-shape tuple ``(e_real_half (), s_re (N_g,), s_im (N_g,),
    ei_bare (N,), ei_disc (N,), ecp_local (N,), n_updates ())``. For inactive
    systems (molecular / full-recompute route) the leaves are inert zeros with
    ``N_g = 0`` so the carry stays uniform and free.
    """
    cp = hamiltonian_data.coulomb_potential_data
    n_e = r_up_carts.shape[0] + r_dn_carts.shape[0]
    if not _gfmc_pot_active(hamiltonian_data):
        z = jnp.zeros((), dtype=jnp.float64)
        zn = jnp.zeros((n_e,), dtype=jnp.float64)
        zg = jnp.zeros((0,), dtype=jnp.float64)
        return (z, zg, zg, zn, zn, zn, z)
    e_real_half, s_re, s_im = compute_ewald_ee_interval_state(r_up_carts, r_dn_carts, cp.ewald_data)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(
        coulomb_potential_data=cp, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )
    ei_bare = jnp.concatenate([ei_up, ei_dn])
    if cp.ecp_flag:
        # ECP composition uses the BARE ei rows only (parcutg=1); carry inert
        # zeros for the discretized rows (statically never consumed), matching
        # the Rust state which skips them for ECP.
        ei_disc = jnp.zeros((n_e,), dtype=jnp.float64)
        r_all = jnp.concatenate([r_up_carts, r_dn_carts], axis=0)
        _empty_dn = jnp.zeros((0, 3), dtype=jnp.float64)
        ecp_local = vmap(
            lambda r: compute_ecp_local_parts_all_pairs(
                coulomb_potential_data=cp,
                r_up_carts=r[None, :],
                r_dn_carts=_empty_dn,
            )
        )(r_all)
    else:
        ei_disc_up, ei_disc_dn = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_potential_data=cp, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts, alat=alat
        )
        ei_disc = jnp.concatenate([ei_disc_up, ei_disc_dn])
        ecp_local = jnp.zeros((n_e,), dtype=jnp.float64)
    return (
        jnp.asarray(e_real_half, dtype=jnp.float64),
        s_re,
        s_im,
        ei_bare,
        ei_disc,
        ecp_local,
        jnp.zeros((), dtype=jnp.float64),
    )


def _gfmc_advance_interval_pot_state(
    hamiltonian_data: Hamiltonian_data,
    pot_state: GFMCIntervalPotState,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    new_r_up_carts: jax.Array,
    new_r_dn_carts: jax.Array,
    has_up_move: jax.Array,
    up_index: jax.Array,
    has_dn_move: jax.Array,
    dn_index: jax.Array,
    alat: float,
) -> GFMCIntervalPotState:
    """Apply the one-electron hop to the interval potential carry.

    ``r_*_carts`` are the PRE-hop positions, ``new_r_*_carts`` the post-hop
    ones. When no electron moved (a suppressed GFMC_t hop after tau exhausted)
    ``old == new`` and every delta is exactly zero; only the liveness counter
    is guarded by the moved mask. Mirrors Rust
    ``IntervalPotState::advance_after_hop`` (same algorithm; vectorized-sum fp
    grouping differs at the documented cross-backend medium level).
    """
    cp = hamiltonian_data.coulomb_potential_data
    (e_real_half, s_re, s_im, ei_bare, ei_disc, ecp_local, n_updates) = pot_state
    n_up = r_up_carts.shape[0]
    moved = jnp.logical_or(has_up_move, has_dn_move)
    if n_up == 0:
        old_pos = r_dn_carts[dn_index]
        new_pos = new_r_dn_carts[dn_index]
        gi = n_up + dn_index
    elif r_dn_carts.shape[0] == 0:
        old_pos = r_up_carts[up_index]
        new_pos = new_r_up_carts[up_index]
        gi = up_index
    else:
        old_pos = jnp.where(has_up_move, r_up_carts[up_index], r_dn_carts[dn_index])
        new_pos = jnp.where(has_up_move, new_r_up_carts[up_index], new_r_dn_carts[dn_index])
        gi = jnp.where(has_up_move, up_index, n_up + dn_index)
    r_all_old = jnp.concatenate([r_up_carts, r_dn_carts], axis=0)
    e_real_half, s_re, s_im = advance_ewald_ee_interval_state(
        e_real_half, s_re, s_im, r_all_old, old_pos, new_pos, cp.ewald_data, moved_index=gi
    )
    # Row updates are guarded by the moved mask so an unmoved sub-step leaves
    # the carry BIT-identical (a same-position single-row recompute is not
    # guaranteed bit-equal to the batched build row under XLA vectorization).
    _empty_dn = jnp.zeros((0, 3), dtype=jnp.float64)
    ei_row_up, _ = compute_coulomb_potential_ei_dispatch(
        coulomb_potential_data=cp, r_up_carts=new_pos[None, :], r_dn_carts=_empty_dn
    )
    ei_bare = ei_bare.at[gi].set(jnp.where(moved, ei_row_up[0], ei_bare[gi]))
    if cp.ecp_flag:
        ecp_row = compute_ecp_local_parts_all_pairs(
            coulomb_potential_data=cp, r_up_carts=new_pos[None, :], r_dn_carts=_empty_dn
        )
        ecp_local = ecp_local.at[gi].set(jnp.where(moved, ecp_row, ecp_local[gi]))
    else:
        ei_disc_row_up, _ = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_potential_data=cp, r_up_carts=new_pos[None, :], r_dn_carts=_empty_dn, alat=alat
        )
        ei_disc = ei_disc.at[gi].set(jnp.where(moved, ei_disc_row_up[0], ei_disc[gi]))
    n_updates = n_updates + jnp.where(moved, 1.0, 0.0)
    return (e_real_half, s_re, s_im, ei_bare, ei_disc, ecp_local, n_updates)


# vmap axis name of the walker batch of the XLA projection drivers. The
# The per-hop inverse screen reduces its refresh decision over this axis
# (``lax.psum``) so that ONE walker-global ``lax.cond`` runs the O(N^3) refresh
# only on hops where some walker needs it (a batched predicate would turn the
# cond into a select that pays the refresh on every hop).
_GFMC_WALKER_AXIS = "gfmc_walkers"


def _streaming_mover(
    has_up_move: jax.Array | bool, up_index: jax.Array | int, dn_index: jax.Array | int, num_up: int, num_dn: int
) -> tuple[jax.Array | bool, jax.Array | int]:
    """Name the electron a GFMC sub-step hands to the streaming cache advances.

    The advances (kinetic grad/lap state, kinetic and ECP candidate meshes)
    rebuild the named electron's rows at its post-hop position, so the name
    must always be an existing electron. A suppressed GFMC_t hop (``tau_left``
    exhausted, positions unchanged) moves none: ``has_up`` is then False and
    ``dn_index`` is 0, which with an empty dn channel named global electron
    ``num_up`` -- one past the end (issue #784 B: the DTN candidate row of the
    last up electron was rebuilt with its own pair, corrupting the cache read
    by the masked sub-steps' measured e_L). The spin is chosen statically
    when one channel is empty; a suppressed hop then re-evaluates an existing
    electron at its unchanged position, which every advance treats as a
    numerical no-op. Two-spin systems are unchanged.

    Args:
        has_up_move: Whether an up electron moved this sub-step.
        up_index: Up-channel index of the moved (or argmax-0) electron.
        dn_index: Dn-channel index of the moved (or argmax-0) electron.
        num_up: Static number of up electrons.
        num_dn: Static number of dn electrons.

    Returns:
        ``(moved_spin_is_up, moved_index)`` naming an existing electron.
    """
    if num_dn == 0:
        return True, up_index
    if num_up == 0:
        return False, dn_index
    return has_up_move, jnp.where(has_up_move, up_index, dn_index)


@jit
def _seed_gfmc_t_geminal_inverse(
    geminal_data: Geminal_data,
    r_up_batch: jax.Array,
    r_dn_batch: jax.Array,
) -> jax.Array:
    """Reuse one executable for the seed and post-branching inverse rebuild."""
    geminals = vmap(compute_geminal_all_elements, in_axes=(None, 0, 0))(geminal_data, r_up_batch, r_dn_batch)
    return _geminal_pinv_batched(geminals)


def _interval_inverse_screen(
    geminal_data: Geminal_data,
    det_state: Det_ratio_streaming_state,
    A_new_inv: jax.Array,
    dependents: Any = None,
    rebuild_dependents: Callable[[jax.Array], Any] | None = None,
    axis_name: str | None = None,
) -> tuple[jax.Array, Any, jax.Array]:
    """Per-hop refresh rule of the carried interval inverse (production).

    After a hop's Sherman--Morrison update, the carried ``G^-1`` of ONE walker
    is screened branch-free by the O(N^2) carried-inverse probe residual against the
    geminal the post-hop streaming caches imply
    (:func:`jqmc.determinant._det_state_inverse_probe_residual`). A walker whose
    residual exceeds ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD`` (the existing
    internal constant; no user knob, no other trigger) gets ``G^-1`` rebuilt
    from scratch -- the partial-pivoting LU inverse of that geminal, whose cache
    columns are fresh orbital evaluations at the current positions (the same
    factorization class as the Rust refresh, ``determinant::inverse``, so both
    backends leave a residual of the same size and take the same decisions on
    the same chain; an SVD pseudo-inverse leaves a residual of order
    eps*cond(G) above the threshold and would refresh on every hop once cond(G)
    exceeds about 1e7) -- and, through ``rebuild_dependents(A_refreshed)``,
    every carried table derived from the old inverse.

    The refresh runs inside ONE ``lax.cond`` taken iff any walker exceeds the
    threshold. Under the walker ``vmap`` the decision is reduced over
    ``axis_name`` with ``lax.psum``, so the cond predicate is walker-global (not
    batched) and stays a real conditional: hops on which no walker needs a
    refresh pay no O(N^3) work. Inside the branch only the flagged walkers are
    replaced (masks); the others are returned bit-identical. ``axis_name=None``
    is the single-walker form.

    Returns:
        ``(A_inv, dependents, refreshed)``; ``refreshed`` is this walker's bool.
    """
    residual = _det_state_inverse_probe_residual(geminal_data, det_state, A_new_inv)
    refreshed = residual > _setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    if axis_name is None:
        any_refreshed = refreshed
    else:
        any_refreshed = lax.psum(refreshed.astype(jnp.int32), axis_name) > 0

    def _refresh(operand: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
        A_inv, deps = operand
        fresh = jnp.linalg.inv(_geminal_from_det_ratio_state(geminal_data, det_state)).astype(A_inv.dtype)
        A_inv = jnp.where(refreshed, fresh, A_inv)
        if rebuild_dependents is not None:
            deps = jax.tree_util.tree_map(lambda new, old: jnp.where(refreshed, new, old), rebuild_dependents(A_inv), deps)
        return A_inv, deps

    A_out, deps_out = lax.cond(any_refreshed, _refresh, lambda operand: operand, (A_new_inv, dependents))
    return A_out, deps_out, refreshed


def _projection_t_core(
    projection_counter: int,
    tau_left: float,
    w_L: float,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    A_old_inv: jnpt.ArrayLike,
    jax_PRNG_key: jnpt.ArrayLike,
    diagonal_kinetic_continuum_elements_up: jnpt.ArrayLike,
    diagonal_kinetic_continuum_elements_dn: jnpt.ArrayLike,
    j3_state: Jastrow_three_body_streaming_state | None,
    interval_RT: jnpt.ArrayLike,
    interval_ecp_RT: jnpt.ArrayLike,
    random_discretized_mesh: bool,
    non_local_move: NonLocalMove,
    alat: float,
    hamiltonian_data: Hamiltonian_data,
    det_ratio_state: Det_ratio_streaming_state | None = None,
    kinetic_mesh_state: Discretized_kinetic_mesh_streaming_state | None = None,
    ecp_mesh_state: Ecp_candidate_mesh_streaming_state | None = None,
    pot_state: GFMCIntervalPotState | None = None,
) -> GFMCProjectionTResult:
    """Single GFMC_t projection step, parameterized by per-electron continuum kinetic energy.

    The body shared by the non-streaming and streaming wrappers. The caller is responsible
    for supplying the continuum per-electron kinetic energies (non-streaming:
    ``compute_kinetic_energy_all_elements_fast_update``; streaming:
    ``_kinetic_energy_from_streaming_state``) and ``j3_state`` (non-streaming:
    None; streaming: the maintained J3 sub-state).

    Returns the same tuple as the non-streaming ``_projection_t`` plus three
    extra fields (``has_up_move``, ``up_index``, ``dn_index``) that the
    streaming wrapper uses to drive ``_advance_kinetic_energy_*``.
    """
    # Key schedule: one rotation-key split per substep, with the first rotation
    # held fixed for the whole projection interval.  Only the mesh-rotation
    # cadence differs from a per-substep rotation; move/escape keys are
    # byte-equal to it.
    first_substep = projection_counter == 0

    # projection counter
    projection_counter = lax.cond(
        tau_left > 0.0,
        lambda pc: pc + 1,
        lambda pc: pc,
        projection_counter,
    )

    # coulomb regularization
    # compute diagonal elements, kinetic part
    diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

    # continuum kinetic energy is supplied by the wrapper (non-streaming:
    # ``compute_kinetic_energy_all_elements_fast_update``; streaming:
    # ``_kinetic_energy_from_streaming_state``).

    # generate a random rotation matrix
    jax_PRNG_key, rotation_key = jax.random.split(jax_PRNG_key)
    kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rotation_key)
    candidate_RT = _sample_lrdmc_rotation_matrix(kinetic_key, random_discretized_mesh).T
    candidate_ecp_RT = _sample_lrdmc_rotation_matrix(ecp_key, random_discretized_mesh).T
    # jnp.where instead of lax.cond: see the miscompile note at the kinetic
    # dispatch below.
    RT = jnp.where(first_substep, candidate_RT, interval_RT)
    ecp_RT = jnp.where(first_substep, candidate_ecp_RT, interval_ecp_RT)

    # compute discretized kinetic energy and mesh (with a random rotation)
    if kinetic_mesh_state is None:
        # By design for GFMC_t's first sub-step: this call is what BUILDS the
        # state the rest of the interval reuses, so it cannot itself be cached.
        with route_site("substep0"):
            mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                compute_discretized_kinetic_energy_fast_update(
                    alat=alat,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    A_old_inv=A_old_inv,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=RT,
                    j3_state=j3_state,
                    det_ratio_state=det_ratio_state,
                )
            )
    else:
        # Streaming path: substep 0 runs UNROLLED outside the projection
        # ``lax.while_loop`` through the ``kinetic_mesh_state is None`` branch
        # above (full kernels + post-hop mesh-state init), so the
        # cached kernel is ALWAYS valid here.  A traced
        # ``first_substep ? full : cached`` select would execute BOTH kernels
        # on every substep under the walker vmap (batched predicate), i.e. a
        # redundant full 6N-mesh evaluation per substep.
        # The cached kernel on a fresh state at the same configuration agrees
        # with the full kernel to ~1e-15 (reduction order).
        mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
            compute_discretized_kinetic_energy_from_streaming_state(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                A_old_inv=A_old_inv,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                mesh_state=kinetic_mesh_state,
                j3_state=j3_state,
                det_ratio_state=det_ratio_state,
            )
        )
    # spin-filp
    elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
    non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
    diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
    # regularizations
    elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
    sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
    non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1)
    sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
    non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
        non_diagonal_kinetic_part_elements, [len(r_up_carts)]
    )

    # Incremental interval potential: a static trace-time route
    # decision. Active => the diagonal ee / per-electron ei rows come from the
    # carried state (advanced per hop below); inactive (molecular, or the
    # debug full-recompute twin via JQMC_GFMC_POT_FULL_RECOMPUTE=1) => the
    # reference per-sub-step dispatch calls. b_L / hop cdf never touch the
    # state: the walk is route-invariant.
    _use_pot = pot_state is not None and _gfmc_pot_active(hamiltonian_data)
    if _use_pot:
        global _GFMC_POT_TRACE_WITNESS
        _GFMC_POT_TRACE_WITNESS += 1
        _n_up_static = r_up_carts.shape[0]
        _n_e_static = _n_up_static + r_dn_carts.shape[0]
        (_pot_e_real_half, _pot_s_re, _pot_s_im, _pot_ei_bare, _pot_ei_disc, _pot_ecp_local, _) = pot_state
        diagonal_bare_coulomb_part_el_el = compute_ewald_ee_from_interval_state(
            _pot_e_real_half,
            _pot_s_re,
            _pot_s_im,
            _n_e_static,
            hamiltonian_data.coulomb_potential_data.ewald_data,
        )
        diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data
        )
        diagonal_bare_coulomb_part_el_ion_elements_up = _pot_ei_bare[:_n_up_static]
        diagonal_bare_coulomb_part_el_ion_elements_dn = _pot_ei_bare[_n_up_static:]
        diagonal_bare_coulomb_part_el_ion_discretized_elements_up = _pot_ei_disc[:_n_up_static]
        diagonal_bare_coulomb_part_el_ion_discretized_elements_dn = _pot_ei_disc[_n_up_static:]
    else:
        # compute diagonal elements, el-el
        diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        # compute diagonal elements, ion-ion
        diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data
        )

        # compute diagonal elements, el-ion
        diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
            compute_coulomb_potential_ei_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
        )

        # compute diagonal elements, el-ion, discretized
        (
            diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
            diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
        ) = compute_coulomb_potential_ei_discretized_dispatch(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            alat=alat,
        )

    # compose discretized el-ion potentials
    diagonal_bare_coulomb_part_el_ion_zv_up = (
        diagonal_bare_coulomb_part_el_ion_elements_up
        + diagonal_kinetic_continuum_elements_up
        - non_diagonal_kinetic_part_elements_up
    )
    # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
    # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
    else:
        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
    diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
        diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
    )
    diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
        sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
    )

    # compose discretized el-ion potentials
    diagonal_bare_coulomb_part_el_ion_zv_dn = (
        diagonal_bare_coulomb_part_el_ion_elements_dn
        + diagonal_kinetic_continuum_elements_dn
        - non_diagonal_kinetic_part_elements_dn
    )
    # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
    # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
    else:
        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
    diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
        diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
    )
    diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
        sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
    )

    # final bare coulomb part
    discretized_diagonal_bare_coulomb_part = (
        diagonal_bare_coulomb_part_el_el
        + diagonal_bare_coulomb_part_ion_ion
        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
    )

    # if-else for all-electron, ECP with tmove, and ECP with dltmove
    # with ECP
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
        # ecp local (carried per-electron rows on the incremental route;
        # electron-major sum -- an fp reordering of the reference channel-major
        # total, bounded by the twin obligation)
        if _use_pot:
            diagonal_ecp_local_part = jnp.sum(_pot_ecp_local)
        else:
            diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

        def _ecp_nonlocal(
            flag_determinant_only: bool,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            def _eval(
                candidate_state: Ecp_candidate_mesh_streaming_state | None,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                return compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    flag_determinant_only=flag_determinant_only,
                    A_old_inv=A_old_inv,
                    RT=ecp_RT,
                    j3_state=j3_state,
                    det_ratio_state=det_ratio_state,
                    candidate_mesh_state=candidate_state,
                )

            if ecp_mesh_state is None:
                return _eval(None)
            # Streaming path: the ECP mesh state is initialized outside the
            # projection loop and advanced per hop, so the cached kernel is
            # always valid -- see the kinetic dispatch note above.
            return _eval(ecp_mesh_state)

        if non_local_move == "tmove":
            # ecp non-local (t-move)
            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = _ecp_nonlocal(False)

            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
            non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

        elif non_local_move == "dltmove":
            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = _ecp_nonlocal(True)

            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

            def _dlt_jastrow(
                j3_candidate_state: J3_candidate_mesh_state | None,
                j2_candidate_state: J2_candidate_mesh_state | None,
                j_dtn_candidate_state: J_DTN_candidate_mesh_state | None,
            ) -> jax.Array:
                # Only carried meshes have a projection-cache obligation. The
                # first substep still builds its state through the full kernel.
                with route_site("projection" if ecp_mesh_state is not None else "substep0"):
                    return _compute_ratio_Jastrow_part_rank1_update(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        old_r_up_carts=r_up_carts,
                        old_r_dn_carts=r_dn_carts,
                        new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                        new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        j3_state=j3_state,
                        j1_candidate_state=(ecp_mesh_state.j1_candidate_state if ecp_mesh_state is not None else None),
                        j3_candidate_state=j3_candidate_state,
                        j2_candidate_state=j2_candidate_state,
                        j_dtn_candidate_state=j_dtn_candidate_state,
                        _route_mesh="ecp" if ecp_mesh_state is not None else None,
                    )

            if ecp_mesh_state is None:
                Jastrow_ratio = _dlt_jastrow(None, None, None)
            else:
                # Streaming path: cached candidate states always valid -- see
                # the kinetic dispatch note above.
                Jastrow_ratio = _dlt_jastrow(
                    ecp_mesh_state.j3_candidate_state,
                    ecp_mesh_state.j2_candidate_state,
                    ecp_mesh_state.j_dtn_candidate_state,
                )
            V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
            non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

        else:
            logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
            raise NotImplementedError

        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
        e_L = (
            diagonal_kinetic_part
            + discretized_diagonal_bare_coulomb_part
            + diagonal_ecp_local_part
            + diagonal_kinetic_part_SP
            + diagonal_ecp_part_SP
            + non_diagonal_sum_hamiltonian
        )

        # Measurement local energy (novar=0 split, cf. TurboRVB enertrue):
        # the CONTINUUM local energy assembled from the per-electron continuum
        # kinetic and the bare el-ion potential -- no lattice hop terms, no
        # sign-flip max clip. By the zero-variance identity (Appendix D, D7)
        # this equals the clipped lattice e_L on every non-clipped
        # configuration; on clipped configurations it is the unbiased bare
        # estimator while ``e_L`` above keeps feeding the propagation weight.
        # The ECP non-local (t-move localized) part is part of the continuum
        # local energy and is kept as-is (TurboRVB's ``vpotoff``).
        e_L_meas = (
            diagonal_bare_coulomb_part_el_el
            + diagonal_bare_coulomb_part_ion_ion
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
            + diagonal_ecp_local_part
            + diagonal_ecp_part_SP
            + non_diagonal_sum_hamiltonian_ecp
        )

        p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
        # Under complex Bloch fixed-phase, take Re for the
        # categorical sampling so the cdf passed to searchsorted is
        # real-valued. Bit-identical at Gamma where p_list is real.
        p_list = jnp.real(p_list)
        non_diagonal_move_probabilities = p_list / p_list.sum()
        non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
            [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
        )
        non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
            [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
        )

    # with all electrons
    else:
        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
        e_L = (
            diagonal_kinetic_part
            + discretized_diagonal_bare_coulomb_part
            + diagonal_kinetic_part_SP
            + non_diagonal_sum_hamiltonian
        )

        # Measurement local energy (novar=0 split): continuum e_L from the
        # per-electron continuum kinetic + BARE el-ion potential. See the ECP
        # branch above for the rationale.
        e_L_meas = (
            diagonal_bare_coulomb_part_el_el
            + diagonal_bare_coulomb_part_ion_ion
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
        )

        p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
        # Under complex Bloch fixed-phase, take Re for the
        # categorical sampling so the cdf passed to searchsorted is
        # real-valued. Bit-identical at Gamma where p_list is real.
        p_list = jnp.real(p_list)
        non_diagonal_move_probabilities = p_list / p_list.sum()
        non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
        non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

    # compute the time the walker remaining in the same configuration
    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
    xi = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
    # Complex Bloch uses the classical fixed-phase approximation:
    # use Re(non_diag) for the escape rate and Re(e_L) for the
    # branching weight so w_L stays real-positive. Im(e_L) enters as
    # a per-walker phase that is dropped here because branching weights must
    # remain real and positive.
    # Bit-identical at Gamma where both are real.
    tau_update = jnp.minimum(tau_left, jnp.log(1 - xi) / jnp.real(non_diagonal_sum_hamiltonian))

    # update weight
    w_L = w_L * jnp.exp(-tau_update * jnp.real(e_L))

    # update tau_left
    tau_left = tau_left - tau_update

    # electron position update
    # random choice
    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
    cdf = jnp.cumsum(non_diagonal_move_probabilities)
    random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
    k = jnp.searchsorted(cdf, random_value)
    proposed_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
    proposed_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

    new_r_up_carts = jnp.where(tau_left <= 0.0, r_up_carts, proposed_r_up_carts)  # '=' is very important!!!
    new_r_dn_carts = jnp.where(tau_left <= 0.0, r_dn_carts, proposed_r_dn_carts)  # '=' is very important!!!

    # update inverse via Sherman-Morrison rank-1 update
    num_up_electrons = r_up_carts.shape[0]
    num_dn_electrons = r_dn_carts.shape[0]

    if num_up_electrons == 0:
        has_up_move = False
        up_index = 0
    else:
        up_diff = jnp.any(r_up_carts != new_r_up_carts, axis=1)
        has_up_move = jnp.any(up_diff)
        up_index = jnp.argmax(up_diff)

    if num_dn_electrons == 0:
        has_dn_move = False
        dn_index = 0
    else:
        dn_diff = jnp.any(r_dn_carts != new_r_dn_carts, axis=1)
        has_dn_move = jnp.any(dn_diff)
        dn_index = jnp.argmax(dn_diff)

    def _update_inv_up_t(_: None) -> jax.Array:
        # Streaming path uses cached AO + paired_dn from
        # ``det_ratio_state``; non-streaming path falls back to the
        # twice-called row helper.
        if det_ratio_state is None:
            v = (
                compute_geminal_up_one_row_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_cart=jnp.reshape(new_r_up_carts[up_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )
                - compute_geminal_up_one_row_elements(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    r_up_cart=jnp.reshape(r_up_carts[up_index], (1, 3)),
                    r_dn_carts=r_dn_carts,
                )
            )[:, None]
        else:
            v = _compute_v_up_move_from_det_ratio_state(
                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                state=det_ratio_state,
                moved_index=up_index,
                r_up_carts_proposed=new_r_up_carts,
            )[:, None]
        u = jax.nn.one_hot(up_index, num_up_electrons)[:, None]
        # The SM rank-1 update returns A_old_inv's dtype, so the result agrees
        # with the no-update (A_old_inv) selection arm, and a complex Bloch
        # A_old_inv keeps its imaginary part. All jnp.where selection arms
        # return the same dtype (A_old_inv.dtype) keeping the select
        # output invariant intact.
        _, A_new_inv = _sherman_morrison_ratio_and_inverse(A_old_inv, u, v, moved_index=up_index, moved_is_up=True)
        return A_new_inv

    def _no_update_t(_: None) -> jax.Array:
        return A_old_inv

    if num_dn_electrons == 0:
        _update_inv_dn_t = _no_update_t
    else:

        def _update_inv_dn_t(_: None) -> jax.Array:
            if det_ratio_state is None:
                u = (
                    compute_geminal_dn_one_column_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_cart=jnp.reshape(new_r_dn_carts[dn_index], (1, 3)),
                    )
                    - compute_geminal_dn_one_column_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_cart=jnp.reshape(r_dn_carts[dn_index], (1, 3)),
                    )
                )[:, None]
            else:
                u = _compute_u_dn_move_from_det_ratio_state(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    state=det_ratio_state,
                    moved_index=dn_index,
                    r_dn_carts_proposed=new_r_dn_carts,
                )[:, None]
            v = jax.nn.one_hot(dn_index, num_up_electrons)[:, None]
            # See note in _update_inv_up_t: consumer-zone explicit cast.
            # Preserve A_old_inv's complex Bloch dtype so its imaginary part
            # survives the SM rank-1 update. All jnp.where selection arms
            # return the same dtype (A_old_inv.dtype) keeping the select
            # output invariant intact.
            _, A_new_inv = _sherman_morrison_ratio_and_inverse(A_old_inv, u, v, moved_index=dn_index, moved_is_up=False)
            return A_new_inv

    if num_up_electrons == 0:
        A_new_inv = A_old_inv
    else:
        # SM-carry selection (GFMC_t). A nested
        # ``lax.cond(has_up_move, up, cond(has_dn_move, dn, no_update))``
        # mis-selects its branch when the body runs inside GFMC_t's
        # ``lax.while_loop`` (the predicate ``has_up_move`` derives from the
        # ``jnp.where(tau_left <= 0.0, ...)``-guarded ``new_r``, which is
        # itself a while-loop carry). The mis-selection returns an
        # ``A_new_inv`` that is NOT ``G(new_r)^-1`` (measured
        # ``max|G . A_new - I| ~ 0.6-0.8`` from the first productive
        # substep) even though ``u``/``v``/``A_old_inv`` are all correct
        # (an inline SM with the identical operands is exact). The
        # eager-vmapped body (warm-up) and GFMC_n (fixed-count
        # ``fori_loop``, unconditional positions) are not affected; GFMC_n
        # keeps the nested cond (verified fp-exact).
        #
        # A flat ``jnp.where`` over unconditionally-evaluated branch
        # results is exactly what a vmapped ``lax.cond`` lowers to (both
        # branches already execute under vmap -- no extra cost), but it
        # is immune to the nested-cond mis-selection. Both branches are
        # numerically safe on the non-taken lane: an unmoved spin yields
        # ``v``/``u == 0`` -> ``det_ratio == 1`` -> the update reduces to
        # ``A_old_inv`` (no NaN).
        A_up_updated = _update_inv_up_t(None)
        A_dn_updated = _update_inv_dn_t(None)
        A_new_inv = jnp.where(
            has_up_move,
            A_up_updated,
            jnp.where(has_dn_move, A_dn_updated, A_old_inv),
        )

    # Walker carry stays unwrapped (in R^3). Wrapping after the SM
    # update would put the running ``A_new_inv`` out of phase with the
    # carried walker positions at twist k != 0, because the complex-
    # Bloch ``compute_AOs`` is wrap-equivariant
    # (``compute_AOs(r + T) = e^{i k . T} compute_AOs(r)``), so wrapping
    # the carry would change the wavefunction value by ``exp(i k . T)``
    # while the SM-evolved inverse was calibrated for the unwrapped
    # position. The mismatch propagates and amplifies through subsequent
    # SM updates. AO evaluation itself is unaffected: ``compute_AOs``
    # wraps internally and re-attaches the Bloch phase via
    # ``_restore_phase`` so the result satisfies the Bloch BC at the
    # input (unwrapped) ``r``.

    # Advance the incremental interval potential to the post-hop config
    # (one-electron delta; exactly a no-op when the hop was suppressed).
    if _use_pot:
        pot_state_new = _gfmc_advance_interval_pot_state(
            hamiltonian_data,
            pot_state,
            r_up_carts,
            r_dn_carts,
            new_r_up_carts,
            new_r_dn_carts,
            has_up_move,
            up_index,
            has_dn_move,
            dn_index,
            alat,
        )
    else:
        pot_state_new = pot_state

    # Slot 0 carries the MEASUREMENT local energy (novar=0 split): every
    # downstream consumer of this slot (e_L_list -> stored_e_L -> get_E /
    # checkpoint observables) is an estimator; the propagation weight has
    # already been applied above with the clipped lattice ``e_L``.
    return (
        e_L_meas,
        projection_counter,
        tau_left,
        w_L,
        new_r_up_carts,
        new_r_dn_carts,
        A_new_inv,
        jax_PRNG_key,
        RT,
        ecp_RT,
        has_up_move,
        up_index,
        dn_index,
        pot_state_new,
    )


@jit
def _sprime_data_from_projection_state(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    A_old_inv: jax.Array,
    kinetic_state: Kinetic_streaming_state,
    kinetic_mesh_state: Discretized_kinetic_mesh_streaming_state,
    alat: float,
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct propagation S-prime data from maintained projection state.

    The force/stress measurement path does not consume this data. This pure
    twin serves the R05 propagation-membership parity gate: it evaluates
    final 6N ratios from maintained tabpip/winv tables, while
    ``derivatives.gfmc._sprime_data`` recomputes the same propagation oracle.
    """
    tc_up, tc_dn = _kinetic_energy_from_streaming_state(kinetic_state)
    _, _, elements = compute_discretized_kinetic_energy_from_streaming_state(
        alat=alat,
        wavefunction_data=hamiltonian_data.wavefunction_data,
        A_old_inv=A_old_inv,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        mesh_state=kinetic_mesh_state,
        j3_state=kinetic_state.j3_state,
        det_ratio_state=kinetic_state.det_state,
    )
    elem_all = elements.reshape(-1, 6)
    sign_flip = jnp.any(jnp.real(elem_all) >= 0, axis=1)
    ndk = jnp.sum(elem_all + 1.0 / (2.0 * alat**2), axis=1)
    ei_up, ei_dn = compute_coulomb_potential_ei_dispatch(hamiltonian_data.coulomb_potential_data, r_up_carts, r_dn_carts)
    eid_up, eid_dn = compute_coulomb_potential_ei_discretized_dispatch(
        hamiltonian_data.coulomb_potential_data, r_up_carts, r_dn_carts, alat
    )
    ei = jnp.concatenate([jnp.asarray(ei_up), jnp.asarray(ei_dn)])
    eid = jnp.concatenate([jnp.asarray(eid_up), jnp.asarray(eid_dn)])
    tc = jnp.concatenate([jnp.asarray(tc_up), jnp.asarray(tc_dn)])
    zv = jnp.real(ei + tc - ndk)
    ei_used = ei if hamiltonian_data.coulomb_potential_data.ecp_flag else eid
    w = (sign_flip & (jnp.real(ei_used) > zv)).astype(jnp.float64)
    return w, elem_all


# One process-wide callable: rebuilding ``jit(vmap(...))`` inside every driver
# invocation gives JAX a new Python function identity and defeats the compile
# cache even though the computation is identical for GFMC_t and GFMC_n.


@partial(jit, static_argnums=(4,))
def _compute_V_elements_t(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    non_local_move: NonLocalMove,
    alat: float,
    ecp_RT: jnpt.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute V elements (non-fast-update path for gradient computation).

    ``RT`` is the kinetic lattice orientation.  ``ecp_RT`` is the independent
    non-local ECP quadrature orientation; the default preserves the one-mesh
    low-level ABI used by derivative oracles that only need measurement e_L.
    """
    if ecp_RT is None:
        ecp_RT = RT
    # compute diagonal elements, kinetic part
    diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

    # compute continuum kinetic energy
    diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = compute_kinetic_energy_all_elements(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # compute discretized kinetic energy and mesh (with a random rotation)
    _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
        alat=alat,
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
    )

    # spin-flip
    elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
    non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
    diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
    # regularizations
    elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
    sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
    non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1)
    sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
    non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
        non_diagonal_kinetic_part_elements, [len(r_up_carts)]
    )

    # compute diagonal elements, el-el
    # MPC (choice A): MPC e-e enters the Hamiltonian so the force
    # grad(_compute_local_energy_t) is MPC-corrected and consistent with w_L.
    diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    # compute diagonal elements, ion-ion
    diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data
    )

    # compute diagonal elements, el-ion
    diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
        compute_coulomb_potential_ei_dispatch(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
    )

    # compute diagonal elements, el-ion, discretized
    (
        diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
        diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
    ) = compute_coulomb_potential_ei_discretized_dispatch(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        alat=alat,
    )

    # compose discretized el-ion potentials
    diagonal_bare_coulomb_part_el_ion_zv_up = (
        diagonal_bare_coulomb_part_el_ion_elements_up
        + diagonal_kinetic_continuum_elements_up
        - non_diagonal_kinetic_part_elements_up
    )
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
    else:
        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
    diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
        diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
    )
    diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
        sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
    )

    # compose discretized el-ion potentials
    diagonal_bare_coulomb_part_el_ion_zv_dn = (
        diagonal_bare_coulomb_part_el_ion_elements_dn
        + diagonal_kinetic_continuum_elements_dn
        - non_diagonal_kinetic_part_elements_dn
    )
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
    else:
        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
    diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
        diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
    )
    diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
        sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
    )

    # final bare coulomb part
    discretized_diagonal_bare_coulomb_part = (
        diagonal_bare_coulomb_part_el_el
        + diagonal_bare_coulomb_part_ion_ion
        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
    )

    # with ECP
    if hamiltonian_data.coulomb_potential_data.ecp_flag:
        diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        if non_local_move == "tmove":
            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                compute_ecp_non_local_parts_nearest_neighbors(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    flag_determinant_only=False,
                    RT=ecp_RT,
                )
            )
            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
            non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

        elif non_local_move == "dltmove":
            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                compute_ecp_non_local_parts_nearest_neighbors(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    flag_determinant_only=True,
                    RT=ecp_RT,
                )
            )
            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

            Jastrow_ref = compute_Jastrow_part_ratio_zone(
                jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )
            Jastrow_on_mesh = vmap(compute_Jastrow_part_ratio_zone, in_axes=(None, 0, 0))(
                hamiltonian_data.wavefunction_data.jastrow_data,
                mesh_non_local_ecp_part_r_up_carts,
                mesh_non_local_ecp_part_r_dn_carts,
            )
            Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
            V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
            non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

        else:
            raise NotImplementedError

        V_diag = (
            diagonal_kinetic_part
            + discretized_diagonal_bare_coulomb_part
            + diagonal_ecp_local_part
            + diagonal_kinetic_part_SP
            + diagonal_ecp_part_SP
        )
        V_nondiag = non_diagonal_sum_hamiltonian

        # Measurement local energy (novar=0 split): continuum e_L. Equals
        # V_diag + V_nondiag on non-clipped configurations (zero-variance
        # identity); this is the composition force/stress differentiate.
        e_L_meas = (
            diagonal_bare_coulomb_part_el_el
            + diagonal_bare_coulomb_part_ion_ion
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
            + diagonal_ecp_local_part
            + diagonal_ecp_part_SP
            + non_diagonal_sum_hamiltonian_ecp
        )

    # with all electrons
    else:
        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
        V_diag = diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP
        V_nondiag = non_diagonal_sum_hamiltonian

        # Measurement local energy (novar=0 split) -- see ECP branch.
        e_L_meas = (
            diagonal_bare_coulomb_part_el_el
            + diagonal_bare_coulomb_part_ion_ion
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
        )

    return (V_diag, V_nondiag, e_L_meas)


# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# separator
num_sep_line = 66

# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


# accumurate weights
@partial(jit, static_argnums=1)
def _compute_G_L_split(w_L: jax.Array, num_gfmc_collect_steps: int) -> tuple[jax.Array, jax.Array]:
    """Return the accumulated weights as binary mantissas and integer exponents.

    ``G_L[j] = prod(w_L[j : j + k]) = mantissa[j] * 2**exponent[j]`` with
    ``k = num_gfmc_collect_steps``. Each factor is split by ``frexp`` and the
    running mantissa is renormalized after every multiplication, so the
    magnitude lives in the exact integer exponent and nothing overflows for any
    window length. Power-of-two scaling is exact, so the mantissa rounding is
    that of the sequential product.

    Note: The dimension of w_L is (num_mcmc, 1)

    """
    A, x = w_L.shape
    if A <= num_gfmc_collect_steps:
        # No complete historical window precedes a retained measurement.
        # Do not trace an oversized dynamic_slice.
        return jnp.empty((0, x), dtype=w_L.dtype), jnp.empty((0, x), dtype=jnp.int32)

    n_out = A - num_gfmc_collect_steps
    factor_mantissa, factor_exponent = jnp.frexp(w_L)

    def multiply_factor(i: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        mantissa, exponent = carry
        mantissa, renormalization = jnp.frexp(mantissa * jax.lax.dynamic_slice(factor_mantissa, (i, 0), (n_out, x)))
        exponent = exponent + jax.lax.dynamic_slice(factor_exponent, (i, 0), (n_out, x)) + renormalization
        return mantissa, exponent

    initial = (jnp.ones((n_out, x), dtype=w_L.dtype), jnp.zeros((n_out, x), dtype=factor_exponent.dtype))
    return jax.lax.fori_loop(0, num_gfmc_collect_steps, multiply_factor, initial)


def _compute_G_L(w_L: jax.Array, num_gfmc_collect_steps: int) -> jax.Array:
    """Return the unscaled accumulated weights for multi-dimensional w_L.

    ``G_L[j] = prod(w_L[j : j + k])``. This raw product leaves the float64
    range once ``k * tau * |E|`` exceeds about 709; the estimators use
    :func:`_compute_retained_G_L` instead.

    Note: The dimension of w_L is (num_mcmc, 1)

    """
    mantissa, exponent = _compute_G_L_split(w_L, num_gfmc_collect_steps)
    return jnp.asarray(np.ldexp(np.asarray(mantissa), np.asarray(exponent)))


def _compute_G_L_debug_split(w_L: jnpt.ArrayLike, num_gfmc_collect_steps: int) -> tuple[jax.Array, jax.Array]:
    """Debug (non-jit) twin of :func:`_compute_G_L_split`.

    Note: The dimension of w_L is (num_mcmc, num_walkers)

    """
    w_L = np.asarray(w_L, dtype=np.float64)
    A, x = w_L.shape
    n_out = max(A - num_gfmc_collect_steps, 0)
    mantissa = np.ones((n_out, x), dtype=np.float64)
    exponent = np.zeros((n_out, x), dtype=np.int64)
    for i in range(num_gfmc_collect_steps if n_out > 0 else 0):
        factor_mantissa, factor_exponent = np.frexp(w_L[i : i + n_out])
        mantissa, renormalization = np.frexp(mantissa * factor_mantissa)
        exponent += factor_exponent + renormalization
    return jnp.asarray(mantissa), jnp.asarray(exponent)


def _compute_G_L_debug(w_L: jnpt.ArrayLike, num_gfmc_collect_steps: int) -> jax.Array:
    """Return the unscaled accumulated weights for multi-dimensional w_L (debug twin of :func:`_compute_G_L`).

    Note: The dimension of w_L is (num_mcmc, num_walkers)

    """
    mantissa, exponent = _compute_G_L_debug_split(w_L, num_gfmc_collect_steps)
    return jnp.asarray(np.ldexp(np.asarray(mantissa), np.asarray(exponent)))


def _compute_retained_G_L(w_L: jnpt.ArrayLike, num_gfmc_collect_steps: int, num_warmup_steps: int) -> npt.NDArray:
    """Accumulated weights of the rows after ``num_warmup_steps``, on one common scale.

    The retained ``(mantissa, exponent)`` rows of :func:`_compute_G_L_split`
    are multiplied by the single exact factor ``2**-M`` of
    :func:`jqmc._checkpoint.scale_accumulated_weights` (``M`` = the largest
    retained exponent), which cancels in every weighted mean and jackknife
    ratio. The weights therefore stay finite for any collect window, while the
    estimates are those of the unscaled products.
    """
    from ._checkpoint import scale_accumulated_weights

    mantissa, exponent = _compute_G_L_split(jnp.asarray(w_L), num_gfmc_collect_steps)
    retained = (np.asarray(mantissa)[num_warmup_steps:], np.asarray(exponent)[num_warmup_steps:])
    return scale_accumulated_weights([retained])[0]


def _compute_retained_G_L_debug(w_L: jnpt.ArrayLike, num_gfmc_collect_steps: int, num_warmup_steps: int) -> npt.NDArray:
    """Debug twin of :func:`_compute_retained_G_L` (built on :func:`_compute_G_L_debug_split`)."""
    from ._checkpoint import scale_accumulated_weights

    mantissa, exponent = _compute_G_L_debug_split(w_L, num_gfmc_collect_steps)
    retained = (np.asarray(mantissa)[num_warmup_steps:], np.asarray(exponent)[num_warmup_steps:])
    return scale_accumulated_weights([retained])[0]


class _GFMCBase:
    """Shared read-only accessors and the position-grad helper for the GFMC drivers.

    The four driver classes (GFMC_t, _GFMC_t_debug,
    GFMC_n, _GFMC_n_debug) inherit the single copy of the observable accessor
    properties and the position-grad object preparation helper defined here.
    Divergent members remain overridden / kept per-class: the ``hamiltonian_data``
    setter and ``num_gfmc_collect_steps`` setter (they differ / call the
    name-mangled ``__init_attributes``), ``average_projection_counter`` (only the
    _t variants store it), ``__validate_stored_shapes`` (GFMC_t vs GFMC_n differ),
    and the debug ``w_L`` / ``_retained_w_L`` accessors (use ``_compute_G_L_debug``). The two
    weight helpers share their algorithm; their mutual agreement is route
    parity, not an independent accumulation oracle. The blocking
    ``MATH-GFMC-ACCUMULATED-WEIGHTS-001`` contract separately compares them
    with direct window products and a scalar-loop reference.
    """

    # MPC (physical = no) defaults. The production drivers overwrite these in
    # ``__init__``; the ``_GFMC_*_debug`` oracle siblings build their state
    # through their own ``__init_attributes`` and never enable MPC, so the class
    # attribute is what makes ``get_dE_mpc`` raise a clean ValueError there.
    _mpc = False
    _mpc_cutoff = MPC_CUTOFF_HA

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        num_walkers: int = 40,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        alat: float = 0.1,
        random_discretized_mesh: bool = True,
        non_local_move: NonLocalMove = "tmove",
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        epsilon_PW: float = 0.0,
        use_swct: bool = False,
        mpc: bool = False,
        mpc_cutoff: float = MPC_CUTOFF_HA,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Set up the shared GFMC driver state (walkers, RNG, stored buffers).

        Subclasses set their own config attribute(s) -- GFMC_t: ``tau``;
        GFMC_n: ``num_mcmc_per_measurement`` / ``E_scf`` -- before delegating
        the common walker / RNG / attribute setup here via ``super().__init__``.
        """
        log_build_info_once(logger)  # Python API runs report their code version once (issue #183)
        # Per-instance communicator injection lets a twist-averaged driver run
        # each twist on its own intra-communicator.
        # Defaults fall back to the module global mpi_comm (byte-identical).
        self._mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        self._world_comm = world_comm_in if world_comm_in is not None else self._mpi_comm
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_size = self._mpi_comm.Get_size()

        _validate_gfmc_alat(alat)

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # attributes
        self._hamiltonian_data = hamiltonian_data
        self._num_walkers = num_walkers
        self._num_gfmc_collect_steps = num_gfmc_collect_steps
        self._mcmc_seed = mcmc_seed
        self._alat = alat
        self._random_discretized_mesh = random_discretized_mesh
        # Validate at the earliest single point the value is known. The native
        # GFMC chunk coerces ``non_local_move`` to the boolean
        # ``dltmove == (non_local_move == "dltmove")`` (see _resolve_native_gfmc_chunk
        # payload), so any unsupported value (None, typos) would silently run tmove
        # on the native chunk while XLA's _projection_t_core raises NotImplementedError.
        # Raise here so both backends reject unsupported moves identically.
        if non_local_move not in ("tmove", "dltmove"):
            raise NotImplementedError(
                f"Unsupported non_local_move={non_local_move!r}; only 'tmove' and 'dltmove' are implemented."
            )
        self._non_local_move = non_local_move

        # derivative flags
        self._comput_position_deriv = comput_position_deriv
        # Stress is requested independently from atomic force. Molecular runs
        # never compute stress even when this flag is enabled.
        self._comput_stress = comput_stress
        self._use_swct = use_swct

        # Pathak-Wagner regularization (DOI: 10.1063/5.0004008)
        self._epsilon_PW = epsilon_PW

        # MPC (physical = no): observable-only finite-size correction. The flags
        # must be set BEFORE the ``hamiltonian_data`` setter fires at the end of
        # this ``__init__`` because ``_init_attributes`` reads them to decide
        # whether to build the (expensive) MPC tables.
        self._mpc = mpc
        self._mpc_cutoff = mpc_cutoff
        if self._mpc and not hamiltonian_data.structure_data.pbc_flag:
            raise ValueError(
                "mpc=True requires a periodic hamiltonian_data (structure_data.pbc_flag=True). "
                "The MPC finite-size correction is defined only for a periodic cell."
            )

        # gfmc branching counter
        self._mcmc_counter = 0

        # Initialization
        # Hierarchical SeedSequence derivation (single derivation point:
        # ``derive_rank_rng_streams``; stream key = [*chain_seed_words, rank]).
        # ``_mpi_seed`` = derived 63-bit JAX base-key int; ``_np_seed_words``
        # = eight uint32 init_by_array words for the NumPy global RNG.
        self._np_seed_words, self._mpi_seed = derive_rank_rng_streams(self._mcmc_seed, self._mpi_rank)
        # Branching owns its stream: repeated runs and checkpoints continue it,
        # independently of initialization or another driver's global RNG use.
        self._branching_rng = np.random.RandomState(self._np_seed_words)
        self._jax_PRNG_key = jax.random.PRNGKey(self._mpi_seed)
        # Use jax.random.split (batched) instead of a Python list-comp of
        # fold_in calls; the latter scaled linearly with num_walkers and
        # dominated init time at large walker counts (e.g. nw = 16384).
        self._jax_PRNG_key_list_init = jax.random.split(self._jax_PRNG_key, self._num_walkers)
        self._jax_PRNG_key_list = self._jax_PRNG_key_list_init

        # initialize random seed
        np.random.seed(self._np_seed_words)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp
        # Cell for PBC wrapping of the seed configuration (None for molecules).
        cell = hamiltonian_data.structure_data.cell if hamiltonian_data.structure_data.pbc_flag else None

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self._num_walkers, charges, coords, cell
        )

        dtype_jnp = jnp.float64
        self._latest_r_up_carts = jnp.asarray(r_carts_up, dtype=dtype_jnp)
        self._latest_r_dn_carts = jnp.asarray(r_carts_dn, dtype=dtype_jnp)

        logger.debug(f"  initial r_up_carts= {self._latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self._latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self._latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self._latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {self._mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self._num_walkers}.")
        logger.info("")

        # init attributes: the ``hamiltonian_data`` setter already applies the
        # diff mask and calls ``_init_attributes`` once; a second explicit call
        # would rebuild the same buffers and, with MPC on, repeat ``build_mpc_data``.
        self.hamiltonian_data = self._hamiltonian_data

    def _init_attributes(self) -> None:
        # mcmc counter
        self._mcmc_counter = 0

        # gfmc accepted/rejected moves
        self._num_survived_walkers = 0
        self._num_killed_walkers = 0

        # Compute dimensions for pre-shaped empty arrays
        n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        n_atoms = self._hamiltonian_data.structure_data.natom

        # Pin the complex Bloch determinant orbital leaf so we know
        # whether to store ``e_L`` as complex128 (twist k != 0) or float64
        # (Gamma). Mirrors the MCMC ``_wf_is_complex`` pattern.
        self._wf_is_complex = np.issubdtype(
            self._hamiltonian_data.wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating
        )

        # dtype for stored numpy arrays
        dtype_np = np.float64
        # e_L is complex at a non-Gamma twist;
        # ``_stored_e_L2`` stays real because it stores ``|e_L|^2``.
        _e_L_dtype = np.complex128 if self._wf_is_complex else dtype_np

        # stored weight (w_L)
        self._stored_w_L = np.zeros((0, 1), dtype=dtype_np)

        # stored local energy (e_L)
        self._stored_e_L = np.zeros((0, 1), dtype=_e_L_dtype)

        # stored local energy (e_L2)
        self._stored_e_L2 = np.zeros((0, 1), dtype=dtype_np)

        # MPC (physical = no): tables + per-walker closure for the passive
        # observable dV = V_ee^MPC - V_ee^Ewald. ``build_mpc_data`` is expensive
        # (FFT extrapolation of g_G) so it is built HERE -- once per
        # ``hamiltonian_data`` -- and never inside the branching loop. The
        # stored buffer stays real float64 at any twist: dV is a difference of
        # two real e-e potentials and carries no e_L phase.
        if self._mpc:
            self._mpc_data = build_mpc_data(self._hamiltonian_data, cutoff_ha=self._mpc_cutoff, comm=self._mpi_comm)
            self._jit_vmap_mpc_correction_ee = jit(vmap(compute_mpc_correction_ee, in_axes=(None, None, 0, 0)))
        else:
            self._mpc_data = None
            self._jit_vmap_mpc_correction_ee = None

        # stored MPC correction dV = V_ee^MPC - V_ee^Ewald, reduced across
        # walkers exactly like e_L (zero-width when mpc is off)
        self._stored_mpc_dV = np.zeros((0, 1), dtype=dtype_np)

        # stored force products (per-walker cross-correlation preserved).
        # force_HF / force_PP are real because their gradient sites project to
        # the physical real component; only the covariance
        # bookkeeping product ``E_L * force_PP`` lifts to complex because
        # ``e_L`` is complex per sample (twist k != 0). The Re projection is
        # taken at binning-time aggregation in ``get_aF``.
        self._stored_force_HF = np.zeros((0, 1, n_atoms, 3), dtype=dtype_np)
        self._stored_force_PP = np.zeros((0, 1, n_atoms, 3), dtype=dtype_np)
        self._stored_E_L_force_PP = np.zeros((0, 1, n_atoms, 3), dtype=_e_L_dtype)

        # Branching-weighted Hellmann-Feynman and Pulay stress tensors
        # (Hartree), shape ``(N_steps, 1, 3, 3)``. ``stress_HF`` / ``stress_PP``
        # are real; ``E_L_stress_PP`` (covariance bookkeeping ``e_L * sigma_PP``)
        # lifts to complex, exactly as the force buffers above. The Re
        # projection is taken at binning time in ``get_aT``.
        self._stored_stress_HF = np.zeros((0, 1, 3, 3), dtype=dtype_np)
        self._stored_stress_PP = np.zeros((0, 1, 3, 3), dtype=dtype_np)
        self._stored_E_L_stress_PP = np.zeros((0, 1, 3, 3), dtype=_e_L_dtype)

        # class-specific stored buffers (GFMC_t: average_projection_counter;
        # GFMC_n: G_L / G_e_L lists for the E_scf update).
        self._init_attributes_class_specific()

    def _init_attributes_class_specific(self) -> None:
        """Hook: subclasses allocate their class-specific stored buffers."""

    # ------------------------------------------------------------------
    # HDF5 checkpoint save / load (shared skeleton)
    # ------------------------------------------------------------------

    def save_to_hdf5(self, filepath: str, *, driver_type: str, extra_config: dict, extra_observables: dict) -> None:
        """Write this rank's GFMC state to a temporary per-rank HDF5 file.

        The file is later merged by :func:`jqmc._checkpoint.merge_rank_checkpoints`.
        Subclasses call this with their class-specific ``driver_type`` and the
        divergent config / observable keys (GFMC_t: ``tau`` +
        ``average_projection_counter``; GFMC_n: ``num_mcmc_per_measurement`` /
        ``E_scf`` + ``G_L`` / ``G_e_L``).

        Args:
            filepath: Output path (e.g. ``._restart_rank0.h5``).
            driver_type: ``"GFMC_t"`` or ``"GFMC_n"``.
            extra_config: Class-specific scalar config keys.
            extra_observables: Class-specific stored observable arrays.
        """
        from ._checkpoint import save_rank_checkpoint

        # -- driver_config (scalar execution parameters) --
        driver_config: dict = {
            "mcmc_seed": self._mcmc_seed,
            "num_walkers": self._num_walkers,
            "num_gfmc_collect_steps": self._num_gfmc_collect_steps,
            "alat": self._alat,
            "random_discretized_mesh": self._random_discretized_mesh,
            "non_local_move": self._non_local_move,
            "comput_position_deriv": self._comput_position_deriv,
            "comput_stress": self._comput_stress,
            "epsilon_PW": self._epsilon_PW,
            "use_swct": self._use_swct,
            "mpc": self._mpc,
            "mpc_cutoff": self._mpc_cutoff,
            "mcmc_counter": self._mcmc_counter,
            "num_survived_walkers": self._num_survived_walkers,
            "num_killed_walkers": self._num_killed_walkers,
            "branching_rng_state_version": 1,
        }
        driver_config.update(extra_config)

        # -- rng_state -- (copy MT19937 only at checkpoint time, not per step)
        branching_state = self._branching_rng.get_state()
        rng_state: dict = {
            "jax_PRNG_key_list": np.asarray(self._jax_PRNG_key_list),
            "jax_PRNG_key_list_init": np.asarray(self._jax_PRNG_key_list_init),
            "mpi_seed": self._mpi_seed,
            "np_seed_words": np.asarray(self._np_seed_words, dtype=np.uint32),
            "branching_rng_algorithm": branching_state[0],
            "branching_rng_keys": branching_state[1],
            "branching_rng_position": branching_state[2],
            "branching_rng_has_gauss": branching_state[3],
            "branching_rng_cached_gaussian": branching_state[4],
        }

        # -- walker_state --
        walker_state: dict = {
            "latest_r_up_carts": np.asarray(self._latest_r_up_carts),
            "latest_r_dn_carts": np.asarray(self._latest_r_dn_carts),
        }

        # -- observables --
        observables: dict = {}
        _obs_map = {
            "e_L": self._stored_e_L,
            "e_L2": self._stored_e_L2,
            "w_L": self._stored_w_L,
            "mpc_dV": self._stored_mpc_dV,
            "force_HF": self._stored_force_HF,
            "force_PP": self._stored_force_PP,
            "E_L_force_PP": self._stored_E_L_force_PP,
            "stress_HF": self._stored_stress_HF,
            "stress_PP": self._stored_stress_PP,
            "E_L_stress_PP": self._stored_E_L_stress_PP,
        }
        _obs_map.update(extra_observables)
        for key, val in _obs_map.items():
            observables[key] = np.asarray(val) if val.size > 0 else np.empty(0)

        save_rank_checkpoint(
            filepath,
            driver_type=driver_type,
            driver_config=driver_config,
            rng_state=rng_state,
            walker_state=walker_state,
            observables=observables,
        )

    @staticmethod
    def _load_obs(arr: npt.ArrayLike | None, default: np.ndarray) -> np.ndarray:
        """Return ``arr`` as an ndarray, or ``default`` if it is missing/empty."""
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            return default
        return np.asarray(arr)

    @classmethod
    def load_from_hdf5(
        cls,
        filepath: str,
        rank: int | None = None,
        *,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> Self:
        """Restore a GFMC driver instance from a merged HDF5 checkpoint.

        This bypasses ``__init__`` entirely and reconstructs all internal
        attributes from the stored data and the embedded ``hamiltonian_data``.
        The class-specific config / observable keys are restored by the
        ``_load_class_specific`` hook.

        Args:
            filepath: Path to the merged checkpoint (e.g. ``restart.h5``).
            rank: MPI rank to load. Defaults to the rank of the resolved
                ``mpi_comm`` (matches what ``__init__`` would have used).
            mpi_comm_in: Per-instance comm, matching ``__init__``. ``None``
                falls back to the module-level ``mpi_comm``.
            world_comm_in: All-rank comm, matching ``__init__``. ``None``
                falls back to ``mpi_comm_in``.

        Returns:
            A fully restored driver instance (``cls``).
        """
        from ._checkpoint import (
            _validate_restart_rank_topology,
            load_hamiltonian_from_checkpoint,
            load_rank_checkpoint_for_restart,
            raise_on_restart_inconsistency,
        )

        _mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        _world_comm = world_comm_in if world_comm_in is not None else _mpi_comm
        _validate_restart_rank_topology(filepath, _mpi_comm.Get_size())
        if rank is None:
            rank = _mpi_comm.Get_rank()

        data, problems = load_rank_checkpoint_for_restart(filepath, rank)
        # The restored scalar domains are checked first, by their own
        # collective validators (one call on every rank, missing -> None).
        cls._validate_restored_domains({} if data is None else data["driver_config"], _mpi_comm)
        # The Hamiltonian is the shared root group: identical on every rank.
        hamiltonian_data = load_hamiltonian_from_checkpoint(filepath)
        if data is not None:
            cfg = data["driver_config"]
            rng = data["rng_state"]
            ws = data["walker_state"]
            obs = data["observables"]
            problems = cls._restart_state_problems(cfg, rng, ws, obs, rank, hamiltonian_data, filepath)
        # Every rank checks its own group against the keys the writer always
        # writes and against its peers BEFORE any restored value gates a
        # collective (issues #805-#807): a missing key (e.g. non_local_move or
        # mcmc_counter), a counter that does not match the stored history, or a
        # disagreement between ranks fails identically on every rank.
        raise_on_restart_inconsistency(_mpi_comm, context=f"{cls.__name__} restart from {filepath!r}", rank=rank, **problems)

        # Create an empty instance without calling __init__
        obj = cls.__new__(cls)
        log_build_info_once(logger)  # a restart-only Python API run also reports its code version (PR #184)

        # Per-instance comms (parallel of __init__). ``run`` reads
        # ``self._mpi_rank`` on entry, so missing these raises AttributeError.
        obj._mpi_comm = _mpi_comm
        obj._world_comm = _world_comm
        obj._mpi_rank = _mpi_comm.Get_rank()
        obj._mpi_size = _mpi_comm.Get_size()

        # -- Scalar config (shared) --
        obj._mcmc_seed = cfg["mcmc_seed"]
        obj._num_walkers = cfg["num_walkers"]
        obj._num_gfmc_collect_steps = cfg["num_gfmc_collect_steps"]
        obj._alat = cfg["alat"]
        obj._random_discretized_mesh = bool(cfg["random_discretized_mesh"])
        obj._non_local_move = cfg["non_local_move"]
        obj._comput_position_deriv = bool(cfg["comput_position_deriv"])
        obj._comput_stress = bool(cfg["comput_stress"])
        obj._epsilon_PW = cfg["epsilon_PW"]
        obj._use_swct = bool(cfg["use_swct"])

        # MPC flags must land BEFORE the ``hamiltonian_data`` setter below:
        # ``_init_attributes`` reads them to rebuild the MPC tables + closure.
        obj._mpc = bool(cfg["mpc"])
        obj._mpc_cutoff = float(cfg["mpc_cutoff"])

        # Counters
        obj._mcmc_counter = cfg["mcmc_counter"]
        obj._num_survived_walkers = cfg["num_survived_walkers"]
        obj._num_killed_walkers = cfg["num_killed_walkers"]

        # -- Hamiltonian data (apply DiffMask as the normal setter does) --
        obj._hamiltonian_data = hamiltonian_data
        obj.hamiltonian_data = hamiltonian_data  # triggers setter -> DiffMask + _init_attributes

        # -- Overwrite _init_attributes results with loaded state --
        obj._mcmc_counter = cfg["mcmc_counter"]
        obj._num_survived_walkers = cfg["num_survived_walkers"]
        obj._num_killed_walkers = cfg["num_killed_walkers"]

        # -- RNG state -- (restore the SAVED derived values; never re-derive
        # the advanced key list -- resume must continue the stored streams)
        obj._mpi_seed = rng["mpi_seed"]
        obj._jax_PRNG_key = jax.random.PRNGKey(rng["mpi_seed"])
        obj._jax_PRNG_key_list = jnp.array(rng["jax_PRNG_key_list"])
        obj._jax_PRNG_key_list_init = jnp.array(rng["jax_PRNG_key_list_init"])
        obj._np_seed_words = np.asarray(rng["np_seed_words"], dtype=np.uint32)

        # The branching MT19937 state was validated collectively above.
        obj._branching_rng = np.random.RandomState(obj._np_seed_words)
        obj._branching_rng.set_state(
            (
                "MT19937",
                np.asarray(rng["branching_rng_keys"]),
                rng["branching_rng_position"],
                rng["branching_rng_has_gauss"],
                rng["branching_rng_cached_gaussian"],
            )
        )

        # -- Walker state --
        # (walker shapes were validated collectively above)
        from ._checkpoint import validate_restored_walker_finiteness_collectively

        validate_restored_walker_finiteness_collectively(
            _mpi_comm,
            driver=cls.__name__,
            filepath=filepath,
            r_up_carts=ws["latest_r_up_carts"],
            r_dn_carts=ws["latest_r_dn_carts"],
        )
        dtype_jnp = jnp.float64
        obj._latest_r_up_carts = jnp.asarray(ws["latest_r_up_carts"], dtype=dtype_jnp)
        obj._latest_r_dn_carts = jnp.asarray(ws["latest_r_dn_carts"], dtype=dtype_jnp)

        # -- Observables (shared) --
        n_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        n_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        n_atoms = hamiltonian_data.structure_data.natom

        obj._stored_e_L = cls._load_obs(obs.get("e_L"), np.zeros((0, 1)))
        obj._stored_e_L2 = cls._load_obs(obs.get("e_L2"), np.zeros((0, 1)))
        obj._stored_w_L = cls._load_obs(obs.get("w_L"), np.zeros((0, 1)))
        obj._stored_mpc_dV = cls._load_obs(obs.get("mpc_dV"), obj._stored_mpc_dV)
        obj._stored_force_HF = cls._load_obs(obs.get("force_HF"), np.zeros((0, 1, n_atoms, 3)))
        obj._stored_force_PP = cls._load_obs(obs.get("force_PP"), np.zeros((0, 1, n_atoms, 3)))
        obj._stored_E_L_force_PP = cls._load_obs(obs.get("E_L_force_PP"), np.zeros((0, 1, n_atoms, 3)))
        obj._stored_stress_HF = cls._load_obs(obs.get("stress_HF"), np.zeros((0, 1, 3, 3)))
        obj._stored_stress_PP = cls._load_obs(obs.get("stress_PP"), np.zeros((0, 1, 3, 3)))
        obj._stored_E_L_stress_PP = cls._load_obs(obs.get("E_L_stress_PP"), np.zeros((0, 1, 3, 3)))

        # -- Class-specific config + observables --
        obj._load_class_specific(cfg, obs)

        # GFMC checkpoints store rank-root averaged observables with a singleton
        # walker axis. Collapse the current positions into the same diagnostic
        # axis while retaining their maximum absolute coordinate in an error.
        _restart_positions = np.concatenate(
            [np.asarray(obj._latest_r_up_carts).reshape(-1, 3), np.asarray(obj._latest_r_dn_carts).reshape(-1, 3)],
            axis=0,
        ).reshape(1, -1, 3)
        _raise_collective_nonfinite_observables(
            obj._mpi_comm,
            driver=cls.__name__,
            stage="checkpoint load",
            first_step=0,
            observables={
                "stored_e_L": obj._stored_e_L,
                "stored_e_L2": obj._stored_e_L2,
                "stored_w_L": obj._stored_w_L,
            },
            r_up_carts=_restart_positions,
            r_dn_carts=np.empty((1, 0, 3)),
        )

        return obj

    def _load_class_specific(self, cfg: Mapping[str, Any], obs: Mapping[str, Any]) -> None:
        """Hook: subclasses restore their class-specific config + observables."""

    # driver_config keys that ``save_to_hdf5`` always writes for every GFMC
    # driver; ``_RESTART_CLASS_CONFIG_KEYS`` adds the class-specific ones.
    _RESTART_CONFIG_KEYS: tuple[str, ...] = (
        "mcmc_seed",
        "num_walkers",
        "num_gfmc_collect_steps",
        "alat",
        "random_discretized_mesh",
        "non_local_move",
        "comput_position_deriv",
        "comput_stress",
        "epsilon_PW",
        "use_swct",
        "mpc",
        "mpc_cutoff",
        "mcmc_counter",
        "num_survived_walkers",
        "num_killed_walkers",
    )
    _RESTART_CLASS_CONFIG_KEYS: tuple[str, ...] = ()
    # Per-rank branching statistics: the only restored scalars that may differ between ranks.
    _RESTART_PER_RANK_CONFIG_KEYS = frozenset({"num_survived_walkers", "num_killed_walkers"})
    _BRANCHING_RNG_FIELDS = frozenset(
        {
            "branching_rng_algorithm",
            "branching_rng_keys",
            "branching_rng_position",
            "branching_rng_has_gauss",
            "branching_rng_cached_gaussian",
        }
    )

    @classmethod
    def _restart_state_problems(
        cls,
        cfg: Mapping[str, Any],
        rng: Mapping[str, Any],
        ws: Mapping[str, Any],
        obs: Mapping[str, Any],
        rank: int,
        hamiltonian_data: Hamiltonian_data,
        filepath: str,
    ) -> dict[str, Any]:
        """This rank's missing keys, damaged state and to-be-agreed values for a restart."""
        from ._checkpoint import history_length_errors, missing_restart_keys, walker_shape_errors

        missing = missing_restart_keys("driver_config", cfg, cls._RESTART_CONFIG_KEYS + cls._RESTART_CLASS_CONFIG_KEYS)
        missing += missing_restart_keys(
            "rng_state", rng, ("jax_PRNG_key_list", "jax_PRNG_key_list_init", "mpi_seed", "np_seed_words")
        )
        walker_missing = missing_restart_keys("walker_state", ws, ("latest_r_up_carts", "latest_r_dn_carts"))
        missing += walker_missing
        errors: list[tuple[str, str]] = []
        if not walker_missing and "num_walkers" in cfg:
            geminal = hamiltonian_data.wavefunction_data.geminal_data
            errors += walker_shape_errors(
                ws,
                num_walkers=cfg["num_walkers"],
                num_electron_up=geminal.num_electron_up,
                num_electron_dn=geminal.num_electron_dn,
                filepath=filepath,
            )
        branching_version = cfg.get("branching_rng_state_version")
        present_branching_fields = {key for key in rng if key.startswith("branching_rng_")}
        if branching_version is None:
            # Always written since the branching stream became restart state; a
            # legacy checkpoint without it is rejected, never reseeded.
            errors.append(
                (
                    "driver_config/branching_rng_state_version",
                    "Invalid GFMC branching RNG checkpoint version: the key is missing (legacy checkpoint).",
                )
            )
        elif type(branching_version) is not int or branching_version != 1:
            errors.append(("driver_config", "Invalid GFMC branching RNG checkpoint version."))
        elif present_branching_fields != cls._BRANCHING_RNG_FIELDS:
            errors.append(("rng_state", "Incomplete or unknown GFMC branching RNG checkpoint fields."))
        else:
            keys = np.asarray(rng["branching_rng_keys"])
            position = rng["branching_rng_position"]
            has_gauss = rng["branching_rng_has_gauss"]
            cached_gaussian = rng["branching_rng_cached_gaussian"]
            if (
                type(rng["branching_rng_algorithm"]) is not str
                or rng["branching_rng_algorithm"] != "MT19937"
                or keys.shape != (624,)
                or keys.dtype.kind != "u"
                or keys.dtype.itemsize != 4
                or not np.any(keys)
                or type(position) is not int
                or not 0 <= position <= 624
                or type(has_gauss) is not int
                or has_gauss not in (0, 1)
                or type(cached_gaussian) is not float
                or not np.isfinite(cached_gaussian)
            ):
                errors.append(("rng_state", "Invalid GFMC branching RNG checkpoint state."))
        counter = cfg.get("mcmc_counter")
        if counter is not None:
            if type(counter) is not int or counter < 0:
                errors.append(("driver_config/mcmc_counter", f"must be a non-negative integer, got {counter!r}"))
            else:
                # The averaged histories are root-owned: rank 0 stores one row
                # per step, every other rank stores none.
                errors += history_length_errors(obs, ("e_L", "e_L2", "w_L"), counter if rank == 0 else 0, counter)
                errors += cls._class_restart_history_errors(obs, counter, rank)
        agreed = {key: value for key, value in cfg.items() if key not in cls._RESTART_PER_RANK_CONFIG_KEYS}
        return {"missing": missing, "errors": errors, "agreed": agreed}

    @classmethod
    def _class_restart_history_errors(cls, obs: Mapping[str, Any], counter: int, rank: int) -> list[tuple[str, str]]:
        """Hook: class-specific stored histories that must match the restored counter."""
        return []

    @classmethod
    def _validate_restored_domains(cls, cfg: Mapping[str, Any], comm: MPI.Comm) -> None:
        """Collectively validate the restored scalar domains (subclasses extend)."""
        _validate_gfmc_alat(cfg.get("alat"), comm=comm, driver=cls.__name__)

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self._mcmc_counter - self._num_gfmc_collect_steps

    @property
    def num_walkers(self) -> int:
        """The number of walkers."""
        return self._num_walkers

    @property
    def alat(self) -> float:
        """Return alat."""
        return self._alat

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the unscaled accumulated weight array. dim: (mcmc_counter, 1).

        The raw products overflow float64 for long collect windows; the
        estimators use :meth:`_retained_w_L`, which shares one exact
        power-of-two scale and leaves every ratio unchanged.
        """
        return _compute_G_L(np.asarray(self._stored_w_L), self._num_gfmc_collect_steps)

    def _retained_w_L(self, num_warmup_steps: int) -> npt.NDArray:
        """Accumulated weights after ``num_warmup_steps``, scaled by one common power of two."""
        return _compute_retained_G_L(np.asarray(self._stored_w_L), self._num_gfmc_collect_steps, num_warmup_steps)

    @property
    def bare_w_L(self) -> npt.NDArray:
        """Return the stored (bare) weight array. dim: (mcmc_counter, 1)."""
        return np.asarray(self._stored_w_L)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, 1)."""
        return np.asarray(self._stored_e_L)[self._num_gfmc_collect_steps :]

    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L2 array. dim: (mcmc_counter, 1)."""
        return np.asarray(self._stored_e_L2)[self._num_gfmc_collect_steps :]

    @property
    def mpc_dV(self) -> npt.NDArray:
        """Return the stored MPC correction dV = V_ee^MPC - V_ee^Ewald. dim: (mcmc_counter, 1).

        Empty (zero rows) when the driver was built with ``mpc=False``. See :meth:`get_dE_mpc`.
        """
        return np.asarray(self._stored_mpc_dV)[self._num_gfmc_collect_steps :]

    @property
    def force_HF(self) -> npt.NDArray:
        """Return the stored force_HF array. dim: (mcmc_counter, 1, num_atoms, 3)."""
        return np.asarray(self._stored_force_HF)[self._num_gfmc_collect_steps :]

    @property
    def force_PP(self) -> npt.NDArray:
        """Return the stored force_PP array. dim: (mcmc_counter, 1, num_atoms, 3)."""
        return np.asarray(self._stored_force_PP)[self._num_gfmc_collect_steps :]

    @property
    def E_L_force_PP(self) -> npt.NDArray:
        """Return the stored E_L * force_PP array. dim: (mcmc_counter, 1, num_atoms, 3)."""
        return np.asarray(self._stored_E_L_force_PP)[self._num_gfmc_collect_steps :]

    @property
    def stress_HF(self) -> npt.NDArray:
        """Return the stored HF stress tensor. dim: (mcmc_counter, 1, 3, 3)."""
        return np.asarray(self._stored_stress_HF)[self._num_gfmc_collect_steps :]

    @property
    def stress_PP(self) -> npt.NDArray:
        """Return the stored Pulay stress tensor. dim: (mcmc_counter, 1, 3, 3)."""
        return np.asarray(self._stored_stress_PP)[self._num_gfmc_collect_steps :]

    @property
    def E_L_stress_PP(self) -> npt.NDArray:
        """Return the stored E_L * sigma_PP array (covariance bookkeeping). dim: (mcmc_counter, 1, 3, 3)."""
        return np.asarray(self._stored_E_L_stress_PP)[self._num_gfmc_collect_steps :]

    @property
    def comput_position_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. atomic positions."""
        return self._comput_position_deriv

    @property
    def comput_stress(self) -> bool:
        """Return whether periodic stress observables are computed."""
        return self._comput_stress

    def _resolve_native_gfmc_chunk(self, driver: Literal["t", "n"]) -> NativeGFMCChunk | None:
        """Resolve the native GFMC projection-interval chunk gate.

        Mirrors ``jqmc_mcmc.run``'s ``rust_active`` resolution. One native
        ``run_gfmc_projection_{t,n}`` call per walker per branching step replaces the JAX
        projection ``lax.while_loop`` / ``fori_loop``; Python keeps the MPI branching /
        reconfiguration (collective across ranks), the bin statistics and the I/O.

        Both GFMC_n and GFMC_t use the native production chain whenever the requested
        physics is supported. Injected-RNG equivalence is tested through the Rust entry
        points directly and is not a production driver mode.

        * COMPLEX-twist: engaged in both drivers, energy-only and with
          ``comput_position_deriv`` / ``comput_stress``. The complex
          ``gfmc_interval_{n,t}_complex`` kernels reuse the verified complex e_L kernels
          with fixed-phase Re-projection and fold DLA ``dltmove`` too; the complex ECP
          non-local force AND strain kernels carry the DLA split. Periodic MPC rides the
          native chunk too (``mpc_data`` is threaded into the complex kernels). The only
          refused combination is a complex AO leaf on a non-periodic structure.
        * ``comput_position_deriv``: every real family is folded into the native chunk
          (``_rs_hf_native`` / ``_rs_pp_native`` are unconditionally true, MPC included).

        Returns ``None`` only when Rust dispatch is inactive, or a statics dict for
        the native run() chunk loop. Unsupported input with Rust active raises
        ``RuntimeError`` instead of selecting the per-step JAX path.
        Every call records its outcome in ``self._native_gfmc_chunk_route``:
        ``"native"`` (engaged), ``"jax:rust-inactive"`` (the Rust-on-CPU flag is off or
        the backend is not CPU) or ``"native-refused:<reason>"`` (Rust active but the
        requested physics is not native-serviceable). The refusal is observable
        from the driver as well as the exception.
        """
        # --- flag_rust_on_CPU dispatch, matching jqmc_mcmc.run -------------------
        rust_active = _rust_cpu_backend_active()
        if not rust_active:
            self._native_gfmc_chunk_route = "jax:rust-inactive"
            return None
        _pbc = bool(self._hamiltonian_data.structure_data.pbc_flag)
        _complex = bool(self._wf_is_complex)
        # Force/stress fold-in gate, matching jqmc_mcmc's ``_rs_obs_chunk_force_ok``.
        # A ``comput_position_deriv`` GFMC run engages the native chunk only when every
        # position-derivative observable is native-serviceable; today every real family
        # (MPC included) is, so this gate never refuses. The GFMC
        # HF force / Pulay force ride the parity-pinned LRDMC kernels folded into
        # ``run_gfmc_projection_{t,n}``; SWCT uses a fused native contraction,
        # while walker weighting stays in Python. (Periodic
        # Bare-AE force is included because the periodic kernels accept empty ECP arrays.
        # MPC is a passive energy-only observable (physical = no) with no force / stress
        # contribution, so it never forces a JAX fallback: the native HF force runs on
        # periodic-MPC runs exactly as on plain Ewald runs.
        _rs_hf_native = True
        _rs_pp_native = True
        # A ``_rs_gfmc_hf_family_ok = (not _pbc) or _ecp_flag`` gate would
        # incorrectly exclude periodic bare-AE force/stress. The underlying cores
        # (``_gfmc_grad_e_L_ecp_pbc`` / ``_gfmc_stress_ecp_pbc_{jnone,withj}`` -- the SAME
        # kernels ``run_gfmc_projection_{t,n}`` fold in) are not ECP-specific: the
        # local/non-local ECP arrays they read from ``coulomb_potential_data`` degrade to
        # empty when ``ecp_flag=False``, so the "ECP" cores compute the correct bare-AE
        # physics for free. tests/test_rs_kernel/test_ungate_gfmc_pbc_ae_force.py pins them
        # against the XLA continuum-measurement oracle to ~1e-14 on bare-AE fixtures (jnone / J12 /
        # DTN, force AND stress), and test_gfmc_chain_modes.py's gate matrix pins the
        # driver engaging the native chunk on a periodic bare-AE force run. Periodic
        # bare-AE force/stress is therefore native.  There is no real-Gamma
        # determinant-leaf guard: those GFMC stress kernels assemble pGTO, PW and
        # B-spline kinetic-strain contributions additively (including
        # pGTO+B-spline composites).  The complex path has the same support:
        # pure PW/B-spline leaves contribute no nuclear Pulay rows, while their
        # electron-axis kinetic force/stress is handled by the leaf-specific
        # complex kernels.  Direct force+stress parity is blocking in
        # test_gfmc_complex_force_stress.py, so there is no type gate or JAX
        # fallback here.
        # The complex-twist HF stress kernel
        # (``gfmc_stress_ecp_pbc_complex`` -> ``gfmc_ecp_nonlocal_strain_complex``) carries
        # the DLA split -- det-only amplitude clamp + det-only strain-Pulay / drift against the
        # matching det-only base -- exactly like its force twin
        # (``gfmc_ecp_nonlocal_force_complex``) and the real stress twin
        # (``gfmc_stress_ecp_pbc_withj``); ``dltmove`` is threaded from this dispatch through
        # ``PbcStressHfComplexArgs::compute_gfmc`` to the kernel, so complex ``dltmove``
        # position-derivatives are supported. Pinned by
        # tests/test_rs_kernel/test_gfmc_complex_force_stress.py, whose force AND stress rows
        # run BOTH ``non_local_move`` values against the XLA oracle.
        _rs_gfmc_force_chunk_ok = _rs_hf_native and _rs_pp_native
        _want_force = bool(self._comput_position_deriv)
        _want_stress = bool(_pbc and self._comput_stress)
        if _complex:
            # Complex-twist chunk for both drivers. The complex
            # ``gfmc_interval_{n,t}_complex`` reuse the verified complex e_L kernels with
            # fixed-phase Re-projection. The production chain uses native RNG/mesh
            # generation and is statistically guarded.
            #
            # The chunk is not energy-only. The
            # complex HF force / Pulay force / periodic HF+Pulay stress are folded out of
            # ``run_gfmc_projection_{n,t}_complex`` by ``gfmc_force_block_complex`` through
            # the same parity-pinned ``gfmc_{grad_e_l,stress}_ecp_pbc_complex`` cores the
            # free ``_gfmc_{grad_e_L,stress}_ecp_pbc_complex`` wrappers drive (pinned vs
            # the XLA continuum-measurement oracle in test_gfmc_complex_force_stress.py), so
            # a complex position-derivative run is native. The ONLY refusal is a complex
            # AO leaf on a non-periodic structure (``and _pbc`` below).
            eligible = (not _want_force or _rs_gfmc_force_chunk_ok) and _pbc
        else:
            eligible = not _want_force or _rs_gfmc_force_chunk_ok
        # The complex ``gfmc_interval_{n,t}_complex``
        # kernels fold BOTH tmove and the DLA ``dltmove`` non-local move (the
        # ``dltmove`` fold into the complex chunk mirrors XLA's ``_projection_t_core``:
        # SP gets max(Re(det_only),0), the FN off-diagonal is min(Re(det_only),0) *
        # Jastrow_ratio, with the fixed-phase Re at the source on the det-only ratio).
        # Complex eligibility also covers MPC.
        # ``run_gfmc_projection_{n,t}_complex`` take an ``mpc_data`` argument (threaded through
        # ``gfmc_interval_{n,t}_complex`` -> ``gfmc_lrdmc_{ae,ecp}_complex``), and those kernels
        # perform the Ewald-ee -> MPC-ee substitution exactly like the real path.
        # Because complex projection receives ``mpc_data``, complex-twist MPC
        # rides the native chunk with the correct MPC e_L.
        if not eligible:
            _reason = "complex-aos-without-pbc" if (_complex and not _pbc) else "position-derivative-not-native"
            self._native_gfmc_chunk_route = f"native-refused:{_reason}"
            raise RuntimeError(
                f"GFMC_{driver} native projection chunk cannot service the requested input: {_reason} "
                f"(complex={_complex}, pbc={_pbc}, position_deriv={self._comput_position_deriv}). "
                "Rust-on-CPU is enabled; automatic XLA fallback is disabled. "
                "Set JQMC_FLAG_RUST_ON_CPU=0 to explicitly select the XLA path."
            )
        try:
            import jqmc_rs
        except ImportError as exc:  # strict raise -- no silent JAX fallback
            # Exception-type contract: mirror jqmc_mcmc.py (RuntimeError, not ImportError)
            # so "flag on but jqmc_rs not importable" surfaces identically on both drivers.
            raise RuntimeError(
                "flag_rust_on_CPU is enabled on the CPU path but the "
                "native extension 'jqmc_rs' could not be imported. Build it with:\n"
                "    pip install -e .   (from the repository root; needs a Rust toolchain)\n"
                "or set JQMC_FLAG_RUST_ON_CPU=0 to force the JAX path."
            ) from exc
        # rayon build_global runs once per process, so the env override
        # (JQMC_RS_NUM_THREADS) can only take effect if applied before the
        # first batch kernel; idempotent across repeated run() calls.
        jqmc_rs.configure_threads(None)
        # Widen floating/complex leaves for native transport in either mode.
        # This preserves mixed-built values; arithmetic uses the separately
        # supplied precision policy below, not the transport dtype alone.
        _ham_f = jax.tree_util.tree_map(_leaf_to_fp64, self._hamiltonian_data)
        _gem_f = _ham_f.wavefunction_data.geminal_data
        _jas_f = _ham_f.wavefunction_data.jastrow_data
        _cp_f = _ham_f.coulomb_potential_data
        # Cross-run() ctx cache keyed on ham identity (checkpoint restarts / repeated
        # run() calls reuse; a swapped-in ham rebuilds -- mirrors MCMC's ctx cache).
        _precision_policy = precision_policy_snapshot()
        _flags = (driver, _pbc, _complex, _want_force, _want_stress, tuple(_precision_policy.items()))
        _cached = getattr(self, "_gfmc_rs_run_ctx", None)
        if (
            _cached is not None
            and getattr(self, "_gfmc_rs_run_ctx_ham", None) is self._hamiltonian_data
            and getattr(self, "_gfmc_rs_run_ctx_flags", None) == _flags
        ):
            _ctx = _cached
        else:
            # want_force/want_stress build the SAME force/stress ``*_Args`` the MCMC chunk
            # uses (molecular ae/ecp or periodic ecp + periodic stress); ``run_gfmc_projection``
            # drives them via the GFMC (LRDMC) compute methods on the projection final config.
            # complex=True builds the complex-twist GfmcArgs (complex geminal + complex J3)
            # for the energy-only ``run_gfmc_projection_{n,t}_complex`` interval.
            _ctx = jqmc_rs.build_obs_ctx(
                _gem_f,
                _jas_f,
                _cp_f,
                _complex,
                _pbc,
                _want_stress,
                want_force=_want_force,
                want_gfmc=True,
                precision_policy=_precision_policy,
            )
            self._gfmc_rs_run_ctx = _ctx
            self._gfmc_rs_run_ctx_ham = self._hamiltonian_data
            self._gfmc_rs_run_ctx_flags = _flags
        # ECP non-local quadrature: the SAME Nv_default base mesh jqmc's
        # ``compute_ecp_non_local_parts_nearest_neighbors`` uses (NN=1); the native
        # kernel RT-rotates it per sub-step internally.
        _mesh = _NV_BASE_MESH[Nv_default]
        logger.info(
            f"Native GFMC projection chunk ENGAGED (driver={driver}, pbc={_pbc}, "
            f"want_force={_want_force}, want_stress={_want_stress})."
        )
        self._native_gfmc_chunk_route = "native"
        return {
            "ctx": _ctx,
            "ecp_w": np.asarray(_mesh.weights, dtype=np.float64),
            "ecp_gp": np.asarray(_mesh.grid_points, dtype=np.float64),
            "dltmove": self._non_local_move == "dltmove",
            "want_force": _want_force,
            "want_stress": _want_stress,
            "complex": _complex,
        }

    def _prepare_position_grad_objects(self) -> tuple[Wavefunction_data, Hamiltonian_data]:
        """Return (Wavefunction_data, Hamiltonian_data) -- both UNMASKED.

        The position mask is applied IN-TRACE inside the local grad closures
        built in :py:meth:`run` (the ``_e_L_*_with_position_mask`` wrappers),
        so JAX does not backpropagate through the wf variational parameters
        during LRDMC force / stress calculation. An eager
        ``with_param_grad_mask(opt_*_param=False)`` here would be wiped at grad
        time (same JAX semantic as the param-grad path): the forces would stay
        numerically correct because ``accumulate_position_grad`` reduces to
        position-only contributions, but the reverse-mode tape would be built
        across the whole wf, which runs out of memory on B-spline geminals.
        Regression test:
        ``tests/test_jqmc_kernel/test_param_grad_mask_production_path.py::test_position_grad_mask_freezes_wf_params_*``.
        """
        return self._hamiltonian_data.wavefunction_data, self._hamiltonian_data

    def get_aT(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        _return_hydrostatic: bool = False,
    ) -> GFMCStressEstimate:
        """Symmetric stress tensor with jackknife statistics (LRDMC, PBC only).

        Mirror of :py:meth:`jqmc.jqmc_mcmc.MCMC.get_aT`. LRDMC stores one
        accumulator per rank (branching weights already absorbed into
        the per-step walker average), so the bin-block estimator omits
        the explicit walker-weighting present in :py:class:`MCMC`.

        Contract (fixed static ansatz): the stored stress_HF / stress_PP
        buffers are strain derivatives of the local energy and of
        ``ln|Psi_G|`` of the guiding function at a FIXED ansatz. The strain
        ``F = I + eps`` maps the cell, every nucleus and every walker electron
        as ``x -> F x``; the non-local ECP quadrature directions stay fixed, so
        each strained quadrature point lies on the sphere of the strained
        electron-nucleus distance; basis exponents and
        contraction coefficients, MO / geminal coefficients and Jastrow
        parameters are held fixed (comoving data such as plane-wave reciprocal
        indices and lattice-image index sets move with the cell). This tensor
        is therefore NOT the equation-of-state derivative of a re-optimized
        ground-state energy; the two differ by the response of the frozen
        parameters to the strain.

        Returns:
            ``(sigma_mean, sigma_std)`` each shape ``(3, 3)``,
            Hartree/Bohr^3. Molecular structures return zeros.
        """
        if not self._comput_stress:
            raise ValueError("Stress data were not computed; construct the GFMC driver with comput_stress=True.")

        # The early returns must honor _return_hydrostatic: get_P_hydrostatic
        # unpacks four values on every rank (rank != 0 of an MPI run included)
        # and also for a molecular structure. The zeros mirror the tensor convention:
        # the root rank owns the reduced observable.
        if not self._hamiltonian_data.structure_data.pbc_flag:
            zeros = (np.zeros((3, 3)), np.zeros((3, 3)))
            return (*zeros, 0.0, 0.0) if _return_hydrostatic else zeros
        if self._mpi_rank != 0:
            zeros = (np.zeros((3, 3)), np.zeros((3, 3)))
            return (*zeros, 0.0, 0.0) if _return_hydrostatic else zeros

        w_L = self._retained_w_L(num_mcmc_warmup_steps)
        e_L = self.e_L[num_mcmc_warmup_steps:]
        stress_HF = self.stress_HF[num_mcmc_warmup_steps:]
        stress_PP = self.stress_PP[num_mcmc_warmup_steps:]
        E_L_stress_PP = self.E_L_stress_PP[num_mcmc_warmup_steps:]

        M = stress_HF.shape[0]
        _require(w_L.shape == (M, 1), "")
        _require(e_L.shape == (M, 1), "")
        _require(stress_HF.shape == (M, 1, 3, 3), "")
        _require(stress_PP.shape == (M, 1, 3, 3), "")
        _require(E_L_stress_PP.shape == (M, 1, 3, 3), "")

        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_stress_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_HF), num_mcmc_bin_blocks, axis=0)
        w_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_PP), num_mcmc_bin_blocks, axis=0)
        w_L_E_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_stress_PP), num_mcmc_bin_blocks, axis=0)

        w_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split])))
        w_L_e_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split])))

        def _bin_tensor(splits: Sequence[npt.NDArray]) -> np.ndarray:
            tensor_sum = np.array([np.sum(arr, axis=0) for arr in splits])
            return tensor_sum.reshape(tensor_sum.shape[0] * tensor_sum.shape[1], 3, 3)

        w_L_stress_HF_binned = _bin_tensor(w_L_stress_HF_split)
        w_L_stress_PP_binned = _bin_tensor(w_L_stress_PP_split)
        w_L_E_L_stress_PP_binned = _bin_tensor(w_L_E_L_stress_PP_split)

        w_L_sum = float(np.sum(w_L_binned))
        # Keep the (possibly complex at a non-Gamma twist) dtype through the
        # jackknife; the physical-real projection happens on the
        # assembled sigma below. A float() cast here would drop Im(e_L) and
        # emit a ComplexWarning.
        w_L_e_L_sum = np.sum(w_L_e_L_binned)
        w_L_stress_HF_sum = np.sum(w_L_stress_HF_binned, axis=0)
        w_L_stress_PP_sum = np.sum(w_L_stress_PP_binned, axis=0)
        w_L_E_L_stress_PP_sum = np.sum(w_L_E_L_stress_PP_binned, axis=0)

        M_local = w_L_binned.size

        sigma_HF_jn = np.array(
            [(w_L_stress_HF_sum - w_L_stress_HF_binned[j]) / (w_L_sum - w_L_binned[j]) for j in range(M_local)]
        )
        sigma_Pulay_jn = 2.0 * np.array(
            [
                (
                    (w_L_E_L_stress_PP_sum - w_L_E_L_stress_PP_binned[j]) / (w_L_sum - w_L_binned[j])
                    - (
                        (w_L_e_L_sum - w_L_e_L_binned[j])
                        / (w_L_sum - w_L_binned[j])
                        * (w_L_stress_PP_sum - w_L_stress_PP_binned[j])
                        / (w_L_sum - w_L_binned[j])
                    )
                )
                for j in range(M_local)
            ]
        )

        # Real projection of the physical observable mirrors the
        # _debug twins and the get_aF force path (np.real(HF + Pulay)).
        sigma_jn = np.real(sigma_HF_jn + sigma_Pulay_jn)
        sigma_jn = 0.5 * (sigma_jn + sigma_jn.transpose(0, 2, 1))
        omega = float(np.abs(np.linalg.det(np.asarray(self._hamiltonian_data.structure_data._cell_jnp))))
        sigma_jn = sigma_jn / omega

        sigma_mean = np.mean(sigma_jn, axis=0)
        var_sigma = np.mean((sigma_jn - sigma_mean) ** 2, axis=0)
        sigma_std = np.sqrt((M_local - 1) * var_sigma)
        if _return_hydrostatic:
            # The hydrostatic pressure is a
            # linear functional of these same replicates, so its error bar must
            # be taken over P_jn = -tr(sigma_jn)/3 rather than by quadrature of
            # the three diagonal bars, which discards their covariance.
            p_hydro_jn = -np.trace(sigma_jn, axis1=1, axis2=2) / 3.0
            p_hydro_mean = float(np.mean(p_hydro_jn))
            p_hydro_std = float(np.sqrt((M_local - 1) * np.mean((p_hydro_jn - p_hydro_mean) ** 2)))
            return sigma_mean, sigma_std, p_hydro_mean, p_hydro_std
        return sigma_mean, sigma_std

    def get_P_hydrostatic(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Hydrostatic pressure with the covariance-aware (trace-first) error bar.

        Mirror of :py:meth:`jqmc.jqmc_mcmc.MCMC.get_P_hydrostatic`:
        the three normal stresses are strongly correlated, so quadrature of
        their separate bars is wrong; the jackknife is taken over the trace of
        the same replicates.

        Returns:
            tuple[float, float]: ``(P_hydro_mean, P_hydro_std)`` in Ha/bohr^3;
            ``(0.0, 0.0)`` for molecular structures.
        """
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return 0.0, 0.0
        _sigma_mean, _sigma_std, p_mean, p_std = self.get_aT(
            num_mcmc_warmup_steps, num_mcmc_bin_blocks, _return_hydrostatic=True
        )
        return p_mean, p_std

    def get_aP(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Normal-pressure components for LRDMC (mirror of :py:meth:`MCMC.get_aP`)."""
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3,)), np.zeros((3,))
        sigma_mean, sigma_std = self.get_aT(num_mcmc_warmup_steps, num_mcmc_bin_blocks)
        return -np.diag(sigma_mean), np.diag(sigma_std)

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        _observable: npt.NDArray | None = None,
    ) -> tuple[float, float, float, float]:
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks
            _observable (npt.NDArray | None): private; replaces the per-step local energy
                by this observable while keeping the same weights and binning. ``None``
                uses ``e_L``.

        Return:
            tuple[float, float, float, float]:
                The mean and standard error of the total energy and of its variance,
                estimated by the Jackknife method with the Args. (E_mean, E_std, Var_mean, Var_std).
        """
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if num_mcmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            logger.warning(f"num_mcmc_warmup_steps should be larger than {GFMC_MIN_WARMUP_STEPS}")
        if num_mcmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            logger.warning(f"num_mcmc_bin_blocks should be larger than {GFMC_MIN_BIN_BLOCKS}")

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        if num_mcmc_bin_blocks < self._mpi_size or self._mpi_size == 1:
            if self._mpi_rank == 0:
                # ``_observable`` (private) substitutes the per-step local quantity
                # while keeping the SAME weights, collect-step products and binning,
                # so ``e_L + dV_MPC`` inherits this jackknife exactly.
                e_L = (self.e_L if _observable is None else _observable)[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
                _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
                w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
                w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

                w_L_binned_local = np.array(w_L_binned)
                w_L_e_L_binned_local = np.array(w_L_e_L_binned)
                w_L_e_L2_binned_local = np.array(w_L_e_L2_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local)
                w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
                w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local)

                ## jackknie binned samples
                M_local = w_L_binned_local.size
                M_total = M_local

                # Take Re at the end of the binning
                # step so the energy observable and variance are real on the
                # complex Bloch path. w_L weights are already real (fixed-phase).
                # The variance subtracts the modulus square ``|<e_L>|**2`` of the
                # complex replicate, as MCMC.get_E does; ``(Re<e_L>)**2`` would add
                # ``Im(<e_L>)**2``. Bit-identical at Gamma where e_L is real.
                E_jackknife_binned_local = np.array(
                    [
                        (w_L_e_L_binned_local_sum - w_L_e_L_binned_local[m]) / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )
                abs_E_sq_local = np.abs(E_jackknife_binned_local) ** 2
                E_jackknife_binned_local = np.real(E_jackknife_binned_local)

                E2_jackknife_binned_local = np.real(
                    np.array(
                        [
                            (w_L_e_L2_binned_local_sum - w_L_e_L2_binned_local[m])
                            / (w_L_binned_local_sum - w_L_binned_local[m])
                            for m in range(M_local)
                        ]
                    )
                )

                Var_jackknife_binned_local = E2_jackknife_binned_local - abs_E_sq_local

                # Two-pass jackknife std (centered sum of squares) to avoid
                # catastrophic cancellation in <x^2> - <x>^2.

                # E: 1st pass -- mean, 2nd pass -- centered sum of squares
                E_mean = np.sum(E_jackknife_binned_local) / M_local
                E_var = np.sum((E_jackknife_binned_local - E_mean) ** 2) / M_local
                E_std = np.sqrt((M_local - 1) * E_var)

                # Var: 1st pass -- mean, 2nd pass -- centered sum of squares
                Var_mean = np.sum(Var_jackknife_binned_local) / M_total
                Var_var = np.sum((Var_jackknife_binned_local - Var_mean) ** 2) / M_total
                Var_std = np.sqrt((M_total - 1) * Var_var)

            else:
                E_mean = None
                E_std = None
                Var_mean = None
                Var_std = None

            # MPI broadcast. All four calls belong OUTSIDE the rank-0 branch, as
            # in ``get_E_mpc`` and ``get_aF``: a bcast is collective, so root and
            # non-root must issue the same calls in the same order. Guarding only
            # E_mean/E_std by the ``else`` made root issue two and everyone else
            # four -- root's Var_mean/Var_std paired with the others' E_mean/E_std
            # (so non-root ranks silently took the variance as the energy) and the
            # trailing two had no partner, leaving every non-root rank blocked
            # forever while rank 0 walked on to the checkpoint barrier. Reached
            # whenever ``num_mcmc_bin_blocks < mpi_size`` with more than one rank.
            if self._mpi_size > 1:
                E_mean = self._mpi_comm.bcast(E_mean, root=0)
                E_std = self._mpi_comm.bcast(E_std, root=0)
                Var_mean = self._mpi_comm.bcast(Var_mean, root=0)
                Var_std = self._mpi_comm.bcast(Var_std, root=0)

        else:
            if self._mpi_rank == 0:
                e_L = (self.e_L if _observable is None else _observable)[num_mcmc_warmup_steps:]
                e_L2 = self.e_L2[num_mcmc_warmup_steps:]
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
                _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
                w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
                w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))
            else:
                w_L_binned = []
                w_L_e_L_binned = []
                w_L_e_L2_binned = []

            if self._mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, self._mpi_size)
                w_L_e_L_binned_split = np.array_split(w_L_e_L_binned, self._mpi_size)
                w_L_e_L2_binned_split = np.array_split(w_L_e_L2_binned, self._mpi_size)
            else:
                w_L_binned_split = None
                w_L_e_L_binned_split = None
                w_L_e_L2_binned_split = None

            # scatter
            w_L_binned_local = self._mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_e_L_binned_local = self._mpi_comm.scatter(w_L_e_L_binned_split, root=0)
            w_L_e_L2_binned_local = self._mpi_comm.scatter(w_L_e_L2_binned_split, root=0)

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local)
            w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
            w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local)

            ## glolbal sum
            w_L_binned_global_sum = self._mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
            w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)
            w_L_e_L2_binned_global_sum = self._mpi_comm.allreduce(w_L_e_L2_binned_local_sum, op=MPI.SUM)

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

            # Take Re at the end of the binning step
            # so the energy observable and variance are real on the complex
            # Bloch path. Mirrors the non-MPI branch above (variance with the
            # modulus square ``|<e_L>|**2``); w_L weights are already real
            # (fixed-phase). Bit-identical at Gamma where e_L is real.
            E_jackknife_binned_local = np.array(
                [
                    (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )
            abs_E_sq_local = np.abs(E_jackknife_binned_local) ** 2
            E_jackknife_binned_local = np.real(E_jackknife_binned_local)

            E2_jackknife_binned_local = np.real(
                np.array(
                    [
                        (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )
            )

            Var_jackknife_binned_local = E2_jackknife_binned_local - abs_E_sq_local

            # Two-pass jackknife std (centered sum of squares) to avoid
            # catastrophic cancellation in <x^2> - <x>^2.

            # E: 1st pass -- global mean
            sum_E_global = self._mpi_comm.allreduce(np.sum(E_jackknife_binned_local), op=MPI.SUM)
            E_mean = sum_E_global / M_total

            # E: 2nd pass -- centered sum of squares (numerically stable)
            sumsq_centered_E_global = self._mpi_comm.allreduce(np.sum((E_jackknife_binned_local - E_mean) ** 2), op=MPI.SUM)
            E_var = sumsq_centered_E_global / M_total
            E_std = np.sqrt((M_total - 1) * E_var)

            # Var: 1st pass -- global mean
            sum_Var_global = self._mpi_comm.allreduce(np.sum(Var_jackknife_binned_local), op=MPI.SUM)
            Var_mean = sum_Var_global / M_total

            # Var: 2nd pass -- centered sum of squares
            sumsq_centered_Var_global = self._mpi_comm.allreduce(
                np.sum((Var_jackknife_binned_local - Var_mean) ** 2), op=MPI.SUM
            )
            Var_var = sumsq_centered_Var_global / M_total
            Var_std = np.sqrt((M_total - 1) * Var_var)

        # return
        return (E_mean, E_std, Var_mean, Var_std)

    def get_E_mpc(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """MPC-corrected total energy with the JOINT (covariance-aware) error bar.

        ``E_MPC = <e_L> + <dV_MPC>``: both are weighted ratio estimators over the
        same walkers, weights and collect-step products, so quadrature of their
        separate error bars is wrong. Sharing the denominator makes the
        joint jackknife exactly the jackknife of the summed numerator, so this
        delegates to :meth:`get_E` with the substituted observable.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks.

        Returns:
            tuple[float, float]: ``(E_mpc_mean, E_mpc_std)`` in Ha.

        Raises:
            ValueError: If the MPC observable was not enabled at construction.
        """
        if not self._mpc:
            logger.error("get_E_mpc requires mpc=True at construction.")
            raise ValueError("get_E_mpc requires the MPC observable to be enabled: construct the driver with mpc=True.")
        mean, std, _var_mean, _var_std = self.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            _observable=self.e_L + self.mpc_dV,
        )
        return (float(mean), float(std))

    def get_dE_mpc(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Estimate the MPC finite-size correction ``dE_mpc`` with jackknife error bars.

        The corrected total energy is ``E_MPC = E_Ewald + dE_mpc``, where ``E_Ewald`` is the
        value returned by :meth:`get_E` and ``dE_mpc = <V_ee^MPC - V_ee^Ewald>`` is the average
        of the per-branching-step difference stored by :meth:`run`. Both terms come out of ONE
        run evaluated on identical walker configurations, so they are strongly correlated and the
        error bar on the difference is much smaller than the statistical error of either Coulomb
        channel taken alone.

        This is an energy-only correction (``physical = no``): MPC never enters the sampled
        Hamiltonian, so the propagation, the branching weights, the forces (:meth:`get_aF`),
        the stresses (:meth:`get_aT`) and every parameter gradient remain purely Ewald.

        Mixed-estimator caveat: because the walkers are propagated under Ewald, ``<dV>`` is a
        mixed estimate ``<Psi_T| dV |Phi_0> / <Psi_T|Phi_0>``, not a pure one. It is correct to
        first order in ``dV``; the residual bias is second order in the (small) MPC - Ewald
        difference.

        The estimator is exactly the one used by :meth:`get_E`: the weighted ratio of sums
        ``sum(w_L * dV) / sum(w_L)`` over the same collect-step weights
        (``self._retained_w_L``), with the same binning and the same jackknife.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Returns:
            tuple[float, float]: ``(dE_mpc_mean, dE_mpc_std)`` in Ha.

        Raises:
            ValueError: If ``mpc`` was not enabled at construction, or if there are insufficient
                post-warmup samples to form the requested blocks.
        """
        if not self._mpc:
            logger.error("get_dE_mpc requires mpc=True at GFMC construction.")
            raise ValueError("get_dE_mpc requires the MPC observable to be enabled: construct the driver with mpc=True.")

        if num_mcmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            logger.warning(f"num_mcmc_warmup_steps should be larger than {GFMC_MIN_WARMUP_STEPS}")
        if num_mcmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            logger.warning(f"num_mcmc_bin_blocks should be larger than {GFMC_MIN_BIN_BLOCKS}")

        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        if num_mcmc_bin_blocks < self._mpi_size or self._mpi_size == 1:
            if self._mpi_rank == 0:
                mpc_dV = self.mpc_dV[num_mcmc_warmup_steps:]
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(mpc_dV.shape == (M, nw), f"mpc_dV shape {mpc_dV.shape} != ({M}, {nw})")
                # ``mpc_dV`` is real at any twist (difference of two real e-e
                # potentials), so none of the Re projections of ``get_E`` are needed.
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_mpc_dV_split = np.array_split(w_L * mpc_dV, num_mcmc_bin_blocks, axis=0)
                w_L_mpc_dV_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_mpc_dV_split]))

                w_L_binned_local = np.array(w_L_binned)
                w_L_mpc_dV_binned_local = np.array(w_L_mpc_dV_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local)
                w_L_mpc_dV_binned_local_sum = np.sum(w_L_mpc_dV_binned_local)

                ## jackknie binned samples
                M_local = w_L_binned_local.size

                dE_jackknife_binned_local = np.array(
                    [
                        (w_L_mpc_dV_binned_local_sum - w_L_mpc_dV_binned_local[m])
                        / (w_L_binned_local_sum - w_L_binned_local[m])
                        for m in range(M_local)
                    ]
                )

                # Two-pass jackknife std (centered sum of squares), as in ``get_E``.
                dE_mean = np.sum(dE_jackknife_binned_local) / M_local
                dE_var = np.sum((dE_jackknife_binned_local - dE_mean) ** 2) / M_local
                dE_std = np.sqrt((M_local - 1) * dE_var)

            else:
                dE_mean = None
                dE_std = None

            # MPI broadcast
            dE_mean = self._mpi_comm.bcast(dE_mean, root=0)
            dE_std = self._mpi_comm.bcast(dE_std, root=0)

        else:
            if self._mpi_rank == 0:
                mpc_dV = self.mpc_dV[num_mcmc_warmup_steps:]
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(mpc_dV.shape == (M, nw), f"mpc_dV shape {mpc_dV.shape} != ({M}, {nw})")
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_mpc_dV_split = np.array_split(w_L * mpc_dV, num_mcmc_bin_blocks, axis=0)
                w_L_mpc_dV_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_mpc_dV_split]))
            else:
                w_L_binned = []
                w_L_mpc_dV_binned = []

            if self._mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, self._mpi_size)
                w_L_mpc_dV_binned_split = np.array_split(w_L_mpc_dV_binned, self._mpi_size)
            else:
                w_L_binned_split = None
                w_L_mpc_dV_binned_split = None

            # scatter
            w_L_binned_local = self._mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_mpc_dV_binned_local = self._mpi_comm.scatter(w_L_mpc_dV_binned_split, root=0)

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_mpc_dV_binned_local = np.array(w_L_mpc_dV_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local)
            w_L_mpc_dV_binned_local_sum = np.sum(w_L_mpc_dV_binned_local)

            ## glolbal sum
            w_L_binned_global_sum = self._mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
            w_L_mpc_dV_binned_global_sum = self._mpi_comm.allreduce(w_L_mpc_dV_binned_local_sum, op=MPI.SUM)

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

            dE_jackknife_binned_local = np.array(
                [
                    (w_L_mpc_dV_binned_global_sum - w_L_mpc_dV_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )

            # Two-pass jackknife std (centered sum of squares), as in ``get_E``.

            # dE: 1st pass -- global mean
            sum_dE_global = self._mpi_comm.allreduce(np.sum(dE_jackknife_binned_local), op=MPI.SUM)
            dE_mean = sum_dE_global / M_total

            # dE: 2nd pass -- centered sum of squares (numerically stable)
            sumsq_centered_dE_global = self._mpi_comm.allreduce(np.sum((dE_jackknife_binned_local - dE_mean) ** 2), op=MPI.SUM)
            dE_var = sumsq_centered_dE_global / M_total
            dE_std = np.sqrt((M_total - 1) * dE_var)

        logger.devel(f"dE_mpc = {dE_mean} +- {dE_std} Ha.")

        # return
        return (dE_mean, dE_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Return the mean and std of the computed atomic forces.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]:
                The mean and std values of the computed atomic forces
                estimated by the Jackknife method with the Args.
                The dimention of the arrays is (N, 3).
        """
        if num_mcmc_bin_blocks < self._mpi_size or self._mpi_size == 1:
            if self._mpi_rank == 0:
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                e_L = self.e_L[num_mcmc_warmup_steps:]
                force_HF = self.force_HF[num_mcmc_warmup_steps:]
                force_PP = self.force_PP[num_mcmc_warmup_steps:]
                E_L_force_PP = self.E_L_force_PP[num_mcmc_warmup_steps:]
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                n_atoms = self._hamiltonian_data.structure_data.natom
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
                _require(force_HF.shape == (M, nw, n_atoms, 3), f"force_HF shape {force_HF.shape} != ({M}, {nw}, {n_atoms}, 3)")
                _require(force_PP.shape == (M, nw, n_atoms, 3), f"force_PP shape {force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)")
                _require(
                    E_L_force_PP.shape == (M, nw, n_atoms, 3),
                    f"E_L_force_PP shape {E_L_force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)",
                )
                # split and binning with multiple walkers
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
                w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
                w_L_E_L_force_PP_split = np.array_split(
                    np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0
                )

                # binned sum
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

                w_L_force_HF_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_HF_split])
                w_L_force_HF_binned_shape = (
                    w_L_force_HF_sum.shape[0] * w_L_force_HF_sum.shape[1],
                    w_L_force_HF_sum.shape[2],
                    w_L_force_HF_sum.shape[3],
                )
                w_L_force_HF_binned = list(w_L_force_HF_sum.reshape(w_L_force_HF_binned_shape))

                w_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_PP_split])
                w_L_force_PP_binned_shape = (
                    w_L_force_PP_sum.shape[0] * w_L_force_PP_sum.shape[1],
                    w_L_force_PP_sum.shape[2],
                    w_L_force_PP_sum.shape[3],
                )
                w_L_force_PP_binned = list(w_L_force_PP_sum.reshape(w_L_force_PP_binned_shape))

                w_L_E_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_E_L_force_PP_split])
                w_L_E_L_force_PP_binned_shape = (
                    w_L_E_L_force_PP_sum.shape[0] * w_L_E_L_force_PP_sum.shape[1],
                    w_L_E_L_force_PP_sum.shape[2],
                    w_L_E_L_force_PP_sum.shape[3],
                )
                w_L_E_L_force_PP_binned = list(w_L_E_L_force_PP_sum.reshape(w_L_E_L_force_PP_binned_shape))

                w_L_binned_local = np.array(w_L_binned)
                w_L_e_L_binned_local = np.array(w_L_e_L_binned)
                w_L_force_HF_binned_local = np.array(w_L_force_HF_binned)
                w_L_force_PP_binned_local = np.array(w_L_force_PP_binned)
                w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned)

                ## local sum
                w_L_binned_local_sum = np.sum(w_L_binned_local)
                w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
                w_L_force_HF_binned_local_sum = np.sum(w_L_force_HF_binned_local, axis=0)
                w_L_force_PP_binned_local_sum = np.sum(w_L_force_PP_binned_local, axis=0)
                w_L_E_L_force_PP_binned_local_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

                ## jackknie binned samples
                M_local = w_L_binned_local.size

                force_HF_jn_local = -1.0 * np.array(
                    [
                        (w_L_force_HF_binned_local_sum - w_L_force_HF_binned_local[j])
                        / (w_L_binned_local_sum - w_L_binned_local[j])
                        for j in range(M_local)
                    ]
                )

                force_Pulay_jn_local = -2.0 * np.array(
                    [
                        (
                            (w_L_E_L_force_PP_binned_local_sum - w_L_E_L_force_PP_binned_local[j])
                            / (w_L_binned_local_sum - w_L_binned_local[j])
                            - (
                                (w_L_e_L_binned_local_sum - w_L_e_L_binned_local[j])
                                / (w_L_binned_local_sum - w_L_binned_local[j])
                                * (w_L_force_PP_binned_local_sum - w_L_force_PP_binned_local[j])
                                / (w_L_binned_local_sum - w_L_binned_local[j])
                            )
                        )
                        for j in range(M_local)
                    ]
                )

                # Re-project the Hellmann-Feynman plus Pulay covariance.
                # The Pulay branch involves ``e_L * (dlnPsi)*`` which is complex
                # per sample on twist k != 0; the outer Re projection is taken
                # here so the std accumulator below sees a real-valued sample
                # distribution. Bit-identical at Gamma.
                force_jn_local = np.real(force_HF_jn_local + force_Pulay_jn_local)

                # Two-pass jackknife std (centered sum of squares) to avoid
                # catastrophic cancellation in <x^2> - <x>^2.
                mean_force_global = np.sum(force_jn_local, axis=0) / M_local
                var_force_global = np.sum((force_jn_local - mean_force_global) ** 2, axis=0) / M_local

                ## mean and std
                force_mean = mean_force_global
                force_std = np.sqrt((M_local - 1) * var_force_global)

            else:
                force_mean = None
                force_std = None

            # broadcast the results
            force_mean = self._mpi_comm.bcast(force_mean, root=0)
            force_std = self._mpi_comm.bcast(force_std, root=0)

        else:
            if self._mpi_rank == 0:
                w_L = self._retained_w_L(num_mcmc_warmup_steps)
                e_L = self.e_L[num_mcmc_warmup_steps:]
                force_HF = self.force_HF[num_mcmc_warmup_steps:]
                force_PP = self.force_PP[num_mcmc_warmup_steps:]
                E_L_force_PP = self.E_L_force_PP[num_mcmc_warmup_steps:]
                M = self.mcmc_counter - num_mcmc_warmup_steps
                nw = 1  # GFMC observables are reduced across walkers
                n_atoms = self._hamiltonian_data.structure_data.natom
                _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
                _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
                _require(force_HF.shape == (M, nw, n_atoms, 3), f"force_HF shape {force_HF.shape} != ({M}, {nw}, {n_atoms}, 3)")
                _require(force_PP.shape == (M, nw, n_atoms, 3), f"force_PP shape {force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)")
                _require(
                    E_L_force_PP.shape == (M, nw, n_atoms, 3),
                    f"E_L_force_PP shape {E_L_force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)",
                )
                w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
                w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
                w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
                w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
                w_L_E_L_force_PP_split = np.array_split(
                    np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0
                )

                # binned sum
                w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
                w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

                w_L_force_HF_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_HF_split])
                w_L_force_HF_binned_shape = (
                    w_L_force_HF_sum.shape[0] * w_L_force_HF_sum.shape[1],
                    w_L_force_HF_sum.shape[2],
                    w_L_force_HF_sum.shape[3],
                )
                w_L_force_HF_binned = list(w_L_force_HF_sum.reshape(w_L_force_HF_binned_shape))

                w_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_PP_split])
                w_L_force_PP_binned_shape = (
                    w_L_force_PP_sum.shape[0] * w_L_force_PP_sum.shape[1],
                    w_L_force_PP_sum.shape[2],
                    w_L_force_PP_sum.shape[3],
                )
                w_L_force_PP_binned = list(w_L_force_PP_sum.reshape(w_L_force_PP_binned_shape))

                w_L_E_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_E_L_force_PP_split])
                w_L_E_L_force_PP_binned_shape = (
                    w_L_E_L_force_PP_sum.shape[0] * w_L_E_L_force_PP_sum.shape[1],
                    w_L_E_L_force_PP_sum.shape[2],
                    w_L_E_L_force_PP_sum.shape[3],
                )
                w_L_E_L_force_PP_binned = list(w_L_E_L_force_PP_sum.reshape(w_L_E_L_force_PP_binned_shape))

            else:
                w_L_binned = []
                w_L_e_L_binned = []
                w_L_force_HF_binned = []
                w_L_force_PP_binned = []
                w_L_E_L_force_PP_binned = []

            if self._mpi_rank == 0:
                w_L_binned_split = np.array_split(w_L_binned, self._mpi_size)
                w_L_e_L_binned_split = np.array_split(w_L_e_L_binned, self._mpi_size)
                w_L_force_HF_binned_split = np.array_split(w_L_force_HF_binned, self._mpi_size)
                w_L_force_PP_binned_split = np.array_split(w_L_force_PP_binned, self._mpi_size)
                w_L_E_L_force_PP_binned_split = np.array_split(w_L_E_L_force_PP_binned, self._mpi_size)
            else:
                w_L_binned_split = None
                w_L_e_L_binned_split = None
                w_L_force_HF_binned_split = None
                w_L_force_PP_binned_split = None
                w_L_E_L_force_PP_binned_split = None

            # scatter
            w_L_binned_local = self._mpi_comm.scatter(w_L_binned_split, root=0)
            w_L_e_L_binned_local = self._mpi_comm.scatter(w_L_e_L_binned_split, root=0)
            w_L_force_HF_binned_local = self._mpi_comm.scatter(w_L_force_HF_binned_split, root=0)
            w_L_force_PP_binned_local = self._mpi_comm.scatter(w_L_force_PP_binned_split, root=0)
            w_L_E_L_force_PP_binned_local = self._mpi_comm.scatter(w_L_E_L_force_PP_binned_split, root=0)

            # convert to numpy arrays
            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
            w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
            w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

            ## local sum
            w_L_binned_local_sum = np.sum(w_L_binned_local)
            w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
            w_L_force_HF_binned_local_sum = np.sum(w_L_force_HF_binned_local, axis=0)
            w_L_force_PP_binned_local_sum = np.sum(w_L_force_PP_binned_local, axis=0)
            w_L_E_L_force_PP_binned_local_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

            ## glolbal sum
            ### mpi allreduce (scalar)
            w_L_binned_global_sum = self._mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
            w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)

            ### mpi Allreduce (array)
            w_L_force_HF_binned_global_sum = np.empty_like(w_L_force_HF_binned_local_sum)
            w_L_force_PP_binned_global_sum = np.empty_like(w_L_force_PP_binned_local_sum)
            w_L_E_L_force_PP_binned_global_sum = np.empty_like(w_L_E_L_force_PP_binned_local_sum)
            self._mpi_comm.Allreduce(
                [w_L_force_HF_binned_local_sum, MPI.DOUBLE], [w_L_force_HF_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )
            self._mpi_comm.Allreduce(
                [w_L_force_PP_binned_local_sum, MPI.DOUBLE], [w_L_force_PP_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )
            self._mpi_comm.Allreduce(
                [w_L_E_L_force_PP_binned_local_sum, MPI.DOUBLE], [w_L_E_L_force_PP_binned_global_sum, MPI.DOUBLE], op=MPI.SUM
            )

            ## jackknie binned samples
            M_local = w_L_binned_local.size
            M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

            force_HF_jn_local = -1.0 * np.array(
                [
                    (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    for j in range(M_local)
                ]
            )

            force_Pulay_jn_local = -2.0 * np.array(
                [
                    (
                        (w_L_E_L_force_PP_binned_global_sum - w_L_E_L_force_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        - (
                            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                            * (w_L_force_PP_binned_global_sum - w_L_force_PP_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                        )
                    )
                    for j in range(M_local)
                ]
            )

            # Take Re on the combined Hellmann-Feynman plus Pulay
            # jackknife sample so the force observable is real on the complex
            # Bloch path. Mirrors the non-MPI branch above. Bit-identical at
            # Gamma where both pieces are real.
            force_jn_local = np.real(force_HF_jn_local + force_Pulay_jn_local)

            # Two-pass jackknife std (centered sum of squares) to avoid
            # catastrophic cancellation in <x^2> - <x>^2.

            # 1st pass -- global mean
            sum_force_local = np.sum(force_jn_local, axis=0)
            sum_force_global = np.empty_like(sum_force_local)
            self._mpi_comm.Allreduce([sum_force_local, MPI.DOUBLE], [sum_force_global, MPI.DOUBLE], op=MPI.SUM)
            mean_force_global = sum_force_global / M_total

            # 2nd pass -- centered sum of squares (numerically stable)
            sumsq_centered_force_local = np.sum((force_jn_local - mean_force_global) ** 2, axis=0)
            sumsq_centered_force_global = np.empty_like(sumsq_centered_force_local)
            self._mpi_comm.Allreduce(
                [sumsq_centered_force_local, MPI.DOUBLE], [sumsq_centered_force_global, MPI.DOUBLE], op=MPI.SUM
            )
            var_force_global = sumsq_centered_force_global / M_total

            ## mean and std
            force_mean = mean_force_global
            force_std = np.sqrt((M_total - 1) * var_force_global)

        return (force_mean, force_std)


def _validate_gfmc_positive(name: str, value: Any, *, comm: MPI.Comm | None, driver: str) -> None:
    """Require a positive finite GFMC parameter, collectively on restart."""
    try:
        valid = not isinstance(value, (bool, np.bool_)) and math.isfinite(value) and value > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    error = None if valid else f"{name} must be finite and > 0 (got {value!r})."
    if comm is not None and comm.Get_size() > 1:
        for rank, rank_error in enumerate(comm.allgather(error)):
            if rank_error is not None:
                raise ValueError(f"{driver} checkpoint load sampling_rank={rank}: {rank_error}")
    elif error is not None:
        raise ValueError(error)


def _validate_gfmc_tau(tau: float, *, comm: MPI.Comm | None = None) -> None:
    """Require a positive finite projection interval, collectively on restart."""
    _validate_gfmc_positive("tau", tau, comm=comm, driver="GFMC_t")


def _validate_gfmc_alat(alat: float, *, comm: MPI.Comm | None = None, driver: str = "GFMC") -> None:
    """Require a positive finite lattice spacing, collectively on restart.

    The lattice-regularized kinetic term scales as 1/alat**2: ``alat = 0`` divides
    by zero on XLA and stalls the native projection (issue #657), and a negative
    spacing is outside the physical domain.
    """
    _validate_gfmc_positive("alat", alat, comm=comm, driver=driver)


class GFMC_t(_GFMCBase):
    """GFMC class.

    GFMC class. Running GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_gfmc_collect_steps (int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        tau (float):
            finite, strictly positive imaginary time between branchings (Ha^-1)
        alat (float):
            finite, strictly positive discretized grid length (bohr)
        random_discretized_mesh (bool):
            Flag for the random discretization mesh in the kinetic part and in the non-local part of ECPs.
            Valid both for all-electron and ECP calculations.
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. All-electron calculations, do not specify this value.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
        comput_stress (bool):
            if True, compute and retain periodic stress observables. Defaults to False.
        epsilon_PW (float):
            Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is
            regularized near the nodal surface using ``f_epsilon`` from
            `Pathak and Wagner (2020) <https://doi.org/10.1063/5.0004008>`_.
            Default is 0.0 (no regularization).
        use_swct (bool):
            if True, apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
            Default is False for GFMC.
        mpc (bool):
            if True, evaluate the passive MPC finite-size correction: on each stored branching step
            ``dV = V_ee^MPC - V_ee^Ewald`` is evaluated on the same walker configurations (see
            :meth:`get_dE_mpc`); MPC never enters the propagation, the weights or the forces.
            Requires a periodic ``hamiltonian_data``. Default is False.
        mpc_cutoff (float):
            reciprocal-space kinetic-energy cutoff (Ha) of the MPC FFT grid. Ignored when ``mpc``
            is False. Default is ``MPC_CUTOFF_HA``.
        mpi_comm_in (MPI.Comm | None):
            sampling communicator of this instance. ``None`` uses the module global ``mpi_comm``
            (``COMM_WORLD``); the twist-averaged driver passes one communicator per twist.
        world_comm_in (MPI.Comm | None):
            all-rank communicator. ``None`` uses ``mpi_comm_in``.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        num_walkers: int = 40,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        tau: float = 0.1,
        alat: float = 0.1,
        random_discretized_mesh: bool = True,
        non_local_move: NonLocalMove = "tmove",
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        epsilon_PW: float = 0.0,
        use_swct: bool = False,
        mpc: bool = False,
        mpc_cutoff: float = MPC_CUTOFF_HA,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Initialize a GFMC_t driver and set up walker state."""
        _validate_gfmc_tau(tau)
        self._tau = tau
        super().__init__(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            alat=alat,
            random_discretized_mesh=random_discretized_mesh,
            non_local_move=non_local_move,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
            epsilon_PW=epsilon_PW,
            use_swct=use_swct,
            mpc=mpc,
            mpc_cutoff=mpc_cutoff,
            mpi_comm_in=mpi_comm_in,
            world_comm_in=world_comm_in,
        )

    def _init_attributes_class_specific(self) -> None:
        # average projection counter (GFMC_t only)
        self._stored_average_projection_counter = np.zeros((0,), dtype=np.float64)

    def __validate_stored_shapes(self) -> None:
        """Assert that all stored observable arrays have consistent shapes."""
        ns = self._mcmc_counter
        n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        n_atoms = self._hamiltonian_data.structure_data.natom

        expected = {
            "e_L": ((ns, 1), self._stored_e_L),
            "e_L2": ((ns, 1), self._stored_e_L2),
            "w_L": ((ns, 1), self._stored_w_L),
            "average_projection_counter": ((ns,), self._stored_average_projection_counter),
        }
        if self._mpc:
            expected["mpc_dV"] = ((ns, 1), self._stored_mpc_dV)
        if self._comput_position_deriv:
            expected.update(
                {
                    "force_HF": ((ns, 1, n_atoms, 3), self._stored_force_HF),
                    "force_PP": ((ns, 1, n_atoms, 3), self._stored_force_PP),
                    "E_L_force_PP": ((ns, 1, n_atoms, 3), self._stored_E_L_force_PP),
                }
            )
        if self._comput_stress:
            expected.update(
                {
                    "stress_HF": ((ns, 1, 3, 3), self._stored_stress_HF),
                    "stress_PP": ((ns, 1, 3, 3), self._stored_stress_PP),
                    "E_L_stress_PP": ((ns, 1, 3, 3), self._stored_E_L_stress_PP),
                }
            )
        for name, (shape, arr) in expected.items():
            _require(arr.shape == shape, f"GFMC_t stored shape mismatch: {name} expected {shape}, got {arr.shape}")

    # ------------------------------------------------------------------
    # HDF5 checkpoint save / load
    # ------------------------------------------------------------------

    def save_to_hdf5(self, filepath: str) -> None:
        """Write this rank's GFMC_t state to a temporary per-rank HDF5 file."""
        super().save_to_hdf5(
            filepath,
            driver_type="GFMC_t",
            extra_config={"tau": self._tau},
            extra_observables={"average_projection_counter": self._stored_average_projection_counter},
        )

    _RESTART_CLASS_CONFIG_KEYS = ("tau",)

    @classmethod
    def _class_restart_history_errors(cls, obs: Mapping[str, Any], counter: int, rank: int) -> list[tuple[str, str]]:
        """``average_projection_counter`` is stored on every rank, one row per step."""
        from ._checkpoint import history_length_errors

        return history_length_errors(obs, ("average_projection_counter",), counter, counter)

    @classmethod
    def _validate_restored_domains(cls, cfg: Mapping[str, Any], comm: MPI.Comm) -> None:
        """Collectively validate the restored lattice spacing and projection interval."""
        super()._validate_restored_domains(cfg, comm)
        _validate_gfmc_tau(cfg.get("tau"), comm=comm)

    def _load_class_specific(self, cfg: Mapping[str, Any], obs: Mapping[str, Any]) -> None:
        """Restore GFMC_t class-specific config + observables."""
        self._tau = cfg["tau"]
        self._stored_average_projection_counter = self._load_obs(obs.get("average_projection_counter"), np.zeros((0,)))

    # hamiltonian
    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Return hamiltonian_data."""
        return self._hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data: Hamiltonian_data) -> None:
        """Set hamiltonian_data."""
        if self._comput_position_deriv or self._comput_stress:
            self._hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=True, coords=True))
        else:
            self._hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=False))
        self._init_attributes()

    # collecting factor
    @property
    def num_gfmc_collect_steps(self) -> int:
        """Return num_gfmc_collect_steps."""
        return self._num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps: int) -> None:
        """Set num_gfmc_collect_steps."""
        self._num_gfmc_collect_steps = num_gfmc_collect_steps

    @property
    def average_projection_counter(self) -> npt.NDArray:
        """Return the stored average projection counter array."""
        return np.asarray(self._stored_average_projection_counter)

    def run(self, num_mcmc_steps: int = 50, max_time: int = 86400) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_mcmc_steps (int): number of branching steps (reconfigurations of walkers).
            max_time (int): maximum time in sec.
        """
        _check_mpi4py_jax_distribution_consistency()

        # set timer
        timer_projection_init = 0.0
        timer_projection_total = 0.0
        timer_observable = 0.0
        timer_mpi_barrier = 0.0
        timer_collection = 0.0
        timer_stress = 0.0
        timer_reconfiguration = 0.0
        self._mpi_comm.Barrier()
        gfmc_total_start = time.perf_counter()

        # toml(control) filename
        toml_filename = "external_control_gfmc.toml"

        # create the single per-invocation control file (world root owns it;
        # see _init_external_control_file for the twist-average rationale).
        _init_external_control_file(toml_filename, self._world_comm)
        self._mpi_comm.Barrier()

        # Native GFMC_t projection-interval chunk gate. Production uses the native
        # chain; deterministic fixed-input comparisons call the Rust entry point
        # directly. When
        # engaged, one native run_gfmc_projection_t call per walker per branching
        # step replaces the projection lax.while_loop; the JAX projection compiles
        # and per-step A_inv reseeds are skipped. Branching / reconfiguration /
        # binning stay Python (MPI collectives), unchanged.
        _rs_gfmc = self._resolve_native_gfmc_chunk("t")
        _rs_gfmc_chunk = _rs_gfmc is not None
        _rs_mpc_observable = self._mpc and _rust_cpu_backend_active()
        if _rs_gfmc_chunk:
            _rs_obs_ctx = _rs_gfmc["ctx"]
            _rs_ecp_w = _rs_gfmc["ecp_w"]
            _rs_ecp_gp = _rs_gfmc["ecp_gp"]
            _rs_dltmove = _rs_gfmc["dltmove"]
            _rs_want_force = _rs_gfmc["want_force"]
            _rs_want_stress = _rs_gfmc["want_stress"]
            _rs_complex = _rs_gfmc["complex"]

        if _rs_gfmc_chunk:
            # The native interval owns and advances its determinant inverse;
            # no Python/JAX inverse crosses this boundary. Avoid an otherwise
            # unused 128-walker SVD batch in the timed run setup.
            self.__latest_A_old_inv = None
        else:
            self.__latest_A_old_inv = _seed_gfmc_t_geminal_inverse(
                self._hamiltonian_data.wavefunction_data.geminal_data, self._latest_r_up_carts, self._latest_r_dn_carts
            )

        # projection function.
        # _generate_rotation_matrix_t and _projection_t_core live at module scope.

        # Shared core of non-streaming and streaming GFMC_t projection. Two thin
        # wrappers (`_projection_t` non-streaming / `_projection_t_streaming`) below
        # supply the per-electron continuum kinetic energy and the optional
        # ``j3_state`` (None for non-streaming, ``kinetic_state.j3_state`` for
        # streaming) and call this body. Mirrors the ``_body_step_core`` /
        # ``_body_fun_n`` / ``_body_fun_n_streaming`` split in GFMC_n.
        # _projection_t_core hoisted to module scope; the wrappers below call the module fn.

        # Note: This jit drastically accelarates the computation!!
        @partial(jit, static_argnums=(10, 11, 12))
        def _projection_t(
            projection_counter: int,
            tau_left: float,
            w_L: float,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            A_old_inv: jnpt.ArrayLike,
            jax_PRNG_key: jnpt.ArrayLike,
            interval_RT: jnpt.ArrayLike,
            interval_ecp_RT: jnpt.ArrayLike,
            pot_state: GFMCIntervalPotState,
            random_discretized_mesh: bool,
            non_local_move: NonLocalMove,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ) -> GFMCProjectionTLoopState:
            """Non-streaming GFMC_t projection step (no streaming kinetic-energy state).

            Recomputes the per-electron continuum kinetic energy fresh each step
            via :func:`compute_kinetic_energy_all_elements_fast_update` and
            delegates the rest of the body to :func:`_projection_t_core`.
            """
            ke_up, ke_dn = compute_kinetic_energy_all_elements_fast_update(
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                geminal_inverse=A_old_inv,
            )
            (e_L, pc, tl, wL, ru, rd, Ainv, key, RT, ecp_RT, _has_up, _up_idx, _dn_idx, pot_new) = _projection_t_core(
                projection_counter,
                tau_left,
                w_L,
                r_up_carts,
                r_dn_carts,
                A_old_inv,
                jax_PRNG_key,
                ke_up,
                ke_dn,
                None,  # j3_state
                interval_RT,
                interval_ecp_RT,
                random_discretized_mesh,
                non_local_move,
                alat,
                hamiltonian_data,
                pot_state=pot_state,
            )
            return (e_L, pc, tl, wL, ru, rd, Ainv, key, RT, ecp_RT, pot_new)

        # Union signature (candidate mesh states + interval pot state + the
        # per-walker per-hop refresh count): static args are
        # random_discretized_mesh / non_local_move / alat.
        @partial(jit, static_argnums=(14, 15, 16))
        def _projection_t_streaming(
            projection_counter: int,
            tau_left: float,
            w_L: float,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            A_old_inv: jnpt.ArrayLike,
            jax_PRNG_key: jnpt.ArrayLike,
            kinetic_state: Kinetic_streaming_state,
            kinetic_mesh_state: Discretized_kinetic_mesh_streaming_state,
            ecp_mesh_state: Ecp_candidate_mesh_streaming_state | None,
            interval_RT: jnpt.ArrayLike,
            interval_ecp_RT: jnpt.ArrayLike,
            pot_state: GFMCIntervalPotState,
            interval_refreshes: jax.Array,
            random_discretized_mesh: bool,
            non_local_move: NonLocalMove,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ) -> GFMCProjectionTStreamingState:
            """Streaming GFMC_t projection step.

            Reads per-electron kinetic energies from ``kinetic_state`` instead
            of recomputing them, threads ``kinetic_state.j3_state`` into the
            fast-update kernels (discretized kinetic / non-local ECP / rank-1
            Jastrow ratio), delegates the body to :func:`_projection_t_core`,
            then advances the kinetic streaming state to the post-step
            ``(r_up_new, r_dn_new, A_new_inv)``.

            Dispatch is Python-static at the ``run()`` entry point. When
            ``tau_left <= 0.0`` the move is suppressed (positions unchanged,
            A_new_inv == A_old_inv), so the advance is a numerical no-op.

            The kinetic / ECP mesh states are initialized by the UNROLLED
            substep-0 call (``_projection_t_streaming_first``, outside the
            while_loop) and advanced UNCONDITIONALLY here; a traced
            ``first_substep ? init : advance`` select would pay init AND
            advance on every substep under the walker vmap.
            """
            ke_up, ke_dn = _kinetic_energy_from_streaming_state(kinetic_state)
            (e_L, pc, tl, wL, ru, rd, Ainv, key, RT, ecp_RT, has_up, up_idx, dn_idx, pot_new) = _projection_t_core(
                projection_counter,
                tau_left,
                w_L,
                r_up_carts,
                r_dn_carts,
                A_old_inv,
                jax_PRNG_key,
                ke_up,
                ke_dn,
                kinetic_state.j3_state,
                interval_RT,
                interval_ecp_RT,
                random_discretized_mesh,
                non_local_move,
                alat,
                hamiltonian_data,
                det_ratio_state=kinetic_state.det_state,
                kinetic_mesh_state=kinetic_mesh_state,
                ecp_mesh_state=ecp_mesh_state,
                pot_state=pot_state,
            )
            moved_spin_is_up, moved_index = _streaming_mover(has_up, up_idx, dn_idx, r_up_carts.shape[0], r_dn_carts.shape[0])

            def _advance_kinetic(A_new_inv: jax.Array) -> Kinetic_streaming_state:
                return _advance_kinetic_energy_all_elements_streaming_state(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    state=kinetic_state,
                    moved_spin_is_up=moved_spin_is_up,
                    moved_index=moved_index,
                    r_up_carts_new=ru,
                    r_dn_carts_new=rd,
                    A_new_inv=A_new_inv,
                )

            kinetic_state_new = _advance_kinetic(Ainv)
            # Per-hop screen of the carried inverse against the
            # post-hop det caches; a refreshed walker re-runs the det
            # grad/lap advance with the rebuilt inverse.
            Ainv, kinetic_state_new, refreshed = _interval_inverse_screen(
                hamiltonian_data.wavefunction_data.geminal_data,
                kinetic_state_new.det_state,
                Ainv,
                kinetic_state_new,
                _advance_kinetic,
                axis_name=_GFMC_WALKER_AXIS,
            )
            interval_refreshes = interval_refreshes + refreshed.astype(jnp.int32)
            # Unconditional advance: the interval-start init happens OUTSIDE
            # the while_loop (GFMC_n structure), so no first-substep selection
            # remains here.  On a suppressed move (tau exhausted, old == new
            # positions) the advance re-evaluates the clamped electron's rows
            # at unchanged positions -- the existing numerical no-op
            # convention (identical to what every post-first substep already
            # did).  The fully-polarized Python-static shape guards inside the
            # advance kernels are unaffected (they dispatch on candidate array
            # shapes, not on the substep index).
            kinetic_mesh_state_new = _advance_discretized_kinetic_mesh_streaming_state(
                wavefunction_data=hamiltonian_data.wavefunction_data,
                state=kinetic_mesh_state,
                moved_spin_is_up=moved_spin_is_up,
                moved_index=moved_index,
                r_up_carts_old=r_up_carts,
                r_dn_carts_old=r_dn_carts,
                r_up_carts_new=ru,
                r_dn_carts_new=rd,
            )
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                ecp_mesh_state_new = _advance_ecp_candidate_mesh_streaming_state(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    state=ecp_mesh_state,
                    moved_spin_is_up=moved_spin_is_up,
                    moved_index=moved_index,
                    r_up_carts_old=r_up_carts,
                    r_dn_carts_old=r_dn_carts,
                    r_up_carts_new=ru,
                    r_dn_carts_new=rd,
                    RT=ecp_RT,
                )
            else:
                ecp_mesh_state_new = None
            return (
                e_L,
                pc,
                tl,
                wL,
                ru,
                rd,
                Ainv,
                kinetic_state_new,
                kinetic_mesh_state_new,
                ecp_mesh_state_new,
                key,
                RT,
                ecp_RT,
                pot_new,
                interval_refreshes,
            )

        @partial(jit, static_argnums=(12, 13, 14))
        def _projection_t_streaming_first(
            projection_counter: int,
            tau_left: float,
            w_L: float,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            A_old_inv: jnpt.ArrayLike,
            jax_PRNG_key: jnpt.ArrayLike,
            kinetic_state: Kinetic_streaming_state,
            interval_RT: jnpt.ArrayLike,
            interval_ecp_RT: jnpt.ArrayLike,
            pot_state: GFMCIntervalPotState,
            interval_refreshes: jax.Array,
            random_discretized_mesh: bool,
            non_local_move: NonLocalMove,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ) -> GFMCProjectionTStreamingState:
            """Substep-0 of a projection interval, unrolled OUTSIDE the while_loop.

            Runs the FULL kinetic/ECP kernels via the Python-static
            ``kinetic_mesh_state is None`` branches of ``_projection_t_core``
            then initializes both mesh states from the POST-hop positions and
            the substep-0 rotation.  Because this
            step is unrolled eagerly, the loop body never carries a
            first-substep dispatch: no dual kernel execution anywhere.
            ``pot_state`` is consumed and advanced exactly like every other
            substep.
            """
            ke_up, ke_dn = _kinetic_energy_from_streaming_state(kinetic_state)
            (e_L, pc, tl, wL, ru, rd, Ainv, key, RT, ecp_RT, has_up, up_idx, dn_idx, pot_new) = _projection_t_core(
                projection_counter,
                tau_left,
                w_L,
                r_up_carts,
                r_dn_carts,
                A_old_inv,
                jax_PRNG_key,
                ke_up,
                ke_dn,
                kinetic_state.j3_state,
                interval_RT,
                interval_ecp_RT,
                random_discretized_mesh,
                non_local_move,
                alat,
                hamiltonian_data,
                det_ratio_state=kinetic_state.det_state,
                kinetic_mesh_state=None,
                ecp_mesh_state=None,
                pot_state=pot_state,
            )
            moved_spin_is_up, moved_index = _streaming_mover(has_up, up_idx, dn_idx, r_up_carts.shape[0], r_dn_carts.shape[0])

            def _advance_kinetic(A_new_inv: jax.Array) -> Kinetic_streaming_state:
                return _advance_kinetic_energy_all_elements_streaming_state(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    state=kinetic_state,
                    moved_spin_is_up=moved_spin_is_up,
                    moved_index=moved_index,
                    r_up_carts_new=ru,
                    r_dn_carts_new=rd,
                    A_new_inv=A_new_inv,
                )

            kinetic_state_new = _advance_kinetic(Ainv)
            # Per-hop screen of the carried inverse against the
            # post-hop det caches; a refreshed walker re-runs the det
            # grad/lap advance with the rebuilt inverse.
            Ainv, kinetic_state_new, refreshed = _interval_inverse_screen(
                hamiltonian_data.wavefunction_data.geminal_data,
                kinetic_state_new.det_state,
                Ainv,
                kinetic_state_new,
                _advance_kinetic,
                axis_name=_GFMC_WALKER_AXIS,
            )
            interval_refreshes = interval_refreshes + refreshed.astype(jnp.int32)
            kinetic_mesh_state_new = _init_discretized_kinetic_mesh_streaming_state(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=ru,
                r_dn_carts=rd,
                RT=RT,
            )
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                ecp_mesh_state_new = _init_ecp_candidate_mesh_streaming_state(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=ru,
                    r_dn_carts=rd,
                    RT=ecp_RT,
                )
            else:
                ecp_mesh_state_new = None
            return (
                e_L,
                pc,
                tl,
                wL,
                ru,
                rd,
                Ainv,
                kinetic_state_new,
                kinetic_mesh_state_new,
                ecp_mesh_state_new,
                key,
                RT,
                ecp_RT,
                pot_new,
                interval_refreshes,
            )

        # The determinant streaming path (consuming ``kinetic_state.det_state``
        # in the LRDMC mesh kernels and the GFMC inv-update) is always used.
        use_streaming = True

        # Avoid per-run XLA compilation: do not warm the dead non-streaming path
        # while streaming is enabled; construct AOT inputs with the same dtypes as
        # the branching-step inputs; and cache the jitted while-loop drivers by
        # Hamiltonian object identity so later run() calls reuse their executables.
        _driver_cache = getattr(self, "_projection_driver_cache", None)
        _drivers_cached = _driver_cache is not None and _driver_cache[0] is self._hamiltonian_data

        # projection compilation.
        start_init = time.perf_counter()
        projection_counter_list = jnp.zeros(self._num_walkers, dtype=jnp.float64)
        tau_left_list = jnp.full(self._num_walkers, self._tau, dtype=jnp.float64)
        w_L_list = jnp.ones(self._num_walkers, dtype=jnp.float64)
        interval_RT_init = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (self._num_walkers, 3, 3))
        interval_ecp_RT_init = interval_RT_init
        # The native chunk never dispatches the JAX projection graph, so skip its
        # (pure compile-time) warmups entirely.
        if not _drivers_cached and not use_streaming and not _rs_gfmc_chunk:
            # Warm the single-projection non-streaming path only when
            # that path is the one the branching loop dispatches.
            logger.info("Start compilation of the GFMC projection funciton.")
            logger.info("  Compilation is in progress...")
            _pot_state_init_compile = vmap(_gfmc_build_interval_pot_state, in_axes=(None, 0, 0, None))(
                self._hamiltonian_data,
                self._latest_r_up_carts,
                self._latest_r_dn_carts,
                self._alat,
            )
            (_, _, _, _, _, _, _, _, _, _, _) = vmap(
                _projection_t,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
            )(
                projection_counter_list,
                tau_left_list,
                w_L_list,
                self._latest_r_up_carts,
                self._latest_r_dn_carts,
                self.__latest_A_old_inv,
                self._jax_PRNG_key_list,
                interval_RT_init,
                interval_ecp_RT_init,
                _pot_state_init_compile,
                self._random_discretized_mesh,
                self._non_local_move,
                self._alat,
                self._hamiltonian_data,
            )
        if not _drivers_cached and use_streaming and not _rs_gfmc_chunk:
            # Fresh kinetic state for the streaming AOT lower below. No eager
            # ``vmap(_projection_t_streaming)`` execution here: the standalone
            # single-projection executable would be inlined inside the
            # while-loop driver and never reused (measured 16-19 s of compile
            # per run() on a 128-atom system).
            # The kinetic/ECP MESH states are built INSIDE the streaming
            # while-loop driver (from the pre-interval positions and the
            # replayed interval rotation), so the AOT lower needs no
            # placeholder mesh-state lists.
            _init_kinetic_state_list_compile = vmap(_init_kinetic_energy_all_elements_streaming_state, in_axes=(None, 0, 0, 0))(
                self._hamiltonian_data.wavefunction_data,
                self._latest_r_up_carts,
                self._latest_r_dn_carts,
                self.__latest_A_old_inv,
            )
        self._mpi_comm.Barrier()
        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GFMC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        # --- Force gradient infrastructure ---
        # _compute_V_elements_t lives at module scope.

        @partial(jit, static_argnums=(4,))
        def _compute_local_energy_t(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> jax.Array:
            _V_diag, _V_nondiag, e_L_meas = _compute_V_elements_t(
                hamiltonian_data=hamiltonian_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
                non_local_move=non_local_move,
                alat=alat,
            )
            # novar=0 split: forces/stress differentiate the MEASUREMENT
            # (continuum) local energy -- mixed-average estimator, no dw/dR.
            return e_L_meas

        wavefunction_for_position_grads, hamiltonian_for_position_grads = self._prepare_position_grad_objects()

        # In-trace position-gradient mask. An eager mask is ineffective because
        # stop_gradient only acts on values participating in the active trace.
        # Wrapping the local energy / ln_psi here
        # ensures the variational-param freeze actually takes effect on the
        # tracer that grad sees.
        _POSITION_GRAD_MASK_KWARGS_LRDMC = dict(
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=False,
        )

        def _e_L_t_with_position_mask(
            active_positions: ActivePositions,
            base_h: Hamiltonian_data,
            r_up: jax.Array,
            r_dn: jax.Array,
            RT: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> jax.Array:
            h = _inject_active_positions(base_h, active_positions)
            masked_wf = h.wavefunction_data.with_param_grad_mask(**_POSITION_GRAD_MASK_KWARGS_LRDMC)
            return _compute_local_energy_t(h.replace(wavefunction_data=masked_wf), r_up, r_dn, RT, non_local_move, alat)

        # argnums=(0, 2, 3): grad over (active_positions, r_up, r_dn).
        # Active-positions output is the per-walker total dE/dR (or
        # dln|Psi|/dR for the ln_psi path); extracting positions into one
        # differentiable leaf prevents unrelated parameter gradients.
        # For complex Bloch, e_L is complex per sample and jax.grad rejects
        # complex outputs on real inputs. The HF force is Re(de_L/dR), which
        # equals d(Re e_L)/dR for real R, so wrap with jnp.real before grad.
        # Bit-identical at Gamma where e_L is real.
        def _real_e_L_t_with_position_mask(*args: Any, **kwargs: Any) -> jax.Array:
            return jnp.real(_e_L_t_with_position_mask(*args, **kwargs))

        # This autograd closure is the fallback (``_ANALYTIC_GRAD_ENABLED=False``); the default
        # HF force routes through the analytic
        # ``jqmc.derivatives.gfmc.grad_e_L_t_wrt_positions`` kernel (rebound below, after the ln|Psi|
        # closures). This ``jax.grad`` twin differentiates the LRDMC *lattice-discretised* local
        # energy ``_compute_local_energy_t`` (``alat`` regularisation + fixed-node sign-flip,
        # ``w_L``-consistent) -- the exact oracle the analytic kernel is verified against
        # (tests/test_jqmc_kernel/test_gfmc_force_analytic.py).
        _jit_vmap_grad_e_L_t_autograd = jit(
            vmap(
                grad(_real_e_L_t_with_position_mask, argnums=(0, 2, 3)),
                in_axes=(None, None, 0, 0, 0, None, None),
            ),
            static_argnums=(5,),  # non_local_move
        )

        # Uniform signature with the analytic closure (below): accept-and-ignore the trailing
        # ``geminal_inv`` (the autograd fallback recomputes the inverse through its own graph).
        def _jit_vmap_grad_e_L_t(
            active_positions: ActivePositions,
            base_h: Hamiltonian_data,
            r_up_b: jax.Array,
            r_dn_b: jax.Array,
            RT_b: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            geminal_inv: jax.Array | None = None,
        ) -> GFMCPositionGradient:
            return _jit_vmap_grad_e_L_t_autograd(active_positions, base_h, r_up_b, r_dn_b, RT_b, non_local_move, alat)

        # dln|Psi|/dR (Pulay,
        # per-atom accumulate) + dln|Psi|/dr_up,dr_dn (drift) route through the
        # unified ``jqmc.derivatives`` interface (``analytic=False`` selects
        # masked autograd -- ln|Psi| has no potential so it is
        # LRDMC-independent, differentiating the same ``evaluate_ln_wavefunction``
        # as MCMC). Reuse MCMC's module-level LRU-cached ``jit(vmap)`` wrapper so
        # the closure object is shared (JAX compilation-cache hit). Returns
        # ``(grad_ln_Psi_R, grad_ln_Psi_r_up, grad_ln_Psi_r_dn)`` (``"R"`` is the
        # interface's ``accumulate_position_grad`` axis).
        # GFMC production ln|Psi| (drift/Pulay/stress) builds the same
        # ``bridge.batched_grad_ln_Psi`` closure MCMC's run() does;
        # ``bridge._build`` is ``@lru_cache(None)``, so the compiled inner jit is shared.
        # ``analytic`` is driven by this module's ``_ANALYTIC_GRAD_ENABLED``,
        # the same flag MCMC uses.
        _grad_ln_psi_rR = bridge.batched_grad_ln_Psi(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
        _active_positions_t = _extract_active_positions_from_ham(self._hamiltonian_data)

        # LRDMC strained-energy kernels for Hellmann-Feynman and Pulay stress.
        # ``F = I + eps`` is applied to every Structure_data leaf in the
        # Hamiltonian (via ``_apply_strain``) and to the electron coords
        # / ECP mesh / wavefunction; ``jax.jacrev`` at ``eps = 0`` yields
        # the per-walker sigma_HF / sigma_PP. ``non_local_move`` and ``alat`` are
        # forwarded unchanged --- the LRDMC lattice-projector discretisation
        # operates in the strained frame just like in the un-strained one.
        def _compute_local_energy_t_strained(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            RT: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            strain: jax.Array,
        ) -> jax.Array:
            F = jnp.eye(3, dtype=jnp.float64) + strain
            new_h = hamiltonian_data._apply_strain(F)
            new_r_up = r_up_carts @ F.T
            new_r_dn = r_dn_carts @ F.T
            # The lattice / ECP quadrature rotation is not strained: the strained ECP
            # quadrature points stay on the sphere of the strained electron-nucleus
            # distance along fixed directions.
            return _compute_local_energy_t(new_h, new_r_up, new_r_dn, RT, non_local_move, alat)

        # Complex Bloch uses the same physical-real projection as the force
        # gradient; differentiating Re(e_L) yields the physical stress and is
        # a no-op at Gamma.
        def _real_compute_local_energy_t_strained(*args: Any, **kwargs: Any) -> jax.Array:
            return jnp.real(_compute_local_energy_t_strained(*args, **kwargs))

        # Fallback HF stress (autograd twin); the default routes through the analytic
        # ``jqmc.derivatives.gfmc.grad_e_L_t_wrt_strain`` kernel (rebound below). This jacrev of the
        # LRDMC lattice-discretised ``_compute_local_energy_t`` (alat/fixed-node, w_L-consistent)
        # is the oracle the analytic stress kernel is verified against.
        _jit_vmap_local_stress_HF_t_autograd = jit(
            vmap(
                jax.jacrev(_real_compute_local_energy_t_strained, argnums=6),
                in_axes=(None, 0, 0, 0, None, None, None),
            ),
            static_argnums=(4,),  # non_local_move
        )

        # Uniform signature with the analytic stress closure: accept-and-ignore trailing ``geminal_inv``.
        def _jit_vmap_local_stress_HF_t(
            base_h: Hamiltonian_data,
            r_up_b: jax.Array,
            r_dn_b: jax.Array,
            RT_b: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            strain: jax.Array,
            geminal_inv: jax.Array | None = None,
        ) -> jax.Array:
            return _jit_vmap_local_stress_HF_t_autograd(base_h, r_up_b, r_dn_b, RT_b, non_local_move, alat, strain)

        # The Pulay stress
        # dln|Psi|/deps routes through the ``jqmc.derivatives`` interface
        # (``analytic=False`` reuses MCMC's ``_local_log_psi_strained`` at strain 0
        # -- identical to the removed local ``_local_log_psi_strained_t``). ln|Psi|
        # is potential-free so this is LRDMC-independent and maps cleanly.
        _grad_ln_psi_eps = bridge.batched_grad_ln_Psi(wrt="eps", with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)

        # Route HF force/stress from the autograd closures above to the verified
        # analytic continuum-measurement kernels (``jqmc.derivatives.gfmc``) when ``_ANALYTIC_GRAD_ENABLED``
        # (see ``_analytic_hf_enabled_for``; no configuration is carved out to autograd).
        # The analytic kernels return ``(dR (n_atom,3), dr_up, dr_dn)``
        # per walker; wrap ``dR`` back into the ``{"positions": ...}`` active-position dict the driver
        # consumes (``grad_e_L_pos_dict["positions"]``). ``active_positions`` / ``non_local_move`` are
        # unused on this path (the sentinel positions already live on ``base_h``; the ECP non-local is
        # the continuous ``tmove`` reassembly). ``_assert_scope`` inside the kernel only documents
        # the derivative boundary: complex-twist DTN is supported analytically and nothing is
        # rejected at warmup.
        if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
            from .derivatives import gfmc as _gfmc_deriv

            _nlm_t = self._non_local_move  # static: threads tmove/dltmove into the analytic kernels
            # The base-position geminal inverse is threaded as a traced vmapped argument
            # (``geminal_inv``, one per walker) so the analytic measurement derivative reuses it instead of
            # recomputing a fresh SVD-pinv (2 SVDs/step -> 0).  The GFMC_t tau-projection SM carry is
            # NOT G^-1 at the observable positions (measured desync), so the driver passes a FRESH
            # per-step ``_geminal_inv_batched`` (== what the kernel would compute internally, so
            # fp-identical) SHARED across force / stress / the ln|Psi| seeds below.
            _vmap_analytic_force_t = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_t_wrt_positions, _nlm_t)
            _vmap_analytic_stress_t = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_t_wrt_strain, _nlm_t)
            # When force and stress are both consumed each step
            # (position_deriv + PBC), one fused program shares the continuous
            # measurement primal between the two observables; the force closure
            # stashes stress for the same-step consumer.
            _vmap_analytic_both_t = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_t_wrt_positions_and_strain, _nlm_t)
            _fused_stress_stash: dict = {}
            # comput_stress=False -> force-only variant.
            _fuse_force_stress = bool(
                self._comput_position_deriv and self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag
            )

            def _jit_vmap_grad_e_L_t(
                active_positions: ActivePositions,
                base_h: Hamiltonian_data,
                r_up_b: jax.Array,
                r_dn_b: jax.Array,
                RT_b: jax.Array,
                non_local_move: NonLocalMove,
                alat: float,
                geminal_inv: jax.Array | None = None,
            ) -> GFMCPositionGradient:
                if _fuse_force_stress:
                    dR, dru, drd, sHF = _vmap_analytic_both_t(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)
                    _fused_stress_stash["stress"] = sHF
                    return {"positions": dR}, dru, drd
                dR, dru, drd = _vmap_analytic_force_t(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)
                return {"positions": dR}, dru, drd

            def _jit_vmap_local_stress_HF_t(
                base_h: Hamiltonian_data,
                r_up_b: jax.Array,
                r_dn_b: jax.Array,
                RT_b: jax.Array,
                non_local_move: NonLocalMove,
                alat: float,
                strain: jax.Array,
                geminal_inv: jax.Array | None = None,
            ) -> jax.Array:
                if _fuse_force_stress and "stress" in _fused_stress_stash:
                    return _fused_stress_stash.pop("stress")
                return _vmap_analytic_stress_t(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)

        if self._use_swct:
            _jit_vmap_swct_omega_t = jit(vmap(evaluate_swct_omega, in_axes=(None, 0)))
            # Fused twin: ``Omega . field`` without materialising the per-walker
            # (N_a, N_e) weight matrix (swct.evaluate_swct_omega_dot).
            _jit_vmap_swct_omega_dot_t = jit(evaluate_swct_omega_dot_batched)
            _jit_vmap_swct_domega_t = jit(vmap(evaluate_swct_domega, in_axes=(None, 0)))

        if self._comput_position_deriv or self._comput_stress:
            start_init_force = time.perf_counter()
            logger.info("Start compilation of position-observable gradient functions.")
            logger.info("  Compilation is in progress...")
            # GFMC_t observable-inverse policy: the tau-projection carry is
            # fp-exact G^-1 at the observable positions (flat ``jnp.where`` SM
            # selection in ``_projection_t_core``), so the analytic force/stress
            # + ln|Psi| seeds thread it directly (no fresh per-step SVD-pinv).
            # ``GFMC_T_OBS_FRESH_INV=1`` selects the shared fresh pinv
            # (fp-equivalence oracle).
            import os as _os_gfmc_obs

            _obs_use_fresh_inv = bool(_os_gfmc_obs.environ.get("GFMC_T_OBS_FRESH_INV"))
            _dummy_RTs = jnp.stack([jnp.eye(3, dtype=jnp.float64)] * self._num_walkers)
            # GFMC_t: thread the tau-projection SM carry ``self.__latest_A_old_inv``
            # into the analytic HF force / stress and the ln|Psi| drift/Pulay/eps
            # seeds (see the per-step site in the branching loop for the full
            # rationale). The carry is fp-exact ``G^-1`` at the returned
            # positions, so it equals the fresh pinv of the shared
            # ``_geminal_inv_batched`` without that standalone SVD-pinv launch
            # per observable step. Warm-up mirrors the per-step choice so both
            # compile the same graph. ``GFMC_T_OBS_FRESH_INV=1`` selects the
            # shared fresh pinv (equivalence oracle).
            # The native force fold-in never dispatches these JAX HF/Pulay/stress
            # grad kernels (the LRDMC force is folded into ``run_gfmc_projection_t``), so
            # skip their compile warmups on the native path. SWCT is native too,
            # so its JAX warmups are needed only on the XLA path.
            if not _rs_gfmc_chunk:
                if _obs_use_fresh_inv:
                    _gemin_inv_t_warmup = _geminal_inv_batched(
                        hamiltonian_for_position_grads.wavefunction_data.geminal_data,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                    )[1]
                else:
                    _gemin_inv_t_warmup = self.__latest_A_old_inv
                if self._comput_position_deriv:
                    _, _, _ = _jit_vmap_grad_e_L_t(
                        _active_positions_t,
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        _dummy_RTs,
                        self._non_local_move,
                        self._alat,
                        _gemin_inv_t_warmup,
                    )
                    _, _, _ = _grad_ln_psi_rR(
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        _gemin_inv_t_warmup,
                    )
            if self._comput_position_deriv and self._use_swct and not _rs_gfmc_chunk:
                # Warm up SWCT vmap callables here so the very first branching
                # step that consumes them does not pay JIT compile time.
                _ = _jit_vmap_swct_omega_t(self._hamiltonian_data.structure_data, self._latest_r_up_carts)
                _ = _jit_vmap_swct_omega_t(self._hamiltonian_data.structure_data, self._latest_r_dn_carts)
                _ = _jit_vmap_swct_domega_t(self._hamiltonian_data.structure_data, self._latest_r_up_carts)
                _ = _jit_vmap_swct_domega_t(self._hamiltonian_data.structure_data, self._latest_r_dn_carts)
            # HF/Pulay stress kernels fire only on PBC + position derivatives
            # runs in the branching loop. Lower them here so the
            # first branching step that emits stress does not stall on a fresh
            # JIT trace -- otherwise the symptom is exactly the MCMC pattern
            # (step 0 ~0 sec, step 1 ~50 sec) on the GFMC side.
            if not _rs_gfmc_chunk and self._hamiltonian_data.structure_data.pbc_flag and self._comput_stress:
                _zero_eps_warmup = jnp.zeros((3, 3), dtype=jnp.float64)
                _ = _jit_vmap_local_stress_HF_t(
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    _dummy_RTs,
                    self._non_local_move,
                    self._alat,
                    _zero_eps_warmup,
                    _gemin_inv_t_warmup,
                )
                _ = _grad_ln_psi_eps(
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    _gemin_inv_t_warmup,
                )
            end_init_force = time.perf_counter()
            timer_projection_init += end_init_force - start_init_force
            logger.info("End compilation of position-observable gradient functions.")
            logger.info(f"Elapsed Time = {end_init_force - start_init_force:.2f} sec.")
            logger.info("")

        # Main branching loop.
        # Progress and filesystem control polling are host-side diagnostics, not
        # part of the GFMC_t algorithm. Keep them infrequent on short CPU benchmarks.
        gfmc_interval = max(1, int(np.ceil(num_mcmc_steps / 20)))
        control_poll_interval = min(max(num_mcmc_steps, 1), 30)

        # ------------------------------------------------------------------
        # Pre-build the per-branching projection driver. Defined ONCE here
        # (outside the ``for i_branching`` loop) so that:
        #   * The Python closure objects ``_body_t`` / ``_body_t_streaming``
        #     are stable across iterations -> the implicit jit cache key
        #     for ``lax.while_loop`` hits after the first compile.
        #   * Closing over ``self._random_discretized_mesh``,
        #     ``self._non_local_move``, ``self._alat``, and
        #     ``self._hamiltonian_data`` is safe because none of them
        #     change between branching steps within a single ``run()``.
        # If these were defined inside the per-branching loop, every step
        # would create a fresh function identity and trigger a full
        # re-trace + re-compile.
        # ------------------------------------------------------------------
        def _cond_t(carry: GFMCProjectionTLoopState | GFMCProjectionTStreamingState) -> jax.Array:
            tau_left = carry[2]
            return jnp.max(tau_left) > 0.0

        _body_vmap_t = vmap(
            _projection_t,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
        )

        # Build the interval potential once from each walker configuration.
        # Inactive Hamiltonians carry fixed-shape inert leaves.
        _build_pot_vmap = vmap(_gfmc_build_interval_pot_state, in_axes=(None, 0, 0, None))

        def _body_t(carry: GFMCProjectionTLoopState) -> GFMCProjectionTLoopState:
            (_, pcl, tll, wll, ru, rd, Ainv, key, interval_RT, interval_ecp_RT, pot) = carry
            return _body_vmap_t(
                pcl,
                tll,
                wll,
                ru,
                rd,
                Ainv,
                key,
                interval_RT,
                interval_ecp_RT,
                pot,
                self._random_discretized_mesh,
                self._non_local_move,
                self._alat,
                self._hamiltonian_data,
            )

        @jit
        def _run_projection_loop(
            pcl: jax.Array,
            tll: jax.Array,
            wll: jax.Array,
            ru: jax.Array,
            rd: jax.Array,
            Ainv: jax.Array,
            key: jax.Array,
        ) -> GFMCProjectionTLoopState:
            pot0 = _build_pot_vmap(self._hamiltonian_data, ru, rd, self._alat)
            init_carry = _body_vmap_t(
                pcl,
                tll,
                wll,
                ru,
                rd,
                Ainv,
                key,
                interval_RT_init,
                interval_ecp_RT_init,
                pot0,
                self._random_discretized_mesh,
                self._non_local_move,
                self._alat,
                self._hamiltonian_data,
            )
            return lax.while_loop(_cond_t, _body_t, init_carry)

        if use_streaming:
            _ecp_mesh_in_axis = 0 if self._hamiltonian_data.coulomb_potential_data.ecp_flag else None
            _body_vmap_t_streaming = vmap(
                _projection_t_streaming,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, _ecp_mesh_in_axis, 0, 0, 0, 0, None, None, None, None),
                axis_name=_GFMC_WALKER_AXIS,
            )

            def _body_t_streaming(carry: GFMCProjectionTStreamingState) -> GFMCProjectionTStreamingState:
                (_, pcl, tll, wll, ru, rd, Ainv, ks, kms, ems, key, interval_RT, interval_ecp_RT, pot, n_ref) = carry
                return _body_vmap_t_streaming(
                    pcl,
                    tll,
                    wll,
                    ru,
                    rd,
                    Ainv,
                    key,
                    ks,
                    kms,
                    ems,
                    interval_RT,
                    interval_ecp_RT,
                    pot,
                    n_ref,
                    self._random_discretized_mesh,
                    self._non_local_move,
                    self._alat,
                    self._hamiltonian_data,
                )

            _first_vmap_t_streaming = vmap(
                _projection_t_streaming_first,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None),
                axis_name=_GFMC_WALKER_AXIS,
            )

            @jit
            def _run_projection_loop_streaming(
                pcl: jax.Array,
                tll: jax.Array,
                wll: jax.Array,
                ru: jax.Array,
                rd: jax.Array,
                Ainv: jax.Array,
                key: jax.Array,
                ks: Kinetic_streaming_state,
            ) -> GFMCProjectionTStreamingState:
                pot0 = _build_pot_vmap(self._hamiltonian_data, ru, rd, self._alat)
                # Substep 0 is UNROLLED here (outside the while_loop) through
                # the Python-static full-kernel branches of
                # ``_projection_t_core`` -- full-kernel substep-0 values and
                # post-hop mesh-state init -- while the loop body
                # only ever runs the cached kernels + unconditional advances.
                # No dual kernel execution on any substep; substep 0 consumes
                # its key splits inside the unrolled call, so the key stream
                # and rotation cadence are those of a rolled loop.
                init_carry = _first_vmap_t_streaming(
                    pcl,
                    tll,
                    wll,
                    ru,
                    rd,
                    Ainv,
                    key,
                    ks,
                    interval_RT_init,
                    interval_ecp_RT_init,
                    pot0,
                    jnp.zeros(ru.shape[0], dtype=jnp.int32),
                    self._random_discretized_mesh,
                    self._non_local_move,
                    self._alat,
                    self._hamiltonian_data,
                )
                return lax.while_loop(_cond_t, _body_t_streaming, init_carry)

        # ------------------------------------------------------------------
        # Warm up the lax.while_loop driver(s) via AOT compilation, so that
        # the first ``branching step`` does NOT include compile time.
        # ``.lower(*args).compile()`` traces & compiles without executing,
        # so walker state / RNG keys are not consumed.
        # ------------------------------------------------------------------
        start_warmup = time.perf_counter()
        if _drivers_cached:
            # Reuse the instance-cached jitted drivers and their compiled
            # executables); the freshly defined closures above are discarded.
            _run_projection_loop = _driver_cache[1]
            _run_projection_loop_streaming = _driver_cache[2]
        # The native chunk never runs the while-loop drivers, so skip their AOT compiles.
        elif not _rs_gfmc_chunk:
            logger.info("Start compilation of the GFMC projection while_loop driver.")
            logger.info("  Compilation is in progress...")
            if not use_streaming:
                # The non-streaming driver is dead code while streaming
                # is enabled -- compile it only when it will be dispatched.
                _run_projection_loop.lower(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                ).compile()
            if use_streaming:
                _run_projection_loop_streaming.lower(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                    _init_kinetic_state_list_compile,
                ).compile()
        if not _drivers_cached:
            self._projection_driver_cache = (
                self._hamiltonian_data,
                _run_projection_loop,
                _run_projection_loop_streaming,
            )
        self._mpi_comm.Barrier()
        end_warmup = time.perf_counter()
        timer_projection_init += end_warmup - start_warmup
        logger.info("End compilation of the GFMC projection while_loop driver.")
        logger.info(f"Elapsed Time = {end_warmup - start_warmup:.2f} sec.")
        logger.info("")

        _log_memory_usage("gfmc_t_post_warmup", comm=self._mpi_comm)

        logger.info("-Start branching-")
        num_mcmc_done = 0

        # -- Extend stored arrays with zero-padding for new steps --
        # dtype for stored numpy arrays
        dtype_np = np.float64
        # average_projection_counter is stored on all ranks
        self._stored_average_projection_counter = np.concatenate(
            [self._stored_average_projection_counter, np.zeros((num_mcmc_steps,), dtype=dtype_np)]
        )
        # other observables are stored on rank 0 only
        if self._mpi_rank == 0:
            n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
            n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
            n_atoms = self._hamiltonian_data.structure_data.natom

            # Preserve stored_e_L's complex Bloch dtype (complex128
            # on twist k != 0). Bit-identical at Gamma where dtype is float64.
            self._stored_e_L = np.concatenate([self._stored_e_L, np.zeros((num_mcmc_steps, 1), dtype=self._stored_e_L.dtype)])
            self._stored_e_L2 = np.concatenate([self._stored_e_L2, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            self._stored_w_L = np.concatenate([self._stored_w_L, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            if self._mpc:
                self._stored_mpc_dV = np.concatenate([self._stored_mpc_dV, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            if self._comput_position_deriv:
                self._stored_force_HF = np.concatenate(
                    [self._stored_force_HF, np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=dtype_np)]
                )
                self._stored_force_PP = np.concatenate(
                    [self._stored_force_PP, np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=dtype_np)]
                )
                self._stored_E_L_force_PP = np.concatenate(
                    [
                        self._stored_E_L_force_PP,
                        np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=self._stored_E_L_force_PP.dtype),
                    ]
                )
            if self._comput_stress:
                self._stored_stress_HF = np.concatenate(
                    [self._stored_stress_HF, np.zeros((num_mcmc_steps, 1, 3, 3), dtype=dtype_np)]
                )
                self._stored_stress_PP = np.concatenate(
                    [self._stored_stress_PP, np.zeros((num_mcmc_steps, 1, 3, 3), dtype=dtype_np)]
                )
                self._stored_E_L_stress_PP = np.concatenate(
                    [self._stored_E_L_stress_PP, np.zeros((num_mcmc_steps, 1, 3, 3), dtype=self._stored_E_L_stress_PP.dtype)]
                )

        gfmc_loop_start = time.perf_counter()
        for i_branching in range(num_mcmc_steps):
            if i_branching % gfmc_interval == 0:
                progress = (i_branching + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
                gmfc_total_current = time.perf_counter()
                logger.info(
                    f"  branching step = {i_branching + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gfmc_loop_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0
            if _rs_gfmc_chunk:
                projection_counter_list = np.zeros(self._num_walkers, dtype=np.float64)
                tau_left_list = np.full(self._num_walkers, self._tau, dtype=np.float64)
                w_L_list = np.ones(self._num_walkers, dtype=np.float64)
            else:
                projection_counter_list = jnp.zeros(self._num_walkers, dtype=jnp.float64)
                tau_left_list = jnp.full(self._num_walkers, self._tau, dtype=jnp.float64)
                w_L_list = jnp.ones(self._num_walkers, dtype=jnp.float64)

            start_projection = time.perf_counter()
            # If streaming is enabled, build a fresh per-walker kinetic state
            # at the start of each branching step (consistent with the freshly
            # reset projection_counter / tau_left / w_L). The state is then
            # advanced by ``_projection_t_streaming`` inside the while loop.
            if use_streaming and not _rs_gfmc_chunk:
                kinetic_state_list = vmap(_init_kinetic_energy_all_elements_streaming_state, in_axes=(None, 0, 0, 0))(
                    self._hamiltonian_data.wavefunction_data,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                )
                # The kinetic/ECP MESH states are built inside
                # ``_run_projection_loop_streaming`` from the pre-interval
                # walker positions and the replayed interval rotation, so no
                # ``interval_RT_init`` placeholder is needed here.
            # projection loop
            #
            # Keep the projection loop in ``lax.while_loop`` so the entire
            # interval is captured in one jit graph. A Python loop that tests
            # ``np.max(tau_left_list)`` would materialize JAX values on the
            # host and block the GPU once per projection step. The device
            # condition is ``jnp.max(tau_left) > 0.0``; the only host sync is
            # the ``block_until_ready`` call after the loop.
            #
            # The driver functions ``_run_projection_loop_*`` are defined
            # **outside** this ``for i_branching`` loop (see above) so the
            # jit cache hits after the first compile; defining them here
            # would create a fresh Python closure each branching and force
            # a re-trace + re-compile per step (catastrophic slowdown).
            if _rs_gfmc_chunk:
                # ===== NATIVE GFMC_t PROJECTION INTERVAL =====
                # One ``run_gfmc_projection_t`` call per walker advances the whole tau
                # interval natively (escape-time loop + LRDMC e_L + w_L + FN hop, driven
                # by the SAME ``gfmc_lrdmc_{ae,ecp}`` kernels the parity-pinned reference
                # chains use). The native production chain is statistically guarded by
                # tests/test_rs_kernel/test_gfmc_chain_modes.py. e_L is the LAST
                # sub-step's evaluation (jqmc ``_projection_t`` semantics) and
                # ``n_substeps`` feeds the projection counter. The SM carry
                # ``self.__latest_A_old_inv`` is not consumed (the native kernels evaluate
                # self-contained) -- its per-step reseed below is skipped too.
                # One batched call fans the whole tau interval over
                # ALL walkers on the rayon global pool (the Python ``for _w`` walker loop
                # -> Rust ``for_each_walker``). Physical interval results are identical
                # to a per-walker ``run_gfmc_projection_t`` loop; returned keys additionally advance through masked
                # steps to the rank-local batch maximum, matching XLA. The same batch
                # remains thread-order independent, but splitting it can change keys.
                # The returned dict gains a leading walker axis. The complex twin folds the same
                # force/stress blocks under the same keys, so the
                # unpack below is shared. The DLA flag reaches the complex kernels via
                # ``dltmove``; both switches (``want_force`` / ``want_stress``) are supplied.
                _rs_keys = np.asarray(self._jax_PRNG_key_list, dtype=np.uint32)
                _ru_all = np.asarray(self._latest_r_up_carts, dtype=np.float64)
                _rd_all = np.asarray(self._latest_r_dn_carts, dtype=np.float64)
                if _rs_complex:
                    _res = _rs_obs_ctx.run_gfmc_projection_t_complex_batch(
                        _ru_all,
                        _rd_all,
                        self._alat,
                        float(self._tau),
                        _rs_keys,
                        _rs_ecp_w,
                        _rs_ecp_gp,
                        random_mesh=bool(self._random_discretized_mesh),
                        dltmove=bool(self._non_local_move == "dltmove"),
                        want_force=_rs_want_force,
                        want_stress=_rs_want_stress,
                        # Debug/oracle twin knob (IMPL-GFMC-INCREMENTAL-TWIN-001):
                        # the SAME env switch that selects the XLA full-recompute
                        # route also selects it on the native chunk, so the twin
                        # semantics are backend-uniform.
                        debug_full_recompute=not _gfmc_incremental_pot_enabled(),
                        # Screen the projection-carried inverse handed to the
                        # Pulay force/stress fold-in (drifted -> fresh route).
                        inverse_refresh_threshold=float(_setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD),
                    )
                else:
                    _res = _rs_obs_ctx.run_gfmc_projection_t_batch(
                        _ru_all,
                        _rd_all,
                        self._alat,
                        float(self._tau),
                        _rs_keys,
                        _rs_ecp_w,
                        _rs_ecp_gp,
                        random_mesh=bool(self._random_discretized_mesh),
                        dltmove=_rs_dltmove,
                        want_force=_rs_want_force,
                        want_stress=_rs_want_stress,
                        # Debug/oracle twin knob (IMPL-GFMC-INCREMENTAL-TWIN-001):
                        # the SAME env switch that selects the XLA full-recompute
                        # route also selects it on the native chunk, so the twin
                        # semantics are backend-uniform.
                        debug_full_recompute=not _gfmc_incremental_pot_enabled(),
                        # Screen the projection-carried inverse handed to the
                        # Pulay force/stress fold-in (drifted -> fresh route).
                        inverse_refresh_threshold=float(_setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD),
                    )
                _warn_native_j3_slow_fallback_once()
                e_L_list = np.asarray(_res["e_L"])
                w_L_list = np.asarray(_res["w_L"], dtype=np.float64)
                projection_counter_list = np.asarray(_res["n_substeps"], dtype=np.float64)
                self._latest_r_up_carts = np.asarray(_res["r_up"], dtype=np.float64)
                self._latest_r_dn_carts = np.asarray(_res["r_dn"], dtype=np.float64)
                self._jax_PRNG_key_list = np.asarray(_res["final_key"], dtype=np.uint32)
                # Incremental-potential liveness witness (native chunk; zeros on
                # molecular / BITEXACT / debug full-recompute routes).
                if "n_pot_updates" in _res:
                    self._latest_gfmc_pot_updates = np.asarray(_res["n_pot_updates"], dtype=np.float64)
                # Stash native raw grads for the force block below (native chunk path).
                if _rs_want_force:
                    _rs_native_grad_e_L_R = np.asarray(_res["hf_dR"])
                    _rs_native_grad_e_L_r_up = np.asarray(_res["hf_dr_up"])
                    _rs_native_grad_e_L_r_dn = np.asarray(_res["hf_dr_dn"])
                    _rs_native_grad_ln_Psi_dR = np.asarray(_res["pp_dR"])
                    _rs_native_grad_ln_Psi_r_up = np.asarray(_res["pp_dr_up"])
                    _rs_native_grad_ln_Psi_r_dn = np.asarray(_res["pp_dr_dn"])
                if _rs_want_stress:
                    _rs_native_stress_HF = np.asarray(_res["stress_HF"])
                    _rs_native_stress_PP = np.asarray(_res["stress_PP"])
            elif use_streaming:
                (
                    e_L_list,
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    kinetic_state_list,
                    _kinetic_mesh_state_list_out,
                    _ecp_mesh_state_list_out,
                    self._jax_PRNG_key_list,
                    _latest_kinetic_RTs,
                    latest_RTs,
                    _latest_pot_state,
                    _interval_refreshes,
                ) = _run_projection_loop_streaming(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                    kinetic_state_list,
                )
                # In-interval refresh witness (walker-hops whose carried
                # inverse was rebuilt by the per-hop residual screen).
                self._num_interval_inverse_refreshes = getattr(self, "_num_interval_inverse_refreshes", 0) + int(
                    np.sum(np.asarray(_interval_refreshes))
                )
                # Incremental-potential liveness witness (per-walker applied-update
                # counts of the finished interval; zeros when the state is inactive).
                self._latest_gfmc_pot_updates = np.asarray(_latest_pot_state[6])
            else:
                (
                    e_L_list,
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                    _latest_kinetic_RTs,
                    latest_RTs,
                    _latest_pot_state,
                ) = _run_projection_loop(
                    projection_counter_list,
                    tau_left_list,
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                )
                self._latest_gfmc_pot_updates = np.asarray(_latest_pot_state[6])

            # sync. jax arrays computations (JAX projection path only; the native
            # chunk returned host numpy already).
            if not _rs_gfmc_chunk:
                e_L_list.block_until_ready()
                projection_counter_list.block_until_ready()
                tau_left_list.block_until_ready()
                w_L_list.block_until_ready()
                self._latest_r_up_carts.block_until_ready()
                self._latest_r_dn_carts.block_until_ready()
                self.__latest_A_old_inv.block_until_ready()
                self._jax_PRNG_key_list.block_until_ready()

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection

            # evaluate observables
            start_observable = time.perf_counter()
            # e_L evaluation is not necesarily repeated here.

            _want_xla_position_observable = self._comput_position_deriv or (
                self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag
            )
            if not _rs_gfmc_chunk and _want_xla_position_observable:
                if _obs_use_fresh_inv:
                    _gemin_inv_t = _geminal_inv_batched(
                        hamiltonian_for_position_grads.wavefunction_data.geminal_data,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                    )[1]
                else:
                    # An interval that leaves a near-singular start
                    # carries the seed's inverse error to the observables;
                    # refresh the drifted walkers from G at these positions.
                    _gemin_inv_t, _n_inv_refreshed = _refresh_observable_geminal_inv(
                        hamiltonian_for_position_grads.wavefunction_data.geminal_data,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        self.__latest_A_old_inv,
                        kinetic_state_list.det_state if use_streaming else None,
                    )
                    self._num_inverse_refreshes = getattr(self, "_num_inverse_refreshes", 0) + _n_inv_refreshed

            # atomic force related
            if self._comput_position_deriv:
                # Active-positions API: dict[{"positions"}, (num_walkers,
                # n_atom, 3)] returns the total dE_L/dR (HF + Pulay) and
                # dln|Psi|/dR already summed via JAX chain rule, so no
                # per-path summing is needed.
                # GFMC_t: thread the tau-projection SM carry as the shared
                # observable-step geminal inverse. With the flat ``jnp.where``
                # SM selection in ``_projection_t_core``
                # the carry ``self.__latest_A_old_inv`` is fp-exact ``G^-1`` at
                # the returned positions (max|G . A_carry - I| <= ~4e-16 across
                # taus {0.03,0.01,0.001} on H2 AE + H_ecp tmove; a nested
                # ``lax.cond`` gives ~0.6-0.8), so it equals the fresh pinv of
                # the shared ``_geminal_inv_batched``: fp-identical
                # observables (e_L/w_L bit-identical; force/stress <=4e-14 rel)
                # without that standalone shared SVD-pinv launch (1 LAPACK
                # SVD/observable-step). (The analytic force/stress
                # kernels carry their own internal SVDs independent of this
                # threaded inverse.)
                # ``GFMC_T_OBS_FRESH_INV=1`` selects the shared fresh pinv.
                if _rs_gfmc_chunk:
                    # ===== NATIVE FORCE FOLD-IN =====
                    # The HF-force (dE_L/dR,dr) + Pulay (dln|Psi|/dR,dr) grads were folded
                    # OUT of ``run_gfmc_projection_t`` on the projection final config with
                    # the interval's last mesh rotation -- the SAME parity-pinned LRDMC
                    # kernels ``_jit_vmap_grad_e_L_t`` / ``_grad_ln_psi_rR`` drive on the JAX
                    # path (fresh LU, GFMC carries no sweep SM inverse). SWCT
                    # contractions are native too; walker weighting stays in Python.
                    grad_e_L_R = _rs_native_grad_e_L_R
                    grad_e_L_r_up = _rs_native_grad_e_L_r_up
                    grad_e_L_r_dn = _rs_native_grad_e_L_r_dn
                    grad_ln_Psi_dR = _rs_native_grad_ln_Psi_dR
                    grad_ln_Psi_r_up = _rs_native_grad_ln_Psi_r_up
                    grad_ln_Psi_r_dn = _rs_native_grad_ln_Psi_r_dn
                else:
                    grad_e_L_pos_dict, grad_e_L_r_up, grad_e_L_r_dn = _jit_vmap_grad_e_L_t(
                        _active_positions_t,
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        latest_RTs,
                        self._non_local_move,
                        self._alat,
                        _gemin_inv_t,
                    )
                    grad_e_L_r_up.block_until_ready()
                    grad_e_L_r_dn.block_until_ready()
                    grad_e_L_R = grad_e_L_pos_dict["positions"]

                    # dln|Psi|/dR (Pulay) and dln|Psi|/dr drift via the
                    # derivatives interface; "R" is the per-atom accumulate
                    # (``accumulate_position_grad``). Reuses the shared
                    # observable geminal inverse (``_gemin_inv_t`` = carry by default).
                    grad_ln_Psi_dR, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = _grad_ln_psi_rR(
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        _gemin_inv_t,
                    )
                    grad_ln_Psi_r_up.block_until_ready()
                    grad_ln_Psi_r_dn.block_until_ready()

                if self._use_swct and not _rs_gfmc_chunk:
                    # Only domega is precomputed here; the Omega contraction is
                    # fused into the force assembly below.
                    grad_omega_dr_up = _jit_vmap_swct_domega_t(self._hamiltonian_data.structure_data, self._latest_r_up_carts)
                    grad_omega_dr_dn = _jit_vmap_swct_domega_t(self._hamiltonian_data.structure_data, self._latest_r_dn_carts)
                    grad_omega_dr_up.block_until_ready()
                    grad_omega_dr_dn.block_until_ready()
                # (use_swct=False: no omega buffers -- the force assembly below
                # skips the identically-zero SWCT terms instead of contracting
                # full-size zero (nw, n_atoms, N_e) buffers; zero-risk deletion,
                # bit-identical while avoiding an expanded temporary.)

            end_observable = time.perf_counter()
            timer_observable += end_observable - start_observable

            # Barrier before MPI operation
            start_mpi_barrier = time.perf_counter()
            if self._mpi_size > 1:
                self._mpi_comm.Barrier()
            end_mpi_barrier = time.perf_counter()
            timer_mpi_barrier += end_mpi_barrier - start_mpi_barrier

            # Branching starts
            start_collection = time.perf_counter()

            # random number for the later use
            if self._mpi_rank == 0:
                zeta = float(self._branching_rng.random())
            else:
                zeta = None
            if self._mpi_size > 1:
                zeta = self._mpi_comm.bcast(zeta, root=0)

            # jnp.array -> np.array
            w_L_latest = np.asarray(w_L_list)
            e_L_latest = np.asarray(e_L_list)
            e_L2_latest = np.abs(e_L_latest) ** 2

            _raise_collective_nonfinite_observables(
                self._mpi_comm,
                driver="GFMC_t",
                stage="pre-aggregation",
                first_step=self._mcmc_counter + num_mcmc_done,
                observables={"e_L": e_L_latest, "e_L2": e_L2_latest, "w_L": w_L_latest},
                r_up_carts=self._latest_r_up_carts,
                r_dn_carts=self._latest_r_dn_carts,
            )

            # MPC (physical = no): passive observable evaluated on the SAME
            # configurations that were just propagated under Ewald. It never
            # feeds back into e_L, the branching weight w_L, the forces, or any
            # gradient -- it is only accumulated for ``get_dE_mpc``.
            if self._mpc:
                _mpc_ewald_data = self._hamiltonian_data.coulomb_potential_data.ewald_data
                if _rs_mpc_observable:
                    # Native dV = V_ee^MPC - V_ee^Ewald on the SAME Ewald-propagated
                    # collect batch; one native call per
                    # walker. Projection / branching / weight are untouched.
                    import jqmc_rs

                    _mpc_ru = np.asarray(self._latest_r_up_carts)
                    _mpc_rd = np.asarray(self._latest_r_dn_carts)
                    # Rayon walker batch, bit-identical to the scalar loop.
                    mpc_dV_latest = np.asarray(
                        jqmc_rs._compute_mpc_correction_ee_batch(self._mpc_data, _mpc_ewald_data, _mpc_ru, _mpc_rd)
                    )
                else:
                    mpc_dV_latest = np.array(
                        self._jit_vmap_mpc_correction_ee(
                            self._mpc_data,
                            _mpc_ewald_data,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                        )
                    )

            if self._comput_position_deriv:
                grad_e_L_r_up_latest = np.array(grad_e_L_r_up)
                grad_e_L_r_dn_latest = np.array(grad_e_L_r_dn)
                grad_e_L_R_latest = np.array(grad_e_L_R)
                grad_ln_Psi_r_up_latest = np.array(grad_ln_Psi_r_up)
                grad_ln_Psi_r_dn_latest = np.array(grad_ln_Psi_r_dn)
                grad_ln_Psi_dR_latest = np.array(grad_ln_Psi_dR)
                if self._use_swct and not _rs_gfmc_chunk:
                    grad_omega_dr_up_latest = np.array(grad_omega_dr_up)
                    grad_omega_dr_dn_latest = np.array(grad_omega_dr_dn)

            # sum
            nw_sum = len(w_L_latest)
            w_L_sum = np.sum(w_L_latest)
            e_L_sum = np.sum(w_L_latest * e_L_latest)
            # The real variance buffer stores |e_L|^2, mirroring MCMC.
            e_L2_sum = np.sum(w_L_latest * e_L2_latest)
            # MPC: identical intra-step walker reduction to ``e_L_sum`` (same
            # w_L_latest weights, later divided by the same w_L_sum), so that
            # sum(w_L * dV) / sum(w_L) stays consistent with the stored e_L.
            if self._mpc:
                mpc_dV_sum = np.sum(w_L_latest * mpc_dV_latest)
            mod_w_L_latest = w_L_latest
            if self._comput_position_deriv:
                # Pathak-Wagner regularization factor f_eps per walker
                if self._epsilon_PW > 0.0:
                    grad_norm_sq = np.sum(grad_ln_Psi_r_up_latest**2, axis=(1, 2)) + np.sum(
                        grad_ln_Psi_r_dn_latest**2, axis=(1, 2)
                    )
                    nodal_dist = 1.0 / np.sqrt(grad_norm_sq)
                    t = nodal_dist / self._epsilon_PW
                    t2 = t * t
                    t4 = t2 * t2
                    t6 = t4 * t2
                    f_eps = np.where(t < 1.0, 7.0 * t6 - 15.0 * t4 + 9.0 * t2, 1.0)
                    mod_w_L_latest = w_L_latest * f_eps
                else:
                    mod_w_L_latest = w_L_latest

                # Compute per-walker force products BEFORE walker averaging
                if self._use_swct and _rs_gfmc_chunk:
                    force_HF_per_walker, force_PP_per_walker = _native_swct_forces(
                        self._hamiltonian_data.structure_data,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        grad_e_L_R_latest,
                        grad_e_L_r_up_latest,
                        grad_e_L_r_dn_latest,
                        grad_ln_Psi_dR_latest,
                        grad_ln_Psi_r_up_latest,
                        grad_ln_Psi_r_dn_latest,
                    )
                elif self._use_swct:
                    # ``Omega . grad`` fused on device: the (nw, n_atoms, N_e)
                    # weight matrix is a contraction operand, never an output
                    # (A = N_e/3 = 288), so it is never materialised.
                    _sd = self._hamiltonian_data.structure_data
                    _ru_s, _rd_s = self._latest_r_up_carts, self._latest_r_dn_carts
                    # force_HF per walker: shape (n_walkers, n_atoms, 3)
                    force_HF_per_walker = (
                        grad_e_L_R_latest
                        + np.asarray(_jit_vmap_swct_omega_dot_t(_sd, _ru_s, grad_e_L_r_up_latest))
                        + np.asarray(_jit_vmap_swct_omega_dot_t(_sd, _rd_s, grad_e_L_r_dn_latest))
                    )
                    # force_PP per walker: shape (n_walkers, n_atoms, 3)
                    force_PP_per_walker = (
                        grad_ln_Psi_dR_latest
                        + np.asarray(_jit_vmap_swct_omega_dot_t(_sd, _ru_s, grad_ln_Psi_r_up_latest))
                        + np.asarray(_jit_vmap_swct_omega_dot_t(_sd, _rd_s, grad_ln_Psi_r_dn_latest))
                        + 0.5 * (grad_omega_dr_up_latest + grad_omega_dr_dn_latest)
                    )
                else:
                    # SWCT off: Omega/domega are exactly zero, so the einsum
                    # terms are skipped (bit-identical to adding zeros).
                    force_HF_per_walker = grad_e_L_R_latest
                    force_PP_per_walker = grad_ln_Psi_dR_latest
                # E_L * force_PP per walker: shape (n_walkers, n_atoms, 3)
                E_L_force_PP_per_walker = e_L_latest[:, None, None] * force_PP_per_walker

                # Weighted sum of per-walker force products
                # Note: GFMC_t uses w_L directly (no V_diag_E division unlike GFMC_n)
                force_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, force_HF_per_walker)
                force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, force_PP_per_walker)
                E_L_force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, E_L_force_PP_per_walker)

            if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                start_stress = time.perf_counter()
                if _rs_gfmc_chunk and _rs_want_stress:
                    _stress_HF_per_walker = _rs_native_stress_HF
                    _stress_PP_per_walker = _rs_native_stress_PP
                else:
                    _zero_eps = jnp.zeros((3, 3), dtype=jnp.float64)
                    _stress_HF_per_walker = np.asarray(
                        _jit_vmap_local_stress_HF_t(
                            hamiltonian_for_position_grads,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                            latest_RTs,
                            self._non_local_move,
                            self._alat,
                            _zero_eps,
                            _gemin_inv_t,
                        )
                    )
                    _stress_PP_per_walker = np.asarray(
                        _grad_ln_psi_eps(
                            hamiltonian_for_position_grads,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                            _gemin_inv_t,
                        )
                    )
                _E_L_stress_PP_per_walker = e_L_latest[:, None, None] * _stress_PP_per_walker
                stress_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, _stress_HF_per_walker)
                stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, _stress_PP_per_walker)
                E_L_stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest, _E_L_stress_PP_per_walker)
                timer_stress += time.perf_counter() - start_stress

            # reduce
            if self._mpi_size > 1:
                nw_sum = self._mpi_comm.reduce(nw_sum, op=MPI.SUM, root=0)
                w_L_sum = self._mpi_comm.reduce(w_L_sum, op=MPI.SUM, root=0)
                e_L_sum = self._mpi_comm.reduce(e_L_sum, op=MPI.SUM, root=0)
                e_L2_sum = self._mpi_comm.reduce(e_L2_sum, op=MPI.SUM, root=0)
                if self._mpc:
                    mpc_dV_sum = self._mpi_comm.reduce(mpc_dV_sum, op=MPI.SUM, root=0)
                if self._comput_position_deriv:
                    force_HF_weighted_sum = self._mpi_comm.reduce(force_HF_weighted_sum, op=MPI.SUM, root=0)
                    force_PP_weighted_sum = self._mpi_comm.reduce(force_PP_weighted_sum, op=MPI.SUM, root=0)
                    E_L_force_PP_weighted_sum = self._mpi_comm.reduce(E_L_force_PP_weighted_sum, op=MPI.SUM, root=0)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_weighted_sum = self._mpi_comm.reduce(stress_HF_weighted_sum, op=MPI.SUM, root=0)
                    stress_PP_weighted_sum = self._mpi_comm.reduce(stress_PP_weighted_sum, op=MPI.SUM, root=0)
                    E_L_stress_PP_weighted_sum = self._mpi_comm.reduce(E_L_stress_PP_weighted_sum, op=MPI.SUM, root=0)

            if self._mpi_rank == 0:
                # averaged
                w_L_averaged = w_L_sum / nw_sum
                e_L_averaged = e_L_sum / w_L_sum
                e_L2_averaged = e_L2_sum / w_L_sum
                # MPC: same denominator (w_L_sum) as e_L_averaged.
                if self._mpc:
                    mpc_dV_averaged = mpc_dV_sum / w_L_sum
                if self._comput_position_deriv:
                    force_HF_averaged = force_HF_weighted_sum / w_L_sum
                    force_PP_averaged = force_PP_weighted_sum / w_L_sum
                    E_L_force_PP_averaged = E_L_force_PP_weighted_sum / w_L_sum
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_averaged = stress_HF_weighted_sum / w_L_sum
                    stress_PP_averaged = stress_PP_weighted_sum / w_L_sum
                    E_L_stress_PP_averaged = E_L_stress_PP_weighted_sum / w_L_sum

                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                if self._mpc:
                    mpc_dV_averaged = np.expand_dims(mpc_dV_averaged, axis=0)
                if self._comput_position_deriv:
                    force_HF_averaged = np.expand_dims(force_HF_averaged, axis=0)
                    force_PP_averaged = np.expand_dims(force_PP_averaged, axis=0)
                    E_L_force_PP_averaged = np.expand_dims(E_L_force_PP_averaged, axis=0)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_averaged = np.expand_dims(stress_HF_averaged, axis=0)
                    stress_PP_averaged = np.expand_dims(stress_PP_averaged, axis=0)
                    E_L_stress_PP_averaged = np.expand_dims(E_L_stress_PP_averaged, axis=0)

                # store  # This should stored only for MPI-rank = 0 !!!
                self._stored_e_L2[self._mcmc_counter + num_mcmc_done] = np.array(e_L2_averaged)
                self._stored_e_L[self._mcmc_counter + num_mcmc_done] = np.array(e_L_averaged)
                self._stored_w_L[self._mcmc_counter + num_mcmc_done] = np.array(w_L_averaged)
                if self._mpc:
                    self._stored_mpc_dV[self._mcmc_counter + num_mcmc_done] = np.array(mpc_dV_averaged)
                if self._comput_position_deriv:
                    self._stored_force_HF[self._mcmc_counter + num_mcmc_done] = np.array(force_HF_averaged)
                    self._stored_force_PP[self._mcmc_counter + num_mcmc_done] = np.array(force_PP_averaged)
                    self._stored_E_L_force_PP[self._mcmc_counter + num_mcmc_done] = np.array(E_L_force_PP_averaged)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    self._stored_stress_HF[self._mcmc_counter + num_mcmc_done] = np.array(stress_HF_averaged)
                    self._stored_stress_PP[self._mcmc_counter + num_mcmc_done] = np.array(stress_PP_averaged)
                    self._stored_E_L_stress_PP[self._mcmc_counter + num_mcmc_done] = np.array(E_L_stress_PP_averaged)

            if self._mpi_size > 1:
                self._mpi_comm.Barrier()

            end_collection = time.perf_counter()
            timer_collection += end_collection - start_collection

            start_reconfiguration = time.perf_counter()

            # branching
            latest_r_up_carts_before_branching = np.array(self._latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self._latest_r_dn_carts)

            #########################################
            # 1. Gather only the weights to MPI_rank=0 and perform branching calculation
            #########################################
            start_ = time.perf_counter()

            # Each process computes the sum of its local walker weights.
            local_weight_sum = np.sum(w_L_latest)

            # Use pickle-based allreduce here (allowed for this part)
            global_weight_sum = (
                local_weight_sum if self._mpi_size == 1 else self._mpi_comm.allreduce(local_weight_sum, op=MPI.SUM)
            )

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Compute the local probabilities for each walker.
            local_probabilities = w_L_latest / global_weight_sum

            # Compute the local cumulative probabilities.
            # NOTE: MPI reductions for branching probabilities are kept float64
            # unconditionally (regardless of the gfmc precision zone) to avoid
            # population collapse from float32 round-off in the branching step.
            local_cumprob = np.cumsum(local_probabilities)
            local_sum_arr = np.array(np.sum(local_probabilities), dtype=np.float64)
            offset_arr = np.zeros(1, dtype=np.float64)
            if self._mpi_size > 1:
                self._mpi_comm.Exscan([local_sum_arr, MPI.DOUBLE], [offset_arr, MPI.DOUBLE], op=MPI.SUM)
            if self._mpi_rank == 0 or self._mpi_size == 1:
                offset = 0.0
            else:
                offset = float(offset_arr[0])
            local_cumprob += offset

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Gather the local cumulative probability arrays from all processes.
            total_walkers = self.num_walkers * self._mpi_size
            if self._mpi_size == 1:
                global_cumprob = local_cumprob
            else:
                global_cumprob = np.empty(total_walkers, dtype=np.float64)
                self._mpi_comm.Allgather([local_cumprob, MPI.DOUBLE], [global_cumprob, MPI.DOUBLE])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # Total number of walkers across all processes.
            # Compute index range for this rank
            start_ = time.perf_counter()
            start_idx = self._mpi_rank * self.num_walkers
            end_idx = start_idx + self.num_walkers

            # Build only local z-array (length = self.num_walkers)
            z_local = (np.arange(start_idx, end_idx) + zeta) / total_walkers

            # Perform searchsorted and cast the result to int32
            local_chosen_indices = np.searchsorted(global_cumprob, z_local).astype(np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # Gather all local_chosen_indices across ranks using MPI.INT
            start_ = time.perf_counter()

            # Allocate buffer to receive all ranks' indices (Number of indices per rank is identical on every rank)
            if self._mpi_size == 1:
                all_chosen_buf = local_chosen_indices
            else:
                all_chosen_buf = np.empty(self.num_walkers * self._mpi_size, dtype=np.int32)
                # Perform the all-gather operation with 32-bit integers
                self._mpi_comm.Allgather([local_chosen_indices, MPI.INT], [all_chosen_buf, MPI.INT])

            # Use the gathered indices for global statistics
            chosen_walker_indices = all_chosen_buf
            num_survived_walkers = len(np.unique(chosen_walker_indices))
            num_killed_walkers = total_walkers - num_survived_walkers

            # Build the local assignment list of (source_rank, source_local_index)
            local_assignment = [
                (src_global_idx // self.num_walkers, src_global_idx % self.num_walkers)
                for src_global_idx in local_chosen_indices
            ]

            # num projection counter
            ## Compute the local average of the projection counter list.
            ave_projection_counter = np.mean(projection_counter_list)

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 2. In each process, prepare for data exchange based on the new walker selection
            #########################################
            start_ = time.perf_counter()
            reqs = {}
            if self._mpi_size == 1:
                latest_r_up_carts_after_branching = latest_r_up_carts_before_branching[local_chosen_indices].copy()
                latest_r_dn_carts_after_branching = latest_r_dn_carts_before_branching[local_chosen_indices].copy()
            else:
                latest_r_up_carts_after_branching = np.empty_like(latest_r_up_carts_before_branching)
                latest_r_dn_carts_after_branching = np.empty_like(latest_r_dn_carts_before_branching)
                for dest_idx, (src_rank, src_local_idx) in enumerate(local_assignment):
                    if src_rank == self._mpi_rank:
                        # Local copy: no communication needed
                        latest_r_up_carts_after_branching[dest_idx] = latest_r_up_carts_before_branching[src_local_idx]
                        latest_r_dn_carts_after_branching[dest_idx] = latest_r_dn_carts_before_branching[src_local_idx]
                    else:
                        reqs.setdefault(src_rank, []).append((dest_idx, src_local_idx))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 2 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 3. Exchange only the necessary walker data between processes using asynchronous communication
            #########################################

            # 3.1.1: Flatten `reqs` into an (N_req x 3) int32 array of triplets
            start_ = time.perf_counter()
            flat_list = [
                (src_rank, dest_idx, src_local_idx) for src_rank, pairs in reqs.items() for dest_idx, src_local_idx in pairs
            ]
            triplets = np.array(flat_list, dtype=np.int32) if flat_list else np.empty((0, 3), dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.2: Compute how many ints to send to each rank (3 ints per request)
            start_ = time.perf_counter()
            counts_per_rank = np.bincount(triplets[:, 0], minlength=self._mpi_size)  # # reqs per src_rank
            send_counts = (counts_per_rank * 3).astype(np.int32)  # # ints per src_rank
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.3: Post nonblocking Alltoall to exchange counts
            start_ = time.perf_counter()
            if self._mpi_size == 1:
                recv_counts = send_counts.copy()
                req_counts = None
            else:
                recv_counts = np.empty_like(send_counts)
                req_counts = self._mpi_comm.Ialltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.4: Build send_buf while counts exchange is in flight
            start_ = time.perf_counter()
            #   sort by src_rank so that each destination's data is contiguous
            order = np.argsort(triplets[:, 0], kind="mergesort") if triplets.size else np.empty(0, dtype=np.int32)
            sorted_tr = triplets[order]  # shape = (N_req, 3)
            send_buf = sorted_tr.ravel()  # shape = (N_req*3,)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.5: Wait for counts exchange to complete
            start_ = time.perf_counter()
            if req_counts is not None:
                req_counts.Wait()
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.6: Build displacements for send/recv from counts
            start_ = time.perf_counter()
            send_displs = np.zeros_like(send_counts)
            send_displs[1:] = np.cumsum(send_counts)[:-1]
            recv_displs = np.zeros_like(recv_counts)
            recv_displs[1:] = np.cumsum(recv_counts)[:-1]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.7: Allocate recv buffer of the exact size
            start_ = time.perf_counter()
            total_recv = int(np.sum(recv_counts))
            recv_buf = np.empty(total_recv, dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.7 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.8: Post blocking Alltoallv to exchange the triplets
            start_ = time.perf_counter()
            if self._mpi_size == 1:
                recv_buf[:] = send_buf
            else:
                self._mpi_comm.Alltoallv(
                    [send_buf, send_counts, send_displs, MPI.INT], [recv_buf, recv_counts, recv_displs, MPI.INT]
                )
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.8 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.9: Wait for data to arrive and reconstruct per-process request dicts
            start_ = time.perf_counter()
            all_reqs = []
            for p in range(self._mpi_size):
                off = recv_displs[p]
                cnt = recv_counts[p]
                block = recv_buf[off : off + cnt]
                if block.size == 0:
                    all_reqs.append({})
                    continue
                rec_tr = block.reshape(-1, 3)
                proc_dict = {}
                for sr, dest_idx, src_local_idx in rec_tr:
                    proc_dict.setdefault(int(sr), []).append((int(dest_idx), int(src_local_idx)))
                all_reqs.append(proc_dict)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.9 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.10: Filter out empty request dicts
            start_ = time.perf_counter()
            non_empty_all_reqs = [(p, rd) for p, rd in enumerate(all_reqs) if rd]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.10 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-2. Build incoming_reqs: who needs data from me? ---
            start_ = time.perf_counter()
            incoming_reqs = [
                (p, src_local_idx, dest_idx)
                for p, proc_req in non_empty_all_reqs
                if p != self._mpi_rank
                for dest_idx, src_local_idx in proc_req.get(self._mpi_rank, [])
            ]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-3. Post nonblocking receives using Irecv for both up and dn buffers. ---
            start_ = time.perf_counter()
            recv_buffers = {}
            recv_reqs_up = {}
            recv_reqs_dn = {}
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                count = len(req_list)
                shape_up = latest_r_up_carts_before_branching.shape[1:]
                shape_dn = latest_r_dn_carts_before_branching.shape[1:]
                buf_up = np.empty((count, *shape_up), dtype=latest_r_up_carts_before_branching.dtype)
                buf_dn = np.empty((count, *shape_dn), dtype=latest_r_dn_carts_before_branching.dtype)
                recv_buffers[src_rank] = (buf_up, buf_dn)
                recv_reqs_up[src_rank] = self._mpi_comm.Irecv([buf_up, MPI.DOUBLE], source=src_rank, tag=200)
                recv_reqs_dn[src_rank] = self._mpi_comm.Irecv([buf_dn, MPI.DOUBLE], source=src_rank, tag=201)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-4. Prepare and post nonblocking sends using Isend. ---
            start_ = time.perf_counter()
            send_requests = []
            for dest_rank, group in groupby(sorted(incoming_reqs, key=lambda x: x[0]), key=lambda x: x[0]):
                idxs = [src_local for (_, src_local, _) in group]
                buf_up = latest_r_up_carts_before_branching[idxs]
                buf_dn = latest_r_dn_carts_before_branching[idxs]
                send_requests.append(self._mpi_comm.Isend([buf_up, MPI.DOUBLE], dest=dest_rank, tag=200))
                send_requests.append(self._mpi_comm.Isend([buf_dn, MPI.DOUBLE], dest=dest_rank, tag=201))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-5. Wait for all nonblocking sends to complete. ---
            start_ = time.perf_counter()
            MPI.Request.Waitall(send_requests)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-6. Process the received walker data. ---
            start_ = time.perf_counter()
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                recv_reqs_up[src_rank].Wait()
                recv_reqs_dn[src_rank].Wait()
                buf_up, buf_dn = recv_buffers[src_rank]
                dest_idxs = [dest for (dest, _) in req_list]
                latest_r_up_carts_after_branching[dest_idxs] = buf_up
                latest_r_dn_carts_after_branching[dest_idxs] = buf_dn
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # np.array -> jnp.array
            self._num_survived_walkers += num_survived_walkers
            self._num_killed_walkers += num_killed_walkers
            self._stored_average_projection_counter[self._mcmc_counter + num_mcmc_done] = ave_projection_counter
            if _rs_gfmc_chunk:
                self._latest_r_up_carts = latest_r_up_carts_after_branching
                self._latest_r_dn_carts = latest_r_dn_carts_after_branching
            else:
                self._latest_r_up_carts = jnp.asarray(latest_r_up_carts_after_branching, dtype=jnp.float64)
                self._latest_r_dn_carts = jnp.asarray(latest_r_dn_carts_after_branching, dtype=jnp.float64)
            # SM-carry reseed: JAX projection path only (the native chunk evaluates
            # self-contained -- no A_inv crosses the boundary).
            if not _rs_gfmc_chunk:
                self.__latest_A_old_inv = _seed_gfmc_t_geminal_inverse(
                    self._hamiltonian_data.wavefunction_data.geminal_data, self._latest_r_up_carts, self._latest_r_dn_carts
                )
                # block before Barrier so the A_inv GPU work is included in timer_reconfiguration
                # and dispatch-queue skew does not leak into the next step's barrier wait.
                self.__latest_A_old_inv.block_until_ready()

            # Barrier after MPI operation
            if self._mpi_size > 1:
                self._mpi_comm.Barrier()

            # timer end
            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration

            # count up, here is the end of the branching step. Count it before the
            # stop checks below: a step that has run (walkers, RNG and stored
            # observables) must be kept when the loop stops after it.
            num_mcmc_done += 1

            # check current time
            # Use rank 0's decision and broadcast to all ranks to prevent
            # MPI deadlock caused by different ranks disagreeing on whether
            # to break (due to slight differences in gfmc_total_start).
            gmfc_current = time.perf_counter()
            should_stop = bool(max_time < gmfc_current - gfmc_total_start) if self._mpi_rank == 0 else False
            if self._mpi_size > 1:
                should_stop = self._mpi_comm.bcast(should_stop, root=0)
            if should_stop:
                logger.info(f"  Stopping... Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

            # check the control file (stop flag); shared across all ranks / twists
            poll_control = (i_branching + 1) % control_poll_interval == 0 or i_branching + 1 == num_mcmc_steps
            stop_flag = _read_external_stop_flag(toml_filename, self._mpi_comm) if poll_control else False
            if stop_flag:
                logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                logger.info("  Break the mcmc loop.")
                break

        progress = (num_mcmc_done + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
        gmfc_total_current = time.perf_counter()
        logger.info(
            f"  branching step = {num_mcmc_done + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %. Elapsed time = {(gmfc_total_current - gfmc_loop_start):.1f} sec."
        )
        logger.info("")

        # count up
        self._mcmc_counter += num_mcmc_done

        # truncate stored arrays to actual number of steps
        ns = self._mcmc_counter
        # average_projection_counter is stored on all ranks
        self._stored_average_projection_counter = self._stored_average_projection_counter[:ns]
        # other observables are stored on rank 0 only
        if self._mpi_rank == 0:
            self._stored_e_L = self._stored_e_L[:ns]
            self._stored_e_L2 = self._stored_e_L2[:ns]
            self._stored_w_L = self._stored_w_L[:ns]
            if self._mpc:
                self._stored_mpc_dV = self._stored_mpc_dV[:ns]
            if self._comput_position_deriv:
                self._stored_force_HF = self._stored_force_HF[:ns]
                self._stored_force_PP = self._stored_force_PP[:ns]
                self._stored_E_L_force_PP = self._stored_E_L_force_PP[:ns]
            if self._comput_stress:
                self._stored_stress_HF = self._stored_stress_HF[:ns]
                self._stored_stress_PP = self._stored_stress_PP[:ns]
                self._stored_E_L_stress_PP = self._stored_E_L_stress_PP[:ns]
            self.__validate_stored_shapes()

        gfmc_total_end = time.perf_counter()
        timer_gfmc_total = gfmc_total_end - gfmc_total_start
        # The stress block runs inside the walker-observable-collection window; report it separately.
        timer_collection -= timer_stress
        timer_misc = timer_gfmc_total - (
            timer_projection_init
            + timer_projection_total
            + timer_observable
            + timer_mpi_barrier
            + timer_reconfiguration
            + timer_collection
            + timer_stress
        )

        # remove the control file (world root owns it; EAFP-guarded)
        self._mpi_comm.Barrier()
        _remove_external_control_file(toml_filename, self._world_comm)

        # net GFMC time
        timer_net_gfmc_total = timer_gfmc_total - timer_projection_init

        # average among MPI processes
        ave_timer_gfmc_total = self._mpi_comm.allreduce(timer_gfmc_total, op=MPI.SUM) / self._mpi_size
        ave_timer_projection_init = self._mpi_comm.allreduce(timer_projection_init, op=MPI.SUM) / self._mpi_size
        ave_timer_net_gfmc_total = self._mpi_comm.allreduce(timer_net_gfmc_total, op=MPI.SUM) / self._mpi_size
        ave_timer_projection_total = (
            self._mpi_comm.allreduce(timer_projection_total, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        )
        ave_timer_observable = self._mpi_comm.allreduce(timer_observable, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_mpi_barrier = self._mpi_comm.allreduce(timer_mpi_barrier, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_collection = self._mpi_comm.allreduce(timer_collection, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_stress = self._mpi_comm.allreduce(timer_stress, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_reconfiguration = self._mpi_comm.allreduce(timer_reconfiguration, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_misc = self._mpi_comm.allreduce(timer_misc, op=MPI.SUM) / self._mpi_size / num_mcmc_done

        sum_killed_walkers = self._mpi_comm.allreduce(self._num_killed_walkers, op=MPI.SUM)
        sum_survived_walkers = self._mpi_comm.allreduce(self._num_survived_walkers, op=MPI.SUM)
        ave_stored_average_projection_counter = (
            self._mpi_comm.allreduce(np.mean(self._stored_average_projection_counter), op=MPI.SUM) / self._mpi_size
        )

        logger.info(f"Total GFMC time for {num_mcmc_done} branching steps = {ave_timer_gfmc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {ave_timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {ave_timer_net_gfmc_total: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_mcmc_done} branching steps.")
        logger.info(f"  Projection time per branching = {ave_timer_projection_total * 10**3: .3f} msec.")
        logger.info(f"  Time for Observable measurement time per branching = {ave_timer_observable * 10**3: .3f} msec.")
        logger.info(f"  Time for computing de_L/deps and dln_Psi/deps = {ave_timer_stress * 10**3:.2f} msec.")
        logger.info(f"  Time for MPI barrier before branching = {ave_timer_mpi_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for walker observable collections time per branching = {ave_timer_collection * 10**3: .3f} msec.")
        logger.info(f"  Time for walker reconfiguration time per branching = {ave_timer_reconfiguration * 10**3: .3f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(
            f"Survived walkers ratio = {sum_survived_walkers / (sum_survived_walkers + sum_killed_walkers) * 100:.2f} %"
        )
        logger.info(f"Average of the number of projections  = {ave_stored_average_projection_counter:.0f}")
        logger.info("")

        _log_memory_usage("gfmc_t_post_run", comm=self._mpi_comm)


class _GFMC_t_debug(_GFMCBase):
    """GFMC class.

    GFMC class. Running GFMC.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_gfmc_collect_steps (int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        tau (float):
            imaginary time between branchings (Ha^-1)
        alat (float):
            discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. All-electron calculations, do not specify this value.
        random_discretized_mesh (bool):
            Flag for the random discretization mesh in the kinetic part and in the non-local part of ECPs.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
        comput_stress (bool):
            if True, compute and retain periodic stress observables. Defaults to False.
        epsilon_PW (float):
            Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is
            regularized near the nodal surface using ``f_epsilon`` from
            `Pathak and Wagner (2020) <https://doi.org/10.1063/5.0004008>`_.
            Default is 0.0 (no regularization).
        use_swct (bool):
            if True, apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
            Default is False for GFMC.
        mpi_comm_in (MPI.Comm | None):
            sampling communicator of this instance. ``None`` uses the module global ``mpi_comm``
            (``COMM_WORLD``); the twist-averaged driver passes one communicator per twist.
        world_comm_in (MPI.Comm | None):
            all-rank communicator. ``None`` uses ``mpi_comm_in``.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        num_walkers: int = 40,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        tau: float = 0.1,
        alat: float = 0.1,
        non_local_move: NonLocalMove = "tmove",
        random_discretized_mesh: bool = False,
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        epsilon_PW: float = 0.0,
        use_swct: bool = False,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Initialize a _GFMC_t_debug driver and set up walker state."""
        # Communicator injection on the debug
        # sibling so production / debug stay uniform under twist driving.
        # Defaults fall back to module global mpi_comm (byte-identical).
        self._mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        self._world_comm = world_comm_in if world_comm_in is not None else self._mpi_comm
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_size = self._mpi_comm.Get_size()

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # attributes
        self._hamiltonian_data = hamiltonian_data
        self._num_walkers = num_walkers
        self._num_gfmc_collect_steps = num_gfmc_collect_steps
        self._mcmc_seed = mcmc_seed
        self._tau = tau
        self._alat = alat
        self._non_local_move = non_local_move
        self._random_discretized_mesh = random_discretized_mesh

        # derivative flags
        self._comput_position_deriv = comput_position_deriv
        self._comput_stress = comput_stress
        if self._comput_stress and not self._comput_position_deriv:
            raise ValueError("comput_stress=True requires comput_position_deriv=True.")
        self._use_swct = use_swct

        # Pathak-Wagner regularization (DOI: 10.1063/5.0004008)
        self._epsilon_PW = epsilon_PW

        # Initialization
        # Hierarchical SeedSequence derivation (single derivation point:
        # ``derive_rank_rng_streams``; stream key = [*chain_seed_words, rank]).
        # ``_mpi_seed`` = derived 63-bit JAX base-key int; ``_np_seed_words``
        # = eight uint32 init_by_array words for the NumPy global RNG.
        self._np_seed_words, self._mpi_seed = derive_rank_rng_streams(self._mcmc_seed, self._mpi_rank)
        self._branching_rng = np.random.RandomState(self._np_seed_words)
        self._jax_PRNG_key = jax.random.PRNGKey(self._mpi_seed)
        # Use jax.random.split (batched) instead of a Python list-comp of
        # fold_in calls; the latter scaled linearly with num_walkers and
        # dominated init time at large walker counts (e.g. nw = 16384).
        self._jax_PRNG_key_list_init = jax.random.split(self._jax_PRNG_key, self._num_walkers)
        self._jax_PRNG_key_list = self._jax_PRNG_key_list_init

        # initialize random seed
        np.random.seed(self._np_seed_words)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp
        # Cell for PBC wrapping of the seed configuration (None for molecules).
        cell = hamiltonian_data.structure_data.cell if hamiltonian_data.structure_data.pbc_flag else None

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self._num_walkers, charges, coords, cell
        )

        # Electron assignment for all atoms is complete.

        dtype_jnp = jnp.float64
        self._latest_r_up_carts = jnp.asarray(r_carts_up, dtype=dtype_jnp)
        self._latest_r_dn_carts = jnp.asarray(r_carts_dn, dtype=dtype_jnp)

        logger.debug(f"  initial r_up_carts= {self._latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self._latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self._latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self._latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {self._mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self._num_walkers}.")
        logger.info("")

        # SWCT data: the SWCT functions take structure_data directly.

        self.__init_attributes()

    def __init_attributes(self) -> None:
        # mcmc counter
        self._mcmc_counter = 0

        # gfmc accepted/rejected moves
        self._num_survived_walkers = 0
        self._num_killed_walkers = 0

        # e_L is complex per walker at a non-Gamma Bloch twist.
        # Mirrors the MCMC / production GFMC pattern.
        self._wf_is_complex = np.issubdtype(
            self._hamiltonian_data.wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating
        )

        # stored local energy (w_L)
        self._stored_w_L = []

        # stored local energy (e_L)
        self._stored_e_L = []

        # stored local energy (e_L)
        self._stored_e_L2 = []

        # average projection counter
        self._stored_average_projection_counter = []

        # stored force_HF (pre-computed force product)
        self._stored_force_HF = []

        # stored force_PP (pre-computed force product)
        self._stored_force_PP = []

        # stored E_L * force_PP (pre-computed force product)
        self._stored_E_L_force_PP = []

        # stored per-step HF / Pulay stress tensors (1, 3, 3) and e_L * stress_PP.
        # Mirrors the production GFMC buffers so the debug route exposes
        # get_aT / get_aP for the production-vs-debug stress parity.
        self._stored_stress_HF = []
        self._stored_stress_PP = []
        self._stored_E_L_stress_PP = []

    # collecting factor
    @property
    def num_gfmc_collect_steps(self) -> int:
        """Return num_gfmc_collect_steps."""
        return self._num_gfmc_collect_steps

    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Return hamiltonian_data."""
        return self._hamiltonian_data

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        return _compute_G_L_debug(np.array(self._stored_w_L), self._num_gfmc_collect_steps)

    def _retained_w_L(self, num_warmup_steps: int) -> npt.NDArray:
        """Accumulated weights after ``num_warmup_steps``, scaled by one common power of two (debug twin)."""
        return _compute_retained_G_L_debug(np.array(self._stored_w_L), self._num_gfmc_collect_steps, num_warmup_steps)

    @property
    def average_projection_counter(self) -> npt.NDArray:
        """Return the stored average projection counter array."""
        return np.array(self._stored_average_projection_counter)

    def run(self, num_mcmc_steps: int = 50) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_mcmc_steps (int): number of branching steps (reconfigurations of walkers).
        """
        logger.info("")
        logger.info("This is a debugging class! It supposed to be very slow.")
        logger.info("")

        # The ln|Psi| drift (r), Pulay
        # force (R) / Pulay stress (eps) grads route through the unified
        # ``jqmc.derivatives`` interface with ``analytic=_ANALYTIC_GRAD_ENABLED``,
        # looped per walker (this class is the slow oracle). The e_L
        # Hellmann-Feynman force and stress follow the SAME flag: when enabled they use the verified analytic
        # continuum-measurement kernels ``gfmc.grad_e_L_t_wrt_{positions,strain}``,
        # else the novar=0 measurement ``grad`` / ``jacrev`` fallback of
        # ``_compute_local_energy_t_debug`` (flag off only; no configuration is
        # carved out to autograd -- see ``_analytic_hf_enabled_for``).
        # So production AND this debug oracle share ONE derivative backend flag.
        # ln|Psi| is potential-free so its axes are LRDMC-independent.
        # Imported lazily to avoid the ``derivatives._auto`` -> ``jqmc_mcmc``
        # strained-helper import cycle.
        from . import derivatives

        _pos_flags = {"opt_r": True, "opt_R": True}
        _eps_flags = {"opt_eps": True}

        def _generate_rotation_matrix_t_debug(
            alpha: jnpt.ArrayLike,
            beta: jnpt.ArrayLike,
            gamma: jnpt.ArrayLike,
        ) -> jax.Array:
            # Precompute all necessary cosines and sines
            cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
            cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
            cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)

            # Combine the rotations directly
            R = jnp.array(
                [
                    [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                    [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                    [-sin_b, cos_b * sin_a, cos_a * cos_b],
                ],
                dtype=jnp.float64,
            )
            return R

        def _compute_V_elements_t_debug(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> jax.Array:
            """Compute V elements."""
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute discretized kinetic energy and mesh (with a random rotation)
            _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=RT,
            )
            # spin-filp
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            # regularizations
            elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
            sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
            non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1)
            sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
            non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                non_diagonal_kinetic_part_elements, [len(r_up_carts)]
            )

            # compute diagonal elements, el-el
            diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_coulomb_potential_ei_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_coulomb_potential_ei_discretized_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_up = (
                diagonal_bare_coulomb_part_el_ion_elements_up
                + diagonal_kinetic_continuum_elements_up
                - non_diagonal_kinetic_part_elements_up
            )
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
            else:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
            diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
            )
            diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_dn = (
                diagonal_bare_coulomb_part_el_ion_elements_dn
                + diagonal_kinetic_continuum_elements_dn
                - non_diagonal_kinetic_part_elements_dn
            )
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
            else:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
            diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
            )
            diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
            )

            # final bare coulomb part
            discretized_diagonal_bare_coulomb_part = (
                diagonal_bare_coulomb_part_el_el
                + diagonal_bare_coulomb_part_ion_ion
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
            )

            # with ECP
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                # ecp local
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    _, _, V_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        flag_determinant_only=False,
                        RT=RT,
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=True,
                            RT=RT,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                    Jastrow_ref = compute_Jastrow_part_ratio_zone(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = jnp.stack(
                        [
                            compute_Jastrow_part_ratio_zone(
                                hamiltonian_data.wavefunction_data.jastrow_data,
                                mesh_non_local_ecp_part_r_up_carts[_i],
                                mesh_non_local_ecp_part_r_dn_carts[_i],
                            )
                            for _i in range(len(mesh_non_local_ecp_part_r_up_carts))
                        ]
                    )
                    Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                else:
                    raise NotImplementedError

                # novar=0 split (debug twin): forces/stress differentiate the
                # MEASUREMENT (continuum) local energy.
                e_L = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                    + diagonal_ecp_local_part
                    + diagonal_ecp_part_SP
                    + non_diagonal_sum_hamiltonian_ecp
                )

            # with all electrons
            else:
                non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                # novar=0 split (debug twin) -- see ECP branch.
                e_L = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                )

            return e_L

        def _compute_local_energy_t_debug(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> jax.Array:
            return _compute_V_elements_t_debug(hamiltonian_data, r_up_carts, r_dn_carts, RT, non_local_move, alat)

        # Main branching loop.
        gfmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))  # gfmc_projection set print-interval

        logger.info("-Start branching-")

        num_mcmc_done = 0
        for i_branching in range(num_mcmc_steps):
            if i_branching % gfmc_interval == 0:
                progress = (i_branching + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
                logger.info(
                    f"  branching step = {i_branching + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %."
                )

            # Always set the initial weight list to 1.0
            projection_counter_list = jnp.array([0 for _ in range(self._num_walkers)])
            # e_L is complex per walker at a non-Gamma Bloch twist.
            _e_L_dtype_jnp = jnp.complex128 if self._wf_is_complex else jnp.float64
            e_L_list = jnp.array([1.0 for _ in range(self._num_walkers)], dtype=_e_L_dtype_jnp)
            w_L_list = jnp.array([1.0 for _ in range(self._num_walkers)], dtype=jnp.float64)

            logger.devel("  Projection is on going....")

            # projection loop
            projection_counter_list = np.array(projection_counter_list)
            e_L_list = np.array(e_L_list)
            w_L_list = np.array(w_L_list)
            latest_r_up_carts = np.array(self._latest_r_up_carts)
            latest_r_dn_carts = np.array(self._latest_r_dn_carts)
            jax_PRNG_key_list = np.array(self._jax_PRNG_key_list)
            latest_RTs = np.empty((self._num_walkers, 3, 3), dtype=np.float64)

            non_local_move = self._non_local_move
            alat = self._alat
            hamiltonian_data = self._hamiltonian_data

            for i_walker in range(self._num_walkers):
                projection_counter = projection_counter_list[i_walker]
                tau_left = self._tau
                w_L = w_L_list[i_walker]

                r_up_carts = latest_r_up_carts[i_walker]
                r_dn_carts = latest_r_dn_carts[i_walker]
                jax_PRNG_key = jax_PRNG_key_list[i_walker]
                R = jnp.eye(3, dtype=jnp.float64)
                ecp_R = jnp.eye(3, dtype=jnp.float64)

                while tau_left > 0.0:
                    projection_counter += 1

                    # coulomb regularization
                    # compute diagonal elements, kinetic part
                    diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

                    # compute continuum kinetic energy
                    diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                        compute_kinetic_energy_all_elements(
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                    )

                    # Preserve one rotation-key split per substep, but build the
                    # mesh rotation only on the first substep of this projection.
                    jax_PRNG_key, rotation_key = jax.random.split(jax_PRNG_key)
                    if projection_counter == 1:
                        kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rotation_key)
                        R = _sample_lrdmc_rotation_matrix(kinetic_key, self._random_discretized_mesh)
                        ecp_R = _sample_lrdmc_rotation_matrix(ecp_key, self._random_discretized_mesh)

                    # compute discretized kinetic energy and mesh (with a random rotation)
                    mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                        compute_discretized_kinetic_energy(
                            alat=alat,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            RT=R.T,
                        )
                    )
                    # spin-filp
                    elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                    non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
                    diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                    # regularizations
                    elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
                    sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
                    non_diagonal_kinetic_part_elements = jnp.sum(
                        elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1
                    )
                    sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(
                        sign_flip_flags_elements, [len(r_up_carts)]
                    )
                    non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                        non_diagonal_kinetic_part_elements, [len(r_up_carts)]
                    )

                    # compute diagonal elements, el-el
                    diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    # compute diagonal elements, ion-ion
                    diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                    )

                    # compute diagonal elements, el-ion
                    diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                        compute_coulomb_potential_ei_dispatch(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                    )

                    # compute diagonal elements, el-ion, discretized
                    (
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                    ) = compute_coulomb_potential_ei_discretized_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        alat=alat,
                    )

                    # compose discretized el-ion potentials
                    diagonal_bare_coulomb_part_el_ion_zv_up = (
                        diagonal_bare_coulomb_part_el_ion_elements_up
                        + diagonal_kinetic_continuum_elements_up
                        - non_diagonal_kinetic_part_elements_up
                    )
                    # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                    # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
                    else:
                        diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
                    diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                        diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
                    )
                    diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                        sign_flip_flags_elements_up,
                        diagonal_bare_coulomb_part_el_ion_max_up,
                        diagonal_bare_coulomb_part_el_ion_zv_up,
                    )

                    # compose discretized el-ion potentials
                    diagonal_bare_coulomb_part_el_ion_zv_dn = (
                        diagonal_bare_coulomb_part_el_ion_elements_dn
                        + diagonal_kinetic_continuum_elements_dn
                        - non_diagonal_kinetic_part_elements_dn
                    )
                    # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                    # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
                    else:
                        diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
                    diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                        diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
                    )
                    diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                        sign_flip_flags_elements_dn,
                        diagonal_bare_coulomb_part_el_ion_max_dn,
                        diagonal_bare_coulomb_part_el_ion_zv_dn,
                    )

                    # final bare coulomb part
                    discretized_diagonal_bare_coulomb_part = (
                        diagonal_bare_coulomb_part_el_el
                        + diagonal_bare_coulomb_part_ion_ion
                        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                        + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
                    )

                    # if-else for all-electron, ECP with tmove, and ECP with dltmove
                    # with ECP
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                        # ecp local
                        diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )

                        if non_local_move == "tmove":
                            # ecp non-local (t-move)
                            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                                compute_ecp_non_local_parts_nearest_neighbors(
                                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                    wavefunction_data=hamiltonian_data.wavefunction_data,
                                    r_up_carts=r_up_carts,
                                    r_dn_carts=r_dn_carts,
                                    flag_determinant_only=False,
                                    RT=ecp_R.T,
                                )
                            )

                            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                            non_diagonal_sum_hamiltonian = (
                                non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp
                            )

                        elif non_local_move == "dltmove":
                            mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                                compute_ecp_non_local_parts_nearest_neighbors(
                                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                    wavefunction_data=hamiltonian_data.wavefunction_data,
                                    r_up_carts=r_up_carts,
                                    r_dn_carts=r_dn_carts,
                                    flag_determinant_only=True,
                                    RT=ecp_R.T,
                                )
                            )

                            V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                            diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                            Jastrow_ref = compute_Jastrow_part_ratio_zone(
                                jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                            )

                            Jastrow_on_mesh = vmap(compute_Jastrow_part_ratio_zone, in_axes=(None, 0, 0))(
                                hamiltonian_data.wavefunction_data.jastrow_data,
                                mesh_non_local_ecp_part_r_up_carts,
                                mesh_non_local_ecp_part_r_dn_carts,
                            )
                            Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                            V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                            non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                            non_diagonal_sum_hamiltonian = (
                                non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp
                            )

                        else:
                            logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
                            raise NotImplementedError

                        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                        e_L = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                            + non_diagonal_sum_hamiltonian
                        )

                        # Measurement local energy (novar=0 split) -- debug twin of
                        # the production ``_projection_t_core`` assembly.
                        e_L_meas = (
                            diagonal_bare_coulomb_part_el_el
                            + diagonal_bare_coulomb_part_ion_ion
                            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                            + diagonal_ecp_local_part
                            + diagonal_ecp_part_SP
                            + non_diagonal_sum_hamiltonian_ecp
                        )

                        p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
                        # Under complex Bloch fixed-phase, take Re for the
                        # categorical sampling so the cdf passed to searchsorted is
                        # real-valued. Bit-identical at Gamma where p_list is real.
                        p_list = jnp.real(p_list)
                        non_diagonal_move_probabilities = p_list / p_list.sum()
                        non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
                            [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
                        )
                        non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
                            [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
                        )

                    # with all electrons
                    else:
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                        # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                        e_L = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_kinetic_part_SP
                            + non_diagonal_sum_hamiltonian
                        )

                        # Measurement local energy (novar=0 split) -- debug twin.
                        e_L_meas = (
                            diagonal_bare_coulomb_part_el_el
                            + diagonal_bare_coulomb_part_ion_ion
                            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                            + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                        )

                        p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                        # Under complex Bloch fixed-phase, take Re for the
                        # categorical sampling so the cdf passed to searchsorted is
                        # real-valued. Bit-identical at Gamma where p_list is real.
                        p_list = jnp.real(p_list)
                        non_diagonal_move_probabilities = p_list / p_list.sum()
                        non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                        non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

                    logger.devel(f"  e_L={e_L}")

                    # compute the time the walker remaining in the same configuration
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    xi = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                    # Complex Bloch fixed-phase mirrors the GFMC_t production route.
                    # fixed-phase branching discards that phase component.
                    tau_update = jnp.minimum(tau_left, jnp.log(1 - xi) / jnp.real(non_diagonal_sum_hamiltonian))
                    logger.debug(f"  tau_update={tau_update}")

                    # update weight
                    w_L = w_L * jnp.exp(-tau_update * jnp.real(e_L))

                    # update tau_left
                    tau_left = tau_left - tau_update
                    logger.debug(f"tau_left = {tau_left}.")

                    if tau_left <= 0.0:  # '= is very important!!'
                        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                        break
                    # electron position update
                    # random choice
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    cdf = jnp.cumsum(non_diagonal_move_probabilities)
                    random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                    k = jnp.searchsorted(cdf, random_value)
                    r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                    r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

                    # Walker carry stays unwrapped (in R^3), mirroring the
                    # production GFMC_t kernel: the
                    # walker carry must not be wrapped between state updates;
                    # ``compute_AOs`` handles periodicity internally via the
                    # wrap + ``_restore_phase`` pair so AO evaluation at the
                    # unwrapped ``r`` satisfies the Bloch BC exactly.

                projection_counter_list[i_walker] = projection_counter
                # Store the measurement local energy (novar=0 split); the
                # propagation weight above used the clipped lattice e_L.
                e_L_list[i_walker] = e_L_meas
                w_L_list[i_walker] = w_L
                latest_r_up_carts[i_walker] = r_up_carts
                latest_r_dn_carts[i_walker] = r_dn_carts
                jax_PRNG_key_list[i_walker] = jax_PRNG_key
                # The projection evaluates its final local energy with R.T.
                # Preserve that exact interval rotation for the observable
                # derivatives, matching the production GFMC_t carry contract.
                latest_RTs[i_walker] = np.asarray(ecp_R.T)

            # adjust jax_PRNG_key (consistent with the production code)
            num_max_projection = np.max(projection_counter_list)
            for i_walker in range(self._num_walkers):
                jax_PRNG_key = jax_PRNG_key_list[i_walker]
                for _ in range(num_max_projection - projection_counter_list[i_walker]):
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                    jax_PRNG_key, _ = jax.random.split(jax_PRNG_key)
                jax_PRNG_key_list[i_walker] = jax_PRNG_key

            # projection ends
            projection_counter_list = jnp.array(projection_counter_list)
            e_L_list = jnp.asarray(e_L_list, dtype=_e_L_dtype_jnp)
            w_L_list = jnp.asarray(w_L_list, dtype=jnp.float64)
            self._latest_r_up_carts = jnp.asarray(latest_r_up_carts, dtype=jnp.float64)
            self._latest_r_dn_carts = jnp.asarray(latest_r_dn_carts, dtype=jnp.float64)
            self._jax_PRNG_key_list = jnp.array(jax_PRNG_key_list)

            logger.debug("  Projection ends.")

            # atomic force related
            if self._comput_position_deriv:
                # The e_L Hellmann-Feynman force uses the same derivative backend
                # flag as production. When ``_ANALYTIC_GRAD_ENABLED``, route through the verified
                # analytic continuum-measurement kernel
                # ``jqmc.derivatives.gfmc.grad_e_L_t_wrt_positions`` using the same
                # final interval rotation as the debug forward (complex-twist DTN
                # included; ``_assert_scope`` rejects nothing). Otherwise (flag off)
                # fall back to the LRDMC lattice-discretised ``jax.grad`` oracle, whose
                # per-path structure-position cotangents are hand-summed.
                if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
                    from .derivatives import gfmc as _gfmc_deriv

                    _hf_results = [
                        _gfmc_deriv.grad_e_L_t_wrt_positions(
                            self._hamiltonian_data,
                            self._latest_r_up_carts[i],
                            self._latest_r_dn_carts[i],
                            latest_RTs[i],
                            self._alat,
                            non_local_move=self._non_local_move,
                        )
                        for i in range(self._num_walkers)
                    ]
                    grad_e_L_R = jnp.stack([r[0] for r in _hf_results])
                    grad_e_L_r_up = jnp.stack([r[1] for r in _hf_results])
                    grad_e_L_r_dn = jnp.stack([r[2] for r in _hf_results])
                else:
                    # Project complex Bloch local energy to its physical real component.
                    # The physical force differentiates Re(e_L).
                    def _real_compute_local_energy_t_debug(*args: Any, **kwargs: Any) -> jax.Array:
                        return jnp.real(_compute_local_energy_t_debug(*args, **kwargs))

                    _grad_e_L_fn = grad(_real_compute_local_energy_t_debug, argnums=(0, 1, 2))
                    _grad_e_L_results = [
                        _grad_e_L_fn(
                            self._hamiltonian_data,
                            self._latest_r_up_carts[i],
                            self._latest_r_dn_carts[i],
                            latest_RTs[i],
                            self._non_local_move,
                            self._alat,
                        )
                        for i in range(self._num_walkers)
                    ]
                    grad_e_L_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *[r[0] for r in _grad_e_L_results])
                    grad_e_L_r_up = jnp.stack([r[1] for r in _grad_e_L_results])
                    grad_e_L_r_dn = jnp.stack([r[2] for r in _grad_e_L_results])

                    grad_e_L_R = (
                        grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.structure_data.positions
                        + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.structure_data.positions
                        + grad_e_L_h.coulomb_potential_data.structure_data.positions
                    )

                    # Which Jastrow channels carry geometry, and where the leaf
                    # sits inside each, is Jastrow-internal knowledge -- the
                    # driver only asks for their summed contribution
                    # so this driver remains independent of channel membership.
                    _jastrow_R = structure_positions_grad_Jastrow(
                        self._hamiltonian_data.wavefunction_data.jastrow_data,
                        grad_e_L_h.wavefunction_data.jastrow_data,
                    )
                    if _jastrow_R is not None:
                        grad_e_L_R += _jastrow_R

                # dln|Psi|/dr (drift) and dln|Psi|/dR
                # (Pulay) use the derivatives interface, looped per walker, with
                # the ``_ANALYTIC_GRAD_ENABLED`` flag (same backend as production); ln|Psi| is
                # potential-free so it is LRDMC/non_local_move-independent (no carve-out). The "R"
                # axis returns the per-atom force already summed by ``accumulate_position_grad``.
                _grad_ln_psi_results = [
                    derivatives.compute_grad_ln_Psi(
                        self._hamiltonian_data,
                        self._latest_r_up_carts[i],
                        self._latest_r_dn_carts[i],
                        _pos_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                    )
                    for i in range(self._num_walkers)
                ]
                grad_ln_Psi_r_up = jnp.stack([r["r"][0] for r in _grad_ln_psi_results])
                grad_ln_Psi_r_dn = jnp.stack([r["r"][1] for r in _grad_ln_psi_results])
                grad_ln_Psi_dR = jnp.stack([r["R"] for r in _grad_ln_psi_results])

                if self._use_swct:
                    omega_up = jnp.stack(
                        [
                            evaluate_swct_omega(self._hamiltonian_data.structure_data, self._latest_r_up_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    omega_dn = jnp.stack(
                        [
                            evaluate_swct_omega(self._hamiltonian_data.structure_data, self._latest_r_dn_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    grad_omega_dr_up = jnp.stack(
                        [
                            evaluate_swct_domega(self._hamiltonian_data.structure_data, self._latest_r_up_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    grad_omega_dr_dn = jnp.stack(
                        [
                            evaluate_swct_domega(self._hamiltonian_data.structure_data, self._latest_r_dn_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                else:
                    _n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
                    _n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
                    _n_atoms = self._hamiltonian_data.structure_data.natom
                    omega_up = jnp.zeros((self._num_walkers, _n_atoms, _n_up), dtype=jnp.float64)
                    omega_dn = jnp.zeros((self._num_walkers, _n_atoms, _n_dn), dtype=jnp.float64)
                    grad_omega_dr_up = jnp.zeros((self._num_walkers, _n_atoms, 3), dtype=jnp.float64)
                    grad_omega_dr_dn = jnp.zeros((self._num_walkers, _n_atoms, 3), dtype=jnp.float64)

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)

            if self._comput_position_deriv:
                grad_e_L_r_up_latest = np.array(grad_e_L_r_up)
                grad_e_L_r_dn_latest = np.array(grad_e_L_r_dn)
                grad_e_L_R_latest = np.array(grad_e_L_R)
                grad_ln_Psi_r_up_latest = np.array(grad_ln_Psi_r_up)
                grad_ln_Psi_r_dn_latest = np.array(grad_ln_Psi_r_dn)
                grad_ln_Psi_dR_latest = np.array(grad_ln_Psi_dR)
                omega_up_latest = np.array(omega_up)
                omega_dn_latest = np.array(omega_dn)
                grad_omega_dr_up_latest = np.array(grad_omega_dr_up)
                grad_omega_dr_dn_latest = np.array(grad_omega_dr_dn)

                # Debug per-walker Hellmann-Feynman and Pulay stress
                # (sigma_HF = d e_L / d eps, sigma_PP = d ln|Psi| / d eps at
                # eps = 0). Each walker uses the same final interval rotation as
                # the debug projection and HF force above.
                # Periodic only. The HF stress uses the
                # ``_ANALYTIC_GRAD_ENABLED`` flag (analytic continuum-measurement kernel
                # ``gfmc.grad_e_L_t_wrt_strain`` when enabled, else the local LRDMC
                # jacrev fallback); the Pulay stress "eps" axis follows the same flag.
                if self._hamiltonian_data.structure_data.pbc_flag:
                    _zero_eps = jnp.zeros((3, 3), dtype=jnp.float64)
                    if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
                        from .derivatives import gfmc as _gfmc_deriv

                        _stress_HF_latest = np.array(
                            [
                                np.asarray(
                                    _gfmc_deriv.grad_e_L_t_wrt_strain(
                                        self._hamiltonian_data,
                                        self._latest_r_up_carts[i],
                                        self._latest_r_dn_carts[i],
                                        latest_RTs[i],
                                        self._alat,
                                        non_local_move=self._non_local_move,
                                    )
                                )
                                for i in range(self._num_walkers)
                            ]
                        )
                    else:

                        def _real_e_L_t_strained(
                            hamiltonian_data: Hamiltonian_data,
                            r_up_carts: jax.Array,
                            r_dn_carts: jax.Array,
                            RT: jax.Array,
                            non_local_move: NonLocalMove,
                            alat: float,
                            strain: jax.Array,
                        ) -> jax.Array:
                            F = jnp.eye(3, dtype=jnp.float64) + strain
                            new_h = hamiltonian_data._apply_strain(F)
                            return jnp.real(
                                _compute_local_energy_t_debug(
                                    new_h, r_up_carts @ F.T, r_dn_carts @ F.T, RT, non_local_move, alat
                                )
                            )

                        _stress_HF_fn = jax.jacrev(_real_e_L_t_strained, argnums=6)
                        _stress_HF_latest = np.array(
                            [
                                np.asarray(
                                    _stress_HF_fn(
                                        self._hamiltonian_data,
                                        self._latest_r_up_carts[i],
                                        self._latest_r_dn_carts[i],
                                        latest_RTs[i],
                                        self._non_local_move,
                                        self._alat,
                                        _zero_eps,
                                    )
                                )
                                for i in range(self._num_walkers)
                            ]
                        )
                    _stress_PP_latest = np.array(
                        [
                            np.asarray(
                                derivatives.compute_grad_ln_Psi(
                                    self._hamiltonian_data,
                                    self._latest_r_up_carts[i],
                                    self._latest_r_dn_carts[i],
                                    _eps_flags,
                                    analytic=_ANALYTIC_GRAD_ENABLED,
                                )["eps"]
                            )
                            for i in range(self._num_walkers)
                        ]
                    )
                else:
                    _stress_HF_latest = np.zeros((self._num_walkers, 3, 3), dtype=np.float64)
                    _stress_PP_latest = np.zeros((self._num_walkers, 3, 3), dtype=np.float64)

            # jnp.array -> np.array
            latest_r_up_carts_before_branching = np.array(self._latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self._latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = latest_r_up_carts_before_branching.shape
            r_up_carts_gathered_dyad = (self._mpi_rank, latest_r_up_carts_before_branching)
            r_up_carts_gathered_dyad = self._mpi_comm.gather(r_up_carts_gathered_dyad, root=0)

            r_dn_carts_shape = latest_r_dn_carts_before_branching.shape
            r_dn_carts_gathered_dyad = (self._mpi_rank, latest_r_dn_carts_before_branching)
            r_dn_carts_gathered_dyad = self._mpi_comm.gather(r_dn_carts_gathered_dyad, root=0)

            e_L_gathered_dyad = (self._mpi_rank, e_L_latest)
            e_L_gathered_dyad = self._mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = (self._mpi_rank, w_L_latest)
            w_L_gathered_dyad = self._mpi_comm.gather(w_L_gathered_dyad, root=0)

            if self._comput_position_deriv:
                grad_e_L_r_up_dyad = (self._mpi_rank, grad_e_L_r_up_latest)
                grad_e_L_r_dn_dyad = (self._mpi_rank, grad_e_L_r_dn_latest)
                grad_e_L_R_dyad = (self._mpi_rank, grad_e_L_R_latest)
                grad_ln_Psi_r_up_dyad = (self._mpi_rank, grad_ln_Psi_r_up_latest)
                grad_ln_Psi_r_dn_dyad = (self._mpi_rank, grad_ln_Psi_r_dn_latest)
                grad_ln_Psi_dR_dyad = (self._mpi_rank, grad_ln_Psi_dR_latest)
                omega_up_dyad = (self._mpi_rank, omega_up_latest)
                omega_dn_dyad = (self._mpi_rank, omega_dn_latest)
                grad_omega_dr_up_dyad = (self._mpi_rank, grad_omega_dr_up_latest)
                grad_omega_dr_dn_dyad = (self._mpi_rank, grad_omega_dr_dn_latest)
                stress_HF_dyad = (self._mpi_rank, _stress_HF_latest)
                stress_PP_dyad = (self._mpi_rank, _stress_PP_latest)

                grad_e_L_r_up_dyad = self._mpi_comm.gather(grad_e_L_r_up_dyad, root=0)
                grad_e_L_r_dn_dyad = self._mpi_comm.gather(grad_e_L_r_dn_dyad, root=0)
                grad_e_L_R_dyad = self._mpi_comm.gather(grad_e_L_R_dyad, root=0)
                grad_ln_Psi_r_up_dyad = self._mpi_comm.gather(grad_ln_Psi_r_up_dyad, root=0)
                grad_ln_Psi_r_dn_dyad = self._mpi_comm.gather(grad_ln_Psi_r_dn_dyad, root=0)
                grad_ln_Psi_dR_dyad = self._mpi_comm.gather(grad_ln_Psi_dR_dyad, root=0)
                omega_up_dyad = self._mpi_comm.gather(omega_up_dyad, root=0)
                omega_dn_dyad = self._mpi_comm.gather(omega_dn_dyad, root=0)
                grad_omega_dr_up_dyad = self._mpi_comm.gather(grad_omega_dr_up_dyad, root=0)
                grad_omega_dr_dn_dyad = self._mpi_comm.gather(grad_omega_dr_dn_dyad, root=0)
                stress_HF_dyad = self._mpi_comm.gather(stress_HF_dyad, root=0)
                stress_PP_dyad = self._mpi_comm.gather(stress_PP_dyad, root=0)

            # num projection counter
            ## Compute the local average of the projection counter list.
            ave_projection_counter = np.mean(projection_counter_list)

            if self._mpi_rank == 0:
                zeta = float(self._branching_rng.random())
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(self._mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(self._mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(self._mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(self._mpi_size)])
                # The real variance buffer stores |e_L|^2,
                # mirroring the MCMC and GFMC_n production patterns.
                e_L2_averaged = np.sum(w_L_gathered * np.abs(e_L_gathered) ** 2) / np.sum(w_L_gathered)
                e_L_averaged = np.sum(w_L_gathered * e_L_gathered) / np.sum(w_L_gathered)
                w_L_averaged = np.average(w_L_gathered)
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                # store  # This should stored only for MPI-rank = 0 !!!
                self._stored_e_L2.append(e_L2_averaged)
                self._stored_e_L.append(e_L_averaged)
                self._stored_w_L.append(w_L_averaged)

                if self._comput_position_deriv:
                    # dict
                    grad_e_L_r_up_gathered_dict = dict(grad_e_L_r_up_dyad)
                    grad_e_L_r_dn_gathered_dict = dict(grad_e_L_r_dn_dyad)
                    grad_e_L_R_gathered_dict = dict(grad_e_L_R_dyad)
                    grad_ln_Psi_r_up_gathered_dict = dict(grad_ln_Psi_r_up_dyad)
                    grad_ln_Psi_r_dn_gathered_dict = dict(grad_ln_Psi_r_dn_dyad)
                    grad_ln_Psi_dR_gathered_dict = dict(grad_ln_Psi_dR_dyad)
                    omega_up_gathered_dict = dict(omega_up_dyad)
                    omega_dn_gathered_dict = dict(omega_dn_dyad)
                    grad_omega_dr_up_gathered_dict = dict(grad_omega_dr_up_dyad)
                    grad_omega_dr_dn_gathered_dict = dict(grad_omega_dr_dn_dyad)
                    # gathered
                    grad_e_L_r_up_gathered = np.concatenate([grad_e_L_r_up_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_e_L_r_dn_gathered = np.concatenate([grad_e_L_r_dn_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_e_L_R_gathered = np.concatenate([grad_e_L_R_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_ln_Psi_r_up_gathered = np.concatenate(
                        [grad_ln_Psi_r_up_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_ln_Psi_r_dn_gathered = np.concatenate(
                        [grad_ln_Psi_r_dn_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_ln_Psi_dR_gathered = np.concatenate([grad_ln_Psi_dR_gathered_dict[i] for i in range(self._mpi_size)])
                    omega_up_gathered = np.concatenate([omega_up_gathered_dict[i] for i in range(self._mpi_size)])
                    omega_dn_gathered = np.concatenate([omega_dn_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_omega_dr_up_gathered = np.concatenate(
                        [grad_omega_dr_up_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_omega_dr_dn_gathered = np.concatenate(
                        [grad_omega_dr_dn_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    # Pathak-Wagner regularization factor f_eps per walker
                    if self._epsilon_PW > 0.0:
                        grad_norm_sq = np.sum(grad_ln_Psi_r_up_gathered**2, axis=(1, 2)) + np.sum(
                            grad_ln_Psi_r_dn_gathered**2, axis=(1, 2)
                        )
                        nodal_dist = 1.0 / np.sqrt(grad_norm_sq)
                        t = nodal_dist / self._epsilon_PW
                        t2 = t * t
                        t4 = t2 * t2
                        t6 = t4 * t2
                        f_eps = np.where(t < 1.0, 7.0 * t6 - 15.0 * t4 + 9.0 * t2, 1.0)
                        mod_w_L_gathered = w_L_gathered * f_eps
                    else:
                        mod_w_L_gathered = w_L_gathered
                    # weighted sum (using w_L directly, no V_diag_E in GFMC_t)
                    # compute per-walker force products
                    force_HF_per_walker = (
                        grad_e_L_R_gathered
                        + np.einsum("ijk,ikl->ijl", omega_up_gathered, grad_e_L_r_up_gathered)
                        + np.einsum("ijk,ikl->ijl", omega_dn_gathered, grad_e_L_r_dn_gathered)
                    )
                    force_PP_per_walker = (
                        grad_ln_Psi_dR_gathered
                        + np.einsum("ijk,ikl->ijl", omega_up_gathered, grad_ln_Psi_r_up_gathered)
                        + np.einsum("ijk,ikl->ijl", omega_dn_gathered, grad_ln_Psi_r_dn_gathered)
                        + 0.5 * (grad_omega_dr_up_gathered + grad_omega_dr_dn_gathered)
                    )
                    E_L_force_PP_per_walker = e_L_gathered[:, None, None] * force_PP_per_walker
                    # weighted sum
                    force_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, force_HF_per_walker)
                    force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, force_PP_per_walker)
                    E_L_force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, E_L_force_PP_per_walker)
                    # averaged (using sum of w_L)
                    w_L_sum = np.sum(w_L_gathered)
                    force_HF_averaged = force_HF_weighted_sum / w_L_sum
                    force_PP_averaged = force_PP_weighted_sum / w_L_sum
                    E_L_force_PP_averaged = E_L_force_PP_weighted_sum / w_L_sum
                    # add a dummy dim
                    force_HF_averaged = np.expand_dims(force_HF_averaged, axis=0)
                    force_PP_averaged = np.expand_dims(force_PP_averaged, axis=0)
                    E_L_force_PP_averaged = np.expand_dims(E_L_force_PP_averaged, axis=0)
                    # store
                    self._stored_force_HF.append(force_HF_averaged)
                    self._stored_force_PP.append(force_PP_averaged)
                    self._stored_E_L_force_PP.append(E_L_force_PP_averaged)

                    # stress: reconstruct gathered per-walker tensors and weight-
                    # average (mirror the force block; same mod_w_L / w_L_sum).
                    stress_HF_gathered = np.concatenate([dict(stress_HF_dyad)[i] for i in range(self._mpi_size)])
                    stress_PP_gathered = np.concatenate([dict(stress_PP_dyad)[i] for i in range(self._mpi_size)])
                    E_L_stress_PP_per_walker = e_L_gathered[:, None, None] * stress_PP_gathered
                    stress_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, stress_HF_gathered)
                    stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, stress_PP_gathered)
                    E_L_stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered, E_L_stress_PP_per_walker)
                    self._stored_stress_HF.append(np.expand_dims(stress_HF_weighted_sum / w_L_sum, axis=0))
                    self._stored_stress_PP.append(np.expand_dims(stress_PP_weighted_sum / w_L_sum, axis=0))
                    self._stored_E_L_stress_PP.append(np.expand_dims(E_L_stress_PP_weighted_sum / w_L_sum, axis=0))

                # branching
                probabilities = w_L_gathered / w_L_gathered.sum()
                # correlated choice (see F. Becca and S. Sorella, Quantum Monte Carlo Approaches for Correlated Systems (2017), p. 182)
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices_old = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices_old]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices_old]

                num_survived_walkers = len(set(chosen_walker_indices_old))
                num_killed_walkers = len(w_L_gathered) - len(set(chosen_walker_indices_old))
            else:
                num_survived_walkers = None
                num_killed_walkers = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            num_survived_walkers = self._mpi_comm.bcast(num_survived_walkers, root=0)
            num_killed_walkers = self._mpi_comm.bcast(num_killed_walkers, root=0)

            proposed_r_up_carts = self._mpi_comm.bcast(proposed_r_up_carts, root=0)
            proposed_r_dn_carts = self._mpi_comm.bcast(proposed_r_dn_carts, root=0)

            proposed_r_up_carts = proposed_r_up_carts.reshape(
                self._mpi_size, r_up_carts_shape[0], r_up_carts_shape[1], r_up_carts_shape[2]
            )
            proposed_r_dn_carts = proposed_r_dn_carts.reshape(
                self._mpi_size, r_dn_carts_shape[0], r_dn_carts_shape[1], r_dn_carts_shape[2]
            )

            # set new r_up_carts and r_dn_carts, and, np.array -> jnp.array
            latest_r_up_carts_after_branching = proposed_r_up_carts[self._mpi_rank, :, :, :]
            latest_r_dn_carts_after_branching = proposed_r_dn_carts[self._mpi_rank, :, :, :]

            # np.array -> jnp.array
            self._num_survived_walkers += num_survived_walkers
            self._num_killed_walkers += num_killed_walkers
            self._stored_average_projection_counter.append(ave_projection_counter)
            self._latest_r_up_carts = jnp.asarray(latest_r_up_carts_after_branching, dtype=jnp.float64)
            self._latest_r_dn_carts = jnp.asarray(latest_r_dn_carts_after_branching, dtype=jnp.float64)

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

        progress = (num_mcmc_done + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
        logger.info(
            f"  branching step = {num_mcmc_done + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %."
        )
        logger.info("")

        # count up mcmc_counter
        self._mcmc_counter += num_mcmc_done

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[float, float, float, float]:
                The mean and std values of the totat energy and those of the variance
                estimated by the Jackknife method with the Args. (E_mean, E_std, Var_mean, Var_std).
        """
        if self._mpi_rank == 0:
            e_L = self.e_L[num_mcmc_warmup_steps:]
            e_L2 = self.e_L2[num_mcmc_warmup_steps:]
            w_L = self._retained_w_L(num_mcmc_warmup_steps)
            # Shape assertions: (M, nw) from mcmc_counter and num_walkers
            M = self.mcmc_counter - num_mcmc_warmup_steps
            nw = 1  # GFMC observables are reduced across walkers
            _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
            _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
            _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
            w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

            w_L_binned_local = w_L_binned
            w_L_e_L_binned_local = w_L_e_L_binned
            w_L_e_L2_binned_local = w_L_e_L2_binned

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_global_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            # Mirror production get_E and take Re of
            # the per-binned-sample ratio so the energy observable / variance
            # are real on the complex Bloch path; the variance subtracts the
            # modulus square ``|<e_L>|**2``. w_L weights are real already
            # (fixed-phase). Bit-identical at Gamma.
            E_jackknife_binned_complex = np.array(
                [
                    (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )
            abs_E_sq_local = np.abs(E_jackknife_binned_complex) ** 2
            E_jackknife_binned_local = list(np.real(E_jackknife_binned_complex))

            E2_jackknife_binned_local = list(
                np.real(
                    np.array(
                        [
                            (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m])
                            / (w_L_binned_global_sum - w_L_binned_local[m])
                            for m in range(M_local)
                        ]
                    )
                )
            )

            Var_jackknife_binned_local = list(np.array(E2_jackknife_binned_local) - abs_E_sq_local)

            # convert to numpy array
            E_jackknife_binned = np.array(E_jackknife_binned_local)
            Var_jackknife_binned = np.array(Var_jackknife_binned_local)
            M_total = len(E_jackknife_binned)
            logger.debug(f"The number of total binned samples = {M_total}")

            # jackknife mean and std
            E_mean = np.average(E_jackknife_binned)
            E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
            Var_mean = np.average(Var_jackknife_binned)
            Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

            logger.info(f"E = {E_mean} +- {E_std} Ha.")
            logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        else:
            E_mean = None
            E_std = None
            Var_mean = None
            Var_std = None

        # MPI broadcast
        E_mean = self._mpi_comm.bcast(E_mean, root=0)
        E_std = self._mpi_comm.bcast(E_std, root=0)
        Var_mean = self._mpi_comm.bcast(Var_mean, root=0)
        Var_std = self._mpi_comm.bcast(Var_std, root=0)

        # return
        return (E_mean, E_std, Var_mean, Var_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Return the mean and std of the computed atomic forces.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]:
                The mean and std values of the computed atomic forces
                estimated by the Jackknife method with the Args.
                The dimention of the arrays is (N, 3).
        """
        if self._mpi_rank == 0:
            w_L = self._retained_w_L(num_mcmc_warmup_steps)
            e_L = self.e_L[num_mcmc_warmup_steps:]
            force_HF = self.force_HF[num_mcmc_warmup_steps:]
            force_PP = self.force_PP[num_mcmc_warmup_steps:]
            E_L_force_PP = self.E_L_force_PP[num_mcmc_warmup_steps:]
            # Shape assertions: (M, nw) from mcmc_counter and num_walkers
            M = self.mcmc_counter - num_mcmc_warmup_steps
            nw = 1  # GFMC observables are reduced across walkers
            n_atoms = self.hamiltonian_data.structure_data.natom
            _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
            _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
            _require(force_HF.shape == (M, nw, n_atoms, 3), f"force_HF shape {force_HF.shape} != ({M}, {nw}, {n_atoms}, 3)")
            _require(force_PP.shape == (M, nw, n_atoms, 3), f"force_PP shape {force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)")
            _require(
                E_L_force_PP.shape == (M, nw, n_atoms, 3),
                f"E_L_force_PP shape {E_L_force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)",
            )

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
            w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
            w_L_E_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0)

            # binned sum
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

            w_L_force_HF_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_HF_split])
            w_L_force_HF_binned_shape = (
                w_L_force_HF_sum.shape[0] * w_L_force_HF_sum.shape[1],
                w_L_force_HF_sum.shape[2],
                w_L_force_HF_sum.shape[3],
            )
            w_L_force_HF_binned = list(w_L_force_HF_sum.reshape(w_L_force_HF_binned_shape))

            w_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_PP_split])
            w_L_force_PP_binned_shape = (
                w_L_force_PP_sum.shape[0] * w_L_force_PP_sum.shape[1],
                w_L_force_PP_sum.shape[2],
                w_L_force_PP_sum.shape[3],
            )
            w_L_force_PP_binned = list(w_L_force_PP_sum.reshape(w_L_force_PP_binned_shape))

            w_L_E_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_E_L_force_PP_split])
            w_L_E_L_force_PP_binned_shape = (
                w_L_E_L_force_PP_sum.shape[0] * w_L_E_L_force_PP_sum.shape[1],
                w_L_E_L_force_PP_sum.shape[2],
                w_L_E_L_force_PP_sum.shape[3],
            )
            w_L_E_L_force_PP_binned = list(w_L_E_L_force_PP_sum.reshape(w_L_E_L_force_PP_binned_shape))

            w_L_binned_local = np.array(w_L_binned)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned)
            w_L_force_HF_binned_local = np.array(w_L_force_HF_binned)
            w_L_force_PP_binned_local = np.array(w_L_force_PP_binned)
            w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned)

            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_force_HF_binned_global_sum = np.sum(w_L_force_HF_binned_local, axis=0)
            w_L_force_PP_binned_global_sum = np.sum(w_L_force_PP_binned_local, axis=0)
            w_L_E_L_force_PP_binned_global_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            force_HF_jn_local = -1.0 * np.array(
                [
                    (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    for j in range(M_local)
                ]
            )

            force_Pulay_jn_local = -2.0 * np.array(
                [
                    (
                        (w_L_E_L_force_PP_binned_global_sum - w_L_E_L_force_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        - (
                            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                            * (w_L_force_PP_binned_global_sum - w_L_force_PP_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                        )
                    )
                    for j in range(M_local)
                ]
            )

            # Mirror production get_aF and re-project the
            # per-binned-sample HF + Pulay sum so the force observable is real
            # on the complex Bloch path. Bit-identical at Gamma.
            force_jn_local = list(np.real(force_HF_jn_local + force_Pulay_jn_local))

            force_jn = force_jn_local
            M_total = len(force_jn)
            logger.debug(f"The number of total binned samples = {M_total}")

            force_mean = np.average(force_jn, axis=0)
            force_std = np.sqrt(M_total - 1) * np.std(force_jn, axis=0)

            logger.devel(f"force_mean.shape  = {force_mean.shape}.")
            logger.devel(f"force_std.shape  = {force_std.shape}.")
            logger.info(f"force = {force_mean} +- {force_std} Ha.")

        else:
            force_mean = None
            force_std = None

        # broadcast the results
        force_mean = self._mpi_comm.bcast(force_mean, root=0)
        force_std = self._mpi_comm.bcast(force_std, root=0)

        return (force_mean, force_std)

    def get_aT(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Symmetric stress tensor with jackknife statistics (debug route, LRDMC/PBC).

        Independent counterpart of :py:meth:`GFMC_t.get_aT`: same single-rank
        bin-block jackknife (branching weights already absorbed into the
        per-step walker average), assembled from the looped-jacrev stress
        buffers stored by the debug ``run``. Molecular structures return zeros.

        Contract (fixed static ansatz): same as :py:meth:`GFMC_t.get_aT` --
        the strain ``F = I + eps`` maps cell, nuclei and electrons as
        ``x -> F x`` (ECP quadrature directions fixed) with basis exponents / coefficients, MO /
        geminal coefficients and Jastrow parameters held fixed, so the
        tensor is NOT the equation-of-state derivative of a re-optimized
        ground-state energy.
        """
        if not self._comput_stress:
            raise ValueError("Stress data were not computed; construct the GFMC driver with comput_stress=True.")
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3, 3)), np.zeros((3, 3))
        if self._mpi_rank != 0:
            return np.zeros((3, 3)), np.zeros((3, 3))

        w_L = self._retained_w_L(num_mcmc_warmup_steps)
        e_L = self.e_L[num_mcmc_warmup_steps:]
        stress_HF = self.stress_HF[num_mcmc_warmup_steps:]
        stress_PP = self.stress_PP[num_mcmc_warmup_steps:]
        E_L_stress_PP = self.E_L_stress_PP[num_mcmc_warmup_steps:]

        M = stress_HF.shape[0]
        _require(w_L.shape == (M, 1), "")
        _require(e_L.shape == (M, 1), "")
        _require(stress_HF.shape == (M, 1, 3, 3), "")
        _require(stress_PP.shape == (M, 1, 3, 3), "")
        _require(E_L_stress_PP.shape == (M, 1, 3, 3), "")

        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_stress_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_HF), num_mcmc_bin_blocks, axis=0)
        w_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_PP), num_mcmc_bin_blocks, axis=0)
        w_L_E_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_stress_PP), num_mcmc_bin_blocks, axis=0)

        w_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split])))
        w_L_e_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split])))

        def _bin_tensor(splits: Sequence[npt.NDArray]) -> npt.NDArray:
            tensor_sum = np.array([np.sum(arr, axis=0) for arr in splits])
            return tensor_sum.reshape(tensor_sum.shape[0] * tensor_sum.shape[1], 3, 3)

        w_L_stress_HF_binned = _bin_tensor(w_L_stress_HF_split)
        w_L_stress_PP_binned = _bin_tensor(w_L_stress_PP_split)
        w_L_E_L_stress_PP_binned = _bin_tensor(w_L_E_L_stress_PP_split)

        w_L_sum = float(np.sum(w_L_binned))
        # Keep the (possibly complex at a non-Gamma twist) dtype through the
        # jackknife; the physical-real projection happens on the
        # assembled sigma below. A float() cast here would drop Im(e_L) and
        # emit a ComplexWarning.
        w_L_e_L_sum = np.sum(w_L_e_L_binned)
        w_L_stress_HF_sum = np.sum(w_L_stress_HF_binned, axis=0)
        w_L_stress_PP_sum = np.sum(w_L_stress_PP_binned, axis=0)
        w_L_E_L_stress_PP_sum = np.sum(w_L_E_L_stress_PP_binned, axis=0)

        M_local = w_L_binned.size

        sigma_HF_jn = np.array(
            [(w_L_stress_HF_sum - w_L_stress_HF_binned[j]) / (w_L_sum - w_L_binned[j]) for j in range(M_local)]
        )
        sigma_Pulay_jn = 2.0 * np.array(
            [
                (
                    (w_L_E_L_stress_PP_sum - w_L_E_L_stress_PP_binned[j]) / (w_L_sum - w_L_binned[j])
                    - (
                        (w_L_e_L_sum - w_L_e_L_binned[j])
                        / (w_L_sum - w_L_binned[j])
                        * (w_L_stress_PP_sum - w_L_stress_PP_binned[j])
                        / (w_L_sum - w_L_binned[j])
                    )
                )
                for j in range(M_local)
            ]
        )

        sigma_jn = np.real(sigma_HF_jn + sigma_Pulay_jn)
        sigma_jn = 0.5 * (sigma_jn + sigma_jn.transpose(0, 2, 1))
        omega = float(np.abs(np.linalg.det(np.asarray(self._hamiltonian_data.structure_data._cell_jnp))))
        sigma_jn = sigma_jn / omega

        sigma_mean = np.mean(sigma_jn, axis=0)
        var_sigma = np.mean((sigma_jn - sigma_mean) ** 2, axis=0)
        sigma_std = np.sqrt((M_local - 1) * var_sigma)
        return sigma_mean, sigma_std

    def get_aP(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Normal-pressure diagonal for LRDMC (debug route; mirror of :py:meth:`GFMC_t.get_aP`)."""
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3,)), np.zeros((3,))
        if self._mpi_rank != 0:
            return np.zeros((3,)), np.zeros((3,))
        sigma_mean, sigma_std = self.get_aT(num_mcmc_warmup_steps, num_mcmc_bin_blocks)
        return -np.diag(sigma_mean), np.diag(sigma_std)


class GFMC_n(_GFMCBase):
    """GFMC class. Running GFMC with multiple walkers.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_mcmc_per_measurement (int):
            the number of GFMC projections between two branchings (measurements), an integer >= 1.
            Required, no default (as ``[lrdmc-bra] num_mcmc_per_measurement`` in the CLI): omitting it
            or passing ``None`` raises ``ValueError`` before any sampling. One projection advances
            imaginary time by about ``alat**2 / (3 N_e)``, so the value sets the imaginary time per
            measurement and depends on the system and on ``alat``. The given integer is used unchanged.
        num_gfmc_collect_steps (int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        E_scf (float):
            Self-consistent E (Hartree)
        alat (float):
            finite, strictly positive discretized grid length (bohr)
        random_discretized_mesh (bool):
            Flag for the random discretization mesh in the kinetic part and the non-local part of ECPs.
            Valid both for all-electron and ECP calculations.
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. Do not specify this value for all-electron calculations.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
        comput_stress (bool):
            if True, compute and retain periodic stress observables. Defaults to False.
        epsilon_PW (float):
            Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is
            regularized near the nodal surface using ``f_epsilon`` from
            `Pathak and Wagner (2020) <https://doi.org/10.1063/5.0004008>`_.
            Default is 0.0 (no regularization).
        use_swct (bool):
            if True, apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
            Default is False for GFMC.
        mpc (bool):
            if True, evaluate the passive MPC finite-size correction: on each stored branching step
            ``dV = V_ee^MPC - V_ee^Ewald`` is evaluated on the same walker configurations (see
            :meth:`get_dE_mpc`); MPC never enters the propagation, the weights or the forces.
            Requires a periodic ``hamiltonian_data``. Default is False.
        mpc_cutoff (float):
            reciprocal-space kinetic-energy cutoff (Ha) of the MPC FFT grid. Ignored when ``mpc``
            is False. Default is ``MPC_CUTOFF_HA``.
        mpi_comm_in (MPI.Comm | None):
            sampling communicator of this instance. ``None`` uses the module global ``mpi_comm``
            (``COMM_WORLD``); the twist-averaged driver passes one communicator per twist.
        world_comm_in (MPI.Comm | None):
            all-rank communicator. ``None`` uses ``mpi_comm_in``.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int | None = None,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        E_scf: float = 0.0,
        alat: float = 0.1,
        random_discretized_mesh: bool = True,
        non_local_move: NonLocalMove = "tmove",
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        epsilon_PW: float = 0.0,
        use_swct: bool = False,
        mpc: bool = False,
        mpc_cutoff: float = MPC_CUTOFF_HA,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Initialize a GFMC_n driver and set up walker state."""
        _validate_gfmc_alat(alat)
        _require_lrdmc_bra_num_mcmc_per_measurement(num_mcmc_per_measurement)
        _validate_num_mcmc_per_measurement(num_mcmc_per_measurement)
        self._num_mcmc_per_measurement = num_mcmc_per_measurement
        self._E_scf = E_scf
        super().__init__(
            hamiltonian_data=hamiltonian_data,
            num_walkers=num_walkers,
            num_gfmc_collect_steps=num_gfmc_collect_steps,
            mcmc_seed=mcmc_seed,
            alat=alat,
            random_discretized_mesh=random_discretized_mesh,
            non_local_move=non_local_move,
            comput_position_deriv=comput_position_deriv,
            comput_stress=comput_stress,
            epsilon_PW=epsilon_PW,
            use_swct=use_swct,
            mpc=mpc,
            mpc_cutoff=mpc_cutoff,
            mpi_comm_in=mpi_comm_in,
            world_comm_in=world_comm_in,
        )

    def _init_attributes_class_specific(self) -> None:
        # stored G_L and G_e_L for updating the E_scf (kept as lists -- variable count per run)
        self._G_L = []
        self._G_e_L = []

    def __validate_stored_shapes(self) -> None:
        """Assert that all stored observable arrays have consistent shapes."""
        ns = self._mcmc_counter
        n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        n_atoms = self._hamiltonian_data.structure_data.natom

        expected = {
            "e_L": ((ns, 1), self._stored_e_L),
            "e_L2": ((ns, 1), self._stored_e_L2),
            "w_L": ((ns, 1), self._stored_w_L),
        }
        if self._mpc:
            expected["mpc_dV"] = ((ns, 1), self._stored_mpc_dV)
        if self._comput_position_deriv:
            expected.update(
                {
                    "force_HF": ((ns, 1, n_atoms, 3), self._stored_force_HF),
                    "force_PP": ((ns, 1, n_atoms, 3), self._stored_force_PP),
                    "E_L_force_PP": ((ns, 1, n_atoms, 3), self._stored_E_L_force_PP),
                }
            )
        if self._comput_stress:
            expected.update(
                {
                    "stress_HF": ((ns, 1, 3, 3), self._stored_stress_HF),
                    "stress_PP": ((ns, 1, 3, 3), self._stored_stress_PP),
                    "E_L_stress_PP": ((ns, 1, 3, 3), self._stored_E_L_stress_PP),
                }
            )
        for name, (shape, arr) in expected.items():
            _require(arr.shape == shape, f"GFMC_n stored shape mismatch: {name} expected {shape}, got {arr.shape}")

    # ------------------------------------------------------------------
    # HDF5 checkpoint save / load
    # ------------------------------------------------------------------

    def save_to_hdf5(self, filepath: str) -> None:
        """Write this rank's GFMC_n state to a temporary per-rank HDF5 file."""
        super().save_to_hdf5(
            filepath,
            driver_type="GFMC_n",
            extra_config={
                "num_mcmc_per_measurement": self._num_mcmc_per_measurement,
                "E_scf": self._E_scf,
            },
            extra_observables={
                "G_L": np.array(self._G_L) if len(self._G_L) > 0 else np.empty(0),
                "G_e_L": np.array(self._G_e_L) if len(self._G_e_L) > 0 else np.empty(0),
            },
        )

    _RESTART_CLASS_CONFIG_KEYS = ("num_mcmc_per_measurement", "E_scf")

    @classmethod
    def _class_restart_history_errors(cls, obs: Mapping[str, Any], counter: int, rank: int) -> list[tuple[str, str]]:
        """The root-owned E_scf products: one per step after the first on-the-fly collect window."""
        from ._checkpoint import history_length_errors

        expected = max(0, counter - GFMC_ON_THE_FLY_COLLECT_STEPS) if rank == 0 else 0
        return history_length_errors(obs, ("G_L", "G_e_L"), expected, counter)

    @classmethod
    def _validate_restored_domains(cls, cfg: Mapping[str, Any], comm: MPI.Comm) -> None:
        """Collectively validate the restored lattice spacing and measurement interval."""
        super()._validate_restored_domains(cfg, comm)
        _validate_num_mcmc_per_measurement(cfg.get("num_mcmc_per_measurement"), comm=comm)

    def _load_class_specific(self, cfg: Mapping[str, Any], obs: Mapping[str, Any]) -> None:
        """Restore GFMC_n class-specific config + observables."""
        self._num_mcmc_per_measurement = cfg["num_mcmc_per_measurement"]
        self._E_scf = cfg["E_scf"]
        _g_l = obs.get("G_L")
        _g_e_l = obs.get("G_e_L")
        self._G_L = [_g_l[i] for i in range(_g_l.shape[0])] if _g_l is not None and _g_l.size > 0 else []
        self._G_e_L = [_g_e_l[i] for i in range(_g_e_l.shape[0])] if _g_e_l is not None and _g_e_l.size > 0 else []

    # hamiltonian
    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Return hamiltonian_data."""
        return self._hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data: Hamiltonian_data) -> None:
        """Set hamiltonian_data."""
        if self._comput_position_deriv or self._comput_stress:
            # Keep gradients for both params and coords when position derivatives are needed.
            self._hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=True, coords=True))
        else:
            self._hamiltonian_data = apply_diff_mask(hamiltonian_data, DiffMask(params=False, coords=False))
        self._init_attributes()

    # collecting factor
    @property
    def num_gfmc_collect_steps(self) -> int:
        """Return num_gfmc_collect_steps."""
        return self._num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps: int) -> None:
        """Set num_gfmc_collect_steps."""
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            logger.warning(f"num_gfmc_collect_steps should be larger than {GFMC_MIN_COLLECT_STEPS}")
        self._num_gfmc_collect_steps = num_gfmc_collect_steps

    def run(self, num_mcmc_steps: int = 50, max_time: int = 86400) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_mcmc_steps (int): number of branching steps (reconfigurations of walkers).
            max_time (int): maximum time in sec.
        """
        _check_mpi4py_jax_distribution_consistency()

        # set timer
        timer_projection_init = 0.0
        timer_projection_total = 0.0
        timer_e_L = 0.0
        timer_de_L_dR_dr = 0.0
        timer_dln_Psi_dR_dr = 0.0
        timer_dln_Psi_dc = 0.0
        timer_de_L_dc = 0.0
        timer_mpi_barrier = 0.0
        timer_reconfiguration = 0.0
        timer_collection = 0.0
        timer_stress = 0.0
        timer_update_E_scf = 0.0

        # toml(control) filename
        toml_filename = "external_control_gfmc.toml"

        # create the single per-invocation control file (world root owns it;
        # see _init_external_control_file for the twist-average rationale).
        _init_external_control_file(toml_filename, self._world_comm)
        self._mpi_comm.Barrier()

        gfmc_total_start = time.perf_counter()

        # Native GFMC_n projection-interval chunk gate; see
        # _resolve_native_gfmc_chunk for the complete conditions. Production
        # always uses the native RNG and mesh rotations. Fixed-input interval
        # comparisons call the Rust entry point directly from tests.
        # When engaged, one run_gfmc_projection_n call per walker per branching
        # step replaces the JAX projection fori_loop AND the post-projection
        # V-elements e_L evaluation (the chunk returns e_L / V_diag at the final
        # config with the last sub-step's mesh rotation, exactly what
        # _jit_vmap_V_elements_n computes). Branching / reconfiguration / E_scf
        # update / binning stay Python (MPI collectives), unchanged.
        _rs_gfmc = self._resolve_native_gfmc_chunk("n")
        _rs_gfmc_chunk = _rs_gfmc is not None
        _rs_mpc_observable = self._mpc and _rust_cpu_backend_active()
        if _rs_gfmc_chunk:
            _rs_obs_ctx = _rs_gfmc["ctx"]
            _rs_ecp_w = _rs_gfmc["ecp_w"]
            _rs_ecp_gp = _rs_gfmc["ecp_gp"]
            _rs_dltmove = _rs_gfmc["dltmove"]
            _rs_want_force = _rs_gfmc["want_force"]
            _rs_want_stress = _rs_gfmc["want_stress"]
            _rs_complex = _rs_gfmc["complex"]
            # Empty injected-rotation input selects native mesh generation.
            _RS_EMPTY_RTS_BATCH = np.zeros((0, 1, 3, 3), dtype=np.float64)

        # precompute geminal inverses per walker for fast updates across projections

        def _compute_initial_geminal_n(r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
            return compute_geminal_all_elements(
                geminal_data=self._hamiltonian_data.wavefunction_data.geminal_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

        # Reuse the cached jit callable when present. This one is
        # EXECUTED right here (it reseeds the running SM inverse at the
        # current walker positions), so only the compile is skippable.
        _driver_cache_early = getattr(self, "_projection_driver_cache", None)
        if _driver_cache_early is not None and _driver_cache_early["hamiltonian"] is self._hamiltonian_data:
            _jit_vmap_A_inv_n = _driver_cache_early["A_inv"]
        else:
            # The shared seed / refresh inverse of the whole walker batch (LU-first with an SVD
            # fallback off the CPU backend, issue #533).
            _jit_vmap_A_inv_n = jit(
                lambda r_up, r_dn: _geminal_pinv_batched(vmap(_compute_initial_geminal_n, in_axes=(0, 0))(r_up, r_dn))
            )
        self.__latest_A_old_inv = _jit_vmap_A_inv_n(self._latest_r_up_carts, self._latest_r_dn_carts)

        # projection function.
        @jit
        def _generate_rotation_matrix_n(
            alpha: jnpt.ArrayLike,
            beta: jnpt.ArrayLike,
            gamma: jnpt.ArrayLike,
        ) -> jax.Array:
            # Precompute all necessary cosines and sines
            cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
            cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
            cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)

            # Combine the rotations directly
            R = jnp.array(
                [
                    [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                    [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                    [-sin_b, cos_b * sin_a, cos_a * cos_b],
                ],
                dtype=jnp.float64,
            )
            return R

        @partial(jit, static_argnums=(6, 7, 8))
        def _projection_n(
            init_w_L: float,
            init_r_up_carts: jnpt.ArrayLike,
            init_r_dn_carts: jnpt.ArrayLike,
            init_A_old_inv: jnpt.ArrayLike,
            init_jax_PRNG_key: jnpt.ArrayLike,
            E_scf: float,
            num_mcmc_per_measurement: int,
            random_discretized_mesh: bool,
            non_local_move: NonLocalMove,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ) -> GFMCProjectionNResult:
            """Do projection, compatible with vmap.

            Do projection for a set of (r_up_cart, r_dn_cart).

            Args:
                init_w_L (float): weight before projection
                init_r_up_carts (jnpt.ArrayLike): up-electron positions (N_e^up, 3) before projection
                init_r_dn_carts (jnpt.ArrayLike): dn-electron positions (N_e^dn, 3) before projection
                init_A_old_inv (jnpt.ArrayLike): cached inverse geminal matrix before projection
                init_jax_PRNG_key (jnpt.ArrayLike): JAX PRNG key before projection
                E_scf (float): Self-consistent E (Hartree)
                num_mcmc_per_measurement (int): the number of GFMC projections per measurement
                random_discretized_mesh (bool): Flag for the random discretization mesh in the kinetic part.
                non_local_move (str): treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
                alat (float): discretized grid length (bohr)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data

            Returns:
                latest_w_L (float): weight after the final projection
                latest_r_up_carts (N_e^up, 3) after the final projection
                latest_r_dn_carts (N_e^dn, 3) after the final projection
                latest_A_old_inv: cached inverse geminal matrix after the final projection
                latest_RT (3, 3) rotation matrix used in the last projection
                latest_V_diag (float): diagonal part of H (importance sampled) at the last projection
                latest_V_nondiag (float): non-diagonal part of H (importance sampled) at the last projection
            """

            @jit
            def _body_step_core(
                i: jax.Array,
                w_L: jax.Array,
                r_up_carts: jax.Array,
                r_dn_carts: jax.Array,
                A_old_inv: jax.Array,
                diagonal_kinetic_continuum_elements_up: jax.Array,
                diagonal_kinetic_continuum_elements_dn: jax.Array,
                j3_state: Jastrow_three_body_streaming_state | None = None,
                det_ratio_state: Det_ratio_streaming_state | None = None,
                kinetic_mesh_state: Discretized_kinetic_mesh_streaming_state | None = None,
                ecp_mesh_state: Ecp_candidate_mesh_streaming_state | None = None,
                pot_state: GFMCIntervalPotState | None = None,
            ) -> GFMCProjectionNStepResult:
                """Single GFMC projection step, parameterized by per-electron continuum kinetic energy.

                Shared so that both the non-streaming path
                (which recomputes ``compute_kinetic_energy_all_elements_fast_update`` from
                scratch every step) and the streaming path (which reads the kinetic
                energy from a maintained ``Kinetic_streaming_state``) can share the
                identical post-kinetic logic. Returns the carry components plus the
                ``(moved_spin_is_up, moved_index)`` pair that the streaming path needs
                to advance its auxiliary state.

                The optional ``j3_state`` (a ``Jastrow_three_body_streaming_state``
                consistent with the current ``r_{up,dn}_carts``) lets the streaming
                path avoid re-evaluating J3 AOs / W,U / cross-vec auxiliaries inside
                the discretized-mesh kinetic ratio and the ECP non-local ratio. Pass
                ``None`` (default) on the non-streaming path; the callees fall back to fresh
                AO evaluation when ``j3_state is None``.
                """
                # compute diagonal elements, kinetic part
                diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

                # The first random rotation is held fixed for this projection
                # interval.  ``rotation_keys`` is still generated for every
                # substep below, so the PRNG key schedule is one split per substep.
                R = interval_R
                ecp_R = interval_ecp_R

                # compute discretized kinetic energy and mesh (with a random rotation)
                if kinetic_mesh_state is None:
                    # Same contract as the GFMC_t twin: the state-building
                    # sub-step cannot consume the state it produces.
                    with route_site("substep0"):
                        mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                            compute_discretized_kinetic_energy_fast_update(
                                alat=alat,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                A_old_inv=A_old_inv,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                RT=R.T,
                                j3_state=j3_state,
                                det_ratio_state=det_ratio_state,
                            )
                        )
                else:
                    mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                        compute_discretized_kinetic_energy_from_streaming_state(
                            alat=alat,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            A_old_inv=A_old_inv,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            mesh_state=kinetic_mesh_state,
                            j3_state=j3_state,
                            det_ratio_state=det_ratio_state,
                        )
                    )
                # spin-filp
                elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
                diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                # regularizations
                elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
                sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
                non_diagonal_kinetic_part_elements = jnp.sum(
                    elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1
                )
                sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(
                    sign_flip_flags_elements, [len(r_up_carts)]
                )
                non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                    non_diagonal_kinetic_part_elements, [len(r_up_carts)]
                )

                # Incremental interval potential: static trace-time
                # route decision -- see the GFMC_t `_projection_t_core` note.
                _use_pot = pot_state is not None and _gfmc_pot_active(hamiltonian_data)
                if _use_pot:
                    global _GFMC_POT_TRACE_WITNESS
                    _GFMC_POT_TRACE_WITNESS += 1
                    _n_up_static = r_up_carts.shape[0]
                    _n_e_static = _n_up_static + r_dn_carts.shape[0]
                    (_pot_e_real_half, _pot_s_re, _pot_s_im, _pot_ei_bare, _pot_ei_disc, _pot_ecp_local, _) = pot_state
                    diagonal_bare_coulomb_part_el_el = compute_ewald_ee_from_interval_state(
                        _pot_e_real_half,
                        _pot_s_re,
                        _pot_s_im,
                        _n_e_static,
                        hamiltonian_data.coulomb_potential_data.ewald_data,
                    )
                    diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                    )
                    diagonal_bare_coulomb_part_el_ion_elements_up = _pot_ei_bare[:_n_up_static]
                    diagonal_bare_coulomb_part_el_ion_elements_dn = _pot_ei_bare[_n_up_static:]
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_up = _pot_ei_disc[:_n_up_static]
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_dn = _pot_ei_disc[_n_up_static:]
                else:
                    # compute diagonal elements, el-el
                    diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    # compute diagonal elements, ion-ion
                    diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                    )

                    # compute diagonal elements, el-ion
                    diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                        compute_coulomb_potential_ei_dispatch(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                    )

                    # compute diagonal elements, el-ion, discretized
                    (
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                        diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                    ) = compute_coulomb_potential_ei_discretized_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        alat=alat,
                    )

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_up = (
                    diagonal_bare_coulomb_part_el_ion_elements_up
                    + diagonal_kinetic_continuum_elements_up
                    - non_diagonal_kinetic_part_elements_up
                )
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
                diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
                )
                diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                    sign_flip_flags_elements_up,
                    diagonal_bare_coulomb_part_el_ion_max_up,
                    diagonal_bare_coulomb_part_el_ion_zv_up,
                )

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_dn = (
                    diagonal_bare_coulomb_part_el_ion_elements_dn
                    + diagonal_kinetic_continuum_elements_dn
                    - non_diagonal_kinetic_part_elements_dn
                )
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
                diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
                )
                diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                    sign_flip_flags_elements_dn,
                    diagonal_bare_coulomb_part_el_ion_max_dn,
                    diagonal_bare_coulomb_part_el_ion_zv_dn,
                )

                # final bare coulomb part
                discretized_diagonal_bare_coulomb_part = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
                )

                # if-else for all-electron, ECP with tmove, and ECP with dltmove
                # with ECP
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    # ecp local (carried per-electron rows on the incremental
                    # route; electron-major sum -- an fp reordering of the
                    # reference channel-major total, twin-bounded)
                    if _use_pot:
                        diagonal_ecp_local_part = jnp.sum(_pot_ecp_local)
                    else:
                        diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                A_old_inv=A_old_inv,
                                RT=ecp_R.T,
                                j3_state=j3_state,
                                det_ratio_state=det_ratio_state,
                                candidate_mesh_state=ecp_mesh_state,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    elif non_local_move == "dltmove":
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                A_old_inv=A_old_inv,
                                RT=ecp_R.T,
                                j3_state=j3_state,
                                det_ratio_state=det_ratio_state,
                                candidate_mesh_state=ecp_mesh_state,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        with route_site("projection" if ecp_mesh_state is not None else "substep0"):
                            Jastrow_ratio = _compute_ratio_Jastrow_part_rank1_update(
                                jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                                old_r_up_carts=r_up_carts,
                                old_r_dn_carts=r_dn_carts,
                                new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                                new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                                j3_state=j3_state,
                                j1_candidate_state=(ecp_mesh_state.j1_candidate_state if ecp_mesh_state is not None else None),
                                _route_mesh="ecp" if ecp_mesh_state is not None else None,
                                j3_candidate_state=(ecp_mesh_state.j3_candidate_state if ecp_mesh_state is not None else None),
                                j2_candidate_state=(ecp_mesh_state.j2_candidate_state if ecp_mesh_state is not None else None),
                                j_dtn_candidate_state=(
                                    ecp_mesh_state.j_dtn_candidate_state if ecp_mesh_state is not None else None
                                ),
                            )

                        V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    else:
                        logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
                        raise NotImplementedError

                    # probability
                    p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
                    # Under complex Bloch fixed-phase, take Re for the
                    # categorical sampling so the cdf passed to searchsorted is
                    # real-valued. Bit-identical at Gamma where p_list is real.
                    p_list = jnp.real(p_list)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
                    )
                    non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
                    )

                # with all electrons
                else:
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                    # Under complex Bloch fixed-phase, take Re for the
                    # categorical sampling so the cdf passed to searchsorted is
                    # real-valued. Bit-identical at Gamma where p_list is real.
                    p_list = jnp.real(p_list)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                    non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP
                    )

                # compute b_L_bar
                # Complex Bloch uses the classical fixed-phase approximation
                # for GFMC_n. The branching factor b_x is derived from a
                # projector step exp(-(H - E_scf)); for complex Bloch we
                # project both diagonal and non-diagonal pieces onto their Re
                # parts so b_x stays real and w_L stays real-positive. Im
                # parts contribute only a per-walker phase, which fixed-phase
                # branching discards.
                # Bit-identical at Gamma where everything is real.
                b_x_bar = -1.0 * jnp.real(non_diagonal_sum_hamiltonian)

                # compute bar_b_L
                b_x = 1.0 / (jnp.real(diagonal_sum_hamiltonian) - E_scf) * b_x_bar

                # update weight
                w_L = w_L * b_x

                # electron position update
                # random choice
                move_key = move_keys[i]
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(move_key, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                proposed_r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                proposed_r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

                num_up_electrons = r_up_carts.shape[0]
                num_dn_electrons = r_dn_carts.shape[0]

                if num_up_electrons == 0:
                    has_up_move = False
                    up_index = 0
                else:
                    up_diff = jnp.any(r_up_carts != proposed_r_up_carts, axis=1)
                    has_up_move = jnp.any(up_diff)
                    up_index = jnp.argmax(up_diff)

                if num_dn_electrons == 0:
                    has_dn_move = False
                    dn_index = 0
                else:
                    dn_diff = jnp.any(r_dn_carts != proposed_r_dn_carts, axis=1)
                    has_dn_move = jnp.any(dn_diff)
                    dn_index = jnp.argmax(dn_diff)

                def _update_inv_up_n(_: None) -> jax.Array:
                    # v construction: streaming path uses cached AO +
                    # paired_dn from ``det_ratio_state``; non-streaming path falls
                    # back to the twice-called row helper.
                    if det_ratio_state is None:
                        v = (
                            compute_geminal_up_one_row_elements(
                                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                r_up_cart=jnp.reshape(proposed_r_up_carts[up_index], (1, 3)),
                                r_dn_carts=r_dn_carts,
                            )
                            - compute_geminal_up_one_row_elements(
                                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                r_up_cart=jnp.reshape(r_up_carts[up_index], (1, 3)),
                                r_dn_carts=r_dn_carts,
                            )
                        )[:, None]
                    else:
                        v = _compute_v_up_move_from_det_ratio_state(
                            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                            state=det_ratio_state,
                            moved_index=up_index,
                            r_up_carts_proposed=proposed_r_up_carts,
                        )[:, None]
                    u = jax.nn.one_hot(up_index, num_up_electrons)[:, None]
                    # The SM rank-1 update returns A_old_inv's dtype, so the result
                    # agrees with the _no_update_n lax.cond branch, and a complex
                    # Bloch A_old_inv keeps its imaginary part. Both lax.cond branches return
                    # the same dtype (A_old_inv.dtype) keeping the cond-output
                    # invariant intact.
                    _, A_new_inv = _sherman_morrison_ratio_and_inverse(A_old_inv, u, v, moved_index=up_index, moved_is_up=True)
                    return A_new_inv

                def _no_update_n(_: None) -> jax.Array:
                    return A_old_inv

                if num_dn_electrons == 0:
                    _update_inv_dn_n = _no_update_n
                else:

                    def _update_inv_dn_n(_: None) -> jax.Array:
                        if det_ratio_state is None:
                            u = (
                                compute_geminal_dn_one_column_elements(
                                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                    r_up_carts=r_up_carts,
                                    r_dn_cart=jnp.reshape(proposed_r_dn_carts[dn_index], (1, 3)),
                                )
                                - compute_geminal_dn_one_column_elements(
                                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                    r_up_carts=r_up_carts,
                                    r_dn_cart=jnp.reshape(r_dn_carts[dn_index], (1, 3)),
                                )
                            )[:, None]
                        else:
                            u = _compute_u_dn_move_from_det_ratio_state(
                                geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                                state=det_ratio_state,
                                moved_index=dn_index,
                                r_dn_carts_proposed=proposed_r_dn_carts,
                            )[:, None]
                        v = jax.nn.one_hot(dn_index, num_up_electrons)[:, None]
                        # See note in _update_inv_up_n: consumer-zone explicit cast.
                        # Preserve A_old_inv's complex Bloch dtype so its imaginary part
                        # survives the SM rank-1 update. Both lax.cond branches return
                        # the same dtype (A_old_inv.dtype) keeping the cond-output
                        # invariant intact.
                        _, A_new_inv = _sherman_morrison_ratio_and_inverse(
                            A_old_inv, u, v, moved_index=dn_index, moved_is_up=False
                        )
                        return A_new_inv

                if num_up_electrons == 0:
                    A_new_inv = A_old_inv
                else:
                    A_new_inv = lax.cond(
                        has_up_move,
                        _update_inv_up_n,
                        lambda __: lax.cond(has_dn_move, _update_inv_dn_n, _no_update_n, operand=None),
                        operand=None,
                    )

                # Advance the incremental interval potential to the post-hop
                # config (one-electron delta) BEFORE the position carry is
                # shadowed by the proposed positions.
                if _use_pot:
                    pot_state_new = _gfmc_advance_interval_pot_state(
                        hamiltonian_data,
                        pot_state,
                        r_up_carts,
                        r_dn_carts,
                        proposed_r_up_carts,
                        proposed_r_dn_carts,
                        has_up_move,
                        up_index,
                        has_dn_move,
                        dn_index,
                        alat,
                    )
                else:
                    pot_state_new = pot_state

                # Walker carry stays unwrapped (in R^3):
                # wrapping the carry between SM updates puts the running
                # ``A_new_inv`` out of phase with the carried walker at twist
                # k != 0 (``compute_AOs`` is wrap-equivariant). Mirror of the
                # MCMC ``_accepted_fun`` fix.
                r_up_carts = proposed_r_up_carts
                r_dn_carts = proposed_r_dn_carts

                # ``moved_index`` is whichever of (up_index, dn_index) is the
                # one whose spin actually moved this step. Used by the streaming
                # path to identify the column-changed by the rank-1 update.
                # GFMC_n always applies its hop, but the mover is named through
                # the same static per-spin rule as GFMC_t so that no advance can
                # ever be handed a nonexistent electron (issue #784 B).
                moved_spin_is_up, moved_index = _streaming_mover(
                    has_up_move, up_index, dn_index, num_up_electrons, num_dn_electrons
                )

                return (
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    R.T,
                    ecp_R.T,
                    A_new_inv,
                    diagonal_sum_hamiltonian,
                    non_diagonal_sum_hamiltonian,
                    moved_spin_is_up,
                    moved_index,
                    pot_state_new,
                )

            @jit
            def _body_fun_n(i: jax.Array, carry: GFMCProjectionNLoopState) -> GFMCProjectionNLoopState:
                """Non-streaming GFMC projection body -- recomputes kinetic energies fresh per step."""
                (
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    RT,
                    ecp_RT,
                    A_old_inv,
                    pot_state,
                    _,
                    _,
                ) = carry

                # compute continuum kinetic energy from scratch (non-streaming path).
                ke_up, ke_dn = compute_kinetic_energy_all_elements_fast_update(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    geminal_inverse=A_old_inv,
                )

                (
                    w_L_new,
                    r_up_new,
                    r_dn_new,
                    RT_new,
                    ecp_RT_new,
                    A_new_inv,
                    diag_sum_H,
                    nondiag_sum_H,
                    _moved_spin_is_up,
                    _moved_index,
                    pot_state_new,
                ) = _body_step_core(i, w_L, r_up_carts, r_dn_carts, A_old_inv, ke_up, ke_dn, pot_state=pot_state)

                return (
                    w_L_new,
                    r_up_new,
                    r_dn_new,
                    RT_new,
                    ecp_RT_new,
                    A_new_inv,
                    pot_state_new,
                    diag_sum_H,
                    nondiag_sum_H,
                )

            @jit
            def _body_fun_n_streaming(
                i: jax.Array,
                carry: GFMCProjectionNStreamingState,
            ) -> GFMCProjectionNStreamingState:
                """Streaming GFMC projection body -- reads kinetic energies from a maintained
                ``Kinetic_streaming_state`` (J3 incrementally; J1/J2/det rebuilt) and
                advances the state at the end of each step.

                Dispatch is Python-static at the ``_projection_n`` entry point.
                """
                (
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    RT,
                    ecp_RT,
                    A_old_inv,
                    kinetic_state,
                    kinetic_mesh_state,
                    ecp_mesh_state,
                    pot_state,
                    _,
                    _,
                    interval_refreshes,
                ) = carry

                ke_up, ke_dn = _kinetic_energy_from_streaming_state(kinetic_state)

                (
                    w_L_new,
                    r_up_new,
                    r_dn_new,
                    RT_new,
                    ecp_RT_new,
                    A_new_inv,
                    diag_sum_H,
                    nondiag_sum_H,
                    moved_spin_is_up,
                    moved_index,
                    pot_state_new,
                ) = _body_step_core(
                    i,
                    w_L,
                    r_up_carts,
                    r_dn_carts,
                    A_old_inv,
                    ke_up,
                    ke_dn,
                    j3_state=kinetic_state.j3_state,
                    det_ratio_state=kinetic_state.det_state,
                    kinetic_mesh_state=kinetic_mesh_state,
                    ecp_mesh_state=ecp_mesh_state,
                    pot_state=pot_state,
                )

                def _advance_kinetic(A_inv: jax.Array) -> Kinetic_streaming_state:
                    return _advance_kinetic_energy_all_elements_streaming_state(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        state=kinetic_state,
                        moved_spin_is_up=moved_spin_is_up,
                        moved_index=moved_index,
                        r_up_carts_new=r_up_new,
                        r_dn_carts_new=r_dn_new,
                        A_new_inv=A_inv,
                    )

                kinetic_state_new = _advance_kinetic(A_new_inv)
                # Per-hop screen of the carried inverse against the
                # post-hop det caches (walker-global cond over the vmap axis).
                A_new_inv, kinetic_state_new, refreshed = _interval_inverse_screen(
                    hamiltonian_data.wavefunction_data.geminal_data,
                    kinetic_state_new.det_state,
                    A_new_inv,
                    kinetic_state_new,
                    _advance_kinetic,
                    axis_name=_GFMC_WALKER_AXIS,
                )
                kinetic_mesh_state_new = _advance_discretized_kinetic_mesh_streaming_state(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    state=kinetic_mesh_state,
                    moved_spin_is_up=moved_spin_is_up,
                    moved_index=moved_index,
                    r_up_carts_old=r_up_carts,
                    r_dn_carts_old=r_dn_carts,
                    r_up_carts_new=r_up_new,
                    r_dn_carts_new=r_dn_new,
                )
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    ecp_mesh_state_new = _advance_ecp_candidate_mesh_streaming_state(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        state=ecp_mesh_state,
                        moved_spin_is_up=moved_spin_is_up,
                        moved_index=moved_index,
                        r_up_carts_old=r_up_carts,
                        r_dn_carts_old=r_dn_carts,
                        r_up_carts_new=r_up_new,
                        r_dn_carts_new=r_dn_new,
                        RT=interval_ecp_R.T,
                    )
                else:
                    ecp_mesh_state_new = None

                return (
                    w_L_new,
                    r_up_new,
                    r_dn_new,
                    RT_new,
                    ecp_RT_new,
                    A_new_inv,
                    kinetic_state_new,
                    kinetic_mesh_state_new,
                    ecp_mesh_state_new,
                    pot_state_new,
                    diag_sum_H,
                    nondiag_sum_H,
                    interval_refreshes + refreshed.astype(jnp.int32),
                )

            def _split_body(current_key: jax.Array, _: None) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                current_key, rot_key = jax.random.split(current_key)
                current_key, move_key = jax.random.split(current_key)
                return current_key, (rot_key, move_key)

            latest_jax_PRNG_key, (rotation_keys, move_keys) = lax.scan(
                _split_body,
                init_jax_PRNG_key,
                xs=None,
                length=num_mcmc_per_measurement,
            )
            kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rotation_keys[0])
            interval_R = _sample_lrdmc_rotation_matrix(kinetic_key, random_discretized_mesh)
            interval_ecp_R = _sample_lrdmc_rotation_matrix(ecp_key, random_discretized_mesh)

            # The determinant streaming path (consuming ``kinetic_state.det_state``
            # in the LRDMC mesh kernels and the GFMC inv-update) is always used.
            # Mirrors the GFMC_t dispatch policy.
            use_streaming = True

            init_pot_state = _gfmc_build_interval_pot_state(hamiltonian_data, init_r_up_carts, init_r_dn_carts, alat)
            if use_streaming:
                init_kinetic_state = _init_kinetic_energy_all_elements_streaming_state(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=init_r_up_carts,
                    r_dn_carts=init_r_dn_carts,
                    geminal_inverse=init_A_old_inv,
                )
                init_kinetic_mesh_state = _init_discretized_kinetic_mesh_streaming_state(
                    alat=alat,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=init_r_up_carts,
                    r_dn_carts=init_r_dn_carts,
                    RT=interval_R.T,
                )
                init_ecp_mesh_state = (
                    _init_ecp_candidate_mesh_streaming_state(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=init_r_up_carts,
                        r_dn_carts=init_r_dn_carts,
                        RT=interval_ecp_R.T,
                    )
                    if hamiltonian_data.coulomb_potential_data.ecp_flag
                    else None
                )
                (
                    latest_w_L,
                    latest_r_up_carts,
                    latest_r_dn_carts,
                    latest_RT,
                    latest_ecp_RT,
                    latest_A_old_inv,
                    _latest_kinetic_state,
                    _latest_kinetic_mesh_state,
                    _latest_ecp_mesh_state,
                    _latest_pot_state,
                    latest_diagonal_sum_hamiltonian,
                    latest_non_diagonal_sum_hamiltonian,
                    latest_interval_refreshes,
                ) = jax.lax.fori_loop(
                    0,
                    num_mcmc_per_measurement,
                    _body_fun_n_streaming,
                    (
                        init_w_L,
                        init_r_up_carts,
                        init_r_dn_carts,
                        jnp.eye(3, dtype=jnp.float64),
                        jnp.eye(3, dtype=jnp.float64),
                        init_A_old_inv,
                        init_kinetic_state,
                        init_kinetic_mesh_state,
                        init_ecp_mesh_state,
                        init_pot_state,
                        # For complex Bloch, diagonal_sum_hamiltonian
                        # / non_diagonal_sum_hamiltonian carry-out are complex
                        # on the twist k != 0 path. Use init_A_old_inv.dtype so
                        # the fori_loop input carry dtype matches the body
                        # output. Bit-identical at Gamma where A_old_inv is
                        # real.
                        jnp.asarray(0.0, dtype=init_A_old_inv.dtype),
                        jnp.asarray(0.0, dtype=init_A_old_inv.dtype),
                        jnp.zeros((), dtype=jnp.int32),
                    ),
                )
            else:
                latest_interval_refreshes = jnp.zeros((), dtype=jnp.int32)
                (
                    latest_w_L,
                    latest_r_up_carts,
                    latest_r_dn_carts,
                    latest_RT,
                    latest_ecp_RT,
                    latest_A_old_inv,
                    _latest_pot_state,
                    latest_diagonal_sum_hamiltonian,
                    latest_non_diagonal_sum_hamiltonian,
                ) = jax.lax.fori_loop(
                    0,
                    num_mcmc_per_measurement,
                    _body_fun_n,
                    (
                        init_w_L,
                        init_r_up_carts,
                        init_r_dn_carts,
                        jnp.eye(3, dtype=jnp.float64),
                        jnp.eye(3, dtype=jnp.float64),
                        init_A_old_inv,
                        init_pot_state,
                        # Complex Bloch follows the same fixed-phase real projection as
                        # the streaming variant above.
                        jnp.asarray(0.0, dtype=init_A_old_inv.dtype),
                        jnp.asarray(0.0, dtype=init_A_old_inv.dtype),
                    ),
                )

            return (
                latest_w_L,
                latest_r_up_carts,
                latest_r_dn_carts,
                latest_A_old_inv,
                latest_jax_PRNG_key,
                latest_RT,
                latest_ecp_RT,
                latest_diagonal_sum_hamiltonian,
                latest_non_diagonal_sum_hamiltonian,
                # Incremental-potential liveness counter (applied one-electron
                # updates of the finished interval; 0 when the state is inactive).
                _latest_pot_state[6],
                _latest_kinetic_state,
                _latest_kinetic_mesh_state,
                # In-interval refresh witness (hops of this walker
                # whose carried inverse was rebuilt by the per-hop screen).
                latest_interval_refreshes,
            )

        @partial(jit, static_argnums=(5, 7))
        def _compute_V_elements_n(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            kinetic_RT: jnpt.ArrayLike,
            ecp_RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
            use_fast_update: bool = True,
            A_old_inv: jnpt.ArrayLike | None = None,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Compute V elements.

            ``A_old_inv``: the running Sherman-Morrison geminal
            inverse maintained by the projection loop for exactly
            ``(r_up_carts, r_dn_carts)``. When provided (the per-step
            observable path), the fresh SVD-pinv seed below is skipped so this
            graph lowers zero SVDs. When ``None`` (trace-time Python default;
            e.g. the autograd ``_compute_local_energy_n`` fallback), the
            behaviour is unchanged (fresh thresholded SVD pseudoinverse).
            """
            # coulomb regularization
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            if A_old_inv is not None:
                # Reuse the running Sherman-Morrison inverse for the continuum
                # kinetic energy too (same closed form; the internal SVD-pinv
                # of ``compute_kinetic_energy_all_elements`` does not lower).
                # Identical to the per-substep projection-loop usage.
                diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                    compute_kinetic_energy_all_elements_fast_update(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        geminal_inverse=A_old_inv,
                    )
                )
            else:
                diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                    compute_kinetic_energy_all_elements(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

            if use_fast_update:
                if A_old_inv is None:
                    # precompute geminal inverse for fast updates: the shared seed
                    # pseudo-inverse (row/column-equilibrated SVD, ``rcond_svd`` cutoff,
                    # conjugate factors for complex Bloch geminals).
                    geminal = compute_geminal_all_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                    A_old_inv = _geminal_pinv_from_matrix(geminal)
                # Otherwise use the running Sherman-Morrison inverse threaded from the
                # projection loop is used verbatim (no fresh SVD lowers).

                # compute discretized kinetic energy and mesh (with a random rotation)
                # Measurement-point e_L: this is the OBSERVABLE evaluation, not
                # a projection sub-step, and it currently takes no mesh state.
                with route_site("observable"):
                    _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy_fast_update(
                        alat=alat,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        A_old_inv=A_old_inv,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=kinetic_RT,
                    )
            else:
                A_old_inv = None
                # compute discretized kinetic energy and mesh (with a random rotation)
                _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
                    alat=alat,
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    RT=kinetic_RT,
                )
            # spin-filp
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            # regularizations
            elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
            sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
            non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1)
            sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
            non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                non_diagonal_kinetic_part_elements, [len(r_up_carts)]
            )

            # compute diagonal elements, el-el
            diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_coulomb_potential_ei_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_coulomb_potential_ei_discretized_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_up = (
                diagonal_bare_coulomb_part_el_ion_elements_up
                + diagonal_kinetic_continuum_elements_up
                - non_diagonal_kinetic_part_elements_up
            )
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
            else:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
            diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
            )
            diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_dn = (
                diagonal_bare_coulomb_part_el_ion_elements_dn
                + diagonal_kinetic_continuum_elements_dn
                - non_diagonal_kinetic_part_elements_dn
            )
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
            else:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
            diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
            )
            diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
            )

            # final bare coulomb part
            discretized_diagonal_bare_coulomb_part = (
                diagonal_bare_coulomb_part_el_el
                + diagonal_bare_coulomb_part_ion_ion
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
            )

            # with ECP
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                # ecp local
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    if use_fast_update:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                A_old_inv=A_old_inv,
                                RT=ecp_RT,
                            )
                        )
                    else:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                RT=ecp_RT,
                            )
                        )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                elif non_local_move == "dltmove":
                    if use_fast_update:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors_fast_update(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                A_old_inv=A_old_inv,
                                RT=ecp_RT,
                            )
                        )
                    else:
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                RT=ecp_RT,
                            )
                        )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                    if use_fast_update:
                        Jastrow_ratio = _compute_ratio_Jastrow_part_rank1_update(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            old_r_up_carts=r_up_carts,
                            old_r_dn_carts=r_dn_carts,
                            new_r_up_carts_arr=mesh_non_local_ecp_part_r_up_carts,
                            new_r_dn_carts_arr=mesh_non_local_ecp_part_r_dn_carts,
                        )
                    else:
                        Jastrow_ref = compute_Jastrow_part_ratio_zone(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                        Jastrow_on_mesh = vmap(compute_Jastrow_part_ratio_zone, in_axes=(None, 0, 0))(
                            hamiltonian_data.wavefunction_data.jastrow_data,
                            mesh_non_local_ecp_part_r_up_carts,
                            mesh_non_local_ecp_part_r_dn_carts,
                        )
                        Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                else:
                    raise NotImplementedError

                V_diag = (
                    diagonal_kinetic_part
                    + discretized_diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
                    + diagonal_kinetic_part_SP
                    + diagonal_ecp_part_SP
                )

                V_nondiag = non_diagonal_sum_hamiltonian

                # Measurement local energy (novar=0 split, cf. TurboRVB
                # enertrue with novar=0): continuum e_L from the per-electron
                # continuum kinetic + BARE el-ion potential, no lattice hop
                # terms, no sign-flip max clip. Equals V_diag + V_nondiag on
                # non-clipped configurations by the zero-variance identity.
                e_L_meas = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                    + diagonal_ecp_local_part
                    + diagonal_ecp_part_SP
                    + non_diagonal_sum_hamiltonian_ecp
                )

            # with all electrons
            else:
                non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic

                V_diag = diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP

                V_nondiag = non_diagonal_sum_hamiltonian

                # Measurement local energy (novar=0 split) -- see ECP branch.
                e_L_meas = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                )

            return (V_diag, V_nondiag, e_L_meas)

        @partial(jit, static_argnums=(4, 6))
        def _compute_local_energy_n(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
            use_fast_update: bool = True,
        ) -> jax.Array:
            _V_diag, _V_nondiag, e_L_meas = _compute_V_elements_n(
                hamiltonian_data=hamiltonian_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                kinetic_RT=RT,
                ecp_RT=RT,
                non_local_move=non_local_move,
                alat=alat,
                use_fast_update=use_fast_update,
            )
            # novar=0 split: forces/stress differentiate the MEASUREMENT
            # (continuum) local energy -- mixed-average estimator, no dw/dR.
            return e_L_meas

        wavefunction_for_position_grads, hamiltonian_for_position_grads = self._prepare_position_grad_objects()

        # Pre-compile all vmap callables once so the JAX jit cache is always hit
        # on subsequent calls (avoids per-step re-trace / re-compile overhead).
        # static_argnums mirrors the @partial(jit, static_argnums=...) on the inner
        # functions so that Python-level conditionals are resolved at compile time.
        _jit_vmap_projection_n = jit(
            vmap(
                _projection_n,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None),
                axis_name=_GFMC_WALKER_AXIS,
            ),
            static_argnums=(6, 7, 8),  # num_mcmc_per_measurement, random_discretized_mesh, non_local_move
        )
        _jit_vmap_V_elements_n = jit(
            # The trailing axis-0 argument is the per-walker running SM inverse.
            # (``self.__latest_A_old_inv``), so the per-step e_L recompute lowers
            # zero fresh SVDs (the inline pinv seed above is skipped).
            vmap(_compute_V_elements_n, in_axes=(None, 0, 0, 0, 0, None, None, None, 0)),
            static_argnums=(5, 7),  # non_local_move, use_fast_update
        )
        # In-trace position-gradient mask. As in the t-move route, an eager
        # mask is ineffective because stop_gradient only acts on traced values.
        _POSITION_GRAD_MASK_KWARGS_LRDMC_N = dict(
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=False,
        )

        def _e_L_n_with_position_mask(
            active_positions: ActivePositions,
            base_h: Hamiltonian_data,
            r_up: jax.Array,
            r_dn: jax.Array,
            RT: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            use_fast_update: bool,
        ) -> jax.Array:
            h = _inject_active_positions(base_h, active_positions)
            masked_wf = h.wavefunction_data.with_param_grad_mask(**_POSITION_GRAD_MASK_KWARGS_LRDMC_N)
            return _compute_local_energy_n(
                h.replace(wavefunction_data=masked_wf), r_up, r_dn, RT, non_local_move, alat, use_fast_update
            )

        # Project the complex Bloch local energy to its physical real component
        # before differentiating the force.
        def _real_e_L_n_with_position_mask(*args: Any, **kwargs: Any) -> jax.Array:
            return jnp.real(_e_L_n_with_position_mask(*args, **kwargs))

        # Fallback HF force (autograd twin); the default routes through the analytic
        # ``jqmc.derivatives.gfmc.grad_e_L_n_wrt_positions`` kernel (rebound below). This ``jax.grad``
        # of the LRDMC lattice-discretised ``_compute_local_energy_n`` (alat/fixed-node,
        # ``w_L``-consistent) is the oracle the analytic kernel is verified against.
        _jit_vmap_grad_e_L_n_autograd = jit(
            vmap(
                grad(_real_e_L_n_with_position_mask, argnums=(0, 2, 3)),
                in_axes=(None, None, 0, 0, 0, None, None, None),
            ),
            static_argnums=(5, 7),  # non_local_move, use_fast_update
        )

        # Uniform signature with the analytic closure: accept-and-ignore the trailing ``geminal_inv``
        # (the autograd fallback recomputes the inverse through its own graph).
        def _jit_vmap_grad_e_L_n(
            active_positions: ActivePositions,
            base_h: Hamiltonian_data,
            r_up_b: jax.Array,
            r_dn_b: jax.Array,
            RT_b: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            use_fast_update: bool,
            geminal_inv: jax.Array | None = None,
        ) -> GFMCPositionGradient:
            return _jit_vmap_grad_e_L_n_autograd(
                active_positions, base_h, r_up_b, r_dn_b, RT_b, non_local_move, alat, use_fast_update
            )

        # dln|Psi|/dR (Pulay) and
        # dln|Psi|/dr drift route through the ``jqmc.derivatives`` interface
        # (``analytic=False``). ln|Psi| is potential-free -> LRDMC-independent
        # and differentiates the same ``evaluate_ln_wavefunction`` as MCMC, so it
        # maps cleanly. Reuse MCMC's module-level LRU-cached ``jit(vmap)`` wrapper;
        # returns ``(grad_ln_Psi_R, grad_ln_Psi_r_up, grad_ln_Psi_r_dn)`` (the "R"
        # accumulate is ``accumulate_position_grad``).
        # GFMC production ln|Psi| (drift/Pulay/stress) builds the same
        # ``bridge.batched_grad_ln_Psi`` closure MCMC's run() does;
        # ``bridge._build`` is ``@lru_cache(None)``, so the compiled inner jit is shared.
        # ``analytic`` is driven by this module's ``_ANALYTIC_GRAD_ENABLED``,
        # the same flag MCMC uses.
        _grad_ln_psi_rR = bridge.batched_grad_ln_Psi(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
        _active_positions_n = _extract_active_positions_from_ham(self._hamiltonian_data)

        # LRDMC strained-energy kernels for Hellmann-Feynman and Pulay stress mirror
        # of the GFMC_t variant (see ``GFMC_t.run``). ``F = I + eps`` is applied
        # to every Structure_data leaf in the Hamiltonian (via ``_apply_strain``)
        # and to the electron coords / ECP mesh / wavefunction; ``jax.jacrev``
        # at ``eps = 0`` yields the per-walker sigma_HF / sigma_PP.
        # ``non_local_move`` and ``alat`` are forwarded unchanged --- the LRDMC
        # lattice-projector discretisation operates in the strained frame just
        # like in the un-strained one.
        def _compute_local_energy_n_strained(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jax.Array,
            r_dn_carts: jax.Array,
            RT: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            strain: jax.Array,
        ) -> jax.Array:
            F = jnp.eye(3, dtype=jnp.float64) + strain
            new_h = hamiltonian_data._apply_strain(F)
            new_r_up = r_up_carts @ F.T
            new_r_dn = r_dn_carts @ F.T
            # Rotation not strained: fixed ECP quadrature directions.
            new_RT = RT
            # use_fast_update=False (eager recompute): the fast-update kinetic
            # path builds the geminal inverse via an SVD pseudo-inverse
            # (``jnp.where(s > eps_rcond * s[0], 1/s, 0)``), whose strain
            # derivative is NaN (dropped / near-degenerate singular values make
            # the masked 1/s branch non-differentiable). The eager path has a
            # clean strain gradient and is value-identical. Matches the GFMC_n
            # force kernel (``_jit_vmap_grad_e_L_n`` is called with
            # use_fast_update=False) and the eager _GFMC_n_debug oracle. The
            # strain derivative is a per-measurement observable, not the hot
            # move path, so the fast-update speedup is irrelevant here.
            return _compute_local_energy_n(new_h, new_r_up, new_r_dn, new_RT, non_local_move, alat, use_fast_update=False)

        # Complex Bloch uses the same physical-real projection as the force
        # gradient; differentiating Re(e_L) yields the physical stress and is
        # a no-op at Gamma.
        def _real_compute_local_energy_n_strained(*args: Any, **kwargs: Any) -> jax.Array:
            return jnp.real(_compute_local_energy_n_strained(*args, **kwargs))

        # Fallback HF stress (autograd twin); the default routes through the analytic
        # ``jqmc.derivatives.gfmc.grad_e_L_n_wrt_strain`` kernel (rebound below). This jacrev of the
        # LRDMC lattice-discretised ``_compute_local_energy_n`` (alat/fixed-node, w_L-consistent)
        # is the oracle the analytic stress kernel is verified against.
        _jit_vmap_local_stress_HF_n_autograd = jit(
            vmap(
                jax.jacrev(_real_compute_local_energy_n_strained, argnums=6),
                in_axes=(None, 0, 0, 0, None, None, None),
            ),
            static_argnums=(4,),  # non_local_move
        )

        # Uniform signature with the analytic stress closure: accept-and-ignore trailing ``geminal_inv``.
        def _jit_vmap_local_stress_HF_n(
            base_h: Hamiltonian_data,
            r_up_b: jax.Array,
            r_dn_b: jax.Array,
            RT_b: jax.Array,
            non_local_move: NonLocalMove,
            alat: float,
            strain: jax.Array,
            geminal_inv: jax.Array | None = None,
        ) -> jax.Array:
            return _jit_vmap_local_stress_HF_n_autograd(base_h, r_up_b, r_dn_b, RT_b, non_local_move, alat, strain)

        # The Pulay stress
        # dln|Psi|/deps routes through the ``jqmc.derivatives`` interface
        # (``analytic=False`` reuses MCMC's ``_local_log_psi_strained`` at strain 0
        # -- identical to the removed local ``_local_log_psi_strained_n``).
        _grad_ln_psi_eps = bridge.batched_grad_ln_Psi(wrt="eps", with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)

        # Route HF force/stress to the verified analytic continuum-measurement kernels
        # (``jqmc.derivatives.gfmc``) when enabled -- same reconciliation as GFMC_t.run above
        # (``e_L^n == e_L^t``, so the ``_n`` kernels delegate to the ``_t`` foundation).
        # Complex-twist DTN is supported; ``_assert_scope`` rejects nothing.
        if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
            from .derivatives import gfmc as _gfmc_deriv

            _nlm_n = self._non_local_move  # static: threads tmove/dltmove into the analytic kernels
            # The base-position geminal inverse is threaded as a traced vmapped
            # arg so the analytic measurement derivative reuses it (2 SVDs/step -> 0). GFMC_n's running SM
            # carry ``self.__latest_A_old_inv`` IS ``G^-1`` at the observable positions (verified
            # fp-exact today, <=2e-14), so the driver passes it directly (no fresh recompute).
            _vmap_analytic_force_n = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_n_wrt_positions, _nlm_n)
            _vmap_analytic_stress_n = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_n_wrt_strain, _nlm_n)
            # One fused program shares the continuous measurement primal between
            # force and stress when both are consumed each step; the force
            # closure stashes stress for the same-step consumer.
            _vmap_analytic_both_n = _walker_gfmc_kernel(_gfmc_deriv.grad_e_L_n_wrt_positions_and_strain, _nlm_n)
            _fused_stress_stash_n: dict = {}
            # comput_stress=False -> force-only variant.
            _fuse_force_stress_n = bool(
                self._comput_position_deriv and self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag
            )

            def _jit_vmap_grad_e_L_n(
                active_positions: ActivePositions,
                base_h: Hamiltonian_data,
                r_up_b: jax.Array,
                r_dn_b: jax.Array,
                RT_b: jax.Array,
                non_local_move: NonLocalMove,
                alat: float,
                use_fast_update: bool,
                geminal_inv: jax.Array | None = None,
            ) -> GFMCPositionGradient:
                if _fuse_force_stress_n:
                    dR, dru, drd, sHF = _vmap_analytic_both_n(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)
                    _fused_stress_stash_n["stress"] = sHF
                    return {"positions": dR}, dru, drd
                dR, dru, drd = _vmap_analytic_force_n(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)
                return {"positions": dR}, dru, drd

            def _jit_vmap_local_stress_HF_n(
                base_h: Hamiltonian_data,
                r_up_b: jax.Array,
                r_dn_b: jax.Array,
                RT_b: jax.Array,
                non_local_move: NonLocalMove,
                alat: float,
                strain: jax.Array,
                geminal_inv: jax.Array | None = None,
            ) -> jax.Array:
                if _fuse_force_stress_n and "stress" in _fused_stress_stash_n:
                    return _fused_stress_stash_n.pop("stress")
                return _vmap_analytic_stress_n(base_h, r_up_b, r_dn_b, RT_b, alat, geminal_inv)

        if self._use_swct:
            _jit_vmap_swct_omega_n = jit(vmap(evaluate_swct_omega, in_axes=(None, 0)))
            # Fused twin (see swct.evaluate_swct_omega_dot / the GFMC_t site).
            _jit_vmap_swct_omega_dot_n = jit(evaluate_swct_omega_dot_batched)
            _jit_vmap_swct_domega_n = jit(vmap(evaluate_swct_domega, in_axes=(None, 0)))

        # Every jitted callable above is run()-local, so a second run() call
        # would recreate and recompile the whole set. Cache them on the
        # instance (keyed on the hamiltonian object identity -- swapping the
        # hamiltonian invalidates the cache, its leaves are closure constants)
        # and skip the compile warmups on a cache hit.
        _driver_cache = getattr(self, "_projection_driver_cache", None)
        _drivers_cached = _driver_cache is not None and _driver_cache["hamiltonian"] is self._hamiltonian_data
        if _drivers_cached:
            _jit_vmap_A_inv_n = _driver_cache["A_inv"]
            _jit_vmap_projection_n = _driver_cache["projection"]
            _jit_vmap_V_elements_n = _driver_cache["V_elements"]
            _jit_vmap_grad_e_L_n = _driver_cache["grad_e_L"]
            _grad_ln_psi_rR = _driver_cache["grad_ln_psi_rR"]
            _grad_ln_psi_eps = _driver_cache["grad_ln_psi_eps"]
            _jit_vmap_local_stress_HF_n = _driver_cache["local_stress_HF"]
            if self._use_swct:
                _jit_vmap_swct_omega_n = _driver_cache["swct_omega"]
                _jit_vmap_swct_omega_dot_n = _driver_cache["swct_omega_dot"]
                _jit_vmap_swct_domega_n = _driver_cache["swct_domega"]
        else:
            _new_cache = {
                "hamiltonian": self._hamiltonian_data,
                "A_inv": _jit_vmap_A_inv_n,
                "projection": _jit_vmap_projection_n,
                "V_elements": _jit_vmap_V_elements_n,
                "grad_e_L": _jit_vmap_grad_e_L_n,
                "grad_ln_psi_rR": _grad_ln_psi_rR,
                "grad_ln_psi_eps": _grad_ln_psi_eps,
                "local_stress_HF": _jit_vmap_local_stress_HF_n,
            }
            if self._use_swct:
                _new_cache["swct_omega"] = _jit_vmap_swct_omega_n
                _new_cache["swct_omega_dot"] = _jit_vmap_swct_omega_dot_n
                _new_cache["swct_domega"] = _jit_vmap_swct_domega_n
            self._projection_driver_cache = _new_cache

        # projection compilation.
        start_init = time.perf_counter()
        w_L_list = jnp.array([1.0 for _ in range(self._num_walkers)], dtype=jnp.float64)
        # The native chunk never dispatches the JAX projection or V-elements graphs,
        # so skip their pure compile-time warmups. The chunk gate
        # excludes comput_position_deriv, so the force warmups below cannot need
        # the warmup ``RTs`` on the native path. On a driver-cache hit the
        # executables are already warm, so skip every warmup below.
        if not _drivers_cached and not _rs_gfmc_chunk:
            logger.info("Start compilation of the GFMC projection funciton.")
            logger.info("  Compilation is in progress...")
            (
                _,
                _,
                _,
                _,
                _,
                kinetic_RTs,
                RTs,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = _jit_vmap_projection_n(
                w_L_list,
                self._latest_r_up_carts,
                self._latest_r_dn_carts,
                self.__latest_A_old_inv,
                self._jax_PRNG_key_list,
                self._E_scf,
                self._num_mcmc_per_measurement,
                self._random_discretized_mesh,
                self._non_local_move,
                self._alat,
                self._hamiltonian_data,
            )

            # compile the e_L recomputation path for debug parity
            _, _, _ = _jit_vmap_V_elements_n(
                self._hamiltonian_data,
                self._latest_r_up_carts,
                self._latest_r_dn_carts,
                kinetic_RTs,
                RTs,
                self._non_local_move,
                self._alat,
                True,
                # Running Sherman-Morrison inverse; no fresh SVD-pinv in this graph.
                self.__latest_A_old_inv,
            )

        if not _drivers_cached and (self._comput_position_deriv or self._comput_stress):
            # On the native force fold-in path the JAX HF/Pulay/stress grad
            # kernels are never dispatched (the LRDMC force is folded into
            # ``run_gfmc_projection_n``), so skip their compile warmups -- and the
            # warmup ``RTs`` (JAX-projection only) is undefined there. SWCT is
            # native too, so its JAX warmups belong only to the XLA path.
            if not _rs_gfmc_chunk and self._comput_position_deriv:
                _, _, _ = _jit_vmap_grad_e_L_n(
                    _active_positions_n,
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    RTs,
                    self._non_local_move,
                    self._alat,
                    False,
                    # GFMC_n's running SM inverse is G^-1 at the observable positions.
                    # -- share it with the analytic HF force so the kernel lowers with no fresh SVD-pinv.
                    self.__latest_A_old_inv,
                )
                _, _, _ = _grad_ln_psi_rR(
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    # Reuse the running SM inverse paired with the current
                    # walker positions by the projection loop) -- no fresh SVD-pinv per step.
                    self.__latest_A_old_inv,
                )
            if self._comput_position_deriv and self._use_swct and not _rs_gfmc_chunk:
                # Warm up SWCT vmap callables so they are not JIT-compiled
                # inside the main MCMC loop.
                _ = _jit_vmap_swct_omega_n(self._hamiltonian_data.structure_data, self._latest_r_up_carts)
                _ = _jit_vmap_swct_omega_n(self._hamiltonian_data.structure_data, self._latest_r_dn_carts)
                _ = _jit_vmap_swct_domega_n(self._hamiltonian_data.structure_data, self._latest_r_up_carts)
                _ = _jit_vmap_swct_domega_n(self._hamiltonian_data.structure_data, self._latest_r_dn_carts)
            # Stress kernels fire only on the PBC + position-derivative
            # path (see the branching loop below). Lower them here so the first
            # branching step that emits stress does not stall on a fresh JIT
            # trace (mirrors the GFMC_t stress warmup).
            if not _rs_gfmc_chunk and self._hamiltonian_data.structure_data.pbc_flag and self._comput_stress:
                _zero_eps_warmup = jnp.zeros((3, 3), dtype=jnp.float64)
                _ = _jit_vmap_local_stress_HF_n(
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    RTs,
                    self._non_local_move,
                    self._alat,
                    _zero_eps_warmup,
                    self.__latest_A_old_inv,
                )
                _ = _grad_ln_psi_eps(
                    hamiltonian_for_position_grads,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    # Reuse the running SM inverse paired with the current
                    # walker positions by the projection loop) -- no fresh SVD-pinv per step.
                    self.__latest_A_old_inv,
                )
        self._mpi_comm.Barrier()
        end_init = time.perf_counter()
        timer_projection_init += end_init - start_init
        logger.info("End compilation of the GFMC projection funciton.")
        logger.info(f"Elapsed Time = {timer_projection_init:.2f} sec.")
        logger.info("")

        _log_memory_usage("gfmc_n_post_warmup", comm=self._mpi_comm)

        # MAIN MCMC loop from here !!!
        logger.info("Start GFMC")
        num_mcmc_done = 0

        # -- Extend stored arrays with zero-padding for new steps (rank 0 only) --
        if self._mpi_rank == 0:
            n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
            n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
            n_atoms = self._hamiltonian_data.structure_data.natom

            # dtype for stored numpy arrays
            dtype_np = np.float64

            # Preserve stored_e_L's complex Bloch dtype (complex128
            # on twist k != 0). Bit-identical at Gamma where dtype is float64.
            self._stored_e_L = np.concatenate([self._stored_e_L, np.zeros((num_mcmc_steps, 1), dtype=self._stored_e_L.dtype)])
            self._stored_e_L2 = np.concatenate([self._stored_e_L2, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            self._stored_w_L = np.concatenate([self._stored_w_L, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            if self._mpc:
                self._stored_mpc_dV = np.concatenate([self._stored_mpc_dV, np.zeros((num_mcmc_steps, 1), dtype=dtype_np)])
            if self._comput_position_deriv:
                self._stored_force_HF = np.concatenate(
                    [self._stored_force_HF, np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=dtype_np)]
                )
                self._stored_force_PP = np.concatenate(
                    [self._stored_force_PP, np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=dtype_np)]
                )
                self._stored_E_L_force_PP = np.concatenate(
                    [
                        self._stored_E_L_force_PP,
                        np.zeros((num_mcmc_steps, 1, n_atoms, 3), dtype=self._stored_E_L_force_PP.dtype),
                    ]
                )
            if self._comput_stress:
                self._stored_stress_HF = np.concatenate(
                    [self._stored_stress_HF, np.zeros((num_mcmc_steps, 1, 3, 3), dtype=dtype_np)]
                )
                self._stored_stress_PP = np.concatenate(
                    [self._stored_stress_PP, np.zeros((num_mcmc_steps, 1, 3, 3), dtype=dtype_np)]
                )
                self._stored_E_L_stress_PP = np.concatenate(
                    [
                        self._stored_E_L_stress_PP,
                        np.zeros((num_mcmc_steps, 1, 3, 3), dtype=self._stored_E_L_stress_PP.dtype),
                    ]
                )

        mcmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))
        progress_interval = max(1, int(np.ceil(num_mcmc_steps / 20)))
        # Polling a TOML file and broadcasting it after every millisecond-scale
        # branch is disproportionately expensive.  A maximum 30-branch polling
        # cadence retains prompt external cancellation (well below the workflow
        # scheduler's 30 s poll) without putting filesystem I/O in the hot loop.
        control_poll_interval = min(max(num_mcmc_steps, 1), 30)

        gfmc_loop_start = time.perf_counter()
        for i_mcmc_step in range(num_mcmc_steps):
            if i_mcmc_step % progress_interval == 0:
                progress = (i_mcmc_step + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
                gfmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: GFMC step = {i_mcmc_step + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %. Elapsed time = {(gfmc_total_current - gfmc_loop_start):.1f} sec."
                )

            # Always set the initial weight list to 1.0.  Keep the native
            # projection path in NumPy between branches; converting these
            # arrays to JAX and straight back on every branch only dispatches
            # device copies and is not part of the GFMC algorithm.
            if _rs_gfmc_chunk:
                w_L_list = np.ones(self._num_walkers, dtype=np.float64)
            else:
                w_L_list = jnp.ones(self._num_walkers, dtype=jnp.float64)

            start_projection = time.perf_counter()

            if _rs_gfmc_chunk:
                # ===== NATIVE GFMC_n PROJECTION INTERVAL =====
                # One ``run_gfmc_projection_n`` call per walker: the fixed
                # ``num_mcmc_per_measurement`` sub-steps (discrete projector w_L +
                # FN hop) AND the final-config e_L / V_diag measurement (last
                # sub-step's mesh rotation -- exactly what the JAX path's
                # ``_jit_vmap_V_elements_n`` block below computes) fused in Rust,
                # driven by the SAME parity-pinned ``gfmc_lrdmc_{ae,ecp}`` kernels.
                # Mesh rotations are generated natively.
                # One batched call fans the fixed-length interval over
                # ALL walkers on the rayon global pool (the Python ``for _w`` walker loop ->
                # Rust ``for_each_walker``). Per-walker semantics/results are identical to a
                # per-walker ``run_gfmc_projection_n`` loop
                # (bit-exact at any thread count -- per-walker threefry keys
                # make the walk thread-order independent). The empty 4-D rotation stack
                # selects native mesh generation.
                # The complex twin folds the same force/stress blocks under the same keys
                # so the unpack below is shared; ``final_key``
                # stays byte-equal to the JAX projection's.
                _rs_keys = np.asarray(self._jax_PRNG_key_list, dtype=np.uint32)
                _ru_all = np.asarray(self._latest_r_up_carts, dtype=np.float64)
                _rd_all = np.asarray(self._latest_r_dn_carts, dtype=np.float64)
                _rts_batch = _RS_EMPTY_RTS_BATCH
                if _rs_complex:
                    _res = _rs_obs_ctx.run_gfmc_projection_n_complex_batch(
                        _ru_all,
                        _rd_all,
                        self._alat,
                        float(self._E_scf),
                        int(self._num_mcmc_per_measurement),
                        _rs_keys,
                        _rts_batch,
                        _rs_ecp_w,
                        _rs_ecp_gp,
                        random_mesh=bool(self._random_discretized_mesh),
                        dltmove=bool(self._non_local_move == "dltmove"),
                        want_force=_rs_want_force,
                        want_stress=_rs_want_stress,
                        # Debug/oracle twin knob (IMPL-GFMC-INCREMENTAL-TWIN-001):
                        # the SAME env switch that selects the XLA full-recompute
                        # route also selects it on the native chunk, so the twin
                        # semantics are backend-uniform.
                        debug_full_recompute=not _gfmc_incremental_pot_enabled(),
                        # Screen the projection-carried inverse handed to the
                        # Pulay force/stress fold-in (drifted -> fresh route).
                        inverse_refresh_threshold=float(_setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD),
                    )
                else:
                    _res = _rs_obs_ctx.run_gfmc_projection_n_batch(
                        _ru_all,
                        _rd_all,
                        self._alat,
                        float(self._E_scf),
                        int(self._num_mcmc_per_measurement),
                        _rs_keys,
                        _rts_batch,
                        _rs_ecp_w,
                        _rs_ecp_gp,
                        random_mesh=bool(self._random_discretized_mesh),
                        dltmove=_rs_dltmove,
                        want_force=_rs_want_force,
                        want_stress=_rs_want_stress,
                        # Debug/oracle twin knob (IMPL-GFMC-INCREMENTAL-TWIN-001):
                        # the SAME env switch that selects the XLA full-recompute
                        # route also selects it on the native chunk, so the twin
                        # semantics are backend-uniform.
                        debug_full_recompute=not _gfmc_incremental_pot_enabled(),
                        # Screen the projection-carried inverse handed to the
                        # Pulay force/stress fold-in (drifted -> fresh route).
                        inverse_refresh_threshold=float(_setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD),
                    )
                _warn_native_j3_slow_fallback_once()
                w_L_list = np.asarray(_res["w_L"], dtype=np.float64)
                _e_L_np = np.asarray(_res["e_L"])
                _V_diag_np = np.asarray(_res["V_diag"], dtype=np.float64)
                self._latest_r_up_carts = np.asarray(_res["r_up"], dtype=np.float64)
                self._latest_r_dn_carts = np.asarray(_res["r_dn"], dtype=np.float64)
                self._jax_PRNG_key_list = np.asarray(_res["final_key"], dtype=np.uint32)
                # Incremental-potential liveness witness (native chunk; zeros on
                # molecular / BITEXACT / debug full-recompute routes).
                if "n_pot_updates" in _res:
                    self._latest_gfmc_pot_updates = np.asarray(_res["n_pot_updates"], dtype=np.float64)
                if _rs_want_force:
                    _rs_native_grad_e_L_R = np.asarray(_res["hf_dR"])
                    _rs_native_grad_e_L_r_up = np.asarray(_res["hf_dr_up"])
                    _rs_native_grad_e_L_r_dn = np.asarray(_res["hf_dr_dn"])
                    _rs_native_grad_ln_Psi_dR = np.asarray(_res["pp_dR"])
                    _rs_native_grad_ln_Psi_r_up = np.asarray(_res["pp_dr_up"])
                    _rs_native_grad_ln_Psi_r_dn = np.asarray(_res["pp_dr_dn"])
                if _rs_want_stress:
                    _rs_native_stress_HF = np.asarray(_res["stress_HF"])
                    _rs_native_stress_PP = np.asarray(_res["stress_PP"])
            else:
                # projection loop
                (
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                    latest_kinetic_RTs,
                    latest_RTs,
                    V_diag_list,
                    V_nondiag_list,
                    _pot_updates_list,
                    _kinetic_state_list_out_n,
                    _kinetic_mesh_state_list_out_n,
                    _interval_refreshes_n,
                ) = _jit_vmap_projection_n(
                    w_L_list,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    self._jax_PRNG_key_list,
                    self._E_scf,
                    self._num_mcmc_per_measurement,
                    self._random_discretized_mesh,
                    self._non_local_move,
                    self._alat,
                    self._hamiltonian_data,
                )
                # In-interval refresh witness.
                self._num_interval_inverse_refreshes = getattr(self, "_num_interval_inverse_refreshes", 0) + int(
                    np.sum(np.asarray(_interval_refreshes_n))
                )
                # Incremental-potential liveness witness (per-walker applied-update
                # counts of the finished interval; zeros when inactive).
                self._latest_gfmc_pot_updates = np.asarray(_pot_updates_list)

                # sync. jax arrays computations.
                w_L_list.block_until_ready()
                self._latest_r_up_carts.block_until_ready()
                self._latest_r_dn_carts.block_until_ready()
                self.__latest_A_old_inv.block_until_ready()
                self._jax_PRNG_key_list.block_until_ready()
                # The running inverse feeds the measurement e_L, forces
                # and stresses below; refresh walkers whose carried inverse
                # drifted (O(N^2) probe against G rebuilt at these positions).
                self.__latest_A_old_inv, _n_inv_refreshed = _refresh_observable_geminal_inv(
                    self._hamiltonian_data.wavefunction_data.geminal_data,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    self.__latest_A_old_inv,
                    None if _kinetic_state_list_out_n is None else _kinetic_state_list_out_n.det_state,
                )
                self._num_inverse_refreshes = getattr(self, "_num_inverse_refreshes", 0) + _n_inv_refreshed

            end_projection = time.perf_counter()
            timer_projection_total += end_projection - start_projection
            logger.devel(f"    timer_projection_total = {(end_projection - start_projection) * 1e3:.2f} msec.")

            # evaluate observables
            start_e_L = time.perf_counter()
            if _rs_gfmc_chunk:
                # The chunk already measured e_L / V_diag at the final config with
                # the last sub-step's mesh rotation (byte-for-byte the same
                # composition as ``_jit_vmap_V_elements_n``'s V_diag + V_nondiag).
                V_diag_list = _V_diag_np
                e_L_list = _e_L_np
            else:
                V_diag_list, V_nondiag_list, e_L_meas_list = _jit_vmap_V_elements_n(
                    self._hamiltonian_data,
                    self._latest_r_up_carts,
                    self._latest_r_dn_carts,
                    latest_kinetic_RTs,
                    latest_RTs,
                    self._non_local_move,
                    self._alat,
                    True,
                    # Running SM inverse, synchronized with the positions the
                    # projection loop just returned) -- no fresh SVD-pinv per step.
                    self.__latest_A_old_inv,
                )
                # novar=0 split: the estimator numerator is the MEASUREMENT
                # (continuum) local energy; V_diag_list keeps the propagation
                # composition for the 1/(V_diag - E_scf) weighting below.
                e_L_list = e_L_meas_list
                e_L_list.block_until_ready()

            end_e_L = time.perf_counter()
            timer_e_L += end_e_L - start_e_L

            # atomic force related
            if self._comput_position_deriv:
                start = time.perf_counter()
                if _rs_gfmc_chunk:
                    # ===== NATIVE FORCE FOLD-IN (GFMC_n) =====
                    # HF-force + Pulay grads folded out of ``run_gfmc_projection_n`` on the
                    # final config with the interval's last mesh rotation (same LRDMC kernels
                    # the JAX ``_jit_vmap_grad_e_L_n`` / ``_grad_ln_psi_rR`` drive; fresh LU).
                    grad_e_L_R = _rs_native_grad_e_L_R
                    grad_e_L_r_up = _rs_native_grad_e_L_r_up
                    grad_e_L_r_dn = _rs_native_grad_e_L_r_dn
                    grad_ln_Psi_dR = _rs_native_grad_ln_Psi_dR
                    grad_ln_Psi_r_up = _rs_native_grad_ln_Psi_r_up
                    grad_ln_Psi_r_dn = _rs_native_grad_ln_Psi_r_dn
                    end = time.perf_counter()
                    timer_de_L_dR_dr += end - start
                else:
                    grad_e_L_pos_dict, grad_e_L_r_up, grad_e_L_r_dn = _jit_vmap_grad_e_L_n(
                        _active_positions_n,
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        latest_RTs,
                        self._non_local_move,
                        self._alat,
                        False,
                        # Share the running SM inverse with the analytic HF force.
                        self.__latest_A_old_inv,
                    )
                    grad_e_L_r_up.block_until_ready()
                    grad_e_L_r_dn.block_until_ready()
                    end = time.perf_counter()
                    timer_de_L_dR_dr += end - start
                    logger.devel(f"    timer_de_L_dR_dr = {(end - start) * 1e3:.2f} msec.")
                    grad_e_L_R = grad_e_L_pos_dict["positions"]

                    start = time.perf_counter()
                    # dln|Psi|/dR (Pulay) and dln|Psi|/dr drift via the
                    # derivatives interface; "R" is the per-atom accumulate
                    # (``accumulate_position_grad``).
                    grad_ln_Psi_dR, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = _grad_ln_psi_rR(
                        hamiltonian_for_position_grads,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        # Reuse the running SM inverse paired with the current
                        # walker positions by the projection loop) -- no fresh SVD-pinv per step.
                        self.__latest_A_old_inv,
                    )
                    grad_ln_Psi_r_up.block_until_ready()
                    grad_ln_Psi_r_dn.block_until_ready()
                    end = time.perf_counter()
                    timer_dln_Psi_dR_dr += end - start

                if self._use_swct and not _rs_gfmc_chunk:
                    # Only domega is precomputed; the Omega contraction is fused
                    # into the force assembly (swct.evaluate_swct_omega_dot), so
                    # the per-walker (N_a, N_e) weight matrix never exists.
                    grad_omega_dr_up = _jit_vmap_swct_domega_n(
                        self._hamiltonian_data.structure_data,
                        self._latest_r_up_carts,
                    )

                    grad_omega_dr_dn = _jit_vmap_swct_domega_n(
                        self._hamiltonian_data.structure_data,
                        self._latest_r_dn_carts,
                    )

                    grad_omega_dr_up.block_until_ready()
                    grad_omega_dr_dn.block_until_ready()
                # (use_swct=False: no omega buffers -- the force assembly below
                # skips the identically-zero SWCT terms instead of contracting
                # full-size zero (nw, n_atoms, N_e) buffers; zero-risk deletion,
                # bit-identical while avoiding an expanded temporary.)

            # Barrier before MPI operation
            start_mpi_barrier = time.perf_counter()
            if self._mpi_size > 1:
                self._mpi_comm.Barrier()
            end_mpi_barrier = time.perf_counter()
            timer_mpi_barrier += end_mpi_barrier - start_mpi_barrier

            # Branching starts
            start_collection = time.perf_counter()

            # random number for the later use
            if self._mpi_rank == 0:
                zeta = float(self._branching_rng.random())
            else:
                zeta = None
            zeta = self._mpi_comm.bcast(zeta, root=0)

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)
            # For complex Bloch fixed-phase, take Re of the diagonal
            # H expectation so the branching weight ratio ``w_L / V_diag_E``
            # stays real. Fixed-phase branching discards the imaginary phase
            # component. Bit-identical at Gamma where V_diag_list is real.
            V_diag_E_latest = np.real(np.array(V_diag_list)) - self._E_scf
            e_L2_latest = np.abs(e_L_latest) ** 2

            _raise_collective_nonfinite_observables(
                self._mpi_comm,
                driver="GFMC_n",
                stage="pre-aggregation",
                first_step=self._mcmc_counter + num_mcmc_done,
                observables={
                    "e_L": e_L_latest,
                    "e_L2": e_L2_latest,
                    "w_L": w_L_latest,
                    "V_diag_minus_E_scf": V_diag_E_latest,
                },
                r_up_carts=self._latest_r_up_carts,
                r_dn_carts=self._latest_r_dn_carts,
            )

            # MPC (physical = no): passive observable evaluated on the SAME
            # configurations that were just propagated under Ewald. It never
            # feeds back into e_L, the branching weight w_L, the forces, or any
            # gradient -- it is only accumulated for ``get_dE_mpc``.
            if self._mpc:
                _mpc_ewald_data = self._hamiltonian_data.coulomb_potential_data.ewald_data
                if _rs_mpc_observable:
                    # Native dV = V_ee^MPC - V_ee^Ewald on the SAME Ewald-propagated
                    # collect batch; one native call per
                    # walker. Projection / branching / weight are untouched.
                    import jqmc_rs

                    _mpc_ru = np.asarray(self._latest_r_up_carts)
                    _mpc_rd = np.asarray(self._latest_r_dn_carts)
                    # Rayon walker batch, bit-identical to the scalar loop.
                    mpc_dV_latest = np.asarray(
                        jqmc_rs._compute_mpc_correction_ee_batch(self._mpc_data, _mpc_ewald_data, _mpc_ru, _mpc_rd)
                    )
                else:
                    mpc_dV_latest = np.array(
                        self._jit_vmap_mpc_correction_ee(
                            self._mpc_data,
                            _mpc_ewald_data,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                        )
                    )

            if self._comput_position_deriv:
                grad_e_L_r_up_latest = np.array(grad_e_L_r_up)
                grad_e_L_r_dn_latest = np.array(grad_e_L_r_dn)
                grad_e_L_R_latest = np.array(grad_e_L_R)
                grad_ln_Psi_r_up_latest = np.array(grad_ln_Psi_r_up)
                grad_ln_Psi_r_dn_latest = np.array(grad_ln_Psi_r_dn)
                grad_ln_Psi_dR_latest = np.array(grad_ln_Psi_dR)
                if self._use_swct and not _rs_gfmc_chunk:
                    grad_omega_dr_up_latest = np.array(grad_omega_dr_up)
                    grad_omega_dr_dn_latest = np.array(grad_omega_dr_dn)

            # sum
            nw_sum = len(w_L_latest)
            w_L_sum = np.sum(w_L_latest)
            w_L_weighted_sum = np.sum(w_L_latest / V_diag_E_latest)
            e_L_weighted_sum = np.sum(w_L_latest / V_diag_E_latest * e_L_latest)
            # MPC: identical intra-step walker reduction to ``e_L_weighted_sum``
            # (same w_L_latest / V_diag_E_latest weights, later divided by the
            # same w_L_weighted_sum), so that sum(w_L * dV) / sum(w_L) stays
            # consistent with the stored e_L.
            if self._mpc:
                mpc_dV_weighted_sum = np.sum(w_L_latest / V_diag_E_latest * mpc_dV_latest)
            # The real e_L2 buffer stores ``|e_L|^2``,
            # mirroring the MCMC pattern. ``np.abs(e_L_latest)**2`` is real;
            # ``V_diag_E_latest`` was Re-projected at the source on the
            # fixed-phase complex Bloch path at its source.
            e_L2_weighted_sum = np.sum(w_L_latest / V_diag_E_latest * e_L2_latest)
            mod_w_L_latest = w_L_latest
            if self._comput_position_deriv:
                # Pathak-Wagner regularization factor f_eps per walker
                if self._epsilon_PW > 0.0:
                    grad_norm_sq = np.sum(grad_ln_Psi_r_up_latest**2, axis=(1, 2)) + np.sum(
                        grad_ln_Psi_r_dn_latest**2, axis=(1, 2)
                    )
                    nodal_dist = 1.0 / np.sqrt(grad_norm_sq)
                    t = nodal_dist / self._epsilon_PW
                    t2 = t * t
                    t4 = t2 * t2
                    t6 = t4 * t2
                    f_eps = np.where(t < 1.0, 7.0 * t6 - 15.0 * t4 + 9.0 * t2, 1.0)
                    mod_w_L_latest = w_L_latest * f_eps
                else:
                    mod_w_L_latest = w_L_latest

                # Compute per-walker force products BEFORE walker averaging
                if self._use_swct and _rs_gfmc_chunk:
                    force_HF_per_walker, force_PP_per_walker = _native_swct_forces(
                        self._hamiltonian_data.structure_data,
                        self._latest_r_up_carts,
                        self._latest_r_dn_carts,
                        grad_e_L_R_latest,
                        grad_e_L_r_up_latest,
                        grad_e_L_r_dn_latest,
                        grad_ln_Psi_dR_latest,
                        grad_ln_Psi_r_up_latest,
                        grad_ln_Psi_r_dn_latest,
                    )
                elif self._use_swct:
                    _sd = self._hamiltonian_data.structure_data
                    _ru_s, _rd_s = self._latest_r_up_carts, self._latest_r_dn_carts
                    # force_HF per walker: shape (n_walkers, n_atoms, 3)
                    force_HF_per_walker = (
                        grad_e_L_R_latest
                        + np.asarray(_jit_vmap_swct_omega_dot_n(_sd, _ru_s, grad_e_L_r_up_latest))
                        + np.asarray(_jit_vmap_swct_omega_dot_n(_sd, _rd_s, grad_e_L_r_dn_latest))
                    )
                    # force_PP per walker: shape (n_walkers, n_atoms, 3)
                    force_PP_per_walker = (
                        grad_ln_Psi_dR_latest
                        + np.asarray(_jit_vmap_swct_omega_dot_n(_sd, _ru_s, grad_ln_Psi_r_up_latest))
                        + np.asarray(_jit_vmap_swct_omega_dot_n(_sd, _rd_s, grad_ln_Psi_r_dn_latest))
                        + 0.5 * (grad_omega_dr_up_latest + grad_omega_dr_dn_latest)
                    )
                else:
                    # With SWCT off, Omega and domega are identically zero.
                    force_HF_per_walker = grad_e_L_R_latest
                    force_PP_per_walker = grad_ln_Psi_dR_latest
                # E_L * force_PP per walker: shape (n_walkers, n_atoms, 3)
                E_L_force_PP_per_walker = e_L_latest[:, None, None] * force_PP_per_walker

                # Weighted sum of per-walker force products
                # Note: GFMC_n uses mod_w_L / V_diag_E weighting
                force_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, force_HF_per_walker)
                force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, force_PP_per_walker)
                E_L_force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, E_L_force_PP_per_walker)

            if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                start_stress = time.perf_counter()
                if _rs_gfmc_chunk and _rs_want_stress:
                    stress_HF_per_walker = _rs_native_stress_HF
                    stress_PP_per_walker = _rs_native_stress_PP
                else:
                    _zero_eps = jnp.zeros((3, 3), dtype=jnp.float64)
                    stress_HF_per_walker = np.asarray(
                        _jit_vmap_local_stress_HF_n(
                            hamiltonian_for_position_grads,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                            latest_RTs,
                            self._non_local_move,
                            self._alat,
                            _zero_eps,
                            self.__latest_A_old_inv,
                        )
                    )
                    stress_PP_per_walker = np.asarray(
                        _grad_ln_psi_eps(
                            hamiltonian_for_position_grads,
                            self._latest_r_up_carts,
                            self._latest_r_dn_carts,
                            self.__latest_A_old_inv,
                        )
                    )
                E_L_stress_PP_per_walker = e_L_latest[:, None, None] * stress_PP_per_walker
                stress_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, stress_HF_per_walker)
                stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, stress_PP_per_walker)
                E_L_stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_latest / V_diag_E_latest, E_L_stress_PP_per_walker)
                timer_stress += time.perf_counter() - start_stress
            # The production Rust/Rayon mode has one MPI rank; its local sums
            # already are the global sums.  Avoid eleven Python-to-MPI calls
            # per branch while retaining the existing collective path for MPI.
            if self._mpi_size > 1:
                nw_sum = self._mpi_comm.reduce(nw_sum, op=MPI.SUM, root=0)
                w_L_sum = self._mpi_comm.reduce(w_L_sum, op=MPI.SUM, root=0)
                w_L_weighted_sum = self._mpi_comm.reduce(w_L_weighted_sum, op=MPI.SUM, root=0)
                e_L_weighted_sum = self._mpi_comm.reduce(e_L_weighted_sum, op=MPI.SUM, root=0)
                e_L2_weighted_sum = self._mpi_comm.reduce(e_L2_weighted_sum, op=MPI.SUM, root=0)
                if self._mpc:
                    mpc_dV_weighted_sum = self._mpi_comm.reduce(mpc_dV_weighted_sum, op=MPI.SUM, root=0)
                if self._comput_position_deriv:
                    force_HF_weighted_sum = self._mpi_comm.reduce(force_HF_weighted_sum, op=MPI.SUM, root=0)
                    force_PP_weighted_sum = self._mpi_comm.reduce(force_PP_weighted_sum, op=MPI.SUM, root=0)
                    E_L_force_PP_weighted_sum = self._mpi_comm.reduce(E_L_force_PP_weighted_sum, op=MPI.SUM, root=0)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_weighted_sum = self._mpi_comm.reduce(stress_HF_weighted_sum, op=MPI.SUM, root=0)
                    stress_PP_weighted_sum = self._mpi_comm.reduce(stress_PP_weighted_sum, op=MPI.SUM, root=0)
                    E_L_stress_PP_weighted_sum = self._mpi_comm.reduce(E_L_stress_PP_weighted_sum, op=MPI.SUM, root=0)

            if self._mpi_rank == 0:
                # averaged
                w_L_averaged = w_L_sum / nw_sum
                e_L_averaged = e_L_weighted_sum / w_L_weighted_sum
                e_L2_averaged = e_L2_weighted_sum / w_L_weighted_sum
                # MPC: same denominator (w_L_weighted_sum) as e_L_averaged.
                if self._mpc:
                    mpc_dV_averaged = mpc_dV_weighted_sum / w_L_weighted_sum
                if self._comput_position_deriv:
                    force_HF_averaged = force_HF_weighted_sum / w_L_weighted_sum
                    force_PP_averaged = force_PP_weighted_sum / w_L_weighted_sum
                    E_L_force_PP_averaged = E_L_force_PP_weighted_sum / w_L_weighted_sum
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_averaged = stress_HF_weighted_sum / w_L_weighted_sum
                    stress_PP_averaged = stress_PP_weighted_sum / w_L_weighted_sum
                    E_L_stress_PP_averaged = E_L_stress_PP_weighted_sum / w_L_weighted_sum
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                if self._mpc:
                    mpc_dV_averaged = np.expand_dims(mpc_dV_averaged, axis=0)
                if self._comput_position_deriv:
                    force_HF_averaged = np.expand_dims(force_HF_averaged, axis=0)
                    force_PP_averaged = np.expand_dims(force_PP_averaged, axis=0)
                    E_L_force_PP_averaged = np.expand_dims(E_L_force_PP_averaged, axis=0)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    stress_HF_averaged = np.expand_dims(stress_HF_averaged, axis=0)
                    stress_PP_averaged = np.expand_dims(stress_PP_averaged, axis=0)
                    E_L_stress_PP_averaged = np.expand_dims(E_L_stress_PP_averaged, axis=0)

                # store  # This should stored only for MPI-rank = 0 !!!
                self._stored_e_L2[self._mcmc_counter + num_mcmc_done] = np.array(e_L2_averaged)
                self._stored_e_L[self._mcmc_counter + num_mcmc_done] = np.array(e_L_averaged)
                self._stored_w_L[self._mcmc_counter + num_mcmc_done] = np.array(w_L_averaged)
                if self._mpc:
                    self._stored_mpc_dV[self._mcmc_counter + num_mcmc_done] = np.array(mpc_dV_averaged)
                if self._comput_position_deriv:
                    self._stored_force_HF[self._mcmc_counter + num_mcmc_done] = np.array(force_HF_averaged)
                    self._stored_force_PP[self._mcmc_counter + num_mcmc_done] = np.array(force_PP_averaged)
                    self._stored_E_L_force_PP[self._mcmc_counter + num_mcmc_done] = np.array(E_L_force_PP_averaged)
                if self._comput_stress and self._hamiltonian_data.structure_data.pbc_flag:
                    self._stored_stress_HF[self._mcmc_counter + num_mcmc_done] = np.array(stress_HF_averaged)
                    self._stored_stress_PP[self._mcmc_counter + num_mcmc_done] = np.array(stress_PP_averaged)
                    self._stored_E_L_stress_PP[self._mcmc_counter + num_mcmc_done] = np.array(E_L_stress_PP_averaged)

            if self._mpi_size > 1:
                self._mpi_comm.Barrier()

            end_collection = time.perf_counter()
            timer_collection += end_collection - start_collection
            logger.devel(f"    timer_collection = {(end_collection - start_collection) * 1e3:.2f} msec.")

            # branching
            start_reconfiguration = time.perf_counter()
            latest_r_up_carts_before_branching = np.array(self._latest_r_up_carts)
            latest_r_dn_carts_before_branching = np.array(self._latest_r_dn_carts)

            #########################################
            # 1. Gather only the weights to MPI_rank=0 and perform branching calculation
            #########################################
            start_ = time.perf_counter()

            # Each process computes the sum of its local walker weights.
            local_weight_sum = np.sum(w_L_latest)

            # Use pickle-based allreduce here (allowed for this part)
            global_weight_sum = (
                local_weight_sum if self._mpi_size == 1 else self._mpi_comm.allreduce(local_weight_sum, op=MPI.SUM)
            )

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Compute the local probabilities for each walker.
            local_probabilities = w_L_latest / global_weight_sum

            # Compute the local cumulative probabilities.
            # NOTE: MPI reductions for branching probabilities are kept float64
            # unconditionally (regardless of the gfmc precision zone) to avoid
            # population collapse from float32 round-off in the branching step.
            local_cumprob = np.cumsum(local_probabilities)
            local_sum_arr = np.array(np.sum(local_probabilities), dtype=np.float64)
            offset_arr = np.zeros(1, dtype=np.float64)
            if self._mpi_size > 1:
                self._mpi_comm.Exscan([local_sum_arr, MPI.DOUBLE], [offset_arr, MPI.DOUBLE], op=MPI.SUM)
            if self._mpi_rank == 0:
                offset = 0.0
            else:
                offset = float(offset_arr[0])
            local_cumprob += offset

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            start_ = time.perf_counter()

            # Gather the local cumulative probability arrays from all processes.
            total_walkers = self.num_walkers * self._mpi_size
            if self._mpi_size == 1:
                global_cumprob = local_cumprob
            else:
                global_cumprob = np.empty(total_walkers, dtype=np.float64)
                self._mpi_comm.Allgather([local_cumprob, MPI.DOUBLE], [global_cumprob, MPI.DOUBLE])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # Total number of walkers across all processes.
            # Compute index range for this rank
            start_ = time.perf_counter()
            start_idx = self._mpi_rank * self.num_walkers
            end_idx = start_idx + self.num_walkers

            # Build only local z-array (length = self.num_walkers)
            z_local = (np.arange(start_idx, end_idx) + zeta) / total_walkers

            # Perform searchsorted and cast the result to int32
            local_chosen_indices = np.searchsorted(global_cumprob, z_local).astype(np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # Gather all local_chosen_indices across ranks using MPI.INT
            start_ = time.perf_counter()

            # Allocate buffer to receive all ranks' indices (Number of indices per rank is identical on every rank)
            if self._mpi_size == 1:
                all_chosen_buf = local_chosen_indices
            else:
                all_chosen_buf = np.empty(self.num_walkers * self._mpi_size, dtype=np.int32)

                # Perform the all-gather operation with 32-bit integers
                self._mpi_comm.Allgather([local_chosen_indices, MPI.INT], [all_chosen_buf, MPI.INT])

            # Use the gathered indices for global statistics
            chosen_walker_indices = all_chosen_buf
            num_survived_walkers = len(np.unique(chosen_walker_indices))
            num_killed_walkers = total_walkers - num_survived_walkers

            # Build the local assignment list of (source_rank, source_local_index)
            local_assignment = [
                (src_global_idx // self.num_walkers, src_global_idx % self.num_walkers)
                for src_global_idx in local_chosen_indices
            ]

            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 2. In each process, prepare for data exchange based on the new walker selection
            #########################################
            start_ = time.perf_counter()
            reqs = {}
            if self._mpi_size == 1:
                # The production Rayon path uses one MPI rank.  NumPy's
                # advanced indexing performs the complete systematic-resample
                # copy in C and replaces 128 pairs of Python slice assignments.
                latest_r_up_carts_after_branching = latest_r_up_carts_before_branching[local_chosen_indices].copy()
                latest_r_dn_carts_after_branching = latest_r_dn_carts_before_branching[local_chosen_indices].copy()
            else:
                latest_r_up_carts_after_branching = np.empty_like(latest_r_up_carts_before_branching)
                latest_r_dn_carts_after_branching = np.empty_like(latest_r_dn_carts_before_branching)

                for dest_idx, (src_rank, src_local_idx) in enumerate(local_assignment):
                    if src_rank == self._mpi_rank:
                        # Local copy: no communication needed
                        latest_r_up_carts_after_branching[dest_idx] = latest_r_up_carts_before_branching[src_local_idx]
                        latest_r_dn_carts_after_branching[dest_idx] = latest_r_dn_carts_before_branching[src_local_idx]
                    else:
                        reqs.setdefault(src_rank, []).append((dest_idx, src_local_idx))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 2 = {(end_ - start_) * 1e3:.3f} msec.")

            #########################################
            # 3. Exchange only the necessary walker data between processes using asynchronous communication
            #########################################

            # 3.1.1: Flatten `reqs` into an (N_req x 3) int32 array of triplets
            start_ = time.perf_counter()
            flat_list = [
                (src_rank, dest_idx, src_local_idx) for src_rank, pairs in reqs.items() for dest_idx, src_local_idx in pairs
            ]
            triplets = np.array(flat_list, dtype=np.int32) if flat_list else np.empty((0, 3), dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.1 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.2: Compute how many ints to send to each rank (3 ints per request)
            start_ = time.perf_counter()
            counts_per_rank = np.bincount(triplets[:, 0], minlength=self._mpi_size)  # # reqs per src_rank
            send_counts = (counts_per_rank * 3).astype(np.int32)  # # ints per src_rank
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.3: Post nonblocking Alltoall to exchange counts
            start_ = time.perf_counter()
            if self._mpi_size == 1:
                recv_counts = send_counts.copy()
                req_counts = None
            else:
                recv_counts = np.empty_like(send_counts)
                req_counts = self._mpi_comm.Ialltoall([send_counts, MPI.INT], [recv_counts, MPI.INT])
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.4: Build send_buf while counts exchange is in flight
            start_ = time.perf_counter()
            #   sort by src_rank so that each destination's data is contiguous
            order = np.argsort(triplets[:, 0], kind="mergesort") if triplets.size else np.empty(0, dtype=np.int32)
            sorted_tr = triplets[order]  # shape = (N_req, 3)
            send_buf = sorted_tr.ravel()  # shape = (N_req*3,)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.5: Wait for counts exchange to complete
            start_ = time.perf_counter()
            if req_counts is not None:
                req_counts.Wait()
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.6: Build displacements for send/recv from counts
            start_ = time.perf_counter()
            send_displs = np.zeros_like(send_counts)
            send_displs[1:] = np.cumsum(send_counts)[:-1]
            recv_displs = np.zeros_like(recv_counts)
            recv_displs[1:] = np.cumsum(recv_counts)[:-1]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.7: Allocate recv buffer of the exact size
            start_ = time.perf_counter()
            total_recv = int(np.sum(recv_counts))
            recv_buf = np.empty(total_recv, dtype=np.int32)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.7 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.8: Post blocking Alltoallv to exchange the triplets
            start_ = time.perf_counter()
            if self._mpi_size == 1:
                recv_buf[:] = send_buf
            else:
                self._mpi_comm.Alltoallv(
                    [send_buf, send_counts, send_displs, MPI.INT], [recv_buf, recv_counts, recv_displs, MPI.INT]
                )
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.8 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.9: Wait for data to arrive and reconstruct per-process request dicts
            start_ = time.perf_counter()
            all_reqs = []
            for p in range(self._mpi_size):
                off = recv_displs[p]
                cnt = recv_counts[p]
                block = recv_buf[off : off + cnt]
                if block.size == 0:
                    all_reqs.append({})
                    continue
                rec_tr = block.reshape(-1, 3)
                proc_dict = {}
                for sr, dest_idx, src_local_idx in rec_tr:
                    proc_dict.setdefault(int(sr), []).append((int(dest_idx), int(src_local_idx)))
                all_reqs.append(proc_dict)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.9 = {(end_ - start_) * 1e3:.3f} msec.")

            # 3.1.10: Filter out empty request dicts
            start_ = time.perf_counter()
            non_empty_all_reqs = [(p, rd) for p, rd in enumerate(all_reqs) if rd]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.1.10 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-2. Build incoming_reqs: who needs data from me? ---
            start_ = time.perf_counter()
            incoming_reqs = [
                (p, src_local_idx, dest_idx)
                for p, proc_req in non_empty_all_reqs
                if p != self._mpi_rank
                for dest_idx, src_local_idx in proc_req.get(self._mpi_rank, [])
            ]
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.2 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-3. Post nonblocking receives using Irecv for both up and dn buffers. ---
            start_ = time.perf_counter()
            recv_buffers = {}
            recv_reqs_up = {}
            recv_reqs_dn = {}
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                count = len(req_list)
                shape_up = latest_r_up_carts_before_branching.shape[1:]
                shape_dn = latest_r_dn_carts_before_branching.shape[1:]
                buf_up = np.empty((count, *shape_up), dtype=latest_r_up_carts_before_branching.dtype)
                buf_dn = np.empty((count, *shape_dn), dtype=latest_r_dn_carts_before_branching.dtype)
                recv_buffers[src_rank] = (buf_up, buf_dn)
                recv_reqs_up[src_rank] = self._mpi_comm.Irecv([buf_up, MPI.DOUBLE], source=src_rank, tag=200)
                recv_reqs_dn[src_rank] = self._mpi_comm.Irecv([buf_dn, MPI.DOUBLE], source=src_rank, tag=201)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.3 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-4. Prepare and post nonblocking sends using Isend. ---
            start_ = time.perf_counter()
            send_requests = []
            for dest_rank, group in groupby(sorted(incoming_reqs, key=lambda x: x[0]), key=lambda x: x[0]):
                idxs = [src_local for (_, src_local, _) in group]
                buf_up = latest_r_up_carts_before_branching[idxs]
                buf_dn = latest_r_dn_carts_before_branching[idxs]
                send_requests.append(self._mpi_comm.Isend([buf_up, MPI.DOUBLE], dest=dest_rank, tag=200))
                send_requests.append(self._mpi_comm.Isend([buf_dn, MPI.DOUBLE], dest=dest_rank, tag=201))
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.4 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-5. Wait for all nonblocking sends to complete. ---
            start_ = time.perf_counter()
            MPI.Request.Waitall(send_requests)
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.5 = {(end_ - start_) * 1e3:.3f} msec.")

            # --- 3-6. Process the received walker data. ---
            start_ = time.perf_counter()
            for src_rank, req_list in reqs.items():
                if not req_list:
                    continue
                recv_reqs_up[src_rank].Wait()
                recv_reqs_dn[src_rank].Wait()
                buf_up, buf_dn = recv_buffers[src_rank]
                dest_idxs = [dest for (dest, _) in req_list]
                latest_r_up_carts_after_branching[dest_idxs] = buf_up
                latest_r_dn_carts_after_branching[dest_idxs] = buf_dn
            end_ = time.perf_counter()
            logger.devel(f"    timer_reconfigration step 3.6 = {(end_ - start_) * 1e3:.3f} msec.")

            # here update the walker positions!!
            self._num_survived_walkers += num_survived_walkers
            self._num_killed_walkers += num_killed_walkers
            if _rs_gfmc_chunk:
                self._latest_r_up_carts = latest_r_up_carts_after_branching
                self._latest_r_dn_carts = latest_r_dn_carts_after_branching
            else:
                self._latest_r_up_carts = jnp.asarray(latest_r_up_carts_after_branching, dtype=jnp.float64)
                self._latest_r_dn_carts = jnp.asarray(latest_r_dn_carts_after_branching, dtype=jnp.float64)
            # SM-carry reseed: JAX projection path only (the native chunk evaluates
            # self-contained -- no A_inv crosses the boundary).
            if not _rs_gfmc_chunk:
                self.__latest_A_old_inv = _jit_vmap_A_inv_n(self._latest_r_up_carts, self._latest_r_dn_carts)
                # block before Barrier so the A_inv GPU work is included in timer_reconfiguration
                # and dispatch-queue skew does not leak into the next step's barrier wait.
                self.__latest_A_old_inv.block_until_ready()

            if self._mpi_size > 1:
                self._mpi_comm.Barrier()

            end_reconfiguration = time.perf_counter()
            timer_reconfiguration += end_reconfiguration - start_reconfiguration
            logger.devel(f"    timer_reconfiguration total = {(end_reconfiguration - start_reconfiguration) * 1e3:.2f} msec.")

            # update E_scf
            start_update_E_scf = time.perf_counter()

            ## parameters for E_scf
            num_gfmc_collect_steps = GFMC_ON_THE_FLY_COLLECT_STEPS
            num_gfmc_bin_blocks = GFMC_ON_THE_FLY_BIN_BLOCKS
            # History and update phase belong to the entire chain, not to this
            # run() call. A restart must not discard its first K new products
            # or repeat the initial jackknife warmup with an already-filled G.
            chain_step = self._mcmc_counter + i_mcmc_step

            # (A) accumulate __G_L / __G_e_L every step (after enough stored_w_L)
            if self._mpi_rank == 0:
                if chain_step >= num_gfmc_collect_steps:
                    e_L = self._stored_e_L[self._mcmc_counter + num_mcmc_done]
                    w_L = self._stored_w_L[
                        self._mcmc_counter + num_mcmc_done - num_gfmc_collect_steps : self._mcmc_counter + num_mcmc_done
                    ]
                    G_L = np.prod(w_L, axis=0)
                    self._G_L.append(G_L)
                    self._G_e_L.append(G_L * e_L)

            # (B) E_scf update schedule:
            #     - rapid phase (chain_step < mcmc_interval = N/100): update every step
            #     - thereafter:                                        update every mcmc_interval steps
            #     - skip when there are not yet enough G_L samples for jackknife
            n_G_L = max(0, chain_step - num_gfmc_collect_steps + 1)
            have_enough = n_G_L >= num_gfmc_bin_blocks
            in_rapid_phase = chain_step < mcmc_interval
            on_throttle = (chain_step + 1) % mcmc_interval == 0

            if have_enough and (in_rapid_phase or on_throttle):
                if self._mpi_rank == 0:
                    # Skip the bad-regime stored_w_L that bleed into __G_L via the K-product.
                    # During the very early phase the available samples are limited, so
                    # clamp to keep at least num_gfmc_bin_blocks samples for jackknife.
                    num_gfmc_warmup_steps = min(
                        num_gfmc_collect_steps + num_gfmc_bin_blocks,
                        max(n_G_L - num_gfmc_bin_blocks, 0),
                    )
                    G_eq = np.array(self._G_L[num_gfmc_warmup_steps:])
                    G_e_L_eq = np.array(self._G_e_L[num_gfmc_warmup_steps:])
                    G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
                    G_e_L_binned = np.array([np.sum(G_e_L_list) for G_e_L_list in G_e_L_split])
                    G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
                    G_binned = np.array([np.sum(G_list) for G_list in G_split])
                    G_e_L_binned_sum = np.sum(G_e_L_binned)
                    G_binned_sum = np.sum(G_binned)
                    E_jackknife = [
                        (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)
                    ]
                    # Physical-real projection at the end of the
                    # jackknife: ``E_jackknife`` is complex on twist k != 0
                    # (the cross-correlator numerator carries Im); the energy
                    # observable is the Re part. Bit-identical at Gamma.
                    E_mean = float(np.real(np.average(E_jackknife)))
                    E_std = float(np.sqrt(num_gfmc_bin_blocks - 1) * np.std(np.real(E_jackknife)))
                else:
                    E_mean = None
                    E_std = None

                if self._mpi_size > 1:
                    E_mean = self._mpi_comm.bcast(E_mean, root=0)
                    E_std = self._mpi_comm.bcast(E_std, root=0)

                self._E_scf = E_mean
                logger.debug(f"    E_scf = {self._E_scf:.5f} +- {E_std:.5f} Ha.")

            if self._mpi_size > 1:
                self._mpi_comm.Barrier()
            end_update_E_scf = time.perf_counter()
            timer_update_E_scf += end_update_E_scf - start_update_E_scf
            logger.devel(f"    timer_update_E_scf = {(end_update_E_scf - start_update_E_scf) * 1e3:.2f} msec.")

            # count up, here is the end of the branching step. Count it before the
            # stop checks below: a step that has run (walkers, RNG and stored
            # observables) must be kept when the loop stops after it.
            num_mcmc_done += 1

            gfmc_current = time.perf_counter()

            # Use rank 0's decision and broadcast to all ranks to prevent
            # MPI deadlock caused by different ranks disagreeing on whether
            # to break (due to slight differences in gfmc_total_start).
            should_stop = bool(max_time < gfmc_current - gfmc_total_start) if self._mpi_rank == 0 else False
            if self._mpi_size > 1:
                should_stop = self._mpi_comm.bcast(should_stop, root=0)
            if should_stop:
                logger.info(f"  Stopping... Max_time = {max_time} sec. exceeds.")
                logger.info("  Break the branching loop.")
                break

            # Check the control file periodically; shared across all ranks / twists.
            poll_control = (i_mcmc_step + 1) % control_poll_interval == 0 or i_mcmc_step + 1 == num_mcmc_steps
            stop_flag = _read_external_stop_flag(toml_filename, self._mpi_comm) if poll_control else False
            if stop_flag:
                logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                logger.info("  Break the mcmc loop.")
                break

        progress = (num_mcmc_done + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
        gfmc_total_current = time.perf_counter()
        logger.info(
            f"  Progress: GFMC step = {num_mcmc_done + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %. Elapsed time = {(gfmc_total_current - gfmc_loop_start):.1f} sec."
        )
        logger.info("")

        # count up mcmc_counter
        self._mcmc_counter += num_mcmc_done

        # -- Truncate stored arrays to actual size (rank 0 only) --
        if self._mpi_rank == 0:
            ns = self._mcmc_counter
            self._stored_e_L = self._stored_e_L[:ns]
            self._stored_e_L2 = self._stored_e_L2[:ns]
            self._stored_w_L = self._stored_w_L[:ns]
            if self._mpc:
                self._stored_mpc_dV = self._stored_mpc_dV[:ns]
            if self._comput_position_deriv:
                self._stored_force_HF = self._stored_force_HF[:ns]
                self._stored_force_PP = self._stored_force_PP[:ns]
                self._stored_E_L_force_PP = self._stored_E_L_force_PP[:ns]
            if self._comput_stress:
                self._stored_stress_HF = self._stored_stress_HF[:ns]
                self._stored_stress_PP = self._stored_stress_PP[:ns]
                self._stored_E_L_stress_PP = self._stored_E_L_stress_PP[:ns]
            self.__validate_stored_shapes()

        gfmc_total_end = time.perf_counter()
        timer_gfmc_total = gfmc_total_end - gfmc_total_start
        # The stress block runs inside the walker-observable-collection window; report it separately.
        timer_collection -= timer_stress
        timer_misc = timer_gfmc_total - (
            timer_projection_init
            + timer_projection_total
            + timer_e_L
            + timer_de_L_dR_dr
            + timer_dln_Psi_dR_dr
            + timer_dln_Psi_dc
            + timer_de_L_dc
            + timer_stress
            + timer_mpi_barrier
            + timer_collection
            + timer_reconfiguration
            + timer_update_E_scf
        )

        # remove the control file (world root owns it; EAFP-guarded)
        self._mpi_comm.Barrier()
        _remove_external_control_file(toml_filename, self._world_comm)

        # net GFMC time
        timer_net_gfmc_total = timer_gfmc_total - timer_projection_init

        # average among MPI processes
        ave_timer_gfmc_total = self._mpi_comm.allreduce(timer_gfmc_total, op=MPI.SUM) / self._mpi_size
        ave_timer_projection_init = self._mpi_comm.allreduce(timer_projection_init, op=MPI.SUM) / self._mpi_size
        ave_timer_net_gfmc_total = self._mpi_comm.allreduce(timer_net_gfmc_total, op=MPI.SUM) / self._mpi_size
        ave_timer_projection_total = (
            self._mpi_comm.allreduce(timer_projection_total, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        )
        ave_timer_e_L = self._mpi_comm.allreduce(timer_e_L, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_de_L_dR_dr = self._mpi_comm.allreduce(timer_de_L_dR_dr, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_dln_Psi_dR_dr = self._mpi_comm.allreduce(timer_dln_Psi_dR_dr, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_dln_Psi_dc = self._mpi_comm.allreduce(timer_dln_Psi_dc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_de_L_dc = self._mpi_comm.allreduce(timer_de_L_dc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_mpi_barrier = self._mpi_comm.allreduce(timer_mpi_barrier, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_collection = self._mpi_comm.allreduce(timer_collection, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_stress = self._mpi_comm.allreduce(timer_stress, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_reconfiguration = self._mpi_comm.allreduce(timer_reconfiguration, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_update_E_scf = self._mpi_comm.allreduce(timer_update_E_scf, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        ave_timer_misc = self._mpi_comm.allreduce(timer_misc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        sum_killed_walkers = self._mpi_comm.allreduce(self._num_killed_walkers, op=MPI.SUM)
        sum_survived_walkers = self._mpi_comm.allreduce(self._num_survived_walkers, op=MPI.SUM)

        logger.info(f"Total GFMC time for {num_mcmc_done} branching steps = {ave_timer_gfmc_total: .3f} sec.")
        logger.info(f"Pre-compilation time for GFMC = {ave_timer_projection_init: .3f} sec.")
        logger.info(f"Net GFMC time without pre-compilations = {ave_timer_net_gfmc_total: .3f} sec.")
        logger.info(f"Elapsed times per branching, averaged over {num_mcmc_done} branching steps.")
        logger.info(f"  Projection between branching = {ave_timer_projection_total * 10**3: .3f} msec.")
        logger.info(f"  Time for computing e_L = {ave_timer_e_L * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {ave_timer_de_L_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {ave_timer_dln_Psi_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {ave_timer_dln_Psi_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {ave_timer_de_L_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/deps and dln_Psi/deps = {ave_timer_stress * 10**3:.2f} msec.")
        logger.info(f"  Time for MPI barrier before branching = {ave_timer_mpi_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for walker observable collections time per branching = {ave_timer_collection * 10**3: .3f} msec.")
        logger.info(f"  Time for walker reconfiguration time per branching = {ave_timer_reconfiguration * 10**3: .3f} msec.")
        logger.info(f"  Time for updating E_scf = {ave_timer_update_E_scf * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(
            f"Survived walkers ratio = {sum_survived_walkers / (sum_survived_walkers + sum_killed_walkers) * 100:.2f} %. Ideal is ~ 98 %. Adjust num_mcmc_per_measurement."
        )
        logger.info("")

        _log_memory_usage("gfmc_n_post_run", comm=self._mpi_comm)


class _GFMC_n_debug(_GFMCBase):
    """GFMC class. Running GFMC with multiple walkers.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        num_walkers (int):
            the number of walkers
        num_mcmc_per_measurement (int):
            the number of GFMC projections between two branchings (measurements), an integer >= 1.
            Required, no default (as ``[lrdmc-bra] num_mcmc_per_measurement`` in the CLI): omitting it
            or passing ``None`` raises ``ValueError`` before any sampling. One projection advances
            imaginary time by about ``alat**2 / (3 N_e)``, so the value sets the imaginary time per
            measurement and depends on the system and on ``alat``. The given integer is used unchanged.
        num_gfmc_collect_steps (int):
            the number of steps to collect the GFMC data
        mcmc_seed (int):
            seed for the MCMC chain.
        E_scf (float):
            Self-consistent E (Hartree)
        alat (float):
            discretized grid length (bohr)
        non_local_move (str):
            treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
            Valid only for ECP calculations. Do not specify this value for all-electron calculations.
        random_discretized_mesh (bool):
            Flag for the random discretization mesh in the kinetic part and in the non-local part of ECPs.
        comput_position_deriv (bool):
            if True, compute the derivatives of E wrt. atomic positions.
        comput_stress (bool):
            if True, compute and retain periodic stress observables. Defaults to False.
        epsilon_PW (float):
            Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is
            regularized near the nodal surface using ``f_epsilon`` from
            `Pathak and Wagner (2020) <https://doi.org/10.1063/5.0004008>`_.
            Default is 0.0 (no regularization).
        use_swct (bool):
            if True, apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
            Default is False for GFMC.
        mpi_comm_in (MPI.Comm | None):
            sampling communicator of this instance. ``None`` uses the module global ``mpi_comm``
            (``COMM_WORLD``); the twist-averaged driver passes one communicator per twist.
        world_comm_in (MPI.Comm | None):
            all-rank communicator. ``None`` uses ``mpi_comm_in``.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int | None = None,
        num_gfmc_collect_steps: int = 5,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        E_scf: float = 0.0,
        alat: float = 0.1,
        non_local_move: NonLocalMove = "tmove",
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        random_discretized_mesh: bool = False,
        epsilon_PW: float = 0.0,
        use_swct: bool = False,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Initialize a _GFMC_n_debug driver and set up walker state."""
        # Communicator injection on the debug
        # sibling. Defaults fall back to module global mpi_comm.
        self._mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        self._world_comm = world_comm_in if world_comm_in is not None else self._mpi_comm
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_size = self._mpi_comm.Get_size()

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # attributes
        self._hamiltonian_data = hamiltonian_data
        self._num_walkers = num_walkers
        _require_lrdmc_bra_num_mcmc_per_measurement(num_mcmc_per_measurement)
        self._num_mcmc_per_measurement = num_mcmc_per_measurement
        self._num_gfmc_collect_steps = num_gfmc_collect_steps
        self._mcmc_seed = mcmc_seed
        self._E_scf = E_scf
        self._alat = alat
        self._random_discretized_mesh = random_discretized_mesh
        self._non_local_move = non_local_move

        # derivative flags
        self._comput_position_deriv = comput_position_deriv
        self._comput_stress = comput_stress
        if self._comput_stress and not self._comput_position_deriv:
            raise ValueError("comput_stress=True requires comput_position_deriv=True.")
        self._use_swct = use_swct

        # Pathak-Wagner regularization (DOI: 10.1063/5.0004008)
        self._epsilon_PW = epsilon_PW

        # Initialization
        # Hierarchical SeedSequence derivation (single derivation point:
        # ``derive_rank_rng_streams``; stream key = [*chain_seed_words, rank]).
        # ``_mpi_seed`` = derived 63-bit JAX base-key int; ``_np_seed_words``
        # = eight uint32 init_by_array words for the NumPy global RNG.
        self._np_seed_words, self._mpi_seed = derive_rank_rng_streams(self._mcmc_seed, self._mpi_rank)
        self._branching_rng = np.random.RandomState(self._np_seed_words)
        self._jax_PRNG_key = jax.random.PRNGKey(self._mpi_seed)
        # Use jax.random.split (batched) instead of a Python list-comp of
        # fold_in calls; the latter scaled linearly with num_walkers and
        # dominated init time at large walker counts (e.g. nw = 16384).
        self._jax_PRNG_key_list_init = jax.random.split(self._jax_PRNG_key, self._num_walkers)
        self._jax_PRNG_key_list = self._jax_PRNG_key_list_init

        # initialize random seed
        np.random.seed(self._np_seed_words)

        # Place electrons around each nucleus with improved spin assignment
        ## check the number of electrons
        tot_num_electron_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        tot_num_electron_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - np.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

        coords = hamiltonian_data.structure_data._positions_cart_jnp
        # Cell for PBC wrapping of the seed configuration (None for molecules).
        cell = hamiltonian_data.structure_data.cell if hamiltonian_data.structure_data.pbc_flag else None

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self._num_walkers, charges, coords, cell
        )

        # Electron assignment for all atoms is complete.

        dtype_jnp = jnp.float64
        self._latest_r_up_carts = jnp.asarray(r_carts_up, dtype=dtype_jnp)
        self._latest_r_dn_carts = jnp.asarray(r_carts_dn, dtype=dtype_jnp)

        logger.debug(f"  initial r_up_carts= {self._latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self._latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self._latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self._latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {self._mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self._num_walkers}.")
        logger.info("")

        # SWCT data: the SWCT functions take structure_data directly.

        # init attributes
        self.__init_attributes()

    def __init_attributes(self) -> None:
        # mcmc counter
        self._mcmc_counter = 0

        # gfmc accepted/rejected moves
        self._num_survived_walkers = 0
        self._num_killed_walkers = 0

        # e_L is complex per walker at a non-Gamma Bloch twist.
        # Mirrors the MCMC / production GFMC pattern.
        self._wf_is_complex = np.issubdtype(
            self._hamiltonian_data.wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating
        )

        # stored local energy (w_L)
        self._stored_w_L = []

        # stored local energy (e_L)
        self._stored_e_L = []

        # stored local energy (e_L2)
        self._stored_e_L2 = []

        # stored force_HF (pre-computed force product)
        self._stored_force_HF = []

        # stored force_PP (pre-computed force product)
        self._stored_force_PP = []

        # stored E_L * force_PP (pre-computed force product)
        self._stored_E_L_force_PP = []

        # stored per-step HF / Pulay stress tensors (1, 3, 3) and e_L * stress_PP.
        # Mirrors the production GFMC buffers so the debug route exposes
        # get_aT / get_aP for the production-vs-debug stress parity.
        self._stored_stress_HF = []
        self._stored_stress_PP = []
        self._stored_E_L_stress_PP = []

    # collecting factor
    @property
    def num_gfmc_collect_steps(self) -> int:
        """Return num_gfmc_collect_steps."""
        return self._num_gfmc_collect_steps

    @num_gfmc_collect_steps.setter
    def num_gfmc_collect_steps(self, num_gfmc_collect_steps: int) -> None:
        """Set num_gfmc_collect_steps."""
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            logger.warning(f"num_gfmc_collect_steps should be larger than {GFMC_MIN_COLLECT_STEPS}.")
        self._num_gfmc_collect_steps = num_gfmc_collect_steps

    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Return hamiltonian_data."""
        return self._hamiltonian_data

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, 1)."""
        return _compute_G_L_debug(np.array(self._stored_w_L), self._num_gfmc_collect_steps)

    def _retained_w_L(self, num_warmup_steps: int) -> npt.NDArray:
        """Accumulated weights after ``num_warmup_steps``, scaled by one common power of two (debug twin)."""
        return _compute_retained_G_L_debug(np.array(self._stored_w_L), self._num_gfmc_collect_steps, num_warmup_steps)

    def run(self, num_mcmc_steps: int = 50) -> None:
        """Run LRDMC with multiple walkers.

        Args:
            num_mcmc_steps (int): number of branching steps (reconfigurations of walkers).
        """
        logger.info("")
        logger.info("This is a debugging class! It supposed to be very slow.")
        logger.info("")

        # The ln|Psi| drift (r), Pulay
        # force (R) / Pulay stress (eps) grads route through the unified
        # ``jqmc.derivatives`` interface with ``analytic=_ANALYTIC_GRAD_ENABLED``
        # selected by the shared backend flag, looped per walker. The same flag
        # unifies the e_L Hellmann-Feynman force / stress onto the SAME flag: when
        # enabled they use the verified analytic continuum-measurement kernels
        # ``gfmc.grad_e_L_n_wrt_{positions,strain}`` (RT = latest_RT[i]),
        # else the LRDMC lattice-discretised ``grad`` / ``jacrev`` fallback
        # of ``_compute_local_energy_n_debug`` (flag off only; no configuration is
        # carved out to autograd -- see ``_analytic_hf_enabled_for``). So production AND
        # this debug oracle share ONE derivative backend flag. ln|Psi| is
        # potential-free -> LRDMC-independent. Imported lazily to avoid the
        # ``derivatives._auto`` -> ``jqmc_mcmc`` strained-helper import cycle.
        from . import derivatives

        _pos_flags = {"opt_r": True, "opt_R": True}
        _eps_flags = {"opt_eps": True}

        def _generate_rotation_matrix_n_debug(
            alpha: jnpt.ArrayLike,
            beta: jnpt.ArrayLike,
            gamma: jnpt.ArrayLike,
        ) -> jax.Array:
            # Precompute all necessary cosines and sines
            cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
            cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
            cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)

            # Combine the rotations directly
            R = jnp.array(
                [
                    [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
                    [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
                    [-sin_b, cos_b * sin_a, cos_a * cos_b],
                ],
                dtype=jnp.float64,
            )
            return R

        def _projection_n_debug(
            init_w_L: float,
            init_r_up_carts: jnpt.ArrayLike,
            init_r_dn_carts: jnpt.ArrayLike,
            init_jax_PRNG_key: jnpt.ArrayLike,
            E_scf: float,
            num_mcmc_per_measurement: int,
            non_local_move: NonLocalMove,
            alat: float,
            hamiltonian_data: Hamiltonian_data,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            """Do projection, compatible with vmap.

            Do projection for a set of (r_up_cart, r_dn_cart).

            Args:
                init_w_L (float): weight before projection
                init_r_up_carts (jnpt.ArrayLike): up-electron positions (N_e^up, 3) before projection
                init_r_dn_carts (jnpt.ArrayLike): dn-electron positions (N_e^dn, 3) before projection
                init_jax_PRNG_key (jnpt.ArrayLike): JAX PRNG key before projection
                E_scf (float): Self-consistent E (Hartree)
                num_mcmc_per_measurement (int): the number of GFMC projections per measurement
                non_local_move (str): treatment of the spin-flip term. tmove (Casula's T-move) or dtmove (Determinant Locality Approximation with Casula's T-move)
                alat (float): discretized grid length (bohr)
                hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data

            Returns:
                latest_w_L (float): weight after the final projection
                latest_r_up_carts (N_e^up, 3) after the final projection
                latest_r_dn_carts (N_e^dn, 3) after the final projection
            """
            w_L, r_up_carts, r_dn_carts, jax_PRNG_key = init_w_L, init_r_up_carts, init_r_dn_carts, init_jax_PRNG_key
            R = jnp.eye(3, dtype=jnp.float64)
            ecp_R = jnp.eye(3, dtype=jnp.float64)

            for projection_step in range(num_mcmc_per_measurement):
                # compute diagonal elements, kinetic part
                diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

                # compute continuum kinetic energy
                diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                    compute_kinetic_energy_all_elements(
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # One rotation-key split per substep; the first rotation is held
                # for every substep in this projection interval.
                jax_PRNG_key, rotation_key = jax.random.split(jax_PRNG_key)
                if projection_step == 0:
                    kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rotation_key)
                    R = _sample_lrdmc_rotation_matrix(kinetic_key, self._random_discretized_mesh)
                    ecp_R = _sample_lrdmc_rotation_matrix(ecp_key, self._random_discretized_mesh)

                # compute discretized kinetic energy and mesh (with a random rotation)
                mesh_kinetic_part_r_up_carts, mesh_kinetic_part_r_dn_carts, elements_non_diagonal_kinetic_part = (
                    compute_discretized_kinetic_energy(
                        alat=alat,
                        wavefunction_data=hamiltonian_data.wavefunction_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                        RT=R.T,
                    )
                )
                # spin-filp
                elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
                non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
                diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
                # regularizations
                elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
                sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
                non_diagonal_kinetic_part_elements = jnp.sum(
                    elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1
                )
                sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(
                    sign_flip_flags_elements, [len(r_up_carts)]
                )
                non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                    non_diagonal_kinetic_part_elements, [len(r_up_carts)]
                )

                # compute diagonal elements, el-el
                diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                # compute diagonal elements, ion-ion
                diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data
                )

                # compute diagonal elements, el-ion
                diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                    compute_coulomb_potential_ei_dispatch(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )
                )

                # compute diagonal elements, el-ion, discretized
                (
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                    diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
                ) = compute_coulomb_potential_ei_discretized_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                    alat=alat,
                )

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_up = (
                    diagonal_bare_coulomb_part_el_ion_elements_up
                    + diagonal_kinetic_continuum_elements_up
                    - non_diagonal_kinetic_part_elements_up
                )
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
                diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
                )
                diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                    sign_flip_flags_elements_up,
                    diagonal_bare_coulomb_part_el_ion_max_up,
                    diagonal_bare_coulomb_part_el_ion_zv_up,
                )

                # compose discretized el-ion potentials
                diagonal_bare_coulomb_part_el_ion_zv_dn = (
                    diagonal_bare_coulomb_part_el_ion_elements_dn
                    + diagonal_kinetic_continuum_elements_dn
                    - non_diagonal_kinetic_part_elements_dn
                )
                # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
                # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
                else:
                    diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
                diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                    diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
                )
                diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                    sign_flip_flags_elements_dn,
                    diagonal_bare_coulomb_part_el_ion_max_dn,
                    diagonal_bare_coulomb_part_el_ion_zv_dn,
                )

                # final bare coulomb part
                discretized_diagonal_bare_coulomb_part = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
                )

                # if-else for all-electron, ECP with tmove, and ECP with dltmove
                # with ECP
                if hamiltonian_data.coulomb_potential_data.ecp_flag:
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    # ecp local
                    diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    if non_local_move == "tmove":
                        # ecp non-local (t-move)
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=False,
                                RT=ecp_R.T,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    elif non_local_move == "dltmove":
                        mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                            compute_ecp_non_local_parts_nearest_neighbors(
                                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                                wavefunction_data=hamiltonian_data.wavefunction_data,
                                r_up_carts=r_up_carts,
                                r_dn_carts=r_dn_carts,
                                flag_determinant_only=True,
                                RT=ecp_R.T,
                            )
                        )

                        V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                        diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                        Jastrow_ref = compute_Jastrow_part_ratio_zone(
                            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                        )
                        Jastrow_on_mesh = jnp.stack(
                            [
                                compute_Jastrow_part_ratio_zone(
                                    hamiltonian_data.wavefunction_data.jastrow_data,
                                    mesh_non_local_ecp_part_r_up_carts[_i],
                                    mesh_non_local_ecp_part_r_dn_carts[_i],
                                )
                                for _i in range(len(mesh_non_local_ecp_part_r_up_carts))
                            ]
                        )
                        Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)

                        V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                        non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                        non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                        diagonal_sum_hamiltonian = (
                            diagonal_kinetic_part
                            + discretized_diagonal_bare_coulomb_part
                            + diagonal_ecp_local_part
                            + diagonal_kinetic_part_SP
                            + diagonal_ecp_part_SP
                        )

                    else:
                        logger.error(f"non_local_move = {non_local_move} is not yet implemented.")
                        raise NotImplementedError

                    # probability
                    p_list = jnp.concatenate([jnp.ravel(elements_non_diagonal_kinetic_part_FN), jnp.ravel(V_nonlocal_FN)])
                    # Under complex Bloch fixed-phase, take Re for the
                    # categorical sampling so the cdf passed to searchsorted is
                    # real-valued. Bit-identical at Gamma where p_list is real.
                    p_list = jnp.real(p_list)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_up_carts, mesh_non_local_ecp_part_r_up_carts], axis=0
                    )
                    non_diagonal_move_mesh_r_dn_carts = jnp.concatenate(
                        [mesh_kinetic_part_r_dn_carts, mesh_non_local_ecp_part_r_dn_carts], axis=0
                    )

                # with all electrons
                else:
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic
                    # compute local energy, i.e., sum of all the hamiltonian (with importance sampling)
                    p_list = jnp.ravel(elements_non_diagonal_kinetic_part_FN)
                    # Under complex Bloch fixed-phase, take Re for the
                    # categorical sampling so the cdf passed to searchsorted is
                    # real-valued. Bit-identical at Gamma where p_list is real.
                    p_list = jnp.real(p_list)
                    non_diagonal_move_probabilities = p_list / p_list.sum()
                    non_diagonal_move_mesh_r_up_carts = mesh_kinetic_part_r_up_carts
                    non_diagonal_move_mesh_r_dn_carts = mesh_kinetic_part_r_dn_carts

                    diagonal_sum_hamiltonian = (
                        diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP
                    )

                # compute b_L_bar
                # Complex Bloch fixed-phase mirrors the GFMC_n production route.
                # Fixed-phase branching discards the imaginary phase component.
                b_x_bar = -1.0 * jnp.real(non_diagonal_sum_hamiltonian)

                # compute bar_b_L
                b_x = 1.0 / (jnp.real(diagonal_sum_hamiltonian) - E_scf) * b_x_bar

                # update weight
                w_L = w_L * b_x

                # electron position update
                # random choice
                jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                cdf = jnp.cumsum(non_diagonal_move_probabilities)
                random_value = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
                k = jnp.searchsorted(cdf, random_value)
                r_up_carts = non_diagonal_move_mesh_r_up_carts[k]
                r_dn_carts = non_diagonal_move_mesh_r_dn_carts[k]

                # Walker carry stays unwrapped (in R^3), mirroring the
                # production GFMC_n kernel and ``_GFMC_t_debug``.

            latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key, latest_RT, latest_ecp_RT = (
                w_L,
                r_up_carts,
                r_dn_carts,
                jax_PRNG_key,
                R.T,
                ecp_R.T,
            )

            return (latest_w_L, latest_r_up_carts, latest_r_dn_carts, latest_jax_PRNG_key, latest_RT, latest_ecp_RT)

        def _compute_V_elements_n_debug(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            kinetic_RT: jnpt.ArrayLike,
            ecp_RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Compute V elements."""
            # coulomb regularization
            # compute diagonal elements, kinetic part
            diagonal_kinetic_part = 3.0 / alat**2 * (len(r_up_carts) + len(r_dn_carts))

            # compute continuum kinetic energy
            diagonal_kinetic_continuum_elements_up, diagonal_kinetic_continuum_elements_dn = (
                compute_kinetic_energy_all_elements(
                    wavefunction_data=hamiltonian_data.wavefunction_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute discretized kinetic energy and mesh (with a random rotation)
            _, _, elements_non_diagonal_kinetic_part = compute_discretized_kinetic_energy(
                alat=alat,
                wavefunction_data=hamiltonian_data.wavefunction_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                RT=kinetic_RT,
            )
            # spin-filp
            elements_non_diagonal_kinetic_part_FN = jnp.minimum(elements_non_diagonal_kinetic_part, 0.0)
            non_diagonal_sum_hamiltonian_kinetic = jnp.sum(elements_non_diagonal_kinetic_part_FN)
            diagonal_kinetic_part_SP = jnp.sum(jnp.maximum(elements_non_diagonal_kinetic_part, 0.0))
            # regularizations
            elements_non_diagonal_kinetic_part_all = elements_non_diagonal_kinetic_part.reshape(-1, 6)
            sign_flip_flags_elements = jnp.any(elements_non_diagonal_kinetic_part_all >= 0, axis=1)
            non_diagonal_kinetic_part_elements = jnp.sum(elements_non_diagonal_kinetic_part_all + 1.0 / (2.0 * alat**2), axis=1)
            sign_flip_flags_elements_up, sign_flip_flags_elements_dn = jnp.split(sign_flip_flags_elements, [len(r_up_carts)])
            non_diagonal_kinetic_part_elements_up, non_diagonal_kinetic_part_elements_dn = jnp.split(
                non_diagonal_kinetic_part_elements, [len(r_up_carts)]
            )

            # compute diagonal elements, el-el
            diagonal_bare_coulomb_part_el_el = compute_coulomb_potential_ee_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
            )

            # compute diagonal elements, ion-ion
            diagonal_bare_coulomb_part_ion_ion = compute_coulomb_potential_ii_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data
            )

            # compute diagonal elements, el-ion
            diagonal_bare_coulomb_part_el_ion_elements_up, diagonal_bare_coulomb_part_el_ion_elements_dn = (
                compute_coulomb_potential_ei_dispatch(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )

            # compute diagonal elements, el-ion, discretized
            (
                diagonal_bare_coulomb_part_el_ion_discretized_elements_up,
                diagonal_bare_coulomb_part_el_ion_discretized_elements_dn,
            ) = compute_coulomb_potential_ei_discretized_dispatch(
                coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                r_up_carts=r_up_carts,
                r_dn_carts=r_dn_carts,
                alat=alat,
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_up = (
                diagonal_bare_coulomb_part_el_ion_elements_up
                + diagonal_kinetic_continuum_elements_up
                - non_diagonal_kinetic_part_elements_up
            )
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_elements_up
            else:
                diagonal_bare_coulomb_part_el_ion_ei_up = diagonal_bare_coulomb_part_el_ion_discretized_elements_up
            diagonal_bare_coulomb_part_el_ion_max_up = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_up, diagonal_bare_coulomb_part_el_ion_ei_up
            )
            diagonal_bare_coulomb_part_el_ion_opt_up = jnp.where(
                sign_flip_flags_elements_up, diagonal_bare_coulomb_part_el_ion_max_up, diagonal_bare_coulomb_part_el_ion_zv_up
            )

            # compose discretized el-ion potentials
            diagonal_bare_coulomb_part_el_ion_zv_dn = (
                diagonal_bare_coulomb_part_el_ion_elements_dn
                + diagonal_kinetic_continuum_elements_dn
                - non_diagonal_kinetic_part_elements_dn
            )
            # The singularity of the bare coulomb potential is cut only for all-electron calculations. (c.f., parcutg=2 in TurboRVB).
            # The discretized bare coulomb potential is replaced with the standard bare coulomb potential without any truncation for effective core potential calculations. (c.f., parcutg=1 in TurboRVB).
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_elements_dn
            else:
                diagonal_bare_coulomb_part_el_ion_ei_dn = diagonal_bare_coulomb_part_el_ion_discretized_elements_dn
            diagonal_bare_coulomb_part_el_ion_max_dn = jnp.maximum(
                diagonal_bare_coulomb_part_el_ion_zv_dn, diagonal_bare_coulomb_part_el_ion_ei_dn
            )
            diagonal_bare_coulomb_part_el_ion_opt_dn = jnp.where(
                sign_flip_flags_elements_dn, diagonal_bare_coulomb_part_el_ion_max_dn, diagonal_bare_coulomb_part_el_ion_zv_dn
            )

            # final bare coulomb part
            discretized_diagonal_bare_coulomb_part = (
                diagonal_bare_coulomb_part_el_el
                + diagonal_bare_coulomb_part_ion_ion
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_up)
                + jnp.sum(diagonal_bare_coulomb_part_el_ion_opt_dn)
            )

            # with ECP
            if hamiltonian_data.coulomb_potential_data.ecp_flag:
                # ecp local
                diagonal_ecp_local_part = compute_ecp_local_parts_all_pairs(
                    coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )

                if non_local_move == "tmove":
                    # ecp non-local (t-move)
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=False,
                            RT=ecp_RT,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))
                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                elif non_local_move == "dltmove":
                    mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, _ = (
                        compute_ecp_non_local_parts_nearest_neighbors(
                            coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
                            wavefunction_data=hamiltonian_data.wavefunction_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts,
                            flag_determinant_only=True,
                            RT=ecp_RT,
                        )
                    )

                    V_nonlocal_FN = jnp.minimum(V_nonlocal, 0.0)
                    diagonal_ecp_part_SP = jnp.sum(jnp.maximum(V_nonlocal, 0.0))

                    Jastrow_ref = compute_Jastrow_part_ratio_zone(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Jastrow_on_mesh = jnp.stack(
                        [
                            compute_Jastrow_part_ratio_zone(
                                hamiltonian_data.wavefunction_data.jastrow_data,
                                mesh_non_local_ecp_part_r_up_carts[_i],
                                mesh_non_local_ecp_part_r_dn_carts[_i],
                            )
                            for _i in range(len(mesh_non_local_ecp_part_r_up_carts))
                        ]
                    )
                    Jastrow_ratio = jnp.exp(Jastrow_on_mesh - Jastrow_ref)
                    V_nonlocal_FN = V_nonlocal_FN * Jastrow_ratio

                    non_diagonal_sum_hamiltonian_ecp = jnp.sum(V_nonlocal_FN)
                    non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic + non_diagonal_sum_hamiltonian_ecp

                else:
                    raise NotImplementedError

                V_diag = (
                    diagonal_kinetic_part
                    + discretized_diagonal_bare_coulomb_part
                    + diagonal_ecp_local_part
                    + diagonal_kinetic_part_SP
                    + diagonal_ecp_part_SP
                )

                V_nondiag = non_diagonal_sum_hamiltonian

                # Measurement local energy (novar=0 split, cf. TurboRVB
                # enertrue with novar=0): continuum e_L from the per-electron
                # continuum kinetic + BARE el-ion potential, no lattice hop
                # terms, no sign-flip max clip. Equals V_diag + V_nondiag on
                # non-clipped configurations by the zero-variance identity.
                e_L_meas = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                    + diagonal_ecp_local_part
                    + diagonal_ecp_part_SP
                    + non_diagonal_sum_hamiltonian_ecp
                )

            # with all electrons
            else:
                non_diagonal_sum_hamiltonian = non_diagonal_sum_hamiltonian_kinetic

                V_diag = diagonal_kinetic_part + discretized_diagonal_bare_coulomb_part + diagonal_kinetic_part_SP

                V_nondiag = non_diagonal_sum_hamiltonian

                # Measurement local energy (novar=0 split) -- see ECP branch.
                e_L_meas = (
                    diagonal_bare_coulomb_part_el_el
                    + diagonal_bare_coulomb_part_ion_ion
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_up + diagonal_kinetic_continuum_elements_up)
                    + jnp.sum(diagonal_bare_coulomb_part_el_ion_elements_dn + diagonal_kinetic_continuum_elements_dn)
                )

            return (V_diag, V_nondiag, e_L_meas)

        def _compute_local_energy_n_debug(
            hamiltonian_data: Hamiltonian_data,
            r_up_carts: jnpt.ArrayLike,
            r_dn_carts: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            non_local_move: NonLocalMove,
            alat: float,
        ) -> jax.Array:
            _V_diag, _V_nondiag, e_L_meas = _compute_V_elements_n_debug(
                hamiltonian_data, r_up_carts, r_dn_carts, RT, RT, non_local_move, alat
            )
            # novar=0 split (debug twin): differentiate the MEASUREMENT e_L.
            return e_L_meas

        # MAIN MCMC loop from here !!!
        logger.info("Start GFMC")
        num_mcmc_done = 0
        mcmc_interval = int(np.maximum(num_mcmc_steps / 100, 1))

        for i_mcmc_step in range(num_mcmc_steps):
            if i_mcmc_step % mcmc_interval == 0:
                progress = (i_mcmc_step + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0

                logger.info(
                    f"  Progress: GFMC step = {i_mcmc_step + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %."
                )

            # Always set the initial weight list to 1.0
            w_L_list = jnp.array([1.0 for _ in range(self._num_walkers)], dtype=jnp.float64)

            logger.devel("  Projection is on going....")

            # projection loop
            _proj_results = [
                _projection_n_debug(
                    w_L_list[i],
                    self._latest_r_up_carts[i],
                    self._latest_r_dn_carts[i],
                    self._jax_PRNG_key_list[i],
                    self._E_scf,
                    self._num_mcmc_per_measurement,
                    self._non_local_move,
                    self._alat,
                    self._hamiltonian_data,
                )
                for i in range(self._num_walkers)
            ]
            w_L_list = jnp.stack([r[0] for r in _proj_results])
            self._latest_r_up_carts = jnp.stack([r[1] for r in _proj_results])
            self._latest_r_dn_carts = jnp.stack([r[2] for r in _proj_results])
            self._jax_PRNG_key_list = jnp.stack([r[3] for r in _proj_results])
            latest_RT = jnp.stack([r[4] for r in _proj_results])
            latest_ecp_RT = jnp.stack([r[5] for r in _proj_results])

            # projection ends
            logger.devel("  Projection ends.")

            # evaluate observables
            # V_diag and e_L
            _V_results = [
                _compute_V_elements_n_debug(
                    self._hamiltonian_data,
                    self._latest_r_up_carts[i],
                    self._latest_r_dn_carts[i],
                    latest_RT[i],
                    latest_ecp_RT[i],
                    self._non_local_move,
                    self._alat,
                )
                for i in range(self._num_walkers)
            ]
            V_diag_list = jnp.stack([r[0] for r in _V_results])
            V_nondiag_list = jnp.stack([r[1] for r in _V_results])
            e_L_meas_list = jnp.stack([r[2] for r in _V_results])
            # novar=0 split (debug twin): measurement numerator is the
            # continuum e_L; V_diag_list keeps the propagation composition
            # for the 1/(V_diag - E_scf) weighting.
            e_L_list = e_L_meas_list

            if self._non_local_move == "tmove":
                # The LRDMC measurement retains complex kinetic energy but
                # localizes ECP with real candidate weights. Continuum VMC
                # retains Im(Vnl), so it is not the same complex observable.
                # For tmove, max(Re V,0)+min(Re V,0)=Re V candidate by candidate.
                def _e_L_check_fn(h, ru, rd, rotation):
                    t_up, t_dn = compute_kinetic_energy_all_elements(h.wavefunction_data, ru, rd)
                    potential = compute_coulomb_potential(
                        h.coulomb_potential_data, ru, rd, rotation, wavefunction_data=h.wavefunction_data
                    )
                    return jnp.sum(t_up) + jnp.sum(t_dn) + jnp.real(potential)

                e_list_debug = jnp.stack(
                    [
                        _e_L_check_fn(
                            self._hamiltonian_data,
                            self._latest_r_up_carts[i],
                            self._latest_r_dn_carts[i],
                            latest_ecp_RT[i],
                        )
                        for i in range(self._num_walkers)
                    ]
                )
                # e_L crosses ao_eval/jastrow_eval/det_eval/coulomb/
                # local_energy; bound the agreement by the weakest zone (fp32 in
                # mixed precision).
                _atol_eL, _rtol_eL = get_tolerance_min(
                    ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "local_energy"),
                    "strict",
                )
                if np.max(np.abs(e_L_list - e_list_debug)) > _rtol_eL:
                    logger.info(f"max(e_list - e_list_debug) = {np.max(np.abs(e_L_list - e_list_debug))}.")
                    logger.info(f"w_L_list = {w_L_list}.")
                    logger.info(f"e_L_list = {e_L_list}.")
                    logger.info(f"e_list_debug = {e_list_debug}.")
                np.testing.assert_allclose(np.array(e_L_list), np.array(e_list_debug), atol=_atol_eL, rtol=_rtol_eL)

            # atomic force related
            if self._comput_position_deriv:
                # The e_L Hellmann-Feynman force uses the same derivative backend
                # flag as production. When ``_ANALYTIC_GRAD_ENABLED``, route through the verified
                # analytic continuum-measurement kernel
                # ``gfmc.grad_e_L_n_wrt_positions`` (RT = latest_ecp_RT[i] like the debug forward;
                # complex-twist DTN included, ``_assert_scope`` rejects nothing).
                # Otherwise (flag off) fall back to the LRDMC lattice-discretised
                # ``jax.grad`` oracle (hand-summed structure grads).
                if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
                    from .derivatives import gfmc as _gfmc_deriv

                    _hf_results = [
                        _gfmc_deriv.grad_e_L_n_wrt_positions(
                            self._hamiltonian_data,
                            self._latest_r_up_carts[i],
                            self._latest_r_dn_carts[i],
                            latest_ecp_RT[i],
                            self._alat,
                            non_local_move=self._non_local_move,
                        )
                        for i in range(self._num_walkers)
                    ]
                    grad_e_L_R = jnp.stack([r[0] for r in _hf_results])
                    grad_e_L_r_up = jnp.stack([r[1] for r in _hf_results])
                    grad_e_L_r_dn = jnp.stack([r[2] for r in _hf_results])
                else:
                    # Project complex Bloch local energy to its physical real component.
                    # The physical force differentiates Re(e_L).
                    def _real_compute_local_energy_n_debug(*args: Any, **kwargs: Any) -> jax.Array:
                        return jnp.real(_compute_local_energy_n_debug(*args, **kwargs))

                    _grad_e_L_fn = grad(_real_compute_local_energy_n_debug, argnums=(0, 1, 2))
                    _grad_e_L_results = [
                        _grad_e_L_fn(
                            self._hamiltonian_data,
                            self._latest_r_up_carts[i],
                            self._latest_r_dn_carts[i],
                            latest_ecp_RT[i],
                            self._non_local_move,
                            self._alat,
                        )
                        for i in range(self._num_walkers)
                    ]
                    grad_e_L_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *[r[0] for r in _grad_e_L_results])
                    grad_e_L_r_up = jnp.stack([r[1] for r in _grad_e_L_results])
                    grad_e_L_r_dn = jnp.stack([r[2] for r in _grad_e_L_results])

                    grad_e_L_R = (
                        grad_e_L_h.wavefunction_data.geminal_data.orb_data_up_spin.structure_data.positions
                        + grad_e_L_h.wavefunction_data.geminal_data.orb_data_dn_spin.structure_data.positions
                        + grad_e_L_h.coulomb_potential_data.structure_data.positions
                    )

                    # Which Jastrow channels carry geometry, and where the leaf
                    # sits inside each, is Jastrow-internal knowledge -- the
                    # driver only asks for their summed contribution
                    # so this driver remains independent of channel membership.
                    _jastrow_R = structure_positions_grad_Jastrow(
                        self._hamiltonian_data.wavefunction_data.jastrow_data,
                        grad_e_L_h.wavefunction_data.jastrow_data,
                    )
                    if _jastrow_R is not None:
                        grad_e_L_R += _jastrow_R

                # dln|Psi|/dr (drift) and dln|Psi|/dR
                # (Pulay) use the derivatives interface, looped per walker, with
                # the ``_ANALYTIC_GRAD_ENABLED`` flag (same backend as production); ln|Psi| is
                # potential-free -> LRDMC/non_local_move-independent (no carve-out). The "R" axis
                # returns the per-atom force already summed by ``accumulate_position_grad``.
                _grad_ln_psi_results = [
                    derivatives.compute_grad_ln_Psi(
                        self._hamiltonian_data,
                        self._latest_r_up_carts[i],
                        self._latest_r_dn_carts[i],
                        _pos_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                    )
                    for i in range(self._num_walkers)
                ]
                grad_ln_Psi_r_up = jnp.stack([r["r"][0] for r in _grad_ln_psi_results])
                grad_ln_Psi_r_dn = jnp.stack([r["r"][1] for r in _grad_ln_psi_results])
                grad_ln_Psi_dR = jnp.stack([r["R"] for r in _grad_ln_psi_results])

                if self._use_swct:
                    omega_up = jnp.stack(
                        [
                            evaluate_swct_omega(self._hamiltonian_data.structure_data, self._latest_r_up_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    omega_dn = jnp.stack(
                        [
                            evaluate_swct_omega(self._hamiltonian_data.structure_data, self._latest_r_dn_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    grad_omega_dr_up = jnp.stack(
                        [
                            evaluate_swct_domega(self._hamiltonian_data.structure_data, self._latest_r_up_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                    grad_omega_dr_dn = jnp.stack(
                        [
                            evaluate_swct_domega(self._hamiltonian_data.structure_data, self._latest_r_dn_carts[i])
                            for i in range(self._num_walkers)
                        ]
                    )
                else:
                    _n_up = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
                    _n_dn = self._hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
                    _n_atoms = self._hamiltonian_data.structure_data.natom
                    omega_up = jnp.zeros((self._num_walkers, _n_atoms, _n_up), dtype=jnp.float64)
                    omega_dn = jnp.zeros((self._num_walkers, _n_atoms, _n_dn), dtype=jnp.float64)
                    grad_omega_dr_up = jnp.zeros((self._num_walkers, _n_atoms, 3), dtype=jnp.float64)
                    grad_omega_dr_dn = jnp.zeros((self._num_walkers, _n_atoms, 3), dtype=jnp.float64)

            # jnp.array -> np.array
            w_L_latest = np.array(w_L_list)
            e_L_latest = np.array(e_L_list)
            # Under complex Bloch fixed-phase, take Re of the diagonal
            # H expectation so the branching weight ratio ``w_L / V_diag_E``
            # stays real. Fixed-phase branching discards the imaginary phase
            # component. Bit-identical at Gamma where V_diag_list is real.
            V_diag_E_latest = np.real(np.array(V_diag_list)) - self._E_scf

            if self._comput_position_deriv:
                grad_e_L_r_up_latest = np.array(grad_e_L_r_up)
                grad_e_L_r_dn_latest = np.array(grad_e_L_r_dn)
                grad_e_L_R_latest = np.array(grad_e_L_R)
                grad_ln_Psi_r_up_latest = np.array(grad_ln_Psi_r_up)
                grad_ln_Psi_r_dn_latest = np.array(grad_ln_Psi_r_dn)
                grad_ln_Psi_dR_latest = np.array(grad_ln_Psi_dR)
                omega_up_latest = np.array(omega_up)
                omega_dn_latest = np.array(omega_dn)
                grad_omega_dr_up_latest = np.array(grad_omega_dr_up)
                grad_omega_dr_dn_latest = np.array(grad_omega_dr_dn)

                # Debug per-walker Hellmann-Feynman and Pulay stress.
                # RT = latest_ecp_RT[i] matches the debug force convention above.
                # Periodic only. The HF stress uses the
                # ``_ANALYTIC_GRAD_ENABLED`` flag (analytic continuum-measurement kernel
                # ``gfmc.grad_e_L_n_wrt_strain`` when enabled, else the local LRDMC
                # jacrev fallback); the Pulay stress "eps" axis follows the same flag.
                if self._hamiltonian_data.structure_data.pbc_flag:
                    _zero_eps = jnp.zeros((3, 3), dtype=jnp.float64)

                    if _analytic_hf_enabled_for(self._hamiltonian_data, self._non_local_move):
                        from .derivatives import gfmc as _gfmc_deriv

                        _stress_HF_latest = np.array(
                            [
                                np.asarray(
                                    _gfmc_deriv.grad_e_L_n_wrt_strain(
                                        self._hamiltonian_data,
                                        self._latest_r_up_carts[i],
                                        self._latest_r_dn_carts[i],
                                        latest_ecp_RT[i],
                                        self._alat,
                                        non_local_move=self._non_local_move,
                                    )
                                )
                                for i in range(self._num_walkers)
                            ]
                        )
                    else:

                        def _real_e_L_n_strained(
                            hamiltonian_data: Hamiltonian_data,
                            r_up_carts: jax.Array,
                            r_dn_carts: jax.Array,
                            RT: jax.Array,
                            non_local_move: NonLocalMove,
                            alat: float,
                            strain: jax.Array,
                        ) -> jax.Array:
                            F = jnp.eye(3, dtype=jnp.float64) + strain
                            new_h = hamiltonian_data._apply_strain(F)
                            return jnp.real(
                                _compute_local_energy_n_debug(
                                    new_h, r_up_carts @ F.T, r_dn_carts @ F.T, RT, non_local_move, alat
                                )
                            )

                        _stress_HF_fn = jax.jacrev(_real_e_L_n_strained, argnums=6)
                        _stress_HF_latest = np.array(
                            [
                                np.asarray(
                                    _stress_HF_fn(
                                        self._hamiltonian_data,
                                        self._latest_r_up_carts[i],
                                        self._latest_r_dn_carts[i],
                                        latest_ecp_RT[i],
                                        self._non_local_move,
                                        self._alat,
                                        _zero_eps,
                                    )
                                )
                                for i in range(self._num_walkers)
                            ]
                        )
                    _stress_PP_latest = np.array(
                        [
                            np.asarray(
                                derivatives.compute_grad_ln_Psi(
                                    self._hamiltonian_data,
                                    self._latest_r_up_carts[i],
                                    self._latest_r_dn_carts[i],
                                    _eps_flags,
                                    analytic=_ANALYTIC_GRAD_ENABLED,
                                )["eps"]
                            )
                            for i in range(self._num_walkers)
                        ]
                    )
                else:
                    _stress_HF_latest = np.zeros((self._num_walkers, 3, 3), dtype=np.float64)
                    _stress_PP_latest = np.zeros((self._num_walkers, 3, 3), dtype=np.float64)

            # jnp.array -> np.array
            latest_r_up_carts_before_branching_old = np.array(self._latest_r_up_carts)
            latest_r_dn_carts_before_branching_old = np.array(self._latest_r_dn_carts)

            # MPI reduce
            r_up_carts_shape = latest_r_up_carts_before_branching_old.shape
            r_up_carts_gathered_dyad = (self._mpi_rank, latest_r_up_carts_before_branching_old)
            r_dn_carts_shape = latest_r_dn_carts_before_branching_old.shape
            r_dn_carts_gathered_dyad = (self._mpi_rank, latest_r_dn_carts_before_branching_old)

            r_up_carts_gathered_dyad = self._mpi_comm.gather(r_up_carts_gathered_dyad, root=0)
            r_dn_carts_gathered_dyad = self._mpi_comm.gather(r_dn_carts_gathered_dyad, root=0)

            # MPI reduce
            e_L_gathered_dyad = (self._mpi_rank, e_L_latest)
            w_L_gathered_dyad = (self._mpi_rank, w_L_latest)
            V_diag_E_gathered_dyad = (self._mpi_rank, V_diag_E_latest)
            if self._comput_position_deriv:
                grad_e_L_r_up_dyad = (self._mpi_rank, grad_e_L_r_up_latest)
                grad_e_L_r_dn_dyad = (self._mpi_rank, grad_e_L_r_dn_latest)
                grad_e_L_R_dyad = (self._mpi_rank, grad_e_L_R_latest)
                grad_ln_Psi_r_up_dyad = (self._mpi_rank, grad_ln_Psi_r_up_latest)
                grad_ln_Psi_r_dn_dyad = (self._mpi_rank, grad_ln_Psi_r_dn_latest)
                grad_ln_Psi_dR_dyad = (self._mpi_rank, grad_ln_Psi_dR_latest)
                omega_up_dyad = (self._mpi_rank, omega_up_latest)
                omega_dn_dyad = (self._mpi_rank, omega_dn_latest)
                grad_omega_dr_up_dyad = (self._mpi_rank, grad_omega_dr_up_latest)
                grad_omega_dr_dn_dyad = (self._mpi_rank, grad_omega_dr_dn_latest)
                stress_HF_dyad = (self._mpi_rank, _stress_HF_latest)
                stress_PP_dyad = (self._mpi_rank, _stress_PP_latest)

            e_L_gathered_dyad = self._mpi_comm.gather(e_L_gathered_dyad, root=0)
            w_L_gathered_dyad = self._mpi_comm.gather(w_L_gathered_dyad, root=0)
            V_diag_E_gathered_dyad = self._mpi_comm.gather(V_diag_E_gathered_dyad, root=0)
            if self._comput_position_deriv:
                grad_e_L_r_up_dyad = self._mpi_comm.gather(grad_e_L_r_up_dyad, root=0)
                grad_e_L_r_dn_dyad = self._mpi_comm.gather(grad_e_L_r_dn_dyad, root=0)
                grad_e_L_R_dyad = self._mpi_comm.gather(grad_e_L_R_dyad, root=0)
                grad_ln_Psi_r_up_dyad = self._mpi_comm.gather(grad_ln_Psi_r_up_dyad, root=0)
                grad_ln_Psi_r_dn_dyad = self._mpi_comm.gather(grad_ln_Psi_r_dn_dyad, root=0)
                grad_ln_Psi_dR_dyad = self._mpi_comm.gather(grad_ln_Psi_dR_dyad, root=0)
                omega_up_dyad = self._mpi_comm.gather(omega_up_dyad, root=0)
                omega_dn_dyad = self._mpi_comm.gather(omega_dn_dyad, root=0)
                grad_omega_dr_up_dyad = self._mpi_comm.gather(grad_omega_dr_up_dyad, root=0)
                grad_omega_dr_dn_dyad = self._mpi_comm.gather(grad_omega_dr_dn_dyad, root=0)
                stress_HF_dyad = self._mpi_comm.gather(stress_HF_dyad, root=0)
                stress_PP_dyad = self._mpi_comm.gather(stress_PP_dyad, root=0)

            if self._mpi_rank == 0:
                # dict
                r_up_carts_gathered_dict = dict(r_up_carts_gathered_dyad)
                r_dn_carts_gathered_dict = dict(r_dn_carts_gathered_dyad)
                e_L_gathered_dict = dict(e_L_gathered_dyad)
                w_L_gathered_dict = dict(w_L_gathered_dyad)
                V_diag_E_gathered_dict = dict(V_diag_E_gathered_dyad)
                if self._comput_position_deriv:
                    grad_e_L_r_up_gathered_dict = dict(grad_e_L_r_up_dyad)
                    grad_e_L_r_dn_gathered_dict = dict(grad_e_L_r_dn_dyad)
                    grad_e_L_R_gathered_dict = dict(grad_e_L_R_dyad)
                    grad_ln_Psi_r_up_gathered_dict = dict(grad_ln_Psi_r_up_dyad)
                    grad_ln_Psi_r_dn_gathered_dict = dict(grad_ln_Psi_r_dn_dyad)
                    grad_ln_Psi_dR_gathered_dict = dict(grad_ln_Psi_dR_dyad)
                    omega_up_gathered_dict = dict(omega_up_dyad)
                    omega_dn_gathered_dict = dict(omega_dn_dyad)
                    grad_omega_dr_up_gathered_dict = dict(grad_omega_dr_up_dyad)
                    grad_omega_dr_dn_gathered_dict = dict(grad_omega_dr_dn_dyad)
                # gathered
                r_up_carts_gathered = np.concatenate([r_up_carts_gathered_dict[i] for i in range(self._mpi_size)])
                r_dn_carts_gathered = np.concatenate([r_dn_carts_gathered_dict[i] for i in range(self._mpi_size)])
                e_L_gathered = np.concatenate([e_L_gathered_dict[i] for i in range(self._mpi_size)])
                w_L_gathered = np.concatenate([w_L_gathered_dict[i] for i in range(self._mpi_size)])
                V_diag_E_gathered = np.concatenate([V_diag_E_gathered_dict[i] for i in range(self._mpi_size)])
                if self._comput_position_deriv:
                    grad_e_L_r_up_gathered = np.concatenate([grad_e_L_r_up_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_e_L_r_dn_gathered = np.concatenate([grad_e_L_r_dn_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_e_L_R_gathered = np.concatenate([grad_e_L_R_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_ln_Psi_r_up_gathered = np.concatenate(
                        [grad_ln_Psi_r_up_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_ln_Psi_r_dn_gathered = np.concatenate(
                        [grad_ln_Psi_r_dn_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_ln_Psi_dR_gathered = np.concatenate([grad_ln_Psi_dR_gathered_dict[i] for i in range(self._mpi_size)])
                    omega_up_gathered = np.concatenate([omega_up_gathered_dict[i] for i in range(self._mpi_size)])
                    omega_dn_gathered = np.concatenate([omega_dn_gathered_dict[i] for i in range(self._mpi_size)])
                    grad_omega_dr_up_gathered = np.concatenate(
                        [grad_omega_dr_up_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                    grad_omega_dr_dn_gathered = np.concatenate(
                        [grad_omega_dr_dn_gathered_dict[i] for i in range(self._mpi_size)]
                    )
                # sum
                w_L_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered)
                e_L_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered * e_L_gathered)
                # Mirror the GFMC_n production path:
                # (|e_L|^2 buffer). See the matching note at the production site.
                e_L2_weighted_sum = np.sum(w_L_gathered / V_diag_E_gathered * np.abs(e_L_gathered) ** 2)
                if self._comput_position_deriv:
                    # Pathak-Wagner regularization factor f_eps per walker
                    if self._epsilon_PW > 0.0:
                        grad_norm_sq = np.sum(grad_ln_Psi_r_up_gathered**2, axis=(1, 2)) + np.sum(
                            grad_ln_Psi_r_dn_gathered**2, axis=(1, 2)
                        )
                        nodal_dist = 1.0 / np.sqrt(grad_norm_sq)
                        t = nodal_dist / self._epsilon_PW
                        t2 = t * t
                        t4 = t2 * t2
                        t6 = t4 * t2
                        f_eps = np.where(t < 1.0, 7.0 * t6 - 15.0 * t4 + 9.0 * t2, 1.0)
                        mod_w_L_gathered = w_L_gathered * f_eps
                    else:
                        mod_w_L_gathered = w_L_gathered

                    # compute per-walker force products
                    force_HF_per_walker = (
                        grad_e_L_R_gathered
                        + np.einsum("ijk,ikl->ijl", omega_up_gathered, grad_e_L_r_up_gathered)
                        + np.einsum("ijk,ikl->ijl", omega_dn_gathered, grad_e_L_r_dn_gathered)
                    )
                    force_PP_per_walker = (
                        grad_ln_Psi_dR_gathered
                        + np.einsum("ijk,ikl->ijl", omega_up_gathered, grad_ln_Psi_r_up_gathered)
                        + np.einsum("ijk,ikl->ijl", omega_dn_gathered, grad_ln_Psi_r_dn_gathered)
                        + 0.5 * (grad_omega_dr_up_gathered + grad_omega_dr_dn_gathered)
                    )
                    E_L_force_PP_per_walker = e_L_gathered[:, None, None] * force_PP_per_walker
                    # weighted sum (using mod_w_L / V_diag_E)
                    force_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, force_HF_per_walker)
                    force_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, force_PP_per_walker)
                    E_L_force_PP_weighted_sum = np.einsum(
                        "i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, E_L_force_PP_per_walker
                    )
                    # averaged
                    force_HF_averaged = force_HF_weighted_sum / w_L_weighted_sum
                    force_PP_averaged = force_PP_weighted_sum / w_L_weighted_sum
                    E_L_force_PP_averaged = E_L_force_PP_weighted_sum / w_L_weighted_sum
                    # add a dummy dim
                    force_HF_averaged = np.expand_dims(force_HF_averaged, axis=0)
                    force_PP_averaged = np.expand_dims(force_PP_averaged, axis=0)
                    E_L_force_PP_averaged = np.expand_dims(E_L_force_PP_averaged, axis=0)
                    # store
                    self._stored_force_HF.append(force_HF_averaged)
                    self._stored_force_PP.append(force_PP_averaged)
                    self._stored_E_L_force_PP.append(E_L_force_PP_averaged)

                    # stress: reconstruct gathered per-walker tensors and weight-
                    # average with the GFMC_n branching weight (mod_w_L / V_diag_E,
                    # normalised by w_L_weighted_sum -- same as the force block).
                    stress_HF_gathered = np.concatenate([dict(stress_HF_dyad)[i] for i in range(self._mpi_size)])
                    stress_PP_gathered = np.concatenate([dict(stress_PP_dyad)[i] for i in range(self._mpi_size)])
                    E_L_stress_PP_per_walker = e_L_gathered[:, None, None] * stress_PP_gathered
                    stress_HF_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, stress_HF_gathered)
                    stress_PP_weighted_sum = np.einsum("i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, stress_PP_gathered)
                    E_L_stress_PP_weighted_sum = np.einsum(
                        "i,ijk->jk", mod_w_L_gathered / V_diag_E_gathered, E_L_stress_PP_per_walker
                    )
                    self._stored_stress_HF.append(np.expand_dims(stress_HF_weighted_sum / w_L_weighted_sum, axis=0))
                    self._stored_stress_PP.append(np.expand_dims(stress_PP_weighted_sum / w_L_weighted_sum, axis=0))
                    self._stored_E_L_stress_PP.append(np.expand_dims(E_L_stress_PP_weighted_sum / w_L_weighted_sum, axis=0))
                # averaged
                w_L_averaged = np.average(w_L_gathered)
                e_L_averaged = e_L_weighted_sum / w_L_weighted_sum
                e_L2_averaged = e_L2_weighted_sum / w_L_weighted_sum
                # add a dummy dim
                e_L2_averaged = np.expand_dims(e_L2_averaged, axis=0)
                e_L_averaged = np.expand_dims(e_L_averaged, axis=0)
                w_L_averaged = np.expand_dims(w_L_averaged, axis=0)
                # store  # This should stored only for MPI-rank = 0 !!!
                self._stored_e_L2.append(e_L2_averaged)
                self._stored_e_L.append(e_L_averaged)
                self._stored_w_L.append(w_L_averaged)
                # Start branching
                logger.devel(f"w_L_gathered = {w_L_gathered}")
                probabilities = w_L_gathered / w_L_gathered.sum()
                logger.devel(f"probabilities = {probabilities}")

                # correlated choice (see F. Becca and S. Sorella, Quantum Monte Carlo Approaches for Correlated Systems (2017), p. 182)
                zeta = float(self._branching_rng.random())
                z_list = [(alpha + zeta) / len(probabilities) for alpha in range(len(probabilities))]
                logger.devel(f"z_list = {z_list}")
                cumulative_prob = np.cumsum(probabilities)
                chosen_walker_indices_old = np.array(
                    [next(idx for idx, prob in enumerate(cumulative_prob) if z <= prob) for z in z_list]
                )
                proposed_r_up_carts = r_up_carts_gathered[chosen_walker_indices_old]
                proposed_r_dn_carts = r_dn_carts_gathered[chosen_walker_indices_old]

                num_survived_walkers = len(set(chosen_walker_indices_old))
                num_killed_walkers = len(w_L_gathered) - len(set(chosen_walker_indices_old))
            else:
                num_survived_walkers = None
                num_killed_walkers = None
                proposed_r_up_carts = None
                proposed_r_dn_carts = None

            num_survived_walkers = self._mpi_comm.bcast(num_survived_walkers, root=0)
            num_killed_walkers = self._mpi_comm.bcast(num_killed_walkers, root=0)

            proposed_r_up_carts = self._mpi_comm.bcast(proposed_r_up_carts, root=0)
            proposed_r_dn_carts = self._mpi_comm.bcast(proposed_r_dn_carts, root=0)

            proposed_r_up_carts = proposed_r_up_carts.reshape(
                self._mpi_size, r_up_carts_shape[0], r_up_carts_shape[1], r_up_carts_shape[2]
            )
            proposed_r_dn_carts = proposed_r_dn_carts.reshape(
                self._mpi_size, r_dn_carts_shape[0], r_dn_carts_shape[1], r_dn_carts_shape[2]
            )

            # set new r_up_carts and r_dn_carts, and, np.array -> jnp.array
            latest_r_up_carts_after_branching = proposed_r_up_carts[self._mpi_rank, :, :, :]
            latest_r_dn_carts_after_branching = proposed_r_dn_carts[self._mpi_rank, :, :, :]

            # here update the walker positions!!
            self._num_survived_walkers += num_survived_walkers
            self._num_killed_walkers += num_killed_walkers
            self._latest_r_up_carts = jnp.asarray(latest_r_up_carts_after_branching, dtype=jnp.float64)
            self._latest_r_dn_carts = jnp.asarray(latest_r_dn_carts_after_branching, dtype=jnp.float64)

            # update E_scf
            num_gfmc_collect_steps = GFMC_ON_THE_FLY_COLLECT_STEPS
            num_gfmc_bin_blocks = GFMC_ON_THE_FLY_BIN_BLOCKS
            chain_step = self._mcmc_counter + i_mcmc_step

            # E_scf update schedule:
            #   - rapid phase (chain_step < mcmc_interval = N/100): update every step
            #   - thereafter:                                        update every mcmc_interval steps
            #   - skip when there are not yet enough G_L samples for jackknife
            n_G_L = max(0, chain_step - num_gfmc_collect_steps + 1)
            have_enough = n_G_L >= num_gfmc_bin_blocks
            in_rapid_phase = chain_step < mcmc_interval
            on_throttle = (chain_step + 1) % mcmc_interval == 0

            if have_enough and (in_rapid_phase or on_throttle):
                num_gfmc_warmup_steps = min(
                    num_gfmc_collect_steps + num_gfmc_bin_blocks,
                    max(n_G_L - num_gfmc_bin_blocks, 0),
                )
                self._E_scf, E_scf_std = self.get_E_on_the_fly(
                    num_gfmc_warmup_steps=num_gfmc_warmup_steps,
                    num_gfmc_bin_blocks=num_gfmc_bin_blocks,
                    num_gfmc_collect_steps=num_gfmc_collect_steps,
                )
                logger.debug(f"    E_scf = {self._E_scf:.5f} +- {E_scf_std:.5f} Ha.")

            # count up, here is the end of the branching step.
            num_mcmc_done += 1

        progress = (num_mcmc_done + self._mcmc_counter) / (num_mcmc_steps + self._mcmc_counter) * 100.0
        logger.info(
            f"  Progress: GFMC step = {num_mcmc_done + self._mcmc_counter}/{num_mcmc_steps + self._mcmc_counter}: {progress:.1f} %."
        )
        logger.info("")

        # count up mcmc_counter
        self._mcmc_counter += num_mcmc_done

    def get_E_on_the_fly(
        self, num_gfmc_warmup_steps: int = 3, num_gfmc_bin_blocks: int = 10, num_gfmc_collect_steps: int = 2
    ) -> float:
        """Get e_L."""
        logger.devel("- Comput. e_L -")
        if self._mpi_rank == 0:
            e_L_eq = self._stored_e_L[num_gfmc_warmup_steps + num_gfmc_collect_steps :]
            w_L_eq = self._stored_w_L[num_gfmc_warmup_steps:]
            logger.devel("  Progress: Computing G_eq and G_e_L_eq.")

            # For complex Bloch, stored_e_L is a Python list of
            # complex (or real) per-step averages; convert to array first,
            # then take Re explicitly under the fixed-phase convention, so
            # the on-the-fly E_scf used for
            # the GFMC_n branching shift is real-valued. No-op at Gamma.
            w_L_eq = jnp.asarray(w_L_eq, dtype=jnp.float64)
            e_L_eq = jnp.asarray(jnp.real(jnp.asarray(e_L_eq)), dtype=jnp.float64)
            G_eq = _compute_G_L_debug(w_L_eq, num_gfmc_collect_steps)
            G_e_L_eq = e_L_eq * G_eq
            G_eq = np.array(G_eq)
            G_e_L_eq = np.array(G_e_L_eq)

            logger.devel(f"  Progress: Computing binned G_e_L_eq and G_eq with # binned blocks = {num_gfmc_bin_blocks}.")
            G_e_L_split = np.array_split(G_e_L_eq, num_gfmc_bin_blocks)
            G_e_L_binned = np.array([np.sum(G_e_L_list) for G_e_L_list in G_e_L_split])
            G_split = np.array_split(G_eq, num_gfmc_bin_blocks)
            G_binned = np.array([np.sum(G_list) for G_list in G_split])

            logger.devel(f"  Progress: Computing jackknife samples with # binned blocks = {num_gfmc_bin_blocks}.")

            G_e_L_binned_sum = np.sum(G_e_L_binned)
            G_binned_sum = np.sum(G_binned)

            E_jackknife = [
                (G_e_L_binned_sum - G_e_L_binned[m]) / (G_binned_sum - G_binned[m]) for m in range(num_gfmc_bin_blocks)
            ]

            logger.devel("  Progress: Computing jackknife mean and std.")
            E_mean = np.average(E_jackknife)
            E_std = np.sqrt(num_gfmc_bin_blocks - 1) * np.std(E_jackknife)
            E_mean = float(E_mean)
            E_std = float(E_std)
        else:
            E_mean = None
            E_std = None

        E_mean = self._mpi_comm.bcast(E_mean, root=0)
        E_std = self._mpi_comm.bcast(E_std, root=0)

        return E_mean, E_std

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Return the mean and std of the computed local energy.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[float, float, float, float]:
                The mean and std values of the totat energy and those of the variance
                estimated by the Jackknife method with the Args. (E_mean, E_std, Var_mean, Var_std).
        """
        if self._mpi_rank == 0:
            e_L = self.e_L[num_mcmc_warmup_steps:]
            e_L2 = self.e_L2[num_mcmc_warmup_steps:]
            w_L = self._retained_w_L(num_mcmc_warmup_steps)
            # Shape assertions: (M, nw) from mcmc_counter and num_walkers
            M = self.mcmc_counter - num_mcmc_warmup_steps
            nw = 1  # GFMC observables are reduced across walkers
            _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
            _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
            _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
            w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
            w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

            w_L_binned_local = w_L_binned
            w_L_e_L_binned_local = w_L_e_L_binned
            w_L_e_L2_binned_local = w_L_e_L2_binned

            w_L_binned_local = np.array(w_L_binned_local)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
            w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_e_L2_binned_global_sum = np.sum(w_L_e_L2_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            # Mirror production get_E and take Re of
            # the per-binned-sample ratio so the energy observable / variance
            # are real on the complex Bloch path; the variance subtracts the
            # modulus square ``|<e_L>|**2``. w_L weights are real already
            # (fixed-phase). Bit-identical at Gamma.
            E_jackknife_binned_complex = np.array(
                [
                    (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                    for m in range(M_local)
                ]
            )
            abs_E_sq_local = np.abs(E_jackknife_binned_complex) ** 2
            E_jackknife_binned_local = list(np.real(E_jackknife_binned_complex))

            E2_jackknife_binned_local = list(
                np.real(
                    np.array(
                        [
                            (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m])
                            / (w_L_binned_global_sum - w_L_binned_local[m])
                            for m in range(M_local)
                        ]
                    )
                )
            )

            Var_jackknife_binned_local = list(np.array(E2_jackknife_binned_local) - abs_E_sq_local)

            # MPI allreduce
            E_jackknife_binned = E_jackknife_binned_local
            Var_jackknife_binned = Var_jackknife_binned_local
            M_total = len(E_jackknife_binned)
            logger.debug(f"The number of total binned samples = {M_total}")

            # jackknife mean and std
            E_mean = np.average(E_jackknife_binned)
            E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
            Var_mean = np.average(Var_jackknife_binned)
            Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

            logger.info(f"E = {E_mean} +- {E_std} Ha.")
            logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        else:
            E_mean = None
            E_std = None
            Var_mean = None
            Var_std = None

        # broadcast the results
        E_mean = self._mpi_comm.bcast(E_mean, root=0)
        E_std = self._mpi_comm.bcast(E_std, root=0)
        Var_mean = self._mpi_comm.bcast(Var_mean, root=0)
        Var_std = self._mpi_comm.bcast(Var_std, root=0)

        # return the results
        return (E_mean, E_std, Var_mean, Var_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Return the mean and std of the computed atomic forces.

        Args:
            num_mcmc_warmup_steps (int): the number of warmup steps.
            num_mcmc_bin_blocks (int): the number of binning blocks

        Return:
            tuple[npt.NDArray, npt.NDArray]:
                The mean and std values of the computed atomic forces
                estimated by the Jackknife method with the Args.
                The dimention of the arrays is (N, 3).
        """
        if self._mpi_rank == 0:
            w_L = self._retained_w_L(num_mcmc_warmup_steps)
            e_L = self.e_L[num_mcmc_warmup_steps:]
            force_HF = self.force_HF[num_mcmc_warmup_steps:]
            force_PP = self.force_PP[num_mcmc_warmup_steps:]
            E_L_force_PP = self.E_L_force_PP[num_mcmc_warmup_steps:]
            # Shape assertions: (M, nw) from mcmc_counter and num_walkers
            M = self.mcmc_counter - num_mcmc_warmup_steps
            nw = 1  # GFMC observables are reduced across walkers
            n_atoms = self.hamiltonian_data.structure_data.natom
            _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
            _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
            _require(force_HF.shape == (M, nw, n_atoms, 3), f"force_HF shape {force_HF.shape} != ({M}, {nw}, {n_atoms}, 3)")
            _require(force_PP.shape == (M, nw, n_atoms, 3), f"force_PP shape {force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)")
            _require(
                E_L_force_PP.shape == (M, nw, n_atoms, 3),
                f"E_L_force_PP shape {E_L_force_PP.shape} != ({M}, {nw}, {n_atoms}, 3)",
            )

            # split and binning with multiple walkers
            w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
            w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
            w_L_force_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_HF), num_mcmc_bin_blocks, axis=0)
            w_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, force_PP), num_mcmc_bin_blocks, axis=0)
            w_L_E_L_force_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_force_PP), num_mcmc_bin_blocks, axis=0)

            # binned sum
            w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
            w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

            w_L_force_HF_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_HF_split])
            w_L_force_HF_binned_shape = (
                w_L_force_HF_sum.shape[0] * w_L_force_HF_sum.shape[1],
                w_L_force_HF_sum.shape[2],
                w_L_force_HF_sum.shape[3],
            )
            w_L_force_HF_binned = list(w_L_force_HF_sum.reshape(w_L_force_HF_binned_shape))

            w_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_force_PP_split])
            w_L_force_PP_binned_shape = (
                w_L_force_PP_sum.shape[0] * w_L_force_PP_sum.shape[1],
                w_L_force_PP_sum.shape[2],
                w_L_force_PP_sum.shape[3],
            )
            w_L_force_PP_binned = list(w_L_force_PP_sum.reshape(w_L_force_PP_binned_shape))

            w_L_E_L_force_PP_sum = np.array([np.sum(arr, axis=0) for arr in w_L_E_L_force_PP_split])
            w_L_E_L_force_PP_binned_shape = (
                w_L_E_L_force_PP_sum.shape[0] * w_L_E_L_force_PP_sum.shape[1],
                w_L_E_L_force_PP_sum.shape[2],
                w_L_E_L_force_PP_sum.shape[3],
            )
            w_L_E_L_force_PP_binned = list(w_L_E_L_force_PP_sum.reshape(w_L_E_L_force_PP_binned_shape))

            w_L_binned_local = np.array(w_L_binned)
            w_L_e_L_binned_local = np.array(w_L_e_L_binned)
            w_L_force_HF_binned_local = np.array(w_L_force_HF_binned)
            w_L_force_PP_binned_local = np.array(w_L_force_PP_binned)
            w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned)

            w_L_binned_global_sum = np.sum(w_L_binned_local, axis=0)
            w_L_e_L_binned_global_sum = np.sum(w_L_e_L_binned_local, axis=0)
            w_L_force_HF_binned_global_sum = np.sum(w_L_force_HF_binned_local, axis=0)
            w_L_force_PP_binned_global_sum = np.sum(w_L_force_PP_binned_local, axis=0)
            w_L_E_L_force_PP_binned_global_sum = np.sum(w_L_E_L_force_PP_binned_local, axis=0)

            M_local = w_L_binned_local.size
            logger.debug(f"The number of local binned samples = {M_local}")

            force_HF_jn_local = -1.0 * np.array(
                [
                    (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    for j in range(M_local)
                ]
            )

            force_Pulay_jn_local = -2.0 * np.array(
                [
                    (
                        (w_L_E_L_force_PP_binned_global_sum - w_L_E_L_force_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        - (
                            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                            * (w_L_force_PP_binned_global_sum - w_L_force_PP_binned_local[j])
                            / (w_L_binned_global_sum - w_L_binned_local[j])
                        )
                    )
                    for j in range(M_local)
                ]
            )

            # Mirror production get_aF and re-project the
            # per-binned-sample HF + Pulay sum so the force observable is real
            # on the complex Bloch path. Bit-identical at Gamma.
            force_jn_local = list(np.real(force_HF_jn_local + force_Pulay_jn_local))

            force_jn = force_jn_local
            M_total = len(force_jn)
            logger.debug(f"The number of total binned samples = {M_total}")

            force_mean = np.average(force_jn, axis=0)
            force_std = np.sqrt(M_total - 1) * np.std(force_jn, axis=0)

            logger.devel(f"force_mean.shape  = {force_mean.shape}.")
            logger.devel(f"force_std.shape  = {force_std.shape}.")
            logger.info(f"force = {force_mean} +- {force_std} Ha.")

        else:
            force_mean = None
            force_std = None

        # broadcast the results
        force_mean = self._mpi_comm.bcast(force_mean, root=0)
        force_std = self._mpi_comm.bcast(force_std, root=0)

        return (force_mean, force_std)

    def get_aT(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Symmetric stress tensor with jackknife statistics (debug route, LRDMC/PBC).

        Independent counterpart of :py:meth:`GFMC_n.get_aT`: same single-rank
        bin-block jackknife (branching weights already absorbed into the
        per-step walker average), assembled from the looped-jacrev stress
        buffers stored by the debug ``run``. Molecular structures return zeros.

        Contract (fixed static ansatz): same as :py:meth:`GFMC_n.get_aT` --
        the strain ``F = I + eps`` maps cell, nuclei and electrons as
        ``x -> F x`` (ECP quadrature directions fixed) with basis exponents / coefficients, MO /
        geminal coefficients and Jastrow parameters held fixed, so the
        tensor is NOT the equation-of-state derivative of a re-optimized
        ground-state energy.
        """
        if not self._comput_stress:
            raise ValueError("Stress data were not computed; construct the GFMC driver with comput_stress=True.")
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3, 3)), np.zeros((3, 3))
        if self._mpi_rank != 0:
            return np.zeros((3, 3)), np.zeros((3, 3))

        w_L = self._retained_w_L(num_mcmc_warmup_steps)
        e_L = self.e_L[num_mcmc_warmup_steps:]
        stress_HF = self.stress_HF[num_mcmc_warmup_steps:]
        stress_PP = self.stress_PP[num_mcmc_warmup_steps:]
        E_L_stress_PP = self.E_L_stress_PP[num_mcmc_warmup_steps:]

        M = stress_HF.shape[0]
        _require(w_L.shape == (M, 1), "")
        _require(e_L.shape == (M, 1), "")
        _require(stress_HF.shape == (M, 1, 3, 3), "")
        _require(stress_PP.shape == (M, 1, 3, 3), "")
        _require(E_L_stress_PP.shape == (M, 1, 3, 3), "")

        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_stress_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_HF), num_mcmc_bin_blocks, axis=0)
        w_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_PP), num_mcmc_bin_blocks, axis=0)
        w_L_E_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_stress_PP), num_mcmc_bin_blocks, axis=0)

        w_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split])))
        w_L_e_L_binned = np.array(list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split])))

        def _bin_tensor(splits: Sequence[npt.NDArray]) -> npt.NDArray:
            tensor_sum = np.array([np.sum(arr, axis=0) for arr in splits])
            return tensor_sum.reshape(tensor_sum.shape[0] * tensor_sum.shape[1], 3, 3)

        w_L_stress_HF_binned = _bin_tensor(w_L_stress_HF_split)
        w_L_stress_PP_binned = _bin_tensor(w_L_stress_PP_split)
        w_L_E_L_stress_PP_binned = _bin_tensor(w_L_E_L_stress_PP_split)

        w_L_sum = float(np.sum(w_L_binned))
        # Keep the (possibly complex at a non-Gamma twist) dtype through the
        # jackknife; the physical-real projection happens on the
        # assembled sigma below. A float() cast here would drop Im(e_L) and
        # emit a ComplexWarning.
        w_L_e_L_sum = np.sum(w_L_e_L_binned)
        w_L_stress_HF_sum = np.sum(w_L_stress_HF_binned, axis=0)
        w_L_stress_PP_sum = np.sum(w_L_stress_PP_binned, axis=0)
        w_L_E_L_stress_PP_sum = np.sum(w_L_E_L_stress_PP_binned, axis=0)

        M_local = w_L_binned.size

        sigma_HF_jn = np.array(
            [(w_L_stress_HF_sum - w_L_stress_HF_binned[j]) / (w_L_sum - w_L_binned[j]) for j in range(M_local)]
        )
        sigma_Pulay_jn = 2.0 * np.array(
            [
                (
                    (w_L_E_L_stress_PP_sum - w_L_E_L_stress_PP_binned[j]) / (w_L_sum - w_L_binned[j])
                    - (
                        (w_L_e_L_sum - w_L_e_L_binned[j])
                        / (w_L_sum - w_L_binned[j])
                        * (w_L_stress_PP_sum - w_L_stress_PP_binned[j])
                        / (w_L_sum - w_L_binned[j])
                    )
                )
                for j in range(M_local)
            ]
        )

        sigma_jn = np.real(sigma_HF_jn + sigma_Pulay_jn)
        sigma_jn = 0.5 * (sigma_jn + sigma_jn.transpose(0, 2, 1))
        omega = float(np.abs(np.linalg.det(np.asarray(self._hamiltonian_data.structure_data._cell_jnp))))
        sigma_jn = sigma_jn / omega

        sigma_mean = np.mean(sigma_jn, axis=0)
        var_sigma = np.mean((sigma_jn - sigma_mean) ** 2, axis=0)
        sigma_std = np.sqrt((M_local - 1) * var_sigma)
        return sigma_mean, sigma_std

    def get_aP(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> GFMCArrayEstimate:
        """Normal-pressure diagonal for LRDMC (debug route; mirror of :py:meth:`GFMC_n.get_aP`)."""
        if not self._hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3,)), np.zeros((3,))
        if self._mpi_rank != 0:
            return np.zeros((3,)), np.zeros((3,))
        sigma_mean, sigma_std = self.get_aT(num_mcmc_warmup_steps, num_mcmc_bin_blocks)
        return -np.diag(sigma_mean), np.diag(sigma_std)
