"""QMC module (VMC / MCMC).

Precision Zones:
    This module owns no Precision Zone; it consumes the zones of the
    wavefunction and Hamiltonian modules it calls. The shared SR/LM
    implementation behind ``run_optimize`` lives in :mod:`jqmc.jqmc_optimizer`.

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
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from logging import getLogger
from typing import Any, Literal, TypeAlias, TypeVar

import jax
import numpy as np
import numpy.typing as npt
from jax import jit, lax, vmap
from jax import numpy as jnp
from mpi4py import MPI

from ._build_info import log_build_info_once
from ._driver_helpers import (
    _check_mpi4py_jax_distribution_consistency,
    _initialize_jax_distributed,
    _generate_init_electron_configurations,
    _init_external_control_file,
    _log_memory_usage,
    _read_external_stop_flag,
    _remove_external_control_file,
    _require,
    _raise_collective_nonfinite_observables,
    _validate_num_mcmc_per_measurement,
    derive_rank_rng_streams,
)
from . import _complex_update
from ._precision import precision_policy_snapshot
from . import _setting as _setting_module
from ._setting import (
    ANALYTIC_GRAD_ENABLED,
    MPC_CUTOFF_HA,
    Nv_default,
    OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
    flag_rust_on_CPU,
    get_eps,
    walker_map,
)
from .coulomb_potential import (
    Coulomb_potential_data,
    icosahedron_sym_mesh_Nv12,
    octahedron_sym_mesh_Nv6,
    octahedron_sym_mesh_Nv18,
    tetrahedron_sym_mesh_Nv4,
)
from .determinant import (
    AGP_data,
    AGPn_data,
    Det_ratio_streaming_state,
    SD_data,
    _advance_det_ratio_streaming_state,
    _compute_dn_column_from_det_ratio_state,
    _compute_u_dn_move_from_det_ratio_state,
    _compute_up_row_from_det_ratio_state,
    _compute_v_up_move_from_det_ratio_state,
    _geminal_from_det_ratio_state,
    _equilibrated_geminal_lu_inverse_and_residual,
    _equilibrated_geminal_pinv,
    _init_det_ratio_streaming_state,
    _sherman_morrison_ratio_and_inverse,
    compute_AS_regularization_factor,
    compute_AS_regularization_factor_fast_update,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
)
from .hamiltonians import (
    Hamiltonian_data,
    compute_local_energy,
    compute_local_energy_fast,
)
from .jqmc_optimizer import (
    BlockDerivativeSource,
    DeviceBlockDerivativeSource,
    DeviceJackknifeBinMetadata,
    DeviceJackknifeBlockMoments,
    DeviceOptimizationBatch,
    EnergyEstimate,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    ParameterSelection,
    OptimizationSampler,
    validate_derivative_sources,
)
from .jastrow_factor import (
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_ratio_state,
    _compute_ratio_Jastrow_part_rank1_update,
    advance_Jastrow_ratio_state,
    compute_Jastrow_part_ratio_zone,
    init_Jastrow_ratio_state,
)
from .mpc import build_mpc_data, compute_mpc_correction_ee


def _diagnostic_storage_derivatives(sources: Sequence[BlockDerivativeSource]) -> dict[str, np.ndarray]:
    """Materialize deprecated public/debug derivative arrays, never solver input."""
    result: dict[str, np.ndarray] = {}
    for source in sources:
        if source.direct is not None:
            values = np.real(source.direct)
        elif source.d_pbar is None:
            values = source.d_p
        else:
            values = source.d_p + np.conj(source.d_pbar)
        result[source.name] = np.asarray(values)
    return result


@partial(jit, donate_argnums=(0, 1))
def _accumulate_device_source_moments(
    sum_acc: jax.Array,
    conjugate_energy_acc: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    energies: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate one bounded step chunk into the two exact source moments."""
    trailing_axes = values.ndim - weights.ndim
    expansion = (...,) + (None,) * trailing_axes
    weight_factor = weights[expansion]
    energy_factor = energies[expansion]
    weighted = weight_factor * values
    return (
        sum_acc + jnp.sum(weighted, axis=0),
        conjugate_energy_acc + jnp.sum(weight_factor * jnp.conj(values) * energy_factor, axis=0),
    )


def _bounded_device_binned_source_component(
    values: npt.ArrayLike,
    weights: jax.Array,
    local_energies: jax.Array,
    num_bin_blocks: int,
    *,
    sample_batch_size: int = OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
) -> tuple[jax.Array, jax.Array]:
    """Build per-walker jackknife moments without a full weighted tensor."""
    host_values = np.asarray(values)
    if host_values.ndim < 2:
        raise ValueError("derivative source moments require step and walker axes.")
    num_steps, num_walkers = host_values.shape[:2]
    if tuple(weights.shape) != (num_steps, num_walkers) or tuple(local_energies.shape) != (num_steps, num_walkers):
        raise ValueError("source moments require matching step-by-walker weights and energies.")
    if not 1 <= num_bin_blocks <= num_steps:
        raise ValueError("source moment bin count must be between one and the number of steps.")
    if sample_batch_size <= 0:
        raise ValueError("source moment sample batch size must be positive.")

    steps_per_chunk = max(1, sample_batch_size // num_walkers)
    trailing_shape = host_values.shape[2:]
    sum_dtype = jnp.result_type(host_values.dtype, weights.dtype)
    energy_dtype = jnp.result_type(host_values.dtype, weights.dtype, local_energies.dtype)
    sum_bins = []
    conjugate_energy_bins = []
    for step_indices in np.array_split(np.arange(num_steps), num_bin_blocks):
        bin_start = int(step_indices[0])
        bin_stop = int(step_indices[-1]) + 1
        sum_acc = jnp.zeros((num_walkers, *trailing_shape), dtype=sum_dtype)
        conjugate_energy_acc = jnp.zeros((num_walkers, *trailing_shape), dtype=energy_dtype)
        for start in range(bin_start, bin_stop, steps_per_chunk):
            stop = min(start + steps_per_chunk, bin_stop)
            sum_acc, conjugate_energy_acc = _accumulate_device_source_moments(
                sum_acc,
                conjugate_energy_acc,
                jnp.asarray(host_values[start:stop]),
                weights[start:stop],
                local_energies[start:stop],
            )
            sum_acc.block_until_ready()
            conjugate_energy_acc.block_until_ready()
        sum_bins.append(sum_acc)
        conjugate_energy_bins.append(conjugate_energy_acc)
    return (
        jnp.concatenate(tuple(sum_bins), axis=0),
        jnp.concatenate(tuple(conjugate_energy_bins), axis=0),
    )


from .structure import (
    Structure_data,
    _find_nearest_index_jnp,
    _get_min_dist_rel_R_cart_jnp,
    _get_min_dist_rel_R_cart_np,
)
from .swct import evaluate_swct_domega, evaluate_swct_omega, evaluate_swct_omega_dot_batched
from .wavefunction import VariationalParameterBlock, Wavefunction_data, _project_linear_tie_storage, evaluate_ln_wavefunction

GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
NativeDerivative: TypeAlias = np.ndarray | np.generic | float
NativeDerivativeTuple: TypeAlias = tuple[NativeDerivative | None, ...]
NativeDerivativePayload: TypeAlias = NativeDerivative | NativeDerivativeTuple
NativeArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
NativeOptionalTriple: TypeAlias = tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
BasisBlockLayout: TypeAlias = tuple[tuple[str, int], ...]
ParameterGradientResult: TypeAlias = dict[str, np.ndarray] | tuple[BlockDerivativeSource, ...]
DerivativeSourceStorage: TypeAlias = dict[str, dict[str, list[np.ndarray]]]
ArrayEstimate: TypeAlias = tuple[npt.NDArray, npt.NDArray]
StressEstimate: TypeAlias = ArrayEstimate | tuple[npt.NDArray, npt.NDArray, float, float]
GeminalInverseState: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
MCMCAcceptRejectState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Det_ratio_streaming_state,
    Jastrow_ratio_state,
]
MCMCSweepState: TypeAlias = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Det_ratio_streaming_state,
    Jastrow_ratio_state,
]
PytreeLeaf = TypeVar("PytreeLeaf")
PytreeOwner = TypeVar("PytreeOwner")

# Production parameter-gradient, force, and stress closures route through the
# unified ``jqmc.derivatives`` interface with ``analytic=_ANALYTIC_GRAD_ENABLED``.
# The default lives in ``jqmc._setting.ANALYTIC_GRAD_ENABLED`` (DEFAULT TRUE -- the analytic backend
# is the production default; set it False there to revert production to the ``_auto`` autograd path).
# The ``_MCMC_debug`` oracle driver uses the
# SAME ``_ANALYTIC_GRAD_ENABLED`` backend flag as production, so debug and production share one
# derivative backend (cross-backend parity caused fp32/complex trouble). The debug driver remains an
# INDEPENDENT eager per-walker LOOP implementation; only the backend flag is unified.
_ANALYTIC_GRAD_ENABLED = ANALYTIC_GRAD_ENABLED


# Bridge between the two parameter-block namespaces: the MCMC
# per-block gradient flags are keyed by the ``VariationalParameterBlock.name`` values
# emitted by ``Wavefunction_data.get_variational_blocks`` (block names), whereas the
# ``with_param_grad_mask`` / ``jqmc.derivatives`` layer is keyed by the ``opt_*``
# keyword names. This mapping is the ONE place that pairs the two namespaces. The
# per-block DEFAULT convention (parameter blocks True, "basis" blocks False) is NOT
# duplicated here -- it is derived from the single source
# ``jqmc.derivatives.bridge._MASK_DEFAULTS`` keyed by the ``opt_*`` names.
_PARAM_BLOCK_TO_MASK_KW = {
    "j1_param": "opt_J1_param",
    "j2_param": "opt_J2_param",
    "j3_matrix": "opt_J3_param",
    "lambda_matrix": "opt_geminal",
    "j3_basis_exp": "opt_J3_basis_exp",
    "j3_basis_coeff": "opt_J3_basis_coeff",
    "lambda_basis_exp": "opt_lambda_basis_exp",
    "lambda_basis_coeff": "opt_lambda_basis_coeff",
    "j3_poly_alpha": "opt_JDTN_u_param",
    "j3_poly_beta": "opt_JDTN_chi_param",
    "j3_poly_gamma": "opt_JDTN_f_param",
}


class _NativeFreeKernels:
    """``jqmc_rs`` free kernels evaluated under one explicit precision policy.

    A free (context-less) kernel parses its Python data when it is called, and
    the parsed leaves capture the thread's native construction policy. Production
    never configures that standalone setting, so without a scope a mixed run would
    silently evaluate these kernels in float64. Every call is wrapped in
    ``jqmc_rs.NativeConstructionPrecision(policy)``; the process-wide standalone
    setting is neither read nor changed.
    """

    def __init__(self, module: Any, precision_policy: dict[str, str]) -> None:
        self._module = module
        self._policy = dict(precision_policy)

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` only runs for names not found normally.  Read the two
        # instance fields through ``__dict__`` so that an introspection of a
        # half-constructed instance (e.g. a tracer inside ``__init__``) gets an
        # AttributeError instead of recursing into ``__getattr__('_module')``.
        fields = self.__dict__
        if "_module" not in fields or "_policy" not in fields:
            raise AttributeError(name)
        module, policy = fields["_module"], fields["_policy"]
        kernel = getattr(module, name)
        if not callable(kernel):
            return kernel

        def scoped(*args: Any, **kwargs: Any) -> Any:
            with module.NativeConstructionPrecision(policy):
                return kernel(*args, **kwargs)

        return scoped


# --- Native (Rust) variational-parameter gradient wiring --------------------------
# O_k = d ln|Psi|/dc and dE_L/dc for optimization observables are assembled
# per walker from the free ``jqmc_rs``
# c-axis kernels (``grad_ln_Psi_*`` / ``grad_ln_det_wrt_*`` for O_k, ``grad_eL_wrt_*``
# + the ECP non-local add ``grad_eL_ecp_wrt_*`` for dE_L/dc). The exact
# family->kernel->block-name assembly (incl. the SD/AGPn direct-MO blocks + the
# closed-shell ``C_dn = conj(C_up)`` tie) MIRRORS ``derivatives.bridge.
# _reduce_c_param_grads`` and the analytic-matrix mirror suite
# (``tests/test_rs_kernel/test_rs_analytic_matrix.py``), so the native flat dict is
# key-for-key / shape-for-shape identical to the JAX path's stored set. The free
# kernels recompute their own geminal inverse internally (they take no geminal_inv),
# matching the scalar kernels; they are parity-checked against the XLA analytic c-axis gradient.
class _RsComplexPdCtx:
    """Complex-twist adapter for the parameter-derivative method surface.

    The monolithic native ``ParamDerivArgs`` parses a REAL geminal. The complex
    Jastrow families (DTN and J1 / J2 / J3 ``j_matrix``) have a fixed-per-run
    native context that shares the sweep/e_L complex geminal (critical for grid-sized
    B-spline storage) and parses the Jastrow / J3 orbital / ECP tables once, instead of
    once per walker and family; the remaining complex c-axis kernels (geminal and basis
    families, and every family when no native context is supplied) continue through
    their free functions:
    ``grad_ln_det_{mo,lambda}_complex``,
    ``grad_T_mo_withJ_complex``, ``grad_eL_ecp_wrt_mo_coeff_complex``, and the
    ``grad_eL_*_complex`` Jastrow/DTN family).

    This adapter exposes both routes under the SAME ``pd_*`` method names the real
    ``NativeObsCtx`` uses, keeping family/tie logic and block names shared in
    ``_rs_native_c_param_grads``; complex-only contributions are selected explicitly. O_k for
    the Jastrow families is served by the REAL kernels -- the Jastrow is real and its
    ``dln_Psi/dc`` does not see the determinant at all.

    Families WITHOUT a complex kernel are never reached from here: the driver raises
    ``NotImplementedError`` up front (see ``_rs_pd_complex_gap``), never falls back.
    """

    def __init__(
        self,
        geminal_data: GeminalData,
        jastrow_data: Jastrow_data,
        coulomb_potential_data: Coulomb_potential_data,
        native_ctx: Any | None = None,
    ) -> None:
        import jqmc_rs

        # The free kernels run under the policy of the run's native context (the
        # Python snapshot it was built with); a direct unit-test construction
        # without a context uses the active Python policy.
        policy = native_ctx.precision_policy() if native_ctx is not None else precision_policy_snapshot()
        self._rs = _NativeFreeKernels(jqmc_rs, policy)
        self._gem = geminal_data
        self._jas = jastrow_data
        self._cp = coulomb_potential_data
        # The fixed-per-run context owns the single parsed complex geminal. DTN is
        # delegated there so a B-spline geminal is not re-expanded for every walker.
        # Optional for direct unit tests of the remaining free-kernel adapter methods.
        self._native_ctx = native_ctx
        # The once-parsed complex Jastrow tables exist only on a context built with
        # ``want_param_deriv=True``; any other context keeps the free kernels.
        self._native_jastrow = native_ctx if (native_ctx is not None and native_ctx.has_complex_param_deriv()) else None
        self._j1 = getattr(jastrow_data, "jastrow_one_body_data", None)
        self._j2 = getattr(jastrow_data, "jastrow_two_body_data", None)
        self._j3 = getattr(jastrow_data, "jastrow_three_body_data", None)
        self._dtn = getattr(jastrow_data, "jastrow_dtn_data", None)

    # ---- O_k = dln_Psi/dc -------------------------------------------------------
    def pd_ok_j1(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        return self._rs.grad_ln_Psi_jastrow_1b_param(self._j1, r_up, r_dn)

    def pd_ok_j2(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        return self._rs.grad_ln_Psi_jastrow_2b_param(self._j2, r_up, r_dn)

    def pd_ok_j3(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        # COMPLEX twin: at a twist the J3 orbital leaf is complex-Bloch, and for the
        # MO-wrapped composite / plane-wave / B-spline leaves the REAL entry cannot even
        # ingest it (``AOs_bspline_complex_data`` has no ``_complex``,
        # ``AOs_plane_wave_complex_data`` no ``_num_ao_kernel``). ``A = Re(orbital)`` with
        # Re taken AFTER the MO contraction, so the returned O_k stays real.
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_ok_j3_complex(r_up, r_dn)
        return self._rs.grad_ln_Psi_jastrow_3b_jmatrix_complex(self._j3, r_up, r_dn)

    def pd_ok_dtn(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeOptionalTriple:
        if self._native_ctx is not None:
            return self._native_ctx.pd_ok_dtn(r_up, r_dn)
        return self._rs.grad_ln_Psi_dtn_params(self._dtn, r_up, r_dn)

    def pd_ok_mo(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        return self._rs.grad_ln_det_mo_complex(self._gem, r_up, r_dn)

    def pd_ok_lambda(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        return self._rs.grad_ln_det_lambda_complex(self._gem, r_up, r_dn)

    def pd_ok_lambda_diag(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        # Optimizer sources are d log(Psi), not d log|Psi|, even for real
        # occupations. The diagnostic natural-orbital export intentionally
        # projects the latter; reduce the existing holomorphic dense kernel here.
        dense = self.pd_ok_lambda(r_up, r_dn)
        n = self._gem.lambda_diag.shape[0]
        return np.diagonal(dense[:, :n]), dense[:, n:]

    # ---- dE_L/dc kinetic --------------------------------------------------------
    def pd_el_j1(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_j1_complex(r_up, r_dn)
        return self._rs.grad_eL_wrt_jastrow_1b_param_complex(self._gem, self._jas, r_up, r_dn, True)

    def pd_el_j2(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_j2_complex(r_up, r_dn)
        return self._rs.grad_eL_wrt_jastrow_2b_param_complex(self._gem, self._jas, r_up, r_dn, True)

    def pd_el_j3(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_j3_complex(r_up, r_dn)
        return self._rs.grad_eL_wrt_jastrow_3b_jmatrix_complex(self._gem, self._jas, r_up, r_dn)

    def pd_el_dtn(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeOptionalTriple:
        if self._native_ctx is not None:
            return self._native_ctx.pd_el_dtn(r_up, r_dn)
        return self._rs.grad_eL_dtn_kinetic_params_complex(self._gem, self._jas, r_up, r_dn, True)

    def pd_el_mo(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        geminal_inv: npt.ArrayLike | None = None,
    ) -> NativeArrayPair:
        return self._rs.grad_T_mo_withJ_complex(self._gem, self._jas, r_up, r_dn, geminal_inv)

    def pd_el_lambda(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeDerivative:
        return self._rs.grad_T_lambda_withJ_complex(self._gem, self._jas, r_up, r_dn)

    def pd_el_lambda_diag(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        dense = self.pd_el_lambda(r_up, r_dn)
        n = self._gem.lambda_diag.shape[0]
        return np.diagonal(dense[:, :n]), dense[:, n:]

    # ---- dE_L/dc ECP non-local add ---------------------------------------------
    def pd_el_ecp_j1(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_ecp_j1_complex(r_up, r_dn, weights, grid_points)
        return self._rs.grad_eL_ecp_wrt_jastrow_1b_param_complex(
            self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points, True
        )

    def pd_el_ecp_j2(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_ecp_j2_complex(r_up, r_dn, weights, grid_points)
        return self._rs.grad_eL_ecp_wrt_jastrow_2b_param_complex(
            self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points, True
        )

    def pd_el_ecp_j3(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeDerivative:
        if self._native_jastrow is not None:
            return self._native_jastrow.pd_el_ecp_j3_complex(r_up, r_dn, weights, grid_points)
        return self._rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex(
            self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points
        )

    def pd_el_ecp_dtn(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeOptionalTriple:
        if self._native_ctx is not None:
            return self._native_ctx.pd_el_ecp_dtn(r_up, r_dn, weights, grid_points)
        return self._rs.grad_eL_ecp_dtn_params_complex(self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points, True)

    # ---- Walker batches of the Jastrow families (``_pd_batch_base`` / ``_pd_batch_ecp``) ----
    # Served by the native context's rayon twins whenever the scalar method above is,
    # so the walkers of one rank run in parallel instead of one after another; walker
    # ``w`` equals the scalar call on walker ``w``. Without a native context the scalar
    # method is looped per walker.
    def _batch(
        self,
        native: Any | None,
        name: str,
        native_batch: str,
        r_up_b: np.ndarray,
        r_dn_b: np.ndarray,
        *ecp: npt.ArrayLike,
    ) -> NativeDerivativePayload:
        if native is not None:
            return getattr(native, native_batch)(r_up_b, r_dn_b, *ecp)
        scalar = getattr(self, name)
        if ecp:
            weights, gp_b = ecp
            return _pd_stack([scalar(r_up_b[w], r_dn_b[w], weights, gp_b[w]) for w in range(r_up_b.shape[0])])
        return _pd_stack([scalar(r_up_b[w], r_dn_b[w]) for w in range(r_up_b.shape[0])])

    def pd_ok_j3_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_ok_j3", "pd_ok_j3_complex_batch", r_up_b, r_dn_b)

    def pd_el_j1_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_j1", "pd_el_j1_complex_batch", r_up_b, r_dn_b)

    def pd_el_j2_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_j2", "pd_el_j2_complex_batch", r_up_b, r_dn_b)

    def pd_el_j3_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_j3", "pd_el_j3_complex_batch", r_up_b, r_dn_b)

    def pd_el_ecp_j1_batch(
        self, r_up_b: np.ndarray, r_dn_b: np.ndarray, weights: npt.ArrayLike, gp_b: np.ndarray
    ) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_ecp_j1", "pd_el_ecp_j1_complex_batch", r_up_b, r_dn_b, weights, gp_b)

    def pd_el_ecp_j2_batch(
        self, r_up_b: np.ndarray, r_dn_b: np.ndarray, weights: npt.ArrayLike, gp_b: np.ndarray
    ) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_ecp_j2", "pd_el_ecp_j2_complex_batch", r_up_b, r_dn_b, weights, gp_b)

    def pd_el_ecp_j3_batch(
        self, r_up_b: np.ndarray, r_dn_b: np.ndarray, weights: npt.ArrayLike, gp_b: np.ndarray
    ) -> NativeDerivativePayload:
        return self._batch(self._native_jastrow, "pd_el_ecp_j3", "pd_el_ecp_j3_complex_batch", r_up_b, r_dn_b, weights, gp_b)

    def pd_ok_dtn_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_ctx, "pd_ok_dtn", "pd_ok_dtn_complex_batch", r_up_b, r_dn_b)

    def pd_el_dtn_batch(self, r_up_b: np.ndarray, r_dn_b: np.ndarray) -> NativeDerivativePayload:
        return self._batch(self._native_ctx, "pd_el_dtn", "pd_el_dtn_complex_batch", r_up_b, r_dn_b)

    def pd_el_ecp_dtn_batch(
        self, r_up_b: np.ndarray, r_dn_b: np.ndarray, weights: npt.ArrayLike, gp_b: np.ndarray
    ) -> NativeDerivativePayload:
        return self._batch(self._native_ctx, "pd_el_ecp_dtn", "pd_el_ecp_dtn_complex_batch", r_up_b, r_dn_b, weights, gp_b)

    def pd_el_ecp_mo(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeArrayPair:
        return self._rs.grad_eL_ecp_wrt_mo_coeff_complex(self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points)

    def pd_el_ecp_lambda(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeDerivative:
        return self._rs.grad_eL_ecp_wrt_lambda_complex(
            self._gem,
            self._jas,
            self._cp,
            r_up,
            r_dn,
            weights,
            grid_points,
        )

    def pd_el_ecp_lambda_diag(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeArrayPair:
        dense = self.pd_el_ecp_lambda(r_up, r_dn, weights, grid_points)
        n = self._gem.lambda_diag.shape[0]
        return np.diagonal(dense[:, :n]), dense[:, n:]

    # ---- Geminal AO radial basis optimization (complex Bloch) ------------------
    # The free ``*_wrt_radial_complex`` kernels return the per-spin dict
    # ``{orb_data_{up,dn}_spin: (dexp, dcoeff)}``; the assembler (and the real
    # ``NativeObsCtx.pd_*_basis``) want ``(exp, coeff)`` with up+dn CONCATENATED,
    # matching ``bridge._reduce_c_param_grads``' ``lambda_basis_*``.
    @staticmethod
    def _basis_cat(
        d: dict[str, tuple[npt.ArrayLike, npt.ArrayLike]],
    ) -> NativeArrayPair:
        up, dn = d["orb_data_up_spin"], d["orb_data_dn_spin"]
        return (
            np.concatenate([np.asarray(up[0]), np.asarray(dn[0])]),
            np.concatenate([np.asarray(up[1]), np.asarray(dn[1])]),
        )

    def pd_ok_basis(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        return self._basis_cat(self._rs.grad_ln_det_wrt_radial_complex(self._gem, r_up, r_dn))

    def pd_el_basis(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        return self._basis_cat(self._rs.grad_eL_wrt_radial_complex(self._gem, r_up, r_dn, self._jas, True))

    def pd_el_ecp_basis(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeArrayPair:
        return self._basis_cat(
            self._rs.grad_eL_ecp_wrt_radial_complex(self._gem, self._jas, self._cp, r_up, r_dn, weights, grid_points, True)
        )

    # ---- J3 orbital radial basis-opt (complex Bloch) ---------------------------
    def pd_ok_j3_basis(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        return self._rs.grad_ln_Psi_jastrow_3b_radial_complex(self._jas, r_up, r_dn)

    def pd_el_j3_basis(self, r_up: npt.ArrayLike, r_dn: npt.ArrayLike) -> NativeArrayPair:
        return self._rs.grad_eL_wrt_jastrow_3b_radial_complex(self._gem, self._jas, r_up, r_dn)

    def pd_el_ecp_j3_basis(
        self,
        r_up: npt.ArrayLike,
        r_dn: npt.ArrayLike,
        weights: npt.ArrayLike,
        grid_points: npt.ArrayLike,
    ) -> NativeArrayPair:
        return self._rs.grad_eL_ecp_wrt_jastrow_3b_radial_complex(
            self._gem,
            self._jas,
            self._cp,
            r_up,
            r_dn,
            weights,
            grid_points,
        )


def _rs_pd_complex_gap(
    hamiltonian_data: Hamiltonian_data,
    mask: Mapping[str, bool],
    *,
    ecp_flag: bool,
    want_e_L: bool,
) -> str | None:
    """Reason string when the complex-twist c-axis assembler has no kernel for a
    REQUESTED family, else ``None``. The driver turns this into a strict
    ``NotImplementedError`` -- never a silent JAX fallback.

    Dense AGP lambda routes are fully served: the with-Jastrow kinetic kernel includes
    the ``-sum gJ.gD`` cross term and the complex ECP non-local add is wired below.
    AGPn uses the same native dense holomorphic kernels, reduced onto occupation
    and unpaired leaves without projecting away their optimizer-source phase.

    The J3 ``j_matrix`` has no gap: ``grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex``
    implements the complex-mesh O_k identity with array-valued
    ``dJ3/dj_matrix``, and ``_RsComplexPdCtx.pd_el_ecp_j3`` calls it.
    """
    del hamiltonian_data, mask, ecp_flag, want_e_L
    return None


def _rs_wrap_home_cell(
    structure_data: Structure_data,
    r_b: Sequence[npt.ArrayLike],
) -> list[np.ndarray]:
    """Map a per-walker list of Cartesian positions onto their home-cell representatives.

    Thin numpy front end to the production ``structure._wrap_into_cell_if_pbc``
    convention (fractional ``x - floor(x)``, then back to Cartesian) used by
    ``compute_AOs``: the native c-axis kernels must see the same representative jqmc's
    AO evaluation does. Molecular structures pass through untouched.
    """
    from .structure import _wrap_into_cell_if_pbc

    stacked = jnp.asarray(np.stack([np.asarray(r, dtype=np.float64) for r in r_b]))
    wrapped = np.asarray(_wrap_into_cell_if_pbc(structure_data, stacked), dtype=np.float64)
    return [np.ascontiguousarray(wrapped[w]) for w in range(wrapped.shape[0])]


def _rs_j3_basis_layout(j3d: Jastrow_three_body_data) -> tuple[BasisBlockLayout, BasisBlockLayout]:
    """Return the exact flat J3 radial block layout published by production.

    ``get_basis_variational_blocks`` is the same leaf protocol used by
    ``Jastrow_data.collect_param_grads`` and ``Wavefunction_data``.  It therefore
    covers both a simple GTO leaf (``j3_basis_*``) and GTO children of a composite
    (``j3_child<i>_basis_*``), while a pure PW/B-spline returns an empty layout.
    Sizes are retained so the single native concatenated radial vector can be split
    back into the child-qualified blocks without guessing names or child order.
    """
    from .molecular_orbital import MOs_data

    orb = j3d.orb_data
    aos = orb.aos_data if isinstance(orb, MOs_data) else orb
    blocks = aos.get_basis_variational_blocks(
        opt_basis_exp=True,
        opt_basis_coeff=True,
        block_name_prefix="j3_",
        state_token=("rust-j3-basis-layout", 0),
        sync_scope="sampler",
    )
    exp = tuple((block.name, int(np.asarray(block.values).size)) for block in blocks if block.name.endswith("_basis_exp"))
    coeff = tuple((block.name, int(np.asarray(block.values).size)) for block in blocks if block.name.endswith("_basis_coeff"))
    if tuple(size for _name, size in exp) != tuple(size for _name, size in coeff):
        raise ValueError(f"inconsistent J3 radial exp/coeff layout: exp={exp}, coeff={coeff}")
    return exp, coeff


def _rs_c_param_derivative_sources(
    blocks: Sequence[VariationalParameterBlock],
    flat: Mapping[str, np.ndarray],
    *,
    tied_mo: NativeArrayPair | None = None,
) -> tuple[BlockDerivativeSource, ...]:
    """Publish exact optimizer sources from the native c-axis kernel payload.

    The Rust kernels return the same ambient-storage derivatives as the JAX
    analytical bridge.  Most blocks therefore map mechanically from the flat
    native payload to their declared derivative policy.  A time-reversal-tied
    MO block is the one exception: the optimizer needs the unfused up/down
    components as ``(d_p, d_pbar)``, while the original diagnostic dictionary
    stores their folded ``d_p + conj(d_pbar)`` value.
    """
    published_blocks = tuple(blocks)
    sources = []
    for block in published_blocks:
        if block.name == "mo_coefficients_up" and tied_mo is not None:
            grad_up, grad_dn = tied_mo
            if block.derivative_policy == "direct_real":
                source = BlockDerivativeSource.build(block=block, direct=grad_up + grad_dn)
            elif block.derivative_policy == "wirtinger_pair":
                source = BlockDerivativeSource.build(block=block, d_p=grad_up, d_pbar=grad_dn)
            else:
                raise ValueError("a time-reversal MO block must be direct_real or wirtinger_pair.")
        else:
            if block.name not in flat:
                raise ValueError(f"native derivative payload is missing variational block {block.name!r}.")
            values = flat[block.name]
            if block.derivative_policy == "direct_real":
                source = BlockDerivativeSource.build(block=block, direct=values)
            elif block.derivative_policy == "holomorphic":
                source = BlockDerivativeSource.build(block=block, d_p=values)
            elif block.derivative_policy == "wirtinger_pair":
                source = BlockDerivativeSource.build(block=block, d_p=values, d_pbar=np.zeros_like(values))
            else:
                raise ValueError(f"unknown derivative policy {block.derivative_policy!r} for {block.name!r}.")
        sources.append(source)
    return validate_derivative_sources(published_blocks, sources)


def _pd_stack(per_walker: Sequence[NativeDerivativePayload]) -> NativeDerivativePayload:
    """Stack a per-walker list of ``pd_*`` returns with a leading walker axis.

    The Python-loop fallback used by :func:`_pd_batch_base` / :func:`_pd_batch_ecp`
    when the ctx exposes no ``*_batch`` method (the complex-twist adapter's geminal and
    basis families, and its Jastrow families without a native context). Scalar
    (float) and single-array returns become ``(nw, ...)``; a tuple return keeps its
    arity, and a component that is ``None`` for every walker (an unused DTN u/chi/f
    channel -- the flag is per-run constant, so all walkers agree) stays ``None``.
    """
    first = per_walker[0]
    if isinstance(first, tuple):
        return tuple(
            None if all(x[k] is None for x in per_walker) else np.asarray([np.asarray(x[k]) for x in per_walker])
            for k in range(len(first))
        )
    return np.asarray(per_walker)


def _pd_batch_base(
    ctx: Any,
    name: str,
    r_up_b: npt.ArrayLike,
    r_dn_b: npt.ArrayLike,
) -> NativeDerivativePayload:
    """Batched ``ctx.<name>`` over the walker axis.

    Prefers the rayon batch method ``ctx.<name>_batch(r_up_b, r_dn_b)`` (the real
    ``NativeObsCtx`` path, and the complex-twist adapter's Jastrow families); falls back
    to a per-walker Python loop over the scalar ``ctx.<name>`` where no ``_batch`` exists.
    Both return the same leading-walker-axis structure, so the assembler is uniform.
    """
    # The home-cell wrap hands back a per-walker LIST; the rayon batch entry needs a
    # contiguous (nw, N, 3) array. Stack once (a no-op when already an ndarray).
    r_up_b = np.asarray(r_up_b)
    r_dn_b = np.asarray(r_dn_b)
    batched = getattr(ctx, name + "_batch", None)
    if batched is not None:
        return batched(r_up_b, r_dn_b)
    fn = getattr(ctx, name)
    return _pd_stack([fn(r_up_b[w], r_dn_b[w]) for w in range(r_up_b.shape[0])])


def _pd_batch_ecp(
    ctx: Any,
    name: str,
    r_up_b: npt.ArrayLike,
    r_dn_b: npt.ArrayLike,
    weights: npt.ArrayLike,
    gp_b: npt.ArrayLike,
) -> NativeDerivativePayload:
    """ECP twin of :func:`_pd_batch_base`. ``weights`` is shared across walkers
    (RT-invariant); ``gp_b`` is the per-walker rotated grid ``ecp_gp @ RT[w]``
    (nw, n_grid, 3), so walker ``w`` receives ``gp_b[w]`` -- byte-identical to the
    scalar ``_gp(w)``.
    """
    r_up_b = np.asarray(r_up_b)
    r_dn_b = np.asarray(r_dn_b)
    batched = getattr(ctx, name + "_batch", None)
    if batched is not None:
        return batched(r_up_b, r_dn_b, weights, gp_b)
    fn = getattr(ctx, name)
    return _pd_stack([fn(r_up_b[w], r_dn_b[w], weights, gp_b[w]) for w in range(r_up_b.shape[0])])


def _rs_native_c_param_grads(
    ctx: Any,
    hamiltonian_data: Hamiltonian_data,
    r_up_b: npt.ArrayLike,
    r_dn_b: npt.ArrayLike,
    mask: Mapping[str, bool],
    nw: int,
    *,
    quantity: Literal["ln_Psi", "e_L"],
    ecp_flag: bool = False,
    ecp_w: npt.ArrayLike | None = None,
    ecp_gp: npt.ArrayLike | None = None,
    rts_b: npt.ArrayLike | None = None,
    complex_twist: bool = False,
    source_blocks: Sequence[VariationalParameterBlock] | None = None,
    geminal_inv: npt.ArrayLike | None = None,
) -> ParameterGradientResult:
    """Assemble the flat block-name dict of native per-walker c-axis grads.

    ``quantity`` is ``"ln_Psi"`` (O_k) or ``"e_L"`` (dE_L/dc). For ``"e_L"`` under an
    ECP the kinetic dT/dc is summed with the non-local dV_ecp_nl/dc mesh-O_k add
    (``pd_el_ecp_*``, RT-rotated base mesh per walker -- byte-identical to the
    e_L / force mesh rotation). Only families active in ``mask`` (and present in the
    wavefunction) are emitted, matching the JAX stored set after the downstream flag
    filter. Positions arrive as per-walker numpy and, on a periodic run, are mapped onto
    their home-cell representatives here (see the wrap block below).

    ``ctx`` is the native ``NativeObsCtx`` on the real path and ``_RsComplexPdCtx`` (the
    free complex-kernel adapter) at a twist. Shared ``pd_*`` methods handle the
    common families; AGPn reduces its ECP lambda source onto the natural-orbital leaves.

    The c-axis kernels are called as methods on the once-built ``NativeObsCtx``
    (``build_obs_ctx(want_param_deriv=True)``,
    ``ParamDerivArgs``) instead of the module-level free wrappers, so the whole
    Geminal / Jastrow / ECP ``from_py`` parse is paid ONCE per run() rather than per
    walker per family. Same kernels + same owned data -> byte-identical to the free
    path (pinned by the A/B dispatch rows). ``hamiltonian_data`` is used ONLY for the
    Python-side family/tie introspection (gem type, DTN u/chi/f flags, time_reversal,
    lambda_unpaired size); no per-walker Hamiltonian re-parse remains.

    With ``source_blocks`` omitted, return the flat diagnostic dictionary,
    projecting real AGPn parameter leaves but preserving complex unpaired leaves.
    Supplying the canonical optimizer block list
    returns validated :class:`BlockDerivativeSource` objects instead, retaining
    the two exact components of a complex time-reversal tie.
    """
    from .determinant import AGPn_data, SD_data, _gto_leaves_of

    wf = hamiltonian_data.wavefunction_data
    jas = wf.jastrow_data
    gem = wf.geminal_data
    is_eL = quantity == "e_L"
    flat: dict[str, np.ndarray] = {}
    tied_mo_sources = None

    # PERIODIC HOME-CELL WRAP. The free O_k c-axis AO kernels
    # (``grad_ln_det_*`` / ``grad_j3_jmatrix`` / ``grad_*_basis``) re-evaluate the AO
    # stack from the raw walker coordinate, while jqmc's ``compute_AOs`` evaluates
    # every periodic AO at the HOME-CELL representative (``_wrap_into_cell_if_pbc``)
    # and sums images from there. A walker that has roamed whole lattice vectors out
    # therefore made the native geminal / J3 c-axis blocks disagree with XLA by 100%
    # (measured: lattice-shift invariance violated by rel 1.0 on ``mo_coefficients_up``
    # / ``j3_matrix``, while the XLA block is shift-invariant to ~1e-15). Wrapping the
    # INPUT is exact for real Gamma because dln_Psi/dc is lattice-shift invariant.
    # Complex Bloch kernels are the exception: ``ComplexOrb`` performs the wrap and
    # restores the lattice phase internally.  Pre-wrapping here would erase that
    # restore phase, which is observable in the real J3 orbital entering J3.
    #
    # BUT the wrap is applied ONLY to the AO-based O_k blocks. The Jastrow-metric
    # blocks (J1 / J2 / DTN) must keep the RAW coordinate: jqmc evaluates their
    # electron distances with the Jastrow's OWN ``structure_data``, which for the
    # two-body J2 commonly carries ``pbc_flag == False`` -> a molecular (Euclidean)
    # metric that is NOT wrap-invariant. Wrapping each electron independently into
    # ``[0, 1)`` splits a nearby pair across the cell and diverges from XLA by up to
    # ~0.45. Periodic (smooth / min-image) J1/J2/DTN are
    # wrap-invariant, so RAW is exact for them too. The dE_L/dc (``is_eL``) kernels
    # wrap the AO half INTERNALLY (``param_grad_el::wrap_both`` / ``kinetic_drift``)
    # while keeping the Jastrow metric on the raw input, so they always receive RAW
    # coordinates here; only the O_k AO blocks are pre-wrapped.
    if bool(hamiltonian_data.structure_data.pbc_flag) and not is_eL and not complex_twist:
        r_up_ao = _rs_wrap_home_cell(hamiltonian_data.structure_data, r_up_b)
        r_dn_ao = _rs_wrap_home_cell(hamiltonian_data.structure_data, r_dn_b)
    else:
        r_up_ao, r_dn_ao = r_up_b, r_dn_b

    # Per-walker rotated ECP grid, precomputed once (nw, n_grid, 3). Built walker by
    # walker so ``gp_b[w]`` is byte-identical to ``ecp_gp @ rts_b[w]``; the
    # matmul is tiny (n_grid x 3 @ 3 x 3) and off the rayon-parallel kernel path.
    gp_b = np.stack([ecp_gp @ rts_b[w] for w in range(nw)]) if (ecp_gp is not None and rts_b is not None) else None

    # ---- Jastrow families (J1 / J2 scalar, J3 j_matrix, DTN u/chi/f) ----
    if jas is not None:
        j1d = jas.jastrow_one_body_data
        j2d = jas.jastrow_two_body_data
        j3d = jas.jastrow_three_body_data
        dtn = jas.jastrow_dtn_data

        if mask.get("opt_J1_param", False) and j1d is not None:
            if is_eL:
                v = _pd_batch_base(ctx, "pd_el_j1", r_up_b, r_dn_b)
                if ecp_flag:
                    v = v + _pd_batch_ecp(ctx, "pd_el_ecp_j1", r_up_b, r_dn_b, ecp_w, gp_b)
            else:
                v = _pd_batch_base(ctx, "pd_ok_j1", r_up_b, r_dn_b)
            flat["j1_param"] = np.asarray(v)

        if mask.get("opt_J2_param", False) and j2d is not None:
            if is_eL:
                v = _pd_batch_base(ctx, "pd_el_j2", r_up_b, r_dn_b)
                if ecp_flag:
                    v = v + _pd_batch_ecp(ctx, "pd_el_ecp_j2", r_up_b, r_dn_b, ecp_w, gp_b)
            else:
                v = _pd_batch_base(ctx, "pd_ok_j2", r_up_b, r_dn_b)
            flat["j2_param"] = np.asarray(v)

        if mask.get("opt_J3_param", False) and j3d is not None:
            # J3 reads an AO (orbital) stack -> home-cell coords (``r_*_ao``) on the
            # real path.  ComplexOrb wraps and restores Bloch phase internally, so a
            # complex-twist O_k receives raw coordinates, as do all e_L kernels.
            if is_eL:
                v = _pd_batch_base(ctx, "pd_el_j3", r_up_ao, r_dn_ao)
                if ecp_flag:
                    v = v + _pd_batch_ecp(ctx, "pd_el_ecp_j3", r_up_ao, r_dn_ao, ecp_w, gp_b)
            else:
                v = _pd_batch_base(ctx, "pd_ok_j3", r_up_ao, r_dn_ao)
            flat["j3_matrix"] = np.asarray(v)

        # ---- J3 ORBITAL AO radial basis-opt (``opt_J3_basis_{exp,coeff}``) ----
        # Mirrors ``Jastrow_data.collect_param_grads``:
        # the ``j3_basis_exp`` / ``j3_basis_coeff`` blocks exist only when the J3
        # orbital resolves to a GTO AO leaf (bare or MO-backed) -- a PW / B-spline /
        # composite J3 orbital publishes none on either route, so the native branch
        # must stay silent for it too or the block SETS diverge. Unlike the geminal
        # ``lambda_basis_*`` pair there is ONE shared orbital, so the native method
        # returns a single shell-rank vector per block (up + dn summed in the kernel).
        # Same coordinate policy as ``j3_matrix`` above: pre-wrapped ``r_*_ao`` only
        # for real O_k; complex O_k and dE_L/dc wrap their AO half internally.
        _want_j3_basis = mask.get("opt_J3_basis_exp", False) or mask.get("opt_J3_basis_coeff", False)
        _j3_exp_layout, _j3_coeff_layout = _rs_j3_basis_layout(j3d) if _want_j3_basis and j3d is not None else ((), ())
        if _want_j3_basis and j3d is not None and _j3_exp_layout:
            if is_eL:
                jexp, jcoeff = _pd_batch_base(ctx, "pd_el_j3_basis", r_up_ao, r_dn_ao)
                if ecp_flag:
                    eex, eco = _pd_batch_ecp(ctx, "pd_el_ecp_j3_basis", r_up_ao, r_dn_ao, ecp_w, gp_b)
                    jexp, jcoeff = np.asarray(jexp) + np.asarray(eex), np.asarray(jcoeff) + np.asarray(eco)
            else:
                jexp, jcoeff = _pd_batch_base(ctx, "pd_ok_j3_basis", r_up_ao, r_dn_ao)
            jexp = np.asarray(jexp)
            jcoeff = np.asarray(jcoeff)

            def _publish_layout(
                values: np.ndarray,
                layout: BasisBlockLayout,
                enabled: bool,
            ) -> None:
                if not enabled:
                    return
                offset = 0
                for name, size in layout:
                    flat[name] = values[..., offset : offset + size]
                    offset += size
                if offset != values.shape[-1]:
                    raise ValueError(
                        "native J3 radial vector does not match production block layout: "
                        f"native_size={values.shape[-1]}, layout={layout}"
                    )

            _publish_layout(jexp, _j3_exp_layout, mask.get("opt_J3_basis_exp", False))
            _publish_layout(jcoeff, _j3_coeff_layout, mask.get("opt_J3_basis_coeff", False))

        if dtn is not None:
            if is_eL:
                _a, _b, _g = _pd_batch_base(ctx, "pd_el_dtn", r_up_b, r_dn_b)
                if ecp_flag:
                    ea, eb, eg = _pd_batch_ecp(ctx, "pd_el_ecp_dtn", r_up_b, r_dn_b, ecp_w, gp_b)
                    _a = _a if ea is None else np.asarray(_a) + np.asarray(ea)
                    _b = _b if eb is None else np.asarray(_b) + np.asarray(eb)
                    _g = _g if eg is None else np.asarray(_g) + np.asarray(eg)
            else:
                _a, _b, _g = _pd_batch_base(ctx, "pd_ok_dtn", r_up_b, r_dn_b)
            if mask.get("opt_JDTN_u_param", False) and dtn.use_u:
                flat["j3_poly_alpha"] = np.asarray(_a)
            if mask.get("opt_JDTN_chi_param", False) and dtn.use_chi:
                flat["j3_poly_beta"] = np.asarray(_b)
            if mask.get("opt_JDTN_f_param", False) and dtn.use_f:
                flat["j3_poly_gamma"] = np.asarray(_g)

    # ---- Geminal / determinant (lambda for AGP; direct-MO blocks for SD/AGPn) ----
    # AO-based -> home-cell coords (``r_*_ao``): pre-wrapped for O_k, raw for e_L
    # (the kernel wraps its AO half internally). See the wrap block above.
    if mask.get("opt_geminal", False) and gem is not None:
        if isinstance(gem, (SD_data, AGPn_data)):
            if is_eL:
                if geminal_inv is None:
                    gu_s, gd_s = _pd_batch_base(ctx, "pd_el_mo", r_up_ao, r_dn_ao)
                else:
                    mo_r_up = np.asarray(r_up_ao)
                    mo_r_dn = np.asarray(r_dn_ao)
                    mo_geminal_inv = np.asarray(geminal_inv)
                    mo_batch = getattr(ctx, "pd_el_mo_batch", None)
                    if mo_batch is not None:
                        gu_s, gd_s = mo_batch(mo_r_up, mo_r_dn, mo_geminal_inv)
                    else:
                        gu_s, gd_s = _pd_stack(
                            [
                                ctx.pd_el_mo(mo_r_up[walker], mo_r_dn[walker], mo_geminal_inv[walker])
                                for walker in range(mo_r_up.shape[0])
                            ]
                        )
                if ecp_flag:
                    eu, ed = _pd_batch_ecp(ctx, "pd_el_ecp_mo", r_up_ao, r_dn_ao, ecp_w, gp_b)
                    gu_s, gd_s = np.asarray(gu_s) + np.asarray(eu), np.asarray(gd_s) + np.asarray(ed)
            else:
                gu_s, gd_s = _pd_batch_base(ctx, "pd_ok_mo", r_up_ao, r_dn_ao)
            gu_s, gd_s = np.asarray(gu_s), np.asarray(gd_s)
            if isinstance(gem, AGPn_data):
                if is_eL:
                    diag_s, unp_s = _pd_batch_base(ctx, "pd_el_lambda_diag", r_up_ao, r_dn_ao)
                    if ecp_flag:
                        if complex_twist:
                            ed, eu = _pd_batch_ecp(ctx, "pd_el_ecp_lambda_diag", r_up_ao, r_dn_ao, ecp_w, gp_b)
                        else:
                            # The native context already evaluates the dense ECP
                            # gradient in one walker batch. Apply the same linear
                            # leaf map as grad_ecp_lambda_diag without reparsing
                            # the Hamiltonian or repeating its mesh reduction.
                            dense = np.asarray(_pd_batch_ecp(ctx, "pd_el_ecp_lambda", r_up_ao, r_dn_ao, ecp_w, gp_b))
                            n = gem.lambda_diag.shape[0]
                            ed = np.diagonal(dense[..., :n], axis1=-2, axis2=-1)
                            eu = dense[..., n:]
                        diag_s, unp_s = np.asarray(diag_s) + np.asarray(ed), np.asarray(unp_s) + np.asarray(eu)
                else:
                    diag_s, unp_s = _pd_batch_base(ctx, "pd_ok_lambda_diag", r_up_ao, r_dn_ao)
            if bool(gem.time_reversal):
                tied_mo_sources = (gu_s, gd_s)
                flat["mo_coefficients_up"] = gu_s + np.conj(gd_s)
            else:
                flat["mo_coefficients_up"] = gu_s
                flat["mo_coefficients_dn"] = gd_s
            if isinstance(gem, AGPn_data):
                flat["lambda_diag"] = np.asarray(diag_s)
                if np.asarray(gem.lambda_unpaired).size > 0:
                    flat["lambda_unpaired"] = np.asarray(unp_s)
        else:
            if is_eL:
                v = _pd_batch_base(ctx, "pd_el_lambda", r_up_ao, r_dn_ao)
                if ecp_flag:
                    v = v + _pd_batch_ecp(ctx, "pd_el_ecp_lambda", r_up_ao, r_dn_ao, ecp_w, gp_b)
            else:
                v = _pd_batch_base(ctx, "pd_ok_lambda", r_up_ao, r_dn_ao)
            flat["lambda_matrix"] = np.asarray(v)

    # ---- Geminal AO radial basis-opt (GTO both spins) ----
    # Mirrors ``bridge._reduce_c_param_grads``: emit ``lambda_basis_exp`` /
    # ``lambda_basis_coeff`` (up+dn concatenated) whenever a basis block is optimised
    # AND both spin leaves are GTO. Each block is stored iff its own flag is active
    # (the run() downstream flag filter drops the other), matching the JAX stored set.
    _want_basis = mask.get("opt_lambda_basis_exp", False) or mask.get("opt_lambda_basis_coeff", False)
    if _want_basis and gem is not None and _gto_leaves_of(gem.orb_data_up_spin) and _gto_leaves_of(gem.orb_data_dn_spin):
        if is_eL:
            exp_s, coeff_s = _pd_batch_base(ctx, "pd_el_basis", r_up_ao, r_dn_ao)
            if ecp_flag:
                eex, eco = _pd_batch_ecp(ctx, "pd_el_ecp_basis", r_up_ao, r_dn_ao, ecp_w, gp_b)
                exp_s, coeff_s = np.asarray(exp_s) + np.asarray(eex), np.asarray(coeff_s) + np.asarray(eco)
        else:
            exp_s, coeff_s = _pd_batch_base(ctx, "pd_ok_basis", r_up_ao, r_dn_ao)
        if mask.get("opt_lambda_basis_exp", False):
            flat["lambda_basis_exp"] = np.asarray(exp_s)
        if mask.get("opt_lambda_basis_coeff", False):
            flat["lambda_basis_coeff"] = np.asarray(coeff_s)

    if source_blocks is not None:
        return _rs_c_param_derivative_sources(source_blocks, flat, tied_mo=tied_mo_sources)
    if complex_twist and isinstance(gem, AGPn_data):
        # Preserve the legacy diagnostic API's real-output convention for real
        # parameter leaves. Exact optimizer sources returned above retain Im.
        for name in ("lambda_diag", "lambda_unpaired"):
            if name in flat and not np.iscomplexobj(getattr(gem, name)):
                flat[name] = np.real(flat[name])
    return flat


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

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# separator
num_sep_line = 66


def _leaf_to_fp64(x: PytreeLeaf) -> PytreeLeaf:
    """Coerce a pytree array leaf to the Rust native sweep's fp64 contract.

    The Rust sweep's ``from_py`` extracts every floating dataclass leaf as
    ``PyReadonlyArray<f64>`` (and complex as ``Complex64``). Under a mixed-precision
    build zone the geminal / jastrow dataclasses carry float32 (/ complex64) leaves.
    Native CPU dispatch supports both precision modes; its transport boundary
    still requires widened arrays. Promote FLOATING -> float64 and COMPLEX ->
    complex128 (so complex-twist leaves stay complex, never flattened to real); leave
    integer / bool / structural leaves untouched, and PRESERVE the array backend
    (numpy stays numpy, jax stays jax) so the Rust extract -- which reads some leaves
    as a numpy ``PyReadonlyArray`` directly (no ``np.asarray`` wrapper) -- still sees
    the type it expects. Widening preserves the already-rounded values; it does
    not reconstruct full-precision input data. Native arithmetic precision is
    selected separately by the zone policy supplied to the run context.
    """
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            return x.astype(np.float64)
        if np.issubdtype(x.dtype, np.complexfloating):
            return x.astype(np.complex128)
        return x
    if isinstance(x, jax.Array):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            return x.astype(jnp.complex128)
        return x
    return x


def _native_swct_forces(structure, r_up, r_dn, hf_R, hf_up, hf_dn, pp_R, pp_up, pp_dn):
    """Assemble native HF/Pulay forces without a JAX SWCT or dense weight matrix.

    HF and Pulay share one fused native geometry pass per spin. Only the
    contracted six-component fields and summed weight derivatives cross the
    boundary; its scratch space never scales as atoms times electrons.
    """
    import jqmc_rs

    corrections = []
    for positions, hf, pp in ((r_up, hf_up, pp_up), (r_dn, hf_dn, pp_dn)):
        field = np.concatenate((np.asarray(hf), np.asarray(pp)), axis=-1)
        field = np.asarray(field, dtype=np.complex128 if np.iscomplexobj(field) else np.float64)
        corrections.append(jqmc_rs._swct_corrections_batched(structure, np.asarray(positions, dtype=np.float64), field))
    (up, domega_up), (dn, domega_dn) = corrections
    force_HF = hf_R + up[..., :3] + dn[..., :3]
    force_PP = pp_R + up[..., 3:] + dn[..., 3:] + 0.5 * (domega_up + domega_dn)
    return force_HF, force_PP


def _native_energy_chunk_route(
    *,
    complex_wf: bool,
    want_force: bool,
    want_stress: bool,
    want_param_deriv: bool,
    real_force_chunk_fused: bool,
) -> bool:
    """Whether the requested observables select the native whole-chunk MCMC route.

    Single source of truth for BOTH ``MCMC.run`` decisions that depend on it: the
    early JAX-init skip for complex chunks and the authoritative native chunk
    gate. Deriving both from this one predicate keeps them from disagreeing
    (an early skip that looked only at ``comput_position_deriv`` and
    diverged on complex-twist stress-only runs).

    Parameter derivatives always keep the per-step dispatch. Real-path chunks serve
    energy, stress and, when fused, force. Complex-twist chunks are energy-only:
    any requested force or stress keeps the per-step dispatch.
    """
    if want_param_deriv:
        return False
    if complex_wf:
        return not want_force and not want_stress
    return not want_force or real_force_chunk_fused


# ECP non-local quadrature base meshes keyed by ``Nv``.
# ``compute_coulomb_potential`` selects one of these by ``Nv`` and rotates its
# ``grid_points`` by the per-measurement ``RT`` (``grid_points @ RT``). The native
# ``jqmc_rs.compute_local_energy`` takes the ALREADY-rotated grid_points + weights
# directly (RT == eye in the ported ECP kernel tests), so the driver replays the
# same base-mesh + RT rotation the JAX ``compute_local_energy_fast`` applies.
_NV_BASE_MESH = {
    4: tetrahedron_sym_mesh_Nv4,
    6: octahedron_sym_mesh_Nv6,
    12: icosahedron_sym_mesh_Nv12,
    18: octahedron_sym_mesh_Nv18,
}


# MPI related
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def allreduce_sum(comm: "MPI.Comm", local: npt.NDArray) -> npt.NDArray:
    """``Allreduce`` over ``op=SUM`` that handles real *and* complex dtypes.

    Behavior by dtype:

    * ``float64`` / ``float32`` -- byte-identical to an in-place
      ``Allreduce([local, MPI.DOUBLE], [out, MPI.DOUBLE], op=MPI.SUM)`` call.
    * ``complex128`` -- reduce through the ``float64`` view of the buffer
      (size doubles to ``2N``); SUM is linear so ``Re`` and ``Im`` parts
      reduce independently and bit-exactly, and the result is reinterpreted
      back into ``complex128`` for the caller.

    The local buffer is not modified. Returns a freshly-allocated array with
    the same shape and dtype as ``local``, holding the SUM across ``comm``.
    """
    out = np.empty_like(local)
    dtype = local.dtype

    if dtype == np.complex128:
        # Bit-exact complex SUM via the real ``(2N,)`` view.
        local_view = np.ascontiguousarray(local).view(np.float64)
        out_view = out.view(np.float64)
        comm.Allreduce([local_view, MPI.DOUBLE], [out_view, MPI.DOUBLE], op=MPI.SUM)
        return out

    if dtype == np.float64:
        mpi_dtype = MPI.DOUBLE
    elif dtype == np.float32:
        mpi_dtype = MPI.FLOAT
    elif dtype == np.complex64:
        # This reduction accepts complex data only in complex128. Reject a
        # stray complex64 array loudly so it surfaces
        # as a clear error rather than a silent truncation.
        raise NotImplementedError(
            "allreduce_sum: complex64 buffers are forbidden by the precision "
            "policy (no fp32 complex). "
            "Promote to complex128 at the producer site."
        )
    else:
        raise TypeError(f"allreduce_sum: unsupported dtype {dtype}. Supported: float64, float32, complex128.")

    comm.Allreduce([local, mpi_dtype], [out, mpi_dtype], op=MPI.SUM)
    return out


class MCMC(OptimizationSampler):
    """Production VMC/MCMC driver with multiple walkers.

    This class drives Metropolis-Hastings sampling for many independent walkers in parallel
    (vectorized with ``jax.vmap``) and stores all observables needed by downstream analysis
    and optimization. All public methods are part of the supported API; private helpers are
    internal and subject to change.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int = 16,
        Dt: float = 2.0,
        epsilon_AS: float = 1e-1,
        comput_log_WF_param_deriv: bool = False,
        comput_e_L_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        comput_stress: bool = False,
        random_discretized_mesh: bool = True,
        use_swct: bool = True,
        mpc: bool = False,
        mpc_cutoff: float = MPC_CUTOFF_HA,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
        retain_derivative_sources: bool = True,
    ) -> None:
        """Build an MCMC driver and initialize walker state.

        Args:
            hamiltonian_data (Hamiltonian_data): Problem definition; ``sanity_check`` is called before sampling.
            mcmc_seed (int | tuple[int, ...], optional): Chain-seed key. The RNG stream is
                ``SeedSequence([*key_words, mpi_rank])`` (see
                ``jqmc._driver_helpers.derive_rank_rng_streams``); scalar for
                non-twist runs, ``(mcmc_seed, twist_index)`` from the twist
                drivers. Defaults to 34467.
            num_walkers (int, optional): Number of walkers on this rank. Defaults to 40.
            num_mcmc_per_measurement (int, optional): Proposals between observable evaluations. Defaults to 16.
            Dt (float, optional): Electron displacement scale (bohr). Defaults to 2.0.
            epsilon_AS (float, optional): Regularization exponent for antisymmetric stabilization. Defaults to 1e-1.
            comput_log_WF_param_deriv (bool, optional): Keep variational parameter derivatives (d ln Psi / dc). Defaults to False.
            comput_e_L_param_deriv (bool, optional): Keep local energy variational parameter derivatives (de_L / dc). Defaults to False.
            comput_position_deriv (bool, optional): Keep nuclear position derivatives. Defaults to False.
            comput_stress (bool, optional): Keep periodic stress observables. Defaults to False.
            random_discretized_mesh (bool, optional): Randomize quadrature mesh for non-local ECP terms. Defaults to True.
            use_swct (bool, optional): Apply Space Warp Coordinate Transformation to atomic forces. Defaults to True.
            mpc (bool, optional): Enable the passive MPC (Model Periodic Coulomb) finite-size
                correction. Under the ``physical = no`` convention MPC never enters the sampled
                Hamiltonian: propagation, ``e_L``, forces and every parameter gradient stay purely
                Ewald, and on each stored step the difference ``dV = V_ee^MPC - V_ee^Ewald`` is
                additionally evaluated on the SAME walker configurations (see
                :meth:`get_dE_mpc`). Requires a periodic ``hamiltonian_data``
                (``structure_data.pbc_flag = True``). Defaults to False.
            mpc_cutoff (float, optional): Reciprocal-space kinetic-energy cutoff (Ha) for the MPC
                ``f_G`` / density FFT grid; the only numerical knob of canonical MPC. Ignored when
                ``mpc`` is False. Defaults to ``MPC_CUTOFF_HA``.
            mpi_comm_in (MPI.Comm | None, optional): This twist's per-instance sampling
                / per-twist determinant communicator.
                ``None`` falls back to the module global ``mpi_comm`` (= ``COMM_WORLD``)
                so single-twist callers are byte-for-byte unchanged. The twist-averaged
                driver passes a distinct ``intra_comm`` per twist.
            world_comm_in (MPI.Comm | None, optional): All-rank comm used by the
                shared (Jastrow) variational block's two-comm reduction (section
                14.2.1). ``None`` falls back to ``mpi_comm_in`` so a single-twist
                run reduces shared and per-twist columns identically (byte-for-byte
                unchanged). The twist driver passes the all-rank ``world_comm``.
            retain_derivative_sources (bool, optional): Keep a separate copy of every exact
                parameter-derivative source. ``False`` stores a real ``direct`` source only once,
                as its identical folded entry (``dln_Psi_dc`` / ``de_L_dc``), and rebuilds it from
                there for ``optimization_batch``; complex ``d_p``/``d_pbar`` sources and blocks
                without a folded entry are still kept. Every result is unchanged. The CLI
                ``mcmc`` job uses ``False``. Defaults to True.

        Notes:
            - Seeds are folded with MPI rank and walker index to avoid correlation.
            - Initial electron configurations are placed near nuclei with balanced spins.
            - Logger prints walker and Hamiltonian diagnostics during initialization.
        """
        _validate_num_mcmc_per_measurement(num_mcmc_per_measurement)
        log_build_info_once(logger)  # Python API runs report their code version once (issue #183)
        # Per-instance communicator storage with
        # backward-compatible defaults. ``self._mpi_comm`` carries the per-
        # twist sampling/det comm; ``self._world_comm`` carries the all-rank
        # shared-Jastrow comm. When both args are None they fall back to the
        # module global ``mpi_comm`` so a single-twist call is byte-identical
        # to the single-communicator construction; the twist driver
        # passes both explicitly. Routing the class body through these
        # attributes keeps the per-twist sampling communicator separate from
        # the all-rank shared-parameter communicator.
        self._mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        self._world_comm = world_comm_in if world_comm_in is not None else self._mpi_comm
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_size = self._mpi_comm.Get_size()
        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        self.__comput_log_WF_param_deriv = comput_log_WF_param_deriv
        self.__comput_e_L_param_deriv = comput_e_L_param_deriv
        self.__retain_derivative_sources = bool(retain_derivative_sources)
        self.__comput_position_deriv = comput_position_deriv
        self.__comput_stress = comput_stress
        self.__random_discretized_mesh = random_discretized_mesh
        self.__use_swct = use_swct
        # MPC (physical = no): observable-only finite-size correction. The flag
        # must be set BEFORE the ``hamiltonian_data`` setter fires at the end of
        # ``__init__`` because ``__init_attributes`` reads it to decide whether
        # to build the (expensive) MPC tables.
        self.__mpc = mpc
        self.__mpc_cutoff = mpc_cutoff
        if self.__mpc and not hamiltonian_data.structure_data.pbc_flag:
            raise ValueError(
                "mpc=True requires a periodic hamiltonian_data (structure_data.pbc_flag=True). "
                "The MPC finite-size correction is defined only for a periodic cell."
            )

        # check sanity of hamiltonian_data
        hamiltonian_data.sanity_check()

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data
        self.__param_grad_flags = self.__default_param_grad_flags()

        # optimizer runtime container (method + hyperparameters)
        self.__optimizer_runtime = None
        self.__ensure_optimizer_runtime()

        # optimization counter
        self.__i_opt = 0

        # seeds -- hierarchical SeedSequence derivation. The single
        # derivation point is ``derive_rank_rng_streams``; the stream key is
        # ``[*chain_seed_words, mpi_rank]`` (scalar user seed for non-twist
        # runs, ``(mcmc_seed, twist_index)`` from the twist drivers).
        # ``__mpi_seed`` is the derived 63-bit JAX base-key integer;
        # ``__np_seed_words`` are the eight uint32 init_by_array words for
        # the NumPy legacy global RNG.
        self.__np_seed_words, self.__mpi_seed = derive_rank_rng_streams(self.__mcmc_seed, self._mpi_rank)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        # Use jax.random.split to obtain ``num_walkers`` independent PRNGKeys in
        # one batched call; a ``[fold_in(..., nw) for nw ...]`` Python loop
        # scales linearly with num_walkers and dominates init time at large
        # walker counts (e.g. nw = 16384).
        self.__jax_PRNG_key_list = jax.random.split(self.__jax_PRNG_key, self.__num_walkers)

        # initialize random seed (multi-word init_by_array path)
        np.random.seed(self.__np_seed_words)

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

        # check if only up electrons are updated
        if tot_num_electron_dn == 0:
            logger.info("  Only up electrons are updated in the MCMC.")
            self.__only_up_electron = True
        else:
            self.__only_up_electron = False

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords, cell
        )

        # Electron assignment for all atoms is complete.

        dtype_jnp = jnp.float64
        self.__latest_r_up_carts = jnp.asarray(r_carts_up, dtype=dtype_jnp)
        self.__latest_r_dn_carts = jnp.asarray(r_carts_dn, dtype=dtype_jnp)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {self._mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

        # init_attributes: the ``hamiltonian_data`` setter already calls
        # ``__init_attributes`` once. A second explicit call rebuilt the same
        # buffers and, with MPC on, repeated the expensive ``build_mpc_data``.
        self.hamiltonian_data = self.__hamiltonian_data

    def __init_attributes(self) -> None:
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves
        self.__accepted_moves = 0
        self.__rejected_moves = 0
        # Number of walker-steps whose carried inverse was refreshed
        # (diagnostic only; not persisted).
        self.__num_inverse_refreshes = 0

        # Compute dimensions for pre-shaped empty arrays
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        # The twist-regime flag is baked here, not in
        # ``__init__``) so it's re-derived on every state restoration --
        # ``MCMC.load_from_hdf5`` rebuilds the object via the
        # ``hamiltonian_data`` setter, which calls ``__init_attributes``
        # without re-running ``__init__``. The determinant owns the scalar
        # type, including final tiled providers that are not AO/MO leaves.
        self._wf_is_complex = np.issubdtype(
            self.__hamiltonian_data.wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating
        )

        # dtype for stored numpy arrays
        dtype_np = np.float64
        # e_L is complex at a non-Gamma twist;
        # ``__stored_e_L2`` stays real because it stores ``|e_L|^2``.
        _e_L_dtype = np.complex128 if self._wf_is_complex else dtype_np

        # stored weight (w_L)
        self.__stored_w_L = np.zeros((0, nw), dtype=dtype_np)

        # stored local energy (e_L)
        self.__stored_e_L = np.zeros((0, nw), dtype=_e_L_dtype)

        # stored local energy (e_L2)
        self.__stored_e_L2 = np.zeros((0, nw), dtype=dtype_np)

        # MPC (physical = no): tables + per-walker closure for the passive
        # observable dV = V_ee^MPC - V_ee^Ewald. ``build_mpc_data`` is expensive
        # (FFT extrapolation of g_G) so it is built HERE -- once per
        # ``hamiltonian_data`` -- and never inside the MCMC loop. The stored
        # buffer stays real float64 at any twist: dV is a difference of two real
        # e-e potentials and carries no e_L phase.
        if self.__mpc:
            self.__mpc_data = build_mpc_data(self.__hamiltonian_data, cutoff_ha=self.__mpc_cutoff, comm=self._mpi_comm)
            self.__jit_vmap_mpc_correction_ee = jit(vmap(compute_mpc_correction_ee, in_axes=(None, None, 0, 0)))
        else:
            self.__mpc_data = None
            self.__jit_vmap_mpc_correction_ee = None

        # stored MPC correction dV = V_ee^MPC - V_ee^Ewald (zero-width when mpc is off)
        self.__stored_mpc_dV = np.zeros((0, nw), dtype=dtype_np)

        # stored force_HF per walker (HF force = de_L/dR + Omega . de_L/dr)
        self.__stored_force_HF = np.zeros((0, nw, n_atoms, 3), dtype=dtype_np)

        # stored force_PP per walker (Pulay force = dln_Psi/dR + Omega . dln_Psi/dr + 1/2 * d_omega/dr)
        self.__stored_force_PP = np.zeros((0, nw, n_atoms, 3), dtype=dtype_np)

        # stored E_L * force_PP per walker (for covariance in Pulay force)
        # E_L * force_PP carries the complex e_L on twist k != 0;
        # the outer Re projection is taken at the binning-time aggregation in
        # ``get_aF``. Mirrors the GFMC ``__stored_E_L_force_PP`` pattern.
        self.__stored_E_L_force_PP = np.zeros((0, nw, n_atoms, 3), dtype=_e_L_dtype)

        # Per-walker Hellmann-Feynman stress tensor (Hartree, per step).
        # Buffer shape is ``(N_steps, N_walkers, 3, 3)`` regardless of pbc_flag
        # (molecular runs fill zeros without a kernel call --- see
        # ``MCMC.run``); this keeps downstream layouts uniform across periodic
        # and molecular paths.
        self.__stored_stress_HF = np.zeros((0, nw, 3, 3), dtype=dtype_np)

        # Per-walker Pulay stress tensor sigma_PP = dln|Psi|/deps
        # and the covariance bookkeeping term E_L * sigma_PP. Same shape +
        # short-circuit conventions as :pyattr:`__stored_stress_HF` (zeros on
        # the molecular path; estimator uses the cov(E_L, sigma_PP) form).
        self.__stored_stress_PP = np.zeros((0, nw, 3, 3), dtype=dtype_np)
        # E_L * stress_PP carries complex e_L; the binning reduction takes Re.
        self.__stored_E_L_stress_PP = np.zeros((0, nw, 3, 3), dtype=_e_L_dtype)

        # stored parameter gradients keyed by block name
        self.__stored_log_WF_param_grads: dict[str, list] = defaultdict(list)

        # stored local energy parameter gradients keyed by block name (de_L / dc)
        self.__stored_e_L_param_grads: dict[str, list] = defaultdict(list)

        # Exact derivative components. These coexist with the
        # original folded dictionaries above; only the private original adapter may
        # collapse them before the atomic real-coordinate solver cutover.
        self.__stored_log_WF_derivative_sources: dict[str, dict[str, list]] = {}
        self.__stored_e_L_derivative_sources: dict[str, dict[str, list]] = {}
        # Blocks whose exact source is a real ``direct`` component kept only as
        # the identical folded entry above (``retain_derivative_sources=False``).
        self.__log_WF_sources_in_param_grads: set[str] = set()
        self.__e_L_sources_in_param_grads: set[str] = set()

    def __store_derivative_sources(
        self,
        storage: DerivativeSourceStorage,
        folded_names: set[str],
        folded: dict[str, list],
        sources: Sequence[BlockDerivativeSource | DeviceBlockDerivativeSource],
        *,
        folded_stored: bool,
    ) -> None:
        """Keep this step's exact sources, or refer to the identical folded entry.

        With ``retain_derivative_sources=False`` a real ``direct`` source is not
        copied: the folded entry appended in the same step is ``np.real(direct)``,
        i.e. the same array, and ``_optimization_derivative_sources`` rebuilds
        ``direct`` from it. A step that cannot use the folded entry first
        restores the block's earlier steps into ``storage``.
        """
        retained = []
        for source in sources:
            name = source.name
            reusable = (
                not self.__retain_derivative_sources
                and folded_stored
                and self.__param_grad_flags.get(name, True)
                and not isinstance(source, DeviceBlockDerivativeSource)
                and source.d_p is None
                and source.d_pbar is None
                and source.direct is not None
                and not np.iscomplexobj(source.direct)
            )
            if reusable and (name in folded_names or (name not in storage and not folded.get(name))):
                folded_names.add(name)
                continue
            if name in folded_names:
                folded_names.discard(name)
                storage[name] = {"direct": list(folded[name])}
            retained.append(source)
        self.__append_derivative_sources(storage, retained)

    @staticmethod
    def __append_derivative_sources(
        storage: DerivativeSourceStorage,
        sources: Sequence[BlockDerivativeSource | DeviceBlockDerivativeSource],
    ) -> None:
        for source in sources:
            components = storage.setdefault(source.name, {})
            for component_name in ("direct", "d_p", "d_pbar"):
                values = getattr(source, component_name)
                if values is not None:
                    # A large MCMC executable already consumes most of GB200
                    # HBM.  Retaining O/D samples there leaves no workspace
                    # for any post-sampling JIT (H32 observed 135.10 GiB
                    # requests from both stack and dynamic-update kernels).
                    # Spill each completed slice to the large host memory;
                    # publication moves one completed component back to the
                    # device before the sharded optimizer begins.
                    if isinstance(source, DeviceBlockDerivativeSource):
                        stored = np.asarray(values)
                    elif isinstance(values, np.ndarray) and values.base is None and not values.flags.writeable:
                        # A built host source already owns an immutable copy
                        # of this step's slice; keep it instead of copying.
                        stored = values
                    else:
                        stored = np.array(values)
                    components.setdefault(component_name, []).append(stored)

    def __device_optimizer_enabled(self) -> bool:
        self.__ensure_optimizer_runtime()
        hyperparameters = self.__optimizer_runtime.get("hyperparameters") or {}
        return bool(hyperparameters.get("use_device_optimizer", False))

    @staticmethod
    def __published_derivative_sources(
        storage: DerivativeSourceStorage,
        blocks: Sequence[VariationalParameterBlock],
        num_warmup_steps: int = 0,
    ) -> tuple[BlockDerivativeSource, ...]:
        if not blocks:
            if storage:
                raise ValueError("derivative-source storage is nonempty but no variational blocks were published.")
            return ()
        sources = []
        for block in blocks:
            if block.name not in storage:
                raise ValueError(f"stored derivative sources are missing variational block {block.name!r}.")
            # Stack only the post-warmup steps into one new private buffer and
            # freeze it, so build() and every later validation share it
            # instead of copying it again.
            components = {}
            for name, values in storage[block.name].items():
                stacked = np.array(values[num_warmup_steps:])
                stacked.setflags(write=False)
                components[name] = stacked
            sources.append(BlockDerivativeSource.build(block=block, **components))
        return validate_derivative_sources(blocks, sources)

    def __validate_stored_shapes(self) -> None:
        """Assert that all stored observable arrays have consistent shapes."""
        ns = self.__mcmc_counter  # expected number of stored steps
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        expected = {
            "e_L": ((ns, nw), self.__stored_e_L),
            "e_L2": ((ns, nw), self.__stored_e_L2),
            "w_L": ((ns, nw), self.__stored_w_L),
        }
        if self.__mpc:
            expected["mpc_dV"] = ((ns, nw), self.__stored_mpc_dV)
        if self.__comput_position_deriv:
            expected.update(
                {
                    "force_HF": ((ns, nw, n_atoms, 3), self.__stored_force_HF),
                    "force_PP": ((ns, nw, n_atoms, 3), self.__stored_force_PP),
                    "E_L_force_PP": ((ns, nw, n_atoms, 3), self.__stored_E_L_force_PP),
                }
            )
        expected.update(
            {
                "stress_HF": (((ns if self.__comput_stress else 0), nw, 3, 3), self.__stored_stress_HF),
                "stress_PP": (((ns if self.__comput_stress else 0), nw, 3, 3), self.__stored_stress_PP),
                "E_L_stress_PP": (((ns if self.__comput_stress else 0), nw, 3, 3), self.__stored_E_L_stress_PP),
            }
        )

        for name, (shape, arr) in expected.items():
            _require(arr.shape == shape, f"stored shape mismatch: {name}.shape={arr.shape}, expected={shape}")

    @staticmethod
    def __default_param_grad_flags() -> dict[str, bool]:
        """Per-block gradient flags in their default state, keyed by block name.

        The default convention (parameter blocks True, "basis" blocks False) is NOT
        hardcoded here: it is derived from the single source
        ``jqmc.derivatives.bridge._MASK_DEFAULTS`` via the
        block-name -> ``opt_*`` keyword bridge ``_PARAM_BLOCK_TO_MASK_KW``. Adding a
        block to ``get_variational_blocks`` + the bridge auto-propagates its default.
        The ``j3_poly_*`` blocks are the DTN polynomial Jastrow channels (one per
        channel; each enabled polynomial family supplies one block).
        """
        from .derivatives.bridge import _MASK_DEFAULTS

        return {block: bool(_MASK_DEFAULTS[kw]) for block, kw in _PARAM_BLOCK_TO_MASK_KW.items()}

    def __param_gradient_flags(self) -> dict[str, bool]:
        """Return a copy of the current per-block gradient flags."""
        return dict(self.__param_grad_flags)

    def __set_param_gradient_flags(self, **flags: bool | None) -> None:
        """Update per-block gradient flags (True enables, False disables)."""
        allowed = set(self.__param_grad_flags)
        for name, value in flags.items():
            if name not in allowed:
                raise ValueError(f"Unknown variational block '{name}'.")
            if value is None:
                continue
            self.__param_grad_flags[name] = bool(value)

    def __needs_param_grad_mask(self) -> bool:
        return any(not enabled for enabled in self.__param_grad_flags.values())

    def __geminal_needs_active_param_path(self) -> bool:
        """True when a direct-MO-opt geminal (``SD_data`` / ``AGPn_data``) is being
        optimized, forcing the active-params grad closure regardless of masking.

        Direct-MO-opt geminals expose ``mo_coefficients_up`` (+ ``lambda_diag`` /
        ``lambda_unpaired`` for AGPn) as their variational blocks (see
        ``Wavefunction_data.get_variational_blocks``), whereas the full-wavefunction-
        pytree ``collect_param_grads`` path keys the geminal grad as the dense
        ``lambda_matrix`` -- correct ONLY for the original ``AGP_data``. When every block
        is active (no mask needed, e.g. an all-blocks-on optimization) the pytree path
        would otherwise be selected and (a) mis-key the geminal grad (block/grad-name
        mismatch -> the ``get_dln_WF`` size assertion), and (b) for a closed-shell
        singlet miss the ``C_dn = conj(C_up)`` tie that the active-params closure
        reconstructs. So take the active path whenever the geminal is optimized and is
        of a direct-MO-opt type. (Latent since the geminal class split; surfaced once
        the ``opt_JDTN_cutoffs`` block removal made a truly all-blocks-active run
        reachable.)"""
        if not self.__param_grad_flags.get("lambda_matrix", True):
            return False
        gem = getattr(self.__hamiltonian_data.wavefunction_data, "geminal_data", None)
        return type(gem).__name__ in ("SD_data", "AGPn_data")

    def __resolve_param_grad_mask_kwargs(self) -> dict[str, bool]:
        """Return the with_param_grad_mask kwargs dict for the current flag set.

        Pure-Python dict (compile-time constants) -- must be closed over by the
        in-trace grad closure so jit treats the flag values as static.

        Translates the block-name-keyed ``__param_grad_flags`` into the ``opt_*``
        keyword namespace via ``_PARAM_BLOCK_TO_MASK_KW``; an absent block falls back
        to the single-source default ``_MASK_DEFAULTS``. In
        practice ``__param_grad_flags`` always carries every block (initialised from
        ``__default_param_grad_flags``), so the fallback is a belt-and-braces default.
        """
        from .derivatives.bridge import _MASK_DEFAULTS

        flags = self.__param_grad_flags
        return {kw: bool(flags.get(block, _MASK_DEFAULTS[kw])) for block, kw in _PARAM_BLOCK_TO_MASK_KW.items()}

    # driver_config keys that ``save_to_hdf5`` always writes. A restart requires
    # every one of them (plus ``param_grad_flag_<block>`` for every block).
    _RESTART_CONFIG_KEYS = (
        "mcmc_seed",
        "num_walkers",
        "num_mcmc_per_measurement",
        "Dt",
        "epsilon_AS",
        "comput_log_WF_param_deriv",
        "comput_e_L_param_deriv",
        "retain_derivative_sources",
        "comput_position_deriv",
        "comput_stress",
        "random_discretized_mesh",
        "use_swct",
        "mpc",
        "mpc_cutoff",
        "mcmc_counter",
        "accepted_moves",
        "rejected_moves",
        "i_opt",
    )
    # Per-rank move statistics: the only restored scalars that may differ between ranks.
    _RESTART_PER_RANK_CONFIG_KEYS = frozenset({"accepted_moves", "rejected_moves"})

    @classmethod
    def __restart_state_problems(
        cls,
        cfg: dict[str, Any],
        rng: dict[str, Any],
        ws: dict[str, Any],
        obs: dict[str, Any],
        opt: dict[str, Any] | None,
        hamiltonian_data: Hamiltonian_data,
        filepath: str,
    ) -> dict[str, Any]:
        """This rank's missing keys, damaged state and to-be-agreed values for a restart."""
        from ._checkpoint import history_length_errors, missing_restart_keys, walker_shape_errors

        config_keys = cls._RESTART_CONFIG_KEYS + tuple(f"param_grad_flag_{k}" for k in cls.__default_param_grad_flags())
        missing = missing_restart_keys("driver_config", cfg, config_keys)
        missing += missing_restart_keys("rng_state", rng, ("jax_PRNG_key_list", "mpi_seed", "np_seed_words"))
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
        counter = cfg.get("mcmc_counter")
        if counter is not None:
            if type(counter) is not int or counter < 0:
                errors.append(("driver_config/mcmc_counter", f"must be a non-negative integer, got {counter!r}"))
            else:
                errors += history_length_errors(obs, ("e_L", "e_L2", "w_L"), counter, counter)
        # Folded derivative-source blocks must carry their folded derivatives.
        for prefix, obs_key in (
            ("log_derivative_source_in_param_grads_", "param_grads"),
            ("e_L_derivative_source_in_param_grads_", "e_L_param_grads"),
        ):
            folded = obs.get(obs_key, {})
            names = sorted(key[len(prefix) :] for key, value in cfg.items() if key.startswith(prefix) and bool(value))
            lacking = [name for name in names if not (isinstance(folded, dict) and np.size(folded.get(name, ())))]
            if lacking:
                errors.append((f"observables/{obs_key}", f"lacks the folded derivatives of source blocks {lacking}"))
        agreed = {key: value for key, value in cfg.items() if key not in cls._RESTART_PER_RANK_CONFIG_KEYS}
        agreed["optimizer_state"] = "absent" if opt is None else "present"
        if opt is not None:
            # An optimizer_state group always carries its lineage id and revision.
            lineage, revision = opt.get("wavefunction_lineage_id"), opt.get("wavefunction_revision")
            if not lineage:
                missing.append("optimizer_state/wavefunction_lineage_id")
            else:
                agreed["optimizer_state/wavefunction_lineage_id"] = lineage
            if revision is None:
                missing.append("optimizer_state/wavefunction_revision")
            else:
                agreed["optimizer_state/wavefunction_revision"] = revision
        return {"missing": missing, "errors": errors, "agreed": agreed}

    def __ensure_optimizer_runtime(self) -> None:
        # Reached together by every rank of the sampling communicator: a fresh
        # driver, or a restart whose ranks ALL lack optimizer_state (decided
        # collectively in ``load_from_hdf5``). A restored runtime always carries
        # its lineage id and revision, so no rank-local repair happens here.
        if getattr(self, "_MCMC__optimizer_runtime", None) is None:
            lineage_local = uuid.uuid4().hex if self._mpi_rank == 0 else None
            lineage_id = self._mpi_comm.bcast(lineage_local, root=0)
            self.__optimizer_runtime = {
                "method": None,
                "hyperparameters": None,
                "wavefunction_lineage_id": lineage_id,
                "wavefunction_revision": 0,
            }

    def __set_optimizer_runtime(
        self,
        *,
        method: str | None,
        hyperparameters: dict[str, Any] | None,
    ) -> None:
        self.__ensure_optimizer_runtime()
        hyper_copy = dict(hyperparameters) if hyperparameters is not None else None
        lineage_id = self.__optimizer_runtime["wavefunction_lineage_id"]
        revision = int(self.__optimizer_runtime["wavefunction_revision"])
        self.__optimizer_runtime = {
            "method": method,
            "hyperparameters": hyper_copy,
            "wavefunction_lineage_id": lineage_id,
            "wavefunction_revision": revision,
        }

    # ====================================================================== #
    # OptimizationSampler implementation
    # ====================================================================== #
    # This single section is the complete boundary consumed by
    # jqmc_optimizer.run_optimize. jqmc_optimizer imports no sampler class;
    # GFMC_t/GFMC_n can implement the same contract without inheriting MCMC.

    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Access the mutable Hamiltonian state optimized by this sampler."""
        return self.__hamiltonian_data

    @hamiltonian_data.setter
    def hamiltonian_data(self, hamiltonian_data: Hamiltonian_data) -> None:
        """Set the Hamiltonian state and reinitialize sampler storage."""
        self.__hamiltonian_data = hamiltonian_data
        self.__init_attributes()

    @property
    def comm(self) -> MPI.Comm:
        return self._mpi_comm

    @property
    def world_comm(self) -> MPI.Comm:
        return self._world_comm

    @property
    def rank(self) -> int:
        return self._mpi_rank

    @property
    def size(self) -> int:
        return self._mpi_size

    @property
    def optimization_step_index(self) -> int:
        return int(self.__i_opt)

    @property
    def wavefunction_state_token(self) -> tuple[str, int]:
        self.__ensure_optimizer_runtime()
        return (
            str(self.__optimizer_runtime["wavefunction_lineage_id"]),
            int(self.__optimizer_runtime["wavefunction_revision"]),
        )

    def advance_wavefunction_revision(self) -> None:
        self.__ensure_optimizer_runtime()
        self.__optimizer_runtime["wavefunction_revision"] = int(self.__optimizer_runtime["wavefunction_revision"]) + 1

    def derivative_capabilities(self) -> tuple[bool, bool]:
        return self.__comput_log_WF_param_deriv, self.__comput_e_L_param_deriv

    def configure_optimization(
        self,
        *,
        method: str,
        hyperparameters: dict[str, Any],
    ) -> None:
        self.__set_optimizer_runtime(method=method, hyperparameters=hyperparameters)

    def select_parameters(self, flags: dict[str, bool]) -> None:
        self.__set_param_gradient_flags(**flags)

    def sample(self, *, num_steps: int, max_time: int) -> None:
        self.run(num_mcmc_steps=num_steps, max_time=max_time)

    def estimate_energy(self, *, num_warmup_steps: int, num_bin_blocks: int) -> EnergyEstimate:
        values = self.get_E(
            num_mcmc_warmup_steps=num_warmup_steps,
            num_mcmc_bin_blocks=num_bin_blocks,
        )
        return EnergyEstimate(*values)

    def validate_wavefunction_candidate(self, candidate: Wavefunction_data) -> None:
        """Reject a non-finite log wavefunction on the current walkers before publication.

        This uses the candidate's actual image expansion and configured
        precision without changing accepted sampler state or native caches.
        A finite log magnitude is the condition: harmless amplitude overflow
        from a constant normalization is not a rejection reason.
        """
        values = jax.vmap(evaluate_ln_wavefunction, in_axes=(None, 0, 0))(
            candidate, self.__latest_r_up_carts, self.__latest_r_dn_carts
        )
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError("Candidate wavefunction has a non-finite log magnitude on current walkers.")

    def variational_blocks(self, parameters: ParameterSelection) -> list[VariationalParameterBlock]:
        return self.hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=self.wavefunction_state_token,
            **parameters.variational_block_kwargs(),
        )

    @staticmethod
    def __flatten_derivative_sources(
        blocks: Sequence[VariationalParameterBlock],
        sources: Sequence[BlockDerivativeSource],
    ) -> tuple[BlockDerivativeSource, ...]:
        flattened = []
        for block, source in zip(blocks, sources, strict=True):
            components = {}
            for name in ("direct", "d_p", "d_pbar"):
                values = getattr(source, name)
                if values is not None:
                    trailing = tuple(block.shape)
                    shape = (-1, *trailing) if trailing else (-1,)
                    components[name] = np.asarray(values).reshape(shape)
            flattened.append(BlockDerivativeSource.build(block=block, **components))
        return validate_derivative_sources(blocks, flattened)

    @staticmethod
    def __binned_source_moments(
        block: VariationalParameterBlock,
        source: BlockDerivativeSource,
        weights: npt.NDArray,
        local_energies: npt.NDArray,
        num_bin_blocks: int,
    ) -> JackknifeBlockMoments:
        trailing_axes = len(block.shape)
        expansion = (...,) + (None,) * trailing_axes
        weight_factor = weights[expansion]
        energy_factor = local_energies[expansion]

        def _bin_sums(weighted: np.ndarray) -> np.ndarray:
            splits = np.array_split(weighted, num_bin_blocks, axis=0)
            return np.concatenate([np.sum(part, axis=0) for part in splits], axis=0)

        def _multiply_owned(left: npt.ArrayLike, right: npt.ArrayLike, *, out: np.ndarray) -> np.ndarray:
            # Same operands and order as ``left * right``; reuse the private
            # sample-sized workspace ``out`` when the result dtype allows.
            if np.result_type(left, right) == out.dtype:
                return np.multiply(left, right, out=out)
            return np.multiply(left, right)

        sums: dict[str, np.ndarray] = {}
        conjugate_energy_sums: dict[str, np.ndarray] = {}
        for name in ("direct", "d_p", "d_pbar"):
            values = getattr(source, name)
            if values is not None:
                component = np.asarray(values)
                weighted = weight_factor * component
                sums[name] = _bin_sums(weighted)
                if np.iscomplexobj(component):
                    # (w * conj(O)) * e_L in one conjugated workspace.
                    del weighted
                    conjugated = np.conj(component)
                    conjugated = _multiply_owned(weight_factor, conjugated, out=conjugated)
                else:
                    # conj is the identity on a real source: w * conj(O) is
                    # the w * O workspace itself (same operands, same bits).
                    conjugated = weighted
                conjugated = _multiply_owned(conjugated, energy_factor, out=conjugated)
                conjugate_energy_sums[name] = _bin_sums(conjugated)
                del conjugated
        return JackknifeBlockMoments.build(
            block=block,
            source_sum=BlockDerivativeSource.build(block=block, **sums),
            conjugate_source_energy_sum=BlockDerivativeSource.build(block=block, **conjugate_energy_sums),
        )

    def __next_optimization_batch_token(self) -> str:
        if not hasattr(self, "_MCMC__batch_provider_identity"):
            identity_local = uuid.uuid4().hex if self._mpi_rank == 0 else None
            self.__batch_provider_identity = self._mpi_comm.bcast(identity_local, root=0)
            self.__batch_sequence = 0
        if self._mpi_rank == 0:
            next_sequence = int(self.__batch_sequence) + 1
            publication_local = (f"{self.__batch_provider_identity}:{next_sequence}", next_sequence)
        else:
            publication_local = None
        token, next_sequence = self._mpi_comm.bcast(publication_local, root=0)
        self.__batch_sequence = int(next_sequence)
        return str(token)

    def optimization_batch(
        self,
        *,
        num_warmup_steps: int,
        num_bin_blocks: int,
        blocks: list[VariationalParameterBlock],
        need_local_energy_derivatives: bool,
    ) -> OptimizationBatch | DeviceOptimizationBatch:
        if self.__device_optimizer_enabled():
            return self.__device_optimization_batch(
                num_warmup_steps=num_warmup_steps,
                num_bin_blocks=num_bin_blocks,
                blocks=blocks,
                need_local_energy_derivatives=need_local_energy_derivatives,
            )
        weights_grid = np.asarray(self.w_L[num_warmup_steps:])
        energies_grid = np.asarray(self.e_L[num_warmup_steps:])
        local_shape_valid = weights_grid.ndim == 2 and energies_grid.shape == weights_grid.shape
        if not self._mpi_comm.allreduce(local_shape_valid, op=MPI.LAND):
            raise ValueError("MCMC optimization samples must have matching (step, walker) weight/energy grids.")
        num_steps, num_walkers = weights_grid.shape
        local_block_valid = num_bin_blocks >= 1 and num_bin_blocks <= num_steps
        if not self._mpi_comm.allreduce(local_block_valid, op=MPI.LAND):
            raise ValueError("optimization jackknife step blocks must be between one and the number of sampled steps.")
        global_num_samples = self._mpi_comm.allreduce(num_steps * num_walkers, op=MPI.SUM)
        global_num_bins = self._mpi_comm.allreduce(num_bin_blocks * num_walkers, op=MPI.SUM)
        if global_num_samples < 2 or global_num_bins < 2:
            raise ValueError("optimization jackknife requires at least two global samples and two nonempty global bins.")

        log_sources_grid = self._optimization_derivative_sources(
            blocks=blocks,
            num_warmup_steps=num_warmup_steps,
            local_energy=False,
        )
        if need_local_energy_derivatives:
            energy_sources_grid = self._optimization_derivative_sources(
                blocks=blocks,
                num_warmup_steps=num_warmup_steps,
                local_energy=True,
            )
        else:
            energy_sources_grid = None

        weight_splits = np.array_split(weights_grid, num_bin_blocks, axis=0)
        energy_splits = np.array_split(weights_grid * energies_grid, num_bin_blocks, axis=0)
        bin_sample_counts = np.concatenate([np.full(num_walkers, part.shape[0], dtype=np.int64) for part in weight_splits])
        bin_weight_sums = np.concatenate([np.sum(part, axis=0) for part in weight_splits])
        bin_weighted_energy_sums = np.concatenate([np.sum(part, axis=0) for part in energy_splits])
        moments = tuple(
            self.__binned_source_moments(
                block,
                source,
                weights_grid,
                energies_grid,
                num_bin_blocks,
            )
            for block, source in zip(blocks, log_sources_grid, strict=True)
        )

        def _freeze(values: npt.ArrayLike) -> np.ndarray:
            frozen = np.array(values, copy=True)
            frozen.setflags(write=False)
            return frozen

        weights = _freeze(np.ravel(weights_grid))
        local_energies = _freeze(np.ravel(energies_grid))
        flattened_log_sources = self.__flatten_derivative_sources(blocks, log_sources_grid)
        flattened_energy_sources = (
            None if energy_sources_grid is None else self.__flatten_derivative_sources(blocks, energy_sources_grid)
        )
        bin_sample_counts = _freeze(bin_sample_counts)
        bin_weight_sums = _freeze(bin_weight_sums)
        bin_weighted_energy_sums = _freeze(bin_weighted_energy_sums)

        # Publish an identity only after every numeric payload has been built
        # and frozen.  Rank 0 owns the monotonic sequence and broadcasts the
        # complete opaque token over the sampler communicator.
        batch_token = self.__next_optimization_batch_token()
        metadata = JackknifeBinMetadata.build(
            blocks=blocks,
            batch_token=batch_token,
            bin_sample_counts=bin_sample_counts,
            bin_weight_sums=bin_weight_sums,
            bin_weighted_energy_sums=bin_weighted_energy_sums,
            log_source_moments=moments,
        )
        return OptimizationBatch.build(
            blocks=blocks,
            batch_token=batch_token,
            state_token=self.wavefunction_state_token,
            weights=weights,
            local_energies=local_energies,
            log_derivative_sources=flattened_log_sources,
            local_energy_derivative_sources=flattened_energy_sources,
            jackknife_bin_metadata=metadata,
        )

    def __device_published_derivative_sources(
        self,
        storage: DerivativeSourceStorage,
        blocks: Sequence[VariationalParameterBlock],
        num_warmup_steps: int,
        *,
        local_energy: bool,
    ) -> tuple[DeviceBlockDerivativeSource, ...]:
        if any(block.name not in storage for block in blocks):
            # Test/dummy samplers may override the established provider rather
            # than execute MCMC.run.  Preserve that structural boundary while
            # converting its exact source publication directly to devices.
            published = self._optimization_derivative_sources(
                blocks=blocks,
                num_warmup_steps=num_warmup_steps,
                local_energy=local_energy,
            )
            return tuple(
                DeviceBlockDerivativeSource.build_host(
                    block=block,
                    direct=source.direct,
                    d_p=source.d_p,
                    d_pbar=source.d_pbar,
                )
                for block, source in zip(blocks, published, strict=True)
            )
        sources = []
        for block in blocks:
            if block.name not in storage:
                raise ValueError(f"stored derivative sources are missing variational block {block.name!r}.")
            components = {}
            for name, values in storage[block.name].items():
                stacked_host = np.stack(values[num_warmup_steps:], axis=0)
                values.clear()
                components[name] = stacked_host
            # Both O and D remain host-staged.  Production moves only bounded
            # O chunks to the device while constructing resident SR chunks;
            # reduced LM already follows the same bounded transfer contract.
            payload = {name: components.get(name) for name in ("direct", "d_p", "d_pbar")}
            sources.append(DeviceBlockDerivativeSource.build_host(block=block, **payload))
        return tuple(sources)

    @staticmethod
    def __device_binned_source_moments(
        block: VariationalParameterBlock,
        source: DeviceBlockDerivativeSource,
        weights: jax.Array,
        local_energies: jax.Array,
        num_bin_blocks: int,
    ) -> DeviceJackknifeBlockMoments:
        routes = ({}, {})
        for name in ("direct", "d_p", "d_pbar"):
            values = getattr(source, name)
            if values is not None:
                source_sum, conjugate_source_energy_sum = _bounded_device_binned_source_component(
                    values,
                    weights,
                    local_energies,
                    num_bin_blocks,
                )
                routes[0][name] = source_sum
                routes[1][name] = conjugate_source_energy_sum
        return DeviceJackknifeBlockMoments(
            name=block.name,
            source_sum=DeviceBlockDerivativeSource.build(block=block, **routes[0]),
            conjugate_source_energy_sum=DeviceBlockDerivativeSource.build(block=block, **routes[1]),
        )

    def __device_optimization_batch(
        self,
        *,
        num_warmup_steps: int,
        num_bin_blocks: int,
        blocks: list[VariationalParameterBlock],
        need_local_energy_derivatives: bool,
    ) -> DeviceOptimizationBatch:
        """Publish host-staged full sources plus bounded device statistics."""
        weights_grid = jnp.asarray(self.w_L[num_warmup_steps:])
        energies_grid = jnp.asarray(self.e_L[num_warmup_steps:])
        if weights_grid.ndim != 2 or energies_grid.shape != weights_grid.shape:
            raise ValueError("MCMC optimization samples must have matching (step, walker) weight/energy grids.")
        num_steps, num_walkers = weights_grid.shape
        if not 1 <= num_bin_blocks <= num_steps:
            raise ValueError("optimization jackknife step blocks must be between one and the number of sampled steps.")
        log_sources_grid = self.__device_published_derivative_sources(
            self.__stored_log_WF_derivative_sources,
            blocks,
            num_warmup_steps,
            local_energy=False,
        )
        energy_sources_grid = (
            self.__device_published_derivative_sources(
                self.__stored_e_L_derivative_sources,
                blocks,
                num_warmup_steps,
                local_energy=True,
            )
            if need_local_energy_derivatives
            else None
        )
        weight_splits = jnp.array_split(weights_grid, num_bin_blocks, axis=0)
        energy_splits = jnp.array_split(weights_grid * energies_grid, num_bin_blocks, axis=0)
        counts = jnp.concatenate([jnp.full(num_walkers, part.shape[0], dtype=jnp.int64) for part in weight_splits])
        weight_sums = jnp.concatenate([jnp.sum(part, axis=0) for part in weight_splits])
        energy_sums = jnp.concatenate([jnp.sum(part, axis=0) for part in energy_splits])
        moments = tuple(
            self.__device_binned_source_moments(block, source, weights_grid, energies_grid, num_bin_blocks)
            for block, source in zip(blocks, log_sources_grid, strict=True)
        )
        batch_token = self.__next_optimization_batch_token()
        metadata = DeviceJackknifeBinMetadata(
            batch_token=batch_token,
            bin_sample_counts=counts,
            bin_weight_sums=weight_sums,
            bin_weighted_energy_sums=energy_sums,
            log_source_moments=moments,
        )

        def flatten_sources(
            sources: Sequence[DeviceBlockDerivativeSource] | None,
        ) -> tuple[DeviceBlockDerivativeSource, ...] | None:
            if sources is None:
                return None
            flattened = []
            for block, source in zip(blocks, sources, strict=True):
                components = {}
                for name in ("direct", "d_p", "d_pbar"):
                    values = getattr(source, name)
                    if values is not None:
                        components[name] = values.reshape((-1, *block.shape))
                payload = {name: components.get(name) for name in ("direct", "d_p", "d_pbar")}
                flattened.append(DeviceBlockDerivativeSource.build_host(block=block, **payload))
            return tuple(flattened)

        batch = DeviceOptimizationBatch(
            batch_token=batch_token,
            state_token=self.wavefunction_state_token,
            weights=weights_grid.reshape(-1),
            local_energies=energies_grid.reshape(-1),
            log_derivative_sources=flatten_sources(log_sources_grid),
            local_energy_derivative_sources=flatten_sources(energy_sources_grid),
            jackknife_bin_metadata=metadata,
        )
        # The stacked publication owns the production source payload.
        # Drop the per-step device references immediately so XLA can reuse
        # their buffers during coordinate transformation and SR centering.
        self.__stored_log_WF_derivative_sources = {}
        self.__stored_e_L_derivative_sources = {}
        self.__log_WF_sources_in_param_grads = set()
        self.__e_L_sources_in_param_grads = set()
        return batch

    def _optimization_derivative_sources(
        self,
        *,
        blocks: list[VariationalParameterBlock],
        num_warmup_steps: int = 0,
        local_energy: bool = False,
    ) -> tuple[BlockDerivativeSource, ...]:
        """Return raw derivative sources for the real-coordinate optimizer batch."""
        if local_energy:
            storage = self.__stored_e_L_derivative_sources
            folded_names, folded = self.__e_L_sources_in_param_grads, self.__stored_e_L_param_grads
        else:
            storage = self.__stored_log_WF_derivative_sources
            folded_names, folded = self.__log_WF_sources_in_param_grads, self.__stored_log_WF_param_grads
        if folded_names:
            storage = {**storage, **{name: {"direct": folded[name]} for name in folded_names}}
        return self.__published_derivative_sources(storage, blocks, num_warmup_steps)

    def advance_optimization_steps(self, count: int) -> None:
        self.__i_opt += int(count)

    # ======================== end OptimizationSampler ===================== #

    def run(self, num_mcmc_steps: int = 0, max_time: int = 86400) -> None:
        """Execute Metropolis-Hastings sampling for all walkers.

        Args:
            num_mcmc_steps (int, optional): Metropolis updates per walker; values <= 0 are no-ops. Defaults to 0.
            max_time (int, optional): Wall-clock budget in seconds. Defaults to 86400.

        Notes:
            - Creates ``external_control_mcmc.toml`` to allow external stop requests.
            - Accumulates energies, weights, forces, and wavefunction gradients into public buffers (``w_L``, ``e_L``, ``dln_Psi_*`` etc.).
            - Logs timing statistics and acceptance ratios at the end of the run.
        """
        _check_mpi4py_jax_distribution_consistency()

        # --- flag_rust_on_CPU dispatch -----------------------------------------
        # Resolve INLINE: the env override (if set) wins over the module flag;
        # the native sweep is used on CPU in both full and mixed precision once
        # ``jqmc_rs`` imports. GPU stays on the JAX vmap kernels. If the flag + CPU hold
        # but the extension is missing, raise rather than silently falling back
        # to JAX. Only the production ``MCMC`` branches; ``_MCMC_debug``
        # is untouched, so the prod-vs-debug equivalence tests validate the port.
        _env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
        if _env_rust is not None:
            _flag_rust = _env_rust.strip().lower() not in ("0", "false", "no", "off", "")
        else:
            _flag_rust = flag_rust_on_CPU
        rust_active = _flag_rust and jax.default_backend() == "cpu"
        if rust_active:
            try:
                import jqmc_rs
            except ImportError as _exc:
                raise RuntimeError(
                    "flag_rust_on_CPU is enabled on the CPU path but the "
                    "native extension 'jqmc_rs' could not be imported. Build it with:\n"
                    "    pip install -e .   (from the repository root; needs a Rust toolchain)\n"
                    "or set JQMC_FLAG_RUST_ON_CPU=0 to force the JAX path."
                ) from _exc
            # rayon build_global runs once per process, so the env override
            # (JQMC_RS_NUM_THREADS) can only take effect if applied before the
            # first batch kernel; idempotent across repeated run() calls.
            jqmc_rs.configure_threads(None)

        # MPC is a passive, energy-only observable (physical = no): it never
        # enters e_L, so the sampled/propagated local energy is always the plain
        # Ewald kernel. The MPC correction dV = V_ee^MPC - V_ee^Ewald is
        # accumulated separately on the stored walker batch (see the mpc_dV block
        # in the sampling loop) and reported via get_dE_mpc.
        _e_L_fast = _jit_vmap_e_L_fast

        # The c-axis (variational
        # parameter) grads route through the unified ``jqmc.derivatives``
        # interface (``analytic=False`` selects masked autograd). The
        # resolved ``opt_flags`` are the whole gradient whitelist -- passed to
        # the interface, which applies the mask / stop_gradient internally. The
        # driver keeps only the per-walker jit(vmap) wrapper + block-name
        # reducer. The closures are built by calling ``bridge.batched_grad_*``
        # directly with the axis/mask statics: the bridge's ``_build`` is
        # itself ``@lru_cache(None)`` on that same static key, so run_optimize's
        # repeated run() calls reuse the SAME compiled inner-jit object (no
        # per-step retrace + recompile), so no driver-side LRU wrapper is needed.
        from .derivatives.bridge import (
            _batched_c_source,
            _collect_c_param_derivative_sources,
            batched_grad_e_L,
            batched_grad_ln_Psi,
        )

        if self.__comput_log_WF_param_deriv or self.__comput_e_L_param_deriv:
            _c_mask_kwargs = self.__resolve_param_grad_mask_kwargs()
            _source_blocks = self.__hamiltonian_data.wavefunction_data.get_variational_blocks(
                state_token=self.wavefunction_state_token,
                **_c_mask_kwargs,
            )
            if self.__comput_log_WF_param_deriv:
                _grad_ln_psi_c = _batched_c_source(
                    "ln_Psi",
                    mask=_c_mask_kwargs,
                    with_geminal_inv=True,
                    analytic=_ANALYTIC_GRAD_ENABLED,
                )
            if self.__comput_e_L_param_deriv:
                _grad_e_L_c = _batched_c_source(
                    "e_L",
                    mask=_c_mask_kwargs,
                    with_geminal_inv=True,
                    analytic=_ANALYTIC_GRAD_ENABLED,
                )
        # The position (r/R) and strain (eps)
        # grads route through the same interface. Each closure is the per-walker
        # jit(vmap) wrapper of the single-walker ``compute_grad_*(..., analytic
        # =False)[axis]``; the r+R axes are bundled (one trace emits the fused
        # ``(dR, dr_up, dr_dn)`` tuple). Built directly from ``bridge`` so
        # run_optimize's repeated run() calls hit the bridge ``_build`` LRU (SAME
        # compiled inner-jit object, no per-step recompile).
        _vmc_fuse = bool(
            self.__comput_position_deriv and self.__comput_stress and self.__hamiltonian_data.structure_data.pbc_flag
        )
        _vmc_fused_stash: dict[str, jax.Array] = {}
        if self.__comput_position_deriv:
            _grad_e_L_rR = batched_grad_e_L(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
            _grad_ln_psi_rR = batched_grad_ln_Psi(wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
        if self.__comput_stress:
            _grad_e_L_eps = batched_grad_e_L(wrt="eps", with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
            _grad_ln_psi_eps = batched_grad_ln_Psi(wrt="eps", with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
        if _vmc_fuse:
            # On PBC force+stress runs the r/R and eps axes are
            # consumed in the SAME measurement step, so ONE fused program lets XLA CSE the
            # shared ECP non-local primal (_ecp_nl_common base/ratio/drift) between the R and
            # eps subgraphs.  The force site stashes the eps block for the same-step stress
            # consumer; closures are LRU-built (no compile until first call).
            _grad_e_L_rReps = batched_grad_e_L(wrt=("r", "R", "eps"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED)
            _grad_ln_psi_rReps = batched_grad_ln_Psi(
                wrt=("r", "R", "eps"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED
            )

        # timer_counter
        timer_mcmc_total = 0.0
        timer_mcmc_update_init = 0.0
        timer_mcmc_update = 0.0
        timer_e_L = 0.0
        timer_de_L_dR_dr = 0.0
        timer_dln_Psi_dR_dr = 0.0
        timer_dln_Psi_dc = 0.0
        timer_de_L_dc = 0.0
        timer_stress = 0.0
        timer_MPI_barrier = 0.0

        # mcmc timer starts
        self._mpi_comm.Barrier()
        mcmc_total_start = time.perf_counter()

        # toml(control) filename
        toml_filename = "external_control_mcmc.toml"

        # create the single per-invocation control file (world root owns it;
        # see _init_external_control_file for the twist-average rationale).
        _init_external_control_file(toml_filename, self._world_comm)
        self._mpi_comm.Barrier()

        # MCMC electron position update function
        mcmc_update_init_start = time.perf_counter()

        # All JIT kernels and vmap wrappers are defined at module level so
        # that repeated calls to run() (e.g. inside run_optimize) reuse the
        # same Python function objects and hit JAX's compilation cache instead
        # of triggering a full re-compilation each time.

        # The complex energy-only Rust chunk reconstructs G/G^-1 and both ratio
        # states internally, then returns their final values. Initialising the
        # same state through JAX is therefore dead work on that route. It is also
        # catastrophically large for B-splines: the nested electron/walker vmap
        # materialises the local K^3 coefficient gather for every walker at once.
        # Resolve this exact subset early (the authoritative chunk gate below
        # asserts the same decision) so no asynchronous JAX allocation can overlap
        # the fp64 Hamiltonian coercion and Rust context construction.
        # The requested-observable flags below are the ONLY ones the native
        # context and the authoritative chunk gate read; both gates go through
        # ``_native_energy_chunk_route`` so they cannot diverge.
        # ``want_stress`` is the explicit periodic-stress request; force-only
        # runs must not construct or invoke the stress kernels.
        _rs_want_force = bool(self.__comput_position_deriv)
        _rs_want_stress = bool(self.__comput_stress and self.__hamiltonian_data.structure_data.pbc_flag)
        _rs_want_param_deriv = bool(self.__comput_log_WF_param_deriv or self.__comput_e_L_param_deriv)
        _rs_complex_energy_chunk_early = bool(
            rust_active
            and self._wf_is_complex
            and _native_energy_chunk_route(
                complex_wf=True,
                want_force=_rs_want_force,
                want_stress=_rs_want_stress,
                want_param_deriv=_rs_want_param_deriv,
                # Read only on the real path; this mirror is complex-only.
                real_force_chunk_fused=False,
            )
        )
        if _rs_complex_energy_chunk_early:
            geminal = geminal_inv = None
            det_ratio_state_warmup = jastrow_state_warmup = None
        else:
            geminal, geminal_inv, _, _ = _geminal_inv_batched(
                self.__hamiltonian_data.wavefunction_data.geminal_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

            # Build the slim state used both by the JIT warm-up and by the real
            # chain. Warm-up calls do not advance the stored walker positions, so
            # rebuilding these same AO/J3/DTN states below only duplicated work.
            det_ratio_state_warmup = _jit_vmap_init_det_ratio_state(
                self.__hamiltonian_data.wavefunction_data.geminal_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )
            # Same throwaway-for-trace pattern for the Jastrow ratio sub-states.
            # The Jastrow layer decides which of its channels own a cache (the
            # dispatch there is Python-static, so the carry shape stays fixed
            # regardless of which terms eventually grow one); this driver just asks
            # for the container; channel membership remains owned by the Jastrow layer.
            jastrow_state_warmup = _jit_vmap_init_jastrow_ratio_state(
                self.__hamiltonian_data.wavefunction_data.jastrow_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
            )

        # A resumed walker can have finite stored energies but an invalid state
        # when the wavefunction is rebuilt from its restored position (the
        # observed mixed-precision escape failure). The backend-specific fresh
        # determinant and local-energy checks below reuse the state that the
        # production run already constructs. Fresh runs pay no extra work.
        _restart_validation_pending = bool(getattr(self, "_MCMC__restart_validation_pending", False))

        dtype_jnp = jnp.float64
        dtype_np = np.float64

        RTs = jnp.broadcast_to(jnp.eye(3, dtype=dtype_jnp), (len(self.__jax_PRNG_key_list), 3, 3))

        _restart_e_L = None
        if _restart_validation_pending and not rust_active:
            _restart_geminal = np.asarray(geminal)
            _restart_sign, _restart_logabsdet = np.linalg.slogdet(_restart_geminal)
            _restart_matrix_finite = np.all(np.isfinite(_restart_geminal), axis=(-2, -1))
            _restart_logabsdet = np.where(
                _restart_matrix_finite & np.isfinite(_restart_logabsdet) & (_restart_sign != 0),
                _restart_logabsdet,
                np.nan,
            )
            _restart_e_L = _e_L_fast(
                self.__hamiltonian_data,
                self.__latest_r_up_carts,
                self.__latest_r_dn_carts,
                RTs,
                geminal_inv,
                det_ratio_state_warmup,
            )
            _restart_e_L.block_until_ready()
            _raise_collective_nonfinite_observables(
                self._mpi_comm,
                driver="MCMC",
                stage="restart state rebuild",
                first_step=self.__mcmc_counter,
                observables={"ln_abs_det": _restart_logabsdet, "e_L": np.asarray(_restart_e_L)},
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            self.__restart_validation_pending = False

        # --- Native position-derivative coverage (warm-up skip) -------------------
        # When the native (Rust) path fully serves a position-derivative channel, the
        # matching JAX kernel (``_grad_e_L_rR`` / ``_grad_ln_psi_rR`` / ``_grad_e_L_eps``
        # / ``_grad_ln_psi_eps``) is NEVER called in the main loop (each is guarded by
        # ``not (rust_active and <native gate>)`` there). Its warm-up trace/compile is
        # then pure waste -- and for a DTN(+B-spline) determinant the strained-energy /
        # strained-log-Psi (stress) XLA compile alone peaks tens of GB per rank, which
        # can exhaust host memory in a 16-rank periodic B-spline force+stress run. So skip warming a JAX
        # position-deriv kernel the native path makes dead.
        #
        # Static (ham-only) mirror of the AUTHORITATIVE gates computed in the
        # rust_active statics block below (``_rs_hf_native`` / ``_rs_pp_native`` /
        # ``_rs_stress_{hf,pp}_native``), which that block harmlessly
        # recomputes for the main loop. Kept in sync with that block. Skipping a warm-up
        # call is CORRECTNESS-SAFE regardless of any drift: an over-eager skip only costs
        # a lazy first-use recompile, never a wrong result, because the authoritative
        # gate below -- not this mirror -- decides what the main loop actually calls.
        # Defaults False when the native path is off -> every JAX kernel is warmed as
        # before on JAX-only runs, where the JAX apparatus is the compute.
        _warm_hf_native = _warm_pp_native = False
        _warm_stress_hf_native = _warm_stress_pp_native = False
        if rust_active:
            if self.__comput_position_deriv:
                _warm_hf_native = True
                _warm_pp_native = True
            if self.__comput_stress and self.__hamiltonian_data.structure_data.pbc_flag:
                # Direct Gamma/twist pGTO+B-spline parity lives in test_vmc_stress.py.
                _warm_stress_hf_native = True
                _warm_stress_pp_native = True

        # Warm-up compilation: trigger JIT tracing on the first run() call
        # so that the MCMC loop does not stall on the first step.
        # On subsequent calls the JAX cache is already warm and these calls
        # return almost instantly, but we still skip them to avoid the overhead.
        if not getattr(self, "_MCMC__mcmc_kernels_warmed_up", False):
            logger.info("Start compilation of the MCMC_update funciton.")
            # The JAX update kernels are unused when the native sweep is active,
            # so skip their (slow) trace/compile. The e_L / gradient kernels
            # below stay JAX and are still warmed up.
            if not rust_active:
                if self.__only_up_electron:
                    _ = _jit_vmap_update_up(
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        self.__jax_PRNG_key_list,
                        self.__num_mcmc_per_measurement,
                        self.__hamiltonian_data,
                        self.__Dt,
                        self.__epsilon_AS,
                        geminal_inv,
                        geminal,
                        det_ratio_state_warmup,
                        jastrow_state_warmup,
                    )
                else:
                    _ = _jit_vmap_update(
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        self.__jax_PRNG_key_list,
                        self.__num_mcmc_per_measurement,
                        self.__hamiltonian_data,
                        self.__Dt,
                        self.__epsilon_AS,
                        geminal_inv,
                        geminal,
                        det_ratio_state_warmup,
                        jastrow_state_warmup,
                    )
            # The JAX ``_e_L_fast`` local-energy kernel is DEAD when the native
            # path serves e_L. Its authoritative main-loop gate is
            # ``not (rust_active and _rs_eL_native)``, and
            # ``_rs_eL_native`` is set UNCONDITIONALLY True inside the rust_active
            # statics block below: native e_L covers every real/complex/MPC
            # ham -- no e_L path falls back to JAX). So the static mirror is simply
            # ``rust_active``: skip warming ``_e_L_fast`` whenever the native sweep
            # is active. Skipping is CORRECTNESS-SAFE -- an over-eager skip only
            # costs a lazy first-use recompile, never a wrong result (same argument
            # as the position-deriv warm-up-skip gates above). Warmed as before on
            # the JAX-only path, where ``_e_L_fast`` IS the compute.
            if not rust_active and _restart_e_L is None:
                _ = _e_L_fast(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                    geminal_inv,
                    det_ratio_state_warmup,
                )
            if not _rs_complex_energy_chunk_early:
                _ = _jit_vmap_as_reg_fast(geminal, geminal_inv)
            # Lower the rotation-matrix generator now so the first measurement
            # step does not pay JIT compile time. Cost is sub-millisecond
            # compared to the stress / force traces but still worth pinning.
            if self.__random_discretized_mesh:
                _ = _jit_vmap_generate_RTs(self.__jax_PRNG_key_list)
            if self.__comput_position_deriv:
                # Skip the JAX HF / Pulay force warm-up when the native path serves it
                # (dead in the main loop otherwise -- see the warm-up-skip gates above).
                if not _warm_hf_native:
                    if _vmc_fuse:
                        # The main loop runs the fused r/R/eps program on PBC
                        # force+stress runs -- warm that one trace (it subsumes the eps
                        # warm below; warming both would pay the huge strained compile twice).
                        _, _, _, _ = _grad_e_L_rReps(
                            self.__hamiltonian_data,
                            self.__latest_r_up_carts,
                            self.__latest_r_dn_carts,
                            RTs,
                            geminal_inv,
                        )
                    else:
                        _, _, _ = _grad_e_L_rR(
                            self.__hamiltonian_data,
                            self.__latest_r_up_carts,
                            self.__latest_r_dn_carts,
                            RTs,
                            geminal_inv,
                        )
                if not _warm_pp_native:
                    if _vmc_fuse:
                        _, _, _, _ = _grad_ln_psi_rReps(
                            self.__hamiltonian_data,
                            self.__latest_r_up_carts,
                            self.__latest_r_dn_carts,
                            geminal_inv,
                        )
                    else:
                        _, _, _ = _grad_ln_psi_rR(
                            self.__hamiltonian_data,
                            self.__latest_r_up_carts,
                            self.__latest_r_dn_carts,
                            geminal_inv,
                        )

                if self.__use_swct and not rust_active:
                    _ = _jit_vmap_swct_omega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_up_carts,
                    )

                    _ = _jit_vmap_swct_omega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_dn_carts,
                    )

                    _ = _jit_vmap_swct_domega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_up_carts,
                    )

                    _ = _jit_vmap_swct_domega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_dn_carts,
                    )
            # HF/Pulay stress is independent of atomic force. Lower the eps-only
            # program for stress-only runs; force+stress may instead use the fused
            # r/R/eps program above.
            if self.__comput_stress and self.__hamiltonian_data.structure_data.pbc_flag:
                # Skip the JAX HF / Pulay STRESS warm-up when the native stress path
                # serves it (dead in the main loop otherwise). This strained-energy /
                # strained-log-Psi compile is the tens-of-GB-per-rank DTN(+B-spline)
                # peak that can exhaust host memory in a 16-rank periodic B-spline
                # force+stress run.
                # The fused r/R/eps warmup above already lowered the eps axis
                # (same one program the loop runs); the separate eps warms are only
                # needed on the non-fused (native-force asymmetric) configurations.
                if not _warm_stress_hf_native and not (_vmc_fuse and not _warm_hf_native):
                    _ = _grad_e_L_eps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        RTs,
                        geminal_inv,
                    )
                if not _warm_stress_pp_native and not (_vmc_fuse and not _warm_pp_native):
                    _ = _grad_ln_psi_eps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        geminal_inv,
                    )

            # The JAX c-axis param-deriv kernels are DEAD when the native path is
            # active: the main-loop gates are ``rust_active and _rs_okc_wanted`` /
            # ``rust_active and _rs_elc_wanted``, and inside the rust_active statics
            # block those are set UNCONDITIONALLY to the ``comput_*_param_deriv``
            # flags (complex gaps raise loudly up front, never fall back), so the
            # static mirror here is simply ``rust_active`` -- same argument as the
            # ``_e_L_fast`` skip above. This jacrev-scale trace/compile is the
            # tens-of-GB-per-rank warm-up peak on large systems (same class as the
            # strained-energy stress compile documented above).
            if self.__comput_log_WF_param_deriv and not rust_active:
                _ = _grad_ln_psi_c(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    geminal_inv,
                )

            if self.__comput_e_L_param_deriv and not rust_active:
                _ = _grad_e_L_c(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                    geminal_inv,
                )

            self.__mcmc_kernels_warmed_up = True
            self._mpi_comm.Barrier()
            mcmc_update_init_end = time.perf_counter()
            timer_mcmc_update_init += mcmc_update_init_end - mcmc_update_init_start
            logger.info("End compilation of the MCMC_update funciton.")
            logger.info(f"Elapsed Time = {mcmc_update_init_end - mcmc_update_init_start:.2f} sec.")
            logger.info("")
        else:
            self._mpi_comm.Barrier()
            logger.info("Skipping compilation (JAX cache is warm from previous run).")
            logger.info("")

        _log_memory_usage("mcmc_post_warmup", comm=self._mpi_comm)

        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        num_mcmc_done = 0
        mcmc_interval = max(1, int(num_mcmc_steps / 100))  # %

        # -- Extend stored arrays with zero-padding for new steps --
        nw = self.__num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom

        # Preserve the e_L storage dtype across
        # resizes (``complex128`` at non-Gamma twists, ``float64`` otherwise).
        _e_L_dtype_run = self.__stored_e_L.dtype
        self.__stored_e_L = np.concatenate([self.__stored_e_L, np.zeros((num_mcmc_steps, nw), dtype=_e_L_dtype_run)])
        self.__stored_e_L2 = np.concatenate([self.__stored_e_L2, np.zeros((num_mcmc_steps, nw), dtype=dtype_np)])
        self.__stored_w_L = np.concatenate([self.__stored_w_L, np.zeros((num_mcmc_steps, nw), dtype=dtype_np)])
        _finite_check_done = 0
        if self.__mpc:
            self.__stored_mpc_dV = np.concatenate([self.__stored_mpc_dV, np.zeros((num_mcmc_steps, nw), dtype=dtype_np)])
        if self.__comput_position_deriv:
            self.__stored_force_HF = np.concatenate(
                [self.__stored_force_HF, np.zeros((num_mcmc_steps, nw, n_atoms, 3), dtype=dtype_np)]
            )
            self.__stored_force_PP = np.concatenate(
                [self.__stored_force_PP, np.zeros((num_mcmc_steps, nw, n_atoms, 3), dtype=dtype_np)]
            )
            self.__stored_E_L_force_PP = np.concatenate(
                [self.__stored_E_L_force_PP, np.zeros((num_mcmc_steps, nw, n_atoms, 3), dtype=self.__stored_E_L_force_PP.dtype)]
            )
        if self.__comput_stress:
            self.__stored_stress_HF = np.concatenate(
                [self.__stored_stress_HF, np.zeros((num_mcmc_steps, nw, 3, 3), dtype=dtype_np)]
            )
            self.__stored_stress_PP = np.concatenate(
                [self.__stored_stress_PP, np.zeros((num_mcmc_steps, nw, 3, 3), dtype=dtype_np)]
            )
            self.__stored_E_L_stress_PP = np.concatenate(
                [self.__stored_E_L_stress_PP, np.zeros((num_mcmc_steps, nw, 3, 3), dtype=self.__stored_E_L_stress_PP.dtype)]
            )

        # The chain starts at exactly the same coordinates used above. Reuse
        # those already-materialized values instead of dispatching four
        # duplicate JAX initializers after the pre-compilation timer.
        det_ratio_state = det_ratio_state_warmup
        jastrow_state = jastrow_state_warmup

        # ENERGY whole-run chunk gate; set True inside the rust_active
        # block below when the run is energy-only + real-path. False elsewhere
        # (JAX path / force / param-deriv / complex) -> the per-step loop runs.
        _rs_energy_chunk = False

        # OPT-observable request flags: "did the
        # caller ask for O_k / dE_L/dc on the native path", nothing more. Set from the
        # ``comput_*_param_deriv`` constructor kwargs inside the rust_active block below;
        # False on the JAX path -> the per-step JAX derivatives interface runs. The two are
        # independent (the O_k and dE_L/dc driver kwargs are).
        #
        # Named ``_wanted``, NOT ``_native``: these flags carry no capability
        # decision (an unimplemented family raises at run() entry instead), and a
        # ``_native`` name would also read as a capability predicate to the
        # ``test_no_capability_gates`` matcher.
        _rs_okc_wanted = False
        _rs_elc_wanted = False

        # --- Native sweep statics ---------------------------------------------
        # Hoisted ONCE per run(): the Rust view of the Hamiltonian is invariant
        # for the whole MCMC chain (variational params change only between run()
        # calls -- run_optimize re-enters run() and re-hoists). The geminal /
        # jastrow dataclasses are passed straight through (their jnp leaves are
        # host-resident on CPU; ``from_py`` coerces via ``np.asarray`` on the
        # Rust side). ``geminal`` is complex128 at a
        # non-Gamma Bloch twist -> route to the complex sweep, which rebuilds
        # its streaming state internally (so we recompute the geminal trio fresh
        # from the swept positions rather than marshalling it back).
        if rust_active:
            # The Rust sweep is fp64-only: promote any fp32 (/ complex64) dataclass
            # leaves to float64 (/ complex128) before ``from_py``. Needed when the ham
            # was built under a mixed-precision zone but the run is full-precision (CLI
            # toml reconfigures precision -> rust_active). No-op in production (already
            # fp64). Complex leaves stay complex128 (complex-twist correctness).
            # The sweep-side Geminal/Jastrow/J3 are extracted once into the run
            # ctx (``SweepArgs`` inside ``build_obs_ctx``, from the same fp64-coerced
            # ``_rs_gem_data_f`` / ``_rs_jas_data_f`` below), so there is no per-step
            # ``tree_map(_leaf_to_fp64, ...)`` of the geminal/jastrow sub-objects.
            _rs_positions = np.asarray(self.__hamiltonian_data.structure_data.positions, dtype=dtype_np)
            if self.__hamiltonian_data.coulomb_potential_data.ecp_flag:
                _rs_charges = np.asarray(self.__hamiltonian_data.structure_data.atomic_numbers, dtype=dtype_np) - np.asarray(
                    self.__hamiltonian_data.coulomb_potential_data.z_cores, dtype=dtype_np
                )
            else:
                _rs_charges = np.asarray(self.__hamiltonian_data.structure_data.atomic_numbers, dtype=dtype_np)
            if self.__hamiltonian_data.structure_data.pbc_flag:
                _rs_cell = np.asarray(self.__hamiltonian_data.structure_data.cell, dtype=dtype_np)
                _rs_n_max = int(self.__hamiltonian_data.structure_data._n_max)
            else:
                _rs_cell = None
                _rs_n_max = 0
            # complex Bloch twist -> route to the complex sweep. Decide from the AO
            # TYPE (``self._wf_is_complex``, the recursion-safe detector set at
            # __init_attributes) rather than from ``np.iscomplexobj(geminal)``: the AO
            # type is the invariant that fixes which ``from_py`` / kernel the sweep
            # marshals into, whereas a geminal matrix value could in principle be
            # numerically real at a special twist and mis-route a complex AO leaf into
            # the real ``OrbData`` (whose real ``from_py`` then reads real-only attrs
            # the complex leaf lacks). The two agree for every physical twist here.
            _rs_complex = bool(self._wf_is_complex)

            # --- Native local-energy statics -----------------------------------
            # The observable e_L routes through ``jqmc_rs.compute_local_energy``
            # (1:1 ``compute_local_energy`` -- T + Ewald/bare-Coulomb + ECP local +
            # ECP non-local). It reads the WHOLE Hamiltonian object, so the fp32 ->
            # fp64 leaf coercion for mixed-built Hamiltonians must
            # cover EVERY float leaf, not just geminal/jastrow: coerce the entire
            # ``Hamiltonian_data`` once here (tree_map preserves int/bool/structural
            # leaves and the array backend -- ``_leaf_to_fp64``). This widens
            # mixed data for transport; the context separately receives the
            # current arithmetic zone policy. The native kernel is self-contained and
            # MPI-safe: its internal geminal inverse is a pure-Rust O(N^3) solve
            # (no XLA lowering -> no ``lapack_dgesdd_ffi`` custom-call), so the
            # MPI lowering trap is structurally avoided -- unlike the
            # JAX param-grad path it never emits LAPACK into an HLO graph.
            _rs_ham_eL = jax.tree_util.tree_map(_leaf_to_fp64, self.__hamiltonian_data)
            # ECP non-local base mesh (Nv_default); grid_points get RT-rotated per
            # measurement below. Weights are RT-invariant.
            _rs_one_move_mesh = _NV_BASE_MESH[Nv_default]
            _rs_ecp_w = np.asarray(_rs_one_move_mesh.weights, dtype=dtype_np)
            _rs_ecp_gp = np.asarray(_rs_one_move_mesh.grid_points, dtype=dtype_np)
            # e_L runs native on ``compute_local_energy_fast`` (real) /
            # ``compute_local_energy_fast_complex`` (complex-twist) -- T + Coulomb/
            # Ewald + ECP local/non-local, reusing the sweep SM inverse -- for every
            # REAL full-precision Hamiltonian (molecular + periodic real-Gamma) and
            # every COMPLEX-twist (k != Gamma) ham. Molecular is bit-identical to
            # jqmc at strict tol; periodic real-Gamma / complex-twist are bit-close
            # at strict tol (sphe / cart pGTO, pure PW, pure B-spline, composites).
            # The complex-twist kinetic reuses the running COMPLEX SM inverse and
            # evaluates grad^2 G with the batch complex AO, which home-cell-wraps
            # INTERNALLY (its WrapCtx) -- so it needs NO ``via_single`` (unlike the
            # real periodic path) and does not collapse far from the home cell. The
            # complex ECP non-local retains its imaginary component like the
            # XLA VMC forward: Im(e_L) == Im(T + V_nl). Fixed-phase projection
            # belongs only to the LRDMC candidate/sign-split boundary.
            # MPC is native too: the built ``MPC_data`` is
            # marshalled in below and the native ``compute_local_energy_fast`` /
            # ``compute_local_energy_fast_complex`` swap the Ewald e-e channel for the
            # finite-size-corrected ``E_mpc_ee`` (same kinetic SM-inverse + ECP pieces,
            # only the ee channel changes). So the native e_L is UNCONDITIONAL -- no
            # e_L path falls back to JAX. The walker chain is native in every case;
            # this flag only selects the e_L backend.
            _rs_eL_native = True
            # --- Native molecular Hellmann-Feynman force -----------------------
            # The HF force = dE_L/dR + Omega.dE_L/dr uses the VMC CONTINUOUS local-
            # energy gradient (``compute_grad_e_L`` wrt R/r). It is native for the
            # MOLECULAR case (AE + ECP) via ``vmc_grad_e_L_ae`` / ``vmc_grad_e_L_ecp``
            # -- the ``gfmc_diag_grads`` smooth base (dT+dV_bare+DTN kinetic) WITHOUT
            # the LRDMC S' discretization, + ECP local/non-local force -- reusing the
            # sweep SM ``geminal_inv`` (near-singular-stable, no fresh LU). Molecular
            # AE (cart/sphe) + ECP-cart are bit-identical to jqmc at strict tol.
            # The HF force and Pulay force are also native for the
            # PERIODIC real-Gamma (``vmc_grad_e_L_pbc`` / ``vmc_grad_ln_psi_pbc``) AND
            # COMPLEX-twist (``vmc_grad_e_L_pbc_complex`` / ``vmc_grad_ln_psi_pbc_complex``)
            # cases -- the ``gfmc_grad_e_l_ecp_pbc`` smooth base (periodic det+J kinetic +
            # Ewald ee/ei/ii + ECP local/non-local continuum) WITHOUT the LRDMC S'
            # discretization, reusing the sweep SM ``geminal_inv`` (real path; the complex
            # path rebuilds the complex inverse fresh, matching the e_L path). Periodic /
            # twist rows match jqmc at MEDIUM tol (the same periodic-kinetic sphe-pGTO
            # grad/lap fp-amplification floor the e_L carries, not a
            # wiring error: two independent native cores land at the same ~1e-6). Only
            # stress / O_k / dE_L_dc gradients stay JAX here.
            #
            # The force splits into HF (dE_L/dR, kinetic-involving) and Pulay
            # (dln|Psi|/dR, pure WF derivative). They are gated SEPARATELY:
            #
            # * ``_rs_pp_native`` (Pulay): native for MOLECULAR + periodic real-Gamma +
            #   complex-twist (pGTO AND pw / bspline / pgto(+)pw / pgto(+)bspline). The
            #   Pulay force is the WF drift (grad ln|Psi|, NOT grad^2), which matches the JAX
            #   oracle even for a far-roamed periodic walker (validated == JAX at far configs,
            #   ~1e-8). Complex non-pGTO leaves follow the same route.
            #
            # * ``_rs_hf_native`` (HF): native for MOLECULAR + periodic real-Gamma +
            #   complex-twist (pGTO + all non-pGTO leaves) -- the SAME coverage as the Pulay
            #   gate. The far-configuration determinant-kinetic
            #   grad^2 G BATCH-vs-via_single convention mismatch does NOT materialize
            #   (measured: jnone/j12 real AND twist HF force == JAX at ~1e-7 for walkers
            #   scattered up to several lattice vectors out, both ginv conventions -- the wrap
            #   inside ``gem_forward``/``pgto_deriv`` already keeps every det AO-derivative
            #   full-image-consistent). Two far-config conventions matter: (1) the J3
            #   drift/value AO is evaluated at the home-cell-wrapped coordinate
            #   (``compute_grads_and_laplacian_jastrow_three_body`` + ``jastrow_value``,
            #   jqmc ``compute_AOs`` convention; the RAW coordinate gives ~1e-2 in the
            #   far j123 HF force) and (2) the DTN force kernels use the min-image
            #   displacement (``forces_dtn`` via ``min_image_disp``, matching the DTN
            #   value/drift kernels; the RAW Euclidean displacement gives ~7e-1 in
            #   the far jdtn HF force).
            #
            #   Periodic real-Gamma genuinely-MO-wrapped J3 (num_mo !=
            #   num_ao, e.g. production lithium 1<-26) is native too -- ``j3_pbc_ctx_owned``
            #   lowers it to its exact AO-equivalent (``lower_j3_to_ao``: raw-AO orbital +
            #   ``M_ao = C^T M_mo C``), matching the molecular Pulay/HF path. Complex leaves:
            #   complex-twist non-pGTO leaves (pure PW / pure bspline / pgto(+)pw / pgto(+)bspline)
            #   carry the complex PW + B-spline electron-axis kinetic-force term
            #   (``complex_det_kinetic_grads`` + ``AosBsplineComplexData::det_egrad_fields``),
            #   so they need no separate gate.
            #
            _rs_pbc = bool(self.__hamiltonian_data.structure_data.pbc_flag)
            # The periodic-MPC HF force density-grid response
            # (``mpc_response::hf_force_dR_response``, part of the native
            # ``vmc_grad_e_l_pbc`` nuclear force) is the ~1.4e-2 dR channel that the
            # frozen-grid ``grad_mpc_ee_forces`` omits (electron AND nuclear axes), so
            # MPC HF force runs native.
            #
            # Complex-twist MPC position derivatives are native:
            # the complex HF-force core (``vmc_grad_e_L_pbc_complex``)
            # and the complex HF-stress core (``vmc_grad_e_L_eps_pbc_complex``) carry the MPC
            # channel -- ``grad_mpc_ee_forces`` / ``grad_mpc_ee_strain`` (WF-independent term1
            # + interp-origin, twist-agnostic) plus the twist-dependent density-grid-response
            # (``mpc_response::hf_force_dR_response_complex`` / ``mpc_stress_dgrid_response_complex``,
            # conj(psi) complex with Re taken LAST). Twist
            # MPC HF force + HF stress run native, threaded via ``_rs_mpc_data`` at the
            # dispatch sites below.
            _rs_hf_native = True
            _rs_pp_native = True
            # The position-deriv block is gated per channel by ``_rs_hf_native`` /
            # ``_rs_pp_native`` at the dispatch sites below; there is no combined
            # force gate.

            # --- Native VMC stress leaf support ------------------------------------
            # The real and complex stress aggregates assemble pGTO, PW and B-spline
            # kinetic-strain contributions additively.  This includes pure leaves,
            # pGTO+PW, and pGTO+B-spline composites; the B-spline value Pulay is zero
            # at fixed fractional coordinates.
            # MPC stress density-grid response (dgrid/deps) is native:
            # ``mpc_response::mpc_stress_dgrid_response`` supplies the
            # AO-at-grid affine density virial + the ``u_G(eps)`` reciprocal-strain + the
            # explicit ``dV`` terms, added onto the term1 pair virial inside the native stress
            # cores (``gfmc_stress_ecp_pbc_{jnone,withj}`` / ``vmc_grad_e_l_eps_pbc``).
            # MPC stress runs native through the ``_rs_mpc_data`` threaded to
            # ``grad_e_L_eps_pbc`` at the dispatch site.
            _rs_stress_hf_native = True
            _rs_stress_pp_native = True

            # --- Native optimization-observable capability ---
            # Native parameter derivatives fail loudly for an unsupported family
            # instead of silently routing an entire observable to JAX.
            # ``_RsComplexPdCtx`` exposes the same ``pd_*`` method surface as the
            # real context: DTN uses the shared-geminal native subset and the remaining
            # complex families use their free kernels. MPC's density-grid-response VJP
            # is threaded through ``_rs_native_c_param_grads(..., mpc_data=_rs_mpc_data)``
            # on the dE_L/dc geminal block; O_k is potential-free and therefore
            # MPC-blind. Radial O_k and
            # dE_L use the native basis path, while unsupported complex basis
            # optimization raises. Positions are wrapped to the same home-cell
            # representatives used by ``compute_AOs``. PW, B-spline, and pGTO
            # composites use their native leaf kernels.
            _rs_pd_want = bool(self.__comput_log_WF_param_deriv or self.__comput_e_L_param_deriv)
            # Complex-twist MPC dE_L/dc is native:
            # ``mpc_response::dele_dc_geminal_mo_complex`` (conj(psi), Re LAST) supplies the
            # geminal MO density response, threaded through ``_rs_native_c_param_grads(...,
            # mpc_data=_rs_mpc_data)``. O_k is potential-free and MPC-blind.
            if _rs_pd_want and _rs_complex:
                _rs_pd_gap = _rs_pd_complex_gap(
                    self.__hamiltonian_data,
                    _c_mask_kwargs,
                    ecp_flag=bool(self.__hamiltonian_data.coulomb_potential_data.ecp_flag),
                    want_e_L=bool(self.__comput_e_L_param_deriv),
                )
                if _rs_pd_gap is not None:
                    raise NotImplementedError(
                        f"flag_rust_on_CPU: complex-twist param-derivatives are native, EXCEPT {_rs_pd_gap}. "
                        "Disable flag_rust_on_CPU to use the JAX "
                        "param-deriv path for this family."
                    )
            # J3 AO-basis param-derivatives (``opt_J3_basis_exp`` / ``opt_J3_basis_coeff``)
            # are native, besides the GEMINAL (``lambda_basis_*``) radial kernels. The
            # J3 orbital's own radial derivative has both legs
            # -- ``param_grad_basis::grad_j3_radial`` (O_k, 1:1
            # ``basis.grad_J3_wrt_basis``), ``grad_j3_radial_kinetic`` (dT/dc, 1:1
            # ``basis.grad_T_wrt_J3_basis``) and ``param_grad_ecp::grad_ecp_j3_radial``
            # (the dV_ecp_nl/dc mesh add) -- reached through the ctx methods
            # ``pd_{ok,el}_j3_basis`` / ``pd_el_ecp_j3_basis``, so the assembler emits the
            # blocks and ``get_dln_WF`` sizes line up on both routes.
            _rs_okc_wanted = bool(self.__comput_log_WF_param_deriv)
            _rs_elc_wanted = bool(self.__comput_e_L_param_deriv)
            _rs_ecp_flag = bool(self.__hamiltonian_data.coulomb_potential_data.ecp_flag)
            _rs_gem_data_f = _rs_ham_eL.wavefunction_data.geminal_data
            _rs_jas_data_f = _rs_ham_eL.wavefunction_data.jastrow_data
            _rs_cp_data_f = _rs_ham_eL.coulomb_potential_data
            # MPC statics: build the ``MPC_data`` ONCE per run() (variational params
            # change only between run() calls -- re-hoisted here), from the same
            # fp64-coerced ham the native kernel reads. Built with the driver's
            # resolved grid_shape / eta so it is bit-identical to the JAX
            # ``_mpc_coulomb_energy`` build (which threads the same statics through
            # ``build_mpc_data``). ``None`` when MPC is off (plain Ewald/bare e_L).

            # --- Native observable context ------------------------------------------
            # Every per-walker native observable call (e_L, the vmc_grad_e_L_* force /
            # Pulay / stress cores) re-parsed the WHOLE Hamiltonian from Python
            # (GeminalData / JastrowData / EwaldData asarray+tolist, pgto_force_leaves,
            # every ECP channel array) -- 80% of the lithium HF-force call, paid AGAIN
            # in the Pulay + stress calls. The Hamiltonian is FIXED during a run (only
            # r_up/r_dn/geminal_inv/RT change per step), so extract it ONCE into owned
            # Rust structs here (re-hoisted per run() = per opt step, next to the ham
            # coercion above) and thread the handle through the per-step calls below.
            # Built from the SAME fp64-coerced sub-objects the wrappers read, so it is
            # bit-identical. ``want_stress`` is the explicit periodic-stress request;
            # force-only runs must not construct or invoke the stress kernels.
            # Build the force/Pulay Args only when this run keeps
            # position derivatives (``comput_position_deriv``); the stress Args need the
            # additional pGTO/composite leaf gate. An energy-only run thus skips the
            # ~4.8 s MolForceArgs/PbcForceArgs pgto_force_leaves extraction entirely.
            # ``_rs_want_force`` / ``_rs_want_stress`` are resolved ONCE above, next to
            # the early complex-chunk mirror, so the ctx and both chunk gates read the
            # same requested-observable flags.
            # Build ``ParamDerivArgs`` (the
            # once-parsed Geminal / Jastrow / ECP the c-axis O_k + dE_L/dc kernels reuse)
            # ONLY when this run keeps a native opt observable -- default off leaves the
            # existing energy/force/stress callers untouched (mirrors ``want_gfmc``).
            # ``_rs_want_param_deriv`` (== ``_rs_okc_wanted or _rs_elc_wanted``) is
            # likewise resolved once above.
            # Cross-run() cache: reuse the once-built ctx when the SAME ham object drives
            # a repeat run() (same requested observables). The optimizer swaps in a NEW
            # hamiltonian_data on a param update -> ``is`` fails -> rebuilt (invalidation
            # is automatic; a stale ctx cannot survive a param change). We hold a
            # reference to the ham so its id() cannot be recycled under us.
            _rs_precision_policy = precision_policy_snapshot()
            _rs_ctx_flags = (
                _rs_complex,
                _rs_pbc,
                _rs_want_force,
                _rs_want_stress,
                _rs_want_param_deriv,
                tuple(_rs_precision_policy.items()),
            )
            _rs_cached_ctx = getattr(self, "_MCMC__rs_run_ctx", None)
            if (
                _rs_cached_ctx is not None
                and getattr(self, "_MCMC__rs_run_ctx_ham", None) is self.__hamiltonian_data
                and getattr(self, "_MCMC__rs_run_ctx_flags", None) == _rs_ctx_flags
            ):
                _rs_obs_ctx = _rs_cached_ctx
            else:
                _rs_obs_ctx = jqmc_rs.build_obs_ctx(
                    _rs_gem_data_f,
                    _rs_jas_data_f,
                    _rs_cp_data_f,
                    _rs_complex,
                    _rs_pbc,
                    _rs_want_stress,
                    _rs_want_force,
                    False,
                    # The real path builds ParamDerivArgs.  The complex path builds the
                    # shared-geminal DTN subset; the adapter below keeps serving the other
                    # complex families through their parity-pinned free kernels.
                    _rs_want_param_deriv,
                    precision_policy=_rs_precision_policy,
                )
                self.__rs_run_ctx = _rs_obs_ctx
                self.__rs_run_ctx_ham = self.__hamiltonian_data
                self.__rs_run_ctx_flags = _rs_ctx_flags

            # c-axis param-deriv context: the native ObsCtx on the real path, the complex
            # adapter (shared native DTN + free kernels for other families) at a twist.
            # Both expose the same ``pd_*`` surface, keeping assembly single-sourced.
            _rs_pd_ctx = (
                _RsComplexPdCtx(_rs_gem_data_f, _rs_jas_data_f, _rs_cp_data_f, _rs_obs_ctx)
                if (_rs_want_param_deriv and _rs_complex)
                else _rs_obs_ctx
            )

            # --- Energy whole-run chunk delegation gate ----------------------------
            # Energy-only REAL-path runs delegate each ``mcmc_interval``-sized CHUNK of
            # the step loop to ONE native call (``run_energy_chunk``: sweep + e_L fused
            # in Rust), collapsing the per-step Python boundary (draw / marshal / store
            # / dispatch, ~2.5 ms/step on water) to per-chunk. Gated to: native e_L
            # (always native) and energy-only (no position / param derivatives -- those
            # keep the per-step dispatch when their per-walker gradient cores
            # still fire). The complex-twist energy-only path uses the same native RNG,
            # native e_L, and native AS-weight policy as the real production chunk.
            # Position-derivative observables are folded into the
            # SAME chunk loop so a FORCE run (``comput_position_deriv=True``) also earns the
            # whole-run boundary savings. The chunk computes the HF force + Pulay force (and,
            # for periodic, the HF + Pulay stress) per step in Rust and returns the raw grad
            # buffers; SWCT contractions are native too, while final force/stress
            # assembly stays on the host (below). The force chunk is gated to the
            # real path. Every force and stress leaf is native at this point; a
            # capability conjunction here would silently add a JAX fallback.
            # The force flag is a chunk-fusion switch, not a capability decision.
            _rs_obs_chunk_force_ok = self.__comput_position_deriv and _rs_hf_native and _rs_pp_native
            # Production has one native-chain policy: generate proposal normals, mesh
            # rotations, and AS weights inside Rust. Deterministic injected-RNG checks call
            # the Rust chunk entry points directly from tests; they are not a driver mode.
            # MPC (physical = no) is a PASSIVE energy-only observable: the chunk runs the
            # Ewald chain natively and additionally hands back the per-measurement configs
            # (``want_steps`` below), on which the driver evaluates ``dV_mpc`` AFTER the
            # chunk -- exactly the batch the per-step path used. MPC never reaches the
            # chunk's own e_L formula (no MPC table crosses the boundary), so it does not
            # force the per-step dispatch. The observable part of this gate (real path:
            # each requested observable serviceable independently, including force-only
            # and stress-only; complex twist: energy-only) is the SAME predicate the early
            # complex-chunk JAX-init skip above uses.
            _rs_energy_chunk = _rs_eL_native and _native_energy_chunk_route(
                complex_wf=_rs_complex,
                want_force=_rs_want_force,
                want_stress=_rs_want_stress,
                want_param_deriv=_rs_want_param_deriv,
                real_force_chunk_fused=bool(_rs_obs_chunk_force_ok),
            )
            if _rs_complex_energy_chunk_early != bool(_rs_energy_chunk and _rs_complex):
                # Explicit raise, not ``assert``: production validation must
                # survive ``python -O`` (raise-contract gate).
                raise RuntimeError("early complex-chunk JAX-init skip diverged from the authoritative native chunk gate")
            # Force/stress fold-in flags threaded to ``run_energy_chunk`` (below). Stress is
            # periodic-only (the ``pbc_shf``/``pbc_spp`` Args are built for the PBC family
            # only; molecular runs compute no stress -- the leaf gate is already folded in).
            # Complex chunks are energy-only, so both are False there.
            _rs_chunk_force = _rs_energy_chunk and _rs_want_force
            _rs_chunk_stress = _rs_energy_chunk and _rs_want_stress
            # Passive MPC observable on the chunk route: the chunk returns the per-measurement
            # configs (``want_steps``) so the driver can evaluate ``dV_mpc`` on the SAME batch,
            # at the SAME measurement granularity as the per-step path. ``want_force`` already
            # returns those configs, so ``want_steps`` only adds the force-off MPC case.
            _rs_chunk_mpc = _rs_energy_chunk and self.__mpc
            # Per-step (non-chunk) fast-mode proposal-normal RNG. Drawing the per-step
            # Gaussian-proposal normals via a JAX ``fori_loop`` threefry replay EVERY step
            # (``_jit_vmap_draw_injected_normals``) is the optimization-run bottleneck:
            # ~15 ms/step at 1 MPI rank growing to ~28 ms/step at 16 ranks (per-worker
            # XLA-accum contention), dwarfing the ~2 ms native sweep + <1 ms O_k assembly.
            # Per-step native routes (OPT / observables that do not use the chunk) draw
            # proposal normals with numpy from a generator seeded, at EVERY step, by the
            # rank and the CURRENT per-walker threefry keys (``_rs_step_normals`` below).
            # The keys are the only carried RNG state: they advance identically however
            # the run is split and are checkpointed, so ``run(a) + run(b)`` and a
            # save/merge/load continuation replay the uninterrupted chain (issue #135; a
            # generator seeded once per ``run()`` call would restart its stream at each call).
            _rs_rank_word = np.asarray([int(mpi_rank)], dtype=np.uint32)

            def _rs_step_normals(keys_u32: np.ndarray, num_moves: int) -> np.ndarray:
                seed = np.concatenate((_rs_rank_word, np.asarray(keys_u32, dtype=np.uint32).ravel()))
                return np.random.default_rng(seed).standard_normal((keys_u32.shape[0], num_moves))

            # Real native sweeps (per-step and energy chunk) advance the carried
            # geminal by rank-1 DELTAS against their own cached orbital columns, so
            # the seed must come from that same native evaluation. A JAX seed differs
            # from it at the ao_eval rounding level (~3e-7 of max|G| under the fp32
            # mixed zone); the delta updates keep that difference forever as an
            # absolute offset, and once an electron wanders far out its decayed row
            # is dominated by the offset: its acceptance ratios no longer follow
            # |Psi|^2 and the walker runs away. Real routes only: the
            # complex sweeps keep their own (unchanged) seeding.
            if not _rs_complex and geminal is not None:
                geminal = jnp.asarray(
                    _rs_obs_ctx.sweep_seed_geminal_batch(
                        np.asarray(self.__latest_r_up_carts, dtype=dtype_np),
                        np.asarray(self.__latest_r_dn_carts, dtype=dtype_np),
                        _rs_cell,
                        _rs_n_max,
                    ),
                    dtype=dtype_jnp,
                )
                geminal_inv = _geminal_pinv_from_matrix_batched(geminal)

            if _restart_validation_pending:
                _rs_restart_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                _rs_restart_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                if _rs_complex:
                    # The complex energy-chunk route intentionally skips the
                    # potentially huge JAX AO/B-spline seed. Build the same
                    # complex geminal natively, then form only its small batched
                    # pseudo-inverse for the one-time restart verdict.
                    _rs_restart_geminal = np.asarray(
                        _rs_obs_ctx.sweep_seed_geminal_complex_batch(
                            _rs_restart_r_up,
                            _rs_restart_r_dn,
                        )
                    )
                    _rs_restart_geminal_inv = np.asarray(_geminal_pinv_from_matrix_batched(jnp.asarray(_rs_restart_geminal)))
                else:
                    _rs_restart_geminal = np.asarray(geminal)
                    _rs_restart_geminal_inv = np.asarray(geminal_inv)
                _rs_restart_sign, _rs_restart_logabsdet = np.linalg.slogdet(_rs_restart_geminal)
                _rs_restart_matrix_finite = np.all(np.isfinite(_rs_restart_geminal), axis=(-2, -1))
                _rs_restart_logabsdet = np.where(
                    _rs_restart_matrix_finite & np.isfinite(_rs_restart_logabsdet) & (_rs_restart_sign != 0),
                    _rs_restart_logabsdet,
                    np.nan,
                )
                _rs_restart_grid_points = _rs_ecp_gp @ np.asarray(RTs, dtype=dtype_np)
                if _rs_complex:
                    _rs_restart_e_L = np.asarray(
                        _rs_obs_ctx.e_L_complex_batch(
                            _rs_restart_r_up,
                            _rs_restart_r_dn,
                            _rs_ecp_w,
                            _rs_restart_grid_points,
                            _rs_restart_geminal_inv,
                        )
                    )
                else:
                    _rs_restart_e_L = np.asarray(
                        _rs_obs_ctx.e_L_batch(
                            _rs_restart_r_up,
                            _rs_restart_r_dn,
                            _rs_ecp_w,
                            _rs_restart_grid_points,
                            _rs_restart_geminal_inv,
                        )
                    )
                _raise_collective_nonfinite_observables(
                    self._mpi_comm,
                    driver="MCMC",
                    stage="restart state rebuild",
                    first_step=self.__mcmc_counter,
                    observables={"ln_abs_det": _rs_restart_logabsdet, "e_L": _rs_restart_e_L},
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                self.__restart_validation_pending = False

            # Requested native observables either have a native route or raised above.
            # There is deliberately no per-observable JAX fallback in this block.

        mcmc_loop_start = time.perf_counter()
        if _rs_energy_chunk:
            # ================= Native energy-chunk loop ===========================
            # One ``run_energy_chunk`` call per walker advances a whole chunk of steps
            # (sweep + e_L threaded in Rust). w_L / e_L2 / storage / stop-flag /
            # max_time / progress are handled here at CHUNK granularity.
            # Empty test-injection arrays select native random-number and mesh
            # generation in ``run_energy_chunk_batch``. Batch-shaped: leading
            # walker axis so the batch entry's shape checks pass.
            _RS_EMPTY_NORMALS_B = np.zeros((nw, 0, 0), dtype=dtype_np)
            _RS_EMPTY_GRIDS_B = np.zeros((nw, 0, 0, 0), dtype=dtype_np)
            # A one-percent progress cadence is far too fine for the Rayon batch
            # path on small molecular jobs: for the standard 500-step benchmark
            # it launches and joins the whole walker pool 100 times for only five
            # sweeps per launch. Energy-only runs amortize that synchronization
            # over at most 500 sweeps. Force/MPC chunks return per-step position /
            # derivative buffers, so cap those at 50 to bound memory and stop-time
            # latency.
            _rs_chunk_cap = 50 if (_rs_chunk_force or _rs_chunk_mpc) else 500
            _rs_chunk_interval = min(_rs_chunk_cap, num_mcmc_steps)
            _chunk_step0 = 0
            while _chunk_step0 < num_mcmc_steps:
                progress = (_chunk_step0 + self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                mcmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: MCMC step = {_chunk_step0 + self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(mcmc_total_current - mcmc_loop_start):.1f} sec."
                )
                _clen = int(min(_rs_chunk_interval, num_mcmc_steps - _chunk_step0))

                start = time.perf_counter()
                _rs_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                _rs_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                if _rs_complex:
                    # Complex fast chunk: the native complex sweep recomputes the geminal
                    # trio internally from the swept positions (it takes no geminal input),
                    # so no complex geminal crosses the boundary. Pass unused (nw,1,1) real
                    # placeholders for the ``run_energy_chunk`` real ``geminal``/``geminal_inv``
                    # positional args (the ``complex=True`` branch ignores them).
                    _rs_gem = np.zeros((nw, 1, 1), dtype=dtype_np)
                    _rs_geminv = _rs_gem
                else:
                    _rs_gem = np.asarray(geminal, dtype=dtype_np)
                    _rs_geminv = np.asarray(geminal_inv, dtype=dtype_np)
                _rs_keys = np.asarray(self.__jax_PRNG_key_list, dtype=np.uint32)

                # ============ One native walker-batch chunk ==========================
                # The whole chunk is fanned over walkers INSIDE Rust (rayon), with no
                # ``for _w in range(nw)`` Python loop of per-walker ``run_energy_chunk``
                # calls. The batched dict carries a leading walker axis on every block; it is
                # unpacked below into the SAME per-walker aggregates the storage / SWCT-drift /
                # MPC assembly consumes -- byte-identical to a per-walker loop (pinned by
                # the thread-count bit-exact compare and ``test_obs_ctx_equivalence``,
                # which still drives the per-walker entry against the per-step kernels).
                # Normals, mesh rotations, and w_L are generated natively. The base mesh
                # is unrotated; Rust applies the per-step rotations internally.
                _res_batch = _rs_obs_ctx.run_energy_chunk_batch(
                    _rs_positions,
                    _rs_charges,
                    _rs_r_up,
                    _rs_r_dn,
                    _rs_gem,
                    _rs_geminv,
                    self.__Dt,
                    self.__epsilon_AS,
                    _rs_keys,
                    _RS_EMPTY_NORMALS_B,
                    _RS_EMPTY_GRIDS_B,
                    _rs_ecp_w,
                    self.__num_mcmc_per_measurement,
                    _rs_cell,
                    _rs_n_max,
                    native_rng=True,
                    random_mesh=bool(self.__random_discretized_mesh),
                    base_grid_points=_rs_ecp_gp,
                    n_steps_fast=_clen,
                    want_force=_rs_chunk_force,
                    want_stress=_rs_chunk_stress,
                    want_steps=_rs_chunk_mpc,
                    complex=_rs_complex,
                    # Per-step carried-inverse refresh inside the chunk
                    # (same probe residual / threshold as the per-step route).
                    inverse_refresh_threshold=float(_setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD),
                )
                self.__num_inverse_refreshes += int(np.sum(np.asarray(_res_batch["inverse_refreshes"])))

                # ---- batched unpack: walker-axis-leading blocks -> the per-walker aggregates
                # the storage / assembly below expects (cheap numpy slicing, no native calls) --
                _eL_ck = np.asarray(_res_batch["e_L"], dtype=_e_L_dtype_run).T  # (clen, nw)
                _ru_last = list(np.asarray(_res_batch["r_up"]))  # nw x (n_up, 3)
                _rd_last = list(np.asarray(_res_batch["r_dn"]))
                _key_last = list(np.asarray(_res_batch["final_key"]))  # nw x (2,)
                _ao_up_l = list(np.asarray(_res_batch["ao_up"]))
                _ao_dn_l = list(np.asarray(_res_batch["ao_dn"]))
                _pdn_l = list(np.asarray(_res_batch["paired_dn"]))
                _pul_l = list(np.asarray(_res_batch["paired_up_lambda"]))
                _acc_ck = int(np.sum(np.asarray(_res_batch["accepted"])))
                _wL_ck = np.asarray(_res_batch["w_L"], dtype=dtype_np).T  # (clen, nw)
                _gfin_l = list(np.asarray(_res_batch["geminal"]))  # nw x (n, m)
                _gifin_l = list(np.asarray(_res_batch["geminal_inv"]))
                if _rs_chunk_force:
                    # Raw HF/Pulay gradient buffers and per-step positions for the SWCT
                    # drift, per walker (clen, ...); stacked over walkers by the assembly below.
                    _hfR_l = list(np.asarray(_res_batch["hf_dR"]))  # nw x (clen, n_atoms, 3)
                    _hfru_l = list(np.asarray(_res_batch["hf_dr_up"]))  # nw x (clen, n_up, 3)
                    _hfrd_l = list(np.asarray(_res_batch["hf_dr_dn"]))  # nw x (clen, n_dn, 3)
                    _ppR_l = list(np.asarray(_res_batch["pp_dR"]))
                    _ppru_l = list(np.asarray(_res_batch["pp_dr_up"]))
                    _pprd_l = list(np.asarray(_res_batch["pp_dr_dn"]))
                    _rus_l = list(np.asarray(_res_batch["r_up_steps"]))  # nw x (clen, n_up, 3)
                    _rds_l = list(np.asarray(_res_batch["r_dn_steps"]))
                if _rs_chunk_stress:
                    _sHF_l = list(np.asarray(_res_batch["stress_HF"]))  # nw x (clen, 3, 3)
                    _sPP_l = list(np.asarray(_res_batch["stress_PP"]))
                if _rs_chunk_mpc:
                    # Passive MPC: per-walker per-measurement post-sweep configs (``want_steps``);
                    # ``dV_mpc`` is evaluated on these after the loop; the double loop
                    # below keeps its per-(step, walker) shape.
                    _mpc_rus_l = list(np.asarray(_res_batch["r_up_steps"]))  # nw x (clen, n_up, 3)
                    _mpc_rds_l = list(np.asarray(_res_batch["r_dn_steps"]))
                end = time.perf_counter()
                timer_mcmc_update += end - start

                _eL2_ck = np.abs(_eL_ck) ** 2
                _raise_collective_nonfinite_observables(
                    self._mpi_comm,
                    driver="MCMC",
                    stage="native energy chunk",
                    first_step=self.__mcmc_counter + num_mcmc_done,
                    observables={"e_L": _eL_ck, "e_L2": _eL2_ck, "w_L": _wL_ck},
                    r_up_carts=np.stack(_ru_last),
                    r_dn_carts=np.stack(_rd_last),
                )
                self.__restart_validation_pending = False

                # --- observable storage (byte-identical to the per-step path) --------
                start = time.perf_counter()
                _dst = self.__mcmc_counter + num_mcmc_done
                self.__stored_e_L[_dst : _dst + _clen] = _eL_ck
                self.__stored_e_L2[_dst : _dst + _clen] = _eL2_ck

                self.__stored_w_L[_dst : _dst + _clen] = _wL_ck
                # Complex-twist returns a complex final configuration; preserve it even
                # though the complex sweep reconstructs it from positions next chunk.
                _gfin_dtype = jnp.complex128 if _rs_complex else dtype_jnp
                geminal = jnp.asarray(np.stack(_gfin_l), dtype=_gfin_dtype)  # (nw, n, m)
                geminal_inv = jnp.asarray(np.stack(_gifin_l), dtype=_gfin_dtype)
                end = time.perf_counter()
                timer_e_L += end - start

                # --- MPC (physical = no): passive energy-only observable on the chunk route.
                # Evaluated on the SAME per-measurement Ewald-propagated configs the chunk
                # returned (``want_steps``), at the SAME (measurement, walker) granularity the
                # per-step block uses. dV_mpc = V_ee^MPC - V_ee^Ewald is NOT fed back into e_L,
                # the weights, acceptance, or any gradient -- only accumulated for get_dE_mpc.
                # The chunk is native-only, so this uses the native correction kernel (the
                # JAX/rust-off run stays on the per-step dispatch, which owns the JAX path).
                if _rs_chunk_mpc:
                    import jqmc_rs

                    _mpc_ewald_data = self.__hamiltonian_data.coulomb_potential_data.ewald_data
                    _mpc_rus = np.stack(_mpc_rus_l, axis=1)  # (clen, nw, n_up, 3)
                    _mpc_rds = np.stack(_mpc_rds_l, axis=1)  # (clen, nw, n_dn, 3)
                    _mpc_dV_ck = np.empty((_clen, nw), dtype=dtype_np)
                    for _i in range(_clen):
                        # Rayon walker batch: one native call per
                        # measurement over all nw walkers; bit-identical to the per-walker loop.
                        _mpc_dV_ck[_i] = jqmc_rs._compute_mpc_correction_ee_batch(
                            self.__mpc_data, _mpc_ewald_data, _mpc_rus[_i], _mpc_rds[_i]
                        )
                    self.__stored_mpc_dV[_dst : _dst + _clen] = _mpc_dV_ck

                # --- Position-derivative assembly (SWCT drift + host einsums) ---------
                # The chunk returned the raw per-step HF / Pulay grad buffers (and periodic
                # stress) in Rust; here we request native SWCT contractions and build the
                # force_HF / force_PP / stress host einsums -- the SAME math the per-step
                # path runs (below), just batched over the (step, walker) axes of this chunk.
                if _rs_chunk_force:
                    start = time.perf_counter()
                    _B = _clen * nw
                    # Stack per-walker (clen, ...) -> (clen, nw, ...) [step-major], then flatten
                    # the (step, walker) axes to one batch B for native SWCT.
                    _hfR = np.stack(_hfR_l, axis=1).reshape(_B, n_atoms, 3)
                    _hfru = np.stack(_hfru_l, axis=1)  # (clen, nw, n_up, 3)
                    _hfrd = np.stack(_hfrd_l, axis=1)
                    _ppR = np.stack(_ppR_l, axis=1).reshape(_B, n_atoms, 3)
                    _ppru = np.stack(_ppru_l, axis=1)
                    _pprd = np.stack(_pprd_l, axis=1)
                    _rus = np.stack(_rus_l, axis=1)  # (clen, nw, n_up, 3)
                    _rds = np.stack(_rds_l, axis=1)
                    _n_up = _rus.shape[2]
                    _n_dn = _rds.shape[2]
                    _ru_flat = _rus.reshape(_B, _n_up, 3)
                    _rd_flat = _rds.reshape(_B, _n_dn, 3)
                    _hfru = _hfru.reshape(_B, _n_up, 3)
                    _hfrd = _hfrd.reshape(_B, _n_dn, 3)
                    _ppru = _ppru.reshape(_B, _n_up, 3)
                    _pprd = _pprd.reshape(_B, _n_dn, 3)
                    if self.__use_swct:
                        _force_HF, _force_PP = _native_swct_forces(
                            self.__hamiltonian_data.structure_data,
                            _ru_flat,
                            _rd_flat,
                            _hfR,
                            _hfru,
                            _hfrd,
                            _ppR,
                            _ppru,
                            _pprd,
                        )
                        _force_HF = _force_HF.reshape(_clen, nw, n_atoms, 3)
                        _force_PP = _force_PP.reshape(_clen, nw, n_atoms, 3)
                    else:
                        # SWCT off: Omega/domega are exactly zero, so the einsum
                        # terms vanish identically. Skipping them (instead of
                        # contracting full-size zero (B, n_atoms, N_e) buffers)
                        # is a zero-risk deletion -- bit-identical result
                        # while preserving the same block order.
                        _force_HF = _hfR.reshape(_clen, nw, n_atoms, 3)
                        _force_PP = _ppR.reshape(_clen, nw, n_atoms, 3)
                    # E_L weight matches the per-step ``np.einsum("w,wjk->wjk", e_L, ...)`` (real
                    # for the real-Gamma chunk; the complex family never rides this path).
                    _eL_w = _eL_ck[:, :, None, None]  # (clen, nw, 1, 1)
                    self.__stored_force_HF[_dst : _dst + _clen] = _force_HF
                    self.__stored_force_PP[_dst : _dst + _clen] = _force_PP
                    self.__stored_E_L_force_PP[_dst : _dst + _clen] = _eL_w * _force_PP
                    end = time.perf_counter()
                    timer_de_L_dR_dr += end - start

                if _rs_chunk_stress:
                    start = time.perf_counter()
                    _eL_w = _eL_ck[:, :, None, None]
                    _stress_HF = np.stack(_sHF_l, axis=1)  # (clen, nw, 3, 3)
                    _stress_PP = np.stack(_sPP_l, axis=1)
                    self.__stored_stress_HF[_dst : _dst + _clen] = _stress_HF
                    self.__stored_stress_PP[_dst : _dst + _clen] = _stress_PP
                    self.__stored_E_L_stress_PP[_dst : _dst + _clen] = _eL_w * _stress_PP
                    end = time.perf_counter()
                    timer_de_L_dR_dr += end - start

                # --- advance driver state to the chunk's final config ----------------
                self.__latest_r_up_carts = jnp.asarray(np.stack(_ru_last), dtype=dtype_jnp)
                self.__latest_r_dn_carts = jnp.asarray(np.stack(_rd_last), dtype=dtype_jnp)
                self.__jax_PRNG_key_list = jnp.asarray(np.stack(_key_last).astype(np.uint32))
                # The chunk emits the sweep streaming state; complex-twist carries COMPLEX
                # AO / lambda leaves (``dtype_jnp`` is real -> use complex128 to preserve
                # them). This state seeds no further JAX ratio kernel in chunk mode (the
                # whole run is native), so it is carried only for shape/lifecycle uniformity.
                _det_dtype = jnp.complex128 if _rs_complex else dtype_jnp
                det_ratio_state = Det_ratio_streaming_state(
                    ao_up=jnp.asarray(np.stack(_ao_up_l), dtype=_det_dtype),
                    ao_dn=jnp.asarray(np.stack(_ao_dn_l), dtype=_det_dtype),
                    paired_dn=jnp.asarray(np.stack(_pdn_l), dtype=_det_dtype),
                    paired_up_lambda=jnp.asarray(np.stack(_pul_l), dtype=_det_dtype),
                )
                _total_moves_ck = _clen * nw * self.__num_mcmc_per_measurement
                self.__accepted_moves = self.__accepted_moves + _acc_ck
                self.__rejected_moves = self.__rejected_moves + (_total_moves_ck - _acc_ck)

                num_mcmc_done += _clen
                _chunk_step0 += _clen

                # max_time / stop_flag at chunk boundary (coarser than per-step).
                mcmc_current = time.perf_counter()
                should_stop = bool(max_time < mcmc_current - mcmc_total_start) if self._mpi_rank == 0 else False
                should_stop = self._mpi_comm.bcast(should_stop, root=0)
                if should_stop:
                    logger.info(f"  Stopping... max_time = {max_time} sec. exceeds.")
                    logger.info("  Break the mcmc loop.")
                    break
                stop_flag = _read_external_stop_flag(toml_filename, self._mpi_comm)
                if stop_flag:
                    logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                    logger.info("  Break the mcmc loop.")
                    break

        for i_mcmc_step in range(0) if _rs_energy_chunk else range(num_mcmc_steps):
            if i_mcmc_step % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                mcmc_total_current = time.perf_counter()
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(mcmc_total_current - mcmc_loop_start):.1f} sec."
                )

            # electron positions are goint to be updated!
            start = time.perf_counter()
            if rust_active:
                # Replace the JAX vmap update with the
                # native sweep -- one sweep per measurement runs
                # ``num_mcmc_per_measurement`` single-electron moves internally.
                # The Gaussian-proposal normals are pre-drawn here with numpy from
                # the step's current keys (``_rs_step_normals``; NOT the JAX
                # ``_draw_injected_normals`` replay, so the chain is not
                # byte-identical to the JAX path); Rust draws every other value
                # (electron/axis choice, Metropolis-b) bit-exactly from the same
                # threefry key words and returns the advanced key. Because the
                # normals are a pure function of the carried keys, split runs and
                # checkpoint continuations reproduce one run. Positions stay UNWRAPPED
                # across the boundary; the complex Bloch orbital wraps internally.
                # Walkers loop in Python (production runs nw=1/rank); the observable
                # e_L stays JAX on this per-step route.
                _rs_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                _rs_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                _rs_keys = np.asarray(self.__jax_PRNG_key_list, dtype=np.uint32)
                # One Gaussian per proposal per walker; the sweep draws the rest of the
                # move (choice/index/axis/Metropolis-b) from k0/k1 inside Rust.
                _rs_normals = _rs_step_normals(_rs_keys, self.__num_mcmc_per_measurement).astype(dtype_np)
                # Streaming geminal / inverse handed to the sweep: complex128 at a twist,
                # float64 on the real path. Both are the PREVIOUS measurement's swept
                # Sherman-Morrison state (near-singular-stable; no fresh LU per sweep).
                _sweep_dtype = np.complex128 if _rs_complex else dtype_np
                _rs_gem = np.asarray(geminal, dtype=_sweep_dtype)
                _rs_geminv = np.asarray(geminal_inv, dtype=_sweep_dtype)
                # The whole walker axis is fanned
                # inside Rust (``for_each_walker`` -> rayon) instead of this Python
                # ``for _w`` loop. ``*_batch`` is the exact per-walker sweep with a
                # leading walker axis on every array (r_up/r_dn/geminal/geminal_inv
                # ``(nw,..)``, keys ``(nw,2)``, normals ``(nw,num_moves)``); the
                # per-walker threefry key makes ``batch[w]`` bit-identical to a
                # per-walker call and thread-count independent (verified threads=1 vs 4).
                if _rs_complex:
                    _res = _rs_obs_ctx.mcmc_sweep_complex_batch(
                        _rs_positions,
                        _rs_charges,
                        _rs_r_up,
                        _rs_r_dn,
                        _rs_gem,
                        _rs_geminv,
                        self.__Dt,
                        self.__epsilon_AS,
                        _rs_keys,
                        _rs_normals,
                        _rs_cell,
                        _rs_n_max,
                    )
                else:
                    _res = _rs_obs_ctx.mcmc_sweep_batch(
                        _rs_positions,
                        _rs_charges,
                        _rs_r_up,
                        _rs_r_dn,
                        _rs_gem,
                        _rs_geminv,
                        self.__Dt,
                        self.__epsilon_AS,
                        _rs_keys,
                        _rs_normals,
                        _rs_cell,
                        _rs_n_max,
                    )
                _acc = np.asarray(_res["accepted"])
                accepted_moves_nw = _acc
                rejected_moves_nw = self.__num_mcmc_per_measurement - _acc
                self.__latest_r_up_carts = jnp.asarray(np.asarray(_res["r_up"]), dtype=dtype_jnp)
                self.__latest_r_dn_carts = jnp.asarray(np.asarray(_res["r_dn"]), dtype=dtype_jnp)
                self.__jax_PRNG_key_list = jnp.asarray(np.asarray(_res["final_key"]).astype(np.uint32))
                # Complex and real sweeps both emit the advanced streaming state
                # (geminal + Sherman-Morrison inverse + det-ratio tables); carry it
                # forward instead of rebuilding it from the swept positions. ``dtype_jnp``
                # is REAL, so the twist state must be kept complex128 (as the fast-chunk
                # path already does) or the Bloch phase would be silently truncated.
                _state_dtype = jnp.complex128 if _rs_complex else dtype_jnp
                geminal = jnp.asarray(np.asarray(_res["geminal"]), dtype=_state_dtype)
                geminal_inv = jnp.asarray(np.asarray(_res["geminal_inv"]), dtype=_state_dtype)
                det_ratio_state = Det_ratio_streaming_state(
                    ao_up=jnp.asarray(np.asarray(_res["ao_up"]), dtype=_state_dtype),
                    ao_dn=jnp.asarray(np.asarray(_res["ao_dn"]), dtype=_state_dtype),
                    paired_dn=jnp.asarray(np.asarray(_res["paired_dn"]), dtype=_state_dtype),
                    paired_up_lambda=jnp.asarray(np.asarray(_res["paired_up_lambda"]), dtype=_state_dtype),
                )
            elif self.__only_up_electron:
                (
                    accepted_moves_nw,
                    rejected_moves_nw,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    geminal_inv,
                    geminal,
                    det_ratio_state,
                    jastrow_state,
                ) = _jit_vmap_update_up(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                    det_ratio_state,
                    jastrow_state,
                )
            else:
                (
                    accepted_moves_nw,
                    rejected_moves_nw,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    geminal_inv,
                    geminal,
                    det_ratio_state,
                    jastrow_state,
                ) = _jit_vmap_update(
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    self.__jax_PRNG_key_list,
                    self.__num_mcmc_per_measurement,
                    self.__hamiltonian_data,
                    self.__Dt,
                    self.__epsilon_AS,
                    geminal_inv,
                    geminal,
                    det_ratio_state,
                    jastrow_state,
                )
            self.__latest_r_up_carts.block_until_ready()
            self.__latest_r_dn_carts.block_until_ready()
            # The carried Sherman--Morrison inverse feeds every consumer
            # below (fast e_L incl. Coulomb/ECP fast ratios, AS weight, forces,
            # stresses, parameter derivatives) AND the next sweep. Refresh it
            # from the same carried geminal for walkers whose O(N^2) probe
            # residual shows drift (XLA and native per-step sweeps alike).
            geminal_inv, _n_inv_refreshed = _refresh_running_geminal_inv(geminal, geminal_inv)
            self.__num_inverse_refreshes += _n_inv_refreshed
            end = time.perf_counter()
            timer_mcmc_update += end - start

            # store vmapped outcomes
            self.__accepted_moves = self.__accepted_moves + int(np.sum(np.asarray(accepted_moves_nw)))
            self.__rejected_moves = self.__rejected_moves + int(np.sum(np.asarray(rejected_moves_nw)))

            # generate rotation matrices (for non-local ECPs)
            if self.__random_discretized_mesh:
                RTs = _jit_vmap_generate_RTs(self.__jax_PRNG_key_list)
            else:
                RTs = jnp.broadcast_to(jnp.eye(3, dtype=dtype_jnp), (len(self.__jax_PRNG_key_list), 3, 3))

            # Evaluate observables each MCMC cycle
            start = time.perf_counter()
            if rust_active:
                # Native local energy on the
                # swept walkers. Per walker: rotate the ECP base mesh by this
                # measurement's RT (``grid_points @ RT``, matching
                # ``compute_coulomb_potential``) and call the aggregate native
                # kernel. Positions stay unwrapped across the boundary.
                # Reuse the sweep-emitted running geminal_inv (SM inverse) for the
                # native e_L exactly as the JAX path threads ``with_geminal_inv``:
                # the fresh-LU ``compute_local_energy`` NaNs at a near-singular
                # geminal (the AS-regularized SM inverse is stable), and reusing it
                # keeps the kinetic + ECP-nl bit-identical to
                # ``compute_local_energy_fast`` -- and avoids a redundant O(N^3)
                # solve and avoids lowering a fresh LAPACK inverse under MPI.
                _rs_RTs = np.asarray(RTs, dtype=dtype_np)
                _rs_r_up_eL = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                _rs_r_dn_eL = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                # At a complex twist the geminal_inv is complex128 and the
                # aggregate returns (re, im); keep the running COMPLEX SM inverse
                # (the fresh ``inverse_complex`` NaNs at a near-singular geminal).
                # Real path stays float64 -> scalar e_L.
                _rs_geminv_eL = np.asarray(geminal_inv, dtype=(np.complex128 if _rs_complex else dtype_np))
                # The walker axis is fanned inside
                # Rust (``for_each_walker``). The per-measurement ECP mesh rotation stays
                # in numpy so the grid fed to the kernel is byte-identical to the
                # per-walker ``_rs_ecp_gp @ _rs_RTs[_w]`` (batched matmul == per-slice,
                # verified). ``e_L[_batch]`` returns a ``(nw,)`` vector.
                _gp_all = _rs_ecp_gp @ _rs_RTs  # (nw, ng, 3)
                if _rs_complex:
                    # ``_rs_mpc_data`` is None off the MPC path -> the native kernel
                    # takes the plain Ewald e-e branch; Some(MPC_data) -> the Ewald e-e
                    # is swapped for the finite-size E_mpc_ee.
                    _e_L_arr = np.asarray(
                        _rs_obs_ctx.e_L_complex_batch(_rs_r_up_eL, _rs_r_dn_eL, _rs_ecp_w, _gp_all, _rs_geminv_eL)
                    )
                else:
                    _e_L_arr = np.asarray(_rs_obs_ctx.e_L_batch(_rs_r_up_eL, _rs_r_dn_eL, _rs_ecp_w, _gp_all, _rs_geminv_eL))
                e_L_step = jnp.asarray(_e_L_arr.astype(_e_L_dtype_run))
            else:
                e_L_step = _e_L_fast(
                    self.__hamiltonian_data,
                    self.__latest_r_up_carts,
                    self.__latest_r_dn_carts,
                    RTs,
                    geminal_inv,
                    det_ratio_state,
                )
                e_L_step.block_until_ready()
            end = time.perf_counter()
            timer_e_L += end - start

            self.__stored_e_L[self.__mcmc_counter + num_mcmc_done] = np.array(e_L_step)
            # The variance buffer stores ``|e_L|^2``
            # (real, non-negative). ``np.abs(e_L)**2 == e_L**2`` for a real
            # ``e_L`` so the Gamma real-energy path is byte-identical.
            self.__stored_e_L2[self.__mcmc_counter + num_mcmc_done] = np.array(np.abs(e_L_step) ** 2)

            # MPC (physical = no): passive observable evaluated on the SAME
            # configurations that were just propagated under Ewald. It is not
            # fed back into e_L, the weights, the acceptance, or any gradient --
            # only accumulated for ``get_dE_mpc``.
            if self.__mpc:
                _mpc_ewald_data = self.__hamiltonian_data.coulomb_potential_data.ewald_data
                if rust_active:
                    # Native dV = V_ee^MPC - V_ee^Ewald on the SAME Ewald-propagated
                    # walker batch. One native call per
                    # walker; the trajectory / e_L / weights are untouched.
                    import jqmc_rs

                    _mpc_ru = np.asarray(self.__latest_r_up_carts)
                    _mpc_rd = np.asarray(self.__latest_r_dn_carts)
                    # Rayon walker batch; bit-identical to the scalar loop.
                    mpc_dV_step = np.asarray(
                        jqmc_rs._compute_mpc_correction_ee_batch(self.__mpc_data, _mpc_ewald_data, _mpc_ru, _mpc_rd)
                    )
                else:
                    mpc_dV_step = self.__jit_vmap_mpc_correction_ee(
                        self.__mpc_data,
                        _mpc_ewald_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                    )
                self.__stored_mpc_dV[self.__mcmc_counter + num_mcmc_done] = np.array(mpc_dV_step)

            # AS weights
            R_AS_step = _jit_vmap_as_reg_fast(geminal, geminal_inv)
            w_L_step = _as_walker_weight(R_AS_step, self.__epsilon_AS)

            self.__stored_w_L[self.__mcmc_counter + num_mcmc_done] = np.array(w_L_step)

            # Inspect at the existing one-percent progress cadence instead of
            # adding one MPI collective to every measurement. A bad value is
            # never returned, published or checkpointed; failure can lag by at
            # most one progress interval. The helper gathers strings only on
            # failure and raises identically on every sampling rank.
            _completed_steps = num_mcmc_done + 1
            if _completed_steps % mcmc_interval == 0 or _completed_steps == num_mcmc_steps:
                _check_start = self.__mcmc_counter + _finite_check_done
                _check_stop = self.__mcmc_counter + _completed_steps
                _raise_collective_nonfinite_observables(
                    self._mpi_comm,
                    driver="MCMC",
                    stage="measurement",
                    first_step=_check_start,
                    observables={
                        "e_L": self.__stored_e_L[_check_start:_check_stop],
                        "e_L2": self.__stored_e_L2[_check_start:_check_stop],
                        "w_L": self.__stored_w_L[_check_start:_check_stop],
                    },
                    r_up_carts=self.__latest_r_up_carts,
                    r_dn_carts=self.__latest_r_dn_carts,
                )
                _finite_check_done = _completed_steps
                self.__restart_validation_pending = False

            if self.__comput_position_deriv:
                start = time.perf_counter()
                # de_L/dr_up, de_L/dr_dn (drift), and
                # de_L/dR (Hellmann-Feynman force) via the derivatives interface
                # (analytic=False), fused into one jit(vmap) trace. The "r" axis
                # differentiates w.r.t. r_up/r_dn only (so the ECP wf-ratio param
                # tape is not built); the "R" axis returns the per-atom (n_atom,
                # 3) force already summed by accumulate_position_grad inside the
                # interface (no host-side walk needed).
                if rust_active:
                    # Native VMC HF-force core per walker
                    # (molecular + periodic real-Gamma + complex-twist pGTO),
                    # reusing the sweep Sherman-Morrison ``geminal_inv`` (no redundant solve,
                    # no LAPACK inverse lowered under MPI, stable at a near-singular geminal;
                    # as for the native e_L above).
                    # Returns (de_L/dR (n_atom,3), de_L/dr_up (n_up,3), de_L/dr_dn
                    # (n_dn,3)); the ECP variant rotates the base mesh by this
                    # measurement's RT (``grid_points @ RT``, as the native e_L does)
                    # and threads the non-local weights/grid. Stress stays on JAX.
                    # Positions stay unwrapped.
                    _rsf_RTs = np.asarray(RTs, dtype=dtype_np)
                    _rsf_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                    _rsf_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                    # Running SM geminal_inv: complex128 at a twist, float64 otherwise.
                    # The complex force/Pulay cores reuse it for every base-config
                    # determinant field, exactly as the real ones do.
                    _rsf_geminv = np.asarray(geminal_inv, dtype=(np.complex128 if _rs_complex else dtype_np))
                    # The walker axis is fanned
                    # inside Rust. The ECP mesh rotation stays in numpy (batched matmul
                    # == the per-walker ``_rs_ecp_gp @ _rsf_RTs[_w]``, byte-exact);
                    # each ``*_batch`` returns ``(dR, dr_up, dr_dn)`` with a leading
                    # walker axis.
                    if _rs_complex:
                        # COMPLEX-twist periodic HF force (real, Re-projected). The
                        # sweep's running COMPLEX SM geminal_inv is reused for the base config.
                        _gp_all = _rs_ecp_gp @ _rsf_RTs
                        _dR_b, _dru_b, _drd_b = _rs_obs_ctx.grad_e_L_pbc_complex_batch(
                            _rsf_r_up, _rsf_r_dn, _rs_ecp_w, _gp_all, _rsf_geminv
                        )
                    elif _rs_pbc:
                        # PERIODIC real-Gamma HF force (AE bare Ewald or ECP), reusing
                        # the sweep SM geminal_inv. MPC swaps the Ewald e-e when active.
                        _gp_all = _rs_ecp_gp @ _rsf_RTs
                        _dR_b, _dru_b, _drd_b = _rs_obs_ctx.grad_e_L_pbc_batch(
                            _rsf_r_up, _rsf_r_dn, _rs_ecp_w, _gp_all, _rsf_geminv
                        )
                    elif _rs_ecp_flag:
                        _gp_all = _rs_ecp_gp @ _rsf_RTs
                        _dR_b, _dru_b, _drd_b = _rs_obs_ctx.grad_e_L_ecp_batch(
                            _rsf_r_up, _rsf_r_dn, _rs_ecp_w, _gp_all, _rsf_geminv
                        )
                    else:
                        _dR_b, _dru_b, _drd_b = _rs_obs_ctx.grad_e_L_ae_batch(_rsf_r_up, _rsf_r_dn, _rsf_geminv)
                    grad_e_L_R = np.asarray(_dR_b)
                    grad_e_L_r_up_step = np.asarray(_dru_b)
                    grad_e_L_r_dn_step = np.asarray(_drd_b)
                elif _vmc_fuse:
                    # In the fused force+stress program, the eps block rides the same
                    # trace (shared ECP non-local primal via CSE) and is stashed for the
                    # same-step stress consumer below.
                    grad_e_L_r_up_step, grad_e_L_r_dn_step, grad_e_L_R, _eps_hf = _grad_e_L_rReps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        RTs,
                        geminal_inv,
                    )
                    _vmc_fused_stash["stress_HF"] = _eps_hf
                    grad_e_L_r_up_step.block_until_ready()
                    grad_e_L_r_dn_step.block_until_ready()
                else:
                    grad_e_L_r_up_step, grad_e_L_r_dn_step, grad_e_L_R = _grad_e_L_rR(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        RTs,
                        geminal_inv,
                    )
                    grad_e_L_r_up_step.block_until_ready()
                    grad_e_L_r_dn_step.block_until_ready()
                end = time.perf_counter()
                timer_de_L_dR_dr += end - start

                start = time.perf_counter()
                logger.devel("    Evaluating dln_Psi/dR and dln_Psi/dr ...")
                # dln|Psi|/dR (Pulay, per-atom
                # accumulate) + dln|Psi|/dr drift via the derivatives interface
                # (analytic=False), fused into one trace (dR, dr_up, dr_dn).
                if rust_active:
                    # Native VMC Pulay wavefunction-gradient
                    # core per walker -- (dln|Psi|/dR (n_atom,3), dln|Psi|/dr_up
                    # (n_up,3), dln|Psi|/dr_dn (n_dn,3)) = the JAX ``_grad_ln_psi_rR``
                    # (``compute_grad_ln_Psi`` wrt R/r). ln|Psi| is LRDMC-independent
                    # so the SAME ``gfmc_grad_ln_psi`` building block serves VMC (no S'
                    # discretization here -- verified bit-exact vs JAX, unlike the HF
                    # e_L force which needed a VMC-specific aggregate). Reuses the sweep
                    # SM ``geminal_inv`` for the determinant Pulay + drift (the Jastrow
                    # J1/J2/J3/DTN pieces are inverse-free; the AO-basis J3 kernel
                    # already handles a production MO-wrapped J3 -- no as_raw_ao). The
                    # Pulay force has no ECP/mesh dependence (pure WF derivative), so AE
                    # and ECP share this one call. Periodic/twist Pulay stays JAX (later
                    # phase). Positions stay unwrapped.
                    _rsp_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                    _rsp_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                    # Running SM geminal_inv (complex128 at a twist, float64 otherwise);
                    # the complex Pulay/drift core reuses it for the base config.
                    _rsp_geminv = np.asarray(geminal_inv, dtype=(np.complex128 if _rs_complex else dtype_np))
                    # The walker axis is fanned
                    # inside Rust. Pure WF derivative (no ECP mesh) -> AE and ECP share
                    # one call; each ``*_batch`` returns ``(dR, dr_up, dr_dn)`` batched.
                    if _rs_complex:
                        # COMPLEX-twist Pulay (Re-projected holomorphic Pulay/drift).
                        _pR_b, _pru_b, _prd_b = _rs_obs_ctx.grad_ln_psi_pbc_complex_batch(_rsp_r_up, _rsp_r_dn, _rsp_geminv)
                    elif _rs_pbc:
                        # PERIODIC real-Gamma Pulay (det Pulay + J1/J3/DTN), reusing
                        # the sweep SM geminal_inv. AE and ECP share this call.
                        _pR_b, _pru_b, _prd_b = _rs_obs_ctx.grad_ln_psi_pbc_batch(_rsp_r_up, _rsp_r_dn, _rsp_geminv)
                    else:
                        _pR_b, _pru_b, _prd_b = _rs_obs_ctx.grad_ln_psi_batch(_rsp_r_up, _rsp_r_dn, _rsp_geminv)
                    grad_ln_Psi_dR = np.asarray(_pR_b)
                    grad_ln_Psi_r_up_step = np.asarray(_pru_b)
                    grad_ln_Psi_r_dn_step = np.asarray(_prd_b)
                elif _vmc_fuse:
                    # Fused Pulay force+stress trace; eps is stashed for the stress step.
                    grad_ln_Psi_dR, grad_ln_Psi_r_up_step, grad_ln_Psi_r_dn_step, _eps_pp = _grad_ln_psi_rReps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        geminal_inv,
                    )
                    _vmc_fused_stash["stress_PP"] = _eps_pp
                    grad_ln_Psi_r_up_step.block_until_ready()
                    grad_ln_Psi_r_dn_step.block_until_ready()
                else:
                    grad_ln_Psi_dR, grad_ln_Psi_r_up_step, grad_ln_Psi_r_dn_step = _grad_ln_psi_rR(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        geminal_inv,
                    )
                    grad_ln_Psi_r_up_step.block_until_ready()
                    grad_ln_Psi_r_dn_step.block_until_ready()
                end = time.perf_counter()
                timer_dln_Psi_dR_dr += end - start

                if self.__use_swct and not rust_active:
                    grad_omega_dr_up_step = _jit_vmap_swct_domega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_up_carts,
                    )
                    grad_omega_dr_dn_step = _jit_vmap_swct_domega(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_dn_carts,
                    )
                # Compute per-walker force products preserving cross-correlations
                _grad_e_L_r_up_np = np.array(grad_e_L_r_up_step)  # (nw, n_up, 3)
                _grad_e_L_r_dn_np = np.array(grad_e_L_r_dn_step)  # (nw, n_dn, 3)
                _grad_e_L_R_np = np.array(grad_e_L_R)  # (nw, n_atoms, 3)
                _grad_ln_Psi_r_up_np = np.array(grad_ln_Psi_r_up_step)  # (nw, n_up, 3)
                _grad_ln_Psi_r_dn_np = np.array(grad_ln_Psi_r_dn_step)  # (nw, n_dn, 3)
                _grad_ln_Psi_dR_np = np.array(grad_ln_Psi_dR)  # (nw, n_atoms, 3)

                if self.__use_swct and rust_active:
                    _force_HF, _force_PP = _native_swct_forces(
                        self.__hamiltonian_data.structure_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        _grad_e_L_R_np,
                        _grad_e_L_r_up_np,
                        _grad_e_L_r_dn_np,
                        _grad_ln_Psi_dR_np,
                        _grad_ln_Psi_r_up_np,
                        _grad_ln_Psi_r_dn_np,
                    )
                elif self.__use_swct:
                    # ``Omega . grad`` fused on device -- the (nw, n_atoms, N_e)
                    # weight matrix is never materialised (see the chunked block
                    # above and ``swct.evaluate_swct_omega_dot``).
                    _sd = self.__hamiltonian_data.structure_data
                    _ru_s, _rd_s = self.__latest_r_up_carts, self.__latest_r_dn_carts
                    _grad_omega_dr_up_np = np.array(grad_omega_dr_up_step)  # (nw, n_atoms, 3)
                    _grad_omega_dr_dn_np = np.array(grad_omega_dr_dn_step)  # (nw, n_atoms, 3)

                    # force_HF = de_L/dR + Omega_up . de_L/dr_up + Omega_dn . de_L/dr_dn
                    _force_HF = (
                        _grad_e_L_R_np
                        + np.asarray(_jit_vmap_swct_omega_dot(_sd, _ru_s, _grad_e_L_r_up_np))
                        + np.asarray(_jit_vmap_swct_omega_dot(_sd, _rd_s, _grad_e_L_r_dn_np))
                    )  # (nw, n_atoms, 3)

                    # force_PP = dln_Psi/dR + Omega . dln_Psi/dr + 1/2 * sum_i d_omega/d_r_i
                    _force_PP = (
                        _grad_ln_Psi_dR_np
                        + np.asarray(_jit_vmap_swct_omega_dot(_sd, _ru_s, _grad_ln_Psi_r_up_np))
                        + np.asarray(_jit_vmap_swct_omega_dot(_sd, _rd_s, _grad_ln_Psi_r_dn_np))
                        + 0.5 * (_grad_omega_dr_up_np + _grad_omega_dr_dn_np)
                    )  # (nw, n_atoms, 3)
                else:
                    # SWCT off: Omega/domega are exactly zero, so the einsum terms
                    # vanish identically. Skipping them (instead of contracting
                    # full-size zero (nw, n_atoms, N_e) buffers) is a zero-risk
                    # deletion while preserving block order and values.
                    _force_HF = _grad_e_L_R_np  # (nw, n_atoms, 3)
                    _force_PP = _grad_ln_Psi_dR_np  # (nw, n_atoms, 3)

                _e_L_np = np.array(e_L_step)  # (nw,)
                _E_L_force_PP = np.einsum("w,wjk->wjk", _e_L_np, _force_PP)  # (nw, n_atoms, 3)

                self.__stored_force_HF[self.__mcmc_counter + num_mcmc_done] = _force_HF
                self.__stored_force_PP[self.__mcmc_counter + num_mcmc_done] = _force_PP
                self.__stored_E_L_force_PP[self.__mcmc_counter + num_mcmc_done] = _E_L_force_PP

            # Per-walker Hellmann-Feynman and Pulay stress tensors are
            # independent of the atomic-force request.
            if self.__comput_stress and self.__hamiltonian_data.structure_data.pbc_flag:
                # de_L/deps (HF stress) and dln|Psi|/deps (Pulay stress) via the
                # derivatives interface. The force+stress XLA route may already
                # have produced these values in its fused r/R/eps program.
                start = time.perf_counter()
                if rust_active:
                    _rss_RTs = np.asarray(RTs, dtype=dtype_np)
                    _rss_r_up = np.asarray(self.__latest_r_up_carts, dtype=dtype_np)
                    _rss_r_dn = np.asarray(self.__latest_r_dn_carts, dtype=dtype_np)
                    _rss_geminv = np.asarray(geminal_inv, dtype=(np.complex128 if _rs_complex else dtype_np))
                    _gp_all = _rs_ecp_gp @ _rss_RTs
                    if _rs_complex:
                        _stress_HF_np = np.asarray(
                            _rs_obs_ctx.grad_e_L_eps_pbc_complex_batch(_rss_r_up, _rss_r_dn, _rs_ecp_w, _gp_all, _rss_geminv)
                        )
                    else:
                        _stress_HF_np = np.asarray(
                            _rs_obs_ctx.grad_e_L_eps_pbc_batch(_rss_r_up, _rss_r_dn, _rs_ecp_w, _gp_all, _rss_geminv)
                        )
                elif "stress_HF" in _vmc_fused_stash:
                    _stress_HF_np = np.asarray(_vmc_fused_stash.pop("stress_HF"))
                else:
                    _stress_HF_step = _grad_e_L_eps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        RTs,
                        geminal_inv,
                    )
                    _stress_HF_np = np.asarray(_stress_HF_step)

                if rust_active:
                    _rss_geminv_pp = np.asarray(geminal_inv, dtype=(np.complex128 if _rs_complex else dtype_np))
                    if _rs_complex:
                        _stress_PP_np = np.asarray(
                            _rs_obs_ctx.grad_ln_psi_eps_pbc_complex_batch(_rss_r_up, _rss_r_dn, _rss_geminv_pp)
                        )
                    else:
                        _stress_PP_np = np.asarray(_rs_obs_ctx.grad_ln_psi_eps_pbc_batch(_rss_r_up, _rss_r_dn, _rss_geminv_pp))
                elif "stress_PP" in _vmc_fused_stash:
                    _stress_PP_np = np.asarray(_vmc_fused_stash.pop("stress_PP"))
                else:
                    _stress_PP_step = _grad_ln_psi_eps(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        geminal_inv,
                    )
                    _stress_PP_np = np.asarray(_stress_PP_step)

                _e_L_np = np.array(e_L_step)
                self.__stored_stress_HF[self.__mcmc_counter + num_mcmc_done] = _stress_HF_np
                self.__stored_stress_PP[self.__mcmc_counter + num_mcmc_done] = _stress_PP_np
                self.__stored_E_L_stress_PP[self.__mcmc_counter + num_mcmc_done] = np.einsum(
                    "w,wij->wij", _e_L_np, _stress_PP_np
                )
                end = time.perf_counter()
                timer_stress += end - start

            if self.__comput_log_WF_param_deriv:
                start = time.perf_counter()
                if rust_active and _rs_okc_wanted:
                    derivative_sources = _rs_native_c_param_grads(
                        _rs_pd_ctx,
                        self.__hamiltonian_data,
                        np.asarray(self.__latest_r_up_carts, dtype=dtype_np),
                        np.asarray(self.__latest_r_dn_carts, dtype=dtype_np),
                        _c_mask_kwargs,
                        nw,
                        quantity="ln_Psi",
                        complex_twist=_rs_complex,
                        source_blocks=_source_blocks,
                    )
                else:
                    # The JAX analytical bridge publishes the same exact
                    # ambient-storage derivative components as the native path.
                    grad_ln_Psi_wf = _grad_ln_psi_c(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        geminal_inv,
                    )
                    derivative_sources = _collect_c_param_derivative_sources(
                        self.__hamiltonian_data.wavefunction_data,
                        grad_ln_Psi_wf,
                        _source_blocks,
                        self.__num_walkers,
                        _c_mask_kwargs,
                        device=self.__device_optimizer_enabled(),
                    )
                self.__store_derivative_sources(
                    self.__stored_log_WF_derivative_sources,
                    self.__log_WF_sources_in_param_grads,
                    self.__stored_log_WF_param_grads,
                    derivative_sources,
                    folded_stored=not self.__device_optimizer_enabled(),
                )
                if not self.__device_optimizer_enabled():
                    flat_param_grads = _diagnostic_storage_derivatives(derivative_sources)
                    for name, grad_val in flat_param_grads.items():
                        if not self.__param_grad_flags.get(name, True):
                            continue
                        if hasattr(grad_val, "block_until_ready"):
                            grad_val.block_until_ready()
                        self.__stored_log_WF_param_grads[name].append(np.array(grad_val))

                end = time.perf_counter()
                timer_dln_Psi_dc += end - start

            if self.__comput_e_L_param_deriv:
                start = time.perf_counter()
                # Gradient flows through full compute_local_energy including
                # ECP wf-ratio Psi(r')/Psi(r), so d(V_ECP_NL)/dc is included.
                # Routed through the derivatives interface.
                # (analytic=False); mask / stop_gradient are internal to it.

                # Diagnostics: check walker positions and RTs before grad call
                _r_up = np.asarray(self.__latest_r_up_carts)
                _r_dn = np.asarray(self.__latest_r_dn_carts)
                _RTs = np.asarray(RTs)
                _nan_r_up = int(np.sum(~np.isfinite(_r_up)))
                _nan_r_dn = int(np.sum(~np.isfinite(_r_dn)))
                _nan_RTs = int(np.sum(~np.isfinite(_RTs)))
                logger.devel(
                    f"    [de_L/dc] r_up_carts: shape={_r_up.shape} non-finite={_nan_r_up}/{_r_up.size} "
                    f"    min={np.nanmin(_r_up):.3e} max={np.nanmax(_r_up):.3e}"
                )
                if _r_dn.size > 0:
                    logger.devel(
                        f"    [de_L/dc] r_dn_carts: shape={_r_dn.shape} non-finite={_nan_r_dn}/{_r_dn.size} "
                        f"    min={np.nanmin(_r_dn):.3e} max={np.nanmax(_r_dn):.3e}"
                    )
                else:
                    logger.devel(f"    [de_L/dc] r_dn_carts: shape={_r_dn.shape} (empty -- no down-spin electrons)")
                logger.devel(f"    [de_L/dc] RTs: shape={_RTs.shape} non-finite={_nan_RTs}/{_RTs.size}")

                # A long Sherman--Morrison chain accumulates route-dependent
                # roundoff in the running inverse.  This is normally invisible,
                # but an ill-conditioned geminal can amplify a Rust/XLA state
                # difference into the c-axis derivative.  One Newton--Schulz
                # residual correction restores G^{-1} to first order without a
                # fresh SVD/LU (important under MPI), and is applied identically
                # before either derivative backend consumes the state.
                _geminal_inv_c = _refine_running_geminal_inv_batched(geminal, geminal_inv)

                if rust_active and _rs_elc_wanted:
                    derivative_sources = _rs_native_c_param_grads(
                        _rs_pd_ctx,
                        self.__hamiltonian_data,
                        np.asarray(self.__latest_r_up_carts, dtype=dtype_np),
                        np.asarray(self.__latest_r_dn_carts, dtype=dtype_np),
                        _c_mask_kwargs,
                        nw,
                        quantity="e_L",
                        ecp_flag=_rs_ecp_flag,
                        ecp_w=_rs_ecp_w,
                        ecp_gp=_rs_ecp_gp,
                        rts_b=np.asarray(RTs, dtype=dtype_np),
                        complex_twist=_rs_complex,
                        source_blocks=_source_blocks,
                        geminal_inv=np.asarray(
                            _geminal_inv_c,
                            dtype=(np.complex128 if _rs_complex else dtype_np),
                        ),
                    )
                else:
                    grad_e_L_wf = _grad_e_L_c(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts,
                        self.__latest_r_dn_carts,
                        RTs,
                        _geminal_inv_c,
                    )
                    derivative_sources = _collect_c_param_derivative_sources(
                        self.__hamiltonian_data.wavefunction_data,
                        grad_e_L_wf,
                        _source_blocks,
                        self.__num_walkers,
                        _c_mask_kwargs,
                        device=self.__device_optimizer_enabled(),
                    )
                self.__store_derivative_sources(
                    self.__stored_e_L_derivative_sources,
                    self.__e_L_sources_in_param_grads,
                    self.__stored_e_L_param_grads,
                    derivative_sources,
                    folded_stored=not self.__device_optimizer_enabled(),
                )
                if not self.__device_optimizer_enabled():
                    flat_param2_grads = _diagnostic_storage_derivatives(derivative_sources)
                    for _pname, _pval in flat_param2_grads.items():
                        _parr = np.asarray(_pval)
                        _pnan = int(np.sum(~np.isfinite(_parr)))
                        logger.devel(
                            f"    [de_L/dc] flat_param2_grads['{_pname}']: shape={_parr.shape} "
                            f"    non-finite={_pnan}/{_parr.size} ({_pnan / _parr.size:.1%})"
                        )
                    for name, grad_val in flat_param2_grads.items():
                        if not self.__param_grad_flags.get(name, True):
                            continue
                        if hasattr(grad_val, "block_until_ready"):
                            grad_val.block_until_ready()
                        self.__stored_e_L_param_grads[name].append(np.array(grad_val))

                end = time.perf_counter()
                timer_de_L_dc += end - start

            num_mcmc_done += 1

            # check max time
            # Use rank 0's decision and broadcast to all ranks to prevent
            # MPI deadlock caused by different ranks disagreeing on whether
            # to break (due to slight differences in mcmc_total_start).
            mcmc_current = time.perf_counter()
            should_stop = bool(max_time < mcmc_current - mcmc_total_start) if self._mpi_rank == 0 else False
            should_stop = self._mpi_comm.bcast(should_stop, root=0)
            if should_stop:
                logger.info(f"  Stopping... max_time = {max_time} sec. exceeds.")
                logger.info("  Break the mcmc loop.")
                break

            # check the control file (stop flag); shared across all ranks / twists
            stop_flag = _read_external_stop_flag(toml_filename, self._mpi_comm)
            if stop_flag:
                logger.info(f"  Stopping... stop_flag in {toml_filename} is true.")
                logger.info("  Break the mcmc loop.")
                break

        progress = (num_mcmc_done + self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        mcmc_total_current = time.perf_counter()
        logger.info(
            f"  Progress: MCMC step = {num_mcmc_done + self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %. Elapsed time = {(mcmc_total_current - mcmc_loop_start):.1f} sec."
        )

        if not _rs_energy_chunk and _finite_check_done < num_mcmc_done:
            _check_start = self.__mcmc_counter + _finite_check_done
            _check_stop = self.__mcmc_counter + num_mcmc_done
            _raise_collective_nonfinite_observables(
                self._mpi_comm,
                driver="MCMC",
                stage="measurement",
                first_step=_check_start,
                observables={
                    "e_L": self.__stored_e_L[_check_start:_check_stop],
                    "e_L2": self.__stored_e_L2[_check_start:_check_stop],
                    "w_L": self.__stored_w_L[_check_start:_check_stop],
                },
                r_up_carts=self.__latest_r_up_carts,
                r_dn_carts=self.__latest_r_dn_carts,
            )
            self.__restart_validation_pending = False

        # Barrier after MCMC operation
        start = time.perf_counter()
        self._mpi_comm.Barrier()
        end = time.perf_counter()
        timer_MPI_barrier += end - start

        logger.info("End MCMC")
        logger.info("")

        # count up the mcmc counter
        # count up mcmc_counter
        self.__mcmc_counter += num_mcmc_done

        # -- Truncate stored arrays to actual number of steps completed --
        self.__stored_e_L = self.__stored_e_L[: self.__mcmc_counter]
        self.__stored_e_L2 = self.__stored_e_L2[: self.__mcmc_counter]
        self.__stored_w_L = self.__stored_w_L[: self.__mcmc_counter]
        if self.__mpc:
            self.__stored_mpc_dV = self.__stored_mpc_dV[: self.__mcmc_counter]
        if self.__comput_position_deriv:
            self.__stored_force_HF = self.__stored_force_HF[: self.__mcmc_counter]
            self.__stored_force_PP = self.__stored_force_PP[: self.__mcmc_counter]
            self.__stored_E_L_force_PP = self.__stored_E_L_force_PP[: self.__mcmc_counter]
        if self.__comput_stress:
            self.__stored_stress_HF = self.__stored_stress_HF[: self.__mcmc_counter]
            self.__stored_stress_PP = self.__stored_stress_PP[: self.__mcmc_counter]
            self.__stored_E_L_stress_PP = self.__stored_E_L_stress_PP[: self.__mcmc_counter]

        # test the shapes of stored arrays are consistent with the number of MCMC steps done and the number of walkers
        self.__validate_stored_shapes()

        mcmc_total_end = time.perf_counter()
        timer_mcmc_total += mcmc_total_end - mcmc_total_start
        timer_misc = timer_mcmc_total - (
            timer_mcmc_update_init
            + timer_mcmc_update
            + timer_e_L
            + timer_de_L_dR_dr
            + timer_dln_Psi_dR_dr
            + timer_dln_Psi_dc
            + timer_de_L_dc
            + timer_stress
            + timer_MPI_barrier
        )

        # remove the control file (world root owns it; EAFP-guarded)
        self._mpi_comm.Barrier()
        _remove_external_control_file(toml_filename, self._world_comm)

        # net MCMC time
        timer_net_mcmc_total = timer_mcmc_total - timer_mcmc_update_init

        # average among MPI processes
        ave_timer_mcmc_total = self._mpi_comm.allreduce(timer_mcmc_total, op=MPI.SUM) / self._mpi_size
        ave_timer_mcmc_update_init = self._mpi_comm.allreduce(timer_mcmc_update_init, op=MPI.SUM) / self._mpi_size
        ave_timer_net_mcmc_total = self._mpi_comm.allreduce(timer_net_mcmc_total, op=MPI.SUM) / self._mpi_size
        if num_mcmc_done > 0:
            ave_timer_mcmc_update = self._mpi_comm.allreduce(timer_mcmc_update, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_e_L = self._mpi_comm.allreduce(timer_e_L, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_de_L_dR_dr = self._mpi_comm.allreduce(timer_de_L_dR_dr, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_dln_Psi_dR_dr = self._mpi_comm.allreduce(timer_dln_Psi_dR_dr, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_dln_Psi_dc = self._mpi_comm.allreduce(timer_dln_Psi_dc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_de_L_dc = self._mpi_comm.allreduce(timer_de_L_dc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_stress = self._mpi_comm.allreduce(timer_stress, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_MPI_barrier = self._mpi_comm.allreduce(timer_MPI_barrier, op=MPI.SUM) / self._mpi_size / num_mcmc_done
            ave_timer_misc = self._mpi_comm.allreduce(timer_misc, op=MPI.SUM) / self._mpi_size / num_mcmc_done
        else:
            ave_timer_mcmc_update = 0.0
            ave_timer_e_L = 0.0
            ave_timer_de_L_dR_dr = 0.0
            ave_timer_dln_Psi_dR_dr = 0.0
            ave_timer_dln_Psi_dc = 0.0
            ave_timer_de_L_dc = 0.0
            ave_timer_stress = 0.0
            ave_timer_MPI_barrier = 0.0
            ave_timer_misc = 0.0
        ave_stored_w_L = self._mpi_comm.allreduce(np.mean(self.__stored_w_L), op=MPI.SUM) / self._mpi_size
        sum_accepted_moves = self._mpi_comm.allreduce(int(self.__accepted_moves), op=MPI.SUM)
        sum_rejected_moves = self._mpi_comm.allreduce(int(self.__rejected_moves), op=MPI.SUM)

        logger.info(f"Total elapsed time for MCMC {num_mcmc_done} steps. = {ave_timer_mcmc_total:.2f} sec.")
        logger.info(f"Pre-compilation time for MCMC = {ave_timer_mcmc_update_init:.2f} sec.")
        logger.info(f"Net total time for MCMC = {ave_timer_net_mcmc_total:.2f} sec.")
        logger.info(f"Elapsed times per MCMC step, averaged over {num_mcmc_done} steps.")
        logger.info(f"  Time for MCMC update = {ave_timer_mcmc_update * 10**3:.2f} msec.")
        logger.info(f"  Time for computing e_L = {ave_timer_e_L * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dR and de_L/dr = {ave_timer_de_L_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dR and dln_Psi/dr = {ave_timer_dln_Psi_dR_dr * 10**3:.2f} msec.")
        logger.info(f"  Time for computing dln_Psi/dc = {ave_timer_dln_Psi_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/dc = {ave_timer_de_L_dc * 10**3:.2f} msec.")
        logger.info(f"  Time for computing de_L/deps and dln_Psi/deps = {ave_timer_stress * 10**3:.2f} msec.")
        logger.info(f"  Time for MPI barrier after MCMC update = {ave_timer_MPI_barrier * 10**3:.2f} msec.")
        logger.info(f"  Time for misc. (others) = {ave_timer_misc * 10**3:.2f} msec.")
        logger.info(f"Average of walker weights is {ave_stored_w_L:.3f}. Ideal is ~ 0.800. Adjust epsilon_AS.")
        if sum_accepted_moves + sum_rejected_moves > 0:
            logger.info(
                f"Acceptance ratio is {sum_accepted_moves / (sum_accepted_moves + sum_rejected_moves) * 100:.2f} %.  Ideal is ~ 50.00%. Adjust Dt."
            )
        else:
            logger.info("Acceptance ratio is N/A (no moves performed).")
        logger.info("")

        _log_memory_usage("mcmc_post_run", comm=self._mpi_comm)

    def get_E(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        _observable: npt.NDArray | None = None,
    ) -> tuple[float, float, float, float]:
        """Estimate total energy and variance with jackknife error bars.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.
            _observable (npt.NDArray | None, optional): Private; replaces the per-step local energy
                by this observable while keeping the same weights and binning. ``None`` uses ``e_L``.

        Returns:
            tuple[float, float, float, float]: ``(E_mean, E_std, Var_mean, Var_std)`` aggregated across MPI ranks.

        Raises:
            ValueError: If there are insufficient post-warmup samples to form the requested blocks.

        Notes:
            The warmup and block counts are the caller's choice: no minimum is enforced beyond the post-warmup
            samples needed to form the blocks. All reductions are MPI-aware.
        """
        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        # ``_observable`` (private) substitutes the per-step local quantity while
        # keeping the SAME weights and binning, so a derived ratio estimator such
        # as ``e_L + dV_MPC`` inherits this jackknife exactly.
        e_L = (self.e_L if _observable is None else _observable)[num_mcmc_warmup_steps:]
        e_L2 = self.e_L2[num_mcmc_warmup_steps:]
        w_L = self.w_L[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) from mcmc_counter and num_walkers
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
        _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
        _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")

        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
        w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
        w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_e_L2_binned_local = w_L_e_L2_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

        # Shape assertions: binned arrays must be 1D (num_mcmc_bin_blocks * nw,)
        M_binned = num_mcmc_bin_blocks * nw
        _require(w_L_binned_local.shape == (M_binned,), f"w_L_binned_local shape {w_L_binned_local.shape} != ({M_binned},)")
        _require(
            w_L_e_L_binned_local.shape == (M_binned,),
            f"w_L_e_L_binned_local shape {w_L_e_L_binned_local.shape} != ({M_binned},)",
        )
        _require(
            w_L_e_L2_binned_local.shape == (M_binned,),
            f"w_L_e_L2_binned_local shape {w_L_e_L2_binned_local.shape} != ({M_binned},)",
        )

        ## local sum
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_e_L2_binned_local_sum = np.sum(w_L_e_L2_binned_local)

        ## glolbal sum
        w_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_binned_local), op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L_binned_local), op=MPI.SUM)
        w_L_e_L2_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L2_binned_local), op=MPI.SUM)

        ## jackknie binned samples
        M_local = w_L_binned_local.size
        M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

        E_jackknife_binned_local = np.array(
            [
                (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]
        )

        E2_jackknife_binned_local = np.array(
            [
                (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
                for m in range(M_local)
            ]
        )

        # With a complex ``e_L``, the
        # per-bin jackknife E is complex (``e_L`` is complex per walker, only
        # its expectation is real). The physical observable is the real part;
        # the imaginary part is a fixed-phase contamination diagnostic. The
        # variance estimator must use the modulus square ``|<e_L>|**2`` --
        # the naive ``<e_L>**2`` on a complex bin mean adds the unphysical
        # ``-Im(<e>)**2`` contribution. Bit-identical on the real path
        # (``np.abs(x)**2 == x**2``, ``np.real(x) == x``, ``np.imag(x) == 0``).
        _E_is_complex = np.iscomplexobj(E_jackknife_binned_local)
        if _E_is_complex:
            abs_E_sq_local = np.abs(E_jackknife_binned_local) ** 2
            Im_E_local = np.imag(E_jackknife_binned_local)
            E_jackknife_binned_local = np.real(E_jackknife_binned_local)
        else:
            abs_E_sq_local = E_jackknife_binned_local**2

        Var_jackknife_binned_local = E2_jackknife_binned_local - abs_E_sq_local

        # Two-pass jackknife std (centered sum of squares) to avoid catastrophic
        # cancellation in <x^2> - <x>^2 when M_total grows large (many walkers).

        # First pass: global mean after the physical-real projection above.
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
        sumsq_centered_Var_global = self._mpi_comm.allreduce(np.sum((Var_jackknife_binned_local - Var_mean) ** 2), op=MPI.SUM)
        Var_var = sumsq_centered_Var_global / M_total
        Var_std = np.sqrt((M_total - 1) * Var_var)

        # Fixed-phase Im<E> diagnostic.
        # ``Im<E>`` should be zero in expectation for a Hermitian H when the
        # down-spin leaf uses -k and conjugated up-spin coefficients; a large
        # value flags a broken complex-grad
        # path (e.g. an unconjugated store, a real-cast in determinant.py)
        # or a non-Hermitian lambda. Downgraded to ``warning`` (not
        # ``assert``) because a noisy short run can produce
        # large transient ``Im<E>`` without the chain being broken.
        if _E_is_complex:
            sum_Im_E_global = self._mpi_comm.allreduce(np.sum(Im_E_local), op=MPI.SUM)
            Im_E_mean = float(sum_Im_E_global / M_total)
            if E_std > 0 and abs(Im_E_mean) > 2.0 * E_std:
                logger.warning(
                    "Complex-energy guard: |Im<E>| = %.3e exceeds 2 * E_std = %.3e",
                    Im_E_mean,
                    E_std,
                )
            else:
                logger.devel("Im<E> = %.3e (E_std = %.3e, OK)", Im_E_mean, E_std)

        logger.devel(f"E = {E_mean} +- {E_std} Ha.")
        logger.devel(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        return (E_mean, E_std, Var_mean, Var_std)

    def get_dE_mpc(
        self,
        num_mcmc_warmup_steps: int = 0,
        num_mcmc_bin_blocks: int = 5,
    ) -> tuple[float, float]:
        """Estimate the MPC finite-size correction ``dE_mpc`` with jackknife error bars.

        The corrected total energy is ``E_MPC = E_Ewald + dE_mpc``, where ``E_Ewald`` is the
        value returned by :meth:`get_E` and ``dE_mpc = <V_ee^MPC - V_ee^Ewald>`` is the average
        of the per-step, per-walker difference stored by :meth:`run`. Both terms come out of ONE
        run evaluated on identical walker configurations, so they are strongly correlated and the
        error bar on the difference is much smaller than the statistical error of either Coulomb
        channel taken alone.

        This is an energy-only correction (``physical = no``): MPC never enters the sampled
        Hamiltonian, so forces (:meth:`get_aF`), stresses and all variational parameter gradients
        remain purely Ewald.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 0.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 5.

        Returns:
            tuple[float, float]: ``(dE_mpc_mean, dE_mpc_std)`` in Ha, aggregated across MPI ranks.

        Raises:
            ValueError: If ``mpc`` was not enabled at construction, or if there are insufficient
                post-warmup samples to form the requested blocks.
        """
        if not self.__mpc:
            logger.error("get_dE_mpc requires mpc=True at MCMC construction.")
            raise ValueError("get_dE_mpc requires the MPC observable to be enabled: construct the MCMC with mpc=True.")

        if self.mcmc_counter < num_mcmc_warmup_steps:
            logger.error("mcmc_counter should be larger than num_mcmc_warmup_steps")
            raise ValueError
        if self.mcmc_counter - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            logger.error("(mcmc_counter - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")
            raise ValueError

        mpc_dV = self.mpc_dV[num_mcmc_warmup_steps:]
        w_L = self.w_L[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) from mcmc_counter and num_walkers
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
        _require(mpc_dV.shape == (M, nw), f"mpc_dV shape {mpc_dV.shape} != ({M}, {nw})")

        # Identical weighted binning to ``get_E``: sum ``w`` and ``w * O`` within
        # each block, ravel over walkers, then jackknife the ratio of global sums.
        # ``mpc_dV`` is real at any twist (difference of two real e-e potentials),
        # so none of the complex-e_L projections of ``get_E`` are needed here.
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_mpc_dV_split = np.array_split(w_L * mpc_dV, num_mcmc_bin_blocks, axis=0)
        w_L_mpc_dV_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_mpc_dV_split]))

        # MCMC case
        w_L_binned_local = np.array(w_L_binned)
        w_L_mpc_dV_binned_local = np.array(w_L_mpc_dV_binned)

        # Shape assertions: binned arrays must be 1D (num_mcmc_bin_blocks * nw,)
        M_binned = num_mcmc_bin_blocks * nw
        _require(w_L_binned_local.shape == (M_binned,), f"w_L_binned_local shape {w_L_binned_local.shape} != ({M_binned},)")
        _require(
            w_L_mpc_dV_binned_local.shape == (M_binned,),
            f"w_L_mpc_dV_binned_local shape {w_L_mpc_dV_binned_local.shape} != ({M_binned},)",
        )

        ## glolbal sum
        w_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_binned_local), op=MPI.SUM)
        w_L_mpc_dV_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_mpc_dV_binned_local), op=MPI.SUM)

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

        return (dE_mean, dE_std)

    def get_E_mpc(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """MPC-corrected total energy with the JOINT (covariance-aware) error bar.

        ``E_MPC = <e_L> + <dV_MPC>`` where both averages are weighted ratios over
        the SAME walkers and weights, so they are correlated and combining their
        separate error bars in quadrature is wrong (measured 0.0422 vs the
        correct 0.0190 on a regression fixture). Because the two estimators share
        the denominator, their joint jackknife is exactly the jackknife of the
        summed numerator, so this delegates to the standard estimator with the
        substituted observable ``e_L + dV_MPC``.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks.

        Returns:
            tuple[float, float]: ``(E_mpc_mean, E_mpc_std)`` in Ha.

        Raises:
            ValueError: If the MPC observable was not enabled at construction.
        """
        if not self.__mpc:
            logger.error("get_E_mpc requires mpc=True at construction.")
            raise ValueError("get_E_mpc requires the MPC observable to be enabled: construct the driver with mpc=True.")
        mean, std, _var_mean, _var_std = self.get_E(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            _observable=self.e_L + self.mpc_dV,
        )
        return (float(mean), float(std))

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> ArrayEstimate:
        """Compute Hellmann-Feynman + Pulay forces with jackknife statistics.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to drop for warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: ``(force_mean, force_std)`` shaped ``(num_atoms, 3)`` in Hartree/bohr.

        Notes:
            Uses stored per-walker weights, energies, and pre-computed force products accumulated during :meth:`run`; reductions are MPI-aware.
        """
        w_L = self.w_L[num_mcmc_warmup_steps:]
        e_L = self.e_L[num_mcmc_warmup_steps:]
        force_HF = self.__stored_force_HF[num_mcmc_warmup_steps:]
        force_PP = self.__stored_force_PP[num_mcmc_warmup_steps:]
        E_L_force_PP = self.__stored_E_L_force_PP[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) for scalars; (M, nw, n_atoms, 3) for forces
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        n_atoms = self.__hamiltonian_data.structure_data.natom
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

        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_force_HF_binned_local = w_L_force_HF_binned
        w_L_force_PP_binned_local = w_L_force_PP_binned
        w_L_E_L_force_PP_binned_local = w_L_E_L_force_PP_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        # Shape assertions: scalar bins (num_bins * nw,); force bins (num_bins * nw, n_atoms, 3)
        M_binned = num_mcmc_bin_blocks * nw
        _require(w_L_binned_local.shape == (M_binned,), f"w_L_binned_local shape {w_L_binned_local.shape} != ({M_binned},)")
        _require(
            w_L_e_L_binned_local.shape == (M_binned,),
            f"w_L_e_L_binned_local shape {w_L_e_L_binned_local.shape} != ({M_binned},)",
        )
        _require(
            w_L_force_HF_binned_local.shape == (M_binned, n_atoms, 3),
            f"w_L_force_HF_binned_local shape {w_L_force_HF_binned_local.shape} != ({M_binned}, {n_atoms}, 3)",
        )
        _require(
            w_L_force_PP_binned_local.shape == (M_binned, n_atoms, 3),
            f"w_L_force_PP_binned_local shape {w_L_force_PP_binned_local.shape} != ({M_binned}, {n_atoms}, 3)",
        )
        _require(
            w_L_E_L_force_PP_binned_local.shape == (M_binned, n_atoms, 3),
            f"w_L_E_L_force_PP_binned_local shape {w_L_E_L_force_PP_binned_local.shape} != ({M_binned}, {n_atoms}, 3)",
        )

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

        # MPI Allreduce for the force array, routed through
        ### the complex-aware helper so ``w * E_L * F_PP`` (E_L is complex at
        ### non-Gamma) reduces with ``Im`` intact. Real path bit-identical.
        w_L_force_HF_binned_global_sum = allreduce_sum(self._mpi_comm, w_L_force_HF_binned_local_sum)
        w_L_force_PP_binned_global_sum = allreduce_sum(self._mpi_comm, w_L_force_PP_binned_local_sum)
        w_L_E_L_force_PP_binned_global_sum = allreduce_sum(self._mpi_comm, w_L_E_L_force_PP_binned_local_sum)

        ## jackknie binned samples
        M_local = w_L_binned_local.size
        M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

        force_HF_jn_local = -1.0 * np.array(
            [
                (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
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

        _require(
            force_HF_jn_local.shape == force_Pulay_jn_local.shape,
            f"force_HF_jn shape {force_HF_jn_local.shape} != force_Pulay_jn shape {force_Pulay_jn_local.shape}",
        )
        # Atomic force is a physical real observable;
        # project to real BEFORE the variance pass so std is variance of the
        # physical force (not of the complex sample, whose Im part vanishes
        # only in expectation). Identity on the Gamma real path.
        force_jn_local = (force_HF_jn_local + force_Pulay_jn_local).real

        # Two-pass jackknife std (centered sum of squares) to avoid catastrophic
        # cancellation in <x^2> - <x>^2 when M_total grows large.

        # 1st pass -- global mean
        sum_force_local = np.sum(force_jn_local, axis=0)
        sum_force_global = allreduce_sum(self._mpi_comm, sum_force_local)
        mean_force_global = sum_force_global / M_total

        # 2nd pass -- centered sum of squares (numerically stable)
        sumsq_centered_force_local = np.sum((force_jn_local - mean_force_global) ** 2, axis=0)
        sumsq_centered_force_global = allreduce_sum(self._mpi_comm, sumsq_centered_force_local)
        var_force_global = sumsq_centered_force_global / M_total

        ## mean and std
        force_mean = mean_force_global
        force_std = np.sqrt((M_total - 1) * var_force_global)

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.devel(f"force = {force_mean} +- {force_std} Ha.")

        return (force_mean, force_std)

    def get_aT(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        _return_hydrostatic: bool = False,
    ) -> StressEstimate:
        """Compute the symmetric stress tensor with jackknife statistics.

        Mirrors :py:meth:`get_aF` but the leaf shape is ``(3, 3)`` instead
        of ``(N_atom, 3)``. The estimator is the covariance-corrected
        Hellmann-Feynman + Pulay form

        .. math::
            \\sigma_{\\alpha\\beta} = \\langle \\sigma^{\\rm HF}_{\\alpha\\beta} \\rangle
                + 2 \\left( \\langle E_L\\,\\sigma^{\\rm PP}_{\\alpha\\beta} \\rangle
                  - \\langle E_L \\rangle \\langle \\sigma^{\\rm PP}_{\\alpha\\beta} \\rangle
                \\right)

        symmetrised (``\\sigma + \\sigma.T) / 2``) and divided by the cell
        volume ``\\Omega = |\\det(\\text{cell})|`` (Hartree/Bohr^3).

        Contract (fixed static ansatz): ``sigma^HF = d e_L / d eps`` and
        ``sigma^PP = d ln|Psi| / d eps`` are strain derivatives at a FIXED
        ansatz. The strain ``F = I + eps`` maps the cell, every nucleus and every
        walker electron as ``x -> F x``; the non-local ECP quadrature directions
        stay fixed, so each strained quadrature point lies on the sphere of the
        strained electron-nucleus distance and the rotation-averaged estimator is
        the exact strain derivative of the angular integral; basis
        exponents and contraction coefficients, MO / geminal coefficients and
        Jastrow parameters are held fixed (comoving data such as plane-wave
        reciprocal indices and lattice-image index sets move with the cell).
        This tensor is therefore NOT the equation-of-state derivative of a
        re-optimized ground-state energy; the two differ by the response of
        the frozen parameters to the strain.

        Args:
            num_mcmc_warmup_steps: Samples to drop for warmup.
            num_mcmc_bin_blocks: Number of jackknife blocks.
            _return_hydrostatic: Private; if True, also return the hydrostatic
                pressure ``P = -tr(sigma)/3`` and its jackknife error, taken over the
                same replicates, as two extra trailing elements.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: ``(sigma_mean, sigma_std)`` each
            shape ``(3, 3)`` in Hartree/Bohr^3.

        Notes:
            Molecular structures (``pbc_flag=False``) return
            ``(zeros((3, 3)), zeros((3, 3)))`` immediately --- the sigma buffers
            carry no contribution on the molecular code path so no
            run-time check inside the kernel is needed.
        """
        if not self.__comput_stress:
            raise ValueError("Stress data were not retained; construct MCMC with comput_stress=True.")

        # The early returns must honor _return_hydrostatic: get_P_hydrostatic
        # unpacks four values on every rank (rank != 0 of an MPI run included)
        # and also for a molecular structure. The zeros mirror the tensor convention:
        # the root rank owns the reduced observable.
        if not self.__hamiltonian_data.structure_data.pbc_flag:
            zeros = (np.zeros((3, 3)), np.zeros((3, 3)))
            return (*zeros, 0.0, 0.0) if _return_hydrostatic else zeros

        w_L = self.w_L[num_mcmc_warmup_steps:]
        e_L = self.e_L[num_mcmc_warmup_steps:]
        stress_HF = self.__stored_stress_HF[num_mcmc_warmup_steps:]
        stress_PP = self.__stored_stress_PP[num_mcmc_warmup_steps:]
        E_L_stress_PP = self.__stored_E_L_stress_PP[num_mcmc_warmup_steps:]

        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), "")
        _require(e_L.shape == (M, nw), "")
        _require(stress_HF.shape == (M, nw, 3, 3), "")
        _require(stress_PP.shape == (M, nw, 3, 3), "")
        _require(E_L_stress_PP.shape == (M, nw, 3, 3), "")

        # Weighted bin split (mirror ``get_aF`` structure with (3, 3) leaf).
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_stress_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_HF), num_mcmc_bin_blocks, axis=0)
        w_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_PP), num_mcmc_bin_blocks, axis=0)
        w_L_E_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_stress_PP), num_mcmc_bin_blocks, axis=0)

        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

        def _bin_tensor(splits: Sequence[npt.NDArray]) -> list[np.ndarray]:
            tensor_sum = np.array([np.sum(arr, axis=0) for arr in splits])
            return list(tensor_sum.reshape(tensor_sum.shape[0] * tensor_sum.shape[1], 3, 3))

        w_L_stress_HF_binned = _bin_tensor(w_L_stress_HF_split)
        w_L_stress_PP_binned = _bin_tensor(w_L_stress_PP_split)
        w_L_E_L_stress_PP_binned = _bin_tensor(w_L_E_L_stress_PP_split)

        w_L_binned_local = np.array(w_L_binned)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned)
        w_L_stress_HF_binned_local = np.array(w_L_stress_HF_binned)
        w_L_stress_PP_binned_local = np.array(w_L_stress_PP_binned)
        w_L_E_L_stress_PP_binned_local = np.array(w_L_E_L_stress_PP_binned)

        ## Local + global sums
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_stress_HF_binned_local_sum = np.sum(w_L_stress_HF_binned_local, axis=0)
        w_L_stress_PP_binned_local_sum = np.sum(w_L_stress_PP_binned_local, axis=0)
        w_L_E_L_stress_PP_binned_local_sum = np.sum(w_L_E_L_stress_PP_binned_local, axis=0)

        w_L_binned_global_sum = self._mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)
        w_L_stress_HF_binned_global_sum = np.empty_like(w_L_stress_HF_binned_local_sum)
        w_L_stress_PP_binned_global_sum = np.empty_like(w_L_stress_PP_binned_local_sum)
        w_L_E_L_stress_PP_binned_global_sum = np.empty_like(w_L_E_L_stress_PP_binned_local_sum)
        self._mpi_comm.Allreduce(
            [w_L_stress_HF_binned_local_sum, MPI.DOUBLE],
            [w_L_stress_HF_binned_global_sum, MPI.DOUBLE],
            op=MPI.SUM,
        )
        self._mpi_comm.Allreduce(
            [w_L_stress_PP_binned_local_sum, MPI.DOUBLE],
            [w_L_stress_PP_binned_global_sum, MPI.DOUBLE],
            op=MPI.SUM,
        )
        self._mpi_comm.Allreduce(
            [w_L_E_L_stress_PP_binned_local_sum, MPI.DOUBLE],
            [w_L_E_L_stress_PP_binned_global_sum, MPI.DOUBLE],
            op=MPI.SUM,
        )

        ## Jackknife per block (delete-one).
        M_local = w_L_binned_local.size
        M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

        sigma_HF_jn_local = np.array(
            [
                (w_L_stress_HF_binned_global_sum - w_L_stress_HF_binned_local[j])
                / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )
        sigma_Pulay_jn_local = 2.0 * np.array(
            [
                (
                    (w_L_E_L_stress_PP_binned_global_sum - w_L_E_L_stress_PP_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    - (
                        (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        * (w_L_stress_PP_binned_global_sum - w_L_stress_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                    )
                )
                for j in range(M_local)
            ]
        )

        # Real projection of the physical observable mirrors the
        # _MCMC_debug twin. No-op when the stored stress accumulators are
        # already real.
        sigma_jn_local = np.real(sigma_HF_jn_local + sigma_Pulay_jn_local)
        # Symmetrise per block (the asymmetric piece is rotation-noise that
        # averages to zero only in the limit; symmetrising per block tracks
        # the physical stress without polluting the jackknife stderr with
        # walker-level anti-symmetric fluctuations).
        sigma_jn_local = 0.5 * (sigma_jn_local + sigma_jn_local.transpose(0, 2, 1))
        # Volume normalisation: cell volume Omega = |det(cell)| from the
        # construction-time structure (the AD path doesn't change Omega, it
        # only differentiates through it -- so the volume is just a
        # scalar normalisation that turns sigma_tilde (Hartree) into sigma (Hartree/Bohr^3)).
        omega = float(np.abs(np.linalg.det(np.asarray(self.__hamiltonian_data.structure_data._cell_jnp))))
        sigma_jn_local = sigma_jn_local / omega

        # Two-pass jackknife mean + std (same numerically stable pattern
        # as ``get_aF``).
        sum_sigma_local = np.sum(sigma_jn_local, axis=0)
        sum_sigma_global = np.empty_like(sum_sigma_local)
        self._mpi_comm.Allreduce([sum_sigma_local, MPI.DOUBLE], [sum_sigma_global, MPI.DOUBLE], op=MPI.SUM)
        sigma_mean = sum_sigma_global / M_total

        sumsq_centered_local = np.sum((sigma_jn_local - sigma_mean) ** 2, axis=0)
        sumsq_centered_global = np.empty_like(sumsq_centered_local)
        self._mpi_comm.Allreduce([sumsq_centered_local, MPI.DOUBLE], [sumsq_centered_global, MPI.DOUBLE], op=MPI.SUM)
        var_sigma = sumsq_centered_global / M_total
        sigma_std = np.sqrt((M_total - 1) * var_sigma)

        if _return_hydrostatic:
            # The hydrostatic pressure is a
            # LINEAR functional of the same replicates, so its error must be
            # taken over P_jn = -tr(sigma_jn)/3 directly. Combining the three
            # diagonal error bars in quadrature discards their covariance and
            # is wrong in both directions (measured 3.76 vs 6.40 on a regression
            # probe). The replicates are available only here, which is why the
            # correct value cannot be reconstructed by the caller.
            p_hydro_jn_local = -np.trace(sigma_jn_local, axis1=1, axis2=2) / 3.0
            sum_p_global = self._mpi_comm.allreduce(float(np.sum(p_hydro_jn_local)), op=MPI.SUM)
            p_hydro_mean = sum_p_global / M_total
            sumsq_p_global = self._mpi_comm.allreduce(float(np.sum((p_hydro_jn_local - p_hydro_mean) ** 2)), op=MPI.SUM)
            p_hydro_std = float(np.sqrt((M_total - 1) * sumsq_p_global / M_total))
            return sigma_mean, sigma_std, float(p_hydro_mean), p_hydro_std

        return sigma_mean, sigma_std

    def get_P_hydrostatic(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> tuple[float, float]:
        """Hydrostatic pressure with the covariance-aware (trace-first) error bar.

        ``P_hydro = -tr(sigma)/3``. The three normal stresses are strongly
        correlated, so propagating their separate error bars in quadrature is
        wrong. This takes the jackknife over the trace of the same
        replicates, which includes every diagonal covariance by construction.

        Args:
            num_mcmc_warmup_steps: Samples to drop for warmup.
            num_mcmc_bin_blocks: Number of jackknife blocks.

        Returns:
            tuple[float, float]: ``(P_hydro_mean, P_hydro_std)`` in Ha/bohr^3;
            ``(0.0, 0.0)`` for molecular structures.
        """
        if not self.__hamiltonian_data.structure_data.pbc_flag:
            return 0.0, 0.0
        _sigma_mean, _sigma_std, p_mean, p_std = self.get_aT(
            num_mcmc_warmup_steps, num_mcmc_bin_blocks, _return_hydrostatic=True
        )
        return p_mean, p_std

    def get_aP(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> ArrayEstimate:
        """Normal-pressure components ``(P_x, P_y, P_z) = -(sigma_xx, sigma_yy, sigma_zz)``.

        The hydrostatic pressure is the arithmetic mean of the three. The
        diagonal triple is returned (instead of the scalar mean) so users
        can detect anisotropy under uniaxial strain or directional
        confinement without recomputing.

        Args:
            num_mcmc_warmup_steps: Samples to drop for warmup.
            num_mcmc_bin_blocks: Number of jackknife blocks.

        Returns:
            tuple[npt.NDArray, npt.NDArray]: ``(P_mean, P_std)`` each shape
            ``(3,)`` in Hartree/Bohr^3. Molecular structures return
            ``(zeros(3), zeros(3))`` immediately.
        """
        if not self.__hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3,)), np.zeros((3,))
        sigma_mean, sigma_std = self.get_aT(num_mcmc_warmup_steps, num_mcmc_bin_blocks)
        return -np.diag(sigma_mean), np.diag(sigma_std)

    def get_dln_WF(
        self,
        blocks: list[VariationalParameterBlock],
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: Sequence[int] | None = None,
    ) -> npt.NDArray:
        """Assemble per-sample derivatives of ln Psi w.r.t. variational parameters.

        Args:
            blocks (list[VariationalParameterBlock]): Ordered variational blocks used for concatenation.
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            chosen_param_index (list | None, optional): Optional subset of flattened indices; ``None`` keeps all. Defaults to None.

        Returns:
            npt.NDArray: ``O_matrix`` with shape ``(M, num_walkers, K)`` after warmup, where ``K`` follows the provided blocks (or subset).

        Notes:
            Validates the concatenated gradient size against block metadata and uses gradients stored during :meth:`run`.
        """
        grad_map = self.dln_Psi_dc

        # Collect gradients in the order of the provided variational blocks.
        dln_Psi_dc_list = []
        matched_blocks = []
        for block in blocks:
            if block.name in grad_map:
                dln_Psi_dc_list.append(grad_map[block.name])
                matched_blocks.append(block)

        # ``O_matrix`` dtype follows the widest block's
        # stored gradient (``complex128`` once any param block carries a
        # complex gradient and is ``float64`` otherwise.
        # A float64 seed buffer would silently down-cast the first complex
        # concatenate.
        if dln_Psi_dc_list:
            _O_dtype = np.result_type(*[a.dtype for a in dln_Psi_dc_list])
        else:
            _O_dtype = np.float64
        O_matrix = np.empty((self.mcmc_counter, self.num_walkers, 0), dtype=_O_dtype)

        for dln_Psi_dc, _ in zip(dln_Psi_dc_list, matched_blocks):
            logger.devel(f"dln_Psi_dc.shape={dln_Psi_dc.shape}.")
            if dln_Psi_dc.ndim == 2:  # scalar variational param.
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], 1)
            else:
                dln_Psi_dc_reshaped = dln_Psi_dc.reshape(
                    dln_Psi_dc.shape[0], dln_Psi_dc.shape[1], int(np.prod(dln_Psi_dc.shape[2:]))
                )
            O_matrix = np.concatenate((O_matrix, dln_Psi_dc_reshaped), axis=2)

        # basic sanity check for consistency with blocks
        total_size_from_blocks = sum(block.size for block in blocks)
        _require(
            total_size_from_blocks == O_matrix.shape[2],
            "Mismatch between total block size and O_matrix parameter dimension. "
            f"total_block_size={total_size_from_blocks} O_matrix_K={O_matrix.shape[2]}; "
            f"block_names={[(b.name, int(b.size)) for b in blocks]}; "
            f"grad_map_keys={sorted(grad_map.keys())}; "
            f"blocks_without_grad={[(b.name, int(b.size)) for b in blocks if b.name not in grad_map]}",
        )

        # ------------------------------------------------------------------
        # Project O_matrix for blocks with declared real-linear ties.
        # This ensures that f, S, theta all respect the symmetry automatically,
        # removing the need for downstream symmetrization of S/N ratios,
        # parameter updates, etc.
        # Derive this temporary original-storage projection solely from the
        # immutable COO tie. The same data drives exact real-coordinate
        # forward and adjoint actions.
        # ------------------------------------------------------------------
        _sym_start = 0
        for block in blocks:
            _sym_end = _sym_start + block.size
            if block.tangent_spec.kind == "linear_tie":
                _require(
                    block.tangent_spec.linear_tie is not None,
                    f"linear_tie block {block.name!r} has no linear-tie specification.",
                )
                _blk_slice = O_matrix[:, :, _sym_start:_sym_end]
                _orig_shape = _blk_slice.shape  # (M, nw, block_size)
                _flat_2d = _blk_slice.reshape(-1, block.size)  # (M*nw, block_size)
                _flat_2d[:] = _project_linear_tie_storage(
                    _flat_2d,
                    block.tangent_spec.linear_tie,
                    field=block.field,
                )
                O_matrix[:, :, _sym_start:_sym_end] = _flat_2d.reshape(_orig_shape)
            _sym_start = _sym_end

        logger.devel(f"O_matrix.shape = {O_matrix.shape}")
        if chosen_param_index is None:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:]
        else:
            O_matrix_chosen = O_matrix[num_mcmc_warmup_steps:, :, chosen_param_index]
        logger.devel(f"O_matrix_chosen.shape = {O_matrix_chosen.shape}")

        # ------------------------------------------------------------------
        # DEBUG: per-sample derivative statistics
        # Useful for diagnosing NaN propagation when epsilon_AS > 0.
        # ------------------------------------------------------------------
        _om = np.asarray(O_matrix_chosen)
        _om_nan = int(np.sum(~np.isfinite(_om)))
        _om_fin = _om[np.isfinite(_om)]
        # ``O_matrix`` may be complex at a
        # non-Gamma twist and follows the widest active parameter block. A
        # real-only active set legitimately remains real. When complex,
        # ``float(np.min(complex))`` raises ComplexWarning, so diagnose the
        # magnitude instead -- preserves the "is anything blowing up" signal
        # without truncating Im. Bit-identical at Gamma.
        if np.iscomplexobj(_om_fin):
            _om_fin_diag = np.abs(_om_fin)
        else:
            _om_fin_diag = _om_fin
        logger.devel(
            "[dln_Psi/dc] shape=%s  NaN or Inf=%d/%d  min=%.3e  max=%.3e  mean=%.3e  std=%.3e",
            _om.shape,
            _om_nan,
            _om.size,
            float(np.min(_om_fin_diag)) if _om_fin_diag.size else float("nan"),
            float(np.max(_om_fin_diag)) if _om_fin_diag.size else float("nan"),
            float(np.mean(_om_fin_diag)) if _om_fin_diag.size else float("nan"),
            float(np.std(_om_fin_diag)) if _om_fin_diag.size else float("nan"),
        )
        if _om_nan > 0:
            nan_param_cols = np.where(np.any(~np.isfinite(_om.reshape(-1, _om.shape[-1])), axis=0))[0]
            logger.devel(
                "[dln_Psi/dc] param indices with NaN or Inf (first 30): %s",
                nan_param_cols[:30].tolist(),
            )

        return O_matrix_chosen

    def get_gF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
        chosen_param_index: list | None = None,
        *,
        blocks: list[VariationalParameterBlock],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Evaluate generalized forces (dE/dc_k) with jackknife error bars.

        Args:
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            num_mcmc_bin_blocks (int, optional): Number of jackknife blocks. Defaults to 10.
            chosen_param_index (list | None, optional): Optional subset of flattened indices. Defaults to None.
            blocks (list[VariationalParameterBlock]): Required, keyword-only. Ordered variational
                blocks that select and order the parameters, as passed to :meth:`get_dln_WF`
                (for example ``wavefunction_data.get_variational_blocks(...)`` with the
                optimization flags of the run). There is no default block list.

        Returns:
            tuple[npt.NDArray, npt.NDArray]:
                ``(generalized_force_mean, generalized_force_std)`` as arrays over
                the filtered parameter space.

        Notes:
            Reuses :meth:`get_dln_WF` after warmup and applies jackknife statistics across MPI ranks.
        """
        w_L = self.w_L[num_mcmc_warmup_steps:]
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))

        e_L = self.e_L[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) from mcmc_counter and num_walkers
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
        _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")

        w_L_e_L_split = np.array_split(np.einsum("iw,iw->iw", w_L, e_L), num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

        O_matrix = self.get_dln_WF(
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            chosen_param_index=chosen_param_index,
            blocks=blocks,
        )

        # Shape assertions: O_matrix is (M, nw, K)
        K_gF = O_matrix.shape[2] if O_matrix.ndim == 3 else 0
        _require(O_matrix.shape == (M, nw, K_gF), f"O_matrix shape {O_matrix.shape} != ({M}, {nw}, {K_gF})")

        w_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, O_matrix), num_mcmc_bin_blocks, axis=0)
        w_L_O_matrix_sum = np.array([np.sum(arr, axis=0) for arr in w_L_O_matrix_split])
        w_L_O_matrix_binned_shape = (
            w_L_O_matrix_sum.shape[0] * w_L_O_matrix_sum.shape[1],
            w_L_O_matrix_sum.shape[2],
        )
        w_L_O_matrix_binned = list(w_L_O_matrix_sum.reshape(w_L_O_matrix_binned_shape))

        # With stored ``O_matrix = O_k`` (un-conjugated), the standard generalised-force
        # formula ``f_k = -2 Re<conj(O_k) (e_L - <e_L>)>`` requires ``conj`` on
        # the ``O`` factor inside the ``<e_L * O>`` accumulator. Bit-identical
        # on the real path (``conj`` of float64 is the identity).
        e_L_O_matrix = np.einsum("iw,iwj->iwj", e_L, np.conj(O_matrix))
        w_L_e_L_O_matrix_split = np.array_split(np.einsum("iw,iwj->iwj", w_L, e_L_O_matrix), num_mcmc_bin_blocks, axis=0)
        w_L_e_L_O_matrix_sum = np.array([np.sum(arr, axis=0) for arr in w_L_e_L_O_matrix_split])
        w_L_e_L_O_matrix_binned_shape = (
            w_L_e_L_O_matrix_sum.shape[0] * w_L_e_L_O_matrix_sum.shape[1],
            w_L_e_L_O_matrix_sum.shape[2],
        )
        w_L_e_L_O_matrix_binned = list(w_L_e_L_O_matrix_sum.reshape(w_L_e_L_O_matrix_binned_shape))

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_O_matrix_binned_local = w_L_O_matrix_binned
        w_L_e_L_O_matrix_binned_local = w_L_e_L_O_matrix_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_O_matrix_binned_local = np.array(w_L_O_matrix_binned_local)
        w_L_e_L_O_matrix_binned_local = np.array(w_L_e_L_O_matrix_binned_local)

        # Shape assertions: scalar bins (num_bins * nw,); O bins (num_bins * nw, K)
        M_binned = num_mcmc_bin_blocks * nw
        _require(w_L_binned_local.shape == (M_binned,), f"w_L_binned_local shape {w_L_binned_local.shape} != ({M_binned},)")
        _require(
            w_L_e_L_binned_local.shape == (M_binned,),
            f"w_L_e_L_binned_local shape {w_L_e_L_binned_local.shape} != ({M_binned},)",
        )
        _require(
            w_L_O_matrix_binned_local.shape == (M_binned, K_gF),
            f"w_L_O_matrix_binned_local shape {w_L_O_matrix_binned_local.shape} != ({M_binned}, {K_gF})",
        )
        _require(
            w_L_e_L_O_matrix_binned_local.shape == (M_binned, K_gF),
            f"w_L_e_L_O_matrix_binned_local shape {w_L_e_L_O_matrix_binned_local.shape} != ({M_binned}, {K_gF})",
        )

        ## local sum
        w_L_binned_local_sum = np.sum(w_L_binned_local)
        w_L_e_L_binned_local_sum = np.sum(w_L_e_L_binned_local)
        w_L_O_matrix_binned_local_sum = np.sum(w_L_O_matrix_binned_local, axis=0)
        w_L_e_L_O_matrix_binned_local_sum = np.sum(w_L_e_L_O_matrix_binned_local, axis=0)

        ## glolbal sum
        ### allreduce (scalar)
        w_L_binned_global_sum = self._mpi_comm.allreduce(w_L_binned_local_sum, op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(w_L_e_L_binned_local_sum, op=MPI.SUM)

        # Allreduce for the derivative array, routed through the
        ### complex-aware helper -- bit-identical for real buffers
        ### (``allreduce_sum`` dispatches to the same ``MPI.DOUBLE`` path),
        ### and preserves ``Im`` via the ``float64`` byte-view for complex.
        w_L_O_matrix_binned_global_sum = allreduce_sum(self._mpi_comm, w_L_O_matrix_binned_local_sum)
        w_L_e_L_O_matrix_binned_global_sum = allreduce_sum(self._mpi_comm, w_L_e_L_O_matrix_binned_local_sum)

        ## jackknie binned samples
        M_local = w_L_binned_local.size
        M_total = self._mpi_comm.allreduce(M_local, op=MPI.SUM)

        eL_O_jn_local = np.array(
            [
                (w_L_e_L_O_matrix_binned_global_sum - w_L_e_L_O_matrix_binned_local[j])
                / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        eL_jn_local = np.array(
            [
                (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        O_jn_local = np.array(
            [
                (w_L_O_matrix_binned_global_sum - w_L_O_matrix_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )

        # Shape assertions: jackknife arrays before einsum; M_local = M_binned
        _require(eL_jn_local.shape == (M_binned,), f"eL_jn_local shape {eL_jn_local.shape} != ({M_binned},)")
        _require(O_jn_local.shape == (M_binned, K_gF), f"O_jn_local shape {O_jn_local.shape} != ({M_binned}, {K_gF})")
        _require(eL_O_jn_local.shape == (M_binned, K_gF), f"eL_O_jn_local shape {eL_O_jn_local.shape} != ({M_binned}, {K_gF})")

        # The jackknife mean of ``conj(O)`` equals
        # ``conj(<O>)`` by linearity, so the cross-term picks up ``conj`` on
        # ``O_jn_local``. Bit-identical at Gamma.
        bar_eL_bar_O_jn_local = np.einsum("i,ij->ij", eL_jn_local, np.conj(O_jn_local))

        # The generalized force stays complex:
        # generalized_force_complex_mc): f = -2 dE/dc* = -2 <w conj(dO)(e_L - <e_L>)>. Re drives Re(c),
        # Im drives Im(c); projecting to Re here would freeze the imaginary DOF
        # of every complex parameter. Identity on the Gamma real path (the
        # covariance is real there).
        force_local = -2.0 * (eL_O_jn_local - bar_eL_bar_O_jn_local)  # (M_local, D)

        # Two-pass jackknife std (centered sum of squares) to avoid catastrophic
        # cancellation in <x^2> - <x>^2 when M_total grows large.

        # 1st pass -- global mean (complex at a twist)
        sum_local = np.sum(force_local, axis=0)  # shape (D,)
        sum_global = allreduce_sum(self._mpi_comm, sum_local)
        mean_global = sum_global / M_total

        # 2nd pass -- centered sum of squares (numerically stable). For a
        # complex estimator the variance is E|x - <x>|^2 = var(Re) + var(Im)
        # (real, used only for |f|/std diagnostics); |.|^2 == (.)^2 on the
        # real Gamma path.
        sumsq_centered_local = np.sum(np.abs(force_local - mean_global) ** 2, axis=0)  # shape (D,)
        sumsq_centered_global = allreduce_sum(self._mpi_comm, sumsq_centered_local)
        var_global = sumsq_centered_global / M_total

        ## mean and std
        generalized_force_mean = mean_global
        generalized_force_std = np.sqrt((M_total - 1) * var_global)

        logger.devel(f"generalized_force_mean = {generalized_force_mean}")
        logger.devel(f"generalized_force_std = {generalized_force_std}")
        logger.devel(f"generalized_force_mean.shape = {generalized_force_mean.shape}")
        logger.devel(f"generalized_force_std.shape = {generalized_force_std.shape}")

        return (
            generalized_force_mean,
            generalized_force_std,
        )  # (L vector, L vector)

    @staticmethod
    def _safe_stats(arr: npt.NDArray, name: str = "array") -> dict:
        """Calculate statistics safely, handling all-NaN arrays without warnings.

        Args:
            arr: Input array
            name: Name for logging

        Returns:
            dict with keys: shape, nan_count, nan_frac, min, max, mean, std
        """
        stats = {
            "shape": arr.shape,
            "nan_count": int(np.sum(np.isnan(arr))),
            "nan_frac": float(np.sum(np.isnan(arr)) / arr.size) if arr.size > 0 else 1.0,
        }

        # At a non-Gamma twist the derivative caches
        # (de_L/dc, O_k) are complex, and ``float(np.nanmin(complex))`` raises
        # ComplexWarning (unordered complex + real cast). Diagnose the MAGNITUDE
        # instead -- mirrors the ``get_dln_WF`` / SR-averages diagnostic. Guarded
        # on ``iscomplexobj`` so the real Gamma path is untouched (``np.abs`` of a
        # real array would change the sign of ``min``/``max``); bit-identical there.
        arr_mag = np.abs(arr) if np.iscomplexobj(arr) else arr

        # Only compute statistics if there are non-NaN values
        if stats["nan_frac"] < 1.0:
            with np.errstate(all="ignore"):  # Suppress warnings
                stats["min"] = float(np.nanmin(arr_mag))
                stats["max"] = float(np.nanmax(arr_mag))
                stats["mean"] = float(np.nanmean(arr_mag))
                stats["std"] = float(np.nanstd(arr_mag))
        else:
            stats["min"] = float("nan")
            stats["max"] = float("nan")
            stats["mean"] = float("nan")
            stats["std"] = float("nan")

        return stats

    @staticmethod
    def _classify_block_columns_shared(blocks: list, chosen_param_index: list | None = None) -> npt.NDArray:
        """Per-column boolean mask: True for SHARED (Jastrow) variational
        parameters, False for PER-TWIST (det = lambda / MO) parameters.

        A twist-averaged SR step
        reduces the Jastrow ``F`` / ``S`` columns over ``self._world_comm``
        (shared across all twists) and the det columns over
        ``self._mpi_comm`` (per-twist intra-comm). The classification is
        purely by ``block.name`` -- determinant blocks are the explicit
        ``lambda_matrix`` / ``mo_coefficients`` keys produced by
        :py:meth:`Geminal_data.apply_block_update`; everything else
        (J1, J2, J3, J-DTN, jastrow_*_param) is shared.

        Args:
            blocks: Ordered variational-parameter blocks (each with
                ``name`` and ``size`` attributes).
            chosen_param_index: Optional flattened-index subset; mirrors
                the same argument on :py:meth:`get_dln_WF`. When supplied
                the mask is sliced to the subset.

        Returns:
            ``np.ndarray`` of ``dtype=bool`` and length ``K`` (the total
            parameter count after the optional subset).
        """
        # Determinant blocks are per twist and reduce over the intra-twist
        # communicator. Keep ``lambda_matrix`` for the original AGP_data path
        # and include the AGPn / SD direct-MO optimization block names; each
        # parameter block reports its declared storage coordinates.
        _DET_BLOCK_NAMES = {
            "lambda_matrix",
            "mo_coefficients",
            "mo_coefficients_up",
            "mo_coefficients_dn",
            "lambda_diag",
            "lambda_unpaired",
        }
        is_shared_full: list[bool] = []
        for block in blocks:
            shared = block.name not in _DET_BLOCK_NAMES
            is_shared_full.extend([shared] * int(block.size))
        is_shared = np.asarray(is_shared_full, dtype=bool)
        if chosen_param_index is not None:
            is_shared = is_shared[np.asarray(chosen_param_index, dtype=np.int64)]
        return is_shared

    @staticmethod
    def _planb_reduce_matrix(
        M_local: npt.NDArray,
        is_shared: npt.NDArray,
        world_comm: MPI.Comm,
        intra_comm: MPI.Comm,
        total_w_world: float,
        total_w_intra: float,
    ) -> npt.NDArray:
        """Reduce a twist-decomposed SR/LM matrix over two communicators.

        Routes the shared-shared (Jastrow-Jastrow) sub-block through the
        world comm and the per-twist (det-det) sub-block through the
        intra comm, then divides each block by its block-appropriate
        normaliser. The shared-det / det-shared cross blocks are
        explicitly zeroed by the shared/twist-local block-diagonal
        approximation, which drops the off-diagonal Fisher entries.

        Single-twist callers (``world_comm is intra_comm``) short-circuit
        to a single reduction over the shared comm and the function is
        byte-identical to the ordinary single-communicator form.

        Args:
            M_local: ``(K, K)`` per-rank local matrix.
            is_shared: ``(K,)`` bool, True where the column belongs to a
                shared (Jastrow) parameter.
            world_comm: All-rank comm for the shared block.
            intra_comm: Per-twist comm for the det block.
            total_w_world: ``sum_w`` reduced over ``world_comm``.
            total_w_intra: ``sum_w`` reduced over ``intra_comm``.

        Returns:
            ``(K, K)`` matrix with the shared/twist-local block structure.
        """
        if world_comm is intra_comm:
            M = allreduce_sum(intra_comm, M_local)
            return M / total_w_intra
        # Different twists may legitimately enter with float64 (Gamma) and
        # complex128 (non-Gamma) local arrays.  MPI collectives require every
        # participant to use the same buffer datatype, so negotiate the
        # world-wide field before any block reduction.
        world_is_complex = bool(world_comm.allreduce(int(np.iscomplexobj(M_local)), op=MPI.MAX))
        work_dtype = np.complex128 if world_is_complex else np.float64
        M_local = np.asarray(M_local, dtype=work_dtype)
        K = M_local.shape[0]
        M_global = np.zeros((K, K), dtype=work_dtype)
        J_idx = np.where(is_shared)[0]
        D_idx = np.where(~is_shared)[0]
        if J_idx.size > 0:
            M_JJ_local = M_local[np.ix_(J_idx, J_idx)]
            # Force a contiguous view -- the ``np.ix_`` fancy-index above
            # returns a non-contiguous slice in general, and the MPI
            # buffer-protocol path (``allreduce_sum`` -> ``MPI.DOUBLE``
            # view) requires a C-contiguous source.
            M_JJ_local = np.ascontiguousarray(M_JJ_local)
            M_JJ_global = allreduce_sum(world_comm, M_JJ_local)
            M_global[np.ix_(J_idx, J_idx)] = M_JJ_global / total_w_world
        if D_idx.size > 0:
            M_DD_local = np.ascontiguousarray(M_local[np.ix_(D_idx, D_idx)])
            M_DD_global = allreduce_sum(intra_comm, M_DD_local)
            M_global[np.ix_(D_idx, D_idx)] = M_DD_global / total_w_intra
        return M_global

    @staticmethod
    def _planb_reduce_vector(
        v_local: npt.NDArray,
        is_shared: npt.NDArray,
        world_comm: MPI.Comm,
        intra_comm: MPI.Comm,
        total_w_world: float,
        total_w_intra: float,
    ) -> npt.NDArray:
        """Two-comm reduction of a ``(K,)`` SR/LM vector (e.g. ``f``).

        The Jastrow entries reduce over the world communicator and the
        determinant entries over the twist-local communicator, each divided
        by its block-appropriate normalizer.
        Single-twist short-circuits to the single-comm path.
        """
        if world_comm is intra_comm:
            return allreduce_sum(intra_comm, v_local) / total_w_intra
        # See the matrix twin above: Gamma/non-Gamma ranks must agree on the
        # MPI datatype before reducing the shared slice.
        world_is_complex = bool(world_comm.allreduce(int(np.iscomplexobj(v_local)), op=MPI.MAX))
        work_dtype = np.complex128 if world_is_complex else np.float64
        v_local = np.asarray(v_local, dtype=work_dtype)
        v_global = np.zeros(v_local.shape, dtype=work_dtype)
        J_idx = np.where(is_shared)[0]
        D_idx = np.where(~is_shared)[0]
        if J_idx.size > 0:
            v_J_global = allreduce_sum(world_comm, np.ascontiguousarray(v_local[J_idx]))
            v_global[J_idx] = v_J_global / total_w_world
        if D_idx.size > 0:
            v_D_global = allreduce_sum(intra_comm, np.ascontiguousarray(v_local[D_idx]))
            v_global[D_idx] = v_D_global / total_w_intra
        return v_global

    def get_aH(
        self,
        blocks: list,
        g: npt.NDArray | None = None,
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: list | None = None,
        return_matrices: bool = False,
        collective_obs: npt.NDArray | None = None,
    ) -> tuple:
        r"""Compute aSR scalars or full LM matrices.

        **aSR mode** (``return_matrices=False``, default):

        Using the decomposition

        .. math::

            E_{\alpha+\delta\alpha}
            = \frac{H_0 + 2\gamma H_1 + \gamma^2 H_2}{1 + \gamma^2 S_2},

        where :math:`\delta\alpha = \gamma g` and :math:`g = S^{-1}f`.

        Returns ``(H_0, H_1, H_2, S_2)`` -- scalars for aSR gamma optimization.
        Requires ``g`` (natural gradient direction).

        **LM mode** (``return_matrices=True``):

        Builds the full S, K, B matrices and force vector f for
        the Linear Method generalized eigenvalue problem.
        ``g`` is ignored.

        Returns ``(H_0, f_vec, S_matrix, K_matrix, B_matrix, bra_row)``.

        Args:
            g (npt.NDArray | None): Natural gradient vector :math:`g = S^{-1}f`.
                Required when ``return_matrices=False``. Ignored when ``return_matrices=True``.
            blocks (list): Ordered variational blocks (same ordering as used for ``g``).
            num_mcmc_warmup_steps (int, optional): Samples to discard as warmup. Defaults to 50.
            chosen_param_index (list | None, optional): Optional subset of flattened parameter indices. Defaults to None.
            return_matrices (bool, optional): If True, return full matrices for LM. Defaults to False.
            collective_obs (npt.NDArray | None, optional): Precomputed collective-variable
                observable ``O_SR = dO @ g`` per sample, shape ``(N,)``, for the LM
                matrices; when given, only the ``chosen_param_index`` subspace of
                ``d ln Psi / dc`` is fetched. Defaults to None.

        Returns:
            tuple: ``(H_0, H_1, H_2, S_2)`` if ``return_matrices=False``, or
            ``(H_0, f_vec, S_matrix, K_matrix, B_matrix, bra_row)`` if
            ``return_matrices=True``.

        Raises:
            RuntimeError: If ``compute_log_WF_param_deriv`` or ``comput_e_L_param_deriv`` is False.
        """
        if not self.__comput_log_WF_param_deriv:
            raise RuntimeError("get_aH requires compute_log_WF_param_deriv=True.")
        if not self.__comput_e_L_param_deriv:
            raise RuntimeError("get_aH requires comput_e_L_param_deriv=True (for B matrix / de_L/dc).")

        # Diagnostic output at entry
        logger.devel("[get_aH] ENTRY (return_matrices=%s)", return_matrices)
        if g is not None:
            g_stats = self._safe_stats(g, "g")
            logger.devel(
                f"[get_aH] INPUT g: shape={g_stats['shape']} "
                f"NaN={g_stats['nan_count']}/{g.size} min={g_stats['min']:.3e} "
                f"max={g_stats['max']:.3e} mean={g_stats['mean']:.3e}"
            )
        else:
            logger.devel("[get_aH] INPUT g: None (LM mode)")

        # ---- raw samples after warmup ----
        w_L = self.w_L[num_mcmc_warmup_steps:]  # (M, nw)
        e_L = self.e_L[num_mcmc_warmup_steps:]  # (M, nw)

        # ---- O_matrix: d ln Psi / dc  shape (M, nw, K) ----
        # When building LM matrices with a collective variable:
        #   - If collective_obs is provided (pre-computed O_SR), only fetch subspace params.
        #   - If g is provided but no collective_obs, fetch full params (to compute O_SR = dO @ g).
        _need_full_O = return_matrices and g is not None and collective_obs is None
        _cpi_for_dln = None if _need_full_O else chosen_param_index
        O_matrix = self.get_dln_WF(
            blocks=blocks,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            chosen_param_index=_cpi_for_dln,
        )  # (M, nw, K_full) or (M, nw, K_sub)

        # Diagnostics: block.values (to detect parameter divergence)
        for block in blocks:
            vals = np.asarray(block.values)
            val_stats = self._safe_stats(vals, f"{block.name}.values")
            logger.devel(
                f"[get_aH] block.values['{block.name}']: shape={val_stats['shape']} "
                f"NaN={val_stats['nan_count']}/{vals.size} ({val_stats['nan_frac']:.2%}) "
                f"min={val_stats['min']:.3e} max={val_stats['max']:.3e} "
                f"mean={val_stats['mean']:.3e} std={val_stats['std']:.3e}"
            )

        # ---- dE_matrix: de_L / dc  shape (M, nw, K) ----
        de_L_dc_map = self.de_L_dc

        # Diagnostics: check de_L_dc_map
        logger.devel(f"[get_aH] de_L_dc_map keys: {list(de_L_dc_map.keys())}")
        for block_name, arr in de_L_dc_map.items():
            arr_stats = self._safe_stats(arr, f"de_L_dc[{block_name}]")
            logger.devel(
                f"[get_aH] de_L_dc['{block_name}']: shape={arr_stats['shape']} "
                f"NaN={arr_stats['nan_count']}/{arr.size} ({arr_stats['nan_frac']:.2%}) "
                f"min={arr_stats['min']:.3e} max={arr_stats['max']:.3e} "
                f"mean={arr_stats['mean']:.3e}"
            )

        de_L_dc_list = []
        for block in blocks:
            if block.name in de_L_dc_map:
                arr = de_L_dc_map[block.name]
                if arr.ndim == 2:
                    arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
                else:
                    arr = arr.reshape(arr.shape[0], arr.shape[1], int(np.prod(arr.shape[2:])))
                de_L_dc_list.append(arr)
        dE_matrix = np.concatenate(de_L_dc_list, axis=2)[num_mcmc_warmup_steps:]  # (M, nw, K)

        # Slice dE_matrix to match O_matrix, UNLESS we need full dE for
        # collective variable dE_SR computation (collective_obs path).
        _need_full_dE = return_matrices and g is not None and collective_obs is not None
        if _need_full_dE:
            pass  # keep full dE_matrix; will slice after computing dE_SR
        elif _cpi_for_dln is not None or (chosen_param_index is not None and not (return_matrices and g is not None)):
            dE_matrix = dE_matrix[:, :, chosen_param_index]
        # else: LM fallback (no collective_obs) -- keep full dE_matrix, slice later

        # Diagnostics: dE_matrix
        dE_matrix_stats = self._safe_stats(dE_matrix, "dE_matrix")
        logger.devel(
            f"[get_aH] dE_matrix: shape={dE_matrix_stats['shape']} "
            f"NaN={dE_matrix_stats['nan_count']}/{dE_matrix.size} ({dE_matrix_stats['nan_frac']:.2%}) "
            f"min={dE_matrix_stats['min']:.3e} max={dE_matrix_stats['max']:.3e} "
            f"mean={dE_matrix_stats['mean']:.3e}"
        )

        # ---- flatten (M, nw) -> (N,) ----
        w_flat = np.ravel(w_L)  # (N,)
        e_flat = np.ravel(e_L)  # (N,)
        N = w_flat.shape[0]
        K = O_matrix.shape[2]
        K_dE = dE_matrix.shape[2]  # may differ from K when dE is kept full for collective_obs
        O_flat = O_matrix.reshape(N, K)  # (N, K)
        dE_flat = dE_matrix.reshape(N, K_dE)  # (N, K_dE)

        # Shape assertions
        _require(w_flat.shape == (N,), f"w_flat shape {w_flat.shape} != ({N},)")
        _require(e_flat.shape == (N,), f"e_flat shape {e_flat.shape} != ({N},)")
        _require(O_flat.shape == (N, K), f"O_flat shape {O_flat.shape} != ({N}, {K})")
        _require(dE_flat.shape == (N, K_dE), f"dE_flat shape {dE_flat.shape} != ({N}, {K_dE})")
        if collective_obs is not None:
            _require(collective_obs.shape == (N,), f"collective_obs shape {collective_obs.shape} != ({N},)")
            _require(K_dE >= K, f"dE must be full when collective_obs is used: K_dE={K_dE} < K={K}")
        else:
            _require(K_dE == K, f"O and dE dimension mismatch: K={K} != K_dE={K_dE}")
        if g is not None:
            _require(g.shape[0] == K_dE, f"g dimension {g.shape[0]} != K_dE={K_dE}")

        # Diagnostics: dE_flat
        dE_flat_stats = self._safe_stats(dE_flat, "dE_flat")
        logger.devel(
            f"[get_aH] dE_flat: shape={dE_flat_stats['shape']} "
            f"NaN={dE_flat_stats['nan_count']}/{dE_flat.size} ({dE_flat_stats['nan_frac']:.2%}) "
            f"min={dE_flat_stats['min']:.3e} max={dE_flat_stats['max']:.3e} "
            f"mean={dE_flat_stats['mean']:.3e} std={dE_flat_stats['std']:.3e}"
        )

        # ---- MPI-aware weighted averages ----
        total_w_local = np.sum(w_flat)
        we_local = np.dot(w_flat, e_flat)
        wO_local = np.einsum("i,ik->k", w_flat, O_flat)  # (K,)
        wdE_local = np.einsum("i,ik->k", w_flat, dE_flat)  # (K_dE,)

        # Diagnostics: wdE_local (BEFORE MPI Allreduce)
        wdE_local_stats = self._safe_stats(wdE_local, "wdE_local")
        logger.devel(
            f"[get_aH] wdE_local (BEFORE Allreduce): shape={wdE_local_stats['shape']} "
            f"NaN={wdE_local_stats['nan_count']}/{wdE_local.size} ({wdE_local_stats['nan_frac']:.2%}) "
            f"min={wdE_local_stats['min']:.3e} max={wdE_local_stats['max']:.3e} "
            f"mean={wdE_local_stats['mean']:.3e} std={wdE_local_stats['std']:.3e}"
        )

        # A twist-averaged SR step
        # needs BOTH the per-twist intra-comm total weight (for det /
        # MO column normalisers) AND the all-rank world-comm total
        # weight (for shared Jastrow column normalisers). For a single-
        # twist run ``self._world_comm is self._mpi_comm`` so the two
        # values coincide and the downstream shared/local routing reduces to
        # the single-communicator path bit-for-bit.
        total_w = self._mpi_comm.allreduce(total_w_local, op=MPI.SUM)
        if self._world_comm is self._mpi_comm:
            total_w_world = total_w
        else:
            total_w_world = self._world_comm.allreduce(total_w_local, op=MPI.SUM)

        # ``e_bar`` / ``O_bar`` / ``dE_bar`` are means used for centering
        # below; they stay on the per-twist (intra) reduction. The Plan
        # B world-reduction handles the SR matrix / vector assembly,
        # not the per-walker centering.
        #
        # ``wO`` and ``wdE`` are covariance quantities;
        # their means ``O_bar`` / ``dE_bar`` center ``dO`` / ``ddE``, which feed
        # the Hermitian S / K / B via ``conj().T``, so the Im part MATTERS and
        # must survive the reduction. Route through the complex-aware
        # ``allreduce_sum`` (bit-identical MPI.DOUBLE path for real buffers,
        # preserves Im via the float64 byte-view for complex); a real
        # ``MPI.DOUBLE`` buffer would drop Im (and fail on a complex
        # send-buffer / real recv-buffer datatype mismatch at k != 0).
        we_global = allreduce_sum(self._mpi_comm, np.atleast_1d(we_local))
        wO_global = allreduce_sum(self._mpi_comm, wO_local)
        wdE_global = allreduce_sum(self._mpi_comm, wdE_local)

        # Diagnostics: wdE_global (AFTER Allreduce)
        wdE_global_stats = self._safe_stats(wdE_global, "wdE_global")
        logger.devel(
            f"[get_aH] wdE_global (AFTER Allreduce): shape={wdE_global_stats['shape']} "
            f"NaN={wdE_global_stats['nan_count']}/{wdE_global.size} ({wdE_global_stats['nan_frac']:.2%}) "
            f"min={wdE_global_stats['min']:.3e} max={wdE_global_stats['max']:.3e} "
            f"mean={wdE_global_stats['mean']:.3e} std={wdE_global_stats['std']:.3e}"
        )

        # ``e_bar`` (== ``H_0``) is a physical-real energy;
        # a complex-twist ``e_L`` gives it a finite-sampling Im part that is a
        # projection artifact, not signal. Deliberately ``np.real``-project it
        # (matches the SR ``e_L_bar`` fix / ``get_gF``) -- identity at Gamma. The
        # Im part cancels in the centered force regardless (``sum_w dO == 0``).
        e_bar = np.real(we_global[0]) / total_w  # scalar (physical-real energy)
        O_bar = wO_global / total_w  # (K,) -- covariance mean, Im preserved
        dE_bar = wdE_global / total_w  # (K,) -- covariance mean, Im preserved

        # Diagnostics: dE_bar
        dE_bar_stats = self._safe_stats(dE_bar, "dE_bar")
        logger.devel(
            f"[get_aH] dE_bar: shape={dE_bar_stats['shape']} "
            f"NaN={dE_bar_stats['nan_count']}/{dE_bar.size} ({dE_bar_stats['nan_frac']:.2%}) "
            f"min={dE_bar_stats['min']:.3e} max={dE_bar_stats['max']:.3e} "
            f"mean={dE_bar_stats['mean']:.3e} std={dE_bar_stats['std']:.3e}"
        )

        # ---- H_0 ----
        H_0 = e_bar

        # ---- f_k = -2 <w (e_L - E)(O_k - O_bar_k)>  (generalized force) ----
        # Classify columns by parameter
        # block and route each column slice through the appropriate
        # comm. Single-twist callers (world == intra) short-circuit
        # to the single-communicator SUM via the scope-aware vector reduction.
        is_shared_col = self._classify_block_columns_shared(blocks, chosen_param_index)
        # Sanity: when fetching the full O_matrix for the SR collective
        # variable path (``_need_full_O``) the column count is
        # ``K_full``, not the subset width. Recompute the mask without
        # the chosen-param-index subset in that one case so the column
        # count matches dO_flat.
        if is_shared_col.size != K:
            is_shared_col = self._classify_block_columns_shared(blocks)
            _require(
                is_shared_col.size == K,
                f"Shared/local column-classifier length {is_shared_col.size} != f-vector K={K}; "
                "check blocks / chosen_param_index passthrough.",
            )
        dO_flat = O_flat - O_bar[np.newaxis, :]  # (N, K)
        # The generalized force is the COMPLEX covariance
        # ``f_k = -2 <w conj(O_k - <O_k>)(e_L - <e_L>)> = -2 dE/dc*``
        # Re drives Re(c), while Im drives
        # Im(c); no Re projection -- ``conj`` is the identity on the real Gamma
        # path (bit-identical there).
        f_local = -2.0 * np.einsum("i,i,ik->k", w_flat, (e_flat - e_bar), np.conj(dO_flat))
        f_vec = self._planb_reduce_vector(f_local, is_shared_col, self._world_comm, self._mpi_comm, total_w_world, total_w)
        # f_vec is the covariance-only force. The full derivative
        # dE/dc = 2cov + D (chain rule) also contains the finite-sample
        # <de_L/dc> term D; in this sign convention (f = -dE/dc) the full
        # force would be f_vec - dE_bar. The SR/aSR direction g = S^-1 f
        # uses the covariance-only force, here and in the second get_aH
        # (sample-store path) of this file.

        if return_matrices:
            # ---- LM mode: build full S, K, B matrices ----
            ddE_flat = dE_flat - dE_bar[np.newaxis, :]  # (N, K)

            # If g (SR direction) is provided, add collective variable to subspace.
            if g is not None:
                if collective_obs is not None:
                    # Memory-efficient path: O_SR pre-computed during SR solve.
                    # dO_flat is already subspace-only (fetched with chosen_param_index).
                    O_SR = collective_obs  # (N_local,)
                    # dE_SR needs full ddE -- ddE_flat is full K_full here because
                    # dE_matrix was NOT sliced by _cpi_for_dln (see else branch below).
                    dE_SR = ddE_flat @ g  # (N,)
                    # Slice ddE to subspace (dO_flat is already subspace)
                    if chosen_param_index is not None:
                        ddE_flat = ddE_flat[:, chosen_param_index]
                else:
                    # Fallback: compute O_SR from full dO (needs full O_matrix).
                    O_SR = dO_flat @ g  # (N,)
                    dE_SR = ddE_flat @ g  # (N,)
                    if chosen_param_index is not None:
                        dO_flat = dO_flat[:, chosen_param_index]
                        ddE_flat = ddE_flat[:, chosen_param_index]

                # Prepend collective variable column
                dO_flat = np.column_stack([O_SR, dO_flat])  # (N, p+1)
                ddE_flat = np.column_stack([dE_SR, ddE_flat])  # (N, p+1)
                K = dO_flat.shape[1]

                # Recompute f_vec for the extended parameter space.
                # The prepended SR collective
                # variable column is treated as PER-TWIST (det-side) --
                # it carries the SR direction in the det subspace, so
                # the extended-space ``is_shared`` mask gets a leading
                # ``False`` and the existing mask is appended after.
                #
                # ``is_shared_col`` is either subspace-sized because it was already
                # subset by ``chosen_param_index``, or full-sized on the
                # ``need_full_O`` recompute branch. Subset only in the latter case so
                # a mask matching dO_flat's (subspace + 1) column count.
                _target_pre_concat = K - 1
                if chosen_param_index is not None and is_shared_col.size != _target_pre_concat:
                    is_shared_for_dO = is_shared_col[chosen_param_index]
                else:
                    is_shared_for_dO = is_shared_col
                _is_shared_ext = np.concatenate(([False], is_shared_for_dO))
                # Use the same complex covariance force as the base space above;
                # the prepended SR collective column is complex
                # at k != 0 too. Bit-identical at Gamma.
                f_local_ext = -2.0 * np.einsum("i,i,ik->k", w_flat, (e_flat - e_bar), np.conj(dO_flat))
                f_vec = self._planb_reduce_vector(
                    f_local_ext, _is_shared_ext, self._world_comm, self._mpi_comm, total_w_world, total_w
                )
                # Carry the extended mask into the S/K/B reductions below.
                is_shared_col = _is_shared_ext

            # Shape assertions before matrix construction
            _require(
                dO_flat.shape == ddE_flat.shape,
                f"dO/ddE shape mismatch after collective variable: dO={dO_flat.shape} ddE={ddE_flat.shape}",
            )
            _require(dO_flat.shape[0] == N, f"dO_flat samples {dO_flat.shape[0]} != N={N}")
            _require(f_vec.shape == (K,), f"f_vec shape {f_vec.shape} != ({K},)")

            # S_matrix = <w * conj(O_k) * O_l>_w  (Hermitian PSD overlap).
            # Use ``T -> conj().T`` so the left
            # factor carries the conjugation; bit-identical on the real
            # path since ``conj`` of a float64 array is the identity.
            # With JAX's ``grad(ln|Psi|, c) = O`` (un-conjugated), the
            # formula ``conj(dO).T @ (w * dO)`` evaluates to
            # ``<conj(O_k) O_l * w>``.
            #
            # The matrix reduction
            # routes Jastrow-Jastrow columns through ``world_comm`` and
            # det-det columns through ``intra_comm``; the cross blocks
            # (Jastrow <-> det) are dropped by the shared/twist-local
            # block-diagonal Fisher-information approximation. Single-twist
            # callers short-circuit to a single-comm reduction (the
            # single-communicator path, byte-identical).
            w_dO = w_flat[:, np.newaxis] * dO_flat  # (N, K)
            S_local = dO_flat.conj().T @ w_dO  # (K, K)
            S_matrix = self._planb_reduce_matrix(
                S_local, is_shared_col, self._world_comm, self._mpi_comm, total_w_world, total_w
            )

            # K_matrix = <w * e_L * conj(O_k) * O_l>_w.
            we_dO = (w_flat * e_flat)[:, np.newaxis] * dO_flat  # (N, K)
            K_local = dO_flat.conj().T @ we_dO  # (K, K)
            K_matrix = self._planb_reduce_matrix(
                K_local, is_shared_col, self._world_comm, self._mpi_comm, total_w_world, total_w
            )

            # B_matrix = <w * conj(ddE_k) * O_l>_w.
            w_ddE = w_flat[:, np.newaxis] * ddE_flat  # (N, K)
            B_local = w_ddE.conj().T @ dO_flat  # (K, K)
            B_matrix = self._planb_reduce_matrix(
                B_local, is_shared_col, self._world_comm, self._mpi_comm, total_w_world, total_w
            )

            # Final shape assertions
            _require(S_matrix.shape == (K, K), f"S_matrix shape {S_matrix.shape} != ({K}, {K})")
            _require(K_matrix.shape == (K, K), f"K_matrix shape {K_matrix.shape} != ({K}, {K})")
            _require(B_matrix.shape == (K, K), f"B_matrix shape {B_matrix.shape} != ({K}, {K})")
            _require(f_vec.shape == (K,), f"f_vec shape {f_vec.shape} != ({K},)")

            # ---- non-symmetric bra row (doc: lm_storage_augmented_matrices) --
            # H_0a = <dO_a de_L> + D_a: the UNCONJUGATED covariance plus the
            # mean local-energy derivative. The ket column stays -f_vec/2
            # (conjugated covariance). Under exact sampling the two coincide
            # (Hermiticity); their difference is the finite-sampling
            # information the non-symmetric LM retains (far root cause).
            # ``dO_flat`` here is the (possibly CV-extended) column set, so the
            # unconjugated covariance covers every pencil column in one pass.
            f_noconj_local = -2.0 * np.einsum("i,i,ik->k", w_flat, (e_flat - e_bar), dO_flat)
            f_noconj = self._planb_reduce_vector(
                f_noconj_local, is_shared_col, self._world_comm, self._mpi_comm, total_w_world, total_w
            )
            if g is not None:
                # Extended space: the CV column's mean derivative is the
                # UNCENTERED <de_L/dc> . g (dE_bar and g are full-width in both
                # collective branches); parameter columns use the subspace
                # slice matching ddE above.
                dE_bar_sub = dE_bar if dE_bar.shape[0] == K - 1 else dE_bar[chosen_param_index]
                D_vec = np.concatenate([np.atleast_1d(dE_bar @ g), dE_bar_sub])
            else:
                D_vec = dE_bar if dE_bar.shape[0] == K else dE_bar[chosen_param_index]
            bra_row = -0.5 * f_noconj + D_vec

            _p_label = f"p={K}" + (" (incl. SR collective)" if g is not None else "")
            logger.info(
                f"  LM matrices: {_p_label}, H_0={H_0:.6f}, "
                f"||f||={np.linalg.norm(f_vec):.3e}, "
                f"||D||=||<de_L/dc>||={np.linalg.norm(D_vec):.3e}, "
                f"diag(S): min={np.min(np.diag(S_matrix)):.3e} max={np.max(np.diag(S_matrix)):.3e}"
            )
            return H_0, f_vec, S_matrix, K_matrix, B_matrix, bra_row

        # ---- aSR mode: scalar projections ----
        _require(g is not None, "g is required for aSR mode (return_matrices=False)")

        # ---- H_1: NON-SYMMETRIC pencil average along g ----
        # The rational model E(gamma) = (H_0 + 2 gamma H_1 + gamma^2 H_2)/(1 +
        # gamma^2 S_2) uses the SUM of the two off-diagonal pencil elements,
        # 2 H_1 = H_01 + H_10. Evaluated by their own estimators (non-symmetric
        # linear method.
        # ``lm_storage_augmented_matrices``):
        #   H_10 (ket)  = C_g  = -1/2 Re(vdot(g, f))          (covariance form)
        #   H_01 (bra)  = C_g + D_g,  D_g = Re(vdot(g, <de_L/dc>))
        # Under exact |Psi|^2 sampling D_g = 0 (Hermiticity) and this reduces
        # to the symmetric H_1 = C_g. On an imperfectly equilibrated
        # sample D_g is finite (measured on a periodic slab: O(1-8) Ha along DTN
        # axes) and discarding it makes the gamma model blind to the true slope
        # (a 31 mHa uphill walk of the optimizer was measured). The inner
        # products carry conj on the DIRECTION factor (vdot); vdot == dot at
        # Gamma (bit-identical there when D_g = 0).
        H_1_ket = -0.5 * np.real(np.vdot(g, f_vec))
        D_g = np.real(np.vdot(g, dE_bar))
        H_1 = H_1_ket + 0.5 * D_g
        logger.info(f"aSR: H_1(ket/cov) = {H_1_ket:.6f}, D_g = <g, de_L/dc> = {D_g:.6f}.")

        # ---- S_2 = g^T S g = <w * (g^T dO)^2>_w  (exact, computed from samples) ----
        # Do NOT use S_2 = g^T f = -2*H_1.  The SR solved (S_scaled + sr_epsilon*I) g_scaled = b,
        # so  g^T f = g^T S g + sr_epsilon * ||g_scaled||^2.
        # The sr_epsilon correction term overestimates S_2, making the denominator of
        # E(gamma) = (H0 + 2*gamma*H1 + gamma^2*H2) / (1 + gamma^2*S2) grow too fast,
        # which in turn makes the optimal gamma unrealistically small.
        gdO_flat = dO_flat @ g  # (N,)
        # Use ``|x|**2`` so the metric stays
        # real-positive when ``g`` and ``dO`` are complex. Bit-identical
        # on the real path (``np.abs(x)**2 == x**2`` for real ``x``).
        gSg_local = np.dot(w_flat, np.abs(gdO_flat) ** 2)
        gSg_arr = np.empty(1)
        self._mpi_comm.Allreduce([np.array([gSg_local]), MPI.DOUBLE], [gSg_arr, MPI.DOUBLE], op=MPI.SUM)
        S_2 = float(gSg_arr[0]) / total_w

        # ---- K matrix contribution: g^H K g = <w * e_L * |g^T dO|^2>_w ----
        # ``e_L`` is complex at k != 0, so ``gKg_local`` is
        # complex -- reduce through the complex-aware ``allreduce_sum`` (a real
        # ``MPI.DOUBLE`` buffer cannot carry Im), then ``np.real``-project the
        # PHYSICAL-REAL curvature scalar. Bit-identical at Gamma.
        gKg_local = np.einsum("i,i,i->", w_flat, e_flat, np.abs(gdO_flat) ** 2)
        gKg_arr = allreduce_sum(self._mpi_comm, np.atleast_1d(gKg_local))
        gKg = np.real(gKg_arr[0]) / total_w

        # ---- B matrix contribution: g^H B g = <w * conj(g^T dE) * (g^T dO)>_w ----
        # ``B_kl = <w * conj(ddE_k) * O_l>`` preserves Hermitian pairing,
        # so the quadratic form picks up ``conj`` on the ``ddE`` side. Bit-
        # identical on the real path.
        ddE_flat = dE_flat - dE_bar[np.newaxis, :]  # (N, K)
        gdE_flat = ddE_flat @ g  # (N,)
        # ``conj(gdE) * gdO`` is complex at k != 0;
        # reduce through ``allreduce_sum`` then ``np.real``-project the
        # PHYSICAL-REAL scalar (a real ``MPI.DOUBLE`` buffer cannot carry Im).
        # Bit-identical at Gamma.
        gBg_local = np.einsum("i,i,i->", w_flat, np.conj(gdE_flat), gdO_flat)
        gBg_arr = allreduce_sum(self._mpi_comm, np.atleast_1d(gBg_local))
        gBg = np.real(gBg_arr[0]) / total_w

        # ---- H_2 = g^T (B + K) g ----
        H_2 = gBg + gKg

        logger.info(f"aSR: H_0 = {H_0:.6f}, H_1 = {H_1:.6f}, H_2 = {H_2:.6f}, S_2 = {S_2:.6f}")
        return H_0, H_1, H_2, S_2

    def run_optimize(
        self,
        num_mcmc_steps: int = 100,
        num_opt_steps: int = 1,
        wf_dump_freq: int = 10,
        dump_wavefunction: bool = True,
        max_time: int = 86400,
        num_mcmc_warmup_steps: int = 0,
        num_mcmc_bin_blocks: int = 100,
        opt_J1_param: bool = True,
        opt_J2_param: bool = True,
        opt_J3_param: bool = True,
        opt_geminal: bool = False,
        opt_J3_basis_exp: bool = False,
        opt_J3_basis_coeff: bool = False,
        opt_lambda_basis_exp: bool = False,
        opt_lambda_basis_coeff: bool = False,
        opt_JDTN_u_param: bool = True,
        opt_JDTN_chi_param: bool = True,
        opt_JDTN_f_param: bool = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_device_optimizer: bool = True,
        fix_geminal_rank: int | None = None,
        energy_aggregator: Callable[[EnergyEstimate], EnergyEstimate] | None = None,
        sampler_weight: float | None = None,
        twist_index: int | None = None,
        *,
        _manage_external_control: bool = True,
        _more_steps_follow: bool = False,
    ) -> Literal["max_time", "stop_flag"] | None:
        """Compatibility facade for the shared wave-function optimizer.

        The implementation lives in :mod:`jqmc.jqmc_optimizer`.  MCMC remains
        responsible for sampling and estimator production.  The return value
        is :func:`jqmc.jqmc_optimizer.run_optimize`'s stop reason:
        ``"max_time"`` or ``"stop_flag"`` after an early exit, else ``None``.

        ``dump_wavefunction=False`` disables every
        ``hamiltonian_data_opt_step_*.h5`` write of this call (periodic, final,
        ``max_time`` and external-stop-flag exits alike).  The twist-averaged
        driver uses it: a dump made here would hold only the world-rank-0
        twist, so the driver writes the complete multi-twist combined file
        instead.

        ``sampler_weight`` and ``twist_index`` are set by the twist-averaged
        driver; ``twist_index`` labels the twist-local step diagnostics.
        """
        from .jqmc_optimizer import (
            OptimizationConfig,
            ParameterSelection,
            run_optimize,
        )

        config = OptimizationConfig(
            num_sampling_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            wavefunction_dump_frequency=wf_dump_freq,
            dump_wavefunction=dump_wavefunction,
            max_time=max_time,
            num_warmup_steps=num_mcmc_warmup_steps,
            num_bin_blocks=num_mcmc_bin_blocks,
            parameters=ParameterSelection(
                opt_J1_param=opt_J1_param,
                opt_J2_param=opt_J2_param,
                opt_J3_param=opt_J3_param,
                opt_geminal=opt_geminal,
                opt_J3_basis_exp=opt_J3_basis_exp,
                opt_J3_basis_coeff=opt_J3_basis_coeff,
                opt_lambda_basis_exp=opt_lambda_basis_exp,
                opt_lambda_basis_coeff=opt_lambda_basis_coeff,
                opt_JDTN_u_param=opt_JDTN_u_param,
                opt_JDTN_chi_param=opt_JDTN_chi_param,
                opt_JDTN_f_param=opt_JDTN_f_param,
            ),
            optimizer_kwargs=dict(optimizer_kwargs or {}),
            use_device_optimizer=use_device_optimizer,
            fix_geminal_rank=fix_geminal_rank,
            energy_aggregator=energy_aggregator,
            sampler_weight=sampler_weight,
            twist_index=twist_index,
            _manage_external_control=_manage_external_control,
            _more_steps_follow=_more_steps_follow,
        )
        return run_optimize(self, config)

    # ------------------------------------------------------------------ #
    #  HDF5 checkpoint save / load
    # ------------------------------------------------------------------ #

    def save_to_hdf5(self, filepath: str) -> None:
        """Write this rank's state to a temporary per-rank HDF5 file.

        The file is later merged by :func:`jqmc.checkpoint.merge_rank_checkpoints`.
        All ranks of this driver's sampling communicator must call this method;
        fresh-wavefunction validation fails collectively before any rank writes.

        Args:
            filepath: Output path (e.g. ``._restart_rank0.h5``).
        """
        from ._checkpoint import save_rank_checkpoint

        # Checkpoint payloads store positions, so a finite carried energy does
        # not prove that the saved walker can rebuild its wavefunction. Use the
        # selected backend and arithmetic policy, as run() does, and no radius
        # or small-determinant cutoff. Validate positions collectively first so
        # no rank enters a native kernel with a non-finite coordinate.
        r_up = np.asarray(self.__latest_r_up_carts, dtype=np.float64)
        r_dn = np.asarray(self.__latest_r_dn_carts, dtype=np.float64)
        finite_positions = np.all(np.isfinite(r_up), axis=(1, 2)) & np.all(np.isfinite(r_dn), axis=(1, 2))
        _raise_collective_nonfinite_observables(
            self._mpi_comm,
            driver="MCMC",
            stage="checkpoint save",
            first_step=self.__mcmc_counter,
            observables={"positions_finite": np.where(finite_positions, 1.0, np.nan)},
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )
        env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
        use_rust = flag_rust_on_CPU if env_rust is None else env_rust.strip().lower() not in ("0", "false", "no", "off", "")
        if use_rust and jax.default_backend() == "cpu":
            import jqmc_rs

            ham = jax.tree_util.tree_map(_leaf_to_fp64, self.__hamiltonian_data)
            ctx = jqmc_rs.build_obs_ctx(
                ham.wavefunction_data.geminal_data,
                ham.wavefunction_data.jastrow_data,
                ham.coulomb_potential_data,
                bool(self._wf_is_complex),
                bool(ham.structure_data.pbc_flag),
                False,
                want_force=False,
                precision_policy=precision_policy_snapshot(),
            )
            if self._wf_is_complex:
                fresh_geminal = np.asarray(ctx.sweep_seed_geminal_complex_batch(r_up, r_dn))
            else:
                cell = np.asarray(ham.structure_data.cell, dtype=np.float64) if ham.structure_data.pbc_flag else None
                n_max = int(ham.structure_data._n_max) if cell is not None else 0
                fresh_geminal = np.asarray(ctx.sweep_seed_geminal_batch(r_up, r_dn, cell, n_max))
        else:
            # Reuse the ordinary per-walker JIT, avoiding an extra walker-vmap
            # allocation for large B-spline/composite orbital tables at save.
            geminal_data = self.__hamiltonian_data.wavefunction_data.geminal_data
            fresh_geminal = np.stack(
                [np.asarray(compute_geminal_all_elements(geminal_data, up, dn)) for up, dn in zip(r_up, r_dn, strict=True)]
            )
        sign, logabsdet = np.linalg.slogdet(fresh_geminal)
        valid = np.all(np.isfinite(fresh_geminal), axis=(-2, -1)) & (sign != 0) & np.isfinite(logabsdet)
        _raise_collective_nonfinite_observables(
            self._mpi_comm,
            driver="MCMC",
            stage="checkpoint save",
            first_step=self.__mcmc_counter,
            observables={"ln_abs_det": np.where(valid, logabsdet, np.nan)},
            r_up_carts=r_up,
            r_dn_carts=r_dn,
        )

        # -- driver_config (scalar execution parameters) --
        driver_config: dict[str, Any] = {
            "mcmc_seed": self.__mcmc_seed,
            "num_walkers": self.__num_walkers,
            "num_mcmc_per_measurement": self.__num_mcmc_per_measurement,
            "Dt": self.__Dt,
            "epsilon_AS": self.__epsilon_AS,
            "comput_log_WF_param_deriv": self.__comput_log_WF_param_deriv,
            "comput_e_L_param_deriv": self.__comput_e_L_param_deriv,
            "retain_derivative_sources": self.__retain_derivative_sources,
            "comput_position_deriv": self.__comput_position_deriv,
            "comput_stress": self.__comput_stress,
            "random_discretized_mesh": self.__random_discretized_mesh,
            "use_swct": self.__use_swct,
            "mpc": self.__mpc,
            "mpc_cutoff": self.__mpc_cutoff,
            "mcmc_counter": self.__mcmc_counter,
            "accepted_moves": self.__accepted_moves,
            "rejected_moves": self.__rejected_moves,
            "i_opt": self.__i_opt,
        }

        # -- param_grad_flags (per-block gradient on/off) --
        for k, v in self.__param_grad_flags.items():
            driver_config[f"param_grad_flag_{k}"] = v

        # -- derivative sources kept only as their folded entries --
        for k in sorted(self.__log_WF_sources_in_param_grads):
            driver_config[f"log_derivative_source_in_param_grads_{k}"] = True
        for k in sorted(self.__e_L_sources_in_param_grads):
            driver_config[f"e_L_derivative_source_in_param_grads_{k}"] = True

        # -- rng_state --
        rng_state: dict[str, Any] = {
            "jax_PRNG_key_list": np.asarray(self.__jax_PRNG_key_list),
            "mpi_seed": self.__mpi_seed,
            "np_seed_words": np.asarray(self.__np_seed_words, dtype=np.uint32),
        }

        # -- walker_state --
        walker_state: dict[str, Any] = {
            "latest_r_up_carts": np.asarray(self.__latest_r_up_carts),
            "latest_r_dn_carts": np.asarray(self.__latest_r_dn_carts),
        }

        # -- observables --
        observables: dict[str, Any] = {}

        _obs_map = {
            "e_L": self.__stored_e_L,
            "e_L2": self.__stored_e_L2,
            "w_L": self.__stored_w_L,
            "mpc_dV": self.__stored_mpc_dV,
            "force_HF": self.__stored_force_HF,
            "force_PP": self.__stored_force_PP,
            "E_L_force_PP": self.__stored_E_L_force_PP,
            "stress_HF": self.__stored_stress_HF,
            "stress_PP": self.__stored_stress_PP,
            "E_L_stress_PP": self.__stored_E_L_stress_PP,
        }
        for key, val in _obs_map.items():
            arr = np.asarray(val) if val.size > 0 else np.empty(0)
            observables[key] = arr

        # dict-keyed parameter gradients
        param_grads: dict[str, np.ndarray] = {}
        for name, val_list in self.__stored_log_WF_param_grads.items():
            if len(val_list) > 0:
                param_grads[name] = np.asarray(val_list)
        observables["param_grads"] = param_grads

        e_L_param_grads: dict[str, np.ndarray] = {}
        for name, val_list in self.__stored_e_L_param_grads.items():
            if len(val_list) > 0:
                e_L_param_grads[name] = np.asarray(val_list)
        observables["e_L_param_grads"] = e_L_param_grads

        def _flatten_source_storage(storage: dict[str, dict[str, list]]) -> dict[str, np.ndarray]:
            flattened: dict[str, np.ndarray] = {}
            for name, components in storage.items():
                for component, values in components.items():
                    if values:
                        flattened[f"{name}__component__{component}"] = np.asarray(values)
            return flattened

        observables["log_derivative_sources"] = _flatten_source_storage(self.__stored_log_WF_derivative_sources)
        observables["e_L_derivative_sources"] = _flatten_source_storage(self.__stored_e_L_derivative_sources)

        # -- optimizer_state --
        self.__ensure_optimizer_runtime()
        optimizer_state = self.__optimizer_runtime

        save_rank_checkpoint(
            filepath,
            driver_type="MCMC",
            driver_config=driver_config,
            rng_state=rng_state,
            walker_state=walker_state,
            observables=observables,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def load_from_hdf5(
        cls,
        filepath: str,
        rank: int | None = None,
        *,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
        require_local_energy_derivatives: bool = False,
    ) -> "MCMC":
        """Restore an MCMC instance from a merged HDF5 checkpoint.

        This bypasses ``__init__`` entirely and reconstructs all internal
        attributes from the stored data and the embedded ``hamiltonian_data``.

        Args:
            filepath: Path to the merged checkpoint (e.g. ``restart.h5``).
            rank: MPI rank to load. Defaults to the module-level ``mpi_comm``
                rank (matches what ``__init__`` would have used).
            mpi_comm_in: This twist's per-instance sampling / per-twist det
                comm. ``None`` falls back to the module-level ``mpi_comm``
                so the single-twist restart path is unchanged. The twist-
                averaged driver passes a per-twist ``intra_comm`` here on
                resume so twist-local determinant updates use the correct
                communicator after loading.
            world_comm_in: All-rank comm for the shared-Jastrow
                reduction. ``None`` falls back to ``mpi_comm_in``.
            require_local_energy_derivatives: Enable their collection for
                VMC's linear method. Every rank must be able to supply
                derivatives for its retained samples.

        Returns:
            A fully restored :class:`MCMC` instance.
        """
        from ._checkpoint import (
            _validate_restart_rank_topology,
            load_hamiltonian_from_checkpoint,
            load_rank_checkpoint_for_restart,
            raise_on_restart_inconsistency,
        )

        # Restore the per-instance comms FIRST so ``hamiltonian_data``
        # setter and any property accesses below have the right values.
        _mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        _world_comm = world_comm_in if world_comm_in is not None else _mpi_comm
        _validate_restart_rank_topology(filepath, _mpi_comm.Get_size())
        if rank is None:
            rank = _mpi_comm.Get_rank()

        data, problems = load_rank_checkpoint_for_restart(filepath, rank)
        # The restored scalar domain is checked first by its own collective
        # validator (one call on every rank, missing -> None).
        _validate_num_mcmc_per_measurement(
            None if data is None else data["driver_config"].get("num_mcmc_per_measurement"), comm=_mpi_comm
        )
        # The Hamiltonian is the shared root group: identical on every rank.
        hamiltonian_data = load_hamiltonian_from_checkpoint(filepath)
        if data is not None:
            cfg = data["driver_config"]
            rng = data["rng_state"]
            ws = data["walker_state"]
            obs = data["observables"]
            opt = data["optimizer_state"]
            problems = cls._MCMC__restart_state_problems(cfg, rng, ws, obs, opt, hamiltonian_data, filepath)

        # Every rank checks its own group against the keys the writer always
        # writes and against its peers BEFORE any restored value gates a
        # collective (issues #805-#807). A missing key, a counter that does not
        # match the stored history, or a disagreement between ranks fails
        # identically on every rank.
        restart_records = raise_on_restart_inconsistency(
            _mpi_comm, context=f"MCMC restart from {filepath!r}", rank=rank, **problems
        )
        saved_e_L_param_deriv = bool(cfg["comput_e_L_param_deriv"])
        if require_local_energy_derivatives:
            # All ranks must reject together before entering sampling collectives.
            missing_derivatives = not saved_e_L_param_deriv and any(
                np.size(obs.get(name, ())) for name in ("e_L", "e_L2", "w_L")
            )
            if _mpi_comm.allreduce(missing_derivatives, op=MPI.LOR):
                raise ValueError(
                    f"Cannot enable local-energy parameter derivatives during restart from {filepath!r}: "
                    "the checkpoint contains retained samples without these derivatives."
                )

        # Create an empty instance without calling __init__
        obj = cls.__new__(cls)
        log_build_info_once(logger)  # a restart-only Python API run also reports its code version (PR #184)

        # -- Per-instance comms (parallel of __init__ lines 535-538;
        # MCMC.run / get_E / twist-decomposed get_aH read these). Without them
        # the loaded MCMC raises AttributeError on the first sampling
        # step -- a latent bug in the pre-edit load path. --
        obj._mpi_comm = _mpi_comm
        obj._world_comm = _world_comm
        obj._mpi_rank = _mpi_comm.Get_rank()
        obj._mpi_size = _mpi_comm.Get_size()

        # -- Scalar config --
        obj._MCMC__mcmc_seed = cfg["mcmc_seed"]
        obj._MCMC__num_walkers = cfg["num_walkers"]
        obj._MCMC__num_mcmc_per_measurement = cfg["num_mcmc_per_measurement"]
        obj._MCMC__Dt = cfg["Dt"]
        obj._MCMC__epsilon_AS = cfg["epsilon_AS"]
        obj._MCMC__comput_log_WF_param_deriv = bool(cfg["comput_log_WF_param_deriv"])
        obj._MCMC__comput_e_L_param_deriv = saved_e_L_param_deriv or require_local_energy_derivatives
        obj._MCMC__retain_derivative_sources = bool(cfg["retain_derivative_sources"])
        obj._MCMC__comput_position_deriv = bool(cfg["comput_position_deriv"])
        obj._MCMC__comput_stress = bool(cfg["comput_stress"])
        obj._MCMC__random_discretized_mesh = bool(cfg["random_discretized_mesh"])
        obj._MCMC__use_swct = bool(cfg["use_swct"])
        # MPC flags must land BEFORE the ``hamiltonian_data`` setter below:
        # ``__init_attributes`` reads them to rebuild the MPC tables + closure.
        obj._MCMC__mpc = bool(cfg["mpc"])
        obj._MCMC__mpc_cutoff = float(cfg["mpc_cutoff"])

        # Counters
        obj._MCMC__mcmc_counter = cfg["mcmc_counter"]
        obj._MCMC__accepted_moves = cfg["accepted_moves"]
        obj._MCMC__rejected_moves = cfg["rejected_moves"]
        obj._MCMC__i_opt = cfg["i_opt"]

        # -- param_grad_flags --
        flags = cls._MCMC__default_param_grad_flags()
        for k in flags:
            flags[k] = bool(cfg[f"param_grad_flag_{k}"])
        obj._MCMC__param_grad_flags = flags

        # -- Hamiltonian data (store state and reinitialize sampler attributes) --
        obj._MCMC__hamiltonian_data = hamiltonian_data
        obj.hamiltonian_data = hamiltonian_data  # triggers setter -> __init_attributes

        # -- Overwrite __init_attributes results with loaded state --
        obj._MCMC__mcmc_counter = cfg["mcmc_counter"]
        obj._MCMC__accepted_moves = cfg["accepted_moves"]
        obj._MCMC__rejected_moves = cfg["rejected_moves"]

        # -- RNG state -- (restore the SAVED derived values; never re-derive
        # the advanced key list -- resume must continue the stored streams)
        obj._MCMC__mpi_seed = rng["mpi_seed"]
        obj._MCMC__jax_PRNG_key = jax.random.PRNGKey(rng["mpi_seed"])
        obj._MCMC__jax_PRNG_key_list = jnp.array(rng["jax_PRNG_key_list"])
        obj._MCMC__np_seed_words = np.asarray(rng["np_seed_words"], dtype=np.uint32)

        # -- Walker state --
        # (walker shapes were validated collectively above)
        from ._checkpoint import validate_restored_walker_finiteness_collectively

        validate_restored_walker_finiteness_collectively(
            _mpi_comm,
            driver="MCMC",
            filepath=filepath,
            r_up_carts=ws["latest_r_up_carts"],
            r_dn_carts=ws["latest_r_dn_carts"],
        )
        obj._MCMC__latest_r_up_carts = jnp.array(ws["latest_r_up_carts"])
        obj._MCMC__latest_r_dn_carts = jnp.array(ws["latest_r_dn_carts"])

        # -- only_up_electron (derived from hamiltonian) --
        tot_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        obj._MCMC__only_up_electron = tot_dn == 0

        # -- Observables (keep __init_attributes defaults for missing/empty data) --
        def _load_obs(obs_arr: npt.ArrayLike | None, default: np.ndarray) -> np.ndarray:
            """Return loaded ndarray if non-empty, otherwise keep the shaped default."""
            if obs_arr is None or (isinstance(obs_arr, np.ndarray) and obs_arr.size == 0):
                return default
            return np.asarray(obs_arr)

        def _to_list(arr: npt.ArrayLike | None) -> list[np.ndarray]:
            """Convert ndarray back to list-of-arrays (for dict-keyed param grads)."""
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                return []
            return [arr[i] for i in range(arr.shape[0])]

        obj._MCMC__stored_e_L = _load_obs(obs.get("e_L"), obj._MCMC__stored_e_L)
        obj._MCMC__stored_e_L2 = _load_obs(obs.get("e_L2"), obj._MCMC__stored_e_L2)
        obj._MCMC__stored_w_L = _load_obs(obs.get("w_L"), obj._MCMC__stored_w_L)
        obj._MCMC__stored_mpc_dV = _load_obs(obs.get("mpc_dV"), obj._MCMC__stored_mpc_dV)
        obj._MCMC__stored_force_HF = _load_obs(obs.get("force_HF"), obj._MCMC__stored_force_HF)
        obj._MCMC__stored_force_PP = _load_obs(obs.get("force_PP"), obj._MCMC__stored_force_PP)
        obj._MCMC__stored_E_L_force_PP = _load_obs(obs.get("E_L_force_PP"), obj._MCMC__stored_E_L_force_PP)
        if obj._MCMC__comput_stress:
            obj._MCMC__stored_stress_HF = _load_obs(obs.get("stress_HF"), obj._MCMC__stored_stress_HF)
            obj._MCMC__stored_stress_PP = _load_obs(obs.get("stress_PP"), obj._MCMC__stored_stress_PP)
            obj._MCMC__stored_E_L_stress_PP = _load_obs(obs.get("E_L_stress_PP"), obj._MCMC__stored_E_L_stress_PP)

        # dict-keyed parameter gradients
        pg = obs.get("param_grads", {})
        stored_pg: dict[str, list] = defaultdict(list)
        if isinstance(pg, dict):
            for name, arr in pg.items():
                stored_pg[name] = _to_list(arr)
        obj._MCMC__stored_log_WF_param_grads = stored_pg

        epg = obs.get("e_L_param_grads", {})
        stored_epg: dict[str, list] = defaultdict(list)
        if isinstance(epg, dict):
            for name, arr in epg.items():
                stored_epg[name] = _to_list(arr)
        obj._MCMC__stored_e_L_param_grads = stored_epg

        def _restore_source_storage(payload: dict[str, npt.ArrayLike]) -> DerivativeSourceStorage:
            restored: dict[str, dict[str, list]] = {}
            if not isinstance(payload, dict):
                return restored
            for key, arr in payload.items():
                if "__component__" not in key:
                    raise ValueError(f"invalid derivative-source checkpoint key {key!r}.")
                name, component = key.rsplit("__component__", 1)
                if component not in ("direct", "d_p", "d_pbar"):
                    raise ValueError(f"invalid derivative-source component {component!r} in checkpoint.")
                restored.setdefault(name, {})[component] = _to_list(arr)
            return restored

        obj._MCMC__stored_log_WF_derivative_sources = _restore_source_storage(obs.get("log_derivative_sources", {}))
        obj._MCMC__stored_e_L_derivative_sources = _restore_source_storage(obs.get("e_L_derivative_sources", {}))

        def _restore_folded_names(prefix: str, folded: dict[str, list]) -> set[str]:
            names = {key[len(prefix) :] for key, value in cfg.items() if key.startswith(prefix) and bool(value)}
            missing = sorted(name for name in names if not folded.get(name))
            if missing:
                raise ValueError(f"checkpoint {filepath!r} lacks the folded derivatives of source blocks {missing}.")
            return names

        obj._MCMC__log_WF_sources_in_param_grads = _restore_folded_names("log_derivative_source_in_param_grads_", stored_pg)
        obj._MCMC__e_L_sources_in_param_grads = _restore_folded_names("e_L_derivative_source_in_param_grads_", stored_epg)

        _raise_collective_nonfinite_observables(
            obj._mpi_comm,
            driver="MCMC",
            stage="checkpoint load",
            first_step=0,
            observables={
                "stored_e_L": obj._MCMC__stored_e_L,
                "stored_e_L2": obj._MCMC__stored_e_L2,
                "stored_w_L": obj._MCMC__stored_w_L,
            },
            r_up_carts=obj._MCMC__latest_r_up_carts,
            r_dn_carts=obj._MCMC__latest_r_dn_carts,
        )
        obj._MCMC__restart_validation_pending = True

        # -- Optimizer runtime -- (the presence of optimizer_state agrees on
        # every rank, checked above). Only when EVERY rank lacks it is this a
        # fresh lineage, e.g. an mcmc checkpoint continued by a vmc job; then
        # every rank takes the same lineage broadcast.
        if all(record["agreed"]["optimizer_state"] == "present" for record in restart_records):
            obj._MCMC__optimizer_runtime = opt
        else:
            obj._MCMC__optimizer_runtime = None
            obj._MCMC__ensure_optimizer_runtime()

        return obj

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Number of Metropolis steps accumulated (rows in stored observables)."""
        return self.__mcmc_counter

    @property
    def num_walkers(self) -> int:
        """The number of walkers."""
        return self.__num_walkers

    @property
    def num_inverse_refreshes(self) -> int:
        """Return how many walker-steps refreshed a drifted carried geminal inverse."""
        return self.__num_inverse_refreshes

    @property
    def acceptance_ratio(self) -> float | None:
        """Fraction of accepted single-electron moves for this rank, in [0, 1].

        Computed from the rank-local accepted / (accepted + rejected) move
        counters accumulated across all :meth:`run` calls on this instance.
        Returns ``None`` when no moves have been proposed yet (ratio undefined).

        The globally MPI-reduced ratio is what the run logs as
        ``Acceptance ratio is X.XX %``; for single-rank runs the two coincide.
        The workflow-side Dt calibration reads that log line via
        :mod:`jqmc_workflow.output_parser` (``_RE_ACCEPTANCE``), so this
        property is primarily for direct/programmatic use and tests.
        """
        total = self.__accepted_moves + self.__rejected_moves
        if total <= 0:
            return None
        return self.__accepted_moves / total

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, num_walkers)."""
        return np.asarray(self.__stored_w_L)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, num_walkers)."""
        return np.asarray(self.__stored_e_L)

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L^2 array. dim: (mcmc_counter, num_walkers)."""
        return np.asarray(self.__stored_e_L2)

    # observables
    @property
    def mpc_dV(self) -> npt.NDArray:
        """Return the stored MPC correction dV = V_ee^MPC - V_ee^Ewald. dim: (mcmc_counter, num_walkers).

        Empty (zero rows) when the driver was built with ``mpc=False``. See :meth:`get_dE_mpc`.
        """
        return np.asarray(self.__stored_mpc_dV)

    @property
    def force_HF_stored(self) -> npt.NDArray:
        """Return the stored force_HF array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_force_HF)

    @property
    def force_PP_stored(self) -> npt.NDArray:
        """Return the stored force_PP array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_force_PP)

    @property
    def E_L_force_PP_stored(self) -> npt.NDArray:
        """Return the stored E_L * force_PP array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.asarray(self.__stored_E_L_force_PP)

    @property
    def stress_HF_stored(self) -> npt.NDArray:
        """Return the stored HF stress tensor array. dim: (mcmc_counter, num_walkers, 3, 3).

        Periodic systems carry per-walker sigma_HF (Hartree); molecular runs
        carry zeros of the same shape.
        """
        return np.asarray(self.__stored_stress_HF)

    @property
    def stress_PP_stored(self) -> npt.NDArray:
        """Return the stored Pulay stress tensor array. dim: (mcmc_counter, num_walkers, 3, 3)."""
        return np.asarray(self.__stored_stress_PP)

    @property
    def E_L_stress_PP_stored(self) -> npt.NDArray:
        """Return the stored E_L * sigma_PP array (covariance bookkeeping). dim: (mcmc_counter, num_walkers, 3, 3)."""
        return np.asarray(self.__stored_E_L_stress_PP)

    @property
    def dln_Psi_dc(self) -> dict[str, npt.NDArray]:
        """Return stored parameter gradients (d ln Psi / dc) keyed by block name."""
        return {name: np.asarray(values) for name, values in self.__stored_log_WF_param_grads.items()}

    @property
    def de_L_dc(self) -> dict[str, npt.NDArray]:
        """Return stored local energy parameter gradients (de_L / dc) keyed by block name."""
        return {name: np.asarray(values) for name, values in self.__stored_e_L_param_grads.items()}

    @property
    def comput_position_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. atomic positions."""
        return self.__comput_position_deriv

    @property
    def comput_stress(self) -> bool:
        """Return whether periodic stress observables are retained."""
        return self.__comput_stress

    @property
    def i_opt(self) -> int:
        """SR/LM optimization step counter -- the cumulative count of
        completed SR/LM iterations across all ``run_optimize`` calls on
        this instance. Survives :meth:`save_to_hdf5` / :meth:`load_from_hdf5`
        so a chunked twist-averaged optimizer resuming mid-stream can ask
        "where did we leave off?"
        without touching name-mangled internals.
        """
        return int(self.__i_opt)

    @property
    def compute_log_WF_param_deriv(self) -> bool:
        """Return the flag for computing the derivatives of E wrt. variational parameters."""
        return self.__comput_log_WF_param_deriv

    @property
    def comput_e_L_param_deriv(self) -> bool:
        """Return the flag for computing the local energy derivatives (de_L/dc) wrt. variational parameters."""
        return self.__comput_e_L_param_deriv


# ---------------------------------------------------------------------------
# Module-level JIT kernels for MCMC sampling
#
# These functions are defined at module scope so that they are compiled only
# once by JAX regardless of how many times ``MCMC.run()`` is called (a local
# closure re-created on every ``run()`` invocation would force a full
# re-compilation each time).
# ---------------------------------------------------------------------------


def _haar_rotation_matrix_from_uniform(uniforms: jax.Array) -> jax.Array:
    """Map three U[0,1) variates to Haar SO(3) in the Rz-Ry-Rx convention.

    For ``R = Rz(gamma) Ry(beta) Rx(alpha)``, invariant measure is
    ``d alpha d gamma d(sin beta)``.  Drawing beta itself uniformly is not
    Haar and over-samples orientations whose first axis approaches +/-z.
    """
    dtype_jnp = jnp.float64
    alpha = 2 * jnp.pi * uniforms[0]
    beta = jnp.arcsin(2 * uniforms[1] - 1)
    gamma = 2 * jnp.pi * uniforms[2]
    cos_a, sin_a = jnp.cos(alpha), jnp.sin(alpha)
    cos_b, sin_b = jnp.cos(beta), jnp.sin(beta)
    cos_g, sin_g = jnp.cos(gamma), jnp.sin(gamma)
    R = jnp.array(
        [
            [cos_b * cos_g, cos_g * sin_a * sin_b - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_b],
            [cos_b * sin_g, cos_a * cos_g + sin_a * sin_b * sin_g, cos_a * sin_b * sin_g - cos_g * sin_a],
            [-sin_b, cos_b * sin_a, cos_a * cos_b],
        ],
        dtype=dtype_jnp,
    )
    return R


@jit
def _generate_rotation_matrix(jax_PRNG_key: jax.Array) -> jax.Array:
    """Sample a Haar-distributed random 3x3 rotation matrix and return R.T."""
    _, subkey = jax.random.split(jax_PRNG_key)
    uniforms = jax.random.uniform(subkey, shape=(3,), minval=0.0, maxval=1.0)
    R = _haar_rotation_matrix_from_uniform(uniforms)
    return R.T


@jit
def _geminal_inv_single(
    geminal_data: GeminalData,
    I: jax.Array,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> GeminalInverseState:
    """Build G and invert via SVD-based pseudoinverse (single sample).

    For a SVD ``G = U @ diag(s) @ Vh`` (where ``Vh`` is the matrix returned by
    ``jnp.linalg.svd`` and equals the conjugate transpose of ``V``), the
    pseudoinverse is ``G^+ = V @ diag(1/s) @ U^H = conj(Vh).T @ diag(1/s) @
    conj(U).T``. The ``conj`` on both ``Vh`` and ``U`` is required for
    complex Bloch geminals (non-Gamma twist); dropping it (``Vh.T``,
    ``U.T``) silently builds the WRONG matrix and seeds the Sherman-Morrison
    running inverse with a large initial error, which then contaminates
    every observable (e_L kinetic, ECP non-local) on the MCMC chain. The
    Gamma / real path is byte-identical to the unconjugated form because
    ``conj(x) == x`` for real ``x``. See ``determinant.py``'s
    ``_ln_det_bwd`` and ``compute_grads_and_laplacian_ln_Det`` for the
    matching convention.
    """
    G = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # Same pseudo-inverse arithmetic as the carried-inverse refresh.
    Ginv = _geminal_pinv_from_matrix(G)
    return G, Ginv, jnp.zeros_like(G), jnp.zeros(G.shape[0], dtype=jnp.int32)


@jit
def _geminal_inv_batched(
    geminal_data: GeminalData,
    r_up_batch: jax.Array,
    r_dn_batch: jax.Array,
) -> GeminalInverseState:
    """Batched geminal inverse over walkers.

    The geminals are built per walker; their seed inverses are computed for the whole batch
    (:func:`_geminal_pinv_batched`: the equilibrated LU inverse of every walker, with the
    pseudo-inverse of :func:`_geminal_inv_single` for a walker whose LU result fails the guard,
    issues #533 and #605).
    """
    G_b = vmap(compute_geminal_all_elements, in_axes=(None, 0, 0))(geminal_data, r_up_batch, r_dn_batch)
    Ginv_b = _geminal_pinv_batched(G_b)
    return G_b, Ginv_b, jnp.zeros_like(G_b), jnp.zeros(G_b.shape[:2], dtype=jnp.int32)


@jit
def _refine_running_geminal_inv_batched(geminal: jax.Array, geminal_inv: jax.Array) -> jax.Array:
    """Apply one Newton--Schulz residual correction to a walker batch.

    ``X + X @ (I - G @ X)`` squares the inverse residual while preserving the
    running Sherman--Morrison path and avoiding a new factorization.  The
    operation is dtype-generic, including complex Bloch geminals.
    """
    eye = jnp.eye(geminal_inv.shape[-1], dtype=geminal_inv.dtype)
    return geminal_inv + geminal_inv @ (eye - geminal @ geminal_inv)


def _geminal_pinv_from_matrix(G: jax.Array) -> jax.Array:
    """Return the row/column-equilibrated, thresholded SVD pseudo-inverse of one geminal.

    The single pseudo-inverse that seeds or re-anchors a carried ``G^-1``:
    the MCMC run seed (:func:`_geminal_inv_single`, and the native-route seed
    in ``MCMC.run``), the boundary refresh of MCMC and GFMC
    (:func:`_refresh_running_geminal_inv_masked`) and the GFMC_t / GFMC_n
    projection seeds.

    ``G`` is graded -- the row of an electron far from
    every nucleus (and the column of a far down electron) decays with the
    orbitals while the rest of ``G`` stays O(1). Such a ``G`` can be perfectly
    invertible and yet have ``s_min / s_max`` below ``rcond_svd`` (an up and a
    down electron about 10 bohr out in all-electron N: 3e-22), so a plain
    thresholded SVD drops a real direction and the Sherman--Morrison ratio of
    the far electron collapses (1.0 instead of 3.6); walkers seeded or
    refreshed there could never move back. Rows and then columns are therefore
    scaled by powers of two (exact in floating point) that bring their
    infinity norms into ``[0.5, 1)``; the pseudo-inverse of the equilibrated
    matrix ``A = R G C`` with the SAME ``rcond_svd`` cutoff is unscaled as
    ``C A^+ R``, which equals ``G^-1`` whenever ``G`` is invertible.

    Zero / non-finite guard: a row or column whose norm is zero (e.g. an
    exactly underflowed AO row) or non-finite keeps scale 1, so it is not
    blown up and the cutoff still zeroes its singular value. A truly singular
    ``G`` therefore stays singular: the result is finite but ``G X != I``,
    which the carried-inverse probe residual reports, and non-finite inputs propagate.
    Scale exponents are clipped to ``GEMINAL_EQUILIBRATION_MAX_EXPONENT``
    (issue #605), so the result is finite for every finite ``G`` (without the
    clip a far up and a far down electron overflow the unscaling to NaN).
    Conjugate factors keep complex Bloch geminals Hermitian-consistent.
    """
    return _equilibrated_geminal_pinv(G, get_eps("rcond_svd", jnp.float64))


def _geminal_pinv_batched(G: jax.Array) -> jax.Array:
    """Seed/refresh inverses of a walker batch of geminals ``(nw, N, N)`` (issues #533, #605).

    Every walker, on every backend, gets the LU inverse of its equilibrated geminal
    (:func:`jqmc.determinant._equilibrated_geminal_lu_inverse_and_residual`). It is kept when it is
    finite and its scale-free probe residual ``max|A (Y P) - P|`` is at most
    ``GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD``; any other walker gets the equilibrated SVD
    pseudo-inverse (:func:`_geminal_pinv_from_matrix`), i.e. exactly the SVD result.

    The residual of an invertible geminal stays at eps*cond, so the guard is far above the
    eps*cond level of any walker the sampler reaches; compared against the carried-inverse refresh
    threshold instead, an LU-first route would run the SVD on most refreshes, and the SVD's own
    residual would be the larger one. On a GH200 the batched SVD is about 1,000 times slower than the
    LU (N = 80 real, 256 walkers: 840 ms vs 0.9 ms; N = 64 complex: 719 ms vs 0.47 ms). The SVD
    batch therefore runs only when some walker needs it (``lax.cond`` at batch level, so it is not
    turned into a select by a walker ``vmap``).
    """
    lu_inverse, residual = vmap(_equilibrated_geminal_lu_inverse_and_residual)(G)
    threshold = float(_setting_module.GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD)
    accepted = jnp.all(jnp.isfinite(lu_inverse), axis=(-2, -1)) & (residual <= threshold)
    return lax.cond(
        jnp.all(accepted),
        lambda: lu_inverse,
        lambda: jnp.where(accepted[:, None, None], lu_inverse, vmap(_geminal_pinv_from_matrix)(G)),
    )


_geminal_pinv_from_matrix_batched = jit(_geminal_pinv_batched)


@jit
def _running_geminal_inv_probe_residual(geminal: jax.Array, geminal_inv: jax.Array) -> jax.Array:
    """Return the O(N^2) probe residual ``max|G (X P) - P|`` per walker.

    ``P`` holds two fixed, deterministic probe columns (all ones and
    ``cos(1 + 0.7 j)``), so the indicator costs four matrix-vector products per
    walker and never factorizes. ``G X - I`` is dimensionless, so the value is
    compared directly with ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``. Works for
    real and complex (Bloch) geminals; a non-finite inverse yields NaN, which
    never triggers a refresh (the existing non-finite handling is unchanged).
    """
    n = geminal_inv.shape[-1]
    index = jnp.arange(n, dtype=jnp.float64)
    probes = jnp.stack([jnp.ones(n, dtype=jnp.float64), jnp.cos(1.0 + 0.7 * index)], axis=1).astype(geminal_inv.dtype)
    residual = geminal @ (geminal_inv @ probes) - probes
    return jnp.max(jnp.abs(residual), axis=(-2, -1))


@jit
def _refresh_running_geminal_inv_masked(
    geminal: jax.Array,
    geminal_inv: jax.Array,
    triggered: jax.Array,
) -> jax.Array:
    """Replace the triggered walkers' inverses by the pseudo-inverse of ``geminal``.

    Untriggered walkers are selected bit-identical. Called only on steps where
    at least one walker triggered, so the O(N^3) batch never runs otherwise.
    """
    fresh = _geminal_pinv_batched(geminal).astype(geminal_inv.dtype)
    return jnp.where(triggered[:, None, None], fresh, geminal_inv)


def _refresh_running_geminal_inv(
    geminal: jax.Array,
    geminal_inv: jax.Array,
    threshold: float | None = None,
) -> tuple[jax.Array, int]:
    """Refresh drifted carried inverses from the same carried geminal.

    A running Sherman--Morrison inverse seeded at (or passing through) a
    near-singular geminal keeps that configuration's absolute error after the
    walker leaves the near-singular region; rank-1 updates never repair it, so
    every consumer of the carried inverse (fast local energy, Coulomb/ECP fast
    ratios, HF/Pulay forces, stresses, AS regularization, parameter
    derivatives) would disagree with an inverse recomputed from the same ``G``.
    This boundary check evaluates the O(N^2) probe residual per walker and
    replaces ``geminal_inv`` by the seed's SVD pseudo-inverse of the SAME
    ``geminal`` only for walkers whose residual exceeds ``threshold`` (default
    ``SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD``). Untriggered walkers are
    returned bit-identical, and the O(N^3) refresh runs only on steps where a
    walker triggers, so the per-move cost stays O(N^2).

    Args:
        geminal: Carried geminal matrices, shape ``(nw, N_up, N_up)``.
        geminal_inv: Carried inverses, same shape (fp64 or complex128).
        threshold: Probe-residual threshold; ``inf`` disables the refresh.

    Returns:
        ``(geminal_inv, n_refreshed)``.
    """
    if threshold is None:
        threshold = _setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    if not np.isfinite(threshold):
        return geminal_inv, 0
    triggered = np.asarray(_running_geminal_inv_probe_residual(geminal, geminal_inv)) > threshold
    if not triggered.any():
        return geminal_inv, 0
    refreshed = _refresh_running_geminal_inv_masked(jnp.asarray(geminal), jnp.asarray(geminal_inv), jnp.asarray(triggered))
    return refreshed, int(np.count_nonzero(triggered))


_jit_vmap_geminal_all_elements = jit(vmap(compute_geminal_all_elements, in_axes=(None, 0, 0)))


@jit
def _det_state_probe_residual(
    geminal_data: GeminalData, det_state: Det_ratio_streaming_state, geminal_inv: jax.Array
) -> jax.Array:
    """Probe residual of a running inverse against the geminal its streaming caches imply (one dispatch)."""
    geminal = vmap(_geminal_from_det_ratio_state, in_axes=(None, 0))(geminal_data, det_state)
    return _running_geminal_inv_probe_residual(geminal, geminal_inv)


def _refresh_observable_geminal_inv(
    geminal_data: GeminalData,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
    det_state: Det_ratio_streaming_state | None = None,
    threshold: float | None = None,
) -> tuple[jax.Array, int]:
    """Boundary refresh for drivers that carry the inverse but not ``G``.

    The GFMC projection intervals advance only the running inverse (and, when
    streaming, the determinant caches). The drift probe uses the geminal the
    running inverse actually inverts: the one implied by the streaming caches
    (no orbital evaluation), or the geminal rebuilt from the positions when no
    cache is available. Triggered walkers are replaced by the pseudo-inverse of
    the geminal rebuilt at these positions -- exactly the inverse the
    ``GFMC_T_OBS_FRESH_INV`` fresh route threads -- so the rebuild runs only on
    steps where a walker drifted.
    """
    if threshold is None:
        threshold = _setting_module.SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD
    if not np.isfinite(threshold):
        return geminal_inv, 0
    r_up_carts = jnp.asarray(r_up_carts)
    r_dn_carts = jnp.asarray(r_dn_carts)
    rebuilt = None
    if det_state is None:
        rebuilt = _jit_vmap_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
        residual = _running_geminal_inv_probe_residual(rebuilt, geminal_inv)
    else:
        residual = _det_state_probe_residual(geminal_data, det_state, geminal_inv)
    triggered = np.asarray(residual) > threshold
    if not triggered.any():
        return geminal_inv, 0
    if rebuilt is None:
        rebuilt = _jit_vmap_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    refreshed = _refresh_running_geminal_inv_masked(rebuilt, jnp.asarray(geminal_inv), jnp.asarray(triggered))
    return refreshed, int(np.count_nonzero(triggered))


def _as_guiding_ratio(R_AS_p: jax.Array, R_AS_o: jax.Array, epsilon_AS: float | jax.Array) -> jax.Array:
    """AS guiding-function factor ``(max(R_p, eps) / R_p) / (max(R_o, eps) / R_o)`` of one proposal (issues #131/#134).

    With ``epsilon_AS == 0`` the guiding function is psi itself and the factor is
    exactly 1, whatever the AS factors are (as the native sweep, which skips AS
    there). Otherwise the expression above is used for positive factors; a
    degenerate (exactly singular, ``R_AS == 0``) proposed state has guiding
    amplitude 0 and is rejected (factor 0, never ``eps / 0 = inf``), and a
    degenerate current state is left by any non-degenerate proposal (factor
    ``inf``). No ``0 * inf`` or ``0 / 0`` is formed.
    """
    R_AS_p_eps = jnp.maximum(R_AS_p, epsilon_AS)
    R_AS_o_eps = jnp.maximum(R_AS_o, epsilon_AS)
    safe_p = jnp.where(R_AS_p > 0.0, R_AS_p, 1.0)
    safe_o = jnp.where(R_AS_o > 0.0, R_AS_o, 1.0)
    ratio = (jnp.where(R_AS_p > 0.0, R_AS_p_eps, 1.0) / safe_p) / (jnp.where(R_AS_o > 0.0, R_AS_o_eps, 1.0) / safe_o)
    ratio = jnp.where(R_AS_p > 0.0, jnp.where(R_AS_o > 0.0, ratio, jnp.inf), 0.0)
    return jnp.where(epsilon_AS > 0.0, ratio, jnp.ones_like(ratio))


def _as_walker_weight(R_AS: jax.Array, epsilon_AS: float | jax.Array) -> jax.Array:
    """Walker reweighting factor ``(R_AS / max(R_AS, eps))**2`` (issue #131).

    Exactly 1 for ``epsilon_AS == 0`` (psi is its own guiding function; the
    bare expression would give ``0 / 0 = NaN`` for ``R_AS == 0``), and the
    expression above for ``epsilon_AS > 0``.
    """
    R_AS_eps = jnp.maximum(R_AS, epsilon_AS)
    weight = (R_AS / jnp.where(R_AS_eps > 0.0, R_AS_eps, 1.0)) ** 2
    return jnp.where(epsilon_AS > 0.0, weight, jnp.ones_like(weight))


@partial(jit, static_argnums=3)
def _update_electron_positions(
    init_r_up_carts: jax.Array,
    init_r_dn_carts: jax.Array,
    jax_PRNG_key: jax.Array,
    num_mcmc_per_measurement: int,
    hamiltonian_data: Hamiltonian_data,
    Dt: float,
    epsilon_AS: float,
    geminal_inv_init: jax.Array,
    geminal_init: jax.Array,
    det_ratio_state_init: Det_ratio_streaming_state,
    jastrow_state_init: Jastrow_ratio_state,
) -> MCMCSweepState:
    """Update electron positions based on the MH method.

    Args:
        init_r_up_carts (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
        init_r_dn_carts (jnpt.ArrayLike): down electron position. dim: (N_e^dn, 3)
        jax_PRNG_key (jnpt.ArrayLike): jax PRIN key.
        num_mcmc_per_measurement (int): the number of iterarations (i.e. the number of proposal in updating electron positions.)
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
        Dt (float): the step size in the MH method.
        epsilon_AS (float): the exponent of the AS regularization.
        geminal_inv_init: Sherman-Morrison running inverse for the geminal at
            ``(init_r_up_carts, init_r_dn_carts)``.
        geminal_init: Geminal matrix at ``(init_r_up_carts, init_r_dn_carts)``.
        det_ratio_state_init: ``Det_ratio_streaming_state`` consistent with
            ``(init_r_up_carts, init_r_dn_carts)``. Built once at the MCMC
            chain entry by :func:`_init_det_ratio_streaming_state` and
            advanced via :func:`_advance_det_ratio_streaming_state` on accept.
        jastrow_state_init: ``Jastrow_ratio_state`` container consistent
            with ``(init_r_up_carts, init_r_dn_carts)``. Today only
            ``.j3_state`` and/or ``.dtn_state`` are populated when their
            corresponding Jastrow terms are present, built at chain entry
            and advanced only on accept.
            Passing ``.j3_state`` to ``_compute_ratio_Jastrow_part_rank1_update``
            avoids the per-step ``O(n_ao^2 * N_e)`` matmul rebuild. The
            DTN similarly reuses electron-distance polynomial tables. The
            container absorbs these caches without changing the
            ``fori_loop`` carry signature.

    Returns:
        jax_PRNG_key (jnpt.ArrayLike): updated jax_PRNG_key.
        accepted_moves (int): the number of accepted moves
        rejected_moves (int): the number of rejected moves
        updated_r_up_cart (jnpt.ArrayLike): up electron position. dim: (N_e^up, 3)
        updated_r_dn_cart (jnpt.ArrayLike): down electron position. dim: (N_e^down, 3)
    """
    dtype_jnp = jnp.float64
    accepted_moves = 0
    rejected_moves = 0
    r_up_carts = jnp.asarray(init_r_up_carts, dtype=dtype_jnp)
    r_dn_carts = jnp.asarray(init_r_dn_carts, dtype=dtype_jnp)
    geminal = geminal_init
    geminal_inv = geminal_inv_init
    det_ratio_state = det_ratio_state_init
    jastrow_state = jastrow_state_init

    def body_fun(_: jax.Array, carry: MCMCSweepState) -> MCMCSweepState:
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ) = carry
        total_electrons = len(r_up_carts) + len(r_dn_carts)
        num_up_electrons = len(r_up_carts)

        # Choose randomly if the electron comes from up or dn
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        rand_num = jax.random.randint(subkey, shape=(), minval=0, maxval=total_electrons)

        # boolean: "up" or "dn"
        # is_up == True -> up,False -> dn
        is_up = rand_num < len(r_up_carts)

        # an index chosen from up electons
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_up_carts))

        # an index chosen from dn electrons
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        dn_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_dn_carts))

        selected_electron_index = jnp.where(is_up, up_index, dn_index)

        # choose an up or dn electron from old_r_cart
        old_r_cart = jnp.where(is_up, r_up_carts[selected_electron_index], r_dn_carts[selected_electron_index])

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

        # charges
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers)

        # PBC fix: use the minimum-image displacement so the proposal step
        # size ``sigma = f_l(|r - R_nearest_image|) * Dt`` collapses near
        # nuclei (Kato cusp protection) even when the nearest nucleus image
        # lies outside the home cell. Using ``coords[i]`` directly here
        # would compute ``|r - R_home_cell|`` which can be much larger than
        # the true min-image distance (e.g. electron near a corner of a
        # cubic cell with the nearest atom image one lattice translation
        # away), inflating sigma and crippling acceptance / autocorrelation
        # near image cusps. Molecular cells short-circuit min-image inside
        # the helper, so this is a no-op for ``pbc_flag=False``.
        Z = charges[nearest_atom_index]
        rel_R_min = _get_min_dist_rel_R_cart_jnp(
            structure_data=hamiltonian_data.structure_data,
            r_cart=old_r_cart,
            i_atom=nearest_atom_index,
            dtype=dtype_jnp,
        )
        norm_r_R = jnp.linalg.norm(rel_R_min)
        f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        sigma = f_l * Dt
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        g = jax.random.normal(subkey, shape=()) * sigma

        # choose x,y,or,z
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

        # plug g into g_vector
        g_vector = jnp.zeros(3, dtype=dtype_jnp)
        g_vector = g_vector.at[random_index].set(g)

        new_r_cart = old_r_cart + g_vector

        # set proposed r_up_carts and r_dn_carts.
        proposed_r_up_carts = lax.cond(
            is_up,
            lambda _: r_up_carts.at[selected_electron_index].set(new_r_cart),
            lambda _: r_up_carts,
            operand=None,
        )

        proposed_r_dn_carts = lax.cond(
            is_up,
            lambda _: r_dn_carts,
            lambda _: r_dn_carts.at[selected_electron_index].set(new_r_cart),
            operand=None,
        )

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

        # PBC fix: same rationale as the forward proposal -- use the
        # minimum-image displacement so f_prime_l matches the actual
        # cusp-protected step size at the new position.
        Z = charges[nearest_atom_index]
        rel_R_min_new = _get_min_dist_rel_R_cart_jnp(
            structure_data=hamiltonian_data.structure_data,
            r_cart=new_r_cart,
            i_atom=nearest_atom_index,
            dtype=dtype_jnp,
        )
        norm_r_R = jnp.linalg.norm(rel_R_min_new)
        f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        T_ratio = (f_l / f_prime_l) * jnp.exp(
            -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
            * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
        )

        # Jastrow ratio via dedicated fast-update API (includes exp).
        # ``jastrow_state.j3_state`` (when not None) lets the J3 block reuse
        # cached ``aos_*`` and ``j3_mat @ aos_*`` / ``j3_mat.T @ aos_*``
        # precontracts instead of recomputing the O(n_ao^2 * N_e) matmuls
        # each step.
        J_ratio = _compute_ratio_Jastrow_part_rank1_update(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            old_r_up_carts=r_up_carts,
            old_r_dn_carts=r_dn_carts,
            new_r_up_carts_arr=jnp.expand_dims(proposed_r_up_carts, axis=0),
            new_r_dn_carts_arr=jnp.expand_dims(proposed_r_dn_carts, axis=0),
            j3_state=jastrow_state.j3_state,
            dtn_state=jastrow_state.dtn_state,
        )[0]

        # Determinant part, fast update using the matrix determinant lemma.
        # The v / u construction reuses the cached AO + paired matrices in
        # ``det_ratio_state`` so the bulk-side AO eval and ``lambda @ ao_dn``
        # GEMM are skipped.
        # Consumer-side explicit cast: cast both lax.cond branches to the
        # local float64 dtype so the branches agree under mixed precision
        # (one_hot defaults to fp32, slim helpers return det_ratio zone).
        # The rank-one update vectors
        # ``v`` / ``u`` go complex at a non-Gamma Bloch geminal because
        # ``_compute_v_up`` / ``_compute_u_dn`` carry the complex paired-
        # dn / paired-up-lambda contraction. Promote ``dtype_vu`` against
        # the geminal lambda dtype AND the streaming-state paired buffer
        # dtype so the lax.cond branches agree on the widest dtype
        # (complex128 at non-Gamma, float64 at Gamma) even when lambda is
        # real (RHF singlet) but the AO leaves are complex.
        dtype_vu = jnp.result_type(
            dtype_jnp,
            hamiltonian_data.wavefunction_data.geminal_data._determinant_dtype_jnp,
            det_ratio_state.paired_dn.dtype,
            det_ratio_state.paired_up_lambda.dtype,
        )
        v = lax.cond(
            is_up,
            lambda _: jnp.asarray(
                _compute_v_up_move_from_det_ratio_state(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    state=det_ratio_state,
                    moved_index=selected_electron_index,
                    r_up_carts_proposed=proposed_r_up_carts,
                )[:, None],
                dtype=dtype_vu,
            ),
            lambda _: jnp.asarray(
                jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None],
                dtype=dtype_vu,
            ),
            operand=None,
        )

        u = lax.cond(
            is_up,
            lambda _: jnp.asarray(
                jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None],  # (N_up, 1)
                dtype=dtype_vu,
            ),
            lambda _: jnp.asarray(
                _compute_u_dn_move_from_det_ratio_state(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    state=det_ratio_state,
                    moved_index=selected_electron_index,
                    r_dn_carts_proposed=proposed_r_dn_carts,
                )[:, None],
                dtype=dtype_vu,
            ),
            operand=None,
        )

        # Determinant ratio and rank-1 inverse update:
        # det(A+uv^T)/det(A) = 1 + v^T A^{-1} u
        # Issue #127: the moved electron selects the exact column / row of the
        # new inverse on a large ratio (bit-identical otherwise).
        Det_T_ratio, geminal_inv_new = _sherman_morrison_ratio_and_inverse(
            geminal_inv, u, v, moved_index=selected_electron_index, moved_is_up=is_up
        )

        geminal_new = lax.cond(
            is_up,
            # Row update: row[i, :] += Deltarow_i  (v is (N_cols,1) -> squeeze last dim)
            lambda _: geminal.at[selected_electron_index, :].add(v.squeeze(-1)),
            # Column update: col[:, j] += Deltacol_j (u is (N_up,1) -> squeeze last dim)
            lambda _: geminal.at[:, selected_electron_index].add(u.squeeze(-1)),
            operand=None,
        )
        # Issue #134: the AS factor of the proposed state uses the moved row /
        # column rebuilt from the new orbital values (G + delta cancels to zero
        # for a tiny ratio) and the moved inverse column / row in its exact
        # Sherman-Morrison form. The carried geminal_new / geminal_inv_new are
        # unchanged.
        geminal_new_AS, geminal_inv_new_AS = lax.cond(
            is_up,
            lambda _: (
                _complex_update.at_set(
                    geminal_new,
                    jnp.s_[selected_electron_index, :],
                    jnp.asarray(
                        _compute_up_row_from_det_ratio_state(
                            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                            state=det_ratio_state,
                            moved_index=selected_electron_index,
                            r_up_carts_proposed=proposed_r_up_carts,
                        ),
                        dtype=geminal_new.dtype,
                    ),
                ),
                _complex_update.at_set(
                    geminal_inv_new,
                    jnp.s_[:, selected_electron_index],
                    jnp.asarray(geminal_inv[:, selected_electron_index] / Det_T_ratio, dtype=geminal_inv_new.dtype),
                ),
            ),
            lambda _: (
                _complex_update.at_set(
                    geminal_new,
                    jnp.s_[:, selected_electron_index],
                    jnp.asarray(
                        _compute_dn_column_from_det_ratio_state(
                            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                            state=det_ratio_state,
                            moved_index=selected_electron_index,
                            r_dn_carts_proposed=proposed_r_dn_carts,
                        ),
                        dtype=geminal_new.dtype,
                    ),
                ),
                _complex_update.at_set(
                    geminal_inv_new,
                    jnp.s_[selected_electron_index, :],
                    jnp.asarray(geminal_inv[selected_electron_index, :] / Det_T_ratio, dtype=geminal_inv_new.dtype),
                ),
            ),
            operand=None,
        )
        # compute AS regularization factors (issue #131: robust to tiny rows)
        R_AS_p = compute_AS_regularization_factor_fast_update(geminal_new_AS, geminal_inv_new_AS)
        R_AS_o = compute_AS_regularization_factor_fast_update(geminal, geminal_inv)

        # modified trial WFs
        R_AS_ratio = _as_guiding_ratio(R_AS_p, R_AS_o, epsilon_AS)
        WF_ratio = J_ratio * Det_T_ratio

        # compute R_ratio (|.|^2 via (r * conj(r)).real; bit-identical for real r,
        # keeps the Metropolis ratio real-positive once Det_T_ratio is complex)
        _ratio = R_AS_ratio * WF_ratio
        R_ratio = (_ratio * jnp.conj(_ratio)).real

        acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio], dtype=dtype_jnp))

        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

        # On accept, advance the slim ratio state to the new configuration.
        # The advance refreshes one column of state.ao_{up,dn} and the
        # corresponding row/column of state.paired_{up_lambda,dn} from a
        # fresh single-electron AO eval (no rank-1 drift).
        det_ratio_state_new = _advance_det_ratio_streaming_state(
            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
            state=det_ratio_state,
            moved_spin_is_up=is_up,
            moved_index=selected_electron_index,
            r_up_carts_new=proposed_r_up_carts,
            r_dn_carts_new=proposed_r_dn_carts,
        )

        # On accept, advance whichever ratio sub-states this Jastrow's channels
        # own. Which channels those are, and what each one's advance kernel
        # wants, is decided inside the Jastrow layer; the dispatch there is
        # Python-static so the container shape is fixed for the loop carry.
        jastrow_state_new = advance_Jastrow_ratio_state(
            hamiltonian_data.wavefunction_data.jastrow_data,
            jastrow_state,
            moved_spin_is_up=is_up,
            moved_index=selected_electron_index,
            r_up_carts_new=proposed_r_up_carts,
            r_dn_carts_new=proposed_r_dn_carts,
        )

        def _accepted_fun(_: None) -> MCMCAcceptRejectState:
            # Move accepted. Walker positions are kept UNWRAPPED; ``compute_AOs``
            # in ``atomic_orbital.py`` wraps internally (line ``r_carts =
            # _wrap_into_cell_if_pbc(self.structure_data, r_carts)``) before
            # the per-image expansion, and the ``_restore_phase`` factor on
            # the complex Bloch leaves picks up the ``e^{i k . T}`` phase
            # from the unwrapped position. Wrapping here, AFTER the
            # SM update + det_ratio_state advance, would leave the running
            # ``geminal_inv`` and the streaming AO cache calibrated for the
            # UNWRAPPED proposed position while the carry walker r is
            # WRAPPED. At Gamma the wrap_restore phase is 1 so the mismatch
            # is benign; on the complex-Bloch path it seeds a phase error
            # at every wrap event that propagates through subsequent SM
            # updates (Si primitive cc-pVDZ at k=(0.25,0.25,0.25): VMC E
            # -10.05 with that wrap, -7.55 = pyscf HF without it).
            return (
                accepted_moves + 1,
                rejected_moves,
                proposed_r_up_carts,
                proposed_r_dn_carts,
                geminal_inv_new,
                geminal_new,
                det_ratio_state_new,
                jastrow_state_new,
            )

        def _rejected_fun(_: None) -> MCMCAcceptRejectState:
            # Move rejected
            return (
                accepted_moves,
                rejected_moves + 1,
                r_up_carts,
                r_dn_carts,
                geminal_inv,
                geminal,
                det_ratio_state,
                jastrow_state,
            )

        # judge accept or reject the propsed move using jax.lax.cond
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ) = lax.cond(b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None)

        carry = (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        )
        return carry

    # main MCMC loop
    (
        accepted_moves,
        rejected_moves,
        r_up_carts,
        r_dn_carts,
        jax_PRNG_key,
        geminal_inv,
        geminal,
        det_ratio_state,
        jastrow_state,
    ) = jax.lax.fori_loop(
        0,
        num_mcmc_per_measurement,
        body_fun,
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ),
    )

    return (
        accepted_moves,
        rejected_moves,
        r_up_carts,
        r_dn_carts,
        jax_PRNG_key,
        geminal_inv,
        geminal,
        det_ratio_state,
        jastrow_state,
    )


@partial(jit, static_argnums=3)
def _update_electron_positions_only_up_electron(
    init_r_up_carts: jax.Array,
    init_r_dn_carts: jax.Array,
    jax_PRNG_key: jax.Array,
    num_mcmc_per_measurement: int,
    hamiltonian_data: Hamiltonian_data,
    Dt: float,
    epsilon_AS: float,
    geminal_inv_init: jax.Array,
    geminal_init: jax.Array,
    det_ratio_state_init: Det_ratio_streaming_state,
    jastrow_state_init: Jastrow_ratio_state,
) -> MCMCSweepState:
    """Update electron positions based on the MH method (up-spin electrons only).

    See :func:`_update_electron_positions` for the slim state ``det_ratio_state_init``
    contract; here only up-electrons move so ``state.ao_dn`` and ``state.paired_dn``
    stay constant for the entire chain.

    ``jastrow_state_init`` mirrors the signature of :func:`_update_electron_positions`
    so the vmap wrappers stay symmetric, but this variant evaluates the Jastrow
    factor from two whole values rather than the rank-1 ratio API, so the
    container is only carried through (never consumed or advanced). Both whole
    values come from :func:`compute_Jastrow_part_ratio_zone`, i.e. the
    ``jastrow_ratio`` zone that owns the accept ratio, not ``jastrow_eval``
    (float32 in mixed), whose rounding of ``|J|`` the difference would keep
    (issue #781). A future refactor to the ratio path would also
    light up the same Jastrow streaming-cache savings here.
    """
    dtype_jnp = jnp.float64
    accepted_moves = 0
    rejected_moves = 0
    r_up_carts = jnp.asarray(init_r_up_carts, dtype=dtype_jnp)
    r_dn_carts = jnp.asarray(init_r_dn_carts, dtype=dtype_jnp)
    geminal_inv = geminal_inv_init
    geminal = geminal_init
    det_ratio_state = det_ratio_state_init
    jastrow_state = jastrow_state_init

    def body_fun(_: jax.Array, carry: MCMCSweepState) -> MCMCSweepState:
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ) = carry
        num_up_electrons = len(r_up_carts)

        # dummy jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)

        # Choose randomly if the electron comes from up or dn
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_up_carts))
        selected_electron_index = up_index

        # dummy jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)

        # choose an up or dn electron from old_r_cart
        old_r_cart = r_up_carts[selected_electron_index]

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

        # charges
        if hamiltonian_data.coulomb_potential_data.ecp_flag:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                hamiltonian_data.coulomb_potential_data.z_cores
            )
        else:
            charges = jnp.array(hamiltonian_data.structure_data.atomic_numbers)

        # PBC fix (see the rank-1 sibling for full rationale): use the
        # minimum-image displacement so the cusp-protection step size is
        # correct even when the nearest nucleus image lies outside the
        # home cell. Molecular cells short-circuit min-image inside the
        # helper, so this is a no-op for ``pbc_flag=False``.
        Z = charges[nearest_atom_index]
        rel_R_min = _get_min_dist_rel_R_cart_jnp(
            structure_data=hamiltonian_data.structure_data,
            r_cart=old_r_cart,
            i_atom=nearest_atom_index,
            dtype=dtype_jnp,
        )
        norm_r_R = jnp.linalg.norm(rel_R_min)
        f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        sigma = f_l * Dt
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        g = jax.random.normal(subkey, shape=()) * sigma

        # choose x,y,or,z
        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

        # plug g into g_vector
        g_vector = jnp.zeros(3, dtype=dtype_jnp)
        g_vector = g_vector.at[random_index].set(g)

        new_r_cart = old_r_cart + g_vector

        # set proposed r_up_carts and r_dn_carts.
        proposed_r_up_carts = r_up_carts.at[selected_electron_index].set(new_r_cart)
        proposed_r_dn_carts = r_dn_carts

        # choose the nearest atom index
        nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

        # PBC fix: same rationale as the forward proposal.
        Z = charges[nearest_atom_index]
        rel_R_min_new = _get_min_dist_rel_R_cart_jnp(
            structure_data=hamiltonian_data.structure_data,
            r_cart=new_r_cart,
            i_atom=nearest_atom_index,
            dtype=dtype_jnp,
        )
        norm_r_R = jnp.linalg.norm(rel_R_min_new)
        f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

        T_ratio = (f_l / f_prime_l) * jnp.exp(
            -(jnp.linalg.norm(new_r_cart - old_r_cart) ** 2)
            * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
        )

        # original trial WFs
        Jastrow_T_p = compute_Jastrow_part_ratio_zone(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            r_up_carts=proposed_r_up_carts,
            r_dn_carts=proposed_r_dn_carts,
        )

        Jastrow_T_o = compute_Jastrow_part_ratio_zone(
            jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        Jastrow_T_p = jnp.asarray(Jastrow_T_p, dtype=dtype_jnp)
        Jastrow_T_o = jnp.asarray(Jastrow_T_o, dtype=dtype_jnp)

        # Determinant part, fast update using the matrix determinant lemma.
        # v is built from the cached AO + paired_dn -- skips the
        # lambda @ ao_dn GEMM and the bulk-side AO eval that calling
        # ``compute_geminal_up_one_row_elements`` would pay.
        v = _compute_v_up_move_from_det_ratio_state(
            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
            state=det_ratio_state,
            moved_index=selected_electron_index,
            r_up_carts_proposed=proposed_r_up_carts,
        )[:, None]

        u = jax.nn.one_hot(selected_electron_index, num_up_electrons)[:, None]

        # Determinant ratio and rank-1 inverse update:
        # det(A+uv^T)/det(A) = 1 + v^T A^{-1} u
        # The updated inverse keeps ``geminal_inv.dtype`` (the helper's default),
        # as in the two-spin kernel: a complex Bloch-twist inverse must stay
        # complex, and both lax.cond branches below must return the same dtype.
        Det_T_ratio, geminal_inv_new = _sherman_morrison_ratio_and_inverse(
            geminal_inv, u, v, moved_index=selected_electron_index, moved_is_up=True
        )

        geminal_new = geminal.at[selected_electron_index, :].add(v.squeeze(-1))

        # Issue #134: AS factor of the proposed state from the directly rebuilt
        # moved row and the exact moved inverse column (see the two-spin kernel).
        geminal_new_AS = _complex_update.at_set(
            geminal_new,
            jnp.s_[selected_electron_index, :],
            jnp.asarray(
                _compute_up_row_from_det_ratio_state(
                    geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                    state=det_ratio_state,
                    moved_index=selected_electron_index,
                    r_up_carts_proposed=proposed_r_up_carts,
                ),
                dtype=geminal_new.dtype,
            ),
        )
        geminal_inv_new_AS = _complex_update.at_set(
            geminal_inv_new,
            jnp.s_[:, selected_electron_index],
            jnp.asarray(geminal_inv[:, selected_electron_index] / Det_T_ratio, dtype=geminal_inv_new.dtype),
        )

        # compute AS regularization factors (issue #131: robust to tiny rows)
        R_AS_p = compute_AS_regularization_factor_fast_update(geminal_new_AS, geminal_inv_new_AS)
        R_AS_o = compute_AS_regularization_factor_fast_update(geminal, geminal_inv)

        # modified trial WFs
        R_AS_ratio = _as_guiding_ratio(R_AS_p, R_AS_o, epsilon_AS)
        # Promote to the local float64 dtype without dropping a complex phase: the
        # acceptance below uses |ratio|^2, not Re(ratio)^2.
        Det_T_ratio = jnp.asarray(Det_T_ratio, dtype=jnp.result_type(dtype_jnp, Det_T_ratio.dtype))
        WF_ratio = jnp.exp(Jastrow_T_p - Jastrow_T_o) * (Det_T_ratio)

        # compute R_ratio (|.|^2 via (r * conj(r)).real; bit-identical for real r,
        # keeps the Metropolis ratio real-positive once Det_T_ratio is complex)
        _ratio = R_AS_ratio * WF_ratio
        R_ratio = (_ratio * jnp.conj(_ratio)).real

        acceptance_ratio = jnp.min(jnp.array([1.0, R_ratio * T_ratio], dtype=dtype_jnp))

        jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
        b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

        # On accept, advance the slim ratio state (always up-move in this
        # variant -- state.ao_dn / state.paired_dn stay constant).
        det_ratio_state_new = _advance_det_ratio_streaming_state(
            geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
            state=det_ratio_state,
            moved_spin_is_up=jnp.bool_(True),
            moved_index=selected_electron_index,
            r_up_carts_new=proposed_r_up_carts,
            r_dn_carts_new=proposed_r_dn_carts,
        )

        def _accepted_fun(_: None) -> MCMCAcceptRejectState:
            # Move accepted. ``jastrow_state`` is carried through unchanged:
            # this variant uses the original full-Jastrow path
            # (``compute_Jastrow_part_ratio_zone``) rather than the rank-1 ratio API,
            # so the Jastrow ratio-streaming cache is not consumed or
            # advanced here. Walker positions are kept UNWRAPPED here; see
            # the matching note in ``_update_electron_positions`` for why
            # MCMC-side wrapping is not done (running ``geminal_inv`` /
            # streaming-cache calibration mismatch with the wrapped carry
            # on the complex Bloch path).
            return (
                accepted_moves + 1,
                rejected_moves,
                proposed_r_up_carts,
                proposed_r_dn_carts,
                geminal_inv_new,
                geminal_new,
                det_ratio_state_new,
                jastrow_state,
            )

        def _rejected_fun(_: None) -> MCMCAcceptRejectState:
            # Move rejected
            return (
                accepted_moves,
                rejected_moves + 1,
                r_up_carts,
                r_dn_carts,
                geminal_inv,
                geminal,
                det_ratio_state,
                jastrow_state,
            )

        # judge accept or reject the propsed move using jax.lax.cond
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ) = lax.cond(b < acceptance_ratio, _accepted_fun, _rejected_fun, operand=None)

        carry = (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        )
        return carry

    # main MCMC loop
    (
        accepted_moves,
        rejected_moves,
        r_up_carts,
        r_dn_carts,
        jax_PRNG_key,
        geminal_inv,
        geminal,
        det_ratio_state,
        jastrow_state,
    ) = jax.lax.fori_loop(
        0,
        num_mcmc_per_measurement,
        body_fun,
        (
            accepted_moves,
            rejected_moves,
            r_up_carts,
            r_dn_carts,
            jax_PRNG_key,
            geminal_inv,
            geminal,
            det_ratio_state,
            jastrow_state,
        ),
    )

    return (
        accepted_moves,
        rejected_moves,
        r_up_carts,
        r_dn_carts,
        jax_PRNG_key,
        geminal_inv,
        geminal,
        det_ratio_state,
        jastrow_state,
    )


# Module-level batched-map/jit wrappers for MCMC kernels.
# Created once at import time so subsequent MCMC.run() calls reuse
# the same Python function objects and hit JAX's compilation cache.
@partial(jit, static_argnums=3)
def _jit_vmap_update(
    r_up: jax.Array,
    r_dn: jax.Array,
    keys: jax.Array,
    n_steps: int,
    ham: Hamiltonian_data,
    Dt: float,
    eps: float,
    ginv: jax.Array,
    geminal: jax.Array,
    det_state: Det_ratio_streaming_state,
    jas_state: Jastrow_ratio_state,
) -> MCMCSweepState:
    xs = (r_up, r_dn, keys, ginv, geminal, det_state, jas_state)

    def one_walker(
        x: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            Det_ratio_streaming_state,
            Jastrow_ratio_state,
        ],
    ) -> MCMCSweepState:
        ru, rd, key, gi, g, ds, js = x
        return _update_electron_positions(ru, rd, key, n_steps, ham, Dt, eps, gi, g, ds, js)

    return walker_map(lambda *x: one_walker(x), *xs)


@partial(jit, static_argnums=3)
def _jit_vmap_update_up(
    r_up: jax.Array,
    r_dn: jax.Array,
    keys: jax.Array,
    n_steps: int,
    ham: Hamiltonian_data,
    Dt: float,
    eps: float,
    ginv: jax.Array,
    geminal: jax.Array,
    det_state: Det_ratio_streaming_state,
    jas_state: Jastrow_ratio_state,
) -> MCMCSweepState:
    xs = (r_up, r_dn, keys, ginv, geminal, det_state, jas_state)

    def one_walker(
        x: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            Det_ratio_streaming_state,
            Jastrow_ratio_state,
        ],
    ) -> MCMCSweepState:
        ru, rd, key, gi, g, ds, js = x
        return _update_electron_positions_only_up_electron(ru, rd, key, n_steps, ham, Dt, eps, gi, g, ds, js)

    return walker_map(lambda *x: one_walker(x), *xs)


_jit_vmap_init_det_ratio_state = jit(
    vmap(_init_det_ratio_streaming_state, in_axes=(None, 0, 0)),
)


def _draw_injected_normals(jax_PRNG_key: jax.Array, num_mcmc_per_measurement: int) -> jax.Array:
    """Bit-exact Gaussian-proposal normals for the ``jqmc_rs`` native sweep.

    Replays the EXACT ``jax.random.split`` order of the JAX MCMC move
    (:func:`_update_electron_positions` / its up-only sibling): six splits per
    proposal, the 4th subkey being the Gaussian-proposal draw. Only that
    ``jax.random.normal`` value is injected into ``jqmc_rs._mcmc_sweep``; the
    other five threefry draws (electron up/dn choice, up/dn indices, Cartesian
    axis, Metropolis ``b``) are reproduced bit-for-bit inside Rust from the same
    threefry key stream. Injecting the erfinv-based normal keeps the whole
    accept/reject sequence byte-identical to the JAX path. Because the erfinv
    ULP match is not guaranteed, the normal is supplied rather than re-derived
    in Rust). The key advance is outcome-independent (every split fires before
    the accept/reject branch), so all proposals' normals are pre-drawn here.
    """

    return _draw_injected_normals_with_key(jax_PRNG_key, num_mcmc_per_measurement)[1]


def _draw_injected_normals_with_key(
    jax_PRNG_key: jax.Array,
    num_mcmc_per_measurement: int,
) -> tuple[jax.Array, jax.Array]:
    """Like :func:`_draw_injected_normals` but ALSO returns the ADVANCED threefry
    key (``(key_after, normals)``). The 6-split-per-move advance is byte-identical
    to the native sweep's key threading, so ``key_after`` == the sweep's
    ``final_key`` for the same starting key -- this is what lets the ENERGY
    whole-run chunk pre-draw a whole chunk's normals + rotation matrices
    in a single JAX scan whose per-step key chain stays aligned with the Rust
    sweep loop."""

    def body_fun(i: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        key, normals = carry
        key, _sk = jax.random.split(key)  # electron up/dn choice
        key, _sk = jax.random.split(key)  # up index
        key, _sk = jax.random.split(key)  # dn index
        key, subkey = jax.random.split(key)  # Gaussian proposal (injected)
        key, _sk = jax.random.split(key)  # Cartesian axis
        key, _sk = jax.random.split(key)  # Metropolis b
        g = jax.random.normal(subkey, shape=())
        return key, normals.at[i].set(g)

    normals0 = jnp.zeros((num_mcmc_per_measurement,), dtype=jnp.float64)
    key_after, normals = jax.lax.fori_loop(0, num_mcmc_per_measurement, body_fun, (jax_PRNG_key, normals0))
    return key_after, normals


# vmap over walkers (each carries its own threefry key); num moves is static.
_jit_vmap_draw_injected_normals = jit(
    vmap(_draw_injected_normals, in_axes=(0, None)),
    static_argnums=1,
)
_jit_vmap_init_jastrow_ratio_state = jit(
    vmap(init_Jastrow_ratio_state, in_axes=(None, 0, 0)),
)
_jit_vmap_e_L_fast = jit(vmap(compute_local_energy_fast, in_axes=(None, 0, 0, 0, 0, 0)))
_jit_vmap_generate_RTs = jit(vmap(_generate_rotation_matrix, in_axes=0))
_jit_vmap_as_reg_fast = jit(vmap(compute_AS_regularization_factor_fast_update, in_axes=(0, 0)))
_jit_vmap_swct_omega = jit(vmap(evaluate_swct_omega, in_axes=(None, 0)))
# Fused twin: contracts the SWCT weights with a per-electron field WITHOUT
# materialising the (N_a, N_e) weight matrix per walker (see
# ``swct.evaluate_swct_omega_dot``). ``_jit_vmap_swct_omega`` is kept for the
# warm-up compiles and as the reference the equivalence test compares against.
_jit_vmap_swct_omega_dot = jit(evaluate_swct_omega_dot_batched)
_jit_vmap_swct_domega = jit(vmap(evaluate_swct_domega, in_axes=(None, 0)))


# ---------------------------------------------------------------------------
# The c-axis (variational
# parameter) gradients O_k = d ln|Psi| / dc and de_L/dc route through the
# unified ``jqmc.derivatives`` interface with ``analytic=False`` (the autograd
# twin, numerically the masked autograd). ``stop_gradient`` /
# ``with_param_grad_mask`` are encapsulated inside ``derivatives`` (the caller
# passes only the resolved ``opt_flags``); the driver keeps the per-walker
# jit(vmap) wrapper and the block-name reducer below.
#
# The interface returns the RAW ``Wavefunction_data``-shaped grad for the
# ``"c"`` axis; ``_reduce_c_param_grads`` collapses the batched grad to the flat
# block-name dict production stores (bit-identical to the active-parameters
# closure, including the SD_data/AGPn_data direct-MO blocks + closed-shell tie).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Active-positions extract / inject for the position-grad path.
#
# All :class:`Structure_data` instances
# reachable from a Hamiltonian carry an identically shaped ``(n_atom, 3)``
# ``positions`` array (``_apply_image_expansion`` does not touch
# ``structure_data.positions`` -- only the AO arrays). So the natural
# "active" coordinate for the position-grad path is a single shared
# ``(n_atom, 3)`` array injected into every reachable ``structure_data
# .positions`` leaf; differentiating w.r.t. it returns the total
# ``dE / dR_atom`` summed across all paths via JAX's chain rule. Unlike an
# ``accumulate_position_grad`` walk over a wf-pytree-shaped output, this
# -- crucially -- shrinks the vmap-broadcast output
# pytree from "every leaf of the Hamiltonian" (incl. spline_coefficients
# on B-spline) to a single ``(num_walkers, n_atom, 3)`` array.
# ---------------------------------------------------------------------------


def _extract_active_positions_from_ham(ham: Hamiltonian_data) -> dict[str, jax.Array]:
    """Return ``{"positions": (n_atom, 3) jax.Array}`` for the
    position-grad active dict. One shared sentinel is enough because
    every reachable ``structure_data.positions`` has the same shape."""
    return {"positions": jnp.asarray(ham.structure_data.positions)}


def _inject_active_positions(obj: PytreeOwner, active: Mapping[str, jax.Array]) -> PytreeOwner:
    """Replace every ``structure_data.positions`` leaf reachable from
    ``obj`` with ``active["positions"]``. Path-based on purpose
    (matches the production leak-detection test's path check). Other
    Structure_data leaves (``vec_a/b/c``, ``recip_cell``, etc.) are
    left untouched; force/Pulay paths don't read cell-cotangents.

    Placeholder leaves whose shape does not match the sentinel (e.g.
    ``Jastrow_two_body_data.structure_data.positions`` shaped ``(0,)``)
    are left untouched so the forward eval doesn't see a shape
    mismatch."""
    if not active or "positions" not in active:
        return obj
    new_pos = active["positions"]
    target_shape = new_pos.shape
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(obj)
    new_leaves = []
    for path, leaf in leaves_with_path:
        path_str = jax.tree_util.keystr(path)
        if (
            path_str.endswith(".positions")
            and ".structure_data." in path_str
            and leaf is not None
            and getattr(leaf, "shape", None) == target_shape
        ):
            new_leaves.append(new_pos)
        else:
            new_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ---------------------------------------------------------------------------
# Module-level LRU caches for the in-trace masked grad closures.
#
# ``MCMC.run_optimize`` re-assigns ``self.hamiltonian_data`` once per step,
# which triggers ``__init_attributes``. Building fresh ``jit(...)`` closures
# there gives JAX a brand-new function object on every step -> trace + HLO
# cache misses -> ~60 second recompile every step on B-spline / PW geminals.
# Caching on ``mask_kwargs_items`` returns the same
# function object across calls so the JAX cache hits and the recompile goes
# away. Keys must be fully hashable: ``mask_kwargs`` is converted to a
# sorted tuple of ``(name, bool)`` pairs.
# ---------------------------------------------------------------------------


# The strained-forward kernels (``_local_energy_strained`` and
# ``_local_log_psi_strained``) and their jit(vmap(jacrev(...))) stress helpers
# (``_jit_vmap_local_stress_HF`` / ``_jit_vmap_local_stress_PP``) live in the
# derivatives layer (``jqmc.derivatives._auto``) -- they are wavefunction /
# Hamiltonian-level forward helpers, so the eps/stress jacrev consumers own them.


def _debug_collect_param_grads(
    wavefunction_data: Wavefunction_data,
    grad_wavefunction: Wavefunction_data,
    param_flags: dict[str, bool],
) -> dict[str, object]:
    """Collect walker-stacked parameter gradients into block-name arrays for ``_MCMC_debug``.

    ``Wavefunction_data.collect_param_grads`` is used unchanged except for
    ``AGPn_data``: its inherited geminal collector reads the dense
    ``lambda_matrix`` compatibility property, which materialises
    ``diag(lambda_diag)`` with host numpy and cannot accept a gradient carrying
    a leading walker axis.  AGPn publishes the direct natural-orbital blocks
    instead (``Wavefunction_data.get_variational_blocks``): ``mo_coefficients_up``
    (the ``time_reversal`` tie folds the embedded down-spin gradient as
    ``grad_up + conj(grad_dn)``; without the tie ``mo_coefficients_dn`` is its
    own block), ``lambda_diag``, ``lambda_unpaired`` for open shell, and the
    GTO ``lambda_basis_*`` blocks of the base collector.  Real-field blocks are
    stored as their real part, the storage convention of the production
    diagnostic arrays.
    """
    geminal_data = wavefunction_data.geminal_data
    if not isinstance(geminal_data, AGPn_data):
        return wavefunction_data.collect_param_grads(grad_wavefunction)

    from .derivatives.bridge import _project_real_param_grads
    from .determinant import _embed_time_reversal_down_rows, _gto_leaves_of

    grads: dict[str, object] = {}
    if wavefunction_data.jastrow_data is not None and grad_wavefunction.jastrow_data is not None:
        grads.update(wavefunction_data.jastrow_data.collect_param_grads(grad_wavefunction.jastrow_data))
    grad_geminal = grad_wavefunction.geminal_data
    if _gto_leaves_of(geminal_data.orb_data_up_spin) and _gto_leaves_of(geminal_data.orb_data_dn_spin):
        grads["lambda_basis_exp"] = jnp.concatenate([grad_geminal.ao_exponents_up, grad_geminal.ao_exponents_dn], axis=-1)
        grads["lambda_basis_coeff"] = jnp.concatenate(
            [grad_geminal.ao_coefficients_up, grad_geminal.ao_coefficients_dn], axis=-1
        )
    grad_up = grad_geminal.orb_data_up_spin.mo_coefficients
    grad_dn = grad_geminal.orb_data_dn_spin.mo_coefficients
    if geminal_data.time_reversal:
        grads["mo_coefficients_up"] = grad_up + jnp.conj(_embed_time_reversal_down_rows(grad_up, grad_dn, xp=jnp))
    else:
        grads["mo_coefficients_up"] = grad_up
        grads["mo_coefficients_dn"] = grad_dn
    grads["lambda_diag"] = grad_geminal.lambda_diag
    if np.asarray(geminal_data.lambda_unpaired).size > 0:
        grads["lambda_unpaired"] = grad_geminal.lambda_unpaired
    return dict(_project_real_param_grads(wavefunction_data, grads, param_flags))


class _MCMC_debug:
    """MCMC with multiple walker class.

    This is for debugging purpose, to check the correctness of the MCMC implementation and the derivatives.

    MCMC class. Running MCMC with multiple walkers. The independent 'num_walkers' MCMCs are
    vectrized via the jax-vmap function.

    Args:
        hamiltonian_data (Hamiltonian_data): an instance of Hamiltonian_data.
        mcmc_seed (int): seed for the MCMC chain.
        num_walkers (int): the number of walkers.
        num_mcmc_per_measurement (int): the number of MCMC steps between a value (e.g., local energy) measurement.
        Dt (float): electron move step (bohr)
        epsilon_AS (float): the exponent of the AS regularization
        comput_log_WF_param_deriv (bool): if True, compute the derivatives of E wrt. variational parameters.
        comput_e_L_param_deriv (bool, optional): Keep local energy variational parameter derivatives (de_L / dc).
        comput_position_deriv (bool): if True, compute the derivatives of E wrt. atomic positions.
        random_discretized_mesh (bool): randomize the quadrature mesh for non-local ECP terms.
        use_swct (bool): apply Space Warp Coordinate Transformation to atomic forces.
        mpi_comm_in (MPI.Comm | None): sampling communicator. ``None`` uses the module global
            ``mpi_comm`` (``COMM_WORLD``).
        world_comm_in (MPI.Comm | None): all-rank communicator. ``None`` uses ``mpi_comm_in``.
    """

    def __init__(
        self,
        hamiltonian_data: Hamiltonian_data | None = None,
        mcmc_seed: "int | tuple[int, ...]" = 34467,
        num_walkers: int = 40,
        num_mcmc_per_measurement: int = 16,
        Dt: float = 2.0,
        epsilon_AS: float = 1e-1,
        comput_log_WF_param_deriv: bool = False,
        comput_e_L_param_deriv: bool = False,
        comput_position_deriv: bool = False,
        random_discretized_mesh: bool = True,
        use_swct: bool = True,
        mpi_comm_in: MPI.Comm | None = None,
        world_comm_in: MPI.Comm | None = None,
    ) -> None:
        """Initialize a MCMC class, creating list holding results."""
        # Mirror the production communicator ownership:
        # production MCMC comm injection on the debug sibling so the
        # production-vs-debug dual-path stays uniform under twist driving.
        # Defaults fall back to the module-global ``self._mpi_comm``, keeping
        # single-twist runs byte-for-byte unchanged.
        self._mpi_comm = mpi_comm_in if mpi_comm_in is not None else mpi_comm
        self._world_comm = world_comm_in if world_comm_in is not None else self._mpi_comm
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_size = self._mpi_comm.Get_size()

        self.__mcmc_seed = mcmc_seed
        self.__num_walkers = num_walkers
        self.__num_mcmc_per_measurement = num_mcmc_per_measurement
        self.__Dt = Dt
        self.__epsilon_AS = epsilon_AS
        self.__comput_log_WF_param_deriv = comput_log_WF_param_deriv
        self.__comput_e_L_param_deriv = comput_e_L_param_deriv
        self.__comput_position_deriv = comput_position_deriv
        self.__random_discretized_mesh = random_discretized_mesh
        self.__use_swct = use_swct

        # set hamiltonian_data
        self.__hamiltonian_data = hamiltonian_data

        # seeds -- hierarchical SeedSequence derivation, byte-identical to the
        # production MCMC class (see derive_rank_rng_streams for the key layout)
        # so MCMC <-> _MCMC_debug parity tests stay aligned.
        self.__np_seed_words, self.__mpi_seed = derive_rank_rng_streams(self.__mcmc_seed, self._mpi_rank)
        self.__jax_PRNG_key = jax.random.PRNGKey(self.__mpi_seed)
        # Use jax.random.split (batched) to match the production MCMC class.
        self.__jax_PRNG_key_list = jax.random.split(self.__jax_PRNG_key, self.__num_walkers)

        # initialize random seed (multi-word init_by_array path)
        np.random.seed(self.__np_seed_words)

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

        # check if only up electrons are updated
        if tot_num_electron_dn == 0:
            self.__only_up_electron = True
        else:
            self.__only_up_electron = False

        coords = hamiltonian_data.structure_data._positions_cart_jnp
        # Cell for PBC wrapping of the seed configuration (None for molecules).
        cell = hamiltonian_data.structure_data.cell if hamiltonian_data.structure_data.pbc_flag else None

        ## generate initial electron configurations
        r_carts_up, r_carts_dn, up_owner, dn_owner = _generate_init_electron_configurations(
            tot_num_electron_up, tot_num_electron_dn, self.__num_walkers, charges, coords, cell
        )

        # Electron assignment for all atoms is complete.

        dtype_jnp = jnp.float64
        self.__latest_r_up_carts = jnp.asarray(r_carts_up, dtype=dtype_jnp)
        self.__latest_r_dn_carts = jnp.asarray(r_carts_dn, dtype=dtype_jnp)

        logger.debug(f"  initial r_up_carts= {self.__latest_r_up_carts}")
        logger.debug(f"  initial r_dn_carts = {self.__latest_r_dn_carts}")
        logger.debug(f"  initial r_up_carts.shape = {self.__latest_r_up_carts.shape}")
        logger.debug(f"  initial r_dn_carts.shape = {self.__latest_r_dn_carts.shape}")
        logger.debug("")

        # print out the number of walkers/MPI processes
        logger.info(f"The number of MPI process = {self._mpi_size}.")
        logger.info(f"The number of walkers assigned for each MPI process = {self.__num_walkers}.")
        logger.info("")

        # SWCT data: the SWCT functions take structure_data directly.

        # init_attributes
        self.__init_attributes()

    def __init_attributes(self) -> None:
        # mcmc counter
        self.__mcmc_counter = 0

        # mcmc accepted/rejected moves (kept on device; convert to host only when logging)
        self.__accepted_moves = jnp.array(0, dtype=jnp.int64)
        self.__rejected_moves = jnp.array(0, dtype=jnp.int64)

        # stored weight (w_L)
        self.__stored_w_L = []

        # stored local energy (e_L)
        self.__stored_e_L = []

        # stored local energy (e_L2)
        self.__stored_e_L2 = []

        # stored de_L / dR
        self.__stored_grad_e_L_dR = []

        # stored de_L / dr_up
        self.__stored_grad_e_L_r_up = []

        # stored de_L / dr_dn
        self.__stored_grad_e_L_r_dn = []

        # stored dln_Psi / dr_up
        self.__stored_grad_ln_Psi_r_up = []

        # stored dln_Psi / dr_dn
        self.__stored_grad_ln_Psi_r_dn = []

        # stored dln_Psi / dR
        self.__stored_grad_ln_Psi_dR = []

        # stored Omega_up (SWCT)
        self.__stored_omega_up = []

        # stored Omega_dn (SWCT)
        self.__stored_omega_dn = []

        # stored sum_i d omega/d r_i for up spins (SWCT)
        self.__stored_grad_omega_r_up = []

        # stored sum_i d omega/d r_i for dn spins (SWCT)
        self.__stored_grad_omega_r_dn = []

        # stored per-walker HF / Pulay stress tensors (3, 3) and e_L * stress_PP.
        # Mirrors the production MCMC buffers so the debug route exposes get_aT /
        # get_aP for the production-vs-debug stress parity.
        self.__stored_stress_HF = []
        self.__stored_stress_PP = []
        self.__stored_E_L_stress_PP = []

        # stored parameter gradients keyed by block name
        self.__stored_log_WF_param_grads: dict[str, list] = defaultdict(list)

        # stored de_L/dc parameter gradients keyed by block name
        self.__stored_e_L_param_grads: dict[str, list] = defaultdict(list)

    def run(self, num_mcmc_steps: int = 0) -> None:
        """Launch MCMCs with the set multiple walkers.

        Args:
            num_mcmc_steps (int):
                the number of total mcmc steps per walker.
        """
        logger.info("")
        logger.info("This is a debugging class! It supposed to be very slow.")
        logger.info("")

        dtype_jnp = jnp.float64
        dtype_np = np.float64

        _e_L_fn = compute_local_energy

        # The independent-loop debug
        # driver routes every gradient (position r, force R, stress eps, and the
        # c-axis param grads) through the SAME unified ``jqmc.derivatives``
        # interface. The backend is
        # ``analytic=_ANALYTIC_GRAD_ENABLED`` -- the SAME flag production uses --
        # so debug and production always agree on the derivative backend. The
        # mask / ``stop_gradient`` is fully encapsulated inside ``derivatives``; it
        # does not execute on the debug run path. The driver keeps only its
        # per-walker wrapper --
        # here the deliberately eager Python loop (this class is the slow oracle),
        # NOT production's jit(vmap). Imported lazily: ``derivatives._auto`` lazily
        # imports this module's strained-forward helpers for its ``eps`` axis, so a
        # module-level import would cycle. The all-block c-axis mask (every
        # parameter block ``True`` makes the mask the identity)
        # reproduces the debug's full-parameter gradient.
        from . import derivatives

        _c_all_param_flags = {k: True for k in derivatives._PARAM_FLAG_KEYS}
        _pos_flags = {"opt_r": True, "opt_R": True}
        _eps_flags = {"opt_eps": True}

        # MAIN MCMC loop from here !!!
        logger.info("Start MCMC")
        num_mcmc_done = 0
        mcmc_interval = max(1, int(num_mcmc_steps / 10))  # %

        for i_mcmc_step in range(num_mcmc_steps):
            if i_mcmc_step % mcmc_interval == 0:
                progress = (i_mcmc_step + self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
                logger.info(
                    f"  Progress: MCMC step = {i_mcmc_step + self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %"
                )

            accepted_moves_nw = np.zeros(self.__num_walkers, dtype=np.int32)
            rejected_moves_nw = np.zeros(self.__num_walkers, dtype=np.int32)
            latest_r_up_carts = np.array(self.__latest_r_up_carts)
            latest_r_dn_carts = np.array(self.__latest_r_dn_carts)
            jax_PRNG_key_list = np.array(self.__jax_PRNG_key_list)

            for i_walker in range(self.__num_walkers):
                accepted_moves = 0
                rejected_moves = 0
                r_up_carts = latest_r_up_carts[i_walker]
                r_dn_carts = latest_r_dn_carts[i_walker]
                jax_PRNG_key = jax_PRNG_key_list[i_walker]

                num_mcmc_per_measurement = self.__num_mcmc_per_measurement
                hamiltonian_data = self.__hamiltonian_data
                Dt = self.__Dt
                epsilon_AS = self.__epsilon_AS

                for _ in range(num_mcmc_per_measurement):
                    total_electrons = len(r_up_carts) + len(r_dn_carts)

                    # Choose randomly if the electron comes from up or dn
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    rand_num = jax.random.randint(subkey, shape=(), minval=0, maxval=total_electrons)

                    # boolean: "up" or "dn"
                    # is_up == True -> up,False -> dn
                    is_up = rand_num < len(r_up_carts)

                    # an index chosen from up electons
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    up_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_up_carts))

                    # an index chosen from dn electrons
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    dn_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(r_dn_carts))

                    if is_up:
                        selected_electron_index = up_index
                        old_r_cart = r_up_carts[selected_electron_index]
                    else:
                        selected_electron_index = dn_index
                        old_r_cart = r_dn_carts[selected_electron_index]

                    # choose the nearest atom index
                    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, old_r_cart)

                    # charges
                    if hamiltonian_data.coulomb_potential_data.ecp_flag:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers) - jnp.array(
                            hamiltonian_data.coulomb_potential_data.z_cores
                        )
                    else:
                        charges = np.array(hamiltonian_data.structure_data.atomic_numbers)

                    # PBC fix (see the production rank-1 sibling for the
                    # full rationale): use the minimum-image displacement
                    # so the cusp-protection step size is correct even
                    # when the nearest nucleus image lies outside the home
                    # cell. Molecular cells short-circuit min-image inside
                    # the helper, so this is a no-op for ``pbc_flag=False``.
                    Z = charges[nearest_atom_index]
                    rel_R_min = _get_min_dist_rel_R_cart_np(
                        structure_data=hamiltonian_data.structure_data,
                        r_cart=np.asarray(old_r_cart, dtype=dtype_np),
                        i_atom=int(nearest_atom_index),
                    )
                    norm_r_R = np.linalg.norm(rel_R_min)
                    f_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                    sigma = f_l * Dt
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    g = jax.random.normal(subkey, shape=()) * sigma

                    # choose x,y,or,z
                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    random_index = jax.random.randint(subkey, shape=(), minval=0, maxval=3)

                    # plug g into g_vector
                    g_vector = np.zeros(3, dtype=dtype_np)
                    g_vector[random_index] = g

                    new_r_cart = old_r_cart + g_vector

                    # set proposed r_up_carts and r_dn_carts.
                    if is_up:
                        proposed_r_up_carts = r_up_carts.copy()
                        proposed_r_up_carts[selected_electron_index] = new_r_cart
                        proposed_r_dn_carts = r_dn_carts
                    else:
                        proposed_r_up_carts = r_up_carts
                        proposed_r_dn_carts = r_dn_carts.copy()
                        proposed_r_dn_carts[selected_electron_index] = new_r_cart

                    # choose the nearest atom index
                    nearest_atom_index = _find_nearest_index_jnp(hamiltonian_data.structure_data, new_r_cart)

                    # PBC fix: same rationale as the forward proposal.
                    Z = charges[nearest_atom_index]
                    rel_R_min_new = _get_min_dist_rel_R_cart_np(
                        structure_data=hamiltonian_data.structure_data,
                        r_cart=np.asarray(new_r_cart, dtype=dtype_np),
                        i_atom=int(nearest_atom_index),
                    )
                    norm_r_R = np.linalg.norm(rel_R_min_new)
                    f_prime_l = 1 / Z**2 * (1 + Z**2 * norm_r_R) / (1 + norm_r_R)

                    T_ratio = (f_l / f_prime_l) * jnp.exp(
                        -(np.linalg.norm(new_r_cart - old_r_cart) ** 2)
                        * (1.0 / (2.0 * f_prime_l**2 * Dt**2) - 1.0 / (2.0 * f_l**2 * Dt**2))
                    )

                    # original trial WFs
                    Jastrow_T_p = compute_Jastrow_part_ratio_zone(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )

                    Jastrow_T_o = compute_Jastrow_part_ratio_zone(
                        jastrow_data=hamiltonian_data.wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    Det_T_p = compute_det_geminal_all_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )

                    Det_T_o = compute_det_geminal_all_elements(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    # compute AS regularization factors, R_AS and R_AS_eps
                    R_AS_p = compute_AS_regularization_factor(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=proposed_r_up_carts,
                        r_dn_carts=proposed_r_dn_carts,
                    )
                    R_AS_o = compute_AS_regularization_factor(
                        geminal_data=hamiltonian_data.wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts,
                    )

                    # modified trial WFs (issues #131/#134: same guiding factor as the kernels)
                    R_AS_ratio = _as_guiding_ratio(R_AS_p, R_AS_o, epsilon_AS)
                    WF_ratio = np.exp(Jastrow_T_p - Jastrow_T_o) * (Det_T_p / Det_T_o)

                    # compute R_ratio (|.|^2 via (r * conj(r)).real; bit-identical for
                    # real r, keeps the Metropolis ratio real-positive when complex)
                    _ratio = R_AS_ratio * WF_ratio
                    R_ratio = (_ratio * np.conj(_ratio)).real

                    logger.devel(f"R_ratio, T_ratio = {R_ratio}, {T_ratio}")
                    acceptance_ratio = np.min(jnp.array([1.0, R_ratio * T_ratio], dtype=dtype_jnp))
                    logger.devel(f"acceptance_ratio = {acceptance_ratio}")

                    jax_PRNG_key, subkey = jax.random.split(jax_PRNG_key)
                    b = jax.random.uniform(subkey, shape=(), minval=0.0, maxval=1.0)

                    if b < acceptance_ratio:
                        accepted_moves += 1
                        # Keep walker positions UNWRAPPED here, mirroring the
                        # JIT accept sites above (see those for the rationale:
                        # ``compute_AOs`` wraps internally + the complex Bloch
                        # leaf's ``wrap_restore`` is NOT idempotent under an
                        # external wrap of the walker position).
                        r_up_carts = np.asarray(proposed_r_up_carts, dtype=dtype_np)
                        r_dn_carts = np.asarray(proposed_r_dn_carts, dtype=dtype_np)
                    else:
                        rejected_moves += 1

                accepted_moves_nw[i_walker] = accepted_moves
                rejected_moves_nw[i_walker] = rejected_moves
                latest_r_up_carts[i_walker] = r_up_carts
                latest_r_dn_carts[i_walker] = r_dn_carts
                jax_PRNG_key_list[i_walker] = jax_PRNG_key

            # store vmapped outcomes
            self.__accepted_moves = self.__accepted_moves + np.sum(accepted_moves_nw)
            self.__rejected_moves = self.__rejected_moves + np.sum(rejected_moves_nw)
            self.__latest_r_up_carts = jnp.asarray(latest_r_up_carts, dtype=dtype_jnp)
            self.__latest_r_dn_carts = jnp.asarray(latest_r_dn_carts, dtype=dtype_jnp)
            self.__jax_PRNG_key_list = jnp.array(jax_PRNG_key_list)

            # generate rotation matrices (for non-local ECPs)
            RTs = []
            for jax_PRNG_key in self.__jax_PRNG_key_list:
                if self.__random_discretized_mesh:
                    RTs.append(_generate_rotation_matrix(jax_PRNG_key))
                else:
                    RTs.append(jnp.eye(3, dtype=dtype_jnp))
            RTs = jnp.array(RTs)

            # evaluate observables
            e_L = jnp.stack(
                [
                    _e_L_fn(self.__hamiltonian_data, self.__latest_r_up_carts[i], self.__latest_r_dn_carts[i], RTs[i])
                    for i in range(self.__num_walkers)
                ]
            )
            self.__stored_e_L.append(e_L)
            # The variance buffer is the real quantity ``|e_L|^2``.
            self.__stored_e_L2.append(np.abs(e_L) ** 2)

            # compute AS regularization factors, R_AS and R_AS_eps
            R_AS = jnp.stack(
                [
                    compute_AS_regularization_factor(
                        self.__hamiltonian_data.wavefunction_data.geminal_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                    )
                    for i in range(self.__num_walkers)
                ]
            )
            w_L = _as_walker_weight(R_AS, self.__epsilon_AS)
            self.__stored_w_L.append(w_L)

            if self.__comput_position_deriv:
                # de_L/dr (drift) and de_L/dR
                # (Hellmann-Feynman force) via the derivatives interface
                # (analytic=_ANALYTIC_GRAD_ENABLED), looped per walker. The "r" axis returns the
                # ``(grad_r_up, grad_r_dn)`` drift pair; the "R" axis returns the
                # per-atom force already summed by ``accumulate_position_grad``
                # inside the interface (bit-identical to applying it host-side to
                # the ham grad, since accumulate_position_grad is linear per walker).
                _grad_e_L_results = [
                    derivatives.compute_grad_e_L(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                        RTs[i],
                        _pos_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                    )
                    for i in range(self.__num_walkers)
                ]
                grad_e_L_r_up = jnp.stack([r["r"][0] for r in _grad_e_L_results])
                grad_e_L_r_dn = jnp.stack([r["r"][1] for r in _grad_e_L_results])

                self.__stored_grad_e_L_r_up.append(grad_e_L_r_up)
                self.__stored_grad_e_L_r_dn.append(grad_e_L_r_dn)

                grad_e_L_R = jnp.stack([r["R"] for r in _grad_e_L_results])
                self.__stored_grad_e_L_dR.append(grad_e_L_R)

                # dln|Psi|/dr (drift) and dln|Psi|/dR
                # (Pulay) via the derivatives interface (analytic=_ANALYTIC_GRAD_ENABLED),
                # looped per walker. Same axis contract as the e_L path above.
                _grad_ln_psi_results = [
                    derivatives.compute_grad_ln_Psi(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                        _pos_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                    )
                    for i in range(self.__num_walkers)
                ]
                grad_ln_Psi_r_up = jnp.stack([r["r"][0] for r in _grad_ln_psi_results])
                grad_ln_Psi_r_dn = jnp.stack([r["r"][1] for r in _grad_ln_psi_results])

                self.__stored_grad_ln_Psi_r_up.append(grad_ln_Psi_r_up)
                self.__stored_grad_ln_Psi_r_dn.append(grad_ln_Psi_r_dn)

                grad_ln_Psi_dR = jnp.stack([r["R"] for r in _grad_ln_psi_results])
                self.__stored_grad_ln_Psi_dR.append(grad_ln_Psi_dR)

                if self.__use_swct:
                    omega_up = jnp.stack(
                        [
                            evaluate_swct_omega(self.__hamiltonian_data.structure_data, self.__latest_r_up_carts[i])
                            for i in range(self.__num_walkers)
                        ]
                    )

                    omega_dn = jnp.stack(
                        [
                            evaluate_swct_omega(self.__hamiltonian_data.structure_data, self.__latest_r_dn_carts[i])
                            for i in range(self.__num_walkers)
                        ]
                    )

                    grad_omega_dr_up = jnp.stack(
                        [
                            evaluate_swct_domega(self.__hamiltonian_data.structure_data, self.__latest_r_up_carts[i])
                            for i in range(self.__num_walkers)
                        ]
                    )

                    grad_omega_dr_dn = jnp.stack(
                        [
                            evaluate_swct_domega(self.__hamiltonian_data.structure_data, self.__latest_r_dn_carts[i])
                            for i in range(self.__num_walkers)
                        ]
                    )
                else:
                    n_up = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
                    n_dn = self.__hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
                    n_atoms = self.__hamiltonian_data.structure_data.natom
                    omega_up = jnp.zeros((self.__num_walkers, n_atoms, n_up), dtype=dtype_jnp)
                    omega_dn = jnp.zeros((self.__num_walkers, n_atoms, n_dn), dtype=dtype_jnp)
                    grad_omega_dr_up = jnp.zeros((self.__num_walkers, n_atoms, 3), dtype=dtype_jnp)
                    grad_omega_dr_dn = jnp.zeros((self.__num_walkers, n_atoms, 3), dtype=dtype_jnp)

                self.__stored_omega_up.append(omega_up)
                self.__stored_omega_dn.append(omega_dn)
                self.__stored_grad_omega_r_up.append(grad_omega_dr_up)
                self.__stored_grad_omega_r_dn.append(grad_omega_dr_dn)

                # Debug per-walker Hellmann-Feynman and Pulay
                # stress tensors (sigma_HF = d e_L / d eps, sigma_PP =
                # d ln|Psi| / d eps, both at eps = 0), routed through the
                # derivatives interface (analytic=_ANALYTIC_GRAD_ENABLED), looped per walker. Its
                # "eps" axis REUSES this module's ``_real_local_energy_strained`` /
                # ``_local_log_psi_strained`` strained-forward helpers via the same
                # jacrev, so the 3x3 tensors retain the original values.
                # Periodic only --
                # molecular runs return zeros from get_aT, so leaving the buffers
                # empty there matches production bookkeeping.
                if self.__hamiltonian_data.structure_data.pbc_flag:
                    _stress_HF = jnp.stack(
                        [
                            derivatives.compute_grad_e_L(
                                self.__hamiltonian_data,
                                self.__latest_r_up_carts[i],
                                self.__latest_r_dn_carts[i],
                                RTs[i],
                                _eps_flags,
                                analytic=_ANALYTIC_GRAD_ENABLED,
                            )["eps"]
                            for i in range(self.__num_walkers)
                        ]
                    )  # (nw, 3, 3)
                    _stress_PP = jnp.stack(
                        [
                            derivatives.compute_grad_ln_Psi(
                                self.__hamiltonian_data,
                                self.__latest_r_up_carts[i],
                                self.__latest_r_dn_carts[i],
                                _eps_flags,
                                analytic=_ANALYTIC_GRAD_ENABLED,
                            )["eps"]
                            for i in range(self.__num_walkers)
                        ]
                    )  # (nw, 3, 3)
                    _stress_PP_np = np.asarray(_stress_PP)
                    self.__stored_stress_HF.append(np.asarray(_stress_HF))
                    self.__stored_stress_PP.append(_stress_PP_np)
                    self.__stored_E_L_stress_PP.append(np.einsum("w,wij->wij", np.asarray(e_L), _stress_PP_np))

            if self.__comput_log_WF_param_deriv or self.__comput_e_L_param_deriv:
                # MPI dgesdd guard: thread a concrete ``geminal_inv`` into the analytic
                # c-axis kernels so their lowered graph contains NO LAPACK SVD --
                # ``resolve_geminal_inv(G, None)`` would recompute a fresh
                # ``geminal_pinv`` whose ``lapack_dgesdd_ffi`` custom call is
                # unavailable on some MPI host backends. The inverse comes from the
                # same jitted ``_geminal_inv_batched`` program the production sweep
                # seeds from, so the value is the
                # identical SVD pinv, just materialised OUTSIDE the c-axis graph.
                _, _gem_inv_b, _, _ = _geminal_inv_batched(
                    self.__hamiltonian_data.wavefunction_data.geminal_data,
                    jnp.asarray(self.__latest_r_up_carts),
                    jnp.asarray(self.__latest_r_dn_carts),
                )

            if self.__comput_log_WF_param_deriv:
                # O_k = d ln|Psi| / dc via the
                # derivatives interface (analytic=_ANALYTIC_GRAD_ENABLED), looped per walker. The
                # all-block c-axis mask is the identity, equivalent to the unmasked
                # ``grad(evaluate_ln_wavefunction, argnums=0)``), so each call
                # returns the RAW ``Wavefunction_data``-shaped grad; the debug's
                # own ``collect_param_grads`` / ``flatten_param_grads`` reduction is
                # kept unchanged so the stored block-name layout is identical.
                _grad_ln_psi_p_results = [
                    derivatives.compute_grad_ln_Psi(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                        _c_all_param_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                        geminal_inv=_gem_inv_b[i],
                    )["c"]
                    for i in range(self.__num_walkers)
                ]
                grad_ln_Psi_h = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *_grad_ln_psi_p_results)

                param_grads = _debug_collect_param_grads(
                    self.__hamiltonian_data.wavefunction_data, grad_ln_Psi_h, _c_all_param_flags
                )
                flat_param_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    param_grads, self.__num_walkers
                )

                for name, grad_val in flat_param_grads.items():
                    self.__stored_log_WF_param_grads[name].append(grad_val)
                    if hasattr(grad_val, "block_until_ready"):
                        grad_val.block_until_ready()

            if self.__comput_e_L_param_deriv:
                # de_L/dc via the derivatives
                # interface (analytic=_ANALYTIC_GRAD_ENABLED), looped per walker. The all-block
                # c-axis mask is the identity, equivalent to the unmasked gradient of
                # ``_e_L_fn(ham.replace(wavefunction_data=wf), ...)`` wrt wf), so
                # each call returns the RAW ``Wavefunction_data``-shaped grad; the
                # debug's own collect/flatten reduction is kept unchanged.
                _e_L_wf_results = [
                    derivatives.compute_grad_e_L(
                        self.__hamiltonian_data,
                        self.__latest_r_up_carts[i],
                        self.__latest_r_dn_carts[i],
                        RTs[i],
                        _c_all_param_flags,
                        analytic=_ANALYTIC_GRAD_ENABLED,
                        geminal_inv=_gem_inv_b[i],
                    )["c"]
                    for i in range(self.__num_walkers)
                ]
                grad_e_L_wf = jax.tree_util.tree_map(lambda *arrs: jnp.stack(arrs), *_e_L_wf_results)
                flat_grads = self.__hamiltonian_data.wavefunction_data.flatten_param_grads(
                    _debug_collect_param_grads(self.__hamiltonian_data.wavefunction_data, grad_e_L_wf, _c_all_param_flags),
                    self.__num_walkers,
                )
                from .derivatives.bridge import _project_real_param_grads

                flat_grads = _project_real_param_grads(
                    self.__hamiltonian_data.wavefunction_data,
                    flat_grads,
                    _c_all_param_flags,
                )
                for name, grad_val in flat_grads.items():
                    self.__stored_e_L_param_grads[name].append(grad_val)

            num_mcmc_done += 1

        progress = (num_mcmc_done + self.__mcmc_counter) / (num_mcmc_steps + self.__mcmc_counter) * 100.0
        logger.info(
            f"  Progress: MCMC step = {num_mcmc_done + self.__mcmc_counter}/{num_mcmc_steps + self.__mcmc_counter}: {progress:.1f} %"
        )

        logger.info("End MCMC")
        logger.info("")

        self.__mcmc_counter += num_mcmc_done

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
        e_L = self.e_L[num_mcmc_warmup_steps:]
        e_L2 = self.e_L2[num_mcmc_warmup_steps:]
        w_L = self.w_L[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) from mcmc_counter and num_walkers
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
        _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
        _require(e_L2.shape == (M, nw), f"e_L2 shape {e_L2.shape} != ({M}, {nw})")

        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))
        w_L_e_L2_split = np.array_split(w_L * e_L2, num_mcmc_bin_blocks, axis=0)
        w_L_e_L2_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L2_split]))

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_e_L2_binned_local = w_L_e_L2_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_e_L2_binned_local = np.array(w_L_e_L2_binned_local)

        # Shape assertions: binned arrays must be 1D (num_mcmc_bin_blocks * nw,)
        M_binned = num_mcmc_bin_blocks * nw
        _require(w_L_binned_local.shape == (M_binned,), f"w_L_binned_local shape {w_L_binned_local.shape} != ({M_binned},)")
        _require(
            w_L_e_L_binned_local.shape == (M_binned,),
            f"w_L_e_L_binned_local shape {w_L_e_L_binned_local.shape} != ({M_binned},)",
        )
        _require(
            w_L_e_L2_binned_local.shape == (M_binned,),
            f"w_L_e_L2_binned_local shape {w_L_e_L2_binned_local.shape} != ({M_binned},)",
        )

        w_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L2_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L2_binned_local, axis=0), op=MPI.SUM)

        M_local = w_L_binned_local.size
        logger.debug(f"The number of local binned samples = {M_local}")

        E_jackknife_binned_local = [
            (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
            for m in range(M_local)
        ]

        E2_jackknife_binned_local = [
            (w_L_e_L2_binned_global_sum - w_L_e_L2_binned_local[m]) / (w_L_binned_global_sum - w_L_binned_local[m])
            for m in range(M_local)
        ]

        # Mirror the production get_E
        # Var-formula and Im<E> diagnostic on the debug class so the
        # production-vs-debug dual-path check stays meaningful at non-Gamma.
        # Bit-identical on the real path.
        _E_local_arr = np.array(E_jackknife_binned_local)
        _E2_local_arr = np.array(E2_jackknife_binned_local)
        _E_is_complex = np.iscomplexobj(_E_local_arr)
        if _E_is_complex:
            abs_E_sq_local = np.abs(_E_local_arr) ** 2
        else:
            abs_E_sq_local = _E_local_arr**2
        Var_jackknife_binned_local = list(_E2_local_arr - abs_E_sq_local)

        # MPI allreduce
        E_jackknife_binned = self._mpi_comm.allreduce(E_jackknife_binned_local, op=MPI.SUM)
        Var_jackknife_binned = self._mpi_comm.allreduce(Var_jackknife_binned_local, op=MPI.SUM)
        E_jackknife_binned = np.array(E_jackknife_binned)
        Var_jackknife_binned = np.array(Var_jackknife_binned)
        M_total = len(E_jackknife_binned)
        logger.debug(f"The number of total binned samples = {M_total}")

        # Project the complex jackknife mean to the real
        # physical observable; log Im<E> as a fixed-phase contamination guard.
        if _E_is_complex:
            Im_E_binned = np.imag(E_jackknife_binned)
            E_jackknife_binned = np.real(E_jackknife_binned)
        else:
            Im_E_binned = None

        # jackknife mean and std (real after projection)
        E_mean = np.average(E_jackknife_binned)
        E_std = np.sqrt(M_total - 1) * np.std(E_jackknife_binned)
        Var_mean = np.average(Var_jackknife_binned)
        Var_std = np.sqrt(M_total - 1) * np.std(Var_jackknife_binned)

        if Im_E_binned is not None:
            Im_E_mean = float(np.average(Im_E_binned))
            if E_std > 0 and abs(Im_E_mean) > 2.0 * E_std:
                logger.warning(
                    "Complex-energy guard: |Im<E>| = %.3e exceeds 2 * E_std = %.3e",
                    Im_E_mean,
                    E_std,
                )
            else:
                logger.devel("Im<E> = %.3e (E_std = %.3e, OK)", Im_E_mean, E_std)

        logger.info(f"E = {E_mean} +- {E_std} Ha.")
        logger.info(f"Var(E) = {Var_mean} +- {Var_std} Ha^2.")

        return (E_mean, E_std, Var_mean, Var_std)

    def get_aF(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> ArrayEstimate:
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
        w_L = self.w_L[num_mcmc_warmup_steps:]
        e_L = self.e_L[num_mcmc_warmup_steps:]
        de_L_dR = self.de_L_dR[num_mcmc_warmup_steps:]
        de_L_dr_up = self.de_L_dr_up[num_mcmc_warmup_steps:]
        de_L_dr_dn = self.de_L_dr_dn[num_mcmc_warmup_steps:]
        dln_Psi_dr_up = self.dln_Psi_dr_up[num_mcmc_warmup_steps:]
        dln_Psi_dr_dn = self.dln_Psi_dr_dn[num_mcmc_warmup_steps:]
        dln_Psi_dR = self.dln_Psi_dR[num_mcmc_warmup_steps:]
        omega_up = self.omega_up[num_mcmc_warmup_steps:]
        omega_dn = self.omega_dn[num_mcmc_warmup_steps:]
        domega_dr_up = self.domega_dr_up[num_mcmc_warmup_steps:]
        domega_dr_dn = self.domega_dr_dn[num_mcmc_warmup_steps:]

        # Shape assertions: (M, nw) for scalars; (M, nw, n_atoms, 3) for R-derivatives
        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        n_atoms = self.hamiltonian_data.structure_data.natom
        n_up = de_L_dr_up.shape[2] if de_L_dr_up.ndim >= 3 else 0
        n_dn = de_L_dr_dn.shape[2] if de_L_dr_dn.ndim >= 3 else 0
        _require(w_L.shape == (M, nw), f"w_L shape {w_L.shape} != ({M}, {nw})")
        _require(e_L.shape == (M, nw), f"e_L shape {e_L.shape} != ({M}, {nw})")
        _require(de_L_dR.shape == (M, nw, n_atoms, 3), f"de_L_dR shape {de_L_dR.shape} != ({M}, {nw}, {n_atoms}, 3)")
        _require(dln_Psi_dR.shape == (M, nw, n_atoms, 3), f"dln_Psi_dR shape {dln_Psi_dR.shape} != ({M}, {nw}, {n_atoms}, 3)")
        _require(de_L_dr_up.shape == (M, nw, n_up, 3), f"de_L_dr_up shape {de_L_dr_up.shape} != ({M}, {nw}, {n_up}, 3)")
        _require(de_L_dr_dn.shape == (M, nw, n_dn, 3), f"de_L_dr_dn shape {de_L_dr_dn.shape} != ({M}, {nw}, {n_dn}, 3)")
        _require(omega_up.shape == (M, nw, n_atoms, n_up), f"omega_up shape {omega_up.shape} != ({M}, {nw}, {n_atoms}, {n_up})")
        _require(omega_dn.shape == (M, nw, n_atoms, n_dn), f"omega_dn shape {omega_dn.shape} != ({M}, {nw}, {n_atoms}, {n_dn})")
        _require(
            domega_dr_up.shape == (M, nw, n_atoms, 3),
            f"domega_dr_up shape {domega_dr_up.shape} != ({M}, {nw}, {n_atoms}, 3)",
        )
        _require(
            domega_dr_dn.shape == (M, nw, n_atoms, 3),
            f"domega_dr_dn shape {domega_dr_dn.shape} != ({M}, {nw}, {n_atoms}, 3)",
        )

        force_HF = (
            de_L_dR + np.einsum("iwjk,iwkl->iwjl", omega_up, de_L_dr_up) + np.einsum("iwjk,iwkl->iwjl", omega_dn, de_L_dr_dn)
        )

        force_PP = (
            dln_Psi_dR
            + np.einsum("iwjk,iwkl->iwjl", omega_up, dln_Psi_dr_up)
            + np.einsum("iwjk,iwkl->iwjl", omega_dn, dln_Psi_dr_dn)
            + 1.0 / 2.0 * (domega_dr_up + domega_dr_dn)
        )

        E_L_force_PP = np.einsum("iw,iwjk->iwjk", e_L, force_PP)

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

        # MCMC case
        w_L_binned_local = w_L_binned
        w_L_e_L_binned_local = w_L_e_L_binned
        w_L_force_HF_binned_local = w_L_force_HF_binned
        w_L_force_PP_binned_local = w_L_force_PP_binned
        w_L_E_L_force_PP_binned_local = w_L_E_L_force_PP_binned

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        w_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L_binned_local, axis=0), op=MPI.SUM)
        w_L_force_HF_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_force_HF_binned_local, axis=0), op=MPI.SUM)
        w_L_force_PP_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_force_PP_binned_local, axis=0), op=MPI.SUM)
        w_L_E_L_force_PP_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_E_L_force_PP_binned_local, axis=0), op=MPI.SUM)

        M_local = w_L_binned_local.size
        logger.debug(f"The number of local binned samples = {M_local}")

        force_HF_jn_local = -1.0 * np.array(
            [
                (w_L_force_HF_binned_global_sum - w_L_force_HF_binned_local[j]) / (w_L_binned_global_sum - w_L_binned_local[j])
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

        _require(
            force_HF_jn_local.shape == force_Pulay_jn_local.shape,
            f"force_HF_jn shape {force_HF_jn_local.shape} != force_Pulay_jn shape {force_Pulay_jn_local.shape}",
        )
        # Mirror production MCMC get_aF and re-project
        # the per-binned-sample HF + Pulay sum so the force observable is
        # real-valued on the complex Bloch path. Bit-identical at Gamma.
        force_jn_local = list(np.real(force_HF_jn_local + force_Pulay_jn_local))

        # MPI allreduce
        force_jn = self._mpi_comm.allreduce(force_jn_local, op=MPI.SUM)
        force_jn = np.array(force_jn)
        M_total = len(force_jn)
        logger.debug(f"The number of total binned samples = {M_total}")

        force_mean = np.average(force_jn, axis=0)
        force_std = np.sqrt(M_total - 1) * np.std(force_jn, axis=0)

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.info(f"force = {force_mean} +- {force_std} Ha.")

        w_L_binned_local = np.array(w_L_binned_local)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned_local)
        w_L_force_HF_binned_local = np.array(w_L_force_HF_binned_local)
        w_L_force_PP_binned_local = np.array(w_L_force_PP_binned_local)
        w_L_E_L_force_PP_binned_local = np.array(w_L_E_L_force_PP_binned_local)

        logger.devel(f"force_mean.shape  = {force_mean.shape}.")
        logger.devel(f"force_std.shape  = {force_std.shape}.")
        logger.devel(f"force = {force_mean} +- {force_std} Ha.")

        return (force_mean, force_std)

    def get_aT(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> ArrayEstimate:
        """Compute the symmetric stress tensor with jackknife statistics (debug route).

        Independent counterpart of :py:meth:`MCMC.get_aT`: same covariance-
        corrected Hellmann-Feynman + Pulay estimator, symmetrised and divided by
        the cell volume, but assembled from the looped-jacrev stress buffers
        stored by the debug ``run``. Molecular structures return
        ``(zeros((3, 3)), zeros((3, 3)))`` (the buffers carry no contribution).

        Contract (fixed static ansatz): same as :py:meth:`MCMC.get_aT` -- the
        strain ``F = I + eps`` maps cell, nuclei and electrons as ``x -> F x``
        (ECP quadrature directions fixed) with basis exponents / coefficients, MO / geminal
        coefficients and Jastrow parameters held fixed, so the tensor is NOT
        the equation-of-state derivative of a re-optimized ground-state energy.
        """
        if not self.__hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3, 3)), np.zeros((3, 3))

        w_L = self.w_L[num_mcmc_warmup_steps:]
        e_L = self.e_L[num_mcmc_warmup_steps:]
        stress_HF = self.stress_HF_stored[num_mcmc_warmup_steps:]
        stress_PP = self.stress_PP_stored[num_mcmc_warmup_steps:]
        E_L_stress_PP = self.E_L_stress_PP_stored[num_mcmc_warmup_steps:]

        M = self.mcmc_counter - num_mcmc_warmup_steps
        nw = self.num_walkers
        _require(w_L.shape == (M, nw), "")
        _require(e_L.shape == (M, nw), "")
        _require(stress_HF.shape == (M, nw, 3, 3), "")
        _require(stress_PP.shape == (M, nw, 3, 3), "")
        _require(E_L_stress_PP.shape == (M, nw, 3, 3), "")

        # Weighted bin split (mirror ``get_aF`` structure with (3, 3) leaf).
        w_L_split = np.array_split(w_L, num_mcmc_bin_blocks, axis=0)
        w_L_e_L_split = np.array_split(w_L * e_L, num_mcmc_bin_blocks, axis=0)
        w_L_stress_HF_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_HF), num_mcmc_bin_blocks, axis=0)
        w_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, stress_PP), num_mcmc_bin_blocks, axis=0)
        w_L_E_L_stress_PP_split = np.array_split(np.einsum("iw,iwjk->iwjk", w_L, E_L_stress_PP), num_mcmc_bin_blocks, axis=0)

        w_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_split]))
        w_L_e_L_binned = list(np.ravel([np.sum(arr, axis=0) for arr in w_L_e_L_split]))

        def _bin_tensor(splits: Sequence[npt.NDArray]) -> list[np.ndarray]:
            tensor_sum = np.array([np.sum(arr, axis=0) for arr in splits])
            return list(tensor_sum.reshape(tensor_sum.shape[0] * tensor_sum.shape[1], 3, 3))

        w_L_stress_HF_binned = _bin_tensor(w_L_stress_HF_split)
        w_L_stress_PP_binned = _bin_tensor(w_L_stress_PP_split)
        w_L_E_L_stress_PP_binned = _bin_tensor(w_L_E_L_stress_PP_split)

        w_L_binned_local = np.array(w_L_binned)
        w_L_e_L_binned_local = np.array(w_L_e_L_binned)
        w_L_stress_HF_binned_local = np.array(w_L_stress_HF_binned)
        w_L_stress_PP_binned_local = np.array(w_L_stress_PP_binned)
        w_L_E_L_stress_PP_binned_local = np.array(w_L_E_L_stress_PP_binned)

        w_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_binned_local, axis=0), op=MPI.SUM)
        w_L_e_L_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_e_L_binned_local, axis=0), op=MPI.SUM)
        w_L_stress_HF_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_stress_HF_binned_local, axis=0), op=MPI.SUM)
        w_L_stress_PP_binned_global_sum = self._mpi_comm.allreduce(np.sum(w_L_stress_PP_binned_local, axis=0), op=MPI.SUM)
        w_L_E_L_stress_PP_binned_global_sum = self._mpi_comm.allreduce(
            np.sum(w_L_E_L_stress_PP_binned_local, axis=0), op=MPI.SUM
        )

        M_local = w_L_binned_local.size

        sigma_HF_jn_local = np.array(
            [
                (w_L_stress_HF_binned_global_sum - w_L_stress_HF_binned_local[j])
                / (w_L_binned_global_sum - w_L_binned_local[j])
                for j in range(M_local)
            ]
        )
        sigma_Pulay_jn_local = 2.0 * np.array(
            [
                (
                    (w_L_E_L_stress_PP_binned_global_sum - w_L_E_L_stress_PP_binned_local[j])
                    / (w_L_binned_global_sum - w_L_binned_local[j])
                    - (
                        (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                        * (w_L_stress_PP_binned_global_sum - w_L_stress_PP_binned_local[j])
                        / (w_L_binned_global_sum - w_L_binned_local[j])
                    )
                )
                for j in range(M_local)
            ]
        )

        # Re(.) on the complex Bloch path (Gamma is bit-identical); mirrors the
        # real-cast the production get_aF applies before symmetrising.
        sigma_jn_local = np.real(sigma_HF_jn_local + sigma_Pulay_jn_local)
        sigma_jn_local = 0.5 * (sigma_jn_local + sigma_jn_local.transpose(0, 2, 1))
        omega = float(np.abs(np.linalg.det(np.asarray(self.__hamiltonian_data.structure_data._cell_jnp))))
        sigma_jn_local = sigma_jn_local / omega

        sigma_jn = self._mpi_comm.allreduce(list(sigma_jn_local), op=MPI.SUM)
        sigma_jn = np.array(sigma_jn)
        M_total = len(sigma_jn)

        sigma_mean = np.average(sigma_jn, axis=0)
        sigma_std = np.sqrt(M_total - 1) * np.std(sigma_jn, axis=0)

        return sigma_mean, sigma_std

    def get_aP(
        self,
        num_mcmc_warmup_steps: int = 50,
        num_mcmc_bin_blocks: int = 10,
    ) -> ArrayEstimate:
        """Normal-pressure diagonal ``(P_x, P_y, P_z) = -(sigma_xx, sigma_yy, sigma_zz)`` (debug route).

        Molecular structures return ``(zeros(3), zeros(3))``.
        """
        if not self.__hamiltonian_data.structure_data.pbc_flag:
            return np.zeros((3,)), np.zeros((3,))
        sigma_mean, sigma_std = self.get_aT(num_mcmc_warmup_steps, num_mcmc_bin_blocks)
        return -np.diag(sigma_mean), np.diag(sigma_std)

    # hamiltonian
    @property
    def hamiltonian_data(self) -> Hamiltonian_data:
        """Return hamiltonian_data."""
        return self.__hamiltonian_data

    # dimensions of observables
    @property
    def mcmc_counter(self) -> int:
        """Return current MCMC counter."""
        return self.__mcmc_counter

    @property
    def num_walkers(self) -> int:
        """The number of walkers."""
        return self.__num_walkers

    # weights
    @property
    def w_L(self) -> npt.NDArray:
        """Return the stored weight array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_w_L)

    # observables
    @property
    def e_L(self) -> npt.NDArray:
        """Return the stored e_L array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_e_L)

    # observables
    @property
    def e_L2(self) -> npt.NDArray:
        """Return the stored e_L^2 array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_e_L2)

    @property
    def de_L_dR(self) -> npt.NDArray:
        """Return the stored de_L/dR array. dim: (mcmc_counter, num_walkers)."""
        return np.array(self.__stored_grad_e_L_dR)

    @property
    def de_L_dr_up(self) -> npt.NDArray:
        """Return the stored de_L/dr_up array. dim: (mcmc_counter, num_walkers, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_e_L_r_up)

    @property
    def de_L_dr_dn(self) -> npt.NDArray:
        """Return the stored de_L/dr_dn array. dim: (mcmc_counter, num_walkers, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_e_L_r_dn)

    @property
    def dln_Psi_dr_up(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_up array. dim: (mcmc_counter, num_walkers, num_electrons_up, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_up)

    @property
    def dln_Psi_dr_dn(self) -> npt.NDArray:
        """Return the stored dln_Psi/dr_down array. dim: (mcmc_counter, num_walkers, num_electrons_dn, 3)."""
        return np.array(self.__stored_grad_ln_Psi_r_dn)

    @property
    def dln_Psi_dR(self) -> npt.NDArray:
        """Return the stored dln_Psi/dR array. dim: (mcmc_counter, num_walkers, num_atoms, 3)."""
        return np.array(self.__stored_grad_ln_Psi_dR)

    @property
    def omega_up(self) -> npt.NDArray:
        """Return the stored Omega (for up electrons) array. dim: (mcmc_counter, num_walkers, num_atoms, num_electrons_up)."""
        return np.array(self.__stored_omega_up)

    @property
    def omega_dn(self) -> npt.NDArray:
        """Return the stored Omega (for down electrons) array. dim: (mcmc_counter, num_walkers, num_atoms, num_electons_dn)."""
        return np.array(self.__stored_omega_dn)

    @property
    def domega_dr_up(self) -> npt.NDArray:
        """Return the stored dOmega/dr_up array. dim: (mcmc_counter, num_walkers, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_up)

    @property
    def domega_dr_dn(self) -> npt.NDArray:
        """Return the stored dOmega/dr_dn array. dim: (mcmc_counter, num_walkers, num_electons_dn, 3)."""
        return np.array(self.__stored_grad_omega_r_dn)

    @property
    def stress_HF_stored(self) -> npt.NDArray:
        """Return the stored HF stress tensor. dim: (mcmc_counter, num_walkers, 3, 3)."""
        return np.array(self.__stored_stress_HF)

    @property
    def stress_PP_stored(self) -> npt.NDArray:
        """Return the stored Pulay stress tensor. dim: (mcmc_counter, num_walkers, 3, 3)."""
        return np.array(self.__stored_stress_PP)

    @property
    def E_L_stress_PP_stored(self) -> npt.NDArray:
        """Return the stored e_L * stress_PP tensor. dim: (mcmc_counter, num_walkers, 3, 3)."""
        return np.array(self.__stored_E_L_stress_PP)

    @property
    def dln_Psi_dc(self) -> dict[str, npt.NDArray]:
        """Return stored d ln Psi / dc parameter gradients keyed by block name."""
        return {name: np.array(values) for name, values in self.__stored_log_WF_param_grads.items()}

    @property
    def de_L_dc(self) -> dict[str, npt.NDArray]:
        """Return stored de_L / dc parameter gradients keyed by block name."""
        return {name: np.array(values) for name, values in self.__stored_e_L_param_grads.items()}

    @property
    def comput_e_L_param_deriv(self) -> bool:
        """Return whether de_L/dc computation is enabled."""
        return self.__comput_e_L_param_deriv

    def get_aH(
        self,
        blocks: list,
        g: npt.NDArray | None = None,
        num_mcmc_warmup_steps: int = 50,
        chosen_param_index: list | None = None,
        return_matrices: bool = False,
    ) -> tuple:
        r"""Compute H_0, H_1, H_2, S_2 for accelerated SR (aSR) gamma optimization.

        This is the debug (single-rank, no vmap, explicit-step) counterpart of
        ``MCMC.get_aH``.  Every intermediate quantity is computed in a named
        numpy step that maps directly to the mathematical formula so that the
        logic can be followed line by line.

        .. note::

            ``return_matrices=True`` (LM mode) returns the same six matrix
            components as the production class, including the non-symmetric
            bra row.

        Mathematical background
        -----------------------
        The energy along the aSR step  alpha -> alpha + gamma * g  is
        approximated as

            E(alpha + gamma*g)  =    (H0 + 2*gamma*H1 + gamma^2*H2)
                                    --------------------------------
                                          1 + gamma^2 * S2

        where g = S^{-1} f is the natural gradient and:

            H0  = E_alpha                              (current energy)
            H1  = -1/2 * g^T f                         (f = generalized force)
            H2  = g^T (B + K) g                        (curvature along g)
            S2  = g^T S g = g^T f = -2*H1             (overlap along g)

        Intermediate quantities (N = total samples, K = # params):

            w(i)        sample weight                   shape (N,)
            e_L(i)      local energy                    shape (N,)
            W           sum_i w(i)                      scalar
            E_bar       sum_i w(i) e_L(i) / W           scalar
            O_{i,k}     d ln Psi(i) / d c_k             shape (N, K)
            O_bar_k     sum_i w(i) O_{i,k} / W          shape (K,)
            dO_{i,k}    O_{i,k} - O_bar_k               shape (N, K)  centered
            dE_{i,k}    d e_L(i) / d c_k                shape (N, K)
            dE_bar_k    sum_i w(i) dE_{i,k} / W         shape (K,)
            ddE_{i,k}   dE_{i,k} - dE_bar_k             shape (N, K)  centered
            f_k         -2/W * sum_i w(i)(e_L-E_bar) dO_{i,k}   shape (K,)

        Args:
            blocks: Ordered variational blocks (same ordering as g).
            g: Natural gradient vector S^{-1}f. Required for aSR mode.
            num_mcmc_warmup_steps: Samples to discard as warmup.
            chosen_param_index: Optional subset of flattened indices.
            return_matrices: If True, return the full LM matrices and bra row.

        Returns:
            ``(H_0, H_1, H_2, S_2)`` in aSR mode, or
            ``(H_0, f_vec, S_matrix, K_matrix, B_matrix, bra_row)`` in LM mode.
        """
        if not self.__comput_log_WF_param_deriv:
            raise RuntimeError("get_aH requires compute_log_WF_param_deriv=True.")
        if not self.__comput_e_L_param_deriv:
            raise RuntimeError("get_aH requires comput_e_L_param_deriv=True.")

        # -- Step 1: Raw samples after warmup ----------------------------------
        # e_L_2d, w_L_2d have shape (M_steps, num_walkers).
        e_L_2d = self.e_L[num_mcmc_warmup_steps:]  # (M, nw)
        w_L_2d = self.w_L[num_mcmc_warmup_steps:]  # (M, nw)

        # Shape assertions: (M, nw) from mcmc_counter and num_walkers
        M_expected = self.mcmc_counter - num_mcmc_warmup_steps
        nw_expected = self.num_walkers
        _require(w_L_2d.shape == (M_expected, nw_expected), f"w_L_2d shape {w_L_2d.shape} != ({M_expected}, {nw_expected})")
        _require(e_L_2d.shape == (M_expected, nw_expected), f"e_L_2d shape {e_L_2d.shape} != ({M_expected}, {nw_expected})")

        # Flatten (M, nw) -> (N,) so each sample is one entry.
        M, nw = e_L_2d.shape
        N = M * nw
        e_L = e_L_2d.ravel()  # (N,)
        w = w_L_2d.ravel()  # (N,)

        # -- Step 2: Build O_matrix  (d ln Psi / dc)  shape (N, K) ------------
        # dln_Psi_dc is a dict  block_name -> array (M, nw, ...).
        # We gather blocks in the same order as `blocks` and flatten the
        # parameter dimensions to get a single (N, K) matrix.
        dln_Psi_dc_map = self.dln_Psi_dc
        O_cols = []
        for block in blocks:
            if block.name not in dln_Psi_dc_map:
                # grad flag was off for this block; treat as zero
                O_cols.append(np.zeros((N, block.size)))
                continue
            arr = dln_Psi_dc_map[block.name][num_mcmc_warmup_steps:]  # (M, nw, ...)
            arr = arr.reshape(N, -1)  # (N, block.size)
            O_cols.append(arr)
        O_matrix = np.concatenate(O_cols, axis=1)  # (N, K)
        K = O_matrix.shape[1]

        # Shape assertion
        _require(O_matrix.shape == (N, K), f"O_matrix shape {O_matrix.shape} != ({N}, {K})")

        # Project O_matrix for blocks with declared real-linear ties
        # (mirrors the projection in MCMC.get_dln_WF).
        _sym_start = 0
        for block in blocks:
            _sym_end = _sym_start + block.size
            if block.tangent_spec.kind == "linear_tie":
                _require(
                    block.tangent_spec.linear_tie is not None,
                    f"linear_tie block {block.name!r} has no linear-tie specification.",
                )
                O_matrix[:, _sym_start:_sym_end] = _project_linear_tie_storage(
                    O_matrix[:, _sym_start:_sym_end],
                    block.tangent_spec.linear_tie,
                    field=block.field,
                )
            _sym_start = _sym_end

        # -- Step 3: Build dE_matrix  (de_L / dc)  shape (N, K) ---------------
        de_L_dc_map = self.de_L_dc
        dE_cols = []
        for block in blocks:
            if block.name not in de_L_dc_map:
                dE_cols.append(np.zeros((N, block.size)))
                continue
            arr = de_L_dc_map[block.name][num_mcmc_warmup_steps:]  # (M, nw, ...)
            arr = arr.reshape(N, -1)  # (N, block.size)
            dE_cols.append(arr)
        dE_matrix = np.concatenate(dE_cols, axis=1)  # (N, K)

        # Shape assertion
        _require(dE_matrix.shape == (N, K), f"dE_matrix shape {dE_matrix.shape} != ({N}, {K})")

        # Apply optional parameter-index subset.
        if chosen_param_index is not None:
            O_matrix = O_matrix[:, chosen_param_index]
            dE_matrix = dE_matrix[:, chosen_param_index]
            g = g[chosen_param_index]

        # Shape assertions after optional subset
        K_sub = O_matrix.shape[1]
        _require(O_matrix.shape == (N, K_sub), f"O_matrix shape {O_matrix.shape} != ({N}, {K_sub})")
        _require(dE_matrix.shape == (N, K_sub), f"dE_matrix shape {dE_matrix.shape} != ({N}, {K_sub})")
        _require(w.shape == (N,), f"w shape {w.shape} != ({N},)")
        _require(e_L.shape == (N,), f"e_L shape {e_L.shape} != ({N},)")
        if g is not None:
            _require(g.shape == (K_sub,), f"g shape {g.shape} != ({K_sub},)")

        # -- Step 4: MPI-aware weighted averages --------------------------------
        W_local = float(np.sum(w))
        W = self._mpi_comm.allreduce(W_local, op=MPI.SUM)

        we_local = np.dot(w, e_L)
        we_global = self._mpi_comm.allreduce(we_local, op=MPI.SUM)
        # ``E_bar`` (== ``H_0``) is a physical-real energy;
        # ``np.real``-project the complex-twist finite-sampling Im part instead of
        # the silent ``float()`` cast (ComplexWarning at k != 0). Identity at Gamma.
        E_bar = np.real(we_global) / W

        K_sub = O_matrix.shape[1]
        # ``wO`` and ``wdE`` are covariance means; center
        # ``dO`` / ``ddE`` for the Hermitian S / K / B), so Im MUST survive --
        # route through the complex-aware ``allreduce_sum`` (bit-identical
        # MPI.DOUBLE for real; a real buffer cannot carry Im at k != 0).
        wO_local = w @ O_matrix  # (K_sub,)
        wO_global = allreduce_sum(self._mpi_comm, wO_local)
        O_bar = wO_global / W

        wdE_local = w @ dE_matrix  # (K_sub,)
        wdE_global = allreduce_sum(self._mpi_comm, wdE_local)
        dE_bar = wdE_global / W

        # -- Step 5: H_0  (current energy estimate) --------------------------
        H_0 = E_bar

        # -- Step 6: Centered observables --------------------------------------
        dO = O_matrix - O_bar[np.newaxis, :]  # (N, K)  O_k(i) - <O_k>
        ddE = dE_matrix - dE_bar[np.newaxis, :]  # (N, K)  dE_k(i) - <dE_k>

        # -- Step 7: Generalized force  f_k = -2/W sum_i w_i (e_L_i - E_bar) conj(dO_{i,k})
        # COMPLEX covariance force ``f_k = -2 <w conj(dO_k)(e_L - E_bar)> = -2 dE/dc*``
        # This matches ``get_gF`` and the SR
        # path). ``conj`` is the identity on the real Gamma path.
        de = e_L - E_bar  # (N,)  local energy fluctuation
        f_local = np.ascontiguousarray(-2.0 * ((w * de) @ np.conj(dO)))  # (K,)
        f_vec = allreduce_sum(self._mpi_comm, f_local) / W
        # f_vec is the covariance-only force (see the first get_aH for the
        # omitted <de_L/dc> term).

        # -- LM mode: build full matrices -------------------------------------
        if return_matrices:
            # If g (SR direction) is provided, prepend collective variable
            if g is not None:
                O_SR = dO @ g  # (N,)
                dE_SR = ddE @ g  # (N,)
                dO = np.column_stack([O_SR, dO])  # (N, K+1)
                ddE = np.column_stack([dE_SR, ddE])  # (N, K+1)
                # Recompute f_vec for the extended space with the same complex
                # covariance force as the base space.
                f_local_ext = np.ascontiguousarray(-2.0 * ((w * de) @ np.conj(dO)))
                f_vec = allreduce_sum(self._mpi_comm, f_local_ext) / W

            # S_{k,k'} = (conj(dO)^T @ diag(w) @ dO) / W.
            w_dO = w[:, np.newaxis] * dO
            S_local = dO.conj().T @ w_dO
            S_matrix = np.empty_like(S_local)
            self._mpi_comm.Allreduce(S_local, S_matrix, op=MPI.SUM)
            S_matrix /= W

            # K_{k,k'} = (conj(dO)^T @ diag(w * e_L) @ dO) / W.
            we_dO = (w * e_L)[:, np.newaxis] * dO
            K_local = dO.conj().T @ we_dO
            K_matrix = np.empty_like(K_local)
            self._mpi_comm.Allreduce(K_local, K_matrix, op=MPI.SUM)
            K_matrix /= W

            # B_{k,k'} = (conj(ddE)^T @ diag(w) @ dO) / W.
            w_ddE = w[:, np.newaxis] * ddE
            B_local = w_ddE.conj().T @ dO
            B_matrix = np.empty_like(B_local)
            self._mpi_comm.Allreduce(B_local, B_matrix, op=MPI.SUM)
            B_matrix /= W

            # Non-symmetric bra row (doc: lm_storage_augmented_matrices):
            # H_0a = <dO_a de_L> + D_a (UNCONJUGATED covariance + mean local-
            # energy derivative). Mirrors the production get_aH.
            f_noconj_local = np.ascontiguousarray(-2.0 * ((w * de) @ dO))
            f_noconj = allreduce_sum(self._mpi_comm, f_noconj_local) / W
            if g is not None:
                D_vec = np.concatenate([np.atleast_1d(dE_bar @ g), dE_bar])
            else:
                D_vec = dE_bar
            bra_row = -0.5 * f_noconj + D_vec

            return H_0, f_vec, S_matrix, K_matrix, B_matrix, bra_row

        # -- aSR mode: scalar projections along g -----------------------------
        _require(g is not None, "g is required for aSR mode (return_matrices=False)")

        # -- Step 8: H_1, NON-SYMMETRIC pencil average along g ----------------
        # 2 H_1 = H_01 + H_10 with H_10 (ket) = -1/2 Re(vdot(g, f)) and
        # H_01 (bra) = H_10 + D_g, D_g = Re(vdot(g, <de_L/dc>)) -- the
        # non-symmetric linear method (mirrors the production get_aH; see
        # while retaining the finite sample-level derivative in the bra row.
        # Conjugate the direction factor; vdot equals dot at Gamma.
        H_1_ket = -0.5 * np.real(np.vdot(g, f_vec))
        D_g = np.real(np.vdot(g, dE_bar))
        H_1 = H_1_ket + 0.5 * D_g

        # -- Step 9: S_2 = g^T S g = <w (g^T dO)^2>_w  (exact, computed from samples) --
        # Do NOT use S_2 = g^T f (= -2*H_1).  The SR solved
        # (S_scaled + sr_epsilon*I) g_scaled = b, so
        #   g^T f = g^T S g + sr_epsilon * ||g_scaled||^2.
        # Using g^T f overestimates S_2, makes the denominator of
        # E(gamma) grow too fast, and drives the optimal gamma to be unrealistically small.
        gdO = dO @ g  # (N,)  g-projected centered observable
        # Use ``|x|**2``; this is bit-identical to ``x**2`` at Gamma.
        S_2_local = float(np.dot(w, np.abs(gdO) ** 2).real)
        S_2 = self._mpi_comm.allreduce(S_2_local, op=MPI.SUM) / W

        # -- Step 10: K matrix contribution  g^H K g -------------------------
        #
        #   K_{k,k'} = 1/W sum_i  w_i * e_L_i * conj(dO_{i,k}) * dO_{i,k'}
        #
        #   g^H K g  = 1/W sum_i  w_i * e_L_i * |sum_k g_k dO_{i,k}|^2
        # This remains bit-identical on the real path.
        gKg_local = float(np.dot(w * e_L, np.abs(gdO) ** 2).real)
        gKg = self._mpi_comm.allreduce(gKg_local, op=MPI.SUM) / W

        # -- Step 11: B matrix contribution  g^H B g -------------------------
        #
        #   B_{k,k'} = 1/W sum_i  w_i * conj(ddE_{i,k}) * dO_{i,k'}  (Hermitian).
        #
        #   g^H B g  = 1/W sum_i  w_i * conj(sum_k g_k ddE_{i,k}) * (sum_k' g_k' dO_{i,k'})
        # This remains bit-identical on the real path.
        gdE = ddE @ g  # (N,)  g-projected centered de_L
        gBg_local = float(np.dot(w * np.conj(gdE), gdO).real)
        gBg = self._mpi_comm.allreduce(gBg_local, op=MPI.SUM) / W

        # -- Step 12: H_2 = g^T (B + K) g ------------------------------------
        H_2 = gBg + gKg

        return H_0, H_1, H_2, S_2
