"""Batched derivative closures matching the production driver convention.

The MCMC/GFMC drivers build a gradient closure once and call it per step over a
walker batch (``jit(vmap(...))`` with the hamiltonian on a ``None`` axis so it is a
TRACED input, not a closure constant -- otherwise every parameter update would
re-jit). These factories wrap the per-config dispatch (``compute_grad_*``) into that
shape; the walker axis is mapped by ``jqmc._setting.walker_map`` (all walkers at once
when they fit ``WALKER_STREAM_BUDGET_BYTES``, else in measured chunks). Production driver builders delegate to
these factories, making this module the single source of truth for gradient
closures.

The factories keep the single-axis ``wrt=``/``mask=`` factory signature
(the driver adapters and tests build one closure per axis), plus a FUSED
``wrt=("r", "R")`` entry that emits the position (drift + Pulay/HF force) tuple in
one trace because the production driver builds r and R together. The underlying
call uses the whitelist ``flags`` interface. ``analytic`` defaults to
``jqmc._setting.ANALYTIC_GRAD_ENABLED`` (read at build time), matching the driver's
module-level flag: ``True`` == STRICT analytic (an unsupported wavefunction raises
``NotImplementedError`` at trace time with no silent ``_auto`` fallback);
``False`` routes every axis through the ``_auto`` autograd twin.

``mask`` follows the ``with_param_grad_mask`` convention it always had: ``None`` =
no mask (every parameter block active, basis included); a partial dict is expanded
with the ``with_param_grad_mask`` defaults (J1/J2/J3/geminal/DTN True, basis False)
before being expressed as the whitelist ``flags``.

The r/R/eps analytic kernels operate on the original basis without calling
``cartesianize_orbitals``. This preserves the already-qualified production path
and avoids an unnecessary representation change for complex Hamiltonians. Complex
spherical and Cartesian pGTO conversion preserves the Bloch leaf, twist, and AO/MO
value, gradient, and Laplacian, but the force/stress closure does not require that
conversion.
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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Literal, TypeAlias

import jax
import numpy as np
from jax import numpy as jnp
from jax import typing as jnpt

from .. import _setting
from .._setting import walker_map
from ..atomic_orbital import AOs_base_data
from ..hamiltonians import Hamiltonian_data
from ..molecular_orbital import MOs_data
from ..wavefunction import VariationalParameterBlock, Wavefunction_data
from . import _PARAM_FLAG_KEYS, compute_grad_e_L, compute_grad_ln_Psi

GradientAxis: TypeAlias = Literal["c", "r", "R", "eps"]
GradientAxes: TypeAlias = GradientAxis | tuple[GradientAxis, ...]
GradientResult: TypeAlias = Wavefunction_data | jax.Array | tuple[jax.Array, ...]
BatchedGradient: TypeAlias = Callable[..., GradientResult]
GradientQuantity: TypeAlias = Literal["ln_Psi", "e_L"]
_FlatParameterGrads: TypeAlias = dict[str, np.ndarray]

# ``with_param_grad_mask`` keyword defaults (see wavefunction.Wavefunction_data):
# an absent key defaults to True for the parameter blocks and False for the basis
# blocks. ``mask=None`` means NO mask = every leaf differentiates = all blocks True.
_MASK_DEFAULTS = {k: not ("basis" in k) for k in _PARAM_FLAG_KEYS}


def _mask_key(
    wrt: GradientAxes,
    mask: Mapping[str, bool] | None,
) -> frozenset[str] | None:
    """Resolve a ``with_param_grad_mask`` request to a hashable frozenset of the TRUE
    param-block keys (``None`` when unmasked or a physical axis -- mask only affects c)."""
    if mask is None or wrt != "c":
        return None
    flags = dict(_MASK_DEFAULTS)
    flags.update({k: bool(v) for k, v in dict(mask).items()})
    return frozenset(k for k, v in flags.items() if v)


def _resolve_analytic(analytic: bool | None) -> bool:
    return _setting.ANALYTIC_GRAD_ENABLED if analytic is None else bool(analytic)


def _num_ao_static(orb_data: AOs_base_data | MOs_data) -> int:
    """Static AO count of one spin channel (0 when the leaf exposes none)."""
    from jqmc.orbital_tiling import TiledOrbitalSet_data

    if isinstance(orb_data, TiledOrbitalSet_data):
        # Bound transient per-electron primitive kernels, including the local
        # harmonic-to-monomial derivative expansion. Final occupied rank alone
        # underestimates pGTO/ECP mesh memory by the retained-image multiplicity.
        count = 0
        for block in orb_data.blocks:
            leaf = block.backend
            if hasattr(leaf, "_num_ao_kernel"):
                factor = 1
                if "sphe" in type(leaf).__name__:
                    l = max(leaf.angular_momentums, default=0)
                    factor = (l + 1) * (l + 2) // 2
                count += leaf._num_ao_kernel * factor
            else:
                count += leaf.num_ao
        return count
    aos = getattr(orb_data, "aos_data", orb_data)
    try:
        return int(aos.num_ao)
    except (AttributeError, TypeError):
        return 0


def _walker_batched(fn: Callable[..., GradientResult]) -> BatchedGradient:
    """``jit`` of ``fn(ham, *per_walker)`` mapped over the walker axis.

    Equivalent to ``jax.jit(jax.vmap(fn, in_axes=(None, 0, ...)))``: the Hamiltonian
    stays a traced, unbatched input and every other argument is batched on axis 0.
    The walkers are processed by ``jqmc._setting.walker_map``, i.e. all at once when
    the measured per-walker workspace times the walker count fits
    ``WALKER_STREAM_BUDGET_BYTES``, else in chunks that do.
    """

    def batched(ham: object, *walker_args: jax.Array) -> GradientResult:
        return walker_map(lambda *x: fn(ham, *x), *walker_args)

    return jax.jit(batched)


# Cache the inner jit(vmap(...)) objects so repeated factory calls (e.g. once per
# MCMC.run() across a run_optimize loop) REUSE the same compiled function instead of
# building a fresh jit -> recompiling every call. Keyed on ALL graph-shaping statics:
# quantity, axis (``wrt``), ``fast``, ``analytic``, the resolved ``mask_key``
# frozenset (maxsize is unbounded but the driver only ever presents a handful of
# masks), and whether the running Sherman-Morrison
# ``geminal_inv`` is threaded, and whether the private source path retains
# complete complex real-parameter directions.
@lru_cache(maxsize=None)
def _build(
    quantity: GradientQuantity,
    wrt: GradientAxes,
    fast: bool,
    analytic: bool,
    mask_key: frozenset[str] | None,
    with_geminal_inv: bool,
    source_mode: bool = False,
) -> BatchedGradient:
    resolved_mask = None if mask_key is None else {key: (key in mask_key) for key in _PARAM_FLAG_KEYS}
    if isinstance(wrt, tuple):
        flags = {f"opt_{axis}": True for axis in wrt}
    elif wrt in ("r", "R", "eps"):
        flags = {f"opt_{wrt}": True}
    elif wrt == "c":
        flags = {key: True for key in _PARAM_FLAG_KEYS} if resolved_mask is None else dict(_MASK_DEFAULTS)
        if resolved_mask is not None:
            flags.update({key: bool(value) for key, value in resolved_mask.items()})
    else:
        raise ValueError(f"unknown gradient axis {wrt!r}; expected 'c'/'r'/'R'/'eps' or a fused axis tuple")

    # Fused force outputs keep the driver-facing tuple order: local-energy
    # derivatives put the electron drift rows first, while log-wavefunction
    # derivatives put the nuclear Pulay row first. Stress is appended last.
    if isinstance(wrt, tuple):
        if "eps" in wrt:
            if quantity == "e_L":
                sel = lambda out: (out["r"][0], out["r"][1], out["R"], out["eps"])
            else:
                sel = lambda out: (out["R"], out["r"][0], out["r"][1], out["eps"])
        elif quantity == "e_L":
            sel = lambda out: (out["r"][0], out["r"][1], out["R"])
        else:
            sel = lambda out: (out["R"], out["r"][0], out["r"][1])
    else:
        sel = lambda out: out[wrt]
    if quantity == "ln_Psi":
        if with_geminal_inv:
            fn = lambda ham, r_up, r_dn, ginv: sel(  # noqa: E731
                compute_grad_ln_Psi(
                    ham,
                    r_up,
                    r_dn,
                    flags,
                    analytic=analytic,
                    fast=fast,
                    geminal_inv=ginv,
                    source_mode=source_mode,
                )
            )
            return _walker_batched(fn)
        fn = lambda ham, r_up, r_dn: sel(  # noqa: E731
            compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=analytic, fast=fast, source_mode=source_mode)
        )
        return _walker_batched(fn)
    if with_geminal_inv:
        fn = lambda ham, r_up, r_dn, RT, ginv: sel(  # noqa: E731
            compute_grad_e_L(
                ham,
                r_up,
                r_dn,
                RT,
                flags,
                analytic=analytic,
                fast=fast,
                geminal_inv=ginv,
                source_mode=source_mode,
            )
        )
        return _walker_batched(fn)
    fn = lambda ham, r_up, r_dn, RT: sel(  # noqa: E731
        compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=analytic, fast=fast, source_mode=source_mode)
    )
    return _walker_batched(fn)


# Compatibility single-axis inner-jit accessors for tests and profiling harnesses
# that reach for the RAW jitted object (e.g. ``bridge._jit_e_L_giv("c", False).lower(...)``).
# ``analytic`` defaults to True to preserve the raw accessor contract.
@lru_cache(maxsize=None)
def _jit_ln_psi(wrt: GradientAxes, fast: bool, analytic: bool = True) -> BatchedGradient:
    return _build("ln_Psi", wrt, fast, analytic, None, False)


@lru_cache(maxsize=None)
def _jit_e_L(wrt: GradientAxes, fast: bool, analytic: bool = True) -> BatchedGradient:
    return _build("e_L", wrt, fast, analytic, None, False)


@lru_cache(maxsize=None)
def _jit_ln_psi_giv(wrt: GradientAxes, fast: bool, analytic: bool = True) -> BatchedGradient:
    return _build("ln_Psi", wrt, fast, analytic, None, True)


@lru_cache(maxsize=None)
def _jit_e_L_giv(wrt: GradientAxes, fast: bool, analytic: bool = True) -> BatchedGradient:
    return _build("e_L", wrt, fast, analytic, None, True)


def cache_clear() -> None:
    """Drop the inner-jit LRU caches AND the underlying XLA compiled programs.

    The ``_build`` and compatibility ``_jit_*`` accessor LRU caches are unbounded:
    a single process that builds closures for MANY different hamiltonians accumulates
    compiled programs until LLVM mmap failures (measured). Production never needs this
    (one run builds a handful of closures for a single ham), but a multi-fixture test
    harness that walks many distinct hams MUST call this between fixtures. Clearing the
    LRU caches drops the jitted-wrapper references; ``jax.clear_caches()`` is the lever
    that actually frees the compiled executables held by JAX's own compilation cache."""
    _build.cache_clear()
    _jit_ln_psi.cache_clear()
    _jit_e_L.cache_clear()
    _jit_ln_psi_giv.cache_clear()
    _jit_e_L_giv.cache_clear()
    jax.clear_caches()


def batched_grad_ln_Psi(
    *,
    wrt: GradientAxes,
    mask: Mapping[str, bool] | None = None,
    fast: bool = False,
    with_geminal_inv: bool = False,
    analytic: bool | None = None,
) -> BatchedGradient:
    """Return a batched ``compute_grad_ln_Psi``; call as ``f(ham, r_up_batch, r_dn_batch)``.

    With ``with_geminal_inv=True``, the closure takes an extra per-walker
    ``geminal_inv`` batch -- ``f(ham, r_up, r_dn, geminal_inv)`` -- reusing the running SM
    inverse so NO fresh SVD-pinv lowers. ``wrt=("r","R")`` fuses drift + Pulay force into
    one trace. ``analytic=None`` reads the configured default at build time;
    ``True`` is strict analytic and ``False`` uses the autograd twin. ``mask``
    applies only to the parameter axis. The closure evaluates the original
    orbital basis without Cartesian conversion.
    """
    analytic = _resolve_analytic(analytic)
    jitted = _build("ln_Psi", wrt, fast, analytic, _mask_key(wrt, mask), with_geminal_inv)
    if with_geminal_inv:

        def f(
            ham: Hamiltonian_data,
            r_up: jnpt.ArrayLike,
            r_dn: jnpt.ArrayLike,
            geminal_inv: jnpt.ArrayLike,
        ) -> GradientResult:
            return jitted(ham, r_up, r_dn, geminal_inv)

        return f

    def f(
        ham: Hamiltonian_data,
        r_up: jnpt.ArrayLike,
        r_dn: jnpt.ArrayLike,
    ) -> GradientResult:
        return jitted(ham, r_up, r_dn)

    return f


def batched_grad_e_L(
    *,
    wrt: GradientAxes,
    mask: Mapping[str, bool] | None = None,
    fast: bool = False,
    with_geminal_inv: bool = False,
    analytic: bool | None = None,
) -> BatchedGradient:
    """Return a batched ``compute_grad_e_L``; call as ``f(ham, r_up_batch, r_dn_batch, RT_batch)``.

    With ``with_geminal_inv=True``, the closure takes an extra per-walker
    ``geminal_inv`` batch -- ``f(ham, r_up, r_dn, RT, geminal_inv)`` -- reusing the running SM
    inverse so NO fresh SVD-pinv lowers. ``wrt=("r","R")`` fuses drift + HF force into one
    trace. ``analytic=None`` reads the configured default at build time;
    ``True`` is strict analytic and ``False`` uses the autograd twin. ``mask``
    applies only to the parameter axis. The closure evaluates the original
    orbital basis without Cartesian conversion.
    """
    analytic = _resolve_analytic(analytic)
    jitted = _build("e_L", wrt, fast, analytic, _mask_key(wrt, mask), with_geminal_inv)
    if with_geminal_inv:

        def f(
            ham: Hamiltonian_data,
            r_up: jnpt.ArrayLike,
            r_dn: jnpt.ArrayLike,
            RT: jnpt.ArrayLike,
            geminal_inv: jnpt.ArrayLike,
        ) -> GradientResult:
            return jitted(ham, r_up, r_dn, RT, geminal_inv)

        return f

    def f(
        ham: Hamiltonian_data,
        r_up: jnpt.ArrayLike,
        r_dn: jnpt.ArrayLike,
        RT: jnpt.ArrayLike,
    ) -> GradientResult:
        return jitted(ham, r_up, r_dn, RT)

    return f


def _batched_c_source(
    quantity: GradientQuantity,
    *,
    mask: Mapping[str, bool] | None = None,
    with_geminal_inv: bool = False,
    analytic: bool | None = None,
) -> BatchedGradient:
    """Return a provider closure that retains exact complex c-axis sources."""
    analytic = _resolve_analytic(analytic)
    jitted = _build(quantity, "c", False, analytic, _mask_key("c", mask), with_geminal_inv, True)
    if quantity == "ln_Psi":
        if with_geminal_inv:
            return lambda ham, r_up, r_dn, geminal_inv: jitted(ham, r_up, r_dn, geminal_inv)
        return lambda ham, r_up, r_dn: jitted(ham, r_up, r_dn)
    if with_geminal_inv:
        return lambda ham, r_up, r_dn, RT, geminal_inv: jitted(ham, r_up, r_dn, RT, geminal_inv)
    return lambda ham, r_up, r_dn, RT: jitted(ham, r_up, r_dn, RT)


def _project_real_param_grads(
    wf: Wavefunction_data,
    flat: Mapping[str, np.ndarray],
    mask_kwargs: Mapping[str, bool],
) -> _FlatParameterGrads:
    """Project real-parameter leaves for the compatibility reducer."""
    blocks = wf.get_variational_blocks(
        state_token=("real-leaf-gradient-reducer", 0),
        **mask_kwargs,
    )
    real_names = {block.name for block in blocks if block.derivative_policy == "direct_real"}
    return {name: (np.real(values) if name in real_names else values) for name, values in flat.items()}


def _reduce_c_param_grads(
    wf: Wavefunction_data,
    grad_wf_batched: Wavefunction_data,
    num_walkers: int,
    mask_kwargs: Mapping[str, bool],
) -> _FlatParameterGrads:
    """Collapse a batched RAW ``Wavefunction_data`` grad (O_k or de_L/dc) to the
    flat block-name dict production stores.

    For legacy ``AGP_data`` (dense ``lambda_matrix`` field) the full
    ``collect_param_grads`` / ``flatten_param_grads`` path is exact. For the
    direct-MO geminals (``SD_data`` / ``AGPn_data``), this batched bridge must
    still assemble the geminal blocks directly.  Their ordinary
    ``collect_param_grads`` methods are defined for unbatched parameter
    cotangents; the exact derivative sources handled here carry a leading
    walker axis and require explicit source-shape handling.  We therefore
    collect only the Jastrow blocks via the WF pytree and build the geminal
    blocks directly:
    ``mo_coefficients_up`` (+ ``lambda_diag`` / ``lambda_unpaired`` for AGPn,
    ``mo_coefficients_dn`` for open shell) plus the GTO ``lambda_basis_*``
    blocks (up + dn concatenated, the same convention as
    ``Geminal_data.collect_param_grads``). The closed-shell singlet tie
    ``C_dn = conj(C_up)`` (``time_reversal=True``) is one variational block, so
    its grad is ``grad_up + conj(grad_dn)`` -- reconstructing exactly what the
    active-parameter closure differentiated. For
    ROHF/kROHF, ``grad_dn`` is first embedded in the paired leading rows of
    the larger up-spin block; excess up rows receive zero down contribution.
    """
    from ..determinant import (
        AGPn_data,
        SD_data,
        _embed_time_reversal_down_rows,
        _gto_leaves_of,
    )

    gem = wf.geminal_data
    if not isinstance(gem, (SD_data, AGPn_data)):
        flat = wf.flatten_param_grads(wf.collect_param_grads(grad_wf_batched), num_walkers)
        return _project_real_param_grads(wf, flat, mask_kwargs)

    ggem = grad_wf_batched.geminal_data
    raw: dict[str, object] = {}
    if wf.jastrow_data is not None and grad_wf_batched.jastrow_data is not None:
        raw.update(wf.jastrow_data.collect_param_grads(grad_wf_batched.jastrow_data))
    flat = wf.flatten_param_grads(raw, num_walkers)

    # Geminal AO basis blocks (GTO both spins), mirroring
    # ``Geminal_data.collect_param_grads`` but without the ``lambda_matrix``
    # property. Emitted only when a basis block is optimised (run()'s
    # downstream filter drops them otherwise, so this matches the stored set).
    if (mask_kwargs.get("opt_lambda_basis_exp", False) or mask_kwargs.get("opt_lambda_basis_coeff", False)) and (
        _gto_leaves_of(gem.orb_data_up_spin) and _gto_leaves_of(gem.orb_data_dn_spin)
    ):
        flat["lambda_basis_exp"] = np.concatenate([np.asarray(ggem.ao_exponents_up), np.asarray(ggem.ao_exponents_dn)], axis=-1)
        flat["lambda_basis_coeff"] = np.concatenate(
            [np.asarray(ggem.ao_coefficients_up), np.asarray(ggem.ao_coefficients_dn)], axis=-1
        )

    if mask_kwargs.get("opt_geminal", False):
        gu = np.asarray(ggem.orb_data_up_spin.mo_coefficients)
        if bool(gem.time_reversal):
            gd = _embed_time_reversal_down_rows(
                gu,
                np.asarray(ggem.orb_data_dn_spin.mo_coefficients),
                xp=np,
            )
            flat["mo_coefficients_up"] = gu + np.conj(gd)
        else:
            flat["mo_coefficients_up"] = gu
            flat["mo_coefficients_dn"] = np.asarray(ggem.orb_data_dn_spin.mo_coefficients)
        if isinstance(gem, AGPn_data):
            flat["lambda_diag"] = np.asarray(ggem.lambda_diag)
            if np.asarray(gem.lambda_unpaired).size > 0:
                flat["lambda_unpaired"] = np.asarray(ggem.lambda_unpaired)
    return _project_real_param_grads(wf, flat, mask_kwargs)


def _collect_c_param_derivative_sources(
    wf: Wavefunction_data,
    grad_wf_batched: Wavefunction_data,
    blocks: Sequence[VariationalParameterBlock],
    num_walkers: int,
    mask_kwargs: Mapping[str, bool],
    *,
    device: bool = False,
) -> tuple[object, ...]:
    """Build exact derivative sources from one raw batched wave-function grad.

    Unlike :func:`_reduce_c_param_grads`, this path never folds a conjugate-tied
    down-spin leaf into the up-spin leaf and never projects a real parameter's
    complex directional derivative.  The caller supplies the canonical block
    publication so source policy, shape, order, and name validation all remain
    mechanical.
    """
    from ..determinant import (
        AGPn_data,
        SD_data,
        _embed_time_reversal_down_rows,
        _gto_leaves_of,
    )
    from ..jqmc_optimizer import (
        BlockDerivativeSource,
        DeviceBlockDerivativeSource,
        validate_derivative_sources,
        validate_device_derivative_sources,
    )

    xp = jnp if device else np
    source_type = DeviceBlockDerivativeSource if device else BlockDerivativeSource

    published_blocks = tuple(blocks)
    gem = wf.geminal_data
    raw_by_name: dict[str, np.ndarray] = {}
    tied_mo: tuple[np.ndarray, np.ndarray] | None = None

    if isinstance(gem, (SD_data, AGPn_data)):
        ggem = grad_wf_batched.geminal_data
        raw: dict[str, object] = {}
        if wf.jastrow_data is not None and grad_wf_batched.jastrow_data is not None:
            raw.update(wf.jastrow_data.collect_param_grads(grad_wf_batched.jastrow_data))
        if device:
            raw_by_name.update({name: jnp.asarray(values) for name, values in raw.items()})
        else:
            raw_by_name.update(wf.flatten_param_grads(raw, num_walkers))

        if (mask_kwargs.get("opt_lambda_basis_exp", False) or mask_kwargs.get("opt_lambda_basis_coeff", False)) and (
            _gto_leaves_of(gem.orb_data_up_spin) and _gto_leaves_of(gem.orb_data_dn_spin)
        ):
            raw_by_name["lambda_basis_exp"] = xp.concatenate(
                [xp.asarray(ggem.ao_exponents_up), xp.asarray(ggem.ao_exponents_dn)], axis=-1
            )
            raw_by_name["lambda_basis_coeff"] = xp.concatenate(
                [xp.asarray(ggem.ao_coefficients_up), xp.asarray(ggem.ao_coefficients_dn)], axis=-1
            )

        if mask_kwargs.get("opt_geminal", False):
            gu = xp.asarray(ggem.orb_data_up_spin.mo_coefficients)
            gd = xp.asarray(ggem.orb_data_dn_spin.mo_coefficients)
            if bool(gem.time_reversal):
                tied_mo = (
                    gu,
                    _embed_time_reversal_down_rows(gu, gd, xp=xp),
                )
            else:
                raw_by_name["mo_coefficients_up"] = gu
                raw_by_name["mo_coefficients_dn"] = gd
            if isinstance(gem, AGPn_data):
                raw_by_name["lambda_diag"] = xp.asarray(ggem.lambda_diag)
                if np.asarray(gem.lambda_unpaired).size > 0:
                    raw_by_name["lambda_unpaired"] = xp.asarray(ggem.lambda_unpaired)
    else:
        raw = wf.collect_param_grads(grad_wf_batched)
        if device:
            raw_by_name.update({name: jnp.asarray(values) for name, values in raw.items()})
        else:
            raw_by_name.update(wf.flatten_param_grads(raw, num_walkers))

    sources = []
    for block in published_blocks:
        if block.name == "mo_coefficients_up" and tied_mo is not None:
            gu, gd = tied_mo
            if block.derivative_policy == "direct_real":
                source = source_type.build(block=block, direct=gu + gd)
            elif block.derivative_policy == "wirtinger_pair":
                source = source_type.build(block=block, d_p=gu, d_pbar=gd)
            else:
                raise ValueError("a time-reversal MO block must be direct_real or wirtinger_pair.")
        else:
            if block.name not in raw_by_name:
                raise ValueError(f"raw derivative payload is missing variational block {block.name!r}.")
            values = raw_by_name[block.name]
            if block.derivative_policy == "direct_real":
                source = source_type.build(block=block, direct=values)
            elif block.derivative_policy == "holomorphic":
                source = source_type.build(block=block, d_p=values)
            else:
                # A complex linear-tie block (the legacy dense AGP lambda)
                # is holomorphic in its full ambient storage.  Its conjugate
                # dependence is introduced by the separately published
                # real-linear tangent map, so the exact ambient B component
                # is structurally zero.
                source = source_type.build(block=block, d_p=values, d_pbar=xp.zeros_like(values))
        sources.append(source)
    if device:
        return validate_device_derivative_sources(published_blocks, sources)
    return validate_derivative_sources(published_blocks, sources)
