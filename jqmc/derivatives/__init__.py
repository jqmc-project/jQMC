"""Analytic gradients of ``ln|Psi|`` and ``e_L`` (autograd-free, Rust-portable).

Public API::

    compute_grad_ln_Psi(ham, r_up, r_dn, flags, *, analytic: bool)
    compute_grad_e_L  (ham, r_up, r_dn, RT, flags, *, analytic: bool)

``flags`` is ONE dict of ``opt_flags`` -- the SAME optimization flags production
already carries -- describing every differentiation target: the parameter blocks
(``opt_J1_param`` ... ``opt_JDTN_f_param``) AND the physical axes ``opt_r`` /
``opt_R`` / ``opt_eps``. The caller passes only ``opt_flags``; the mask /
``stop_gradient`` that prunes the frozen branches is derived internally. ``c``
and ``r``/``R``/``eps`` are the same "what do we
differentiate w.r.t." concept, so there is no separate selector argument.

There is NO ``opt_c``: the parameter-block flags ARE the c-axis. The c-axis is
computed iff at least one parameter block is optimized (its flag is ``True``);
the block flags then double as the c-axis mask. A block whose flag is ``False``
or absent is frozen (pruned) -- ``flags`` is a whitelist of the branches to
differentiate, exactly like the ``opt_flags`` an optimizer sets. Both the
analytic and ``_auto`` paths prune the same branches.

The result is a ``dict`` keyed by the requested axes (``"c"``/``"r"``/``"R"``/
``"eps"``), each holding that axis' gradient.

``analytic: bool``:
  * ``True``  -> the assembled analytic path. A flagged axis/block the analytic
    path does not implement raises ``NotImplementedError`` (NO silent ``_auto``
    fallback -- the analytic path is the product; ``_auto`` is a test oracle).
  * ``False`` -> the ``_auto`` autograd twin used as the independent reference
    by equivalence tests and debug drivers.

mask / ``stop_gradient`` is encapsulated here: callers pass flags, never
``with_param_grad_mask`` / ``stop_gradient`` directly.

New analytic components register their supported feature in ``_capability``;
this public dispatch remains independent of component-specific implementations.
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

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Literal, TypeAlias

import jax
from jax import typing as jnpt

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.wavefunction import Wavefunction_data

from . import _capability
from . import _auto

# The analytic implementations are MANDATORY, and their import errors must propagate.
#
# These imports must not be wrapped in
# ``except Exception: _assemble = None`` / ``except Exception: pass`` (which would
# silently fall back to ``_auto`` when an implementation is incomplete). The
# analytic modules are mandatory, so such a guard actively hides regressions: a
# NameError/syntax error anywhere in the analytic tree would silently un-register the
# capabilities and degrade EVERY route to ``_auto`` (autograd), which hollows the
# analytic-vs-``_auto`` parity tests into a tautological ``_auto`` vs ``_auto`` compare --
# and the Rust port's reference IS the analytic implementation.
#
# The analytic Jastrow derivatives live in ``assemble`` / ``kinetic`` / ``dtn`` /
# ``basis``; there is no ``jqmc.derivatives.jastrow`` module.
from . import assemble as _assemble  # Analytic geminal/Jastrow/AO/Coulomb assembly.
from . import ao as _ao  # noqa: F401  -- self-registers capabilities on import
from . import geminal as _geminal  # noqa: F401
from . import fast as _fast  # noqa: F401

DerivativeValue: TypeAlias = Wavefunction_data | jax.Array | tuple[jax.Array, jax.Array]
DerivativeQuantity: TypeAlias = Literal["ln_Psi", "e_L"]
DerivativeAxis: TypeAlias = Literal["c", "r", "R", "eps"]
DerivativeResult: TypeAlias = dict[str, DerivativeValue]


# --- flag vocabulary -------------------------------------------------------
# Parameter-block flags == the ``with_param_grad_mask`` keyword set; they
# configure the c-axis mask (which blocks propagate a gradient).
_PARAM_FLAG_KEYS = frozenset(
    {
        "opt_J1_param",
        "opt_J2_param",
        "opt_J3_param",
        "opt_geminal",
        "opt_J3_basis_exp",
        "opt_J3_basis_coeff",
        "opt_lambda_basis_exp",
        "opt_lambda_basis_coeff",
        "opt_JDTN_u_param",
        "opt_JDTN_chi_param",
        "opt_JDTN_f_param",
    }
)
# Physical-axis switches (the c-axis has NO switch -- it is implied by the
# parameter-block flags).
_AXIS_TO_WRT = {"opt_r": "r", "opt_R": "R", "opt_eps": "eps"}
_AXIS_FLAG_KEYS = frozenset(_AXIS_TO_WRT)


def _split_flags(flags: dict) -> tuple[dict, list[str]]:
    """Split a unified ``opt_flags`` dict into ``(param_mask, axes)``.

    ``param_mask`` is the COMPLETE ``with_param_grad_mask`` dict: every parameter
    block gets an explicit bool (``flags`` is a whitelist -- a block absent or
    ``False`` is frozen/pruned). ``axes`` is the ordered list of requested
    internal axis tokens drawn from ``("c", "r", "R", "eps")``: ``c`` iff any
    parameter block is optimized (its flag is ``True``), plus each physical axis
    whose ``opt_*`` flag is truthy.
    """
    unknown = set(flags) - _PARAM_FLAG_KEYS - _AXIS_FLAG_KEYS
    if unknown:
        raise ValueError(
            f"unknown gradient flags {sorted(unknown)}; allowed = parameter blocks "
            f"{sorted(_PARAM_FLAG_KEYS)} + axes {sorted(_AXIS_FLAG_KEYS)}"
        )
    param_mask = {k: bool(flags.get(k, False)) for k in _PARAM_FLAG_KEYS}
    axes: list[str] = []
    if any(param_mask.values()):  # c-axis iff >=1 parameter block is optimized
        axes.append("c")
    for opt_key, wrt in (("opt_r", "r"), ("opt_R", "R"), ("opt_eps", "eps")):
        if flags.get(opt_key):
            axes.append(wrt)
    return param_mask, axes


# --------------------------------------------------------------------------- #
# Run the selected analytic numeric kernel under ``jax.jit`` for
# EAGER callers (the analytic-vs-_auto harness + GFMC oracle tests). Eager, each
# ``jnp`` op in ``_assemble.compute_grad_*`` compiles as its own tiny XLA program
# (~1841 for N_ae jnone) -- the ROOT CAUSE of the "XLA-accumulation SIGABRT" that
# crashes the shared xdist suite. Wrapping the SAME call in one outer ``jax.jit``
# collapses it to single-digit compiles. PRODUCTION already jit(vmap)s these via
# the ``bridge``; only the eager public-entry path floods.
#
# The DISPATCH plumbing stays OUTSIDE jit / at Python level (``jit_ao_serveable``
# AO-type branching, ``_split_flags``, the per-axis loop --
# they read ham STRUCTURE and raise NotImplementedError, so they cannot be
# traced). The jit boundary is AROUND the numeric ``_assemble`` kernel call only.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _jitted_assemble(
    quantity: DerivativeQuantity,
    wrt: DerivativeAxis,
    fast: bool,
    mask_key: frozenset[str] | None,
    source_mode: bool = False,
) -> Callable[..., DerivativeValue]:
    """Cached ``jax.jit`` wrapper around the selected ``_assemble`` kernel.

    All STATIC (graph-shaping) inputs are closure-captured via the lru_cache key so
    repeated eager calls REUSE the compiled function (second call = 0 compiles):

      * ``quantity`` ("ln_Psi"/"e_L"), ``wrt`` axis, ``fast`` flag;
      * ``mask_key`` -- the param mask made hashable as a ``frozenset`` of the TRUE
        block keys (reconstructed to the full ``with_param_grad_mask`` dict here).

    TRACED (per-config) inputs are jit arguments: the ham pytree (its static leaves
    -- AO types/shapes -- live in the treedef), ``r_up``/``r_dn``/``RT`` and
    ``geminal_inv`` (the running SM inverse; ``None`` = SVD-pinv)."""
    mask = None if mask_key is None else {k: (k in mask_key) for k in _PARAM_FLAG_KEYS}
    if quantity == "ln_Psi":

        def _ln_psi(
            hamiltonian_data: Hamiltonian_data,
            r_up: jax.Array,
            r_dn: jax.Array,
            geminal_inv: jax.Array | None,
        ) -> DerivativeValue:
            kwargs = {"wrt": wrt, "mask": mask, "fast": fast, "geminal_inv": geminal_inv}
            if source_mode:
                kwargs["source_mode"] = True
            return _assemble.compute_grad_ln_Psi(hamiltonian_data, r_up, r_dn, **kwargs)

        return jax.jit(_ln_psi)

    def _e_L(
        hamiltonian_data: Hamiltonian_data,
        r_up: jax.Array,
        r_dn: jax.Array,
        RT: jax.Array,
        geminal_inv: jax.Array | None,
    ) -> DerivativeValue:
        kwargs = {"wrt": wrt, "mask": mask, "fast": fast, "geminal_inv": geminal_inv}
        if source_mode:
            kwargs["source_mode"] = True
        return _assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, **kwargs)

    return jax.jit(_e_L)


def _analytic_axis(
    quantity: DerivativeQuantity,
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike | None,
    wrt: DerivativeAxis,
    param_mask: Mapping[str, bool],
    fast: bool,
    geminal_inv: jnpt.ArrayLike | None,
    source_mode: bool = False,
) -> DerivativeValue:
    """Compute ONE analytic axis gradient, routing the numeric ``_assemble`` kernel
    through ``jax.jit`` when it is jit-safe (the big compile-flood win), else eager.

    Eager evaluation is used for any AO leaf not in the JIT-safe
    set (``jit_ao_serveable``): exotic PW/bspline analytic kernels are eager-only.
    (DTN is jit-safe: ``derivatives/dtn.py`` folds its electron/atom/pair loops into
    vectorized ``jnp``/``vmap``.)
    """
    # ``_assemble`` is imported unconditionally, so the dispatch needs no
    # ``_assemble is None`` case; there is no fall back to ``_auto`` here.
    eager = not _capability.jit_ao_serveable(hamiltonian_data)
    if eager:
        if quantity == "ln_Psi":
            kwargs = {"wrt": wrt, "mask": param_mask, "fast": fast, "geminal_inv": geminal_inv}
            if source_mode:
                kwargs["source_mode"] = True
            return _assemble.compute_grad_ln_Psi(hamiltonian_data, r_up, r_dn, **kwargs)
        kwargs = {"wrt": wrt, "mask": param_mask, "fast": fast, "geminal_inv": geminal_inv}
        if source_mode:
            kwargs["source_mode"] = True
        return _assemble.compute_grad_e_L(hamiltonian_data, r_up, r_dn, RT, **kwargs)
    # NO sphe->Cartesian hoist here (unlike the batched production ``bridge``). Measured on every 8-set
    # AO type (molecular cart/sphe-GTO, periodic sphe-pgto, complex-twist sphe-pgto-complex): the
    # analytic ln_Psi/e_L r/R/eps kernels jit-COMPILE and match ``_auto`` on the ORIGINAL basis with NO
    # cartesianize -- the c2s conversion the hoist guarded against does not raise here (the geminal-MO
    # c2s reads STATIC angular/shape data, not the traced exponents). Complex pGTO basis conversion
    # preserves the Bloch subtype, twist, and AO/MO value/gradient/Laplacian, but the complete r/R/eps
    # closure has not been qualified on a Cartesianized complex Hamiltonian. Passing the untouched ham is
    # the established production contract and avoids an unnecessary representation change; the c-axis
    # never needed the hoist either.
    mask_key = frozenset(k for k, v in param_mask.items() if v)
    if quantity == "ln_Psi":
        return _jitted_assemble("ln_Psi", wrt, fast, mask_key, source_mode)(hamiltonian_data, r_up, r_dn, geminal_inv)
    return _jitted_assemble("e_L", wrt, fast, mask_key, source_mode)(hamiltonian_data, r_up, r_dn, RT, geminal_inv)


def _grad_bundled(
    quantity: DerivativeQuantity,
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike | None,
    flags: Mapping[str, bool],
    analytic: bool,
    fast: bool,
    geminal_inv: jnpt.ArrayLike | None,
    source_mode: bool = False,
) -> DerivativeResult:
    """Run the flagged axes through the analytic (``analytic=True``) or ``_auto``
    (``analytic=False``) per-axis kernels and bundle the results by axis."""
    param_mask, axes = _split_flags(flags)
    out: dict[str, object] = {}
    for wrt in axes:
        if analytic:
            # NO capability PRE-CHECK gate: the analytic path is
            # the product, so it is ALWAYS attempted. A genuinely-unimplemented axis/leaf surfaces as a
            # natural NotImplementedError raised by the deep kernels (ao_deriv.py etc.) or the
            # mask-conditional raise in assemble.py (masked-but-unsupported J3 basis-opt) -- never a
            # silent _auto fallback, and never a redundant registry gate that can under-claim an
            # implemented axis. The numeric kernel runs under jax.jit (single-digit compiles instead of
            # thousands) -- see _analytic_axis.
            out[wrt] = _analytic_axis(
                quantity,
                hamiltonian_data,
                r_up,
                r_dn,
                RT,
                wrt,
                param_mask,
                fast,
                geminal_inv,
                source_mode,
            )
            continue
        mod = _auto
        if quantity == "ln_Psi":
            out[wrt] = mod.compute_grad_ln_Psi(
                hamiltonian_data,
                r_up,
                r_dn,
                wrt=wrt,
                mask=param_mask,
                fast=fast,
                geminal_inv=geminal_inv,
                source_mode=source_mode,
            )
        else:
            out[wrt] = mod.compute_grad_e_L(
                hamiltonian_data,
                r_up,
                r_dn,
                RT,
                wrt=wrt,
                mask=param_mask,
                fast=fast,
                geminal_inv=geminal_inv,
                source_mode=source_mode,
            )
    return out


def compute_grad_ln_Psi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    flags: Mapping[str, bool],
    *,
    analytic: bool,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> dict[str, DerivativeValue]:
    """Gradient(s) of ``ln|Psi|``. See the module docstring for the flag interface.

    ``compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=bool)`` -> ``dict``
    keyed by requested axis.
    """
    return _grad_bundled("ln_Psi", hamiltonian_data, r_up, r_dn, None, flags, bool(analytic), fast, geminal_inv, source_mode)


def compute_grad_e_L(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    flags: Mapping[str, bool],
    *,
    analytic: bool,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> dict[str, DerivativeValue]:
    """Gradient(s) of ``e_L``. See the module docstring for the flag interface.

    ``compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=bool)`` -> ``dict``
    keyed by requested axis.
    """
    return _grad_bundled("e_L", hamiltonian_data, r_up, r_dn, RT, flags, bool(analytic), fast, geminal_inv, source_mode)
