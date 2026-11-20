"""Autograd (``_auto``) twins of the production gradient layer.

These permanent test oracles operate on one ``(r_up, r_dn)`` configuration;
the bridge vectorizes them over walkers and exposes the production closure shape.
The active-parameter mask is applied IN-TRACE
(``with_param_grad_mask`` inside the differentiated function -- a ``stop_gradient``
applied eagerly to a concrete pytree is a silent no-op), energy derivatives are
``jnp.real``-wrapped (so HF force/stress = d(Re e_L)/d.), and O_k is the un-conjugated
``grad(ln|Psi|)``. Param grads are returned as the flat block-name dict via
``collect_param_grads`` (matching production); frozen blocks come back as zeros.

``mask`` is the dict of ``with_param_grad_mask`` flags (or ``None`` = all active).
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

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import typing as jnpt

from jqmc.determinant import AGPn_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy, compute_local_energy_fast
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

DerivativeAxis: TypeAlias = Literal["c", "r", "R", "eps"]
DerivativeValue: TypeAlias = Wavefunction_data | jax.Array | tuple[jax.Array, jax.Array]
_StrainedArgument: TypeAlias = Hamiltonian_data | jax.Array


# Hellmann-Feynman stress kernel.
# ``strain`` is a (3, 3) matrix; the deformation gradient is F = I + strain.
# F is applied to every Structure_data leaf in the Hamiltonian (cell + atomic
# positions via ``_apply_strain``) and to the electron coordinates using the
# row-vector convention ``r' = r @ F.T``. The ECP quadrature rotation ``RT`` is
# NOT strained: each strained quadrature point stays on the sphere of the
# strained electron-nucleus distance along a fixed direction, so the rotation
# average of the derivative is the exact strain derivative of the angular
# integral. Straining it as ``RT @ F.T`` moved the points off that sphere and
# biased the non-local ECP stress.
# At ``strain == 0`` the function returns the un-strained ``compute_local_energy``
# value to floating-point identity; ``jax.jacrev(..., argnums=4)`` then yields
# the per-walker sigma_HF tensor of shape ``(3, 3)``.
def _local_energy_strained(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    strain: jax.Array,
) -> jax.Array:
    F = jnp.eye(3, dtype=jnp.float64) + strain
    new_h = hamiltonian_data._apply_strain(F)
    new_r_up = r_up_carts @ F.T
    new_r_dn = r_dn_carts @ F.T
    return compute_local_energy(new_h, new_r_up, new_r_dn, RT)


# The physical local energy is real, so complex Bloch values are projected to
# their real component before differentiating with respect to strain.
def _real_local_energy_strained(
    *args: _StrainedArgument,
    **kwargs: _StrainedArgument,
) -> jax.Array:
    return jnp.real(_local_energy_strained(*args, **kwargs))


_jit_vmap_local_stress_HF = jax.jit(jax.vmap(jax.jacrev(_real_local_energy_strained, argnums=4), in_axes=(None, 0, 0, 0, None)))


# Pulay stress kernel (dln|Psi|/deps via jacrev).
# Mirrors :py:func:`_local_energy_strained` but operates only on the
# wavefunction tree (no Hamiltonian / coulomb / Ewald path; ``log|Psi|``
# does not depend on RT). The strain ``F = I + eps`` is applied to the
# wavefunction's structure_data leaves and to the electron coordinates;
# ``jax.jacrev`` then yields the per-walker sigma_PP tensor of shape ``(3, 3)``.
def _local_log_psi_strained(
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    strain: jax.Array,
) -> jax.Array:
    F = jnp.eye(3, dtype=jnp.float64) + strain
    new_wf = wavefunction_data._apply_strain(F)
    new_r_up = r_up_carts @ F.T
    new_r_dn = r_dn_carts @ F.T
    return evaluate_ln_wavefunction(new_wf, new_r_up, new_r_dn)


_jit_vmap_local_stress_PP = jax.jit(jax.vmap(jax.jacrev(_local_log_psi_strained, argnums=3), in_axes=(None, 0, 0, None)))


def _flags(mask: dict[str, bool] | None) -> dict[str, bool]:
    if isinstance(mask, dict):
        return mask
    return getattr(mask, "flags", None) or {}


def _PENDING(wrt: str) -> NotImplementedError:
    return NotImplementedError(f"_auto does not implement the requested derivative axis wrt={wrt!r}.")


def compute_grad_ln_Psi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    *,
    wrt: DerivativeAxis,
    mask: dict[str, bool] | None = None,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> DerivativeValue:
    wf = hamiltonian_data.wavefunction_data
    flags = _flags(mask)
    if wrt == "c":  # O_k = d ln|Psi| / d c (un-conjugated). RAW Wavefunction_data-shaped
        # grad (jit-able); the host-side ``collect_param_grads`` -> flat block dict is
        # applied by the scaffold/driver, exactly as production keeps it outside jit.
        def _ln(w: Wavefunction_data) -> jax.Array:
            wm = w.with_param_grad_mask(**flags) if flags else w
            return evaluate_ln_wavefunction(wm, r_up, r_dn)

        if not source_mode:
            return jax.grad(lambda w: jnp.real(_ln(w)))(wf)
        grad_real = jax.grad(lambda w: jnp.real(_ln(w)))(wf)
        grad_imag = jax.grad(lambda w: jnp.imag(_ln(w)))(wf)

        def _source(g_real: jax.Array, g_imag: jax.Array, parameter: jax.Array) -> jax.Array:
            return g_real if jnp.iscomplexobj(parameter) else g_real + 1.0j * g_imag

        return jax.tree_util.tree_map(_source, grad_real, grad_imag, wf)
    if wrt == "R":  # Pulay d ln|Psi| / d R  (per-atom, via the structure-position grad)
        grad_wf = jax.grad(lambda w: jnp.real(evaluate_ln_wavefunction(w, r_up, r_dn)))(wf)
        return wf.accumulate_position_grad(grad_wf)
    if wrt == "eps":  # Pulay stress d ln|Psi| / d eps (periodic); reuse the strained fwd (this module)
        return jax.jacrev(_local_log_psi_strained, argnums=3)(wf, r_up, r_dn, jnp.zeros((3, 3)))
    if wrt == "r":  # drift d ln|Psi| / d r_up, d r_dn
        return jax.grad(lambda a, b: jnp.real(evaluate_ln_wavefunction(wf, a, b)), argnums=(0, 1))(r_up, r_dn)
    raise _PENDING(wrt)


def compute_grad_e_L(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    *,
    wrt: DerivativeAxis,
    mask: dict[str, bool] | None = None,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> DerivativeValue:
    flags = _flags(mask)
    _lenergy = compute_local_energy

    # A supplied inverse selects the production-fast contract.  The oracle
    # differentiates that exact mixed-precision formula, but rebuilds a fresh
    # inverse from the value-only geminal so AD sees d(G^-1) rather than
    # treating the caller's running state as a constant.  This is test/debug
    # code only; the production analytical route continues to consume the
    # already-maintained Sherman--Morrison inverse without a factorization.
    def _fast_energy(h: Hamiltonian_data, a: jax.Array, b: jax.Array, rotation: jax.Array) -> jax.Array:
        G = compute_geminal_all_elements(h.wavefunction_data.geminal_data, a, b)
        return compute_local_energy_fast(h, a, b, rotation, jnp.linalg.inv(G))

    if wrt == "c":  # d e_L / d c (linear method). RAW Wavefunction_data-shaped grad
        # (jit-able); host-side ``collect_param_grads`` is applied by the scaffold/driver.
        def _el(h: Hamiltonian_data) -> jax.Array:
            wm = h.wavefunction_data.with_param_grad_mask(**flags) if flags else h.wavefunction_data
            hm = h.replace(wavefunction_data=wm)
            if geminal_inv is not None:
                return _fast_energy(hm, r_up, r_dn, RT)
            return _lenergy(hm, r_up, r_dn, RT)

        grad_real = jax.grad(lambda h: jnp.real(_el(h)))(hamiltonian_data).wavefunction_data
        gd = hamiltonian_data.wavefunction_data.geminal_data
        factorized_lambda = (
            isinstance(gd, AGPn_data)
            and gd.lambda_unpaired.size == 0
            and int(gd.orb_num_up) == int(r_up.shape[0])
            and int(gd.orb_num_dn) == int(r_dn.shape[0])
        )
        if factorized_lambda:
            # det(O_up.T diag(lambda) O_dn) factorizes into
            # det(O_up.T) prod(lambda) det(O_dn).  The local energy is exactly
            # normalization-independent, so its lambda derivative is zero.
            # Raw mixed AD of the hybrid fast evaluator leaves round-off here
            # because its value-only inverse and fused spatial tables have
            # different rounding boundaries; use the exact factorized oracle.
            grad_real = grad_real.replace(
                geminal_data=grad_real.geminal_data.replace(lambda_diag=jnp.zeros_like(grad_real.geminal_data.lambda_diag))
            )
        if not source_mode:
            return grad_real
        grad_imag = jax.grad(lambda h: jnp.imag(_el(h)))(hamiltonian_data).wavefunction_data
        if factorized_lambda:
            grad_imag = grad_imag.replace(
                geminal_data=grad_imag.geminal_data.replace(lambda_diag=jnp.zeros_like(grad_imag.geminal_data.lambda_diag))
            )
        wf_params = hamiltonian_data.wavefunction_data

        def _complex_derivative(g_real: jax.Array, g_imag: jax.Array, parameter: jax.Array) -> jax.Array:
            # A real leaf owns one physical direction, so source mode retains its complete
            # complex derivative.  A complex storage leaf follows the existing
            # holomorphic-source convention: differentiating the real part
            # already returns the unconjugated holomorphic derivative of the
            # complex VMC energy, including its unprojected ECP contribution.
            if jnp.iscomplexobj(parameter):
                return g_real
            return g_real + 1.0j * g_imag

        return jax.tree_util.tree_map(_complex_derivative, grad_real, grad_imag, wf_params)
    if wrt == "R":  # Hellmann-Feynman d e_L / d R  (per-atom)
        energy = _fast_energy if geminal_inv is not None else _lenergy
        grad_ham = jax.grad(lambda h: jnp.real(energy(h, r_up, r_dn, RT)))(hamiltonian_data)
        return hamiltonian_data.accumulate_position_grad(grad_ham)
    if wrt == "eps":  # HF stress d e_L / d eps (periodic); reuse the production strained fwd
        if geminal_inv is not None:

            def _fast_strained(strain: jax.Array) -> jax.Array:
                F = jnp.eye(3, dtype=strain.dtype) + strain
                new_h = hamiltonian_data._apply_strain(F)
                new_r_up = r_up @ F.T
                new_r_dn = r_dn @ F.T
                # Quadrature directions fixed under the strain.
                return jnp.real(_fast_energy(new_h, new_r_up, new_r_dn, RT))

            return jax.jacrev(_fast_strained)(jnp.zeros((3, 3)))
        return jax.jacrev(_real_local_energy_strained, argnums=4)(hamiltonian_data, r_up, r_dn, RT, jnp.zeros((3, 3)))
    if wrt == "r":  # d e_L / d r_up, d r_dn (SWCT term)
        if geminal_inv is not None:
            return jax.grad(
                lambda a, b: jnp.real(_fast_energy(hamiltonian_data, a, b, RT)),
                argnums=(0, 1),
            )(r_up, r_dn)
        return jax.grad(lambda a, b: jnp.real(_lenergy(hamiltonian_data, a, b, RT)), argnums=(0, 1))(r_up, r_dn)
    raise _PENDING(wrt)
