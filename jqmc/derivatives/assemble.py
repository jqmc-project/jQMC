"""Assemble analytical wavefunction and local-energy derivatives.

The module combines geminal, Jastrow, radial-basis, force, and stress pieces
into the pytrees returned by ``compute_grad_ln_Psi`` and
``compute_grad_e_L``. Capability routing happens before these strict
analytical entry points; a requested radial parameter carried by an
unsupported basis raises instead of silently producing a zero derivative.
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

import math
from collections.abc import Mapping
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import typing as jnpt

from jqmc.atomic_orbital import AOs_base_data, compute_AOs, compute_AOs_grad
from jqmc.derivatives.fast import geminal_value_tables, resolve_geminal_inv
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data

GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
DerivativeAxis: TypeAlias = Literal["c", "r", "R", "eps"]
DerivativeValue: TypeAlias = Wavefunction_data | jax.Array | tuple[jax.Array, jax.Array]
OrbitalData: TypeAlias = AOs_base_data | MOs_data
RadialGradient: TypeAlias = tuple[jax.Array, jax.Array]


# Geminal parameter derivatives of ln|det G|.
def grad_ln_det_wrt_lambda(
    geminal_data: GeminalData,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    """d ln|det G| / d lambda_matrix (AGP); shape == lambda_matrix.shape."""
    gd = geminal_data
    M_dn = gd.orb_num_dn
    N_dn = gd.num_electron_dn
    A = gd.compute_orb_api(gd.orb_data_up_spin, r_up_carts).T  # (N_up, M_up)
    B = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn_carts)  # (M_dn, N_dn)
    lam = gd._lambda_matrix_jnp
    lam_paired, lam_unpaired = jnp.hsplit(lam, [M_dn])
    G = jnp.hstack([A @ lam_paired @ B, A @ lam_unpaired])  # (N_up, N_up)
    # Reuse the running Sherman-Morrison inverse supplied by MCMC/GFMC so the
    # optimization graph does not compute a fresh SVD. Without one, use the
    # same thresholded SVD-pinv as a freshly evaluated forward path.
    P = resolve_geminal_inv(G, geminal_inv) @ A  # (N_up, M_up)
    grad_paired = P[:N_dn].T @ B.T  # (M_up, M_dn)
    grad_unpaired = P[N_dn:].T  # (M_up, N_up - N_dn)
    return jnp.hstack([grad_paired, grad_unpaired])


def grad_ln_det_wrt_mo_coeff(
    geminal_data: GeminalData,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> tuple[jax.Array, jax.Array]:
    """(d ln|det G| / d C_up, d/d C_dn) for SD / AGPn-MO; each shaped like its leaf."""
    gd = geminal_data
    M_dn = gd.orb_num_dn
    N_dn = gd.num_electron_dn
    orb_up = gd.compute_orb_api(gd.orb_data_up_spin, r_up_carts)  # (m_up, N_up)
    orb_dn = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn_carts)  # (m_dn, N_dn)
    A_up = compute_AOs(gd.orb_data_up_spin.aos_data, r_up_carts)  # (num_ao_up, N_up)
    A_dn = compute_AOs(gd.orb_data_dn_spin.aos_data, r_dn_carts)  # (num_ao_dn, N_dn)
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        G_inv = resolve_geminal_inv(G, geminal_inv)
        grad_orb_up = jnp.concatenate(
            [orb_dn @ G_inv[:N_dn, :], G_inv[N_dn:, :]],
            axis=0,
        )
        grad_C_up = grad_orb_up @ A_up.T
        Q = orb_up[:N_dn].T
        grad_orb_dn = (G_inv @ Q)[:N_dn].T
        grad_C_dn = grad_orb_dn @ A_dn.T
        return grad_C_up, grad_C_dn

    lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
    G = jnp.hstack([orb_up.T @ lam_p @ orb_dn, orb_up.T @ lam_unp])  # (N_up, N_up)
    W = jnp.hstack([lam_p @ orb_dn, lam_unp])  # (m_up, N_up)
    G_inv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else SVD-pinv
    grad_orb_up = W @ G_inv  # W @ G^{-1}
    grad_C_up = grad_orb_up @ A_up.T  # (m_up, num_ao_up)
    Q = orb_up.T @ lam_p  # (N_up, m_dn)
    grad_orb_dn = (G_inv @ Q)[:N_dn].T  # (m_dn, N_dn)
    grad_C_dn = grad_orb_dn @ A_dn.T  # (m_dn, num_ao_dn)
    return grad_C_up, grad_C_dn


# Jastrow parameter derivatives of J.
def _pair_distances(sd: Structure_data, disp: jax.Array) -> jax.Array:
    """||disp|| per last axis. PBC -> Williamson smooth periodic distance (the same
    kernel the J1/J2 value evaluators use); molecular -> Euclidean norm. ``disp`` has
    shape (..., 3); returns shape (...). pbc_flag is static (pytree_node=False)."""
    if sd.pbc_flag:
        from jqmc.jastrow_factor import _smooth_periodic_distance
        from jqmc.structure import _inv_3x3

        cell = jnp.asarray(sd._cell_jnp, jnp.float64)
        inv_cell = _inv_3x3(cell)
        flat = disp.reshape(-1, 3)
        d = jax.vmap(lambda x: _smooth_periodic_distance(x, cell, inv_cell))(flat)
        return d.reshape(disp.shape[:-1])
    return jnp.linalg.norm(disp, axis=-1)


# Taylor coefficients of q(x) = 1 - (1 + x) exp(-x) = sum_{n>=2} (-1)^n (n-1) x^n / n!, n = 2..13.
_Q_TAYLOR_COEFFS = tuple((-1.0) ** n * (n - 1) / math.factorial(n) for n in range(2, 14))
_Q_TAYLOR_SWITCH = 0.25


def _one_minus_one_plus_x_exp_neg_x(x: jax.Array) -> jax.Array:
    """``q(x) = 1 - (1 + x) exp(-x)`` without cancellation at small ``|x|``.

    ``q`` is O(x^2) while both of its terms are O(x): the subtraction form
    loses every digit at ``x ~ 2**-26`` in fp64, and even the ``expm1`` form
    ``-expm1(-x) - x exp(-x)`` keeps a relative error ~eps/x (the rounding of
    ``exp(-x)`` near 1). Below ``|x| = 1/4`` the Taylor series through
    ``x**13`` is used (first omitted term < 3e-17 relative); above it
    ``(1 - e) - x e`` with ``e = exp(-x)`` has no significant cancellation
    (``q(1/4) = 0.026``, a few 1e-15 relative). Double-``where`` keeps both
    branches finite for reverse mode.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < _Q_TAYLOR_SWITCH
    xs = jnp.where(small, x, 0.0)
    poly = jnp.full_like(xs, _Q_TAYLOR_COEFFS[-1])
    for coeff in reversed(_Q_TAYLOR_COEFFS[:-1]):
        poly = poly * xs + coeff
    e = jnp.exp(-x)
    return jnp.where(small, xs * xs * poly, (1.0 - e) - x * e)


def _exp_jastrow_param_kernel(p: jax.Array, x: jax.Array) -> jax.Array:
    """``d/dp`` of the exp-form pair kernel ``(1 - exp(-x)) / (2 p)`` at fixed ``x / p``.

    Equals ``-(1 - exp(-x)) / (2 p^2) + x exp(-x) / (2 p^2) = -q(x) / (2 p^2)``
    with ``x = p d`` (J2) or ``x = p c d`` (J1, times its prefactor), evaluated
    through :func:`_one_minus_one_plus_x_exp_neg_x` so the O(x^2) result keeps
    full relative precision at small separations. Shared by
    :func:`grad_J1_wrt_param`, :func:`grad_J2_wrt_param` and the one-move mesh
    ``one_move_mesh.dV_ecp_dc_fast`` J2 block.
    """
    return -_one_minus_one_plus_x_exp_neg_x(x) / (2.0 * p**2)


def grad_J1_wrt_param(
    j1_data: Jastrow_one_body_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> jax.Array:
    sd = j1_data.structure_data
    R = jnp.asarray(sd.positions)
    Z = jnp.asarray(sd.atomic_numbers) - jnp.asarray(j1_data.core_electrons)
    p = j1_data.jastrow_1b_param
    c = (2.0 * Z) ** 0.25
    pref = -((2.0 * Z) ** 0.75) if j1_data.jastrow_1b_type == "exp" else -2.0 * Z
    r_all = jnp.concatenate([jnp.asarray(r_up_carts), jnp.asarray(r_dn_carts)], axis=0)
    d = _pair_distances(sd, r_all[:, None, :] - R[None, :, :])
    if j1_data.jastrow_1b_type == "exp":
        dterm = pref * _exp_jastrow_param_kernel(p, p * c * d)
    elif j1_data.jastrow_1b_type == "pade":
        dterm = -pref * c * d**2 / (2.0 * (1.0 + p * c * d) ** 2)
    else:
        raise ValueError(j1_data.jastrow_1b_type)
    return jnp.sum(dterm)


def grad_J2_wrt_param(
    j2_data: Jastrow_two_body_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> jax.Array:
    p = j2_data.jastrow_2b_param
    sd = j2_data.structure_data
    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)
    anti = _pair_distances(sd, r_up[:, None, :] - r_dn[None, :, :]).ravel()
    parts = [anti]
    for r in (r_up, r_dn):
        i, j = jnp.triu_indices(len(r), k=1)
        parts.append(_pair_distances(sd, r[i] - r[j]))
    d = jnp.concatenate(parts)
    if j2_data.jastrow_2b_type == "pade":
        dterm = -(d**2) / (2.0 * (1.0 + p * d) ** 2)
    elif j2_data.jastrow_2b_type == "exp":
        dterm = _exp_jastrow_param_kernel(p, p * d)
    else:
        raise ValueError(j2_data.jastrow_2b_type)
    return jnp.sum(dterm)


def grad_J3_wrt_jmatrix(
    j3_data: Jastrow_three_body_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
) -> jax.Array:
    A_up = jnp.real(j3_data.compute_orb_api(j3_data.orb_data, r_up_carts))  # (orb_num, N_up)
    A_dn = jnp.real(j3_data.compute_orb_api(j3_data.orb_data, r_dn_carts))
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    K_up = jnp.tril(jnp.ones((n_up, n_up)), k=-1)
    K_dn = jnp.tril(jnp.ones((n_dn, n_dn)), k=-1)
    s_up = A_up.sum(axis=1)
    s_dn = A_dn.sum(axis=1)
    grad_M = A_up @ K_up.T @ A_up.T + A_dn @ K_dn.T @ A_dn.T + jnp.outer(s_up, s_dn)
    grad_j1col = s_up + s_dn
    return jnp.hstack([grad_M, grad_j1col[:, None]])


def _set_geminal_lambda_grad(
    g_gd: GeminalData,
    gd: GeminalData,
    full_lambda_grad: jax.Array,
    *,
    preserve_complex_diag: bool = False,
) -> GeminalData:
    """Write the analytic d/d(lambda_matrix) grad onto the geminal grad pytree's
    variational leaf, matching what ``collect_param_grads`` reads.

    On AGPn, ``lambda_matrix`` is the materialised
    ``hstack([diag(lambda_diag), lambda_unpaired])`` property.  Its paired
    cotangent must therefore be projected onto the stored real diagonal leaf;
    zero. SD optimizes only its occupied MO coefficient blocks and has no
    lambda storage, so no lambda cotangent is written. On bare AGP
    ``lambda_matrix`` is a stored leaf and is set directly.
    """
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        return g_gd
    elif "lambda_diag" in gd.__dataclass_fields__:
        paired = full_lambda_grad[:, : gd.orb_num_dn]
        return g_gd.replace(
            lambda_diag=jnp.diag(paired) if preserve_complex_diag else jnp.real(jnp.diag(paired)),
            lambda_unpaired=full_lambda_grad[:, gd.orb_num_dn :],
        )
    if "lambda_unpaired" in gd.__dataclass_fields__:
        return g_gd
    return g_gd.replace(lambda_matrix=full_lambda_grad)


def _set_geminal_mo_grad(
    g_gd: GeminalData,
    gd: GeminalData,
    gC_up: jax.Array,
    gC_dn: jax.Array,
) -> GeminalData:
    """Write the d/d(mo_coefficients) per-spin-channel grads onto the geminal grad
    pytree's MO leaves. For MO-backed geminals (SD / AGPn-MO) the production param-grad
    mask optimizes ``mo_coefficients_up`` / ``_dn`` directly (get_variational_blocks with
    opt_geminal=True), so the masked driver path reads these leaves. Harmless to the
    collect_param_grads (lambda_matrix) path, which ignores mo_coefficients.

    The two spin channels are stored on their OWN leaves (``gC_up`` -> up leaf,
    ``gC_dn`` -> dn leaf) -- the SAME raw two-leaf structure ``_auto`` produces (its
    ``jax.grad`` populates both ``orb_data_up_spin`` and ``orb_data_dn_spin``
    mo_coefficients). The time-reversal tie ``C_dn = conj(C_up)`` (both channels SHARE
    one variational block) is folded to the single ``mo_coefficients_up`` block LATER,
    at the OUTERMOST reduction (``derivatives.bridge._reduce_c_param_grads``: ``gu + conj(gd)``),
    exactly as it is for ``_auto``.

    Folding early here (``gC_up + conj(gC_dn)`` onto the up leaf) is incorrect:
    it is called once per intermediate config, and in the ECP non-local dV_nl/dc path a
    COMPLEX per-grid weight ``W_g`` multiplies the folded grad AFTER this call. Because
    ``W_g * conj(gC_dn) != conj(W_g * gC_dn)`` the early fold drops the correct Im on a
    complex-twist geminal. Deferring the
    conj-fold to the linear (weight-multiply then accumulate) reduction makes the
    channels compose correctly under the complex weight. For real orbitals (real Gamma /
    molecular) ``gC_dn`` is real, so early-vs-late fold are bit-identical -- this change
    is a no-op there (and the reduction's ``conj`` is likewise a no-op)."""
    if type(gd.orb_data_up_spin).__name__ != "MOs_data":
        return g_gd
    return g_gd.replace(
        orb_data_up_spin=g_gd.orb_data_up_spin.replace(mo_coefficients=gC_up),
        orb_data_dn_spin=g_gd.orb_data_dn_spin.replace(mo_coefficients=gC_dn),
    )


def _geminal_has_opt_basis(gd: GeminalData) -> bool:
    """True iff the geminal orbital carries a GTO/pGTO radial basis to optimize.

    Supported leaves are real or complex Cartesian/spherical GTO and pGTO
    bases, either directly or inside a composite. Composite PW and B-spline
    children carry no radial parameters and contribute empty gradients. The
    basis kernels concatenate GTO child gradients in child order, and
    ``_set_basis_grad`` restores that order onto the gradient pytree.

    The ECP non-local path keeps complex-twist determinant cotangents
    holomorphic until they are multiplied by complex mesh weights. Other
    consumers use the real radial-parameter derivative.
    """
    od = gd.orb_data_up_spin
    aos = od.aos_data if type(od).__name__ == "MOs_data" else od
    single_leaf_types = (
        "AOs_cart_gto_data",
        "AOs_sphe_gto_data",
        "AOs_cart_pgto_data",
        "AOs_sphe_pgto_data",
        "AOs_sphe_pgto_complex_data",
        "AOs_cart_pgto_complex_data",
    )
    composite_gto = (
        "AOs_cart_gto_data",
        "AOs_sphe_gto_data",
        "AOs_cart_pgto_data",
        "AOs_sphe_pgto_data",
        "AOs_sphe_pgto_complex_data",
        "AOs_cart_pgto_complex_data",
    )
    if type(aos).__name__ == "AOs_composite_data":
        return any(type(c).__name__ in composite_gto for c in aos.children)
    return type(aos).__name__ in single_leaf_types


def _require_geminal_basis_opt(gd: GeminalData, mask: Mapping[str, bool] | None) -> None:
    """Raise iff the param mask REQUESTS the geminal (lambda) radial basis-opt block
    (``opt_lambda_basis_exp`` / ``opt_lambda_basis_coeff`` True) but the geminal orbital
    leaf carries no analytic-optimizable radial basis (``_geminal_has_opt_basis`` is
    False) -- the silent-wrong case a plain skip would answer with a zero contribution.

    Like :func:`_require_j3_basis_opt`, this is a no-op when the block is
    frozen or no parameter mask was supplied. A parameterless orbital also
    legitimately contributes no radial gradient.
    """
    if not mask:
        return
    if not (mask.get("opt_lambda_basis_exp") or mask.get("opt_lambda_basis_coeff")):
        return
    if _geminal_has_opt_basis(gd):
        return
    # A geminal leaf that STRUCTURALLY carries no radial basis params (plane-wave / B-spline, or a
    # composite whose children are all such paramless leaves) has a legitimately empty/zero
    # lambda-basis grad, so skip-to-zero. The guard still
    # RAISES only for a leaf carrying GTO/pGTO radial params of an unsupported variant -- of which
    # there are currently NONE (all 6 cart/sphe gto/pgto incl. complex-twist are registered in
    # _capability), so this raise is currently unreachable; kept as a guard for future radial bases.
    if _capability._orb_no_radial_basis(gd.orb_data_up_spin) and _capability._orb_no_radial_basis(gd.orb_data_dn_spin):
        return
    leaf = type(gd.orb_data_up_spin).__name__
    raise NotImplementedError(
        "analytic geminal basis-opt gradient (opt_lambda_basis_exp / opt_lambda_basis_coeff) is "
        f"not implemented for the geminal orbital leaf {leaf!r}: the param mask requests it but this "
        "leaf carries an unsupported analytic-optimizable radial basis and the strict analytical "
        "route cannot return a valid derivative."
    )


def _j3_has_opt_basis(jd: Jastrow_data) -> bool:
    """True iff the three-body Jastrow orbital carries an analytic-optimizable GTO radial basis.

    Wired for a SINGLE-leaf real cart/sphe GTO, a real pgto leaf, a complex-twist sphe-pgto leaf, an
    MO-backed J3 orbital wrapping any of those AO leaves (O_k via basis.grad_J3_wrt_basis, kinetic
    via basis.grad_T_wrt_J3_basis -- both contract through the MO coefficients for the MO case; the
    complex-twist kinetic cotangent is Re-projected), AND a real-Gamma composite (pgto(+)PW) J3
    orbital, bare or MO-backed (the shared general contraction kernels iterate the children exactly
    as the geminal lambda-basis path; PW / B-spline children carry no radial params). Delegated
    to the single _capability gate to avoid drift, so its basis blocks are left zero here."""
    j3 = jd.jastrow_three_body_data
    if j3 is None:
        return False
    return _capability._j3_basis_opt_orb_ok(j3.orb_data)


def _require_j3_basis_opt(jd: Jastrow_data, mask: Mapping[str, bool] | None) -> None:
    """Raise iff the param mask REQUESTS the J3-orbital radial basis-opt block
    (``opt_J3_basis_exp`` / ``opt_J3_basis_coeff`` True) but the present three-body
    Jastrow leaf carries no analytic-optimizable radial basis (``_j3_has_opt_basis``
    is False) -- the silent-wrong case (mask asks for the derivative, the leaf cannot
    supply it, so a plain skip would return a zero contribution).

    A frozen block legitimately contributes zero, so this guard is a no-op
    when the mask does not request it or when no mask was supplied. A
    parameterless orbital also has no radial derivative to provide.
    """
    if not mask:
        return
    if not (mask.get("opt_J3_basis_exp") or mask.get("opt_J3_basis_coeff")):
        return
    j3 = jd.jastrow_three_body_data
    if j3 is None or _j3_has_opt_basis(jd):
        return
    # A leaf that STRUCTURALLY carries no radial basis params (plane-wave / B-spline, or a
    # composite whose children are all such paramless leaves) has a legitimately empty/zero
    # J3-basis grad, so skip-to-zero. The guard still RAISES only for a leaf that DOES carry
    # GTO/pGTO radial params of an unsupported variant (the genuine silent-wrong case) -- of which
    # there are currently NONE (all cart/sphe gto/pgto incl. complex-twist are registered), so this
    # raise is currently unreachable; kept as a guard for future radial bases.
    if _capability._orb_no_radial_basis(j3.orb_data):
        return
    leaf = type(j3.orb_data).__name__
    raise NotImplementedError(
        "analytic J3 basis-opt gradient (opt_J3_basis_exp / opt_J3_basis_coeff) is not implemented "
        f"for the three-body-Jastrow orbital leaf {leaf!r}: the param mask requests it but this leaf "
        "carries an unsupported analytic-optimizable radial basis and the strict analytical route "
        "cannot return a valid derivative."
    )


def _set_j3_basis_grad(
    g_jd: Jastrow_data,
    jd: Jastrow_data,
    gZ: jax.Array,
    gc: jax.Array,
) -> Jastrow_data:
    """Write the analytic d/d radial.{exponents,coefficients} grad onto the J3 grad pytree's
    orbital leaf. collect_param_grads reads j3_basis_exp/coeff from
    ``grad.jastrow_three_body_data.ao_exponents/ao_coefficients`` (which resolve
    ``orb_data.radial`` for a GTO/pGTO leaf or ``orb_data.aos_data.radial`` for an MO leaf). Gated by
    _j3_has_opt_basis (single-leaf cart/sphe GTO, real / complex-twist pgto, MO-backed, or a
    composite). Composite: the flat ``(gZ, gc)`` are the per-child GTO shell grads concatenated in
    child order (``basis._basis_contract_*_general`` appends ``zeros(0)`` for PW / B-spline) --
    split back onto each GTO child's radial exactly as the geminal ``_set_basis_grad`` does. (The
    flat reducer publishes these as per-child ``j3_child<i>_basis_*`` blocks.)"""
    g_j3 = g_jd.jastrow_three_body_data
    g_orb = g_j3.orb_data
    is_mo = type(g_orb).__name__ == "MOs_data"
    g_aos = g_orb.aos_data if is_mo else g_orb  # MO-backed J3: write onto the underlying AO leaf's radial.
    if type(g_aos).__name__ == "AOs_composite_data":
        new_children = []
        off = 0
        for c in g_aos.children:
            if "gto" in type(c).__name__:
                n = c.radial.exponents.shape[0]
                rad = c.radial.replace(exponents=gZ[off : off + n], coefficients=gc[off : off + n])
                new_children.append(c.replace(radial=rad))
                off += n
            else:
                new_children.append(c)
        new_aos = g_aos.replace(children=tuple(new_children))
    else:
        new_aos = g_aos.replace(radial=g_aos.radial.replace(exponents=gZ, coefficients=gc))
    new_orb = g_orb.replace(aos_data=new_aos) if is_mo else new_aos
    return g_jd.replace(jastrow_three_body_data=g_j3.replace(orb_data=new_orb))


def _set_basis_grad(
    g_gd: GeminalData,
    up_grads: RadialGradient,
    dn_grads: RadialGradient,
) -> GeminalData:
    """Write the d/d radial.{exponents,coefficients} grads onto the geminal grad
    pytree's AO leaf (collect_param_grads reads ao_exponents_up/dn = radial.exponents)."""

    def _set(g_orb: OrbitalData, gZ: jax.Array, gc: jax.Array) -> OrbitalData:
        is_mo = type(g_orb).__name__ == "MOs_data"
        g_aos = g_orb.aos_data if is_mo else g_orb
        if type(g_aos).__name__ == "AOs_composite_data":
            # Composite (pgto(+)PW): the flat (gZ, gc) are the per-child GTO shell
            # grads concatenated in child order (basis._basis_contract_*_general
            # appends zeros(0) for PW / B-spline). Split back onto each GTO child's
            # radial; PW / B-spline children keep their zero cotangent.
            new_children = []
            off = 0
            for c in g_aos.children:
                if "gto" in type(c).__name__:
                    n = c.radial.exponents.shape[0]
                    rad = c.radial.replace(exponents=gZ[off : off + n], coefficients=gc[off : off + n])
                    new_children.append(c.replace(radial=rad))
                    off += n
                else:
                    new_children.append(c)
            new_aos = g_aos.replace(children=tuple(new_children))
        else:
            new_aos = g_aos.replace(radial=g_aos.radial.replace(exponents=gZ, coefficients=gc))
        return g_orb.replace(aos_data=new_aos) if is_mo else new_aos

    return g_gd.replace(
        orb_data_up_spin=_set(g_gd.orb_data_up_spin, up_grads[0], up_grads[1]),
        orb_data_dn_spin=_set(g_gd.orb_data_dn_spin, dn_grads[0], dn_grads[1]),
    )


#: Wave-function pytree leaves that hold static orbital tables, which are never variational
#: parameters of any block (``wavefunction._ORB_BASIS_PAYLOAD_FIELDS``). Their c-axis
#: gradient is identically zero; as full-size zeros they dominated the per-walker output of
#: the batched c-axis programs (B-spline control points: 77 MiB per walker for Si8, so SR/LM
#: VMC ran out of GPU memory; issue #562).
_C_GRAD_STATIC_TABLE_FIELDS = frozenset({"spline_coefficients"})


def _c_grad_template(wf: Wavefunction_data) -> Wavefunction_data:
    """Zero c-axis gradient pytree of ``wf`` with every static orbital table zero-size.

    Every other leaf is ``zeros_like`` its value, as before; the derivative kernels write the
    variational leaves into this template, and the driver's block reducers read only those.
    """

    def zero(path: tuple[object, ...], leaf: jax.Array) -> jax.Array:
        name = getattr(path[-1], "name", None) if path else None
        if name in _C_GRAD_STATIC_TABLE_FIELDS:
            return jnp.zeros((0,), dtype=jnp.result_type(leaf))
        return jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(zero, wf)


def _O_k(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    *,
    holomorphic_basis: bool = False,
    geminal_inv: jax.Array | None = None,
    mask: Mapping[str, bool] | None = None,
    source_mode: bool = False,
) -> Wavefunction_data:
    """O_k = d ln|Psi| / dc as a Wavefunction_data-shaped grad pytree (zeros for the
    frozen/basis leaves; the host-side ``collect_param_grads`` keys out the blocks).
    Covered set with basis-opt OFF (the production default): geminal + Jastrow only."""
    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    grad_wf = _c_grad_template(wf)

    # --- geminal block (collect_param_grads reads grad.lambda_matrix; the masked
    # driver path reads mo_coefficients_up/dn when opt_geminal is on) ---
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        g_gd = grad_wf.geminal_data
    elif "lambda_diag" in gd.__dataclass_fields__:
        # For bare-lambda AGPn, lambda_matrix is a property; the stored variational leaves are
        # lambda_diag (paired occupations) + lambda_unpaired. Use the closed-form bare-lambda VJP
        # (geminal.dln_det_dlambda_bare) and write onto those leaves -- collect_param_grads then
        # materialises the lambda_matrix property from those AGPn leaves.
        # Canonical SD never enters this lambda branch.
        from . import geminal as _gem

        orb_up_v = gd.compute_orb_api(gd.orb_data_up_spin, r_up)  # (n, N_up)
        orb_dn_v = gd.compute_orb_api(gd.orb_data_dn_spin, r_dn)  # (n, N_dn)
        d_diag, d_unp = _gem.dln_det_dlambda_bare(
            orb_up_v,
            orb_dn_v,
            gd.lambda_diag,
            gd.lambda_unpaired,
            geminal_inv=geminal_inv,
            preserve_complex_diag=source_mode,
        )
        g_gd = grad_wf.geminal_data.replace(lambda_diag=d_diag, lambda_unpaired=d_unp)
    else:
        g_gd = _set_geminal_lambda_grad(
            grad_wf.geminal_data,
            gd,
            grad_ln_det_wrt_lambda(gd, r_up, r_dn, geminal_inv=geminal_inv),
            preserve_complex_diag=source_mode,
        )
    if type(gd.orb_data_up_spin).__name__ == "MOs_data":
        gCu, gCd = grad_ln_det_wrt_mo_coeff(gd, r_up, r_dn, geminal_inv=geminal_inv)
        g_gd = _set_geminal_mo_grad(g_gd, gd, gCu, gCd)
    _require_geminal_basis_opt(gd, mask)  # Requested unsupported radial basis must fail loudly.
    if _geminal_has_opt_basis(gd):  # (p)gto / composite radial basis-opt block (O_k)
        from . import basis

        # holomorphic_basis=True ONLY from the ECP dV_nl/dc path (complex-Bloch twist keeps Im(O_k));
        # the default (direct ln_Psi/c) Re-projects to the real d ln|det|/d radial.
        g_gd = _set_basis_grad(
            g_gd, *basis.grad_ln_det_wrt_basis(gd, r_up, r_dn, holomorphic=holomorphic_basis, geminal_inv=geminal_inv)
        )

    # --- Jastrow blocks (channels actually present) ---
    g_jd = grad_wf.jastrow_data
    if jd.jastrow_one_body_data is not None:
        gj1 = jnp.asarray(grad_J1_wrt_param(jd.jastrow_one_body_data, r_up, r_dn))
        g_jd = g_jd.replace(jastrow_one_body_data=g_jd.jastrow_one_body_data.replace(jastrow_1b_param=gj1))
    if jd.jastrow_two_body_data is not None:
        gj2 = jnp.asarray(grad_J2_wrt_param(jd.jastrow_two_body_data, r_up, r_dn))
        g_jd = g_jd.replace(jastrow_two_body_data=g_jd.jastrow_two_body_data.replace(jastrow_2b_param=gj2))
    if jd.jastrow_three_body_data is not None:
        gj3 = grad_J3_wrt_jmatrix(jd.jastrow_three_body_data, r_up, r_dn)
        g_jd = g_jd.replace(jastrow_three_body_data=g_jd.jastrow_three_body_data.replace(j_matrix=gj3))
    _require_j3_basis_opt(jd, mask)  # Requested unsupported radial basis must fail loudly.
    if _j3_has_opt_basis(jd):  # J3-orbital radial basis-opt block (O_k value: d J3/d radial)
        from . import basis

        # ALWAYS Re-projected -- unlike the determinant O_k above (holomorphic_basis threads Im for the
        # complex-twist ECP dV_nl/dc), the three-body Jastrow consumes Re(orb) so d J3/d radial is a
        # REAL derivative; the complex per-grid weight's Re is taken by compute_grad_e_L.
        g_jd = _set_j3_basis_grad(g_jd, jd, *basis.grad_J3_wrt_basis(jd, r_up, r_dn))
    if jd.jastrow_dtn_data is not None:  # DTN O_k (u/chi/f indep params; eager)
        from . import dtn

        dg = dtn.grad_dtn_wrt_param(jd.jastrow_dtn_data, r_up, r_dn)
        g_jd = g_jd.replace(jastrow_dtn_data=g_jd.jastrow_dtn_data.replace(**{k: v for k, v in dg.items()}))

    return grad_wf.replace(geminal_data=g_gd, jastrow_data=g_jd)


# Nuclear derivative of ln|Psi|. AO translation derivatives are scattered to
# their home atoms; periodic image AOs are folded before that scatter.
def _orb_pieces(orb_data: OrbitalData) -> tuple[AOs_base_data, jax.Array]:
    if type(orb_data).__name__ == "MOs_data":
        return orb_data.aos_data, jnp.asarray(orb_data._mo_coefficients_jnp)
    return orb_data, jnp.eye(orb_data.num_ao)


def _ao_grad_stack(aos_data: AOs_base_data, r_carts: jax.Array) -> jax.Array:
    gx, gy, gz = compute_AOs_grad(aos_data, r_carts)
    return jnp.stack([gx, gy, gz], axis=0)  # (3, num_ao, N)


def _scatter_atoms(s_dc: jax.Array, nucleus_index: jax.Array, n_atom: int) -> jax.Array:
    idx = jnp.asarray(nucleus_index, dtype=jnp.int32)
    return jax.ops.segment_sum(s_dc.T, idx, num_segments=n_atom)  # (n_atom, 3)


def _pgto_home_atom(aos: AOs_base_data) -> jax.Array:
    """(num_ao,) home atom of each folded AO. Molecular GTO: nucleus_index is already
    per-AO. Periodic pGTO gradients fold lattice-image AOs back to molecular rank
    (num_ao), while ``_nucleus_index_jnp`` is the longer per-image table. The
    per-folded-AO atom therefore comes from the ragged kernel-to-home and atom maps
    (``_ragged_kernel_to_home`` / ``_ragged_kernel_atom``)."""
    if getattr(aos, "_ragged_kernel_to_home", ()):
        home = np.asarray(aos._ragged_kernel_to_home)  # (M,) kernel-AO -> folded AO
        katom = np.asarray(aos._ragged_kernel_atom)  # (M,) kernel-AO -> atom
        out = np.empty(int(aos.num_ao), dtype=np.int64)
        out[home] = katom
        return jnp.asarray(out, jnp.int32)
    return jnp.asarray(aos._nucleus_index_jnp, jnp.int32)


def grad_lndet_wrt_R(
    geminal_data: GeminalData,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    take_real: bool = True,
    geminal_inv: jnpt.ArrayLike | None = None,
) -> jax.Array:
    from . import pgto_deriv

    gd = geminal_data
    M_dn = gd.orb_num_dn
    N_dn = gd.num_electron_dn
    # Without a running inverse, G comes from the fused value table that the drift
    # (kinetic._det_grad_field) inverts, so d/dR and d/dr contract one G^{-1} (#762).
    orb_up, orb_dn = geminal_value_tables(gd, r_up, r_dn, geminal_inv)
    from jqmc.determinant import SD_data

    if isinstance(gd, SD_data):
        G = jnp.hstack([orb_up[:N_dn].T @ orb_dn, orb_up[N_dn:].T])
        Q = orb_up[:N_dn].T
    else:
        lam_p, lam_unp = jnp.hsplit(gd._lambda_matrix_jnp, [M_dn])
        G = jnp.hstack([orb_up.T @ lam_p @ orb_dn, orb_up.T @ lam_unp])
        W = jnp.hstack([lam_p @ orb_dn, lam_unp])
        Q = orb_up.T @ lam_p
    # The base-position running inverse is a constant cotangent field here:
    # the nuclear response differentiates the AO translation, not the inverse.
    G_inv = resolve_geminal_inv(G, geminal_inv)  # fast: running SM inverse; else SVD-pinv (matches forward)
    if isinstance(gd, SD_data):
        M_up = jnp.concatenate([orb_dn @ G_inv[:N_dn], G_inv[N_dn:]], axis=0)
    else:
        M_up = W @ G_inv
    M_dn_coeff = (G_inv @ Q)[:N_dn].T
    n_atom = gd.orb_data_up_spin.structure_data.natom
    # take_real=True -> d ln|Psi|/dR (the Pulay force; Re for complex Bloch). take_real=False keeps
    # the holomorphic complex d lnPsi/dR for the ECP wf-ratio (where Re must be taken on the product).
    # Per AO leaf: pgto uses the Bloch translation identity (-grad_r), plane-wave / B-spline carry no
    # nuclear dependence (dAO/dR = 0) -> contribute nothing (composite descends to its children).
    out = jnp.zeros((n_atom, 3), dtype=jnp.float64 if take_real else jnp.complex128)
    for orb_data, coeff, r in ((gd.orb_data_up_spin, M_up, r_up), (gd.orb_data_dn_spin, M_dn_coeff, r_dn)):
        from jqmc.orbital_tiling import TiledOrbitalSet_data
        from .orbital_tiling import tiled_field_vjp

        if isinstance(orb_data, TiledOrbitalSet_data):
            contribution = tiled_field_vjp(orb_data, r, coeff, wrt="R")
            out = out + (jnp.real(contribution) if take_real else contribution)
            continue
        for aos, C, kind in pgto_deriv.orbital_leaves(orb_data):
            if kind in ("pw", "bspline"):
                continue  # plane-wave / B-spline carry no nuclear dependence -> dAO/dR = 0
            CM = jnp.asarray(C).T @ coeff
            gA = _ao_grad_stack(aos, r)  # forward AO grad (handles gto + pgto natively)
            s = -jnp.einsum("ck,dck->dc", CM, gA)  # -1 = translation identity
            sc = _scatter_atoms(s, _pgto_home_atom(aos), n_atom)
            out = out + (jnp.real(sc) if take_real else sc)
    return out


def grad_J1_wrt_R(
    j1_data: Jastrow_one_body_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
) -> jax.Array:
    from .kinetic import _dist_and_derivs

    sd = j1_data.structure_data
    R = jnp.asarray(sd.positions)
    Z = jnp.asarray(sd.atomic_numbers) - jnp.asarray(j1_data.core_electrons)
    p = j1_data.jastrow_1b_param
    c = (2.0 * Z) ** 0.25
    pref = -((2.0 * Z) ** 0.75) if j1_data.jastrow_1b_type == "exp" else -2.0 * Z
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    # disp = R - r so grad_d = d(d)/d(disp) = grad_{R} d (PBC: periodic distance grad).
    diff = R[None, :, :] - r_all[:, None, :]  # (Ne, nA, 3)
    dist, grad_dR, _, _ = _dist_and_derivs(sd, diff)
    if j1_data.jastrow_1b_type == "exp":
        dterm_dr = pref[None, :] * (c[None, :] / 2.0) * jnp.exp(-p * c[None, :] * dist)
    elif j1_data.jastrow_1b_type == "pade":
        dterm_dr = pref[None, :] / (2.0 * (1.0 + p * c[None, :] * dist) ** 2)
    else:
        raise ValueError(j1_data.jastrow_1b_type)
    return jnp.asarray(jnp.einsum("ea,ead->ad", dterm_dr, grad_dR))


def _j3_mo_to_bare_ao(j3_data: Jastrow_three_body_data) -> Jastrow_three_body_data:
    """Return an EQUIVALENT bare-AO three-body Jastrow when the orbital leaf is MO-backed.

    A polynomial J3 is bilinear in the orbital values with a symmetric coupling block ``M`` plus a
    linear final column ``j1col`` (see ``Jastrow_three_body_data``). If the orbital is MO-backed
    (``orbital = C @ AO``), the SAME J3 value/derivatives are reproduced by a bare-AO J3 on the
    underlying AO leaf with the coupling folded through the MO coefficients::

        square' = C.T @ square @ C,   j1col' = C.T @ j1col

    -- exactly the identity ``Jastrow_three_body_data.to_cartesian`` uses for sphe->cart (with the
    c2s matrix in place of ``C``). This lets the R/eps Pulay/stress machinery -- written for a bare
    AO leaf (its home-atom scatter / sphe->cart / AO-derivative kernels are AO-rank) -- serve an
    MO-backed J3 orbital without a dedicated MO kernel. No-op for a bare AO leaf (bit-identical)."""
    orb = j3_data.orb_data
    if type(orb).__name__ != "MOs_data":
        return j3_data
    C = jnp.asarray(orb._mo_coefficients_jnp)  # (num_mo, num_ao); traced-safe (jit bridge)
    # A complex MO J3 is Re(C @ AO) at orbital rank.  Folding its real
    # j_matrix through complex C would produce a complex AO-space matrix and
    # does not preserve the per-orbital Re operation.
    if jnp.issubdtype(C.dtype, jnp.complexfloating):
        return j3_data
    Jm = jnp.asarray(j3_data._j_matrix_jnp)  # (num_mo, num_mo + 1)
    square_ao = C.T @ Jm[:, :-1] @ C  # (num_ao, num_ao) symmetric
    j1_ao = C.T @ Jm[:, -1]  # (num_ao,)
    j_ao = jnp.concatenate([square_ao, j1_ao[:, None]], axis=1)  # (num_ao, num_ao + 1)
    return j3_data.replace(orb_data=orb.aos_data, j_matrix=j_ao)


def _j3_orb_value(orb_data: OrbitalData, r: jax.Array) -> jax.Array:
    """Re-projected J3 orbital values ``(num_orb, N)`` for ANY J3 leaf: MO-backed, bare cart/sphe
    GTO / pgto, OR a bare composite (pgto(+)PW). ``Jastrow_three_body_data.compute_orb_api`` only
    dispatches cart/sphe-GTO / MO leaves, so the bare-composite leaf produced by
    ``_j3_mo_to_bare_ao`` (an MO-backed composite J3 folded to AO rank) needs this leaf-agnostic
    forward (``compute_AOs`` handles every AO container, including composites)."""
    from jqmc.atomic_orbital import compute_AOs
    from jqmc.molecular_orbital import compute_MOs

    if type(orb_data).__name__ == "MOs_data":
        return jnp.real(compute_MOs(orb_data, r))
    return jnp.real(compute_AOs(orb_data, r))


def grad_J3_wrt_R(
    j3_data: Jastrow_three_body_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
) -> jax.Array:
    from . import pgto_deriv

    j3_data = _j3_mo_to_bare_ao(j3_data)  # MO-backed J3 -> equivalent bare-AO J3 (bit-identical value)
    A_up = _j3_orb_value(j3_data.orb_data, r_up)
    A_dn = _j3_orb_value(j3_data.orb_data, r_dn)
    n_up, n_dn = A_up.shape[1], A_dn.shape[1]
    K_up = jnp.tril(jnp.ones((n_up, n_up)), k=-1)
    K_dn = jnp.tril(jnp.ones((n_dn, n_dn)), k=-1)
    Jm = j3_data._j_matrix_jnp
    M = Jm[:, :-1]
    j1col = Jm[:, -1]
    s_up = A_up.sum(1)
    s_dn = A_dn.sum(1)
    gA_up = j1col[:, None] + M @ A_up @ K_up + M.T @ A_up @ K_up.T + (M @ s_dn)[:, None]
    gA_dn = j1col[:, None] + M @ A_dn @ K_dn + M.T @ A_dn @ K_dn.T + (M.T @ s_up)[:, None]
    aos = j3_data.orb_data
    leaves = pgto_deriv.orbital_leaves(aos)
    n_atom = jnp.asarray(leaves[0][0].structure_data.positions).shape[0]
    mo_rank = type(aos).__name__ == "MOs_data"
    out = jnp.zeros((n_atom, 3))
    # Per-AO-leaf translation identity (a composite descends to its children, exactly as the det
    # force grad_lndet_wrt_R does): atom-centred GTO/pGTO children scatter -grad_r onto their home
    # atoms; plane-wave / B-spline children carry NO nuclear dependence (dAO/dR = 0) -> skipped.
    # A single bare leaf is one "child" with identity coupling -- bit-identical to the legacy path.
    for coeff, r in ((gA_up, r_up), (gA_dn, r_dn)):
        off = 0
        for leaf, C, kind in leaves:
            n = int(leaf.num_ao)
            if kind in ("pw", "bspline"):
                off += n
                continue
            dA = _ao_grad_stack(leaf, r)
            if mo_rank:
                coeff_leaf = jnp.asarray(C).T @ coeff
                s = -jnp.real(jnp.einsum("ck,dck->dc", coeff_leaf, dA))
            else:
                s = -jnp.einsum("ck,dck->dc", jnp.real(coeff[off : off + n]), jnp.real(dA))
            out = out + _scatter_atoms(s, _pgto_home_atom(leaf), n_atom)  # pgto folds to molecular rank
            off += n
    return out


def _force_lnPsi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    take_real: bool = True,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """d ln|Psi| / dR (n_atom, 3), molecular: det + J1 + J3 (J2 has no R dependence). take_real=False
    returns the holomorphic complex d lnPsi/dR (the ECP wf-ratio needs the complex form).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the det Pulay force."""
    wf = hamiltonian_data.wavefunction_data
    jd = wf.jastrow_data
    out = grad_lndet_wrt_R(wf.geminal_data, r_up, r_dn, take_real=take_real, geminal_inv=geminal_inv)
    if jd.jastrow_one_body_data is not None:
        out = out + grad_J1_wrt_R(jd.jastrow_one_body_data, r_up, r_dn)
    if jd.jastrow_three_body_data is not None:
        out = out + grad_J3_wrt_R(jd.jastrow_three_body_data, r_up, r_dn)
    if jd.jastrow_dtn_data is not None:
        from . import dtn

        dtn_R = dtn.grad_dtn_wrt_R(jd.jastrow_dtn_data, r_up, r_dn)
        out = out + (dtn_R if take_real else jnp.asarray(dtn_R, out.dtype))
    return out


def _struct_positions(geminal_data: GeminalData) -> jax.Array:
    return geminal_data.orb_data_up_spin.structure_data._positions_cart_jnp


def _stress_lnPsi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """Pulay stress d ln|Psi| / d eps for a molecular affine strain. Strain
    r' = (I+eps) r, R' = (I+eps) R contracts the position gradients with the
    positions: d/deps_cd = sum_e (d lnPsi/d r_e)_c r_e,d + sum_I (d lnPsi/d R_I)_c R_I,d.
    Reuses the det drift field + Jastrow drift (d/dr) and _force_lnPsi (d/dR).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by both the det drift
    field and the det Pulay force (the eps virial contracts base-position fields, no autodiff-pinv)."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _det_grad_field

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    R = _struct_positions(gd)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    gdu, gdd = _det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    drift = jnp.concatenate([jnp.asarray(gdu), jnp.asarray(gdd)], axis=0)
    if any(
        getattr(jd, f) is not None
        for f in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data", "jastrow_dtn_data")
    ):
        gju, gjd, _, _ = compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn)
        drift = drift + jnp.concatenate([jnp.asarray(gju), jnp.asarray(gjd)], axis=0)
    dR = jnp.asarray(_force_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv))
    stress = jnp.einsum("ec,ed->cd", drift, r_all) + jnp.einsum("ic,id->cd", dR, R)
    return jnp.asarray(stress)


def _drift_lnPsi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d ln|Psi| / d r per electron (the drift): det field gD + Jastrow grad gJ.
    Returns (up (n_up, 3), dn (n_dn, 3)).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the det drift field."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _det_grad_field

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    gDu, gDd = _det_grad_field(gd, r_up, r_dn, geminal_inv=geminal_inv)
    du, dd = jnp.asarray(gDu), jnp.asarray(gDd)
    if any(
        getattr(jd, f) is not None
        for f in ("jastrow_one_body_data", "jastrow_two_body_data", "jastrow_three_body_data", "jastrow_dtn_data")
    ):
        gju, gjd, _, _ = compute_grads_and_laplacian_Jastrow_part(jd, r_up, r_dn)
        du = du + jnp.asarray(gju)
        dd = dd + jnp.asarray(gjd)
    # The drift of ln|Psi| is Re(grad ln Psi); for complex-Bloch (twist) orbitals grad ln Psi
    # is complex and the real part is the sampling drift (jnp.real is a no-op for real systems).
    return jnp.real(du), jnp.real(dd)


def compute_grad_ln_Psi(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    *,
    wrt: DerivativeAxis,
    mask: Mapping[str, bool] | None = None,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> DerivativeValue:
    if wrt == "c":
        # O_k is holomorphic and is not conjugated, so it is correct for a
        # complex-Bloch twist too. The basis
        # block is Re-projected inside grad_ln_det_wrt_basis (default holomorphic=False -> real
        # d ln|det|/d radial, matching the real d ln|Psi|/d radial oracle); only the ECP dV_nl/dc
        # opts into the un-projected holomorphic basis O_k (via _O_k(holomorphic_basis=True)).
        # Reuse the running Sherman-Morrison inverse so the optimization graph
        # does not lower a fresh SVD.
        return _O_k(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv, mask=mask, source_mode=source_mode)
    if wrt == "r":
        # The drift = Re(grad ln Psi) (handled in _drift_lnPsi), exact for real and complex Bloch.
        return _drift_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
    if wrt == "R":
        # PBC Pulay force IS analytic (real Gamma AND complex twist): the Bloch translation identity
        # d phi/dR = -d phi/dr holds (image phase depends on the integer image index, not R), the
        # periodic det grad folds image-AOs to molecular rank (home-atom scatter, Re for twist), the
        # J1 force uses the periodic distance grad, and DTN is in _force_lnPsi (dtn.grad_dtn_wrt_R).
        return _force_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
    if wrt == "eps":
        if hamiltonian_data.structure_data.pbc_flag:
            # Periodic Pulay stress: affine virial PLUS the cell-strain (image-T + smooth-distance
            # lattice-length) terms + the DTN min-image displacement virial -- all inside
            # pbc_stress.stress_lnPsi_pbc (single source, so the ECP wf-ratio stress gets DTN too).
            from . import pbc_stress

            return pbc_stress.stress_lnPsi_pbc(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
        # molecular: the position-virial _stress_lnPsi already captures DTN (drift + dR both include it).
        return _stress_lnPsi(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
    # Unreachable: the dispatch (`__init__._split_flags`) only ever emits wrt in {c, r, R, eps},
    # all handled above. Reaching here means that closure invariant was violated (a dispatch bug),
    # NOT a missing feature -- so this is an internal assertion, not a NotImplementedError.
    raise AssertionError(f"compute_grad_ln_Psi: unexpected wrt={wrt!r} (dispatch emits only c/r/R/eps)")


# Local-energy parameter derivative. Bare Coulomb is parameter-independent;
# ECP non-local terms are added separately through the displaced-mesh O_k.
def _dE_L_dc(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
    mask: Mapping[str, bool] | None = None,
    *,
    source_mode: bool = False,
) -> Wavefunction_data:
    from . import kinetic
    from jqmc.determinant import SD_data

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    grad_wf = _c_grad_template(wf)
    # The kinetic reverse pass reuses the running ``geminal_inv`` (base
    # walker position) instead of a fresh SVD-pinv, so the optimize de_L/dc graph carries no
    # lapack_dgesdd custom call. geminal_inv is None only off the driver (equivalence harness),
    # where resolve_geminal_inv falls back to the thresholded SVD-pinv (unchanged behaviour).
    if isinstance(gd, SD_data):
        # A canonical SD has no geminal-eigenvalue leaf: all determinant
        # freedom lives in the occupied MO rows.
        g_gd = grad_wf.geminal_data
    else:
        g_gd = _set_geminal_lambda_grad(
            grad_wf.geminal_data,
            gd,
            kinetic.grad_T_wrt_lambda_withJ(gd, jd, r_up, r_dn, geminal_inv=geminal_inv),
            preserve_complex_diag=True,
        )
    if type(gd.orb_data_up_spin).__name__ == "MOs_data":
        cu, cd = kinetic.grad_T_wrt_mo_withJ(gd, jd, r_up, r_dn, geminal_inv=geminal_inv)
        g_gd = _set_geminal_mo_grad(g_gd, gd, cu, cd)
    _require_geminal_basis_opt(gd, mask)  # Requested unsupported radial basis must fail loudly.
    if _geminal_has_opt_basis(gd):  # (p)gto / composite radial basis-opt block (dT/dc; dV_bare/dc=0)
        from . import basis

        g_gd = _set_basis_grad(
            g_gd,
            *basis.grad_T_wrt_basis(
                gd,
                jd,
                r_up,
                r_dn,
                geminal_inv=geminal_inv,
                holomorphic=source_mode,
            ),
        )
    g_jd = grad_wf.jastrow_data
    if jd.jastrow_one_body_data is not None:
        gj1 = jnp.asarray(kinetic.grad_T_wrt_J1(gd, jd, r_up, r_dn, geminal_inv=geminal_inv))
        g_jd = g_jd.replace(jastrow_one_body_data=g_jd.jastrow_one_body_data.replace(jastrow_1b_param=gj1))
    if jd.jastrow_two_body_data is not None:
        gj2 = jnp.asarray(kinetic.grad_T_wrt_J2(gd, jd, r_up, r_dn, geminal_inv=geminal_inv))
        g_jd = g_jd.replace(jastrow_two_body_data=g_jd.jastrow_two_body_data.replace(jastrow_2b_param=gj2))
    if jd.jastrow_three_body_data is not None:  # J3 kinetic dT/d j_matrix
        gj3 = jnp.asarray(kinetic.grad_T_wrt_J3(gd, jd, r_up, r_dn, geminal_inv=geminal_inv))
        g_jd = g_jd.replace(jastrow_three_body_data=g_jd.jastrow_three_body_data.replace(j_matrix=gj3))
    _require_j3_basis_opt(jd, mask)  # Requested unsupported radial basis must fail loudly.
    if _j3_has_opt_basis(jd):  # J3-orbital radial basis-opt block (kinetic dT/dc; dV_bare/dc=0)
        from . import basis

        g_jd = _set_j3_basis_grad(g_jd, jd, *basis.grad_T_wrt_J3_basis(gd, jd, r_up, r_dn, geminal_inv=geminal_inv))
    if jd.jastrow_dtn_data is not None:  # DTN dT/dc (u/chi/f channels; eager)
        from . import dtn

        dg = dtn.grad_T_dtn_wrt_param(jd.jastrow_dtn_data, jd, gd, r_up, r_dn, geminal_inv=geminal_inv)
        g_jd = g_jd.replace(jastrow_dtn_data=g_jd.jastrow_dtn_data.replace(**{k: v for k, v in dg.items()}))
    return grad_wf.replace(geminal_data=g_gd, jastrow_data=g_jd)


def _dV_ecp_dc(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> Wavefunction_data:
    """Differentiate the non-local ECP potential through displaced-mesh O_k values.

    V_nl = sum_g V_grid_g * Psi(r'_g)/Psi(r), with c-independent weights, so
    dV_nl/dc = sum_g V_grid_g * O_k(r'_g) - sumV * O_k(r), where O_k = d lnPsi/dc is the
    FULL covered-set O_k (geminal + Jastrow) reused at the mesh-displaced configs and
    V_grid already carries the Psi(r')/Psi(r) ratio. Uses the production NN-truncated
    mesh (``..._nearest_neighbors``) so it matches the production forward exactly.

    Complex-Bloch (twist) note: the forward ``e_L`` non-local part is Re(V_nl), so the
    correct c-derivative is Re[sum_g W_g (O_k(r'_g) - O_k(r))] with W_g = B_g * wf_ratio_g
    the COMPLEX (holomorphic) per-grid weight. ``Re`` and the complex ``O_k`` do NOT commute,
    so the real (already-Re-projected) forward weight would drop the Im(wf_ratio).Im(O_k)
    cross term. We therefore request the un-projected complex weights (the ``_holomorphic``
    entry) and pair them with the holomorphic ``O_k`` (via ``_O_k(holomorphic_basis=True)``, which asks
    ``grad_ln_det_wrt_basis`` to skip its default Re projection for THIS path only -- every other
    O_k consumer keeps the real-projected basis grad); the caller (compute_grad_e_L) takes ``Re`` on
    the assembled c-grad. At the real Gamma point the weights are already real, so bit-identical."""
    from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_holomorphic

    wf = hamiltonian_data.wavefunction_data
    cp = hamiltonian_data.coulomb_potential_data
    # On supported molecular real inputs, the weighted O_k mesh sum
    # collapses into bounded GEMMs (sumV = sum_g W_g, so only the rank-1 per-config
    # O_k DELTAS survive) -- see one_move_mesh.dV_ecp_dc_fast.  Unsupported cases keep the
    # streamed accumulation below.
    from . import one_move_mesh

    if one_move_mesh.fast_supported_c(hamiltonian_data):
        from .ecp_force import _ecp_nl_common

        # Base-position SM inverse threads into the rank-1 mesh setup (det_base) so no fresh SVD-pinv.
        c = _ecp_nl_common(hamiltonian_data, r_up, r_dn, RT, None, geminal_inv=geminal_inv)
        return one_move_mesh.dV_ecp_dc_fast(hamiltonian_data, r_up, r_dn, c)
    # holomorphic -> keep the un-projected complex per-grid weights (no-op at Gamma).
    mesh_up, mesh_dn, V_grid, sumV = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(cp, wf, r_up, r_dn, RT)
    V_grid = jnp.asarray(V_grid)
    mesh_up = jnp.asarray(mesh_up)
    mesh_dn = jnp.asarray(mesh_dn)
    # V_grid carries the value-only determinant ratio det(G'_g) / det(G).  The mesh O_k receive
    # per-config inverses and therefore read that table; the base O_k must use the same base
    # inverse (one_move_mesh.ratio_table_geminal_inv; a supplied running inverse is unchanged).
    geminal_inv = one_move_mesh.ratio_table_geminal_inv(wf.geminal_data, r_up, r_dn, geminal_inv)
    # holomorphic_basis=True: pair the complex holomorphic basis O_k with the complex per-grid weight
    # (the Re is deferred to compute_grad_e_L's final tree_map). No-op at real Gamma / molecular.
    ok_base = _O_k(
        hamiltonian_data,
        r_up,
        r_dn,
        holomorphic_basis=True,
        geminal_inv=geminal_inv,
        source_mode=True,
    )
    # The mesh O_k reduction sum_g V_grid_g O_k(r'_g) is accumulated inside a
    # bounded chunked scan (ecp_force._mesh_wsum).  A full vmap (or even a chunk-streamed
    # STACK) of the per-config O_k parameter-gradient pytrees is mesh-sized -- the O_k
    # lambda/MO/J3 leaves are ~N_AO^2 per config (138 GB for the e_L/c mesh of a water system).  Live
    # memory is the running weighted sum + one chunk of O_k fields.  Same per-config
    # math; the g-reduction is regrouped chunk-wise (fp-benign reassociation).
    from .mesh_stream import _mesh_wsum

    # Per-config displaced geminal inverse via the EXACT rank-1 Sherman-Morrison update off the base
    # (threaded) running inverse -- the ECP move displaces ONE electron (rank-1 row/column of G), so no
    # fresh SVD-pinv is needed at any mesh config.  Threading it into the streamed _O_k keeps this
    # complex-twist path free of ``lapack_[dz]gesdd`` (a LAPACK custom call that is unavailable on some
    # MPI host backends).  Off the
    # driver (no running inverse) the base is the value-only ratio-table inverse resolved above.
    ginv_cfg = one_move_mesh.sm_geminal_inv_configs(wf.geminal_data, r_up, r_dn, mesh_up, mesh_dn, geminal_inv=geminal_inv)
    ok_wsum = _mesh_wsum(
        lambda a, c, gi: _O_k(hamiltonian_data, a, c, holomorphic_basis=True, geminal_inv=gi, source_mode=True),
        V_grid,
        (mesh_up, mesh_dn, ginv_cfg),
    )
    return jax.tree_util.tree_map(lambda m, base: m - sumV * base, ok_wsum, ok_base)


def _stress_eL(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """HF stress d e_L / d eps for a molecular affine strain. Strain r' = (I+eps) r,
    R' = (I+eps) R contracts the e_L position gradients with the positions:
    d/deps_cd = sum_e (d e_L/d r_e)_c r_e,d + sum_I (d e_L/d R_I)_c R_I,d.
    d e_L/d r = dT/dr (with Jastrow) + dV_bare/dr, while the nuclear
    derivative is dT/dR + dV/dR.

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by dT/dr and dT/dR."""
    from . import force, kinetic

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cp = hamiltonian_data.coulomb_potential_data
    R = _struct_positions(gd)
    r_all = jnp.concatenate([jnp.asarray(r_up), jnp.asarray(r_dn)], axis=0)
    deL_dr = kinetic.grad_T_wrt_r(gd, jd, r_up, r_dn, geminal_inv=geminal_inv) + force.grad_Vbare_wrt_r(cp, r_up, r_dn)
    deL_dR = force.grad_T_wrt_R_withJ(gd, jd, r_up, r_dn, geminal_inv=geminal_inv) + force.grad_V_wrt_R(cp, r_up, r_dn)
    stress = jnp.einsum("ec,ed->cd", deL_dr, r_all) + jnp.einsum("ic,id->cd", deL_dR, R)
    return jnp.asarray(stress)


def _deL_dr(
    hamiltonian_data: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    RT: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """d e_L / d r per electron = dT/dr (with Jastrow) + dV_bare/dr + dV_ecp/dr. Molecular.
    Returns (up (n_up, 3), dn (n_dn, 3)).

    ``geminal_inv`` (fast path): base-position running SM ``G^{-1}`` reused by the kinetic dT/dr and
    the ECP non-local base setup (mesh-config re-evals use their own inverse)."""
    import jax.numpy as _jnp

    from . import force, kinetic

    wf = hamiltonian_data.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cp = hamiltonian_data.coulomb_potential_data
    deL = kinetic.grad_T_wrt_r(gd, jd, r_up, r_dn, geminal_inv=geminal_inv)
    if hamiltonian_data.structure_data.pbc_flag:  # periodic Coulomb drift = Ewald e-e + Ewald e-i
        from . import ewald

        ewd = cp.ewald_data
        deL = deL + _jnp.asarray(ewald.grad_Eee_wrt_r(ewd, r_up, r_dn))
        deL = deL + _jnp.asarray(ewald.grad_Eei_wrt_r(cp, ewd, r_up, r_dn))
    else:
        deL = deL + force.grad_Vbare_wrt_r(cp, r_up, r_dn)
    if getattr(cp, "ecp_flag", False):  # ECP local + non-local electron gradient (eager)
        from . import ecp_force

        deL = deL + _jnp.asarray(ecp_force.grad_V_ecp_wrt_r(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv))
    if jd.jastrow_dtn_data is not None:  # DTN kinetic drift (NN-cutoff u/chi/f)
        from . import dtn

        deL = deL + dtn.grad_T_dtn_wrt_r(jd.jastrow_dtn_data, jd, gd, r_up, r_dn, geminal_inv=geminal_inv)
    n_up = r_up.shape[0]
    return deL[:n_up], deL[n_up:]


def compute_grad_e_L(
    hamiltonian_data: Hamiltonian_data,
    r_up: jnpt.ArrayLike,
    r_dn: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    *,
    wrt: DerivativeAxis,
    mask: Mapping[str, bool] | None = None,
    fast: bool = False,
    geminal_inv: jnpt.ArrayLike | None = None,
    source_mode: bool = False,
) -> DerivativeValue:
    if wrt == "r":
        cp = hamiltonian_data.coulomb_potential_data
        jd = hamiltonian_data.wavefunction_data.jastrow_data
        # Complex-Bloch (twist) e_L/r is analytic: the periodic kinetic drift + Ewald e-e/e-i
        # Coulomb drift + ECP electron-gradient, with the ECP wf-ratio kept holomorphic and Re
        # taken on the product (ecp_force._O_drift take_real=False). DTN drift = dtn.grad_T_dtn_wrt_r
        # (added inside _deL_dr).
        return _deL_dr(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
    if wrt == "c":
        cp = hamiltonian_data.coulomb_potential_data
        jd = hamiltonian_data.wavefunction_data.jastrow_data
        # Real Gamma PBC e_L/c IS analytic: dV_ewald/dc = dV_bare/dc = 0 (c-independent); the geminal
        # kinetic is pbc-aware; J1/J2 kinetic dT/dc uses the periodic distance's closed-form grad+lap;
        # dV_ecp/dc reuses the basis/pbc-agnostic O_k(r') mesh identity; complex-Bloch (twist) is
        # holomorphic -> jnp.real at the end. DTN dV_ecp/dc + DTN kinetic dT/dc are already threaded
        # through _dV_ecp_dc / _dE_L_dc (dtn.grad_dtn_wrt_param / grad_T_dtn_wrt_param, min-image).
        # J3 kinetic dT/d j_matrix (kinetic.grad_T_wrt_J3, ao_deriv.j3_basis_derivs is periodic-pgto-
        # aware) is analytic for molecular AND periodic (real Gamma + twist; j3_basis_derivs Re-projects).
        out = _dE_L_dc(
            hamiltonian_data,
            r_up,
            r_dn,
            geminal_inv=geminal_inv,
            mask=mask,
            source_mode=source_mode,
        )  # kinetic dT/dc (dV_local/dc = 0)
        if getattr(cp, "ecp_flag", False):  # ECP non-local dV_nl/dc via O_k(r')
            dvecp = _dV_ecp_dc(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
            out = jax.tree_util.tree_map(lambda a, b: a + b, out, dvecp)
        if source_mode:
            # Source mode retains the complete complex derivative even for a
            # real parameter leaf. The storage adapter alone restores the real-leaf
            # projection required by the current solver.
            return out
        # The public derivative contract optimizes the real part of e_L. The
        # projection is selective by PARAMETER dtype, so real blocks remain
        # real while a complex MO/lambda leaf keeps its non-conjugated
        # holomorphic cotangent.  Source mode above is the only path that
        # bypasses this boundary.
        wf_params = hamiltonian_data.wavefunction_data
        return jax.tree_util.tree_map(lambda g, p: g if jnp.iscomplexobj(p) else jnp.real(g), out, wf_params)
    if wrt == "R":
        cp = hamiltonian_data.coulomb_potential_data
        jd = hamiltonian_data.wavefunction_data.jastrow_data
        # Real Gamma PBC e_L/R IS analytic: periodic kinetic force (folded pgto AO higher
        # derivatives + smooth-distance J1 side), Ewald HF force, and the periodic ECP force
        # (min-image via the jnp _get_min_dist helper). Complex-Bloch (twist) is analytic too:
        # the ECP force/_force_lnPsi keep ratio/O_R holomorphic and Re the assembled dVdR at the
        # end (take_real=False). DTN kinetic force = dtn.grad_T_dtn_wrt_R (added below).
        from . import force

        gd = hamiltonian_data.wavefunction_data.geminal_data
        out = force.grad_T_wrt_R_withJ(gd, jd, r_up, r_dn, geminal_inv=geminal_inv) + force.grad_V_wrt_R(cp, r_up, r_dn)
        if jd.jastrow_dtn_data is not None:  # DTN kinetic nuclear force (Jastrow-side, NN-cutoff u/chi/f)
            from . import dtn

            out = out + dtn.grad_T_dtn_wrt_R(jd.jastrow_dtn_data, jd, gd, r_up, r_dn, geminal_inv=geminal_inv)
        if getattr(cp, "ecp_flag", False):  # ECP local + non-local nuclear force
            from . import ecp_force

            out = out + ecp_force.grad_V_ecp_wrt_R(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
        return out
    if wrt == "eps":
        cp = hamiltonian_data.coulomb_potential_data
        jd = hamiltonian_data.wavefunction_data.jastrow_data
        # Complex-Bloch (twist) e_L/eps is analytic: periodic kinetic stress + Ewald stress + ECP
        # stress, with the det-side Pulay stress (pbc_stress.stress_lnPsi_pbc take_real=False) and
        # ECP wf-ratio kept holomorphic and Re taken on the product. DTN kinetic stress =
        # dtn.grad_T_dtn_wrt_strain (added below).
        gd = hamiltonian_data.wavefunction_data.geminal_data
        if hamiltonian_data.structure_data.pbc_flag:
            # Real Gamma PBC e_L/eps = periodic kinetic stress (orbital + Jastrow) + Ewald (ii+ei+ee)
            # stress + ECP stress (local affine virial + non-local mesh, with stress_lnPsi at the mesh
            # configs for the wf-ratio cell term).
            from . import ewald, pbc_stress

            out = pbc_stress.stress_eL_kinetic_pbc(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
            out = out + ewald.grad_V_ewald_wrt_strain(cp, r_up, r_dn)
            if jd.jastrow_dtn_data is not None:  # DTN kinetic stress (NN-cutoff displacement virial)
                from . import dtn

                out = out + dtn.grad_T_dtn_wrt_strain(jd.jastrow_dtn_data, jd, gd, r_up, r_dn, geminal_inv=geminal_inv)
            if getattr(cp, "ecp_flag", False):
                from . import ecp_force

                out = out + ecp_force.grad_V_ecp_wrt_strain(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
            return out
        # Molecular affine stress d e_L/d eps = T + V_bare virial (_stress_eL) + ECP local/non-local
        # strain (ecp_force.grad_V_ecp_wrt_strain_mol: local affine virial + non-local geometry strain +
        # non-local wf-ratio MOLECULAR Pulay stress in CLOSED FORM off the rank-1 config drift/force
        # fields -- no autodiff-pinv, so the axis lowers with ZERO lapack_dgesdd under the shared inverse)
        # + DTN kinetic strain. Removes the LAST production _auto fallback (molecular-ECP e_L/eps).
        out = _stress_eL(hamiltonian_data, r_up, r_dn, geminal_inv=geminal_inv)
        if getattr(cp, "ecp_flag", False):  # molecular ECP local + non-local stress (eager)
            from . import ecp_force

            out = out + ecp_force.grad_V_ecp_wrt_strain_mol(hamiltonian_data, r_up, r_dn, RT, geminal_inv=geminal_inv)
        if jd.jastrow_dtn_data is not None:  # molecular AE/ECP DTN kinetic stress
            from . import dtn

            out = out + dtn.grad_T_dtn_wrt_strain(jd.jastrow_dtn_data, jd, gd, r_up, r_dn, geminal_inv=geminal_inv)
        return out
    # Unreachable: as in compute_grad_ln_Psi above, the dispatch only emits wrt in {c, r, R, eps},
    # all handled above. Reaching here is a dispatch-invariant violation (a bug), not a missing
    # feature -- internal assertion, not NotImplementedError.
    raise AssertionError(f"compute_grad_e_L: unexpected wrt={wrt!r} (dispatch emits only c/r/R/eps)")


# Capability registration for the analytical kernels in this module. These
# sets describe supported leaves; strict dispatch still raises for an
# unsupported requested axis instead of silently using an automatic gradient.
from . import _capability  # noqa: E402

# JIT-safe AO leaves use JAX geometry fields because the Hamiltonian is a
# traced, non-vmapped bridge argument. Periodic and composite conversions stay
# inside the leaf kernels rather than applying the spherical-to-Cartesian hoist.
_capability.SUPPORTED_AO_TYPES_JIT.update(
    {
        "AOs_cart_gto_data",
        "AOs_sphe_gto_data",
        "AOs_cart_pgto_data",
        "AOs_sphe_pgto_data",
        # Complex-Bloch periodic pGTO. Analytical kernels stay holomorphic in
        # the orbitals; drift, force, stress, and e_L project the assembled product.
        "AOs_sphe_pgto_complex_data",
        # Plane waves have no nuclear derivative. Their drift and kinetic
        # strain use closed-form reciprocal-space kernels.
        "AOs_plane_wave_data",
        # B-splines also have no nuclear derivative. Separable-spline fields
        # provide drift and metric-strain kinetic contributions.
        "AOs_bspline_data",
        # Complex plane waves use g+k in the same value/derivative formulas;
        # complex Cartesian pGTO uses the complex periodic-GTO kernel.
        "AOs_plane_wave_complex_data",
        "AOs_cart_pgto_complex_data",
        # Complex B-splines combine separable-spline fields with the Bloch
        # phase product rule and still have no nuclear derivative.
        "AOs_bspline_complex_data",
    }
)
# J3 radial basis gradients support real/complex Cartesian and spherical
# GTO/pGTO leaves. MO wrappers and composites are handled by the capability
# predicate without adding wrapper class names to this leaf set.
_capability.SUPPORTED_J3_BASIS_OPT_AO.update(
    {
        "AOs_cart_gto_data",
        "AOs_sphe_gto_data",
        "AOs_cart_pgto_data",
        "AOs_sphe_pgto_data",
        "AOs_sphe_pgto_complex_data",
        "AOs_cart_pgto_complex_data",
    }
)
