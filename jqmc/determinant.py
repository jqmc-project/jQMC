"""Determinant module.

Precision Zones:
    - ``det_eval``: geminal matrix elements and the log-determinant
      (compute_geminal_*, compute_ln_det_*, compute_det_*).
    - ``det_grad_lap``: gradient and Laplacian of ln|Det|
      (compute_grads_and_laplacian_ln_Det*).
    - ``det_ratio``: rank-1 determinant-ratio updates.

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

# python modules
from collections.abc import Callable
from logging import getLogger
from types import ModuleType

# jqmc module
from typing import TYPE_CHECKING, Literal, NoReturn, Self, TypeVar

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit, lax, vmap

from . import _complex_update
from ._precision import cast_to_zone, get_dtype_jnp, get_dtype_np
from ._setting import (
    AO_OVERLAP_EIGENVALUE_FLOOR,
    ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
    GEMINAL_EQUILIBRATION_MAX_EXPONENT,
    GEMINAL_REPRESENTATION_ATOL,
    RTOL_BIT_EXACT,
    as_linear_min_sq_norm,
    atol_consistency,
    get_eps,
    rtol_consistency,
    sm_large_ratio_threshold,
)
from .atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    AOs_composite_data,
    AOs_GTO_base_data,
    AOs_sphe_gto_data,
    _aos_cart_to_sphe,
    _aos_sphe_to_cart,
    _apply_strain_to_aos,
    _compute_AOs_debug,
    _structure_data_equal,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    compute_overlap_matrix,
)
from .molecular_orbital import (
    MOs_data,
    _compute_MOs_debug,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
    compute_MOs_value_grad_lap,
)

from .orbital_tiling import (
    TiledOrbitalSet_data,
    _compute_tiled_orbitals_debug,
    compute_tiled_orbitals,
    compute_tiled_orbitals_grad,
    compute_tiled_orbitals_laplacian,
    compute_tiled_orbitals_value_grad_lap,
    conjugate_tiled_orbitals,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid circular dependency
    from .wavefunction import VariationalParameterBlock

_OrbitalDataT = TypeVar("_OrbitalDataT", bound=AOs_base_data | MOs_data)
_FrozenOverlap = tuple[npt.ArrayLike, npt.ArrayLike] | None

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# ---------------------------------------------------------------------------
# Private-helper naming convention (used throughout this module).
#
# Many defs here are private (leading ``_``) but structurally required rather
# than incidental. A reader can classify a helper from its name:
#
#   ``_*_fwd`` / ``_*_bwd``
#       The two halves of a ``jax.custom_vjp`` rule (forward residual +
#       backward cotangent). Always come in a pair bound to a public kernel.
#   ``_init_*_state`` / ``_advance_*_state``
#       The ``lax.scan``-style streaming pair: build the initial carry, then
#       advance it by one accepted single-electron move (MCMC / GFMC / LRDMC).
#   ``_*_debug``
#       Test-only reference oracle (slow, direct recompute) paired with a
#       fast production kernel; kept for equivalence tests.
#   ``_get_info`` / ``_logger_info`` / ``_determinant_dtype_jnp``
#       Cross-module accessors: private-named (per-subclass, not public API)
#       but called widely across jqmc (loggers / dispatch / kinetic).
#   ``_lambda_matrix_jnp`` remains an AGP/AGPn-only internal accessor; SD
#       deliberately has no lambda interface.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AO/MO basis helpers (module-level): type-dispatch over AO leaves / MOs_data /
# composite containers, plus the MPC overlap guard. These unwrap the orbital
# container and route GTO-radial (exponents / coefficients) reads and writes.
# ---------------------------------------------------------------------------


def _is_ao_leaf(orb: AOs_base_data | MOs_data) -> bool:
    """Return True iff ``orb`` is an AO leaf (any AOs_base_data subclass except MOs_data).

    Covers all AO leaves -- GTO (cart/sphe), periodic pGTO (cart/sphe),
    plane-wave, B-spline, and the composite container -- while excluding
    the MO wrapper that also stores ``aos_data`` internally.
    """
    return isinstance(orb, AOs_base_data) and not isinstance(orb, MOs_data)


def _overlap_matrix_traceable(aos_data: AOs_base_data) -> npt.NDArray:
    r"""``compute_overlap_matrix(aos_data)`` that is safe under ``jax.grad`` of the whole ham.

    The AO overlap kernel (sphe->cart detile + analytic Gaussian integrals) is a
    host-side numpy construction (shell-matching + Hermite recursion, e.g.
    :func:`atomic_orbital._aos_sphe_to_cart` /
    :func:`atomic_orbital._compute_overlap_matrix_cart_analytic`) that reads the
    AO ``exponents`` / ``coefficients`` / centres as concrete numpy and cannot
    run on JAX tracers. It is only invoked here, by the AO-AGP branch of
    :meth:`natural_orbital_density_coefficients_jnp`, to build the MPC one-body
    density. When that runs inside a ``jax.grad`` w.r.t. the ham, EVERY ham
    float leaf is a tracer and the direct call raises
    ``TracerArrayConversionError``.

    Route the overlap through a host callback and hold it CONSTANT w.r.t. the
    differentiation, which freezes the natural-orbital coeff/occ w.r.t.
    geometry. (MPC itself differentiates nothing -- it is a
    passive, energy-only observable; see :mod:`jqmc.mpc`. The freeze is
    what keeps this host-side numpy kernel out of any enclosing trace.) The
    overlap is ``lambda``-
    independent, so freezing ``S`` does NOT freeze the c-axis: the natural-orbital
    factorisation of ``f = S_up^{1/2} lambda (S_dn^{1/2})^T`` stays differentiable w.r.t.
    ``lambda`` (with ``S`` fixed) exactly as the analytic
    ``mpc.grad_Vmpc_ee_wrt_c_AGP`` density-matrix VJP assumes.

    Forward VALUE is bit-identical to the direct call: concrete (untraced) leaves
    take the direct path unchanged. Traced leaves are REFUSED (loud, per the
    strict no-silent-fallback rule): the caller must HOIST the overlap out of
    the trace -- compute it eagerly from the CONCRETE ham and thread it down as
    ``frozen_overlap`` (``natural_orbital_density_coefficients_jnp`` /
    ``derivatives._auto._frozen_mpc_overlap``). A closure-captured concrete
    array is a constant under AD (zero tangent automatically), realising the
    same matched-approximation freeze WITHOUT any in-trace host callback (a
    ``pure_callback`` here would materialise the AO leaves and run the host
    Hermite integrator inside every jacrev evaluation -- a host-callback /
    memory load that can kill shared xdist workers).
    """
    leaves = jax.tree_util.tree_leaves(aos_data)
    if not any(isinstance(x, jax.core.Tracer) for x in leaves):
        return compute_overlap_matrix(aos_data)
    raise NotImplementedError(
        "compute_overlap_matrix is a host-side numpy kernel and cannot run on JAX tracers. "
        "Hoist it out of the trace: compute the overlap eagerly from the CONCRETE ham and pass "
        "it via natural_orbital_density_coefficients_jnp(frozen_overlap=(S_up, S_dn)) "
        "(threaded through jqmc.mpc.build_mpc_data). "
        "The frozen closure constant reproduces the matched-approximation coeff/occ freeze exactly."
    )


def _unwrap_aos_container(orb: _OrbitalDataT) -> AOs_base_data:
    """Return the AO container (leaf or composite) under ``orb``, unwrapping ``MOs_data``."""
    return orb.aos_data if isinstance(orb, MOs_data) else orb


def _gto_leaves_of(orb: AOs_base_data | MOs_data) -> list[AOs_GTO_base_data]:
    """GTO-family AO leaves owned by ``orb``, in canonical order.

    Handles a bare GTO leaf, a ``MOs_data`` over a GTO leaf, an
    :class:`AOs_composite_data` (its GTO/pGTO children in child order) and a
    ``MOs_data`` over a composite. Non-GTO children (plane-wave / B-spline)
    carry no primitive exponents / coefficients and contribute nothing.
    """
    aos = _unwrap_aos_container(orb)
    if isinstance(aos, AOs_GTO_base_data):
        return [aos]
    if isinstance(aos, AOs_composite_data):
        return [c for c in aos.children if isinstance(c, AOs_GTO_base_data)]
    return []


def _concat_gto_radial(
    orb: AOs_base_data | MOs_data,
    attr: Literal["exponents", "coefficients"],
) -> jax.Array:
    """Concatenate ``radial.<attr>`` (``exponents`` / ``coefficients``) over the
    GTO leaves of ``orb`` in canonical order.

    A single GTO leaf returns its own array; a composite (pgto(+)PW) returns the
    per-child GTO radial arrays concatenated in child order -- matching the
    analytic basis-grad layout (``derivatives.basis._basis_contract_*_general``
    iterates the same children and concatenates the same way). Raises when
    ``orb`` owns no GTO basis (pure PW / B-spline).
    """
    leaves = _gto_leaves_of(orb)
    if not leaves:
        raise NotImplementedError(
            f"basis-exponent/coefficient optimization is only defined for GTO-family AOs; "
            f"{type(_unwrap_aos_container(orb)).__name__} has no primitive exponents"
        )
    parts = [jnp.asarray(getattr(leaf.radial, attr)) for leaf in leaves]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)


def _replace_gto_radial(
    orb: _OrbitalDataT,
    new_vals: npt.ArrayLike | jax.Array,
    attr: Literal["exponents", "coefficients"],
) -> _OrbitalDataT:
    """Return ``orb`` with its GTO leaves' ``radial.<attr>`` overwritten from the
    flat concatenated ``new_vals`` (canonical GTO-leaf order).

    JAX-safe: the split sizes come from the original (static) leaf shapes, so
    ``new_vals`` may be a tracer (the in-trace analytic lambda-basis grad path).
    A single GTO leaf takes the whole vector; a composite splits it per GTO
    child (PW / B-spline children pass through unchanged).
    """
    aos = _unwrap_aos_container(orb)
    if not _gto_leaves_of(orb):
        raise NotImplementedError(
            f"basis-exponent/coefficient optimization is only defined for GTO-family AOs; "
            f"{type(aos).__name__} has no primitive exponents"
        )
    if isinstance(aos, AOs_GTO_base_data):
        new_aos = aos.replace(radial=aos.radial.replace(**{attr: new_vals}))
    else:  # AOs_composite_data
        new_children = []
        off = 0
        for c in aos.children:
            if isinstance(c, AOs_GTO_base_data):
                n = int(np.asarray(getattr(c.radial, attr)).shape[-1])
                new_children.append(c.replace(radial=c.radial.replace(**{attr: new_vals[off : off + n]})))
                off += n
            else:
                new_children.append(c)
        new_aos = aos.replace(children=tuple(new_children))
    return orb.replace(aos_data=new_aos) if isinstance(orb, MOs_data) else new_aos


def _apply_composite_leaf_basis_update(
    aos: AOs_composite_data,
    leaf_block: "VariationalParameterBlock",
) -> AOs_composite_data:
    """Split a leaf-level basis block (``basis_exp`` / ``basis_coeff``) across the
    composite's GTO children (canonical order) and let each child consume its
    slice via its own :meth:`apply_basis_block_update` (which casts dtype).
    """
    name = leaf_block.name
    if name.endswith("basis_exp"):
        attr = "exponents"
    elif name.endswith("basis_coeff"):
        attr = "coefficients"
    else:
        return aos
    vals = np.asarray(leaf_block.values)
    new_children = []
    off = 0
    for c in aos.children:
        if isinstance(c, AOs_GTO_base_data):
            n = int(np.asarray(getattr(c.radial, attr)).shape[-1])
            sub = vals[off : off + n]
            off += n
            child_block = leaf_block.replace(name=name, values=sub, shape=sub.shape, size=int(sub.size))
            new_children.append(c.apply_basis_block_update(child_block))
        else:
            new_children.append(c)
    return aos.replace(children=tuple(new_children))


# ---------------------------------------------------------------------------
# Overlap linear-algebra helper: Hermitian square root (S^{1/2}, S^{-1/2}) for
# the natural-orbital / MPC factorisation.
# ---------------------------------------------------------------------------


def _hermitian_sqrt(
    overlap: npt.NDArray,
    eig_clip_tol: float = AO_OVERLAP_EIGENVALUE_FLOOR,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build ``(S^{1/2}, S^{-1/2})`` for a Hermitian PSD overlap matrix.

    Consolidates the ``eigh + eigvecs @ diag(...) @ eigvecs.conj().T`` pattern
    repeated across four sites in the geminal pipeline:

    * :py:meth:`Geminal_data.convert_from_AOs_to_MOs`

    The shared edit is the Hermitian-conjugate ``eigvecs.conj().T`` that makes
    the square-root correct on a complex Bloch AO overlap. On a real symmetric
    input the ``conj`` is a no-op and the result is byte-identical to the real
    Gamma-point ``eigvecs @ diag(...) @ eigvecs.T`` form.

    Numerical-stability clip: periodic AO overlaps (PW, B-spline, composite,
    image-folded pGTO) are assembled from numerical integration / image sums
    and can carry tiny non-positive eigenvalues from round-off. Lift them to
    a small positive floor so ``S^{-1/2}`` stays finite without raising for
    what is really a positive-semidefinite operator.

    Args:
        overlap: Hermitian PSD overlap matrix ``(M, M)``. The caller is
            expected to have already enforced ``S = 0.5*(S + S^H)`` if a
            non-Hermitian round-off component is feared.
        eig_clip_tol: Floor for the eigenvalues before inversion.

    Returns:
        ``(sqrt_overlap, inv_sqrt_overlap)``, both ``(M, M)`` with the same
        dtype as ``overlap`` (real for real input, complex128 for complex
        input).
    """
    eigvals, eigvecs = np.linalg.eigh(overlap)
    eigvals = np.maximum(np.real(eigvals), eig_clip_tol)
    sqrt_overlap = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    inv_sqrt_overlap = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    return sqrt_overlap, inv_sqrt_overlap


def _magnitude_order(values: npt.NDArray) -> npt.NDArray[np.int64]:
    """Stable descending-``abs`` order for natural-orbital occupations."""
    return np.argsort(-np.abs(np.asarray(values)), kind="stable")


def _validate_factor_rank(target: int, available: int, n_up: int, *, context: str) -> int:
    """Validate an exact structural factor rank."""
    target = int(target)
    if target < int(n_up):
        raise ValueError(f"{context}: target rank {target} is below num_electron_up={n_up}.")
    if target > int(available):
        raise ValueError(f"{context}: target rank {target} exceeds the available rank {available}.")
    return target


def _embed_time_reversal_down_rows(
    grad_up: npt.ArrayLike | jax.Array,
    grad_dn: npt.ArrayLike | jax.Array,
    *,
    xp: ModuleType,
) -> npt.NDArray | jax.Array:
    """Embed a tied down-spin covector into the full up-spin row shape."""
    up_shape = tuple(grad_up.shape)
    dn_shape = tuple(grad_dn.shape)
    if len(up_shape) < 2 or len(dn_shape) != len(up_shape):
        raise ValueError(
            f"time-reversal MO gradients require matching ranks with trailing (MO, AO) axes; got up={up_shape}, dn={dn_shape}."
        )
    if up_shape[:-2] != dn_shape[:-2] or up_shape[-1] != dn_shape[-1]:
        raise ValueError(f"time-reversal MO gradients require identical sample and AO axes; got up={up_shape}, dn={dn_shape}.")
    missing_rows = up_shape[-2] - dn_shape[-2]
    if missing_rows < 0:
        raise ValueError(
            f"time-reversal down-spin gradient has more occupied rows than up-spin: up={up_shape[-2]}, dn={dn_shape[-2]}."
        )
    padding = [(0, 0)] * (len(dn_shape) - 2) + [
        (0, missing_rows),
        (0, 0),
    ]
    return xp.pad(grad_dn, padding)


def _require_time_reversal_arrays(
    paired: npt.NDArray,
    overlap_up: npt.NDArray,
    overlap_dn: npt.NDArray,
    *,
    context: str,
) -> None:
    """Enforce the declared time-reversal algebra before factorization."""
    if not np.allclose(overlap_dn, np.conj(overlap_up), atol=atol_consistency, rtol=rtol_consistency):
        raise ValueError(f"{context}: time-reversal overlap requires S_dn == conj(S_up).")
    if not np.allclose(paired, np.asarray(paired).conj().T, atol=atol_consistency, rtol=rtol_consistency):
        raise ValueError(f"{context}: declared time-reversal paired core must be Hermitian.")


def _require_time_reversal_mos(geminal: "_GeminalBase", paired: npt.NDArray, *, context: str) -> None:
    """Enforce the MO-factor and AO-overlap time-reversal identities."""
    C_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
    C_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients)
    if not np.allclose(C_dn, np.conj(C_up), atol=atol_consistency, rtol=rtol_consistency):
        raise ValueError(f"{context}: time-reversal factors require C_dn == conj(C_up).")
    S_up = np.asarray(compute_overlap_matrix(geminal.orb_data_up_spin.aos_data))
    S_dn = np.asarray(compute_overlap_matrix(geminal.orb_data_dn_spin.aos_data))
    S_up = 0.5 * (S_up + S_up.conj().T)
    S_dn = 0.5 * (S_dn + S_dn.conj().T)
    _require_time_reversal_arrays(paired, S_up, S_dn, context=context)


def _validate_magnitude_truncation(
    occupations: npt.NDArray,
    target: int,
    unpaired: npt.NDArray | None,
    *,
    context: str,
    require_sorted: bool,
) -> None:
    """Reject order-dependent or state-dropping factor truncations."""
    values = np.asarray(occupations)
    magnitudes = np.abs(values)
    if require_sorted and np.any(magnitudes[1:] > magnitudes[:-1] + atol_consistency):
        raise ValueError(f"{context}: occupations must be sorted by descending abs(lambda) before truncation.")
    if target < values.size and np.isclose(
        magnitudes[target - 1],
        magnitudes[target],
        atol=atol_consistency,
        rtol=rtol_consistency,
    ):
        raise ValueError(f"{context}: ambiguous truncation at equal-magnitude occupations.")
    if unpaired is not None and target < np.asarray(unpaired).shape[0]:
        discarded = np.asarray(unpaired)[target:]
        if not np.allclose(discarded, 0.0, atol=atol_consistency, rtol=rtol_consistency):
            raise ValueError(f"{context}: truncation would discard a non-negligible unpaired component.")


_SAME_TWIST_REGENERATE = (
    "Hamiltonians whose spin channels share one twist (the (+k, +k) convention for spin-polarized "
    "inputs) are not supported; regenerate the file from its QE / TREXIO source with the current converter."
)


def _require_opposite_spin_twists(orb_up: AOs_base_data | MOs_data, orb_dn: AOs_base_data | MOs_data) -> None:
    """Fail unless the down-spin channel is a Bloch basis at the opposite twist of the up channel.

    The spin channels always sit at opposite twists: the up channel at ``+k``
    and the down channel at ``-k`` (for an independent channel, the conjugate
    of its own ``+k`` orbitals; its basis need not equal the up basis). Each
    channel's twist is the common ``k_point`` of all its complex Bloch AO
    leaves, composite children included; it must be unique within the
    channel, and ``k_dn = -k_up``, both modulo integer reciprocal vectors.
    Each channel's leaves must also carry Bloch phase data that belong to
    their own twist, which is what can be checked from a single leaf: periodic-Gaussian image phases equal
    ``exp(2 pi i k . n)`` for the leaf's own ``k`` (so a ``+k`` Gaussian leaf
    merely relabelled ``-k`` fails). Plane-wave and spline leaves are Bloch
    functions of their labelled twist for any Miller list or control points,
    and tiled spline leaves validate their primitive-k unfolding in their own
    ``sanity_check``. Real / Gamma / molecular channels carry no twist.

    Raises:
        ValueError: If the channels are not an opposite-twist pair.
    """

    def fail(reason: str) -> NoReturn:
        raise ValueError(
            f"the down-spin channel is not at the opposite twist (+k, -k) of the up-spin channel: {reason}. "
            f"{_SAME_TWIST_REGENERATE}"
        )

    def same_twist(a: np.ndarray, b: np.ndarray) -> bool:
        # Twists are fractional reciprocal coordinates, defined modulo integer
        # reciprocal-lattice vectors (e.g. supercell images S k of one sector).
        difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return np.allclose(difference - np.rint(difference), 0.0, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT)

    def leaves(aos: AOs_base_data):
        if isinstance(aos, AOs_composite_data):
            for child in aos.children:
                yield from leaves(child)
        else:
            yield aos

    def phases_match_twist(leaf: AOs_base_data) -> bool:
        actual = np.asarray(leaf._image_phase_factors)
        if not leaf._image_indices:
            # Same rule as ``_promote_pgto_to_complex``: no image table, unit phases.
            expected = np.ones(int(leaf._n_image), dtype=np.complex128)
        else:
            indices = np.asarray(leaf._image_indices, dtype=np.float64).reshape(-1, 3)
            expected = np.exp(2j * np.pi * (indices @ np.asarray(leaf.k_point, dtype=np.float64)))
        return actual.shape == expected.shape and np.allclose(
            actual, expected, atol=GEMINAL_REPRESENTATION_ATOL, rtol=RTOL_BIT_EXACT
        )

    def channel_twist(orb: AOs_base_data | MOs_data, spin: str) -> np.ndarray | None:
        aos = orb.aos_data if isinstance(orb, MOs_data) else orb
        twist = None
        for leaf in leaves(aos):
            k_point = getattr(leaf, "k_point", None)
            if k_point is None:
                continue
            k = np.asarray(k_point, dtype=np.float64)
            if twist is None:
                twist = k
            elif not same_twist(k, twist):
                fail(
                    f"the {spin}-spin AO leaves carry different twists "
                    f"{tuple(float(c) for c in twist)} and {tuple(float(c) for c in k)}"
                )
            if hasattr(leaf, "_image_phase_factors") and not phases_match_twist(leaf):
                fail(
                    f"a {spin}-spin periodic-Gaussian leaf has image phases that do not belong to its twist "
                    f"{tuple(float(c) for c in k)}"
                )
        return twist

    k_up = channel_twist(orb_up, "up")
    k_dn = channel_twist(orb_dn, "down")
    if k_up is None and k_dn is None:
        return
    zero = np.zeros(3)
    k_up = zero if k_up is None else k_up
    k_dn = zero if k_dn is None else k_dn
    if not same_twist(k_dn, -k_up):
        fail(f"k_dn={tuple(float(c) for c in k_dn)} is not -k_up for k_up={tuple(float(c) for c in k_up)}")


def require_opposite_spin_twists(geminal: "AGP_data | AGPn_data | SD_data") -> None:
    """Apply the opposite-twist spin pairing check to a reconstructed geminal.

    Restart readers rebuild geminals without their validating constructors.
    This applies the same refusal to every geminal class: ordinary SD, AGP and
    AGPn through :func:`_require_opposite_spin_twists`, tiled SD through its
    ``(+k_s, -k_s)`` provider validation.

    Raises:
        ValueError: If the spin channels are not an opposite-twist pair.
    """
    if getattr(geminal, "is_tiled_representation", False):
        geminal.sanity_check()
        return
    if min(int(geminal.num_electron_up), int(geminal.num_electron_dn)) > 0:
        _require_opposite_spin_twists(geminal.orb_data_up_spin, geminal.orb_data_dn_spin)


# ---------------------------------------------------------------------------
# Geminal base mixin. Shared methods (AO-basis accessors, param-grad
# accumulation, strain, cart<->sphe conversion, projection, AGP<->MO reproject)
# for AGP_data / AGPn_data / SD_data.
# ---------------------------------------------------------------------------


class _GeminalBase:
    """Shared method mixin for ``AGP_data``, ``AGPn_data`` and ``SD_data``.

    Plain Python class (NOT a ``@struct.dataclass``) used as a method-only
    mixin for the three geminal flavours. Holds methods whose bodies depend
    only on fields that all three subclasses expose:
    ``orb_data_up_spin``, ``orb_data_dn_spin``, ``num_electron_up`` and
    ``num_electron_dn``. Lambda-dependent operations remain subtype-specific;
    canonical ``SD_data`` has no lambda API.

    Flax ``@struct.dataclass`` supports inheriting from a plain Python class
    that declares no data fields; the canonical mixin pattern.
    """

    # --- AO basis property accessors (for basis optimization) ---

    @property
    def ao_exponents_up(self) -> jax.Array:
        """AO Gaussian exponents for spin-up orbitals (jnp view of underlying numpy storage).

        GTO-family only. A single (p)GTO leaf returns its own exponents; a
        composite (pgto(+)PW) returns the concatenation of its GTO children's
        exponents in child order (PW / B-spline children contribute nothing).
        Pure PW / B-spline leaves have no primitive exponents and raise.
        """
        return _concat_gto_radial(self.orb_data_up_spin, "exponents")

    @property
    def ao_exponents_dn(self) -> jax.Array:
        """AO Gaussian exponents for spin-down orbitals (jnp view of underlying numpy storage).

        GTO-family only; see :attr:`ao_exponents_up` for the composite layout.
        """
        return _concat_gto_radial(self.orb_data_dn_spin, "exponents")

    @property
    def ao_coefficients_up(self) -> jax.Array:
        """AO contraction coefficients for spin-up orbitals (jnp view of underlying numpy storage).

        GTO-family only; see :attr:`ao_exponents_up` for the composite layout.
        """
        return _concat_gto_radial(self.orb_data_up_spin, "coefficients")

    @property
    def ao_coefficients_dn(self) -> jax.Array:
        """AO contraction coefficients for spin-down orbitals (jnp view of underlying numpy storage).

        GTO-family only; see :attr:`ao_exponents_up` for the composite layout.
        """
        return _concat_gto_radial(self.orb_data_dn_spin, "coefficients")

    def _replace_orb_radial(
        self,
        orb_data: _OrbitalDataT,
        new_vals: npt.ArrayLike | jax.Array,
        attr: Literal["exponents", "coefficients"],
    ) -> _OrbitalDataT:
        """Replace a GTO radial array (``attr`` = ``"exponents"`` / ``"coefficients"``).

        Handles a GTO AO / MO over GTO AO / composite. The flat ``new_vals`` is
        split per GTO child for a composite; PW / Spline leaves carry no such
        radial array (raises for a pure-PW orb).
        """
        return _replace_gto_radial(orb_data, new_vals, attr)

    def with_updated_ao_exponents(
        self, new_exp_up: npt.NDArray[np.float64], new_exp_dn: npt.NDArray[np.float64]
    ) -> "Geminal_data":
        """Return a new instance with updated AO exponents for both spins."""
        return self.replace(
            orb_data_up_spin=self._replace_orb_radial(self.orb_data_up_spin, new_exp_up, "exponents"),
            orb_data_dn_spin=self._replace_orb_radial(self.orb_data_dn_spin, new_exp_dn, "exponents"),
        )

    def with_updated_ao_coefficients(
        self, new_coeff_up: npt.NDArray[np.float64], new_coeff_dn: npt.NDArray[np.float64]
    ) -> "Geminal_data":
        """Return a new instance with updated AO contraction coefficients for both spins."""
        return self.replace(
            orb_data_up_spin=self._replace_orb_radial(self.orb_data_up_spin, new_coeff_up, "coefficients"),
            orb_data_dn_spin=self._replace_orb_radial(self.orb_data_dn_spin, new_coeff_dn, "coefficients"),
        )

    def _apply_basis_block_delegated(self, block: "VariationalParameterBlock", kind: str) -> "Geminal_data":
        """Delegate a ``lambda_basis_*`` block update to the AO leaves' protocol.

        Rather than unwrapping the up / dn AO leaves inline here and
        overwriting ``exponents`` / ``coefficients``, this
        (1) symmetrises the concatenated up+dn vector, (2) splits it
        into up / dn slices, (3) builds a leaf-level
        :class:`VariationalParameterBlock` named ``"basis_exp"`` /
        ``"basis_coeff"`` for each side, and (4) hands off to the AO
        leaf's :meth:`apply_basis_block_update`. The leaf protocol is
        defined on :class:`AOs_base_data`; GTO leaves implement it,
        future PW / Spline leaves inherit the no-op default.

        ``kind`` is ``"exp"`` or ``"coeff"`` --- the only difference
        between the two lambda-basis paths.
        """
        # No shell-sharing symmetriser: the nested radial leaf is already
        # de-duplicated to one DOF per shell-primitive.
        # Delegate the original values, not an ao_eval-quantized copy.
        # The receiving GTO leaf enforces fp64 model storage.
        vals = np.asarray(block.values)
        suffix = "basis_exp" if kind == "exp" else "basis_coeff"
        n_up = len(self.ao_exponents_up if kind == "exp" else self.ao_coefficients_up)
        up_vals, dn_vals = vals[:n_up], vals[n_up:]
        up_block = block.replace(name=suffix, values=up_vals, shape=up_vals.shape, size=int(up_vals.size))
        dn_block = block.replace(name=suffix, values=dn_vals, shape=dn_vals.shape, size=int(dn_vals.size))
        new_orb_up = self._delegate_leaf_basis_update(self.orb_data_up_spin, up_block)
        new_orb_dn = self._delegate_leaf_basis_update(self.orb_data_dn_spin, dn_block)
        return self.replace(orb_data_up_spin=new_orb_up, orb_data_dn_spin=new_orb_dn)

    @staticmethod
    def _delegate_leaf_basis_update(
        orb_data: _OrbitalDataT,
        leaf_block: "VariationalParameterBlock",
    ) -> _OrbitalDataT:
        """Call the AO leaf protocol on ``orb_data``, transparently unwrapping ``MOs_data``.

        Mirrors :meth:`_replace_orb_radial` in MOs_data handling:
        a ``MOs_data`` wraps an AO leaf; the update
        flows to the inner ``aos_data`` and the MOs_data is rebuilt with
        the new inner.
        """
        aos = _unwrap_aos_container(orb_data)
        if isinstance(aos, AOs_GTO_base_data):
            new_aos = aos.apply_basis_block_update(leaf_block)
        elif isinstance(aos, AOs_composite_data) and _gto_leaves_of(orb_data):
            new_aos = _apply_composite_leaf_basis_update(aos, leaf_block)
        else:
            raise NotImplementedError(
                f"basis-exponent/coefficient optimization is only defined for GTO-family AOs; "
                f"{type(aos).__name__} has no primitive exponents"
            )
        return orb_data.replace(aos_data=new_aos) if isinstance(orb_data, MOs_data) else new_aos

    def symmetrize_lambda(self, mat: npt.NDArray) -> npt.NDArray:
        """Symmetrize a lambda matrix according to the ``time_reversal`` flag.

        Type-level dispatch on ``self.time_reversal``:

        * ``True`` (closed-shell singlet, RHF/kRHF): enforce Hermitian
          (complex) / symmetric (real) on the paired block of ``mat``.
          ``0.5 * (A + A^H)``. Unpaired columns are left untouched.
        * ``False`` (open-shell / FM / UHF / kUHF): return ``mat``
          unchanged. The two channels are variationally independent and
          no Hermitian constraint applies.

        The dispatch is on the declared flag only; there is no runtime
        auto-detect of whether ``self.lambda_matrix`` happens to be
        symmetric within ``atol_consistency``.

        This method is the **single source of truth** for all symmetry
        operations on the lambda matrix.  Both
        :meth:`apply_block_update` (parameter enforcement) and the MCMC
        driver (SN-metric symmetrization, selection-mask expansion)
        use this method, so the symmetry logic is never duplicated.
        (The MCMC driver wraps it with flatten/unflatten in
        :meth:`~Wavefunction_data.get_variational_blocks`.)

        Args:
            mat: 2-D lambda matrix to symmetrize.

        Returns:
            Symmetrized matrix when ``time_reversal=True``, otherwise the
            input unchanged. 1-D or empty inputs are also returned
            unchanged.
        """
        if not self.time_reversal:
            return mat
        if self.lambda_matrix is None:
            return mat
        lam = np.asarray(self.lambda_matrix)
        if lam.ndim != 2:
            return mat
        # Hermitian when complex (time-reversal
        # pairing makes the paired block Hermitian rather than just
        # real-symmetric); bit-identical on the real path since
        # ``a.conj().T == a.T`` for real ``a``. The unpaired columns are
        # not touched in either case.
        _T = (lambda a: a.conj().T) if np.iscomplexobj(mat) else (lambda a: a.T)
        if lam.shape[0] == lam.shape[1]:
            return 0.5 * (mat + _T(mat))
        n_paired = lam.shape[0]
        out = mat.copy()
        p = out[:, :n_paired]
        out[:, :n_paired] = 0.5 * (p + _T(p))
        return out

    def accumulate_position_grad(self, grad_geminal: "Geminal_data") -> jax.Array:
        """Aggregate position gradients from geminal-related structures."""
        grad = 0.0
        if hasattr(grad_geminal, "orb_data_up_spin"):
            grad += grad_geminal.orb_data_up_spin.structure_data.positions
        if hasattr(grad_geminal, "orb_data_dn_spin"):
            grad += grad_geminal.orb_data_dn_spin.structure_data.positions
        return grad

    def accumulate_strain_grad(self, grad_geminal: "Geminal_data") -> jax.Array:
        r"""Aggregate strain cotangents from geminal-related structures.

        Each ``orb_data_*.structure_data`` carries cotangents for both
        ``positions`` (cell-AD: shape ``(N_atom, 3)``) and the three
        lattice rows ``vec_a/b/c`` (each shape ``(3,)``).
        """
        from .structure import _accumulate_strain_grad_per_leaf

        sigma_tilde = jnp.zeros((3, 3), dtype=jnp.float64)
        if hasattr(grad_geminal, "orb_data_up_spin"):
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_geminal.orb_data_up_spin.structure_data,
                self.orb_data_up_spin.structure_data,
            )
        if hasattr(grad_geminal, "orb_data_dn_spin"):
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_geminal.orb_data_dn_spin.structure_data,
                self.orb_data_dn_spin.structure_data,
            )
        return sigma_tilde

    def _apply_strain(self, F: jax.Array) -> "Geminal_data":
        """Return a new ``Geminal_data`` with ``F`` applied to every owned ``Structure_data``.

        Handles both pure-AO orbital containers (where ``structure_data``
        is a direct field on ``orb_data_*``) and MO orbital containers
        (where ``structure_data`` is a property that dereferences
        ``orb_data_*.aos_data.structure_data``). The actual AO strain is
        delegated to :py:func:`atomic_orbital._apply_strain_to_aos`, which
        recurses into :class:`AOs_composite_data` children (the composite
        exposes ``structure_data`` via property, not field, so a direct
        ``.replace(structure_data=...)`` raises ``TypeError`` --- see
        regression test ``test_stress_HF_composite_aos_kernel_runs``).
        """

        def _strain_orb(orb: _OrbitalDataT | None) -> _OrbitalDataT | None:
            if orb is None:
                return None
            if hasattr(orb, "aos_data"):
                return orb.replace(aos_data=_apply_strain_to_aos(orb.aos_data, F))
            return _apply_strain_to_aos(orb, F)

        kwargs = {}
        if hasattr(self, "orb_data_up_spin"):
            kwargs["orb_data_up_spin"] = _strain_orb(self.orb_data_up_spin)
        if hasattr(self, "orb_data_dn_spin"):
            kwargs["orb_data_dn_spin"] = _strain_orb(self.orb_data_dn_spin)
        return self.replace(**kwargs)

    def collect_param_grads(self, grad_geminal: "Geminal_data") -> dict[str, object]:
        """Collect parameter gradients into a flat dict keyed by block name.

        ``lambda_basis_exp`` / ``lambda_basis_coeff`` blocks require a
        GTO-family basis. A single (p)GTO leaf contributes its radial
        exponents / coefficients; a composite (pgto(+)PW) contributes its
        GTO children's radial params concatenated in child order (PW /
        B-spline children contribute nothing). The blocks are omitted when a
        spin channel owns no GTO basis (pure PW / B-spline) so callers do not
        attempt to optimize non-existent basis parameters.
        """
        grads: dict[str, any] = {}
        if hasattr(grad_geminal, "lambda_matrix"):
            grads["lambda_matrix"] = grad_geminal.lambda_matrix
        # AO basis gradients (up + dn concatenated along the parameter axis).
        # Meaningful when each spin channel carries at least one GTO-family leaf
        # (a single (p)gto leaf or a composite with >=1 (p)gto child).
        if _gto_leaves_of(self.orb_data_up_spin) and _gto_leaves_of(self.orb_data_dn_spin):
            grads["lambda_basis_exp"] = jnp.concatenate([grad_geminal.ao_exponents_up, grad_geminal.ao_exponents_dn], axis=-1)
            grads["lambda_basis_coeff"] = jnp.concatenate(
                [grad_geminal.ao_coefficients_up, grad_geminal.ao_coefficients_dn], axis=-1
            )
        return grads

    def to_cartesian(self) -> "Geminal_data":
        """Convert spherical orbitals to Cartesian and transform the lambda matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_cartesian``
        (the lambda matrix is unchanged). For canonical ``SD_data``, convert
        the two occupied-MO objects and preserve the lambda-free SD subtype.
        For Cartesian inputs, return self.
        """
        if isinstance(self, SD_data):
            converted = self.replace(
                orb_data_up_spin=self.orb_data_up_spin.to_cartesian(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_cartesian(),
            )
            converted.sanity_check()
            return converted
        if isinstance(self.orb_data_up_spin, MOs_data) or isinstance(self.orb_data_dn_spin, MOs_data):
            if not (isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)):
                raise ValueError("Cartesian conversion requires both spin channels to use MOs or both to use AOs.")
            return AGP_data(
                num_electron_up=self.num_electron_up,
                num_electron_dn=self.num_electron_dn,
                orb_data_up_spin=self.orb_data_up_spin.to_cartesian(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_cartesian(),
                lambda_matrix=self.lambda_matrix,
                time_reversal=self.time_reversal,
            )

        if isinstance(self.orb_data_up_spin, AOs_cart_gto_data) and isinstance(self.orb_data_dn_spin, AOs_cart_gto_data):
            return self
        if not isinstance(self.orb_data_up_spin, (AOs_sphe_gto_data, AOs_cart_gto_data)) or not isinstance(
            self.orb_data_dn_spin, (AOs_sphe_gto_data, AOs_cart_gto_data)
        ):
            raise ValueError(
                "Cartesian/spherical conversion is only defined for GTO-type AOs (or MOs over GTO AOs); "
                "PW/Spline/composite AO leaves have no cart/sphe distinction."
            )
        aos_up_cart, transform_up = _aos_sphe_to_cart(self.orb_data_up_spin)
        aos_dn_cart, transform_dn = _aos_sphe_to_cart(self.orb_data_dn_spin)

        lambda_matrix = np.asarray(
            self.lambda_matrix
        )  # dtype-preserving (a float64 cast would raise on a complex-twist lambda)
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(lambda_matrix, [self.orb_num_dn])
        lambda_paired_cart = transform_up.T @ lambda_matrix_paired @ transform_dn
        lambda_unpaired_cart = transform_up.T @ lambda_matrix_unpaired
        lambda_cart = np.hstack([lambda_paired_cart, lambda_unpaired_cart])
        lambda_cart = lambda_cart.astype(np.asarray(self.lambda_matrix).dtype, copy=False)

        return AGP_data(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=aos_up_cart,
            orb_data_dn_spin=aos_dn_cart,
            lambda_matrix=lambda_cart,
            time_reversal=self.time_reversal,
        )

    def to_spherical(self) -> "Geminal_data":
        """Convert Cartesian orbitals to spherical and transform the lambda matrix.

        Raw-AO coefficients use the complete molecular overlap metric or the
        periodic Bloch metric ``S(k)``; the same global reverse maps are
        applied to the paired and unpaired lambda blocks.
        If the underlying orbitals are MOs, defer to ``MOs_data.to_spherical``
        (the lambda matrix is unchanged). For canonical ``SD_data``, convert
        the two occupied-MO objects and preserve the lambda-free SD subtype.
        For spherical inputs, return self.
        """
        if isinstance(self, SD_data):
            converted = self.replace(
                orb_data_up_spin=self.orb_data_up_spin.to_spherical(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_spherical(),
            )
            converted.sanity_check()
            return converted
        if isinstance(self.orb_data_up_spin, MOs_data) or isinstance(self.orb_data_dn_spin, MOs_data):
            if not (isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)):
                raise ValueError("Spherical conversion requires both spin channels to use MOs or both to use AOs.")
            return AGP_data(
                num_electron_up=self.num_electron_up,
                num_electron_dn=self.num_electron_dn,
                orb_data_up_spin=self.orb_data_up_spin.to_spherical(),
                orb_data_dn_spin=self.orb_data_dn_spin.to_spherical(),
                lambda_matrix=self.lambda_matrix,
                time_reversal=self.time_reversal,
            )

        if isinstance(self.orb_data_up_spin, AOs_sphe_gto_data) and isinstance(self.orb_data_dn_spin, AOs_sphe_gto_data):
            return self
        if not isinstance(self.orb_data_up_spin, (AOs_sphe_gto_data, AOs_cart_gto_data)) or not isinstance(
            self.orb_data_dn_spin, (AOs_sphe_gto_data, AOs_cart_gto_data)
        ):
            raise ValueError(
                "Spherical/cartesian conversion is only defined for GTO-type AOs (or MOs over GTO AOs); "
                "PW/Spline/composite AO leaves have no cart/sphe distinction."
            )
        aos_up_sphe, transform_up = _aos_cart_to_sphe(self.orb_data_up_spin)
        aos_dn_sphe, transform_dn = _aos_cart_to_sphe(self.orb_data_dn_spin)

        lambda_matrix = np.asarray(
            self.lambda_matrix
        )  # dtype-preserving (a float64 cast would raise on a complex-twist lambda)
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(lambda_matrix, [self.orb_num_dn])
        lambda_paired_sph = transform_up.T @ lambda_matrix_paired @ transform_dn
        lambda_unpaired_sph = transform_up.T @ lambda_matrix_unpaired
        lambda_sph = np.hstack([lambda_paired_sph, lambda_unpaired_sph])
        storage_dtype = np.result_type(
            np.asarray(self.lambda_matrix).dtype,
            np.asarray(transform_up).dtype,
            np.asarray(transform_dn).dtype,
        )
        lambda_sph = lambda_sph.astype(storage_dtype, copy=False)

        return AGP_data(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=aos_up_sphe,
            orb_data_dn_spin=aos_dn_sphe,
            lambda_matrix=lambda_sph,
            time_reversal=self.time_reversal,
        )

    @classmethod
    def convert_from_MOs_to_AOs(cls, geminal_data: "Geminal_data") -> "Geminal_data":
        """Convert MOs to AOs.

        AO-type independent: works for any AO leaf (GTO, periodic pGTO,
        plane-wave, B-spline, composite) since the transformation only
        uses ``mo_coefficients`` (no GTO-specific kernels).

        Always returns an :class:`AGP_data`. The output of an AO/MO
        conversion is intrinsically a dense-lambda AGP shape, so the
        classmethod hardcodes the constructor rather than dispatching on
        ``cls`` (which would not match the ``AGPn_data`` / ``SD_data``
        signatures).
        """
        if _is_ao_leaf(geminal_data.orb_data_up_spin) and _is_ao_leaf(geminal_data.orb_data_dn_spin):
            return geminal_data
        if isinstance(geminal_data, SD_data):
            return geminal_data.to_AGP()
        if isinstance(geminal_data.orb_data_up_spin, MOs_data) and isinstance(geminal_data.orb_data_dn_spin, MOs_data):
            # split mo_lambda_matrix
            mo_lambda_matrix_paired, mo_lambda_matrix_unpaired = np.hsplit(
                geminal_data.lambda_matrix, [geminal_data.orb_num_dn]
            )

            # extract AOs data
            aos_data_up_spin = geminal_data.orb_data_up_spin.aos_data
            aos_data_dn_spin = geminal_data.orb_data_dn_spin.aos_data

            # convert MOs lambda to AO lambda
            aos_lambda_matrix_paired = np.dot(
                geminal_data.orb_data_up_spin.mo_coefficients.T,
                np.dot(mo_lambda_matrix_paired, geminal_data.orb_data_dn_spin.mo_coefficients),
            )
            aos_lambda_matrix_unpaired = np.dot(geminal_data.orb_data_up_spin.mo_coefficients.T, mo_lambda_matrix_unpaired)
            aos_lambda_matrix = np.hstack([aos_lambda_matrix_paired, aos_lambda_matrix_unpaired])
            return AGP_data(
                geminal_data.num_electron_up,
                geminal_data.num_electron_dn,
                aos_data_up_spin,
                aos_data_dn_spin,
                aos_lambda_matrix,
                geminal_data.time_reversal,
            )
        raise NotImplementedError

    @classmethod
    def convert_from_AOs_to_MOs(
        cls,
        geminal_data: "Geminal_data",
        num_eigenvectors: int | str = "all",
    ) -> "Geminal_data":
        r"""Factor an AO geminal in an S-orthonormal MO basis.

        Args:
            geminal_data: Input geminal representation.
            num_eigenvectors: ``"all"`` for a full AO-space factorization. If
                an integer ``N`` is supplied, construct exactly ``N`` factor
                rows, including the rows required by an open-shell unpaired
                block.

        Returns:
            Geminal_data in MO representation. Always an :class:`AGP_data`;
            AO/MO conversion has a dense-lambda AGP shape.

        Notes:
            Retained factors are ordered by decreasing occupation magnitude.
            The detailed natural-orbital factorization is work in progress.
        """
        if isinstance(geminal_data.orb_data_up_spin, MOs_data) and isinstance(geminal_data.orb_data_dn_spin, MOs_data):
            return geminal_data

        if not (_is_ao_leaf(geminal_data.orb_data_up_spin) and _is_ao_leaf(geminal_data.orb_data_dn_spin)):
            raise NotImplementedError

        aos_data_up_spin = geminal_data.orb_data_up_spin
        aos_data_dn_spin = geminal_data.orb_data_dn_spin

        if aos_data_up_spin.num_ao != aos_data_dn_spin.num_ao:
            raise ValueError("AO->MO conversion currently requires identical up/down AO dimensions.")

        if type(aos_data_up_spin) is not type(aos_data_dn_spin):
            raise ValueError("AO->MO conversion requires matching AO basis types for up/down channels.")

        ao_dim = aos_data_up_spin.num_ao
        ne_up = int(geminal_data.num_electron_up)
        ne_dn = int(geminal_data.num_electron_dn)
        num_unpaired = ne_up - ne_dn  # >= 0

        if isinstance(num_eigenvectors, str):
            if num_eigenvectors != "all":
                raise ValueError(f"num_eigenvectors must be 'all' or int, got {num_eigenvectors}.")
            num_mo = ao_dim
            use_truncated_mode = False
        elif isinstance(num_eigenvectors, (int, np.integer)):
            requested_num_mo = int(num_eigenvectors)
            if requested_num_mo <= 0:
                raise ValueError(f"num_eigenvectors={requested_num_mo} must be positive.")
            num_mo = _validate_factor_rank(
                requested_num_mo,
                ao_dim,
                ne_up,
                context="AO->MO conversion",
            )
            use_truncated_mode = num_mo < ao_dim
            num_truncated = ao_dim - num_mo
            if use_truncated_mode:
                logger.info(
                    "AO->MO truncated mode: kept %d factor rows out of %d (truncated %d).",
                    num_mo,
                    ao_dim,
                    num_truncated,
                )
        else:
            raise ValueError(f"num_eigenvectors must be 'all' or int, got {type(num_eigenvectors)}.")

        _validate_factor_rank(num_mo, ao_dim, ne_up, context="AO->MO conversion")

        ao_lambda_matrix_paired, ao_lambda_matrix_unpaired = np.hsplit(
            np.asarray(geminal_data.lambda_matrix), [geminal_data.orb_num_dn]
        )
        # lambda dtype follows the stored
        # matrix (complex Hermitian at non-Gamma; real at Gamma).
        # Bit-identical on the real path.
        _lambda_dtype = np.asarray(geminal_data.lambda_matrix).dtype
        ao_lambda_matrix_paired = np.asarray(ao_lambda_matrix_paired, dtype=_lambda_dtype)
        ao_lambda_matrix_unpaired = np.asarray(ao_lambda_matrix_unpaired, dtype=_lambda_dtype)

        # The overlap dtype follows the AO leaf (complex for Bloch pGTO at
        # non-Gamma; real otherwise). Dropping the float64 hardcast is the
        # complex-extension entry point.
        overlap_up = np.asarray(compute_overlap_matrix(aos_data_up_spin))
        overlap_dn = np.asarray(compute_overlap_matrix(aos_data_dn_spin))

        # Hermitian symmetrise (complex-aware).
        overlap_up = 0.5 * (overlap_up + overlap_up.conj().T)
        overlap_dn = 0.5 * (overlap_dn + overlap_dn.conj().T)

        # Numerical-stability clip for S^{1/2} / S^{-1/2}: periodic AO
        # overlaps (PW, B-spline, composite, image-folded pGTO) are assembled
        # from numerical integration / image sums and can carry tiny non-
        # positive eigenvalues from round-off. The shared ``_hermitian_sqrt``
        # helper applies the eig_clip_tol floor.
        eig_clip_tol = AO_OVERLAP_EIGENVALUE_FLOOR
        # Pre-eigh diagnostic on the clip; the helper applies the floor.
        # Note: eigh returns real eigenvalues for Hermitian input regardless
        # of whether the storage dtype is real or complex.
        _min_eig_up_pre = float(np.real(np.linalg.eigvalsh(overlap_up)[0]))
        _min_eig_dn_pre = float(np.real(np.linalg.eigvalsh(overlap_dn)[0]))
        if _min_eig_up_pre < eig_clip_tol:
            logger.debug(
                "AO->MO: clipping up-spin overlap eigenvalues below %.1e (min=%.3e).",
                eig_clip_tol,
                _min_eig_up_pre,
            )
        if _min_eig_dn_pre < eig_clip_tol:
            logger.debug(
                "AO->MO: clipping dn-spin overlap eigenvalues below %.1e (min=%.3e).",
                eig_clip_tol,
                _min_eig_dn_pre,
            )

        sqrt_overlap_up, inv_sqrt_overlap_up = _hermitian_sqrt(overlap_up, eig_clip_tol)
        sqrt_overlap_dn, inv_sqrt_overlap_dn = _hermitian_sqrt(overlap_dn, eig_clip_tol)

        # The AO geminal is bilinear, so the right Lowdin factor carries a
        # transpose, not an adjoint.  With row-oriented MO storage this gives
        # lambda_AO = S_up^-1/2 B S_dn^-T/2.
        f_biorth_repr = sqrt_overlap_up @ ao_lambda_matrix_paired @ sqrt_overlap_dn.T
        time_reversal = bool(geminal_data.time_reversal)
        if time_reversal:
            _require_time_reversal_arrays(
                ao_lambda_matrix_paired,
                overlap_up,
                overlap_dn,
                context="AO->MO conversion",
            )
            f_biorth_repr = 0.5 * (f_biorth_repr + f_biorth_repr.conj().T)
            all_evals, all_vectors_up = np.linalg.eigh(f_biorth_repr)
            order = _magnitude_order(all_evals)
            all_evals = np.asarray(all_evals[order], dtype=np.float64)
            all_vectors_up = np.asarray(all_vectors_up[:, order])
            all_vectors_dn = np.conj(all_vectors_up)
        else:
            all_vectors_up, all_evals, v_h = np.linalg.svd(f_biorth_repr, full_matrices=False)
            all_evals = np.asarray(all_evals, dtype=np.float64)
            # B = U diag(s) Vh and C_dn = Vh S_dn^-T/2, hence the column
            # vectors used by the row-oriented builder are Vh.T.
            all_vectors_dn = np.asarray(v_h.T)

        n_paired_keep = num_mo
        if num_unpaired > 0 and use_truncated_mode:
            n_paired_keep = num_mo - num_unpaired
            if n_paired_keep < ne_dn:
                raise ValueError(
                    f"AO->MO conversion: target rank leaves fewer paired rows ({n_paired_keep}) than num_electron_dn={ne_dn}."
                )
        if use_truncated_mode:
            _validate_magnitude_truncation(
                all_evals,
                n_paired_keep,
                None,
                context="AO->MO conversion",
                require_sorted=False,
            )
        selected_evals = np.asarray(all_evals[:n_paired_keep], dtype=np.float64)
        selected_vectors_up = np.asarray(all_vectors_up[:, :n_paired_keep])
        selected_vectors_dn = np.asarray(all_vectors_dn[:, :n_paired_keep])

        if use_truncated_mode:
            discarded_evals = all_evals[n_paired_keep:]
            discarded_norm = float(np.linalg.norm(discarded_evals))
            logger.debug(
                "AO->MO factor truncation: discarded %d paired factor values of "
                "B = S_up^{1/2} lambda S_dn^{T/2}; "
                "discarded-value norm = %.6e.",
                discarded_evals.size,
                discarded_norm,
            )
            logger.debug(
                "[MOOPT-TRACE] AO->MO retained paired factor values: %s",
                np.array2string(selected_evals, precision=10, separator=", "),
            )

        # ------------------------------------------------------------------
        # Augment up-spin MO basis with unpaired directions  (Ne_up > Ne_dn)
        #
        # A fixed-rank open-shell representation reserves one factor row per
        # excess-up column.  Those rows are constructed from the unpaired AO
        # block after projection against *every* retained paired row.  This
        # avoids both an arbitrary null-space orientation and a rank loss in
        # the unpaired Slater block.
        # ------------------------------------------------------------------
        if num_unpaired > 0 and use_truncated_mode:
            orth_unpaired = sqrt_overlap_up @ ao_lambda_matrix_unpaired
            if selected_vectors_up.shape[1] > 0:
                orth_unpaired -= selected_vectors_up @ (selected_vectors_up.conj().T @ orth_unpaired)

            # Keep every independent residual unpaired direction first, then
            # fill any already-contained column with discarded paired/null
            # vectors.  Every candidate is projected against all retained
            # rows, not only the occupied subset.
            u_unpaired, s_unpaired, _ = np.linalg.svd(orth_unpaired, full_matrices=False)
            scale = float(s_unpaired[0]) if s_unpaired.size else 0.0
            rank_tol = max(atol_consistency, rtol_consistency * scale)
            residual_rank = int(np.count_nonzero(s_unpaired > rank_tol))
            appended = np.asarray(u_unpaired[:, :residual_rank])
            need = num_unpaired - residual_rank
            if need > 0:
                fallback = np.asarray(all_vectors_up[:, n_paired_keep:])
                basis = np.hstack([selected_vectors_up, appended])
                if basis.shape[1] > 0:
                    fallback = fallback - basis @ (basis.conj().T @ fallback)
                q_fallback, r_fallback = np.linalg.qr(fallback, mode="reduced")
                usable = np.abs(np.diag(r_fallback)) > rank_tol
                q_fallback = q_fallback[:, usable]
                if q_fallback.shape[1] < need:
                    raise ValueError("AO->MO conversion: unpaired augmentation is rank deficient.")
                appended = np.hstack([appended, q_fallback[:, :need]])

            selected_vectors_up = np.hstack([selected_vectors_up, appended[:, :num_unpaired]])
            if time_reversal:
                selected_vectors_dn = np.conj(selected_vectors_up)
            else:
                selected_vectors_dn = np.hstack([selected_vectors_dn, all_vectors_dn[:, n_paired_keep:num_mo]])
            selected_evals = np.concatenate([selected_evals, np.zeros(num_unpaired, dtype=np.float64)])
            logger.info(
                "AO->MO: augmented up-spin MO basis with %d unpaired-derived vectors after %d retained paired/null-space rows.",
                num_unpaired,
                n_paired_keep,
            )

        p_matrix_cols_up = inv_sqrt_overlap_up @ selected_vectors_up
        p_matrix_cols_dn = inv_sqrt_overlap_dn @ selected_vectors_dn
        mo_coefficients_up = p_matrix_cols_up.T
        mo_coefficients_dn = p_matrix_cols_dn.T

        mo_lambda_matrix_paired = np.diag(selected_evals)
        mo_lambda_matrix_unpaired = mo_coefficients_up.conj() @ overlap_up @ ao_lambda_matrix_unpaired

        mo_lambda_matrix = np.hstack([mo_lambda_matrix_paired, mo_lambda_matrix_unpaired])

        mos_data_up_spin = MOs_data(num_mo=num_mo, aos_data=aos_data_up_spin, mo_coefficients=mo_coefficients_up)
        mos_data_dn_spin = MOs_data(num_mo=num_mo, aos_data=aos_data_dn_spin, mo_coefficients=mo_coefficients_dn)

        # <psi_i|psi_j> = sum_ab conj(C_ia)
        # S_ab C_jb is sesquilinear -- the orthonormality check is
        # (C.conj()) @ S @ C.T. Bit-identical on the real path
        # (``.conj()`` is identity for float64).
        psp_up = mo_coefficients_up.conj() @ overlap_up @ mo_coefficients_up.T
        psp_dn = mo_coefficients_dn.conj() @ overlap_dn @ mo_coefficients_dn.T
        identity = np.eye(num_mo, dtype=psp_up.dtype)
        psp_up_diff = psp_up - identity
        psp_dn_diff = psp_dn - identity

        up_frob_norm = float(np.linalg.norm(psp_up_diff))
        up_max_abs_diff = float(np.max(np.abs(psp_up_diff))) if psp_up_diff.size else 0.0
        up_is_orthonormal = bool(np.allclose(psp_up, identity, atol=atol_consistency, rtol=rtol_consistency))

        dn_frob_norm = float(np.linalg.norm(psp_dn_diff))
        dn_max_abs_diff = float(np.max(np.abs(psp_dn_diff))) if psp_dn_diff.size else 0.0
        dn_is_orthonormal = bool(np.allclose(psp_dn, identity, atol=atol_consistency, rtol=rtol_consistency))

        logger.debug(
            "[MOOPT-TRACE] AO->MO orthogonality check (up): allclose(C*SC^T, I)=%s, ||C*SC^T-I||_F=%.3e, max|C*SC^T-I|=%.3e",
            up_is_orthonormal,
            up_frob_norm,
            up_max_abs_diff,
        )
        logger.debug(
            "[MOOPT-TRACE] AO->MO orthogonality check (dn): allclose(C*SC^T, I)=%s, ||C*SC^T-I||_F=%.3e, max|C*SC^T-I|=%.3e",
            dn_is_orthonormal,
            dn_frob_norm,
            dn_max_abs_diff,
        )
        if not up_is_orthonormal or not dn_is_orthonormal:
            raise FloatingPointError(
                f"AO->MO conversion failed S-orthonormality: up residual={up_frob_norm:.3e}, dn residual={dn_frob_norm:.3e}."
            )

        return AGP_data(
            num_electron_up=geminal_data.num_electron_up,
            num_electron_dn=geminal_data.num_electron_dn,
            orb_data_up_spin=mos_data_up_spin,
            orb_data_dn_spin=mos_data_dn_spin,
            lambda_matrix=mo_lambda_matrix,
            time_reversal=geminal_data.time_reversal,
        )


# ---------------------------------------------------------------------------
# AGP_data -- pairing-matrix (lambda) geminal in AO or MO representation.
# Per-subclass mixin methods (_get_info / _logger_info / _lambda_matrix_jnp)
# are the cross-module accessors described in the naming convention above.
# ---------------------------------------------------------------------------


# @dataclass
@struct.dataclass
class AGP_data(_GeminalBase):
    """Geminal (AGP) parameters and orbital references.

    Args:
        num_electron_up (int): Number of spin-up electrons.
        num_electron_dn (int): Number of spin-down electrons.
        orb_data_up_spin (AOs_data | MOs_data): Basis/orbitals for spin-up electrons.
        orb_data_dn_spin (AOs_data | MOs_data): Basis/orbitals for spin-down electrons.
        lambda_matrix (npt.NDArray[np.float64]): Geminal pairing matrix with shape
            ``(orb_num_up, orb_num_dn + num_electron_up - num_electron_dn)``. dtype: float64.

    Notes:
        - For closed shells, ``orb_num_up == orb_num_dn`` and ``lambda_matrix`` is square.
        - For open shells, the right block encodes unpaired spin-up orbitals.
    """

    num_electron_up: int = struct.field(pytree_node=False, default=0)  #: Number of spin-up electrons.
    num_electron_dn: int = struct.field(pytree_node=False, default=0)  #: Number of spin-down electrons.
    orb_data_up_spin: AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data = struct.field(
        pytree_node=True, default_factory=lambda: AOs_sphe_gto_data()
    )  #: Orbital data (AOs or MOs) for spin-up electrons.
    orb_data_dn_spin: AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data = struct.field(
        pytree_node=True, default_factory=lambda: AOs_sphe_gto_data()
    )  #: Orbital data (AOs or MOs) for spin-down electrons.
    lambda_matrix: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )  #: Geminal pairing matrix; see class notes for expected shape. dtype: float64.
    time_reversal: bool = struct.field(pytree_node=False, default=True)
    """Coefficient-tying flag (restricted vs independent spin channels),
    consumed by ``symmetrize_lambda`` and the optimizer. It does not select
    the spin twists: at a complex Bloch twist the down-spin orbitals are
    always the ``with_negated_k`` (``-k``) partner of their own channel's
    ``+k`` orbitals, for either value (enforced by ``sanity_check``).

    * ``True``: closed-shell singlet (RHF / kRHF). ``lambda_paired`` is
      Hermitian (complex) / symmetric (real). Requires
      ``num_electron_up == num_electron_dn``.
    * ``False``: open-shell / FM / broken-symmetry (UHF / kUHF). The two
      channels are variationally independent; no Hermitian constraint on
      ``lambda_paired``.
    """

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.orb_num_up != self.orb_num_dn:
            raise ValueError(
                f"The number of up and down orbitals ({self.orb_num_up}, {self.orb_num_dn}) should be the same such that the lambda_matrix is square."
            )
        if self.lambda_matrix.shape != (
            self.orb_num_up,
            self.orb_num_dn + (self.num_electron_up - self.num_electron_dn),
        ):
            raise ValueError(
                f"dim. of lambda_matrix = {self.lambda_matrix.shape} is incompatible with the expected one "
                + f"= ({self.orb_num_up}, {self.orb_num_dn + (self.num_electron_up - self.num_electron_dn)}).",
            )
        if not isinstance(self.num_electron_up, (int, np.integer)):
            raise ValueError(f"num_electron_up = {type(self.num_electron_up)} must be an int.")
        if not isinstance(self.num_electron_dn, (int, np.integer)):
            raise ValueError(f"num_electron_dn = {type(self.num_electron_dn)} must be an int.")

        # ``time_reversal=True`` covers both the closed-shell singlet (RHF /
        # kRHF, ``N_up == N_dn``) and the restricted open-shell (ROHF /
        # kROHF, ``N_up > N_dn``) cases. In ROHF the paired core retains
        # the TR-paired Restricted structure (mo_dn = conj(mo_up) on the
        # first ``N_dn`` MOs) while the singly-occupied alpha orbitals live
        # in ``mo_up[N_dn:N_up]`` and enter ``lambda_unpaired`` only -- the
        # dn channel never reads them. The truly unrestricted case (UHF /
        # kUHF) sets ``time_reversal=False`` because alpha and beta are
        # variationally independent. The two open-shell cases share the
        # same electron-count pattern and cannot be distinguished here;
        # the loader / caller chooses ``time_reversal`` per its provenance.
        if self.time_reversal and self.num_electron_up < self.num_electron_dn:
            raise ValueError(
                f"time_reversal=True with num_electron_up < num_electron_dn "
                f"({self.num_electron_up} < {self.num_electron_dn}) is not supported; "
                f"the alpha-major (``N_up >= N_dn``) convention is assumed throughout "
                f"the geminal kernel (paired block dimensioned by ``N_dn``, "
                f"unpaired columns carry the alpha excess)."
            )

        # dtype-consistency rule: the lambda matrix
        # must match the orbital leaves' real/complex regime, and the two
        # spin sectors must agree.
        from .atomic_orbital import is_complex_aos as _is_complex_aos

        def _orb_is_complex(orb: AOs_base_data | MOs_data) -> bool:
            if _is_complex_aos(orb):
                return True
            inner_aos = getattr(orb, "aos_data", None)
            return inner_aos is not None and _is_complex_aos(inner_aos)

        up_cplx = _orb_is_complex(self.orb_data_up_spin)
        dn_cplx = _orb_is_complex(self.orb_data_dn_spin)
        if up_cplx != dn_cplx:
            raise ValueError(
                f"spin sectors disagree on twist regime: up complex={up_cplx}, dn complex={dn_cplx} "
                "(both must be real or both must be complex Bloch leaves at opposite twists)."
            )
        lam_dtype = np.asarray(self.lambda_matrix).dtype
        if up_cplx:
            if lam_dtype != np.complex128:
                raise ValueError(f"complex orbital leaves require complex128 lambda_matrix, got dtype = {lam_dtype}.")
        else:
            if not np.issubdtype(lam_dtype, np.floating):
                raise ValueError(f"real orbital leaves require real lambda_matrix, got dtype = {lam_dtype}.")

        if min(int(self.num_electron_up), int(self.num_electron_dn)) > 0:
            # An empty spin channel enters no determinant: nothing to pair.
            _require_opposite_spin_twists(self.orb_data_up_spin, self.orb_data_dn_spin)
        self.orb_data_up_spin.sanity_check()
        self.orb_data_dn_spin.sanity_check()

    @classmethod
    def build(
        cls,
        *,
        num_electron_up: int,
        num_electron_dn: int,
        orb_data_up_spin: AOs_base_data | MOs_data,
        orb_data_dn_spin: AOs_base_data | MOs_data,
        lambda_matrix: npt.ArrayLike,
        time_reversal: bool | None = None,
    ) -> Self:
        """Build a validated :class:`Geminal_data` from orbital data and a pairing matrix.

        Blessed factory: coerces ``lambda_matrix`` to the ``det_eval``
        zone dtype, constructs the instance, and runs
        :py:meth:`sanity_check`. Orbital containers may be AO leaves /
        composites or :class:`MOs_data`. Direct construction
        (``Geminal_data(...)``) remains supported.

        Zone-aware storage plus zone-aware update keeps the JIT cache
        key stable between iter 1 (fresh init) and iter 2+ (post-SR
        step), since :py:meth:`apply_block_update` writes the updated
        ``lambda_matrix`` in the same zone dtype.

        Args:
            num_electron_up: Number of up-spin electrons.
            num_electron_dn: Number of down-spin electrons.
            orb_data_up_spin: Up-spin orbital data (AO leaf / composite or :class:`MOs_data`).
            orb_data_dn_spin: Down-spin orbital data (AO leaf / composite or :class:`MOs_data`).
            lambda_matrix: Pairing matrix in the basis of ``orb_data_*``; coerced to the
                ``det_eval`` zone dtype.
            time_reversal: Coefficient-tying flag (see ``AGP_data.time_reversal``). When ``None``
                (default), inferred from
                the electron counts: ``True`` if
                ``num_electron_up == num_electron_dn`` (closed-shell),
                ``False`` otherwise. Pass ``True`` / ``False`` explicitly
                to override (e.g., broken-symmetry AGP on a closed-shell
                system).
        """
        # Keep the storage dtype rule zone-aware
        # for the real path (so fp32 mixed mode is unchanged) AND lift to
        # ``complex128`` when the caller already provides a complex lambda
        # (the TREXIO loader at a non-Gamma twist + the Hermitian paired
        # block).
        _src = np.asarray(lambda_matrix)
        _dt = np.complex128 if np.iscomplexobj(_src) else get_dtype_np("det_eval")
        if time_reversal is None:
            # Closed-shell singlet by default; open-shell automatically
            # disables time-reversal pairing so the build factory does not
            # raise on the contradiction.
            time_reversal = int(num_electron_up) == int(num_electron_dn)
        data = cls(
            num_electron_up=int(num_electron_up),
            num_electron_dn=int(num_electron_dn),
            orb_data_up_spin=orb_data_up_spin,
            orb_data_dn_spin=orb_data_dn_spin,
            lambda_matrix=np.ascontiguousarray(_src, dtype=_dt),
            time_reversal=bool(time_reversal),
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        """Return a list of strings containing the information stored in the attributes."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  dim. of lambda_matrix = {self.lambda_matrix.shape}")
        info_lines.append(
            f"  lambda_matrix is symmetric? = {np.allclose(self.lambda_matrix[: self.orb_num_up, : self.orb_num_up], self.lambda_matrix[: self.orb_num_up, : self.orb_num_up].T)}"
        )
        lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(self.lambda_matrix, [self.orb_num_dn])
        info_lines.append(f"  lambda_matrix_paired.shape = {lambda_matrix_paired.shape}")
        info_lines.append(f"  lambda_matrix_unpaired.shape = {lambda_matrix_unpaired.shape}")
        info_lines.append(f"  num_electron_up = {self.num_electron_up}")
        info_lines.append(f"  num_electron_dn = {self.num_electron_dn}")
        info_lines.extend(self.orb_data_up_spin._get_info())
        info_lines.extend(self.orb_data_dn_spin._get_info())

        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    # --- AO basis property accessors (for basis optimization) ---

    @property
    def _lambda_matrix_jnp(self) -> jax.Array:
        """Return lambda_matrix as a jax.Array (jnp view of the underlying numpy storage).

        Dtype follows the numpy storage (``float64`` for the real Gamma-point
        path, ``complex128`` once the loader populates a complex paired /
        unpaired lambda at a non-Gamma twist).
        Consumers promote with :func:`jnp.result_type` at the use site.
        """
        return jnp.asarray(self.lambda_matrix)

    @property
    def _determinant_dtype_jnp(self) -> jnp.dtype:
        """Working dtype used by determinant kernels."""
        return self._lambda_matrix_jnp.dtype

    def apply_block_update(self, block: "VariationalParameterBlock") -> "Geminal_data":
        """Apply a single variational-parameter block update to this Geminal object.

        This method is the Geminal-specific counterpart of
        :meth:`Wavefunction_data.apply_block_updates`.  It receives a generic
        :class:`VariationalParameterBlock` whose ``values`` have already been
        updated (typically by ``block.apply_update`` inside the SR/MCMC driver),
        and interprets that block according to the structure of the geminal
        (lambda) matrix.

        Responsibilities of this method are:

        * Map the block name (currently ``"lambda_matrix"``) to the internal
          geminal parameters.
        * Handle the splitting of a rectangular lambda matrix into paired and
          unpaired parts when needed.
        * Enforce Geminal-specific structural constraints from the declared
          ``time_reversal`` ansatz flag.  Optimizer-facing tangent metadata
          carries the matching immutable linear-tie map.

        All details about how the lambda parameters are stored and constrained
        live here (and in the surrounding ``Geminal_data`` class), not in
        :class:`VariationalParameterBlock` or in the optimizer.  This keeps the
        SR/MCMC machinery and the block abstraction structure-agnostic: adding
        new Geminal parameters should only require updating the block
        construction in ``Wavefunction_data.get_variational_blocks`` and adding
        the corresponding handling in this method.
        """
        if block.name == "lambda_matrix":
            # Write back into the det_eval zone, matching the
            # init dtype produced by :py:meth:`Geminal_data.build`. Keeps
            # the JIT cache key stable across SR iterations under mixed
            # precision (otherwise iter 1 fp64 vs iter 2+ in whatever
            # dtype the SR delta promoted to would recompile the geminal
            # kernels).
            # The SR delta block for lambda goes complex once the storage
            # dtype flips to ``complex128``; downcasting
            # to ``det_eval`` (real) would silently drop ``Im(delta)`` every
            # iteration. Lift to ``complex128`` when the input is already
            # complex; otherwise the zone-aware ``det_eval`` lookup is
            # preserved so fp32 mixed mode stays unchanged.
            _dt = np.complex128 if np.iscomplexobj(block.values) else get_dtype_np("det_eval")
            lambda_new = np.asarray(block.values, dtype=_dt)

            # Symmetrize unconditionally -- the method is a no-op for non-symmetric matrices.
            lambda_new = self.symmetrize_lambda(lambda_new)

            return Geminal_data(
                num_electron_up=self.num_electron_up,
                num_electron_dn=self.num_electron_dn,
                orb_data_up_spin=self.orb_data_up_spin,
                orb_data_dn_spin=self.orb_data_dn_spin,
                lambda_matrix=lambda_new,
                time_reversal=self.time_reversal,
            )
        if block.name == "lambda_basis_exp":
            return self._apply_basis_block_delegated(block, kind="exp")
        if block.name == "lambda_basis_coeff":
            return self._apply_basis_block_delegated(block, kind="coeff")

        return self

    @property
    def orb_num_up(self) -> int:
        """orb_num_up.

        The number of atomic orbitals or molecular orbitals for up electrons,
        depending on the instance stored in the attribute orb_data_up.

        Return:
            int: The number of atomic orbitals or molecular orbitals for up electrons.

        Raises:
            NotImplementedError:
                If the instance of orb_data_up_spin is neither AOs_data nor MOs_data.

        """
        if _is_ao_leaf(self.orb_data_up_spin):
            return self.orb_data_up_spin.num_ao
        if isinstance(self.orb_data_up_spin, MOs_data):
            return self.orb_data_up_spin.num_mo
        raise NotImplementedError

    @property
    def orb_num_dn(self) -> int:
        """orb_num_dn.

        The number of atomic orbitals or molecular orbitals for down electrons,
        depending on the instance stored in the attribute orb_data_up.

        Return:
            int: The number of atomic orbitals or molecular orbitals for down electrons.

        Raises:
            NotImplementedError:
                If the instance of orb_data_dn_spin is neither AOs_data nor MOs_data.
        """
        if _is_ao_leaf(self.orb_data_dn_spin):
            return self.orb_data_dn_spin.num_ao
        if isinstance(self.orb_data_dn_spin, MOs_data):
            return self.orb_data_dn_spin.num_mo
        raise NotImplementedError

    @property
    def is_mo_representation(self) -> bool:
        """Whether both spin channels are represented by molecular orbitals."""
        return isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data)

    @property
    def is_ao_representation(self) -> bool:
        """Whether both spin channels are represented by atomic orbitals.

        An AO representation is any AOs_base_data leaf that is not an
        MOs_data wrapper (covers GTO, periodic pGTO, plane-wave, B-spline
        and composite AO containers).
        """
        return _is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin)

    @property
    def compute_orb_api(self) -> Callable[..., jax.Array]:
        """Function for computing AOs or MOs.

        The api method to compute AOs or MOs corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin

        Return:
            Callable: The api method to compute AOs or MOs.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if _is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin):
            return compute_AOs
        if isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs
        raise NotImplementedError

    @property
    def compute_orb_grad_api(self) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
        """Function for computing AOs or MOs grads.

        The api method to compute AOs or MOs grads corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin.

        Return:
            Callable: The api method to compute AOs or MOs grads.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if _is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin):
            return compute_AOs_grad
        if isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs_grad
        raise NotImplementedError

    @property
    def compute_orb_laplacian_api(self) -> Callable[..., jax.Array]:
        """Function for computing AOs or MOs laplacians.

        The api method to compute AOs or MOs laplacians corresponding to instances
        stored in self.orb_data_up_spin and self.orb_data_dn_spin.

        Return:
            Callable: The api method to compute AOs or MOs laplacians.

        Raises:
            NotImplementedError:
                If the instances of orb_data_up_spin/orb_data_dn_spin are
                neither AOs_data/AOs_data nor MOs_data/MOs_data.
        """
        if _is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin):
            return compute_AOs_laplacian
        if isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs_laplacian
        raise NotImplementedError

    @property
    def compute_orb_value_grad_lap_api(
        self,
    ) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        """Fused (value, grad, laplacian) api for AOs or MOs.

        Returns a callable that yields ``(val, gx, gy, gz, lap)`` in a single
        dispatch -- used by the streaming advance hot path so the heavy block
        (``exp``, polynomial chain, ``S_l_m``) is shared across val/grad/lap
        instead of being recomputed three times.
        """
        if _is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin):
            return compute_AOs_value_grad_lap
        if isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            return compute_MOs_value_grad_lap
        raise NotImplementedError

    def natural_orbital_density_coefficients_jnp(
        self,
        frozen_overlap: _FrozenOverlap = None,
    ) -> tuple[
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
    ]:
        r"""JAX-traceable natural-orbital coefficients and occupations for density evaluation.

        Returns the per-spin AO leaves, AO-basis orbital coefficients, and
        natural-orbital occupations needed to evaluate the spin-summed
        one-body density represented by the geminal. The result is traceable
        with ``jax.grad`` so density-dependent forces can differentiate
        through ion positions.

        For MO representations, the stored ``mo_coefficients`` are used as the
        natural orbitals. For closed-shell AO-AGP representations, the natural
        orbitals and occupations are obtained from the paired geminal.
        Open-shell AO-AGP (unpaired AO block) is not supported.

        Returns:
            tuple ``(aos_up, coeff_up, n_up, aos_dn, coeff_dn, n_dn)`` where
            ``aos_*`` are the AO leaves to evaluate, ``coeff_*`` have shape
            ``(num_orb_*, num_ao_*)`` (AO-basis orbital coefficients), and
            ``n_*`` have shape ``(num_orb_*,)`` (occupations, fp64). The
            density built from these integrates to ``N_up + N_dn`` exactly for
            a Slater determinant and approximately (leading-order) for a
            dense-lambda AGP; the density build renormalises to ``N_e`` on the
            grid regardless.

        Raises:
            NotImplementedError: For open-shell AO-AGP (unpaired AO block), or
                for orbital containers that are neither a matched MO/MO pair
                nor a matched AO-leaf/AO-leaf pair.

        Notes:
            The detailed natural-orbital density construction is work in
            progress.
        """
        orb_num_dn = self.orb_num_dn

        # ---- MO representation: direct read (production path). ----
        if isinstance(self.orb_data_up_spin, MOs_data) and isinstance(self.orb_data_dn_spin, MOs_data):
            # Complex Bloch: preserve the complex MO dtype so the
            # downstream MPC builder can compute ``|psi|^2`` correctly. A
            # ``dtype=jnp.float64`` hard-cast would drop Im at twist k != 0.
            # Zone is ``det_eval``: these are geminal (MO + lambda) quantities;
            # the downstream Coulomb consumer is on the ``coulomb`` zone but
            # its inputs are produced here by determinant code.
            coeff_up = cast_to_zone(self.orb_data_up_spin._mo_coefficients_jnp, "det_eval")
            coeff_dn = cast_to_zone(self.orb_data_dn_spin._mo_coefficients_jnp, "det_eval")
            lambda_matrix = cast_to_zone(self.lambda_matrix, "det_eval")
            paired = lambda_matrix[:, :orb_num_dn]
            unpaired = lambda_matrix[:, orb_num_dn:]
            # Down occupations: |diag(paired)|. Up occupations: |diag(paired)|
            # plus the per-up-orbital weight of the unpaired block (each
            # singly-occupied up orbital contributes 1 in the Slater case).
            diag_paired = jnp.abs(jnp.diagonal(paired))
            n_dn = diag_paired
            n_up = diag_paired + jnp.sum(jnp.abs(unpaired), axis=1)
            return (
                self.orb_data_up_spin.aos_data,
                coeff_up,
                n_up,
                self.orb_data_dn_spin.aos_data,
                coeff_dn,
                n_dn,
            )

        # ---- AO (AGP) representation: jnp natural-orbital decomposition. ----
        if not (_is_ao_leaf(self.orb_data_up_spin) and _is_ao_leaf(self.orb_data_dn_spin)):
            raise NotImplementedError(
                "natural_orbital_density_coefficients_jnp requires a matched MO/MO or AO-leaf/AO-leaf orbital pair."
            )

        num_unpaired = self.num_electron_up - self.num_electron_dn
        if num_unpaired != 0:
            raise NotImplementedError(
                "natural_orbital_density_coefficients_jnp supports only "
                "closed-shell AO-AGP (num_electron_up == num_electron_dn); "
                "got open-shell AGP with "
                f"{num_unpaired} unpaired electrons."
            )

        aos_data_up_spin = self.orb_data_up_spin
        aos_data_dn_spin = self.orb_data_dn_spin

        # Complex Bloch: preserve complex AO/lambda dtype.
        # Zone is ``det_eval``: the work below is geminal SVD on
        # ``S_up^{1/2} @ lambda @ S_dn^{1/2}``, matching the det_eval zone
        # definition ("geminal matrix elements + log-det / SVD / AS reg").
        lambda_matrix = cast_to_zone(self.lambda_matrix, "det_eval")
        ao_lambda_paired = lambda_matrix[:, :orb_num_dn]

        # Hoisted-overlap path (MPC-aware ``_auto`` twin): ``frozen_overlap`` carries the
        # (S_up, S_dn) computed EAGERLY from the concrete ham, captured as closure constants
        # -- zero tangent under AD, matching the analytic matched-approximation freeze of the
        # natural-orbital coeff/occ w.r.t. geometry (S is lambda-independent, so the c-axis
        # SVD below stays lambda-differentiable). Concrete callers (driver / assemble) leave
        # it None and take the direct host-kernel path, bit-identically.
        if frozen_overlap is not None:
            s_up_raw, s_dn_raw = frozen_overlap
            overlap_up = cast_to_zone(jnp.asarray(s_up_raw), "det_eval")
            overlap_dn = cast_to_zone(jnp.asarray(s_dn_raw), "det_eval")
        else:
            overlap_up = cast_to_zone(_overlap_matrix_traceable(aos_data_up_spin), "det_eval")
            overlap_dn = cast_to_zone(_overlap_matrix_traceable(aos_data_dn_spin), "det_eval")
        # Hermitian symmetrisation (defensive against FP rounding). For a
        # complex Hermitian S the conjugate transpose is required:
        # ``S + S.T`` averages S with S.conj() (since S^H = S => S.T = S.conj()),
        # leaving only Re(S) and destroying the imaginary part of S. ``.conj().T``
        # is bit-identical to ``.T`` on the real Gamma path (``.conj()`` is a
        # no-op for real arrays) and produces the correct Hermitian projector on
        # the complex Bloch, non-Gamma path.
        overlap_up = 0.5 * (overlap_up + overlap_up.conj().T)
        overlap_dn = 0.5 * (overlap_dn + overlap_dn.conj().T)

        # Numerical-stability clip (mirrors the numpy convert path): lift tiny
        # non-positive overlap eigenvalues to a small positive floor so
        # S^{-1/2} stays finite. jnp.maximum keeps this AD-safe (no Python
        # branch on traced values).
        eig_clip_tol = AO_OVERLAP_EIGENVALUE_FLOOR
        eigvals_up, eigvecs_up = jnp.linalg.eigh(overlap_up)
        eigvals_dn, eigvecs_dn = jnp.linalg.eigh(overlap_dn)
        eigvals_up = jnp.maximum(eigvals_up, eig_clip_tol)
        eigvals_dn = jnp.maximum(eigvals_dn, eig_clip_tol)

        # Hermitian S^{1/2} = U diag(sqrt(eigvals)) U^H, where U^H = U.conj().T
        # (eigh of complex Hermitian returns complex unitary U). ``.conj().T``
        # is a no-op for the real-eigvecs Gamma path and is required on the
        # complex Bloch, non-Gamma path.
        sqrt_up = eigvecs_up @ jnp.diag(jnp.sqrt(eigvals_up)) @ eigvecs_up.conj().T
        inv_sqrt_up = eigvecs_up @ jnp.diag(1.0 / jnp.sqrt(eigvals_up)) @ eigvecs_up.conj().T
        sqrt_dn = eigvecs_dn @ jnp.diag(jnp.sqrt(eigvals_dn)) @ eigvecs_dn.conj().T
        inv_sqrt_dn = eigvecs_dn @ jnp.diag(1.0 / jnp.sqrt(eigvals_dn)) @ eigvecs_dn.conj().T

        # The AO geminal sum_ab chi_a(r1) lambda_ab chi_b(r2) is bilinear and the
        # MO rows satisfy conj(C) S C^T = I. With lambda = C_up^T diag(L) C_dn,
        # Q_up = S_up^{1/2} C_up^T and Q_dn = S_dn^{1/2} C_dn^T are isometries
        # and C_dn (S_dn^{1/2})^T = Q_dn^T, so
        #     S_up^{1/2} lambda (S_dn^{1/2})^T = Q_up diag(L) Q_dn^T
        # is an SVD: U = Q_up, V^H = Q_dn^T. The right factor is therefore a
        # TRANSPOSE (same convention as ``convert_from_AOs_to_MOs``), and the dn
        # natural orbitals are C_dn = V^H (S_dn^{-1/2})^T, i.e. P_dn =
        # S_dn^{-1/2} (V^H)^T. Using ``S_dn^{1/2}`` (the adjoint) instead breaks
        # this at a complex twist, where S_dn = conj(S_up) is not real; on a
        # real (Gamma) overlap the two forms coincide.
        f_biorth = sqrt_up @ ao_lambda_paired @ sqrt_dn.T
        u_mat, singular_vals, v_h = jnp.linalg.svd(f_biorth, full_matrices=False)

        # Natural orbitals in the AO basis: P = S^{-1/2} (left/right singular
        # vectors); ``coeff_*`` rows are the per-orbital AO coefficients.
        p_up = inv_sqrt_up @ u_mat
        p_dn = inv_sqrt_dn @ v_h.T
        coeff_up = p_up.T
        coeff_dn = p_dn.T
        n_up = jnp.abs(singular_vals)
        n_dn = jnp.abs(singular_vals)

        return (
            aos_data_up_spin,
            coeff_up,
            n_up,
            aos_data_dn_spin,
            coeff_dn,
            n_dn,
        )


# Compatibility alias: ``Geminal_data`` is another name of ``AGP_data``, the
# AGP-flavoured geminal class (with ``AGPn_data`` and ``SD_data`` siblings).
# The alias keeps existing imports, isinstance checks, and constructor calls
# working; ``AGP_data`` is the canonical name.
Geminal_data = AGP_data


# ---------------------------------------------------------------------------
# S-metric polar retraction. Used by ``AGPn_data.apply_block_update`` and
# ``SD_data.apply_block_update`` to bring MO coefficients back onto the
# Stiefel manifold after a raw SR / LM step.
# ---------------------------------------------------------------------------


def _smetric_polar_retract(
    C: npt.NDArray,
    aos_data: AOs_base_data,
    *,
    metric_sqrt: npt.NDArray | None = None,
) -> npt.NDArray:
    """Project ``C`` back onto the Stiefel manifold in the S-metric.

    Given a post-step MO coefficient matrix ``C`` of shape ``(n, N_AO)``,
    form ``M = C.conj() @ S @ C.T`` and return
    ``qf_S(C) = conj(M^{-1/2}) @ C`` so that the row-oriented result
    satisfies ``C_new.conj() @ S @ C_new.T = I``.

    Args:
        C: Post-step MO coefficient matrix, shape ``(n, N_AO)``. Can be
           real or complex.
        aos_data: AO leaf used to compute the overlap matrix ``S`` when no
            frozen optimizer metric is supplied.
        metric_sqrt: Optional frozen ``S^{1/2}`` published with the optimizer
            block.  Supplying it guarantees that the tangent action and
            finite-step retraction use exactly the same metric.

    Returns:
        Same-shape MO coefficient matrix, S-orthonormalised.
    """
    if metric_sqrt is None:
        S = np.asarray(cast_to_zone(compute_overlap_matrix(aos_data), "det_eval"))
        S = 0.5 * (S + S.conj().T)  # Hermitian symmetrise (round-off guard)
    else:
        sqrt = np.asarray(metric_sqrt)
        S = sqrt @ sqrt
        S = 0.5 * (S + S.conj().T)
    C_arr = np.asarray(C)
    # Promote to a common working dtype: complex if either S or C is complex.
    work_dtype = np.result_type(C_arr.dtype, S.dtype)
    C_arr = C_arr.astype(work_dtype, copy=False)
    S = S.astype(work_dtype, copy=False)

    M = C_arr.conj() @ S @ C_arr.T
    # Numerical guard: enforce Hermitian and a tiny positive floor on the
    # eigenvalues before inversion (matches the discipline elsewhere in
    # this module, see :py:func:`_hermitian_sqrt`).
    M = 0.5 * (M + M.conj().T)
    _, M_inv_sqrt = _hermitian_sqrt(M)
    C_new = M_inv_sqrt.conj() @ C_arr

    # Independent postcondition: row-oriented complex coefficients use
    # C* S C^T = I.  This assertion catches transpose/adjoint mistakes even
    # when a Gamma-point fixture makes every quantity accidentally real.
    gram = C_new.conj() @ S @ C_new.T
    identity = np.eye(C_new.shape[0], dtype=gram.dtype)
    if not np.allclose(gram, identity, atol=atol_consistency, rtol=rtol_consistency):
        residual = float(np.linalg.norm(gram - identity))
        raise FloatingPointError(
            f"S-metric polar retraction failed to restore row orthonormality; ||C* S C^T - I||={residual:.3e}."
        )
    return C_new.astype(C_arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# AGPn_data -- Antisymmetrized Geminal Power in natural-orbital occupation
# form (MO representation; lambda = diag(occupations)).
# ---------------------------------------------------------------------------


@jax.custom_vjp
def _embed_real_parameter_in_complex_arithmetic(x: jax.Array) -> jax.Array:
    """Embed a real physical parameter in complex arithmetic.

    AGPn occupations remain real variational parameters at non-Gamma twist,
    while their MO contraction is complex.  A plain real-to-complex cast has
    the correct forward value, but JAX's transpose rule later converts the
    complex cotangent back to the real leaf implicitly.  Under the systematic
    ``ComplexWarning`` guard that implicit (though physical) projection is an
    error.  This boundary makes the same projection explicit in the VJP.
    """
    return jax.lax.complex(x, jnp.zeros_like(x))


def _embed_real_parameter_in_complex_arithmetic_fwd(x: jax.Array) -> tuple[jax.Array, None]:
    return jax.lax.complex(x, jnp.zeros_like(x)), None


def _embed_real_parameter_in_complex_arithmetic_bwd(
    _res: None,
    cotangent: jax.Array,
) -> tuple[jax.Array]:
    return (jnp.real(cotangent),)


_embed_real_parameter_in_complex_arithmetic.defvjp(
    _embed_real_parameter_in_complex_arithmetic_fwd,
    _embed_real_parameter_in_complex_arithmetic_bwd,
)


@struct.dataclass
class AGPn_data(_GeminalBase):
    """Antisymmetrized Geminal Power, natural-orbital occupation form.

    AGPn (``Lambda_MO = diag(lambda_diag) | lambda_unpaired``) stores the
    geminal in the MO basis with a diagonal paired pairing block whose
    entries are the natural-orbital occupations, and an open-shell
    ``lambda_unpaired`` block holding the unpaired MO -> excess-up
    coupling. This is the rank-``n`` truncation of the full AGP
    (``AGP_data``) in the natural-orbital basis.

    Attributes:
        num_electron_up: Number of spin-up electrons.
        num_electron_dn: Number of spin-down electrons.
        orb_data_up_spin: MOs_data for spin-up orbitals.
        orb_data_dn_spin: MOs_data for spin-down orbitals (related to up via
            time-reversal when ``time_reversal=True``).
        lambda_diag: Diagonal of the natural-orbital occupation matrix,
            shape ``(n,)``. Always real (occupations are real even on the
            complex Bloch path).
        lambda_unpaired: Open-shell unpaired block, shape
            ``(n, n_unpaired)`` with ``n_unpaired = num_electron_up -
            num_electron_dn``. Empty (shape ``(n, 0)``) for closed-shell.
            dtype follows the AO leaf complex / real regime -- same rule as
            ``AGP_data.lambda_matrix``. Initialised
            from the loader's unpaired block by ``build``, then the
            occupations evolve as variational parameters
            (``apply_block_update("lambda_unpaired")``).
        time_reversal: Coefficient-tying flag. See ``AGP_data.time_reversal``.
    """

    num_electron_up: int = struct.field(pytree_node=False, default=0)
    num_electron_dn: int = struct.field(pytree_node=False, default=0)
    orb_data_up_spin: MOs_data = struct.field(pytree_node=True, default_factory=lambda: MOs_data())
    orb_data_dn_spin: MOs_data = struct.field(pytree_node=True, default_factory=lambda: MOs_data())
    lambda_diag: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )
    lambda_unpaired: npt.NDArray = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    time_reversal: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_mo(self) -> int:
        """Number of MOs (= rank of the natural-orbital expansion)."""
        return int(self.lambda_diag.shape[0])

    # --- duck-type interface compatible with AGP_data so the existing
    # geminal-eval kernel (_compute_geminal_all_elements, det / grad /
    # laplacian helpers) accepts an AGPn_data instance without changes.

    @property
    def orb_num_up(self) -> int:
        return int(self.orb_data_up_spin.num_mo)

    @property
    def orb_num_dn(self) -> int:
        return int(self.orb_data_dn_spin.num_mo)

    @property
    def _lambda_matrix_jnp(self) -> jax.Array:
        """Jnp view of the materialised lambda block
        ``hstack([diag(lambda_diag), lambda_unpaired])``. Shape
        ``(n, n + n_unpaired)``.  Complex Bloch MOs select complex local
        arithmetic even though ``lambda_diag`` itself remains a real physical
        parameter; its VJP explicitly projects the complex cotangent back to
        that real parameter space.
        """
        diag_values = jnp.asarray(self.lambda_diag)
        unp = jnp.asarray(self.lambda_unpaired)
        complex_mo_arithmetic = jnp.iscomplexobj(self.orb_data_up_spin._mo_coefficients_jnp)
        if complex_mo_arithmetic:
            diag_values = _embed_real_parameter_in_complex_arithmetic(diag_values)
            if not jnp.iscomplexobj(unp):
                unp = _embed_real_parameter_in_complex_arithmetic(unp)
        diag_part = jnp.diag(diag_values)
        if unp.shape[1] == 0:
            return diag_part
        work = jnp.result_type(diag_part.dtype, unp.dtype)
        return jnp.hstack([diag_part.astype(work, copy=False), unp.astype(work, copy=False)])

    @property
    def _determinant_dtype_jnp(self) -> jnp.dtype:
        """Working dtype used by determinant kernels."""
        return self._lambda_matrix_jnp.dtype

    @property
    def is_mo_representation(self) -> bool:
        return True

    @property
    def is_ao_representation(self) -> bool:
        return False

    @property
    def compute_orb_api(self) -> Callable[..., jax.Array]:
        return compute_MOs

    @property
    def compute_orb_grad_api(self) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
        return compute_MOs_grad

    @property
    def compute_orb_laplacian_api(self) -> Callable[..., jax.Array]:
        return compute_MOs_laplacian

    @property
    def compute_orb_value_grad_lap_api(
        self,
    ) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        return compute_MOs_value_grad_lap

    def natural_orbital_density_coefficients_jnp(
        self,
        frozen_overlap: _FrozenOverlap = None,
    ) -> tuple[
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
    ]:
        """MO-basis natural-orbital coefficients + occupations for the MPC density.

        Mirrors :py:meth:`AGP_data.natural_orbital_density_coefficients_jnp`
        but reads the natural-orbital occupations directly from
        ``lambda_diag`` (which IS the stored diagonal); no
        ``diag(lambda_diag)`` materialisation. Open-shell adds the
        ``|lambda_unpaired|.sum(axis=1)`` contribution to the up-spin
        occupations.
        """
        # Complex Bloch: preserve complex MO dtype. See AGP_data
        # natural_orbital_density_coefficients_jnp for rationale.
        coeff_up = cast_to_zone(self.orb_data_up_spin._mo_coefficients_jnp, "det_eval")
        coeff_dn = cast_to_zone(self.orb_data_dn_spin._mo_coefficients_jnp, "det_eval")
        n_dn = jnp.abs(jnp.asarray(self.lambda_diag, dtype=jnp.float64))
        unp = jnp.asarray(self.lambda_unpaired)
        if unp.shape[1] == 0:
            n_up = n_dn
        else:
            n_up = n_dn + jnp.sum(jnp.abs(unp), axis=1)
        return (
            self.orb_data_up_spin.aos_data,
            coeff_up,
            n_up,
            self.orb_data_dn_spin.aos_data,
            coeff_dn,
            n_dn,
        )

    @property
    def lambda_matrix(self) -> npt.NDArray:
        """Materialise ``hstack([diag(lambda_diag), lambda_unpaired])`` for
        downstream compat with code that expects ``AGP_data.lambda_matrix``.

        Shape is ``(num_mo, num_mo)`` for closed-shell and
        ``(num_mo, num_mo + n_unpaired)`` for open-shell.
        """
        diag_part = np.diag(np.asarray(self.lambda_diag))
        # The materialised AGPn lambda follows the MO coefficient dtype
        # (complex128 off-Gamma, float64 at Gamma); ``lambda_diag`` itself
        # stays real because occupations are real.
        _mo_dtype = np.asarray(self.orb_data_up_spin.mo_coefficients).dtype
        if np.issubdtype(_mo_dtype, np.complexfloating):
            diag_part = diag_part.astype(np.complex128, copy=False)
        unp = np.asarray(self.lambda_unpaired)
        if unp.shape[1] == 0:
            return diag_part
        work = np.result_type(diag_part.dtype, unp.dtype)
        return np.hstack([diag_part.astype(work, copy=False), unp.astype(work, copy=False)])

    def sanity_check(self) -> None:
        """Validate the fields and the time-reversal invariant."""
        if not isinstance(self.num_electron_up, (int, np.integer)):
            raise ValueError(f"num_electron_up = {type(self.num_electron_up)} must be an int.")
        if not isinstance(self.num_electron_dn, (int, np.integer)):
            raise ValueError(f"num_electron_dn = {type(self.num_electron_dn)} must be an int.")
        # See ``AGP_data.sanity_check`` for the ROHF / kROHF semantics of
        # ``time_reversal=True`` with open-shell electron counts.
        if self.time_reversal and self.num_electron_up < self.num_electron_dn:
            raise ValueError(
                f"time_reversal=True with num_electron_up < num_electron_dn "
                f"({self.num_electron_up} < {self.num_electron_dn}) is not supported; "
                f"alpha-major convention assumed."
            )

        if not isinstance(self.orb_data_up_spin, MOs_data):
            raise ValueError(f"AGPn_data.orb_data_up_spin must be MOs_data; got {type(self.orb_data_up_spin).__name__}.")
        if not isinstance(self.orb_data_dn_spin, MOs_data):
            raise ValueError(f"AGPn_data.orb_data_dn_spin must be MOs_data; got {type(self.orb_data_dn_spin).__name__}.")

        n_up_mo = int(self.orb_data_up_spin.num_mo)
        n_dn_mo = int(self.orb_data_dn_spin.num_mo)
        n = self.num_mo
        if n_up_mo != n_dn_mo:
            raise ValueError(
                f"AGPn_data MO counts disagree: up num_mo={n_up_mo}, dn num_mo={n_dn_mo}. "
                f"Both spin channels must share the same MO count."
            )
        if n != n_up_mo:
            raise ValueError(f"lambda_diag length ({n}) must match orb_data_up_spin.num_mo ({n_up_mo}).")

        # lambda_diag is always real -- natural-orbital occupations are real
        # invariants, even on the complex Bloch path.
        if np.iscomplexobj(np.asarray(self.lambda_diag)):
            raise ValueError("AGPn_data.lambda_diag must be real; natural-orbital occupations are real physical invariants.")

        # lambda_unpaired shape contract: rows
        # match num_mo, cols match (num_electron_up - num_electron_dn).
        unp = np.asarray(self.lambda_unpaired)
        n_unpaired = int(self.num_electron_up) - int(self.num_electron_dn)
        if unp.ndim != 2:
            raise ValueError(f"AGPn_data.lambda_unpaired must be a 2D array; got ndim={unp.ndim}.")
        if unp.shape[1] != n_unpaired:
            raise ValueError(
                f"AGPn_data.lambda_unpaired columns ({unp.shape[1]}) must equal "
                f"num_electron_up - num_electron_dn ({n_unpaired})."
            )
        if n_unpaired > 0 and unp.shape[0] != n:
            raise ValueError(f"AGPn_data.lambda_unpaired rows ({unp.shape[0]}) must equal num_mo ({n}) when n_unpaired > 0.")

        if min(int(self.num_electron_up), int(self.num_electron_dn)) > 0:
            # An empty spin channel enters no determinant: nothing to pair.
            _require_opposite_spin_twists(self.orb_data_up_spin, self.orb_data_dn_spin)
        self.orb_data_up_spin.sanity_check()
        self.orb_data_dn_spin.sanity_check()

    @classmethod
    def build(
        cls,
        *,
        num_electron_up: int,
        num_electron_dn: int,
        mos_up: MOs_data,
        mos_dn: MOs_data | None = None,
        lambda_diag: npt.NDArray | None = None,
        lambda_unpaired: npt.NDArray | None = None,
        time_reversal: bool | None = None,
    ) -> "AGPn_data":
        """Construct an ``AGPn_data`` with default-init and time-reversal
        handling.

        Args:
            num_electron_up: Number of spin-up electrons.
            num_electron_dn: Number of spin-down electrons.
            mos_up: Up-spin MOs_data.
            mos_dn: Down-spin MOs_data. If omitted and ``time_reversal=True``,
                derived as the conjugate of ``mos_up``.
            lambda_diag: Diagonal natural-orbital occupations, shape
                ``(num_mo,)``. If omitted, initialised to
                ``[1.0]*N_occ + [0.0]*(num_mo - N_occ)`` where
                ``N_occ = num_electron_dn`` (closed-shell convention).
            lambda_unpaired: Open-shell unpaired block, shape
                ``(num_mo, num_electron_up - num_electron_dn)``. If omitted
                and the system is open-shell, initialised to an
                identity-selector pattern (1 at row ``n_dn + k`` for column
                ``k``, zero elsewhere). Canonical SD loaders do not construct
                AGPn or retain these extra rows. For
                closed-shell the default is the empty array ``(0, 0)``.
                The unpaired occupations are variational on AGPn (see
                ``apply_block_update("lambda_unpaired")``).
            time_reversal: See ``AGP_data.build``; if ``None``, inferred
                from electron counts (True iff equal).
        """
        n_up = int(num_electron_up)
        n_dn = int(num_electron_dn)
        if time_reversal is None:
            time_reversal = n_up == n_dn

        if time_reversal and mos_dn is None:
            # BCS / TR-pair auto-derive: dn-channel orbital is phi^{-k} =
            # conj(phi^k), built from (conj(C^k), -k AO leaf). Real / Gamma:
            # conj is identity AND -k == k, so collapses byte-identically to
            # ``mos_dn = mos_up.replace(...)`` with shared coefficients / leaf.
            new_coeffs = np.conj(np.asarray(mos_up.mo_coefficients))
            if hasattr(mos_up.aos_data, "with_negated_k"):
                _aos_dn = mos_up.aos_data.with_negated_k()
            else:
                _aos_dn = mos_up.aos_data
            mos_dn = MOs_data(
                num_mo=int(mos_up.num_mo),
                aos_data=_aos_dn,
                mo_coefficients=new_coeffs,
            )
        elif mos_dn is None:
            raise ValueError(
                "mos_dn is required when time_reversal=False; the two spin channels "
                "are variationally independent in the open-shell / FM regime."
            )

        num_mo = int(mos_up.num_mo)
        if lambda_diag is None:
            n_occ = n_dn
            if n_occ > num_mo:
                raise ValueError(
                    f"num_electron_dn={n_occ} exceeds num_mo={num_mo}. Supply an explicit common-rank AGPn orbital basis."
                )
            lambda_diag = np.concatenate(
                [
                    np.ones(n_occ, dtype=np.float64),
                    np.zeros(num_mo - n_occ, dtype=np.float64),
                ]
            )

        n_unpaired = n_up - n_dn
        if lambda_unpaired is None:
            if n_unpaired == 0:
                # closed-shell: shape (0, 0) so the field is a benign empty
                # tensor that round-trips through HDF5 / replace() cleanly.
                lambda_unpaired = np.zeros((0, 0), dtype=np.float64)
            else:
                # Default to an explicit identity-selector initialization at
                # rows [n_dn : n_dn + n_unpaired], zero elsewhere. dtype
                # follows the MO leaf complex regime so the real/complex
                # dtype invariants hold. The unpaired occupations are
                # variational from this initial value.
                mo_dtype = np.asarray(mos_up.mo_coefficients).dtype
                _dt = np.complex128 if np.issubdtype(mo_dtype, np.complexfloating) else np.float64
                lambda_unpaired = np.pad(
                    np.eye(n_unpaired, dtype=_dt),
                    ((n_dn, num_mo - n_dn - n_unpaired), (0, 0)),
                )

        data = cls(
            num_electron_up=n_up,
            num_electron_dn=n_dn,
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            lambda_diag=np.ascontiguousarray(np.asarray(lambda_diag, dtype=np.float64)),
            lambda_unpaired=np.ascontiguousarray(np.asarray(lambda_unpaired)),
            time_reversal=bool(time_reversal),
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        """Return a list of strings describing the instance for logging."""
        info_lines = [
            "**" + self.__class__.__name__,
            f"  num_mo = {self.num_mo}",
            f"  lambda_diag (head 5) = {np.asarray(self.lambda_diag)[:5]}",
            f"  num_electron_up = {self.num_electron_up}",
            f"  num_electron_dn = {self.num_electron_dn}",
            f"  time_reversal = {self.time_reversal}",
        ]
        info_lines.extend(self.orb_data_up_spin._get_info())
        info_lines.extend(self.orb_data_dn_spin._get_info())
        return info_lines

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)

    def to_AGP(self) -> AGP_data:
        """Materialise to an ``AGP_data`` in MO basis with
        ``lambda_matrix = hstack([diag(lambda_diag), lambda_unpaired])``.
        Downstream-compat fallback for code paths that expect the dense
        ``AGP_data`` interface. Closed- and open-shell both supported
        (the ``lambda_unpaired`` block is carried through via the
        ``lambda_matrix`` property).
        """
        return AGP_data.build(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=self.orb_data_up_spin,
            orb_data_dn_spin=self.orb_data_dn_spin,
            lambda_matrix=self.lambda_matrix,
            time_reversal=self.time_reversal,
        )

    def apply_block_update(self, block: "VariationalParameterBlock") -> "AGPn_data":
        r"""Apply a variational-parameter block update.

        Recognised blocks:

        * ``"mo_coefficients_up"``: the SR / LM driver supplies the
          post-step MO coefficient values for the up-spin channel. This
          method retracts them to preserve orthonormality and, when
          ``time_reversal=True``, derives the dn-spin coefficients as
          ``conj(C_up)``.
        * ``"mo_coefficients_dn"``: open-shell / UHF only
          (``time_reversal=False``). Independent orthonormality-preserving
          retraction on the dn-channel coefficients using
          ``orb_data_dn_spin.aos_data`` for the overlap matrix. A no-op
          when ``time_reversal=True`` (the dn channel is derived from
          the up channel via the ``mo_coefficients_up`` block and the
          dn block should never be exposed by ``get_variational_blocks``
          in that regime).
        * ``"lambda_diag"``: unconstrained Euclidean step on R^n. The
          block.values is the post-step occupation vector.
        * ``"lambda_unpaired"``: unconstrained Euclidean step on
          ``(num_mo, n_unpaired)``. Same treatment as
          ``lambda_diag`` -- no projection, no constraint.

        Other block names are no-ops (this geminal does not own them).

        Notes:
            The detailed retraction construction is work in progress.
        """
        if block.name == "mo_coefficients_up":
            new_C_up = np.asarray(block.values)
            new_C_up = _smetric_polar_retract(
                new_C_up,
                self.orb_data_up_spin.aos_data,
                metric_sqrt=block.tangent_spec.metric_sqrt,
            )
            new_mos_up = self.orb_data_up_spin.replace(
                mo_coefficients=new_C_up,
            )
            if self.time_reversal:
                new_C_dn = np.conj(new_C_up)
                new_mos_dn = self.orb_data_dn_spin.replace(
                    mo_coefficients=new_C_dn,
                )
            else:
                new_mos_dn = self.orb_data_dn_spin
            return self.replace(
                orb_data_up_spin=new_mos_up,
                orb_data_dn_spin=new_mos_dn,
            )
        if block.name == "mo_coefficients_dn":
            if self.time_reversal:
                # Defensive no-op: get_variational_blocks should not
                # have exposed this block when time_reversal=True.
                return self
            new_C_dn = np.asarray(block.values)
            new_C_dn = _smetric_polar_retract(
                new_C_dn,
                self.orb_data_dn_spin.aos_data,
                metric_sqrt=block.tangent_spec.metric_sqrt,
            )
            new_mos_dn = self.orb_data_dn_spin.replace(mo_coefficients=new_C_dn)
            return self.replace(orb_data_dn_spin=new_mos_dn)
        if block.name == "lambda_diag":
            # Unconstrained Euclidean step. ``lambda_diag`` is real even
            # on the complex Bloch path; coerce here in case
            # the SR / LM produced a residual imaginary component.
            new_diag = np.asarray(block.values, dtype=np.float64)
            return self.replace(lambda_diag=new_diag)
        if block.name == "lambda_unpaired":
            return self.replace(lambda_unpaired=np.asarray(block.values))
        return self


# ---------------------------------------------------------------------------
# SD_data -- Single Slater Determinant in occupied-MO form.
# ---------------------------------------------------------------------------


@struct.dataclass
class SD_data(_GeminalBase):
    """Single Slater Determinant in occupied orbitals.

    ``build_tiled`` consumes final orbitals from primitive k-block providers,
    with local operators and no global MO matrix. These orbitals are frozen.
    The ordinary ``build`` route below uses dense or implicit-identity MOs.

    ``SD_data`` stores exactly the occupied spin orbitals: ``N_up`` rows in
    ``orb_data_up_spin.mo_coefficients`` and ``N_dn`` rows in
    ``orb_data_dn_spin.mo_coefficients``.  It has no lambda leaf, implicit
    lambda property, or virtual-MO rows.  The Slater matrix is formed directly
    from those occupied orbitals.  For ``N_up >= N_dn`` it is

    ``hstack([Phi_up[:N_dn].T @ Phi_dn, Phi_up[N_dn:].T])``.

    This is numerically equivalent to the
    ``diag(1, ..., 1, 0, ..., 0)`` AGP-compatible representation, but makes
    the physical SD manifold explicit: orbital relaxation moves the occupied
    rows through the full AO column space; it does not activate additional
    geminal occupations.  Finite extra natural-orbital occupations belong to
    :class:`AGPn_data`.

    Attributes:
        num_electron_up: Number of spin-up electrons.
        num_electron_dn: Number of spin-down electrons.
        orb_data_up_spin: MOs_data for spin-up orbitals.
        orb_data_dn_spin: MOs_data for spin-down orbitals (related to up via
            time-reversal when ``time_reversal=True``).
        time_reversal: Coefficient-tying flag. See ``AGP_data.time_reversal``.
    """

    num_electron_up: int = struct.field(pytree_node=False, default=0)
    num_electron_dn: int = struct.field(pytree_node=False, default=0)
    orb_data_up_spin: MOs_data | TiledOrbitalSet_data = struct.field(pytree_node=True, default_factory=lambda: MOs_data())
    orb_data_dn_spin: MOs_data | TiledOrbitalSet_data = struct.field(pytree_node=True, default_factory=lambda: MOs_data())
    time_reversal: bool = struct.field(pytree_node=False, default=True)

    @property
    def num_mo(self) -> int:
        """Number of occupied up-spin orbitals (there are no virtual rows)."""
        return int(self.orb_data_up_spin.num_mo)

    @property
    def num_occ_up(self) -> int:
        return int(self.num_electron_up)

    @property
    def num_occ_dn(self) -> int:
        return int(self.num_electron_dn)

    @property
    def orb_num_up(self) -> int:
        return int(self.orb_data_up_spin.num_mo)

    @property
    def orb_num_dn(self) -> int:
        return int(self.orb_data_dn_spin.num_mo)

    @property
    def _determinant_dtype_jnp(self) -> jnp.dtype:
        """Working dtype derived from the two occupied-MO coefficient leaves."""
        if self.is_tiled_representation:
            return jnp.dtype(jnp.complex128)
        return jnp.result_type(
            jnp.asarray(self.orb_data_up_spin.mo_coefficients).dtype,
            jnp.asarray(self.orb_data_dn_spin.mo_coefficients).dtype,
        )

    @property
    def is_tiled_representation(self) -> bool:
        return isinstance(self.orb_data_up_spin, TiledOrbitalSet_data)

    @property
    def is_mo_representation(self) -> bool:
        return True

    @property
    def is_ao_representation(self) -> bool:
        return False

    @property
    def compute_orb_api(self) -> Callable[..., jax.Array]:
        return compute_tiled_orbitals if self.is_tiled_representation else compute_MOs

    @property
    def compute_orb_grad_api(self) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
        return compute_tiled_orbitals_grad if self.is_tiled_representation else compute_MOs_grad

    @property
    def compute_orb_laplacian_api(self) -> Callable[..., jax.Array]:
        return compute_tiled_orbitals_laplacian if self.is_tiled_representation else compute_MOs_laplacian

    @property
    def compute_orb_value_grad_lap_api(
        self,
    ) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        return compute_tiled_orbitals_value_grad_lap if self.is_tiled_representation else compute_MOs_value_grad_lap

    def natural_orbital_density_coefficients_jnp(
        self,
        frozen_overlap: _FrozenOverlap = None,
    ) -> tuple[
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
        "AOs_sphe_gto_data | AOs_cart_gto_data",
        jax.Array,
        jax.Array,
    ]:
        """MO-basis natural-orbital coefficients + occupations for the MPC density.

        Mirrors :py:meth:`AGP_data.natural_orbital_density_coefficients_jnp`
        The stored rows are precisely the occupied orbitals, hence every
        returned occupation is one and there is no virtual tail.
        """
        if self.is_tiled_representation:
            raise NotImplementedError(
                "tiled SD density uses final orbital values; a global natural-orbital coefficient matrix is unavailable."
            )
        coeff_up = cast_to_zone(self.orb_data_up_spin._mo_coefficients_jnp, "det_eval")
        coeff_dn = cast_to_zone(self.orb_data_dn_spin._mo_coefficients_jnp, "det_eval")
        n_up = jnp.ones((self.num_occ_up,), dtype=jnp.float64)
        n_dn = jnp.ones((self.num_occ_dn,), dtype=jnp.float64)
        return (
            self.orb_data_up_spin.aos_data,
            coeff_up,
            n_up,
            self.orb_data_dn_spin.aos_data,
            coeff_dn,
            n_dn,
        )

    def sanity_check(self) -> None:
        if not isinstance(self.num_electron_up, (int, np.integer)):
            raise ValueError(f"num_electron_up = {type(self.num_electron_up)} must be an int.")
        if not isinstance(self.num_electron_dn, (int, np.integer)):
            raise ValueError(f"num_electron_dn = {type(self.num_electron_dn)} must be an int.")
        # See ``AGP_data.sanity_check`` for the ROHF / kROHF semantics of
        # ``time_reversal=True`` with open-shell electron counts.
        if self.time_reversal and self.num_electron_up < self.num_electron_dn:
            raise ValueError(
                f"time_reversal=True with num_electron_up < num_electron_dn "
                f"({self.num_electron_up} < {self.num_electron_dn}) is not supported; "
                f"alpha-major convention assumed."
            )

        if isinstance(self.orb_data_up_spin, TiledOrbitalSet_data) or isinstance(self.orb_data_dn_spin, TiledOrbitalSet_data):
            up, down = self.orb_data_up_spin, self.orb_data_dn_spin
            if not isinstance(up, TiledOrbitalSet_data) or not isinstance(down, TiledOrbitalSet_data):
                raise ValueError("tiled SD requires two tiled spin providers.")
            up.sanity_check()
            down.sanity_check()
            if self.num_electron_up < self.num_electron_dn or self.num_electron_dn < 0:
                raise ValueError("tiled SD requires N_up >= N_down >= 0.")
            if up.num_mo != self.num_occ_up or down.num_mo != self.num_occ_dn:
                raise ValueError("tiled SD requires exactly the occupied rows, without global prefix truncation.")
            if up.spin != "up" or down.spin != "down":
                raise ValueError("tiled SD requires explicit up/down spin labels.")
            if up.tiling_matrix != down.tiling_matrix or not _structure_data_equal(up.structure_data, down.structure_data):
                raise ValueError("tiled SD spin providers must describe the same physical supercell.")
            residue = np.asarray(up.supertwist) + np.asarray(down.supertwist)
            if not np.allclose(residue, np.rint(residue), atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT):
                raise ValueError("tiled SD requires BCS spin sectors (+k_s, -k_s).")
            if self.time_reversal:
                expected = conjugate_tiled_orbitals(up, spin="down")
                leaves, tree = jax.tree_util.tree_flatten(expected)
                actual, actual_tree = jax.tree_util.tree_flatten(down)
                if tree != actual_tree or any(not np.array_equal(x, y) for x, y in zip(leaves, actual, strict=True)):
                    raise ValueError("restricted tiled SD requires the explicit conjugate source partner.")
            return

        if not isinstance(self.orb_data_up_spin, MOs_data):
            raise ValueError(f"SD_data.orb_data_up_spin must be MOs_data; got {type(self.orb_data_up_spin).__name__}.")
        if not isinstance(self.orb_data_dn_spin, MOs_data):
            raise ValueError(f"SD_data.orb_data_dn_spin must be MOs_data; got {type(self.orb_data_dn_spin).__name__}.")
        if min(int(self.num_electron_up), int(self.num_electron_dn)) > 0:
            # An empty spin channel enters no determinant: nothing to pair.
            _require_opposite_spin_twists(self.orb_data_up_spin, self.orb_data_dn_spin)

        if int(self.orb_data_up_spin.num_mo) != self.num_occ_up:
            raise ValueError(
                "SD_data must store exactly num_electron_up occupied up-spin rows; "
                f"got num_mo={self.orb_data_up_spin.num_mo}, expected {self.num_occ_up}."
            )
        if int(self.orb_data_dn_spin.num_mo) != self.num_occ_dn:
            raise ValueError(
                "SD_data must store exactly num_electron_dn occupied down-spin rows; "
                f"got num_mo={self.orb_data_dn_spin.num_mo}, expected {self.num_occ_dn}."
            )
        if self.time_reversal and (self.orb_data_up_spin.identity or self.orb_data_dn_spin.identity):
            if not (self.orb_data_up_spin.identity and self.orb_data_dn_spin.identity):
                raise ValueError("SD_data time-reversal partners must both use implicit identity MOs or both use dense MOs.")
            if self.num_occ_up != self.num_occ_dn:
                raise ValueError("implicit identity time-reversal MOs currently require a closed shell (N_up == N_dn).")
        elif self.time_reversal:
            expected_dn = np.conj(np.asarray(self.orb_data_up_spin.mo_coefficients)[: self.num_occ_dn])
            actual_dn = np.asarray(self.orb_data_dn_spin.mo_coefficients)
            if actual_dn.shape != expected_dn.shape or not np.allclose(
                actual_dn,
                expected_dn,
                atol=GEMINAL_REPRESENTATION_ATOL,
                rtol=RTOL_BIT_EXACT,
            ):
                raise ValueError(
                    "SD_data time_reversal=True requires down-spin occupied "
                    "coefficients to equal conj(up[:num_electron_dn]). Use "
                    "time_reversal=False for independent spin orbitals."
                )

        self.orb_data_up_spin.sanity_check()
        self.orb_data_dn_spin.sanity_check()

    @classmethod
    def build_tiled(cls, *, up_source: TiledOrbitalSet_data, down_source: TiledOrbitalSet_data | None = None) -> "SD_data":
        """Build frozen SD with explicit BCS (+k_s, -k_s) spin sectors.

        Both supplied sources are in the positive sector. An omitted down
        source explicitly requests the restricted shared-source construction;
        independent LSDA sources are never inferred from electron counts.
        """
        up_source.sanity_check()
        restricted = down_source is None
        source = up_source if restricted else down_source
        source.sanity_check()
        if source.tiling_matrix != up_source.tiling_matrix or not _structure_data_equal(
            source.structure_data, up_source.structure_data
        ):
            raise ValueError("tiled spin sources must share the physical supercell.")
        delta = np.asarray(source.supertwist) - np.asarray(up_source.supertwist)
        if not np.allclose(delta, np.rint(delta), atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT):
            raise ValueError("both tiled source spin channels must be in the same +k_s sector before BCS conjugation.")
        data = cls(
            num_electron_up=up_source.num_mo,
            num_electron_dn=source.num_mo,
            orb_data_up_spin=up_source.replace(spin="up"),
            orb_data_dn_spin=conjugate_tiled_orbitals(source, spin="down"),
            time_reversal=restricted,
        )
        data.sanity_check()
        return data

    @classmethod
    def build(
        cls,
        *,
        num_electron_up: int,
        num_electron_dn: int,
        mos_up: MOs_data,
        mos_dn: MOs_data | None = None,
        time_reversal: bool | None = None,
        fix_geminal_rank: int | None = None,
    ) -> "SD_data":
        """Construct an ``SD_data``.

        Args:
            num_electron_up: Number of up-spin electrons. Equal to
                ``num_electron_dn`` for a closed-shell singlet; allowed to
                differ for open-shell.
            num_electron_dn: Number of down-spin electrons.
            mos_up: Up-spin MOs_data. Only the first ``num_electron_up``
                occupied rows are retained.
            mos_dn: Down-spin MOs_data. Auto-derived from ``mos_up`` via
                ``conj`` when ``time_reversal=True`` and ``mos_dn`` is
                ``None``. Only the first ``num_electron_dn`` occupied rows
                are retained.
            time_reversal: See ``AGP_data.build``; if ``None``, inferred
                from electron counts.
            fix_geminal_rank: Ignored by ``SD_data``; rank is structurally
                ``N_occ``. A non-``None`` value triggers a warning.
        """
        if isinstance(mos_up, TiledOrbitalSet_data) or isinstance(mos_dn, TiledOrbitalSet_data):
            raise ValueError("use SD_data.build_tiled with explicit positive-sector sources for tiled orbitals.")
        n_up = int(num_electron_up)
        n_dn = int(num_electron_dn)
        if time_reversal is None:
            time_reversal = n_up == n_dn

        if fix_geminal_rank is not None:
            logger.warning(
                f"SD_data ignores fix_geminal_rank={fix_geminal_rank}; "
                f"rank is structurally N_occ={n_dn}. Did you mean AGPn_data?"
            )

        if int(mos_up.num_mo) < n_up:
            raise ValueError(f"SD_data needs {n_up} occupied up-spin MOs; only {mos_up.num_mo} were supplied.")
        if mos_up.identity:
            if int(mos_up.num_mo) != n_up:
                raise ValueError(
                    "implicit identity up-spin MOs cannot be row-truncated; build the tiled AO leaf with exactly N_up rows."
                )
        else:
            mos_up = mos_up.replace(
                num_mo=n_up,
                mo_coefficients=np.ascontiguousarray(np.asarray(mos_up.mo_coefficients)[:n_up]),
            )

        if time_reversal and mos_dn is None:
            # BCS / TR-pair auto-derive: dn-channel = (conj(C^k), -k AO leaf).
            # Real / Gamma collapses to (C, +k AO leaf) byte-identically.
            if hasattr(mos_up.aos_data, "with_negated_k"):
                _aos_dn = mos_up.aos_data.with_negated_k()
            else:
                _aos_dn = mos_up.aos_data
            if mos_up.identity:
                if n_dn != n_up:
                    raise ValueError("implicit identity time-reversal MOs currently require N_up == N_dn.")
                mos_dn = MOs_data.build_identity(aos_data=_aos_dn)
            else:
                new_coeffs = np.conj(np.asarray(mos_up.mo_coefficients)[:n_dn])
                mos_dn = MOs_data(
                    num_mo=n_dn,
                    aos_data=_aos_dn,
                    mo_coefficients=new_coeffs,
                )
        elif mos_dn is None:
            raise ValueError(
                "mos_dn is required when time_reversal=False; the two spin channels "
                "are variationally independent in the open-shell / FM regime."
            )

        else:
            if int(mos_dn.num_mo) < n_dn:
                raise ValueError(f"SD_data needs {n_dn} occupied down-spin MOs; only {mos_dn.num_mo} were supplied.")
            if mos_dn.identity:
                if int(mos_dn.num_mo) != n_dn:
                    raise ValueError(
                        "implicit identity down-spin MOs cannot be row-truncated; build the tiled AO leaf with exactly N_dn rows."
                    )
            else:
                mos_dn = mos_dn.replace(
                    num_mo=n_dn,
                    mo_coefficients=np.ascontiguousarray(np.asarray(mos_dn.mo_coefficients)[:n_dn]),
                )

        data = cls(
            num_electron_up=n_up,
            num_electron_dn=n_dn,
            orb_data_up_spin=mos_up,
            orb_data_dn_spin=mos_dn,
            time_reversal=bool(time_reversal),
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        info_lines = [
            "**" + self.__class__.__name__,
            f"  occupied MO rows (up, dn) = ({self.num_occ_up}, {self.num_occ_dn})",
            f"  num_electron_up = {self.num_electron_up}",
            f"  num_electron_dn = {self.num_electron_dn}",
            f"  time_reversal = {self.time_reversal}",
        ]
        info_lines.extend(self.orb_data_up_spin._get_info())
        info_lines.extend(self.orb_data_dn_spin._get_info())
        return info_lines

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)

    def collect_param_grads(self, grad_geminal: "SD_data") -> dict[str, object]:
        """Collect occupied-MO gradients without routing through lambda storage."""
        if self.is_tiled_representation:
            raise NotImplementedError("tiled orbital/geminal optimization is frozen in this release.")
        grads = super().collect_param_grads(grad_geminal)
        grad_up = grad_geminal.orb_data_up_spin.mo_coefficients
        grad_dn = grad_geminal.orb_data_dn_spin.mo_coefficients
        if self.time_reversal:
            grad_dn_embedded = _embed_time_reversal_down_rows(
                grad_up,
                grad_dn,
                xp=jnp,
            )
            grads["mo_coefficients_up"] = grad_up + jnp.conj(grad_dn_embedded)
        else:
            grads["mo_coefficients_up"] = grad_up
            grads["mo_coefficients_dn"] = grad_dn
        return grads

    def symmetrize_lambda(self, mat: npt.NDArray) -> NoReturn:
        """Reject obsolete lambda operations on a canonical occupied SD."""
        raise TypeError("SD_data has no lambda matrix to symmetrize; update its occupied MO coefficient blocks instead.")

    def to_AGPn(self) -> "AGPn_data":
        """Promote a closed-shell occupied SD to rank-``Nocc`` AGPn.

        No virtual orbitals are invented.  Callers that want a larger AGPn
        manifold must supply those natural orbitals explicitly when building
        the AGPn object.
        """
        if self.is_tiled_representation:
            raise NotImplementedError("tiled SD cannot be promoted to an optimizable AGPn.")
        if self.num_occ_up != self.num_occ_dn:
            raise ValueError(
                "Open-shell SD -> AGPn requires explicit common-rank natural orbitals; no virtual rows are stored on SD_data."
            )
        return AGPn_data.build(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            mos_up=self.orb_data_up_spin,
            mos_dn=self.orb_data_dn_spin,
            lambda_diag=np.ones((self.num_occ_dn,), dtype=np.float64),
            time_reversal=self.time_reversal,
        )

    def to_AGP(self) -> AGP_data:
        """Convert the occupied Slater factors directly to an AO-basis AGP."""
        if self.is_tiled_representation:
            raise NotImplementedError("tiled SD cannot materialize a global AO coefficient matrix.")
        C_up = np.asarray(self.orb_data_up_spin.mo_coefficients)
        C_dn = np.asarray(self.orb_data_dn_spin.mo_coefficients)
        paired = C_up[: self.num_occ_dn].T @ C_dn
        unpaired = C_up[self.num_occ_dn : self.num_occ_up].T
        lambda_ao = np.hstack([paired, unpaired])
        return AGP_data.build(
            num_electron_up=self.num_electron_up,
            num_electron_dn=self.num_electron_dn,
            orb_data_up_spin=self.orb_data_up_spin.aos_data,
            orb_data_dn_spin=self.orb_data_dn_spin.aos_data,
            lambda_matrix=lambda_ao,
            time_reversal=self.time_reversal,
        )

    def apply_block_update(self, block: "VariationalParameterBlock") -> "SD_data":
        r"""Apply a variational-parameter block update.

        Recognised blocks:

        * ``"mo_coefficients_up"``: orthonormality-preserving retraction on
          the up-channel coefficients. When ``time_reversal=True`` the
          dn-channel is auto-derived as ``conj(C_up)``.
        * ``"mo_coefficients_dn"``: open-shell / UHF only
          (``time_reversal=False``). Independent orthonormality-preserving
          retraction on the dn-channel coefficients. A no-op when
          ``time_reversal=True``.

        ``SD_data`` exposes no lambda blocks. Other block names are no-ops.

        Notes:
            The detailed retraction construction is work in progress.
        """
        if self.is_tiled_representation:
            if block.name in (
                "mo_coefficients_up",
                "mo_coefficients_dn",
                "lambda_matrix",
                "lambda_basis_exp",
                "lambda_basis_coeff",
            ):
                raise NotImplementedError("tiled orbital/geminal optimization is frozen in this release.")
            return self
        if block.name == "mo_coefficients_up":
            new_C_up = np.asarray(block.values)
            new_C_up = _smetric_polar_retract(
                new_C_up,
                self.orb_data_up_spin.aos_data,
                metric_sqrt=block.tangent_spec.metric_sqrt,
            )
            new_mos_up = self.orb_data_up_spin.replace(
                mo_coefficients=new_C_up,
            )
            if self.time_reversal:
                new_C_dn = np.conj(new_C_up[: self.num_occ_dn])
                new_mos_dn = self.orb_data_dn_spin.replace(
                    mo_coefficients=new_C_dn,
                )
            else:
                new_mos_dn = self.orb_data_dn_spin
            return self.replace(
                orb_data_up_spin=new_mos_up,
                orb_data_dn_spin=new_mos_dn,
            )
        if block.name == "mo_coefficients_dn":
            if self.time_reversal:
                # Defensive no-op: get_variational_blocks should not
                # have exposed this block when time_reversal=True.
                return self
            new_C_dn = np.asarray(block.values)
            new_C_dn = _smetric_polar_retract(
                new_C_dn,
                self.orb_data_dn_spin.aos_data,
                metric_sqrt=block.tangent_spec.metric_sqrt,
            )
            new_mos_dn = self.orb_data_dn_spin.replace(mo_coefficients=new_C_dn)
            return self.replace(orb_data_dn_spin=new_mos_dn)
        return self


# ---------------------------------------------------------------------------
# Conversion API (free functions).
#
# These are dispatch-style free functions that route to the appropriate
# per-class method. They are the user-facing entry points; the per-class
# methods (``AGPn_data.to_AGP``, ``SD_data.to_AGPn``, etc.) remain available
# for callers that already hold a concretely-typed instance.
# ---------------------------------------------------------------------------


def to_AGP(geminal: AGP_data | AGPn_data | SD_data) -> AGP_data:
    """Materialise ``geminal`` to an ``AGP_data`` (full dense lambda).

    Dispatches on the concrete subtype:

    * ``AGP_data``: returned as-is (no copy).
    * ``AGPn_data``: ``lambda_matrix = diag(lambda_diag)``.
    * ``SD_data``: directly contracts occupied factors into an AO-basis
      ``AGP_data``. No lambda is created or retained on the source SD.
    """
    if isinstance(geminal, AGP_data):
        return geminal
    if isinstance(geminal, (AGPn_data, SD_data)):
        return geminal.to_AGP()
    raise TypeError(f"to_AGP: unsupported geminal type {type(geminal).__name__}. Expected AGP_data, AGPn_data, or SD_data.")


def to_AGPn(
    geminal: AGP_data | AGPn_data | SD_data,
    n: int | None = None,
) -> AGPn_data:
    r"""Convert ``geminal`` to an ``AGPn_data`` with the natural-orbital
    diagonal form.

    Args:
        geminal: Source geminal in any of the three supported subtypes.
        n: Target rank. If ``None``, preserves the current rank (no
            truncation). If ``int < current_num_mo``, truncates the
            natural-orbital occupations to the top ``n``.

    Notes:
        * ``AGP_data`` -> ``AGPn_data`` routes AO representations through
          ``AGP_data.convert_from_AOs_to_MOs`` and converts MO
          representations into the diagonal natural-orbital form.
          Open-shell unpaired columns are transformed by the same up-side
          basis change and stored in ``AGPn_data.lambda_unpaired``.
        * ``AGPn_data`` -> ``AGPn_data`` is a no-op or rank truncation.
        * closed-shell ``SD_data`` -> ``AGPn_data`` uses exactly its
          occupied rows and an all-one ``lambda_diag``. It invents neither
          virtual rows nor an unpaired lambda block.

        The detailed natural-orbital construction is work in progress.
    """
    if isinstance(geminal, AGP_data):
        n_up = int(geminal.num_electron_up)
        n_dn = int(geminal.num_electron_dn)
        n_unpaired = n_up - n_dn
        if n_unpaired < 0:
            raise ValueError(
                f"to_AGPn(AGP_data): num_electron_up ({n_up}) < num_electron_dn ({n_dn}); "
                f"unpaired block dimensions would be negative."
            )

        if _is_ao_leaf(geminal.orb_data_up_spin):
            # AO-rep: route through the existing ``convert_from_AOs_to_MOs``
            # machinery. Its output is an MO-basis AGP_data with
            # ``lambda_matrix = hstack([diag(s), mo_lambda_unpaired])``
            # where ``mo_lambda_unpaired`` has shape ``(num_mo, n_unpaired)``
            # (empty in the closed-shell case).
            num_eig = int(n) if n is not None else "all"
            agp_mo = AGP_data.convert_from_AOs_to_MOs(geminal, num_eigenvectors=num_eig)
            num_mo_after = int(agp_mo.orb_data_up_spin.num_mo)
            lam_after = np.asarray(agp_mo.lambda_matrix)
            paired_after = lam_after[:, :num_mo_after]
            unpaired_after = lam_after[:, num_mo_after:] if n_unpaired > 0 else None
            lambda_diag_raw = np.diag(paired_after)
            if np.iscomplexobj(lambda_diag_raw) and not np.allclose(
                np.imag(lambda_diag_raw), 0.0, atol=atol_consistency, rtol=rtol_consistency
            ):
                raise ValueError("to_AGPn(AGP_data): diagonal occupations are not real within tolerance.")
            lambda_diag_full = np.asarray(np.real_if_close(lambda_diag_raw), dtype=np.float64)
            return AGPn_data.build(
                num_electron_up=int(agp_mo.num_electron_up),
                num_electron_dn=int(agp_mo.num_electron_dn),
                mos_up=agp_mo.orb_data_up_spin,
                mos_dn=agp_mo.orb_data_dn_spin,
                lambda_diag=lambda_diag_full,
                lambda_unpaired=(np.ascontiguousarray(unpaired_after) if unpaired_after is not None else None),
                time_reversal=getattr(agp_mo, "time_reversal", True),
            )

        # MO-rep AGP_data: split ``lambda_matrix`` into paired and unpaired
        # blocks, SVD the paired block to diagonalise it, and absorb U / Vh
        # into the MO coefficients. The unpaired block is left-multiplied
        # by U.conj().T to follow the basis change. Open-shell support:
        # AGPn carries the transformed ``lambda_unpaired`` block. Canonical
        # SD does not enter this AGP -> AGPn conversion branch.
        lambda_mat = np.asarray(geminal.lambda_matrix)
        num_mo_now = int(geminal.orb_data_up_spin.num_mo)
        if lambda_mat.shape[1] < num_mo_now:
            raise ValueError(
                f"to_AGPn(AGP_data): MO-rep lambda_matrix has fewer columns ({lambda_mat.shape[1]}) than num_mo ({num_mo_now})."
            )
        paired_mat = lambda_mat[:, :num_mo_now]
        unpaired_mat = lambda_mat[:, num_mo_now:] if n_unpaired > 0 else None
        if unpaired_mat is not None and unpaired_mat.shape[1] != n_unpaired:
            raise ValueError(
                f"to_AGPn(AGP_data): MO-rep lambda unpaired columns "
                f"({unpaired_mat.shape[1]}) do not match n_unpaired ({n_unpaired})."
            )
        if paired_mat.shape[0] != paired_mat.shape[1]:
            raise ValueError(f"to_AGPn(AGP_data): MO-rep paired lambda must be square, got shape {paired_mat.shape}.")
        C_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
        C_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients)
        time_reversal = bool(getattr(geminal, "time_reversal", True))
        if time_reversal:
            _require_time_reversal_mos(geminal, paired_mat, context="to_AGPn(AGP_data)")
            paired_hermitian = 0.5 * (paired_mat + paired_mat.conj().T)
            s_vals, u_mat = np.linalg.eigh(paired_hermitian)
            order = _magnitude_order(s_vals)
            s_vals = np.asarray(s_vals[order], dtype=np.float64)
            u_mat = np.asarray(u_mat[:, order])
            new_C_up = u_mat.T @ C_up
            new_C_dn = np.conj(new_C_up)
        else:
            u_mat, s_vals, v_h = np.linalg.svd(paired_mat, full_matrices=False)
            s_vals = np.asarray(s_vals, dtype=np.float64)
            # Bilinear absorption: (U.T C_up).T diag(s) (Vh C_dn)
            # reproduces C_up.T (U diag(s) Vh) C_dn.
            new_C_up = u_mat.T @ C_up
            new_C_dn = v_h @ C_dn

        # The unpaired columns transform with the Hermitian left inverse of
        # the up-side basis change, for both the TR and independent cases.
        new_unpaired = u_mat.conj().T @ unpaired_mat if unpaired_mat is not None else None

        n_target = (
            int(s_vals.size)
            if n is None
            else _validate_factor_rank(
                int(n),
                int(s_vals.size),
                n_up,
                context="to_AGPn(AGP_data)",
            )
        )
        _validate_magnitude_truncation(
            s_vals,
            n_target,
            new_unpaired,
            context="to_AGPn(AGP_data)",
            require_sorted=False,
        )
        new_C_up = new_C_up[:n_target]
        new_C_dn = new_C_dn[:n_target]
        new_lambda_diag = np.asarray(s_vals[:n_target], dtype=np.float64)
        if new_unpaired is not None:
            new_unpaired = new_unpaired[:n_target]

        new_mos_up = MOs_data(
            num_mo=n_target,
            aos_data=geminal.orb_data_up_spin.aos_data,
            mo_coefficients=new_C_up,
        )
        new_mos_dn = MOs_data(
            num_mo=n_target,
            aos_data=geminal.orb_data_dn_spin.aos_data,
            mo_coefficients=new_C_dn,
        )
        return AGPn_data.build(
            num_electron_up=n_up,
            num_electron_dn=n_dn,
            mos_up=new_mos_up,
            mos_dn=new_mos_dn,
            lambda_diag=new_lambda_diag,
            lambda_unpaired=(np.ascontiguousarray(new_unpaired) if new_unpaired is not None else None),
            time_reversal=time_reversal,
        )
    if isinstance(geminal, SD_data):
        agpn = geminal.to_AGPn()
        if n is None or int(n) == agpn.num_mo:
            return agpn
        # Caller asked for a truncation: pass through the AGPn path.
        return to_AGPn(agpn, n=n)
    if isinstance(geminal, AGPn_data):
        n_up = int(geminal.num_electron_up)
        diag = np.asarray(geminal.lambda_diag)
        unp = np.asarray(geminal.lambda_unpaired)
        n_target = (
            geminal.num_mo
            if n is None
            else _validate_factor_rank(
                int(n),
                geminal.num_mo,
                n_up,
                context="to_AGPn(AGPn_data)",
            )
        )
        if geminal.time_reversal:
            _require_time_reversal_mos(geminal, np.diag(diag), context="to_AGPn(AGPn_data)")
        _validate_magnitude_truncation(
            diag,
            n_target,
            unp if unp.shape[1] > 0 else None,
            context="to_AGPn(AGPn_data)",
            require_sorted=True,
        )
        if n_target == geminal.num_mo:
            return geminal
        # Top-n truncation: keep the first n entries of lambda_diag and the
        # corresponding MO rows. This is a one-shot operation at
        # construction; lambda_diag is already
        # sorted descending by the loader / SVD convention. For open-shell
        # the unpaired block is also truncated to the same row count
        # (n_target >= n_up keeps all identity rows intact).
        new_lambda = diag[:n_target]
        new_C_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)[:n_target]
        new_C_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients)[:n_target]
        new_mos_up = MOs_data(num_mo=n_target, aos_data=geminal.orb_data_up_spin.aos_data, mo_coefficients=new_C_up)
        new_mos_dn = MOs_data(num_mo=n_target, aos_data=geminal.orb_data_dn_spin.aos_data, mo_coefficients=new_C_dn)
        if unp.shape[1] == 0:
            new_unp = None
        else:
            new_unp = unp[:n_target]
        return AGPn_data.build(
            num_electron_up=geminal.num_electron_up,
            num_electron_dn=geminal.num_electron_dn,
            mos_up=new_mos_up,
            mos_dn=new_mos_dn,
            lambda_diag=new_lambda,
            lambda_unpaired=new_unp,
            time_reversal=geminal.time_reversal,
        )
    raise TypeError(f"to_AGPn: unsupported geminal type {type(geminal).__name__}. Expected AGP_data, AGPn_data, or SD_data.")


def to_SD(geminal: AGP_data | AGPn_data | SD_data) -> SD_data:
    """Demote ``geminal`` to ``SD_data`` (single Slater).

    Valid only when the source geminal already represents an SD (occupation
    pattern is exactly ``[1]*N_occ + [0]*N_virt``).

    Raises:
        ValueError: if the source geminal carries non-trivial pair
            correlation (any ``lambda_diag[i] in (0, 1)`` violated).
    """
    if isinstance(geminal, SD_data):
        return geminal
    if isinstance(geminal, AGPn_data):
        n_occ = int(geminal.num_electron_dn)
        diag = np.asarray(geminal.lambda_diag)
        is_sd = (
            diag.shape[0] >= n_occ
            and np.allclose(diag[:n_occ], 1.0, atol=GEMINAL_REPRESENTATION_ATOL)
            and np.allclose(diag[n_occ:], 0.0, atol=GEMINAL_REPRESENTATION_ATOL)
        )
        if not is_sd:
            raise ValueError(
                "to_SD(AGPn_data): the source carries non-trivial pair "
                "correlation (lambda_diag is not [1]*N_occ + [0]*N_virt). "
                "Use to_AGPn or to_AGP for representations that retain the "
                "occupations."
            )
        return SD_data.build(
            num_electron_up=geminal.num_electron_up,
            num_electron_dn=geminal.num_electron_dn,
            mos_up=geminal.orb_data_up_spin,
            mos_dn=geminal.orb_data_dn_spin,
            time_reversal=geminal.time_reversal,
        )
    if isinstance(geminal, AGP_data):
        raise NotImplementedError("to_SD(AGP_data) requires AGP -> AGPn conversion first; see to_AGPn.")
    raise TypeError(f"to_SD: unsupported geminal type {type(geminal).__name__}. Expected AGP_data, AGPn_data, or SD_data.")


def dispatch_loaded_geminal(
    *,
    num_electron_up: int,
    num_electron_dn: int,
    orb_data_up_spin: AOs_base_data | MOs_data,
    orb_data_dn_spin: AOs_base_data | MOs_data,
    lambda_matrix: npt.ArrayLike,
    time_reversal: bool | None = None,
    atol: float = GEMINAL_REPRESENTATION_ATOL,
) -> AGP_data | AGPn_data | SD_data:
    """Load-time geminal-subclass dispatcher.

    Given an explicit lambda-bearing
    ``(orb_data_up, orb_data_dn, lambda_matrix)`` triple, pick the most
    specific :class:`AGP_data` / :class:`AGPn_data` / :class:`SD_data`
    instance that exactly represents it.  Canonical QE, TREXIO, and HDF5 SD
    loaders do not use this boundary: they construct occupied-only
    :class:`SD_data` directly.  This dispatcher remains for lambda-based AGP /
    AGPn twist payloads and explicit in-memory representation conversion.

    Dispatch table:

    =======================================================  ============
    source form                                               target
    =======================================================  ============
    AO orb_data                                               ``AGP_data``
    MO orb_data, lambda_paired diagonal & I on first N_occ    ``SD_data`` (selector folded into occupied MO rows)
    MO orb_data, lambda_paired diagonal with general values   ``AGPn_data`` (``lambda_diag`` + ``lambda_unpaired`` variational)
    MO orb_data, lambda_paired non-diagonal                   ``ValueError`` (caller must call ``to_AGP`` explicitly)
    =======================================================  ============

    Open-shell: the paired sub-block (first ``n_dn`` columns) decides the
    target class via the rules above.  For SD, the unpaired selector is
    folded into the excess occupied up-spin rows at this boundary; for
    AGPn it remains the variational ``lambda_unpaired`` field.

    Args:
        num_electron_up: Number of up-spin electrons.
        num_electron_dn: Number of down-spin electrons.
        orb_data_up_spin: Loaded up-spin orbital data.
        orb_data_dn_spin: Loaded down-spin orbital data.
        lambda_matrix: Loaded lambda in the same basis as ``orb_data``.
        time_reversal: If ``None``, inferred from electron counts (``True``
            iff equal -- the closed-shell-singlet default).
        atol: Tolerance for the "approximately diagonal" / "approximately I"
            checks on an explicitly supplied lambda representation.
    """
    if _is_ao_leaf(orb_data_up_spin) or _is_ao_leaf(orb_data_dn_spin):
        return AGP_data.build(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            orb_data_up_spin=orb_data_up_spin,
            orb_data_dn_spin=orb_data_dn_spin,
            lambda_matrix=lambda_matrix,
            time_reversal=time_reversal,
        )

    if not (isinstance(orb_data_up_spin, MOs_data) and isinstance(orb_data_dn_spin, MOs_data)):
        raise TypeError(
            f"dispatch_loaded_geminal: orb_data must be an AO leaf or MOs_data; "
            f"got up={type(orb_data_up_spin).__name__}, dn={type(orb_data_dn_spin).__name__}."
        )

    n_up = int(num_electron_up)
    n_dn = int(num_electron_dn)
    if time_reversal is None:
        time_reversal = n_up == n_dn

    lam = np.asarray(lambda_matrix)
    n_mo_dn = int(orb_data_dn_spin.num_mo)
    n_unpaired = n_up - n_dn

    # Split paired (first n_mo_dn columns) and unpaired (remaining) blocks.
    # The TREXIO / QE loaders pack ``lambda = hstack([paired, unpaired])``
    # with shapes ``(n_mo, n_mo)`` and ``(n_mo, n_unpaired)`` respectively.
    if lam.shape[1] < n_mo_dn:
        raise ValueError(
            f"dispatch_loaded_geminal: lambda_matrix has fewer columns "
            f"({lam.shape[1]}) than the dn MO count ({n_mo_dn}); cannot split paired / unpaired."
        )
    paired = lam[:, :n_mo_dn]
    unpaired = lam[:, n_mo_dn:]
    if unpaired.shape[1] != n_unpaired:
        raise ValueError(
            f"dispatch_loaded_geminal: unpaired-block columns ({unpaired.shape[1]}) "
            f"do not match num_electron_up - num_electron_dn ({n_unpaired})."
        )

    # Closed/open-shell shared: check paired diagonality.
    paired_diag = np.diag(paired)
    paired_off_diag = paired - np.diag(paired_diag)
    if not np.allclose(paired_off_diag, 0.0, atol=atol):
        # Refuse to silently materialise a non-diagonal MO
        # paired lambda. The caller can convert explicitly via ``to_AGP``.
        raise ValueError(
            "dispatch_loaded_geminal: loaded MO-basis paired lambda is non-diagonal; "
            "cannot dispatch to AGPn_data (diag form) or SD_data (identity form). "
            "Call ``to_AGP(loaded_geminal)`` explicitly to materialise an "
            "AO-basis AGP_data."
        )

    # Diagonal occupations must be real (natural-orbital occupations
    # are physical real invariants even on the complex Bloch path).
    if np.iscomplexobj(paired_diag) and not np.allclose(np.imag(paired_diag), 0.0, atol=atol):
        raise ValueError(
            "dispatch_loaded_geminal: loaded MO-basis paired lambda has a non-trivial "
            "imaginary diagonal; natural-orbital occupations must be real."
        )
    diag_real = np.real(paired_diag).astype(np.float64)

    # SD pattern on the paired block: [1]*N_occ + [0]*N_virt.
    # ``N_occ = num_electron_dn`` in the closed-shell convention (paired
    # block) and equals num_electron_dn even open-shell (paired rows are
    # for the lower-spin electrons; the excess up electrons go into the
    # unpaired block on the right).
    n_occ = n_dn
    is_sd = (
        n_occ <= n_mo_dn and np.allclose(diag_real[:n_occ], 1.0, atol=atol) and np.allclose(diag_real[n_occ:], 0.0, atol=atol)
    )
    # Materialise an empty (0, 0) lambda_unpaired for closed-shell AGPn.
    # For open-shell AGPn, pass through the supplied unpaired columns.
    if n_unpaired == 0:
        unpaired_arg = None
    else:
        unpaired_arg = np.ascontiguousarray(unpaired)

    if is_sd:
        # Fold the AGP-compatible selector form into the occupied up-spin rows
        # exactly once at this representation boundary.  No lambda survives
        # on the returned SD_data.  General unpaired columns are linear
        # combinations of the supplied up MOs, so absorb them into the excess
        # occupied orbitals instead of preserving a structural selector.
        C_up = np.asarray(orb_data_up_spin.mo_coefficients)
        paired_occ = C_up[:n_dn]
        if n_unpaired:
            excess_occ = np.asarray(unpaired).T @ C_up
            C_up_occ = np.vstack([paired_occ, excess_occ])
        else:
            C_up_occ = paired_occ
        mos_up_occ = orb_data_up_spin.replace(
            num_mo=n_up,
            mo_coefficients=np.ascontiguousarray(C_up_occ),
        )
        return SD_data.build(
            num_electron_up=num_electron_up,
            num_electron_dn=num_electron_dn,
            mos_up=mos_up_occ,
            mos_dn=orb_data_dn_spin,
            time_reversal=time_reversal,
        )

    # Diagonal but non-SD -> AGPn_data with the natural-orbital occupations.
    return AGPn_data.build(
        num_electron_up=num_electron_up,
        num_electron_dn=num_electron_dn,
        mos_up=orb_data_up_spin,
        mos_dn=orb_data_dn_spin,
        lambda_diag=diag_real,
        lambda_unpaired=unpaired_arg,
        time_reversal=time_reversal,
    )


# ===========================================================================
# DETERMINANT KERNELS
# ===========================================================================
# The remaining top-level defs are the numerical kernels: log-det / det /
# antisymmetrization, geminal matrix elements, ratio-update, and the
# grads-&-Laplacian of ln|Det|. Public ``compute_*`` entry points are paired
# with ``_*_debug`` oracles and (for AD) ``_*_fwd`` / ``_*_bwd`` custom_vjp
# halves, per the naming convention documented at the top of the module.
# ---------------------------------------------------------------------------

_GeminalLike = AGP_data | AGPn_data | SD_data
_OrbitalTables = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
_GradLapResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _geminal_equilibration_scales(G: jax.Array, max_exponent: int = 1021) -> tuple[jax.Array, jax.Array]:
    """Exact power-of-two row scales of ``G`` and column scales of the row-scaled ``G``.

    Each scale brings an infinity norm into ``[0.5, 1)``; a zero or non-finite norm keeps scale 1.
    The exponents are clipped to ``max_exponent``: 1021 keeps every scale a normal power of two,
    and the pseudo-inverse passes ``GEMINAL_EQUILIBRATION_MAX_EXPONENT`` (issue #605) so that its
    unscaling cannot overflow. The scales are piecewise constant in ``G`` (zero derivative almost
    everywhere), so they are computed under ``stop_gradient``.
    """

    def _pow2_scale(norm: jax.Array) -> jax.Array:
        # norm = m * 2**e with 0.5 <= m < 1  ->  scale 2**-e (exact power of two).
        _, exponent = jnp.frexp(norm)
        exponent = jnp.clip(exponent, -max_exponent, max_exponent)
        usable = jnp.isfinite(norm) & (norm > 0)
        return jnp.where(usable, jnp.ldexp(jnp.ones_like(norm), -exponent), jnp.ones_like(norm))

    # The scales are piecewise constant in G (their derivative is zero almost
    # everywhere): stop_gradient keeps any differentiated caller's tangent the
    # one of the plain inverse and keeps frexp out of the autodiff graph.
    row_scale = _pow2_scale(jax.lax.stop_gradient(jnp.max(jnp.abs(G), axis=1)))
    col_scale = _pow2_scale(jax.lax.stop_gradient(jnp.max(jnp.abs(G * row_scale[:, jnp.newaxis]), axis=0)))
    return row_scale, col_scale


def _equilibrated_geminal_pinv(G: jax.Array, eps_rcond: float) -> jax.Array:
    """Invert a geminal after exact power-of-two row and column equilibration.

    This is the common recipe for a fresh determinant inverse and the sampler
    seed/refresh inverse. Threshold the equilibrated matrix with the caller's
    unchanged rcond, then undo its scales. In particular, open-shell paired
    columns scale quadratically with an MO normalization and unpaired columns
    linearly: a raw relative SVD cutoff must not discard those directions.
    Zero/nonfinite rows and columns keep unit scales; truly singular directions
    retain the existing thresholded behavior. Scales have zero AD tangent. The
    scale exponents are clipped to ``GEMINAL_EQUILIBRATION_MAX_EXPONENT`` (issue
    #605): the unscale factor is then at most 2**956 and ``|A^+|`` at most
    ``1/(rcond s_max)``, so the result is finite for every finite ``G``; a row or
    column with an infinity norm below about 1e-144 is only partly scaled.
    """

    row_scale, col_scale = _geminal_equilibration_scales(G, GEMINAL_EQUILIBRATION_MAX_EXPONENT)
    A = G * row_scale[:, jnp.newaxis]
    A = A * col_scale[jnp.newaxis, :]
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    s_inv = jnp.where(s > eps_rcond * s[0], 1.0 / s, 0.0)
    A_pinv = (jnp.conj(Vt).T * s_inv[jnp.newaxis, :]) @ jnp.conj(U).T
    return col_scale[:, jnp.newaxis] * A_pinv * row_scale[jnp.newaxis, :]


def _equilibrated_geminal_lu_inverse(G: jax.Array) -> jax.Array:
    """LU inverse of a geminal after the same power-of-two equilibration as :func:`_equilibrated_geminal_pinv`.

    ``C (R G C)^-1 R`` with the row and column scales of the pseudo-inverse recipe. For an
    invertible ``G`` it equals the equilibrated pseudo-inverse (whose ``rcond_svd`` cutoff of
    1e-20 removes nothing from an invertible fp64 matrix) to round-off. It does not threshold:
    the caller checks it with :func:`_equilibrated_geminal_lu_inverse_and_residual` and falls back
    to the pseudo-inverse for a numerically singular ``G`` (issue #605).
    """
    return _equilibrated_geminal_lu_inverse_and_residual(G)[0]


def _equilibrated_geminal_lu_inverse_and_residual(G: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Equilibrated LU inverse of one geminal and its scale-free probe residual (issue #605).

    Returns ``(C Y R, max|A (Y P) - P|)`` with ``A = R G C``, ``Y = A^-1`` and ``P`` the two fixed
    probe columns (all ones and ``cos(1 + 0.7 j)``). The residual is measured on the equilibrated
    matrix, so a graded but invertible ``G`` (a far electron) does not inflate it, while an LU
    that silently returns garbage for a numerically rank-deficient ``G`` gives a residual of order
    one. A non-finite inverse gives a non-finite residual. The LU keeps the full exponent range, so
    a representable inverse of a strongly graded ``G`` (a row of order 1e-200) stays exact; an
    inverse that overflows is non-finite and the caller falls back to the clipped pseudo-inverse.
    """
    row_scale, col_scale = _geminal_equilibration_scales(G)
    A = (G * row_scale[:, jnp.newaxis]) * col_scale[jnp.newaxis, :]
    Y = jnp.linalg.inv(A)
    n = G.shape[-1]
    index = jnp.arange(n, dtype=jnp.float64)
    probes = jnp.stack([jnp.ones(n, dtype=jnp.float64), jnp.cos(1.0 + 0.7 * index)], axis=1).astype(Y.dtype)
    residual = jnp.max(jnp.abs(A @ (Y @ probes) - probes))
    return col_scale[:, jnp.newaxis] * Y * row_scale[jnp.newaxis, :], residual


# ---------------------------------------------------------------------------
# ln|Det| of the geminal (custom_vjp): _ln_det_fwd / _ln_det_bwd.
# ---------------------------------------------------------------------------


@jax.custom_vjp
@jit
def compute_ln_det_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute :math:`\ln|\det G|` for the geminal matrix.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        float: Scalar log-determinant of the geminal matrix.
    """
    G = compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    G = cast_to_zone(G, "det_eval", geminal_data._determinant_dtype_jnp)
    return jnp.linalg.slogdet(G)[1]


# Forward pass for custom VJP.
def _ln_det_fwd(
    geminal_data: _GeminalLike,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, tuple[_GeminalLike, jax.Array, jax.Array, jax.Array]]:
    """Forward pass for custom VJP.

    The custom derivative is needed for ln |Det(G)| because the jax native grad
    and hessian introduce numerical instability. The custom derivative uses SVD
    on an equilibrated G to compute its inverse, avoiding NaN near singularity
    (small singular values are zeroed rather than producing 1/~0 from LU).

    Returns:
        - primal output: ln|det(G)|
        - residuals: (inputs and equilibrated inverse) for the backward pass
    """
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    G = cast_to_zone(G, "det_eval", geminal_data._determinant_dtype_jnp)
    ln_det = jnp.linalg.slogdet(G)[1]
    inverse = _equilibrated_geminal_pinv(G, get_eps("rcond_svd", get_dtype_jnp("det_eval")))
    return ln_det, (geminal_data, r_up_carts, r_dn_carts, inverse)


# Backward pass for custom VJP.
def _ln_det_bwd(
    res: tuple[_GeminalLike, jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[_GeminalLike, jax.Array, jax.Array]:
    r"""Backward pass for custom VJP of :func:`compute_ln_det_geminal_all_elements`.

    Computes ``d ln|det G| / dG = (G^{-1})^T`` using a thresholded SVD
    pseudoinverse, avoiding NaN that would arise from LU with near-zero pivots.
    When the autodiff path takes the Hessian of ``ln|det G|``, JAX
    differentiates *through* this backward pass, so the second derivative
    implicitly accounts for the pseudoinverse structure, including projection
    terms when singular values are zeroed. The analytic Laplacian in
    :func:`compute_grads_and_laplacian_ln_Det` assumes the simpler
    ``d(G^{-1}) = -G^{-1} dG G^{-1}`` and becomes approximate in that case.

    Args:
        res: residuals from forward pass
        g: cotangent of the primal output

    Returns:
        Gradients with respect to (geminal_data, r_up_carts, r_dn_carts)
    """
    geminal_data, r_up_carts, r_dn_carts, X = res

    # d ln|det G| / dG = (G^{-1})^T, scaled by incoming cotangent g
    grad_G = g * X.T

    # Now backpropagate through compute_geminal_all_elements_jax
    _, vjp_fun = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    # Apply VJP to produce gradients for each input
    return vjp_fun(grad_G)


# Register the custom VJP rule !!
compute_ln_det_geminal_all_elements.defvjp(_ln_det_fwd, _ln_det_bwd)


# ---------------------------------------------------------------------------
# Fast ln|Det| (custom_vjp) reusing a precomputed geminal_inv:
# _ln_det_fast_fwd / _ln_det_fast_bwd.
# ---------------------------------------------------------------------------


@jax.custom_vjp
def compute_ln_det_geminal_all_elements_fast(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
) -> jax.Array:
    r"""Compute :math:`\ln|\det G|` using pre-computed ``geminal_inv`` in gradients.

    Mirrors :func:`compute_ln_det_geminal_all_elements` in the forward direction.
    The **backward pass** replaces the implicit ``G^{-1}`` computation that JAX
    would normally perform (via a fresh LU decomposition) with the pre-computed
    ``geminal_inv`` -- the Sherman-Morrison running inverse.  This avoids
    catastrophic NaN for near-singular geminal matrices sampled when
    ``epsilon_AS > 0``.

    Args:
        geminal_data: Geminal parameters (lambda matrix etc.).
        r_up_carts: Cartesian coordinates of up-spin electrons ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of down-spin electrons ``(N_dn, 3)``.
        geminal_inv: Pre-computed inverse geminal matrix ``(N_up, N_up)``.

    Returns:
        Scalar :math:`\ln|\det G|`.

    Warning:
        ``geminal_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}`` exactly
        at the supplied electron positions.  This is only guaranteed when the
        inverse is maintained via **single-electron (rank-1) Sherman-Morrison
        updates** starting from a freshly initialized LU inverse -- the pattern
        used in the MCMC loop.  Passing an inverse that corresponds to different
        electron positions silently produces incorrect gradients.
    """
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    G = cast_to_zone(G, "det_eval", geminal_data._determinant_dtype_jnp)
    return jnp.linalg.slogdet(G)[1]


def _ln_det_fast_fwd(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inv: jax.Array,
) -> tuple[jax.Array, tuple[_GeminalLike, jax.Array, jax.Array, jax.Array]]:
    G = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    G = cast_to_zone(G, "det_eval", geminal_data._determinant_dtype_jnp)
    val = jnp.linalg.slogdet(G)[1]
    # Save inputs for backward (geminal_inv replaces G^{-1} in bwd)
    return val, (geminal_data, r_up_carts, r_dn_carts, geminal_inv)


def _ln_det_fast_bwd(
    res: tuple[_GeminalLike, jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[_GeminalLike, jax.Array, jax.Array, jax.Array]:
    geminal_data, r_up_carts, r_dn_carts, geminal_inv = res
    geminal_inv = cast_to_zone(geminal_inv, "det_eval", geminal_data._determinant_dtype_jnp)
    # d(ln|det G|)/d(G_{ij}) = (G^{-T})_{ij}
    # Use the pre-computed inverse instead of re-solving.
    G_bar = g * geminal_inv.T  # cotangent w.r.t. G, shape (N_up, N_up)
    # Propagate cotangent back through G = compute_geminal_all_elements(...)
    _, vjp_fn = jax.vjp(compute_geminal_all_elements, geminal_data, r_up_carts, r_dn_carts)
    grad_geminal_data, grad_r_up, grad_r_dn = vjp_fn(G_bar)
    # geminal_inv is a non-differentiable constant; return zeros.
    return grad_geminal_data, grad_r_up, grad_r_dn, jnp.zeros_like(geminal_inv)


# Register the custom VJP rule !!
compute_ln_det_geminal_all_elements_fast.defvjp(_ln_det_fast_fwd, _ln_det_fast_bwd)


# ---------------------------------------------------------------------------
# det (non-log) of the geminal + _compute_det_geminal_all_elements_debug oracle.
# ---------------------------------------------------------------------------


@jit
def compute_det_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Compute :math:`\det G` for the geminal matrix.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        float: Scalar determinant of the geminal matrix.
    """
    G = compute_geminal_all_elements(geminal_data=geminal_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
    # Dtype promotion -- a complex geminal must keep ``Im`` for
    # ``jax.linalg.det`` to return the genuine complex determinant.
    G = cast_to_zone(G, "det_eval")
    return jnp.linalg.det(G)


def _compute_det_geminal_all_elements_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> np.float64 | np.complex128:
    """Test-only NumPy determinant oracle using a directly rebuilt geminal.

    Preserve the complex phase and the configured determinant precision. This
    slow reference supports numerical tests; it is not a production fallback.
    """
    G = _compute_geminal_all_elements_debug(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    # ``det_eval`` is real by default, but a non-Gamma geminal carries a
    # physical complex phase.  Promote against the computed matrix just as
    # the production JAX kernel does instead of silently discarding ``Im`` in
    # this independent NumPy oracle.
    dtype_np = np.result_type(get_dtype_np("det_eval"), np.asarray(G).dtype)
    G = np.asarray(G, dtype=dtype_np)
    return np.linalg.det(G)


# ---------------------------------------------------------------------------
# Attaccalite-Sorella (AS) antisymmetrization regularization: fast-update
# entry, _compute_AS_regularization_factor_debug oracle, and the direct kernel.
# ---------------------------------------------------------------------------


def compute_AS_regularization_factor_fast_update(
    geminal: npt.NDArray[np.float64], geminal_inv: npt.NDArray[np.float64]
) -> jax.Array:
    """Compute Attaccalite-Sorella regularization via fast update.

    Args:
        geminal: Geminal matrix with shape ``(N_up, N_up)``.
        geminal_inv: Inverse geminal matrix with shape ``(N_up, N_up)``.

    Returns:
        jax.Array: Scalar AS regularization factor.
    """
    # The strictly-real ``det_ratio`` zone would strip ``Im`` before the
    # ``|x|^2`` reductions below; promote against
    # the input dtype so ``geminal/geminal_inv`` keep ``complex128`` at non-Gamma.
    geminal = cast_to_zone(geminal, "det_ratio", geminal_inv.dtype)
    geminal_inv = cast_to_zone(geminal_inv, "det_ratio", geminal.dtype)
    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    F = jnp.sum(jnp.abs(geminal_inv) ** 2)

    # Compute the scale factor from the full square determinant matrix G.
    # Rows and columns are matrix directions, not two spin-resolved definitions
    # of S: in an open-shell geminal the last N_up-N_dn columns are unpaired
    # orbitals rather than down-electron columns, but they remain genuine columns
    # of det(G) and must therefore participate in the minimum.
    S = jnp.minimum(
        jnp.min(jnp.sum(jnp.abs(geminal) ** 2, axis=1)),  # minimum full-row norm
        jnp.min(jnp.sum(jnp.abs(geminal) ** 2, axis=0)),  # minimum full-column norm
    )

    # compute R_AS
    # Issue #131: a far electron makes one row of G tiny; its squares underflow
    # (S = 0) while F overflows, but R_AS is invariant under that row's scale.
    # The linear product is kept (bit-identical) only where it is exact to
    # rounding; otherwise R_AS comes from the max-abs-scaled ln S + ln F. Only a
    # genuinely degenerate state (an exactly zero row/column, or a non-finite
    # G / G^-1) gives R_AS = 0, so the walker is fully down-weighted.
    SF = S * F
    return _as_factor_linear_or_log(geminal, geminal_inv, S, F, SF, theta)


def _ln_sum_abs_sq_scaled(x: jax.Array, axis: int | None) -> jax.Array:
    """``ln sum |x|^2`` over ``axis`` evaluated with max-abs scaling (no underflow / overflow).

    An all-zero slice gives ``-inf``; a slice holding ``inf`` or ``NaN`` gives ``NaN``.
    """
    a = jnp.abs(x)
    m = jnp.max(a, axis=axis, keepdims=True)
    scale = jnp.where(m > 0.0, m, 1.0)
    s = jnp.sum((a / scale) ** 2, axis=axis)
    m = jnp.squeeze(m, axis=axis)
    scale = jnp.squeeze(scale, axis=axis)
    return jnp.where(m > 0.0, 2.0 * jnp.log(scale) + jnp.log(s), jnp.where(m == 0.0, -jnp.inf, jnp.nan))


def _as_factor_linear_or_log(
    geminal: jax.Array, geminal_inv: jax.Array, S: jax.Array, F: jax.Array, SF: jax.Array, theta: float
) -> jax.Array:
    """Return ``(S F)^(-theta)``: linear where exact to rounding, else from scaled logarithms (issue #131).

    ``S``, ``F`` and ``SF`` are the caller's linear-scale values; the linear
    branch is bit-identical to the plain ``SF ** (-theta)``. Outside
    :func:`jqmc._setting.as_linear_min_sq_norm` (or with a non-finite ``F`` /
    ``SF``) the factor is ``exp(-theta (ln S + ln F))`` with ``ln S`` the minimum
    of the max-abs-scaled row / column logarithms of ``geminal`` and ``ln F`` that
    of ``geminal_inv``; 0 only if one of them is not finite.
    """
    real_dtype = jnp.real(SF).dtype
    linear_ok = (S >= as_linear_min_sq_norm(real_dtype)) & jnp.isfinite(F) & jnp.isfinite(SF) & (SF > 0.0)
    R_linear = jnp.where(linear_ok, SF, 1.0) ** (-theta)
    ln_S = jnp.minimum(
        jnp.min(_ln_sum_abs_sq_scaled(geminal, axis=1)),
        jnp.min(_ln_sum_abs_sq_scaled(geminal, axis=0)),
    )
    ln_F = _ln_sum_abs_sq_scaled(geminal_inv, axis=None)
    log_ok = jnp.isfinite(ln_S) & jnp.isfinite(ln_F)
    R_log = jnp.exp(-theta * jnp.where(log_ok, ln_S + ln_F, 0.0))
    return jnp.where(linear_ok, R_linear, jnp.where(log_ok, R_log, 0.0))


def _compute_AS_regularization_factor_debug(
    geminal_data: Geminal_data, r_up_carts: npt.NDArray[np.float64], r_dn_carts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """See compute_AS_regularization_factor_jax."""
    geminal = _compute_geminal_all_elements_debug(geminal_data, r_up_carts, r_dn_carts)
    # Promote against the input dtype so
    # ``complex128`` survives at non-Gamma (the ``det_eval`` zone is strictly
    # real); bit-identical at Gamma where ``geminal`` is already real.
    dtype_np = np.result_type(get_dtype_np("det_eval"), np.asarray(geminal).dtype)
    geminal = np.asarray(geminal, dtype=dtype_np)

    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    geminal_inv = np.linalg.inv(geminal)
    F = np.sum(np.abs(geminal_inv) ** 2)

    # Compute the scale factor over every row and every column of the full
    # square determinant matrix.  Unpaired-orbital columns are included.
    S = min(
        np.min(np.sum(np.abs(geminal) ** 2, axis=1)),  # minimum full-row norm
        np.min(np.sum(np.abs(geminal) ** 2, axis=0)),  # minimum full-column norm
    )

    # compute R_AS
    R_AS = (S * F) ** (-theta)

    return R_AS


@jit
def compute_AS_regularization_factor(geminal_data: Geminal_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
    """Compute Attaccalite-Sorella regularization from electron coordinates.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Scalar AS regularization factor.
    """
    geminal = compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)
    # Promote against the input dtype so
    # ``complex128`` survives at non-Gamma.
    geminal = cast_to_zone(geminal, "det_eval")

    # compute the AS factor
    theta = 3.0 / 8.0

    # compute F \equiv the square of Frobenius norm of geminal_inv
    # Use SVD with conservative threshold to avoid Inf from 1/sigma^2 for tiny sigma
    # ``svd`` of a complex matrix returns real ``sigma`` -> ``get_eps`` lookup
    # below must use a REAL dtype (the function only handles fp32/fp64 zones).
    # ``sigma`` and ``sigma_sq_inv`` (and hence ``F``) are real even when
    # ``geminal`` is complex; keep them in the real ``det_eval`` zone so the
    # downstream ``S * F`` ordering test (``> 0``) stays well-defined.
    _real_dtype = get_dtype_jnp("det_eval")
    eps_rcond = get_eps("rcond_svd", _real_dtype)
    sigma = jnp.linalg.svd(geminal, compute_uv=False)
    sigma_sq_inv = jnp.where(
        sigma > eps_rcond * sigma[0],
        jnp.asarray(1.0, dtype=_real_dtype) / (sigma**2),
        jnp.asarray(0.0, dtype=_real_dtype),
    )
    F = jnp.sum(sigma_sq_inv)

    # Compute the scale factor from the full square determinant matrix G.
    # This is a row/column matrix norm, not a spin-channel split.  In particular,
    # the unpaired-orbital columns of an open-shell G are included.
    S = jnp.minimum(
        jnp.min(jnp.sum(jnp.abs(geminal) ** 2, axis=1)),  # minimum full-row norm
        jnp.min(jnp.sum(jnp.abs(geminal) ** 2, axis=0)),  # minimum full-column norm
    )

    # compute R_AS
    # Issue #131: S*F can be 0*inf = NaN for a tiny row (S underflows, F
    # overflows) although R_AS is invariant under that row's scale; outside the
    # linear range the factor comes from the scaled ln S + ln F, with ln F taken
    # from the kept 1/sigma. Only a degenerate state gives 0.
    SF = S * F
    inv_sigma = jnp.where(
        sigma > eps_rcond * sigma[0],
        jnp.asarray(1.0, dtype=_real_dtype) / sigma,
        jnp.asarray(0.0, dtype=_real_dtype),
    )
    return _as_factor_linear_or_log(geminal, inv_sigma, S, F, SF, theta)


# ---------------------------------------------------------------------------
# Geminal matrix-element kernels: full G(r_up, r_dn), the jitted core, and the
# _compute_geminal_all_elements_debug oracle.
# ---------------------------------------------------------------------------


def _assemble_geminal_from_orbitals(
    geminal_data: _GeminalLike,
    orb_matrix_up: jax.Array,
    orb_matrix_dn: jax.Array,
) -> jax.Array:
    """Assemble ``G`` without materialising an SD identity selector.

    ``SD_data`` is contracted directly from its occupied rows.  AGP/AGPn keep
    their established lambda contraction.  The subtype is static in JAX's
    pytree signature, so this Python dispatch does not introduce a runtime
    branch in compiled kernels.
    """
    if isinstance(geminal_data, SD_data):
        n_dn = int(geminal_data.num_occ_dn)
        paired = orb_matrix_up[:n_dn].T @ orb_matrix_dn
        unpaired = orb_matrix_up[n_dn : geminal_data.num_occ_up].T
        return jnp.hstack([paired, unpaired])

    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_data._lambda_matrix_jnp, [geminal_data.orb_num_dn])
    geminal_paired = orb_matrix_up.T @ (lambda_matrix_paired @ orb_matrix_dn)
    geminal_unpaired = orb_matrix_up.T @ lambda_matrix_unpaired
    return jnp.hstack([geminal_paired, geminal_unpaired])


def _lambda_blocks_jnp(geminal_data: AGP_data | AGPn_data) -> list[jax.Array]:
    """Return AGP/AGPn lambda blocks; SD deliberately has no such operator."""
    if isinstance(geminal_data, SD_data):
        raise TypeError("SD_data has no lambda blocks; use occupied-orbital contraction helpers.")
    return jnp.hsplit(geminal_data._lambda_matrix_jnp, [geminal_data.orb_num_dn])


def _contract_paired_dn(geminal_data: _GeminalLike, dn_values: jax.Array) -> jax.Array:
    """Apply the paired map to a dn-orbital value/derivative table."""
    if isinstance(geminal_data, SD_data):
        return dn_values
    paired, _ = _lambda_blocks_jnp(geminal_data)
    return paired @ dn_values


def _contract_up_paired(geminal_data: _GeminalLike, up_values: jax.Array) -> jax.Array:
    """Apply the paired map from the up side; output shape ``(..., N, M_dn)``."""
    if isinstance(geminal_data, SD_data):
        return jnp.swapaxes(up_values[..., : geminal_data.num_occ_dn, :], -2, -1)
    paired, _ = _lambda_blocks_jnp(geminal_data)
    return jnp.swapaxes(up_values, -2, -1) @ paired


def _contract_up_unpaired(geminal_data: _GeminalLike, up_values: jax.Array) -> jax.Array:
    """Build the excess-up columns directly for SD or through AGP lambda."""
    if isinstance(geminal_data, SD_data):
        return jnp.swapaxes(
            up_values[..., geminal_data.num_occ_dn : geminal_data.num_occ_up, :],
            -2,
            -1,
        )
    _, unpaired = _lambda_blocks_jnp(geminal_data)
    return jnp.swapaxes(up_values, -2, -1) @ unpaired


def _contract_up_with_paired_dn(
    geminal_data: _GeminalLike,
    up_values: jax.Array,
    paired_dn: jax.Array,
) -> jax.Array:
    """Contract an up table against :func:`_contract_paired_dn` output."""
    if isinstance(geminal_data, SD_data):
        up_values = up_values[..., : geminal_data.num_occ_dn, :]
    return jnp.swapaxes(up_values, -2, -1) @ paired_dn


def _geminal_spatial_derivative_tables(
    geminal_data: _GeminalLike,
    orb_up: jax.Array,
    orb_up_grads: jax.Array,
    orb_up_lap: jax.Array,
    orb_dn: jax.Array,
    orb_dn_grads: jax.Array,
    orb_dn_lap: jax.Array,
    dtype: jnp.dtype,
) -> dict[str, jax.Array]:
    """Build first/second spatial derivative tables for AGP or direct SD."""
    paired_dn = _contract_paired_dn(geminal_data, orb_dn)
    if isinstance(geminal_data, SD_data):
        n_dn = geminal_data.num_occ_dn
        n_up = geminal_data.num_occ_up
        paired_dn_grads = orb_dn_grads
        paired_dn_lap = orb_dn_lap
        paired_up = orb_up[:n_dn].T
        grad_up_paired = jnp.einsum("gia,aj->gij", jnp.swapaxes(orb_up_grads[:, :n_dn, :], 1, 2), paired_dn)
        grad_up_unpaired = jnp.swapaxes(orb_up_grads[:, n_dn:n_up, :], 1, 2)
        grad_dn_paired = jnp.einsum("ia,gaj->gij", paired_up, paired_dn_grads)
        lap_up_paired = orb_up_lap[:n_dn].T @ paired_dn
        lap_up_unpaired = orb_up_lap[n_dn:n_up].T
        lap_dn_paired = paired_up @ paired_dn_lap
    else:
        paired, unpaired = _lambda_blocks_jnp(geminal_data)
        paired = jnp.asarray(paired, dtype=dtype)
        unpaired = jnp.asarray(unpaired, dtype=dtype)
        paired_dn_grads = jnp.einsum("ab,gbn->gan", paired, orb_dn_grads)
        paired_dn_lap = paired @ orb_dn_lap
        paired_up = orb_up.T @ paired
        grad_up_paired = jnp.einsum("gia,aj->gij", jnp.swapaxes(orb_up_grads, 1, 2), paired_dn)
        grad_up_unpaired = jnp.einsum("gia,ak->gik", jnp.swapaxes(orb_up_grads, 1, 2), unpaired)
        grad_dn_paired = jnp.einsum("ia,gaj->gij", orb_up.T, paired_dn_grads)
        lap_up_paired = orb_up_lap.T @ paired_dn
        lap_up_unpaired = orb_up_lap.T @ unpaired
        lap_dn_paired = orb_up.T @ paired_dn_lap

    n_unpaired = geminal_data.num_electron_up - geminal_data.num_electron_dn
    grad_dn_unpaired = jnp.zeros((3, geminal_data.num_electron_up, n_unpaired), dtype=dtype)
    lap_dn_unpaired = jnp.zeros((geminal_data.num_electron_up, n_unpaired), dtype=dtype)
    return {
        "paired_dn": paired_dn,
        "paired_dn_grads": paired_dn_grads,
        "paired_dn_lap": paired_dn_lap,
        "paired_up": paired_up,
        "grad_up": jnp.concatenate([grad_up_paired, grad_up_unpaired], axis=2),
        "grad_dn": jnp.concatenate([grad_dn_paired, grad_dn_unpaired], axis=2),
        "lap_up": jnp.hstack([lap_up_paired, lap_up_unpaired]),
        "lap_dn": jnp.hstack([lap_dn_paired, lap_dn_unpaired]),
    }


def _assemble_geminal_from_orbitals_numpy(
    geminal_data: _GeminalLike,
    orb_matrix_up: npt.NDArray,
    orb_matrix_dn: npt.NDArray,
) -> npt.NDArray:
    """NumPy debug sibling of :func:`_assemble_geminal_from_orbitals`."""
    if isinstance(geminal_data, SD_data):
        n_dn = int(geminal_data.num_occ_dn)
        paired = orb_matrix_up[:n_dn].T @ orb_matrix_dn
        unpaired = orb_matrix_up[n_dn : geminal_data.num_occ_up].T
        return np.hstack([paired, unpaired])

    lambda_matrix_paired, lambda_matrix_unpaired = np.hsplit(np.asarray(geminal_data.lambda_matrix), [geminal_data.orb_num_dn])
    return np.hstack(
        [
            orb_matrix_up.T @ lambda_matrix_paired @ orb_matrix_dn,
            orb_matrix_up.T @ lambda_matrix_unpaired,
        ]
    )


def compute_geminal_all_elements(geminal_data: Geminal_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
    r"""Compute geminal matrix :math:`G` for all electron pairs.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Geminal matrix with shape ``(N_up, N_up)`` combining paired and unpaired blocks.
    """
    # NOTE: do not pre-cast r_*_carts here. r_*_carts is only forwarded to
    # ``_compute_geminal_all_elements`` (which in turn calls ``compute_orb_api``
    # -> ``compute_AOs``); the AO kernels reconstruct ``r - R`` in float64
    # internally to avoid catastrophic cancellation, and a wrapper-level
    # downcast would defeat that guard. Arithmetic in this function uses
    # ``ao_matrix_*`` / ``lambda_matrix_*`` which are cast at their own use
    # sites, so r_*_carts itself does not need casting here.
    if len(r_up_carts) != geminal_data.num_electron_up or len(r_dn_carts) != geminal_data.num_electron_dn:
        logger.info(
            f"Number of up and dn electrons (N_up, N_dn) = ({len(r_up_carts)}, {len(r_dn_carts)}) are not consistent "
            + f"with the expected values. (N_up, N_dn) = {geminal_data.num_electron_up}, {geminal_data.num_electron_dn})"
        )
        raise ValueError

    if len(r_up_carts) != len(r_dn_carts):
        if len(r_up_carts) - len(r_dn_carts) < 0:
            logger.error(
                f"Number of up electron is smaller than dn electrons. (N_up - N_dn = {len(r_up_carts) - len(r_dn_carts)})"
            )
            raise ValueError

    geminal = _compute_geminal_all_elements(geminal_data, r_up_carts, r_dn_carts)

    if geminal.shape != (len(r_up_carts), len(r_up_carts)):
        logger.error(
            f"geminal.shape = {geminal.shape} is inconsistent with the expected one = {(len(r_up_carts), len(r_up_carts))}"
        )
        raise ValueError

    return geminal


@jit
def _compute_geminal_all_elements(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """See compute_geminal_all_elements_api."""
    # Promote against the lambda dtype so the
    # ``complex128`` storage propagates through the matmul; the
    # strictly-real ``det_eval`` zone would silently drop ``Im`` and break
    # ``jax.grad`` on a complex paired block.
    determinant_dtype = geminal_data._determinant_dtype_jnp

    # orb_matrix_* may be produced in the ao_eval / mo_eval zone (potentially
    # float32). Explicitly upcast to the det_eval zone here so the matmul does
    # not rely on JAX implicit type promotion (fp32 x fp64 -> fp64).
    orb_matrix_up = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts),
        "det_eval",
        determinant_dtype,
    )
    orb_matrix_dn = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts),
        "det_eval",
        determinant_dtype,
    )
    return _assemble_geminal_from_orbitals(geminal_data, orb_matrix_up, orb_matrix_dn)


def _compute_geminal_all_elements_debug(
    geminal_data: Geminal_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """See compute_geminal_all_elements_api."""
    dtype_np = get_dtype_np("det_eval")
    r_up_carts = np.asarray(r_up_carts, dtype=dtype_np)
    r_dn_carts = np.asarray(r_dn_carts, dtype=dtype_np)
    up_orbitals = geminal_data.orb_data_up_spin
    down_orbitals = geminal_data.orb_data_dn_spin
    up_debug = (
        _compute_tiled_orbitals_debug
        if isinstance(up_orbitals, TiledOrbitalSet_data)
        else (_compute_MOs_debug if isinstance(up_orbitals, MOs_data) else _compute_AOs_debug)
    )
    down_debug = (
        _compute_tiled_orbitals_debug
        if isinstance(down_orbitals, TiledOrbitalSet_data)
        else (_compute_MOs_debug if isinstance(down_orbitals, MOs_data) else _compute_AOs_debug)
    )
    orb_matrix_up_raw = up_debug(up_orbitals, r_up_carts)
    orb_matrix_dn_raw = down_debug(down_orbitals, r_dn_carts)
    # Dtype promotion -- pick a working dtype that carries ``Im``
    # if any orb / lambda block is complex. ``np.result_type`` of two real
    # dtypes returns the real one, so the real Gamma path stays byte-identical.
    if isinstance(geminal_data, SD_data):
        _parameter_dtype = geminal_data._determinant_dtype_jnp
    else:
        _parameter_dtype = np.asarray(geminal_data.lambda_matrix).dtype
    _work_dtype = np.result_type(
        dtype_np, _parameter_dtype, np.asarray(orb_matrix_up_raw).dtype, np.asarray(orb_matrix_dn_raw).dtype
    )
    orb_matrix_up = np.asarray(orb_matrix_up_raw, dtype=_work_dtype)
    orb_matrix_dn = np.asarray(orb_matrix_dn_raw, dtype=_work_dtype)
    return _assemble_geminal_from_orbitals_numpy(geminal_data, orb_matrix_up, orb_matrix_dn)


# ---------------------------------------------------------------------------
# Single-row / single-column geminal elements: the fast-update primitives for
# one moved electron (one up row / one dn column).
# ---------------------------------------------------------------------------


@jax.jit
def compute_geminal_up_one_row_elements(
    geminal_data: AGP_data | AGPn_data | SD_data,
    r_up_cart: jax.Array,  # shape: (1, 3)
    r_dn_carts: jax.Array,  # shape: (N_dn, 3)
) -> jax.Array:
    """Single row of the geminal matrix for one spin-up electron.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_cart: Cartesian coordinate for one spin-up electron with shape ``(1, 3)``.
            ``compute_orb_api`` requires a 2D ``(N, 3)`` batch; pass a single
            electron as ``(1, 3)``, not ``(3,)``.
        r_dn_carts: Cartesian coordinates for all spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        jax.Array: Row vector with shape ``(N_dn + N_unpaired,)``.
    """
    # r_*_cart(s) are only forwarded to ``compute_orb_api``; do not pre-cast
    # here (would defeat the AO kernels' fp64 ``r - R`` reconstruction).
    # Orbital values:
    # - up: single position -> 1D vector (n_orb_up,)
    # - dn: batched positions -> (n_orb_dn, N_dn)
    # Explicitly upcast to the det_ratio zone (compute_orb_api may return
    # ao_eval / mo_eval dtype, e.g. fp32 for AGP) to avoid relying on JAX
    # implicit type promotion in the lambda matmul below.
    orb_up_vec = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_cart),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    orb_up_vec = jnp.reshape(orb_up_vec, (-1,))  # ensure (n_orb_up,)
    orb_matrix_dn = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    # ensure (n_orb_dn, N_dn)

    if isinstance(geminal_data, SD_data):
        row_paired = orb_up_vec[: geminal_data.num_occ_dn] @ orb_matrix_dn
        row_unpaired = orb_up_vec[geminal_data.num_occ_dn : geminal_data.num_occ_up]
    else:
        lambda_matrix_paired, lambda_matrix_unpaired = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_paired = cast_to_zone(lambda_matrix_paired, "det_ratio", geminal_data._determinant_dtype_jnp)
        lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_ratio", geminal_data._determinant_dtype_jnp)
        row_paired = orb_up_vec @ (lambda_matrix_paired @ orb_matrix_dn)
        row_unpaired = orb_up_vec @ lambda_matrix_unpaired

    # Concatenate horizontally to match the full geminal row
    row = jnp.hstack([row_paired, row_unpaired])  # (N_dn + num_unpaired,)
    return row


@jax.jit
def compute_geminal_dn_one_column_elements(
    geminal_data: AGP_data | AGPn_data | SD_data,
    r_up_carts: jax.Array,  # shape: (N_up, 3)
    r_dn_cart: jax.Array,  # shape: (1, 3)
) -> jax.Array:
    """Single column of the geminal matrix for one spin-down electron.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_cart: Cartesian coordinate for one spin-down electron with shape ``(1, 3)``.
            ``compute_orb_api`` requires a 2D ``(N, 3)`` batch; pass a single
            electron as ``(1, 3)``, not ``(3,)``.

    Returns:
        jax.Array: Column vector for the paired block with shape ``(N_up,)``.
    """
    # r_*_cart(s) are only forwarded to ``compute_orb_api``; do not pre-cast
    # here (would defeat the AO kernels' fp64 ``r - R`` reconstruction).
    # Orbital values:
    # - up: batched positions -> (n_orb_up, N_up)
    # - dn: single position -> 1D vector (n_orb_dn,)
    # Explicitly upcast to the det_ratio zone (compute_orb_api may return
    # ao_eval / mo_eval dtype, e.g. fp32 for AGP) to avoid relying on JAX
    # implicit type promotion in the lambda matmul below.
    orb_matrix_up = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )  # (n_orb_up, N_up)

    orb_dn_vec = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_cart),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    orb_dn_vec = jnp.reshape(orb_dn_vec, (-1,))  # (n_orb_dn,)

    if isinstance(geminal_data, SD_data):
        col = orb_matrix_up[: geminal_data.num_occ_dn].T @ orb_dn_vec
    else:
        lambda_matrix_paired, _ = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_paired = cast_to_zone(lambda_matrix_paired, "det_ratio", geminal_data._determinant_dtype_jnp)
        col = orb_matrix_up.T @ (lambda_matrix_paired @ orb_dn_vec)

    return col


# ---------------------------------------------------------------------------
# Determinant-ratio kernels for MCMC moves: rank-1 update, the split-spin
# variant (up-move vs dn-move blocks), and the _*_debug oracle.
# ---------------------------------------------------------------------------


@jit
def _compute_ratio_determinant_part_rank1_update(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
) -> jax.Array:
    r"""Determinant ratio :math:`\det G(\mathbf r')/\det G(\mathbf r)` for batched moves.

    Optimized for the non-local ECP mesh where *exactly one* electron moves per grid.
    We identify the moved electron once, batch-evaluate the new AO/MO row or column,
    and apply a rank-1 update formula ``row @ A_old_inv[:, idx]`` or
    ``A_old_inv[idx, :] @ col`` with ``vmap``.

    Production-only fast path: it has no direct ``_debug`` sibling by design.
    Its determinant-ratio output is guarded indirectly via the use-site
    recompute in ``_compute_ratio_determinant_part_debug``.

    Args:
        geminal_data: Geminal parameters and orbital references.
        A_old_inv: Inverse geminal matrix for the reference configuration with shape ``(N_up, N_up)``.
        old_r_up_carts: Original spin-up electron coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Original spin-down electron coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates per grid with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates per grid with shape ``(N_grid, N_dn, 3)``.

    Returns:
        jax.Array: Determinant ratios per grid with shape ``(N_grid,)``.

    Warning:
        Each proposed configuration in ``new_r_up_carts_arr`` / ``new_r_dn_carts_arr``
        must differ from ``old_r_up_carts`` / ``old_r_dn_carts`` in **exactly one
        electron**.  The moved electron is identified via ``jnp.argmax`` on the
        change mask; if two or more electrons differ in the same config, only the
        first changed electron is detected and the ratio is silently incorrect.
        This function is intended exclusively for the non-local ECP integration
        grid generated by the MCMC loop, where exactly one electron is displaced
        per grid point by construction.
    """
    # Forward A_old_inv and old/new r_up/dn_carts as-is (Principle 3a -- no
    # parameter rebind). Module-level forwards (compute_det_geminal_all_elements,
    # compute_orb_api) handle their own use-site casts. Inline arithmetic below
    # casts at the use site (Principle 3b) -- see jnp.dot with A_old_inv.
    # Promotion against the lambda storage dtype
    # propagates ``complex128`` through every ``cast_to_zone`` site below
    # (lambda_paired/unpaired + orb_matrix_*_old + A_old_inv) without
    # touching the cast sites individually. ``.T`` stays algebraic (no
    # conjugation): the Sherman-Morrison identity
    # ``orb_up.T @ lambda @ orb_dn`` must match :func:`_compute_geminal_all_elements`
    # bit-for-bit, which is what makes the kinetic kernel holomorphic by
    # construction.
    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # Degenerate cases (no up or no down) fall back to full determinant evaluation.
    if num_up == 0 or num_dn == 0:
        det_x = compute_det_geminal_all_elements(geminal_data, old_r_up_carts, old_r_dn_carts)
        det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
            geminal_data, new_r_up_carts_arr, new_r_dn_carts_arr
        )
        return det_xp / det_x

    # Which electron moved on each grid? Only one electron moves per grid by construction.
    delta_up = new_r_up_carts_arr - old_r_up_carts
    delta_dn = new_r_dn_carts_arr - old_r_dn_carts
    moved_up_mask = jnp.any(delta_up != 0.0, axis=2)  # (N_grid, N_up)
    moved_dn_mask = jnp.any(delta_dn != 0.0, axis=2)  # (N_grid, N_dn)
    moved_up_exists = jnp.any(moved_up_mask, axis=1)

    # Indices of the moved electron per grid (argmax is fine because only one is non-zero).
    idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)  # (N_grid,)
    idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)  # (N_grid,)

    # Gather the moved coordinates for batch AO/MO evaluation.
    r_up_new = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1)  # (N_grid, 1, 3)
    r_dn_new = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1)  # (N_grid, 1, 3)

    # Flatten grid axis for a single batched AO/MO evaluation (reduces JAX HLO count).
    r_up_new_flat = r_up_new.reshape(-1, 3)  # (N_grid, 3)
    r_dn_new_flat = r_dn_new.reshape(-1, 3)  # (N_grid, 3)

    # Precompute old AO matrices once. Explicitly upcast to the det_ratio zone
    # (compute_orb_api may return ao_eval / mo_eval dtype, e.g. fp32 for AGP) to avoid
    # relying on JAX implicit type promotion in the lambda matmuls below.
    orb_matrix_up_old = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, old_r_up_carts),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    orb_matrix_dn_old = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, old_r_dn_carts),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )

    # Batched AO for moved electrons (up) -> rows.
    #
    # Associativity: the chain is
    #   row_paired = (orb_up_new^T @ lambda_paired) @ orb_dn_old.
    # Naively materialising ``orb_up_new^T @ lambda_paired`` produces a
    # ``(G, n_orb_dn)`` intermediate that is enormous on the ECP / discretized
    # kinetic mesh (G ~= walker * Nv * NN or walker * 6 * n_elec, easily 1-100 M
    # rows for f64 at GH200 scale).  Pre-contracting on the small side instead,
    #   M_paired := lambda_paired @ orb_dn_old   # (n_orb_up, N_dn)
    #   row_paired = orb_up_new^T @ M_paired     # (G, N_dn)
    # turns the big middle gemm into a small ``(n_orb_up, n_orb_dn) x (n_orb_dn, N_dn)``
    # constant-size product (executed once per walker) followed by a single
    # ``(G, n_orb_up) x (n_orb_up, N_dn)`` gemm whose output is the final
    # small ``(G, N_dn)`` row block.  Roughly (n_orb + N) / N FLOPs are saved
    # and the (G, n_orb) intermediate is eliminated, lifting the gemm AI from
    # the DRAM-bound regime to ridge.
    orb_up_new_batch = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_new_flat),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )  # (n_orb_up, G)
    M_paired_up = _contract_paired_dn(geminal_data, orb_matrix_dn_old)
    row_paired = _contract_up_with_paired_dn(geminal_data, orb_up_new_batch, M_paired_up)
    row_unpaired = _contract_up_unpaired(geminal_data, orb_up_new_batch)
    new_rows_up = jnp.hstack([row_paired, row_unpaired])  # (G, N_up)

    # Batched AO for moved electrons (dn) -> columns.
    # Same associativity reorder for the dn block:
    #   cols = orb_up_old^T @ (lambda_paired @ orb_dn_new) -> (orb_up_old^T @ lambda_paired) @ orb_dn_new.
    # ``M_dn`` has shape ``(N_up, n_orb_dn)`` (small) instead of ``(n_orb_up, G)``.
    orb_dn_new_batch = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_new_flat),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )  # (n_orb_dn, G)
    M_paired_dn = _contract_up_paired(geminal_data, orb_matrix_up_old)
    cols = jnp.dot(M_paired_dn, orb_dn_new_batch)  # (N_up, G)
    new_cols_dn = cols.T  # (G, N_up)

    # rank-1 determinant ratios for up-move grids and dn-move grids.
    # Use matrix-matrix contractions first to maximize BLAS/TensorCore utilization,
    # then extract the moved-electron component per grid.
    # Cast A_old_inv to the det_ratio zone at the use site (Principle 3b).
    A_old_inv_z = cast_to_zone(A_old_inv, "det_ratio", geminal_data._determinant_dtype_jnp)
    up_all_cols = jnp.dot(new_rows_up, A_old_inv_z)  # (N_grid, N_up)
    det_ratio_up = jnp.take_along_axis(up_all_cols, idx_up[:, None], axis=1).reshape(-1)

    dn_all_rows = jnp.dot(A_old_inv_z, new_cols_dn.T).T  # (N_grid, N_up)
    det_ratio_dn = jnp.take_along_axis(dn_all_rows, idx_dn[:, None], axis=1).reshape(-1)

    # Select per grid based on which spin moved.
    determinant_ratios = jnp.where(moved_up_exists, det_ratio_up, det_ratio_dn)
    return determinant_ratios


@struct.dataclass
class Det_candidate_mesh_state:
    """Candidate-orbital cache for a fixed one-electron mesh.

    The candidate lattice is stored electron-major, candidate-minor.  Its
    orbital values depend only on the candidate position, so an accepted hop
    invalidates exactly one electron's contiguous candidate block.  The
    current-configuration AO/precontract tables remain owned by
    :class:`Det_ratio_streaming_state`.
    """

    orb_up_candidates: jax.Array
    orb_dn_candidates: jax.Array
    candidates_per_electron: int = struct.field(pytree_node=False)


@struct.dataclass
class Det_winv_state:
    """TurboRVB ``winvup/winvdo``-style determinant weights.

    ``up_weights`` is ``C_up @ A^{-1}`` with shape ``(M_up, N_up)`` and
    ``dn_weights`` is ``A^{-1} @ C_dn`` with shape ``(N_up, M_dn)``.  A
    candidate determinant ratio is therefore one orbital-vector dot product.

    DERIVED QUANTITY, NOT A CHAINED STATE.  The weights are rebuilt from the
    running ``A_old_inv`` and the determinant streaming state at every mesh
    evaluation (two ``O(M*N^2)`` GEMMs -- a factor ``~6N/N = 6`` cheaper than
    the oracle row-build contraction).  Maintaining these tables with their
    own Sherman--Morrison advance chain would compound that chain's rounding
    against the driver's ``A_old_inv`` chain and, amplified by small
    accepted-ratio (``1/rho``) events, diverge from the oracle over
    production-length hop chains (6.5e-3 in fp64, 4.2e-2 in mixed at 150
    adversarial hops), inflating the LRDMC hopping rate and exploding hop
    counts (pinned by ``test_lrdmc_mesh_chain_consistency.py``).  Deriving the weights fresh
    each evaluation keeps the winv ratio kernel's per-candidate speedup while
    making divergence structurally impossible; no advance function exists.
    """

    up_weights: jax.Array
    dn_weights: jax.Array


@jit
def _init_det_winv_state(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    det_ratio_state: "Det_ratio_streaming_state | Det_streaming_state",
) -> Det_winv_state:
    """Derive candidate-ratio weights from the current inverse and orbital state.

    Assemble ``C_up`` so each geminal row is ``orb_up.T @ C_up``, then form
    ``C_up @ A_old_inv`` and ``A_old_inv @ C_dn`` for the two spin sectors.
    Kinetic and ECP mesh consumers rebuild these weights at each evaluation;
    the weights do not carry a separate inverse-update chain.
    """
    dtype = geminal_data._determinant_dtype_jnp
    A_inv = cast_to_zone(A_old_inv, "det_ratio", dtype)
    paired_dn = cast_to_zone(det_ratio_state.paired_dn, "det_ratio", dtype)
    num_up_electrons = det_ratio_state.ao_up.shape[1]
    num_dn_electrons = det_ratio_state.ao_dn.shape[1]
    if isinstance(geminal_data, SD_data):
        # SD evaluates exactly N_up occupied up orbitals.  The first N_dn
        # participate in the paired block; the excess orbitals are the
        # explicit unpaired determinant columns.
        coeff_up = jnp.zeros((det_ratio_state.ao_up.shape[0], num_up_electrons), dtype=paired_dn.dtype)
        coeff_up = coeff_up.at[: geminal_data.num_occ_dn, :num_dn_electrons].set(paired_dn)
        num_unpaired_electrons = num_up_electrons - num_dn_electrons
        if num_unpaired_electrons:
            coeff_up = coeff_up.at[
                geminal_data.num_occ_dn : geminal_data.num_occ_up,
                num_dn_electrons:num_up_electrons,
            ].set(jnp.eye(num_unpaired_electrons, dtype=paired_dn.dtype))
    else:
        _, unpaired = _lambda_blocks_jnp(geminal_data)
        coeff_up = jnp.hstack([paired_dn, cast_to_zone(unpaired, "det_ratio", dtype, paired_dn.dtype)])
    coeff_dn = cast_to_zone(det_ratio_state.paired_up_lambda, "det_ratio", dtype, A_inv.dtype)
    return Det_winv_state(
        up_weights=jnp.dot(coeff_up, A_inv),
        dn_weights=jnp.dot(A_inv, coeff_dn),
    )


@jit
def _init_det_candidate_mesh_state(
    geminal_data: Geminal_data,
    candidate_up_positions: jax.Array,
    candidate_dn_positions: jax.Array,
) -> Det_candidate_mesh_state:
    """Evaluate determinant orbitals once for a fixed candidate lattice."""
    k = candidate_up_positions.shape[1] if candidate_up_positions.shape[0] else candidate_dn_positions.shape[1]
    up_flat = candidate_up_positions.reshape(-1, 3)
    dn_flat = candidate_dn_positions.reshape(-1, 3)
    orb_up = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, up_flat),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    orb_dn = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, dn_flat),
        "det_ratio",
        geminal_data._determinant_dtype_jnp,
    )
    return Det_candidate_mesh_state(
        orb_up_candidates=orb_up,
        orb_dn_candidates=orb_dn,
        candidates_per_electron=k,
    )


@jit
def _advance_det_candidate_mesh_state(
    geminal_data: Geminal_data,
    state: Det_candidate_mesh_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    candidate_up_positions_new: jax.Array,
    candidate_dn_positions_new: jax.Array,
) -> Det_candidate_mesh_state:
    """Refresh only the moved electron's candidate-orbital block."""
    k = state.candidates_per_electron

    def _up(_: None) -> Det_candidate_mesh_state:
        pos = lax.dynamic_slice(candidate_up_positions_new, (moved_index, 0, 0), (1, k, 3)).reshape(k, 3)
        fresh = cast_to_zone(
            geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, pos),
            "det_ratio",
            geminal_data._determinant_dtype_jnp,
        )
        start = moved_index * k
        return state.replace(orb_up_candidates=_complex_update.dynamic_update_slice(state.orb_up_candidates, fresh, (0, start)))

    def _dn(_: None) -> Det_candidate_mesh_state:
        pos = lax.dynamic_slice(candidate_dn_positions_new, (moved_index, 0, 0), (1, k, 3)).reshape(k, 3)
        fresh = cast_to_zone(
            geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, pos),
            "det_ratio",
            geminal_data._determinant_dtype_jnp,
        )
        start = moved_index * k
        return state.replace(orb_dn_candidates=_complex_update.dynamic_update_slice(state.orb_dn_candidates, fresh, (0, start)))

    if candidate_up_positions_new.shape[0] == 0:
        return _dn(None)
    if candidate_dn_positions_new.shape[0] == 0:
        return _up(None)
    # tree_map + jnp.where instead of lax.cond: the batched-predicate cond
    # form miscompiles on XLA CPU (jax 0.7.x) inside the GFMC_t vmapped
    # while_loop (leaf mixing across branches); under vmap both branches
    # execute anyway, so this costs nothing extra. See the matching note in
    # jqmc_gfmc._projection_t_core.
    return jax.tree_util.tree_map(lambda a, b: jnp.where(moved_spin_is_up, a, b), _up(None), _dn(None))


def _compute_ratio_determinant_from_candidate_mesh_state(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    det_ratio_state: "Det_ratio_streaming_state | Det_streaming_state",
    candidate_state: Det_candidate_mesh_state,
) -> jax.Array:
    """Contract cached candidate orbitals with the current determinant state."""
    lambda_dtype = geminal_data._determinant_dtype_jnp
    k = candidate_state.candidates_per_electron
    num_up = det_ratio_state.ao_up.shape[1]
    num_dn = det_ratio_state.ao_dn.shape[1]
    idx_up = jnp.repeat(jnp.arange(num_up, dtype=jnp.int32), k)
    idx_dn = jnp.repeat(jnp.arange(num_dn, dtype=jnp.int32), k)
    M_paired_up = cast_to_zone(det_ratio_state.paired_dn, "det_ratio", lambda_dtype)
    M_paired_dn = cast_to_zone(det_ratio_state.paired_up_lambda, "det_ratio", lambda_dtype)
    orb_up = cast_to_zone(candidate_state.orb_up_candidates, "det_ratio", lambda_dtype, M_paired_up.dtype)
    orb_dn = cast_to_zone(candidate_state.orb_dn_candidates, "det_ratio", lambda_dtype, M_paired_dn.dtype)

    row_paired = _contract_up_with_paired_dn(geminal_data, orb_up, M_paired_up)
    row_unpaired = _contract_up_unpaired(geminal_data, orb_up)
    new_rows_up = jnp.hstack([row_paired, row_unpaired])
    A_old_inv_z = cast_to_zone(A_old_inv, "det_ratio", lambda_dtype, M_paired_up.dtype)
    A_col_for_up = jnp.take(A_old_inv_z, idx_up, axis=1).T
    det_ratio_up = jnp.sum(new_rows_up * A_col_for_up, axis=1)

    new_cols_dn = jnp.dot(M_paired_dn, orb_dn).T
    A_row_for_dn = jnp.take(A_old_inv_z, idx_dn, axis=0)
    det_ratio_dn = jnp.sum(A_row_for_dn * new_cols_dn, axis=1)
    return jnp.concatenate([det_ratio_up, det_ratio_dn])


def _compute_ratio_determinant_from_winv_state(
    geminal_data: Geminal_data,
    candidate_state: Det_candidate_mesh_state,
    winv_state: Det_winv_state,
) -> jax.Array:
    """Contract candidate orbitals with maintained ``winv`` weights.

    Unlike :func:`_compute_ratio_determinant_from_candidate_mesh_state`, this
    production kernel contains no dense inverse/precontract reconstruction.
    The full sibling intentionally remains unchanged as the debug oracle.
    """
    dtype = geminal_data._determinant_dtype_jnp
    k = candidate_state.candidates_per_electron
    num_up = winv_state.up_weights.shape[1]
    # dn_weights contains N_up rows because it is the full A^-1 C_dn table;
    # only the first N_dn rows correspond to movable down electrons.
    num_dn = candidate_state.orb_dn_candidates.shape[1] // k
    orb_up = cast_to_zone(
        candidate_state.orb_up_candidates,
        "det_ratio",
        dtype,
        winv_state.up_weights.dtype,
    ).reshape((candidate_state.orb_up_candidates.shape[0], num_up, k))
    orb_dn = cast_to_zone(
        candidate_state.orb_dn_candidates,
        "det_ratio",
        dtype,
        winv_state.dn_weights.dtype,
    ).reshape((candidate_state.orb_dn_candidates.shape[0], num_dn, k))
    up_weights = cast_to_zone(winv_state.up_weights, "det_ratio", dtype, orb_up.dtype)
    dn_weights = cast_to_zone(winv_state.dn_weights[:num_dn], "det_ratio", dtype, orb_dn.dtype)
    ratio_up = jnp.einsum("mik,mi->ik", orb_up, up_weights).reshape(-1)
    ratio_dn = jnp.einsum("im,mik->ik", dn_weights, orb_dn).reshape(-1)
    return jnp.concatenate([ratio_up, ratio_dn])


def _compute_ratio_determinant_part_split_spin(
    geminal_data: Geminal_data,
    A_old_inv: jax.Array,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_shifted: jax.Array,
    new_r_dn_shifted: jax.Array,
    det_ratio_state: "Det_ratio_streaming_state | None" = None,
) -> jax.Array:
    r"""Determinant ratio for a block-structured mesh where up and dn electrons move separately.

    Avoids computing MOs for the unchanged spin block.  Compared with
    ``_compute_ratio_determinant_part_rank1_update`` called on the concatenated
    (G_up + G_dn) array, this evaluates only ``G_up`` up-spin AOs and ``G_dn``
    dn-spin AOs instead of evaluating both spins for all ``G_up + G_dn`` configs.

    Production-only fast path: it has no direct ``_debug`` sibling by design.
    Its determinant-ratio output is guarded indirectly via the use-site
    recompute in ``_compute_ratio_determinant_part_debug``.

    Args:
        geminal_data: Geminal parameters and orbital references.
        A_old_inv: Inverse geminal matrix with shape ``(N_up, N_up)``.
        old_r_up_carts: Reference up-spin coordinates ``(N_up, 3)``.
        old_r_dn_carts: Reference dn-spin coordinates ``(N_dn, 3)``.
        new_r_up_shifted: Up-block proposed coords ``(G_up, N_up, 3)``.  Exactly one
            up electron differs from ``old_r_up_carts`` per config.
        new_r_dn_shifted: Dn-block proposed coords ``(G_dn, N_dn, 3)``.  Exactly one
            dn electron differs from ``old_r_dn_carts`` per config.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or a
            superset like :class:`Det_streaming_state`) that is consistent
            with ``(old_r_up_carts, old_r_dn_carts)``. When provided, this
            function reuses ``state.ao_up``, ``state.ao_dn``,
            ``state.paired_dn``, and ``state.paired_up_lambda`` to skip the
            heavy ``compute_orb_api`` calls for the bulk old configuration
            and the two ``lambda_paired @ ao_*`` precontracts. Pass ``None``
            (default) for the uncached path used by debug / observation code.

    Returns:
        jax.Array: Concatenated determinant ratios ``(G_up + G_dn,)``.

    Warning:
        Each config in ``new_r_up_shifted`` must differ from ``old_r_up_carts``
        in **exactly one up electron**, and each config in ``new_r_dn_shifted``
        must differ from ``old_r_dn_carts`` in **exactly one dn electron**.
        The moved electron is located via ``jnp.argmax`` on the change mask;
        if two or more electrons differ in the same config, only the first is
        detected and the ratio is silently incorrect.  This function is intended
        exclusively for the block-structured non-local ECP grids produced by
        the MCMC loop.
    """
    # Forward A_old_inv and old/new r_up/dn coords as-is (Principle 3a -- no
    # parameter rebind). Module-level forwards (compute_orb_api,
    # _compute_ratio_determinant_part_rank1_update) handle their own use-site
    # casts. A_old_inv is cast at the use site (Principle 3b) below.
    # Promotion via cast_to_zone (which folds
    # each input's actual dtype into the result_type) so the casts below
    # preserve Im on complex Bloch (lambda real for RHF singlet, but AOs /
    # paired tables / A_old_inv are complex at twist k != 0).
    lambda_dtype = geminal_data._determinant_dtype_jnp
    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # Degenerate cases fall back to the general function on empty slices.
    # (Slim state plumb is irrelevant here -- those code paths are not in the
    # production hot loop.)
    if num_up == 0 or num_dn == 0:
        combined_up = jnp.concatenate(
            [new_r_up_shifted, jnp.broadcast_to(old_r_up_carts[None], (new_r_dn_shifted.shape[0], num_up, 3))],
            axis=0,
        )
        combined_dn = jnp.concatenate(
            [jnp.broadcast_to(old_r_dn_carts[None], (new_r_up_shifted.shape[0], num_dn, 3)), new_r_dn_shifted],
            axis=0,
        )
        return _compute_ratio_determinant_part_rank1_update(
            geminal_data, A_old_inv, old_r_up_carts, old_r_dn_carts, combined_up, combined_dn
        )

    if det_ratio_state is None:
        # Original path: evaluate old-side AO matrices and precontracts from scratch.
        # Cast via cast_to_zone so a complex AO survives the cast (a plain
        # ``.astype(dtype_jnp)`` with a real ``dtype_jnp`` silently drops Im
        # on complex Bloch and makes the fast and slow det_ratio paths
        # disagree).
        orb_matrix_dn_old = cast_to_zone(
            geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, old_r_dn_carts),
            "det_ratio",
            lambda_dtype,
        )
        orb_matrix_up_old = cast_to_zone(
            geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, old_r_up_carts),
            "det_ratio",
            lambda_dtype,
        )
        M_paired_up = _contract_paired_dn(geminal_data, orb_matrix_dn_old)
        M_paired_dn = _contract_up_paired(geminal_data, orb_matrix_up_old)
    else:
        # Streaming path: reuse cached AO + paired tables maintained by the
        # walker. Cast at use site via cast_to_zone (Principle 3b).
        # state.paired_dn and state.paired_up_lambda are mathematically equal
        # to lambda_paired @ ao_dn_old and ao_up_old.T @ lambda_paired
        # respectively (refreshed on every accepted single-electron move).
        M_paired_up = cast_to_zone(det_ratio_state.paired_dn, "det_ratio", lambda_dtype)
        M_paired_dn = cast_to_zone(det_ratio_state.paired_up_lambda, "det_ratio", lambda_dtype)

    # --- UP BLOCK: up electron moved, dn unchanged -----------------------------
    delta_up = new_r_up_shifted - old_r_up_carts  # (G_up, N_up, 3)
    moved_up_mask = jnp.any(delta_up != 0.0, axis=2)  # (G_up, N_up)
    idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)  # (G_up,)
    r_up_new_flat = jnp.take_along_axis(new_r_up_shifted, idx_up[:, None, None], axis=1).reshape(-1, 3)

    # Only evaluate up-spin MOs for the moved electron positions.
    #
    # Associativity: pre-contract on the small side
    #   M_paired := lambda_paired @ orb_dn_old   # (n_orb_up, N_dn)
    # so the chain ``(orb_up_new^T @ lambda_paired) @ orb_dn_old`` becomes
    #   row_paired = orb_up_new^T @ M_paired     # (G_up, N_dn)
    # eliminating the (G_up, n_orb_dn) middle intermediate. See the matching
    # comment in ``_compute_ratio_determinant_part_rank1_update`` for the
    # GH200 motivation.
    orb_up_new_batch = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_new_flat),
        "det_ratio",
        lambda_dtype,
        M_paired_up.dtype,
    )  # (n_orb_up, G_up)
    row_paired = _contract_up_with_paired_dn(geminal_data, orb_up_new_batch, M_paired_up)
    row_unpaired = _contract_up_unpaired(geminal_data, orb_up_new_batch)
    new_rows_up = jnp.hstack([row_paired, row_unpaired])  # (G_up, N_up)

    # Cast A_old_inv to the det_ratio zone at the use site (Principle 3b).
    A_old_inv_z = cast_to_zone(A_old_inv, "det_ratio", lambda_dtype, M_paired_up.dtype)
    A_col_for_up = jnp.take(A_old_inv_z, idx_up, axis=1).T  # (G_up, N_up)
    det_ratio_up_block = jnp.sum(new_rows_up * A_col_for_up, axis=1)  # (G_up,)

    # --- DN BLOCK: dn electron moved, up unchanged -----------------------------
    delta_dn = new_r_dn_shifted - old_r_dn_carts  # (G_dn, N_dn, 3)
    moved_dn_mask = jnp.any(delta_dn != 0.0, axis=2)  # (G_dn, N_dn)
    idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)  # (G_dn,)
    r_dn_new_flat = jnp.take_along_axis(new_r_dn_shifted, idx_dn[:, None, None], axis=1).reshape(-1, 3)

    # Only evaluate dn-spin MOs for the moved electron positions.
    # Associativity: pre-contract ``M_dn := orb_up_old^T @ lambda_paired``
    # (N_up, n_orb_dn), then ``cols.T = orb_dn_new^T @ M_dn^T = (M_dn @ orb_dn_new).T``.
    orb_dn_new_batch = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_new_flat),
        "det_ratio",
        lambda_dtype,
        M_paired_dn.dtype,
    )  # (n_orb_dn, G_dn)
    new_cols_dn = jnp.dot(M_paired_dn, orb_dn_new_batch).T  # (G_dn, N_up)

    A_row_for_dn = jnp.take(A_old_inv_z, idx_dn, axis=0)  # (G_dn, N_up)
    det_ratio_dn_block = jnp.sum(A_row_for_dn * new_cols_dn, axis=1)  # (G_dn,)

    return jnp.concatenate([det_ratio_up_block, det_ratio_dn_block])


def _compute_ratio_determinant_part_debug(
    geminal_data: Geminal_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """See _api method."""
    dtype_np = get_dtype_np("det_ratio")
    old_r_up_carts = np.asarray(old_r_up_carts, dtype=dtype_np)
    old_r_dn_carts = np.asarray(old_r_dn_carts, dtype=dtype_np)
    new_r_up_carts_arr = np.asarray(new_r_up_carts_arr, dtype=dtype_np)
    new_r_dn_carts_arr = np.asarray(new_r_dn_carts_arr, dtype=dtype_np)
    return np.array(
        [
            compute_det_geminal_all_elements(geminal_data, new_r_up_carts, new_r_dn_carts)
            / compute_det_geminal_all_elements(geminal_data, old_r_up_carts, old_r_dn_carts)
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr, strict=True)
        ]
    )


# ---------------------------------------------------------------------------
# Grads & Laplacian of ln|Det| (custom_vjp): the kinetic-energy primitive.
# _grads_lap_body is the shared core; _grads_lap_fwd / _grads_lap_bwd are the
# custom_vjp halves.
# ---------------------------------------------------------------------------


def _orbital_value_grad_lap_tables(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> _OrbitalTables:
    """Evaluate fused orbital tables in the determinant derivative zone.

    Reusing the fused value table to assemble ``G`` is essential.  A separate
    call through ``compute_orb_api`` routes AO values through mixed-mode
    ``ao_eval`` (fp32), so its inverse and the fp64 spatial derivatives would
    otherwise describe two different geminal matrices.
    """
    orb_up, grad_up_x, grad_up_y, grad_up_z, lap_up = geminal_data.compute_orb_value_grad_lap_api(
        geminal_data.orb_data_up_spin, r_up_carts
    )
    orb_dn, grad_dn_x, grad_dn_y, grad_dn_z, lap_dn = geminal_data.compute_orb_value_grad_lap_api(
        geminal_data.orb_data_dn_spin, r_dn_carts
    )
    orb_up = cast_to_zone(orb_up, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    orb_dn = cast_to_zone(orb_dn, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    grads_up = jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=0)
    grads_dn = jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=0)
    return orb_up, grads_up, lap_up, orb_dn, grads_dn, lap_dn


def _geminal_for_grads_lap(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Build ``G`` through the fused value/gradient/Laplacian route."""
    orb_up, _grads_up, _lap_up, orb_dn, _grads_dn, _lap_dn = _orbital_value_grad_lap_tables(
        geminal_data, r_up_carts, r_dn_carts
    )
    return _assemble_geminal_from_orbitals(geminal_data, orb_up, orb_dn)


def _grads_lap_from_orbital_tables(
    geminal_data: Geminal_data,
    geminal_inverse: jax.Array,
    orb_up: jax.Array,
    grads_up: jax.Array,
    lap_up: jax.Array,
    orb_dn: jax.Array,
    grads_dn: jax.Array,
    lap_dn: jax.Array,
) -> _GradLapResult:
    """Assemble determinant log-gradients from precomputed fused tables."""
    dtype_jnp = jnp.result_type(get_dtype_jnp("det_grad_lap"), geminal_data._determinant_dtype_jnp)
    tables = _geminal_spatial_derivative_tables(
        geminal_data,
        orb_up,
        grads_up,
        lap_up,
        orb_dn,
        grads_dn,
        lap_dn,
        dtype_jnp,
    )
    geminal_grad_up = tables["grad_up"]
    geminal_grad_dn = tables["grad_dn"]
    geminal_laplacian_up = tables["lap_up"]
    geminal_laplacian_dn = tables["lap_dn"]

    grad_ln_D_up_stack = jnp.einsum("gij,ji->gi", geminal_grad_up, geminal_inverse)
    grad_ln_D_dn_stack = jnp.einsum("ij,gji->gi", geminal_inverse, geminal_grad_dn)
    grad_ln_D_up = grad_ln_D_up_stack.T
    grad_ln_D_dn = grad_ln_D_dn_stack.T

    grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z = grad_ln_D_up_stack
    grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z = grad_ln_D_dn_stack
    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_laplacian_up, geminal_inverse)
    lap_ln_D_dn = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", geminal_inverse, geminal_laplacian_dn)

    n_dn = geminal_data.num_electron_dn
    return grad_ln_D_up, grad_ln_D_dn[:n_dn], lap_ln_D_up, lap_ln_D_dn[:n_dn]


@jax.custom_vjp
def compute_grads_and_laplacian_ln_Det(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    r"""Gradients and Laplacians of :math:`\ln|\det G|` for each electron.

    Computes :math:`\nabla_i \ln|\det G|` and :math:`\nabla_i^2 \ln|\det G|`
    using the analytic formulas:

    .. math::

        \nabla_i \ln|\det G| &= \mathrm{tr}(G^{-1} \nabla_i G) \\
        \nabla_i^2 \ln|\det G| &= \mathrm{tr}(G^{-1} \nabla_i^2 G)
            - |\nabla_i \ln|\det G||^2

    These formulas rely on the matrix identity
    :math:`d(G^{-1}) = -G^{-1}\, dG\, G^{-1}`, which holds **only for the
    true inverse**.  When ``G`` is near-singular, :math:`G^{-1}` is computed
    via a thresholded SVD pseudoinverse :math:`G^{\dagger}`, which coincides
    with :math:`G^{-1}` only when no singular values are zeroed by
    ``EPS_rcond_SVD``.  If the threshold zeroes non-negligible singular
    values, :math:`G^{\dagger} \neq G^{-1}`, and the analytic Laplacian
    becomes inaccurate because the true pseudoinverse derivative contains
    additional projection terms:

    .. math::

        d(G^{\dagger}) = -G^{\dagger}\,dG\,G^{\dagger}
            + G^{\dagger} G^{\dagger T} dG^T (I - G G^{\dagger})
            + (I - G^{\dagger} G)\, dG^T\, G^{\dagger T} G^{\dagger}

    Autodiff of ``compute_ln_det_geminal_all_elements`` differentiates through
    the SVD pseudoinverse in ``_ln_det_bwd`` and includes the projection terms.
    The kinetic autodiff reference differentiates the normalized wavefunction
    directly. The analytic path here uses the true-inverse derivative above.
    Consequently, ``EPS_rcond_SVD`` must be kept very small (e.g. ``1e-20``)
    so that only truly zero singular values are zeroed and
    :math:`G^{\dagger} = G^{-1}` in practice.

    In production MCMC, the ``_fast`` variant receives a Sherman-Morrison
    running true inverse, so this limitation does not apply there.

    The function uses a custom VJP to avoid NaN in the backward pass when G is
    near-singular. The standard JAX SVD backward computes ``1/(s_i^2 - s_j^2)``
    terms that blow up for degenerate singular values. The custom VJP instead uses
    the matrix identity ``d(G^{-1}) = -G^{-1} dG G^{-1}`` with a thresholded SVD
    pseudoinverse, bypassing differentiation through SVD entirely.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            - Gradients for spin-up electrons with shape ``(N_up, 3)``.
            - Gradients for spin-down electrons with shape ``(N_dn, 3)``.
            - Laplacians for spin-up electrons with shape ``(N_up,)``.
            - Laplacians for spin-down electrons with shape ``(N_dn,)``.
    """
    # Promotion against
    # the geminal lambda dtype so a complex Bloch lambda lifts ``dtype_jnp``
    # from the real ``det_grad_lap`` zone to ``complex128``. A plain
    # ``jnp.asarray(G, dtype=dtype_jnp)`` hardcast silently drops ``Im``
    # on the complex path and breaks the kinetic-energy chain end-to-end
    # (VMC without Jastrow on an HF wavefunction at non-Gamma then gives
    # E_VMC != E_HF). Bit-identical at Gamma where lambda is real.
    dtype_jnp = jnp.result_type(
        get_dtype_jnp("det_grad_lap"),
        geminal_data._determinant_dtype_jnp,
    )
    # NOTE: do not pre-cast r_*_carts. They are forwarded to
    # ``compute_orb_value_grad_lap_api``; the AO kernels reconstruct ``r - R``
    # in float64 internally and a wrapper-level downcast would defeat that
    # guard. Arithmetic here uses orbital/lambda tables cast at their use sites.
    # Evaluate the fused orbital tables once. In mixed mode their value table
    # follows ao_grad_lap (fp64), unlike the separate ao_eval path (fp32).
    orbital_tables = _orbital_value_grad_lap_tables(geminal_data, r_up_carts, r_dn_carts)
    orb_up, _grads_up, _lap_up, orb_dn, _grads_dn, _lap_dn = orbital_tables

    # Compute G_inv from the same orbital values used for its derivatives.
    G = _assemble_geminal_from_orbitals(geminal_data, orb_up, orb_dn)
    # Preserve the conservative cutoff after removing harmless row/column scales.
    geminal_inverse = _equilibrated_geminal_pinv(G, get_eps("rcond_svd", dtype_jnp))

    return _grads_lap_from_orbital_tables(geminal_data, geminal_inverse, *orbital_tables)


def _grads_lap_body(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure functional core of grad/laplacian computation.

    Contains the same computation as ``compute_grads_and_laplacian_ln_Det_fast``
    but without the ``@jit`` decorator or ``None``-check, so it can be safely
    passed to ``jax.vjp`` inside the custom VJP backward pass without creating
    a dependency on the public fast function.
    """
    # r_*_carts are only forwarded; do not pre-cast (see the public function
    # ``compute_grads_and_laplacian_ln_Det`` for the rationale).
    geminal_inverse = cast_to_zone(geminal_inverse, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    orbital_tables = _orbital_value_grad_lap_tables(geminal_data, r_up_carts, r_dn_carts)
    return _grads_lap_from_orbital_tables(geminal_data, geminal_inverse, *orbital_tables)


def _grads_lap_fwd(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[_GradLapResult, tuple[_GeminalLike, jax.Array, jax.Array, jax.Array]]:
    """Forward pass: compute stable G_inv and primal outputs."""
    dtype_jnp = jnp.result_type(get_dtype_jnp("det_grad_lap"), geminal_data._determinant_dtype_jnp)
    # r_*_carts are only forwarded; do not pre-cast (see
    # ``compute_grads_and_laplacian_ln_Det`` for the rationale).
    orbital_tables = _orbital_value_grad_lap_tables(geminal_data, r_up_carts, r_dn_carts)
    orb_up, _grads_up, _lap_up, orb_dn, _grads_dn, _lap_dn = orbital_tables
    G = _assemble_geminal_from_orbitals(geminal_data, orb_up, orb_dn)
    G_inv_stable = _equilibrated_geminal_pinv(G, get_eps("rcond_svd", dtype_jnp))
    primals = _grads_lap_from_orbital_tables(geminal_data, G_inv_stable, *orbital_tables)
    return primals, (geminal_data, r_up_carts, r_dn_carts, G_inv_stable)


def _grads_lap_bwd(
    res: tuple[_GeminalLike, jax.Array, jax.Array, jax.Array],
    g: _GradLapResult,
) -> tuple[_GeminalLike, jax.Array, jax.Array]:
    r"""Backward pass using the matrix identity ``d(G^{-1}) = -G^{-1} dG G^{-1}``.

    The gradient has two contributions:

    1. **Direct**: ``d/d(\lambda)`` via ``\lambda -> AO matrices -> G_grad/G_lap -> output``.
       Obtained by differentiating :func:`_grads_lap_body` w.r.t.
       all inputs (including ``G_inv_stable`` as an explicit argument).
    2. **Inverse**: ``d/d(\lambda)`` via ``\lambda -> G -> G^{-1} -> output``.
       Obtained by converting ``G_inv_bar -> G_bar`` via
       ``-G_inv.T @ G_inv_bar @ G_inv.T``, then propagating ``G_bar`` back
       through :func:`_geminal_for_grads_lap`.

    Neither path requires differentiating through ``jnp.linalg.svd``, so
    degenerate singular values cannot produce NaN.

    .. note::

        Step 2 uses ``d(G^{-1}) = -G^{-1} dG G^{-1}``, which is exact only
        when ``G_inv_stable`` is the true inverse.  If ``EPS_rcond_SVD`` is
        large enough to zero out non-negligible singular values, this identity
        becomes approximate -- the full pseudoinverse derivative has additional
        projection terms (see the docstring of
        :func:`compute_grads_and_laplacian_ln_Det` for details).  Keep
        ``EPS_rcond_SVD`` very small (e.g. ``1e-20``) to avoid this.
    """
    geminal_data, r_up_carts, r_dn_carts, G_inv_stable = res
    G_inv_stable = cast_to_zone(G_inv_stable, "det_grad_lap", geminal_data._determinant_dtype_jnp)

    # Step 1: differentiate _grads_lap_body w.r.t. all args.
    # This gives direct gradients (AO path) and G_inv_bar (cotangent for G_inv).
    _, vjp_fn = jax.vjp(
        _grads_lap_body,
        geminal_data,
        r_up_carts,
        r_dn_carts,
        G_inv_stable,
    )
    # Complex Bloch: when the geminal is complex, JAX's
    # auto-VJP of _grads_lap_body casts the complex intermediate
    # cotangents to real for the real-input slots (r_up_carts, r_dn_carts).
    # This cast IS the physical Re(de_L/dR) projection of the
    # Hellmann-Feynman force estimator and is
    # mathematically intentional, not a buggy contamination. The VJP
    # transpose of a real -> complex promotion emits no ComplexWarning
    # under the pinned JAX, so no warning filter is needed here
    # (tests/test_jqmc_kernel/test_grads_lap_bwd_complex_warning.py).
    d_geminal_direct, d_r_up_direct, d_r_dn_direct, G_inv_bar = vjp_fn(g)

    # Step 2: convert G_inv_bar -> G_bar using d(G^{-1}) = -G^{-1} dG G^{-1}:
    #   <G_inv_bar, dG^{-1}> = -<G_inv.T @ G_inv_bar @ G_inv.T, dG>
    G_bar_from_inv = -(G_inv_stable.T @ G_inv_bar @ G_inv_stable.T)

    # Step 3: propagate G_bar through the same fused value route that built the
    # forward-pass inverse. The cotangent must match that det_grad_lap output.
    _, vjp_fn2 = jax.vjp(_geminal_for_grads_lap, geminal_data, r_up_carts, r_dn_carts)
    # Cast target follows the lambda dtype
    # (complex at non-Gamma) so the cotangent matches the geminal primal
    # output dtype on the complex Bloch path. Bit-identical at Gamma.
    G_bar_from_inv = cast_to_zone(G_bar_from_inv, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    d_geminal_inv, d_r_up_inv, d_r_dn_inv = vjp_fn2(G_bar_from_inv)

    # Total: sum both contributions.
    d_geminal = jax.tree_util.tree_map(lambda a, b: a + b, d_geminal_direct, d_geminal_inv)
    return d_geminal, d_r_up_direct + d_r_up_inv, d_r_dn_direct + d_r_dn_inv


compute_grads_and_laplacian_ln_Det.defvjp(_grads_lap_fwd, _grads_lap_bwd)


# ---------------------------------------------------------------------------
# Grads & Laplacian of ln|Det| -- fast variant reusing a precomputed
# geminal_inverse (no custom_vjp; value-only fast recompute).
# ---------------------------------------------------------------------------


@jit
def compute_grads_and_laplacian_ln_Det_fast(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Gradients and Laplacians of ln det G using a precomputed inverse.

    Args:
        geminal_data: Geminal parameters and orbital references.
        r_up_carts: Cartesian coordinates of spin-up electrons with shape ``(N_up, 3)``.
        r_dn_carts: Cartesian coordinates of spin-down electrons with shape ``(N_dn, 3)``.
        geminal_inverse: Precomputed inverse of the geminal matrix ``G`` at the
            supplied ``(r_up_carts, r_dn_carts)``.

    Returns:
        Gradients (up/down) and Laplacians (up/down) of ln det G per electron.

    Warning:
        ``geminal_inverse`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  This is only guaranteed when
        the inverse is maintained via **single-electron (rank-1) Sherman-Morrison
        updates** starting from a freshly initialized LU inverse -- the pattern
        used in the MCMC loop.  Passing an inverse that corresponds to different
        electron positions silently produces incorrect kinetic energy.
    """
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for fast evaluation")

    dtype_jnp = jnp.result_type(get_dtype_jnp("det_grad_lap"), geminal_data._determinant_dtype_jnp)
    # r_*_carts and geminal_inverse are only forwarded; do not pre-cast
    # (Principle 3a -- no parameter rebind). geminal_inverse is cast to the
    # det_grad_lap zone at each einsum use site below (Principle 3b).

    # Explicitly upcast AO/MO forward values to the det_grad_lap zone
    # (compute_orb_api may return ao_eval / mo_eval dtype, e.g. fp32 for AGP) to avoid
    # relying on JAX implicit type promotion in the lambda/gradient matmuls below.
    # Single fused dispatch shares the heavy block (exp / poly / S_l_m) across
    # val/grad/lap.
    ao_matrix_up, ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z, ao_matrix_laplacian_up = (
        geminal_data.compute_orb_value_grad_lap_api(geminal_data.orb_data_up_spin, r_up_carts)
    )
    ao_matrix_dn, ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z, ao_matrix_laplacian_dn = (
        geminal_data.compute_orb_value_grad_lap_api(geminal_data.orb_data_dn_spin, r_dn_carts)
    )
    ao_matrix_up = cast_to_zone(ao_matrix_up, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    ao_matrix_dn = cast_to_zone(ao_matrix_dn, "det_grad_lap", geminal_data._determinant_dtype_jnp)

    ao_up_grads = jnp.stack([ao_matrix_up_grad_x, ao_matrix_up_grad_y, ao_matrix_up_grad_z], axis=0)
    ao_dn_grads = jnp.stack([ao_matrix_dn_grad_x, ao_matrix_dn_grad_y, ao_matrix_dn_grad_z], axis=0)

    # Only the electron-diagonal contractions are needed here. Contract the
    # inverse before the orbital derivatives, avoiding four full derivative
    # matrices per spin. These are bilinear products, including complex Bloch
    # orbitals: transpose is intentional, never conjugate transpose.
    G_inv = cast_to_zone(geminal_inverse, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    n_dn = geminal_data.num_electron_dn
    if isinstance(geminal_data, SD_data):
        n_up = geminal_data.num_occ_up
        weights_up = jnp.concatenate([ao_matrix_dn @ G_inv[:n_dn], G_inv[n_dn:]], axis=0)
        weights_dn = ao_matrix_up[:n_dn] @ G_inv[:n_dn].T
        ao_up_grads = ao_up_grads[:, :n_up]
        ao_matrix_laplacian_up = ao_matrix_laplacian_up[:n_up]
    else:
        paired, unpaired = _lambda_blocks_jnp(geminal_data)
        paired = jnp.asarray(paired, dtype=dtype_jnp)
        unpaired = jnp.asarray(unpaired, dtype=dtype_jnp)
        weights_up = jnp.concatenate([paired @ ao_matrix_dn, unpaired], axis=1) @ G_inv
        weights_dn = paired.T @ (ao_matrix_up @ G_inv[:n_dn].T)

    grad_ln_D_up = jnp.einsum("gan,an->ng", ao_up_grads, weights_up)
    grad_ln_D_dn = jnp.einsum("gan,an->ng", ao_dn_grads, weights_dn)
    lap_ln_D_up = jnp.einsum("an,an->n", ao_matrix_laplacian_up, weights_up) - jnp.sum(grad_ln_D_up**2, axis=1)
    lap_ln_D_dn = jnp.einsum("an,an->n", ao_matrix_laplacian_dn, weights_dn) - jnp.sum(grad_ln_D_dn**2, axis=1)
    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


# ---------------------------------------------------------------------------
# Streaming variant of ``compute_grads_and_laplacian_ln_Det_fast``.
#
# Maintains (a) AO/grad/lap tables, (b) lambda_p x ao_dn intermediates, and
# (c) the full geminal grad/lap matrices, all consistent with the current
# (r_up, r_dn). A single-electron move advances them in O(n_ao^2 + n_ao*N_e
# + N_e^2) per call, vs. O(n_ao^2*N_e + n_ao*N_e^2) for fresh recompute.
# ---------------------------------------------------------------------------


@struct.dataclass
class Det_streaming_state:
    """Auxiliary tables required to evaluate ``nablaln|Det|`` / ``nabla^2ln|Det|``
    incrementally under single-electron moves.

    The field ``paired_up_lambda`` makes this state a structural superset of
    :class:`Det_ratio_streaming_state`, so it can be consumed directly by
    ratio kernels (``_compute_ratio_determinant_part_split_spin`` and
    friends) via duck-typing. The four "ratio fields" (``ao_up``,
    ``ao_dn``, ``paired_dn``, ``paired_up_lambda``) carry the same invariants
    as the slim state.
    """

    ao_up: jax.Array
    ao_dn: jax.Array
    ao_up_grads: jax.Array
    ao_dn_grads: jax.Array
    ao_up_lap: jax.Array
    ao_dn_lap: jax.Array
    paired_dn: jax.Array
    paired_dn_grads: jax.Array
    paired_dn_lap: jax.Array
    paired_up_lambda: jax.Array  # (N_up, n_ao_dn) = ao_up.T @ lambda_paired
    geminal_grad_up: jax.Array
    geminal_grad_dn: jax.Array
    geminal_lap_up: jax.Array
    geminal_lap_dn: jax.Array
    grad_ln_D_up: jax.Array
    grad_ln_D_dn: jax.Array
    lap_ln_D_up: jax.Array
    lap_ln_D_dn: jax.Array


def _det_streaming_finalize(
    geminal_grad_up: jax.Array,
    geminal_grad_dn: jax.Array,
    geminal_lap_up: jax.Array,
    geminal_lap_dn: jax.Array,
    G_inv: jax.Array,
    n_dn: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Contract the fast-path ``geminal_*`` fields with ``G_inv``.

    Mirrors the tail of ``compute_grads_and_laplacian_ln_Det_fast`` so the
    streaming and fresh paths produce bit-comparable results when fed
    identical ``geminal_*`` and ``G_inv``.
    """
    grad_ln_D_up_stack = jnp.einsum("gij,ji->gi", geminal_grad_up, G_inv)
    grad_ln_D_dn_stack = jnp.einsum("ij,gji->gi", G_inv, geminal_grad_dn)

    grad_ln_D_up = grad_ln_D_up_stack.T
    grad_ln_D_dn_full = grad_ln_D_dn_stack.T

    grad_ln_D_up_x, grad_ln_D_up_y, grad_ln_D_up_z = grad_ln_D_up_stack
    grad_ln_D_dn_x, grad_ln_D_dn_y, grad_ln_D_dn_z = grad_ln_D_dn_stack

    lap_ln_D_up = -(
        grad_ln_D_up_x * grad_ln_D_up_x + grad_ln_D_up_y * grad_ln_D_up_y + grad_ln_D_up_z * grad_ln_D_up_z
    ) + jnp.einsum("ij,ji->i", geminal_lap_up, G_inv)
    lap_ln_D_dn_full = -(
        grad_ln_D_dn_x * grad_ln_D_dn_x + grad_ln_D_dn_y * grad_ln_D_dn_y + grad_ln_D_dn_z * grad_ln_D_dn_z
    ) + jnp.einsum("ij,ji->i", G_inv, geminal_lap_dn)

    grad_ln_D_dn = grad_ln_D_dn_full[:n_dn]
    lap_ln_D_dn = lap_ln_D_dn_full[:n_dn]
    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


@jit
def _init_grads_laplacian_ln_Det_streaming_state(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    geminal_inverse: jax.Array,
) -> Det_streaming_state:
    """Initialize the det streaming state at ``(r_up, r_dn)``.

    Cost is dominated by the same ``lambda_p x ao_dn`` and AO einsums as
    :func:`compute_grads_and_laplacian_ln_Det_fast`; the streaming path
    additionally retains the intermediate tables needed by the rank-one
    advance.
    """
    if geminal_inverse is None:
        raise ValueError("geminal_inverse must be provided for streaming init")

    # Evaluate AO values, gradients, and Laplacians (forward r_*_carts unchanged so the
    # underlying kernels reconstruct r-R in fp64 -- Principle 3b). Single fused
    # dispatch shares the heavy block (exp / poly / S_l_m) across val/grad/lap.
    ao_up, ao_up_gx, ao_up_gy, ao_up_gz, ao_up_lap = geminal_data.compute_orb_value_grad_lap_api(
        geminal_data.orb_data_up_spin, r_up_carts
    )
    ao_dn, ao_dn_gx, ao_dn_gy, ao_dn_gz, ao_dn_lap = geminal_data.compute_orb_value_grad_lap_api(
        geminal_data.orb_data_dn_spin, r_dn_carts
    )

    # Promote dtype_jnp against the actual AO/grad/lap dtypes so
    # the casts below preserve Im on complex Bloch (AO leaves can be complex
    # while lambda is real for RHF singlet). See the analogous handling in
    # jqmc/wavefunction.py (compute_kinetic_energy_all_elements*).
    dtype_jnp = jnp.result_type(
        get_dtype_jnp("det_grad_lap"),
        geminal_data._determinant_dtype_jnp,
        ao_up.dtype,
        ao_dn.dtype,
    )

    ao_up = ao_up.astype(dtype_jnp)
    ao_dn = ao_dn.astype(dtype_jnp)
    ao_up_grads = jnp.asarray(jnp.stack([ao_up_gx, ao_up_gy, ao_up_gz], axis=0), dtype=dtype_jnp)
    ao_dn_grads = jnp.asarray(jnp.stack([ao_dn_gx, ao_dn_gy, ao_dn_gz], axis=0), dtype=dtype_jnp)
    ao_up_lap = jnp.asarray(ao_up_lap, dtype=dtype_jnp)
    ao_dn_lap = jnp.asarray(ao_dn_lap, dtype=dtype_jnp)

    tables = _geminal_spatial_derivative_tables(
        geminal_data,
        ao_up,
        ao_up_grads,
        ao_up_lap,
        ao_dn,
        ao_dn_grads,
        ao_dn_lap,
        dtype_jnp,
    )
    paired_dn = tables["paired_dn"]
    paired_dn_grads = tables["paired_dn_grads"]
    paired_dn_lap = tables["paired_dn_lap"]
    paired_up_lambda = tables["paired_up"]
    geminal_grad_up = tables["grad_up"]
    geminal_grad_dn = tables["grad_dn"]
    geminal_lap_up = tables["lap_up"]
    geminal_lap_dn = tables["lap_dn"]

    # Contract with G_inv to expose per-electron gradients and Laplacians.
    G_inv = geminal_inverse.astype(dtype_jnp)
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = _det_streaming_finalize(
        geminal_grad_up,
        geminal_grad_dn,
        geminal_lap_up,
        geminal_lap_dn,
        G_inv,
        geminal_data.num_electron_dn,
    )

    return Det_streaming_state(
        ao_up=ao_up,
        ao_dn=ao_dn,
        ao_up_grads=ao_up_grads,
        ao_dn_grads=ao_dn_grads,
        ao_up_lap=ao_up_lap,
        ao_dn_lap=ao_dn_lap,
        paired_dn=paired_dn,
        paired_dn_grads=paired_dn_grads,
        paired_dn_lap=paired_dn_lap,
        paired_up_lambda=paired_up_lambda,
        geminal_grad_up=geminal_grad_up,
        geminal_grad_dn=geminal_grad_dn,
        geminal_lap_up=geminal_lap_up,
        geminal_lap_dn=geminal_lap_dn,
        grad_ln_D_up=grad_ln_D_up,
        grad_ln_D_dn=grad_ln_D_dn,
        lap_ln_D_up=lap_ln_D_up,
        lap_ln_D_dn=lap_ln_D_dn,
    )


def _advance_sd_grads_laplacian_ln_Det_streaming_state(
    geminal_data: SD_data,
    state: Det_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
    A_new_inv: jax.Array,
) -> Det_streaming_state:
    """Row/column advance of the direct occupied-SD derivative state.

    ``SD_data`` stores exactly the occupied orbitals, so the Slater matrix is
    ``G = [Phi_up[:N_dn]^T Phi_dn | Phi_up[N_dn:]^T]`` and the state built by
    :func:`_init_grads_laplacian_ln_Det_streaming_state` holds

    * ``paired_dn = ao_dn`` (and its gradient / Laplacian tables),
    * ``paired_up_lambda = ao_up[:N_dn].T``,
    * ``geminal_grad_up[:, i, :] = [grad_i Phi_up[:N_dn]^T Phi_dn | grad_i Phi_up[N_dn:]^T]``
      and the analogous Laplacian / dn tables.

    Moving up electron ``k`` changes column ``k`` of the up tables and row
    ``k`` of the four geminal tables; moving dn electron ``k`` changes column
    ``k`` of the dn tables and column ``k`` (paired block) of the geminal
    tables. Both are written with one single-point orbital evaluation at the
    new position, the same casts as the fresh build, and the transpose without
    conjugation (Convention B; the dn conjugation lives in the MO coefficients).
    The contraction with ``A_new_inv`` reuses :func:`_det_streaming_finalize`.

    Cost: one orbital point plus ``O(N_e^2)`` table work and the ``O(N_e^2)``
    finalize, instead of the full all-electron orbital re-evaluation.
    """
    dtype = state.ao_up.dtype
    n_dn = geminal_data.num_occ_dn
    n_up = geminal_data.num_occ_up
    num_up = state.ao_up.shape[1]
    num_dn = state.ao_dn.shape[1]
    G_inv = A_new_inv.astype(dtype)

    def _single_point(orb_data, r_row):
        ao_v, gx, gy, gz, ao_lap = geminal_data.compute_orb_value_grad_lap_api(orb_data, jnp.expand_dims(r_row, axis=0))
        val = ao_v[:, 0].astype(dtype)
        grad = jnp.asarray(jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=0), dtype=dtype)
        lap = jnp.asarray(ao_lap[:, 0], dtype=dtype)
        return val, grad, lap

    def _finalize(new_state: Det_streaming_state) -> Det_streaming_state:
        grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = _det_streaming_finalize(
            new_state.geminal_grad_up,
            new_state.geminal_grad_dn,
            new_state.geminal_lap_up,
            new_state.geminal_lap_dn,
            G_inv,
            num_dn,
        )
        return new_state.replace(
            grad_ln_D_up=grad_ln_D_up,
            grad_ln_D_dn=grad_ln_D_dn,
            lap_ln_D_up=lap_ln_D_up,
            lap_ln_D_dn=lap_ln_D_dn,
        )

    def _branch_up(_: None) -> Det_streaming_state:
        k = moved_index
        val, grad, lap = _single_point(geminal_data.orb_data_up_spin, r_up_carts_new[k])
        zeros_grad = jnp.zeros((3, num_up - num_dn), dtype=dtype)
        zeros_lap = jnp.zeros((num_up - num_dn,), dtype=dtype)
        row_grad_up = jnp.concatenate(
            [jnp.einsum("ga,aj->gj", grad[:, :n_dn], state.paired_dn), grad[:, n_dn:n_up]],
            axis=1,
        )
        row_grad_dn = jnp.concatenate([jnp.einsum("a,gaj->gj", val[:n_dn], state.paired_dn_grads), zeros_grad], axis=1)
        row_lap_up = jnp.concatenate([lap[:n_dn] @ state.paired_dn, lap[n_dn:n_up]], axis=0)
        row_lap_dn = jnp.concatenate([val[:n_dn] @ state.paired_dn_lap, zeros_lap], axis=0)
        return _finalize(
            state.replace(
                ao_up=_complex_update.at_set(state.ao_up, jnp.s_[:, k], val),
                ao_up_grads=_complex_update.at_set(state.ao_up_grads, jnp.s_[:, :, k], grad),
                ao_up_lap=_complex_update.at_set(state.ao_up_lap, jnp.s_[:, k], lap),
                paired_up_lambda=_complex_update.at_set(state.paired_up_lambda, jnp.s_[k, :], val[:n_dn]),
                geminal_grad_up=_complex_update.at_set(state.geminal_grad_up, jnp.s_[:, k, :], row_grad_up),
                geminal_grad_dn=_complex_update.at_set(state.geminal_grad_dn, jnp.s_[:, k, :], row_grad_dn),
                geminal_lap_up=_complex_update.at_set(state.geminal_lap_up, jnp.s_[k, :], row_lap_up),
                geminal_lap_dn=_complex_update.at_set(state.geminal_lap_dn, jnp.s_[k, :], row_lap_dn),
            )
        )

    def _branch_dn(_: None) -> Det_streaming_state:
        k = moved_index
        val, grad, lap = _single_point(geminal_data.orb_data_dn_spin, r_dn_carts_new[k])
        new_ao_dn = _complex_update.at_set(state.ao_dn, jnp.s_[:, k], val)
        new_ao_dn_grads = _complex_update.at_set(state.ao_dn_grads, jnp.s_[:, :, k], grad)
        new_ao_dn_lap = _complex_update.at_set(state.ao_dn_lap, jnp.s_[:, k], lap)
        col_grad_up = jnp.einsum("gai,a->gi", state.ao_up_grads[:, :n_dn, :], val)
        col_grad_dn = jnp.einsum("ia,ga->gi", state.paired_up_lambda, grad)
        col_lap_up = state.ao_up_lap[:n_dn].T @ val
        col_lap_dn = state.paired_up_lambda @ lap
        return _finalize(
            state.replace(
                ao_dn=new_ao_dn,
                ao_dn_grads=new_ao_dn_grads,
                ao_dn_lap=new_ao_dn_lap,
                paired_dn=new_ao_dn,
                paired_dn_grads=new_ao_dn_grads,
                paired_dn_lap=new_ao_dn_lap,
                geminal_grad_up=_complex_update.at_set(state.geminal_grad_up, jnp.s_[:, :, k], col_grad_up),
                geminal_grad_dn=_complex_update.at_set(state.geminal_grad_dn, jnp.s_[:, :, k], col_grad_dn),
                geminal_lap_up=_complex_update.at_set(state.geminal_lap_up, jnp.s_[:, k], col_lap_up),
                geminal_lap_dn=_complex_update.at_set(state.geminal_lap_dn, jnp.s_[:, k], col_lap_dn),
            )
        )

    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    # Same tree_map + jnp.where selection as the AGP advance (see the note there).
    return jax.tree_util.tree_map(lambda a, b: jnp.where(moved_spin_is_up, a, b), _branch_up(None), _branch_dn(None))


@jit
def _advance_grads_laplacian_ln_Det_streaming_state(
    geminal_data: Geminal_data,
    state: Det_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
    A_new_inv: jax.Array,
) -> Det_streaming_state:
    """Advance the det streaming state after a single-electron move.

    The new ``(r_up_carts_new, r_dn_carts_new)`` differ from the configuration
    represented by ``state`` in *exactly one* electron position, identified by
    ``(moved_spin_is_up, moved_index)``. ``A_new_inv`` is the Sherman-Morrison
    inverse of ``G(r_up_new, r_dn_new)`` provided by the caller (typically
    ``_body_step_core`` in ``jqmc_gfmc.py``).

    Cost: ``O(n_ao^2 + n_ao * N_e + N_e^2)`` per call.
    """
    dtype_jnp = jnp.result_type(get_dtype_jnp("det_grad_lap"), geminal_data._determinant_dtype_jnp)

    if isinstance(geminal_data, SD_data):
        # The direct occupied-SD state has no lambda operator: advance the
        # moved electron's row/column of the occupied-orbital tables directly
        # ``_init_grads_laplacian_ln_Det_streaming_state`` stays
        # the from-scratch correctness reference.
        return _advance_sd_grads_laplacian_ln_Det_streaming_state(
            geminal_data,
            state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
            A_new_inv,
        )

    lambda_matrix_paired, lambda_matrix_unpaired = jnp.hsplit(geminal_data._lambda_matrix_jnp, [geminal_data.orb_num_dn])
    lambda_matrix_paired = cast_to_zone(lambda_matrix_paired, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_grad_lap", geminal_data._determinant_dtype_jnp)

    num_up = state.ao_up.shape[1]
    num_dn = state.ao_dn.shape[1]
    G_inv = cast_to_zone(A_new_inv, "det_grad_lap", geminal_data._determinant_dtype_jnp)

    def _branch_up(_: None) -> Det_streaming_state:
        # --- Single-point AO evaluation at r_up_new[k] -------------------
        # Single fused dispatch shares the heavy block (exp / poly / S_l_m)
        # across val/grad/lap -- replaces three separate compute_orb_* calls.
        r_new = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        ao_v, gx, gy, gz, ao_lap = geminal_data.compute_orb_value_grad_lap_api(geminal_data.orb_data_up_spin, r_new)
        ao_col = cast_to_zone(ao_v[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_up,)
        grad_col = cast_to_zone(
            jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=0), "det_grad_lap", geminal_data._determinant_dtype_jnp
        )  # (3, n_ao_up)
        lap_col = cast_to_zone(ao_lap[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_up,)

        new_ao_up = _complex_update.at_set(state.ao_up, jnp.s_[:, moved_index], ao_col)
        new_ao_up_grads = _complex_update.at_set(state.ao_up_grads, jnp.s_[:, :, moved_index], grad_col)
        new_ao_up_lap = _complex_update.at_set(state.ao_up_lap, jnp.s_[:, moved_index], lap_col)

        # --- paired_up_lambda row: ao_col @ lambda_paired -> (n_ao_dn,)
        # depends on ao_up only, so on dn-move it stays unchanged.
        new_paired_up_lambda_row = jnp.dot(ao_col, lambda_matrix_paired)
        new_paired_up_lambda = _complex_update.at_set(state.paired_up_lambda, jnp.s_[moved_index, :], new_paired_up_lambda_row)

        # --- Row k of geminal_* (paired || unpaired) ----------------------
        # row of geminal_grad_up: einsum("ga,aj->gj", grad_col, paired_dn) ||
        #                        einsum("ga,ak->gk", grad_col, lambda_u)
        row_grad_up_paired = jnp.einsum("ga,aj->gj", grad_col, state.paired_dn)
        row_grad_up_unpaired = jnp.einsum("ga,ak->gk", grad_col, lambda_matrix_unpaired)
        row_grad_up = jnp.concatenate([row_grad_up_paired, row_grad_up_unpaired], axis=1)
        new_geminal_grad_up = _complex_update.at_set(state.geminal_grad_up, jnp.s_[:, moved_index, :], row_grad_up)

        # row of geminal_grad_dn: paired = einsum("a,gaj->gj", ao_col, paired_dn_grads),
        # unpaired = zeros.
        row_grad_dn_paired = jnp.einsum("a,gaj->gj", ao_col, state.paired_dn_grads)
        row_grad_dn_unpaired = jnp.zeros((3, num_up - num_dn), dtype=dtype_jnp)
        row_grad_dn = jnp.concatenate([row_grad_dn_paired, row_grad_dn_unpaired], axis=1)
        new_geminal_grad_dn = _complex_update.at_set(state.geminal_grad_dn, jnp.s_[:, moved_index, :], row_grad_dn)

        # row of geminal_lap_up: lap_col @ paired_dn || lap_col @ lambda_u
        row_lap_up_paired = lap_col @ state.paired_dn
        row_lap_up_unpaired = lap_col @ lambda_matrix_unpaired
        row_lap_up = jnp.concatenate([row_lap_up_paired, row_lap_up_unpaired], axis=0)
        new_geminal_lap_up = _complex_update.at_set(state.geminal_lap_up, jnp.s_[moved_index, :], row_lap_up)

        # row of geminal_lap_dn: ao_col @ paired_dn_lap || zeros
        row_lap_dn_paired = ao_col @ state.paired_dn_lap
        row_lap_dn_unpaired = jnp.zeros((num_up - num_dn,), dtype=dtype_jnp)
        row_lap_dn = jnp.concatenate([row_lap_dn_paired, row_lap_dn_unpaired], axis=0)
        new_geminal_lap_dn = _complex_update.at_set(state.geminal_lap_dn, jnp.s_[moved_index, :], row_lap_dn)

        grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = _det_streaming_finalize(
            new_geminal_grad_up,
            new_geminal_grad_dn,
            new_geminal_lap_up,
            new_geminal_lap_dn,
            G_inv,
            num_dn,
        )

        return state.replace(
            ao_up=new_ao_up,
            ao_up_grads=new_ao_up_grads,
            ao_up_lap=new_ao_up_lap,
            paired_up_lambda=new_paired_up_lambda,
            geminal_grad_up=new_geminal_grad_up,
            geminal_grad_dn=new_geminal_grad_dn,
            geminal_lap_up=new_geminal_lap_up,
            geminal_lap_dn=new_geminal_lap_dn,
            grad_ln_D_up=grad_ln_D_up,
            grad_ln_D_dn=grad_ln_D_dn,
            lap_ln_D_up=lap_ln_D_up,
            lap_ln_D_dn=lap_ln_D_dn,
        )

    def _branch_dn(_: None) -> Det_streaming_state:
        # --- Single-point AO evaluation at r_dn_new[k] -------------------
        # Single fused dispatch (see _branch_up).
        r_new = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        ao_v, gx, gy, gz, ao_lap = geminal_data.compute_orb_value_grad_lap_api(geminal_data.orb_data_dn_spin, r_new)
        ao_col = cast_to_zone(ao_v[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_dn,)
        grad_col = cast_to_zone(
            jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=0), "det_grad_lap", geminal_data._determinant_dtype_jnp
        )  # (3, n_ao_dn)
        lap_col = cast_to_zone(ao_lap[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_dn,)

        new_ao_dn = _complex_update.at_set(state.ao_dn, jnp.s_[:, moved_index], ao_col)
        new_ao_dn_grads = _complex_update.at_set(state.ao_dn_grads, jnp.s_[:, :, moved_index], grad_col)
        new_ao_dn_lap = _complex_update.at_set(state.ao_dn_lap, jnp.s_[:, moved_index], lap_col)

        # --- Column k of the paired_dn family ----------------------------
        new_paired_dn_col = lambda_matrix_paired @ ao_col  # (n_ao_up,)
        new_paired_dn = _complex_update.at_set(state.paired_dn, jnp.s_[:, moved_index], new_paired_dn_col)

        new_paired_dn_grads_col = jnp.einsum("ab,gb->ga", lambda_matrix_paired, grad_col)  # (3, n_ao_up)
        new_paired_dn_grads = _complex_update.at_set(state.paired_dn_grads, jnp.s_[:, :, moved_index], new_paired_dn_grads_col)

        new_paired_dn_lap_col = lambda_matrix_paired @ lap_col  # (n_ao_up,)
        new_paired_dn_lap = _complex_update.at_set(state.paired_dn_lap, jnp.s_[:, moved_index], new_paired_dn_lap_col)

        # --- Column k of geminal_* (paired block only; unpaired
        #              columns are independent of dn). --------------------
        # geminal_grad_up[:, :, k] paired-block update:
        col_grad_up = jnp.einsum("gia,a->gi", jnp.swapaxes(state.ao_up_grads, 1, 2), new_paired_dn_col)
        new_geminal_grad_up = _complex_update.at_set(state.geminal_grad_up, jnp.s_[:, :, moved_index], col_grad_up)

        # geminal_grad_dn[:, :, k] paired-block update:
        col_grad_dn = jnp.einsum("ai,ga->gi", state.ao_up, new_paired_dn_grads_col)
        new_geminal_grad_dn = _complex_update.at_set(state.geminal_grad_dn, jnp.s_[:, :, moved_index], col_grad_dn)

        # geminal_lap_up[:, k] paired-block update:
        col_lap_up = state.ao_up_lap.T @ new_paired_dn_col
        new_geminal_lap_up = _complex_update.at_set(state.geminal_lap_up, jnp.s_[:, moved_index], col_lap_up)

        # geminal_lap_dn[:, k] paired-block update:
        col_lap_dn = state.ao_up.T @ new_paired_dn_lap_col
        new_geminal_lap_dn = _complex_update.at_set(state.geminal_lap_dn, jnp.s_[:, moved_index], col_lap_dn)

        grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = _det_streaming_finalize(
            new_geminal_grad_up,
            new_geminal_grad_dn,
            new_geminal_lap_up,
            new_geminal_lap_dn,
            G_inv,
            num_dn,
        )

        return state.replace(
            ao_dn=new_ao_dn,
            ao_dn_grads=new_ao_dn_grads,
            ao_dn_lap=new_ao_dn_lap,
            paired_dn=new_paired_dn,
            paired_dn_grads=new_paired_dn_grads,
            paired_dn_lap=new_paired_dn_lap,
            geminal_grad_up=new_geminal_grad_up,
            geminal_grad_dn=new_geminal_grad_dn,
            geminal_lap_up=new_geminal_lap_up,
            geminal_lap_dn=new_geminal_lap_dn,
            grad_ln_D_up=grad_ln_D_up,
            grad_ln_D_dn=grad_ln_D_dn,
            lap_ln_D_up=lap_ln_D_up,
            lap_ln_D_dn=lap_ln_D_dn,
        )

    # Edge case: zero-electron spin sectors collapse the cond.
    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    # tree_map + jnp.where instead of lax.cond: the batched-predicate cond
    # form miscompiles on XLA CPU (jax 0.7.x) inside the GFMC vmapped
    # projection loops (leaf mixing across branches, observed in the J3
    # grads advance with the incremental interval potential); under vmap
    # both branches execute anyway, so this costs nothing extra. See the
    # matching note in jqmc_gfmc._projection_t_core.
    return jax.tree_util.tree_map(lambda a, b: jnp.where(moved_spin_is_up, a, b), _branch_up(None), _branch_dn(None))


# ---------------------------------------------------------------------------
# Det_ratio_streaming_state -- slim, value-only streaming state for ratio
# kernels (MCMC walk body, GFMC inv-update, LRDMC mesh ratio kernels).
#
# Compared with Det_streaming_state above, this slim variant carries only
# the four fields needed for ratio computation; no grad/lap fields.
#
# Invariants maintained at every accepted single-electron move:
#     ao_up            == compute_orb_api(orb_data_up_spin, r_up_carts)
#     ao_dn            == compute_orb_api(orb_data_dn_spin, r_dn_carts)
#     paired_dn        == lambda_matrix_paired @ ao_dn
#     paired_up_lambda == ao_up.T @ lambda_matrix_paired
#
# All four fields are *refreshed* (not rank-1 advanced) on accept: when an
# electron moves, the corresponding column or row is recomputed from the
# fresh AO at the new position. This means drift accumulation does not
# occur in the slim state itself.
# ---------------------------------------------------------------------------


@struct.dataclass
class Det_ratio_streaming_state:
    """Value-only AO + paired tables for ratio kernels.

    Fields:
        ao_up: ``(n_ao_up, N_up)`` -- per-electron AO evaluations for up spin.
        ao_dn: ``(n_ao_dn, N_dn)`` -- per-electron AO evaluations for dn spin.
        paired_dn: ``(n_ao_up, N_dn)`` -- precomputed ``lambda_paired @ ao_dn``.
        paired_up_lambda: ``(N_up, n_ao_dn)`` -- precomputed ``ao_up.T @ lambda_paired``.
    """

    ao_up: jax.Array
    ao_dn: jax.Array
    paired_dn: jax.Array
    paired_up_lambda: jax.Array


@jit
def _init_det_ratio_streaming_state(
    geminal_data: "Geminal_data",
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Det_ratio_streaming_state:
    """Initialize the slim ratio streaming state at ``(r_up_carts, r_dn_carts)``.

    Performs one full AO evaluation per spin and two ``(n_ao, n_ao) @ (n_ao, N_e)``
    matmuls. Subsequent advances refresh only one column/row per accepted
    single-electron move.
    """
    # The cast target follows the lambda dtype
    # (complex at non-Gamma Bloch geminal) so the streaming-state cache
    # does not silently truncate Im. Bit-identical at Gamma.
    # Forward r_up/dn_carts as-is (Principle 3a). compute_orb_api handles
    # its own use-site casts.
    ao_up = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_up_carts),
        "det_grad_lap",
        geminal_data._determinant_dtype_jnp,
    )
    ao_dn = cast_to_zone(
        geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_dn_carts),
        "det_grad_lap",
        geminal_data._determinant_dtype_jnp,
    )

    paired_dn = _contract_paired_dn(geminal_data, ao_dn)
    paired_up_lambda = _contract_up_paired(geminal_data, ao_up)

    return Det_ratio_streaming_state(
        ao_up=ao_up,
        ao_dn=ao_dn,
        paired_dn=paired_dn,
        paired_up_lambda=paired_up_lambda,
    )


@jit
def _geminal_from_det_ratio_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state | Det_streaming_state,
) -> jax.Array:
    """Return the geminal ``G`` implied by the streaming determinant caches.

    Rows are assembled exactly as the Sherman--Morrison ``v`` rows of
    :func:`_compute_v_up_move_from_det_ratio_state` (``det_ratio`` zone), so
    the result is the matrix a cache-driven running inverse actually inverts
    (no orbital evaluation; ``O(N^2 n_ao)``). Used by the carried-inverse drift probe.
    """
    dtype = geminal_data._determinant_dtype_jnp
    ao_up = cast_to_zone(state.ao_up, "det_ratio", dtype)
    paired_dn = cast_to_zone(state.paired_dn, "det_ratio", dtype)
    if isinstance(geminal_data, SD_data):
        paired = ao_up[: geminal_data.num_occ_dn].T @ paired_dn
        unpaired = ao_up[geminal_data.num_occ_dn : geminal_data.num_occ_up].T
    else:
        _, lambda_matrix_unpaired = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_ratio", dtype)
        paired = ao_up.T @ paired_dn
        unpaired = ao_up.T @ lambda_matrix_unpaired
    return jnp.concatenate([paired, unpaired], axis=1)


def _det_state_inverse_probe_residual(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state | Det_streaming_state,
    geminal_inv: jax.Array,
) -> jax.Array:
    """Probe residual of a carried inverse against the cache-implied geminal, O(N^2).

    Returns ``max|G (X P) - P|`` for ONE walker, where ``G`` is the geminal of
    :func:`_geminal_from_det_ratio_state` and ``P`` the two deterministic probe
    columns of ``jqmc_mcmc._running_geminal_inv_probe_residual`` (all ones and
    ``cos(1 + 0.7 j)``). ``G`` is never formed: ``G y`` is applied in its
    factored form ``ao_up^T (paired_dn y_p) (+ unpaired block)``, so the cost is
    O(N^2 + N n_ao) per call (per-hop screen; the Rust twin is
    ``determinant::det_state_inverse_probe_residual``). A non-finite inverse
    yields NaN, which never triggers a refresh.
    """
    dtype = geminal_data._determinant_dtype_jnp
    n = geminal_inv.shape[-1]
    index = jnp.arange(n, dtype=jnp.float64)
    probes = jnp.stack([jnp.ones(n, dtype=jnp.float64), jnp.cos(1.0 + 0.7 * index)], axis=1).astype(geminal_inv.dtype)
    ao_up = cast_to_zone(state.ao_up, "det_ratio", dtype)
    paired_dn = cast_to_zone(state.paired_dn, "det_ratio", dtype)
    n_dn = paired_dn.shape[1]
    y = geminal_inv @ probes
    if isinstance(geminal_data, SD_data):
        g_y = (
            ao_up[: geminal_data.num_occ_dn].T @ (paired_dn @ y[:n_dn])
            + ao_up[geminal_data.num_occ_dn : geminal_data.num_occ_up].T @ y[n_dn:]
        )
    else:
        _, lambda_matrix_unpaired = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_ratio", dtype)
        g_y = ao_up.T @ (paired_dn @ y[:n_dn] + lambda_matrix_unpaired @ y[n_dn:])
    return jnp.max(jnp.abs(g_y - probes))


@jit
def _advance_det_ratio_streaming_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Det_ratio_streaming_state:
    """Advance the slim ratio streaming state after a single-electron move.

    The new ``(r_up_carts_new, r_dn_carts_new)`` differ from the configuration
    represented by ``state`` in *exactly one* electron position, identified by
    ``(moved_spin_is_up, moved_index)``.

    On up-move: refresh ``ao_up[:, k]`` and ``paired_up_lambda[k, :]``.
        ``paired_dn`` is unchanged because it depends on ``ao_dn`` only.
    On dn-move: refresh ``ao_dn[:, k]`` and ``paired_dn[:, k]``.
        ``paired_up_lambda`` is unchanged because it depends on ``ao_up`` only.

    Cost: O(n_ao + n_ao^2) per call -- one single-electron AO eval plus one
    ``(n_ao,) @ (n_ao, n_ao)`` matvec.
    """

    def _branch_up(_: None) -> Det_ratio_streaming_state:
        # Single-electron AO eval at the new up position.
        r_new = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        ao_v = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_new)
        ao_col = cast_to_zone(ao_v[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_up,)

        new_ao_up = _complex_update.at_set(state.ao_up, jnp.s_[:, moved_index], ao_col)
        # paired_up_lambda[k, :] = ao_col @ lambda_paired   (n_ao_dn,)
        if isinstance(geminal_data, SD_data):
            new_paired_up_lambda_row = ao_col[: geminal_data.num_occ_dn]
        else:
            lambda_matrix_paired, _ = _lambda_blocks_jnp(geminal_data)
            new_paired_up_lambda_row = jnp.dot(ao_col, lambda_matrix_paired)
        new_paired_up_lambda = _complex_update.at_set(state.paired_up_lambda, jnp.s_[moved_index, :], new_paired_up_lambda_row)
        # paired_dn unchanged (depends on ao_dn only).
        return state.replace(
            ao_up=new_ao_up,
            paired_up_lambda=new_paired_up_lambda,
        )

    def _branch_dn(_: None) -> Det_ratio_streaming_state:
        # Single-electron AO eval at the new dn position.
        r_new = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        ao_v = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_new)
        ao_col = cast_to_zone(ao_v[:, 0], "det_grad_lap", geminal_data._determinant_dtype_jnp)  # (n_ao_dn,)

        new_ao_dn = _complex_update.at_set(state.ao_dn, jnp.s_[:, moved_index], ao_col)
        # paired_dn[:, k] = lambda_paired @ ao_col   (n_ao_up,)
        if isinstance(geminal_data, SD_data):
            new_paired_dn_col = ao_col
        else:
            lambda_matrix_paired, _ = _lambda_blocks_jnp(geminal_data)
            new_paired_dn_col = jnp.dot(lambda_matrix_paired, ao_col)
        new_paired_dn = _complex_update.at_set(state.paired_dn, jnp.s_[:, moved_index], new_paired_dn_col)
        # paired_up_lambda unchanged (depends on ao_up only).
        return state.replace(
            ao_dn=new_ao_dn,
            paired_dn=new_paired_dn,
        )

    # Edge case: zero-electron spin sectors collapse the cond at trace time.
    num_up = state.ao_up.shape[1]
    num_dn = state.ao_dn.shape[1]
    if num_up == 0:
        return _branch_dn(None)
    if num_dn == 0:
        return _branch_up(None)
    # tree_map + jnp.where instead of lax.cond: the batched-predicate cond
    # form miscompiles on XLA CPU (jax 0.7.x) inside the GFMC vmapped
    # projection loops (leaf mixing across branches, observed in the J3
    # grads advance with the incremental interval potential); under vmap
    # both branches execute anyway, so this costs nothing extra. See the
    # matching note in jqmc_gfmc._projection_t_core.
    return jax.tree_util.tree_map(lambda a, b: jnp.where(moved_spin_is_up, a, b), _branch_up(None), _branch_dn(None))


# ---------------------------------------------------------------------------
# Det-ratio move extractors: build the rank-1 update vectors (v for an up move,
# u for a dn move) from the slim Det_ratio_streaming_state.
# ---------------------------------------------------------------------------


def _sherman_morrison_ratio_and_inverse(
    A_old_inv: jax.Array,
    u: jax.Array,
    v: jax.Array,
    output_dtype: npt.DTypeLike | None = None,
    *,
    moved_index: jax.Array | int | None = None,
    moved_is_up: jax.Array | bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the determinant ratio and production rank-one inverse update.

    This is the single arithmetic implementation consumed by MCMC, GFMC_n,
    and GFMC_t.  Keeping the operation un-jitted lets each caller inline it
    into its existing compiled kernel without introducing a call boundary or
    changing the operation order.  The shared entry point also lets
    the conditioned inverse-chain gate exercise production code rather than a
    test-local copy of the Sherman--Morrison formula.

    Large-ratio guard (issue #127): production callers pass the moved
    electron (``moved_index`` and ``moved_is_up``; an up move has
    ``u = e_k``, a dn move ``v = e_k``). The new inverse's column ``k`` (up)
    or row ``k`` (dn) is then exactly ``X e_k / R`` (``e_k^T X / R``), but
    the rank-1 form computes it as ``X e_k (1 - (R - 1) / R)`` and loses
    ``log10(eps |R|)`` digits -- every digit once ``|R| > 1 / eps`` (an
    electron returning from a deep orbital tail), after which the next move's
    ratio and Metropolis decision are wrong. When ``|R|`` exceeds
    :func:`jqmc._setting.sm_large_ratio_threshold` of the inverse dtype, that
    column / row is overwritten by its exact value (an O(N) write; the rest of
    the update is accurate because ``X e_k / R`` multiplies every other
    correction). Below the threshold the result is bit-identical to the
    plain rank-1 form. Without ``moved_index`` the plain rank-1 update is
    returned.
    """
    Ainv_u = A_old_inv @ u
    vT_Ainv = v.T @ A_old_inv
    det_ratio = 1.0 + (v.T @ Ainv_u)[0, 0]
    if output_dtype is None:
        output_dtype = A_old_inv.dtype
    A_new_inv = jnp.asarray(A_old_inv - (Ainv_u @ vT_Ainv) / det_ratio, dtype=output_dtype)
    if moved_index is not None:
        large = jnp.abs(det_ratio) > sm_large_ratio_threshold(A_old_inv.dtype)
        exact_col = jnp.asarray(Ainv_u[:, 0] / det_ratio, dtype=output_dtype)
        exact_row = jnp.asarray(vT_Ainv[0, :] / det_ratio, dtype=output_dtype)
        A_new_inv = _complex_update.at_set(
            A_new_inv, jnp.s_[:, moved_index], jnp.where(large & moved_is_up, exact_col, A_new_inv[:, moved_index])
        )
        A_new_inv = _complex_update.at_set(
            A_new_inv,
            jnp.s_[moved_index, :],
            jnp.where(large & jnp.logical_not(moved_is_up), exact_row, A_new_inv[moved_index, :]),
        )
    return det_ratio, A_new_inv


@jit
def _compute_v_up_move_from_det_ratio_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state,
    moved_index: jax.Array,
    r_up_carts_proposed: jax.Array,
) -> jax.Array:
    """Build ``v = (G_new[i, :] - G_old[i, :])`` for an up-move from cached state.

    Used in the MCMC walk body instead of calling
    ``compute_geminal_up_one_row_elements`` twice: it reuses ``state.ao_up``
    (old AO column) and ``state.paired_dn`` (= ``lambda_paired @ ao_dn``), so
    the heavy ``lambda @ ao_dn`` GEMM and the N_dn-bulk AO evaluation are not
    paid every step.

    Args:
        geminal_data: Geminal parameters (used for orb_data_up_spin and
            ``_lambda_matrix_jnp`` unpaired block).
        state: Streaming state consistent with the current ``(r_up_carts,
            r_dn_carts)``.
        moved_index: Up-electron index that is being proposed to move.
        r_up_carts_proposed: Proposed up-spin coordinates ``(N_up, 3)``;
            differs from current ``r_up_carts`` in row ``moved_index`` only.

    Returns:
        ``v`` with shape ``(N_up,)`` (= N_dn + num_unpaired).
    """
    # Single-electron AO eval at the proposed position.
    r_new = jnp.expand_dims(r_up_carts_proposed[moved_index], axis=0)  # (1, 3)
    ao_up_new_2d = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_new)
    ao_up_new = cast_to_zone(ao_up_new_2d[:, 0], "det_ratio", geminal_data._determinant_dtype_jnp)  # (n_ao_up,)

    # Cached old AO column (use-site cast; Principle 3b).
    ao_up_old = cast_to_zone(state.ao_up[:, moved_index], "det_ratio", geminal_data._determinant_dtype_jnp)
    delta_ao_up = ao_up_new - ao_up_old  # (n_ao_up,)

    # Paired block:  delta_ao_up @ paired_dn  -> (N_dn,)
    paired_dn = cast_to_zone(state.paired_dn, "det_ratio", geminal_data._determinant_dtype_jnp)
    if isinstance(geminal_data, SD_data):
        v_paired = jnp.dot(delta_ao_up[: geminal_data.num_occ_dn], paired_dn)
    else:
        v_paired = jnp.dot(delta_ao_up, paired_dn)

    if isinstance(geminal_data, SD_data):
        v_unpaired = delta_ao_up[geminal_data.num_occ_dn : geminal_data.num_occ_up]
    else:
        _, lambda_matrix_unpaired = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_ratio", geminal_data._determinant_dtype_jnp)
        v_unpaired = jnp.dot(delta_ao_up, lambda_matrix_unpaired)

    return jnp.concatenate([v_paired, v_unpaired])  # (N_up,)


@jit
def _compute_u_dn_move_from_det_ratio_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state,
    moved_index: jax.Array,
    r_dn_carts_proposed: jax.Array,
) -> jax.Array:
    """Build ``u = G_new[:, j] - G_old[:, j]`` for a dn-move from cached state.

    Used in the MCMC walk body instead of calling
    ``compute_geminal_dn_one_column_elements`` twice: it reuses
    ``state.ao_dn`` (old AO column) and ``state.paired_up_lambda``
    (= ``ao_up.T @ lambda_paired``).

    Args:
        geminal_data: Geminal parameters (used for orb_data_dn_spin).
        state: Streaming state consistent with the current ``(r_up_carts,
            r_dn_carts)``.
        moved_index: Dn-electron index that is being proposed to move.
        r_dn_carts_proposed: Proposed dn-spin coordinates ``(N_dn, 3)``;
            differs from current ``r_dn_carts`` in row ``moved_index`` only.

    Returns:
        ``u`` with shape ``(N_up,)``.
    """
    r_new = jnp.expand_dims(r_dn_carts_proposed[moved_index], axis=0)
    ao_dn_new_2d = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_new)
    ao_dn_new = cast_to_zone(ao_dn_new_2d[:, 0], "det_ratio", geminal_data._determinant_dtype_jnp)  # (n_ao_dn,)

    ao_dn_old = cast_to_zone(state.ao_dn[:, moved_index], "det_ratio", geminal_data._determinant_dtype_jnp)
    delta_ao_dn = ao_dn_new - ao_dn_old  # (n_ao_dn,)

    # u = paired_up_lambda @ delta_ao_dn   -> (N_up,)
    paired_up_lambda = cast_to_zone(state.paired_up_lambda, "det_ratio", geminal_data._determinant_dtype_jnp)
    return jnp.dot(paired_up_lambda, delta_ao_dn)


@jit
def _compute_up_row_from_det_ratio_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state,
    moved_index: jax.Array,
    r_up_carts_proposed: jax.Array,
) -> jax.Array:
    """Build the proposed geminal row ``G_new[k, :]`` of an up-move directly from cached state (issue #134).

    Same contraction as :func:`_compute_v_up_move_from_det_ratio_state` with the
    new orbital column in place of the difference ``new - old``. The AS factor of
    the proposed state uses this row instead of ``G[k, :] + v``: for a tiny
    determinant ratio (``|G_new[k, :]| < eps |G[k, :]|``) the sum cancels to
    round-off or exactly zero, and an exactly zero row made ``R_AS = 0`` and the
    AS ratio ``eps / 0 = inf`` (a forced acceptance).

    Returns:
        ``G_new[k, :]`` with shape ``(N_up,)`` (= N_dn + num_unpaired).
    """
    r_new = jnp.expand_dims(r_up_carts_proposed[moved_index], axis=0)  # (1, 3)
    ao_up_new_2d = geminal_data.compute_orb_api(geminal_data.orb_data_up_spin, r_new)
    ao_up_new = cast_to_zone(ao_up_new_2d[:, 0], "det_ratio", geminal_data._determinant_dtype_jnp)  # (n_ao_up,)
    paired_dn = cast_to_zone(state.paired_dn, "det_ratio", geminal_data._determinant_dtype_jnp)
    if isinstance(geminal_data, SD_data):
        row_paired = jnp.dot(ao_up_new[: geminal_data.num_occ_dn], paired_dn)
        row_unpaired = ao_up_new[geminal_data.num_occ_dn : geminal_data.num_occ_up]
    else:
        row_paired = jnp.dot(ao_up_new, paired_dn)
        _, lambda_matrix_unpaired = _lambda_blocks_jnp(geminal_data)
        lambda_matrix_unpaired = cast_to_zone(lambda_matrix_unpaired, "det_ratio", geminal_data._determinant_dtype_jnp)
        row_unpaired = jnp.dot(ao_up_new, lambda_matrix_unpaired)
    return jnp.concatenate([row_paired, row_unpaired])  # (N_up,)


@jit
def _compute_dn_column_from_det_ratio_state(
    geminal_data: "Geminal_data",
    state: Det_ratio_streaming_state,
    moved_index: jax.Array,
    r_dn_carts_proposed: jax.Array,
) -> jax.Array:
    """Build the proposed geminal column ``G_new[:, k]`` of a dn-move directly from cached state (issue #134).

    Same contraction as :func:`_compute_u_dn_move_from_det_ratio_state` with the
    new orbital column in place of the difference; see
    :func:`_compute_up_row_from_det_ratio_state`.

    Returns:
        ``G_new[:, k]`` with shape ``(N_up,)``.
    """
    r_new = jnp.expand_dims(r_dn_carts_proposed[moved_index], axis=0)
    ao_dn_new_2d = geminal_data.compute_orb_api(geminal_data.orb_data_dn_spin, r_new)
    ao_dn_new = cast_to_zone(ao_dn_new_2d[:, 0], "det_ratio", geminal_data._determinant_dtype_jnp)  # (n_ao_dn,)
    paired_up_lambda = cast_to_zone(state.paired_up_lambda, "det_ratio", geminal_data._determinant_dtype_jnp)
    return jnp.dot(paired_up_lambda, ao_dn_new)


# ---------------------------------------------------------------------------
# Grads & Laplacian of ln|Det| -- test-only oracles: _*_fast_debug (fast path
# vs direct recompute) and _*_auto (jax.grad reference).
# ---------------------------------------------------------------------------


def _compute_grads_and_laplacian_ln_Det_fast_debug(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Debug helper that uses auto-diff to validate the fast path."""
    # Pure delegation to ``_compute_grads_and_laplacian_ln_Det_auto``;
    # no cast needed at this wrapper.
    # Use auto-diff as the reference (independent implementation).
    grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn = _compute_grads_and_laplacian_ln_Det_auto(
        geminal_data=geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    return grad_ln_D_up, grad_ln_D_dn, lap_ln_D_up, lap_ln_D_dn


@jit
def _compute_grads_and_laplacian_ln_Det_auto(
    geminal_data: Geminal_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Auto-diff version of grads and laplacian of ln Det.

    Uses autodiff on ln|det(G)| to compute gradients w.r.t. electron positions
    and per-electron Laplacians.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). Cast
    # to the det_grad_lap zone at the use site before passing as the
    # differentiation operand to grad/jacfwd (Principle 3b).
    r_up_z = cast_to_zone(r_up_carts, "det_grad_lap", geminal_data._determinant_dtype_jnp)
    r_dn_z = cast_to_zone(r_dn_carts, "det_grad_lap", geminal_data._determinant_dtype_jnp)

    def ln_det_fn(r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
        return compute_ln_det_geminal_all_elements(geminal_data, r_up, r_dn)

    grad_ln_D_up = jax.grad(ln_det_fn, argnums=0)(r_up_z, r_dn_z)
    grad_ln_D_dn = jax.grad(ln_det_fn, argnums=1)(r_up_z, r_dn_z)

    def grad_up_fn(r_up: jax.Array) -> jax.Array:
        return jax.grad(ln_det_fn, argnums=0)(r_up, r_dn_z)

    def grad_dn_fn(r_dn: jax.Array) -> jax.Array:
        return jax.grad(ln_det_fn, argnums=1)(r_up_z, r_dn)

    jac_up = jax.jacfwd(grad_up_fn)(r_up_z)
    jac_dn = jax.jacfwd(grad_dn_fn)(r_dn_z)

    laplacian_ln_D_up = jnp.einsum("ijij->i", jac_up)
    laplacian_ln_D_dn = jnp.einsum("ijij->i", jac_dn)

    return grad_ln_D_up, grad_ln_D_dn, laplacian_ln_D_up, laplacian_ln_D_dn
