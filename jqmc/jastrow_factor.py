"""Jastrow module.

Precision Zones:
    - ``jastrow_eval``: forward Jastrow evaluation (compute_Jastrow_part, J1/J2/J3/DTN).
    - ``jastrow_grad_lap``: Jastrow derivatives (compute_grads_and_laplacian_Jastrow_*).
    - ``jastrow_grad_lap_carry``: the carried DTN gradient/Laplacian state of the
      GFMC/LRDMC projection.
    - ``jastrow_ratio``: Jastrow ratio and update functions.

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
import itertools
from collections.abc import Callable, Sequence
from functools import partial

# set logger
from logging import getLogger

# jqmc module
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, TypeAlias

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import grad, hessian, jit, lax, vmap
from jax import typing as jnpt

from ._precision import get_dtype_jnp, get_dtype_np
from ._route_probe import note_route
from ._setting import (
    ATOL_J3_COUPLING_SYMMETRY,
    JASTROW_DTN_ECP_GRID_BUDGET_BYTES,
    JASTROW_DTN_PAIR_BUDGET_BYTES,
    RTOL_J3_COUPLING_SYMMETRY,
    EPS_safe_distance,
    measured_stream_batch,
)
from .atomic_orbital import (
    AOs_base_data,
    AOs_bspline_complex_data,
    AOs_cart_gto_data,
    AOs_cart_pgto_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_complex_data,
    _aos_cart_to_sphe,
    _aos_sphe_to_cart,
    _apply_strain_to_aos,
    _compute_AOs_debug,
    check_ao_eval_value_range,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
)
from .molecular_orbital import (
    MOs_data,
    _compute_MOs_debug,
    compute_MOs,
    compute_MOs_grad,
    compute_MOs_laplacian,
    compute_MOs_value_grad_lap,
)
from .structure import (
    Structure_data,
    _check_minimum_image_box_size,
    _inv_3x3,
    _minimum_image_reduction_plan,
    _minimum_image_shell_half_widths,
    _minimum_image_shell_shifts,
)

if TYPE_CHECKING:  # typing-only import to avoid circular dependency
    from .wavefunction import VariationalParameterBlock

# set logger
logger = getLogger("jqmc").getChild(__name__)

# Storage dtype of every Jastrow variational parameter leaf (J1 ``jastrow_1b_param``,
# J2 ``jastrow_2b_param``, J3 ``j_matrix``, DTN ``alpha|beta|gamma_indep``) at
# construction, basis conversion, optimizer update and checkpoint load. Float64 by
# construction in both modes, like the on-disk data it is loaded from and saved to
# (``_precision.py``, "Basis-data storage accessors"): the evaluation kernels cast it to
# their own zone at the use site (Principle 3b), whereas storing it in the
# ``jastrow_eval`` dtype rounded the parameters for every zone that consumes them,
# including the float64 ``jastrow_ratio`` / ``jastrow_grad_lap_carry`` zones, and
# quantized optimizer steps below one float32 ulp away (issue #781).
_JASTROW_PARAMETER_STORAGE_DTYPE = np.float64


# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

_NestedFloatList: TypeAlias = float | list["_NestedFloatList"]
_StaticFloatTuple: TypeAlias = float | tuple["_StaticFloatTuple", ...]
_PairGradLap: TypeAlias = tuple[jax.Array, jax.Array]
_GradLapResult: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_DtnValueGradLap: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
_DtnLeg: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
_FiniteDifferenceEval: TypeAlias = Callable[[np.ndarray], jax.Array]
_OrbitalApis: TypeAlias = tuple[
    Callable[..., jax.Array],
    Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    Callable[..., jax.Array],
    Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]],
]


@jax.custom_jvp
def _periodic_sincos_residual(theta: jax.Array) -> jax.Array:
    """Evaluate sin(theta)*cos(theta)-theta without losing its cubic term."""
    square = theta * theta
    series = theta * square * (-2 / 3 + square * (2 / 15 + square * (-4 / 315 + square * (2 / 2835))))
    # The degree-nine remainder is below roundoff relative to the cubic term.
    # This is an arithmetic series selector, not a comparison tolerance.
    cutoff = jnp.finfo(theta.dtype).eps ** 0.25
    return jnp.where(square < cutoff, series, jnp.sin(theta) * jnp.cos(theta) - theta)


@_periodic_sincos_residual.defjvp
def _periodic_sincos_residual_jvp(primals, tangents):
    """Keep the exact mathematical tangent instead of subtracting near-unit terms."""
    (theta,), (dtheta,) = primals, tangents
    return _periodic_sincos_residual(theta), -2 * jnp.sin(theta) ** 2 * dtheta


@jax.custom_jvp
def _periodic_safe_sqrt(square: jax.Array) -> jax.Array:
    """Square root with the established zero tangent convention at coincidence."""
    return jnp.sqrt(square)


@_periodic_safe_sqrt.defjvp
def _periodic_safe_sqrt_jvp(primals, tangents):
    """A masked zero cannot leak an infinite square-root tangent into outer AD."""
    (square,), (dsquare,) = primals, tangents
    positive = square > 0
    safe = jnp.where(positive, square, jnp.ones_like(square))
    tangent = jnp.where(positive, 0.5 * dsquare / jnp.sqrt(safe), jnp.zeros_like(dsquare))
    return _periodic_safe_sqrt(square), tangent


def _periodic_geometry_components(x: jax.Array, cell: jax.Array, inv: jax.Array):
    """Positive full-Gram distance square, half-gradient and half-Laplacian.

    All arithmetic stays in the caller's dtype. The Cartesian leading term
    is explicit to retain near-cusp cell derivatives. Scalar component
    contractions allow value-only callers to discard derivative work and
    derivative consumers to share geometry through a single reduction.
    """
    fractional = tuple(sum(x[..., k] * inv[k, i] for k in range(3)) for i in range(3))
    images = tuple(jnp.rint(value) for value in fractional)
    local = tuple(x[..., k] - sum(images[i] * cell[i, k] for i in range(3)) for k in range(3))
    theta = tuple(jnp.pi * sum(local[k] * inv[k, i] for k in range(3)) for i in range(3))
    sine, cosine = tuple(jnp.sin(value) for value in theta), tuple(jnp.cos(value) for value in theta)
    z = tuple(value * value for value in sine)
    h = tuple(s * c for s, c in zip(sine, cosine, strict=True))
    correction = tuple(_periodic_sincos_residual(value) / jnp.pi for value in theta)
    gram = jnp.sum(cell[:, None, :] * cell[None, :, :], axis=-1)
    inverse_gram = jnp.sum(inv[:, :, None] * inv[:, None, :], axis=0)
    diagonal = tuple(gram[i, i] for i in range(3))
    inverse_diagonal = tuple(inverse_gram[i, i] for i in range(3))
    off_h = tuple(sum(gram[i, j] * h[j] for j in range(3) if i != j) for i in range(3))
    cartesian = tuple(local[k] + sum(correction[i] * cell[i, k] for i in range(3)) for k in range(3))
    square = sum(value * value for value in cartesian)
    square += sum(diagonal[i] * z[i] * z[i] for i in range(3)) / jnp.pi**2
    numerator = tuple(cartesian[k] - (2 / jnp.pi) * sum(z[i] * off_h[i] * inv[k, i] for i in range(3)) for k in range(3))
    trace = 3 + sum((2 * diagonal[i] * inverse_diagonal[i] - 4) * z[i] for i in range(3))
    trace += 8 * sum(gram[i, j] * inverse_gram[i, j] * z[i] * z[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    trace -= 4 * sum(
        gram[i, j] * (inverse_diagonal[i] + inverse_diagonal[j]) * h[i] * h[j] for i, j in ((0, 1), (0, 2), (1, 2))
    )
    return square, numerator, trace


@jit
def _smooth_periodic_distance(
    r_d_cart: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
) -> jax.Array:
    r"""Positive full-Gram periodic distance for conventional J1/J2.

    With row-lattice C, fractional s=x inv(C), h=sin(pi*s) cos(pi*s),
    and z=sin(pi*s)^2, the adopted law is
    d^2=(||h C||^2 + sum_i ||C_i||^2 z_i^2)/pi^2.
    It has a Euclidean cusp and agrees with the per-axis sine distance
    in orthogonal cells. Its orthogonal-cell shear response is NOT the
    response of a diagonal-only extension.

    Evaluation uses a local image solely to expose the Cartesian leading
    term; caller positions are not wrapped or rebound. This is a batched
    (...,3) helper and respects the dtype selected by its caller. Value
    consumers reconstruct geometry in fp64 before their zone cast;
    ratio consumers retain their own explicitly cast geometry.

    The distance is not Cartesian-differentiable at coincidence. Its
    zero tangent there is a numerical AD convention, not a physical
    derivative or a claim that the distance is C-infinity on the torus.
    """
    d2, _, _ = _periodic_geometry_components(r_d_cart, cell, inv_cell)
    # A masked zero must not leak an infinite sqrt tangent through outer AD.
    # This convention is applied before any caller-specific distance clamp.
    return _periodic_safe_sqrt(d2)


def _smooth_periodic_distance_grad_lap(
    r_d_cart: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
    eps: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Closed Cartesian derivatives of the positive full-Gram periodic law.

    Return (d, grad, |grad|^2, lap) in the jastrow_grad_lap zone, with
    the existing EPS distance clamp. For squared distance S, numerator
    m=grad(S)/2 and trace T=lap(S)/2, grad(d)=m/d and
    lap(d)=(T-|grad(d)|^2)/d. Their stabilized expressions retain the
    Euclidean leading term without subtracting large Gram constants.

    The helper is batched over leading displacement axes. A zero-safe
    sqrt tangent prevents masked periodic coincidences from poisoning
    outer AD; callers retain their existing self-pair masks and EPS
    policies. No autodiff is used to calculate these derivatives.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    x = jnp.asarray(r_d_cart, dtype=dtype_jnp)
    cell_j = jnp.asarray(cell, dtype=dtype_jnp)
    inv_cell_j = jnp.asarray(inv_cell, dtype=dtype_jnp)
    square, numerator, trace = _periodic_geometry_components(x, cell_j, inv_cell_j)
    d = jnp.maximum(_periodic_safe_sqrt(square), eps)
    grad = jnp.stack(numerator, axis=-1) / d[..., None]
    q = jnp.sum(grad * grad, axis=-1)
    lap = (trace - q) / d
    return d, grad, q, lap


def _smooth_periodic_distance_debug(
    r_d_cart: npt.ArrayLike,
    cell: npt.ArrayLike,
    inv_cell: npt.ArrayLike,
) -> float:
    """fp64 numpy reference for ``_smooth_periodic_distance``.

    Helper-free, literal transcription of the formula, used to
    cross-check the production (JIT/jax) implementation.
    """
    r_d_cart = np.asarray(r_d_cart, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    inv_cell = np.asarray(inv_cell, dtype=np.float64)
    r_frac = r_d_cart @ inv_cell
    phase = np.pi * r_frac
    vector = np.sin(2 * phase) @ cell
    height = 2 * np.sin(phase) ** 2
    square = np.dot(vector, vector) + np.dot(np.sum(cell * cell, axis=1), height * height)
    return float(np.sqrt(square) / (2 * np.pi))


@struct.dataclass
class Jastrow_one_body_data:
    r"""One-body Jastrow parameters and structure metadata.

    The one-body term models electron-nucleus correlations.  Two functional
    forms are available, selected by ``jastrow_1b_type``:

    * ``'exp'`` (default) -- exponential form:

      .. math::

         f(r_{eN}) = -A \, \frac{1}{2a} \bigl(1 - e^{-a\,c\,r_{eN}}\bigr)

    * ``'pade'`` -- Pade form:

      .. math::

         f(r_{eN}) = -A \, \frac{c\,r_{eN}}{2\,(1 + a\,c\,r_{eN})}

    where :math:`A = (2 Z_{\text{eff}})^{3/4}`, :math:`c = (2 Z_{\text{eff}})^{1/4}`,
    and :math:`a` is ``jastrow_1b_param``.

    The numerical value is returned without the ``exp`` wrapper; callers
    attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_1b_param (float): Parameter *a* controlling the one-body decay.
        jastrow_1b_type (str): Functional form -- ``'exp'`` or ``'pade'``.
            Stored as a compile-time constant (``pytree_node=False``).
        structure_data (Structure_data): Nuclear positions and charges.
        core_electrons (tuple[float]): Removed core electrons per nucleus (for ECPs).
    """

    jastrow_1b_param: float = struct.field(pytree_node=True, default=1.0)  #: One-body Jastrow exponent parameter.
    jastrow_1b_type: str = struct.field(pytree_node=False, default="exp")  #: Functional form: ``'exp'`` or ``'pade'``.
    structure_data: Structure_data = struct.field(
        pytree_node=True, default_factory=Structure_data
    )  #: Nuclear structure data providing positions and atomic numbers.
    core_electrons: list[float] | tuple[float] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Effective core-electron counts aligned with ``structure_data``.

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if not np.isfinite(self.jastrow_1b_param):
            raise ValueError(f"jastrow_1b_param = {self.jastrow_1b_param} must be finite.")
        if self.jastrow_1b_param < 0.0:
            raise ValueError(f"jastrow_1b_param = {self.jastrow_1b_param} must be non-negative.")
        if self.jastrow_1b_type not in ("exp", "pade"):
            raise ValueError(f"jastrow_1b_type = '{self.jastrow_1b_type}' must be 'exp' or 'pade'.")
        if len(self.core_electrons) != len(self.structure_data.positions):
            raise ValueError(
                f"len(core_electrons) = {len(self.core_electrons)} must be the same as len(structure_data.positions) = {len(self.structure_data.positions)}."
            )
        if not isinstance(self.core_electrons, (list, tuple)):
            raise ValueError(f"core_electrons = {type(self.core_electrons)} must be a list or tuple.")
        self.structure_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 1b param = {self.jastrow_1b_param}")
        info_lines.append(f"  1b Jastrow functional form is the {self.jastrow_1b_type} type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_one_body_data(
        cls,
        jastrow_1b_param: npt.ArrayLike,
        structure_data: Structure_data,
        core_electrons: Sequence[float],
        jastrow_1b_type: Literal["exp", "pade"] = "exp",
    ) -> Self:
        """Initialization."""
        jastrow_one_body_data = cls(
            jastrow_1b_param=np.asarray(jastrow_1b_param, dtype=_JASTROW_PARAMETER_STORAGE_DTYPE).reshape(()),
            jastrow_1b_type=jastrow_1b_type,
            structure_data=structure_data,
            core_electrons=core_electrons,
        )
        return jastrow_one_body_data

    @classmethod
    def build(
        cls,
        *,
        jastrow_1b_param: npt.ArrayLike,
        structure_data: Structure_data,
        core_electrons: Sequence[float],
        jastrow_1b_type: Literal["exp", "pade"] = "exp",
    ) -> Self:
        """Build a validated :class:`Jastrow_one_body_data`.

        Blessed factory: delegates to
        :py:meth:`init_jastrow_one_body_data` (which normalizes the scalar
        parameter) and then runs :py:meth:`sanity_check`. Direct
        construction remains supported.
        """
        data = cls.init_jastrow_one_body_data(jastrow_1b_param, structure_data, core_electrons, jastrow_1b_type)
        data.sanity_check()
        return data


@partial(jit, static_argnames=("zone",))
def compute_Jastrow_one_body(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    zone: str = "jastrow_eval",
) -> jax.Array:
    r"""Evaluate the one-body Jastrow :math:`J_1` (without ``exp``) for given coordinates.

    This routine returns the scalar ``J`` value; callers attach ``exp(J)`` to
    the wavefunction.

    Args:
        jastrow_one_body_data: One-body Jastrow parameters and structure data.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.
        zone: Precision zone of the evaluation (default ``"jastrow_eval"``, the
            zone of whole Jastrow values). The whole-value ratio routes pass
            ``"jastrow_ratio"`` (see :func:`compute_Jastrow_part_ratio_zone`).

    Returns:
        float: One-body Jastrow value (before exponentiation).
    """
    # NOTE: do not pre-cast r_*_carts. ``one_body_jastrow_kernel`` reconstructs
    # ``r - R`` in float64 internally to avoid catastrophic cancellation;
    # a wrapper-level downcast would defeat that guard.
    dtype_jnp = get_dtype_jnp(zone)
    # Retrieve structure data and convert to JAX arrays
    sd = jastrow_one_body_data.structure_data
    # Keep both operands in their geometry precision until r - R is formed.
    R_carts = sd._positions_cart_jnp
    atomic_numbers = jnp.array(sd.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.array(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    effective_charges = atomic_numbers - core_electrons
    j1b = jnp.asarray(jastrow_one_body_data.jastrow_1b_param, dtype=dtype_jnp)

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    # PBC dispatch: swap the electron--nucleus distance for the
    # Williamson smooth periodic distance. ``sd.pbc_flag`` is
    # ``pytree_node=False`` so this resolves at trace time -- the molecular
    # branch is unaffected.
    if sd.pbc_flag:
        cell_jnp = sd._cell_jnp.astype(jnp.float64)
        inv_cell_jnp = _inv_3x3(cell_jnp)

        # Wrap-policy compliance: ``_smooth_periodic_distance`` already depends
        # only on the fractional part of ``r_cart - R_cart`` (``sin(pi * frac)``
        # has period 1), so it is translation-invariant by construction. No
        # wrap of the input positions is needed or permitted here -- the wrap
        # policy reserves walker-position wrapping for the AO kernels (paired
        # with ``wrap_restore``) and the deliberate rebuild hook.
        def _eN_distance(r_cart: jax.Array, R_cart: jax.Array) -> jax.Array:
            return _smooth_periodic_distance(
                r_cart - R_cart,
                cell_jnp,
                inv_cell_jnp,
            ).astype(dtype_jnp)
    else:

        def _eN_distance(r_cart: jax.Array, R_cart: jax.Array) -> jax.Array:
            # Reconstruct r - R in caller-supplied precision (fp64 from MCMC
            # walker state) via JAX promotion, then downcast to jastrow_eval.
            diff = (r_cart - R_cart).astype(dtype_jnp)
            if dtype_jnp == jnp.float32:
                # Avoid batch-dependent dot lowering for this short norm.
                return jnp.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            return jnp.linalg.norm(diff)

    if j1b_type == "exp":

        def one_body_jastrow_kernel(
            param: float,
            coeff: float,
            r_cart: jnpt.ArrayLike,
            R_cart: jnpt.ArrayLike,
        ) -> jax.Array:
            """Exponential form of J1."""
            r_eN = _eN_distance(r_cart, R_cart)
            # ``-expm1(-x)`` == ``1 - exp(-x)`` without the small-r subtraction
            # cancellation (fp32 ``1 - exp(-x)`` is exactly 0 at r = 2**-26).
            return 1.0 / (2.0 * param) * (-jnp.expm1(-param * coeff * r_eN))

        def atom_contrib(r_cart: jax.Array, R_cart: jax.Array, Z_eff: jax.Array) -> jax.Array:
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            return -((2.0 * Z_eff) ** (3.0 / 4.0)) * one_body_jastrow_kernel(j1b, coeff, r_cart, R_cart)

    elif j1b_type == "pade":

        def atom_contrib(r_cart: jax.Array, R_cart: jax.Array, Z_eff: jax.Array) -> jax.Array:
            """Cusp-normalized Pade J1: -Z_eff*r/(1 + a*(2*Z_eff)^{1/4}*r)."""
            r_eN = _eN_distance(r_cart, R_cart)
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            return -Z_eff * r_eN / (1.0 + j1b * coeff * r_eN)

    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    def _sum_contributions(values):
        # Keep full, periodic and long reductions on their existing path.
        # Expanding long trees regresses large molecular batches on XLA CPU;
        # only short mixed molecular sums need a fixed scalar/vmap ordering.
        if dtype_jnp != jnp.float32 or sd.pbc_flag or values.shape[0] > 8:
            return jnp.sum(values)
        if values.shape[0] == 0:
            return jnp.zeros((), dtype=values.dtype)
        while values.shape[0] > 1:
            if values.shape[0] % 2:
                values = jnp.concatenate((values, jnp.zeros_like(values[:1])))
            values = values[::2] + values[1::2]
        return values[0]

    # Sum the contributions from all atoms for a single electron
    def electron_contrib(
        r_cart: jax.Array,
        R_carts: jax.Array,
        effective_charges: jax.Array,
    ) -> jax.Array:
        # Apply vmap over positions and effective_charges
        return _sum_contributions(jax.vmap(atom_contrib, in_axes=(None, 0, 0))(r_cart, R_carts, effective_charges))

    if sd.pbc_flag:
        # J1 is spin independent. One value batch avoids duplicate geometry
        # preparation. Atom-major pairs make electron coordinates contiguous
        # for the trigonometric kernels; the radial law is unchanged.
        electrons = jnp.concatenate((jnp.asarray(r_up_carts), jnp.asarray(r_dn_carts)), axis=0)
        pairs = jax.vmap(jax.vmap(atom_contrib, in_axes=(0, None, None)), in_axes=(None, 0, 0))(
            electrons, R_carts, effective_charges
        )
        return jnp.sum(pairs)

    # Sum contributions for all spin-up electrons
    J1_up = _sum_contributions(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_up_carts, R_carts, effective_charges))
    # Sum contributions for all spin-down electrons
    J1_dn = _sum_contributions(jax.vmap(electron_contrib, in_axes=(0, None, None))(r_dn_carts, R_carts, effective_charges))

    return J1_up + J1_dn


def _compute_Jastrow_one_body_molecular_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Independent fp64 NumPy reference for molecular J1."""
    dtype_np = np.float64
    positions = jastrow_one_body_data.structure_data.positions
    atomic_numbers = jastrow_one_body_data.structure_data.atomic_numbers
    core_electrons = jastrow_one_body_data.core_electrons
    effective_charges = np.array(atomic_numbers, dtype=dtype_np) - np.array(core_electrons, dtype=dtype_np)

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    def one_body_jastrow_exp(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Exponential form of J1."""
        return 1.0 / (2.0 * param) * (1.0 - np.exp(-param * coeff * np.linalg.norm(r_cart - R_cart)))

    def one_body_jastrow_pade(
        param: float, coeff: float, r_cart: npt.NDArray[np.float64], R_cart: npt.NDArray[np.float64]
    ) -> float:
        """Pade form of J1."""
        r_eN = np.linalg.norm(r_cart - R_cart)
        return coeff * r_eN / (2.0 * (1.0 + param * coeff * r_eN))

    if j1b_type == "exp":
        _j1_kernel = one_body_jastrow_exp
    elif j1b_type == "pade":
        _j1_kernel = one_body_jastrow_pade
    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    J1_up = 0.0
    for r_up in r_up_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_up += -((2.0 * Z_eff) ** (3.0 / 4.0)) * _j1_kernel(jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart)

    J1_dn = 0.0
    for r_up in r_dn_carts:
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            J1_dn += -((2.0 * Z_eff) ** (3.0 / 4.0)) * _j1_kernel(jastrow_one_body_data.jastrow_1b_param, coeff, r_up, R_cart)

    J1 = J1_up + J1_dn

    return J1


def _compute_Jastrow_one_body_pbc_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """fp64 numpy reference for the PBC branch of ``compute_Jastrow_one_body``.

    Helper-free reimplementation: the positive full-Gram
    periodic distance is inlined per electron--nucleus pair (no call to
    ``_smooth_periodic_distance*``) so the production-vs-debug
    cross-check validates the entire helper chain in one comparison.
    """
    dtype_np = np.float64
    sd = jastrow_one_body_data.structure_data
    positions = np.asarray(sd.positions, dtype=np.float64)
    atomic_numbers = sd.atomic_numbers
    core_electrons = jastrow_one_body_data.core_electrons
    effective_charges = np.array(atomic_numbers, dtype=dtype_np) - np.array(core_electrons, dtype=dtype_np)
    cell = np.asarray(sd.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    a = float(jastrow_one_body_data.jastrow_1b_param)
    j1b_type = jastrow_one_body_data.jastrow_1b_type

    J1 = 0.0
    for r_cart in np.concatenate([np.asarray(r_up_carts), np.asarray(r_dn_carts)], axis=0):
        for R_cart, Z_eff in zip(positions, effective_charges, strict=True):
            r_frac = (np.asarray(r_cart, dtype=np.float64) - R_cart) @ inv_cell
            phase = np.pi * r_frac
            vector = np.sin(2 * phase) @ cell
            height = 2 * np.sin(phase) ** 2
            square = np.dot(vector, vector) + np.dot(np.sum(cell * cell, axis=1), height * height)
            r_eN = float(np.sqrt(square) / (2 * np.pi))
            coeff = (2.0 * Z_eff) ** (1.0 / 4.0)
            prefactor = -((2.0 * Z_eff) ** (3.0 / 4.0))
            if j1b_type == "exp":
                J1 += prefactor / (2.0 * a) * (1.0 - np.exp(-a * coeff * r_eN))
            elif j1b_type == "pade":
                J1 += prefactor * coeff * r_eN / (2.0 * (1.0 + a * coeff * r_eN))
            else:
                raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")
    return float(J1)


def _compute_grads_and_laplacian_Jastrow_one_body_debug(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Numerical gradients and Laplacian for one-body Jastrow (debug)."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5
    r_up_carts = np.array(r_up_carts, dtype=dtype_np)
    r_dn_carts = np.array(r_dn_carts, dtype=dtype_np)

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J_p_x_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_p_x_r_up_carts, r_dn_carts)
        J_p_y_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_p_y_r_up_carts, r_dn_carts)
        J_p_z_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_p_z_r_up_carts, r_dn_carts)

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J_m_x_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_m_x_r_up_carts, r_dn_carts)
        J_m_y_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_m_y_r_up_carts, r_dn_carts)
        J_m_z_up = compute_Jastrow_one_body(jastrow_one_body_data, diff_m_z_r_up_carts, r_dn_carts)

        grad_x_up.append((J_p_x_up - J_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J_p_y_up - J_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J_p_z_up - J_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J_p_x_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_p_x_r_dn_carts)
        J_p_y_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_p_y_r_dn_carts)
        J_p_z_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_p_z_r_dn_carts)

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J_m_x_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_m_x_r_dn_carts)
        J_m_y_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_m_y_r_dn_carts)
        J_m_z_dn = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, diff_m_z_r_dn_carts)

        grad_x_dn.append((J_p_x_dn - J_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J_p_y_dn - J_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J_p_z_dn - J_m_z_dn) / (2.0 * diff_h))

    grad_J1_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J1_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian (4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    diff_h2 = 1.0e-3
    J_ref = compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn_carts)

    def _eval_up(r_up: np.ndarray) -> jax.Array:
        return compute_Jastrow_one_body(jastrow_one_body_data, r_up, r_dn_carts)

    def _eval_dn(r_dn: np.ndarray) -> jax.Array:
        return compute_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn)

    def _fd4_second_deriv(
        eval_fn: _FiniteDifferenceEval,
        r_carts: np.ndarray,
        r_i: int,
        dim: int,
        h: float,
        f0: jax.Array,
    ) -> jax.Array:
        r_p1 = r_carts.copy()
        r_p2 = r_carts.copy()
        r_m1 = r_carts.copy()
        r_m2 = r_carts.copy()
        r_p1[r_i][dim] += h
        r_p2[r_i][dim] += 2 * h
        r_m1[r_i][dim] -= h
        r_m2[r_i][dim] -= 2 * h
        return (-eval_fn(r_p2) + 16 * eval_fn(r_p1) - 30 * f0 + 16 * eval_fn(r_m1) - eval_fn(r_m2)) / (12 * h**2)

    lap_J1_up = np.zeros(len(r_up_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_up_carts):
        lap_J1_up[r_i] = (
            _fd4_second_deriv(_eval_up, r_up_carts, r_i, 0, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 1, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 2, diff_h2, J_ref)
        )

    lap_J1_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_dn_carts):
        lap_J1_dn[r_i] = (
            _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 0, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 1, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 2, diff_h2, J_ref)
        )

    return grad_J1_up, grad_J1_dn, lap_J1_up, lap_J1_dn


@jit
def _compute_grads_and_laplacian_Jastrow_one_body_auto(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Auto-diff gradients and Laplacian for one-body Jastrow."""
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts = jnp.asarray(r_up_carts)
    r_dn_carts = jnp.asarray(r_dn_carts)

    grad_J1_up = grad(compute_Jastrow_one_body, argnums=1)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    grad_J1_dn = grad(compute_Jastrow_one_body, argnums=2)(jastrow_one_body_data, r_up_carts, r_dn_carts)

    hessian_J1_up = hessian(compute_Jastrow_one_body, argnums=1)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    laplacian_J1_up = jnp.einsum("ijij->i", hessian_J1_up)

    hessian_J1_dn = hessian(compute_Jastrow_one_body, argnums=2)(jastrow_one_body_data, r_up_carts, r_dn_carts)
    laplacian_J1_dn = jnp.einsum("ijij->i", hessian_J1_dn)

    return tuple(value.astype(dtype_jnp) for value in (grad_J1_up, grad_J1_dn, laplacian_J1_up, laplacian_J1_dn))


def _compute_grads_and_laplacian_Jastrow_one_body_pbc(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Closed-form PBC J1 grad/lap on the Williamson smooth periodic metric.

    Chain rule on the smooth periodic distance ``d(r_i - R_I)``::

        nabla_i   J1 = sum_I f'(d) (nabla d)
        nabla_i^2 J1 = sum_I ( f''(d) |nabla d|^2 + f'(d) nabla^2 d )

    with ``(d, nabla d, |nabla d|^2, nabla^2 d)`` from
    :func:`_smooth_periodic_distance_grad_lap` (closed form, no autodiff) and
    the same kernel derivatives ``f'`` / ``f''`` as the molecular branch of
    :func:`compute_grads_and_laplacian_Jastrow_one_body`. 1:1 with the Rust
    reference ``j1_grad_lap_one_spin`` + ``smooth_dist``
    (``jqmc/rs/src/jastrow_factor.rs``); validated against the retained
    ``_compute_grads_and_laplacian_Jastrow_one_body_auto`` oracle in
    ``tests/test_jqmc_kernel/test_jastrow.py``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    sd = jastrow_one_body_data.structure_data
    positions = sd._positions_cart_jnp
    cell = sd._cell_jnp.astype(jnp.float64)
    inv_cell = _inv_3x3(cell)
    atomic_numbers = jnp.asarray(sd.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.asarray(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    z_eff = atomic_numbers - core_electrons

    a = jastrow_one_body_data.jastrow_1b_param
    c = (2.0 * z_eff) ** (1.0 / 4.0)
    # Effective numerator: Pade includes the distance scale c in 2*Z_eff.
    A = (2.0 * z_eff) ** (3.0 / 4.0) if jastrow_one_body_data.jastrow_1b_type == "exp" else 2.0 * z_eff
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)
    j1b_type = jastrow_one_body_data.jastrow_1b_type

    def _grad_lap_one_spin(r_carts: jax.Array) -> _PairGradLap:
        diff = r_carts[:, None, :] - positions[None, :, :]  # (n_e, n_nuc, 3)
        square, numerator, trace = _periodic_geometry_components(
            diff.astype(dtype_jnp), cell.astype(dtype_jnp), inv_cell.astype(dtype_jnp)
        )
        d = jnp.maximum(_periodic_safe_sqrt(square), eps)
        reciprocal = 1 / d
        gradient = tuple(component * reciprocal for component in numerator)
        q = sum(component * component for component in gradient)
        if j1b_type == "exp":
            # f(d) = -A/(2a) * (1 - exp(-a*c*d)); f' = -A*(c/2)*exp(-a*c*d);
            # f'' = A*(a*c^2/2)*exp(-a*c*d)  (identical to the molecular branch).
            exp_term = jnp.exp(-a * c[None, :] * d)
            fprime = -A[None, :] * (c[None, :] / 2.0) * exp_term
            scaled = fprime * reciprocal
            lap = scaled * (trace - (1 + a * c[None, :] * d) * q)
        elif j1b_type == "pade":
            # f(d) = -Z_eff*d/(1+a*c*d).
            # f' = -Z_eff/(1+a*c*d)^2; f'' = 2*Z_eff*a*c/(1+a*c*d)^3.
            denom = 1.0 + a * c[None, :] * d
            fprime = -A[None, :] / (2.0 * denom * denom)
            scaled = fprime * reciprocal
            lap = scaled * (trace - (3 - 2 / denom) * q)
        else:
            raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")
        # Reduce the four same-layout fields together. This shares all per-pair
        # geometry, without a heterogeneous component axis that prevents SIMD.
        operands = tuple(scaled * component for component in numerator) + (lap,)
        zeros = tuple(jnp.zeros((), value.dtype) for value in operands)

        def add_four(left, right):
            return tuple(left[i] + right[i] for i in range(4))

        reduced = lax.reduce(operands, zeros, add_four, dimensions=(1,))
        return jnp.stack(reduced[:3], axis=-1), reduced[3]

    grad_up, lap_up = _grad_lap_one_spin(jnp.asarray(r_up_carts))
    grad_dn, lap_dn = _grad_lap_one_spin(jnp.asarray(r_dn_carts))
    return grad_up, grad_dn, lap_up, lap_dn


@jit
def compute_grads_and_laplacian_Jastrow_one_body(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and per-electron Laplacians for the one-body Jastrow.

    Args:
        jastrow_one_body_data: One-body Jastrow parameters and structure data.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    # PBC dispatch: the analytic kernel below assumes the Cartesian e-N
    # distance ``jnp.linalg.norm(r_i - R_I)``. For periodic systems the same
    # chain rule runs on the Williamson smooth periodic metric via the
    # CLOSED-FORM ``_smooth_periodic_distance_grad_lap`` -- no autodiff
    # (production must not trace generic AD; an ``_auto`` dispatch here costs
    # a ~O(nw N_e N_atom) hessian tape: 6.87 GB at N_atom=108 /
    # N_e=432 / nw=8 measured on an A30). ``pbc_flag`` is
    # ``pytree_node=False`` so this resolves at trace time and the molecular
    # HLO is unaffected.
    if jastrow_one_body_data.structure_data.pbc_flag:
        return _compute_grads_and_laplacian_Jastrow_one_body_pbc(
            jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )

    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    positions = jastrow_one_body_data.structure_data._positions_cart_jnp
    atomic_numbers = jnp.asarray(jastrow_one_body_data.structure_data.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.asarray(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    z_eff = atomic_numbers - core_electrons

    a = jastrow_one_body_data.jastrow_1b_param
    c = (2.0 * z_eff) ** (1.0 / 4.0)
    # Preserve the derivative operation order; only the Pade coefficient changes.
    A = (2.0 * z_eff) ** (3.0 / 4.0) if jastrow_one_body_data.jastrow_1b_type == "exp" else 2.0 * z_eff

    eps = EPS_safe_distance

    j1b_type = jastrow_one_body_data.jastrow_1b_type

    if j1b_type == "exp":

        def _grad_lap_one_spin(r_carts: jax.Array) -> _PairGradLap:
            diff = (r_carts[:, None, :] - positions[None, :, :]).astype(dtype_jnp)
            r = jnp.linalg.norm(diff, axis=-1)
            r_safe = jnp.maximum(r, eps)
            exp_term = jnp.exp(-a * c[None, :] * r_safe)

            fprime = -A[None, :] * (c[None, :] / 2.0) * exp_term
            grad = jnp.sum((fprime[..., None] * diff) / r_safe[..., None], axis=1)

            fsecond = A[None, :] * (a * c[None, :] * c[None, :] / 2.0) * exp_term
            lap = fsecond - A[None, :] * c[None, :] * exp_term / r_safe
            lap_e = jnp.sum(lap, axis=1)
            return grad, lap_e

    elif j1b_type == "pade":

        def _grad_lap_one_spin(r_carts: jax.Array) -> _PairGradLap:
            # f(r) = -Z_eff*r/(1+a*c*r).
            # f'(r) = -Z_eff/(1+a*c*r)^2
            # f''(r) = 2*Z_eff*a*c/(1+a*c*r)^3
            # lap contribution per atom: f''(r) + 2*f'(r)/r
            diff = (r_carts[:, None, :] - positions[None, :, :]).astype(dtype_jnp)  # (n_e, n_nuc, 3)
            r = jnp.linalg.norm(diff, axis=-1)  # (n_e, n_nuc)
            r_safe = jnp.maximum(r, eps)
            denom = 1.0 + a * c[None, :] * r_safe  # (n_e, n_nuc)

            fprime = -A[None, :] / (2.0 * denom * denom)  # f'(r)
            grad = jnp.sum((fprime[..., None] * diff) / r_safe[..., None], axis=1)  # (n_e, 3)

            fsecond = A[None, :] * a * c[None, :] / (denom * denom * denom)  # f''(r)
            lap = fsecond + 2.0 * fprime / r_safe
            lap_e = jnp.sum(lap, axis=1)
            return grad, lap_e

    else:
        raise ValueError(f"Unknown jastrow_1b_type: {j1b_type}")

    grad_up, lap_up = _grad_lap_one_spin(jnp.asarray(r_up_carts))
    grad_dn, lap_dn = _grad_lap_one_spin(jnp.asarray(r_dn_carts))

    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# J1 streaming state.
#
# J1 is a per-electron sum over atoms with no inter-electron coupling, so a
# single-electron move only affects one row of the per-electron grad/lap.
# The state simply caches those rows; advance recomputes the moved row in
# ``O(n_atom)`` rather than the fresh O(N_e * n_atom).
# ---------------------------------------------------------------------------


@struct.dataclass
class Jastrow_one_body_streaming_state:
    """Cached per-electron J1 grad/lap consistent with current ``(r_up, r_dn)``."""

    grad_J1_up: jax.Array  # (N_up, 3)
    grad_J1_dn: jax.Array  # (N_dn, 3)
    lap_J1_up: jax.Array  # (N_up,)
    lap_J1_dn: jax.Array  # (N_dn,)


@jit
def _init_grads_laplacian_Jastrow_one_body_streaming_state(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_one_body_streaming_state:
    """One-shot init equivalent in cost to ``compute_grads_and_laplacian_Jastrow_one_body``."""
    g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_up_carts, r_dn_carts)
    return Jastrow_one_body_streaming_state(grad_J1_up=g_up, grad_J1_dn=g_dn, lap_J1_up=l_up, lap_J1_dn=l_dn)


@jit
def _advance_grads_laplacian_Jastrow_one_body_streaming_state(
    jastrow_one_body_data: Jastrow_one_body_data,
    state: Jastrow_one_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_one_body_streaming_state:
    """Advance J1 state after a single-electron move.

    Recomputes one electron's row by re-running the existing one-spin kernel
    on a single-row slice. Cost: ``O(n_atom)``.
    """
    num_up = state.grad_J1_up.shape[0]
    num_dn = state.grad_J1_dn.shape[0]

    def _branch_up(_: None) -> Jastrow_one_body_streaming_state:
        # Reuse the full-batch kernel on a length-1 slice -- gives one row that
        # we slot back into the cached state.
        r_slice = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        g_row, _, l_row, _ = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_slice, r_slice[:0])
        new_grad = state.grad_J1_up.at[moved_index].set(g_row[0])
        new_lap = state.lap_J1_up.at[moved_index].set(l_row[0])
        return state.replace(grad_J1_up=new_grad, lap_J1_up=new_lap)

    def _branch_dn(_: None) -> Jastrow_one_body_streaming_state:
        r_slice = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        # Pass empty up so only the dn branch contributes (J1 is per-spin
        # independent -- feeding empty up has zero effect on dn output).
        _, g_row, _, l_row = compute_grads_and_laplacian_Jastrow_one_body(jastrow_one_body_data, r_slice[:0], r_slice)
        new_grad = state.grad_J1_dn.at[moved_index].set(g_row[0])
        new_lap = state.lap_J1_dn.at[moved_index].set(l_row[0])
        return state.replace(grad_J1_dn=new_grad, lap_J1_dn=new_lap)

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


@struct.dataclass
class Jastrow_two_body_data:
    r"""Two-body Jastrow parameter container.

    The two-body term models electron-electron correlations.  Two functional
    forms are available, selected by ``jastrow_2b_type``:

    * ``'pade'`` (default) -- Pade form:

      .. math::

         f(r_{ee}) = \frac{r_{ee}}{2\,(1 + a\,r_{ee})}

    * ``'exp'`` -- exponential form:

      .. math::

         f(r_{ee}) = \frac{1}{2a}\bigl(1 - e^{-a\,r_{ee}}\bigr)

    where :math:`a` is ``jastrow_2b_param``.

    Values are returned without exponentiation; callers use ``exp(J)``
    when constructing the wavefunction.

    Args:
        jastrow_2b_param (float): Parameter *a* for the two-body Jastrow part.
        jastrow_2b_type (str): Functional form -- ``'pade'`` or ``'exp'``.
            Stored as a compile-time constant (``pytree_node=False``).
    """

    jastrow_2b_param: float = struct.field(pytree_node=True, default=1.0)  #: Two-body Jastrow parameter.
    jastrow_2b_type: str = struct.field(pytree_node=False, default="pade")  #: Functional form: ``'pade'`` or ``'exp'``.
    #: Nuclear / cell structure data. Needed for the PBC branch of
    #: ``compute_Jastrow_two_body``, which reads ``structure_data.pbc_flag``
    #: at trace time and routes to the Williamson smooth periodic distance
    #: when ``True``. Molecular instances keep the default
    #: ``Structure_data()`` (``pbc_flag=False``) and the field is unused.
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=Structure_data)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if not np.isfinite(self.jastrow_2b_param):
            raise ValueError(f"jastrow_2b_param = {self.jastrow_2b_param} must be finite.")
        if self.jastrow_2b_param < 0.0:
            raise ValueError(f"jastrow_2b_param = {self.jastrow_2b_param} must be non-negative.")
        if self.jastrow_2b_type not in ("exp", "pade"):
            raise ValueError(f"jastrow_2b_type = '{self.jastrow_2b_type}' must be 'exp' or 'pade'.")
        self.structure_data.sanity_check()

    def __setstate__(self, state: dict) -> None:
        # Backwards compat for pickle fixtures created before ``structure_data``
        # was added. ``pickle.load`` bypasses ``__init__`` and so does not
        # apply ``default_factory``; supply the molecular default when the
        # field is missing from the state dict.
        if "structure_data" not in state:
            state = dict(state)
            state["structure_data"] = Structure_data()
        self.__dict__.update(state)

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Jastrow 2b param = {self.jastrow_2b_param}")
        info_lines.append(f"  2b Jastrow functional form is the {self.jastrow_2b_type} type.")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def init_jastrow_two_body_data(
        cls,
        jastrow_2b_param: npt.ArrayLike = 1.0,
        jastrow_2b_type: Literal["exp", "pade"] = "pade",
        structure_data: Structure_data | None = None,
    ) -> Self:
        """Initialization.

        ``structure_data`` is optional for backward compatibility with the
        molecular-only call sites; defaults to ``Structure_data()`` (an
        empty ``pbc_flag=False`` instance), which is what the molecular
        path expects. A periodic Hamiltonian must pass its periodic
        ``Structure_data``: :py:meth:`Hamiltonian_data.sanity_check` rejects
        a J2 whose periodicity differs from the Hamiltonian structure.
        Only legacy checkpoints without the J2 structure
        subgroup are back-filled on load.
        """
        if structure_data is None:
            structure_data = Structure_data()
        jastrow_two_body_data = cls(
            jastrow_2b_param=np.asarray(jastrow_2b_param, dtype=_JASTROW_PARAMETER_STORAGE_DTYPE).reshape(()),
            jastrow_2b_type=jastrow_2b_type,
            structure_data=structure_data,
        )
        return jastrow_two_body_data

    @classmethod
    def build(
        cls,
        *,
        jastrow_2b_param: npt.ArrayLike = 1.0,
        jastrow_2b_type: Literal["exp", "pade"] = "pade",
        structure_data: Structure_data | None = None,
    ) -> Self:
        """Build a validated :class:`Jastrow_two_body_data`.

        Blessed factory: delegates to
        :py:meth:`init_jastrow_two_body_data` (which normalizes the scalar
        parameter and defaults ``structure_data``) and then runs
        :py:meth:`sanity_check`. Direct construction remains supported.
        """
        data = cls.init_jastrow_two_body_data(jastrow_2b_param, jastrow_2b_type, structure_data)
        data.sanity_check()
        return data


@partial(jit, static_argnames=("zone",))
def compute_Jastrow_two_body(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    zone: str = "jastrow_eval",
) -> jax.Array:
    r"""Evaluate the two-body Jastrow :math:`J_2` (Pade form) without exponentiation.

    This returns ``J`` and callers attach ``exp(J)`` to the wavefunction.

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.
        zone: Precision zone of the evaluation (default ``"jastrow_eval"``, the
            zone of whole Jastrow values). The whole-value ratio routes pass
            ``"jastrow_ratio"`` (see :func:`compute_Jastrow_part_ratio_zone`).

    Returns:
        float: Two-body Jastrow value (before exponentiation).
    """
    # NOTE: do not pre-cast r_*_carts. ``two_body_jastrow_pade``/``_exp``
    # reconstruct ``r_i - r_j`` in float64 internally to avoid catastrophic
    # cancellation; a wrapper-level downcast would defeat that guard.
    dtype_jnp = get_dtype_jnp(zone)
    j2b_param = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    j2b_type = jastrow_two_body_data.jastrow_2b_type
    sd = jastrow_two_body_data.structure_data

    # PBC dispatch: swap the electron--electron distance for the
    # Williamson smooth periodic distance. ``sd.pbc_flag`` is
    # ``pytree_node=False`` so this resolves at trace time -- the molecular
    # branch is unaffected.
    if sd.pbc_flag:
        cell_jnp = sd._cell_jnp.astype(jnp.float64)
        inv_cell_jnp = _inv_3x3(cell_jnp)

        # Wrap-policy compliance: ``_smooth_periodic_distance`` already depends
        # only on the fractional part of ``r_cart_i - r_cart_j``, so input
        # wrapping is a no-op. See the matching note in
        # ``compute_jastrow_one_body``.
        def _ee_distance(r_cart_i: jax.Array, r_cart_j: jax.Array) -> jax.Array:
            return _smooth_periodic_distance(
                r_cart_i - r_cart_j,
                cell_jnp,
                inv_cell_jnp,
            ).astype(dtype_jnp)
    else:

        def _ee_distance(r_cart_i: jax.Array, r_cart_j: jax.Array) -> jax.Array:
            # Reconstruct r_i - r_j in caller-supplied precision (fp64 from
            # MCMC walker state) via JAX promotion, then downcast to
            # jastrow_eval.
            diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
            return jnp.linalg.norm(diff)

    def two_body_jastrow_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> jax.Array:
        """Exponential form of J2."""
        r_ij = _ee_distance(r_cart_i, r_cart_j)
        # ``-expm1(-x)`` == ``1 - exp(-x)`` without the small-r cancellation.
        return 1.0 / (2.0 * param) * (-jnp.expm1(-param * r_ij))

    def two_body_jastrow_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> jax.Array:
        """Pade form of J2."""
        r_ij = _ee_distance(r_cart_i, r_cart_j)
        return r_ij / 2.0 * (1.0 + param * r_ij) ** (-1.0)

    if j2b_type == "pade":
        two_body_jastrow_anti_parallel = two_body_jastrow_pade
        two_body_jastrow_parallel = two_body_jastrow_pade
    elif j2b_type == "exp":
        two_body_jastrow_anti_parallel = two_body_jastrow_exp
        two_body_jastrow_parallel = two_body_jastrow_exp
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    vmap_two_body_jastrow_anti_parallel_spins = vmap(
        vmap(two_body_jastrow_anti_parallel, in_axes=(None, None, 0)), in_axes=(None, 0, None)
    )

    two_body_jastrow_anti_parallel_val = jnp.sum(vmap_two_body_jastrow_anti_parallel_spins(j2b_param, r_up_carts, r_dn_carts))

    def compute_parallel_sum(r_carts: jax.Array) -> jax.Array:
        num_particles = r_carts.shape[0]
        idx_i, idx_j = jnp.triu_indices(num_particles, k=1)
        r_i = r_carts[idx_i]
        r_j = r_carts[idx_j]
        vmap_two_body_jastrow_parallel_spins = vmap(two_body_jastrow_parallel, in_axes=(None, 0, 0))(j2b_param, r_i, r_j)
        return jnp.sum(vmap_two_body_jastrow_parallel_spins)

    two_body_jastrow_parallel_up = compute_parallel_sum(r_up_carts)
    two_body_jastrow_parallel_dn = compute_parallel_sum(r_dn_carts)

    two_body_jastrow = two_body_jastrow_anti_parallel_val + two_body_jastrow_parallel_up + two_body_jastrow_parallel_dn

    return two_body_jastrow


def _compute_Jastrow_two_body_molecular_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """See _api method."""
    j2b_type = jastrow_two_body_data.jastrow_2b_type

    def two_body_jastrow_exp(param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]) -> float:
        """Exponential form of J2."""
        return 1.0 / (2.0 * param) * (1.0 - np.exp(-param * np.linalg.norm(r_cart_i - r_cart_j)))

    def two_body_jastrow_pade(param: float, r_cart_i: npt.NDArray[np.float64], r_cart_j: npt.NDArray[np.float64]) -> float:
        """Pade form of J2."""
        return np.linalg.norm(r_cart_i - r_cart_j) / 2.0 * (1.0 + param * np.linalg.norm(r_cart_i - r_cart_j)) ** (-1.0)

    if j2b_type == "pade":
        _j2_anti = two_body_jastrow_pade
        _j2_para = two_body_jastrow_pade
    elif j2b_type == "exp":
        _j2_anti = two_body_jastrow_exp
        _j2_para = two_body_jastrow_exp
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    two_body_jastrow = (
        np.sum(
            [
                _j2_anti(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart,
                    r_cart_j=r_dn_cart,
                )
                for (r_up_cart, r_dn_cart) in itertools.product(r_up_carts, r_dn_carts)
            ]
        )
        + np.sum(
            [
                _j2_para(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_up_cart_i,
                    r_cart_j=r_up_cart_j,
                )
                for (r_up_cart_i, r_up_cart_j) in itertools.combinations(r_up_carts, 2)
            ]
        )
        + np.sum(
            [
                _j2_para(
                    param=jastrow_two_body_data.jastrow_2b_param,
                    r_cart_i=r_dn_cart_i,
                    r_cart_j=r_dn_cart_j,
                )
                for (r_dn_cart_i, r_dn_cart_j) in itertools.combinations(r_dn_carts, 2)
            ]
        )
    )

    return two_body_jastrow


def _compute_Jastrow_two_body_pbc_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """fp64 numpy reference for the PBC branch of ``compute_Jastrow_two_body``.

    Helper-free reimplementation: the positive full-Gram
    periodic distance is inlined per electron--electron pair (no call to
    ``_smooth_periodic_distance*``) so the production-vs-debug
    cross-check validates the entire helper chain in one comparison.
    """
    sd = jastrow_two_body_data.structure_data
    cell = np.asarray(sd.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    a = float(jastrow_two_body_data.jastrow_2b_param)
    j2b_type = jastrow_two_body_data.jastrow_2b_type

    def _u(r_ij: float) -> float:
        if j2b_type == "exp":
            return 1.0 / (2.0 * a) * (1.0 - np.exp(-a * r_ij))
        if j2b_type == "pade":
            return r_ij / (2.0 * (1.0 + a * r_ij))
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    def _r_ij(r_i: np.ndarray, r_j: np.ndarray) -> float:
        r_frac = (np.asarray(r_i, dtype=np.float64) - np.asarray(r_j, dtype=np.float64)) @ inv_cell
        phase = np.pi * r_frac
        vector = np.sin(2 * phase) @ cell
        height = 2 * np.sin(phase) ** 2
        square = np.dot(vector, vector) + np.dot(np.sum(cell * cell, axis=1), height * height)
        return float(np.sqrt(square) / (2 * np.pi))

    J2 = 0.0
    for r_up, r_dn in itertools.product(r_up_carts, r_dn_carts):
        J2 += _u(_r_ij(r_up, r_dn))
    for r_i, r_j in itertools.combinations(r_up_carts, 2):
        J2 += _u(_r_ij(r_i, r_j))
    for r_i, r_j in itertools.combinations(r_dn_carts, 2):
        J2 += _u(_r_ij(r_i, r_j))
    return float(J2)


def _phase_free_j3_orbitals(orb_data: AOs_base_data | MOs_data) -> AOs_base_data | MOs_data:
    """Copy a reference orbital basis without its Bloch phase for shared J3.

    MO, plane-wave and spline coefficients are retained exactly. This does
    not recompute Gamma-point orbitals: their fixed cell-periodic functions
    define the shared J3 basis, whose real parts the existing J3 kernels use.
    Call once on a reference basis, never separately on each twist's MOs.
    Determinant orbitals are immutable and are not modified by this copy.
    """
    if isinstance(orb_data, MOs_data):
        return orb_data.replace(aos_data=_phase_free_j3_orbitals(orb_data.aos_data))
    if isinstance(orb_data, AOs_composite_data):
        return orb_data.replace(children=tuple(_phase_free_j3_orbitals(child) for child in orb_data.children))
    if isinstance(orb_data, (AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data)):
        # Retain complex storage for a potentially complex fixed MO matrix,
        # while removing BOTH image-sum and wrap-restore Bloch phases.
        return orb_data.replace(
            k_point=(0.0, 0.0, 0.0),
            _image_phase_factors=np.ones_like(np.asarray(orb_data._image_phase_factors)),
        )
    if isinstance(orb_data, (AOs_plane_wave_complex_data, AOs_bspline_complex_data)):
        return orb_data.replace(k_point=(0.0, 0.0, 0.0))
    if isinstance(orb_data, (AOs_cart_gto_data, AOs_sphe_gto_data)):
        return orb_data.replace(k_point=None)
    return orb_data


def _validate_j3_phase_free_basis(orb_data: AOs_base_data | MOs_data, *, concrete_only: bool = False) -> None:
    """Reject phase-bearing J3 at construction/load, outside numerical kernels.

    ``concrete_only`` (construction-time use) inspects the static ``k_point``
    always and the image-phase leaf only when it is a host ``ndarray``, so
    pytree unflattening with tracers or placeholder leaves is never evaluated.
    """
    from .atomic_orbital import AOs_bspline_tiled_complex_data
    from .orbital_tiling import TiledOrbitalSet_data

    if isinstance(orb_data, (TiledOrbitalSet_data, AOs_bspline_tiled_complex_data)):
        raise ValueError("J3 requires an independent, untiled physical-supercell basis; determinant tiling is not a J3 basis.")
    if isinstance(orb_data, MOs_data):
        _validate_j3_phase_free_basis(orb_data.aos_data, concrete_only=concrete_only)
        return
    if isinstance(orb_data, AOs_composite_data):
        for child in orb_data.children:
            _validate_j3_phase_free_basis(child, concrete_only=concrete_only)
        return
    k_point = getattr(orb_data, "k_point", None)
    if k_point is not None and np.any(np.asarray(k_point) != 0):
        raise ValueError(
            "J3 requires a phase-free basis shared by all twists; a nonzero Bloch k_point is not allowed. "
            "Rebuild the shared J3 reference with the converter rather than projecting a Bloch orbital to its real part."
        )
    image_phases = getattr(orb_data, "_image_phase_factors", None)
    if concrete_only and not isinstance(image_phases, np.ndarray):
        return
    if image_phases is not None and np.any(np.asarray(image_phases) != 1):
        raise ValueError("J3 requires phase-free image factors equal to one, not stale Bloch image phases.")


# @dataclass
@struct.dataclass
class Jastrow_three_body_data:
    r"""Three-body Jastrow parameters and orbital references.

    The three-body term uses a square symmetric J3 block plus a last-column
    J1-like vector. Values are returned without
    exponentiation; callers attach ``exp(J)`` to the wavefunction. The canonical
    J3 orbital basis is real (Gamma); complex non-Gamma orbital leaves are not a
    canonical J3 basis and are rejected at construction/load validation.
    Detailed periodic basis conventions are work in progress.

    Args:
        orb_data (AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data): Basis/orbital data used for both spins.
        j_matrix (npt.NDArray[np.float64]): J matrix with shape ``(orb_num, orb_num + 1)``. dtype: float64.
    """

    orb_data: AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data = struct.field(
        pytree_node=True, default_factory=AOs_sphe_gto_data
    )  #: Orbital basis (AOs or MOs) shared across spins.
    j_matrix: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )  #: J3/J1 matrix; square block plus final column. dtype: float64.

    def __post_init__(self) -> None:
        """Reject a phase-bearing J3 basis on every construction path.

        ``build`` validates through ``sanity_check``, but direct construction,
        ``init_jastrow_three_body_data`` and ``replace`` bypassed it, so a
        Bloch-phased J3 reached the kernels and the Rust home-cell wrap and the
        XLA image sum silently disagreed. Only concrete metadata is inspected.
        """
        _validate_j3_phase_free_basis(self.orb_data, concrete_only=True)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        _validate_j3_phase_free_basis(self.orb_data)
        self.orb_data.sanity_check()
        if not np.all(np.isfinite(self.j_matrix)):
            raise ValueError("J3 coupling matrix must be finite.")
        # The J3 AOs are evaluated in the ao_eval zone as well: mixed precision
        # rejects a basis whose AO values leave its range (issue #111).
        check_ao_eval_value_range(self.orb_data.aos_data if isinstance(self.orb_data, MOs_data) else self.orb_data)
        if self.j_matrix.shape != (
            self.orb_num,
            self.orb_num + 1,
        ):
            raise ValueError(
                f"dim. of j_matrix = {self.j_matrix.shape} is incompatible with the expected one "
                + f"= ({self.orb_num}, {self.orb_num + 1}).",
            )
        # The J3 coupling block must be SYMMETRIC. The Jastrow factor is bosonic
        # (symmetric under particle exchange), and the optimiser keeps the J3
        # matrix symmetric, so this always holds. Several streaming/ratio kernels
        # exploit it (``j3_mat.T @ x == j3_mat @ x``, so the second matvec and the
        # ``U = j3_mat.T @ aos`` cached tables are dropped). Assert here rather
        # than symmetrising on load, so a violation (which would silently change
        # results in those kernels) is surfaced.
        j3_block = np.asarray(self.j_matrix[:, :-1])
        if j3_block.size and not np.allclose(
            j3_block, j3_block.T, rtol=RTOL_J3_COUPLING_SYMMETRY, atol=ATOL_J3_COUPLING_SYMMETRY
        ):
            raise ValueError(
                "J3 coupling block j_matrix[:, :-1] must be symmetric (bosonic Jastrow); "
                f"max|J - J^T| = {np.max(np.abs(j3_block - j3_block.T)):.3e}. "
                "The streaming/ratio kernels rely on this."
            )

    def _get_info(self) -> list[str]:
        """Return a list of strings containing the information stored in the attributes."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  dim. of jastrow_3b_matrix = {self.j_matrix.shape}")
        info_lines.append(
            f"  j3 part of the jastrow_3b_matrix is symmetric? = {np.allclose(self.j_matrix[:, :-1], self.j_matrix[:, :-1].T)}"
        )
        # Replace orb_data.logger_info() with orb_data.get_info() output.
        info_lines.extend(self.orb_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def orb_num(self) -> int:
        """Get number of atomic orbitals.

        Returns:
            int: get number of atomic orbitals.
        """
        return self.orb_data._num_orb

    @property
    def compute_orb_api(self) -> Callable[..., npt.NDArray[np.float64]]:
        """Function for computing AOs or MOs.

        The api method to compute AOs or MOs corresponding to instances
        stored in self.orb_data

        Return:
            Callable: The api method to compute AOs or MOs.

        Raises:
            NotImplementedError:
                If the instances of orb_data is neither AOs_data nor MOs_data.
        """
        # ``AOs_composite_data`` (e.g. the QE pgto+pw composite) is a valid AO leaf: it
        # arrives here after ``_j3_mo_to_bare_ao`` folds an MO-backed composite J3 down to
        # its bare-AO leaf, and ``compute_AOs`` dispatches through ``aos_data._compute_AOs``
        # for every ``AOs_base_data`` subtype (composite included). Recognising it keeps the
        # ECP-mesh J3 base (r/R/eps e_L axes) from raising a bare NotImplementedError.
        if isinstance(self.orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data, AOs_composite_data)):
            return compute_AOs
        if isinstance(self.orb_data, MOs_data):
            return compute_MOs
        raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    @property
    def _j_matrix_jnp(self) -> jax.Array:
        """Return j_matrix as a jax.Array (jnp view of the underlying numpy storage)."""
        # Lift-only fp64 basis-data storage accessor (see _precision.py exemption);
        # consumer casts to its own zone at use site.
        return jnp.asarray(self.j_matrix, dtype=jnp.float64)

    @property
    def ao_exponents(self) -> jax.Array:
        """AO Gaussian exponents (jnp view of underlying numpy storage)."""
        if isinstance(self.orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
            return jnp.asarray(self.orb_data.radial.exponents)
        if isinstance(self.orb_data, MOs_data):
            return jnp.asarray(self.orb_data.aos_data.radial.exponents)
        raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    @property
    def ao_coefficients(self) -> jax.Array:
        """AO contraction coefficients (jnp view of underlying numpy storage)."""
        if isinstance(self.orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
            return jnp.asarray(self.orb_data.radial.coefficients)
        if isinstance(self.orb_data, MOs_data):
            return jnp.asarray(self.orb_data.aos_data.radial.coefficients)
        raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    def with_updated_ao_exponents(self, new_exp: npt.NDArray[np.float64]) -> "Jastrow_three_body_data":
        """Return a new instance with updated AO exponents."""
        if isinstance(self.orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
            return self.replace(orb_data=self.orb_data._with_updated_basis_parameter("exponents", new_exp))
        if isinstance(self.orb_data, MOs_data):
            new_aos = self.orb_data.aos_data._with_updated_basis_parameter("exponents", new_exp)
            return self.replace(orb_data=self.orb_data.replace(aos_data=new_aos))
        raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    def with_updated_ao_coefficients(self, new_coeff: npt.NDArray[np.float64]) -> "Jastrow_three_body_data":
        """Return a new instance with updated AO contraction coefficients."""
        if isinstance(self.orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
            return self.replace(orb_data=self.orb_data._with_updated_basis_parameter("coefficients", new_coeff))
        if isinstance(self.orb_data, MOs_data):
            new_aos = self.orb_data.aos_data._with_updated_basis_parameter("coefficients", new_coeff)
            return self.replace(orb_data=self.orb_data.replace(aos_data=new_aos))
        raise NotImplementedError(f"Unsupported orb_data type: {type(self.orb_data)}")

    @classmethod
    def init_jastrow_three_body_data(
        cls,
        orb_data: AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data,
        random_init: bool = False,
        random_scale: float = 0.01,
        seed: int | None = None,
    ) -> Self:
        """Initialization.

        Args:
            orb_data: Orbital container (AOs or MOs) used to size the J-matrix.
            random_init: If True, initialize with small random values instead of zeros (for tests).
            random_scale: Upper bound of uniform sampler when random_init is True (default 0.01).
            seed: Optional seed for deterministic initialization when random_init is True.
        """
        dtype_np = _JASTROW_PARAMETER_STORAGE_DTYPE
        if random_init:
            rng = np.random.default_rng(seed)
            j_matrix = rng.uniform(0.0, random_scale, size=(orb_data._num_orb, orb_data._num_orb + 1)).astype(dtype_np)
            # The J3 coupling block must be SYMMETRIC (bosonic Jastrow; the
            # optimiser preserves this). Symmetrise the square block in place;
            # the final column is the j1-like vector and is left untouched.
            j_matrix[:, :-1] = 0.5 * (j_matrix[:, :-1] + j_matrix[:, :-1].T)
        else:
            j_matrix = np.zeros((orb_data._num_orb, orb_data._num_orb + 1), dtype=dtype_np)

        jastrow_three_body_data = cls(
            orb_data=orb_data,
            j_matrix=j_matrix,
        )
        return jastrow_three_body_data

    @classmethod
    def build(
        cls,
        *,
        orb_data: AOs_sphe_gto_data | AOs_cart_gto_data | MOs_data,
        random_init: bool = False,
        random_scale: float = 0.01,
        seed: int | None = None,
    ) -> Self:
        """Build a validated :class:`Jastrow_three_body_data`.

        Blessed factory: delegates to
        :py:meth:`init_jastrow_three_body_data` (which sizes / initializes
        the J-matrix from ``orb_data``) and then runs
        :py:meth:`sanity_check`. Direct construction remains supported.
        """
        data = cls.init_jastrow_three_body_data(orb_data, random_init, random_scale, seed)
        data.sanity_check()
        return data

    def to_cartesian(self) -> "Jastrow_three_body_data":
        """Convert spherical orbitals to Cartesian and transform the J-matrix.

        If the underlying orbitals are MOs, defer to ``MOs_data.to_cartesian``
        (the J-matrix size is unchanged). For Cartesian inputs, return self.
        """
        if isinstance(self.orb_data, AOs_cart_gto_data):
            return self
        if isinstance(self.orb_data, MOs_data):
            return Jastrow_three_body_data(orb_data=self.orb_data.to_cartesian(), j_matrix=self.j_matrix)
        if not isinstance(self.orb_data, AOs_sphe_gto_data):
            raise ValueError("Cartesian conversion is only available from spherical AOs or MOs.")

        aos_cart, transform_matrix = _aos_sphe_to_cart(self.orb_data)

        dtype_np = _JASTROW_PARAMETER_STORAGE_DTYPE
        square_sph = np.asarray(self.j_matrix[:, :-1], dtype=dtype_np)
        j1_sph = np.asarray(self.j_matrix[:, -1], dtype=dtype_np)
        square_cart = transform_matrix.T @ square_sph @ transform_matrix
        j1_cart = transform_matrix.T @ j1_sph

        j_matrix_cart = np.zeros((aos_cart.num_ao, aos_cart.num_ao + 1), dtype=dtype_np)
        j_matrix_cart[:, :-1] = square_cart
        j_matrix_cart[:, -1] = j1_cart

        return Jastrow_three_body_data(orb_data=aos_cart, j_matrix=j_matrix_cart)

    def to_spherical(self) -> "Jastrow_three_body_data":
        """Convert Cartesian orbitals to spherical and transform the J-matrix.

        Cartesian AO coefficients use the global molecular overlap metric (or
        periodic Bloch ``S(k)`` metric); the returned map is applied to both
        sides of the quadratic J3 block and to the linear J1 block.  The
        canonical J3 orbital basis is real (Gamma), as documented by this
        class; complex non-Gamma orbital leaves are not a canonical J3 basis.
        If the underlying orbitals are MOs, defer to ``MOs_data.to_spherical``
        (the J-matrix size is unchanged). For spherical inputs, return self.
        """
        if isinstance(self.orb_data, AOs_sphe_gto_data):
            return self
        if isinstance(self.orb_data, MOs_data):
            return Jastrow_three_body_data(orb_data=self.orb_data.to_spherical(), j_matrix=self.j_matrix)
        if not isinstance(self.orb_data, AOs_cart_gto_data):
            raise ValueError("Spherical conversion is only available from Cartesian AOs or MOs.")

        aos_sphe, cart_to_sphe = _aos_cart_to_sphe(self.orb_data)

        dtype_np = _JASTROW_PARAMETER_STORAGE_DTYPE
        square_cart = np.asarray(self.j_matrix[:, :-1], dtype=dtype_np)
        j1_cart = np.asarray(self.j_matrix[:, -1], dtype=dtype_np)
        square_sph = cart_to_sphe.T @ square_cart @ cart_to_sphe
        j1_sph = cart_to_sphe.T @ j1_cart

        j_matrix_sph = np.zeros((aos_sphe.num_ao, aos_sphe.num_ao + 1), dtype=dtype_np)
        j_matrix_sph[:, :-1] = square_sph
        j_matrix_sph[:, -1] = j1_sph

        return Jastrow_three_body_data(orb_data=aos_sphe, j_matrix=j_matrix_sph)


def compute_Jastrow_three_body(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    zone: str = "jastrow_eval",
) -> jax.Array:
    r"""Evaluate the three-body Jastrow :math:`J_3` (analytic) without exponentiation.

    The square J3 block couples electron pairs and the last column acts as a
    J1-like vector. Returned value
    is ``J``; attach ``exp(J)`` externally.

    Args:
        jastrow_three_body_data: Three-body Jastrow parameters and orbitals.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.
        zone: Precision zone of the evaluation (default ``"jastrow_eval"``, the
            zone of whole Jastrow values). The whole-value ratio routes pass
            ``"jastrow_ratio"`` (see :func:`compute_Jastrow_part_ratio_zone`).

    Returns:
        float: Three-body Jastrow value (before exponentiation).
    """
    # r_*_carts forwarded unchanged to ``compute_orb_api`` (the AO/MO kernels
    # reconstruct ``r - R`` in float64 internally); do not pre-cast here.
    dtype_jnp = get_dtype_jnp(zone)
    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    num_electron_up = len(r_up_carts)
    num_electron_dn = len(r_dn_carts)

    aos_up = jnp.array(
        jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_up_carts)), dtype=dtype_jnp
    )
    aos_dn = jnp.array(
        jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, r_dn_carts)), dtype=dtype_jnp
    )

    j1_matrix_up = j_matrix[:, -1]
    j1_matrix_dn = j_matrix[:, -1]
    j3_matrix_up_up = j_matrix[:, :-1]
    j3_matrix_dn_dn = j_matrix[:, :-1]
    j3_matrix_up_dn = j_matrix[:, :-1]

    e_up = jnp.ones(num_electron_up, dtype=dtype_jnp).T
    e_dn = jnp.ones(num_electron_dn, dtype=dtype_jnp).T

    # Same-spin pair term in closed form. The dense expression builds
    # X = A.T @ M @ A -- an (N_e, N_e) matrix via two O(N_e^2 num_orb)
    # gemms -- multiplies it by a strict-triangular ones matrix and takes the
    # trace, i.e. the strict-upper-triangle SUM of X (A = 2.9e6, ~3 GB for a
    # large periodic cell, plus the FLOP waste). The strict-upper
    # sum is exactly
    #     sum_{i<j} X_ij = sum_j p_j.T M a_j = sum(P * (M @ A)),
    #     P[:, j] = sum_{i<j} A[:, i]   (exclusive column prefix sum)
    # -- one (num_orb, num_orb) x (num_orb, N) gemm plus a cumsum, no
    # (N_e, N_e) intermediate, and valid for ARBITRARY M. Do NOT assume M is
    # symmetric: symmetrising M via 0.5 * (M + M.T) equals the strict-upper
    # sum only when M is symmetric; the analytic derivative path keeps the
    # asymmetric convention, and real converted J3 matrices (e.g.
    # N_ae_ccpvdz_cart) are NOT symmetric (pinned by
    # test_j3_molecular_axes_vs_auto).
    def _pair_term(aos: jax.Array, m: jax.Array) -> jax.Array:
        p_excl = jnp.cumsum(aos, axis=1) - aos  # (num_orb, N): exclusive prefix
        return jnp.sum(p_excl * (m @ aos))

    J3 = (
        j1_matrix_up @ aos_up @ e_up
        + j1_matrix_dn @ aos_dn @ e_dn
        + _pair_term(aos_up, j3_matrix_up_up)
        + _pair_term(aos_dn, j3_matrix_dn_dn)
        + e_up.T @ aos_up.T @ j3_matrix_up_dn @ aos_dn @ e_dn
    )

    return J3


def _compute_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    *,
    zone: str = "jastrow_eval",
) -> float:
    """See _api method.

    ``zone`` selects the precision zone the reference accumulates in. The
    default ``"jastrow_eval"`` mirrors :func:`compute_Jastrow_three_body`.
    :func:`_compute_ratio_Jastrow_part_debug` passes ``"jastrow_ratio"``: it
    subtracts two accumulated J3 values, and that cancellation must happen in
    the ratio zone (fp64 in both modes), not in the fp32 mixed-mode
    ``jastrow_eval`` zone.
    """
    orbitals = jastrow_three_body_data.orb_data
    orbital_debug = _compute_MOs_debug if isinstance(orbitals, MOs_data) else _compute_AOs_debug
    dtype = get_dtype_np(zone)
    aos_up = np.asarray(np.real(orbital_debug(orbitals, r_up_carts)), dtype=dtype)
    aos_dn = np.asarray(np.real(orbital_debug(orbitals, r_dn_carts)), dtype=dtype)
    matrix = np.asarray(jastrow_three_body_data.j_matrix, dtype=dtype)

    # compute one body
    J_1_up = dtype(0)
    j1_vector_up = matrix[:, -1]
    for i in range(len(r_up_carts)):
        ao_up = aos_up[:, i]
        for al in range(len(ao_up)):
            J_1_up += j1_vector_up[al] * ao_up[al]

    J_1_dn = dtype(0)
    j1_vector_dn = matrix[:, -1]
    for i in range(len(r_dn_carts)):
        ao_dn = aos_dn[:, i]
        for al in range(len(ao_dn)):
            J_1_dn += j1_vector_dn[al] * ao_dn[al]

    # compute three-body
    J_3_up_up = dtype(0)
    j3_matrix_up_up = matrix[:, :-1]
    for i in range(len(r_up_carts)):
        for j in range(i + 1, len(r_up_carts)):
            ao_up_i = aos_up[:, i]
            ao_up_j = aos_up[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_up_j)):
                    J_3_up_up += j3_matrix_up_up[al, bm] * ao_up_i[al] * ao_up_j[bm]

    J_3_dn_dn = dtype(0)
    j3_matrix_dn_dn = matrix[:, :-1]
    for i in range(len(r_dn_carts)):
        for j in range(i + 1, len(r_dn_carts)):
            ao_dn_i = aos_dn[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_dn_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_dn_dn += j3_matrix_dn_dn[al, bm] * ao_dn_i[al] * ao_dn_j[bm]

    J_3_up_dn = dtype(0)
    j3_matrix_up_dn = matrix
    for i in range(len(r_up_carts)):
        for j in range(len(r_dn_carts)):
            ao_up_i = aos_up[:, i]
            ao_dn_j = aos_dn[:, j]
            for al in range(len(ao_up_i)):
                for bm in range(len(ao_dn_j)):
                    J_3_up_dn += j3_matrix_up_dn[al, bm] * ao_up_i[al] * ao_dn_j[bm]

    J3 = J_1_up + J_1_dn + J_3_up_up + J_3_dn_dn + J_3_up_dn

    return J3


# =====================================================================
# DTN (Drummond--Towler--Needs) polynomial Jastrow (J1 + J2 + J3 family).
# Cutoff polynomial with factor (L - r)^C Theta(L - r): u (e-e), chi (e-n),
# f (e-e-n). Note (L - r)^C = (-1)^C (r - L)^C: the (r - L)^C sign convention
# of CASINO-style write-ups must NOT be used for the inhomogeneous cusp biases.
# Cusps are encoded by linear constraints on the polynomial coefficients via
# the inflation maps M_alpha (with bias b_alpha), M_beta (with b_beta),
# and M_gamma (homogeneous). Distance kernel dispatches on
# Structure_data.pbc_flag (true minimum-image distance in PBC).
# =====================================================================

#: Default dimensionless periodic DTN cutoff ratio for every channel
#: (u, chi and f).  ``0.95`` is the Wigner--Seitz / inscribed-sphere radius
#: convention (QMCPACK) with a 5% margin from the minimum-image seam: the
#: cutoffs resolve to ``0.95 * R_ins`` for u / chi and ``0.95 * R_ins / 2``
#: for f, where ``R_ins`` = ``Structure_data._wigner_seitz_inscribed_radius()``
#: is half the shortest non-zero lattice vector (basis independent).  The comoving cutoff policy keeps the direct stress consistent
#: with the equation of state at any ratio.
DTN_DEFAULT_CUTOFF_RATIO = 0.95


def _dtn_default_cutoffs(
    structure_data: Structure_data,
    cutoff_fraction: float,
    cutoff_abs_cap: float,
) -> tuple[float, float, float]:
    """Geometry-derived default cutoffs ``(L_u, L_chi, L_f)``.

    No optimisation -- ship good geometry-derived defaults:

    * PBC: ``L_u=L_chi=cutoff_fraction*R_inscribed`` and
      ``L_f=cutoff_fraction*R_inscribed/2``, with ``R_inscribed`` the
      Wigner--Seitz inscribed radius (half the shortest non-zero lattice
      vector; a property of the lattice, not of the cell basis).  Thus the one
      dimensionless ratio has identical semantics for every channel: a
      fraction of that channel's geometric safe maximum.
    * ``L_u`` (e-e, long ranged), molecular:
        - Molecular: ``min(R_max + 10, cutoff_abs_cap)``, ``R_max`` = max
          distance from the molecular centroid to any atom. The *additive*
          +10 Bohr keeps L_u long for small molecules -- the e-e correlation
          length is not bounded by molecular size (a multiplicative rule would
          undershoot, e.g. ~1 Bohr for H2).
    * ``L_chi`` (e-N -- short), molecular:
        ``2.5 * nn`` (``nn`` = median nearest-neighbour ion distance), clamped
        to ``cutoff_abs_cap``. Single-atom / no-neighbour geometries (no
        ``nn``) fall back to the ``L_u`` policy.
    * ``L_f`` (e-e-N), molecular: same ``2.5 * nn`` rule. Under PBC the
        channel maximum is ``R_inscribed / 2`` because the f-term's reach in the
        PAIR separation is ``2 L_f`` (two electrons inside the same ion
        sphere), and the independent minimum-image ``r_ij`` used by the
        standard (CASINO/QMCPACK-compatible) DTN convention is smooth only
        inside the WS inscribed radius. ``2 L_f <= R_inscribed`` therefore
        is a SMOOTHNESS requirement, not a tuning choice: violating it puts
        the min-image seam (a gradient discontinuity of ``r_ij``) inside the
        f-term's support, ln(Psi) acquires a kink surface whose kinetic
        surface-delta pointwise e_L sampling never sees, and the VMC
        optimizer collapses the sampled energy below the fixed-node bound
        (observed on a periodic slab). CASINO enforces the
        same ``L_f <= R_ins/2`` bound on its f-like cutoffs.
    """
    if structure_data.pbc_flag:
        pbc_bound = cutoff_fraction * structure_data._wigner_seitz_inscribed_radius()
        return float(pbc_bound), float(pbc_bound), float(0.5 * pbc_bound)

    positions = np.asarray(structure_data.positions, dtype=np.float64)
    if positions.shape[0] > 0:
        centroid = positions.mean(axis=0)
        R_max = float(np.max(np.linalg.norm(positions - centroid, axis=1)))
    else:
        R_max = 0.0
    L_u = float(min(R_max + 10.0, cutoff_abs_cap))

    nn = structure_data._median_nn_distance()
    if nn is None:
        # No neighbour pair (single atom) -> reuse the L_u policy for the
        # short cutoffs as well (f keeps its halved PBC bound).
        L_chi = L_u
        L_f = L_u
    else:
        L_chi = min(2.5 * nn, cutoff_abs_cap)
        L_f = L_chi
    return float(L_u), float(L_chi), float(L_f)


def _dtn_gamma_constraint_matrix(
    n_order_eN: int,
    n_order_ee: int,
    cutoff_smoothness: int,
    L_f: float,
) -> np.ndarray:
    """Raw homogeneous f-channel constraint matrix for one species."""
    N_eN = int(n_order_eN)
    N_ee = int(n_order_ee)
    C = int(cutoff_smoothness)
    raw_size = (N_eN + 1) * (N_eN + 1) * (N_ee + 1)

    def idx(l: int, m: int, n: int) -> int:
        return (l * (N_eN + 1) + m) * (N_ee + 1) + n

    rows: list[np.ndarray] = []
    for l in range(N_eN + 1):
        for m in range(l):
            for n in range(N_ee + 1):
                row = np.zeros(raw_size, dtype=np.float64)
                row[idx(l, m, n)] = 1.0
                row[idx(m, l, n)] = -1.0
                rows.append(row)
    for total in range(2 * N_eN + 1):
        row = np.zeros(raw_size, dtype=np.float64)
        for l in range(N_eN + 1):
            m = total - l
            if 0 <= m <= N_eN:
                row[idx(l, m, 1)] += 1.0
        rows.append(row)
    for total in range(N_eN + N_ee + 1):
        row = np.zeros(raw_size, dtype=np.float64)
        for m in range(N_eN + 1):
            n = total - m
            if 0 <= n <= N_ee:
                row[idx(0, m, n)] += C
                row[idx(1, m, n)] -= float(L_f)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _dtn_constraint_transport_tangents(
    *,
    n_species: int,
    z_eff_per_species: np.ndarray,
    pseudized_per_species: np.ndarray,
    use_u: bool,
    use_chi: bool,
    use_f: bool,
    n_order_u: int,
    n_order_chi: int,
    n_order_eN: int,
    n_order_ee: int,
    L_u: float,
    L_chi: np.ndarray,
    L_f: np.ndarray,
    cutoff_smoothness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Linear maps for the comoving-cutoff full-coefficient tangent.

    Let ``t = ln(ell_V)`` and hence ``dL/dt=L``.  If a full coefficient
    vector obeys ``C(L) f=d(L)``, the derivative of the closest Euclidean
    projection into the neighbouring constraint manifold is

    ``f_dot = C(L)^+ [d_dot - C_dot f] = T f + q``.

    These maps are independent of the optimizer coordinates and therefore can
    be stored as static metadata.  Applying them to the *current inflated full
    coefficients* makes the policy derivative invariant to the SVD null-basis
    gauge and to the radial-RMS normalization of the optimizer coordinates.

    The inhomogeneous rows use the evaluators' ``(L - r)^C`` radial form:
    ``C(L) = [C, -L, 0, ...]`` with ``d(L) = -c_s / L^(C-1)`` for the u-term
    and ``d(L) = +cusp_I / L^(C-1)`` for the chi-term, for every ``C``,
    hence ``d_dot = -(C - 1) d``.
    """
    Cpow = int(cutoff_smoothness)

    def affine_map(Cmat: np.ndarray, Cdot: np.ndarray, ddot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cp = np.linalg.pinv(Cmat, rcond=1.0e-10)
        return -cp @ Cdot, cp @ ddot

    Na = int(n_order_u) + 1
    Ta = np.zeros((2, Na, Na), dtype=np.float64)
    qa = np.zeros((2, Na), dtype=np.float64)
    if use_u:
        Cmat = np.zeros((1, Na), dtype=np.float64)
        Cmat[0, 0], Cmat[0, 1] = Cpow, -float(L_u)
        Cdot = np.zeros_like(Cmat)
        Cdot[0, 1] = -float(L_u)
        for s, cusp in enumerate((0.25, 0.5)):
            # (L - r)^C cutoff: u'(0) = cusp  <=>  C alpha_0 - L alpha_1 = -cusp / L^(C-1).
            rhs = -cusp / float(L_u) ** (Cpow - 1)
            Ta[s], qa[s] = affine_map(Cmat, Cdot, np.asarray([-(Cpow - 1) * rhs]))

    Nb = int(n_order_chi) + 1
    Tb = np.zeros((n_species, Nb, Nb), dtype=np.float64)
    qb = np.zeros((n_species, Nb), dtype=np.float64)
    if use_chi:
        for species in range(n_species):
            L = float(L_chi[species])
            Cmat = np.zeros((1, Nb), dtype=np.float64)
            Cmat[0, 0], Cmat[0, 1] = Cpow, -L
            Cdot = np.zeros_like(Cmat)
            Cdot[0, 1] = -L
            cusp = 0.0 if bool(pseudized_per_species[species]) else float(z_eff_per_species[species])
            # (L - r)^C cutoff: chi'(0) = -Z  <=>  C beta_0 - L beta_1 = +Z / L^(C-1).
            rhs = cusp / L ** (Cpow - 1)
            Tb[species], qb[species] = affine_map(Cmat, Cdot, np.asarray([-(Cpow - 1) * rhs]))

    Ng = (int(n_order_eN) + 1) ** 2 * (int(n_order_ee) + 1)
    Tg = np.zeros((n_species, Ng, Ng), dtype=np.float64)
    qg = np.zeros((n_species, Ng), dtype=np.float64)
    if use_f:
        N_eN, N_ee = int(n_order_eN), int(n_order_ee)

        def idx(l: int, m: int, n: int) -> int:
            return (l * (N_eN + 1) + m) * (N_ee + 1) + n

        n_static_rows = N_eN * (N_eN + 1) // 2 * (N_ee + 1) + (2 * N_eN + 1)
        for species in range(n_species):
            L = float(L_f[species])
            Cmat = _dtn_gamma_constraint_matrix(N_eN, N_ee, Cpow, L)
            Cdot = np.zeros_like(Cmat)
            for row_index, total in enumerate(range(N_eN + N_ee + 1), start=n_static_rows):
                for m in range(N_eN + 1):
                    n = total - m
                    if 0 <= n <= N_ee:
                        Cdot[row_index, idx(1, m, n)] -= L
            Tg[species], qg[species] = affine_map(Cmat, Cdot, np.zeros(Cmat.shape[0], dtype=np.float64))
    return Ta, qa, Tb, qb, Tg, qg


def _normalize_inflation_columns(M: np.ndarray, basis_scale: np.ndarray) -> np.ndarray:
    """Rescale each independent column of an inflation map to unit basis scale.

    The DTN basis functions carry wildly different magnitudes between
    channels: the u / chi terms scale as a single cutoff ``L^C`` (times
    ``r^k``), while the f term scales as a *double* cutoff ``L^{2C}``
    (times ``r_i^l r_j^m r_ij^n``). At ``L ~ 3`` Bohr, ``C = 4`` this makes
    ``O_f = dlnPsi/dgamma`` roughly two orders of magnitude larger than
    ``O_u / O_chi``. The natural-gradient (SR / LM) update is invariant to
    this rescaling *in exact arithmetic*, but in float64 the wildly
    different column scales (a) inflate ``cond(S)`` when DTN is co-optimised
    with O(1)-scale blocks (e.g. geminal / MO), and (b) cause
    catastrophic-cancellation loss when ``S`` / ``f`` are assembled as
    Monte-Carlo sums. Folding an analytic per-column scale into the
    inflation map makes every variational coefficient an O(1)-natural
    variable so ``O = dlnPsi/dc`` is O(1) across all channels.

    The column scale is the L2 norm of ``basis_scale * M[..., i]`` over the
    full-coefficient axes, i.e. the analytic magnitude of the basis
    derivative that independent coefficient ``i`` controls. Scaling only the
    *columns* of ``M`` leaves ``span(M)`` (hence every cusp / symmetry /
    cuspless constraint) exactly intact; the inhomogeneous bias ``b`` is
    untouched, so the inhomogeneous cusp still holds. The reparametrisation
    is physics-preserving: the representable family of Jastrow factors and
    the default (zero-coefficient) value are unchanged.

    Args:
        M: Inflation map ``(*lead, *full_axes, n_indep)`` whose last axis
            indexes the independent coefficients and whose middle axes index
            the full polynomial coefficients.
        basis_scale: Analytic basis magnitude broadcastable to
            ``(*lead, *full_axes)`` (positive).

    Returns:
        A copy of ``M`` with each independent column divided by its scale.
    """
    if M.shape[-1] == 0:
        return M
    # weighted = basis_scale (broadcast over the indep axis) * M
    weighted = basis_scale[..., np.newaxis] * M
    # L2 norm over all full-coefficient axes (everything except the leading
    # axis -- spin/species -- and the trailing indep axis).
    full_axes = tuple(range(1, M.ndim - 1))
    col_norm = np.sqrt(np.sum(weighted * weighted, axis=full_axes, keepdims=True))  # (lead, 1.., 1, n_indep)
    # Guard against an all-zero column (cannot happen for a non-trivial
    # null-space basis with strictly positive basis_scale, but stay safe).
    col_norm = np.where(col_norm > 0.0, col_norm, 1.0)
    return M / col_norm


def _dtn_radial_rms(cutoff_power: int, poly_power: int, L: float, *, with_cutoff: bool, n_grid: int = 256) -> float:
    """RMS magnitude of a 1-D DTN radial basis factor over its support ``[0, L]``.

    The basis-derivative magnitude ``O = dlnPsi/dc`` is what conditions the
    SR / LM linear system; to make it O(1)-natural per coefficient we need
    the *typical* (not peak) magnitude of each radial factor under the
    physical radial measure ``r^2 dr``. Evaluating the cutoff at its peak
    ``r = 0`` (``L^C``) badly over-weights the f-term's *double* cutoff
    relative to the u / chi single cutoff: at a typical separation
    ``r ~ L/2`` the factor ``(L - r)^C`` is already an order of magnitude
    below its peak, and that suppression compounds quadratically for the
    product of two cutoffs. The radial RMS captures this correctly and
    parameter-free.

    Args:
        cutoff_power: ``C`` (cutoff smoothness); ignored when ``with_cutoff`` is False.
        poly_power: monomial power ``k`` (or ``l`` / ``m`` / ``n``).
        L: cutoff length (Bohr) defining the support ``[0, L]``.
        with_cutoff: ``True`` for a cutoff-bearing factor ``(L - r)^C r^k``
            (u / chi, and each e-n factor of f); ``False`` for the bare e-e
            factor ``r^n`` of the f-term (no cutoff of its own).
        n_grid: radial quadrature resolution.

    Returns:
        ``sqrt(<f^2>)`` with ``<.>`` the ``r^2 dr`` average over ``[0, L]``.
    """
    r = np.linspace(0.0, L, n_grid)
    f = r**poly_power
    if with_cutoff:
        f = f * np.maximum(L - r, 0.0) ** cutoff_power
    w = r * r  # radial measure r^2 dr
    denom = float(np.sum(w))
    if denom <= 0.0:
        return 1.0
    return float(np.sqrt(np.sum(f * f * w) / denom))


# Residual-norm threshold for a pivot of ``_canonical_null_space_basis``: a
# projected unit vector already in the span of the earlier columns leaves a
# residual at rounding level (~1e-15), and a new direction leaves one of
# order its projector diagonal.
_NULL_BASIS_PIVOT_TOL = 1.0e-6


def _canonical_null_space_basis(C_mat: np.ndarray) -> np.ndarray:
    """Orthonormal null-space basis of ``C_mat`` that does not depend on the LAPACK build.

    The rows ``Vt[rank:]`` of an SVD span the null space, but inside that
    subspace the LAPACK implementation may return any rotation, and it does
    differ between builds (OpenBLAS on aarch64 and on x86_64 return different
    bases for the same f-term constraints, issue #528). The span is fixed, so
    the orthogonal projector ``P = N N^T`` is the same on every platform up
    to rounding. Gram-Schmidt over the projected unit vectors ``P e_j``, in
    index order ``j = 0, 1, ...``, therefore gives the same basis everywhere.
    A projected unit vector joins the basis when its residual norm is above
    ``_NULL_BASIS_PIVOT_TOL``; otherwise it already lies in the span of the
    earlier columns. Each kept column has a positive component at its own
    pivot index ``j``, because that component equals the squared residual
    norm. That is the sign convention.

    Args:
        C_mat: Constraint rows ``(n_rows, raw_size)``.

    Returns:
        ``(raw_size, n_null)`` array with orthonormal columns spanning
        ``{x : C_mat x = 0}``.
    """
    raw_size = C_mat.shape[1]
    if C_mat.shape[0] == 0:
        return np.eye(raw_size, dtype=np.float64)
    _, S, Vt = np.linalg.svd(C_mat, full_matrices=True)
    s_max = float(S.max()) if S.size > 0 else 0.0
    rank_tol = 1.0e-10 * max(s_max, 1.0)
    rank = int(np.sum(S > rank_tol))
    n_null = raw_size - rank
    if n_null == 0:
        return np.zeros((raw_size, 0), dtype=np.float64)
    null_rows = Vt[rank:]
    projector = null_rows.T @ null_rows
    basis: list[np.ndarray] = []
    for j in range(raw_size):
        v = projector[:, j].copy()
        # Two Gram-Schmidt passes keep the columns orthonormal to rounding.
        for _ in range(2):
            for q in basis:
                v -= float(q @ v) * q
        norm = float(np.linalg.norm(v))
        if norm > _NULL_BASIS_PIVOT_TOL:
            basis.append(v / norm)
            if len(basis) == n_null:
                break
    if len(basis) != n_null:
        raise RuntimeError(f"canonical null-space basis found {len(basis)} of {n_null} directions")
    return np.stack(basis, axis=1)


def _build_dtn_constraint_matrices(
    *,
    n_species: int,
    z_eff_per_species: np.ndarray,
    pseudized_per_species: np.ndarray,
    use_u: bool,
    use_chi: bool,
    use_f: bool,
    n_order_u: int,
    n_order_chi: int,
    n_order_eN: int,
    n_order_ee: int,
    L_u: float,
    L_chi: np.ndarray,
    L_f: np.ndarray,
    cutoff_smoothness: int,
    normalize_basis: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inflation matrices for the DTN polynomial Jastrow.

    Returns ``(M_alpha, b_alpha, M_beta, b_beta, M_gamma)`` such that the full
    polynomial coefficients are reconstructed as

        alpha_full[s, k] = sum_i M_alpha[s, k, i] alpha_indep[s, i] + b_alpha[s, k]
        beta_full [I, k] = sum_i M_beta [I, k, i] beta_indep [I, i] + b_beta [I, k]
        gamma_full[I, l, m, n] = sum_i M_gamma[I, l, m, n, i] gamma_indep[I, i]

    Constraint algebra (for the evaluators' radial form
    ``(L - r)^C sum_k c_k r^k``, whose slope at ``r = 0`` is
    ``L^{C-1} (L c_1 - C c_0)`` for EVERY integer ``C``):

      * ``alpha`` (u-term): one inhomogeneous e-e cusp constraint per spin
        channel. Eliminate ``alpha_{s, 1}`` in favour of ``alpha_{s, 0}``
        plus the fixed bias ``b_alpha[s, 1] = c_s / L_u^C`` with cusp value
        ``c_s = 1/4`` (parallel) / ``1/2`` (antiparallel).
      * ``beta`` (chi-term): one e-n cusp constraint per species, eliminating
        ``beta_{I, 1}`` with ``b_beta[I, 1] = -cusp_I / L_chi^C``. Pseudized
        species use zero slope (``cusp_I = 0``); all-electron species use
        ``dchi/dr|_0+ = -Z_eff`` (``cusp_I = Z_eff``).
      * A ``(r - L)^C`` derivation would multiply both biases by ``(-1)^C``
        and flip the imposed slopes for odd ``C``; it must not be used.
      * ``gamma`` (f-term): permutation symmetry on ``(l, m)``, e-e cuspless
        slab (``Sum_{l+m=K} gamma_{l, m, 1} = 0``), and grouped e-n
        cuspless rows
        (``Sum_{m+n=k} (C gamma_{0,m,n} - L_f gamma_{1,m,n}) = 0``).
        Reduced to a per-species orthonormal null-space basis that is the
        same on every platform (``_canonical_null_space_basis``).

    Shapes returned (with inactive terms zero-padded to a uniform leading
    axis so downstream callers can ``replace`` independent of which terms
    are active):

      * ``M_alpha``: ``(2, N_u + 1, n_alpha_indep)``
      * ``b_alpha``: ``(2, N_u + 1)``
      * ``M_beta`` : ``(n_species, N_chi + 1, n_beta_indep)``
      * ``b_beta`` : ``(n_species, N_chi + 1)``
      * ``M_gamma``: ``(n_species, N_eN + 1, N_eN + 1, N_ee + 1, n_gamma_indep)``
    """
    C = int(cutoff_smoothness)
    N_u = int(n_order_u)
    N_chi = int(n_order_chi)
    N_eN = int(n_order_eN)
    N_ee = int(n_order_ee)
    n_species = int(n_species)

    # --- alpha (u-term) --------------------------------------------------
    if use_u:
        n_alpha_indep = N_u  # 1 cusp constraint per channel
        M_alpha = np.zeros((2, N_u + 1, n_alpha_indep), dtype=np.float64)
        b_alpha = np.zeros((2, N_u + 1), dtype=np.float64)
        for s in range(2):
            c_s = 0.25 if s == 0 else 0.5  # 0: parallel, 1: antiparallel
            # The evaluators use u_s(r) = (L - r)^C sum_k alpha_k r^k, so
            # u'_s(0) = L^{C-1} (L alpha_1 - C alpha_0) = c_s
            #   -> alpha_1 = (C/L) alpha_0 + c_s / L^C   (for every C, even or odd)
            M_alpha[s, 0, 0] = 1.0
            M_alpha[s, 1, 0] = C / L_u
            b_alpha[s, 1] = c_s / (L_u**C)
            for k in range(2, N_u + 1):
                M_alpha[s, k, k - 1] = 1.0
    else:
        M_alpha = np.zeros((2, N_u + 1, 0), dtype=np.float64)
        b_alpha = np.zeros((2, N_u + 1), dtype=np.float64)

    # --- beta (chi-term) ------------------------------------------------
    if use_chi:
        n_beta_indep = N_chi
        M_beta = np.zeros((n_species, N_chi + 1, n_beta_indep), dtype=np.float64)
        b_beta = np.zeros((n_species, N_chi + 1), dtype=np.float64)
        # ECP presence, not Z_eff < Z, decides pseudization. A zero-core ECP
        # such as H-ccECP still removes the Coulomb singularity and therefore
        # requires chi'(0)=0 even though Z_eff equals the bare atomic number.
        for I in range(n_species):
            Z_eff_I = float(z_eff_per_species[I])
            L_I = float(L_chi[I])
            cusp_value = 0.0 if bool(pseudized_per_species[I]) else Z_eff_I
            # chi_I(r) = (L - r)^C sum_k beta_k r^k, so
            # chi'_I(0) = L^{C-1} (L beta_1 - C beta_0) = -cusp_value
            #   -> beta_1 = (C/L) beta_0 - cusp_value / L^C   (for every C, even or odd)
            M_beta[I, 0, 0] = 1.0
            M_beta[I, 1, 0] = C / L_I
            b_beta[I, 1] = -cusp_value / (L_I**C)
            for k in range(2, N_chi + 1):
                M_beta[I, k, k - 1] = 1.0
    else:
        M_beta = np.zeros((n_species, N_chi + 1, 0), dtype=np.float64)
        b_beta = np.zeros((n_species, N_chi + 1), dtype=np.float64)

    # --- gamma (f-term) -------------------------------------------------
    if use_f:
        raw_size = (N_eN + 1) * (N_eN + 1) * (N_ee + 1)

        def _idx(l: int, m: int, n: int) -> int:
            return (l * (N_eN + 1) + m) * (N_ee + 1) + n

        M_gamma_list: list[np.ndarray] = []
        for I in range(n_species):
            L_I = float(L_f[I])
            rows: list[np.ndarray] = []
            # (1) permutation symmetry: gamma_{l,m,n} - gamma_{m,l,n} = 0 for l > m
            for l in range(N_eN + 1):
                for m in range(l):
                    for n in range(N_ee + 1):
                        row = np.zeros(raw_size, dtype=np.float64)
                        row[_idx(l, m, n)] = 1.0
                        row[_idx(m, l, n)] = -1.0
                        rows.append(row)
            # (2) e-e cuspless: Sum_{l+m=K} gamma_{l,m,1} = 0 for K = 0..2*N_eN
            for K in range(2 * N_eN + 1):
                row = np.zeros(raw_size, dtype=np.float64)
                for l in range(N_eN + 1):
                    m = K - l
                    if 0 <= m <= N_eN:
                        row[_idx(l, m, 1)] += 1.0
                rows.append(row)
            # (3) e-n cuspless at r_iI -> 0.  At coalescence r_ij=r_jI,
            # so coefficients are grouped by the common total power k=m+n:
            #   Sum_{m+n=k} (C gamma_{0,m,n} - L gamma_{1,m,n}) = 0.
            for K in range(N_eN + N_ee + 1):
                row = np.zeros(raw_size, dtype=np.float64)
                for m in range(N_eN + 1):
                    n = K - m
                    if 0 <= n <= N_ee:
                        row[_idx(0, m, n)] += C
                        row[_idx(1, m, n)] -= L_I
                rows.append(row)

            C_mat = np.asarray(rows, dtype=np.float64)
            null_basis = _canonical_null_space_basis(C_mat)  # (raw_size, n_indep)
            M_I = null_basis.reshape((N_eN + 1, N_eN + 1, N_ee + 1, -1))
            M_gamma_list.append(M_I)

        # Per-species null-space dimensions are identical for matching
        # (orders, C, L_I) but a hetero-species cell with different L_I
        # could in principle differ. Pad to the maximum so the stacked
        # array stays rectangular; trailing zero columns are inert.
        n_indep_max = max((m.shape[-1] for m in M_gamma_list), default=0)
        M_gamma = np.zeros(
            (n_species, N_eN + 1, N_eN + 1, N_ee + 1, n_indep_max),
            dtype=np.float64,
        )
        for I, m in enumerate(M_gamma_list):
            M_gamma[I, :, :, :, : m.shape[-1]] = m
    else:
        M_gamma = np.zeros(
            (n_species, N_eN + 1, N_eN + 1, N_ee + 1, 0),
            dtype=np.float64,
        )

    # --- basis pre-normalization (physics-preserving conditioning) -------
    # Fold an analytic per-column scale into each inflation map so the
    # variational coefficients become O(1)-natural variables (``O =
    # dlnPsi/dc`` is O(1) across u / chi / f). See
    # ``_normalize_inflation_columns`` for the rationale (float64
    # conditioning + cancellation when DTN is co-optimised with O(1) blocks).
    # The per-coefficient scale is the radial RMS of its basis factor(s)
    # under the physical ``r^2 dr`` measure (``_dtn_radial_rms``), composed
    # multiplicatively for the f-term's two e-n cutoff factors and one bare
    # e-e factor. Using the radial RMS (not the cutoff peak ``L^C``) is what
    # makes the single- vs double-cutoff channels land on the same O(1)
    # scale: the peak over-weights the f double cutoff quadratically.
    #   u / chi (single cutoff):  scale_k       = rms[(L-r)^C r^k]
    #   f       (double cutoff):  scale_{l,m,n} = rms[(L-r)^C r^l]
    #                                            * rms[(L-r)^C r^m]
    #                                            * rms[r^n]          (bare e-e)
    if normalize_basis:
        if use_u and M_alpha.shape[-1] > 0:
            scale_u = np.array([_dtn_radial_rms(C, k, L_u, with_cutoff=True) for k in range(N_u + 1)])
            scale_alpha = np.broadcast_to(scale_u, (2, N_u + 1))
            M_alpha = _normalize_inflation_columns(M_alpha, scale_alpha)
        if use_chi and M_beta.shape[-1] > 0:
            scale_beta = np.stack(
                [
                    np.array([_dtn_radial_rms(C, k, float(L_chi[I]), with_cutoff=True) for k in range(N_chi + 1)])
                    for I in range(n_species)
                ],
                axis=0,
            )  # (n_species, N_chi + 1)
            M_beta = _normalize_inflation_columns(M_beta, scale_beta)
        if use_f and M_gamma.shape[-1] > 0:
            scale_gamma = np.zeros((n_species, N_eN + 1, N_eN + 1, N_ee + 1), dtype=np.float64)
            for I in range(n_species):
                L_I = float(L_f[I])
                rms_en = np.array([_dtn_radial_rms(C, p, L_I, with_cutoff=True) for p in range(N_eN + 1)])  # e-n factor
                rms_ee = np.array([_dtn_radial_rms(C, p, L_I, with_cutoff=False) for p in range(N_ee + 1)])  # bare e-e
                scale_gamma[I] = rms_en[:, None, None] * rms_en[None, :, None] * rms_ee[None, None, :]
            M_gamma = _normalize_inflation_columns(M_gamma, scale_gamma)

    return M_alpha, b_alpha, M_beta, b_beta, M_gamma


def _dtn_static_tuple(a: npt.ArrayLike) -> tuple[_StaticFloatTuple, ...]:
    """Convert an array to a hashable nested tuple for ``pytree_node=False`` storage.

    The constraint-inflation maps (``M_alpha`` etc.) are non-variational static
    metadata, but numpy arrays are unhashable and cannot be JAX static aux_data
    (two same-structure DTN instances would collide the jit treedef). A nested
    tuple of Python floats is hashable and compares by value, and every consumer
    already wraps the field in ``np.asarray`` / ``jnp.asarray`` so it round-trips
    to the original array (constant-folded under jit -- no runtime cost)."""

    def _t(v: _NestedFloatList) -> _StaticFloatTuple:
        return tuple(_t(e) for e in v) if isinstance(v, list) else v

    return _t(np.asarray(a).tolist())


@struct.dataclass
class Jastrow_DTN_data:
    r"""Drummond--Towler--Needs polynomial Jastrow (J1 + J2 + J3 family).

    A single cutoff-polynomial container carrying the u-term (electron-electron
    J2), chi-term (electron-nucleus J1) and f-term (electron-electron-nucleus
    J3). Each channel is independently activated by the static ``use_u`` /
    ``use_chi`` / ``use_f`` flags. The DTN term is mutually exclusive with the
    Pade J1 / J2 + analytic J3 family.

    The distance kernel uses the Euclidean norm for molecular systems and the
    minimum-image distance for periodic systems. Periodic cutoffs must be small
    enough that the minimum image is unique (below the Wigner--Seitz inscribed
    radius). Detailed derivation of the polynomial constraints is work in
    progress.
    """

    structure_data: Structure_data = struct.field(
        pytree_node=True, default_factory=Structure_data
    )  #: Nuclear structure data (positions + lattice).
    species_index: tuple[int, ...] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Per-atom index into the unique-species table (length ``natom``).
    z_eff_per_species: tuple[float, ...] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Per-species effective charge ``Z - Z_core`` (length ``n_species``).

    cutoff_smoothness: int = struct.field(pytree_node=False, default=4)  # ``C``
    use_u: bool = struct.field(pytree_node=False, default=True)
    use_chi: bool = struct.field(pytree_node=False, default=True)
    use_f: bool = struct.field(pytree_node=False, default=True)
    apply_en_cusp: bool = struct.field(
        pytree_node=False, default=True
    )  #: Serialization marker; builders always set True and sanity_check rejects False.

    n_order_u: int = struct.field(pytree_node=False, default=8)
    n_order_chi: int = struct.field(pytree_node=False, default=8)
    n_order_eN: int = struct.field(pytree_node=False, default=4)
    n_order_ee: int = struct.field(pytree_node=False, default=4)

    # PBC cutoffs are dimensionless fractions of the channel's safe
    # Wigner--Seitz maximum at the reference geometry.  ``L_*`` remain cached
    # static Bohr values for the hot forward kernels; the stress contract
    # differentiates the equivalent smooth volumetric law L=c*|det(cell)|^(1/3)
    # analytically.  Molecular cutoffs remain absolute because there is no cell.
    cutoff_ratio_u: float = struct.field(pytree_node=False, default=0.0)
    cutoff_ratio_chi: tuple = struct.field(pytree_node=False, default_factory=tuple)
    cutoff_ratio_f: tuple = struct.field(pytree_node=False, default_factory=tuple)
    # Cached cutoff lengths and constraint maps.
    L_u: float = struct.field(pytree_node=False, default_factory=lambda: 5.0)  #: Scalar shared e-e cutoff (Bohr).
    L_chi: tuple = struct.field(
        pytree_node=False, default_factory=lambda: ()
    )  #: Per-species e-n cutoff (Bohr); length ``n_species``.
    L_f: tuple = struct.field(
        pytree_node=False, default_factory=lambda: ()
    )  #: Per-species e-e-n cutoff (Bohr); length ``n_species``.
    # Integer shell ``(N_a, N_b, N_c)`` searched after the parallelepiped
    # minimum-image wrap in the basis ``mic_basis_change @ cell``
    # (``_dtn_minimum_image``), static so the kernels unroll it at trace time.
    # ``build`` sets both fields with ``structure._minimum_image_reduction_plan``
    # so the minimum image is exact below the WS inscribed radius on any cell
    # (issue #68); ``(0, 0, 0)`` with the identity basis (molecular,
    # orthorhombic, and checkpoints written before these fields) is the plain
    # wrap. ``sanity_check`` rejects a shell too small for the stored cutoffs.
    mic_shell_half_widths: tuple = struct.field(pytree_node=False, default=(0, 0, 0))
    # Integer unimodular ``M``: the kernels wrap in the basis ``M @ cell`` of
    # the same lattice (identity unless the given basis is far from reduced;
    # ``structure._minimum_image_reduction_plan``), and the shell above is
    # sized for that basis.
    mic_basis_change: tuple = struct.field(pytree_node=False, default=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    # Comoving cutoff scale ``ell_V / ell_V^0`` (pytree LEAF, default ``1.0``). Every stored /
    # optimised state carries ``1.0``; ``Jastrow_data._apply_strain`` sets ``det(F)^(1/3)`` so that
    # autograd / finite differences through a strained Hamiltonian follow the comoving cutoff
    # policy: all kernels read cutoffs through ``_dtn_effective_cutoffs`` (``L * cutoff_scale``)
    # and ``_dtn_inflate_*`` transport the full coefficients to first order along the
    # minimum-norm constraint tangent (``policy_T_* f + policy_q_*``) times ``ln(cutoff_scale)``.
    # At ``cutoff_scale == 1.0`` both reduce bitwise to the plain cutoffs / inflation.
    cutoff_scale: float = struct.field(pytree_node=True, default=1.0)

    # Independent polynomial coefficients. Shapes set by ``build``; the
    # last axis lives in the variational subspace produced by the cusp /
    # symmetry constraints, the leading axes are spin (alpha) / species
    # (beta, gamma).
    alpha_indep: np.ndarray = struct.field(pytree_node=True, default_factory=lambda: np.zeros((2, 0), dtype=np.float64))
    beta_indep: np.ndarray = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    gamma_indep: np.ndarray = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0, 0), dtype=np.float64))

    # Precomputed constraint inflation maps. Linear in the independent
    # coefficients, with biases ``b_*`` carrying the inhomogeneous cusp
    # contributions. ``pytree_node=False`` -- they are *constraint-structure*
    # metadata derived from (orders, cutoffs, smoothness, en-cusp policy),
    # not variational. See ``_build_dtn_constraint_matrices``.
    # Stored as hashable nested tuples (see ``_dtn_static_tuple``): numpy arrays
    # are unhashable and so cannot live in a ``pytree_node=False`` static field.
    # Consumers wrap them in ``np.asarray`` / ``jnp.asarray`` (unchanged).
    M_alpha: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: _dtn_static_tuple(np.zeros((2, 1, 0), dtype=np.float64)),
    )
    b_alpha: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: _dtn_static_tuple(np.zeros((2, 1), dtype=np.float64)),
    )
    M_beta: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1, 0), dtype=np.float64)),
    )
    b_beta: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1), dtype=np.float64)),
    )
    M_gamma: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1, 1, 1, 0), dtype=np.float64)),
    )
    # Full-coefficient transport derivative ``f_dot=T f+q`` for a common
    # logarithmic PBC cutoff dilation.  These maps are static, while ``f`` is
    # reconstructed from the current variational parameters on every call.
    policy_T_alpha: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((2, 1, 1), dtype=np.float64))
    )
    policy_q_alpha: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((2, 1), dtype=np.float64))
    )
    policy_T_beta: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1, 1), dtype=np.float64))
    )
    policy_q_beta: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1), dtype=np.float64))
    )
    policy_T_gamma: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1, 1), dtype=np.float64))
    )
    policy_q_gamma: tuple = struct.field(
        pytree_node=False, default_factory=lambda: _dtn_static_tuple(np.zeros((0, 1), dtype=np.float64))
    )

    def sanity_check(self) -> None:
        """Verify shapes, smoothness and cutoff geometric bounds."""
        if not self.apply_en_cusp:
            raise ValueError(
                "Legacy J_DTN data with apply_en_cusp=False is unsafe: a free "
                "chi linear coefficient creates a nuclear kink, so the raw local "
                "energy contains -chi'(0)/r and its variance can diverge "
                "(observed 357518 -> 0.687 Ha^2 after cusp projection). Regenerate "
                "the Hamiltonian with this jQMC version and rerun VMC optimization. "
                "See doc/notes/wavefunction.md, 'Drummond-Towler-Needs (DTN) Jastrow'."
            )
        if self.cutoff_smoothness < 3:
            raise ValueError(
                f"cutoff_smoothness = {self.cutoff_smoothness} must be >= 3 "
                "(C^2 is the minimum the Laplacian-based kinetic estimator needs)."
            )
        # Lower bound: a cutoff L <= 0 makes its channel identically zero (issue #589).
        for name, values in (("L_u", self.L_u), ("L_chi", self.L_chi), ("L_f", self.L_f)):
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(arr) & (arr > 0.0)):
                raise ValueError(f"{name} = {arr.tolist()} must contain finite lengths > 0 (Bohr).")
        self._check_cusp_bias_signs()
        n_atom = len(self.structure_data.positions)
        if len(self.species_index) != n_atom:
            raise ValueError(f"species_index length {len(self.species_index)} != n_atom {n_atom}.")
        n_species = len(self.z_eff_per_species)
        if n_species > 0 and max(self.species_index, default=-1) >= n_species:
            raise ValueError("species_index contains an entry >= len(z_eff_per_species).")
        if np.asarray(self.L_chi).shape != (n_species,):
            raise ValueError(f"L_chi shape {np.asarray(self.L_chi).shape} != (n_species={n_species},).")
        if np.asarray(self.L_f).shape != (n_species,):
            raise ValueError(f"L_f shape {np.asarray(self.L_f).shape} != (n_species={n_species},).")
        if self.structure_data.pbc_flag:
            if not (0.0 < self.cutoff_ratio_u < 1.0):
                raise ValueError("Periodic cutoff_ratio_u must be strictly between zero and one.")
            for name, ratios in (("cutoff_ratio_chi", self.cutoff_ratio_chi), ("cutoff_ratio_f", self.cutoff_ratio_f)):
                arr = np.asarray(ratios, dtype=np.float64)
                if arr.shape != (n_species,) or np.any(arr <= 0.0) or np.any(arr >= 1.0):
                    raise ValueError(f"Periodic {name} must contain one ratio in (0,1) per species.")
        # Geometric bound on PBC cutoffs: a minimum-image distance is smooth
        # (one unique nearest image) only inside the WS inscribed radius (half
        # the shortest lattice vector), so the hard cutoff must vanish before
        # that sphere. The image search below it is the shell check further down.
        R_inscribed = self.structure_data._wigner_seitz_inscribed_radius()
        if not isinstance(self.cutoff_scale, jax.core.Tracer):
            _scale = np.asarray(self.cutoff_scale, dtype=np.float64)
            if _scale.shape != () or not np.isfinite(_scale) or not (_scale > 0.0):
                raise ValueError(f"cutoff_scale must be a positive finite scalar, got {self.cutoff_scale!r}.")
        L_u_val = float(np.asarray(self.L_u).reshape(()))
        if not (L_u_val < R_inscribed):
            raise ValueError(f"L_u = {L_u_val} must be < R_inscribed = {R_inscribed} to avoid PBC image wrap.")
        for I in range(n_species):
            if not (float(self.L_chi[I]) < R_inscribed):
                raise ValueError(f"L_chi[{I}] = {float(self.L_chi[I])} must be < R_inscribed = {R_inscribed}.")
            # f-term: the pair separation reaches 2*L_f inside the term's
            # support, and the independent minimum-image r_ij (standard
            # CASINO/QMCPACK-compatible DTN convention) is smooth only inside
            # the WS inscribed radius. 2*L_f < R_inscribed is therefore a
            # SMOOTHNESS requirement: violating it puts the min-image seam
            # (a kink of ln Psi) inside the support, breaks the
            # <e_L> = <Psi|H|Psi> identity (the kink's kinetic surface-delta
            # is never sampled) and lets the optimizer collapse the sampled
            # energy below the fixed-node bound (observed on a periodic slab).
            if not (2.0 * float(self.L_f[I]) < R_inscribed):
                raise ValueError(
                    f"L_f[{I}] = {float(self.L_f[I])} must satisfy 2*L_f < R_inscribed = {R_inscribed} "
                    "(minimum-image r_ij must remain smooth throughout the f-term support)."
                )
        half_widths = tuple(int(h) for h in self.mic_shell_half_widths)
        if len(half_widths) != 3 or any(h < 0 for h in half_widths):
            raise ValueError(f"mic_shell_half_widths must be three non-negative ints, got {self.mic_shell_half_widths!r}.")
        basis_change = np.asarray(self.mic_basis_change, dtype=np.float64)
        if (
            basis_change.shape != (3, 3)
            or np.any(basis_change != np.rint(basis_change))
            or abs(round(float(np.linalg.det(basis_change)))) != 1
        ):
            raise ValueError(f"mic_basis_change must be an integer unimodular 3x3 matrix, got {self.mic_basis_change!r}.")
        if self.structure_data.pbc_flag:
            # The kernels' minimum image is the parallelepiped wrap in the basis
            # ``mic_basis_change @ cell`` plus this static shell
            # (``_dtn_minimum_image``). It must recover the true minimum image
            # for every separation a cutoff can reach: L_u, L_chi and the
            # f-term pair reach 2*L_f (issue #68).
            reach = max([L_u_val] + [float(x) for x in self.L_chi] + [2.0 * float(x) for x in self.L_f])
            wrap_basis = basis_change @ np.asarray(self.structure_data.cell, dtype=np.float64)
            required = _minimum_image_shell_half_widths(wrap_basis, reach)
            if any(h < r for h, r in zip(half_widths, required)):
                raise ValueError(
                    f"mic_shell_half_widths = {half_widths} is smaller than {required}, the image shell needed "
                    f"for the cutoff reach {reach} on this cell; rebuild with Jastrow_DTN_data.build."
                )
            # The XLA/NumPy kernels expand this shell eagerly and the Rust kernel
            # loops over the shell of the same wrap basis; reject an infeasible
            # one here instead of attempting it (PR #82).
            _check_minimum_image_box_size(half_widths, "DTN minimum-image shell")
            _check_minimum_image_box_size(_minimum_image_shell_half_widths(wrap_basis), "DTN native minimum-image shell")
        self.structure_data.sanity_check()

    def _check_cusp_bias_signs(self) -> None:
        """Reject serialized cusp biases derived for the wrong ``(r - L)^C`` sign.

        The evaluators use ``(L - r)^C sum_k c_k r^k``, whose slope at ``r = 0``
        is ``L^{C-1} (L c_1 - C c_0)``.  The inflation maps ``M_*`` carry zero
        slope, so the stored biases alone fix the cusp slopes: positive
        (``+1/4`` / ``+1/2``) for the u-term and ``-Z_eff`` (all-electron) or
        exactly zero (pseudized) for the chi-term.  Older builders derived
        the biases for ``(r - L)^C``, which multiplies them by ``(-1)^C``: for
        odd ``C`` such data imposes ``-1/4``, ``-1/2`` and ``+Z_eff``.  This
        is a sign test, not a tolerance test: the stale and the correct slopes
        differ in sign, so no numerical threshold is involved.
        """
        C = int(self.cutoff_smoothness)

        def slope(bias: np.ndarray, L: float) -> np.ndarray:
            return L ** (C - 1) * (L * bias[..., 1] - C * bias[..., 0])

        stale = []
        b_alpha = np.asarray(self.b_alpha, dtype=np.float64)
        if self.use_u and b_alpha.ndim == 2 and b_alpha.shape[1] >= 2:
            ee = slope(b_alpha, float(np.asarray(self.L_u, dtype=np.float64).reshape(())))
            if not np.all(ee > 0.0):
                stale.append(f"e-e cusp slopes {ee.tolist()} (expected +1/4, +1/2)")
        b_beta = np.asarray(self.b_beta, dtype=np.float64)
        if self.use_chi and b_beta.ndim == 2 and b_beta.shape[1] >= 2:
            en = slope(b_beta, np.asarray(self.L_chi, dtype=np.float64).reshape(-1))
            if np.any(en > 0.0):
                stale.append(f"e-n cusp slopes {en.tolist()} (expected -Z_eff or 0)")
        if stale:
            raise ValueError(
                "J_DTN cusp biases have the wrong sign for the (L - r)^C cutoff form: "
                + "; ".join(stale)
                + f". This is serialized odd-C (C={C}) data written by an older jQMC version, whose "
                "cusp constraints were derived for (r - L)^C and imposed the opposite e-e and "
                "e-n slopes. Regenerate the Hamiltonian with this jQMC version and rerun the "
                "VMC optimization."
            )

    @classmethod
    def build(
        cls,
        *,
        structure_data: Structure_data,
        z_eff_per_species: Sequence[float],
        pseudized_per_species: Sequence[bool],
        species_index: Sequence[int] | None = None,
        use_u: bool = True,
        use_chi: bool = True,
        use_f: bool = True,
        cutoff_smoothness: int = 4,
        n_order_u: int = 8,
        n_order_chi: int = 8,
        n_order_eN: int = 4,
        n_order_ee: int = 4,
        cutoff_abs_cap: float = 8.0,
        cutoff_ratio_u: float | None = None,
        cutoff_ratio_chi: float | Sequence[float] | None = None,
        cutoff_ratio_f: float | Sequence[float] | None = None,
        L_u: float | None = None,
        L_chi: float | Sequence[float] | None = None,
        L_f: float | Sequence[float] | None = None,
        random_init: bool = False,
        seed: int | None = None,
        normalize_basis: bool = True,
    ) -> "Jastrow_DTN_data":
        r"""Blessed factory: resolve cutoffs, build constraint data, run sanity_check.

        Periodic inputs use dimensionless ``cutoff_ratio_*`` fractions of the
        channel maximum: ``R_inscribed`` for u/chi and ``R_inscribed/2`` for f,
        where ``R_inscribed`` is the Wigner--Seitz inscribed radius. Default is
        ``DTN_DEFAULT_CUTOFF_RATIO`` = 0.95. Absolute ``L_*`` inputs are rejected
        for PBC; molecular inputs retain absolute ``L_*``.

        ``normalize_basis`` (default ``True``) folds an analytic per-column
        scale into the constraint maps so each variational coefficient is an
        O(1)-natural variable. This is physics-preserving and improves numerical
        conditioning when DTN is co-optimised with O(1)-scale blocks. Pass
        ``False`` to recover the raw cutoff-magnitude basis.

        ``z_eff_per_species`` and ``pseudized_per_species`` are required.
        ``Structure_data`` contains neither ECP core charges nor ECP identity,
        so callers must derive both arrays from the Coulomb/ECP data rather than
        infer pseudization from charge alone. Detailed construction of the
        constraint maps is work in progress.
        """
        # The cusp constraints of an active channel fix its linear coefficient
        # (u, chi: k = 1; f: l = 1 and n = 1), so its polynomial orders must be
        # >= 1; a smaller order is rejected here.
        for active, name, order in (
            (use_u, "n_order_u", n_order_u),
            (use_chi, "n_order_chi", n_order_chi),
            (use_f, "n_order_eN", n_order_eN),
            (use_f, "n_order_ee", n_order_ee),
        ):
            if active and int(order) < 1:
                raise ValueError(
                    f"{name} = {order} must be >= 1 for an active DTN channel (its cusp constraint fixes the linear coefficient)."
                )
        # Default species table = one species per distinct atomic_number.
        atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.int64)
        if species_index is None:
            unique_Z, inverse = np.unique(atomic_numbers, return_inverse=True)
            species_index_arr = inverse.astype(np.int32)
        else:
            species_index_arr = np.asarray(species_index, dtype=np.int32)
        n_species = int(species_index_arr.max(initial=-1)) + 1 if species_index_arr.size > 0 else 0

        z_eff_arr = np.asarray(z_eff_per_species, dtype=np.float64)
        if z_eff_arr.shape != (n_species,):
            raise ValueError(f"z_eff_per_species shape {z_eff_arr.shape} != (n_species={n_species},).")
        pseudized_arr = np.asarray(pseudized_per_species, dtype=np.bool_)
        if pseudized_arr.shape != (n_species,):
            raise ValueError(f"pseudized_per_species shape {pseudized_arr.shape} != (n_species={n_species},).")

        # A species that mixes elements has no single nuclear charge and would
        # silently impose one atom's cusp on another atom's nucleus. Reject it
        # even though ECP identity is supplied independently by the caller.
        for _species in range(n_species):
            _members = np.unique(atomic_numbers[species_index_arr == _species])
            if _members.size == 0:
                raise ValueError(f"species {_species} has no atoms; species_index must label species 0..n_species-1.")
            if _members.size > 1:
                raise ValueError(
                    f"species {_species} mixes atomic numbers {_members.tolist()}; "
                    "one species must be one element, otherwise its electron-nucleus "
                    "cusp and Z_eff are ambiguous."
                )

        if structure_data.pbc_flag:
            if any(value is not None for value in (L_u, L_chi, L_f)):
                raise ValueError(
                    "Absolute periodic DTN cutoffs are forbidden; use cutoff_ratio_u, cutoff_ratio_chi and cutoff_ratio_f."
                )

            def ratio_array(value: float | Sequence[float] | None, default: float) -> np.ndarray:
                if value is None:
                    return np.full(n_species, default, dtype=np.float64)
                if np.ndim(value) == 0:
                    return np.full(n_species, float(value), dtype=np.float64)
                return np.asarray(value, dtype=np.float64)

            ratio_u = float(DTN_DEFAULT_CUTOFF_RATIO if cutoff_ratio_u is None else cutoff_ratio_u)
            ratio_chi = ratio_array(cutoff_ratio_chi, DTN_DEFAULT_CUTOFF_RATIO)
            ratio_f = ratio_array(cutoff_ratio_f, DTN_DEFAULT_CUTOFF_RATIO)
            if not (0.0 < ratio_u < 1.0):
                raise ValueError("cutoff_ratio_u must be in (0,1).")
            if ratio_chi.shape != (n_species,) or np.any(ratio_chi <= 0.0) or np.any(ratio_chi >= 1.0):
                raise ValueError("cutoff_ratio_chi must contain one value in (0,1) per species.")
            if ratio_f.shape != (n_species,) or np.any(ratio_f <= 0.0) or np.any(ratio_f >= 1.0):
                raise ValueError("cutoff_ratio_f must contain one value in (0,1) per species.")
            R_inscribed = float(structure_data._wigner_seitz_inscribed_radius())
            L_u_val = ratio_u * R_inscribed
            L_chi_arr = ratio_chi * R_inscribed
            L_f_arr = ratio_f * (0.5 * R_inscribed)
        else:
            if any(value is not None for value in (cutoff_ratio_u, cutoff_ratio_chi, cutoff_ratio_f)):
                raise ValueError("Molecular DTN cutoffs are absolute; cutoff_ratio_* requires a periodic cell.")
            # A cutoff is a length: the (L - r)^C factor vanishes for r >= L, so
            # L <= 0 would silently remove the channel and its Kato cusp (issue #589).
            if not (np.isfinite(float(cutoff_abs_cap)) and float(cutoff_abs_cap) > 0.0):
                raise ValueError(f"cutoff_abs_cap = {cutoff_abs_cap} must be a finite length > 0 (Bohr).")
            L_u_default, L_chi_default, L_f_default = _dtn_default_cutoffs(
                structure_data, DTN_DEFAULT_CUTOFF_RATIO, cutoff_abs_cap
            )
            L_u_val = float(L_u_default if L_u is None else L_u)
            if L_chi is None:
                L_chi_arr = np.full(n_species, L_chi_default, dtype=np.float64)
            elif np.ndim(L_chi) == 0:
                L_chi_arr = np.full(n_species, float(L_chi), dtype=np.float64)
            else:
                L_chi_arr = np.asarray(L_chi, dtype=np.float64)
            if L_f is None:
                L_f_arr = np.full(n_species, L_f_default, dtype=np.float64)
            elif np.ndim(L_f) == 0:
                L_f_arr = np.full(n_species, float(L_f), dtype=np.float64)
            else:
                L_f_arr = np.asarray(L_f, dtype=np.float64)
            for name, values in (("L_u", np.asarray(L_u_val)), ("L_chi", L_chi_arr), ("L_f", L_f_arr)):
                if not np.all(np.isfinite(values) & (values > 0.0)):
                    raise ValueError(f"{name} = {values.tolist()} must be a finite length > 0 (Bohr).")
            ratio_u = 0.0
            ratio_chi = np.zeros(n_species, dtype=np.float64)
            ratio_f = np.zeros(n_species, dtype=np.float64)

        M_alpha, b_alpha, M_beta, b_beta, M_gamma = _build_dtn_constraint_matrices(
            n_species=n_species,
            z_eff_per_species=z_eff_arr,
            pseudized_per_species=pseudized_arr,
            use_u=use_u,
            use_chi=use_chi,
            use_f=use_f,
            n_order_u=n_order_u,
            n_order_chi=n_order_chi,
            n_order_eN=n_order_eN,
            n_order_ee=n_order_ee,
            L_u=L_u_val,
            L_chi=L_chi_arr,
            L_f=L_f_arr,
            cutoff_smoothness=cutoff_smoothness,
            normalize_basis=normalize_basis,
        )
        Ta, qa, Tb, qb, Tg, qg = _dtn_constraint_transport_tangents(
            n_species=n_species,
            z_eff_per_species=z_eff_arr,
            pseudized_per_species=pseudized_arr,
            use_u=use_u,
            use_chi=use_chi,
            use_f=use_f,
            n_order_u=n_order_u,
            n_order_chi=n_order_chi,
            n_order_eN=n_order_eN,
            n_order_ee=n_order_ee,
            L_u=L_u_val,
            L_chi=L_chi_arr,
            L_f=L_f_arr,
            cutoff_smoothness=cutoff_smoothness,
        )

        # Static minimum-image plan (wrap basis + image shell) for the cell, so
        # the kernels find the true minimum image below the WS inscribed radius
        # (issue #68).
        mic_plan = (
            _minimum_image_reduction_plan(np.asarray(structure_data.cell, dtype=np.float64))
            if structure_data.pbc_flag
            else (((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
        )

        n_alpha_indep = M_alpha.shape[-1]
        n_beta_indep = M_beta.shape[-1]
        n_gamma_indep = M_gamma.shape[-1]

        # Storage dtype of the variational indep coefficients: float64 by
        # construction (_JASTROW_PARAMETER_STORAGE_DTYPE). It matches the
        # apply_block_update and post-load dtype, so iter 1 (post-load) and
        # iter 2+ (post-SR) share the same JIT cache key; every DTN kernel
        # inflates the coefficients in its own zone (issue #781).
        dtype_np_jastrow = _JASTROW_PARAMETER_STORAGE_DTYPE

        if random_init:
            rng = np.random.default_rng(seed)
            alpha_indep = (0.01 * rng.standard_normal((2, n_alpha_indep))).astype(dtype_np_jastrow)
            beta_indep = (0.01 * rng.standard_normal((n_species, n_beta_indep))).astype(dtype_np_jastrow)
            gamma_indep = (0.01 * rng.standard_normal((n_species, n_gamma_indep))).astype(dtype_np_jastrow)
        else:
            alpha_indep = np.zeros((2, n_alpha_indep), dtype=np.float64)
            if n_alpha_indep > 0:
                # Start u from the monotone cusp-satisfying member of the same space:
                # alpha_1 = 0, so u_s(r) = -(c_s L_u / C) (1 - r/L_u)^C. The bare cusp
                # bias alone (alpha_indep = 0) gives u_s = c_s r (L_u - r)^C / L_u^C,
                # whose bump (~0.08 c_s L_u for C = 4) attracts electrons beyond
                # r ~ L_u/(C+1); in large cells (64-atom Si, L_u ~ 10 bohr) the VMC
                # walkers cluster before SR can remove it.
                alpha_indep[:, 0] = -np.asarray(b_alpha)[:, 1] / np.asarray(M_alpha)[:, 1, 0]
            alpha_indep = alpha_indep.astype(dtype_np_jastrow)
            beta_indep = np.zeros((n_species, n_beta_indep), dtype=dtype_np_jastrow)
            gamma_indep = np.zeros((n_species, n_gamma_indep), dtype=dtype_np_jastrow)

        data = cls(
            structure_data=structure_data,
            species_index=tuple(int(x) for x in species_index_arr.tolist()),
            z_eff_per_species=tuple(float(x) for x in z_eff_arr.tolist()),
            cutoff_smoothness=int(cutoff_smoothness),
            use_u=bool(use_u),
            use_chi=bool(use_chi),
            use_f=bool(use_f),
            apply_en_cusp=True,
            n_order_u=int(n_order_u),
            n_order_chi=int(n_order_chi),
            n_order_eN=int(n_order_eN),
            n_order_ee=int(n_order_ee),
            cutoff_ratio_u=float(ratio_u),
            cutoff_ratio_chi=_dtn_static_tuple(ratio_chi),
            cutoff_ratio_f=_dtn_static_tuple(ratio_f),
            L_u=float(np.asarray(L_u_val).reshape(())),
            L_chi=_dtn_static_tuple(L_chi_arr),
            L_f=_dtn_static_tuple(L_f_arr),
            alpha_indep=alpha_indep,
            beta_indep=beta_indep,
            gamma_indep=gamma_indep,
            M_alpha=_dtn_static_tuple(M_alpha),
            b_alpha=_dtn_static_tuple(b_alpha),
            M_beta=_dtn_static_tuple(M_beta),
            b_beta=_dtn_static_tuple(b_beta),
            M_gamma=_dtn_static_tuple(M_gamma),
            policy_T_alpha=_dtn_static_tuple(Ta),
            policy_q_alpha=_dtn_static_tuple(qa),
            policy_T_beta=_dtn_static_tuple(Tb),
            policy_q_beta=_dtn_static_tuple(qb),
            policy_T_gamma=_dtn_static_tuple(Tg),
            policy_q_gamma=_dtn_static_tuple(qg),
            mic_shell_half_widths=mic_plan[1],
            mic_basis_change=mic_plan[0],
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        info = ["**" + self.__class__.__name__]
        info.append(
            f"  terms: use_u={self.use_u}, use_chi={self.use_chi}, use_f={self.use_f}; "
            f"electron-nucleus cusp constrained; C={self.cutoff_smoothness}"
        )
        info.append(f"  orders: N_u={self.n_order_u}, N_chi={self.n_order_chi}, N_eN={self.n_order_eN}, N_ee={self.n_order_ee}")
        info.append(
            f"  cutoffs: L_u={float(np.asarray(self.L_u)):.4f}, "
            f"L_chi={np.asarray(self.L_chi).tolist()}, "
            f"L_f={np.asarray(self.L_f).tolist()}"
        )
        if self.structure_data.pbc_flag:
            info.append(
                f"  cutoff ratios: u={self.cutoff_ratio_u:.4f}, "
                f"chi={np.asarray(self.cutoff_ratio_chi).tolist()}, "
                f"f={np.asarray(self.cutoff_ratio_f).tolist()} (comoving volumetric stress policy)"
            )
        info.append(
            f"  indep coefs: alpha={self.alpha_indep.shape}, beta={self.beta_indep.shape}, gamma={self.gamma_indep.shape}"
        )
        return info

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)


def _dtn_cutoff_scale(data: "Jastrow_DTN_data") -> jax.Array:
    """Comoving cutoff scale ``ell_V / ell_V^0`` as an fp64 scalar (``1.0`` unless strained)."""
    return jnp.asarray(data.cutoff_scale, dtype=jnp.float64).reshape(())


def _dtn_effective_cutoffs(data: "Jastrow_DTN_data", dtype_jnp: jnpt.DTypeLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(L_u, L_chi, L_f)`` actually used by the kernels: the stored cutoffs times
    ``cutoff_scale`` (exact identity at ``cutoff_scale == 1.0``)."""
    s = _dtn_cutoff_scale(data)
    L_u = (jnp.asarray(data.L_u, dtype=jnp.float64).reshape(()) * s).astype(dtype_jnp)
    L_chi = (jnp.asarray(data.L_chi, dtype=jnp.float64) * s).astype(dtype_jnp)
    L_f = (jnp.asarray(data.L_f, dtype=jnp.float64) * s).astype(dtype_jnp)
    return L_u, L_chi, L_f


def _dtn_transport_full(
    data: "Jastrow_DTN_data",
    full: jax.Array,
    T: tuple,
    q: tuple,
    dtype_jnp: jnpt.DTypeLike,
) -> jax.Array:
    """First-order comoving transport ``full + ln(cutoff_scale) * (T full + q)`` (periodic only).

    ``T``/``q`` are the static minimum-norm constraint tangents of ``_dtn_constraint_transport_tangents``
    evaluated at the stored cutoffs, so the result is exact to first order in ``ln(cutoff_scale)``,
    which is all a strain derivative at ``F = I`` needs. ``full`` is returned unchanged for molecules
    and whenever the tangent shapes do not match (legacy checkpoints without policy metadata).
    """
    if not data.structure_data.pbc_flag:
        return full
    T_arr = jnp.asarray(T, dtype=dtype_jnp)
    q_arr = jnp.asarray(q, dtype=dtype_jnp)
    lead = full.shape[0]
    flat = full.reshape((lead, -1))
    if T_arr.shape != (lead, flat.shape[1], flat.shape[1]) or q_arr.shape != flat.shape:
        return full
    delta = jnp.log(_dtn_cutoff_scale(data)).astype(dtype_jnp)
    flat = flat + delta * (jnp.einsum("Ikl,Il->Ik", T_arr, flat) + q_arr)
    return flat.reshape(full.shape)


def _dtn_inflate_alpha(data: "Jastrow_DTN_data", dtype_jnp: jnpt.DTypeLike) -> jax.Array:
    """alpha_full[s, k] = M_alpha[s, k, i] alpha_indep[s, i] + b_alpha[s, k] (+ comoving transport)."""
    M = jnp.asarray(data.M_alpha, dtype=dtype_jnp)
    b = jnp.asarray(data.b_alpha, dtype=dtype_jnp)
    indep = jnp.asarray(data.alpha_indep, dtype=dtype_jnp)
    if not data.use_u:
        return jnp.zeros(M.shape[:2], dtype=dtype_jnp)
    full = jnp.einsum("ski,si->sk", M, indep) + b
    return _dtn_transport_full(data, full, data.policy_T_alpha, data.policy_q_alpha, dtype_jnp)


def _dtn_inflate_beta(data: "Jastrow_DTN_data", dtype_jnp: jnpt.DTypeLike) -> jax.Array:
    """beta_full[I, k] = M_beta[I, k, i] beta_indep[I, i] + b_beta[I, k]."""
    M = jnp.asarray(data.M_beta, dtype=dtype_jnp)
    b = jnp.asarray(data.b_beta, dtype=dtype_jnp)
    indep = jnp.asarray(data.beta_indep, dtype=dtype_jnp)
    if not data.use_chi:
        return jnp.zeros(M.shape[:2], dtype=dtype_jnp)
    full = jnp.einsum("Iki,Ii->Ik", M, indep) + b
    return _dtn_transport_full(data, full, data.policy_T_beta, data.policy_q_beta, dtype_jnp)


def _dtn_inflate_gamma(data: "Jastrow_DTN_data", dtype_jnp: jnpt.DTypeLike) -> jax.Array:
    """gamma_full[I, l, m, n] = M_gamma[I, l, m, n, i] gamma_indep[I, i] (homogeneous)."""
    M = jnp.asarray(data.M_gamma, dtype=dtype_jnp)
    indep = jnp.asarray(data.gamma_indep, dtype=dtype_jnp)
    if not data.use_f:
        return jnp.zeros(M.shape[:4], dtype=dtype_jnp)
    full = jnp.einsum("Ilmni,Ii->Ilmn", M, indep)
    return _dtn_transport_full(data, full, data.policy_T_gamma, data.policy_q_gamma, dtype_jnp)


def _dtn_cutoff_policy_full_coefficients(
    data: "Jastrow_DTN_data", dtype_jnp: jnpt.DTypeLike
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``d(alpha,beta,gamma)_full / d ln(ell_V)``.

    The derivative follows the minimum-norm transport of the current full
    coefficient tensors into the neighbouring cutoff-dependent cusp/null
    constraint manifold.  It is identically zero for molecules.
    """
    alpha = _dtn_inflate_alpha(data, dtype_jnp)
    beta = _dtn_inflate_beta(data, dtype_jnp)
    gamma = _dtn_inflate_gamma(data, dtype_jnp)
    if not data.structure_data.pbc_flag:
        return jnp.zeros_like(alpha), jnp.zeros_like(beta), jnp.zeros_like(gamma)
    Ta = jnp.asarray(data.policy_T_alpha, dtype=dtype_jnp)
    qa = jnp.asarray(data.policy_q_alpha, dtype=dtype_jnp)
    Tb = jnp.asarray(data.policy_T_beta, dtype=dtype_jnp)
    qb = jnp.asarray(data.policy_q_beta, dtype=dtype_jnp)
    Tg = jnp.asarray(data.policy_T_gamma, dtype=dtype_jnp)
    qg = jnp.asarray(data.policy_q_gamma, dtype=dtype_jnp)
    alpha_dot = jnp.einsum("skl,sl->sk", Ta, alpha) + qa
    beta_dot = jnp.einsum("Ikl,Il->Ik", Tb, beta) + qb
    gamma_flat = gamma.reshape((gamma.shape[0], -1))
    gamma_dot = (jnp.einsum("Ikl,Il->Ik", Tg, gamma_flat) + qg).reshape(gamma.shape)
    return alpha_dot, beta_dot, gamma_dot


class _DtnMinimumImage(NamedTuple):
    """Static DTN minimum-image plan: wrap basis ``basis_change @ cell`` and the integer shell searched after the wrap."""

    basis_change: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    shifts: tuple[tuple[int, int, int], ...]


_DTN_IDENTITY_BASIS_CHANGE = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _dtn_minimum_image_plan(data: "Jastrow_DTN_data") -> _DtnMinimumImage:
    """Static minimum-image plan of a DTN instance (wrap basis plus image shell).

    Expanded from the static ``mic_basis_change`` / ``mic_shell_half_widths``
    fields (see :py:func:`jqmc.structure._minimum_image_reduction_plan`). A
    molecular structure and an orthorhombic cell get the identity basis and no
    shifts, i.e. the plain wrap.
    """
    if not data.structure_data.pbc_flag:
        return _DtnMinimumImage(_DTN_IDENTITY_BASIS_CHANGE, ())
    basis_change = tuple(tuple(int(x) for x in row) for row in data.mic_basis_change)
    shifts = _minimum_image_shell_shifts(tuple(int(h) for h in data.mic_shell_half_widths))
    return _DtnMinimumImage(basis_change, shifts)


def _dtn_minimum_image(
    diff: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
) -> jax.Array:
    r"""DTN minimum-image displacement of ``diff`` (``(..., 3)``), exact below the WS inscribed radius.

    With ``B = mic.basis_change @ cell`` (the same lattice; the identity for
    every standard cell), first the parallelepiped wrap
    ``d_w = (f - round(f)) @ B`` with ``f = diff @ inv(B)``, then the shortest
    of ``d_w`` and ``d_w + n @ B`` over the static ``mic.shifts``. The plan
    comes from :py:func:`jqmc.structure._minimum_image_reduction_plan`, so every
    displacement whose minimum image is shorter than the WS inscribed radius
    (the bound on every DTN cutoff reach) gets that image (issue #68). The
    wrap alone is exact only below half the smallest interplanar spacing of
    the wrap basis, which is smaller on oblique cells (FCC, BCC, hexagonal,
    non-reduced bases). Beyond the cutoffs the returned image may be longer
    than the minimum, but every lattice-equivalent displacement is at least
    as long as the minimum image, so it stays outside every cutoff and the DTN
    terms still vanish there.

    With the identity basis and no shifts (orthorhombic cells) this is exactly
    the earlier wrap expression. Ties keep the earlier candidate (strict
    ``<``), so the wrapped ``d_w`` wins whenever it is already a minimum
    image. The selected image index is locally constant inside the cutoff, so
    the Euclidean derivative forms of :func:`_dtn_scalar_dist_with_derivs`
    still hold, and AD through the selected candidate (``B`` is ``cell`` times
    a constant integer matrix) is the derivative of the true minimum image.

    Args:
        diff: Cartesian displacements, shape ``(..., 3)``.
        cell: Lattice vectors as rows, fp64 ``(3, 3)``.
        inv_cell: ``inv(cell)``, fp64 ``(3, 3)``.
        mic: Static plan from :func:`_dtn_minimum_image_plan`.

    Returns:
        jax.Array: Minimum-image displacements, fp64, shape of ``diff``.
    """
    if mic.basis_change != _DTN_IDENTITY_BASIS_CHANGE:
        change = np.asarray(mic.basis_change, dtype=np.float64)
        cell = jnp.asarray(change) @ cell
        inv_cell = inv_cell @ jnp.asarray(np.rint(np.linalg.inv(change)))
    frac = diff @ inv_cell
    wrapped = (frac - jnp.round(frac)) @ cell
    if not mic.shifts:
        return wrapped
    best = wrapped
    best_d2 = jnp.sum(wrapped * wrapped, axis=-1)
    for n in mic.shifts:
        candidate = wrapped + (float(n[0]) * cell[0] + float(n[1]) * cell[1] + float(n[2]) * cell[2])
        d2 = jnp.sum(candidate * candidate, axis=-1)
        take = d2 < best_d2
        best = jnp.where(take[..., None], candidate, best)
        best_d2 = jnp.where(take, d2, best_d2)
    return best


def _dtn_minimum_image_np(
    diff: npt.NDArray[np.float64],
    cell: npt.NDArray[np.float64],
    inv_cell: npt.NDArray[np.float64],
    mic: _DtnMinimumImage,
) -> npt.NDArray[np.float64]:
    """NumPy twin of :func:`_dtn_minimum_image` for the ``_debug`` evaluator (``(3,)`` input)."""
    if mic.basis_change != _DTN_IDENTITY_BASIS_CHANGE:
        change = np.asarray(mic.basis_change, dtype=np.float64)
        cell = change @ cell
        inv_cell = inv_cell @ np.rint(np.linalg.inv(change))
    frac = np.asarray(diff, dtype=np.float64) @ inv_cell
    wrapped = (frac - np.round(frac)) @ cell
    best, best_d2 = wrapped, float(wrapped @ wrapped)
    for n in mic.shifts:
        candidate = wrapped + np.asarray(n, dtype=np.float64) @ cell
        d2 = float(candidate @ candidate)
        if d2 < best_d2:
            best, best_d2 = candidate, d2
    return best


def compute_Jastrow_DTN(
    data: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    zone: str = "jastrow_eval",
) -> jax.Array:
    """Evaluate the DTN polynomial Jastrow ``J = J_u + J_chi + J_f``.

    Returns the scalar log-amplitude *before* exponentiation; callers
    attach ``exp(J)`` to the wavefunction (matching ``compute_Jastrow_part``).
    Gradients / Laplacians are produced by AD over this value
    (``_compute_grads_and_laplacian_Jastrow_DTN_auto``). ``zone`` selects the
    precision zone of the evaluation, coefficients included (default
    ``"jastrow_eval"``; the whole-value ratio routes pass ``"jastrow_ratio"``).
    """
    dtype_jnp = get_dtype_jnp(zone)
    sd = data.structure_data
    C = data.cutoff_smoothness

    # Inflate constrained -> full coefficient tensors.
    alpha_full = _dtn_inflate_alpha(data, dtype_jnp)
    beta_full = _dtn_inflate_beta(data, dtype_jnp)
    gamma_full = _dtn_inflate_gamma(data, dtype_jnp)

    # Distance kernel dispatch.
    mic = _dtn_minimum_image_plan(data)
    if sd.pbc_flag:
        cell = sd._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)
        r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
        r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

        # Wrap-policy compliance: the minimum-image distance depends only on
        # ``frac(a - b)`` -- translating ``a`` or ``b`` by a lattice vector
        # shifts ``u_frac`` by an integer, which ``round`` removes -- so input
        # wrapping is a no-op. See the matching note in ``compute_jastrow_one_body``.
        def r_fn(a: jax.Array, b: jax.Array) -> jax.Array:
            # True minimum-image distance. DTN's hard ``(L - r)^C``
            # cutoff needs the real interparticle distance, NOT the Williamson
            # smooth metric (which shares J1/J2's short-range distortion on
            # non-orthorhombic cells); valid because every cutoff ``L <
            # R_inscribed`` pins a unique image and ``_dtn_minimum_image``
            # recovers it on any cell (wrap + static shell, issue #68).
            # Reconstruct in fp64 via the always-fp64 ``inv_cell``
            # (Principle 3a/3b). The image selection is fp64; the Euclidean
            # block (d2, sqrt) runs in the jastrow_eval zone dtype, exactly like
            # ``_dtn_scalar_dist_with_derivs`` and the Rust value kernel.
            u_min = _dtn_minimum_image(a - b, cell, inv_cell, mic).astype(dtype_jnp)
            d2 = jnp.dot(u_min, u_min)
            # Safe sqrt: keep the backward clean at coincident samples (the AD
            # oracle ``_compute_grads_and_laplacian_Jastrow_DTN_auto``
            # differentiates this value), mirroring ``_smooth_periodic_distance``.
            d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
            return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)
    else:
        r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
        r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

        def r_fn(a: jax.Array, b: jax.Array) -> jax.Array:
            # Safe sqrt (double-where) mirroring the PBC branch above: a
            # coincident sample (``a == b`` -> ``d2 == 0``, e.g. a masked
            # self-pair reached by the ECP mesh / ratio path) must keep the
            # backward AD-clean. A bare ``jnp.linalg.norm`` leaks a ``0/0``
            # gradient through ``sqrt`` that survives the downstream mask
            # (``0 * NaN == NaN``) and NaNs the ``d e_L / d r`` / stress paths.
            u = (a - b).astype(dtype_jnp)
            d2 = jnp.dot(u, u)
            d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
            return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)

    L_u = _dtn_effective_cutoffs(data, dtype_jnp)[0]
    L_chi = _dtn_effective_cutoffs(data, dtype_jnp)[1]
    L_f = _dtn_effective_cutoffs(data, dtype_jnp)[2]
    R_carts = sd._positions_cart_jnp.astype(dtype_jnp)
    species_idx = jnp.asarray(data.species_index, dtype=jnp.int32)

    J = jnp.zeros((), dtype=dtype_jnp)

    # Probe-only min-image distance: mirrors ``r_fn`` bit-for-bit but threads
    # cell/inv through args, so the measured-budget probes below never capture an
    # outer tracer (``measured_bytes_per_config`` lowers them on avals). On the
    # molecular branch cell/inv are unused ``eye(3)`` placeholders; ``sd.pbc_flag``
    # and ``dtype_jnp`` are static.
    _probe_cell = cell if sd.pbc_flag else jnp.eye(3, dtype=jnp.float64)
    _probe_inv = inv_cell if sd.pbc_flag else jnp.eye(3, dtype=jnp.float64)

    def _probe_rfn(
        a: jax.Array,
        b: jax.Array,
        cell_p: jax.Array,
        inv_p: jax.Array,
    ) -> jax.Array:
        if sd.pbc_flag:
            u_min = _dtn_minimum_image(a - b, cell_p, inv_p, mic)
            d2 = jnp.dot(u_min, u_min)
        else:
            u = (a - b).astype(dtype_jnp)
            d2 = jnp.dot(u, u)
        d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
        return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)

    # --- u-term (J2 e-e) ---------------------------------------------
    if data.use_u:
        N_u = data.n_order_u
        k_arr_u = jnp.arange(N_u + 1, dtype=dtype_jnp)

        def u_kernel(r_ee: jax.Array, alpha_ch: jax.Array) -> jax.Array:
            cutoff = jnp.maximum(L_u - r_ee, 0.0) ** C
            powers = r_ee**k_arr_u
            return cutoff * jnp.dot(alpha_ch, powers)

        # Measured-budget probe for the per-pair u-workspace (scalar dist +
        # (N_u+1,) power buffer). One kernel shape serves _uu/_dd/_ud; only the
        # pair count n and the (r_a, r_b, alpha_ch) tracers in ``data`` differ.
        def _u_fwd_probe(
            pdata: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            p: jax.Array,
        ) -> jax.Array:
            r_a, r_b, alpha_ch, L_u_p, cell_p, inv_p = pdata
            r_ee = _probe_rfn(r_a[p[0]], r_b[p[1]], cell_p, inv_p)
            cutoff = jnp.maximum(L_u_p - r_ee, 0.0) ** C
            powers = r_ee ** jnp.arange(N_u + 1, dtype=dtype_jnp)
            return cutoff * jnp.dot(alpha_ch, powers)

        N_up = r_up.shape[0]
        N_dn = r_dn.shape[0]
        # A plain vmap over the DTN u-channel pair axis
        # vmap holds every pair's (N_u+1) power buffer + cutoff at once, so peak
        # scales with N_pair (O(N_e^2), measured O(N^2) on silicon: N=4->0.08 MB,
        # N=16->1.27 MB). The measured-budget batch keeps that many pairs live per
        # step and scans the rest; the outer reduction is a plain ``jnp.sum`` over the
        # *materialised* per-pair array (each pair independent), so the value
        # matches the vmap to fp64 rounding (measured exact on silicon, rel
        # 1.2e-16 on water; batch >= N_pair degenerates to the vmap). Mirrors the
        # chi-term stream below and the grad/lap twin per_pair
        # (:~3794). Measured fwd temp silicon 1.27 MB -> 0.05 MB at batch 4.
        if N_up >= 2:
            i_idx, j_idx = jnp.triu_indices(N_up, k=1)

            def _uu(p: jax.Array) -> jax.Array:
                return u_kernel(r_fn(r_up[p[0]], r_up[p[1]]), alpha_full[0])

            pairs = jnp.stack([i_idx, j_idx], axis=1)
            _uu_batch = measured_stream_batch(
                _u_fwd_probe,
                (r_up, r_up, alpha_full[0], L_u, _probe_cell, _probe_inv),
                (jax.ShapeDtypeStruct((2,), pairs.dtype),),
                pairs.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(_uu, pairs, batch_size=_uu_batch))
        if N_dn >= 2:
            i_idx, j_idx = jnp.triu_indices(N_dn, k=1)

            def _dd(p: jax.Array) -> jax.Array:
                return u_kernel(r_fn(r_dn[p[0]], r_dn[p[1]]), alpha_full[0])

            pairs = jnp.stack([i_idx, j_idx], axis=1)
            _dd_batch = measured_stream_batch(
                _u_fwd_probe,
                (r_dn, r_dn, alpha_full[0], L_u, _probe_cell, _probe_inv),
                (jax.ShapeDtypeStruct((2,), pairs.dtype),),
                pairs.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(_dd, pairs, batch_size=_dd_batch))
        if N_up >= 1 and N_dn >= 1:
            i_grid, j_grid = jnp.meshgrid(jnp.arange(N_up), jnp.arange(N_dn), indexing="ij")
            pairs = jnp.stack([i_grid.ravel(), j_grid.ravel()], axis=1)

            def _ud(p: jax.Array) -> jax.Array:
                return u_kernel(r_fn(r_up[p[0]], r_dn[p[1]]), alpha_full[1])

            _ud_batch = measured_stream_batch(
                _u_fwd_probe,
                (r_up, r_dn, alpha_full[1], L_u, _probe_cell, _probe_inv),
                (jax.ShapeDtypeStruct((2,), pairs.dtype),),
                pairs.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(_ud, pairs, batch_size=_ud_batch))

    # --- chi-term (J1 e-n) -------------------------------------------
    if data.use_chi:
        N_chi = data.n_order_chi
        k_arr_chi = jnp.arange(N_chi + 1, dtype=dtype_jnp)

        def chi_atom_contrib(r_e: jax.Array, R_I: jax.Array, sp_I: jax.Array) -> jax.Array:
            r_eI = r_fn(r_e, R_I)
            L_I = L_chi[sp_I]
            cutoff = jnp.maximum(L_I - r_eI, 0.0) ** C
            powers = r_eI**k_arr_chi
            return cutoff * jnp.dot(beta_full[sp_I], powers)

        def chi_electron_contrib(r_e: jax.Array) -> jax.Array:
            return jnp.sum(vmap(chi_atom_contrib, in_axes=(None, 0, 0))(r_e, R_carts, species_idx))

        # Measured-budget probe for the per-electron chi workspace
        # ((N_atom, N_chi+1) block). Static C/N_chi/dtype/pbc closed over; tracers
        # (R_carts, species_idx, beta_full, L_chi, cell) threaded via ``data``.
        def _chi_fwd_probe(
            pdata: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            r_e: jax.Array,
        ) -> jax.Array:
            R_c, sp_idx, beta_f, L_chi_p, cell_p, inv_p = pdata

            def _atom(R_I: jax.Array, sp_I: jax.Array) -> jax.Array:
                r_eI = _probe_rfn(r_e, R_I, cell_p, inv_p)
                cutoff = jnp.maximum(L_chi_p[sp_I] - r_eI, 0.0) ** C
                powers = r_eI ** jnp.arange(N_chi + 1, dtype=dtype_jnp)
                return cutoff * jnp.dot(beta_f[sp_I], powers)

            return jnp.sum(vmap(_atom, in_axes=(0, 0))(R_c, sp_idx))

        # Stream the electron axis like the grad/lap twin (per_up_chi / per_dn_chi
        # in compute_grads_and_laplacian_Jastrow_DTN): a plain vmap holds every
        # electron's (N_atom, N_chi+1) chi intermediate at once. The
        # measured-budget batch keeps that many electrons live per
        # step and scans the rest; electrons are independent and the outer
        # reduction is a plain sum, so only the summation order moves (fp64
        # round-off ~1e-14), and batch >= N_e degenerates to the vmap.
        _chi_pdata = (R_carts, species_idx, beta_full, L_chi, _probe_cell, _probe_inv)
        if r_up.shape[0] > 0:
            _chi_up_batch = measured_stream_batch(
                _chi_fwd_probe,
                _chi_pdata,
                (jax.ShapeDtypeStruct((3,), r_up.dtype),),
                r_up.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(chi_electron_contrib, r_up, batch_size=_chi_up_batch))
        if r_dn.shape[0] > 0:
            _chi_dn_batch = measured_stream_batch(
                _chi_fwd_probe,
                _chi_pdata,
                (jax.ShapeDtypeStruct((3,), r_dn.dtype),),
                r_dn.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(chi_electron_contrib, r_dn, batch_size=_chi_dn_batch))

    # --- f-term (J3 e-e-n) -------------------------------------------
    if data.use_f:
        N_eN = data.n_order_eN
        N_ee = data.n_order_ee
        k_eN = jnp.arange(N_eN + 1, dtype=dtype_jnp)
        k_ee = jnp.arange(N_ee + 1, dtype=dtype_jnp)

        all_e = jnp.concatenate([r_up, r_dn], axis=0)
        N_tot = all_e.shape[0]
        if N_tot >= 2:
            i_idx, j_idx = jnp.triu_indices(N_tot, k=1)
            pairs = jnp.stack([i_idx, j_idx], axis=1)

            def f_pair_atom(
                r_e1: jax.Array,
                r_e2: jax.Array,
                r_ij: jax.Array,
                R_I: jax.Array,
                sp_I: jax.Array,
            ) -> jax.Array:
                r_iI = r_fn(r_e1, R_I)
                r_jI = r_fn(r_e2, R_I)
                L_I = L_f[sp_I]
                cutoff_i = jnp.maximum(L_I - r_iI, 0.0) ** C
                cutoff_j = jnp.maximum(L_I - r_jI, 0.0) ** C
                pow_i = r_iI**k_eN
                pow_j = r_jI**k_eN
                pow_ij = r_ij**k_ee
                inner = jnp.einsum("lmn,l,m,n->", gamma_full[sp_I], pow_i, pow_j, pow_ij)
                return cutoff_i * cutoff_j * inner

            def f_pair(p: jax.Array) -> jax.Array:
                r_e1 = all_e[p[0]]
                r_e2 = all_e[p[1]]
                r_ij = r_fn(r_e1, r_e2)
                return jnp.sum(vmap(f_pair_atom, in_axes=(None, None, None, 0, 0))(r_e1, r_e2, r_ij, R_carts, species_idx))

            # Measured-budget probe for the per-pair f-workspace ((N_atom, poly^3)
            # contraction). Static C/N_eN/N_ee/dtype/pbc closed over; tracers
            # (all_e, R_carts, species_idx, gamma_full, L_f, cell) via ``data``.
            def _f_fwd_probe(
                pdata: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
                p: jax.Array,
            ) -> jax.Array:
                all_e_p, R_c, sp_idx, gamma_f, L_f_p, cell_p, inv_p = pdata
                r_e1 = all_e_p[p[0]]
                r_e2 = all_e_p[p[1]]
                r_ij = _probe_rfn(r_e1, r_e2, cell_p, inv_p)

                def _atom(R_I: jax.Array, sp_I: jax.Array) -> jax.Array:
                    r_iI = _probe_rfn(r_e1, R_I, cell_p, inv_p)
                    r_jI = _probe_rfn(r_e2, R_I, cell_p, inv_p)
                    L_I = L_f_p[sp_I]
                    cutoff_i = jnp.maximum(L_I - r_iI, 0.0) ** C
                    cutoff_j = jnp.maximum(L_I - r_jI, 0.0) ** C
                    pow_i = r_iI ** jnp.arange(N_eN + 1, dtype=dtype_jnp)
                    pow_j = r_jI ** jnp.arange(N_eN + 1, dtype=dtype_jnp)
                    pow_ij = r_ij ** jnp.arange(N_ee + 1, dtype=dtype_jnp)
                    inner = jnp.einsum("lmn,l,m,n->", gamma_f[sp_I], pow_i, pow_j, pow_ij)
                    return cutoff_i * cutoff_j * inner

                return jnp.sum(vmap(_atom, in_axes=(0, 0))(R_c, sp_idx))

            # Streamed like the grad/lap twin: the f-channel
            # evaluates a (n_atom, poly, poly, poly) contraction per PAIR, and a
            # plain vmap over N_pair = N_tot (N_tot - 1) / 2 holds every pair's
            # block at once -- 12.6 GiB per walker at 108 atoms / 432 electrons.
            # Pairs are independent and the outer reduction is a plain sum, so
            # blocking only changes the summation order (fp64 round-off).
            _f_batch = measured_stream_batch(
                _f_fwd_probe,
                (all_e, R_carts, species_idx, gamma_full, L_f, _probe_cell, _probe_inv),
                (jax.ShapeDtypeStruct((2,), pairs.dtype),),
                pairs.shape[0],
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            J = J + jnp.sum(jax.lax.map(f_pair, pairs, batch_size=_f_batch))

    return J


# ---------------------------------------------------------------------------
# DTN delta ratio (rank-1 fast update) for single-electron moves.
#
# Replaces the brute-force
#   vmap(compute_Jastrow_DTN, in_axes=(None, 0, 0))(dtn, new_r_up_arr, new_r_dn_arr)
# pattern in ``_compute_ratio_Jastrow_part_rank1_update`` /
# ``_compute_ratio_Jastrow_part_split_spin``. Each ECP non-local /
# MCMC-update / LRDMC-projection grid point differs from the reference
# configuration in EXACTLY ONE electron, so the full O(N_e^2 N_atom) DTN
# evaluation per grid is wasteful: only terms involving the moved electron
# change. The delta formulation reduces the per-grid intermediate from
# (N_grid, N_pair=N_e*(N_e-1)/2, N_atom, poly_dims) to
# (N_grid, N_e-1, N_atom, poly_dims) -- a ~N_e/2 factor reduction on the
# huge ``f`` channel intermediates that dominated peak GPU memory on
# silicon (~5 GB savings in ECP non-local, ~9 GB in LRDMC kinetic_disc).
# ---------------------------------------------------------------------------


def _delta_J_DTN_one_electron_moved(
    dtn: "Jastrow_DTN_data",
    r_moved_old: jax.Array,
    r_moved_new: jax.Array,
    same_spin_arr: jax.Array,
    moved_idx_in_same: jax.Array,
    opp_spin_arr: jax.Array,
    dtype_jnp: jnpt.DTypeLike,
) -> jax.Array:
    """``J_DTN(new) - J_DTN(old)`` where exactly one electron moved.

    All channels (u, chi, f) are evaluated only on the pairs / triples that
    involve the moved electron. Other-electron-only contributions cancel
    exactly between old and new.

    Args:
        dtn: ``Jastrow_DTN_data`` instance.
        r_moved_old: ``(3,)`` position of the moved electron BEFORE the move.
        r_moved_new: ``(3,)`` position AFTER the move.
        same_spin_arr: ``(N_same, 3)`` ALL same-spin electron positions
            (the moved electron itself is in here at index ``moved_idx_in_same``;
            its contribution is masked out below to avoid the self pair).
        moved_idx_in_same: scalar int, index of the moved electron within
            ``same_spin_arr`` (so the self pair can be masked).
        opp_spin_arr: ``(N_opp, 3)`` opposite-spin electron positions.
        dtype_jnp: dtype in which the channel contributions are evaluated.

    Returns:
        Scalar ``delta J_DTN``.
    """
    sd = dtn.structure_data
    C = dtn.cutoff_smoothness

    alpha_full = _dtn_inflate_alpha(dtn, dtype_jnp)
    beta_full = _dtn_inflate_beta(dtn, dtype_jnp)
    gamma_full = _dtn_inflate_gamma(dtn, dtype_jnp)

    L_u = _dtn_effective_cutoffs(dtn, dtype_jnp)[0]
    L_chi = _dtn_effective_cutoffs(dtn, dtype_jnp)[1]
    L_f = _dtn_effective_cutoffs(dtn, dtype_jnp)[2]
    R_carts = sd._positions_cart_jnp.astype(dtype_jnp)
    species_idx = jnp.asarray(dtn.species_index, dtype=jnp.int32)
    mic = _dtn_minimum_image_plan(dtn)

    if sd.pbc_flag:
        cell = sd._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)

        # Wrap-policy compliance: the minimum-image distance depends only on
        # ``frac(a - b)`` (a lattice-vector shift of ``a`` or ``b`` is removed
        # by ``round``), so wrapping the moved-electron / same-spin / opp-spin
        # position arrays is a no-op for both the function value and its grad /
        # lap. See the matching note in ``compute_jastrow_one_body``.
        def r_fn(a: jax.Array, b: jax.Array) -> jax.Array:
            # True minimum-image distance (see ``compute_Jastrow_DTN``). The
            # safe sqrt matters here: the same-spin block evaluates the masked
            # self pair ``r_fn(r_moved_old, r_moved_old) = 0`` before the mask
            # zeroes it, so ``u_min = 0`` must stay AD-clean.
            u_min = _dtn_minimum_image(a - b, cell, inv_cell, mic)
            d2 = jnp.dot(u_min, u_min)
            d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
            return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)
    else:

        def r_fn(a: jax.Array, b: jax.Array) -> jax.Array:
            # Safe sqrt (double-where) mirroring the PBC branch above and
            # ``compute_Jastrow_DTN``. The same-spin ratio blocks evaluate the
            # masked self pair ``r_fn(r_moved_old, r_moved_old) = 0`` (and the
            # f-channel ``r_ij_old = 0``) BEFORE the mask zeroes it, so a bare
            # ``jnp.linalg.norm`` leaks a ``0/0`` gradient through ``sqrt`` that
            # survives the mask (``0 * NaN == NaN``) and NaNs the ECP-nonlocal
            # ``d e_L / d r`` / stress autograd paths.
            u = (a - b).astype(dtype_jnp)
            d2 = jnp.dot(u, u)
            d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
            return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)

    delta = jnp.zeros((), dtype=dtype_jnp)

    # -------------------- u channel ----------------------------------------
    if dtn.use_u:
        N_u = dtn.n_order_u
        k_arr_u = jnp.arange(N_u + 1, dtype=dtype_jnp)

        def u_kernel(r_ee: jax.Array, alpha_ch: jax.Array) -> jax.Array:
            cutoff = jnp.maximum(L_u - r_ee, 0.0) ** C
            powers = r_ee**k_arr_u
            return cutoff * jnp.dot(alpha_ch, powers)

        # Same-spin (alpha[0]): mask out self pair (moved electron paired with itself).
        n_same = same_spin_arr.shape[0]
        if n_same > 0:
            same_mask = (jnp.arange(n_same) != moved_idx_in_same).astype(dtype_jnp)

            def _u_same(r_j: jax.Array, m: jax.Array) -> jax.Array:
                r_new = r_fn(r_moved_new, r_j)
                r_old = r_fn(r_moved_old, r_j)
                return m * (u_kernel(r_new, alpha_full[0]) - u_kernel(r_old, alpha_full[0]))

            delta = delta + jnp.sum(jax.vmap(_u_same)(same_spin_arr, same_mask))

        # Opposite-spin (alpha[1]): no self-mask needed.
        n_opp = opp_spin_arr.shape[0]
        if n_opp > 0:

            def _u_opp(r_j: jax.Array) -> jax.Array:
                r_new = r_fn(r_moved_new, r_j)
                r_old = r_fn(r_moved_old, r_j)
                return u_kernel(r_new, alpha_full[1]) - u_kernel(r_old, alpha_full[1])

            delta = delta + jnp.sum(jax.vmap(_u_opp)(opp_spin_arr))

    # -------------------- chi channel --------------------------------------
    # chi depends only on the moved electron's position and atoms; the
    # OTHER electrons' chi contributions cancel between old and new.
    if dtn.use_chi:
        N_chi = dtn.n_order_chi
        k_arr_chi = jnp.arange(N_chi + 1, dtype=dtype_jnp)

        def chi_atom(r_e: jax.Array, R_I: jax.Array, sp_I: jax.Array) -> jax.Array:
            r_eI = r_fn(r_e, R_I)
            L_I = L_chi[sp_I]
            cutoff = jnp.maximum(L_I - r_eI, 0.0) ** C
            powers = r_eI**k_arr_chi
            return cutoff * jnp.dot(beta_full[sp_I], powers)

        def _chi_delta_atom(R_I: jax.Array, sp_I: jax.Array) -> jax.Array:
            return chi_atom(r_moved_new, R_I, sp_I) - chi_atom(r_moved_old, R_I, sp_I)

        delta = delta + jnp.sum(jax.vmap(_chi_delta_atom)(R_carts, species_idx))

    # -------------------- f channel ----------------------------------------
    # Triples (moved, j, I): only triples involving the moved electron change.
    # Iterate j over (other_same + opp) and atom I.
    #
    # Contract the (l, m, n) polynomial indices for the WHOLE (j, atom) block in
    # ONE batched einsum ("Almn,Al,jAm,jn->jA") instead of a
    # vmap(vmap(scalar einsum "lmn,l,m,n->")). The per-instance scalar einsum was
    # lowered by XLA to a standalone transpose of the (n_atom, 5, 5, ..., n_elec)
    # outer product -- the single dominant, memory-bound e_L kernel
    # (loop_transpose_fusion, ~36% of e_L). The batched
    # contraction keeps the (j, atom) instance axes as einsum batch dims, so XLA
    # lowers it to a batched matmul with no separate transpose. Bit-identical up
    # to fp64 round-off (different reduction order in the contraction).
    if dtn.use_f:
        N_eN = dtn.n_order_eN
        N_ee = dtn.n_order_ee
        k_eN = jnp.arange(N_eN + 1, dtype=dtype_jnp)
        k_ee = jnp.arange(N_ee + 1, dtype=dtype_jnp)
        L_atoms = L_f[species_idx]  # (n_atom,)
        gamma_atoms = gamma_full[species_idx]  # (n_atom, l, m, n)

        # Moved electron <-> atoms (shared across all j; new and old positions).
        r_iI_new = jax.vmap(lambda R_I: r_fn(r_moved_new, R_I))(R_carts)  # (n_atom,)
        r_iI_old = jax.vmap(lambda R_I: r_fn(r_moved_old, R_I))(R_carts)  # (n_atom,)
        cutoff_i_new = jnp.maximum(L_atoms - r_iI_new, 0.0) ** C  # (n_atom,)
        cutoff_i_old = jnp.maximum(L_atoms - r_iI_old, 0.0) ** C
        pow_i_new = r_iI_new[:, None] ** k_eN  # (n_atom, N_eN+1)
        pow_i_old = r_iI_old[:, None] ** k_eN

        def _f_delta_block(j_arr: jax.Array, mask: "jax.Array | None") -> jax.Array:
            # j_arr: (n_j, 3) other-electron positions; mask: (n_j,) or None.
            # Other electron <-> atoms (identical for new/old; only i moves).
            r_jI = jax.vmap(lambda r_j: jax.vmap(lambda R_I: r_fn(r_j, R_I))(R_carts))(j_arr)  # (n_j, n_atom)
            cutoff_j = jnp.maximum(L_atoms[None, :] - r_jI, 0.0) ** C  # (n_j, n_atom)
            pow_j = r_jI[..., None] ** k_eN  # (n_j, n_atom, N_eN+1)
            # Moved electron <-> other electron (new vs old).
            r_ij_new = jax.vmap(lambda r_j: r_fn(r_moved_new, r_j))(j_arr)  # (n_j,)
            r_ij_old = jax.vmap(lambda r_j: r_fn(r_moved_old, r_j))(j_arr)
            pow_ij_new = r_ij_new[:, None] ** k_ee  # (n_j, N_ee+1)
            pow_ij_old = r_ij_old[:, None] ** k_ee
            # inner[j, A] = sum_lmn gamma[A,l,m,n] pow_i[A,l] pow_j[j,A,m] pow_ij[j,n]
            inner_new = jnp.einsum("Almn,Al,jAm,jn->jA", gamma_atoms, pow_i_new, pow_j, pow_ij_new)
            inner_old = jnp.einsum("Almn,Al,jAm,jn->jA", gamma_atoms, pow_i_old, pow_j, pow_ij_old)
            new_v = cutoff_i_new[None, :] * cutoff_j * inner_new  # (n_j, n_atom)
            old_v = cutoff_i_old[None, :] * cutoff_j * inner_old
            per_j = jnp.sum(new_v - old_v, axis=1)  # (n_j,)
            if mask is not None:
                per_j = per_j * mask
            return jnp.sum(per_j)

        # Same-spin: mask out self pair.
        n_same = same_spin_arr.shape[0]
        if n_same > 0:
            same_mask = (jnp.arange(n_same) != moved_idx_in_same).astype(dtype_jnp)
            delta = delta + _f_delta_block(same_spin_arr, same_mask)
        # Opposite-spin: no self-mask.
        n_opp = opp_spin_arr.shape[0]
        if n_opp > 0:
            delta = delta + _f_delta_block(opp_spin_arr, None)

    return delta


def _delta_J_DTN_mesh_probe(
    data: tuple["Jastrow_DTN_data", jax.Array, jax.Array],
    new_r_up_one: jax.Array,
    new_r_dn_one: jax.Array,
) -> jax.Array:
    """(data, *per_config) probe form of ``_compute_delta_J_DTN_per_config``.

    ``data = (dtn, old_r_up_carts, old_r_dn_carts)`` (axis-None avals);
    ``new_r_up_one`` / ``new_r_dn_one`` are the streamed per-config coords. Used
    only by ``measured_bytes_per_config`` / ``verified_stream_batch`` to size the
    ECP-grid batch; the map still runs the closure over ``dtn`` for its values.
    """
    dtn, old_r_up_carts, old_r_dn_carts = data
    return _compute_delta_J_DTN_per_config(dtn, old_r_up_carts, old_r_dn_carts, new_r_up_one, new_r_dn_one)


def _compute_delta_J_DTN_per_config(
    dtn: "Jastrow_DTN_data",
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_one: jax.Array,
    new_r_dn_one: jax.Array,
) -> jax.Array:
    """Per-config ``delta J_DTN`` for the mixed up/dn-move case.

    Detects which spin's electron moved via ``jnp.argmax`` on the change
    mask (matches the J1/J2/J3 ``rank1_update`` pattern: exactly one
    electron differs). Computes deltas for both possible cases (cheap --
    each is O(N_e * N_atom)) and selects with ``jnp.where``; vmap-compatible.
    """
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    n_up = old_r_up_carts.shape[0]
    n_dn = old_r_dn_carts.shape[0]

    diff_up = new_r_up_one - old_r_up_carts
    diff_dn = new_r_dn_one - old_r_dn_carts
    moved_up_mask = jnp.any(diff_up != 0.0, axis=1) if n_up > 0 else jnp.zeros((0,), dtype=bool)
    moved_dn_mask = jnp.any(diff_dn != 0.0, axis=1) if n_dn > 0 else jnp.zeros((0,), dtype=bool)
    up_moved_exists = jnp.any(moved_up_mask) if n_up > 0 else jnp.asarray(False)
    idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32)) if n_up > 0 else jnp.asarray(0)
    idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32)) if n_dn > 0 else jnp.asarray(0)

    # Degenerate: one spin has no electrons -> the other spin must have moved.
    if n_up == 0:
        return _delta_J_DTN_one_electron_moved(
            dtn,
            jnp.take(old_r_dn_carts, idx_dn, axis=0),
            jnp.take(new_r_dn_one, idx_dn, axis=0),
            old_r_dn_carts,
            idx_dn,
            old_r_up_carts,
            dtype_jnp,
        )
    if n_dn == 0:
        return _delta_J_DTN_one_electron_moved(
            dtn,
            jnp.take(old_r_up_carts, idx_up, axis=0),
            jnp.take(new_r_up_one, idx_up, axis=0),
            old_r_up_carts,
            idx_up,
            old_r_dn_carts,
            dtype_jnp,
        )

    delta_up_case = _delta_J_DTN_one_electron_moved(
        dtn,
        jnp.take(old_r_up_carts, idx_up, axis=0),
        jnp.take(new_r_up_one, idx_up, axis=0),
        old_r_up_carts,
        idx_up,
        old_r_dn_carts,
        dtype_jnp,
    )
    delta_dn_case = _delta_J_DTN_one_electron_moved(
        dtn,
        jnp.take(old_r_dn_carts, idx_dn, axis=0),
        jnp.take(new_r_dn_one, idx_dn, axis=0),
        old_r_dn_carts,
        idx_dn,
        old_r_up_carts,
        dtype_jnp,
    )
    return jnp.where(up_moved_exists, delta_up_case, delta_dn_case)


def _delta_J_DTN_up_moved_probe(
    data: tuple["Jastrow_DTN_data", jax.Array, jax.Array],
    new_r_up_one: jax.Array,
) -> jax.Array:
    """(data, per_config) probe form of ``_compute_delta_J_DTN_per_config_up_moved``."""
    dtn, old_r_up_carts, old_r_dn_carts = data
    return _compute_delta_J_DTN_per_config_up_moved(dtn, old_r_up_carts, old_r_dn_carts, new_r_up_one)


def _delta_J_DTN_dn_moved_probe(
    data: tuple["Jastrow_DTN_data", jax.Array, jax.Array],
    new_r_dn_one: jax.Array,
) -> jax.Array:
    """(data, per_config) probe form of ``_compute_delta_J_DTN_per_config_dn_moved``."""
    dtn, old_r_up_carts, old_r_dn_carts = data
    return _compute_delta_J_DTN_per_config_dn_moved(dtn, old_r_up_carts, old_r_dn_carts, new_r_dn_one)


def _compute_delta_J_DTN_per_config_up_moved(
    dtn: "Jastrow_DTN_data",
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_one: jax.Array,
) -> jax.Array:
    """Per-config ``delta J_DTN`` for the up-only-moved split block.

    Used by ``_compute_ratio_Jastrow_part_split_spin`` UP block where the
    caller has already partitioned the grid so only up electrons differ --
    no spin dispatch needed (saves one delta-computation branch).
    """
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    diff_up = new_r_up_one - old_r_up_carts
    idx_up = jnp.argmax(jnp.any(diff_up != 0.0, axis=1).astype(jnp.int32))
    return _delta_J_DTN_one_electron_moved(
        dtn,
        jnp.take(old_r_up_carts, idx_up, axis=0),
        jnp.take(new_r_up_one, idx_up, axis=0),
        old_r_up_carts,
        idx_up,
        old_r_dn_carts,
        dtype_jnp,
    )


def _compute_delta_J_DTN_per_config_dn_moved(
    dtn: "Jastrow_DTN_data",
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_dn_one: jax.Array,
) -> jax.Array:
    """Per-config ``delta J_DTN`` for the dn-only-moved split block."""
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    diff_dn = new_r_dn_one - old_r_dn_carts
    idx_dn = jnp.argmax(jnp.any(diff_dn != 0.0, axis=1).astype(jnp.int32))
    return _delta_J_DTN_one_electron_moved(
        dtn,
        jnp.take(old_r_dn_carts, idx_dn, axis=0),
        jnp.take(new_r_dn_one, idx_dn, axis=0),
        old_r_dn_carts,
        idx_dn,
        old_r_up_carts,
        dtype_jnp,
    )


def _compute_Jastrow_DTN_debug(
    data: Jastrow_DTN_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    *,
    zone: str = "jastrow_eval",
) -> float:
    """Pure-numpy oracle: re-derive ``J_u + J_chi + J_f`` from scratch.

    Shares no numerical helper with :func:`compute_Jastrow_DTN`; the constraint
    inflation, distance formulae and term sums are implemented inline so
    a production bug cannot hide in a shared helper. Also serves as the
    brute-force reference for the rank-1 ratio path. Stored constraint and
    transport maps remain shared input data; their construction is not an
    independent physical reference here. ``zone`` selects the accumulation
    precision zone (default ``"jastrow_eval"``); the ratio reference passes
    ``"jastrow_ratio"`` so the J(R') - J(R) cancellation happens in that zone.
    """
    dtype_np = get_dtype_np(zone)
    sd = data.structure_data
    C = int(data.cutoff_smoothness)
    r_up = np.asarray(r_up_carts, dtype=dtype_np)
    r_dn = np.asarray(r_dn_carts, dtype=dtype_np)

    # Inflate independent coefficients via the precomputed maps.
    M_alpha = np.asarray(data.M_alpha, dtype=dtype_np)
    b_alpha = np.asarray(data.b_alpha, dtype=dtype_np)
    M_beta = np.asarray(data.M_beta, dtype=dtype_np)
    b_beta = np.asarray(data.b_beta, dtype=dtype_np)
    M_gamma = np.asarray(data.M_gamma, dtype=dtype_np)
    if data.use_u:
        alpha_full = np.einsum("ski,si->sk", M_alpha, np.asarray(data.alpha_indep, dtype=dtype_np)) + b_alpha
    else:
        alpha_full = np.zeros(M_alpha.shape[:2], dtype=dtype_np)
    if data.use_chi:
        beta_full = np.einsum("Iki,Ii->Ik", M_beta, np.asarray(data.beta_indep, dtype=dtype_np)) + b_beta
    else:
        beta_full = np.zeros(M_beta.shape[:2], dtype=dtype_np)
    if data.use_f:
        gamma_full = np.einsum("Ilmni,Ii->Ilmn", M_gamma, np.asarray(data.gamma_indep, dtype=dtype_np))
    else:
        gamma_full = np.zeros(M_gamma.shape[:4], dtype=dtype_np)

    species_idx = np.asarray(data.species_index, dtype=np.int64)
    R_carts = np.asarray(sd.positions, dtype=dtype_np)
    # Independently apply the stored first-order transport maps. The maps are
    # fixture data, but neither inflation nor transport calls production code.
    _scale = float(np.asarray(data.cutoff_scale, dtype=np.float64).reshape(()))
    if sd.pbc_flag and _scale != 1.0:
        transported = []
        for full, enabled, T, q in (
            (alpha_full, data.use_u, data.policy_T_alpha, data.policy_q_alpha),
            (beta_full, data.use_chi, data.policy_T_beta, data.policy_q_beta),
            (gamma_full, data.use_f, data.policy_T_gamma, data.policy_q_gamma),
        ):
            if enabled:
                flat = full.reshape((full.shape[0], -1))
                tangent = np.asarray(T, dtype=dtype_np)
                offset = np.asarray(q, dtype=dtype_np)
                if tangent.shape == (flat.shape[0], flat.shape[1], flat.shape[1]) and offset.shape == flat.shape:
                    delta = np.asarray(np.log(_scale), dtype=dtype_np)
                    full = (flat + delta * (np.einsum("Ikl,Il->Ik", tangent, flat) + offset)).reshape(full.shape)
            transported.append(full)
        alpha_full, beta_full, gamma_full = transported
    L_u_val = float(np.asarray(data.L_u).reshape(())) * _scale
    L_chi_arr = np.asarray(data.L_chi, dtype=dtype_np) * _scale
    L_f_arr = np.asarray(data.L_f, dtype=dtype_np) * _scale

    if sd.pbc_flag:
        cell = np.asarray([sd.vec_a, sd.vec_b, sd.vec_c], dtype=np.float64)
        inv_cell = np.linalg.inv(cell)
        mic = _dtn_minimum_image_plan(data)

        def r_fn(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
            # True minimum-image distance (numpy oracle for the PBC branch of
            # ``compute_Jastrow_DTN``'s ``r_fn``).
            d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            d_min = _dtn_minimum_image_np(d, cell, inv_cell, mic)
            return float(np.linalg.norm(d_min))

        # Wrap electron positions into the home cell -- matches ``compute_Jastrow_DTN``
        # so the chi / f terms evaluate at the same canonical position.
        def _wrap(arr: np.ndarray) -> np.ndarray:
            frac = arr @ inv_cell
            return (frac - np.floor(frac)) @ cell

        if r_up.shape[0] > 0:
            r_up = _wrap(np.asarray(r_up, dtype=np.float64)).astype(dtype_np)
        if r_dn.shape[0] > 0:
            r_dn = _wrap(np.asarray(r_dn, dtype=np.float64)).astype(dtype_np)
    else:

        def r_fn(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
            return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))

    J = 0.0

    # --- u-term ---
    if data.use_u:
        N_u = data.n_order_u

        def _u_value(r_ee: float, alpha_ch: np.ndarray) -> float:
            if r_ee >= L_u_val:
                return 0.0
            cutoff = (L_u_val - r_ee) ** C
            poly = 0.0
            for k in range(N_u + 1):
                poly += alpha_ch[k] * (r_ee**k)
            return cutoff * poly

        # parallel (uu)
        for i in range(r_up.shape[0]):
            for j in range(i + 1, r_up.shape[0]):
                J += _u_value(r_fn(r_up[i], r_up[j]), alpha_full[0])
        # parallel (dd)
        for i in range(r_dn.shape[0]):
            for j in range(i + 1, r_dn.shape[0]):
                J += _u_value(r_fn(r_dn[i], r_dn[j]), alpha_full[0])
        # antiparallel (ud)
        for i in range(r_up.shape[0]):
            for j in range(r_dn.shape[0]):
                J += _u_value(r_fn(r_up[i], r_dn[j]), alpha_full[1])

    # --- chi-term ---
    if data.use_chi:
        N_chi = data.n_order_chi

        def _chi_value(r: float, beta_I: np.ndarray, L_I: float) -> float:
            if r >= L_I:
                return 0.0
            cutoff = (L_I - r) ** C
            poly = 0.0
            for k in range(N_chi + 1):
                poly += beta_I[k] * (r**k)
            return cutoff * poly

        for elec_set in (r_up, r_dn):
            for i in range(elec_set.shape[0]):
                for I in range(R_carts.shape[0]):
                    sp = int(species_idx[I])
                    r_eI = r_fn(elec_set[i], R_carts[I])
                    J += _chi_value(r_eI, beta_full[sp], float(L_chi_arr[sp]))

    # --- f-term ---
    if data.use_f:
        N_eN = data.n_order_eN
        N_ee = data.n_order_ee
        all_e = np.concatenate([r_up, r_dn], axis=0) if (r_up.shape[0] + r_dn.shape[0]) > 0 else r_up
        N_tot = all_e.shape[0]
        for i in range(N_tot):
            for j in range(i + 1, N_tot):
                r_ij = r_fn(all_e[i], all_e[j])
                for I in range(R_carts.shape[0]):
                    sp = int(species_idx[I])
                    L_I = float(L_f_arr[sp])
                    r_iI = r_fn(all_e[i], R_carts[I])
                    r_jI = r_fn(all_e[j], R_carts[I])
                    if r_iI >= L_I or r_jI >= L_I:
                        continue
                    cutoff_i = (L_I - r_iI) ** C
                    cutoff_j = (L_I - r_jI) ** C
                    inner = 0.0
                    for l in range(N_eN + 1):
                        for m in range(N_eN + 1):
                            for n in range(N_ee + 1):
                                inner += gamma_full[sp, l, m, n] * (r_iI**l) * (r_jI**m) * (r_ij**n)
                    J += cutoff_i * cutoff_j * inner

    return float(J)


@jit
def _compute_grads_and_laplacian_Jastrow_DTN_auto(
    data: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-electron grads / Laplacians of ``J_DTN`` via AD.

    Matches the calling convention of
    ``_compute_grads_and_laplacian_Jastrow_three_body_auto``; the
    Laplacian is the trace of the per-electron Hessian. ``cutoff_smoothness >= 3``
    guarantees the Laplacian is continuous at the cutoff edge.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)
    grad_up = grad(compute_Jastrow_DTN, argnums=1)(data, r_up, r_dn)
    grad_dn = grad(compute_Jastrow_DTN, argnums=2)(data, r_up, r_dn)
    hess_up = hessian(compute_Jastrow_DTN, argnums=1)(data, r_up, r_dn)
    hess_dn = hessian(compute_Jastrow_DTN, argnums=2)(data, r_up, r_dn)
    lap_up = jnp.einsum("ijij->i", hess_up) if r_up.shape[0] > 0 else jnp.zeros((0,), dtype=dtype_jnp)
    lap_dn = jnp.einsum("ijij->i", hess_dn) if r_dn.shape[0] > 0 else jnp.zeros((0,), dtype=dtype_jnp)
    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# Analytic per-electron gradient + Laplacian of J_DTN.
#
# The autodiff reference ``_compute_grads_and_laplacian_Jastrow_DTN_auto`` builds
# the full (3N_e)x(3N_e) Hessian via ``jax.hessian`` and takes its diagonal
# trace as the Laplacian. That tape is O(N_e^4 * N_atom) and explodes XLA
# memory whenever ``grad(compute_local_energy)`` (3rd-order AD) is
# differentiated on top of it -- the J_DTN-only OOM seen on silicon +
# pGTO+B-spline / pGTO+PW geminals.
#
# This analytic path computes per-electron derivatives by chain rule on the
# scalar distance r and the scalar polynomial * cutoff kernel:
#   nabla_i [f(r)] = f'(r) nabla_i r
#   nabla_i^2 [f(r)] = f''(r) |nabla_i r|^2 + f'(r) nabla_i^2 r
# The polynomial * cutoff derivatives f, f', f'' are closed-form (no AD);
# nabla r and nabla^2 r are taken via autodiff on the SCALAR distance
# (tiny 3-var tape -- pure forward pass per pair). Total cost is
# O(N_e^2 * N_atom) -- same scaling as compute_Jastrow_DTN -- with the
# expensive nested AD removed.
#
# Same calling convention as
# ``_compute_grads_and_laplacian_Jastrow_DTN_auto``; the latter is kept as
# the autodiff reference that the analytic path is checked against.
# ---------------------------------------------------------------------------


def _dtn_scalar_dist_with_derivs_auto(
    r_a: jax.Array,
    r_b: jax.Array,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    """Autodiff oracle for :func:`_dtn_scalar_dist_with_derivs`.

    PBC branch takes ``jax.grad`` + ``jax.jvp`` over the scalar
    minimum-image distance to assemble grad and Laplacian. ``round`` has
    zero derivative a.e. (the image index is locally constant inside the
    cutoff), so the AD recovers the Euclidean grad / Laplacian of
    ``u_min`` -- the analytic forms produced directly by
    :func:`_dtn_scalar_dist_with_derivs`. Kept as the reference against
    which that analytic helper is validated (mirrors the kernel-level
    :func:`_dtn_f_atom_value_grad_hess_auto` oracle for the f-channel
    polynomial kernel).
    """
    u = (r_a - r_b).astype(dtype_jnp)
    if pbc_flag:

        def r_of_u(u_in: jax.Array) -> jax.Array:
            # Minimum-image distance, matching ``compute_Jastrow_DTN``'s r_fn.
            u_min = _dtn_minimum_image(u_in, cell, inv_cell, mic)
            d2 = jnp.dot(u_min, u_min)
            d2_safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
            return jnp.where(d2 > 0.0, jnp.sqrt(d2_safe), jnp.zeros_like(d2)).astype(dtype_jnp)

        r = r_of_u(u)
        grad_r = jax.grad(r_of_u)(u)
        g = jax.grad(r_of_u)

        def diag_one(e: jax.Array) -> jax.Array:
            _, hv = jax.jvp(g, (u,), (e,))
            return jnp.dot(hv, e)

        basis = jnp.eye(3, dtype=u.dtype)
        lap_r = jnp.sum(jax.vmap(diag_one)(basis))
    else:
        # Double-where safety (same as :func:`_dtn_scalar_dist_with_derivs`) so
        # higher-order AD through this oracle stays finite at coincident samples.
        d2 = jnp.dot(u, u)
        is_pos = d2 > 0.0
        d2_safe = jnp.where(is_pos, d2, jnp.ones_like(d2))
        r_pos = jnp.sqrt(d2_safe)
        r_den = jnp.where(is_pos, r_pos, jnp.ones_like(d2))
        r = jnp.where(is_pos, r_pos, jnp.zeros_like(d2))
        grad_r = jnp.where(is_pos, u / r_den, jnp.zeros_like(u))
        lap_r = jnp.where(is_pos, jnp.asarray(2.0, dtype=dtype_jnp) / r_den, jnp.zeros_like(d2))
    return r, grad_r, lap_r


def _dtn_scalar_dist_with_derivs(
    r_a: jax.Array,
    r_b: jax.Array,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    r"""Scalar distance ``r = ||r_a - r_b||`` (minimum-image under PBC) plus
    its gradient (3,) and Laplacian (scalar) w.r.t. ``r_a``.

    PBC: the **true minimum-image** distance (DTN's hard
    ``(L - r)^C`` cutoff needs the real interparticle distance, not the
    Williamson smooth metric -- see ``compute_Jastrow_DTN``). Inside the
    cutoff every ``L < R_inscribed`` pins a unique image, which
    :func:`_dtn_minimum_image` returns on any cell (``mic``), so the
    image index is locally constant: ``d(u_min)/d(r_a) = I`` and the distance
    derivatives collapse to the **plain Euclidean** forms

    .. math::
       \nabla_{r_a} r = u_{\min} / r, \qquad \nabla_{r_a}^2 r = 2 / r,

    i.e. identical to the molecular branch on the min-image displacement
    ``u_min`` -- simpler than Williamson smooth-distance closed forms. This
    is an AD-free path: under the
    ``vmap(per_pair) -> vmap(per_atom)`` scaffold of
    :func:`_dtn_f_term_pair_contribution` the helper fires O(N_pair *
    N_atom) times per kinetic-energy eval (~330k calls on a
    96-electron / 36-atom periodic cell).

    Molecular: closed-form Euclidean.

    Safe-at-zero: at ``d^2 == 0`` (a coincident sample, e.g. the u-channel
    self-pair before the mask) ``sqrt(dot(u, u))`` carries a ``0/0``
    gradient that a plain ``maximum(r, eps)`` floor does NOT cure (the
    floor guards the value and the explicit ``1/r`` but not the AD tape
    through ``sqrt``). The where-guard routes the dead branch through
    ``d2_safe == 1`` so every derivative order is clean; the in-limit
    value is ``(r, grad_r, lap_r) = (0, 0, 0)``, keeping the higher-order
    AD paths that wrap this helper (``grad(compute_local_energy)`` etc.)
    finite.
    """
    u = r_a - r_b
    if pbc_flag:
        # Minimum-image displacement (``_dtn_minimum_image``: wrap plus the
        # static shell, exact below the WS inscribed radius on any cell);
        # reconstruct before rounding to the radial evaluation zone.
        u = _dtn_minimum_image(u, cell, inv_cell, mic)
    u = u.astype(dtype_jnp)

    # Euclidean distance r = ||u|| and its grad (3,) / Laplacian (scalar) wrt
    # r_a. PBC feeds the min-image displacement computed above; molecular feeds
    # the raw displacement. Double-where safety keeps every derivative order
    # finite at a coincident sample (``u == 0``) per the docstring.
    d2 = jnp.dot(u, u)
    is_pos = d2 > 0.0
    d2_safe = jnp.where(is_pos, d2, jnp.ones_like(d2))
    r_pos = jnp.sqrt(d2_safe)
    r_den = jnp.where(is_pos, r_pos, jnp.ones_like(d2))  # never 0 -> all-order AD-clean
    r = jnp.where(is_pos, r_pos, jnp.zeros_like(d2))
    grad_r = jnp.where(is_pos, u / r_den, jnp.zeros_like(u))
    lap_r = jnp.where(is_pos, jnp.asarray(2.0, dtype=dtype_jnp) / r_den, jnp.zeros_like(d2))
    return r, grad_r, lap_r


def _dtn_cutoff_with_derivs(
    r: jax.Array,
    L: jax.Array,
    C: int,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    """Closed-form ``cutoff(r) = max(L - r, 0)^C`` and its 1st/2nd derivs."""
    t = L - r
    pos = t > 0.0
    one = jnp.asarray(1.0, dtype=dtype_jnp)
    zero = jnp.asarray(0.0, dtype=dtype_jnp)
    t_safe = jnp.where(pos, t, one)
    cutoff = jnp.where(pos, t_safe**C, zero)
    cutoff_p = jnp.where(pos, -C * t_safe ** (C - 1), zero)
    cutoff_pp = jnp.where(pos, C * (C - 1) * t_safe ** (C - 2), zero)
    return cutoff, cutoff_p, cutoff_pp


def _dtn_poly1d_with_derivs(
    r: jax.Array,
    coeffs: jax.Array,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    """Closed-form ``P(r) = sum_k coeffs[k] r^k`` and its 1st/2nd derivs.

    ``r >= 0`` always (it is a distance). Powers are evaluated safely:
    contributions for ``k=0`` (and ``k=1`` for ``P''``) carry a zero
    coefficient so the ``r^-1`` / ``r^-2`` branches drop out without
    producing NaNs.
    """
    K = coeffs.shape[0]
    k = jnp.arange(K, dtype=dtype_jnp)
    pow_k = r**k
    P = jnp.dot(coeffs, pow_k)
    pow_km1 = r ** jnp.maximum(k - 1, 0)
    P_p = jnp.dot(coeffs * k, pow_km1)
    pow_km2 = r ** jnp.maximum(k - 2, 0)
    P_pp = jnp.dot(coeffs * k * (k - 1), pow_km2)
    return P, P_p, P_pp


def _dtn_radial_kernel_with_derivs(
    r: jax.Array,
    coeffs: jax.Array,
    L: jax.Array,
    C: int,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    """``f(r) = cutoff(r) * P(r)``; returns (f, f', f'') by analytic
    product rule. Shared between the u and chi channels."""
    cutoff, c_p, c_pp = _dtn_cutoff_with_derivs(r, L, C, dtype_jnp)
    P, P_p, P_pp = _dtn_poly1d_with_derivs(r, coeffs, dtype_jnp)
    f = cutoff * P
    fp = c_p * P + cutoff * P_p
    fpp = c_pp * P + 2.0 * c_p * P_p + cutoff * P_pp
    return f, fp, fpp


def _dtn_f_atom_kernel_scalar(
    r_iI: jax.Array,
    r_jI: jax.Array,
    r_ij: jax.Array,
    gamma_I: jax.Array,
    L_I: jax.Array,
    C: int,
    N_eN: int,
    N_ee: int,
    dtype_jnp: jnpt.DTypeLike,
) -> jax.Array:
    """Scalar f-channel pair-atom contribution.

    ``F(r_iI, r_jI, r_ij) = cutoff(r_iI) * cutoff(r_jI) *
                            sum_{l,m,n} gamma_I[l,m,n] r_iI^l r_jI^m r_ij^n``.
    """
    cutoff_i = jnp.maximum(L_I - r_iI, jnp.asarray(0.0, dtype=dtype_jnp)) ** C
    cutoff_j = jnp.maximum(L_I - r_jI, jnp.asarray(0.0, dtype=dtype_jnp)) ** C
    k_eN = jnp.arange(N_eN + 1, dtype=dtype_jnp)
    k_ee = jnp.arange(N_ee + 1, dtype=dtype_jnp)
    pow_i = r_iI**k_eN
    pow_j = r_jI**k_eN
    pow_ij = r_ij**k_ee
    G = jnp.einsum("lmn,l,m,n->", gamma_I, pow_i, pow_j, pow_ij)
    return cutoff_i * cutoff_j * G


def _dtn_f_atom_value_grad_hess_auto(
    r_iI: jax.Array,
    r_jI: jax.Array,
    r_ij: jax.Array,
    gamma_I: jax.Array,
    L_I: jax.Array,
    C: int,
    N_eN: int,
    N_ee: int,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    """Auto-diff kernel-level oracle for the f-channel scalar kernel.

    Returns ``(F, grad_F (3,), hess_F (3, 3))`` via ``jax.grad`` and
    ``jax.hessian`` on the 3-var tape ``_dtn_f_atom_kernel_scalar``. Kept as
    the reference against which the analytic
    ``_dtn_f_atom_value_grad_hess`` is checked (mirrors the role of
    ``_compute_grads_and_laplacian_Jastrow_DTN_auto`` at the per-J level but
    confined to a single (i, j, I) atom-pair contribution -- easier to debug
    when an analytic derivative breaks).
    """

    def k(x: jax.Array) -> jax.Array:
        return _dtn_f_atom_kernel_scalar(x[0], x[1], x[2], gamma_I, L_I, C, N_eN, N_ee, dtype_jnp)

    x = jnp.stack([r_iI, r_jI, r_ij])
    F = k(x)
    grad_F = jax.grad(k)(x)
    hess_F = jax.hessian(k)(x)
    return F, grad_F, hess_F


def _dtn_f_atom_value_grad_hess(
    r_iI: jax.Array,
    r_jI: jax.Array,
    r_ij: jax.Array,
    gamma_I: jax.Array,
    L_I: jax.Array,
    C: int,
    N_eN: int,
    N_ee: int,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnValueGradLap:
    r"""Analytic ``(F, grad_F (3,), hess_F (3, 3))`` for the f-channel scalar
    kernel w.r.t. ``(r_iI, r_jI, r_ij)``.

    The ``jax.hessian`` path on the 3-var tape is the reference
    :func:`_dtn_f_atom_value_grad_hess_auto`. Mirrors the u / chi 1D path
    (:func:`_dtn_radial_kernel_with_derivs`) lifted to the 3-variable
    cutoff-times-polynomial structure

    .. math::
       F(x, y, z) = g(x)\,h(y)\,G(x, y, z),
       \qquad G = \sum_{lmn} \gamma_{lmn}\, x^l y^m z^n

    where ``g(x) = h(x) = cutoff(x; L, C)`` (the cutoff is on the two
    e-N distances only; ``z = r_ij`` is uncut). The 9 partials of ``G``
    are obtained from the same per-axis "stacked power table"
    ``[r^k, k r^{k-1}, k(k-1) r^{k-2}]`` trick as the 1D helpers above,
    contracted with ``gamma_I`` through a single
    ``einsum('lmn,al,bm,cn->abc')`` that yields the full (3, 3, 3) tensor
    of value / 1st / 2nd partials of ``G``. The product-rule combination
    with ``g, g', g''`` (and the symmetric pair for ``y``) then assembles
    ``F``, ``grad_F`` and ``hess_F`` without any AD pass.

    Motivation: an AD path materialises many small intermediate tensors
    under the outer ``vmap(per_pair) -> vmap(per_atom)`` of
    :func:`_dtn_f_term_pair_contribution`, dominating the JDTN kinetic
    energy of a 96-electron / 36-atom PBC cell (~70 % of ``e_L``). The analytic
    formulation removes the second-order AD tape and lets XLA fuse the
    pair-atom kernel as a single dense computation.
    """
    # Cutoffs only on r_iI, r_jI; not on r_ij.
    g_x, gp_x, gpp_x = _dtn_cutoff_with_derivs(r_iI, L_I, C, dtype_jnp)
    g_y, gp_y, gpp_y = _dtn_cutoff_with_derivs(r_jI, L_I, C, dtype_jnp)

    def _power_table(r: jax.Array, K: int) -> jax.Array:
        """``(3, K)`` stack of ``[r^k, k r^{k-1}, k(k-1) r^{k-2}]`` for
        ``k = 0..K-1``. Negative exponents are clamped (the surviving
        coefficient is zero, so the term drops without producing NaNs --
        same safety pattern as :func:`_dtn_poly1d_with_derivs`)."""
        k = jnp.arange(K, dtype=dtype_jnp)
        pow_k = r**k
        pow_km1 = r ** jnp.maximum(k - 1, 0)
        pow_km2 = r ** jnp.maximum(k - 2, 0)
        return jnp.stack([pow_k, k * pow_km1, k * (k - 1) * pow_km2])

    Px = _power_table(r_iI, N_eN + 1)  # (3, N_eN+1)
    Py = _power_table(r_jI, N_eN + 1)  # (3, N_eN+1)
    Pz = _power_table(r_ij, N_ee + 1)  # (3, N_ee+1)

    # T[a, b, c] = sum_{lmn} gamma[l,m,n] * Px[a,l] * Py[b,m] * Pz[c,n].
    # T[0,0,0] = G; T[1,0,0] = dG/dx; T[2,0,0] = d^2G/dx^2; etc.
    T = jnp.einsum("lmn,al,bm,cn->abc", gamma_I, Px, Py, Pz)
    G = T[0, 0, 0]
    G_x, G_y, G_z = T[1, 0, 0], T[0, 1, 0], T[0, 0, 1]
    G_xx, G_yy, G_zz = T[2, 0, 0], T[0, 2, 0], T[0, 0, 2]
    G_xy, G_xz, G_yz = T[1, 1, 0], T[1, 0, 1], T[0, 1, 1]

    # Product rule on F = g(x) h(y) G.
    gh = g_x * g_y
    F = gh * G

    F_x = gp_x * g_y * G + gh * G_x
    F_y = g_x * gp_y * G + gh * G_y
    F_z = gh * G_z
    grad_F = jnp.stack([F_x, F_y, F_z])

    F_xx = gpp_x * g_y * G + 2.0 * gp_x * g_y * G_x + gh * G_xx
    F_yy = g_x * gpp_y * G + 2.0 * g_x * gp_y * G_y + gh * G_yy
    F_zz = gh * G_zz
    F_xy = gp_x * gp_y * G + gp_x * g_y * G_y + g_x * gp_y * G_x + gh * G_xy
    F_xz = gp_x * g_y * G_z + gh * G_xz
    F_yz = g_x * gp_y * G_z + gh * G_yz

    hess_F = jnp.array(
        [
            [F_xx, F_xy, F_xz],
            [F_xy, F_yy, F_yz],
            [F_xz, F_yz, F_zz],
        ]
    )
    return F, grad_F, hess_F


def _dtn_u_term_contributions(
    r_self: jax.Array,
    r_others: jax.Array,
    alpha_ch: jax.Array,
    L_u: jax.Array,
    C: int,
    self_mask: jax.Array,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _PairGradLap:
    """For one electron (``r_self``), accumulate the u-kernel gradient + Laplacian
    contributions from all electrons in ``r_others``. ``self_mask[k] = 1.0``
    means include pair ``(self, r_others[k])``; ``0.0`` masks it out (used for
    same-spin loops to drop the self-pair)."""

    def per_other(r_other: jax.Array, m: jax.Array) -> _PairGradLap:
        r, grad_r, lap_r = _dtn_scalar_dist_with_derivs(r_self, r_other, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        # Replace the (r, grad, lap) with safe placeholders when masked so the
        # downstream r^k / cutoff don't NaN -- the mask zeros the contribution.
        r_safe = jnp.where(m > 0.0, r, jnp.asarray(1.0, dtype=dtype_jnp))
        grad_r_safe = jnp.where(m > 0.0, grad_r, jnp.zeros_like(grad_r))
        lap_r_safe = jnp.where(m > 0.0, lap_r, jnp.asarray(0.0, dtype=dtype_jnp))
        _, fp, fpp = _dtn_radial_kernel_with_derivs(r_safe, alpha_ch, L_u, C, dtype_jnp)
        grad_c = m * fp * grad_r_safe
        lap_c = m * (fpp * jnp.dot(grad_r_safe, grad_r_safe) + fp * lap_r_safe)
        return grad_c, lap_c

    grads, laps = jax.vmap(per_other)(r_others, self_mask)
    return jnp.sum(grads, axis=0), jnp.sum(laps)


def _dtn_chi_term_contributions(
    r_self: jax.Array,
    R_carts: jax.Array,
    species_idx: jax.Array,
    beta_full: jax.Array,
    L_chi: jax.Array,
    C: int,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _PairGradLap:
    """For one electron, accumulate chi-kernel gradient + Laplacian
    contributions from all nuclei."""

    def per_atom(R_I: jax.Array, sp_I: jax.Array) -> _PairGradLap:
        r, grad_r, lap_r = _dtn_scalar_dist_with_derivs(r_self, R_I, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        L_I = L_chi[sp_I]
        _, fp, fpp = _dtn_radial_kernel_with_derivs(r, beta_full[sp_I], L_I, C, dtype_jnp)
        grad_c = fp * grad_r
        lap_c = fpp * jnp.dot(grad_r, grad_r) + fp * lap_r
        return grad_c, lap_c

    grads, laps = jax.vmap(per_atom)(R_carts, species_idx)
    return jnp.sum(grads, axis=0), jnp.sum(laps)


def _dtn_f_term_pair_contribution(
    r_e_i: jax.Array,
    r_e_j: jax.Array,
    R_carts: jax.Array,
    species_idx: jax.Array,
    gamma_full: jax.Array,
    L_f: jax.Array,
    C: int,
    N_eN: int,
    N_ee: int,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _GradLapResult:
    """For the pair (e_i, e_j) -- summed over all nuclei -- return
    ``(grad_i, lap_i, grad_j, lap_j)`` from the f-channel.

    The scalar kernel ``F(r_iI, r_jI, r_ij)`` is autodiffed (3-var tape) for
    its value, 3-vector gradient and 3x3 Hessian. The per-electron
    derivatives are then assembled by chain rule with the geometric
    derivatives ``(grad_iI, lap_iI)``, ``(grad_jI, lap_jI)``,
    ``(grad_ij, lap_ij)``.
    """
    # r_ij and its derivatives are common to all nuclei; compute once.
    r_ij, g_ij_i, lap_ij_i = _dtn_scalar_dist_with_derivs(r_e_i, r_e_j, pbc_flag, cell, inv_cell, mic, dtype_jnp)
    # nabla_j r_ij = -nabla_i r_ij ; Laplacian of r_ij is the same wrt either
    # electron (it depends only on the displacement and the cell metric).
    g_ij_j = -g_ij_i
    lap_ij_j = lap_ij_i

    def per_atom(R_I: jax.Array, sp_I: jax.Array) -> _GradLapResult:
        r_iI, g_iI, lap_iI = _dtn_scalar_dist_with_derivs(r_e_i, R_I, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        r_jI, g_jI, lap_jI = _dtn_scalar_dist_with_derivs(r_e_j, R_I, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        L_I = L_f[sp_I]
        _, grad_F, hess_F = _dtn_f_atom_value_grad_hess(r_iI, r_jI, r_ij, gamma_full[sp_I], L_I, C, N_eN, N_ee, dtype_jnp)
        # Chain rule: F depends on (r_iI, r_jI, r_ij). For nabla_i F:
        #   nabla_i F = F_x * g_iI + F_z * g_ij_i        (F_y branch is constant in r_i)
        # For nabla_j F:
        #   nabla_j F = F_y * g_jI + F_z * g_ij_j
        # Laplacians (sum-of-diagonal-Hessian on r_i, then on r_j) include the
        # mixed terms via hess_F entries.
        F_x, F_y, F_z = grad_F[0], grad_F[1], grad_F[2]
        H_xx, H_yy, H_zz = hess_F[0, 0], hess_F[1, 1], hess_F[2, 2]
        H_xz, H_yz = hess_F[0, 2], hess_F[1, 2]

        grad_i = F_x * g_iI + F_z * g_ij_i
        lap_i = (
            H_xx * jnp.dot(g_iI, g_iI)
            + 2.0 * H_xz * jnp.dot(g_iI, g_ij_i)
            + H_zz * jnp.dot(g_ij_i, g_ij_i)
            + F_x * lap_iI
            + F_z * lap_ij_i
        )

        grad_j = F_y * g_jI + F_z * g_ij_j
        lap_j = (
            H_yy * jnp.dot(g_jI, g_jI)
            + 2.0 * H_yz * jnp.dot(g_jI, g_ij_j)
            + H_zz * jnp.dot(g_ij_j, g_ij_j)
            + F_y * lap_jI
            + F_z * lap_ij_j
        )

        return grad_i, lap_i, grad_j, lap_j

    grads_i, laps_i, grads_j, laps_j = jax.vmap(per_atom)(R_carts, species_idx)
    return (
        jnp.sum(grads_i, axis=0),
        jnp.sum(laps_i),
        jnp.sum(grads_j, axis=0),
        jnp.sum(laps_j),
    )


# ---------------------------------------------------------------------------
# Deduplicated / atom-batched f-channel grad+lap (production fast path).
#
# The reference ``_dtn_f_term_pair_contribution`` fires ``_dtn_scalar_dist_with_derivs``
# TWICE per (pair, atom) (= 2 * N_pair * N_atom; ~330k trig-heavy calls on a
# 96-electron / 36-atom periodic cell) even though only N_e * N_atom (= ~3.5k)
# electron-atom legs are distinct -- a ~95x redundancy. It also builds the
# full (3,3) ``hess_F`` per (pair, atom) via ``jnp.array`` scalar packing and
# computes the cross term ``F_xy`` that the chain rule never consumes.
#
# The fast path below precomputes the per-(electron, atom) "leg" data once
# (cutoff value + 2 derivs, the [r^k, k r^{k-1}, k(k-1) r^{k-2}] power table,
# and the scalar-distance grad/Laplacian), then for each electron pair gathers
# the two legs, batches the gamma einsum over atoms, and assembles the
# per-electron grad/Laplacian with pure array ops (no scalar packing, no F_xy).
# Math is identical to the reference;
# the win is fewer trig evals + XLA-friendlier dense tensor ops, so it
# accelerates BOTH the CPU and the GPU/XLA kinetic-energy path.
# ---------------------------------------------------------------------------


def _dtn_power_table(r: jax.Array, K: int, dtype_jnp: jnpt.DTypeLike) -> jax.Array:
    """``(3, K)`` stack of ``[r^k, k r^{k-1}, k(k-1) r^{k-2}]`` for ``k=0..K-1``.

    Negative exponents are clamped (the surviving coefficient is zero so the
    term drops without NaNs) -- the same safety pattern as the nested
    ``_power_table`` in :func:`_dtn_f_atom_value_grad_hess`.
    """
    k = jnp.arange(K, dtype=dtype_jnp)
    pow_k = r**k
    pow_km1 = r ** jnp.maximum(k - 1, 0)
    pow_km2 = r ** jnp.maximum(k - 2, 0)
    return jnp.stack([pow_k, k * pow_km1, k * (k - 1) * pow_km2])


def _dtn_f_legs(
    all_e: jax.Array,
    R_carts: jax.Array,
    L_f_atoms: jax.Array,
    C: int,
    N_eN: int,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _DtnLeg:
    """Per-(electron, atom) f-channel leg data, deduplicated.

    ``L_f_atoms`` is the per-ATOM e-e-N cutoff (= ``L_f[species_index]``).
    Returns a tuple of arrays each leading-shaped ``(N_e, N_atom, ...)``:
    cutoff ``g`` / ``g'`` / ``g''`` (scalars), the e-N power table ``Px``
    ``(3, N_eN+1)``, the scalar-distance gradient ``gr`` ``(3,)`` and its
    Laplacian ``lap`` (scalar). One leg serves both the i- and j-roles of a
    pair (the cutoff, power table and own-electron distance derivatives are
    identical regardless of which slot the electron occupies).
    """

    def per_e(r_e: jax.Array) -> _DtnLeg:
        def per_atom(R_I: jax.Array, L_I: jax.Array) -> _DtnLeg:
            r_eI, gr_eI, lap_eI = _dtn_scalar_dist_with_derivs(r_e, R_I, pbc_flag, cell, inv_cell, mic, dtype_jnp)
            g, gp, gpp = _dtn_cutoff_with_derivs(r_eI, L_I, C, dtype_jnp)
            Px = _dtn_power_table(r_eI, N_eN + 1, dtype_jnp)  # (3, N_eN+1)
            return g, gp, gpp, Px, gr_eI, lap_eI

        return jax.vmap(per_atom)(R_carts, L_f_atoms)

    return jax.vmap(per_e)(all_e)


def _dtn_f_pair_grad_lap_fast(
    leg_i: _DtnLeg,
    leg_j: _DtnLeg,
    gamma_atoms: jax.Array,
    Pz: jax.Array,
    g_ij_i: jax.Array,
    lap_ij_i: jax.Array,
) -> _GradLapResult:
    """f-channel ``(grad_i, lap_i, grad_j, lap_j)`` for one electron pair from
    precomputed legs, batched over all atoms.

    ``leg_i`` / ``leg_j`` are the gathered ``(N_atom, ...)`` legs of the two
    electrons; ``gamma_atoms`` is ``gamma_full[species_idx]`` ``(N_atom, ...)``;
    ``Pz`` is the shared e-e power table ``(3, N_ee+1)``; ``g_ij_i`` / ``lap_ij_i``
    are the e-e distance grad/Laplacian wrt electron i.
    """
    g_x, gp_x, gpp_x, Px, gr_iI, lap_iI = leg_i
    g_y, gp_y, gpp_y, Py, gr_jI, lap_jI = leg_j
    g_ij_j = -g_ij_i
    lap_ij_j = lap_ij_i

    # T[A,a,b,c] = sum_{lmn} gamma[A,l,m,n] Px[A,a,l] Py[A,b,m] Pz[c,n].
    # In mixed mode this reduction is cancellation-sensitive once the kinetic
    # term is differentiated into a final stress tensor.  The default einsum
    # planner contracts l -> m -> n; explicitly contracting n -> m -> l keeps
    # the same three contractions and the same zone dtype while avoiding that
    # unstable fp32 reduction order.  Do not let planner-version changes choose
    # the failing order again.
    T = jnp.einsum(
        "Almn,Aal,Abm,cn->Aabc",
        gamma_atoms,
        Px,
        Py,
        Pz,
        optimize=[(0, 3), (1, 2), (0, 1)],
    )
    G = T[:, 0, 0, 0]
    G_x, G_y, G_z = T[:, 1, 0, 0], T[:, 0, 1, 0], T[:, 0, 0, 1]
    G_xx, G_yy, G_zz = T[:, 2, 0, 0], T[:, 0, 2, 0], T[:, 0, 0, 2]
    G_xz, G_yz = T[:, 1, 0, 1], T[:, 0, 1, 1]  # G_xy (T[:,1,1,0]) is unused.

    gh = g_x * g_y
    F_x = gp_x * g_y * G + gh * G_x
    F_y = g_x * gp_y * G + gh * G_y
    F_z = gh * G_z
    F_xx = gpp_x * g_y * G + 2.0 * gp_x * g_y * G_x + gh * G_xx
    F_yy = g_x * gpp_y * G + 2.0 * g_x * gp_y * G_y + gh * G_yy
    F_zz = gh * G_zz
    F_xz = gp_x * g_y * G_z + gh * G_xz
    F_yz = g_x * gp_y * G_z + gh * G_yz

    # Chain rule (vectorized over atoms, then summed). nabla_i F = F_x grad(r_iI)
    # + F_z grad(r_ij); nabla_j F = F_y grad(r_jI) + F_z grad(r_ij_j).
    dij_ij = jnp.dot(g_ij_i, g_ij_i)
    grad_i = jnp.sum(F_x[:, None] * gr_iI + F_z[:, None] * g_ij_i[None, :], axis=0)
    dii = jnp.sum(gr_iI * gr_iI, axis=1)
    di_ij = gr_iI @ g_ij_i
    lap_i = jnp.sum(F_xx * dii + 2.0 * F_xz * di_ij + F_zz * dij_ij + F_x * lap_iI + F_z * lap_ij_i)

    grad_j = jnp.sum(F_y[:, None] * gr_jI + F_z[:, None] * g_ij_j[None, :], axis=0)
    djj = jnp.sum(gr_jI * gr_jI, axis=1)
    dj_ij = gr_jI @ g_ij_j
    lap_j = jnp.sum(F_yy * djj + 2.0 * F_yz * dj_ij + F_zz * dij_ij + F_y * lap_jI + F_z * lap_ij_j)

    return grad_i, lap_i, grad_j, lap_j


@jit
def compute_grads_and_laplacian_Jastrow_DTN(
    data: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic per-electron grads and Laplacians of ``J_DTN``.

    Production path for the kinetic-energy estimator and for the de_L/dc
    back-propagation through it; the hessian-based
    ``_compute_grads_and_laplacian_Jastrow_DTN_auto`` is its cross-check
    reference. Evaluated in the ``jastrow_grad_lap`` zone, coefficients
    included.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    return _dtn_grads_and_laplacian_in_dtype(data, r_up_carts, r_dn_carts, dtype_jnp, dtype_jnp)


# Zones whose dtype selects the shared DTN coefficient set of the Rust
# ``JastrowDtnData`` (``dtn.rs``, ``DTN_SHARED_COEFFICIENT_ZONE``): the zone of its
# consumers (gradient / Laplacian, drift, Pulay, strain). The value and ratio
# kernels and the carried grad/lap read their own ``jastrow_eval`` /
# ``jastrow_ratio`` / ``jastrow_grad_lap_carry`` copies, so those zones do not lower
# the shared set, nor does the shared set lower them (issues #766, #774, #781).
_DTN_RUST_COEFFICIENT_ZONES = ("jastrow_grad_lap",)


def _dtn_carry_coefficient_dtype() -> type:
    """Precision of the DTN coefficients the carried DTN grad/lap state consumes.

    The carry consumes its coefficients in its own ``jastrow_grad_lap_carry`` zone
    (float64 in both modes), not in the ``jastrow_grad_lap`` dtype of the Rust
    shared set (:data:`_DTN_RUST_COEFFICIENT_ZONES`), which the carry does not
    consume (issue #781). The Rust carry (``DtnGradLapCache``) reads the
    carry-zone copy of its ``JastrowDtnData`` (``JastrowDtnData::zone``), so both
    backends carry the same quantity.
    """
    return get_dtype_np("jastrow_grad_lap_carry")


def _dtn_zone_coefficients(
    inflate: Callable[["Jastrow_DTN_data", jnpt.DTypeLike], jax.Array],
    data: "Jastrow_DTN_data",
    dtype_jnp: jnpt.DTypeLike,
    coefficient_dtype: jnpt.DTypeLike,
) -> jax.Array:
    """One full DTN coefficient tensor (``_dtn_inflate_alpha|beta|gamma``) for arithmetic in ``dtype_jnp``.

    When ``coefficient_dtype`` is ``dtype_jnp`` (every public zone entry, and the
    carry in full precision) the tensor is inflated in that zone. Otherwise (the
    carry in mixed precision: fp64 arithmetic, fp32 coefficients) it is inflated
    in ``dtype_jnp`` and rounded once to ``coefficient_dtype``. Inflating in fp32
    would make the rounded set depend on how the inflation is compiled (a jitted
    seed and an eagerly traced delta differed by up to 7e-12 in gamma, which the
    carried state accumulated). The fp64 inflation differs between compilation
    paths only at the fp64 roundoff level, which the single rounding to fp32
    removes, so the seed, every delta and a fresh evaluation consume the same
    coefficients.
    """
    if np.dtype(coefficient_dtype) == np.dtype(dtype_jnp):
        return inflate(data, dtype_jnp)
    return inflate(data, dtype_jnp).astype(coefficient_dtype).astype(dtype_jnp)


@jit
def _compute_grads_and_laplacian_Jastrow_DTN_carry(
    data: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Seed of the carried DTN grad/lap state, in the ``jastrow_grad_lap_carry`` zone.

    Same analytic kernel as :func:`compute_grads_and_laplacian_Jastrow_DTN`,
    with coefficients inflated by :func:`_dtn_carry_coefficient_dtype`. In full
    precision both entries are the same computation.
    """
    return _dtn_grads_and_laplacian_in_dtype(
        data, r_up_carts, r_dn_carts, get_dtype_jnp("jastrow_grad_lap_carry"), _dtn_carry_coefficient_dtype()
    )


def _dtn_grads_and_laplacian_in_dtype(
    data: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    dtype_jnp: jnpt.DTypeLike,
    coefficient_dtype: jnpt.DTypeLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Body of the DTN grad/lap entries: arithmetic in ``dtype_jnp``, coefficients
    of precision ``coefficient_dtype`` (:func:`_dtn_zone_coefficients`)."""
    sd = data.structure_data
    C = data.cutoff_smoothness
    N_eN = data.n_order_eN
    N_ee = data.n_order_ee
    mic = _dtn_minimum_image_plan(data)

    alpha_full = _dtn_zone_coefficients(_dtn_inflate_alpha, data, dtype_jnp, coefficient_dtype)
    beta_full = _dtn_zone_coefficients(_dtn_inflate_beta, data, dtype_jnp, coefficient_dtype)
    gamma_full = _dtn_zone_coefficients(_dtn_inflate_gamma, data, dtype_jnp, coefficient_dtype)

    if sd.pbc_flag:
        cell = sd._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)
        # Wrap-policy compliance: ``_dtn_scalar_dist_with_derivs`` reduces
        # ``r_a - r_b`` to its minimum image (``round`` removes lattice-vector
        # offsets), so wrapping the input positions is a no-op for both the
        # value and its grad / lap. Preserve the absolute coordinates until
        # the subtraction and image reconstruction, then round the displacement.
        geometry_dtype = jnp.float64
        pbc_flag = True
    else:
        cell = jnp.eye(3, dtype=jnp.float64)
        inv_cell = jnp.eye(3, dtype=jnp.float64)
        geometry_dtype = dtype_jnp
        pbc_flag = False
    r_up = jnp.asarray(r_up_carts, dtype=geometry_dtype)
    r_dn = jnp.asarray(r_dn_carts, dtype=geometry_dtype)

    L_u = _dtn_effective_cutoffs(data, dtype_jnp)[0]
    L_chi = _dtn_effective_cutoffs(data, dtype_jnp)[1]
    L_f = _dtn_effective_cutoffs(data, dtype_jnp)[2]
    R_carts = sd._positions_cart_jnp.astype(geometry_dtype)
    species_idx = jnp.asarray(data.species_index, dtype=jnp.int32)

    # Measured-budget probes for the streamed grad/lap channels: each calls the
    # same module-level leaf evaluator the map uses, with static C/pbc_flag/dtype
    # closed over and the tracer inputs threaded via ``data`` avals (so the probe
    # lowers without capturing an outer tracer). The maps still stream over
    # electron INDICES; the per-config workspace of the leaf is identical.
    def _u_glap_probe(
        pdata: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        r_self: jax.Array,
    ) -> _PairGradLap:
        r_others, alpha_ch, L_u_p, self_mask, cell_p, inv_p = pdata
        return _dtn_u_term_contributions(
            r_self, r_others, alpha_ch, L_u_p, C, self_mask, pbc_flag, cell_p, inv_p, mic, dtype_jnp
        )

    def _chi_glap_probe(
        pdata: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        r_self: jax.Array,
    ) -> _PairGradLap:
        R_c, sp_idx, beta_f, L_chi_p, cell_p, inv_p = pdata
        return _dtn_chi_term_contributions(r_self, R_c, sp_idx, beta_f, L_chi_p, C, pbc_flag, cell_p, inv_p, mic, dtype_jnp)

    N_up = r_up.shape[0]
    N_dn = r_dn.shape[0]

    grad_up = jnp.zeros((N_up, 3), dtype=dtype_jnp)
    grad_dn = jnp.zeros((N_dn, 3), dtype=dtype_jnp)
    lap_up = jnp.zeros((N_up,), dtype=dtype_jnp)
    lap_dn = jnp.zeros((N_dn,), dtype=dtype_jnp)

    # --- u channel (J2 e-e) --------------------------------------------
    if data.use_u:
        # Same-spin up-up: alpha_full[0]. For each electron i, sum over j != i.
        if N_up >= 2:
            mask_up = 1.0 - jnp.eye(N_up, dtype=dtype_jnp)

            def per_up(idx: jax.Array) -> _PairGradLap:
                return _dtn_u_term_contributions(
                    r_up[idx],
                    r_up,
                    alpha_full[0],
                    L_u,
                    C,
                    mask_up[idx],
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            g_up_uu, l_up_uu = jax.vmap(per_up)(jnp.arange(N_up))
            grad_up = grad_up + g_up_uu
            lap_up = lap_up + l_up_uu
        # Same-spin dn-dn.
        if N_dn >= 2:
            mask_dn = 1.0 - jnp.eye(N_dn, dtype=dtype_jnp)

            def per_dn(idx: jax.Array) -> _PairGradLap:
                return _dtn_u_term_contributions(
                    r_dn[idx],
                    r_dn,
                    alpha_full[0],
                    L_u,
                    C,
                    mask_dn[idx],
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            g_dn_dd, l_dn_dd = jax.vmap(per_dn)(jnp.arange(N_dn))
            grad_dn = grad_dn + g_dn_dd
            lap_dn = lap_dn + l_dn_dd
        # Opposite-spin up-dn: alpha_full[1]. Each up sums over all dn (and vv).
        if N_up >= 1 and N_dn >= 1:
            ones_dn = jnp.ones((N_dn,), dtype=dtype_jnp)

            def per_up_opp(idx: jax.Array) -> _PairGradLap:
                return _dtn_u_term_contributions(
                    r_up[idx],
                    r_dn,
                    alpha_full[1],
                    L_u,
                    C,
                    ones_dn,
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            # Streamed on the same knob as the f-channel pair loop below: each
            # electron's u / chi contribution builds a (N_sig, ...) block, and a
            # plain vmap held every electron's at once -- 95 MiB per walker,
            # 12 GiB at 108 atoms / 432 electrons, nw = 128. Electrons are
            # independent; only the summation order changes (fp64 round-off).
            _up_opp_batch = measured_stream_batch(
                _u_glap_probe,
                (r_dn, alpha_full[1], L_u, ones_dn, cell, inv_cell),
                (jax.ShapeDtypeStruct((3,), r_up.dtype),),
                N_up,
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            g_up_ud, l_up_ud = jax.lax.map(per_up_opp, jnp.arange(N_up), batch_size=_up_opp_batch)
            grad_up = grad_up + g_up_ud
            lap_up = lap_up + l_up_ud

            ones_up = jnp.ones((N_up,), dtype=dtype_jnp)

            def per_dn_opp(idx: jax.Array) -> _PairGradLap:
                return _dtn_u_term_contributions(
                    r_dn[idx],
                    r_up,
                    alpha_full[1],
                    L_u,
                    C,
                    ones_up,
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            _dn_opp_batch = measured_stream_batch(
                _u_glap_probe,
                (r_up, alpha_full[1], L_u, ones_up, cell, inv_cell),
                (jax.ShapeDtypeStruct((3,), r_dn.dtype),),
                N_dn,
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            g_dn_du, l_dn_du = jax.lax.map(per_dn_opp, jnp.arange(N_dn), batch_size=_dn_opp_batch)
            grad_dn = grad_dn + g_dn_du
            lap_dn = lap_dn + l_dn_du

    # --- chi channel (J1 e-N) ------------------------------------------
    if data.use_chi:
        if N_up >= 1:

            def per_up_chi(idx: jax.Array) -> _PairGradLap:
                return _dtn_chi_term_contributions(
                    r_up[idx],
                    R_carts,
                    species_idx,
                    beta_full,
                    L_chi,
                    C,
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            _up_chi_batch = measured_stream_batch(
                _chi_glap_probe,
                (R_carts, species_idx, beta_full, L_chi, cell, inv_cell),
                (jax.ShapeDtypeStruct((3,), r_up.dtype),),
                N_up,
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            g_up_chi, l_up_chi = jax.lax.map(per_up_chi, jnp.arange(N_up), batch_size=_up_chi_batch)
            grad_up = grad_up + g_up_chi
            lap_up = lap_up + l_up_chi
        if N_dn >= 1:

            def per_dn_chi(idx: jax.Array) -> _PairGradLap:
                return _dtn_chi_term_contributions(
                    r_dn[idx],
                    R_carts,
                    species_idx,
                    beta_full,
                    L_chi,
                    C,
                    pbc_flag,
                    cell,
                    inv_cell,
                    mic,
                    dtype_jnp,
                )

            _dn_chi_batch = measured_stream_batch(
                _chi_glap_probe,
                (R_carts, species_idx, beta_full, L_chi, cell, inv_cell),
                (jax.ShapeDtypeStruct((3,), r_dn.dtype),),
                N_dn,
                JASTROW_DTN_PAIR_BUDGET_BYTES,
            )
            g_dn_chi, l_dn_chi = jax.lax.map(per_dn_chi, jnp.arange(N_dn), batch_size=_dn_chi_batch)
            grad_dn = grad_dn + g_dn_chi
            lap_dn = lap_dn + l_dn_chi

    # --- f channel (J3 e-e-N) ------------------------------------------
    if data.use_f:
        all_e = jnp.concatenate([r_up, r_dn], axis=0) if (N_up + N_dn) > 0 else r_up
        N_tot = N_up + N_dn
        if N_tot >= 2:
            gamma_atoms = gamma_full[species_idx]  # (N_atom, N_eN+1, N_eN+1, N_ee+1)
            L_f_atoms = L_f[species_idx]  # (N_atom,)

            # Per-electron f-channel (grad (N_tot,3), lap (N_tot,)) as a differentiable
            # function of (all_e, gamma_atoms, R_c, L_f, cell, inv_cell). ALL differentiable
            # inputs are explicit args so every derivative jQMC takes through the f-channel
            # is faithful: e_L kinetic (d/d ae), SR parameter gradients (d/d gam_a), atomic
            # FORCES (d/d R_c) and stress (d/d cell).
            def _f_channel_ref(
                ae: jax.Array,
                gam_a: jax.Array,
                R_c: jax.Array,
                Lf_c: jax.Array,
                cell_c: jax.Array,
                inv_c: jax.Array,
            ) -> _PairGradLap:
                legs = _dtn_f_legs(ae, R_c, Lf_c, C, N_eN, pbc_flag, cell_c, inv_c, mic, dtype_jnp)
                i_idx, j_idx = jnp.triu_indices(N_tot, k=1)

                def per_pair(p: jax.Array) -> _GradLapResult:
                    i, j = p[0], p[1]
                    r_ij, g_ij_i, lap_ij_i = _dtn_scalar_dist_with_derivs(ae[i], ae[j], pbc_flag, cell_c, inv_c, mic, dtype_jnp)
                    Pz = _dtn_power_table(r_ij, N_ee + 1, dtype_jnp)  # (3, N_ee+1)
                    leg_i = tuple(leaf[i] for leaf in legs)
                    leg_j = tuple(leaf[j] for leaf in legs)
                    return _dtn_f_pair_grad_lap_fast(leg_i, leg_j, gam_a, Pz, g_ij_i, lap_ij_i)

                # Measured-budget probe: same per-pair f-workspace as per_pair, with
                # static pbc_flag/N_ee/dtype closed over and the tracers (ae, legs,
                # gam_a, cell) threaded via ``data`` avals.
                def _f_glap_probe(
                    pdata: tuple[jax.Array, _DtnLeg, jax.Array, jax.Array, jax.Array],
                    p: jax.Array,
                ) -> _GradLapResult:
                    ae_p, legs_p, gam_a_p, cell_p, inv_p = pdata
                    i, j = p[0], p[1]
                    r_ij, g_ij_i, lap_ij_i = _dtn_scalar_dist_with_derivs(
                        ae_p[i], ae_p[j], pbc_flag, cell_p, inv_p, mic, dtype_jnp
                    )
                    Pz = _dtn_power_table(r_ij, N_ee + 1, dtype_jnp)
                    leg_i = tuple(leaf[i] for leaf in legs_p)
                    leg_j = tuple(leaf[j] for leaf in legs_p)
                    return _dtn_f_pair_grad_lap_fast(leg_i, leg_j, gam_a_p, Pz, g_ij_i, lap_ij_i)

                pairs = jnp.stack([i_idx, j_idx], axis=1)
                # Stream the e-e-N (f-channel) over electron pairs with lax.map instead
                # of vmap. vmap materialises every pair's per-atom intermediate
                # simultaneously -- the einsum "Almn,Aal,Abm,cn->Aabc" in
                # ``_dtn_f_pair_grad_lap_fast``
                # blows up to f64[nw, N_pair, N_atom, 3,5,5] (~37 GiB at 128 atoms, nw=32) and
                # OOMs, capping the walker count. The measured-budget batch processes
                # that many pairs per step (vmapped) and scans the rest, so peak memory
                # holds that many pairs' (N_atom, ...) intermediates instead of all
                # N_pair while staying fast. Under the outer per-walker vmap each step
                # also carries the nw axis. Equivalent to the vmap form to fp64 round-off
                # (~1e-14); batch >= N_pair degenerates to the vmap.
                _f_glap_batch = measured_stream_batch(
                    _f_glap_probe,
                    (ae, legs, gam_a, cell_c, inv_c),
                    (jax.ShapeDtypeStruct((2,), pairs.dtype),),
                    pairs.shape[0],
                    JASTROW_DTN_PAIR_BUDGET_BYTES,
                )
                grads_i, laps_i, grads_j, laps_j = jax.lax.map(per_pair, pairs, batch_size=_f_glap_batch)
                grad_pe = jnp.zeros((N_tot, 3), dtype=dtype_jnp)
                lap_pe = jnp.zeros((N_tot,), dtype=dtype_jnp)
                grad_pe = grad_pe.at[i_idx].add(grads_i)
                grad_pe = grad_pe.at[j_idx].add(grads_j)
                lap_pe = lap_pe.at[i_idx].add(laps_i)
                lap_pe = lap_pe.at[j_idx].add(laps_j)
                return grad_pe, lap_pe

            grad_per_e, lap_per_e = _f_channel_ref(all_e, gamma_atoms, R_carts, L_f_atoms, cell, inv_cell)

            grad_up = grad_up + grad_per_e[:N_up]
            grad_dn = grad_dn + grad_per_e[N_up:]
            lap_up = lap_up + lap_per_e[:N_up]
            lap_dn = lap_dn + lap_per_e[N_up:]

    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# Incremental (rank-one) DTN grad/Laplacian advance
#
# Recomputing the WHOLE DTN grad/Laplacian at the new positions every GFMC
# substep is costly: a branching step contains ~220 substeps on a 128-atom
# system (tau=0.05 / alat=0.30). A one-electron move only changes the terms
# CONTAINING the moved electron -- O(N_atom * N_e) instead of
# O(N_atom * N_e^2), a ~65x structural saving paid ~220 times per step.
# This removes a full DTN recomputation from every accepted substep.
# ---------------------------------------------------------------------------


def _dtn_u_pair_terms_per_other(
    r_self: jax.Array,
    r_others: jax.Array,
    coeffs_per_other: jax.Array,
    L_u: jax.Array,
    C: int,
    self_mask: jax.Array,
    pbc_flag: bool,
    cell: jax.Array,
    inv_cell: jax.Array,
    mic: _DtnMinimumImage,
    dtype_jnp: jnpt.DTypeLike,
) -> _PairGradLap:
    """UNSUMMED twin of :func:`_dtn_u_term_contributions`.

    Returns the per-partner ``(grad (N, 3), lap (N,))`` contributions to
    ``r_self`` instead of their sum, because the incremental advance also needs
    each partner's OWN row: for a pair the partner's gradient contribution is
    ``-grad`` (``nabla_j r_ij = -nabla_i r_ij``) and its Laplacian contribution
    is ``+lap`` (``|nabla r|^2`` and ``lap r`` are symmetric in the two
    electrons). ``coeffs_per_other`` is per-partner so the same call covers the
    same-spin and opposite-spin u channels of a single moved electron.
    """

    def per_other(r_other: jax.Array, m: jax.Array, coeffs: jax.Array) -> _PairGradLap:
        r, grad_r, lap_r = _dtn_scalar_dist_with_derivs(r_self, r_other, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        # Masked lanes get safe placeholders so the r^k / cutoff chain cannot
        # produce NaNs that survive the multiplication by a zero mask.
        r_safe = jnp.where(m > 0.0, r, jnp.asarray(1.0, dtype=dtype_jnp))
        grad_r_safe = jnp.where(m > 0.0, grad_r, jnp.zeros_like(grad_r))
        lap_r_safe = jnp.where(m > 0.0, lap_r, jnp.asarray(0.0, dtype=dtype_jnp))
        _, fp, fpp = _dtn_radial_kernel_with_derivs(r_safe, coeffs, L_u, C, dtype_jnp)
        grad_c = m * fp * grad_r_safe
        lap_c = m * (fpp * jnp.dot(grad_r_safe, grad_r_safe) + fp * lap_r_safe)
        return grad_c, lap_c

    return jax.vmap(per_other)(r_others, self_mask, coeffs_per_other)


def _dtn_terms_touching_one(
    data: "Jastrow_DTN_data",
    all_e: jax.Array,
    m_idx: jax.Array,
    r_m: jax.Array,
    num_up: int,
    dtype_jnp: jnpt.DTypeLike,
    coefficient_dtype: jnpt.DTypeLike,
) -> _PairGradLap:
    """Every DTN grad/Laplacian term that CONTAINS electron ``m_idx``, with that
    electron placed at ``r_m``, scattered to the rows of ALL electrons.

    Arithmetic runs in ``dtype_jnp``; the DTN coefficients have precision
    ``coefficient_dtype`` (:func:`_dtn_zone_coefficients`, the caller's zone rule).

    ``all_e`` supplies the partner positions; its own ``m_idx`` row is never
    read (the self pair is masked), so the SAME array serves the "old" and
    "new" evaluations of an advance -- only ``r_m`` differs between them.

    Returns ``(grad (N_e, 3), lap (N_e,))``. Cost ``O(N_atom * N_e)``.
    """
    sd = data.structure_data
    C = data.cutoff_smoothness
    N_eN = data.n_order_eN
    N_ee = data.n_order_ee
    mic = _dtn_minimum_image_plan(data)
    n_e = all_e.shape[0]

    if sd.pbc_flag:
        cell = sd._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)
        pbc_flag = True
    else:
        cell = jnp.eye(3, dtype=jnp.float64)
        inv_cell = jnp.eye(3, dtype=jnp.float64)
        pbc_flag = False

    R_carts = sd._positions_cart_jnp.astype(dtype_jnp)
    species_idx = jnp.asarray(data.species_index, dtype=jnp.int32)
    idx = jnp.arange(n_e)
    notself = idx != m_idx
    w = notself.astype(dtype_jnp)

    grad = jnp.zeros((n_e, 3), dtype=dtype_jnp)
    lap = jnp.zeros((n_e,), dtype=dtype_jnp)

    # --- chi (e-N): a one-electron term, so ONLY the mover's row changes ----
    if data.use_chi:
        beta_full = _dtn_zone_coefficients(_dtn_inflate_beta, data, dtype_jnp, coefficient_dtype)
        L_chi = _dtn_effective_cutoffs(data, dtype_jnp)[1]
        g_chi, l_chi = _dtn_chi_term_contributions(
            r_m, R_carts, species_idx, beta_full, L_chi, C, pbc_flag, cell, inv_cell, mic, dtype_jnp
        )
        grad = grad.at[m_idx].add(g_chi)
        lap = lap.at[m_idx].add(l_chi)

    # --- u (e-e): pairs (m, j) -- mover's row plus one term per partner -----
    if data.use_u and n_e >= 2:
        alpha_full = _dtn_zone_coefficients(_dtn_inflate_alpha, data, dtype_jnp, coefficient_dtype)
        L_u = _dtn_effective_cutoffs(data, dtype_jnp)[0]
        # Spin channel by GLOBAL index parity: same-spin -> alpha_full[0],
        # opposite-spin -> alpha_full[1] (no argmax spin detection).
        same_spin = (idx < num_up) == (m_idx < num_up)
        coeffs = jnp.where(same_spin[:, None], alpha_full[0][None, :], alpha_full[1][None, :])
        g_pair, l_pair = _dtn_u_pair_terms_per_other(r_m, all_e, coeffs, L_u, C, w, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        grad = grad.at[m_idx].add(jnp.sum(g_pair, axis=0))
        lap = lap.at[m_idx].add(jnp.sum(l_pair))
        # Partner rows: gradient flips sign, Laplacian does not.
        grad = grad - g_pair
        lap = lap + l_pair

    # --- f (e-e-N): pairs (m, j) over atoms --------------------------------
    if data.use_f and n_e >= 2:
        gamma_full = _dtn_zone_coefficients(_dtn_inflate_gamma, data, dtype_jnp, coefficient_dtype)
        L_f = _dtn_effective_cutoffs(data, dtype_jnp)[2]
        gamma_atoms = gamma_full[species_idx]
        L_f_atoms = L_f[species_idx]
        legs_all = _dtn_f_legs(all_e, R_carts, L_f_atoms, C, N_eN, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        leg_m = tuple(
            leaf[0] for leaf in _dtn_f_legs(r_m[None, :], R_carts, L_f_atoms, C, N_eN, pbc_flag, cell, inv_cell, mic, dtype_jnp)
        )
        # The self pair would evaluate a zero e-e distance (NaN in grad r);
        # feed it a displaced placeholder and mask the result.
        safe_others = jnp.where(notself[:, None], all_e, r_m[None, :] + jnp.ones((1, 3), dtype=all_e.dtype))

        def per_j(j: jax.Array) -> _GradLapResult:
            r_mj, g_ij_i, lap_ij_i = _dtn_scalar_dist_with_derivs(r_m, safe_others[j], pbc_flag, cell, inv_cell, mic, dtype_jnp)
            Pz = _dtn_power_table(r_mj, N_ee + 1, dtype_jnp)
            leg_j = tuple(leaf[j] for leaf in legs_all)
            return _dtn_f_pair_grad_lap_fast(leg_m, leg_j, gamma_atoms, Pz, g_ij_i, lap_ij_i)

        g_i, l_i, g_j, l_j = jax.vmap(per_j)(idx)
        grad = grad.at[m_idx].add(jnp.sum(g_i * w[:, None], axis=0))
        lap = lap.at[m_idx].add(jnp.sum(l_i * w))
        grad = grad + g_j * w[:, None]
        lap = lap + l_j * w

    return grad, lap


@struct.dataclass
class Jastrow_DTN_grad_lap_streaming_state:
    """Carried DTN grad/Laplacian at the current configuration.

    Fields:
      * ``grad_up`` / ``grad_dn`` / ``lap_up`` / ``lap_dn``: the DTN-ONLY
        per-electron totals, in the ``jastrow_grad_lap_carry`` zone (fp64 in
        both modes). The seed, every per-hop delta and the accumulation run in
        that zone, with the coefficients of :func:`_dtn_zone_coefficients`
        (one set for the seed and every delta): an fp32 seed would carry its
        rounding unchanged over the whole interval, which a fresh evaluation
        re-draws at every hop. This is the contract of the Rust
        ``DtnGradLapCache``.
      * ``r_up_carts`` / ``r_dn_carts``: the PRE-move configuration. Load-bearing:
        the advance receives only the NEW coordinates, so the subtraction leg
        needs the old position of the mover. Kept at caller precision for
        ``fori_loop`` carry-shape stability, exactly as
        ``Jastrow_two_body_streaming_state`` does for the same reason.

    Lifetime matches the other kinetic sub-states: freshly initialised at each
    branching / interval boundary, advanced in between, never persisted across.
    """

    grad_up: jax.Array = struct.field(pytree_node=True)
    grad_dn: jax.Array = struct.field(pytree_node=True)
    lap_up: jax.Array = struct.field(pytree_node=True)
    lap_dn: jax.Array = struct.field(pytree_node=True)
    r_up_carts: jax.Array = struct.field(pytree_node=True)
    r_dn_carts: jax.Array = struct.field(pytree_node=True)


def _init_grads_laplacian_Jastrow_DTN_streaming_state(
    data: "Jastrow_DTN_data",
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_DTN_grad_lap_streaming_state:
    """Build the state from one full recompute at ``(r_up_carts, r_dn_carts)``, in the carry zone."""
    g_up, g_dn, l_up, l_dn = _compute_grads_and_laplacian_Jastrow_DTN_carry(data, r_up_carts, r_dn_carts)
    return Jastrow_DTN_grad_lap_streaming_state(
        grad_up=g_up,
        grad_dn=g_dn,
        lap_up=l_up,
        lap_dn=l_dn,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )


def _advance_grads_laplacian_Jastrow_DTN_streaming_state(
    data: "Jastrow_DTN_data",
    state: Jastrow_DTN_grad_lap_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_DTN_grad_lap_streaming_state:
    """Advance the DTN grad/Laplacian after a SINGLE-electron move.

    ``(r_up_carts_new, r_dn_carts_new)`` differ from the configuration the state
    represents in exactly one electron, identified by
    ``(moved_spin_is_up, moved_index)`` -- the same contract as the J1/J2/J3
    advances. A no-op step must be passed through unchanged by the caller.

    The update rule is uniform over every row::

        G_new = G_old - PairSum(q, .) + PairSum(p, .)

    where ``PairSum(x, .)`` is every term containing the mover with the mover at
    ``x``, scattered to both roles. Terms without the mover are untouched, which
    is what makes this ``O(N_atom * N_e)`` instead of ``O(N_atom * N_e^2)``.
    The delta and the accumulation run in the ``jastrow_grad_lap_carry`` zone.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap_carry")
    coefficient_dtype = _dtn_carry_coefficient_dtype()
    num_up = r_up_carts_new.shape[0]
    n_e = num_up + r_dn_carts_new.shape[0]

    m_idx = jnp.where(moved_spin_is_up, moved_index, num_up + moved_index)
    all_e_new = jnp.concatenate([r_up_carts_new, r_dn_carts_new], axis=0).astype(dtype_jnp)
    all_e_old = jnp.concatenate([state.r_up_carts, state.r_dn_carts], axis=0).astype(dtype_jnp)
    p = all_e_new[m_idx]
    q = all_e_old[m_idx]

    # Partner positions are identical in both configurations (only the mover
    # differs), and the mover's own row is masked out of both passes, so ONE
    # partner array serves them.
    g_new, l_new = _dtn_terms_touching_one(data, all_e_new, m_idx, p, num_up, dtype_jnp, coefficient_dtype)
    g_old, l_old = _dtn_terms_touching_one(data, all_e_new, m_idx, q, num_up, dtype_jnp, coefficient_dtype)

    grad_all = jnp.concatenate([state.grad_up, state.grad_dn], axis=0).astype(dtype_jnp) + (g_new - g_old)
    lap_all = jnp.concatenate([state.lap_up, state.lap_dn], axis=0).astype(dtype_jnp) + (l_new - l_old)
    assert grad_all.shape[0] == n_e

    return state.replace(
        grad_up=grad_all[:num_up],
        grad_dn=grad_all[num_up:],
        lap_up=lap_all[:num_up],
        lap_dn=lap_all[num_up:],
        r_up_carts=r_up_carts_new,
        r_dn_carts=r_dn_carts_new,
    )


def _jastrow_structure_is_unset(structure_data: Structure_data) -> bool:
    """Whether ``structure_data`` is the empty molecular default ``Structure_data()``.

    ``Jastrow_two_body_data`` defaults to it; on load,
    ``_apply_jastrow_structure_backfill_post_load`` replaces it by a periodic
    top-level structure (legacy checkpoints without the J2 subgroup).
    """
    return (not structure_data.pbc_flag) and np.asarray(structure_data.positions).size == 0


def _check_jastrow_periodicity_match(
    name: str, structure_data: Structure_data, ref_name: str, ref_structure_data: Structure_data
) -> None:
    """Raise ``ValueError`` unless ``structure_data`` has the periodicity (and lattice) of the reference.

    A Jastrow component evaluates its distances with its *own*
    ``structure_data`` (``pbc_flag`` selects the molecular or periodic
    kernel, ``vec_a|b|c`` the lattice). A periodic Hamiltonian whose Jastrow
    component carries a molecular structure is not lattice-periodic, and a
    molecular Hamiltonian whose component carries a cell applies minimum-image
    distances to a finite system. Both are rejected (issue #147).
    """
    pbc = bool(structure_data.pbc_flag)
    ref_pbc = bool(ref_structure_data.pbc_flag)
    if pbc != ref_pbc:
        hint = ""
        if name == "jastrow_two_body_data" and _jastrow_structure_is_unset(structure_data):
            hint = (
                " Jastrow_two_body_data.init_jastrow_two_body_data defaults to the molecular Structure_data(); "
                "pass structure_data=<the periodic Structure_data> for a periodic Hamiltonian."
            )
        raise ValueError(
            f"{name}.structure_data.pbc_flag={pbc} does not match {ref_name}.pbc_flag={ref_pbc}: "
            f"every Jastrow component must use the periodicity of the Hamiltonian structure.{hint}"
        )
    if ref_pbc:
        cell = np.asarray([structure_data.vec_a, structure_data.vec_b, structure_data.vec_c], dtype=np.float64)
        ref_cell = np.asarray([ref_structure_data.vec_a, ref_structure_data.vec_b, ref_structure_data.vec_c], dtype=np.float64)
        if cell.shape != ref_cell.shape or not np.array_equal(cell, ref_cell):
            raise ValueError(
                f"{name}.structure_data lattice vectors {cell.tolist()} differ from {ref_name} lattice vectors "
                f"{ref_cell.tolist()}: every periodic Jastrow component must use the Hamiltonian cell."
            )


def _orbital_leaf_structures(name: str, orb_data) -> list[tuple[str, Structure_data]]:
    """``(name, structure_data)`` of every leaf of a J3 orbital basis.

    ``MOs_data.structure_data`` forwards to its AO basis, and
    ``AOs_composite_data.structure_data`` forwards to its first child only, so
    a composite child on another cell would be invisible to a check of the
    forwarded structure. Every composite child is therefore listed
    (issue #147).
    """
    if isinstance(orb_data, MOs_data) and isinstance(orb_data.aos_data, AOs_composite_data):
        name, orb_data = f"{name}.aos_data", orb_data.aos_data
    if isinstance(orb_data, AOs_composite_data):
        return [(f"{name}.children[{i}]", child.structure_data) for i, child in enumerate(orb_data.children)]
    return [(name, orb_data.structure_data)]


@struct.dataclass
class Jastrow_data:
    """Jastrow dataclass.

    The class contains data for evaluating a Jastrow function.

    Args:
        jastrow_one_body_data (Jastrow_one_body_data):
            An instance of Jastrow_one_body_data. If None, the one-body Jastrow is turned off.
        jastrow_two_body_data (Jastrow_two_body_data):
            An instance of Jastrow_two_body_data. If None, the two-body Jastrow is turned off.
        jastrow_three_body_data (Jastrow_three_body_data):
            An instance of Jastrow_three_body_data. if None, the three-body Jastrow is turned off.
        jastrow_dtn_data (Jastrow_DTN_data | None):
            Optional container for the Drummond--Towler--Needs polynomial
            Jastrow (J1 + J2 + J3 family). The DTN term carries its own
            J1 / J2 / J3 channels (selected by ``use_chi`` / ``use_u`` /
            ``use_f``) and is *mutually exclusive* with the
            Pade J1 / J2 + analytic J3 family.
    """

    jastrow_one_body_data: Jastrow_one_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional one-body Jastrow component.
    jastrow_two_body_data: Jastrow_two_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional two-body Jastrow component.
    jastrow_three_body_data: Jastrow_three_body_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional analytic three-body Jastrow component.
    jastrow_dtn_data: Jastrow_DTN_data | None = struct.field(
        pytree_node=True, default=None
    )  #: Optional DTN polynomial Jastrow component (J1 + J2 + J3 family).

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        # Check cross-component structures before child validation so a
        # mismatched composite reports the offending Jastrow child path.
        # Components that carry a populated structure must agree on the
        # periodicity and lattice among themselves. The empty default J2
        # structure is resolved against the Hamiltonian structure by
        # :py:meth:`check_structure_consistency` (issue #147).
        populated = [(n, sd) for n, sd in self._component_structures() if not _jastrow_structure_is_unset(sd)]
        for name, sd in populated[1:]:
            _check_jastrow_periodicity_match(name, sd, f"{populated[0][0]}.structure_data", populated[0][1])
        if self.jastrow_one_body_data is not None:
            self.jastrow_one_body_data.sanity_check()
        if self.jastrow_two_body_data is not None:
            self.jastrow_two_body_data.sanity_check()
        if self.jastrow_three_body_data is not None:
            self.jastrow_three_body_data.sanity_check()
        if self.jastrow_dtn_data is not None:
            self.jastrow_dtn_data.sanity_check()

    def _component_structures(self) -> list[tuple[str, Structure_data]]:
        """``(name, structure_data)`` of every present component whose kernels read a structure."""
        out = []
        if self.jastrow_one_body_data is not None:
            out.append(("jastrow_one_body_data", self.jastrow_one_body_data.structure_data))
        if self.jastrow_two_body_data is not None:
            out.append(("jastrow_two_body_data", self.jastrow_two_body_data.structure_data))
        if self.jastrow_three_body_data is not None:
            out.extend(_orbital_leaf_structures("jastrow_three_body_data.orb_data", self.jastrow_three_body_data.orb_data))
        if self.jastrow_dtn_data is not None:
            out.append(("jastrow_dtn_data", self.jastrow_dtn_data.structure_data))
        return out

    def check_structure_consistency(self, structure_data: Structure_data) -> None:
        """Check that every Jastrow component has the periodicity and lattice of ``structure_data``.

        Called by :py:meth:`Hamiltonian_data.sanity_check` with the top-level
        structure, so an accepted Hamiltonian evaluates the same Jastrow in
        memory and after an HDF5 round trip / checkpoint restart.

        Raises:
            ValueError: If a component's ``structure_data.pbc_flag`` differs from
                ``structure_data.pbc_flag``, or a periodic component uses another cell.
        """
        for name, sd in self._component_structures():
            _check_jastrow_periodicity_match(name, sd, "Hamiltonian_data.structure_data", structure_data)

    @classmethod
    def build(
        cls,
        *,
        jastrow_one_body_data: "Jastrow_one_body_data | None" = None,
        jastrow_two_body_data: "Jastrow_two_body_data | None" = None,
        jastrow_three_body_data: "Jastrow_three_body_data | None" = None,
        jastrow_dtn_data: "Jastrow_DTN_data | None" = None,
    ) -> "Jastrow_data":
        """Build a validated :class:`Jastrow_data` from its component terms.

        Blessed factory: assembles the optional one- / two- / three-body
        and DTN-polynomial components, then runs :py:meth:`sanity_check`. Any
        component left as ``None`` turns that term off. The DTN term and the
        Pade J1 / J2 + AO J3 family are mutually exclusive *by tool-
        layer construction* -- this container does not police the choice.
        """
        data = cls(
            jastrow_one_body_data=jastrow_one_body_data,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
            jastrow_dtn_data=jastrow_dtn_data,
        )
        data.sanity_check()
        return data

    def to_cartesian(self) -> "Jastrow_data":
        """Return a copy with the three-body term converted to Cartesian AOs/MOs.

        Only the analytic J3 block is transformed; one- and two-body parts are
        unchanged. If no three-body data is present, the current instance is
        returned.
        """
        if self.jastrow_three_body_data is None:
            return self

        return Jastrow_data(
            jastrow_one_body_data=self.jastrow_one_body_data,
            jastrow_two_body_data=self.jastrow_two_body_data,
            jastrow_three_body_data=self.jastrow_three_body_data.to_cartesian(),
            jastrow_dtn_data=self.jastrow_dtn_data,
        )

    def to_spherical(self) -> "Jastrow_data":
        """Return a copy with the three-body term converted to spherical AOs/MOs.

        Only the analytic J3 block is transformed; one- and two-body parts are
        unchanged. If no three-body data is present, the current instance is
        returned.
        """
        if self.jastrow_three_body_data is None:
            return self

        return Jastrow_data(
            jastrow_one_body_data=self.jastrow_one_body_data,
            jastrow_two_body_data=self.jastrow_two_body_data,
            jastrow_three_body_data=self.jastrow_three_body_data.to_spherical(),
            jastrow_dtn_data=self.jastrow_dtn_data,
        )

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information from Jastrow data attributes."""
        info_lines = []
        # Replace jastrow_one_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_one_body_data is not None:
            info_lines.extend(self.jastrow_one_body_data._get_info())
        # Replace jastrow_two_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_two_body_data is not None:
            info_lines.extend(self.jastrow_two_body_data._get_info())
        # Replace jastrow_three_body_data.logger_info() with its get_info() output if available.
        if self.jastrow_three_body_data is not None:
            info_lines.extend(self.jastrow_three_body_data._get_info())
        if self.jastrow_dtn_data is not None:
            info_lines.extend(self.jastrow_dtn_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def apply_block_update(self, block: "VariationalParameterBlock") -> "Jastrow_data":
        """Apply a single variational-parameter block update to this Jastrow object.

        This method is the Jastrow-specific counterpart of
        :meth:`Wavefunction_data.apply_block_updates`.  It receives a generic
        :class:`VariationalParameterBlock` whose ``values`` have already been
        updated (typically by ``block.apply_update`` inside the SR/MCMC driver),
        and interprets that block according to Jastrow semantics.

        Responsibilities of this method are:

        * Map the block name (e.g. ``"j1_param"``, ``"j2_param"``,
          ``"j3_matrix"``) to the corresponding internal Jastrow field(s).
        * Enforce Jastrow-specific structural constraints when copying the
          block values into the internal arrays.  In particular, for the
          three-body Jastrow term (J3) this includes:

          - Handling the case where only the last column is variational and the
            rest of the matrix is constrained.
          - Handling the fully square J3 matrix case.
          - Enforcing the required symmetry of the square J3 block.

        By keeping all J1/J2/J3 interpretation and constraints in this method
        (and in the surrounding ``Jastrow_data`` class), the optimizer and
        :class:`VariationalParameterBlock` remain completely structure-agnostic.
        To introduce a new Jastrow parameter, extend the block construction
        in ``Wavefunction_data.get_variational_blocks`` and add the
        corresponding handling here, without touching the SR/MCMC driver.
        """
        # Parameter storage is float64 by construction (_JASTROW_PARAMETER_STORAGE_DTYPE);
        # the evaluation kernels cast to their own zones (issue #781).
        dtype_np = _JASTROW_PARAMETER_STORAGE_DTYPE
        j1 = self.jastrow_one_body_data
        j2 = self.jastrow_two_body_data
        j3 = self.jastrow_three_body_data
        dtn = self.jastrow_dtn_data

        if block.name == "j1_param" and j1 is not None:
            new_param = np.asarray(block.values, dtype=dtype_np).reshape(())
            j1 = Jastrow_one_body_data(
                jastrow_1b_param=new_param,
                structure_data=j1.structure_data,
                core_electrons=j1.core_electrons,
                jastrow_1b_type=j1.jastrow_1b_type,
            )
        elif block.name == "j2_param" and j2 is not None:
            new_param = np.asarray(block.values, dtype=dtype_np).reshape(())
            j2 = Jastrow_two_body_data(
                jastrow_2b_param=new_param,
                jastrow_2b_type=j2.jastrow_2b_type,
                structure_data=j2.structure_data,
            )
        elif block.name == "j3_matrix" and j3 is not None:
            j3_new = np.array(block.values, dtype=dtype_np)

            # Symmetrize unconditionally -- the method is a no-op for non-symmetric matrices.
            j3_new = self.symmetrize_j3(j3_new)

            j3 = Jastrow_three_body_data(orb_data=j3.orb_data, j_matrix=j3_new)
        elif block.name.startswith("j3_") and block.name.endswith("basis_exp") and j3 is not None:
            j3 = self._apply_j3_basis_block_delegated(j3, block, kind="exp")
        elif block.name.startswith("j3_") and block.name.endswith("basis_coeff") and j3 is not None:
            j3 = self._apply_j3_basis_block_delegated(j3, block, kind="coeff")
        elif block.name == "j3_poly_alpha" and dtn is not None:
            new_arr = np.asarray(block.values, dtype=dtype_np).reshape(dtn.alpha_indep.shape)
            dtn = dtn.replace(alpha_indep=new_arr)
        elif block.name == "j3_poly_beta" and dtn is not None:
            new_arr = np.asarray(block.values, dtype=dtype_np).reshape(dtn.beta_indep.shape)
            dtn = dtn.replace(beta_indep=new_arr)
        elif block.name == "j3_poly_gamma" and dtn is not None:
            new_arr = np.asarray(block.values, dtype=dtype_np).reshape(dtn.gamma_indep.shape)
            dtn = dtn.replace(gamma_indep=new_arr)

        return Jastrow_data(
            jastrow_one_body_data=j1,
            jastrow_two_body_data=j2,
            jastrow_three_body_data=j3,
            jastrow_dtn_data=dtn,
        )

    def symmetrize_j3(self, mat: npt.ArrayLike) -> npt.NDArray:
        """Enforce the structural J3 symmetry on ``mat``.

        :class:`Jastrow_three_body_data` requires an ``(n, n + 1)`` matrix
        whose square block is symmetric.  The constraint is therefore a
        declared ansatz fact, never inferred from current numerical values.

        This method is the **single source of truth** for all symmetry
        operations on the j3 matrix.  Both :meth:`apply_block_update`
        (parameter enforcement) and the MCMC driver (SN-metric
        symmetrization, selection-mask expansion) use this method,
        so the symmetry logic is never duplicated.  (The MCMC driver
        wraps it with flatten/unflatten in
        :meth:`~Wavefunction_data.get_variational_blocks`.)

        .. note::

           When molecular or crystal spatial symmetry is incorporated in
           the future, **only this method** needs to be extended (e.g. to
           average over a symmetry-group orbit) -- every call site
           automatically follows.

        Args:
            mat: 2-D j3 matrix to symmetrize.

        Returns:
            Symmetrized matrix.
        """
        j3 = self.jastrow_three_body_data
        if j3 is None:
            return mat
        expected_shape = np.asarray(j3.j_matrix).shape
        if len(expected_shape) != 2 or expected_shape[1] != expected_shape[0] + 1:
            raise ValueError(f"invalid stored J3 shape: {expected_shape}.")
        out = np.asarray(mat).copy()
        if out.shape != expected_shape:
            raise ValueError(f"J3 update shape {out.shape} does not match stored shape {expected_shape}.")
        sq_new = out[:, :-1]
        out[:, :-1] = 0.5 * (sq_new + sq_new.T)
        return out

    @staticmethod
    def _apply_j3_basis_block_delegated(
        j3: "Jastrow_three_body_data",
        block: "VariationalParameterBlock",
        kind: str,
    ) -> "Jastrow_three_body_data":
        """Delegate a ``j3_basis_*`` block update to the AO leaf's protocol.

        Symmetrise the values, then hand off to the
        AO leaf's :meth:`apply_basis_block_update` via
        :meth:`AOs_base_data.apply_basis_block_update`. Future PW /
        Spline leaves inherit the no-op default --- they have no
        ``basis_exp`` / ``basis_coeff`` parameters to optimise --- and
        will quietly skip this update path.

        ``kind`` is ``"exp"`` or ``"coeff"``.
        """
        suffix = "basis_exp" if kind == "exp" else "basis_coeff"
        assert block.name.endswith(suffix)
        # No symmetriser: the nested radial leaf is already one DOF per shell.
        # Preserve optimizer values across delegation; only the evaluation
        # kernel, never the stored basis parameters, uses the ao_eval dtype.
        new_arr = np.asarray(block.values)
        leaf_block = block.replace(
            # Strip only the Jastrow-owner prefix.  For a simple GTO this is
            # ``basis_*``; for a composite it preserves ``child<i>_basis_*``
            # so AOs_composite_data can route the update to the right child.
            name=block.name.removeprefix("j3_"),
            values=new_arr,
            shape=new_arr.shape,
            size=int(new_arr.size),
        )
        # Unwrap MOs_data: the leaf protocol lives on AOs_*_data; the
        # MOs wrapper just exposes the inner AO leaf.
        if isinstance(j3.orb_data, MOs_data):
            new_aos = j3.orb_data.aos_data.apply_basis_block_update(leaf_block)
            new_orb_data = j3.orb_data.replace(aos_data=new_aos)
        else:
            new_orb_data = j3.orb_data.apply_basis_block_update(leaf_block)
        return Jastrow_three_body_data(orb_data=new_orb_data, j_matrix=j3.j_matrix)

    def accumulate_position_grad(self, grad_jastrow: "Jastrow_data") -> jax.Array:
        """Aggregate position gradients from all active Jastrow components."""
        grad = 0.0
        if grad_jastrow.jastrow_one_body_data is not None:
            grad += grad_jastrow.jastrow_one_body_data.structure_data.positions
        if grad_jastrow.jastrow_three_body_data is not None:
            grad += grad_jastrow.jastrow_three_body_data.orb_data.structure_data.positions
        if grad_jastrow.jastrow_dtn_data is not None:
            grad += grad_jastrow.jastrow_dtn_data.structure_data.positions
        return grad

    def accumulate_strain_grad(self, grad_jastrow: "Jastrow_data") -> jax.Array:
        """Aggregate Cauchy-Born strain cotangents from all active Jastrow components.

        Mirrors :py:meth:`accumulate_position_grad`. Each active body's
        ``structure_data`` contributes
        ``grad_cell.T @ cell + grad_positions.T @ positions``.
        """
        from .structure import _accumulate_strain_grad_per_leaf

        sigma_tilde = jnp.zeros((3, 3), dtype=jnp.float64)
        if grad_jastrow.jastrow_one_body_data is not None:
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_jastrow.jastrow_one_body_data.structure_data,
                self.jastrow_one_body_data.structure_data,
            )
        if grad_jastrow.jastrow_two_body_data is not None:
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_jastrow.jastrow_two_body_data.structure_data,
                self.jastrow_two_body_data.structure_data,
            )
        if grad_jastrow.jastrow_three_body_data is not None:
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_jastrow.jastrow_three_body_data.orb_data.structure_data,
                self.jastrow_three_body_data.orb_data.structure_data,
            )
        if grad_jastrow.jastrow_dtn_data is not None:
            sigma_tilde += _accumulate_strain_grad_per_leaf(
                grad_jastrow.jastrow_dtn_data.structure_data,
                self.jastrow_dtn_data.structure_data,
            )
        return sigma_tilde

    def _apply_strain(self, F: jax.Array) -> "Jastrow_data":
        """Return a new ``Jastrow_data`` with ``F`` applied to every owned ``Structure_data``."""
        from .structure import _strain_structure_data

        new_j1 = self.jastrow_one_body_data
        if new_j1 is not None:
            new_j1 = new_j1.replace(structure_data=_strain_structure_data(new_j1.structure_data, F))
        new_j2 = self.jastrow_two_body_data
        if new_j2 is not None:
            new_j2 = new_j2.replace(structure_data=_strain_structure_data(new_j2.structure_data, F))
        new_j3 = self.jastrow_three_body_data
        if new_j3 is not None:
            orb = new_j3.orb_data
            # ``_apply_strain_to_aos`` recurses into AOs_composite_data
            # children (composite has no structure_data field --- it exposes
            # the property via children[0]); calling .replace(structure_data=...)
            # on it raises TypeError. The MOs_data wrapper just delegates
            # to its inner aos_data leaf.
            if isinstance(orb, MOs_data):
                new_orb = orb.replace(aos_data=_apply_strain_to_aos(orb.aos_data, F))
            else:
                new_orb = _apply_strain_to_aos(orb, F)
            new_j3 = new_j3.replace(orb_data=new_orb)
        new_dtn = self.jastrow_dtn_data
        if new_dtn is not None:
            new_sd = _strain_structure_data(new_dtn.structure_data, F)
            if new_dtn.structure_data.pbc_flag:
                # Comoving cutoff policy (doc/notes/wavefunction.md, sec:dtn_comoving_cutoff): the periodic cutoffs follow
                # ``ell_V = V^(1/3)`` and the constrained coefficients ride the minimum-norm constraint
                # tangent. Recording ``det(F)^(1/3)`` in the pytree leaf ``cutoff_scale`` makes every
                # derivative taken THROUGH ``_apply_strain`` (autograd oracles, the non-analytic
                # driver fallback, finite differences) agree with the analytic stress kernels, which
                # add the same policy term in closed form.
                lin = jnp.cbrt(jnp.linalg.det(jnp.asarray(F, dtype=jnp.float64)))
                new_scale = _dtn_cutoff_scale(new_dtn) * lin
                new_dtn = new_dtn.replace(structure_data=new_sd, cutoff_scale=new_scale)
            else:
                new_dtn = new_dtn.replace(structure_data=new_sd)
        return self.replace(
            jastrow_one_body_data=new_j1,
            jastrow_two_body_data=new_j2,
            jastrow_three_body_data=new_j3,
            jastrow_dtn_data=new_dtn,
        )

    def collect_param_grads(self, grad_jastrow: "Jastrow_data") -> dict[str, object]:
        """Collect parameter gradients into a flat dict keyed by block name."""
        grads: dict[str, object] = {}
        if grad_jastrow.jastrow_one_body_data is not None:
            grads["j1_param"] = grad_jastrow.jastrow_one_body_data.jastrow_1b_param
        if grad_jastrow.jastrow_two_body_data is not None:
            grads["j2_param"] = grad_jastrow.jastrow_two_body_data.jastrow_2b_param
        if grad_jastrow.jastrow_three_body_data is not None:
            grads["j3_matrix"] = grad_jastrow.jastrow_three_body_data.j_matrix
            # AO basis gradients are published through the same leaf protocol
            # that declares optimizer blocks.  This covers simple GTO leaves
            # (``j3_basis_*``), parameterless leaves (no blocks), and each GTO
            # child of a composite (``j3_child<i>_basis_*``).
            grad_j3_orb = grad_jastrow.jastrow_three_body_data.orb_data
            grad_j3_aos = grad_j3_orb.aos_data if isinstance(grad_j3_orb, MOs_data) else grad_j3_orb
            for name, values in grad_j3_aos.get_basis_variational_payloads(
                opt_basis_exp=True,
                opt_basis_coeff=True,
                block_name_prefix="j3_",
            ):
                grads[name] = values
        if grad_jastrow.jastrow_dtn_data is not None:
            dtn_grad = grad_jastrow.jastrow_dtn_data
            if self.jastrow_dtn_data is not None and self.jastrow_dtn_data.use_u:
                grads["j3_poly_alpha"] = dtn_grad.alpha_indep
            if self.jastrow_dtn_data is not None and self.jastrow_dtn_data.use_chi:
                grads["j3_poly_beta"] = dtn_grad.beta_indep
            if self.jastrow_dtn_data is not None and self.jastrow_dtn_data.use_f:
                grads["j3_poly_gamma"] = dtn_grad.gamma_indep
        return grads


def structure_positions_grad_Jastrow(jastrow_data: Jastrow_data, grad_jastrow_data: Jastrow_data) -> jax.Array | None:
    """Sum the nuclear-position derivative held by every Jastrow channel with geometry.

    Some Jastrow channels carry a ``structure_data`` leaf and therefore
    contribute to ``d/dR``; others (the plain two-body) do not, and where the
    leaf lives differs per channel. That is Jastrow-internal knowledge, so it
    belongs here rather than in a driver: a driver that enumerates the channels
    itself has to be edited whenever a channel with geometry is added.

    Args:
        jastrow_data: The active Jastrow, used only to test which channels exist.
        grad_jastrow_data: The Jastrow leaf of a derivative pytree with the same
            structure (e.g. from ``jax.grad`` of the local energy).

    Returns:
        The summed ``(N_atom, 3)`` contribution, or ``None`` when no active
        channel carries ``structure_data`` -- so the caller adds nothing rather
        than needing a shape to build zeros from.
    """
    total = None
    for holder, leaf in (
        (jastrow_data.jastrow_one_body_data, getattr(grad_jastrow_data, "jastrow_one_body_data", None)),
        (
            jastrow_data.jastrow_three_body_data,
            getattr(getattr(grad_jastrow_data, "jastrow_three_body_data", None), "orb_data", None),
        ),
        (jastrow_data.jastrow_dtn_data, getattr(grad_jastrow_data, "jastrow_dtn_data", None)),
    ):
        if holder is None or leaf is None:
            continue
        contribution = leaf.structure_data.positions
        total = contribution if total is None else total + contribution
    return total


def compute_Jastrow_part(
    jastrow_data: Jastrow_data, r_up_carts: jax.Array, r_dn_carts: jax.Array, zone: str = "jastrow_eval"
) -> jax.Array:
    """Evaluate the total Jastrow ``J = J1 + J2 + J3`` (without exponentiation).

    The returned scalar ``J`` excludes the ``exp`` factor; callers apply ``exp(J)`` to the wavefunction. Both the
    analytic three-body and optional NN three-body contributions are included.

    Args:
        jastrow_data: Collection of active Jastrow components (J1/J2/J3/NN).
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.
        zone: Precision zone of every component and of the sum (default
            ``"jastrow_eval"``, the zone of whole Jastrow values).

    Returns:
        float: Total Jastrow value before exponentiation.
    """
    # r_*_carts forwarded unchanged to the sub-Jastrow kernels (each handles
    # its own zone management). Do not pre-cast.
    dtype_jnp = get_dtype_jnp(zone)

    # Typed identities are part of the zone contract.  Python weak scalars make
    # an empty Jastrow return float64 in mixed mode and then violate every
    # `lax.switch` branch that pairs it with a float32 jastrow_eval identity.
    J1 = jnp.zeros((), dtype=dtype_jnp)
    J2 = jnp.zeros((), dtype=dtype_jnp)
    J3 = jnp.zeros((), dtype=dtype_jnp)

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        J1 += compute_Jastrow_one_body(jastrow_data.jastrow_one_body_data, r_up_carts, r_dn_carts, zone=zone)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        J2 += compute_Jastrow_two_body(jastrow_data.jastrow_two_body_data, r_up_carts, r_dn_carts, zone=zone)

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += compute_Jastrow_three_body(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts, zone=zone)

    # DTN polynomial (J1 + J2 + J3 family); carries its own channels.
    if jastrow_data.jastrow_dtn_data is not None:
        J3 = J3 + compute_Jastrow_DTN(jastrow_data.jastrow_dtn_data, r_up_carts, r_dn_carts, zone=zone)

    J = J1 + J2 + J3

    return J


def compute_Jastrow_part_ratio_zone(jastrow_data: Jastrow_data, r_up_carts: jax.Array, r_dn_carts: jax.Array) -> jax.Array:
    """Whole Jastrow value ``J`` evaluated in the ``jastrow_ratio`` zone, for whole-value ratios.

    The routes that form a Jastrow ratio ``exp(J(R') - J(R))`` from two whole
    values (full-recompute reference / fallback / debug ECP and discretized-kinetic
    meshes, the up-electron-only MCMC accept ratio, the autograd force fallback)
    must take both values from this entry: the precision policy assigns that ratio
    to ``jastrow_ratio`` (float64 in both modes), while :func:`compute_Jastrow_part`
    rounds ``J`` itself in ``jastrow_eval`` (float32 in mixed), a rounding of ``|J|``
    the difference keeps and no later upcast recovers. Every component,
    its parameters and coefficients and the sum are evaluated in ``jastrow_ratio``;
    AO values keep their ``ao_eval`` precision. In full precision this is
    :func:`compute_Jastrow_part` bit for bit. Differentiable like it.
    """
    return compute_Jastrow_part(jastrow_data, r_up_carts, r_dn_carts, zone="jastrow_ratio")


def _compute_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    *,
    zone: str = "jastrow_eval",
) -> float:
    """Independent NumPy reference for the full Jastrow value.

    ``zone`` is forwarded to the J3 and DTN references, which accumulate in
    that precision zone (the J1 / J2 references are always fp64).
    """
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        j1_data = jastrow_data.jastrow_one_body_data
        j1_reference = (
            _compute_Jastrow_one_body_pbc_debug
            if j1_data.structure_data.pbc_flag
            else _compute_Jastrow_one_body_molecular_debug
        )
        J1 += j1_reference(j1_data, r_up_carts, r_dn_carts)

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        j2_data = jastrow_data.jastrow_two_body_data
        j2_reference = (
            _compute_Jastrow_two_body_pbc_debug
            if j2_data.structure_data.pbc_flag
            else _compute_Jastrow_two_body_molecular_debug
        )
        J2 += j2_reference(j2_data, r_up_carts, r_dn_carts)

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        J3 += _compute_Jastrow_three_body_debug(jastrow_data.jastrow_three_body_data, r_up_carts, r_dn_carts, zone=zone)

    # DTN polynomial (oracle numpy implementation).
    if jastrow_data.jastrow_dtn_data is not None:
        J3 += _compute_Jastrow_DTN_debug(jastrow_data.jastrow_dtn_data, r_up_carts, r_dn_carts, zone=zone)

    J = J1 + J2 + J3

    return J


@jit
def _compute_Jastrow_one_body_electron_contrib_ratio(
    jastrow_one_body_data: Jastrow_one_body_data,
    r_cart: jax.Array,
) -> jax.Array:
    """Evaluate one electron's J1 contribution in the ratio precision zone.

    The regular forward evaluator belongs to ``jastrow_eval`` and is fp32 in
    mixed mode.  Calling it twice and subtracting the two accumulated values
    defeats the fp64 ``jastrow_ratio`` policy.  The rank-1 path instead needs
    the moved electron's old/new contributions evaluated directly in its own
    zone so cancellation occurs in fp64.
    """
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    sd = jastrow_one_body_data.structure_data
    positions = sd._positions_cart_jnp.astype(dtype_jnp)
    atomic_numbers = jnp.asarray(sd.atomic_numbers, dtype=dtype_jnp)
    core_electrons = jnp.asarray(jastrow_one_body_data.core_electrons, dtype=dtype_jnp)
    effective_charges = atomic_numbers - core_electrons
    param = jnp.asarray(jastrow_one_body_data.jastrow_1b_param, dtype=dtype_jnp)
    r = jnp.asarray(r_cart, dtype=dtype_jnp)

    diffs = r[None, :] - positions
    if sd.pbc_flag:
        cell = sd._cell_jnp.astype(dtype_jnp)
        inv_cell = _inv_3x3(cell)
        distances = _smooth_periodic_distance(diffs, cell, inv_cell)
    else:
        distances = jnp.linalg.norm(diffs, axis=1)

    coefficients = (2.0 * effective_charges) ** 0.25
    prefactors = -((2.0 * effective_charges) ** 0.75)
    if jastrow_one_body_data.jastrow_1b_type == "exp":
        radial = (-jnp.expm1(-param * coefficients * distances)) / (2.0 * param)
    elif jastrow_one_body_data.jastrow_1b_type == "pade":
        # A*c = 2*Z_eff includes the numerator distance scaling of u(c*r).
        prefactors = -2.0 * effective_charges
        radial = distances / (2.0 * (1.0 + param * coefficients * distances))
    else:
        raise ValueError(f"Unknown jastrow_1b_type: {jastrow_one_body_data.jastrow_1b_type}")
    return jnp.sum(prefactors * radial)


def _compute_ratio_Jastrow_part_rank1_update(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    j1_candidate_state: "J1_candidate_mesh_state | None" = None,
    j3_candidate_state: "J3_candidate_mesh_state | None" = None,
    j2_candidate_state: "J2_candidate_mesh_state | None" = None,
    j_dtn_candidate_state: "J_DTN_candidate_mesh_state | None" = None,
    dtn_state: "Jastrow_DTN_ratio_state | None" = None,
    _dtn_excluded: bool = False,
    _route_mesh: str | None = None,
) -> jax.Array:
    r"""Compute :math:`\exp(J(\mathbf r'))/\exp(J(\mathbf r))` for batched moves.

    The ratio includes the exponential and takes ``jax.Array`` inputs. The
    return is one ratio per proposed grid configuration.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference spin-up coordinates with shape ``(N_up, 3)``.
        old_r_dn_carts: Reference spin-down coordinates with shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Proposed spin-up coordinates with shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Proposed spin-down coordinates with shape ``(N_grid, N_dn, 3)``.
        j3_state: Optional cached J3 auxiliaries consistent with
            ``(old_r_up_carts, old_r_dn_carts)``. When provided, the J3 block
            reuses ``aos_*``, ``j3_mat @ aos_*``, ``j3_mat.T @ aos_*`` from the
            state instead of recomputing them -- saves per-call ``O(n_ao^2 * N_e)``
            in matmul work. Pass ``None`` (default) to recompute from scratch
            (the one-shot path used outside the projection loop).
        j1_candidate_state: Optional projection-scoped J1 candidate cache
            (``(Ne, K)`` log-ratio table) maintained by the fixed-mesh
            streaming containers. J1 has no cross-electron term, so unlike
            J2/J3/DTN its table needs no per-hop correction pass over the
            unmoved rows -- only the moved electron's own row is ever
            rebuilt. Same flattening contract as ``j2_candidate_state``.
        j3_candidate_state: Optional projection-scoped J3-orbital candidate
            cache maintained by the fixed-mesh streaming containers; ``None``
            evaluates the J3 factor directly.
        j2_candidate_state: Optional projection-scoped J2 log-ratio tables
            (up and dn) maintained by the fixed-mesh streaming containers, in
            this function's grid ordering; ``None`` evaluates the J2 factor
            directly.
        j_dtn_candidate_state: Optional projection-scoped DTN candidate cache
            (``(Ne, K)`` log-delta table) maintained by the fixed-mesh
            streaming containers. When provided, the DTN factor is read from
            the table instead of the per-config evaluators; its up-then-dn,
            m-major / k-minor flattening matches this function's grid
            ordering (same contract as ``j2_candidate_state``).
        dtn_state: Optional cached DTN distance/polynomial tables consistent
            with the old coordinates. MCMC supplies it; ECP/LRDMC callers may
            omit it and retain the stateless rank-1 path.
        _dtn_excluded: MODULE-INTERNAL. Set only by
            ``compute_ratio_Jastrow_one_move_mesh``, which supplies DTN from its
            fused ``(m, k)`` kernel instead. ``True`` statically drops the DTN
            leaf HERE, before any branch can evaluate it, so exactly one of
            {this function's generic DTN branch, the fused factor} contributes
            DTN (enforced by a double-count positive-control test). With it set, ``dtn_state`` is consequently
            ignored (the dropped leaf disables every branch that would read it).

            Callers outside this module must NOT set it: the exact-once
            obligation it creates violates channel ownership. Use
            ``compute_ratio_Jastrow_one_move_mesh``, whose contract -- "returns
            the whole Jastrow ratio" -- cannot be double counted.
        _route_mesh: MODULE-INTERNAL. ``"kinetic"`` / ``"ecp"`` when this call
            serves one of the two candidate LATTICES that carry a
            projection-scoped cache, so each channel can report to
            ``_route_probe`` which route it compiled. ``None`` (the default,
            and what MCMC's single-move caller leaves it at) disables the
            report: a single proposed move is not a mesh and has no candidate
            cache to miss.

    Returns:
        jax.Array: Jastrow ratios per grid with shape ``(N_grid,)`` (includes ``exp``).

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
    # Forward old/new r_up/dn_carts as-is (Principle 3a -- no parameter rebind).
    # Module-level forwards (compute_Jastrow_part, compute_Jastrow_one_body,
    # compute_orb_api, NN_Jastrow.apply) handle their own use-site casts.
    # Inline arithmetic in the local J1/J2/J3 closures below casts at the diff
    # site (Principle 3b) -- for r-r differences the operand is reconstructed in
    # caller-supplied precision (fp64 from MCMC walker state) before downcast.
    dtype_jnp = get_dtype_jnp("jastrow_ratio")

    if _dtn_excluded:
        # Exact-once: drop the DTN leaf BEFORE any branch below --
        # both generic DTN branches read ``jastrow_data.jastrow_dtn_data``,
        # so this ONE static exclusion governs every return path.
        jastrow_data = jastrow_data.replace(jastrow_dtn_data=None)

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]
    n_grid = new_r_up_carts_arr.shape[0]
    # A fully spin-polarised configuration (one spin channel empty) takes the
    # same per-channel rank-one formulas as a two-spin one. Every electron
    # identification below (``jnp.argmax`` over a spin's electron axis, and the
    # gathers indexed by it) is resolved statically per spin: an empty channel
    # has no electron that can move, so its branch is never built (an argmax
    # over a zero-size axis raises at trace time). Earlier this case recomputed
    # two whole ``compute_Jastrow_part`` values and subtracted them, i.e. formed
    # J(R') - J(R) in ``jastrow_eval`` (float32 in mixed) instead of this
    # function's float64 ``jastrow_ratio`` zone (issue #775).

    J_ratio = jnp.ones(n_grid, dtype=dtype_jnp)

    # J1 part. Evaluate only the moved electron, directly in jastrow_ratio.
    # Reusing compute_Jastrow_one_body here would evaluate two accumulated J1
    # values in mixed-mode fp32 and only promote after the lossy subtraction.
    if jastrow_data.jastrow_one_body_data is not None and j1_candidate_state is not None:
        if _route_mesh is not None:
            note_route("j1", _route_mesh, cached=True)
        J_ratio *= jnp.exp(
            jnp.concatenate([j1_candidate_state.log_ratio_up.reshape(-1), j1_candidate_state.log_ratio_dn.reshape(-1)])
        )
    if jastrow_data.jastrow_one_body_data is not None and j1_candidate_state is None:
        # No candidate cache supplied (MCMC single-move, or a caller outside
        # the fixed-mesh streaming containers): fall back to the per-config
        # evaluator. On a mesh with no cache this is genuinely the uncached
        # route -- both ``j1_new`` and ``j1_old`` below are re-evaluated over
        # all N_grid candidates every hop.
        if _route_mesh is not None:
            note_route("j1", _route_mesh, cached=False)
        j1_data = jastrow_data.jastrow_one_body_data
        delta_up = new_r_up_carts_arr - old_r_up_carts[None, :, :]
        delta_dn = new_r_dn_carts_arr - old_r_dn_carts[None, :, :]
        moved_up_mask = jnp.any(delta_up != 0.0, axis=2)
        moved_dn_mask = jnp.any(delta_dn != 0.0, axis=2)
        moved_up = jnp.any(moved_up_mask, axis=1)
        # Static per-spin electron identification (empty channel: no branch).
        if num_up > 0:
            idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)
            r_up_new = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1).reshape(n_grid, 3)
            r_up_old = jnp.take(old_r_up_carts, idx_up, axis=0)
        if num_dn > 0:
            idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)
            r_dn_new = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1).reshape(n_grid, 3)
            r_dn_old = jnp.take(old_r_dn_carts, idx_dn, axis=0)
        if num_dn == 0:
            r_new, r_old = r_up_new, r_up_old
        elif num_up == 0:
            r_new, r_old = r_dn_new, r_dn_old
        else:
            r_new = jnp.where(moved_up[:, None], r_up_new, r_dn_new)
            r_old = jnp.where(moved_up[:, None], r_up_old, r_dn_old)
        j1_new = vmap(_compute_Jastrow_one_body_electron_contrib_ratio, in_axes=(None, 0))(j1_data, r_new)
        j1_old = vmap(_compute_Jastrow_one_body_electron_contrib_ratio, in_axes=(None, 0))(j1_data, r_old)
        J_ratio *= jnp.exp(j1_new - j1_old)

    def _two_body_jastrow_exp(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Exponential form of J2."""
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        return 1.0 / (2.0 * param) * (-jnp.expm1(-param * jnp.linalg.norm(diff)))

    def _two_body_jastrow_pade(param: float, r_cart_i: jnpt.ArrayLike, r_cart_j: jnpt.ArrayLike) -> float:
        """Pade form of J2."""
        # Reconstruct diff in caller-supplied precision then downcast (Principle 3b).
        diff = (r_cart_i - r_cart_j).astype(dtype_jnp)
        return jnp.linalg.norm(diff) / 2.0 * (1.0 + param * jnp.linalg.norm(diff)) ** (-1.0)

    # Select the functional form based on type
    if jastrow_data.jastrow_two_body_data is not None:
        _j2b_type = jastrow_data.jastrow_two_body_data.jastrow_2b_type
    else:
        _j2b_type = "pade"

    if _j2b_type == "pade":
        two_body_jastrow_anti_parallel_spins = _two_body_jastrow_pade
        two_body_jastrow_parallel_spins = _two_body_jastrow_pade
    else:
        two_body_jastrow_anti_parallel_spins = _two_body_jastrow_exp
        two_body_jastrow_parallel_spins = _two_body_jastrow_exp

    def compute_one_grid_J2(
        jastrow_2b_param: jax.Array,
        new_r_up_carts: jax.Array,
        new_r_dn_carts: jax.Array,
        old_r_up_carts: jax.Array,
        old_r_dn_carts: jax.Array,
    ) -> jax.Array:
        delta_up = new_r_up_carts - old_r_up_carts
        delta_dn = new_r_dn_carts - old_r_dn_carts
        up_moved = jnp.any(delta_up != 0)
        if num_up == 0:
            nonzero_dn = jnp.any(delta_dn != 0, axis=1)
            idx = jnp.argmax(nonzero_dn)
            up_moved = False
        elif num_dn == 0:
            nonzero_up = jnp.any(delta_up != 0, axis=1)
            idx = jnp.argmax(nonzero_up)
            up_moved = True
        else:
            nonzero_up = jnp.any(delta_up != 0, axis=1)
            nonzero_dn = jnp.any(delta_dn != 0, axis=1)
            idx_up = jnp.argmax(nonzero_up)
            idx_dn = jnp.argmax(nonzero_dn)
            # jnp.where instead of lax.cond: batched-predicate cond miscompiles
            # on XLA CPU (jax 0.7.x) inside the vmapped GFMC loops; see the
            # matching notes on the streaming-advance kernels.
            idx = jnp.where(up_moved, idx_up, idx_dn)

        def up_case(
            jastrow_2b_param: jax.Array,
            new_r_up_carts: jax.Array,
            new_r_dn_carts: jax.Array,
            old_r_up_carts: jax.Array,
            old_r_dn_carts: jax.Array,
        ) -> jax.Array:
            new_r_up_carts_extracted = jnp.expand_dims(new_r_up_carts[idx], axis=0)  # shape=(1,3)
            old_r_up_carts_extracted = jnp.expand_dims(old_r_up_carts[idx], axis=0)  # shape=(1,3)
            J2_up_up_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, new_r_up_carts
                )
            )
            J2_up_up_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_up_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_up_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_up_carts_extracted, old_r_dn_carts
                )
            )
            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        def dn_case(
            jastrow_2b_param: jax.Array,
            new_r_up_carts: jax.Array,
            new_r_dn_carts: jax.Array,
            old_r_up_carts: jax.Array,
            old_r_dn_carts: jax.Array,
        ) -> jax.Array:
            new_r_dn_carts_extracted = jnp.expand_dims(new_r_dn_carts[idx], axis=0)  # shape=(1,3)
            old_r_dn_carts_extracted = jnp.expand_dims(old_r_dn_carts[idx], axis=0)  # shape=(1,3)
            J2_dn_dn_new = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, new_r_dn_carts_extracted, new_r_dn_carts
                )
            )
            J2_dn_dn_old = jnp.sum(
                vmap(two_body_jastrow_parallel_spins, in_axes=(None, None, 0))(
                    jastrow_2b_param, old_r_dn_carts_extracted, old_r_dn_carts
                )
            )
            J2_up_dn_new = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, new_r_dn_carts_extracted
                )
            )
            J2_up_dn_old = jnp.sum(
                vmap(two_body_jastrow_anti_parallel_spins, in_axes=(None, 0, None))(
                    jastrow_2b_param, old_r_up_carts, old_r_dn_carts_extracted
                )
            )

            return jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_dn_dn_new - J2_dn_dn_old)

        if num_up == 0:
            return dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)
        if num_dn == 0:
            return up_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts)

        # jnp.where over both computed branches instead of lax.cond (same
        # miscompile note as above; a batched cond executed both anyway).
        return jnp.where(
            up_moved,
            up_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
            dn_case(jastrow_2b_param, new_r_up_carts, new_r_dn_carts, old_r_up_carts, old_r_dn_carts),
        )

    # J2 part. Projection-scoped meshes carry the log-ratio table and advance
    # it from the accepted one-electron hop; other routes retain the established
    # full per-grid evaluator below.
    if jastrow_data.jastrow_two_body_data is not None and j2_candidate_state is not None:
        if _route_mesh is not None:
            note_route("j2", _route_mesh, cached=True)
        J_ratio *= jnp.exp(
            jnp.concatenate([j2_candidate_state.log_ratio_up.reshape(-1), j2_candidate_state.log_ratio_dn.reshape(-1)])
        )
    if jastrow_data.jastrow_two_body_data is not None and j2_candidate_state is None:
        if _route_mesh is not None:
            note_route("j2", _route_mesh, cached=False)
        j2_param = jastrow_data.jastrow_two_body_data.jastrow_2b_param
        _j2_type = jastrow_data.jastrow_two_body_data.jastrow_2b_type

        if _j2_type == "pade":

            def _j2_from_dist(dist: jax.Array, param: jax.Array) -> jax.Array:
                return dist / 2.0 * (1.0 + param * dist) ** (-1.0)
        else:

            def _j2_from_dist(dist: jax.Array, param: jax.Array) -> jax.Array:
                return 1.0 / (2.0 * param) * (-jnp.expm1(-param * dist))

        # The OLD-side per-electron pair-sum baseline via
        # ``compute_pairwise_sums(old, old)`` would be O(N^2), but only the moved-
        # electron row is consumed per grid, so the OLD-side per-grid sum is
        # computed on demand via the same ``_batch_pairwise_sum`` used for the
        # NEW side -- total work O(N * N_grid). Both same-spin sums EXCLUDE the
        # moved electron's own column ``skip`` (masked to zero): on the OLD side
        # its exact value is ``J2(0) = 0``, but the molecular Gram-form distance
        # ``sqrt(max(|a|^2 + |a|^2 - 2 a.a, 0))`` rounds to a ``sqrt(eps64)``-sized
        # artefact instead, and on the NEW side the pair (r', r_old) is not a pair
        # of the moved configuration at all. Masking the column removes both
        # without a separate self-pair correction (PR #780: the
        # unmasked OLD self pair left a 1.49e-8 log-ratio error in both modes).

        delta_up_all = new_r_up_carts_arr - old_r_up_carts
        delta_dn_all = new_r_dn_carts_arr - old_r_dn_carts
        moved_up_mask = jnp.any(delta_up_all != 0.0, axis=2)
        moved_dn_mask = jnp.any(delta_dn_all != 0.0, axis=2)
        moved_up_exists = jnp.any(moved_up_mask, axis=1)
        # Static per-spin electron identification (empty channel: no branch).
        if num_up > 0:
            idx_up = jnp.argmax(moved_up_mask.astype(jnp.int32), axis=1)
            r_up_new = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1).reshape(-1, 3)
            r_up_old = jnp.take(old_r_up_carts, idx_up, axis=0)
        if num_dn > 0:
            idx_dn = jnp.argmax(moved_dn_mask.astype(jnp.int32), axis=1)
            r_dn_new = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1).reshape(-1, 3)
            r_dn_old = jnp.take(old_r_dn_carts, idx_dn, axis=0)

        # PBC dispatch: the electron--electron distance used by the
        # batched pair sums must match the Williamson
        # smooth periodic distance employed by the production J2 evaluator
        # (``compute_Jastrow_two_body``); otherwise the rank-1 ratio disagrees with
        # the full recompute under PBC. ``sd_j2.pbc_flag`` is ``pytree_node=False``
        # so this resolves at trace time -- the molecular branch is unaffected.
        sd_j2 = jastrow_data.jastrow_two_body_data.structure_data
        if sd_j2.pbc_flag:
            cell_j2 = sd_j2._cell_jnp.astype(jnp.float64)
            inv_cell_j2 = _inv_3x3(cell_j2)

            def _batch_pairwise_sum(
                points_a: jax.Array, points_b: jax.Array, param: jax.Array, skip: jax.Array | None = None
            ) -> jax.Array:
                pa = points_a.astype(dtype_jnp)
                pb = points_b.astype(dtype_jnp)
                diff = pa[:, None, :] - pb[None, :, :]  # (M, K, 3)
                d = _smooth_periodic_distance(diff, cell_j2.astype(dtype_jnp), inv_cell_j2.astype(dtype_jnp))
                vals = _j2_from_dist(d, param)
                if skip is not None:
                    vals = jnp.where(jnp.arange(vals.shape[1])[None, :] == skip[:, None], 0.0, vals)
                return jnp.sum(vals, axis=1)
        else:

            def _batch_pairwise_sum(
                points_a: jax.Array, points_b: jax.Array, param: jax.Array, skip: jax.Array | None = None
            ) -> jax.Array:
                # Cast operands to the jastrow_ratio zone at the arithmetic use site
                # (Principle 3b). Inputs may arrive in caller-supplied precision; cast
                # before consuming as norm/dot operands to keep the function in-zone.
                pa = points_a.astype(dtype_jnp)
                pb = points_b.astype(dtype_jnp)
                norm_a2 = jnp.sum(pa * pa, axis=1, keepdims=True)
                norm_b2 = jnp.sum(pb * pb, axis=1, keepdims=True).T
                dots = jnp.dot(pa, pb.T)
                d2 = jnp.maximum(norm_a2 + norm_b2 - 2.0 * dots, 0.0)
                safe_d2 = jnp.where(d2 > 0, d2, jnp.ones_like(d2))
                d = jnp.where(d2 > 0, jnp.sqrt(safe_d2), jnp.zeros_like(d2))
                vals = _j2_from_dist(d, param)
                if skip is not None:
                    vals = jnp.where(jnp.arange(vals.shape[1])[None, :] == skip[:, None], 0.0, vals)
                return jnp.sum(vals, axis=1)

        # Up-move branch contributions (all grids in batch)
        if num_up > 0:
            up_up_new = _batch_pairwise_sum(r_up_new, old_r_up_carts, j2_param, skip=idx_up)
            up_up_old = _batch_pairwise_sum(r_up_old, old_r_up_carts, j2_param, skip=idx_up)

            up_dn_new = _batch_pairwise_sum(r_up_new, old_r_dn_carts, j2_param)
            up_dn_old = _batch_pairwise_sum(r_up_old, old_r_dn_carts, j2_param)
            J2_ratio_up = jnp.exp((up_up_new - up_up_old) + (up_dn_new - up_dn_old))

        # Down-move branch contributions (all grids in batch)
        if num_dn > 0:
            dn_dn_new = _batch_pairwise_sum(r_dn_new, old_r_dn_carts, j2_param, skip=idx_dn)
            dn_dn_old = _batch_pairwise_sum(r_dn_old, old_r_dn_carts, j2_param, skip=idx_dn)

            dn_up_new = _batch_pairwise_sum(r_dn_new, old_r_up_carts, j2_param)
            dn_up_old = _batch_pairwise_sum(r_dn_old, old_r_up_carts, j2_param)
            J2_ratio_dn = jnp.exp((dn_dn_new - dn_dn_old) + (dn_up_new - dn_up_old))

        if num_dn == 0:
            J2_ratio = J2_ratio_up
        elif num_up == 0:
            J2_ratio = J2_ratio_dn
        else:
            J2_ratio = jnp.where(moved_up_exists, J2_ratio_up, J2_ratio_dn)

        J_ratio *= jnp.ravel(J2_ratio)

    # J3 part  (batched AO evaluation -- avoids per-config compute_orb_api inside vmap)
    if jastrow_data.jastrow_three_body_data is not None:
        j3d = jastrow_data.jastrow_three_body_data
        j3_mat = j3d._j_matrix_jnp[:, :-1]  # (n_ao, n_ao)  shared for up-up / dn-dn / up-dn
        j1_vec = j3d._j_matrix_jnp[:, -1]  # (n_ao,)

        # Old AOs evaluated once.
        # When ``j3_state`` is supplied, the cached AOs/W/U/cross_vec from the
        # streaming state are consistent with ``(old_r_up_carts, old_r_dn_carts)``
        # by contract -- we just dtype-cast into the jastrow_ratio zone and skip
        # the recomputation. Python-static dispatch (j3_state is None vs not).
        if j3_state is None:
            aos_up_old = jnp.array(jnp.real(j3d.compute_orb_api(j3d.orb_data, old_r_up_carts)), dtype=dtype_jnp)  # (n_ao, N_up)
            aos_dn_old = jnp.array(jnp.real(j3d.compute_orb_api(j3d.orb_data, old_r_dn_carts)), dtype=dtype_jnp)  # (n_ao, N_dn)
        else:
            aos_up_old = j3_state.aos_up.astype(dtype_jnp)
            aos_dn_old = j3_state.aos_dn.astype(dtype_jnp)

        N_batch = new_r_up_carts_arr.shape[0]

        # Detect which spin moved and which electron index per config. The
        # index of an empty spin channel is never built (static per spin).
        delta_up_batch = new_r_up_carts_arr - old_r_up_carts[None]  # (N, N_up, 3)
        delta_dn_batch = new_r_dn_carts_arr - old_r_dn_carts[None]  # (N, N_dn, 3)
        up_moved_batch = jnp.any(delta_up_batch != 0.0, axis=(1, 2))  # (N,) bool
        if num_up > 0:
            idx_up = jnp.argmax(jnp.any(delta_up_batch != 0.0, axis=2).astype(jnp.int32), axis=1)  # (N,)
            r_new_up_moved = jnp.take_along_axis(new_r_up_carts_arr, idx_up[:, None, None], axis=1).reshape(N_batch, 3)
        if num_dn > 0:
            idx_dn = jnp.argmax(jnp.any(delta_dn_batch != 0.0, axis=2).astype(jnp.int32), axis=1)  # (N,)
            r_new_dn_moved = jnp.take_along_axis(new_r_dn_carts_arr, idx_dn[:, None, None], axis=1).reshape(N_batch, 3)

        # New position of the moved electron per config (select spin block)
        if num_dn == 0:
            r_new_moved = r_new_up_moved
        elif num_up == 0:
            r_new_moved = r_new_dn_moved
        else:
            r_new_moved = jnp.where(up_moved_batch[:, None], r_new_up_moved, r_new_dn_moved)  # (N, 3)

        # Single batched AO evaluation for all N configs (replaces N per-config calls inside vmap).
        # Old AOs are obtained by column-slicing the already-computed
        # ``aos_up_old`` / ``aos_dn_old`` (consistent with old_r_*_carts), avoiding
        # a redundant ``compute_orb_api`` call on ``r_old_moved`` -- mirrors the
        # gather trick already used in ``_compute_ratio_Jastrow_part_split_spin``.
        if j3_candidate_state is None:
            if _route_mesh is not None:
                note_route("j3", _route_mesh, cached=False)
            aos_new_batch = jnp.array(jnp.real(j3d.compute_orb_api(j3d.orb_data, r_new_moved)), dtype=dtype_jnp)  # (n_ao, N)
        else:
            if _route_mesh is not None:
                note_route("j3", _route_mesh, cached=True)
            # The one-move mesh is flattened up-electron-major first, then
            # down-electron-major, exactly matching the configuration arrays.
            aos_new_batch = jnp.concatenate(
                [
                    j3_candidate_state.aos_up_candidates.astype(dtype_jnp),
                    j3_candidate_state.aos_dn_candidates.astype(dtype_jnp),
                ],
                axis=1,
            )
        if num_dn == 0:
            aos_old_batch = jnp.take(aos_up_old, idx_up, axis=1)  # (n_ao, N)
        elif num_up == 0:
            aos_old_batch = jnp.take(aos_dn_old, idx_dn, axis=1)  # (n_ao, N)
        else:
            aos_up_old_at_idx = jnp.take(aos_up_old, idx_up, axis=1)  # (n_ao, N)
            aos_dn_old_at_idx = jnp.take(aos_dn_old, idx_dn, axis=1)  # (n_ao, N)
            aos_old_batch = jnp.where(up_moved_batch[None, :], aos_up_old_at_idx, aos_dn_old_at_idx)  # (n_ao, N)
        # A configuration that moves no electron (an ECP quadrature point on
        # the electron itself) falls into the dn branch with idx_dn = 0, while
        # a cached candidate AO column belongs to the candidate's own electron.
        # Its ratio is exactly 1 (issue #652).
        any_moved_batch = up_moved_batch | jnp.any(delta_dn_batch != 0.0, axis=(1, 2))  # (N,)
        aos_p_batch = jnp.where(any_moved_batch[None, :], aos_new_batch - aos_old_batch, 0.0)  # (n_ao, N)

        # Precompute constant products (independent of config). With a
        # streaming state, all four matmuls are read directly from the cache
        # -- that's the main per-step ``O(n_ao^2 * N_e)`` saving. Note that
        # ``j3_state.j3_mat_T_aos_*`` stores ``j3_mat.T @ aos_*`` of shape
        # ``(n_ao, N_*)``, while we want ``U_* = aos_*.T @ j3_mat`` of shape
        # ``(N_*, n_ao)`` -- these are transposes of each other.
        if j3_state is None:
            W_up = jnp.dot(j3_mat, aos_up_old)  # (n_ao, N_up)  = j3_mat @ A_up
            U_up = jnp.dot(aos_up_old.T, j3_mat)  # (N_up, n_ao)  = A_up.T @ j3_mat
            W_dn = jnp.dot(j3_mat, aos_dn_old)  # (n_ao, N_dn)
            U_dn = jnp.dot(aos_dn_old.T, j3_mat)  # (N_dn, n_ao)
            dn_cross_vec = j3_mat @ jnp.sum(aos_dn_old, axis=1)  # (n_ao,): UP cross term constant
            up_cross_vec = jnp.sum(aos_up_old, axis=1) @ j3_mat  # (n_ao,): DN cross term constant
        else:
            W_up = j3_state.j3_mat_aos_up.astype(dtype_jnp)
            W_dn = j3_state.j3_mat_aos_dn.astype(dtype_jnp)
            U_up = j3_state.j3_mat_T_aos_up.astype(dtype_jnp).T
            U_dn = j3_state.j3_mat_T_aos_dn.astype(dtype_jnp).T
            # cross_vec are precomputed and rank-1 advanced in the streaming
            # state (``j3_mat_aos_dn_rowsum`` / ``j3_mat_T_aos_up_rowsum``).
            # Equivalences:
            #   dn_cross_vec = j3_mat @ sum(aos_dn, axis=1)
            #                = sum(j3_mat @ aos_dn, axis=1) = sum(W_dn, axis=1)
            #                = j3_mat_aos_dn_rowsum
            #   up_cross_vec = sum(aos_up, axis=1) @ j3_mat
            #                = sum(j3_mat.T @ aos_up, axis=1)
            #                = j3_mat_T_aos_up_rowsum
            # Reading the cached ``(n_ao,)`` rowsum avoids the per-call
            # ``(W, n_ao, N_e)`` HBM-bound reduction.
            dn_cross_vec = j3_state.j3_mat_aos_dn_rowsum.astype(dtype_jnp)
            up_cross_vec = j3_state.j3_mat_T_aos_up_rowsum.astype(dtype_jnp)

        # Q index: idx_up for UP configs, idx_dn for DN configs
        if num_dn == 0:
            idx_for_Q = idx_up
        elif num_up == 0:
            idx_for_Q = idx_dn
        else:
            idx_for_Q = jnp.where(up_moved_batch, idx_up, idx_dn)  # (N,)

        # term1: J1-like contribution (identical formula for UP and DN)
        term1 = j1_vec @ aos_p_batch  # (N,)

        # UP formula  -----------------------------------------------------------
        # Use tensordot with explicit contracting axes (n_ao = axis 0 of both
        # operands) instead of ``aos_p_batch.T @ W_up``: under vmap on the
        # walker axis XLA's transpose-folding does not fold the ``.T`` into
        # the dot, materialising an explicit ``transpose`` kernel that is
        # HBM-bound. Expressing the contraction via
        # ``dot_general(contract=[0]x[0])`` avoids the materialisation.
        V_up = jnp.tensordot(aos_p_batch, W_up, axes=((0,), (0,)))  # (N, N_up)
        P_up = jnp.dot(U_up, aos_p_batch)  # (N_up, N)
        Q_up_c = (idx_for_Q[:, None] < jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (N, N_up)
        Q_up_r = (idx_for_Q[:, None] > jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (N, N_up)
        term2_up = jnp.sum(V_up * Q_up_c, axis=1)  # (N,)
        term3_up = jnp.sum(P_up.T * Q_up_r, axis=1)  # (N,)
        term4_up = dn_cross_vec @ aos_p_batch  # (N,)
        J3_log_up = term1 + term2_up + term3_up + term4_up

        # DN formula  -----------------------------------------------------------
        # See UP-formula comment above re: tensordot-vs-``.T``-then-dot.
        V_dn = jnp.tensordot(aos_p_batch, W_dn, axes=((0,), (0,)))  # (N, N_dn)
        P_dn = jnp.dot(U_dn, aos_p_batch)  # (N_dn, N)
        Q_dn_c = (idx_for_Q[:, None] < jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (N, N_dn)
        Q_dn_r = (idx_for_Q[:, None] > jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (N, N_dn)
        term2_dn = jnp.sum(V_dn * Q_dn_c, axis=1)  # (N,)
        term3_dn = jnp.sum(P_dn.T * Q_dn_r, axis=1)  # (N,)
        term4_dn = up_cross_vec @ aos_p_batch  # (N,)
        J3_log_dn = term1 + term2_dn + term3_dn + term4_dn

        # Select UP or DN formula per config
        if num_up == 0:
            J3_ratio = jnp.exp(J3_log_dn)
        elif num_dn == 0:
            J3_ratio = jnp.exp(J3_log_up)
        else:
            J3_ratio = jnp.exp(jnp.where(up_moved_batch, J3_log_up, J3_log_dn))

        J_ratio *= J3_ratio

    # DTN part. Projection-scoped meshes carry the (Ne, K) log-delta table and
    # advance it from the accepted one-electron hop (mirrors the J2 cache
    # above); its up-then-dn, m-major / k-minor flattening matches this
    # function's grid ordering. Other routes retain the per-config evaluators
    # below.
    if jastrow_data.jastrow_dtn_data is not None and j_dtn_candidate_state is not None:
        if _route_mesh is not None:
            note_route("dtn", _route_mesh, cached=True)
        J_ratio *= jnp.exp(j_dtn_candidate_state.log_delta.reshape(-1))
    # DTN polynomial part (rank-1 delta -- exactly one electron differs per
    # grid config). Each per-grid ``delta J_DTN`` only touches pairs/triples
    # involving the moved electron, cutting the vmap intermediate from
    # ``(N_grid, N_pair=N_e*(N_e-1)/2, ...)`` to ``(N_grid, N_e-1, ...)`` --
    # the (N_e/2)-factor reduction on the dominant ``f`` channel intermediate
    # is what reclaims ~5 GB in ECP non-local (silicon scale). See
    # ``_compute_delta_J_DTN_per_config`` for the per-config delta formula
    # and its derivation. No streaming state is needed -- DTN has no AO
    # matmul to amortize, unlike J3's ``j3_mat @ aos_*``.
    if jastrow_data.jastrow_dtn_data is not None and j_dtn_candidate_state is None:
        if _route_mesh is not None:
            note_route("dtn", _route_mesh, cached=False)
        dtn = jastrow_data.jastrow_dtn_data
        if dtn_state is None:
            # Streamed on the same knob as the ECP-ratio twin: the mixed
            # (non-cached) DTN delta rebuilds a per-config block, and a plain
            # vmap over the G quadrature / move configurations would hold all
            # of them at once -- ~6 MiB per config
            # for every config. Configs are independent, so blocking is a
            # pure peak reduction.
            _delta_specs = (
                jax.ShapeDtypeStruct((num_up, 3), new_r_up_carts_arr.dtype),
                jax.ShapeDtypeStruct((num_dn, 3), new_r_dn_carts_arr.dtype),
            )
            _delta_batch = measured_stream_batch(
                _delta_J_DTN_mesh_probe,
                (dtn, old_r_up_carts, old_r_dn_carts),
                _delta_specs,
                n_grid,
                JASTROW_DTN_ECP_GRID_BUDGET_BYTES,
            )
            delta_J_DTN_per_config = jax.lax.map(
                lambda x: _compute_delta_J_DTN_per_config(dtn, old_r_up_carts, old_r_dn_carts, x[0], x[1]),
                (new_r_up_carts_arr, new_r_dn_carts_arr),
                batch_size=_delta_batch,
            )
        else:
            all_e = jnp.concatenate((old_r_up_carts, old_r_dn_carts), axis=0)

            def _moved_index_and_row(
                new_up: jax.Array, new_dn: jax.Array, old_up: jax.Array, old_dn: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                # Static per spin: an empty channel's argmax is never built.
                if num_up > 0:
                    diff_up = jnp.any(new_up != old_up, axis=1)
                    idx_up = jnp.argmax(diff_up.astype(jnp.int32))
                if num_dn > 0:
                    diff_dn = jnp.any(new_dn != old_dn, axis=1)
                    idx_dn = jnp.argmax(diff_dn.astype(jnp.int32))
                if num_dn == 0:
                    return idx_up, new_up[idx_up]
                if num_up == 0:
                    return num_up + idx_dn, new_dn[idx_dn]
                up_moved = jnp.any(diff_up)
                moved_index = jnp.where(up_moved, idx_up, num_up + idx_dn)
                r_new = jnp.where(up_moved, new_up[idx_up], new_dn[idx_dn])
                return moved_index, r_new

            def _cached_delta(new_up: jax.Array, new_dn: jax.Array) -> jax.Array:
                moved_index, r_new = _moved_index_and_row(new_up, new_dn, old_r_up_carts, old_r_dn_carts)
                return _delta_J_DTN_from_ratio_state(dtn, dtn_state, moved_index, r_new, all_e, num_up)

            # Measured-budget probe for the cached branch: reconstructs the
            # _cached_delta per-config workspace with the static num_up closed over
            # and the tracers (dtn, dtn_state, old coords, all_e) threaded via data.
            def _cached_delta_probe(
                pdata: tuple[Jastrow_DTN_data, "Jastrow_DTN_ratio_state", jax.Array, jax.Array, jax.Array],
                new_up: jax.Array,
                new_dn: jax.Array,
            ) -> jax.Array:
                dtn_p, dtn_state_p, old_up_p, old_dn_p, all_e_p = pdata
                moved_index, r_new = _moved_index_and_row(new_up, new_dn, old_up_p, old_dn_p)
                return _delta_J_DTN_from_ratio_state(dtn_p, dtn_state_p, moved_index, r_new, all_e_p, num_up)

            # Apply the same bounded stream to the cached branch (~3 MiB per config).
            _cached_specs = (
                jax.ShapeDtypeStruct((num_up, 3), new_r_up_carts_arr.dtype),
                jax.ShapeDtypeStruct((num_dn, 3), new_r_dn_carts_arr.dtype),
            )
            _cached_batch = measured_stream_batch(
                _cached_delta_probe,
                (dtn, dtn_state, old_r_up_carts, old_r_dn_carts, all_e),
                _cached_specs,
                n_grid,
                JASTROW_DTN_ECP_GRID_BUDGET_BYTES,
            )
            delta_J_DTN_per_config = jax.lax.map(
                lambda x: _cached_delta(x[0], x[1]),
                (new_r_up_carts_arr, new_r_dn_carts_arr),
                batch_size=_cached_batch,
            )
        J_ratio *= jnp.exp(delta_J_DTN_per_config)

    return J_ratio


def compute_ratio_Jastrow_one_move_mesh(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_carts_arr: jax.Array,
    new_r_dn_carts_arr: jax.Array,
    shifts: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    j1_candidate_state: "J1_candidate_mesh_state | None" = None,
    j3_candidate_state: "J3_candidate_mesh_state | None" = None,
    j2_candidate_state: "J2_candidate_mesh_state | None" = None,
    j_dtn_candidate_state: "J_DTN_candidate_mesh_state | None" = None,
) -> jax.Array:
    r"""Whole-Jastrow ratio over a ONE-MOVE MESH, dispatched per channel here.

    The caller describes the mesh STRUCTURE -- an ``(m, k)`` lattice in which
    electron ``m`` is displaced by ``shifts[k]`` and nothing else moves -- and
    receives the complete :math:`\exp(J(\mathbf r'))/\exp(J(\mathbf r))`. Which
    channels exist, which of them have a fused mesh kernel, and how their
    factors combine are decided HERE; no caller may know or need to know. The
    contract is simply "this returns the whole Jastrow ratio", which cannot be
    double counted by construction.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference spin-up coordinates, shape ``(N_up, 3)``.
        old_r_dn_carts: Reference spin-down coordinates, shape ``(N_dn, 3)``.
        new_r_up_carts_arr: Mesh spin-up coordinates, shape ``(N_grid, N_up, 3)``.
        new_r_dn_carts_arr: Mesh spin-down coordinates, shape ``(N_grid, N_dn, 3)``.
        shifts: Displacement lattice, shape ``(k, 3)``. ``N_grid == N_e * k``
            with the up-then-dn, shift-minor flattening the fused kernels assume
            (pinned by the flattening-order gate).
        j3_state: Optional cached J3 auxiliaries for the old coordinates.
        j1_candidate_state: Optional projection-scoped J1 candidate cache
            maintained by the fixed-mesh streaming containers; ``None``
            (default) keeps the stateless kernels.
        j3_candidate_state: Optional projection-scoped J3 candidate cache
            (same contract as ``j1_candidate_state``).
        j2_candidate_state: Optional projection-scoped J2 candidate cache
            (same contract as ``j1_candidate_state``).
        j_dtn_candidate_state: Optional projection-scoped DTN candidate cache
            (same contract as ``j1_candidate_state``).

    Returns:
        jax.Array: Jastrow ratios, shape ``(N_grid,)`` (includes ``exp``).
    """
    dtn = jastrow_data.jastrow_dtn_data
    # Per-channel dispatch. DTN owns a fused (m, k) kernel, so it is excluded
    # from the generic per-config route and supplied from that kernel instead;
    # every other channel has no mesh form and stays on the generic route.
    J_ratio = _compute_ratio_Jastrow_part_rank1_update(
        jastrow_data=jastrow_data,
        old_r_up_carts=old_r_up_carts,
        old_r_dn_carts=old_r_dn_carts,
        new_r_up_carts_arr=new_r_up_carts_arr,
        new_r_dn_carts_arr=new_r_dn_carts_arr,
        j3_state=j3_state,
        j1_candidate_state=j1_candidate_state,
        j3_candidate_state=j3_candidate_state,
        j2_candidate_state=j2_candidate_state,
        _dtn_excluded=dtn is not None,
        _route_mesh="kinetic",
    )
    if dtn is not None:
        # DTN is served HERE (exact-once: the generic route dropped its leaf),
        # so this is the branch that decides the DTN route for the 6N lattice.
        if j_dtn_candidate_state is not None:
            note_route("dtn", "kinetic", cached=True)
            # Projection-scoped cache: the (Ne, K) table IS the dense kernel's
            # output for the current configuration, maintained incrementally.
            # Row-major reshape matches the mesh flattening (up block m-major /
            # k-minor, then dn) exactly like the dense return.
            J_ratio = J_ratio * jnp.exp(j_dtn_candidate_state.log_delta.reshape(-1))
        else:
            note_route("dtn", "kinetic", cached=False)
            delta_fused = _compute_delta_J_DTN_mesh_dense(dtn, old_r_up_carts, old_r_dn_carts, shifts)
            J_ratio = J_ratio * jnp.exp(delta_fused.reshape(-1))
    return J_ratio


@struct.dataclass
class J3_candidate_mesh_state:
    """J3-orbital cache for a fixed one-electron candidate lattice.

    This cache is deliberately independent of the determinant candidate
    cache: J3 and the determinant are allowed to use different orbital/AO
    bases.  An accepted hop refreshes only one contiguous candidate block.
    """

    aos_up_candidates: jax.Array
    aos_dn_candidates: jax.Array
    candidates_per_electron: int = struct.field(pytree_node=False)


@struct.dataclass
class J1_candidate_mesh_state:
    """Incrementally maintained J1 log-ratio table for one-move candidates.

    J1 is a pure per-electron sum (each electron's contribution depends only
    on its own position relative to the atoms) with NO cross-electron
    coupling, unlike J2/J3/DTN. A hop therefore changes only the moved
    electron's own row -- every other electron's candidate row is exactly
    unchanged, so the advance needs no correction pass over the rest of the
    table at all.
    """

    log_ratio_up: jax.Array
    log_ratio_dn: jax.Array


def _j1_candidate_row(
    j1_data: Jastrow_one_body_data,
    r_i: jax.Array,
    candidates_i: jax.Array,
) -> jax.Array:
    """log(J1(candidate)/J1(r_i)) for one electron's K candidates."""
    old = _compute_Jastrow_one_body_electron_contrib_ratio(j1_data, r_i)
    new = vmap(_compute_Jastrow_one_body_electron_contrib_ratio, in_axes=(None, 0))(j1_data, candidates_i)
    return new - old


@jit
def _init_j1_candidate_mesh_state(
    j1_data: Jastrow_one_body_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    candidate_up: jax.Array,
    candidate_dn: jax.Array,
) -> J1_candidate_mesh_state:
    log_up = vmap(_j1_candidate_row, in_axes=(None, 0, 0))(j1_data, r_up, candidate_up)
    log_dn = vmap(_j1_candidate_row, in_axes=(None, 0, 0))(j1_data, r_dn, candidate_dn)
    return J1_candidate_mesh_state(log_ratio_up=log_up, log_ratio_dn=log_dn)


@jit
def _advance_j1_candidate_mesh_state(
    j1_data: Jastrow_one_body_data,
    state: J1_candidate_mesh_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_new: jax.Array,
    r_dn_new: jax.Array,
    candidate_up_new: jax.Array,
    candidate_dn_new: jax.Array,
) -> J1_candidate_mesh_state:
    """Rebuild only the moved electron's row (see the class docstring: J1 has
    no cross-electron term, so this is the WHOLE advance -- contrast
    J2/J3/DTN, which must additionally correct every other row)."""
    n_up = state.log_ratio_up.shape[0]
    new_base = jnp.concatenate([r_up_new, r_dn_new], axis=0)
    candidates_new = jnp.concatenate([candidate_up_new, candidate_dn_new], axis=0)
    global_index = jnp.where(moved_spin_is_up, moved_index, n_up + moved_index)
    logs = jnp.concatenate([state.log_ratio_up, state.log_ratio_dn], axis=0)

    moved_pos = new_base[global_index]
    moved_candidates = lax.dynamic_slice(candidates_new, (global_index, 0, 0), (1, candidates_new.shape[1], 3))[0]
    row = _j1_candidate_row(j1_data, moved_pos, moved_candidates)[None, :]
    logs = lax.dynamic_update_slice(logs, row, (global_index, 0))
    return J1_candidate_mesh_state(log_ratio_up=logs[:n_up], log_ratio_dn=logs[n_up:])


@struct.dataclass
class J2_candidate_mesh_state:
    """Incrementally maintained J2 log-ratio table for one-move candidates."""

    log_ratio_up: jax.Array
    log_ratio_dn: jax.Array


def _j2_mesh_distance(
    j2_data: Jastrow_two_body_data,
    a: jax.Array,
    b: jax.Array,
    dtype_jnp: jnpt.DTypeLike,
) -> jax.Array:
    diff = a - b
    if j2_data.structure_data.pbc_flag:
        cell = j2_data.structure_data._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)
        return _smooth_periodic_distance(diff.astype(dtype_jnp), cell.astype(dtype_jnp), inv_cell.astype(dtype_jnp))
    d = diff.astype(dtype_jnp)
    d2 = jnp.sum(d * d, axis=-1)
    safe = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
    return jnp.where(d2 > 0.0, jnp.sqrt(safe), jnp.zeros_like(d2))


def _j2_value_from_distance(j2_data: Jastrow_two_body_data, distance: jax.Array) -> jax.Array:
    param = j2_data.jastrow_2b_param
    if j2_data.jastrow_2b_type == "pade":
        return distance / 2.0 * (1.0 + param * distance) ** (-1.0)
    return 1.0 / (2.0 * param) * (-jnp.expm1(-param * distance))


@jit
def _init_j2_candidate_mesh_state(
    j2_data: Jastrow_two_body_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    candidate_up: jax.Array,
    candidate_dn: jax.Array,
) -> J2_candidate_mesh_state:
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    base = jnp.concatenate([r_up, r_dn], axis=0)
    candidates = jnp.concatenate([candidate_up, candidate_dn], axis=0)
    # (m, k, j): candidate m,k against every current electron j.
    d_new = _j2_mesh_distance(j2_data, candidates[:, :, None, :], base[None, None, :, :], dtype_jnp)
    new_sum = jnp.sum(_j2_value_from_distance(j2_data, d_new), axis=2)
    d_old = _j2_mesh_distance(j2_data, base[:, None, :], base[None, :, :], dtype_jnp)
    old_sum = jnp.sum(_j2_value_from_distance(j2_data, d_old), axis=1)
    # The candidate-vs-old-base sum contains j=m, whereas the proposed
    # configuration's self term is f(candidate,candidate)=0.
    self_d = _j2_mesh_distance(j2_data, candidates, base[:, None, :], dtype_jnp)
    logs = new_sum - _j2_value_from_distance(j2_data, self_d) - old_sum[:, None]
    n_up = r_up.shape[0]
    return J2_candidate_mesh_state(log_ratio_up=logs[:n_up], log_ratio_dn=logs[n_up:])


@jit
def _advance_j2_candidate_mesh_state(
    j2_data: Jastrow_two_body_data,
    state: J2_candidate_mesh_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_old: jax.Array,
    r_dn_old: jax.Array,
    r_up_new: jax.Array,
    r_dn_new: jax.Array,
    candidate_up_new: jax.Array,
    candidate_dn_new: jax.Array,
) -> J2_candidate_mesh_state:
    """Apply the one-electron J2 delta to all rows; rebuild the moved row."""
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    n_up = r_up_old.shape[0]
    old_base = jnp.concatenate([r_up_old, r_dn_old], axis=0)
    new_base = jnp.concatenate([r_up_new, r_dn_new], axis=0)
    candidates_old = jnp.concatenate(
        [
            # Unmoved rows are identical in old/new candidate arrays.  The
            # moved row is overwritten below, so using the new table here is
            # harmless for it.
            candidate_up_new,
            candidate_dn_new,
        ],
        axis=0,
    )
    logs = jnp.concatenate([state.log_ratio_up, state.log_ratio_dn], axis=0)
    global_index = jnp.where(moved_spin_is_up, moved_index, n_up + moved_index)
    old_pos = old_base[global_index]
    new_pos = new_base[global_index]

    cand_new_d = _j2_mesh_distance(j2_data, candidates_old, new_pos, dtype_jnp)
    base_new_d = _j2_mesh_distance(j2_data, new_base[:, None, :], new_pos, dtype_jnp)
    cand_old_d = _j2_mesh_distance(j2_data, candidates_old, old_pos, dtype_jnp)
    base_old_d = _j2_mesh_distance(j2_data, old_base[:, None, :], old_pos, dtype_jnp)
    correction = (
        _j2_value_from_distance(j2_data, cand_new_d)
        - _j2_value_from_distance(j2_data, base_new_d)
        - _j2_value_from_distance(j2_data, cand_old_d)
        + _j2_value_from_distance(j2_data, base_old_d)
    )
    logs = logs + correction

    # Rebuild only the moved electron's row, O(K*N_e), not the whole
    # O(K*N_e^2) table.
    moved_candidates = lax.dynamic_slice(candidates_old, (global_index, 0, 0), (1, candidates_old.shape[1], 3))[0]
    moved_base = new_base[global_index]
    d_row_new = _j2_mesh_distance(j2_data, moved_candidates[:, None, :], new_base[None, :, :], dtype_jnp)
    row_new_sum = jnp.sum(_j2_value_from_distance(j2_data, d_row_new), axis=1)
    d_row_old = _j2_mesh_distance(j2_data, moved_base, new_base, dtype_jnp)
    row_old_sum = jnp.sum(_j2_value_from_distance(j2_data, d_row_old))
    d_self = _j2_mesh_distance(j2_data, moved_candidates, moved_base, dtype_jnp)
    moved_row = (row_new_sum - _j2_value_from_distance(j2_data, d_self) - row_old_sum)[None, :]
    logs = lax.dynamic_update_slice(logs, moved_row, (global_index, 0))
    return J2_candidate_mesh_state(log_ratio_up=logs[:n_up], log_ratio_dn=logs[n_up:])


@jit
def _init_j3_candidate_mesh_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    candidate_up_positions: jax.Array,
    candidate_dn_positions: jax.Array,
) -> J3_candidate_mesh_state:
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    k = candidate_up_positions.shape[1] if candidate_up_positions.shape[0] else candidate_dn_positions.shape[1]
    up_flat = candidate_up_positions.reshape(-1, 3)
    dn_flat = candidate_dn_positions.reshape(-1, 3)
    aos_up = jnp.asarray(
        jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, up_flat)),
        dtype=dtype_jnp,
    )
    aos_dn = jnp.asarray(
        jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, dn_flat)),
        dtype=dtype_jnp,
    )
    return J3_candidate_mesh_state(
        aos_up_candidates=aos_up,
        aos_dn_candidates=aos_dn,
        candidates_per_electron=k,
    )


@jit
def _advance_j3_candidate_mesh_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    state: J3_candidate_mesh_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    candidate_up_positions_new: jax.Array,
    candidate_dn_positions_new: jax.Array,
) -> J3_candidate_mesh_state:
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    k = state.candidates_per_electron

    def _up(_: None) -> J3_candidate_mesh_state:
        pos = lax.dynamic_slice(candidate_up_positions_new, (moved_index, 0, 0), (1, k, 3)).reshape(k, 3)
        fresh = jnp.asarray(
            jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, pos)),
            dtype=dtype_jnp,
        )
        return state.replace(aos_up_candidates=lax.dynamic_update_slice(state.aos_up_candidates, fresh, (0, moved_index * k)))

    def _dn(_: None) -> J3_candidate_mesh_state:
        pos = lax.dynamic_slice(candidate_dn_positions_new, (moved_index, 0, 0), (1, k, 3)).reshape(k, 3)
        fresh = jnp.asarray(
            jnp.real(jastrow_three_body_data.compute_orb_api(jastrow_three_body_data.orb_data, pos)),
            dtype=dtype_jnp,
        )
        return state.replace(aos_dn_candidates=lax.dynamic_update_slice(state.aos_dn_candidates, fresh, (0, moved_index * k)))

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


@struct.dataclass
class J_DTN_candidate_mesh_state:
    """Incrementally maintained DTN log-delta table for one-move candidates.

    Fields live in the GLOBAL electron frame (up-then-dn), unlike the split
    ``J2_candidate_mesh_state`` storage: the advance corrects every row and
    scatters exactly one mover row, so a single concatenated table avoids the
    per-spin branch (and thus any tree-select) entirely. Consumers split with
    the STATIC ``num_up`` they already hold.

    - ``log_delta``: ``(Ne, K)`` -- ``J_DTN(candidate) - J_DTN(current)`` for
      every (electron ``m``, candidate ``k``); exactly the output of
      :func:`_compute_delta_J_DTN_mesh_dense` for the CURRENT configuration.
    - ``cand_r_eI``: ``(Ne, K, Na)`` -- candidate--atom distances. Unmoved
      rows are geometry-invariant across hops (a candidate of electron ``m``
      depends only on ``r_m``), so caching them spares the advance the full
      candidate--atom distance rebuild; only the f-channel correction and the
      mover-row scatter read/write it.
    """

    log_delta: jax.Array
    cand_r_eI: jax.Array


@jit
def _init_j_dtn_candidate_mesh_state(
    dtn: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    candidate_up: jax.Array,
    candidate_dn: jax.Array,
) -> J_DTN_candidate_mesh_state:
    """Build the DTN candidate table from scratch (once per projection interval).

    ``candidate_up`` / ``candidate_dn`` are the container's ``(N_spin, K, 3)``
    per-electron candidate lattices. For the kinetic mesh they equal
    ``r + shifts`` built in the SAME caller precision the dense kernel uses
    internally, so the initial ``log_delta`` matches
    :func:`_compute_delta_J_DTN_mesh_dense` bit-for-bit (identical einsum
    shapes and strings via the shared :func:`_dtn_candidate_delta_rows`).
    """
    all_e = jnp.concatenate((r_up, r_dn), axis=0)
    candidates = jnp.concatenate((candidate_up, candidate_dn), axis=0)  # (Ne, K, 3)
    pos = jnp.concatenate((all_e[:, None, :], candidates), axis=1)  # (Ne, K+1, 3)
    delta, cand_r_eI = _dtn_candidate_delta_rows(dtn, pos, all_e, r_up.shape[0], jnp.arange(all_e.shape[0]))
    return J_DTN_candidate_mesh_state(log_delta=delta, cand_r_eI=cand_r_eI)


@jit
def _advance_j_dtn_candidate_mesh_state(
    dtn: Jastrow_DTN_data,
    state: J_DTN_candidate_mesh_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_old: jax.Array,
    r_dn_old: jax.Array,
    r_up_new: jax.Array,
    r_dn_new: jax.Array,
    candidate_up_new: jax.Array,
    candidate_dn_new: jax.Array,
) -> J_DTN_candidate_mesh_state:
    """Apply the one-electron DTN delta to all rows; rebuild the moved row.

    Same shape as :func:`_advance_j2_candidate_mesh_state`: with electron
    ``m0`` hopping ``o -> n``, every OTHER row's ``delta(m, k)`` changes only
    through its pair terms with ``m0``, giving the four-term correction

        ``P(c_{m,k}, n) - P(r_m, n) - P(c_{m,k}, o) + P(r_m, o)``

    where ``P`` is the pair channel sum: the u term
    ``cut_u(r) * sum_o alpha[s, o] r^o`` (spin channel ``s`` from GLOBAL index
    parity of ``m`` vs ``m0``) plus the atom-summed f term
    ``sum_I cut_I(a) cut_I(b) sum_{l,m',n'} gamma_I |a-R_I|^l |b-R_I|^{m'} |a-b|^{n'}``.
    chi has no pair dependence, so it appears only in the mover-row rebuild.
    The mover's own row is recomputed EXACTLY (fresh evaluation via the shared
    dense-algebra core) -- correction round-off accumulates on non-moved rows
    only, matching the J2 cache's accepted behaviour.

    Cost: O(Ne*K*Na*poly) for the f correction + O(K*Ne*Na*poly) for the row
    rebuild, vs the dense kernel's O(Ne*K*Ne*Na*poly) full rebuild.
    """
    # Zero-electron edge (Python-static, BEFORE any traced op): nothing can
    # move, the table is empty. A FULLY-POLARIZED system (one spin empty) is
    # absorbed by the global-frame concatenations below; its mover is clamped
    # onto the table with the other row reads (see ``global_index``).
    if candidate_up_new.shape[0] == 0 and candidate_dn_new.shape[0] == 0:
        return state

    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    n_up = r_up_old.shape[0]
    old_base = jnp.concatenate((r_up_old, r_dn_old), axis=0)  # (Ne, 3)
    new_base = jnp.concatenate((r_up_new, r_dn_new), axis=0)
    # Unmoved rows are identical in old/new candidate arrays (a candidate of
    # electron m depends only on r_m). The moved row is rebuilt below, so
    # using the new lattice everywhere is harmless for it.
    candidates = jnp.concatenate((candidate_up_new, candidate_dn_new), axis=0)  # (Ne, K, 3)
    n_e = new_base.shape[0]
    n_k = candidates.shape[1]
    C = dtn.cutoff_smoothness
    atoms = dtn.structure_data._positions_cart_jnp
    species = jnp.asarray(dtn.species_index, dtype=jnp.int32)

    # Clamp the mover onto the table ONCE so every consumer names the same
    # electron. The gathers / dynamic slices below clamp an out-of-range index
    # anyway, but the u-channel spin parity and the ``_dtn_candidate_delta_rows``
    # self mask read the index itself: a nonexistent mover (``n_up + 0`` with an
    # empty dn channel, issue #784 B) rebuilt the last electron's row with its
    # own pair. In range (every real hop) the clamp is the identity.
    global_index = jnp.clip(jnp.where(moved_spin_is_up, moved_index, n_up + moved_index), 0, n_e - 1)
    old_pos = old_base[global_index]
    new_pos = new_base[global_index]

    # Distances of every candidate and every unmoved base position to the
    # mover's old/new positions -- the only fresh e-e geometry the
    # corrections need. Row m0's entries are garbage-tolerant (finite by the
    # safe-norm/0^0 conventions) and overwritten by the rebuild.
    cand_flat = candidates.reshape(n_e * n_k, 3)
    d_cand_new = _dtn_distance_rows(dtn, cand_flat, new_pos[None, :], dtype_jnp).reshape(n_e, n_k)
    d_cand_old = _dtn_distance_rows(dtn, cand_flat, old_pos[None, :], dtype_jnp).reshape(n_e, n_k)
    d_base_new = _dtn_distance_rows(dtn, new_base, new_pos[None, :], dtype_jnp)[:, 0]  # (Ne,)
    d_base_old = _dtn_distance_rows(dtn, old_base, old_pos[None, :], dtype_jnp)[:, 0]

    corr = jnp.zeros((n_e, n_k), dtype=dtype_jnp)

    if dtn.use_u:
        alpha_full = _dtn_inflate_alpha(dtn, dtype_jnp)
        L_u = _dtn_effective_cutoffs(dtn, dtype_jnp)[0]
        k_u = jnp.arange(dtn.n_order_u + 1, dtype=dtype_jnp)
        same = (jnp.arange(n_e) < n_up) == (global_index < n_up)  # (Ne,) parity vs the mover
        alpha_rows = jnp.where(same[:, None], alpha_full[0][None, :], alpha_full[1][None, :])  # (Ne, o)
        poly_cand_new = jnp.einsum("mko,mo->mk", d_cand_new[..., None] ** k_u, alpha_rows)
        poly_cand_old = jnp.einsum("mko,mo->mk", d_cand_old[..., None] ** k_u, alpha_rows)
        u_cand_new = jnp.maximum(L_u - d_cand_new, 0.0) ** C * poly_cand_new
        u_cand_old = jnp.maximum(L_u - d_cand_old, 0.0) ** C * poly_cand_old
        u_base_new = jnp.maximum(L_u - d_base_new, 0.0) ** C * jnp.einsum("mo,mo->m", d_base_new[..., None] ** k_u, alpha_rows)
        u_base_old = jnp.maximum(L_u - d_base_old, 0.0) ** C * jnp.einsum("mo,mo->m", d_base_old[..., None] ** k_u, alpha_rows)
        corr = corr + (u_cand_new - u_cand_old) - (u_base_new - u_base_old)[:, None]

    if dtn.use_f:
        gamma_atoms = _dtn_inflate_gamma(dtn, dtype_jnp)[species]  # (Na, l, m', n)
        L_f_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[2][species]  # (Na,)
        k_eN = jnp.arange(dtn.n_order_eN + 1, dtype=dtype_jnp)
        k_ee = jnp.arange(dtn.n_order_ee + 1, dtype=dtype_jnp)
        # Mover-side atom factors at the new/old positions with the mover's
        # eN polynomial and cutoff folded in -- the j-side ``g_side`` of the
        # dense kernel restricted to j = m0.
        r_xI = _dtn_distance_rows(dtn, jnp.stack((new_pos, old_pos)), atoms, dtype_jnp)  # (2, Na)
        cut_x = jnp.maximum(L_f_atoms[None, :] - r_xI, 0.0) ** C
        pow_x = r_xI[..., None] ** k_eN  # (2, Na, m')
        g_x = jnp.einsum("Ilmn,xIm,xI->xIln", gamma_atoms, pow_x, cut_x)  # (2, Na, l, n)
        # Row-side legs: candidates from the CACHED candidate--atom distances,
        # base positions fresh (O(Ne*Na); new_base == old_base off the mover
        # row, which is overwritten anyway).
        cut_cand = jnp.maximum(L_f_atoms[None, None, :] - state.cand_r_eI, 0.0) ** C  # (Ne, K, Na)
        leg_cand = cut_cand[..., None] * (state.cand_r_eI[..., None] ** k_eN)  # (Ne, K, Na, l)
        r_bI = _dtn_distance_rows(dtn, new_base, atoms, dtype_jnp)  # (Ne, Na)
        cut_b = jnp.maximum(L_f_atoms[None, :] - r_bI, 0.0) ** C
        leg_base = cut_b[..., None] * (r_bI[..., None] ** k_eN)  # (Ne, Na, l)
        f_cand_new = jnp.einsum("mkIl,Iln,mkn->mk", leg_cand, g_x[0], d_cand_new[..., None] ** k_ee)
        f_cand_old = jnp.einsum("mkIl,Iln,mkn->mk", leg_cand, g_x[1], d_cand_old[..., None] ** k_ee)
        f_base_new = jnp.einsum("mIl,Iln,mn->m", leg_base, g_x[0], d_base_new[..., None] ** k_ee)
        f_base_old = jnp.einsum("mIl,Iln,mn->m", leg_base, g_x[1], d_base_old[..., None] ** k_ee)
        corr = corr + (f_cand_new - f_cand_old) - (f_base_new - f_base_old)[:, None]

    log_delta = state.log_delta + corr

    # Rebuild only the moved electron's row EXACTLY on the post-hop
    # configuration: O(K*Ne*Na*poly), not the whole O(Ne*K*Ne*Na*poly) table.
    # chi enters here (it has no pair dependence, hence no correction above).
    cand_row = lax.dynamic_slice(candidates, (global_index, 0, 0), (1, n_k, 3))  # (1, K, 3)
    pos_row = jnp.concatenate((new_pos[None, None, :], cand_row), axis=1)  # (1, K+1, 3)
    row_delta, row_r_eI = _dtn_candidate_delta_rows(dtn, pos_row, new_base, n_up, global_index[None])
    log_delta = lax.dynamic_update_slice(log_delta, row_delta, (global_index, 0))
    cand_r_eI = lax.dynamic_update_slice(state.cand_r_eI, row_r_eI, (global_index, 0, 0))
    return J_DTN_candidate_mesh_state(log_delta=log_delta, cand_r_eI=cand_r_eI)


def _compute_ratio_Jastrow_part_split_spin(
    jastrow_data: Jastrow_data,
    old_r_up_carts: jax.Array,
    old_r_dn_carts: jax.Array,
    new_r_up_shifted: jax.Array,
    new_r_dn_shifted: jax.Array,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    j1_candidate_state: "J1_candidate_mesh_state | None" = None,
    j3_candidate_state: "J3_candidate_mesh_state | None" = None,
    j2_candidate_state: "J2_candidate_mesh_state | None" = None,
    j_dtn_candidate_state: "J_DTN_candidate_mesh_state | None" = None,
) -> jax.Array:
    r"""Jastrow ratio for a block-structured mesh where up and dn electrons move separately.

    Avoids computing spin-block contributions for the unchanged spin.  Compared
    with ``_compute_ratio_Jastrow_part_rank1_update`` called on the concatenated
    (G_up + G_dn) array, this evaluates only the up-spin J3 formula for the
    ``G_up`` configs and only the dn-spin J3 formula for the ``G_dn`` configs,
    replacing the ``jnp.where`` merge with separate block computations.  For
    J3 the old AOs of the moved electron are obtained by column-slicing the
    already-computed ``aos_up_old`` / ``aos_dn_old`` matrices, avoiding two
    extra ``compute_orb_api`` calls.

    When called from the projection streaming path, the caller may pass
    ``j3_state`` to skip recomputing ``aos_*_old`` and the ``W``/``U``/cross_vec
    products -- see ``_compute_ratio_Jastrow_part_rank1_update`` for the exact
    correspondence.

    Args:
        jastrow_data: Active Jastrow components.
        old_r_up_carts: Reference up-spin coordinates ``(N_up, 3)``.
        old_r_dn_carts: Reference dn-spin coordinates ``(N_dn, 3)``.
        new_r_up_shifted: Up-block proposed coords ``(G_up, N_up, 3)``.  Exactly one
            up electron differs from ``old_r_up_carts`` per config.
        new_r_dn_shifted: Dn-block proposed coords ``(G_dn, N_dn, 3)``.  Exactly one
            dn electron differs from ``old_r_dn_carts`` per config.
        j3_state: Optional cached J3 auxiliaries consistent with the old
            coordinates (see ``_compute_ratio_Jastrow_part_rank1_update``).
        j1_candidate_state: Optional projection-scoped J1 candidate cache
            maintained by the fixed-mesh streaming containers; ``None``
            (default) evaluates the factor directly.
        j3_candidate_state: Optional projection-scoped J3 candidate cache
            (same contract as ``j1_candidate_state``).
        j2_candidate_state: Optional projection-scoped J2 candidate cache
            (same contract as ``j1_candidate_state``).
        j_dtn_candidate_state: Optional projection-scoped DTN candidate cache
            (same contract as ``j1_candidate_state``).

    Returns:
        jax.Array: Concatenated Jastrow ratios ``(G_up + G_dn,)`` (includes ``exp``).

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
    # Forward old/new r_up/dn_carts as-is (Principle 3a -- no parameter rebind).
    # Module-level forwards (compute_Jastrow_one_body, compute_orb_api,
    # _compute_ratio_Jastrow_part_rank1_update, NN_Jastrow.apply) handle their
    # own use-site casts. Inline diffs in the local J2 _safe_norm closure cast
    # operands to the jastrow_ratio zone at the use site (Principle 3b).
    dtype_jnp = get_dtype_jnp("jastrow_ratio")

    num_up = old_r_up_carts.shape[0]
    num_dn = old_r_dn_carts.shape[0]

    # A fully spin-polarised configuration (one spin channel empty) runs the
    # same block formulas: every block of an empty channel is skipped
    # statically (its argmax over a zero-size electron axis would raise at
    # trace time, and nothing in it can move), so J(R') - J(R) stays in
    # ``jastrow_ratio`` as for two spins. Earlier this case delegated to the
    # rank-one kernel's whole ``compute_Jastrow_part`` recompute, i.e. to
    # ``jastrow_eval`` (float32 in mixed; issue #775).
    g_up = new_r_up_shifted.shape[0]
    g_dn = new_r_dn_shifted.shape[0]

    # Precompute moved-electron indices for both blocks; a block exists only
    # when its spin channel has electrons (``has_up`` / ``has_dn``).
    has_up = num_up > 0
    has_dn = num_dn > 0
    if has_up:
        delta_up_block = new_r_up_shifted - old_r_up_carts[None]  # (G_up, N_up, 3)
        idx_up_block = jnp.argmax(jnp.any(delta_up_block != 0.0, axis=2).astype(jnp.int32), axis=1)  # (G_up,)
        moved_up_block = jnp.any(delta_up_block != 0.0, axis=(1, 2))  # (G_up,)
        # New / old position of the moved electron in the block.
        r_up_moved = jnp.take_along_axis(new_r_up_shifted, idx_up_block[:, None, None], axis=1).reshape(g_up, 3)  # (G_up, 3)
        r_up_old_moved = old_r_up_carts[idx_up_block]  # (G_up, 3)
    if has_dn:
        delta_dn_block = new_r_dn_shifted - old_r_dn_carts[None]  # (G_dn, N_dn, 3)
        idx_dn_block = jnp.argmax(jnp.any(delta_dn_block != 0.0, axis=2).astype(jnp.int32), axis=1)  # (G_dn,)
        moved_dn_block = jnp.any(delta_dn_block != 0.0, axis=(1, 2))  # (G_dn,)
        r_dn_moved = jnp.take_along_axis(new_r_dn_shifted, idx_dn_block[:, None, None], axis=1).reshape(g_dn, 3)  # (G_dn, 3)
        r_dn_old_moved = old_r_dn_carts[idx_dn_block]  # (G_dn, 3)

    J_up = jnp.ones(g_up, dtype=dtype_jnp)
    J_dn = jnp.ones(g_dn, dtype=dtype_jnp)

    # -- J1 part --------------------------------------------------------------
    if jastrow_data.jastrow_one_body_data is not None and j1_candidate_state is not None:
        note_route("j1", "ecp", cached=True)
        J_up = J_up * jnp.exp(j1_candidate_state.log_ratio_up.reshape(-1))
        J_dn = J_dn * jnp.exp(j1_candidate_state.log_ratio_dn.reshape(-1))
    if jastrow_data.jastrow_one_body_data is not None and j1_candidate_state is None:
        note_route("j1", "ecp", cached=False)
        j1_data = jastrow_data.jastrow_one_body_data

        if has_up:
            # UP block: only the moved up electron contributes to the J1 change.
            def compute_J1_up_one(r_up_new: jax.Array, r_up_old: jax.Array) -> jax.Array:
                j1_new = _compute_Jastrow_one_body_electron_contrib_ratio(j1_data, r_up_new)
                j1_old = _compute_Jastrow_one_body_electron_contrib_ratio(j1_data, r_up_old)
                return jnp.exp(j1_new - j1_old)

            J1_up_block = vmap(compute_J1_up_one)(r_up_moved, r_up_old_moved)  # (G_up,)
            J_up = J_up * jnp.ravel(J1_up_block)

        if has_dn:
            # DN block: only the moved dn electron contributes.
            def compute_J1_dn_one(r_dn_new: jax.Array, r_dn_old: jax.Array) -> jax.Array:
                j1_new = _compute_Jastrow_one_body_electron_contrib_ratio(j1_data, r_dn_new)
                j1_old = _compute_Jastrow_one_body_electron_contrib_ratio(j1_data, r_dn_old)
                return jnp.exp(j1_new - j1_old)

            J1_dn_block = vmap(compute_J1_dn_one)(r_dn_moved, r_dn_old_moved)  # (G_dn,)
            J_dn = J_dn * jnp.ravel(J1_dn_block)

    # -- J2 part --------------------------------------------------------------
    if jastrow_data.jastrow_two_body_data is not None and j2_candidate_state is not None:
        note_route("j2", "ecp", cached=True)
        J_up = J_up * jnp.exp(j2_candidate_state.log_ratio_up.reshape(-1))
        J_dn = J_dn * jnp.exp(j2_candidate_state.log_ratio_dn.reshape(-1))
    if jastrow_data.jastrow_two_body_data is not None and j2_candidate_state is None:
        note_route("j2", "ecp", cached=False)
        j2_param = jastrow_data.jastrow_two_body_data.jastrow_2b_param
        _j2_type_split = jastrow_data.jastrow_two_body_data.jastrow_2b_type

        # PBC dispatch: the electron--electron distance must match
        # the Williamson smooth periodic distance used by the production J2
        # evaluator (``compute_Jastrow_two_body``); otherwise the split-spin
        # ratio disagrees with the full recompute under PBC -- as in
        # ``_compute_ratio_Jastrow_part_rank1_update``. ``sd_j2.pbc_flag`` is
        # ``pytree_node=False`` so this resolves at trace time (molecular branch
        # byte-identical). ``_safe_norm`` reduces the last axis, so the single
        # definition serves both the (M, K, 3) pairwise and (G, 3) self callers.
        sd_j2_split = jastrow_data.jastrow_two_body_data.structure_data
        if sd_j2_split.pbc_flag:
            cell_j2_split = sd_j2_split._cell_jnp.astype(jnp.float64)
            inv_cell_j2_split = _inv_3x3(cell_j2_split)

            def _safe_norm(diff: jax.Array) -> jax.Array:
                # Cast diff (caller-supplied precision) to the jastrow_ratio zone
                # at the use site (Principle 3b); `diff` itself stays frozen (3a).
                d = diff.astype(dtype_jnp)
                return _smooth_periodic_distance(d, cell_j2_split.astype(dtype_jnp), inv_cell_j2_split.astype(dtype_jnp))
        else:

            def _safe_norm(diff: jax.Array) -> jax.Array:
                # Cast diff (reconstructed in caller-supplied precision by the
                # caller, e.g. `pos1 - pos2`) to the jastrow_ratio zone at the use
                # site (Principle 3b). New variable name `d` keeps the parameter
                # `diff` itself frozen (Principle 3a).
                d = diff.astype(dtype_jnp)
                sq = jnp.sum(d**2, axis=-1)
                return jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, jnp.ones_like(sq))), jnp.zeros_like(sq))

        if _j2_type_split == "pade":

            def _j2_from_dist_split(dist: jax.Array, param: jax.Array) -> jax.Array:
                return dist / 2.0 * (1.0 + param * dist) ** (-1.0)
        else:

            def _j2_from_dist_split(dist: jax.Array, param: jax.Array) -> jax.Array:
                return 1.0 / (2.0 * param) * (-jnp.expm1(-param * dist))

        def _pairwise_sums(pos1: jax.Array, pos2: jax.Array) -> jax.Array:
            """Row-wise sum of two-body terms: pos1 -> all of pos2."""
            dists = _safe_norm(pos1[:, None, :] - pos2[None, :, :])
            return jnp.sum(_j2_from_dist_split(dists, j2_param), axis=1)

        J2_sum_up_up = _pairwise_sums(old_r_up_carts, old_r_up_carts)  # (N_up,)
        J2_sum_up_dn = _pairwise_sums(old_r_up_carts, old_r_dn_carts)  # (N_up,)
        J2_sum_dn_dn = _pairwise_sums(old_r_dn_carts, old_r_dn_carts)  # (N_dn,)
        J2_sum_dn_up = _pairwise_sums(old_r_dn_carts, old_r_up_carts)  # (N_dn,)

        if has_up:
            # UP block: moved up electron interacts with all old up and dn electrons.
            dists_up_up_new = _safe_norm(r_up_moved[:, None, :] - old_r_up_carts[None, :, :])  # (G_up, N_up)
            J2_up_up_new = jnp.sum(_j2_from_dist_split(dists_up_up_new, j2_param), axis=1)  # (G_up,)
            # The sum above includes the self-term f(r_new, r_old[idx]) but the correct self-term is
            # f(r_new, r_new) = 0.  Subtract the spurious contribution.
            dists_self_up = _safe_norm(r_up_moved - r_up_old_moved)  # (G_up,)
            J2_up_up_new = J2_up_up_new - _j2_from_dist_split(dists_self_up, j2_param)
            J2_up_up_old = J2_sum_up_up[idx_up_block]  # (G_up,)  self-term is f(r_old,r_old)=0 already
            dists_up_dn_new = _safe_norm(r_up_moved[:, None, :] - old_r_dn_carts[None, :, :])  # (G_up, N_dn)
            J2_up_dn_new = jnp.sum(_j2_from_dist_split(dists_up_dn_new, j2_param), axis=1)  # (G_up,)
            J2_up_dn_old = J2_sum_up_dn[idx_up_block]  # (G_up,)
            J_up = J_up * jnp.exp(J2_up_dn_new - J2_up_dn_old + J2_up_up_new - J2_up_up_old)

        if has_dn:
            # DN block: moved dn electron interacts with all old dn and up electrons.
            dists_dn_dn_new = _safe_norm(r_dn_moved[:, None, :] - old_r_dn_carts[None, :, :])  # (G_dn, N_dn)
            J2_dn_dn_new = jnp.sum(_j2_from_dist_split(dists_dn_dn_new, j2_param), axis=1)  # (G_dn,)
            # Same self-term correction for down block.
            dists_self_dn = _safe_norm(r_dn_moved - r_dn_old_moved)  # (G_dn,)
            J2_dn_dn_new = J2_dn_dn_new - _j2_from_dist_split(dists_self_dn, j2_param)
            J2_dn_dn_old = J2_sum_dn_dn[idx_dn_block]  # (G_dn,)
            dists_dn_up_new = _safe_norm(r_dn_moved[:, None, :] - old_r_up_carts[None, :, :])  # (G_dn, N_up)
            J2_dn_up_new = jnp.sum(_j2_from_dist_split(dists_dn_up_new, j2_param), axis=1)  # (G_dn,)
            J2_dn_up_old = J2_sum_dn_up[idx_dn_block]  # (G_dn,)
            J_dn = J_dn * jnp.exp(J2_dn_up_new - J2_dn_up_old + J2_dn_dn_new - J2_dn_dn_old)

    # -- J3 part --------------------------------------------------------------
    if jastrow_data.jastrow_three_body_data is not None:
        j3d = jastrow_data.jastrow_three_body_data
        j3_mat = j3d._j_matrix_jnp[:, :-1]  # (n_ao, n_ao)
        j1_vec = j3d._j_matrix_jnp[:, -1]  # (n_ao,)

        # Old AOs evaluated once; column slices give old AO at each moved position.
        # When the streaming state is provided, all four matmuls and both
        # cross_vec products come from the cache (Python-static dispatch).
        if j3_state is None:
            aos_up_old = jnp.array(jnp.real(j3d.compute_orb_api(j3d.orb_data, old_r_up_carts)), dtype=dtype_jnp)  # (n_ao, N_up)
            aos_dn_old = jnp.array(jnp.real(j3d.compute_orb_api(j3d.orb_data, old_r_dn_carts)), dtype=dtype_jnp)  # (n_ao, N_dn)
            # Precompute constant products (shared between blocks).
            W_up = jnp.dot(j3_mat, aos_up_old)  # (n_ao, N_up)
            U_up = jnp.dot(aos_up_old.T, j3_mat)  # (N_up, n_ao)
            W_dn = jnp.dot(j3_mat, aos_dn_old)  # (n_ao, N_dn)
            U_dn = jnp.dot(aos_dn_old.T, j3_mat)  # (N_dn, n_ao)
            dn_cross_vec = j3_mat @ jnp.sum(aos_dn_old, axis=1)  # (n_ao,): UP cross term constant
            up_cross_vec = jnp.sum(aos_up_old, axis=1) @ j3_mat  # (n_ao,): DN cross term constant
        else:
            aos_up_old = j3_state.aos_up.astype(dtype_jnp)
            aos_dn_old = j3_state.aos_dn.astype(dtype_jnp)
            W_up = j3_state.j3_mat_aos_up.astype(dtype_jnp)
            W_dn = j3_state.j3_mat_aos_dn.astype(dtype_jnp)
            U_up = j3_state.j3_mat_T_aos_up.astype(dtype_jnp).T
            U_dn = j3_state.j3_mat_T_aos_dn.astype(dtype_jnp).T
            # See ``_compute_ratio_Jastrow_part_rank1_update`` for why the
            # cross_vec are read from the cached rank-1-advanced rowsums.
            dn_cross_vec = j3_state.j3_mat_aos_dn_rowsum.astype(dtype_jnp)
            up_cross_vec = j3_state.j3_mat_T_aos_up_rowsum.astype(dtype_jnp)

        note_route("j3", "ecp", cached=j3_candidate_state is not None)

        if has_up:
            # -- UP BLOCK ---------------------------------------------------------
            # New AOs at the moved up-electron positions; old AOs by column-slice.
            if j3_candidate_state is None:
                aos_up_new_moved = jnp.array(
                    jnp.real(j3d.compute_orb_api(j3d.orb_data, r_up_moved)), dtype=dtype_jnp
                )  # (n_ao, G_up)
            else:
                aos_up_new_moved = j3_candidate_state.aos_up_candidates.astype(dtype_jnp)
            aos_up_old_moved = aos_up_old[:, idx_up_block]  # (n_ao, G_up)
            # A candidate that coincides with the current configuration (an ECP
            # quadrature point on the electron itself, e.g. after a t-move under
            # the interval-fixed mesh rotation) has no moved electron: argmax
            # then returns 0, while the cached candidate AOs belong to the
            # candidate's own electron. Its ratio is exactly 1 (issue #652).
            aos_p_up = jnp.where(moved_up_block[None, :], aos_up_new_moved - aos_up_old_moved, 0.0)  # (n_ao, G_up)

            term1_up = j1_vec @ aos_p_up  # (G_up,)
            # tensordot avoids the explicit transpose of ``aos_p_up``; see
            # ``_compute_ratio_Jastrow_part_rank1_update`` for the same rewrite rationale.
            V_up_block = jnp.tensordot(aos_p_up, W_up, axes=((0,), (0,)))  # (G_up, N_up)
            P_up_block = jnp.dot(U_up, aos_p_up)  # (N_up, G_up)
            Q_up_c = (idx_up_block[:, None] < jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (G_up, N_up)
            Q_up_r = (idx_up_block[:, None] > jnp.arange(num_up)[None, :]).astype(dtype_jnp)  # (G_up, N_up)
            term2_up = jnp.sum(V_up_block * Q_up_c, axis=1)  # (G_up,)
            term3_up = jnp.sum(P_up_block.T * Q_up_r, axis=1)  # (G_up,)
            term4_up = dn_cross_vec @ aos_p_up  # (G_up,)
            J_up = J_up * jnp.exp(term1_up + term2_up + term3_up + term4_up)

        if has_dn:
            # -- DN BLOCK ---------------------------------------------------------
            # New AOs at the moved dn-electron positions; old AOs by column-slice.
            if j3_candidate_state is None:
                aos_dn_new_moved = jnp.array(
                    jnp.real(j3d.compute_orb_api(j3d.orb_data, r_dn_moved)), dtype=dtype_jnp
                )  # (n_ao, G_dn)
            else:
                aos_dn_new_moved = j3_candidate_state.aos_dn_candidates.astype(dtype_jnp)
            aos_dn_old_moved = aos_dn_old[:, idx_dn_block]  # (n_ao, G_dn)
            aos_p_dn = jnp.where(moved_dn_block[None, :], aos_dn_new_moved - aos_dn_old_moved, 0.0)  # (n_ao, G_dn)

            term1_dn = j1_vec @ aos_p_dn  # (G_dn,)
            # See UP-block comment re: tensordot.
            V_dn_block = jnp.tensordot(aos_p_dn, W_dn, axes=((0,), (0,)))  # (G_dn, N_dn)
            P_dn_block = jnp.dot(U_dn, aos_p_dn)  # (N_dn, G_dn)
            Q_dn_c = (idx_dn_block[:, None] < jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (G_dn, N_dn)
            Q_dn_r = (idx_dn_block[:, None] > jnp.arange(num_dn)[None, :]).astype(dtype_jnp)  # (G_dn, N_dn)
            term2_dn = jnp.sum(V_dn_block * Q_dn_c, axis=1)  # (G_dn,)
            term3_dn = jnp.sum(P_dn_block.T * Q_dn_r, axis=1)  # (G_dn,)
            term4_dn = up_cross_vec @ aos_p_dn  # (G_dn,)
            J_dn = J_dn * jnp.exp(term1_dn + term2_dn + term3_dn + term4_dn)

    # -- DTN polynomial part (rank-1 delta -- split-spin form) ---------------
    # Projection-scoped meshes carry the (Ne, K) log-delta table and advance
    # it from the accepted one-electron hop (mirrors the J2 cache above); the
    # per-electron candidate rows flatten m-major / k-minor, matching this
    # function's block config ordering.
    if jastrow_data.jastrow_dtn_data is not None and j_dtn_candidate_state is not None:
        note_route("dtn", "ecp", cached=True)
        J_up = J_up * jnp.exp(j_dtn_candidate_state.log_delta[:num_up].reshape(-1))
        J_dn = J_dn * jnp.exp(j_dtn_candidate_state.log_delta[num_up:].reshape(-1))
    # The split mesh already partitions configs by which spin moved, so the
    # per-block delta helpers (no spin dispatch) avoid the redundant
    # other-spin branch evaluation of the mixed-grid path. Same memory win
    # as the mixed path: per-block vmap intermediate scales with N_e-1
    # instead of N_pair.
    if jastrow_data.jastrow_dtn_data is not None and j_dtn_candidate_state is None:
        note_route("dtn", "ecp", cached=False)
        dtn = jastrow_data.jastrow_dtn_data
        # Stream the ECP-mesh grid configs with lax.map(batch_size=...) instead
        # of a plain vmap over all G configs. The per-config f-channel
        # intermediate (N_atom, poly, poly, N_e) -- materialised for every config
        # at once by vmap -- is the entire e_L peak and OOMs at higher walker
        # counts (24.4 GiB at 32 walkers on a periodic B-spline cell). The measured-budget batch holds only
        # that many configs live; the scan walks the rest. Pure memory<->throughput
        # knob (results change only at fp64 round-off); see
        # _setting.JASTROW_DTN_ECP_GRID_BUDGET_BYTES. This is the ECP-ratio twin of
        # the kinetic-path DTN pair budget (compute_grads_and_laplacian_Jastrow_DTN).
        if has_up:
            # UP block: only r_up changes.
            _delta_up_batch = measured_stream_batch(
                _delta_J_DTN_up_moved_probe,
                (dtn, old_r_up_carts, old_r_dn_carts),
                (jax.ShapeDtypeStruct((num_up, 3), new_r_up_shifted.dtype),),
                new_r_up_shifted.shape[0],
                JASTROW_DTN_ECP_GRID_BUDGET_BYTES,
            )
            delta_up_per_config = jax.lax.map(
                lambda cfg: _compute_delta_J_DTN_per_config_up_moved(dtn, old_r_up_carts, old_r_dn_carts, cfg),
                new_r_up_shifted,
                batch_size=_delta_up_batch,
            )
            J_up = J_up * jnp.exp(delta_up_per_config)
        if has_dn:
            # DN block: only r_dn changes.
            _delta_dn_batch = measured_stream_batch(
                _delta_J_DTN_dn_moved_probe,
                (dtn, old_r_up_carts, old_r_dn_carts),
                (jax.ShapeDtypeStruct((num_dn, 3), new_r_dn_shifted.dtype),),
                new_r_dn_shifted.shape[0],
                JASTROW_DTN_ECP_GRID_BUDGET_BYTES,
            )
            delta_dn_per_config = jax.lax.map(
                lambda cfg: _compute_delta_J_DTN_per_config_dn_moved(dtn, old_r_up_carts, old_r_dn_carts, cfg),
                new_r_dn_shifted,
                batch_size=_delta_dn_batch,
            )
            J_dn = J_dn * jnp.exp(delta_dn_per_config)

    return jnp.concatenate([J_up, J_dn])


def _compute_ratio_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    old_r_up_carts: npt.NDArray[np.float64],
    old_r_dn_carts: npt.NDArray[np.float64],
    new_r_up_carts_arr: npt.NDArray[np.float64],
    new_r_dn_carts_arr: npt.NDArray[np.float64],
) -> npt.NDArray:
    """Independent full-recompute reference for the rank-1 ratio path.

    Both J(R) and J(R') are accumulated in the ``jastrow_ratio`` zone before
    they are subtracted. Accumulating them in the mixed-mode fp32
    ``jastrow_eval`` zone and promoting only after the subtraction loses
    ~eps32 * |J| of the log-ratio to cancellation (the same defect the rank-1
    J1 path avoids with ``_compute_Jastrow_one_body_electron_contrib_ratio``).
    """
    dtype_np = get_dtype_np("jastrow_ratio")
    jastrow_old = _compute_Jastrow_part_debug(jastrow_data, old_r_up_carts, old_r_dn_carts, zone="jastrow_ratio")
    return np.array(
        [
            np.exp(
                _compute_Jastrow_part_debug(jastrow_data, new_r_up_carts, new_r_dn_carts, zone="jastrow_ratio") - jastrow_old
            )
            for new_r_up_carts, new_r_dn_carts in zip(new_r_up_carts_arr, new_r_dn_carts_arr, strict=True)
        ],
        dtype=dtype_np,
    )


@jit
def compute_grads_and_laplacian_Jastrow_part(
    jastrow_data: Jastrow_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    r"""Per-electron gradients and Laplacians of the full Jastrow :math:`J`.

    Analytic paths are used for J1/J2/J3 when available; the NN three-body
    term (if present) is handled via autodiff. Values are returned per electron
    (not summed) to match downstream kinetic-energy estimators.

    Args:
        jastrow_data: Active Jastrow components.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``
            and Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up = jnp.asarray(r_up_carts, dtype=dtype_jnp)
    r_dn = jnp.asarray(r_dn_carts, dtype=dtype_jnp)

    grad_J_up = jnp.zeros_like(r_up)
    grad_J_dn = jnp.zeros_like(r_dn)
    lap_J_up = jnp.zeros((r_up.shape[0],), dtype=dtype_jnp)
    lap_J_dn = jnp.zeros((r_dn.shape[0],), dtype=dtype_jnp)

    # one-body (analytic)
    if jastrow_data.jastrow_one_body_data is not None:
        grad_J1_up, grad_J1_dn, lap_J1_up, lap_J1_dn = compute_grads_and_laplacian_Jastrow_one_body(
            jastrow_data.jastrow_one_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J1_up
        grad_J_dn = grad_J_dn + grad_J1_dn
        lap_J_up = lap_J_up + lap_J1_up
        lap_J_dn = lap_J_dn + lap_J1_dn

    # two-body (analytic)
    if jastrow_data.jastrow_two_body_data is not None:
        grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn = compute_grads_and_laplacian_Jastrow_two_body(
            jastrow_data.jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J2_up
        grad_J_dn = grad_J_dn + grad_J2_dn
        lap_J_up = lap_J_up + lap_J2_up
        lap_J_dn = lap_J_dn + lap_J2_dn

    # three-body (analytic)
    if jastrow_data.jastrow_three_body_data is not None:
        grad_J3_up, grad_J3_dn, lap_J3_up, lap_J3_dn = compute_grads_and_laplacian_Jastrow_three_body(
            jastrow_data.jastrow_three_body_data,
            r_up_carts,
            r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J3_up
        grad_J_dn = grad_J_dn + grad_J3_dn
        lap_J_up = lap_J_up + lap_J3_up
        lap_J_dn = lap_J_dn + lap_J3_dn

    # DTN polynomial (analytic -- production path).
    # The autodiff reference ``_compute_grads_and_laplacian_Jastrow_DTN_auto``
    # uses ``jax.hessian`` and builds an O(N_e^4 * N_atom) tape, OOMing when
    # ``grad(compute_local_energy)`` differentiates through the kinetic energy
    # (3rd-order AD on top). ``compute_grads_and_laplacian_Jastrow_DTN``
    # reduces that to O(N_e^2 * N_atom) by chain-ruling closed-form polynomial
    # derivatives with autodiff on the scalar distance kernel only.
    if jastrow_data.jastrow_dtn_data is not None:
        grad_dtn_up, grad_dtn_dn, lap_dtn_up, lap_dtn_dn = compute_grads_and_laplacian_Jastrow_DTN(
            jastrow_data.jastrow_dtn_data, r_up_carts, r_dn_carts
        )
        grad_J_up = grad_J_up + grad_dtn_up
        grad_J_dn = grad_J_dn + grad_dtn_dn
        lap_J_up = lap_J_up + lap_dtn_up
        lap_J_dn = lap_J_dn + lap_dtn_dn

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


@jit
def _compute_grads_and_laplacian_Jastrow_part_auto(
    jastrow_data: Jastrow_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_data.

    Args:
        jastrow_data (Jastrow_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J and the sum of laplacians of J at (r_up_carts, r_dn_carts).
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts_jnp = jnp.array(r_up_carts, dtype=dtype_jnp)
    r_dn_carts_jnp = jnp.array(r_dn_carts, dtype=dtype_jnp)

    grad_J_up = jnp.zeros_like(r_up_carts_jnp)
    grad_J_dn = jnp.zeros_like(r_dn_carts_jnp)
    lap_J_up = jnp.zeros((r_up_carts_jnp.shape[0],), dtype=dtype_jnp)
    lap_J_dn = jnp.zeros((r_dn_carts_jnp.shape[0],), dtype=dtype_jnp)

    # one-body
    if jastrow_data.jastrow_one_body_data is not None:
        grad_J1_up = grad(compute_Jastrow_one_body, argnums=1)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        grad_J1_dn = grad(compute_Jastrow_one_body, argnums=2)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )

        hessian_J1_up = hessian(compute_Jastrow_one_body, argnums=1)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        laplacian_J1_up = jnp.einsum("ijij->i", hessian_J1_up)

        hessian_J1_dn = hessian(compute_Jastrow_one_body, argnums=2)(
            jastrow_data.jastrow_one_body_data, r_up_carts_jnp, r_dn_carts_jnp
        )
        laplacian_J1_dn = jnp.einsum("ijij->i", hessian_J1_dn)

        grad_J_up = grad_J_up + grad_J1_up
        grad_J_dn = grad_J_dn + grad_J1_dn
        lap_J_up = lap_J_up + laplacian_J1_up
        lap_J_dn = lap_J_dn + laplacian_J1_dn

    # two-body
    if jastrow_data.jastrow_two_body_data is not None:
        grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn = _compute_grads_and_laplacian_Jastrow_two_body_auto(
            jastrow_data.jastrow_two_body_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        grad_J_up = grad_J_up + grad_J2_up
        grad_J_dn = grad_J_dn + grad_J2_dn
        lap_J_up = lap_J_up + lap_J2_up
        lap_J_dn = lap_J_dn + lap_J2_dn

    # three-body
    if jastrow_data.jastrow_three_body_data is not None:
        grad_J3_up_add, grad_J3_dn_add, lap_J3_up_add, lap_J3_dn_add = _compute_grads_and_laplacian_Jastrow_three_body_auto(
            jastrow_data.jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        grad_J_up = grad_J_up + grad_J3_up_add
        grad_J_dn = grad_J_dn + grad_J3_dn_add
        lap_J_up = lap_J_up + lap_J3_up_add
        lap_J_dn = lap_J_dn + lap_J3_dn_add

    # DTN polynomial (autodiff)
    if jastrow_data.jastrow_dtn_data is not None:
        grad_dtn_up, grad_dtn_dn, lap_dtn_up, lap_dtn_dn = _compute_grads_and_laplacian_Jastrow_DTN_auto(
            jastrow_data.jastrow_dtn_data, r_up_carts, r_dn_carts
        )
        grad_J_up = grad_J_up + grad_dtn_up
        grad_J_dn = grad_J_dn + grad_dtn_dn
        lap_J_up = lap_J_up + lap_dtn_up
        lap_J_dn = lap_J_dn + lap_dtn_dn

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


def _compute_grads_and_laplacian_Jastrow_part_debug(
    jastrow_data: Jastrow_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Numerical gradients and Laplacian for the full Jastrow factor.

    Uses central finite differences to approximate gradients and the
    sum of Laplacians of J at (r_up_carts, r_dn_carts).
    """
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    r_up_carts = np.array(r_up_carts, dtype=dtype_np)
    r_dn_carts = np.array(r_dn_carts, dtype=dtype_np)

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J_p_x_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_x_r_up_carts, r_dn_carts=r_dn_carts)
        J_p_y_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_y_r_up_carts, r_dn_carts=r_dn_carts)
        J_p_z_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_p_z_r_up_carts, r_dn_carts=r_dn_carts)

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J_m_x_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_x_r_up_carts, r_dn_carts=r_dn_carts)
        J_m_y_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_y_r_up_carts, r_dn_carts=r_dn_carts)
        J_m_z_up = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=diff_m_z_r_up_carts, r_dn_carts=r_dn_carts)

        grad_x_up.append((J_p_x_up - J_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J_p_y_up - J_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J_p_z_up - J_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J_p_x_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_x_r_dn_carts)
        J_p_y_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_y_r_dn_carts)
        J_p_z_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_p_z_r_dn_carts)

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J_m_x_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_x_r_dn_carts)
        J_m_y_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_y_r_dn_carts)
        J_m_z_dn = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=diff_m_z_r_dn_carts)

        grad_x_dn.append((J_p_x_dn - J_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J_p_y_dn - J_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J_p_z_dn - J_m_z_dn) / (2.0 * diff_h))

    grad_J_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian (4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    diff_h2 = 1.0e-3
    J_ref = compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)

    def _eval_up(r_up: np.ndarray) -> jax.Array:
        return compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up, r_dn_carts=r_dn_carts)

    def _eval_dn(r_dn: np.ndarray) -> jax.Array:
        return compute_Jastrow_part(jastrow_data=jastrow_data, r_up_carts=r_up_carts, r_dn_carts=r_dn)

    def _fd4_second_deriv(
        eval_fn: _FiniteDifferenceEval,
        r_carts: np.ndarray,
        r_i: int,
        dim: int,
        h: float,
        f0: jax.Array,
    ) -> jax.Array:
        r_p1 = r_carts.copy()
        r_p2 = r_carts.copy()
        r_m1 = r_carts.copy()
        r_m2 = r_carts.copy()
        r_p1[r_i][dim] += h
        r_p2[r_i][dim] += 2 * h
        r_m1[r_i][dim] -= h
        r_m2[r_i][dim] -= 2 * h
        return (-eval_fn(r_p2) + 16 * eval_fn(r_p1) - 30 * f0 + 16 * eval_fn(r_m1) - eval_fn(r_m2)) / (12 * h**2)

    lap_J_up = np.zeros(len(r_up_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_up_carts):
        lap_J_up[r_i] = (
            _fd4_second_deriv(_eval_up, r_up_carts, r_i, 0, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 1, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 2, diff_h2, J_ref)
        )

    lap_J_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_dn_carts):
        lap_J_dn[r_i] = (
            _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 0, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 1, diff_h2, J_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 2, diff_h2, J_ref)
        )

    return grad_J_up, grad_J_dn, lap_J_up, lap_J_dn


@jit
def _compute_grads_and_laplacian_Jastrow_two_body_auto(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_two_body_data.

    The method is for computing the gradients and the sum of laplacians of J at (r_up_carts, r_dn_carts)
    with a given Jastrow_two_body_data.

    Args:
        jastrow_two_body_data (Jastrow_two_body_data): an instance of Jastrow_two_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(twobody) and the sum of laplacians of J(twobody) at (r_up_carts, r_dn_carts).
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    r_up_carts = jnp.array(r_up_carts, dtype=dtype_jnp)
    r_dn_carts = jnp.array(r_dn_carts, dtype=dtype_jnp)

    # compute grad
    grad_J2_up = grad(compute_Jastrow_two_body, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    grad_J2_dn = grad(compute_Jastrow_two_body, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)

    # compute laplacians
    hessian_J2_up = hessian(compute_Jastrow_two_body, argnums=1)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    laplacian_J2_up = jnp.einsum("ijij->i", hessian_J2_up)

    hessian_J2_dn = hessian(compute_Jastrow_two_body, argnums=2)(jastrow_two_body_data, r_up_carts, r_dn_carts)
    laplacian_J2_dn = jnp.einsum("ijij->i", hessian_J2_dn)

    return grad_J2_up, grad_J2_dn, laplacian_J2_up, laplacian_J2_dn


def _compute_grads_and_laplacian_Jastrow_two_body_pbc(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Closed-form PBC J2 grad/lap: the molecular dense-pair assembly on the
    Williamson smooth periodic metric.

    Identical uu / dd / ud dense-block structure to the molecular branch of
    :func:`compute_grads_and_laplacian_Jastrow_two_body` (antisymmetric
    ``grad_pair``, symmetric ``lap_pair``, masked diagonal), with the pair
    terms supplied by the closed-form :func:`_j2_pair_terms_pbc` -- which is
    also what the streaming advance uses, keeping fresh and streaming kernels
    bit-consistent under PBC. No autodiff anywhere; validated against
    the retained ``_compute_grads_and_laplacian_Jastrow_two_body_auto`` oracle
    in ``tests/test_jqmc_kernel/test_jastrow.py``. The coincident-pair NaN
    guard for outer autodiff (safe substitution + zeroed rows) lives inside
    ``_j2_pair_terms_pbc``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    a = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)
    j2b_type = jastrow_two_body_data.jastrow_2b_type
    sd = jastrow_two_body_data.structure_data
    cell = sd._cell_jnp.astype(jnp.float64)
    inv_cell = _inv_3x3(cell)

    # Keep the caller-supplied (fp64 walker) coordinates until the pair
    # displacements are formed; ``_j2_pair_terms_pbc`` downcasts each
    # displacement at its use site (Principle 3a/3b, as the streaming advance
    # does). Pre-casting the absolute coordinates to the grad/lap zone would
    # round a small separation at a large common coordinate away.
    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)
    num_up = r_up.shape[0]
    num_dn = r_dn.shape[0]

    grad_up = jnp.zeros(r_up.shape, dtype=dtype_jnp)
    grad_dn = jnp.zeros(r_dn.shape, dtype=dtype_jnp)
    lap_up = jnp.zeros((num_up,), dtype=dtype_jnp)
    lap_dn = jnp.zeros((num_dn,), dtype=dtype_jnp)

    def pair_terms(diff: jax.Array) -> _PairGradLap:
        return _j2_pair_terms_pbc(j2b_type, a, eps, diff, cell, inv_cell)

    # up-up pairs (dense (N_up, N_up); the diagonal is zeroed inside
    # ``_j2_pair_terms_pbc`` via the coincident-pair guard AND masked here).
    if num_up > 1:
        diff_uu = r_up[:, None, :] - r_up[None, :, :]
        mask_uu = 1.0 - jnp.eye(num_up, dtype=dtype_jnp)
        grad_pair_uu, lap_pair_uu = pair_terms(diff_uu)
        grad_up = grad_up + jnp.sum(grad_pair_uu * mask_uu[..., None], axis=1)
        lap_up = lap_up + jnp.sum(lap_pair_uu * mask_uu, axis=1)

    # dn-dn pairs
    if num_dn > 1:
        diff_dd = r_dn[:, None, :] - r_dn[None, :, :]
        mask_dd = 1.0 - jnp.eye(num_dn, dtype=dtype_jnp)
        grad_pair_dd, lap_pair_dd = pair_terms(diff_dd)
        grad_dn = grad_dn + jnp.sum(grad_pair_dd * mask_dd[..., None], axis=1)
        lap_dn = lap_dn + jnp.sum(lap_pair_dd * mask_dd, axis=1)

    # up-dn pairs (all combinations; grad_pair is antisymmetric in the
    # displacement, lap_pair symmetric, so the dn-side contributions are the
    # negated-gradient / same-Laplacian sums over the transposed block).
    if (num_up > 0) and (num_dn > 0):
        diff_ud = r_up[:, None, :] - r_dn[None, :, :]
        grad_pair_ud, lap_pair_ud = pair_terms(diff_ud)
        grad_up = grad_up + jnp.sum(grad_pair_ud, axis=1)
        lap_up = lap_up + jnp.sum(lap_pair_ud, axis=1)
        grad_dn = grad_dn - jnp.sum(grad_pair_ud, axis=0)
        lap_dn = lap_dn + jnp.sum(lap_pair_ud, axis=0)

    return grad_up, grad_dn, lap_up, lap_dn


@jit
def compute_grads_and_laplacian_Jastrow_two_body(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and Laplacians for the two-body Jastrow.

    Supports both ``'pade'`` and ``'exp'`` functional forms, selected via
    ``jastrow_two_body_data.jastrow_2b_type``. Returns per-electron quantities (not summed).

    Args:
        jastrow_two_body_data: Two-body Jastrow parameter container.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    # PBC dispatch: the analytic kernel below assumes the Cartesian e-e
    # distance ``jnp.linalg.norm(r_i - r_j)``. For periodic systems the same
    # dense pair assembly runs with the CLOSED-FORM smooth-periodic pair terms
    # ``_j2_pair_terms_pbc`` -- no autodiff (production must not trace
    # generic AD).
    # ``pbc_flag`` is ``pytree_node=False`` so this resolves at trace time and
    # the molecular HLO is unaffected.
    if jastrow_two_body_data.structure_data.pbc_flag:
        return _compute_grads_and_laplacian_Jastrow_two_body_pbc(
            jastrow_two_body_data,
            r_up_carts,
            r_dn_carts,
        )

    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    a = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)

    # Keep the caller-supplied (fp64 walker) coordinates until the pair
    # displacements are formed, then downcast each displacement to the
    # grad/lap zone (Principle 3a/3b, as ``_j2_pair_terms`` does for the
    # streaming advance). Pre-casting the absolute coordinates would round a
    # small separation at a large common coordinate away.
    r_up = jnp.asarray(r_up_carts)
    r_dn = jnp.asarray(r_dn_carts)

    num_up = r_up.shape[0]
    num_dn = r_dn.shape[0]

    grad_up = jnp.zeros(r_up.shape, dtype=dtype_jnp)
    grad_dn = jnp.zeros(r_dn.shape, dtype=dtype_jnp)
    lap_up = jnp.zeros((num_up,), dtype=dtype_jnp)
    lap_dn = jnp.zeros((num_dn,), dtype=dtype_jnp)

    j2b_type = jastrow_two_body_data.jastrow_2b_type

    if j2b_type == "pade":
        # f(r) = r / (2*(1 + a*r))
        # f'(r) = 1 / (2*(1+a*r)^2)
        # f''(r) = - a / ((1+a*r)^3)
        def pair_terms(diff: jax.Array) -> _PairGradLap:
            r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            r = jnp.maximum(r, eps)
            denom = 1.0 + a * r
            f_prime = 0.5 / (denom * denom)
            grad_coeff = f_prime / r
            lap = -a / (denom * denom * denom) + (2.0 * f_prime) / r
            return grad_coeff[..., None] * diff, lap

    elif j2b_type == "exp":
        # f(r) = 1/(2a) * (1 - exp(-a*r))
        # f'(r) = (1/2) * exp(-a*r)
        # f''(r) = -(a/2) * exp(-a*r)
        def pair_terms(diff: jax.Array) -> _PairGradLap:
            r = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            r = jnp.maximum(r, eps)
            exp_term = jnp.exp(-a * r)
            f_prime = 0.5 * exp_term
            grad_coeff = f_prime / r
            lap = -(a / 2.0) * exp_term + (2.0 * f_prime) / r
            return grad_coeff[..., None] * diff, lap

    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")

    # up-up pairs: dense (N_up, N_up) instead of triu + scatter-add.
    #
    # Background: a ``triu_indices(N, k=1)`` + ``.at[idx_i].add(...)``
    # path emits four ``scatter-add`` ops per spin (grad_i/grad_j/lap_i/lap_j).
    # Because ``idx_i`` repeats values, XLA must respect sequential semantics and
    # lowers each scatter to a ``while_loop`` of ``trip_count = N*(N-1)/2``
    # (~= 4 small kernel launches per iteration), which dominates host launch
    # overhead in ``compute_kinetic_energy_all_elements_fast_update`` (8 such
    # while-loops total -> ~16k launches per call for N=32). Replacing with a
    # dense (N,N) reduction collapses the work into a handful of fused
    # elementwise + reduction kernels with no scatter.
    #
    # Correctness: ``grad_pair[i,j] = grad_coeff(|r_i-r_j|) * (r_i-r_j)`` is
    # antisymmetric and ``lap_pair[i,j]`` is symmetric in (i,j), so summing over
    # ``j`` reproduces the (i,j) and (j,i) contributions of a
    # triu-loop. The diagonal i==j is masked out (``r=0`` is clamped to ``eps``
    # by ``pair_terms`` and would otherwise add a spurious ~1/eps^2 term to the
    # Laplacian). The grad diagonal is mathematically zero (grad_pair prop diff)
    # but is masked too to avoid 0*finite issues if ``eps`` is very small.
    # NaN-safety note: the diagonal i==j has ``diff = 0``. Even though we mask
    # the diagonal out of the sum, ``pair_terms`` evaluates ``r = sqrt(sum diff^2)``
    # whose first derivative is ``1/(2*r)`` and is ``inf`` at ``r=0``. Under
    # higher-order AD (kinetic energy gradients used for forces), this leaks
    # ``0 * inf = NaN`` through the chain rule even though ``maximum(r, eps)``
    # clamps the value. Replace the diagonal of ``diff`` with a safe nonzero
    # vector ``(1, 0, 0)`` before evaluating ``pair_terms``; the mask still
    # zeroes that entry in the sum, so the final result is unchanged.
    if num_up > 1:
        diff_uu = (r_up[:, None, :] - r_up[None, :, :]).astype(dtype_jnp)
        eye_uu = jnp.eye(num_up, dtype=dtype_jnp)
        safe_offset_uu = jnp.stack([eye_uu, jnp.zeros_like(eye_uu), jnp.zeros_like(eye_uu)], axis=-1)
        diff_uu_safe = diff_uu + safe_offset_uu
        grad_pair_uu, lap_pair_uu = pair_terms(diff_uu_safe)
        mask_uu = 1.0 - eye_uu
        grad_up = grad_up + jnp.sum(grad_pair_uu * mask_uu[..., None], axis=1)
        lap_up = lap_up + jnp.sum(lap_pair_uu * mask_uu, axis=1)

    # dn-dn pairs: same dense reformulation as up-up above.
    if num_dn > 1:
        diff_dd = (r_dn[:, None, :] - r_dn[None, :, :]).astype(dtype_jnp)
        eye_dd = jnp.eye(num_dn, dtype=dtype_jnp)
        safe_offset_dd = jnp.stack([eye_dd, jnp.zeros_like(eye_dd), jnp.zeros_like(eye_dd)], axis=-1)
        diff_dd_safe = diff_dd + safe_offset_dd
        grad_pair_dd, lap_pair_dd = pair_terms(diff_dd_safe)
        mask_dd = 1.0 - eye_dd
        grad_dn = grad_dn + jnp.sum(grad_pair_dd * mask_dd[..., None], axis=1)
        lap_dn = lap_dn + jnp.sum(lap_pair_dd * mask_dd, axis=1)

    # up-dn pairs (all combinations)
    if (num_up > 0) and (num_dn > 0):
        diff_ud = (r_up[:, None, :] - r_dn[None, :, :]).astype(dtype_jnp)
        grad_pair, lap_pair = pair_terms(diff_ud)
        grad_up = grad_up + jnp.sum(grad_pair, axis=1)
        grad_dn = grad_dn - jnp.sum(grad_pair, axis=0)
        lap_up = lap_up + jnp.sum(lap_pair, axis=1)
        lap_dn = lap_dn + jnp.sum(lap_pair, axis=0)

    return grad_up, grad_dn, lap_up, lap_dn


# ---------------------------------------------------------------------------
# J2 streaming state.
#
# When a single electron k of spin sigma moves, only pair contributions involving
# k change. The state caches per-electron grad/lap and the previous (r_up,
# r_dn) so the advance can compute the per-pair delta for that electron.
# Cost: O(N_e) per advance, vs O(N_e^2) fresh.
# ---------------------------------------------------------------------------


@struct.dataclass
class Jastrow_two_body_streaming_state:
    """Cached J2 grad/lap and electron coordinates consistent with the state."""

    r_up_carts: jax.Array  # (N_up, 3) -- config used for the cached J2 quantities
    r_dn_carts: jax.Array  # (N_dn, 3)
    grad_J2_up: jax.Array  # (N_up, 3)
    grad_J2_dn: jax.Array  # (N_dn, 3)
    lap_J2_up: jax.Array  # (N_up,)
    lap_J2_dn: jax.Array  # (N_dn,)


def _j2_pair_terms(j2b_type: str, a: jax.Array, eps: jax.Array, diff: jax.Array) -> _PairGradLap:
    """Single-pair grad / lap contributions for the two-body Jastrow.

    Mirrors the closures inside ``compute_grads_and_laplacian_Jastrow_two_body``
    so init and advance share the exact arithmetic. ``j2b_type`` is JIT-static
    (Jastrow_two_body_data marks it ``pytree_node=False``).

    Callers may construct ``diff`` in caller-supplied precision (e.g. fp64
    walker coords for ``r - r_new``); cast at the arithmetic use site here so
    pair-term outputs always live in this function's own zone
    (``jastrow_grad_lap``) regardless of input dtype (Principle 3b -- and
    required for fori_loop carry-shape stability under mixed precision,
    where state.r_up_carts is stored in fp64 but state.grad_J2_up lives in
    the grad/lap zone). The cast target is fetched via
    ``get_dtype_jnp("jastrow_grad_lap")`` rather than reading ``a.dtype``,
    so the function declares its own zone explicitly instead of inheriting
    it from a caller-supplied argument.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    d = diff.astype(dtype_jnp)
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    r = jnp.maximum(r, eps)
    if j2b_type == "pade":
        denom = 1.0 + a * r
        f_prime = 0.5 / (denom * denom)
        grad_coeff = f_prime / r
        lap = -a / (denom * denom * denom) + (2.0 * f_prime) / r
    elif j2b_type == "exp":
        exp_term = jnp.exp(-a * r)
        f_prime = 0.5 * exp_term
        grad_coeff = f_prime / r
        lap = -(a / 2.0) * exp_term + (2.0 * f_prime) / r
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")
    return grad_coeff[..., None] * d, lap


def _j2_pair_terms_pbc(
    j2b_type: str,
    a: jax.Array,
    eps: jax.Array,
    diff: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
) -> _PairGradLap:
    """PBC counterpart of ``_j2_pair_terms`` -- CLOSED FORM, no autodiff.

    The molecular helper differentiates ``f(|diff|)`` analytically; under PBC
    the electron-electron distance is the Williamson smooth periodic distance,
    and the per-pair gradient / Laplacian follow by the chain rule on the
    closed-form metric derivatives of
    :func:`_smooth_periodic_distance_grad_lap`::

        grad f(d) = f'(d) * grad d
        lap  f(d) = f''(d) * |grad d|^2 + f'(d) * lap d

    (derivative w.r.t. ``r_i`` equals the derivative w.r.t. ``diff`` since
    ``d diff / d r_i = I``). 1:1 with the Rust reference ``j2_grad_lap`` +
    ``smooth_dist`` (``jqmc/rs/src/jastrow_factor.rs``). Shared by the fresh
    PBC kernel ``_compute_grads_and_laplacian_Jastrow_two_body_pbc`` and the
    streaming advance, keeping the two bit-consistent under PBC. Accepts any
    batched ``diff`` of shape ``(..., 3)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    d_in = diff.astype(dtype_jnp)

    # The self pair (diff == 0) is masked out by the caller, but the OUTER
    # autodiff used by the kinetic-force paths differentiates through these
    # pair terms, and ``sqrt`` at d=0 leaks NaN through that chain rule even
    # when the value is eps-clamped. Substitute a safe nonzero displacement
    # for those rows and zero their outputs; the caller's mask is unchanged
    # so the final accumulation is identical.
    norm = jnp.sqrt(jnp.sum(d_in * d_in, axis=-1))
    is_zero = norm < eps
    safe_vec = jnp.array([1.0, 0.0, 0.0], dtype=dtype_jnp)
    d_safe = jnp.where(is_zero[..., None], safe_vec, d_in)

    d, g_metric, q, lap_d = _smooth_periodic_distance_grad_lap(d_safe, cell, inv_cell, eps)
    if j2b_type == "pade":
        # f(d) = d / (2*(1 + a*d)); f' = 1/(2*(1+a*d)^2); f'' = -a/((1+a*d)^3)
        denom = 1.0 + a * d
        f_prime = 0.5 / (denom * denom)
        f_second = -a / (denom * denom * denom)
    elif j2b_type == "exp":
        # f(d) = (1/(2a))*(1 - exp(-a*d)); f' = exp(-a*d)/2; f'' = -(a/2)*exp(-a*d)
        exp_term = jnp.exp(-a * d)
        f_prime = 0.5 * exp_term
        f_second = -(a / 2.0) * exp_term
    else:
        raise ValueError(f"Unknown jastrow_2b_type: {j2b_type}")
    g = f_prime[..., None] * g_metric
    lap = f_second * q + f_prime * lap_d

    g = jnp.where(is_zero[..., None], jnp.zeros_like(g), g)
    lap = jnp.where(is_zero, jnp.zeros_like(lap), lap)
    return g, lap


@jit
def _init_grads_laplacian_Jastrow_two_body_streaming_state(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_two_body_streaming_state:
    """Build a J2 state at ``(r_up, r_dn)`` via the existing fresh kernel.

    Stores ``r_up_carts`` / ``r_dn_carts`` in caller-supplied precision
    (Principle 3a -- no rebind). Under mixed precision the carry-shape
    must match what ``advance`` writes back via
    ``state.r_*.at[moved_index].set(r_up_carts_new[moved_index])``;
    ``r_up_carts_new`` arrives in fp64 (walker state), so the cached
    coords must be fp64 too. Diffs are downcast to the jastrow_grad_lap
    zone at the arithmetic use site inside ``_j2_pair_terms``
    (Principle 3b).
    """
    g_up, g_dn, l_up, l_dn = compute_grads_and_laplacian_Jastrow_two_body(jastrow_two_body_data, r_up_carts, r_dn_carts)
    return Jastrow_two_body_streaming_state(
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        grad_J2_up=g_up,
        grad_J2_dn=g_dn,
        lap_J2_up=l_up,
        lap_J2_dn=l_dn,
    )


@jit
def _advance_grads_laplacian_Jastrow_two_body_streaming_state(
    jastrow_two_body_data: Jastrow_two_body_data,
    state: Jastrow_two_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_two_body_streaming_state:
    """Advance J2 state after a single-electron move.

    Computes only the pair-contribution deltas that involve the moved
    electron: ``O(N_e)`` operations per call.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    j2b_type = jastrow_two_body_data.jastrow_2b_type
    a = jnp.asarray(jastrow_two_body_data.jastrow_2b_param, dtype=dtype_jnp)
    eps = jnp.asarray(EPS_safe_distance, dtype=dtype_jnp)

    # PBC dispatch: under periodic boundaries the per-pair
    # grad/lap must use the Williamson smooth periodic distance, matching the
    # auto-diff full kernel that ``_init`` uses. ``pbc_flag`` is
    # ``pytree_node=False`` so this resolves at trace time; the molecular HLO
    # is unaffected.
    sd_j2 = jastrow_two_body_data.structure_data
    if sd_j2.pbc_flag:
        cell_j2 = sd_j2._cell_jnp.astype(jnp.float64)
        inv_cell_j2 = _inv_3x3(cell_j2)

        def _pair(diff: jax.Array) -> _PairGradLap:
            return _j2_pair_terms_pbc(j2b_type, a, eps, diff, cell_j2, inv_cell_j2)
    else:

        def _pair(diff: jax.Array) -> _PairGradLap:
            return _j2_pair_terms(j2b_type, a, eps, diff)

    num_up = state.r_up_carts.shape[0]
    num_dn = state.r_dn_carts.shape[0]

    def _branch_up(_: None) -> Jastrow_two_body_streaming_state:
        r_old = state.r_up_carts[moved_index]
        r_new = r_up_carts_new[moved_index]

        # --- Same-spin (up-up) pairs (k, i) for i != k --------------------
        # Old & new diffs both place 0 at i=k (state.r_up_carts[k]=r_old vs r_old,
        # r_up_carts_new[k]=r_new vs r_new), so masking is implicit at i=k for new
        # but not for old. Mask out the i=k row explicitly to avoid contaminating k
        # via the eps-clamped self-pair value.
        diff_old_uu = r_old - state.r_up_carts  # (N_up, 3); k-th = 0
        diff_new_uu = r_new - r_up_carts_new  # (N_up, 3); k-th = 0
        grad_old_uu, lap_old_uu = _pair(diff_old_uu)
        grad_new_uu, lap_new_uu = _pair(diff_new_uu)
        delta_grad_uu = grad_new_uu - grad_old_uu  # (N_up, 3)
        delta_lap_uu = lap_new_uu - lap_old_uu  # (N_up,)
        mask_uu = (jnp.arange(num_up) != moved_index).astype(dtype_jnp)
        delta_grad_uu = delta_grad_uu * mask_uu[:, None]
        delta_lap_uu = delta_lap_uu * mask_uu

        # i != k: grad_up[i] -= delta_grad_uu[i], lap_up[i] += delta_lap_uu[i]
        new_grad_up = state.grad_J2_up - delta_grad_uu
        new_lap_up = state.lap_J2_up + delta_lap_uu
        # k:   grad_up[k] += sum delta_grad_uu, lap_up[k] += sum delta_lap_uu
        new_grad_up = new_grad_up.at[moved_index].add(jnp.sum(delta_grad_uu, axis=0))
        new_lap_up = new_lap_up.at[moved_index].add(jnp.sum(delta_lap_uu, axis=0))

        # --- Cross-spin (up-dn) pairs (k, j) for all j -------------------
        diff_old_ud = r_old[None, :] - state.r_dn_carts  # (N_dn, 3)
        diff_new_ud = r_new[None, :] - state.r_dn_carts  # (N_dn, 3) (r_dn unchanged)
        grad_old_ud, lap_old_ud = _pair(diff_old_ud)
        grad_new_ud, lap_new_ud = _pair(diff_new_ud)
        delta_grad_ud = grad_new_ud - grad_old_ud
        delta_lap_ud = lap_new_ud - lap_old_ud

        new_grad_up = new_grad_up.at[moved_index].add(jnp.sum(delta_grad_ud, axis=0))
        new_lap_up = new_lap_up.at[moved_index].add(jnp.sum(delta_lap_ud, axis=0))
        new_grad_dn = state.grad_J2_dn - delta_grad_ud
        new_lap_dn = state.lap_J2_dn + delta_lap_ud

        new_r_up = state.r_up_carts.at[moved_index].set(r_new)
        return state.replace(
            r_up_carts=new_r_up,
            grad_J2_up=new_grad_up,
            grad_J2_dn=new_grad_dn,
            lap_J2_up=new_lap_up,
            lap_J2_dn=new_lap_dn,
        )

    def _branch_dn(_: None) -> Jastrow_two_body_streaming_state:
        r_old = state.r_dn_carts[moved_index]
        r_new = r_dn_carts_new[moved_index]

        # --- Same-spin (dn-dn) -------------------------------------------
        diff_old_dd = r_old - state.r_dn_carts
        diff_new_dd = r_new - r_dn_carts_new
        grad_old_dd, lap_old_dd = _pair(diff_old_dd)
        grad_new_dd, lap_new_dd = _pair(diff_new_dd)
        delta_grad_dd = grad_new_dd - grad_old_dd
        delta_lap_dd = lap_new_dd - lap_old_dd
        mask_dd = (jnp.arange(num_dn) != moved_index).astype(dtype_jnp)
        delta_grad_dd = delta_grad_dd * mask_dd[:, None]
        delta_lap_dd = delta_lap_dd * mask_dd

        new_grad_dn = state.grad_J2_dn - delta_grad_dd
        new_lap_dn = state.lap_J2_dn + delta_lap_dd
        new_grad_dn = new_grad_dn.at[moved_index].add(jnp.sum(delta_grad_dd, axis=0))
        new_lap_dn = new_lap_dn.at[moved_index].add(jnp.sum(delta_lap_dd, axis=0))

        # --- Cross-spin (up-dn): grad_up[i] receives +grad_pair(r_up[i] - r_dn[k])
        # so for dn-k moving, the deltas flip signs vs the up branch:
        #   diff = r_up[i] - r_dn_*  ->  diff_new for r_dn[k]=r_new is r_up[i] - r_new
        diff_old_du = state.r_up_carts - r_old[None, :]  # (N_up, 3)
        diff_new_du = state.r_up_carts - r_new[None, :]  # (N_up, 3)
        grad_old_du, lap_old_du = _pair(diff_old_du)
        grad_new_du, lap_new_du = _pair(diff_new_du)
        delta_grad_du = grad_new_du - grad_old_du
        delta_lap_du = lap_new_du - lap_old_du

        # grad_up[i] += delta_grad_du[i]  (sign +)
        new_grad_up = state.grad_J2_up + delta_grad_du
        new_lap_up = state.lap_J2_up + delta_lap_du
        # grad_dn[k] -= sum_i delta_grad_du[i]  (sign - accumulated at k)
        new_grad_dn = new_grad_dn.at[moved_index].add(-jnp.sum(delta_grad_du, axis=0))
        new_lap_dn = new_lap_dn.at[moved_index].add(jnp.sum(delta_lap_du, axis=0))

        new_r_dn = state.r_dn_carts.at[moved_index].set(r_new)
        return state.replace(
            r_dn_carts=new_r_dn,
            grad_J2_up=new_grad_up,
            grad_J2_dn=new_grad_dn,
            lap_J2_up=new_lap_up,
            lap_J2_dn=new_lap_dn,
        )

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


def _compute_grads_and_laplacian_Jastrow_two_body_debug(
    jastrow_two_body_data: Jastrow_two_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """See _api method."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J2_p_x_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_y_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_p_z_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J2_m_x_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_y_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J2_m_z_up = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J2_p_x_up - J2_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J2_p_y_up - J2_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J2_p_z_up - J2_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J2_p_x_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J2_p_y_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J2_p_z_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J2_m_x_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J2_m_y_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J2_m_z_dn = compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J2_p_x_dn - J2_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J2_p_y_dn - J2_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J2_p_z_dn - J2_m_z_dn) / (2.0 * diff_h))

    grad_J2_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J2_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian (4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    diff_h2 = 1.0e-3

    J2_ref = compute_Jastrow_two_body(
        jastrow_two_body_data=jastrow_two_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    def _eval_up(r_up: np.ndarray) -> jax.Array:
        return compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn_carts,
        )

    def _eval_dn(r_dn: np.ndarray) -> jax.Array:
        return compute_Jastrow_two_body(
            jastrow_two_body_data=jastrow_two_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn,
        )

    def _fd4_second_deriv(
        eval_fn: _FiniteDifferenceEval,
        r_carts: np.ndarray,
        r_i: int,
        dim: int,
        h: float,
        f0: jax.Array,
    ) -> jax.Array:
        r_p1 = r_carts.copy()
        r_p2 = r_carts.copy()
        r_m1 = r_carts.copy()
        r_m2 = r_carts.copy()
        r_p1[r_i][dim] += h
        r_p2[r_i][dim] += 2 * h
        r_m1[r_i][dim] -= h
        r_m2[r_i][dim] -= 2 * h
        return (-eval_fn(r_p2) + 16 * eval_fn(r_p1) - 30 * f0 + 16 * eval_fn(r_m1) - eval_fn(r_m2)) / (12 * h**2)

    lap_J2_up = np.zeros(len(r_up_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_up_carts):
        lap_J2_up[r_i] = (
            _fd4_second_deriv(_eval_up, r_up_carts, r_i, 0, diff_h2, J2_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 1, diff_h2, J2_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 2, diff_h2, J2_ref)
        )

    lap_J2_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_dn_carts):
        lap_J2_dn[r_i] = (
            _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 0, diff_h2, J2_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 1, diff_h2, J2_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 2, diff_h2, J2_ref)
        )

    return grad_J2_up, grad_J2_dn, lap_J2_up, lap_J2_dn


@jit
def _compute_grads_and_laplacian_Jastrow_three_body_auto(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Function for computing grads and laplacians with a given Jastrow_three_body_data.

    The method is for computing the gradients and the sum of laplacians of J3 at (r_up_carts, r_dn_carts)
    with a given Jastrow_three_body_data.

    Args:
        jastrow_three_body_data (Jastrow_three_body_data): an instance of Jastrow_three_body_data class
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        the gradients(x,y,z) of J(threebody) and the sum of laplacians of J(threebody) at (r_up_carts, r_dn_carts).
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). Cast to
    # the jastrow_grad_lap zone at the use site (Principle 3b) before passing as
    # the differentiation operand to grad/hessian.
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")

    # compute grad
    grad_J3_up = grad(compute_Jastrow_three_body, argnums=1)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )

    grad_J3_dn = grad(compute_Jastrow_three_body, argnums=2)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )

    # compute laplacians
    hessian_J3_up = hessian(compute_Jastrow_three_body, argnums=1)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )
    laplacian_J3_up = jnp.einsum("ijij->i", hessian_J3_up)

    hessian_J3_dn = hessian(compute_Jastrow_three_body, argnums=2)(
        jastrow_three_body_data, r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)
    )
    laplacian_J3_dn = jnp.einsum("ijij->i", hessian_J3_dn)

    return grad_J3_up, grad_J3_dn, laplacian_J3_up, laplacian_J3_dn


@jit
def compute_grads_and_laplacian_Jastrow_three_body(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Analytic gradients and Laplacians for the three-body Jastrow.

    The functional form is unchanged; this routine leverages analytic AO/MO
    gradients and Laplacians. Per-electron derivatives are returned (not
    summed), matching kinetic-energy estimator expectations.

    Args:
        jastrow_three_body_data: Three-body Jastrow parameters and orbitals.
        r_up_carts: Spin-up electron coordinates with shape ``(N_up, 3)``.
        r_dn_carts: Spin-down electron coordinates with shape ``(N_dn, 3)``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            Gradients for up/down electrons with shapes ``(N_up, 3)`` and ``(N_dn, 3)``,
            Laplacians for up/down electrons with shapes ``(N_up,)`` and ``(N_dn,)``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    orb_data = jastrow_three_body_data.orb_data

    if isinstance(orb_data, MOs_data):
        compute_orb_vgl = compute_MOs_value_grad_lap
    elif isinstance(orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
        compute_orb_vgl = compute_AOs_value_grad_lap
    else:
        raise NotImplementedError

    # r_*_carts forwarded unchanged to ``compute_orb_vgl``; do not pre-cast
    # (the AO/MO kernels reconstruct ``r - R`` in float64 internally). Single
    # fused dispatch shares the heavy block (exp / poly / S_l_m) across
    # val/grad/lap.
    aos_up, grad_up_x, grad_up_y, grad_up_z, lap_up = map(jnp.real, compute_orb_vgl(orb_data, r_up_carts))
    aos_dn, grad_dn_x, grad_dn_y, grad_dn_z, lap_dn = map(jnp.real, compute_orb_vgl(orb_data, r_dn_carts))
    aos_up = jnp.asarray(aos_up, dtype=dtype_jnp)  # (n_orb, n_up)
    aos_dn = jnp.asarray(aos_dn, dtype=dtype_jnp)  # (n_orb, n_dn)

    grad_up = jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=-1)  # (n_orb, n_up, 3)
    grad_dn = jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=-1)  # (n_orb, n_dn, 3)

    lap_up = jnp.asarray(lap_up, dtype=dtype_jnp)  # (n_orb, n_up)
    lap_dn = jnp.asarray(lap_dn, dtype=dtype_jnp)  # (n_orb, n_dn)

    j1_vec = jastrow_three_body_data._j_matrix_jnp[:, -1].astype(dtype_jnp)  # (n_orb,)
    j3_mat = jastrow_three_body_data._j_matrix_jnp[:, :-1].astype(dtype_jnp)  # (n_orb, n_orb)

    # The scalar J3 sums ordered electron pairs, with all up electrons before
    # all down electrons. Exclusive prefix/suffix sums reproduce its two
    # orientations without dense electron-by-electron triangular masks. Keep
    # J and J.T distinct: accepted roundoff asymmetry and derivatives with
    # respect to individual coupling entries must retain their original value.
    aos = jnp.concatenate([aos_up, aos_dn], axis=1)
    zero = jnp.zeros_like(aos[:, :1])
    before = jnp.concatenate([zero, jnp.cumsum(aos, axis=1)[:, :-1]], axis=1)
    after = jnp.concatenate([jnp.cumsum(aos[:, ::-1], axis=1)[:, :-1][:, ::-1], zero], axis=1)
    coefficients = j1_vec[:, None] + j3_mat @ after + j3_mat.T @ before
    g_up, g_dn = jnp.split(coefficients, [aos_up.shape[1]], axis=1)

    grad_J3_up = jnp.einsum("on,onj->nj", g_up, grad_up)
    grad_J3_dn = jnp.einsum("on,onj->nj", g_dn, grad_dn)

    lap_up_contrib = jnp.einsum("on,on->n", g_up, lap_up)
    lap_dn_contrib = jnp.einsum("on,on->n", g_dn, lap_dn)

    return grad_J3_up, grad_J3_dn, lap_up_contrib, lap_dn_contrib


# ---------------------------------------------------------------------------
# J3 streaming state (single-electron rank-1 advance for projection loops)
# ---------------------------------------------------------------------------
#
# The functions below maintain a per-walker auxiliary state that lets us
# advance the per-electron J3 gradients/Laplacians by O(n_ao^2 + n_ao*N_e)
# per single-electron move, instead of recomputing them from scratch at
# O(n_ao^2 * N_e + n_ao * N_e^2). Used by the GFMC projection inner loop
# (jqmc_gfmc.py:_body_fun_n_streaming).
#
# Lifetime: the state is freshly initialized at each branching boundary
# (when _projection_n is re-entered) and advanced for at most
# `num_mcmc_per_measurement` steps inside the fori_loop, mirroring the
# Sherman-Morrison `A_old_inv`. No persistence across branchings.


@struct.dataclass
class Jastrow_three_body_streaming_state:
    """Auxiliary state for streaming J3 grad/Laplacian updates.

    All fields are evaluated at the current ``(r_up_carts, r_dn_carts)``.
    Advancing the state via :func:`_advance_grads_laplacian_Jastrow_three_body_streaming_state`
    after a single-electron move keeps every field consistent with the new
    configuration, with cost ``O(n_ao^2 + n_ao * N_e)`` per step.

    Fields (shapes use ``n_orb`` for the orbital dimension; for MO-based
    three-body the same ``n_orb`` is used, since orbitals are evaluated by
    ``compute_MOs`` to dimension ``orb_data._num_orb``):

    - ``aos_up`` / ``aos_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)`` orbital values.
    - ``grad_aos_up`` / ``grad_aos_dn``: ``(n_orb, N_up, 3)`` / ``(n_orb, N_dn, 3)``.
    - ``lap_aos_up`` / ``lap_aos_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)``.
    - ``j3_mat_aos_up`` / ``j3_mat_aos_dn``: ``j3_mat @ aos_*`` (shapes match aos_*).
    - ``j3_mat_T_aos_up`` / ``j3_mat_T_aos_dn``: ``j3_mat.T @ aos_*``.
    - ``j3_mat_aos_dn_rowsum`` / ``j3_mat_T_aos_up_rowsum``: ``(n_orb,)``
      row-sums ``sum(j3_mat_aos_dn, axis=1)`` / ``sum(j3_mat_T_aos_up, axis=1)``.
      Equivalent to ``j3_mat @ sum(aos_dn, axis=1)`` and
      ``sum(aos_up, axis=1) @ j3_mat`` and consumed by
      :func:`_compute_ratio_Jastrow_part_rank1_update` as the
      ``dn_cross_vec`` / ``up_cross_vec``. Caching avoids the
      ``(W, n_orb, N_e)`` HBM-bound reduction every ratio call.
    - ``g_up`` / ``g_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)`` ``dJ/dA`` per electron.
    - ``grad_J3_up`` / ``grad_J3_dn``: ``(N_up, 3)`` / ``(N_dn, 3)`` per-electron grad.
    - ``lap_J3_up`` / ``lap_J3_dn``: ``(N_up,)`` / ``(N_dn,)`` per-electron lap.
    """

    aos_up: jax.Array = struct.field(pytree_node=True)
    aos_dn: jax.Array = struct.field(pytree_node=True)
    grad_aos_up: jax.Array = struct.field(pytree_node=True)
    grad_aos_dn: jax.Array = struct.field(pytree_node=True)
    lap_aos_up: jax.Array = struct.field(pytree_node=True)
    lap_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_dn_rowsum: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_up_rowsum: jax.Array = struct.field(pytree_node=True)
    g_up: jax.Array = struct.field(pytree_node=True)
    g_dn: jax.Array = struct.field(pytree_node=True)
    grad_J3_up: jax.Array = struct.field(pytree_node=True)
    grad_J3_dn: jax.Array = struct.field(pytree_node=True)
    lap_J3_up: jax.Array = struct.field(pytree_node=True)
    lap_J3_dn: jax.Array = struct.field(pytree_node=True)


def _three_body_orb_apis(jastrow_three_body_data: Jastrow_three_body_data) -> _OrbitalApis:
    """Pick the correct orbital evaluation backends (AO or MO).

    Returned as a Python tuple so callers can dispatch statically (the
    orb_data type is JIT-static via the @struct.dataclass). Includes the
    fused ``(val, gx, gy, gz, lap)`` dispatcher used by the streaming advance
    hot path so the heavy block (exp / poly / S_l_m) is shared across
    val/grad/lap.
    """
    orb_data = jastrow_three_body_data.orb_data
    if isinstance(orb_data, MOs_data):
        return (
            compute_MOs,
            compute_MOs_grad,
            compute_MOs_laplacian,
            compute_MOs_value_grad_lap,
        )
    if isinstance(orb_data, (AOs_sphe_gto_data, AOs_cart_gto_data)):
        return (
            compute_AOs,
            compute_AOs_grad,
            compute_AOs_laplacian,
            compute_AOs_value_grad_lap,
        )
    raise NotImplementedError(f"Unsupported orb_data type: {type(orb_data)}")


def _j3_streaming_table_dtype() -> type:
    """Dtype of the ratio-read tables of :class:`Jastrow_three_body_streaming_state`.

    The orbital values ``aos_*``, the ``j3_mat @ aos_*`` tables and their row-sums are
    read by the ``jastrow_ratio`` kernels (``_compute_ratio_Jastrow_part_rank1_update``,
    ``_compute_ratio_Jastrow_part_split_spin``: the GFMC kinetic and ECP meshes) as well
    as by the ``jastrow_grad_lap`` update of ``g``. They are therefore kept in the higher
    precision of the two zones (float64 in both modes), so the ratio never consumes
    tables rounded and rank-one advanced for the grad/lap zone (issue #781); the grad/lap
    arithmetic casts them to its own zone at the use site. In full precision this is the
    ``jastrow_grad_lap`` dtype.
    """
    return jnp.promote_types(get_dtype_jnp("jastrow_grad_lap"), get_dtype_jnp("jastrow_ratio"))


@jit
def _init_grads_laplacian_Jastrow_three_body_streaming_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_three_body_streaming_state:
    """Initialize the J3 streaming state from a configuration ``(r_up, r_dn)``.

    This is a one-shot evaluation equivalent in cost to
    :func:`compute_grads_and_laplacian_Jastrow_three_body`; it additionally
    materializes the auxiliary tables required by the rank-1 advance.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    # Ratio-read tables (aos, W = j3_mat @ aos, row-sums): see _j3_streaming_table_dtype.
    dtype_tab = _j3_streaming_table_dtype()
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, compute_orb_grad, compute_orb_lapl, compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    # AO/MO tables (forward r_*_carts unchanged so the underlying kernels can
    # reconstruct r-R in float64 -- Principle 3b). Single fused dispatch shares
    # the heavy block (exp / poly / S_l_m) across val/grad/lap.
    aos_up, grad_up_x, grad_up_y, grad_up_z, lap_aos_up = map(jnp.real, compute_orb_vgl(orb_data, r_up_carts))
    aos_dn, grad_dn_x, grad_dn_y, grad_dn_z, lap_aos_dn = map(jnp.real, compute_orb_vgl(orb_data, r_dn_carts))
    aos_up = jnp.asarray(aos_up, dtype=dtype_tab)
    aos_dn = jnp.asarray(aos_dn, dtype=dtype_tab)
    grad_aos_up = jnp.asarray(jnp.stack([grad_up_x, grad_up_y, grad_up_z], axis=-1), dtype=dtype_jnp)
    grad_aos_dn = jnp.asarray(jnp.stack([grad_dn_x, grad_dn_y, grad_dn_z], axis=-1), dtype=dtype_jnp)
    lap_aos_up = jnp.asarray(lap_aos_up, dtype=dtype_jnp)
    lap_aos_dn = jnp.asarray(lap_aos_dn, dtype=dtype_jnp)

    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    j1_vec = j_matrix[:, -1]
    j3_mat_tab = jastrow_three_body_data._j_matrix_jnp[:, :-1].astype(dtype_tab)

    num_up = aos_up.shape[1]
    num_dn = aos_dn.shape[1]

    j3_mat_aos_up = j3_mat_tab @ aos_up
    j3_mat_aos_dn = j3_mat_tab @ aos_dn
    # J3 symmetric (sanity_check asserts it): the U tables == the W tables --
    # alias instead of a second matmul.
    j3_mat_T_aos_up = j3_mat_aos_up
    j3_mat_T_aos_dn = j3_mat_aos_dn
    # Row-sums consumed by ``_compute_ratio_Jastrow_part_rank1_update`` as
    # ``dn_cross_vec`` / ``up_cross_vec`` (see field docstring above).
    j3_mat_aos_dn_rowsum = jnp.sum(j3_mat_aos_dn, axis=1)
    j3_mat_T_aos_up_rowsum = jnp.sum(j3_mat_T_aos_up, axis=1)

    upper_up = jnp.triu(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=1)
    lower_up = jnp.tril(jnp.ones((num_up, num_up), dtype=dtype_jnp), k=-1)
    upper_dn = jnp.triu(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=1)
    lower_dn = jnp.tril(jnp.ones((num_dn, num_dn), dtype=dtype_jnp), k=-1)

    # g is grad/lap-zone arithmetic: the tables are cast to that zone at the use site.
    W_up = j3_mat_aos_up.astype(dtype_jnp)
    W_dn = j3_mat_aos_dn.astype(dtype_jnp)
    g_up = j1_vec[:, None] + W_up @ lower_up + W_up @ upper_up + W_dn @ jnp.ones((num_dn, 1), dtype=dtype_jnp)
    g_dn = j1_vec[:, None] + W_dn @ lower_dn + W_dn @ upper_dn + W_up @ jnp.ones((num_up, 1), dtype=dtype_jnp)

    grad_J3_up = jnp.einsum("on,onj->nj", g_up, grad_aos_up)
    grad_J3_dn = jnp.einsum("on,onj->nj", g_dn, grad_aos_dn)
    lap_J3_up = jnp.einsum("on,on->n", g_up, lap_aos_up)
    lap_J3_dn = jnp.einsum("on,on->n", g_dn, lap_aos_dn)

    return Jastrow_three_body_streaming_state(
        aos_up=aos_up,
        aos_dn=aos_dn,
        grad_aos_up=grad_aos_up,
        grad_aos_dn=grad_aos_dn,
        lap_aos_up=lap_aos_up,
        lap_aos_dn=lap_aos_dn,
        j3_mat_aos_up=j3_mat_aos_up,
        j3_mat_aos_dn=j3_mat_aos_dn,
        j3_mat_T_aos_up=j3_mat_T_aos_up,
        j3_mat_T_aos_dn=j3_mat_T_aos_dn,
        j3_mat_aos_dn_rowsum=j3_mat_aos_dn_rowsum,
        j3_mat_T_aos_up_rowsum=j3_mat_T_aos_up_rowsum,
        g_up=g_up,
        g_dn=g_dn,
        grad_J3_up=grad_J3_up,
        grad_J3_dn=grad_J3_dn,
        lap_J3_up=lap_J3_up,
        lap_J3_dn=lap_J3_dn,
    )


@jit
def _advance_grads_laplacian_Jastrow_three_body_streaming_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    state: Jastrow_three_body_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_three_body_streaming_state:
    """Advance the J3 streaming state after a single-electron move.

    The new ``(r_up_carts_new, r_dn_carts_new)`` differ from the configuration
    represented by ``state`` in *exactly one* electron position, identified by
    ``(moved_spin_is_up, moved_index)``. If neither spin actually moved (e.g. a
    no-op step), the state should still be passed through unchanged by the
    caller -- this routine assumes a real one-electron displacement.

    Cost: ``O(n_ao^2 + n_ao * N_e)`` per call, dominated by two ``n_ao``-sized
    matvecs ``j3_mat @ delta_aos`` and one full einsum over ``g``.
    """
    dtype_jnp = get_dtype_jnp("jastrow_grad_lap")
    # Ratio-read tables (aos, W, row-sums) advance in their own dtype
    # (_j3_streaming_table_dtype); g and the grad/lap outputs in jastrow_grad_lap.
    dtype_tab = _j3_streaming_table_dtype()
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, compute_orb_grad, compute_orb_lapl, compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    j3_mat = jastrow_three_body_data._j_matrix_jnp[:, :-1].astype(dtype_tab)

    num_up = state.aos_up.shape[1]
    num_dn = state.aos_dn.shape[1]

    def _branch_up(_: None) -> Jastrow_three_body_streaming_state:
        # Single-point AO eval at the moved electron's new position.
        # NB: forward r_up_carts_new unchanged (Principle 3b -- fp64 r-R
        # reconstruction inside the kernels). Single fused dispatch shares
        # the heavy block (exp / poly / S_l_m) across val/grad/lap.
        r_new = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        ao_v, gx, gy, gz, ao_lap = map(jnp.real, compute_orb_vgl(orb_data, r_new))
        aos_new_col = jnp.asarray(ao_v[:, 0], dtype=dtype_tab)
        grad_aos_new_col = jnp.asarray(jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=-1), dtype=dtype_jnp)
        lap_aos_new_col = jnp.asarray(ao_lap[:, 0], dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_up[:, moved_index]
        d_J = j3_mat @ delta_aos  # (n_orb,)
        d_JT = d_J  # J3 symmetric: d_JT == d_J

        # Update auxiliary matmuls at the moved column (U == W, so one update).
        new_j3_mat_aos_up = state.j3_mat_aos_up.at[:, moved_index].add(d_J)
        new_j3_mat_T_aos_up = new_j3_mat_aos_up
        # j3_mat_aos_dn / j3_mat_T_aos_dn unchanged (depend on aos_dn).

        # g_up update:
        #   term A (j3_mat_aos_up @ lower_up): col j gets +d_J for j < k.
        #   term B (j3_mat_T_aos_up @ upper_up): col j gets +d_JT for j > k.
        #   term C (cross-spin via aos_dn): unchanged.
        #   col k itself: unchanged (strict triangulars set k-th col to 0).
        col_idx_up = jnp.arange(num_up)
        mask_lt = (col_idx_up < moved_index).astype(dtype_jnp)
        mask_gt = (col_idx_up > moved_index).astype(dtype_jnp)
        d_J_gl = d_J.astype(dtype_jnp)  # grad/lap-zone copy for the g update
        new_g_up = state.g_up + d_J_gl[:, None] * mask_lt[None, :] + d_J_gl[:, None] * mask_gt[None, :]

        # g_dn update: term C is (j3_mat.T @ aos_up) @ ones_up, so the change
        # is sum_k Delta(j3_mat.T @ aos_up)[:, k] = d_JT (single column changed).
        # Same vector added to every dn column.
        new_g_dn = state.g_dn + d_J_gl[:, None]

        # Update aos/grad_aos/lap_aos at the moved column.
        new_aos_up = state.aos_up.at[:, moved_index].set(aos_new_col)
        new_grad_aos_up = state.grad_aos_up.at[:, moved_index, :].set(grad_aos_new_col)
        new_lap_aos_up = state.lap_aos_up.at[:, moved_index].set(lap_aos_new_col)

        # Recompute per-electron grad_J3_*, lap_J3_* via einsum on updated
        # tables. Cost: O(n_ao * N_e * 3) -- within target asymptotics.
        grad_J3_up = jnp.einsum("on,onj->nj", new_g_up, new_grad_aos_up)
        grad_J3_dn = jnp.einsum("on,onj->nj", new_g_dn, state.grad_aos_dn)
        lap_J3_up = jnp.einsum("on,on->n", new_g_up, new_lap_aos_up)
        lap_J3_dn = jnp.einsum("on,on->n", new_g_dn, state.lap_aos_dn)

        return state.replace(
            aos_up=new_aos_up,
            grad_aos_up=new_grad_aos_up,
            lap_aos_up=new_lap_aos_up,
            j3_mat_aos_up=new_j3_mat_aos_up,
            j3_mat_T_aos_up=new_j3_mat_T_aos_up,
            # Only j3_mat_T_aos_up changes (column ``moved_index``); the row-sum
            # picks up exactly ``d_JT``. The dn row-sum is unchanged.
            j3_mat_T_aos_up_rowsum=state.j3_mat_T_aos_up_rowsum + d_JT,
            g_up=new_g_up,
            g_dn=new_g_dn,
            grad_J3_up=grad_J3_up,
            grad_J3_dn=grad_J3_dn,
            lap_J3_up=lap_J3_up,
            lap_J3_dn=lap_J3_dn,
        )

    def _branch_dn(_: None) -> Jastrow_three_body_streaming_state:
        # Single fused dispatch (see _branch_up).
        r_new = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        ao_v, gx, gy, gz, ao_lap = map(jnp.real, compute_orb_vgl(orb_data, r_new))
        aos_new_col = jnp.asarray(ao_v[:, 0], dtype=dtype_tab)
        grad_aos_new_col = jnp.asarray(jnp.stack([gx[:, 0], gy[:, 0], gz[:, 0]], axis=-1), dtype=dtype_jnp)
        lap_aos_new_col = jnp.asarray(ao_lap[:, 0], dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_dn[:, moved_index]
        d_J = j3_mat @ delta_aos

        new_j3_mat_aos_dn = state.j3_mat_aos_dn.at[:, moved_index].add(d_J)
        new_j3_mat_T_aos_dn = new_j3_mat_aos_dn  # U == W (symmetric)

        col_idx_dn = jnp.arange(num_dn)
        mask_lt = (col_idx_dn < moved_index).astype(dtype_jnp)
        mask_gt = (col_idx_dn > moved_index).astype(dtype_jnp)
        d_J_gl = d_J.astype(dtype_jnp)  # grad/lap-zone copy for the g update
        new_g_dn = state.g_dn + d_J_gl[:, None] * mask_lt[None, :] + d_J_gl[:, None] * mask_gt[None, :]

        # g_up term C is (j3_mat @ aos_dn) @ ones_dn, change = d_J for every up col.
        new_g_up = state.g_up + d_J_gl[:, None]

        new_aos_dn = state.aos_dn.at[:, moved_index].set(aos_new_col)
        new_grad_aos_dn = state.grad_aos_dn.at[:, moved_index, :].set(grad_aos_new_col)
        new_lap_aos_dn = state.lap_aos_dn.at[:, moved_index].set(lap_aos_new_col)

        grad_J3_up = jnp.einsum("on,onj->nj", new_g_up, state.grad_aos_up)
        grad_J3_dn = jnp.einsum("on,onj->nj", new_g_dn, new_grad_aos_dn)
        lap_J3_up = jnp.einsum("on,on->n", new_g_up, state.lap_aos_up)
        lap_J3_dn = jnp.einsum("on,on->n", new_g_dn, new_lap_aos_dn)

        return state.replace(
            aos_dn=new_aos_dn,
            grad_aos_dn=new_grad_aos_dn,
            lap_aos_dn=new_lap_aos_dn,
            j3_mat_aos_dn=new_j3_mat_aos_dn,
            j3_mat_T_aos_dn=new_j3_mat_T_aos_dn,
            # Only j3_mat_aos_dn changes (column ``moved_index``); the row-sum
            # picks up exactly ``d_J``. The up row-sum is unchanged.
            j3_mat_aos_dn_rowsum=state.j3_mat_aos_dn_rowsum + d_J,
            g_up=new_g_up,
            g_dn=new_g_dn,
            grad_J3_up=grad_J3_up,
            grad_J3_dn=grad_J3_dn,
            lap_J3_up=lap_J3_up,
            lap_J3_dn=lap_J3_dn,
        )

    # Edge case: zero-electron spin sector -- no advance possible, just no-op.
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


# -----------------------------------------------------------------------------
# Slim ratio-only J3 streaming state (8 fields, no grad/lap).
#
# Mirrors the ``Det_ratio_streaming_state`` / ``Det_streaming_state`` split:
# ``_compute_ratio_Jastrow_part_rank1_update`` only reads the 8 fields below
# (aos_*, j3_mat_aos_*, j3_mat_T_aos_*, rowsums), so the MCMC walker update
# path -- which never consumes grad/lap -- can carry just these and skip the
# remaining 10 fields the full state holds for LRDMC kinetic-energy work.
#
# Cuts ~63% of the per-walker state bytes (grad_aos_* alone is 3x larger via
# the trailing xyz axis), removing the bulk of the DtoD plumbing traffic the
# full state would cost in an MCMC ``fori_loop`` carry.
# -----------------------------------------------------------------------------


@struct.dataclass
class Jastrow_DTN_ratio_state:
    """Geometry cache used by the DTN single-electron ratio kernel.

    Electron--nucleus and electron--electron polynomial tables depend only on
    the current walker geometry.  Keeping them in the MCMC carry means a
    proposal evaluates only the moved row; accepted moves replace that row
    (and the symmetric electron--electron column).
    """

    r_eI: jax.Array = struct.field(pytree_node=True)
    cutoff_eI: jax.Array = struct.field(pytree_node=True)
    pow_eI: jax.Array = struct.field(pytree_node=True)
    r_ee: jax.Array = struct.field(pytree_node=True)
    pow_ee: jax.Array = struct.field(pytree_node=True)


def _dtn_distance_rows(
    dtn: Jastrow_DTN_data,
    points_a: jax.Array,
    points_b: jax.Array,
    dtype_jnp: jnpt.DTypeLike,
) -> jax.Array:
    """Pair distances between two point arrays, with DTN's min-image rule."""
    a = points_a.astype(dtype_jnp)
    b = points_b.astype(dtype_jnp)
    diff = a[:, None, :] - b[None, :, :]
    if dtn.structure_data.pbc_flag:
        cell = dtn.structure_data._cell_jnp.astype(jnp.float64)
        inv_cell = _inv_3x3(cell)
        diff = _dtn_minimum_image(diff.astype(jnp.float64), cell, inv_cell, _dtn_minimum_image_plan(dtn)).astype(dtype_jnp)
    d2 = jnp.sum(diff * diff, axis=-1)
    safe_d2 = jnp.where(d2 > 0.0, d2, jnp.ones_like(d2))
    return jnp.where(d2 > 0.0, jnp.sqrt(safe_d2), jnp.zeros_like(d2))


def _dtn_f_pair_grad_lap_atom(
    leg_a: _DtnLeg,
    leg_b: _DtnLeg,
    gamma_I: jax.Array,
    Pz: jax.Array,
    g_ab_a: jax.Array,
    lap_ab_a: jax.Array,
) -> _GradLapResult:
    """Single-ATOM f-channel ``(grad_a, lap_a, grad_b, lap_b)`` for one pair.

    Identical arithmetic to ``_dtn_f_pair_grad_lap_fast`` with the atom axis
    removed (scalars/`(3,)`/`(3, N+1)` instead of `(N_atom, ...)`), so summing
    the per-atom results reproduces the dense atom-summed pair kernel exactly.
    Legs whose cutoff scalars ``g/gp/gpp`` are zeroed (invalid/padded slots)
    contribute EXACTLY 0 to both the a- and b-roles.
    """
    g_x, gp_x, gpp_x, Px, gr_iI, lap_iI = leg_a
    g_y, gp_y, gpp_y, Py, gr_jI, lap_jI = leg_b
    g_ab_b = -g_ab_a
    lap_ab_b = lap_ab_a

    T = jnp.einsum("lmn,al,bm,cn->abc", gamma_I, Px, Py, Pz)  # (3, 3, 3)
    G = T[0, 0, 0]
    G_x, G_y, G_z = T[1, 0, 0], T[0, 1, 0], T[0, 0, 1]
    G_xx, G_yy, G_zz = T[2, 0, 0], T[0, 2, 0], T[0, 0, 2]
    G_xz, G_yz = T[1, 0, 1], T[0, 1, 1]

    gh = g_x * g_y
    F_x = gp_x * g_y * G + gh * G_x
    F_y = g_x * gp_y * G + gh * G_y
    F_z = gh * G_z
    F_xx = gpp_x * g_y * G + 2.0 * gp_x * g_y * G_x + gh * G_xx
    F_yy = g_x * gpp_y * G + 2.0 * g_x * gp_y * G_y + gh * G_yy
    F_zz = gh * G_zz
    F_xz = gp_x * g_y * G_z + gh * G_xz
    F_yz = g_x * gp_y * G_z + gh * G_yz

    dij_ij = jnp.dot(g_ab_a, g_ab_a)
    grad_a = F_x * gr_iI + F_z * g_ab_a
    dii = jnp.dot(gr_iI, gr_iI)
    di_ij = jnp.dot(gr_iI, g_ab_a)
    lap_a = F_xx * dii + 2.0 * F_xz * di_ij + F_zz * dij_ij + F_x * lap_iI + F_z * lap_ab_a

    grad_b = F_y * gr_jI + F_z * g_ab_b
    djj = jnp.dot(gr_jI, gr_jI)
    dj_ij = jnp.dot(gr_jI, g_ab_b)
    lap_b = F_yy * djj + 2.0 * F_yz * dj_ij + F_zz * dij_ij + F_y * lap_jI + F_z * lap_ab_b

    return grad_a, lap_a, grad_b, lap_b


def _dtn_candidate_delta_rows(
    dtn: Jastrow_DTN_data,
    pos: jax.Array,
    all_e: jax.Array,
    num_up: int,
    row_indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Shared (rows, candidates) DTN delta core -- the dense-mesh algebra.

    This is the body of :func:`_compute_delta_J_DTN_mesh_dense` generalised
    over WHICH electrons own the rows: the dense kinetic kernel passes every
    electron (``row_indices = arange(Ne)``) while the candidate-cache advance
    passes a single traced row (the accepted mover). The einsum strings and
    channel algebra are IDENTICAL to an inline dense body, so the dense
    wrapper's numerics do not depend on the factoring.

    Args:
        dtn: ``Jastrow_DTN_data``.
        pos: ``(M, K+1, 3)`` -- per row, column 0 is the row electron's
            UNMOVED position and columns ``1..K`` are its candidate positions
            (caller precision; cast to the jastrow_ratio zone at the distance
            sites, matching the early-cast chain).
        all_e: ``(Ne, 3)`` unmoved ensemble (up-then-dn).
        num_up: Static up-electron count (spin-channel parity).
        row_indices: ``(M,)`` GLOBAL electron index of each row (static
            ``arange`` for the dense mesh, traced ``(1,)`` for the advance's
            mover-row rebuild). Drives the ``j != m`` self mask and the u
            same/opposite-spin channel selection.

    Returns:
        tuple: ``(delta, cand_r_eI)`` where ``delta`` is the ``(M, K)``
        candidate-minus-unmoved log-Jastrow table and ``cand_r_eI`` is the
        ``(M, K, Na)`` candidate--atom distance block (columns ``1..K`` only)
        that the candidate cache stores for its f-channel corrections.
    """
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    sd = dtn.structure_data
    C = dtn.cutoff_smoothness
    n_m = pos.shape[0]
    n_k1 = pos.shape[1]
    n_e = all_e.shape[0]
    atoms = sd._positions_cart_jnp.astype(dtype_jnp)
    n_a = atoms.shape[0]
    species_idx = jnp.asarray(dtn.species_index, dtype=jnp.int32)

    flat = pos.reshape(n_m * n_k1, 3)

    # Shared distance tensors cast at the consumer boundary: moved electron vs
    # atoms (chi + f moved side) and vs the UNMOVED ensemble (u + f e-e side).
    r_eI = _dtn_distance_rows(dtn, flat, atoms, dtype_jnp).reshape(n_m, n_k1, n_a)
    r_ee = _dtn_distance_rows(dtn, flat, all_e, dtype_jnp).reshape(n_m, n_k1, n_e)
    self_mask = (row_indices[:, None] != jnp.arange(n_e)[None, :]).astype(dtype_jnp)  # (m, j)

    val = jnp.zeros((n_m, n_k1), dtype=dtype_jnp)

    if dtn.use_u:
        alpha_full = _dtn_inflate_alpha(dtn, dtype_jnp)
        L_u = _dtn_effective_cutoffs(dtn, dtype_jnp)[0]
        k_u = jnp.arange(dtn.n_order_u + 1, dtype=dtype_jnp)
        cut_u = jnp.maximum(L_u - r_ee, 0.0) ** C  # (m, K+1, j)
        pow_u = r_ee[..., None] ** k_u  # (m, K+1, j, o)
        spin_up = jnp.arange(n_e) < num_up
        same = (row_indices < num_up)[:, None] == spin_up[None, :]  # (m, j) global-index parity
        alpha_mj = jnp.where(same[..., None], alpha_full[0][None, None, :], alpha_full[1][None, None, :])
        val = val + jnp.einsum("mkj,mkjo,mjo,mj->mk", cut_u, pow_u, alpha_mj, self_mask)

    if dtn.use_chi:
        beta_full = _dtn_inflate_beta(dtn, dtype_jnp)
        L_chi_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[1][species_idx]  # (Na,)
        beta_atoms = beta_full[species_idx]  # (Na, o)
        k_chi = jnp.arange(dtn.n_order_chi + 1, dtype=dtype_jnp)
        cut_chi = jnp.maximum(L_chi_atoms[None, None, :] - r_eI, 0.0) ** C  # (m, K+1, I)
        pow_chi = r_eI[..., None] ** k_chi
        val = val + jnp.einsum("mkI,mkIo,Io->mk", cut_chi, pow_chi, beta_atoms)

    if dtn.use_f:
        gamma_full = _dtn_inflate_gamma(dtn, dtype_jnp)
        gamma_atoms = gamma_full[species_idx]  # (Na, l, m', n)
        L_f_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[2][species_idx]
        k_eN = jnp.arange(dtn.n_order_eN + 1, dtype=dtype_jnp)
        k_ee = jnp.arange(dtn.n_order_ee + 1, dtype=dtype_jnp)
        # Moved-side legs over ALL atoms.
        cut_i = jnp.maximum(L_f_atoms[None, None, :] - r_eI, 0.0) ** C  # (m, K+1, I)
        leg_i = cut_i[..., None] * (r_eI[..., None] ** k_eN)  # (m, K+1, I, l)
        # j-side ONCE per walker: dense rows for the unmoved ensemble. Every
        # electron is a lane of every atom, so the membership one-hot is the
        # identity and the strict cutoff alone decides the lane's weight
        # (0 outside) -- identical semantics to the per-config gather,
        # amortised over all configs.
        r_jI = _dtn_distance_rows(dtn, all_e, atoms, dtype_jnp)  # (j, I)
        cut_j = jnp.maximum(L_f_atoms[:, None] - r_jI.T, 0.0) ** C  # (I, j)
        pow_j = r_jI.T[..., None] ** k_eN  # (I, j, m')
        # Atom-side kernel with the j polynomial folded in, then one batched
        # contraction over (I, l) -- the same "no per-atom dispatch" lowering
        # lesson as the per-config einsum.
        g_side = jnp.einsum("Ilmn,Ijm,Ij->Ijln", gamma_atoms, pow_j, cut_j)  # (I, j, l, n)
        t_mid = jnp.einsum("mkIl,Ijln->mkjn", leg_i, g_side)  # (m, K+1, j, n)
        pow_ij = r_ee[..., None] ** k_ee  # (m, K+1, j, n)
        val = val + jnp.einsum("mkjn,mkjn,mj->mk", t_mid, pow_ij, self_mask)

    # (M, K): candidate minus unmoved column, plus the candidate--atom block.
    return val[:, 1:] - val[:, :1], r_eI[:, 1:, :]


def _compute_delta_J_DTN_mesh_dense(
    dtn: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    shifts: jax.Array,
) -> jax.Array:
    """DENSE fused DTN delta for the WHOLE kinetic mesh -- no neighbor list.

    ONE pass computes ``delta J_DTN(m, k)`` for every (moved electron ``m``,
    axial shift ``k``) of the LRDMC kinetic lattice, instead of ``Ne * K``
    independent per-config kernels. Every moved-side tensor is batched over the
    ``(m, K+1)`` candidate lattice (column 0 = the unmoved position, so the old
    values come from the same pass) and the j-side f tensors are built ONCE.

    Channel contracts (the ``(m, k)`` FACTORIZATION is shared by all three):
      * u   : rank-one row of the moved electron vs ALL ``Ne`` electrons;
        same/opposite spin selected by GLOBAL index parity
        ``(j < num_up) == (m < num_up)`` -- no argmax detection.
      * chi : moved electron vs ALL ``Na`` atoms.
      * f   : moved-side legs vs ALL ``Na`` atoms; j-side lanes are ALL ``Ne``
        electrons for every atom, so the per-atom membership one-hot of the
        neighbor-list form IS the identity and disappears -- the ``(I, j)``
        lane weight is just the strict cutoff, and the self pair is excluded
        by the same ``j != m`` mask as u. This is the exact same collapse the
        dense drift/force mesh kernels make
        (``derivatives.dtn.dtn_drift_mesh_dense``).

    Args:
        dtn: ``Jastrow_DTN_data``.
        r_up_carts: ``(N_up, 3)`` unmoved up-electron positions (the mesh
            caller's zone-cast array).
        r_dn_carts: ``(N_dn, 3)`` unmoved dn-electron positions (the mesh
            caller's zone-cast array).
        shifts: ``(K, 3)`` displacement lattice (already rotated, caller
            dtype) -- candidate ``k`` of electron ``m`` is ``r_m + shifts[k]``,
            the SAME arithmetic as the mesh builder's ``eye``-broadcast.

    Returns:
        ``(Ne, K)`` delta matrix, up electrons first -- a row-major reshape
        matches the mesh flattening (up block m-major / k-minor, then dn).
    """
    num_up = r_up_carts.shape[0]
    all_e = jnp.concatenate((r_up_carts, r_dn_carts), axis=0)
    n_e = all_e.shape[0]

    # Candidate lattice (m, K+1, 3): column 0 = unmoved, columns 1..K shifted.
    # ``r_m + shifts[k]`` in caller precision -- bit-identical to the mesh
    # builder's ``r + eye*shift`` construction. The channel algebra lives in
    # ``_dtn_candidate_delta_rows`` (shared with the candidate-cache init and
    # mover-row rebuild); the second output (candidate--atom distances) is
    # dead here and DCE'd by XLA.
    pos = jnp.concatenate((all_e[:, None, :], all_e[:, None, :] + shifts[None, :, :]), axis=1)
    delta, _cand_r_eI = _dtn_candidate_delta_rows(dtn, pos, all_e, num_up, jnp.arange(n_e))
    return delta  # (Ne, K): candidate minus unmoved column


@jit
def _init_jastrow_dtn_ratio_state(
    dtn: Jastrow_DTN_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_DTN_ratio_state:
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    all_e = jnp.concatenate((r_up_carts, r_dn_carts), axis=0)
    atoms = dtn.structure_data._positions_cart_jnp
    r_eI = _dtn_distance_rows(dtn, all_e, atoms, dtype_jnp)
    L_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[2][jnp.asarray(dtn.species_index, dtype=jnp.int32)]
    cutoff_eI = jnp.maximum(L_atoms[None, :] - r_eI, 0.0) ** dtn.cutoff_smoothness
    max_eI_order = max(dtn.n_order_eN, dtn.n_order_chi)
    pow_eI = r_eI[..., None] ** jnp.arange(max_eI_order + 1, dtype=dtype_jnp)
    r_ee = _dtn_distance_rows(dtn, all_e, all_e, dtype_jnp)
    max_ee_order = max(dtn.n_order_ee, dtn.n_order_u)
    pow_ee = r_ee[..., None] ** jnp.arange(max_ee_order + 1, dtype=dtype_jnp)
    return Jastrow_DTN_ratio_state(r_eI=r_eI, cutoff_eI=cutoff_eI, pow_eI=pow_eI, r_ee=r_ee, pow_ee=pow_ee)


def _delta_J_DTN_from_ratio_state(
    dtn: Jastrow_DTN_data,
    state: Jastrow_DTN_ratio_state,
    moved_index: jax.Array,
    r_new: jax.Array,
    all_e: jax.Array,
    num_up: int,
    skip_f: bool = False,
) -> jax.Array:
    """DTN(new)-DTN(old) using cached old geometry and one new row."""
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    atoms = dtn.structure_data._positions_cart_jnp
    species = jnp.asarray(dtn.species_index, dtype=jnp.int32)
    L_f_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[2][species]
    r_eI_new = _dtn_distance_rows(dtn, r_new[None, :], atoms, dtype_jnp)[0]
    cutoff_eI_new = jnp.maximum(L_f_atoms - r_eI_new, 0.0) ** dtn.cutoff_smoothness
    max_eI_order = max(dtn.n_order_eN, dtn.n_order_chi)
    pow_eI_new = r_eI_new[:, None] ** jnp.arange(max_eI_order + 1, dtype=dtype_jnp)

    r_ee_new = _dtn_distance_rows(dtn, r_new[None, :], all_e, dtype_jnp)[0]
    max_ee_order = max(dtn.n_order_ee, dtn.n_order_u)
    pow_ee_new = r_ee_new[:, None] ** jnp.arange(max_ee_order + 1, dtype=dtype_jnp)
    mask = (jnp.arange(all_e.shape[0]) != moved_index).astype(dtype_jnp)
    delta = jnp.zeros((), dtype=dtype_jnp)

    if dtn.use_u:
        alpha = _dtn_inflate_alpha(dtn, dtype_jnp)
        L_u = _dtn_effective_cutoffs(dtn, dtype_jnp)[0]
        cutoff_new = jnp.maximum(L_u - r_ee_new, 0.0) ** dtn.cutoff_smoothness
        cutoff_old = jnp.maximum(L_u - state.r_ee[moved_index], 0.0) ** dtn.cutoff_smoothness
        moved_up = moved_index < num_up
        other_up = jnp.arange(all_e.shape[0]) < num_up
        same_spin = other_up == moved_up
        channel = jnp.where(same_spin, 0, 1)
        u_new = cutoff_new * jnp.sum(alpha[channel] * pow_ee_new[..., : dtn.n_order_u + 1], axis=1)
        u_old = cutoff_old * jnp.sum(alpha[channel] * state.pow_ee[moved_index, ..., : dtn.n_order_u + 1], axis=1)
        delta = delta + jnp.sum(mask * (u_new - u_old))

    if dtn.use_chi:
        beta = _dtn_inflate_beta(dtn, dtype_jnp)
        L_chi_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[1][species]
        chi_cut_new = jnp.maximum(L_chi_atoms - r_eI_new, 0.0) ** dtn.cutoff_smoothness
        chi_cut_old = jnp.maximum(L_chi_atoms - state.r_eI[moved_index], 0.0) ** dtn.cutoff_smoothness
        chi_new = chi_cut_new * jnp.sum(beta[species] * pow_eI_new[..., : dtn.n_order_chi + 1], axis=1)
        chi_old = chi_cut_old * jnp.sum(beta[species] * state.pow_eI[moved_index, ..., : dtn.n_order_chi + 1], axis=1)
        delta = delta + jnp.sum(chi_new - chi_old)

    if dtn.use_f and not skip_f:
        gamma = _dtn_inflate_gamma(dtn, dtype_jnp)[species]
        pow_i_new_f = pow_eI_new[..., : dtn.n_order_eN + 1]
        pow_i_old_f = state.pow_eI[moved_index, ..., : dtn.n_order_eN + 1]
        pow_j_f = state.pow_eI[..., : dtn.n_order_eN + 1]
        pow_ij_new_f = pow_ee_new[..., : dtn.n_order_ee + 1]
        pow_ij_old_f = state.pow_ee[moved_index, ..., : dtn.n_order_ee + 1]
        inner_new = jnp.einsum("Almn,Al,jAm,jn->jA", gamma, pow_i_new_f, pow_j_f, pow_ij_new_f)
        inner_old = jnp.einsum(
            "Almn,Al,jAm,jn->jA",
            gamma,
            pow_i_old_f,
            pow_j_f,
            pow_ij_old_f,
        )
        f_new = cutoff_eI_new[None, :] * state.cutoff_eI * inner_new
        f_old = state.cutoff_eI[moved_index][None, :] * state.cutoff_eI * inner_old
        delta = delta + jnp.sum(mask * jnp.sum(f_new - f_old, axis=1))
    return delta


@jit
def _advance_jastrow_dtn_ratio_state(
    dtn: Jastrow_DTN_data,
    state: Jastrow_DTN_ratio_state,
    moved_index: jax.Array,
    r_new: jax.Array,
    all_e_new: jax.Array,
) -> Jastrow_DTN_ratio_state:
    """Commit one accepted electron move to the cached DTN geometry."""
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    atoms = dtn.structure_data._positions_cart_jnp
    species = jnp.asarray(dtn.species_index, dtype=jnp.int32)
    r_eI = _dtn_distance_rows(dtn, r_new[None, :], atoms, dtype_jnp)[0]
    L_atoms = _dtn_effective_cutoffs(dtn, dtype_jnp)[2][species]
    cutoff_eI = jnp.maximum(L_atoms - r_eI, 0.0) ** dtn.cutoff_smoothness
    max_eI_order = max(dtn.n_order_eN, dtn.n_order_chi)
    pow_eI = r_eI[:, None] ** jnp.arange(max_eI_order + 1, dtype=dtype_jnp)
    r_ee = _dtn_distance_rows(dtn, r_new[None, :], all_e_new, dtype_jnp)[0]
    max_ee_order = max(dtn.n_order_ee, dtn.n_order_u)
    pow_ee = r_ee[:, None] ** jnp.arange(max_ee_order + 1, dtype=dtype_jnp)
    new_r_ee = state.r_ee.at[moved_index, :].set(r_ee).at[:, moved_index].set(r_ee)
    new_pow_ee = state.pow_ee.at[moved_index, :, :].set(pow_ee).at[:, moved_index, :].set(pow_ee)
    return state.replace(
        r_eI=state.r_eI.at[moved_index].set(r_eI),
        cutoff_eI=state.cutoff_eI.at[moved_index].set(cutoff_eI),
        pow_eI=state.pow_eI.at[moved_index].set(pow_eI),
        r_ee=new_r_ee,
        pow_ee=new_pow_ee,
    )


@struct.dataclass
class Jastrow_three_body_ratio_state:
    """Slim ratio-only J3 streaming state.

    Subset of :class:`Jastrow_three_body_streaming_state` containing exactly
    the 8 fields read by :func:`_compute_ratio_Jastrow_part_rank1_update` when
    a J3 cache is supplied. Field names match the full state so the ratio
    kernel consumes either via duck-typing (mirrors the
    ``Det_ratio_streaming_state`` / ``Det_streaming_state`` pattern).

    Fields (shapes use ``n_orb`` for the orbital dimension; for MO-based
    three-body the same ``n_orb`` is used, since orbitals are evaluated by
    ``compute_MOs`` to dimension ``orb_data._num_orb``):

    - ``aos_up`` / ``aos_dn``: ``(n_orb, N_up)`` / ``(n_orb, N_dn)`` orbital values.
    - ``j3_mat_aos_up`` / ``j3_mat_aos_dn``: ``j3_mat @ aos_*`` (shapes match aos_*).
    - ``j3_mat_T_aos_up`` / ``j3_mat_T_aos_dn``: ``j3_mat.T @ aos_*``.
    - ``j3_mat_aos_dn_rowsum`` / ``j3_mat_T_aos_up_rowsum``: ``(n_orb,)`` -- the
      ``dn_cross_vec`` / ``up_cross_vec`` consumed by the ratio kernel.
    """

    aos_up: jax.Array = struct.field(pytree_node=True)
    aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_up: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_dn: jax.Array = struct.field(pytree_node=True)
    j3_mat_aos_dn_rowsum: jax.Array = struct.field(pytree_node=True)
    j3_mat_T_aos_up_rowsum: jax.Array = struct.field(pytree_node=True)


@jit
def _init_jastrow_ratio_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_three_body_ratio_state:
    """Initialize the slim ratio-only J3 streaming state at ``(r_up, r_dn)``.

    Value-only AO evaluation (no grad/lap), four ``(n_orb, n_orb) @ (n_orb, N_e)``
    matmuls, and two ``(n_orb,)`` row-sums. Subsequent advances refresh only
    one column per accepted single-electron move.

    Cost: dominated by 2 ``compute_orb`` calls + 4 matmuls, equivalent to the
    j3_state-less branch of :func:`_compute_ratio_Jastrow_part_rank1_update`.
    """
    # Use the ratio-zone dtype to match what
    # ``_compute_ratio_Jastrow_part_rank1_update`` will cast its inputs to;
    # this avoids a re-cast on first consumption.
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, _compute_orb_grad, _compute_orb_lapl, _compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    # Value-only AO eval. Forward r_*_carts unchanged so the underlying
    # kernels reconstruct r-R in fp64 (Principle 3b).
    aos_up = jnp.asarray(jnp.real(compute_orb(orb_data, r_up_carts)), dtype=dtype_jnp)
    aos_dn = jnp.asarray(jnp.real(compute_orb(orb_data, r_dn_carts)), dtype=dtype_jnp)

    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    j3_mat = j_matrix[:, :-1]

    j3_mat_aos_up = j3_mat @ aos_up
    j3_mat_aos_dn = j3_mat @ aos_dn
    # J3 is symmetric (sanity_check asserts it), so j3_mat.T @ x == j3_mat @ x:
    # the U tables equal the W tables -- alias instead of a second matmul.
    j3_mat_T_aos_up = j3_mat_aos_up
    j3_mat_T_aos_dn = j3_mat_aos_dn
    # Row-sums consumed as ``dn_cross_vec`` / ``up_cross_vec`` by the ratio kernel.
    j3_mat_aos_dn_rowsum = jnp.sum(j3_mat_aos_dn, axis=1)
    j3_mat_T_aos_up_rowsum = jnp.sum(j3_mat_T_aos_up, axis=1)

    return Jastrow_three_body_ratio_state(
        aos_up=aos_up,
        aos_dn=aos_dn,
        j3_mat_aos_up=j3_mat_aos_up,
        j3_mat_aos_dn=j3_mat_aos_dn,
        j3_mat_T_aos_up=j3_mat_T_aos_up,
        j3_mat_T_aos_dn=j3_mat_T_aos_dn,
        j3_mat_aos_dn_rowsum=j3_mat_aos_dn_rowsum,
        j3_mat_T_aos_up_rowsum=j3_mat_T_aos_up_rowsum,
    )


@jit
def _advance_jastrow_ratio_state(
    jastrow_three_body_data: Jastrow_three_body_data,
    state: Jastrow_three_body_ratio_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_three_body_ratio_state:
    """Advance the slim ratio-only J3 streaming state after a single-electron move.

    Mirrors :func:`_advance_grads_laplacian_Jastrow_three_body_streaming_state`
    but with the grad/lap field plumbing removed:

    - Single-point AO eval at the moved electron's new position (value only,
      no grad/lap so the per-call ``compute_orb_vgl`` becomes ``compute_orb``).
    - ``delta_aos = aos_new - state.aos[:, moved_index]``,
      ``d_J = j3_mat @ delta_aos``, ``d_JT = j3_mat.T @ delta_aos``.
    - Refresh the moved column of ``j3_mat_aos_*`` / ``j3_mat_T_aos_*`` via
      ``.at[:, moved_index].add(...)`` and the corresponding row-sum.

    Cost: ``O(n_ao + n_ao^2)`` per call -- one single-electron AO eval plus
    two ``n_ao``-sized matvecs. Strictly less work than the full advance:
    the grad/lap einsums and the ``g_up`` / ``g_dn`` triangular updates are
    not needed for the ratio kernel.
    """
    dtype_jnp = get_dtype_jnp("jastrow_ratio")
    orb_data = jastrow_three_body_data.orb_data
    compute_orb, _compute_orb_grad, _compute_orb_lapl, _compute_orb_vgl = _three_body_orb_apis(jastrow_three_body_data)

    j_matrix = jastrow_three_body_data._j_matrix_jnp.astype(dtype_jnp)
    j3_mat = j_matrix[:, :-1]

    num_up = state.aos_up.shape[1]
    num_dn = state.aos_dn.shape[1]

    def _branch_up(_: None) -> Jastrow_three_body_ratio_state:
        # Single-electron AO eval at the new up position (value only).
        r_new = jnp.expand_dims(r_up_carts_new[moved_index], axis=0)  # (1, 3)
        aos_new_col = jnp.asarray(jnp.real(compute_orb(orb_data, r_new)[:, 0]), dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_up[:, moved_index]
        d_J = j3_mat @ delta_aos  # (n_orb,)
        # J3 symmetric: d_JT == d_J, and the U table stays == the W table, so the
        # second matvec and the second column-add are dropped.
        new_aos_up = state.aos_up.at[:, moved_index].set(aos_new_col)
        new_j3_mat_aos_up = state.j3_mat_aos_up.at[:, moved_index].add(d_J)
        return state.replace(
            aos_up=new_aos_up,
            j3_mat_aos_up=new_j3_mat_aos_up,
            j3_mat_T_aos_up=new_j3_mat_aos_up,  # U == W (symmetric)
            j3_mat_T_aos_up_rowsum=state.j3_mat_T_aos_up_rowsum + d_J,
        )

    def _branch_dn(_: None) -> Jastrow_three_body_ratio_state:
        r_new = jnp.expand_dims(r_dn_carts_new[moved_index], axis=0)
        aos_new_col = jnp.asarray(jnp.real(compute_orb(orb_data, r_new)[:, 0]), dtype=dtype_jnp)

        delta_aos = aos_new_col - state.aos_dn[:, moved_index]
        d_J = j3_mat @ delta_aos
        # J3 symmetric: d_JT == d_J, U table stays == W.
        new_aos_dn = state.aos_dn.at[:, moved_index].set(aos_new_col)
        new_j3_mat_aos_dn = state.j3_mat_aos_dn.at[:, moved_index].add(d_J)
        return state.replace(
            aos_dn=new_aos_dn,
            j3_mat_aos_dn=new_j3_mat_aos_dn,
            j3_mat_T_aos_dn=new_j3_mat_aos_dn,  # U == W (symmetric)
            j3_mat_aos_dn_rowsum=state.j3_mat_aos_dn_rowsum + d_J,
        )

    # Edge case: zero-electron spin sector -- the cond collapses at trace time.
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


# -----------------------------------------------------------------------------
# Jastrow_ratio_state container.
#
# Mirrors the ``Kinetic_streaming_state`` pattern in ``wavefunction.py``: a
# flax dataclass holding per-Jastrow ratio-streaming sub-states so the
# MCMC/GFMC fori_loop carry is a single pytree node regardless of how many
# Jastrow terms can carry ratio caches. J3 stores orbital contractions and
# DTN stores geometry polynomial tables; J1/J2/NN remain stateless. Future
# terms add a field here without touching driver carry signatures.
# -----------------------------------------------------------------------------


@struct.dataclass
class Jastrow_ratio_state:
    """Container for per-Jastrow ratio-streaming sub-states (MCMC carry).

    ``None`` fields contribute no leaves, so the container collapses to an
    empty pytree when no term carries a ratio cache. Callers thread this one
    container through their loop carry for both J3 and DTN.
    """

    j3_state: Jastrow_three_body_ratio_state | None = struct.field(
        pytree_node=True,
        default=None,
    )
    dtn_state: Jastrow_DTN_ratio_state | None = struct.field(pytree_node=True, default=None)


def init_Jastrow_ratio_state(
    jastrow_data: Jastrow_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> Jastrow_ratio_state:
    """Build whichever ratio sub-states this Jastrow's channels own.

    WHICH channels keep a ratio cache is Jastrow-internal knowledge. A caller
    that branches on ``jastrow_three_body_data is not None`` and then on
    ``jastrow_dtn_data is not None`` has to be edited every time a channel grows
    or loses a cache; this function absorbs that policy.

    The dispatch is Python-static -- it reads only ``is None`` on the data
    leaves, never a traced value -- so the returned container has a fixed pytree
    shape for a given ``jastrow_data`` and is safe as a ``fori_loop`` carry.
    Absent channels contribute ``None``, i.e. no leaves at all.

    Args:
        jastrow_data: Active Jastrow components.
        r_up_carts: Spin-up coordinates, shape ``(N_up, 3)``.
        r_dn_carts: Spin-down coordinates, shape ``(N_dn, 3)``.

    Returns:
        Jastrow_ratio_state: One container for every cache-owning channel.
    """
    j3 = jastrow_data.jastrow_three_body_data
    dtn = jastrow_data.jastrow_dtn_data
    return Jastrow_ratio_state(
        j3_state=None if j3 is None else _init_jastrow_ratio_state(j3, r_up_carts, r_dn_carts),
        dtn_state=None if dtn is None else _init_jastrow_dtn_ratio_state(dtn, r_up_carts, r_dn_carts),
    )


def advance_Jastrow_ratio_state(
    jastrow_data: Jastrow_data,
    state: Jastrow_ratio_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
) -> Jastrow_ratio_state:
    """Commit one accepted single-electron move to every cache-owning channel.

    Twin of :func:`init_Jastrow_ratio_state`. Besides the per-channel dispatch,
    this also absorbs the per-channel CALLING CONVENTIONS: J3 advances on the
    ``(spin, local index)`` pair, whereas DTN wants a GLOBAL electron index, the
    moved electron's new position, and the concatenated new configuration.
    Reconstructing those at the call site is exactly the kind of channel-internal
    detail a driver should not hold.

    Args:
        jastrow_data: Active Jastrow components.
        state: The container returned by :func:`init_Jastrow_ratio_state`, valid
            for the PRE-move configuration.
        moved_spin_is_up: Traced bool -- the moved electron's spin.
        moved_index: Traced index of the moved electron WITHIN its spin block.
        r_up_carts_new: Post-move spin-up coordinates, shape ``(N_up, 3)``.
        r_dn_carts_new: Post-move spin-down coordinates, shape ``(N_dn, 3)``.

    Returns:
        Jastrow_ratio_state: The advanced container, same pytree shape.
    """
    j3 = jastrow_data.jastrow_three_body_data
    dtn = jastrow_data.jastrow_dtn_data
    j3_state_new = state.j3_state
    if j3 is not None:
        j3_state_new = _advance_jastrow_ratio_state(
            jastrow_three_body_data=j3,
            state=state.j3_state,
            moved_spin_is_up=moved_spin_is_up,
            moved_index=moved_index,
            r_up_carts_new=r_up_carts_new,
            r_dn_carts_new=r_dn_carts_new,
        )
    dtn_state_new = state.dtn_state
    if dtn is not None:
        num_up = r_up_carts_new.shape[0]
        all_e_new = jnp.concatenate((r_up_carts_new, r_dn_carts_new), axis=0)
        global_moved_index = jnp.where(moved_spin_is_up, moved_index, num_up + moved_index)
        r_new = all_e_new[global_moved_index]
        dtn_state_new = _advance_jastrow_dtn_ratio_state(
            dtn,
            state.dtn_state,
            global_moved_index,
            r_new,
            all_e_new,
        )
    return state.replace(j3_state=j3_state_new, dtn_state=dtn_state_new)


def _compute_grads_and_laplacian_Jastrow_three_body_debug(
    jastrow_three_body_data: Jastrow_three_body_data,
    r_up_carts: np.ndarray,
    r_dn_carts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """See _api method."""
    dtype_np = get_dtype_np("jastrow_grad_lap")
    diff_h = 1.0e-5

    # grad up
    grad_x_up = []
    grad_y_up = []
    grad_z_up = []
    for r_i, _ in enumerate(r_up_carts):
        diff_p_x_r_up_carts = r_up_carts.copy()
        diff_p_y_r_up_carts = r_up_carts.copy()
        diff_p_z_r_up_carts = r_up_carts.copy()
        diff_p_x_r_up_carts[r_i][0] += diff_h
        diff_p_y_r_up_carts[r_i][1] += diff_h
        diff_p_z_r_up_carts[r_i][2] += diff_h

        J3_p_x_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_y_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_p_z_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_p_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        diff_m_x_r_up_carts = r_up_carts.copy()
        diff_m_y_r_up_carts = r_up_carts.copy()
        diff_m_z_r_up_carts = r_up_carts.copy()
        diff_m_x_r_up_carts[r_i][0] -= diff_h
        diff_m_y_r_up_carts[r_i][1] -= diff_h
        diff_m_z_r_up_carts[r_i][2] -= diff_h

        J3_m_x_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_x_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_y_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_y_r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        J3_m_z_up = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=diff_m_z_r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        grad_x_up.append((J3_p_x_up - J3_m_x_up) / (2.0 * diff_h))
        grad_y_up.append((J3_p_y_up - J3_m_y_up) / (2.0 * diff_h))
        grad_z_up.append((J3_p_z_up - J3_m_z_up) / (2.0 * diff_h))

    # grad dn
    grad_x_dn = []
    grad_y_dn = []
    grad_z_dn = []
    for r_i, _ in enumerate(r_dn_carts):
        diff_p_x_r_dn_carts = r_dn_carts.copy()
        diff_p_y_r_dn_carts = r_dn_carts.copy()
        diff_p_z_r_dn_carts = r_dn_carts.copy()
        diff_p_x_r_dn_carts[r_i][0] += diff_h
        diff_p_y_r_dn_carts[r_i][1] += diff_h
        diff_p_z_r_dn_carts[r_i][2] += diff_h

        J3_p_x_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_x_r_dn_carts,
        )
        J3_p_y_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_y_r_dn_carts,
        )
        J3_p_z_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_p_z_r_dn_carts,
        )

        diff_m_x_r_dn_carts = r_dn_carts.copy()
        diff_m_y_r_dn_carts = r_dn_carts.copy()
        diff_m_z_r_dn_carts = r_dn_carts.copy()
        diff_m_x_r_dn_carts[r_i][0] -= diff_h
        diff_m_y_r_dn_carts[r_i][1] -= diff_h
        diff_m_z_r_dn_carts[r_i][2] -= diff_h

        J3_m_x_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_x_r_dn_carts,
        )
        J3_m_y_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_y_r_dn_carts,
        )
        J3_m_z_dn = compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=diff_m_z_r_dn_carts,
        )

        grad_x_dn.append((J3_p_x_dn - J3_m_x_dn) / (2.0 * diff_h))
        grad_y_dn.append((J3_p_y_dn - J3_m_y_dn) / (2.0 * diff_h))
        grad_z_dn.append((J3_p_z_dn - J3_m_z_dn) / (2.0 * diff_h))

    grad_J3_up = np.array([grad_x_up, grad_y_up, grad_z_up], dtype=dtype_np).T
    grad_J3_dn = np.array([grad_x_dn, grad_y_dn, grad_z_dn], dtype=dtype_np).T

    # laplacian (4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    diff_h2 = 1.0e-3

    J3_ref = compute_Jastrow_three_body(
        jastrow_three_body_data=jastrow_three_body_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    def _eval_up(r_up: np.ndarray) -> jax.Array:
        return compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up,
            r_dn_carts=r_dn_carts,
        )

    def _eval_dn(r_dn: np.ndarray) -> jax.Array:
        return compute_Jastrow_three_body(
            jastrow_three_body_data=jastrow_three_body_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn,
        )

    def _fd4_second_deriv(
        eval_fn: _FiniteDifferenceEval,
        r_carts: np.ndarray,
        r_i: int,
        dim: int,
        h: float,
        f0: jax.Array,
    ) -> jax.Array:
        r_p1 = r_carts.copy()
        r_p2 = r_carts.copy()
        r_m1 = r_carts.copy()
        r_m2 = r_carts.copy()
        r_p1[r_i][dim] += h
        r_p2[r_i][dim] += 2 * h
        r_m1[r_i][dim] -= h
        r_m2[r_i][dim] -= 2 * h
        return (-eval_fn(r_p2) + 16 * eval_fn(r_p1) - 30 * f0 + 16 * eval_fn(r_m1) - eval_fn(r_m2)) / (12 * h**2)

    lap_J3_up = np.zeros(len(r_up_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_up_carts):
        lap_J3_up[r_i] = (
            _fd4_second_deriv(_eval_up, r_up_carts, r_i, 0, diff_h2, J3_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 1, diff_h2, J3_ref)
            + _fd4_second_deriv(_eval_up, r_up_carts, r_i, 2, diff_h2, J3_ref)
        )

    lap_J3_dn = np.zeros(len(r_dn_carts), dtype=dtype_np)
    for r_i, _ in enumerate(r_dn_carts):
        lap_J3_dn[r_i] = (
            _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 0, diff_h2, J3_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 1, diff_h2, J3_ref)
            + _fd4_second_deriv(_eval_dn, r_dn_carts, r_i, 2, diff_h2, J3_ref)
        )

    return grad_J3_up, grad_J3_dn, lap_J3_up, lap_J3_dn
