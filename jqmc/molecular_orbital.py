"""Molecular Orbital module.

Precision Zones:
    - ``mo_eval``: forward MO evaluation (compute_MOs).
    - ``mo_grad``: MO gradient (compute_MOs_grad).
    - ``mo_lap``: MO Laplacian (compute_MOs_laplacian).

The fused :func:`compute_MOs_value_grad_lap` API returns all three at
once and casts each output into its corresponding zone at the matmul
use site. Use it when value, gradient, and Laplacian are all needed at
the same call site; otherwise prefer the standalone APIs.
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
from logging import getLogger

# jax modules
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit


from ._precision import cast_to_zone, get_dtype_jnp
from .atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    AOs_sphe_gto_data,
    _aos_cart_to_sphe,
    _aos_cart_to_sphe_with_metric,
    _aos_sphe_to_cart,
    _compute_AOs_debug,
    _compute_AOs_grad_autodiff,
    _compute_AOs_laplacian_autodiff,
    check_ao_eval_value_range,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    compute_AOs_value_grad_lap,
    is_complex_aos,
)
from .structure import Structure_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


@struct.dataclass
class MOs_data:
    """Molecular orbital (MO) coefficients and metadata.

    Holds the contraction matrix that maps atomic orbitals (AOs) to molecular
    orbitals (MOs). MO values are obtained as
    ``mo_coefficients @ AO_values``. Real AO data uses float64 coefficients;
    complex Bloch AO data uses complex128 coefficients.

    Attributes:
        num_mo (int): Number of molecular orbitals.
        aos_data (AOs_base_data): AO leaf or composite supplying the AO values.
        mo_coefficients: Real or complex coefficient matrix of shape
            ``(num_mo, num_ao)``. Rows correspond to MOs and columns to AOs.

    Examples:
        Minimal runnable setup (2 AOs -> 1 MO)::

            import numpy as np
            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_sphe_gto_data
            from jqmc.molecular_orbital import MOs_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            aos = AOs_sphe_gto_data(
                structure_data=structure,
                nucleus_index=[0, 1],
                num_ao=2,
                num_ao_prim=2,
                orbital_indices=[0, 1],
                exponents=[1.0, 1.2],
                coefficients=[1.0, 0.8],
                angular_momentums=[0, 0],
                magnetic_quantum_numbers=[0, 0],
            )
            aos.sanity_check()

            mo_coeffs = np.array([[0.7, 0.7]], dtype=float)  # shape (1, 2)
            mos = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=mo_coeffs)
            mos.sanity_check()
    """

    #: Number of molecular orbitals.
    num_mo: int = struct.field(pytree_node=False, default=0)
    #: AO definition supplying centers, exponents/coefficients, angular data, and contraction mapping.
    aos_data: AOs_base_data = struct.field(pytree_node=True, default_factory=lambda: AOs_sphe_gto_data())
    #: MO coefficient matrix, shape ``(num_mo, num_ao)``. dtype: float64 or complex128.
    mo_coefficients: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )
    #: Whether the AO -> MO map is the implicit square identity.  This avoids
    #: storing and multiplying a dense O(N^2) eye for frozen occupied orbitals.
    identity: bool = struct.field(pytree_node=False, default=False)

    def sanity_check(self) -> None:
        """Validate internal consistency.

        Ensures ``mo_coefficients`` matches ``(num_mo, M_c)``, verifies
        ``num_mo`` is an int, and delegates AO validation to
        ``aos_data.sanity_check()``. ``M_c`` is the molecular AO count
        ``aos_data.num_ao`` on every leaf (molecular and periodic alike).
        The image-expanded pgto kernel emits AOs at rank
        ``_n_image * num_ao`` but the pgto leaf post-reduces them to
        ``num_ao`` before this matrix sees them, so the contracted
        variational space is unchanged regardless of ``pbc_flag``.

        Raises:
            ValueError: If coefficient shape or ``num_mo`` type is invalid, or if ``aos_data`` fails its check.
            AOEvalRangeError: (a ``ValueError``) in mixed precision only, when a primitive of
                ``aos_data`` leaves the ``ao_eval`` dtype range
                (:func:`jqmc.atomic_orbital.check_ao_eval_value_range`).
        """
        molecular_ao_dim = self.aos_data.num_ao
        if self.identity:
            if self.num_mo != molecular_ao_dim:
                raise ValueError(f"implicit identity MOs require num_mo == num_ao; got {self.num_mo} and {molecular_ao_dim}.")
            if np.asarray(self.mo_coefficients).size != 0:
                raise ValueError("implicit identity MOs must store an empty mo_coefficients sentinel.")
        elif self.mo_coefficients.shape != (self.num_mo, molecular_ao_dim):
            raise ValueError(
                f"dim. of ao_coefficients = {self.mo_coefficients.shape} is wrong. "
                f"Inconsistent with the expected value = {(self.num_mo, molecular_ao_dim)}"
            )
        if not isinstance(self.num_mo, (int, np.integer)):
            raise ValueError(f"num_mo = {type(self.num_mo)} must be an int.")
        # The MO matrix dtype must match the AO leaf's twist regime. A complex
        # Bloch-pGTO leaf REQUIRES ``complex128`` MO coefficients (the AO
        # phase is unit-modulus, so dropping ``Im`` on the MO side would
        # silently force the wavefunction back onto the real path), and a
        # real AO leaf forbids complex MOs (no way to consume them
        # consistently). Cross-cutting failure is loud here rather than
        # producing junk far downstream.
        mo_dtype = np.asarray(self.mo_coefficients).dtype
        if is_complex_aos(self.aos_data):
            if mo_dtype != np.complex128:
                raise ValueError(f"complex Bloch-pGTO leaf requires complex128 mo_coefficients, got dtype = {mo_dtype}.")
        else:
            if not np.issubdtype(mo_dtype, np.floating):
                raise ValueError(
                    f"real AO leaf requires real mo_coefficients, got dtype = {mo_dtype} "
                    "(complex MOs are only allowed with a non-Gamma complex Bloch-pGTO leaf)."
                )
        self.aos_data.sanity_check()
        # Mixed precision only: an orbital set whose AO values leave the ao_eval
        # dtype range is rejected here, where the basis is configured into a
        # wavefunction (geminal / Hamiltonian / TREXIO and HDF5 load / driver
        # setup), instead of being evaluated silently wrong (issue #111).
        check_ao_eval_value_range(self.aos_data)

    @classmethod
    def build(
        cls,
        *,
        aos_data: AOs_base_data,
        mo_coefficients: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    ) -> "MOs_data":
        """Build a validated :class:`MOs_data` from AO data and a coefficient matrix.

        The blessed factory: it coerces ``mo_coefficients`` to a contiguous
        float64 ``(num_mo, num_ao)`` matrix, derives ``num_mo`` from its row
        count, constructs the instance, and runs :py:meth:`sanity_check`.
        Direct construction (``MOs_data(...)``) remains supported.

        Args:
            aos_data: AO definition (any leaf or composite); its ``num_ao``
                must equal the coefficient matrix column count.
            mo_coefficients: MO coefficient matrix of shape ``(num_mo, num_ao)``.

        Returns:
            MOs_data: The validated instance.
        """
        # Keep ``complex128`` when given complex (the TREXIO loader
        # routes complex MOs through here at non-Gamma twists); otherwise
        # coerce to ``float64`` (the real Gamma-point path).
        source_coefficients = np.asarray(mo_coefficients)
        storage_dtype = np.complex128 if np.iscomplexobj(source_coefficients) else np.float64
        contiguous_coefficients = np.ascontiguousarray(source_coefficients, dtype=storage_dtype)
        mos_data = cls(
            num_mo=int(contiguous_coefficients.shape[0]),
            aos_data=aos_data,
            mo_coefficients=contiguous_coefficients,
        )
        mos_data.sanity_check()
        return mos_data

    @classmethod
    def build_identity(cls, *, aos_data: AOs_base_data) -> "MOs_data":
        """Build an implicit ``C = I`` MO wrapper without an O(N^2) matrix.

        The representation is intended for frozen occupied orbitals.  Orbital
        evaluation dispatches directly to the AO leaf; callers that explicitly
        request a dense coefficient matrix materialise it lazily via
        :attr:`_mo_coefficients_jnp`. Core frozen-node paths, including MPC
        density construction, preserve the implicit representation.
        """
        dtype = np.complex128 if is_complex_aos(aos_data) else np.float64
        mos_data = cls(
            num_mo=int(aos_data.num_ao),
            aos_data=aos_data,
            mo_coefficients=np.empty((0,), dtype=dtype),
            identity=True,
        )
        mos_data.sanity_check()
        return mos_data

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        info_lines.append("**" + self.__class__.__name__)
        info_lines.append(f"  Number of MOs = {self.num_mo}")
        info_lines.append(
            "  MO coefficients = implicit identity" if self.identity else f"  dim. of MOs coeff = {self.mo_coefficients.shape}"
        )
        # Replace aos_data.logger_info() with aos_data.get_info() output.
        info_lines.extend(self.aos_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def structure_data(self) -> Structure_data:
        """Return structure_data of the aos_data instance."""
        return self.aos_data.structure_data

    @property
    def _num_orb(self) -> int:
        """Return the number of orbitals."""
        return self.num_mo

    @property
    def _mo_coefficients_jnp(self) -> jax.Array:
        """Return MO coefficients as a jax.Array (jnp view of the underlying numpy storage).

        Dtype follows the numpy storage (``float64`` for the real Gamma-point
        path, ``complex128`` once the loader populates the matrix with
        a complex MO at a non-Gamma twist). Consumer
        promotes/casts to its own zone at the use site via
        :func:`jnp.result_type`.
        """
        if self.identity:
            dtype = jnp.complex128 if is_complex_aos(self.aos_data) else jnp.float64
            return jnp.eye(self.num_mo, dtype=dtype)
        return jnp.asarray(self.mo_coefficients)

    def to_cartesian(self) -> "MOs_data":
        """Convert spherical AOs to Cartesian AOs and transform MO coefficients.

        Returns a new ``MOs_data`` whose AO leaf has Cartesian angular form.
        Molecular orbitals are preserved by applying an analytic
        (radial-independent) transformation per angular momentum shell.
        Complex Bloch pGTO inputs retain their concrete complex leaf, twist,
        and image phases. For Cartesian inputs the instance is returned unchanged.

        Raises:
            ValueError: If the underlying AOs are not spherical or the shell completeness
                check (``2l+1`` functions per shell) fails.
        """
        if isinstance(self.aos_data, AOs_cart_gto_data):
            return self
        if not isinstance(self.aos_data, AOs_sphe_gto_data):
            raise ValueError("Cartesian conversion is only available from spherical AOs.")
        aos_cart, transform_matrix = _aos_sphe_to_cart(self.aos_data)
        # Dtype promotion against the storage dtype, so a complex
        # ``mo_coefficients`` keeps ``Im`` through the matmul. Casting straight
        # to ``get_dtype_jnp("mo_eval")`` instead would silently drop ``Im``
        # (mo_eval is strictly real), and a trailing ``.astype(orig.dtype)``
        # only re-labels the array -- it cannot recover the lost imaginary part.
        src = np.asarray(self.mo_coefficients)
        out_dtype = np.result_type(np.dtype(get_dtype_jnp("mo_eval")), src.dtype)
        transform = np.asarray(transform_matrix, dtype=out_dtype)
        cart_coeffs = transform if self.identity else src.astype(out_dtype) @ transform
        cart_coeffs = cart_coeffs.astype(src.dtype, copy=False)

        return MOs_data(num_mo=self.num_mo, aos_data=aos_cart, mo_coefficients=cart_coeffs)

    def to_spherical(self) -> "MOs_data":
        """Project Cartesian MOs onto the pure-spherical AO subspace.

        Returns a new ``MOs_data`` whose AO leaf has spherical angular form.
        The angular embedding follows the CP2K solid-harmonic convention, and
        the reverse map minimizes the complete AO-function residual in the
        global molecular overlap metric or periodic Bloch metric ``S(k)``.
        The operation is exact for MOs already in the pure-spherical subspace;
        a general Cartesian-native MO can also contain lower-angular-momentum
        trace components, which are intentionally discarded.
        Complex Bloch pGTO inputs retain their concrete complex leaf, twist,
        and image phases. For spherical inputs the instance is returned unchanged.

        Raises:
            ValueError: If the underlying AOs are not Cartesian or the shell completeness
                check (``(l+1)(l+2)/2`` functions per shell with matching exps/coefs/orders)
                fails.
        """
        if isinstance(self.aos_data, AOs_sphe_gto_data):
            return self
        if not isinstance(self.aos_data, AOs_cart_gto_data):
            raise ValueError("Spherical conversion is only available from Cartesian AOs.")
        aos_sphe, cart_to_sphe = _aos_cart_to_sphe(self.aos_data)
        # Dtype promotion against the storage dtype --
        # see :py:meth:`to_cartesian` for the rationale.
        src = np.asarray(self.mo_coefficients)
        storage_dtype = np.result_type(src.dtype, np.asarray(cart_to_sphe).dtype)
        out_dtype = np.result_type(np.dtype(get_dtype_jnp("mo_eval")), storage_dtype)
        transform = np.asarray(cart_to_sphe, dtype=out_dtype)
        sph_coeffs = transform if self.identity else src.astype(out_dtype) @ transform
        sph_coeffs = sph_coeffs.astype(storage_dtype, copy=False)

        return MOs_data(num_mo=self.num_mo, aos_data=aos_sphe, mo_coefficients=sph_coeffs)

    def _spherical_projection_residual(self) -> npt.NDArray[np.float64]:
        r"""Return the relative norm of the part :py:meth:`to_spherical` discards, per MO.

        For each MO row ``c`` in the native Cartesian basis, the discarded
        function is ``r = c - (c R) T`` (``R`` the global S-metric reverse
        map, ``T`` the spherical embedding of :func:`_aos_cart_to_sphe_with_metric`),
        and the returned value is the scale-free ratio
        ``sqrt(conj(r) S r^T / conj(c) S c^T)`` in the same overlap metric
        ``S`` (molecular, or periodic Bloch ``S(k)``) that defines ``R``.
        It is zero for MOs in the pure-spherical subspace and for spherical
        inputs. Host-side fp64/complex128 setup arithmetic.
        """
        if isinstance(self.aos_data, AOs_sphe_gto_data):
            return np.zeros(self.num_mo, dtype=np.float64)
        if not isinstance(self.aos_data, AOs_cart_gto_data):
            raise ValueError("Spherical conversion is only available from Cartesian AOs.")
        _, cart_to_sphe, transform, overlap = _aos_cart_to_sphe_with_metric(self.aos_data)
        overlap = np.asarray(overlap)
        coeffs = np.eye(self.num_mo, overlap.shape[0]) if self.identity else np.asarray(self.mo_coefficients)
        dtype = np.result_type(coeffs.dtype, np.asarray(cart_to_sphe).dtype, overlap.dtype, np.float64)
        coeffs = coeffs.astype(dtype, copy=False)
        residual = coeffs - (coeffs @ np.asarray(cart_to_sphe, dtype=dtype)) @ np.asarray(transform, dtype=dtype)
        residual_norm_sq = np.einsum("ij,jk,ik->i", residual.conj(), overlap.astype(dtype, copy=False), residual).real
        norm_sq = np.einsum("ij,jk,ik->i", coeffs.conj(), overlap.astype(dtype, copy=False), coeffs).real
        return np.sqrt(np.maximum(residual_norm_sq, 0.0) / norm_sq)


@jit
def compute_MOs(mos_data: MOs_data, r_carts: jax.Array) -> jax.Array:
    """Evaluate molecular orbitals at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64 (same convention as
            AO evaluators).

    Returns:
        jax.Array: MO values with shape ``(num_mo, N_e)``.
    """
    # Heavy AO evaluation runs in ``ao_eval`` precision (e.g. fp32 in mixed mode),
    # but the (small) MO matmul and the returned MO matrix are kept in the
    # ``determinant`` precision (fp64 by default).  This avoids amplifying fp32
    # round-off through downstream determinant / kinetic / energy paths while
    # preserving the speed of the AO kernels. Dtype promotion: when any input is
    # complex (non-Gamma twist), ``jnp.result_type`` lifts ``out_dtype``
    # to ``complex128`` automatically, so the matmul is dtype-correct without
    # adding a new ``mo_eval_complex`` precision zone.
    ao_values = compute_AOs(aos_data=mos_data.aos_data, r_carts=r_carts)
    if mos_data.identity:
        return cast_to_zone(ao_values, "mo_eval")
    mo_coefficients = mos_data._mo_coefficients_jnp
    mo_values = jnp.dot(
        cast_to_zone(mo_coefficients, "mo_eval", ao_values.dtype),
        cast_to_zone(ao_values, "mo_eval", mo_coefficients.dtype),
    )

    return mo_values


def _compute_MOs_debug(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Evaluate MO values with the independent NumPy AO oracle.

    Args:
        mos_data: MO coefficients and the underlying AO definition.
        r_carts: Electron Cartesian coordinates with shape ``(N_e, 3)``.

    Returns:
        Real or complex MO values with shape ``(num_mo, N_e)``.
    """
    aos = _compute_AOs_debug(aos_data=mos_data.aos_data, r_carts=r_carts)
    answer = aos if mos_data.identity else np.dot(mos_data.mo_coefficients, aos)
    return answer


@jit
def _compute_MOs_laplacian_autodiff(
    mos_data: MOs_data,
    r_carts: jax.Array | npt.NDArray[np.float64],
) -> jax.Array:
    """Evaluate MO Laplacians with the autodiff AO oracle.

    Args:
        mos_data: MO coefficients and the underlying AO definition.
        r_carts: Electron Cartesian coordinates with shape ``(N_e, 3)``.

    Returns:
        MO Laplacians with shape ``(num_mo, N_e)``.
    """
    ao_laplacian = _compute_AOs_laplacian_autodiff(mos_data.aos_data, r_carts)
    mo_matrix_laplacian = ao_laplacian if mos_data.identity else jnp.dot(mos_data._mo_coefficients_jnp, ao_laplacian)

    return mo_matrix_laplacian


@jit
def compute_MOs_laplacian(mos_data: MOs_data, r_carts: jax.Array) -> jax.Array:
    """Compute MO laplacians at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64.

    Returns:
        jax.Array: Laplacians of each MO, shape ``(num_mo, N_e)``.
    """
    # Dtype promotion: consumer casts to ``mo_lap`` zone, then
    # ``jnp.result_type`` lifts to ``complex128`` if either input is complex.
    ao_lap = compute_AOs_laplacian(mos_data.aos_data, r_carts)
    if mos_data.identity:
        return cast_to_zone(ao_lap, "mo_lap")
    mo_coefficients = mos_data._mo_coefficients_jnp
    return jnp.dot(
        cast_to_zone(mo_coefficients, "mo_lap", ao_lap.dtype),
        cast_to_zone(ao_lap, "mo_lap", mo_coefficients.dtype),
    )


def _compute_MOs_laplacian_debug(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
) -> jax.Array:
    """Approximate MO Laplacians with a fourth-order central stencil.

    Args:
        mos_data: MO coefficients and the underlying AO definition.
        r_carts: Electron Cartesian coordinates with shape ``(N_e, 3)``.

    Returns:
        Finite-difference MO Laplacians with shape ``(num_mo, N_e)``.
    """
    # Laplacians of MOs (numerical, 4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    # Larger h is viable with the 4th-order stencil (O(h^4) truncation).
    diff_h = 1.0e-3

    mo_matrix = compute_MOs(mos_data, r_carts)

    def _compute_shifted_mos(cartesian_axis: int, step_multiplier: int) -> jax.Array:
        shifted_r_carts = r_carts.copy()
        shifted_r_carts[:, cartesian_axis] += step_multiplier * diff_h
        return compute_MOs(mos_data, shifted_r_carts)

    def _compute_second_derivative(cartesian_axis: int) -> jax.Array:
        mo_values_plus_one_step = _compute_shifted_mos(cartesian_axis, +1)
        mo_values_plus_two_steps = _compute_shifted_mos(cartesian_axis, +2)
        mo_values_minus_one_step = _compute_shifted_mos(cartesian_axis, -1)
        mo_values_minus_two_steps = _compute_shifted_mos(cartesian_axis, -2)
        return (
            -mo_values_plus_two_steps
            + 16 * mo_values_plus_one_step
            - 30 * mo_matrix
            + 16 * mo_values_minus_one_step
            - mo_values_minus_two_steps
        ) / (12 * diff_h**2)

    mo_matrix_grad2_x = _compute_second_derivative(0)
    mo_matrix_grad2_y = _compute_second_derivative(1)
    mo_matrix_grad2_z = _compute_second_derivative(2)

    mo_matrix_laplacian = mo_matrix_grad2_x + mo_matrix_grad2_y + mo_matrix_grad2_z

    return mo_matrix_laplacian


@jit
def compute_MOs_grad(
    mos_data: MOs_data, r_carts: jax.Array
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Compute MO gradients (x, y, z components) at electron coordinates.

    Args:
        mos_data (MOs_data): MO/AO definition and coefficient matrix.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in float64.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Gradients per component
        ``(grad_x, grad_y, grad_z)``, each with shape ``(num_mo, N_e)``.
    """
    # Dtype promotion: see :func:`compute_MOs_laplacian` for the
    # rationale. All three components share the same dtype.
    ao_grad_x, ao_grad_y, ao_grad_z = compute_AOs_grad(mos_data.aos_data, r_carts)
    if mos_data.identity:
        return (
            cast_to_zone(ao_grad_x, "mo_grad"),
            cast_to_zone(ao_grad_y, "mo_grad"),
            cast_to_zone(ao_grad_z, "mo_grad"),
        )
    mo_coefficients = mos_data._mo_coefficients_jnp
    coefficients_for_gradient = cast_to_zone(mo_coefficients, "mo_grad", ao_grad_x.dtype)
    mo_grad_x = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_grad_x, "mo_grad", mo_coefficients.dtype))
    mo_grad_y = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_grad_y, "mo_grad", mo_coefficients.dtype))
    mo_grad_z = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_grad_z, "mo_grad", mo_coefficients.dtype))

    return mo_grad_x, mo_grad_y, mo_grad_z


@jit
def compute_MOs_value_grad_lap(
    mos_data: MOs_data, r_carts: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused evaluation of MO values, Cartesian gradients, and Laplacians.

    Calls :func:`compute_AOs_value_grad_lap` once and applies
    ``mo_coefficients @ *`` to each AO output. Each MO output is cast
    into its own precision zone at the matrix multiplication:

    * ``val`` -> ``mo_eval`` (fp32 in mixed mode, fp64 in full)
    * ``gx`` / ``gy`` / ``gz`` -> ``mo_grad`` (fp64)
    * ``lap`` -> ``mo_lap`` (fp64)

    For value-only / grad-only / lap-only call sites, prefer the
    standalone APIs (``compute_MOs`` / ``compute_MOs_grad`` /
    ``compute_MOs_laplacian``) -- JAX DCE does not reliably eliminate
    unused outputs across this function's ``@jit`` boundary.

    Returns:
        tuple: ``(val, gx, gy, gz, lap)``, each of shape ``(num_mo, N_e)``.
    """
    ao_val, ao_gx, ao_gy, ao_gz, ao_lap = compute_AOs_value_grad_lap(aos_data=mos_data.aos_data, r_carts=r_carts)
    if mos_data.identity:
        return (
            cast_to_zone(ao_val, "mo_eval"),
            cast_to_zone(ao_gx, "mo_grad"),
            cast_to_zone(ao_gy, "mo_grad"),
            cast_to_zone(ao_gz, "mo_grad"),
            cast_to_zone(ao_lap, "mo_lap"),
        )
    # AO outputs live in their own zones (ao_val: ao_eval; ao_g*/ao_lap:
    # ao_grad_lap). Dtype promotion: cast each into the matching MO zone, with
    # ``jnp.result_type`` lifting to ``complex128`` if either operand is complex.
    mo_coefficients = mos_data._mo_coefficients_jnp

    mo_val = jnp.dot(
        cast_to_zone(mo_coefficients, "mo_eval", ao_val.dtype),
        cast_to_zone(ao_val, "mo_eval", mo_coefficients.dtype),
    )
    coefficients_for_gradient = cast_to_zone(mo_coefficients, "mo_grad", ao_gx.dtype)
    mo_gx = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_gx, "mo_grad", mo_coefficients.dtype))
    mo_gy = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_gy, "mo_grad", mo_coefficients.dtype))
    mo_gz = jnp.dot(coefficients_for_gradient, cast_to_zone(ao_gz, "mo_grad", mo_coefficients.dtype))
    mo_lap = jnp.dot(
        cast_to_zone(mo_coefficients, "mo_lap", ao_lap.dtype),
        cast_to_zone(ao_lap, "mo_lap", mo_coefficients.dtype),
    )

    return mo_val, mo_gx, mo_gy, mo_gz, mo_lap


@jit
def _compute_MOs_grad_autodiff(
    mos_data: MOs_data,
    r_carts: jax.Array | npt.NDArray[np.float64],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Cartesian MO gradients with the autodiff AO oracle.

    Args:
        mos_data: MO coefficients and the underlying AO definition.
        r_carts: Electron Cartesian coordinates with shape ``(N_e, 3)``.

    Returns:
        Gradient components ``(grad_x, grad_y, grad_z)``, each with shape
        ``(num_mo, N_e)``.
    """
    # Dtype promotion: mirror of :func:`compute_MOs_grad`.
    mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z = _compute_AOs_grad_autodiff(mos_data.aos_data, r_carts)
    if mos_data.identity:
        return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z
    mo_coefficients = mos_data._mo_coefficients_jnp
    coefficients_for_gradient = cast_to_zone(mo_coefficients, "mo_grad", mo_matrix_grad_x.dtype)
    mo_matrix_grad_x = jnp.dot(
        coefficients_for_gradient,
        cast_to_zone(mo_matrix_grad_x, "mo_grad", mo_coefficients.dtype),
    )
    mo_matrix_grad_y = jnp.dot(
        coefficients_for_gradient,
        cast_to_zone(mo_matrix_grad_y, "mo_grad", mo_coefficients.dtype),
    )
    mo_matrix_grad_z = jnp.dot(
        coefficients_for_gradient,
        cast_to_zone(mo_matrix_grad_z, "mo_grad", mo_coefficients.dtype),
    )

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z


def _compute_MOs_grad_debug(
    mos_data: MOs_data,
    r_carts: npt.NDArray[np.float64],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Approximate Cartesian MO gradients with central differences.

    Args:
        mos_data: MO coefficients and the underlying AO definition.
        r_carts: Electron Cartesian coordinates with shape ``(N_e, 3)``.

    Returns:
        Gradient components ``(grad_x, grad_y, grad_z)``, each with shape
        ``(num_mo, N_e)``.
    """
    # Gradients of MOs (numerical)
    diff_h = 1.0e-5

    # grad x
    plus_x_r_carts = r_carts.copy()
    plus_x_r_carts[:, 0] += diff_h
    mo_values_plus_x = compute_MOs(mos_data, plus_x_r_carts)
    minus_x_r_carts = r_carts.copy()
    minus_x_r_carts[:, 0] -= diff_h
    mo_values_minus_x = compute_MOs(mos_data, minus_x_r_carts)

    # grad y
    plus_y_r_carts = r_carts.copy()
    plus_y_r_carts[:, 1] += diff_h
    mo_values_plus_y = compute_MOs(mos_data, plus_y_r_carts)
    minus_y_r_carts = r_carts.copy()
    minus_y_r_carts[:, 1] -= diff_h
    mo_values_minus_y = compute_MOs(mos_data, minus_y_r_carts)

    # grad z
    plus_z_r_carts = r_carts.copy()
    plus_z_r_carts[:, 2] += diff_h
    mo_values_plus_z = compute_MOs(mos_data, plus_z_r_carts)
    minus_z_r_carts = r_carts.copy()
    minus_z_r_carts[:, 2] -= diff_h
    mo_values_minus_z = compute_MOs(mos_data, minus_z_r_carts)

    mo_matrix_grad_x = (mo_values_plus_x - mo_values_minus_x) / (2.0 * diff_h)
    mo_matrix_grad_y = (mo_values_plus_y - mo_values_minus_y) / (2.0 * diff_h)
    mo_matrix_grad_z = (mo_values_plus_z - mo_values_minus_z) / (2.0 * diff_h)

    return mo_matrix_grad_x, mo_matrix_grad_y, mo_matrix_grad_z
