"""Hamiltonian module.

Precision Zones:
    - ``local_energy``: assembly of the local energy from the kinetic (T) and
      Coulomb (V) parts (compute_local_energy*).

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
import collections.abc
import dataclasses
import importlib
import weakref
from logging import getLogger
from os import PathLike
from typing import Any, TypeVar

import h5py

# JAX
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit
from jax import typing as jnpt

from ._precision import get_dtype_jnp
from ._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    MO_ORTHONORMALITY_WARNING_THRESHOLD,
)
from .atomic_orbital import (
    AOs_base_data,
    AOs_cart_gto_data,
    AOs_cart_pgto_data,
    AOs_composite_data,
    AOs_bspline_tiled_complex_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_data,
    _apply_image_expansion,
    _apply_image_expansion_complex,
    _detile_ragged_to_molecular,
    compute_overlap_matrix,
)
from .coulomb_potential import (
    Coulomb_potential_data,
    Ewald_data,
    build_ewald_data,
    compute_coulomb_potential,
    compute_coulomb_potential_fast,
)
from .determinant import Det_ratio_streaming_state, Det_streaming_state
from .jastrow_factor import Jastrow_data
from .molecular_orbital import MOs_data
from .structure import Structure_data, _strain_structure_data
from .wavefunction import (
    Wavefunction_data,
    _compute_kinetic_energy_auto,
    compute_kinetic_energy,
    compute_kinetic_energy_all_elements_fast_update,
)

T = TypeVar("T")

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# separator
num_sep_line = 66


def _is_twist_average_layout(group: h5py.Group) -> bool:
    """Recognize a combined input from its payload, including incomplete ones.

    The actual reader validates completeness. Keeping partial combined inputs
    on that route prevents a damaged file from falling back to a single twist.
    """
    return "shared" in group or "twists" in group


def _require_twist_average_layout(group: h5py.Group) -> int:
    """Validate the combined container and return its declared twist count."""
    for name in ("_meta", "shared", "twists"):
        if name not in group or not isinstance(group[name], h5py.Group):
            raise ValueError(f"twist-averaged input requires the /{name} group")
    count = group["_meta"].attrs.get("n_twist")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError("twist-averaged input requires a positive integer /_meta/n_twist")
    if len(group["twists"]) != count:
        raise ValueError("twist-averaged input requires exactly the declared twist groups")
    expected = {f"twist_{index:03d}" for index in range(int(count))}
    if set(group["twists"]) != expected or any(not isinstance(child, h5py.Group) for child in group["twists"].values()):
        raise ValueError("twist-averaged input requires exactly the declared twist groups")
    return int(count)


@struct.dataclass
class Hamiltonian_data:
    """Hamiltonian dataclass.

    The class contains data for computing Kinetic and Potential energy terms.

    Args:
        structure_data (Structure_data): an instance of Structure_data
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data

    Notes:
        Here are the differentiable arguments, i.e., pytree_node = True
        This information is a little bit tricky in terms of a principle of the object-oriented programming,
        'Don't ask, but tell' (i.e., the Hamiltonian_data knows the details of the other classes
        too much), but there is no other choice to dynamically switch on and off pytree_nodes depending
        on optimized variational parameters chosen by a user because @dataclass is statistically generated.

        WF parameters related:
            - lambda in wavefunction_data.geminal_data (determinant.py)
            - jastrow_2b_param in wavefunction_data.jastrow_data.jastrow_two_body_data (jastrow_factor.py)
            - j_matrix in wavefunction_data.jastrow_data.jastrow_three_body_data (jastrow_factor.py)

        Atomic positions related:
            - positions in hamiltonian_data.structure_data (this file)
            - positions in wavefunction_data.geminal_data.mos_data/aos_data.structure_data (molecular_orbital.py/atomic_orbital.py)
            - positions in wavefunction_data.jastrow_data.jastrow_three_body_data.mos_data/aos_data.structure_data (jastrow_factor.py)
            - positions in Coulomb_potential_data.structure_data (coulomb_potential.py)
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    coulomb_potential_data: Coulomb_potential_data = struct.field(
        pytree_node=True, default_factory=lambda: Coulomb_potential_data()
    )
    wavefunction_data: Wavefunction_data = struct.field(pytree_node=True, default_factory=lambda: Wavefunction_data())

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments. After the
        component checks pass, the MO orthonormality of the determinant part
        is diagnosed by :py:func:`_warn_mo_orthonormality`; that diagnostic
        only logs a warning and never raises.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument,
                or a Jastrow component's periodicity / lattice differs from ``structure_data``.
        """
        self.structure_data.sanity_check()
        self.coulomb_potential_data.sanity_check()
        self.wavefunction_data.sanity_check()
        from .atomic_orbital import _structure_data_equal
        from .orbital_tiling import TiledOrbitalSet_data

        orbitals = self.wavefunction_data.geminal_data.orb_data_up_spin
        if isinstance(orbitals, TiledOrbitalSet_data):
            if self.coulomb_potential_data.ewald_data is None:
                raise ValueError("tiled orbitals require a periodic physical Hamiltonian")
            physical_structures = (
                orbitals.structure_data,
                self.coulomb_potential_data.structure_data,
                self.coulomb_potential_data.ewald_data.structure_data,
            )
            if any(not _structure_data_equal(s, self.structure_data) for s in physical_structures):
                raise ValueError("tiled orbitals, Coulomb and Hamiltonian must share the physical supercell structure")
        if self.wavefunction_data.jastrow_data is not None:
            # Jastrow components evaluate with their own structure; their
            # periodicity must be the Hamiltonian's (issue #147).
            self.wavefunction_data.jastrow_data.check_structure_consistency(self.structure_data)
        _warn_mo_orthonormality(self.wavefunction_data)

    @classmethod
    def build(
        cls,
        *,
        structure_data: Structure_data,
        coulomb_potential_data: Coulomb_potential_data,
        wavefunction_data: Wavefunction_data,
    ) -> "Hamiltonian_data":
        """Build a validated :class:`Hamiltonian_data` (the top-level container).

        Blessed factory: assembles the structure, Coulomb, and wavefunction
        components, constructs the instance, and runs :py:meth:`sanity_check`
        (which recursively validates each component and the periodic
        ``pbc_flag`` / ``ewald_data`` contract). Direct construction
        (``Hamiltonian_data(...)``) remains supported.
        """
        data = cls(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        """Return a list of strings representing the logged information."""
        info_lines = []
        # Add the top separator line.
        info_lines.append("=" * num_sep_line)
        # Replace attribute logger_info() calls with their get_info() outputs.
        info_lines.extend(self.structure_data._get_info())
        info_lines.extend(self.coulomb_potential_data._get_info())
        info_lines.extend(self.wavefunction_data._get_info())
        # Add the bottom separator line.
        info_lines.append("=" * num_sep_line)
        return info_lines

    def _logger_info(self) -> None:
        """Log the information from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    def accumulate_position_grad(self, grad_hamiltonian: "Hamiltonian_data") -> jax.Array:
        """Aggregate position gradients from Hamiltonian components (structure + wavefunction)."""
        grad = grad_hamiltonian.structure_data.positions
        grad += grad_hamiltonian.coulomb_potential_data.structure_data.positions
        # The in-trace Ewald ii/ee/ei kernels read
        # ``ewald_data.structure_data._cell_jnp`` and (for ii)
        # ``_positions_cart_jnp``, so ion-ion position grads flow into the
        # ``ewald_data.structure_data`` leaf. Walk it here to recover the
        # ion-ion HF contribution.
        if (
            grad_hamiltonian.coulomb_potential_data.ewald_data is not None
            and self.coulomb_potential_data.ewald_data is not None
        ):
            ewald_grad_sd = grad_hamiltonian.coulomb_potential_data.ewald_data.structure_data
            if ewald_grad_sd is not None:
                grad += ewald_grad_sd.positions
        if self.wavefunction_data is not None and grad_hamiltonian.wavefunction_data is not None:
            grad += self.wavefunction_data.accumulate_position_grad(grad_hamiltonian.wavefunction_data)
        return grad

    def accumulate_strain_grad(self, grad_hamiltonian: "Hamiltonian_data") -> jax.Array:
        r"""Aggregate Cauchy-Born strain cotangents across the Hamiltonian tree.

        Returns the ``(3, 3) fp64`` strain-cotangent matrix

        .. math::
            \tilde{\sigma}_{\alpha\beta}
                = \sum_\text{leaves} \left[
                    (\nabla_\text{cell} E)^T \cdot \text{cell}
                    + (\nabla_\text{positions} E)^T \cdot \text{positions}
                  \right]

        summed over every ``Structure_data`` leaf reachable from this
        Hamiltonian (own ``structure_data``, ``coulomb_potential_data``,
        and the wavefunction tree). The returned tensor is generally
        non-symmetric; the estimator path symmetrises and divides by
        volume to obtain the physical stress tensor :math:`\sigma`.

        Returns ``jnp.zeros((3, 3))`` for molecular structures
        (``self.structure_data.pbc_flag == False``) so callers can treat
        the output buffer uniformly across periodic and molecular runs.

        Args:
            grad_hamiltonian: Hamiltonian-shaped cotangent pytree, e.g.
                ``jax.grad(local_energy)(self, ...)``.

        Returns:
            jax.Array: ``(3, 3) fp64`` strain-cotangent matrix.
        """
        if not self.structure_data.pbc_flag:
            return jnp.zeros((3, 3), dtype=jnp.float64)
        from .structure import _accumulate_strain_grad_per_leaf

        sigma_tilde = _accumulate_strain_grad_per_leaf(grad_hamiltonian.structure_data, self.structure_data)
        sigma_tilde = sigma_tilde + _accumulate_strain_grad_per_leaf(
            grad_hamiltonian.coulomb_potential_data.structure_data,
            self.coulomb_potential_data.structure_data,
        )
        # The in-trace Ewald ii/ee/ei kernels read
        # cell (and ii reads positions) from ``ewald_data.structure_data``,
        # so cell- and position-cotangents from those channels land on
        # ``ewald_data.structure_data`` rather than
        # ``coulomb_potential_data.structure_data``. Walk it here.
        if (
            grad_hamiltonian.coulomb_potential_data.ewald_data is not None
            and self.coulomb_potential_data.ewald_data is not None
        ):
            sigma_tilde = sigma_tilde + _accumulate_strain_grad_per_leaf(
                grad_hamiltonian.coulomb_potential_data.ewald_data.structure_data,
                self.coulomb_potential_data.ewald_data.structure_data,
            )
        if self.wavefunction_data is not None and grad_hamiltonian.wavefunction_data is not None:
            sigma_tilde = sigma_tilde + self.wavefunction_data.accumulate_strain_grad(grad_hamiltonian.wavefunction_data)
        return sigma_tilde

    def _apply_strain(self, F: jax.Array) -> "Hamiltonian_data":
        """Return a new ``Hamiltonian_data`` with the affine deformation ``F`` applied.

        Walks every ``Structure_data`` leaf reachable from this Hamiltonian
        (top-level, ``coulomb_potential_data``, the Ewald rebuild's leaf,
        and the wavefunction tree) and applies
        :py:func:`jqmc.structure._strain_structure_data` to each. This is
        the constructive dual of :py:meth:`accumulate_strain_grad`: when
        ``F = I + eps`` and ``compute_local_energy`` is evaluated on the
        strained Hamiltonian + strained electron coordinates (ECP quadrature
        rotation unchanged), :py:func:`jax.jacrev` with respect to ``eps`` at zero
        recovers the local stress kernel.

        Contract (fixed static ansatz): only geometry moves. ``F`` maps the
        cell rows and every nucleus (the caller maps the electrons the same way
        and keeps the ECP quadrature directions fixed); basis exponents and contraction coefficients,
        MO / geminal coefficients and Jastrow parameters are held fixed, and
        comoving data (plane-wave reciprocal indices, lattice-image and Ewald
        index sets) move with the cell. Derivatives taken through this map
        are fixed-ansatz strain derivatives, NOT the equation-of-state
        derivative of a re-optimized ground-state energy.

        Args:
            F: ``(3, 3) fp64`` deformation gradient.

        Returns:
            Hamiltonian_data: a new instance with every leaf strained.
        """
        new_sd = _strain_structure_data(self.structure_data, F)
        new_cpd_sd = _strain_structure_data(self.coulomb_potential_data.structure_data, F)
        new_cpd = self.coulomb_potential_data.replace(structure_data=new_cpd_sd)
        if self.coulomb_potential_data.ewald_data is not None:
            new_ewd_sd = _strain_structure_data(self.coulomb_potential_data.ewald_data.structure_data, F)
            new_ewd = self.coulomb_potential_data.ewald_data.replace(structure_data=new_ewd_sd)
            new_cpd = new_cpd.replace(ewald_data=new_ewd)
        new_wd = self.wavefunction_data
        if new_wd is not None:
            new_wd = new_wd._apply_strain(F)
        return self.replace(
            structure_data=new_sd,
            coulomb_potential_data=new_cpd,
            wavefunction_data=new_wd,
        )

    def save_to_hdf5(self, filepath: str | PathLike[str] = "jqmc.h5") -> None:
        """Save Hamiltonian data to an HDF5 file.

        Stores the Hamiltonian payload together with ``jqmc_version`` and
        ``jqmc_git_commit`` provenance. The reader validates the
        stored data rather than requiring a schema-version attribute.

        Args:
            filepath (str, optional): file path
        """
        from ._build_info import build_info

        info = build_info()
        with h5py.File(filepath, "w") as f:
            f.attrs["jqmc_version"] = info.version
            f.attrs["jqmc_git_commit"] = info.git_commit
            _save_dataclass_to_hdf5(f, self)

    @staticmethod
    def load_from_hdf5(filepath: str | PathLike[str] = "jqmc.h5") -> "Hamiltonian_data":
        """Load Hamiltonian data from an HDF5 file.

        Recognizes the actual single-Hamiltonian payload and rejects combined
        twist containers. Obsolete version attributes do not select a reader.
        After load, every periodic ``AOs_*_data`` is re-expanded via
        :py:func:`_apply_image_expansion_post_load`.

        .. warning::

            **Round-trip equality is precision-mode-dependent.** The six
            post-load hooks below (``_apply_image_expansion_post_load``,
            ``_apply_ewald_rebuild_post_load``, ``_apply_jastrow_structure_backfill_post_load``,
            ``_apply_{ao_basis,geminal,dtn}_dtype_alignment_post_load``)
            normalise the loaded values to the **active precision zone**
            so the JIT cache stays stable across SR iterations. In the current
            mixed map this casts AO exponents / coefficients and DTN params
            from float64 to float32; ``det_eval`` remains float64, so a real
            lambda_matrix stays float64 (and a complex one stays complex128). The
            ``save_to_hdf5`` -> ``load_from_hdf5`` round-trip is therefore
            **NOT bit-exact across precision modes**:

            - full mode: identity (fp64/complex128 arithmetic).
            - mixed mode: loaded values match the original
              up to the loosest zone tolerance among ``(ao_eval, jastrow_eval,
              det_eval)``. Use :py:func:`jqmc._setting.get_tol_hdf5_roundtrip`
              for any round-trip equality assertion (NEVER inline
              ``pytest.approx(abs=1e-14)`` or ``np.testing.assert_array_equal``;
              both fail under mixed precision). The project policy is
              "every tolerance in jqmc/_setting.py".

        Args:
            filepath (str, optional): file path

        Returns:
            Hamiltonian_data: An instance of Hamiltonian_data.
        """
        with h5py.File(filepath, "r") as f:
            if _is_twist_average_layout(f):
                raise ValueError(
                    "This is a twist-averaged input layout; "
                    "Hamiltonian_data.load_from_hdf5 cannot read it. Use "
                    "jqmc._twist_average.load_twist(filepath, twist_index) "
                    "instead."
                )
            obj = _load_dataclass_from_hdf5(Hamiltonian_data, f)
        obj = _apply_image_expansion_post_load(obj)
        obj = _apply_ewald_rebuild_post_load(obj)
        obj = _apply_jastrow_structure_backfill_post_load(obj)
        obj = _apply_ao_basis_dtype_alignment_post_load(obj)
        obj = _apply_geminal_dtype_alignment_post_load(obj)
        obj = _apply_dtn_dtype_alignment_post_load(obj)
        # The recursive dataclass loader cannot run full validation before
        # periodic AO leaves have been re-expanded and retwisted. Close every
        # invariant only after all post-load fixups. Besides the canonical SD
        # boundary, this rejects persisted DTN cutoffs that violate the
        # min-image smoothness bound (2*L_f < R_inscribed) before the loaded
        # Hamiltonian can be evaluated.
        obj.sanity_check()
        return obj


def _aos_overlap_unavailable_reason(aos_data: AOs_base_data) -> str | None:
    """Why :py:func:`compute_overlap_matrix` gives no usable overlap for ``aos_data`` (``None`` if it does).

    Every concrete AO leaf owns an overlap kernel except a molecular
    (open-boundary) :class:`AOs_composite_data`, which has no shared cell to
    integrate over (its ``_compute_overlap_matrix`` raises
    ``NotImplementedError`` by design); the MO orthonormality diagnostic
    therefore skips it explicitly. The standalone complex B-spline leaf is
    measured like every other leaf: its overlap is the exact cell Gram
    matrix.
    """
    if isinstance(aos_data, AOs_composite_data) and not aos_data.structure_data.pbc_flag:
        return "molecular AO composite (no shared cell)"
    return None


def _compute_mo_gram_matrix(
    mos_data: MOs_data,
    overlap_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    r"""Return the MO Gram matrix :math:`M = \overline{C}\,S\,C^{T}` in fp64 / complex128.

    ``C = mos_data.mo_coefficients`` has shape ``(num_mo, num_ao)`` (rows are
    MOs) and ``overlap_matrix`` is the AO overlap of ``mos_data.aos_data`` in
    the :py:func:`compute_overlap_matrix` convention
    :math:`S_{\mu\nu} = \int \overline{\chi_\mu}\,\chi_\nu` (one cell for
    periodic leaves, all space for molecular ones). With
    :math:`\phi_i = \sum_\mu C_{i\mu}\chi_\mu` this gives
    :math:`M_{ij} = \langle\phi_i|\phi_j\rangle = \int \overline{\phi_i}\,\phi_j`.
    Both inputs are promoted to float64 / complex128 regardless of the
    precision mode.
    """
    overlap = np.asarray(overlap_matrix)
    if mos_data.identity:
        return overlap.astype(np.complex128 if np.iscomplexobj(overlap) else np.float64)
    coefficients = np.asarray(mos_data.mo_coefficients)
    dtype = np.complex128 if (np.iscomplexobj(coefficients) or np.iscomplexobj(overlap)) else np.float64
    coefficients = coefficients.astype(dtype)
    overlap = overlap.astype(dtype)
    return coefficients.conj() @ overlap @ coefficients.T


def _compute_mo_orthonormality_metrics(
    mos_data: MOs_data,
    overlap_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> tuple[float, float]:
    r"""Return the MO normalization deviation and normalized non-orthogonality.

    With :math:`M` from :py:func:`_compute_mo_gram_matrix`, the two returned
    numbers are :math:`\max_i |M_{ii} - 1|` and
    :math:`\max_{i \ne j} |M_{ij}| / \sqrt{M_{ii} M_{jj}}` (zero for a single
    MO). The second is invariant to a common or per-MO rescaling, so a global
    ``sqrt(Omega)`` convention factor only shows up in the first. A
    non-positive diagonal entry (a null MO row) contributes no off-diagonal
    ratio; its normalization deviation is already at least one.
    """
    return _orthonormality_metrics_from_gram(_compute_mo_gram_matrix(mos_data, overlap_matrix))


def _orthonormality_metrics_from_gram(
    gram: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> tuple[float, float]:
    """The two metrics of :py:func:`_compute_mo_orthonormality_metrics` from an MO Gram matrix."""
    diagonal = np.real(np.diag(gram))
    normalization_deviation = float(np.max(np.abs(diagonal - 1.0)))
    scale = np.sqrt(np.clip(np.outer(diagonal, diagonal), 0.0, None))
    ratio = np.divide(np.abs(gram), scale, out=np.zeros(scale.shape, dtype=np.float64), where=scale > 0.0)
    np.fill_diagonal(ratio, 0.0)
    non_orthogonality = float(np.max(ratio))
    return normalization_deviation, non_orthogonality


#: Metrics of the MO orthonormality diagnostic per ``MOs_data`` object, kept
#: while that object is alive (``id`` -> (weak reference, metrics)). A
#: Hamiltonian loaded from HDF5 and then passed to MCMC / GFMC is checked twice
#: in one run; the second check reuses the metrics instead of the AO overlap
#: (issue #247). The warning itself is still emitted on every check.
_MO_ORTHONORMALITY_METRICS: dict[int, tuple[weakref.ref, tuple[float, float]]] = {}


def _mo_orthonormality_metrics_once(
    orb_data: MOs_data,
    dense_overlap: collections.abc.Callable[[AOs_base_data], npt.ArrayLike],
) -> tuple[float, float]:
    """Metrics of ``orb_data``, computed once per object.

    The Gram matrix comes from the AO leaf's ``_overlap_gram``, which uses the
    closed-form overlap of plane-wave leaves and blocks (no ``num_ao^2`` array)
    and ``dense_overlap`` otherwise.
    """
    key = id(orb_data)
    cached = _MO_ORTHONORMALITY_METRICS.get(key)
    if cached is not None and cached[0]() is orb_data:
        return cached[1]
    aos_data = orb_data.aos_data
    coefficients = np.asarray(orb_data.mo_coefficients)
    metrics = _orthonormality_metrics_from_gram(aos_data._overlap_gram(coefficients, coefficients, dense_overlap))
    _MO_ORTHONORMALITY_METRICS[key] = (
        weakref.ref(orb_data, lambda _ref, key=key: _MO_ORTHONORMALITY_METRICS.pop(key, None)),
        metrics,
    )
    return metrics


def _warn_mo_orthonormality(wavefunction_data: Wavefunction_data) -> None:
    """Warn when the determinant MOs are not orthonormal in the AO metric.

    Diagnostic only: never raises, never mutates. For every
    :class:`MOs_data` reachable from ``wavefunction_data.geminal_data``
    (``orb_data_up_spin`` / ``orb_data_dn_spin`` of SD, AGPn, or an AGP whose
    orbital data are MOs), the metrics of
    :py:func:`_compute_mo_orthonormality_metrics` are compared with
    :data:`jqmc._setting.MO_ORTHONORMALITY_WARNING_THRESHOLD`, and one
    ``logger.warning`` is emitted per channel where either exceeds it (or is
    not finite). The two numbers are always logged at debug level. Skipped
    explicitly (debug log only): no geminal, AO-basis orbital data (no MOs),
    an empty MO set, and AO leaves without a defined overlap (see
    :py:func:`_aos_overlap_unavailable_reason`). The Gram matrix comes from the
    AO leaf's ``_overlap_gram``: plane-wave leaves and the plane-wave blocks of
    composites use their closed-form overlap and never form a ``num_ao^2``
    array; other leaves use the dense overlap, computed at most once per AO
    leaf object per call (issue #247). The metrics are computed once per
    ``MOs_data`` object for as long as it lives, so the HDF5 load and the MCMC /
    GFMC initialization of one run share them; the warning is still emitted on
    every check. This runs from :py:meth:`Hamiltonian_data.sanity_check`
    (build, HDF5 load, MCMC / GFMC initialization), never per optimization step.
    """
    geminal_data = wavefunction_data.geminal_data
    if geminal_data is None:
        logger.debug("MO orthonormality check skipped: the wavefunction has no geminal.")
        return
    geminal_name = type(geminal_data).__name__
    overlap_by_aos: dict[int, Any] = {}

    def dense_overlap(leaf: AOs_base_data) -> npt.ArrayLike:
        # One dense overlap per AO leaf object per call (the up/dn channels share it).
        overlap = overlap_by_aos.get(id(leaf))
        if overlap is None:
            overlap = np.asarray(compute_overlap_matrix(leaf))
            overlap_by_aos[id(leaf)] = overlap
        return overlap

    orb_data_up = geminal_data.orb_data_up_spin
    orb_data_dn = geminal_data.orb_data_dn_spin
    if orb_data_dn is orb_data_up:
        channels = ((f"{geminal_name} spin-up/dn", orb_data_up),)
    else:
        channels = ((f"{geminal_name} spin-up", orb_data_up), (f"{geminal_name} spin-dn", orb_data_dn))
    for label, orb_data in channels:
        if not isinstance(orb_data, MOs_data):
            logger.debug(f"MO orthonormality check skipped for {label}: orbital data is {type(orb_data).__name__}, not MOs.")
            continue
        if int(orb_data.num_mo) == 0:
            logger.debug(f"MO orthonormality check skipped for {label}: no MOs.")
            continue
        aos_data = orb_data.aos_data
        if orb_data.identity and isinstance(aos_data, AOs_bspline_tiled_complex_data):
            # The exact tiled overlap is block diagonal in primitive k, but
            # materialising its full supercell-rank matrix would reintroduce an
            # O(N^2) setup allocation solely for this optional warning.
            logger.debug(f"MO orthonormality check skipped for {label}: implicit tiled identity basis.")
            continue
        unavailable_reason = _aos_overlap_unavailable_reason(aos_data)
        if unavailable_reason is not None:
            logger.debug(f"MO orthonormality check skipped for {label}: no usable AO overlap, {unavailable_reason}.")
            continue
        normalization_deviation, non_orthogonality = _mo_orthonormality_metrics_once(orb_data, dense_overlap)
        logger.debug(
            f"MO orthonormality ({label}): max|M_ii - 1| = {normalization_deviation:.3e}, "
            f"max_(i!=j) |M_ij|/sqrt(M_ii M_jj) = {non_orthogonality:.3e}."
        )
        # Written as "not (both within)" so that non-finite metrics (NaN / inf
        # MO coefficients) also warn instead of silently passing.
        within = (
            normalization_deviation <= MO_ORTHONORMALITY_WARNING_THRESHOLD
            and non_orthogonality <= MO_ORTHONORMALITY_WARNING_THRESHOLD
        )
        if not within:
            logger.warning(
                f"MOs are not orthonormal in the AO metric M = conj(C) S C^T ({label}, {int(orb_data.num_mo)} MOs): "
                f"max|M_ii - 1| = {normalization_deviation:.3e}, "
                f"max_(i!=j) |M_ij|/sqrt(M_ii M_jj) = {non_orthogonality:.3e} "
                f"(warning threshold {MO_ORTHONORMALITY_WARNING_THRESHOLD:.1e}). "
                "Hint: QE-derived periodic MOs written before the unit-cell normalization convention carry a common "
                "sqrt(Omega) scale and are expected to warn; physics is unaffected. Newer QE conversions are "
                "unit-cell normalized (pure pgto also orthonormalized); truncated pw and the pgto+pw / "
                "pgto+bspline kinds can still warn on their natural projection non-orthogonality, which "
                "does not affect a single determinant."
            )


def get_num_electrons(filepath: str = "hamiltonian_data.h5") -> tuple[int, int]:
    """Return ``(num_electron_up, num_electron_dn)`` from a single-twist Hamiltonian HDF5.

    Reads the electron counts from ``wavefunction_data.geminal_data`` of a saved
    :class:`Hamiltonian_data`. The object is reconstructed via
    :meth:`Hamiltonian_data.load_from_hdf5` (rather than a raw h5py dataset read)
    so the counts are correct for every geminal class and representation.

    This is a pure *fact* accessor exposed for jqmc-workflow's ``'auto'``
    resolution of ``num_mcmc_per_measurement`` (``= 4 * total electrons``). The
    driver never calls it: the policy (the x4 rule) lives in the workflow, the
    mechanism (reading a fact from the h5) lives here.

    Args:
        filepath: Path to a single-twist ``Hamiltonian_data`` HDF5 file.

    Returns:
        ``(num_electron_up, num_electron_dn)`` as Python ints.

    Raises:
        ValueError: propagated from :meth:`Hamiltonian_data.load_from_hdf5` when
            ``filepath`` is a twist-averaged (combined) combined input. Electron counts
            are twist-invariant, but the single-twist loader cannot consume combined inputs.
    """
    ham = Hamiltonian_data.load_from_hdf5(filepath)
    geminal_data = ham.wavefunction_data.geminal_data
    return int(geminal_data.num_electron_up), int(geminal_data.num_electron_dn)


def compute_local_energy(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
) -> float:
    """Compute Local Energy.

    The method is for computing the local energy at (r_up_carts, r_dn_carts).

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike):
            Rotation matrix. equiv R.T used for non-local part. It does not affect all-electron calculations.

    Returns:
        float: The value of local energy (e_L) with the given wavefunction (float)
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). Each
    # downstream consumer (compute_kinetic_energy, compute_coulomb_potential)
    # casts to its own zone at the use site.

    T = compute_kinetic_energy(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    V = compute_coulomb_potential(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    # Cast scalar zone outputs to local_energy zone at the sum (Principle 3b).
    # Promote dtype_jnp against the actual T / V dtypes so the sum preserves
    # Im on complex Bloch (twist k != 0). Bit-identical at Gamma where T and
    # V are both real.
    dtype_jnp = jnp.result_type(get_dtype_jnp("local_energy"), T.dtype, V.dtype)
    return T.astype(dtype_jnp) + V.astype(dtype_jnp)


def compute_local_energy_fast(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
    geminal_inverse: jnpt.ArrayLike,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> jax.Array:
    """Compute local energy using a precomputed geminal inverse.

    Identical to :func:`compute_local_energy` but avoids re-computing the
    LU decomposition of the geminal matrix by reusing ``geminal_inverse``
    supplied by the caller (e.g. the Sherman-Morrison inverse maintained
    inside the MCMC loop).  When the geminal matrix is near-singular the
    fresh LU decomposition inside :func:`compute_local_energy` produces
    NaN, whereas the Sherman-Morrison inverse has already been regularized
    by the AS acceptance/rejection, making this variant numerically safer.

    Args:
        hamiltonian_data (Hamiltonian_data):
            An instance of Hamiltonian_data.
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons with shape ``(N_up, 3)``.
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of down-spin electrons with shape ``(N_dn, 3)``.
        RT (jnpt.ArrayLike):
            Rotation matrix (R.T) used for the non-local ECP part.
        geminal_inverse (jnpt.ArrayLike):
            Precomputed inverse of the geminal matrix ``G(r_up_carts, r_dn_carts)``
            with shape ``(N_up, N_up)``.  Typically the Sherman-Morrison running
            inverse from the MCMC loop.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or
            superset) consistent with ``(r_up_carts, r_dn_carts)``.
            Forwarded to ``compute_coulomb_potential_fast`` so the non-local
            ECP ratio kernel can skip the AO/precontract recomputation.

    Returns:
        float: Local energy :math:`e_L` at the supplied configuration.

    Warning:
        ``geminal_inverse`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}``
        exactly at the supplied electron positions.  Correctness is only
        guaranteed when the inverse is maintained via **single-electron
        (rank-1) Sherman-Morrison updates** starting from a freshly
        initialized LU inverse -- the pattern used in the MCMC loop.
        Passing an inverse from a different configuration silently produces
        incorrect kinetic energy.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). Each
    # downstream consumer casts to its own zone at the use site.

    T_up_elements, T_dn_elements = compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        geminal_inverse=geminal_inverse,
    )
    T = jnp.sum(T_up_elements) + jnp.sum(T_dn_elements)

    V = compute_coulomb_potential_fast(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        A_old_inv=geminal_inverse,
        wavefunction_data=hamiltonian_data.wavefunction_data,
        det_ratio_state=det_ratio_state,
    )

    # Cast scalar zone outputs to local_energy zone at the sum (Principle 3b).
    # Promote dtype_jnp against the actual T / V dtypes so the sum preserves
    # Im on complex Bloch (twist k != 0). Bit-identical at Gamma where T and
    # V are both real.
    dtype_jnp = jnp.result_type(get_dtype_jnp("local_energy"), T.dtype, V.dtype)
    return T.astype(dtype_jnp) + V.astype(dtype_jnp)


@jit
def _compute_local_energy_auto(
    hamiltonian_data: Hamiltonian_data,
    r_up_carts: jnpt.ArrayLike,
    r_dn_carts: jnpt.ArrayLike,
    RT: jnpt.ArrayLike,
) -> float | complex:
    """Compute the local-energy reference using phase-aware kinetic autodiff.

    Args:
        hamiltonian_data (Hamiltonian_data):
            an instance of Hamiltonian_data
        r_up_carts (jnpt.ArrayLike):
            Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (jnpt.ArrayLike):
            Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (jnpt.ArrayLike):
            Rotation matrix. equiv R.T used for non-local part. It does not affect all-electron calculations.

    Returns:
        Real or complex local energy at the supplied electron coordinates.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). Each
    # downstream consumer casts to its own zone at the use site.

    T = _compute_kinetic_energy_auto(
        wavefunction_data=hamiltonian_data.wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    V = compute_coulomb_potential(
        coulomb_potential_data=hamiltonian_data.coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        wavefunction_data=hamiltonian_data.wavefunction_data,
    )

    # Cast scalar zone outputs to local_energy zone at the sum (Principle 3b).
    # Promote dtype_jnp against the actual T / V dtypes so the sum preserves
    # Im on complex Bloch (twist k != 0). Bit-identical at Gamma where T and
    # V are both real.
    dtype_jnp = jnp.result_type(get_dtype_jnp("local_energy"), T.dtype, V.dtype)
    return T.astype(dtype_jnp) + V.astype(dtype_jnp)


def _reconstruct_dataclass(cls: type[T], obj: Any) -> T:
    """Restore a dataclass instance from an object.

    Reconstructs an instance of `cls` from `obj`, handling missing/extra fields.
    This is useful for backward compatibility when loading pickled objects
    where the class definition might have changed (e.g. new fields added).

    """
    if not dataclasses.is_dataclass(cls):
        return obj

    kwargs = {}
    for field in dataclasses.fields(cls):
        field_name = field.name
        if hasattr(obj, field_name):
            val = getattr(obj, field_name)

            # Recursively reconstruct if the field type is a dataclass
            # This handles nested dataclasses that might also have changed.
            # We check if field.type is a class and is a dataclass.
            # Note: This assumes field.type is the actual class, not a string.
            if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
                val = _reconstruct_dataclass(field.type, val)

            kwargs[field_name] = val
        # If field is missing in obj, we skip it so cls() uses its default value/factory.

    return cls(**kwargs)


def _save_item(group: h5py.Group, name: str, value: Any) -> None:
    """Helper to save an item to HDF5 group."""
    if value is None:
        return

    if hasattr(value, "device_buffer") or isinstance(value, jnp.ndarray):
        value = np.array(value)

    if isinstance(value, np.ndarray):
        group.create_dataset(name, data=value)
    elif dataclasses.is_dataclass(value):
        subgroup = group.create_group(name)
        _save_dataclass_to_hdf5(subgroup, value)
    elif isinstance(value, (dict, collections.abc.Mapping)):
        subgroup = group.create_group(name)
        subgroup.attrs["_is_dict"] = True
        for k, v in value.items():
            _save_item(subgroup, str(k), v)
    elif isinstance(value, (list, tuple)):
        if len(value) > 0:
            if all(isinstance(v, (int, float, bool, str, np.number, np.bool_)) for v in value):
                group.create_dataset(name, data=value)
            elif all(isinstance(v, (np.ndarray, jnp.ndarray)) for v in value):
                subgroup = group.create_group(name)
                subgroup.attrs["_is_list"] = True
                for i, v in enumerate(value):
                    v_np = np.array(v) if hasattr(v, "device_buffer") or isinstance(v, jnp.ndarray) else v
                    subgroup.create_dataset(str(i), data=v_np)
            else:
                subgroup = group.create_group(name)
                subgroup.attrs["_is_list"] = True
                for i, v in enumerate(value):
                    _save_item(subgroup, str(i), v)
        else:
            group.create_dataset(name, data=np.array([]))
    elif isinstance(value, (int, float, bool, str)):
        group.attrs[name] = value
    else:
        try:
            group.attrs[name] = value
        except Exception:
            pass


def _detile_aos_data_for_save(
    aos_data: AOs_cart_pgto_data | AOs_sphe_pgto_data,
) -> AOs_cart_gto_data | AOs_sphe_gto_data:
    """Recover the physical home basis with its independent radial identity.

    Saving and accepted basis updates use the same structural inverse, so
    coincident radial values never merge independent optimizer coordinates.
    Load rebuilds the image envelope at the stored truncation tolerance.
    """
    if int(aos_data.num_ao) <= 0:
        raise RuntimeError("Expanded pgto leaf has num_ao <= 0 -- the pgto factory did not set this field correctly.")
    return _detile_ragged_to_molecular(aos_data)


# =============================================================================
# HDF5 Persistence: Light-Save / Load-Rebuild Contract
# =============================================================================
# The checkpoint layer maintains forward-compatibility by persisting only the
# minimal "setup-scalars" needed to reconstruct derived heavy tables at load
# time, rather than storing full pre-computed arrays.
#
# For periodic AOs_*_pgto_data:
#   - Save: de-tile to molecular GTO form (image-expansion is reversible)
#   - Persist: image_truncation_tol group attribute
#   - Load: re-expand via _apply_image_expansion_post_load with saved tol
#   - Benefit: forward-compatible across changes to image-enumeration heuristic
#
# For periodic Ewald_data:
#   - Save: persist (eta, r_real_cut, g_max), integer indices + structure_data reference
#   - Drop: real_image_list, g_vectors, g_factors, ii_energy (all derived)
#   - Load: rebuild via build_ewald_data(eta_override=saved_eta)
#   - Benefit: forward-compatible across changes to reciprocal-cutoff sweep
#
# Molecular checkpoints short-circuit the AO/Ewald logic (pbc_flag=False;
# Ewald_data=None). The pattern ensures every fixture round-trips bit-
# identically while enabling schema evolution without breaking old checkpoints.
def _strip_derived_arrays_for_save(ewald_data: Ewald_data) -> Ewald_data:
    """Strip derived arrays from ``Ewald_data`` before HDF5 serialisation.

    See module preamble for the light-save contract. Persist only the three
    setup-scalars ``(eta, r_real_cut, g_max)``, integer index tuples and the
    ``structure_data`` reference. The heavy tables ``real_image_list`` / ``g_vectors`` /
    ``g_factors`` and the precomputed ``ii_energy`` are rebuilt at load
    time via :py:func:`build_ewald_data` with ``eta_override=eta``, so the
    checkpoint stays forward-compatible across changes to the
    image-enumeration heuristic or the reciprocal-cutoff sweep.

    Molecular-context safety: ``Ewald_data`` only appears under
    ``Coulomb_potential_data`` for periodic systems; the recursive
    serializer never reaches this helper on a molecular checkpoint
    because the parent has ``ewald_data=None``.
    """
    return ewald_data.replace(
        real_image_list=None,
        g_vectors=None,
        g_factors=None,
        ii_energy=0.0,
    )


def _save_dataclass_to_hdf5(group: h5py.Group, obj: Any) -> None:
    """Recursively save a dataclass to an HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to save to.
        obj (Any): The dataclass instance to save.
    """
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Object {obj} is not a dataclass.")

    # A persisted SD is
    # exactly the occupied up/down MO rows and never a diagonal-lambda AGP
    # surrogate. Validate the actual payload, not an auxiliary format marker.
    from .determinant import SD_data
    from .orbital_tiling import TiledOrbitalSet_data
    from .orbital_tiling_hdf5 import save_tiled_orbitals

    if isinstance(obj, TiledOrbitalSet_data):
        save_tiled_orbitals(group, obj, {})
        return

    if isinstance(obj, SD_data):
        obj.sanity_check()
        if int(obj.orb_data_up_spin.num_mo) != int(obj.num_electron_up):
            raise ValueError("occupied SD HDF5 has non-canonical up-spin MO row count")
        if int(obj.orb_data_dn_spin.num_mo) != int(obj.num_electron_dn):
            raise ValueError("occupied SD HDF5 has non-canonical down-spin MO row count")

    # Periodic ``AOs_*_pgto_data`` is persisted in molecular form (de-tiled)
    # as the matching ``AOs_*_gto_data`` so the checkpoint is forward-
    # compatible across changes to the image-set heuristic. The
    # ``image_truncation_tol`` group attr lets the loader re-expand from
    # the saved molecular form via
    # :py:func:`jqmc.atomic_orbital._apply_image_expansion`.
    if isinstance(obj, (AOs_cart_pgto_data, AOs_sphe_pgto_data)):
        group.attrs["image_truncation_tol"] = float(getattr(obj, "_image_truncation_tol", AO_IMAGE_TRUNCATION_ATOL))
        obj = _detile_aos_data_for_save(obj)

    # Periodic ``Ewald_data`` keeps setup scalars and integer indices on disk;
    # heavy arrays / ii_energy are rebuilt at load time via build_ewald_data
    # with eta_override=saved_eta (see module preamble for details).
    if isinstance(obj, Ewald_data):
        obj = _strip_derived_arrays_for_save(obj)

    # Save class name for verification/reconstruction
    group.attrs["_class_name"] = obj.__class__.__name__
    group.attrs["_module_name"] = obj.__class__.__module__

    tiled_payloads = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if field.name == "ao_coefficient_scales" and not value:
            # An absent field already denotes the legacy unit-scale layout.
            continue
        if isinstance(value, TiledOrbitalSet_data):
            save_tiled_orbitals(group.create_group(field.name), value, tiled_payloads)
        else:
            _save_item(group, field.name, value)


# Returns Any: an HDF5 group may encode a Python list, a dict, or a nested
# dataclass (dispatched by the _is_list / _is_dict / _class_name attrs).
def _load_group(grp: h5py.Group) -> Any:
    """Reconstruct a list / dict / nested dataclass from an HDF5 group."""
    if grp.attrs.get("_is_list"):
        lst = []
        # Combine keys from subgroups/datasets and attributes
        all_keys = set(grp.keys())
        for k in grp.attrs.keys():
            if k.isdigit():
                all_keys.add(k)

        sorted_keys = sorted(all_keys, key=int)
        for k in sorted_keys:
            if k in grp:
                lst.append(_load_item(grp[k]))
            elif k in grp.attrs:
                lst.append(grp.attrs[k])
        return lst
    if grp.attrs.get("_is_dict"):
        return {k: _load_item(grp[k]) for k in grp.keys()}
    # Dataclass group
    class_name = grp.attrs.get("_class_name")
    module_name = grp.attrs.get("_module_name")
    if class_name and module_name:
        module = importlib.import_module(module_name)
        sub_cls = getattr(module, class_name)
        return _load_dataclass_from_hdf5(sub_cls, grp)
    # Fallback for dicts saved without _is_dict or unknown structures
    return {k: _load_item(grp[k]) for k in grp.keys()}


# Returns Any: dispatches to dataset decoding (ndarray/str) or _load_group
# (list/dict/dataclass), so the value type depends on the node.
def _load_item(item: h5py.Group | h5py.Dataset) -> Any:
    """Load an HDF5 node: dispatch to the dataset or group loader (recursive)."""
    if isinstance(item, h5py.Dataset):
        value = item[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                return np.char.decode(value, "utf-8")
            if value.dtype.kind == "O" and value.size > 0 and isinstance(value.flat[0], bytes):
                return np.array([entry.decode("utf-8") for entry in value.flat]).reshape(value.shape)
        return value
    return _load_group(item)


def _coerce_nested_tuple(val: Any) -> Any:
    """Restore an HDF5 sequence as recursively hashable tuple metadata.

    ``_save_item`` represents nested tuples as nested HDF5 groups.  Loading
    those groups can therefore produce a tuple whose children are lists or
    ndarrays.  A top-level ``tuple(...)`` is insufficient for
    ``pytree_node=False`` fields: JAX compares the complete auxiliary value
    when looking up a cached JIT, and an ndarray nested at any depth makes
    that equality ambiguous.  Canonicalise every sequence level and every
    ndarray row to tuples of Python / NumPy scalar leaves.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return tuple(_coerce_nested_tuple(item) for item in val)
    return val


def _load_dataclass_from_hdf5(cls: type[T], group: h5py.Group) -> T:
    """Recursively load a dataclass from an HDF5 group.

    Args:
        cls (Type[T]): The class to reconstruct.
        group (h5py.Group): The HDF5 group to load from.

    Returns:
        T: The reconstructed dataclass instance.
    """
    from .determinant import SD_data
    from .orbital_tiling import TiledOrbitalSet_data
    from .orbital_tiling_hdf5 import load_tiled_orbitals

    if cls is TiledOrbitalSet_data:
        return load_tiled_orbitals(group, {})

    if cls is SD_data:
        if any(name.startswith("lambda_") for name in group.keys()):
            raise ValueError("occupied SD HDF5 must not contain lambda datasets")
        providers = []
        for name in ("orb_data_up_spin", "orb_data_dn_spin"):
            if name not in group or not isinstance(group[name], h5py.Group):
                raise ValueError(f"SD HDF5 requires the {name} orbital provider")
            provider = group[name].attrs.get("_class_name")
            if provider not in ("MOs_data", "TiledOrbitalSet_data"):
                raise ValueError("SD HDF5 has an unsupported orbital provider class")
            providers.append(provider)
        if providers[0] != providers[1]:
            raise ValueError("SD HDF5 cannot mix tiled and ordinary orbital providers")

    init_args = {}
    tiled_payloads = {}

    for field in dataclasses.fields(cls):
        if field.name in group:
            node = group[field.name]
            if cls is SD_data and field.name in ("orb_data_up_spin", "orb_data_dn_spin"):
                is_tiled = node.attrs.get("_class_name") == "TiledOrbitalSet_data"
                val = load_tiled_orbitals(node, tiled_payloads) if is_tiled else _load_item(node)
            else:
                val = _load_item(node)

            # Type conversion for list/tuple/array.  Tuple-annotated fields
            # are pytree-static throughout the data model, so recursively
            # restore them to hashable tuples.  Nested HDF5 groups otherwise
            # leave inner lists / ndarrays behind (notably DTN's M_* / b_*
            # constraint tensors), which crashes JAX on the *second* JIT cache
            # lookup with ambiguous ndarray metadata equality.
            _declares_tuple = field.type is tuple or "tuple" in str(field.type)
            if isinstance(val, np.ndarray) and val.size == 0:
                if field.default_factory is list:
                    val = []
                elif field.default_factory is tuple:
                    val = ()
            elif (
                isinstance(val, np.ndarray)
                and (field.type is list or field.type is tuple or "list" in str(field.type) or "tuple" in str(field.type))
                and not ("Array" in str(field.type) or "ndarray" in str(field.type))
            ):
                val = val.tolist()
                # An ``isinstance(field.type, type)`` gate would only match
                # the literal ``tuple`` type, not generic aliases such as
                # ``tuple[int, ...]`` / ``tuple[str, ...]`` (which are
                # ``typing._GenericAlias`` instances). String-form
                # detection catches both; without this, periodic
                # ``Structure_data`` round-trips through HDF5 came back
                # as ``list`` per-field, which JAX's pytree-metadata
                # equality treats as distinct from the freshly-built
                # tuple form, silently invalidating JIT caches and
                # eventually raising ``Exception raised while checking
                # equality of metadata fields of pytree`` at the next
                # MCMC compile.
                if _declares_tuple:
                    val = _coerce_nested_tuple(val)
            elif isinstance(val, (list, tuple)) and _declares_tuple:
                val = _coerce_nested_tuple(val)

            # Convert np.ndarray or list/tuple to jax.Array for fields typed as jax.Array.
            # Note: fields typed `npt.NDArray[np.float64]` (string-form annotation) must
            # NOT trigger this branch -- they are stored as numpy arrays. Exclude both
            # "ndarray" (resolved form) and "NDArray" (npt alias form).
            if (
                isinstance(val, (np.ndarray, list, tuple))
                and "Array" in str(field.type)
                and "ndarray" not in str(field.type)
                and "NDArray" not in str(field.type)
                and "list" not in str(field.type)
                and "tuple" not in str(field.type)
            ):
                # No float64 hard-cast here, so a
                # ``complex128`` MO / lambda matrix written by a complex
                # checkpoint comes back as ``complex128``. ``jnp.asarray``
                # without ``dtype=`` follows the source dataset dtype, so
                # the real path stays bit-identical (the source was always
                # float64 in real-only inputs).
                val = jnp.asarray(val)

            init_args[field.name] = val
        elif field.name in group.attrs:
            val = group.attrs[field.name]
            # HDF5 attrs come back as numpy
            # scalars (``np.int64`` / ``np.float64`` / ``np.bool_``); for
            # pytree-static metadata fields ``pytree_node=False`` JAX's
            # cache-equality check distinguishes ``np.int64(8)`` from
            # ``int(8)`` and silently invalidates compiled jaxprs, eventually
            # raising ``Exception raised while checking equality of metadata
            # fields of pytree``. Normalise back to the Python primitive
            # when the declared field type is the primitive itself, so
            # freshly-built and HDF5-loaded instances stay pytree-identical.
            field_type = field.type
            if field_type is bool or str(field_type) == "bool":
                val = bool(val)
            elif isinstance(val, np.integer) and (field_type is int or str(field_type) == "int"):
                val = int(val)
            elif isinstance(val, np.floating) and (field_type is float or str(field_type) == "float"):
                val = float(val)
            elif isinstance(val, (bytes, np.bytes_)) and (field_type is str or str(field_type) == "str"):
                val = val.decode("utf-8")
            init_args[field.name] = val

    # Check for missing fields and fill with defaults
    for field in dataclasses.fields(cls):
        if field.name not in init_args:
            if field.default is not dataclasses.MISSING:
                init_args[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                init_args[field.name] = field.default_factory()

    obj = cls(**init_args)

    if isinstance(obj, Structure_data):
        # ``_n_max`` is derived from the lattice and owned by
        # ``Structure_data.build``; a persisted value may predate the
        # issue #67 box bound, so recompute it on every restore path
        # (Hamiltonian files, checkpoints, twist-averaged inputs).
        obj = obj._with_n_max_recomputed()

    if isinstance(obj, SD_data):
        # Full AO/MO dtype validation runs after the generic post-load
        # retwist/image-expansion fixups. At this layer enforce only the SD
        # schema invariant that is independent of those fixups.
        if int(obj.orb_data_up_spin.num_mo) != int(obj.num_electron_up):
            raise ValueError("occupied SD HDF5 has non-canonical up-spin MO row count")
        if int(obj.orb_data_dn_spin.num_mo) != int(obj.num_electron_dn):
            raise ValueError("occupied SD HDF5 has non-canonical down-spin MO row count")
        if obj.is_tiled_representation:
            obj.sanity_check()

    # PBC: stash ``image_truncation_tol`` from the group attr onto the
    # AOs_*_data instance so the post-load fixup (Hamiltonian_data.load_from_hdf5
    # -> _apply_image_expansion_post_load) can re-run
    # :py:func:`jqmc.atomic_orbital._apply_image_expansion` with the saved
    # tolerance. The attribute is non-pytree (instance attribute, not a
    # dataclass field), so it does not affect JIT semantics.
    if isinstance(obj, (AOs_cart_gto_data, AOs_sphe_gto_data)) and "image_truncation_tol" in group.attrs:
        object.__setattr__(obj, "_image_truncation_tol", float(group.attrs["image_truncation_tol"]))

    return obj


def _expand_orb_data_for_pbc(
    orb_data: AOs_base_data | MOs_data | None,
    top_structure: Structure_data,
) -> AOs_base_data | MOs_data | None:
    """Walk an orbital sub-tree (``MOs_data`` or ``AOs_*_data``) and rebuild
    every molecular GTO leaf as the matching pgto leaf via
    :py:func:`jqmc.atomic_orbital._apply_image_expansion`.

    Stashed ``_image_truncation_tol`` (set during HDF5 load via the
    ``image_truncation_tol`` group attr) is consumed here; if absent, the
    default :data:`jqmc._setting.AO_IMAGE_TRUNCATION_ATOL` is used ---
    consistent with the default in the builder. Already-expanded pgto
    leaves (loaded directly without going through the de-tile save path)
    pass through unchanged. Returns the original ``orb_data`` for
    unknown node types.
    """
    if orb_data is None:
        return None
    if isinstance(orb_data, (AOs_cart_pgto_data, AOs_sphe_pgto_data)):
        # Already a pgto leaf (e.g., loaded checkpoint that was not detiled,
        # or a freshly-built instance threaded through here). Leave it as is.
        # NOTE: we don't auto-promote a real pgto leaf to its
        # complex sibling here, because by the time we reach this branch the
        # checkpoint either (i) was detiled before save and now hits the
        # gto-leaf branch below (where the k_point-aware dispatcher fires),
        # or (ii) was saved already in pgto form (uncommon, persistence flow
        # writes the detiled form). Hitting (ii) on a non-Gamma load is an
        # explicit user error -- the loader must restart from the detiled
        # form so the complex leaf is built fresh.
        return orb_data
    if isinstance(orb_data, (AOs_cart_gto_data, AOs_sphe_gto_data)):
        tol = float(getattr(orb_data, "_image_truncation_tol", AO_IMAGE_TRUNCATION_ATOL))
        # Dispatch on the *leaf's own* ``k_point``
        # (preserved through the detile/save/load round-trip via
        # ``AOs_GTO_base_data.k_point``). Non-None -> promote to the complex
        # Bloch-pGTO leaf (carries ``_image_phase_factors``); None -> real
        # leaf (Gamma / molecular).
        if orb_data.k_point is None:
            return _apply_image_expansion(orb_data, top_structure, tol=tol)
        return _apply_image_expansion_complex(orb_data, top_structure, k_point=orb_data.k_point, tol=tol)
    if isinstance(orb_data, MOs_data):
        new_aos = _expand_orb_data_for_pbc(orb_data.aos_data, top_structure)
        return orb_data.replace(aos_data=new_aos)
    if isinstance(orb_data, AOs_composite_data):
        # Recurse into composite children so a detiled pgto child saved as
        # a molecular GTO leaf is re-expanded back to its pgto form. Other
        # leaf kinds (PW / B-spline) pass through unchanged via the default
        # branch inside this helper.
        new_children = tuple(_expand_orb_data_for_pbc(c, top_structure) for c in orb_data.children)
        return orb_data.replace(children=new_children)
    return orb_data


def _restore_shared_jastrow_after_load(jastrow: Jastrow_data) -> Jastrow_data:
    """Restore periodic J3 images in a standalone shared Jastrow checkpoint.

    The recursive writer de-tiles pGTO leaves even without a Hamiltonian
    wrapper. Restore those images before attaching the common block to any
    twist. A saved nonzero J3 twist is invalid, not a request to retwist or
    silently migrate the correlation factor. The J3 AO radial arrays are
    restored as fp64 model storage, as the Hamiltonian loaders do.
    """
    from .jastrow_factor import _validate_j3_phase_free_basis

    j3 = jastrow.jastrow_three_body_data
    if j3 is not None:
        # A detiled complex pGTO is stored as a molecular GTO with k=0.
        # Its complex MO matrix is valid only after image reconstruction,
        # so defer the full sanity check until the live AO type is restored.
        # Reject phase-bearing input before reconstruction can change it.
        _validate_j3_phase_free_basis(j3.orb_data)
        # Expansion uses the reference cell; reject mismatched children before
        # that reconstruction could normalize their serialized structures.
        Jastrow_data(jastrow_three_body_data=j3).check_structure_consistency(j3.orb_data.structure_data)
        orbital = _expand_orb_data_for_pbc(j3.orb_data, j3.orb_data.structure_data)
        # AO radial arrays are fp64 model storage in both precision modes, as
        # restored by the Hamiltonian loaders (issues #124, #136): a legacy
        # float32 shared Jastrow is widened exactly.
        orbital = _align_orb_data_basis_storage(orbital)
        jastrow = jastrow.replace(jastrow_three_body_data=j3.replace(orb_data=orbital))
    jastrow.sanity_check()
    return jastrow


def _apply_image_expansion_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Re-expand every ``AOs_*_data`` reachable under a loaded ``Hamiltonian_data``.

    Persistence layer stores ``AOs_*_data`` in molecular form (de-tiled) so
    that checkpoints stay forward-compatible. On load, every periodic
    ``AOs_*_data`` instance must be re-expanded using the saved
    ``image_truncation_tol`` group attr (stashed onto the instance during
    :py:func:`_load_dataclass_from_hdf5`); molecular Hamiltonians
    (``structure_data.pbc_flag == False``) short-circuit to a no-op.

    Walks:
      * ``ham.wavefunction_data.geminal_data.orb_data_{up,dn}_spin`` --- AO
        or MO containing the determinant / AGP basis;
      * ``ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data``
        --- AO or MO used by the three-body Jastrow factor (if present).

    The top-level ``ham.structure_data`` and
    ``ham.coulomb_potential_data.structure_data`` remain molecular --- they
    are not orbital-basis bookkeeping. The expanded inner structures are
    only carried on the ``AOs_*_data`` instances.
    """
    if not ham.structure_data.pbc_flag:
        return ham

    top_structure = ham.structure_data
    wf = ham.wavefunction_data
    if wf is None:
        return ham

    gd = wf.geminal_data
    new_orb_up = _expand_orb_data_for_pbc(gd.orb_data_up_spin, top_structure)
    new_orb_dn = _expand_orb_data_for_pbc(gd.orb_data_dn_spin, top_structure)
    new_gd = gd.replace(orb_data_up_spin=new_orb_up, orb_data_dn_spin=new_orb_dn)

    jd = wf.jastrow_data
    if jd is not None and getattr(jd, "jastrow_three_body_data", None) is not None:
        j3 = jd.jastrow_three_body_data
        new_j3_orb = _expand_orb_data_for_pbc(j3.orb_data, top_structure)
        new_j3 = j3.replace(orb_data=new_j3_orb)
        new_jd = jd.replace(jastrow_three_body_data=new_j3)
        new_wf = wf.replace(geminal_data=new_gd, jastrow_data=new_jd)
    else:
        new_wf = wf.replace(geminal_data=new_gd)

    return ham.replace(wavefunction_data=new_wf)


def _apply_ewald_rebuild_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Rebuild ``Ewald_data`` heavy tables after an HDF5 load.

    The HDF5 ``light save`` path (see module preamble and
    :py:func:`_strip_derived_arrays_for_save`) persists only the three
    setup-scalars ``(eta, r_real_cut, g_max)``, integer index tuples and the
    ``structure_data`` reference on every periodic ``Ewald_data``. The
    heavy arrays ``real_image_list`` / ``g_vectors`` / ``g_factors`` and
    the precomputed ``ii_energy`` therefore arrive as ``None`` / ``0.0``
    on load and must be reconstructed via :py:func:`build_ewald_data`
    with ``eta_override=saved_eta`` so the loaded Hamiltonian is
    immediately usable by the Coulomb dispatch shims.

    Molecular Hamiltonians (``structure_data.pbc_flag == False``)
    short-circuit to a no-op --- the ``Ewald_data`` field is ``None`` on
    every molecular instance.

    Returns the input ``ham`` unchanged when no rebuild is required.
    """
    if not ham.structure_data.pbc_flag:
        return ham
    cd = ham.coulomb_potential_data
    if cd is None or cd.ewald_data is None:
        return ham
    saved_eta = float(cd.ewald_data.eta)
    saved_r_real_cut = float(cd.ewald_data.r_real_cut)
    saved_g_max = float(cd.ewald_data.g_max)
    # HDF5 strips Cartesian tables but persists integer tuples. Preserve that
    # exact real-space topology across enumeration-policy updates. Calls on
    # an already-built Hamiltonian (e.g. after strain) still reselect tables.
    saved_real_indices = (
        cd.ewald_data.real_image_indices if cd.ewald_data.real_image_list is None and cd.ewald_data.real_image_indices else None
    )
    gd = ham.wavefunction_data.geminal_data
    n_electron = int(gd.num_electron_up) + int(gd.num_electron_dn)
    if saved_eta <= 0.0:
        # Fresh default-constructed Ewald_data (eta=0, r_real_cut=0, g_max=0)
        # --- nothing to rebuild from; fall through to the heuristic with
        # the default tol.
        new_ewald = build_ewald_data(
            ham.structure_data,
            cd,
            tol=1e-10,
            n_electron=n_electron,
        )
    else:
        # Restart path: rebuild from the saved (eta, r_real_cut, g_max)
        # triple verbatim so the image and G-vector sets reproduce the
        # original run bit-for-bit, independent of any later change to
        # the heuristic or the default tol.
        new_ewald = build_ewald_data(
            ham.structure_data,
            cd,
            eta_override=saved_eta,
            r_real_cut_override=saved_r_real_cut,
            g_max_override=saved_g_max,
            n_electron=n_electron,
            real_image_indices_override=saved_real_indices,
        )
    return ham.replace(
        coulomb_potential_data=cd.replace(ewald_data=new_ewald),
    )


def _apply_jastrow_structure_backfill_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Backfill ``Jastrow_two_body_data.structure_data`` from the top-level structure.

    ``Jastrow_two_body_data`` carries a ``structure_data`` field so the
    periodic Jastrow branch can read ``pbc_flag`` / ``cell``. Pre-PBC checkpoints have no
    such subgroup; the recursive loader fills it with the empty
    ``default_factory=Structure_data()``. This fixup substitutes the
    top-level ``Hamiltonian_data.structure_data`` whenever the loaded
    ``Jastrow_two_body_data.structure_data`` looks default.

    No-op when:
      * the top-level ``Hamiltonian_data.structure_data`` is molecular
        (``pbc_flag=False``). Molecular runs never enter the periodic
        Jastrow branch, so the empty default carried by the loaded
        ``Jastrow_two_body_data.structure_data`` is harmless and must
        be preserved bit-for-bit to keep HDF5 round-trip equality.
      * ``wavefunction_data`` or ``jastrow_data`` is missing,
      * ``jastrow_two_body_data`` is missing,
      * the loaded ``structure_data`` is already populated (a
        post-PBC checkpoint with a periodic structure).
    """
    if not ham.structure_data.pbc_flag:
        return ham
    wf = ham.wavefunction_data
    if wf is None or wf.jastrow_data is None:
        return ham
    j2 = wf.jastrow_data.jastrow_two_body_data
    if j2 is None:
        return ham
    if np.asarray(j2.structure_data.positions).size != 0 or j2.structure_data.pbc_flag:
        return ham  # already populated by a post-PBC checkpoint
    new_j2 = j2.replace(structure_data=ham.structure_data)
    new_jd = wf.jastrow_data.replace(jastrow_two_body_data=new_j2)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=new_jd))


def _align_ao_leaf_dtype(
    leaf: AOs_base_data | None,
    target_dtype: npt.DTypeLike,
) -> AOs_base_data | None:
    """Recast the radial ``exponents`` / ``coefficients`` of a GTO AO leaf
    (or any composite that wraps one) to ``target_dtype``. Non-GTO leaves
    (PW, B-spline) carry no radial leaf and are returned unchanged.
    AOs_composite_data forwards to children.

    The dtype-carrying radial arrays live on the nested
    per-shell ``radial`` leaf (:class:`AO_radial_shell_data`); the leaf-level
    ``exponents`` / ``coefficients`` are read-only derived properties
    expanded from it, so the recast must go through ``radial``.
    """
    if leaf is None:
        return leaf
    # Composite: recurse into children list.
    children = getattr(leaf, "children", None)
    if children is not None:
        new_children = [_align_ao_leaf_dtype(c, target_dtype) for c in children]
        if all(nc is c for nc, c in zip(new_children, children)):
            return leaf
        return leaf.replace(children=tuple(new_children))
    # GTO-like leaf: the radial Gaussian basis lives on the nested ``radial``
    # leaf. Non-GTO leaves (PW / B-spline) have no ``radial`` field.
    radial = getattr(leaf, "radial", None)
    if radial is None:
        return leaf
    exp_arr = getattr(radial, "exponents", None)
    coef_arr = getattr(radial, "coefficients", None)
    if exp_arr is None or coef_arr is None:
        return leaf
    if exp_arr.dtype == target_dtype and coef_arr.dtype == target_dtype:
        return leaf
    return leaf.replace(
        radial=radial.replace(
            exponents=exp_arr.astype(target_dtype),
            coefficients=coef_arr.astype(target_dtype),
        )
    )


def _align_orb_data_basis_storage(
    orb_data: AOs_base_data | MOs_data | None,
) -> AOs_base_data | MOs_data | None:
    """Recast the GTO AO radial arrays reachable from ``orb_data`` to the
    fp64 model-storage dtype (exact widening; no-op when already fp64).

    ``MOs_data`` wraps an ``AOs_*_data`` via ``.aos_data``; an AO leaf has
    the radial arrays directly; a composite recurses into its children.
    """
    # Model storage dtype, independent of the precision mode (see the
    # basis-data storage exemption in ``_precision.py``).
    target = np.dtype(np.float64)
    if orb_data is None:
        return orb_data
    inner = getattr(orb_data, "aos_data", None)
    if inner is not None:
        new_inner = _align_ao_leaf_dtype(inner, target)
        if new_inner is inner:
            return orb_data
        return orb_data.replace(aos_data=new_inner)
    return _align_ao_leaf_dtype(orb_data, target)


def _apply_ao_basis_dtype_alignment_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Re-cast every GTO AO leaf's ``exponents`` / ``coefficients`` to
    the fp64 model-storage dtype after HDF5 load.

    AO basis parameters are model storage, fp64 in both precision modes:
    TREXIO construction and
    :py:meth:`AOs_GTO_base_data.apply_basis_block_update` keep them fp64, and
    the ``_exponents_jnp`` / ``_coefficients_jnp`` accessors are lift-only
    (see the basis-data storage exemption in ``_precision.py``); the
    evaluation kernels own their ``ao_eval`` zone casts. Older checkpoints
    may carry float32 leaves (from when the block update wrote the
    ``ao_eval`` dtype); lifting them to fp64 keeps the loaded pytree dtype
    identical to an in-memory Hamiltonian, so the JIT cache is shared.

    The load must not quantize to ``ao_eval``: the Cartesian primitive
    normalization is formed from the fp64 storage exponents (issue #111),
    so a reloaded Hamiltonian would otherwise evaluate different AO values
    from the in-memory one it was saved from in mixed precision (issue #124).

    Walks: ``geminal_data.orb_data_{up,dn}_spin`` (and their inner
    ``aos_data`` if wrapped in :class:`MOs_data`) plus
    ``jastrow_three_body_data.orb_data``. Composite leaves recurse.
    No-op when every leaf is already fp64.
    """
    wf = ham.wavefunction_data
    if wf is None:
        return ham
    new_wf = wf
    if wf.geminal_data is not None:
        new_up = _align_orb_data_basis_storage(wf.geminal_data.orb_data_up_spin)
        new_dn = _align_orb_data_basis_storage(wf.geminal_data.orb_data_dn_spin)
        if (new_up is not wf.geminal_data.orb_data_up_spin) or (new_dn is not wf.geminal_data.orb_data_dn_spin):
            new_wf = new_wf.replace(
                geminal_data=wf.geminal_data.replace(
                    orb_data_up_spin=new_up,
                    orb_data_dn_spin=new_dn,
                )
            )
    if wf.jastrow_data is not None and wf.jastrow_data.jastrow_three_body_data is not None:
        j3 = wf.jastrow_data.jastrow_three_body_data
        new_orb = _align_orb_data_basis_storage(j3.orb_data)
        if new_orb is not j3.orb_data:
            new_jd = new_wf.jastrow_data.replace(
                jastrow_three_body_data=j3.replace(orb_data=new_orb),
            )
            new_wf = new_wf.replace(jastrow_data=new_jd)
    if new_wf is wf:
        return ham
    return ham.replace(wavefunction_data=new_wf)


def _apply_geminal_dtype_alignment_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Re-cast ``Geminal_data.lambda_matrix`` to the ``det_eval`` zone
    dtype after HDF5 load.

    Same rationale as :py:func:`_apply_dtn_dtype_alignment_post_load`:
    older checkpoints stored ``lambda_matrix`` as ``float64``
    unconditionally, while an alternate precision policy may assign
    ``det_eval`` another component width. Aligning load and post-SR updates
    prevents dtype-signature recompilation. Under the current full and mixed
    maps ``det_eval`` is float64, so real lambda matrices are unchanged. No-op
    when the dtype already matches the zone.

    At a non-Gamma twist the stored
    ``lambda_matrix`` is ``complex128``; the alignment target must
    preserve the complex part rather than silently downgrading to
    ``float64`` (which would drop Im and break
    ``Geminal_data.sanity_check``'s complex-leaf <-> complex-lambda
    contract). Use ``np.result_type`` so the cast promotes through the
    widest dtype that covers both the zone target and the stored lambda.
    Bit-identical on real input (real -> real ``result_type`` is the
    target itself).
    """
    from ._precision import get_dtype_np
    from .determinant import AGP_data

    wf = ham.wavefunction_data
    if wf is None or wf.geminal_data is None:
        return ham
    geminal = wf.geminal_data
    # SD_data / AGPn_data store no dense ``lambda_matrix`` field.  SD is
    # evaluated directly from its occupied MO rows; AGPn carries a real
    # ``lambda_diag``.  Neither needs this AGP-only alignment.
    if not isinstance(geminal, AGP_data):
        return ham
    target_dtype = np.result_type(get_dtype_np("det_eval"), geminal.lambda_matrix.dtype)
    if geminal.lambda_matrix.dtype == target_dtype:
        return ham
    new_geminal = geminal.replace(lambda_matrix=geminal.lambda_matrix.astype(target_dtype))
    return ham.replace(wavefunction_data=wf.replace(geminal_data=new_geminal))


def _apply_dtn_dtype_alignment_post_load(ham: "Hamiltonian_data") -> "Hamiltonian_data":
    """Re-cast ``Jastrow_DTN_data`` independent coefficients to their float64 storage dtype after HDF5 load.

    The DTN ``alpha_indep`` / ``beta_indep`` / ``gamma_indep`` are stored in
    ``jastrow_factor._JASTROW_PARAMETER_STORAGE_DTYPE`` (float64 in both modes) at
    construction and after every SR update (``apply_block_update``). A checkpoint
    written by an older mixed-precision run may hold them as float32; aligning
    the loaded dtype to the storage dtype keeps iter 1 (fresh load) and iter 2+
    (post-SR) on the same JIT cache slot (a slot change costs ~30 s of
    recompilation on B-spline pgto geminals). The loaded values are kept; no zone ever lowers the
    stored coefficients (issue #781). No-op when they are already float64 or no
    DTN block is present.
    """
    from .jastrow_factor import _JASTROW_PARAMETER_STORAGE_DTYPE

    wf = ham.wavefunction_data
    if wf is None or wf.jastrow_data is None:
        return ham
    dtn = wf.jastrow_data.jastrow_dtn_data
    if dtn is None:
        return ham
    target_dtype = np.dtype(_JASTROW_PARAMETER_STORAGE_DTYPE)
    if dtn.alpha_indep.dtype == target_dtype and dtn.beta_indep.dtype == target_dtype and dtn.gamma_indep.dtype == target_dtype:
        return ham
    # L_u / L_chi / L_f are static (float / nested tuple) and consumed with an
    # explicit dtype at eval time, so they need no cast here.
    new_dtn = dtn.replace(
        alpha_indep=np.asarray(dtn.alpha_indep).astype(target_dtype),
        beta_indep=np.asarray(dtn.beta_indep).astype(target_dtype),
        gamma_indep=np.asarray(dtn.gamma_indep).astype(target_dtype),
    )
    new_jd = wf.jastrow_data.replace(jastrow_dtn_data=new_dtn)
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=new_jd))
