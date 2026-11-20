"""Structure module.

Precision Zones:
    This module owns no Precision Zone.

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
import math
import numbers
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING

# JAX
import jax
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit
from jax import numpy as jnp
from jax.typing import DTypeLike
from numpy import linalg as LA

from ._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_TILING_FRACTIONAL_BOUNDARY_SNAP,
    DECIMAL_RECIPROCAL_CELL_DUALITY,
    MINIMUM_IMAGE_SHELL_MAX_CANDIDATES,
    RTOL_BIT_EXACT,
)

if TYPE_CHECKING:
    from .qe_wrapper import QEReference

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# separator
num_sep_line = 50


@struct.dataclass
class Structure_data:
    """Atomic structure and cell metadata.

    Stores Cartesian coordinates (Bohr), optional lattice vectors for PBC, and basic
    element metadata used by AOs/MOs, Coulomb, and Hamiltonian builders.

    Attributes:
        positions (npt.NDArray[np.float64]): Atomic Cartesian coordinates with shape ``(N, 3)`` in Bohr. dtype: float64.
        pbc_flag (bool): Whether periodic boundary conditions are active. If ``True``, lattice
            vectors ``vec_a|b|c`` must be provided; otherwise they must be empty.
        vec_a (list[float] | tuple[float]): Lattice vector **a** (Bohr) when ``pbc_flag=True``.
        vec_b (list[float] | tuple[float]): Lattice vector **b** (Bohr) when ``pbc_flag=True``.
        vec_c (list[float] | tuple[float]): Lattice vector **c** (Bohr) when ``pbc_flag=True``.
        atomic_numbers (list[int] | tuple[int]): Atomic numbers ``Z`` for each site (len ``N``).
        element_symbols (list[str] | tuple[str]): Element symbols for each site (len ``N``).
        atomic_labels (list[str] | tuple[str]): Human-readable labels for each site (len ``N``).

    Examples:
        Minimal H2 setup (Bohr)::

            import numpy as np
            from jqmc import Structure_data

            structure = Structure_data(
                positions=np.array([[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]]),
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )
            structure.sanity_check()
    """

    #: Atomic Cartesian coordinates with shape ``(N, 3)`` in Bohr. dtype: float64.
    positions: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64))
    #: Whether periodic boundary conditions are active.
    pbc_flag: bool = struct.field(pytree_node=False, default=False)
    #: Lattice vector **a** in Bohr (requires ``pbc_flag=True``).
    #:
    #: ``vec_{a,b,c}`` are ``pytree_node=True`` ndarrays rather than
    #: ``pytree_node=False`` static tuples, so the lattice cell is a
    #: differentiable leaf and stress AD
    #: (``jacrev(local_energy)`` w.r.t. cell) gets a non-NaN cotangent.
    #: The ``__post_init__`` normalizer accepts list / tuple / ndarray
    #: inputs and stores the canonical numpy form, so existing kwarg
    #: literals (``vec_a=(10.0, 0.0, 0.0)``) continue to work without
    #: surgery at every call site.
    vec_a: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0,), dtype=np.float64))
    #: Lattice vector **b** in Bohr (requires ``pbc_flag=True``). See :py:attr:`vec_a`.
    vec_b: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0,), dtype=np.float64))
    #: Lattice vector **c** in Bohr (requires ``pbc_flag=True``). See :py:attr:`vec_a`.
    vec_c: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0,), dtype=np.float64))
    #: Atomic numbers ``Z`` per site (len ``N``).
    atomic_numbers: list[int] | tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Element symbols per site (len ``N``).
    element_symbols: list[str] | tuple[str, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Human-readable labels per site (len ``N``).
    atomic_labels: list[str] | tuple[str, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Half-width of the lattice-image search box for minimum-image enumeration.
    #: Static (``pytree_node=False``) so JIT-compiled MIC helpers can consume it
    #: as a compile-time constant. Computed from the cell vectors by
    #: :py:meth:`build` (via :py:meth:`_compute_n_max`) when ``pbc_flag=True``;
    #: the molecular default ``1`` is used otherwise. Depends only on the
    #: lattice (not on positions), so ``Structure_data.replace(positions=...)``
    #: leaves ``_n_max`` invariant and the JIT cache is not invalidated.
    #: ``sanity_check`` rejects a periodic instance left at the molecular
    #: default (``_n_max == 1``), which signals a bypass of ``build``.
    _n_max: int = struct.field(pytree_node=False, default=1)

    def __post_init__(self) -> None:
        """Normalize lattice vectors to ndarray (does NOT populate ``_n_max``).

        Sole responsibility: coerce ``vec_a/b/c`` to ``np.ndarray`` of dtype
        float64. The declared field type is a
        ``pytree_node=True`` ndarray, but call sites may pass list /
        tuple literals (``vec_a=(10.0, 0.0, 0.0)``). The normalizer coerces
        both forms to the canonical ndarray, keeping the kwarg API stable
        while enabling stress AD through the cell pytree leaf.

        Derived static state (``_n_max``) is **not** computed here. It is
        owned exclusively by :py:meth:`build` (via :py:meth:`_compute_n_max`),
        so the JAX tracer (``replace(vec_a=...)`` under ``jacrev``) and the
        pytree SENTINEL ``object()`` leaf (``vmap`` ``in_axes`` flattening)
        flow through ``__init__`` / ``__post_init__`` untouched -- no
        defensive tracer/sentinel guard is needed. A periodic structure
        constructed via ``build`` carries the correct ``_n_max >= 2``;
        :py:meth:`replace` preserves it (``_n_max`` is ``pytree_node=False``
        and invariant under small strain); the HDF5 loader recomputes it from
        the restored lattice (:py:meth:`_with_n_max_recomputed`).
        :py:meth:`sanity_check` rejects a periodic structure
        left at the molecular default ``_n_max == 1`` (the unambiguous
        signature of a construction path that bypassed ``build``).
        """
        # Normalize lattice vectors to ndarray. Bypasses the frozen-
        # dataclass setter via ``object.__setattr__``. JAX tracers and
        # existing ndarrays pass through unchanged --- only list / tuple
        # literals (from kwarg construction) are coerced. Coercing a tracer
        # here would break ``replace(vec_a=jnp_traced)`` under ``jacrev``.
        for name in ("vec_a", "vec_b", "vec_c"):
            val = getattr(self, name)
            if isinstance(val, (list, tuple)):
                object.__setattr__(self, name, np.asarray(val, dtype=np.float64))
            elif isinstance(val, np.ndarray) and val.dtype != np.float64:
                object.__setattr__(self, name, val.astype(np.float64))

    @staticmethod
    def _compute_n_max(
        vec_a: npt.NDArray[np.float64],
        vec_b: npt.NDArray[np.float64],
        vec_c: npt.NDArray[np.float64],
    ) -> int:
        r"""Half-width of the lattice-image search box from the cell vectors.

        Returns ``max(n_legacy, n_proof)``. ``n_proof`` makes the box of
        :py:func:`_minimum_image_displacement` (parallelepiped wrap in the
        given basis, then the ``(2 n_max + 1)^3`` candidates
        ``|n_k| <= n_max``) provably contain every minimum image, for any
        cell basis of the lattice (issue #67). Write the wrapped displacement
        as ``d_w = f @ cell`` with ``|f_k| <= 1/2`` and the minimum image as
        ``d* = d_w + n @ cell``. With the reciprocal columns
        ``g_k = inv(cell)[:, k]`` (``cell @ g_k = e_k``),

        .. math::
           n_k = d^* \cdot g_k - f_k, \qquad
           |n_k| \le \rho\, |g_k| + \tfrac12,

        where :math:`\rho \ge |d^*|` bounds the covering radius of the
        lattice. Babai's nearest-plane bound gives
        :math:`\rho = \tfrac12 \sqrt{\sum_i |b_i^*|^2}` for the Gram-Schmidt
        vectors :math:`b_i^*` of *any* basis of the lattice; the LLL-reduced
        basis of :py:func:`_lll_reduce_basis` keeps it small. Hence
        ``n_proof = max_k ceil(rho |g_k| + 1/2)``. The bound depends on the
        input basis only through ``|g_k|``, so a non-reduced basis gets a
        wider (but still finite and sufficient) box.

        ``n_legacy`` is the earlier heuristic

        .. math::
           n_\text{legacy} = \left\lceil \frac{0.5 \, \|T_a + T_b + T_c\|}{L_\text{min}} \right\rceil + 1,

        where :math:`L_\text{min} = V / \max_i |T_j \times T_k|` is the cell
        inradius (same definition as :py:meth:`cell_inradius`). It is kept as
        a floor so every cell on which it already sufficed keeps its static
        box (and its JIT shapes and persisted value) unchanged; on such cells
        the selected minimum image is the same. It is not sufficient on its
        own: it only looks at the ``a + b + c`` body diagonal, while the
        wrapped displacement can reach half the longest of the four body
        diagonals ``|+-a +-b +-c|``.

        Inputs must be concrete ``(3,)`` float ndarrays so :py:meth:`build`
        can compute the static value before constructing an instance.

        Args:
            vec_a: Lattice vector **a** as a ``(3,)`` float ndarray (Bohr).
            vec_b: Lattice vector **b** as a ``(3,)`` float ndarray (Bohr).
            vec_c: Lattice vector **c** as a ``(3,)`` float ndarray (Bohr).

        Returns:
            int: The image-box half-width ``n_max``.
        """
        va = np.asarray(vec_a, dtype=np.float64)
        vb = np.asarray(vec_b, dtype=np.float64)
        vc = np.asarray(vec_c, dtype=np.float64)
        half_diag = 0.5 * float(np.linalg.norm(va + vb + vc))
        volume = float(np.abs(np.dot(va, np.cross(vb, vc))))
        face_areas = np.array(
            [
                np.linalg.norm(np.cross(vb, vc)),
                np.linalg.norm(np.cross(vc, va)),
                np.linalg.norm(np.cross(va, vb)),
            ],
            dtype=np.float64,
        )
        inradius = float(volume / np.max(face_areas))
        n_legacy = int(np.ceil(half_diag / inradius)) + 1

        cell = np.stack([va, vb, vc])
        reduced = _lll_reduce_basis(cell)
        gram_schmidt = _gram_schmidt_rows(reduced)
        covering_radius_bound = 0.5 * float(np.sqrt(np.sum(gram_schmidt * gram_schmidt)))
        dual_norms = np.linalg.norm(np.linalg.inv(cell), axis=0)  # |g_k| = |inv(cell)[:, k]|
        n_proof = int(np.max(np.ceil(covering_radius_bound * dual_norms + 0.5)))
        return max(n_legacy, n_proof)

    def _wigner_seitz_inscribed_radius(self) -> float:
        """Radius of the largest sphere inscribed in the Wigner-Seitz cell.

        The WS cell is bounded by the perpendicular bisector planes of the
        lattice vectors, so its inscribed radius is half the shortest non-zero
        lattice vector, ``min_{T != 0} |T| / 2``. It is a property of the
        lattice alone: the same lattice written in any cell basis gives the
        same value (issue #68). Computed by :py:func:`_shortest_lattice_vector_length`
        (LLL reduction plus a provably sufficient integer search).

        Every displacement shorter than this radius is the unique minimum
        image of its lattice class, so a minimum-image distance below it is
        smooth. Consumers that select the image must therefore recover the
        true minimum image up to this radius; for the DTN Jastrow that is
        :py:func:`_minimum_image_shell_half_widths` on top of the
        parallelepiped wrap (the wrap alone is exact only up to half the
        smallest interplanar spacing of the given basis, which is smaller on
        oblique cells).

        For an orthorhombic cell (all rows exactly orthogonal,
        :py:func:`_orthogonal_axes`) the shortest lattice vector is the
        shortest row, which equals the smallest interplanar spacing; that
        branch keeps the expression ``0.5 / max_j |g_j|`` with
        ``g_j = inv_cell[:, j]`` (``a / 2`` for a cubic cell of side ``a``),
        so orthorhombic results are bit-identical to earlier releases.

        Returns ``+inf`` for a molecular (non-PBC) structure so callers can
        apply the bound unconditionally.
        """
        if not self.pbc_flag:
            return float("inf")
        cell = np.asarray([self.vec_a, self.vec_b, self.vec_c], dtype=np.float64)
        if all(_orthogonal_axes(cell)):
            inv_cell = np.linalg.inv(cell)
            recip_norms = np.linalg.norm(inv_cell, axis=0)  # |inv_cell[:, j]| = |g_j|
            return float(0.5 / np.max(recip_norms))
        return 0.5 * _shortest_lattice_vector_length(cell)

    def _median_nn_distance(self) -> float | None:
        """Median nearest-neighbour ion-ion distance (minimum-image under PBC).

        Returns ``None`` for a single-atom / no-pair geometry. Supercell-size
        independent (tiling the cell leaves it unchanged) yet density-adaptive.
        """
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.shape[0] < 2:
            return None
        diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
        if self.pbc_flag:
            cell = np.asarray([self.vec_a, self.vec_b, self.vec_c], dtype=np.float64)
            inv_cell = np.linalg.inv(cell)
            flat = diff.reshape(-1, 3)
            diff = np.stack([_minimum_image_displacement_debug(d, cell, inv_cell, self._n_max) for d in flat]).reshape(
                diff.shape
            )
        d2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        return float(np.median(nn))

    @classmethod
    def build(
        cls,
        *,
        positions: npt.NDArray[np.float64] | list | tuple = (),
        pbc_flag: bool = False,
        vec_a: npt.NDArray[np.float64] | list | tuple = (),
        vec_b: npt.NDArray[np.float64] | list | tuple = (),
        vec_c: npt.NDArray[np.float64] | list | tuple = (),
        atomic_numbers: list[int] | tuple[int, ...] = (),
        element_symbols: list[str] | tuple[str, ...] = (),
        atomic_labels: list[str] | tuple[str, ...] = (),
    ) -> "Structure_data":
        """Construct a validated Structure_data with derived static fields computed.

        The blessed factory: it normalizes inputs (list / tuple -> float64
        ndarray for the cell and positions; sequences -> tuple for the
        per-atom metadata), computes the derived static field ``_n_max``
        (via :py:meth:`_compute_n_max` when ``pbc_flag=True`` and the lattice
        is well-formed), constructs the instance, and runs
        :py:meth:`sanity_check`. ``build`` exists so derived state has a single
        explicit owner and so the construction contract is uniform across all
        dataclasses.

        Direct construction (``Structure_data(...)``) is still allowed, but
        ``__post_init__`` only normalizes ``vec_a/b/c`` -- it does **not**
        compute ``_n_max``. A caller taking that route must pass ``_n_max``
        itself (the TREXIO loader passes :py:meth:`_compute_n_max`; the HDF5
        loader recomputes it via :py:meth:`_with_n_max_recomputed`); otherwise a periodic instance
        is left at the molecular default ``_n_max == 1``, which
        :py:meth:`sanity_check` rejects.

        Args:
            positions: Atomic Cartesian coordinates ``(N, 3)`` in Bohr.
            pbc_flag: Whether periodic boundary conditions are active.
            vec_a: Lattice vector **a** (Bohr); required when ``pbc_flag=True``.
            vec_b: Lattice vector **b** (Bohr); required when ``pbc_flag=True``.
            vec_c: Lattice vector **c** (Bohr); required when ``pbc_flag=True``.
            atomic_numbers: Atomic numbers ``Z`` per site (len ``N``).
            element_symbols: Element symbols per site (len ``N``).
            atomic_labels: Human-readable labels per site (len ``N``).

        Returns:
            Structure_data: The validated instance.
        """
        va = np.asarray(vec_a, dtype=np.float64)
        vb = np.asarray(vec_b, dtype=np.float64)
        vc = np.asarray(vec_c, dtype=np.float64)
        pos = np.asarray(positions, dtype=np.float64)
        n_max = 1
        if pbc_flag and va.shape == (3,) and vb.shape == (3,) and vc.shape == (3,):
            n_max = cls._compute_n_max(va, vb, vc)
        data = cls(
            positions=pos,
            pbc_flag=pbc_flag,
            vec_a=va,
            vec_b=vb,
            vec_c=vc,
            atomic_numbers=tuple(atomic_numbers),
            element_symbols=tuple(element_symbols),
            atomic_labels=tuple(atomic_labels),
            _n_max=n_max,
        )
        data.sanity_check()
        return data

    def _with_n_max_recomputed(self) -> "Structure_data":
        """Return this structure with ``_n_max`` recomputed from its lattice vectors.

        Restore hook for every HDF5 / checkpoint load (called by
        :py:func:`jqmc.hamiltonians._load_dataclass_from_hdf5` on each
        ``Structure_data`` group it reconstructs). ``_n_max`` is derived
        state owned by :py:meth:`build`; a persisted value is not trusted,
        because a file written before issue #67 may carry the old
        ``a + b + c``-only box, which misses minimum images on a non-reduced
        cell basis. The value is therefore recomputed with
        :py:meth:`_compute_n_max` under exactly the condition :py:meth:`build`
        uses (``pbc_flag=True`` and three ``(3,)`` lattice vectors), so a
        loaded structure is identical to a freshly built one.

        When the recomputed value equals the persisted one (every file
        written since issue #67, and every cell on which the old box already
        sufficed, e.g. orthogonal or reduced cells) ``self`` is returned
        unchanged. Otherwise a warning reports the change and a replaced
        instance is returned. Molecular / malformed structures are returned
        unchanged (``sanity_check`` keeps rejecting the latter).

        Returns:
            Structure_data: ``self`` or a copy with the recomputed ``_n_max``.
        """
        if not self.pbc_flag:
            return self
        va = np.asarray(self.vec_a, dtype=np.float64)
        vb = np.asarray(self.vec_b, dtype=np.float64)
        vc = np.asarray(self.vec_c, dtype=np.float64)
        if not (va.shape == (3,) and vb.shape == (3,) and vc.shape == (3,)):
            return self
        n_max = int(self._compute_n_max(va, vb, vc))
        if n_max == self._n_max:
            return self
        logger.warning(
            f"Structure_data restored from HDF5 carried _n_max = {self._n_max}; recomputed _n_max = {n_max} "
            "from the lattice vectors (the persisted minimum-image box comes from an older jQMC version and can miss "
            "minimum images on a non-reduced cell basis)."
        )
        return self.replace(_n_max=n_max)

    @classmethod
    def from_qe_reference(cls, qe_reference: "QEReference") -> "Structure_data":
        """Map a QE-projection reference's cell + atoms to a periodic Structure_data.

        QE stores lengths in Bohr (atomic units), which is also the jqmc convention,
        so the cell rows are copied verbatim. ``element_symbols`` come from
        ``qe_reference.atom_elements`` (each atom's element resolved from its UPF
        PP_HEADER, QE's own convention), while ``atomic_labels`` keep the raw QE
        species labels ('O' vs 'O1'); atomic numbers are mapped from the elements
        via :func:`jqmc.trexio_wrapper._convert_from_atomic_labels_to_atomic_numbers`.

        This factory carries no notion of k (k lives on the
        AO leaf, not the structure); callers that need the twist should extract it
        from ``qe_reference`` separately.

        Args:
            qe_reference: A :class:`jqmc.qe_wrapper.QEReference` instance.

        Returns:
            Structure_data: The validated instance.
        """
        # Deferred import: trexio_wrapper imports Structure_data at module
        # scope, so importing it here at module scope would be circular.
        from .trexio_wrapper import _convert_from_atomic_labels_to_atomic_numbers

        cell = qe_reference.cell
        labels = list(qe_reference.atom_labels)
        # Element identity is the UPF-resolved symbol (QE's own convention),
        # kept distinct from the possibly-tagged QE label so an AFM setup that
        # writes 'O' / 'O1' for opposite spins round-trips: element_symbols
        # ('O') feed Z, atomic_labels ('O1') stay the per-species pseudo key.
        elements = list(qe_reference.atom_elements)
        if len(elements) != len(labels):
            raise ValueError(
                "QEReference.atom_elements is empty or length-mismatched; the QE path "
                "resolves each atom's element from its UPF PP_HEADER (see "
                "jqmc.qe_wrapper._resolve_atom_elements). Load the reference via "
                "load_qe_reference so this is populated."
            )
        z_list = _convert_from_atomic_labels_to_atomic_numbers(elements)
        structure_data = cls.build(
            positions=np.asarray(qe_reference.atom_positions, dtype=np.float64),
            pbc_flag=True,
            vec_a=tuple(float(x) for x in cell.a[0]),
            vec_b=tuple(float(x) for x in cell.a[1]),
            vec_c=tuple(float(x) for x in cell.a[2]),
            atomic_numbers=tuple(int(z) for z in z_list),
            element_symbols=tuple(elements),
            atomic_labels=tuple(labels),
        )
        structure_data.sanity_check()
        return structure_data

    def sanity_check(self) -> None:
        """Validate consistency of positions, labels, and lattice metadata.

        Ensures all per-atom arrays share length ``N``, lattice vectors are either all
        present (length 3 each) when ``pbc_flag=True`` or all empty when ``pbc_flag=False``,
        and basic types are as expected.

        Raises:
            ValueError: If list lengths mismatch, lattice vectors are inconsistent with
                ``pbc_flag``, or field types are incorrect.
        """
        if len(self.element_symbols) != len(self.atomic_numbers):
            raise ValueError("The length of element_symbols and atomic_numbers must be the same.")
        if len(self.atomic_labels) != len(self.atomic_numbers):
            raise ValueError("The length of atomic_labels and atomic_numbers must be the same.")
        if len(self.positions) != len(self.atomic_numbers):
            raise ValueError("The length of positions and atomic_numbers must be the same.")
        if not isinstance(self.pbc_flag, bool):
            raise ValueError(f"pbc_flag = {type(self.pbc_flag)} must be a boolean.")
        if self.pbc_flag:
            if len(self.vec_a) != 3 or len(self.vec_b) != 3 or len(self.vec_c) != 3:
                raise ValueError("The length of lattice vectors must be 3.")
            # Guardrail: a periodic structure must carry a computed image-box
            # half-width. The minimum-image formula
            # ``n_max = ceil(0.5 |T_a+T_b+T_c| / L_min) + 1`` is >= 2 for any
            # real cell (ceil of a positive ratio is >= 1, plus the +1
            # margin), so ``_n_max == 1`` is the unambiguous signature of a
            # construction path that bypassed ``build`` (the molecular
            # default leaked into a periodic instance). Reject it here so the
            # failure is loud instead of a silently undersized minimum-image
            # search box.
            if self._n_max < 2:
                raise ValueError(
                    f"pbc_flag=True but _n_max = {self._n_max} (< 2). A periodic Structure_data "
                    "must be created via Structure_data.build(...), which computes _n_max from the "
                    "cell. Direct construction (Structure_data(...)) leaves _n_max at the molecular "
                    "default and undersizes the minimum-image search box."
                )
        else:
            if len(self.vec_a) != 0 or len(self.vec_b) != 0 or len(self.vec_c) != 0:
                raise ValueError("The lattice vectors must be empty.")

        # Lattice vectors are normalized to ndarray in
        # __post_init__; accept any of the three forms (the post_init makes
        # them ndarray, but a user-constructed instance might still come
        # in fresh).
        if not isinstance(self.vec_a, (list, tuple, np.ndarray)):
            raise ValueError(f"vec_a = {type(self.vec_a)} must be a list, tuple, or ndarray.")
        if not isinstance(self.vec_b, (list, tuple, np.ndarray)):
            raise ValueError(f"vec_b = {type(self.vec_b)} must be a list, tuple, or ndarray.")
        if not isinstance(self.vec_c, (list, tuple, np.ndarray)):
            raise ValueError(f"vec_c = {type(self.vec_c)} must be a list, tuple, or ndarray.")
        if not isinstance(self.atomic_numbers, (list, tuple)):
            raise ValueError(f"atomic_numbers = {type(self.atomic_numbers)} must be a list or tuple.")
        if not isinstance(self.element_symbols, (list, tuple)):
            raise ValueError(f"element_symbols = {type(self.element_symbols)} must be a list or tuple.")
        if not isinstance(self.atomic_labels, (list, tuple)):
            raise ValueError(f"atomic_labels = {type(self.atomic_labels)} must be a list or tuple.")

    def _get_info(self) -> list[str]:
        """Return human-readable summary lines for logging."""
        info_lines: list[str] = ["**" + self.__class__.__name__, f"  PBC flag = {self.pbc_flag}"]
        if self.pbc_flag:
            info_lines.append(f"  vec A = {self.vec_a} Bohr")
            info_lines.append(f"  vec B = {self.vec_b} Bohr")
            info_lines.append(f"  vec C = {self.vec_c} Bohr")
        info_lines.append("  " + "-" * num_sep_line)
        info_lines.append("  element, label, Z, x, y, z in cartesian (Bohr)")
        info_lines.append("  " + "-" * num_sep_line)
        for atomic_number, element_symbol, atomic_label, position in zip(
            self.atomic_numbers, self.element_symbols, self.atomic_labels, self._positions_cart_np, strict=True
        ):
            info_lines.append(
                f"  {element_symbol:s}, {atomic_label:s}, {atomic_number:.1f}, "
                f"{position[0]:.8f}, {position[1]:.8f}, {position[2]:.8f}"
            )
        info_lines.append("  " + "-" * num_sep_line)
        return info_lines

    def _logger_info(self) -> None:
        """Log the information obtained from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def cell(self) -> npt.NDArray[np.float64]:
        """Lattice vectors as a ``(3, 3)`` matrix in Bohr (``[a, b, c]``)."""
        dtype_np = np.float64
        cell = np.array([self.vec_a, self.vec_b, self.vec_c], dtype=dtype_np)
        return cell

    @property
    def _cell_np(self) -> npt.NDArray[np.float64]:
        """Numpy alias of :py:attr:`cell`.

        Same shape ``(3, 3)`` as :py:attr:`cell`; introduced for symmetry
        with :py:attr:`_cell_jnp` so callers can use a single name regardless
        of fp / numpy preference.
        """
        return self.cell

    @property
    def _cell_jnp(self) -> jax.Array:
        """JAX view of :py:attr:`cell`; AD-active under stress.

        Returns a ``(3, 3)`` JAX array stacked from ``vec_{a,b,c}``. Because
        those fields are ``pytree_node=True`` ndarrays, the
        ``cell``-rank cotangent flows through this property to downstream
        consumers (``T_n = m @ cell`` for pgto image translations,
        ``G_m = m @ (2*pi*inv(cell).T)`` for plane-wave reciprocal vectors,
        ``frac = r @ inv(cell)`` for the B-spline metric, and Ewald
        rebuilds).
        """
        return jnp.stack(
            [
                jnp.asarray(self.vec_a, dtype=jnp.float64),
                jnp.asarray(self.vec_b, dtype=jnp.float64),
                jnp.asarray(self.vec_c, dtype=jnp.float64),
            ],
            axis=0,
        )

    @property
    def recip_cell(self) -> npt.NDArray[np.float64]:
        r"""Reciprocal lattice vectors ``(3, 3)`` in Bohr^{-1}.

        Uses the standard definition

        .. math::
           G_a = 2\pi \frac{T_b \times T_c}{T_a \cdot (T_b \times T_c)},
           \quad G_b = 2\pi \frac{T_c \times T_a}{T_b \cdot (T_c \times T_a)},
           \quad G_c = 2\pi \frac{T_a \times T_b}{T_c \cdot (T_a \times T_b)},

        and asserts the orthonormality condition :math:`T_i \cdot G_j = 2\pi\,\delta_{ij}`.
        """
        dtype_np = np.float64
        va = np.asarray(self.vec_a, dtype=dtype_np)
        vb = np.asarray(self.vec_b, dtype=dtype_np)
        vc = np.asarray(self.vec_c, dtype=dtype_np)
        recip_a = 2 * np.pi * (np.cross(vb, vc)) / (np.dot(va, np.cross(vb, vc)))
        recip_b = 2 * np.pi * (np.cross(vc, va)) / (np.dot(vb, np.cross(vc, va)))
        recip_c = 2 * np.pi * (np.cross(va, vb)) / (np.dot(vc, np.cross(va, vb)))

        # check if the implementations are correct
        lattice_vec_list = [va, vb, vc]
        recip_vec_list = [recip_a, recip_b, recip_c]
        for (lattice_vec_i, lattice_vec), (recip_vec_j, recip_vec) in itertools.product(
            enumerate(lattice_vec_list), enumerate(recip_vec_list)
        ):
            if lattice_vec_i == recip_vec_j:
                np.testing.assert_almost_equal(
                    np.dot(lattice_vec, recip_vec), 2 * np.pi, decimal=DECIMAL_RECIPROCAL_CELL_DUALITY
                )
            else:
                np.testing.assert_almost_equal(np.dot(lattice_vec, recip_vec), 0.0, decimal=DECIMAL_RECIPROCAL_CELL_DUALITY)

        recip_cell = np.array([recip_a, recip_b, recip_c], dtype=dtype_np)
        return recip_cell

    @property
    def lattice_vec_a(self) -> tuple[np.float64, ...]:
        """Return lattice vector A (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector A (in Bohr).

        """
        return tuple(self.cell[0])

    @property
    def lattice_vec_b(self) -> tuple[np.float64, ...]:
        """Return lattice vector B (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector B (in Bohr).

        """
        return tuple(self.cell[1])

    @property
    def lattice_vec_c(self) -> tuple[np.float64, ...]:
        """Return lattice vector C (in Bohr).

        Returns:
            tuple[np.float64]: the lattice vector C (in Bohr).

        """
        return tuple(self.cell[2])

    @property
    def recip_vec_a(self) -> tuple[np.float64, ...]:
        """Return reciprocal lattice vector A (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector A (in Bohr).

        """
        return tuple(self.recip_cell[0])

    @property
    def recip_vec_b(self) -> tuple[np.float64, ...]:
        """Return reciprocal lattice vector B (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector B (in Bohr).

        """
        return tuple(self.recip_cell[1])

    @property
    def recip_vec_c(self) -> tuple[np.float64, ...]:
        """Return reciprocal lattice vector C (in Bohr).

        Returns:
            tuple[np.float64]: the reciprocal lattice vector C (in Bohr).

        """
        return tuple(self.recip_cell[2])

    @property
    def norm_vec_a(self) -> np.float64:
        """Return the norm of the lattice vector A (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector A (in Bohr).

        """
        return LA.norm(self.vec_a)

    @property
    def norm_vec_b(self) -> np.float64:
        """Return the norm of the lattice vector B (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector C (in Bohr).

        """
        return LA.norm(self.vec_b)

    @property
    def norm_vec_c(self) -> np.float64:
        """Return the norm of the lattice vector C (in Bohr).

        Returns:
            np.float64: the norm of the lattice vector C (in Bohr).

        """
        return LA.norm(self.vec_c)

    @property
    def _positions_cart_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions as ``numpy.ndarray`` with shape ``(N, 3)`` in Bohr."""
        dtype_np = np.float64
        return np.array(self.positions, dtype=dtype_np)

    @property
    def _positions_cart_jnp(self) -> jax.Array:
        """Atomic positions as ``jax.Array`` with shape ``(N, 3)`` in Bohr."""
        dtype_jnp = jnp.float64
        return jnp.array(self.positions, dtype=dtype_jnp)

    @property
    def _positions_frac_np(self) -> npt.NDArray[np.float64]:
        """Fractional (crystal) coordinates as ``numpy.ndarray`` with shape ``(N, 3)``."""
        dtype_np = np.float64
        h = np.array([self.vec_a, self.vec_b, self.vec_c], dtype=dtype_np)
        positions_frac = np.array(
            [np.dot(np.asarray(pos, dtype=dtype_np), np.linalg.inv(h)) for pos in self._positions_cart_np],
            dtype=dtype_np,
        )
        return positions_frac

    @property
    def natom(self) -> int:
        """The number of atoms in the system.

        Returns:
            int:The number of atoms in the system.
        """
        return len(self.atomic_numbers)

    @property
    def ntyp(self) -> int:
        """The number of element types in the system.

        Returns:
            int: The number of element types in the system.
        """
        return len(list(set(self.atomic_numbers)))

    def cell_inradius(self) -> float:
        r"""Perpendicular distance between opposing parallelepiped faces (minimum).

        Returns:

        .. math::
           L_\text{min} = \min_i \frac{V}{|T_j \times T_k|},

        where :math:`(i, j, k)` runs over the three cyclic pairs and
        :math:`V = T_a \cdot (T_b \times T_c)` is the cell volume in Bohr\ :sup:`3`.
        This is the inradius of the parallelepiped (the largest sphere that fits
        inside the home cell).

        Depends only on the lattice vectors ``vec_a/b/c`` --- not on atomic
        positions --- so the value is stable across
        :py:meth:`Structure_data.replace` updates that change ``positions``.
        Sizes the minimum-image search box (``_n_max``) and the pGTO image
        enumeration bounding box.

        Returns:
            float: Cell inradius in Bohr. Raises ``ValueError`` for non-PBC structures.
        """
        if not self.pbc_flag:
            raise ValueError("cell_inradius is defined only for pbc_flag=True structures.")
        cell = self.cell
        volume = float(np.abs(np.dot(cell[0], np.cross(cell[1], cell[2]))))
        face_areas = np.array(
            [
                np.linalg.norm(np.cross(cell[1], cell[2])),
                np.linalg.norm(np.cross(cell[2], cell[0])),
                np.linalg.norm(np.cross(cell[0], cell[1])),
            ],
            dtype=np.float64,
        )
        return float(volume / np.max(face_areas))

    def primitive_image_list(
        self,
        exponent: float,
        angular_L: int,
        tol: float = AO_IMAGE_TRUNCATION_ATOL,
    ) -> npt.NDArray[np.float64]:
        r"""Lattice translations needed to periodise a primitive Gaussian.

        For a primitive :math:`g_p(r) = (r - R_\mu)^{\alpha} \exp(-\alpha_p |r - R_\mu|^2)`
        centred at :math:`R_\mu \in \Omega`, returns the set of integer lattice
        translations :math:`\{T_n\}` whose image copy contributes above ``tol``
        to the value at any point in the home cell :math:`\Omega`.

        The envelope :math:`r^{L_p} \exp(-\alpha_p r^2)` peaks at
        :math:`r_\text{peak} = \sqrt{L_p / (2 \alpha_p)}` and falls below
        ``tol`` outside the radius

        .. math::
           r_\text{cut} =
             \sqrt{\frac{\max(-\ln\text{tol}, 1)}{\alpha_p}} +
             r_\text{peak}\, \max\!\left(1, \frac{\ln(1/\text{tol})}{\max(L_p, 1)}\right),

        where the ``max(L_p, 1)`` guard keeps the s-orbital branch finite
        (:math:`r_\text{peak} = 0`, second term vanishes,
        :math:`r_\text{cut} = \sqrt{-\ln\text{tol} / \alpha_p}`).

        A translation :math:`T_n` is retained when its closest approach to the
        Minkowski sum :math:`\Omega - \Omega = \{\alpha T_a + \beta T_b + \gamma T_c :
        |\alpha|, |\beta|, |\gamma| \le 1\}` is at most :math:`r_\text{cut}`,
        i.e., the fractional offset of :math:`T_n` beyond the
        :math:`(-1, 1)^3` box, mapped back to Cartesian, has norm
        :math:`\le r_\text{cut}`. The criterion depends only on
        :math:`(\alpha_p, L_p, \text{tol})` and the lattice --- **not** on
        :math:`R_\mu` --- so the same list is reused across every primitive
        sharing :math:`(\alpha_p, L_p)`.

        Runs once at setup; numpy / fp64 throughout. Not a hot-path call.

        Args:
            exponent: Primitive Gaussian exponent :math:`\alpha_p` (Bohr\ :sup:`-2`).
            angular_L: Total Cartesian angular momentum :math:`L_p = n_x + n_y + n_z`.
            tol: Numerical truncation on the envelope. Defaults to
                :data:`jqmc._setting.AO_IMAGE_TRUNCATION_ATOL`.

        Returns:
            npt.NDArray[np.float64]: Lattice translations :math:`T_n` with
            shape ``(N_img, 3)`` in Bohr. Always includes :math:`T_0 = 0`.
        """
        if not self.pbc_flag:
            raise ValueError("primitive_image_list is defined only for pbc_flag=True structures.")
        return _primitive_image_list(self.cell, exponent, angular_L, tol)


def normalize_tiling_matrix(tiling: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return a validated integer ``(3, 3)`` supercell tiling matrix.

    A length-three input is the common diagonal shorthand ``(n1, n2, n3)``.
    A flat length-nine input is reshaped row-major for command-line callers.
    The full integer matrix follows the row-lattice convention
    ``supercell = S @ primitive_cell``.
    """
    # Object conversion preserves Python integers before range checking; casting
    # through float64 loses valid integers above 2**53 and can change det(S).
    raw = np.asarray(tiling, dtype=object)
    if raw.shape == (3,):
        raw = np.diag(raw)
    elif raw.shape == (9,):
        raw = raw.reshape(3, 3)
    if raw.shape != (3, 3):
        raise ValueError("tiling must be a finite length-3 diagonal vector, flat length-9 matrix, or 3x3 integer matrix.")
    entries = []
    limits = np.iinfo(np.int64)
    for value in raw.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError("tiling entries must be finite integers, not booleans, strings, or complex numbers.")
        if not isinstance(value, numbers.Integral) and (not math.isfinite(value) or value != math.trunc(value)):
            raise ValueError("tiling entries must be finite integers.")
        integer = int(value)
        if not limits.min <= integer <= limits.max:
            raise ValueError("tiling entries must fit in int64.")
        entries.append(integer)
    matrix = np.asarray(entries, dtype=np.int64).reshape(3, 3)
    determinant, _ = _tiling_determinant_adjugate(matrix)
    if determinant == 0:
        raise ValueError("tiling matrix must be nonsingular.")
    return matrix


def _tiling_determinant_adjugate(matrix: npt.ArrayLike) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Return the exact determinant and adjugate of an integer 3x3 matrix.

    Python integer intermediates avoid overflow even when the entries fit in
    int64 but the cofactors do not. Callers validate the input shape/type first.
    """
    a, b, c, d, e, f, g, h, i = (int(value) for value in np.asarray(matrix).flat)
    adjugate = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    determinant = a * adjugate[0][0] + b * adjugate[1][0] + c * adjugate[2][0]
    return determinant, adjugate


def _tiling_quotient_residues(matrix: npt.ArrayLike) -> tuple[tuple[int, int, int], ...]:
    """Enumerate integer numerators of Z^3 / (Z^3 S), denominator |det(S)|."""
    determinant, adjugate = _tiling_determinant_adjugate(matrix)
    modulus = abs(determinant)
    sign = 1 if determinant > 0 else -1
    generators = tuple(tuple(sign * value % modulus for value in row) for row in adjugate)
    residues = {(0, 0, 0)}
    frontier = [(0, 0, 0)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            candidate = tuple((value + step) % modulus for value, step in zip(current, generator, strict=True))
            if candidate not in residues:
                residues.add(candidate)
                frontier.append(candidate)
    if len(residues) != modulus:
        raise RuntimeError("integer tiling quotient enumeration produced the wrong number of cosets.")
    return tuple(sorted(residues))


def tiling_translation_fractions(tiling: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""Representatives of the primitive translations inside a supercell.

    Returns ``|det(S)|`` fractional rows in the supercell basis.  The finite
    quotient is enumerated exactly with the integer adjugate: for primitive
    translation ``n``, its supercell fraction is ``n @ inv(S) (mod 1)``.
    This avoids a geometry-dependent bounding-box search for skew or
    non-diagonal tilings.
    """
    matrix = normalize_tiling_matrix(tiling)
    determinant, _ = _tiling_determinant_adjugate(matrix)
    return np.asarray(_tiling_quotient_residues(matrix), dtype=np.float64) / abs(determinant)


def tile_structure_data(primitive: Structure_data, tiling: npt.ArrayLike) -> Structure_data:
    """Replicate a periodic primitive structure into ``S @ primitive.cell``.

    Atom order is translation-major and primitive-atom-minor.  Species and QE
    pseudopotential labels are repeated unchanged so per-species ECP lookup
    continues to work on the returned supercell.
    """
    if not primitive.pbc_flag:
        raise ValueError("tile_structure_data requires a periodic primitive structure.")
    matrix = normalize_tiling_matrix(tiling)
    supercell = matrix.astype(np.float64) @ primitive.cell
    inv_primitive = np.linalg.inv(primitive.cell)
    inv_matrix = np.linalg.inv(matrix.astype(np.float64))
    atom_frac_primitive = np.asarray(primitive.positions, dtype=np.float64) @ inv_primitive
    translations = tiling_translation_fractions(matrix)
    positions: list[np.ndarray] = []
    for translation in translations:
        frac_super = atom_frac_primitive @ inv_matrix + translation[None, :]
        frac_super -= np.floor(frac_super)
        # Snap the roundoff representation of the upper boundary back to 0.
        frac_super[
            np.isclose(
                frac_super,
                1.0,
                rtol=RTOL_BIT_EXACT,
                atol=ATOL_TILING_FRACTIONAL_BOUNDARY_SNAP,
            )
        ] = 0.0
        positions.extend(frac_super @ supercell)
    repetitions = len(translations)
    return Structure_data.build(
        positions=np.asarray(positions, dtype=np.float64),
        pbc_flag=True,
        vec_a=supercell[0],
        vec_b=supercell[1],
        vec_c=supercell[2],
        atomic_numbers=tuple(primitive.atomic_numbers) * repetitions,
        element_symbols=tuple(primitive.element_symbols) * repetitions,
        atomic_labels=tuple(primitive.atomic_labels) * repetitions,
    )


def _find_nearest_index_np(structure: Structure_data, r_cart: npt.ArrayLike) -> np.integer:
    """Return the nearest atom index for one Cartesian position.

    Args:
        structure: Atomic positions and optional periodic cell.
        r_cart: Cartesian position with shape ``(3,)`` in Bohr.

    Returns:
        NumPy integer containing the nearest nucleus index.
    """
    return _find_nearest_nucleus_indices_np(structure, r_cart, 1)[0]


def _find_nearest_index_jnp(structure: Structure_data, r_cart: jax.Array) -> jax.Array:
    """Return the nearest atom index for one Cartesian position.

    Args:
        structure: Atomic positions and optional periodic cell.
        r_cart: Cartesian position with shape ``(3,)`` in Bohr.

    Returns:
        Scalar integer JAX array containing the nearest nucleus index.
    """
    return _find_nearest_nucleus_indices_jnp(structure, r_cart, 1)[0]


def _find_nearest_nucleus_indices_np(
    structure_data: Structure_data,
    r_cart: npt.ArrayLike,
    number_of_nuclei: int,
) -> npt.NDArray[np.intp]:
    """Return nearest nucleus indices for one Cartesian position.

    Distances use direct Cartesian displacements for molecules and exact
    Wigner-Seitz minimum-image displacements for periodic cells, including
    oblique cells.

    Args:
        structure_data: Atomic positions and optional periodic cell.
        r_cart: Cartesian position with shape ``(3,)`` in Bohr.
        number_of_nuclei: Number of nearest indices to return.

    Returns:
        Integer NumPy array with shape ``(number_of_nuclei,)``, ordered by
        increasing distance.
    """
    dtype_np = np.float64
    positions = structure_data._positions_cart_np
    r_cart = np.array(r_cart, dtype=dtype_np)
    diffs = positions - r_cart
    if structure_data.pbc_flag:
        cell = structure_data.cell
        inv_cell = np.linalg.inv(cell)
        diffs = np.stack([_minimum_image_displacement_debug(d, cell, inv_cell, structure_data._n_max) for d in diffs])

    distances = np.sqrt(np.sum(diffs**2, axis=1))
    nearest_indices = np.argsort(distances)
    return nearest_indices[:number_of_nuclei]


@partial(jit, static_argnums=2)
def _find_nearest_nucleus_indices_jnp(
    structure_data: Structure_data,
    r_cart: jax.Array,
    number_of_nuclei: int,
) -> jax.Array:
    """Return nearest nucleus indices for one Cartesian position.

    Distances use direct Cartesian displacements for molecules and exact
    Wigner-Seitz minimum-image displacements for periodic cells, including
    oblique cells. ``number_of_nuclei`` is static under JIT because it fixes
    the result shape.

    Args:
        structure_data: Atomic positions and optional periodic cell.
        r_cart: Cartesian position with shape ``(3,)`` in Bohr.
        number_of_nuclei: Number of nearest indices to return.

    Returns:
        Integer JAX array with shape ``(number_of_nuclei,)``, ordered by
        increasing distance.
    """
    dtype_jnp = jnp.float64
    positions = structure_data._positions_cart_jnp
    r_cart = jnp.array(r_cart, dtype=dtype_jnp)
    diffs = positions - r_cart
    if structure_data.pbc_flag:
        cell = structure_data._cell_jnp.astype(dtype_jnp)
        inv_cell = _inv_3x3(cell)
        diffs = jax.vmap(_minimum_image_displacement, in_axes=(0, None, None, None))(
            diffs, cell, inv_cell, structure_data._n_max
        )

    distances = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    nearest_indices = jnp.argsort(distances)
    return nearest_indices[:number_of_nuclei]


def _get_min_dist_rel_R_cart_np(
    structure_data: Structure_data,
    r_cart: npt.ArrayLike,
    i_atom: int | np.integer,
) -> npt.NDArray[np.float64]:
    """Return the minimum-image atom-to-electron displacement.

    The returned vector is ``R_atom - r_cart`` for molecules. Under PBC it is
    reduced to the exact Wigner-Seitz minimum image, including for oblique
    cells. The debug path always uses NumPy fp64 arithmetic.

    Args:
        structure_data: Atomic positions and optional periodic cell.
        r_cart: Electron Cartesian position with shape ``(3,)`` in Bohr.
        i_atom: Target nucleus index.

    Returns:
        Minimum-image displacement with shape ``(3,)`` in Bohr.
    """
    dtype_np = np.float64
    r_cart = np.array(r_cart, dtype=dtype_np)
    R_cart = structure_data._positions_cart_np[i_atom]
    diff = R_cart - r_cart
    if structure_data.pbc_flag:
        cell = structure_data.cell
        inv_cell = np.linalg.inv(cell)
        diff = _minimum_image_displacement_debug(diff, cell, inv_cell, structure_data._n_max)
    return diff


@partial(jit, static_argnames=("dtype",))
def _get_min_dist_rel_R_cart_jnp(
    structure_data: Structure_data,
    r_cart: jax.Array,
    i_atom: int | jax.Array,
    dtype: DTypeLike | None = None,
) -> jax.Array:
    """Return the minimum-image atom-to-electron displacement.

    The returned vector is ``R_atom - r_cart`` for molecules. Under PBC it is
    reduced to the exact Wigner-Seitz minimum image, including for oblique
    cells. Subtraction happens before the result is cast to ``dtype`` so the
    caller controls the local precision zone without losing upstream
    coordinate precision.

    ``dtype`` is marked static for ``jit`` because it is a Python scalar type
    (e.g. ``jnp.float32``) and cannot be traced as an abstract array.

    Args:
        structure_data: Atomic positions and optional periodic cell.
        r_cart: Electron Cartesian position with shape ``(3,)`` in Bohr.
        i_atom: Target nucleus index; may be a scalar tracer under ``vmap``.
        dtype: Output arithmetic dtype. Defaults to ``jnp.float64``.

    Returns:
        Minimum-image displacement with shape ``(3,)``.
    """
    if dtype is None:
        dtype = jnp.float64
    # Keep coordinate precision through subtraction and cast only the result
    # down to the caller's local precision zone.
    R_cart = structure_data._positions_cart_jnp[i_atom]
    diff = (R_cart - r_cart).astype(dtype)
    if structure_data.pbc_flag:
        cell = structure_data._cell_jnp.astype(dtype)
        inv_cell = _inv_3x3(cell)
        diff = _minimum_image_displacement(diff, cell, inv_cell, structure_data._n_max)
    return diff


def _strain_structure_data(sd: "Structure_data", F: jax.Array) -> "Structure_data":
    r"""Apply the affine deformation gradient :math:`F = I + \varepsilon` to a leaf.

    Returns a new ``Structure_data`` with:

    - ``positions[I, :] = F @ positions[I, :]`` (row-vector convention)
    - ``vec_a/b/c = F @ vec_a/b/c``

    This is the dual of :py:func:`_accumulate_strain_grad_per_leaf`: under
    a strain ``\varepsilon``, the lattice and atomic positions transform
    as :math:`x \to F x`, and ``\partial / \partial \varepsilon`` of any
    quantity computed downstream collapses (via the chain rule) to the
    Cauchy-Born projection the accumulator returns at ``\varepsilon = 0``.

    Molecular instances (``pbc_flag=False``) pass through with strained
    positions only --- the empty ``vec_a/b/c`` arrays are preserved.

    Args:
        sd: forward-value ``Structure_data`` leaf.
        F: ``(3, 3) fp64`` deformation gradient. Under stress AD it is a
            JAX tracer; outside AD it can be a constant identity for a
            pass-through.

    Returns:
        Structure_data: new leaf with strained positions and lattice vectors.
    """
    dtype = jnp.float64
    F = F.astype(dtype)
    pos = jnp.asarray(sd.positions, dtype=dtype)
    if pos.size == 0:
        new_pos = sd.positions
    else:
        # row-vector convention: each position is a row, F multiplies from left
        # so r_new = F @ r => positions_new = positions @ F.T
        new_pos = pos @ F.T
    if sd.pbc_flag and sd.vec_a.size == 3:
        va = jnp.asarray(sd.vec_a, dtype=dtype)
        vb = jnp.asarray(sd.vec_b, dtype=dtype)
        vc = jnp.asarray(sd.vec_c, dtype=dtype)
        new_va = F @ va
        new_vb = F @ vb
        new_vc = F @ vc
        return sd.replace(positions=new_pos, vec_a=new_va, vec_b=new_vb, vec_c=new_vc)
    return sd.replace(positions=new_pos)


def _accumulate_strain_grad_per_leaf(sd_grad: "Structure_data", sd_value: "Structure_data") -> jax.Array:
    r"""Cauchy-Born strain-cotangent contribution from one ``Structure_data`` leaf.

    Returns the ``(3, 3)`` matrix

    .. math::
        \sigma_{\alpha\beta}^{\text{leaf}}
            = (\nabla_\text{cell} E)^T \cdot \text{cell}
            + (\nabla_\text{positions} E)^T \cdot \text{positions}

    where ``cell`` stacks ``vec_a/b/c`` as rows. The two terms are the
    Cauchy-Born identity for the strain Jacobian under the affine
    transform :math:`x \to (I + \varepsilon)\,x`. Producing this
    matrix per ``Structure_data`` instance is correct because every leaf
    instance is an independent JAX cotangent target; the upstream
    accumulator (``Hamiltonian_data.accumulate_strain_grad``) sums these
    contributions across the tree.

    Args:
        sd_grad: ``Structure_data`` whose ``vec_a/b/c`` and ``positions``
            carry JAX cotangents (the output of ``jax.grad``).
        sd_value: The corresponding forward-value ``Structure_data`` whose
            ``_cell_jnp`` and ``_positions_cart_jnp`` supply the
            contraction factors.

    Returns:
        ``(3, 3) fp64`` strain-cotangent contribution. The returned matrix
        is generally non-symmetric; the top-level caller
        (``Hamiltonian_data.accumulate_strain_grad``) is responsible for
        symmetrisation if downstream code requires it.
    """
    dtype = jnp.float64
    cell_grad = sd_grad._cell_jnp.astype(dtype)  # (3, 3)
    cell_value = sd_value._cell_jnp.astype(dtype)  # (3, 3)
    sigma = cell_grad.T @ cell_value
    pos_grad = jnp.asarray(sd_grad.positions, dtype=dtype)
    pos_value = jnp.asarray(sd_value.positions, dtype=dtype)
    if pos_grad.size != 0 and pos_value.size != 0:
        sigma = sigma + pos_grad.T @ pos_value
    return sigma


@jit
def _wrap_into_cell(r_cart: jax.Array, cell: jax.Array, inv_cell: jax.Array) -> jax.Array:
    """Wrap Cartesian positions into the home parallelepiped.

    Maps an arbitrary Cartesian point onto its representative in
    :math:`[0, 1)^3` fractional coordinates, then back to Cartesian. The
    trailing axis (length 3) is contracted by ``inv_cell``; any leading axes
    broadcast through unchanged, so the same helper serves single-electron
    inputs ``(3,)`` and batched inputs such as ``(n_spin, NN, Nv, 3)`` or
    ``(n_walker, N_e, 3)``.

    Output dtype matches input dtype; ``cell`` / ``inv_cell`` carry the
    fp64 precision needed for an exact ``floor`` round-down on positions
    of order :math:`10^{10}` Bohr.

    Args:
        r_cart: Cartesian coordinates with shape ``(..., 3)``.
        cell: Lattice vectors as rows, shape ``(3, 3)``.
        inv_cell: Inverse of ``cell``, shape ``(3, 3)``.

    Returns:
        jax.Array: Wrapped Cartesian coordinates with the same shape and
        dtype as ``r_cart``.
    """
    frac = r_cart @ inv_cell
    frac = frac - jnp.floor(frac)
    return frac @ cell


def _inv_3x3(m: jax.Array) -> jax.Array:
    r"""Analytic inverse of a 3x3 matrix (adjugate / determinant).

    ``jnp.linalg.inv`` lowers even a 3x3 to the cuSOLVER ``getrf`` +
    ``triangularSolve`` LU path, which explodes into ~11 tiny grid=1 kernels
    per call. In the MCMC single-electron AO-eval hot loop that ``inv(cell)``
    runs every move and measured ~38% of the per-move orbital-eval kernels
    (ncu); the closed form below is a handful of elementwise ops instead.

    Numerically equivalent to ``jnp.linalg.inv`` for an invertible 3x3 to
    machine precision (AO output differs by ~1e-12; a production MCMC
    energy is identical to 5+ digits at fixed seed). Stays differentiable, so
    the ``cell`` cotangent for stress AD flows through exactly.
    Measured: per-move MCMC update 365 -> 327 ms/step (-10%) on a GPU.
    """
    c00 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    c01 = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    c02 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    c10 = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
    c11 = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    c12 = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]
    c20 = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
    c21 = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
    c22 = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    det = m[0, 0] * c00 + m[0, 1] * c01 + m[0, 2] * c02
    adj = jnp.array([[c00, c10, c20], [c01, c11, c21], [c02, c12, c22]], dtype=m.dtype)
    return adj / det


def _det_3x3(m: jax.Array) -> jax.Array:
    """Analytic determinant of a 3x3 matrix (cofactor expansion along row 0).

    Same motivation as :py:func:`_inv_3x3`: ``jnp.linalg.det`` lowers even a
    3x3 to the cuSOLVER ``getrf`` LU path, so a traced ``det(cell)`` drags the
    whole LU custom-call machinery into the module for what is three products
    of 2x2 minors. Callers wanting the cell VOLUME still need ``jnp.abs``.

    Numerically equivalent to ``jnp.linalg.det`` for a 3x3 to machine
    precision, and differentiable, so the ``cell`` cotangent for stress AD
    flows through unchanged.
    """
    c00 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    c01 = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    c02 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    return m[0, 0] * c00 + m[0, 1] * c01 + m[0, 2] * c02


def _wrap_into_cell_if_pbc(structure_data: "Structure_data", r_cart: jax.Array) -> jax.Array:
    """Wrap positions into the home cell, or pass them through for molecules.

    Geometry-agnostic front end to :py:func:`_wrap_into_cell`: callers no
    longer branch on ``pbc_flag`` themselves. ``pbc_flag`` is a static
    (``pytree_node=False``) field, so the ``if`` resolves at JIT trace time
    and the molecular branch compiles to an identity (no ``cell`` inverse,
    no ``floor``). The PBC branch builds ``cell`` / ``inv_cell`` in fp64
    internally so call sites stay free of lattice boilerplate.

    Args:
        structure_data: Structure carrying ``pbc_flag`` and lattice vectors.
        r_cart: Cartesian coordinates with shape ``(..., 3)``.

    Returns:
        jax.Array: For PBC, ``r_cart`` wrapped into ``[0, 1)^3`` fractional
        and mapped back to Cartesian; for molecules, ``r_cart`` unchanged.
    """
    if not structure_data.pbc_flag:
        return r_cart
    # Use the JAX-aware ``_cell_jnp`` so the cell
    # cotangent flows through ``inv(cell)`` into the wrap (and on into
    # ``compute_AOs`` for stress AD). ``jnp.asarray(structure_data.cell)``
    # would route through the numpy ``cell`` property and lose the
    # tracer.
    cell = structure_data._cell_jnp
    inv_cell = _inv_3x3(cell)
    return _wrap_into_cell(r_cart, cell, inv_cell)


def _gram_schmidt_rows(basis: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthogonalisation (no normalisation) of the rows of ``basis``."""
    basis = np.asarray(basis, dtype=np.float64)
    gs = basis.copy()
    for i in range(1, basis.shape[0]):
        for m in range(i):
            gs[i] -= (basis[i] @ gs[m]) / (gs[m] @ gs[m]) * gs[m]
    return gs


def _lll_reduce_basis(cell: np.ndarray) -> np.ndarray:
    r"""LLL-reduced basis (``delta = 3/4``) of the lattice spanned by the rows of ``cell``.

    Only unimodular integer row operations are applied (size reduction and
    swaps), so the returned rows span the same lattice. The iteration is
    capped; callers must use the result only in bounds that are valid for
    *any* basis of the lattice (the reduction only makes them tighter), so
    correctness never relies on the reduction having converged.

    Static helper: concrete ``(3, 3)`` ndarray in, ``(3, 3)`` ndarray out
    (Bohr). Shared by :py:meth:`Structure_data._compute_n_max` and
    :py:func:`_shortest_lattice_vector_length`.
    """
    basis = np.asarray(cell, dtype=np.float64).copy()
    k, n_steps = 1, 0
    while k < 3 and n_steps < 1000:
        n_steps += 1
        for j in range(k - 1, -1, -1):
            gs = _gram_schmidt_rows(basis)
            mu = (basis[k] @ gs[j]) / (gs[j] @ gs[j])
            if abs(mu) > 0.5:
                basis[k] -= np.round(mu) * basis[j]
        gs = _gram_schmidt_rows(basis)
        mu = (basis[k] @ gs[k - 1]) / (gs[k - 1] @ gs[k - 1])
        if gs[k] @ gs[k] >= (0.75 - mu * mu) * (gs[k - 1] @ gs[k - 1]):
            k += 1
        else:
            basis[[k - 1, k]] = basis[[k, k - 1]]
            k = max(k - 1, 1)
    return basis


def _minimum_image_box_size(half_widths) -> int:
    """Exact number ``prod_k (2 N_k + 1)`` of integer vectors with ``|n_k| <= N_k`` (Python int, no overflow)."""
    size = 1
    for h in half_widths:
        size *= 2 * int(h) + 1
    return size


def _check_minimum_image_box_size(half_widths, what: str) -> None:
    """Raise ``ValueError`` if the integer box ``|n_k| <= half_widths[k]`` exceeds ``MINIMUM_IMAGE_SHELL_MAX_CANDIDATES``.

    Guards the eager expansions of the static minimum-image helpers, so an
    infeasible box (an extreme unreduced cell basis) fails with a clear error
    instead of attempting the allocation. Every box at or below the limit is
    left untouched.
    """
    size = _minimum_image_box_size(half_widths)
    if size > MINIMUM_IMAGE_SHELL_MAX_CANDIDATES:
        raise ValueError(
            f"The {what} with half-widths {tuple(int(h) for h in half_widths)} has {size} integer candidates, "
            f"more than MINIMUM_IMAGE_SHELL_MAX_CANDIDATES = {MINIMUM_IMAGE_SHELL_MAX_CANDIDATES}. The cell basis is "
            "too far from reduced to enumerate its images; pass an LLL- or Niggli-reduced basis of the same lattice."
        )


def _shortest_lattice_vector_length(cell: np.ndarray, n_search: int = 2) -> float:
    r"""Length of the shortest non-zero vector of the lattice spanned by the rows of ``cell``.

    The result depends only on the lattice, not on the choice of cell basis.
    The basis is first LLL-reduced (:py:func:`_lll_reduce_basis`), which keeps
    the search box small. The integer search box is then provably sufficient
    for any basis ``b``: the shortest non-zero lattice vector satisfies
    ``|T| <= l = min_i |b_i|``, and its integer coefficients are
    ``n_k = T . inv(b)[:, k]``, so ``|n_k| <= l |inv(b)[:, k]|``. Each axis is
    searched over ``|n_k| <= max(n_search, floor(l |inv(b)[:, k]|) + 1)``;
    ``n_search`` is only a lower bound on the half-width, and the bound holds
    even if the reduction stops early (issue #53).

    Static helper: concrete ``(3, 3)`` ndarray in, Python float out (Bohr).
    Shared by :py:meth:`Structure_data._wigner_seitz_inscribed_radius` and
    :py:func:`jqmc.mpc.wigner_seitz_radius`.
    """
    cell = np.asarray(cell, dtype=np.float64)
    basis = _lll_reduce_basis(cell)
    shortest_row = float(np.min(np.linalg.norm(basis, axis=1)))
    dual_norms = np.linalg.norm(np.linalg.inv(basis), axis=0)  # |inv(basis)[:, k]|
    half_widths = [max(int(n_search), int(np.floor(shortest_row * d)) + 1) for d in dual_norms]
    _check_minimum_image_box_size(half_widths, "shortest-lattice-vector search box")
    axes = [np.arange(-h, h + 1, dtype=np.float64) for h in half_widths]
    n_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n_grid = n_grid[np.any(n_grid != 0.0, axis=1)]
    lengths = np.linalg.norm(n_grid @ basis, axis=1)
    return float(np.min(lengths))


def _orthogonal_axes(cell: np.ndarray) -> tuple[bool, bool, bool]:
    """Per axis ``k``: whether row ``k`` of ``cell`` is exactly orthogonal to the other two rows.

    The test is on the exact floating-point dot products (no tolerance): an
    axis-aligned orthorhombic cell has exactly zero off-diagonal Gram entries,
    so it takes the orthogonal branches of
    :py:meth:`Structure_data._wigner_seitz_inscribed_radius` and
    :py:func:`_minimum_image_shell_half_widths`. A cell that is orthogonal
    only up to round-off is treated as oblique, which is always valid there
    (it only takes the general formula).
    """
    cell = np.asarray(cell, dtype=np.float64)
    gram = cell @ cell.T
    return tuple(bool(all(gram[k, i] == 0.0 for i in range(3) if i != k)) for k in range(3))


def _minimum_image_shell_half_widths(cell: np.ndarray, radius: float | None = None) -> tuple[int, int, int]:
    r"""Integer shell around the parallelepiped wrap that recovers every minimum image shorter than ``radius``.

    The parallelepiped wrap ``d_w = d - round(d @ inv(cell)) @ cell`` (one
    rounding in the given basis, as the DTN Jastrow kernels use) returns the
    minimum image only when it is shorter than half the smallest interplanar
    spacing of the given parallelepiped. On oblique cells (FCC, BCC,
    hexagonal, any non-reduced basis) that is smaller than the Wigner-Seitz
    inscribed radius, so separations between the two radii can be wrapped to
    a longer image (issue #68). Searching the candidates
    ``d_w + n @ cell`` with ``|n_k| <= N_k`` and keeping the shortest fixes
    that, with

    .. math::
       N_k = \lceil \rho\, |g_k| + \tfrac12 \rceil - 1, \qquad
       g_k = \mathrm{inv(cell)}[:, k].

    Proof: write ``d_w = f @ cell`` with ``|f_k| <= 1/2`` and the minimum
    image ``d* = d_w + n @ cell``. Then ``n_k = d* . g_k - f_k``, so
    ``|n_k| <= |d*| |g_k| + 1/2 < rho |g_k| + 1/2`` whenever
    ``|d*| < rho``, i.e. ``|n_k| <= N_k``.

    ``radius=None`` uses ``rho = min_i |b_i| / 2`` (half the shortest *basis*
    vector). That is at least the Wigner-Seitz inscribed radius
    (half the shortest *lattice* vector), so the shell then recovers the
    minimum image of every separation below the WS inscribed radius, the
    largest cutoff the DTN Jastrow accepts. It needs no lattice reduction, so
    the Rust ``CellInfo`` evaluates the same rule.

    An axis that is exactly orthogonal to the other two (:py:func:`_orthogonal_axes`)
    has ``|g_k| = 1 / |b_k|`` and ``rho <= |b_k| / 2``, so ``N_k = 0``; it is
    set directly so that round-off in ``|g_k|`` cannot add a shell.
    ``radius`` must not exceed ``min_i |b_i| / 2`` for that shortcut to hold;
    callers pass a DTN cutoff, which is below the WS inscribed radius.
    An orthorhombic cell therefore gets ``(0, 0, 0)``: the wrap alone is exact.

    Args:
        cell: Lattice vectors as rows, concrete ``(3, 3)`` array (Bohr).
        radius: Largest minimum-image length that must be recovered (Bohr);
            ``None`` for half the shortest basis vector.

    Returns:
        tuple[int, int, int]: ``(N_a, N_b, N_c)``.
    """
    cell = np.asarray(cell, dtype=np.float64)
    rho = 0.5 * float(np.min(np.linalg.norm(cell, axis=1))) if radius is None else float(radius)
    dual_norms = np.linalg.norm(np.linalg.inv(cell), axis=0)  # |g_k| = |inv(cell)[:, k]|
    orthogonal = _orthogonal_axes(cell)
    return tuple(0 if orthogonal[k] else max(int(np.ceil(rho * float(dual_norms[k]) + 0.5)) - 1, 0) for k in range(3))


def _minimum_image_reduction_plan(
    cell: np.ndarray,
) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], tuple[int, int, int]]:
    """Static plan ``(M, half_widths)`` for an exact minimum image below the WS inscribed radius.

    ``M`` is an integer unimodular matrix and ``half_widths`` the shell of
    :py:func:`_minimum_image_shell_half_widths` for the basis ``M @ cell`` (the
    same lattice). Consumers wrap a displacement in ``M @ cell`` and search
    that shell. Two candidates are compared by the shell size
    ``prod(2 N_k + 1)`` (exact Python-int arithmetic, :py:func:`_minimum_image_box_size`): the given basis (``M = I``) and the LLL-reduced basis
    of :py:func:`_lll_reduce_basis` (``M = rint(reduced @ inv(cell))``). The
    given basis is kept unless the reduced one needs a strictly smaller shell,
    so an orthorhombic cell keeps ``(I, (0, 0, 0))`` (the plain wrap, bit for
    bit) and the standard FCC/BCC/hexagonal cells keep their own basis. For an
    LLL-reduced basis (``delta = 3/4``) in three dimensions
    ``prod_i |b_i| <= 2^{3/2} V``, so ``rho |g_k| <= 2^{1/2}`` and every
    ``N_k <= 1``: a non-reduced input basis costs at most 26 extra candidates
    instead of a shell that grows with its skew. Correctness does not depend on
    the reduction: the shell is always computed for the basis actually used.

    Args:
        cell: Lattice vectors as rows, concrete ``(3, 3)`` array (Bohr).

    Returns:
        tuple: ``(M, half_widths)`` with ``M`` a nested int tuple (rows).
    """
    cell = np.asarray(cell, dtype=np.float64)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    given = _minimum_image_shell_half_widths(cell)
    transform = np.rint(_lll_reduce_basis(cell) @ np.linalg.inv(cell))
    if abs(round(float(np.linalg.det(transform)))) != 1:
        return identity, given
    reduced = _minimum_image_shell_half_widths(transform @ cell)

    # Exact Python-int cardinalities: a fixed-width product overflows for an
    # extreme unreduced basis and would keep its huge shell (PR #82).
    if _minimum_image_box_size(reduced) < _minimum_image_box_size(given):
        return tuple(tuple(int(x) for x in row) for row in transform), reduced
    return identity, given


def _minimum_image_shell_shifts(half_widths: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    """Non-zero integer shifts ``n`` with ``|n_k| <= half_widths[k]`` (static, lexicographic order).

    Empty for ``(0, 0, 0)``, so an orthorhombic cell keeps the plain wrap.
    Raises ``ValueError`` (:py:func:`_check_minimum_image_box_size`) instead of
    expanding a shell larger than ``MINIMUM_IMAGE_SHELL_MAX_CANDIDATES``.
    """
    _check_minimum_image_box_size(half_widths, "minimum-image shell")
    axes = [range(-int(h), int(h) + 1) for h in half_widths]
    return tuple(n for n in itertools.product(*axes) if any(n))


@jit
def _min_image_fractional_parallelepiped(diff: jax.Array, inv_cell: jax.Array) -> jax.Array:
    """Parallelepiped-rounded fractional minimum image (NOT Wigner-Seitz).

    Returns the fractional offset of ``diff`` after rounding to
    :math:`(-1/2, 1/2]^3` along each lattice axis. The result is
    **exact for orthogonal cells** but does **not** give the geometrically
    nearest image on oblique cells (FCC primitive, hcp, rhombohedral,
    triclinic): along oblique axes the true nearest image can lie outside
    the parallelepiped wrap.

    Used by Jastrow ``p(x)`` periodization where any periodicity-preserving
    wrap suffices (``sin`` has period 1, so the choice of representative
    does not change :math:`J_1` / :math:`J_2`). For Coulomb / ECP /
    nearest-nucleus dispatch use :py:func:`_minimum_image_displacement`,
    which is exact on arbitrary cells.

    Args:
        diff: Cartesian displacement with shape ``(3,)``.
        inv_cell: Inverse lattice matrix, shape ``(3, 3)``.

    Returns:
        jax.Array: Wrapped fractional offset, shape ``(3,)``, in ``(-1/2, 1/2]``.
    """
    diff_frac = diff @ inv_cell
    return diff_frac - jnp.round(diff_frac)


@partial(jit, static_argnums=3)
def _minimum_image_displacement(
    diff: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
    n_max: int,
) -> jax.Array:
    r"""Wigner-Seitz minimum-image displacement, exact for arbitrary cells.

    Returns the lattice-translated copy of ``diff`` with the smallest
    Euclidean norm. The result is geometrically correct for orthorhombic,
    monoclinic, triclinic, FCC primitive, hcp, and rhombohedral cells; the
    naive ``round(diff @ inv_cell)`` parallelepiped trick is exact for
    orthogonal cells only.

    The algorithm enumerates the :math:`(2 n_\text{max} + 1)^3` integer
    lattice translations in a box of half-width ``n_max`` (already wrapped
    once to the parallelepiped to keep the search box compact) and selects
    the closest. With ``n_max`` carried as a Python ``int`` and ``static``
    under JIT, the candidate count is a compile-time constant; no SIMT
    divergence on GPU, no ragged ``lax.scan``.

    Caller responsibilities:
      * ``cell`` and ``inv_cell`` carry the desired numerical precision
        (selectable-precision modules cast before the call); this function
        does no internal cast.
      * ``n_max`` is supplied by ``Structure_data._n_max``, sized by
        :py:meth:`Structure_data._compute_n_max`, whose bound provably
        encloses every Wigner-Seitz nearest image for any cell basis
        (including non-reduced bases, issue #67).

    Args:
        diff: Cartesian displacement, shape ``(3,)``, caller dtype.
        cell: Lattice vectors as rows, shape ``(3, 3)``.
        inv_cell: Inverse of ``cell``, shape ``(3, 3)``.
        n_max: Static half-width of the integer-image enumeration box.

    Returns:
        jax.Array: Wigner-Seitz minimum-image displacement, shape ``(3,)``,
        in the same dtype as ``diff``.
    """
    diff_frac = diff @ inv_cell
    diff_frac = diff_frac - jnp.round(diff_frac)
    diff_wrapped = diff_frac @ cell

    n_range = jnp.arange(-n_max, n_max + 1)
    na, nb, nc = jnp.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_grid = jnp.stack([na.reshape(-1), nb.reshape(-1), nc.reshape(-1)], axis=-1).astype(cell.dtype)
    T_images = n_grid @ cell

    candidates = diff_wrapped[None, :] + T_images
    d2 = jnp.sum(candidates * candidates, axis=-1)
    best = jnp.argmin(d2)
    return candidates[best]


def _minimum_image_displacement_debug(
    diff: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Reference (numpy, fp64, explicit loops) for ``_minimum_image_displacement``.

    Matches the production version line-for-line.
    Carries no JIT, no ``vmap``, and runs in fp64 unconditionally
    so it serves as the golden-value oracle for cross-checks. Sits beside
    its production sibling; downstream callers (e.g.,
    ``_find_nearest_nucleus_indices_np``) consume it directly on the numpy
    code path.
    """
    diff = np.asarray(diff, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    inv_cell = np.asarray(inv_cell, dtype=np.float64)
    diff_frac = diff @ inv_cell
    diff_frac = diff_frac - np.round(diff_frac)
    diff_wrapped = diff_frac @ cell
    best_d2, best_v = np.inf, diff_wrapped
    for na in range(-n_max, n_max + 1):
        for nb in range(-n_max, n_max + 1):
            for nc in range(-n_max, n_max + 1):
                T = na * cell[0] + nb * cell[1] + nc * cell[2]
                v = diff_wrapped + T
                d2 = float(np.dot(v, v))
                if d2 < best_d2:
                    best_d2, best_v = d2, v
    return best_v


def _primitive_image_radius(exponent: float, angular_L: int, tol: float) -> float:
    r"""Truncation radius for the primitive envelope :math:`r^{L} e^{-\alpha r^2}`.

    Solves :math:`r^L e^{-\alpha r^2} \le \text{tol}` as an *absolute* envelope
    bound. Taking logs gives :math:`\alpha r^2 \ge -\ln\text{tol} + L \ln r`,
    which is solved by fixed-point iteration starting from the ``L = 0`` root
    :math:`r_0 = \sqrt{-\ln\text{tol} / \alpha}` (the Gaussian-only cutoff).
    Each iteration adds the polynomial correction :math:`L \ln r / \alpha`
    under the square root; convergence is fast (< 5 iterations to within
    fp64 round-off for ``tol >= 1e-30`` and ``L <= 6``).

    A closed-form upper bound using a
    ``r_peak * log_inv_tol / L`` correction over-estimates by a factor
    of ``log_inv_tol`` (=27.6 for ``tol = 1e-12``), inflating ``r_cut`` by
    ~4-5x and the image count by ~50-100x for diffuse L >= 1 primitives
    (e.g. Si ccECP-ccpvtz p-shells with alpha ~ 0.056 give ``n_image = 6,931``
    where the correct value is ~180).

    ``tol`` bounds the bare envelope, not an AO value. The periodic AO builder
    (:py:func:`jqmc.atomic_orbital._build_ragged_image_layout`) therefore passes a
    per-primitive tolerance already divided by the primitive's normalised
    prefactor (:py:func:`jqmc.atomic_orbital._primitive_image_truncation_tolerances`,
    issue #120); the pair-overlap lattice callers pass their own term tolerance.
    """
    log_inv_tol = max(-float(np.log(tol)), 1.0)
    if angular_L <= 0:
        return float(np.sqrt(log_inv_tol / exponent))
    r = float(np.sqrt(log_inv_tol / exponent))
    for _ in range(5):
        r = float(np.sqrt((log_inv_tol + angular_L * np.log(max(r, 1.0))) / exponent))
    return r


def _image_candidate_grid(cell: np.ndarray, r_cut: float) -> np.ndarray:
    r"""Integer image candidates ``(n_a, n_b, n_c)`` as a float ``(K, 3)`` array.

    Enumerates every lattice translation whose closest-approach test *could*
    pass for the given ``r_cut``: a cube of half-width
    ``ceil(r_cut / L_min) + 1`` where ``L_min`` is the cell inradius (the
    ``+1`` covers the unit-width fractional box each per-image test sweeps).
    Shared by the cell-cell (:py:func:`_primitive_image_list_indices`) and
    per-atom (:py:func:`_primitive_image_list_indices_per_atom`) selectors so
    both scan the same candidate set.
    """
    cell = np.asarray(cell, dtype=np.float64)
    # cell_inradius (without needing a Structure_data instance).
    volume = float(np.abs(np.dot(cell[0], np.cross(cell[1], cell[2]))))
    face_areas = np.array(
        [
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ],
        dtype=np.float64,
    )
    L_min = float(volume / np.max(face_areas))
    bound = int(np.ceil(r_cut / L_min)) + 1
    n_range = np.arange(-bound, bound + 1)
    na, nb, nc = np.meshgrid(n_range, n_range, n_range, indexing="ij")
    return np.stack([na.ravel(), nb.ravel(), nc.ravel()], axis=-1).astype(np.float64)


def _split_length(vec: np.ndarray) -> tuple[np.ndarray, float, int]:
    r"""Scale-safe ``(unit, mantissa, exponent)`` with ``vec = unit * mantissa * 2**exponent``.

    The vector is first divided by the power of two of its largest entry (exact), so its sum
    of squares neither underflows nor overflows for any finite input (issue #66:
    ``|1e-200|^2`` underflows to zero, ``|1e170|^2`` overflows). ``mantissa`` is in
    ``[0.5, 1)``; a zero vector returns a zero unit, mantissa ``0`` and exponent ``0``.
    """
    vec = np.asarray(vec, dtype=np.float64)
    peak = float(np.max(np.abs(vec)))
    if peak == 0.0:
        return np.zeros_like(vec), 0.0, 0
    shift = math.frexp(peak)[1]
    scaled = np.ldexp(vec, -shift)
    length = math.sqrt(float(scaled @ scaled))
    mantissa, exponent = math.frexp(length)
    return scaled / length, mantissa, exponent + shift


def _length_key(vec: np.ndarray) -> tuple[int, float]:
    """Totally ordered, scale-safe key of ``|vec|`` (exponent first, then mantissa)."""
    _, mantissa, exponent = _split_length(vec)
    return (exponent, mantissa) if mantissa > 0.0 else (-(2**31), 0.0)


def _lagrange_reduce_pair(row_a: np.ndarray, row_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Lagrange-Gauss reduction of two lattice vectors, with the integer transform.

    Returns ``(reduced, transform)`` with ``reduced = transform @ [row_a; row_b]``,
    ``transform`` a unimodular integer ``(2, 2)`` matrix (stored as float64) and
    ``reduced`` a basis of the same 2D lattice whose unit vectors satisfy
    ``|cos(r_0, r_1)| <= 1/2``, so the Gram matrix of the two unit vectors has
    ``det >= 3/4`` and is well conditioned however skewed the input pair is
    (issue #66 floating-point domain). While ``|cos| > 1/2`` a Lagrange step
    ``r_1 <- r_1 - mu r_0`` (``|r_0| <= |r_1|``, ``mu = round((r_1 . r_0) / |r_0|^2)``)
    strictly shortens the longer vector, so the loop terminates.

    Scale safe (issue #66): lengths and cosines come from
    :py:func:`_split_length` and ``mu = cos |r_1| / |r_0|`` is formed from the
    unit vectors and power-of-two lengths, so no squared length is formed. A
    step whose ``mu`` is not representable in float64 ends the reduction.
    """
    reduced = np.array([row_a, row_b], dtype=np.float64)
    transform = np.eye(2, dtype=np.float64)
    while True:
        if _length_key(reduced[0]) > _length_key(reduced[1]):
            reduced = reduced[::-1].copy()
            transform = transform[::-1].copy()
        (unit_0, mantissa_0, exponent_0), (unit_1, mantissa_1, exponent_1) = (_split_length(row) for row in reduced)
        cos = float(unit_0 @ unit_1)
        if not abs(cos) > 0.5:
            return reduced, transform
        with np.errstate(over="ignore"):
            mu = float(np.round(np.ldexp(cos * mantissa_1 / mantissa_0, exponent_1 - exponent_0)))
        if mu == 0.0 or not math.isfinite(mu):
            return reduced, transform
        with np.errstate(over="ignore", invalid="ignore"):
            shortened = reduced[1] - mu * reduced[0]
        if not (np.all(np.isfinite(shortened)) and _length_key(shortened) < _length_key(reduced[1])):
            return reduced, transform
        reduced[1] = shortened
        transform[1] = transform[1] - mu * transform[0]


def _scaled_combination(
    coeffs: np.ndarray, units: np.ndarray, mantissas: np.ndarray, exponents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Scale-safe ``x = coeffs @ rows`` for rows ``units[i] * mantissas[i] * 2**exponents[i]``.

    Returns ``(scaled, exponent)`` with ``x[k] = scaled[k] * 2**exponent[k]``: each term is
    divided by the power of two of the largest term of its row ``k``, so no term overflows and
    only terms below ``2**-1074`` of the largest one underflow (below float64 resolution).
    """
    weights = coeffs * mantissas[None, :]
    term_exponents = np.frexp(weights)[1] + exponents[None, :]
    nonzero = weights != 0.0
    exponent = np.where(nonzero.any(axis=1), np.max(np.where(nonzero, term_exponents, -(2**30)), axis=1), 0)
    scaled = np.ldexp(weights, exponents[None, :] - exponent[:, None])
    return scaled @ units, exponent


def _images_within_rcut_from_box(
    cell: np.ndarray,
    r_cut: float,
    n_candidates: np.ndarray,
    box_low: np.ndarray,
    box_high: np.ndarray,
) -> np.ndarray:
    r"""Closest-approach image selection under the cell Gram metric.

    For each candidate image ``n`` (row of ``n_candidates``),
    ``[box_low, box_high]`` is the *fractional* displacement set swept by the
    closest-approach test (cell-cell: ``[n - 1, n + 1]``; per-atom:
    ``[-n - s_A, 1 - n - s_A]``). Keeps ``n`` iff that box comes within
    ``r_cut`` (Bohr) of the origin.

    The box-constrained quadratic ``min_v |v @ cell|^2`` (``G = cell cell^T``
    positive definite) is solved **exactly** by active-set enumeration
    (issue #66). Each of the three coordinates is either at its lower bound,
    at its upper bound, or free (27 patterns: the interior, 6 faces, 12 edges
    and 8 vertices of the box). For a pattern, the free coordinates minimise
    ``|v @ cell|`` with the bounded ones held fixed (the stationarity equations
    ``G_ff v_f = -G_fx v_x``), and are then clipped into the box. Every pattern
    therefore yields a feasible point, whose value bounds the minimum from
    above, and the pattern of the true minimiser (its interior coordinates
    free, the others at their active bounds) reproduces the minimiser itself.
    The minimum over the 27 patterns is the exact closest-approach distance for
    any cell basis, including non-reduced ones. (The earlier fixed 20-sweep
    coordinate descent returned a non-converged feasible point there, i.e. an
    overestimate, and dropped contributing images.) A candidate is kept iff
    some pattern's feasible point lies within ``r_cut``.

    Floating-point domain: the full Gram matrix is never formed or inverted,
    because for a strongly non-reduced basis (e.g. rows ``(1, 1e8, 0)``,
    ``(0, 1, 0)``, ``(0, 0, 1)``) its float64 rounding loses positive
    definiteness. Distances are evaluated in Cartesian form ``|v @ cell|``. A
    single free coordinate is a projection onto one cell vector. Two free
    coordinates are a projection onto the plane of two cell vectors, solved
    on a Lagrange-Gauss-reduced pair (:py:func:`_lagrange_reduce_pair`, a
    unimodular integer change of the pair, mapped back exactly), whose 2x2
    Gram is well conditioned by construction. All three free gives
    ``v = 0``. The supported domain is every finite cell whose volume exceeds
    ``eps * |a| |b| |c|`` (float64 ``eps``); a numerically singular or
    non-finite cell raises ``ValueError`` instead of returning a wrong image
    set.

    Scale safety (issue #66, e.g. ``diag(1e-200, 1e100, 1e100)``): no
    squared length, Gram entry or squared distance is ever formed in absolute
    units, so nothing under- or overflows for any finite cell of the domain.
    Each cell row is split into a unit vector and a power-of-two length
    (:py:func:`_split_length`); the domain test is ``|det(a_hat, b_hat, c_hat)|
    > eps``, the same condition free of under/overflow; projections are
    ``-(offset . a_hat) / |a|``; the reduced pair is solved in its unit basis;
    and Cartesian points and the cutoff comparison carry a per-candidate
    power-of-two exponent (:py:func:`_scaled_combination`). A pattern whose
    two-free solve is not finite in float64 (only possible for a nearly
    parallel pair whose length ratio exceeds the float64 range, so that its
    reduction step is not representable) keeps the candidate, so an image is
    never dropped by it.

    A per-axis ``sign(n) . max(|n| - 1, 0)`` formula is correct only for
    ORTHOGONAL cells; on oblique cells the off-diagonal Gram terms couple the
    axes. The independent oracle :py:func:`_primitive_image_list_debug`
    computes the same distance with Cartesian point-to-face / point-to-edge
    geometry instead.

    Returns:
        ndarray: kept integer indices, shape ``(N_img, 3)``, dtype int64.

    Raises:
        ValueError: if ``cell`` is non-finite or numerically singular.
    """
    cell = np.asarray(cell, dtype=np.float64)
    box_low = np.asarray(box_low, dtype=np.float64)
    box_high = np.asarray(box_high, dtype=np.float64)
    if not np.all(np.isfinite(cell)):
        raise ValueError("periodic image selection needs a finite, numerically non-singular cell basis (non-finite entry).")
    splits = [_split_length(row) for row in cell]
    units = np.array([split[0] for split in splits])
    mantissas = np.array([split[1] for split in splits])
    exponents = np.array([split[2] for split in splits], dtype=np.int64)
    normalized_volume = abs(float(np.dot(units[0], np.cross(units[1], units[2]))))
    if not normalized_volume > np.finfo(np.float64).eps:
        raise ValueError(
            "periodic image selection needs a finite, numerically non-singular cell basis "
            f"(volume / (|a| |b| |c|) = {normalized_volume!r} <= float64 eps); "
            "the rows of `cell` are (nearly) linearly dependent."
        )
    reduced_pairs = {}
    for pair in ((0, 1), (0, 2), (1, 2)):
        reduced, transform = _lagrange_reduce_pair(cell[pair[0]], cell[pair[1]])
        pair_splits = [_split_length(row) for row in reduced]
        pair_units = np.array([split[0] for split in pair_splits])
        pair_mantissas = np.array([split[1] for split in pair_splits])
        pair_exponents = np.array([split[2] for split in pair_splits], dtype=np.int64)
        reduced_pairs[pair] = (pair_units, pair_mantissas, pair_exponents, transform)
    keep = np.zeros(box_low.shape[0], dtype=bool)
    for pattern in itertools.product((-1, 0, 1), repeat=3):
        free = [axis for axis in range(3) if pattern[axis] == 0]
        fixed = [axis for axis in range(3) if pattern[axis] != 0]
        v = np.zeros_like(box_low)
        for axis in fixed:
            v[:, axis] = box_low[:, axis] if pattern[axis] < 0 else box_high[:, axis]
        unsolved = np.zeros(box_low.shape[0], dtype=bool)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            if free and fixed:
                # Cartesian fixed part = offset * 2**offset_exponent (per candidate).
                offset, offset_exponent = _scaled_combination(v[:, fixed], units[fixed], mantissas[fixed], exponents[fixed])
            if len(free) == 3:
                v_free = np.zeros((box_low.shape[0], 3), dtype=np.float64)  # interior: v = 0
            elif len(free) == 1:
                axis = free[0]
                v_free = np.ldexp(-(offset @ units[axis]) / mantissas[axis], offset_exponent - exponents[axis])[:, None]
            elif len(free) == 2:
                pair_units, pair_mantissas, pair_exponents, transform = reduced_pairs[(free[0], free[1])]
                cos = float(pair_units[0] @ pair_units[1])
                rhs = -(offset @ pair_units.T)  # (K, 2): -(offset . e_hat_j)
                det = 1.0 - cos * cos
                sigma = np.stack([rhs[:, 0] - cos * rhs[:, 1], rhs[:, 1] - cos * rhs[:, 0]], axis=-1) / det
                # weight_j = sigma_j 2**offset_exponent / |e_j|, then v_free = weights @ transform,
                # factored by the common power of two of the two weights so that it never mixes inf and 0.
                weight_exponents = offset_exponent[:, None] - pair_exponents[None, :]
                common = np.max(weight_exponents, axis=1)
                weights = np.ldexp(sigma / pair_mantissas[None, :], weight_exponents - common[:, None])
                v_free = np.ldexp(weights @ transform, common[:, None])
            if free:
                unsolved = np.any(np.isnan(v_free), axis=1)
                v[:, free] = np.clip(v_free, box_low[:, free], box_high[:, free])
            x, x_exponent = _scaled_combination(v, units, mantissas, exponents)
            peak_exponent = np.frexp(np.max(np.abs(x), axis=1))[1]
            x = np.ldexp(x, -peak_exponent[:, None])
            length = np.sqrt(np.einsum("ki,ki->k", x, x))
            within = length <= np.ldexp(r_cut, -(x_exponent + peak_exponent))
        keep |= within | unsolved
    return n_candidates[keep].astype(np.int64)


def _primitive_image_list_indices(
    cell: np.ndarray,
    exponent: float,
    angular_L: int,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> np.ndarray:
    r"""Integer ``(n_a, n_b, n_c)`` triplets for the primitive image set.

    Sibling of :py:func:`_primitive_image_list`. Returns the **integer**
    indices of the kept lattice translations instead of their Cartesian
    values ``T_n = n @ cell``. The image-set selection logic is identical
    to the Cartesian sibling --- the Cartesian helper is implemented as a
    thin wrapper that converts these indices to Bohr.

    Used by the periodic AO image-expansion builder
    (:py:func:`jqmc.atomic_orbital._apply_image_expansion`) where set-style
    unions over distinct ``(alpha, L)`` keys require hashable integer
    triplets, not floats.

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, Bohr.
        exponent: Primitive Gaussian exponent :math:`\alpha_p`.
        angular_L: Total Cartesian angular momentum :math:`L_p`.
        tol: Numerical envelope-truncation tolerance.

    Returns:
        ndarray: Integer indices with shape ``(N_img, 3)``, dtype int64.
    """
    cell = np.asarray(cell, dtype=np.float64)
    r_cut = _primitive_image_radius(exponent, angular_L, tol)
    n = _image_candidate_grid(cell, r_cut)
    # Cell-to-cell closest approach: T_n vs the Minkowski difference
    # Omega - Omega (a parallelepiped of fractional side 2 centred at the
    # origin), i.e. the per-row fractional box [n - 1, n + 1].
    return _images_within_rcut_from_box(cell, r_cut, n, n - 1.0, n + 1.0)


def _primitive_image_list_indices_per_atom(
    cell: np.ndarray,
    exponent: float,
    angular_L: int,
    frac_pos: tuple[float, float, float] | np.ndarray,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> np.ndarray:
    r"""Integer image triplets for ONE atom's primitive image set (per-atom truncation).

    Tighter sibling of :py:func:`_primitive_image_list_indices` that replaces
    the cell-to-cell closest approach (``Omega - Omega``, fractional side 2)
    with the **atom-to-cell** closest approach (``Omega - R_A``, fractional
    side 1, shifted by the atom's fractional position ``s_A``). Keeps image
    ``n`` iff atom ``A``'s ``n``-th copy can come within ``r_cut`` of SOME
    electron in the home cell ``Omega = [0, 1)^3``:

    .. math:: \mathrm{dist}(T_n,\ \Omega - R_A) \le r_\mathrm{cut}.

    Because ``Omega - R_A`` is a subset of ``Omega - Omega`` for an atom inside
    ``Omega``, the per-atom set is then a subset of the cell-cell set; for a
    centred, localised shell it collapses to ``{T_0}``, whereas the cell-cell
    metric floors at the full 27-image nearest-neighbour block (every neighbour
    has cell-cell distance 0, so it is kept for any positive ``r_cut``). This
    is the floor-breaking lever for periodic AO image truncation.

    ``frac_pos`` is the atom's ACTUAL (unwrapped) fractional position, the one
    the kernel centres ``R_A + T_n`` are built from; it is not folded into
    ``Omega``. Writing ``s_A = w_A + f_A`` with ``f_A = floor(s_A)``, the box
    of image ``n`` equals the box of image ``m = n + f_A`` for the wrapped
    ``w_A``, so the selection runs on ``w_A`` (keeping the candidate grid
    centred) and returns ``m - f_A``. The kept set is the set of copies that
    reach the home cell, so it moves continuously with ``R_A`` across a cell
    face. Image ``-f_A`` is always present; ``T_0`` is present whenever the
    atom lies in ``Omega``.

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, Bohr.
        exponent: Primitive Gaussian exponent :math:`\alpha_p`.
        angular_L: Total Cartesian angular momentum :math:`L_p`.
        frac_pos: Atom fractional position ``s_A = R_A inv(cell)``, NOT wrapped.
        tol: Numerical envelope-truncation tolerance.

    Returns:
        ndarray: Integer indices with shape ``(N_img, 3)``, dtype int64.
    """
    cell = np.asarray(cell, dtype=np.float64)
    s_A = np.asarray(frac_pos, dtype=np.float64).reshape(3)
    f_A = np.floor(s_A)
    w_A = s_A - f_A  # in [0, 1] (1.0 only by rounding; the box test is valid there)
    r_cut = _primitive_image_radius(exponent, angular_L, tol)
    n = _image_candidate_grid(cell, r_cut)
    # Atom-to-cell closest approach: (electron in Omega) - R_A - T_n sweeps the
    # fractional box [-n - s_A, 1 - n - s_A] (side 1, shifted by the atom),
    # evaluated as the box of m = n + f_A for the wrapped w_A.
    kept = _images_within_rcut_from_box(cell, r_cut, n, -n - w_A, 1.0 - n - w_A)
    return kept - f_A.astype(np.int64)[None, :]


def _primitive_image_list(
    cell: np.ndarray,
    exponent: float,
    angular_L: int,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> np.ndarray:
    r"""Vectorised primitive image enumeration (production setup helper).

    See :py:meth:`Structure_data.primitive_image_list` for the contract. This
    free function is the numpy production implementation; the explicit-loop
    oracle :py:func:`_primitive_image_list_debug` sits beside it as the
    golden cross-check. Delegates index selection to
    :py:func:`_primitive_image_list_indices` and emits the Cartesian
    translations via the same per-axis weighted sum used by ``_debug`` ---
    routing through ``np.matmul`` / BLAS would introduce a different
    floating-point reduction order and break bit-for-bit agreement.

    Returns:
        ndarray: Translations :math:`T_n` with shape ``(N_img, 3)``, fp64.
    """
    cell = np.asarray(cell, dtype=np.float64)
    indices = _primitive_image_list_indices(cell, exponent, angular_L, tol)
    na_kept = indices[:, 0].astype(np.float64)
    nb_kept = indices[:, 1].astype(np.float64)
    nc_kept = indices[:, 2].astype(np.float64)
    return na_kept[:, None] * cell[0] + nb_kept[:, None] * cell[1] + nc_kept[:, None] * cell[2]


def _primitive_image_list_debug(
    cell: np.ndarray,
    exponent: float,
    angular_L: int,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> np.ndarray:
    r"""Reference (numpy, fp64, explicit loops) for ``_primitive_image_list``.

    Independent oracle (issue #66): it shares only the envelope radius
    :py:func:`_primitive_image_radius` with production. The candidate range
    and the closest-approach test are computed differently:

    * candidates: per-axis dual-norm bound. A kept image has a point
      ``x = v @ cell`` of its box with ``|x| <= r_cut``; ``v_k = x . g_k``
      with ``g_k = inv(cell)[:, k]`` gives ``|v_k| <= r_cut |g_k|`` and, since
      ``v_k`` lies in ``[n_k - 1, n_k + 1]``,
      ``|n_k| <= floor(r_cut |g_k|) + 1``;
    * distance: Cartesian geometry of the parallelepiped
      ``P_n = {v @ cell : v in [n - 1, n + 1]^3}``. The origin is inside iff
      every ``n_k - 1 <= 0 <= n_k + 1``; otherwise the closest point lies on
      one of the six faces, and the distance to a face (a parallelogram) is
      the perpendicular distance when the foot of the perpendicular lies on
      it, else the distance to the nearest of its four edges (point-to-segment
      with a clamped parameter).

    Explicit loops, fp64, no Gram matrix and no active-set enumeration; not
    consumed on any hot path.
    """
    cell = np.asarray(cell, dtype=np.float64)
    r_cut = _primitive_image_radius(exponent, angular_L, tol)
    inv_cell = np.linalg.inv(cell)
    bounds = [int(np.floor(r_cut * np.linalg.norm(inv_cell[:, k]))) + 1 for k in range(3)]

    def point_to_segment(start, edge):
        """Distance from the origin to the segment ``start + t edge``, ``t in [0, 1]``."""
        t = -float(start @ edge) / float(edge @ edge)
        t = max(0.0, min(1.0, t))
        return float(np.linalg.norm(start + t * edge))

    def point_to_parallelogram(corner, edge_1, edge_2):
        """Distance from the origin to ``corner + s edge_1 + t edge_2``, ``s, t in [0, 1]``."""
        m = np.array([[edge_1 @ edge_1, edge_1 @ edge_2], [edge_1 @ edge_2, edge_2 @ edge_2]])
        s, t = np.linalg.solve(m, -np.array([corner @ edge_1, corner @ edge_2]))
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return float(np.linalg.norm(corner + s * edge_1 + t * edge_2))
        return min(
            point_to_segment(corner, edge_1),
            point_to_segment(corner, edge_2),
            point_to_segment(corner + edge_1, edge_2),
            point_to_segment(corner + edge_2, edge_1),
        )

    images = []
    for na in range(-bounds[0], bounds[0] + 1):
        for nb in range(-bounds[1], bounds[1] + 1):
            for nc in range(-bounds[2], bounds[2] + 1):
                n = (na, nb, nc)
                low = [float(n_k) - 1.0 for n_k in n]
                high = [float(n_k) + 1.0 for n_k in n]
                if all(low[k] <= 0.0 <= high[k] for k in range(3)):
                    dist = 0.0
                else:
                    dist = np.inf
                    for axis in range(3):
                        j, k = (axis + 1) % 3, (axis + 2) % 3
                        edge_1 = (high[j] - low[j]) * cell[j]
                        edge_2 = (high[k] - low[k]) * cell[k]
                        for side in (low[axis], high[axis]):
                            corner = side * cell[axis] + low[j] * cell[j] + low[k] * cell[k]
                            dist = min(dist, point_to_parallelogram(corner, edge_1, edge_2))
                if dist <= r_cut:
                    images.append(na * cell[0] + nb * cell[1] + nc * cell[2])
    return np.asarray(images, dtype=np.float64)
