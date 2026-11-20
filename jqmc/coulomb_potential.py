"""Effective core potential module.

Module containing classes and methods related to Effective core potential
and bare Coulomb potentials

Precision Zones:
    - ``coulomb``: all functions in this module.

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

import itertools
from functools import partial
from logging import getLogger
from typing import NamedTuple, TypeAlias

# JAX
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import jit, lax, vmap
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erfc as jnp_erfc
from scipy.special import erfc as np_erfc
from scipy.special import eval_legendre

from contextlib import AbstractContextManager, nullcontext

from ._precision import cast_to_zone, get_dtype_jnp, get_dtype_np
from ._route_probe import note_route, route_site
from ._setting import ATOL_EWALD_CHARGE_NEUTRALITY, EPS_safe_distance, NN_default, Nv_default
from .determinant import (
    Det_candidate_mesh_state,
    Det_ratio_streaming_state,
    Det_streaming_state,
    _advance_det_candidate_mesh_state,
    _compute_ratio_determinant_from_winv_state,
    _compute_ratio_determinant_part_split_spin,
    _init_det_candidate_mesh_state,
    _init_det_winv_state,
    compute_det_geminal_all_elements,
    compute_geminal_all_elements,
)
from .jastrow_factor import (
    J1_candidate_mesh_state,
    J2_candidate_mesh_state,
    J3_candidate_mesh_state,
    J_DTN_candidate_mesh_state,
    Jastrow_three_body_streaming_state,
    _advance_j3_candidate_mesh_state,
    _advance_j1_candidate_mesh_state,
    _advance_j2_candidate_mesh_state,
    _advance_j_dtn_candidate_mesh_state,
    _compute_ratio_Jastrow_part_split_spin,
    _init_j3_candidate_mesh_state,
    _init_j1_candidate_mesh_state,
    _init_j2_candidate_mesh_state,
    _init_j_dtn_candidate_mesh_state,
    compute_Jastrow_part_ratio_zone,
)
from .structure import (
    Structure_data,
    _find_nearest_nucleus_indices_jnp,
    _det_3x3,
    _inv_3x3,
    _find_nearest_nucleus_indices_np,
    _get_min_dist_rel_R_cart_jnp,
    _get_min_dist_rel_R_cart_np,
    _minimum_image_displacement,
)
from .wavefunction import Wavefunction_data

# set logger
logger = getLogger("jqmc").getChild(__name__)

_ArrayPair: TypeAlias = tuple[jax.Array, jax.Array]
_EcpPaddedParameters: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_EcpMeshBlock: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
_EcpNonlocalResult: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


@jit
def _legendre_tablated(n: int, x: float) -> float:
    """Tabulated Legendre polynomials P_n(x) for n = 0..6 (cc-ECP angular momenta).

    See https://en.wikipedia.org/wiki/Legendre_polynomials . JAX-friendly
    (jnp.select over the supported degrees).
    """
    conditions = [n == 0, n == 1, n == 2, n == 3, n == 4, n == 5, n == 6]
    P_n = [
        1,
        x,
        1.0 / 2.0 * (3.0 * x**2 - 1),
        1.0 / 2.0 * (5.0 * x**3 - 3.0 * x),
        1.0 / 8.0 * (35.0 * x**4 - 30.0 * x**2 + 3.0),
        1.0 / 8.0 * (63.0 * x**5 - 70 * x**3 + 15 * x),
        1.0 / 16.0 * (231.0 * x**6 - 315.0 * x**4 + 105.0 * x**2 - 5),
    ]
    return jnp.select(conditions, P_n, jnp.nan)


# non local PPs, Mesh Info. taken from Mitas's paper [J. Chem. Phys., 95, 5, (1991)]
# why namedtuple? because it should be immutable
class _Mesh(NamedTuple):
    Nv: int
    weights: list[float]
    grid_points: npt.NDArray[np.float64]


# Tetrahedron symmetry quadrature (Nv=4)
q = 1 / np.sqrt(3)
A = 1.0 / 4.0
tetrahedron_sym_mesh_Nv4 = _Mesh(
    Nv=4,
    weights=[A, A, A, A],
    grid_points=np.array([[q, q, q], [q, -q, -q], [-q, q, -q], [-q, -q, q]]),
)

# Octahedron symmetry quadrature (Nv=6)
A = 1.0 / 6.0
octahedron_sym_mesh_Nv6 = _Mesh(
    Nv=6,
    weights=[A, A, A, A, A, A],
    grid_points=np.array(
        [
            [+1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, +1.0],
            [0.0, 0.0, -1.0],
        ]
    ),
)

# Icosahedron symmetry quadrature (Nv=12)
A = 1.0 / 12.0
B = 1.0 / 12.0
grid_points_sphe = np.array(
    [
        [0.0, 0.0],
        [np.pi, 0.0],
        [np.arctan(2), (2.0 * 0 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 1 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 2 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 3 * np.pi) / 5.0],
        [np.arctan(2), (2.0 * 4 * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 0) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 1) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 2) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 3) + 1.0) * np.pi) / 5.0],
        [np.pi - np.arctan(2), (((2.0 * 4) + 1.0) * np.pi) / 5.0],
    ]
)
theta = grid_points_sphe[:, 0]
phi = grid_points_sphe[:, 1]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
grid_points_cart = np.vstack((x, y, z)).T
icosahedron_sym_mesh_Nv12 = _Mesh(Nv=12, weights=[A, A, B, B, B, B, B, B, B, B, B, B], grid_points=grid_points_cart)

# Octahedron symmetry quadrature (Nv=18)
A = 1.0 / 30.0
B = 1.0 / 15.0
p = 1.0 / np.sqrt(2)
octahedron_sym_mesh_Nv18 = _Mesh(
    Nv=18,
    weights=[A, A, A, A, A, A, B, B, B, B, B, B, B, B, B, B, B, B],
    grid_points=np.array(
        [
            [+1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, +1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, +1.0],
            [0.0, 0.0, -1.0],
            [+p, +p, 0.0],
            [+p, -p, 0.0],
            [-p, +p, 0.0],
            [-p, -p, 0.0],
            [+p, 0.0, +p],
            [+p, 0.0, -p],
            [-p, 0.0, +p],
            [-p, 0.0, -p],
            [0.0, +p, +p],
            [0.0, -p, +p],
            [0.0, +p, -p],
            [0.0, -p, -p],
        ]
    ),
)


@struct.dataclass
class Ewald_data:
    r"""Precomputed Ewald-summation tables for periodic Coulomb evaluation.

    Populated by :py:func:`build_ewald_data` from a ``Structure_data`` and
    ``Coulomb_potential_data`` pair under tin-foil boundary conditions.
    Detailed Ewald parameter-selection and channel decomposition are work in
    progress; this container exposes the setup scalars and cached tables
    needed by the periodic Coulomb API.

    Attributes:
        eta (float): Ewald splitting parameter.
        r_real_cut (float): Real-space lattice-image cutoff in Bohr.
        g_max (float): Reciprocal-space cutoff in Bohr^-1.
        real_image_list (array-like | None): Cached Cartesian lattice-image
            list, shape ``(N_n, 3)``.
        g_vectors (array-like | None): Cached reciprocal vectors, shape
            ``(N_g, 3)``.
        g_factors (array-like | None): Cached per-G factors, shape ``(N_g,)``.
        ii_energy (float | None): Cached setup-time ion-ion energy in Ha.
        structure_data (Structure_data): Structure used at build time.

    Notes:
        * fp64 throughout (``ewald`` precision zone, see
          :mod:`jqmc._precision`) in both ``"full"`` and ``"mixed"`` modes.
        * HDF5 round-trip persists only ``(eta, r_real_cut, g_max)``, integer
          index tuples and ``structure_data``; heavy tables are rebuilt on
          load via :py:func:`build_ewald_data`.
        * The cached Cartesian fields describe the structure at build time.
          Runtime backends rebuild tables from the static integer index sets
          and the current ``structure_data`` cell, so moved ions or a strained
          cell are evaluated for the current structure.
    """

    #: Structure (positions + cell) the Ewald tables were built for.
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=Structure_data)
    #: Ewald split parameter (Bohr^-1). Larger eta narrows the real-space
    #: erfc and broadens the reciprocal Gaussian (exp(-G^2/4 eta^2)).
    eta: float = struct.field(pytree_node=True, default=0.0)
    #: Real-space image cutoff (Bohr). Lattice translations T_n with
    #: ``|T_n| <= r_real_cut`` are enumerated.
    r_real_cut: float = struct.field(pytree_node=True, default=0.0)
    #: Reciprocal-space cutoff (Bohr^-1). Reciprocal vectors G with
    #: ``|G| <= g_max`` (and ``G != 0``) are enumerated.
    g_max: float = struct.field(pytree_node=True, default=0.0)
    #: Real-space lattice translations T_n, shape ``(N_n, 3)``, fp64.
    #: Always includes ``T_0 = (0, 0, 0)``.
    real_image_list: jax.Array | None = struct.field(pytree_node=True, default=None)
    #: Non-zero reciprocal vectors G, shape ``(N_g, 3)``, fp64.
    g_vectors: jax.Array | None = struct.field(pytree_node=True, default=None)
    #: Per-G factor ``(4 pi / V) * exp(-G^2 / (4 eta^2)) / G^2``, shape ``(N_g,)``, fp64.
    g_factors: jax.Array | None = struct.field(pytree_node=True, default=None)
    #: Precomputed ion-ion Ewald energy (Ha, scalar fp64). Includes the
    #: ion self-energy ``-(eta / sqrt(pi)) * sum_I Z_I**2``.
    ii_energy: float = struct.field(pytree_node=True, default=0.0)
    #: Integer lattice indices for the real-space images, flat tuple of
    #: length ``3 * N_n`` (row-major over ``(N_n, 3)``). Static
    #: ``pytree_node=False`` so the index set is fixed at construction
    #: time --- cell-AD then differentiates ``T_n = idx @ cell`` through
    #: cell only. ``(0, 0, 0)`` first (matches the numpy enumerator).
    #: Empty tuple for default-constructed instances.
    real_image_indices: tuple[int, ...] = struct.field(pytree_node=False, default=())
    #: Integer reciprocal indices for the non-zero G-vectors, flat tuple
    #: of length ``3 * N_g`` (row-major over ``(N_g, 3)``). Same
    #: ``pytree_node=False`` rationale as :py:attr:`real_image_indices`.
    g_indices: tuple[int, ...] = struct.field(pytree_node=False, default=())

    @property
    def _real_image_indices_jnp(self) -> jax.Array:
        """``(N_n, 3) int32`` view of :py:attr:`real_image_indices`."""
        flat = jnp.asarray(self.real_image_indices, dtype=jnp.int32)
        return flat.reshape(-1, 3)

    @property
    def _g_indices_jnp(self) -> jax.Array:
        """``(N_g, 3) int32`` view of :py:attr:`g_indices`."""
        flat = jnp.asarray(self.g_indices, dtype=jnp.int32)
        return flat.reshape(-1, 3)

    def sanity_check(self) -> None:
        """Validate internal consistency.

        Heavy arrays may be ``None`` for default-constructed instances
        (the convention used while a ``Coulomb_potential_data`` is still
        being assembled before :py:func:`build_ewald_data` runs). Once
        populated, ``real_image_list`` shape ``(N_n, 3)``, ``g_vectors``
        shape ``(N_g, 3)``, and ``g_factors`` shape ``(N_g,)`` must
        agree on ``N_g``.

        Raises:
            ValueError: If field shapes or scalar types are inconsistent.
        """
        if not isinstance(self.eta, (float, np.floating, jnp.floating)):
            raise ValueError(f"eta = {type(self.eta)} must be a float.")
        if self.real_image_list is not None:
            arr = np.asarray(self.real_image_list)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"real_image_list.shape = {arr.shape} must be (N_n, 3).")
        if self.g_vectors is not None and self.g_factors is not None:
            gv = np.asarray(self.g_vectors)
            gf = np.asarray(self.g_factors)
            if gv.ndim != 2 or gv.shape[1] != 3:
                raise ValueError(f"g_vectors.shape = {gv.shape} must be (N_g, 3).")
            if gf.ndim != 1 or gf.shape[0] != gv.shape[0]:
                raise ValueError(f"g_factors.shape = {gf.shape} must be (N_g,) with N_g = {gv.shape[0]}.")
        self.structure_data.sanity_check()

    def _get_info(self) -> list[str]:
        """Return a human-readable summary."""
        info_lines = ["**" + self.__class__.__name__]
        info_lines.append(f"  eta = {float(self.eta):.6f} Bohr^-1")
        info_lines.append(f"  r_real_cut = {float(self.r_real_cut):.4f} Bohr")
        info_lines.append(f"  g_max = {float(self.g_max):.4f} Bohr^-1")
        if self.real_image_list is not None:
            n_images = int(np.asarray(self.real_image_list).shape[0])
            info_lines.append(f"  N_real_images = {n_images}")
        if self.g_vectors is not None:
            n_g = int(np.asarray(self.g_vectors).shape[0])
            info_lines.append(f"  N_g = {n_g}")
        info_lines.append(f"  ii_energy = {float(self.ii_energy):.10f} Ha")
        return info_lines

    def _logger_info(self) -> None:
        """Log the information from :py:meth:`_get_info` via logger.info."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def build(
        cls,
        *,
        structure_data: Structure_data,
        coulomb_potential_data: "Coulomb_potential_data",
        tol: float = 1e-10,
        eta_override: float | None = None,
        r_real_cut_override: float | None = None,
        g_max_override: float | None = None,
        n_electron: int | None = None,
    ) -> "Ewald_data":
        """Build a validated :class:`Ewald_data` for a periodic system.

        Blessed factory: delegates to :py:func:`build_ewald_data` (which
        selects ``(eta, r_real_cut, g_max)``, enumerates the real / reciprocal
        image sets, and precomputes the ion-ion energy), then runs
        :py:meth:`sanity_check`. Direct construction remains supported (used
        by the HDF5 loader, which rebuilds heavy tables separately).
        """
        data = build_ewald_data(
            structure_data,
            coulomb_potential_data,
            tol=tol,
            eta_override=eta_override,
            r_real_cut_override=r_real_cut_override,
            g_max_override=g_max_override,
            n_electron=n_electron,
        )
        data.sanity_check()
        return data


@struct.dataclass
class Coulomb_potential_data:
    """Container for bare Coulomb and effective core potential (ECP) parameters.

    Args:
        structure_data (Structure_data): Underlying nuclear geometry and metadata.
        ecp_flag (bool): Whether ECPs are present. When ``True``, all ECP arrays must be populated.
        z_cores (list[float] | tuple[float]): Core electrons removed per atom; length ``natom``.
        max_ang_mom_plus_1 (list[int] | tuple[int]): ``l_max + 1`` for each atom; length ``natom``.
        num_ecps (int): Total number of ECP projector terms across all atoms and angular momenta.
        ang_moms (list[int] | tuple[int]): Angular momentum ``l`` per ECP term; length ``num_ecps``.
        nucleus_index (list[int] | tuple[int]): Atom index per ECP term; length ``num_ecps``.
        exponents (list[float] | tuple[float]): Gaussian exponents per ECP term; length ``num_ecps``.
        coefficients (list[float] | tuple[float]): Prefactors per ECP term; length ``num_ecps``.
        powers (list[int] | tuple[int]): Polynomial powers per ECP term; length ``num_ecps``.

    Notes:
        - When ``ecp_flag`` is ``False``, all ECP-related sequences must be empty and ``num_ecps`` should be 0.
        - Arrays are stored as Python lists/tuples for pytrees; conversion to ``jax.Array`` happens in the compute kernels.
    """

    structure_data: Structure_data = struct.field(
        pytree_node=True, default_factory=Structure_data
    )  #: Nuclear geometry and atom metadata.
    ecp_flag: bool = struct.field(pytree_node=False, default=False)  #: Whether ECP parameters are active.
    z_cores: list[float] | tuple[float] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Core electrons removed per atom (len = natom).
    max_ang_mom_plus_1: list[int] | tuple[int] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: ``l_max + 1`` per atom (len = natom).
    num_ecps: int = struct.field(pytree_node=False, default=0)  #: Total ECP projector terms across all atoms.
    ang_moms: list[int] | tuple[int] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Angular momentum ``l`` per ECP term (len = num_ecps).
    nucleus_index: list[int] | tuple[int] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Atom index per ECP term (len = num_ecps).
    exponents: list[float] | tuple[float] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Gaussian exponents per ECP term (len = num_ecps).
    coefficients: list[float] | tuple[float] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Prefactors per ECP term (len = num_ecps).
    powers: list[int] | tuple[int] = struct.field(
        pytree_node=False, default_factory=tuple
    )  #: Polynomial powers per ECP term (len = num_ecps).
    #: Precomputed Ewald-summation tables (PBC only). ``None`` for
    #: ``structure_data.pbc_flag == False`` (molecular) and for periodic
    #: instances that have not yet been processed by :py:func:`build_ewald_data`.
    #: After ``read_trexio_file`` returns on a periodic fixture, this is
    #: guaranteed populated --- the loader enforces a strict
    #: ``pbc_flag == True iff ewald_data is not None`` contract.
    ewald_data: "Ewald_data | None" = struct.field(pytree_node=True, default=None)

    def sanity_check(self) -> None:
        """Check attributes of the class.

        This function checks the consistencies among the arguments.

        Raises:
            ValueError: If there is an inconsistency in a dimension of a given argument.
        """
        if self.ecp_flag:
            if len(self.z_cores) != self.structure_data.natom:
                raise ValueError("dim. of self.z_cores is wrong")
            if len(self.max_ang_mom_plus_1) != self.structure_data.natom:
                raise ValueError("dim. of self.max_ang_mom_plus_1 is wrong")
            if len(self.ang_moms) != self.num_ecps:
                raise ValueError("dim. of self.num_ecps is wrong")
            if len(self.nucleus_index) != self.num_ecps:
                raise ValueError("dim. of self.nucleus_index is wrong")
            if len(self.exponents) != self.num_ecps:
                raise ValueError("dim. of self.ang_moms is wrong")
            if len(self.coefficients) != self.num_ecps:
                raise ValueError("dim. of self.coefficients is wrong")
            if len(self.powers) != self.num_ecps:
                raise ValueError("dim. of self.powers is wrong")

            # Semantic check, not just dimensional. Nothing downstream can catch a
            # violation: the local/non-local split compares ang_mom against
            # max_ang_mom_plus_1 (so a uniformly shifted converter still splits
            # correctly), and the global-l_max padding in the
            # ``*_non_local_part_with_global_lmax_plus1`` views fills a missing l
            # with a zero-coefficient dummy. A QE->jqmc converter that counted the
            # local channel as an l value therefore ran to completion with an
            # empty, silently zero-padded l=1 channel, and only a spec-conforming
            # external reader noticed.
            #
            # What is pathological is a DECLARED l with no term at all -- the
            # atom's own l_max promises a projector that does not exist. What is
            # NOT pathological is a term whose coefficient happens to be 0.0:
            # PySCF emits exactly that as a placeholder for H/He, and the
            # cross-atom padding is built from such dummies by design. So the test
            # is on the presence of TERMS per l, never on their magnitude.
            ang = np.asarray(self.ang_moms, dtype=np.int64)
            nuc = np.asarray(self.nucleus_index, dtype=np.int64)
            for i_atom in range(self.structure_data.natom):
                tags = ang[nuc == i_atom]
                if tags.size == 0:
                    continue
                lmax_p1 = int(self.max_ang_mom_plus_1[i_atom])
                if not np.any(tags == lmax_p1):
                    raise ValueError(
                        f"atom {i_atom}: no ECP term tagged with "
                        f"max_ang_mom_plus_1 = {lmax_p1}, so the local channel is "
                        f"missing. Tags present: {sorted(set(int(t) for t in tags))}. "
                        "The local channel must be tagged l_max + 1."
                    )
                missing = sorted(set(range(lmax_p1)) - {int(t) for t in tags})
                if missing:
                    raise ValueError(
                        f"atom {i_atom}: max_ang_mom_plus_1 = {lmax_p1} declares "
                        f"non-local channels l = 0..{lmax_p1 - 1}, but l = {missing} "
                        "carry no terms at all. A declared l must have terms (their "
                        "coefficients may legitimately be 0.0 -- that is how PySCF "
                        "writes H/He placeholders -- but the terms must exist). "
                        "Note l_max counts the NON-LOCAL channels only."
                    )
                if tags.max() > lmax_p1:
                    raise ValueError(
                        f"atom {i_atom}: ang_mom {int(tags.max())} exceeds "
                        f"max_ang_mom_plus_1 = {lmax_p1}; no tag may sit above the "
                        "local channel."
                    )
        else:
            if len(self.z_cores) != 0:
                raise ValueError("dim. of self.z_cores is wrong")
            if len(self.max_ang_mom_plus_1) != 0:
                raise ValueError("dim. of self.max_ang_mom_plus_1 is wrong")
            if len(self.ang_moms) != 0:
                raise ValueError("dim. of self.ang_moms is wrong")
            if len(self.nucleus_index) != 0:
                raise ValueError("dim. of self.nucleus_index is wrong")
            if len(self.exponents) != 0:
                raise ValueError("dim. of self.exponents is wrong")
            if len(self.coefficients) != 0:
                raise ValueError("dim. of self.coefficients is wrong")
            if len(self.powers) != 0:
                raise ValueError("dim. of self.powers is wrong")

        if not isinstance(self.ecp_flag, bool):
            raise ValueError(f"ecp_flag = {type(self.ecp_flag)} must be a bool.")
        if not isinstance(self.z_cores, (list, tuple)):
            logger.warning(f"z_cores = {type(self.z_cores)} should be a list or tuple.")
        if not isinstance(self.max_ang_mom_plus_1, (list, tuple)):
            logger.warning(f"max_ang_mom_plus_1 = {type(self.max_ang_mom_plus_1)} should be a list or tuple.")
        if not isinstance(self.num_ecps, (int, np.integer)):
            raise ValueError(f"num_ecps = {type(self.num_ecps)} must be an int.")
        if not isinstance(self.ang_moms, (list, tuple)):
            logger.warning(f"ang_moms = {type(self.ang_moms)} should be a list or tuple.")
        if not isinstance(self.exponents, (list, tuple)):
            logger.warning(f"exponents = {type(self.exponents)} should be a list or tuple.")
        if not isinstance(self.coefficients, (list, tuple)):
            logger.warning(f"coefficients = {type(self.coefficients)} should be a list or tuple.")
        if not isinstance(self.powers, (list, tuple)):
            logger.warning(f"powers = {type(self.powers)} should be a list or tuple.")

        # PBC strict contract: structure_data.pbc_flag == True iff ewald_data
        # is populated. Loaders (read_trexio_file, Hamiltonian_data.load_from_hdf5)
        # enforce this on construction; sanity_check warns when a user-built
        # instance forgot the build_ewald_data step.
        if self.structure_data.pbc_flag and self.ewald_data is None:
            logger.warning(
                "Coulomb_potential_data has structure_data.pbc_flag=True but "
                "ewald_data is None; call build_ewald_data(structure_data, "
                "self) before any periodic Coulomb evaluation."
            )
        if self.ewald_data is not None:
            self.ewald_data.sanity_check()

        self.structure_data.sanity_check()

    @classmethod
    def build(
        cls,
        *,
        structure_data: Structure_data,
        ecp_flag: bool = False,
        z_cores: list[float] | tuple[float, ...] = (),
        max_ang_mom_plus_1: list[int] | tuple[int, ...] = (),
        num_ecps: int = 0,
        ang_moms: list[int] | tuple[int, ...] = (),
        nucleus_index: list[int] | tuple[int, ...] = (),
        exponents: list[float] | tuple[float, ...] = (),
        coefficients: list[float] | tuple[float, ...] = (),
        powers: list[int] | tuple[int, ...] = (),
        ewald_data: "Ewald_data | None" = None,
    ) -> "Coulomb_potential_data":
        """Build a validated :class:`Coulomb_potential_data`.

        Blessed factory: assembles the bare-Coulomb / ECP fields, constructs
        the instance, and runs :py:meth:`sanity_check`. For a periodic system
        pass ``ewald_data`` (built via :py:meth:`Ewald_data.build` /
        :py:func:`build_ewald_data`) so the strict
        ``pbc_flag == True iff ewald_data is not None`` contract holds.
        Direct construction (``Coulomb_potential_data(...)``) remains supported.
        """
        data = cls(
            structure_data=structure_data,
            ecp_flag=ecp_flag,
            z_cores=z_cores,
            max_ang_mom_plus_1=max_ang_mom_plus_1,
            num_ecps=num_ecps,
            ang_moms=ang_moms,
            nucleus_index=nucleus_index,
            exponents=exponents,
            coefficients=coefficients,
            powers=powers,
            ewald_data=ewald_data,
        )
        data.sanity_check()
        return data

    def _get_info(self) -> list[str]:
        """Return a list of strings containing the attribute information."""
        info_lines = ["**" + self.__class__.__name__]
        info_lines.append(f"  ecp_flag = {self.ecp_flag}")
        if self.ewald_data is not None:
            info_lines.extend(self.ewald_data._get_info())
        return info_lines

    def _logger_info(self) -> None:
        """Log the information from get_info() using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def _effective_charges(self) -> npt.NDArray:
        """effective_charges.

        Return nucleus charge (all-electron) or effective charge (with ECP)

        Return:
            npt.NDAarray: nucleus charge (effective charge)
        """
        if self.ecp_flag:
            return np.array(self.structure_data.atomic_numbers) - np.array(self.z_cores)
        return np.array(self.structure_data.atomic_numbers)

    @property
    def pseudized_per_atom(self) -> npt.NDArray[np.bool_]:
        """Return whether each nucleus is represented by an ECP.

        ECP identity cannot be inferred from the removed core charge: a
        zero-core pseudopotential such as H-ccECP has ``z_core == 0`` but its
        local potential is finite at the nucleus.  The authoritative signal
        is therefore the presence of an ECP angular-momentum channel or an
        ECP term carrying that nucleus index.
        """
        pseudized = np.zeros(self.structure_data.natom, dtype=np.bool_)
        if not self.ecp_flag:
            return pseudized
        pseudized |= np.asarray(self.max_ang_mom_plus_1, dtype=np.int64) > 0
        for atom_index in self.nucleus_index:
            if not 0 <= int(atom_index) < self.structure_data.natom:
                raise ValueError(f"ECP nucleus_index {atom_index} is outside [0, {self.structure_data.natom}).")
            pseudized[int(atom_index)] = True
        return pseudized

    @property
    def _global_max_ang_mom_plus_1(self) -> int:
        """The maximum number of ang_mom_plus_1 among all atoms."""
        return np.max(self.max_ang_mom_plus_1)

    def _validate_xla_projector_channels(self) -> None:
        """Reject unsupported XLA projectors using static ECP metadata at trace time.

        Local-channel tags are not Legendre degrees. Other backends need not
        share XLA's tabulated degree limit, so this is not a container invariant.
        Check declared degrees too: padding materializes every declared channel,
        including zero-coefficient ones for which zero times NaN is still NaN.
        """
        nonlocal_degrees = self._ang_mom_non_local_part
        if any(tag < 0 or tag > 7 for tag in self.max_ang_mom_plus_1) or np.any(
            (nonlocal_degrees < 0) | (nonlocal_degrees > 6)
        ):
            raise NotImplementedError(
                "XLA non-local ECP supports only l=0..6; "
                f"local-channel tags={self.max_ang_mom_plus_1}, "
                f"non-local degrees={nonlocal_degrees.tolist()}."
            )

    @property
    def _ang_mom_local_part(self) -> npt.NDArray:
        """ang_mom_local_part.

        Return angular momentum of the local part (i.e., = max_ang_mom_plus1)

        Return:
            npt.NDAarray: momentum of the local part (effective charge)
        """
        return np.array(self.max_ang_mom_plus_1)

    @property
    def _ang_mom_non_local_part(self) -> npt.NDArray:
        """ang_mom_non_local_part.

        Return angular momentum of the non_local part (i.e., = max_ang_mom_plus1)

        Return:
            npt.NDAarray: momentum of the non_local part (effective charge)
        """
        return np.array(self.ang_moms)[self._non_local_part_index]

    @property
    def _local_part_index(self) -> npt.NDArray:
        """local_part_index.

        Return a list containing index of the local part

        Return:
            npt.NDAarray: a list containing index of the local part
        """
        local_part_index = np.array(
            [
                i
                for i, v in enumerate(self.nucleus_index)
                if v in range(self.structure_data.natom) and self.ang_moms[i] == self.max_ang_mom_plus_1[v]
            ],
            dtype=np.int64,
        )
        return local_part_index

    @property
    def _non_local_part_index(self) -> npt.NDArray:
        """non_local_part_index.

        Return a list containing index of the non-local part

        Return:
            npt.NDAarray: a list containing index of the non-local part
        """
        non_local_part_index = np.array(
            [
                i
                for i, v in enumerate(self.nucleus_index)
                if v in range(self.structure_data.natom) and self.ang_moms[i] != self.max_ang_mom_plus_1[v]
            ],
            dtype=np.int64,
        )
        return non_local_part_index

    @property
    def _nucleus_index_local_part(self) -> npt.NDArray:
        """nucleus_index local_part.

        Return a list containing nucleus_index of the local part

        Return:
            npt.NDAarray: a list containing nucleus_index of the local part
        """
        return np.array(self.nucleus_index)[self._local_part_index]

    @property
    def _nucleus_index_non_local_part(self) -> npt.NDArray:
        """nucleus_index non_local_part.

        Return a list containing nucleus_index of the non-local part

        Return:
            npt.NDAarray: a list containing nucleus_index of the non-local part
        """
        return np.array(self.nucleus_index)[self._non_local_part_index]

    @property
    def _exponents_local_part(self) -> npt.NDArray:
        """Exponents local_part.

        Return a list containing exponents of the local part

        Return:
            npt.NDAarray: a list containing exponents of the local part
        """
        return np.array(self.exponents)[self._local_part_index]

    @property
    def _exponents_non_local_part(self) -> npt.NDArray:
        """Exponents non_local_part.

        Return a list containing exponents of the non-local part

        Return:
            npt.NDAarray: a list containing exponents of the non-local part
        """
        return np.array(self.exponents)[self._non_local_part_index]

    @property
    def _coefficients_local_part(self) -> npt.NDArray:
        """Coefficients local_part.

        Return a list containing coefficients of the local part

        Return:
            npt.NDAarray: a list containing coefficients of the local part
        """
        return np.array(self.coefficients)[self._local_part_index]

    @property
    def _coefficients_non_local_part(self) -> npt.NDArray:
        """Coefficients non_local_part.

        Return a list containing coefficients of the non-local part

        Return:
            npt.NDAarray: a list containing coefficients of the non-local part
        """
        return np.array(self.coefficients)[self._non_local_part_index]

    @property
    def _powers_local_part(self) -> npt.NDArray:
        """Powers local_part.

        Return a list containing powers of the local part

        Return:
            npt.NDAarray: a list containing powers of the local part
        """
        return np.array(self.powers)[self._local_part_index]

    @property
    def _powers_non_local_part(self) -> npt.NDArray:
        """Powers non_local_part.

        Return a list containing powers of the non-local part

        Return:
            npt.NDAarray: a list containing powers of the non-local part
        """
        return np.array(self.powers)[self._non_local_part_index]

    @property
    def _n_atom(self) -> int:
        """Number of atoms in the system (rows of the padded per-atom ECP tables).

        Sized from the structure, not from the largest ECP nucleus index: an
        all-electron atom listed after the last ECP atom must still own a row
        (of zero coefficients), or a gather at its index clamps to another
        atom's ECP parameters.
        """
        return int(self.structure_data.natom)

    @property
    def _non_local_ecp_atom_mask(self) -> npt.NDArray[np.bool_]:
        """Per-atom mask of the nuclei that carry a non-local ECP channel.

        These are the candidate centres of the nearest-neighbour non-local
        projector: an electron projects onto its NN nearest nuclei among them,
        never onto an all-electron nucleus (which projects nothing).
        """
        mask = np.zeros(self.structure_data.natom, dtype=np.bool_)
        mask[np.asarray(self._nucleus_index_non_local_part, dtype=np.int64)] = True
        return mask

    @property
    def _padded_parameters_tuple(self) -> _EcpPaddedParameters:
        """Padding parameters for jit(vmap).

        Ensure that each atom has the global max ang_mom and the same number of
        params for jit vmap. Padded jnp.arrays are returned.
        If an atom's local max ang_mom is less than the global max,
        append a dummy element with ang_mom = global_ang_mom_non_local_part.

        Returns updated lists of:

            - ang_mom_non_local_part_padded_jnp
            - exponents_non_local_part_padded_jnp
            - coefficients_non_local_part_padded_jnp
            - powers_non_local_part_padded_jnp

        so that each atom's maximum ang_mom matches the global max,
        and the same dim. for i_atom (nucleus index).
        """
        # 1) One group per atom of the structure; atoms without non-local terms
        #    (all-electron or local-only) keep an empty group -> zero-coefficient row.
        n_atom = self._n_atom

        # 2) Prepare a temporary data structure to group items by atom
        #    grouped[i_atom] = [(ang_mom, exponent), (ang_mom, exponent), ...]
        grouped = [[] for _ in range(n_atom)]
        for i in range(len(self._nucleus_index_non_local_part)):
            i_atom = self._nucleus_index_non_local_part[i]
            l_val = self._ang_mom_non_local_part[i]
            expo = self._exponents_non_local_part[i]
            coeff = self._coefficients_non_local_part[i]
            power = self._powers_non_local_part[i]
            grouped[i_atom].append((l_val, expo, coeff, power))

        # 3) For each atom, if the local max ang_mom is less than the global, append one dummy element
        for i_atom in range(n_atom):
            if len(grouped[i_atom]) == 0:
                # If the atom has no elements at all, skip or add one by choice
                continue

            local_max = max(pair[0] for pair in grouped[i_atom])
            if local_max < self._global_max_ang_mom_plus_1:
                # Append one element with ang_mom = global and expo, coeff, power = 0.0
                grouped[i_atom].append((self._global_max_ang_mom_plus_1, 0.0, 0.0, 0.0))

        # 4) Flatten the data structure back into lists
        nucleus_index_with_global_lmax_plus1 = []
        new_ang_mom = []
        new_exponents = []
        new_coefficients = []
        new_powers = []
        for i_atom in range(n_atom):
            for l_val, expo, coeff, power in grouped[i_atom]:
                nucleus_index_with_global_lmax_plus1.append(i_atom)
                new_ang_mom.append(l_val)
                new_exponents.append(expo)
                new_coefficients.append(coeff)
                new_powers.append(power)

        nucleus_index_with_global_lmax_plus1 = np.array(nucleus_index_with_global_lmax_plus1)
        ang_mom_with_global_lmax_plus1 = np.array(new_ang_mom)
        exponents_global_lmax_plus1 = np.array(new_exponents)
        coefficients_global_lmax_plus1 = np.array(new_coefficients)
        powers_global_lmax_plus1 = np.array(new_powers)

        # 5) count the max of i_atom appearance (i.e., max num of params.)
        counts = np.bincount(nucleus_index_with_global_lmax_plus1)
        max_param_num = counts.max()

        # 6) padding ang_mom_non_local_part_padded_np
        ang_mom_non_local_part_padded_np = np.zeros((self._n_atom, max_param_num), dtype=ang_mom_with_global_lmax_plus1.dtype)
        row_counts = np.zeros((self._n_atom,), dtype=np.int32)
        for i in range(ang_mom_with_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            ang_mom_non_local_part_padded_np[i_atom, j] = ang_mom_with_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 7) padding exponents_non_local_part_padded_np
        exponents_non_local_part_padded_np = np.zeros((self._n_atom, max_param_num), dtype=exponents_global_lmax_plus1.dtype)
        row_counts = np.zeros((self._n_atom,), dtype=np.int32)
        for i in range(exponents_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            exponents_non_local_part_padded_np[i_atom, j] = exponents_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 8) padding coefficients_non_local_part_padded_np
        coefficients_non_local_part_padded_np = np.zeros(
            (self._n_atom, max_param_num), dtype=coefficients_global_lmax_plus1.dtype
        )
        row_counts = np.zeros((self._n_atom,), dtype=np.int32)
        for i in range(coefficients_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            coefficients_non_local_part_padded_np[i_atom, j] = coefficients_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        # 9) padding coefficients_non_local_part_padded_np
        powers_non_local_part_padded_np = np.zeros((self._n_atom, max_param_num), dtype=powers_global_lmax_plus1.dtype)
        row_counts = np.zeros((self._n_atom,), dtype=np.int32)
        for i in range(powers_global_lmax_plus1.shape[0]):
            i_atom = nucleus_index_with_global_lmax_plus1[i]
            j = row_counts[i_atom]
            powers_non_local_part_padded_np[i_atom, j] = powers_global_lmax_plus1[i]
            row_counts[i_atom] += 1

        ang_mom_non_local_part_padded_jnp = jnp.array(ang_mom_non_local_part_padded_np)
        exponents_non_local_part_padded_jnp = jnp.array(exponents_non_local_part_padded_np)
        coefficients_non_local_part_padded_jnp = jnp.array(coefficients_non_local_part_padded_np)
        powers_non_local_part_padded_jnp = jnp.array(powers_non_local_part_padded_np)

        return (
            ang_mom_non_local_part_padded_jnp,
            exponents_non_local_part_padded_jnp,
            coefficients_non_local_part_padded_jnp,
            powers_non_local_part_padded_jnp,
        )


def _find_nearest_ecp_nucleus_indices_np(
    coulomb_potential_data: Coulomb_potential_data,
    r_cart: npt.ArrayLike,
    NN: int,
) -> npt.NDArray[np.intp]:
    """Return the NN projector centres of one electron: the nearest nuclei carrying a non-local ECP channel.

    Nuclei with a non-local channel are ranked first, by increasing (minimum-image)
    distance; ties keep the lower index. An all-electron nucleus never hosts a
    projector, so it must not displace an ECP centre even when it is nearer. Only
    when NN exceeds the number of ECP nuclei do the remaining slots fall to the
    other nuclei, whose zero-coefficient rows in the padded tables project nothing
    (this keeps the result shape ``(NN,)`` static).
    """
    structure_data = coulomb_potential_data.structure_data
    order = _find_nearest_nucleus_indices_np(
        structure_data=structure_data, r_cart=r_cart, number_of_nuclei=structure_data.natom
    )
    is_ecp = coulomb_potential_data._non_local_ecp_atom_mask[order]
    ecp_first = np.argsort(~is_ecp, kind="stable")
    return order[ecp_first][:NN]


def _find_nearest_ecp_nucleus_indices_jnp(
    coulomb_potential_data: Coulomb_potential_data,
    r_cart: jax.Array,
    NN: int,
) -> jax.Array:
    """JAX twin of :func:`_find_nearest_ecp_nucleus_indices_np` (static result shape)."""
    structure_data = coulomb_potential_data.structure_data
    order = _find_nearest_nucleus_indices_jnp(
        structure_data=structure_data, r_cart=r_cart, number_of_nuclei=structure_data.natom
    )
    is_ecp = jnp.asarray(coulomb_potential_data._non_local_ecp_atom_mask)[order]
    ecp_first = jnp.argsort(jnp.logical_not(is_ecp).astype(jnp.int32), stable=True)
    return order[ecp_first][:NN]


def _compute_ecp_local_parts_all_pairs_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Compute ecp local parts.

    The method is for computing the local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)

    Returns:
        float: The sum of local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). The
    # accumulated scalar V_local is cast to the coulomb zone before return.
    dtype_np = get_dtype_np("coulomb")

    V_local = 0.0
    for i_atom in range(coulomb_potential_data.structure_data.natom):
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom]
        ang_moms = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
        exponents = [coulomb_potential_data.exponents[i] for i in nucleus_indices]
        coefficients = [coulomb_potential_data.coefficients[i] for i in nucleus_indices]
        powers = [coulomb_potential_data.powers[i] for i in nucleus_indices]
        ang_mom_indices = [i for i, v in enumerate(ang_moms) if v == max_ang_mom_plus_1]
        exponents = [exponents[i] for i in ang_mom_indices]
        coefficients = [coefficients[i] for i in ang_mom_indices]
        powers = [powers[i] for i in ang_mom_indices]

        for r_up_cart in r_up_carts:
            rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_up_cart,
                i_atom=i_atom,
            )
            V_local += np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                [
                    a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents, strict=True)
                ]
            )
        for r_dn_cart in r_dn_carts:
            rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_dn_cart,
                i_atom=i_atom,
            )
            V_local += np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                [
                    a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                    for a, n, b in zip(coefficients, powers, exponents, strict=True)
                ]
            )
    # Cast accumulator to coulomb zone (Principle 3b).
    return np.asarray(V_local, dtype=dtype_np)


def _compute_ecp_non_local_parts_all_pairs_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> _EcpNonlocalResult:
    """Compute ecp non-local parts, considering all nucleus-electron pairs.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[npt.NDArray]: The list of grids for up electrons on which the non-local part is computed.
        list[npt.NDArray]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
    """
    # Preserve the caller's coordinate arrays and cast the rotation matrix only
    # where it enters the Coulomb-zone grid calculation.
    dtype_np = get_dtype_np("coulomb")

    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    # Cast at the use site so upstream coordinate precision remains unchanged.
    grid_points = grid_points @ np.asarray(RT, dtype=dtype_np)  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    mesh_non_local_ecp_part = []
    V_nonlocal = []
    sum_V_nonlocal = 0.0

    if flag_determinant_only:
        jastrow_denominator = 1.0
    else:
        jastrow_denominator = compute_Jastrow_part_ratio_zone(
            jastrow_data=wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

    det_denominator = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    for i_atom in range(coulomb_potential_data.structure_data.natom):
        max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
        nucleus_indices = [i for i, v in enumerate(coulomb_potential_data.nucleus_index) if v == i_atom]

        ang_moms_all = [coulomb_potential_data.ang_moms[i] for i in nucleus_indices]
        exponents_all = [coulomb_potential_data.exponents[i] for i in nucleus_indices]
        coefficients_all = [coulomb_potential_data.coefficients[i] for i in nucleus_indices]
        powers_all = [coulomb_potential_data.powers[i] for i in nucleus_indices]

        for ang_mom in range(max_ang_mom_plus_1):
            ang_mom_indices = [i for i, v in enumerate(ang_moms_all) if v == ang_mom]
            exponents = [exponents_all[i] for i in ang_mom_indices]
            coefficients = [coefficients_all[i] for i in ang_mom_indices]
            powers = [powers_all[i] for i in ang_mom_indices]

            # up electrons
            for r_up_i, r_up_cart in enumerate(r_up_carts):
                rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_up_cart,
                    i_atom=i_atom,
                )
                V_l = np.linalg.norm(rel_R_cart_min_dist) ** -2.0 * np.sum(
                    [
                        a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
                        for a, n, b in zip(coefficients, powers, exponents, strict=True)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points, strict=True):
                    r_up_carts_on_mesh = r_up_carts.copy()
                    r_up_carts_on_mesh[r_up_i] = (
                        r_up_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        ((vec_delta) / np.linalg.norm(vec_delta)),
                    )

                    if flag_determinant_only:
                        jastrow_numerator = 1
                    else:
                        jastrow_numerator = compute_Jastrow_part_ratio_zone(
                            jastrow_data=wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts_on_mesh,
                            r_dn_carts=r_dn_carts,
                        )

                    det_numerator = compute_det_geminal_all_elements(
                        geminal_data=wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts_on_mesh,
                        r_dn_carts=r_dn_carts,
                    )

                    wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio

                    mesh_non_local_ecp_part.append((r_up_carts_on_mesh, r_dn_carts))
                    V_nonlocal.append(V_l * P_l)
                    sum_V_nonlocal += V_l * P_l

            # dn electrons
            for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
                rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_dn_cart,
                    i_atom=i_atom,
                )
                V_l = np.linalg.norm(rel_R_cart_min_dist) ** -2 * np.sum(
                    [
                        a * np.linalg.norm(rel_R_cart_min_dist) ** n * np.exp(-b * np.linalg.norm(rel_R_cart_min_dist) ** 2)
                        for a, n, b in zip(coefficients, powers, exponents, strict=True)
                    ]
                )

                for weight, vec_delta in zip(weights, grid_points, strict=True):
                    r_dn_carts_on_mesh = r_dn_carts.copy()
                    r_dn_carts_on_mesh[r_dn_i] = (
                        r_dn_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                    )

                    cos_theta = np.dot(
                        -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                        vec_delta / np.linalg.norm(vec_delta),
                    )

                    if flag_determinant_only:
                        jastrow_numerator = 1.0
                    else:
                        jastrow_numerator = compute_Jastrow_part_ratio_zone(
                            jastrow_data=wavefunction_data.jastrow_data,
                            r_up_carts=r_up_carts,
                            r_dn_carts=r_dn_carts_on_mesh,
                        )
                    det_numerator = compute_det_geminal_all_elements(
                        geminal_data=wavefunction_data.geminal_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts_on_mesh,
                    )

                    wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator

                    P_l = (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                    mesh_non_local_ecp_part.append((r_up_carts, r_dn_carts_on_mesh))
                    V_nonlocal.append(V_l * P_l)
                    sum_V_nonlocal += V_l * P_l

    mesh_non_local_ecp_part_r_up_carts = np.array([up for up, _ in mesh_non_local_ecp_part])
    mesh_non_local_ecp_part_r_dn_carts = np.array([dn for _, dn in mesh_non_local_ecp_part])
    V_nonlocal = np.array(V_nonlocal)

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


def _compute_ecp_non_local_parts_nearest_neighbors_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> _EcpNonlocalResult:
    """Compute ecp non-local parts.

    The method is for computing the non-local part of the given ECPs at (r_up_carts, r_dn_carts)
    with a cutoff considering only up to NN-th nearest neighbors. A very straightforward (so very slow) implementation.
    Just for debudding purpose.

    Convention: the ratios are the holomorphic ``exp(J' - J) * det' / det``, so for a
    complex-Bloch (twist) wavefunction ``V_nonlocal`` and ``sum_V_nonlocal`` are complex
    and equal :func:`compute_ecp_non_local_parts_nearest_neighbors_holomorphic`. The
    fixed-phase production entry :func:`compute_ecp_non_local_parts_nearest_neighbors`
    returns ``Re`` of each per-grid term. Quadrature points are not wrapped into the home
    cell here (see the comment in the body).

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.
        flag_determinant_only (bool): If True, only the determinant part is considered for the non-local ECP part.

    Returns:
        list[npt.NDArray]: The list of grids for up electrons on which the non-local part is computed.
        list[npt.NDArray]: The list of grids for dn electrons on which the non-local part is computed.
        list[float]: The list of non-local part of the given ECPs with r_up_carts and r_dn_carts.
        float: sum of the V_nonlocal
    """
    # Preserve the caller's coordinate arrays and cast the rotation matrix only
    # where it enters the Coulomb-zone grid calculation.
    dtype_np = get_dtype_np("coulomb")

    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    # Cast at the use site so upstream coordinate precision remains unchanged.
    grid_points = grid_points @ np.asarray(RT, dtype=dtype_np)  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    # PBC quadrature points r_q = R_I^nearest + |r_iI| * Omega_q can land
    # outside the home parallelepiped. They are passed to Psi UNWRAPPED, as in
    # the production helpers compute_ecp_non_local_parts_nearest_neighbors{,
    # _fast_update}: the home-cell wrap is centralised in compute_AOs, which
    # also restores the Bloch phase exp(i k.T) of the lattice vector T it
    # removes. An explicit wrap here (a frac-floor) would move r_q by T BEFORE
    # that guarantor runs, so at a non-Gamma twist every wrapped mesh value
    # would carry a spurious exp(-i k.T) and the ratio Psi(r_q)/Psi(r) would be
    # wrong (WF-PATTERN-008: Re sum 0.018 against 0.061). Such a wrap is a no-op
    # at Gamma and for molecules.

    V_nonlocal = []
    sum_V_nonlocal = 0.0

    if flag_determinant_only:
        jastrow_denominator = 1.0
    else:
        jastrow_denominator = compute_Jastrow_part_ratio_zone(
            jastrow_data=wavefunction_data.jastrow_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

    det_denominator = compute_det_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )

    i_atom_np = np.array(coulomb_potential_data._nucleus_index_non_local_part)
    ang_mom_np = np.array(coulomb_potential_data._ang_mom_non_local_part)
    exponent_np = np.array(coulomb_potential_data._exponents_non_local_part)
    coefficient_np = np.array(coulomb_potential_data._coefficients_non_local_part)
    power_np = np.array(coulomb_potential_data._powers_non_local_part)

    # up electrons
    up_mesh_non_local_ecp_part_up = []
    up_mesh_non_local_ecp_part_dn = []

    for r_up_i, r_up_cart in enumerate(r_up_carts):
        i_atom_list = _find_nearest_ecp_nucleus_indices_np(
            coulomb_potential_data=coulomb_potential_data,
            r_cart=r_up_cart,
            NN=NN,
        )

        for i_atom in i_atom_list:
            rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_up_cart,
                i_atom=i_atom,
            )

            max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
            i_target = [i for i, v in enumerate(i_atom_np) if v == i_atom]
            ang_moms = ang_mom_np[i_target]
            exponents = exponent_np[i_target]
            coefficients = coefficient_np[i_target]
            powers = power_np[i_target]

            # dim (#ang_moms, 1) # note: ang_moms include the same l because each l has seveal (exp, coeff, and powers).
            # jax.ops.segsum cares about it later.
            V_l_list = (
                np.linalg.norm(rel_R_cart_min_dist) ** -2.0
                * coefficients
                * np.linalg.norm(rel_R_cart_min_dist) ** powers
                * np.exp(-exponents * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
            )

            weight_list = []
            cos_theta_list = []
            wf_ratio_list = []
            for weight, vec_delta in zip(weights, grid_points, strict=True):
                weight_list.append(weight)
                r_up_carts_on_mesh = r_up_carts.copy()
                new_pos_up = r_up_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                r_up_carts_on_mesh[r_up_i] = new_pos_up
                up_mesh_non_local_ecp_part_up.append(r_up_carts_on_mesh)
                up_mesh_non_local_ecp_part_dn.append(r_dn_carts)

                cos_theta = np.dot(
                    -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                    ((vec_delta) / np.linalg.norm(vec_delta)),
                )
                cos_theta_list.append(cos_theta)

                if flag_determinant_only:
                    jastrow_numerator = 1.0
                else:
                    jastrow_numerator = compute_Jastrow_part_ratio_zone(
                        jastrow_data=wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts_on_mesh,
                        r_dn_carts=r_dn_carts,
                    )

                det_numerator = compute_det_geminal_all_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts_on_mesh,
                    r_dn_carts=r_dn_carts,
                )

                wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator
                wf_ratio_list.append(wf_ratio)

            for ang_mom in range(max_ang_mom_plus_1):
                ll_target = [i for i, v in enumerate(ang_moms) if v == ang_mom]
                V_l = np.sum([V_l_list[ll] for ll in ll_target])
                P_l = np.array(
                    [
                        (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                        for cos_theta, weight, wf_ratio in zip(cos_theta_list, weight_list, wf_ratio_list, strict=True)
                    ]
                )
                ans = list(V_l * P_l)
                V_nonlocal += ans
                sum_V_nonlocal += np.sum(ans)

    # dn electrons
    dn_mesh_non_local_ecp_part_up = []
    dn_mesh_non_local_ecp_part_dn = []

    for r_dn_i, r_dn_cart in enumerate(r_dn_carts):
        i_atom_list = _find_nearest_ecp_nucleus_indices_np(
            coulomb_potential_data=coulomb_potential_data,
            r_cart=r_dn_cart,
            NN=NN,
        )

        for i_atom in i_atom_list:
            rel_R_cart_min_dist = _get_min_dist_rel_R_cart_np(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r_dn_cart,
                i_atom=i_atom,
            )

            max_ang_mom_plus_1 = coulomb_potential_data.max_ang_mom_plus_1[i_atom]
            i_target = [i for i, v in enumerate(i_atom_np) if v == i_atom]
            ang_moms = ang_mom_np[i_target]
            exponents = exponent_np[i_target]
            coefficients = coefficient_np[i_target]
            powers = power_np[i_target]

            # dim (#ang_moms, 1) # note: ang_moms include the same l because each l has seveal (exp, coeff, and powers).
            # jax.ops.segsum cares about it later.
            V_l_list = (
                np.linalg.norm(rel_R_cart_min_dist) ** -2.0
                * coefficients
                * np.linalg.norm(rel_R_cart_min_dist) ** powers
                * np.exp(-exponents * np.linalg.norm(rel_R_cart_min_dist) ** 2.0)
            )

            weight_list = []
            cos_theta_list = []
            wf_ratio_list = []
            for weight, vec_delta in zip(weights, grid_points, strict=True):
                weight_list.append(weight)
                r_dn_carts_on_mesh = r_dn_carts.copy()
                new_pos_dn = r_dn_cart + rel_R_cart_min_dist + np.linalg.norm(rel_R_cart_min_dist) * vec_delta
                r_dn_carts_on_mesh[r_dn_i] = new_pos_dn
                dn_mesh_non_local_ecp_part_up.append(r_up_carts)
                dn_mesh_non_local_ecp_part_dn.append(r_dn_carts_on_mesh)

                cos_theta = np.dot(
                    -1.0 * (rel_R_cart_min_dist) / np.linalg.norm(rel_R_cart_min_dist),
                    ((vec_delta) / np.linalg.norm(vec_delta)),
                )
                cos_theta_list.append(cos_theta)

                if flag_determinant_only:
                    jastrow_numerator = 1.0
                else:
                    jastrow_numerator = compute_Jastrow_part_ratio_zone(
                        jastrow_data=wavefunction_data.jastrow_data,
                        r_up_carts=r_up_carts,
                        r_dn_carts=r_dn_carts_on_mesh,
                    )

                det_numerator = compute_det_geminal_all_elements(
                    geminal_data=wavefunction_data.geminal_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts_on_mesh,
                )

                wf_ratio = np.exp(jastrow_numerator - jastrow_denominator) * det_numerator / det_denominator
                wf_ratio_list.append(wf_ratio)

            for ang_mom in range(max_ang_mom_plus_1):
                ll_target = [i for i, v in enumerate(ang_moms) if v == ang_mom]
                V_l = np.sum([V_l_list[ll] for ll in ll_target])
                P_l = np.array(
                    [
                        (2 * ang_mom + 1) * eval_legendre(ang_mom, cos_theta) * weight * wf_ratio
                        for cos_theta, weight, wf_ratio in zip(cos_theta_list, weight_list, wf_ratio_list, strict=True)
                    ]
                )
                ans = list(V_l * P_l)
                V_nonlocal += ans
                sum_V_nonlocal += np.sum(ans)

    mesh_non_local_ecp_part_r_up_carts = np.array(up_mesh_non_local_ecp_part_up + dn_mesh_non_local_ecp_part_up)
    mesh_non_local_ecp_part_r_dn_carts = np.array(up_mesh_non_local_ecp_part_dn + dn_mesh_non_local_ecp_part_dn)
    V_nonlocal = np.array(V_nonlocal)

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


def _compute_ecp_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
) -> float:
    """Compute ecp local and non-local parts.

    The method is for computing the local and non-local part of the given ECPs at (r_up_carts, r_dn_carts).
    A very straightforward (so very slow) implementation. Just for debudding purpose.

    Args:
        coulomb_potential_data (Coulomb_potential_data): an instance of Coulomb_potential_data
        wavefunction_data (Wavefunction_data): an instance of Wavefunction_data
        r_up_carts (npt.NDArray[np.float64]): Cartesian coordinates of up-spin electrons (dim: N_e^{up}, 3)
        r_dn_carts (npt.NDArray[np.float64]): Cartesian coordinates of dn-spin electrons (dim: N_e^{dn}, 3)
        RT (npt.NDArray): Rotation matrix. equiv R.T used for non-local part
        NN (int): Consider only up to NN-th nearest neighbors.
        Nv (int): The number of quadrature points for the spherical part.

    Returns:
        float: The sum of non-local part of the given ECPs with r_up_carts and r_dn_carts.
    """
    # Forward r_up/dn_carts/RT as-is (Principle 3a -- no parameter rebind).

    ecp_local_parts = _compute_ecp_local_parts_all_pairs_debug(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    _, _, _, ecp_nonlocal_parts = _compute_ecp_non_local_parts_nearest_neighbors_debug(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        Nv=Nv,
        NN=NN,
        flag_determinant_only=False,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


@jit
def compute_ecp_local_parts_all_pairs(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Compute local ECP contribution over all nucleus-electron pairs.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.

    Returns:
        float: Total local ECP energy for the provided electron coordinates.
    """

    # Compute the local part. To understand the flow, please refer to the debug version.
    def compute_V_l(
        r_cart: jax.Array,
        i_atom: jax.Array,
        exponent: jax.Array,
        coefficient: jax.Array,
        power: jax.Array,
    ) -> jax.Array:
        rel_R_cart_min_dist = _get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_cart,
            i_atom=i_atom,
            dtype=dtype_jnp,
        )
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficient
            * jnp.linalg.norm(rel_R_cart_min_dist) ** power
            * jnp.exp(-exponent * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    # Compute the local part V_l for a up electron.
    # This is activate when the given ang_mom == max_ang_mom_plus_1
    # i.e. the projection is not needed for the highest angular momentum
    # To understand the flow, please refer to the debug version.
    def compute_V_local(
        r_cart: jax.Array,
        i_atom: jax.Array,
        exponent: jax.Array,
        coefficient: jax.Array,
        power: jax.Array,
    ) -> jax.Array:
        V_l = compute_V_l(r_cart, i_atom, exponent, coefficient, power)
        return V_l

    # vectrize compute_ecp_up and compute_ecp_dn
    vmap_vmap_compute_ecp_up = vmap(
        vmap(
            compute_V_local,
            in_axes=(0, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0),
    )
    vmap_vmap_compute_ecp_dn = vmap(
        vmap(
            compute_V_local,
            in_axes=(0, None, None, None, None),
        ),
        in_axes=(None, 0, 0, 0, 0),
    )

    # Vectrized (flatten) arguments are prepared here.
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here. _get_min_dist_rel_R_cart_jnp
    # (called inside compute_V_l) reconstructs R - r in fp64 internally; pre-casting to
    # fp32 here would destroy precision before that reconstruction.
    dtype_jnp = get_dtype_jnp("coulomb")
    r_up_carts_jnp = r_up_carts
    r_dn_carts_jnp = r_dn_carts

    dtype_np = get_dtype_np("coulomb")
    i_atom_np = np.array(coulomb_potential_data._nucleus_index_local_part)
    exponent_np = np.array(coulomb_potential_data._exponents_local_part, dtype=dtype_np)
    coefficient_np = np.array(coulomb_potential_data._coefficients_local_part, dtype=dtype_np)
    power_np = np.array(coulomb_potential_data._powers_local_part, dtype=dtype_np)

    V_ecp_up = jnp.sum(
        vmap_vmap_compute_ecp_up(
            r_up_carts_jnp,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    )

    V_ecp_dn = jnp.sum(
        vmap_vmap_compute_ecp_dn(
            r_dn_carts_jnp,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    )

    V_ecp = V_ecp_up + V_ecp_dn

    return V_ecp


@partial(jit, static_argnums=(5, 6, 7, 8))
def _compute_ecp_non_local_parts_nearest_neighbors_impl(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
    project_real: bool = True,
) -> _EcpNonlocalResult:
    """Shared implementation for the NN non-local ECP contribution.

    Two thin public entries wrap this: :func:`compute_ecp_non_local_parts_nearest_neighbors`
    (``project_real=True``, forward-energy semantics) and
    :func:`compute_ecp_non_local_parts_nearest_neighbors_holomorphic`
    (``project_real=False``, un-projected complex weights). See those docstrings.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        NN (int): Number of nearest nuclei to include for each electron.
        Nv (int): Number of quadrature points on the sphere.
        flag_determinant_only (bool): If True, ignore Jastrow in the wavefunction ratio.
        project_real (bool): If True, a complex-Bloch (twist) wavefunction has ``Re`` applied
            to ``V_l x P_l`` BEFORE the contraction (forward-energy real-part semantics). If
            False, the HOLOMORPHIC complex per-grid weights are kept un-projected. The
            projection is intentionally at the pre-einsum position (moving ``Re`` after the
            einsum re-introduces a complex-cotangent-to-real cast on the einsum's strictly-real
            ``V_l`` slot that trips ``_strict_complex_cast_guard``).

    Returns:
        tuple[list[jax.Array], list[jax.Array], jax.Array, float]:
            - Mesh-displaced ``r_up_carts`` per configuration.
            - Mesh-displaced ``r_dn_carts`` per configuration.
            - Non-local ECP contributions per configuration (flattened).
            - Scalar sum of all non-local contributions.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here. They are forwarded to
    # compute_Jastrow_part / compute_det_geminal_all_elements (fp64 zones); pre-casting
    # to the coulomb (fp32) zone would silently down-cast inputs to those fp64 zones.
    # RT is also forwarded as-is (Principle 3a); cast at the use site below.
    # ``dtype_jnp`` keeps the real ``coulomb`` zone for
    # geometry / mesh / position constants (they are real-valued); only the
    # FOUR ratio casts (``jastrow_{x,xp}``, ``det_{x,xp}``) lift to complex
    # via a separate ``_ratio_dtype`` so ``Im`` survives at non-Gamma.
    # Static (trace-time) flag on the determinant orb leaf -> outer ``Re()``.
    coulomb_potential_data._validate_xla_projector_channels()
    dtype_jnp = get_dtype_jnp("coulomb")
    _is_complex_wf = np.issubdtype(wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating)

    if Nv == 4:
        weights = jnp.array(tetrahedron_sym_mesh_Nv4.weights, dtype=dtype_jnp)
        grid_points = jnp.array(tetrahedron_sym_mesh_Nv4.grid_points, dtype=dtype_jnp)
    elif Nv == 6:
        weights = jnp.array(octahedron_sym_mesh_Nv6.weights, dtype=dtype_jnp)
        grid_points = jnp.array(octahedron_sym_mesh_Nv6.grid_points, dtype=dtype_jnp)
    elif Nv == 12:
        weights = jnp.array(icosahedron_sym_mesh_Nv12.weights, dtype=dtype_jnp)
        grid_points = jnp.array(icosahedron_sym_mesh_Nv12.grid_points, dtype=dtype_jnp)
    elif Nv == 18:
        weights = jnp.array(octahedron_sym_mesh_Nv18.weights, dtype=dtype_jnp)
        grid_points = jnp.array(octahedron_sym_mesh_Nv18.grid_points, dtype=dtype_jnp)
    else:
        raise NotImplementedError

    # Cast RT to coulomb zone at the use site (Principle 3b).
    grid_points = grid_points @ RT.astype(dtype_jnp)  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)
    grid_norm = jnp.linalg.norm(grid_points, axis=1, keepdims=True)

    # jnp variables
    ang_mom_all, exponent_all, coefficient_all, power_all = coulomb_potential_data._padded_parameters_tuple
    global_max_ang_mom_plus_1 = coulomb_potential_data._global_max_ang_mom_plus_1

    # stored
    non_local_ecp_part_r_carts_up = jnp.zeros((0, len(r_up_carts), 3), dtype=dtype_jnp)
    non_local_ecp_part_r_carts_dn = jnp.zeros((0, len(r_dn_carts), 3), dtype=dtype_jnp)

    @jit
    def compute_V_l(
        rel_R_cart_min_dist: jax.Array,
        exponents: jax.Array,
        coefficients: jax.Array,
        powers: jax.Array,
    ) -> jax.Array:
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficients
            * jnp.linalg.norm(rel_R_cart_min_dist) ** powers
            * jnp.exp(-exponents * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    @jit
    def compute_P_l(
        ang_mom: jax.Array,
        cos_theta: jax.Array,
        weight: jax.Array,
        wf_ratio: jax.Array,
    ) -> jax.Array:
        P_l = (2 * ang_mom + 1) * _legendre_tablated(ang_mom, cos_theta) * weight * wf_ratio
        return P_l

    def _build_mesh_for_spin(r_carts: jax.Array, other_carts: jax.Array) -> _EcpMeshBlock:
        n_spin = r_carts.shape[0]
        n_other = other_carts.shape[0]
        if n_spin == 0:
            return (
                jnp.zeros((0, n_spin, 3), dtype=dtype_jnp),
                jnp.zeros((0, n_other, 3), dtype=dtype_jnp),
                jnp.zeros((n_spin, 0, global_max_ang_mom_plus_1), dtype=dtype_jnp),
                jnp.zeros((0,), dtype=dtype_jnp),
                jnp.zeros((0,), dtype=dtype_jnp),
            )

        i_atom_lists = vmap(
            lambda r_cart: _find_nearest_ecp_nucleus_indices_jnp(
                coulomb_potential_data=coulomb_potential_data,
                r_cart=r_cart,
                NN=NN,
            )
        )(r_carts)

        def _rels_for_electron(r_cart: jax.Array, i_atom_list: jax.Array) -> jax.Array:
            return vmap(
                lambda i_atom: _get_min_dist_rel_R_cart_jnp(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_cart,
                    i_atom=i_atom,
                    dtype=dtype_jnp,
                )
            )(i_atom_list)

        rels = vmap(_rels_for_electron)(r_carts, i_atom_lists)  # (n_spin, NN, 3)
        rel_norm = jnp.linalg.norm(rels, axis=-1, keepdims=True)
        offsets = rels[..., None, :] + rel_norm[..., None, :] * grid_points[None, None, :, :]
        updated_carts = r_carts[:, None, None, :] + offsets  # (n_spin, NN, Nv, 3)

        # PBC quadrature points r_q = R_I^nearest + r_iI * Omega_q can land
        # outside the home parallelepiped. The home-cell wrap that keeps the
        # truncated AO image sum accurate is centralised in the
        # periodicity guarantor ``compute_AOs`` (jqmc/atomic_orbital.py), which
        # wraps every Psi-evaluation coordinate just before the image sum, so
        # an explicit wrap of ``updated_carts`` here would be redundant.
        # ``rels`` (the projector argument) must stay unwrapped -- wrapping it
        # would corrupt the ECP angular / radial dependency -- and it does,
        # since it never flows into ``compute_AOs``.
        delta = updated_carts - r_carts[:, None, None, :]
        one_hot = jax.nn.one_hot(jnp.arange(n_spin), n_spin)
        delta_full = delta[..., None, :] * one_hot[:, None, None, :, None]
        base = r_carts[None, None, None, :, :]
        r_carts_on_mesh = base + delta_full  # (n_spin, NN, Nv, n_spin, 3)
        if n_other == 0:
            other_carts_on_mesh = jnp.zeros((n_spin, NN, grid_points.shape[0], 0, 3), dtype=dtype_jnp)
        else:
            other_carts_on_mesh = jnp.broadcast_to(other_carts, (n_spin, NN, grid_points.shape[0], n_other, 3))

        ang_moms = ang_mom_all[i_atom_lists]
        exponents = exponent_all[i_atom_lists]
        coefficients = coefficient_all[i_atom_lists]
        powers = power_all[i_atom_lists]

        def _V_l_mapped(
            rel: jax.Array,
            ang_mom: jax.Array,
            exponent: jax.Array,
            coefficient: jax.Array,
            power: jax.Array,
        ) -> jax.Array:
            # NOTE: see compute_ecp_non_local_parts_nearest_neighbors_fast_update
            # below for the rationale -- `jax.ops.segment_sum` inside vmap(vmap(...))
            # lowers to a 4096*32*1*L-iter while_loop on GPU. For small L
            # (typically 2-3) a masked dense reduce is dramatically faster.
            V_l_vmapped = compute_V_l(rel, exponent, coefficient, power)
            mask = ang_mom[:, None] == jnp.arange(global_max_ang_mom_plus_1)[None, :]
            return jnp.where(mask, V_l_vmapped[:, None], 0.0).sum(axis=0)

        V_l_mapped = vmap(vmap(_V_l_mapped, in_axes=(0, 0, 0, 0, 0)))(rels, ang_moms, exponents, coefficients, powers)

        rel_unit = -rels / rel_norm
        grid_unit = grid_points / grid_norm
        cos_theta = jnp.einsum("ijn,kn->ijk", rel_unit, grid_unit)
        weight = jnp.broadcast_to(weights, cos_theta.shape)

        r_mesh = r_carts_on_mesh.reshape(-1, n_spin, 3)
        if n_other == 0:
            other_mesh = jnp.zeros((r_mesh.shape[0], 0, 3), dtype=dtype_jnp)
        else:
            other_mesh = other_carts_on_mesh.reshape(-1, n_other, 3)
        return r_mesh, other_mesh, V_l_mapped, cos_theta.reshape(-1), weight.reshape(-1)

    up_mesh_r_up, up_mesh_r_dn, V_l_up_base, cos_up, weight_up = _build_mesh_for_spin(r_up_carts, r_dn_carts)
    dn_mesh_r_dn, dn_mesh_r_up, V_l_dn_base, cos_dn, weight_dn = _build_mesh_for_spin(r_dn_carts, r_up_carts)

    non_local_ecp_part_r_carts_up = jnp.concatenate([up_mesh_r_up, dn_mesh_r_up], axis=0)
    non_local_ecp_part_r_carts_dn = jnp.concatenate([up_mesh_r_dn, dn_mesh_r_dn], axis=0)
    non_local_ecp_part_r_carts_up = jnp.array(non_local_ecp_part_r_carts_up)
    non_local_ecp_part_r_carts_dn = jnp.array(non_local_ecp_part_r_carts_dn)

    # jastrow_ratio
    if flag_determinant_only:
        jastrow_x = 1.0
        jastrow_xp = 1.0
    else:
        jastrow_x = compute_Jastrow_part_ratio_zone(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts)
        jastrow_xp = vmap(compute_Jastrow_part_ratio_zone, in_axes=(None, 0, 0))(
            wavefunction_data.jastrow_data, non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn
        )

    # det_ratio
    det_x = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
    det_xp = vmap(compute_det_geminal_all_elements, in_axes=(None, 0, 0))(
        wavefunction_data.geminal_data, non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn
    )

    # Cast all ratio inputs to the local coulomb zone dtype before arithmetic.
    # The strictly-real ``coulomb``
    # zone would silently drop ``Im`` from the determinant ratio before the
    # outer ``Re(...)`` wrap is reached; ``_ratio_dtype`` lifts to
    # ``complex128`` when any input is complex (non-Gamma) and stays real at
    # Gamma (bit-identical to the real Gamma-point path). Note ``jastrow_x/xp`` are Python
    # ``1.0`` under ``flag_determinant_only=True``, so promote via the
    # already-jnp ``det_*`` only.
    jastrow_x = cast_to_zone(jastrow_x, "coulomb", det_x.dtype, det_xp.dtype)
    jastrow_xp = cast_to_zone(jastrow_xp, "coulomb", det_x.dtype, det_xp.dtype)
    det_x = cast_to_zone(det_x, "coulomb", det_xp.dtype)
    det_xp = cast_to_zone(det_xp, "coulomb", det_x.dtype)
    wf_ratio_all = jnp.exp(jastrow_xp - jastrow_x) * det_xp / det_x

    # Split ratios for up/dn blocks to avoid big concat of V_l / cos / weight.
    g_up = cos_up.shape[0]
    wf_ratio_up = wf_ratio_all[:g_up]
    wf_ratio_dn = wf_ratio_all[g_up:]

    def _contract_chunk(
        V_l_base: jax.Array,
        cos_flat: jax.Array,
        weight_flat: jax.Array,
        wf_ratio_flat: jax.Array,
    ) -> _ArrayPair:
        cos_flat = jnp.array(cos_flat)
        weight_flat = jnp.array(weight_flat)
        wf_ratio_flat = jnp.array(wf_ratio_flat)

        # map from grid index -> rel index to reuse V_l_base without duplication
        num_rel = V_l_base.shape[0] * V_l_base.shape[1]
        rel_indices = jnp.repeat(jnp.arange(num_rel), grid_points.shape[0])
        V_l_flat = V_l_base.reshape(num_rel, global_max_ang_mom_plus_1)
        V_l_per_grid = V_l_flat[rel_indices]

        P_l_chunk = vmap(vmap(compute_P_l, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None))(
            jnp.arange(global_max_ang_mom_plus_1), cos_flat, weight_flat, wf_ratio_flat
        )
        # Mirror the fast path -- apply the Re
        # projection to ``P_l_chunk`` BEFORE the einsum so the V_l x P_l
        # contraction stays real-valued throughout. Projecting only at the
        # outer ``Re(sum_V_nonlocal)`` / ``Re(V_nonlocal)`` at the bottom of the
        # function would leave the backward
        # pass casting a complex cotangent to real on the einsum's V_l slot
        # (V_l is strictly real), tripping the ``_strict_complex_cast_guard``
        # autouse fixture in the test suite. Bit-identical at Gamma where
        # P_l_chunk is already real-valued.
        if _is_complex_wf and project_real:
            P_l_chunk = jnp.real(P_l_chunk)
        elif _is_complex_wf:
            # Holomorphic path: promote the real V_l to the complex P_l dtype so
            # the backward pass never casts a complex cotangent into it.
            V_l_per_grid = V_l_per_grid.astype(P_l_chunk.dtype)
        V_nonlocal_chunk = jnp.einsum("gl,lg->g", V_l_per_grid, P_l_chunk)

        def _scan_sum(carry: jax.Array, elems: _ArrayPair) -> tuple[jax.Array, None]:
            v_l_row, p_l_row = elems
            return carry + jnp.dot(v_l_row, p_l_row), None

        sum_chunk, _ = jax.lax.scan(
            _scan_sum,
            0.0,
            (V_l_per_grid, jnp.moveaxis(P_l_chunk, 0, 1)),
        )
        return V_nonlocal_chunk, sum_chunk

    V_nonlocal_up, sum_up = _contract_chunk(V_l_up_base, cos_up, weight_up, wf_ratio_up)
    V_nonlocal_dn, sum_dn = _contract_chunk(V_l_dn_base, cos_dn, weight_dn, wf_ratio_dn)

    V_nonlocal = jnp.concatenate([V_nonlocal_up, V_nonlocal_dn], axis=0)
    sum_V_nonlocal = sum_up + sum_dn
    # Re was already applied inside ``_contract_chunk`` BEFORE the einsum; the
    # outer wrap is therefore unnecessary (kept for clarity / single-source-of-
    # truth would re-introduce the backward-pass cast hazard). ``sum_V_nonlocal``
    # and ``V_nonlocal`` are real here on both paths.

    return non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn, V_nonlocal, sum_V_nonlocal


def compute_ecp_non_local_parts_nearest_neighbors(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> _EcpNonlocalResult:
    r"""Compute real fixed-phase non-local ECP candidates with a nearest-neighbor cutoff.

    For complex-Bloch (twist) wavefunctions the returned per-grid
    ``V_nonlocal`` and ``sum_V_nonlocal`` are real. At the real Gamma point
    the weights are already real.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        NN (int): Number of nearest nuclei to include for each electron.
        Nv (int): Number of quadrature points on the sphere.
        flag_determinant_only (bool): If True, ignore Jastrow in the wavefunction ratio.

    Returns:
        tuple[list[jax.Array], list[jax.Array], jax.Array, float]:
            - Mesh-displaced ``r_up_carts`` per configuration.
            - Mesh-displaced ``r_dn_carts`` per configuration.
            - Non-local ECP contributions per configuration (flattened, real).
            - Scalar sum of all non-local contributions (real).
    """
    return _compute_ecp_non_local_parts_nearest_neighbors_impl(
        coulomb_potential_data,
        wavefunction_data,
        r_up_carts,
        r_dn_carts,
        RT,
        NN,
        Nv,
        flag_determinant_only,
        True,
    )


def compute_ecp_non_local_parts_nearest_neighbors_holomorphic(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    r"""Compute un-projected (holomorphic) non-local ECP weights.

    Unlike :func:`compute_ecp_non_local_parts_nearest_neighbors`, the ``Re``
    projection is not applied. For complex-Bloch wavefunctions the returned
    per-grid ``V_nonlocal`` and ``sum_V_nonlocal`` may be complex; at the real
    Gamma point they are real.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        NN (int): Number of nearest nuclei to include for each electron.
        Nv (int): Number of quadrature points on the sphere.
        flag_determinant_only (bool): If True, ignore Jastrow in the wavefunction ratio.

    Returns:
        tuple[list[jax.Array], list[jax.Array], jax.Array, float]:
            - Mesh-displaced ``r_up_carts`` per configuration.
            - Mesh-displaced ``r_dn_carts`` per configuration.
            - Non-local ECP contributions per configuration (flattened, possibly complex).
            - Scalar sum of all non-local contributions (possibly complex).
    """
    return _compute_ecp_non_local_parts_nearest_neighbors_impl(
        coulomb_potential_data,
        wavefunction_data,
        r_up_carts,
        r_dn_carts,
        RT,
        NN,
        Nv,
        flag_determinant_only,
        False,
    )


def _ecp_candidate_positions_for_spin(
    coulomb_potential_data: Coulomb_potential_data,
    r_carts: jax.Array,
    RT: jax.Array,
    NN: int,
    Nv: int,
) -> jax.Array:
    """Candidate positions only, in electron/neighbor/quadrature order."""
    dtype_jnp = get_dtype_jnp("coulomb")
    if Nv == 4:
        grid_points = jnp.array(tetrahedron_sym_mesh_Nv4.grid_points, dtype=dtype_jnp)
    elif Nv == 6:
        grid_points = jnp.array(octahedron_sym_mesh_Nv6.grid_points, dtype=dtype_jnp)
    elif Nv == 12:
        grid_points = jnp.array(icosahedron_sym_mesh_Nv12.grid_points, dtype=dtype_jnp)
    elif Nv == 18:
        grid_points = jnp.array(octahedron_sym_mesh_Nv18.grid_points, dtype=dtype_jnp)
    else:
        raise NotImplementedError
    grid_points = grid_points @ RT.astype(dtype_jnp)
    if r_carts.shape[0] == 0:
        return jnp.zeros((0, NN * Nv, 3), dtype=dtype_jnp)
    i_atom_lists = vmap(
        lambda r: _find_nearest_ecp_nucleus_indices_jnp(
            coulomb_potential_data=coulomb_potential_data,
            r_cart=r,
            NN=NN,
        )
    )(r_carts)

    def _rels(r: jax.Array, indices: jax.Array) -> jax.Array:
        return vmap(
            lambda i: _get_min_dist_rel_R_cart_jnp(
                structure_data=coulomb_potential_data.structure_data,
                r_cart=r,
                i_atom=i,
                dtype=dtype_jnp,
            )
        )(indices)

    rels = vmap(_rels)(r_carts, i_atom_lists)
    rel_norm = jnp.linalg.norm(rels, axis=-1, keepdims=True)
    offsets = rels[..., None, :] + rel_norm[..., None, :] * grid_points[None, None, :, :]
    return (r_carts[:, None, None, :] + offsets).reshape(r_carts.shape[0], NN * Nv, 3)


@struct.dataclass
class Ecp_candidate_mesh_streaming_state:
    """Projection-scoped determinant/J3 candidate cache for the ECP mesh."""

    candidate_up_positions: jax.Array
    candidate_dn_positions: jax.Array
    det_candidate_state: Det_candidate_mesh_state
    j1_candidate_state: J1_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j3_candidate_state: J3_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j2_candidate_state: J2_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    j_dtn_candidate_state: J_DTN_candidate_mesh_state | None = struct.field(pytree_node=True, default=None)
    NN: int = struct.field(pytree_node=False, default=NN_default)
    Nv: int = struct.field(pytree_node=False, default=Nv_default)


@partial(jit, static_argnums=(5, 6))
def _init_ecp_candidate_mesh_streaming_state(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
) -> Ecp_candidate_mesh_streaming_state:
    up = _ecp_candidate_positions_for_spin(coulomb_potential_data, r_up_carts, RT, NN, Nv)
    dn = _ecp_candidate_positions_for_spin(coulomb_potential_data, r_dn_carts, RT, NN, Nv)
    det = _init_det_candidate_mesh_state(wavefunction_data.geminal_data, up, dn)
    j1_data = wavefunction_data.jastrow_data.jastrow_one_body_data
    j1 = _init_j1_candidate_mesh_state(j1_data, r_up_carts, r_dn_carts, up, dn) if j1_data is not None else None
    j3_data = wavefunction_data.jastrow_data.jastrow_three_body_data
    j3 = _init_j3_candidate_mesh_state(j3_data, up, dn) if j3_data is not None else None
    j2_data = wavefunction_data.jastrow_data.jastrow_two_body_data
    j2 = _init_j2_candidate_mesh_state(j2_data, r_up_carts, r_dn_carts, up, dn) if j2_data is not None else None
    dtn_data = wavefunction_data.jastrow_data.jastrow_dtn_data
    j_dtn = _init_j_dtn_candidate_mesh_state(dtn_data, r_up_carts, r_dn_carts, up, dn) if dtn_data is not None else None
    return Ecp_candidate_mesh_streaming_state(
        candidate_up_positions=up,
        candidate_dn_positions=dn,
        det_candidate_state=det,
        j1_candidate_state=j1,
        j3_candidate_state=j3,
        j2_candidate_state=j2,
        j_dtn_candidate_state=j_dtn,
        NN=NN,
        Nv=Nv,
    )


@jit
def _advance_ecp_candidate_mesh_streaming_state(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    state: Ecp_candidate_mesh_streaming_state,
    moved_spin_is_up: jax.Array,
    moved_index: jax.Array,
    r_up_carts_old: jax.Array,
    r_dn_carts_old: jax.Array,
    r_up_carts_new: jax.Array,
    r_dn_carts_new: jax.Array,
    RT: jax.Array,
) -> Ecp_candidate_mesh_streaming_state:
    up = _ecp_candidate_positions_for_spin(coulomb_potential_data, r_up_carts_new, RT, state.NN, state.Nv)
    dn = _ecp_candidate_positions_for_spin(coulomb_potential_data, r_dn_carts_new, RT, state.NN, state.Nv)
    det = _advance_det_candidate_mesh_state(
        wavefunction_data.geminal_data,
        state.det_candidate_state,
        moved_spin_is_up,
        moved_index,
        up,
        dn,
    )
    j1_data = wavefunction_data.jastrow_data.jastrow_one_body_data
    j1 = (
        _advance_j1_candidate_mesh_state(
            j1_data,
            state.j1_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_new,
            r_dn_carts_new,
            up,
            dn,
        )
        if j1_data is not None
        else None
    )
    j3_data = wavefunction_data.jastrow_data.jastrow_three_body_data
    j3 = (
        _advance_j3_candidate_mesh_state(j3_data, state.j3_candidate_state, moved_spin_is_up, moved_index, up, dn)
        if j3_data is not None
        else None
    )
    j2_data = wavefunction_data.jastrow_data.jastrow_two_body_data
    j2 = (
        _advance_j2_candidate_mesh_state(
            j2_data,
            state.j2_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_old,
            r_dn_carts_old,
            r_up_carts_new,
            r_dn_carts_new,
            up,
            dn,
        )
        if j2_data is not None
        else None
    )
    dtn_data = wavefunction_data.jastrow_data.jastrow_dtn_data
    j_dtn = (
        _advance_j_dtn_candidate_mesh_state(
            dtn_data,
            state.j_dtn_candidate_state,
            moved_spin_is_up,
            moved_index,
            r_up_carts_old,
            r_dn_carts_old,
            r_up_carts_new,
            r_dn_carts_new,
            up,
            dn,
        )
        if dtn_data is not None
        else None
    )
    return state.replace(
        candidate_up_positions=up,
        candidate_dn_positions=dn,
        det_candidate_state=det,
        j1_candidate_state=j1,
        j3_candidate_state=j3,
        j2_candidate_state=j2,
        j_dtn_candidate_state=j_dtn,
    )


@partial(jit, static_argnums=(6, 7, 8, 12))
def _compute_ecp_non_local_parts_nearest_neighbors_fast_update_impl(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
    candidate_mesh_state: "Ecp_candidate_mesh_streaming_state | None" = None,
    project_real: bool = True,
) -> tuple[list, list, list, float]:
    """Fast-update variant of non-local ECP contributions (nearest neighbors).

    This variant reuses the inverse geminal matrix to compute determinant ratios
    and uses Jastrow ratios, avoiding full recomputation for each mesh point.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        A_old_inv (jax.Array): Inverse geminal matrix evaluated at ``(r_up_carts, r_dn_carts)``.
        NN (int): Number of nearest nuclei to include for each electron.
        Nv (int): Number of quadrature points on the sphere.
        flag_determinant_only (bool): If True, ignore Jastrow in the wavefunction ratio.
        j3_state: Optional cached J3 streaming auxiliaries consistent with
            ``(r_up_carts, r_dn_carts)``. Forwarded to the split-spin Jastrow
            ratio kernel so it can skip per-call ``aos_*_old``/``W``/``U``/cross_vec
            recomputation. Use the value carried in the projection's
            ``Kinetic_streaming_state.j3_state``; pass ``None`` (default) for
            the one-shot path used by observation/MCMC code.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or a
            superset like :class:`Det_streaming_state`) consistent with
            ``(r_up_carts, r_dn_carts)``. Forwarded to
            ``_compute_ratio_determinant_part_split_spin`` so it can skip
            the bulk-side AO eval and the two ``lambda_paired @ ao_*``
            precontracts.
        candidate_mesh_state: Optional :class:`Ecp_candidate_mesh_streaming_state`
            (projection-scoped determinant / Jastrow candidate caches for the
            ECP mesh) consistent with ``(r_up_carts, r_dn_carts)``; ``None``
            evaluates the candidates directly.
        project_real (bool): If True, a complex-Bloch (twist) wavefunction has
            ``Re`` applied to the per-candidate weights (real fixed-phase
            semantics); if False, the holomorphic complex weights are kept.

    Returns:
        tuple[list[jax.Array], list[jax.Array], jax.Array, float]:
            - Mesh-displaced ``r_up_carts`` per configuration.
            - Mesh-displaced ``r_dn_carts`` per configuration.
            - Non-local ECP contributions per configuration (flattened).
            - Scalar sum of all non-local contributions.

    Warning:
        ``A_old_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}`` exactly
        at the supplied electron positions.  Correctness is only guaranteed when
        the inverse is maintained via **single-electron (rank-1) Sherman-Morrison
        updates** starting from a freshly initialized LU inverse -- the pattern
        used in the MCMC loop.  Passing an inverse from a different configuration
        silently produces incorrect non-local ECP contributions.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here (forwarded to fp64 Jastrow/det zones).
    # RT is also forwarded as-is (Principle 3a); cast at the use site below.
    # ``dtype_jnp`` stays in the real ``coulomb`` zone
    # for geometry / mesh constants; ratio casts are individually lifted
    # below via a separate ``_ratio_dtype``.
    coulomb_potential_data._validate_xla_projector_channels()
    dtype_jnp = get_dtype_jnp("coulomb")
    _is_complex_wf = np.issubdtype(wavefunction_data.geminal_data._determinant_dtype_jnp, np.complexfloating)

    if Nv == 4:
        weights = jnp.array(tetrahedron_sym_mesh_Nv4.weights, dtype=dtype_jnp)
        grid_points = jnp.array(tetrahedron_sym_mesh_Nv4.grid_points, dtype=dtype_jnp)
    elif Nv == 6:
        weights = jnp.array(octahedron_sym_mesh_Nv6.weights, dtype=dtype_jnp)
        grid_points = jnp.array(octahedron_sym_mesh_Nv6.grid_points, dtype=dtype_jnp)
    elif Nv == 12:
        weights = jnp.array(icosahedron_sym_mesh_Nv12.weights, dtype=dtype_jnp)
        grid_points = jnp.array(icosahedron_sym_mesh_Nv12.grid_points, dtype=dtype_jnp)
    elif Nv == 18:
        weights = jnp.array(octahedron_sym_mesh_Nv18.weights, dtype=dtype_jnp)
        grid_points = jnp.array(octahedron_sym_mesh_Nv18.grid_points, dtype=dtype_jnp)
    else:
        raise NotImplementedError

    # Cast RT to coulomb zone at the use site (Principle 3b).
    grid_points = grid_points @ RT.astype(dtype_jnp)  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)
    grid_norm = jnp.linalg.norm(grid_points, axis=1, keepdims=True)

    # jnp variables
    ang_mom_all, exponent_all, coefficient_all, power_all = coulomb_potential_data._padded_parameters_tuple
    global_max_ang_mom_plus_1 = coulomb_potential_data._global_max_ang_mom_plus_1

    # stored
    non_local_ecp_part_r_carts_up = jnp.zeros((0, len(r_up_carts), 3), dtype=dtype_jnp)
    non_local_ecp_part_r_carts_dn = jnp.zeros((0, len(r_dn_carts), 3), dtype=dtype_jnp)

    @jit
    def compute_V_l(
        rel_R_cart_min_dist: jax.Array,
        exponents: jax.Array,
        coefficients: jax.Array,
        powers: jax.Array,
    ) -> jax.Array:
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficients
            * jnp.linalg.norm(rel_R_cart_min_dist) ** powers
            * jnp.exp(-exponents * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    @jit
    def compute_P_l(
        ang_mom: jax.Array,
        cos_theta: jax.Array,
        weight: jax.Array,
        wf_ratio: jax.Array,
    ) -> jax.Array:
        P_l = (2 * ang_mom + 1) * _legendre_tablated(ang_mom, cos_theta) * weight * wf_ratio
        return P_l

    def _build_mesh_for_spin(r_carts: jax.Array, other_carts: jax.Array) -> _EcpMeshBlock:
        n_spin = r_carts.shape[0]
        n_other = other_carts.shape[0]
        if n_spin == 0:
            return (
                jnp.zeros((0, n_spin, 3), dtype=dtype_jnp),
                jnp.zeros((0, n_other, 3), dtype=dtype_jnp),
                jnp.zeros((global_max_ang_mom_plus_1, 0), dtype=dtype_jnp),
                jnp.zeros((0,), dtype=dtype_jnp),
                jnp.zeros((0,), dtype=dtype_jnp),
            )

        i_atom_lists = vmap(
            lambda r_cart: _find_nearest_ecp_nucleus_indices_jnp(
                coulomb_potential_data=coulomb_potential_data,
                r_cart=r_cart,
                NN=NN,
            )
        )(r_carts)

        def _rels_for_electron(r_cart: jax.Array, i_atom_list: jax.Array) -> jax.Array:
            return vmap(
                lambda i_atom: _get_min_dist_rel_R_cart_jnp(
                    structure_data=coulomb_potential_data.structure_data,
                    r_cart=r_cart,
                    i_atom=i_atom,
                    dtype=dtype_jnp,
                )
            )(i_atom_list)

        rels = vmap(_rels_for_electron)(r_carts, i_atom_lists)  # (n_spin, NN, 3)
        rel_norm = jnp.linalg.norm(rels, axis=-1, keepdims=True)
        offsets = rels[..., None, :] + rel_norm[..., None, :] * grid_points[None, None, :, :]
        updated_carts = r_carts[:, None, None, :] + offsets  # (n_spin, NN, Nv, 3)

        # PBC quadrature points r_q = R_I^nearest + r_iI * Omega_q can land
        # outside the home parallelepiped. The home-cell wrap that keeps the
        # truncated AO image sum accurate is centralised in the
        # periodicity guarantor ``compute_AOs`` (jqmc/atomic_orbital.py), which
        # wraps every Psi-evaluation coordinate just before the image sum, so
        # an explicit wrap of ``updated_carts`` here would be redundant.
        # ``rels`` (the projector argument) must stay unwrapped -- wrapping it
        # would corrupt the ECP angular / radial dependency -- and it does,
        # since it never flows into ``compute_AOs``.
        delta = updated_carts - r_carts[:, None, None, :]
        one_hot = jax.nn.one_hot(jnp.arange(n_spin), n_spin)
        delta_full = delta[..., None, :] * one_hot[:, None, None, :, None]
        base = r_carts[None, None, None, :, :]
        r_carts_on_mesh = base + delta_full  # (n_spin, NN, Nv, n_spin, 3)
        if n_other == 0:
            other_carts_on_mesh = jnp.zeros((n_spin, NN, grid_points.shape[0], 0, 3), dtype=dtype_jnp)
        else:
            other_carts_on_mesh = jnp.broadcast_to(other_carts, (n_spin, NN, grid_points.shape[0], n_other, 3))

        ang_moms = ang_mom_all[i_atom_lists]
        exponents = exponent_all[i_atom_lists]
        coefficients = coefficient_all[i_atom_lists]
        powers = power_all[i_atom_lists]

        def _V_l_mapped(
            rel: jax.Array,
            ang_mom: jax.Array,
            exponent: jax.Array,
            coefficient: jax.Array,
            power: jax.Array,
        ) -> jax.Array:
            # NOTE: not `jax.ops.segment_sum(V_l_vmapped, ang_mom,
            # num_segments=global_max_ang_mom_plus_1)`: inside the outer
            # vmap(vmap(...)) that scatter lowers to a 1-element-per-iter
            # GPU while_loop with batch dim flattened to 4096*32*1*L =
            # ~262k iters/call, dominating the LRDMC launch storm. For small L
            # (typically 2-3 for cc-ECP) a masked dense reduce avoids the scatter entirely.
            V_l_vmapped = compute_V_l(rel, exponent, coefficient, power)
            mask = ang_mom[:, None] == jnp.arange(global_max_ang_mom_plus_1)[None, :]
            return jnp.where(mask, V_l_vmapped[:, None], 0.0).sum(axis=0)

        V_l_mapped = vmap(vmap(_V_l_mapped, in_axes=(0, 0, 0, 0, 0)))(rels, ang_moms, exponents, coefficients, powers)
        V_l_dup = jnp.repeat(V_l_mapped[:, :, :, None], grid_points.shape[0], axis=3)
        V_l_all = jnp.moveaxis(V_l_dup, 2, 0).reshape(global_max_ang_mom_plus_1, -1)

        rel_unit = -rels / rel_norm
        grid_unit = grid_points / grid_norm
        cos_theta = jnp.einsum("ijn,kn->ijk", rel_unit, grid_unit)
        weight = jnp.broadcast_to(weights, cos_theta.shape)

        r_mesh = r_carts_on_mesh.reshape(-1, n_spin, 3)
        if n_other == 0:
            other_mesh = jnp.zeros((r_mesh.shape[0], 0, 3), dtype=dtype_jnp)
        else:
            other_mesh = other_carts_on_mesh.reshape(-1, n_other, 3)
        return r_mesh, other_mesh, V_l_all, cos_theta.reshape(-1), weight.reshape(-1)

    up_mesh_r_up, up_mesh_r_dn, V_l_up, cos_up, weight_up = _build_mesh_for_spin(r_up_carts, r_dn_carts)
    dn_mesh_r_dn, dn_mesh_r_up, V_l_dn, cos_dn, weight_dn = _build_mesh_for_spin(r_dn_carts, r_up_carts)

    non_local_ecp_part_r_carts_up = jnp.concatenate([up_mesh_r_up, dn_mesh_r_up], axis=0)
    non_local_ecp_part_r_carts_dn = jnp.concatenate([up_mesh_r_dn, dn_mesh_r_dn], axis=0)
    non_local_ecp_part_r_carts_up = jnp.array(non_local_ecp_part_r_carts_up)
    non_local_ecp_part_r_carts_dn = jnp.array(non_local_ecp_part_r_carts_dn)

    # wavefunction ratio.
    # A supplied candidate state means this call IS a projection sub-step; with
    # none, the label stays whatever the caller established (MCMC / observable
    # / force), so a gate can hold the projection to a stricter standard than
    # the paths that legitimately have no interval-scoped mesh to reuse.
    def _ecp_site() -> AbstractContextManager[None]:
        # A fresh context per use: ``route_site`` is a single-use generator CM.
        return route_site("projection") if candidate_mesh_state is not None else nullcontext()

    if candidate_mesh_state is None:
        with _ecp_site():
            note_route("det", "ecp", cached=False)
        det_ratio = _compute_ratio_determinant_part_split_spin(
            geminal_data=wavefunction_data.geminal_data,
            A_old_inv=A_old_inv,
            old_r_up_carts=r_up_carts,
            old_r_dn_carts=r_dn_carts,
            new_r_up_shifted=up_mesh_r_up,
            new_r_dn_shifted=dn_mesh_r_dn,
            det_ratio_state=det_ratio_state,
        )
    else:
        with _ecp_site():
            note_route("det", "ecp", cached=True)
        # winv weights are DERIVED from the running inverse at every
        # evaluation -- never chained (see ``Det_winv_state``).
        det_winv = _init_det_winv_state(wavefunction_data.geminal_data, A_old_inv, det_ratio_state)
        det_ratio = _compute_ratio_determinant_from_winv_state(
            wavefunction_data.geminal_data,
            candidate_mesh_state.det_candidate_state,
            det_winv,
        )
    # Complex promotion for the THREE ratio casts (det,
    # jastrow, wf_ratio_chunk). The strictly-real ``coulomb`` zone would
    # silently drop ``Im`` BEFORE the outer ``Re(...)`` wrap at the return.
    if flag_determinant_only:
        det_ratio = cast_to_zone(det_ratio, "coulomb")
        _ratio_dtype = det_ratio.dtype
        wf_ratio_all = det_ratio
    else:
        with _ecp_site():
            jastrow_ratio = _compute_ratio_Jastrow_part_split_spin(
                jastrow_data=wavefunction_data.jastrow_data,
                old_r_up_carts=r_up_carts,
                old_r_dn_carts=r_dn_carts,
                new_r_up_shifted=up_mesh_r_up,
                new_r_dn_shifted=dn_mesh_r_dn,
                j3_state=j3_state,
                j1_candidate_state=(candidate_mesh_state.j1_candidate_state if candidate_mesh_state is not None else None),
                j3_candidate_state=(candidate_mesh_state.j3_candidate_state if candidate_mesh_state is not None else None),
                j2_candidate_state=(candidate_mesh_state.j2_candidate_state if candidate_mesh_state is not None else None),
                j_dtn_candidate_state=(
                    candidate_mesh_state.j_dtn_candidate_state if candidate_mesh_state is not None else None
                ),
            )
        det_ratio = cast_to_zone(det_ratio, "coulomb", jastrow_ratio.dtype)
        jastrow_ratio = cast_to_zone(jastrow_ratio, "coulomb", det_ratio.dtype)
        _ratio_dtype = det_ratio.dtype
        wf_ratio_all = det_ratio * jastrow_ratio

    # Split ratios for up/dn blocks to avoid big concat of V_l / cos / weight.
    g_up = V_l_up.shape[1]
    wf_ratio_up = wf_ratio_all[:g_up]
    wf_ratio_dn = wf_ratio_all[g_up:]

    def _contract_chunk(
        V_l_chunk: jax.Array,
        cos_chunk: jax.Array,
        weight_chunk: jax.Array,
        wf_ratio_chunk: jax.Array,
    ) -> _ArrayPair:
        cos_chunk = jnp.array(cos_chunk)
        weight_chunk = jnp.array(weight_chunk)
        wf_ratio_chunk = jnp.array(wf_ratio_chunk, dtype=_ratio_dtype)
        P_l_chunk = vmap(vmap(compute_P_l, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None))(
            jnp.arange(global_max_ang_mom_plus_1), cos_chunk, weight_chunk, wf_ratio_chunk
        )
        # Take the real part of
        # P_l_chunk BEFORE the einsum with the real V_l_chunk so the
        # backward pass through jax.grad (used by the position-derivative
        # path with ``comput_position_deriv=True``) never has to cast a
        # complex cotangent into the real V_l_chunk -- which would fire a
        # ComplexWarning at the implicit dtype demotion. The arithmetic
        # is identical to ``jnp.real(sum(V_l * P_l)) = sum(V_l * Re(P_l))``
        # because V_l is strictly real. Bit-identical at Gamma where
        # P_l_chunk is already real-valued.
        if _is_complex_wf and project_real:
            P_l_chunk = jnp.real(P_l_chunk)
        elif _is_complex_wf:
            # Holomorphic (continuum VMC) path keeps a complex P_l: promote the
            # strictly real V_l to that dtype first so the einsum VJP never casts
            # a complex cotangent into the real V_l leaf (ComplexWarning).
            # Value-identical to the implicit einsum promotion since Im(V_l) = 0.
            V_l_chunk = V_l_chunk.astype(P_l_chunk.dtype)
        V_nonlocal_chunk = jnp.einsum("lg,lg->g", V_l_chunk, P_l_chunk)

        sum_chunk = jnp.sum(V_nonlocal_chunk)
        return V_nonlocal_chunk, sum_chunk

    V_nonlocal_up, sum_up = _contract_chunk(V_l_up, cos_up, weight_up, wf_ratio_up)
    V_nonlocal_dn, sum_dn = _contract_chunk(V_l_dn, cos_dn, weight_dn, wf_ratio_dn)

    V_nonlocal = jnp.concatenate([V_nonlocal_up, V_nonlocal_dn], axis=0)
    sum_V_nonlocal = sum_up + sum_dn
    # Fixed-phase callers project each candidate before sign splitting;
    # continuum VMC callers retain its complex matrix element.

    return non_local_ecp_part_r_carts_up, non_local_ecp_part_r_carts_dn, V_nonlocal, sum_V_nonlocal


def compute_ecp_non_local_parts_nearest_neighbors_fast_update(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
    candidate_mesh_state: "Ecp_candidate_mesh_streaming_state | None" = None,
) -> tuple[list, list, list, float]:
    """Real fixed-phase ECP candidates using the shared inverse-update kernel.

    The inverse and optional streaming states must match the input positions.
    Per-candidate real projection is retained for LRDMC sign splitting.
    """
    return _compute_ecp_non_local_parts_nearest_neighbors_fast_update_impl(
        coulomb_potential_data,
        wavefunction_data,
        r_up_carts,
        r_dn_carts,
        RT,
        A_old_inv,
        NN,
        Nv,
        flag_determinant_only,
        j3_state,
        det_ratio_state,
        candidate_mesh_state,
        True,
    )


def compute_ecp_non_local_parts_nearest_neighbors_fast_update_holomorphic(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
    j3_state: "Jastrow_three_body_streaming_state | None" = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
    candidate_mesh_state: "Ecp_candidate_mesh_streaming_state | None" = None,
) -> tuple[list, list, list, float]:
    """Complex continuum ECP candidates using the same inverse-update kernel.

    Unlike the fixed-phase wrapper, retain Im(Vnl) for VMC H Psi/Psi. All
    geometry, ratio and streaming work is shared; no second ECP implementation.
    The inverse and optional streaming states must match the input positions.
    """
    return _compute_ecp_non_local_parts_nearest_neighbors_fast_update_impl(
        coulomb_potential_data,
        wavefunction_data,
        r_up_carts,
        r_dn_carts,
        RT,
        A_old_inv,
        NN,
        Nv,
        flag_determinant_only,
        j3_state,
        det_ratio_state,
        candidate_mesh_state,
        False,
    )


@partial(jit, static_argnums=(5, 6))
def compute_ecp_non_local_parts_all_pairs(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    Nv: int = Nv_default,
    flag_determinant_only: bool = False,
) -> tuple[list, list, list, float]:
    """Compute non-local ECP contribution considering all nucleus-electron pairs.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        Nv (int): Number of quadrature points on the sphere.
        flag_determinant_only (bool): If True, ignore Jastrow in the wavefunction ratio.

    Returns:
        tuple[list[jax.Array], list[jax.Array], jax.Array, float]:
            - Mesh-displaced ``r_up_carts`` per configuration.
            - Mesh-displaced ``r_dn_carts`` per configuration.
            - Non-local ECP contributions per configuration (flattened).
            - Scalar sum of all non-local contributions.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here (forwarded to fp64 Jastrow/det zones).
    # RT is also forwarded as-is (Principle 3a); cast at the use site below.

    # PBC reject: the all-pairs variant enumerates molecular
    # atoms with no image awareness, which is incompatible with the PBC
    # nearest-image projector. Callers must use
    # ``compute_ecp_non_local_parts_nearest_neighbors`` (the NN-based variant
    # that consumes the minimum-image-convention helpers) instead. ``pbc_flag`` is
    # pytree_node=False so this resolves at trace time.
    if coulomb_potential_data.structure_data.pbc_flag:
        raise NotImplementedError(
            "compute_ecp_non_local_parts_all_pairs is unsupported in PBC "
            "(it enumerates molecular atoms with no image awareness -- use "
            "compute_ecp_non_local_parts_nearest_neighbors instead)."
        )

    dtype_jnp = get_dtype_jnp("coulomb")

    if Nv == 4:
        weights = tetrahedron_sym_mesh_Nv4.weights
        grid_points = tetrahedron_sym_mesh_Nv4.grid_points
    elif Nv == 6:
        weights = octahedron_sym_mesh_Nv6.weights
        grid_points = octahedron_sym_mesh_Nv6.grid_points
    elif Nv == 12:
        weights = icosahedron_sym_mesh_Nv12.weights
        grid_points = icosahedron_sym_mesh_Nv12.grid_points
    elif Nv == 18:
        weights = octahedron_sym_mesh_Nv18.weights
        grid_points = octahedron_sym_mesh_Nv18.grid_points
    else:
        raise NotImplementedError

    # Cast RT to coulomb zone at the use site (Principle 3b).
    grid_points = grid_points @ RT.astype(dtype_jnp)  # rotate the grid points. dim. (N,3) @ (3,3) = (N,3)

    r_up_carts_on_mesh, r_dn_carts_on_mesh, V_ecp_up, V_ecp_dn, sum_V_nonlocal = (
        compute_ecp_non_local_part_all_pairs_jax_weights_grid_points(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            weights=weights,
            grid_points=grid_points,
            flag_determinant_only=int(flag_determinant_only),
        )
    )

    _, uq_indices = np.unique(coulomb_potential_data._nucleus_index_non_local_part, return_index=True)
    r_up_carts_on_mesh = r_up_carts_on_mesh[uq_indices]
    r_dn_carts_on_mesh = r_dn_carts_on_mesh[uq_indices]

    nucleus_index_non_local_part = np.array(coulomb_potential_data._nucleus_index_non_local_part, dtype=np.int32)
    num_segments = len(set(coulomb_potential_data._nucleus_index_non_local_part))
    # NOTE: a masked dense reduce (matmul over the kr-pair axis) instead of two
    # `jax.ops.segment_sum(..., num_segments=num_segments)`, so the
    # XLA scatter-pathology that bites V_l (see compute_ecp_non_local_parts_nearest_neighbors_fast_update)
    # cannot resurface here. n_kr_pairs and num_segments
    # are both small (~tens), so the dense reduce is essentially free.
    nucleus_index_non_local_part_jnp = jnp.asarray(nucleus_index_non_local_part)
    _aggregator_mask = nucleus_index_non_local_part_jnp[:, None] == jnp.arange(num_segments)[None, :]
    V_ecp_up = jnp.einsum("ks,k...->s...", _aggregator_mask.astype(V_ecp_up.dtype), V_ecp_up)
    V_ecp_dn = jnp.einsum("ks,k...->s...", _aggregator_mask.astype(V_ecp_dn.dtype), V_ecp_dn)

    r_up_new_shape = (np.prod(r_up_carts_on_mesh.shape[:3]),) + r_up_carts_on_mesh.shape[3:]
    r_up_carts_on_mesh = r_up_carts_on_mesh.reshape(r_up_new_shape)
    r_dn_new_shape = (np.prod(r_dn_carts_on_mesh.shape[:3]),) + r_dn_carts_on_mesh.shape[3:]
    r_dn_carts_on_mesh = r_dn_carts_on_mesh.reshape(r_dn_new_shape)

    V_ecp_up_new_shape = (np.prod(V_ecp_up.shape[:3]),)
    V_ecp_up = V_ecp_up.reshape(V_ecp_up_new_shape)
    V_ecp_dn_new_shape = (np.prod(V_ecp_dn.shape[:3]),)
    V_ecp_dn = V_ecp_dn.reshape(V_ecp_dn_new_shape)

    # Repeat up-spin electrons for down-spin configurations
    r_up_carts_repeated_dn = jnp.repeat(
        r_up_carts[None, :, :], r_dn_carts_on_mesh.shape[0], axis=0
    )  # Shape: (num_dn_configs, N_up, 3)
    # Repeat down-spin electrons for up-spin configurations
    r_dn_carts_repeated_up = jnp.repeat(
        r_dn_carts[None, :, :], r_up_carts_on_mesh.shape[0], axis=0
    )  # Shape: (num_up_configs, N_dn, 3)
    # Combine configurations
    mesh_non_local_ecp_part_r_up_carts = jnp.concatenate(
        [r_up_carts_on_mesh, r_up_carts_repeated_dn], axis=0
    )  # Shape: (N_configs, N_up, 3)
    mesh_non_local_ecp_part_r_dn_carts = jnp.concatenate(
        [r_dn_carts_repeated_up, r_dn_carts_on_mesh], axis=0
    )  # Shape: (N_configs, N_dn, 3)

    V_nonlocal = jnp.concatenate([V_ecp_up, V_ecp_dn], axis=0)

    return mesh_non_local_ecp_part_r_up_carts, mesh_non_local_ecp_part_r_dn_carts, V_nonlocal, sum_V_nonlocal


@jit  # this jit drastically accelarates the computation!
def compute_ecp_non_local_part_all_pairs_jax_weights_grid_points(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    weights: list,
    grid_points: npt.NDArray[np.float64],
    flag_determinant_only: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    """Vectorized non-local ECP projection over all pairs with provided quadrature.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        weights (list[float]): Quadrature weights for angular integration.
        grid_points (npt.NDArray[np.float64]): Quadrature grid points with shape ``(Nv, 3)``.
        flag_determinant_only (int): If 1, skip Jastrow in the wavefunction ratio; if 0, include it.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
            - Mesh-displaced up-spin coordinates per configuration.
            - Mesh-displaced down-spin coordinates per configuration.
            - Non-local contributions for up-spin mesh points.
            - Non-local contributions for down-spin mesh points.
            - Scalar sum of all non-local contributions.
    """
    coulomb_potential_data._validate_xla_projector_channels()

    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here. They are forwarded to
    # compute_Jastrow_part / compute_det_geminal_all_elements (fp64 zones); pre-casting
    # to the coulomb (fp32) zone would silently down-cast inputs to those fp64 zones.
    dtype_jnp = get_dtype_jnp("coulomb")
    weights = jnp.array(weights, dtype=dtype_jnp)
    grid_points = jnp.array(grid_points, dtype=dtype_jnp)

    jastrow_denominator = lax.switch(
        flag_determinant_only,
        (
            compute_Jastrow_part_ratio_zone,
            lambda *args, **kwargs: jnp.ones((), dtype=get_dtype_jnp("jastrow_ratio")),
        ),
        *(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts),
    )

    det_denominator = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up_carts, r_dn_carts)

    # Compute the local part. To understand the flow, please refer to the debug version.
    @jit
    def compute_V_l(
        r_cart: jax.Array,
        i_atom: jax.Array,
        exponent: jax.Array,
        coefficient: jax.Array,
        power: jax.Array,
    ) -> jax.Array:
        rel_R_cart_min_dist = _get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_cart,
            i_atom=i_atom,
            dtype=dtype_jnp,
        )
        V_l = (
            jnp.linalg.norm(rel_R_cart_min_dist) ** -2.0
            * coefficient
            * jnp.linalg.norm(rel_R_cart_min_dist) ** power
            * jnp.exp(-exponent * (jnp.linalg.norm(rel_R_cart_min_dist) ** 2))
        )

        return V_l

    # Compute the Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    @jit
    def compute_P_l_up(
        ang_mom: jax.Array,
        r_up_i: jax.Array,
        r_up_cart: jax.Array,
        i_atom: jax.Array,
        weight: jax.Array,
        vec_delta: jax.Array,
    ) -> _ArrayPair:
        rel_R_cart_min_dist = _get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_up_cart,
            i_atom=i_atom,
            dtype=dtype_jnp,
        )
        r_up_carts_on_mesh = r_up_carts
        r_up_carts_on_mesh = r_up_carts_on_mesh.at[r_up_i].set(
            r_up_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_up = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )

        jastrow_numerator_up = lax.switch(
            flag_determinant_only,
            (
                compute_Jastrow_part_ratio_zone,
                lambda *args, **kwargs: jnp.ones((), dtype=get_dtype_jnp("jastrow_ratio")),
            ),
            *(wavefunction_data.jastrow_data, r_up_carts_on_mesh, r_dn_carts),
        )

        det_numerator_up = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up_carts_on_mesh, r_dn_carts)

        # Consumer-zone explicit cast: cast ALL upstream values to the local
        # coulomb zone dtype before any arithmetic, so the wf_ratio computation
        # never relies on JAX implicit fp32 x fp64 -> fp64 promotion and never
        # borrows another zone's dtype.
        jastrow_numerator_up = jnp.asarray(jastrow_numerator_up, dtype=dtype_jnp)
        jastrow_denominator_c = jnp.asarray(jastrow_denominator, dtype=dtype_jnp)
        det_numerator_up = jnp.asarray(det_numerator_up, dtype=dtype_jnp)
        det_denominator_c = jnp.asarray(det_denominator, dtype=dtype_jnp)
        wf_ratio_up = jnp.exp(jastrow_numerator_up - jastrow_denominator_c) * det_numerator_up / det_denominator_c

        P_l_up = (2 * ang_mom + 1) * _legendre_tablated(ang_mom, cos_theta_up) * weight * wf_ratio_up

        return r_up_carts_on_mesh, P_l_up

    # Compute the Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    @jit
    def compute_P_l_dn(
        ang_mom: jax.Array,
        r_dn_i: jax.Array,
        r_dn_cart: jax.Array,
        i_atom: jax.Array,
        weight: jax.Array,
        vec_delta: jax.Array,
    ) -> _ArrayPair:
        rel_R_cart_min_dist = _get_min_dist_rel_R_cart_jnp(
            structure_data=coulomb_potential_data.structure_data,
            r_cart=r_dn_cart,
            i_atom=i_atom,
            dtype=dtype_jnp,
        )
        r_dn_carts_on_mesh = r_dn_carts
        r_dn_carts_on_mesh = r_dn_carts_on_mesh.at[r_dn_i].set(
            r_dn_cart + rel_R_cart_min_dist + jnp.linalg.norm(rel_R_cart_min_dist) * vec_delta
        )

        cos_theta_dn = jnp.dot(
            -1.0 * (rel_R_cart_min_dist) / jnp.linalg.norm(rel_R_cart_min_dist),
            ((vec_delta) / jnp.linalg.norm(vec_delta)),
        )

        jastrow_numerator_dn = lax.switch(
            flag_determinant_only,
            (
                compute_Jastrow_part_ratio_zone,
                lambda *args, **kwargs: jnp.ones((), dtype=get_dtype_jnp("jastrow_ratio")),
            ),
            *(wavefunction_data.jastrow_data, r_up_carts, r_dn_carts_on_mesh),
        )

        det_numerator_dn = compute_det_geminal_all_elements(wavefunction_data.geminal_data, r_up_carts, r_dn_carts_on_mesh)

        # Consumer-zone explicit cast (see compute_P_l_up).
        jastrow_numerator_dn = jnp.asarray(jastrow_numerator_dn, dtype=dtype_jnp)
        jastrow_denominator_c = jnp.asarray(jastrow_denominator, dtype=dtype_jnp)
        det_numerator_dn = jnp.asarray(det_numerator_dn, dtype=dtype_jnp)
        det_denominator_c = jnp.asarray(det_denominator, dtype=dtype_jnp)
        wf_ratio_dn = jnp.exp(jastrow_numerator_dn - jastrow_denominator_c) * det_numerator_dn / det_denominator_c

        P_l_dn = (2 * ang_mom + 1) * _legendre_tablated(ang_mom, cos_theta_dn) * weight * wf_ratio_dn
        return r_dn_carts_on_mesh, P_l_dn

    # Vectrize the functions
    vmap_compute_P_l_up = vmap(compute_P_l_up, in_axes=(None, None, None, None, 0, 0))
    vmap_compute_P_l_dn = vmap(compute_P_l_dn, in_axes=(None, None, None, None, 0, 0))

    # Compute the local part V_l * Projection of WF. for a up electron
    # To understand the flow, please refer to the debug version.
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
    @jit
    def compute_V_nonlocal_up(
        r_up_i: jax.Array,
        r_up_cart: jax.Array,
        ang_mom: jax.Array,
        i_atom: jax.Array,
        exponent: jax.Array,
        coefficient: jax.Array,
        power: jax.Array,
    ) -> _ArrayPair:
        V_l_up = compute_V_l(r_up_cart, i_atom, exponent, coefficient, power)
        r_up_carts_on_mesh, P_l_up = vmap_compute_P_l_up(ang_mom, r_up_i, r_up_cart, i_atom, weights, grid_points)
        return r_up_carts_on_mesh, (V_l_up * P_l_up)

    # Compute the local part V_l * Projection of WF. for a down electron
    # To understand the flow, please refer to the debug version.
    # vmap in_axes=(0, 0, None, None, None, None, None) and in_axes=(None, None, 0, 0, 0, 0, 0)
    @jit
    def compute_V_nonlocal_dn(
        r_dn_i: jax.Array,
        r_dn_cart: jax.Array,
        ang_mom: jax.Array,
        i_atom: jax.Array,
        exponent: jax.Array,
        coefficient: jax.Array,
        power: jax.Array,
    ) -> _ArrayPair:
        V_l_dn = compute_V_l(r_dn_cart, i_atom, exponent, coefficient, power)
        r_dn_carts_on_mesh, P_l_dn = vmap_compute_P_l_dn(ang_mom, r_dn_i, r_dn_cart, i_atom, weights, grid_points)
        return r_dn_carts_on_mesh, (V_l_dn * P_l_dn)

    # vectrize compute_ecp_up and compute_ecp_dn
    vmap_vmap_compute_ecp_up = vmap(
        vmap(compute_V_nonlocal_up, in_axes=(0, 0, None, None, None, None, None)), in_axes=(None, None, 0, 0, 0, 0, 0)
    )
    vmap_vmap_compute_ecp_dn = vmap(
        vmap(compute_V_nonlocal_dn, in_axes=(0, 0, None, None, None, None, None)), in_axes=(None, None, 0, 0, 0, 0, 0)
    )

    # Vectrized (flatten) arguments are prepared here.
    # NOTE: Keep r_up_carts/r_dn_carts as fp64 here; the inner closures (compute_P_l_*)
    # forward derived mesh coords to fp64 Jastrow/det zones, so any fp32 down-cast here
    # would propagate to those zones.
    r_up_i_jnp = jnp.arange(len(r_up_carts))
    r_up_carts_jnp = r_up_carts
    r_dn_i_jnp = jnp.arange(len(r_dn_carts))
    r_dn_carts_jnp = r_dn_carts

    i_atom_np = jnp.array(coulomb_potential_data._nucleus_index_non_local_part)
    ang_mom_np = jnp.array(coulomb_potential_data._ang_mom_non_local_part)
    exponent_np = jnp.array(coulomb_potential_data._exponents_non_local_part, dtype=dtype_jnp)
    coefficient_np = jnp.array(coulomb_potential_data._coefficients_non_local_part, dtype=dtype_jnp)
    power_np = jnp.array(coulomb_potential_data._powers_non_local_part, dtype=dtype_jnp)

    # An empty vmap still traces the per-electron scatter, which is invalid
    # for a zero-length spin block. Shapes are static, so normal nonempty
    # calculations retain exactly the same compiled arithmetic.
    if len(r_up_carts):
        r_up_carts_on_mesh, V_ecp_up = vmap_vmap_compute_ecp_up(
            r_up_i_jnp,
            r_up_carts_jnp,
            ang_mom_np,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    else:
        shape = (len(ang_mom_np), 0, len(weights))
        r_up_carts_on_mesh = jnp.empty(shape + (0, 3), dtype=r_up_carts.dtype)
        V_ecp_up = jnp.empty(shape, dtype=dtype_jnp)

    if len(r_dn_carts):
        r_dn_carts_on_mesh, V_ecp_dn = vmap_vmap_compute_ecp_dn(
            r_dn_i_jnp,
            r_dn_carts_jnp,
            ang_mom_np,
            i_atom_np,
            exponent_np,
            coefficient_np,
            power_np,
        )
    else:
        shape = (len(ang_mom_np), 0, len(weights))
        r_dn_carts_on_mesh = jnp.empty(shape + (0, 3), dtype=r_dn_carts.dtype)
        V_ecp_dn = jnp.empty(shape, dtype=dtype_jnp)

    sum_V_nonlocal = jnp.sum(V_ecp_up) + jnp.sum(V_ecp_dn)

    return r_up_carts_on_mesh, r_dn_carts_on_mesh, V_ecp_up, V_ecp_dn, sum_V_nonlocal


@partial(jit, static_argnums=(5, 6))
def compute_ecp_coulomb_potential(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
) -> float:
    """Compute total ECP energy (local + non-local) for a configuration.

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        NN (int): Number of nearest nuclei to include for each electron in the non-local term.
        Nv (int): Number of quadrature points on the sphere.

    Returns:
        float: Sum of local and non-local ECP contributions for the given geometry.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts/RT here (forwarded to downstream
    # functions that handle their own use-site casts -- Principle 3a).

    ecp_local_parts = compute_ecp_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    # Compute inverse geminal matrix once and reuse it for rank-1 det-ratio updates
    # (avoids full LU per mesh point in the slow path, ~10x speedup for ECP non-local)
    G = compute_geminal_all_elements(
        geminal_data=wavefunction_data.geminal_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
    )
    lu, piv = jsp_linalg.lu_factor(G)
    A_old_inv = jsp_linalg.lu_solve((lu, piv), jnp.eye(G.shape[0], dtype=G.dtype))

    _, _, _, ecp_nonlocal_parts = compute_ecp_non_local_parts_nearest_neighbors_fast_update_holomorphic(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        A_old_inv=A_old_inv,
        NN=NN,
        Nv=Nv,
        flag_determinant_only=False,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


def compute_ecp_coulomb_potential_fast(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> jax.Array:
    """Compute total ECP energy (local + non-local) using a pre-computed geminal inverse.

    This is the fast variant of :func:`compute_ecp_coulomb_potential`.  Instead of
    performing a fresh LU factorisation of the current-configuration geminal matrix,
    it accepts ``A_old_inv`` directly.  This avoids NaN when the current configuration
    is near-singular--provided ``A_old_inv`` is the inverse of a nearby, well-conditioned
    reference configuration (e.g., the previous MCMC step).

    Args:
        coulomb_potential_data (Coulomb_potential_data): ECP parameters and structure data.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ratios.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``).
        A_old_inv (jax.Array): Pre-computed inverse of the reference geminal matrix.
        NN (int): Number of nearest nuclei to include for each electron in the non-local term.
        Nv (int): Number of quadrature points on the sphere.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or
            superset). Forwarded to the non-local ECP ratio kernel so it can
            skip the bulk-side AO eval and ``lambda @ ao_*`` precontracts.

    Returns:
        float: Sum of local and non-local ECP contributions for the given geometry.

    Warning:
        ``A_old_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}`` exactly at the
        supplied electron positions.  Correctness is only guaranteed when the inverse is
        maintained via **single-electron (rank-1) Sherman-Morrison updates** starting from
        a freshly initialized LU inverse -- the pattern used in the MCMC loop.  If multiple
        electrons have moved simultaneously, the Sherman-Morrison rank-1 update used inside
        :func:`compute_ecp_non_local_parts_nearest_neighbors_fast_update` becomes incorrect
        and the non-local ratios will be silently wrong.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts/RT here (forwarded to downstream
    # functions that handle their own use-site casts -- Principle 3a).

    ecp_local_parts = compute_ecp_local_parts_all_pairs(
        coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
    )

    _, _, _, ecp_nonlocal_parts = compute_ecp_non_local_parts_nearest_neighbors_fast_update_holomorphic(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        A_old_inv=A_old_inv,
        NN=NN,
        Nv=Nv,
        flag_determinant_only=False,
        det_ratio_state=det_ratio_state,
    )

    V_ecp = ecp_local_parts + ecp_nonlocal_parts

    return V_ecp


def _compute_ecp_coulomb_potential_fast_debug(
    coulomb_potential_data: Coulomb_potential_data,
    wavefunction_data: Wavefunction_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> jax.Array:
    """Debug oracle for :func:`compute_ecp_coulomb_potential_fast`.

    Mirrors the fast signature but discards ``A_old_inv`` (and
    ``det_ratio_state``) and routes through the non-fast sibling
    :func:`compute_ecp_coulomb_potential`, which performs a fresh LU
    factorisation of the current-configuration geminal matrix. The guardrail
    test asserts the fast Sherman-Morrison path reproduces this from-scratch
    reference whenever ``A_old_inv`` is exact (see the ``Warning`` in
    :func:`compute_ecp_coulomb_potential_fast`).
    """
    return compute_ecp_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        NN=NN,
        Nv=Nv,
    )


def _compute_bare_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> float:
    """Debug reference for the total bare Coulomb energy (ion-ion + electron-ion + electron-electron).

    Straightforward (slow) numpy implementation that sums ``Z_a Z_b / |r_a - r_b|`` over every
    distinct particle pair; just for debugging purposes.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). The
    # accumulated scalar is cast to the coulomb zone before return (Principle 3b).
    dtype_np = get_dtype_np("coulomb")

    R_carts = coulomb_potential_data.structure_data._positions_cart_np
    R_charges = coulomb_potential_data._effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    all_carts = np.vstack([R_carts, r_up_carts, r_dn_carts])
    all_charges = np.hstack([R_charges, r_up_charges, r_dn_charges])

    bare_coulomb_potential = np.sum(
        [
            (Z_a * Z_b) / np.linalg.norm(r_a - r_b)
            for (Z_a, r_a), (Z_b, r_b) in itertools.combinations(zip(all_charges, all_carts, strict=True), 2)
        ]
    )

    return np.asarray(bare_coulomb_potential, dtype=dtype_np)


@jit
def compute_bare_coulomb_potential(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Compute the total Coulomb interaction (ion-ion, electron-ion, electron-electron).

    Despite the ``bare`` in the name (preserved for API stability),
    this composite routes through the
    Coulomb dispatch shims and therefore returns the **Ewald lattice
    sum under tin-foil BC** for periodic ``coulomb_potential_data``
    (``structure_data.pbc_flag == True``). For molecular inputs the
    dispatch falls through to the bare 1/r implementations and the
    emitted HLO matches the pre-PBC molecular path byte-for-byte.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure and charges (effective if ECPs present).
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.

    Returns:
        float: Total Coulomb energy (Ewald total for periodic inputs).
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts here. Downstream el_ion / el_el helpers
    # reconstruct r_i - r_j in fp64 internally; pre-casting would destroy that precision.
    interactions_ion_ion = compute_coulomb_potential_ii_dispatch(coulomb_potential_data)
    interactions_el_ion_elements_up, interactions_el_ion_elements_dn = compute_coulomb_potential_ei_dispatch(
        coulomb_potential_data, r_up_carts, r_dn_carts
    )
    interactions_el_el = compute_coulomb_potential_ee_dispatch(coulomb_potential_data, r_up_carts, r_dn_carts)

    return (
        interactions_ion_ion
        + jnp.sum(interactions_el_ion_elements_up)
        + jnp.sum(interactions_el_ion_elements_dn)
        + interactions_el_el
    )


@jit
def compute_bare_coulomb_potential_el_ion_element_wise(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Element-wise electron-ion Coulomb interactions.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure and charges (effective if ECPs present).
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.

    Returns:
        tuple[jax.Array, jax.Array]: Element-wise ion-electron interactions for up spins and down spins (shape ``(N_up,)`` and ``(N_dn,)``).
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts. el_ion_interaction reconstructs
    # r_i - r_j in fp64 internally to avoid catastrophic cancellation; a fp32 pre-cast
    # here would silently destroy that precision before the reconstruction can take effect.
    dtype_jnp = get_dtype_jnp("coulomb")
    dtype_np = get_dtype_np("coulomb")
    R_carts = jnp.array(coulomb_potential_data.structure_data._positions_cart_jnp, dtype=dtype_jnp)
    R_charges = np.array(coulomb_potential_data._effective_charges, dtype=dtype_np)
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=dtype_np)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=dtype_np)

    # Define a function to compute interaction for a pair
    def el_ion_interaction(
        Z_i: jax.Array,
        Z_j: jax.Array,
        r_i: jax.Array,
        r_j: jax.Array,
    ) -> jax.Array:
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # coulomb zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_i - r_j).astype(dtype_jnp)
        distance = jnp.linalg.norm(diff, axis=1)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions_R_r_up = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0))(R_charges, r_up_charges, R_carts, r_up_carts), axis=1
    )
    interactions_R_r_dn = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0))(R_charges, r_dn_charges, R_carts, r_dn_carts), axis=1
    )

    return interactions_R_r_up, interactions_R_r_dn


@jit
def compute_discretized_bare_coulomb_potential_el_ion_element_wise(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    alat: float,
) -> tuple[jax.Array, jax.Array]:
    """Element-wise electron-ion Coulomb interactions with distance floor ``alat``.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure and charges (effective if ECPs present).
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        alat (float): Minimum allowed distance to avoid divergence.

    Returns:
        tuple[jax.Array, jax.Array]: Element-wise ion-electron interactions for up spins and down spins (shape ``(N_up,)`` and ``(N_dn,)``).
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts. el_ion_interaction reconstructs
    # r_i - r_j in fp64 internally to avoid catastrophic cancellation.
    dtype_jnp = get_dtype_jnp("coulomb")
    dtype_np = get_dtype_np("coulomb")
    R_carts = jnp.array(coulomb_potential_data.structure_data._positions_cart_jnp, dtype=dtype_jnp)
    R_charges = np.array(coulomb_potential_data._effective_charges, dtype=dtype_np)
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=dtype_np)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=dtype_np)

    # Define a function to compute interaction for a pair
    def el_ion_interaction(
        Z_i: jax.Array,
        Z_j: jax.Array,
        r_i: jax.Array,
        r_j: jax.Array,
        alat: jax.Array,
    ) -> jax.Array:
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # coulomb zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_i - r_j).astype(dtype_jnp)
        distance = jnp.maximum(jnp.linalg.norm(diff, axis=1), alat)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions_R_r_up = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0, None))(R_charges, r_up_charges, R_carts, r_up_carts, alat),
        axis=1,
    )
    interactions_R_r_dn = jnp.sum(
        jax.vmap(el_ion_interaction, in_axes=(None, 0, None, 0, None))(R_charges, r_dn_charges, R_carts, r_dn_carts, alat),
        axis=1,
    )

    return interactions_R_r_up, interactions_R_r_dn


def _compute_bare_coulomb_potential_el_ion_element_wise_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Debug reference for the element-wise electron-ion bare Coulomb interactions.

    Straightforward (slow) numpy implementation returning the per-electron sums over nuclei;
    just for debugging purposes.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). The
    # accumulators are cast to the coulomb zone before return (Principle 3b).
    dtype_np = get_dtype_np("coulomb")
    R_carts = coulomb_potential_data.structure_data._positions_cart_np  # fp64 storage accessor
    R_charges = coulomb_potential_data._effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    interactions_R_r_up = np.zeros(len(r_up_carts))
    interactions_R_r_dn = np.zeros(len(r_dn_carts))

    for i, (r_up_charge, r_up_cart) in enumerate(zip(r_up_charges, r_up_carts, strict=True)):
        # Reconstruct R - r in caller-supplied precision then downcast to the
        # coulomb zone at the use site (Principle 3b).
        interactions_R_r_up[i] = np.sum(
            [
                (R_charge * r_up_charge) / np.linalg.norm((R_cart - r_up_cart).astype(dtype_np))
                for R_charge, R_cart in zip(R_charges, R_carts, strict=True)
            ]
        )

    for i, (r_dn_charge, r_dn_cart) in enumerate(zip(r_dn_charges, r_dn_carts, strict=True)):
        interactions_R_r_dn[i] = np.sum(
            [
                (R_charge * r_dn_charge) / np.linalg.norm((R_cart - r_dn_cart).astype(dtype_np))
                for R_charge, R_cart in zip(R_charges, R_carts, strict=True)
            ]
        )

    return (
        np.asarray(interactions_R_r_up, dtype=dtype_np),
        np.asarray(interactions_R_r_dn, dtype=dtype_np),
    )


def _compute_discretized_bare_coulomb_potential_el_ion_element_wise_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    alat: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Debug reference for the element-wise electron-ion bare Coulomb interactions with an ``alat`` floor.

    Straightforward (slow) numpy implementation of the per-electron sums over nuclei, with the
    pair distance floored at ``alat``; just for debugging purposes.
    """
    # Forward r_up/dn_carts as-is (Principle 3a -- no parameter rebind). The
    # accumulators are cast to the coulomb zone before return (Principle 3b).
    dtype_np = get_dtype_np("coulomb")
    R_carts = coulomb_potential_data.structure_data._positions_cart_np  # fp64 storage accessor
    R_charges = coulomb_potential_data._effective_charges
    r_up_charges = [-1 for _ in range(len(r_up_carts))]
    r_dn_charges = [-1 for _ in range(len(r_dn_carts))]

    interactions_R_r_up = np.zeros(len(r_up_carts))
    interactions_R_r_dn = np.zeros(len(r_dn_carts))

    for i, (r_up_charge, r_up_cart) in enumerate(zip(r_up_charges, r_up_carts, strict=True)):
        # Reconstruct R - r in caller-supplied precision then downcast to the
        # coulomb zone at the use site (Principle 3b).
        interactions_R_r_up[i] = np.sum(
            [
                (R_charge * r_up_charge) / np.maximum(np.linalg.norm((R_cart - r_up_cart).astype(dtype_np)), alat)
                for R_charge, R_cart in zip(R_charges, R_carts, strict=True)
            ]
        )

    for i, (r_dn_charge, r_dn_cart) in enumerate(zip(r_dn_charges, r_dn_carts, strict=True)):
        interactions_R_r_dn[i] = np.sum(
            [
                (R_charge * r_dn_charge) / np.maximum(np.linalg.norm((R_cart - r_dn_cart).astype(dtype_np)), alat)
                for R_charge, R_cart in zip(R_charges, R_carts, strict=True)
            ]
        )

    return (
        np.asarray(interactions_R_r_up, dtype=dtype_np),
        np.asarray(interactions_R_r_dn, dtype=dtype_np),
    )


@jit
def compute_bare_coulomb_potential_el_el(
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Electron-electron Coulomb interaction energy.

    Args:
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.

    Returns:
        float: Electron-electron Coulomb energy.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts. el_el_interaction reconstructs
    # r_i - r_j in fp64 internally to avoid catastrophic cancellation.
    dtype_np = get_dtype_np("coulomb")
    dtype_jnp = get_dtype_jnp("coulomb")
    r_up_charges = np.full(len(r_up_carts), -1.0, dtype=dtype_np)
    r_dn_charges = np.full(len(r_dn_carts), -1.0, dtype=dtype_np)

    all_charges = np.hstack([r_up_charges, r_dn_charges])
    all_carts = jnp.vstack([r_up_carts, r_dn_carts])

    # Number of particles
    N_np = all_charges.shape[0]
    N_jnp = all_carts.shape[0]

    # Generate all unique pairs indices (i < j)
    idx_i_np, idx_j_np = np.triu_indices(N_np, k=1)
    idx_i_jnp, idx_j_jnp = jnp.triu_indices(N_jnp, k=1)

    # Extract charges and positions for each pair
    Z_i = all_charges[idx_i_np]  # Shape: (M,)
    Z_j = all_charges[idx_j_np]  # Shape: (M,)
    r_i = all_carts[idx_i_jnp]  # Shape: (M, D)
    r_j = all_carts[idx_j_jnp]  # Shape: (M, D)

    # Define a function to compute interaction for a pair
    def el_el_interaction(
        Z_i: jax.Array,
        Z_j: jax.Array,
        r_i: jax.Array,
        r_j: jax.Array,
    ) -> jax.Array:
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # coulomb zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_i - r_j).astype(dtype_jnp)
        distance = jnp.linalg.norm(diff)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions = jax.vmap(el_el_interaction)(Z_i, Z_j, r_i, r_j)  # Shape: (M,)

    # Sum all interactions
    bare_coulomb_potential_el_el = jnp.sum(interactions)

    return bare_coulomb_potential_el_el


@jit
def compute_bare_coulomb_potential_ion_ion(
    coulomb_potential_data: Coulomb_potential_data,
) -> float:
    """Ion-ion Coulomb interaction energy.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure and charges (effective if ECPs present).

    Returns:
        float: Ion-ion Coulomb energy.
    """
    dtype_jnp = get_dtype_jnp("coulomb")
    dtype_np = get_dtype_np("coulomb")
    R_carts = jnp.array(coulomb_potential_data.structure_data._positions_cart_jnp, dtype=dtype_jnp)
    R_charges = np.array(coulomb_potential_data._effective_charges, dtype=dtype_np)

    all_charges = R_charges
    all_carts = R_carts

    # Number of particles
    N_np = all_charges.shape[0]
    N_jnp = all_carts.shape[0]

    # Generate all unique pairs indices (i < j)
    idx_i_np, idx_j_np = np.triu_indices(N_np, k=1)
    idx_i_jnp, idx_j_jnp = jnp.triu_indices(N_jnp, k=1)

    # Extract charges and positions for each pair
    Z_i = all_charges[idx_i_np]  # Shape: (M,)
    Z_j = all_charges[idx_j_np]  # Shape: (M,)
    r_i = all_carts[idx_i_jnp]  # Shape: (M, D)
    r_j = all_carts[idx_j_jnp]  # Shape: (M, D)

    # Define a function to compute interaction for a pair
    def ion_ion_interaction(
        Z_i: jax.Array,
        Z_j: jax.Array,
        r_i: jax.Array,
        r_j: jax.Array,
    ) -> jax.Array:
        # Reconstruct r_i - r_j in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion when one operand is fp64, then downcast to the
        # coulomb zone. Avoids catastrophic cancellation without hardcoding fp64.
        diff = (r_i - r_j).astype(dtype_jnp)
        distance = jnp.linalg.norm(diff)
        interaction = (Z_i * Z_j) / distance
        return interaction

    # Vectorize the function over all pairs
    interactions = jax.vmap(ion_ion_interaction)(Z_i, Z_j, r_i, r_j)  # Shape: (M,)

    # Sum all interactions
    bare_coulomb_potential_ion_ion = jnp.sum(interactions)

    return bare_coulomb_potential_ion_ion


@jit
def compute_bare_coulomb_potential_el_ion(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> float:
    """Total electron-ion Coulomb interaction energy.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure and charges (effective if ECPs present).
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.

    Returns:
        float: Electron-ion Coulomb energy.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts (forwarded to el_ion_element_wise which reconstructs in fp64).
    interactions_el_ion_elements_up, interactions_el_ion_elements_dn = compute_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data, r_up_carts, r_dn_carts
    )

    return jnp.sum(interactions_el_ion_elements_up) + jnp.sum(interactions_el_ion_elements_dn)


# =============================================================================
# Ewald summation (Gamma-point PBC, tin-foil boundary conditions)
# =============================================================================
# Real-space and reciprocal-space tables are precomputed at setup time
# via build_ewald_data and stored on Ewald_data; the per-step runtime
# kernels (compute_ewald_potential_{ee,ei,ii}) consume the tables.

#: Non-ad soft reference for the reciprocal-space extent
#: ``g_max . max(|T_a|, |T_b|, |T_c|) / (2 pi)``. **Not a gate.** Used as a
#: ceiling together with a ``r_real_cut <= L_min`` cap, it would make
#: anisotropic (slab / vacuum) cells infeasible. The QE-style setup selects
#: ``eta`` by minimising the estimated evaluation cost and enumerates
#: whatever real-space image / reciprocal-vector counts the cell needs, so
#: this constant is only a rough documentation scale kept for imports;
#: nothing gates on it.
_EWALD_G_BUDGET = 32.0
#: Number of ``eta`` samples on the log-grid the cost-minimising selector
#: scans (see :py:func:`_eta_iteration_heuristic`). Pure setup-time numpy.
_EWALD_ETA_GRID_POINTS = 200


_EWALD_DEFAULT_TOL = 1e-10


def _enumerate_real_images(
    cell: np.ndarray, r_real_cut: float, *, legacy_envelope: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    r"""Enumerate the lattice translations needed for the short-range Ewald sum.

    Every consumer folds the pair displacement into the centred cell or
    chooses a nearest image, whose norm is no larger. Convexity bounds the
    norm throughout the centred parallelepiped by its eight corners:

    .. math::
       C = \max_{s_i \in \{-1/2,+1/2\}} |s A|.

    Consequently every translation contributing at ``|d + T| <= r_real_cut``
    satisfies ``|T| <= r_real_cut + C``. This holds for skew cells too and
    includes either representative at a half-cell tie. Raw, unfolded pair
    differences are not supported by this bound.

    The kernels retain erfc tails rather than applying a hard cutoff. For
    intentionally undersized custom cutoffs, ``legacy_envelope=True`` uses
    the ``r_real_cut + sum_i |a_i|`` envelope and its finite sum.

    We enumerate exactly those. The enumeration must not cap
    ``r_real_cut`` at the shortest lattice vector; for a slab or vacuum
    cell that cap would drop in-plane images at ``L_min - d``
    even though they sit well inside ``r_real_cut``, preventing real-space
    Ewald convergence. For a compact bulk cell
    the ball is small and this returns a handful-to-dozens of images.

    Always includes :math:`T_0 = (0, 0, 0)` as the first row. fp64 numpy
    throughout.

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, Bohr.
        r_real_cut: Real-space cutoff radius (Bohr).
        legacy_envelope: Use the ``r_real_cut + sum_i |a_i|`` envelope for short custom cutoffs.

    Returns:
        tuple: ``(images_cart, images_int)`` --- Cartesian translations
        with shape ``(N_n, 3)`` fp64 and the corresponding integer
        triples with shape ``(N_n, 3)`` int64. The order is ``T_0`` first,
        then ascending ``|n_a| + |n_b| + |n_c|`` with lexicographic
        tie-break.
    """
    cell = np.asarray(cell, dtype=np.float64)
    if legacy_envelope:
        R = float(r_real_cut) + float(np.sum(np.linalg.norm(cell, axis=1)))
    else:
        corners = np.stack(np.meshgrid(*[[-0.5, 0.5]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        corner_radius = float(np.max(np.linalg.norm(corners @ cell, axis=1)))
        # Outward rounding includes translations exactly on the bounding sphere.
        R = np.nextafter(float(r_real_cut) + corner_radius, np.inf)
    # Per-direction index bound: for T = n @ cell, n = T @ inv(cell), so
    # |n_i| <= |T| * |inv_cell[:, i]| <= R * |inv_cell[:, i]|. The columns of
    # inv(cell) are the (2pi-free) dual basis; |inv_cell[:, i]| = 1 / d_i is
    # the reciprocal of the i-th interplanar spacing.
    inv_cell = np.linalg.inv(cell)
    dual_norms = np.linalg.norm(inv_cell, axis=0)  # |inv_cell[:, i]|
    bounds = np.ceil(R * dual_norms).astype(np.int64) + 1  # +1 guards fp round-off
    n_a = np.arange(-int(bounds[0]), int(bounds[0]) + 1)
    n_b = np.arange(-int(bounds[1]), int(bounds[1]) + 1)
    n_c = np.arange(-int(bounds[2]), int(bounds[2]) + 1)
    na_g, nb_g, nc_g = np.meshgrid(n_a, n_b, n_c, indexing="ij")
    idx = np.stack([na_g.ravel(), nb_g.ravel(), nc_g.ravel()], axis=1).astype(np.int64)
    T_all = idx.astype(np.float64) @ cell  # (M, 3)
    keep = np.linalg.norm(T_all, axis=1) <= R
    idx = idx[keep]
    # Order: T_0 = (0,0,0) first, then ascending L1 with lexicographic tie-break.
    order = sorted(
        range(idx.shape[0]),
        key=lambda k: (
            int(abs(idx[k, 0]) + abs(idx[k, 1]) + abs(idx[k, 2])),
            int(idx[k, 0]),
            int(idx[k, 1]),
            int(idx[k, 2]),
        ),
    )
    indices = idx[order]
    images_cart = indices.astype(np.float64) @ cell
    return images_cart, indices


def _enumerate_g_vectors(
    cell: np.ndarray,
    g_max: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Enumerate non-zero reciprocal vectors :math:`G` with :math:`|G| \le g_\text{max}`.

    Returns the Cartesian G-vectors plus the per-G prefactor
    :math:`(4 \pi / V) \exp(-G^2 / 4 \eta^2) / G^2` used by the
    reciprocal-space Ewald sum. fp64 numpy throughout.

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, Bohr.
        g_max: Reciprocal cutoff (Bohr^-1).
        eta: Ewald split parameter (Bohr^-1).

    Returns:
        tuple: ``(g_vectors, g_factors)`` with shapes ``(N_g, 3)`` and
        ``(N_g,)``, both fp64.
    """
    cell = np.asarray(cell, dtype=np.float64)
    volume = float(np.abs(np.dot(cell[0], np.cross(cell[1], cell[2]))))
    # Reciprocal lattice rows: G_i = 2 pi (T_j x T_k) / V (cyclic permutation).
    recip = (
        (2.0 * np.pi)
        * np.array(
            [
                np.cross(cell[1], cell[2]),
                np.cross(cell[2], cell[0]),
                np.cross(cell[0], cell[1]),
            ],
            dtype=np.float64,
        )
        / volume
    )
    # Per-axis index box. For G = n @ recip the direct lattice vectors are the
    # dual basis (a_i . b_j = 2 pi delta_ij), so n_i = G . a_i / (2 pi) and
    # |G| <= g_max => |n_i| <= g_max |a_i| / (2 pi). This holds for any basis;
    # a bound g_max / min|b_i| is valid only for an orthogonal
    # reciprocal basis and drops G vectors on strongly skewed cell bases.
    bounds = np.ceil(g_max * np.linalg.norm(cell, axis=1) / (2.0 * np.pi)).astype(np.int64) + 1  # +1 guards fp round-off
    # Vectorized enumeration: build all (ga, gb, gc) on a meshgrid, exclude
    # G = 0, then mask by |G| <= g_max. The triple Python loop would dominate
    # setup time on small cells where the bound is >= 20 (e.g. eta ~ 7 on a
    # 2-Bohr cubic Madelung fixture); meshgrid keeps this O(prod(bounds))
    # operation in numpy and finishes in ~100 ms for bounds of 25.
    ga_grid, gb_grid, gc_grid = np.meshgrid(
        np.arange(-int(bounds[0]), int(bounds[0]) + 1),
        np.arange(-int(bounds[1]), int(bounds[1]) + 1),
        np.arange(-int(bounds[2]), int(bounds[2]) + 1),
        indexing="ij",
    )
    ga_int = ga_grid.ravel()
    gb_int = gb_grid.ravel()
    gc_int = gc_grid.ravel()
    ga_flat = ga_int.astype(np.float64)
    gb_flat = gb_int.astype(np.float64)
    gc_flat = gc_int.astype(np.float64)
    nonzero = ~((ga_flat == 0) & (gb_flat == 0) & (gc_flat == 0))
    ga_flat, gb_flat, gc_flat = ga_flat[nonzero], gb_flat[nonzero], gc_flat[nonzero]
    ga_int, gb_int, gc_int = ga_int[nonzero], gb_int[nonzero], gc_int[nonzero]
    G_all = ga_flat[:, None] * recip[0] + gb_flat[:, None] * recip[1] + gc_flat[:, None] * recip[2]
    g_norm = np.linalg.norm(G_all, axis=-1)
    keep = g_norm <= g_max
    g_vectors = G_all[keep]
    g_indices = np.stack([ga_int[keep], gb_int[keep], gc_int[keep]], axis=1).astype(np.int64)
    if g_vectors.shape[0] == 0:
        g_vectors = np.zeros((0, 3), dtype=np.float64)
        g_factors = np.zeros((0,), dtype=np.float64)
        g_indices = np.zeros((0, 3), dtype=np.int64)
    else:
        g2 = g_norm[keep] ** 2
        g_factors = (4.0 * np.pi / volume) * np.exp(-g2 / (4.0 * eta * eta)) / g2
    return g_vectors, g_factors, g_indices


def _eta_iteration_heuristic(
    cell: np.ndarray,
    tol: float,
    eta_override: float | None,
    n_ion: int = 1,
    n_elec: int | None = None,
) -> tuple[float, float, float]:
    r"""Pick ``(eta, r_real_cut, g_max)`` for the given lattice and tolerance.

    QE-style single-knob selection. For any trial ``eta`` the per-channel
    truncation error is pinned to ``tol`` by

    .. math::
       r_\text{real\_cut} = \frac{\sqrt{-\ln\text{tol}}}{\eta}, \qquad
       g_\text{max} = 2\,\eta\,\sqrt{-\ln\text{tol}} ,

    which makes ``erfc(eta r_real_cut) ~ tol`` and
    ``exp(-g_max^2 / (4 eta^2)) = tol``. The **total energy is independent
    of ``eta``** to within ``tol`` for any positive ``eta`` (the Ewald
    identity), so ``eta`` is a pure cost knob: large ``eta`` shrinks the
    real-space image ball but grows the reciprocal-vector count, and vice
    versa. This is exact for **any** cell -- cubic, anisotropic slab with
    vacuum, or monoclinic -- with no ``r_real_cut <= L_min`` cap and no
    scalar ``G``-budget that could make a legitimate cell infeasible. The
    builder therefore never raises for a physical cell, however anisotropic.

    Selection (``eta_override is None``): scan a log-grid of ``eta`` and
    minimise a cheap estimate of the per-evaluation cost

    .. math::
       \text{cost}(\eta) \approx N_\text{tot}^2\, n_\text{images}(\eta)
                                + N_\text{tot}\, n_G(\eta),

    with ``N_tot = n_ion + n_elec``, ``n_images ~ (4pi/3) R^3 / V`` for
    ``R = r_real_cut + sum_i |a_i|`` (the enumeration radius of
    :py:func:`_enumerate_real_images`), and
    ``n_G ~ (4pi/3) g_max^3 / ((2pi)^3 / V)``. The quadratic weight on the
    image term reflects the ``O(N^2)`` real-space pair loop; any reasonable
    monotone model would do -- the *choice* of ``eta`` cannot change the
    energies beyond ``tol`` (pinned by the eta-invariance tests).

    When ``eta_override`` is supplied, skip the scan and return the pinned
    ``(r_real_cut, g_max)`` for that ``eta`` (consumed by the eta-invariance
    tests and the checkpoint-reload path).

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, Bohr.
        tol: Numerical tolerance on the per-channel truncation error
            (default 1e-10 in :py:func:`build_ewald_data`).
        eta_override: Caller-supplied eta to bypass the selection.
            ``None`` selects via the cost scan above.
        n_ion: Number of ions in the cell (cost weight). Default 1.
        n_elec: Typical electron count (cost weight). ``None`` falls back to
            ``n_ion`` (a neutral-cell proxy).

    Returns:
        tuple: ``(eta, r_real_cut, g_max)`` floats, fp64.
    """
    cell = np.asarray(cell, dtype=np.float64)
    volume = float(np.abs(np.linalg.det(cell)))
    log_inv_tol = max(-float(np.log(tol)), 1.0)
    sqrt_log = float(np.sqrt(log_inv_tol))

    if eta_override is not None:
        eta = float(eta_override)
        return eta, sqrt_log / eta, 2.0 * eta * sqrt_log

    face_areas = np.array(
        [
            np.linalg.norm(np.cross(cell[1], cell[2])),
            np.linalg.norm(np.cross(cell[2], cell[0])),
            np.linalg.norm(np.cross(cell[0], cell[1])),
        ],
        dtype=np.float64,
    )
    L_min = float(volume / np.max(face_areas))  # cell inradius (WS diameter bound)
    # Conservative cost estimate, independent of the image-list enumeration:
    # the eta-selection policy does not follow the image list.
    d_cell = float(np.sum(np.linalg.norm(cell, axis=1)))

    n_tot = float(n_ion) + (float(n_elec) if n_elec is not None else float(n_ion))

    # Log-grid of eta centred on the scale where r_real_cut ~ L_min, spanning
    # 1e-2..1e2 of it so the cost minimum is safely bracketed for both compact
    # (reciprocal-dominated) and vacuum (real-space-dominated) cells.
    eta_ref = sqrt_log / L_min
    etas = eta_ref * np.geomspace(1.0e-2, 1.0e2, _EWALD_ETA_GRID_POINTS)
    r_cuts = sqrt_log / etas
    g_maxs = 2.0 * etas * sqrt_log
    R = r_cuts + d_cell
    four_pi_over_3 = 4.0 * np.pi / 3.0
    n_images_est = four_pi_over_3 * R**3 / volume
    v_recip = (2.0 * np.pi) ** 3 / volume
    n_g_est = four_pi_over_3 * g_maxs**3 / v_recip
    cost = n_tot**2 * n_images_est + n_tot * n_g_est
    eta = float(etas[int(np.argmin(cost))])

    return eta, sqrt_log / eta, 2.0 * eta * sqrt_log


def _compute_ewald_ii_energy(
    charges: np.ndarray,
    positions: np.ndarray,
    eta: float,
    real_image_list: np.ndarray,
    g_vectors: np.ndarray,
    g_factors: np.ndarray,
    volume: float,
    cell: np.ndarray,
) -> float:
    r"""Precompute the ion-ion Ewald energy. fp64 numpy.

    Sums four contributions:

      * Real-space ``(1/2) sum_{I,J,n}' Z_I Z_J erfc(eta r) / r``
        where the prime excludes ``(I=J, n=0)``.
      * Reciprocal-space ``(1/2) sum_{G!=0} g_factors[G] |S(G)|^2``
        with structure factor ``S(G) = sum_I Z_I exp(-i G . R_I)``.
      * Per-ion self-energy ``-(eta / sqrt(pi)) sum_I Z_I^2``.
      * Uniform-background (jellium) term
        ``-(pi / (2 eta^2 V)) * (sum_I Z_I)^2`` (the ``G=0`` /
        neutralizing-background contribution for the *charged* ion
        sub-lattice; ``sum_I Z_I = +Q_ion != 0``).

    The ion sub-lattice carries net charge ``+Q_ion`` and is therefore
    NOT neutral on its own, so its background term is non-zero and must
    be retained for the ``ii`` channel to (a) be independent of the
    (non-physical) split ``eta`` and (b) match the standard charged-cell
    Ewald convention (e.g. pyscf ``cell.energy_nuc()``). The matching
    ``ee`` and ``ei`` background terms cancel it for an overall-neutral
    cell, so the *total* Coulomb energy is unchanged.

    The pair displacement ``R_I - R_J`` is folded to the
    parallelepiped-centred image ``d - round(d @ inv(cell)) @ cell``
    before ``real_image_list`` is added, so the result does not depend on
    which periodic copy of a nucleus the input lists (issue #52):
    :py:func:`_enumerate_real_images` is complete only for
    ``|d| <= max_{s_i=+/-1/2} |s @ cell|``, which the fold guarantees.
    ``cell`` holds the lattice vectors as rows, shape ``(3, 3)``, Bohr.
    """
    charges = np.asarray(charges, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    real_image_list = np.asarray(real_image_list, dtype=np.float64)
    g_vectors = np.asarray(g_vectors, dtype=np.float64)
    g_factors = np.asarray(g_factors, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    sqrt_pi = float(np.sqrt(np.pi))

    # Real-space: vectorized over (I, J, n). Fold R_I - R_J to the
    # parallelepiped-centred image first (see docstring).
    R_IJ = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    R_IJ = R_IJ - np.round(R_IJ @ np.linalg.inv(cell)) @ cell
    v_IJn = R_IJ[:, :, None, :] + real_image_list[None, None, :, :]  # (N, N, N_n, 3)
    d_IJn = np.linalg.norm(v_IJn, axis=-1)  # (N, N, N_n)
    safe_d = np.where(d_IJn > 0.0, d_IJn, 1.0)
    contrib = np_erfc(eta * safe_d) / safe_d
    contrib = np.where(d_IJn > 0.0, contrib, 0.0)  # exclude r=0 entries
    ZIZJ = charges[:, None, None] * charges[None, :, None]  # (N, N, 1)
    E_real = 0.5 * float(np.sum(ZIZJ * contrib))

    # Reciprocal-space: structure-factor form.
    GR = g_vectors @ positions.T  # (N_g, N)
    S_real = np.cos(GR) @ charges  # (N_g,)
    S_imag = np.sin(GR) @ charges
    S_abs_sq = S_real * S_real + S_imag * S_imag
    E_recip = 0.5 * float(np.sum(g_factors * S_abs_sq))

    # Self-energy (compensates reciprocal-sum self-image inclusion).
    E_self = -(eta / sqrt_pi) * float(np.sum(charges * charges))

    # Uniform-background (jellium) term for the charged ion sub-lattice.
    Q_tot = float(np.sum(charges))
    E_background = -(np.pi / (2.0 * eta * eta * float(volume))) * Q_tot * Q_tot

    return float(E_real + E_recip + E_self + E_background)


# -----------------------------------------------------------------------------
# In-trace Ewald rebuild for cell-AD.
# -----------------------------------------------------------------------------
# Under stress AD, the cell flows as a JAX leaf, so the Ewald tables T_n,
# G_m, g_factors, and the ion-ion energy must be derived from the cell at
# trace time rather than read from cached buffers. We keep the cached
# fields on Ewald_data as a non-AD fast path for the no-AD case (the
# numpy _compute_ewald_ii_energy oracle above) and add the JAX in-trace
# equivalents below. Storage layout: ``real_image_indices_int`` and
# ``g_indices_int`` (pytree_node=False static tuples) carry the integer
# triples; helpers below reconstitute the Cartesian tables on demand.


@jit
def _ewald_tables_from_cell(
    eta: jax.Array,
    real_image_indices: jax.Array,
    g_indices: jax.Array,
    cell: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Rebuild Ewald Cartesian tables from integer indices + cell.

    Mirrors the numpy setup-time enumeration so the table values are
    bit-identical (up to fp64 reassociation) for the same integer index
    sets. Used by the cell-AD path: the integer index sets are static
    pytree leaves, the cell is a traced JAX leaf, and the tables become
    differentiable functions of cell.

    Args:
        eta: Ewald split parameter, scalar fp64.
        real_image_indices: ``(N_n, 3) int32``, lattice index triples
            (``T_0 = (0, 0, 0)`` first).
        g_indices: ``(N_g, 3) int32``, reciprocal index triples
            (``G != 0``).
        cell: ``(3, 3) fp64`` lattice vectors as rows.

    Returns:
        ``(T_n, G_m, g_factors, volume)`` where ``T_n = idx @ cell``,
        ``G_m = g_idx @ (2 pi inv(cell).T)``, ``g_factors = (4 pi / V)
        exp(-G^2 / 4 eta^2) / G^2``, and ``volume = |det(cell)|``.
    """
    dtype = jnp.float64
    cell = cell.astype(dtype)
    real_image_indices = real_image_indices.astype(dtype)
    g_indices = g_indices.astype(dtype)
    eta = jnp.asarray(eta, dtype=dtype)

    T_n = real_image_indices @ cell  # (N_n, 3)
    # Reciprocal lattice rows: 2 pi * inv(cell).T  =>  G_m = g_idx @ recip.
    recip = (2.0 * jnp.pi) * _inv_3x3(cell).T  # (3, 3)
    G_m = g_indices @ recip  # (N_g, 3)
    volume = jnp.abs(_det_3x3(cell))
    G2 = jnp.sum(G_m * G_m, axis=-1)  # (N_g,)
    # G != 0 by construction (enumerator excludes the origin); the divide
    # is safe at runtime, but use a where-guard so the VJP at G=0 is finite
    # if a downstream caller ever passes the origin index.
    safe_G2 = jnp.where(G2 > 0.0, G2, 1.0)
    g_factors = jnp.where(
        G2 > 0.0,
        (4.0 * jnp.pi / volume) * jnp.exp(-safe_G2 / (4.0 * eta * eta)) / safe_G2,
        0.0,
    )
    return T_n, G_m, g_factors, volume


@jit
def _compute_ewald_ii_energy_jax(
    charges: jax.Array,
    positions: jax.Array,
    eta: jax.Array,
    real_image_indices: jax.Array,
    g_indices: jax.Array,
    cell: jax.Array,
) -> jax.Array:
    r"""In-trace JAX mirror of :py:func:`_compute_ewald_ii_energy`.

    Differentiable through ``positions`` (force path) and ``cell``
    (stress path). The numpy oracle above stays as the setup-time fast
    path and the cross-check reference.
    """
    dtype = jnp.float64
    charges = charges.astype(dtype)
    positions = positions.astype(dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    T_n, G_m, g_factors, volume = _ewald_tables_from_cell(eta, real_image_indices, g_indices, cell)
    sqrt_pi = jnp.sqrt(jnp.asarray(jnp.pi, dtype=dtype))

    # Real-space (mirror numpy oracle: vectorized over (I, J, n)).
    # Fold R_I - R_J to the parallelepiped-centred image BEFORE adding T_n:
    # ``_enumerate_real_images`` only guarantees completeness for
    # displacements inside the centred-cell corner bound, so nuclei outside
    # the home cell would otherwise lose real-space images (issue #52). The
    # fold ``d - round(d @ inv(cell)) @ cell`` keeps ``|d| <= sum_i |a_i| / 2``
    # for any input; the rounded integers are piecewise constant, so the fold
    # stays differentiable w.r.t. positions and cell (strain AD).
    cell_f = cell.astype(dtype)
    R_IJ = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    R_IJ = R_IJ - jnp.round(R_IJ @ _inv_3x3(cell_f)) @ cell_f
    v_IJn = R_IJ[:, :, None, :] + T_n[None, None, :, :]  # (N, N, N_n, 3)
    # SQUARED-distance guard for NaN-free VJP at v=0 (same pattern as ee/ei).
    v_sq = jnp.sum(v_IJn * v_IJn, axis=-1)  # (N, N, N_n)
    safe_v_sq = jnp.where(v_sq > 0.0, v_sq, 1.0)
    d_IJn = jnp.sqrt(safe_v_sq)
    contrib = jnp_erfc(eta * d_IJn) / d_IJn
    contrib = jnp.where(v_sq > 0.0, contrib, 0.0)
    ZIZJ = charges[:, None, None] * charges[None, :, None]  # (N, N, 1)
    E_real = 0.5 * jnp.sum(ZIZJ * contrib)

    # Reciprocal-space (structure factor).
    GR = G_m @ positions.T  # (N_g, N)
    S_real = jnp.cos(GR) @ charges  # (N_g,)
    S_imag = jnp.sin(GR) @ charges
    S_abs_sq = S_real * S_real + S_imag * S_imag
    E_recip = 0.5 * jnp.sum(g_factors * S_abs_sq)

    # Self-energy.
    E_self = -(eta / sqrt_pi) * jnp.sum(charges * charges)

    # Uniform-background (jellium) term for the charged ion sub-lattice
    # (mirror of the numpy oracle _compute_ewald_ii_energy).
    Q_tot = jnp.sum(charges)
    E_background = -(jnp.pi / (2.0 * eta * eta * volume)) * Q_tot * Q_tot

    return E_real + E_recip + E_self + E_background


def _check_charge_neutrality_for_ewald(
    coulomb_potential_data: Coulomb_potential_data,
    n_electron: int,
) -> None:
    r"""Reject non-neutral periodic inputs before Ewald construction.

    The Ewald ``_ee`` / ``_ei`` / ``_ii`` channel formulas in this module
    each carry their own uniform-background (jellium) term for their
    charged sub-system. Those three background terms sum to
    ``-(pi / (2 eta^2 V)) * (sum_all q)^2``, which is zero -- and
    eta-independent -- only when the cell is overall neutral
    (``sum_I Z_I_effective == N_electron``). A non-neutral input would
    leave a residual eta-dependent (and therefore unphysical) total
    Coulomb energy, so this helper fails fast at every periodic
    construction path.

    Effective charges (bare Z for all-electron atoms, ``Z - z_core`` for
    ECP atoms) are summed and compared against ``n_electron``. Both
    quantities are integer-valued in any physical setup, so the
    tolerance is set to ``1e-9`` --- well below the smallest meaningful
    deviation but loose enough to absorb floating-point noise on
    accumulated ``Z`` arrays.

    Args:
        coulomb_potential_data: Provides ``_effective_charges``; the sum
            is the total ionic valence charge in the simulation cell.
        n_electron: Total electron count (``n_up + n_dn``).

    Raises:
        ValueError: When ``|sum_I Z_I_effective - n_electron| > 1e-9``;
            the message reports both numbers so the caller can correct
            the input file or the wavefunction setup.
    """
    total_ion_charge = float(np.sum(np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)))
    if abs(total_ion_charge - float(n_electron)) > ATOL_EWALD_CHARGE_NEUTRALITY:
        raise ValueError(
            "Ewald construction requires a charge-neutral simulation cell, "
            f"but sum(Z_I_effective) = {total_ion_charge} and "
            f"n_electron = {n_electron} (diff = {total_ion_charge - n_electron}). "
            "Charged periodic cells require a uniform-background correction "
            "that is out of scope for this Ewald implementation. Verify that the "
            "input file electron count matches the total ionic valence "
            "charge after ECP core removal."
        )


def build_ewald_data(
    structure_data: Structure_data,
    coulomb_potential_data: Coulomb_potential_data,
    tol: float = _EWALD_DEFAULT_TOL,
    eta_override: float | None = None,
    r_real_cut_override: float | None = None,
    g_max_override: float | None = None,
    n_electron: int | None = None,
    *,
    real_image_indices_override: tuple[int, ...] | None = None,
) -> Ewald_data:
    r"""Precompute the Ewald-summation tables for a periodic system.

    Calls the parameter-selection step to choose or restore
    ``(eta, r_real_cut, g_max)``, enumerates real-space lattice images and
    nonzero G-vectors, and precomputes the ion-ion Ewald energy. Detailed
    numerical derivations of the selection and channel decomposition are work
    in progress. The resulting :py:class:`Ewald_data` is stored on
    ``Coulomb_potential_data.ewald_data`` by the TREXIO loader (strict
    contract: ``pbc_flag == True iff ewald_data is not None``).

    fp64 numpy throughout (the ``ewald`` precision zone; see
    :mod:`jqmc._precision`). Setup-time only; never on the MCMC hot path.

    Charge neutrality is required for production periodic systems:
    ``sum_I Z_I_effective == N_electron``. Callers that pass ``n_electron``
    get this checked at construction time; omitting it skips the check and is
    intended only for test fixtures.

    Args:
        structure_data: Periodic lattice + atomic positions.
        coulomb_potential_data: Provides the effective charges
            (``_effective_charges``: bare Z for all-electron, ``Z - z_core``
            for ECP).
        tol: Per-channel truncation tolerance. Default ``1e-10``;
            ``1e-12`` is the default for
            the AO image-list truncation (a separate knob).
        eta_override: Optional caller-supplied eta to bypass the iteration
            heuristic --- used by the eta-invariance test and by the
            checkpoint reload path to restore the exact saved split.
        r_real_cut_override: Optional caller-supplied real-space cutoff
            (Bohr). Any subset of the three setup scalars may be
            overridden independently; unspecified ones are derived from
            ``tol`` (and, for ``eta``, the cost selector). Supplying all
            three together reproduces a saved run bit-for-bit (checkpoint
            reload) regardless of later heuristic / ``tol`` changes.
        g_max_override: Optional caller-supplied reciprocal cutoff
            (Bohr^-1). See ``r_real_cut_override``.
        n_electron: Total electron count (up + dn). When supplied,
            enforces charge neutrality via
            :py:func:`_check_charge_neutrality_for_ewald`; mismatched
            inputs raise ``ValueError``. ``None`` skips the check.
        real_image_indices_override: Saved flattened integer triples, origin
            first. Restore a checkpoint's exact real-space topology rather
            than selecting it again, whatever enumeration policy wrote it.

    Returns:
        Ewald_data: Populated with ``eta``, ``r_real_cut``, ``g_max``,
        ``real_image_list``, ``g_vectors``, ``g_factors``, and ``ii_energy``.
        The ``structure_data`` field is set to the input ``structure_data``.

    Raises:
        ValueError: When ``structure_data.pbc_flag`` is ``False`` or when
            ``n_electron`` is supplied and disagrees with the total ionic
            valence charge.
    """
    if not structure_data.pbc_flag:
        raise ValueError("build_ewald_data requires structure_data.pbc_flag=True.")
    if n_electron is not None:
        _check_charge_neutrality_for_ewald(coulomb_potential_data, n_electron)
    cell = np.asarray(structure_data.cell, dtype=np.float64)
    # Any subset of (eta, r_real_cut, g_max) may be overridden; the rest are
    # derived from ``tol`` (via the eta cost-selector for eta, and the pinned
    # closed forms for the cutoffs). Supplying all three reproduces a saved
    # run bit-for-bit (checkpoint reload); supplying only eta is the
    # eta-invariance path; supplying e.g. only r_real_cut tightens one knob.
    n_ion = int(structure_data.natom)
    eta_h, r_h, g_h = _eta_iteration_heuristic(cell, tol, eta_override, n_ion=n_ion, n_elec=n_electron)
    eta = eta_h  # already == eta_override when that was supplied
    r_real_cut = float(r_real_cut_override) if r_real_cut_override is not None else r_h
    g_max = float(g_max_override) if g_max_override is not None else g_h
    # A short explicit cutoff can intentionally retain appreciable erfc tails
    # in the ``r_real_cut + sum_i |a_i|`` envelope; that finite sum is kept. Use only saved scalars
    # and the existing default tolerance so reload does not depend on its tol.
    legacy_envelope = r_real_cut < float(np.sqrt(-np.log(_EWALD_DEFAULT_TOL))) / eta
    if real_image_indices_override is None:
        real_image_list, real_image_indices = _enumerate_real_images(cell, r_real_cut, legacy_envelope=legacy_envelope)
    else:
        saved_indices = np.asarray(real_image_indices_override)
        if (
            saved_indices.ndim != 1
            or saved_indices.size == 0
            or saved_indices.size % 3
            or not np.issubdtype(saved_indices.dtype, np.integer)
            or np.any(saved_indices[:3] != 0)
        ):
            raise ValueError("Saved Ewald real image indices must be flattened integer triples with the origin first.")
        real_image_indices = saved_indices.astype(np.int64).reshape(-1, 3)
        real_image_list = real_image_indices.astype(np.float64) @ cell
    g_vectors, g_factors, g_indices = _enumerate_g_vectors(cell, g_max, eta)
    charges = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    positions = np.asarray(structure_data._positions_cart_np, dtype=np.float64)
    volume = float(np.abs(np.linalg.det(cell)))
    ii_energy = _compute_ewald_ii_energy(
        charges,
        positions,
        eta,
        real_image_list,
        g_vectors,
        g_factors,
        volume,
        cell,
    )
    return Ewald_data(
        structure_data=structure_data,
        eta=float(eta),
        r_real_cut=float(r_real_cut),
        g_max=float(g_max),
        real_image_list=jnp.asarray(real_image_list, dtype=jnp.float64),
        g_vectors=jnp.asarray(g_vectors, dtype=jnp.float64),
        g_factors=jnp.asarray(g_factors, dtype=jnp.float64),
        ii_energy=float(ii_energy),
        real_image_indices=tuple(int(x) for x in real_image_indices.reshape(-1)),
        g_indices=tuple(int(x) for x in g_indices.reshape(-1)),
    )


@jit
def compute_ewald_potential_ii(
    coulomb_potential_data: Coulomb_potential_data,
    ewald_data: Ewald_data,
) -> jax.Array:
    r"""Return the ion-ion Ewald energy (per geometry, scalar).

    The in-trace JAX path
    :py:func:`_compute_ewald_ii_energy_jax` rebuilds the Cartesian tables
    from the integer index sets ``ewald_data.real_image_indices`` and
    ``ewald_data.g_indices`` and ``ewald_data.structure_data._cell_jnp``,
    so the energy is differentiable w.r.t. both positions (force path)
    and cell (stress path). The cached ``ii_energy`` field on
    ``ewald_data`` is a setup-time value for the structure given to
    :py:func:`build_ewald_data`; no runtime path consumes it. The native
    Rust route likewise rebuilds the tables from the same index sets and
    evaluates the ion-ion energy at ``ewald_data.structure_data``.

    Args:
        coulomb_potential_data: Provides ``_effective_charges``.
        ewald_data: Carries ``eta`` plus the integer index sets and the
            structure (positions + cell) the tables resolve against.

    Returns:
        jax.Array: Scalar Ewald ion-ion energy (Ha), ``ewald`` zone fp64.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    charges = jnp.asarray(coulomb_potential_data._effective_charges, dtype=dtype_jnp)
    positions = ewald_data.structure_data._positions_cart_jnp.astype(dtype_jnp)
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    return _compute_ewald_ii_energy_jax(
        charges,
        positions,
        jnp.asarray(ewald_data.eta, dtype=dtype_jnp),
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )


def _compute_ewald_potential_ii_debug(
    coulomb_potential_data: Coulomb_potential_data,
    ewald_data: Ewald_data,
) -> float:
    r"""Self-contained reference for :py:func:`compute_ewald_potential_ii`.

    Recomputes the ion-ion Ewald energy from scratch using only
    ``ewald_data`` table values (``eta``, ``real_image_list``,
    ``g_vectors``, ``g_factors``) plus the system charges and positions.
    Does NOT call :py:func:`build_ewald_data`, :py:func:`_enumerate_real_images`,
    :py:func:`_enumerate_g_vectors`, :py:func:`_eta_iteration_heuristic`,
    or :py:func:`_compute_ewald_ii_energy` --- the helper chain is
    re-implemented inline with explicit Python loops so a single
    bit-identical agreement against ``ewald_data.ii_energy`` validates
    the entire production helper chain.

    fp64 numpy; setup-time / test-time only.
    """
    charges = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    positions = np.asarray(
        coulomb_potential_data.structure_data._positions_cart_np,
        dtype=np.float64,
    )
    eta = float(ewald_data.eta)
    real_images = np.asarray(ewald_data.real_image_list, dtype=np.float64)
    g_vectors = np.asarray(ewald_data.g_vectors, dtype=np.float64)
    g_factors = np.asarray(ewald_data.g_factors, dtype=np.float64)
    sqrt_pi = float(np.sqrt(np.pi))
    N_atom = positions.shape[0]

    # Real-space contribution. Decompose into distinct-atom pairs (I < J,
    # all images including n=0) plus self-image pairs (I = J, n != 0,
    # with 1/2 to compensate the T <-> -T double-listing in real_image_list).
    #
    # Each pair displacement is first folded to the parallelepiped-centred
    # image r - round(r @ inv(cell)) @ cell, so a nucleus listed several
    # cells outside the home cell keeps all real-space images (issue #52).
    cell = np.asarray(coulomb_potential_data.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    E_real_pairs = 0.0
    for I in range(N_atom):
        for J in range(I + 1, N_atom):
            r_IJ = positions[J] - positions[I]
            n_IJ = [float(np.round(np.dot(r_IJ, inv_cell[:, k]))) for k in range(3)]
            r_IJ = r_IJ - (n_IJ[0] * cell[0] + n_IJ[1] * cell[1] + n_IJ[2] * cell[2])
            for T in real_images:
                d = float(np.linalg.norm(r_IJ + T))
                if d > 0.0:
                    E_real_pairs += charges[I] * charges[J] * float(np_erfc(eta * d)) / d
    E_real_self = 0.0
    for I in range(N_atom):
        for T in real_images:
            d = float(np.linalg.norm(T))
            if d > 0.0:
                E_real_self += 0.5 * charges[I] * charges[I] * float(np_erfc(eta * d)) / d

    # Reciprocal-space contribution. Re-implements the structure-factor
    # form ``E_recip = 0.5 * sum_G g_factors[G] * |S(G)|^2`` independently
    # from :py:func:`_compute_ewald_ii_energy` (no helper call). The
    # numpy broadcast ``g_vectors @ positions.T`` is just dense linear
    # algebra --- no JIT, no vmap, no lax.scan; the self-contained _debug
    # rule excludes JAX kernel fusion and our own production helpers, not
    # primitive numpy operations.
    phases = g_vectors @ positions.T  # (N_g, N_atom)
    S_real_vec = (charges[None, :] * np.cos(phases)).sum(axis=-1)  # (N_g,)
    S_imag_vec = (charges[None, :] * np.sin(phases)).sum(axis=-1)
    E_recip = 0.5 * float(np.sum(g_factors * (S_real_vec**2 + S_imag_vec**2)))

    # Self-energy.
    E_self = 0.0
    for I in range(N_atom):
        E_self += -(eta / sqrt_pi) * charges[I] * charges[I]

    # Uniform-background (jellium) term for the charged ion sub-lattice.
    volume = float(np.abs(np.linalg.det(cell)))
    Q_tot = float(np.sum(charges))
    E_background = -(np.pi / (2.0 * eta * eta * volume)) * Q_tot * Q_tot

    return float(E_real_pairs + E_real_self + E_recip + E_self + E_background)


@jit
def compute_ewald_potential_ee(
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    ewald_data: Ewald_data,
) -> jax.Array:
    r"""Pair-level Ewald electron-electron Coulomb potential under tin-foil BC.

    Returns the scalar Ewald e-e energy for the supplied electron positions.
    The ``ee`` channel includes its own self/background terms; cross-channel
    cancellation for a neutral cell and detailed numerical derivations are
    work in progress.

    fp64 throughout (``ewald`` zone). Cartesian tables are rebuilt in-trace
    from the integer index sets chosen by :py:func:`build_ewald_data` and
    ``ewald_data.structure_data.cell``; cached ``real_image_list`` /
    ``g_vectors`` / ``g_factors`` are not read.

    Args:
        r_up_carts: Up-spin electron positions, shape ``(N_up, 3)``, fp64.
        r_dn_carts: Down-spin electron positions, shape ``(N_dn, 3)``, fp64.
        ewald_data: Precomputed Ewald tables for the periodic system.

    Returns:
        jax.Array: Scalar e-e Ewald energy (Ha), ``ewald`` zone fp64.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    # Rebuild tables in-trace from integer indices +
    # cell so the energy is differentiable through cell. The cached
    # ``real_image_list`` / ``g_vectors`` / ``g_factors`` on ewald_data are
    # retained for the HDF5 round-trip / non-AD debug oracle but are no
    # longer consumed on the runtime path.
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    real_images, g_vectors, g_factors, _ = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )

    # Concatenate up + down electrons (all have charge q = -1; q^2 = 1).
    # Reconstruct r-r' in caller-supplied precision (fp64 walker state),
    # then downcast at use site (Principle 3a/3b).
    r_all = jnp.concatenate(
        [r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)],
        axis=0,
    )
    N_e = r_all.shape[0]
    sqrt_pi = jnp.sqrt(jnp.asarray(jnp.pi, dtype=dtype_jnp))

    # Real-space: (1/2) sum_{i,j,n} (skip i=j and n=0) erfc(eta r) / r.
    # q_i q_j = (-1)(-1) = +1, so the charge prefactor is unity. The
    # (1/2) handles the (i, j) ordering double-count plus the (T, -T)
    # double-listing for i=j self-images.
    #
    # Wrap ``r_ij`` to the Wigner-Seitz minimum image before adding the
    # truncated ``real_images`` list. The cutoff ``r_real_cut =
    # sqrt(-log tol)/eta`` is typically smaller than the smallest lattice
    # vector, so ``real_images = [(0,0,0)]`` is the only entry; using the
    # raw home-cell ``r_ij`` would drop the dominant erfc contribution
    # whenever an electron sits near a different periodic copy of its
    # partner. The reciprocal-space sum is intrinsically periodic and
    # does not compensate -- without the wrap, Gamma-point Bloch
    # translation invariance is broken by an order-Ha amount per close
    # pair (pinned by
    # ``test_compute_ewald_potential_ee_pbc_lattice_translation_invariance``).
    inv_cell = _inv_3x3(cell)
    n_max = ewald_data.structure_data._n_max
    r_ij = r_all[:, None, :] - r_all[None, :, :]  # (N, N, 3)
    r_ij = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_ij)
    v_ijn = r_ij[:, :, None, :] + real_images[None, None, :, :]  # (N, N, N_n, 3)
    # Guard against r=0 (i=j, n=0 self-pair, ALWAYS present on the diagonal)
    # at the SQUARED-distance level: jnp.sum(v**2) has a finite gradient (2v,
    # =0 at v=0), whereas jnp.linalg.norm(v) has the 0/0 = NaN VJP at v=0. The
    # outer jnp.where on v_sq_ijn then excludes those self entries from the sum
    # while keeping the backward pass NaN-free (the masked branch never feeds a
    # NaN cotangent). Off-diagonal forward values are unchanged (sqrt(sum v^2)).
    v_sq_ijn = jnp.sum(v_ijn**2, axis=-1)  # (N, N, N_n)
    safe_v_sq = jnp.where(v_sq_ijn > 0.0, v_sq_ijn, 1.0)
    d_ijn = jnp.sqrt(safe_v_sq)  # (N, N, N_n)
    contrib = jnp_erfc(eta * d_ijn) / d_ijn
    contrib = jnp.where(v_sq_ijn > 0.0, contrib, 0.0)  # exclude r=0 self-pairs
    E_real = 0.5 * jnp.sum(contrib)

    # Reciprocal-space: structure factor. q_e = -1 => S_real picks up the
    # sign, but |S(G)|^2 = S_real^2 + S_imag^2 absorbs it.
    GR = g_vectors @ r_all.T  # (N_g, N_e)
    S_real_vec = jnp.cos(GR).sum(axis=-1)  # (N_g,)
    S_imag_vec = jnp.sin(GR).sum(axis=-1)
    E_recip = 0.5 * jnp.sum(g_factors * (S_real_vec**2 + S_imag_vec**2))

    # Self-energy: -(eta / sqrt(pi)) sum_i q_i^2 = -(eta / sqrt(pi)) N_e.
    E_self = -(eta / sqrt_pi) * jnp.asarray(N_e, dtype=dtype_jnp)

    # Uniform-background (jellium) term for the charged electron gas.
    # sum_i q_i = -N_e, so (sum q)^2 = N_e^2. This cancels the ion-ion and
    # electron-ion background terms for an overall-neutral cell, leaving the
    # total Coulomb energy unchanged, while making the ee channel eta-
    # independent and matching the standard charged-cell Ewald convention.
    volume = jnp.abs(_det_3x3(cell))
    Ne_f = jnp.asarray(N_e, dtype=dtype_jnp)
    E_background = -(jnp.asarray(jnp.pi, dtype=dtype_jnp) / (2.0 * eta * eta * volume)) * Ne_f * Ne_f

    return E_real + E_recip + E_self + E_background


# ---------------------------------------------------------------------------
# Incremental (single-electron-move) Ewald ee state for the GFMC/LRDMC
# projection interval.
# XLA mirror of the Rust ``ewald.rs::EwaldEeState``: the projection hop moves
# exactly ONE electron, so the ee energy admits a carried state -- the
# real-space running half-sum plus the electron structure factor S(G) --
# advanced per hop at O(N*n_img + n_g) instead of the full O(N^2*n_img)
# recompute. The build is numerically the SAME reductions as
# ``compute_ewald_potential_ee`` (bit-identical at the build configuration);
# only later ``advance`` steps introduce fp-reordering drift, pinned by the
# incremental-vs-full twin obligation (IMPL-GFMC-INCREMENTAL-TWIN-001).
# ---------------------------------------------------------------------------


def compute_ewald_ee_interval_state(
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    ewald_data: Ewald_data,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Build the carried ee Ewald interval state ``(E_real_half, S_re, S_im)``.

    Detailed interval-state estimator construction is work in progress. The
    returned state is the input to
    :func:`compute_ewald_ee_from_interval_state`.

    Args:
        r_up_carts: Up-spin electron positions ``(N_up, 3)`` fp64.
        r_dn_carts: Down-spin electron positions ``(N_dn, 3)`` fp64.
        ewald_data: Precomputed Ewald tables.

    Returns:
        tuple: ``(E_real_half (scalar), S_re (N_g,), S_im (N_g,))`` fp64.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    real_images, g_vectors, _, _ = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )
    r_all = jnp.concatenate(
        [r_up_carts.astype(dtype_jnp), r_dn_carts.astype(dtype_jnp)],
        axis=0,
    )
    inv_cell = _inv_3x3(cell)
    n_max = ewald_data.structure_data._n_max
    r_ij = r_all[:, None, :] - r_all[None, :, :]
    r_ij = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_ij)
    v_ijn = r_ij[:, :, None, :] + real_images[None, None, :, :]
    v_sq_ijn = jnp.sum(v_ijn**2, axis=-1)
    safe_v_sq = jnp.where(v_sq_ijn > 0.0, v_sq_ijn, 1.0)
    d_ijn = jnp.sqrt(safe_v_sq)
    contrib = jnp_erfc(eta * d_ijn) / d_ijn
    contrib = jnp.where(v_sq_ijn > 0.0, contrib, 0.0)
    E_real_half = 0.5 * jnp.sum(contrib)
    GR = g_vectors @ r_all.T
    S_re = jnp.cos(GR).sum(axis=-1)
    S_im = jnp.sin(GR).sum(axis=-1)
    return E_real_half, S_re, S_im


def compute_ewald_ee_from_interval_state(
    E_real_half: jax.Array,
    S_re: jax.Array,
    S_im: jax.Array,
    n_electron: int,
    ewald_data: Ewald_data,
) -> jax.Array:
    r"""Total ee Ewald energy from the carried interval state.

    Consumes the state produced by :func:`compute_ewald_ee_interval_state`
    and returns the scalar ee Ewald energy. Detailed numerical composition is
    work in progress.

    Args:
        E_real_half: Carried real-space half-sum (scalar).
        S_re: Electron structure factor, real half ``(N_g,)``.
        S_im: Electron structure factor, imaginary half ``(N_g,)``.
        n_electron: Static total electron count.
        ewald_data: Precomputed Ewald tables.

    Returns:
        jax.Array: Scalar ee Ewald energy (Ha).
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    _, _, g_factors, volume = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )
    sqrt_pi = jnp.sqrt(jnp.asarray(jnp.pi, dtype=dtype_jnp))
    E_recip = 0.5 * jnp.sum(g_factors * (S_re**2 + S_im**2))
    E_self = -(eta / sqrt_pi) * jnp.asarray(n_electron, dtype=dtype_jnp)
    Ne_f = jnp.asarray(n_electron, dtype=dtype_jnp)
    E_background = -(jnp.asarray(jnp.pi, dtype=dtype_jnp) / (2.0 * eta * eta * volume)) * Ne_f * Ne_f
    return E_real_half + E_recip + E_self + E_background


def _ewald_ee_point_image_sum(
    point: jax.Array,
    r_others: jax.Array,
    eta: jax.Array,
    real_images: jax.Array,
    cell: jax.Array,
    inv_cell: jax.Array,
    n_max: int,
    exclude_index: jax.Array | None = None,
) -> jax.Array:
    r"""``sum_j sum_n erfc(eta d)/d`` of points against a set of electrons.

    ``d = |min_image(point - r_j) + T_n|`` with the ``d = 0`` self term masked
    -- the real-space kernel of the incremental ee update. A single ``(3,)``
    point returns a scalar; ``(P, 3)`` returns one sum per point. Keeping the
    point axis between partner and image axes batches old/new evaluations.
    An optional moved row is masked before reduction, avoiding subtraction of
    a huge artificial self interaction while retaining the full-N array layout.
    """
    dtype_jnp = get_dtype_jnp("ewald")

    def displacements(p):
        diff = p[None, :].astype(dtype_jnp) - r_others.astype(dtype_jnp)
        metric = cell @ cell.T
        off_diagonal = metric - jnp.diag(jnp.diag(metric))

        def orthogonal(ds):
            fractional = ds @ inv_cell
            return (fractional - jnp.floor(fractional + 0.5)) @ cell

        def general(ds):
            return jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max))(ds)

        # Exact, dynamic metric check: a sheared cell must use the general
        # nearest-image search, including when the cell is an AD input.
        return jax.lax.cond(jnp.all(off_diagonal == 0), orthogonal, general, diff)

    diff = jax.vmap(displacements, out_axes=1)(jnp.atleast_2d(point))
    v = diff[:, :, None, :] + real_images[None, None, :, :]
    v_sq = jnp.sum(v**2, axis=-1)
    safe_v_sq = jnp.where(v_sq > 0.0, v_sq, 1.0)
    d = jnp.sqrt(safe_v_sq)
    contrib = jnp_erfc(eta * d) / d
    contrib = jnp.where(v_sq > 0.0, contrib, 0.0)
    if exclude_index is not None:
        contrib = jnp.where((jnp.arange(r_others.shape[0]) != exclude_index)[:, None, None], contrib, 0.0)
    sums = jnp.sum(contrib, axis=(0, 2))
    return sums[0] if point.ndim == 1 else sums


def advance_ewald_ee_interval_state(
    E_real_half: jax.Array,
    S_re: jax.Array,
    S_im: jax.Array,
    r_all_old: jax.Array,
    old_pos: jax.Array,
    new_pos: jax.Array,
    ewald_data: Ewald_data,
    moved_index: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""One-electron-move update of the carried ee Ewald interval state.

    Returns an updated ``(E_real_half, S_re, S_im)`` for a single electron hop.
    Detailed periodic estimator derivation is work in progress.

    ``old_pos`` must be a row of ``r_all_old``. If ``old == new``, every
    update delta is exactly zero.

    Args:
        E_real_half: Carried real-space half-sum (scalar).
        S_re: Electron structure factor, real half ``(N_g,)``.
        S_im: Electron structure factor, imaginary half ``(N_g,)``.
        r_all_old: Pre-hop electron positions ``(N, 3)`` (up then dn).
        old_pos: Moved electron's pre-hop position ``(3,)``.
        new_pos: Moved electron's post-hop position ``(3,)``.
        ewald_data: Precomputed Ewald tables.
        moved_index: Index in ``r_all_old`` when known by the caller. Omitting
            it preserves the position-based API; production callers pass
            their already-known index to avoid a redundant row search.

    Returns:
        tuple: Updated ``(E_real_half, S_re, S_im)``.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    real_images, g_vectors, _, _ = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )
    inv_cell = _inv_3x3(cell)
    n_max = ewald_data.structure_data._n_max
    if moved_index is None:
        moved_index = jnp.argmax(jnp.all(r_all_old == old_pos[None, :], axis=1))
    # Compile-time batching choice only; both arms sum the same N-1 partners.
    # Separate point evaluations avoid a large doubled image temporary.
    if r_all_old.shape[0] < 128:
        pair_sums = _ewald_ee_point_image_sum(
            jnp.stack((new_pos, old_pos)), r_all_old, eta, real_images, cell, inv_cell, n_max, moved_index
        )
        d_real = pair_sums[0] - pair_sums[1]
    else:
        phi_new = _ewald_ee_point_image_sum(new_pos, r_all_old, eta, real_images, cell, inv_cell, n_max, moved_index)
        phi_old = _ewald_ee_point_image_sum(old_pos, r_all_old, eta, real_images, cell, inv_cell, n_max, moved_index)
        d_real = phi_new - phi_old
    gr_new = g_vectors @ new_pos.astype(dtype_jnp)
    gr_old = g_vectors @ old_pos.astype(dtype_jnp)
    S_re_new = S_re + (jnp.cos(gr_new) - jnp.cos(gr_old))
    S_im_new = S_im + (jnp.sin(gr_new) - jnp.sin(gr_old))
    return E_real_half + d_real, S_re_new, S_im_new


def _compute_ewald_potential_ee_debug(
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    ewald_data: Ewald_data,
) -> float:
    r"""Self-contained reference for :py:func:`compute_ewald_potential_ee`.

    Re-implements the Ewald e-e formula from scratch using only the
    ``ewald_data`` table values (``eta``, ``real_image_list``,
    ``g_vectors``, ``g_factors``) and the input electron coordinates.
    Does NOT call :py:func:`build_ewald_data`,
    :py:func:`_enumerate_real_images`, :py:func:`_enumerate_g_vectors`,
    or :py:func:`_compute_ewald_ii_energy` --- a single bit-identical
    comparison against the production
    :py:func:`compute_ewald_potential_ee` validates the helper chain.

    fp64 numpy; setup-time / test-time only. Reciprocal-space uses
    numpy broadcasting (not a production helper, not a JAX kernel) to keep
    the runtime tractable on large G-vector counts; real-space stays
    explicit-loop for readability.
    """
    eta = float(ewald_data.eta)
    real_images = np.asarray(ewald_data.real_image_list, dtype=np.float64)
    g_vectors = np.asarray(ewald_data.g_vectors, dtype=np.float64)
    g_factors = np.asarray(ewald_data.g_factors, dtype=np.float64)
    r_all = np.concatenate(
        [
            np.asarray(r_up_carts, dtype=np.float64),
            np.asarray(r_dn_carts, dtype=np.float64),
        ],
        axis=0,
    )
    N_e = r_all.shape[0]
    sqrt_pi = float(np.sqrt(np.pi))

    # Real-space contribution. Distinct pairs (i < j) iterate all images;
    # same-electron self-image pairs (i = j, T != 0) require 1/2 to
    # compensate the T <-> -T double listing in real_images. All q^2 = 1,
    # so the charge factor is unity throughout.
    cell = np.asarray(ewald_data.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    E_real_pairs = 0.0
    for i in range(N_e):
        for j in range(i + 1, N_e):
            r_ij = r_all[j] - r_all[i]
            # Independent NumPy fold; do not reuse the production MIC helper.
            fractional = r_ij @ inv_cell
            r_ij = (fractional - np.rint(fractional)) @ cell
            for T in real_images:
                d = float(np.linalg.norm(r_ij + T))
                if d > 0.0:
                    E_real_pairs += float(np_erfc(eta * d)) / d
    E_real_self = 0.0
    for i in range(N_e):
        for T in real_images:
            d = float(np.linalg.norm(T))
            if d > 0.0:
                E_real_self += 0.5 * float(np_erfc(eta * d)) / d

    # Reciprocal-space contribution (structure-factor form). Self-contained
    # numpy broadcast; no production helpers, no JAX kernel.
    phases = g_vectors @ r_all.T  # (N_g, N_e)
    S_real_vec = np.cos(phases).sum(axis=-1)
    S_imag_vec = np.sin(phases).sum(axis=-1)
    E_recip = 0.5 * float(np.sum(g_factors * (S_real_vec**2 + S_imag_vec**2)))

    # Self-energy: per electron contribution -(eta / sqrt(pi)) summed over
    # N_e (q_e^2 = 1 for every electron).
    E_self = 0.0
    for _ in range(N_e):
        E_self += -(eta / sqrt_pi)

    # Uniform-background (jellium) term for the charged electron gas
    # (sum_i q_i = -N_e, so (sum q)^2 = N_e^2).
    volume = float(np.abs(np.linalg.det(np.asarray(ewald_data.structure_data.cell, dtype=np.float64))))
    E_background = -(np.pi / (2.0 * eta * eta * volume)) * float(N_e) * float(N_e)

    return float(E_real_pairs + E_real_self + E_recip + E_self + E_background)


@jit
def compute_ewald_potential_ei(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    ewald_data: Ewald_data,
) -> tuple[jax.Array, jax.Array]:
    r"""Per-electron Ewald electron-ion Coulomb potential under tin-foil BC.

    Returns per-electron contribution arrays ``((N_up,), (N_dn,))`` for
    electrons interacting with ions. Detailed Ewald channel decomposition and
    background-term derivation are work in progress.

    **ECP path**: ``coulomb_potential_data._effective_charges`` returns
    :math:`Z_\text{eff} = Z - z_\text{core}` for ECP atoms and bare
    :math:`Z` for all-electron atoms. Short-range non-local projectors and
    local-channel deviations from :math:`-Z_\text{eff}/r` are handled by
    :py:func:`compute_ecp_coulomb_potential`; this function supplies only the
    long-range Coulomb piece in the e-I path.

    fp64 throughout (``ewald`` zone).

    Args:
        coulomb_potential_data: Provides ``_effective_charges`` and
            ``structure_data._positions_cart_jnp``.
        r_up_carts: Up-spin electron positions, shape ``(N_up, 3)``, fp64.
        r_dn_carts: Down-spin electron positions, shape ``(N_dn, 3)``, fp64.
        ewald_data: Precomputed Ewald tables for the periodic system.

    Returns:
        tuple[jax.Array, jax.Array]: Per-electron contributions
        ``(E_ei_up, E_ei_dn)`` with shapes ``(N_up,)`` and ``(N_dn,)``,
        both ``ewald`` zone fp64.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    # In-trace rebuild from integer indices + cell
    # so the energy is differentiable through cell.
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    real_images, g_vectors, g_factors, _ = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )
    Z_I = jnp.asarray(coulomb_potential_data._effective_charges, dtype=dtype_jnp)
    R_I = coulomb_potential_data.structure_data._positions_cart_jnp.astype(dtype_jnp)

    # Uniform-background (jellium) cross term for the electron-ion channel.
    # The whole-system background -(pi / (2 eta^2 V)) (sum_all q)^2 has the
    # cross contribution -(pi / (eta^2 V)) (sum_I Z_I)(sum_i q_i); with
    # sum_i q_i = -N_e this is +(pi / (eta^2 V)) Q_ion N_e, i.e. a per-electron
    # constant +(pi / (eta^2 V)) Q_ion. Together with the ee and ii background
    # terms it cancels for an overall-neutral cell, leaving the total unchanged.
    volume = jnp.abs(_det_3x3(cell))
    ei_background_per_e = (jnp.asarray(jnp.pi, dtype=dtype_jnp) / (eta * eta * volume)) * jnp.sum(Z_I)

    # Ion structure-factor halves (cached: same for both spin channels).
    GR_ion = g_vectors @ R_I.T  # (N_g, N_atom)
    SI_real = (Z_I[None, :] * jnp.cos(GR_ion)).sum(axis=-1)  # (N_g,)
    SI_imag = (Z_I[None, :] * jnp.sin(GR_ion)).sum(axis=-1)

    # Min-image helper for r_iI: see the parallel comment in
    # ``compute_ewald_potential_ee`` for the rationale (truncated real_images
    # would otherwise drop the dominant erfc contribution when an electron
    # is in a different periodic copy than its nucleus).
    inv_cell = _inv_3x3(cell)
    n_max = ewald_data.structure_data._n_max

    def _per_spin(r_carts: jax.Array) -> jax.Array:
        r_carts = r_carts.astype(dtype_jnp)
        # Real-space: per-electron sum_I Z_I sum_n erfc(eta r) / r with prefactor -1
        # (electron-ion charge product q_e * Z_I = -Z_I).
        r_iI = r_carts[:, None, :] - R_I[None, :, :]  # (N_e, N_atom, 3)
        r_iI = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_iI)
        v_iIn = r_iI[:, :, None, :] + real_images[None, None, :, :]  # (N_e, N_atom, N_n, 3)
        # Guard against accidental r=0 (electron exactly on a nucleus image).
        # Physically r > 0 always (ECP electron never sits on a nucleus;
        # AE Kato cusp is in the Jastrow factor) but the guard keeps
        # the JIT path NaN-free under AD. Guard at the SQUARED-distance level:
        # jnp.sum(v**2) has a finite gradient (2v) at v=0, whereas
        # jnp.linalg.norm(v) has the 0/0 = NaN VJP that leaks through the
        # outer jnp.where (the classic double-where gotcha).
        v_sq_iIn = jnp.sum(v_iIn**2, axis=-1)  # (N_e, N_atom, N_n)
        safe_v_sq = jnp.where(v_sq_iIn > 0.0, v_sq_iIn, 1.0)
        d_iIn = jnp.sqrt(safe_v_sq)  # (N_e, N_atom, N_n)
        contrib_real = jnp_erfc(eta * d_iIn) / d_iIn
        contrib_real = jnp.where(v_sq_iIn > 0.0, contrib_real, 0.0)
        E_real_per_e = -jnp.sum(Z_I[None, :, None] * contrib_real, axis=(-2, -1))  # (N_e,)

        # Reciprocal-space: cos(G . (r_i - R_I)) = cos(G.r_i) cos(G.R_I)
        #                                          + sin(G.r_i) sin(G.R_I)
        # so sum_I Z_I cos(G . (r_i - R_I)) = cos(G.r_i) * SI_real + sin(G.r_i) * SI_imag.
        Gr_e = g_vectors @ r_carts.T  # (N_g, N_e)
        cos_Gr = jnp.cos(Gr_e)
        sin_Gr = jnp.sin(Gr_e)
        per_g_per_e = cos_Gr * SI_real[:, None] + sin_Gr * SI_imag[:, None]  # (N_g, N_e)
        E_recip_per_e = -jnp.sum(g_factors[:, None] * per_g_per_e, axis=0)  # (N_e,)
        return E_real_per_e + E_recip_per_e + ei_background_per_e

    return _per_spin(r_up_carts), _per_spin(r_dn_carts)


def _compute_ewald_potential_ei_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    ewald_data: Ewald_data,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Self-contained reference for :py:func:`compute_ewald_potential_ei`.

    Re-implements the Ewald e-I per-electron formula using only
    ``ewald_data`` tables, ``coulomb_potential_data._effective_charges``,
    and the input electron coordinates. No helper-function call
    (``_enumerate_*``, ``_compute_ewald_*``, ``build_ewald_data``) ---
    bit-identical agreement against the production
    :py:func:`compute_ewald_potential_ei` validates the entire helper
    chain in one cross-check.

    Real-space part uses explicit Python loops (small ``N_e`` * ``N_atom``
    * ``N_n``); reciprocal-space uses numpy broadcasting (large ``N_g``
    counts dominate setup time on small cells; broadcast keeps the
    runtime tractable without invoking any JAX kernel or production helper).
    """
    Z_I = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    R_I = np.asarray(
        coulomb_potential_data.structure_data._positions_cart_np,
        dtype=np.float64,
    )
    eta = float(ewald_data.eta)
    real_images = np.asarray(ewald_data.real_image_list, dtype=np.float64)
    g_vectors = np.asarray(ewald_data.g_vectors, dtype=np.float64)
    g_factors = np.asarray(ewald_data.g_factors, dtype=np.float64)
    N_atom = R_I.shape[0]
    cell = np.asarray(ewald_data.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)

    # Uniform-background (jellium) cross term, per electron (see
    # compute_ewald_potential_ei): +(pi / (eta^2 V)) * sum_I Z_I.
    volume = float(np.abs(np.linalg.det(np.asarray(coulomb_potential_data.structure_data.cell, dtype=np.float64))))
    ei_background_per_e = (np.pi / (eta * eta * volume)) * float(np.sum(Z_I))

    # Ion structure-factor halves (consumed twice; cache once).
    GR_ion = g_vectors @ R_I.T  # (N_g, N_atom)
    SI_real = (Z_I[None, :] * np.cos(GR_ion)).sum(axis=-1)  # (N_g,)
    SI_imag = (Z_I[None, :] * np.sin(GR_ion)).sum(axis=-1)

    def _per_spin(r_carts: np.ndarray) -> np.ndarray:
        r_carts = np.asarray(r_carts, dtype=np.float64)
        N_e = r_carts.shape[0]
        out = np.zeros(N_e, dtype=np.float64)
        if N_e == 0:
            return out
        # Real-space: explicit triple loop (i, I, T) summing the electron-ion
        # pair contribution with the q_e * Z_I = -Z_I sign convention.
        for i in range(N_e):
            for I_idx in range(N_atom):
                # Independent centred-cell fold keeps the bounded table complete.
                fractional = (r_carts[i] - R_I[I_idx]) @ inv_cell
                displacement = (fractional - np.rint(fractional)) @ cell
                for T in real_images:
                    d = float(np.linalg.norm(displacement - T))
                    if d > 0.0:
                        out[i] += -Z_I[I_idx] * float(np_erfc(eta * d)) / d
        # Reciprocal-space (numpy broadcast; self-contained).
        Gr_e = g_vectors @ r_carts.T  # (N_g, N_e)
        cos_Gr = np.cos(Gr_e)
        sin_Gr = np.sin(Gr_e)
        per_g_per_e = cos_Gr * SI_real[:, None] + sin_Gr * SI_imag[:, None]  # (N_g, N_e)
        out += -np.sum(g_factors[:, None] * per_g_per_e, axis=0)
        out += ei_background_per_e
        return out

    return _per_spin(r_up_carts), _per_spin(r_dn_carts)


@jit
def compute_ewald_potential_ei_discretized(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    ewald_data: Ewald_data,
    alat: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Per-electron Ewald electron-ion Coulomb with LRDMC ``alat`` distance floor.

    Periodic analogue of
    :py:func:`compute_discretized_bare_coulomb_potential_el_ion_element_wise`.
    Applies ``r -> max(r, alat)`` to the real-space short-range e-I
    contribution; the reciprocal-space contribution passes through unchanged.
    Detailed Ewald numerical derivation is work in progress.

    fp64 throughout (``ewald`` zone).

    Args:
        coulomb_potential_data: Provides ``_effective_charges`` and
            ``structure_data._positions_cart_jnp``.
        r_up_carts: Up-spin electron positions, shape ``(N_up, 3)``, fp64.
        r_dn_carts: Down-spin electron positions, shape ``(N_dn, 3)``, fp64.
        ewald_data: Precomputed Ewald tables for the periodic system.
        alat: LRDMC lattice spacing; the real-space short-range
            contribution applies the floor ``r -> max(r, alat)``.

    Returns:
        tuple[jax.Array, jax.Array]: Per-electron contributions
        ``(E_ei_up, E_ei_dn)`` with shapes ``(N_up,)`` and ``(N_dn,)``,
        both ``ewald`` zone fp64.
    """
    dtype_jnp = get_dtype_jnp("ewald")
    eta = jnp.asarray(ewald_data.eta, dtype=dtype_jnp)
    # In-trace rebuild from integer indices + cell
    # so the energy is differentiable through cell.
    cell = ewald_data.structure_data._cell_jnp.astype(dtype_jnp)
    real_images, g_vectors, g_factors, _ = _ewald_tables_from_cell(
        eta,
        ewald_data._real_image_indices_jnp,
        ewald_data._g_indices_jnp,
        cell,
    )
    Z_I = jnp.asarray(coulomb_potential_data._effective_charges, dtype=dtype_jnp)
    R_I = coulomb_potential_data.structure_data._positions_cart_jnp.astype(dtype_jnp)
    alat_j = jnp.asarray(alat, dtype=dtype_jnp)

    # Uniform-background (jellium) cross term, per electron (see
    # compute_ewald_potential_ei). The reciprocal/background pieces are
    # alat-independent, so the discretised variant carries the same term and
    # still reduces to compute_ewald_potential_ei at alat = 0.
    volume = jnp.abs(_det_3x3(cell))
    ei_background_per_e = (jnp.asarray(jnp.pi, dtype=dtype_jnp) / (eta * eta * volume)) * jnp.sum(Z_I)

    # Ion structure-factor halves (cached: same for both spin channels).
    GR_ion = g_vectors @ R_I.T  # (N_g, N_atom)
    SI_real = (Z_I[None, :] * jnp.cos(GR_ion)).sum(axis=-1)  # (N_g,)
    SI_imag = (Z_I[None, :] * jnp.sin(GR_ion)).sum(axis=-1)

    # Min-image helper for r_iI (see compute_ewald_potential_ee/_ei rationale).
    inv_cell = _inv_3x3(cell)
    n_max = ewald_data.structure_data._n_max

    def _per_spin(r_carts: jax.Array) -> jax.Array:
        r_carts = r_carts.astype(dtype_jnp)
        # Real-space: per-electron sum_I Z_I sum_n erfc(eta * safe_r) / safe_r
        # with safe_r = max(|r_i - R_I - T_n|, alat). The alat floor is the
        # LRDMC regulariser; identical structure to the molecular
        # `_discretized` variant, just applied inside the image sum.
        #
        # AD safety (matching the unregularised `compute_ewald_potential_ei`
        # pattern): build the distance from ``sqrt(sum v^2 + EPS_safe_distance)``
        # so the VJP through the square-root never sees the ``0/0`` divergence
        # ``jnp.linalg.norm(v)`` produces at ``v == 0``. The ``EPS_safe_distance``
        # offset perturbs the forward value by ``< 1e-6`` Bohr at distances
        # ``> 1`` Bohr, negligible against the ``alat`` floor (and dominated by
        # the LRDMC discretisation error anyway). The outer ``jnp.maximum``
        # then provides the LRDMC regularisation on top of the AD-clean
        # ``d_iIn``.
        r_iI = r_carts[:, None, :] - R_I[None, :, :]  # (N_e, N_atom, 3)
        r_iI = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(r_iI)
        v_iIn = r_iI[:, :, None, :] + real_images[None, None, :, :]  # (N_e, N_atom, N_n, 3)
        v_sq_iIn = jnp.sum(v_iIn**2, axis=-1)  # (N_e, N_atom, N_n)
        d_iIn = jnp.sqrt(v_sq_iIn + EPS_safe_distance)
        safe_d = jnp.maximum(d_iIn, alat_j)
        contrib_real = jnp_erfc(eta * safe_d) / safe_d
        E_real_per_e = -jnp.sum(Z_I[None, :, None] * contrib_real, axis=(-2, -1))  # (N_e,)

        # Reciprocal-space: unchanged from the unregularised variant.
        # cos(G . (r_i - R_I)) = cos(G.r_i) cos(G.R_I) + sin(G.r_i) sin(G.R_I)
        # so sum_I Z_I cos(G . (r_i - R_I)) = cos(G.r_i) SI_real + sin(G.r_i) SI_imag.
        Gr_e = g_vectors @ r_carts.T  # (N_g, N_e)
        cos_Gr = jnp.cos(Gr_e)
        sin_Gr = jnp.sin(Gr_e)
        per_g_per_e = cos_Gr * SI_real[:, None] + sin_Gr * SI_imag[:, None]  # (N_g, N_e)
        E_recip_per_e = -jnp.sum(g_factors[:, None] * per_g_per_e, axis=0)  # (N_e,)
        return E_real_per_e + E_recip_per_e + ei_background_per_e

    return _per_spin(r_up_carts), _per_spin(r_dn_carts)


def _compute_ewald_potential_ei_discretized_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    ewald_data: Ewald_data,
    alat: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Self-contained reference for :py:func:`compute_ewald_potential_ei_discretized`.

    Re-implements the Ewald e-I per-electron formula with the LRDMC
    ``alat`` floor applied to the real-space short-range contribution.
    No helper-function call (``_enumerate_*``, ``_compute_ewald_*``,
    ``build_ewald_data``) --- bit-identical agreement against the
    production :py:func:`compute_ewald_potential_ei_discretized`
    validates the entire helper chain in one cross-check.

    Real-space part uses explicit Python loops (small ``N_e`` * ``N_atom``
    * ``N_n``); reciprocal-space uses numpy broadcasting.
    """
    Z_I = np.asarray(coulomb_potential_data._effective_charges, dtype=np.float64)
    R_I = np.asarray(
        coulomb_potential_data.structure_data._positions_cart_np,
        dtype=np.float64,
    )
    eta = float(ewald_data.eta)
    real_images = np.asarray(ewald_data.real_image_list, dtype=np.float64)
    g_vectors = np.asarray(ewald_data.g_vectors, dtype=np.float64)
    g_factors = np.asarray(ewald_data.g_factors, dtype=np.float64)
    alat_f = float(alat)
    N_atom = R_I.shape[0]
    cell = np.asarray(ewald_data.structure_data.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)

    # Uniform-background (jellium) cross term, per electron (see
    # compute_ewald_potential_ei): +(pi / (eta^2 V)) * sum_I Z_I.
    volume = float(np.abs(np.linalg.det(np.asarray(coulomb_potential_data.structure_data.cell, dtype=np.float64))))
    ei_background_per_e = (np.pi / (eta * eta * volume)) * float(np.sum(Z_I))

    # Ion structure-factor halves (consumed twice; cache once).
    GR_ion = g_vectors @ R_I.T  # (N_g, N_atom)
    SI_real = (Z_I[None, :] * np.cos(GR_ion)).sum(axis=-1)  # (N_g,)
    SI_imag = (Z_I[None, :] * np.sin(GR_ion)).sum(axis=-1)

    def _per_spin(r_carts: np.ndarray) -> np.ndarray:
        r_carts = np.asarray(r_carts, dtype=np.float64)
        N_e = r_carts.shape[0]
        out = np.zeros(N_e, dtype=np.float64)
        if N_e == 0:
            return out
        # Real-space with alat floor.
        for i in range(N_e):
            for I_idx in range(N_atom):
                # Independent centred-cell fold keeps the bounded table complete.
                fractional = (r_carts[i] - R_I[I_idx]) @ inv_cell
                displacement = (fractional - np.rint(fractional)) @ cell
                for T in real_images:
                    d = float(np.linalg.norm(displacement - T))
                    safe_d = max(d, alat_f)
                    out[i] += -Z_I[I_idx] * float(np_erfc(eta * safe_d)) / safe_d
        # Reciprocal-space (unchanged).
        Gr_e = g_vectors @ r_carts.T  # (N_g, N_e)
        cos_Gr = np.cos(Gr_e)
        sin_Gr = np.sin(Gr_e)
        per_g_per_e = cos_Gr * SI_real[:, None] + sin_Gr * SI_imag[:, None]  # (N_g, N_e)
        out += -np.sum(g_factors[:, None] * per_g_per_e, axis=0)
        out += ei_background_per_e
        return out

    return _per_spin(r_up_carts), _per_spin(r_dn_carts)


# =============================================================================
# Three-prong (+ 1 raise-only) Coulomb dispatch shims
# =============================================================================
# Single chokepoint that routes ee / ei / ii calls to either the bare 1/r
# implementation (molecular) or the Ewald lattice-sum implementation (PBC).
# The branch is decided by ``coulomb_potential_data.structure_data.pbc_flag``,
# which is ``pytree_node=False`` and therefore resolved at JIT trace time.
# Molecular and periodic instances JIT-cache separately; the emitted HLO for
# a molecular ``Coulomb_potential_data`` is byte-identical to a direct call
# to the corresponding bare function.
#
# Public-API invariance under PBC requires that every
# call site routes through these shims; the bare
# functions stay available as ``_debug`` oracles and for the existing
# molecular regression tests.


@jit
def compute_coulomb_potential_ee_dispatch(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    """Dispatch the electron-electron Coulomb potential by ``pbc_flag``.

    Periodic (``pbc_flag=True``): forward to
    :py:func:`compute_ewald_potential_ee` using
    ``coulomb_potential_data.ewald_data`` (populated at setup time by
    :py:func:`build_ewald_data`).

    Molecular (``pbc_flag=False``): forward to
    :py:func:`compute_bare_coulomb_potential_el_el`. The first argument
    (``coulomb_potential_data``) is unused on the molecular path but is
    kept in the shim signature so all four dispatch shims share a uniform
    leading-argument convention.
    """
    if coulomb_potential_data.structure_data.pbc_flag:
        return compute_ewald_potential_ee(
            r_up_carts,
            r_dn_carts,
            coulomb_potential_data.ewald_data,
        )
    return compute_bare_coulomb_potential_el_el(r_up_carts, r_dn_carts)


@jit
def compute_coulomb_potential_ei_dispatch(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch the per-electron electron-ion Coulomb potential by ``pbc_flag``.

    Periodic: forward to :py:func:`compute_ewald_potential_ei`. The ECP
    Z_eff path is handled inside that function via
    ``coulomb_potential_data._effective_charges``.

    Molecular: forward to
    :py:func:`compute_bare_coulomb_potential_el_ion_element_wise`. Returns
    per-electron arrays ``((N_up,), (N_dn,))`` in both branches.
    """
    if coulomb_potential_data.structure_data.pbc_flag:
        return compute_ewald_potential_ei(
            coulomb_potential_data,
            r_up_carts,
            r_dn_carts,
            coulomb_potential_data.ewald_data,
        )
    return compute_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data,
        r_up_carts,
        r_dn_carts,
    )


@jit
def compute_coulomb_potential_ii_dispatch(
    coulomb_potential_data: Coulomb_potential_data,
) -> jax.Array:
    """Dispatch the ion-ion Coulomb potential by ``pbc_flag``.

    Periodic: forward to :py:func:`compute_ewald_potential_ii`, which
    evaluates the ion-ion energy in-trace from the integer index sets and
    the current ``ewald_data.structure_data`` (not the cached
    ``ewald_data.ii_energy``).

    Molecular: forward to :py:func:`compute_bare_coulomb_potential_ion_ion`.
    """
    if coulomb_potential_data.structure_data.pbc_flag:
        return compute_ewald_potential_ii(
            coulomb_potential_data,
            coulomb_potential_data.ewald_data,
        )
    return compute_bare_coulomb_potential_ion_ion(coulomb_potential_data)


@jit
def compute_coulomb_potential_ei_discretized_dispatch(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    alat: float,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch the LRDMC-discretised electron-ion Coulomb potential.

    Periodic (``pbc_flag=True``): forward to
    :py:func:`compute_ewald_potential_ei_discretized` --- the Ewald
    per-electron e-I energy with the LRDMC ``alat`` distance floor
    applied to the real-space short-range contribution. The reciprocal
    contribution is bounded and passes through unchanged.

    Molecular (``pbc_flag=False``): forward unchanged to
    :py:func:`compute_discretized_bare_coulomb_potential_el_ion_element_wise`.
    """
    if coulomb_potential_data.structure_data.pbc_flag:
        return compute_ewald_potential_ei_discretized(
            coulomb_potential_data,
            r_up_carts,
            r_dn_carts,
            coulomb_potential_data.ewald_data,
            alat,
        )
    return compute_discretized_bare_coulomb_potential_el_ion_element_wise(
        coulomb_potential_data,
        r_up_carts,
        r_dn_carts,
        alat,
    )


def _compute_coulomb_potential_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: npt.NDArray[np.float64],
    r_dn_carts: npt.NDArray[np.float64],
    RT: npt.NDArray = np.eye(3),
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """Debug reference for the total Coulomb energy (bare + ECP).

    Straightforward (slow) implementation that adds the bare Coulomb debug value and, when
    ``ecp_flag`` is set, the ECP debug value; just for debugging purposes.
    """
    # Forward r_up/dn_carts and RT as-is (Principle 3a -- no parameter rebind).
    # Each downstream debug function casts to its own zone at the use site;
    # the accumulated scalar is cast to the coulomb zone before return.
    dtype_np = get_dtype_np("coulomb")

    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = _compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = _compute_bare_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
        )

        ecp_coulomb_potential = _compute_ecp_coulomb_potential_debug(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            RT=RT,
            Nv=Nv,
            NN=NN,
        )

    return np.asarray(bare_coulomb_potential + ecp_coulomb_potential, dtype=dtype_np)


def compute_coulomb_potential(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data = None,
) -> float:
    """Compute total Coulomb energy including bare and ECP terms.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure, charges, and ECP parameters.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``) for non-local ECP.
        NN (int): Number of nearest nuclei to include for each electron in the non-local term.
        Nv (int): Number of quadrature points on the sphere.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ECP ratios; required when ``ecp_flag`` is True.

    Returns:
        float: Sum of bare Coulomb (ion-ion, electron-ion, electron-electron) and ECP (local + non-local) energies.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts/RT (forwarded to downstream
    # functions that handle their own use-site casts -- Principle 3a).

    # all-electron
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        ecp_coulomb_potential = 0

    # pseudo-potential
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        ecp_coulomb_potential = compute_ecp_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            RT=RT,
            NN=NN,
            Nv=Nv,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


def compute_coulomb_potential_fast(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> jax.Array:
    """Compute total Coulomb energy using a pre-computed geminal inverse for ECP non-local terms.

    This is the fast variant of :func:`compute_coulomb_potential`.  For ECP systems the
    non-local Psi(r')/Psi(r) ratios are evaluated via
    :func:`compute_ecp_coulomb_potential_fast`, which accepts ``A_old_inv`` directly
    and therefore avoids the fresh LU factorisation performed by the standard path.
    This prevents NaN when the current-configuration geminal matrix is nearly singular.

    Args:
        coulomb_potential_data (Coulomb_potential_data): Structure, charges, and ECP parameters.
        r_up_carts (jax.Array): Up-spin electron Cartesian coordinates with shape ``(N_up, 3)`` and ``float64`` dtype.
        r_dn_carts (jax.Array): Down-spin electron Cartesian coordinates with shape ``(N_dn, 3)`` and ``float64`` dtype.
        RT (jax.Array): Rotation matrix applied to quadrature grid points (shape ``(3, 3)``) for non-local ECP.
        A_old_inv (jax.Array): Pre-computed inverse of the reference geminal matrix.
        NN (int): Number of nearest nuclei to include for each electron in the non-local term.
        Nv (int): Number of quadrature points on the sphere.
        wavefunction_data (Wavefunction_data): Wavefunction (geminal + Jastrow) used for ECP ratios; required when ``ecp_flag`` is True.
        det_ratio_state: Optional :class:`Det_ratio_streaming_state` (or
            superset) consistent with ``(r_up_carts, r_dn_carts)``.
            Forwarded to :func:`compute_ecp_coulomb_potential_fast` so the
            non-local ECP ratio kernel can skip the bulk-side AO eval and
            ``lambda @ ao_*`` precontracts.

    Returns:
        float: Sum of bare Coulomb (ion-ion, electron-ion, electron-electron) and ECP (local + non-local) energies.

    Warning:
        ``A_old_inv`` **must** equal ``G(r_up_carts, r_dn_carts)^{-1}`` exactly at the
        supplied electron positions.  Correctness is only guaranteed when the inverse is
        maintained via **single-electron (rank-1) Sherman-Morrison updates** starting from
        a freshly initialized LU inverse -- the pattern used in the MCMC loop.  If multiple
        electrons have moved simultaneously the underlying Sherman-Morrison rank-1 update is
        incorrect and non-local ratios will be silently wrong.
    """
    # NOTE: Do NOT pre-cast r_up_carts/r_dn_carts/RT (forwarded to downstream
    # functions that handle their own use-site casts -- Principle 3a).

    # all-electron -- no ECP, no need for A_old_inv
    if not coulomb_potential_data.ecp_flag:
        bare_coulomb_potential = compute_bare_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )
        ecp_coulomb_potential = 0

    # pseudo-potential -- use pre-computed inverse to avoid fresh LU
    else:
        bare_coulomb_potential = compute_bare_coulomb_potential(
            coulomb_potential_data=coulomb_potential_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts
        )

        ecp_coulomb_potential = compute_ecp_coulomb_potential_fast(
            coulomb_potential_data=coulomb_potential_data,
            r_up_carts=r_up_carts,
            r_dn_carts=r_dn_carts,
            wavefunction_data=wavefunction_data,
            RT=RT,
            A_old_inv=A_old_inv,
            NN=NN,
            Nv=Nv,
            det_ratio_state=det_ratio_state,
        )

    return bare_coulomb_potential + ecp_coulomb_potential


def _compute_coulomb_potential_fast_debug(
    coulomb_potential_data: Coulomb_potential_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
    RT: jax.Array,
    A_old_inv: jax.Array,
    NN: int = NN_default,
    Nv: int = Nv_default,
    wavefunction_data: Wavefunction_data | None = None,
    det_ratio_state: Det_ratio_streaming_state | Det_streaming_state | None = None,
) -> jax.Array:
    """Debug oracle for :func:`compute_coulomb_potential_fast`.

    Discards ``A_old_inv`` / ``det_ratio_state`` and delegates to the non-fast
    sibling :func:`compute_coulomb_potential` (fresh LU per ECP non-local
    evaluation). Covers both the bare-only and ECP branches; the guardrail test
    asserts the fast path reproduces it when ``A_old_inv`` is exact.
    """
    return compute_coulomb_potential(
        coulomb_potential_data=coulomb_potential_data,
        r_up_carts=r_up_carts,
        r_dn_carts=r_dn_carts,
        RT=RT,
        NN=NN,
        Nv=Nv,
        wavefunction_data=wavefunction_data,
    )
