"""TREXIO wrapper modules.

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

# import python modules
# logger
import itertools
from logging import getLogger

import numpy as np
import numpy.typing as npt
import scipy
import trexio

from .atomic_orbital import (
    AO_radial_shell_data,
    AOs_cart_gto_data,
    AOs_cart_pgto_complex_data,
    AOs_cart_pgto_data,
    AOs_sphe_gto_data,
    AOs_sphe_pgto_complex_data,
    AOs_sphe_pgto_data,
    build_AOs_cart_pgto_complex_data,
    build_AOs_sphe_pgto_complex_data,
)
from .coulomb_potential import Coulomb_potential_data, build_ewald_data
from .determinant import SD_data
from .molecular_orbital import MOs_data

from .structure import Structure_data

logger = getLogger("jqmc").getChild(__name__)

#: QMC fixes the electron numbers, so a periodic TREXIO block is accepted only
#: when its occupations are integers that sum to the file's electron numbers.
#: Same contract as ``qe_to_jqmc.OCC_BINARY_TOL`` on the QE route.
MO_OCCUPATION_INTEGER_TOL = 1.0e-6


def _validate_trexio_kpoint_mo_mapping(file_r, n_kpts: int) -> np.ndarray | None:
    """Validate the MO-to-k-point map required by a multi-k TREXIO source.

    ``pbc_k_point`` identifies the Bloch vectors, while ``mo_k_point`` binds
    every MO row to one of those vectors. A multi-k file without that binding
    is ambiguous because selecting ``twist_ik`` would otherwise reuse the
    unfiltered global MO matrix for every k point.
    """
    has_mapping = trexio.has_mo_k_point(file_r)
    if n_kpts > 1 and not has_mapping:
        raise ValueError("multi-k TREXIO requires mo_k_point to associate every MO row with one pbc_k_point entry.")
    if not has_mapping:
        return None
    mo_num = int(trexio.read_mo_num(file_r))
    try:
        mapping = np.asarray(trexio.read_mo_k_point(file_r))
    except trexio.Error as err:
        raise ValueError(f"TREXIO mo_k_point cannot be read consistently with mo_num={mo_num}.") from err
    if mapping.shape != (mo_num,):
        raise ValueError(f"TREXIO mo_k_point shape {mapping.shape} does not match mo_num={mo_num}.")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("TREXIO mo_k_point entries must be integer k-point indices.")
    mapping = mapping.astype(np.int64, copy=False)
    invalid = (mapping < 0) | (mapping >= n_kpts)
    if np.any(invalid):
        bad = int(mapping[np.flatnonzero(invalid)[0]])
        raise ValueError(f"TREXIO mo_k_point index {bad} is outside pbc_k_point range [0, {n_kpts}).")
    missing = np.flatnonzero(np.bincount(mapping, minlength=n_kpts) == 0)
    if missing.size:
        raise ValueError(f"TREXIO pbc_k_point entries {missing.tolist()} have no associated MO rows.")
    return mapping


def _check_periodic_occupations(
    mo_occupation: npt.NDArray[np.float64],
    mo_spin: npt.NDArray[np.int64],
    num_ele_up: int,
    num_ele_dn: int,
    twist_ik: int,
) -> None:
    """Raise ``ValueError`` unless one k-block holds the fixed electron numbers.

    Selecting the first occupied rows of a smeared block, or of a metallic
    k-point that holds more electrons than the file's per-cell count, would
    build a determinant that does not correspond to the mean-field state.

    Args:
        mo_occupation: Occupations of the MO rows of one k-point.
        mo_spin: ``mo_spin`` labels of the same rows.
        num_ele_up: Number of up-spin electrons.
        num_ele_dn: Number of down-spin electrons.
        twist_ik: Zero-based k index, used in the error message.

    Raises:
        ValueError: If an occupation is non-finite or out of range, the
            block does not sum to the electron numbers, or restricted
            occupations give other spin populations.
    """
    remedy = (
        "QMC fixes the electron numbers at every twist (no grand-canonical TABC yet), "
        "so smeared or metallic mean-field states are not accepted."
    )
    occ = np.asarray(mo_occupation, dtype=np.float64)
    spin = np.asarray(mo_spin)
    # NaN fails every comparison below, so it would hide the electron count.
    if not np.all(np.isfinite(occ)):
        raise ValueError(f"TREXIO k-point {twist_ik}: occupations must be finite. {remedy}")
    spin_dependent = bool(np.any(spin != 0))
    if spin_dependent:
        channels = (("up", occ[spin == 0], num_ele_up), ("dn", occ[spin != 0], num_ele_dn))
        max_occupation = 1
    else:
        channels = (("total", occ, num_ele_up + num_ele_dn),)
        max_occupation = 2
    for label, block, expected in channels:
        nearest = np.rint(block)
        if (
            np.any(np.abs(block - nearest) > MO_OCCUPATION_INTEGER_TOL)
            or np.any(nearest < 0)
            or np.any(nearest > max_occupation)
        ):
            raise ValueError(
                f"TREXIO k-point {twist_ik}: {label} occupations must be integers in [0, {max_occupation}]. {remedy}"
            )
        total = float(block.sum())
        if abs(total - expected) > MO_OCCUPATION_INTEGER_TOL:
            raise ValueError(f"TREXIO k-point {twist_ik}: {label} occupations sum to {total:g}, expected {expected}. {remedy}")
    if not spin_dependent:
        # Restricted rows: 2 = up + dn and 1 = up only, as read_trexio_file
        # builds the up and dn determinants. The total alone does not fix
        # the spin populations ([1, 1] and [2, 0] both sum to 2).
        nearest = np.rint(occ)
        n_double = int(np.count_nonzero(nearest == 2))
        n_single = int(np.count_nonzero(nearest == 1))
        if (n_double + n_single, n_double) != (num_ele_up, num_ele_dn):
            raise ValueError(
                f"TREXIO k-point {twist_ik}: restricted occupations hold {n_double} doubly and {n_single} singly "
                f"occupied orbitals ({n_double + n_single} up, {n_double} dn), expected {num_ele_up} up and "
                f"{num_ele_dn} dn. {remedy}"
            )


def read_trexio_file(
    trexio_file: str,
    store_tuple: bool = False,
    twist_ik: int | None = None,
) -> tuple[
    Structure_data,
    "AOs_sphe_gto_data | AOs_cart_gto_data | AOs_sphe_pgto_data | AOs_cart_pgto_data | AOs_sphe_pgto_complex_data | AOs_cart_pgto_complex_data",
    MOs_data,
    MOs_data,
    SD_data,
    Coulomb_potential_data,
]:
    """Load a TREXIO HDF5 file into jqmc data containers.

    Multi-k selection: ``twist_ik`` (0-based) selects one twist from a
    multi-k TREXIO save (``pbc_k_point_num > 1``). The MO rows are
    filtered to those carrying ``mo_k_point == twist_ik`` and the
    corresponding fractional ``k_point`` is stored on the determinant AO leaf
    that owns Bloch phase information (and selects its complex Bloch-pGTO
    representation). ``twist_ik = None`` (default)
    keeps the single-twist behaviour: the first ``pbc_k_point`` row is
    used and the MO matrix is consumed as-is.

    The returned SD is canonical at this boundary: occupation data selects
    exactly ``num_electron_up`` and ``num_electron_dn`` rows. Virtual TREXIO
    bands are never exposed through :class:`SD_data`, including in the
    multi-twist path.

    Args:
        trexio_file (str): Path to the TREXIO file to read (HDF5 backend expected).
        store_tuple (bool, optional): Store list-like fields as tuples for immutability
            (useful in tests with JAX/Flax), at the cost of slower production runs.
            Defaults to ``False``.
        twist_ik (int or None, optional): Zero-based k-point row to select from
            a multi-k TREXIO file. ``None`` selects the first row and preserves
            the ordinary single-twist behavior.

    Returns:
        tuple: ``(structure_data, aos_data, mos_data_up, mos_data_dn, geminal_data, coulomb_potential_data)``
        where:

        - ``structure_data`` is a `Structure_data` describing atoms and geometry.
        - ``aos_data`` depends on the basis angular form (Cartesian or
          spherical) and the periodicity: `AOs_cart_gto_data` /
          `AOs_sphe_gto_data` for a molecule; `AOs_cart_pgto_data` /
          `AOs_sphe_pgto_data` for a single-k Gamma periodic file;
          `AOs_cart_pgto_complex_data` / `AOs_sphe_pgto_complex_data` (the
          ``+k`` Bloch leaf) for a non-Gamma twist or a multi-k file.
        - ``mos_data_up`` and ``mos_data_dn`` are `MOs_data` for spin-up/down
          orbitals; with a complex AO leaf, ``mos_data_dn`` is built on the
          separate ``-k`` leaf.
        - ``geminal_data`` is an `SD_data` built from the occupied MO rows.
        - ``coulomb_potential_data`` is a `Coulomb_potential_data` describing (E)CPs.

    Raises:
        ValueError: If a complex MO is supplied without an unambiguous
            non-Gamma Bloch context, if occupation data cannot fill the SD,
            or if AO/cell metadata are inconsistent.

    Notes:
        - Periodic boundary conditions are supported at Gamma and non-Gamma
          twists; ``cell_a/b/c`` are read into
          ``Structure_data.vec_a/b/c``.
        - Periodic ``aos_data`` is returned as an ``AOs_*_pgto_data`` leaf
          in image-expanded form (``aos_data._n_image >= 1``) --- the
          loader constructs it directly via
          :py:meth:`jqmc.atomic_orbital.AOs_cart_pgto_data.build` or
          :py:meth:`jqmc.atomic_orbital.AOs_sphe_pgto_data.build` (wrapped by
          :py:func:`jqmc.atomic_orbital.build_AOs_cart_pgto_complex_data` /
          :py:func:`jqmc.atomic_orbital.build_AOs_sphe_pgto_complex_data` for a
          complex Bloch leaf) using
          :data:`jqmc._setting.AO_IMAGE_TRUNCATION_ATOL` as the truncation.
          Molecular ``aos_data`` is returned as an ``AOs_*_gto_data`` leaf
          and is never image-expanded.
        - Periodic ``coulomb_potential_data`` is returned with its
          ``ewald_data`` field populated by
          :py:func:`jqmc.coulomb_potential.build_ewald_data` using
          ``tol = 1e-10``. Molecular ``coulomb_potential_data`` keeps
          ``ewald_data = None``. The strict contract
          ``pbc_flag == True iff ewald_data is not None`` is enforced at
          loader exit.
        - Molecular orbitals are real at Gamma and may be complex at a
          non-Gamma twist. The TREXIO file stores the MO coefficients of
          each spin set at its twist ``+k``. The loaded wavefunction uses
          ``+k`` for the up channel and ``-k`` for the down channel: the
          down-spin orbitals are the conjugated coefficients of that
          channel's own ``+k`` set on the separate ``-k`` AO leaf, i.e. the
          complex conjugates of those ``+k`` orbitals. Restricted inputs
          take the alpha set for the down channel; unrestricted inputs keep
          the independent beta set. A geminal whose down channel is not at
          ``-k`` of its up channel fails ``sanity_check``.

    Examples:
        >>> from jqmc import read_trexio_file
        >>> structure_data, aos_data, mos_up, mos_dn, geminal_data, coulomb_data = read_trexio_file("molecule.h5")
        >>> structure_data.atomic_labels[:3]
        ['O', 'H', 'H']
    """
    dtype_np = np.float64

    # read a trexio file
    file_r = trexio.File(
        trexio_file,
        mode="r",
        back_end=trexio.TREXIO_HDF5,
    )

    # check if the system is PBC or not.
    try:
        periodic = trexio.read_pbc_periodic(file_r)
    except trexio.Error:
        periodic = False
    mo_k_point_per_mo = None
    if periodic:
        pbc_flag = True
        cell_a = trexio.read_cell_a(file_r)
        cell_b = trexio.read_cell_b(file_r)
        cell_c = trexio.read_cell_c(file_r)
        # Accept a non-Gamma k_point. Modern TREXIO files always carry
        # ``pbc_k_point``; treat its absence as a malformed fixture. The
        # fractional k_point is threaded into the determinant AO leaf below
        # (``None`` at single-k Gamma so the real-path dispatch stays
        # byte-identical; a 3-tuple otherwise).
        # NOTE on convention: TREXIO's ``pbc_k_point`` is stored in
        # **fractional** units of reciprocal lattice vectors (per the
        # TREXIO schema), which is also jQMC's internal convention. A nonzero
        # loaded value is logged below, so a file written in Cartesian
        # units is visible in the run log.
        try:
            k_point = np.asarray(trexio.read_pbc_k_point(file_r), dtype=dtype_np)
        except trexio.Error as err:
            raise ValueError("Periodic TREXIO file is missing pbc_k_point. Write [0, 0, 0] for the Gamma point.") from err
        # TREXIO's ``pbc_k_point`` natively supports multi-k storage as an
        # ``(n_kpts, 3)`` matrix. The single-twist reader path selects the
        # first k entry; ``read_trexio_kpoint_list`` enumerates all rows
        # for the twist-averaged converter. A legacy 1-D ``(3,)`` array
        # (pre-multi-k pyscf+trexio writers) is treated as a single
        # k-point so existing single-twist fixtures keep loading
        # unchanged.
        if k_point.ndim == 2:
            n_kpts = int(k_point.shape[0])
            if n_kpts == 0:
                raise ValueError("Periodic TREXIO file has pbc_k_point with zero rows; expected at least one k-point.")
            mo_k_point_per_mo = _validate_trexio_kpoint_mo_mapping(file_r, n_kpts)
            # ``twist_ik`` (0-based) selects which row of ``pbc_k_point``
            # to use. The default ``twist_ik = None`` keeps the ``ik=0``
            # behaviour so single-twist callers are byte-identical.
            _selected_ik = 0 if twist_ik is None else int(twist_ik)
            if not (0 <= _selected_ik < n_kpts):
                raise ValueError(f"twist_ik={twist_ik} out of range; pbc_k_point has {n_kpts} rows.")
            if n_kpts > 1 and twist_ik is None:
                logger.warning(
                    "TREXIO file has %d k-points; the single-twist reader uses "
                    "ik=0. Use jqmc._twist_average converters or pass twist_ik=N "
                    "for a specific twist.",
                    n_kpts,
                )
            k_point_vec = k_point[_selected_ik]
            _is_multi_k = n_kpts > 1
        elif k_point.ndim == 1:
            if twist_ik not in (None, 0):
                raise ValueError(
                    f"twist_ik={twist_ik} requested but the TREXIO file stores a "
                    "single k-point (1-D pbc_k_point). Only twist_ik=0 / None is valid."
                )
            k_point_vec = k_point
            _is_multi_k = False
        else:
            raise ValueError(f"pbc_k_point has unexpected shape {k_point.shape}; expected (3,) or (n_kpts, 3).")
        # A multi-k Bloch storage routes **every** twist (including any
        # ``xk == (0, 0, 0)`` row in the mesh) through the complex Bloch
        # path. Collapsing the Gamma slice to ``k_point_frac = None``
        # would pair a real AO leaf with the file's complex128 MO block
        # (``mo_coefficient + 1j * mo_coefficient_im`` with ``Im == 0``
        # at the Gamma row) and break the geminal ``sanity_check``
        # complex-leaf <-> complex-lambda contract. A single-k Gamma
        # file keeps the real-path dispatch.
        if np.allclose(k_point_vec, 0.0) and not _is_multi_k:
            k_point_frac = None
        else:
            k_point_frac = tuple(float(x) for x in k_point_vec)
            if np.allclose(k_point_vec, 0.0):
                logger.info("Multi-k Gamma slice loaded as complex Bloch (k_point fractional = (0, 0, 0)).")
            else:
                logger.info(f"Non-Gamma twist loaded, k_point (fractional) = {k_point_frac}")
    else:
        pbc_flag = False
        cell_a = cell_b = cell_c = None
        k_point_frac = None
    logger.info(f"pbc_flag = {pbc_flag}")

    # read electron num
    num_ele_up = trexio.read_electron_up_num(file_r)
    num_ele_dn = trexio.read_electron_dn_num(file_r)

    if num_ele_up - num_ele_dn != 0:
        spin_polarized = True
    else:
        spin_polarized = False
    logger.info(f"num_ele_up = {num_ele_up}, num_ele_dn = {num_ele_dn}, spin_polarized = {spin_polarized}")

    # read structure info.
    # TREXIO keeps element identity and label as two independent nucleus fields:
    # ``nucleus_charge`` is the full nuclear charge Z (core electrons removed by
    # an ECP are tracked separately in the ecp group, read below), and
    # ``nucleus_label`` is a free-form per-nucleus tag ('O' vs 'O1'). We map
    # charge -> atomic_numbers / element_symbols and nucleus_label ->
    # atomic_labels, so a tagged (e.g. AFM) file round-trips without ever
    # parsing the label string for its element.
    labels_r = trexio.read_nucleus_label(file_r)
    charges_r = trexio.read_nucleus_charge(file_r)
    coords_r = trexio.read_nucleus_coord(file_r)

    # Reading basis sets info
    basis_shell_num = trexio.read_basis_shell_num(file_r)
    basis_shell_index = trexio.read_basis_shell_index(file_r)
    basis_nucleus_index = trexio.read_basis_nucleus_index(file_r)
    basis_shell_ang_mom = trexio.read_basis_shell_ang_mom(file_r)
    basis_shell_factor = trexio.read_basis_shell_factor(file_r)
    basis_shell_index = trexio.read_basis_shell_index(file_r)
    basis_exponent = trexio.read_basis_exponent(file_r)
    basis_coefficient = trexio.read_basis_coefficient(file_r)
    basis_prim_factor = trexio.read_basis_prim_factor(file_r)

    # ao info
    ao_cartesian = trexio.read_ao_cartesian(file_r)
    ao_num = trexio.read_ao_num(file_r)
    ao_shell = (
        np.asarray(trexio.read_ao_shell(file_r)) if ao_cartesian and not pbc_flag and trexio.has_ao_shell(file_r) else None
    )
    if ao_shell is not None and (ao_shell.shape != (ao_num,) or np.any((ao_shell < 0) | (ao_shell >= basis_shell_num))):
        raise ValueError("TREXIO Cartesian AO shell IDs do not match the declared basis.")
    ao_normalization = trexio.read_ao_normalization(file_r)

    # mo info
    mo_coefficient_real = trexio.read_mo_coefficient(file_r)
    mo_occupation = trexio.read_mo_occupation(file_r)

    # mo spin check
    mo_spin = trexio.read_mo_spin(file_r)

    # A multi-k TREXIO file stacks the per-k MO blocks vertically into one
    # ``(n_kpts * num_mo_per_k, num_ao)`` matrix and tags each row with
    # ``mo_k_point[i]`` (0-based k index).
    # When ``twist_ik`` is requested we filter to that twist's
    # rows before any downstream MO-occupation / spin / complex handling
    # so all of those operate on the per-k block. Same for ``mo_occupation``
    # and ``mo_spin`` -- both are per-MO arrays of length ``num_mo_total``.
    # Single-k files (no ``mo_k_point`` dataset) are untouched.
    _twist_ik_selected = 0 if twist_ik is None else int(twist_ik)
    if mo_k_point_per_mo is not None:
        if mo_k_point_per_mo.size != mo_coefficient_real.shape[0]:
            raise ValueError(
                f"TREXIO mo_k_point length {mo_k_point_per_mo.size} != mo_coefficient row count {mo_coefficient_real.shape[0]}."
            )
        _row_mask = mo_k_point_per_mo == _twist_ik_selected
        if not np.any(_row_mask):
            raise ValueError(
                f"TREXIO file has no MOs with mo_k_point == {_twist_ik_selected}; "
                f"available k indices: {sorted(set(int(x) for x in mo_k_point_per_mo))[:10]}..."
            )
        mo_coefficient_real = np.asarray(mo_coefficient_real)[_row_mask]
        mo_occupation = np.asarray(mo_occupation)[_row_mask]
        mo_spin = np.asarray(mo_spin)[_row_mask]
        if twist_ik is not None:
            logger.info(
                "twist_ik=%d -> %d MOs selected from the multi-k MO block.",
                _twist_ik_selected,
                int(np.sum(_row_mask)),
            )

    if all(x == 0 for x in mo_spin):
        spin_dependent = False
    else:
        spin_dependent = True

    logger.info(f"spin_dependent MOs = {spin_dependent}")

    if pbc_flag:
        _check_periodic_occupations(mo_occupation, mo_spin, num_ele_up, num_ele_dn, _twist_ik_selected)

    # MO complex check: consume a complex MO at a non-Gamma twist; leave a
    # real MO as-is at Gamma. The TREXIO file ships the **up-spin** MO set
    # ``C^{+k}``; the down-spin geminal orbital is built later in this
    # loader with ``mo_coefficients = conj(C^{+k})`` on the corresponding
    # ``-k`` AO leaf.
    # NOTE on variable name: ``mo_coefficient_real`` is the long-standing
    # downstream identifier for the loaded MO matrix; its dtype widens to
    # ``complex128`` at non-Gamma but the name is kept so
    # every downstream slice / construction site works unchanged.
    if trexio.has_mo_coefficient_im(file_r):
        # A single-k Gamma file with complex Im is ambiguous (Bloch vs
        # inversion-symmetric spin-orbit; both store the same way). In a
        # multi-k file the Gamma slice is routed through the complex
        # Bloch path above (``k_point_frac == (0, 0, 0)``, not ``None``),
        # so this guard only fires when there is genuinely no Bloch
        # context.
        if k_point_frac is None:
            raise ValueError(
                "TREXIO file has a complex MO (mo_coefficient_im) but k_point is Gamma. "
                "A complex MO with k = (0,0,0) is ambiguous (Bloch and inversion-symmetric "
                "spin-orbit components are stored the same way); convert to a real MO or "
                "set a non-zero k_point in the TREXIO writer."
            )
        mo_im = np.asarray(trexio.read_mo_coefficient_im(file_r), dtype=np.float64)
        # ``mo_im`` is stored in the same layout as ``mo_coefficient``
        # (one row per per-k MO, stacked across k). Apply the same row
        # mask as above before combining.
        if trexio.has_mo_k_point(file_r):
            mo_im = mo_im[_row_mask]
        mo_coefficient_real = (np.asarray(mo_coefficient_real, dtype=np.float64) + 1j * mo_im).astype(np.complex128)
        # Gauge fix: each Bloch band has an arbitrary global phase from
        # the DFT solver. Normalise per band so the first non-negligible
        # coefficient is real-positive
        # (``C[b] *= conj(c_pivot)/|c_pivot|``). One ``O(num_mo * num_ao)``
        # pass before any slicing -- so the gauge-fixed amplitudes flow
        # uniformly into both the up-spin geminal orbital and its
        # conjugate down-spin sibling.
        _GAUGE_PIVOT_TOL = 1e-8
        _abs_C = np.abs(mo_coefficient_real)
        _mask = _abs_C > _GAUGE_PIVOT_TOL
        # First above-threshold column per band; fall back to column 0 if
        # an entire band is zero (degenerate / unoccupied placeholder).
        _pivot_col = np.where(_mask.any(axis=1), _mask.argmax(axis=1), 0)
        _pivot_c = mo_coefficient_real[np.arange(mo_coefficient_real.shape[0]), _pivot_col]
        _pivot_abs = np.abs(_pivot_c)
        _phase = np.where(_pivot_abs > 0.0, np.conj(_pivot_c) / np.where(_pivot_abs > 0.0, _pivot_abs, 1.0), 1.0)
        mo_coefficient_real = mo_coefficient_real * _phase[:, None]
        logger.info(
            f"Complex MO coefficients loaded (down-spin built as conj(C) "
            f"downstream). Gauge-fixed {mo_coefficient_real.shape[0]} bands."
        )
    else:
        mo_coefficient_real = np.asarray(mo_coefficient_real, dtype=np.float64)

    # Pseudo potentials info
    if trexio.has_ecp_num(file_r):
        ecp_flag = True
        ecp_max_ang_mom_plus_1 = trexio.read_ecp_max_ang_mom_plus_1(file_r)
        ecp_z_core = trexio.read_ecp_z_core(file_r)
        ecp_num = trexio.read_ecp_num(file_r)
        ecp_ang_mom = trexio.read_ecp_ang_mom(file_r)
        ecp_nucleus_index = trexio.read_ecp_nucleus_index(file_r)
        ecp_exponent = trexio.read_ecp_exponent(file_r)
        ecp_coefficient = trexio.read_ecp_coefficient(file_r)
        ecp_power = trexio.read_ecp_power(file_r)
    else:
        ecp_flag = False
    file_r.close()

    # Element identity from the true atomic number Z, kept distinct from the
    # raw ``nucleus_label`` (which QE-origin files may tag 'O' / 'O1' for AFM /
    # inequivalent sites). Crucially, when an ECP is present TREXIO stores the
    # *valence* charge in ``nucleus_charge`` (core electrons live in the ecp
    # group), so the full Z is the valence charge plus the per-nucleus
    # ``ecp_z_core``; with no ECP the charge is already the full Z. This is why
    # we do NOT parse the element from the label string: label 'O1' would be
    # unmappable, whereas 6.0 + z_core(2) = 8 -> 'O' is unambiguous.
    z_core_per_nucleus = [int(round(float(zc))) for zc in ecp_z_core] if ecp_flag else [0] * len(charges_r)
    if len(z_core_per_nucleus) != len(charges_r):
        raise ValueError(
            f"TREXIO ecp_z_core length {len(z_core_per_nucleus)} != nucleus count "
            f"{len(charges_r)}; cannot reconstruct atomic numbers."
        )
    atomic_numbers_r = [int(round(float(z))) + zc for z, zc in zip(charges_r, z_core_per_nucleus)]
    element_symbols_r = _convert_from_atomic_numbers_to_atomic_labels(atomic_numbers_r)

    # Structure_data instance
    if pbc_flag:
        if store_tuple:
            vec_a, vec_b, vec_c = tuple(cell_a), tuple(cell_b), tuple(cell_c)
        else:
            vec_a, vec_b, vec_c = list(cell_a), list(cell_b), list(cell_c)
    else:
        if store_tuple:
            vec_a, vec_b, vec_c = (), (), ()
        else:
            vec_a, vec_b, vec_c = [], [], []

    # Derived static field _n_max owned by Structure_data (via _compute_n_max).
    # Computed here so both the store_tuple (tuple) and the non-store_tuple
    # (list) branches preserve their container-type performance distinction
    # while still populating the periodic minimum-image search-box half-width.
    if pbc_flag:
        _n_max = Structure_data._compute_n_max(
            np.asarray(vec_a, dtype=np.float64),
            np.asarray(vec_b, dtype=np.float64),
            np.asarray(vec_c, dtype=np.float64),
        )
    else:
        _n_max = 1

    if store_tuple:
        structure_data = Structure_data(
            pbc_flag=pbc_flag,
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=tuple(atomic_numbers_r),
            element_symbols=tuple(element_symbols_r),
            atomic_labels=tuple(labels_r),
            positions=np.array(coords_r, dtype=dtype_np),
            _n_max=_n_max,
        )
    else:
        structure_data = Structure_data(
            pbc_flag=pbc_flag,
            vec_a=vec_a,
            vec_b=vec_b,
            vec_c=vec_c,
            atomic_numbers=list(atomic_numbers_r),
            element_symbols=list(element_symbols_r),
            atomic_labels=list(labels_r),
            positions=np.array(coords_r, dtype=dtype_np),
            _n_max=_n_max,
        )

    # ao spherical part check
    if ao_cartesian:
        logger.info("Cartesian basis functions.")
        # AOs_data instance
        ao_num_count = 0
        ao_prim_num_count = 0

        # values to be stored
        nucleus_index = []
        atomic_center_carts = []
        angular_momentums = []
        orbital_indices = []
        exponents = []
        coefficients = []
        polynominal_order_x = []
        polynominal_order_y = []
        polynominal_order_z = []
        radial_indices = []
        radial_exponents = []
        radial_coefficients = []
        radial_shells = []
        component_scales = []

        for i_shell in range(basis_shell_num):
            b_nucleus_index = basis_nucleus_index[i_shell]
            b_coord = list(coords_r[b_nucleus_index])
            b_ang_mom = basis_shell_ang_mom[i_shell]
            poly_orders = ["".join(p) for p in itertools.combinations_with_replacement("xyz", b_ang_mom)]
            poly_x = [poly_order.count("x") for poly_order in poly_orders]
            poly_y = [poly_order.count("y") for poly_order in poly_orders]
            poly_z = [poly_order.count("z") for poly_order in poly_orders]
            num_ao_mag_moms = len(poly_orders)

            ao_nucleus_index = [b_nucleus_index for _ in range(num_ao_mag_moms)]
            ao_coords = [b_coord for _ in range(num_ao_mag_moms)]
            ao_ang_moms = [b_ang_mom for _ in range(num_ao_mag_moms)]

            b_prim_indices = [i for i, v in enumerate(basis_shell_index) if v == i_shell]
            b_prim_num = len(b_prim_indices)

            N_n_dup_fuctorial_part = [
                (scipy.special.factorial(nx) * scipy.special.factorial(ny) * scipy.special.factorial(nz))
                / (scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz))
                for nx, ny, nz in zip(poly_x, poly_y, poly_z, strict=True)
            ]

            N_n_dup_Z_part = [
                (2.0 * basis_exponent[k] / np.pi) ** (3.0 / 2.0) * (8.0 * basis_exponent[k]) ** b_ang_mom
                for k in b_prim_indices
            ]
            b_prim_exponents = [basis_exponent[k] for k in b_prim_indices]

            ao_exponents = b_prim_exponents * num_ao_mag_moms
            ao_coefficients_list = []
            for p in range(num_ao_mag_moms):
                ao_coefficients_list += [
                    basis_shell_factor[i_shell]
                    * basis_prim_factor[k]
                    / np.sqrt(N_n_dup_Z_part[i] * N_n_dup_fuctorial_part[p])
                    * basis_coefficient[k]
                    for i, k in enumerate(b_prim_indices)
                ]

            if ao_shell is None:
                ao_ids = list(range(ao_num_count, ao_num_count + num_ao_mag_moms))
            else:
                ao_ids = np.flatnonzero(ao_shell == i_shell).tolist()
                if len(ao_ids) != num_ao_mag_moms:
                    raise ValueError(f"TREXIO Cartesian shell {i_shell} has the wrong number of AOs.")
            orbital_indices_all = [ao_id for ao_id in ao_ids for _ in range(b_prim_num)]

            ao_coefficients = [
                ao_coefficients_list[k] * ao_normalization[orbital_indices_all[k]] for k in range(len(ao_coefficients_list))
            ]
            if not pbc_flag:
                # The file's shell identity, rather than rounded component
                # coefficients, determines the optimizer's radial coordinates.
                start = len(radial_exponents)
                radial_indices.extend(list(range(start, start + b_prim_num)) * num_ao_mag_moms)
                radial_exponents.extend(b_prim_exponents)
                radial_shells.extend([i_shell] * b_prim_num)
                reference_norm = float(ao_normalization[ao_ids[0]]) or 1.0
                radial_coefficients.extend(c * reference_norm for c in ao_coefficients_list[:b_prim_num])
                component_scales.extend(
                    np.sqrt(N_n_dup_fuctorial_part[0] / part) * (float(ao_normalization[ao_id]) / reference_norm)
                    for ao_id, part in zip(ao_ids, N_n_dup_fuctorial_part, strict=True)
                )
            ao_num_count += num_ao_mag_moms
            ao_prim_num_count += num_ao_mag_moms * b_prim_num

            nucleus_index += ao_nucleus_index
            atomic_center_carts += ao_coords
            angular_momentums += ao_ang_moms
            polynominal_order_x += poly_x
            polynominal_order_y += poly_y
            polynominal_order_z += poly_z
            orbital_indices += orbital_indices_all
            exponents += ao_exponents
            coefficients += ao_coefficients

        if ao_num_count != ao_num:
            logger.error(f"ao_num_count = {ao_num_count} is inconsistent with the read ao_num = {ao_num}")
            raise ValueError

        if ao_shell is not None:
            # Shell traversal computes each component's angular metadata in
            # order within its shell. Restore the file's AO row order (the MO
            # coefficient columns) even when shell IDs are interleaved.
            ao_order = np.argsort(ao_shell, kind="stable").argsort()
            nucleus_index = np.asarray(nucleus_index)[ao_order].tolist()
            angular_momentums = np.asarray(angular_momentums)[ao_order].tolist()
            polynominal_order_x = np.asarray(polynominal_order_x)[ao_order].tolist()
            polynominal_order_y = np.asarray(polynominal_order_y)[ao_order].tolist()
            polynominal_order_z = np.asarray(polynominal_order_z)[ao_order].tolist()
            component_scales = np.asarray(component_scales)[ao_order].tolist()
            primitive_order = np.argsort(orbital_indices, kind="stable")
            orbital_indices = np.asarray(orbital_indices)[primitive_order].tolist()
            exponents = np.asarray(exponents)[primitive_order].tolist()
            coefficients = np.asarray(coefficients)[primitive_order].tolist()
            radial_indices = np.asarray(radial_indices)[primitive_order].tolist()

        # Single branch over (pbc_flag, store_tuple). Periodic systems go
        # through the pgto factory; molecular through the gto leaf.
        if store_tuple:
            cart_kwargs = dict(
                structure_data=structure_data,
                nucleus_index=tuple(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=tuple(angular_momentums),
                polynominal_order_x=tuple(polynominal_order_x),
                polynominal_order_y=tuple(polynominal_order_y),
                polynominal_order_z=tuple(polynominal_order_z),
                orbital_indices=tuple(orbital_indices),
                exponents=np.array(exponents, dtype=dtype_np),
                coefficients=np.array(coefficients, dtype=dtype_np),
            )
        else:
            cart_kwargs = dict(
                structure_data=structure_data,
                nucleus_index=list(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=list(angular_momentums),
                polynominal_order_x=list(polynominal_order_x),
                polynominal_order_y=list(polynominal_order_y),
                polynominal_order_z=list(polynominal_order_z),
                orbital_indices=list(orbital_indices),
                exponents=np.array(exponents, dtype=dtype_np),
                coefficients=np.array(coefficients, dtype=dtype_np),
            )
        if pbc_flag:
            # Dispatch the determinant pgto leaf on ``k_point_frac``.
            # Non-Gamma -> complex Bloch-pGTO; Gamma / molecular -> real
            # pgto / gto as before. The Jastrow 3-body basis stays real
            # Gamma pGTO and is built elsewhere from a ``k_point=None``
            # structure regardless.
            if k_point_frac is not None:
                aos_data = build_AOs_cart_pgto_complex_data(**cart_kwargs, k_point=tuple(float(c) for c in k_point_frac))
                # BCS / TR-pair convention: the dn-spin geminal evaluates phi^{-k}
                # = conj(phi^k) on a separate -k Bloch leaf so the engine's
                # geminal kernel sees a true time-reversal pair without any
                # spin-asymmetric conj at the orbital-evaluation level.
                minus_k = tuple(-float(c) for c in k_point_frac)
                aos_data_dn = build_AOs_cart_pgto_complex_data(**cart_kwargs, k_point=minus_k)
            else:
                aos_data = AOs_cart_pgto_data.build(**cart_kwargs)
                aos_data_dn = aos_data  # Gamma: dn shares the +k = -k leaf
        else:
            aos_data = AOs_cart_gto_data._from_full_basis(
                **cart_kwargs,
                radial=AO_radial_shell_data(
                    exponents=np.asarray(radial_exponents, dtype=dtype_np),
                    coefficients=np.asarray(radial_coefficients, dtype=dtype_np),
                    shell_prim_to_shell=tuple(int(shell) for shell in radial_shells),
                    nucleus_index=tuple(int(atom) for atom in basis_nucleus_index),
                    angular_momentum=tuple(int(l) for l in basis_shell_ang_mom),
                ),
                ao_prim_to_shell_prim=radial_indices,
                ao_coefficient_scales=tuple(float(scale) for scale in component_scales),
            )
            aos_data_dn = aos_data  # molecular: no Bloch phase, dn shares the leaf
    else:
        logger.debug("Spherical basis functions.")
        # AOs_data instance
        ao_num_count = 0
        ao_prim_num_count = 0

        # values to be stored
        nucleus_index = []
        atomic_center_carts = []
        angular_momentums = []
        magnetic_quantum_numbers = []
        orbital_indices = []
        exponents = []
        coefficients = []

        for i_shell in range(basis_shell_num):
            b_nucleus_index = basis_nucleus_index[i_shell]
            b_coord = list(coords_r[b_nucleus_index])
            b_ang_mom = basis_shell_ang_mom[i_shell]
            ao_mag_mom_list = [0] + [i * (-1) ** j for i in range(1, b_ang_mom + 1) for j in range(2)]
            num_ao_mag_moms = len(ao_mag_mom_list)

            ao_nucleus_index = [b_nucleus_index for _ in range(num_ao_mag_moms)]
            ao_coords = [b_coord for _ in range(num_ao_mag_moms)]
            ao_ang_moms = [b_ang_mom for _ in range(num_ao_mag_moms)]

            b_prim_indices = [i for i, v in enumerate(basis_shell_index) if v == i_shell]
            b_prim_num = len(b_prim_indices)
            b_normalizations = [
                np.sqrt(
                    (
                        2.0 ** (2 * b_ang_mom + 3)
                        * scipy.special.factorial(b_ang_mom + 1)
                        * (2 * basis_exponent[k]) ** (b_ang_mom + 1.5)
                    )
                    / (scipy.special.factorial(2 * b_ang_mom + 2) * np.sqrt(np.pi))
                )
                for k in b_prim_indices
            ]
            b_prim_exponents = [basis_exponent[k] for k in b_prim_indices]
            b_prim_coefficients = [
                basis_shell_factor[i_shell]
                * basis_prim_factor[k]
                * np.sqrt(4 * np.pi)
                / np.sqrt(2 * b_ang_mom + 1)
                / b_normalizations[i]
                * basis_coefficient[k]
                for i, k in enumerate(b_prim_indices)
            ]
            orbital_indices_all = [ao_num_count + j for j in range(num_ao_mag_moms) for _ in range(b_prim_num)]
            ao_exponents = b_prim_exponents * num_ao_mag_moms
            ao_coefficients_list = b_prim_coefficients * num_ao_mag_moms
            ao_coefficients = [
                ao_coefficients_list[k] * ao_normalization[orbital_indices_all[k]] for k in range(len(ao_coefficients_list))
            ]
            ao_num_count += num_ao_mag_moms
            ao_prim_num_count += num_ao_mag_moms * b_prim_num

            nucleus_index += ao_nucleus_index
            atomic_center_carts += ao_coords
            angular_momentums += ao_ang_moms
            magnetic_quantum_numbers += ao_mag_mom_list
            orbital_indices += orbital_indices_all
            exponents += ao_exponents
            coefficients += ao_coefficients

        if ao_num_count != ao_num:
            logger.error(f"ao_num_count = {ao_num_count} is inconsistent with the read ao_num = {ao_num}")
            raise ValueError

        if store_tuple:
            sphe_kwargs = dict(
                structure_data=structure_data,
                nucleus_index=tuple(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=tuple(angular_momentums),
                magnetic_quantum_numbers=tuple(magnetic_quantum_numbers),
                orbital_indices=tuple(orbital_indices),
                exponents=np.array(exponents, dtype=dtype_np),
                coefficients=np.array(coefficients, dtype=dtype_np),
            )
        else:
            sphe_kwargs = dict(
                structure_data=structure_data,
                nucleus_index=list(nucleus_index),
                num_ao=ao_num_count,
                num_ao_prim=ao_prim_num_count,
                angular_momentums=list(angular_momentums),
                magnetic_quantum_numbers=list(magnetic_quantum_numbers),
                orbital_indices=list(orbital_indices),
                exponents=np.array(exponents, dtype=dtype_np),
                coefficients=np.array(coefficients, dtype=dtype_np),
            )
        if pbc_flag:
            # Mirror of the cart branch -- complex Bloch-pGTO leaf at
            # non-Gamma, real pgto at Gamma. See the cart branch above
            # for the Jastrow-side scope note.
            if k_point_frac is not None:
                aos_data = build_AOs_sphe_pgto_complex_data(**sphe_kwargs, k_point=tuple(float(c) for c in k_point_frac))
                minus_k = tuple(-float(c) for c in k_point_frac)
                aos_data_dn = build_AOs_sphe_pgto_complex_data(**sphe_kwargs, k_point=minus_k)
            else:
                aos_data = AOs_sphe_pgto_data.build(**sphe_kwargs)
                aos_data_dn = aos_data  # Gamma: dn shares the +k = -k leaf
        else:
            aos_data = AOs_sphe_gto_data._from_full_basis(**sphe_kwargs)
            aos_data_dn = aos_data  # molecular: dn shares the leaf

    # Periodic systems: the pgto factory above already returned an
    # image-tiled instance (``aos_data._n_image >= 1``).
    #
    # Strict contract:
    #   - pbc_flag == True  iff  aos_data is an AOs_*_pgto_data leaf
    #   - pbc_flag == False iff  aos_data is an AOs_*_gto_data leaf
    # Default truncation tol comes from jqmc._setting.AO_IMAGE_TRUNCATION_ATOL;
    # callers wanting a different value pass tol= to the factory directly.

    # MOs_data instance
    threshold_mo_occ = 1.0e-6

    # dn channel at non-Gamma: every spin treatment uses opposite twists. The
    # dn orbital is the complex conjugate of its own channel's +k orbital,
    # conj(phi^{k}) = phi^{-k} (the channel Hamiltonian is real: no vector
    # potential, no spin-orbit coupling). Restricted files (RHF / ROHF) take
    # the alpha set for that channel; unrestricted files (UHF / LSDA) take
    # the independent beta set.
    #
    # It is realised by two operations that must BOTH be applied: ``conj(C)`` on
    # the coefficients AND a separate -k Bloch leaf (``aos_data_dn``). With real
    # primitives and psi_k(r) = e^{ik.r} u_k(r) one has chi_a^{-k} =
    # conj(chi_a^{k}), so together they compose to
    #     phi_dn(r) = sum_a conj(C_a) chi_a^{-k}(r) = conj(phi_up(r)).
    # At Gamma (and for a molecule) the -k leaf IS the +k leaf and ``conj`` is a
    # no-op, so the construction reduces byte-identically to a shared
    # ``mo_dn = mo_up`` leaf.
    #
    # The pitfall is applying only ONE of the two -- conjugating the
    # coefficients while still evaluating on the +k leaf, or flipping the leaf
    # without conjugating. Either half gives a mixed-phase orbital that is
    # neither phi^{k} nor phi^{-k}, and that bug produced VMC energies ~2 Ha
    # BELOW the pyscf HF reference at twist k=(0.25,0.25,0.25) for Si primitive
    # cc-pVDZ. Keep the two in lockstep when touching this branch.

    if not spin_dependent:  # RHF (closed shell) or ORHF (open shell restricted)
        mo_indices = [i for (i, v) in enumerate(mo_spin) if v == 0]
        mo_coefficient_real_up = mo_coefficient_real[mo_indices]
        # BCS / TR-pair convention: the dn-channel orbital is phi^{-k} =
        # conj(phi^k), implemented as (mo_coefficients_dn = conj(C^k)) on the
        # -k Bloch AO leaf (``aos_data_dn``). Both the coefficient conj AND
        # the AO-leaf k-flip are required: combining only one yields a
        # mixed-phase orbital that is neither phi^k nor phi^{-k}. ``np.conj``
        # is a no-op on the real / Gamma path, where ``aos_data_dn == aos_data``;
        # the dn-channel evaluation then reduces byte-identically to the
        # ``mo_dn = mo_up`` shared-leaf form.
        mo_coefficient_real_dn = np.conj(mo_coefficient_real_up)
        mo_occ = mo_occupation[mo_indices]
        occupied = [i for (i, v) in enumerate(mo_occ) if v >= threshold_mo_occ]
        if len(occupied) < num_ele_up:
            raise ValueError(
                f"TREXIO SD requires {num_ele_up} occupied up-spin orbitals; occupation data supplies {len(occupied)}."
            )
        # Restricted open shell (ROHF / ROKS): TREXIO ``mo_occupation`` counts
        # electrons per spatial orbital (2 = alpha + beta, 1 = alpha only), so
        # the dn determinant is the doubly occupied MOs, NOT the first N_dn
        # stored rows: a non-aufbau file (e.g. occupations [2, 1, 2, 2, 2] from
        # Delta-SCF / MOM or a user-set mo_occ) stores a singly occupied MO
        # before doubly occupied ones. The up rows are therefore ordered by
        # decreasing occupation (paired block first, as SD_data with
        # time_reversal=True requires dn == conj(up[:N_dn])); reordering up
        # rows only flips the global sign of det_up. The sort is stable, so
        # closed-shell and aufbau-ordered files keep their storage order
        # byte-identically.
        up_indices = sorted(occupied[:num_ele_up], key=lambda i: -float(mo_occ[i]))
        dn_indices = up_indices[:num_ele_dn]
        mos_data_up = MOs_data(
            num_mo=num_ele_up,
            mo_coefficients=np.ascontiguousarray(mo_coefficient_real_up[up_indices]),
            aos_data=aos_data,
        )
        mos_data_dn = MOs_data(
            num_mo=num_ele_dn,
            mo_coefficients=np.ascontiguousarray(mo_coefficient_real_dn[dn_indices]),
            aos_data=aos_data_dn,
        )

    else:
        # ``spin_dependent=True`` branch: MOs stored separately per spin (UHF /
        # kUHF; also catches broken-symmetry / UHF-as-RHF where ``spin_polarized=False``
        # but alpha and beta orbitals are stored independently).
        mo_indices_up = [i for (i, v) in enumerate(mo_spin) if v == 0]
        mo_indices_dn = [i for (i, v) in enumerate(mo_spin) if v == 1]
        mo_coefficient_real_up = mo_coefficient_real[mo_indices_up]
        # Unrestricted (UHF / kUHF / LSDA) dn channel: the stored beta
        # orbitals stay variationally independent of alpha. As on every
        # route, the down channel sits at the opposite twist: the complex
        # conjugate of the beta channel's own +k orbitals, conj(C_beta) on
        # the -k AO leaf ``aos_data_dn`` (never conj(C_alpha)). Both
        # operations are identities on the real / Gamma path.
        mo_coefficient_real_dn = np.conj(mo_coefficient_real[mo_indices_dn])
        mo_occ_up = mo_occupation[mo_indices_up]
        mo_occ_dn = mo_occupation[mo_indices_dn]

        mo_considered_indices_up = [i for (i, v) in enumerate(mo_occ_up) if v >= threshold_mo_occ]
        mo_considered_indices_dn = [i for (i, v) in enumerate(mo_occ_dn) if v >= threshold_mo_occ]

        if len(mo_considered_indices_up) < len(mo_considered_indices_dn):
            raise ValueError(
                f"The number of occ. orbitals for up spins = {len(mo_considered_indices_up)} should be larger than those of down spins = {len(mo_considered_indices_dn)}."
            )

        if len(mo_considered_indices_up) < num_ele_up or len(mo_considered_indices_dn) < num_ele_dn:
            raise ValueError(
                "TREXIO occupation data does not supply the requested occupied rows: "
                f"up={len(mo_considered_indices_up)}/{num_ele_up}, dn={len(mo_considered_indices_dn)}/{num_ele_dn}."
            )
        mos_data_up = MOs_data(
            num_mo=num_ele_up,
            mo_coefficients=np.ascontiguousarray(mo_coefficient_real_up[mo_considered_indices_up[:num_ele_up]]),
            aos_data=aos_data,
        )
        mos_data_dn = MOs_data(
            num_mo=num_ele_dn,
            mo_coefficients=np.ascontiguousarray(mo_coefficient_real_dn[mo_considered_indices_dn[:num_ele_dn]]),
            aos_data=aos_data_dn,
        )

    # ``time_reversal`` records coefficient tying only (the spin twists are
    # always opposite): True for *restricted* spin treatments (RHF /
    # kRHF / ROHF / kROHF, where the dn channel is variationally tied to
    # ``conj(up)`` on the paired core) and False for *unrestricted* (UHF
    # / kUHF, where alpha and beta are variationally independent). The
    # ``spin_dependent`` TREXIO flag is the discriminator: restricted
    # solvers store a single MO set (False), unrestricted solvers store
    # two independent sets (True). ROHF therefore correctly keeps the
    # paired-core BCS / TR-pair structure even though ``N_up > N_dn``.
    _time_reversal_flag = not spin_dependent

    geminal_data = SD_data.build(
        num_electron_up=num_ele_up,
        num_electron_dn=num_ele_dn,
        mos_up=mos_data_up,
        mos_dn=mos_data_dn,
        time_reversal=_time_reversal_flag,
    )

    # Coulomb_potential_data instance
    if ecp_flag:
        if store_tuple:
            coulomb_potential_data = Coulomb_potential_data(
                structure_data=structure_data,
                ecp_flag=True,
                z_cores=tuple(ecp_z_core),
                max_ang_mom_plus_1=tuple(ecp_max_ang_mom_plus_1),
                num_ecps=ecp_num,
                ang_moms=tuple(ecp_ang_mom),
                nucleus_index=tuple(ecp_nucleus_index),
                exponents=tuple(ecp_exponent),
                coefficients=tuple(ecp_coefficient),
                powers=tuple(ecp_power + 2),
            )
        else:
            coulomb_potential_data = Coulomb_potential_data(
                structure_data=structure_data,
                ecp_flag=True,
                z_cores=list(ecp_z_core),
                max_ang_mom_plus_1=list(ecp_max_ang_mom_plus_1),
                num_ecps=ecp_num,
                ang_moms=list(ecp_ang_mom),
                nucleus_index=list(ecp_nucleus_index),
                exponents=list(ecp_exponent),
                coefficients=list(ecp_coefficient),
                powers=list(ecp_power + 2),
            )
    else:
        coulomb_potential_data = Coulomb_potential_data(structure_data=structure_data, ecp_flag=False)

    # Periodic systems: precompute the Ewald tables so the strict contract
    #   pbc_flag == True iff coulomb_potential_data.ewald_data is not None
    # holds at every loader exit. Downstream code
    # (Coulomb dispatch shims, compute_local_energy) can then assume the
    # tables are populated without further checks. Molecular systems
    # keep ewald_data = None and the dispatch shims fall through to the
    # bare 1/r implementations.
    if pbc_flag:
        ewald_data = build_ewald_data(
            structure_data,
            coulomb_potential_data,
            tol=1e-10,
            n_electron=int(num_ele_up) + int(num_ele_dn),
        )
        coulomb_potential_data = coulomb_potential_data.replace(ewald_data=ewald_data)

    return (
        structure_data,
        aos_data,
        mos_data_up,
        mos_data_dn,
        geminal_data,
        coulomb_potential_data,
    )


# ---------------------------------------------------------------------------
# Multi-twist TREXIO -> combined HDF5 converter
# ---------------------------------------------------------------------------
def read_trexio_kpoint_list(trexio_file: str) -> list[tuple[tuple[float, float, float], float]]:
    """Enumerate ``(k_point_fractional, weight)`` for every twist in a multi-k TREXIO save.

    The TREXIO ``pbc_k_point`` slot
    stores k-points in fractional reciprocal-lattice coordinates (matching
    the determinant AO leaf's ``k_point`` field); the ``pbc_k_point_weight`` slot
    stores the IBZ weights (typically normalised so they sum to 1.0 in
    the TREXIO convention). This helper exposes both in a single shot so
    the driver / converter can plan the per-twist work and the BZ-mean
    combine without re-opening the file.

    Args:
        trexio_file: Path to a TREXIO HDF5 save with ``pbc_k_point`` set.

    Returns:
        Length-``n_kpts`` list of ``(k_frac_tuple, weight)``. ``k_frac_tuple``
        is a 3-tuple of Python floats; weights are *not* normalised
        (:func:`jqmc._twist_average.normalize_twist_weights` performs the
        ``W_k = w_k / sum_k w_k`` normalisation at the BZ-mean combine
        site).

    Raises:
        ValueError: If the file is missing ``pbc_k_point``, its shape is
            invalid, or a multi-k source lacks a complete valid
            ``mo_k_point`` association.
    """
    f = trexio.File(trexio_file, mode="r", back_end=trexio.TREXIO_HDF5)
    try:
        if not trexio.has_pbc_k_point(f):
            raise ValueError(f"{trexio_file}: missing pbc_k_point; not a periodic Bloch save.")
        k_arr = np.asarray(trexio.read_pbc_k_point(f), dtype=np.float64)
        if k_arr.ndim == 1:
            if k_arr.shape != (3,):
                raise ValueError(f"{trexio_file}: 1-D pbc_k_point has unexpected shape {k_arr.shape}; expected (3,).")
            k_arr = k_arr.reshape(1, 3)
        elif k_arr.ndim != 2 or k_arr.shape[1] != 3:
            raise ValueError(f"{trexio_file}: pbc_k_point has unexpected shape {k_arr.shape}; expected (n_kpts, 3) or (3,).")
        n_kpts = k_arr.shape[0]
        _validate_trexio_kpoint_mo_mapping(f, n_kpts)
        if trexio.has_pbc_k_point_weight(f):
            w_arr = np.asarray(trexio.read_pbc_k_point_weight(f), dtype=np.float64).reshape(-1)
            if w_arr.size != n_kpts:
                raise ValueError(f"{trexio_file}: pbc_k_point_weight length {w_arr.size} != n_kpts {n_kpts}.")
        else:
            # Pre-multi-k writers may omit the weight; fall back to uniform.
            w_arr = np.full(n_kpts, 1.0 / float(n_kpts), dtype=np.float64)
    finally:
        del f
    return [((float(k_arr[i, 0]), float(k_arr[i, 1]), float(k_arr[i, 2])), float(w_arr[i])) for i in range(n_kpts)]


def convert_trexio_to_twist_average_input(
    trexio_file: str,
    output_path: str,
    *,
    jastrow_data: "object | None" = None,
    ewald_tol: float = 1.0e-10,
) -> None:
    """End-to-end multi-twist TREXIO -> combined HDF5 conversion.

    Walks the
    multi-k TREXIO save one twist at a time, loads the occupation-selected
    per-k MO block via :func:`read_trexio_file`, builds a per-twist complex
    Bloch SD (up rows: the twist's ``+k`` coefficients; down rows: the
    conjugate of the down channel's own ``+k`` set, ``conj(mo_up)`` for a
    restricted file and the conjugated beta set for an unrestricted file;
    the loaded Hamiltonian puts the up channel at ``+k`` and the down
    channel at ``-k``), and writes the
    occupied-row-only combined HDF5 input via
    :func:`jqmc._twist_average.save_twist_average_input`. The shared
    template Hamiltonian is taken from the ``ik=0`` twist (structure,
    Coulomb / ECP, AO basis are k-independent; the per-twist datasets
    override the geminal block at load time via
    :func:`jqmc._twist_average.load_twist`).

    Args:
        trexio_file: Path to a multi-k TREXIO HDF5 save.
        output_path: Destination combined HDF5 (``hamiltonian_data.h5``).
        jastrow_data: Optional :class:`jqmc.jastrow_factor.Jastrow_data`
            to attach to the shared template. ``None`` uses the default
            empty Jastrow (the user wires real Jastrow params later if
            the optimisation is to start from a non-trivial guess).
        ewald_tol: Forwarded to :func:`build_ewald_data` for the shared
            template's Coulomb block.
    """
    from ._twist_average import save_twist_average_input
    from .hamiltonians import Hamiltonian_data
    from .jastrow_factor import Jastrow_data
    from .wavefunction import Wavefunction_data

    kpts_meta = read_trexio_kpoint_list(trexio_file)
    n_twist = len(kpts_meta)
    if n_twist < 2:
        raise ValueError(f"{trexio_file}: stores {n_twist} k-point; a twist-average input needs at least two.")
    logger.info("TREXIO multi-twist convert: %d k-points found in %s", n_twist, trexio_file)

    # 1) Build the shared template from ik=0 (structure / Coulomb / ECP /
    #    AO basis are k-independent; the geminal carries placeholder
    #    occupied MO rows that get overridden per-twist at load time).
    structure_data, _aos, _mos_up, _mos_dn, geminal_template, coulomb_potential_data = read_trexio_file(
        trexio_file, store_tuple=True, twist_ik=0
    )
    if jastrow_data is None:
        jastrow_data = Jastrow_data()
    template_ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(
            jastrow_data=jastrow_data,
            geminal_data=geminal_template,
        ),
    )
    template_ham.sanity_check()

    # 2) Walk every twist; capture per-k occupied MOs + (k_frac, weight).
    k_points_per_twist: list = []
    weights_per_twist: list[float] = []
    mo_up_per_twist: list[np.ndarray] = []
    mo_dn_per_twist: list[np.ndarray] = []
    for ik in range(n_twist):
        sd_k, _, _, _, gem_k, _ = read_trexio_file(
            trexio_file,
            store_tuple=True,
            twist_ik=ik,
        )
        # The per-twist fractional k lives on the AO leaf (``aos_data.k_point``);
        # the structure carries no k. For an unfolded multi-k save the ``ik=0``
        # entry is ``(0, 0, 0)`` rather than ``None`` (every twist of a
        # multi-k save goes through the complex Bloch path).
        _kp_k = gem_k.orb_data_up_spin.aos_data.k_point
        if _kp_k is None:
            k_points_per_twist.append((0.0, 0.0, 0.0))
        else:
            k_points_per_twist.append(tuple(float(x) for x in _kp_k))
        weights_per_twist.append(float(kpts_meta[ik][1]))
        mo_up_per_twist.append(np.asarray(gem_k.orb_data_up_spin.mo_coefficients))
        mo_dn_per_twist.append(np.asarray(gem_k.orb_data_dn_spin.mo_coefficients))

    # 3) Emit the combined HDF5. ``save_twist_average_input`` writes
    #    ``/_meta`` + ``/shared`` + ``/twists`` with /_meta + /shared + /twists/twist_*.
    save_twist_average_input(
        filepath=output_path,
        template_hamiltonian=template_ham,
        k_points=k_points_per_twist,
        weights=weights_per_twist,
        mo_coefficients_up_per_twist=mo_up_per_twist,
        mo_coefficients_dn_per_twist=mo_dn_per_twist,
    )
    logger.info(
        "TREXIO multi-twist convert: wrote combined HDF5 (n_twist=%d, sum_weights=%.6f) -> %s",
        n_twist,
        float(np.sum(weights_per_twist)),
        output_path,
    )


def _convert_from_atomic_numbers_to_atomic_labels(charges_r: list[int]) -> list[str]:
    """Map atomic numbers (Z) to element symbols, up to Z = 86."""
    atomic_number_to_element = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
    }

    labels_r = []

    for charge in charges_r:
        if charge <= 0:
            raise ValueError("Atomic number must be greater than 0.")
        if charge > 86:
            raise NotImplementedError("Atomic numbers above 86 are not implemented.")

        if charge in atomic_number_to_element:
            labels_r.append(atomic_number_to_element[charge])
        else:
            raise ValueError(f"No element for atomic number: {charge}")

    return labels_r


def _convert_from_atomic_labels_to_atomic_numbers(labels_r: list[str]) -> list[int]:
    """Mapping of element symbols to their atomic numbers up to 86."""
    element_to_number = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
        "In": 49,
        "Sn": 50,
        "Sb": 51,
        "Te": 52,
        "I": 53,
        "Xe": 54,
        "Cs": 55,
        "Ba": 56,
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Pm": 61,
        "Sm": 62,
        "Eu": 63,
        "Gd": 64,
        "Tb": 65,
        "Dy": 66,
        "Ho": 67,
        "Er": 68,
        "Tm": 69,
        "Yb": 70,
        "Lu": 71,
        "Hf": 72,
        "Ta": 73,
        "W": 74,
        "Re": 75,
        "Os": 76,
        "Ir": 77,
        "Pt": 78,
        "Au": 79,
        "Hg": 80,
        "Tl": 81,
        "Pb": 82,
        "Bi": 83,
        "Po": 84,
        "At": 85,
        "Rn": 86,
    }

    # Convert labels to atomic numbers, checking for validity
    atomic_numbers = []
    for label in labels_r:
        if label in element_to_number:
            atomic_number = element_to_number[label]
            if atomic_number > 86:
                raise NotImplementedError("Atomic numbers above 86 are not implemented.")
            atomic_numbers.append(atomic_number)
        else:
            raise ValueError(f"No atomic number found for the label '{label}'")
    return atomic_numbers
