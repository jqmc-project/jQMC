"""Quantum ESPRESSO plane-wave reference reader (HDF5 + XML).

This module loads a PWSCF ground state so it can be projected onto the
all-primitive pGTO basis seeded by :mod:`jqmc.upf_wrapper`. It reads the two
artefacts QE writes into ``<prefix>.save/``:

* ``wfc<ik>.hdf5`` -- the plane-wave Kohn--Sham orbitals: integer
  ``MillerIndices`` and the interleaved real/imag coefficient block ``evc``.
  Gamma-point runs store only one of each ``(G, -G)`` pair; the real orbital
  satisfies ``c(-G) = c(G)^*`` so the half set is expanded here.
* ``data-file-schema.xml`` -- the direct lattice ``a1/a2/a3`` (Bohr), the PW
  cutoff ``ecutwfc``, the atomic positions, and the band eigenvalues /
  occupations.

Everything is returned as plain NumPy in Hartree atomic units (lengths in
Bohr). The plane-wave convention is ``psi(r) = sum_G c_G exp(iG.r)`` with the
kinetic cutoff ``0.5 |G|^2 <= ecutwfc``.

Precision Zones:
    This module owns no Precision Zone; it computes in NumPy / h5py fp64.

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

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from glob import glob
from logging import getLogger

import numpy as np
import numpy.typing as npt

logger = getLogger("jqmc").getChild(__name__)

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class QECell:
    """Direct/reciprocal lattice of a QE run (Bohr / Bohr^-1, atomic units).

    ``a[i]`` is the i-th direct lattice vector (row); ``b[i]`` the matching
    reciprocal vector with ``a[i] . b[j] = 2*pi*delta_ij`` -- the basis in which
    the wfc Miller indices are expressed.
    """

    a: npt.NDArray[np.float64]

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """Reciprocal lattice rows with ``a[i] . b[j] = 2*pi*delta_ij``."""
        return _TWO_PI * np.linalg.inv(self.a).T

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.a)))


@dataclass(frozen=True)
class QEWavefunction:
    """Plane-wave Kohn--Sham orbitals from a single ``wfc<ik>.hdf5``.

    Attributes:
        millers: Integer reciprocal indices, shape ``(n_g, 3)``. For a
            ``gamma_only`` run these are already expanded to the full
            ``(G, -G)`` set.
        coeffs: Complex PW coefficients ``c_{n,G}``, shape ``(n_bnd, n_g)``.
        gamma_only: Whether the source file used the Gamma-point half storage.
        k_point: Crystal-coordinate k point ``xk`` (zeros at Gamma).
    """

    millers: npt.NDArray[np.int64]
    coeffs: npt.NDArray[np.complex128]
    gamma_only: bool
    k_point: npt.NDArray[np.float64]

    @property
    def n_bands(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def n_gvectors(self) -> int:
        return int(self.coeffs.shape[1])

    def g_vectors(self, cell: QECell) -> npt.NDArray[np.float64]:
        """Cartesian reciprocal vectors ``G = millers @ b`` for this wfc."""
        return self.millers.astype(np.float64) @ cell.b


@dataclass(frozen=True)
class QEBands:
    """Band eigenvalues / occupations from ``data-file-schema.xml`` (Gamma)."""

    eigenvalues: npt.NDArray[np.float64]  # Hartree
    occupations: npt.NDArray[np.float64]
    n_electrons: float
    fermi_energy: float | None  # Hartree

    def occupied_band_indices(self, occ_tol: float = 1.0e-6) -> npt.NDArray[np.int64]:
        """Indices of bands with non-negligible occupation."""
        return np.nonzero(self.occupations > occ_tol)[0].astype(np.int64)


@dataclass(frozen=True)
class QEReference:
    """A fully-loaded QE reference: cell, atoms, cutoff, bands and wavefunction."""

    cell: QECell
    atom_labels: tuple[str, ...]
    atom_positions: npt.NDArray[np.float64]  # (n_atom, 3) Cartesian Bohr
    ecutwfc: float  # Hartree (0.5|G|^2 <= ecutwfc)
    fft_grid: tuple[int, int, int]  # dense FFT mesh (nr1, nr2, nr3)
    bands: QEBands
    wfc: QEWavefunction
    # Authoritative per-atom element symbol, resolved from the UPF that each
    # species' (possibly tagged) label points at -- NOT parsed from the label
    # string, which QE lets users tag freely ('O' vs 'O1', 'Fe2+' vs 'Fe3+')
    # for AFM / distinct-environment setups. Defaults to () so lightweight
    # unit-test references (that never build a Structure_data) stay valid;
    # ``load_qe_reference`` always populates it (see ``_resolve_atom_elements``).
    atom_elements: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# wfc HDF5
# ---------------------------------------------------------------------------
def expand_gamma_half(
    millers: npt.NDArray[np.int64], coeffs: npt.NDArray[np.complex128]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.complex128]]:
    """Expand a ``gamma_only`` half G-set to the full set via ``c(-G)=c(G)^*``.

    QE stores one of each ``(G, -G)`` pair (and ``G=0`` once) for Gamma-point
    real orbitals. The conjugate partners are appended so the orbital can be
    summed on a real grid / projected without aliasing.

    Returns ``(millers_full (M,3), coeffs_full (n_bnd,M))`` with
    ``M = 2*n_g - 1`` (``G=0`` is not duplicated).
    """
    nonzero = np.any(millers != 0, axis=1)
    millers_full = np.concatenate([millers, -millers[nonzero]], axis=0)
    coeffs_full = np.concatenate([coeffs, np.conj(coeffs[:, nonzero])], axis=1)
    return millers_full, coeffs_full


def load_qe_wfc(wfc_path: str) -> QEWavefunction:
    """Read a PWSCF ``wfc<ik>.hdf5`` plane-wave wavefunction file.

    Applies the stored ``scale_factor`` and expands ``gamma_only`` storage to
    the full G set.
    """
    import h5py

    if not os.path.exists(wfc_path):
        raise FileNotFoundError(f"{wfc_path} not found.")

    with h5py.File(wfc_path, "r") as f:
        attrs = dict(f.attrs)
        millers = np.asarray(f["MillerIndices"], dtype=np.int64)
        evc = np.asarray(f["evc"], dtype=np.float64)

    # Noncollinear / spin-orbit runs (``npol = 2``) store each band as a
    # two-component spinor: ``npol * igwx`` complex PW columns per band
    # ([up | dn] blocks). Only spin-restricted and LSDA saves (``npol = 1``)
    # are supported; projecting a spinor block against ``igwx`` Miller
    # indices would silently keep the first component only.
    npol_attr = attrs.get("npol")
    npol = 1 if npol_attr is None else int(np.asarray(npol_attr).reshape(-1)[0])
    if npol != 1:
        raise ValueError(
            f"{wfc_path}: npol={npol} (noncollinear / spin-orbit spinor "
            "wavefunction) is not supported; the QE -> jQMC converter accepts only spin-restricted "
            "(nspin=1) and LSDA (nspin=2) saves with npol=1."
        )
    if evc.ndim != 2 or millers.ndim != 2 or evc.shape[1] != 2 * millers.shape[0]:
        raise ValueError(
            f"{wfc_path}: evc block shape {evc.shape} does not hold one interleaved real/imag "
            f"column pair per Miller index ({millers.shape[0]} MillerIndices); a wider block "
            "indicates a noncollinear (npol=2) spinor save, which is not supported."
        )

    # Assemble into one complex buffer. Writing the interleaved real and
    # imaginary views separately avoids retaining multiple full-size
    # wavefunction temporaries, and scaling in place avoids another copy.
    coeffs = np.empty((evc.shape[0], evc.shape[1] // 2), dtype=np.complex128)
    coeffs.real = evc[:, 0::2]
    coeffs.imag = evc[:, 1::2]
    del evc
    coeffs *= float(attrs.get("scale_factor", 1.0))

    gamma_only_attr = attrs.get("gamma_only")
    if isinstance(gamma_only_attr, (bytes, np.bytes_)):
        gamma_only_attr = gamma_only_attr.decode()
    if isinstance(gamma_only_attr, str):
        gamma_only = ".true." in gamma_only_attr.lower() or gamma_only_attr.strip().lower() in ("t", "true")
    else:
        gamma_only = bool(gamma_only_attr)
    if gamma_only:
        millers, coeffs = expand_gamma_half(millers, coeffs)

    xk = np.asarray(attrs.get("xk", np.zeros(3)), dtype=np.float64).reshape(3)
    return QEWavefunction(millers=millers, coeffs=coeffs, gamma_only=gamma_only, k_point=xk)


# ---------------------------------------------------------------------------
# data-file-schema.xml
# ---------------------------------------------------------------------------
def _floats(text: str | None) -> npt.NDArray[np.float64]:
    return np.array([float(x) for x in (text or "").split()], dtype=np.float64)


def _find_scoped(root: ET.Element, rel_path: str) -> ET.Element | None:
    """Find ``rel_path`` preferring the converged ``output`` block.

    The QE schema repeats ``atomic_structure``/``basis_set``/``band_structure``
    under both ``input`` and ``output``; a bare ``.//`` match would return the
    input copy first and (for ``findall``) duplicate the structure. Scope to
    ``output`` first and fall back to ``.//`` for older schemas.
    """
    out = root.find("output")
    if out is not None:
        hit = out.find(rel_path)
        if hit is not None:
            return hit
    return root.find(".//" + rel_path)


def load_qe_cell(xml_path: str) -> QECell:
    """Read the direct lattice ``a1/a2/a3`` (Bohr rows) from the QE XML."""
    root = ET.parse(xml_path).getroot()
    cell = _find_scoped(root, "atomic_structure/cell")
    if cell is None:
        raise ValueError(f"No atomic_structure/cell in {xml_path}.")
    a = np.array(
        [_floats(cell.find(tag).text) for tag in ("a1", "a2", "a3")],
        dtype=np.float64,
    )
    return QECell(a=a)


def load_qe_atoms(xml_path: str) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """Read atom labels and Cartesian positions (Bohr) from the QE XML."""
    root = ET.parse(xml_path).getroot()
    positions_el = _find_scoped(root, "atomic_structure/atomic_positions")
    if positions_el is None:
        raise ValueError(f"No atomic_structure/atomic_positions in {xml_path}.")
    atoms = positions_el.findall("atom")
    labels = tuple((a.get("name") or "").strip() for a in atoms)
    positions = np.array([_floats(a.text) for a in atoms], dtype=np.float64)
    return labels, positions


def load_qe_species_pseudo(xml_path: str) -> dict[str, str]:
    """Map each QE species label to its UPF filename from ``<atomic_species>``.

    QE lets chemically / magnetically inequivalent sites of one element carry
    tagged labels (``O`` vs ``O1``, ``Fe2+`` vs ``Fe3+``), each pointing at its
    own UPF. Returns ``{label: pseudo_file}`` so callers can recover the
    authoritative element from the pseudopotential rather than from the
    free-form label. The XML carries ``<atomic_species>`` in both the input and
    output sections with identical content, so we dedupe by label (first wins).
    """
    root = ET.parse(xml_path).getroot()
    species_pseudo: dict[str, str] = {}
    for species in root.iter("species"):
        name = species.get("name")
        pseudo_el = species.find("pseudo_file")
        if name is None or pseudo_el is None or not (pseudo_el.text or "").strip():
            continue
        name = name.strip()
        if name not in species_pseudo:
            species_pseudo[name] = pseudo_el.text.strip()
    return species_pseudo


def _resolve_atom_elements(labels: tuple[str, ...], xml_path: str, save_dir: str) -> tuple[str, ...]:
    """Resolve each atom's element from the UPF its species label points at.

    This is the QE-consistent, unambiguous source of element identity: QE
    itself takes the element from the pseudopotential (``upf%psd``), never by
    parsing the species label. Raises with an actionable message when a
    species has no ``<pseudo_file>`` or the UPF is not co-located in the save
    dir -- deliberately no fall back to label-string parsing, which would
    silently mis-identify tagged labels (e.g. ``CO`` -> carbon).
    """
    from .upf_wrapper import parse_upf_element

    species_pseudo = load_qe_species_pseudo(xml_path)
    element_of_label: dict[str, str] = {}
    for label in labels:
        if label in element_of_label:
            continue
        pseudo_file = species_pseudo.get(label)
        if pseudo_file is None:
            raise ValueError(
                f"QE species {label!r} has no <pseudo_file> in the XML atomic_species "
                f"block, so its element cannot be determined. jQMC reads the element "
                f"from the pseudopotential's PP_HEADER, not from the label string."
            )
        upf_path = os.path.join(save_dir, pseudo_file)
        if not os.path.isfile(upf_path):
            raise FileNotFoundError(
                f"QE species {label!r} -> UPF {pseudo_file!r} not found in {save_dir}; "
                f"the .upf must accompany the save so jQMC can read the element from "
                f"its PP_HEADER."
            )
        element_of_label[label] = parse_upf_element(upf_path)
    return tuple(element_of_label[label] for label in labels)


def load_qe_ecutwfc(xml_path: str) -> float:
    """Read the plane-wave kinetic cutoff ``ecutwfc`` (Hartree) from the QE XML."""
    root = ET.parse(xml_path).getroot()
    ec = _find_scoped(root, "basis_set/ecutwfc")
    if ec is None or ec.text is None:
        raise ValueError(f"No basis_set/ecutwfc in {xml_path}.")
    return float(ec.text)


def load_qe_fft_grid(xml_path: str) -> tuple[int, int, int]:
    """Read the dense FFT mesh ``(nr1, nr2, nr3)`` from ``basis_set/fft_grid``.

    This is the real-space sampling grid on which the PW orbitals are exact
    (every stored Miller index lies within its Nyquist range), so it is the
    natural quadrature grid for the PW -> pGTO projection.
    """
    root = ET.parse(xml_path).getroot()
    fg = _find_scoped(root, "basis_set/fft_grid")
    if fg is None:
        raise ValueError(f"No basis_set/fft_grid in {xml_path}.")
    return (int(fg.get("nr1")), int(fg.get("nr2")), int(fg.get("nr3")))


def load_qe_lsda(xml_path: str) -> bool:
    """Whether the QE run is spin-polarized LSDA (``nspin = 2``).

    Reads ``band_structure/lsda`` from ``data-file-schema.xml``. LSDA saves
    store the two spin channels as separate wavefunction files
    (``wfcup<ik>.hdf5`` / ``wfcdw<ik>.hdf5``) and concatenate the per-spin
    eigenvalues / occupations inside each ``ks_energies`` block
    (up block of ``nbnd_up`` entries first, then ``nbnd_dw`` down entries).
    """
    root = ET.parse(xml_path).getroot()
    bs = _find_scoped(root, "band_structure")
    if bs is None:
        return False
    el = bs.find("lsda")
    return el is not None and (el.text or "").strip().lower() == "true"


def load_qe_noncollinear(xml_path: str) -> bool:
    """Whether the QE run is noncollinear or spin-orbit (two-component spinors).

    QE records ``noncolin`` / ``spinorbit`` flags in the ``input/spin``,
    ``output/magnetization`` and ``output/band_structure`` blocks of
    ``data-file-schema.xml``. Any flag set to ``true`` marks a spinor
    (``npol = 2``) save, which the QE -> jQMC converter does not support.
    """
    root = ET.parse(xml_path).getroot()
    for tag in ("noncolin", "spinorbit"):
        for el in root.iter(tag):
            if (el.text or "").strip().lower() == "true":
                return True
    return False


def load_qe_bands(xml_path: str, ik: int = 1, spin: str | None = None) -> QEBands:
    """Read per-k eigenvalues, occupations and electron count (Hartree).

    Args:
        xml_path: ``data-file-schema.xml`` path.
        ik: 1-based k-point index. ``1`` selects the first ``ks_energies``
            block (the Gamma point for a single-twist run, or the first
            IBZ entry for a multi-k run).
        spin: ``None`` for a spin-restricted run. For an LSDA
            (``nspin = 2``) save pass ``"up"`` or ``"dw"`` to select the
            spin channel: QE concatenates the two channels inside one
            ``ks_energies`` block (first ``nbnd_up`` entries are up, the
            remaining ``nbnd_dw`` are down). ``n_electrons`` stays the
            *total* valence electron count in both cases; the per-channel
            electron count is the sum of the returned ``occupations``.

    A multi-k QE save has one ``ks_energies`` block per k-point, so
    ``load_qe_reference(path, ik=N)`` must read the N-th block, not
    always the first. Defaults to ``ik=1`` so existing call sites are
    unchanged.
    """
    root = ET.parse(xml_path).getroot()
    bs = _find_scoped(root, "band_structure")
    if bs is None:
        raise ValueError(f"No band_structure in {xml_path}.")
    ks_blocks = bs.findall("ks_energies")
    if not ks_blocks:
        raise ValueError(f"No ks_energies in {xml_path}.")
    if not (1 <= ik <= len(ks_blocks)):
        raise ValueError(f"ik={ik} out of range; data-file-schema.xml has {len(ks_blocks)} k-points.")
    ks = ks_blocks[ik - 1]
    eig = _floats(ks.find("eigenvalues").text)
    occ = _floats(ks.find("occupations").text)
    nelec = float(bs.find("nelec").text)
    fermi_el = bs.find("fermi_energy")
    fermi = float(fermi_el.text) if fermi_el is not None and fermi_el.text else None

    lsda_el = bs.find("lsda")
    is_lsda = lsda_el is not None and (lsda_el.text or "").strip().lower() == "true"
    if spin is not None and not is_lsda:
        raise ValueError(f"spin={spin!r} requested but {xml_path} is not an LSDA (nspin=2) save.")
    if is_lsda:
        if spin not in ("up", "dw"):
            raise ValueError(
                f"{xml_path} is an LSDA (nspin=2) save; pass spin='up' or spin='dw' to select the channel (got spin={spin!r})."
            )
        nbnd_up_el = bs.find("nbnd_up")
        nbnd_dw_el = bs.find("nbnd_dw")
        if nbnd_up_el is None or nbnd_dw_el is None:
            raise ValueError(f"LSDA save {xml_path} is missing band_structure/nbnd_up or nbnd_dw.")
        nbnd_up = int(nbnd_up_el.text)
        nbnd_dw = int(nbnd_dw_el.text)
        if eig.size != nbnd_up + nbnd_dw or occ.size != nbnd_up + nbnd_dw:
            raise ValueError(
                f"LSDA save {xml_path}: ks_energies arrays have {eig.size} entries, "
                f"expected nbnd_up + nbnd_dw = {nbnd_up} + {nbnd_dw}."
            )
        if spin == "up":
            eig, occ = eig[:nbnd_up], occ[:nbnd_up]
        else:
            eig, occ = eig[nbnd_up:], occ[nbnd_up:]
    return QEBands(eigenvalues=eig, occupations=occ, n_electrons=nelec, fermi_energy=fermi)


def load_qe_kpoint_list(xml_path: str) -> list[tuple[npt.NDArray[np.float64], float]]:
    """Enumerate all k-points + weights from ``data-file-schema.xml``.

    QE stores ``<k_point>`` text in ``2 pi / alat`` Cartesian units and
    ``weight`` as the (un-normalised) IBZ multiplicity weight. This helper
    converts to ``1/bohr`` Cartesian (matching ``QEWavefunction.k_point``)
    by multiplying by ``2 pi / alat`` so the caller can feed the result
    directly into :func:`build_gto_pw_matrix` without further unit
    handling.

    Args:
        xml_path: ``data-file-schema.xml`` path.

    Returns:
        Length-``n_kpts`` list of ``(xk_bohr_inv, weight)`` pairs. ``xk_*``
        has shape ``(3,)`` and ``dtype=float64``. The weights are *not*
        normalised here -- :func:`jqmc._twist_average.normalize_twist_weights`
        does the BZ-mean normalisation at the driver side.

    Raises:
        ValueError: If the XML is missing ``ks_energies`` or
            ``atomic_structure[@alat]``.
    """
    root = ET.parse(xml_path).getroot()
    bs = _find_scoped(root, "band_structure")
    if bs is None:
        raise ValueError(f"No band_structure in {xml_path}.")
    # alat from <atomic_structure alat="..."> (Bohr).
    atom_struct = _find_scoped(root, "atomic_structure")
    if atom_struct is None or "alat" not in atom_struct.attrib:
        raise ValueError(f"{xml_path}: missing atomic_structure[@alat]; cannot scale k-points to 1/bohr.")
    alat = float(atom_struct.attrib["alat"])
    two_pi_per_alat = 2.0 * float(np.pi) / alat
    out: list[tuple[npt.NDArray[np.float64], float]] = []
    for ks in bs.findall("ks_energies"):
        kp_el = ks.find("k_point")
        if kp_el is None:
            continue
        xk_qe = _floats(kp_el.text).reshape(3)
        xk_bohr_inv = xk_qe * two_pi_per_alat
        weight = float(kp_el.attrib.get("weight", "1.0"))
        out.append((xk_bohr_inv, weight))
    if not out:
        raise ValueError(f"{xml_path}: no ks_energies / k_point blocks found.")
    return out


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------
def _find_save_dir(path: str) -> str:
    """Resolve ``path`` to a QE ``*.save`` directory.

    Accepts the ``.save`` directory itself, its parent (containing exactly one
    ``*.save``), or a path to a file inside it.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.isdir(path) and path.rstrip("/").endswith(".save"):
        return path
    candidates = sorted(glob(os.path.join(path, "*.save")))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No *.save directory found under {path!r}.")
    raise ValueError(f"Multiple *.save directories under {path!r}: {candidates}.")


def qe_save_is_lsda(path: str) -> bool:
    """Whether the QE save at ``path`` is a spin-polarized (``nspin = 2``) run."""
    save_dir = _find_save_dir(path)
    xml_path = os.path.join(save_dir, "data-file-schema.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"{xml_path} not found.")
    return load_qe_lsda(xml_path)


def truncate_qe_reference_bands(reference: QEReference, n_keep: int) -> QEReference:
    """Return a copy of ``reference`` keeping only the lowest ``n_keep`` bands.

    QE orders bands by eigenvalue, so slicing the leading ``n_keep`` rows of
    the PW coefficient block (and the matching eigenvalue / occupation
    entries) keeps the energetically lowest states. Used by the LSDA
    (``nspin = 2``) conversion path to give both spin channels the *same*
    MO count (= the majority-channel occupied count): the unrestricted
    geminal packs a square paired lambda block, so ``num_mo_up`` must equal
    ``num_mo_dn`` even though the minority channel occupies fewer bands.
    """
    n_keep = int(n_keep)
    if not (0 < n_keep <= reference.wfc.n_bands):
        raise ValueError(f"n_keep={n_keep} out of range; the wavefunction has {reference.wfc.n_bands} bands.")
    if reference.bands.eigenvalues.size < n_keep or reference.bands.occupations.size < n_keep:
        raise ValueError(
            f"n_keep={n_keep} exceeds the XML band count "
            f"({reference.bands.eigenvalues.size} eigenvalues, {reference.bands.occupations.size} occupations)."
        )
    wfc = reference.wfc
    bands = reference.bands
    return QEReference(
        cell=reference.cell,
        atom_labels=reference.atom_labels,
        atom_positions=reference.atom_positions,
        ecutwfc=reference.ecutwfc,
        fft_grid=reference.fft_grid,
        bands=QEBands(
            eigenvalues=bands.eigenvalues[:n_keep],
            occupations=bands.occupations[:n_keep],
            n_electrons=bands.n_electrons,
            fermi_energy=bands.fermi_energy,
        ),
        wfc=QEWavefunction(
            millers=wfc.millers,
            coeffs=wfc.coeffs[:n_keep],
            gamma_only=wfc.gamma_only,
            k_point=wfc.k_point,
        ),
        atom_elements=reference.atom_elements,
    )


def load_qe_reference(path: str, ik: int = 1, spin: str | None = None) -> QEReference:
    """Load a complete QE reference from a ``*.save`` directory.

    Args:
        path: A ``*.save`` directory, its parent, or any file inside it.
        ik: 1-based k-point index selecting ``wfc<ik>.hdf5`` (1 at Gamma).
        spin: ``None`` for a spin-restricted save (reads ``wfc<ik>.hdf5``).
            For an LSDA (``nspin = 2``) save pass ``"up"`` / ``"dw"`` to
            read ``wfcup<ik>.hdf5`` / ``wfcdw<ik>.hdf5`` together with the
            matching half of the ``ks_energies`` eigenvalue / occupation
            arrays. Loading an LSDA save without ``spin`` (or vice versa)
            raises ``ValueError``.

    Returns:
        QEReference bundling cell, atoms, cutoff, bands and the wavefunction.
    """
    save_dir = _find_save_dir(path)
    xml_path = os.path.join(save_dir, "data-file-schema.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"{xml_path} not found.")
    if load_qe_noncollinear(xml_path):
        raise ValueError(
            f"{save_dir} is a noncollinear / spin-orbit save (noncolin or spinorbit = true, npol=2 spinor "
            "wavefunctions); the QE -> jQMC converter supports only spin-restricted (nspin=1) and "
            "LSDA (nspin=2) saves."
        )
    is_lsda = load_qe_lsda(xml_path)
    if is_lsda and spin not in ("up", "dw"):
        raise ValueError(
            f"{save_dir} is an LSDA (nspin=2) save; load each channel explicitly "
            "with spin='up' / spin='dw' (the QE -> jQMC UKS path loads both)."
        )
    if not is_lsda and spin is not None:
        raise ValueError(f"spin={spin!r} requested but {save_dir} is not an LSDA (nspin=2) save.")
    wfc_name = f"wfc{ik}.hdf5" if spin is None else f"wfc{spin}{ik}.hdf5"
    wfc_path = os.path.join(save_dir, wfc_name)
    if not os.path.exists(wfc_path):
        raise FileNotFoundError(f"{wfc_path} not found.")

    cell = load_qe_cell(xml_path)
    labels, positions = load_qe_atoms(xml_path)
    atom_elements = _resolve_atom_elements(labels, xml_path, save_dir)
    ecutwfc = load_qe_ecutwfc(xml_path)
    fft_grid = load_qe_fft_grid(xml_path)
    # Forward ``ik`` so a multi-k save delivers the per-k eigenvalues /
    # occupations from the matching ks_energies block. Bit-identical for
    # single-k Gamma fixtures where ``ik == 1`` selects the first (and
    # only) block.
    bands = load_qe_bands(xml_path, ik=ik, spin=spin)
    wfc = load_qe_wfc(wfc_path)
    if wfc.n_bands != bands.eigenvalues.size:
        raise ValueError(
            f"{wfc_path}: PW coefficient block has {wfc.n_bands} bands but the XML "
            f"ks_energies block (spin={spin!r}) has {bands.eigenvalues.size}."
        )

    logger.info(
        "Loaded QE reference from %s%s: %d atoms, %d bands, %d G-vectors, ecutwfc=%.3f Ha.",
        save_dir,
        "" if spin is None else f" (spin={spin})",
        len(labels),
        wfc.n_bands,
        wfc.n_gvectors,
        ecutwfc,
    )
    return QEReference(
        cell=cell,
        atom_labels=labels,
        atom_positions=positions,
        ecutwfc=ecutwfc,
        fft_grid=fft_grid,
        bands=bands,
        wfc=wfc,
        atom_elements=atom_elements,
    )
