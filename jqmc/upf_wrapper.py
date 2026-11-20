"""UPF pseudopotential reader: pGTO exponent init + analytic ECP extraction.

This module turns a Quantum ESPRESSO ``.upf`` pseudopotential into

1. an initial **all-primitive** Gaussian basis for the PW -> pGTO projection
   pipeline, and
2. an **analytic ccECP block** (local + non-local Gaussian semi-local terms)
   for the Coulomb Hamiltonian.

Both come from a single UPF file --- the converter does not need a separate
ccECP text file.

pGTO basis init (PSWFC fit). Parses the pseudo-atomic wavefunction channels
``chi_l(r) = r * R_l(r)`` stored in ``PP_PSWFC`` (both the legacy v1 text
layout and the v2 XML layout). These are the smooth, pseudized atomic valence
orbitals QE ships for ``projwfc.x`` PDOS / ``startingwfc='atomic'`` / DFT+U
projectors --- the core cusp and all-electron core nodes are already removed,
so their Gaussian content sits at moderate exponents. Each channel's radial
part is least-squares fit by a small set of *primitive* Gaussians

    R_l(r) ~ sum_p c_p r^l exp(-alpha_p r^2)

with **no contraction**. The fitted exponent ladder and the s/p/d... angular
channels seed the variational pGTO block; the exponents are optimised
downstream against the real solid Kohn--Sham orbitals.

Analytic ECP ([Gaussinfo] section). ccECP UPF files emitted by Opium with the
``rpt`` flag include a ``[Gaussinfo]`` block inside ``PP_INFO`` carrying the
original analytic Gaussian semi-local form

    V_l(r) = sum_k c_k r^(n_k - 2) exp(-alpha_k r^2)

with channels ordered ``l = 0, 1, ..., l_max - 1, local``. :func:`parse_gaussinfo`
extracts this directly into an :class:`ECPData` (jQMC / TREXIO convention,
local-first ang_mom tagging) so the same UPF file supplies both basis seeding
and ECP terms. Standard ONCV / GBRV / PAW UPF files do **not** carry
``[Gaussinfo]`` and are rejected with a clear error.

Exponent initialisation is provided by the PSWFC fit through
``ExponentInitStrategy``. It derives the Gaussian basis seed from the atomic
pseudo-wavefunction data in the UPF file; the subsequent solid-state
optimization is documented in the atomic-orbital guide.

Precision Zones:
    This module owns no Precision Zone; it computes in NumPy / SciPy fp64.

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

import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from ._setting import (
    AE_PGTO_CELL_SAFETY,
    PGTO_TIGHT_MAX_PER_SHELL,
    PGTO_TIGHT_RATIO,
    PW_PGTO_TAIL_TOL,
)

# Hardcoded chemistry-grade pGTO basis for bare-Coulomb AE H. See the
# extended rationale in ``generate_ae_even_tempered_shells``. The single
# real consumer of bare-Coulomb AE in jqmc today is ``H.coulomb-ae.UPF``;
# other elements ship UPFs with PSWFC channels and go through the
# ``fit_pswfc_channel_auto`` path. Values from cc-pVDZ uncontracted H.
#
# Tight s extension (75, 39) added on top of the cc-pVDZ core: the
# bare-Coulomb AE path does NOT run ``_extend_shells_tight`` (that helper
# is PSWFC/ECP-only), so without these the basis topped out at alpha=13.01
# and could not resolve the e-N cusp region. The Jastrow chi fixes the cusp
# *slope* (-Z), but the near-nucleus radial *shape* is the determinant's job
# and needs tight Gaussians. The tight ceiling is set by what the QE
# reference actually resolves: ecutwfc=150 Ha -> G_max=17.3 Bohr^-1, and the
# pGTO projection low-passes any Gaussian at G_max, so exponents past
# alpha ~ G_max^2/4 ~ 75 are truncated (and exceed the FFT-grid-safe ceiling
# alpha_max_safe ~ 61). Even-tempered ratio ~2 from 13.01 -> 39 -> 75.
_AE_H_S_EXPONENTS: tuple[float, ...] = (75.0, 39.0, 13.01, 1.962, 0.4446)
_AE_H_P_EXPONENT: float = 0.7270

logger = getLogger("jqmc").getChild(__name__)

# Angular-momentum letter <-> integer.
_L_FROM_LETTER = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5}
_LETTER_FROM_L = {v: k for k, v in _L_FROM_LETTER.items()}


class PSWFCFitError(RuntimeError):
    r"""least_squares did not report success, so its fitted exponents are not used."""


@dataclass(frozen=True)
class PSWFChannel:
    """A single pseudo-atomic wavefunction channel from ``PP_PSWFC``.

    ``chi`` stores ``chi_l(r) = r * R_l(r)`` on the pseudopotential radial mesh
    ``r`` (Bohr), exactly as UPF stores it. ``label`` is the UPF tag (e.g.
    ``"3D"``), ``angular_l`` the integer angular momentum.
    """

    label: str
    angular_l: int
    occupation: float
    chi: npt.NDArray[np.float64]


@dataclass(frozen=True)
class UPFData:
    """Parsed subset of a UPF pseudopotential relevant to exponent init.

    For all-electron bare-Coulomb UPFs (``is_coulomb_ae=True``) the
    ``channels`` tuple is empty: the file ships only the 1/r local
    potential, no PSWFC. :class:`UPFExponentInit` dispatches on this flag
    and generates the initial primitive ladder even-tempered from scratch
    instead of fitting PSWFC.
    """

    element: str
    z_valence: float
    r_mesh: npt.NDArray[np.float64]
    channels: tuple[PSWFChannel, ...]
    l_max_upf: int
    is_coulomb_ae: bool

    @property
    def angular_ls(self) -> tuple[int, ...]:
        """Distinct angular momenta present among the PSWFC channels (sorted)."""
        return tuple(sorted({c.angular_l for c in self.channels}))


@dataclass(frozen=True)
class PrimitiveShell:
    """An all-primitive fitted shell: ``R_l(r) = sum_p c_p r^l e^{-alpha_p r^2}``.

    ``exponents``/``coefficients`` are the raw primitive parameters (no
    contraction). ``rel_error`` is the radial-L2 relative fit residual and
    ``overlap_cond`` the condition number of the same-l primitive overlap
    matrix (the ill-conditioning proxy).
    """

    angular_l: int
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    rel_error: float
    overlap_cond: float


@dataclass(frozen=True)
class ECPData:
    """Analytic ccECP block extracted from a UPF ``[Gaussinfo]`` section.

    The terms are flattened across channels and ordered local-first then
    ``l = 0, 1, ..., l_max - 1`` (the jQMC / TREXIO convention also used by
    :class:`jqmc.coulomb_potential.Coulomb_potential_data`). ``ang_moms``
    encodes the local channel as ``l_max + 1`` and each non-local channel as
    its ``l``.

    Attributes:
        element: Element symbol parsed from ``PP_HEADER`` (e.g. ``"Cu"``).
        z_core: Number of core electrons removed by the pseudopotential
            (``Z_atomic - Z_valence``).
        max_ang_mom_plus_1: ``l_max + 1`` (the local-channel angular tag).
        ang_moms: Per-term angular momentum (length ``num_ecps``); local terms
            carry ``max_ang_mom_plus_1``, non-local terms carry ``l``.
        exponents: Per-term Gaussian exponent ``alpha_k`` (length ``num_ecps``).
        coefficients: Per-term coefficient ``c_k`` (length ``num_ecps``).
        powers: Per-term radial power ``n_k`` (length ``num_ecps``).
    """

    element: str
    z_core: int
    max_ang_mom_plus_1: int
    ang_moms: tuple[int, ...]
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    powers: tuple[int, ...]

    @property
    def num_ecps(self) -> int:
        """Total number of ECP terms across all channels."""
        return len(self.ang_moms)


# ---------------------------------------------------------------------------
# UPF parsing
# ---------------------------------------------------------------------------
def parse_upf(path: str) -> UPFData:
    """Parse a UPF file (auto-detecting v1 text vs v2 XML layout)."""
    with open(path) as fh:
        text = fh.read()
    # v2 is well-formed XML with a root <UPF version="2.x">; v1 is loose text
    # with bare <PP_*> tags and no XML declaration / root attributes.
    if re.search(r"<UPF\b[^>]*version\s*=\s*[\"']2", text):
        return _parse_upf_v2_xml(text)
    return _parse_upf_v1_text(text)


def parse_upf_element(path: str) -> str:
    """Read *only* the element symbol from a UPF ``PP_HEADER`` (v1 text or v2 XML).

    Element identity on jQMC's QE path comes from the pseudopotential the
    species was actually computed with -- QE's own convention -- rather than
    from parsing the (free-form) species label, which QE lets users tag
    arbitrarily (``O`` vs ``O1``, ``Fe2+`` vs ``Fe3+``) to give distinct
    UPFs / magnetic moments to chemically inequivalent sites. Header-only so
    it stays robust for pseudos whose ``PP_PSWFC`` / mesh blocks a full
    :func:`parse_upf` would reject (e.g. single-basis ``pw`` / ``bspline``
    kinds that never fit the UPF).

    The symbol is normalised to periodic-table casing (``FE`` -> ``Fe``) so
    the downstream ``element -> Z`` lookup is case-robust.
    """
    with open(path) as fh:
        text = fh.read()
    if re.search(r"<UPF\b[^>]*version\s*=\s*[\"']2", text):
        root = ET.fromstring(text)
        header = _find_child(root, "PP_HEADER")
        if header is None:
            raise ValueError(f"UPF {path!r}: missing PP_HEADER; cannot read element symbol.")
        element = (header.get("element") or "").strip()
    else:
        header = _read_between(text, "<PP_HEADER>", "</PP_HEADER>")
        element = header.split("Element", 1)[0].split()[-1]
    if not element:
        raise ValueError(f"UPF {path!r}: PP_HEADER carries no element symbol.")
    return element.capitalize()


def _floats(block: str) -> npt.NDArray[np.float64]:
    return np.array([float(x) for x in block.split()], dtype=np.float64)


def _read_between(text: str, start_tag: str, end_tag: str) -> str:
    return text.split(start_tag, 1)[1].split(end_tag, 1)[0]


def _parse_upf_v1_text(text: str) -> UPFData:
    """Parse the legacy (QE < 5) plain-text UPF layout."""
    header = _read_between(text, "<PP_HEADER>", "</PP_HEADER>")
    element = header.split("Element", 1)[0].split()[-1]
    z_valence = float(re.search(r"([-\d.eE+]+)\s+Z valence", header).group(1))
    # v1 has no XML attributes; bare-Coulomb AE is signalled in the prose
    # ('1/r' as pseudo_type / 'Coulomb' marker). l_max is reported on its
    # own line in the header block.
    is_coulomb_ae = bool(re.search(r"\b1/r\b|Coulomb", header, re.IGNORECASE))
    l_max_match = re.search(r"(-?\d+)\s+Max angular momentum component", header)
    l_max_upf = int(l_max_match.group(1)) if l_max_match else 0

    r = _floats(_read_between(text, "<PP_R>", "</PP_R>"))
    n = r.size

    channels: list[PSWFChannel] = []
    if "<PP_PSWFC>" in text and "</PP_PSWFC>" in text:
        pswfc = _read_between(text, "<PP_PSWFC>", "</PP_PSWFC>")
        # Each channel begins with a header line: 'S    0   2.00   Wavefunction'.
        header_re = re.compile(r"^\s*([SPDFGH])\s+(\d+)\s+([-\d.eE+]+)\s+Wavefunction\s*$", re.M)
        matches = list(header_re.finditer(pswfc))
        for i, h in enumerate(matches):
            letter, l_str, occ_str = h.group(1), h.group(2), h.group(3)
            seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(pswfc)
            vals = _floats(pswfc[h.end() : seg_end])[:n]
            channels.append(
                PSWFChannel(
                    label=f"{letter}",
                    angular_l=int(l_str),
                    occupation=float(occ_str),
                    chi=vals,
                )
            )
    if not channels and not is_coulomb_ae:
        raise ValueError(f"No PP_PSWFC channels found in v1 UPF for {element}.")
    return UPFData(
        element=element,
        z_valence=z_valence,
        r_mesh=r,
        channels=tuple(channels),
        l_max_upf=l_max_upf,
        is_coulomb_ae=is_coulomb_ae,
    )


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem.iter():
        if _strip_ns(child.tag) == name:
            return child
    return None


def _parse_upf_v2_xml(text: str) -> UPFData:
    """Parse the QE >= 5 XML UPF layout (``<UPF version="2.0.1">``)."""
    root = ET.fromstring(text)
    header = _find_child(root, "PP_HEADER")
    if header is None:
        raise ValueError("UPF v2: missing PP_HEADER.")
    element = (header.get("element") or "").strip()
    z_valence = float(header.get("z_valence", "nan"))
    is_coulomb_ae = (header.get("is_coulomb") or "").strip().lower() == "true"
    try:
        l_max_upf = int(header.get("l_max", "0") or 0)
    except ValueError:
        l_max_upf = 0

    mesh = _find_child(root, "PP_R")
    if mesh is None:
        raise ValueError("UPF v2: missing PP_R mesh.")
    r = _floats(mesh.text or "")
    n = r.size

    pswfc = _find_child(root, "PP_PSWFC")
    if pswfc is None and not is_coulomb_ae:
        raise ValueError(f"UPF v2: missing PP_PSWFC for {element}.")
    channels: list[PSWFChannel] = []
    if pswfc is not None:
        for child in pswfc:
            name = _strip_ns(child.tag)
            if not name.startswith("PP_CHI"):
                continue
            l = int(child.get("l"))
            label = (child.get("label") or _LETTER_FROM_L.get(l, str(l))).strip()
            occ = float(child.get("occupation", "0") or 0.0)
            vals = _floats(child.text or "")[:n]
            channels.append(PSWFChannel(label=label, angular_l=l, occupation=occ, chi=vals))
    if not channels and not is_coulomb_ae:
        raise ValueError(f"UPF v2: no PP_CHI channels for {element}.")
    return UPFData(
        element=element,
        z_valence=z_valence,
        r_mesh=r,
        channels=tuple(channels),
        l_max_upf=l_max_upf,
        is_coulomb_ae=is_coulomb_ae,
    )


# ---------------------------------------------------------------------------
# Analytic ccECP extraction from the UPF [Gaussinfo] block
# ---------------------------------------------------------------------------
def _parse_gaussinfo_body(body: str) -> tuple[int, list[list[tuple[int, float, float]]]]:
    """Parse a ``[Gaussinfo]`` body into ``(z_val, channels)``.

    Format::

        Z_val   N_chan                  # N_chan = l_max + 1
        n_0  n_1  ...  n_{N_chan - 1}   # terms per channel
        n_pow  alpha  coef              # one row per term
        ...

    Channel order is ``l = 0, 1, ..., l_max - 1, local`` (local last). Returns
    the channel list in that same order; the caller re-tags to TREXIO's
    local-first convention.
    """
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"[Gaussinfo]: need at least 2 header lines, got {len(lines)}.")
    head = lines[0].split()
    z_val = int(round(float(head[0])))
    n_chan = int(head[1])
    term_counts = [int(x) for x in lines[1].split()]
    if len(term_counts) != n_chan:
        raise ValueError(
            f"[Gaussinfo]: header says {n_chan} channels but term-count line has {len(term_counts)} entries ({lines[1]!r})."
        )
    pos = 2
    channels: list[list[tuple[int, float, float]]] = []
    for nc in term_counts:
        ch: list[tuple[int, float, float]] = []
        for _ in range(nc):
            if pos >= len(lines):
                raise ValueError(f"[Gaussinfo]: truncated; expected {sum(term_counts)} term rows.")
            tok = lines[pos].split()
            if len(tok) < 3:
                raise ValueError(f"[Gaussinfo]: malformed term row {lines[pos]!r}.")
            n_pow, alpha, coef = int(tok[0]), float(tok[1]), float(tok[2])
            ch.append((n_pow, alpha, coef))
            pos += 1
        channels.append(ch)
    return z_val, channels


def parse_gaussinfo(upf_path: str) -> ECPData:
    """Parse the analytic ccECP from a ccECP UPF's ``[Gaussinfo]`` block.

    Reads ``PP_HEADER`` for the element and ``Z_valence`` (used to compute
    ``z_core = Z_atomic - Z_valence``), then parses the ``[Gaussinfo]`` body
    and re-tags the channels to the jQMC / TREXIO local-first convention
    consumed by :class:`jqmc.coulomb_potential.Coulomb_potential_data`.

    Args:
        upf_path: Path to a ccECP UPF file.

    Returns:
        ECPData: ``ang_moms`` use ``l_max + 1`` for the local channel and the
        bare ``l`` for non-local channels; terms are emitted local-first then
        ``l = 0, 1, ..., l_max - 1``.

    Raises:
        ValueError: If the UPF has no ``[Gaussinfo]`` section. The QE -> jqmc
            converter derives the analytic ccECP (local + non-local terms)
            from this block; standard UPF files (ONCV / GBRV / PAW / SSSP)
            do not include it. Use a ccECP UPF (e.g. from
            pseudopotentiallibrary.org) or regenerate the pseudopotential with
            Opium using the ``rpt`` flag (``opium <elem> ... upf rpt``).
    """
    from .trexio_wrapper import _convert_from_atomic_labels_to_atomic_numbers

    with open(upf_path) as fh:
        text = fh.read()

    # Gaussinfo lives inside the human-readable PP_INFO header and ends at
    # the next bracket-headed section or the PP_INFO closing tag.
    gaussinfo_match = re.search(
        r"\[Gaussinfo\]\s*\n(.*?)(?=\n\s*\[|\n\s*</PP_INFO>)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    body = gaussinfo_match.group(1) if gaussinfo_match else None
    if body is None:
        raise ValueError(
            f"{upf_path!r}: no [Gaussinfo] section. The QE -> jqmc converter "
            "extracts the analytic ccECP (local + non-local terms) from the "
            "[Gaussinfo] block embedded in ccECP UPF files. Standard UPFs "
            "(ONCV / GBRV / PAW / SSSP) do not include this block. Use a "
            "ccECP UPF (e.g. from pseudopotentiallibrary.org) or regenerate "
            "the pseudopotential with Opium using the 'rpt' flag (e.g. "
            "'opium <elem> <log> ae ps nl tc2 ke upf rpt')."
        )

    upf = parse_upf(upf_path)
    z_atomic = int(_convert_from_atomic_labels_to_atomic_numbers([upf.element])[0])
    z_core = int(round(float(z_atomic) - float(upf.z_valence)))

    _z_val_gauss, channels = _parse_gaussinfo_body(body)
    if not channels:
        raise ValueError(f"{upf_path!r}: [Gaussinfo] declares zero channels.")

    local_terms = channels[-1]
    nonlocal_channels = channels[:-1]  # ordered l = 0, 1, ..., l_max
    if not nonlocal_channels:
        raise ValueError(
            f"{upf_path!r}: [Gaussinfo] declares a local channel with no non-local "
            "channels. A pure-local ECP has no angular projector to tag, so the "
            "local-first TREXIO layout is undefined here."
        )
    # ``l_max`` is the highest angular momentum among the NON-LOCAL projectors, so
    # it counts ``nonlocal_channels``: the local channel is the term the others are
    # projected against, not an l value of its own. TREXIO then tags the local
    # channel ``l_max + 1``. Counting ``channels`` would shift every tag by one --
    # ccECP H/Li/O have a single non-local channel (l=0) plus the local one, so the
    # tags would be max_ang_mom_plus_1 = 2 with ang_moms = (2, 2, 2, 0) where
    # the spec -- and PySCF's dump of the same system -- say 1 and (1, 1, 1, 0).
    # jQMC itself would not notice (local/non-local split compares ang_mom against
    # max_ang_mom_plus_1, and both shift together, so the extra empty l=1 channel
    # contributes exactly 0.0; e_L identical to 1e-16), but a TREXIO file carrying such tags makes any
    # spec-conforming reader lose the local channel.
    l_max = len(nonlocal_channels) - 1

    ang_moms: list[int] = []
    exponents: list[float] = []
    coefficients: list[float] = []
    powers: list[int] = []

    def _emit(tag: int, terms: list[tuple[int, float, float]]) -> None:
        for n_pow, alpha, coef in terms:
            ang_moms.append(tag)
            powers.append(int(n_pow))
            exponents.append(float(alpha))
            coefficients.append(float(coef))

    _emit(l_max + 1, local_terms)
    for l, ch in enumerate(nonlocal_channels):
        _emit(l, ch)

    # Every declared non-local channel must actually carry terms, and the local
    # tag must be exactly one above the highest of them. Violating either is
    # SILENT downstream: Coulomb_potential_data pads a missing l with a
    # zero-coefficient dummy (see its `_*_non_local_part_with_global_lmax_plus1`
    # padding), which is legitimate when a light element sits next to a heavier
    # one, but which also swallows a converter that over-declares l_max. That is
    # how the off-by-one below went unnoticed until a spec-conforming external
    # reader (CHAMP) lost the local channel. Fail here, at the source.
    declared = {int(l) for l in range(l_max + 1)}
    present = {int(a) for a in ang_moms if int(a) != l_max + 1}
    if present != declared:
        raise ValueError(
            f"{upf_path!r}: inconsistent ccECP channels. Declared non-local "
            f"l = {sorted(declared)} (l_max = {l_max}) but the [Gaussinfo] terms "
            f"carry l = {sorted(present)}. Every declared l must have terms; "
            "l_max counts the NON-LOCAL channels only, and the local channel is "
            f"tagged l_max + 1 = {l_max + 1}."
        )
    if any(int(a) > l_max + 1 for a in ang_moms):
        raise ValueError(
            f"{upf_path!r}: ang_mom exceeds the local tag l_max + 1 = {l_max + 1}: {sorted({int(a) for a in ang_moms})}."
        )

    return ECPData(
        element=upf.element,
        z_core=z_core,
        max_ang_mom_plus_1=l_max + 1,
        ang_moms=tuple(ang_moms),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        powers=tuple(powers),
    )


# ---------------------------------------------------------------------------
# Primitive-GTO fit of a PSWFC channel
# ---------------------------------------------------------------------------
def _radial_quadrature_weight(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Radial L2 weight ``w = r^2 dr`` on a (generally logarithmic) UPF mesh.

    ``int_0^inf |R(r)|^2 r^2 dr`` is the physical norm; ``dr`` is approximated
    by ``np.gradient(r)`` which is exact to second order on smooth meshes.
    """
    dr = np.gradient(r)
    return np.clip((r**2) * dr, 0.0, None)


def fit_pswfc_channel(
    r: npt.NDArray[np.float64],
    chi: npt.NDArray[np.float64],
    angular_l: int,
    n_gto: int,
    alpha_lo: float = 0.1,
    alpha_hi: float = 30.0,
    alpha_ratio_min: float = 1.5,
    max_nfev: int = 4000,
) -> PrimitiveShell:
    """Fit ``R_l(r) = chi/r`` by ``sum_p c_p r^l exp(-alpha_p r^2)``.

    The log-exponents are optimised by Levenberg--Marquardt-style least squares
    (scipy ``trf``); the linear coefficients ``c_p`` are solved in closed form
    at each step. A soft penalty enforces a minimum exponent ratio
    ``alpha_ratio_min`` between neighbours to prevent exponent collapse (which
    would ill-condition the primitive overlap).

    Args:
        r: Radial mesh (Bohr), shape ``(n,)``.
        chi: ``chi_l(r) = r R_l(r)`` from the UPF channel, shape ``(n,)``.
        angular_l: Angular momentum ``l`` of the channel.
        n_gto: Number of primitive Gaussians.
        alpha_lo: Smallest exponent (Bohr^-2) of the initial even-tempered
            (geometric) exponent guess.
        alpha_hi: Largest exponent (Bohr^-2) of the initial even-tempered
            exponent guess.
        alpha_ratio_min: Minimum neighbouring-exponent ratio (collapse guard).
        max_nfev: Max least-squares function evaluations.

    Returns:
        PrimitiveShell with exponents sorted ascending.
    """
    r = np.asarray(r, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    R = np.zeros_like(chi)
    nz = r > 1e-8
    R[nz] = chi[nz] / r[nz]

    wq = _radial_quadrature_weight(r)
    sqrt_w = np.sqrt(wq)

    log_a0 = np.log(np.geomspace(alpha_lo, alpha_hi, n_gto))

    def _design(
        log_a: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alphas = np.exp(log_a)
        basis = (r[:, None] ** angular_l) * np.exp(-alphas[None, :] * (r[:, None] ** 2))
        return basis, alphas

    def _solve_c(
        log_a: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        basis, alphas = _design(log_a)
        a_mat = basis * sqrt_w[:, None]
        b_vec = R * sqrt_w
        c, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        return c, basis, alphas

    def _residual(log_a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        c, basis, _ = _solve_c(log_a)
        fit_err = (basis @ c - R) * sqrt_w
        gaps = np.diff(np.sort(log_a))
        penalty = np.clip(np.log(alpha_ratio_min) - gaps, 0.0, None) * 5.0
        return np.concatenate([fit_err, penalty])

    sol = least_squares(_residual, log_a0, method="trf", max_nfev=max_nfev)
    # A fit least_squares does not report as a success is never used (issue
    # 917), including the max_nfev limit (status 0): raise with its status.
    if not sol.success:
        raise PSWFCFitError(
            f"PSWFC Gaussian fit (l={angular_l}, n_gto={n_gto}) failed: least_squares status={int(sol.status)}, "
            f"nfev={int(sol.nfev)}: {sol.message}"
        )
    c, basis, alphas = _solve_c(sol.x)

    num = float(np.sqrt(np.sum(((basis @ c - R) ** 2) * wq)))
    den = float(np.sqrt(np.sum((R**2) * wq)))
    rel = num / den if den > 0 else float("nan")

    # Same-l primitive overlap condition number (constants cancel in cond):
    #   S_pq propto (alpha_p + alpha_q)^{-(l + 3/2)}.
    a_sum = alphas[:, None] + alphas[None, :]
    overlap = a_sum ** (-(angular_l + 1.5))
    cond = float(np.linalg.cond(overlap))

    order = np.argsort(alphas)
    return PrimitiveShell(
        angular_l=angular_l,
        exponents=tuple(float(a) for a in alphas[order]),
        coefficients=tuple(float(x) for x in c[order]),
        rel_error=rel,
        overlap_cond=cond,
    )


def fit_pswfc_channel_auto(
    r: npt.NDArray[np.float64],
    chi: npt.NDArray[np.float64],
    angular_l: int,
    rel_target: float = 1.0e-2,
    cond_max: float = 1.0e6,
    n_min: int = 2,
    n_max: int = 6,
    **fit_kwargs: float | int,
) -> PrimitiveShell:
    """Auto-select the smallest ``n_gto`` meeting ``rel_target`` and ``cond_max``.

    Sweeps ``n_gto = n_min..n_max`` and returns the first shell whose
    ``rel_error <= rel_target`` and ``overlap_cond <= cond_max``. If none
    qualifies, returns the best ``rel_error`` shell that stays under
    ``cond_max`` (falling back to the smallest-cond shell if all exceed it).

    This removes ``n_gto`` from the set of system-dependent knobs: the number
    of primitives per channel is decided by the fidelity / conditioning
    thresholds, not hardcoded.
    """
    candidates: list[PrimitiveShell] = []
    for n_gto in range(n_min, n_max + 1):
        shell = fit_pswfc_channel(r, chi, angular_l, n_gto, **fit_kwargs)
        candidates.append(shell)
        if shell.rel_error <= rel_target and shell.overlap_cond <= cond_max:
            return shell
    well_conditioned = [s for s in candidates if s.overlap_cond <= cond_max]
    if well_conditioned:
        return min(well_conditioned, key=lambda s: s.rel_error)
    return min(candidates, key=lambda s: s.overlap_cond)


# ---------------------------------------------------------------------------
# Tight-primitive extension (always-on, applied by UPFExponentInit)
# ---------------------------------------------------------------------------
# A UPF-seeded basis follows the PSWFC channels of the pseudopotential and
# typically has alpha_max ~ a few -- giving a Fourier reach of only
# |G| ~ sqrt(2 alpha_max) Bohr^-1, well below the QE wfc cutoff. Adding
# tight primitives via an even-tempered geometric extension at the tight
# end of each shell extends the basis into the high-G regime so the
# downstream projection (and any VMC consumer) can reach what the PSWFC
# fit alone cannot. Knobs:
#   - g_target (Bohr^-1) -- required by UPFExponentInit; production
#     callers derive from ``sqrt(2 * ref.ecutwfc)``
#   - PGTO_TIGHT_RATIO (even-tempered ratio; jqmc/_setting.py)
#   - PGTO_TIGHT_MAX_PER_SHELL (cap to prevent basis blowup; same)
#   - PW_PGTO_TAIL_TOL (alpha-vs-G coupling, shared with alpha_min_from_gcut)
# No cond(S) gate: with max_per_shell=3 and ratio=2.5 the alpha range only
# widens by ~15x, so the worst-case observed shell cond stays well under
# the projection's svd_rcond=1e-8 truncation threshold. A gate would only
# silently truncate extension without giving the caller actionable signal.
# FFT-grid resolvability is checked at projection time inside
# ``optimize_mixed_basis``; no constant for it on purpose.


def _gaussian_radial_overlap_cond(
    exponents: tuple[float, ...] | npt.NDArray[np.float64],
    angular_l: int,
) -> float:
    """Condition number of the same-l radial Gaussian overlap matrix.

    Uses the closed-form ``S_pq propto (alpha_p + alpha_q)^{-(l + 3/2)}``
    (constants cancel in ``cond``). Same convention as
    :func:`fit_pswfc_channel` so the returned value is comparable to the
    ``overlap_cond`` field of un-extended shells. O(N^2) for N exponents
    plus a 1-time eigh on the small NxN overlap; trivial.
    """
    alphas = np.asarray(exponents, dtype=np.float64)
    if alphas.size == 0:
        return float("nan")
    a_sum = alphas[:, None] + alphas[None, :]
    overlap = a_sum ** (-(angular_l + 1.5))
    return float(np.linalg.cond(overlap))


def _extend_shells_tight(
    shells: list[PrimitiveShell],
    *,
    G_target: float,
    tail_tol: float = PW_PGTO_TAIL_TOL,
    ratio: float = PGTO_TIGHT_RATIO,
    max_per_shell: int = PGTO_TIGHT_MAX_PER_SHELL,
) -> list[PrimitiveShell]:
    """Even-tempered tight-primitive extension of every shell.

    For each shell, append primitives at ``alpha_max * ratio^k`` (k=1, 2, ...)
    until either stop condition fires:

      (a) new alpha exceeds ``alpha_target = G_target^2 / (4 |ln(tail_tol)|)``
          (G-space reach achieved -- inverse of :func:`alpha_min_from_gcut`);
      (b) ``max_per_shell`` primitives have been added (basis-blowup guard).

    FFT-grid resolvability is NOT checked here (the helper has no access
    to the runtime FFT grid). That check lives in
    :func:`jqmc.pw_pgto_projection.optimize_mixed_basis`, which warns
    when any primitive's alpha exceeds ``alpha_max_safe = 1/(2 dx^2)``
    for the cell + FFT grid actually in use.

    Returns a NEW list of :class:`PrimitiveShell` with extended exponents,
    unit-1.0 coefficients for the appended primitives (all-primitive
    convention -- the projection layer picks the optimal linear
    combination), and the ``overlap_cond`` field RECOMPUTED on the
    extended exponent set so downstream diagnostics reflect the actual
    basis. ``rel_error`` is kept as-is (it describes the original PSWFC
    fit and is meaningless on the extended set; see docstring of
    :class:`PrimitiveShell`).

    The extension is unconditional in :class:`UPFExponentInit`; this
    helper is exposed for callers (tests, debugging, alternative
    workflows) that want to apply it explicitly.
    """
    if tail_tol <= 0.0 or tail_tol >= 1.0:
        raise ValueError(f"tail_tol must be in (0, 1), got {tail_tol}.")
    if ratio <= 1.0:
        raise ValueError(f"ratio must be > 1, got {ratio}.")
    alpha_target = float(G_target) ** 2 / (4.0 * abs(np.log(tail_tol)))

    out: list[PrimitiveShell] = []
    for sh in shells:
        if not sh.exponents:
            out.append(sh)
            continue
        alpha_max_existing = max(sh.exponents)
        new_alphas: list[float] = []
        a = alpha_max_existing
        for _ in range(int(max_per_shell)):
            a = a * float(ratio)
            if a > alpha_target:
                break
            new_alphas.append(a)
        if not new_alphas:
            out.append(sh)
            continue
        new_exps = tuple(sh.exponents) + tuple(new_alphas)
        new_coefs = tuple(sh.coefficients) + tuple(1.0 for _ in new_alphas)
        new_cond = _gaussian_radial_overlap_cond(new_exps, sh.angular_l)
        out.append(
            PrimitiveShell(
                angular_l=sh.angular_l,
                exponents=new_exps,
                coefficients=new_coefs,
                rel_error=sh.rel_error,
                overlap_cond=new_cond,
            )
        )
        logger.info(
            "extend_shells_tight: l=%d alpha_max %.4g -> %.4g (added %d "
            "primitive(s) via ratio=%.2f, alpha_target=%.4g); new overlap_cond=%.2e.",
            sh.angular_l,
            alpha_max_existing,
            max(new_exps),
            len(new_alphas),
            ratio,
            alpha_target,
            new_cond,
        )
    return out


# ---------------------------------------------------------------------------
# All-electron bare-Coulomb seeding (no PSWFC to fit)
# ---------------------------------------------------------------------------
def generate_ae_even_tempered_shells(
    *,
    g_target: float,
    upf_lmax: int,
    atomic_number: int,
    cell_min_edge: float = 0.0,
    cell_safety: float = AE_PGTO_CELL_SAFETY,
) -> list[PrimitiveShell]:
    """Hardcoded chemistry-grade pGTO basis for bare-Coulomb AE UPFs.

    The only bare-Coulomb UPF (no PSWFC channels, no projectors, just the
    ``1/r`` local potential) that anyone converts in practice is
    ``H.coulomb-ae.UPF``. Every other AE / ECP UPF ships PSWFC channels
    and goes through the regular ``fit_pswfc_channel_auto`` path, which
    derives orbital-matched exponents from ``chi(r)`` directly.

    A generic Z-scaled tempered ladder does not work on the auto-LS side:
    a chemistry-grade basis needs the natural valence scale, the cusp
    tightness, and the diffuse tail to be tuned per element independently,
    and no single formula reproduces all three. The downstream
    ``optimize_mixed_basis`` prunes most of such a ladder, leaving 2
    surviving s primitives even when the call site asks for 3s + 1p.

    The H exponents are therefore hardcoded to the cc-pVDZ uncontracted
    choice (``s: 13.01, 1.962, 0.4446 ; p: 0.7270``), and any other atomic
    number raises ``NotImplementedError``. A non-H bare-Coulomb UPF needs
    its own hardcoded chemistry-grade exponents here, not a generic ladder.

    The cell-aware floor still applies as a
    diagnostic: if the smallest hardcoded exponent's e-fold exceeds
    ``cell_min_edge / 2``, a WARNING is emitted (the basis is not
    trimmed -- the basis is the basis). For H4 in a 5x6x7 Bohr cell the
    smallest s exponent ``0.4446`` corresponds to e-fold ``1.5 Bohr``,
    well within ``cell_min / 2 = 2.5 Bohr``, so the warning never fires
    for any realistic H workflow.

    Args:
        g_target: Unused. Kept for API compatibility with the caller
            (``UPFExponentInit.shells_for_element``), which derives it
            from QE's ``ecutwfc``.
        upf_lmax: Unused. Kept for API compatibility. The hardcoded basis
            always emits s + p (3 s primitives + 1 p polarisation
            primitive); H bare-Coulomb has ``upf_lmax = 0`` so the p
            shell is the standard single polarisation above.
        atomic_number: Element Z. Must be 1 (H). Any other value raises
            ``NotImplementedError``.
        cell_min_edge: Shortest lattice-vector length (Bohr). When
            ``> 0`` a warning is emitted if the smallest hardcoded
            exponent's e-fold exceeds ``cell_safety / cell_min_edge^2``.
        cell_safety: Threshold multiplier for the warning. Default 4
            corresponds to a one-image self-overlap of ~ 3.7% at the
            threshold.

    Returns:
        Two ``PrimitiveShell`` entries: ``s`` (3 exponents) and ``p``
        (1 exponent). ``rel_error`` is ``nan`` (no fit was performed);
        ``overlap_cond`` is computed on each shell for diagnostics.

    Raises:
        NotImplementedError: When ``atomic_number != 1``. The bare-
            Coulomb path is reserved for H; other elements should reach
            this code only through a UPF parser bug.
    """
    if atomic_number != 1:
        raise NotImplementedError(
            f"generate_ae_even_tempered_shells: only H (Z=1) is supported on the "
            f"bare-Coulomb AE path; got atomic_number={atomic_number}. Other "
            f"elements ship UPFs with PSWFC channels and should go through the "
            f"standard fit_pswfc_channel_auto path instead. If you genuinely "
            f"need a bare-Coulomb basis for element Z={atomic_number}, add the "
            f"corresponding hardcoded chemistry-grade exponents to "
            f"_AE_*_S_EXPONENTS / _AE_*_P_EXPONENT in upf_wrapper.py."
        )
    if cell_min_edge < 0.0:
        raise ValueError(f"cell_min_edge must be non-negative, got {cell_min_edge}.")
    if cell_safety <= 0.0:
        raise ValueError(f"cell_safety must be positive, got {cell_safety}.")

    s_exps = _AE_H_S_EXPONENTS
    p_exp = _AE_H_P_EXPONENT

    # Cell-aware safety warning. The hardcoded smallest exponent
    # (0.4446 -> e-fold 1.5 Bohr) is well within any reasonable cell;
    # this warning only fires for pathologically tight cells (< ~ 3 Bohr).
    if cell_min_edge > 0.0:
        alpha_floor_warn = float(cell_safety) / (float(cell_min_edge) ** 2)
        smallest = min(s_exps)
        if smallest < alpha_floor_warn:
            cross_cell_overlap = 2.0 * float(np.exp(-smallest * float(cell_min_edge) ** 2))
            logger.warning(
                "generate_ae_even_tempered_shells: smallest H s exponent "
                "(%.4f) is below the cell-wrapping safety floor "
                "(%.4f = %.1f / %.2f^2). Estimated cross-cell self-overlap "
                "%.1f%%; pgto LS conditioning may degrade. Either raise "
                "cell_min_edge or accept the loss.",
                smallest,
                alpha_floor_warn,
                float(cell_safety),
                float(cell_min_edge),
                100.0 * cross_cell_overlap,
            )

    shells = [
        PrimitiveShell(
            angular_l=0,
            exponents=s_exps,
            coefficients=(1.0,) * len(s_exps),
            rel_error=float("nan"),
            overlap_cond=_gaussian_radial_overlap_cond(s_exps, 0),
        ),
        PrimitiveShell(
            angular_l=1,
            exponents=(p_exp,),
            coefficients=(1.0,),
            rel_error=float("nan"),
            overlap_cond=_gaussian_radial_overlap_cond((p_exp,), 1),
        ),
    ]

    logger.info(
        "generate_ae_even_tempered_shells: hardcoded H (cc-pVDZ-derived) shells -> s%s, p(%.4f)",
        list(s_exps),
        p_exp,
    )

    return shells


# ---------------------------------------------------------------------------
# Pluggable exponent-initialisation strategy
# ---------------------------------------------------------------------------
class ExponentInitStrategy(ABC):
    """Produce the initial all-primitive shells for a given element.

    Implementations seed the variational pGTO block. The exponents are refined
    downstream against the solid Kohn--Sham orbitals, so a strategy only needs
    to supply physically reasonable, well-conditioned starting ladders and the
    correct angular channels.
    """

    @abstractmethod
    def shells_for_element(self, element: str) -> list[PrimitiveShell]:
        """Return the primitive shells (one per UPF angular channel) for ``element``."""
        raise NotImplementedError


class UPFExponentInit(ExponentInitStrategy):
    """UPF ``PP_PSWFC`` -> primitive-GTO fit strategy with tight extension.

    Each returned shell carries both (a) the primitives fitted to the UPF
    PSWFC channel (alpha range typical of pseudo-atomic wavefunctions:
    ~0.01 to ~5) and (b) a small tail of TIGHT primitives appended via
    :func:`_extend_shells_tight` (even-tempered ratio
    ``PGTO_TIGHT_RATIO`` until the ``g_target`` Fourier reach is
    achieved). The extension is **always on** because the PSWFC-only
    basis cannot reach the high-G content the QE wfc carries (typical
    ecutwfc_QE ~ 100 Ry has g_max ~ 17 Bohr^-1, while a PSWFC-only
    basis tops out at g_reach ~ 10). All pGTO-bearing conversion kinds
    (``pgto``, ``pgto+pw``, ``pgto+bspline``) inherit the wider basis
    transparently.

    Args:
        upf_paths: Mapping ``element -> .upf path``. Element keys are matched
            case-insensitively against the UPF ``PP_HEADER`` element.
        rel_target: Relative fit-error target per PSWFC channel, forwarded
            to :func:`fit_pswfc_channel_auto`. Default ``1e-2``.
        cond_max: Maximum primitive-overlap condition number per channel,
            forwarded to :func:`fit_pswfc_channel_auto`. Default ``1e6``.
        n_min: Smallest number of primitives tried per channel, forwarded
            to :func:`fit_pswfc_channel_auto`. Default ``2``.
        n_max: Largest number of primitives tried per channel, forwarded
            to :func:`fit_pswfc_channel_auto`. Default ``6``.
        g_target: Target Fourier reach (Bohr^-1) for the tight extension.
            **Required.** Physically right value is the QE reference's
            ``g_max_QE = sqrt(2 ecutwfc_Ha)`` -- pGTO need not reach past
            what psi itself contains. CLI paths in ``jqmc_tool`` compute
            this from ``ref.ecutwfc`` automatically. There is no
            system-blind default because a hardcoded value would either
            over-extend (wasting basis for low-ecutwfc systems) or
            under-extend (missing high-G for high-ecutwfc / AE systems).
        cell_min_edge: Shortest lattice-vector length (Bohr), forwarded to
            :func:`generate_ae_even_tempered_shells` for bare-Coulomb
            all-electron UPFs, where a value ``> 0`` enables the cell-aware
            diffuse-exponent diagnostic. Ignored by the PSWFC-fit path.
            Must be non-negative. Default ``0.0`` (disabled).
    """

    def __init__(
        self,
        upf_paths: dict[str, str],
        *,
        g_target: float,
        rel_target: float = 1.0e-2,
        cond_max: float = 1.0e6,
        n_min: int = 2,
        n_max: int = 6,
        cell_min_edge: float = 0.0,
    ) -> None:
        if g_target <= 0.0:
            raise ValueError(f"g_target must be positive (Bohr^-1), got {g_target}.")
        if cell_min_edge < 0.0:
            raise ValueError(f"cell_min_edge must be non-negative, got {cell_min_edge}.")
        self._upf_paths = {k.capitalize(): v for k, v in upf_paths.items()}
        self._rel_target = rel_target
        self._cond_max = cond_max
        self._n_min = n_min
        self._n_max = n_max
        self._g_target = float(g_target)
        # ``cell_min_edge > 0`` activates the cell-aware diffuse floor in
        # the AE bare-Coulomb ladder generator (see
        # :func:`generate_ae_even_tempered_shells`). The PSWFC-fit path
        # (non-bare-Coulomb UPFs) ignores this field -- fitted exponents
        # already track the PSWFC shape and don't need the floor.
        self._cell_min_edge = float(cell_min_edge)
        self._cache: dict[str, list[PrimitiveShell]] = {}
        self._upf_cache: dict[str, UPFData] = {}

    def _load_upf(self, element: str) -> UPFData:
        key = element.capitalize()
        if key in self._upf_cache:
            return self._upf_cache[key]
        if key not in self._upf_paths:
            raise KeyError(f"No UPF path registered for element {element!r}.")
        upf = parse_upf(self._upf_paths[key])
        self._upf_cache[key] = upf
        return upf

    def z_valence_for_element(self, element: str) -> float:
        """Per-element valence-electron count (``Z_valence``) parsed from UPF."""
        return float(self._load_upf(element).z_valence)

    def is_coulomb_ae_for_element(self, element: str) -> bool:
        """Return whether an element's UPF is a bare-Coulomb AE potential."""
        return bool(self._load_upf(element).is_coulomb_ae)

    def ecp_for_element(self, element: str) -> ECPData:
        """Analytic ccECP parsed from the UPF's ``[Gaussinfo]`` block."""
        key = element.capitalize()
        if key not in self._upf_paths:
            raise KeyError(f"No UPF path registered for element {element!r}.")
        return parse_gaussinfo(self._upf_paths[key])

    def shells_for_element(self, element: str) -> list[PrimitiveShell]:
        key = element.capitalize()
        if key in self._cache:
            return self._cache[key]
        upf = self._load_upf(element)
        # Bare-Coulomb all-electron UPFs (e.g. H.coulomb-ae.UPF) carry no
        # PSWFC chi(r) to fit -- the file is purely the 1/r local potential
        # plus a radial mesh. Switch to an even-tempered ladder built from
        # scratch using g_target and the UPF-advertised l_max + 1.
        if upf.is_coulomb_ae:
            # The AE bare-Coulomb ladder is a hardcoded cc-pVDZ-derived H basis,
            # so the atomic number is needed only to enforce that: Z must be 1,
            # and ``generate_ae_even_tempered_shells`` raises
            # ``NotImplementedError`` for anything else. Z does not scale the
            # exponents (there is no generic Z^2/2 hydrogenic rule).
            # Resolve Z from the UPF element label via the trexio_wrapper helper
            # also used by ``parse_gaussinfo`` above.
            from .trexio_wrapper import _convert_from_atomic_labels_to_atomic_numbers

            atomic_number = int(_convert_from_atomic_labels_to_atomic_numbers([upf.element])[0])
            shells = generate_ae_even_tempered_shells(
                g_target=self._g_target,
                upf_lmax=upf.l_max_upf,
                atomic_number=atomic_number,
                cell_min_edge=self._cell_min_edge,
            )
            self._cache[key] = shells
            return shells
        fitted = [
            fit_pswfc_channel_auto(
                upf.r_mesh,
                ch.chi,
                ch.angular_l,
                rel_target=self._rel_target,
                cond_max=self._cond_max,
                n_min=self._n_min,
                n_max=self._n_max,
            )
            for ch in upf.channels
        ]
        # Always-on tight-primitive extension. g_target comes from the
        # constructor (required, typically derived from the QE
        # reference's ecutwfc by the caller). The helper recomputes
        # overlap_cond on the extended primitive set so the diagnostic
        # stays truthful.
        shells = _extend_shells_tight(fitted, G_target=self._g_target)
        self._cache[key] = shells
        return shells


class EvenTemperedExponentInit(ExponentInitStrategy):
    """Manual even-tempered override (no pseudopotential / target needed).

    A geometric exponent ladder ``geomspace(alpha_min, alpha_max, n)`` is shared
    across angular channels ``l = 0..l_max``, with unit coefficients. Intended
    only as an explicit manual override / debugging fallback --- the UPF and
    (future) PW-driven strategies supply system-matched ladders instead.
    """

    def __init__(self, alpha_min: float, alpha_max: float, n: int, l_max: int) -> None:
        if n < 1:
            raise ValueError("EvenTemperedExponentInit requires n >= 1.")
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._n = n
        self._l_max = l_max

    def shells_for_element(self, element: str) -> list[PrimitiveShell]:
        if self._n == 1:
            exps = (float(self._alpha_max),)
        else:
            exps = tuple(float(a) for a in np.geomspace(self._alpha_min, self._alpha_max, self._n))
        coeffs = (1.0,) * len(exps)
        return [
            PrimitiveShell(
                angular_l=l,
                exponents=exps,
                coefficients=coeffs,
                rel_error=float("nan"),
                overlap_cond=float("nan"),
            )
            for l in range(self._l_max + 1)
        ]
