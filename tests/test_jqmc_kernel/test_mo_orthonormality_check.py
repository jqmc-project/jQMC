"""MO orthonormality warning in ``Hamiltonian_data.sanity_check``."""

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

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL, MO_ORTHONORMALITY_WARNING_THRESHOLD
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_cart_gto_data,
    AOs_composite_data,
    AOs_sphe_gto_data,
    _apply_image_expansion,
    _promote_pgto_to_complex,
    compute_overlap_matrix,
)
from jqmc.determinant import AGP_data, SD_data
from jqmc.hamiltonians import (
    Hamiltonian_data,
    _compute_mo_gram_matrix,
    _compute_mo_orthonormality_metrics,
    _warn_mo_orthonormality,
)
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data, compute_MOs
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
_WARNING_PREFIX = "MOs are not orthonormal"

# Molecular (cart, sphe), periodic Gamma and periodic twist fixtures; the
# H4 / N2 fixtures carry more than one occupied MO per spin, so the
# off-diagonal (orthogonality) metric is live.
_ORTHONORMAL_FIXTURES = [
    "H2_ae_ccpvdz_sphe.h5",
    "N2_ecp_ccpvtz_cart.h5",
    "H4_pbc_ae_cart.h5",
    "H4_pbc_ae_k025_cart.h5",
    "lithium_k025.h5",
]
_MULTI_MO_FIXTURES = ["N2_ecp_ccpvtz_cart.h5", "H4_pbc_ae_cart.h5", "H4_pbc_ae_k025_cart.h5"]


def _load(name):
    """``(structure, geminal, coulomb)`` of a committed TREXIO fixture."""
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(str(_TREXIO_DIR / name), store_tuple=True)
    return structure, geminal, coulomb


def _hamiltonian(structure, geminal, coulomb):
    """Validated Hamiltonian through the blessed factory (runs ``sanity_check``)."""
    return Hamiltonian_data.build(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )


def _with_up_coefficients(geminal, coefficients, *, time_reversal=None):
    """Same SD with replaced up-spin rows.

    By default the time-reversed down channel is re-derived from the new rows.
    With ``time_reversal=False`` the down channel is an independent copy of the
    new rows (needed for a NaN row, which the time-reversal identity rejects).
    """
    mos_up = geminal.orb_data_up_spin.replace(mo_coefficients=np.asarray(coefficients))
    mos_dn = None if time_reversal is None else geminal.orb_data_dn_spin.replace(mo_coefficients=np.asarray(coefficients))
    return SD_data.build(
        num_electron_up=geminal.num_electron_up,
        num_electron_dn=geminal.num_electron_dn,
        mos_up=mos_up,
        mos_dn=mos_dn,
        time_reversal=time_reversal,
    )


def _orthonormality_warnings(caplog):
    return [rec for rec in caplog.records if rec.levelno == logging.WARNING and rec.getMessage().startswith(_WARNING_PREFIX)]


def _channel_metrics(geminal):
    out = []
    for orb_data in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        out.append(_compute_mo_orthonormality_metrics(orb_data, np.asarray(compute_overlap_matrix(orb_data.aos_data))))
    return out


@pytest.mark.parametrize("trexio_file", _ORTHONORMAL_FIXTURES)
def test_orthonormal_fixture_mos_do_not_warn(trexio_file, caplog):
    """Orthonormal MOs build a Hamiltonian without the MO orthonormality warning.

    The committed PySCF-derived TREXIO fixtures (molecular Cartesian and
    spherical, periodic Gamma, periodic twist) are orthonormal in the AO metric
    M = conj(C) S C^T to round-off, so ``Hamiltonian_data.build`` (which runs
    ``sanity_check``) must not emit the warning, and both metrics must be at or
    below the production threshold in each channel.
    """
    structure, geminal, coulomb = _load(trexio_file)
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        _hamiltonian(structure, geminal, coulomb)
    assert _orthonormality_warnings(caplog) == []
    for normalization_deviation, non_orthogonality in _channel_metrics(geminal):
        assert normalization_deviation <= MO_ORTHONORMALITY_WARNING_THRESHOLD
        assert non_orthogonality <= MO_ORTHONORMALITY_WARNING_THRESHOLD


@pytest.mark.parametrize("trexio_file", _ORTHONORMAL_FIXTURES)
def test_uniformly_scaled_mos_warn_on_normalization(trexio_file, caplog):
    """A common MO scale (the old sqrt(Omega) QE convention) warns on normalization only.

    Every up-spin row is multiplied by 2 (molecular) or sqrt(Omega) (periodic)
    and the down channel is re-derived by time reversal. The normalization
    deviation must exceed the threshold while the scale-invariant
    non-orthogonality stays within it, and one warning per spin channel must
    name the channel, report both numbers and carry the unit-cell normalization hint. The
    build must not raise.
    """
    structure, geminal, coulomb = _load(trexio_file)
    scale = np.sqrt(abs(np.linalg.det(np.asarray(structure.cell)))) if structure.pbc_flag else 2.0
    scaled = _with_up_coefficients(geminal, scale * np.asarray(geminal.orb_data_up_spin.mo_coefficients))
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        _hamiltonian(structure, scaled, coulomb)
    warnings = _orthonormality_warnings(caplog)
    assert len(warnings) == 2
    assert "spin-up" in warnings[0].getMessage() and "spin-dn" in warnings[1].getMessage()
    for record in warnings:
        message = record.getMessage()
        assert "max|M_ii - 1|" in message and "max_(i!=j) |M_ij|/sqrt(M_ii M_jj)" in message
        assert "unit-cell normalization convention" in message and "sqrt(Omega)" in message
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    for normalization_deviation, non_orthogonality in _channel_metrics(scaled):
        assert normalization_deviation > MO_ORTHONORMALITY_WARNING_THRESHOLD
        assert non_orthogonality <= MO_ORTHONORMALITY_WARNING_THRESHOLD
        np.testing.assert_allclose(normalization_deviation, scale**2 - 1.0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("trexio_file", _MULTI_MO_FIXTURES)
def test_mixed_mos_warn_on_orthogonality(trexio_file, caplog):
    """A unit-norm but non-orthogonal MO set warns on orthogonality only.

    The first occupied up-spin MO is replaced by (phi_0 + w phi_1)/sqrt(2)
    with w = 1 (real) or w = i (complex twist). Its norm stays one, but its
    normalized overlap with phi_1 is exactly 1/sqrt(2): the non-orthogonality
    must exceed the threshold and equal 1/sqrt(2) in the strict det_eval
    policy, the normalization deviation must stay within the threshold, and
    the build must warn for both spin channels without raising. Rescaling the
    mixed set by 2 (molecular) or sqrt(Omega) (periodic) must leave the
    non-orthogonality at 1/sqrt(2) (it is normalized by sqrt(M_ii M_jj)).
    """
    structure, geminal, coulomb = _load(trexio_file)
    coefficients = np.array(geminal.orb_data_up_spin.mo_coefficients)
    weight = 1j if np.iscomplexobj(coefficients) else 1.0
    coefficients[0] = (coefficients[0] + weight * coefficients[1]) / np.sqrt(2.0)
    mixed = _with_up_coefficients(geminal, coefficients)
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        _hamiltonian(structure, mixed, coulomb)
    assert len(_orthonormality_warnings(caplog)) == 2
    scale = np.sqrt(abs(np.linalg.det(np.asarray(structure.cell)))) if structure.pbc_flag else 2.0
    mixed_scaled = _with_up_coefficients(geminal, scale * coefficients)
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    for normalization_deviation, non_orthogonality in _channel_metrics(mixed):
        assert normalization_deviation <= MO_ORTHONORMALITY_WARNING_THRESHOLD
        assert non_orthogonality > MO_ORTHONORMALITY_WARNING_THRESHOLD
        np.testing.assert_allclose(non_orthogonality, 1.0 / np.sqrt(2.0), atol=atol, rtol=rtol)
    for _normalization_deviation, non_orthogonality in _channel_metrics(mixed_scaled):
        np.testing.assert_allclose(non_orthogonality, 1.0 / np.sqrt(2.0), atol=atol, rtol=rtol)


def test_scaled_mos_warn_again_when_loaded_from_hdf5(tmp_path, caplog):
    """The warning is re-emitted by ``load_from_hdf5`` (which runs ``sanity_check``).

    A periodic Gamma Hamiltonian with sqrt(Omega)-scaled MOs (an old-convention
    file) is saved and loaded back; loading must succeed and warn once per
    spin channel.
    """
    structure, geminal, coulomb = _load("H4_pbc_ae_cart.h5")
    scale = np.sqrt(abs(np.linalg.det(np.asarray(structure.cell))))
    scaled = _with_up_coefficients(geminal, scale * np.asarray(geminal.orb_data_up_spin.mo_coefficients))
    hamiltonian = _hamiltonian(structure, scaled, coulomb)
    path = tmp_path / "hamiltonian_data.h5"
    hamiltonian.save_to_hdf5(str(path))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        Hamiltonian_data.load_from_hdf5(str(path))
    assert len(_orthonormality_warnings(caplog)) == 2


def test_degenerate_and_unsupported_orbitals_never_raise(caplog):
    """The diagnostic never raises and skips what it cannot measure.

    A null MO row (M_ii = 0) and a non-finite MO row warn without raising.
    AO-basis AGP orbital data (no MOs) and a molecular AO composite (no overlap
    defined) are skipped with a debug record and no warning. A standalone
    complex B-spline leaf is measured, not skipped: its MO scaled by 5 warns on
    normalization in both channels (the judgment against the independent cell
    quadrature is in ``test_bspline_complex_overlap.py``).
    """
    structure, geminal, coulomb = _load("N2_ecp_ccpvtz_cart.h5")
    for bad_value in (0.0, np.nan):
        coefficients = np.array(geminal.orb_data_up_spin.mo_coefficients)
        coefficients[0] = bad_value
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="jqmc"):
            _hamiltonian(structure, _with_up_coefficients(geminal, coefficients, time_reversal=False), coulomb)
        assert len(_orthonormality_warnings(caplog)) == 2

    ao_geminal = AGP_data.convert_from_MOs_to_AOs(_load("H2_ae_ccpvdz_sphe.h5")[1])
    assert not isinstance(ao_geminal.orb_data_up_spin, MOs_data)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="jqmc"):
        _warn_mo_orthonormality(Wavefunction_data(geminal_data=ao_geminal, jastrow_data=Jastrow_data()))
    assert _orthonormality_warnings(caplog) == []
    assert any("not MOs" in rec.getMessage() for rec in caplog.records if rec.levelno == logging.DEBUG)

    mos = geminal.orb_data_up_spin
    composite = AOs_composite_data.build([mos.aos_data, mos.aos_data])
    composite_mos = MOs_data(
        num_mo=mos.num_mo,
        aos_data=composite,
        mo_coefficients=np.concatenate([np.asarray(mos.mo_coefficients), np.asarray(mos.mo_coefficients)], axis=1),
    )
    composite_sd = SD_data.build(
        num_electron_up=geminal.num_electron_up, num_electron_dn=geminal.num_electron_dn, mos_up=composite_mos
    )
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="jqmc"):
        _warn_mo_orthonormality(Wavefunction_data(geminal_data=composite_sd, jastrow_data=Jastrow_data()))
    assert _orthonormality_warnings(caplog) == []
    assert any("molecular AO composite" in rec.getMessage() for rec in caplog.records if rec.levelno == logging.DEBUG)

    # A standalone complex B-spline leaf is not skipped:
    # its overlap is the exact cell Gram matrix, so a single random band with
    # MO coefficient 5 is measured with M_00 = 25 S_00 and warns on
    # normalization in both spin channels (S_00 is far from 1/25 here).
    bspline_structure = make_pbc_structure(CELL_FIXTURES["cubic_10bohr"])
    rng = np.random.default_rng(3)
    spline_shape = (8, 8, 8)
    bspline_leaf = AOs_bspline_complex_data.build(
        structure_data=bspline_structure,
        spline_coefficients=rng.normal(size=(1, *spline_shape)) + 1j * rng.normal(size=(1, *spline_shape)),
        spline_shape=spline_shape,
        k_point=(0.25, 0.0, 0.0),
    )
    bspline_mos = MOs_data(num_mo=1, aos_data=bspline_leaf, mo_coefficients=5.0 * np.eye(1, dtype=np.complex128))
    bspline_sd = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=bspline_mos)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="jqmc"):
        _warn_mo_orthonormality(Wavefunction_data(geminal_data=bspline_sd, jastrow_data=Jastrow_data()))
    assert len(_orthonormality_warnings(caplog)) == 2
    assert not any("skipped" in rec.getMessage() for rec in caplog.records if rec.levelno == logging.DEBUG)
    overlap_00 = float(np.real(np.asarray(compute_overlap_matrix(bspline_leaf))[0, 0]))
    for normalization, non_orthogonality in _channel_metrics(bspline_sd):
        atol, rtol = get_tolerance_min(("det_eval",), "strict")
        np.testing.assert_allclose(normalization, abs(25.0 * overlap_00 - 1.0), atol=atol, rtol=rtol)
        assert non_orthogonality == 0.0


def _two_atom_triclinic_leaf(ao_kind):
    """Small periodic pGTO leaf (Li s,p,s + H s) in the skewed triclinic cell."""
    structure = make_pbc_structure(
        CELL_FIXTURES["triclinic_skewed"],
        positions=np.array([[1.0, 1.2, 0.8], [2.6, 2.1, 1.9]], dtype=np.float64),
        atomic_numbers=(3, 1),
        element_symbols=("Li", "H"),
        atomic_labels=("Li1", "H1"),
    )
    common = dict(
        structure_data=structure,
        nucleus_index=(0, 0, 0, 0, 0, 1),
        num_ao=6,
        num_ao_prim=6,
        orbital_indices=(0, 1, 2, 3, 4, 5),
        exponents=np.array([0.3, 0.6, 0.6, 0.6, 2.0, 0.5], dtype=np.float64),
        coefficients=np.ones(6),
        angular_momentums=(0, 1, 1, 1, 0, 0),
    )
    if ao_kind == "sphe":
        base = AOs_sphe_gto_data._from_full_basis(**common, magnetic_quantum_numbers=(0, -1, 0, 1, 0, 0))
    else:
        base = AOs_cart_gto_data._from_full_basis(
            **common,
            polynominal_order_x=(0, 1, 0, 0, 0, 0),
            polynominal_order_y=(0, 0, 1, 0, 0, 0),
            polynominal_order_z=(0, 0, 0, 1, 0, 0),
        )
    return _apply_image_expansion(base, structure, tol=AO_IMAGE_TRUNCATION_ATOL)


@pytest.mark.parametrize("ao_kind", ["cart", "sphe"])
@pytest.mark.parametrize("twist", ["gamma", "general-k"])
def test_mo_gram_convention_matches_cell_grid_integral(ao_kind, twist):
    """The check's MO Gram matrix equals the cell integral of conj(phi_i) phi_j.

    Three generic MOs (random real at Gamma, random complex at the general
    twist k = (0.27, -0.1, 0.13)) on a two-atom pGTO leaf in the skewed
    triclinic cell. M = conj(C) S C^T from the production overlap must match
    an independent 40^3 fractional-midpoint integral of conj(phi_i) phi_j
    built from production MO values, within the medium ao_eval/mo_eval zone
    policy (the midpoint rule is spectrally accurate for these smooth periodic
    integrands; measured deviation 3e-8 full, 5e-6 mixed). At the twist, the two wrong conventions
    (conjugating the ket side, i.e. M^T, and using conj(S)) must both be
    rejected by the same comparison, so the check cannot report false
    non-orthogonality at general k.
    """
    leaf = _two_atom_triclinic_leaf(ao_kind)
    rng = np.random.default_rng(11)
    coefficients = rng.normal(size=(3, leaf.num_ao))
    if twist == "general-k":
        leaf = _promote_pgto_to_complex(leaf, (0.27, -0.1, 0.13))
        coefficients = coefficients + 1j * rng.normal(size=(3, leaf.num_ao))
    mos = MOs_data(num_mo=3, aos_data=leaf, mo_coefficients=coefficients)
    overlap = np.asarray(compute_overlap_matrix(leaf))
    gram = _compute_mo_gram_matrix(mos, overlap)

    cell = np.asarray(leaf.structure_data.cell, dtype=np.float64)
    num_grid_points = 40
    steps = (np.arange(num_grid_points, dtype=np.float64) + 0.5) / num_grid_points
    fa, fb, fc = np.meshgrid(steps, steps, steps, indexing="ij")
    r_carts = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1) @ cell
    volume_element = abs(np.linalg.det(cell)) / num_grid_points**3
    values = np.asarray(compute_MOs(mos, jnp.asarray(r_carts)))
    values = values.astype(np.complex128 if np.iscomplexobj(values) else np.float64)
    gram_grid = (values.conj() @ values.T) * volume_element

    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval"), "medium")
    assert np.all(np.isfinite(gram)) and np.all(np.isfinite(gram_grid))
    np.testing.assert_allclose(gram, gram_grid, atol=atol, rtol=rtol)
    if twist == "general-k":
        assert np.max(np.abs(np.imag(overlap))) > atol
        ket_conjugated = gram.T
        conjugated_overlap = coefficients.conj() @ overlap.conj() @ coefficients.T
        assert not np.allclose(ket_conjugated, gram_grid, atol=atol, rtol=rtol)
        assert not np.allclose(conjugated_overlap, gram_grid, atol=atol, rtol=rtol)
