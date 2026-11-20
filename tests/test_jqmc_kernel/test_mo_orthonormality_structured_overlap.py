"""Cost-bounded MO orthonormality diagnostic of ``Hamiltonian_data.sanity_check`` (issue #247).

The diagnostic forms ``M = conj(C) S C^T``. A plane-wave leaf has the closed-form
overlap ``S = V I`` (complex) or a diagonal half-space overlap (real), and a
periodic composite is the sum of its child blocks, so ``M`` must be formed
without the ``num_ao x num_ao`` matrix, which would need 1.34 TiB for
303,144 plane waves. The structured Gram matrix is checked against the
production dense overlap on small leaves, its allocation is measured with
``tracemalloc``, a second check of the same orbitals reuses the metrics while
and still warning.

Memory safety of this module: every case that a dense-overlap implementation would
form as a dense plane-wave overlap is kept below 4,000 functions (a few hundred MB). The
97,336-function case first forms a one-MO Gram matrix with a dense-overlap
callback that raises, so a dense implementation fails before allocating anything.
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

import logging
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from jqmc import hamiltonians
from jqmc._precision import get_tolerance_min
from jqmc.atomic_orbital import (
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    AOs_plane_wave_data,
    compute_overlap_matrix,
)
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data, _warn_mo_orthonormality
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._opposite_spin_twist_cases import opposite_twist_partner
from tests.test_jqmc_kernel.test_composite_overlap_blocks import _composite, _pgto, _plane_waves

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
_WARNING_PREFIX = "MOs are not orthonormal"

#: case -> (twist fixture, leaf kind)
_LEAVES = {
    "pw-gamma-half-space": ("gamma", "plane-wave"),
    "pw-gamma-signed-pair": ("gamma", "plane-wave-signed-pair"),
    "pw-twist": ("twist", "plane-wave"),
    "composite-pw-gamma": ("gamma", "composite-plane-wave"),
    "composite-pw-twist": ("twist", "composite-plane-wave"),
    "composite-bspline-twist": ("twist", "composite-bspline"),
    "pgto-twist": ("twist", "pgto"),
}


def _no_dense_overlap(leaf):
    raise AssertionError(f"dense overlap of {type(leaf).__name__} requested")


def _leaf(case):
    twist, kind = _LEAVES[case]
    structure, pgto = _pgto(twist)
    if kind == "plane-wave":
        return _plane_waves(structure, pgto)
    if kind == "plane-wave-signed-pair":
        # cos(G.r) for both G and -G: not a half-space basis, so the overlap has off-diagonal entries.
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(structure.cell, dtype=np.float64)).T
        g_vectors = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 1]], dtype=np.float64) @ reciprocal
        return AOs_plane_wave_data.build(structure, g_vectors, np.zeros(4, dtype=int))
    if kind == "pgto":
        return pgto
    return _composite(twist, kind.removeprefix("composite-"))[3]


def _coefficients(num_mo, num_ao, complex_valued, seed=7):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(num_mo, num_ao))
    if complex_valued:
        coefficients = coefficients + 1j * rng.normal(size=(num_mo, num_ao))
    return coefficients


def _plane_wave_leaf(num_ao_per_axis, *, complex_valued):
    """A cubic grid of reciprocal vectors on the H2 periodic cell (Gamma real or k = 1/4 complex)."""
    structure, pgto = _pgto("twist" if complex_valued else "gamma")
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(structure.cell, dtype=np.float64)).T
    half = num_ao_per_axis // 2
    axis = np.arange(-half, num_ao_per_axis - half)
    millers = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    if complex_valued:
        return AOs_plane_wave_complex_data.build(structure, millers @ reciprocal, k_point=pgto.k_point)
    # Real half-space basis: G = 0 (cos) and, for each G with a positive leading non-zero
    # component, one cos and one sin function.
    positive = [m for m in millers if tuple(m) > (0, 0, 0)]
    g_vectors = np.concatenate([np.zeros((1, 3)), np.repeat(np.asarray(positive) @ reciprocal, 2, axis=0)])
    phases = np.array([0] + [0, 1] * len(positive))
    return AOs_plane_wave_data.build(structure, g_vectors, phases)


@pytest.mark.parametrize("case", list(_LEAVES))
def test_structured_mo_gram_equals_the_dense_overlap_gram(case):
    """``_overlap_gram`` equals ``conj(C) S C^T`` with the production dense ``compute_overlap_matrix``."""
    aos_data = _leaf(case)
    dense = np.asarray(compute_overlap_matrix(aos_data))
    coefficients = _coefficients(3, int(aos_data.num_ao), np.iscomplexobj(dense) or case.endswith("twist"))
    dtype = np.complex128 if np.iscomplexobj(coefficients) or np.iscomplexobj(dense) else np.float64
    reference = coefficients.astype(dtype).conj() @ dense.astype(dtype) @ coefficients.astype(dtype).T
    dense_requests = []

    def dense_overlap(leaf):
        dense_requests.append(type(leaf).__name__)
        return np.asarray(compute_overlap_matrix(leaf))

    structured = aos_data._overlap_gram(coefficients, coefficients, dense_overlap)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval"), "strict")
    np.testing.assert_allclose(structured, reference, atol=atol * np.max(np.abs(reference)), rtol=rtol)
    # Only the pure plane-wave leaves (half-space real, complex) never request a dense overlap.
    plane_wave_only = case.startswith("pw-gamma-half") or case == "pw-twist"
    assert (dense_requests == []) == plane_wave_only


@pytest.mark.parametrize("leaf", ["pw-gamma", "pw-twist", "composite-pw-twist"])
def test_plane_wave_mo_gram_allocates_no_dense_overlap(leaf):
    """The Gram matrix of a 16^3 plane-wave basis, alone or beside a pGTO leaf, allocates far less than one dense overlap."""
    complex_valued = leaf.endswith("twist")
    aos_data = _plane_wave_leaf(16, complex_valued=complex_valued)
    if leaf.startswith("composite"):
        aos_data = AOs_composite_data.build((_pgto("twist")[1], aos_data))
    num_ao = int(aos_data.num_ao)
    coefficients = _coefficients(4, num_ao, complex_valued)
    dense_bytes = 16 * num_ao * num_ao
    tracemalloc.start()
    try:
        gram = aos_data._overlap_gram(coefficients, coefficients, lambda leaf: np.asarray(compute_overlap_matrix(leaf)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert gram.shape == (4, 4)
    assert np.all(np.isfinite(gram))
    # A dense overlap is dense_bytes; the structured Gram (and a composite's pGTO x plane-wave cross block) is O(n_AO).
    assert peak < dense_bytes // 16


def test_large_plane_wave_hamiltonian_check_is_memory_bounded(caplog):
    """The MO check of 97,336 plane waves (151 GB as a dense overlap) stays within a few C-sized arrays."""
    aos_data = _plane_wave_leaf(46, complex_valued=True)
    coefficients = _coefficients(4, int(aos_data.num_ao), True)
    # Required before any allocation: a dense implementation fails here, not in a 151 GB np.eye.
    assert aos_data._overlap_gram(coefficients[:1], coefficients[:1], _no_dense_overlap).shape == (1, 1)
    mos = MOs_data.build(aos_data=aos_data, mo_coefficients=coefficients)
    geminal = SD_data.build(
        num_electron_up=4, num_electron_dn=4, mos_up=mos, mos_dn=opposite_twist_partner(mos), time_reversal=False
    )
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    tracemalloc.start()
    try:
        with caplog.at_level(logging.WARNING, logger="jqmc"):
            _warn_mo_orthonormality(wavefunction)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * coefficients.nbytes
    # Random coefficients are not orthonormal: the diagnostic ran and warned.
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any(message.startswith(_WARNING_PREFIX) for message in messages)


def _scaled_periodic_hamiltonian():
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(
        str(_TREXIO_DIR / "H4_pbc_ae_cart.h5"), store_tuple=True
    )
    scale = np.sqrt(abs(np.linalg.det(np.asarray(structure.cell))))
    mos_up = geminal.orb_data_up_spin.replace(mo_coefficients=scale * np.asarray(geminal.orb_data_up_spin.mo_coefficients))
    scaled = SD_data.build(num_electron_up=geminal.num_electron_up, num_electron_dn=geminal.num_electron_dn, mos_up=mos_up)
    wavefunction = Wavefunction_data(geminal_data=scaled, jastrow_data=Jastrow_data())
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)


def test_second_check_of_the_same_orbitals_reuses_the_metrics_and_still_warns(monkeypatch, tmp_path, caplog):
    """HDF5 load and the later driver check share the overlap work; each check still warns per spin channel."""
    path = tmp_path / "hamiltonian_data.h5"
    _scaled_periodic_hamiltonian().save_to_hdf5(str(path))
    calls = []
    production_overlap = hamiltonians.compute_overlap_matrix
    monkeypatch.setattr(hamiltonians, "compute_overlap_matrix", lambda aos: calls.append(id(aos)) or production_overlap(aos))
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        loaded = Hamiltonian_data.load_from_hdf5(str(path))
        first_warnings = [r for r in caplog.records if r.getMessage().startswith(_WARNING_PREFIX)]
        first_calls = len(calls)
        caplog.clear()
        loaded.sanity_check()  # the check MCMC / GFMC run on the loaded object
        second_warnings = [r for r in caplog.records if r.getMessage().startswith(_WARNING_PREFIX)]
    assert first_calls >= 1
    assert len(calls) == first_calls
    assert len(first_warnings) == 2
    assert [r.getMessage() for r in second_warnings] == [r.getMessage() for r in first_warnings]
