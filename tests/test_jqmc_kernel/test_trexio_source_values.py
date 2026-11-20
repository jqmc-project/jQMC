"""Deterministic TREXIO-to-storage value checks, independently indexed from raw fields."""

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

import math
from pathlib import Path

import numpy as np
import pytest
import trexio

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc.trexio_wrapper import read_trexio_file

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _raw_fields(path):
    """Read source arrays without using jQMC conversion or normalization helpers."""
    source = trexio.File(str(path), mode="r", back_end=trexio.TREXIO_HDF5)
    try:
        raw = {
            "electrons": [trexio.read_electron_up_num(source), trexio.read_electron_dn_num(source)],
            "positions": trexio.read_nucleus_coord(source),
            "charges": trexio.read_nucleus_charge(source),
            "labels": trexio.read_nucleus_label(source),
            "shell_atom": trexio.read_basis_nucleus_index(source),
            "shell_l": trexio.read_basis_shell_ang_mom(source),
            "shell_factor": trexio.read_basis_shell_factor(source),
            "prim_shell": trexio.read_basis_shell_index(source),
            "exponents": trexio.read_basis_exponent(source),
            "coefficients": trexio.read_basis_coefficient(source),
            "prim_factor": trexio.read_basis_prim_factor(source),
            "ao_shell": trexio.read_ao_shell(source),
            "ao_normalization": trexio.read_ao_normalization(source),
            "cartesian": bool(trexio.read_ao_cartesian(source)),
            "pbc": bool(trexio.read_pbc_periodic(source)) if trexio.has_pbc_periodic(source) else False,
            "ecp": bool(trexio.has_ecp_num(source)),
        }
        if raw["pbc"]:
            raw["cell"] = np.stack([trexio.read_cell_a(source), trexio.read_cell_b(source), trexio.read_cell_c(source)])
            raw["k_points"] = np.asarray(trexio.read_pbc_k_point(source)).reshape(-1, 3)
        if raw["ecp"]:
            raw["ecp_fields"] = {
                "z_cores": trexio.read_ecp_z_core(source),
                "max_ang_mom_plus_1": trexio.read_ecp_max_ang_mom_plus_1(source),
                "ang_moms": trexio.read_ecp_ang_mom(source),
                "nucleus_index": trexio.read_ecp_nucleus_index(source),
                "exponents": trexio.read_ecp_exponent(source),
                "coefficients": trexio.read_ecp_coefficient(source),
                # TREXIO's r**n / r**2 storage becomes jQMC's radial power.
                "powers": trexio.read_ecp_power(source) + 2,
            }
            raw["ecp_num"] = trexio.read_ecp_num(source)
        return raw
    finally:
        source.close()


def _basis_reference(aos, raw):
    """Map each real kernel primitive to raw AO/shell/primitive records.

    The independent Gaussian L2 normalization converts the stored jQMC
    contraction back to the coefficient multiplying the unnormalized
    Cartesian polynomial or real spherical solid harmonic. This checks
    conversion/storage, not any evaluated wavefunction or physical oracle.
    """
    ao_shell = np.asarray(raw["ao_shell"], dtype=np.int64)
    prim_shell = np.asarray(raw["prim_shell"], dtype=np.int64)
    home = np.asarray(aos._ragged_kernel_to_home if raw["pbc"] else np.arange(aos.num_ao), dtype=np.int64)
    assert len(home) == aos._num_ao_kernel
    np.testing.assert_array_equal(np.unique(home), np.arange(len(ao_shell), dtype=np.int64))
    kernel_shell = ao_shell[home]
    expected_l = np.asarray(raw["shell_l"], dtype=np.int64)[kernel_shell]
    expected_atom = np.asarray(raw["shell_atom"], dtype=np.int64)[kernel_shell]
    atoms = np.asarray(aos._ragged_kernel_atom if raw["pbc"] else aos.nucleus_index, dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(aos.angular_momentums, dtype=np.int64), expected_l)
    np.testing.assert_array_equal(atoms, expected_atom)
    orbital_indices = np.asarray(aos.orbital_indices, dtype=np.int64)
    exponents = np.asarray(aos.exponents)
    coefficients = np.asarray(aos.coefficients)
    assert exponents.dtype == np.float64 and coefficients.dtype == np.float64
    assert exponents.size == coefficients.size == orbital_indices.size > 0
    reference_exponents = np.empty_like(exponents)
    raw_contractions = np.empty_like(coefficients)
    normalizations = np.empty_like(coefficients)
    visited = np.zeros(orbital_indices.size, dtype=np.int64)
    for kernel_ao, home_ao in enumerate(home):
        shell = int(ao_shell[home_ao])
        angular = int(raw["shell_l"][shell])
        source_primitives = np.flatnonzero(prim_shell == shell)
        target_primitives = np.flatnonzero(orbital_indices == kernel_ao)
        assert source_primitives.size == target_primitives.size > 0
        component = int(np.flatnonzero(np.flatnonzero(ao_shell == shell) == home_ao)[0])
        if raw["cartesian"]:
            powers = [(x, y, angular - x - y) for x in range(angular, -1, -1) for y in range(angular - x, -1, -1)]
            nx, ny, nz = powers[component]
            assert (
                aos.polynominal_order_x[kernel_ao],
                aos.polynominal_order_y[kernel_ao],
                aos.polynominal_order_z[kernel_ao],
            ) == (nx, ny, nz)
            double_factorials = math.prod(math.prod(range(1, 2 * n, 2)) for n in (nx, ny, nz))
        else:
            magnetic = [0, *[m * sign for m in range(1, angular + 1) for sign in (1, -1)]]
            assert aos.magnetic_quantum_numbers[kernel_ao] == magnetic[component]
        for target, primitive in zip(target_primitives, source_primitives, strict=True):
            exponent = float(raw["exponents"][primitive])
            reference_exponents[target] = exponent
            raw_contractions[target] = (
                raw["coefficients"][primitive]
                * raw["prim_factor"][primitive]
                * raw["shell_factor"][shell]
                * raw["ao_normalization"][home_ao]
            )
            if raw["cartesian"]:
                normalizations[target] = (2 * exponent / math.pi) ** 0.75 * math.sqrt(
                    (4 * exponent) ** angular / double_factorials
                )
            else:
                normalizations[target] = math.sqrt(
                    (2 * angular + 1) / (4 * math.pi) * 2 * (2 * exponent) ** (angular + 1.5) / math.gamma(angular + 1.5)
                )
            visited[target] += 1
    np.testing.assert_array_equal(visited, np.ones_like(visited))
    np.testing.assert_array_equal(exponents, reference_exponents)
    return coefficients * normalizations, raw_contractions


@pytest.mark.parametrize("store_tuple", [True, False])
@pytest.mark.parametrize(
    "trexio_file,twist_ik",
    [
        ("H2_ae_ccpvdz_cart.h5", 0),
        ("H2_ae_ccpvdz_sphe.h5", 0),
        ("H2_ecp_ccpvdz_cart.h5", 0),
        ("H2_ecp_ccpvtz.h5", 0),
        ("water_ccecp_ccpvqz.h5", 0),
        ("N_ae_ccpvdz_cart.h5", 0),
        ("Li_ecp_ccpvdz_cart.h5", 0),
        ("H2_pbc_ae_cart.h5", 0),
        ("H2_pbc_ae_sphe.h5", 0),
        ("H2_pbc_ae_k025_cart.h5", 0),
        ("H2_pbc_ae_k025_sphe.h5", 0),
        ("lithium_cart.h5", 0),
        ("lithium.h5", 0),
        ("lithium_k025_cart.h5", 0),
        ("lithium_k025.h5", 0),
        ("lithium_twist_222.h5", 0),
        ("lithium_twist_222.h5", 3),
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="input-storage conversion; no electronic coordinates are evaluated"
)
def test_trexio_storage_matches_raw_source(trexio_file, twist_ik, store_tuple):
    """Loaded nuclear, electron, basis, ECP and cell data match their raw TREXIO records.

    The explicit committed-file matrix covers Cartesian/spherical, AE/ECP,
    molecular/Gamma/non-Gamma, closed/open shell (including one empty spin),
    and two distinct rows of multi-twist storage, through tuple and list input.
    It evaluates no electronic configuration: no positions, seed or cond(G)
    selection is used. Nuclear positions are input data, not electron positions.
    Electron counts, charges, AO angular/atom/primitive mappings, exponents,
    ECP channels, cell and signed spin k-points match exactly. Every kernel
    primitive participates, including the periodic image copies. Gaussian
    contraction factors are reconstructed from raw normalization data using
    independent scalar Gaussian integrals and existing central fp64 Python
    bounds. I/O storage is fp64 in both modes; no full/mixed output comparison
    is made. MO coefficient fidelity is covered by separate spin/phase tests.
    """
    path = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
    structure, aos, mo_up, mo_down, determinant, coulomb = read_trexio_file(
        trexio_file=str(path),
        store_tuple=store_tuple,
        twist_ik=twist_ik,
    )
    raw = _raw_fields(path)
    assert np.asarray(structure.positions).dtype == np.float64, "coordinate storage must be fp64"
    assert structure.pbc_flag is raw["pbc"] and coulomb.ecp_flag is raw["ecp"]
    assert isinstance(structure.atomic_numbers, tuple if store_tuple else list)
    assert tuple(structure.atomic_labels) == tuple(raw["labels"])
    np.testing.assert_array_equal(np.asarray(structure.positions), np.asarray(raw["positions"]))
    np.testing.assert_array_equal(
        np.asarray([determinant.num_electron_up, determinant.num_electron_dn], dtype=np.int64),
        np.asarray(raw["electrons"], dtype=np.int64),
    )
    z_core = np.asarray(raw["ecp_fields"]["z_cores"]) if raw["ecp"] else np.zeros(len(raw["charges"]))
    np.testing.assert_array_equal(
        np.asarray(structure.atomic_numbers), np.rint(np.asarray(raw["charges"]) + z_core).astype(np.int64)
    )
    if raw["ecp"]:
        assert coulomb.num_ecps == raw["ecp_num"] > 0
        for field, expected in raw["ecp_fields"].items():
            actual = np.asarray(getattr(coulomb, field))
            if field in ("coefficients", "exponents"):
                assert actual.dtype == np.float64, "ECP storage must be fp64"
            expected = np.asarray(expected, dtype=actual.dtype)
            assert actual.size > 0
            np.testing.assert_array_equal(actual, expected)
    if raw["pbc"]:
        assert np.asarray(structure.cell).dtype == np.float64, "cell storage must be fp64"
        np.testing.assert_array_equal(np.asarray(structure.cell), raw["cell"])
        expected_k = raw["k_points"][twist_ik]
        if len(raw["k_points"]) == 1 and np.array_equal(expected_k, np.zeros(3)):
            assert mo_up.aos_data.k_point is None and mo_down.aos_data.k_point is None
        else:
            actual_k = np.asarray(mo_up.aos_data.k_point)
            np.testing.assert_array_equal(actual_k, expected_k)
            expected_down_k = -expected_k if determinant.time_reversal else expected_k
            actual_down_k = np.asarray(mo_down.aos_data.k_point)
            np.testing.assert_array_equal(actual_down_k, expected_down_k)
    else:
        assert len(structure.vec_a) == len(structure.vec_b) == len(structure.vec_c) == 0
    actual_contractions, expected_contractions = _basis_reference(aos, raw)
    assert np.all(np.isfinite(actual_contractions)) and np.any(expected_contractions != 0)
    np.testing.assert_allclose(
        actual_contractions,
        expected_contractions,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        equal_nan=False,
    )


@pytest.mark.parametrize("trexio_file,twist_ik,store_tuple", [("lithium_k025_cart.h5", 0, True)])
@pytest.mark.parametrize(
    "fault", ["positions", "electrons", "exponents", "coefficients", "ecp", "cell", "k", "primitive-map", "storage-dtype"]
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="input-storage corruption check; no electronic coordinates are evaluated"
)
def test_raw_source_verdict_rejects_loader_corruption(monkeypatch, trexio_file, twist_ik, store_tuple, fault):
    """Each of nine corrupted loader outputs fails after the identical baseline passes.

    Committed non-Gamma Cartesian Li provides nonzero ECP and basis payloads,
    both spin sectors and a cell. The raw-file reader stays untouched. Corrupt
    nuclear positions, electron count, basis exponent/coefficient, ECP coefficient,
    cell, k sign, primitive map or ECP storage dtype in the returned object, never the source file.
    No electron configuration, tolerance change or statistical test is involved.
    """
    import sys

    test_trexio_storage_matches_raw_source(trexio_file, twist_ik, store_tuple)
    original = read_trexio_file

    def corrupt(*args, **kwargs):
        structure, aos, up, down, determinant, coulomb = original(*args, **kwargs)
        if fault == "positions":
            structure = structure.replace(positions=np.asarray(structure.positions) + 1)
        elif fault == "electrons":
            determinant = determinant.replace(num_electron_up=determinant.num_electron_up + 1)
        elif fault in ("exponents", "coefficients"):
            value = np.asarray(getattr(aos.radial, fault)).copy()
            value[np.argmax(np.abs(value))] *= 1 + 1e-6
            aos = aos.replace(radial=aos.radial.replace(**{fault: value}))
        elif fault == "ecp":
            value = np.asarray(coulomb.coefficients).copy()
            value[np.argmax(np.abs(value))] += 1
            coulomb = coulomb.replace(coefficients=tuple(value))
        elif fault == "cell":
            structure = structure.replace(vec_a=tuple(np.asarray(structure.vec_a) + 1))
        elif fault == "k":
            up = up.replace(aos_data=up.aos_data.replace(k_point=tuple(-np.asarray(up.aos_data.k_point))))
        elif fault == "storage-dtype":
            coulomb = coulomb.replace(coefficients=tuple(np.asarray(coulomb.coefficients, dtype=np.float32)))
        else:
            mapping = np.asarray(aos.ao_prim_to_shell_prim).copy()
            exponents = np.asarray(aos.radial.exponents)
            alternatives = np.flatnonzero(exponents != exponents[mapping[0]])
            assert alternatives.size > 0, "primitive-map fault requires a distinct exponent"
            mapping[0] = alternatives[0]
            aos = aos.replace(ao_prim_to_shell_prim=tuple(mapping))
        return structure, aos, up, down, determinant, coulomb

    monkeypatch.setattr(sys.modules[__name__], "read_trexio_file", corrupt)
    with pytest.raises(AssertionError, match="(not equal|Not equal|Mismatched|storage must be fp64)"):
        test_trexio_storage_matches_raw_source(trexio_file, twist_ik, store_tuple)
