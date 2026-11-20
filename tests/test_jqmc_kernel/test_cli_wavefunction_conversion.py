"""Deterministic contracts for the Hamiltonian wavefunction conversion command."""

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

import hashlib
from pathlib import Path

import numpy as np
import pytest
from click import unstyle
from typer.testing import CliRunner

from jqmc._precision import get_tolerance_min
from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_tool import hamiltonian_app
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction, evaluate_wavefunction
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel.test_hamiltonian import assert_dataclasses_equal


pytestmark = [pytest.mark.without_chdir]


@pytest.mark.heavy
@pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_pbc_ae_k025_sphe.h5"])
@pytest.mark.parametrize("target", ["jagp", "jagpn"])
@pytest.mark.parametrize("custom_output", [False, True])
@pytest.mark.parametrize(
    "r_up,r_dn",
    [
        (((0.21, 0.32, 0.43),), ((0.73, -0.24, 0.57),)),
        (((-0.31, 0.41, 0.26),), ((0.62, 0.53, -0.37),)),
        (((0.81, -0.23, 0.35),), ((-0.42, 0.64, 0.78),)),
    ],
    ids=["configuration-0", "configuration-1", "configuration-2"],
)
def test_cli_conversion_preserves_wavefunction_and_hamiltonian(
    tmp_path, monkeypatch, trexio_file, target, custom_output, r_up, r_dn
):
    """The real conversion CLI preserves signed/complex Psi and log magnitude at three fixed positions after HDF5 reload.

    The input is a closed-shell two-electron TREXIO SD with a nonzero Pade J2.
    Molecular Cartesian and non-Gamma spherical periodic AO inputs exercise
    real and complex values. Output ansatz, unchanged input bytes, output path
    and the untouched Hamiltonian/Jastrow fields are also checked. Positions
    are explicit, with no conditioning rejection or random sampling.
    """
    fixture = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_file
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(fixture), store_tuple=False)
    assert isinstance(geminal, SD_data)
    jastrow = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.25, jastrow_2b_type="pade", structure_data=structure
        )
    )
    source = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal),
    )
    source_path = tmp_path / "source.h5"
    source.save_to_hdf5(source_path)
    before_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    # Compare both routes from the same serialized, dtype-aligned input.
    baseline = Hamiltonian_data.load_from_hdf5(source_path)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / ("chosen-output.h5" if custom_output else "hamiltonian_data_conv.h5")
    args = ["conv-wf", str(source_path), "--convert-to", target]
    if custom_output:
        args.extend(["--output", str(output)])
    result = CliRunner().invoke(hamiltonian_app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert output.is_file()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before_digest
    if custom_output:
        assert not (tmp_path / "hamiltonian_data_conv.h5").exists()
    converted = Hamiltonian_data.load_from_hdf5(output)
    assert type(converted.wavefunction_data.geminal_data) is (AGP_data if target == "jagp" else AGPn_data)
    assert_dataclasses_equal(converted.structure_data, baseline.structure_data)
    assert_dataclasses_equal(converted.coulomb_potential_data, baseline.coulomb_potential_data)
    assert_dataclasses_equal(converted.wavefunction_data.jastrow_data, baseline.wavefunction_data.jastrow_data)
    r_up = np.asarray(r_up, dtype=np.float64)
    r_dn = np.asarray(r_dn, dtype=np.float64)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "jastrow_eval", "wf_eval"), "strict")
    expected = np.asarray(evaluate_wavefunction(baseline.wavefunction_data, r_up, r_dn))
    actual = np.asarray(evaluate_wavefunction(converted.wavefunction_data, r_up, r_dn))
    expected_log = np.asarray(evaluate_ln_wavefunction(baseline.wavefunction_data, r_up, r_dn))
    actual_log = np.asarray(evaluate_ln_wavefunction(converted.wavefunction_data, r_up, r_dn))
    assert np.isfinite(expected) and np.isfinite(actual)
    assert expected != 0
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_log, expected_log, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("target", [None, "invalid-ansatz", "jsd"])
@pytest.mark.parametrize("force_color", [False, True])
def test_cli_conversion_rejects_invalid_target_before_loading(tmp_path, monkeypatch, target, force_color):
    """Invalid targets fail before Hamiltonian I/O, with the same option diagnostic under forced ANSI styling."""
    if force_color:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    calls = []

    def forbidden_load(*args, **kwargs):
        calls.append(args)
        raise AssertionError("invalid target must not load a Hamiltonian")

    monkeypatch.setattr(Hamiltonian_data, "load_from_hdf5", forbidden_load)
    args = ["conv-wf", str(tmp_path / "absent.h5")]
    if target is not None:
        args.extend(["--convert-to", target])
    result = CliRunner().invoke(hamiltonian_app, args)
    assert result.exit_code == 2, (result.output, result.exception)
    if force_color:
        assert "\x1b[" in result.output
    assert "convert-to" in unstyle(result.output)
    assert not calls
    assert not (tmp_path / "hamiltonian_data_conv.h5").exists()


def _fixed_configuration(hamiltonian, seed):
    """Explicit electron positions around the atom (seeded, no sampling or conditioning screen)."""
    geminal = hamiltonian.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    center = np.asarray(hamiltonian.structure_data.positions)[0]
    r_up = center + rng.normal(scale=0.8, size=(int(geminal.num_electron_up), 3))
    r_dn = center + rng.normal(scale=0.8, size=(int(geminal.num_electron_dn), 3))
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("existing_output", [False, True], ids=["no-existing-output", "existing-output"])
def test_cli_conversion_refuses_unequal_ao_agp_without_writing(tmp_path, existing_output):
    """conv-wf -c jagp refuses an AGPn whose up and down AO sets differ in size, before writing.

    Independent reference for "not representable": the AO-basis paired lambda is
    C_up^T diag(lambda) C_dn with shape (num_ao_up, num_ao_dn), while an AGP
    pairs the two spin channels through a square paired block, so unequal AO
    counts admit no AGP. The public outcome is checked: non-zero exit with the
    AGP dimension diagnostic, no success message, and no file created or replaced
    at the output path.
    """
    source = build_canonical_hamiltonian("WF-PATTERN-014")
    geminal = source.wavefunction_data.geminal_data
    assert isinstance(geminal, AGPn_data)
    num_ao_up = geminal.orb_data_up_spin.aos_data.num_ao
    num_ao_dn = geminal.orb_data_dn_spin.aos_data.num_ao
    assert num_ao_up != num_ao_dn
    source_path = tmp_path / "source.h5"
    source.save_to_hdf5(source_path)
    output = tmp_path / "converted.h5"
    previous = b"previous conversion output"
    if existing_output:
        output.write_bytes(previous)
    result = CliRunner().invoke(hamiltonian_app, ["conv-wf", str(source_path), "--convert-to", "jagp", "--output", str(output)])
    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, ValueError), (result.output, result.exception)
    assert f"({num_ao_up}, {num_ao_dn}) should be the same" in str(result.exception)
    assert "Hamiltonian data is saved" not in result.output
    if existing_output:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()


@pytest.mark.heavy
@pytest.mark.parametrize(
    "pattern_id,target",
    [("WF-PATTERN-009", "jagp"), ("WF-PATTERN-014", "jagpn")],
    ids=["equal-ao-agpn-to-jagp", "unequal-ao-agpn-to-jagpn"],
)
def test_cli_conversion_of_canonical_agpn_reloads_and_preserves_wavefunction(tmp_path, pattern_id, target):
    """Representable AGPn conversions still write a file that reloads and evaluates the source wavefunction.

    Controls for the refused case: the equal-AO-count AGPn to AGP conversion, and the
    unequal-AO-count AGPn of the refused case converted to the representable
    AGPn target. The reference is the serialized source Hamiltonian evaluated at
    three explicit electron configurations (signed/complex Psi and ln|Psi|).
    """
    source_path = tmp_path / "source.h5"
    build_canonical_hamiltonian(pattern_id).save_to_hdf5(source_path)
    baseline = Hamiltonian_data.load_from_hdf5(source_path)
    output = tmp_path / "converted.h5"
    result = CliRunner().invoke(hamiltonian_app, ["conv-wf", str(source_path), "--convert-to", target, "--output", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    converted = Hamiltonian_data.load_from_hdf5(output)
    assert type(converted.wavefunction_data.geminal_data) is (AGP_data if target == "jagp" else AGPn_data)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "jastrow_eval", "wf_eval"), "strict")
    for seed in range(3):
        r_up, r_dn = _fixed_configuration(baseline, seed)
        expected = np.asarray(evaluate_wavefunction(baseline.wavefunction_data, r_up, r_dn))
        actual = np.asarray(evaluate_wavefunction(converted.wavefunction_data, r_up, r_dn))
        expected_log = np.asarray(evaluate_ln_wavefunction(baseline.wavefunction_data, r_up, r_dn))
        actual_log = np.asarray(evaluate_ln_wavefunction(converted.wavefunction_data, r_up, r_dn))
        assert np.isfinite(expected) and np.isfinite(actual)
        assert expected != 0
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual_log, expected_log, atol=atol, rtol=rtol)
