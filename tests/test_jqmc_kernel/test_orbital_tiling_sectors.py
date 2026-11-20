"""Complete supercell twist sectors, source conversion and ragged persistence.

These adapter tests do not certify the separate six-route force/stress matrix.
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

import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from jqmc._precision import get_tolerance_min
from jqmc._setting import (
    ATOL_NONVACUITY_DISTINCTION,
    ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
    RTOL_BIT_EXACT,
    RTOL_NONVACUITY_DISTINCTION,
)
from jqmc._twist_average import TwistAverageWriter, load_twist, load_twist_average_input, normalize_twist_weights
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jqmc_tool import _cli, typer_click_qe, typer_click_trexio
from jqmc.orbital_tiling import compute_tiled_orbitals, group_tiling_k_points, tiling_k_points
from jqmc.orbital_tiling_sources import select_tiling_sector, trexio_tiling_sectors
from tests.test_jqmc_kernel.test_orbital_tiling_hdf5 import _hamiltonian
from tests.test_jqmc_kernel.test_orbital_tiling_j3 import _supercell_j3
from tests.test_jqmc_kernel.test_qe_orbital_tiling import _synthetic_source

pytestmark = [pytest.mark.without_chdir]


@pytest.mark.light
@pytest.mark.parametrize("matrix", [(2, 1, 1), (2, 1, 0, 0, 2, 0, 0, 0, -1)])
def test_sector_partition_preserves_cosets_ids_and_weights(matrix):
    """Shuffled/reciprocally shifted input recovers two complete unequal-weight sectors."""
    twists = [(0, 0, 0), (0.23, 0.17, 0.31)]
    ks = [tiling_k_points(matrix, twist) for twist in twists]
    count = len(ks[0])
    points = np.concatenate(ks)
    weights = np.repeat([0.3 / count, 0.7 / count], count)
    permutation = np.random.default_rng(53).permutation(len(points))
    points = points[permutation] + np.random.default_rng(54).integers(-2, 3, points.shape)
    weights = weights[permutation]
    groups = group_tiling_k_points(matrix, points, weights)
    assert len(groups) == 2
    for twist, expected_weight in zip(twists, (0.3, 0.7), strict=True):
        group = select_tiling_sector(groups, twist)
        actual = points[list(group[1])] - tiling_k_points(matrix, twist)
        np.testing.assert_allclose(actual, np.rint(actual), atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT)
        np.testing.assert_allclose(group[2], expected_weight, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT)
        assert select_tiling_sector([group]) is group
    np.testing.assert_allclose(
        normalize_twist_weights([g[2] for g in groups]).sum(), 1, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT
    )
    with pytest.raises(ValueError, match="multiple supertwists"):
        select_tiling_sector(groups)
    with pytest.raises(ValueError, match="no complete"):
        select_tiling_sector(groups, (0.11, 0.22, 0.33))


@pytest.mark.light
@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "weights", "tiny_weights", "nan", "complex", "huge", "negative", "overflow"]
)
def test_sector_partition_rejects_invalid_mesh(invalid):
    """Incomplete or numerically invalid source sectors fail before conversion."""
    points = tiling_k_points((2, 1, 1))
    weights = np.array([0.5, 0.5])
    if invalid == "missing":
        points, weights = points[:1], weights[:1]
    elif invalid == "duplicate":
        points[1] = points[0]
    elif invalid == "weights":
        weights[1] = 0.4
    elif invalid == "tiny_weights":
        weights = np.array([1e-100, 2e-100])
    elif invalid == "nan":
        points[0, 0] = np.nan
    elif invalid == "complex":
        points = points.astype(complex) + 1j
    elif invalid == "huge":
        points[0, 0] = 2**54
    elif invalid == "negative":
        weights[0] = -1
    else:
        weights[:] = np.finfo(float).max
    with pytest.raises(ValueError):
        group_tiling_k_points((2, 1, 1), points, weights)


@pytest.mark.light
@pytest.mark.parametrize("corruption", ["missing", "length", "out_of_range", "uncovered_k"])
def test_trexio_tiling_rejects_invalid_mo_k_point_mapping(corruption, tmp_path):
    """Every k point in a multi-k TREXIO source owns an explicit nonempty MO block."""
    from jqmc.trexio_wrapper import read_trexio_kpoint_list

    source = Path(__file__).parents[1] / "trexio_example_files/lithium_twist_222.h5"
    damaged = tmp_path / "damaged_multik.h5"
    shutil.copy(source, damaged)
    with h5py.File(damaged, "r+") as handle:
        dataset = handle["mo/mo_k_point"]
        mapping = dataset[...]
        if corruption == "missing":
            del handle["mo/mo_k_point"]
        elif corruption == "length":
            del handle["mo/mo_k_point"]
            handle["mo"].create_dataset("mo_k_point", data=mapping[:-1])
        elif corruption == "out_of_range":
            mapping[0] = len(handle["pbc/pbc_k_point"])
            dataset[...] = mapping
        else:
            mapping[mapping == mapping.max()] = 0
            dataset[...] = mapping
    message = {
        "missing": "requires mo_k_point",
        "length": "consistently with mo_num",
        "out_of_range": "outside pbc_k_point range",
        "uncovered_k": "have no associated MO rows",
    }[corruption]
    with pytest.raises(ValueError, match=message):
        read_trexio_kpoint_list(str(damaged))


@pytest.mark.light
@pytest.mark.parametrize("source", ["qe", "trexio"])
def test_cli_tiling_selection_is_validated_before_source_io(source):
    """Both source CLIs reject inconsistent matrix and sector options before opening input."""
    cli, command = (
        (typer_click_qe, ["missing-source"]) if source == "qe" else (typer_click_trexio, ["convert-to", "missing-source"])
    )
    for options in (
        ["--tiling", "2 0 1"],
        ["--tiling", "2 1 1", "--supertwist", "nan 0 0"],
        ["--supertwist", "0 0 0"],
        ["--tiling", "2 1 1", "--supertwist", "0 0 0", "--twist-average"],
    ):
        result = CliRunner().invoke(cli, command + options)
        assert result.exit_code != 0
        assert "Invalid value" in result.output, result.output


@pytest.mark.light
@pytest.mark.parametrize("source", ["qe", "trexio"])
def test_workflow_transports_tiling_sector_and_ensemble_options(source):
    """Both Workflows retain one matrix argument and explicit sector/ensemble options."""
    from jqmc_workflow.qe_workflow import QE_Workflow
    from jqmc_workflow.wf_workflow import WF_Workflow

    cls, kwargs = (QE_Workflow, {"qe_save_dir": "/x", "kind": "pw"}) if source == "qe" else (WF_Workflow, {"trexio_file": "/x"})
    for options in ({"supertwist": [0.2, -0.1, 0]}, {"twist_average": True}):
        wf = cls(**kwargs, tiling=[2, 1, 1], **options)
        command = wf._build_command()
        assert command[command.index("--tiling") + 1] == "2 1 1"
        if "supertwist" in options:
            assert command[command.index("--supertwist") + 1] == "0.2 -0.1 0"
        else:
            assert "--twist-average" in command
        assert all(wf.configure()[key] == value for key, value in options.items())


@pytest.mark.light
@pytest.mark.parametrize("mode", ["gamma", "general", "average"])
def test_qe_tiling_sector_conversion_and_storage(mode, tmp_path, monkeypatch):
    """Synthetic source/projection plumbing exercises the production QE sector conversion."""
    from jqmc.jqmc_qe import run_qe2jqmc

    original, projections = _synthetic_source("bspline")
    twists = [(0, 0, 0)] if mode == "gamma" else [(0.23, 0.17, 0.31)]
    if mode == "average":
        twists = [(0, 0, 0), (0.23, 0.17, 0.31)]
    references = [
        replace(original[i], wfc=replace(original[i].wfc, k_point=k @ original[i].cell.b))
        for twist in twists
        for i, k in enumerate(tiling_k_points((2, 1, 1), twist))
    ]
    weights = [0.15, 0.15, 0.35, 0.35] if mode == "average" else [0.5, 0.5]
    monkeypatch.setattr("jqmc.qe_wrapper.qe_save_is_lsda", lambda path: False)
    monkeypatch.setattr("jqmc.qe_wrapper._find_save_dir", lambda path: str(tmp_path))
    monkeypatch.setattr(
        "jqmc.qe_wrapper.load_qe_kpoint_list", lambda path: list(zip([r.wfc.k_point for r in references], weights, strict=True))
    )
    monkeypatch.setattr("jqmc.qe_wrapper.load_qe_reference", lambda path, ik=1: references[ik - 1])
    monkeypatch.setattr("jqmc.pw_pgto_projection.taper_qe_reference", lambda ref, **kw: (ref, None))
    monkeypatch.setattr("jqmc.pw_pgto_projection.project_pw_to_spline", lambda ref, **kw: projections[0])
    path = str(tmp_path / "converted.h5")
    run_qe2jqmc(
        {"qe_save_dir": str(tmp_path), "kind": "bspline", "tiling": [2, 1, 1], "twist_average": mode == "average"},
        {"j3_basis_type": "none"},
        path,
    )
    if mode == "average":
        meta = load_twist_average_input(path)
        assert meta.n_twist == 2
        np.testing.assert_allclose(meta.weights, [0.3, 0.7], atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT)
    for index, twist in enumerate(twists):
        ham = load_twist(path, index) if mode == "average" else Hamiltonian_data.load_from_hdf5(path)
        sd = ham.wavefunction_data.geminal_data
        np.testing.assert_allclose(
            sd.orb_data_up_spin.supertwist, twist, atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT
        )
        assert sd.num_electron_up == 2 and ham.structure_data.natom == 2
        assert all(block.source_band_indices == (2,) for block in sd.orb_data_up_spin.blocks)


@pytest.mark.heavy
@pytest.mark.parametrize("mode", ["general", "average"])
def test_real_trexio_tiling_cli_keeps_supercell_j3(mode, tmp_path):
    """Actual PySCF TREXIO data passes CLI, GTO block storage and independent J3 construction."""
    source = Path(__file__).parents[1] / "trexio_example_files/lithium_twist_222.h5"
    sectors = trexio_tiling_sectors(str(source), (2, 1, 1))
    output = str(tmp_path / "trexio-tiled.h5")
    options = ["--twist-average"] if mode == "average" else ["--supertwist", " ".join(str(x) for x in sectors[1][0])]
    result = CliRunner().invoke(
        _cli, ["trexio", "convert-to", str(source), "-o", output, "--tiling", "2 1 1", "-j3", "ao-small", *options]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    selected = sectors if mode == "average" else [sectors[1]]
    for index, sector in enumerate(selected):
        ham = load_twist(output, index) if mode == "average" else Hamiltonian_data.load_from_hdf5(output)
        sd = ham.wavefunction_data.geminal_data
        np.testing.assert_allclose(
            sd.orb_data_up_spin.supertwist, sector[0], atol=ATOL_TWIST_CHECKPOINT_KPOINT_MATCH, rtol=RTOL_BIT_EXACT
        )
        assert ham.structure_data.natom == 4 and sd.num_electron_up == 2
        j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
        np.testing.assert_array_equal(j3.orb_data.structure_data.cell, ham.structure_data.cell)
        from jqmc.hamiltonians import _detile_aos_data_for_save

        assert set(_detile_aos_data_for_save(j3.orb_data).nucleus_index) == set(range(4))
        assert sd.orb_data_up_spin.blocks[0].source_band_indices != sd.orb_data_up_spin.blocks[1].source_band_indices


@pytest.mark.light
@pytest.mark.parametrize("corruption", ["none", "provider_class", "sector", "weight", "payload"])
def test_tiled_twist_payload_validation_and_nonzero_j3_roundtrip(corruption, tmp_path):
    """Ragged twist payloads retain nonzero J3 and aliases while rejecting corruption."""
    ham = _hamiltonian("pw", True)
    j3 = _supercell_j3(ham.structure_data)
    ham = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3)
        )
    )
    path = str(tmp_path / "twists.h5")
    with TwistAverageWriter(path, ham, 1) as writer:
        writer.write_tiled_twist(0, 2.0, ham.wavefunction_data.geminal_data)
    with h5py.File(path, "r+") as f:
        group = f["twists/twist_000"]
        assert group["geminal"].id == f["shared/wavefunction_data/geminal_data"].id
        assert "geminal_representation" not in group.attrs
        assert "pbc_schema_version" not in f["_meta"].attrs
        if corruption == "provider_class":
            group["geminal/orb_data_up_spin"].attrs["_class_name"] = "MOs_data"
        elif corruption == "sector":
            group["k_point"][0] += 0.25
        elif corruption == "weight":
            group["weight"][()] = 0.0
        elif corruption == "payload":
            del group["geminal"]
    if corruption != "none":
        with pytest.raises(ValueError):
            load_twist(path, 0)
        return
    restored = load_twist(path, 0)
    np.testing.assert_array_equal(restored.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix, j3.j_matrix)
    points = np.asarray(ham.structure_data.positions) + [0.23, 0.35, -0.12]
    original = ham.wavefunction_data.geminal_data.orb_data_up_spin
    loaded = restored.wavefunction_data.geminal_data.orb_data_up_spin
    atol, rtol = get_tolerance_min(("tiling_eval", "ao_eval"), "medium")
    np.testing.assert_allclose(
        compute_tiled_orbitals(loaded, points), compute_tiled_orbitals(original, points), atol=atol, rtol=rtol
    )
    for down in restored.wavefunction_data.geminal_data.orb_data_dn_spin.blocks:
        assert any(down.backend is up.backend and down.coefficients is up.coefficients for up in loaded.blocks)


@pytest.mark.light
@pytest.mark.parametrize("basis", ["cart", "sphe"])
def test_trexio_uhf_tiled_down_channel_is_conjugated_beta(basis):
    """Tiled TREXIO UHF: the positive down source is beta at +k and the down orbitals are conj(beta).

    KUHF oxygen (N_up = 4 > N_dn = 2 per k) in the general 2x1x1 sector
    k_s = (0.22, 0.13, 0.09). The ordinary TREXIO reader stores each down
    channel as conj(C_beta) on the -k leaf (pinned by
    PHY-TREXIO-LSDA-SPIN-CHANNELS-001). The tiled positive-sector down
    source must undo that once -- its rows are the +k beta rows, not their
    conjugate -- and the SD's -k_s down orbitals must equal the reader's
    conj(beta) orbitals (Bloch-extended to the supercell, with the 1/sqrt(N)
    tiling normalization) at every point.
    """
    from jqmc.molecular_orbital import compute_MOs
    from jqmc.orbital_tiling_sources import read_tiled_trexio_sector
    from jqmc.trexio_wrapper import read_trexio_file

    path = str(Path(__file__).parents[1] / "trexio_example_files" / f"oxygen_tiling_sectors_{basis}.h5")
    sector = next(s for s in trexio_tiling_sectors(path, (2, 1, 1)) if np.any(np.asarray(s[0]) != 0))
    _, _, _, down, sd, _ = read_tiled_trexio_sector(path, (2, 1, 1), sector, expand_j3_basis=False)
    assert not sd.time_reversal and (sd.num_electron_up, sd.num_electron_dn) == (8, 4)
    reader = {index: read_trexio_file(path, store_tuple=True, twist_ik=index)[3] for index in sector[1]}
    rows_by_k = {tuple(np.round(np.asarray(m.aos_data.k_point), 12)): m for m in reader.values()}
    points = np.asarray(sd.orb_data_up_spin.structure_data.positions)[[0, 0, 1, 1]] + [
        [0.41, -0.22, 0.35],
        [-0.63, 0.18, 0.27],
        [0.12, 0.57, -0.44],
        [0.71, 0.33, 0.58],
    ]
    expected_rows = []
    for block in down.blocks:
        # A conjugated block reports k = -k_source; its reader leaf sits at that -k.
        mos = rows_by_k[tuple(np.round(np.asarray(block.k_point), 12))]
        np.testing.assert_array_equal(np.asarray(block.coefficients), np.conj(np.asarray(mos.mo_coefficients)))
        assert not np.allclose(
            np.asarray(block.coefficients),
            np.asarray(mos.mo_coefficients),
            atol=ATOL_NONVACUITY_DISTINCTION,
            rtol=RTOL_NONVACUITY_DISTINCTION,
        )
        expected_rows.append(np.asarray(compute_MOs(mos, points)) / np.sqrt(len(down.blocks)))
    atol, rtol = get_tolerance_min(("ao_eval", "ao_bloch_eval", "mo_eval", "tiling_eval"), "strict")
    np.testing.assert_allclose(
        np.asarray(compute_tiled_orbitals(down, points)), np.concatenate(expected_rows), atol=atol, rtol=rtol
    )
