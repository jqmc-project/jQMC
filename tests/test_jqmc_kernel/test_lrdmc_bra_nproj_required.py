"""lrdmc-bra: ``num_mcmc_per_measurement`` (GFMC_n projections per measurement) is a required input (#690).

The number of GFMC projections between two measurements sets the imaginary time
per measurement (about ``num_mcmc_per_measurement * alat**2 / (3 N_e)``), the
walker survival per branching and with it the warmup/collect windows and the
cost. It depends on the system and on ``alat``, so it has no default and no
``alat``-derived value: a fresh run without it stops before sampling with an
error naming the key (CLI single-twist and twist-averaged routes, Python API
``GFMC_n`` / ``_GFMC_n_debug`` and ``run_twist_average_gfmc``). An explicit
value is used unchanged, and a single-twist restart may omit it (the
checkpoint's frozen value is kept; a different explicit value is refused by the
frozen-key guard).

The CLI cases stop at the driver boundary: ``GFMC_n`` (or the twist driver)
is replaced by a recorder, so no Hamiltonian is evaluated and no electron is
moved.
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

import copy
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import tomlkit

import jqmc.jqmc_cli as jqmc_cli
from jqmc.jqmc_miscs import cli_parameters

pytestmark = [
    pytest.mark.with_chdir,
    pytest.mark.gdd_electron_configuration(
        mode="not-required",
        reason="TOML-to-driver argument resolution; the driver is replaced or refused before any electron is moved",
    ),
]

_REQUIRED_MESSAGE = "num_mcmc_per_measurement ([lrdmc-bra], GFMC projections per measurement) is required"


class _StopAtDriver(Exception):
    """Raised by the recorder once the driver arguments are captured."""


def _write_toml(path: Path, control: dict, section: dict) -> None:
    doc = tomlkit.document()
    doc.add("control", tomlkit.item(control))
    doc.add("lrdmc-bra", tomlkit.item(section))
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _install_gfmc_n_recorder(monkeypatch, captured: dict) -> None:
    class _RecordingGFMCn:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _StopAtDriver

    monkeypatch.setattr(jqmc_cli, "GFMC_n", _RecordingGFMCn)
    monkeypatch.setattr(jqmc_cli.Hamiltonian_data, "load_from_hdf5", staticmethod(lambda _path: object()))


def _single_hamiltonian_h5(tmp_path) -> str:
    """An HDF5 file without the combined twist layout, so the CLI takes the single-twist route.

    ``Hamiltonian_data.load_from_hdf5`` is stubbed by the callers; only the
    twist-dispatch probe opens this file.
    """
    path = tmp_path / "hamiltonian_data.h5"
    with h5py.File(path, "w"):
        pass
    return str(path)


def _write_fresh_input(tmp_path, monkeypatch, section: dict) -> None:
    toml_path = tmp_path / "lrdmc.toml"
    control = {"job_type": "lrdmc-bra", "hamiltonian_h5": _single_hamiltonian_h5(tmp_path)}
    _write_toml(toml_path, control, section)
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])


def _fresh_section(**extra) -> dict:
    section = {"num_mcmc_steps": 100, "E_scf": -17.0}
    section.update(extra)
    return section


@pytest.mark.light
@pytest.mark.parametrize("alat", [None, 0.30, 0.10])
def test_cli_fresh_run_without_key_stops_before_the_driver(tmp_path, monkeypatch, capsys, alat):
    """A fresh lrdmc-bra TOML without num_mcmc_per_measurement exits non-zero naming the key; GFMC_n is never built.

    ``alat = None`` omits ``alat`` too: no lattice spacing (default or given)
    supplies a projection count.
    """
    captured: dict = {}
    _install_gfmc_n_recorder(monkeypatch, captured)
    section = _fresh_section() if alat is None else _fresh_section(alat=alat)
    _write_fresh_input(tmp_path, monkeypatch, section)
    with pytest.raises(SystemExit) as excinfo:
        jqmc_cli._cli()
    assert excinfo.value.code == 1
    assert captured == {}
    assert "num_mcmc_per_measurement should be specified." in capsys.readouterr().out


@pytest.mark.light
@pytest.mark.parametrize(("alat", "explicit"), [(0.10, 30), (0.10, 40), (0.30, 7), (0.20, 1)])
def test_cli_explicit_value_is_used_unchanged(tmp_path, monkeypatch, alat, explicit):
    """An explicit num_mcmc_per_measurement reaches GFMC_n as written, whatever alat is."""
    captured: dict = {}
    _install_gfmc_n_recorder(monkeypatch, captured)
    _write_fresh_input(tmp_path, monkeypatch, _fresh_section(alat=alat, num_mcmc_per_measurement=explicit))
    with pytest.raises(_StopAtDriver):
        jqmc_cli._cli()
    assert captured["alat"] == alat
    assert captured["num_mcmc_per_measurement"] == explicit


@pytest.mark.light
@pytest.mark.parametrize("toml_value", [None, 30, 31])
def test_cli_restart_keeps_checkpoint_value(tmp_path, monkeypatch, toml_value):
    """A restart does not need the key: the checkpoint's frozen value is kept; the guard is unchanged.

    The checkpoint was written with 30 projections at alat = 0.10. An omitted
    key restarts with 30 (no "should be specified" exit), an equal explicit
    value restarts with 30, and a different explicit value is refused by the
    frozen-key guard as before.
    """
    import jqmc._checkpoint as checkpoint

    ckpt_cfg = {
        "num_mcmc_per_measurement": 30,
        "alat": 0.10,
        "non_local_move": "tmove",
        "num_gfmc_collect_steps": 5,
        "E_scf": -17.2,
    }
    restored: dict = {}

    class _LoadedDriver:
        _num_mcmc_per_measurement = ckpt_cfg["num_mcmc_per_measurement"]

    class _RecordingGFMCn:
        def __init__(self, **kwargs):
            raise AssertionError("a restart must not construct a fresh GFMC_n")

        @classmethod
        def load_from_hdf5(cls, filepath, rank=0):
            restored["driver"] = _LoadedDriver()
            raise _StopAtDriver

    monkeypatch.setattr(jqmc_cli, "GFMC_n", _RecordingGFMCn)
    monkeypatch.setattr(checkpoint, "_validate_restart_rank_topology", lambda *_a, **_k: None)
    monkeypatch.setattr(checkpoint, "load_driver_config_from_checkpoint", lambda *_a, **_k: dict(ckpt_cfg))
    section = {"num_mcmc_steps": 100, "alat": 0.10}
    if toml_value is not None:
        section["num_mcmc_per_measurement"] = toml_value
    toml_path = tmp_path / "lrdmc.toml"
    control = {
        "job_type": "lrdmc-bra",
        "hamiltonian_h5": _single_hamiltonian_h5(tmp_path),
        "restart": True,
        "restart_chk": str(tmp_path / "restart.h5"),
    }
    _write_toml(toml_path, control, section)
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

    if toml_value == 31:
        with pytest.raises(ValueError, match="num_mcmc_per_measurement"):
            jqmc_cli._cli()
        assert "driver" not in restored
        return
    with pytest.raises(_StopAtDriver):
        jqmc_cli._cli()
    assert restored["driver"]._num_mcmc_per_measurement == 30


@pytest.fixture
def combined_twist_file(tmp_path):
    """Two-twist combined input of the lithium fixture, loadable by the twist dispatch."""
    from tests.test_jqmc_kernel.test_cli_twist_dispatch import _write_combined_twist_input

    return _write_combined_twist_input(tmp_path, 2)


@pytest.mark.light
@pytest.mark.parametrize(("explicit", "restart"), [(None, False), (None, True), (30, False), (30, True)])
def test_twist_dispatch_requires_the_key(combined_twist_file, tmp_path, monkeypatch, explicit, restart):
    """The twist-averaged lrdmc-bra route refuses a TOML without the key and forwards an explicit value unchanged.

    A twist-averaged resume re-runs every incomplete twist from the TOML
    settings (it has no checkpoint-driven continuation of a sampler), so the
    key is required there as well; this matches every other key of the route
    that has no default.
    """
    import jqmc._twist_average as twist_average

    captured: dict = {}

    def fake_gfmc(**kwargs):
        captured.update(kwargs)
        raise _StopAtDriver

    monkeypatch.setattr(twist_average, "run_twist_average_gfmc", fake_gfmc)
    section = {"num_mcmc_steps": 100, "alat": 0.10, "E_scf": -0.5}
    if explicit is not None:
        section["num_mcmc_per_measurement"] = explicit
    kwargs = dict(
        job_type="lrdmc-bra",
        hamiltonian_h5=combined_twist_file,
        parameters=copy.deepcopy(cli_parameters),
        dict_toml={"lrdmc-bra": section},
        mcmc_seed=1,
        number_of_walkers=1,
        mpc=False,
        mpc_cutoff=0.0,
        max_time=60,
        restart=restart,
        restart_chk=str(tmp_path / "twist_checkpoints"),
        mpi_comm=None,
        mpi_rank=0,
    )
    if explicit is None:
        with pytest.raises(ValueError, match=r"required key \[lrdmc-bra\]\.num_mcmc_per_measurement"):
            jqmc_cli._run_twist_average_dispatch(**kwargs)
        assert captured == {}
        return
    with pytest.raises(_StopAtDriver):
        jqmc_cli._run_twist_average_dispatch(**kwargs)
    assert captured["gfmc_kwargs"]["alat"] == 0.10
    assert captured["gfmc_kwargs"]["num_mcmc_per_measurement"] == explicit


@pytest.mark.light
@pytest.mark.parametrize("exclude_comment", [False, True])
def test_generated_template_marks_the_key_required(tmp_path, exclude_comment):
    """jqmc-tool lrdmc generate-input writes the key as a required placeholder, like num_mcmc_steps."""
    import toml as _toml

    from jqmc.jqmc_tool import lrdmc_generate_input

    assert cli_parameters["lrdmc-bra"]["num_mcmc_per_measurement"] is None
    path = tmp_path / "lrdmc.toml"
    lrdmc_generate_input(flag=True, filename=str(path), exclude_comment=exclude_comment)
    text = path.read_text(encoding="utf-8")
    section = _toml.loads(text)["lrdmc-bra"]
    assert section["num_mcmc_per_measurement"] == section["num_mcmc_steps"] == "None"
    assert section["alat"] == 0.30
    assert "max(1, round(40" not in text
    assert ("Required, no default" in text) is (not exclude_comment)


@pytest.fixture(scope="module")
def h_ae_hamiltonian():
    """Committed all-electron H atom (cc-pVDZ, Cartesian), no Jastrow: enough to build a GFMC_n driver."""
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    fixture = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
    assert fixture.is_file(), f"committed fixture missing: {fixture}"
    sd, _, _, _, gem, cpd = read_trexio_file(str(fixture), store_tuple=True)
    return Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=gem),
    )


@pytest.mark.light
@pytest.mark.parametrize("driver_name", ["GFMC_n", "_GFMC_n_debug"])
@pytest.mark.parametrize("alat", [0.30, 0.10])
@pytest.mark.parametrize("given", ["omitted", "passed_none"])
def test_python_api_without_value_is_refused(h_ae_hamiltonian, driver_name, alat, given):
    """GFMC_n / _GFMC_n_debug without num_mcmc_per_measurement (or with None) raise ValueError naming the key."""
    from jqmc import jqmc_gfmc

    driver_class = getattr(jqmc_gfmc, driver_name)
    kwargs = {"hamiltonian_data": h_ae_hamiltonian, "num_walkers": 1, "mcmc_seed": 12345, "alat": alat}
    if given == "passed_none":
        kwargs["num_mcmc_per_measurement"] = None
    with pytest.raises(ValueError, match=re.escape(_REQUIRED_MESSAGE)):
        driver_class(**kwargs)


@pytest.mark.light
@pytest.mark.parametrize("driver_name", ["GFMC_n", "_GFMC_n_debug"])
@pytest.mark.parametrize(("alat", "explicit"), [(0.10, 16), (0.10, 30), (0.30, 7), (0.20, 1)])
def test_python_api_explicit_value_is_used_unchanged(h_ae_hamiltonian, driver_name, alat, explicit):
    """An explicit num_mcmc_per_measurement is stored exactly as given, whatever alat is."""
    from jqmc import jqmc_gfmc

    driver_class = getattr(jqmc_gfmc, driver_name)
    driver = driver_class(
        hamiltonian_data=h_ae_hamiltonian, num_walkers=1, mcmc_seed=12345, alat=alat, num_mcmc_per_measurement=explicit
    )
    assert driver._num_mcmc_per_measurement == explicit


@pytest.mark.light
@pytest.mark.parametrize("gfmc_kind", ["n", "n_debug"])
def test_twist_average_gfmc_without_value_is_refused(h_ae_hamiltonian, monkeypatch, gfmc_kind):
    """run_twist_average_gfmc for GFMC_n refuses gfmc_kwargs without the key before any driver is built."""
    from jqmc import jqmc_gfmc
    from jqmc._twist_average import run_twist_average_gfmc

    class _MustNotBuild:
        def __init__(self, **kwargs):
            raise AssertionError("no GFMC_n driver may be built without num_mcmc_per_measurement")

    monkeypatch.setattr(jqmc_gfmc, "GFMC_n", _MustNotBuild)
    monkeypatch.setattr(jqmc_gfmc, "_GFMC_n_debug", _MustNotBuild)
    with pytest.raises(ValueError, match=re.escape(_REQUIRED_MESSAGE)):
        run_twist_average_gfmc(
            hamiltonian_list=[h_ae_hamiltonian],
            weights=[1.0],
            gfmc_kind=gfmc_kind,
            num_mcmc_steps=40,
            num_walkers=1,
            gfmc_kwargs={"alat": 0.30, "E_scf": -0.5},
        )
