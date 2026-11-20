"""CLI job == directly constructed driver with the same non-default TOML values.

The restart-equality nodes (test_cli_restart_values.py) run the TOML -> driver
plumbing on both sides of the comparison, so a dropped or miswired key (tau,
alat, E_scf, non_local_move, epsilon_PW, mpc, ...) is not detected there. Here
each job type is run once through ``jqmc_cli._cli()`` with NON-default values
and once through the public Python API with the same values, the same
HDF5 Hamiltonian and the same seed, both on the XLA backend.

Verdicts (deterministic, same computation on both sides): checkpoint walker
positions and threefry keys are exact; stored e_L / e_L2 / w_L (and, where
requested, force / stress / MPC buffers) and the public energy tuple agree at
the same-mode strict restart tolerance; for vmc the optimized J2 parameter too.

Rows: molecular H2 ccECP cc-pVDZ Cartesian with exp J1 + Pade J2 for mcmc,
vmc, lrdmc-bra and lrdmc-tau (dltmove, so the non-local move key matters), and
the periodic lithium.h5 cell (Li ccECP, J1 + J2) for mcmc and lrdmc-tau with
[coulomb] mpc = true, atomic forces, stress and a non-default epsilon_PW.
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
import logging
from pathlib import Path
import sys

import numpy as np
import pytest
import toml

from jqmc import jqmc_cli
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
_SEED = 24680
_WALKERS = 2
_RESTART_ZONES = ("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy")
_MCMC_STEPS = 12
_MCMC_ANALYSIS = {"num_mcmc_warmup_steps": 2, "num_mcmc_bin_blocks": 2}
_GFMC_ANALYSIS = {"num_mcmc_warmup_steps": GFMC_MIN_WARMUP_STEPS, "num_mcmc_bin_blocks": GFMC_MIN_BIN_BLOCKS}

# Non-default values per job type (the shipped defaults are in jqmc_miscs.cli_parameters).
_JOB_VALUES = {
    "mcmc": dict(
        num_mcmc_steps=_MCMC_STEPS,
        num_mcmc_per_measurement=3,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        Dt=1.7,
        epsilon_AS=1.0e-5,
    ),
    "vmc": dict(
        num_mcmc_steps=_MCMC_STEPS,
        num_mcmc_per_measurement=2,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        Dt=1.7,
        epsilon_AS=1.0e-5,
        num_opt_steps=1,
    ),
    "lrdmc-bra": dict(
        num_mcmc_steps=GFMC_MIN_STEPS,
        num_mcmc_per_measurement=3,
        alat=0.25,
        non_local_move="dltmove",
        num_gfmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
        num_gfmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        E_scf=-1.3,
    ),
    "lrdmc-tau": dict(
        num_mcmc_steps=GFMC_MIN_STEPS,
        tau=0.07,
        alat=0.25,
        non_local_move="dltmove",
        num_gfmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
        num_gfmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
    ),
}
_VMC_OPTIMIZER = {"method": "sr", "delta": 0.002, "epsilon": 0.1, "cg_flag": False, "use_lm": False}
# Periodic rows add forces, stress, SWCT, a non-default epsilon_PW and MPC.
_PERIODIC_VALUES = dict(atomic_force=True, compute_stress=True, use_swct=True)
_PERIODIC_EPSILON_PW = 1.0
_PERIODIC_MCMC_DT = 1.5


def _write_hamiltonian(root: Path, system: str) -> Path:
    stem = {"molecule": "H2_ecp_ccpvdz_cart.h5", "periodic": "lithium.h5"}[system]
    structure, _aos, _mos_up, _mos_dn, geminal, coulomb = read_trexio_file(str(_TREXIO_DIR / stem), store_tuple=True)
    jastrow = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure,
            core_electrons=core_electrons_of(coulomb, len(structure.atomic_numbers)),
            jastrow_1b_type="exp",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
        ),
        jastrow_three_body_data=None,
    )
    hamiltonian = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal),
    )
    path = root / "hamiltonian.h5"
    hamiltonian.save_to_hdf5(str(path))
    return path


def _job_values(job_type: str, system: str) -> dict:
    values = dict(_JOB_VALUES[job_type])
    if system == "periodic":
        values.update(_PERIODIC_VALUES)
        if job_type == "mcmc":
            values["Dt"] = _PERIODIC_MCMC_DT
        else:
            values["epsilon_PW"] = _PERIODIC_EPSILON_PW
    return values


def _run_cli(root: Path, hamiltonian_h5: Path, job_type: str, system: str, monkeypatch) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters[job_type])
    params.update(_job_values(job_type, system))
    if job_type == "vmc":
        for key in params:
            if key.startswith("opt_"):
                params[key] = key == "opt_J2_param"
        params["optimizer_kwargs"] = dict(_VMC_OPTIMIZER)
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": _SEED,
            "number_of_walkers": _WALKERS,
            "hamiltonian_h5": str(hamiltonian_h5),
            "restart": False,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: params,
    }
    if system == "periodic":
        payload["coulomb"] = {"mpc": True}
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload))
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    checkpoint = str(root / "restart.h5")
    driver_class = {"mcmc": MCMC, "vmc": MCMC, "lrdmc-bra": GFMC_n, "lrdmc-tau": GFMC_t}[job_type]
    return {
        "state": load_rank_checkpoint(checkpoint, rank=0),
        "driver": driver_class.load_from_hdf5(checkpoint),
        "hamiltonian": load_hamiltonian_from_checkpoint(checkpoint),
    }


def _run_direct(root: Path, hamiltonian_h5: Path, job_type: str, system: str, monkeypatch):
    values = _job_values(job_type, system)
    periodic = system == "periodic"
    hamiltonian = Hamiltonian_data.load_from_hdf5(str(hamiltonian_h5))
    root.mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as patch:
        patch.chdir(root)
        if job_type in ("mcmc", "vmc"):
            driver = MCMC(
                hamiltonian_data=hamiltonian,
                Dt=values["Dt"],
                mcmc_seed=_SEED,
                num_walkers=_WALKERS,
                num_mcmc_per_measurement=values["num_mcmc_per_measurement"],
                epsilon_AS=values["epsilon_AS"],
                comput_position_deriv=bool(values.get("atomic_force", False)),
                comput_stress=bool(values.get("compute_stress", False)),
                comput_log_WF_param_deriv=job_type == "vmc",
                comput_e_L_param_deriv=False,
                use_swct=bool(values.get("use_swct", True)),
                mpc=periodic,
            )
            if job_type == "mcmc":
                driver.run(num_mcmc_steps=values["num_mcmc_steps"])
            else:
                driver.run_optimize(
                    num_mcmc_steps=values["num_mcmc_steps"],
                    num_opt_steps=values["num_opt_steps"],
                    num_mcmc_warmup_steps=values["num_mcmc_warmup_steps"],
                    num_mcmc_bin_blocks=values["num_mcmc_bin_blocks"],
                    opt_J1_param=False,
                    opt_J2_param=True,
                    opt_J3_param=False,
                    opt_geminal=False,
                    optimizer_kwargs=dict(_VMC_OPTIMIZER),
                    use_device_optimizer=False,
                )
            return driver
        common = dict(
            hamiltonian_data=hamiltonian,
            mpc=periodic,
            num_walkers=_WALKERS,
            num_gfmc_collect_steps=values["num_gfmc_collect_steps"],
            mcmc_seed=_SEED,
            alat=values["alat"],
            non_local_move=values["non_local_move"],
            comput_position_deriv=bool(values.get("atomic_force", False)),
            comput_stress=bool(values.get("compute_stress", False)),
            epsilon_PW=values.get("epsilon_PW", 0.0),
            use_swct=bool(values.get("use_swct", False)),
        )
        if job_type == "lrdmc-bra":
            driver = GFMC_n(**common, num_mcmc_per_measurement=values["num_mcmc_per_measurement"], E_scf=values["E_scf"])
        else:
            driver = GFMC_t(**common, tau=values["tau"])
        driver.run(num_mcmc_steps=values["num_mcmc_steps"])
    return driver


def _exact(actual, reference, label):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape and actual.size > 0, f"{label}: shape {actual.shape} vs {reference.shape}"
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), f"{label}: operand not finite"
    np.testing.assert_array_equal(actual, reference, err_msg=label)


def _close(actual, reference, label):
    atol, rtol = get_tolerance_min(_RESTART_ZONES, "strict")
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape and actual.size > 0, f"{label}: shape {actual.shape} vs {reference.shape}"
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _direct_walk(driver, job_type):
    if job_type in ("mcmc", "vmc"):
        return (
            driver._MCMC__latest_r_up_carts,
            driver._MCMC__latest_r_dn_carts,
            driver._MCMC__jax_PRNG_key_list,
        )
    return driver._latest_r_up_carts, driver._latest_r_dn_carts, driver._jax_PRNG_key_list


def _observable_names(job_type, system):
    if job_type == "vmc":
        return ()
    names = ["e_L", "e_L2", "w_L"]
    if system == "periodic":
        if job_type == "mcmc":
            names += ["force_HF_stored", "force_PP_stored", "stress_HF_stored", "stress_PP_stored", "mpc_dV"]
        else:
            names += ["force_HF", "force_PP", "stress_HF", "stress_PP", "mpc_dV"]
    return tuple(names)


def _assert_cli_equals_direct(cli, direct, job_type, system):
    state = cli["state"]
    r_up, r_dn, keys = _direct_walk(direct, job_type)
    _exact(state["walker_state"]["latest_r_up_carts"], r_up, f"{job_type} CLI vs direct: r_up")
    _exact(state["walker_state"]["latest_r_dn_carts"], r_dn, f"{job_type} CLI vs direct: r_dn")
    _exact(state["rng_state"]["jax_PRNG_key_list"], keys, f"{job_type} CLI vs direct: keys")
    if job_type == "vmc":  # an optimizer run has no stored observables; its verdict is the updated J2
        _close(
            cli["hamiltonian"].wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            direct.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            "vmc CLI vs direct: J2",
        )
        return
    for name in _observable_names(job_type, system):
        _close(getattr(cli["driver"], name), getattr(direct, name), f"{job_type} CLI vs direct: {name}")
    analysis = _GFMC_ANALYSIS if job_type.startswith("lrdmc") else _MCMC_ANALYSIS
    _close(cli["driver"].get_E(**analysis), direct.get_E(**analysis), f"{job_type} CLI vs direct: E")
    if system == "periodic":
        _close(cli["driver"].get_aF(**analysis), direct.get_aF(**analysis), f"{job_type} CLI vs direct: aF")
        _close(cli["driver"].get_aT(**analysis), direct.get_aT(**analysis), f"{job_type} CLI vs direct: aT")


_ROWS = [
    pytest.param("mcmc", "molecule", id="mcmc-molecule"),
    pytest.param("vmc", "molecule", id="vmc-molecule"),
    pytest.param("lrdmc-bra", "molecule", id="lrdmc-bra-molecule"),
    pytest.param("lrdmc-tau", "molecule", id="lrdmc-tau-molecule"),
    pytest.param("mcmc", "periodic", id="mcmc-periodic"),
    pytest.param("lrdmc-tau", "periodic", id="lrdmc-tau-periodic"),
]


@pytest.mark.parametrize("job_type,system", _ROWS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_job_equals_direct_driver_with_the_same_toml_values(tmp_path, monkeypatch, job_type, system):
    """The CLI job equals the directly constructed driver fed the same non-default TOML values.

    Values differ from the shipped defaults: mcmc/vmc Dt 1.7 (periodic 1.5),
    epsilon_AS 1e-5, 3 (vmc 2) moves per measurement; vmc one SR step on J2
    with delta 0.002, epsilon 0.1; lrdmc-bra alat 0.25, dltmove, E_scf -1.3,
    3 projections per measurement; lrdmc-tau tau 0.07, alat 0.25, dltmove;
    periodic rows also mpc, forces, stress, SWCT and epsilon_PW 1.0 bohr. Seed
    24680, 2 walkers. Walk and keys exact; buffers, energies (and periodic
    force/stress estimates, vmc J2) at the strict restart tolerance. A
    dropped key in the CLI branch changes the walk or the stored values.
    """
    hamiltonian_h5 = _write_hamiltonian(tmp_path, system)
    cli = _run_cli(tmp_path / "cli", hamiltonian_h5, job_type, system, monkeypatch)
    direct = _run_direct(tmp_path / "direct", hamiltonian_h5, job_type, system, monkeypatch)
    _assert_cli_equals_direct(cli, direct, job_type, system)


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_equivalence_rejects_a_changed_driver_value(tmp_path, monkeypatch):
    """A direct lrdmc-tau run with the shipped tau instead of the TOML tau fails the CLI equivalence verdict."""
    hamiltonian_h5 = _write_hamiltonian(tmp_path, "molecule")
    cli = _run_cli(tmp_path / "cli", hamiltonian_h5, "lrdmc-tau", "molecule", monkeypatch)
    direct = _run_direct(tmp_path / "direct", hamiltonian_h5, "lrdmc-tau", "molecule", monkeypatch)
    _assert_cli_equals_direct(cli, direct, "lrdmc-tau", "molecule")
    shipped_tau = float(jqmc_cli.cli_parameters["lrdmc-tau"]["tau"])
    assert shipped_tau != _JOB_VALUES["lrdmc-tau"]["tau"]
    with monkeypatch.context() as patch:
        patch.setitem(_JOB_VALUES, "lrdmc-tau", {**_JOB_VALUES["lrdmc-tau"], "tau": shipped_tau})
        changed = _run_direct(tmp_path / "changed", hamiltonian_h5, "lrdmc-tau", "molecule", monkeypatch)
    with pytest.raises(AssertionError, match="lrdmc-tau CLI vs direct"):
        _assert_cli_equals_direct(cli, changed, "lrdmc-tau", "molecule")
