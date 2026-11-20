"""Deterministic actual CLI versus public two-twist orchestrator equivalence."""

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

import h5py
from mpi4py import MPI
import numpy as np
import pytest
import toml

from jqmc import jqmc_cli
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import is_mixed_precision_enabled
from jqmc._twist_average import (
    load_twist,
    run_twist_average_from_hamiltonians,
    run_twist_average_gfmc,
    run_twist_average_optimize,
    save_twist_average_input,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file, read_trexio_kpoint_list
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel.test_cli_driver_equivalence import (
    _GFMC_ANALYSIS,
    _JOB_VALUES,
    _MCMC_ANALYSIS,
    _SEED,
    _TREXIO_DIR,
    _VMC_OPTIMIZER,
    _WALKERS,
    _close,
    _exact,
)

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir, pytest.mark.mpi_only]
_WEIGHTS = [0.25, 0.75]
_FIXTURE = _TREXIO_DIR / "lithium_twist_222.h5"


def _write_twist_input(path):
    kpoints = read_trexio_kpoint_list(str(_FIXTURE))
    orbitals_up, orbitals_down = [], []
    template = None
    for index in range(2):
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(_FIXTURE), store_tuple=True, twist_ik=index)
        if template is None:
            jastrow = Jastrow_data(
                jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
                )
            )
            template = Hamiltonian_data(
                structure_data=structure,
                coulomb_potential_data=coulomb,
                wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=jastrow),
            )
        orbitals_up.append(np.asarray(geminal.orb_data_up_spin.mo_coefficients))
        orbitals_down.append(np.asarray(geminal.orb_data_dn_spin.mo_coefficients))
    save_twist_average_input(
        str(path),
        template,
        [kpoints[0][0], kpoints[1][0]],
        _WEIGHTS,
        orbitals_up,
        orbitals_down,
    )


def _run_actual_cli(root, source, job_type, monkeypatch):
    comm = MPI.COMM_WORLD
    parameters = copy.deepcopy(jqmc_cli.cli_parameters[job_type])
    parameters.update(_JOB_VALUES[job_type])
    if job_type == "vmc":
        parameters.update({name: name == "opt_J2_param" for name in parameters if name.startswith("opt_")})
        parameters["optimizer_kwargs"] = dict(_VMC_OPTIMIZER)
    else:
        parameters["use_swct"] = False
    config = {
        "control": {
            "job_type": job_type,
            "hamiltonian_h5": str(source),
            "restart": False,
            "restart_chk": str(root / "checkpoint"),
            "mcmc_seed": _SEED,
            "number_of_walkers": _WALKERS,
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: parameters,
    }
    if comm.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / "input.toml").write_text(toml.dumps(config))
    comm.Barrier()
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(root / "input.toml")])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    comm.Barrier()


def _run_direct_twist(root, source, job_type, monkeypatch):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    hamiltonians = [load_twist(str(source), index) for index in range(2)]
    values = _JOB_VALUES[job_type]
    common = dict(
        hamiltonian_list=hamiltonians,
        weights=_WEIGHTS,
        mcmc_seed=_SEED,
        num_walkers=_WALKERS,
        world_comm=comm,
        checkpoint_dir=str(root / "checkpoint"),
        resume=False,
    )
    with monkeypatch.context() as patch:
        patch.chdir(root)
        if job_type == "mcmc":
            run_twist_average_from_hamiltonians(
                **common,
                num_mcmc_steps=values["num_mcmc_steps"],
                num_mcmc_warmup_steps=values["num_mcmc_warmup_steps"],
                num_mcmc_bin_blocks=values["num_mcmc_bin_blocks"],
                compute_force=False,
                compute_stress=False,
                mcmc_kwargs={
                    "Dt": values["Dt"],
                    "epsilon_AS": values["epsilon_AS"],
                    "num_mcmc_per_measurement": values["num_mcmc_per_measurement"],
                    "use_swct": False,
                },
            )
        elif job_type == "vmc":
            run_twist_average_optimize(
                **common,
                source_hamiltonian_path=str(source),
                mcmc_kwargs={
                    "Dt": values["Dt"],
                    "epsilon_AS": values["epsilon_AS"],
                    "num_mcmc_per_measurement": values["num_mcmc_per_measurement"],
                },
                optimize_kwargs={
                    "num_mcmc_steps": values["num_mcmc_steps"],
                    "num_opt_steps": values["num_opt_steps"],
                    "wf_dump_freq": 1,
                    "num_mcmc_warmup_steps": values["num_mcmc_warmup_steps"],
                    "num_mcmc_bin_blocks": values["num_mcmc_bin_blocks"],
                    "opt_J1_param": False,
                    "opt_J2_param": True,
                    "opt_J3_param": False,
                    "opt_geminal": False,
                    "opt_J3_basis_exp": False,
                    "opt_J3_basis_coeff": False,
                    "opt_lambda_basis_exp": False,
                    "opt_lambda_basis_coeff": False,
                    "opt_JDTN_u_param": False,
                    "opt_JDTN_chi_param": False,
                    "opt_JDTN_f_param": False,
                    "optimizer_kwargs": dict(_VMC_OPTIMIZER),
                    "use_device_optimizer": False,
                },
            )
        else:
            kwargs = {
                "alat": values["alat"],
                "non_local_move": values["non_local_move"],
                "num_gfmc_collect_steps": values["num_gfmc_collect_steps"],
                "use_swct": False,
            }
            if job_type == "lrdmc-bra":
                kwargs.update(E_scf=values["E_scf"], num_mcmc_per_measurement=values["num_mcmc_per_measurement"])
            else:
                kwargs["tau"] = values["tau"]
            run_twist_average_gfmc(
                **common,
                gfmc_kind="n" if job_type == "lrdmc-bra" else "tau",
                num_mcmc_steps=values["num_mcmc_steps"],
                num_mcmc_warmup_steps=values["num_gfmc_warmup_steps"],
                num_mcmc_bin_blocks=values["num_gfmc_bin_blocks"],
                gfmc_kwargs=kwargs,
                compute_force=False,
                compute_stress=False,
            )
    comm.Barrier()


def _assert_twist_checkpoint_equivalence(cli_path, direct_path, source, job_type, index):
    actual = load_rank_checkpoint(str(cli_path), rank=0)
    reference = load_rank_checkpoint(str(direct_path), rank=0)
    for key in ("latest_r_up_carts", "latest_r_dn_carts"):
        _exact(actual["walker_state"][key], reference["walker_state"][key], f"{job_type} twist {index}: {key}")
    _exact(actual["rng_state"]["jax_PRNG_key_list"], reference["rng_state"]["jax_PRNG_key_list"], "twist RNG")
    if job_type in ("mcmc", "vmc", "lrdmc-bra"):
        requested = _JOB_VALUES[job_type]["num_mcmc_per_measurement"]
        _exact(actual["driver_config"]["num_mcmc_per_measurement"], requested, "CLI requested measurement count")
        _exact(reference["driver_config"]["num_mcmc_per_measurement"], requested, "direct requested measurement count")
    with h5py.File(cli_path, "r") as cli, h5py.File(direct_path, "r") as direct:
        _exact(cli["_meta"].attrs["normalized_twist_weight"], _WEIGHTS[index], "CLI twist weight")
        _exact(direct["_meta"].attrs["normalized_twist_weight"], _WEIGHTS[index], "direct twist weight")
    cli_ham = load_hamiltonian_from_checkpoint(str(cli_path))
    direct_ham = load_hamiltonian_from_checkpoint(str(direct_path))
    input_ham = load_twist(str(source), index)
    for ham in (cli_ham, direct_ham):
        for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
            actual_orbital = getattr(ham.wavefunction_data.geminal_data, spin)
            input_orbital = getattr(input_ham.wavefunction_data.geminal_data, spin)
            _exact(actual_orbital.aos_data.k_point, input_orbital.aos_data.k_point, "twist k identity")
            _exact(actual_orbital.mo_coefficients, input_orbital.mo_coefficients, "twist orbitals")
    if job_type == "vmc":
        initial_j2 = np.asarray(input_ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
        updated_j2 = np.asarray(direct_ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)
        assert np.any(updated_j2 != initial_j2), "the optimizer evidence must exercise a nonzero J2 update"
        _close(
            cli_ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            direct_ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param,
            "CLI versus direct twist-optimized J2",
        )
    else:
        driver_class = {"mcmc": MCMC, "lrdmc-bra": GFMC_n, "lrdmc-tau": GFMC_t}[job_type]
        cli_driver = driver_class.load_from_hdf5(str(cli_path), mpi_comm_in=MPI.COMM_SELF, world_comm_in=MPI.COMM_SELF)
        direct_driver = driver_class.load_from_hdf5(str(direct_path), mpi_comm_in=MPI.COMM_SELF, world_comm_in=MPI.COMM_SELF)
        for name in ("e_L", "e_L2", "w_L"):
            _close(getattr(cli_driver, name), getattr(direct_driver, name), f"{job_type} twist {index}: {name}")
        analysis = _MCMC_ANALYSIS if job_type == "mcmc" else _GFMC_ANALYSIS
        _close(cli_driver.get_E(**analysis), direct_driver.get_E(**analysis), "twist energy tuple")


@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_twist_job_equals_direct_orchestrator(tmp_path, monkeypatch, job_type):
    """Actual two-rank CLI and direct twist drivers agree for nondefault MCMC/VMC/LRDMC inputs.

    Lithium TREXIO twists zero/one, Pade J2, weights 1/4 and 3/4, seed 24680
    and two walkers exercise real MPI splitting. Check exact walker/RNG/twist
    identity; compare stored GFMC histories and energy tuples or optimized
    J2 at the existing strict same-mode policy. No statistical acceptance.
    """
    comm = MPI.COMM_WORLD
    assert comm.size == 2, "this evidence requires two real MPI ranks"
    root = Path(comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0))
    source = root / "combined.h5"
    if comm.rank == 0:
        _write_twist_input(source)
    comm.Barrier()
    _run_actual_cli(root / "cli", source, job_type, monkeypatch)
    _run_direct_twist(root / "direct", source, job_type, monkeypatch)
    index = comm.rank
    _assert_twist_checkpoint_equivalence(
        root / "cli" / "checkpoint" / f"twist_{index:03d}.h5",
        root / "direct" / "checkpoint" / f"twist_{index:03d}.h5",
        source,
        job_type,
        index,
    )
