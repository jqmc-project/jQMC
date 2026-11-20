"""Same-input numerical oracles for actual CLI checkpoint continuation."""

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
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint, merge_rank_checkpoints
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_STEPS,
    GFMC_MIN_WARMUP_STEPS,
    GFMC_ON_THE_FLY_COLLECT_STEPS,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.output_parser import parse_force_table, parse_pressure_table, parse_stress_table
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _write_hamiltonian(root, trexio_name="H2_ae_ccpvdz_cart.h5"):
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / trexio_name
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    hamiltonian = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(
            geminal_data=geminal,
            jastrow_data=Jastrow_data(
                jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=1.0, jastrow_2b_type="pade"
                )
            ),
        ),
    )
    hamiltonian.save_to_hdf5(str(root / "hamiltonian.h5"))


# Frozen keys left out of the continuation TOML; the CLI restores them from the checkpoint.
_OMITTED_ON_RESTART = {
    "mcmc": ("control", "number_of_walkers", "mcmc", "Dt"),
    "vmc": ("control", "number_of_walkers", "vmc", "Dt"),
    "lrdmc-bra": ("control", "number_of_walkers", "lrdmc-bra", "alat"),
    "lrdmc-tau": ("control", "number_of_walkers", "lrdmc-tau", "alat"),
}


def _run_cli(root, job_type, restart, multiplier, monkeypatch, *, mcmc_seed, overrides=None, omit_job_keys=()):
    """Run the CLI once and return its log messages."""
    root.mkdir(exist_ok=True)
    params = copy.deepcopy(jqmc_cli.cli_parameters[job_type])
    if job_type in ("mcmc", "vmc"):
        params.update(
            num_mcmc_steps=8 if job_type == "vmc" else 8 * multiplier,
            num_mcmc_per_measurement=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=2,
        )
    else:
        params.update(
            num_mcmc_steps=GFMC_MIN_STEPS * multiplier,
            num_gfmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_gfmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        )
        if job_type == "lrdmc-bra":
            params["num_mcmc_per_measurement"] = 1
            if restart:
                # The adaptive value is restored; reasserting the initial
                # guess is deliberately rejected by the frozen-key guard.
                params.pop("E_scf")
    if job_type == "vmc":
        params["num_opt_steps"] = multiplier
        for key in params:
            if key.startswith("opt_"):
                params[key] = key == "opt_J2_param"
        params["optimizer_kwargs"].update(method="sr", delta=0.01, use_lm=False)
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": mcmc_seed,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root.parent / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: params,
    }
    if restart:
        control_section, control_key, section, key = _OMITTED_ON_RESTART[job_type]
        del payload[control_section][control_key]
        del payload[section][key]
    for section, values in (overrides or {}).items():
        payload.setdefault(section, {}).update(values)
    for key in omit_job_keys:
        del payload[job_type][key]
    input_file = root / "input.toml"
    input_file.write_text(toml.dumps(payload))
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    messages = []
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            # The CLI installs its own handlers; record every message it emits.
            patch.setattr(logging.Logger, "callHandlers", _recording(logging.Logger.callHandlers, messages))
            jqmc_cli._cli()
    finally:
        logger.handlers, logger.level = handlers, level
    return messages


def _recording(call_handlers, messages):
    def call(self, record):
        if self.name == "jqmc" or self.name.startswith("jqmc."):
            messages.append(record.getMessage())
        return call_handlers(self, record)

    return call


@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_restart_preserves_continuous_state_and_energy(tmp_path, monkeypatch, job_type):
    """Actual CLI restart matches a continuous run on fixed H2 AE with nonzero Pade J2.

    The committed Cartesian-GTO/MO H2 input has one up and one down electron,
    open boundaries, J2 parameter 1 and no J1/J3/DTN. Two walkers use seed 12345;
    positions are production-generated with no cond(G) filter. MCMC compares
    16 measurements with 8+8; VMC compares two J2-only SR updates with 1+1,
    each using eight measurements. Both GFMC algorithms compare 80 with 40+40
    measurements under the central minimum-step policy, crossing the adaptive
    shift warmup boundary. The load-balanced shift-update interval is one in
    both runs; this does not claim equivalence of different update cadences.

    Checkpoint positions, JAX/branching RNG state and counters are exact.
    Stored e_L/e_L2/w_L, final reduced-energy tuple, J2 parameter and the
    load-balanced adaptive E_scf/G histories use the existing same-mode strict
    zone tolerance, never error bars. UUID namespaces differ between fresh
    runs, but lineage must persist within the resumed run. No sampler is mocked.
    VMC clears sample buffers after updating its wavefunction: require that
    empty state, then compare four further measurements from both restored
    final optimizers, rather than pretending that a cleared buffer has energy.
    """
    _write_hamiltonian(tmp_path)
    _run_cli(tmp_path / "continuous", job_type, False, 2, monkeypatch, mcmc_seed=12345)
    _run_cli(tmp_path / "resumed", job_type, False, 1, monkeypatch, mcmc_seed=12345)
    continuous_file = str(tmp_path / "continuous" / "restart.h5")
    resumed_file = str(tmp_path / "resumed" / "restart.h5")
    split_state = load_rank_checkpoint(resumed_file, rank=0)
    driver_class = {"mcmc": MCMC, "vmc": MCMC, "lrdmc-bra": GFMC_n, "lrdmc-tau": GFMC_t}[job_type]
    # Check every declared scalar frozen field against this real driver's
    # checkpoint. A shared guard bypass or a missing checkpoint mapping must
    # fail before loading the sampler, even when the setting is a new flag.
    from jqmc._restart_policy import TOML_TO_CFG

    checkpoint_bytes = Path(resumed_file).read_bytes()

    def unexpected_sampler_load(*args, **kwargs):
        pytest.fail("a frozen-setting conflict reached sampler loading")

    checked = set()
    for section in ("control", job_type):
        for name, kind in jqmc_cli.cli_parameters[f"{section}_restart_policy"].items():
            if kind != "frozen" or name == "hamiltonian_h5":
                continue
            cfg_key = TOML_TO_CFG[(section, name)]
            assert cfg_key in split_state["driver_config"], (job_type, section, name, cfg_key)
            saved = split_state["driver_config"][cfg_key]
            if isinstance(saved, (bool, np.bool_)):
                conflicting = not saved
            elif isinstance(saved, (int, float, np.integer, np.floating)):
                conflicting = saved.item() + 1 if isinstance(saved, np.generic) else saved + 1
            else:
                assert name == "non_local_move" and saved in ("tmove", "dltmove"), (name, saved)
                conflicting = "dltmove" if saved == "tmove" else "tmove"
            with monkeypatch.context() as patch:
                patch.setattr(driver_class, "load_from_hdf5", unexpected_sampler_load)
                with pytest.raises(ValueError, match=rf"\[{section}\] {name}.*frozen on restart"):
                    _run_cli(
                        tmp_path / "resumed",
                        job_type,
                        True,
                        1,
                        patch,
                        mcmc_seed=12345,
                        overrides={section: {name: conflicting}},
                    )
            assert Path(resumed_file).read_bytes() == checkpoint_bytes
            checked.add((section, name))
    assert checked == {
        (section, name)
        for section in ("control", job_type)
        for name, kind in jqmc_cli.cli_parameters[f"{section}_restart_policy"].items()
        if kind == "frozen" and name != "hamiltonian_h5"
    }
    restart_log = _run_cli(tmp_path / "resumed", job_type, True, 1, monkeypatch, mcmc_seed=12345)
    # The restart log states the values the run used: the omitted frozen keys
    # with their checkpoint values (never the defaults), and the MPI/walker layout.
    key = _OMITTED_ON_RESTART[job_type][3]
    restored = {"number_of_walkers": 2, key: split_state["driver_config"][key]}
    for name, value in restored.items():
        assert f"The value of {name} = {value} (restored from the checkpoint)." in restart_log
        assert not any(line.startswith(f"The default value of {name} =") for line in restart_log)
    assert "The number of MPI process = 1." in restart_log
    assert "The number of walkers assigned for each MPI process = 2." in restart_log
    reference = load_rank_checkpoint(continuous_file, rank=0)
    actual = load_rank_checkpoint(resumed_file, rank=0)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        assert actual["walker_state"][name].shape == reference["walker_state"][name].shape == (2, 1, 3)
        assert np.isfinite(actual["walker_state"][name]).all() and np.isfinite(reference["walker_state"][name]).all()
        np.testing.assert_array_equal(actual["walker_state"][name], reference["walker_state"][name])
    np.testing.assert_array_equal(actual["rng_state"]["jax_PRNG_key_list"], reference["rng_state"]["jax_PRNG_key_list"])
    assert actual["driver_config"]["mcmc_counter"] == reference["driver_config"]["mcmc_counter"]
    if job_type.startswith("lrdmc"):
        for name in (
            "branching_rng_keys",
            "branching_rng_position",
            "branching_rng_has_gauss",
            "branching_rng_cached_gaussian",
        ):
            np.testing.assert_array_equal(actual["rng_state"][name], reference["rng_state"][name])
    else:
        assert actual["optimizer_state"]["wavefunction_lineage_id"] == split_state["optimizer_state"]["wavefunction_lineage_id"]
        assert actual["optimizer_state"]["wavefunction_revision"] == reference["optimizer_state"]["wavefunction_revision"]
    observed_driver = driver_class.load_from_hdf5(resumed_file)
    expected_driver = driver_class.load_from_hdf5(continuous_file)
    if job_type == "vmc":
        assert actual["driver_config"]["i_opt"] == reference["driver_config"]["i_opt"] == 2
        assert all(
            name not in actual["observables"] and name not in reference["observables"] for name in ("e_L", "e_L2", "w_L")
        )
        observed_driver.run(num_mcmc_steps=4)
        expected_driver.run(num_mcmc_steps=4)
    for name in ("e_L", "e_L2", "w_L"):
        observed, expected = (
            (np.asarray(getattr(observed_driver, name)), np.asarray(getattr(expected_driver, name)))
            if job_type == "vmc"
            else (actual["observables"][name], reference["observables"][name])
        )
        assert observed.size > 0 and expected.size > 0
        assert np.isfinite(observed).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)
    if job_type == "lrdmc-bra":
        np.testing.assert_allclose(actual["driver_config"]["E_scf"], reference["driver_config"]["E_scf"], atol=atol, rtol=rtol)
        for name in ("G_L", "G_e_L"):
            observed, expected = actual["observables"][name], reference["observables"][name]
            assert observed.shape == expected.shape == (2 * GFMC_MIN_STEPS - GFMC_ON_THE_FLY_COLLECT_STEPS, 1)
            assert np.isfinite(observed).all() and np.isfinite(expected).all()
            np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)
    analysis = (
        {"num_mcmc_warmup_steps": GFMC_MIN_WARMUP_STEPS, "num_mcmc_bin_blocks": GFMC_MIN_BIN_BLOCKS}
        if job_type.startswith("lrdmc")
        else {"num_mcmc_warmup_steps": 0, "num_mcmc_bin_blocks": 2}
    )
    observed_energy = np.asarray(observed_driver.get_E(**analysis))
    expected_energy = np.asarray(expected_driver.get_E(**analysis))
    assert np.isfinite(observed_energy).all() and np.isfinite(expected_energy).all()
    np.testing.assert_allclose(observed_energy, expected_energy, atol=atol, rtol=rtol)
    observed_j2 = load_hamiltonian_from_checkpoint(
        resumed_file
    ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
    expected_j2 = load_hamiltonian_from_checkpoint(
        continuous_file
    ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
    assert np.isfinite(observed_j2).all() and np.isfinite(expected_j2).all()
    np.testing.assert_allclose(observed_j2, expected_j2, atol=atol, rtol=rtol)
    if job_type == "vmc":
        assert np.any(np.asarray(observed_j2) != 1.0), "the nonzero J2 optimization must actually update its parameter"


def _save_merged_checkpoint(driver, root):
    """Write ``root/restart.h5`` exactly as the CLI does for one rank."""
    root.mkdir(exist_ok=True)
    rank_file = str(root / "rank{rank}.h5")
    driver.save_to_hdf5(rank_file.format(rank=0))
    merge_rank_checkpoints(
        output_path=str(root / "restart.h5"),
        mpi_size=1,
        driver_type="MCMC",
        hamiltonian_data=driver.hamiltonian_data,
        tmp_pattern=rank_file,
    )
    return load_rank_checkpoint(str(root / "restart.h5"), rank=0)


@pytest.mark.parametrize(
    "trexio_name", ["H2_ae_ccpvdz_cart.h5", "H2_pbc_ae_k025_cart.h5"], ids=["real-molecule", "complex-twist"]
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_mcmc_parameter_derivatives_continue_unchanged(tmp_path, monkeypatch, trexio_name):
    """The CLI mcmc job with parameter_derivatives = true keeps every result and continuation.

    Inputs: the committed H2 AE Cartesian MO input (real, open boundaries) and
    the H2 periodic AE input at twist k = (0.25, 0.25, 0.25) (complex
    orbitals), each with Pade J2 parameter 1, two walkers and seed 12345.
    Independent reference: the same sampling through the Python API with its
    default ``MCMC(retain_derivative_sources=True)``, which keeps a separate
    copy of every exact derivative source, saved and merged as the CLI does.
    The CLI runs 8 measurements, restarts for 8 more, and a plain-SR
    (``use_lm = false``) J2-only ``vmc`` job then continues that ``mcmc``
    checkpoint; the same ``vmc`` continuation also runs from the reference
    checkpoint. Same seed, same route and same precision mode on both sides,
    so the stored energies, weights and folded parameter derivatives after
    each stage and the optimized J2 parameter must be identical.
    """
    _write_hamiltonian(tmp_path, trexio_name)
    derivatives = {"mcmc": {"parameter_derivatives": True}}
    cli_root, api_root = tmp_path / "cli", tmp_path / "api"
    _run_cli(cli_root, "mcmc", False, 1, monkeypatch, mcmc_seed=12345, overrides=derivatives)
    cli_stages = [load_rank_checkpoint(str(cli_root / "restart.h5"), rank=0)]
    _run_cli(cli_root, "mcmc", True, 1, monkeypatch, mcmc_seed=12345, overrides=derivatives)
    cli_stages.append(load_rank_checkpoint(str(cli_root / "restart.h5"), rank=0))

    cfg = cli_stages[0]["driver_config"]
    reference = MCMC(
        hamiltonian_data=Hamiltonian_data.load_from_hdf5(str(tmp_path / "hamiltonian.h5")),
        mcmc_seed=12345,
        num_walkers=2,
        num_mcmc_per_measurement=1,
        Dt=cfg["Dt"],
        epsilon_AS=cfg["epsilon_AS"],
        comput_position_deriv=bool(cfg["comput_position_deriv"]),
        comput_stress=bool(cfg["comput_stress"]),
        comput_log_WF_param_deriv=True,
        use_swct=bool(cfg["use_swct"]),
        mpc=bool(cfg["mpc"]),
        mpc_cutoff=cfg["mpc_cutoff"],
    )
    reference.run(num_mcmc_steps=8)
    api_stages = [_save_merged_checkpoint(reference, api_root)]
    reference = MCMC.load_from_hdf5(str(api_root / "restart.h5"), rank=0)
    reference.run(num_mcmc_steps=8)
    api_stages.append(_save_merged_checkpoint(reference, api_root))

    for stage, (observed, expected) in enumerate(zip(cli_stages, api_stages, strict=True)):
        assert observed["driver_config"]["mcmc_counter"] == expected["driver_config"]["mcmc_counter"] == 8 * (stage + 1)
        for name in ("e_L", "w_L"):
            assert np.isfinite(expected["observables"][name]).all()
            np.testing.assert_array_equal(observed["observables"][name], expected["observables"][name])
        observed_grads = observed["observables"]["param_grads"]
        expected_grads = expected["observables"]["param_grads"]
        assert set(observed_grads) == set(expected_grads) and "j2_param" in expected_grads
        for block in sorted(expected_grads):
            assert expected_grads[block].shape[:2] == (8 * (stage + 1), 2)
            np.testing.assert_array_equal(observed_grads[block], expected_grads[block])

    optimized = []
    for root in (cli_root, api_root):
        _run_cli(root, "vmc", True, 1, monkeypatch, mcmc_seed=12345)
        after = load_rank_checkpoint(str(root / "restart.h5"), rank=0)
        assert after["driver_config"]["i_opt"] == 1
        optimized.append(
            load_hamiltonian_from_checkpoint(
                str(root / "restart.h5")
            ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
        )
    observed_j2, expected_j2 = (np.asarray(value) for value in optimized)
    assert np.isfinite(expected_j2).all() and np.any(expected_j2 != 1.0), "the SR continuation must update J2"
    np.testing.assert_array_equal(observed_j2, expected_j2)


@pytest.fixture
def sr_vmc_restart(tmp_path, monkeypatch):
    """A real one-update CLI checkpoint from plain SR."""
    _write_hamiltonian(tmp_path)
    resumed = tmp_path / "resumed"
    _run_cli(resumed, "vmc", False, 1, monkeypatch, mcmc_seed=12345)
    checkpoint = str(resumed / "restart.h5")
    before = load_rank_checkpoint(checkpoint, rank=0)
    assert before["driver_config"]["comput_log_WF_param_deriv"] is True
    assert before["driver_config"]["comput_e_L_param_deriv"] is False
    assert before["driver_config"]["i_opt"] == 1
    assert "e_L" not in before["observables"]
    return resumed, checkpoint, before


_VMC_LM = {"vmc": {"optimizer_kwargs": {"method": "sr", "delta": 0.01, "use_lm": True, "lm_subspace_dim": 1}}}


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_vmc_restart_can_enable_lm_at_an_accepted_update(sr_vmc_restart, monkeypatch):
    """The real CLI can start collecting LM derivatives at an SR update boundary."""
    resumed, checkpoint, before = sr_vmc_restart
    restart_log = _run_cli(resumed, "vmc", True, 1, monkeypatch, mcmc_seed=12345, overrides=_VMC_LM)
    assert any("use_lm=True, lm_subspace_dim=1: Linear Method" in line for line in restart_log)
    after = load_rank_checkpoint(checkpoint, rank=0)
    assert after["driver_config"]["comput_e_L_param_deriv"] is True
    assert after["driver_config"]["i_opt"] == before["driver_config"]["i_opt"] + 1
    assert after["optimizer_state"]["wavefunction_lineage_id"] == before["optimizer_state"]["wavefunction_lineage_id"]
    assert after["optimizer_state"]["wavefunction_revision"] == before["optimizer_state"]["wavefunction_revision"] + 1
    assert after["rng_state"]["mpi_seed"] == before["rng_state"]["mpi_seed"]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_vmc_restart_rejects_lm_with_retained_sr_samples(sr_vmc_restart, monkeypatch):
    """Previously sampled rows without LM derivatives cannot be used for LM."""
    resumed, checkpoint, _ = sr_vmc_restart
    driver = MCMC.load_from_hdf5(checkpoint, rank=0)
    driver.run(num_mcmc_steps=4)
    rank_file = str(resumed / "rank{rank}.h5")
    driver.save_to_hdf5(rank_file.format(rank=0))
    merge_rank_checkpoints(
        output_path=checkpoint,
        mpi_size=1,
        driver_type="MCMC",
        hamiltonian_data=driver.hamiltonian_data,
        tmp_pattern=rank_file,
    )
    retained = load_rank_checkpoint(checkpoint, rank=0)["observables"]
    assert retained["e_L"].size > 0
    assert not retained.get("e_L_param_grads")
    checkpoint_bytes = Path(checkpoint).read_bytes()
    with pytest.raises(ValueError, match=r"local-energy parameter derivatives during restart.*retained samples"):
        _run_cli(resumed, "vmc", True, 1, monkeypatch, mcmc_seed=12345, overrides=_VMC_LM)
    assert Path(checkpoint).read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("job_type", ["mcmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_restart_preserves_mpc_history_and_reporting(tmp_path, monkeypatch, job_type):
    """MPC continuation inherits the checkpoint and agrees with uninterrupted periodic sampling."""
    from tests.test_jqmc_kernel.test_cli_driver_equivalence import _write_hamiltonian as _write_equivalence_hamiltonian

    _write_equivalence_hamiltonian(tmp_path, "periodic")
    cutoff = 11.0
    fresh = {"coulomb": {"mpc": True, "mpc_cutoff": cutoff}}
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    continuous_log = _run_cli(continuous, job_type, False, 2, monkeypatch, mcmc_seed=24680, overrides=fresh)
    _run_cli(resumed, job_type, False, 1, monkeypatch, mcmc_seed=24680, overrides=fresh)
    restart_log = _run_cli(resumed, job_type, True, 1, monkeypatch, mcmc_seed=24680)

    assert "The value of mpc = True (restored from the checkpoint)." in restart_log
    assert f"The value of mpc_cutoff = {cutoff} (restored from the checkpoint)." in restart_log
    for label in ("MPC correction: dE_MPC =", "Total Energy (MPC-corrected): E_MPC ="):
        expected_summary = [line for line in continuous_log if label in line]
        observed_summary = [line for line in restart_log if label in line]
        assert len(expected_summary) == 1
        assert observed_summary == expected_summary
    assert any("MPC correction: " in line and f"(cutoff = {cutoff} Ha)" in line for line in restart_log)

    checkpoint = resumed / "restart.h5"
    reference = load_rank_checkpoint(str(continuous / "restart.h5"), rank=0)
    actual = load_rank_checkpoint(str(checkpoint), rank=0)
    assert actual["driver_config"]["mpc"] is True and reference["driver_config"]["mpc"] is True
    assert actual["driver_config"]["mpc_cutoff"] == reference["driver_config"]["mpc_cutoff"] == cutoff
    observed = actual["observables"]["mpc_dV"]
    expected = reference["observables"]["mpc_dV"]
    history_shape = (16, 2) if job_type == "mcmc" else (2 * GFMC_MIN_STEPS, 1)
    assert observed.shape == expected.shape == history_shape
    assert np.isfinite(observed).all() and np.isfinite(expected).all()
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)

    checkpoint_bytes = checkpoint.read_bytes()
    for key, value in (("mpc", False), ("mpc_cutoff", 7.0)):
        with pytest.raises(ValueError, match=rf"\[coulomb\] {key}.*frozen on restart"):
            _run_cli(resumed, job_type, True, 1, monkeypatch, mcmc_seed=24680, overrides={"coulomb": {key: value}})
        assert checkpoint.read_bytes() == checkpoint_bytes


def _force_stress_values(messages):
    final_output = "\n".join(messages[messages.index("Final output(s):") :])
    forces = parse_force_table(final_output)
    stress = parse_stress_table(final_output)
    pressure = parse_pressure_table(final_output)
    assert forces is not None, "missing final atomic force table"
    assert stress is not None, "missing or incomplete final stress table"
    assert pressure is not None, "missing or incomplete final pressure table"
    assert [row["label"] for row in forces] == ["Li", "Li"]
    values = np.asarray(
        [value for row in forces for key, value in row.items() if key != "label"]
        + list(stress.values())
        + list(pressure.values()),
        dtype=float,
    )
    assert np.isfinite(values).all(), "non-finite final force, stress or pressure output"
    return values


@pytest.mark.parametrize("job_type", ["mcmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_restart_preserves_force_stress_reporting(tmp_path, monkeypatch, job_type):
    """Omitted frozen reporting switches recover the uninterrupted periodic Li tables."""
    from tests.test_jqmc_kernel.test_cli_driver_equivalence import _write_hamiltonian as _write_equivalence_hamiltonian

    _write_equivalence_hamiltonian(tmp_path, "periodic")
    enabled = {"atomic_force": True, "compute_stress": True}
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    expected_log = _run_cli(continuous, job_type, False, 2, monkeypatch, mcmc_seed=24680, overrides={job_type: enabled})
    _run_cli(resumed, job_type, False, 1, monkeypatch, mcmc_seed=24680, overrides={job_type: enabled})
    actual_log = _run_cli(resumed, job_type, True, 1, monkeypatch, mcmc_seed=24680, omit_job_keys=enabled)
    for key in enabled:
        assert f"The value of {key} = True (restored from the checkpoint)." in actual_log

    expected_values = _force_stress_values(expected_log)
    actual_values = _force_stress_values(actual_log)
    np.testing.assert_array_equal(actual_values, expected_values)
