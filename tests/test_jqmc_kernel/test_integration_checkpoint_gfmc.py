"""Tests for GFMC_t / GFMC_n save_to_hdf5 / load_from_hdf5."""

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

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import h5py

jax.config.update("jax_enable_x64", True)

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_STEPS
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

TREXIO_FILES = [
    "H2_ae_ccpvdz_cart.h5",  # all-electron
    "H2_ecp_ccpvdz_cart.h5",  # ECP
]

JASTROW_COMBOS = [
    "none",  # empty Jastrow
    "1b",  # one-body only
    "1b+2b",  # one-body + two-body
    "1b+2b+3b",  # one-body + two-body + three-body
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hamiltonian(trexio_file, jastrow_combo):
    """Build a Hamiltonian_data from a trexio file with a given Jastrow combination."""
    structure_data, aos_data, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, trexio_file),
        store_tuple=True,
    )

    jastrow_one_body_data = None
    jastrow_two_body_data = None
    jastrow_three_body_data = None

    if jastrow_combo in ("1b", "1b+2b", "1b+2b+3b"):
        core_electrons = tuple([0.0] * len(structure_data.positions))
        jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons,
            jastrow_1b_type="pade",
        )

    if jastrow_combo in ("1b+2b", "1b+2b+3b"):
        jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="exp",
        )

    if jastrow_combo in ("1b+2b+3b",):
        jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data,
        )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )
    jastrow_data.sanity_check()

    wfd = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
    wfd.sanity_check()

    hd = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wfd,
    )
    hd.sanity_check()
    return hd


def _make_gfmc_t(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return a GFMC_t instance."""
    from jqmc.jqmc_gfmc import GFMC_t

    return GFMC_t(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _make_gfmc_n(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return a GFMC_n instance."""
    from jqmc.jqmc_gfmc import GFMC_n

    kwargs.setdefault("num_mcmc_per_measurement", 16)
    return GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _save_merge_gfmc_t(gfmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file (GFMC_t)."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    gfmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="GFMC_t",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


def _save_merge_gfmc_n(gfmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file (GFMC_n)."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    gfmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="GFMC_n",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


@pytest.mark.light
@pytest.mark.parametrize("boundary", ["constructor", "checkpoint"])
@pytest.mark.parametrize(
    "tau",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-0.1, id="negative"),
        pytest.param(np.inf, id="infinite"),
        pytest.param(np.nan, id="nan"),
    ],
)
def test_gfmc_t_rejects_invalid_tau(tmp_path, boundary, tau):
    """Both driver entry points reject a non-positive or non-finite projection interval."""
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    if boundary == "constructor":
        with pytest.raises(ValueError, match=r"tau must be finite and > 0"):
            _make_gfmc_t(hd, num_walkers=2, tau=tau)
    else:
        driver = _make_gfmc_t(hd, num_walkers=2, tau=0.1)
        merged = _save_merge_gfmc_t(driver, hd, tmp_path)
        with h5py.File(merged, "r+") as checkpoint:
            checkpoint["rank_0/driver_config"].attrs["tau"] = tau
        with pytest.raises(ValueError, match=r"tau must be finite and > 0"):
            type(driver).load_from_hdf5(merged, rank=0)


@pytest.mark.light
@pytest.mark.parametrize("tau", [pytest.param(0.0, id="zero"), pytest.param(0.1, id="positive")])
def test_lrdmc_tau_cli_domain(tmp_path, tau):
    """The console entry refuses zero tau and still projects and checkpoints a valid run."""
    hd = _build_hamiltonian(TREXIO_FILES[0], "1b+2b")
    hd.save_to_hdf5(tmp_path / "hamiltonian.h5")
    precision = "mixed" if is_mixed_precision_enabled() else "full"
    (tmp_path / "input.toml").write_text(
        f"""
[control]
job_type = "lrdmc-tau"
number_of_walkers = 2
mcmc_seed = 12345
hamiltonian_h5 = "hamiltonian.h5"
restart_chk = "restart.h5"
restart = false

[precision]
mode = "{precision}"

[lrdmc-tau]
num_mcmc_steps = {GFMC_MIN_STEPS}
tau = {tau}
alat = 0.3
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "JQMC_FLAG_RUST_ON_CPU": "0",
    }
    completed = subprocess.run(
        [sys.executable, "-c", "from jqmc._cli_entry import main; main()", "input.toml"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    checkpoint = tmp_path / "restart.h5"
    if tau == 0.0:
        assert completed.returncode != 0, completed.stdout
        assert "tau must be finite and > 0" in completed.stderr
        assert not checkpoint.exists()
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        from jqmc.jqmc_gfmc import GFMC_t

        resumed = GFMC_t.load_from_hdf5(str(checkpoint), rank=0)
        assert len(resumed.average_projection_counter) == GFMC_MIN_STEPS
        assert (resumed.average_projection_counter > 0).all()


_INVALID_ALATS = [
    pytest.param(0.0, id="zero"),
    pytest.param(-0.3, id="negative"),
    pytest.param(np.inf, id="infinite"),
    pytest.param(np.nan, id="nan"),
]


@pytest.mark.light
@pytest.mark.parametrize("driver_name", ["GFMC_t", "GFMC_n"])
@pytest.mark.parametrize("boundary", ["constructor", "checkpoint"])
@pytest.mark.parametrize("alat", _INVALID_ALATS)
def test_gfmc_rejects_invalid_alat(tmp_path, driver_name, boundary, alat):
    """Both LRDMC drivers reject a non-positive or non-finite lattice spacing at both entry points."""
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    make, save_merge = (_make_gfmc_t, _save_merge_gfmc_t) if driver_name == "GFMC_t" else (_make_gfmc_n, _save_merge_gfmc_n)
    if boundary == "constructor":
        with pytest.raises(ValueError, match=r"alat must be finite and > 0"):
            make(hd, num_walkers=2, alat=alat)
    else:
        driver = make(hd, num_walkers=2, alat=0.3)
        merged = save_merge(driver, hd, tmp_path)
        with h5py.File(merged, "r+") as checkpoint:
            checkpoint["rank_0/driver_config"].attrs["alat"] = alat
        with pytest.raises(ValueError, match=r"alat must be finite and > 0"):
            type(driver).load_from_hdf5(merged, rank=0)


@pytest.mark.light
@pytest.mark.parametrize("job_type", ["lrdmc-tau", "lrdmc-bra"])
@pytest.mark.parametrize("rust_on_cpu", [pytest.param("1", id="rust"), pytest.param("0", id="xla")])
@pytest.mark.parametrize("alat", [pytest.param(0.0, id="zero"), pytest.param(-0.3, id="negative")])
def test_lrdmc_cli_rejects_nonpositive_alat(tmp_path, job_type, rust_on_cpu, alat):
    """The console entry refuses a non-positive lattice spacing on both backends and writes no checkpoint.

    A native lrdmc-tau projection that never returns at ``alat = 0`` would hang,
    so the subprocess timeout turns a hang into a test failure.
    """
    hd = _build_hamiltonian(TREXIO_FILES[0], "1b+2b")
    hd.save_to_hdf5(tmp_path / "hamiltonian.h5")
    precision = "mixed" if is_mixed_precision_enabled() else "full"
    driver_keys = "tau = 0.1" if job_type == "lrdmc-tau" else "num_mcmc_per_measurement = 4"
    (tmp_path / "input.toml").write_text(
        f"""
[control]
job_type = "{job_type}"
number_of_walkers = 2
mcmc_seed = 12345
hamiltonian_h5 = "hamiltonian.h5"
restart_chk = "restart.h5"
restart = false

[precision]
mode = "{precision}"

[{job_type}]
num_mcmc_steps = {GFMC_MIN_STEPS}
{driver_keys}
alat = {alat}
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "JQMC_FLAG_RUST_ON_CPU": rust_on_cpu,
    }
    completed = subprocess.run(
        [sys.executable, "-c", "from jqmc._cli_entry import main; main()", "input.toml"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout
    assert "alat must be finite and > 0" in completed.stderr, completed.stderr[-2000:]
    assert not (tmp_path / "restart.h5").exists()


@pytest.mark.light
@pytest.mark.parametrize("driver_name", ["GFMC_t", "GFMC_n", "_GFMC_t_debug", "_GFMC_n_debug"])
def test_branching_rng_initial_stream_is_driver_owned(driver_name):
    """All GFMC drivers preserve the original MT19937 draws independently of other drivers."""
    from jqmc import jqmc_gfmc
    from jqmc._driver_helpers import derive_rank_rng_streams

    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    driver_class = getattr(jqmc_gfmc, driver_name)
    nproj = {"num_mcmc_per_measurement": 16} if driver_name in ("GFMC_n", "_GFMC_n_debug") else {}
    driver = driver_class(hamiltonian_data=hd, num_walkers=2, mcmc_seed=12345, **nproj)
    seed_words, _ = derive_rank_rng_streams(12345, 0)
    reference = np.random.RandomState(seed_words)
    np.testing.assert_array_equal(driver._branching_rng.random(5), reference.random(5))
    driver_class(hamiltonian_data=hd, num_walkers=2, mcmc_seed=9081, **nproj)
    np.random.random(19)
    np.testing.assert_array_equal(driver._branching_rng.random(7), reference.random(7))


@pytest.mark.heavy
@pytest.mark.parametrize("driver_kind", ["t", "n"])
def test_split_run_preserves_actual_branching_draws(driver_kind):
    """Two run calls consume six branching draws and reproduce nonempty uninterrupted observables."""
    from types import SimpleNamespace

    from jqmc._driver_helpers import derive_rank_rng_streams

    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    make_driver = _make_gfmc_t if driver_kind == "t" else _make_gfmc_n
    uninterrupted = make_driver(hd)
    uninterrupted.run(num_mcmc_steps=6)
    split = make_driver(hd)
    stream = split._branching_rng
    draws = []

    def record_draw():
        value = stream.random()
        draws.append(value)
        return value

    split._branching_rng = SimpleNamespace(random=record_draw, get_state=stream.get_state)
    split.run(num_mcmc_steps=3)
    np.random.random(19)
    split.run(num_mcmc_steps=3)
    seed_words, _ = derive_rank_rng_streams(12345, 0)
    expected_draws = np.random.RandomState(seed_words).random(6)
    np.testing.assert_array_equal(draws, expected_draws)
    np.testing.assert_array_equal(split._jax_PRNG_key_list, uninterrupted._jax_PRNG_key_list)
    np.testing.assert_array_equal(split._latest_r_up_carts, uninterrupted._latest_r_up_carts)
    np.testing.assert_array_equal(split._latest_r_dn_carts, uninterrupted._latest_r_dn_carts)
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    assert split.e_L.size > 0 and uninterrupted.e_L.size > 0
    assert split.w_L.size > 0 and uninterrupted.w_L.size > 0
    np.testing.assert_allclose(split._stored_e_L, uninterrupted._stored_e_L, atol=atol, rtol=rtol)
    np.testing.assert_allclose(split.bare_w_L, uninterrupted.bare_w_L, atol=atol, rtol=rtol)
    np.testing.assert_allclose(split.e_L, uninterrupted.e_L, atol=atol, rtol=rtol)
    np.testing.assert_allclose(split.w_L, uninterrupted.w_L, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["t", "n"])
@pytest.mark.parametrize("draw_count", [0, 1, 625])
def test_checkpoint_preserves_complete_branching_rng(tmp_path, driver_kind, draw_count):
    """Checkpoint round trips preserve MT19937 position, words and a populated Gaussian cache."""
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    make_driver = _make_gfmc_t if driver_kind == "t" else _make_gfmc_n
    save_driver = _save_merge_gfmc_t if driver_kind == "t" else _save_merge_gfmc_n
    driver = make_driver(hd, num_walkers=2)
    driver._branching_rng.random(draw_count)
    driver._branching_rng.standard_normal()
    before = driver._branching_rng.get_state()
    assert before[3] == 1
    merged = save_driver(driver, hd, tmp_path)
    resumed = type(driver).load_from_hdf5(merged, rank=0)
    after = resumed._branching_rng.get_state()
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]
    np.testing.assert_array_equal(resumed._branching_rng.standard_normal(3), driver._branching_rng.standard_normal(3))
    np.testing.assert_array_equal(resumed._branching_rng.random(9), driver._branching_rng.random(9))


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["t", "n"])
def test_restart_rejects_rank_local_nonfinite_energy_history(tmp_path, driver_kind):
    """Both production GFMC loaders refuse a poisoned averaged energy history."""
    make_driver = _make_gfmc_t if driver_kind == "t" else _make_gfmc_n
    save_driver = _save_merge_gfmc_t if driver_kind == "t" else _save_merge_gfmc_n
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    driver = make_driver(hd, num_walkers=2)
    merged = save_driver(driver, hd, tmp_path)
    with h5py.File(merged, "r+") as checkpoint:
        # One stored step, consistent with the counter in every other respect,
        # so only the non-finite energy is wrong.
        checkpoint["rank_0/driver_config"].attrs["mcmc_counter"] = 1
        observables = checkpoint["rank_0/observables"]
        history = {"e_L": [[np.inf]], "e_L2": [[1.0]], "w_L": [[1.0]]}
        if driver_kind == "t":
            history["average_projection_counter"] = [1.0]
        for name, values in history.items():
            if name in observables:
                del observables[name]
            observables.create_dataset(name, data=np.array(values))

    with pytest.raises(FloatingPointError, match=rf"(?s)GFMC_{driver_kind}.*checkpoint load.*stored_e_L"):
        type(driver).load_from_hdf5(merged, rank=0)


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["t", "n"])
def test_production_step_rejects_nonfinite_energy_before_aggregation(monkeypatch, driver_kind):
    """Both GFMC production loops bind their walker values to the shared gate."""
    from jqmc import jqmc_gfmc as gfmc_module

    boundary = gfmc_module._raise_collective_nonfinite_observables

    def _inject_nonfinite_energy(comm, **kwargs):
        if kwargs["stage"] == "pre-aggregation":
            observables = dict(kwargs["observables"])
            local_energy = np.asarray(observables["e_L"]).copy()
            local_energy.flat[0] = np.nan
            observables["e_L"] = local_energy
            kwargs["observables"] = observables
        return boundary(comm, **kwargs)

    monkeypatch.setattr(gfmc_module, "_raise_collective_nonfinite_observables", _inject_nonfinite_energy)
    make_driver = _make_gfmc_t if driver_kind == "t" else _make_gfmc_n
    driver = make_driver(_build_hamiltonian(TREXIO_FILES[0], "none"), num_walkers=2)
    with pytest.raises(FloatingPointError, match=rf"(?s)GFMC_{driver_kind}.*pre-aggregation.*field=e_L"):
        driver.run(num_mcmc_steps=1)


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["t", "n"])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "all-missing",
        "legacy",
        "unknown",
        "version-missing",
        "version-unknown",
        "version-bool",
        "algorithm",
        "keys-shape",
        "keys-dtype",
        "keys-zero",
        "position",
        "has-gauss",
        "nan-cache",
    ],
)
def test_checkpoint_rejects_corrupt_branching_rng(tmp_path, driver_kind, corruption):
    """An incomplete, invalid or legacy (absent) branching state is rejected rather than silently reseeded."""
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    make_driver = _make_gfmc_t if driver_kind == "t" else _make_gfmc_n
    save_driver = _save_merge_gfmc_t if driver_kind == "t" else _save_merge_gfmc_n
    driver = make_driver(hd, num_walkers=2)
    merged = save_driver(driver, hd, tmp_path)
    with h5py.File(merged, "r+") as checkpoint:
        rng = checkpoint["rank_0/rng_state"]
        if corruption == "missing":
            del rng.attrs["branching_rng_position"]
        elif corruption in ("all-missing", "legacy"):
            del rng["branching_rng_keys"]
            for key in tuple(rng.attrs):
                if key.startswith("branching_rng_"):
                    del rng.attrs[key]
            if corruption == "legacy":
                # A pre-branching-state checkpoint is rejected, not reseeded.
                del checkpoint["rank_0/driver_config"].attrs["branching_rng_state_version"]
        elif corruption == "unknown":
            rng.attrs["branching_rng_extra"] = 0
        elif corruption == "version-missing":
            del checkpoint["rank_0/driver_config"].attrs["branching_rng_state_version"]
        elif corruption == "version-unknown":
            checkpoint["rank_0/driver_config"].attrs["branching_rng_state_version"] = 2
        elif corruption == "version-bool":
            checkpoint["rank_0/driver_config"].attrs["branching_rng_state_version"] = True
        elif corruption == "algorithm":
            rng.attrs["branching_rng_algorithm"] = "PCG64"
        elif corruption in ("keys-shape", "keys-dtype"):
            keys = rng["branching_rng_keys"][()]
            del rng["branching_rng_keys"]
            rng.create_dataset("branching_rng_keys", data=keys[:-1] if corruption == "keys-shape" else keys.astype(np.int64))
        elif corruption == "keys-zero":
            rng["branching_rng_keys"][...] = 0
        elif corruption == "position":
            rng.attrs["branching_rng_position"] = 625
        elif corruption == "has-gauss":
            rng.attrs["branching_rng_has_gauss"] = 2
        else:
            rng.attrs["branching_rng_cached_gaussian"] = np.nan
    with pytest.raises(ValueError, match="(?:Incomplete or unknown|Invalid) GFMC branching RNG checkpoint"):
        type(driver).load_from_hdf5(merged, rank=0)


@pytest.mark.light
@pytest.mark.parametrize("implementation", ["production", "debug"])
@pytest.mark.parametrize("rows", [0, 1, 4, 5, 6, 9])
@pytest.mark.parametrize("collect", [0, 1, 5])
@pytest.mark.parametrize("walkers", [1, 3])
def test_accumulated_weights_match_independent_window_products(implementation, rows, collect, walkers):
    """Historical weight products match explicit NumPy windows, including the empty-history boundary.

    Parameter-set input only: no wavefunction or electrons. Exact powers of two
    make products bit-exact. Each output precedes one retained measurement;
    at most collect rows produce shape (0, walkers), while a zero-width product
    is one. Both helper implementations must obey the same indexing convention.
    """
    from jqmc.jqmc_gfmc import _compute_G_L, _compute_G_L_debug

    weights = np.power(2.0, np.arange(rows * walkers) % 5 - 2).reshape(rows, walkers)
    reference = np.asarray([np.prod(weights[i - collect : i], axis=0) for i in range(collect, rows)]).reshape(-1, walkers)
    compute = _compute_G_L if implementation == "production" else _compute_G_L_debug
    actual = compute(jax.numpy.asarray(weights), collect)
    assert actual.shape == (max(rows - collect, 0), walkers)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["t", "n"])
def test_restart_rejects_doctored_tied_sd_down_spin(tmp_path, driver_kind):
    """Both GFMC restart drivers reject a corrupted tied-spin occupied SD."""
    if driver_kind == "t":
        from jqmc.jqmc_gfmc import GFMC_t as driver_class

        hamiltonian_data = _build_hamiltonian(TREXIO_FILES[0], "none")
        driver = _make_gfmc_t(hamiltonian_data, num_walkers=2)
        merged = _save_merge_gfmc_t(driver, hamiltonian_data, tmp_path)
    else:
        from jqmc.jqmc_gfmc import GFMC_n as driver_class

        hamiltonian_data = _build_hamiltonian(TREXIO_FILES[0], "none")
        driver = _make_gfmc_n(hamiltonian_data, num_walkers=2)
        merged = _save_merge_gfmc_n(driver, hamiltonian_data, tmp_path)

    with h5py.File(merged, "r+") as checkpoint:
        down_coefficients = checkpoint["hamiltonian_data/wavefunction_data/geminal_data/orb_data_dn_spin/mo_coefficients"]
        down_coefficients[...] = down_coefficients[...] + 0.125

    with pytest.raises(ValueError, match="time_reversal=True requires down-spin occupied coefficients"):
        driver_class.load_from_hdf5(merged, rank=0)


# ===========================================================================
# GFMC_t tests
# ===========================================================================


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestGFMCtSaveLoadRoundtrip:
    """Round-trip tests: GFMC_t -> save -> merge -> load -> compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    @pytest.mark.heavy
    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=3)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert gfmc2.mcmc_counter == gfmc.mcmc_counter
        assert gfmc2.num_walkers == gfmc.num_walkers
        assert gfmc2.alat == gfmc.alat

        # Observables
        np.testing.assert_array_equal(np.asarray(gfmc2.bare_w_L), np.asarray(gfmc.bare_w_L))

    @pytest.mark.heavy
    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(gfmc._latest_r_up_carts)
        r_dn_before = np.asarray(gfmc._latest_r_dn_carts)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._latest_r_dn_carts),
            r_dn_before,
        )

    @pytest.mark.heavy
    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(gfmc._jax_PRNG_key_list)
        keys_init_before = np.asarray(gfmc._jax_PRNG_key_list_init)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._jax_PRNG_key_list),
            keys_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._jax_PRNG_key_list_init),
            keys_init_before,
        )

    @pytest.mark.heavy
    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, survived/killed) are preserved."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=5)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2._mcmc_counter == gfmc._mcmc_counter
        assert gfmc2._num_survived_walkers == gfmc._num_survived_walkers
        assert gfmc2._num_killed_walkers == gfmc._num_killed_walkers

    @pytest.mark.heavy
    def test_continue_run_after_load(self):
        """Conventional-algorithm checkpoint continuation matches uninterrupted values.

        Identical seeds and Hamiltonians define six deterministic steps, with
        one path checkpointed after three; nonempty energies, weights and walker RNG agree.
        """
        from jqmc.jqmc_gfmc import GFMC_t

        uninterrupted = _make_gfmc_t(self.hd)
        uninterrupted.run(num_mcmc_steps=6)
        uninterrupted_branching_rng = uninterrupted._branching_rng.get_state()
        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=3)

        counter_before = gfmc._mcmc_counter

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        gfmc2.run(num_mcmc_steps=3)
        assert gfmc2._mcmc_counter == counter_before + 3
        # The driver-owned stream supplies the actual branching draw. Equal walker
        # outcomes in a short trajectory do not prove this stream continued.
        resumed_branching_rng = gfmc2._branching_rng.get_state()
        assert resumed_branching_rng[0] == uninterrupted_branching_rng[0]
        np.testing.assert_array_equal(resumed_branching_rng[1], uninterrupted_branching_rng[1])
        assert resumed_branching_rng[2:] == uninterrupted_branching_rng[2:]
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
        assert gfmc2.e_L.size > 0 and uninterrupted.e_L.size > 0
        assert gfmc2.w_L.size > 0 and uninterrupted.w_L.size > 0
        np.testing.assert_allclose(gfmc2._stored_e_L, uninterrupted._stored_e_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.bare_w_L, uninterrupted.bare_w_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.e_L, uninterrupted.e_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.w_L, uninterrupted.w_L, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(gfmc2._jax_PRNG_key_list, uninterrupted._jax_PRNG_key_list)
        np.testing.assert_array_equal(gfmc2._latest_r_up_carts, uninterrupted._latest_r_up_carts)
        np.testing.assert_array_equal(gfmc2._latest_r_dn_carts, uninterrupted._latest_r_dn_carts)

    @pytest.mark.heavy
    def test_hamiltonian_embedded(self):
        """The restored GFMC_t has a functional hamiltonian_data."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        hd = gfmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    @pytest.mark.heavy
    def test_gfmc_t_specific_config_roundtrip(self):
        """GFMC_t-specific configs (tau, non_local_move, etc.) survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd, tau=0.05, alat=0.2, non_local_move="tmove")
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2._tau == 0.05
        assert gfmc2._alat == 0.2
        assert gfmc2._non_local_move == "tmove"
        assert gfmc2._num_gfmc_collect_steps == gfmc._num_gfmc_collect_steps

    @pytest.mark.heavy
    def test_force_derivative_roundtrip(self):
        """Force observables survive round-trip when comput_position_deriv=True."""
        from jqmc.jqmc_gfmc import GFMC_t

        gfmc = _make_gfmc_t(self.hd, comput_position_deriv=True)
        gfmc.run(num_mcmc_steps=2)

        merged = _save_merge_gfmc_t(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_t.load_from_hdf5(merged, rank=0)

        assert gfmc2.comput_position_deriv is True
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_force_HF),
            np.asarray(gfmc._stored_force_HF),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_force_PP),
            np.asarray(gfmc._stored_force_PP),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_E_L_force_PP),
            np.asarray(gfmc._stored_E_L_force_PP),
        )


# ===========================================================================
# GFMC_n tests
# ===========================================================================


@pytest.mark.heavy
@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize(
    "trexio_file,jastrow_combo",
    [("H2_ae_ccpvdz_cart.h5", "none"), ("H2_ecp_ccpvdz_cart.h5", "1b+2b+3b")],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_force_enabled_restart_matches_continuous_history(tmp_path, driver_kind, trexio_file, jastrow_combo):
    """Force-enabled GFMC continuation preserves actual histories, positions and RNG.

    Both GFMC algorithms use committed molecular H2 Cartesian GTO/MO with
    one up/one down electron: AE without Jastrow, or ECP with J1+J2+J3.
    Four walkers, seed 12345 and a fixed mesh define six measurements versus
    three, checkpoint save/load, then three. No cond(G) filter is applied.

    Energy/weight and HF, Pulay and energy-times-Pulay histories must be
    nonempty, finite and nonzero, and agree under the existing same-mode
    strict policy. Positions and both RNG states agree exactly. This is a
    restart-route oracle, not independent force physics or a statistical test.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    ham = _build_hamiltonian(trexio_file, jastrow_combo)
    make = _make_gfmc_t if driver_kind == "conventional" else _make_gfmc_n
    save = _save_merge_gfmc_t if driver_kind == "conventional" else _save_merge_gfmc_n
    kwargs = dict(num_walkers=4, mcmc_seed=12345, random_discretized_mesh=False, comput_position_deriv=True)
    continuous = make(ham, **kwargs)
    continuous.run(num_mcmc_steps=6)
    interrupted = make(ham, **kwargs)
    interrupted.run(num_mcmc_steps=3)
    checkpoint = save(interrupted, ham, tmp_path)
    resumed = type(interrupted).load_from_hdf5(checkpoint, rank=0)
    resumed.run(num_mcmc_steps=3)
    assert resumed._mcmc_counter == continuous._mcmc_counter == 6
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    for field in ("_stored_e_L", "_stored_w_L", "_stored_force_HF", "_stored_force_PP", "_stored_E_L_force_PP"):
        actual, reference = np.asarray(getattr(resumed, field)), np.asarray(getattr(continuous, field))
        assert actual.size > 0 and actual.shape == reference.shape
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        assert np.any(reference != 0), field
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=field)
    for field in ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list"):
        actual, reference = np.asarray(getattr(resumed, field)), np.asarray(getattr(continuous, field))
        np.testing.assert_array_equal(actual, reference, err_msg=field)
    actual_rng, reference_rng = resumed._branching_rng.get_state(), continuous._branching_rng.get_state()
    assert actual_rng[0] == reference_rng[0]
    np.testing.assert_array_equal(actual_rng[1], reference_rng[1])
    assert actual_rng[2:] == reference_rng[2:]


@pytest.mark.heavy
@pytest.mark.parametrize("driver_kind", ["conventional", "load-balanced"])
@pytest.mark.parametrize("force_field", ["force_HF", "force_PP", "E_L_force_PP"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_gfmc_force_restart_verdict_rejects_corrupt_history(tmp_path, monkeypatch, driver_kind, force_field):
    """Each resumed force-history channel is checked after a passing real-driver baseline.

    Both algorithms use the same AE H2, no-Jastrow, fixed-seed four-walker
    fixture as the continuation test. Change only one saved force channel's
    first row before the real resumed run; the numerical history comparison
    must fail, not an unrelated I/O or driver exception.
    """
    from jqmc import _checkpoint

    check = test_force_enabled_restart_matches_continuous_history
    check(tmp_path / "baseline", driver_kind, "H2_ae_ccpvdz_cart.h5", "none")
    original = _checkpoint.load_rank_checkpoint

    def corrupted(filepath, rank):
        data = original(filepath, rank)
        data["observables"][force_field][0] += 1
        return data

    monkeypatch.setattr(_checkpoint, "load_rank_checkpoint", corrupted)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        check(tmp_path / "corrupted", driver_kind, "H2_ae_ccpvdz_cart.h5", "none")


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestGFMCnSaveLoadRoundtrip:
    """Round-trip tests: GFMC_n -> save -> merge -> load -> compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    @pytest.mark.heavy
    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=3)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert gfmc2.mcmc_counter == gfmc.mcmc_counter
        assert gfmc2.num_walkers == gfmc.num_walkers
        assert gfmc2.alat == gfmc.alat

        # Observables
        np.testing.assert_array_equal(np.asarray(gfmc2.bare_w_L), np.asarray(gfmc.bare_w_L))

    @pytest.mark.heavy
    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(gfmc._latest_r_up_carts)
        r_dn_before = np.asarray(gfmc._latest_r_dn_carts)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._latest_r_dn_carts),
            r_dn_before,
        )

    @pytest.mark.heavy
    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(gfmc._jax_PRNG_key_list)
        keys_init_before = np.asarray(gfmc._jax_PRNG_key_list_init)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(gfmc2._jax_PRNG_key_list),
            keys_before,
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._jax_PRNG_key_list_init),
            keys_init_before,
        )

    @pytest.mark.heavy
    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, survived/killed) are preserved."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=5)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2._mcmc_counter == gfmc._mcmc_counter
        assert gfmc2._num_survived_walkers == gfmc._num_survived_walkers
        assert gfmc2._num_killed_walkers == gfmc._num_killed_walkers

    @pytest.mark.heavy
    def test_continue_run_after_load(self):
        """Load-balanced-algorithm checkpoint continuation matches uninterrupted values.

        Identical seeds and Hamiltonians define six deterministic steps, with
        one path checkpointed after three; nonempty energies, weights and walker RNG agree.
        """
        from jqmc.jqmc_gfmc import GFMC_n

        uninterrupted = _make_gfmc_n(self.hd)
        uninterrupted.run(num_mcmc_steps=6)
        uninterrupted_branching_rng = uninterrupted._branching_rng.get_state()
        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=3)

        counter_before = gfmc._mcmc_counter

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        gfmc2.run(num_mcmc_steps=3)
        assert gfmc2._mcmc_counter == counter_before + 3
        resumed_branching_rng = gfmc2._branching_rng.get_state()
        assert resumed_branching_rng[0] == uninterrupted_branching_rng[0]
        np.testing.assert_array_equal(resumed_branching_rng[1], uninterrupted_branching_rng[1])
        assert resumed_branching_rng[2:] == uninterrupted_branching_rng[2:]
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
        assert gfmc2.e_L.size > 0 and uninterrupted.e_L.size > 0
        assert gfmc2.w_L.size > 0 and uninterrupted.w_L.size > 0
        np.testing.assert_allclose(gfmc2._stored_e_L, uninterrupted._stored_e_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.bare_w_L, uninterrupted.bare_w_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.e_L, uninterrupted.e_L, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gfmc2.w_L, uninterrupted.w_L, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(gfmc2._jax_PRNG_key_list, uninterrupted._jax_PRNG_key_list)
        np.testing.assert_array_equal(gfmc2._latest_r_up_carts, uninterrupted._latest_r_up_carts)
        np.testing.assert_array_equal(gfmc2._latest_r_dn_carts, uninterrupted._latest_r_dn_carts)

    @pytest.mark.heavy
    def test_hamiltonian_embedded(self):
        """The restored GFMC_n has a functional hamiltonian_data."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        hd = gfmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    @pytest.mark.heavy
    def test_gfmc_n_specific_config_roundtrip(self):
        """GFMC_n-specific configs (E_scf, num_mcmc_per_measurement, etc.) survive round-trip."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd, E_scf=-1.5, alat=0.2, num_mcmc_per_measurement=8)
        gfmc.run(num_mcmc_steps=1)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2._E_scf == -1.5
        assert gfmc2._alat == 0.2
        assert gfmc2._num_mcmc_per_measurement == 8
        assert gfmc2._non_local_move == gfmc._non_local_move
        assert gfmc2._num_gfmc_collect_steps == gfmc._num_gfmc_collect_steps

    @pytest.mark.heavy
    def test_force_derivative_roundtrip(self):
        """Force observables survive round-trip when comput_position_deriv=True."""
        from jqmc.jqmc_gfmc import GFMC_n

        gfmc = _make_gfmc_n(self.hd, comput_position_deriv=True)
        gfmc.run(num_mcmc_steps=2)

        merged = _save_merge_gfmc_n(gfmc, self.hd, self.tmp_path)
        gfmc2 = GFMC_n.load_from_hdf5(merged, rank=0)

        assert gfmc2.comput_position_deriv is True
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_force_HF),
            np.asarray(gfmc._stored_force_HF),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_force_PP),
            np.asarray(gfmc._stored_force_PP),
        )
        np.testing.assert_array_equal(
            np.asarray(gfmc2._stored_E_L_force_PP),
            np.asarray(gfmc._stored_E_L_force_PP),
        )
