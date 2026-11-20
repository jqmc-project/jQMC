"""Tests for MCMC.save_to_hdf5 / MCMC.load_from_hdf5."""

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
from jqmc._precision import get_tolerance_min
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
    """Build a Hamiltonian_data from a trexio file with a given Jastrow combination.

    Args:
        trexio_file: Basename of the trexio file (e.g. ``"H2_ae_ccpvdz_cart.h5"``).
        jastrow_combo: One of ``JASTROW_COMBOS``.

    Returns:
        Hamiltonian_data with the requested Jastrow configuration.
    """
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


def _make_mcmc(hamiltonian_data, num_walkers=4, mcmc_seed=12345, **kwargs):
    """Create and return an MCMC instance."""
    from jqmc.jqmc_mcmc import MCMC

    return MCMC(
        hamiltonian_data=hamiltonian_data,
        num_walkers=num_walkers,
        mcmc_seed=mcmc_seed,
        **kwargs,
    )


def _save_merge(mcmc, hd, tmp_path, mpi_size=1, rank=0):
    """Save one rank and merge into a checkpoint HDF5 file."""
    rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
    mcmc.save_to_hdf5(rank_file)

    merged = str(tmp_path / "restart.h5")
    tmp_pattern = str(tmp_path / "._restart_rank{rank}.h5")
    merge_rank_checkpoints(
        output_path=merged,
        mpi_size=mpi_size,
        driver_type="MCMC",
        hamiltonian_data=hd,
        tmp_pattern=tmp_pattern,
    )
    return merged


@pytest.mark.light
@pytest.mark.parametrize("geminal_class", ["AGP", "AGPn"])
def test_restart_refuses_common_twist_independent_agp_and_agpn(geminal_class, tmp_path):
    """A restart checkpoint whose AGP / AGPn spins share one twist is refused by MCMC.load_from_hdf5.

    An independent-spin (KUHF oxygen, general twist) opposite-twist geminal is
    converted to AGP / AGPn, sampled briefly and checkpointed through the real
    writer; the checkpoint restarts. Rewriting only the stored down-spin twist
    to the up twist gives a common-twist checkpoint, which the restart
    reader must refuse with the regeneration error.
    """
    from jqmc.determinant import to_AGP, to_AGPn
    from jqmc.jqmc_mcmc import MCMC

    source = str(Path(__file__).resolve().parents[1] / "trexio_example_files" / "oxygen_tiling_sectors_sphe.h5")
    structure, _, _, _, sd, coulomb = read_trexio_file(source, store_tuple=True, twist_ik=2)
    assert not sd.time_reversal and sd.num_electron_up > sd.num_electron_dn
    geminal = to_AGP(sd) if geminal_class == "AGP" else to_AGPn(to_AGP(sd))
    hd = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal),
    )
    hd.sanity_check()
    merged = _save_merge(_make_mcmc(hd, num_walkers=2), hd, tmp_path)
    MCMC.load_from_hdf5(merged, rank=0)
    with h5py.File(merged, "r+") as checkpoint:
        geminal_group = checkpoint["hamiltonian_data/wavefunction_data/geminal_data"]
        twists = []
        geminal_group.visit(lambda name: twists.append(name) if name.endswith("k_point") else None)
        down = [name for name in twists if name.startswith("orb_data_dn_spin")]
        assert down
        for name in down:
            geminal_group[name][...] = geminal_group[name.replace("orb_data_dn_spin", "orb_data_up_spin", 1)][...]
    with pytest.raises(ValueError, match="regenerate"):
        MCMC.load_from_hdf5(merged, rank=0)


@pytest.mark.light
def test_restart_rejects_doctored_tied_sd_down_spin(tmp_path):
    """Checkpoint loading enforces the same tied-SD contract as a plain HDF5 load."""
    from jqmc.jqmc_mcmc import MCMC

    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    mcmc = _make_mcmc(hd, num_walkers=2)
    merged = _save_merge(mcmc, hd, tmp_path)
    with h5py.File(merged, "r+") as checkpoint:
        down_coefficients = checkpoint["hamiltonian_data/wavefunction_data/geminal_data/orb_data_dn_spin/mo_coefficients"]
        down_coefficients[...] = down_coefficients[...] + 0.125

    with pytest.raises(ValueError, match="time_reversal=True requires down-spin occupied coefficients"):
        MCMC.load_from_hdf5(merged, rank=0)


@pytest.mark.light
def test_restart_rejects_rank_local_nonfinite_energy_history(tmp_path):
    """A poisoned rank history is rejected before its walker can continue."""
    from jqmc.jqmc_mcmc import MCMC

    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    mcmc = _make_mcmc(hd, num_walkers=2)
    merged = _save_merge(mcmc, hd, tmp_path)
    with h5py.File(merged, "r+") as checkpoint:
        # One stored step, consistent with the counter in every other respect,
        # so only the non-finite energy is wrong.
        checkpoint["rank_0/driver_config"].attrs["mcmc_counter"] = 1
        observables = checkpoint["rank_0/observables"]
        for name, values in {"e_L": [[np.nan, -1.0]], "e_L2": [[1.0, 1.0]], "w_L": [[1.0, 1.0]]}.items():
            if name in observables:
                del observables[name]
            observables.create_dataset(name, data=np.array(values))

    with pytest.raises(FloatingPointError, match=r"(?s)MCMC.*checkpoint load.*step=0.*walker=0.*stored_e_L"):
        MCMC.load_from_hdf5(merged, rank=0)


@pytest.mark.light
@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
def test_restart_rejects_far_walker_when_fresh_determinant_underflows(tmp_path, monkeypatch, rust_enabled):
    """A finite-history tail walker cannot enter the first continuation sweep.

    The doctored coordinate models the production incident: the old observable
    history is finite, but rebuilding Gaussian orbitals at 123 bohr makes the
    fresh determinant zero.  Validation reuses the normal run-seed geminal and
    therefore adds no separate evaluation to ordinary fresh runs.
    """
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    mcmc = _make_mcmc(hd, num_walkers=1)
    merged = _save_merge(mcmc, hd, tmp_path)
    with h5py.File(merged, "r+") as checkpoint:
        coordinates = checkpoint["rank_0/walker_state/latest_r_dn_carts"]
        coordinates[0, 0] = np.array([-115.0, 40.0, 15.0])

    resumed = MCMC.load_from_hdf5(merged, rank=0)
    with pytest.raises(
        FloatingPointError,
        match=r"(?s)MCMC.*restart state rebuild.*walker=0.*ln_abs_det.*max_abs_position=115",
    ):
        resumed.run(num_mcmc_steps=1)


@pytest.mark.light
@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
def test_restart_rejects_nonfinite_fresh_local_energy(tmp_path, monkeypatch, rust_enabled):
    """The selected backend's fresh local energy is part of the restart verdict."""
    from jqmc import jqmc_mcmc as mcmc_module
    from jqmc.jqmc_mcmc import MCMC

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    hd = _build_hamiltonian(TREXIO_FILES[0], "none")
    merged = _save_merge(_make_mcmc(hd, num_walkers=1), hd, tmp_path)
    resumed = MCMC.load_from_hdf5(merged, rank=0)

    if rust_enabled:
        import jqmc_rs

        build_obs_ctx = jqmc_rs.build_obs_ctx

        class _NonfiniteLocalEnergyContext:
            def __init__(self, context):
                self.context = context

            def __getattr__(self, name):
                return getattr(self.context, name)

            def e_L_batch(self, r_up, *args):
                del args
                return np.full(np.shape(r_up)[0], np.nan)

        monkeypatch.setattr(
            jqmc_rs,
            "build_obs_ctx",
            lambda *args, **kwargs: _NonfiniteLocalEnergyContext(build_obs_ctx(*args, **kwargs)),
        )
    else:

        def _nonfinite_local_energy(_ham, r_up, _r_dn, _rts, _geminal_inv, _det_state):
            return jax.device_put(np.full(np.shape(r_up)[0], np.nan))

        monkeypatch.setattr(mcmc_module, "_jit_vmap_e_L_fast", _nonfinite_local_energy)

    with pytest.raises(FloatingPointError, match=r"(?s)MCMC.*restart state rebuild.*walker=0.*field=e_L"):
        resumed.run(num_mcmc_steps=1)


@pytest.mark.light
@pytest.mark.parametrize("rust_enabled", [False, True], ids=["xla", "rust"])
def test_production_step_rejects_nonfinite_energy_before_publication(monkeypatch, rust_enabled):
    """The XLA measurement and native chunk both bind to the shared gate."""
    from jqmc import jqmc_mcmc as mcmc_module

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1" if rust_enabled else "0")
    boundary = mcmc_module._raise_collective_nonfinite_observables

    def _inject_nonfinite_energy(comm, **kwargs):
        if kwargs["stage"] in ("measurement", "native energy chunk"):
            observables = dict(kwargs["observables"])
            local_energy = np.asarray(observables["e_L"]).copy()
            local_energy.flat[0] = np.nan
            observables["e_L"] = local_energy
            kwargs["observables"] = observables
        return boundary(comm, **kwargs)

    monkeypatch.setattr(mcmc_module, "_raise_collective_nonfinite_observables", _inject_nonfinite_energy)
    driver = _make_mcmc(_build_hamiltonian(TREXIO_FILES[0], "none"), num_walkers=1)
    with pytest.raises(FloatingPointError, match=r"(?s)MCMC.*(measurement|native energy chunk).*field=e_L"):
        driver.run(num_mcmc_steps=1)


@pytest.mark.light
def test_complex_rust_restart_rebuild_uses_batched_native_seed(tmp_path, monkeypatch):
    """Complex restart validation reuses one parsed native context for all walkers."""
    import jqmc_rs

    from jqmc.jqmc_mcmc import MCMC
    from tests.test_rs_kernel._rs_fixtures import rs_ham

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham = rs_ham("twist_sphepgto_jnone")
    merged = _save_merge(
        _make_mcmc(ham, num_walkers=1, random_discretized_mesh=False),
        ham,
        tmp_path,
    )
    resumed = MCMC.load_from_hdf5(merged, rank=0)
    build_obs_ctx = jqmc_rs.build_obs_ctx
    seed_calls = []

    class _ComplexSeedRecordingContext:
        def __init__(self, context):
            self.context = context

        def __getattr__(self, name):
            return getattr(self.context, name)

        def sweep_seed_geminal_complex_batch(self, r_up, r_dn):
            seed_calls.append(np.shape(r_up)[0])
            return self.context.sweep_seed_geminal_complex_batch(r_up, r_dn)

    monkeypatch.setattr(
        jqmc_rs,
        "build_obs_ctx",
        lambda *args, **kwargs: _ComplexSeedRecordingContext(build_obs_ctx(*args, **kwargs)),
    )
    resumed.run(num_mcmc_steps=1)
    assert seed_calls == [1]


# ---------------------------------------------------------------------------
# Tests -- basic round-trip (parameterized by trexio file x Jastrow combo)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", JASTROW_COMBOS)
class TestMCMCSaveLoadRoundtrip:
    """Round-trip tests: MCMC -> save -> merge -> load -> compare."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method."""
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    @pytest.mark.heavy
    def test_basic_roundtrip(self):
        """Scalars, walker state, and observables survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=3)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        # Scalar configs
        assert mcmc2.mcmc_counter == mcmc.mcmc_counter
        assert mcmc2.num_walkers == mcmc.num_walkers

        # Observables
        np.testing.assert_array_equal(np.asarray(mcmc2.e_L), np.asarray(mcmc.e_L))
        np.testing.assert_array_equal(np.asarray(mcmc2.e_L2), np.asarray(mcmc.e_L2))
        np.testing.assert_array_equal(np.asarray(mcmc2.w_L), np.asarray(mcmc.w_L))

    @pytest.mark.heavy
    def test_walker_state_roundtrip(self):
        """Walker coordinates survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=2)

        r_up_before = np.asarray(mcmc._MCMC__latest_r_up_carts)
        r_dn_before = np.asarray(mcmc._MCMC__latest_r_dn_carts)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__latest_r_up_carts),
            r_up_before,
        )
        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__latest_r_dn_carts),
            r_dn_before,
        )

    @pytest.mark.heavy
    def test_rng_state_roundtrip(self):
        """RNG keys survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=1)

        keys_before = np.asarray(mcmc._MCMC__jax_PRNG_key_list)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        np.testing.assert_array_equal(
            np.asarray(mcmc2._MCMC__jax_PRNG_key_list),
            keys_before,
        )

    @pytest.mark.heavy
    def test_counters_roundtrip(self):
        """Counters (mcmc_counter, accepted/rejected) are preserved."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=5)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        assert mcmc2._MCMC__mcmc_counter == mcmc._MCMC__mcmc_counter
        assert mcmc2._MCMC__accepted_moves == mcmc._MCMC__accepted_moves
        assert mcmc2._MCMC__rejected_moves == mcmc._MCMC__rejected_moves

    @pytest.mark.heavy
    def test_continue_run_after_load(self):
        """Checkpoint continuation preserves the uninterrupted trajectory and energies.

        Both paths use the same Hamiltonian, seed, precision and four steps;
        the resumed path writes and reloads its state after the second step.
        """
        from jqmc.jqmc_mcmc import MCMC

        uninterrupted = _make_mcmc(self.hd)
        uninterrupted.run(num_mcmc_steps=4)
        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        mcmc2.run(num_mcmc_steps=2)
        assert mcmc2.mcmc_counter == 4
        assert mcmc2.e_L.shape[0] == 4
        atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
        np.testing.assert_allclose(mcmc2.e_L, uninterrupted.e_L, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(mcmc2._MCMC__jax_PRNG_key_list, uninterrupted._MCMC__jax_PRNG_key_list)
        np.testing.assert_array_equal(mcmc2._MCMC__latest_r_up_carts, uninterrupted._MCMC__latest_r_up_carts)
        np.testing.assert_array_equal(mcmc2._MCMC__latest_r_dn_carts, uninterrupted._MCMC__latest_r_dn_carts)

    @pytest.mark.heavy
    def test_hamiltonian_embedded(self):
        """The restored MCMC has a functional hamiltonian_data."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd)
        mcmc.run(num_mcmc_steps=1)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        hd = mcmc2.hamiltonian_data
        assert hd is not None
        n_up = self.hd.wavefunction_data.geminal_data.num_electron_up
        n_dn = self.hd.wavefunction_data.geminal_data.num_electron_dn
        assert hd.wavefunction_data.geminal_data.num_electron_up == n_up
        assert hd.wavefunction_data.geminal_data.num_electron_dn == n_dn

    @pytest.mark.heavy
    def test_param_deriv_roundtrip(self):
        """Legacy gradients and exact C6 sources survive round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        dln_orig = mcmc.dln_Psi_dc
        dln_loaded = mcmc2.dln_Psi_dc
        assert set(dln_orig.keys()) == set(dln_loaded.keys())
        for k in dln_orig:
            np.testing.assert_array_equal(dln_loaded[k], dln_orig[k])

        blocks_orig = mcmc.hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=mcmc.wavefunction_state_token,
            opt_geminal=True,
        )
        blocks_loaded = mcmc2.hamiltonian_data.wavefunction_data.get_variational_blocks(
            state_token=mcmc2.wavefunction_state_token,
            opt_geminal=True,
        )
        sources_orig = mcmc._optimization_derivative_sources(blocks=blocks_orig)
        sources_loaded = mcmc2._optimization_derivative_sources(blocks=blocks_loaded)
        assert tuple(source.name for source in sources_loaded) == tuple(source.name for source in sources_orig)
        for before, after in zip(sources_orig, sources_loaded, strict=True):
            for component in ("direct", "d_p", "d_pbar"):
                before_values = getattr(before, component)
                after_values = getattr(after, component)
                if before_values is None:
                    assert after_values is None
                else:
                    np.testing.assert_array_equal(after_values, before_values)

    @pytest.mark.heavy
    def test_force_derivative_roundtrip(self):
        """A force-only checkpoint preserves force data and omits stress data."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_position_deriv=True, comput_stress=False)
        mcmc.run(num_mcmc_steps=2)

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        with h5py.File(merged, "r") as checkpoint:
            observables = checkpoint["rank_0/observables"]
            assert "stress_HF" not in observables
            assert "stress_PP" not in observables
            assert "E_L_stress_PP" not in observables

        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        assert mcmc2.comput_stress is False
        np.testing.assert_array_equal(np.asarray(mcmc2.force_HF_stored), np.asarray(mcmc.force_HF_stored))
        np.testing.assert_array_equal(np.asarray(mcmc2.force_PP_stored), np.asarray(mcmc.force_PP_stored))
        assert mcmc2.stress_HF_stored.shape[0] == 0
        assert mcmc2.stress_PP_stored.shape[0] == 0
        assert mcmc2.E_L_stress_PP_stored.shape[0] == 0
        with pytest.raises(ValueError, match="comput_stress=True"):
            mcmc2.get_aT(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=1)


# ---------------------------------------------------------------------------
# Tests -- optimizer state round-trip (only combos that have variational params)
# ---------------------------------------------------------------------------

# Jastrow combos that have variational parameters (needed for run_optimize)
_JASTROW_WITH_PARAMS = ["1b", "1b+2b", "1b+2b+3b"]  # NN combo removed (deprecated)


@pytest.mark.parametrize("trexio_file", TREXIO_FILES, ids=lambda f: f.replace(".h5", ""))
@pytest.mark.parametrize("jastrow_combo", _JASTROW_WITH_PARAMS)
class TestMCMCOptimizerRoundtrip:
    """SR optimizer state round-trip through MCMC save/load."""

    @pytest.fixture(autouse=True)
    def _setup(self, trexio_file, jastrow_combo, tmp_path):
        """Build hamiltonian_data once per test method.

        ``run_optimize`` writes ``hamiltonian_data_opt_step_*.h5`` checkpoint
        files relative to the current working directory. The autouse
        ``_isolate_cwd`` fixture in ``conftest.py`` already switches the CWD to
        a private ``tmp_path`` for every test, so parametrized runs (and
        pytest-xdist workers) do not race on the same filename. We still
        capture ``tmp_path`` here for the explicit save/load round-trip paths.
        """
        self.hd = _build_hamiltonian(trexio_file, jastrow_combo)
        self.tmp_path = tmp_path

    @pytest.mark.heavy
    def test_sr_optimizer_roundtrip(self):
        """SR optimizer state survives round-trip."""
        from jqmc.jqmc_mcmc import MCMC

        mcmc = _make_mcmc(self.hd, comput_log_WF_param_deriv=True)
        token_initial = mcmc.wavefunction_state_token
        mcmc.run_optimize(
            num_mcmc_steps=5,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=5,
            optimizer_kwargs={"method": "sr", "delta": 1e-3},
        )

        rt_before = mcmc._MCMC__optimizer_runtime
        assert rt_before["method"] == "sr"
        token_before = mcmc.wavefunction_state_token
        assert token_before[0] == token_initial[0]
        assert token_before[1] == token_initial[1] + 1

        merged = _save_merge(mcmc, self.hd, self.tmp_path)
        mcmc2 = MCMC.load_from_hdf5(merged, rank=0)

        rt_after = mcmc2._MCMC__optimizer_runtime
        assert rt_after["method"] == "sr"
        assert rt_after["hyperparameters"]["delta"] == 1e-3
        assert mcmc2.wavefunction_state_token == token_before
