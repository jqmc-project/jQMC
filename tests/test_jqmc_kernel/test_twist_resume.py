"""Single-twist checkpoint readback, overwrite and no-checkpoint behavior.

These molecular one-twist tests exercise the identity communicator path.
They do not establish partial recovery of a multi-twist sweep: that requires
both cached and freshly sampled twists with nonuniform weights, covered by
``test_twist_partial_resume_mpi.py`` using two actual MPI ranks.
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

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_TIGHT
from jqmc._twist_average import run_twist_average_from_hamiltonians, run_twist_average_gfmc
from jqmc._twist_checkpoint import is_twist_complete, twist_checkpoint_path
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_H_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
_h_available = _H_FIXTURE.is_file()
requires_h_fixture = pytest.mark.skipif(not _h_available, reason="H_ae_ccpvdz_cart.h5 fixture not present")


def _build_h_hamiltonian():
    sd, _, _, _, gem, cpd = read_trexio_file(str(_H_FIXTURE), store_tuple=True)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(),
            geminal_data=gem,
        ),
    )
    ham.sanity_check()
    return ham


_DRIVER_KWARGS = dict(
    mcmc_seed=12345,
    num_walkers=2,
    num_mcmc_steps=20,
    num_mcmc_warmup_steps=5,
    num_mcmc_bin_blocks=3,
)


class TestTwistResume:
    @pytest.mark.light
    @requires_h_fixture
    def test_resume_skips_completed_twist_and_reproduces_uninterrupted_E_TA(self, tmp_path):
        """One completed molecular H twist is read back without rewriting its file.

        The cached energy and single-twist average match the original run
        under the existing comparison tolerance. No second twist is present;
        this is a checkpoint readback test, not mixed cached/fresh aggregation.
        """
        ham = _build_h_hamiltonian()

        # --- partial run: twist 0 only, write its checkpoint ---
        ckpt_partial = tmp_path / "ckpt_partial"
        result_partial = run_twist_average_from_hamiltonians(
            hamiltonian_list=[ham],
            weights=[1.0],
            checkpoint_dir=str(ckpt_partial),
            **_DRIVER_KWARGS,
        )
        assert is_twist_complete(str(ckpt_partial), 0, _DRIVER_KWARGS["num_mcmc_steps"])
        partial_mtime = os.path.getmtime(twist_checkpoint_path(str(ckpt_partial), 0))

        # E_k cached for twist 0 -- will be reused on the resume run below.
        E_k0_partial = result_partial.per_twist_E[0]

        # --- resume run: same checkpoint_dir, ham list of just the same twist ---
        # Ensure file mtime resolution lets us detect a re-write if one happens.
        time.sleep(0.02)
        result_resume = run_twist_average_from_hamiltonians(
            hamiltonian_list=[ham],
            weights=[1.0],
            checkpoint_dir=str(ckpt_partial),
            resume=True,
            **_DRIVER_KWARGS,
        )
        # The completed twist's file was NOT re-written.
        assert os.path.getmtime(twist_checkpoint_path(str(ckpt_partial), 0)) == partial_mtime
        # The cached (E_k, sigma_k) was returned verbatim.
        assert result_resume.per_twist_E[0] == pytest.approx(E_k0_partial, abs=ATOL_PYTHON_FP64_TIGHT)
        # The BZ aggregate matches the original run (no second sample).
        assert result_resume.energy.E_TA == pytest.approx(result_partial.energy.E_TA, abs=ATOL_PYTHON_FP64_TIGHT)

    @pytest.mark.light
    @requires_h_fixture
    def test_resume_false_overwrites_existing_checkpoint(self, tmp_path):
        """With ``resume=False`` (the default) the driver re-runs every twist
        regardless of existing checkpoint files -- the certified-complete
        file is overwritten with a fresh chain. Pins the toggle: only
        explicit ``resume=True`` switches on the skip path."""
        ham = _build_h_hamiltonian()
        ckpt = tmp_path / "ckpt_overwrite"

        run_twist_average_from_hamiltonians(
            hamiltonian_list=[ham],
            weights=[1.0],
            checkpoint_dir=str(ckpt),
            **_DRIVER_KWARGS,
        )
        first_mtime = os.path.getmtime(twist_checkpoint_path(str(ckpt), 0))

        time.sleep(0.02)
        run_twist_average_from_hamiltonians(
            hamiltonian_list=[ham],
            weights=[1.0],
            checkpoint_dir=str(ckpt),
            resume=False,
            **_DRIVER_KWARGS,
        )
        # Without resume the file mtime DID advance -- the twist was re-sampled.
        assert os.path.getmtime(twist_checkpoint_path(str(ckpt), 0)) > first_mtime

    @pytest.mark.light
    @requires_h_fixture
    def test_no_checkpoint_dir_is_a_no_op_relative_to_pre_change_behaviour(self, tmp_path):
        """When ``checkpoint_dir`` is ``None`` (default) the driver behaves
        identically to the pre-checkpoint baseline. Pins that the new
        optional parameter is byte-identical when not used -- the
        pre-existing twist-driver tests must not start seeing checkpoint
        files materialize on disk."""
        ham = _build_h_hamiltonian()
        pre_existing = list(tmp_path.iterdir())  # empty
        result = run_twist_average_from_hamiltonians(
            hamiltonian_list=[ham],
            weights=[1.0],
            **_DRIVER_KWARGS,
        )
        post_run = list(tmp_path.iterdir())
        assert post_run == pre_existing  # no files created
        assert np.isfinite(result.energy.E_TA)


@pytest.mark.light
@requires_h_fixture
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
def test_fresh_run_killed_before_saving_leaves_no_converged_old_twist(tmp_path, monkeypatch, route):
    """A fresh run marks the old twist files unconverged before sampling.

    A fresh run rewrites a twist file only when that twist finishes, and
    jqmc-tool combines every twist file of the directory. A fresh run (seed 2)
    stopped before its twist is saved must therefore leave the complete
    directory of the older run (seed 1) refused by the manifest instead of
    reported as this run, and a resume must re-sample the twist, so that the
    directory average is the resumed run's ``E_TA``.
    """
    from jqmc._twist_checkpoint import load_twist_checkpoint_manifest, load_twist_checkpoint_summary
    from jqmc.jqmc_gfmc import GFMC_t
    from jqmc.jqmc_mcmc import MCMC

    ham = _build_h_hamiltonian()
    checkpoint_dir = str(tmp_path / "checkpoints")

    def run(seed, *, resume=False):
        common = dict(_DRIVER_KWARGS, hamiltonian_list=[ham], weights=[1.0], checkpoint_dir=checkpoint_dir, resume=resume)
        common["mcmc_seed"] = seed
        if route == "mcmc":
            return run_twist_average_from_hamiltonians(**common, mcmc_kwargs={"num_mcmc_per_measurement": 1})
        return run_twist_average_gfmc(**common, gfmc_kind="tau", gfmc_kwargs={"tau": 0.1, "alat": 0.5})

    def directory_average():
        manifest = load_twist_checkpoint_manifest(checkpoint_dir)
        return sum(
            weight * load_twist_checkpoint_summary(checkpoint_dir, idx)["E_k"] for idx, (_, weight) in enumerate(manifest)
        )

    old = run(1)
    assert directory_average() == pytest.approx(old.energy.E_TA, abs=ATOL_PYTHON_FP64_TIGHT)

    def stop(*args, **kwargs):
        raise RuntimeError("sampler stopped before the twist was saved")

    with monkeypatch.context() as patch:
        patch.setattr(MCMC if route == "mcmc" else GFMC_t, "run", stop)
        with pytest.raises(RuntimeError, match="sampler stopped"):
            run(2)
    with pytest.raises(ValueError, match=r"twist_000\.h5: twist is not converged"):
        load_twist_checkpoint_manifest(checkpoint_dir)

    resumed = run(2, resume=True)
    assert resumed.energy.E_TA != old.energy.E_TA
    assert directory_average() == pytest.approx(resumed.energy.E_TA, abs=ATOL_PYTHON_FP64_TIGHT)


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
def test_twist_driver_refuses_stale_checkpoint(tmp_path, route):
    """Both sampling drivers reject a stale identity before returning cached estimates."""
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / "lithium_twist_222.h5"

    def build_hamiltonian(twist_index, j2_param):
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True, twist_ik=twist_index)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=j2_param, structure_data=structure)
        ham = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data(jastrow_two_body_data=j2)),
        )
        ham.sanity_check()
        return ham

    def run(ham, cutoff, *, resume=False):
        common = dict(hamiltonian_list=[ham], weights=[1.0], checkpoint_dir=str(tmp_path), resume=resume, **_DRIVER_KWARGS)
        mpc = {"mpc": True, "mpc_cutoff": cutoff}
        if route == "mcmc":
            return run_twist_average_from_hamiltonians(**common, mcmc_kwargs={"num_mcmc_per_measurement": 1, **mpc})
        return run_twist_average_gfmc(**common, gfmc_kind="tau", gfmc_kwargs={"tau": 0.1, "alat": 0.5, **mpc})

    ham = build_hamiltonian(twist_index=0, j2_param=0.5)
    original = run(ham, 4.0)
    checkpoint = Path(twist_checkpoint_path(str(tmp_path), 0))
    saved_bytes = checkpoint.read_bytes()
    saved_mtime = checkpoint.stat().st_mtime_ns
    assert original.mpc is not None
    assert np.isfinite(original.energy.E_TA) and np.isfinite(original.mpc.energy.E_TA)

    matching = run(ham, 4.0, resume=True)
    assert matching.mpc is not None
    assert matching.energy.E_TA == original.energy.E_TA
    assert matching.mpc.energy.E_TA == original.mpc.energy.E_TA
    assert checkpoint.read_bytes() == saved_bytes
    assert checkpoint.stat().st_mtime_ns == saved_mtime

    changed_wavefunction = build_hamiltonian(twist_index=0, j2_param=2.0)
    changed_k = build_hamiltonian(twist_index=1, j2_param=0.5)
    with pytest.raises(ValueError, match="identity digest mismatch.*refusing to blend"):
        run(changed_wavefunction, 4.0, resume=True)
    with pytest.raises(ValueError, match="was written for k=.*refusing to blend"):
        run(changed_k, 4.0, resume=True)
    with pytest.raises(ValueError, match="refusing to reuse stale MPC observables"):
        run(ham, 8.0, resume=True)
    assert checkpoint.read_bytes() == saved_bytes
    assert checkpoint.stat().st_mtime_ns == saved_mtime


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
def test_twist_resume_rejects_a_changed_ecp(tmp_path, route):
    """An unchanged saved ECP can be reused, but a different ECP cannot."""
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / "Li_ecp_ccpvdz_cart.h5"
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )
    assert coulomb.ecp_flag and coulomb.coefficients and coulomb.exponents

    def run(hamiltonian, *, resume=False):
        common = dict(
            hamiltonian_list=[hamiltonian], weights=[1.0], checkpoint_dir=str(tmp_path), resume=resume, **_DRIVER_KWARGS
        )
        if route == "mcmc":
            return run_twist_average_from_hamiltonians(**common, mcmc_kwargs={"num_mcmc_per_measurement": 1})
        return run_twist_average_gfmc(**common, gfmc_kind="tau", gfmc_kwargs={"tau": 0.1, "alat": 0.5})

    original = run(ham)
    checkpoint = Path(twist_checkpoint_path(str(tmp_path), 0))
    saved = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
    assert np.isfinite(original.energy.E_TA)
    assert run(ham, resume=True).energy.E_TA == original.energy.E_TA
    assert (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns) == saved

    for field in ("coefficients", "exponents"):
        values = getattr(coulomb, field)
        changed = replace(coulomb, **{field: (2.0 * values[0], *values[1:])})
        with pytest.raises(ValueError, match=rf"different ECP \({field} mismatch\).*refusing to blend"):
            run(replace(ham, coulomb_potential_data=changed), resume=True)
    all_electron = type(coulomb)(structure_data=structure)
    with pytest.raises(ValueError, match=r"different ECP \(ecp_flag mismatch\).*refusing to blend"):
        run(replace(ham, coulomb_potential_data=all_electron), resume=True)
    assert (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns) == saved


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau", "gfmc_n"])
def test_twist_resume_rejects_different_sampler_settings(tmp_path, route):
    """Resume only reuses an estimate produced by the requested sampler."""
    ham = _build_h_hamiltonian()
    options = {
        "mcmc": {"num_mcmc_per_measurement": 1, "Dt": 1.25},
        "gfmc_tau": {"tau": 0.1, "alat": 0.5},
        "gfmc_n": {"num_mcmc_per_measurement": 2, "E_scf": -0.5, "alat": 0.5},
    }

    def run(driver, *, resume=False, sampler_changes=None, **control_changes):
        common = {**_DRIVER_KWARGS, **control_changes}
        common.update(hamiltonian_list=[ham], weights=[1.0], checkpoint_dir=str(tmp_path), resume=resume)
        sampler = {**options[driver], **(sampler_changes or {})}
        if driver == "mcmc":
            return run_twist_average_from_hamiltonians(**common, mcmc_kwargs=sampler)
        return run_twist_average_gfmc(**common, gfmc_kind=driver.removeprefix("gfmc_"), gfmc_kwargs=sampler)

    original = run(route)
    checkpoint = Path(twist_checkpoint_path(str(tmp_path), 0))
    saved = checkpoint.read_bytes()
    saved_mtime = checkpoint.stat().st_mtime_ns
    assert np.isfinite(original.per_twist_E[0])
    np.testing.assert_array_equal(run(route, resume=True).per_twist_E, original.per_twist_E)

    different_driver = {"mcmc": "gfmc_tau", "gfmc_tau": "gfmc_n", "gfmc_n": "gfmc_tau"}[route]
    with pytest.raises(ValueError, match=r"written by driver.*refusing to reuse"):
        run(different_driver, resume=True)
    changes = {
        "mcmc": {"Dt": 2.0, "num_mcmc_per_measurement": 3},
        "gfmc_tau": {"tau": 0.4, "alat": 0.1, "non_local_move": "dltmove"},
        "gfmc_n": {"E_scf": -0.4, "alat": 0.1, "num_mcmc_per_measurement": 4},
    }
    for name, value in changes[route].items():
        with pytest.raises(ValueError, match=rf"has {name}=.*refusing to reuse"):
            run(route, resume=True, sampler_changes={name: value})
    with pytest.raises(ValueError, match=r"has num_walkers=.*refusing to reuse"):
        run(route, resume=True, num_walkers=3)
    assert checkpoint.read_bytes() == saved
    assert checkpoint.stat().st_mtime_ns == saved_mtime

    if route == "gfmc_n":
        with h5py.File(checkpoint, "r+") as f:
            del f["_meta"].attrs["initial_E_scf"]
        assert not is_twist_complete(
            str(tmp_path), 0, _DRIVER_KWARGS["num_mcmc_steps"], driver_type="GFMC_n", sampler_config={"E_scf": -0.5}
        )
