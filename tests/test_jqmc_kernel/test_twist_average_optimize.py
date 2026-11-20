"""Twist-averaged SR/LM driver tests.

``run_twist_average_optimize`` is the optimization counterpart of
``run_twist_average_from_hamiltonians``. It plays the orchestrator role
(comm split with equal compute per twist, one MCMC instance
per twist) while delegating the actual SR/LM loop to
``MCMC.run_optimize``. The Plan B math lives in the shared real-coordinate
optimizer: shared Jastrow columns reduce over ``world_comm`` and per-twist
determinant columns reduce over ``intra_comm``. The orchestrator owns the
separate per-step BZ energy gather and two-stage broadcast.

These tests pin:

* the byte-identity exit at ``N_twist == 1`` (no extra
  comm splits, no extra MCMC steps, no extra reductions, no surprise
  parameter delta) -- the optimized Hamiltonian must equal what a plain
  ``MCMC.run_optimize`` would have produced;
* the explicit derivative-cache contract (SR/LM needs both deriv flags
  on; the driver enables them by default and refuses an explicit
  ``False``);
* the same input-validation contract used by the VMC / GFMC drivers
  (empty list, weights-vs-twist-count mismatch).

A two-twist routing correctness check is exercised at the helper
level in ``tests/test_jqmc_kernel/test_twist_average_two_comm.py``; the end-to-end multi-rank
optimization run is not part of this module.
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

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, RTOL_PYTHON_FP64_ZERO
import pytest
from jax.tree_util import tree_flatten

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jqmc._twist_average as twist_average_module
from jqmc._twist_average import run_twist_average_optimize
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_optimizer import EnergyEstimate
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_H_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
_h_available = _H_FIXTURE.is_file()

requires_h_fixture = pytest.mark.skipif(not _h_available, reason="H_ae_ccpvdz_cart.h5 TREXIO fixture not present")


class _RootIntraComm:
    def __init__(self):
        self.broadcasts = []

    def Get_rank(self):
        return 0

    def bcast(self, value, root):
        self.broadcasts.append((value, root))
        return value


class _RootInterComm:
    def __init__(self, gathered):
        self.gathered = gathered
        self.gather_calls = []
        self.broadcasts = []

    def Get_rank(self):
        return 0

    def gather(self, value, root):
        self.gather_calls.append((value, root))
        return self.gathered

    def bcast(self, value, root):
        self.broadcasts.append((value, root))
        return value


class _WorldComm:
    def __init__(self):
        self.barriers = 0

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        del op
        return value

    def Barrier(self):
        self.barriers += 1


def _build_h_hamiltonian_with_j2():
    """Build the H atom Hamiltonian with a J2 Jastrow block enabled.

    A free-standing helper (not a fixture) so the byte-identity test can
    build two independent copies -- one for the reference direct
    ``MCMC.run_optimize`` call, one for the orchestrator. Sharing a
    fixture would let the first call's in-place update leak into the
    second, defeating the byte-identity check.
    """
    sd, _, _, _, gem, cpd = read_trexio_file(str(_H_FIXTURE), store_tuple=True)
    jastrow = Jastrow_data(jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0))
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=gem),
    )
    ham.sanity_check()
    return ham


def _build_h_hamiltonian_with_j1_j2():
    """Retain the single-spin H/J2 input and add an active one-body parameter."""
    ham = _build_h_hamiltonian_with_j2()
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=ham.structure_data,
        core_electrons=tuple(0 for _ in ham.structure_data.atomic_numbers),
        jastrow_1b_type="exp",
    )
    return ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(
            jastrow_data=ham.wavefunction_data.jastrow_data.replace(jastrow_one_body_data=j1)
        )
    )


class TestTwistAverageOptimize:
    @pytest.mark.light
    def test_step_energy_is_aggregated_on_twist_roots_then_broadcast(self):
        """Every optimization step uses the weighted BZ estimate everywhere.

        Only one rank per twist participates in the inter-twist gather.  The
        root computes the BZ mean and independent-chain quadrature error, then
        broadcasts the resulting scalar estimate first among twist roots and
        finally to every sampler rank in each twist.
        """
        local = EnergyEstimate(
            mean=-1.0,
            standard_error=0.2,
            variance=2.0,
            variance_standard_error=0.3,
        )
        other = EnergyEstimate(
            mean=-3.0,
            standard_error=0.4,
            variance=6.0,
            variance_standard_error=0.5,
        )
        inter = _RootInterComm(gathered=[local, other])
        intra = _RootIntraComm()

        actual = twist_average_module._aggregate_twist_optimization_energy(
            local,
            twist_index=0,
            weights=np.asarray([1.0, 3.0]),
            intra_comm=intra,
            inter_comm=inter,
        )

        np.testing.assert_allclose(actual.mean, -2.5, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            actual.standard_error,
            np.sqrt((0.25 * 0.2) ** 2 + (0.75 * 0.4) ** 2),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            actual.variance, 0.25 * 2.0 + 0.75 * 6.0, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            actual.variance_standard_error,
            np.sqrt((0.25 * 0.3) ** 2 + (0.75 * 0.5) ** 2),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        assert inter.gather_calls == [(local, 0)]
        assert inter.broadcasts == [(actual, 0)]
        assert intra.broadcasts == [(actual, 0)]

    @pytest.mark.light
    def test_driver_wires_step_energy_aggregation_into_mcmc_optimizer(self, monkeypatch):
        """Driver wires step energy aggregation into MCMC optimizer."""
        local = EnergyEstimate(-1.0, 0.2, 2.0, 0.3)
        other = EnergyEstimate(-3.0, 0.4, 6.0, 0.5)
        inter = _RootInterComm(gathered=[local, other])
        intra = _RootIntraComm()
        world = _WorldComm()
        hamiltonians = [object(), object()]
        instances = []

        class FakeMCMC:
            def __init__(self, *, hamiltonian_data, **kwargs):
                del kwargs
                self.hamiltonian_data = hamiltonian_data
                self.i_opt = 0
                self.aggregated = None
                instances.append(self)

            def run_optimize(self, *, num_opt_steps, energy_aggregator, **kwargs):
                del kwargs
                self.aggregated = energy_aggregator(local)
                self.i_opt += num_opt_steps

        monkeypatch.setattr("jqmc.jqmc_mcmc.MCMC", FakeMCMC)
        monkeypatch.setattr(
            twist_average_module,
            "split_world_for_twists",
            lambda world_comm, n_twist: SimpleNamespace(
                world_comm=world_comm,
                intra_comm=intra,
                inter_comm=inter,
                twist_index=0,
                n_twist=n_twist,
            ),
        )

        actual_hamiltonian = run_twist_average_optimize(
            hamiltonian_list=hamiltonians,
            weights=[1.0, 3.0],
            world_comm=world,
            optimize_kwargs={"num_opt_steps": 1},
        )

        assert actual_hamiltonian is hamiltonians[0]
        assert len(instances) == 1
        np.testing.assert_allclose(
            instances[0].aggregated.mean, -2.5, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            instances[0].aggregated.standard_error,
            np.sqrt((0.25 * 0.2) ** 2 + (0.75 * 0.4) ** 2),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

    @pytest.mark.light
    def test_twist_kwargs_forward_the_device_optimizer_request_unchanged(self, monkeypatch):
        """The twist driver forwards ``use_device_optimizer`` unchanged; the route follows from the communicators."""
        calls = []

        class FakeMCMC:
            def __init__(self, *, hamiltonian_data, **kwargs):
                del kwargs
                self.hamiltonian_data = hamiltonian_data
                self.i_opt = 0

            def run_optimize(self, **kwargs):
                calls.append(dict(kwargs))
                self.i_opt += int(kwargs["num_opt_steps"])

        world = _WorldComm()
        intra = _RootIntraComm()
        inter = _RootInterComm(gathered=[])
        monkeypatch.setattr("jqmc.jqmc_mcmc.MCMC", FakeMCMC)
        monkeypatch.setattr(
            twist_average_module,
            "split_world_for_twists",
            lambda world_comm, n_twist: SimpleNamespace(
                world_comm=world_comm,
                intra_comm=intra,
                inter_comm=inter,
                twist_index=0,
                n_twist=n_twist,
            ),
        )
        for use_device_optimizer in (True, False):
            for hamiltonians in ([object(), object()], [object()]):
                requested = {"use_device_optimizer": use_device_optimizer, "num_mcmc_steps": 5, "num_opt_steps": 1}
                run_twist_average_optimize(
                    hamiltonian_list=hamiltonians,
                    weights=[1.0] * len(hamiltonians),
                    world_comm=world,
                    optimize_kwargs=requested,
                )
                assert calls[-1]["use_device_optimizer"] is use_device_optimizer
                assert "use_device_collectives" not in calls[-1]
                assert requested == {"use_device_optimizer": use_device_optimizer, "num_mcmc_steps": 5, "num_opt_steps": 1}

    @pytest.mark.light
    def test_twist_progress_guard_raises_instead_of_retrying_same_chunk(self, monkeypatch):
        """Twist progress guard raises instead of retrying same chunk."""

        class FakeMCMC:
            def __init__(self, *, hamiltonian_data, **kwargs):
                del kwargs
                self.hamiltonian_data = hamiltonian_data
                self.i_opt = 0

            def run_optimize(self, **kwargs):
                self.i_opt += int(kwargs["num_opt_steps"])

        class RemoteStalledWorld(_WorldComm):
            def allreduce(self, value, op=None):
                del value, op
                return 0

        monkeypatch.setattr("jqmc.jqmc_mcmc.MCMC", FakeMCMC)
        with pytest.raises(RuntimeError, match="no accepted progress"):
            run_twist_average_optimize(
                hamiltonian_list=[object()],
                weights=[1.0],
                world_comm=RemoteStalledWorld(),
                optimize_kwargs={"num_opt_steps": 1},
            )

        result = run_twist_average_optimize(
            hamiltonian_list=[object()],
            weights=[1.0],
            world_comm=_WorldComm(),
            optimize_kwargs={"num_opt_steps": 1},
        )
        assert result is not None

    # ---------- N_twist == 1 byte-identity ----------

    @pytest.mark.light
    @requires_h_fixture
    def test_n_twist_1_byte_identical_to_direct_run_optimize(self):
        """At ``N_twist == 1`` the orchestrator must produce the same
        optimized Hamiltonian as a direct ``MCMC.run_optimize`` call on the
        same ``world_comm`` (here ``COMM_WORLD``). Pins the
        byte-identity contract -- the comm-split short-circuit + the
        one-iteration loop must not perturb the chain.

        The committed H-atom input has one up electron and no down electrons.
        J1 exp must actually change, while J2 must remain unchanged because
        no electron pair exists. All optimized Hamiltonian leaves must match
        exactly between the fixed-seed direct and single-twist routes. This
        checks deterministic execution, not statistical energy convergence.
        """
        from jqmc.jqmc_mcmc import MCMC

        common_init = dict(
            mcmc_seed=12345,
            num_walkers=2,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        optimize_kwargs = dict(
            num_mcmc_steps=20,
            num_opt_steps=1,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-2, "epsilon": 1e-3},
        )

        # Reference: plain MCMC.run_optimize.
        ref_ham = _build_h_hamiltonian_with_j1_j2()
        assert ref_ham.wavefunction_data.geminal_data.num_electron_up == 1
        assert ref_ham.wavefunction_data.geminal_data.num_electron_dn == 0
        initial_j1 = np.asarray(ref_ham.wavefunction_data.jastrow_data.jastrow_one_body_data.jastrow_1b_param).copy()
        initial_j2 = np.asarray(ref_ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param).copy()
        ref_mcmc = MCMC(hamiltonian_data=ref_ham, **common_init)
        ref_mcmc.run_optimize(**optimize_kwargs)
        ref_jastrow = ref_mcmc.hamiltonian_data.wavefunction_data.jastrow_data
        ref_j1 = np.asarray(ref_jastrow.jastrow_one_body_data.jastrow_1b_param)
        assert np.all(np.isfinite(ref_j1))
        assert not np.array_equal(ref_j1, initial_j1), "the direct optimizer must update the active J1 parameter"
        np.testing.assert_array_equal(ref_jastrow.jastrow_two_body_data.jastrow_2b_param, initial_j2)

        # Orchestrator at N_twist = 1.
        orch_ham = _build_h_hamiltonian_with_j1_j2()
        out_ham = run_twist_average_optimize(
            hamiltonian_list=[orch_ham],
            weights=[1.0],
            mcmc_seed=common_init["mcmc_seed"],
            num_walkers=common_init["num_walkers"],
            optimize_kwargs=optimize_kwargs,
        )
        out_jastrow = out_ham.wavefunction_data.jastrow_data
        out_j1 = np.asarray(out_jastrow.jastrow_one_body_data.jastrow_1b_param)
        assert np.all(np.isfinite(out_j1))
        assert not np.array_equal(out_j1, initial_j1), "the twist optimizer must update the active J1 parameter"
        np.testing.assert_array_equal(out_j1, ref_j1)
        np.testing.assert_array_equal(out_jastrow.jastrow_two_body_data.jastrow_2b_param, initial_j2)
        ref_leaves, ref_tree = tree_flatten(ref_mcmc.hamiltonian_data)
        out_leaves, out_tree = tree_flatten(out_ham)
        assert out_tree == ref_tree
        for out_leaf, ref_leaf in zip(out_leaves, ref_leaves, strict=True):
            np.testing.assert_array_equal(out_leaf, ref_leaf)
            out_array, ref_array = np.asarray(out_leaf), np.asarray(ref_leaf)
            assert out_array.dtype == ref_array.dtype
            assert out_array.tobytes() == ref_array.tobytes()

    # ---------- time budget and external stop of the chunked loop (#354) ----------

    @pytest.mark.light
    @requires_h_fixture
    @pytest.mark.parametrize(
        ("stop", "wf_dump_freq"),
        [("max_time", 1), ("max_time", 5), ("stop_flag", 1), ("stop_flag", 5), ("stop_during_dump", 1)],
    )
    def test_budget_and_stop_request_end_the_whole_chunked_run_like_direct_run_optimize(
        self, tmp_path, monkeypatch, stop, wf_dump_freq
    ):
        """``max_time`` and ``stop = true`` end the twist-averaged run where a direct run stops.

        Independent reference: a direct ``MCMC.run_optimize`` call with the same
        H-atom J1/J2 input, seed, walkers and five requested SR steps, under the
        same stop condition. ``max_time``: the optimizer and driver clocks are
        replaced by one virtual clock that advances 1000 s per optimization
        step (per ``MCMC.sample`` call), with a 2500 s budget, so the direct run
        stops after two steps: the third would end at 3000 s, after the budget,
        and is not started (comparing only the elapsed time would run it).
        The chunked driver must apply the same prediction at the last step of a
        chunk when another chunk follows. ``stop_flag``: the real control file
        ``external_control_opt.toml`` is rewritten to ``stop = true`` as soon as
        the optimizer creates it, so the direct run stops after one step.
        ``stop_during_dump`` writes a single request immediately after saving
        the first real Hamiltonian dump, so the direct run stops after step two.
        The single-twist driver must stop at the same
        step, with the same optimized Hamiltonian (byte identity), and keep a
        non-converged checkpoint of that step. ``wf_dump_freq = 1`` (CLI
        default) splits the run into one-step chunks; ``wf_dump_freq = 5``
        requests a single chunk. No chunk may restart the budget clock or
        recreate the control file with ``stop = false``.
        """
        import jqmc.jqmc_optimizer as optimizer_module
        from jqmc._twist_checkpoint import load_twist_checkpoint_summary
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jqmc_mcmc import MCMC
        from tests._external_stop_request import request_external_stop

        requested_steps = 5
        sample_calls = [0]
        if stop == "max_time":
            real_sample = MCMC.sample

            def counted_sample(self, *args, **kwargs):
                sample_calls[0] += 1
                return real_sample(self, *args, **kwargs)

            def virtual_clock():
                return 1000.0 * sample_calls[0]

            clock = SimpleNamespace(perf_counter=virtual_clock)
            monkeypatch.setattr(MCMC, "sample", counted_sample)
            monkeypatch.setattr(optimizer_module, "time", clock)
            # raising=False: the patch must not depend on the driver importing time.
            monkeypatch.setattr(twist_average_module, "time", clock, raising=False)
            max_time = 2500.0
        else:
            max_time = 86400.0
        optimize_kwargs = dict(
            num_mcmc_steps=20,
            num_opt_steps=requested_steps,
            wf_dump_freq=wf_dump_freq,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            max_time=max_time,
            opt_J1_param=True,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-2, "epsilon": 1e-3},
        )

        def run_once(route):
            run_dir = tmp_path / route
            run_dir.mkdir()
            sample_calls[0] = 0
            with monkeypatch.context() as patch:
                patch.chdir(run_dir)
                requests = request_external_stop(
                    patch, optimizer_module, "external_control_opt.toml", requests=1 if stop == "stop_flag" else 0
                )
                if stop == "stop_during_dump":
                    save = Hamiltonian_data.save_to_hdf5

                    def save_then_request(hamiltonian, filepath="jqmc.h5"):
                        save(hamiltonian, filepath)
                        if Path(filepath).name == "hamiltonian_data_opt_step_1.h5" and requests["requested"] == 0:
                            assert Path(filepath).is_file()
                            control = Path("external_control_opt.toml")
                            assert control.is_file()
                            control.write_text("[external_control]\nstop = true\n", encoding="utf-8")
                            requests["requested"] += 1

                    patch.setattr(Hamiltonian_data, "save_to_hdf5", save_then_request)
                if route == "direct":
                    mcmc = MCMC(
                        hamiltonian_data=_build_h_hamiltonian_with_j1_j2(),
                        mcmc_seed=12345,
                        num_walkers=2,
                        comput_log_WF_param_deriv=True,
                        comput_e_L_param_deriv=True,
                    )
                    reason = mcmc.run_optimize(**optimize_kwargs)
                    return mcmc.i_opt, reason, mcmc.hamiltonian_data, requests
                real_run_optimize = MCMC.run_optimize
                chunks = []

                def spy_run_optimize(mcmc, **kwargs):
                    before = mcmc.i_opt
                    reason = real_run_optimize(mcmc, **kwargs)
                    chunks.append((kwargs["num_opt_steps"], mcmc.i_opt - before, reason))
                    return reason

                patch.setattr(MCMC, "run_optimize", spy_run_optimize)
                out = run_twist_average_optimize(
                    hamiltonian_list=[_build_h_hamiltonian_with_j1_j2()],
                    weights=[1.0],
                    mcmc_seed=12345,
                    num_walkers=2,
                    optimize_kwargs=optimize_kwargs,
                    checkpoint_dir=str(run_dir / "ckpt"),
                )
            return chunks, out, requests

        ref_steps, ref_reason, ref_ham, ref_requests = run_once("direct")
        expected_steps = 1 if stop == "stop_flag" else 2
        expected_reason = "max_time" if stop == "max_time" else "stop_flag"
        expected_requests = 0 if stop == "max_time" else 1
        assert ref_steps == expected_steps, "the reference stop condition must bind before the requested steps"
        assert ref_requests == {"created": 1, "requested": expected_requests}

        chunks, out_ham, requests = run_once("twist")
        assert sum(done for _, done, _ in chunks) == ref_steps, f"chunks (requested, done, reason): {chunks}"
        assert ref_reason == expected_reason, "the direct optimizer must report why it stopped"
        assert chunks[-1][2] == expected_reason, f"the last chunk must be the one that stopped: {chunks}"
        assert all(reason is None for _, _, reason in chunks[:-1]), chunks
        assert requests == {"created": 1, "requested": expected_requests}
        assert not (tmp_path / "twist" / "external_control_opt.toml").exists()
        summary = load_twist_checkpoint_summary(str(tmp_path / "twist" / "ckpt"), 0)
        assert summary["converged"] is False, "an early stop keeps a non-converged, resumable checkpoint"
        assert summary["target_steps"] == requested_steps
        ref_leaves, ref_tree = tree_flatten(ref_ham)
        out_leaves, out_tree = tree_flatten(out_ham)
        assert out_tree == ref_tree
        for out_leaf, ref_leaf in zip(out_leaves, ref_leaves, strict=True):
            np.testing.assert_array_equal(out_leaf, ref_leaf)

    # ---------- derivative-cache contract ----------

    @pytest.mark.light
    @requires_h_fixture
    def test_driver_enables_deriv_caches_by_default(self):
        """SR/LM cannot run without ``comput_log_WF_param_deriv`` and
        ``comput_e_L_param_deriv``. The driver enables them by default so
        the caller does not have to remember the contract; pinned here by
        checking that a call with NO ``mcmc_kwargs`` runs end-to-end."""
        ham = _build_h_hamiltonian_with_j2()
        out_ham = run_twist_average_optimize(
            hamiltonian_list=[ham],
            weights=[1.0],
            mcmc_seed=12345,
            num_walkers=2,
            optimize_kwargs=dict(
                num_mcmc_steps=10,
                num_opt_steps=1,
                num_mcmc_warmup_steps=2,
                num_mcmc_bin_blocks=2,
                opt_J1_param=False,
                opt_J2_param=True,
                opt_J3_param=False,
                opt_geminal=False,
                optimizer_kwargs={"method": "sr", "delta": 1e-2, "epsilon": 1e-3},
            ),
        )
        assert out_ham is not None

    @pytest.mark.light
    @requires_h_fixture
    def test_driver_refuses_explicit_disable_of_deriv_caches(self):
        """If the caller explicitly disables a derivative cache via
        ``mcmc_kwargs`` the driver raises rather than silently producing a
        broken SR step (the alternative -- letting ``get_aH`` raise mid-
        optimization -- would waste compute and obscure the cause)."""
        ham = _build_h_hamiltonian_with_j2()
        with pytest.raises(ValueError, match="comput_log_WF_param_deriv"):
            run_twist_average_optimize(
                hamiltonian_list=[ham],
                weights=[1.0],
                mcmc_kwargs={"comput_log_WF_param_deriv": False},
                optimize_kwargs={"num_mcmc_steps": 5},
            )

    # ---------- input validation (mirrors the VMC / GFMC drivers) ----------

    @pytest.mark.light
    def test_driver_rejects_empty_hamiltonian_list(self):
        """Empty ``hamiltonian_list`` raises; pins the input validation
        contract shared with the VMC / GFMC twist drivers."""
        with pytest.raises(ValueError, match="empty"):
            run_twist_average_optimize(
                hamiltonian_list=[],
                weights=[],
                optimize_kwargs={"num_mcmc_steps": 5},
            )

    @pytest.mark.light
    @requires_h_fixture
    def test_driver_rejects_weights_length_mismatch(self):
        """``len(weights) != n_twist`` raises. Crucial here: a downstream
        BZ-mean energy evaluation that consumes the optimized states will
        use the same weights, so a silent mismatch would bias not just the
        energy but the parameter trajectory of every twist (Plan B routing
        keys on the count, not on the weight values, but the contract is
        pinned uniformly)."""
        ham = _build_h_hamiltonian_with_j2()
        with pytest.raises(ValueError, match="weights size"):
            run_twist_average_optimize(
                hamiltonian_list=[ham, ham],
                weights=[1.0],
                optimize_kwargs={"num_mcmc_steps": 5},
            )

    # ---------- multi-twist opt-step dump (complete combined) ----------

    @pytest.mark.light
    @requires_h_fixture
    def test_multi_twist_opt_step_dump_is_a_complete_combined_input(self, tmp_path, monkeypatch):
        """A twist-averaged optimisation must dump a COMPLETE combined Hamiltonian.

        The single-twist dump inside ``run_optimize`` would hold only the
        world-rank-0 twist, and a downstream MCMC / LRDMC job reading it would
        silently sample a single-k wavefunction.  Given ``checkpoint_dir`` and
        ``source_hamiltonian_path`` the driver instead assembles
        ``dump_dir/hamiltonian_data_opt_step_{i_opt}.h5`` from the per-twist
        checkpoints it has just written, and switches the inner single-twist
        dump off (``dump_wavefunction=False``) so the two can never collide.
        The dump is verified through the loaders a fresh run would use, so it
        is proven to be restartable input, not merely shaped like a combined input.
        """
        import os

        import h5py

        import jqmc._twist_checkpoint as twist_checkpoint_module
        from jqmc._twist_average import (
            is_twist_average_input,
            load_twist,
            load_twist_average_input,
            save_twist_average_input,
        )
        from jqmc._twist_checkpoint import save_shared_state, twist_checkpoint_path
        from jqmc.hamiltonians import _save_dataclass_to_hdf5

        n_twist = 2
        template = _build_h_hamiltonian_with_j2()
        geminal = template.wavefunction_data.geminal_data
        mo_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients, dtype=np.float64)
        mo_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients, dtype=np.float64)
        source = tmp_path / "hamiltonian_data.h5"
        save_twist_average_input(
            filepath=str(source),
            template_hamiltonian=template,
            k_points=[None] * n_twist,
            weights=[1.0] * n_twist,
            mo_coefficients_up_per_twist=[mo_up] * n_twist,
            mo_coefficients_dn_per_twist=[mo_dn] * n_twist,
        )

        # Per-twist "optimized" states: distinct MO scaling per twist so a
        # wrong slot (or a rank-0-twist-only dump) is impossible to miss.
        optimized = {}
        expected_up = {}
        for twist_index in range(n_twist):
            ham = load_twist(str(source), twist_index)
            gem = ham.wavefunction_data.geminal_data
            up = np.asarray(gem.orb_data_up_spin.mo_coefficients) * (1.0 + 0.1 * (twist_index + 1))
            dn = np.conj(up[: gem.num_electron_dn]) if gem.time_reversal else np.asarray(gem.orb_data_dn_spin.mo_coefficients)
            gem = gem.replace(
                orb_data_up_spin=gem.orb_data_up_spin.replace(mo_coefficients=up),
                orb_data_dn_spin=gem.orb_data_dn_spin.replace(mo_coefficients=dn),
            )
            optimized[twist_index] = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=gem))
            expected_up[twist_index] = np.asarray(up)

        # Stand-in for "every twist root wrote its own checkpoint": a single
        # test process plays all twists.
        def fake_save_twist_checkpoint(checkpoint_dir, *, twist_idx, mcmc_or_gfmc, converged, **kwargs):
            del twist_idx, mcmc_or_gfmc, kwargs
            os.makedirs(checkpoint_dir, exist_ok=True)
            for index, ham in optimized.items():
                with h5py.File(twist_checkpoint_path(checkpoint_dir, index), "w") as h5:
                    meta = h5.create_group("_meta")
                    meta.attrs["twist_index"] = index
                    meta.attrs["target_steps"] = 2
                    meta.attrs["converged"] = bool(converged)
                    _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), ham)
            save_shared_state(checkpoint_dir, optimized[0])

        monkeypatch.setattr(twist_checkpoint_module, "save_twist_checkpoint", fake_save_twist_checkpoint)

        calls = []

        class FakeMCMC:
            def __init__(self, *, hamiltonian_data, **kwargs):
                del kwargs
                self.hamiltonian_data = hamiltonian_data
                self.i_opt = 0

            def run_optimize(self, **kwargs):
                calls.append(dict(kwargs))
                self.i_opt += int(kwargs["num_opt_steps"])

        world = _WorldComm()
        monkeypatch.setattr("jqmc.jqmc_mcmc.MCMC", FakeMCMC)
        monkeypatch.setattr(
            twist_average_module,
            "split_world_for_twists",
            lambda world_comm, n_twist: SimpleNamespace(
                world_comm=world_comm,
                intra_comm=_RootIntraComm(),
                inter_comm=_RootInterComm(gathered=[EnergyEstimate(-1.0, 0.2, 2.0, 0.3)] * n_twist),
                twist_index=0,
                n_twist=n_twist,
            ),
        )

        checkpoint_dir = tmp_path / "restart.h5"
        dump_dir = tmp_path / "dumps"
        dump_dir.mkdir()
        run_twist_average_optimize(
            hamiltonian_list=[optimized[0], optimized[1]],
            weights=[1.0, 1.0],
            world_comm=world,
            optimize_kwargs={"num_opt_steps": 2, "wf_dump_freq": 1},
            checkpoint_dir=str(checkpoint_dir),
            source_hamiltonian_path=str(source),
            dump_dir=str(dump_dir),
        )

        # One dump per chunk, numbered exactly like the single-twist dump.
        assert sorted(os.listdir(dump_dir)) == [
            "hamiltonian_data_opt_step_1.h5",
            "hamiltonian_data_opt_step_2.h5",
        ]
        # The inner optimizer must not have written its rank-0-twist-only file.
        assert all(call["dump_wavefunction"] is False for call in calls)
        assert not os.path.exists(tmp_path / "hamiltonian_data_opt_step_1.h5")
        assert not os.path.exists(tmp_path / "hamiltonian_data_opt_step_2.h5")

        dump = str(dump_dir / "hamiltonian_data_opt_step_2.h5")
        assert is_twist_average_input(dump)
        meta = load_twist_average_input(dump)
        assert meta.n_twist == n_twist
        np.testing.assert_array_equal(meta.weights, load_twist_average_input(str(source)).weights)
        for twist_index in range(n_twist):
            reloaded = load_twist(dump, twist_index)
            np.testing.assert_array_equal(
                reloaded.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients,
                expected_up[twist_index],
            )

    @pytest.mark.light
    @requires_h_fixture
    def test_single_twist_keeps_the_historical_opt_step_dump(self, tmp_path):
        """``N_twist == 1`` behaviour is untouched: the optimizer still writes
        ``hamiltonian_data_opt_step_N.h5`` into the working directory itself
        (``dump_wavefunction`` stays ``True``), so no existing single-k
        workflow loses its artifact."""
        import os

        optimize_kwargs = dict(
            num_mcmc_steps=10,
            num_opt_steps=1,
            wf_dump_freq=1,
            num_mcmc_warmup_steps=2,
            num_mcmc_bin_blocks=2,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-2, "epsilon": 1e-3},
        )
        run_twist_average_optimize(
            hamiltonian_list=[_build_h_hamiltonian_with_j2()],
            weights=[1.0],
            mcmc_seed=12345,
            num_walkers=2,
            optimize_kwargs=optimize_kwargs,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        # The autouse ``_isolate_cwd`` fixture puts CWD at the test's tmp dir.
        assert os.path.isfile(os.path.join(os.getcwd(), "hamiltonian_data_opt_step_1.h5"))

    @pytest.mark.light
    def test_multi_twist_without_a_source_refuses_to_dump_a_single_twist_file(self, monkeypatch, caplog):
        """Without ``source_hamiltonian_path`` a multi-twist run dumps NOTHING
        and says so: emitting the rank-0 twist under the canonical opt-step
        name is the defect being fixed, so it must not happen even in this
        degraded configuration."""
        import logging

        calls = []

        class FakeMCMC:
            def __init__(self, *, hamiltonian_data, **kwargs):
                del kwargs
                self.hamiltonian_data = hamiltonian_data
                self.i_opt = 0

            def run_optimize(self, **kwargs):
                calls.append(dict(kwargs))
                self.i_opt += int(kwargs["num_opt_steps"])

        world = _WorldComm()
        monkeypatch.setattr("jqmc.jqmc_mcmc.MCMC", FakeMCMC)
        monkeypatch.setattr(
            twist_average_module,
            "split_world_for_twists",
            lambda world_comm, n_twist: SimpleNamespace(
                world_comm=world_comm,
                intra_comm=_RootIntraComm(),
                inter_comm=_RootInterComm(gathered=[]),
                twist_index=0,
                n_twist=n_twist,
            ),
        )

        with caplog.at_level(logging.WARNING, logger="jqmc"):
            run_twist_average_optimize(
                hamiltonian_list=[object(), object()],
                weights=[1.0, 1.0],
                world_comm=world,
                optimize_kwargs={"num_opt_steps": 1},
            )
        assert calls[-1]["dump_wavefunction"] is False
        assert any("source_hamiltonian_path" in record.message for record in caplog.records)
