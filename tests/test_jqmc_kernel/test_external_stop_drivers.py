"""Real external TOML requests stop non-twist drivers and leave continuable checkpoints."""

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

import numpy as np
import pytest

from jqmc import jqmc_gfmc, jqmc_mcmc, jqmc_optimizer, output_parser
from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import get_tolerance_min
from jqmc._setting import GFMC_MIN_STEPS
from tests._external_stop_request import request_external_stop
from tests.test_jqmc_kernel.test_cli_restart_values import _run_cli, _write_hamiltonian

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_DRIVERS = {
    "mcmc": (jqmc_mcmc, "external_control_mcmc.toml", "mcmc_counter", 8, 1),
    "lrdmc-bra": (jqmc_gfmc, "external_control_gfmc.toml", "mcmc_counter", GFMC_MIN_STEPS, 30),
    "lrdmc-tau": (jqmc_gfmc, "external_control_gfmc.toml", "mcmc_counter", GFMC_MIN_STEPS, 30),
    "vmc": (jqmc_optimizer, "external_control_opt.toml", "i_opt", 2, 1),
}


def _run(root, job_type, steps, monkeypatch, *, restart=False):
    key = "num_opt_steps" if job_type == "vmc" else "num_mcmc_steps"
    messages = _run_cli(root, job_type, restart, 1, monkeypatch, mcmc_seed=12345, overrides={job_type: {key: steps}})
    checkpoint = load_rank_checkpoint(str(root / "restart.h5"), rank=0)
    return checkpoint, output_parser.parse_jqmc_log("\n".join(messages), job_type)


def _assert_stopped(root, job_type, partial, parsed, expected_stop):
    _, filename, counter, _, _ = _DRIVERS[job_type]
    assert partial["driver_config"][counter] == expected_stop, "external stop must keep exactly the first completed boundary"
    assert parsed["stopped_by_stop_flag"] and not parsed["stopped_by_max_time"]
    assert not (root / filename).exists(), "normal termination must remove the owned control file"


def _assert_stop_continuation(tmp_path, monkeypatch, job_type, *, total=None, expected_stop=None):
    """Same seeded H2 chain before/after a real flag request; no sampler or reader replacement."""
    module, filename, counter, default_total, default_stop = _DRIVERS[job_type]
    total = default_total if total is None else total
    expected_stop = default_stop if expected_stop is None else expected_stop
    assert 0 < expected_stop < total
    _write_hamiltonian(tmp_path)
    expected, continuous_log = _run(tmp_path / "continuous", job_type, total, monkeypatch)
    assert not continuous_log["stopped_by_stop_flag"] and not continuous_log["stopped_by_max_time"]
    with monkeypatch.context() as patch:
        requests = request_external_stop(patch, module, filename)
        partial, stop_log = _run(tmp_path / "resumed", job_type, total, patch)
        _assert_stopped(tmp_path / "resumed", job_type, partial, stop_log, expected_stop)
        assert requests == {"created": 1, "requested": 1}
        actual, resumed_log = _run(tmp_path / "resumed", job_type, total - expected_stop, patch, restart=True)
        assert requests == {"created": 2, "requested": 1}
    assert not resumed_log["stopped_by_stop_flag"] and not resumed_log["stopped_by_max_time"]
    assert actual["driver_config"][counter] == expected["driver_config"][counter] == total
    for name in ("latest_r_up_carts", "latest_r_dn_carts"):
        np.testing.assert_array_equal(actual["walker_state"][name], expected["walker_state"][name])
    rng_names = ["jax_PRNG_key_list"]
    if job_type.startswith("lrdmc"):
        rng_names += [
            "branching_rng_keys",
            "branching_rng_position",
            "branching_rng_has_gauss",
            "branching_rng_cached_gaussian",
        ]
    for name in rng_names:
        np.testing.assert_array_equal(actual["rng_state"][name], expected["rng_state"][name])
    atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "det_grad_lap", "coulomb", "local_energy"), "strict")
    if job_type == "vmc":
        actual_j2, expected_j2 = (
            load_hamiltonian_from_checkpoint(
                str(tmp_path / run / "restart.h5")
            ).wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param
            for run in ("resumed", "continuous")
        )
        assert np.any(np.asarray(expected_j2) != 1.0), "the optimization must actually update J2"
        np.testing.assert_allclose(actual_j2, expected_j2, atol=atol, rtol=rtol)
    else:
        for name in ("e_L", "e_L2", "w_L"):
            observed, reference = actual["observables"][name], expected["observables"][name]
            assert observed.shape == reference.shape
            assert observed.shape[0] == total
            assert np.isfinite(observed).all() and np.isfinite(reference).all()
            np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("job_type", list(_DRIVERS))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_external_stop_keeps_real_checkpoint_continuation(tmp_path, monkeypatch, job_type):
    """H2 AE/J2, two walkers: stop at 1 MCMC/VMC or 30 GFMC steps, then finish the unchanged trajectory.

    The shared request helper rewrites the real TOML immediately after creation;
    parsing, broadcasts, sampling, optimization, CLI termination and checkpoint
    restart are production code. The complete run uses 8 MCMC measurements,
    40 GFMC branching steps or 2 J2-only SR updates. Exact state/RNG and existing
    strict same-mode observable/J2 tolerances compare the resumed and continuous
    results. This module exercises XLA; native chunking has a separate witness.
    """
    _assert_stop_continuation(tmp_path, monkeypatch, job_type)
