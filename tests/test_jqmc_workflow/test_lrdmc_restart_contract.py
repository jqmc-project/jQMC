"""LRDMC-bra (GFMC_n) continuation contract: E_scf must not survive restart.

E_scf only seeds the GFMC energy shift on the FIRST submission; the driver
self-consistently updates it via population control and freezes the drifted
value into the checkpoint. jqmc's assert_restart_frozen_consistency
(jqmc/_restart_policy.py) hard-fails a continuation TOML that re-sends the
original (now-stale) static E_scf, because the key is classified "frozen on
restart" -- an explicit conflicting value is a loud error, not a silent
no-op. jqmc_workflow therefore omits E_scf from continuation TOMLs
entirely (this test pins it): an omitted key skips the check and the
checkpoint's self-consistently-evolved value carries forward untouched.

Example: a checkpoint whose E_scf drifted from -17.21329 (the static
VMC-energy seed) to -17.23561... makes every continuation TOML that re-sends
-17.21329 raise ValueError on submission.
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

from __future__ import annotations

import tomllib

import pytest

from jqmc_workflow.lrdmc_workflow import LRDMC_Workflow

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_GFMC_N_OK = {
    "num_gfmc_warmup_steps": 25,
    "num_gfmc_bin_blocks": 10,
    "num_gfmc_collect_steps": 5,
    "num_projection_per_measurement": 40,  # selects GFMC_n (lrdmc-bra)
    "E_scf": -17.21329,
}


def test_lrdmc_bra_first_submission_includes_e_scf(tmp_path):
    """The pilot / first submission (restart=False) seeds E_scf explicitly."""
    input_path = tmp_path / "lrdmc_bra_pilot.toml"
    wf = LRDMC_Workflow(**_GFMC_N_OK)

    wf._generate_input(num_steps=200, input_file=str(input_path), restart=False)

    with input_path.open("rb") as stream:
        section = tomllib.load(stream)["lrdmc-bra"]

    assert section["E_scf"] == pytest.approx(-17.21329)


def test_lrdmc_bra_continuation_omits_e_scf(tmp_path):
    """A restart=True continuation must NOT re-send E_scf (frozen-on-restart).

    Other lrdmc-bra keys (alat, non_local_move, ...) stay static across the
    whole workflow so they trivially match the checkpoint and never trigger
    the guard; E_scf is unique in DRIFTING inside the checkpoint, so it is
    the one key that must be omitted rather than merely matching.
    """
    input_path = tmp_path / "lrdmc_bra_continuation.toml"
    wf = LRDMC_Workflow(**_GFMC_N_OK)

    wf._generate_input(
        num_steps=5000,
        input_file=str(input_path),
        restart=True,
        restart_chk="restart.h5",
    )

    with input_path.open("rb") as stream:
        parsed = tomllib.load(stream)

    assert parsed["control"]["restart"] is True
    section = parsed["lrdmc-bra"]
    assert "E_scf" not in section
    # Sanity: other keys are still present -- only E_scf is special-cased.
    assert section["alat"] == pytest.approx(0.30)
    assert section["num_mcmc_per_measurement"] == 40


@pytest.mark.parametrize("alat", [0.30, 0.10])
def test_lrdmc_bra_input_without_projection_count_is_refused(tmp_path, alat):
    """A GFMC_n input with no given or calibrated projection count is refused, not filled in.

    ``[lrdmc-bra] num_mcmc_per_measurement`` has no default in jqmc, so the
    workflow must not write an input without it nor invent a value.
    """
    input_path = tmp_path / "lrdmc_bra.toml"
    kwargs = {k: v for k, v in _GFMC_N_OK.items() if k != "num_projection_per_measurement"}
    wf = LRDMC_Workflow(alat=alat, target_survived_walkers_ratio=0.97, **kwargs)
    assert wf.num_projection_per_measurement is None

    with pytest.raises(ValueError, match="num_mcmc_per_measurement"):
        wf._generate_input(num_steps=200, input_file=str(input_path), restart=False)
    assert not input_path.exists()


def test_lrdmc_bra_calibrated_projection_count_is_written(tmp_path):
    """The per-call value (calibration pilots) is written unchanged when the workflow has none of its own."""
    input_path = tmp_path / "lrdmc_bra.toml"
    kwargs = {k: v for k, v in _GFMC_N_OK.items() if k != "num_projection_per_measurement"}
    wf = LRDMC_Workflow(alat=0.10, target_survived_walkers_ratio=0.97, **kwargs)

    wf._generate_input(num_steps=200, input_file=str(input_path), restart=False, num_projection_per_measurement=64)

    with input_path.open("rb") as stream:
        section = tomllib.load(stream)["lrdmc-bra"]
    assert section["num_mcmc_per_measurement"] == 64


def test_lrdmc_bra_explicit_projection_count_is_kept_at_small_alat(tmp_path):
    """An explicit num_projection_per_measurement is written unchanged at any alat."""
    input_path = tmp_path / "lrdmc_bra.toml"
    wf = LRDMC_Workflow(alat=0.10, **_GFMC_N_OK)

    wf._generate_input(num_steps=200, input_file=str(input_path), restart=False)

    with input_path.open("rb") as stream:
        section = tomllib.load(stream)["lrdmc-bra"]
    assert section["num_mcmc_per_measurement"] == 40
