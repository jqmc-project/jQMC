"""Tests for jqmc_workflow._error_estimator -- step estimation + parsing helpers.

Layer 1 (pure unit): tmp_path for file-touching tests; no subprocess, no SSH.

Test class layout:
    TestEstimateRequiredSteps       -- CLT step-count math for pilot -> production
    TestEstimateAdditionalSteps     -- additional-steps math for resumed runs
    TestReadAccumulatedMeasurementSteps -- error branches (happy path is integration)
    TestSuffixedName                -- filename helper
    TestFormatDuration              -- duration formatter
    TestParseNetTime                -- jQMC output net-time regex
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

import math

import pytest

from jqmc_workflow._error_estimator import (
    _format_duration,
    estimate_additional_steps,
    estimate_required_steps,
    parse_net_time,
    read_accumulated_measurement_steps,
    suffixed_name,
)

pytestmark = [pytest.mark.with_chdir]


class TestEstimateRequiredSteps:
    """N_required = max(int(pilot_steps * (sigma_p/sigma_t)^2 * walker_ratio), min_steps)."""

    @pytest.mark.light
    def test_halving_error_requires_4x_steps(self):
        # sigma propto 1/sqrt(N) => to halve sigma, multiply N by 4.
        n = estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=0.005)
        assert n == 400

    @pytest.mark.light
    def test_same_error_returns_pilot_steps(self):
        n = estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=0.01)
        assert n == 100

    @pytest.mark.light
    def test_walker_ratio_lt_1_reduces_required_steps(self):
        """walker_ratio = pilot/prod walkers. More prod walkers -> fewer prod steps."""
        n_same = estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=0.005, walker_ratio=1.0)
        n_more = estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=0.005, walker_ratio=0.5)
        assert n_more == n_same // 2

    @pytest.mark.light
    def test_min_steps_floor_applied(self):
        # tiny ratio_sq would give n < 50; floor lifts it.
        n = estimate_required_steps(pilot_steps=10, pilot_error=0.001, target_error=0.01, min_steps=50)
        assert n == 50

    @pytest.mark.light
    def test_target_error_zero_raises(self):
        with pytest.raises(ValueError, match="target_error must be positive"):
            estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=0.0)

    @pytest.mark.light
    def test_target_error_negative_raises(self):
        with pytest.raises(ValueError, match="target_error must be positive"):
            estimate_required_steps(pilot_steps=100, pilot_error=0.01, target_error=-1e-3)

    @pytest.mark.light
    def test_zero_pilot_error_falls_back_to_pilot_steps(self):
        """When pilot stats are unusable, return pilot_steps (with min_steps floor)."""
        n = estimate_required_steps(pilot_steps=100, pilot_error=0.0, target_error=0.005)
        assert n == 100

    @pytest.mark.light
    def test_zero_pilot_error_with_min_steps(self):
        n = estimate_required_steps(pilot_steps=10, pilot_error=0.0, target_error=0.005, min_steps=500)
        assert n == 500

    @pytest.mark.light
    def test_negative_pilot_error_falls_back(self):
        n = estimate_required_steps(pilot_steps=100, pilot_error=-1e-5, target_error=0.005)
        assert n == 100


class TestEstimateAdditionalSteps:
    """additional = max(ceil(accumulated * (current/target)^2) - accumulated, min_additional)."""

    @pytest.mark.light
    def test_zero_when_already_below_target(self):
        n = estimate_additional_steps(accumulated_steps=1000, current_error=0.001, target_error=0.005)
        assert n == 0

    @pytest.mark.light
    def test_zero_when_exactly_at_target(self):
        n = estimate_additional_steps(accumulated_steps=1000, current_error=0.005, target_error=0.005)
        assert n == 0

    @pytest.mark.light
    def test_quadrupling_when_halving_error(self):
        # accumulated=100, current=0.01, target=0.005 => N_total=400, additional=300
        n = estimate_additional_steps(accumulated_steps=100, current_error=0.01, target_error=0.005, min_additional=0)
        assert n == 300

    @pytest.mark.light
    def test_min_additional_floor(self):
        # Tiny gap: needs only ~10 more, but floor lifts to 100 (default).
        n = estimate_additional_steps(accumulated_steps=1000, current_error=0.005001, target_error=0.005)
        assert n == 100  # default min_additional

    @pytest.mark.light
    def test_min_additional_override(self):
        n = estimate_additional_steps(
            accumulated_steps=100,
            current_error=0.01,
            target_error=0.005,
            min_additional=10000,
        )
        assert n == 10000  # floor wins over 300

    @pytest.mark.light
    def test_target_error_zero_raises(self):
        with pytest.raises(ValueError, match="target_error must be positive"):
            estimate_additional_steps(accumulated_steps=100, current_error=0.01, target_error=0.0)

    @pytest.mark.light
    def test_target_error_negative_raises(self):
        with pytest.raises(ValueError, match="target_error must be positive"):
            estimate_additional_steps(accumulated_steps=100, current_error=0.01, target_error=-1e-3)


class TestReadAccumulatedMeasurementSteps:
    """Layer 1 covers the error branches; happy path needs a real HDF5 checkpoint (integration)."""

    @pytest.mark.light
    def test_nonexistent_file_returns_none(self, tmp_path):
        result = read_accumulated_measurement_steps(
            restart_chk_path=str(tmp_path / "does_not_exist.h5"),
            warmup=10,
        )
        assert result is None

    @pytest.mark.light
    def test_unreadable_file_returns_none(self, tmp_path):
        """A path that exists but cannot be loaded as a checkpoint."""
        bad = tmp_path / "garbage.h5"
        bad.write_bytes(b"not an HDF5 file")
        result = read_accumulated_measurement_steps(restart_chk_path=str(bad), warmup=10)
        assert result is None

    @pytest.mark.light
    def test_subtracts_warmup_and_collect(self, tmp_path, monkeypatch):
        """Stub jqmc._checkpoint.load_driver_config_from_checkpoint to inject mcmc_counter."""
        import jqmc._checkpoint as ckpt

        monkeypatch.setattr(ckpt, "load_driver_config_from_checkpoint", lambda *a, **kw: {"mcmc_counter": 250})
        result = read_accumulated_measurement_steps(
            restart_chk_path=str(tmp_path / "fake.h5"),
            warmup=50,
            collect_steps=100,
        )
        # 250 - 100 - 50 = 100
        # First touch the file so isfile -> True
        # Actually read_accumulated does NOT check isfile -- it relies on jqmc loader.
        # The stub bypasses the real loader, so file existence is moot.
        assert result == 100

    @pytest.mark.light
    def test_clamps_negative_to_zero(self, tmp_path, monkeypatch):
        import jqmc._checkpoint as ckpt

        monkeypatch.setattr(ckpt, "load_driver_config_from_checkpoint", lambda *a, **kw: {"mcmc_counter": 10})
        result = read_accumulated_measurement_steps(
            restart_chk_path=str(tmp_path / "fake.h5"),
            warmup=100,
            collect_steps=0,
        )
        assert result == 0  # max(10 - 0 - 100, 0)

    @pytest.mark.light
    def test_missing_mcmc_counter_returns_none(self, tmp_path, monkeypatch):
        import jqmc._checkpoint as ckpt

        monkeypatch.setattr(ckpt, "load_driver_config_from_checkpoint", lambda *a, **kw: {"other_key": 42})
        result = read_accumulated_measurement_steps(
            restart_chk_path=str(tmp_path / "fake.h5"),
            warmup=0,
        )
        assert result is None


class TestSuffixedName:
    @pytest.mark.parametrize(
        "filename,index,expected",
        [
            ("input.toml", 0, "input_0.toml"),
            ("out.o", 2, "out_2.o"),
            ("a.b.c", 7, "a.b_7.c"),  # only the last extension is split
            ("noext", 1, "noext_1"),
        ],
    )
    @pytest.mark.light
    def test_suffix_insertion(self, filename, index, expected):
        assert suffixed_name(filename, index) == expected


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0s"),
            (5, "5s"),
            (59, "59s"),
            (60, "1m 0s"),
            (90, "1m 30s"),
            (3599, "59m 59s"),
            (3600, "1h 0m 0s"),
            (3661, "1h 1m 1s"),
            (86399, "23h 59m 59s"),
            (86400, "1d 0h 0m"),
        ],
    )
    @pytest.mark.light
    def test_canonical_format(self, seconds, expected):
        assert _format_duration(seconds) == expected

    @pytest.mark.light
    def test_negative_seconds_clamped_to_zero(self):
        assert _format_duration(-100) == "0s"


class TestParseNetTime:
    @pytest.mark.light
    def test_returns_none_when_file_missing(self, tmp_path):
        result = parse_net_time(str(tmp_path / "no.o"))
        assert result is None

    @pytest.mark.light
    def test_parses_mcmc_net_time(self, tmp_path):
        out = tmp_path / "mcmc.o"
        out.write_text("noise\nNet total time for MCMC =   6.22 sec.\nmore noise\n")
        assert parse_net_time(str(out)) == pytest.approx(6.22)

    @pytest.mark.light
    def test_parses_lrdmc_net_time(self, tmp_path):
        out = tmp_path / "lrdmc.o"
        out.write_text("Net GFMC time without pre-compilations = 2832.326 sec.\n")
        assert parse_net_time(str(out)) == pytest.approx(2832.326)

    @pytest.mark.light
    def test_lrdmc_takes_priority_over_mcmc_when_both_present(self, tmp_path):
        """The implementation tries LRDMC pattern first."""
        out = tmp_path / "both.o"
        out.write_text("Net GFMC time without pre-compilations = 100.0 sec.\nNet total time for MCMC = 200.0 sec.\n")
        assert parse_net_time(str(out)) == pytest.approx(100.0)

    @pytest.mark.light
    def test_sums_multiple_mcmc_matches(self, tmp_path):
        """VMC prints the MCMC line once per optimization step -- they should sum."""
        out = tmp_path / "vmc.o"
        out.write_text(
            "Net total time for MCMC = 1.5 sec.\nNet total time for MCMC = 2.5 sec.\nNet total time for MCMC = 3.0 sec.\n"
        )
        assert parse_net_time(str(out)) == pytest.approx(7.0)

    @pytest.mark.light
    def test_returns_none_when_no_pattern_matches(self, tmp_path):
        out = tmp_path / "empty.o"
        out.write_text("Some unrelated text.\n")
        assert parse_net_time(str(out)) is None

    @pytest.mark.light
    def test_handles_scientific_notation(self, tmp_path):
        out = tmp_path / "sci.o"
        out.write_text("Net total time for MCMC = 1.5e3 sec.\n")
        assert parse_net_time(str(out)) == pytest.approx(1500.0)

    @pytest.mark.light
    def test_handles_unreadable_file(self, tmp_path, monkeypatch):
        """File exists but open() raises OSError -- returns None."""
        bad = tmp_path / "noread.o"
        bad.write_text("Net total time for MCMC = 1.0 sec.\n")

        real_open = open

        def open_raising(path, *args, **kwargs):
            if str(path) == str(bad):
                raise OSError("simulated read failure")
            return real_open(path, *args, **kwargs)

        import builtins

        monkeypatch.setattr(builtins, "open", open_raising)
        assert parse_net_time(str(bad)) is None


# Sanity: math import is used in estimate_additional_steps (ceil).
assert math.ceil  # noqa: B015
