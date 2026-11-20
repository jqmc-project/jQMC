"""Tests for jqmc_workflow._input_generator -- TOML input file generation.

Layer 1 (pure unit): tmp_path for file writes; no subprocess, no SSH.

Test class layout:
    TestResolveWithDefaults     -- None-value resolution against cli_parameters
    TestGetDefaultParameters    -- per-job_type default fetch + validation
    TestGenerateInputToml       -- emit + atomic write + overrides + validation
    TestDumpWithComments        -- comment-inclusive serializer
    TestTomlValueFormatter      -- _toml_value primitive formatting
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

import os
from pathlib import Path

import pytest
import toml

from jqmc.jqmc_miscs import cli_parameters
from jqmc_workflow._input_generator import (
    _dump_with_comments,
    _toml_value,
    generate_input_toml,
    get_default_parameters,
    resolve_with_defaults,
)

pytestmark = [pytest.mark.with_chdir]


class TestResolveWithDefaults:
    @pytest.mark.light
    def test_explicit_value_kept(self):
        """Explicit value kept."""
        resolved = resolve_with_defaults("mcmc", {"Dt": 3.0})
        assert resolved == {"Dt": 3.0}

    @pytest.mark.light
    def test_none_replaced_by_default(self):
        """None replaced by default."""
        resolved = resolve_with_defaults("mcmc", {"Dt": None})
        # mcmc default Dt is 2.0
        assert resolved["Dt"] == cli_parameters["mcmc"]["Dt"]

    @pytest.mark.light
    def test_unknown_key_resolves_to_none(self):
        """A key not in cli_parameters[section] resolves to None (no default)."""
        resolved = resolve_with_defaults("mcmc", {"nonexistent_param": None})
        assert resolved == {"nonexistent_param": None}

    @pytest.mark.light
    def test_unknown_section_returns_user_values_only(self):
        """Unknown section -> defaults={}, so explicit values pass through unchanged."""
        resolved = resolve_with_defaults("nonexistent_section", {"x": 5, "y": None})
        assert resolved == {"x": 5, "y": None}

    @pytest.mark.light
    def test_mixed_explicit_and_none(self):
        """Mixed explicit and none."""
        resolved = resolve_with_defaults("mcmc", {"Dt": 5.0, "num_mcmc_bin_blocks": None})
        assert resolved["Dt"] == 5.0
        assert resolved["num_mcmc_bin_blocks"] == cli_parameters["mcmc"]["num_mcmc_bin_blocks"]


class TestGetDefaultParameters:
    @pytest.mark.light
    @pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
    def test_returns_control_and_job_type_sections(self, job_type):
        """Returns control and job type sections."""
        params = get_default_parameters(job_type)
        assert "control" in params
        assert job_type in params

    @pytest.mark.light
    def test_control_job_type_field_set(self):
        """Control job type field set."""
        params = get_default_parameters("mcmc")
        assert params["control"]["job_type"] == "mcmc"

    @pytest.mark.light
    def test_unknown_job_type_raises(self):
        """Unknown job type raises."""
        with pytest.raises(ValueError, match="Unknown job_type"):
            get_default_parameters("not_a_real_type")

    @pytest.mark.light
    def test_error_message_lists_valid_choices(self):
        """Error message lists valid choices."""
        with pytest.raises(ValueError) as excinfo:
            get_default_parameters("bogus")
        msg = str(excinfo.value)
        assert "mcmc" in msg
        # *_comments keys are filtered out of the error message.
        assert "control_comments" not in msg

    @pytest.mark.light
    def test_returned_dict_is_deep_copy(self):
        """Mutating the returned dict must not affect cli_parameters."""
        params = get_default_parameters("mcmc")
        params["mcmc"]["Dt"] = 999.0
        assert cli_parameters["mcmc"]["Dt"] != 999.0


class TestGenerateInputToml:
    @pytest.mark.light
    def test_emits_file_at_filename(self, tmp_path, monkeypatch):
        """Emits file at filename."""
        monkeypatch.chdir(tmp_path)
        path = generate_input_toml(
            "mcmc",
            overrides={"mcmc": {"num_mcmc_steps": 100}},
            filename="input.toml",
        )
        assert path == "input.toml"
        assert (tmp_path / "input.toml").exists()

    @pytest.mark.light
    def test_round_trip(self, tmp_path, monkeypatch):
        """Round trip."""
        monkeypatch.chdir(tmp_path)
        generate_input_toml(
            "mcmc",
            overrides={"mcmc": {"num_mcmc_steps": 100, "Dt": 1.5}},
            filename="rt.toml",
        )
        loaded = toml.load(tmp_path / "rt.toml")
        assert loaded["control"]["job_type"] == "mcmc"
        assert loaded["mcmc"]["num_mcmc_steps"] == 100
        assert loaded["mcmc"]["Dt"] == 1.5

    @pytest.mark.light
    def test_missing_required_field_raises(self, tmp_path, monkeypatch):
        """num_mcmc_steps has default=None -> must be provided in overrides."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Required parameter 'num_mcmc_steps'"):
            generate_input_toml("mcmc", overrides=None, filename="missing.toml")

    @pytest.mark.light
    def test_with_comments_includes_comment_lines(self, tmp_path, monkeypatch):
        """With comments includes comment lines."""
        monkeypatch.chdir(tmp_path)
        generate_input_toml(
            "mcmc",
            overrides={"mcmc": {"num_mcmc_steps": 50}},
            filename="commented.toml",
            with_comments=True,
        )
        text = (tmp_path / "commented.toml").read_text()
        assert "[mcmc]" in text
        assert "[control]" in text
        # At least one comment line should be present in the commented form.
        assert any(line.startswith("#") for line in text.splitlines())

    @pytest.mark.light
    def test_atomic_write_replaces_existing(self, tmp_path, monkeypatch):
        """Re-running with new overrides overwrites the old TOML cleanly."""
        monkeypatch.chdir(tmp_path)
        generate_input_toml("mcmc", overrides={"mcmc": {"num_mcmc_steps": 1}}, filename="a.toml")
        generate_input_toml("mcmc", overrides={"mcmc": {"num_mcmc_steps": 2}}, filename="a.toml")
        loaded = toml.load(tmp_path / "a.toml")
        assert loaded["mcmc"]["num_mcmc_steps"] == 2

    @pytest.mark.light
    def test_no_tmpfile_left_behind(self, tmp_path, monkeypatch):
        """The .tmp file used for atomic write must be cleaned up via os.replace."""
        monkeypatch.chdir(tmp_path)
        generate_input_toml("mcmc", overrides={"mcmc": {"num_mcmc_steps": 1}}, filename="z.toml")
        assert not (tmp_path / "z.toml.tmp").exists()

    @pytest.mark.light
    def test_overrides_can_add_new_section(self, tmp_path, monkeypatch):
        """Overrides can add new section."""
        monkeypatch.chdir(tmp_path)
        generate_input_toml(
            "mcmc",
            overrides={"mcmc": {"num_mcmc_steps": 1}, "custom_section": {"x": 42}},
            filename="extra.toml",
        )
        loaded = toml.load(tmp_path / "extra.toml")
        assert loaded["custom_section"]["x"] == 42

    @pytest.mark.light
    def test_overrides_with_scalar_value_replaces_section(self, tmp_path, monkeypatch):
        """Defensive branch: a scalar override at the section level overwrites the whole section."""
        monkeypatch.chdir(tmp_path)
        # toml.dump rejects scalar top-level sections, so we exercise the with_comments path
        # (which uses the custom serializer that handles scalar sections).
        generate_input_toml(
            "mcmc",
            overrides={"mcmc": {"num_mcmc_steps": 1}, "marker_section": "just_a_string"},
            filename="scalar.toml",
            with_comments=True,
        )
        text = (tmp_path / "scalar.toml").read_text()
        assert 'marker_section = "just_a_string"' in text


class TestDumpWithComments:
    @pytest.mark.light
    def test_emits_section_headers(self):
        """Emits section headers."""
        params = {"control": {"job_type": "mcmc"}, "mcmc": {"Dt": 2.0, "num_mcmc_steps": 10}}
        text = _dump_with_comments(params, "mcmc")
        assert "[control]" in text
        assert "[mcmc]" in text

    @pytest.mark.light
    def test_short_comment_kept_unchanged(self, monkeypatch):
        """Comments under 120 chars are emitted verbatim (no truncation)."""
        fake = {
            "mcmc_comments": {"Dt": "short comment"},
            "mcmc": {"Dt": 2.0},
            "control": {},
            "control_comments": {},
        }
        monkeypatch.setattr("jqmc_workflow._input_generator.cli_parameters", fake)
        text = _dump_with_comments({"control": {}, "mcmc": {"Dt": 2.0}}, "mcmc")
        assert "# short comment" in text

    @pytest.mark.light
    def test_truncates_long_comments_at_120_chars(self, monkeypatch):
        """Truncates long comments at 120 chars."""
        long_comment = "x" * 200
        fake = {
            "mcmc_comments": {"Dt": long_comment},
            "mcmc": {"Dt": 2.0},
            "control": {},
            "control_comments": {},
        }
        monkeypatch.setattr("jqmc_workflow._input_generator.cli_parameters", fake)
        text = _dump_with_comments({"control": {}, "mcmc": {"Dt": 2.0}}, "mcmc")
        # Find the comment line and check it is truncated.
        comment_lines = [line for line in text.splitlines() if line.startswith("# ")]
        assert any(line.endswith("...") for line in comment_lines)
        assert all(len(line) <= 122 for line in comment_lines)  # "# " + 120


class TestTomlValueFormatter:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, "true"),
            (False, "false"),
            (42, "42"),
            (3.14, "3.14"),
            ("hello", '"hello"'),
            ('with "quote"', '"with \\"quote\\""'),
            ([1, 2, 3], "[1, 2, 3]"),
            ([True, False], "[true, false]"),
            ([], "[]"),
            ({"a": 1, "b": 2}, "{a = 1, b = 2}"),
            ({}, "{}"),
        ],
    )
    @pytest.mark.light
    def test_primitive_formatting(self, value, expected):
        """Primitive formatting."""
        assert _toml_value(value) == expected

    @pytest.mark.light
    def test_nested_list_of_dicts(self):
        """Nested list of dicts."""
        out = _toml_value([{"x": 1}, {"y": 2}])
        assert out == "[{x = 1}, {y = 2}]"

    @pytest.mark.light
    @pytest.mark.light
    def test_fallback_to_repr_for_unsupported_type(self):
        """Fallback to repr for unsupported type."""

        class Custom:
            def __repr__(self):
                return "<custom>"

        assert _toml_value(Custom()) == "<custom>"
