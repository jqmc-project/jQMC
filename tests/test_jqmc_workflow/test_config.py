"""Tests for jqmc_workflow._config -- config-directory resolution.

Layer 1 (pure unit): manipulates CWD via pytest's `tmp_path` + `monkeypatch.chdir`
and resets the module's cached `_resolved_config_dir` between tests.

Test class layout:
    TestGetConfigDir       -- resolution precedence (local CWD override vs global)
    TestCacheBehavior      -- _resolved_config_dir caching + reset_config_dir
    TestModuleConstants    -- package-level paths (template_dir, test_dir, config_dir)
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
from pathlib import Path

import pytest

from jqmc_workflow import _config as cfg
from jqmc_workflow._config import (
    config_dir,
    get_config_dir,
    reset_config_dir,
    template_dir,
    test_dir,
)

pytestmark = [pytest.mark.with_chdir]


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear cached _resolved_config_dir before and after each test.

    The module caches the first resolved path globally; without resetting,
    earlier tests would leak state into later ones.
    """
    reset_config_dir()
    yield
    reset_config_dir()


class TestGetConfigDir:
    """Local CWD override vs global ~/.jqmc_setting precedence."""

    @pytest.mark.light
    def test_returns_global_when_no_local_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = get_config_dir()
        assert result == os.path.join(os.path.expanduser("~"), ".jqmc_setting")

    @pytest.mark.light
    def test_local_override_takes_precedence(self, tmp_path, monkeypatch):
        local = tmp_path / "jqmc_setting_local"
        local.mkdir()
        monkeypatch.chdir(tmp_path)
        result = get_config_dir()
        # Returned path is absolute and points at the local dir
        assert os.path.isabs(result)
        assert Path(result).resolve() == local.resolve()

    @pytest.mark.light
    def test_local_override_requires_directory_not_file(self, tmp_path, monkeypatch):
        """A file named jqmc_setting_local is ignored; falls back to global."""
        (tmp_path / "jqmc_setting_local").write_text("not a dir")
        monkeypatch.chdir(tmp_path)
        result = get_config_dir()
        assert result == os.path.join(os.path.expanduser("~"), ".jqmc_setting")

    @pytest.mark.light
    def test_local_override_is_resolved_to_absolute_path(self, tmp_path, monkeypatch):
        (tmp_path / "jqmc_setting_local").mkdir()
        monkeypatch.chdir(tmp_path)
        result = get_config_dir()
        assert os.path.isabs(result), "get_config_dir must return an absolute path"


class TestCacheBehavior:
    """Resolution is cached after first call; reset_config_dir clears the cache."""

    @pytest.mark.light
    def test_result_is_cached(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        first = get_config_dir()
        # Mutate CWD AFTER first call -- cached value should NOT change.
        new_cwd = tmp_path / "elsewhere"
        new_cwd.mkdir()
        (new_cwd / "jqmc_setting_local").mkdir()
        monkeypatch.chdir(new_cwd)
        second = get_config_dir()
        assert first == second

    @pytest.mark.light
    def test_reset_clears_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        first = get_config_dir()  # global (no local override)
        # Add a local override and reset -- next call should pick it up.
        (tmp_path / "jqmc_setting_local").mkdir()
        reset_config_dir()
        second = get_config_dir()
        assert first != second
        assert Path(second).resolve() == (tmp_path / "jqmc_setting_local").resolve()

    @pytest.mark.light
    def test_cache_starts_empty(self):
        """The autouse fixture must leave the cache cleared at test entry."""
        assert cfg._resolved_config_dir is None


class TestModuleConstants:
    """Package-level path constants are usable absolute paths."""

    @pytest.mark.light
    def test_template_dir_is_absolute(self):
        assert os.path.isabs(template_dir)
        assert template_dir.endswith("template")

    @pytest.mark.light
    def test_test_dir_is_absolute(self):
        assert os.path.isabs(test_dir)
        assert test_dir.endswith("tests")

    @pytest.mark.light
    def test_config_dir_alias_matches_global(self):
        """config_dir is the back-compat alias for the global path."""
        assert config_dir == os.path.join(os.path.expanduser("~"), ".jqmc_setting")
