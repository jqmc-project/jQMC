"""Runtime warning for the jaxlib 0.9.1/0.9.2 persistent-cache load crash.

jaxlib 0.9.1 and 0.9.2 load an executable from JAX's persistent compilation
cache on a 512 KiB thread-pool stack and link its object files recursively; a
large executable overflows that stack and the process dies with SIGSEGV and no
Python traceback. jqmc keeps its supported jax range and
instead logs one warning when an affected jaxlib runs with the cache enabled.

The version and the jax config are mocked: no real crash is needed. The
affected-release table is ``jqmc._setting.JAXLIB_PERSISTENT_CACHE_LOAD_OVERFLOW``,
shared with the test-harness gate in ``tests/conftest.py``.
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

import logging
import os
import subprocess
import sys

import pytest

from jqmc import _setting

pytestmark = [pytest.mark.with_chdir]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENABLED_DIR = "/nonexistent/jqmc-w54-cache"


class _FakeJaxConfig:
    """Stand-in for ``jax.config`` carrying only the two persistent-cache flags."""

    def __init__(self, cache_dir, enabled=True):
        self.jax_compilation_cache_dir = cache_dir
        self.jax_enable_compilation_cache = enabled


def _warn_once(caplog, monkeypatch, version, config, environ):
    """Call the check from a never-warned state; return (its result, matching warning records)."""
    monkeypatch.setattr(_setting, "_persistent_cache_warning_emitted", False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jqmc._setting"):
        returned = _setting.warn_if_persistent_cache_load_can_overflow(jaxlib_version=version, config=config, environ=environ)
    records = [r for r in caplog.records if r.name == "jqmc._setting" and "persistent compilation cache" in r.getMessage()]
    return returned, records


@pytest.mark.light
@pytest.mark.parametrize("version", ["0.9.1", "0.9.2", "0.9.2.dev20260901"])
def test_warning_fires_for_affected_jaxlib_with_cache_enabled(caplog, monkeypatch, version):
    """Affected jaxlib + cache dir in jax.config -> one WARNING naming bug, versions and remedies."""
    returned, records = _warn_once(caplog, monkeypatch, version, _FakeJaxConfig(_ENABLED_DIR), {})
    assert returned is True
    assert [r.levelname for r in records] == ["WARNING"]
    message = records[0].getMessage()
    assert f"jaxlib {version} with the JAX persistent compilation cache enabled" in message
    assert "jaxlib >= 0.9.1 and < 0.10.0" in message
    assert "SIGSEGV, no Python traceback" in message
    assert "cache-load thread overflows its stack" in message
    assert "Upgrade to jaxlib >= 0.10.0" in message
    assert "JAX_ENABLE_COMPILATION_CACHE=false" in message


@pytest.mark.light
def test_warning_fires_when_cache_dir_comes_from_environment(caplog, monkeypatch):
    """JAX_COMPILATION_CACHE_DIR in the environment enables the cache just like jax.config."""
    environ = {"JAX_COMPILATION_CACHE_DIR": _ENABLED_DIR, "JAX_ENABLE_COMPILATION_CACHE": "true"}
    returned, records = _warn_once(caplog, monkeypatch, "0.9.2", _FakeJaxConfig(None), environ)
    assert returned is True
    assert len(records) == 1


@pytest.mark.light
def test_warning_is_emitted_once_per_process(caplog, monkeypatch):
    """A second check after a warning stays silent."""
    first, records = _warn_once(caplog, monkeypatch, "0.9.2", _FakeJaxConfig(_ENABLED_DIR), {})
    assert first is True
    assert len(records) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jqmc._setting"):
        second = _setting.warn_if_persistent_cache_load_can_overflow(
            jaxlib_version="0.9.2", config=_FakeJaxConfig(_ENABLED_DIR), environ={}
        )
    assert second is False
    assert caplog.records == []


_SILENT_CASES = [
    ("no-cache-dir", "0.9.2", (None, True), {}),
    ("empty-cache-dir", "0.9.2", ("", True), {"JAX_COMPILATION_CACHE_DIR": ""}),
    ("config-switch-off", "0.9.2", (_ENABLED_DIR, False), {}),
    ("env-switch-false", "0.9.1", (_ENABLED_DIR, True), {"JAX_ENABLE_COMPILATION_CACHE": "false"}),
    ("env-switch-zero", "0.9.2", (_ENABLED_DIR, True), {"JAX_ENABLE_COMPILATION_CACHE": "0"}),
    ("env-switch-off-padded", "0.9.2", (_ENABLED_DIR, True), {"JAX_ENABLE_COMPILATION_CACHE": " Off "}),
    ("jaxlib-0.7.2", "0.7.2", (_ENABLED_DIR, True), {}),
    ("jaxlib-0.8.0", "0.8.0", (_ENABLED_DIR, True), {}),
    ("jaxlib-0.9.0", "0.9.0", (_ENABLED_DIR, True), {}),
    ("jaxlib-0.10.0", "0.10.0", (_ENABLED_DIR, True), {}),
    ("jaxlib-0.10.1", "0.10.1", (_ENABLED_DIR, True), {}),
    ("jaxlib-0.10.2", "0.10.2", (_ENABLED_DIR, True), {}),
    ("jaxlib-unparsable", "garbage", (_ENABLED_DIR, True), {}),
]


@pytest.mark.light
@pytest.mark.parametrize("label,version,config_args,environ", _SILENT_CASES, ids=[c[0] for c in _SILENT_CASES])
def test_warning_silent_when_cache_disabled_or_jaxlib_unaffected(caplog, monkeypatch, label, version, config_args, environ):
    """No warning without a cache directory, with the cache switched off, or on an unaffected jaxlib."""
    returned, records = _warn_once(caplog, monkeypatch, version, _FakeJaxConfig(*config_args), environ)
    assert returned is False, label
    assert records == [], label


@pytest.mark.light
def test_warning_never_raises(caplog, monkeypatch):
    """A broken config object or a failing version lookup is swallowed (returns False, logs nothing)."""

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("config unavailable")

    returned, records = _warn_once(caplog, monkeypatch, "0.9.2", _Exploding(), {})
    assert returned is False
    assert records == []

    def _boom():
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(_setting, "installed_jaxlib_version", _boom)
    returned, records = _warn_once(caplog, monkeypatch, None, _FakeJaxConfig(_ENABLED_DIR), {})
    assert returned is False
    assert records == []


_IMPORT_PROBE = (
    "import logging, importlib.metadata as md\n"
    "real = md.version\n"
    "md.version = lambda name: '0.9.2' if name == 'jaxlib' else real(name)\n"
    "logging.basicConfig(level=logging.WARNING, format='%(name)s|%(message)s')\n"
    "import jqmc._precision\n"
    "import jax\n"
    "print('CACHE', jax.config.jax_enable_compilation_cache, bool(jax.config.jax_compilation_cache_dir))\n"
)


def _import_precision(cache_dir):
    """Import ``jqmc._precision`` in a fresh interpreter with jaxlib mocked as 0.9.2."""
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE")}
    env["PYTHONPATH"] = os.pathsep.join([_ROOT, os.environ.get("PYTHONPATH", "")])
    env["JAX_PLATFORMS"] = "cpu"
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    return subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE], capture_output=True, text=True, cwd=_ROOT, env=env, check=False
    )


@pytest.mark.light
def test_import_of_jqmc_precision_runs_the_check_and_changes_nothing():
    """``jqmc._precision`` (imported by every jax-using jqmc module) runs the check once at import.

    With the cache enabled the warning is logged exactly once and the cache stays
    enabled (the check only reports); without a cache directory nothing is logged.
    """
    warning = "jqmc._setting|jaxlib 0.9.2 with the JAX persistent compilation cache enabled"
    enabled = _import_precision(_ENABLED_DIR)
    assert enabled.returncode == 0, enabled.stderr
    assert enabled.stderr.count(warning) == 1, enabled.stderr
    assert "CACHE True True" in enabled.stdout, enabled.stdout
    disabled = _import_precision(None)
    assert disabled.returncode == 0, disabled.stderr
    assert warning not in disabled.stderr, disabled.stderr
    assert "CACHE True False" in disabled.stdout, disabled.stdout
