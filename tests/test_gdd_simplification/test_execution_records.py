"""Execution completion uses pytest outcomes, never reconstructed assertions."""

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

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gdd.engine.execution_records import SCHEMA, completion_errors, declared_case_required, execution_profile, read_records


pytest_plugins = ["pytester"]
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
NODE = "tests/test_science.py::test_energy[ecp-twist]"


def document(kind="collection", *, precision="full", size=1, rank=0, nodes=(NODE,)):
    return {
        "schema": SCHEMA,
        "kind": kind,
        "commit": "abc",
        "run_id": "123-1",
        "precision": precision,
        "execution": {"name": "mpi" if size > 1 else "serial", "world_size": size, "rank": rank},
        "exitstatus": 0,
        "selected": [{"nodeid": node, "excluded": None} for node in nodes],
        "reports": [
            {"nodeid": node, "when": phase, "outcome": "passed"} for node in nodes for phase in ("setup", "call", "teardown")
        ]
        if kind == "execution"
        else [],
    }


def errors(expected, records):
    return completion_errors(expected, records, commit="abc", run_id="123-1")


def test_missing_shard_is_detected_from_independent_collection():
    expected = document(nodes=(NODE, "tests/test_science.py::test_other"))
    result = errors([expected], [document("execution")])
    assert len(result) == 1 and "test_other" in result[0] and "missing" in result[0]


def test_missing_precision_is_not_replaced_by_duplicate_full_runs():
    result = errors([document(), document(precision="mixed")], [document("execution"), document("execution")])
    assert len(result) == 1 and "mixed" in result[0]


@pytest.mark.parametrize(
    "phase,outcome",
    [("call", "passed"), ("call", "skipped"), ("call", "failed"), ("call", "xpassed"), ("teardown", "failed")],
)
def test_nonpassing_phase_is_never_evidence(phase, outcome):
    """Only passed setup, call and teardown phases close a case; ``call-passed`` is the control."""
    record = document("execution")
    next(row for row in record["reports"] if row["when"] == phase)["outcome"] = outcome
    result = errors([document()], [record])
    if outcome == "passed":
        assert result == []
    else:
        assert "not passing evidence" in result[0]


def test_a_good_duplicate_does_not_hide_a_failed_run():
    bad = document("execution")
    bad["reports"][1]["outcome"] = "failed"
    assert errors([document()], [document("execution"), bad])


def test_phases_from_different_sessions_cannot_be_combined():
    first = document("execution")
    second = deepcopy(first)
    first["reports"] = first["reports"][:2]
    second["reports"] = second["reports"][2:]
    assert "no single execution" in errors([document()], [first, second])[0]


@pytest.mark.parametrize("collection", ["required", "extra"])
def test_declared_required_case_cannot_be_waived_by_a_static_skip(collection):
    """A static skip of a required case, or a skip in another collection, waives nothing."""
    if collection == "required":
        expected = document()
        expected["selected"][0].update(required=True, excluded="unexpected disabled test")
        assert errors([expected], [])
    else:
        skipped = document()
        skipped["selected"][0]["excluded"] = "not selected in another lane"
        assert errors([document(), skipped], [])


def test_authored_required_precision_cannot_silently_become_optional():
    execution = {"precision_modes": ["full"], "profiles": [{"name": "serial", "world_size": 1}]}
    node = {"required_precision_modes": ["full", "mixed"], "required_parameter_cases": [{"ecp": True}]}
    profile = {"name": "serial", "world_size": 1}
    assert declared_case_required(node, {"selector": {"ecp": True}}, execution, precision="mixed", profile=profile)
    assert not declared_case_required(node, {"selector": {"ecp": False}}, execution, precision="mixed", profile=profile)


def test_source_factory_selector_preserves_actual_callable_identity():
    """Two functions with identical generic serialization are not interchangeable."""
    from gdd.engine.execution_records import verified_source_selector

    def molecular():
        pass

    def periodic():
        pass

    periodic.__name__ = molecular.__name__
    periodic.__qualname__ = molecular.__qualname__
    selector = {"factory": {"source_expression": "molecular"}, "label": "molecule"}
    item = SimpleNamespace(
        nodeid="tests/test_factory.py::test_factory[molecule]",
        module=SimpleNamespace(molecular=molecular, periodic=periodic),
        callspec=SimpleNamespace(params={"factory": molecular, "label": "molecule"}),
    )
    assert verified_source_selector(item, selector) == selector
    item.callspec.params["factory"] = periodic
    with pytest.raises(AssertionError, match="callable parameter differs"):
        verified_source_selector(item, selector)


def test_empty_required_collection_is_not_green():
    assert errors([document(nodes=())], [document("execution", nodes=())])


@pytest.mark.parametrize("field,value", [("commit", "old"), ("run_id", "another"), ("schema", "old-schema")])
def test_cross_run_or_malformed_record_fails(field, value):
    record = document("execution")
    record[field] = value
    with pytest.raises(ValueError):
        errors([document()], [record])


@pytest.mark.parametrize("kind", ["collection", "execution"])
def test_session_failure_does_not_disappear_behind_passing_rows(kind):
    expected, record = document(), document("execution")
    (expected if kind == "collection" else record)["exitstatus"] = 1
    with pytest.raises(ValueError, match="did not finish successfully"):
        errors([expected], [record])


def test_collect_override_does_not_forge_actual_mpi_world():
    env = {"GDD_EXECUTION_WORLD_SIZE": "2"}
    assert execution_profile(env, collect=True) == {"name": "mpi", "world_size": 2, "rank": 0}
    assert execution_profile(env) == {"name": "serial", "world_size": 1, "rank": 0}


@pytest.mark.parametrize("workers", [0, 2])
def test_pytest_plugin_records_actual_collection_and_all_phases(pytester, monkeypatch, workers):
    """A tiny subprocess demonstrates independent collection then ordinary execution."""
    root = str(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("GDD_COMMIT_SHA", "abc")
    monkeypatch.setenv("GDD_RUN_ID", "123-1")
    monkeypatch.setenv("GDD_RECEIPT_AUTHORITY", "1")
    monkeypatch.delenv("GDD_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("JQMC_TEST_SHARD_COUNT", raising=False)
    monkeypatch.delenv("GDD_EXECUTION_WORLD_SIZE", raising=False)
    for prefix in ("OMPI_COMM_WORLD", "PMI", "MV2_COMM_WORLD"):
        monkeypatch.delenv(f"{prefix}_SIZE", raising=False)
        monkeypatch.delenv(f"{prefix}_RANK", raising=False)
    pytester.makeconftest(
        "from gdd.engine.execution_records import register\ndef pytest_configure(config):\n    register(config)\n"
    )
    pytester.makepyfile(
        "import pytest\n"
        "def test_value():\n"
        "    assert 1 + 1 == 2\n"
        "@pytest.mark.skip(reason='existing unsupported lane')\n"
        "def test_unsupported():\n"
        "    raise AssertionError('not supported')\n"
    )
    expected = pytester.path / "expected"
    records = pytester.path / "records"
    monkeypatch.setenv("GDD_RECEIPT_DIR", str(expected))
    pytester.runpytest_subprocess("--collect-only", "-q").assert_outcomes()
    monkeypatch.setenv("GDD_RECEIPT_DIR", str(records))
    worker_args = ["-n", str(workers)] if workers else []
    pytester.runpytest_subprocess("-q", *worker_args).assert_outcomes(passed=1, skipped=1)
    assert errors(read_records(expected, "collection"), read_records(records, "execution")) == []
    expected_doc = read_records(expected, "collection")[0]
    assert len(expected_doc["before_selection"]) == 2
    assert sum(len(row["reports"]) for row in read_records(records, "execution")) == 5


def test_plugin_retains_runtime_skips_and_xpasses(pytester, monkeypatch):
    """The real pytest report boundary never promotes runtime skip/XPASS to success."""
    root = str(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("GDD_RECEIPT_DIR", str(pytester.path / "records"))
    monkeypatch.setenv("GDD_RECEIPT_AUTHORITY", "1")
    monkeypatch.delenv("GDD_EXECUTION_MODE", raising=False)
    pytester.makeconftest(
        "from gdd.engine.execution_records import register\ndef pytest_configure(config):\n    register(config)\n"
    )
    pytester.makepyfile(
        "import pytest\n"
        "def test_skip():\n"
        "    pytest.skip('runtime unavailable')\n"
        "@pytest.mark.xfail(reason='known bug', strict=False)\n"
        "def test_xpass():\n"
        "    assert True\n"
    )
    pytester.runpytest_subprocess("-q").assert_outcomes(skipped=1, xpassed=1)
    reports = read_records(pytester.path / "records", "execution")[0]["reports"]
    assert {row["outcome"] for row in reports if row["when"] == "call"} == {"skipped", "xpassed"}
