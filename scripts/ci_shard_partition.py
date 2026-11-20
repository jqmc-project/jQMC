"""Duration-weighted execution-shard partition of ``tests/conftest.py``.

``tests/conftest.py::_apply_test_shard`` splits one lane's collected items over
``JQMC_TEST_SHARD_COUNT`` jobs. Its default is a stride over the sorted node ids,
which is blind to cost: it can put several long rows into one shard while the
other shards stay short.

When ``JQMC_TEST_SHARD_WEIGHTS`` names a duration lane, the split is instead a
greedy longest-processing-time (LPT) partition over the measured per-test
minutes committed in ``scripts/ci_shard_durations.json`` (the per-test minutes
of a finished test run). The partition is

* TOTAL and DISJOINT by construction: every position gets exactly one shard
  index, so the union of the ``count`` shards is the collected selection and no
  item runs twice. Durations only decide WHERE an item runs, never WHETHER.
* A pure function of the SORTED node-id list and the committed table, so every
  xdist worker and both ``mpirun -np 2`` ranks compute the same assignment.
* Non-empty for every shard whenever ``count <= len(node_ids)``: weights are
  integers >= 1 and ties go to the lowest shard index, so the first ``count``
  items in LPT order land in ``count`` distinct empty shards.

An item missing from the table (a new test, a renamed parameter) still runs; it
is weighted with the median of its file's measured rows (lane median when the
file has none), so a stale table can only cost balance, never coverage.

Standard library only, so it imports without the scientific stack.
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

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

DURATIONS_PATH = Path(__file__).resolve().with_name("ci_shard_durations.json")
DURATIONS_SCHEMA_VERSION = 1
# Weights are compared as integer tenths of a minute so the partition never
# depends on floating-point summation order.
_TENTHS = 10


def load_durations(path: Path = DURATIONS_PATH) -> dict:
    """Load and shape-check the committed durations table (fail closed)."""
    table = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or table.get("schema_version") != DURATIONS_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported durations table schema")
    tests = table.get("tests")
    jobs = table.get("jobs")
    if not isinstance(tests, dict) or not isinstance(jobs, dict):
        raise ValueError(f"{path}: durations table needs 'tests' and 'jobs' mappings")
    for lane, rows in tests.items():
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f"{path}: tests[{lane!r}] must be a non-empty mapping")
        for node_id, minutes in rows.items():
            if "::" not in node_id or not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes < 0:
                raise ValueError(f"{path}: tests[{lane!r}][{node_id!r}] must map a node id to minutes >= 0")
    for key, minutes in jobs.items():
        if key.count("|") != 3 or not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes < 0:
            raise ValueError(f"{path}: jobs[{key!r}] must map 'lane|env|precision|file' to minutes >= 0")
    return table


def _tenths(minutes: float) -> int:
    return max(1, int(round(minutes * _TENTHS)))


def _median_tenths(values: list[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return max(1, (ordered[middle - 1] + ordered[middle]) // 2)


def item_weights(node_ids: Sequence[str], minutes: Mapping[str, float]) -> list[int]:
    """Integer weights (tenths of a minute, >= 1) for ``node_ids``.

    Unknown ids get the median of their file's measured rows, else the median of
    every measured row, else 1.
    """
    by_file: dict[str, list[int]] = {}
    for node_id, value in minutes.items():
        by_file.setdefault(node_id.split("::", 1)[0], []).append(_tenths(value))
    every = [weight for weights in by_file.values() for weight in weights]
    lane_default = _median_tenths(every) if every else 1
    weights = []
    for node_id in node_ids:
        if node_id in minutes:
            weights.append(_tenths(minutes[node_id]))
        else:
            known = by_file.get(node_id.split("::", 1)[0])
            weights.append(_median_tenths(known) if known else lane_default)
    return weights


def weighted_partition(node_ids: Sequence[str], count: int, minutes: Mapping[str, float]) -> list[int]:
    """Shard index (``0 <= k < count``) for every position of ``node_ids``.

    Greedy LPT: positions are visited by descending weight (ties: node id, then
    position) and each goes to the currently lightest shard (ties: lowest index).
    """
    if count < 1:
        raise ValueError(f"shard count must be >= 1, got {count}")
    weights = item_weights(node_ids, minutes)
    order = sorted(range(len(node_ids)), key=lambda position: (-weights[position], node_ids[position], position))
    loads = [0] * count
    assignment = [-1] * len(node_ids)
    for position in order:
        shard = min(range(count), key=lambda index: (loads[index], index))
        assignment[position] = shard
        loads[shard] += weights[position]
    return assignment


def lane_minutes(table: Mapping, lane: str) -> dict[str, float]:
    """Per-test minutes for one duration lane; an unknown lane is a configuration error."""
    rows = table["tests"].get(lane)
    if rows is None:
        raise KeyError(f"duration lane {lane!r} is not in {DURATIONS_PATH.name} (known: {sorted(table['tests'])})")
    return dict(rows)


def projected_shard_minutes(minutes: Mapping[str, float], path: str, count: int, workers: int = 1) -> list[float]:
    """Projected pytest minutes of each shard of one file from the table alone.

    The file's measured rows are partitioned exactly as the conftest would
    partition them; inside a shard the rows are packed onto ``workers`` xdist
    workers longest-first (``workers=1``: the serial mpirun lanes). This is a
    planning and ordering estimate only -- the live partition runs over the
    collected node ids, which may include rows the table has not measured yet.
    """
    node_ids = sorted(node for node in minutes if node.split("::", 1)[0] == path)
    assignment = weighted_partition(node_ids, count, minutes)
    weights = item_weights(node_ids, minutes)
    projected = []
    for shard in range(count):
        loads = [0] * max(1, workers)
        for weight in sorted((weights[i] for i, k in enumerate(assignment) if k == shard), reverse=True):
            loads[loads.index(min(loads))] += weight
        projected.append(max(loads) / _TENTHS)
    return projected


def select_shard(
    sorted_ids: Sequence[str], index: int, count: int, weights_lane: str = "", table: Mapping | None = None
) -> list[int]:
    """Positions of ``sorted_ids`` that execution shard ``index`` of ``count`` keeps.

    The shard function of ``tests/conftest.py::_apply_test_shard``.
    ``sorted_ids`` must already be sorted. An empty
    ``weights_lane`` is the stride ``[index::count]``; a lane name is the LPT
    partition over the committed durations (``table``, loaded when omitted).
    Raises ``ValueError``/``KeyError``/``OSError`` for an unusable table.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} is outside [0, {count})")
    if list(sorted_ids) != sorted(sorted_ids):
        raise ValueError("select_shard requires sorted node ids")
    if not weights_lane:
        return list(range(index, len(sorted_ids), count))
    minutes = lane_minutes(load_durations() if table is None else table, weights_lane)
    assignment = weighted_partition(sorted_ids, count, minutes)
    return [position for position, shard in enumerate(assignment) if shard == index]
