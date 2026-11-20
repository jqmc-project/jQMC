"""The wired bound-comparison gate on the real tree (``gdd_sync --check`` runs it)."""

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

from pathlib import Path

import pytest

from gdd.engine.case_binding_inventory import derive_case_binding
from gdd.engine.ledger import iter_registered_nodes, load_ledger
from scripts.check_gdd_bound_comparisons import agreement_methods, node_errors

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]
ROOT = Path(__file__).resolve().parents[2]


def _real_and_rewritten(monkeypatch, ledger, guarantee_id, test_file, test_name, old, new):
    """A registered node's recorded observations, and those of its source with ``old`` replaced by ``new``."""
    ledger, test_file = ROOT / ledger, ROOT / test_file
    pytest_node = f"{test_file.relative_to(ROOT).as_posix()}::{test_name}"
    group, node = next(
        (group, node)
        for guarantee, group, node in iter_registered_nodes(load_ledger(ledger), ledger)
        if guarantee["id"] == guarantee_id and node["pytest_node"] == pytest_node
    )
    source = test_file.read_text(encoding="utf-8")
    assert source.count(old) == 1
    mutated = source.replace(old, new)
    read_text = Path.read_text

    def mutated_read(path, *args, **kwargs):
        return mutated if path.resolve() == test_file else read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mutated_read)
    binding = derive_case_binding(
        pytest_node=pytest_node,
        expected_cases=node["expected_cases"],
        verdict_method=group["verdict_method"],
        wavefunction_scope=node["case_binding"]["wavefunction"]["scope"],
    )
    return (
        pytest_node,
        node.get("executable_checks"),
        group["verdict_method"],
        node["case_binding"]["comparison"]["typed_observations"],
        binding["comparison"]["typed_observations"],
    )


def test_fd_verdict_reduced_to_a_finiteness_check_turns_the_gate_red(monkeypatch):
    """The real FD-reference node keeps its bound comparison; a finiteness-only rewrite of its source does not.

    In the evidence of NUM-OPT-COMPLEX-ECP-LM-FD-001 the two ``assert_fd``
    comparisons are replaced by a finiteness assertion; the gate reports the
    missing comparison.
    """
    node, checks, _method, real, rewritten = _real_and_rewritten(
        monkeypatch,
        "gdd/ssot/guarantees/jqmc_optimizer.yaml",
        "NUM-OPT-COMPLEX-ECP-LM-FD-001",
        "tests/test_jqmc_kernel/test_complex_ecp_optimizer_contract.py",
        "test_reduced_lm_update_matches_independent_fd_reference",
        '    assert_fd(solved.q, q, "FD-input LM update")\n    assert_fd(solved.predicted_energy, energy, "FD-input LM eigenvalue")\n',
        "    assert np.all(np.isfinite(solved.q)) and np.isfinite(solved.predicted_energy)\n",
    )
    assert node_errors(node, checks, real) == []
    errors = node_errors(node, checks, rewritten)
    assert len(errors) == 1 and "'complex-ecp-value'" in errors[0] and "is missing" in errors[0], errors


def test_parity_verdict_without_authored_checks_reduced_to_a_finiteness_check_turns_the_gate_red(monkeypatch):
    """A backend-parity node with no authored checks needs its agreement comparison, not finiteness.

    In the evidence of IMPL-AO-RUST-XLA-001,
    ``assert _close(got, ref)`` becomes ``assert np.all(np.isfinite(got))``; the
    remaining ``pgto._n_image >= 1`` is a one-sided constant bound, not agreement.
    """
    node, checks, method, real, rewritten = _real_and_rewritten(
        monkeypatch,
        "gdd/ssot/guarantees/jqmc_rs.yaml",
        "IMPL-AO-RUST-XLA-001",
        "tests/test_rs_kernel/test_atomic_orbital_pgto.py",
        "test_pgto_value_matches_jax",
        "    assert got.shape == (pgto.num_ao, 6), (got.shape, pgto.num_ao)\n    assert _close(got, ref), np.abs(got - ref).max()\n",
        "    assert got.shape == (pgto.num_ao, 6), (got.shape, pgto.num_ao)\n    assert np.all(np.isfinite(got))\n",
    )
    methods = agreement_methods()
    assert checks is None and method in methods
    assert node_errors(node, checks, real, method, methods) == []
    errors = node_errors(node, checks, rewritten, method, methods)
    assert len(errors) == 1 and "no agreement comparison" in errors[0], errors
