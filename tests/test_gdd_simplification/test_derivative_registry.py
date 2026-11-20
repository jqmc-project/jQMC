"""The node-level derivative registry, its in-test self-check and its static call-site pin (issue 789)."""

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

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gdd.engine import derivative_registry as registry
from gdd.engine import evidence

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

NODE = "tests/test_jqmc_kernel/test_toy.py::test_emit[a]"
KEY = "private|leaf|P01"


def _receipt(obligation_id):
    """A genuine, validating case receipt (as the job-completion toy emits)."""
    electrons = evidence.observe_electron_configurations(
        evidence.ElectronConfigurationBinding(
            binding_id="toy-positions",
            mode="required",
            generation_pattern="direct",
            position_source_kind="seeded-random",
            generator_operation="toy-generator",
            random_algorithm="numpy.random.PCG64",
            seed=1,
            configuration_count=1,
            geminal_conditioning_mode="not-limited",
            production_route_capture="toy-route",
        ),
        np.ones((1, 3)),
        np.empty((0, 3)),
    )
    capture = evidence.ProductionRouteCapture("toy-route", lambda args, kwargs: [(args[0], args[1])])
    values = capture.wrap(lambda up, down: up + 1)(np.ones((1, 3)), np.empty((0, 3)))
    compared = evidence.observe_exact(
        evidence.ComparisonBinding("toy-comparison", "CMP-EXACT-VALUES-001", "full", "actual", "reference"),
        [(values, np.full((1, 3), 2.0))],
        electron_receipt=electrons,
    )
    return evidence.CaseReceipt(
        guarantee_id="MATH-DERIV-TOY-001",
        obligation_id=obligation_id,
        pytest_node_id=NODE,
        input_binding=evidence.InputBinding(
            binding_id="toy-input",
            kind="parameter-set",
            builder_operation="toy-builder",
            transformations=(),
            parameter_set_id="toy-parameters",
            parameter_source="toy-source",
            content_snapshot="sha256:toy",
        ),
        electron_configuration=electrons,
        comparison=compared,
        additional_checks=(compared,),
        route_capture=capture.receipt(),
        status="passed",
    )


UNRELATED = "OPS-UNRELATED-TOY-001"


def _emitted(monkeypatch, *obligation_ids, guarantee=None, fragments=()):
    """The payloads emit_completion_record hands to the self-check sink (nothing is written).

    ``fragments`` are ``(upstream, component)`` Rust fragment receipts.
    """
    records = []
    monkeypatch.setattr(evidence, "_SELFCHECK_SINK", records.append)
    monkeypatch.delenv("GDD_RECEIPT_DIR", raising=False)
    for obligation_id in obligation_ids:
        receipt = _receipt(obligation_id)
        evidence.emit_completion_record(replace(receipt, guarantee_id=guarantee) if guarantee else receipt)
    for upstream, component in fragments:
        receipt = replace(_receipt(f"{upstream}::rust-component={component}"), guarantee_id="IMPL-DERIV-RUST-XLA-001")
        evidence.emit_completion_record(
            receipt, record_kind="rust-fragment", upstream_obligation_id=upstream, component_id=component
        )
    return records


def _item(marker=None):
    return SimpleNamespace(
        nodeid=NODE,
        callspec=SimpleNamespace(params={"p": "a"}),
        get_closest_marker=lambda name: marker if name == registry.SYNTHETIC_MARKER else None,
        config=SimpleNamespace(getoption=lambda name, default=None: "full"),
    )


def _plugin(keys=(KEY,), passed=True, lane="xla-full", upstream=()):
    def toy(definition, params, precision):
        if definition != NODE.split("[", 1)[0]:
            return None
        assert params == {"p": "a"} and precision == "full"
        return {"lane": lane, "keys": list(keys), "upstream": list(upstream)}

    plugin = registry.DerivativeSelfCheck(toy)
    plugin._call_passed = passed
    return plugin


FRAGMENT = f"{KEY}::P01"


@pytest.mark.gdd_synthetic_derivative_receipts(reason="drives the self-check with toy receipts")
@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing",
        "invalid",
        "unrelated-guarantee",
        "wrong-kind",
        "wrong-lane",
        "upstream-wrong-guarantee",
    ],
)
def test_selfcheck_rejects_a_missing_extra_misclassified_or_invalid_receipt(monkeypatch, defect):
    """Each receipt must classify (aggregator classify) into exactly its registered (lane, key) and validate.

    ``none`` is the control: a Rust case attests its fragment key, its upstream
    replay classifying as the XLA receipt of its precision. A receipt with the right obligation id but an unrelated guarantee, a
    primary Rust key emitted as a plain case, an FD receipt registered in the
    mixed lane, or an upstream replay of an unrelated guarantee closes nothing:
    the case fails at teardown and is never attested.
    """
    plugin, records = {
        "none": lambda: (
            _plugin((FRAGMENT,), lane="rust-full", upstream=(KEY,)),
            _emitted(monkeypatch, KEY, fragments=((KEY, "P01"),)),
        ),
        "missing": lambda: (_plugin((KEY, f"{KEY}-2")), _emitted(monkeypatch, KEY)),
        "invalid": lambda: (_plugin(), _emitted(monkeypatch, KEY)),
        "unrelated-guarantee": lambda: (_plugin(), _emitted(monkeypatch, KEY, guarantee=UNRELATED)),
        "wrong-kind": lambda: (_plugin((FRAGMENT,), lane="rust-full"), _emitted(monkeypatch, FRAGMENT)),
        "wrong-lane": lambda: (_plugin(lane="mixed"), _emitted(monkeypatch, KEY, guarantee="NUM-DERIV-FD-AUTODIFF-001")),
        "upstream-wrong-guarantee": lambda: (
            _plugin((FRAGMENT,), lane="rust-full", upstream=(KEY,)),
            _emitted(monkeypatch, KEY, guarantee=UNRELATED, fragments=((KEY, "P01"),)),
        ),
    }[defect]()
    if defect == "none":
        plugin.check(_item(), records)
        assert plugin.attestations == {NODE: {"lane": "rust-full", "keys": 1}}
        return
    if defect == "invalid":
        records[0]["status"] = "failed"
    message = "invalid derivative receipt" if defect == "invalid" else "classified derivative receipts differ"
    with pytest.raises(AssertionError, match=message):
        plugin.check(_item(), records)
    assert plugin.attestations == {}


@pytest.mark.gdd_synthetic_derivative_receipts(reason="drives the self-check with a toy receipt")
def test_selfcheck_never_attests_a_failed_case_and_rejects_an_unregistered_emitter(monkeypatch):
    """A failed call is not attested; a case outside the registry may not emit."""
    plugin = _plugin(passed=False)
    plugin.check(_item(), [])
    assert plugin.attestations == {}
    stray = registry.DerivativeSelfCheck(lambda definition, params, precision: None)
    with pytest.raises(AssertionError, match="not a registered emitter"):
        stray.check(_item(), _emitted(monkeypatch, KEY))
    # A marked contract test is outside the registry on purpose.
    stray.check(_item(marker=object()), _emitted(monkeypatch, KEY))
