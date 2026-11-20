"""Tests for jqmc_workflow._phase -- scientific phase / action guard rails.

Layer 1 (pure unit): no subprocess, no SSH, no filesystem.

Test class layout:
    TestScientificPhaseEnum    -- enum value invariants
    TestPhaseTransitions       -- can_advance / advance_phase / rollback_phase
    TestAllowedActions         -- status-dependent action filtering
    TestRequireAction          -- guard-rail raise / pass
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

import pytest

from jqmc_workflow._phase import (
    ALWAYS_ALLOWED_ACTIONS,
    PHASE_ALLOWED_ACTIONS,
    PHASE_TRANSITIONS,
    ScientificPhase,
    advance_phase,
    allowed_actions,
    can_advance,
    require_action,
    rollback_phase,
)
from jqmc_workflow.state import WorkflowStatus

pytestmark = [pytest.mark.without_chdir]


class TestScientificPhaseEnum:
    """Enum values are stable identifiers consumed by downstream tools."""

    @pytest.mark.light
    def test_all_phases_have_string_values(self):
        for phase in ScientificPhase:
            assert isinstance(phase.value, str)
            assert phase.value != ""

    @pytest.mark.light
    def test_phase_values_are_unique(self):
        values = [p.value for p in ScientificPhase]
        assert len(values) == len(set(values))

    @pytest.mark.light
    def test_every_phase_appears_in_transition_table(self):
        for phase in ScientificPhase:
            assert phase in PHASE_TRANSITIONS

    @pytest.mark.light
    def test_every_phase_appears_in_allowed_actions_table(self):
        for phase in ScientificPhase:
            assert phase in PHASE_ALLOWED_ACTIONS


class TestPhaseTransitions:
    """can_advance / advance_phase / rollback_phase."""

    @pytest.mark.light
    def test_can_advance_init_to_scf(self):
        assert can_advance(ScientificPhase.INIT, ScientificPhase.SCF) is True

    @pytest.mark.light
    def test_can_advance_init_to_wf_build(self):
        assert can_advance(ScientificPhase.INIT, ScientificPhase.WF_BUILD) is True

    @pytest.mark.light
    def test_cannot_advance_init_to_vmc(self):
        assert can_advance(ScientificPhase.INIT, ScientificPhase.VMC) is False

    @pytest.mark.light
    def test_cannot_advance_from_completed(self):
        for target in ScientificPhase:
            assert can_advance(ScientificPhase.COMPLETED, target) is False

    @pytest.mark.light
    def test_advance_phase_returns_target_on_valid(self):
        new = advance_phase(ScientificPhase.INIT, ScientificPhase.SCF)
        assert new is ScientificPhase.SCF

    @pytest.mark.light
    def test_advance_phase_raises_on_invalid(self):
        with pytest.raises(ValueError, match="Cannot advance"):
            advance_phase(ScientificPhase.INIT, ScientificPhase.VMC)

    @pytest.mark.light
    def test_advance_phase_error_lists_allowed(self):
        with pytest.raises(ValueError) as excinfo:
            advance_phase(ScientificPhase.INIT, ScientificPhase.VMC)
        msg = str(excinfo.value)
        assert "scf" in msg and "wf_build" in msg

    @pytest.mark.light
    def test_rollback_phase_returns_predecessor(self):
        # WF_BUILD is reachable from INIT and SCF; rollback picks the
        # most recent predecessor by insertion order.
        new = rollback_phase(ScientificPhase.WF_BUILD)
        assert new in {ScientificPhase.INIT, ScientificPhase.SCF}

    @pytest.mark.light
    def test_rollback_phase_raises_on_init(self):
        with pytest.raises(ValueError, match="no predecessor"):
            rollback_phase(ScientificPhase.INIT)

    @pytest.mark.light
    def test_self_loops_are_explicit(self):
        # VMC, MCMC, LRDMC are documented self-loops (re-runnable phases).
        for phase in (ScientificPhase.VMC, ScientificPhase.MCMC, ScientificPhase.LRDMC):
            assert can_advance(phase, phase) is True

    @pytest.mark.parametrize(
        "current,target",
        [
            (ScientificPhase.WF_BUILD, ScientificPhase.VMC_PILOT),
            (ScientificPhase.VMC_PILOT, ScientificPhase.VMC),
            (ScientificPhase.VMC, ScientificPhase.COMPLETED),
            (ScientificPhase.LRDMC_FIT, ScientificPhase.COMPLETED),
        ],
    )
    @pytest.mark.light
    def test_canonical_forward_paths(self, current, target):
        assert can_advance(current, target) is True


class TestAllowedActions:
    """allowed_actions filters by status."""

    @pytest.mark.light
    def test_pending_status_returns_phase_actions_plus_always(self):
        actions = allowed_actions(ScientificPhase.SCF, WorkflowStatus.PENDING)
        assert "run_scf" in actions
        for always in ALWAYS_ALLOWED_ACTIONS:
            assert always in actions

    @pytest.mark.light
    def test_running_status_excludes_configure(self):
        actions = allowed_actions(ScientificPhase.VMC, WorkflowStatus.RUNNING)
        assert "configure_vmc" not in actions
        assert "run_vmc" in actions  # non-configure phase actions retained

    @pytest.mark.light
    def test_failed_status_keeps_only_recover(self):
        actions = allowed_actions(ScientificPhase.VMC, WorkflowStatus.FAILED)
        # Recover-prefixed actions kept
        assert any(a.startswith("recover_") for a in actions)
        # Non-recover phase actions filtered
        assert "run_vmc" not in actions
        assert "configure_vmc" not in actions
        # rollback_phase explicitly appended
        assert "rollback_phase" in actions

    @pytest.mark.light
    def test_cancelled_status_behaves_like_failed(self):
        failed = set(allowed_actions(ScientificPhase.VMC, WorkflowStatus.FAILED))
        cancelled = set(allowed_actions(ScientificPhase.VMC, WorkflowStatus.CANCELLED))
        assert failed == cancelled

    @pytest.mark.light
    def test_completed_phase_has_only_always_allowed(self):
        actions = allowed_actions(ScientificPhase.COMPLETED, WorkflowStatus.PENDING)
        assert set(actions) == set(ALWAYS_ALLOWED_ACTIONS)

    @pytest.mark.light
    def test_always_allowed_present_in_every_status(self):
        for status in WorkflowStatus:
            actions = allowed_actions(ScientificPhase.VMC, status)
            for always in ALWAYS_ALLOWED_ACTIONS:
                assert always in actions, f"missing {always} under status={status.value}"


class TestRequireAction:
    """require_action raises on disallowed, returns None on allowed."""

    @pytest.mark.light
    def test_allowed_action_does_not_raise(self):
        assert require_action("run_scf", ScientificPhase.SCF, WorkflowStatus.PENDING) is None

    @pytest.mark.light
    def test_always_allowed_works_in_any_phase(self):
        for phase in ScientificPhase:
            require_action("advance_phase", phase, WorkflowStatus.PENDING)

    @pytest.mark.light
    def test_disallowed_action_raises(self):
        with pytest.raises(ValueError, match="not allowed"):
            require_action("run_vmc", ScientificPhase.SCF, WorkflowStatus.PENDING)

    @pytest.mark.light
    def test_configure_disallowed_while_running(self):
        with pytest.raises(ValueError):
            require_action("configure_vmc", ScientificPhase.VMC, WorkflowStatus.RUNNING)

    @pytest.mark.light
    def test_non_recover_disallowed_while_failed(self):
        with pytest.raises(ValueError):
            require_action("run_vmc", ScientificPhase.VMC, WorkflowStatus.FAILED)

    @pytest.mark.light
    def test_recover_allowed_while_failed(self):
        require_action("recover_vmc", ScientificPhase.VMC, WorkflowStatus.FAILED)

    @pytest.mark.light
    def test_error_message_includes_phase_status_and_allowed(self):
        with pytest.raises(ValueError) as excinfo:
            require_action("run_vmc", ScientificPhase.SCF, WorkflowStatus.PENDING)
        msg = str(excinfo.value)
        assert "scf" in msg
        assert "pending" in msg
        assert "Allowed" in msg
