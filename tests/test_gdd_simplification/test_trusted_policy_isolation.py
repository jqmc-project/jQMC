"""The protected-contract check decides with the trusted base policy, in isolation."""

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

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import check_gdd_contract_changes as checker


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

REPO = Path(__file__).resolve().parents[2]
CHANGE = "toy.yaml: removed guarantee MATH-TOY-001"


def _git(root, *args):
    subprocess.run(
        ["git", "-c", "user.name=Metadata Test", "-c", "user.email=metadata@example.test", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_trusted_policy_stays_red_without_importing_candidate_guards(tmp_path, monkeypatch, capsys):
    """Run the real isolated launcher, with unrelated guard inputs stubbed only in its toy trusted fixture."""
    root = tmp_path
    for relative in ("scripts", "gdd/engine"):
        shutil.copytree(REPO / relative, root / relative, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    policy = root / "scripts/check_gdd_contract_changes.py"
    source = policy.read_text(encoding="utf-8")
    fixture = (
        f"\ndef validation_errors(base_ref):\n    return ([{CHANGE!r}] if (ROOT / 'protected-change').exists() else [], {{}})\n"
        "\ndef trusted_source_errors(base_ref):\n    return []\n\n"
    )
    source = source.replace('if __name__ == "__main__":', fixture + 'if __name__ == "__main__":')
    policy.write_text(source, encoding="utf-8")
    _git(root, "init", "-q", "-b", "trusted")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "trusted isolated policy fixture")
    monkeypatch.setattr(checker, "ROOT", root)
    assert checker.run_trusted_base_policy("trusted") == 0
    (root / "protected-change").write_text("candidate\n", encoding="utf-8")
    assert checker.run_trusted_base_policy("trusted") == 1
    output = capsys.readouterr()
    assert CHANGE in output.err
    assert "accepted only by merging" in output.err
    (root / "protected-change").unlink()
    candidate_guard = root / "gdd/engine/contract_guard.py"
    marker = root / "candidate-guard-executed"
    candidate_guard.write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\nraise RuntimeError("candidate guard")\n',
        encoding="utf-8",
    )
    assert checker.run_trusted_base_policy("trusted") == 1
    output = capsys.readouterr()
    assert "guard change requires separate review: gdd/engine/contract_guard.py" in output.err
    assert not marker.exists()  # The reported source diff may quote the tripwire; it must never execute it.
