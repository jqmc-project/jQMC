"""Fresh GFMC driver children import their checkout while keeping a private CWD."""

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

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]
ROOT = Path(__file__).resolve().parents[2]
DRIVER_SCRIPT = ROOT / "tests/test_rs_kernel/test_gfmc_pure_leaf_j3_force_stress.py"


@pytest.mark.gdd_no_committed_artifact(
    reason="executes the real child import/bootstrap path and checks source/native identity; no physical trajectory or fixture"
)
def test_pure_leaf_driver_child_imports_from_private_workdir(tmp_path):
    """A wheel-installed child needs neither inherited PYTHONPATH nor a shared CWD."""
    import jqmc_rs
    from jqmc._precision import is_mixed_precision_enabled, precision_policy_snapshot
    from tests.test_rs_kernel.test_build_identity import _source_fnv64

    assert Path.cwd().resolve() == tmp_path.resolve()
    mode = "mixed" if is_mixed_precision_enabled() else "full"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GDD_")}
    env.pop("PYTHONPATH", None)
    env["GDD_RECEIPT_AUTHORITY"] = "0"
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "child-jax-cache")
    output = tmp_path / "child-bootstrap.json"
    child = subprocess.run(
        [sys.executable, str(DRIVER_SCRIPT), "--bootstrap-probe", mode, str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    metadata = json.loads(output.read_text())
    assert metadata["cwd"] == str(tmp_path.resolve())
    assert metadata["pythonpath"] is None and metadata["receipt_authority"] == "0"
    assert metadata["mode"] == mode and metadata["requested_policy"] == precision_policy_snapshot()
    assert metadata["standalone_dispatch"] == jqmc_rs.native_precision_support()
    assert metadata["script_source"] == str(DRIVER_SCRIPT.resolve())
    assert metadata["script_sha256"] == hashlib.sha256(DRIVER_SCRIPT.read_bytes()).hexdigest()
    assert metadata["tests_package_source"] == str(ROOT / "tests/__init__.py")
    assert metadata["jqmc_source"] == str(ROOT / "jqmc/__init__.py")
    assert metadata["native_binary"] == str(Path(jqmc_rs.__file__).resolve())
    assert metadata["native_binary_sha256"] == hashlib.sha256(Path(jqmc_rs.__file__).read_bytes()).hexdigest()
    assert metadata["native_source_fnv64"] == jqmc_rs.__source_fnv64__ == _source_fnv64()
    assert metadata["physical_driver_executed"] is False
