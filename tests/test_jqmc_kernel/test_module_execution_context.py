"""Requested execution context is active when module-scoped models are built."""

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
import subprocess
import sys

import pytest

from jqmc._precision import is_mixed_precision_enabled

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.fixture(scope="module")
def model_creation_context():
    """Capture before the existing function-scoped initialization can run."""
    return {
        "mixed": is_mixed_precision_enabled(),
        "rust_dispatch_environment": os.environ.get("JQMC_FLAG_RUST_ON_CPU"),
    }


def test_module_models_receive_requested_execution_context(request, model_creation_context):
    """Fresh-process module construction uses the lane precision and XLA guard."""
    mode = request.config.getoption("--precision-mode")
    if os.environ.get("JQMC_MODULE_CONTEXT_CHILD") != "1":
        node = "tests/test_jqmc_kernel/test_module_execution_context.py::test_module_models_receive_requested_execution_context"
        child_environment = os.environ.copy()
        for name in ("GDD_RECEIPT_DIR", "PYTEST_ADDOPTS", "JQMC_TEST_SHARD_WEIGHTS"):
            child_environment.pop(name, None)
        child_environment.update(
            JQMC_MODULE_CONTEXT_CHILD="1",
            JQMC_FLAG_RUST_ON_CPU="1",
            JQMC_TEST_SHARD_COUNT="1",
            JQMC_TEST_SHARD_INDEX="0",
            GDD_RECEIPT_AUTHORITY="0",
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--precision-mode=" + mode, node],
            cwd=Path(__file__).resolve().parents[2],
            env=child_environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    expected_mixed = mode == "mixed"
    assert model_creation_context["mixed"] is expected_mixed
    assert model_creation_context["rust_dispatch_environment"] == "0"
