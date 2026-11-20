"""Registry resolving requirement-side GDD operation IDs to engine implementations.

The SSoT (``gdd/ssot/guarantees/derivatives.yaml``) references stable operation
IDs.  Requirement-side IDs -- the ones that expand or pin the reviewed
obligation universe -- must resolve here, through the constrained engine,
never through a test module.  Evidence-side IDs (pytest registration,
fixture construction, runtime receipts) are bound by the pytest adapters and
are intentionally absent from this registry until their own promotion is
reviewed.
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

from collections.abc import Callable
from typing import Any

from gdd.engine import derivative_inventory, derivative_matrix, derivative_requirements, production_entry_discovery


REQUIREMENT_OPERATIONS: dict[str, Callable[[], Any]] = {
    "derivatives.required_cases.v1": lambda: derivative_requirements.REQUIRED_CASES,
    "derivatives.continuous_fixture_matrix.v1": lambda: derivative_matrix._generate_matrix(),
    "derivatives.gfmc_fixture_matrix.v1": derivative_requirements.gfmc_matrix_payload,
    "derivatives.production_inventory.v1": lambda: {
        "branches": derivative_inventory.PRODUCTION_BRANCH_INVENTORY,
        "validity": derivative_inventory.PRODUCTION_VALIDITY_CONSTRAINTS,
        "parameter_leaves": derivative_inventory.PARAMETER_LEAF_INVENTORY,
        "numerical_regimes": derivative_inventory.PRODUCTION_NUMERICAL_REGIMES,
    },
    "derivatives.production_entry_discovery.v1": lambda: (
        production_entry_discovery.discover_external_derivative_references,
        production_entry_discovery.discover_direct_autodiff_calls,
    ),
}


def resolve_requirement_operation(operation_id: str) -> Any:
    """Resolve one requirement-side operation ID; unknown IDs fail loudly."""
    try:
        operation = REQUIREMENT_OPERATIONS[operation_id]
    except KeyError:
        raise KeyError(f"operation ID is not a registered requirement-side engine operation: {operation_id}") from None
    return operation()
