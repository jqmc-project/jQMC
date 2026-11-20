"""Independent minimum parameter-case requirements, not measured physics coverage.

Explicit tuples preserve coupled conditions without imposing a global Cartesian
product. Adding cases is allowed; deleting a required tuple or silently replacing
it while regenerating the derived binding is not.
Separately authored execution profiles are exact launch-scope requirements;
they never override the collected scope or authorize a skipped receipt.
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
import keyword


def execution_profile_errors(node, execution):
    """Require the independently authored launch scope to match the collected scope.

    Requirements never override the source-derived profile or turn an excluded
    run into a passing receipt. This is an exact execution scope, not a minimum
    parameter-coverage set.
    """
    if "required_execution_profiles" not in node:
        return []
    try:

        def profiles(value):
            if not isinstance(value, list) or not value:
                raise ValueError("execution profiles must be a nonempty list")
            identities = []
            for profile in value:
                if not isinstance(profile, dict) or set(profile) != {"name", "world_size"}:
                    raise ValueError("execution profile requires exactly name and world_size")
                name, size = profile["name"], profile["world_size"]
                if type(name) is not str or name not in {"serial", "mpi"}:
                    raise ValueError("execution profile name must be serial or mpi")
                if type(size) is not int or size < 1:
                    raise ValueError("execution profile world_size must be a positive integer")
                if (name == "serial") != (size == 1):
                    raise ValueError("execution profile name and world_size disagree")
                identities.append((name, size))
            if len(set(identities)) != len(identities):
                raise ValueError("execution profiles are duplicated")
            return set(identities)

        required = profiles(node["required_execution_profiles"])
        if not isinstance(execution, dict):
            raise ValueError("required execution profiles have no collected execution contract")
        collected = profiles(execution.get("profiles"))
        if required != collected:
            raise ValueError(
                f"required execution profiles {sorted(required)} differ from collected profiles {sorted(collected)}"
            )
    except (ValueError, TypeError, KeyError) as exc:
        return [f"execution scope: {exc}"]
    return []


def parameter_case_errors(node, cases, *, execution=None):
    """Check authored R against collected D before granting any runtime obligation."""
    profile_errors = execution_profile_errors(node, execution)
    if not {"required_parameter_cases", "required_precision_modes"} & node.keys():
        return profile_errors
    try:
        required = node.get("required_parameter_cases")
        if not isinstance(required, list) or not required:
            raise ValueError("required_parameter_cases must be a nonempty list")
        if not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError("required parameter cases have no collected cases")

        def value_schema(value):
            if type(value) in (str, bool, int, float, type(None)):
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    value_schema(item)
                return
            if isinstance(value, dict) and all(type(name) is str for name in value):
                for item in value.values():
                    value_schema(item)
                return
            raise ValueError("parameter selector values must be JSON data with string mapping keys")

        def key(selector):
            # The collector represents one non-parametrized test by {}. It is
            # an exact singleton requirement, not a wildcard for other inputs.
            if (
                not isinstance(selector, dict)
                or (not selector and (len(required) != 1 or len(cases) != 1))
                or any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) for name in selector)
            ):
                raise ValueError("required/collected parameter selector is malformed")
            value_schema(selector)
            return json.dumps(selector, sort_keys=True, allow_nan=False, separators=(",", ":"))

        def modes(value):
            if (
                not isinstance(value, list)
                or not value
                or any(type(mode) is not str or mode not in {"full", "mixed"} for mode in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError("required/collected precision modes must be distinct full/mixed names")
            return set(value)

        required_modes = modes(node.get("required_precision_modes"))
        if not isinstance(execution, dict):
            raise ValueError("required parameter cases have no collected execution contract")
        execution_modes = modes(execution.get("precision_modes"))
        requested = [key(row) for row in required]
        selectors = [case["selector"] for case in cases]
        observed = [key(row) for row in selectors]
        if len(set(requested)) != len(requested):
            raise ValueError("required parameter tuples are duplicated")
        if len(set(observed)) != len(observed):
            raise ValueError("collected parameter tuples are duplicated; aliases do not add coverage")
        names = set(required[0])
        if any(set(row) != names for row in required + selectors):
            raise ValueError("required and collected parameter names differ")
        missing = set(requested) - set(observed)
        if missing:
            raise ValueError(f"missing required parameter tuples: {sorted(missing)}")
        for identifier, case in zip(observed, cases):
            case_modes = modes(case.get("precision_modes", execution["precision_modes"]))
            if identifier in requested and not required_modes <= execution_modes & case_modes:
                raise ValueError(f"missing required precision modes for {identifier}")
    except (ValueError, TypeError, KeyError) as exc:
        return profile_errors + [f"parameter coverage: {exc}"]
    return profile_errors
