"""Path-coverage predicate for path-filtered static GDD pre-commit hooks.

A static GDD
gate hook may run only when a relevant path changes, so a documentation-only
commit does not re-run every gate. Such a hook stays fail-closed only while
its ``files`` filter selects every protected input tree. This module is the
predicate of ``scripts/check_gdd_contract_changes.py`` (identity of the trusted
``gdd-*`` hooks).

A hook satisfies the predicate when it is ``always_run: true`` (the original
unconditional form), or when ``always_run`` is ``false`` and pre-commit's own
file selection -- ``re.search`` of ``files``, minus ``re.search`` of
``exclude``, with no ``types``/``types_or``/``exclude_types`` narrowing --
selects one representative path under every protected prefix below.

Pure deletions never reach a pre-commit hook (pre-commit stages only
``--diff-filter=ACMRTUXB``); ``scripts/run_gdd_deleted_path_hooks.py`` (hook
``gdd-deleted-path-hooks``, always run) closes that gap.
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

import re
from typing import Any

# One representative path per protected input prefix of the static GDD gates.
PROTECTED_SAMPLE_PATHS = (
    "gdd/ssot/x.yaml",
    "gdd/engine/x.py",
    "gdd/derived/x.yaml",
    "tests/test_jqmc_kernel/test_x.py",
    "jqmc/x.py",
    "jqmc/rs/src/x.rs",
    "scripts/x.py",
    ".pre-commit-config.yaml",
    ".github/workflows/x.yml",
)
DELETED_PATH_HOOK_ID = "gdd-deleted-path-hooks"
DELETED_PATH_HOOK_COMMAND = ("scripts/run_gdd_deleted_path_hooks.py",)
_NARROWING_KEYS = ("types_or", "exclude_types")


def selects(hook: dict[str, Any], path: str) -> bool:
    """Whether pre-commit's ``files``/``exclude`` regexes select ``path`` for ``hook``."""
    files = hook.get("files", "")
    exclude = hook.get("exclude", "^$")
    return bool(re.search(files, path)) and not re.search(exclude, path)


def path_filter_errors(hook: Any, label: str) -> list[str]:
    """Reasons ``hook`` is not an unconditional-equivalent static gate hook (empty = accepted)."""
    if not isinstance(hook, dict):
        return [f"{label}: hook is not a mapping"]
    if hook.get("always_run") is True:
        return []
    if "always_run" in hook and hook["always_run"] is not False:
        return [f"{label}: always_run must be a boolean"]
    files = hook.get("files")
    if not isinstance(files, str) or not files:
        return [f"{label}: always_run is false but no files regex selects the protected GDD inputs"]
    exclude = hook.get("exclude", "^$")
    if not isinstance(exclude, str):
        return [f"{label}: exclude must be a regex string"]
    try:
        re.compile(files)
        re.compile(exclude)
    except re.error as exc:
        return [f"{label}: invalid files/exclude regex: {exc}"]
    errors = []
    types = hook.get("types", ["file"])
    if types != ["file"]:
        errors.append(f"{label}: a path-filtered GDD gate must not narrow by types {types!r}")
    errors.extend(f"{label}: a path-filtered GDD gate must not set {key}" for key in _NARROWING_KEYS if hook.get(key))
    for sample in PROTECTED_SAMPLE_PATHS:
        if not re.search(files, sample):
            errors.append(f"{label}: files regex does not select protected path {sample}")
        elif re.search(exclude, sample):
            errors.append(f"{label}: exclude regex removes protected path {sample}")
    return errors


def path_filter_relaxation_errors(before: dict[str, Any], after: Any, label: str) -> list[str]:
    """The only accepted change of a trusted hook: ``always_run: true`` -> ``false`` + a covering ``files``.

    Every other field (id, name, entry, language, pass_filenames, stages,
    args, additional_dependencies, ...) must be identical; a removed hook is
    rejected by the caller.
    """
    if after == before:
        return []
    if not isinstance(after, dict):
        return [f"removed or changed trusted GDD hook {label}"]
    ignored = {"always_run", "files"}
    rest_before = {key: value for key, value in before.items() if key not in ignored}
    rest_after = {key: value for key, value in after.items() if key not in ignored}
    if (
        rest_before != rest_after
        or before.get("always_run") is not True
        or "files" in before
        or after.get("always_run") is not False
    ):
        return [f"removed or changed trusted GDD hook {label}"]
    return path_filter_errors(after, f"trusted GDD hook {label}")
