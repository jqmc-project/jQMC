"""Atomic-force / stress / pressure table parsers for jQMC output.

The table parsers are implemented in :mod:`jqmc.output_parser` and
re-exported here for the MCMC and LRDMC workflows. This module keeps
:func:`repair_forces_from_output`, which needs the workflow state; its I/O
helper (``_find_output_files``) stays in ``output_parser``.
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

import os
from logging import getLogger

# The table parsers live in ``jqmc.output_parser`` (one implementation for
# every driver); they are re-exported here for the workflow package.
from jqmc.output_parser import (
    _PRESSURE_COMPONENTS,
    _STRESS_COMPONENTS,
    _parse_labelled_ufloat_table,
    parse_force_table,
    parse_pressure_table,
    parse_stress_table,
    parse_ufloat_short,
)

from .output_parser import _find_output_files

__all__ = [
    "_PRESSURE_COMPONENTS",
    "_STRESS_COMPONENTS",
    "_parse_labelled_ufloat_table",
    "parse_force_table",
    "parse_pressure_table",
    "parse_stress_table",
    "parse_ufloat_short",
    "repair_forces_from_output",
]

logger = getLogger("jqmc-workflow").getChild(__name__)


def repair_forces_from_output(work_dir: str) -> bool:
    """Re-parse forces from output files and update ``workflow_state.toml``.

    The force table of the last output file is parsed with
    ``parse_force_table``, which reads scientific notation (e.g.
    ``+3(8)e-05`` as ``3e-05``), and replaces ``result.forces`` in the
    state, so force data read with a wrong exponent is rewritten.

    Returns *True* if the TOML was updated, *False* otherwise.
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return False

    out_files = _find_output_files(work_dir)
    if not out_files:
        return False
    last_out = out_files[-1]

    # Read and parse the force table from the output file
    try:
        with open(last_out) as f:
            text = f.read()
    except Exception:
        return False

    forces = parse_force_table(text)
    if forces is None:
        return False

    # Update the TOML atomically via the canonical state API.
    from .state import _write, read_state

    state = read_state(work_dir)
    state.setdefault("result", {})["forces"] = forces
    _write(work_dir, state)

    logger.info(f"  Repaired forces in {work_dir} from {os.path.basename(last_out)}")
    return True
