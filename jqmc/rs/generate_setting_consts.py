#!/usr/bin/env python3
"""Generate Rust compile-time constants without importing jqmc or its dependencies.

Only the reviewed names below cross the compile boundary. Every definition must
be one top-level, finite nonnegative numeric literal. Parsing the complete AST
also rejects later conditional/function/class shadowing of a selected name.
The generated f64::from_bits uses Python's exact binary64 literal value.
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

import ast
import math
import struct
import sys
from pathlib import Path


NAMES = (
    "FAST_AO_SCREEN_ABS_BOUND",
    "NATIVE_AO_SCREEN_EXPONENT",
    "EPS_safe_distance",
    "EPS_machine_precision",
    "EPS_rcond_SVD",
    "AS_LINEAR_MIN_SQ_NORM_FP64",
    "AS_NORM_CANCELLATION_THRESHOLD_FP64",
    "SM_LARGE_RATIO_THRESHOLD_FP64",
    "SM_INVERSE_REFRESH_RESIDUAL_THRESHOLD",
    "GEMINAL_LU_GUARD_RESIDUAL_THRESHOLD",
    "GEMINAL_EQUILIBRATION_MAX_EXPONENT",
    "NATIVE_DISTANCE_SKIP_THRESHOLD",
    "atol_consistency",
    "rtol_consistency",
    "ATOL_PYTHON_FP64_TIGHT",
    "ATOL_PYTHON_FP64_LOOSE",
    "ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT",
    "ATOL_GFMC_INCREMENTAL_TWIN",
    "ATOL_RS_UNIT_NEAR_MACHINE",
    "ATOL_RS_UNIT_REDUCTION",
    "ATOL_RS_UNIT_TIME_REVERSAL",
    "RTOL_RS_UNIT_RANK1_FORCE",
    "ATOL_RS_ERFINV_LEGACY_CORE",
    "ATOL_RS_ERFINV_LEGACY_TAIL",
    "FAST_ECP_NONLOCAL_RADIAL_CUT",
    "ATOL_REDUCTION_ORDER",
    "RTOL_REDUCTION_ORDER",
    "ATOL_BIT_EXACT",
    "ATOL_FD_VS_ANALYTIC_TIGHT",
    "RTOL_FD_VS_ANALYTIC_TIGHT",
    "RTOL_FASTMATH_EXP_VS_LIBM",
    "RTOL_FASTMATH_ERFC_VS_LIBM",
    "ATOL_FASTMATH_EWALD_IMAGE_SUM_VS_LIBM",
    "ATOL_FASTMATH_PERIODIC_SMOOTH_DIST",
    "RTOL_FASTMATH_PERIODIC_J12_RATIO",
    "RTOL_FASTMATH_F32_VS_LIBM",
    "ATOL_FASTMATH_F32_SINCOS_VS_LIBM",
)


def read_constants(path: Path) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions: dict[str, list[ast.AST]] = {name: [] for name in NAMES}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in definitions:
            definitions[node.id].append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in definitions:
            definitions[node.name].append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in definitions:
                    definitions[name].append(node)
    top = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top[node.target.id] = node.value
    values = {}
    for name, stores in definitions.items():
        if len(stores) != 1 or name not in top:
            raise ValueError(f"{name}: expected exactly one top-level definition, found {len(stores)}")
        value = top[name]
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
            literal = value.operand
        else:
            literal = value
        if not isinstance(literal, ast.Constant) or type(literal.value) not in (int, float):
            raise ValueError(f"{name}: expected a numeric literal")
        number = float(ast.literal_eval(value))
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name}: expected a finite nonnegative numeric literal")
        values[name] = number
    return values


def generate(source: Path, output: Path) -> None:
    values = read_constants(source)
    lines = ["// Generated from jqmc/_setting.py. Do not edit."]
    for name, value in values.items():
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        lines.append(f"pub const {name}: f64 = f64::from_bits(0x{bits:016x});")
    lines.append("pub const MANIFEST: &[(&str, f64)] = &[")
    lines.extend(f'    ("{name}", {name}),' for name in values)
    lines.append("];")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        generate(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError, SyntaxError, OverflowError) as exc:
        raise SystemExit(f"cannot generate Rust central settings: {exc}") from exc
