"""Static correctness gates for public documentation and examples."""

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
import re
import shlex
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT))
from examples.sync_examples import INCLUDE_BLOCK as INCLUDE_PATTERN, INCLUDE_MARKER, included_text

LINK_PATTERN = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
PYTHON_FENCE_PATTERN = re.compile(r"^```(?:python|py)\s*\n([\s\S]*?)^```", re.MULTILINE)
BANNED_COMMANDS = {
    "jqmc-tool qe convert-to": "use 'jqmc-tool qe <save-dir>'",
    "jqmc-tool mcmc analyze-output": "use 'jqmc-tool mcmc compute-energy <restart.h5>'",
    "jqmc-tool lrdmc analyze-output": "use an lrdmc compute/extrapolate command",
}
BANNED_PUBLIC_PHRASES = {
    "opt_J1/J2/J3/JNN_param": "jqmc has no neural-network Jastrow and no opt_JNN_param",
    "opt_JDTN_cutoffs": "DTN cutoffs are fixed data and have no optimization flag",
    "Jastrow_DTN_sparse_state": "jqmc has no DTN sparse neighbor-list implementation",
    "preflight_dtn_sparse_caps": "jqmc has no DTN sparse neighbor-list implementation",
    "DTN_SPARSE_STATE_CAP": "jqmc has no DTN sparse neighbor-list implementation",
    "DTN_SPARSE_ATOM_CAP": "jqmc has no DTN sparse neighbor-list implementation",
    "_compute_delta_J_DTN_mesh_fused": "the current implementation is _compute_delta_J_DTN_mesh_dense",
    "GFMC_n driver does not yet expose stress": "GFMC_n supports stress",
    "Stress is not implemented for the lrdmc-bra": "lrdmc-bra/GFMC_n supports stress",
    "GFMC_n stress is not implemented": "GFMC_n supports stress",
    "Available only on `lrdmc-tau`": "both LRDMC drivers support stress",
    "mandatory when `--jastrow-dtn != none`": "DTN is mutually exclusive with the Pade Jastrow family",
    "adaptive_learning_rate": "use optimizer_kwargs.use_lm",
    '`level` = `"strict"` or `"loose"`': "the supported tolerance levels are strict and medium",
    "jqmc.__version__": "read the installed distribution version with importlib.metadata.version('jqmc')",
    "jax[rocm7]": "the current JAX ROCm extra is jax[rocm7-local]",
    "structure_data.k_point": "the twist is owned by the determinant AO leaf's k_point field",
    "Stage 2 readiness notice": "non-Gamma and complex TREXIO loading are implemented",
    "None of that is shipped": "non-Gamma and complex TREXIO loading are implemented",
    "Stage 2, PLANNED": "describe the current TREXIO and QE converter behavior",
}
# Central analysis minima (jqmc/_setting.py) per tool family, keyed by TOML key / long option name.
ANALYSIS_MINIMA = {
    "mcmc": {"num_mcmc_warmup_steps": "MCMC_MIN_WARMUP_STEPS", "num_mcmc_bin_blocks": "MCMC_MIN_BIN_BLOCKS"},
    "lrdmc": {
        "num_gfmc_warmup_steps": "GFMC_MIN_WARMUP_STEPS",
        "num_gfmc_bin_blocks": "GFMC_MIN_BIN_BLOCKS",
        "num_gfmc_collect_steps": "GFMC_MIN_COLLECT_STEPS",
    },
}
ANALYSIS_TOML_SECTIONS = {"vmc": "mcmc", "mcmc": "mcmc", "lrdmc-bra": "lrdmc", "lrdmc-tau": "lrdmc"}
ANALYSIS_COMMANDS = ("compute-energy", "compute-force", "extrapolate-energy")
ANALYSIS_COMMAND_PATTERN = re.compile(
    rf"^\s*(?:\$\s+|Running:\s+)?jqmc-tool\s+(mcmc|lrdmc)\s+({'|'.join(ANALYSIS_COMMANDS)})(?=\s|$)(.*)$"
)
FENCE_PATTERN = re.compile(r"^[ \t]*```[\w:.-]*[ \t]*\n([\s\S]*?)^[ \t]*```[ \t]*$", re.MULTILINE)
WORKFLOW_SOURCES = {
    "WF_Workflow": ROOT / "jqmc_workflow" / "wf_workflow.py",
    "QE_Workflow": ROOT / "jqmc_workflow" / "qe_workflow.py",
    "VMC_Workflow": ROOT / "jqmc_workflow" / "vmc_workflow.py",
    "MCMC_Workflow": ROOT / "jqmc_workflow" / "mcmc_workflow.py",
    "LRDMC_Workflow": ROOT / "jqmc_workflow" / "lrdmc_workflow.py",
    "LRDMC_Ext_Workflow": ROOT / "jqmc_workflow" / "lrdmc_ext_workflow.py",
}


def public_markdown_files() -> list[Path]:
    """Return Markdown rendered for users or used as its source."""
    files = [ROOT / "README.md", *EXAMPLES.rglob("README.md")]
    files.extend(ROOT.joinpath("doc").glob("*.md"))
    files.extend(ROOT.joinpath("doc", "notes").glob("*.md"))
    return sorted(set(files))


def load_cli_parameter_keys() -> dict[str, set[str]]:
    """Read the CLI parameter key tree without importing jQMC dependencies."""
    path = ROOT / "jqmc" / "jqmc_miscs.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "cli_parameters" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        result: dict[str, set[str]] = {}
        for section_node, values_node in zip(node.value.keys, node.value.values, strict=True):
            if not isinstance(section_node, ast.Constant) or not isinstance(section_node.value, str):
                continue
            if not isinstance(values_node, ast.Dict):
                continue
            result[section_node.value] = {
                key.value for key in values_node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
        return result
    raise RuntimeError(f"Cannot find cli_parameters in {path}")


def load_cli_parameter_defaults() -> dict[str, dict[str, object]]:
    """Read literal CLI defaults without importing jqmc dependencies."""
    path = ROOT / "jqmc" / "jqmc_miscs.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "cli_parameters" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        result: dict[str, dict[str, object]] = {}
        for section_node, values_node in zip(node.value.keys, node.value.values, strict=True):
            if (
                not isinstance(section_node, ast.Constant)
                or not isinstance(section_node.value, str)
                or not isinstance(values_node, ast.Dict)
            ):
                continue
            values: dict[str, object] = {}
            for key, value in zip(values_node.keys, values_node.values, strict=True):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, TypeError):
                    continue
                if not isinstance(literal, dict):
                    values[key.value] = literal
            result[section_node.value] = values
        return result
    raise RuntimeError(f"Cannot find cli_parameters in {path}")


def documented_default_matches(cell: str, expected: object) -> bool:
    """Compare a Markdown Default cell with a literal Python default."""
    normalized = cell.replace("**", "").strip().strip("`").strip()
    if expected is None:
        return normalized.lower() == "required"
    if isinstance(expected, bool):
        return normalized.lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return float(normalized) == float(expected)
        except ValueError:
            return False
    if isinstance(expected, str):
        try:
            parsed = ast.literal_eval(normalized)
        except (ValueError, SyntaxError):
            parsed = normalized
        return parsed == expected
    return True


def load_restart_frozen_keys() -> dict[str, set[str]]:
    """Read frozen restart keys from cli_parameters without importing jQMC."""
    path = ROOT / "jqmc" / "jqmc_miscs.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "cli_parameters" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        result: dict[str, set[str]] = {}
        for section_node, values_node in zip(node.value.keys, node.value.values, strict=True):
            if (
                not isinstance(section_node, ast.Constant)
                or not isinstance(section_node.value, str)
                or not section_node.value.endswith("_restart_policy")
                or not isinstance(values_node, ast.Dict)
            ):
                continue
            section = section_node.value.removesuffix("_restart_policy")
            result[section] = {
                key.value
                for key, value in zip(values_node.keys, values_node.values, strict=True)
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and value.value == "frozen"
            }
        return result
    raise RuntimeError(f"Cannot find cli_parameters in {path}")


def optimizer_keyword_contract() -> set[str]:
    """Extract optimizer_kwargs keys consumed by production run_optimize."""
    path = ROOT / "jqmc" / "jqmc_optimizer.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "run_optimize":
            continue
        return {
            call.args[0].value
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "optimizer_kwargs"
            and call.func.attr in {"get", "pop"}
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        }
    raise RuntimeError(f"Cannot find run_optimize in {path}")


def load_precision_zone_maps() -> dict[str, dict[str, str]]:
    """Read the full/mixed precision maps without importing JAX."""
    path = ROOT / "jqmc" / "_precision.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    result: dict[str, dict[str, str]] = {}
    target_names = {"_FULL_PRECISION": "full", "_MIXED_PRECISION": "mixed"}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        label = target_names.get(node.target.id)
        if label is None or not isinstance(node.value, ast.Dict):
            continue
        result[label] = {
            key.value: value.value
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        }
    if set(result) != {"full", "mixed"}:
        raise RuntimeError(f"Cannot find full/mixed precision maps in {path}")
    return result


def check_toml_files(errors: list[str]) -> None:
    """Reject syntax errors, unknown keys, and mismatched job sections."""
    cli_parameter_keys = load_cli_parameter_keys()
    for path in sorted(EXAMPLES.rglob("*.toml")):
        rel = path.relative_to(ROOT)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"{rel}: invalid TOML: {exc}")
            continue

        for section, values in data.items():
            if not isinstance(values, dict):
                errors.append(f"{rel}: top-level {section!r} is not a TOML table")
                continue
            if section == "precision":
                allowed = {"mode"}
            elif section in cli_parameter_keys:
                allowed = cli_parameter_keys[section]
            else:
                errors.append(f"{rel}: unknown section [{section}]")
                continue
            unknown = sorted(set(values) - allowed)
            if unknown:
                errors.append(f"{rel}: [{section}] has unknown keys: {unknown}")

        control = data.get("control", {})
        job_type = control.get("job_type") if isinstance(control, dict) else None
        if job_type and job_type not in data:
            errors.append(f"{rel}: job_type={job_type!r} but [{job_type}] is absent")

        vmc = data.get("vmc", {})
        optimizer_kwargs = vmc.get("optimizer_kwargs") if isinstance(vmc, dict) else None
        if isinstance(optimizer_kwargs, dict):
            allowed_optimizer_keys = optimizer_keyword_contract()
            if unknown := sorted(set(optimizer_kwargs) - allowed_optimizer_keys):
                errors.append(f"{rel}: [vmc].optimizer_kwargs has unsupported keys: {unknown}")
            method = optimizer_kwargs.get("method", "sr")
            if not isinstance(method, str) or method.lower() != "sr":
                errors.append(f"{rel}: [vmc].optimizer_kwargs.method must be 'sr', got {method!r}")


def load_analysis_minima() -> dict[str, int]:
    """Read the central analysis minima from jqmc/_setting.py without importing jQMC."""
    path = ROOT / "jqmc" / "_setting.py"
    names = {name for family in ANALYSIS_MINIMA.values() for name in family.values()}
    minima = {
        target.id: node.value.value
        for node in ast.parse(path.read_text("utf-8"), filename=str(path)).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    if missing := sorted(names - set(minima)):
        raise RuntimeError(f"Cannot find integer {missing} in {path}")
    return minima


def load_analysis_tool_options() -> dict[tuple[str, str], dict[str, tuple[str, object]]]:
    """Map each mcmc/lrdmc analysis command's flags to (long option name, literal default)."""
    path = ROOT / "jqmc" / "jqmc_tool.py"
    result: dict[tuple[str, str], dict[str, tuple[str, object]]] = {}
    for node in ast.parse(path.read_text("utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "command"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id in ("mcmc_app", "lrdmc_app")
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and decorator.args[0].value in ANALYSIS_COMMANDS
            ):
                continue
            options: dict[str, tuple[str, object]] = {}
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if isinstance(default, ast.Call) and isinstance(default.func, ast.Attribute) and default.func.attr == "Option":
                    flags = [
                        arg.value for arg in default.args[1:] if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    ]
                    long_name = next(flag[2:] for flag in flags if flag.startswith("--"))
                    options.update({flag: (long_name, ast.literal_eval(default.args[0])) for flag in flags})
            result[(decorator.func.value.id.removesuffix("_app"), decorator.args[0].value)] = options
    return result


def published_analysis_commands() -> list[tuple[str, str, str, list[str]]]:
    """Return (location, group, command, arguments) for each analysis command in the example READMEs.

    Every fenced block is scanned, backslash continuations are joined, an optional ``$`` prompt or
    driver-log ``Running:`` prefix is accepted, and shell redirections/pipes end the arguments.
    Lines with a ``->`` arrow are workflow diagrams, not commands.
    """
    commands = []
    for path in sorted(EXAMPLES.glob("*/README.md")):
        content = path.read_text("utf-8")
        for block in FENCE_PATTERN.finditer(content):
            first_line = content.count("\n", 0, block.start(1)) + 1
            logical, start = "", None
            for offset, line in enumerate(block.group(1).splitlines()):
                start = first_line + offset if start is None else start
                if line.endswith("\\"):
                    logical += line[:-1] + " "
                    continue
                match = ANALYSIS_COMMAND_PATTERN.match(logical + line)
                if match and "->" not in match.group(3):
                    words = re.split(r"\s(?:[0-9]?>|\||&&|;)", match.group(3), maxsplit=1)[0]
                    location = f"{path.relative_to(ROOT)}:{start}"
                    commands.append((location, match.group(1), match.group(2), shlex.split(words, comments=True)))
                logical, start = "", None
    return commands


def analysis_command_settings(options: dict[str, tuple[str, object]], arguments: list[str]) -> dict[str, object]:
    """Resolve option values of one analysis command, falling back to the tool defaults."""
    settings = {name: default for name, default in options.values()}
    words = iter(arguments)
    for word in words:
        flag, has_value, inline = word.partition("=") if word.startswith("--") else (word, "", "")
        if flag not in options:
            if word.startswith("-"):
                raise ValueError(f"unknown option {word!r}")
            continue  # positional checkpoint path
        name, default = options[flag]
        if isinstance(default, bool):
            settings[name] = True
            continue
        raw = inline if has_value else next(words, None)
        if raw is None:
            raise ValueError(f"option {flag!r} needs a value")
        settings[name] = raw if default is None else type(default)(raw)
    return settings


def analysis_minimum_errors(where: str, family: str, values: dict, minima: dict[str, int]) -> list[str]:
    """Report each warmup/bin/collect value that is missing or below its central minimum."""
    return [
        f"{where}: {key} = {values.get(key)!r} must be >= {name} = {minima[name]}"
        for key, name in ANALYSIS_MINIMA[family].items()
        if not (isinstance(values.get(key), int) and values[key] >= minima[name])
    ]


def check_analysis_minima(errors: list[str]) -> None:
    """Require shipped example inputs and README analysis commands to meet the central minima."""
    minima = load_analysis_minima()
    for path in sorted(EXAMPLES.rglob("*.toml")):
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError:
            continue  # reported by check_toml_files
        for section, family in ANALYSIS_TOML_SECTIONS.items():
            if isinstance(values := data.get(section), dict):
                errors.extend(analysis_minimum_errors(f"{path.relative_to(ROOT)}: [{section}]", family, values, minima))
    tool_options = load_analysis_tool_options()
    for location, group, command, arguments in published_analysis_commands():
        where = f"{location}: jqmc-tool {group} {command}"
        if (group, command) not in tool_options:
            errors.append(f"{where}: unknown analysis command")
            continue
        try:
            settings = analysis_command_settings(tool_options[(group, command)], arguments)
        except ValueError as exc:
            errors.append(f"{where}: {exc}")
            continue
        errors.extend(analysis_minimum_errors(where, group, settings, minima))


def check_feature_tutorials(errors: list[str]) -> None:
    """Keep the flagship feature examples enabled rather than comment-only."""

    expected = {
        "examples/jqmc-example07/05mcmc_JSD_mixed_mpc/mcmc.toml": {
            ("control", "job_type"): "mcmc",
            ("coulomb", "mpc"): True,
            ("precision", "mode"): "mixed",
        },
        "examples/jqmc-example09/02mcmc_twist_stress/mcmc.toml": {
            ("control", "job_type"): "mcmc",
            ("mcmc", "compute_stress"): True,
            ("coulomb", "mpc"): True,
            ("precision", "mode"): "mixed",
        },
        "examples/jqmc-example09/03lrdmc_twist_stress/lrdmc_tau.toml": {
            ("control", "job_type"): "lrdmc-tau",
            ("lrdmc-tau", "compute_stress"): True,
            ("coulomb", "mpc"): True,
            ("precision", "mode"): "mixed",
        },
        "examples/jqmc-example10/02vmc_open_shell/vmc.toml": {
            ("control", "job_type"): "vmc",
            ("vmc", "opt_geminal"): True,
        },
        "examples/jqmc-example10/03mcmc_open_shell/mcmc.toml": {
            ("control", "job_type"): "mcmc",
            ("precision", "mode"): "mixed",
        },
        "examples/jqmc-example10/04QE_LSDA/mcmc.toml": {
            ("control", "job_type"): "mcmc",
            ("precision", "mode"): "mixed",
        },
    }
    for rel_text, fields in expected.items():
        path = ROOT / rel_text
        if not path.is_file():
            errors.append(f"{rel_text}: required feature tutorial is missing")
            continue
        data = tomllib.loads(path.read_text("utf-8"))
        for (section, key), wanted in fields.items():
            actual = data.get(section, {}).get(key)
            if actual != wanted:
                errors.append(
                    f"{rel_text}: [{section}].{key} must be {wanted!r} for non-vacuous feature coverage, got {actual!r}"
                )

    mpc_paths = (
        ROOT / "examples/jqmc-example07/05mcmc_JSD_mixed_mpc/mcmc.toml",
        ROOT / "examples/jqmc-example09/02mcmc_twist_stress/mcmc.toml",
        ROOT / "examples/jqmc-example09/03lrdmc_twist_stress/lrdmc_tau.toml",
    )
    for mpc_path in mpc_paths:
        if mpc_path.is_file():
            mpc_data = tomllib.loads(mpc_path.read_text("utf-8"))
            cutoff = mpc_data.get("coulomb", {}).get("mpc_cutoff")
            if not isinstance(cutoff, (int, float)) or cutoff <= 0:
                errors.append(f"{mpc_path.relative_to(ROOT)}: [coulomb].mpc_cutoff must be positive")

    open_shell_source = ROOT / "examples/jqmc-example10/01SCF/run_pyscf.py"
    if not open_shell_source.is_file():
        errors.append(f"{open_shell_source.relative_to(ROOT)}: required open-shell source is missing")
    else:
        source = open_shell_source.read_text("utf-8")
        for needle in ("mol.spin = 3", "scf.UHF(mol)"):
            if needle not in source:
                errors.append(f"{open_shell_source.relative_to(ROOT)}: open-shell tutorial must contain {needle!r}")

    lsda_source = ROOT / "examples/jqmc-example10/04QE_LSDA/si_atom_lsda.in"
    if not lsda_source.is_file():
        errors.append(f"{lsda_source.relative_to(ROOT)}: required LSDA source is missing")
    else:
        source = lsda_source.read_text("utf-8")
        for needle in ("nspin         = 2", "tot_magnetization = 2.0", "occupations   = 'fixed'"):
            if needle not in source:
                errors.append(f"{lsda_source.relative_to(ROOT)}: LSDA tutorial must contain {needle!r}")

    twist_workflow_source = ROOT / "examples/jqmc-workflow-example06/run_twist_pipeline.py"
    if not twist_workflow_source.is_file():
        errors.append(f"{twist_workflow_source.relative_to(ROOT)}: required multi-twist workflow tutorial is missing")
    else:
        source = twist_workflow_source.read_text("utf-8")
        for needle in (
            "twist_average=True",
            "opt_geminal=True",
            'PRECISION_MODE = "mixed"',
            'FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))',
            "if n_twist_published != N_TWIST:",
        ):
            if needle not in source:
                errors.append(
                    f"{twist_workflow_source.relative_to(ROOT)}: multi-twist workflow tutorial must contain {needle!r}"
                )

    eos_workflow_source = ROOT / "examples/jqmc-workflow-example04/run_pipelines.py"
    if not eos_workflow_source.is_file():
        errors.append(f"{eos_workflow_source.relative_to(ROOT)}: required solid-state workflow tutorial is missing")
    else:
        source = eos_workflow_source.read_text("utf-8")
        for needle in ('kind="bspline"', "twist_average=True", "LRDMC_Workflow"):
            if needle not in source:
                errors.append(f"{eos_workflow_source.relative_to(ROOT)}: solid-state workflow tutorial must contain {needle!r}")

    index = (EXAMPLES / "README.md").read_text("utf-8")
    for feature in (
        "Twist averaging",
        "Mixed precision",
        "Stress and pressure",
        "MPC",
        "Open-shell",
        "Multi-twist VMC optimization",
        "Equation of state",
        "jqmc-workflow-example04",
    ):
        if feature not in index:
            errors.append(f"examples/README.md: feature index is missing {feature!r}")


def check_cli_reference_keys(errors: list[str]) -> None:
    """Require every TOML input key to appear in the CLI reference table."""
    path = ROOT / "doc" / "api_reference_cli.md"
    content = path.read_text("utf-8")
    expected_by_section = load_cli_parameter_keys()
    defaults_by_section = load_cli_parameter_defaults()
    if "### `precision`" not in content or "| `mode`" not in content:
        errors.append(f"{path.relative_to(ROOT)}: missing [precision].mode documentation")
    for section in ("control", "mcmc", "vmc", "lrdmc-bra", "lrdmc-tau", "coulomb"):
        heading = f"### `{section}`"
        start = content.find(heading)
        if start < 0:
            errors.append(f"{path.relative_to(ROOT)}: missing {heading} section")
            continue
        lines = content[start:].splitlines()
        try:
            table_start = next(index for index, line in enumerate(lines) if line.startswith("| Key"))
        except StopIteration:
            errors.append(f"{path.relative_to(ROOT)}: {heading} has no key table")
            continue
        documented: set[str] = set()
        documented_defaults: dict[str, str] = {}
        for line in lines[table_start + 2 :]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and cells[0].startswith("`") and cells[0].endswith("`"):
                key = cells[0][1:-1]
                documented.add(key)
                if len(cells) > 1:
                    documented_defaults[key] = cells[1]
        expected = expected_by_section[section]
        if missing := sorted(expected - documented):
            errors.append(f"{path.relative_to(ROOT)}: {heading} is missing keys: {missing}")
        if unknown := sorted(documented - expected):
            errors.append(f"{path.relative_to(ROOT)}: {heading} documents unknown keys: {unknown}")
        for key, expected_default in defaults_by_section[section].items():
            cell = documented_defaults.get(key)
            if cell is not None and not documented_default_matches(cell, expected_default):
                errors.append(
                    f"{path.relative_to(ROOT)}: {heading} default for {key!r} differs: "
                    f"expected {expected_default!r}, documented {cell!r}"
                )

    restart_heading = "## Restart inheritance"
    restart_start = content.find(restart_heading)
    if restart_start < 0:
        errors.append(f"{path.relative_to(ROOT)}: missing restart-inheritance documentation")
        return
    restart_content = content[restart_start:]
    documented_frozen: dict[str, set[str]] = {}
    for match in re.finditer(r"^\| `([^`]+)` \| (.*?) \|$", restart_content, re.MULTILINE):
        documented_frozen[match.group(1)] = set(re.findall(r"`([^`]+)`", match.group(2)))
    for section, expected in load_restart_frozen_keys().items():
        documented = documented_frozen.get(section)
        if documented is None:
            errors.append(f"{path.relative_to(ROOT)}: restart table is missing section {section!r}")
        elif documented != expected:
            errors.append(
                f"{path.relative_to(ROOT)}: restart frozen keys for {section!r} differ: "
                f"expected {sorted(expected)}, documented {sorted(documented)}"
            )


def check_tool_reference_surface(errors: list[str]) -> None:
    """Require every jqmc-tool command and long option in its group section."""
    source_path = ROOT / "jqmc" / "jqmc_tool.py"
    doc_path = ROOT / "doc" / "api_reference_cli_tool.md"
    tree = ast.parse(source_path.read_text("utf-8"), filename=str(source_path))
    content = doc_path.read_text("utf-8")
    section_pattern = re.compile(r"^## `([^`]+)` group.*?$(.*?)(?=^## `|\Z)", re.MULTILINE | re.DOTALL)
    sections = {match.group(1): match.group(2) for match in section_pattern.finditer(content)}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        command: str | None = None
        group: str | None = None
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "command"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id.endswith("_app")
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                group = decorator.func.value.id.removesuffix("_app")
                command = decorator.args[0].value
                break
        if group is None or command is None:
            continue
        section = sections.get(group)
        if section is None:
            errors.append(f"{doc_path.relative_to(ROOT)}: missing jqmc-tool group {group!r}")
            continue
        # Typer collapses the sole qe subcommand into ``jqmc-tool qe``.
        if group != "qe" and f"### `{command}`" not in section:
            errors.append(f"{doc_path.relative_to(ROOT)}: group {group!r} is missing command {command!r}")

        defaults = [*node.args.defaults, *node.args.kw_defaults]
        long_options: set[str] = set()
        for default in defaults:
            if not (
                isinstance(default, ast.Call) and isinstance(default.func, ast.Attribute) and default.func.attr == "Option"
            ):
                continue
            long_options.update(
                arg.value
                for arg in default.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
            )
        for option in sorted(long_options):
            if option not in section:
                errors.append(f"{doc_path.relative_to(ROOT)}: jqmc-tool {group} is missing long option {option!r}")


def check_mixed_precision_zone_table(errors: list[str]) -> None:
    """Require the published zone table to match _precision.py exactly."""
    path = ROOT / "doc" / "notes" / "mixed_precision.md"
    documented: dict[str, dict[str, str]] = {"full": {}, "mixed": {}}
    for line in path.read_text("utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip().replace("*", "") for cell in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0].startswith("`") or not cells[0].endswith("`"):
            continue
        zone = cells[0][1:-1]
        dtype_aliases = {"f32": "float32", "f64": "float64"}
        if cells[2] in dtype_aliases and cells[3] in dtype_aliases:
            documented["full"][zone] = dtype_aliases[cells[2]]
            documented["mixed"][zone] = dtype_aliases[cells[3]]
    expected = load_precision_zone_maps()
    for mode in ("full", "mixed"):
        if documented[mode] != expected[mode]:
            missing = sorted(set(expected[mode]) - set(documented[mode]))
            unknown = sorted(set(documented[mode]) - set(expected[mode]))
            wrong = sorted(
                zone for zone in set(expected[mode]) & set(documented[mode]) if expected[mode][zone] != documented[mode][zone]
            )
            errors.append(
                f"{path.relative_to(ROOT)}: {mode} zone table drift: missing={missing}, unknown={unknown}, wrong_dtype={wrong}"
            )


def check_included_blocks(errors: list[str]) -> None:
    """Require README include blocks to equal their source files."""
    for readme in sorted(EXAMPLES.rglob("README.md")):
        content = readme.read_text("utf-8")
        for marker in INCLUDE_MARKER.finditer(content):
            rel_readme = readme.relative_to(ROOT)
            match = INCLUDE_PATTERN.match(content, marker.start())
            if match is None:
                errors.append(f"{rel_readme}: include {marker.group(1)} has no fenced block")
                continue
            suffix = Path(match.group(2).partition("#")[0]).suffix
            language = {".toml": "toml", ".py": "python", ".xyz": "text", ".inp": "text", ".cp2k": "text"}.get(suffix)
            if language and not match.group(3).startswith("```" + language):
                errors.append(f"{rel_readme}: include {marker.group(1)} must use a {language} fence")
            try:
                expected = included_text(readme, match.group(2).strip())
            except (OSError, ValueError) as exc:
                errors.append(f"{rel_readme}: invalid include: {exc}")
                continue
            actual = match.group(4).rstrip() + "\n"
            if actual != expected:
                errors.append(f"{rel_readme}: stale include {match.group(2).strip()}; run 'python examples/sync_examples.py'")


def source_signature(path: Path, class_name: str, method_name: str) -> tuple[set[str], bool]:
    """Read a class method signature without importing MPI-owning modules."""
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                args = child.args
                names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
                names.discard("self")
                return names, args.kwarg is not None
    raise RuntimeError(f"Cannot find {class_name}.{method_name} in {path}")


def callable_signatures() -> dict[str, tuple[set[str], bool]]:
    """Return constructor/method keyword contracts used by examples."""
    signatures = {
        "MCMC": source_signature(ROOT / "jqmc" / "jqmc_mcmc.py", "MCMC", "__init__"),
        "GFMC_n": source_signature(ROOT / "jqmc" / "jqmc_gfmc.py", "GFMC_n", "__init__"),
        "GFMC_t": source_signature(ROOT / "jqmc" / "jqmc_gfmc.py", "GFMC_t", "__init__"),
        "MCMC.run_optimize": source_signature(ROOT / "jqmc" / "jqmc_mcmc.py", "MCMC", "run_optimize"),
    }
    signatures.update({name: source_signature(path, name, "__init__") for name, path in WORKFLOW_SOURCES.items()})
    return signatures


def validate_tree_calls(
    tree: ast.AST,
    rel: Path,
    signatures: dict[str, tuple[set[str], bool]],
    errors: list[str],
    *,
    line_offset: int = 0,
) -> None:
    """Validate keyword calls in one parsed Python tree."""
    allowed_optimizer_keys = optimizer_keyword_contract()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "optimizer_kwargs" or not isinstance(keyword.value, ast.Dict):
                continue
            items = {
                key.value: value
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if unknown := sorted(set(items) - allowed_optimizer_keys):
                errors.append(f"{rel}:{line_offset + keyword.value.lineno}: optimizer_kwargs has unsupported keys: {unknown}")
            method = items.get("method")
            if isinstance(method, ast.Constant) and (not isinstance(method.value, str) or method.value.lower() != "sr"):
                errors.append(
                    f"{rel}:{line_offset + method.lineno}: optimizer_kwargs.method must be 'sr', got {method.value!r}"
                )
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "run_optimize":
            name = "MCMC.run_optimize"
        if name not in signatures:
            continue
        allowed, has_var_keyword = signatures[name]
        if has_var_keyword:
            continue
        unknown = sorted({kw.arg for kw in node.keywords if kw.arg is not None} - allowed)
        if unknown:
            errors.append(f"{rel}:{line_offset + node.lineno}: {name} has unknown keyword arguments: {unknown}")


def workflow_exports() -> set[str]:
    """Read jqmc_workflow.__all__ without importing package dependencies."""
    path = ROOT / "jqmc_workflow" / "__init__.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            return {item.value for item in node.value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)}
    raise RuntimeError(f"Cannot find __all__ in {path}")


def jqmc_exports() -> set[str]:
    """Read jqmc's lazy public export map without importing dependencies."""
    path = ROOT / "jqmc" / "__init__.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "_PUBLIC" for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise RuntimeError(f"Cannot find _PUBLIC in {path}")


def validate_public_imports(
    tree: ast.AST,
    rel: Path,
    errors: list[str],
    *,
    jqmc_public: set[str],
    workflow_public: set[str],
    line_offset: int = 0,
) -> None:
    """Keep user examples on the declared root-level public APIs."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.startswith("jqmc."):
            errors.append(
                f"{rel}:{line_offset + node.lineno}: public examples must import from 'jqmc', "
                f"not internal module '{node.module}'"
            )
        elif node.module == "jqmc":
            unknown = sorted(alias.name for alias in node.names if alias.name not in jqmc_public)
            if unknown:
                errors.append(f"{rel}:{line_offset + node.lineno}: jqmc has no public exports: {unknown}")
        elif node.module == "jqmc_workflow":
            unknown = sorted(alias.name for alias in node.names if alias.name not in workflow_public)
            if unknown:
                errors.append(f"{rel}:{line_offset + node.lineno}: jqmc_workflow has no public exports: {unknown}")


def check_python_examples(errors: list[str]) -> None:
    """Parse scripts and validate supported constructor keyword surfaces."""
    signatures = callable_signatures()
    jqmc_public = jqmc_exports()
    workflow_public = workflow_exports()

    for path in sorted(EXAMPLES.rglob("*.py")):
        rel = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{rel}: Python syntax error: {exc}")
            continue
        validate_public_imports(
            tree,
            rel,
            errors,
            jqmc_public=jqmc_public,
            workflow_public=workflow_public,
        )
        validate_tree_calls(tree, rel, signatures, errors)


def check_markdown_python_blocks(errors: list[str]) -> None:
    """Parse published Python snippets and validate workflow imports/calls."""
    signatures = callable_signatures()
    jqmc_public = jqmc_exports()
    workflow_public = workflow_exports()
    generated = {ROOT / "doc" / "examples.md"}
    for path in public_markdown_files():
        if path in generated:
            continue
        rel = path.relative_to(ROOT)
        content = path.read_text("utf-8")
        for match in PYTHON_FENCE_PATTERN.finditer(content):
            line_offset = content.count("\n", 0, match.start(1))
            try:
                # Included script regions may be indented (function bodies).
                tree = ast.parse(textwrap.dedent(match.group(1)), filename=f"{rel}:{line_offset + 1}")
            except SyntaxError as exc:
                errors.append(f"{rel}:{line_offset + (exc.lineno or 1)}: invalid Python snippet: {exc.msg}")
                continue
            validate_public_imports(
                tree,
                rel,
                errors,
                jqmc_public=jqmc_public,
                workflow_public=workflow_public,
                line_offset=line_offset,
            )
            validate_tree_calls(tree, rel, signatures, errors, line_offset=line_offset)


def check_local_links_and_commands(errors: list[str]) -> None:
    """Check local Markdown targets and commands known not to exist."""
    for path in public_markdown_files():
        rel = path.relative_to(ROOT)
        fenced = False
        for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            for command, replacement in BANNED_COMMANDS.items():
                if command in line:
                    errors.append(f"{rel}:{line_number}: unsupported '{command}'; {replacement}")
            if fenced:
                continue
            for raw_target in LINK_PATTERN.findall(line):
                target = raw_target.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
                if (
                    not target
                    or target.startswith(("http://", "https://", "mailto:", "data:"))
                    or any(char.isspace() for char in target)
                ):
                    continue
                if not (path.parent / target).resolve().exists():
                    errors.append(f"{rel}:{line_number}: missing local link target {raw_target!r}")
        if fenced:
            errors.append(f"{rel}: unclosed fenced code block")


def check_removed_public_surfaces(errors: list[str]) -> None:
    """Reject banned implementation phrases in current user guidance."""
    historical = {ROOT / "doc" / "changelog.md"}
    generated = {ROOT / "doc" / "examples.md"}
    for path in public_markdown_files():
        if path in historical or path in generated:
            continue
        rel = path.relative_to(ROOT)
        for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            for phrase, replacement in BANNED_PUBLIC_PHRASES.items():
                if phrase in line:
                    errors.append(f"{rel}:{line_number}: banned phrase {phrase!r}; {replacement}")


def unsupported_docstring_args(path: Path) -> list[str]:
    """Check class Args sections against actual constructors, without imports."""
    errors = []
    for cls in ast.parse(path.read_text("utf-8")).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        constructor = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if constructor is None or constructor.args.kwarg:
            continue
        doc = ast.get_docstring(cls, clean=False) or ""
        if "Args:" not in doc:
            continue
        section = re.split(r"\n    [A-Za-z][A-Za-z ]*:\s*\n", doc.split("Args:", 1)[1])[0]
        documented = set(re.findall(r"^        (\w+)\s*(?:\([^\n]+?\))?:", section, re.MULTILINE))
        args = constructor.args
        actual = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        if unknown := sorted(documented - actual):
            errors.append(f"{path.name}:{cls.lineno}: {cls.name} documents unsupported constructor arguments {unknown}")
    return errors


def workflow_handoff_errors(tree: ast.AST) -> list[str]:
    """Downstream samplers must use the VMC-published artifact and its rename."""
    errors = []
    for container in ast.walk(tree):
        if not isinstance(container, ast.Call) or not isinstance(container.func, ast.Name) or container.func.id != "Container":
            continue
        kwargs = {item.arg: item.value for item in container.keywords}
        workflow = kwargs.get("workflow")
        if not isinstance(workflow, ast.Call) or not isinstance(workflow.func, ast.Name):
            continue
        if workflow.func.id not in {"MCMC_Workflow", "LRDMC_Workflow", "LRDMC_Ext_Workflow"}:
            continue
        inputs = kwargs.get("input_files")
        if inputs is None:
            continue
        for node in ast.walk(inputs):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "FileFrom":
                continue
            if not node.args or "vmc" not in ast.unparse(node.args[0]).lower():
                continue
            value = node.args[1] if len(node.args) == 2 else None
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "ValueFrom"
                and len(value.args) == 2
                and ast.dump(value.args[0]) == ast.dump(node.args[0])
                and isinstance(value.args[1], ast.Constant)
                and value.args[1].value == "optimized_hamiltonian"
            ):
                errors.append(f"line {node.lineno}: sampler must use the VMC-published optimized_hamiltonian")
            rename = kwargs.get("rename_input_files")
            if not isinstance(inputs, ast.List) or len(inputs.elts) != 1:
                errors.append(f"line {container.lineno}: stage only the optimized trial, not an original trial beside it")
            if not isinstance(rename, ast.List) or ast.literal_eval(rename) != ["hamiltonian_data.h5"]:
                errors.append(f"line {container.lineno}: rename the optimized trial to hamiltonian_data.h5")
    return errors


def vmc_workflow_defaults() -> dict[str, object]:
    """Literal constructor defaults of ``VMC_Workflow``, read from its source."""
    tree = ast.parse(WORKFLOW_SOURCES["VMC_Workflow"].read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args.args[len(node.args.args) - len(node.args.defaults) :]
            return {
                arg.arg: default.value
                for arg, default in zip(args, node.args.defaults, strict=True)
                if isinstance(default, ast.Constant)
            }
    raise RuntimeError("Cannot find VMC_Workflow.__init__")


def workflow_vmc_average_errors(tree: ast.AST, defaults: dict[str, object] | None = None) -> list[str]:
    """Tutorial VMC stages hand the average of the last steps downstream and dump every step of it."""
    defaults = vmc_workflow_defaults() if defaults is None else defaults
    constants = {
        target.id: node.value.value
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    errors = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "VMC_Workflow"):
            continue
        kwargs = {item.arg: item.value for item in call.keywords}
        unresolved = object()

        def value(key, kwargs=kwargs, unresolved=unresolved):
            node = kwargs.get(key)
            if node is None:
                return defaults.get(key, unresolved)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name) and node.id in constants:
                return constants[node.id]
            return unresolved

        settings = {
            key: value(key)
            for key in ("handoff_mode", "handoff_average_steps", "handoff_allow_fewer", "num_opt_steps", "max_continuation")
        }
        if any(item is unresolved for item in settings.values()):
            errors.append(f"line {call.lineno}: VMC_Workflow hand-off settings must be literals or module constants")
            continue
        if settings["handoff_mode"] != "average":
            errors.append(f"line {call.lineno}: hand the average of the last optimization steps downstream")
        if "wf_dump_freq" in kwargs and value("wf_dump_freq") != 1:
            errors.append(f"line {call.lineno}: dump every optimization step (wf_dump_freq = 1) for the averaged hand-off")
        total = settings["num_opt_steps"] * settings["max_continuation"]
        if total < settings["handoff_average_steps"] and not settings["handoff_allow_fewer"]:
            errors.append(
                f"line {call.lineno}: {total} optimization steps cannot give the {settings['handoff_average_steps']}-step average"
            )
    return errors


def check_release_contracts(errors: list[str]) -> None:
    """Validate executable shell syntax, artifact routing and autodoc signatures."""
    for path in sorted((ROOT / "jqmc_workflow").glob("*.py")):
        errors.extend(unsupported_docstring_args(path))
    for path in sorted(EXAMPLES.glob("jqmc-workflow-example*/run*.py")):
        tree = ast.parse(path.read_text("utf-8"))
        errors.extend(f"{path.relative_to(ROOT)}: {error}" for error in workflow_handoff_errors(tree))
        errors.extend(f"{path.relative_to(ROOT)}: {error}" for error in workflow_vmc_average_errors(tree))
    for path in public_markdown_files():
        if path == ROOT / "doc/examples.md":
            continue
        content = path.read_text("utf-8")
        for match in re.finditer(r"^```(?:bash|sh|shell)\s*\n([\s\S]*?)^```", content, re.MULTILINE):
            body = match.group(1)
            line = content.count("\n", 0, match.start()) + 1
            if re.search(r"^% ", body, re.MULTILINE):
                errors.append(f"{path.relative_to(ROOT)}:{line}: remove prompts from copyable command blocks")
            result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
            if result.returncode:
                errors.append(f"{path.relative_to(ROOT)}:{line}: invalid shell block: {result.stderr.strip()}")


def main() -> int:
    """Run all static documentation/example checks."""
    errors: list[str] = []
    check_toml_files(errors)
    check_analysis_minima(errors)
    check_feature_tutorials(errors)
    check_cli_reference_keys(errors)
    check_tool_reference_surface(errors)
    check_mixed_precision_zone_table(errors)
    check_included_blocks(errors)
    check_python_examples(errors)
    check_markdown_python_blocks(errors)
    check_local_links_and_commands(errors)
    check_removed_public_surfaces(errors)
    check_release_contracts(errors)
    from scripts.check_example_paths import check_all

    errors.extend(check_all())
    if errors:
        print("\n".join(errors))
        print(f"docs/examples static checks failed with {len(errors)} error(s)")
        return 1
    print("docs/examples static checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
