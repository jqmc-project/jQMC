"""Release-documentation contracts; calculators/schedulers are not launched."""

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
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from examples.copy_optimized_hamiltonian import average_dumps, copy_dump, last_dumps, latest_dump
from examples.sync_examples import PROJECT_HEADER, included_text
from scripts import check_docs_examples as checks
from scripts.check_docs_html import check_html

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def test_dump_selection_numeric_partial_and_continued(tmp_path):
    """Select the latest numeric optimization step and refuse missing or overwritten dumps."""
    (tmp_path / "hamiltonian_data.h5").write_text("original")
    with pytest.raises(FileNotFoundError):
        latest_dump(tmp_path)
    for step in (9, 91, 100, 200):
        (tmp_path / f"hamiltonian_data_opt_step_{step}.h5").write_text(f"optimized-{step}")
        assert latest_dump(tmp_path).name == f"hamiltonian_data_opt_step_{step}.h5"
    target = tmp_path / "next.h5"
    copy_dump(tmp_path, target)
    assert target.read_text() == "optimized-200"
    assert latest_dump(tmp_path, step=91).name == "hamiltonian_data_opt_step_91.h5"
    with pytest.raises(FileExistsError):
        copy_dump(tmp_path, target)
    assert target.read_text() == "optimized-200"
    with pytest.raises(FileNotFoundError):
        latest_dump(tmp_path, step=150)


def test_average_last_selects_the_numeric_window_and_refuses_short_or_existing(tmp_path):
    """--average-last N passes the N largest numeric steps (ending at --step) to jqmc-tool vmc average-wf."""
    for step in (9, 91, 100, 101, 102, 200):
        (tmp_path / f"hamiltonian_data_opt_step_{step}.h5").write_text(f"optimized-{step}")
    (tmp_path / "hamiltonian_data.h5").write_text("original")
    calls = []

    def run(command, returncode=0):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    target = tmp_path / "next.h5"
    sources = average_dumps(tmp_path, target, 3, run=run)
    assert [path.name for path in sources] == [f"hamiltonian_data_opt_step_{step}.h5" for step in (101, 102, 200)]
    assert calls[-1][-5:] == [*map(str, sources), "-o", str(target)]
    assert calls[-1][1:5] == ["-m", "jqmc.jqmc_tool", "vmc", "average-wf"]
    windowed = average_dumps(tmp_path, target, 4, step=101, run=run)
    assert [int(path.stem.rsplit("_", 1)[1]) for path in windowed] == [9, 91, 100, 101]
    with pytest.raises(ValueError, match="only 6 dumps"):
        average_dumps(tmp_path, target, 20, run=run)
    assert len(average_dumps(tmp_path, target, 20, allow_fewer=True, run=run)) == 6
    with pytest.raises(FileNotFoundError):
        last_dumps(tmp_path, 2, step=150)
    with pytest.raises(ValueError, match="positive"):
        last_dumps(tmp_path, 0)
    with pytest.raises(ValueError, match="refused"):
        average_dumps(tmp_path, target, 2, run=lambda command: run(command, returncode=1))
    target.write_text("existing downstream input")
    count = len(calls)
    with pytest.raises(FileExistsError):
        average_dumps(tmp_path, target, 2, run=run)
    assert len(calls) == count and target.read_text() == "existing downstream input"


@pytest.mark.parametrize(
    "frequency,steps,restart_steps,average,accepted",
    [(1, 150, 0, 20, True), (10, 150, 0, 20, False), (10, 150, 0, 15, True), (1, 10, 15, 20, True), (1, 10, 5, 20, False)],
)
def test_linear_staging_requires_enough_dumps_for_average_last(tmp_path, frequency, steps, restart_steps, average, accepted):
    """The staging model writes one dump per wf_dump_freq steps, continues numbering on restart and counts them."""
    from scripts.check_example_paths import trace_paths

    (tmp_path / "copy_optimized_hamiltonian.py").write_text("")
    example = tmp_path / "example"
    vmc, mcmc = example / "02vmc", example / "03mcmc"
    vmc.mkdir(parents=True)
    mcmc.mkdir()
    (vmc / "hamiltonian_data.h5").write_text("")
    deck = '[control]\njob_type = "vmc"\nrestart = {restart}\n[vmc]\nnum_opt_steps = {steps}\nwf_dump_freq = {frequency}\n'
    (vmc / "vmc.toml").write_text(deck.format(restart="false", steps=steps, frequency=frequency))
    (vmc / "vmc_cont.toml").write_text(deck.format(restart="true", steps=restart_steps, frequency=frequency))
    continuation = "jqmc vmc_cont.toml > out_vmc_cont\n" if restart_steps else ""
    readme = example / "README.md"
    readme.write_text(
        "```bash\ncd 02vmc\njqmc vmc.toml > out_vmc\n"
        f"{continuation}"
        f"cd ../03mcmc\npython ../../copy_optimized_hamiltonian.py --average-last {average} ../02vmc hamiltonian_data.h5\n```\n"
    )
    handoffs = []
    errors = trace_paths(readme, handoffs)
    if accepted:
        assert errors == [] and handoffs == [((mcmc / "hamiltonian_data.h5").resolve(), {"--average-last": average})]
    else:
        assert len(errors) == 1 and f"--average-last {average} but only" in errors[0], errors


def test_published_sampling_handoffs_average_the_last_20_steps():
    """Every tutorial hand-off into an MCMC/LRDMC directory averages the last 20 dumps, with enough dumps."""
    from scripts.check_example_paths import LINEAR_EXAMPLES, trace_paths

    sampling = 0
    for name in LINEAR_EXAMPLES:
        handoffs = []
        assert trace_paths(ROOT / f"examples/jqmc-example{name}/README.md", handoffs) == []
        for destination, options in handoffs:
            if any(destination.parent.glob("mcmc*.toml")) or any(destination.parent.glob("lrdmc*.toml")):
                assert options.get("--average-last") == 20 and "--allow-fewer" not in options, (name, destination)
                sampling += 1
            else:  # a start of a further optimization or conversion copies one dump
                assert "--average-last" not in options, (name, destination)
    assert sampling == 16
    for script in ("jqmc-example12/run_jqmc.py", "jqmc-example13/run_jqmc_mpi.py"):
        source = (ROOT / "examples" / script).read_text()
        assert '"--average-last", str(num_averaged_steps)' in source and "num_averaged_steps = 20" in source
        assert "wf_dump_freq=1," in source
        assert "optimized_hamiltonian = Hamiltonian_data.load_from_hdf5(averaged_file)" in source
        assert "optimized_hamiltonian = mcmc_opt.hamiltonian_data" not in source


@pytest.mark.parametrize("block", ["", "\nSome prose, not a code block.\n"])
def test_dangling_include_is_rejected(tmp_path, monkeypatch, block):
    """Reject documentation includes without a corresponding fenced code block."""
    (tmp_path / "input.toml").write_text('mode = "full"\n')
    (tmp_path / "README.md").write_text("<!-- include: input.toml -->\n" + block)
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "EXAMPLES", tmp_path)
    errors = []
    checks.check_included_blocks(errors)
    assert errors and "no fenced block" in errors[0]


def test_include_with_blank_line_and_named_region(tmp_path, monkeypatch):
    """Resolve a named source region across blank lines and detect stale included code."""
    source = "# docs: configuration-begin\nSTEPS = 200\n# docs: configuration-end\n"
    (tmp_path / "run.py").write_text(source)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- include: run.py#configuration -->\n\n```python\nSTEPS = 200\n```\n")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "EXAMPLES", tmp_path)
    errors = []
    checks.check_included_blocks(errors)
    assert not errors
    assert included_text(readme, "run.py#configuration") == "STEPS = 200\n"
    (tmp_path / "run.py").write_text(source.replace("200", "300"))
    checks.check_included_blocks(errors)
    assert errors and "stale include" in errors[0]


def test_include_omits_only_the_project_header(tmp_path, monkeypatch):
    """A whole-file listing omits the project licence header and must equal the rest of the file."""
    body = "import numpy as np\n\nSTEPS = 200\n"
    (tmp_path / "run.py").write_text('"""Run."""\n\n' + PROJECT_HEADER + "\n" + body)
    readme = tmp_path / "README.md"
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "EXAMPLES", tmp_path)
    for listing, stale in (
        ('"""Run."""\n\n' + body, False),
        ('"""Run."""\n\n' + PROJECT_HEADER + "\n" + body, True),
        ('"""Run."""\n\n' + body.replace("200", "300"), True),
    ):
        readme.write_text(f"<!-- include: run.py -->\n```python\n{listing}```\n")
        errors = []
        checks.check_included_blocks(errors)
        assert bool(errors) == stale and all("stale include" in error for error in errors)


def test_constructor_docstring_gate_is_nonvacuous(tmp_path):
    """Reject documented constructor arguments absent from the actual interface."""
    path = tmp_path / "workflow.py"
    path.write_text(
        'class Conversion:\n    """Local conversion.\n\n    Args:\n'
        '        server (str): Not a constructor parameter.\n    """\n'
        "    def __init__(self, input_file):\n        pass\n"
    )
    assert "server" in " ".join(checks.unsupported_docstring_args(path))
    assert not checks.unsupported_docstring_args(ROOT / "jqmc_workflow/qe_workflow.py")


@pytest.mark.parametrize("name", ["01", "02", "03", "05", "06"])
def test_workflow_handoff_gate_rejects_fixed_step_and_missing_rename(name):
    """Published workflows select the latest optimized Hamiltonian and rename it for downstream input."""
    script = next((ROOT / f"examples/jqmc-workflow-example{name}").glob("run*py"))
    source = script.read_text()
    tree = ast.parse(source)
    assert not checks.workflow_handoff_errors(tree)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "FileFrom"
        and len(n.args) == 2
        and isinstance(n.args[1], ast.Call)
    )
    node.args[1] = ast.Constant("hamiltonian_data_opt_step_100.h5")
    assert checks.workflow_handoff_errors(tree)
    missing_rename = source.replace('rename_input_files=["hamiltonian_data.h5"]', "rename_input_files=[]")
    assert checks.workflow_handoff_errors(ast.parse(missing_rename))


@pytest.mark.parametrize("name", ["01", "02", "03", "04", "05", "06"])
def test_workflow_examples_hand_on_the_average_of_every_dumped_last_step(name):
    """Published VMC stages keep the averaged hand-off, dump every step and run enough steps for it."""
    script = next((ROOT / f"examples/jqmc-workflow-example{name}").glob("run*py"))
    source = script.read_text()
    defaults = checks.vmc_workflow_defaults()
    assert (defaults["handoff_mode"], defaults["handoff_average_steps"], defaults["handoff_allow_fewer"]) == (
        "average",
        20,
        False,
    )
    assert not checks.workflow_vmc_average_errors(ast.parse(source))

    sparse = re.sub(r"^WF_DUMP_FREQ = 1\b", "WF_DUMP_FREQ = 10", source, count=1, flags=re.MULTILINE)
    assert sparse != source
    assert any("wf_dump_freq = 1" in error for error in checks.workflow_vmc_average_errors(ast.parse(sparse)))

    tree = ast.parse(source)
    vmc = next(
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "VMC_Workflow"
    )
    vmc.keywords.append(ast.keyword(arg="handoff_mode", value=ast.Constant("last")))
    assert any("average of the last" in error for error in checks.workflow_vmc_average_errors(tree))

    short = {**defaults, "handoff_average_steps": 10**6}
    assert any("cannot give the" in error for error in checks.workflow_vmc_average_errors(ast.parse(source), short))


@pytest.mark.parametrize("steps", [(9, 91), (20, 100), (20, 100, 200)])
def test_published_vmc_dump_reaches_actual_container_staging(tmp_path, monkeypatch, steps):
    """The container stages the latest optimized dump under the required Hamiltonian filename."""
    from jqmc_workflow import Container, FileFrom, Launcher, ValueFrom, VMC_Workflow, Workflow

    monkeypatch.chdir(tmp_path)
    vmc_dir = tmp_path / "vmc"
    vmc_dir.mkdir()
    (vmc_dir / "hamiltonian_data.h5").write_text("unoptimized")
    for step in steps:
        (vmc_dir / f"hamiltonian_data_opt_step_{step}.h5").write_text(f"optimized-{step}")
    publisher = VMC_Workflow.__new__(VMC_Workflow)
    published = publisher._latest_optimized_hamiltonian(str(vmc_dir))
    upstream = Container(label="vmc", dirname="vmc", workflow=Workflow())
    upstream.output_values["optimized_hamiltonian"] = published
    downstream = Container(
        label="sample",
        dirname="sample",
        workflow=Workflow(),
        input_files=[FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian"))],
        rename_input_files=["hamiltonian_data.h5"],
    )
    launcher = Launcher(workflows=[upstream, downstream], log_level="WARNING", draw_graph=False)
    launcher._resolve_variables(downstream)
    downstream._prepare()
    consumed = Path(downstream.project_dir) / "hamiltonian_data.h5"
    assert consumed.read_text() == f"optimized-{max(steps)}"
    assert sorted(p.name for p in consumed.parent.glob("*.h5")) == ["hamiltonian_data.h5"]


@pytest.mark.parametrize("name,directory", [("06", "03mcmc_JSD"), ("07", "03mcmc_JSD"), ("10", "03mcmc_open_shell")])
def test_published_analysis_commands_one_rank_one_walker(name, directory, monkeypatch):
    """Published analysis commands discard synthetic warmup samples and report the expected finite energy."""
    import tomllib
    from click.testing import CliRunner
    from jqmc import jqmc_tool

    example = ROOT / f"examples/jqmc-example{name}"
    settings = tomllib.loads((example / directory / "mcmc.toml").read_text())["mcmc"]
    count = settings["num_mcmc_steps"]
    warmup = settings["num_mcmc_warmup_steps"]
    bins = settings["num_mcmc_bin_blocks"]
    energy = np.full((count, 1), 3.0)
    energy[:warmup] = -1000.0  # Fails if the displayed command loses warmup.
    monkeypatch.setattr(
        jqmc_tool, "load_observables_from_checkpoint", lambda *a, **k: [{"e_L": energy, "w_L": np.ones_like(energy)}]
    )
    command = next(
        line for line in (example / "README.md").read_text().splitlines() if line.startswith("jqmc-tool mcmc compute-energy ")
    )
    result = CliRunner().invoke(jqmc_tool._cli, shlex.split(command)[1:])
    assert result.exit_code == 0, result.output
    assert f"Total number of binned samples = {bins}" in result.output
    match = re.search(r"^E = (\S+) \+- (\S+) Ha\.", result.output, re.MULTILINE)
    assert match is not None, result.output
    assert all(np.isfinite(float(v)) for v in match.groups())
    assert float(match.group(1)) == 3.0


def test_weighted_statistics_example_is_executable():
    """Execute the weighted-binning documentation snippet and check its exact replica values."""
    source = (ROOT / "doc/notes/error_analysis.md").read_text()
    snippet = next(m.group(1) for m in checks.PYTHON_FENCE_PATTERN.finditer(source) if "replicas =" in m.group(1))
    namespace = {}
    exec(compile(snippet, "weighted-binning-doc-example", "exec"), namespace)
    assert namespace["replicas"] == [9.0, 9.0]


def test_rendered_api_guard_catches_inherited_method_omission(tmp_path):
    """Detect a missing inherited driver method in the rendered API documentation."""
    headings = [
        f"jqmc.{driver}.{method}"
        for driver in ("MCMC", "GFMC_n", "GFMC_t")
        for method in ("get_E", "get_aF", "get_aT", "get_aP", "load_from_hdf5")
    ]
    page = tmp_path / "api_reference.html"
    page.write_text("\n".join(f'<dt id="{name}"></dt>' for name in headings))
    assert not check_html(tmp_path)
    page.write_text(page.read_text().replace('id="jqmc.GFMC_n.get_E"', 'id="lost"'))
    assert any("jqmc.GFMC_n.get_E" in error for error in check_html(tmp_path))


def test_linear_staging_paths_and_original_qe_path_regression(tmp_path):
    """Published staging paths resolve and a corrupted QE save-directory path is detected."""
    from scripts.check_example_paths import check_all, trace_paths

    assert not check_all()
    # Use real supplied inputs but a mutated README in a scratch copy.
    import shutil

    example = tmp_path / "example08"
    shutil.copytree(ROOT / "examples/jqmc-example08", example)
    # Keep the helper reachable at the same relative depth as the real tree.
    shutil.copyfile(ROOT / "examples/copy_optimized_hamiltonian.py", tmp_path / "copy_optimized_hamiltonian.py")
    readme = example / "README.md"
    readme.write_text(readme.read_text().replace("../01DFT/out/h_k025.save", "../01DFT/h_k025.save"))
    errors = trace_paths(readme)
    assert any("h_k025.save/wfc1.hdf5" in error for error in errors)


@pytest.mark.parametrize("nspin", [None, 1, 2])
def test_linear_staging_pw_x_names_restricted_and_lsda_wavefunction_files(tmp_path, nspin):
    """The pw.x model stages wfc1.hdf5, or wfcup1.hdf5/wfcdw1.hdf5 for an LSDA (nspin = 2) deck."""
    from scripts.check_example_paths import trace_paths

    dft = tmp_path / "01DFT"
    dft.mkdir()
    spin_line = "" if nspin is None else f"  nspin = {nspin}\n"
    (dft / "pw.in").write_text(f"&control\n  prefix = 'si'\n  outdir = 'out'\n/\n&system\n{spin_line}/\n")
    (tmp_path / "02vmc").mkdir()
    staged = ("wfcup1.hdf5", "wfcdw1.hdf5") if nspin == 2 else ("wfc1.hdf5",)
    copies = "".join(f"cp ../01DFT/out/si.save/{name} copy_{name}\n" for name in staged)
    readme = tmp_path / "README.md"
    readme.write_text(
        "```bash\n"
        "cd 01DFT\n"
        "pw.x -in pw.in > pw.out\n"
        "cd ../02vmc\n"
        f"{copies}"
        "jqmc-tool qe --kind pgto --output hamiltonian_data.h5 ../01DFT/out/si.save\n"
        "```\n"
    )
    assert trace_paths(readme) == []


@pytest.mark.parametrize(
    "files,accepted",
    [
        (("wfc1.hdf5",), True),
        (("wfcup1.hdf5", "wfcdw1.hdf5"), True),
        (("wfcup1.hdf5",), False),
        ((), False),
    ],
)
def test_linear_staging_qe_conversion_accepts_shipped_restricted_or_lsda_save(tmp_path, files, accepted):
    """jqmc-tool qe accepts a save holding wfc1.hdf5 or both wfcup1/wfcdw1.hdf5, and rejects anything else."""
    from scripts.check_example_paths import trace_paths

    save = tmp_path / "01DFT" / "si.save"
    save.mkdir(parents=True)
    for name in files:
        (save / name).write_bytes(b"")
    readme = tmp_path / "README.md"
    readme.write_text("```bash\njqmc-tool qe --kind pgto --output hamiltonian_data.h5 01DFT/si.save\n```\n")
    errors = trace_paths(readme)
    if accepted:
        assert errors == []
    else:
        assert len(errors) == 1 and "si.save/wfc1.hdf5" in errors[0], errors


def test_scientific_molecular_tutorial_analysis_minima():
    """Every shipped example TOML (molecular and periodic) satisfies the central minimum sample and warmup requirements."""
    import tomllib
    from jqmc._setting import (
        MCMC_MIN_WARMUP_STEPS,
        MCMC_MIN_BIN_BLOCKS,
        GFMC_MIN_WARMUP_STEPS,
        GFMC_MIN_BIN_BLOCKS,
        GFMC_MIN_COLLECT_STEPS,
    )

    checked = set()
    for path in sorted((ROOT / "examples").rglob("*.toml")):
        data = tomllib.loads(path.read_text())
        for section in ("vmc", "mcmc"):
            if section in data:
                checked.add(section)
                assert data[section]["num_mcmc_warmup_steps"] >= MCMC_MIN_WARMUP_STEPS, path
                assert data[section]["num_mcmc_bin_blocks"] >= MCMC_MIN_BIN_BLOCKS, path
        for section in ("lrdmc-bra", "lrdmc-tau"):
            if section in data:
                checked.add(section)
                assert data[section]["num_gfmc_warmup_steps"] >= GFMC_MIN_WARMUP_STEPS, path
                assert data[section]["num_gfmc_bin_blocks"] >= GFMC_MIN_BIN_BLOCKS, path
                assert data[section]["num_gfmc_collect_steps"] >= GFMC_MIN_COLLECT_STEPS, path
    assert {"vmc", "mcmc", "lrdmc-tau"} <= checked, checked


def test_published_analysis_command_lines_satisfy_minima():
    """Every README analysis command, parsed by the real jqmc-tool parser, meets the central minima."""
    from jqmc import jqmc_tool
    from jqmc._setting import (
        MCMC_MIN_WARMUP_STEPS,
        MCMC_MIN_BIN_BLOCKS,
        GFMC_MIN_WARMUP_STEPS,
        GFMC_MIN_BIN_BLOCKS,
        GFMC_MIN_COLLECT_STEPS,
    )

    minima = {
        "mcmc": {"num_mcmc_warmup_steps": MCMC_MIN_WARMUP_STEPS, "num_mcmc_bin_blocks": MCMC_MIN_BIN_BLOCKS},
        "lrdmc": {
            "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
            "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
            "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
        },
    }
    # The static gate reads the same constants and option defaults without importing jqmc.
    assert checks.load_analysis_minima() == {checks.ANALYSIS_MINIMA[g][k]: v for g in minima for k, v in minima[g].items()}
    static_options = checks.load_analysis_tool_options()
    below, seen = [], set()
    for location, group, command, arguments in checks.published_analysis_commands():
        cli = jqmc_tool._cli.commands[group].commands[command]
        params = cli.make_context(command, list(arguments)).params  # actual parser: defaults, order, --long forms
        settings = {opt[2:]: params[p.name] for p in cli.params for opt in p.opts if opt.startswith("--")}
        static = checks.analysis_command_settings(static_options[(group, command)], arguments)
        assert {k: static[k] for k in minima[group]} == {k: settings[k] for k in minima[group]}, location
        below += [
            f"{location}: {key}={settings[key]} < {value}" for key, value in minima[group].items() if settings[key] < value
        ]
        seen.add((group, command))
    assert below == []
    expected = {(g, c) for g in ("mcmc", "lrdmc") for c in ("compute-energy", "compute-force")}
    assert seen == expected | {("lrdmc", "extrapolate-energy")}, seen


def test_solid_state_workflow_example_is_in_the_public_manual_and_gate(tmp_path, monkeypatch):
    """The public manual includes the silicon workflow and its absence is detected by the documentation check."""
    from scripts.render_examples_doc import render_examples_page

    page = render_examples_page()
    assert "## jqmc-workflow-example04: silicon equation of state" in page
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "EXAMPLES", tmp_path / "examples")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/README.md").write_text("")
    errors = []
    checks.check_feature_tutorials(errors)
    assert any("jqmc-workflow-example04/run_pipelines.py: required" in error for error in errors)
    assert any("feature index is missing 'Equation of state'" in error for error in errors)


def test_examples_renderer_keeps_code_blocks_verbatim():
    """Adapt Markdown prose for the manual without altering fenced example code."""
    """Heading demotion skips fenced code, GitHub alerts become MyST admonitions, manual links are rebased."""
    from scripts.render_examples_doc import _convert_github_alerts, _demote_headings_outside_fences

    source = "# Title\n\n```toml\n# comment\n[precision]\n```\n\n## Section\n"
    assert _demote_headings_outside_fences(source) == "## Title\n\n```toml\n# comment\n[precision]\n```\n\n### Section\n"
    assert _convert_github_alerts("> [!NOTE]\n> First line.\n> Second line.\n\nafter\n") == (
        ":::{note}\nFirst line.\nSecond line.\n:::\n\nafter\n"
    )
    page = __import__("scripts.render_examples_doc", fromlist=["render_examples_page"]).render_examples_page()
    assert "[!NOTE]" not in page
    assert "](../../doc/" not in page
    for readme in sorted((ROOT / "examples").glob("jqmc-*/README.md")):
        for block in re.findall(r"^```[^\n]*\n(.*?)^```\s*$", readme.read_text("utf-8"), flags=re.M | re.S):
            assert block in page, f"{readme}: a code block is not reproduced verbatim"
