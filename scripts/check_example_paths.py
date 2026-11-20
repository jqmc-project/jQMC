"""Read-only staging trace for linear CLI tutorials; no calculator is executed.

The model follows cd/copy/TOML commands and synthesizes only declared outputs
of PySCF, QE, converters and samplers. It does not prove scientific execution
or model optional manual alat sweeps and interactive continuation edits.
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
from pathlib import Path
import re
import shlex
import tomllib

ROOT = Path(__file__).resolve().parents[1]
LINEAR_EXAMPLES = ("01", "02", "03", "04", "05", "06", "07", "08", "10")


def _dump_steps(generated: dict[Path, str], directory: Path) -> list[int]:
    """Step numbers of the modeled ``hamiltonian_data_opt_step_N.h5`` dumps in ``directory``."""
    return sorted(
        int(match.group(1))
        for p in generated
        if p.parent == directory and (match := re.fullmatch(r"hamiltonian_data_opt_step_(\d+)\.h5", p.name))
    )


def trace_paths(readme: Path, handoffs: list[tuple[Path, dict]] | None = None) -> list[str]:
    """Trace the staging of one README; append each helper hand-off ``(destination, options)`` to ``handoffs``."""
    origin = readme.parent.resolve()
    cwd = origin
    generated: dict[Path, str] = {}
    directories = set()
    errors = []

    def exists(path):
        return path.exists() or path in generated or path in directories

    def contents(path):
        return generated[path] if path in generated else path.read_text("utf-8")

    def require(path):
        if not exists(path):
            raise ValueError(f"missing {path.relative_to(origin) if path.is_relative_to(origin) else path}")

    for block in re.finditer(r"^```bash\s*\n([\s\S]*?)^```", readme.read_text("utf-8"), re.MULTILINE):
        lines = block.group(1).replace("\\\n", " ").splitlines()
        for line in lines:
            for command in line.split(" && "):
                args = shlex.split(command, comments=True)
                if not args or args[0].startswith("(") or args[0] in {"for", "do", "done", "export"}:
                    # Shell loops and subshells (optional manual sweeps) are not modeled.
                    continue
                if args[0] in {"mpirun", "mpiexec"} and "pw.x" in args:
                    args = args[args.index("pw.x") :]
                op = args[0]
                path = lambda name: (cwd / name).resolve()
                try:
                    if op == "cd":
                        if "$example_run" in args[1]:
                            cwd = origin
                        else:
                            target = path(args[1])
                            require(target)
                            if not target.is_dir() and target not in directories:
                                raise ValueError(f"cd target is not a directory: {target}")
                            cwd = target
                    elif op == "mkdir":
                        for name in args[1:]:
                            if not name.startswith("-"):
                                target = path(name)
                                directories.update([target, *target.parents])
                    elif op in {"cp", "mv"} and not args[1].startswith("-"):
                        source, target = path(args[1]), path(args[2])
                        require(source)
                        if target.is_dir() or target in directories:
                            target /= source.name
                        require(target.parent)
                        generated[target] = contents(source) if source.suffix == ".toml" else "staged"
                    elif op == "python" and args[1].endswith("copy_optimized_hamiltonian.py"):
                        require(path(args[1]))
                        positional, options, rest = [], {}, args[2:]
                        while rest:
                            token = rest.pop(0)
                            if token == "--allow-fewer":
                                options[token] = True
                            elif token in {"--step", "--average-last"}:
                                options[token] = int(rest.pop(0))
                            else:
                                positional.append(token)
                        source, target = path(positional[0]), path(positional[1])
                        require(source)
                        candidates = _dump_steps(generated, source)
                        if "--step" in options:
                            if options["--step"] not in candidates:
                                raise ValueError(f"no generated dump for step {options['--step']} in {source}")
                            candidates = [step for step in candidates if step <= options["--step"]]
                        if not candidates:
                            raise ValueError(f"no generated optimized dump in {source}")
                        wanted = options.get("--average-last")
                        if wanted is not None and len(candidates) < wanted and not options.get("--allow-fewer"):
                            raise ValueError(f"--average-last {wanted} but only {len(candidates)} generated dumps in {source}")
                        require(target.parent)
                        generated[target] = "optimized"
                        if handoffs is not None:
                            handoffs.append((target, options))
                    elif op == "python" and args[1] == "run_pyscf.py":
                        source = path(args[1])
                        require(source)
                        tree = ast.parse(contents(source))
                        literals = {}
                        for node in tree.body:
                            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                                try:
                                    literals[node.targets[0].id] = ast.literal_eval(node.value)
                                except (ValueError, TypeError):
                                    if isinstance(node.value, ast.JoinedStr):
                                        pieces = []
                                        for part in node.value.values:
                                            if isinstance(part, ast.Constant):
                                                pieces.append(str(part.value))
                                            elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                                                spec = "" if part.format_spec is None else part.format_spec.values[0].value
                                                pieces.append(format(literals[part.value.id], spec))
                                        literals[node.targets[0].id] = "".join(pieces)
                        filenames = [
                            n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        ]
                        filenames += [v for v in literals.values() if isinstance(v, str)]
                        for name in filenames:
                            if "\n" not in name and name.endswith((".h5", ".hdf5")):
                                generated[path(name)] = "trexio"
                    elif op.startswith("cp2k"):
                        deck = path(args[args.index("-i") + 1])
                        require(deck)
                        generated[deck.with_name(deck.stem + "-TREXIO.h5")] = "trexio"
                    elif op == "pw.x":
                        flag = "-in" if "-in" in args else "-i"
                        deck = path(args[args.index(flag) + 1])
                        require(deck)
                        values = dict(re.findall(r"(prefix|outdir)\s*=\s*['\"]([^'\"]+)['\"]", contents(deck), re.IGNORECASE))
                        target = path(values.get("outdir", ".")) / (values["prefix"] + ".save")
                        directories.update([target, *target.parents])
                        # pw.x names the first k-point file wfc1.hdf5, or
                        # wfcup1.hdf5/wfcdw1.hdf5 for an LSDA (nspin = 2) run.
                        nspin = re.search(r"\bnspin\s*=\s*(\d+)", contents(deck), re.IGNORECASE)
                        names = ("wfcup1.hdf5", "wfcdw1.hdf5") if nspin and nspin.group(1) == "2" else ("wfc1.hdf5",)
                        for name in names:
                            generated[target / name] = "qe"
                    elif op == "jqmc-tool":
                        if args[1:3] == ["trexio", "convert-to"]:
                            require(path(args[3]))
                            output = next(
                                (args[i + 1] for i, a in enumerate(args) if a in {"-o", "--output"}), "hamiltonian_data.h5"
                            )
                            generated[path(output)] = "hamiltonian"
                        elif args[1] == "qe":
                            savedir = path(next(a for a in args[2:] if a.endswith(".save")))
                            if not (exists(savedir / "wfcup1.hdf5") and exists(savedir / "wfcdw1.hdf5")):
                                require(savedir / "wfc1.hdf5")
                            output = next(
                                (args[i + 1] for i, a in enumerate(args) if a in {"-o", "--output"}), "hamiltonian_data.h5"
                            )
                            generated[path(output)] = "hamiltonian"
                        elif args[1:3] == ["hamiltonian", "conv-wf"]:
                            require(path(args[-1]))
                            generated[path("hamiltonian_data_conv.h5")] = "converted"
                    elif "jqmc" in args:
                        index = args.index("jqmc")
                        deck = path(args[index + 1])
                        require(deck)
                        data = tomllib.loads(contents(deck))
                        control = data["control"]
                        require(path(control.get("hamiltonian_h5", "hamiltonian_data.h5")))
                        if control["job_type"] == "vmc":
                            # Continuations number their dumps after the steps already done in this directory.
                            done = max(_dump_steps(generated, cwd), default=0) if control.get("restart") else 0
                            steps, frequency = int(data["vmc"]["num_opt_steps"]), int(data["vmc"].get("wf_dump_freq", 1))
                            for step in sorted({*range(frequency, steps + 1, frequency), steps}):
                                generated[path(f"hamiltonian_data_opt_step_{done + step}.h5")] = "optimized"
                        generated[path(control.get("restart_chk", "restart.h5"))] = "checkpoint"
                except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
                    errors.append(f"{readme.parent.name}: {command!r}: {exc}")
    return errors


def check_all() -> list[str]:
    return [error for name in LINEAR_EXAMPLES for error in trace_paths(ROOT / f"examples/jqmc-example{name}/README.md")]


if __name__ == "__main__":
    errors = check_all()
    print("\n".join(errors) if errors else "Linear tutorial staging paths passed (calculators modeled, not executed)")
    raise SystemExit(bool(errors))
