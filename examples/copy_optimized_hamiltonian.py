"""Hand an actual VMC dump, never the unoptimized input, to the next step.

By default the largest numeric ``hamiltonian_data_opt_step_N.h5`` is copied.
With ``--average-last N`` the last N dumps (by step number) are averaged with
``jqmc-tool vmc average-wf`` instead; this is the tutorials' standard hand-off.
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

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

DUMP_NAME = re.compile(r"hamiltonian_data_opt_step_([0-9]+)\.h5")


def numbered_dumps(directory: Path) -> dict[int, Path]:
    """All optimization dumps of ``directory`` keyed by their numeric step."""
    candidates = {}
    for path in directory.iterdir():
        match = DUMP_NAME.fullmatch(path.name)
        if path.is_file() and match:
            number = int(match.group(1))
            if number in candidates:
                raise ValueError(f"Ambiguous dumps for step {number} in {directory}")
            candidates[number] = path
    return candidates


def latest_dump(directory: Path, step: int | None = None) -> Path:
    """Select by numeric step, including partial and continued optimizations."""
    candidates = numbered_dumps(directory)
    if not candidates or (step is not None and step not in candidates):
        raise FileNotFoundError(f"No optimized Hamiltonian dump for step {step or 'latest'} in {directory}")
    return candidates[max(candidates) if step is None else step]


def copy_dump(directory: Path, destination: Path, *, step: int | None = None) -> Path:
    """Refuse to overwrite an existing downstream input or a checkpoint."""
    source = latest_dump(directory, step)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    return source


def last_dumps(directory: Path, count: int, *, step: int | None = None, allow_fewer: bool = False) -> list[Path]:
    """The ``count`` dumps with the largest step numbers, ending at ``step`` when given."""
    if count < 1:
        raise ValueError(f"--average-last needs a positive number of steps, got {count}")
    candidates = numbered_dumps(directory)
    if step is not None:
        if step not in candidates:
            raise FileNotFoundError(f"No optimized Hamiltonian dump for step {step} in {directory}")
        candidates = {number: path for number, path in candidates.items() if number <= step}
    selected = sorted(candidates)[-count:]
    if not selected:
        raise FileNotFoundError(f"No optimized Hamiltonian dump in {directory}")
    if len(selected) < count and not allow_fewer:
        raise ValueError(
            f"only {len(selected)} dumps in {directory} for --average-last {count}; "
            "dump every step (wf_dump_freq = 1) or pass --allow-fewer"
        )
    return [candidates[number] for number in selected]


def average_dumps(
    directory: Path,
    destination: Path,
    count: int,
    *,
    step: int | None = None,
    allow_fewer: bool = False,
    run=subprocess.run,
) -> list[Path]:
    """Average the last ``count`` dumps into ``destination`` with ``jqmc-tool vmc average-wf``."""
    sources = last_dumps(directory, count, step=step, allow_fewer=allow_fewer)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    command = [sys.executable, "-m", "jqmc.jqmc_tool", "vmc", "average-wf", *map(str, sources), "-o", str(destination)]
    if run(command).returncode != 0:
        raise ValueError(f"jqmc-tool vmc average-wf refused the dumps of {directory} (see its error above)")
    return sources


def main() -> None:
    """Command-line handoff used by the public tutorials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vmc_directory", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--step",
        type=int,
        help="choose a reviewed dump instead of the largest numeric step (with --average-last: the last step)",
    )
    parser.add_argument(
        "--average-last",
        type=int,
        metavar="N",
        help="average the last N dumps with jqmc-tool vmc average-wf instead of copying",
    )
    parser.add_argument("--allow-fewer", action="store_true", help="with --average-last, accept fewer than N dumps")
    args = parser.parse_args()
    if args.allow_fewer and args.average_last is None:
        parser.error("--allow-fewer requires --average-last")
    try:
        if args.average_last is None:
            source = copy_dump(args.vmc_directory, args.destination, step=args.step)
        else:
            sources = average_dumps(
                args.vmc_directory, args.destination, args.average_last, step=args.step, allow_fewer=args.allow_fewer
            )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Handoff failed: {exc}\n")
    if args.average_last is None:
        print(f"Copied optimized trial: {source} -> {args.destination}")
    else:
        steps = [int(DUMP_NAME.fullmatch(path.name).group(1)) for path in sources]
        print(
            f"Averaged optimized trial: {len(sources)} dumps, steps {steps[0]}-{steps[-1]} of {args.vmc_directory} -> {args.destination}"
        )
        if steps[-1] - steps[0] + 1 != len(steps):
            print("Note: the averaged dumps are not consecutive steps (wf_dump_freq > 1 or missing dumps).")
    print("Selection is by saved step, not a test of scientific convergence.")


if __name__ == "__main__":
    main()
