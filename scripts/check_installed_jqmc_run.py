"""Run short MCMC, VMC and LRDMC calculations with the INSTALLED jqmc, as a user would.

Used by the distribution check (public-build-distributions.yml) and by the index
check after uploads (public-verify-index.yml). It converts a small TREXIO file with
the installed ``jqmc-tool``, runs each job type with the installed ``jqmc``
under ``mpirun -np 2``, and post-processes the results with ``jqmc-tool``.
Every command must succeed, every run must reach "Program ends", and every
energy must be finite. It checks that the package works, not its numbers:
the numerical tests run separately.

    python -I scripts/check_installed_jqmc_run.py tests/trexio_example_files/H2_ecp_ccpvdz_cart.h5
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
import math
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

CONTROL = """[control]
job_type = "{job_type}"
mcmc_seed = 34456
number_of_walkers = 4
max_time = 3600
restart = false
restart_chk = "restart.h5"
hamiltonian_h5 = "hamiltonian_data.h5"
verbosity = "low"
"""
SECTIONS = {
    "mcmc": """[mcmc]
num_mcmc_steps = 60
num_mcmc_per_measurement = 20
num_mcmc_warmup_steps = 10
num_mcmc_bin_blocks = 5
Dt = 2.0
""",
    "vmc": """[vmc]
num_mcmc_steps = 40
num_mcmc_per_measurement = 20
num_mcmc_warmup_steps = 10
num_mcmc_bin_blocks = 5
Dt = 2.0
num_opt_steps = 2
""",
    # jqmc's minimums: 25 warmup steps (E_scf), 10 bin blocks, 5 collect steps.
    "lrdmc-bra": """[lrdmc-bra]
num_mcmc_steps = 100
num_mcmc_per_measurement = 20
alat = 0.3
num_gfmc_warmup_steps = 25
num_gfmc_bin_blocks = 10
num_gfmc_collect_steps = 5
""",
}
# jqmc-tool commands that read each run's results.
POST = {
    "mcmc": ["mcmc", "compute-energy", "restart.h5", "-w", "10", "-b", "5"],
    "vmc": ["vmc", "analyze-output", "out"],
    "lrdmc-bra": ["lrdmc", "compute-energy", "restart.h5", "-w", "25", "-b", "10", "-c", "5"],
}
# "E = -1.13 +- 0.015 Ha." (compute-energy) and "   1  -1.161(18)  ..." rows (analyze-output).
# The value is any token, so that a nan, inf or broken row is read and rejected, not skipped.
ENERGY = {
    "mcmc": r"^E = (\S+) \+- \S+ Ha\.$",
    "vmc": r"^\s*\d+\s+([^\s(]+)",
    "lrdmc-bra": r"^E = (\S+) \+- \S+ Ha\.$",
}
# Energy rows each job must report: one per optimization step for vmc.
ROWS = {"mcmc": 1, "vmc": 2, "lrdmc-bra": 1}


def read_energies(job_type: str, post: str) -> list[float]:
    """Every energy row of a jqmc-tool output; a missing, unreadable or non-finite row fails."""
    tokens = re.findall(ENERGY[job_type], post, re.MULTILINE)
    if len(tokens) != ROWS[job_type]:
        raise SystemExit(f"{job_type}: expected {ROWS[job_type]} energies, found {len(tokens)}:\n{post}")
    try:
        energies = [float(token) for token in tokens]
    except ValueError as error:
        raise SystemExit(f"{job_type}: unreadable energy ({error}):\n{post}") from error
    if not all(math.isfinite(value) for value in energies):
        raise SystemExit(f"{job_type}: non-finite energy in the jqmc-tool output:\n{post}")
    return energies


def run(command: list[str], cwd: Path, log: Path | None = None) -> str:
    print("$", shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=1800)
    text = result.stdout + result.stderr
    if log is not None:
        log.write_text(text)
    if result.returncode != 0:
        # The error lines first: MPI aborts bury them in the middle of the log.
        print("\n".join(sorted({line for line in text.splitlines() if "Error" in line})))
        print(text[-5000:])
        raise SystemExit(f"Failed ({result.returncode}): {shlex.join(command)}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trexio", type=Path, help="small TREXIO file to start from")
    # --oversubscribe: Open MPI refuses 2 ranks on a runner it counts fewer slots on.
    parser.add_argument("--mpirun", default="mpirun --oversubscribe -np 2", help="MPI launcher prefix (default: %(default)s)")
    args = parser.parse_args()

    # The reader itself: a bad later row must stop the check, not drop out of the list.
    assert read_energies("vmc", "   1  -1.1(1)  x\n   2  -1.2(1)  x\n") == [-1.1, -1.2]
    for row in ("nan(nan)", "inf(0)", "-1.1+/-nan"):
        try:
            read_energies("vmc", f"   1  -1.1(1)  x\n   2  {row}  x\n")
        except SystemExit:
            continue
        raise SystemExit(f"The energy reader accepted the row {row}")

    import jqmc
    import jqmc_rs

    # The installed package, not a checkout on sys.path.
    checkout = Path(__file__).resolve().parents[1]
    for module in (jqmc, jqmc_rs):
        if Path(module.__file__).resolve().is_relative_to(checkout):
            raise SystemExit(f"Imported the checkout instead of the installed package: {module.__file__}")
    scripts = Path(sysconfig.get_path("scripts"))
    jqmc_cli, tool = str(scripts / "jqmc"), str(scripts / "jqmc-tool")
    trexio = args.trexio.resolve()

    with tempfile.TemporaryDirectory(prefix="jqmc-installed-run-") as directory:
        work = Path(directory)
        shutil.copy(trexio, work / trexio.name)
        run([tool, "trexio", "convert-to", trexio.name, "-j2", "1.0", "-j3", "mo"], work)
        for job_type, section in SECTIONS.items():
            job = work / job_type
            job.mkdir()
            shutil.copy(work / "hamiltonian_data.h5", job)
            (job / "input.toml").write_text(CONTROL.format(job_type=job_type) + section)
            out = run([*shlex.split(args.mpirun), sys.executable, "-I", jqmc_cli, "input.toml"], job, job / "out")
            if "Program ends" not in out:
                raise SystemExit(f"{job_type}: jqmc did not reach 'Program ends'")
            if job_type != "vmc" and not (job / "restart.h5").is_file():
                raise SystemExit(f"{job_type}: no restart.h5")
            post = run([sys.executable, "-I", tool, *POST[job_type]], job)
            energies = read_energies(job_type, post)
            print(f"{job_type}: OK, E = {', '.join(f'{value:.6f}' for value in energies)} Ha", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
