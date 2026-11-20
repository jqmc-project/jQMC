"""Console entry points that answer --version / --help without MPI or JAX (issue #185).

``jqmc.jqmc_cli`` imports ``mpi4py.MPI`` (which calls ``MPI_Init_thread``) and
JAX at module level. Outside ``srun`` / ``mpirun`` some MPI stacks (Intel MPI on
a login node) abort in that initialization, so ``jqmc --version`` could not
print anything. This module imports only the standard library and
``jqmc._build_info``. It answers ``--version`` / ``-V`` / ``--help`` / ``-h``, no
arguments, and the argument errors itself, and imports the heavy CLI only to
run a job. ``jqmc_cli._cli`` uses the same argument check, so the usage text
and messages have a single source.
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
import sys

from ._build_info import build_info

CLI_USAGE = """usage: jqmc <input.toml>
       jqmc --help | --version

Run a jQMC job. The job type (vmc, mcmc, lrdmc-tau, lrdmc-bra) and every
parameter are read from the TOML input file; see the CLI reference in the
manual (https://jqmc-project.github.io/jQMC/api_reference_cli.html).
Template inputs are generated with "jqmc-tool <job> generate-input".
"""


def _write_version() -> None:
    sys.stdout.write("".join(f"{line}\n" for line in build_info().banner_lines()))


def parse_cli_arguments(argv: list[str]) -> str:
    """Return the input TOML path, or exit for help, version and argument errors.

    No arguments print the usage and exit 2; ``-h`` / ``--help`` print it and
    exit 0; ``-V`` / ``--version`` print the two version lines and exit 0. More
    than one argument, or a path that is not a file, print the usage on stderr
    and exit with the error message.
    """
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(CLI_USAGE)
        sys.exit(0 if argv else 2)
    if argv[0] in ("-V", "--version"):
        _write_version()
        sys.exit(0)
    if len(argv) > 1:
        sys.stderr.write(CLI_USAGE)
        raise SystemExit("jqmc: more than one input TOML file is not acceptable.")
    toml_file = argv[0]
    if not os.path.isfile(toml_file):
        sys.stderr.write(CLI_USAGE)
        raise SystemExit(f"jqmc: input file {toml_file!r} does not exist.")
    return toml_file


def main() -> None:
    """``jqmc`` console script: check the arguments here, then run the job through the MPI-owning CLI."""
    parse_cli_arguments(sys.argv[1:])
    from .jqmc_cli import main as run_job

    return run_job()


def tool_main() -> None:
    """``jqmc-tool`` console script: answer ``--version`` here, then hand over to the full tool CLI."""
    if sys.argv[1:] == ["--version"]:
        _write_version()
        sys.exit(0)
    from .jqmc_tool import _cli_main

    return _cli_main()
