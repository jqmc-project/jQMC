"""WF_Workflow -- TREXIO to hamiltonian_data.h5 conversion.

Wraps ``jqmc-tool trexio convert-to`` which converts a TREXIO file (.h5)
into the internal ``hamiltonian_data.h5`` format, optionally attaching
Jastrow one-body, two-body, and three-body factors.

This workflow runs **locally** (no remote job submission).
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

import os
import subprocess
from logging import getLogger

from .state import WorkflowStatus
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)


class WF_Workflow(Workflow):
    """Convert a TREXIO file to ``hamiltonian_data.h5``.

    Calls ``jqmc-tool trexio convert-to`` under the hood.

    Args:
        trexio_file (str):
            Path to the input TREXIO ``.h5`` file.
        hamiltonian_file (str):
            Output filename (default: ``"hamiltonian_data.h5"``).
        j1_parameter (float, optional):
            Jastrow one-body parameter (``-j1``).
        j1_type (str, optional):
            Jastrow one-body functional form (``--jastrow-1b-type``).
            ``"exp"`` (default) or ``"pade"``.
        j2_parameter (float, optional):
            Jastrow two-body parameter (``-j2``).
        j2_type (str, optional):
            Jastrow two-body functional form (``--jastrow-2b-type``).
            ``"pade"`` (default) or ``"exp"``.
        j3_basis_type (str, optional):
            Jastrow three-body basis-set type (``-j3``).
            One of ``"ao"``, ``"ao-full"``, ``"ao-small"``, ``"ao-medium"``,
            ``"ao-large"``, ``"mo"``, ``"none"``, or ``None`` (disabled).
        jastrow_dtn (str, optional):
            DTN (Drummond-Towler-Needs) polynomial Jastrow preset
            (``--jastrow-dtn``): ``"none"``, ``"u"``, ``"u_chi"``,
            ``"u_f"`` or ``"u_chi_f"``.  When not ``"none"`` the DTN family
            is mutually exclusive with the j1 / j2 / AO j3 family.  For
            this and every ``dtn_*`` argument below, *None* (default) omits
            the flag so the ``jqmc-tool trexio convert-to`` default applies.
        dtn_c (int, optional):
            DTN cutoff polynomial smoothness order C, >= 3 (``--dtn-C``).
        dtn_n_u (int, optional):
            DTN u-term polynomial order N_u (``--dtn-Nu``).
        dtn_n_chi (int, optional):
            DTN chi-term polynomial order N_chi (``--dtn-Nchi``).
        dtn_n_eN (int, optional):
            DTN f-term electron-ion polynomial order N_eN (``--dtn-NeN``).
        dtn_n_ee (int, optional):
            DTN f-term electron-electron polynomial order N_ee
            (``--dtn-Nee``).
        dtn_L_u (float, optional):
            DTN u-term cutoff in bohr (``--dtn-Lu``).  Omit for the
            geometry-derived default.
        dtn_L_chi (float, optional):
            DTN chi-term cutoff in bohr, broadcast across species
            (``--dtn-Lchi``).  Omit for auto.
        dtn_L_f (float, optional):
            DTN f-term cutoff in bohr, broadcast across species
            (``--dtn-Lf``).  Omit for auto.
        dtn_cutoff_ratio_u (float, optional):
            PBC u cutoff ratio relative to the Wigner-Seitz inscribed
            radius R_ins (``--dtn-cutoff-ratio-u``).
        dtn_cutoff_ratio_chi (float, optional):
            PBC chi cutoff ratio relative to R_ins
            (``--dtn-cutoff-ratio-chi``).
        dtn_cutoff_ratio_f (float, optional):
            PBC f cutoff ratio relative to R_ins/2
            (``--dtn-cutoff-ratio-f``).
        dtn_cutoff_abs_cap (float, optional):
            Auto-cutoff absolute cap in bohr (``--dtn-cutoff-abs-cap``).
        ao_conv_to (str, optional):
            Convert AOs after building the Hamiltonian (``--ao-conv-to``).
            ``"cart"``  -> convert to Cartesian AOs,
            ``"sphe"`` -> convert to spherical-harmonic AOs,
            ``None``    -> keep the original representation.
        twist_average (bool):
            Convert every k-point of a multi-k TREXIO file into one
            multi-twist HDF5 file (``--twist-average``). Every twist shares
            one phase-free ``ao*`` J3; incompatible with
            ``j3_basis_type="mo"`` and with ``ao_conv_to``.
        tiling (list[int], optional):
            Primitive k blocks in a supercell (``--tiling``): 3 diagonal
            factors or 9 row-major integer matrix entries.  Any other
            length raises ``ValueError``.  Default *None* (no tiling).
        supertwist (list[float], optional):
            Fractional supercell twist to select (``--supertwist``), three
            finite fractional coordinates.  Requires ``tiling`` and cannot
            be combined with ``twist_average``.  Default *None* (the CLI
            infers the single sector).

    Examples:
        >>> wf = WF_Workflow(
        ...     trexio_file="molecular.h5",
        ...     j1_parameter=1.0,
        ...     j1_type="pade",
        ...     j2_parameter=0.5,
        ...     j2_type="exp",
        ...     j3_basis_type="ao-small",
        ... )
        >>> status, out_files, out_values = wf.launch()

    Notes:
        This workflow runs **locally** -- no remote job submission is
        involved.  It calls ``jqmc-tool trexio convert-to`` via
        :func:`subprocess.run`.

    See Also:
        VMC_Workflow : Optimise the wavefunction produced by this step.
    """

    def __init__(
        self,
        trexio_file: str = "trexio.h5",
        hamiltonian_file: str = "hamiltonian_data.h5",
        j1_parameter: float | None = None,
        j1_type: str | None = None,
        j2_parameter: float | None = None,
        j2_type: str | None = None,
        j3_basis_type: str | None = None,
        jastrow_dtn: str | None = None,
        dtn_c: int | None = None,
        dtn_n_u: int | None = None,
        dtn_n_chi: int | None = None,
        dtn_n_eN: int | None = None,
        dtn_n_ee: int | None = None,
        dtn_L_u: float | None = None,
        dtn_L_chi: float | None = None,
        dtn_L_f: float | None = None,
        dtn_cutoff_ratio_u: float | None = None,
        dtn_cutoff_ratio_chi: float | None = None,
        dtn_cutoff_ratio_f: float | None = None,
        dtn_cutoff_abs_cap: float | None = None,
        ao_conv_to: str | None = None,
        twist_average: bool = False,
        tiling: list[int] | None = None,
        supertwist: list[float] | None = None,
    ):
        super().__init__()
        self.trexio_file = trexio_file
        self.hamiltonian_file = hamiltonian_file
        self.j1_parameter = j1_parameter
        self.j1_type = j1_type
        self.j2_parameter = j2_parameter
        self.j2_type = j2_type
        self.j3_basis_type = j3_basis_type
        self.jastrow_dtn = jastrow_dtn
        self.dtn_c = dtn_c
        self.dtn_n_u = dtn_n_u
        self.dtn_n_chi = dtn_n_chi
        self.dtn_n_eN = dtn_n_eN
        self.dtn_n_ee = dtn_n_ee
        self.dtn_L_u = dtn_L_u
        self.dtn_L_chi = dtn_L_chi
        self.dtn_L_f = dtn_L_f
        self.dtn_cutoff_ratio_u = dtn_cutoff_ratio_u
        self.dtn_cutoff_ratio_chi = dtn_cutoff_ratio_chi
        self.dtn_cutoff_ratio_f = dtn_cutoff_ratio_f
        self.dtn_cutoff_abs_cap = dtn_cutoff_abs_cap

        if ao_conv_to is not None and ao_conv_to not in ("cart", "sphe"):
            raise ValueError(f"ao_conv_to must be None, 'cart', or 'sphe', got {ao_conv_to!r}")
        self.ao_conv_to = ao_conv_to
        self.twist_average = bool(twist_average)
        self.tiling = list(tiling or [])
        if self.tiling and len(self.tiling) not in (3, 9):
            raise ValueError("tiling must contain 3 diagonal factors or 9 row-major matrix entries.")
        self.supertwist = None if supertwist is None else list(supertwist)
        if self.supertwist is not None:
            from math import isfinite

            if len(self.supertwist) != 3 or not all(isfinite(value) for value in self.supertwist):
                raise ValueError("supertwist must contain three finite fractional coordinates.")
            if not self.tiling or self.twist_average:
                raise ValueError("supertwist requires tiling and cannot be combined with twist_average.")

    def _build_command(self) -> list[str]:
        """Build the ``jqmc-tool trexio convert-to`` CLI command (argv list)."""
        cmd = ["jqmc-tool", "trexio", "convert-to", self.trexio_file]
        cmd += ["-o", self.hamiltonian_file]

        if self.j1_parameter is not None:
            cmd += ["-j1", str(self.j1_parameter)]
        if self.j1_type is not None:
            cmd += ["--jastrow-1b-type", str(self.j1_type)]
        if self.j2_parameter is not None:
            cmd += ["-j2", str(self.j2_parameter)]
        if self.j2_type is not None:
            cmd += ["--jastrow-2b-type", str(self.j2_type)]
        if self.j3_basis_type is not None:
            cmd += ["-j3", str(self.j3_basis_type)]
        if self.jastrow_dtn is not None:
            cmd += ["--jastrow-dtn", str(self.jastrow_dtn)]
        if self.dtn_c is not None:
            cmd += ["--dtn-C", str(self.dtn_c)]
        if self.dtn_n_u is not None:
            cmd += ["--dtn-Nu", str(self.dtn_n_u)]
        if self.dtn_n_chi is not None:
            cmd += ["--dtn-Nchi", str(self.dtn_n_chi)]
        if self.dtn_n_eN is not None:
            cmd += ["--dtn-NeN", str(self.dtn_n_eN)]
        if self.dtn_n_ee is not None:
            cmd += ["--dtn-Nee", str(self.dtn_n_ee)]
        if self.dtn_L_u is not None:
            cmd += ["--dtn-Lu", str(self.dtn_L_u)]
        if self.dtn_L_chi is not None:
            cmd += ["--dtn-Lchi", str(self.dtn_L_chi)]
        if self.dtn_L_f is not None:
            cmd += ["--dtn-Lf", str(self.dtn_L_f)]
        if self.dtn_cutoff_ratio_u is not None:
            cmd += ["--dtn-cutoff-ratio-u", str(self.dtn_cutoff_ratio_u)]
        if self.dtn_cutoff_ratio_chi is not None:
            cmd += ["--dtn-cutoff-ratio-chi", str(self.dtn_cutoff_ratio_chi)]
        if self.dtn_cutoff_ratio_f is not None:
            cmd += ["--dtn-cutoff-ratio-f", str(self.dtn_cutoff_ratio_f)]
        if self.dtn_cutoff_abs_cap is not None:
            cmd += ["--dtn-cutoff-abs-cap", str(self.dtn_cutoff_abs_cap)]
        if self.ao_conv_to is not None:
            cmd += ["--ao-conv-to", str(self.ao_conv_to)]
        if self.twist_average:
            cmd.append("--twist-average")
        if self.tiling:
            cmd += ["--tiling", " ".join(str(value) for value in self.tiling)]
        if self.supertwist is not None:
            cmd += ["--supertwist", " ".join(str(value) for value in self.supertwist)]

        return cmd

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        return {
            "trexio_file": self.trexio_file,
            "hamiltonian_file": self.hamiltonian_file,
            "j1_parameter": self.j1_parameter,
            "j1_type": self.j1_type,
            "j2_parameter": self.j2_parameter,
            "j2_type": self.j2_type,
            "j3_basis_type": self.j3_basis_type,
            "twist_average": self.twist_average,
            "tiling": self.tiling,
            "supertwist": self.supertwist,
        }

    async def run(self) -> tuple:
        """Run the TREXIO->hamiltonian conversion (locally).

        Returns:
            tuple:
                ``(status, output_files, output_values)``
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        command = self._build_command()
        logger.info(f"  Running: {' '.join(command)}")

        # Stream the subprocess's stdout / stderr line-by-line so live
        # progress (jqmc-tool's per-step logs) surfaces immediately
        # instead of being buffered until completion. Same pattern as
        # QE_Workflow.run().
        try:
            with subprocess.Popen(
                command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                cwd=_wd,
                bufsize=1,  # line-buffered (text mode)
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    logger.info(line.rstrip())
                rc = proc.wait()
            if rc != 0:
                logger.error(f"Command failed (rc={rc})")
                self.status = WorkflowStatus.FAILED
                return self.status, [], {}
        except FileNotFoundError as e:
            logger.error(f"Command failed: '{command[0]}' not found on PATH ({e})")
            self.status = WorkflowStatus.FAILED
            return self.status, [], {}

        if not os.path.isfile(os.path.join(_wd, self.hamiltonian_file)):
            logger.error(f"Output file not found: {self.hamiltonian_file}")
            self.status = WorkflowStatus.FAILED
            return self.status, [], {}

        self.status = WorkflowStatus.COMPLETED
        self.output_files = [self.hamiltonian_file]
        self.output_values = {"hamiltonian_file": self.hamiltonian_file}
        return self.status, self.output_files, self.output_values
