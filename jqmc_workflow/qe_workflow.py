"""QE_Workflow -- Quantum ESPRESSO to hamiltonian_data.h5 conversion.

Wraps ``jqmc-tool qe`` which projects a Quantum ESPRESSO
plane-wave SCF result (a ``prefix.save`` directory) onto a chosen jQMC
basis and writes the internal ``hamiltonian_data.h5`` format, optionally
attaching Jastrow one-body and two-body factors.

Unlike :class:`WF_Workflow` (which converts a TREXIO GTO / pGTO file), this
workflow takes plane-wave input and supports the full set of periodic output
representations: plane-wave (``pw``), B-spline (``bspline``), periodic GTO
(``pgto``), and the additive mixes ``pgto+pw`` / ``pgto+bspline``.

Runs **locally** via ``subprocess.run`` (same pattern as
:class:`WF_Workflow`). Conversion does not submit a scheduler job;
provision memory and CPU resources on the host launching the workflow.
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

from jqmc._setting import PGTO_EXP_TAIL_EPS_USER_DEFAULT
from jqmc.jastrow_factor import DTN_DEFAULT_CUTOFF_RATIO

from .state import WorkflowStatus
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)

#: Supported projection targets (mirror ``jqmc-tool qe --kind``).
_VALID_KINDS = ("pw", "bspline", "pgto", "pgto+pw", "pgto+bspline")

#: Kinds whose output contains a pGTO block and therefore require UPF
#: pseudopotentials to seed the pGTO exponents.
_KINDS_NEEDING_UPF = ("pgto", "pgto+pw", "pgto+bspline")


class QE_Workflow(Workflow):
    """Convert a Quantum ESPRESSO ``prefix.save`` directory to ``hamiltonian_data.h5``.

    Calls ``jqmc-tool qe`` under the hood. The plane-wave bands are
    projected onto the basis selected by ``kind``; pGTO-bearing kinds
    necessarily truncate the basis, so three truncation knobs are exposed with
    sensible defaults (``ecutwfc`` controlling the number of
    retained plane waves, ``alpha_min`` controlling the diffuse-exponent floor,
    and ``max_cond_number`` capping the overlap-matrix condition number).

    Args:
        qe_save_dir (str):
            Path to the QE ``prefix.save`` directory (``data-file-schema.xml``
            and ``wfc<ik>.hdf5``).
        kind (str):
            Projection target. One of ``"pw"``, ``"bspline"``, ``"pgto"``,
            ``"pgto+pw"``, ``"pgto+bspline"`` (default ``"pw"``).
        upf_paths (list[str], optional):
            UPF pseudopotential mapping, one ``"element=path/to/element.upf"``
            entry per element/species, or auto-discovered from the save XML.
            UPF data must be available for the pGTO-bearing kinds
            (``"pgto"``, ``"pgto+pw"``, ``"pgto+bspline"``). Also the source
            of the analytic ECP: ccECP UPF files emitted by Opium with the
            ``rpt`` flag embed a ``[Gaussinfo]`` block carrying the original
            local + non-local Gaussian semi-local terms, which the converter
            parses directly. Standard ONCV / GBRV / PAW UPF files do not
            include ``[Gaussinfo]`` and are rejected for any element that has
            core electrons removed. ECP identity is retained per atom, so
            mixed all-electron/ECP structures and zero-core ECPs receive the
            appropriate Coulomb interaction and J_DTN cusp constraint.
        twist_average (bool):
            If true, convert every k-point in the QE save into one combined
            twist-average HDF5 via ``jqmc-tool qe --twist-average``. Gamma is
            stored as the uniform complex ``k=(0,0,0)`` twist representation;
            single-k conversion remains real at Gamma. Default ``False``.
        tiling (list[int], optional):
            Three diagonal factors or nine row-major entries of an integer
            tiling matrix for PW/B-spline/pGTO. Complete primitive k cosets
            become physical-supercell twists; compatible with twist_average.
            Emitted as one quoted CLI matrix argument.
        supertwist (list[float], optional):
            Select one fractional supercell twist from the source mesh.
            Requires tiling and is exclusive with twist_average.
        hamiltonian_file (str):
            Output filename (default ``"hamiltonian_data.h5"``).
        ecutwfc (float | str, optional):
            Plane-wave handoff kinetic-energy cutoff. Either a numeric value in
            **Rydberg** (QE convention -- pass the same value as your ``pw.x``
            ``ecutwfc``), or the sentinel string ``"auto"``. Numeric: plane
            waves with ``|G+k|^2/2 <= ecutwfc[Ry]/2`` (Hartree) are retained, i.e.
            ``|G+k| <= sqrt(ecutwfc[Ry])`` Bohr^-1. Used by ``"pw"`` /
            ``"pgto+pw"``; larger ``ecutwfc`` keeps more plane waves (higher
            ``n_PW``). ``"auto"`` is only meaningful for ``kind="pgto+pw"``:
            the converter sweeps ``g_cut`` small -> large and picks the
            smallest meeting the hardcoded overlap residual target
            (``jqmc._setting.PW_PGTO_AUTO_RESIDUAL_TARGET``); pass a numeric
            value instead if you want a different fidelity. ``None`` keeps
            the full QE basis for ``"pw"`` and defaults ``"pgto+pw"`` to
            half the QE maximum ``|G+k|``.
        alpha_min (float, optional):
            Diffuse-pGTO exponent floor: primitives with exponent below this are
            dropped and handed off to the PW / spline block. Used by
            ``"pgto+pw"`` / ``"pgto+bspline"``. ``None`` keeps the kind default
            (no extra drop for ``pgto+pw``; keep-all for ``pgto+bspline``).
        exp_tail_eps (float, optional):
            Exponent-tail bound of the ``pgto+pw`` exponent optimiser: no
            primitive may place more than this fraction of its momentum-space
            norm outside the QE WFC sphere. Defaults to
            ``jqmc._setting.PGTO_EXP_TAIL_EPS_USER_DEFAULT`` (``1e-2``);
            ``None`` or ``0`` disables the bound.
        max_cond_number (float, optional):
            Conditioning control for the pGTO least-squares fit. Maps to the
            canonical-orthogonalisation eigenvalue floor
            ``svd_rcond = 1 / max_cond_number``. Exponent-search adoption also
            checks the raw full emitted-AO overlap, not just the pGTO Schur
            complement; rejected searches restore the seed, so this is not
            a blanket guarantee on every emitted seed basis. ``None`` uses the default
            ``1e8`` (``svd_rcond = 1e-8``). Used by ``"pgto"`` / ``"pgto+pw"`` /
            ``"pgto+bspline"``.
        mesh_factor (float, optional):
            B-spline mesh density relative to the QE FFT grid (used by
            ``"bspline"`` / ``"pgto+bspline"``).  ``None`` (default) omits
            ``--mesh-factor``, so the ``jqmc-tool qe`` default
            ``0.75`` applies.
        spline_order (int, optional):
            B-spline polynomial order (used by ``"bspline"`` /
            ``"pgto+bspline"``; default ``5``).
        j1_parameter (float, optional):
            Jastrow one-body parameter (``-j1``).
        j1_type (str, optional):
            Jastrow one-body functional form (``--jastrow-1b-type``):
            ``"exp"`` (default) or ``"pade"``.
        j2_parameter (float, optional):
            Jastrow two-body parameter (``-j2``). Set ``< 0`` to disable.
        j2_type (str, optional):
            Jastrow two-body functional form (``--jastrow-2b-type``):
            ``"pade"`` (default) or ``"exp"``.
        ewald_tol (float, optional):
            Convergence tolerance forwarded to ``build_ewald_data``
            (``--ewald-tol``).
        ecp_z_cores (list[float], optional):
            Per-atom core-electron counts (``--ecp-z-core``, one per atom).
        j3_basis_type (str, optional):
            Pade-family AO/MO J3 basis selector (not supported with DTN).
            With tiling, ao-small/medium/large/full selects an independent
            UPF-seeded periodic GTO basis on the physical supercell; mo is rejected.
        jastrow_dtn (str, optional):
            DTN preset: ``"u"``, ``"u_chi"``, or ``"u_chi_f"``.
            Mutually exclusive with the Pade J1/J2/J3 family.
        dtn_c (int, optional):
            DTN cutoff smoothness; CLI default 4.
        dtn_n_u (int, optional):
            DTN u polynomial order; CLI default 8.
        dtn_n_chi (int, optional):
            DTN chi polynomial order; CLI default 8. Its cusp constraint is
            mandatory: AE slope -Z_eff, ECP slope zero.
        dtn_n_eN (int, optional):
            DTN f electron-nucleus polynomial order; CLI default 4.
        dtn_n_ee (int, optional):
            DTN f electron-electron polynomial order; CLI default 4.
        dtn_cutoff_ratio_u (float):
            Periodic u cutoff relative to the inscribed radius; default 0.95.
        dtn_cutoff_ratio_chi (float):
            Periodic chi cutoff relative to the inscribed radius; default 0.95.
        dtn_cutoff_ratio_f (float):
            Periodic f cutoff relative to half the inscribed radius; default 0.95.
        verbosity (str, optional):
            Verbosity forwarded to the local converter.
        cleanup_patterns (list, optional):
            Files to clean after successful conversion; keep the Hamiltonian.

    Examples:
        >>> wf = QE_Workflow(
        ...     qe_save_dir="si.save",
        ...     kind="pgto+pw",
        ...     upf_paths=["Si=Si.upf"],
        ...     ecutwfc=20.0,
        ...     max_cond_number=1.0e8,
        ...     j2_parameter=1.0,
        ... )
        >>> status, out_files, out_values = wf.launch()

    See Also:
        WF_Workflow : The TREXIO (GTO / pGTO) counterpart.
        VMC_Workflow : Optimise the wavefunction produced by this step.
    """

    def __init__(
        self,
        qe_save_dir: str,
        kind: str = "pw",
        upf_paths: list[str] | None = None,
        hamiltonian_file: str = "hamiltonian_data.h5",
        ecutwfc: float | str | None = None,
        alpha_min: float | None = None,
        exp_tail_eps: float | None = PGTO_EXP_TAIL_EPS_USER_DEFAULT,
        max_cond_number: float | None = None,
        mesh_factor: float | None = None,
        spline_order: int | None = None,
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
        dtn_cutoff_ratio_u: float = DTN_DEFAULT_CUTOFF_RATIO,
        dtn_cutoff_ratio_chi: float = DTN_DEFAULT_CUTOFF_RATIO,
        dtn_cutoff_ratio_f: float = DTN_DEFAULT_CUTOFF_RATIO,
        ewald_tol: float | None = None,
        ecp_z_cores: list[float] | None = None,
        verbosity: str | None = None,
        cleanup_patterns: list | None = None,
        twist_average: bool = False,
        tiling: list[int] | None = None,
        supertwist: list[float] | None = None,
    ):
        super().__init__(cleanup_patterns=cleanup_patterns)

        if kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
        # ``upf_paths`` may be left empty: jqmc-tool qe auto-discovers
        # from ``<qe_save_dir>/data-file-schema.xml`` at run time.

        self.qe_save_dir = qe_save_dir
        self.kind = kind
        self.upf_paths = upf_paths or []
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
        if self.tiling and kind not in ("pw", "bspline", "pgto", "pgto+pw"):
            raise ValueError("tiling supports pw, bspline, pgto and pgto+pw.")
        self.hamiltonian_file = hamiltonian_file
        self.ecutwfc = ecutwfc
        self.alpha_min = alpha_min
        self.exp_tail_eps = exp_tail_eps
        self.max_cond_number = max_cond_number
        self.mesh_factor = mesh_factor
        self.spline_order = spline_order
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
        self.dtn_cutoff_ratio_u = dtn_cutoff_ratio_u
        self.dtn_cutoff_ratio_chi = dtn_cutoff_ratio_chi
        self.dtn_cutoff_ratio_f = dtn_cutoff_ratio_f
        self.ewald_tol = ewald_tol
        self.ecp_z_cores = ecp_z_cores or []
        self.verbosity = verbosity

    def validate_inputs(self, base_dir: str | None = None) -> dict:
        """Check that the QE ``*.save`` directory exists and has the expected files.

        Implements the pre-flight inspection of the QE input: the ``*.save``
        directory must contain ``data-file-schema.xml`` and at least one
        ``wfc<ik>.hdf5``. The UPF files referenced by ``upf_paths``
        (``element=path`` specs) are checked for existence too.

        Args:
            base_dir: Directory that relative paths are resolved against
                (defaults to the workflow project dir, else the cwd).

        Returns:
            dict: ``{"save_dir", "xml", "wfc_files", "upf"}`` with the
            resolved absolute paths that were found.

        Raises:
            FileNotFoundError: If the save dir, the XML, any ``wfc`` file, or
                a referenced UPF path is missing.
        """
        root = base_dir or self.project_dir or os.getcwd()

        def _resolve(p: str) -> str:
            return p if os.path.isabs(p) else os.path.join(root, p)

        save_dir = _resolve(self.qe_save_dir)
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"QE save directory not found: {save_dir}")

        xml = os.path.join(save_dir, "data-file-schema.xml")
        if not os.path.isfile(xml):
            raise FileNotFoundError(f"QE schema file not found: {xml}")

        wfc_files = sorted(
            os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("wfc") and f.endswith(".hdf5")
        )
        if not wfc_files:
            raise FileNotFoundError(f"No wfc<ik>.hdf5 plane-wave file found in {save_dir}.")

        def _check_specs(specs: list[str], flag: str) -> list[str]:
            found: list[str] = []
            for spec in specs:
                if "=" not in spec:
                    raise ValueError(f"{flag} expects 'element=path', got {spec!r}.")
                _elem, path = spec.split("=", 1)
                resolved = _resolve(path.strip())
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(f"{flag} file not found: {resolved}")
                found.append(resolved)
            return found

        upf_found = _check_specs(self.upf_paths, "--upf")

        logger.info(
            "QE inputs OK: save_dir=%s, %d wfc file(s), %d UPF.",
            save_dir,
            len(wfc_files),
            len(upf_found),
        )
        return {
            "save_dir": save_dir,
            "xml": xml,
            "wfc_files": wfc_files,
            "upf": upf_found,
        }

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        return {
            "qe_save_dir": self.qe_save_dir,
            "kind": self.kind,
            "upf_paths": self.upf_paths,
            "twist_average": self.twist_average,
            "tiling": self.tiling,
            "supertwist": self.supertwist,
            "hamiltonian_file": self.hamiltonian_file,
            "ecutwfc": self.ecutwfc,
            "alpha_min": self.alpha_min,
            "exp_tail_eps": self.exp_tail_eps,
            "max_cond_number": self.max_cond_number,
            "mesh_factor": self.mesh_factor,
            "spline_order": self.spline_order,
            "j1_parameter": self.j1_parameter,
            "j1_type": self.j1_type,
            "j2_parameter": self.j2_parameter,
            "j2_type": self.j2_type,
            "ewald_tol": self.ewald_tol,
            "ecp_z_cores": self.ecp_z_cores,
        }

    def _build_command(self) -> list[str]:
        """Build the ``jqmc-tool qe`` argv list.

        ``upf_paths`` is forwarded only when the user supplied entries
        explicitly; otherwise ``jqmc-tool qe`` auto-discovers from
        ``<qe_save_dir>/data-file-schema.xml``.
        """
        cmd: list[str] = ["jqmc-tool", "qe", self.qe_save_dir, "-k", str(self.kind)]
        cmd += ["-o", self.hamiltonian_file]
        if self.twist_average:
            cmd += ["--twist-average"]
        if self.tiling:
            cmd += ["--tiling", " ".join(str(value) for value in self.tiling)]
        if self.supertwist is not None:
            cmd += ["--supertwist", " ".join(str(value) for value in self.supertwist)]
        for spec in self.upf_paths:
            cmd += ["-u", str(spec)]
        if self.ecutwfc is not None:
            cmd += ["--ecutwfc", str(self.ecutwfc)]
        if self.alpha_min is not None:
            cmd += ["--alpha-min", str(self.alpha_min)]
        if self.exp_tail_eps != PGTO_EXP_TAIL_EPS_USER_DEFAULT:
            # The tool's own default is omitted; None disables as 0.
            cmd += ["--exp-tail-eps", "0" if self.exp_tail_eps is None else str(self.exp_tail_eps)]
        if self.max_cond_number is not None:
            cmd += ["--max-cond-number", str(self.max_cond_number)]
        if self.mesh_factor is not None:
            cmd += ["--mesh-factor", str(self.mesh_factor)]
        if self.spline_order is not None:
            cmd += ["--spline-order", str(self.spline_order)]
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
        cmd += ["--dtn-cutoff-ratio-u", str(self.dtn_cutoff_ratio_u)]
        cmd += ["--dtn-cutoff-ratio-chi", str(self.dtn_cutoff_ratio_chi)]
        cmd += ["--dtn-cutoff-ratio-f", str(self.dtn_cutoff_ratio_f)]
        if self.ewald_tol is not None:
            cmd += ["--ewald-tol", str(self.ewald_tol)]
        for z in self.ecp_z_cores:
            cmd += ["--ecp-z-core", str(z)]
        if self.verbosity is not None:
            cmd += ["--verbosity", str(self.verbosity)]
        return cmd

    async def run(self) -> tuple:
        """Run ``jqmc-tool qe`` locally via ``subprocess.run``
        (same pattern as :class:`WF_Workflow`).

        Returns:
            tuple:
                ``(status, output_files, output_values)``
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        try:
            self.validate_inputs(base_dir=_wd)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"QE input validation failed: {e}")
            self.status = WorkflowStatus.FAILED
            return self.status, [], {}

        command = self._build_command()
        logger.info(f"  Running: {' '.join(command)}")

        # Stream the subprocess's stdout / stderr line-by-line so progress
        # logs (e.g. ``build_gto_design_matrix: group N/M done in Xs``)
        # surface live instead of being buffered until completion. stderr
        # is merged into stdout via ``stderr=STDOUT`` so warnings stay in
        # order relative to stdout chatter (matters for things like
        # `Auto pgto+pw cutoff: ...` followed by per-trial diagnostics).
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
                assert proc.stdout is not None  # for type checkers
                for line in proc.stdout:
                    # Use ``logger.info`` so the parent's logger formatter /
                    # destination apply; strip trailing newline to avoid
                    # double-newlines in handlers that add their own.
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
