"""LRDMC_Workflow -- Lattice-Regularized Diffusion Monte Carlo run.

Generates an LRDMC input TOML, submits ``jqmc`` (job_type=lrdmc-bra or
job_type=lrdmc-tau) on a remote/local machine, monitors until completion,
fetches the checkpoint, and post-processes with
``jqmc-tool lrdmc compute-energy``.

Two operating modes are available:

* **GFMC_n mode** (``job_type=lrdmc-bra``) -- activated when
  *target_survived_walkers_ratio* or *num_projection_per_measurement* is set.
  Uses discrete projections per measurement.
* **GFMC_t mode** (``job_type=lrdmc-tau``) -- activated when
  *time_projection_tau* is used (default).  Uses a continuous imaginary
  time step between projections.  No calibration pilot is needed.
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

import asyncio
import glob
import os
import math
import shutil
import re
import subprocess
import time
import h5py

from logging import getLogger

from ._error_estimator import (
    _format_duration,
    estimate_additional_steps,
    estimate_required_steps,
    parse_net_time,
    read_accumulated_measurement_steps,
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import get_num_mpi, load_queue_data
from .forces_parser import parse_force_table, parse_pressure_table, parse_stress_table
from .output_parser import parse_lrdmc_output
from .state import (
    CompletionStatus,
    WorkflowStatus,
    get_estimation,
    get_job_by_step,
    read_state,
    reconcile_fetched_jobs_recursive,
    set_estimation,
    validate_completion,
)
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)

# Minimal post-processing thresholds (GFMC / LRDMC), used to reject invalid
# parameters before submitting an HPC job (lrdmc_ext_workflow imports them
# from here).
GFMC_MIN_WARMUP_STEPS = 25
GFMC_MIN_BIN_BLOCKS = 10
GFMC_MIN_COLLECT_STEPS = 5


# -- HDF5 electron count ------------------------------------------


def get_num_electrons(hamiltonian_file: str) -> int:
    """Read the total number of electrons from a hamiltonian HDF5 file.

    Args:
        hamiltonian_file (str):
            Path to ``hamiltonian_data.h5``.

    Returns:
        int:
            Total electron count ``num_electron_up + num_electron_dn``.

    Raises:
        RuntimeError:
            If the electron counts cannot be found in the file.
    """
    try:
        with h5py.File(hamiltonian_file, "r") as f:
            geminal = f["wavefunction_data/geminal_data"]
            n_up = int(geminal.attrs["num_electron_up"])
            n_dn = int(geminal.attrs["num_electron_dn"])
        return n_up + n_dn
    except Exception as e:
        raise RuntimeError(f"Cannot read electron counts from {hamiltonian_file}: {e}") from e


# -- Survived walkers ratio parsing -------------------------------

_SURVIVED_PATTERN = re.compile(r"Survived walkers ratio\s*=\s*(\d+\.?\d*)\s*%")


def parse_survived_walkers_ratio(output_file: str) -> float | None:
    """Parse the survived walkers ratio from an LRDMC output file.

    Searches for the line
    ``Survived walkers ratio = <value> %``
    and returns the **last** occurrence as a fraction (0.0-1.0).

    Args:
        output_file (str):
            Path to the jqmc stdout file.

    Returns:
        float or None:
            Survived walkers ratio as a fraction, or *None* if not found.
    """
    last_value = None
    try:
        with open(output_file) as f:
            for line in f:
                m = _SURVIVED_PATTERN.search(line)
                if m:
                    last_value = float(m.group(1)) / 100.0
    except Exception:
        return None
    return last_value


# -- Linear fitting -----------------------------------------------


def fit_num_projection_per_measurement(
    x_values: list[int],
    y_values: list[float],
    target_ratio: float,
) -> int:
    r"""Determine the optimal ``num_projection_per_measurement`` by linear fit.

    Given two or more data points
    ``(num_projection_per_measurement, survived_walkers_ratio)`` the function
    fits a linear model :math:`f(x) = a x + b` via least squares and
    solves for the *x* at which :math:`f(x) = \text{target\_ratio}`.

    Args:
        x_values (list[int]):
            ``num_projection_per_measurement`` values used in calibration runs.
        y_values (list[float]):
            Corresponding survived-walkers ratios (fractions, 0.0-1.0).
        target_ratio (float):
            Target survived-walkers ratio (e.g. 0.97).

    Returns:
        int:
            Optimal ``num_projection_per_measurement`` (rounded up to the nearest
            even integer, minimum 2).

    Raises:
        RuntimeError:
            If the linear fit cannot determine a positive root.
    """
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(x_values)}")

    # -- Fit linear y = a*x + b via least-squares (no numpy) -----
    n = len(x_values)
    sx = sum(x_values)
    sx2 = sum(xi**2 for xi in x_values)
    sy = sum(y_values)
    sxy = sum(xi * yi for xi, yi in zip(x_values, y_values))

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-30:
        raise RuntimeError("Degenerate fit: all x values are identical.")

    a = (n * sxy - sx * sy) / denom
    b = (sy * sx2 - sx * sxy) / denom

    logger.info(f"Linear fit: f(x) = {a:.6g}*x + {b:.6g}")
    for xi, yi in zip(x_values, y_values):
        fitted = a * xi + b
        logger.info(f"  nmpm={xi:>6d}: measured={yi:.4f}, fitted={fitted:.4f}")

    # Solve a*x + b = target_ratio  =>  x = (target_ratio - b) / a
    if abs(a) < 1e-15:
        raise RuntimeError(f"Linear fit slope is ~0 (a={a:.6g}). Cannot solve for target_ratio={target_ratio:.4f}.")

    x_opt = (target_ratio - b) / a
    if x_opt <= 0:
        raise RuntimeError(
            f"Linear fit gives non-positive root x={x_opt:.2f} "
            f"for target_ratio={target_ratio:.4f}. "
            f"Coefficients: a={a:.6g}, b={b:.6g}"
        )

    # Round up to nearest even integer, minimum 2
    result = max(2, int(math.ceil(x_opt)))
    if result % 2 != 0:
        result += 1

    logger.info(f"Optimal num_projection_per_measurement for target ratio {target_ratio:.4f}: raw={x_opt:.2f} -> {result}")
    return result


class LRDMC_Workflow(Workflow):
    r"""Single LRDMC (Lattice-Regularized Diffusion Monte Carlo) run.

    Generates a ``job_type=lrdmc-bra`` (GFMC_n) or ``job_type=lrdmc-tau``
    (GFMC_t) input TOML at a fixed lattice spacing ``alat``, submits
    ``jqmc``, monitors until completion, fetches the checkpoint, and
    post-processes with ``jqmc-tool lrdmc compute-energy`` to extract
    the DMC energy +/- error.

    **Mode selection** (mutually exclusive):

    * **GFMC_t** (default) -- set *time_projection_tau* (default 0.10).
      Uses continuous imaginary-time projection.  Only the error-bar
      pilot is run (no calibration phase).
    * **GFMC_n** -- set *target_survived_walkers_ratio* or
      *num_projection_per_measurement*.  Uses discrete GFMC projections.
      When *target_survived_walkers_ratio* is set (and
      *num_projection_per_measurement* is *None*), an automatic calibration
      pilot determines the optimal *num_projection_per_measurement*.

    The workflow supports two operating modes:

    **Automatic mode** (default, ``num_gfmc_projections=None``):

    1. **Pilot run** (``_0``) -- A short run with ``pilot_steps``
       measurement steps.  The resulting error estimates the steps
       required for ``target_error`` via $\sigma \propto 1/\sqrt{N}$.
       In GFMC_n mode with calibration, three additional short runs
       precede this to determine *num_projection_per_measurement*.
    2. **Production runs** (``_1``, ``_2``, ...) -- Continuation runs
       with the estimated step count.  The loop terminates when the
       error is <= ``target_error`` or ``max_continuation`` is reached.

    **Fixed-step mode** (``num_gfmc_projections`` is set):

    The error-bar pilot (``_pilot_b``) is skipped and ``target_error``
    is ignored.  If calibration is needed (GFMC_n mode with
    ``target_survived_walkers_ratio``), ``_pilot_a`` still runs.
    Each production run uses exactly ``num_gfmc_projections``
    measurement steps, and ``max_continuation`` runs are executed
    unconditionally.

    Args:
        server_machine_name (str):
            Target machine name.
        alat (float):
            Lattice discretization parameter (bohr).
        hamiltonian_file (str):
            Input ``hamiltonian_data.h5``.
        queue_label (str):
            Queue/partition label.
        jobname (str):
            Scheduler job name.
        number_of_walkers (int):
            Walkers per MPI process.
        max_time (int):
            Wall-time limit (seconds).
        num_gfmc_bin_blocks (int):
            Binning blocks for post-processing.
        num_gfmc_warmup_steps (int):
            Warmup steps to discard in post-processing.
        num_gfmc_collect_steps (int):
            Weight-collection steps for energy post-processing.
        time_projection_tau (float, optional):
            Imaginary time between branchings (Ha^-1) for GFMC_t mode,
            written as ``[lrdmc-tau] tau``.  Default ``0.10``.  Ignored when
            *target_survived_walkers_ratio* or *num_projection_per_measurement*
            is set.
        target_survived_walkers_ratio (float, optional):
            Target survived-walkers ratio for automatic
            ``num_projection_per_measurement`` calibration.  Setting this
            activates GFMC_n mode.  The pilot phase runs three short
            calculations at ``Ne*k*(0.3/alat)^2`` projections (k=2,4,6),
            fits a linear model to the observed survived-walkers ratio,
            and picks the value that achieves this target.
        num_projection_per_measurement (int, optional):
            GFMC projections per measurement (GFMC_n mode), written as
            ``[lrdmc-bra] num_mcmc_per_measurement``.  When given
            explicitly, the automatic calibration is skipped.  The key has
            no default: in GFMC_n mode it is either given here or
            calibrated through *target_survived_walkers_ratio*.
        non_local_move (str, optional):
            Non-local move treatment (``"tmove"`` or ``"dltmove"``).  Default from ``jqmc_miscs``.
        E_scf (float, optional):
            Initial energy guess for the GFMC shift (GFMC_n only).
            Default from ``jqmc_miscs``.
        atomic_force (bool, optional):
            Compute atomic forces.  Default from ``jqmc_miscs``.
        compute_stress (bool, optional):
            Compute the symmetric stress tensor (Ha/bohr^3) and the
            diagonal pressure; periodic systems only.  Supported by both
            GFMC_n (lrdmc-bra) and GFMC_t (lrdmc-tau).  When true, the
            ``stress`` / ``pressure`` tables are parsed from the ``jqmc``
            output into ``output_values``.  Default from ``jqmc_miscs``.
        use_swct (bool, optional):
            Apply Space Warp Coordinate Transformation (SWCT) to atomic forces.
            Default is False for LRDMC.
        epsilon_PW (float, optional):
            Pathak-Wagner regularization parameter (Bohr). When > 0,
            the force estimator is regularized near the nodal surface.
            Default from ``jqmc_miscs``.
        mcmc_seed (int, optional):
            Random seed for MCMC.  Default from ``jqmc_miscs``.
        verbosity (str, optional):
            Verbosity level.  Default from ``jqmc_miscs``.
        poll_interval (int):
            Seconds between job-status polls, and between
            submission attempts while the queue's ``max_job_submit`` is reached.
        target_error (float):
            Target statistical error (Ha).
        pilot_steps (int):
            Measurement steps for the pilot estimation run.
        num_gfmc_projections (int, optional):
            Fixed number of measurement steps per production run.
            When set, the error-bar pilot (``_pilot_b``) is skipped,
            ``target_error`` is ignored, and all ``max_continuation``
            production runs are executed unconditionally.  Calibration
            (``_pilot_a``) still runs when needed (GFMC_n mode with
            ``target_survived_walkers_ratio``).  Default *None*
            (automatic mode).
        pilot_queue_label (str, optional):
            Queue label for the pilot run.  Defaults to *queue_label*.
            Use a shorter/smaller queue for the pilot to save resources.
        max_continuation (int):
            Maximum number of production runs after the pilot.
        cleanup_patterns (list[str], optional):
            Glob patterns for files to delete after successful completion
            (e.g. ``["restart.h5", "hamiltonian_opt*.h5"]``).  Local files
            are always removed; remote files are removed only when the
            workflow targets a remote machine.  Default *None* (no cleanup).
        precision_mode (str):
            Precision mode written to the ``[precision]`` section
            (``mode``) of the generated TOML: ``"full"`` (all float64) or
            ``"mixed"`` (mixed precision).  Default ``"full"``.
        mpc (bool, optional):
            Enable the passive MPC (Model Periodic Coulomb) finite-size
            correction (``[coulomb] mpc``).  MPC is an energy-only
            observable: it does not enter the sampled Hamiltonian, the
            branching weight, ``e_L``, forces or parameter gradients; only
            the reported total energy is corrected.  ``True`` enables it,
            ``False`` disables it, *None* (default) omits the key so the
            jqmc default applies.
        mpc_cutoff (float, optional):
            Reciprocal-space kinetic-energy cutoff (Ha) for the MPC
            reciprocal-space sum (``[coulomb] mpc_cutoff``).  A convergence
            parameter.  *None* (default) omits the key so the driver
            default applies.

    Examples:
        GFMC_t mode (default)::

            wf = LRDMC_Workflow(
                server_machine_name="cluster",
                alat=0.3,
                target_error=0.0005,
                number_of_walkers=8,
            )
            status, files, values = wf.launch()
            print(values["energy"], values["energy_error"])

        GFMC_n mode with calibration::

            wf = LRDMC_Workflow(
                server_machine_name="cluster",
                alat=0.3,
                target_error=0.0005,
                target_survived_walkers_ratio=0.97,
                number_of_walkers=8,
            )

        Fixed-step mode (skip error-bar pilot)::

            wf = LRDMC_Workflow(
                server_machine_name="cluster",
                alat=0.3,
                num_gfmc_projections=500,
                max_continuation=3,
                number_of_walkers=8,
            )

        As part of a :class:`Launcher` pipeline::

            enc = Container(
                label="lrdmc-a0.30",
                dirname="03_lrdmc",
                input_files=[FileFrom("mcmc-run", "hamiltonian_data.h5")],
                workflow=LRDMC_Workflow(
                    server_machine_name="cluster",
                    alat=0.3,
                    target_error=0.001,
                ),
            )

    Output Values:
        energy (float):
            DMC energy (Ha).
        energy_error (float):
            Statistical error on ``energy`` (Ha).
        alat (float):
            Lattice spacing used for this run.
        restart_chk (str):
            Basename of the restart checkpoint file.
        forces (object):
            Atomic forces (only when ``atomic_force=True``).
        estimated_steps (int):
            Estimated total measurement steps.
        num_projection_per_measurement (int):
            Number of GFMC projections per measurement
            (GFMC_n mode only).
        time_projection_tau (float):
            Imaginary-time projection step (GFMC_t mode only).

    Notes:
        * For a^2->0 continuum-limit extrapolation, use
          :class:`LRDMC_Ext_Workflow` instead.
        * The pilot is skipped on re-entrance if an estimation already
          exists in ``workflow_state.toml``.

    See Also:
        LRDMC_Ext_Workflow : Multi-alat extrapolation wrapper.
        MCMC_Workflow : VMC production sampling (job_type=mcmc).
        VMC_Workflow : Wavefunction optimisation (job_type=vmc).
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        alat: float = 0.30,
        hamiltonian_file: str = "hamiltonian_data.h5",
        queue_label: str = "default",
        jobname: str = "jqmc-lrdmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        num_gfmc_bin_blocks: int = 10,
        num_gfmc_warmup_steps: int = 25,
        num_gfmc_collect_steps: int = 5,
        # -- [lrdmc-bra / lrdmc-tau] section parameters --
        time_projection_tau: float | None = 0.10,
        target_survived_walkers_ratio: float | None = None,
        num_projection_per_measurement: int | None = None,
        non_local_move: str | None = None,
        E_scf: float | None = None,
        atomic_force: bool | None = None,
        compute_stress: bool | None = None,
        use_swct: bool | None = None,
        epsilon_PW: float | None = None,
        # -- [control] section parameters --
        mcmc_seed: int | None = None,
        verbosity: str | None = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        pilot_steps: int = 100,
        num_gfmc_projections: int | None = None,
        pilot_queue_label: str | None = None,
        max_continuation: int = 1,
        cleanup_patterns: list | None = None,
        # -- [precision] section --
        precision_mode: str = "full",
        # -- [coulomb] section --
        # MPC (Model Periodic Coulomb) finite-size correction. Under the current
        # design MPC is PASSIVE (``physical = no``): it is an energy-only
        # observable. It does NOT enter the Hamiltonian, the branching weight,
        # ``e_L``, ``aF`` or ``gF`` -- forces and parameter gradients stay on the
        # Ewald channel. Only the REPORTED total energy is corrected
        # (``E_MPC = E_Ewald + dE_mpc``).
        #
        # ``mpc``: True enables the correction, False explicitly disables it,
        # None omits the key (and, with ``mpc_cutoff`` also None, the whole
        # ``[coulomb]`` section) from the generated TOML -- the jqmc default.
        mpc: bool | None = None,
        # ``mpc_cutoff``: reciprocal-space kinetic-energy cutoff in HARTREE for
        # the MPC reciprocal-space sum; default 30.0 (matching CASINO's
        # ``mpc_cutoff`` and QMCPACK's ``cutoff``). This is a CONVERGENCE
        # parameter, not a physical one: DOUBLE it until the correction stops
        # changing. None omits the key from the TOML (driver default).
        mpc_cutoff: float | None = None,
    ):
        super().__init__(cleanup_patterns=cleanup_patterns)
        self.server_machine_name = server_machine_name
        self.alat = alat
        self.hamiltonian_file = hamiltonian_file
        self.queue_label = queue_label
        self.jobname = jobname
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        # Validate GFMC post-processing parameters
        if num_gfmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            raise ValueError(
                f"num_gfmc_warmup_steps ({num_gfmc_warmup_steps}) must be "
                f">= {GFMC_MIN_WARMUP_STEPS}. E_scf is unreliable before "
                f"step {GFMC_MIN_WARMUP_STEPS}."
            )
        if num_gfmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            raise ValueError(f"num_gfmc_bin_blocks ({num_gfmc_bin_blocks}) must be >= {GFMC_MIN_BIN_BLOCKS}.")
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            raise ValueError(f"num_gfmc_collect_steps ({num_gfmc_collect_steps}) must be >= {GFMC_MIN_COLLECT_STEPS}.")
        self.num_gfmc_bin_blocks = num_gfmc_bin_blocks
        self.num_gfmc_warmup_steps = num_gfmc_warmup_steps
        self.num_gfmc_collect_steps = num_gfmc_collect_steps
        # Mode selection: GFMC_n vs GFMC_t
        #   GFMC_n: target_survived_walkers_ratio or num_projection_per_measurement is set
        #   GFMC_t: otherwise (uses time_projection_tau)
        self._use_gfmc_n = target_survived_walkers_ratio is not None or num_projection_per_measurement is not None
        # [lrdmc-bra / lrdmc-tau] section
        self.time_projection_tau = time_projection_tau
        self.target_survived_walkers_ratio = target_survived_walkers_ratio
        self.num_projection_per_measurement = num_projection_per_measurement
        self.non_local_move = non_local_move
        self.E_scf = E_scf
        self.atomic_force = atomic_force
        self.compute_stress = compute_stress
        # Stress is supported on both LRDMC drivers: lrdmc-tau (GFMC_t) and
        # lrdmc-bra (GFMC_n) expose the same get_aT / get_aP estimators.
        self.use_swct = use_swct
        self.epsilon_PW = epsilon_PW
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        self.pilot_steps = pilot_steps
        self.num_gfmc_projections = num_gfmc_projections
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation
        # [precision] section
        self.precision_mode = precision_mode
        # [coulomb] section (passive MPC energy-only observable)
        self.mpc = mpc
        self.mpc_cutoff = mpc_cutoff

    @property
    def job_type(self) -> str:
        """Return the jqmc job type string for TOML generation."""
        return "lrdmc-bra" if self._use_gfmc_n else "lrdmc-tau"

    # -- Input generation ------------------------------------------

    def _generate_input(
        self,
        num_steps,
        input_file,
        restart=False,
        restart_chk=None,
        num_projection_per_measurement=None,
    ):
        """Generate LRDMC TOML input file.

        Args:
            num_steps (int):
                Number of measurement steps.
            input_file (str):
                Output filename.
            restart (bool):
                Whether this is a restart run.
            restart_chk (str or None):
                Restart checkpoint filename.
            num_projection_per_measurement (int or None):
                Override for GFMC projections per measurement (GFMC_n only).
                Falls back to ``self.num_projection_per_measurement``. In GFMC_n
                mode one of the two must be set (given or calibrated): the
                key has no default, so ``ValueError`` is raised otherwise.
        """
        jt = self.job_type
        control_ov = resolve_with_defaults(
            "control",
            {
                "number_of_walkers": self.number_of_walkers,
                "max_time": self.max_time,
                "hamiltonian_h5": self.hamiltonian_file,
                "restart": restart,
                "mcmc_seed": self.mcmc_seed,
                "verbosity": self.verbosity,
            },
        )
        if restart and restart_chk:
            control_ov["restart_chk"] = restart_chk

        if self._use_gfmc_n:
            # GFMC_n mode: job_type="lrdmc-bra"
            nmpm = num_projection_per_measurement or self.num_projection_per_measurement
            if nmpm is None:
                # [lrdmc-bra] num_mcmc_per_measurement is required and has no
                # default: it comes from the user or from the calibration.
                raise ValueError(
                    "GFMC_n (lrdmc-bra) input needs num_projection_per_measurement "
                    "([lrdmc-bra] num_mcmc_per_measurement), which has no default: pass it "
                    "explicitly, or set target_survived_walkers_ratio and let the calibration "
                    "(_pilot_a) choose it before the input is written."
                )
            section_ov = resolve_with_defaults(
                "lrdmc-bra",
                {
                    "num_mcmc_steps": num_steps,
                    "alat": self.alat,
                    "num_mcmc_per_measurement": nmpm,
                    "non_local_move": self.non_local_move,
                    "num_gfmc_warmup_steps": self.num_gfmc_warmup_steps,
                    "num_gfmc_bin_blocks": self.num_gfmc_bin_blocks,
                    "num_gfmc_collect_steps": self.num_gfmc_collect_steps,
                    # E_scf seeds the GFMC energy shift on the FIRST
                    # submission only; on restart it is stripped from the
                    # written TOML entirely via the omit_keys= call below
                    # (see the comment at that call site).
                    "E_scf": self.E_scf,
                    "atomic_force": self.atomic_force,
                    "compute_stress": self.compute_stress,
                    "use_swct": self.use_swct,
                    "epsilon_PW": self.epsilon_PW,
                },
            )
        else:
            # GFMC_t mode: job_type="lrdmc-tau"
            section_ov = resolve_with_defaults(
                "lrdmc-tau",
                {
                    "num_mcmc_steps": num_steps,
                    "tau": self.time_projection_tau,
                    "alat": self.alat,
                    "non_local_move": self.non_local_move,
                    "num_gfmc_warmup_steps": self.num_gfmc_warmup_steps,
                    "num_gfmc_bin_blocks": self.num_gfmc_bin_blocks,
                    "num_gfmc_collect_steps": self.num_gfmc_collect_steps,
                    "atomic_force": self.atomic_force,
                    "compute_stress": self.compute_stress,
                    "use_swct": self.use_swct,
                    "epsilon_PW": self.epsilon_PW,
                },
            )

        overrides = {
            "control": control_ov,
            jt: section_ov,
        }
        # Add [precision] section
        overrides["precision"] = {"mode": self.precision_mode}
        # Add [coulomb] keys only when explicitly set; a None value omits that
        # key (and, when both are None, the whole section) -- bit-identical to
        # a TOML with no MPC at all. MPC is energy-only (``physical = no``):
        # nothing here changes e_L, the branching weight, aF or gF.
        coulomb_ov = {}
        if self.mpc is not None:
            coulomb_ov["mpc"] = bool(self.mpc)
        if self.mpc_cutoff is not None:
            coulomb_ov["mpc_cutoff"] = float(self.mpc_cutoff)
        if coulomb_ov:
            overrides["coulomb"] = coulomb_ov

        # E_scf ("frozen on restart") is only a valid SEED for the first
        # submission: the GFMC_n driver self-consistently updates it via
        # population control and freezes the drifted value into the
        # checkpoint. A continuation TOML that re-sends the (necessarily
        # stale) static self.E_scf trips jqmc's
        # assert_restart_frozen_consistency -- omit the key entirely so the
        # checkpoint's evolved value carries forward untouched (an omitted
        # key skips that check; see jqmc._restart_policy).
        omit_keys = {"lrdmc-bra": ["E_scf"]} if (restart and self._use_gfmc_n) else None

        generate_input_toml(
            job_type=jt,
            overrides=overrides,
            filename=input_file,
            omit_keys=omit_keys,
        )

    # -- Submit / poll / fetch -------------------------------------
    # _submit_and_wait() and _make_job() are inherited from Workflow.

    def _get_command(self, input_file: str) -> str:
        """LRDMC reads a TOML; the script line is ``mpirun -np N jqmc <input>``."""
        return f"jqmc {input_file}"

    # -- configure / run ------------------------------------------

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        return {
            "jobname": self.jobname,
            "alat": self.alat,
            "target_error": self.target_error,
            "hamiltonian_file": self.hamiltonian_file,
            "server_machine": self.server_machine_name,
            "number_of_walkers": self.number_of_walkers,
            "max_time": self.max_time,
            "pilot_steps": self.pilot_steps,
            "max_continuation": self.max_continuation,
        }

    def can_resume_after_completed(self, proj_dir: str) -> bool:
        """Return True when a re-launch could still reduce ``energy_error`` toward target.

        A prior ``"completed"`` state may have been written by
        :meth:`_launch_auto` after exhausting ``max_continuation`` without
        meeting ``target_error``.  When the user raises ``max_continuation``
        (or tightens ``target_error``) and relaunches, the recorded
        ``[result].energy_error`` will still exceed ``target_error*1.20``
        and this method returns True so ``Container`` bypasses its
        short-circuit and re-enters the production loop.

        Fixed-step mode (``num_gfmc_projections`` set) has no convergence
        criterion and is never resumed automatically.
        """
        if self.target_error is None or self.num_gfmc_projections is not None:
            return False
        result = read_state(proj_dir).get("result", {})
        err = result.get("energy_error")
        if err is None:
            return False
        return err > self.target_error * 1.20

    async def run(self) -> tuple:
        """Run the LRDMC workflow.

        **Fixed-step mode** (``num_gfmc_projections`` is set):
        The error-bar pilot (``_pilot_b``) is skipped.  Calibration
        (``_pilot_a``) still runs if needed.  Each production run
        uses exactly ``num_gfmc_projections`` steps and all
        ``max_continuation`` runs are executed unconditionally.

        **Automatic mode** (``num_gfmc_projections`` is *None*, default):

        1. Calibration pilot (``_pilot_a``, GFMC_n only) -- Three short
           LRDMC runs to determine ``num_projection_per_measurement``.
        2. Error-bar pilot (``_pilot_b``) -- estimates production steps.
        3. Production runs (``_1``, ``_2``, ...) -- accumulate statistics
           until ``target_error`` is achieved or ``max_continuation``
           is reached.
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        # Reconcile any orphaned "submitted"/"completed" job records whose
        # output file already landed locally with a "Program ends" marker
        # (e.g. workflow killed between job completion and fetch-finalize).
        # Walks pilot subdirectories (``_pilot_a/``, ``_pilot_b/``) too,
        # since each carries its own state file.  Without this, Phase A
        # would break out at the orphan record and the safety-net energy
        # computation would read a stale earlier step.
        n_reconciled = reconcile_fetched_jobs_recursive(_wd)
        if n_reconciled:
            logger.info(f"  Reconciled {n_reconciled} job record(s) to 'fetched' from existing output.")

        # -- Fixed-step mode ---------------------------------------
        if self.num_gfmc_projections is not None:
            return await self._launch_fixed_steps(_wd)

        # -- Automatic mode (pilot + target_error) -----------------
        return await self._launch_auto(_wd)

    async def _launch_fixed_steps(self, _wd):
        """Fixed-step production: skip error-bar pilot, ignore target_error.

        Calibration (``_pilot_a``) still runs when needed (GFMC_n mode
        with ``target_survived_walkers_ratio``).
        """
        estimated_steps = self.num_gfmc_projections

        # -- Phase A: calibrate num_projection_per_measurement (GFMC_n only) --
        need_calibration = (
            self._use_gfmc_n and self.num_projection_per_measurement is None and self.target_survived_walkers_ratio is not None
        )
        if need_calibration:
            await self._run_calibration(_wd)

        mode_info = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
        logger.info("")
        logger.info("-- LRDMC Fixed-step mode " + "-" * 26)
        logger.info(
            f"  num_gfmc_projections = {estimated_steps}\n"
            f"  max_continuation     = {self.max_continuation}\n"
            f"  mode                 = {'GFMC_n' if self._use_gfmc_n else 'GFMC_t'}\n"
            f"  {mode_info}"
        )

        step_files = {}  # {step: (input, output, run_id)}
        last_run = 0
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")

            if status_i == "fetched":
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                logger.info(f"  step {i}: already fetched. Skipping.")
                last_run = i
                continue
            if status_i in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status_i}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    self._generate_input(estimated_steps, os.path.join(_wd, input_i))
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- LRDMC Production run {i}/{self.max_continuation} (a={self.alat}) " + "-" * 10)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            step_files[i] = (input_i, output_i, run_id_i)
            restart_chk = self._find_restart_chk(_wd) if i > 1 else None
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
                step=i,
                run_id=run_id_i,
            )
            last_run = i

            # Post-process energy (informational only, no convergence check)
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error, dE_mpc, dE_mpc_error = self._compute_energy(restart_chk, work_dir=_wd, with_mpc=True)
                if energy is not None:
                    self.output_values["energy"] = energy
                    self.output_values["energy_error"] = error
                    self.output_values["alat"] = self.alat
                    self.output_values["restart_chk"] = restart_chk
                    logger.info(f"  LRDMC energy (a={self.alat}): {energy} +- {error} Ha")
                    self._record_mpc_correction(dE_mpc, dE_mpc_error)
                    if dE_mpc is not None:
                        logger.info(f"  MPC correction: dE_MPC = {dE_mpc} +- {dE_mpc_error} Ha")
                    if self.atomic_force:
                        forces = self._compute_force(restart_chk, work_dir=_wd, output_file=output_i)
                        if forces is not None:
                            self.output_values["forces"] = forces
                    if self.compute_stress:
                        stress, pressure = self._compute_stress(work_dir=_wd, output_file=output_i)
                        if stress is not None:
                            self.output_values["stress"] = stress
                        if pressure is not None:
                            self.output_values["pressure"] = pressure
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_dE_mpc=dE_mpc,
                        last_dE_mpc_error=dE_mpc_error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )

            # -- Abnormal-termination guard (single source of truth) --
            # Fixed-step mode has no convergence criterion, so only the
            # Program-ends / non-finite-energy / missing-energy checks are
            # active here.
            vstatus, vmsg = validate_completion(_wd, self.output_values, require_energy=True)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

        # -- Final energy computation -----------------------------
        last_output = step_files[last_run][1] if last_run in step_files else None
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error, dE_mpc, dE_mpc_error = self._compute_energy(restart_chk, work_dir=_wd, with_mpc=True)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["alat"] = self.alat
                self.output_values["restart_chk"] = restart_chk
                self._record_mpc_correction(dE_mpc, dE_mpc_error)
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd, output_file=last_output)
                    if forces is not None:
                        self.output_values["forces"] = forces
                if self.compute_stress:
                    stress, pressure = self._compute_stress(work_dir=_wd, output_file=last_output)
                    if stress is not None:
                        self.output_values["stress"] = stress
                    if pressure is not None:
                        self.output_values["pressure"] = pressure
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                    last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                    last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                )

        # -- Collect outputs ---------------------------------------
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps
        if self._use_gfmc_n:
            self.output_values["num_projection_per_measurement"] = self.num_projection_per_measurement
        else:
            self.output_values["time_projection_tau"] = self.time_projection_tau
            avg_nmpm = self._resolve_avg_nmpm(_wd)
            if avg_nmpm is not None:
                self.output_values["num_projection_per_measurement"] = avg_nmpm

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    async def _run_calibration(self, _wd):
        """Phase A: calibrate num_projection_per_measurement (GFMC_n only)."""
        h5_src = os.path.join(_wd, self.hamiltonian_file)
        pilot_a_dir = os.path.join(_wd, "_pilot_a")
        os.makedirs(pilot_a_dir, exist_ok=True)

        n_electrons = get_num_electrons(os.path.join(_wd, self.hamiltonian_file))
        alat_scale = (0.3 / self.alat) ** 2
        trial_nmpm = [int(n_electrons * k * alat_scale) for k in (2, 4, 6)]

        logger.info("")
        logger.info(f"-- LRDMC calibration: num_projection_per_measurement (a={self.alat}) " + "-" * 8)
        logger.info(
            f"  Ne={n_electrons}, trial nmpm = {trial_nmpm}, target survived ratio = {self.target_survived_walkers_ratio:.2%}"
        )

        # Run 3 calibration jobs in parallel sub-directories
        calib_tasks = []
        for idx, nmpm_val in enumerate(trial_nmpm, start=1):
            calib_sub = os.path.join(pilot_a_dir, f"_pilot{idx}")
            os.makedirs(calib_sub, exist_ok=True)
            h5_link = os.path.join(calib_sub, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_link):
                try:
                    os.symlink(h5_src, h5_link)
                except OSError:
                    shutil.copy(h5_src, h5_link)

            async def _run_calib(sub_dir, nmpm_v, _idx=idx):
                rec = get_job_by_step(sub_dir, 0)
                status_c = rec.get("status")
                if status_c in ("fetched", "submitted", "completed"):
                    inp_f = rec["input_file"]
                    out_f = rec["output_file"]
                    run_id_c = rec.get("run_id", "")
                    if status_c == "fetched":
                        logger.info(f"  {inp_f} (nmpm={nmpm_v}): already fetched. Skipping.")
                    else:
                        logger.info(f"  {inp_f} (nmpm={nmpm_v}): already {status_c}. Resuming...")
                else:
                    run_id_c = self._new_run_id()
                    inp_f = self._input_filename(0, run_id_c)
                    out_f = self._output_filename(0, run_id_c)
                    self._generate_input(
                        self.pilot_steps,
                        os.path.join(sub_dir, inp_f),
                        num_projection_per_measurement=nmpm_v,
                    )
                logger.info(f"  _pilot{_idx}: nmpm={nmpm_v}, {self.pilot_steps} steps")
                await self._submit_and_wait(
                    inp_f,
                    out_f,
                    work_dir=sub_dir,
                    queue_label=self.pilot_queue_label,
                    step=0,
                    run_id=run_id_c,
                )
                # Parse survived walkers ratio from output
                out_path = os.path.join(sub_dir, out_f)
                ratio = parse_survived_walkers_ratio(out_path)
                logger.info(f"  _pilot{_idx} (nmpm={nmpm_v}): survived ratio = {ratio:.4f}" if ratio else "N/A")
                return nmpm_v, ratio

            calib_tasks.append(asyncio.create_task(_run_calib(calib_sub, nmpm_val)))

        calib_results = await asyncio.gather(*calib_tasks)

        # Collect results and fit
        x_vals = []
        y_vals = []
        for nmpm_v, ratio in calib_results:
            if ratio is not None:
                x_vals.append(nmpm_v)
                y_vals.append(ratio)

        if len(x_vals) < 2:
            raise RuntimeError(
                f"Only {len(x_vals)} calibration runs returned a survived walkers ratio. Need at least 2 for linear fit."
            )

        optimal_nmpm = fit_num_projection_per_measurement(
            x_vals,
            y_vals,
            self.target_survived_walkers_ratio,
        )
        self.num_projection_per_measurement = optimal_nmpm

        logger.info("")
        logger.info(f"-- LRDMC Calibration Summary (a={self.alat}) " + "-" * 18)
        for xv, yv in zip(x_vals, y_vals):
            logger.info(f"  nmpm={xv:>6d}: survived ratio = {yv:.4f}")
        logger.info(f"  target ratio  = {self.target_survived_walkers_ratio:.4f}")
        logger.info(f"  optimal nmpm  = {optimal_nmpm}")
        logger.info("-" * 50)

    async def _launch_auto(self, _wd):
        """Automatic mode: pilot + target_error convergence."""
        estimation = get_estimation(_wd)

        if estimation.get("estimated_steps") is not None:
            estimated_steps = int(estimation["estimated_steps"])
            # Restore calibrated nmpm from saved state (GFMC_n only)
            if self._use_gfmc_n and estimation.get("num_projection_per_measurement") is not None:
                self.num_projection_per_measurement = int(estimation["num_projection_per_measurement"])
            mode_str = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info(
                f"Estimation already done (continuation): estimated_steps={estimated_steps}, {mode_str}. Skipping pilot."
            )
        else:
            # -- Phase A: calibrate num_projection_per_measurement (GFMC_n only) --
            h5_src = os.path.join(_wd, self.hamiltonian_file)

            need_calibration = (
                self._use_gfmc_n
                and self.num_projection_per_measurement is None
                and self.target_survived_walkers_ratio is not None
            )

            if need_calibration:
                await self._run_calibration(_wd)

            # -- Phase B: error-bar pilot (_pilot_b) ---------------
            pilot_b_dir = os.path.join(_wd, "_pilot_b")
            os.makedirs(pilot_b_dir, exist_ok=True)
            h5_link_b = os.path.join(pilot_b_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_link_b):
                try:
                    os.symlink(h5_src, h5_link_b)
                except OSError:
                    shutil.copy(h5_src, h5_link_b)

            input_pb = None
            output_pb = None
            run_id_pb = ""

            recorded_pb = get_job_by_step(pilot_b_dir, 0)
            status_pb = recorded_pb.get("status")
            if status_pb in ("fetched", "submitted", "completed"):
                input_pb = recorded_pb["input_file"]
                output_pb = recorded_pb["output_file"]
                run_id_pb = recorded_pb.get("run_id", "")
                if status_pb == "fetched":
                    logger.info(f"  {input_pb}: already fetched. Skipping pilot_b.")
                else:
                    logger.info(f"  {input_pb}: already {status_pb}. Resuming...")
            else:
                run_id_pb = self._new_run_id()
                input_pb = self._input_filename(0, run_id_pb)
                output_pb = self._output_filename(0, run_id_pb)
                self._generate_input(self.pilot_steps, os.path.join(pilot_b_dir, input_pb))
            mode_info = f"nmpm={self.num_projection_per_measurement}" if self._use_gfmc_n else f"tau={self.time_projection_tau}"
            logger.info("")
            logger.info(f"-- LRDMC error-bar pilot (a={self.alat}) " + "-" * 12)
            logger.info(f"  {input_pb}: {self.pilot_steps} steps, {mode_info} (queue: {self.pilot_queue_label})")

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(
                input_pb, output_pb, work_dir=pilot_b_dir, queue_label=self.pilot_queue_label, step=0, run_id=run_id_pb
            )
            pilot_wall_sec = time.monotonic() - pilot_t0

            restart_chk = self._find_restart_chk(pilot_b_dir)
            if not restart_chk:
                raise RuntimeError("No checkpoint found after pilot run. Cannot estimate required steps.")

            _, pilot_error = self._compute_energy(restart_chk, work_dir=pilot_b_dir)
            if pilot_error is None:
                raise RuntimeError("Could not parse energy error from pilot run.")

            # Walker ratio: pilot queue may have different MPI count
            pilot_qd = load_queue_data(self.server_machine_name, self.pilot_queue_label)
            prod_qd = load_queue_data(self.server_machine_name, self.queue_label)
            pilot_mpi = get_num_mpi(pilot_qd)
            prod_mpi = get_num_mpi(prod_qd)
            walker_ratio = pilot_mpi / prod_mpi

            estimated_steps = estimate_required_steps(
                self.pilot_steps - self.num_gfmc_warmup_steps,
                pilot_error,
                self.target_error,
                walker_ratio=walker_ratio,
                min_steps=self.num_gfmc_bin_blocks,
            )
            # Add warmup back: production also discards warmup steps
            estimated_steps += self.num_gfmc_warmup_steps

            # Time estimate: only Net time scales with step count
            step_ratio = estimated_steps / self.pilot_steps if self.pilot_steps > 0 else 0
            pilot_output_path = os.path.join(pilot_b_dir, output_pb)
            net_pilot_sec = parse_net_time(pilot_output_path)
            if net_pilot_sec is not None and net_pilot_sec > 0:
                overhead_sec = pilot_wall_sec - net_pilot_sec
                est_prod_sec = overhead_sec + net_pilot_sec * step_ratio
            else:
                est_prod_sec = pilot_wall_sec * step_ratio
            logger.info("")
            logger.info(f"-- LRDMC Step Estimation Summary (a={self.alat}) " + "-" * 8)
            logger.info(
                f"  pilot steps       = {self.pilot_steps}\n"
                f"  warmup steps      = {self.num_gfmc_warmup_steps}\n"
                f"  pilot error       = {pilot_error:.6g} Ha\n"
                f"  target error      = {self.target_error:.6g} Ha\n"
                f"  mode              = {'GFMC_n' if self._use_gfmc_n else 'GFMC_t'}\n"
                f"  {'nmpm' if self._use_gfmc_n else 'tau':17s} = {self.num_projection_per_measurement if self._use_gfmc_n else self.time_projection_tau}\n"
                f"  pilot MPI procs   = {pilot_mpi}\n"
                f"  prod. MPI procs   = {prod_mpi}\n"
                f"  walker ratio      = {walker_ratio:.4g}\n"
                f"  estimated steps   = {estimated_steps}\n"
                f"  pilot wall time   = "
                f"{_format_duration(pilot_wall_sec)}\n"
                f"  pilot net time    = "
                f"{_format_duration(net_pilot_sec) if net_pilot_sec else 'N/A'}\n"
                f"  est. prod. time   = "
                f"{_format_duration(est_prod_sec)}"
            )
            logger.info("-" * 50)

            # Save estimation to main working directory
            est_kwargs = dict(
                pilot_steps=self.pilot_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_steps=estimated_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
                pilot_wall_sec=pilot_wall_sec,
                net_pilot_sec=net_pilot_sec or 0,
            )
            if self._use_gfmc_n:
                est_kwargs["num_projection_per_measurement"] = self.num_projection_per_measurement
            else:
                est_kwargs["time_projection_tau"] = self.time_projection_tau
            set_estimation(_wd, **est_kwargs)

        # -- Re-compute energy if post-processing parameters changed --
        _postproc_changed = (
            estimation.get("last_num_gfmc_bin_blocks") != self.num_gfmc_bin_blocks
            or estimation.get("last_num_gfmc_warmup_steps") != self.num_gfmc_warmup_steps
            or estimation.get("last_num_gfmc_collect_steps") != self.num_gfmc_collect_steps
        )
        if _postproc_changed and estimation.get("last_energy") is not None:
            logger.info(
                "  Post-processing parameters changed "
                f"(bin_blocks: {estimation.get('last_num_gfmc_bin_blocks')}"
                f" -> {self.num_gfmc_bin_blocks}, "
                f"warmup: {estimation.get('last_num_gfmc_warmup_steps')}"
                f" -> {self.num_gfmc_warmup_steps}, "
                f"collect: {estimation.get('last_num_gfmc_collect_steps')}"
                f" -> {self.num_gfmc_collect_steps}); "
                "re-computing energy."
            )
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk:
                energy, error, dE_mpc, dE_mpc_error = self._compute_energy(restart_chk, work_dir=_wd, with_mpc=True)
                if energy is not None:
                    set_estimation(
                        _wd,
                        last_energy=energy,
                        last_energy_error=error,
                        last_dE_mpc=dE_mpc,
                        last_dE_mpc_error=dE_mpc_error,
                        last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                        last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                        last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                    )
                    estimation = get_estimation(_wd)

        # -- Early exit if target already met ----------------------
        cached_energy = estimation.get("last_energy")
        cached_error = estimation.get("last_energy_error")
        if cached_energy is not None and cached_error is not None:
            if cached_error <= self.target_error * 1.20:
                restart_chk = self._find_restart_chk(_wd)
                logger.info(
                    f"  Target already achieved (cached): {cached_error:.6g} <= {self.target_error * 1.20:.6g} Ha (target*1.20)"
                )
                # Mode-specific key: avoid writing None (which TOML
                # silently drops, breaking downstream readers).
                if self._use_gfmc_n:
                    mode_extras = {"num_projection_per_measurement": self.num_projection_per_measurement}
                else:
                    mode_extras = {"time_projection_tau": self.time_projection_tau}
                    avg_nmpm = self._resolve_avg_nmpm(_wd)
                    if avg_nmpm is not None:
                        mode_extras["num_projection_per_measurement"] = avg_nmpm
                self.output_values.update(
                    energy=cached_energy,
                    energy_error=cached_error,
                    alat=self.alat,
                    restart_chk=restart_chk or "",
                    estimated_steps=estimated_steps,
                    **mode_extras,
                )
                # Cached correction, written by the same set_estimation
                # call as cached_energy (absent for a non-MPC run).
                self._record_mpc_correction(estimation.get("last_dE_mpc"), estimation.get("last_dE_mpc_error"))
                if self.atomic_force and restart_chk:
                    forces = self._compute_force(restart_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                if self.compute_stress and restart_chk:
                    stress, pressure = self._compute_stress(work_dir=_wd)
                    if stress is not None:
                        self.output_values["stress"] = stress
                    if pressure is not None:
                        self.output_values["pressure"] = pressure
                self.output_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
                self.status = WorkflowStatus.COMPLETED
                return self.status, self.output_files, self.output_values

        # -- Production runs ---------------------------------------
        # Three phases:
        #   A. Scan existing runs -> find resume point
        #   B. Re-estimate from accumulated data (if resuming)
        #   C. Production loop for remaining runs
        #
        # Checkpoint preserves all accumulated statistics across
        # production restarts.  accumulated_measurement tracks
        # total measurement steps (excluding warmup) and is
        # persisted in estimation state for correct resume.

        warmup = self.num_gfmc_warmup_steps
        step_files = {}  # {step: (input, output, run_id)}
        last_run = 0
        first_new_run = self.max_continuation + 1  # assume all done

        # -- Phase A: scan existing runs --
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")
            if status_i == "fetched":
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                logger.info(f"  step {i}: already fetched. Skipping.")
                last_run = i
            else:
                first_new_run = i
                break

        # -- Phase B: re-estimate from accumulated data --
        accumulated_measurement = 0  # measurement steps only (excl. warmup)
        if first_new_run > 1:
            _re_chk = self._find_restart_chk(_wd)
            if _re_chk is None:
                raise RuntimeError(
                    f"Re-estimating from earlier runs: {first_new_run - 1} prior run(s) marked fetched but no restart checkpoint found in {_wd}."
                )
            # mcmc_counter in restart.h5 is the only trustworthy source
            # for accumulated samples (planned step counts over-count
            # when prior runs were cut short by max_time).
            actual = read_accumulated_measurement_steps(
                os.path.join(_wd, _re_chk),
                warmup=warmup,
                collect_steps=self.num_gfmc_collect_steps,
            )
            if actual is None:
                raise RuntimeError(f"Re-estimating from earlier runs: cannot read mcmc_counter from {_re_chk} in {_wd}.")
            accumulated_measurement = actual

            _re_energy, _re_error, _re_dE_mpc, _re_dE_mpc_error = self._compute_energy_cached(
                _re_chk, work_dir=_wd, accumulated=actual, with_mpc=True
            )
            if _re_energy is None or _re_error is None:
                raise RuntimeError(
                    f"Re-estimating from earlier runs: compute-energy failed for {_re_chk} in {_wd}. Cannot decide whether to resume or stop."
                )
            if _re_error <= self.target_error * 1.20:
                logger.info(f"  Target already met after prior runs: {_re_error:.6g} <= {self.target_error * 1.20:.6g} Ha")
                self.output_values.update(
                    energy=_re_energy,
                    energy_error=_re_error,
                    alat=self.alat,
                    restart_chk=_re_chk,
                )
                self._record_mpc_correction(_re_dE_mpc, _re_dE_mpc_error)
                if self.atomic_force:
                    forces = self._compute_force(_re_chk, work_dir=_wd)
                    if forces is not None:
                        self.output_values["forces"] = forces
                if self.compute_stress:
                    stress, pressure = self._compute_stress(work_dir=_wd)
                    if stress is not None:
                        self.output_values["stress"] = stress
                    if pressure is not None:
                        self.output_values["pressure"] = pressure
                first_new_run = self.max_continuation + 1  # skip loop
            else:
                _additional = estimate_additional_steps(
                    accumulated_measurement,
                    _re_error,
                    self.target_error,
                )
                estimated_steps = _additional + warmup
                logger.info(
                    f"  Resuming after {first_new_run - 1} prior run(s): "
                    f"error={_re_error:.6g} Ha > target "
                    f"{self.target_error:.6g} Ha -> "
                    f"{estimated_steps} steps "
                    f"(measurement: {_additional}, warmup: {warmup}, "
                    f"accumulated measurement: {accumulated_measurement})"
                )

        # -- Phase C: production loop --
        _prev_run_steps = None
        for i in range(first_new_run, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status_i = recorded.get("status")

            if status_i in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status_i}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                # First-ever production run starts fresh; all others restart
                if i == 1 and first_new_run == 1:
                    self._generate_input(estimated_steps, os.path.join(_wd, input_i))
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- LRDMC production run {i}/{self.max_continuation} (a={self.alat}) " + "-" * 15)
                logger.info(f"  {input_i}: {estimated_steps} steps")

            step_files[i] = (input_i, output_i, run_id_i)
            need_restart = i > 1 or first_new_run > 1
            restart_chk = self._find_restart_chk(_wd) if need_restart else None
            if need_restart and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            # Estimate run time from the latest available output
            _ref_net = None
            for _j in range(i - 1, 0, -1):
                if _j not in step_files:
                    continue
                _ref_net = parse_net_time(os.path.join(_wd, step_files[_j][1]))
                if _ref_net and _ref_net > 0:
                    _ref_steps = _prev_run_steps if _prev_run_steps else estimated_steps
                    _est_sec = _ref_net * (estimated_steps / _ref_steps) if _ref_steps > 0 else _ref_net
                    logger.info(f"  est. Net run time (w/o JAX compilation) = {_format_duration(_est_sec)}")
                    break
            else:
                _pilot_outs = sorted(glob.glob(os.path.join(_wd, "_pilot_b", "*.out")))
                _ref_net = parse_net_time(_pilot_outs[-1]) if _pilot_outs else None
                if _ref_net and _ref_net > 0:
                    _p_steps = estimation.get("pilot_steps") or self.pilot_steps
                    if _p_steps > 0:
                        logger.info(
                            f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net * estimated_steps / _p_steps)}"
                        )

            await self._submit_and_wait(
                input_i,
                output_i,
                work_dir=_wd,
                extra_from_objects=extra_from,
                step=i,
                run_id=run_id_i,
            )
            _prev_run_steps = estimated_steps
            last_run = i

            # -- Side-effects: compute energy from checkpoint (if any) --
            restart_chk = self._find_restart_chk(_wd)
            if restart_chk is None:
                raise RuntimeError(f"Production loop: run {i} completed but no restart checkpoint found in {_wd}.")
            # mcmc_counter in restart.h5 is the only trustworthy source for
            # accumulated samples (planned step counts over-count when the
            # run was cut short by max_time).
            actual = read_accumulated_measurement_steps(
                os.path.join(_wd, restart_chk),
                warmup=warmup,
                collect_steps=self.num_gfmc_collect_steps,
            )
            if actual is None:
                raise RuntimeError(f"Production loop: cannot read mcmc_counter from {restart_chk} in {_wd}.")
            accumulated_measurement = actual
            energy, error, dE_mpc, dE_mpc_error = self._compute_energy_cached(
                restart_chk, work_dir=_wd, accumulated=actual, with_mpc=True
            )
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["alat"] = self.alat
                self.output_values["restart_chk"] = restart_chk
                logger.info(f"  LRDMC energy (a={self.alat}): {energy} +- {error} Ha")
                self._record_mpc_correction(dE_mpc, dE_mpc_error)
                if dE_mpc is not None:
                    logger.info(f"  MPC correction: dE_MPC = {dE_mpc} +- {dE_mpc_error} Ha")
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd, output_file=output_i)
                    if forces is not None:
                        self.output_values["forces"] = forces
                if self.compute_stress:
                    stress, pressure = self._compute_stress(work_dir=_wd, output_file=output_i)
                    if stress is not None:
                        self.output_values["stress"] = stress
                    if pressure is not None:
                        self.output_values["pressure"] = pressure

                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_dE_mpc=dE_mpc,
                    last_dE_mpc_error=dE_mpc_error,
                    accumulated_measurement_steps=accumulated_measurement,
                    last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                    last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                    last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                )

            # -- Termination decision -- single source of truth --
            vstatus, vmsg = validate_completion(
                _wd,
                self.output_values,
                target_error=self.target_error,
                target_tol=1.20,
            )
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break
            if vstatus == CompletionStatus.OK:
                logger.info(f"  Target error achieved: {vmsg} (run {i}/{self.max_continuation})")
                break
            # INCOMPLETE -- prepare next iteration if we have an error estimate
            if energy is not None:
                if i < self.max_continuation:
                    old_steps = estimated_steps
                    _additional = estimate_additional_steps(
                        accumulated_measurement,
                        error,
                        self.target_error,
                    )
                    estimated_steps = _additional + warmup
                    logger.info(
                        f"  Re-estimated: {old_steps} -> {estimated_steps} steps "
                        f"(measurement: {_additional}, warmup: {warmup}, "
                        f"accumulated measurement: {accumulated_measurement})"
                    )
                else:
                    logger.warning(
                        f"Error {error:.6g} > target "
                        f"{self.target_error:.6g} Ha -- "
                        f"max_continuation ({self.max_continuation}) reached"
                    )

        # -- Final energy computation (safety net) -----------------
        last_output = step_files[last_run][1] if last_run in step_files else None
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            energy, error, dE_mpc, dE_mpc_error = self._compute_energy_cached(restart_chk, work_dir=_wd, with_mpc=True)
            if energy is not None:
                self.output_values["energy"] = energy
                self.output_values["energy_error"] = error
                self.output_values["alat"] = self.alat
                self.output_values["restart_chk"] = restart_chk
                self._record_mpc_correction(dE_mpc, dE_mpc_error)
                if self.atomic_force:
                    forces = self._compute_force(restart_chk, work_dir=_wd, output_file=last_output)
                    if forces is not None:
                        self.output_values["forces"] = forces
                if self.compute_stress:
                    stress, pressure = self._compute_stress(work_dir=_wd, output_file=last_output)
                    if stress is not None:
                        self.output_values["stress"] = stress
                    if pressure is not None:
                        self.output_values["pressure"] = pressure
                set_estimation(
                    _wd,
                    last_energy=energy,
                    last_energy_error=error,
                    last_dE_mpc=dE_mpc,
                    last_dE_mpc_error=dE_mpc_error,
                    accumulated_measurement_steps=accumulated_measurement,
                    last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                    last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                    last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
                )

        # -- Collect outputs ---------------------------------------
        chk_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = chk_files + output_logs
        self.output_values["estimated_steps"] = estimated_steps
        if self._use_gfmc_n:
            self.output_values["num_projection_per_measurement"] = self.num_projection_per_measurement
        else:
            self.output_values["time_projection_tau"] = self.time_projection_tau
            avg_nmpm = self._resolve_avg_nmpm(_wd)
            if avg_nmpm is not None:
                self.output_values["num_projection_per_measurement"] = avg_nmpm

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    # -- Utility methods -------------------------------------------

    def _resolve_avg_nmpm(self, work_dir: str) -> int | None:
        """Parse the GFMC_t output log for the averaged number of projections.

        Returns the rounded ``avg_num_projections`` as an int (>=1), or
        None if the diagnostic is unavailable.  Used to expose the
        averaged nmpm via ``output_values["num_projection_per_measurement"]``
        so that downstream GFMC_n runs can consume it via ``ValueFrom``.
        """
        try:
            diag = parse_lrdmc_output(work_dir)
        except Exception:
            return None
        avg = getattr(diag, "avg_num_projections", None) if diag is not None else None
        if avg is None:
            return None
        try:
            return max(int(round(float(avg))), 1)
        except (TypeError, ValueError):
            return None

    def _find_restart_chk(self, work_dir: str) -> str | None:
        """Locate the LRDMC restart checkpoint file in *work_dir*."""
        for pattern in ["restart.h5", "lrdmc.h5", "*.h5"]:
            matches = sorted(glob.glob(os.path.join(work_dir, pattern)))
            if matches:
                return os.path.basename(matches[-1])
        return None

    def _record_mpc_correction(self, dE_mpc: float | None, dE_mpc_error: float | None):
        """Publish the MPC correction into ``output_values`` when present.

        MPC is a passive, energy-only finite-size correction: ``energy``
        stays the pure-Ewald value and the consumer forms
        ``energy + dE_mpc`` itself, so the two error bars stay
        separable.  The keys are left absent for a run without MPC.

        Args:
            dE_mpc (float or None):
                ``dE_MPC`` from ``jqmc-tool lrdmc compute-energy``.
            dE_mpc_error (float or None):
                Statistical error on the above.
        """
        if dE_mpc is None:
            return
        self.output_values["dE_mpc"] = dE_mpc
        self.output_values["dE_mpc_error"] = dE_mpc_error

    def _compute_energy_cached(
        self,
        restart_chk: str,
        work_dir: str,
        accumulated: int | None = None,
        with_mpc: bool = False,
    ):
        """Return (energy, error) using ``[estimation]`` cache when fresh.

        The cache (``last_energy`` / ``last_energy_error`` in
        ``workflow_state.toml``) is considered fresh when the recorded
        ``accumulated_measurement_steps`` matches the current
        ``restart.h5`` ``mcmc_counter`` *and* the post-processing
        parameters (``-b``, ``-w``, ``-c``) match the workflow's
        current settings.  On a hit, no subprocess is launched.

        On a miss, :meth:`_compute_energy` is invoked and the cache is
        refreshed via :func:`set_estimation` so that subsequent
        invocations within the same or later workflow runs short-circuit.

        The MPC correction (``last_dE_mpc`` / ``last_dE_mpc_error``) is
        always computed and always written in the *same*
        :func:`set_estimation` call as the energy it belongs to, so a
        cache hit can never pair an energy from one run with a
        correction from another.  Writing *None* removes the keys (TOML
        has no null), which is exactly what a run without MPC needs.

        Args:
            restart_chk (str):
                Checkpoint filename (basename).
            work_dir (str):
                Directory in which to run the command.
            accumulated (int, optional):
                Pre-read ``mcmc_counter`` from ``restart.h5``.  Pass when
                the caller has already computed it (Phase B / Phase C) to
                avoid a redundant HDF5 read.
            with_mpc (bool, optional):
                Also return the MPC finite-size correction.  Default
                *False* keeps the ``(energy, error)`` shape.
        """
        if accumulated is None:
            accumulated = read_accumulated_measurement_steps(
                os.path.join(work_dir, restart_chk),
                warmup=self.num_gfmc_warmup_steps,
                collect_steps=self.num_gfmc_collect_steps,
            )
        est = get_estimation(work_dir)
        if (
            accumulated is not None
            and est.get("last_energy") is not None
            and est.get("last_energy_error") is not None
            and est.get("accumulated_measurement_steps") == accumulated
            and est.get("last_num_gfmc_bin_blocks") == self.num_gfmc_bin_blocks
            and est.get("last_num_gfmc_warmup_steps") == self.num_gfmc_warmup_steps
            and est.get("last_num_gfmc_collect_steps") == self.num_gfmc_collect_steps
        ):
            e, err = est["last_energy"], est["last_energy_error"]
            logger.info(
                f"  Energy cached: E = {e} +- {err} Ha "
                f"(acc={accumulated}, b={self.num_gfmc_bin_blocks}, "
                f"w={self.num_gfmc_warmup_steps}, c={self.num_gfmc_collect_steps})"
            )
            if not with_mpc:
                return e, err
            return e, err, est.get("last_dE_mpc"), est.get("last_dE_mpc_error")
        # Always parse the correction so the cache entry is complete
        # regardless of what this particular caller asked for.
        energy, error, dE_mpc, dE_mpc_error = self._compute_energy(restart_chk, work_dir=work_dir, with_mpc=True)
        if energy is not None and accumulated is not None:
            set_estimation(
                work_dir,
                last_energy=energy,
                last_energy_error=error,
                last_dE_mpc=dE_mpc,
                last_dE_mpc_error=dE_mpc_error,
                accumulated_measurement_steps=accumulated,
                last_num_gfmc_bin_blocks=self.num_gfmc_bin_blocks,
                last_num_gfmc_warmup_steps=self.num_gfmc_warmup_steps,
                last_num_gfmc_collect_steps=self.num_gfmc_collect_steps,
            )
        if not with_mpc:
            return energy, error
        return energy, error, dE_mpc, dE_mpc_error

    def _compute_energy(self, restart_chk: str, work_dir: str, with_mpc: bool = False):
        """Run ``jqmc-tool lrdmc compute-energy`` against *restart_chk*.

        Always invokes ``jqmc-tool`` so that the returned (energy, error)
        carry full numerical precision.  Parsing jqmc's printed
        ``Total Energy: E = ... +- ... Ha.`` line is lossy (``%.5f``
        formatting), which is unsuitable for values persisted to
        ``workflow_state.toml`` or compared against ``target_error``.

        Args:
            restart_chk (str):
                Checkpoint filename (basename).
            work_dir (str):
                Directory in which to run the command.
            with_mpc (bool, optional):
                Also return the MPC finite-size correction (*None* when
                the run had ``mpc = false``).  Default *False* keeps the
                ``(energy, error)`` return shape.

        Returns:
            tuple:
                ``(energy, error)``, or ``(energy, error, dE_mpc,
                dE_mpc_error)`` when *with_mpc*.  All-*None* on failure.
        """
        cmd = [
            "jqmc-tool",
            "lrdmc",
            "compute-energy",
            restart_chk,
            "-b",
            str(self.num_gfmc_bin_blocks),
            "-w",
            str(self.num_gfmc_warmup_steps),
            "-c",
            str(self.num_gfmc_collect_steps),
        ]
        logger.info(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
                cwd=work_dir,
            )
            return self._parse_energy_output(result.stdout, with_mpc=with_mpc)
        except FileNotFoundError as e:
            logger.error(f"compute-energy: '{cmd[0]}' not found on PATH ({e})")
            return (None, None, None, None) if with_mpc else (None, None)
        except subprocess.CalledProcessError as e:
            logger.error(f"compute-energy failed: {e.stderr}")
            return (None, None, None, None) if with_mpc else (None, None)

    @staticmethod
    def _parse_energy_output(text: str, with_mpc: bool = False):
        """Parse ``E = <value> +- <error> Ha.`` from jqmc-tool output.

        When the run had ``mpc = true``, jqmc-tool prints two extra lines
        *after* the plain energy line::

            MPC correction: dE_MPC = <value> +- <error> Ha.
            Total Energy (MPC-corrected): E_MPC = <value> +- <error> Ha.

        Neither confuses the plain-energy pattern: in ``dE_MPC =`` and
        ``E_MPC =`` the ``E`` is followed by ``_``, not by whitespace or
        ``=``, so ``search()`` still locks onto the first (pure-Ewald)
        ``E = ...`` line.  MPC is a passive, energy-only finite-size
        correction, so ``energy`` stays the pure-Ewald value and the
        consumer forms ``energy + dE_mpc`` itself -- keeping the two
        error bars separable.

        Args:
            text (str):
                jqmc-tool stdout.
            with_mpc (bool, optional):
                Also return the MPC correction.  Default *False* keeps
                the ``(energy, error)`` return shape.

        Returns:
            tuple:
                ``(energy, error)``, or ``(energy, error, dE_mpc,
                dE_mpc_error)`` when *with_mpc*.  Values are *None* when
                the corresponding line is absent -- in particular the
                correction is *None* for a run without MPC.
        """
        pattern = re.compile(r"E\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+\-\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)")
        match = pattern.search(text)
        energy, error = (float(match.group(1)), float(match.group(2))) if match else (None, None)
        if not with_mpc:
            return energy, error
        mpc_pattern = re.compile(r"dE_MPC\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+\-\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)")
        mpc_match = mpc_pattern.search(text)
        dE_mpc, dE_mpc_error = (float(mpc_match.group(1)), float(mpc_match.group(2))) if mpc_match else (None, None)
        return energy, error, dE_mpc, dE_mpc_error

    def _compute_force(self, restart_chk: str, work_dir: str, output_file: str | None = None):
        """Parse forces from *output_file* or run ``jqmc-tool lrdmc compute-force``.

        When *output_file* is given, forces are read directly from
        the ``jqmc`` stdout (``Atomic Forces:`` table).  Falls back
        to ``jqmc-tool`` when *output_file* is *None* or parsing fails.

        Args:
            restart_chk (str):
                Checkpoint filename (basename).
            work_dir (str):
                Directory in which to run the command.
            output_file (str, optional):
                Stdout filename (basename) of the ``jqmc`` run.

        Returns:
            list of dict or None:
                Each dict has keys ``label``, ``Fx``, ``Fx_err``,
                ``Fy``, ``Fy_err``, ``Fz``, ``Fz_err``.
                Returns *None* on failure.
        """
        # Fast path: parse from jqmc stdout
        if output_file is not None:
            out_path = os.path.join(work_dir, output_file)
            if os.path.isfile(out_path):
                try:
                    with open(out_path) as fh:
                        text = fh.read()
                    forces = parse_force_table(text)
                    if forces:
                        logger.info(f"  Forces from {output_file} (jqmc-tool skipped):")
                        for f in forces:
                            logger.info(
                                f"  {f['label']:8s}"
                                f" Fx={f['Fx']:+.6f}+-{f['Fx_err']:.6f}"
                                f" Fy={f['Fy']:+.6f}+-{f['Fy_err']:.6f}"
                                f" Fz={f['Fz']:+.6f}+-{f['Fz_err']:.6f}"
                                f" Ha/bohr"
                            )
                        return forces
                except OSError:
                    pass

        # Fallback: jqmc-tool
        cmd = [
            "jqmc-tool",
            "lrdmc",
            "compute-force",
            restart_chk,
            "-b",
            str(self.num_gfmc_bin_blocks),
            "-w",
            str(self.num_gfmc_warmup_steps),
            "-c",
            str(self.num_gfmc_collect_steps),
        ]
        logger.info(f"  Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
                cwd=work_dir,
            )
            forces = parse_force_table(result.stdout)
            if forces:
                for f in forces:
                    logger.info(
                        f"  {f['label']:8s}"
                        f" Fx={f['Fx']:+.6f}+-{f['Fx_err']:.6f}"
                        f" Fy={f['Fy']:+.6f}+-{f['Fy_err']:.6f}"
                        f" Fz={f['Fz']:+.6f}+-{f['Fz_err']:.6f}"
                        f" Ha/bohr"
                    )
            return forces
        except FileNotFoundError as e:
            logger.error(f"compute-force: '{cmd[0]}' not found on PATH ({e})")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"compute-force failed: {e.stderr}")
            return None

    def _compute_stress(self, work_dir: str, output_file: str | None = None):
        """Parse ``Stress Tensor`` + ``Pressure`` tables from *output_file*.

        Mirror of :meth:`_compute_force` but stress-only and fast-path-only
        -- ``jqmc-tool`` has no ``compute-stress`` subcommand yet, so we
        return ``(None, None)`` rather than spawning a tool process when
        the output file is missing or the tables are absent. The CLI
        emits both tables via ``_log_stress_section`` when the run was
        launched with ``compute_stress=True``.

        Stress is available for both LRDMC drivers, lrdmc-bra (GFMC_n)
        and lrdmc-tau (GFMC_t); the same tables are parsed in either
        mode.

        Args:
            work_dir: Directory containing *output_file*.
            output_file: Basename of the ``jqmc`` stdout file. If *None*,
                the most recently modified ``*.out`` in *work_dir* is
                used. Returns ``(None, None)`` when no output file is
                found.

        Returns:
            ``(stress_dict, pressure_dict)`` -- either entry may be
            *None* if the corresponding table is absent.
        """
        if output_file is None:
            candidates = sorted(
                glob.glob(os.path.join(work_dir, "*.out")),
                key=os.path.getmtime,
            )
            if not candidates:
                return None, None
            out_path = candidates[-1]
            output_file = os.path.basename(out_path)
        else:
            out_path = os.path.join(work_dir, output_file)
        if not os.path.isfile(out_path):
            return None, None
        try:
            with open(out_path) as fh:
                text = fh.read()
        except OSError:
            return None, None
        stress = parse_stress_table(text)
        pressure = parse_pressure_table(text)
        if stress is not None:
            logger.info(
                f"  Stress from {output_file}: "
                f"sigma_xx={stress['sigma_xx']:+.6f}+-{stress['sigma_xx_err']:.6f} Ha/bohr^3, "
                f"sigma_yy={stress['sigma_yy']:+.6f}+-{stress['sigma_yy_err']:.6f} Ha/bohr^3, "
                f"sigma_zz={stress['sigma_zz']:+.6f}+-{stress['sigma_zz_err']:.6f} Ha/bohr^3"
            )
        if pressure is not None:
            logger.info(
                f"  Pressure from {output_file}: P_hydro={pressure['P_hydro']:+.6f}+-{pressure['P_hydro_err']:.6f} Ha/bohr^3"
            )
        return stress, pressure
