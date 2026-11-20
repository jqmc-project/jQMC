"""VMC_Workflow -- Jastrow / orbital optimization via ``jqmc`` (job_type=vmc).

Generates a VMC input TOML, submits the ``jqmc`` binary on a remote (or local)
machine, monitors until completion, and fetches the results.  The optimized
``hamiltonian_data_opt_step_N.h5`` files and the checkpoint are collected as
outputs -- one code path for single-twist and twist-averaged runs, because jQMC
dumps a complete combined (all twists) Hamiltonian for a twist-averaged
optimisation.
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
import re
import shutil
import subprocess
import sys
import time
from logging import getLogger

from jqmc.jqmc_miscs import cli_parameters

from ._error_estimator import (
    _format_duration,
    estimate_required_steps,
    parse_net_time,
)
from ._input_generator import generate_input_toml, resolve_with_defaults
from ._job import get_num_mpi, load_queue_data
from .state import (
    CompletionStatus,
    WorkflowStatus,
    get_estimation,
    get_job_by_step,
    reconcile_fetched_jobs_recursive,
    set_estimation,
    validate_completion,
)
from .workflow import Workflow

logger = getLogger("jqmc-workflow").getChild(__name__)

#: Hand-off modes of :class:`VMC_Workflow`: average the last steps, or publish the last dump.
HANDOFF_MODES = ("average", "last")
#: Basename of the averaged Hamiltonian that ``handoff_mode="average"`` publishes.
AVERAGED_HAMILTONIAN = "hamiltonian_data_opt_averaged.h5"
_OPT_STEP_DUMP = re.compile(r"hamiltonian_data_opt_step_(\d+)\.h5")


def _last_opt_energy_from_log(output_file: str) -> tuple[float | None, float | None]:
    """Return ``(energy, energy_error)`` of the last VMC opt step in *output_file*.

    Single source of truth shared by :meth:`VMC_Workflow._parse_output`
    and :meth:`VMC_Workflow._parse_last_opt_energy`, so the two cannot drift
    apart (a separate ``E = ... +- ...`` regex can silently drop
    ``nan``/``inf`` lines).  This helper delegates to the canonical
    :func:`output_parser._parse_vmc_log_text` parser so any future
    format change lives in exactly one place.

    A ``nan``/``inf`` energy is returned as a (non-finite) float -- not
    ``None`` -- so the caller's ``math.isfinite`` check in
    :func:`validate_completion` can flag diverged runs.
    """
    try:
        with open(output_file, errors="replace") as f:
            text = f.read()
    except OSError:
        return None, None
    from .output_parser import _parse_vmc_log_text

    steps = _parse_vmc_log_text(text)
    # Prefer the last opt-step block that actually contains an ``E =``
    # line; the very last block may be a header-only stub from a
    # partial write.  Note: ``nan`` is not ``None`` so a diverged final
    # step is still selected, which is the whole point.
    for s in reversed(steps):
        if s.energy is not None and s.energy_error is not None:
            return s.energy, s.energy_error
    return None, None


class VMC_Workflow(Workflow):
    r"""VMC (Variational Monte Carlo) Jastrow / orbital optimisation workflow.

    Generates a ``job_type=vmc`` input TOML, submits ``jqmc``, monitors
    until completion, and collects the optimised Hamiltonian and checkpoint.
    By default the published Hamiltonian is the uniform average of the
    wavefunction parameters of the last ``handoff_average_steps`` (20)
    optimization steps, written by ``jqmc-tool vmc average-wf`` from the
    ``hamiltonian_data_opt_step_N.h5`` dumps with the largest step numbers
    (``hamiltonian_data_opt_averaged.h5``).  With ``handoff_mode="last"`` the
    published Hamiltonian is the last dump itself.  Single-twist and combined
    twist-averaged runs share one publication path.  Under twist averaging
    jQMC writes every dump as a COMPLETE combined file (the canonical shared Jastrow
    plus every per-twist MO/lambda payload), and the average of such dumps is
    a combined file too, so either is directly usable as the next step's
    ``hamiltonian_h5``.

    The workflow supports two modes:

    **Automatic mode** (default, ``num_mcmc_steps=None``):

    1. **Pilot VMC run** (``_0``) -- Runs a short optimisation with
       ``pilot_vmc_steps`` optimisation steps and ``pilot_mcmc_steps``
       MCMC steps per step.  The statistical error of the *last*
       optimisation step is used to estimate the MCMC steps per
       opt-step required to achieve ``target_error`` via
       $\sigma \propto 1/\sqrt{N}$.
    2. **Production VMC runs** (``_1``, ``_2``, ...) -- Full optimisation
       with ``num_opt_steps`` and the estimated MCMC steps per step.
       If a run is interrupted by the wall-time limit, the next
       continuation restarts from the checkpoint.  At most
       ``max_continuation`` runs are attempted.

    **Fixed-step mode** (``num_mcmc_steps`` is set):

    The pilot run is skipped entirely and ``target_error`` is ignored.
    Each production run uses exactly ``num_mcmc_steps`` MCMC steps per
    optimisation step, and ``max_continuation`` runs are executed
    unconditionally.

    Args:
        server_machine_name (str):
            Name of the target machine (must be configured in ``~/.jqmc_setting/``).
        num_opt_steps (int):
            Number of optimization iterations for production runs.
        hamiltonian_file (str):
            Input ``hamiltonian_data.h5``.
        queue_label (str):
            Queue/partition label from ``queue_data.toml``.
        jobname (str):
            Job name for the scheduler.
        number_of_walkers (int):
            Walkers per MPI process.
        max_time (int):
            Wall-time limit in seconds.
        Dt (float, optional):
            MCMC step size (bohr).  Default from ``jqmc_miscs``.
        epsilon_AS (float, optional):
            Attaccalite-Sorella regularization parameter.  Default from ``jqmc_miscs``.
        num_mcmc_per_measurement (int, optional):
            MCMC updates per measurement.  Default from ``jqmc_miscs``.
        num_mcmc_warmup_steps (int, optional):
            Warmup measurement steps to discard.  Default from ``jqmc_miscs``.
        num_mcmc_bin_blocks (int, optional):
            Binning blocks.  Default from ``jqmc_miscs``.
        wf_dump_freq (int, optional):
            Wavefunction dump frequency.  Default from ``jqmc_miscs`` (1,
            every step).  With ``handoff_mode="average"`` every step of the
            averaging window must be dumped, so any value other than 1 is
            rejected at construction.
        opt_J1_param (bool, optional):
            Optimize J1 Jastrow parameters.  Default from ``jqmc_miscs``.
        opt_J2_param (bool, optional):
            Optimize J2 Jastrow parameters.  Default from ``jqmc_miscs``.
        opt_J3_param (bool, optional):
            Optimize J3 Jastrow parameters.  Default from ``jqmc_miscs``.
        opt_geminal (bool, optional):
            Optimize the geminal (lambda matrix for AGP_data, MO coefficients +
            lambda_diag + lambda_unpaired for AGPn_data, MO coefficients
            only for SD_data). Default from ``jqmc_miscs``.
        fix_geminal_rank (int, optional):
            Rank-control flag for the new direct MO-opt path. ``None``
            runs the full-rank AGP-AO ambient opt. A positive ``int``
            triggers ``to_AGPn`` auto-conversion at ``run_optimize``
            entry when the current geminal is ``AGP_data``; the
            optimisation loop then runs uniformly through
            ``apply_block_update`` with S-Grassmannian retraction
            built into ``AGPn_data`` / ``SD_data``. Default ``None``.
        opt_J3_basis_exp (bool, optional):
            Optimize J3 AO Gaussian exponents.  Default from ``jqmc_miscs``.
        opt_J3_basis_coeff (bool, optional):
            Optimize J3 AO contraction coefficients.  Default from ``jqmc_miscs``.
        opt_lambda_basis_exp (bool, optional):
            Optimize Geminal AO Gaussian exponents.  Default from ``jqmc_miscs``.
        opt_lambda_basis_coeff (bool, optional):
            Optimize Geminal AO contraction coefficients.  Default from ``jqmc_miscs``.
        opt_JDTN_u_param (bool, optional):
            Optimize the DTN polynomial e-e channel (``u`` block, ``alpha_indep``,
            J2 analogue).  Default from ``jqmc_miscs`` (``True``).
        opt_JDTN_chi_param (bool, optional):
            Optimize the DTN polynomial e-N channel (``chi`` block, ``beta_indep``,
            J1 analogue).  Default from ``jqmc_miscs`` (``True``).
        opt_JDTN_f_param (bool, optional):
            Optimize the DTN polynomial factorised e-e-N channel (``f`` block,
            ``gamma_indep``, J3 analogue).  Default from ``jqmc_miscs`` (``True``).
        optimizer_kwargs (dict, optional):
            Optimizer configuration dict.  Default from ``jqmc_miscs``.
        use_lm (bool, optional):
            Convenience alias merged into a copy of ``optimizer_kwargs`` as
            its ``use_lm`` key (enable the linear method).  A value that
            conflicts with the same key in ``optimizer_kwargs`` raises
            ``ValueError``.  Default *None* (not set).
        num_opt_param (int, optional):
            Convenience alias merged into a copy of ``optimizer_kwargs`` as
            the optimizer's canonical ``lm_subspace_dim`` key.  Conflicting
            values raise ``ValueError``.  Default *None* (not set).
        delta (float, optional):
            Convenience alias merged into a copy of ``optimizer_kwargs`` as
            its ``delta`` key (prefactor of the SR update).  Conflicting
            values raise ``ValueError``.  Default *None* (not set).
        epsilon (float, optional):
            Convenience alias merged into a copy of ``optimizer_kwargs`` as
            its ``epsilon`` key (regularization strength added to S).
            Conflicting values raise ``ValueError``.  Default *None* (not
            set).
        mcmc_seed (int, optional):
            Random seed for MCMC.  Default from ``jqmc_miscs``.
        verbosity (str, optional):
            Verbosity level.  Default from ``jqmc_miscs``.
        poll_interval (int):
            Seconds between job-status polls, and between
            submission attempts while the queue's ``max_job_submit`` is reached.
        target_error (float):
            Target statistical error (Ha) per optimization step.  Ignored
            when *num_mcmc_steps* is set.
        target_acceptance_ratio (float):
            Target MCMC acceptance ratio for the ``Dt="auto"`` calibration,
            which runs short plain MCMC sampling jobs and fits ``Dt`` to
            this ratio.  Used only when ``Dt="auto"``.  Default 0.5.
        num_mcmc_steps (int, optional):
            Fixed number of MCMC measurement steps per optimisation step.
            When set, the pilot run is skipped and ``target_error`` is
            ignored; each of the ``max_continuation`` production runs uses
            exactly this many MCMC steps per opt step.
        pilot_mcmc_steps (int):
            MCMC steps per opt-step for the pilot run.  Ignored when
            *num_mcmc_steps* is set.
        pilot_vmc_steps (int):
            Number of optimization steps in the pilot run (small; just
            enough to estimate the error bar).
        pilot_queue_label (str, optional):
            Queue label for the pilot run.  Defaults to *queue_label*.
            Use a shorter/smaller queue for the pilot to save resources.
        max_continuation (int):
            Maximum number of production runs after the pilot.
        target_snr (float):
            Target signal-to-noise ratio ``max(|f|/|std f|)`` for force
            convergence.  The workflow continues until the averaged
            S/N drops to or below this threshold.
        snr_avg_window (int):
            Number of trailing optimization steps over which to average
            the signal-to-noise ratio for the convergence check.
            When there are fewer S/N values than this window, all
            available values are averaged.
        energy_slope_sigma_threshold (float, optional):
            Enables the energy-plateau convergence check: a linear fit of
            energy vs. optimization step over the trailing window is
            considered converged when ``slope >= -slope_std *
            energy_slope_sigma_threshold``.  Default *None* (check
            disabled).
        energy_slope_window_size (int):
            Number of trailing optimization steps used for the energy
            slope fit.  Default 5.
        cleanup_patterns (list[str], optional):
            Glob patterns for files to delete after successful completion
            (e.g. ``["restart.h5", "hamiltonian_opt*.h5"]``).  Local files
            are always removed; remote files are removed only when the
            workflow targets a remote machine.  Default *None* (no cleanup).
        handoff_mode (str):
            What ``optimized_hamiltonian`` publishes.  ``"average"``
            (default): the uniform average of the last
            ``handoff_average_steps`` dumps, selected by step number and
            averaged with ``jqmc-tool vmc average-wf`` into
            ``hamiltonian_data_opt_averaged.h5``.  ``"last"``: the dump with
            the largest step number, as before the averaged hand-off.
        handoff_average_steps (int):
            Number of last optimization steps to average (default 20, the
            tutorials' standard).  Ignored with ``handoff_mode="last"``.
            The selected dumps must be consecutive steps; fewer dumps than
            this are refused (and, unless ``external_restart_chk`` is set,
            ``num_opt_steps * max_continuation`` smaller than this is
            refused at construction).  In automatic mode an early
            convergence stop is deferred until enough dumps exist.
        handoff_allow_fewer (bool):
            Average all available dumps when fewer than
            ``handoff_average_steps`` exist instead of refusing.  Default
            *False*.
        external_restart_chk (str, optional):
            Basename of a checkpoint produced by an *earlier, unrelated*
            run and staged into the project directory through
            ``Container.input_files``.  It may be a single ``restart.h5``
            file or a twist checkpoint DIRECTORY
            (``restart.h5/{shared_state.h5,twist_NNN.h5}``).  When set,
            production run 1 is emitted with ``restart = true`` /
            ``restart_chk = <name>`` and the checkpoint is uploaded with
            the job.  On a restart ``num_opt_steps`` counts *additional*
            steps: the driver continues from the checkpoint's completed
            step count and runs ``num_opt_steps`` more.
            Because the checkpoint pins the keys that are *frozen on
            restart* (``mcmc_seed``, ``number_of_walkers``,
            ``num_mcmc_per_measurement``, ``Dt``, ``epsilon_AS``), the
            workflow must be constructed with the same values the original
            run used or the jqmc CLI restart guard rejects the input.
            Default *None* (run 1 starts from scratch).
        precision_mode (str):
            Precision mode written to the ``[precision]`` section
            (``mode``) of the generated TOML: ``"full"`` (all float64) or
            ``"mixed"`` (mixed precision).  Default ``"full"``.
        mpc (bool, optional):
            Value of ``[coulomb] mpc`` in the generated TOML.  The ``vmc``
            job (wavefunction optimization) does not publish a final MPC
            estimate, so the jqmc CLI rejects ``True`` with a ``ValueError``;
            enable MPC in a subsequent MCMC or LRDMC sampling job instead.
            ``False`` writes the key, *None* (default) omits it.
        mpc_cutoff (float, optional):
            Reciprocal-space kinetic-energy cutoff (Ha) for the MPC
            reciprocal-space sum (``[coulomb] mpc_cutoff``).  A convergence
            parameter.  *None* (default) omits the key so the driver
            default applies.

    Examples:
        Standalone launch (automatic mode)::

            wf = VMC_Workflow(
                server_machine_name="cluster",
                num_opt_steps=20,
                target_error=0.001,
                pilot_mcmc_steps=50,
                pilot_vmc_steps=5,
                number_of_walkers=8,
            )
            status, files, values = wf.launch()
            print(values["optimized_hamiltonian"])

        Fixed-step mode (no pilot, no target_error check)::

            wf = VMC_Workflow(
                server_machine_name="cluster",
                num_opt_steps=20,
                num_mcmc_steps=500,
                number_of_walkers=8,
                max_continuation=3,
            )
            status, files, values = wf.launch()

        As part of a :class:`Launcher` pipeline::

            enc = Container(
                label="vmc",
                dirname="01_vmc",
                input_files=[FileFrom("wf", "hamiltonian_data.h5")],
                workflow=VMC_Workflow(
                    server_machine_name="cluster",
                    num_opt_steps=20,
                    target_error=0.001,
                ),
            )

    Output Values:
        optimized_hamiltonian (str):
            Basename of the downstream optimised Hamiltonian:
            ``"hamiltonian_data_opt_averaged.h5"`` with
            ``handoff_mode="average"``, or the last opt-step file (for
            example ``"hamiltonian_data_opt_step_91.h5"``) with
            ``handoff_mode="last"``, for single-twist AND twist-averaged
            runs alike -- under twist averaging jQMC dumps a complete
            combined file holding every twist.  Use with
            ``ValueFrom("vmc", "optimized_hamiltonian")`` to pass the
            filename dynamically to downstream workflows.
        optimized_hamiltonian_handoff (str):
            The ``handoff_mode`` that produced ``optimized_hamiltonian``.
        optimized_hamiltonian_sources (list[str]):
            Basenames of the dumps behind ``optimized_hamiltonian``, in step
            order (one file with ``handoff_mode="last"``).
        optimized_hamiltonian_steps (list[int]):
            Optimization step numbers of ``optimized_hamiltonian_sources``.
        checkpoint (str):
            Basename of the restart checkpoint file.
        num_mcmc_steps (int):
            MCMC steps per optimisation step (fixed-step mode).  In
            automatic mode this key is ``estimated_mcmc_steps`` instead.
        estimated_mcmc_steps (int):
            Estimated MCMC steps per optimisation step (automatic mode).
        energy (float):
            Energy from the last optimisation step (Ha).
        energy_error (float):
            Statistical error on ``energy`` (Ha).
        signal_to_noise (float):
            Average signal-to-noise ratio over the trailing window
            (only when force-convergence is enabled).
        signal_to_noise_last (float):
            Signal-to-noise ratio of the last optimisation step.
        energy_slope (float):
            Slope of energy vs. step from the trailing window
            (only when ``energy_slope_sigma_threshold`` is set).
        energy_slope_std (float):
            Standard deviation of the energy slope.

    Notes:
        * The pilot uses a small number of opt steps (``pilot_vmc_steps``)
          just to estimate the error.  The real optimisation happens in
          production runs with the full ``num_opt_steps``.
        * The estimation is stored in ``workflow_state.toml`` under
          ``[estimation]``; on re-entrance the pilot is skipped.

    See Also:
        MCMC_Workflow : VMC production sampling (job_type=mcmc).
        LRDMC_Workflow : Diffusion Monte Carlo (job_type=lrdmc-bra / lrdmc-tau).
        WF_Workflow : TREXIO -> hamiltonian_data conversion.
    """

    def __init__(
        self,
        server_machine_name: str = "localhost",
        num_opt_steps: int = 20,
        hamiltonian_file: str = "hamiltonian_data.h5",
        queue_label: str = "default",
        jobname: str = "jqmc-vmc",
        number_of_walkers: int = 4,
        max_time: int = 86400,
        # -- [vmc] section parameters --
        Dt: float | str | None = None,
        epsilon_AS: float | None = None,
        num_mcmc_per_measurement: int | str | None = None,
        num_mcmc_warmup_steps: int | None = None,
        num_mcmc_bin_blocks: int | None = None,
        wf_dump_freq: int | None = None,
        opt_J1_param: bool | None = None,
        opt_J2_param: bool | None = None,
        opt_J3_param: bool | None = None,
        opt_geminal: bool | None = None,
        fix_geminal_rank: int | None = None,
        opt_J3_basis_exp: bool | None = None,
        opt_J3_basis_coeff: bool | None = None,
        opt_lambda_basis_exp: bool | None = None,
        opt_lambda_basis_coeff: bool | None = None,
        opt_JDTN_u_param: bool | None = None,
        opt_JDTN_chi_param: bool | None = None,
        opt_JDTN_f_param: bool | None = None,
        optimizer_kwargs: dict | None = None,
        # -- [control] section parameters --
        mcmc_seed: int | None = None,
        verbosity: str | None = None,
        # -- workflow parameters --
        poll_interval: int = 60,
        target_error: float = 0.001,
        target_acceptance_ratio: float = 0.5,
        num_mcmc_steps: int | None = None,
        pilot_mcmc_steps: int = 50,
        pilot_vmc_steps: int = 5,
        pilot_queue_label: str | None = None,
        max_continuation: int = 1,
        target_snr: float | None = None,
        snr_avg_window: int = 5,
        energy_slope_sigma_threshold: float | None = None,
        energy_slope_window_size: int = 5,
        cleanup_patterns: list | None = None,
        # -- hand-off of the optimized wavefunction --
        handoff_mode: str = "average",
        handoff_average_steps: int = 20,
        handoff_allow_fewer: bool = False,
        # -- external continuation --
        # Basename of a checkpoint that already exists in the project
        # directory when the workflow starts (delivered through
        # ``Container.input_files``), i.e. one produced by *some other* run.
        # It may be a single ``restart.h5`` file or a twist checkpoint
        # DIRECTORY (``restart.h5/{shared_state.h5,twist_NNN.h5}``).  When
        # set, production run 1 is generated with ``restart = true`` /
        # ``restart_chk = <name>`` instead of starting from scratch, and the
        # checkpoint is uploaded alongside the job.  ``None`` (default):
        # run 1 always starts fresh.
        external_restart_chk: str | None = None,
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
        # -- optimizer convenience aliases (kept last for positional API stability) --
        use_lm: bool | None = None,
        num_opt_param: int | None = None,
        delta: float | None = None,
        epsilon: float | None = None,
    ):
        super().__init__(cleanup_patterns=cleanup_patterns)
        self.server_machine_name = server_machine_name
        self.num_opt_steps = num_opt_steps
        self.hamiltonian_file = hamiltonian_file
        self.queue_label = queue_label
        self.jobname = jobname
        self.number_of_walkers = number_of_walkers
        self.max_time = max_time
        # [vmc] section
        self.Dt = Dt
        self.epsilon_AS = epsilon_AS
        self.num_mcmc_per_measurement = num_mcmc_per_measurement
        # 'auto' -> 4 * total electrons, resolved once from
        # the input h5 and cached (shared contract with MCMC_Workflow).
        self._resolved_num_mcmc_per_measurement = None
        self.num_mcmc_warmup_steps = num_mcmc_warmup_steps
        self.num_mcmc_bin_blocks = num_mcmc_bin_blocks
        self.wf_dump_freq = wf_dump_freq
        self.opt_J1_param = opt_J1_param
        self.opt_J2_param = opt_J2_param
        self.opt_J3_param = opt_J3_param
        self.opt_geminal = opt_geminal
        self.fix_geminal_rank = fix_geminal_rank
        self.opt_J3_basis_exp = opt_J3_basis_exp
        self.opt_J3_basis_coeff = opt_J3_basis_coeff
        self.opt_lambda_basis_exp = opt_lambda_basis_exp
        self.opt_lambda_basis_coeff = opt_lambda_basis_coeff
        self.opt_JDTN_u_param = opt_JDTN_u_param
        self.opt_JDTN_chi_param = opt_JDTN_chi_param
        self.opt_JDTN_f_param = opt_JDTN_f_param
        # Convenience surface used by workflow clients.  The optimizer itself
        # has one canonical dictionary contract; merge these aliases into a
        # copy so callers can safely reuse their input dict.  ``num_opt_param``
        # is the workflow name for LM's subspace dimension.
        merged_optimizer_kwargs = dict(optimizer_kwargs or {})
        convenience = {
            "use_lm": use_lm,
            "lm_subspace_dim": num_opt_param,
            "delta": delta,
            "epsilon": epsilon,
        }
        for key, value in convenience.items():
            if value is None:
                continue
            if key in merged_optimizer_kwargs and merged_optimizer_kwargs[key] != value:
                raise ValueError(
                    f"optimizer_kwargs[{key!r}]={merged_optimizer_kwargs[key]!r} "
                    f"conflicts with the explicit workflow argument value {value!r}."
                )
            merged_optimizer_kwargs[key] = value
        self.optimizer_kwargs = merged_optimizer_kwargs or None
        # [control] section
        self.mcmc_seed = mcmc_seed
        self.verbosity = verbosity
        # workflow
        self.poll_interval = poll_interval
        self.target_error = target_error
        # Dt='auto' calibrates to this acceptance (pilot_a).
        # Driver asymmetry: for VMC pilot_a is a plain MCMC sampling run.
        self.target_acceptance_ratio = target_acceptance_ratio
        self.num_mcmc_steps = num_mcmc_steps
        self.pilot_mcmc_steps = pilot_mcmc_steps
        self.pilot_vmc_steps = pilot_vmc_steps
        self.pilot_queue_label = pilot_queue_label or queue_label
        self.max_continuation = max_continuation
        self.target_snr = target_snr
        self.snr_avg_window = snr_avg_window
        self.energy_slope_sigma_threshold = energy_slope_sigma_threshold
        self.energy_slope_window_size = energy_slope_window_size
        self.external_restart_chk = external_restart_chk
        self.handoff_mode = handoff_mode
        self.handoff_average_steps = handoff_average_steps
        self.handoff_allow_fewer = handoff_allow_fewer
        self._validate_handoff()
        # [precision] section
        self.precision_mode = precision_mode
        # [coulomb] section (passive MPC energy-only observable)
        self.mpc = mpc
        self.mpc_cutoff = mpc_cutoff

    def _validate_handoff(self):
        """Reject hand-off settings that cannot produce the requested average.

        Raising here, before any job is submitted, keeps a long optimization
        from ending in a hand-off refusal that its settings made certain.
        """
        if self.handoff_mode not in HANDOFF_MODES:
            raise ValueError(f"handoff_mode must be one of {HANDOFF_MODES}, got {self.handoff_mode!r}.")
        steps = self.handoff_average_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"handoff_average_steps must be a positive integer, got {steps!r}.")
        if self.handoff_mode != "average":
            return
        dump_freq = self.wf_dump_freq if self.wf_dump_freq is not None else cli_parameters["vmc"]["wf_dump_freq"]
        if int(dump_freq) != 1:
            raise ValueError(
                f"handoff_mode='average' averages the last {steps} consecutive optimization steps, but "
                f"wf_dump_freq={dump_freq} dumps only every {dump_freq}-th step. Set wf_dump_freq=1 (or leave it unset), "
                "or pass handoff_mode='last' to publish the last dump only."
            )
        if self.external_restart_chk is None and not self.handoff_allow_fewer:
            total = int(self.num_opt_steps) * int(self.max_continuation)
            if total < steps:
                raise ValueError(
                    f"handoff_average_steps={steps} needs at least {steps} optimization steps, but num_opt_steps="
                    f"{self.num_opt_steps} x max_continuation={self.max_continuation} gives at most {total}. Run more "
                    "steps, lower handoff_average_steps, pass handoff_allow_fewer=True, or handoff_mode='last'."
                )

    # -- Input generation ------------------------------------------
    def _resolve_num_mcmc_per_measurement(self, input_file):
        """Expand ``num_mcmc_per_measurement='auto'`` -> 4 * total electrons.

        Shares the read-once/cache contract with
        ``MCMC_Workflow`` via ``_auto_params.resolve_workflow_num_mcmc_per_measurement``.
        For VMC the 'auto' Dt calibration is an MCMC pilot (driver
        asymmetry), but the num_mcmc_per_measurement resolution itself only needs
        the electron count from the input h5.
        """
        from ._auto_params import resolve_workflow_num_mcmc_per_measurement

        resolved, self._resolved_num_mcmc_per_measurement = resolve_workflow_num_mcmc_per_measurement(
            self.num_mcmc_per_measurement,
            self._resolved_num_mcmc_per_measurement,
            self.hamiltonian_file,
            input_file,
        )
        return resolved

    def _generate_input(
        self,
        num_mcmc_steps,
        num_opt_steps,
        input_file,
        restart=False,
        restart_chk=None,
    ):
        """Generate a VMC TOML input file.

        Args:
            num_mcmc_steps (int):
                MCMC measurement steps per optimization step.
            num_opt_steps (int):
                Number of optimization iterations.
            input_file (str):
                Output filename for the TOML.
            restart (bool):
                Whether to restart from a checkpoint.
            restart_chk (str, optional):
                Checkpoint file to restart from.
        """
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
        vmc_ov = resolve_with_defaults(
            "vmc",
            {
                "num_opt_steps": num_opt_steps,
                "num_mcmc_steps": num_mcmc_steps,
                "Dt": self.Dt,
                "epsilon_AS": self.epsilon_AS,
                "num_mcmc_per_measurement": self._resolve_num_mcmc_per_measurement(input_file),
                "num_mcmc_warmup_steps": self.num_mcmc_warmup_steps,
                "num_mcmc_bin_blocks": self.num_mcmc_bin_blocks,
                "wf_dump_freq": self.wf_dump_freq,
                "opt_J1_param": self.opt_J1_param,
                "opt_J2_param": self.opt_J2_param,
                "opt_J3_param": self.opt_J3_param,
                "opt_geminal": (bool(self.opt_geminal) if self.opt_geminal is not None else False),
                "fix_geminal_rank": int(self.fix_geminal_rank) if self.fix_geminal_rank is not None else 0,
                "opt_J3_basis_exp": self.opt_J3_basis_exp,
                "opt_J3_basis_coeff": self.opt_J3_basis_coeff,
                "opt_lambda_basis_exp": self.opt_lambda_basis_exp,
                "opt_lambda_basis_coeff": self.opt_lambda_basis_coeff,
                "opt_JDTN_u_param": self.opt_JDTN_u_param,
                "opt_JDTN_chi_param": self.opt_JDTN_chi_param,
                "opt_JDTN_f_param": self.opt_JDTN_f_param,
                "optimizer_kwargs": self.optimizer_kwargs,
            },
        )
        overrides = {
            "control": control_ov,
            "vmc": vmc_ov,
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
        generate_input_toml(
            job_type="vmc",
            overrides=overrides,
            filename=input_file,
        )

    # -- Submit / poll / fetch -------------------------------------
    # _submit_and_wait() and _make_job() are inherited from Workflow.

    def _get_command(self, input_file: str) -> str:
        """VMC reads a TOML; the script line is ``mpirun -np N jqmc <input>``."""
        return f"jqmc {input_file}"

    # -- configure / run ------------------------------------------

    def configure(self) -> dict:
        """Validate parameters and return configuration summary."""
        mode = "fixed" if self.num_mcmc_steps is not None else "auto"
        return {
            "jobname": self.jobname,
            "num_opt_steps": self.num_opt_steps,
            "num_mcmc_steps": self.num_mcmc_steps,
            "target_error": self.target_error,
            "mode": mode,
            "hamiltonian_file": self.hamiltonian_file,
            "server_machine": self.server_machine_name,
            "number_of_walkers": self.number_of_walkers,
            "max_time": self.max_time,
            "target_snr": self.target_snr,
            "pilot_mcmc_steps": self.pilot_mcmc_steps,
            "pilot_vmc_steps": self.pilot_vmc_steps,
            "max_continuation": self.max_continuation,
            "handoff_mode": self.handoff_mode,
            "handoff_average_steps": self.handoff_average_steps,
            "handoff_allow_fewer": self.handoff_allow_fewer,
        }

    async def run(self) -> tuple:
        """Run the VMC optimization workflow.

        **Fixed-step mode** (``num_mcmc_steps`` is set):
        The pilot run is skipped.  Each production run uses exactly
        ``num_mcmc_steps`` MCMC steps per opt step and all
        ``max_continuation`` runs are executed unconditionally.

        **Automatic mode** (``num_mcmc_steps`` is *None*, default):

        1. Pilot VMC run in ``_pilot/`` with ``pilot_vmc_steps`` opt steps
           and ``pilot_mcmc_steps`` MCMC steps to estimate the required
           MCMC steps per opt step (skipped on continuation).
           May use a different queue (``pilot_queue_label``).
        2. Production VMC runs (``_1``, ``_2``, ...) start from scratch
           with the full ``num_opt_steps`` and estimated MCMC steps until
           all optimization steps complete or ``max_continuation`` is
           reached.
        """
        self._ensure_project_dir()
        _wd = self.project_dir

        # Reconcile any orphaned "submitted"/"completed" job records whose
        # output file already landed locally with a "Program ends" marker
        # (e.g. workflow killed between job completion and fetch-finalize).
        # Walks the pilot subdirectory (``_pilot/``) too, since it carries
        # its own state file.  SNR / slope convergence checks parse those
        # output files, so a stale "submitted" record would otherwise hide
        # the latest data.
        n_reconciled = reconcile_fetched_jobs_recursive(_wd)
        if n_reconciled:
            logger.info(f"  Reconciled {n_reconciled} job record(s) to 'fetched' from existing output.")

        # -- Dt calibration (pilot_a, cross-driver MCMC) -----------
        # For VMC the Dt calibration is a plain MCMC
        # sampling pilot (Dt is a pure-sampling concept); pilot_b / production
        # stay VMC. self.Dt becomes a concrete float afterwards.
        if self.Dt == "auto":
            await self._run_dt_calibration(_wd)

        # -- Fixed-step mode ---------------------------------------
        if self.num_mcmc_steps is not None:
            return await self._launch_fixed_steps(_wd)

        # -- Automatic mode (pilot + target_error) -----------------
        return await self._launch_auto(_wd)

    async def _run_dt_calibration(self, _wd):
        """pilot_a: calibrate ``Dt='auto'`` via a plain MCMC sampling pilot.

        Driver asymmetry: for VMC the Dt calibration runs MCMC sampling jobs (not
        VMC optimization), so pilot_a generates *mcmc-type* input while
        pilot_b / production stay VMC. Uses the shared calibration orchestration.
        """
        from ._auto_params import run_dt_calibration

        self.Dt = await run_dt_calibration(
            self,
            _wd,
            self.hamiltonian_file,
            self.pilot_mcmc_steps,
            self.pilot_queue_label,
            self.target_acceptance_ratio,
            lambda sub_dir, inp_f, dt_v: self._write_mcmc_calibration_input(os.path.join(sub_dir, inp_f), dt_v),
        )

    def _write_mcmc_calibration_input(self, input_path, dt_v):
        """Write an mcmc-type TOML for a VMC pilot_a Dt-calibration run.

        Cross-driver: the calibration is plain MCMC sampling, so this
        emits a ``job_type='mcmc'`` input at the trial ``Dt`` reusing the VMC's
        walker count / seed / precision. Factored out of ``_run_dt_calibration``
        so it can be unit-tested without the async job machinery.
        """
        from ._input_generator import generate_input_toml, resolve_with_defaults

        control_ov = resolve_with_defaults(
            "control",
            {
                "number_of_walkers": self.number_of_walkers,
                "max_time": self.max_time,
                "hamiltonian_h5": self.hamiltonian_file,
                "mcmc_seed": self.mcmc_seed,
                "verbosity": self.verbosity,
            },
        )
        mcmc_ov = resolve_with_defaults(
            "mcmc",
            {
                "num_mcmc_steps": self.pilot_mcmc_steps,
                "Dt": dt_v,
                "epsilon_AS": self.epsilon_AS,
                "num_mcmc_per_measurement": self._resolve_num_mcmc_per_measurement(input_path),
            },
        )
        generate_input_toml(
            job_type="mcmc",
            overrides={"control": control_ov, "mcmc": mcmc_ov, "precision": {"mode": self.precision_mode}},
            filename=input_path,
        )

    async def _launch_fixed_steps(self, _wd):
        """Fixed-step production: skip pilot, ignore target_error."""
        estimated_mcmc_steps = self.num_mcmc_steps
        logger.info("")
        logger.info("-- VMC Fixed-step mode " + "-" * 27)
        logger.info(
            f"  num_mcmc_steps    = {estimated_mcmc_steps}\n"
            f"  num_opt_steps     = {self.num_opt_steps}\n"
            f"  max_continuation  = {self.max_continuation}"
        )

        last_run = 0
        step_files = {}  # {step: (input, output, run_id)}
        for i in range(1, self.max_continuation + 1):
            recorded = get_job_by_step(_wd, i)
            status = recorded.get("status")
            if status == "fetched":
                logger.info(f"  step {i}: already fetched. Skipping.")
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                last_run = i
                continue
            if status in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status}. Resuming...")
            else:
                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    # Run 1 normally starts from scratch; with
                    # ``external_restart_chk`` it continues an optimisation
                    # that some earlier run left in the staged checkpoint.
                    restart_chk = self._resolve_external_restart_chk(_wd)
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=restart_chk is not None,
                        restart_chk=restart_chk,
                    )
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=True,
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- VMC Production run {i}/{self.max_continuation} " + "-" * 10)
                logger.info(f"  {input_i}: {self.num_opt_steps} opt steps x {estimated_mcmc_steps} MCMC steps/step")

            restart_chk = self._find_restart_chk(_wd) if i > 1 else self._resolve_external_restart_chk(_wd)
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
            step_files[i] = (input_i, output_i, run_id_i)
            last_run = i

            logger.info(f"  VMC production run {i}/{self.max_continuation} completed.")

            # Refresh output_values["energy"] from THIS iteration's log
            # before validating -- otherwise validate_completion sees a
            # stale or unset energy and the non-finite check at
            # ``state.py`` silently passes for diverged runs.
            self._parse_output(os.path.join(_wd, output_i))

            # -- Abnormal-termination guard (single source of truth) --
            # target_error=None -> only Program-ends / non-finite-energy
            # checks are active. VMC's SNR/slope convergence is decided
            # separately at end-of-workflow.
            vstatus, vmsg = validate_completion(_wd, self.output_values)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

        # -- Collect outputs ---------------------------------------
        h5_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = h5_files + output_logs

        await self._publish_handoff(_wd)
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            self.output_values["checkpoint"] = restart_chk
        self.output_values["num_mcmc_steps"] = estimated_mcmc_steps

        # Parse last production output for energy
        if last_run in step_files:
            last_output = os.path.join(_wd, step_files[last_run][1])
            self._parse_output(last_output)

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    async def _launch_auto(self, _wd):
        """Automatic mode: pilot + target_error convergence."""
        # -- Pilot estimation (skip on continuation) ---------------
        estimation = get_estimation(_wd)

        if estimation.get("estimated_mcmc_steps") is not None:
            estimated_mcmc_steps = int(estimation["estimated_mcmc_steps"])
            logger.info(
                "Estimation already done (continuation): "
                f"estimated_mcmc_steps={estimated_mcmc_steps} per opt step. "
                "Skipping pilot."
            )
        else:
            # -- Run pilot in a subdirectory -----------------------
            pilot_dir = os.path.join(_wd, "_pilot")
            os.makedirs(pilot_dir, exist_ok=True)

            h5_src = os.path.join(_wd, self.hamiltonian_file)
            h5_dst = os.path.join(pilot_dir, self.hamiltonian_file)
            if os.path.isfile(h5_src) and not os.path.exists(h5_dst):
                try:
                    os.symlink(h5_src, h5_dst)
                except OSError:
                    # Some shared filesystems don't support symlinks (EOPNOTSUPP);
                    # fall back to a plain copy for the pilot staging.
                    shutil.copy(h5_src, h5_dst)

            recorded_0 = get_job_by_step(pilot_dir, 0)
            status_0 = recorded_0.get("status")
            if status_0 in ("fetched", "submitted", "completed"):
                input_0 = recorded_0["input_file"]
                output_0 = recorded_0["output_file"]
                run_id_0 = recorded_0.get("run_id", "")
                if status_0 == "fetched":
                    logger.info(f"  {input_0}: already fetched. Skipping pilot.")
                else:
                    logger.info(f"  {input_0}: already {status_0}. Resuming...")
            else:
                run_id_0 = self._new_run_id()
                input_0 = self._input_filename(0, run_id_0)
                output_0 = self._output_filename(0, run_id_0)
                self._generate_input(
                    self.pilot_mcmc_steps,
                    self.pilot_vmc_steps,
                    os.path.join(pilot_dir, input_0),
                )
            logger.info("")
            logger.info("-- VMC pilot estimation " + "-" * 27)
            logger.info(
                f"  {input_0}: {self.pilot_vmc_steps} opt steps x {self.pilot_mcmc_steps} MCMC steps/step (queue: {self.pilot_queue_label})"
            )

            pilot_t0 = time.monotonic()
            await self._submit_and_wait(
                input_0, output_0, work_dir=pilot_dir, queue_label=self.pilot_queue_label, step=0, run_id=run_id_0
            )
            pilot_wall_sec = time.monotonic() - pilot_t0

            # Parse the last optimization step's error from pilot output
            pilot_output_path = os.path.join(pilot_dir, output_0)
            pilot_energy, pilot_error = self._parse_last_opt_energy(pilot_output_path)
            if pilot_error is None:
                raise RuntimeError(
                    f"Could not parse energy error from the last optimization step of the pilot VMC output ({output_0})."
                )

            logger.info(
                f"Pilot VMC last-step energy: {pilot_energy} +- {pilot_error} Ha (from {self.pilot_mcmc_steps} MCMC steps/step)"
            )

            # Walker ratio: pilot queue may have different MPI count
            pilot_qd = load_queue_data(self.server_machine_name, self.pilot_queue_label)
            prod_qd = load_queue_data(self.server_machine_name, self.queue_label)
            pilot_mpi = get_num_mpi(pilot_qd)
            prod_mpi = get_num_mpi(prod_qd)
            walker_ratio = pilot_mpi / prod_mpi

            warmup = self.num_mcmc_warmup_steps or 0
            # The CLI rejects an input with fewer measurement steps after
            # warm-up than binning blocks, so the estimate must respect the
            # bin-block count that will actually be written to the TOML: the
            # explicit value, or the CLI default when none was given.
            if self.num_mcmc_bin_blocks is not None:
                bin_blocks = self.num_mcmc_bin_blocks
            else:
                from jqmc.jqmc_miscs import cli_parameters

                bin_blocks = cli_parameters["mcmc"]["num_mcmc_bin_blocks"]
            estimated_mcmc_steps = estimate_required_steps(
                self.pilot_mcmc_steps - warmup,
                pilot_error,
                self.target_error,
                walker_ratio=walker_ratio,
                min_steps=bin_blocks,
            )
            # Add warmup back: production also discards warmup steps
            estimated_mcmc_steps += warmup

            total_pilot_cost = self.pilot_vmc_steps * self.pilot_mcmc_steps
            total_prod_cost = self.num_opt_steps * estimated_mcmc_steps
            cost_ratio = total_prod_cost / total_pilot_cost if total_pilot_cost > 0 else 0

            # Time estimate: only Net time scales with step count
            net_pilot_sec = parse_net_time(pilot_output_path)
            if net_pilot_sec is not None and net_pilot_sec > 0:
                overhead_sec = pilot_wall_sec - net_pilot_sec
                est_prod_sec = overhead_sec + net_pilot_sec * cost_ratio
            else:
                est_prod_sec = pilot_wall_sec * cost_ratio
            logger.info("")
            logger.info("-- VMC Step Estimation Summary " + "-" * 19)
            logger.info(
                f"  pilot MCMC/step     = {self.pilot_mcmc_steps}\n"
                f"  warmup steps        = {warmup}\n"
                f"  pilot VMC steps     = {self.pilot_vmc_steps}\n"
                f"  pilot error (last)  = {pilot_error:.6g} Ha\n"
                f"  target error        = {self.target_error:.6g} Ha\n"
                f"  pilot MPI procs     = {pilot_mpi}\n"
                f"  prod. MPI procs     = {prod_mpi}\n"
                f"  walker ratio        = {walker_ratio:.4g}\n"
                f"  estimated MCMC/step = {estimated_mcmc_steps}\n"
                f"  production VMC steps= {self.num_opt_steps}\n"
                f"  total pilot cost    = {total_pilot_cost} steps\n"
                f"  total prod. cost    = {total_prod_cost} steps\n"
                f"  pilot wall time     = {_format_duration(pilot_wall_sec)}\n"
                f"  pilot net time      = "
                f"{_format_duration(net_pilot_sec) if net_pilot_sec else 'N/A'}\n"
                f"  est. production time= {_format_duration(est_prod_sec)}"
            )
            logger.info("-" * 50)

            # Save estimation to main working directory
            set_estimation(
                _wd,
                pilot_mcmc_steps=self.pilot_mcmc_steps,
                pilot_vmc_steps=self.pilot_vmc_steps,
                pilot_error=pilot_error,
                target_error=self.target_error,
                estimated_mcmc_steps=estimated_mcmc_steps,
                pilot_queue_label=self.pilot_queue_label,
                walker_ratio=walker_ratio,
                pilot_wall_sec=pilot_wall_sec,
                net_pilot_sec=net_pilot_sec or 0,
            )

        # -- Production runs ---------------------------------------
        _has_convergence_criteria = self.target_snr is not None or self.energy_slope_sigma_threshold is not None
        last_run = 0
        step_files = {}  # {step: (input, output, run_id)}
        _checked_fetched_convergence = False
        converged = converged_snr = converged_slope = None
        for i in range(1, self.max_continuation + 1):
            # Skip input generation if this step already has a job record
            recorded = get_job_by_step(_wd, i)
            status = recorded.get("status")
            if status == "fetched":
                logger.info(f"  step {i}: already fetched. Skipping.")
                step_files[i] = (recorded["input_file"], recorded["output_file"], recorded.get("run_id", ""))
                last_run = i
                continue
            if status in ("submitted", "completed"):
                input_i = recorded["input_file"]
                output_i = recorded["output_file"]
                run_id_i = recorded.get("run_id", "")
                logger.info(f"  step {i}: already {status}. Resuming...")
            else:
                # -- Re-evaluate convergence from fetched runs -----
                if _has_convergence_criteria and last_run > 0 and not _checked_fetched_convergence:
                    _checked_fetched_convergence = True
                    converged, converged_snr, converged_slope = self._check_convergence(
                        _wd,
                        step_files,
                        last_run,
                    )
                    if converged and self._early_stop_allowed(_wd):
                        logger.info(f"  Convergence already met after fetched runs (step {last_run}). No further runs needed.")
                        break

                run_id_i = self._new_run_id()
                input_i = self._input_filename(i, run_id_i)
                output_i = self._output_filename(i, run_id_i)
                if i == 1:
                    restart_chk = self._resolve_external_restart_chk(_wd)
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=restart_chk is not None,
                        restart_chk=restart_chk,
                    )
                else:
                    restart_chk = self._find_restart_chk(_wd)
                    if restart_chk is None:
                        raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
                    self._generate_input(
                        estimated_mcmc_steps,
                        self.num_opt_steps,
                        os.path.join(_wd, input_i),
                        restart=bool(restart_chk),
                        restart_chk=restart_chk,
                    )
                logger.info("")
                logger.info(f"-- VMC production run {i}/{self.max_continuation} " + "-" * 16)
                logger.info(f"  {input_i}: {self.num_opt_steps} opt steps x {estimated_mcmc_steps} MCMC steps/step")

            restart_chk = self._find_restart_chk(_wd) if i > 1 else self._resolve_external_restart_chk(_wd)
            if i > 1 and restart_chk is None:
                raise RuntimeError(f"No restart checkpoint found for continuation run {i}. Expected .h5 file in {_wd}")
            extra_from = [restart_chk] if restart_chk else []

            # Estimate run time from the latest available output
            _prod_cost = self.num_opt_steps * estimated_mcmc_steps
            _ref_net = None
            for _j in range(i - 1, 0, -1):
                if _j not in step_files:
                    continue
                _ref_net = parse_net_time(os.path.join(_wd, step_files[_j][1]))
                if _ref_net and _ref_net > 0:
                    # All production runs have the same step count
                    logger.info(f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net)}")
                    break
            else:
                # First production run: use pilot output
                _pilot_outs = sorted(glob.glob(os.path.join(_wd, "_pilot", "*.out")))
                _ref_net = parse_net_time(_pilot_outs[-1]) if _pilot_outs else None
                if _ref_net and _ref_net > 0:
                    _p_vmc = estimation.get("pilot_vmc_steps") or self.pilot_vmc_steps
                    _p_mcmc = estimation.get("pilot_mcmc_steps") or self.pilot_mcmc_steps
                    _pilot_cost = _p_vmc * _p_mcmc
                    if _pilot_cost > 0:
                        logger.info(
                            f"  est. Net run time (w/o JAX compilation) = {_format_duration(_ref_net * _prod_cost / _pilot_cost)}"
                        )

            await self._submit_and_wait(input_i, output_i, work_dir=_wd, extra_from_objects=extra_from, step=i, run_id=run_id_i)
            step_files[i] = (input_i, output_i, run_id_i)
            last_run = i
            converged = converged_snr = converged_slope = None

            logger.info(f"  VMC production run {i}/{self.max_continuation} completed.")

            # Refresh output_values["energy"] from THIS iteration's log
            # before validating -- otherwise validate_completion sees a
            # stale or unset energy and the non-finite check at
            # ``state.py`` silently passes for diverged runs.
            self._parse_output(os.path.join(_wd, output_i))

            # -- Abnormal-termination guard (single source of truth) --
            # target_error=None -> only Program-ends / non-finite-energy
            # checks; SNR/slope convergence is evaluated separately below.
            vstatus, vmsg = validate_completion(_wd, self.output_values)
            if vstatus == CompletionStatus.FAILED:
                logger.error(vmsg)
                self.output_values["error"] = vmsg
                self.status = WorkflowStatus.FAILED
                break

            # -- Early exit if convergence criteria met ------------
            if _has_convergence_criteria and i < self.max_continuation:
                converged, converged_snr, converged_slope = self._check_convergence(
                    _wd,
                    step_files,
                    last_run,
                )
                if converged and self._early_stop_allowed(_wd):
                    logger.info(f"  Convergence achieved at run {i}/{self.max_continuation}. Stopping early.")
                    break

        # -- Collect outputs ---------------------------------------
        h5_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.h5")))
        output_logs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(_wd, "*.out")))
        self.output_files = h5_files + output_logs

        await self._publish_handoff(_wd)
        restart_chk = self._find_restart_chk(_wd)
        if restart_chk:
            self.output_values["checkpoint"] = restart_chk
        self.output_values["estimated_mcmc_steps"] = estimated_mcmc_steps

        # Parse last production output for energy
        if last_run in step_files:
            last_output = os.path.join(_wd, step_files[last_run][1])
            self._parse_output(last_output)

        # -- Final convergence check -------------------------------
        if converged is None:
            converged, converged_snr, converged_slope = self._check_convergence(
                _wd,
                step_files,
                last_run,
            )
        if not _has_convergence_criteria:
            logger.info("  No convergence criteria set; treating as converged.")
        elif converged:
            logger.info("  VMC converged (all active checks passed).")
        else:
            reasons = []
            if not converged_snr:
                reasons.append(f"S/N ({self.output_values.get('signal_to_noise', '?'):.3f}) > target ({self.target_snr})")
            if not converged_slope:
                reasons.append(
                    f"energy slope ({self.output_values.get('energy_slope', '?'):.6f} Ha/step) still significantly negative"
                )
            msg = f"VMC NOT converged after {self.max_continuation} continuation run(s): {'; '.join(reasons)}"
            logger.error(f"  {msg}")
            self.status = WorkflowStatus.FAILED

        if self.status != WorkflowStatus.FAILED:
            self.status = WorkflowStatus.COMPLETED
        return self.status, self.output_files, self.output_values

    # -- Utility methods -------------------------------------------

    def _check_convergence(
        self,
        work_dir: str,
        step_files: dict,
        last_run: int,
    ) -> tuple:
        """Evaluate SNR and energy-slope convergence criteria.

        Returns ``(converged, converged_snr, converged_slope)`` where
        each flag is *True* when the criterion is met or not active.
        """
        converged_snr = True
        converged_slope = True

        # -- (A) SNR check --
        if self.target_snr is not None:
            all_snr = []
            for j in range(last_run, 0, -1):
                if j not in step_files:
                    continue
                out_j = os.path.join(work_dir, step_files[j][1])
                all_snr = self._parse_all_snr(out_j) + all_snr
                if len(all_snr) >= self.snr_avg_window:
                    break
            if all_snr:
                window = all_snr[-self.snr_avg_window :]
                avg_snr = sum(window) / len(window)
                self.output_values["signal_to_noise"] = avg_snr
                self.output_values["signal_to_noise_last"] = all_snr[-1]
                logger.info(f"  S/N avg over last {len(window)} step(s): {avg_snr:.3f}  (last step: {all_snr[-1]:.3f})")
                converged_snr = avg_snr <= self.target_snr
            else:
                converged_snr = False
                logger.warning("  Could not parse S/N from production output.")

        # -- (B) Energy slope check --
        if self.energy_slope_sigma_threshold is not None:
            all_energies: list[tuple[float, float]] = []
            for j in range(last_run, 0, -1):
                if j not in step_files:
                    continue
                out_j = os.path.join(work_dir, step_files[j][1])
                all_energies = self._parse_all_energies(out_j) + all_energies
                if len(all_energies) >= self.energy_slope_window_size:
                    break

            window_e = all_energies[-self.energy_slope_window_size :]
            if len(window_e) >= 2:
                energies = [e for e, _ in window_e]
                errors = [s for _, s in window_e]
                slope, slope_std = self._fit_energy_slope(energies, errors)
                self.output_values["energy_slope"] = slope
                self.output_values["energy_slope_std"] = slope_std
                logger.info(f"  Energy slope over last {len(window_e)} step(s): {slope:.6f} \u00b1 {slope_std:.6f} Ha/step")
                converged_slope = slope >= -slope_std * self.energy_slope_sigma_threshold
                if converged_slope:
                    logger.info(
                        f"  Energy plateau: slope ({slope:.6f} Ha/step) >= "
                        f"-{slope_std:.6f} * {self.energy_slope_sigma_threshold}"
                    )
                else:
                    logger.info(
                        f"  Energy still decreasing: slope ({slope:.6f} Ha/step) < "
                        f"-{slope_std:.6f} * {self.energy_slope_sigma_threshold}"
                    )
            else:
                converged_slope = False
                logger.warning(f"  Not enough energy data for slope check ({len(window_e)} < 2 steps).")

        converged = converged_snr and converged_slope
        return converged, converged_snr, converged_slope

    def _resolve_external_restart_chk(self, work_dir: str) -> str | None:
        """Return the externally supplied checkpoint basename, or ``None``.

        ``external_restart_chk`` names a checkpoint that a *previous,
        unrelated* run produced and that the ``Container`` has staged into
        *work_dir* as an input.  It may be a plain ``restart.h5`` file or a
        twist checkpoint DIRECTORY (``restart.h5/shared_state.h5`` plus
        ``restart.h5/twist_NNN.h5``); the jqmc driver resumes from either,
        skipping the optimisation steps already certified by the checkpoint.

        Raises:
            FileNotFoundError: the name was configured but nothing of that
                name exists in *work_dir*.  Failing here means no 64-GPU job
                is submitted that would silently restart from scratch.
        """
        if not self.external_restart_chk:
            return None
        name = os.path.basename(self.external_restart_chk)
        path = os.path.join(work_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"external_restart_chk='{name}' was configured but is not present in {work_dir}. "
                f"Add it to the Container's input_files (a twist checkpoint directory is copied "
                f"recursively) so the continuation actually resumes."
            )
        return name

    def _find_restart_chk(self, work_dir: str) -> str | None:
        """Locate a VMC restart checkpoint file in *work_dir*."""
        for pattern in ["restart.h5", "vmc.h5", "*.h5"]:
            matches = sorted(glob.glob(os.path.join(work_dir, pattern)))
            if matches:
                return os.path.basename(matches[-1])
        return None

    def _latest_optimized_hamiltonian(self, work_dir: str) -> str:
        """Basename of the optimizer's newest wavefunction dump.

        ONE path for every run: jQMC writes ``hamiltonian_data_opt_step_N.h5``
        every ``wf_dump_freq`` steps, and under twist averaging that file is a
        complete combined Hamiltonian holding every twist, so it is directly
        usable as the next step's ``hamiltonian_h5`` either way.  Raises when no
        dump exists -- there is nothing to hand downstream then, and guessing
        (an input file, a single twist of a twist-averaged run) would corrupt
        the next stage silently.
        """

        def _step_number(path: str) -> int:
            m = re.search(r"hamiltonian_data_opt_step_(\d+)\.h5$", path)
            return int(m.group(1)) if m else -1

        opt_files = sorted(glob.glob(os.path.join(work_dir, "hamiltonian_data_opt_step_*.h5")), key=_step_number)
        if not opt_files:
            raise RuntimeError(
                f"VMC in {work_dir} produced no hamiltonian_data_opt_step_*.h5: there is no optimized "
                "Hamiltonian to publish. Check the jqmc output log for the cause."
            )
        return os.path.basename(opt_files[-1])

    # -- Hand-off of the optimized wavefunction --------------------

    @staticmethod
    def _numbered_opt_dumps(work_dir: str) -> dict[int, str]:
        """Basenames of the optimizer dumps in *work_dir*, keyed by step number."""
        dumps: dict[int, str] = {}
        for path in glob.glob(os.path.join(work_dir, "hamiltonian_data_opt_step_*.h5")):
            name = os.path.basename(path)
            match = _OPT_STEP_DUMP.fullmatch(name)
            if not match or not os.path.isfile(path):
                continue
            step = int(match.group(1))
            if step in dumps:
                raise RuntimeError(f"Ambiguous dumps for optimization step {step} in {work_dir}: {dumps[step]} and {name}.")
            dumps[step] = name
        return dumps

    def _select_handoff_dumps(self, work_dir: str) -> list[str]:
        """The last ``handoff_average_steps`` dumps by step number, in step order.

        Refuses (``RuntimeError``) when there are fewer dumps than requested
        (unless ``handoff_allow_fewer``) and when the selected steps are not
        consecutive (a dump frequency above 1 or deleted dumps): the average
        is then not the one requested, and the workflow does not substitute
        another hand-off.
        """
        dumps = self._numbered_opt_dumps(work_dir)
        if not dumps:
            raise RuntimeError(
                f"VMC in {work_dir} produced no hamiltonian_data_opt_step_*.h5: there is no optimized "
                "Hamiltonian to publish. Check the jqmc output log for the cause."
            )
        count = self.handoff_average_steps
        steps = sorted(dumps)[-count:]
        if len(steps) < count and not self.handoff_allow_fewer:
            raise RuntimeError(
                f"VMC in {work_dir} has only {len(steps)} hamiltonian_data_opt_step_*.h5 dump(s) (steps {steps[0]}-"
                f"{steps[-1]}), but handoff_average_steps={count} averages the last {count} optimization steps. Run "
                "more steps, pass handoff_allow_fewer=True to average the available dumps, or handoff_mode='last' "
                "to publish the last dump only."
            )
        if steps[-1] - steps[0] + 1 != len(steps):
            raise RuntimeError(
                f"The last {len(steps)} dumps of {work_dir} are not consecutive optimization steps ({steps}): the "
                "averaged hand-off needs every step of its window dumped (wf_dump_freq=1, no deleted dumps). Pass "
                "handoff_mode='last' to publish the last dump only."
            )
        return [dumps[step] for step in steps]

    def _average_handoff_dumps(self, work_dir: str, sources: list[str]) -> None:
        """Run ``jqmc-tool vmc average-wf`` on *sources* (basenames in *work_dir*).

        The command runs in *work_dir* with the interpreter of this process,
        so the averaged file records the dump basenames as its provenance and
        the jQMC installation that runs the workflow does the averaging.  A
        refusal by the tool (inputs that are not steps of one optimization,
        a non-finite parameter, an MO gauge jump, ...) is raised with the
        tool's own message; nothing is published then.
        """
        if os.path.basename(self.hamiltonian_file) == AVERAGED_HAMILTONIAN:
            raise RuntimeError(
                f"hamiltonian_file={self.hamiltonian_file!r} is the averaged hand-off's output name; rename the input."
            )
        # A refused average must not leave an earlier average behind under the published name.
        stale = os.path.join(work_dir, AVERAGED_HAMILTONIAN)
        if os.path.isfile(stale):
            os.remove(stale)
        command = [sys.executable, "-m", "jqmc.jqmc_tool", "vmc", "average-wf", *sources, "-o", AVERAGED_HAMILTONIAN]
        result = subprocess.run(command, cwd=work_dir, capture_output=True, text=True)
        for line in result.stderr.splitlines():
            if line.startswith("Warning:"):
                logger.warning(f"  average-wf: {line}")
        if result.returncode != 0:
            message = "\n".join(line for line in result.stderr.splitlines() if line.startswith("Error:"))
            raise RuntimeError(
                f"jqmc-tool vmc average-wf refused the last {len(sources)} dumps of {work_dir} "
                f"({sources[0]} .. {sources[-1]}), exit status {result.returncode}: "
                f"{message or result.stderr.strip() or result.stdout.strip()}"
            )

    def _publish_optimized_hamiltonian(self, work_dir: str) -> str:
        """Build the hand-off of *work_dir* and record it with its provenance in ``output_values``."""
        if self.handoff_mode == "last":
            sources = [self._latest_optimized_hamiltonian(work_dir)]
            published = sources[0]
        else:
            sources = self._select_handoff_dumps(work_dir)
            self._average_handoff_dumps(work_dir, sources)
            published = AVERAGED_HAMILTONIAN
        steps = [int(match.group(1)) if (match := _OPT_STEP_DUMP.fullmatch(name)) else -1 for name in sources]
        if self.handoff_mode == "average":
            logger.info(f"  Hand-off: {published} = average of {len(sources)} dumps (steps {steps[0]}-{steps[-1]}).")
        else:
            logger.info(f"  Hand-off: {published} (last dump).")
        self.output_values["optimized_hamiltonian"] = published
        self.output_values["optimized_hamiltonian_handoff"] = self.handoff_mode
        self.output_values["optimized_hamiltonian_sources"] = sources
        self.output_values["optimized_hamiltonian_steps"] = steps
        return published

    async def _publish_handoff(self, work_dir: str) -> None:
        """Publish ``optimized_hamiltonian`` at the end of a run (off the event loop).

        A failed averaged run publishes nothing: averaging a diverged
        trajectory would only replace the failure message with a refusal.
        """
        if self.status == WorkflowStatus.FAILED and self.handoff_mode == "average":
            logger.warning("  VMC failed: no averaged optimized Hamiltonian is published.")
            return
        published = await asyncio.to_thread(self._publish_optimized_hamiltonian, work_dir)
        if published not in self.output_files:
            self.output_files.append(published)

    def _early_stop_allowed(self, work_dir: str) -> bool:
        """Convergence may end the production runs early only once the averaged hand-off has its dumps."""
        if self.handoff_mode != "average" or self.handoff_allow_fewer:
            return True
        available = len(self._numbered_opt_dumps(work_dir))
        if available >= self.handoff_average_steps:
            return True
        logger.info(
            f"  Converged, but only {available} of the {self.handoff_average_steps} dumps that the averaged "
            "hand-off needs exist; running the next continuation."
        )
        return False

    # -- Output parsing --------------------------------------------

    def _parse_output(self, output_file=None):
        """Extract the last optimization step's energy from *output_file*.

        Delegates to :func:`output_parser._parse_vmc_log_text` so there
        is a single source of truth for VMC log parsing -- any future
        format change (or fix like nan/inf support) lives in one place.
        """
        if output_file is None or not os.path.isfile(output_file):
            return
        e, err = _last_opt_energy_from_log(output_file)
        if e is not None and err is not None:
            self.output_values["energy"] = e
            self.output_values["energy_error"] = err
            logger.info(f"  VMC energy: {e} +- {err} Ha")

    @staticmethod
    def _parse_all_snr(output_file):
        """Parse all signal-to-noise ratios from a VMC output file.

        Returns:
            list[float]:
                All maximum generalized-force signal-to-noise values in order, one per
                optimization step.  Empty list if the file is missing
                or contains no S/N lines.
        """
        if not os.path.isfile(output_file):
            return []

        snr_pattern = re.compile(r"Max generalized force signal-to-noise = ([-+]?\d+(?:\.\d+)?)")
        values = []
        try:
            with open(output_file) as f:
                for line in f:
                    m = snr_pattern.search(line)
                    if m:
                        values.append(float(m.group(1)))
        except Exception:
            return []
        return values

    @staticmethod
    def _parse_all_energies(output_file: str) -> list[tuple[float, float]]:
        """Extract per-step ``(energy, energy_error)`` from a VMC output file.

        Uses the existing ``_parse_vmc_log_text()`` parser to obtain
        :class:`VMC_Step_Data` and returns the energy/error pairs.

        Returns:
            list[tuple[float, float]]:
                ``[(E_1, sigma_1), (E_2, sigma_2), ...]`` in file order.
                Empty list if the file is missing or unparseable.
        """
        if not os.path.isfile(output_file):
            return []
        try:
            with open(output_file, errors="replace") as f:
                text = f.read()
            from .output_parser import _parse_vmc_log_text

            steps = _parse_vmc_log_text(text)
            return [(s.energy, s.energy_error) for s in steps if s.energy is not None and s.energy_error is not None]
        except OSError as exc:
            logger.warning(f"_parse_all_energies: cannot read {output_file}: {exc}")
            return []
        except Exception:
            # Log unexpected parser failures rather than swallowing silently.
            logger.exception(f"_parse_all_energies: unexpected error parsing {output_file}")
            return []

    @staticmethod
    def _fit_energy_slope(
        energies: list[float],
        energy_errors: list[float],
    ) -> tuple[float, float]:
        """Weighted linear regression of energy vs optimisation step.

        Model: ``E_k = a + b * k + eps_k``, weight ``w_k = 1 / sigma_k^2``.

        Args:
            energies (list[float]):
                Energy value per optimisation step (length *N* >= 2).
            energy_errors (list[float]):
                Statistical error per step (length *N*, positive).

        Returns:
            slope (float):
                Weighted least-squares slope *b*.
            slope_std (float):
                Standard error of *b*.
        """
        import numpy as np

        E = np.asarray(energies, dtype=float)
        sigma = np.asarray(energy_errors, dtype=float)
        # Replace non-positive sigmas with the median positive sigma so
        # they get a finite (non-inf) weight rather than dividing by zero.
        positive = sigma[sigma > 0]
        floor = float(np.median(positive)) if positive.size else 1.0
        sigma = np.where(sigma > 0, sigma, floor)
        w = 1.0 / sigma**2
        k = np.arange(len(E), dtype=float)

        S = np.sum(w)
        Sk = np.sum(w * k)
        Skk = np.sum(w * k**2)
        SE = np.sum(w * E)
        SkE = np.sum(w * k * E)

        delta = S * Skk - Sk**2
        b = (S * SkE - Sk * SE) / delta
        sigma_b = np.sqrt(S / delta)
        return float(b), float(sigma_b)

    @staticmethod
    def _parse_last_opt_energy(output_file):
        """Parse the last optimization step's energy from a VMC output file.

        Returns:
            tuple:
                ``(energy, error)`` or ``(None, None)``.  ``nan``/``inf``
                are returned as :class:`float` (not ``None``), so callers
                can apply ``math.isfinite`` to detect diverged runs.
        """
        if not os.path.isfile(output_file):
            return None, None
        return _last_opt_energy_from_log(output_file)
