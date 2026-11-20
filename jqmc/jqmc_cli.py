"""command-line module."""

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

# python modules
import copy
import os
import platform
import sys
from collections.abc import Callable
from importlib import import_module
from datetime import datetime
from logging import FileHandler, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np
import numpy.typing as npt
import toml

# MPI
from mpi4py import MPI
from uncertainties import ufloat

from ._build_info import log_build_info_once
from ._cli_entry import CLI_USAGE, parse_cli_arguments
from ._checkpoint import merge_rank_checkpoints

# jQMC
from ._driver_helpers import _check_mpi4py_jax_distribution_consistency, _initialize_jax_distributed, num_sep_line
from ._precision import configure as configure_precision
from ._precision import mode_label as precision_mode_label
from ._precision import zone_detail as precision_zone_detail
from ._restart_policy import is_frozen, restored_frozen_values
from ._setting import (
    GFMC_MIN_BIN_BLOCKS,
    GFMC_MIN_COLLECT_STEPS,
    GFMC_MIN_WARMUP_STEPS,
)
from .hamiltonians import Hamiltonian_data
from .jqmc_gfmc import GFMC_n, GFMC_t, _validate_gfmc_alat, _validate_gfmc_tau
from .jqmc_mcmc import MCMC
from .jqmc_miscs import cli_parameters
from .jqmc_optimizer_kernel import validate_lm_condition_cutoff
from .mpc import _validate_mpc_cutoff

if TYPE_CHECKING:
    from ._twist_average import TwistAverageResult

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# set logger
logger = getLogger("jqmc").getChild(__name__)


def _execution_backend_label() -> str:
    """Return the production backend selected by the CPU dispatch contract.

    Rust augments rather than replaces XLA: JAX still owns setup and the parts of
    the workflow outside the native hot loops.  Report ``XLA + Rust`` only when
    the same flag/platform/precision conditions used by the production drivers
    hold and the native extension can actually be imported.
    """
    from . import _setting

    env_rust = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if env_rust is None:
        rust_requested = _setting.flag_rust_on_CPU
    else:
        rust_requested = env_rust.strip().lower() not in ("0", "false", "no", "off", "")

    if not rust_requested or jax.default_backend() != "cpu":
        return "XLA"

    try:
        import_module("jqmc_rs")
    except (ImportError, OSError):
        return "XLA"
    return "XLA + Rust"


def _rs_thread_line() -> str:
    """The ``Rust rayon threads = N`` banner line (call only when Rust is active).

    Initializes the rayon pool via ``configure_threads(None)`` BEFORE reading the
    width: ``current_num_threads()`` on an uninitialized pool would freeze it at
    rayon's own default, and the drivers' later env-honoring init could no longer
    resize it (``build_global`` is once-per-process). Same precedence as the
    drivers: ``JQMC_RS_NUM_THREADS`` > ``RAYON_NUM_THREADS`` > affinity default,
    so the printed width is the one the batch harness subsequently uses.
    """
    import jqmc_rs

    jqmc_rs.configure_threads(None)
    return f"Rust rayon threads = {jqmc_rs.current_num_threads()} (per MPI rank)."


def _jax_environment_info() -> str:
    """``jax.print_environment_info`` with its ``platform:`` line reduced to OS family and architecture.

    JAX prints ``platform: uname_result(...)``, which carries the host name and
    the full kernel release and build (issue 910). Logs are kept as provenance
    and shared, and neither value is needed to reproduce a calculation, so the
    line becomes ``platform: <system> <machine>`` (for example ``Linux x86_64``).
    Every other line, including the versions, devices and the ``JAX_*``/``XLA_*``
    variables, is unchanged.
    """
    summary = f"platform: {platform.system()} {platform.machine()}"
    lines = jax.print_environment_info(return_string=True).splitlines()
    return "\n".join(summary if line.startswith("platform:") else line for line in lines)


def _print_header() -> None:
    """Print Header."""
    logger.info(f"Program starts {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("-" * 48)
    logger.info("      d8b  .d88888b.  888b     d888  .d8888b.   ")
    logger.info('      Y8P d88P" "Y88b 8888b   d8888 d88P  Y88b  ')
    logger.info("          888     888 88888b.d88888 888    888  ")
    logger.info("     8888 888     888 888Y88888P888 888         ")
    logger.info("      888 888     888 888 Y888P 888 888         ")
    logger.info("      888 888 Y8b 888 888  Y8P  888 888    888  ")
    logger.info("      888 Y88b.Y8b88P 888       888 Y88b  d88P  ")
    logger.info('      888  "Y888888"  888       888   Y8888P    ')
    logger.info("      888        Y8b                            ")
    logger.info("     d88P                                       ")
    logger.info("   888P                           Made on earth ")
    logger.info("-" * 48)
    logger.info("")
    logger.info("jQMC: Python-based real-space ab-initio Quantum Monte Carlo package.")
    log_build_info_once(logger)
    logger.info("")
    logger.info("Authors: Kosuke Nakano [kousuke_1123@icloud.com]")
    logger.info("")
    logger.info("***The 3-Clause BSD License***")
    logger.info("")
    logger.info("Copyright (C) 2024- Kosuke Nakano")
    logger.info("All rights reserved.")
    logger.info("")


def _print_footer() -> None:
    """Print Footer."""
    logger.info(f"Program ends {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")


def _resolve_twist_dispatch(
    hamiltonian_h5: str,
    *,
    restart: bool = False,
    restart_chk: str | None = None,
) -> Literal["single", "twist_average"]:
    """Select the CLI path from the source this run will actually consume.

    An existing restart file is a self-contained single-run checkpoint: its
    loader validates the payload, and the unused initialization Hamiltonian
    must not be read. An existing restart directory selects the twist driver,
    which validates its combined input and checkpoint shards. Neither path
    catches read errors or tries another format.

    Fresh runs, and twist resumes whose checkpoint directory does not yet
    exist, select their route from the current Hamiltonian input. A combined
    input takes the twist path even when it contains only one twist.

    Returns:
        ``"twist_average"`` for a combined twist input, otherwise
        ``"single"``.
    """
    from ._twist_average import is_twist_average_input

    if restart and restart_chk:
        if os.path.isfile(restart_chk):
            return "single"
        if os.path.isdir(restart_chk):
            return "twist_average"
    return "twist_average" if is_twist_average_input(hamiltonian_h5) else "single"


def _log_twist_average_result(result: "TwistAverageResult", *, with_force: bool) -> None:
    """Format + emit the TA energy / force summary.

    Mirrors the single-twist ``logger.info(...)`` block but adds the
    twist dispersion ``s_twist`` -- reported separately from the MC
    error -- and a per-twist table, whose values feed the diagnostic
    "shell-filling oscillation" plot.
    """
    energy = result.energy
    logger.info("Final output(s):")
    logger.info(
        f"  BZ-averaged Energy: E_TA = {energy.E_TA:.5f} +- {energy.E_TA_std:.5f} Ha (s_twist = {energy.s_twist:.5f} Ha)."
    )
    if result.mpc is not None:
        correction = result.mpc.correction
        corrected = result.mpc.energy
        logger.info(
            f"  BZ-averaged MPC correction: dE_MPC = {correction.E_TA:+.5f} +- "
            f"{correction.E_TA_std:.5f} Ha (cutoff = {result.mpc.cutoff} Ha)."
        )
        logger.info(f"  BZ-averaged Total Energy (MPC-corrected): E_MPC = {corrected.E_TA:.5f} +- {corrected.E_TA_std:.5f} Ha.")
    sep = 16 * 3
    logger.info("  Per-twist energy (Level-1 outputs):")
    logger.info("  " + "-" * sep)
    logger.info("  twist  k_point                 E_k (Ha)")
    logger.info("  " + "-" * sep)
    for ik, (k, Ek, sigma_k) in enumerate(zip(result.k_points, result.per_twist_E, result.per_twist_E_std)):
        k_repr = "Gamma" if k is None else f"({k[0]:+.3f},{k[1]:+.3f},{k[2]:+.3f})"
        val = f"{ufloat(float(Ek), float(sigma_k)):+2uS}"
        logger.info(f"  {ik:5d}  {k_repr.ljust(22)}  {val}")
    logger.info("  " + "-" * sep)
    if with_force and result.force is not None:
        logger.info("  BZ-averaged Atomic Forces:")
        F = result.force.F_TA
        F_std = result.force.F_TA_std
        logger.info("  " + "-" * sep)
        logger.info("  atom    Fx(Ha/bohr)  Fy(Ha/bohr)  Fz(Ha/bohr)")
        logger.info("  " + "-" * sep)
        for ia in range(F.shape[0]):
            row = [f"{ufloat(float(F[ia, d]), float(F_std[ia, d])):+2uS}" for d in range(3)]
            logger.info("  " + f"{ia:4d}    " + " ".join(v.ljust(12) for v in row))
        logger.info("  " + "-" * sep)
    if result.stress is not None:
        T = result.stress.F_TA
        T_std = result.stress.F_TA_std
        P_hydro_TA = result.P_hydro
        _log_stress_section(
            T,
            T_std,
            -np.diag(T),
            np.diag(T_std),
            (float(P_hydro_TA.F_TA[0]), float(P_hydro_TA.F_TA_std[0])),
        )


def _collect_mpc_estimates(
    driver: object,
    *,
    num_mcmc_warmup_steps: int,
    num_mcmc_bin_blocks: int,
) -> tuple[float, float, float, float]:
    """Collect the MPC correction and covariance-aware corrected total.

    Keeping this routing shared prevents GFMC CLI branches from accidentally
    forwarding the unrelated VMC warmup local to either estimator.
    """
    kwargs = {
        "num_mcmc_warmup_steps": num_mcmc_warmup_steps,
        "num_mcmc_bin_blocks": num_mcmc_bin_blocks,
    }
    dE_mpc_mean, dE_mpc_std = driver.get_dE_mpc(**kwargs)
    E_mpc_mean, E_mpc_std = driver.get_E_mpc(**kwargs)
    return float(dE_mpc_mean), float(dE_mpc_std), float(E_mpc_mean), float(E_mpc_std)


def _validate_mpc_job_type(job_type: str, *, mpc: bool) -> None:
    """Reject MPC where the CLI cannot publish a well-defined final value."""
    if job_type == "vmc" and mpc:
        raise ValueError(
            "[coulomb].mpc = true is not supported for job_type = 'vmc': wavefunction optimization does not "
            "publish a final MPC estimate. Run an MCMC or LRDMC sampling job after optimization."
        )


def _reject_single_file_twist_checkpoint(restart_chk: str) -> None:
    """Refuse a regular file where the twist-averaged path needs a directory.

    The twist-averaged path stores one ``twist_NNN.h5`` per twist plus
    ``shared_state.h5`` inside the ``restart_chk`` *directory*. A regular file
    at that path (a single-file checkpoint such as the ``restart.h5`` left by
    a single-k run) can neither be resumed nor be
    replaced by that directory, so the run would compute every twist and then
    fail when saving. Stop before any computation instead, for ``restart``
    true or false and for the default name alike. No conversion is attempted.

    Raises:
        ValueError: If ``restart_chk`` names an existing regular file.
    """
    if not restart_chk or not os.path.isfile(restart_chk):
        return
    raise ValueError(
        f"[control] restart_chk = '{restart_chk}' is an existing regular file, but this Hamiltonian is a "
        "twist-averaged (combined) input. Twist-averaged runs, including a one-twist input, use restart_chk "
        "as a checkpoint DIRECTORY that holds one twist_NNN.h5 per twist plus shared_state.h5. A single-file "
        "restart checkpoint (such as the restart.h5 left by a single-k run) cannot be "
        "resumed by, or converted to, this layout. To proceed, set [control] restart_chk to a directory name "
        "that is not an existing file (for example 'restart_twists'), or move the existing file away. With "
        "restart = true the twist-averaged run then starts from the beginning in that directory."
    )


def _merge_section_into_parameters(parameters: dict, dict_toml: dict, section: str) -> None:
    """Mirror of the per-section override loop the single-twist job_type
    blocks run on their entry; called from
    :func:`_run_twist_average_dispatch` so the dispatch sees the user's
    toml-supplied values (otherwise ``parameters[section]`` would still
    carry the cli_parameters defaults, including ``None`` sentinels
    that would crash the inner driver)."""
    if section not in parameters:
        return
    toml_section = dict_toml.get(section, {}) or {}
    for key in parameters[section].keys():
        if key in toml_section:
            parameters[section][key] = toml_section[key]
        elif parameters[section][key] is None:
            raise ValueError(
                f"twist-averaged dispatch: required key [{section}].{key} is not specified in the toml and has no default."
            )


def _log_restored_from_checkpoint(section: str, dict_toml: dict, checkpoint_driver_config: dict, mpi_size: int) -> None:
    """Log what a restart takes from the checkpoint instead of the TOML.

    The frozen keys omitted from the continuation TOML are logged with their
    checkpoint value, and the MPI/walker layout with the same lines a fresh
    run prints, so that the log of a restart states the values actually used.
    """
    for toml_section, key, value in restored_frozen_values(section, dict_toml, checkpoint_driver_config):
        if value is None:
            logger.info(f"[{toml_section}] {key} is restored from the checkpoint.")
        else:
            logger.info(f"The value of {key} = {value} (restored from the checkpoint).")
    logger.info(f"The number of MPI process = {mpi_size}.")
    if "num_walkers" in checkpoint_driver_config:
        logger.info(f"The number of walkers assigned for each MPI process = {checkpoint_driver_config['num_walkers']}.")


def _restart_force_stress_switches(section: str, dict_toml: dict, atomic_force, compute_stress, driver) -> tuple:
    """Force/stress switches of a restart: an omitted key takes the restored driver's value.

    ``atomic_force`` and ``compute_stress`` are frozen on restart. When the
    continuation TOML omits them, the run reports what the checkpoint was
    sampled with (#316) instead of the ``jqmc_miscs`` default. A key the TOML
    sets explicitly keeps the TOML value, as before.
    """
    user_section = dict_toml.get(section) or {}
    if "atomic_force" not in user_section:
        atomic_force = driver.comput_position_deriv
    if "compute_stress" not in user_section:
        compute_stress = driver.comput_stress
    return atomic_force, compute_stress


def _merged_optimizer_kwargs(user_value: object) -> dict:
    """Return the ``[vmc].optimizer_kwargs`` table merged onto the documented defaults.

    The generic per-key override replaces the whole sub-table, so a partial
    user table would otherwise silently drop every documented default it does
    not repeat (e.g. ``delta`` would fall back to the optimizer's internal
    1e-3 instead of the documented 0.15).
    """
    merged = copy.deepcopy(cli_parameters["vmc"]["optimizer_kwargs"])
    if user_value is None:
        return merged
    if not isinstance(user_value, dict):
        raise TypeError("optimizer_kwargs must be a dictionary when provided in the VMC section.")
    merged.update(user_value)
    return merged


# The four documented [control] job_type values (jqmc_miscs.control_comments["job_type"]).
_SUPPORTED_JOB_TYPES = ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau")
_KNOWN_TOML_SECTIONS = ("control", *_SUPPORTED_JOB_TYPES, "coulomb", "precision")
# These documented optimizer options default in run_optimize, outside the shipped TOML template.
_OPTIMIZER_KEYS_WITH_DRIVER_DEFAULTS = {"asr_gamma_max", "asr_fallback_gamma", "lm_reference_weight_min"}


def _validate_job_type(job_type) -> None:
    """Reject a [control] job_type outside the documented set before any input is read.

    No job runs for any other value, so a typo must not end as an apparent
    success (``Program ends``, exit status 0).
    """
    if not (isinstance(job_type, str) and job_type in _SUPPORTED_JOB_TYPES):
        raise ValueError(f"Unsupported [control] job_type = {job_type!r}. Expected one of {' / '.join(_SUPPORTED_JOB_TYPES)}.")


def _warn_unknown_toml_keys(dict_toml: dict, job_type: str) -> None:
    """Warn about TOML tables/keys the CLI does not read (typos are otherwise silent)."""
    unknown_sections = sorted(set(dict_toml) - set(_KNOWN_TOML_SECTIONS))
    if unknown_sections:
        logger.warning(f"Unknown TOML sections ignored: {unknown_sections}.")
    for section in ("control", job_type):
        user_section = dict_toml.get(section)
        if section not in cli_parameters or not isinstance(user_section, dict):
            continue
        unknown_keys = sorted(set(user_section) - set(cli_parameters[section]))
        if unknown_keys:
            logger.warning(f"Unknown [{section}] keys ignored: {unknown_keys}.")
        optimizer_kwargs = user_section.get("optimizer_kwargs")
        if section == "vmc" and isinstance(optimizer_kwargs, dict):
            known_optimizer_keys = cli_parameters[section]["optimizer_kwargs"].keys() | _OPTIMIZER_KEYS_WITH_DRIVER_DEFAULTS
            unknown_optimizer_keys = sorted(optimizer_kwargs.keys() - known_optimizer_keys)
            if unknown_optimizer_keys:
                logger.warning("Unknown [vmc].optimizer_kwargs keys ignored: %s.", unknown_optimizer_keys)
    for section in ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"):
        if section != job_type and section in dict_toml:
            logger.warning(f"[{section}] is ignored because job_type = {job_type!r}.")


def _validate_toml_boolean_keys(dict_toml: dict) -> None:
    """Reject a non-boolean value for a boolean-valued key the CLI reads (#599).

    The boolean keys are those whose ``cli_parameters`` default is a ``bool``,
    including the nested ``[vmc].optimizer_kwargs`` options. A quoted ``"false"``
    or a number would otherwise be read by Python truthiness and silently switch
    the option (e.g. a restart from a stale checkpoint). The sections checked are
    those the CLI reads: ``[control]``, ``[coulomb]`` and the ``job_type`` section.
    Non-table sections are left to the existing format checks.

    Raises:
        ValueError: If such a key holds anything other than a TOML boolean.
    """
    control = dict_toml.get("control")
    job_type = control.get("job_type") if isinstance(control, dict) else None
    job_section = job_type if isinstance(job_type, str) else None
    for section in dict.fromkeys(("control", "coulomb", job_section)):
        user_section = dict_toml.get(section)
        if section not in _KNOWN_TOML_SECTIONS or section not in cli_parameters or not isinstance(user_section, dict):
            continue
        checks = []
        for key, default in cli_parameters[section].items():
            if isinstance(default, bool) and key in user_section:
                checks.append((key, user_section[key]))
            elif isinstance(default, dict) and isinstance(user_section.get(key), dict):
                nested = user_section[key]
                checks += [(f"{key}.{sub}", nested[sub]) for sub, d in default.items() if isinstance(d, bool) and sub in nested]
        for name, value in checks:
            if not isinstance(value, bool):
                raise ValueError(f"[{section}] {name} must be a TOML boolean (true or false, without quotes), got {value!r}.")


class _LazyTwistHamiltonians:
    """List-like per-rank view that loads a twist only when the driver selects it.

    Twist orchestrators need ``len`` and indexed access, but a multi-twist MPI rank
    samples only its assigned index.  Eagerly materialising every HDF5 twist on every
    rank multiplies large PW/B-spline coefficient storage by ``n_twist`` before the
    native backend is even built.
    """

    def __init__(
        self,
        filepath: str,
        n_twist: int,
        loader: Callable[[str, int], Hamiltonian_data],
        k_points: "list | None" = None,
    ) -> None:
        self._filepath = filepath
        self._n_twist = int(n_twist)
        self._loader = loader
        self._cache = {}
        # Cheap per-twist k-point table (from the combined ``/twists`` header via
        # ``load_twist_average_input``). The twist drivers duck-type on this
        # attribute to obtain every twist's k WITHOUT deserialising every
        # Hamiltonian on every rank; ``None`` keeps the fallback (full
        # iteration) for callers that do not have the header at hand.
        self.twist_k_points = list(k_points) if k_points is not None else None

    def __len__(self) -> int:
        return self._n_twist

    def __getitem__(self, index: int | slice) -> Hamiltonian_data | list[Hamiltonian_data]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n_twist))]
        index = int(index)
        if index < 0:
            index += self._n_twist
        if not 0 <= index < self._n_twist:
            raise IndexError(index)
        if index not in self._cache:
            self._cache[index] = self._loader(self._filepath, index)
        return self._cache[index]


def _run_twist_average_dispatch(
    *,
    job_type: str,
    hamiltonian_h5: str,
    parameters: dict,
    dict_toml: dict,
    mcmc_seed: int,
    number_of_walkers: int,
    mpc: bool,
    mpc_cutoff: float,
    max_time: int,
    restart: bool,
    restart_chk: str,
    mpi_comm: MPI.Comm,
    mpi_rank: int,
) -> None:
    """Twist-averaged dispatch -- runs the right
    ``run_twist_average_*`` for ``job_type`` on every twist of the combined
    input file and logs the BZ-mean summary.

    ``restart_chk`` is interpreted as a *checkpoint directory* in this
    mode (``twist_NNN.h5 + shared_state.h5`` layout). When ``restart``
    is true the directory is read for completed twists which are
    skipped; missing twists run as usual and their checkpoints are
    written. An existing regular file at ``restart_chk`` is rejected here,
    before the input is read or any twist runs.
    """
    _reject_single_file_twist_checkpoint(restart_chk)

    from ._twist_average import (
        load_twist,
        load_twist_average_input,
        run_twist_average_from_hamiltonians,
        run_twist_average_gfmc,
        run_twist_average_optimize,
    )

    meta = load_twist_average_input(hamiltonian_h5)
    # Drivers select one twist per MPI rank after communicator splitting. Keep the
    # list API they accept, but do not duplicate every PW/B-spline Hamiltonian on
    # every rank before that selection has happened. The header k-points ride
    # along so the drivers' metadata pass does not force a full materialise.
    ham_list = _LazyTwistHamiltonians(hamiltonian_h5, meta.n_twist, load_twist, k_points=meta.k_points)
    weights = meta.weights
    n_twist = int(meta.n_twist)
    logger.info(
        "Twist-averaged dispatch: %d twists detected in %s (BZ sum w = %.6f).",
        n_twist,
        hamiltonian_h5,
        float(meta.bz_weight_sum),
    )

    def log_summary() -> None:
        # Printed after the section checks, just before the twist driver runs.
        # Rank 0 samples twist 0 (split_world_for_twists assigns rank // group
        # size), so this loads only the Hamiltonian the driver reuses from the
        # lazy view.
        if mpi_rank == 0:
            _log_hamiltonian_summary(ham_list[0])

    # The checkpoint directory is shared across all job_types; on a
    # fresh run we write to it, on resume we skip completed twists.
    checkpoint_dir = restart_chk if restart_chk else None

    if job_type == "mcmc":
        section = "mcmc"
        _merge_section_into_parameters(parameters, dict_toml, section)
        sec = parameters[section]
        mcmc_kwargs = {
            "num_mcmc_per_measurement": sec["num_mcmc_per_measurement"],
            "Dt": sec["Dt"],
            "epsilon_AS": sec["epsilon_AS"],
            "use_swct": sec["use_swct"],
            "mpc": mpc,
            "mpc_cutoff": mpc_cutoff,
            "comput_log_WF_param_deriv": bool(sec["parameter_derivatives"]),
            "retain_derivative_sources": False,
        }
        log_summary()
        result = run_twist_average_from_hamiltonians(
            hamiltonian_list=ham_list,
            weights=weights,
            mcmc_seed=mcmc_seed,
            num_walkers=number_of_walkers,
            num_mcmc_steps=sec["num_mcmc_steps"],
            num_mcmc_warmup_steps=sec["num_mcmc_warmup_steps"],
            num_mcmc_bin_blocks=sec["num_mcmc_bin_blocks"],
            max_time=max_time,
            world_comm=mpi_comm,
            compute_force=bool(sec["atomic_force"]),
            compute_stress=bool(sec["compute_stress"]),
            mcmc_kwargs=mcmc_kwargs,
            checkpoint_dir=checkpoint_dir,
            resume=bool(restart),
        )
        _log_twist_average_result(result, with_force=bool(sec["atomic_force"]))
        return

    if job_type == "vmc":
        section = "vmc"
        _merge_section_into_parameters(parameters, dict_toml, section)
        sec = parameters[section]
        # ``fix_geminal_rank=0`` is the 'not set' sentinel in the new
        # defaults; map it back to None for the run_optimize boundary.
        _fix_rank_raw = sec.get("fix_geminal_rank", 0)
        fix_rank_val = int(_fix_rank_raw) if int(_fix_rank_raw) > 0 else None
        optimizer_kwargs = _merged_optimizer_kwargs(sec.get("optimizer_kwargs"))
        optimize_kwargs = {
            "num_mcmc_steps": sec["num_mcmc_steps"],
            "num_opt_steps": sec["num_opt_steps"],
            "wf_dump_freq": sec["wf_dump_freq"],
            "num_mcmc_warmup_steps": sec["num_mcmc_warmup_steps"],
            "num_mcmc_bin_blocks": sec["num_mcmc_bin_blocks"],
            "max_time": max_time,
            "opt_J1_param": sec["opt_J1_param"],
            "opt_J2_param": sec["opt_J2_param"],
            "opt_J3_param": sec["opt_J3_param"],
            "opt_geminal": bool(sec["opt_geminal"]),
            "fix_geminal_rank": fix_rank_val,
            "opt_J3_basis_exp": sec["opt_J3_basis_exp"],
            "opt_J3_basis_coeff": sec["opt_J3_basis_coeff"],
            "opt_lambda_basis_exp": sec["opt_lambda_basis_exp"],
            "opt_lambda_basis_coeff": sec["opt_lambda_basis_coeff"],
            "opt_JDTN_u_param": sec["opt_JDTN_u_param"],
            "opt_JDTN_chi_param": sec["opt_JDTN_chi_param"],
            "opt_JDTN_f_param": sec["opt_JDTN_f_param"],
            "optimizer_kwargs": optimizer_kwargs,
        }
        use_device_optimizer = jax.default_backend() == "gpu"
        optimize_kwargs["use_device_optimizer"] = use_device_optimizer
        mcmc_kwargs = {
            "num_mcmc_per_measurement": sec["num_mcmc_per_measurement"],
            "Dt": sec["Dt"],
            "epsilon_AS": sec["epsilon_AS"],
            "mpc": mpc,
            "mpc_cutoff": mpc_cutoff,
        }
        log_summary()
        run_twist_average_optimize(
            hamiltonian_list=ham_list,
            weights=weights,
            mcmc_seed=mcmc_seed,
            num_walkers=number_of_walkers,
            world_comm=mpi_comm,
            mcmc_kwargs=mcmc_kwargs,
            optimize_kwargs=optimize_kwargs,
            checkpoint_dir=checkpoint_dir,
            resume=bool(restart),
            # Supplies the shared /shared template + k points + BZ weights for
            # the complete multi-twist hamiltonian_data_opt_step_N.h5 dumps.
            source_hamiltonian_path=hamiltonian_h5,
        )
        logger.info(
            "Twist-averaged optimization complete. Shared Jastrow parameters "
            "were synchronized across twists; determinant parameters were "
            "optimized in their twist-local sampler communicators."
        )
        return

    if job_type in ("lrdmc-bra", "lrdmc-tau"):
        # Both LRDMC variants share the GFMC twist-driver entry point;
        # only the dispatch kind + per-section parameter table differs.
        section = "lrdmc-bra" if job_type == "lrdmc-bra" else "lrdmc-tau"
        # [lrdmc-bra] num_mcmc_per_measurement has no default, so the merge
        # refuses a TOML without it (fresh run or twist resume).
        _merge_section_into_parameters(parameters, dict_toml, section)
        sec = parameters[section]
        gfmc_kind = "n" if job_type == "lrdmc-bra" else "tau"
        if sec["num_gfmc_warmup_steps"] < GFMC_MIN_WARMUP_STEPS:
            raise ValueError(f"num_gfmc_warmup_steps ({sec['num_gfmc_warmup_steps']}) must be >= {GFMC_MIN_WARMUP_STEPS}.")
        if sec["num_gfmc_bin_blocks"] < GFMC_MIN_BIN_BLOCKS:
            raise ValueError(f"num_gfmc_bin_blocks ({sec['num_gfmc_bin_blocks']}) must be >= {GFMC_MIN_BIN_BLOCKS}.")
        if sec["num_gfmc_collect_steps"] < GFMC_MIN_COLLECT_STEPS:
            raise ValueError(f"num_gfmc_collect_steps ({sec['num_gfmc_collect_steps']}) must be >= {GFMC_MIN_COLLECT_STEPS}.")
        if not restart and sec["num_mcmc_steps"] - sec["num_gfmc_warmup_steps"] < sec["num_gfmc_bin_blocks"]:
            raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) must be at least num_gfmc_bin_blocks.")
        gfmc_kwargs = {
            "alat": sec["alat"],
            "num_gfmc_collect_steps": sec["num_gfmc_collect_steps"],
            "use_swct": sec["use_swct"],
            "mpc": mpc,
            "mpc_cutoff": mpc_cutoff,
        }
        if job_type == "lrdmc-tau":
            gfmc_kwargs["tau"] = sec["tau"]
        else:
            gfmc_kwargs["num_mcmc_per_measurement"] = sec["num_mcmc_per_measurement"]
        if "non_local_move" in sec:
            gfmc_kwargs["non_local_move"] = sec["non_local_move"]
        if "epsilon_PW" in sec:
            gfmc_kwargs["epsilon_PW"] = sec["epsilon_PW"]
        if "E_scf" in sec:
            gfmc_kwargs["E_scf"] = sec["E_scf"]
        log_summary()
        result = run_twist_average_gfmc(
            hamiltonian_list=ham_list,
            weights=weights,
            gfmc_kind=gfmc_kind,
            mcmc_seed=mcmc_seed,
            num_walkers=number_of_walkers,
            num_mcmc_steps=sec["num_mcmc_steps"],
            num_mcmc_warmup_steps=sec["num_gfmc_warmup_steps"],
            num_mcmc_bin_blocks=sec["num_gfmc_bin_blocks"],
            max_time=max_time,
            world_comm=mpi_comm,
            compute_force=bool(sec.get("atomic_force", False)),
            compute_stress=bool(sec.get("compute_stress", False)),
            gfmc_kwargs=gfmc_kwargs,
            checkpoint_dir=checkpoint_dir,
            resume=bool(restart),
        )
        _log_twist_average_result(result, with_force=bool(sec.get("atomic_force", False)))
        return

    raise ValueError(
        f"Twist-averaged dispatch: unsupported job_type {job_type!r}. Expected one of {' / '.join(_SUPPORTED_JOB_TYPES)}."
    )


def _log_hamiltonian_summary(hamiltonian_data: Hamiltonian_data) -> None:
    """Print the Hamiltonian summary (tiled-orbital summary included) once per run.

    This is the only place the summary is printed, for every job type
    (mcmc, vmc, lrdmc-bra, lrdmc-tau). A single-twist run prints its
    Hamiltonian; a twist-averaged run prints twist 0, the twist of MPI rank 0,
    which is the rank that writes the log.
    """
    logger.info("=" * num_sep_line)
    logger.info("Printing out information in hamiltonian_data instance.")
    hamiltonian_data._logger_info()


def _log_precision_section() -> None:
    """Log the active precision configuration as a labeled section.

    Output format mirrors the ``hamiltonian_data`` info section: a title
    line, top ``=`` separator, the summary at INFO level, the per-zone
    detail at DEBUG level, and a bottom ``=`` separator.
    """
    logger.info("=" * num_sep_line)
    logger.info("Printing out precision information.")
    logger.info("=" * num_sep_line)
    logger.info("Precision: %s", precision_mode_label())
    logger.debug("Zone detail:")
    for line in precision_zone_detail().split("\n"):
        logger.debug(line)
    logger.info("=" * num_sep_line)
    logger.info("")


def _dump_restart_checkpoint(driver, restart_chk: str, driver_type: str, mpi_comm) -> None:
    """Save per-rank HDF5 files and merge them into the single checkpoint ``restart_chk``."""
    mpi_rank = mpi_comm.Get_rank()
    logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
    logger.info("")

    tmp_h5 = f"._restart_rank{mpi_rank}.h5"
    driver.save_to_hdf5(tmp_h5)

    mpi_comm.Barrier()

    if mpi_rank == 0:
        merge_rank_checkpoints(
            output_path=restart_chk,
            mpi_size=mpi_comm.Get_size(),
            driver_type=driver_type,
            hamiltonian_data=driver.hamiltonian_data,
        )

    mpi_comm.Barrier()


def _read_restart_driver_config(section: str, dict_toml: dict, restart_chk: str, mpi_comm) -> dict:
    """Read this rank's checkpoint driver_config and guard the frozen keys on every rank together.

    The continuation TOML must not set a frozen key to a value that conflicts
    with the checkpoint, or that the checkpoint lacks: the driver restores
    frozen keys from the checkpoint and would otherwise ignore the TOML. Each
    rank checks its own group, and a problem on any rank fails identically on
    every rank, naming the rank (#807).
    """
    from ._checkpoint import load_driver_config_from_checkpoint, raise_on_restart_inconsistency
    from ._restart_policy import assert_restart_frozen_consistency

    mpi_rank = mpi_comm.Get_rank()
    checkpoint_driver_config = load_driver_config_from_checkpoint(restart_chk, rank=mpi_rank)
    errors = []
    try:
        assert_restart_frozen_consistency(section, dict_toml, checkpoint_driver_config)
    except ValueError as exc:
        errors.append(("driver_config", str(exc)))
    raise_on_restart_inconsistency(mpi_comm, context=f"Restart from {restart_chk!r}", rank=mpi_rank, errors=errors)
    _log_restored_from_checkpoint(section, dict_toml, checkpoint_driver_config, mpi_comm.Get_size())
    return checkpoint_driver_config


def _has_enough_samples(driver, num_warmup_steps: int, num_bin_blocks: int, mpi_comm) -> bool:
    """Whether the stored history suffices for the binned estimators (same condition as ``get_E``).

    A run stopped early by ``max_time`` may not have enough samples yet. Its
    checkpoint is already written, so it can be continued with ``restart = true``.
    The decision gates the collective ``get_E``, so every rank takes it from the
    same counter: ranks that disagree fail together instead of leaving a peer
    blocked in ``get_E`` (#806).
    """
    from ._checkpoint import raise_on_restart_inconsistency

    raise_on_restart_inconsistency(
        mpi_comm,
        context="The final-output sample count",
        rank=mpi_comm.Get_rank(),
        agreed={"mcmc_counter": int(driver.mcmc_counter)},
    )
    if driver.mcmc_counter - num_warmup_steps >= num_bin_blocks:
        return True
    logger.warning(
        f"Not enough samples for the final output yet: {max(driver.mcmc_counter, 0)} measurements, "
        f"warmup steps = {num_warmup_steps}, bin blocks = {num_bin_blocks}. "
        "Continue from the checkpoint with restart = true."
    )
    logger.info("")
    return False


def _log_stress_section(
    sigma_mean: npt.ArrayLike,
    sigma_std: npt.ArrayLike,
    P_mean: npt.ArrayLike,
    P_std: npt.ArrayLike,
    P_hydro: tuple[float, float] | None,
) -> None:
    """Log the stress tensor (Ha/bohr^3) and the diagonal pressure (Ha/bohr^3).

    Format mirrors the ``Atomic Forces:`` table emitted alongside it
    (whitespace-separated, ``uncertainties`` short ``+2uS``) so the
    workflow-side parser uses the same ``parse_ufloat_short`` token
    pattern. The six symmetric stress components are printed; the
    diagonal pressure is the negation of the three normal stresses
    plus the hydrostatic mean for convenience.

    Args:
        sigma_mean: ``(3, 3)`` array, stress tensor mean (Ha/bohr^3).
        sigma_std:  ``(3, 3)`` array, stress tensor std (Ha/bohr^3).
        P_mean:     ``(3,)`` array, ``-diag(sigma_mean)``.
        P_std:      ``(3,)`` array, ``diag(sigma_std)``.
        P_hydro:    ``(mean, std)`` of the hydrostatic pressure computed by the
            driver's trace-first jackknife. REQUIRED for a correct uncertainty:
            the three normal stresses are correlated, so quadrature of their
            separate bars is wrong. ``None`` is accepted only for
            callers that have no replicates at all (e.g. re-formatting stored
            means), and then the hydrostatic row is printed without an error bar
            rather than with a silently wrong one.
    """
    sep = 16 * 3
    logger.info("  Stress Tensor (sigma):")
    logger.info("  " + "-" * sep)
    logger.info("  Component   Value (Ha/bohr^3)")
    logger.info("  " + "-" * sep)
    _components = (
        ("sigma_xx", 0, 0),
        ("sigma_yy", 1, 1),
        ("sigma_zz", 2, 2),
        ("sigma_xy", 0, 1),
        ("sigma_xz", 0, 2),
        ("sigma_yz", 1, 2),
    )
    for label, i, j in _components:
        val = f"{ufloat(float(sigma_mean[i, j]), float(sigma_std[i, j])):+2uS}"
        logger.info(f"  {label.ljust(10)}  {val}")
    logger.info("  " + "-" * sep)

    P_hydro_mean = float(P_mean[0] + P_mean[1] + P_mean[2]) / 3.0

    logger.info("  Pressure (P = -diag(sigma)):")
    logger.info("  " + "-" * sep)
    logger.info("  Component   Value (Ha/bohr^3)")
    logger.info("  " + "-" * sep)
    for label, k in (("P_x", 0), ("P_y", 1), ("P_z", 2)):
        val = f"{ufloat(float(P_mean[k]), float(P_std[k])):+2uS}"
        logger.info(f"  {label.ljust(10)}  {val}")
    if P_hydro is None:
        # Refuse loudly instead of emitting a bare value. A P_hydro row without
        # ``val(err)`` makes ``jqmc_workflow.forces_parser.parse_pressure_table``
        # return None, which drops the ENTIRE pressure record -- Px/Py/Pz
        # included -- while the run is still recorded as completed.
        raise ValueError(
            "P_hydro replicates are unavailable, so the hydrostatic pressure has no error bar. "
            "Every caller must pass the covariance-aware (mean, std) pair from get_P_hydrostatic "
            "(single-twist) or TwistAverageResult.P_hydro (BZ-averaged); emitting the value without "
            "its error bar would silently drop the whole pressure table downstream."
        )
    else:
        P_hydro_mean, P_hydro_std = float(P_hydro[0]), float(P_hydro[1])
        val = f"{ufloat(P_hydro_mean, P_hydro_std):+2uS}"
        logger.info(f"  {'P_hydro'.ljust(10)}  {val}")
    logger.info("  " + "-" * sep)


_CLI_USAGE = CLI_USAGE


def _cli() -> None:
    """Main function."""
    # The same check the light console entry (jqmc._cli_entry.main) runs before importing this module.
    toml_file = parse_cli_arguments(sys.argv[1:])
    with open(toml_file) as f:
        dict_toml = toml.load(f)
    removed_twist_controls = {"twist_average", "twist_index"}.intersection(dict_toml.get("control", {}))
    if removed_twist_controls:
        raise ValueError(
            f"Removed [control] key(s): {sorted(removed_twist_controls)}. "
            "Twist dispatch is determined by hamiltonian_h5: a combined /twists input "
            "always runs twist averaging, and an ordinary Hamiltonian runs a single k point."
        )

    # MPI related
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # set verbosity
    try:
        verbosity = dict_toml["control"]["verbosity"]
    except KeyError:
        verbosity = cli_parameters["control"]["verbosity"]

    # set logger level
    if verbosity == "devel":
        logger_level = "DEVEL"
    elif verbosity == "high":
        logger_level = "DEBUG"
    elif verbosity == "low":
        logger_level = "INFO"
    elif verbosity == "mpi-devel":
        logger_level = "MPI-DEVEL"
    elif verbosity == "mpi-high":
        logger_level = "MPI-DEBUG"
    elif verbosity == "mpi-low":
        logger_level = "MPI-INFO"
    else:
        raise ValueError(
            f"Invalid control.verbosity {verbosity!r}; expected one of low, high, devel, mpi-low, mpi-high, mpi-devel."
        )

    log = getLogger("jqmc")
    if log.hasHandlers():
        log.handlers.clear()

    if logger_level == "INFO":
        if mpi_rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("INFO")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "DEBUG":
        if mpi_rank == 0:
            log.setLevel("DEBUG")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("DEBUG")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "DEVEL":
        if mpi_rank == 0:
            log.setLevel("DEVEL")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("DEVEL")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
    elif logger_level == "MPI-INFO":
        log.setLevel("INFO")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("INFO")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    elif logger_level == "MPI-DEBUG":
        log.setLevel("DEBUG")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("DEBUG")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    elif logger_level == "MPI-DEVEL":
        log.setLevel("DEVEL")
        file_handler = FileHandler(f"log_jqmc_mpirank_{mpi_rank}.log", mode="w")
        file_handler.setLevel("DEVEL")
        formatter = Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    else:
        if mpi_rank == 0:
            log.setLevel("INFO")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("INFO")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)
        else:
            log.setLevel("ERROR")
            stream_handler = StreamHandler(sys.stdout)
            stream_handler.setLevel("ERROR")
            handler_format = Formatter("%(message)s")
            stream_handler.setFormatter(handler_format)
            log.addHandler(stream_handler)

    # print header
    _print_header()

    # jax-MPI related
    # Avoid JAX distributed init warnings/hangs due to proxy env vars.
    for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(_proxy_var, None)
    try:
        _initialize_jax_distributed()
        logger.info("JAX distributed initialization is successful.")
        logger.info("")
        jax_distributed_is_initialized = True
    except Exception as e:
        logger.info("Running on CPUs or single GPU. JAX distributed initialization is skipped.")
        logger.debug(f"Distributed initialization Exception: {e}")
        logger.info("")
        jax_distributed_is_initialized = False

    # Surface silently-failed distributed init: ``initialize`` may swallow
    # exceptions (the try/except above) but if we are actually under
    # ``mpirun -n N>=2`` and JAX still sees only 1 process, every downstream
    # device-collective call would silently produce per-rank-local results.
    _check_mpi4py_jax_distribution_consistency()

    if jax_distributed_is_initialized:
        # global JAX device
        global_device_info = jax.devices()
        # local JAX device
        num_devices = jax.local_devices()
        device_info_str = f"Rank {mpi_rank}: {num_devices}"
        local_device_info = mpi_comm.allgather(device_info_str)
        # print recognized XLA devices
        logger.info("*** XLA Global devices recognized by JAX***")
        logger.info(global_device_info)
        logger.info("*** XLA Local devices recognized by JAX***")
        logger.info(local_device_info)
        logger.info("")

    # JAX environment info
    logger.debug(_jax_environment_info())

    logger.info(f"Input file = {toml_file}")
    if not all([type(value) is dict for value in dict_toml.values()]):
        raise ValueError("The format of the toml file is wrong. See the tutorial.")

    logger.info("")
    logger.info("Input parameters are::")
    logger.info("")
    for section, dict_item in dict_toml.items():
        logger.info(f"**section:{section}**")
        for key, item in dict_item.items():
            logger.info(f"  {key}={item}")
        logger.info("")
    # Before any Hamiltonian or checkpoint file is read.
    _validate_toml_boolean_keys(dict_toml)

    # --- precision configuration ---
    precision_section = dict_toml.get("precision", {})
    if not isinstance(precision_section, dict):
        raise ValueError("The [precision] section must be a TOML table.")
    precision_mode = precision_section.get("mode", "full")
    extra_keys = set(precision_section.keys()) - {"mode"}
    if extra_keys:
        logger.warning(
            "Per-zone precision overrides are not supported and are ignored: %s. Only [precision] mode is read.",
            sorted(extra_keys),
        )
    configure_precision(precision_mode)
    logger.info(f"JAX backend = {jax.default_backend()}.")
    _backend = _execution_backend_label()
    logger.info(f"Execution backend = {_backend}.")
    if _backend == "XLA + Rust":
        logger.info(_rs_thread_line())
    logger.info("")

    # default parameters
    # Deep copy: the per-key override loops below assign into the nested
    # section dicts, which a shallow copy would share with the module-level
    # defaults (and with the defaults _merged_optimizer_kwargs reads).
    parameters = copy.deepcopy(cli_parameters)

    # control section
    section = "control"
    # On restart the frozen keys take the checkpoint value, logged once the
    # checkpoint is read (_log_restored_from_checkpoint), not the default.
    restart_requested = bool(dict_toml.get(section, {}).get("restart", parameters[section]["restart"]))
    for key in parameters[section].keys():
        try:
            parameters[section][key] = dict_toml[section][key]
        except KeyError:
            if parameters[section][key] is None:
                logger.error(f"{key} should be specified.")
                sys.exit(1)
            elif not (restart_requested and is_frozen(section, key)):
                logger.warning(f"The default value of {key} = {parameters[section][key]}.")

    job_type = parameters["control"]["job_type"]
    _validate_job_type(job_type)
    mcmc_seed = parameters["control"]["mcmc_seed"]
    number_of_walkers = parameters["control"]["number_of_walkers"]
    max_time = parameters["control"]["max_time"]
    restart = parameters["control"]["restart"]
    restart_chk = parameters["control"]["restart_chk"]
    hamiltonian_h5 = parameters["control"]["hamiltonian_h5"]
    if job_type == "lrdmc-tau" and "tau" in dict_toml.get("lrdmc-tau", {}):
        _validate_gfmc_tau(dict_toml["lrdmc-tau"]["tau"])
    if job_type in ("lrdmc-tau", "lrdmc-bra") and "alat" in dict_toml.get(job_type, {}):
        _validate_gfmc_alat(dict_toml[job_type]["alat"])
    twist_dispatch = _resolve_twist_dispatch(hamiltonian_h5, restart=restart, restart_chk=restart_chk)
    _warn_unknown_toml_keys(dict_toml, job_type)

    # --- [coulomb] section (optional; MPC finite-size correction) ---
    # On fresh runs, omitting it leaves the run pure-Ewald. ``mpc = true``
    # additionally accumulates <V_ee^MPC - V_ee^Ewald> as a passive observable
    # on the Ewald-propagated walkers, so the reported energy gains
    # E_MPC = E_Ewald + <dV> while sampling, branching, forces and parameter
    # gradients are untouched. ``mpc_cutoff`` (Ha) is the convergence knob.
    section = "coulomb"
    coulomb_section = dict_toml.get(section, {})
    if not isinstance(coulomb_section, dict):
        raise ValueError("The [coulomb] section must be a TOML table.")
    for key in parameters[section].keys():
        if key in coulomb_section:
            parameters[section][key] = coulomb_section[key]
    unknown_coulomb_keys = set(coulomb_section.keys()) - set(parameters[section].keys())
    if unknown_coulomb_keys:
        logger.warning(f"Unknown [coulomb] keys ignored: {sorted(unknown_coulomb_keys)}.")
    mpc = parameters["coulomb"]["mpc"]
    if mpc:
        # Reject a non-positive or non-finite cutoff before any driver is set up.
        _validate_mpc_cutoff(parameters["coulomb"]["mpc_cutoff"])
    mpc_cutoff = float(parameters["coulomb"]["mpc_cutoff"])
    _validate_mpc_job_type(job_type, mpc=bool(mpc))

    # ---- twist-averaged dispatch ----
    # A combined twist input always routes to the twist-averaged
    # driver (run_twist_average_from_hamiltonians /
    # run_twist_average_gfmc / run_twist_average_optimize) and short-
    # circuits the single-twist blocks below. Ordinary Hamiltonian files
    # always use the single-twist driver.
    if twist_dispatch == "twist_average":
        # A twist-averaged restart does not read these from a checkpoint.
        if restart:
            for key in parameters["control"]:
                if is_frozen("control", key) and key not in dict_toml.get("control", {}):
                    logger.warning(f"The default value of {key} = {parameters['control'][key]}.")
        _run_twist_average_dispatch(
            job_type=job_type,
            hamiltonian_h5=hamiltonian_h5,
            parameters=parameters,
            dict_toml=dict_toml,
            mcmc_seed=mcmc_seed,
            number_of_walkers=number_of_walkers,
            mpc=mpc,
            mpc_cutoff=mpc_cutoff,
            max_time=max_time,
            restart=restart,
            restart_chk=restart_chk,
            mpi_comm=mpi_comm,
            mpi_rank=mpi_rank,
        )
        _print_footer()
        return

    if restart and job_type in ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"):
        # Reject changed rank counts before even the frozen-config read tries
        # to access rank_N. Twist dispatch above uses its intra-comm in the
        # driver loaders, rather than this single-twist COMM_WORLD size.
        from ._checkpoint import _validate_restart_rank_topology

        _validate_restart_rank_topology(restart_chk, mpi_size)

    # MCMC
    if job_type == "mcmc":
        logger.info("***Variational Monte Carlo***")

        # vmc section
        section = "mcmc"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                elif not (restart and is_frozen(section, key)):
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        # parameters
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        # 'auto' is a jqmc-workflow-level concept. The driver is a pure executor
        # and must receive a concrete integer; the workflow resolves
        # 'auto' (= 4 * total electrons) before writing the TOML.
        if isinstance(num_mcmc_per_measurement, bool) or not isinstance(num_mcmc_per_measurement, int):
            raise ValueError(
                f"num_mcmc_per_measurement = {num_mcmc_per_measurement!r} is not a concrete integer. "
                "The jqmc input needs an explicit integer; 'auto' is accepted only by "
                "jqmc-workflow, which replaces it with an integer when it writes the input."
            )
        num_mcmc_warmup_steps = parameters[section]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters[section]["num_mcmc_bin_blocks"]
        Dt = parameters[section]["Dt"]
        epsilon_AS = parameters[section]["epsilon_AS"]
        atomic_force = parameters[section]["atomic_force"]
        compute_stress = parameters[section]["compute_stress"]
        use_swct = parameters[section]["use_swct"]
        parameter_derivatives = parameters[section]["parameter_derivatives"]

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if not restart:
            if num_mcmc_steps < num_mcmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
            if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            _ckpt_cfg = _read_restart_driver_config(section, dict_toml, restart_chk, mpi_comm)
            mcmc = MCMC.load_from_hdf5(restart_chk, rank=mpi_rank)
            atomic_force, compute_stress = _restart_force_stress_switches(
                section, dict_toml, atomic_force, compute_stress, mcmc
            )
            # Report the MPC settings that the driver loaded for the stored history.
            mpc = bool(_ckpt_cfg["mpc"])
            mpc_cutoff = float(_ckpt_cfg["mpc_cutoff"])
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            # Atomic force and periodic stress are independent observables.
            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                epsilon_AS=epsilon_AS,
                comput_position_deriv=bool(atomic_force),
                comput_stress=bool(compute_stress),
                comput_log_WF_param_deriv=parameter_derivatives,
                retain_derivative_sources=False,
                use_swct=use_swct,
                mpc=mpc,
                mpc_cutoff=mpc_cutoff,
            )
        _log_precision_section()
        _log_hamiltonian_summary(mcmc.hamiltonian_data)
        mcmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        # Write the checkpoint before the final analysis so that a run stopped
        # early by max_time is never lost.
        _dump_restart_checkpoint(mcmc, restart_chk, "MCMC", mpi_comm)
        if _has_enough_samples(mcmc, num_mcmc_warmup_steps, num_mcmc_bin_blocks, mpi_comm):
            E_mean, E_std, Var_mean, Var_std = mcmc.get_E(
                num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            )
            # MPC finite-size correction (passive observable; [coulomb] mpc = true).
            # Reported as the correction AND the corrected total, because the
            # correction is the number to converge against ``mpc_cutoff`` while
            # the corrected total is the number to quote. Sampling, forces and
            # parameter gradients are pure Ewald regardless -- MPC is energy-only.
            if mpc:
                dE_mpc_mean, dE_mpc_std, E_mpc_mean, E_mpc_std = _collect_mpc_estimates(
                    mcmc,
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                )
            if atomic_force:
                f_mean, f_std = mcmc.get_aF(
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                )
            if compute_stress:
                sigma_mean, sigma_std = mcmc.get_aT(
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                )
                P_mean, P_std = mcmc.get_aP(
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                )
                P_hydro = mcmc.get_P_hydrostatic(
                    num_mcmc_warmup_steps=num_mcmc_warmup_steps,
                    num_mcmc_bin_blocks=num_mcmc_bin_blocks,
                )
            logger.info("Final output(s):")
            logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:.5f} Ha.")
            logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:.5f} Ha^2.")
            if mpc:
                logger.info(f"  MPC correction: dE_MPC = {dE_mpc_mean:+.5f} +- {dE_mpc_std:.5f} Ha (cutoff = {mpc_cutoff} Ha).")
                logger.info(f"  Total Energy (MPC-corrected): E_MPC = {E_mpc_mean:.5f} +- {E_mpc_std:.5f} Ha.")
            if atomic_force:
                logger.info("  Atomic Forces:")
                sep = 16 * 3
                logger.info("  " + "-" * sep)
                logger.info("  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
                logger.info("  " + "-" * sep)
                for i in range(len(mcmc.hamiltonian_data.structure_data.atomic_labels)):
                    atomic_label = str(mcmc.hamiltonian_data.structure_data.atomic_labels[i])
                    row_values = [f"{ufloat(f_mean[i, j], f_std[i, j]):+2uS}" for j in range(3)]
                    row_str = "  " + atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
                    logger.info(row_str)
                logger.info("  " + "-" * sep)
            if compute_stress and mcmc.hamiltonian_data.structure_data.pbc_flag:
                _log_stress_section(sigma_mean, sigma_std, P_mean, P_std, P_hydro)
            if mcmc.compute_log_WF_param_deriv:
                logger.info("  Parameter Derivatives are computed.")
            logger.info("")

    # VMC
    if job_type == "vmc":
        logger.info("***WF optimization with Variational Monte Carlo***")

        # vmcopt section
        section = "vmc"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                elif not (restart and is_frozen(section, key)):
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        logger.info("")

        # parameters
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        # 'auto' is a jqmc-workflow-level concept. The driver is a pure executor
        # and must receive a concrete integer; the workflow resolves
        # 'auto' (= 4 * total electrons) before writing the TOML.
        if isinstance(num_mcmc_per_measurement, bool) or not isinstance(num_mcmc_per_measurement, int):
            raise ValueError(
                f"num_mcmc_per_measurement = {num_mcmc_per_measurement!r} is not a concrete integer. "
                "The jqmc input needs an explicit integer; 'auto' is accepted only by "
                "jqmc-workflow, which replaces it with an integer when it writes the input."
            )
        num_mcmc_warmup_steps = parameters[section]["num_mcmc_warmup_steps"]
        num_mcmc_bin_blocks = parameters[section]["num_mcmc_bin_blocks"]
        Dt = parameters[section]["Dt"]
        epsilon_AS = parameters[section]["epsilon_AS"]
        num_opt_steps = parameters[section]["num_opt_steps"]
        wf_dump_freq = parameters[section]["wf_dump_freq"]
        opt_J1_param = parameters[section]["opt_J1_param"]
        opt_J2_param = parameters[section]["opt_J2_param"]
        opt_J3_param = parameters[section]["opt_J3_param"]
        opt_geminal = bool(parameters[section]["opt_geminal"])
        # ``fix_geminal_rank=0`` is the 'not set' sentinel in the new
        # defaults; map it back to None at the run_optimize boundary so
        # the auto-convert is skipped.
        _fix_rank_raw = parameters[section]["fix_geminal_rank"]
        fix_geminal_rank = int(_fix_rank_raw) if int(_fix_rank_raw) > 0 else None
        opt_J3_basis_exp = parameters[section]["opt_J3_basis_exp"]
        opt_J3_basis_coeff = parameters[section]["opt_J3_basis_coeff"]
        opt_lambda_basis_exp = parameters[section]["opt_lambda_basis_exp"]
        opt_lambda_basis_coeff = parameters[section]["opt_lambda_basis_coeff"]
        opt_JDTN_u_param = parameters[section]["opt_JDTN_u_param"]
        opt_JDTN_chi_param = parameters[section]["opt_JDTN_chi_param"]
        opt_JDTN_f_param = parameters[section]["opt_JDTN_f_param"]
        optimizer_kwargs = _merged_optimizer_kwargs(parameters[section]["optimizer_kwargs"])

        vmc_section = dict_toml.get(section, {}) if isinstance(dict_toml.get(section, {}), dict) else {}
        optimizer_method = vmc_section.get("optimizer")
        if optimizer_method is not None and not isinstance(optimizer_method, str):
            raise TypeError("The 'optimizer' key must be a string when provided in the VMC section.")
        if optimizer_method is not None:
            logger.warning("The 'optimizer' key is deprecated. Please move the value under optimizer_kwargs.method instead.")
            optimizer_kwargs.setdefault("method", optimizer_method)

        optimizer_kwargs.setdefault("method", "sr")
        if not isinstance(optimizer_kwargs["method"], str):
            raise TypeError("optimizer_kwargs['method'] must be a string when provided in the VMC section.")
        if optimizer_kwargs["method"].lower() != "sr":
            raise ValueError(
                f"Unsupported optimizer_kwargs['method'] = '{optimizer_kwargs['method']}'. SR (natural "
                "gradient) / linear method is the only supported optimizer; set method = 'sr'."
            )
        optimizer_kwargs["lm_cond"] = validate_lm_condition_cutoff(optimizer_kwargs["lm_cond"])

        # check num_mcmc_steps, num_mcmc_warmup_steps, num_mcmc_bin_blocks
        if num_mcmc_steps < num_mcmc_warmup_steps:
            raise ValueError("num_mcmc_steps should be larger than num_mcmc_warmup_steps")
        if num_mcmc_steps - num_mcmc_warmup_steps < num_mcmc_bin_blocks:
            raise ValueError("(num_mcmc_steps - num_mcmc_warmup_steps) should be larger than num_mcmc_bin_blocks.")

        _need_eL_deriv = bool(optimizer_kwargs.get("use_lm", False))

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            _ckpt_cfg = _read_restart_driver_config(section, dict_toml, restart_chk, mpi_comm)
            mcmc = MCMC.load_from_hdf5(restart_chk, rank=mpi_rank, require_local_energy_derivatives=_need_eL_deriv)
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            mcmc = MCMC(
                hamiltonian_data=hamiltonian_data,
                Dt=Dt,
                mcmc_seed=mcmc_seed,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                epsilon_AS=epsilon_AS,
                comput_position_deriv=False,
                comput_log_WF_param_deriv=True,
                comput_e_L_param_deriv=_need_eL_deriv,
                mpc=mpc,
                mpc_cutoff=mpc_cutoff,
            )
        _log_precision_section()
        _log_hamiltonian_summary(mcmc.hamiltonian_data)
        # Pick the SR backend per JAX device: GPU favours the on-device
        # (jax.shard_map + NCCL) path, CPU favours the host mpi4py + SciPy
        # path. Anything else falls back to the CPU path.
        use_device_optimizer = jax.default_backend() == "gpu"
        mcmc.run_optimize(
            num_mcmc_steps=num_mcmc_steps,
            num_opt_steps=num_opt_steps,
            wf_dump_freq=wf_dump_freq,
            num_mcmc_warmup_steps=num_mcmc_warmup_steps,
            num_mcmc_bin_blocks=num_mcmc_bin_blocks,
            opt_J1_param=opt_J1_param,
            opt_J2_param=opt_J2_param,
            opt_J3_param=opt_J3_param,
            opt_geminal=opt_geminal,
            fix_geminal_rank=fix_geminal_rank,
            opt_J3_basis_exp=opt_J3_basis_exp,
            opt_J3_basis_coeff=opt_J3_basis_coeff,
            opt_lambda_basis_exp=opt_lambda_basis_exp,
            opt_lambda_basis_coeff=opt_lambda_basis_coeff,
            opt_JDTN_u_param=opt_JDTN_u_param,
            opt_JDTN_chi_param=opt_JDTN_chi_param,
            opt_JDTN_f_param=opt_JDTN_f_param,
            max_time=max_time,
            optimizer_kwargs=optimizer_kwargs,
            use_device_optimizer=use_device_optimizer,
        )
        logger.info("")

        logger.info(f"Dump restart checkpoint file(s) to {restart_chk}.")
        logger.info("")

        # Save per-rank HDF5 and merge into a single checkpoint.
        tmp_h5 = f"._restart_rank{mpi_rank}.h5"
        mcmc.save_to_hdf5(tmp_h5)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            merge_rank_checkpoints(
                output_path=restart_chk,
                mpi_size=mpi_size,
                driver_type="MCMC",
                hamiltonian_data=mcmc.hamiltonian_data,
            )

        mpi_comm.Barrier()

    # LRDMC!
    if job_type == "lrdmc-bra":
        logger.info("***Lattice Regularized diffusion Monte Carlo***")

        # lrdmc-bra section
        section = "lrdmc-bra"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    if restart and is_frozen(section, key):
                        # num_mcmc_per_measurement (required, no default): a
                        # restart takes the frozen value from the checkpoint.
                        continue
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                elif not (restart and is_frozen(section, key)):
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        logger.info("")

        # parameters
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        num_mcmc_per_measurement = parameters[section]["num_mcmc_per_measurement"]
        # 'auto' is a jqmc-workflow-level concept. The driver is a pure executor
        # and must receive a concrete integer; the workflow resolves
        # 'auto' (= 4 * total electrons) before writing the TOML. None only
        # reaches here on a restart that omits the key (checkpoint value kept).
        restored_from_checkpoint = restart and num_mcmc_per_measurement is None
        if not restored_from_checkpoint and (
            isinstance(num_mcmc_per_measurement, bool) or not isinstance(num_mcmc_per_measurement, int)
        ):
            raise ValueError(
                f"num_mcmc_per_measurement = {num_mcmc_per_measurement!r} is not a concrete integer. "
                "The jqmc input needs an explicit integer; 'auto' is accepted only by "
                "jqmc-workflow, which replaces it with an integer when it writes the input."
            )
        alat = parameters[section]["alat"]
        non_local_move = parameters[section]["non_local_move"]
        num_gfmc_warmup_steps = parameters[section]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = parameters[section]["num_gfmc_bin_blocks"]
        num_gfmc_collect_steps = parameters[section]["num_gfmc_collect_steps"]
        E_scf = parameters[section]["E_scf"]
        atomic_force = parameters[section]["atomic_force"]
        compute_stress = parameters[section]["compute_stress"]
        use_swct = parameters[section]["use_swct"]
        epsilon_PW = parameters[section]["epsilon_PW"]

        # Enforce GFMC minimum thresholds (E_scf unreliable before step 25)
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

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if not restart:
            if num_mcmc_steps < num_gfmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_gfmc_warmup_steps")
            if num_mcmc_steps - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            _ckpt_cfg = _read_restart_driver_config(section, dict_toml, restart_chk, mpi_comm)
            lrdmc = GFMC_n.load_from_hdf5(restart_chk, rank=mpi_rank)
            atomic_force, compute_stress = _restart_force_stress_switches(
                section, dict_toml, atomic_force, compute_stress, lrdmc
            )
            mpc = bool(_ckpt_cfg["mpc"])
            mpc_cutoff = float(_ckpt_cfg["mpc_cutoff"])
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            # Atomic force and periodic stress are independent observables.
            lrdmc = GFMC_n(
                hamiltonian_data=hamiltonian_data,
                mpc=mpc,
                mpc_cutoff=mpc_cutoff,
                num_walkers=number_of_walkers,
                num_mcmc_per_measurement=num_mcmc_per_measurement,
                num_gfmc_collect_steps=num_gfmc_collect_steps,
                mcmc_seed=mcmc_seed,
                E_scf=E_scf,
                alat=alat,
                non_local_move=non_local_move,
                comput_position_deriv=bool(atomic_force),
                comput_stress=bool(compute_stress),
                epsilon_PW=epsilon_PW,
                use_swct=use_swct,
            )
        _log_precision_section()
        _log_hamiltonian_summary(lrdmc.hamiltonian_data)
        lrdmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        # Write the checkpoint before the final analysis so that a run stopped
        # early by max_time is never lost.
        _dump_restart_checkpoint(lrdmc, restart_chk, "GFMC_n", mpi_comm)
        if _has_enough_samples(lrdmc, num_gfmc_warmup_steps, num_gfmc_bin_blocks, mpi_comm):
            E_mean, E_std, Var_mean, Var_std = lrdmc.get_E(
                num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                num_mcmc_bin_blocks=num_gfmc_bin_blocks,
            )
            # MPC finite-size correction (passive observable; [coulomb] mpc = true).
            # Reported as the correction AND the corrected total, because the
            # correction is the number to converge against ``mpc_cutoff`` while
            # the corrected total is the number to quote. Sampling, forces and
            # parameter gradients are pure Ewald regardless -- MPC is energy-only.
            if mpc:
                dE_mpc_mean, dE_mpc_std, E_mpc_mean, E_mpc_std = _collect_mpc_estimates(
                    lrdmc,
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            if atomic_force:
                f_mean, f_std = lrdmc.get_aF(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            if compute_stress:
                sigma_mean, sigma_std = lrdmc.get_aT(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
                P_mean, P_std = lrdmc.get_aP(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
                P_hydro = lrdmc.get_P_hydrostatic(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            logger.info("Final output(s):")
            logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:.5f} Ha.")
            logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:.5f} Ha^2.")
            if mpc:
                logger.info(f"  MPC correction: dE_MPC = {dE_mpc_mean:+.5f} +- {dE_mpc_std:.5f} Ha (cutoff = {mpc_cutoff} Ha).")
                logger.info(f"  Total Energy (MPC-corrected): E_MPC = {E_mpc_mean:.5f} +- {E_mpc_std:.5f} Ha.")
            if atomic_force:
                logger.info("  Atomic Forces:")
                sep = 16 * 3
                logger.info("  " + "-" * sep)
                logger.info("  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
                logger.info("  " + "-" * sep)
                for i in range(len(lrdmc.hamiltonian_data.structure_data.atomic_labels)):
                    atomic_label = str(lrdmc.hamiltonian_data.structure_data.atomic_labels[i])
                    row_values = [f"{ufloat(f_mean[i, j], f_std[i, j]):+2uS}" for j in range(3)]
                    row_str = "  " + atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
                    logger.info(row_str)
                logger.info("  " + "-" * sep)
            if compute_stress and lrdmc.hamiltonian_data.structure_data.pbc_flag:
                _log_stress_section(sigma_mean, sigma_std, P_mean, P_std, P_hydro)
            logger.info("")

    # LRDMC with fixed time!
    if job_type == "lrdmc-tau":
        logger.info("***Lattice Regularized diffusion Monte Carlo with a fixed projection time***")

        # lrdmc-tau section
        section = "lrdmc-tau"
        for key in parameters[section].keys():
            try:
                parameters[section][key] = dict_toml[section][key]
            except KeyError:
                if parameters[section][key] is None:
                    logger.error(f"{key} should be specified.")
                    sys.exit(1)
                elif not (restart and is_frozen(section, key)):
                    logger.warning(f"The default value of {key} = {parameters[section][key]}.")

        logger.info("")

        # parameters
        num_mcmc_steps = parameters[section]["num_mcmc_steps"]
        tau = parameters[section]["tau"]
        alat = parameters[section]["alat"]
        non_local_move = parameters[section]["non_local_move"]
        num_gfmc_warmup_steps = parameters[section]["num_gfmc_warmup_steps"]
        num_gfmc_bin_blocks = parameters[section]["num_gfmc_bin_blocks"]
        num_gfmc_collect_steps = parameters[section]["num_gfmc_collect_steps"]
        atomic_force = parameters[section]["atomic_force"]
        compute_stress = parameters[section]["compute_stress"]
        use_swct = parameters[section]["use_swct"]
        epsilon_PW = parameters[section]["epsilon_PW"]

        # Enforce GFMC minimum thresholds (same as lrdmc-bra and the twist route)
        if num_gfmc_warmup_steps < GFMC_MIN_WARMUP_STEPS:
            raise ValueError(f"num_gfmc_warmup_steps ({num_gfmc_warmup_steps}) must be >= {GFMC_MIN_WARMUP_STEPS}.")
        if num_gfmc_bin_blocks < GFMC_MIN_BIN_BLOCKS:
            raise ValueError(f"num_gfmc_bin_blocks ({num_gfmc_bin_blocks}) must be >= {GFMC_MIN_BIN_BLOCKS}.")
        if num_gfmc_collect_steps < GFMC_MIN_COLLECT_STEPS:
            raise ValueError(f"num_gfmc_collect_steps ({num_gfmc_collect_steps}) must be >= {GFMC_MIN_COLLECT_STEPS}.")

        # num_branching, num_gmfc_warmup_steps, num_gmfc_bin_blocks, num_gfmc_bin_collect
        if not restart:
            if num_mcmc_steps < num_gfmc_warmup_steps:
                raise ValueError("num_mcmc_steps should be larger than num_gfmc_warmup_steps")
            if num_mcmc_steps - num_gfmc_warmup_steps < num_gfmc_bin_blocks:
                raise ValueError("(num_mcmc_steps - num_gfmc_warmup_steps) should be larger than num_gfmc_bin_blocks.")

        if restart:
            logger.info(f"Read restart checkpoint file(s) from {restart_chk}.")
            _ckpt_cfg = _read_restart_driver_config(section, dict_toml, restart_chk, mpi_comm)
            lrdmc = GFMC_t.load_from_hdf5(restart_chk, rank=mpi_rank)
            atomic_force, compute_stress = _restart_force_stress_switches(
                section, dict_toml, atomic_force, compute_stress, lrdmc
            )
            mpc = bool(_ckpt_cfg["mpc"])
            mpc_cutoff = float(_ckpt_cfg["mpc_cutoff"])
        else:
            hamiltonian_data = Hamiltonian_data.load_from_hdf5(hamiltonian_h5)

            # Atomic force and periodic stress are independent observables.
            lrdmc = GFMC_t(
                hamiltonian_data=hamiltonian_data,
                mpc=mpc,
                mpc_cutoff=mpc_cutoff,
                num_walkers=number_of_walkers,
                tau=tau,
                num_gfmc_collect_steps=num_gfmc_collect_steps,
                mcmc_seed=mcmc_seed,
                alat=alat,
                non_local_move=non_local_move,
                comput_position_deriv=bool(atomic_force),
                comput_stress=bool(compute_stress),
                epsilon_PW=epsilon_PW,
                use_swct=use_swct,
            )
        _log_precision_section()
        _log_hamiltonian_summary(lrdmc.hamiltonian_data)
        lrdmc.run(num_mcmc_steps=num_mcmc_steps, max_time=max_time)
        # Write the checkpoint before the final analysis so that a run stopped
        # early by max_time is never lost.
        _dump_restart_checkpoint(lrdmc, restart_chk, "GFMC_t", mpi_comm)
        if _has_enough_samples(lrdmc, num_gfmc_warmup_steps, num_gfmc_bin_blocks, mpi_comm):
            E_mean, E_std, Var_mean, Var_std = lrdmc.get_E(
                num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                num_mcmc_bin_blocks=num_gfmc_bin_blocks,
            )
            # MPC finite-size correction (passive observable; [coulomb] mpc = true).
            # Reported as the correction AND the corrected total, because the
            # correction is the number to converge against ``mpc_cutoff`` while
            # the corrected total is the number to quote. Sampling, forces and
            # parameter gradients are pure Ewald regardless -- MPC is energy-only.
            if mpc:
                dE_mpc_mean, dE_mpc_std, E_mpc_mean, E_mpc_std = _collect_mpc_estimates(
                    lrdmc,
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            if atomic_force:
                f_mean, f_std = lrdmc.get_aF(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            if compute_stress:
                sigma_mean, sigma_std = lrdmc.get_aT(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
                P_mean, P_std = lrdmc.get_aP(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
                P_hydro = lrdmc.get_P_hydrostatic(
                    num_mcmc_warmup_steps=num_gfmc_warmup_steps,
                    num_mcmc_bin_blocks=num_gfmc_bin_blocks,
                )
            logger.info("Final output(s):")
            logger.info(f"  Total Energy: E = {E_mean:.5f} +- {E_std:.5f} Ha.")
            logger.info(f"  Variance: Var = {Var_mean:.5f} +- {Var_std:.5f} Ha^2.")
            if mpc:
                logger.info(f"  MPC correction: dE_MPC = {dE_mpc_mean:+.5f} +- {dE_mpc_std:.5f} Ha (cutoff = {mpc_cutoff} Ha).")
                logger.info(f"  Total Energy (MPC-corrected): E_MPC = {E_mpc_mean:.5f} +- {E_mpc_std:.5f} Ha.")
            if atomic_force:
                logger.info("  Atomic Forces:")
                sep = 16 * 3
                logger.info("  " + "-" * sep)
                logger.info("  Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)")
                logger.info("  " + "-" * sep)
                for i in range(len(lrdmc.hamiltonian_data.structure_data.atomic_labels)):
                    atomic_label = str(lrdmc.hamiltonian_data.structure_data.atomic_labels[i])
                    row_values = [f"{ufloat(f_mean[i, j], f_std[i, j]):+2uS}" for j in range(3)]
                    row_str = "  " + atomic_label.ljust(8) + " ".join(val.ljust(12) for val in row_values)
                    logger.info(row_str)
                logger.info("  " + "-" * sep)
            if compute_stress and lrdmc.hamiltonian_data.structure_data.pbc_flag:
                _log_stress_section(sigma_mean, sigma_std, P_mean, P_std, P_hydro)
            logger.info("")

    _print_footer()


def _finalize_distributed_runtimes() -> None:
    """Release process-global JAX and MPI runtimes owned by the CLI.

    JAX distributed shutdown must precede MPI finalization because its client
    may still use MPI-discovered peers while tearing down.  ``_cli`` itself is
    intentionally left free of this cleanup: several in-process tests call it
    directly, and finalizing MPI there would invalidate the entire pytest
    process.  The executable entry point :func:`main` is the ownership boundary.
    """
    try:
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()


def _abort_mpi_on_uncaught_exception(exc: BaseException) -> None:
    """Print a rank-prefixed traceback and abort the whole MPI job.

    Without this, a per-rank exception reaches ``main``'s ``finally`` and
    blocks inside ``jax.distributed.shutdown()`` at the coordination-service
    ``Shutdown`` barrier: the healthy ranks are stuck in MPI collectives
    waiting for the crashed peer, so the barrier can never complete. After
    the ~5 min barrier timeout the coordination client ``LOG(FATAL)``s every
    process, which destroys the pending Python tracebacks -- the job dies
    silent with no diagnostics. Aborting the world communicator immediately
    keeps the logs intact and releases the nodes right away.

    Deliberate no-op (normal exception propagation) when:

    * MPI is not initialized / already finalized, or the world size is 1 --
      single-rank runs keep the plain traceback (and pytest calls ``_cli``
      directly, never reaching ``main``);
    * the exception is a clean ``SystemExit`` (code ``0`` / ``None``).
    """
    if isinstance(exc, SystemExit) and (exc.code is None or exc.code == 0):
        return
    try:
        # ``COMM_WORLD`` is probed first: a stubbed/absent MPI module (the
        # entry-point teardown test injects one without it) drops us to the
        # silent no-op before any runtime-state query is recorded.
        world = MPI.COMM_WORLD
        if not MPI.Is_initialized() or MPI.Is_finalized():
            return
        if world.Get_size() < 2:
            return
        rank = world.Get_rank()
    except Exception:
        return
    import traceback

    print(
        f"[jqmc rank {rank}] FATAL: unhandled {type(exc).__name__}; aborting the whole MPI job.",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    world.Abort(1)


def main() -> None:
    """Run the CLI and always close its distributed process runtimes."""
    try:
        return _cli()
    except BaseException as exc:
        # Only returns (falling through to the re-raise + ``finally``) when
        # not running under MPI with >= 2 ranks; otherwise Abort()s the job.
        _abort_mpi_on_uncaught_exception(exc)
        raise
    finally:
        _finalize_distributed_runtimes()


if __name__ == "__main__":
    main()
