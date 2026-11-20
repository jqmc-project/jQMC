"""Deterministic output parser for jQMC stdout/stderr.

This module extracts **facts only** from jQMC log files using regular
expressions.  It contains no heuristic judgments or convergence assessments
-- those belong to a higher-level diagnostics layer.

For VMC it extracts the data of every optimization step, not only the
last value.

The text-level parsing (regular expressions, per-step VMC data, run
metadata and the final-output fields) is implemented in
:mod:`jqmc.output_parser`, which needs no work directory. This module adds
output-file discovery from ``workflow_state.toml`` and the result dataclasses.

Public API
----------
parse_vmc_output(work_dir)
    Parse VMC optimization stdout/stderr -> VMC_Diagnostic_Data.
parse_mcmc_output(work_dir)
    Parse MCMC sampling stdout/stderr -> MCMC_Diagnostic_Data.
parse_lrdmc_output(work_dir)
    Parse LRDMC stdout/stderr -> LRDMC_Diagnostic_Data.
parse_input_params(toml_path)
    Extract key parameters from a TOML input file -> Input_Parameters.
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

import glob
import os
import re
from logging import getLogger

import toml

from jqmc.output_parser import (
    _RE_EXTRAP_ENERGY,
    _RE_RESTART_CHECKPOINT,
    parse_lrdmc_log,
    parse_mcmc_log,
    parse_run_metadata,
    parse_vmc_log,
    parse_vmc_steps,
)

from ._results import (
    Input_Parameters,
    LRDMC_Diagnostic_Data,
    LRDMC_Ext_Diagnostic_Data,
    MCMC_Diagnostic_Data,
    VMC_Diagnostic_Data,
    VMC_Step_Data,
)

logger = getLogger("jqmc-workflow").getChild(__name__)


# -- Text-level parsing ---------------------------------------------
#
# The regular expressions and text parsers live in ``jqmc.output_parser`` so
# that drivers other than jqmc_workflow use the same implementation. This
# module adds work-directory discovery and the result dataclasses.

_STDERR_TAIL_LINES = 200


# -- Internal helpers ----------------------------------------------


def _read_text(path: str) -> str | None:
    """Read a text file, returning *None* if it doesn't exist or can't be read."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return None


def _tail(text: str, max_lines: int = _STDERR_TAIL_LINES) -> str:
    """Return the last *max_lines* lines of *text*."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[-max_lines:])


def _find_input_files(work_dir: str) -> list:
    """Return jQMC input TOML files in *work_dir*, sorted by step index.

    Reads ``workflow_state.toml`` ``[[jobs]]`` records and returns the
    ``input_file`` paths that exist on disk, ordered by ``step``.

    Only jobs in ``"fetched"`` or ``"completed"`` status are considered:
    cancelled / failed / still-submitted jobs may have stale inputs
    that would mislead downstream parsing (e.g. the wrong hamiltonian
    file reference).
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return []
    try:
        state = toml.load(state_path)
    except Exception:
        return []
    files = []
    for job in state.get("jobs", []):
        if job.get("status") not in ("fetched", "completed"):
            continue
        name = job.get("input_file", "")
        if name:
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                files.append((job.get("step", 0), path))
    files.sort()
    return [path for _, path in files]


def _find_hamiltonian_h5(work_dir: str) -> str | None:
    """Extract ``[control] hamiltonian_h5`` from input TOML files in *work_dir*.

    Uses ``workflow_state.toml`` ``[[jobs]]`` to locate input files.
    Returns the value from the last input that contains
    ``[control] hamiltonian_h5``, or *None* if not found.
    """
    for fpath in reversed(_find_input_files(work_dir)):
        try:
            data = toml.load(fpath)
            h5 = data.get("control", {}).get("hamiltonian_h5")
            if h5:
                return h5
        except Exception:
            continue
    return None


def _parse_run_metadata(text: str) -> dict:
    """Extract run-level metadata from a jQMC output file header.

    See :func:`jqmc.output_parser.parse_run_metadata`.
    """
    return parse_run_metadata(text)


def _apply_run_metadata(result, meta: dict) -> None:
    """Apply run metadata dict to a diagnostic data object."""
    result.num_mpi_processes = meta["num_mpi_processes"]
    result.num_walkers_per_process = meta["num_walkers_per_process"]
    result.jax_backend = meta["jax_backend"]
    result.jax_devices = meta["jax_devices"]


def _find_output_files(work_dir: str) -> list:
    """Return jQMC stdout files in *work_dir*, sorted by step index.

    Reads ``workflow_state.toml`` ``[[jobs]]`` records and returns the
    ``output_file`` paths that exist on disk, ordered by ``step``.

    Only jobs in ``"fetched"`` or ``"completed"`` status are considered:
    partial output left behind by cancelled / failed / still-submitted
    jobs would otherwise pollute parser aggregations (timing breakdowns,
    SNR series, force tables) with garbage data.
    """
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if not os.path.isfile(state_path):
        return []
    try:
        state = toml.load(state_path)
    except Exception:
        return []
    files = []
    for job in state.get("jobs", []):
        if job.get("status") not in ("fetched", "completed"):
            continue
        name = job.get("output_file", "")
        if name:
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                files.append((job.get("step", 0), path))
    files.sort()
    return [path for _, path in files]


def _parse_vmc_log_text(text: str) -> list:
    """Parse VMC log text into a list of :class:`VMC_Step_Data`.

    One entry per ``Optimization step`` block, parsed by
    :func:`jqmc.output_parser.parse_vmc_steps`. Lines before the first
    ``Optimization step`` header are ignored.

    Returns:
        list of VMC_Step_Data:
            One entry per optimization step found.
    """
    names = set(VMC_Step_Data.__dataclass_fields__)
    return [VMC_Step_Data(**{k: v for k, v in step.items() if k in names}) for step in parse_vmc_steps(text)]


# -- VMC parser ----------------------------------------------------


def parse_vmc_output(work_dir: str) -> VMC_Diagnostic_Data:
    """Parse VMC optimization output from *work_dir*.

    Discovers output files from ``workflow_state.toml`` ``[[jobs]]``
    records, parses per-step data, and looks for
    ``hamiltonian_data_opt_step_*.h5``.

    Args:
        work_dir (str):
            Path to the VMC working directory.

    Returns:
        VMC_Diagnostic_Data:
            Structured parse result containing per-step data and metadata.
    """
    result = VMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_vmc_output: directory not found: %s", work_dir)
        return result

    # -- Discover and parse stdout files --
    output_files = _find_output_files(work_dir)
    all_steps: list[VMC_Step_Data] = []

    for fpath in output_files:
        text = _read_text(fpath)
        if text is None:
            continue
        file_steps = _parse_vmc_log_text(text)
        all_steps.extend(file_steps)

    result.steps = all_steps

    # -- Run-level metadata (MPI, walkers, JAX) from first output file --
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Extract total_opt_steps from the last output file (the M in N/M)
    if output_files:
        last_text = _read_text(output_files[-1])
        if last_text:
            last = parse_vmc_log(last_text)
            if last["total_opt_steps"] is not None:
                result.total_opt_steps = last["total_opt_steps"]
            # -- Optimization-level timing (appears once after all steps) --
            if last["total_opt_time_sec"] is not None:
                result.total_opt_time_sec = last["total_opt_time_sec"]
            result.opt_timing_breakdown.update(last["opt_timing_breakdown"])

    # -- optimized hamiltonian --
    # Find hamiltonian_data_opt_step_*.h5 and sort numerically by step.
    h5_pattern = os.path.join(work_dir, "hamiltonian_data_opt_step_*.h5")
    h5_files = glob.glob(h5_pattern)
    if h5_files:

        def _h5_step_num(path: str) -> int:
            m = re.search(r"hamiltonian_data_opt_step_(\d+)\.h5$", path)
            return int(m.group(1)) if m else -1

        h5_files.sort(key=_h5_step_num)
        result.optimized_hamiltonian = h5_files[-1]
    else:
        logger.warning(
            "parse_vmc_output: no hamiltonian_data_opt_step_*.h5 found in %s; optimized_hamiltonian will be None.",
            work_dir,
        )

    # -- restart checkpoint --
    # Search all output files for the last "Dump restart checkpoint" line.
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            for m_rc in _RE_RESTART_CHECKPOINT.finditer(text):
                result.restart_checkpoint = m_rc.group(1)
            if result.restart_checkpoint is not None:
                break

    # -- stderr tail --
    stderr_candidates = [
        os.path.join(work_dir, "stderr"),
        os.path.join(work_dir, "err"),
    ]
    # Also check output files for stderr content
    for fpath in stderr_candidates:
        text = _read_text(fpath)
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# -- MCMC parser ---------------------------------------------------


def parse_mcmc_output(work_dir: str) -> MCMC_Diagnostic_Data:
    """Parse MCMC sampling output from *work_dir*.

    Extracts acceptance ratio, walker weights, and net time from stdout.
    Energy/error are extracted from the ``workflow_state.toml`` result
    section (populated by jqmc-tool post-processing) or from stdout
    if ``jqmc-tool mcmc compute-energy`` output is present.  The MPC
    finite-size correction (``dE_mpc`` / ``dE_mpc_error``) comes from
    the same section and stays *None* for a run without MPC.

    Args:
        work_dir (str):
            Path to the MCMC working directory.

    Returns:
        MCMC_Diagnostic_Data:
            Structured parse result.
    """
    result = MCMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_mcmc_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    # -- Run-level metadata (MPI, walkers, JAX) from first output file --
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Parse the last output file (most recent run)
    last_text = None
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            last_text = text
            break

    if last_text:
        # Diagnostics -- the last occurrence of each line
        last = parse_mcmc_log(last_text)
        for name in (
            "avg_walker_weight",
            "acceptance_ratio",
            "total_time_sec",
            "precompilation_time_sec",
            "net_time_sec",
            "restart_checkpoint",
        ):
            if last[name] is not None:
                setattr(result, name, last[name])
        result.timing_breakdown.update(last["timing_breakdown"])

    # -- hamiltonian_data_file from input TOML --
    result.hamiltonian_data_file = _find_hamiltonian_h5(work_dir)

    # -- Energy & forces from workflow_state.toml result section --
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "energy" in res:
                result.energy = float(res["energy"])
            if "energy_error" in res:
                result.energy_error = float(res["energy_error"])
            # MPC correction: absent for a run without MPC (the workflow
            # never writes the keys in that case).
            if "dE_mpc" in res:
                result.dE_mpc = float(res["dE_mpc"])
            if "dE_mpc_error" in res:
                result.dE_mpc_error = float(res["dE_mpc_error"])
            if "forces" in res and isinstance(res["forces"], list):
                result.atomic_forces = res["forces"]
            if "stress" in res and isinstance(res["stress"], dict):
                result.stress_tensor = res["stress"]
            if "pressure" in res and isinstance(res["pressure"], dict):
                result.pressure = res["pressure"]
        except Exception:
            pass

    # -- stderr tail --
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# -- LRDMC parser --------------------------------------------------


def parse_lrdmc_output(work_dir: str) -> LRDMC_Diagnostic_Data:
    """Parse LRDMC calculation output from *work_dir*.

    Extracts survived walkers ratio, average number of projections,
    and net GFMC time from stdout.  Energy/error come from
    ``workflow_state.toml`` result section, as does the MPC finite-size
    correction (``dE_mpc`` / ``dE_mpc_error``), which stays *None* for a
    run without MPC.

    Args:
        work_dir (str):
            Path to the LRDMC working directory.

    Returns:
        LRDMC_Diagnostic_Data:
            Structured parse result.
    """
    result = LRDMC_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_lrdmc_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    # -- Run-level metadata (MPI, walkers, JAX) from first output file --
    if output_files:
        first_text = _read_text(output_files[0])
        if first_text:
            _apply_run_metadata(result, _parse_run_metadata(first_text))

    # Parse the last output file
    last_text = None
    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text:
            last_text = text
            break

    if last_text:
        # Diagnostics -- the last occurrence of each line
        last = parse_lrdmc_log(last_text)
        for name in (
            "survived_walkers_ratio",
            "avg_num_projections",
            "total_time_sec",
            "precompilation_time_sec",
            "net_time_sec",
            "restart_checkpoint",
        ):
            if last[name] is not None:
                setattr(result, name, last[name])
        result.timing_breakdown.update(last["timing_breakdown"])

    # -- hamiltonian_data_file from input TOML --
    result.hamiltonian_data_file = _find_hamiltonian_h5(work_dir)

    # -- Energy & forces from workflow_state.toml result section --
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "energy" in res:
                result.energy = float(res["energy"])
            if "energy_error" in res:
                result.energy_error = float(res["energy_error"])
            # MPC correction: absent for a run without MPC (the workflow
            # never writes the keys in that case).
            if "dE_mpc" in res:
                result.dE_mpc = float(res["dE_mpc"])
            if "dE_mpc_error" in res:
                result.dE_mpc_error = float(res["dE_mpc_error"])
            if "forces" in res and isinstance(res["forces"], list):
                result.atomic_forces = res["forces"]
            if "stress" in res and isinstance(res["stress"], dict):
                result.stress_tensor = res["stress"]
            if "pressure" in res and isinstance(res["pressure"], dict):
                result.pressure = res["pressure"]
        except Exception:
            pass

    # -- stderr tail --
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# -- LRDMC extrapolation parser ------------------------------------


def parse_lrdmc_ext_output(work_dir: str) -> LRDMC_Ext_Diagnostic_Data:
    """Parse LRDMC extrapolation output from *work_dir*.

    Looks for ``For a -> 0 bohr: E = ...`` in the stdout of the
    extrapolation step.

    Args:
        work_dir (str):
            Path to the LRDMC extrapolation working directory.

    Returns:
        LRDMC_Ext_Diagnostic_Data:
            Structured parse result.
    """
    result = LRDMC_Ext_Diagnostic_Data()

    if not os.path.isdir(work_dir):
        logger.warning("parse_lrdmc_ext_output: directory not found: %s", work_dir)
        return result

    output_files = _find_output_files(work_dir)

    for fpath in reversed(output_files):
        text = _read_text(fpath)
        if text is None:
            continue
        m = _RE_EXTRAP_ENERGY.search(text)
        if m:
            result.extrapolated_energy = float(m.group(1))
            result.extrapolated_energy_error = float(m.group(2))
            break

    # -- per-alat results from workflow_state.toml --
    state_path = os.path.join(work_dir, "workflow_state.toml")
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            res = state.get("result", {})
            if "per_alat_results" in res:
                result.per_alat_results = res["per_alat_results"]
        except Exception:
            pass

    # -- stderr tail --
    for name in ("stderr", "err"):
        text = _read_text(os.path.join(work_dir, name))
        if text:
            result.stderr_tail = _tail(text)
            break

    return result


# -- Input parameters parser --------------------------------------


def _get_cli_defaults() -> dict:
    """Return a **deep copy** of ``cli_parameters`` from ``jqmc.jqmc_miscs``.

    Falls back to an empty dict if jqmc is not installed.
    """
    try:
        import copy

        from jqmc.jqmc_miscs import cli_parameters

        return copy.deepcopy(cli_parameters)
    except Exception:
        return {}


def parse_input_params(work_dir: str) -> Input_Parameters:
    """Extract key parameters from a workflow directory, **per input file**.

    For each ``[[jobs]]`` entry in ``workflow_state.toml``, the
    corresponding TOML input file is loaded and merged with the
    default values defined in ``jqmc.jqmc_miscs.cli_parameters``.
    The result is a list of per-input dicts stored in
    ``Input_Parameters.per_input``.

    ``actual_opt_steps`` is read from ``restart.h5`` when available
    (VMC only).

    Args:
        work_dir (str):
            Path to the workflow working directory.

    Returns:
        Input_Parameters:
            Structured parameter data with per-input detail.
    """
    result = Input_Parameters()

    if not os.path.isdir(work_dir):
        logger.warning("parse_input_params: directory not found: %s", work_dir)
        return result

    # -- 1) restart.h5 -> actual_opt_steps (VMC only) --
    restart_path = os.path.join(work_dir, "restart.h5")
    if os.path.isfile(restart_path):
        try:
            import h5py

            with h5py.File(restart_path, "r") as f:
                if "rank_0/driver_config" in f:
                    dc_attrs = dict(f["rank_0/driver_config"].attrs)
                    i_opt = dc_attrs.get("i_opt")
                    if i_opt is not None:
                        result.actual_opt_steps = int(i_opt)
        except Exception as exc:
            logger.warning(
                "parse_input_params: cannot read restart.h5 in %s: %s",
                work_dir,
                exc,
            )

    # -- 2) workflow_state.toml -> [[jobs]] --
    state_path = os.path.join(work_dir, "workflow_state.toml")
    jobs: list = []
    if os.path.isfile(state_path):
        try:
            state = toml.load(state_path)
            jobs = state.get("jobs", [])
        except Exception:
            pass

    # -- 3) Per-input parameter extraction --
    defaults = _get_cli_defaults()

    for job_rec in jobs:
        input_name = job_rec.get("input_file", "")
        output_name = job_rec.get("output_file", "")
        entry: dict = {
            "input_file": input_name,
            "output_file": output_name,
        }

        toml_path = os.path.join(work_dir, input_name) if input_name else ""
        if toml_path and os.path.isfile(toml_path):
            try:
                raw = toml.load(toml_path)
            except Exception:
                raw = {}
        else:
            raw = {}

        # -- [control] section (merge defaults) --
        ctrl_defaults = {k: v for k, v in defaults.get("control", {}).items()}
        ctrl_user = raw.get("control", {})
        ctrl_merged = {**ctrl_defaults, **ctrl_user}
        entry["control"] = ctrl_merged

        job_type = ctrl_merged.get("job_type", "")
        entry["job_type"] = job_type if job_type else ""

        # -- job-type section (merge defaults) --
        if job_type and job_type in defaults:
            jt_defaults = {k: v for k, v in defaults[job_type].items()}
            jt_user = raw.get(job_type, {})
            jt_merged = {**jt_defaults, **jt_user}
            entry[job_type] = jt_merged
        elif job_type:
            # No defaults known -- just use raw values
            entry[job_type] = raw.get(job_type, {})

        result.per_input.append(entry)

    return result
