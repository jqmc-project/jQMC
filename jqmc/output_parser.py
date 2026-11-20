"""Deterministic parser for the text printed by ``jqmc`` and ``jqmc-tool``.

This module extracts **facts only** from jQMC stdout/stderr using regular
expressions. It contains no heuristic judgments or convergence assessments,
and it never touches the file system: every function takes the log *text*
and returns plain, JSON-serialisable Python objects. Any driver can use it:
``jqmc_workflow`` (which adds work-directory discovery and its result
dataclasses on top), AiiDA plugins, notebooks or agents.

Conventions:
    * A key is present with value ``None`` when its line was not found.
    * Ratios printed in percent (acceptance, survived walkers) are returned
      as fractions in ``[0, 1]``.
    * ``*_sec`` values are seconds; ``timing_breakdown`` values are msec,
      as printed.
    * Counters (``mcmc_counter``, ``gfmc_counter``, ``opt_step_counter``)
      are the cumulative values printed by jqmc, including restarted runs.

Public API:
    parse_ufloat_short, parse_force_table, parse_stress_table, parse_pressure_table
        ``val(err)`` tokens and the force / stress / pressure tables.
    parse_run_metadata
        MPI, walker and JAX information from the log header.
    parse_vmc_steps
        One dict per VMC optimization step.
    parse_mcmc_log, parse_vmc_log, parse_lrdmc_log, parse_jqmc_log
        stdout of ``jqmc`` for ``job_type`` mcmc / vmc / lrdmc-bra, lrdmc-tau.
    parse_tool_energy, parse_tool_force, parse_lrdmc_extrapolation
        stdout of ``jqmc-tool {mcmc,lrdmc} compute-energy / compute-force``
        and ``jqmc-tool lrdmc extrapolate-energy``.
    parse_last_exception
        The last ``SomeError: message`` line of a traceback or a
        ``jqmc-tool`` ``Error: ...`` message.
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
import math
import re

__all__ = [
    "parse_environment",
    "parse_force_table",
    "parse_input_parameters",
    "parse_jqmc_log",
    "parse_last_exception",
    "parse_lrdmc_extrapolation",
    "parse_lrdmc_log",
    "parse_mcmc_log",
    "parse_pressure_table",
    "parse_run_metadata",
    "parse_stress_table",
    "parse_system_info",
    "parse_tool_energy",
    "parse_tool_force",
    "parse_twist_average_result",
    "parse_ufloat_short",
    "parse_vmc_log",
    "parse_vmc_steps",
    "unparsed_numeric_lines",
]

# -- Compiled regex patterns ---------------------------------------
#
# All patterns are compiled once at module level for efficiency.

# "Optimization step =   1/10" or "Optimization step = 1/10."
_RE_OPT_STEP = re.compile(r"Optimization\s+step\s*=\s*(\d+)\s*/\s*(\d+)")

# Numeric pattern that also matches ``nan`` / ``inf`` (any case).
# Required so a diverged QMC run surfaces as a non-finite float rather
# than being silently dropped from per-step parsing.
_NUM = r"[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)"

# "E = -76.438901 +- 0.000123 Ha" (energy line).  Anchor the pattern at
# the start of the line so text such as ``Max generalized force = ...`` cannot
# be misidentified as an energy merely because "force" ends in ``e``.
_RE_ENERGY = re.compile(
    rf"^\s*E\s*=\s*({_NUM})\s*\+\-\s*({_NUM})",
    re.IGNORECASE,
)

# Current jQMC output spelling. Legacy force labels are intentionally rejected.
_RE_MAX_FORCE = re.compile(
    rf"^\s*Max\s+generalized\s+force\s*=\s*({_NUM})\s*\+\-\s*({_NUM})",
    re.IGNORECASE,
)

# "Max generalized force signal-to-noise = 126.871."
_RE_SNR = re.compile(
    rf"^\s*Max\s+generalized\s+force\s+signal-to-noise\s*=\s*({_NUM})",
    re.IGNORECASE,
)

# "Average of walker weights is 0.799. Ideal is ~ 0.800. Adjust epsilon_AS."
_RE_WALKER_WEIGHT = re.compile(r"Average of walker weights is\s+(\d+(?:\.\d+)?)")

# "Acceptance ratio is 51.46 %.  Ideal is ~ 50.00%. Adjust Dt."
_RE_ACCEPTANCE = re.compile(r"Acceptance ratio is\s+(\d+(?:\.\d+)?)\s*%")

# "Net total time for MCMC = 9.72 sec."
_RE_NET_MCMC = re.compile(r"Net total time for MCMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# "Total elapsed time for MCMC 1000 steps. = 21.07 sec."
_RE_TOTAL_MCMC = re.compile(
    r"Total elapsed time for MCMC\s+\d+\s+steps\.?\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Pre-compilation time for MCMC = 11.35 sec."
_RE_PRECOMP_MCMC = re.compile(r"Pre-compilation time for MCMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# Per-MCMC-step timing breakdown (msec)
_RE_TIME_MCMC_UPDATE = re.compile(r"Time for MCMC update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_E_L = re.compile(r"Time for computing e_L\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DE_L_DR = re.compile(r"Time for computing de_L/dR and de_L/dr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DLN_PSI_DR = re.compile(r"Time for computing dln_Psi/dR and dln_Psi/dr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DLN_PSI_DC = re.compile(r"Time for computing dln_Psi/dc\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_DE_L_DC = re.compile(r"Time for computing de_L/dc\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_STRESS = re.compile(r"Time for computing de_L/deps and dln_Psi/deps\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MPI_MCMC = re.compile(r"Time for MPI barrier after MCMC update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MISC_MCMC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")

# VMC optimization-level timing (sec, appears once after all opt steps)
# "Total elapsed time for optimization 10 steps. = 123.45 sec."
_RE_TOTAL_OPT = re.compile(
    r"Total elapsed time for optimization\s+\d+\s+steps\.?\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)
# Per-opt-step breakdown (sec)
_RE_OPT_MCMC_RUN = re.compile(r"Time for MCMC run\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_GET_E = re.compile(r"Time for computing E\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_GET_GF = re.compile(r"Time for computing generalized forces \(gF\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
# The label names the method that ran: "SR", "aSR" or "LM" (issue #857).
_RE_OPT_OPTIMIZER = re.compile(r"Time for optimizer \((?:SR|aSR|LM)\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_PARAM_UPDATE = re.compile(r"Time for parameter update\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_MPI_BARRIER = re.compile(r"Time for MPI barrier\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")
_RE_OPT_MISC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# "Net GFMC time without pre-compilations = 2832.326 sec."
_RE_NET_GFMC = re.compile(
    r"Net GFMC time without pre-compilations\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Total GFMC time for 1000 branching steps = 3000.0 sec."
_RE_TOTAL_GFMC = re.compile(
    r"Total GFMC time for\s+\d+\s+branching steps\s*=\s*"
    r"([\d.]+(?:[eE][+-]?\d+)?)\s*sec"
)

# "Pre-compilation time for GFMC = 167.674 sec."
_RE_PRECOMP_GFMC = re.compile(r"Pre-compilation time for GFMC\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# Per-branching timing breakdown (msec) -- GFMC
_RE_TIME_PROJECTION = re.compile(
    r"Projection(?:\s+time per branching|\s+between branching)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec"
)
_RE_TIME_OBSERVABLE = re.compile(r"Time for Observable measurement.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MPI_GFMC = re.compile(r"Time for MPI barrier before branching\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_COLLECTION = re.compile(r"Time for walker observable collections.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_RECONFIGURATION = re.compile(r"Time for walker reconfiguration.*?=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_UPDATE_ESCF = re.compile(r"Time for updating E_scf\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")
_RE_TIME_MISC_GFMC = re.compile(r"Time for misc\.\s*\(others\)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*msec")

# "Survived walkers ratio = 95.2 %"
_RE_SURVIVED = re.compile(r"Survived walkers ratio\s*=\s*(\d+\.?\d*)\s*%")

# "Average of the number of projections  = 123"
_RE_AVG_PROJECTIONS = re.compile(r"Average of the number of projections\s*=\s*(\d+(?:\.\d+)?)")

# "For a -> 0 bohr: E = -76.43 +- 0.01 Ha."  (LRDMC extrapolation)
_RE_EXTRAP_ENERGY = re.compile(
    r"For a -> 0 bohr:\s*E\s*=\s*"
    r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"\s*\+\-\s*"
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
)

# "Dump restart checkpoint file(s) to restart.h5."
_RE_RESTART_CHECKPOINT = re.compile(r"Dump restart checkpoint file\(s\) to\s+(\S+)\.\s*$", re.MULTILINE)

# -- Run-level metadata (header section, appears once per output file) --

# "The number of MPI process = 4."
_RE_MPI_PROCESSES = re.compile(r"The number of MPI process\s*=\s*(\d+)")

# "The number of walkers assigned for each MPI process = 4."
_RE_WALKERS_PER_PROCESS = re.compile(r"The number of walkers assigned for each MPI process\s*=\s*(\d+)")

# "JAX backend = gpu."
_RE_JAX_BACKEND = re.compile(r"JAX backend\s*=\s*(\w+)")

# Fallback: "Running on CPUs or single GPU. JAX distributed initialization is skipped."
_RE_JAX_CPU_FALLBACK = re.compile(r"Running on CPUs or single GPU")

# "*** XLA Global devices recognized by JAX***"
# followed by e.g. "[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]"
_RE_XLA_GLOBAL_HEADER = re.compile(r"\*{3}\s*XLA Global devices recognized by JAX\s*\*{3}")
_RE_XLA_DEVICE_LIST = re.compile(r"\[([^\]]+)\]")

# -- Run outcome and final output of ``jqmc`` -------------------------

# "Program ends 2000-01-01 00:00:00"
_RE_PROGRAM_ENDS = re.compile(r"^Program ends", re.MULTILINE)
# "  Stopping... max_time = 600 sec. exceeds." (GFMC prints "Max_time")
_RE_STOP_MAX_TIME = re.compile(r"Stopping\.\.\. max_time = (\d+(?:\.\d+)?) sec\. exceeds\.", re.IGNORECASE)
# Twist-averaged MCMC / LRDMC: a twist that max_time stopped before its requested steps.
_RE_TWIST_STOPPED = re.compile(r"Twist (\d+) stopped on max_time after (\d+) of (\d+) (measurements|branching steps)")
# "  Stopping... stop_flag in external_control_mcmc.toml is true."
_RE_STOP_FLAG = re.compile(r"Stopping\.\.\. stop_flag in \S+ is true\.")
# "num_mcmc_warmup_steps should be larger than 30"
_RE_WARNING_MIN = re.compile(r"^\s*(\S+ should be larger than .*?)\s*$", re.MULTILINE)
# "  Total Energy: E = -1.16785 +- 0.01522 Ha."
_RE_FINAL_ENERGY = re.compile(rf"^\s*Total Energy: E = ({_NUM}) \+- ({_NUM}) Ha\.", re.MULTILINE | re.IGNORECASE)
# "  Variance: Var = 0.06916 +- 0.00904 Ha^2."
_RE_FINAL_VARIANCE = re.compile(rf"^\s*Variance: Var = ({_NUM}) \+- ({_NUM}) Ha\^2\.", re.MULTILINE | re.IGNORECASE)
# "  MPC correction: dE_MPC = +0.00123 +- 0.00045 Ha (cutoff = 10.0 Ha)."
_RE_FINAL_MPC_CORRECTION = re.compile(rf"^\s*MPC correction: dE_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE | re.IGNORECASE)
# "  Total Energy (MPC-corrected): E_MPC = -1.16662 +- 0.01522 Ha."
_RE_FINAL_MPC_ENERGY = re.compile(
    rf"^\s*Total Energy \(MPC-corrected\): E_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE | re.IGNORECASE
)
# "  Progress: MCMC step = 61/1060: 5.8 %. Elapsed time = 0.5 sec."
_RE_MCMC_PROGRESS = re.compile(r"Progress: MCMC step = (\d+)/(\d+)")
# lrdmc-bra: "  Progress: GFMC step = 100/100: ..."; lrdmc-tau: "  branching step = 100/100: ..."
_RE_GFMC_PROGRESS = re.compile(r"(?:Progress: GFMC step|^\s*branching step) = (\d+)/(\d+)", re.MULTILINE)
# "Number of variational parameters = 3."
_RE_OPT_NUM_PARAMS = re.compile(r"Number of variational parameters = (\d+)\.")
# "Number of real optimization coordinates = 2.": the dimension of the SR/LM system (stored values tied or complex
# counted twice, jqmc_optimizer.variational_parameter_counts)
_RE_OPT_NUM_COORDINATES = re.compile(r"Number of real optimization coordinates = (\d+)\.")
# Twist-averaged VMC: the same two counts over every optimized variable, the shared coordinates once plus each twist's
# own local ones. "Number of real optimization coordinates (all twists) = 9."
_RE_OPT_NUM_PARAMS_ALL_TWISTS = re.compile(r"Number of variational parameters \(all twists\) = (\d+)\.")
_RE_OPT_NUM_COORDINATES_ALL_TWISTS = re.compile(r"Number of real optimization coordinates \(all twists\) = (\d+)\.")
# Twist-averaged VMC, one line per twist: "  Twist 3 local max generalized force signal-to-noise = 12.345."
_RE_TWIST_LOCAL_SNR = re.compile(
    rf"^\s*Twist\s+(\d+)\s+local\s+max\s+generalized\s+force\s+signal-to-noise\s*=\s*({_NUM})",
    re.IGNORECASE,
)
# "Hamiltonian data is dumped as an HDF5 file: hamiltonian_data_opt_step_1.h5."
# Twist-averaged VMC: "Twist-averaged Hamiltonian data (8 twists) is dumped as an HDF5 file: ...".
_RE_OPT_DUMP = re.compile(r"Hamiltonian data (?:\(\d+ twists?\) )?is dumped as an HDF5 file: (\S+?\.h5)\.?\s*$", re.MULTILINE)

# -- jqmc-tool ----------------------------------------------------------

# "E = -1.1735583058119603 +- 0.018008460477648664 Ha."
_RE_TOOL_ENERGY = re.compile(rf"^E = ({_NUM}) \+- ({_NUM}) Ha\.", re.MULTILINE | re.IGNORECASE)
# "Found 2 MPI ranks." / "Found 2 MPI ranks in lrdmc.h5."
_RE_TOOL_NUM_RANKS = re.compile(r"Found (\d+) MPI ranks")
# "Total number of binned samples = 80"
_RE_TOOL_NUM_SAMPLES = re.compile(r"Total number of binned samples = (\d+)")
# "  For a = 0.3 bohr: E = -1.18 +- 0.01 Ha." (also ``E_MPC`` with MPC)
_RE_EXTRAP_PER_ALAT = re.compile(rf"For a = ({_NUM}) bohr: (E|E_MPC) = ({_NUM}) \+- ({_NUM}) Ha", re.IGNORECASE)
# "  For a -> 0 bohr: E_MPC = -1.20 +- 0.04 Ha."
_RE_EXTRAP_MPC_ENERGY = re.compile(rf"For a -> 0 bohr:\s*E_MPC\s*=\s*({_NUM})\s*\+\-\s*({_NUM})", re.IGNORECASE)
# "  Polynomial order = 2."
_RE_EXTRAP_ORDER = re.compile(r"Polynomial order = (\d+)\.")

# MPC results of ``jqmc-tool {mcmc,lrdmc} compute-energy`` (no leading spaces).
_RE_TOOL_MPC_CORRECTION = re.compile(rf"^MPC correction: dE_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE | re.IGNORECASE)
_RE_TOOL_MPC_ENERGY = re.compile(
    rf"^Total Energy \(MPC-corrected\): E_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE | re.IGNORECASE
)
# "Found 8 twist checkpoints."
_RE_TOOL_NUM_TWISTS = re.compile(r"Found (\d+) twist checkpoints")

# -- Twist-averaged final output of ``jqmc`` ----------------------------

# "  BZ-averaged Energy: E_TA = -1.16785 +- 0.01522 Ha (s_twist = 0.00123 Ha)."
_RE_TA_ENERGY = re.compile(rf"^\s*BZ-averaged Energy: E_TA = ({_NUM}) \+- ({_NUM}) Ha \(s_twist = ({_NUM}) Ha\)", re.MULTILINE)
# "  BZ-averaged MPC correction: dE_MPC = +0.00123 +- 0.00045 Ha (cutoff = 10.0 Ha)."
_RE_TA_MPC_CORRECTION = re.compile(rf"^\s*BZ-averaged MPC correction: dE_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE)
# "  BZ-averaged Total Energy (MPC-corrected): E_MPC = -1.16662 +- 0.01522 Ha."
_RE_TA_MPC_ENERGY = re.compile(
    rf"^\s*BZ-averaged Total Energy \(MPC-corrected\): E_MPC = ({_NUM}) \+- ({_NUM}) Ha", re.MULTILINE
)
# Per-twist table row: "      0  Gamma                   -1.1678(15)" or "      1  (+0.250,+0.000,+0.000)  ..."
_RE_TA_TWIST_ROW = re.compile(r"^\s+(\d+)\s+(Gamma|\([^)]*\))\s+(\S+)\s*$", re.MULTILINE)
_RE_TA_TABLE_HEADER = re.compile(r"^\s*Per-twist energy \(Level-1 outputs\):", re.MULTILINE)

# "Twist-averaged dispatch: 8 twists detected in hamiltonian_data.h5 (BZ sum w = 2.000000)."
_RE_TWIST_DISPATCH = re.compile(
    rf"^Twist-averaged dispatch: (\d+) twists? detected in (\S+) \(BZ sum w = ({_NUM})\)\.", re.MULTILINE
)

# -- MPC table setup ------------------------------------------------------

# "MPC setup: building the tables on rank 0 and broadcasting them to 15 ranks."
_RE_MPC_SETUP_RANKS = re.compile(
    r"^MPC setup: building the tables on rank 0 and broadcasting them to (\d+) ranks\.", re.MULTILINE
)
# "MPC setup: cutoff = 30.0 Ha, grid = (64, 64, 64), n_max = 3; g_G is extrapolated from the N, 2N and 4N grids."
_RE_MPC_SETUP_GRID = re.compile(
    rf"^MPC setup: cutoff = ({_NUM}) Ha, grid = \(([\d, ]+)\), n_max = (\d+)(?:; (.*?))?\s*$", re.MULTILINE
)
# "  MPC setup: g_G on the 1N grid (64, 64, 64) done in 1.1 sec."
_RE_MPC_SETUP_STEP = re.compile(rf"^\s+MPC setup: (.+?) done in ({_NUM}) sec\.", re.MULTILINE)
# "MPC setup: done in 14.1 sec."
_RE_MPC_SETUP_DONE = re.compile(rf"^MPC setup: done in ({_NUM}) sec\.", re.MULTILINE)

# -- Header: environment, precision, input -----------------------------

_RE_PROGRAM_STARTS = re.compile(r"^Program starts (.+?)\s*$", re.MULTILINE)
_RE_PROGRAM_ENDS_AT = re.compile(r"^Program ends (.+?)\s*$", re.MULTILINE)
_RE_VERSION = re.compile(r"^\s*version = (\S+)\s*$", re.MULTILINE)
_RE_GIT_COMMIT = re.compile(r"^\s*git commit = (\S+)\s*$", re.MULTILINE)
_RE_LIB_VERSION = re.compile(r"^(jax|jaxlib|numpy|python):\s+(.+?)\s*$", re.MULTILINE)
_RE_DEVICE_INFO = re.compile(r'^device info: (.+?)"?\s*$', re.MULTILINE)
_RE_PROCESS_COUNT = re.compile(r"^process_count: (\d+)\s*$", re.MULTILINE)
_RE_PLATFORM = re.compile(r"^platform: (.+?)\s*$", re.MULTILINE)
_RE_XLA_FLAGS = re.compile(r"^XLA_FLAGS=(.*?)\s*$", re.MULTILINE)
# Every ``JAX_*``/``XLA_*`` environment variable that ``jax.print_environment_info``
# appends to the header (``verbosity = "high"``), e.g. ``JAX_COMPILATION_CACHE_DIR=...``.
_RE_JAX_ENV_VAR = re.compile(r"^((?:JAX|XLA)_\w+)=(.*?)\s*$", re.MULTILINE)
_RE_EXECUTION_BACKEND = re.compile(r"^Execution backend = (.+?)\.\s*$", re.MULTILINE)
_RE_RUST_THREADS = re.compile(r"^Rust rayon threads = (\d+)", re.MULTILINE)
_RE_PRECISION = re.compile(r"^Precision: (.+?)\s*$", re.MULTILINE)
_RE_PRECISION_ZONE = re.compile(r"^  (\w+): (float\d+|bfloat16)\s*$", re.MULTILINE)
# The device (JAX allocator) peak follows the RSS peak on GPU/TPU runs; CPU runs print the RSS field only.
_RE_MEMORY = re.compile(rf"^\[memory\] stage=(\S+), RSS_peak_max=({_NUM}) GB(?:, JAX_peak_max=({_NUM}) GB)?\s*$", re.MULTILINE)
_RE_INPUT_FILE = re.compile(r"^Input file = (.+?)\s*$", re.MULTILINE)
_RE_INPUT_SECTION = re.compile(r"^\*\*section:(\S+?)\*\*\s*$")
_RE_INPUT_ITEM = re.compile(r"^  (\w+)=(.*?)\s*$")
_RE_DEFAULT_VALUE = re.compile(r"^The default value of (\w+) = (.*?)\.?\s*$", re.MULTILINE)
_RE_DEFAULT_ALAT_SCALED = re.compile(r"^The default value of (num_mcmc_per_measurement) = (\d+) \(=", re.MULTILINE)
_RE_RESTORED_VALUE = re.compile(r"^The value of (\w+) = (.*?) \(restored from the checkpoint\)\.\s*$", re.MULTILINE)
_RE_RESTORED_NONSCALAR = re.compile(r"^\[(\S+)\] (\w+) is restored from the checkpoint\.\s*$", re.MULTILINE)

# -- Header: system (Hamiltonian summary) --------------------------------

_RE_PBC_FLAG = re.compile(r"^\s*PBC flag = (True|False)\s*$", re.MULTILINE)
_RE_ATOM_ROW = re.compile(rf"^\s+(\w+), (\w+), ({_NUM}), ({_NUM}), ({_NUM}), ({_NUM})\s*$", re.MULTILINE)
_RE_ECP_FLAG = re.compile(r"^\s*ecp_flag = (True|False)\s*$", re.MULTILINE)
_RE_OCCUPIED_ROWS = re.compile(r"^\s*occupied MO rows \(up, dn\) = \((\d+), (\d+)\)", re.MULTILINE)
_RE_NUM_ELECTRON = re.compile(r"^\s*num_electron_(up|dn) = (\d+)\s*$", re.MULTILINE)
_RE_TIME_REVERSAL = re.compile(r"^\s*time_reversal = (True|False)\s*$", re.MULTILINE)
_RE_NUM_MOS = re.compile(r"^\s*Number of MOs = (\d+)\s*$", re.MULTILINE)
_RE_NUM_AOS = re.compile(r"^\s*Number of AOs = (\d+)\s*$", re.MULTILINE)
_RE_NUM_PRIMITIVE_AOS = re.compile(r"^\s*Number of primitive AOs = (\d+)\s*$", re.MULTILINE)
_RE_MO_COEFF_DIM = re.compile(r"^\s*dim\. of MOs coeff = \((\d+), (\d+)\)", re.MULTILINE)
_RE_J2_PARAM = re.compile(rf"^\s*Jastrow 2b param = ({_NUM})\s*$", re.MULTILINE)
_RE_J2_TYPE = re.compile(r"^\s*2b Jastrow functional form is the (\w+) type\.", re.MULTILINE)
_RE_J3_DIM = re.compile(r"^\s*dim\. of jastrow_3b_matrix = \((\d+), (\d+)\)", re.MULTILINE)
_RE_MO_ORTHONORMALITY = re.compile(
    rf"^MO orthonormality \((\S+) spin-(up|dn)\): max\|M_ii - 1\| = ({_NUM}), "
    rf"max_\(i!=j\) \|M_ij\|/sqrt\(M_ii M_jj\) = ({_NUM})\.",
    re.MULTILINE,
)
_RE_LATTICE_VECTOR = re.compile(r"^\s+vec ([ABC]) = \[([^\]]*)\] Bohr", re.MULTILINE)
_RE_EWALD = re.compile(rf"^\s+(eta|r_real_cut|g_max|N_real_images|N_g|ii_energy) = ({_NUM})(?: \S+)?\s*$", re.MULTILINE)
# Periodic GTOs: the lattice images summed by the AO expansion.
_RE_AO_IMAGES = {
    "num_image_cells": re.compile(r"^\s+Number of lattice image cells = (\d+)\s*$", re.MULTILINE),
    "max_image_index": re.compile(r"^\s+max \|n_a\|, \|n_b\|, \|n_c\| over image indices = (\d+)\s*$", re.MULTILINE),
    "truncation_tolerance": re.compile(rf"^\s+image truncation tolerance = ({_NUM})\s*$", re.MULTILINE),
}
_RE_BSPLINE = {
    "grid": re.compile(r"^\s+spline grid shape \(N0, N1, N2\) = \(([\d, ]+)\)", re.MULTILINE),
    "order": re.compile(r"^\s+spline order \(degree\) = (\d+)", re.MULTILINE),
    "control_points_per_ao": re.compile(r"^\s+control points per AO = (\d+)", re.MULTILINE),
    "total_coefficients": re.compile(r"^\s+total stored coefficients \(num_ao \* cp_per_ao\) = (\d+)", re.MULTILINE),
    "complex_path": re.compile(r"^\s+complex \(k != 0\) path = (True|False)", re.MULTILINE),
}
# Tiled orbitals: the "**TiledOrbitalSet_data (frozen)" summary of each spin channel.
_RE_TILED_HEADER = re.compile(r"^\*\*TiledOrbitalSet_data \(frozen\)\s*$")
_RE_TILED = {
    # The supertwist is printed as plain floats, or as np.float64(...) reprs by older versions.
    "spin": re.compile(r"^  spin=(\w+), supertwist=\(((?:[^()]|np\.float64\([^()]*\))*)\), tiling=(\(.*\))\s*$"),
    "k_blocks": re.compile(r"^  primitive k blocks=(\d+), occupied orbitals=(\d+)\s*$"),
    "basis_sizes": re.compile(r"^  local basis sizes=\(([\d, ]*)\)\s*$"),
    "offsets": re.compile(r"^  orbital offsets=\(([\d, ]*)\), normalization=(\S+)\s*$"),
}
_RE_DTN_TERMS = re.compile(r"^\s+terms: (.*?); (.*?); C=(\d+)\s*$", re.MULTILINE)
_RE_DTN_ORDERS = re.compile(r"^\s+orders: (.*?)\s*$", re.MULTILINE)
_RE_DTN_CUTOFFS = re.compile(r"^\s+cutoffs: (.*?)\s*$", re.MULTILINE)
_RE_DTN_RATIOS = re.compile(r"^\s+cutoff ratios: (.*?)(?: \((.*?)\))?\s*$", re.MULTILINE)
_RE_DTN_COEFS = re.compile(r"^\s+indep coefs: (.*?)\s*$", re.MULTILINE)
_RE_J1_PARAM = re.compile(rf"^\s*Jastrow 1b param = ({_NUM})\s*$", re.MULTILINE)
_RE_J3_SYMMETRIC = re.compile(r"^\s*j3 part of the jastrow_3b_matrix is symmetric\? = (True|False)", re.MULTILINE)
_RE_LAMBDA = re.compile(
    r"^\s*(?:dim\. of (lambda_matrix) = |(lambda_matrix_paired|lambda_matrix_unpaired)\.shape = )\(([\d, ]+)\)", re.MULTILINE
)
_RE_LAMBDA_SYMMETRIC = re.compile(r"^\s*lambda_matrix is symmetric\? = (True|False)", re.MULTILINE)
_RE_INITIAL_SHAPE = re.compile(r"^\s*initial r_(up|dn)_carts\.shape = \(([\d, ]+)\)", re.MULTILINE)

# -- VMC optimizer -------------------------------------------------------

_RE_OPTIMIZER_NAME = re.compile(r"^The chosen optimizer is '(\w+)'\.", re.MULTILINE)
_RE_OPTIMIZER_MODE = re.compile(r"^\s*(use_lm=.*?)\s*$", re.MULTILINE)
_RE_OPTIMIZER_HYPER = re.compile(r"^\s*Hyperparameters: (.*?)\s*$", re.MULTILINE)
_RE_OPT_SAMPLING_STEPS = re.compile(r"^Sampling steps this iteration = (\d+)\.")
_RE_OPT_WARMUP = re.compile(r"^Warmup steps = (\d+)\.")
_RE_OPT_BIN_BLOCKS = re.compile(r"^Bin blocks = (\d+)\.")
_RE_OPT_BATCH = re.compile(
    r"^Optimization batch: route=(\S+), local samples=(\d+), sampler ranks=(\d+), "
    r"real coordinates=(\d+), storage parameters=(\d+)\."
)
_RE_OPT_DERIVATIVE_SOURCE = re.compile(
    r"^\s*Derivative source (\S+): O shape=\(([\d, ]*)\), D shape=(?:\(([\d, ]*)\)|None), policy=(\S+), tangent=(\S+)\."
)
_RE_ASR_GAMMAS = re.compile(rf"^aSR: gamma\+ = ({_NUM}), gamma- = ({_NUM})\.")
_RE_ASR_SELECTED = re.compile(rf"^aSR: selected gamma([+-]) = ({_NUM}) \((.*?)\)\.")
_RE_ASR_NOTE = re.compile(r"^aSR: (selected root .*?)\.?\s*$")
_RE_ASR_FINAL = re.compile(rf"^aSR: final gamma = ({_NUM})\.")
# aSR fallbacks and clipping (jqmc_optimizer_kernel): each line becomes one
# ``asr["fallbacks"]`` record, or ``asr["clipped_discriminant"]``.
_RE_ASR_FALLBACKS = {
    "root_not_finite_or_zero": re.compile(
        rf"^aSR: selected root is not finite/nonzero \(gamma=({_NUM})\); falling back to plain SR \(gamma=({_NUM})\)\."
    ),
    "root_exceeds_gamma_max": re.compile(
        rf"^aSR: selected reference-connected root gamma=({_NUM}) exceeds \|gamma_max\|=({_NUM}); "
        rf"falling back to plain SR with the model's sign \(gamma=({_NUM})\)\."
    ),
    "negative_centered_metric": re.compile(
        rf"^aSR: centered metric contraction S_2=({_NUM}) is negative; falling back to plain SR \(gamma=({_NUM})\)\."
    ),
    "degenerate_denominator": re.compile(
        rf"^aSR: denom = -2\*H_1\*S_2 ~ 0 \(H_1=({_NUM}), S_2=({_NUM})\); no meaningful optimization direction\. "
        rf"Returning gamma = ({_NUM})\."
    ),
    "invalid_model_denominator": re.compile(
        rf"^aSR: selected gamma produced an invalid model denominator; using gamma = ({_NUM})\."
    ),
}
# The numbers of each fallback line, in the order printed.
_ASR_FALLBACK_FIELDS = {
    "root_not_finite_or_zero": ("root_gamma", "gamma"),
    "root_exceeds_gamma_max": ("root_gamma", "gamma_max", "gamma"),
    "negative_centered_metric": ("S_2", "gamma"),
    "degenerate_denominator": ("H_1", "S_2", "gamma"),
    "invalid_model_denominator": ("gamma",),
}
_RE_ASR_DISCRIMINANT = re.compile(rf"^aSR: discriminant is negative \(({_NUM})\); setting it to zero\.")
_RE_DIAG_COORDINATES = re.compile(
    r"^\s*Coordinates: real=(\d+), storage=(\d+), active=(\d+), frozen=(\d+) in (\d+) atomic group\(s\)\."
)
# Twist-averaged VMC: the shared coordinates once plus every twist's local ones.
_RE_DIAG_COORDINATES_ALL_TWISTS = re.compile(
    r"^\s*Coordinates \(all twists\): real=(\d+), storage=(\d+), active=(\d+), frozen=(\d+) in (\d+) atomic group\(s\)\."
)
_RE_DIAG_METRIC = re.compile(
    rf"^\s*SR metric diagonal: all\[min=({_NUM}), max=({_NUM})\], active\[min=({_NUM}), max=({_NUM})\]\."
)
_RE_DIAG_SOLVE = re.compile(
    rf"^\s*SR solve: regime=(\S+), method=(\S+), iterations=(\d+), residual=({_NUM}), "
    rf"tolerance=({_NUM}), converged=(True|False)\."
)
_RE_DIAG_DIRECTION = re.compile(rf"^\s*SR direction: \|\|q_SR\|\|=({_NUM}), max\|q_SR\|=({_NUM})\.")
_RE_DIAG_TOP_HEADER = re.compile(r"^\s*Top (\d+) coordinates by generalized-force signal-to-noise:")
_RE_DIAG_TOP_ROW = re.compile(rf"^\s*(\S+): force=({_NUM}) \+- ({_NUM}), signal-to-noise=({_NUM}), metric_diag=({_NUM})\.")
_RE_DIAG_DECISION = re.compile(rf"^\s*aSR decision: scale=({_NUM}), model_fallback=(True|False)\.")
# Linear method (lm_subspace_dim -1 or > 0): conditioning of the tangent
# coordinates, the selected eigenstate, the subspace and the decision.
_RE_LM_CONDITIONING_INPUT = re.compile(
    rf"^\s*LM conditioning input: coordinates=(\d+), cutoff=({_NUM}), diag\[min=({_NUM}), max=({_NUM})\]\."
)
_RE_LM_CONDITIONING = re.compile(
    rf"^\s*LM conditioning: active=(\d+)/(\d+), tiny-diagonal removed=(\d+), correlation removed=(\d+), "
    rf"min correlation eigenvalue of the active set=({_NUM}) \(cutoff=({_NUM})\)\."
)
_RE_LM_SELECTED_STATE = re.compile(
    rf"^\s*LM selected state: reference weight=({_NUM}), \|\|q\|\|=({_NUM}), max\|q\|=({_NUM})\."
)
_RE_LM_SUBSPACE = re.compile(r"^\s*LM subspace: SR collective \+ (\d+) coordinate\(s\) in (\d+) atomic group\(s\)\.")
_RE_LM_ACCEPTED = re.compile(rf"^\s*LM decision: accepted \(reference weight=({_NUM})\)\.")
_RE_LM_REJECTED = re.compile(rf"^\s*LM decision: rejected; fallback scale=({_NUM}) \((.*)\)\.\s*$")
_RE_LM_REJECTED_WEIGHT = re.compile(rf"^\s*Rejected LM candidate reference weight=({_NUM})\.")
_RE_DIAG_LNPSI = re.compile(rf"^\s*Change in linearized ln\|Psi\| norm = ({_NUM})\.")
_RE_DIAG_FINAL_UPDATE = re.compile(rf"^\s*Final real-coordinate update: \|\|delta_q\|\|=({_NUM}), max\|delta_q\|=({_NUM})\.")
_RE_DIAG_BLOCK = re.compile(
    rf"^\s*\[optimizer block\] (\S+): \|\|force\|\|=({_NUM}), max\|force\|=({_NUM}), max\(S/N\)=({_NUM}), "
    rf"metric_diag\[min=({_NUM}), max=({_NUM})\]\."
)
_RE_DIAG_BLOCK_UPDATE = re.compile(
    rf"^\s*update: \|\|delta_q\|\|=({_NUM}), max\|delta_q\|=({_NUM}), \|\|delta_p\|\|=({_NUM}), max\|delta_p\|=({_NUM})\."
)
# MO blocks: the update's S-Stiefel tangent, printed after the block's update line.
_RE_DIAG_BLOCK_STIEFEL = re.compile(
    rf"^\s*S-Stiefel tangent: \|\|delta_C S\^\(1/2\)\|\|_F=({_NUM}), \|\|delta_C S\^\(1/2\)\|\|_2=({_NUM})\."
)
_RE_SR_PATH = re.compile(r"^SR solver path: regime=(\S+), method=(\S+), backend=(\S+)\s*$")
_RE_THETA_SIZE = re.compile(r"^theta\.size = (\d+)\.")
_RE_THETA_NONZERO = re.compile(r"^np\.count_nonzero\(theta\) = (\d+)\.")
_RE_THETA_RANGE = re.compile(rf"^max\. and min\. of \|theta\| are ({_NUM}) and ({_NUM})\.")
# Per-block update lines of every optimizer step; the label is the step's
# method (SR, aSR or LM), and an all-zero block prints the "no update" form.
_RE_PARAMETER_UPDATE = re.compile(
    rf"^\s*\[(\w+) update\] - block=(\S+) size=(\d+)\s+\|\|delta_p\|\|=({_NUM})\s+max\|delta_p\|=({_NUM})"
    rf"\s+max\|applied update\|=({_NUM})"
)
_RE_PARAMETER_NO_UPDATE = re.compile(r"^\s*\[(\w+) update\] - block=(\S+) size=(\d+)\s+theta=ALL ZERO \(no update\)")
_RE_OPT_UPDATING = re.compile(r"^Updating parameters with optimizer '(\w+)'\.")
_RE_OPT_SAMPLING_TIME = re.compile(r"Time for sampling\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*sec")

# -- LRDMC ---------------------------------------------------------------

# "    E_scf = -1.17111 +- 0.00348 Ha." (adaptive energy shift, printed during the run)
_RE_E_SCF = re.compile(rf"^\s+E_scf = ({_NUM}) \+- ({_NUM}) Ha\.\s*$", re.MULTILINE)

# -- stderr -------------------------------------------------------------

# Last line of a traceback ("ValueError: message", or a bare "ValueError" for
# ``raise ValueError`` without a message) or a jqmc-tool "Error: message".
# jqmc-tool reports bad options as click's ``BadParameter`` / ``UsageError``.
_RE_EXCEPTION = re.compile(r"^((?:\w+\.)*\w*(?:Error|Exception|BadParameter))(?:: (.*?))?\s*$", re.MULTILINE)

_MCMC_BREAKDOWN_PATTERNS = (
    (_RE_TIME_MCMC_UPDATE, "mcmc_update"),
    (_RE_TIME_E_L, "e_L"),
    (_RE_TIME_DE_L_DR, "de_L_dR_dr"),
    (_RE_TIME_DLN_PSI_DR, "dln_Psi_dR_dr"),
    (_RE_TIME_DLN_PSI_DC, "dln_Psi_dc"),
    (_RE_TIME_DE_L_DC, "de_L_dc"),
    (_RE_TIME_STRESS, "stress"),
    (_RE_TIME_MPI_MCMC, "mpi_barrier"),
    (_RE_TIME_MISC_MCMC, "misc"),
)

_OPT_BREAKDOWN_PATTERNS = (
    (_RE_OPT_SAMPLING_TIME, "sampling"),
    (_RE_OPT_MCMC_RUN, "mcmc_run"),
    (_RE_OPT_GET_E, "get_E"),
    (_RE_OPT_GET_GF, "get_gF"),
    (_RE_OPT_OPTIMIZER, "optimizer"),
    (_RE_OPT_PARAM_UPDATE, "param_update"),
    (_RE_OPT_MPI_BARRIER, "mpi_barrier"),
    (_RE_OPT_MISC, "misc"),
)

_GFMC_BREAKDOWN_PATTERNS = (
    (_RE_TIME_PROJECTION, "projection"),
    (_RE_TIME_OBSERVABLE, "observable"),
    (_RE_TIME_MPI_GFMC, "mpi_barrier"),
    (_RE_TIME_COLLECTION, "collection"),
    (_RE_TIME_RECONFIGURATION, "reconfiguration"),
    (_RE_TIME_E_L, "e_L"),
    (_RE_TIME_DE_L_DR, "de_L_dR_dr"),
    (_RE_TIME_DLN_PSI_DR, "dln_Psi_dR_dr"),
    (_RE_TIME_DLN_PSI_DC, "dln_Psi_dc"),
    (_RE_TIME_DE_L_DC, "de_L_dc"),
    (_RE_TIME_STRESS, "stress"),
    (_RE_TIME_UPDATE_ESCF, "update_E_scf"),
    (_RE_TIME_MISC_GFMC, "misc"),
)

_STRESS_COMPONENTS = ("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz")
_PRESSURE_COMPONENTS = ("P_x", "P_y", "P_z", "P_hydro")


# -- Internal helpers ----------------------------------------------


def _last(pattern: re.Pattern, text: str, group: int = 1, cast=float):
    """Return the last match of *pattern* in *text* (group *group*, cast), or ``None``."""
    value = None
    for m in pattern.finditer(text):
        value = m.group(group)
    return None if value is None else cast(value)


def _last_breakdown(patterns, text: str) -> dict:
    """Return ``{key: value of the last match}`` for each found ``(pattern, key)``."""
    result = {}
    for pattern, key in patterns:
        value = _last(pattern, text)
        if value is not None:
            result[key] = value
    return result


def _percent_to_fraction(value: float | None) -> float | None:
    return None if value is None else value / 100.0


def _value(text: str):
    """A printed input value as a Python value: number, bool, None, dict/list literal, or the string itself."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _shape(text: str) -> list:
    return [int(token) for token in text.replace(" ", "").split(",") if token]


def _key_values(text: str) -> dict:
    """``"a=1, b=[1, 2], c=(2, 8), d=x"`` -> ``{"a": 1, "b": [1, 2], "c": (2, 8), "d": "x"}``."""
    result = {}
    depth, start, items = 0, 0, []
    for index, char in enumerate(text):
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            items.append(text[start:index])
            start = index + 1
    items.append(text[start:])
    for item in items:
        key, _, value = item.partition("=")
        if key.strip():
            result[key.strip()] = _value(value.strip())
    return result


# -- val(err) tokens and tables --------------------------------------


def parse_ufloat_short(text: str):
    """Parse uncertainties short format, e.g. ``+0.123(45)`` -> (0.123, 0.045).

    Handles several notations produced by jqmc:

    * ``+0.0114(14)``   -- integer uncertainty digits in last decimal place
    * ``+3(8)e-05``     -- scientific notation with integer uncertainty
    * ``+3.9(3.5)e-05`` -- scientific notation with decimal uncertainty

    Args:
        text (str):
            A single token like ``"+0.0114(14)"``, ``"-1.23(4)"``,
            ``"+3(8)e-05"``, or ``"+3.9(3.5)e-05"``.

    Returns:
        tuple:
            ``(value, uncertainty)`` or ``(None, None)`` on failure.
    """
    m = re.match(r"([+-]?(?:\d+\.?\d*|nan|inf))\((\d+\.?\d*|nan|inf)\)([eE][+-]?\d+)?", text.strip(), re.IGNORECASE)
    if not m:
        return None, None
    val_str = m.group(1)
    unc_str = m.group(2)
    exp_str = m.group(3)  # e.g. "e-05" or None
    val = float(val_str)
    # A non-finite value or error is a diverged run: keep it, as _NUM does.
    if not (math.isfinite(val) and math.isfinite(float(unc_str))):
        return val, float(unc_str)
    if "." in unc_str:
        # Decimal uncertainty: use the value directly at the mantissa scale
        unc = float(unc_str)
    else:
        # Integer uncertainty: digits in the last decimal place of the value
        if "." in val_str:
            n_decimals = len(val_str.split(".")[1])
        else:
            n_decimals = 0
        unc = int(unc_str) * 10 ** (-n_decimals)
    if exp_str is not None:
        exp_factor = 10 ** int(exp_str[1:])  # skip 'e'/'E'
        val *= exp_factor
        unc *= exp_factor
    return val, unc


def parse_force_table(text: str):
    """Parse an ``Atomic Forces:`` table from jqmc or jqmc-tool output.

    Expected format::

        Atomic Forces:
        ------------------------------------------------
        Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
        ------------------------------------------------
        O       -0.0114(14)  -0.0082(11)  -0.0173(13)
        H       +0.0112(11)  +0.0050(7)   +0.0054(9)
        ------------------------------------------------

    Args:
        text (str):
            Full stdout of ``jqmc`` or ``jqmc-tool {mcmc,lrdmc} compute-force``.

    Returns:
        list of dict or None:
            Each dict: ``{label, Fx, Fx_err, Fy, Fy_err, Fz, Fz_err}``.
            Returns *None* when no force data is found.
    """
    lines = text.splitlines()
    forces = []
    in_data = False
    for line in lines:
        if "Atomic Forces:" in line:
            in_data = False
            continue
        stripped = line.strip()
        if "Fx" in stripped:
            in_data = True
            continue
        if stripped.startswith("---"):
            if in_data and forces:
                break
            continue
        if not in_data:
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        label = tokens[0]
        fx, fx_err = parse_ufloat_short(tokens[1])
        fy, fy_err = parse_ufloat_short(tokens[2])
        fz, fz_err = parse_ufloat_short(tokens[3])
        if fx is not None:
            forces.append(
                {
                    "label": label,
                    "Fx": fx,
                    "Fx_err": fx_err,
                    "Fy": fy,
                    "Fy_err": fy_err,
                    "Fz": fz,
                    "Fz_err": fz_err,
                }
            )
    return forces or None


def _parse_labelled_ufloat_table(text: str, header: str, labels: tuple[str, ...]):
    """Parse a ``label  value`` table delimited by ``-----`` separators.

    Used for both the ``Stress Tensor (sigma):`` and the
    ``Pressure (P = -diag(sigma)):`` tables emitted by ``jqmc_cli``.
    A row whose label is not in *labels* is silently skipped, so a
    header line like ``Component   Value (Ha/bohr^3)`` does not
    confuse the parser even though its first token (``Component``)
    is not in the whitelist.
    """
    lines = text.splitlines()
    in_block = False
    rows: dict[str, tuple[float, float]] = {}
    seen_dash_after_header = False
    for line in lines:
        if header in line:
            in_block = True
            seen_dash_after_header = False
            rows = {}
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if stripped.startswith("---"):
            if seen_dash_after_header and rows:
                break
            seen_dash_after_header = True
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        label = tokens[0]
        if label not in labels:
            continue
        val, unc = parse_ufloat_short(tokens[1])
        if val is None:
            continue
        rows[label] = (val, unc)
    if not rows:
        return None
    return rows


def parse_stress_table(text: str):
    """Parse a ``Stress Tensor (sigma):`` table from jqmc stdout.

    Expected format (mirrors :func:`parse_force_table`)::

        Stress Tensor (sigma):
        ------------------------------------------------
        Component   Value (Ha/bohr^3)
        ------------------------------------------------
        sigma_xx    +0.001234(56)
        sigma_yy    +0.001234(56)
        sigma_zz    +0.001234(56)
        sigma_xy    +0.000012(34)
        sigma_xz    +0.000012(34)
        sigma_yz    +0.000012(34)
        ------------------------------------------------

    Args:
        text: Full stdout from a ``jqmc`` MCMC / lrdmc-tau run.

    Returns:
        ``{sigma_xx, sigma_xx_err, ..., sigma_yz, sigma_yz_err}`` or
        *None* when no stress table is found.
    """
    rows = _parse_labelled_ufloat_table(text, "Stress Tensor", _STRESS_COMPONENTS)
    if rows is None:
        return None
    out: dict[str, float] = {}
    for comp in _STRESS_COMPONENTS:
        if comp not in rows:
            return None
        out[comp] = rows[comp][0]
        out[f"{comp}_err"] = rows[comp][1]
    return out


def parse_pressure_table(text: str):
    """Parse a ``Pressure (P = -diag(sigma)):`` table from jqmc stdout.

    Expected format::

        Pressure (P = -diag(sigma)):
        ------------------------------------------------
        Component   Value (Ha/bohr^3)
        ------------------------------------------------
        P_x         -0.001234(56)
        P_y         -0.001234(56)
        P_z         -0.001234(56)
        P_hydro     -0.001234(56)
        ------------------------------------------------

    Returns:
        ``{Px, Px_err, Py, Py_err, Pz, Pz_err, P_hydro, P_hydro_err}``
        or *None* when no pressure table is found.
    """
    rows = _parse_labelled_ufloat_table(text, "Pressure", _PRESSURE_COMPONENTS)
    if rows is None:
        return None
    out: dict[str, float] = {}
    for comp in _PRESSURE_COMPONENTS:
        if comp not in rows:
            return None
        # ``P_x`` -> ``Px`` (drop the underscore so callers can write
        # ``result.pressure["Px"]`` instead of escaping the underscore).
        key = comp.replace("_x", "x").replace("_y", "y").replace("_z", "z") if comp != "P_hydro" else "P_hydro"
        out[key] = rows[comp][0]
        out[f"{key}_err"] = rows[comp][1]
    return out


# -- Run metadata and per-step VMC data --------------------------------


def parse_run_metadata(text: str) -> dict:
    """Extract run-level metadata from a jQMC output file header.

    Parses the header section that appears once per output file before
    any computation output.  Returns a dict with keys::

        {
            "num_mpi_processes": int | None,
            "num_walkers_per_process": int | None,
            "jax_backend": str | None,      # "gpu", "cpu", etc.
            "jax_devices": list | None,      # ["CudaDevice(id=0)", ...]
        }
    """
    meta: dict = {
        "num_mpi_processes": None,
        "num_walkers_per_process": None,
        "jax_backend": None,
        "jax_devices": None,
    }

    for line in text.splitlines():
        # MPI processes
        m = _RE_MPI_PROCESSES.search(line)
        if m:
            meta["num_mpi_processes"] = int(m.group(1))
            continue

        # Walkers per process
        m = _RE_WALKERS_PER_PROCESS.search(line)
        if m:
            meta["num_walkers_per_process"] = int(m.group(1))
            continue

        # JAX backend (explicit)
        m = _RE_JAX_BACKEND.search(line)
        if m:
            meta["jax_backend"] = m.group(1).lower()
            continue

        # JAX backend (fallback: CPU/single GPU)
        if _RE_JAX_CPU_FALLBACK.search(line):
            if meta["jax_backend"] is None:
                meta["jax_backend"] = "cpu"
            continue

    # XLA Global devices -- parse from the full text (may span lines)
    m_header = _RE_XLA_GLOBAL_HEADER.search(text)
    if m_header:
        # The device list is on the next non-empty line after the header
        rest = text[m_header.end() :]
        for next_line in rest.splitlines():
            stripped = next_line.strip()
            if not stripped:
                continue
            m_devs = _RE_XLA_DEVICE_LIST.search(stripped)
            if m_devs:
                # Split "CudaDevice(id=0), CudaDevice(id=1), ..."
                raw_devices = m_devs.group(1)
                meta["jax_devices"] = [d.strip() for d in raw_devices.split(",") if d.strip()]
            break

    return meta


def _new_vmc_step(step: int) -> dict:
    return {
        "step": step,
        "energy": None,
        "energy_error": None,
        "max_force": None,
        "max_force_error": None,
        "signal_to_noise_ratio": None,
        "avg_walker_weight": None,
        "acceptance_ratio": None,
        "total_time_sec": None,
        "precompilation_time_sec": None,
        "net_time_sec": None,
        "timing_breakdown": {},
        "num_variational_parameters": None,
        "num_optimization_coordinates": None,
        "num_variational_parameters_all_twists": None,
        "num_optimization_coordinates_all_twists": None,
        "twist_local_max_signal_to_noise": [],
        "dumped_hamiltonian": None,
        "num_mcmc_steps": None,
        "num_mcmc_warmup_steps": None,
        "num_mcmc_bin_blocks": None,
        "optimization_batch": None,
        "derivative_sources": [],
        "asr": {},
        "lm": {},
        "diagnostics": {},
        "top_coordinates": [],
        "blocks": [],
        "sr_solver_path": None,
        "theta": {},
        "parameter_updates": [],
        "optimizer": None,
        "memory": [],
    }


def parse_vmc_steps(text: str) -> list:
    """Parse VMC log text into one dict per optimization step.

    The VMC output repeats blocks like::

        ==================================================================
        Optimization step = 1/10.
        ==================================================================
        ...
        Net total time for MCMC = 9.72 sec.
        Average of walker weights is 0.799. ...
        Acceptance ratio is 51.46 %. ...
        E = -16.61072 +- 0.00704 Ha
        Max generalized force = 17.984 +- 0.330 Ha/a.u.
        Max generalized force signal-to-noise = 126.871.
        ...

    Lines before the first ``Optimization step`` header are ignored.

    Returns:
        list of dict:
            One entry per optimization step found, with keys ``step``,
            ``energy``, ``energy_error``, ``max_force``, ``max_force_error``,
            ``signal_to_noise_ratio``, ``avg_walker_weight``,
            ``acceptance_ratio``, ``total_time_sec``,
            ``precompilation_time_sec``, ``net_time_sec``,
            ``timing_breakdown``, ``num_variational_parameters``,
            ``num_optimization_coordinates`` and ``dumped_hamiltonian``.
            A twist-averaged run also fills
            ``num_variational_parameters_all_twists``,
            ``num_optimization_coordinates_all_twists`` and
            ``twist_local_max_signal_to_noise`` (``[{"twist", "signal_to_noise"}, ...]``);
            its ``max_force`` and ``signal_to_noise_ratio`` are maxima over the
            shared coordinates and every twist's local ones.
            At ``verbosity = "high"`` it also gives
            ``diagnostics["coordinates_all_twists"]``, top coordinates labelled
            ``block[tK:qJ]`` and one ``blocks`` entry per twist (``block[tK]``)
            for each twist-local block.
            With the linear method (``lm_subspace_dim`` -1 or > 0), ``lm`` holds
            ``conditioning`` (``{active, coordinates, tiny_diagonal_removed,
            correlation_removed, min_correlation_eigenvalue, cutoff}``) and, at
            ``verbosity = "high"``, ``conditioning_input`` (``{coordinates,
            cutoff, diag_min, diag_max}``), ``selected_state``
            (``{reference_weight, q_norm, max_abs_q}``), ``subspace``
            (``{coordinates, atomic_groups}``) and ``decision``
            (``{accepted, reference_weight}``, or ``{accepted, fallback_scale,
            reason[, reference_weight]}`` when rejected).
    """
    steps: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        # -- Optimization step header --
        m = _RE_OPT_STEP.search(line)
        if m:
            current = _new_vmc_step(int(m.group(1)))
            steps.append(current)
            continue

        if current is None:
            continue

        # -- MCMC times of this step --
        m = _RE_NET_MCMC.search(line)
        if m:
            current["net_time_sec"] = float(m.group(1))
            continue
        m = _RE_TOTAL_MCMC.search(line)
        if m:
            current["total_time_sec"] = float(m.group(1))
            continue
        m = _RE_PRECOMP_MCMC.search(line)
        if m:
            current["precompilation_time_sec"] = float(m.group(1))
            continue

        # -- Per-step MCMC timing breakdown (msec) --
        matched = False
        for pattern, key in _MCMC_BREAKDOWN_PATTERNS:
            m = pattern.search(line)
            if m:
                current["timing_breakdown"][key] = float(m.group(1))
                matched = True
                break
        if matched:
            continue

        m = _RE_WALKER_WEIGHT.search(line)
        if m:
            current["avg_walker_weight"] = float(m.group(1))
            continue
        m = _RE_ACCEPTANCE.search(line)
        if m:
            current["acceptance_ratio"] = float(m.group(1)) / 100.0
            continue

        # -- Signal-to-noise and max force (must be checked BEFORE energy) --
        m = _RE_SNR.search(line)
        if m:
            current["signal_to_noise_ratio"] = float(m.group(1))
            continue
        m = _RE_MAX_FORCE.search(line)
        if m:
            current["max_force"] = float(m.group(1))
            current["max_force_error"] = float(m.group(2))
            continue

        m = _RE_ENERGY.search(line)
        if m:
            current["energy"] = float(m.group(1))
            current["energy_error"] = float(m.group(2))
            continue

        m = _RE_OPT_NUM_PARAMS.search(line)
        if m:
            current["num_variational_parameters"] = int(m.group(1))
            continue
        m = _RE_OPT_NUM_COORDINATES.search(line)
        if m:
            current["num_optimization_coordinates"] = int(m.group(1))
            continue
        m = _RE_OPT_NUM_PARAMS_ALL_TWISTS.search(line)
        if m:
            current["num_variational_parameters_all_twists"] = int(m.group(1))
            continue
        m = _RE_OPT_NUM_COORDINATES_ALL_TWISTS.search(line)
        if m:
            current["num_optimization_coordinates_all_twists"] = int(m.group(1))
            continue
        m = _RE_TWIST_LOCAL_SNR.search(line)
        if m:
            current["twist_local_max_signal_to_noise"].append({"twist": int(m.group(1)), "signal_to_noise": float(m.group(2))})
            continue
        m = _RE_OPT_DUMP.search(line)
        if m:
            current["dumped_hamiltonian"] = m.group(1)
            continue

        if _parse_vmc_step_detail(line, current):
            continue

    return steps


def _parse_vmc_step_detail(line: str, step: dict) -> bool:
    """Optimizer details of one step (most need ``verbosity = "high"``); True if *line* was used."""
    simple = (
        (_RE_OPT_SAMPLING_STEPS, "num_mcmc_steps"),
        (_RE_OPT_WARMUP, "num_mcmc_warmup_steps"),
        (_RE_OPT_BIN_BLOCKS, "num_mcmc_bin_blocks"),
    )
    for pattern, key in simple:
        m = pattern.match(line)
        if m:
            step[key] = int(m.group(1))
            return True
    m = _RE_OPT_BATCH.match(line)
    if m:
        step["optimization_batch"] = {
            "route": m.group(1),
            "local_samples": int(m.group(2)),
            "sampler_ranks": int(m.group(3)),
            "real_coordinates": int(m.group(4)),
            "storage_parameters": int(m.group(5)),
        }
        return True
    m = _RE_OPT_DERIVATIVE_SOURCE.match(line)
    if m:
        step["derivative_sources"].append(
            {
                "block": m.group(1),
                "O_shape": _shape(m.group(2)),
                "D_shape": None if m.group(3) is None else _shape(m.group(3)),
                "policy": m.group(4),
                "tangent": m.group(5),
            }
        )
        return True
    asr = step["asr"]
    m = _RE_ASR_GAMMAS.match(line)
    if m:
        asr["gamma_plus"], asr["gamma_minus"] = float(m.group(1)), float(m.group(2))
        return True
    m = _RE_ASR_SELECTED.match(line)
    if m:
        asr["selected_root"], asr["selected_gamma"], asr["selection_rule"] = m.group(1), float(m.group(2)), m.group(3)
        return True
    m = _RE_ASR_FINAL.match(line)
    if m:
        asr["final_gamma"] = float(m.group(1))
        return True
    for reason, pattern in _RE_ASR_FALLBACKS.items():
        m = pattern.match(line)
        if m:
            record = {"reason": reason, **dict(zip(_ASR_FALLBACK_FIELDS[reason], map(float, m.groups()), strict=True))}
            asr.setdefault("fallbacks", []).append(record)
            if reason == "root_not_finite_or_zero":
                asr.setdefault("notes", []).append(line.strip()[len("aSR: ") :].rstrip("."))
            return True
    m = _RE_ASR_DISCRIMINANT.match(line)
    if m:
        asr["clipped_discriminant"] = float(m.group(1))
        return True
    m = _RE_ASR_NOTE.match(line)
    if m:
        asr.setdefault("notes", []).append(m.group(1))
        return True
    diagnostics = step["diagnostics"]
    m = _RE_DIAG_COORDINATES.match(line)
    if m:
        diagnostics["coordinates"] = dict(
            zip(("real", "storage", "active", "frozen", "atomic_groups"), map(int, m.groups()), strict=True)
        )
        return True
    m = _RE_DIAG_COORDINATES_ALL_TWISTS.match(line)
    if m:
        diagnostics["coordinates_all_twists"] = dict(
            zip(("real", "storage", "active", "frozen", "atomic_groups"), map(int, m.groups()), strict=True)
        )
        return True
    m = _RE_DIAG_METRIC.match(line)
    if m:
        diagnostics["metric_diagonal"] = dict(
            zip(("all_min", "all_max", "active_min", "active_max"), map(float, m.groups()), strict=True)
        )
        return True
    m = _RE_DIAG_SOLVE.match(line)
    if m:
        diagnostics["sr_solve"] = {
            "regime": m.group(1),
            "method": m.group(2),
            "iterations": int(m.group(3)),
            "residual": float(m.group(4)),
            "tolerance": float(m.group(5)),
            "converged": m.group(6) == "True",
        }
        return True
    m = _RE_DIAG_DIRECTION.match(line)
    if m:
        diagnostics["sr_direction"] = {"norm": float(m.group(1)), "max_abs": float(m.group(2))}
        return True
    if _RE_DIAG_TOP_HEADER.match(line):
        return True
    m = _RE_DIAG_TOP_ROW.match(line)
    if m:
        step["top_coordinates"].append(
            {
                "coordinate": m.group(1),
                "force": float(m.group(2)),
                "force_error": float(m.group(3)),
                "signal_to_noise": float(m.group(4)),
                "metric_diag": float(m.group(5)),
            }
        )
        return True
    m = _RE_DIAG_DECISION.match(line)
    if m:
        asr["scale"], asr["model_fallback"] = float(m.group(1)), m.group(2) == "True"
        return True
    lm = step["lm"]
    m = _RE_LM_CONDITIONING_INPUT.match(line)
    if m:
        lm["conditioning_input"] = {
            "coordinates": int(m.group(1)),
            "cutoff": float(m.group(2)),
            "diag_min": float(m.group(3)),
            "diag_max": float(m.group(4)),
        }
        return True
    m = _RE_LM_CONDITIONING.match(line)
    if m:
        lm["conditioning"] = {
            "active": int(m.group(1)),
            "coordinates": int(m.group(2)),
            "tiny_diagonal_removed": int(m.group(3)),
            "correlation_removed": int(m.group(4)),
            "min_correlation_eigenvalue": float(m.group(5)),
            "cutoff": float(m.group(6)),
        }
        return True
    m = _RE_LM_SELECTED_STATE.match(line)
    if m:
        lm["selected_state"] = dict(zip(("reference_weight", "q_norm", "max_abs_q"), map(float, m.groups()), strict=True))
        return True
    m = _RE_LM_SUBSPACE.match(line)
    if m:
        lm["subspace"] = {"coordinates": int(m.group(1)), "atomic_groups": int(m.group(2))}
        return True
    m = _RE_LM_ACCEPTED.match(line)
    if m:
        lm["decision"] = {"accepted": True, "reference_weight": float(m.group(1))}
        return True
    m = _RE_LM_REJECTED.match(line)
    if m:
        lm["decision"] = {"accepted": False, "fallback_scale": float(m.group(1)), "reason": m.group(2)}
        return True
    m = _RE_LM_REJECTED_WEIGHT.match(line)
    if m and "decision" in lm:
        lm["decision"]["reference_weight"] = float(m.group(1))
        return True
    m = _RE_DIAG_LNPSI.match(line)
    if m:
        diagnostics["ln_psi_change_norm"] = float(m.group(1))
        return True
    m = _RE_DIAG_FINAL_UPDATE.match(line)
    if m:
        diagnostics["final_update"] = {"norm": float(m.group(1)), "max_abs": float(m.group(2))}
        return True
    m = _RE_DIAG_BLOCK.match(line)
    if m:
        step["blocks"].append(
            {
                "block": m.group(1),
                "force_norm": float(m.group(2)),
                "max_abs_force": float(m.group(3)),
                "max_signal_to_noise": float(m.group(4)),
                "metric_diag_min": float(m.group(5)),
                "metric_diag_max": float(m.group(6)),
                "update": None,
            }
        )
        return True
    m = _RE_DIAG_BLOCK_UPDATE.match(line)
    if m and step["blocks"]:
        step["blocks"][-1]["update"] = dict(
            zip(("delta_q_norm", "max_abs_delta_q", "delta_p_norm", "max_abs_delta_p"), map(float, m.groups()), strict=True)
        )
        return True
    m = _RE_DIAG_BLOCK_STIEFEL.match(line)
    if m and step["blocks"]:
        step["blocks"][-1]["s_stiefel_tangent"] = {
            "frobenius_norm": float(m.group(1)),
            "spectral_norm": float(m.group(2)),
        }
        return True
    m = _RE_SR_PATH.match(line)
    if m:
        step["sr_solver_path"] = {"regime": m.group(1), "method": m.group(2), "backend": m.group(3)}
        return True
    theta = step["theta"]
    m = _RE_THETA_SIZE.match(line)
    if m:
        theta["size"] = int(m.group(1))
        return True
    m = _RE_THETA_NONZERO.match(line)
    if m:
        theta["nonzero"] = int(m.group(1))
        return True
    m = _RE_THETA_RANGE.match(line)
    if m:
        theta["max_abs"], theta["min_abs"] = float(m.group(1)), float(m.group(2))
        return True
    m = _RE_PARAMETER_UPDATE.match(line)
    if m:
        step["parameter_updates"].append(
            {
                "method": m.group(1),
                "block": m.group(2),
                "size": int(m.group(3)),
                "delta_p_norm": float(m.group(4)),
                "max_abs_delta_p": float(m.group(5)),
                "max_abs_applied_update": float(m.group(6)),
            }
        )
        return True
    m = _RE_PARAMETER_NO_UPDATE.match(line)
    if m:
        # theta is exactly zero on this block, so every update norm is 0.
        step["parameter_updates"].append(
            {
                "method": m.group(1),
                "block": m.group(2),
                "size": int(m.group(3)),
                "delta_p_norm": 0.0,
                "max_abs_delta_p": 0.0,
                "max_abs_applied_update": 0.0,
            }
        )
        return True
    m = _RE_OPT_UPDATING.match(line)
    if m:
        step["optimizer"] = m.group(1)
        return True
    m = _RE_MEMORY.match(line)
    if m:
        step["memory"].append(_memory_entry(m))
        return True
    return False


def _memory_entry(m: re.Match) -> dict:
    """One ``[memory]`` line: ``stage``, ``rss_peak_max_gb`` and, when printed, ``jax_peak_max_gb``."""
    entry = {"stage": m.group(1), "rss_peak_max_gb": float(m.group(2))}
    if m.group(3) is not None:
        entry["jax_peak_max_gb"] = float(m.group(3))
    return entry


# -- Header: environment, input and system --------------------------------


def parse_environment(text: str) -> dict:
    """Software, hardware and precision of the run, from the header of a ``jqmc`` log.

    Returns:
        dict: ``jqmc_version``, ``git_commit``, ``library_versions`` (``jax``,
        ``jaxlib``, ``numpy``, ``python``; ``verbosity = "high"`` only),
        ``device_info``, ``process_count``, ``platform`` (OS family and
        architecture, e.g. ``"Linux x86_64"``), ``xla_flags``,
        ``jax_environment_variables`` (``{name: value}`` of the ``JAX_*`` and
        ``XLA_*`` variables echoed by ``jax.print_environment_info``),
        ``execution_backend`` (e.g. ``"XLA + Rust"``), ``rust_threads``,
        ``precision_mode``, ``precision_zones`` (``{zone: dtype}``),
        ``started``/``ended`` (timestamps as printed) and ``memory``
        (``[{stage, rss_peak_max_gb}]``, in order; an entry also has
        ``jax_peak_max_gb`` when its line prints the device peak).
    """
    zones = {}
    detail = re.search(r"^Zone detail:\s*$", text, re.MULTILINE)
    if detail:
        for line in text[detail.end() :].splitlines()[1:]:
            m = re.match(r"^  (\w+): (\S+)\s*$", line)
            if not m:
                break
            zones[m.group(1)] = m.group(2)
    return {
        "jqmc_version": _last(_RE_VERSION, text, cast=str),
        "git_commit": _last(_RE_GIT_COMMIT, text, cast=str),
        "library_versions": {m.group(1): m.group(2) for m in _RE_LIB_VERSION.finditer(text)},
        "device_info": _last(_RE_DEVICE_INFO, text, cast=str),
        "process_count": _last(_RE_PROCESS_COUNT, text, cast=int),
        "platform": _last(_RE_PLATFORM, text, cast=str),
        "xla_flags": _last(_RE_XLA_FLAGS, text, cast=str),
        "jax_environment_variables": {m.group(1): m.group(2) for m in _RE_JAX_ENV_VAR.finditer(text)},
        "execution_backend": _last(_RE_EXECUTION_BACKEND, text, cast=str),
        "rust_threads": _last(_RE_RUST_THREADS, text, cast=int),
        "precision_mode": _last(_RE_PRECISION, text, cast=str),
        "precision_zones": zones,
        "started": _last(_RE_PROGRAM_STARTS, text, cast=str),
        "ended": _last(_RE_PROGRAM_ENDS_AT, text, cast=str),
        "memory": [_memory_entry(m) for m in _RE_MEMORY.finditer(text)],
    }


def parse_input_parameters(text: str) -> dict:
    """The input of the run as jqmc applied it.

    Returns:
        dict:
            ``input_file``; ``explicit`` -- ``{section: {key: value}}`` echoed
            from the input file; ``defaults`` -- ``{key: value}`` of every key
            jqmc filled with its default; ``restored`` -- ``{key: value}`` of
            every key a restart took from the checkpoint (``None`` for
            non-scalar ones such as the Hamiltonian). Values are converted
            to numbers, booleans, dicts or lists where they print as such.
    """
    explicit: dict = {}
    m = re.search(r"^Input parameters are::\s*$", text, re.MULTILINE)
    if m:
        section = None
        for line in text[m.end() :].splitlines()[1:]:
            m_section = _RE_INPUT_SECTION.match(line)
            if m_section:
                section = explicit.setdefault(m_section.group(1), {})
                continue
            m_item = _RE_INPUT_ITEM.match(line)
            if m_item and section is not None:
                section[m_item.group(1)] = _value(m_item.group(2))
                continue
            if not line.strip():
                continue
            break
    defaults = {m.group(1): _value(m.group(2)) for m in _RE_DEFAULT_VALUE.finditer(text)}
    # Logs of earlier versions, which filled an omitted [lrdmc-bra]
    # num_mcmc_per_measurement from alat, carry an explanation after the value.
    for m in _RE_DEFAULT_ALAT_SCALED.finditer(text):
        defaults[m.group(1)] = int(m.group(2))
    restored = {m.group(1): _value(m.group(2)) for m in _RE_RESTORED_VALUE.finditer(text)}
    restored.update({m.group(2): None for m in _RE_RESTORED_NONSCALAR.finditer(text)})
    return {
        "input_file": _last(_RE_INPUT_FILE, text, cast=str),
        "explicit": explicit,
        "defaults": defaults,
        "restored": restored,
    }


def parse_system_info(text: str) -> dict:
    """The system as summarised by jqmc from the Hamiltonian (the first summary printed).

    Returns:
        dict: ``pbc``, ``atoms`` (``[{element, label, Z, position}]``,
        Bohr), ``ecp``, ``num_electron_up``, ``num_electron_dn``,
        ``occupied_mo_rows`` ([up, dn]), ``time_reversal``, ``num_mos``,
        ``num_aos``, ``num_primitive_aos``, ``mo_coefficients_shape``,
        ``jastrow_2b_param``, ``jastrow_2b_type``,
        ``jastrow_3b_matrix_shape``, ``mo_orthonormality``
        (``[{data, spin, max_diag_deviation, max_offdiag}]``),
        ``initial_walker_shapes`` (``{up, dn}``; ``verbosity = "high"``),
        for periodic GTOs, ``ao_images`` (``{num_image_cells,
        max_image_index, truncation_tolerance}``) and, for tiled orbitals,
        ``tiled_orbitals`` (one entry per spin channel, see ``_tiled_orbitals``).
    """

    def first(pattern, cast=str, group=1):
        m = pattern.search(text)
        return None if m is None else cast(m.group(group))

    electrons = {m.group(1): int(m.group(2)) for m in _RE_NUM_ELECTRON.finditer(text)}
    rows = _RE_OCCUPIED_ROWS.search(text)
    mo_dim = _RE_MO_COEFF_DIM.search(text)
    j3_dim = _RE_J3_DIM.search(text)
    return {
        "pbc": first(_RE_PBC_FLAG, lambda v: v == "True"),
        "atoms": [
            {
                "element": m.group(1),
                "label": m.group(2),
                "Z": float(m.group(3)),
                "position": [float(m.group(4)), float(m.group(5)), float(m.group(6))],
            }
            for m in _RE_ATOM_ROW.finditer(text)
        ],
        "ecp": first(_RE_ECP_FLAG, lambda v: v == "True"),
        "num_electron_up": electrons.get("up"),
        "num_electron_dn": electrons.get("dn"),
        "occupied_mo_rows": None if rows is None else [int(rows.group(1)), int(rows.group(2))],
        "time_reversal": first(_RE_TIME_REVERSAL, lambda v: v == "True"),
        "num_mos": first(_RE_NUM_MOS, int),
        "num_aos": first(_RE_NUM_AOS, int),
        "num_primitive_aos": first(_RE_NUM_PRIMITIVE_AOS, int),
        "mo_coefficients_shape": None if mo_dim is None else [int(mo_dim.group(1)), int(mo_dim.group(2))],
        "jastrow_2b_param": first(_RE_J2_PARAM, float),
        "jastrow_2b_type": first(_RE_J2_TYPE),
        "jastrow_3b_matrix_shape": None if j3_dim is None else [int(j3_dim.group(1)), int(j3_dim.group(2))],
        "mo_orthonormality": [
            {
                "data": m.group(1),
                "spin": m.group(2),
                "max_diag_deviation": float(m.group(3)),
                "max_offdiag": float(m.group(4)),
            }
            for m in _RE_MO_ORTHONORMALITY.finditer(text)
        ],
        "initial_walker_shapes": {m.group(1): _shape(m.group(2)) for m in _RE_INITIAL_SHAPE.finditer(text)},
        "lattice_vectors": _lattice_vectors(text),
        "ewald": _first_values(_RE_EWALD, text),
        "bspline": _bspline(text),
        "ao_images": _ao_images(text),
        "jastrow_1b_param": first(_RE_J1_PARAM, float),
        "jastrow_3b_symmetric": first(_RE_J3_SYMMETRIC, lambda v: v == "True"),
        "jastrow_dtn": _jastrow_dtn(text),
        "lambda_matrix": _lambda_matrix(text),
        "tiled_orbitals": _tiled_orbitals(text),
    }


def _tiled_orbitals(text: str) -> list:
    """Tiled orbitals: ``[{spin, supertwist, tiling, num_k_blocks, num_occupied_orbitals, local_basis_sizes,
    orbital_offsets, normalization}]``, one entry per spin channel of the first summary."""
    channels: dict = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not _RE_TILED_HEADER.match(line):
            continue
        matches = [pattern.match(row) for pattern, row in zip(_RE_TILED.values(), lines[index + 1 : index + 5], strict=False)]
        if len(matches) != len(_RE_TILED) or not all(matches):
            raise ValueError(f"incomplete TiledOrbitalSet_data summary after line {index + 1}.")
        spin, k_blocks, basis_sizes, offsets = matches
        supertwist = [float(v) for v in re.sub(r"np\.float64\(([^()]*)\)", r"\1", spin.group(2)).split(",")]
        if len(supertwist) != 3:
            raise ValueError(f"TiledOrbitalSet_data supertwist after line {index + 1} does not have 3 components.")
        channels.setdefault(
            spin.group(1),
            {
                "spin": spin.group(1),
                "supertwist": supertwist,
                "tiling": [list(row) for row in ast.literal_eval(spin.group(3))],
                "num_k_blocks": int(k_blocks.group(1)),
                "num_occupied_orbitals": int(k_blocks.group(2)),
                "local_basis_sizes": _shape(basis_sizes.group(1)),
                "orbital_offsets": _shape(offsets.group(1)),
                "normalization": offsets.group(2),
            },
        )
    return list(channels.values())


def _first_values(pattern: re.Pattern, text: str) -> dict:
    """``{name: value}`` of the first occurrence of each name matched by *pattern* (2 groups)."""
    values: dict = {}
    for m in pattern.finditer(text):
        values.setdefault(m.group(1), _value(m.group(2)))
    return values


def _ao_images(text: str) -> dict | None:
    """Periodic GTOs: ``{num_image_cells, max_image_index, truncation_tolerance}`` (first summary)."""
    found = {name: pattern.search(text) for name, pattern in _RE_AO_IMAGES.items()}
    if found["num_image_cells"] is None:
        return None
    return {
        name: None if m is None else (float(m.group(1)) if name == "truncation_tolerance" else int(m.group(1)))
        for name, m in found.items()
    }


def _lattice_vectors(text: str) -> list | None:
    vectors: dict = {}
    for m in _RE_LATTICE_VECTOR.finditer(text):
        vectors.setdefault(m.group(1), [float(v) for v in m.group(2).split()])
    return [vectors[name] for name in "ABC"] if len(vectors) == 3 else None


def _bspline(text: str) -> dict | None:
    if not _RE_BSPLINE["grid"].search(text):
        return None
    values = {}
    for name, pattern in _RE_BSPLINE.items():
        m = pattern.search(text)
        if m:
            values[name] = (
                _shape(m.group(1)) if name == "grid" else m.group(1) == "True" if name == "complex_path" else int(m.group(1))
            )
    return values


def _jastrow_dtn(text: str) -> dict | None:
    """The DTN Jastrow summary: terms, cusp, C, polynomial orders, cutoffs, ratios and coefficient shapes."""
    m = _RE_DTN_TERMS.search(text)
    if m is None:
        return None
    ratios = _RE_DTN_RATIOS.search(text)
    orders = _RE_DTN_ORDERS.search(text)
    cutoffs = _RE_DTN_CUTOFFS.search(text)
    coefs = _RE_DTN_COEFS.search(text)
    return {
        "terms": _key_values(m.group(1)),
        "cusp": m.group(2),
        "C": int(m.group(3)),
        "orders": _key_values(orders.group(1)) if orders else {},
        "cutoffs": _key_values(cutoffs.group(1)) if cutoffs else {},
        "cutoff_ratios": _key_values(ratios.group(1)) if ratios else {},
        "cutoff_ratio_policy": ratios.group(2) if ratios else None,
        "independent_coefficients": {k: list(v) for k, v in _key_values(coefs.group(1)).items()} if coefs else {},
    }


def _lambda_matrix(text: str) -> dict | None:
    """Shapes of the geminal (JAGP) lambda matrix and its paired/unpaired blocks."""
    shapes: dict = {}
    for m in _RE_LAMBDA.finditer(text):
        shapes.setdefault(m.group(1) or m.group(2), _shape(m.group(3)))
    if not shapes:
        return None
    symmetric = _RE_LAMBDA_SYMMETRIC.search(text)
    return {
        "shape": shapes.get("lambda_matrix"),
        "paired_shape": shapes.get("lambda_matrix_paired"),
        "unpaired_shape": shapes.get("lambda_matrix_unpaired"),
        "symmetric": None if symmetric is None else symmetric.group(1) == "True",
    }


# -- jqmc stdout --------------------------------------------------------


def _parse_common(text: str) -> dict:
    """Fields printed by every ``jqmc`` run: outcome, final output and header metadata."""
    warnings: list = []
    for m in _RE_WARNING_MIN.finditer(text):
        if m.group(1) not in warnings:
            warnings.append(m.group(1))
    result = {
        "program_ended": bool(_RE_PROGRAM_ENDS.search(text)),
        "stopped_by_max_time": bool(_RE_STOP_MAX_TIME.search(text)),
        "stopped_by_stop_flag": bool(_RE_STOP_FLAG.search(text)),
        "incomplete_twists": [
            {"twist": int(m.group(1)), "steps": int(m.group(2)), "requested_steps": int(m.group(3))}
            for m in _RE_TWIST_STOPPED.finditer(text)
        ],
        "warnings": warnings,
        "restart_checkpoint": _last(_RE_RESTART_CHECKPOINT, text, cast=str),
        "energy": _last(_RE_FINAL_ENERGY, text, 1),
        "energy_error": _last(_RE_FINAL_ENERGY, text, 2),
        "variance": _last(_RE_FINAL_VARIANCE, text, 1),
        "variance_error": _last(_RE_FINAL_VARIANCE, text, 2),
        "dE_mpc": _last(_RE_FINAL_MPC_CORRECTION, text, 1),
        "dE_mpc_error": _last(_RE_FINAL_MPC_CORRECTION, text, 2),
        "energy_mpc": _last(_RE_FINAL_MPC_ENERGY, text, 1),
        "energy_mpc_error": _last(_RE_FINAL_MPC_ENERGY, text, 2),
        "twist_average": parse_twist_average_result(text),
        "twist_dispatch": _parse_twist_dispatch(text),
        "mpc_setup": _parse_mpc_setup(text),
    }
    result.update(parse_run_metadata(text))
    result["environment"] = parse_environment(text)
    result["input_parameters"] = parse_input_parameters(text)
    result["system"] = parse_system_info(text)
    return result


def _parse_twist_dispatch(text: str) -> dict | None:
    """The twist-averaged dispatch banner: ``{num_twists, hamiltonian_file, bz_weight_sum}``."""
    m = _RE_TWIST_DISPATCH.search(text)
    if m is None:
        return None
    return {"num_twists": int(m.group(1)), "hamiltonian_file": m.group(2), "bz_weight_sum": float(m.group(3))}


def _parse_mpc_setup(text: str) -> dict | None:
    """The MPC table setup: ranks, cutoff, grid, n_max, extrapolation note and the time of each step."""
    m = _RE_MPC_SETUP_GRID.search(text)
    if m is None:
        return None
    return {
        "broadcast_ranks": _last(_RE_MPC_SETUP_RANKS, text, cast=int),
        "cutoff": float(m.group(1)),
        "grid": _shape(m.group(2)),
        "n_max": int(m.group(3)),
        "note": m.group(4),
        "steps": [{"step": s.group(1), "time_sec": float(s.group(2))} for s in _RE_MPC_SETUP_STEP.finditer(text)],
        "total_time_sec": _last(_RE_MPC_SETUP_DONE, text),
    }


def parse_twist_average_result(text: str) -> dict | None:
    """The BZ-averaged final output of a twist-averaged run, or ``None`` for a single-twist run.

    Returns:
        dict: ``energy``, ``energy_error``, ``s_twist`` (twist dispersion,
        reported apart from the MC error), ``dE_mpc``, ``dE_mpc_error``,
        ``energy_mpc``, ``energy_mpc_error`` (``None`` without MPC) and
        ``per_twist`` (``[{twist, k_point, energy, energy_error}]``;
        ``k_point`` is ``None`` for Gamma).
    """
    m = None
    for m in _RE_TA_ENERGY.finditer(text):
        pass
    if m is None:
        return None
    per_twist = []
    header = None
    for header in _RE_TA_TABLE_HEADER.finditer(text):
        pass
    if header is not None:
        for row in text[header.end() :].splitlines():
            m_row = _RE_TA_TWIST_ROW.match(row)
            if m_row:
                k = m_row.group(2)
                energy, error = parse_ufloat_short(m_row.group(3))
                per_twist.append(
                    {
                        "twist": int(m_row.group(1)),
                        "k_point": None if k == "Gamma" else [float(v) for v in k.strip("()").split(",")],
                        "energy": energy,
                        "energy_error": error,
                    }
                )
            elif per_twist and row.strip().startswith("-"):
                break
    return {
        "energy": float(m.group(1)),
        "energy_error": float(m.group(2)),
        "s_twist": float(m.group(3)),
        "dE_mpc": _last(_RE_TA_MPC_CORRECTION, text, 1),
        "dE_mpc_error": _last(_RE_TA_MPC_CORRECTION, text, 2),
        "energy_mpc": _last(_RE_TA_MPC_ENERGY, text, 1),
        "energy_mpc_error": _last(_RE_TA_MPC_ENERGY, text, 2),
        "per_twist": per_twist,
    }


def parse_mcmc_log(text: str) -> dict:
    """Parse the stdout of ``jqmc`` with ``job_type = "mcmc"``.

    ``mcmc_counter`` is the cumulative number of measurements in the
    checkpoint (restarted runs included); ``mcmc_counter_target`` is the
    value this run was asked to reach. Diagnostics take the last
    occurrence in *text*.

    Returns:
        dict: See the module docstring for conventions.
    """
    result = _parse_common(text)
    result.update(
        {
            "mcmc_counter": _last(_RE_MCMC_PROGRESS, text, 1, int),
            "mcmc_counter_target": _last(_RE_MCMC_PROGRESS, text, 2, int),
            "acceptance_ratio": _percent_to_fraction(_last(_RE_ACCEPTANCE, text)),
            "avg_walker_weight": _last(_RE_WALKER_WEIGHT, text),
            "total_time_sec": _last(_RE_TOTAL_MCMC, text),
            "precompilation_time_sec": _last(_RE_PRECOMP_MCMC, text),
            "net_time_sec": _last(_RE_NET_MCMC, text),
            "timing_breakdown": _last_breakdown(_MCMC_BREAKDOWN_PATTERNS, text),
            "atomic_forces": parse_force_table(text),
            "stress_tensor": parse_stress_table(text),
            "pressure": parse_pressure_table(text),
        }
    )
    return result


def parse_vmc_log(text: str) -> dict:
    """Parse the stdout of ``jqmc`` with ``job_type = "vmc"``.

    ``steps`` holds one dict per optimization step (see
    :func:`parse_vmc_steps`); step numbers are cumulative across restarts.
    ``opt_step_counter`` is the last step printed and ``total_opt_steps``
    the number this run was asked to reach. ``dumped_hamiltonians`` lists
    every ``hamiltonian_data_opt_step_*.h5`` written, in order. The
    optimization-level timing is printed once, after the loop.

    Returns:
        dict: See the module docstring for conventions.
    """
    result = _parse_common(text)
    headers = list(_RE_OPT_STEP.finditer(text))
    opt_breakdown = {}
    for pattern, key in _OPT_BREAKDOWN_PATTERNS:
        m = pattern.search(text)
        if m:
            opt_breakdown[key] = float(m.group(1))
    m_total = _RE_TOTAL_OPT.search(text)
    result.update(
        {
            "steps": parse_vmc_steps(text),
            "opt_step_counter": int(headers[-1].group(1)) if headers else None,
            "total_opt_steps": int(headers[-1].group(2)) if headers else None,
            "dumped_hamiltonians": [m.group(1) for m in _RE_OPT_DUMP.finditer(text)],
            "total_opt_time_sec": float(m_total.group(1)) if m_total else None,
            "opt_timing_breakdown": opt_breakdown,
            "optimizer": {
                "name": _last(_RE_OPTIMIZER_NAME, text, cast=str),
                "mode": _last(_RE_OPTIMIZER_MODE, text, cast=str),
                "hyperparameters": _key_values(h) if (h := _last(_RE_OPTIMIZER_HYPER, text, cast=str)) else {},
            },
        }
    )
    return result


def parse_lrdmc_log(text: str) -> dict:
    """Parse the stdout of ``jqmc`` with ``job_type = "lrdmc-bra"`` or ``"lrdmc-tau"``.

    ``gfmc_counter`` is the cumulative number of branching steps in the
    checkpoint; ``gfmc_counter_target`` is the value this run was asked to
    reach. lrdmc-bra prints ``Progress: GFMC step = i/N`` and lrdmc-tau
    ``branching step = i/N``.

    Returns:
        dict: See the module docstring for conventions.
    """
    result = _parse_common(text)
    result.update(
        {
            "gfmc_counter": _last(_RE_GFMC_PROGRESS, text, 1, int),
            "gfmc_counter_target": _last(_RE_GFMC_PROGRESS, text, 2, int),
            "survived_walkers_ratio": _percent_to_fraction(_last(_RE_SURVIVED, text)),
            "avg_num_projections": _last(_RE_AVG_PROJECTIONS, text),
            "e_scf_history": [
                {"energy": float(m.group(1)), "energy_error": float(m.group(2))} for m in _RE_E_SCF.finditer(text)
            ],
            "total_time_sec": _last(_RE_TOTAL_GFMC, text),
            "precompilation_time_sec": _last(_RE_PRECOMP_GFMC, text),
            "net_time_sec": _last(_RE_NET_GFMC, text),
            "timing_breakdown": _last_breakdown(_GFMC_BREAKDOWN_PATTERNS, text),
            "atomic_forces": parse_force_table(text),
            "stress_tensor": parse_stress_table(text),
            "pressure": parse_pressure_table(text),
        }
    )
    return result


_JOB_TYPE_PARSERS = {
    "mcmc": parse_mcmc_log,
    "vmc": parse_vmc_log,
    "lrdmc-bra": parse_lrdmc_log,
    "lrdmc-tau": parse_lrdmc_log,
}


def parse_jqmc_log(text: str, job_type: str) -> dict:
    """Parse ``jqmc`` stdout according to *job_type*.

    Args:
        text (str): Rank-0 stdout of ``jqmc``.
        job_type (str): ``"mcmc"``, ``"vmc"``, ``"lrdmc-bra"`` or ``"lrdmc-tau"``.

    Returns:
        dict: Output of the job-type parser with ``job_type`` added.

    Raises:
        ValueError: If *job_type* is unknown.
    """
    if job_type not in _JOB_TYPE_PARSERS:
        raise ValueError(f"Unknown job_type = {job_type!r}. Choose from {sorted(_JOB_TYPE_PARSERS)}.")
    result = _JOB_TYPE_PARSERS[job_type](text)
    result["job_type"] = job_type
    result["unparsed_numeric_lines"] = unparsed_numeric_lines(text)
    return result


# -- Completeness -------------------------------------------------------

# A number standing alone (not part of an identifier such as ``j3`` or ``opt_J1_param``).
_RE_NUMERIC_TOKEN = re.compile(r"(?<![\w.])[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")

# Lines with numbers that are deliberately not extracted, with the reason.
_NOT_EXTRACTED = (
    # Basis summary (headers and Gaussian exponents/coefficients): part of the input Hamiltonian file.
    re.compile(rf"^\s+{_NUM}\s+{_NUM}\s*$"),
    re.compile(r"^\s+\*\*basis set for atom index \d+: \S+\*\*\s*$"),
    # Initial walker coordinates (numpy array print); only their shape is extracted.
    re.compile(r"^\s*(?:initial r_(?:up|dn)_carts\s*=\s*)?\[*\s*\[\[?\s*[-+0-9.eE ]+\]+\s*$"),
    # Compilation time, also printed as "Pre-compilation time for ...".
    re.compile(r"^Elapsed Time = [\d.]+ sec\.\s*$"),
    # Headers of timing tables; the step count is the run's own.
    re.compile(r"^Elapsed times per .*, averaged over \d+ .*steps\.\s*$"),
    # Device lists, extracted as ``jax_devices`` (global) and redundant per rank (local).
    re.compile(r"^\[?'?(?:Rank \d+: )?\[?\w*Device\(id=\d+"),
    # Rows of the force, stress and pressure tables (``parse_force_table`` & co.).
    re.compile(r"^\s+\S+(?:\s+[+-]?[\d.]+\([\d.]+\)(?:e[+-]?\d+)?)+\s*$"),
    # Unit line of the stress and pressure tables.
    re.compile(r"^\s+Component\s+Value \(Ha/bohr\^3\)\s*$"),
    # Input echo ("  key=value"), extracted by ``parse_input_parameters``.
    re.compile(r"^  \w+=.*$"),
)


def _coverage_patterns() -> list:
    patterns = []
    for name, value in globals().items():
        if not name.startswith("_RE_") or name in ("_RE_XLA_DEVICE_LIST", "_RE_NUMERIC_TOKEN"):
            continue
        if isinstance(value, re.Pattern):
            patterns.append(value)
        elif isinstance(value, dict):
            patterns.extend(v for v in value.values() if isinstance(v, re.Pattern))
    return patterns


def unparsed_numeric_lines(text: str) -> list:
    """Lines of a ``jqmc`` log that print a number but are not extracted by this module.

    The banner and license above ``All rights reserved.`` are skipped, and the
    lines in ``_NOT_EXTRACTED`` are known non-results. Anything else listed
    here is output that the parser does not structure yet; a consumer can
    still read it, and the tests require the list to be empty for real logs.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "All rights reserved.":
            lines = lines[index + 1 :]
            break
    patterns = _coverage_patterns()
    unparsed = []
    for line in lines:
        if not _RE_NUMERIC_TOKEN.search(line):
            continue
        if any(pattern.search(line) for pattern in patterns) or any(pattern.match(line) for pattern in _NOT_EXTRACTED):
            continue
        unparsed.append(line.strip())
    return unparsed


# -- jqmc-tool stdout ---------------------------------------------------


def parse_tool_energy(text: str) -> dict:
    """Parse ``jqmc-tool {mcmc,lrdmc} compute-energy`` stdout.

    Returns:
        dict: ``{energy, energy_error, dE_mpc, dE_mpc_error, energy_mpc,
        energy_mpc_error, num_mpi_ranks, num_binned_samples, num_twists}``.
        The MPC fields are ``None`` when no MPC estimate was stored;
        ``num_twists`` is ``None`` for a single-twist checkpoint.
    """
    return {
        "energy": _last(_RE_TOOL_ENERGY, text, 1),
        "energy_error": _last(_RE_TOOL_ENERGY, text, 2),
        "dE_mpc": _last(_RE_TOOL_MPC_CORRECTION, text, 1),
        "dE_mpc_error": _last(_RE_TOOL_MPC_CORRECTION, text, 2),
        "energy_mpc": _last(_RE_TOOL_MPC_ENERGY, text, 1),
        "energy_mpc_error": _last(_RE_TOOL_MPC_ENERGY, text, 2),
        "num_mpi_ranks": _last(_RE_TOOL_NUM_RANKS, text, cast=int),
        "num_binned_samples": _last(_RE_TOOL_NUM_SAMPLES, text, cast=int),
        "num_twists": _last(_RE_TOOL_NUM_TWISTS, text, cast=int),
    }


def parse_tool_force(text: str) -> dict:
    """Parse ``jqmc-tool {mcmc,lrdmc} compute-force`` stdout.

    Returns:
        dict: ``{atomic_forces, num_mpi_ranks, num_binned_samples}``.
    """
    return {
        "atomic_forces": parse_force_table(text),
        "num_mpi_ranks": _last(_RE_TOOL_NUM_RANKS, text, cast=int),
        "num_binned_samples": _last(_RE_TOOL_NUM_SAMPLES, text, cast=int),
    }


def parse_lrdmc_extrapolation(text: str) -> dict:
    """Parse ``jqmc-tool lrdmc extrapolate-energy`` stdout.

    Returns:
        dict:
            ``{extrapolated_energy, extrapolated_energy_error,
            extrapolated_energy_mpc, extrapolated_energy_mpc_error,
            polynomial_order, per_alat_results}``. ``per_alat_results`` lists
            ``{alat, quantity, energy, energy_error}``, ``quantity`` being
            ``"E"`` or ``"E_MPC"``.
    """
    return {
        "extrapolated_energy": _last(_RE_EXTRAP_ENERGY, text, 1),
        "extrapolated_energy_error": _last(_RE_EXTRAP_ENERGY, text, 2),
        "extrapolated_energy_mpc": _last(_RE_EXTRAP_MPC_ENERGY, text, 1),
        "extrapolated_energy_mpc_error": _last(_RE_EXTRAP_MPC_ENERGY, text, 2),
        "polynomial_order": _last(_RE_EXTRAP_ORDER, text, cast=int),
        "per_alat_results": [
            {
                "alat": float(m.group(1)),
                "quantity": m.group(2),
                "energy": float(m.group(3)),
                "energy_error": float(m.group(4)),
            }
            for m in _RE_EXTRAP_PER_ALAT.finditer(text)
        ],
    }


def parse_last_exception(text: str) -> str | None:
    """Return the last exception line of a traceback or a jqmc-tool error, if any.

    Args:
        text (str): stderr (or combined stdout/stderr) text.

    Returns:
        str or None: For example ``"ValueError: num_gfmc_warmup_steps (10) must be >= 25."``,
        ``"ValueError"`` for an exception raised without a message, or
        ``"Error: atomic_force was not enabled during the LRDMC run. ..."``.
    """
    message = None
    for m in _RE_EXCEPTION.finditer(text):
        message = f"{m.group(1)}: {m.group(2)}" if m.group(2) else m.group(1)
    return message
