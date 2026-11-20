"""All-electron hydrogen (packed-H2 ortho cell, k=(1/4, 1/4, 1/4)): QE -> jQMC workflow.

This is the *workflow-API* counterpart to the manual TOML tutorial in
``examples/jqmc-example08`` (same physical system, same ansatz). Use this one
when you want the launcher + state machine + automatic continuation + summary
parsing; use the TOML tutorial when you want to see what the workflow API is
generating under the hood.

Pipeline (single chain)
-----------------------

    QE_Workflow          ingest QE PW LDA-PZ output for hydrogen
        kind="pgto+pw"   keep the pGTO block (Slater-friendly) and an
        ecutwfc="auto"   adaptive PW residual; converter sweeps g_cut and
                         stops at the hard-coded overlap-residual target.
        jastrow_dtn="u_chi"  DTN polynomial Jastrow with u and chi only
                             (no f). DTN is mutually exclusive with the
                             Pade J1/J2/J3 family, so those stay at None.
        The 1/r AE Coulomb potential demands a cusp at the nucleus that pure
        Gaussians cannot supply. The DTN chi constraint applies it
        unconditionally for all-electron species.
        |
        v
    VMC_Workflow         optimise Jastrow u/chi + geminal (Slater) coeffs
        (pilot run, then MAX_CONTINUATION production runs of NUM_OPT_STEPS
        steps each; run 2 restarts from the run-1 checkpoint, so 2 x 100 =
        200 optimisation steps in total)
        opt_geminal=True
        opt_JDTN_u_param=True
        opt_JDTN_chi_param=True
        opt_JDTN_f_param=False       (u_chi preset has no f channel)
        opt_J1/J2/J3_param=False (Pade family disabled with DTN)
        |
        v
    MCMC_Workflow        VMC production with atomic_force=True (analytic derivatives);
                         pilot run, then production until TARGET_PROD_ERROR
        |
        v
    LRDMC_Workflow       LRDMC GFMC_t driver (a=0.30, tmove, atomic_force=True)
                         GFMC_t mode is selected automatically when neither
                         ``target_survived_walkers_ratio`` nor
                         ``num_projection_per_measurement`` is set.

Required files
--------------
    qe-pbc/h_scf.in            QE input (LDA-PZ, k=(1/4, 1/4, 1/4))
    qe-pbc/H.coulomb-ae.UPF    All-electron 1/r Coulomb pseudo for H, downloaded (see README.md)
    qe-pbc/out/h_k025.save/    QE output (prefix='h_k025'), NOT shipped: run
                               ``pw.x`` (built with HDF5) in ``qe-pbc/`` first
                               so that it writes ``wfc1.hdf5``. The workflow
                               only converts it; it does not run pw.x.

Usage
-----
    python run_pipelines.py

Note on machine configuration
-----------------------------
``SERVER`` and ``QUEUE_LABEL`` below are placeholders: set them to a machine
and queue label defined in ``jqmc_setting_local/`` (next to this script) or
``~/.jqmc_setting/``. The reference run used one whole 128-core node per job
(128 MPI ranks, 3 h wall-time limit); keep ``MAX_TIME`` (seconds) below the
queue wall-time limit so that jQMC stops and writes its checkpoint first.
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
import math
import os

from jqmc_workflow import (
    Container,
    FileFrom,
    ValueFrom,
    Launcher,
    LRDMC_Workflow,
    MCMC_Workflow,
    QE_Workflow,
    VMC_Workflow,
    parse_lrdmc_output,
    parse_mcmc_output,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Server / queue -- placeholders; adapt to the machine you have configured
# under ``jqmc_setting_local/`` or ``~/.jqmc_setting/``.
# docs: configuration-begin
SERVER = "cluster"
QUEUE_LABEL = "cores-128-mpi-128-omp-1-3h"
LABEL = "h-ae"

QE_SAVE_GLOB = "qe-pbc/**/*.save"

DT = 1.0
TARGET_PROD_ERROR = 5.0e-4
TARGET_VMC_ERROR = 5.0e-3
MAX_TIME = 10000
POLL_INTERVAL = 120
MAX_CONTINUATION = 2
NUM_WALKERS = 1

# VMC optimisation steps per production run (the VMC stage runs
# MAX_CONTINUATION production runs, i.e. 200 steps in total).
NUM_OPT_STEPS = 100
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps

# DTN Jastrow preset: u + chi only (no f).
JASTROW_DTN = "u_chi"

# LRDMC lattice spacing (Bohr) and non-local move scheme.
ALAT = 0.30
NON_LOCAL_MOVE = "tmove"


# docs: configuration-end


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_single_qe_save(base_dir: str) -> str:
    """Locate the (single) QE ``*.save`` directory under ``qe-pbc/``."""
    matches = sorted(glob.glob(os.path.join(base_dir, QE_SAVE_GLOB), recursive=True))
    if not matches:
        raise FileNotFoundError("No QE save directory found. Expected something like 'qe-pbc/out/h_k025.save'.")
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple QE save directories found. Keep one or set QE_SAVE_GLOB more specifically:\n" + "\n".join(matches)
        )
    return os.path.abspath(matches[0])


def format_val(val: float | None, err: float | None) -> str:
    """Format value(error) in compact statistical-physics notation."""
    if val is None or err is None:
        return "N/A"
    if err == 0:
        return f"{val:.5f}(0)"
    n_dec = max(0, -math.floor(math.log10(err)) + 1)
    err_in_last = round(err * 10**n_dec)
    return f"{val:.{n_dec}f}({err_in_last})"


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def build_workflows(base_dir: str) -> tuple[list[Container], dict[str, Container]]:
    """Build a 4-stage chain: QE ingest -> VMC opt -> MCMC -> LRDMC.

    Returns:
    -------
    workflows : list[Container]
        Flat list of Containers (for the Launcher).
    result_containers : dict[str, Container]
        Keyed by ``"mcmc"`` and ``"lrdmc"`` for summary parsing.
    """
    qe_save_dir = _resolve_single_qe_save(base_dir)

    # 1) QE ingest -> hamiltonian_data.h5 with the DTN(u, chi) ansatz.
    #    QE_Workflow runs ``jqmc-tool qe`` locally via subprocess (about
    #    half a minute here); the launcher does not need to stage the
    #    save dir. pw.x itself is NOT run by the workflow.
    wf = Container(
        label="wf",
        dirname="01_wf",
        workflow=QE_Workflow(
            qe_save_dir=qe_save_dir,
            kind="pgto+pw",
            hamiltonian_file="hamiltonian_data.h5",
            ecutwfc="auto",
            # Disable Pade J1/J2/J3 (mutually exclusive with DTN).
            j1_parameter=None,
            j2_parameter=None,
            j3_basis_type=None,
            jastrow_dtn=JASTROW_DTN,
            verbosity="high",
        ),
    )

    # 2) VMC optimise Jastrow(u, chi) + geminal coefficients.
    vmc = Container(
        label="vmc",
        dirname="02_vmc",
        input_files=[FileFrom("wf", "hamiltonian_data.h5")],
        workflow=VMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname=f"vmc-{LABEL}",
            wf_dump_freq=WF_DUMP_FREQ,
            Dt=DT,
            number_of_walkers=NUM_WALKERS,
            num_opt_steps=NUM_OPT_STEPS,
            num_mcmc_warmup_steps=30,
            num_mcmc_bin_blocks=10,
            pilot_mcmc_steps=50,
            pilot_vmc_steps=20,
            # Pade Jastrow channels are off (DTN is active).
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            # DTN channels: optimise u and chi only (the u_chi preset
            # has no f channel).
            opt_JDTN_u_param=True,
            opt_JDTN_chi_param=True,
            opt_JDTN_f_param=False,
            # Slater (geminal) coefficients on top of the Jastrow.
            opt_geminal=True,
            target_error=TARGET_VMC_ERROR,
            optimizer_kwargs={
                "method": "sr",
                "delta": 0.100,
                "epsilon": 0.001,
                "use_lm": True,
            },
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
        ),
    )

    # 3) MCMC production with atomic forces (analytic derivative backend).
    mcmc = Container(
        label="mcmc",
        dirname="03_mcmc",
        input_files=[
            FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")),
        ],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=MCMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname=f"mcmc-{LABEL}",
            Dt=DT,
            number_of_walkers=NUM_WALKERS,
            target_error=TARGET_PROD_ERROR,
            atomic_force=True,
            num_mcmc_warmup_steps=30,
            num_mcmc_bin_blocks=10,
            pilot_steps=100,
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
            verbosity="high",
        ),
    )

    # 4) LRDMC (GFMC_t) with atomic forces.
    #
    #    GFMC_t (the lrdmc-tau driver) is selected automatically when
    #    neither ``target_survived_walkers_ratio`` nor
    #    ``num_projection_per_measurement`` is set. The driver then uses
    #    ``time_projection_tau`` (default 0.10, imaginary time in Ha^-1).
    #
    #    non_local_move is "tmove" (the default) or "dltmove". The choice
    #    only affects non-local ECP terms, so for all-electron H it has no
    #    effect.
    lrdmc = Container(
        label="lrdmc",
        dirname="04_lrdmc",
        input_files=[
            FileFrom("vmc", ValueFrom("vmc", "optimized_hamiltonian")),
        ],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=LRDMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname=f"lrdmc-{LABEL}",
            alat=ALAT,
            non_local_move=NON_LOCAL_MOVE,
            number_of_walkers=NUM_WALKERS,
            target_error=TARGET_PROD_ERROR,
            atomic_force=True,
            num_gfmc_warmup_steps=30,
            num_gfmc_bin_blocks=10,
            num_gfmc_collect_steps=20,
            pilot_steps=100,
            max_time=MAX_TIME,
            poll_interval=POLL_INTERVAL,
            max_continuation=MAX_CONTINUATION,
            verbosity="high",
        ),
    )

    workflows = [wf, vmc, mcmc, lrdmc]
    result_containers = {"mcmc": mcmc, "lrdmc": lrdmc}
    return workflows, result_containers


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def print_summary_table(
    base_dir: str,
    result_containers: dict[str, Container],
) -> None:
    """Print energy + max |F| summary for the MCMC and LRDMC stages."""
    print()
    print("=" * 100)
    print("  Hydrogen Energy & Force Summary  (AE, packed-H2 ortho cell, k=(1/4, 1/4, 1/4))")
    print(f"  LRDMC a = {ALAT},  VMC opt steps = {NUM_OPT_STEPS} x {MAX_CONTINUATION} runs,  DTN preset = {JASTROW_DTN}")
    print("=" * 100)
    print()

    PATTERNS = [
        ("mcmc", "MCMC "),
        ("lrdmc", "LRDMC"),
    ]

    header = f"| {'Pattern':>10} | {'Energy (Ha)':>17} | {'Max|F| (Ha/bohr)':>17} | {'t_net (s)':>10} |"
    separator = f"|{'-' * 12}|{'-' * 19}|{'-' * 19}|{'-' * 12}|"
    print(header)
    print(separator)

    dir_map = {"mcmc": "03_mcmc", "lrdmc": "04_lrdmc"}
    parser_map = {"mcmc": parse_mcmc_output, "lrdmc": parse_lrdmc_output}

    for key, label in PATTERNS:
        ctr = result_containers.get(key)
        if ctr is not None and ctr.output_values:
            e = ctr.output_values.get("energy")
            e_err = ctr.output_values.get("energy_error")
            forces = ctr.output_values.get("forces")
        else:
            e, e_err, forces = None, None, None

        e_str = format_val(e, e_err)

        # Max |F| = max over atoms of sqrt(Fx^2 + Fy^2 + Fz^2).
        f_val, f_err = None, None
        if forces:
            max_norm = -1.0
            for f in forces:
                fx, fy, fz = f["Fx"], f["Fy"], f["Fz"]
                norm = math.sqrt(fx**2 + fy**2 + fz**2)
                if norm > max_norm:
                    max_norm = norm
                    if norm > 0:
                        fx_e, fy_e, fz_e = f["Fx_err"], f["Fy_err"], f["Fz_err"]
                        f_err = math.sqrt((fx * fx_e) ** 2 + (fy * fy_e) ** 2 + (fz * fz_e) ** 2) / norm
                    else:
                        f_err = 0.0
            f_val = max_norm

        f_str = format_val(f_val, f_err)

        work_dir = os.path.join(base_dir, dir_map[key])
        t_net = parser_map[key](work_dir).net_time_sec
        t_str = f"{t_net:.1f}" if t_net is not None else "N/A"

        print(f"| {label:>10} | {e_str:>17} | {f_str:>17} | {t_str:>10} |")

    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    workflows, result_containers = build_workflows(base_dir)
    launcher = Launcher(workflows=workflows, draw_graph=True)
    launcher.launch()

    print_summary_table(base_dir, result_containers)
