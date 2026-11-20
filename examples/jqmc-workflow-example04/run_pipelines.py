"""Diamond silicon: equation of state E(V) and an LRDMC lattice-spacing extrapolation.

Every step -- the QE SCF runs, the QE -> jQMC conversion, the Jastrow
optimisation and the QMC sampling -- is a node of a jqmc_workflow DAG, so the
whole study is restartable with one command.

System: diamond Si, 2-atom FCC primitive cell, ccECP (4 valence electrons per
atom), LDA-PZ determinant from QE, B-spline orbitals and the DTN ``u_chi_f``
Jastrow (the ansatz of ``jqmc-example07``), averaged over the complete shifted
2x2x2 twist grid (8 twists, ``twist_average=True``).

Stage 1 -- volume scan, one independent chain per lattice constant::

    PWscf_Workflow   pw.x SCF through the queue (custom node, defined below)
          |  si.save/
          v
    QE_Workflow      jqmc-tool qe --kind bspline --twist-average --jastrow-dtn u_chi_f
          |  hamiltonian_data.h5
          v
    VMC_Workflow     optimise the DTN Jastrow (B-spline determinant frozen)
          |  hamiltonian_data_opt_averaged.h5 (average of the last 20 dumps)
          +--> MCMC_Workflow    VMC energy
          +--> LRDMC_Workflow   LRDMC energy at alat = LRDMC_ALAT

    -> Birch-Murnaghan fits of E_DFT(V), E_VMC(V) and E_LRDMC(V): a0 and B0.

Stage 2 -- at the scanned lattice constant closest to the LRDMC a0, two more
LRDMC_Workflow runs (EXT_ALATS) with the same trial; together with the stage-1
run they are extrapolated to alat -> 0 by ``jqmc-tool lrdmc extrapolate-energy``.

Usage
-----
    python run_pipelines.py     # runs both stages; rerun the same command to resume

Machine configuration
---------------------
``SERVER`` and the queue labels are placeholders: set them to a machine and
queue labels defined in ``jqmc_setting_local/`` (next to this script) or in
``~/.jqmc_setting/``. The queue template must put ``pw.x`` (built with HDF5)
on ``PATH`` and launch a multiple of 8 MPI ranks (one rank group per twist).
The reference run used one 128-core node per job (128 MPI ranks).
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

import math
import os
import re
import subprocess
import sys

import numpy as np

from jqmc_workflow import (
    Container,
    FileFrom,
    Launcher,
    LRDMC_Workflow,
    MCMC_Workflow,
    QE_Workflow,
    ValueFrom,
    VMC_Workflow,
    Workflow,
    WorkflowStatus,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# docs: configuration-begin
SERVER = "cluster"
QUEUE_LABEL = "cores-128-mpi-128-omp-1-3h"  # VMC / MCMC / LRDMC jobs
PW_QUEUE_LABEL = "cores-128-mpi-128-omp-1-1h"  # pw.x SCF jobs (seconds each)
POLL_INTERVAL = 60

# Cubic lattice constants of the scan (Angstrom).
LATTICE_CONSTANTS = [5.25, 5.35, 5.45, 5.55, 5.65]

# QE (pw.x) settings, as in jqmc-example07 / jqmc-example09.
UPF_FILE = "../jqmc-example07/01DFT/Si.ccECP.upf"
ECUTWFC_RY = 100.0
TWIST_GRID = 2  # complete shifted 2x2x2 Monkhorst-Pack grid -> 8 twists

# Trial wavefunction: B-spline determinant (frozen) + DTN Jastrow.
JASTROW_DTN = "u_chi_f"

# VMC optimisation (fixed-step mode: no pilot run).
NUM_OPT_STEPS = 80
VMC_MCMC_STEPS = 500  # MCMC measurements per optimisation step
WF_DUMP_FREQ = 1  # dump every step: VMC_Workflow hands on the average of the last 20 steps

# Sampling (fixed-step mode: no pilot run).
NUMBER_OF_WALKERS = 4  # per MPI rank
DT = 1.5
MCMC_STEPS = 20000
LRDMC_ALAT = 0.30  # bohr; lattice spacing of the volume scan
LRDMC_STEPS = 20000  # GFMC branching steps per LRDMC run

# Stage 2: extra lattice spacings for the alat -> 0 extrapolation.
EXT_ALATS = [0.20, 0.40]

MAX_TIME = 10000  # seconds; below the 3 h queue limit
# docs: configuration-end

PREFIX = "si"
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
HA_PER_BOHR3_TO_GPA = 29421.015697


def tag(a: float) -> str:
    """Directory/label prefix of a lattice constant, e.g. 5.45 -> 'a5.450'."""
    return f"a{a:.3f}"


# -----------------------------------------------------------------------------
# A custom workflow node: pw.x SCF through the machine's queue template
# -----------------------------------------------------------------------------
class PWscf_Workflow(Workflow):
    """Write ``pw.in`` for diamond Si at lattice constant ``a`` and run ``pw.x``.

    jqmc_workflow has no pw.x node and no public call that submits an
    arbitrary command, so this subclass of the public :class:`Workflow` base
    class reuses the internal job helper of the built-in workflows
    (``_submit_and_wait``): the queue template's ``_COMMAND_`` is
    ``pw.x -nk <n_twist> -in pw.in``, launched by the template's ``mpirun``.
    Downstream nodes receive the ``si.save`` directory through ``FileFrom``.
    """

    def __init__(self, server_machine_name, queue_label, jobname, lattice_constant, poll_interval=60):
        super().__init__()
        self.server_machine_name = server_machine_name
        self.queue_label = queue_label
        self.jobname = jobname
        self.lattice_constant = lattice_constant
        self.poll_interval = poll_interval
        # ``_submit_and_wait`` uploads ``hamiltonian_file`` to a remote
        # machine when it exists; for pw.x that extra input is the UPF.
        self.hamiltonian_file = os.path.basename(UPF_FILE)

    def _pw_input(self) -> str:
        h = self.lattice_constant / 2.0
        n = TWIST_GRID
        return f"""&CONTROL
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = './'
    pseudo_dir  = './'
/
&SYSTEM
    ibrav       = 0
    nat         = 2
    ntyp        = 1
    nbnd        = 8
    ecutwfc     = {ECUTWFC_RY}
    ecutrho     = {4 * ECUTWFC_RY}
    occupations = 'fixed'
    input_dft   = 'pz'
    nosym       = .true.
    noinv       = .true.
/
&ELECTRONS
    conv_thr    = 1.0d-9
    mixing_beta = 0.5
/
ATOMIC_SPECIES
    Si  28.085  {os.path.basename(UPF_FILE)}
CELL_PARAMETERS {{angstrom}}
    0.000000  {h:.6f}  {h:.6f}
    {h:.6f}  0.000000  {h:.6f}
    {h:.6f}  {h:.6f}  0.000000
ATOMIC_POSITIONS {{crystal}}
    Si  0.00  0.00  0.00
    Si  0.25  0.25  0.25
K_POINTS automatic
    {n} {n} {n} 1 1 1
"""

    def _get_command(self, input_file: str) -> str:
        return f"pw.x -nk {TWIST_GRID**3} -in {input_file}"

    def configure(self) -> dict:
        return {"lattice_constant": self.lattice_constant, "twist_grid": TWIST_GRID}

    async def run(self):
        self._ensure_project_dir()
        wd = self.project_dir
        with open(os.path.join(wd, "pw.in"), "w") as f:
            f.write(self._pw_input())
        await self._submit_and_wait(
            "pw.in",
            "pw.out",
            wd,
            fetch_from_objects=["pw.out", f"{PREFIX}.save"],
            step=1,
            run_id="pw",
        )
        with open(os.path.join(wd, "pw.out")) as f:
            text = f.read()
        energies = re.findall(r"^!\s+total energy\s+=\s+(\S+) Ry", text, re.M)
        if "JOB DONE" not in text or not energies:
            self.status = WorkflowStatus.FAILED
            return self.status, [], {"error": "pw.x did not finish; see pw.out"}
        # The Container's completion check looks for jQMC's last log line,
        # "Program ends", in every job output; pw.x ends with "JOB DONE."
        # instead, so record the verified normal termination in that form.
        if "Program ends" not in text:
            with open(os.path.join(wd, "pw.out"), "a") as f:
                f.write("Program ends (pw.x JOB DONE verified by PWscf_Workflow)\n")
        self.status = WorkflowStatus.COMPLETED
        self.output_files = ["pw.out", f"{PREFIX}.save"]
        self.output_values = {"dft_energy_ry": float(energies[-1])}
        return self.status, self.output_files, self.output_values


# -----------------------------------------------------------------------------
# Stage 1: QE -> conversion -> VMC -> MCMC / LRDMC for every lattice constant
# -----------------------------------------------------------------------------
def lrdmc_container(t: str, alat: float, dirname: str, trial) -> Container:
    """LRDMC (GFMC_t, lrdmc-tau) at one lattice spacing with a fixed number of branchings."""
    return Container(
        label=f"lrdmc-{t}-alat{alat:.2f}",
        dirname=dirname,
        input_files=[trial],
        rename_input_files=["hamiltonian_data.h5"],
        workflow=LRDMC_Workflow(
            server_machine_name=SERVER,
            queue_label=QUEUE_LABEL,
            jobname=f"lrdmc-{t}-{alat:.2f}",
            number_of_walkers=NUMBER_OF_WALKERS,
            max_time=MAX_TIME,
            alat=alat,
            non_local_move="tmove",
            num_gfmc_projections=LRDMC_STEPS,
            num_gfmc_warmup_steps=50,
            num_gfmc_bin_blocks=50,
            num_gfmc_collect_steps=20,
            verbosity="high",
            poll_interval=POLL_INTERVAL,
        ),
    )


def build_scan() -> tuple[list[Container], dict]:
    workflows: list[Container] = []
    nodes: dict = {}
    for a in LATTICE_CONSTANTS:
        t = tag(a)
        pw = Container(
            label=f"pw-{t}",
            dirname=f"{t}/01_pw",
            input_files=[UPF_FILE],
            workflow=PWscf_Workflow(SERVER, PW_QUEUE_LABEL, f"pw-{t}", a, poll_interval=POLL_INTERVAL),
        )
        conv = Container(
            label=f"conv-{t}",
            dirname=f"{t}/02_conv",
            input_files=[FileFrom(f"pw-{t}", f"{PREFIX}.save")],
            workflow=QE_Workflow(
                qe_save_dir=f"{PREFIX}.save",
                kind="bspline",
                twist_average=True,
                jastrow_dtn=JASTROW_DTN,
                hamiltonian_file="hamiltonian_data.h5",
            ),
        )
        vmc = Container(
            label=f"vmc-{t}",
            dirname=f"{t}/03_vmc",
            input_files=[FileFrom(f"conv-{t}", "hamiltonian_data.h5")],
            workflow=VMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"vmc-{t}",
                number_of_walkers=NUMBER_OF_WALKERS,
                max_time=MAX_TIME,
                Dt=DT,
                num_opt_steps=NUM_OPT_STEPS,
                num_mcmc_steps=VMC_MCMC_STEPS,
                num_mcmc_per_measurement=40,
                num_mcmc_warmup_steps=30,
                num_mcmc_bin_blocks=10,
                wf_dump_freq=WF_DUMP_FREQ,
                # DTN Jastrow only; the B-spline determinant stays frozen.
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_JDTN_u_param=True,
                opt_JDTN_chi_param=True,
                opt_JDTN_f_param=True,
                opt_geminal=False,
                optimizer_kwargs={"method": "sr", "delta": 0.15, "epsilon": 0.001, "use_lm": True},
                verbosity="high",
                poll_interval=POLL_INTERVAL,
            ),
        )
        trial = FileFrom(f"vmc-{t}", ValueFrom(f"vmc-{t}", "optimized_hamiltonian"))
        mcmc = Container(
            label=f"mcmc-{t}",
            dirname=f"{t}/04_mcmc",
            input_files=[trial],
            rename_input_files=["hamiltonian_data.h5"],
            workflow=MCMC_Workflow(
                server_machine_name=SERVER,
                queue_label=QUEUE_LABEL,
                jobname=f"mcmc-{t}",
                number_of_walkers=NUMBER_OF_WALKERS,
                max_time=MAX_TIME,
                Dt=DT,
                num_mcmc_steps=MCMC_STEPS,
                num_mcmc_per_measurement=40,
                num_mcmc_warmup_steps=30,
                num_mcmc_bin_blocks=10,
                verbosity="high",
                poll_interval=POLL_INTERVAL,
            ),
        )
        lrdmc = lrdmc_container(t, LRDMC_ALAT, f"{t}/05_lrdmc", trial)
        workflows += [pw, conv, vmc, mcmc, lrdmc]
        nodes[a] = {"pw": pw, "vmc": vmc, "mcmc": mcmc, "lrdmc": lrdmc}
    return workflows, nodes


# -----------------------------------------------------------------------------
# Equation of state: Birch-Murnaghan fit, parametric-bootstrap errors
# -----------------------------------------------------------------------------
def birch_murnaghan(v, e0, v0, b0, b0p=4.0):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * ((x - 1.0) ** 3 * b0p + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


def fit_eos(volumes, energies, errors=None, n_boot=2000, seed=0):
    """Fit E(V) with B0' = 4 fixed. Returns (V0, dV0, B0, dB0, chi2/dof); errors None -> no bootstrap."""
    from scipy.optimize import curve_fit

    v, e = np.asarray(volumes), np.asarray(energies)
    p0 = [e.min(), v[np.argmin(e)], 0.003]
    if errors is None:
        popt, _ = curve_fit(birch_murnaghan, v, e, p0=p0)
        return popt[1], None, popt[2], None, None
    s = np.asarray(errors)
    popt, _ = curve_fit(birch_murnaghan, v, e, p0=p0, sigma=s, absolute_sigma=True, maxfev=20000)
    chi2 = float(np.sum(((birch_murnaghan(v, *popt) - e) / s) ** 2) / (len(v) - 3))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        try:
            p_b, _ = curve_fit(birch_murnaghan, v, e + rng.normal(0.0, s), p0=popt, sigma=s, maxfev=20000)
            boot.append(p_b)
        except RuntimeError:
            continue
    boot = np.array(boot)
    return popt[1], boot[:, 1].std(), popt[2], boot[:, 2].std(), chi2


def fmt(val, err):
    """value(error) with two digits of the error, e.g. -7.93121(23)."""
    if val is None or not err:
        return "N/A"
    n_dec = max(0, -math.floor(math.log10(err)) + 1)
    return f"{val:.{n_dec}f}({round(err * 10**n_dec)})"


def summarize_scan(nodes: dict) -> dict:
    print()
    print("=" * 80)
    print("  Si E(V): 2-atom cell, 8 twists, B-spline + DTN u_chi_f (energies per cell)")
    print("=" * 80)
    print(f"| {'a (A)':>6} | {'V (bohr^3)':>10} | {'E_DFT (Ha)':>11} | {'E_VMC (Ha)':>14} | {'E_LRDMC (Ha)':>14} |")
    print(f"|{'-' * 8}|{'-' * 12}|{'-' * 13}|{'-' * 16}|{'-' * 16}|")
    data = {"dft": [], "mcmc": [], "lrdmc": []}
    for a in LATTICE_CONSTANTS:
        v = (a * ANGSTROM_TO_BOHR) ** 3 / 4.0
        e_ry = nodes[a]["pw"].output_values.get("dft_energy_ry")
        e_dft = e_ry / 2.0 if e_ry is not None else None
        if e_dft is not None:
            data["dft"].append((v, e_dft, None))
        cells = []
        for key in ("mcmc", "lrdmc"):
            ov = nodes[a][key].output_values or {}
            e, de = ov.get("energy"), ov.get("energy_error")
            cells.append(fmt(e, de))
            if e is not None and de:
                data[key].append((v, e, de))
        e_dft_s = f"{e_dft:.5f}" if e_dft is not None else "N/A"
        print(f"| {a:6.3f} | {v:10.3f} | {e_dft_s:>11} | {cells[0]:>14} | {cells[1]:>14} |")
    print()
    print("  Birch-Murnaghan fits (B0' = 4 fixed; QMC errors from a parametric bootstrap):")
    fits = {}
    for key, name in (("dft", "DFT (LDA)"), ("mcmc", "VMC"), ("lrdmc", f"LRDMC a={LRDMC_ALAT}")):
        if len(data[key]) < 4:
            print(f"  {name:>14}: not enough points for a fit.")
            continue
        v, e, de = zip(*data[key])
        v0, dv0, b0, db0, chi2 = fit_eos(v, e, None if key == "dft" else de)
        a0 = (4.0 * v0) ** (1.0 / 3.0) / ANGSTROM_TO_BOHR
        fits[key] = a0
        if key == "dft":
            print(f"  {name:>14}: a0 = {a0:.3f} A,  B0 = {b0 * HA_PER_BOHR3_TO_GPA:.1f} GPa")
        else:
            da0 = a0 * dv0 / (3.0 * v0)
            b0_s = fmt(b0 * HA_PER_BOHR3_TO_GPA, db0 * HA_PER_BOHR3_TO_GPA)
            print(f"  {name:>14}: a0 = {fmt(a0, da0)} A,  B0 = {b0_s} GPa,  chi2/dof = {chi2:.2f}")
    print()
    return fits


# -----------------------------------------------------------------------------
# Stage 2: alat -> 0 at the scanned lattice constant closest to the LRDMC a0
# -----------------------------------------------------------------------------
def run_extrapolation(nodes: dict, a_ref: float) -> None:
    t = tag(a_ref)
    trial = os.path.join(t, "03_vmc", nodes[a_ref]["vmc"].output_values["optimized_hamiltonian"])
    extra = [lrdmc_container(t, alat, f"{t}/06_lrdmc_alat{alat:.2f}", trial) for alat in EXT_ALATS]
    Launcher(workflows=extra, log_name="jqmc_workflow_ext.log").launch()

    # The twist-averaged checkpoints are directories (restart.h5/twist_*.h5);
    # jqmc-tool forms the BZ average at every alat and fits E against alat^2.
    runs = {LRDMC_ALAT: nodes[a_ref]["lrdmc"]} | {c.workflow.alat: c for c in extra}
    chks = [os.path.relpath(os.path.join(c.project_dir, c.output_values["restart_chk"])) for _, c in sorted(runs.items())]
    cmd = ["jqmc-tool", "lrdmc", "extrapolate-energy", *chks, "-p", "1", "-b", "50", "-w", "50", "-c", "20"]
    cmd += ["-s", "lrdmc_extrapolation.png"]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("\n".join(line for line in result.stdout.splitlines() if re.search(r"For a|Polynomial|Graph", line)))


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    scan, nodes = build_scan()
    Launcher(workflows=scan, log_name="jqmc_workflow_scan.log").launch()
    fits = summarize_scan(nodes)

    a_fit = fits.get("lrdmc")
    if a_fit is None:
        sys.exit("No LRDMC E(V) fit; inspect the a*/ directories and rerun to resume.")
    a_ref = min(LATTICE_CONSTANTS, key=lambda a: abs(a - a_fit))
    print(f"Stage 2: LRDMC alat -> 0 at a = {a_ref:.3f} A (scan point closest to the LRDMC a0 = {a_fit:.3f} A)")
    run_extrapolation(nodes, a_ref)
