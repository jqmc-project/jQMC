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

import logging
import os
import subprocess
import sys

import jax

# Enable 64-bit floats right after importing JAX, before jQMC is imported and
# before any JAX array or distributed runtime is created, so that every array
# jQMC builds is float64.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from jqmc import (  # noqa: E402
    GFMC_t,
    Hamiltonian_data,
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    MCMC,
    Wavefunction_data,
    read_trexio_file,
)

# MPI Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("jqmc")
if rank == 0:
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.ERROR)  # Suppress logs from other ranks

# JAX configuration. The drivers use JAX collectives across MPI ranks, so the
# JAX distributed runtime must know about the MPI communicator before any
# driver is constructed (the ``jqmc`` command does this automatically).
# HTTP(S) proxy variables can make the gRPC connection of the JAX coordination
# service hang (the ``jqmc`` command removes them too).
for _proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_proxy_var, None)
jax.distributed.initialize(cluster_detection_method="mpi4py")

# =============================================================================
# 0. Settings
# =============================================================================
trexio_file = "water_ccecp_ccpvtz.h5"
hamiltonian_file = "hamiltonian_data.h5"

# The total number of walkers is the same as in the serial jqmc-example12;
# it is split evenly over the MPI ranks.
num_walkers_total = 2000
if num_walkers_total % size != 0:
    if rank == 0:
        logger.error(f"num_walkers_total = {num_walkers_total} is not divisible by {size} MPI processes.")
    sys.exit(1)
num_walkers_per_rank = num_walkers_total // size

if rank == 0:
    logger.info(f"Running on {size} MPI processes, {num_walkers_per_rank} walkers per process.")

if not os.path.exists(trexio_file):
    if rank == 0:
        logger.error(f"{trexio_file} not found. Please run run_pyscf.py first.")
    sys.exit(1)

# =============================================================================
# 1. Convert TREXIO to Hamiltonian
# =============================================================================
if rank == 0:
    logger.info(f"Converting {trexio_file} to {hamiltonian_file}...")

# All ranks read TREXIO (it's read-only)
(structure_data, aos_data, mos_data, _, geminal_data, coulomb_potential_data) = read_trexio_file(trexio_file, store_tuple=True)

# Jastrow factor J2 + J3, as `jqmc-tool trexio convert-to ... -j2 1.0 -j3 mo`.
# No one-body Jastrow: the J1 form imposes the electron-nucleus cusp with the
# effective charge, which the smooth ccECP pseudopotentials do not need.
# Jastrow 2-body (Pade, initial parameter 1.0)
jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)

# Jastrow 3-body in the occupied-MO basis (J matrix initialized to zero)
jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=mos_data)

jastrow_data = Jastrow_data(
    jastrow_one_body_data=None,
    jastrow_two_body_data=jastrow_twobody_data,
    jastrow_three_body_data=jastrow_threebody_data,
)

wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

hamiltonian_data = Hamiltonian_data(
    structure_data=structure_data,
    coulomb_potential_data=coulomb_potential_data,
    wavefunction_data=wavefunction_data,
)

if rank == 0:
    hamiltonian_data.save_to_hdf5(hamiltonian_file)
    logger.info(f"Saved Hamiltonian data to {hamiltonian_file}")


# =============================================================================
# 2. VMC Optimization
# =============================================================================
if rank == 0:
    logger.info("Starting VMC Optimization...")

mcmc_opt = MCMC(
    hamiltonian_data=hamiltonian_data,
    Dt=2.0,
    mcmc_seed=42,
    num_walkers=num_walkers_per_rank,
    num_mcmc_per_measurement=40,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_log_WF_param_deriv=True,  # needed by the SR optimizer
    comput_e_L_param_deriv=True,  # needed by use_lm=True
)

mcmc_opt.run_optimize(
    num_mcmc_steps=200,  # Measurements per optimization step
    num_opt_steps=50,
    wf_dump_freq=1,  # dump every step: the last 20 are averaged below
    num_mcmc_warmup_steps=30,
    num_mcmc_bin_blocks=10,
    opt_J1_param=False,
    opt_J2_param=True,
    opt_J3_param=True,
    opt_geminal=False,
    max_time=3600,
    optimizer_kwargs={"method": "sr", "delta": 0.15, "epsilon": 0.001, "use_lm": True},
)

# Hand the sampling runs the uniform average of the last 20 optimization steps,
# as the command-line tutorials do (copy_optimized_hamiltonian.py runs
# `jqmc-tool vmc average-wf`); mcmc_opt.hamiltonian_data is only the last step.
# Rank 0 wrote the dumps, so rank 0 averages them in a separate non-MPI process
# (the MPI launcher variables are removed from its environment); then every
# rank reads the result.
num_averaged_steps = 20
averaged_file = "hamiltonian_data_averaged.h5"
status = 0
if rank == 0:
    if os.path.exists(averaged_file):
        os.remove(averaged_file)  # left over from a previous run of this script
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "copy_optimized_hamiltonian.py")
    launcher = ("PMI_", "PMIX_", "HYDRA_", "OMPI_", "MPI_LOCAL")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(launcher)}
    command = [sys.executable, helper, "--average-last", str(num_averaged_steps), ".", averaged_file]
    status = subprocess.run(command, env=environment).returncode
if comm.bcast(status, root=0) != 0:
    sys.exit(1)
optimized_hamiltonian = Hamiltonian_data.load_from_hdf5(averaged_file)


# =============================================================================
# 3. VMC Sampling
# =============================================================================
if rank == 0:
    logger.info("Starting VMC Sampling...")

mcmc_prod = MCMC(
    hamiltonian_data=optimized_hamiltonian,
    Dt=2.0,
    mcmc_seed=123,
    num_walkers=num_walkers_per_rank,
    num_mcmc_per_measurement=40,
    epsilon_AS=0.0,
    comput_position_deriv=False,
    comput_log_WF_param_deriv=False,
)

num_vmc_steps = 1000
mcmc_prod.run(num_mcmc_steps=num_vmc_steps, max_time=3600)

# get_E performs MPI reduction internally
vmc_E_mean, vmc_E_std, _, _ = mcmc_prod.get_E(
    num_mcmc_warmup_steps=int(num_vmc_steps * 0.1),
    num_mcmc_bin_blocks=10,
)

if rank == 0:
    logger.info(f"VMC Energy: {vmc_E_mean:.5f} +/- {vmc_E_std:.5f} Ha")


# =============================================================================
# 4. LRDMC with multiple alats
# =============================================================================
alats = [0.3, 0.2, 0.1]
lrdmc_results = []

for alat in alats:
    if rank == 0:
        logger.info(f"Starting LRDMC with alat={alat}...")

    lrdmc = GFMC_t(
        hamiltonian_data=optimized_hamiltonian,
        num_walkers=num_walkers_per_rank,
        tau=0.10,  # imaginary time between two branchings (1/Ha)
        num_gfmc_collect_steps=20,
        mcmc_seed=456,
        alat=alat,
        non_local_move="tmove",
        comput_position_deriv=False,
    )

    num_lrdmc_steps = 3000
    lrdmc.run(num_mcmc_steps=num_lrdmc_steps, max_time=3600)

    # get_E performs MPI reduction internally
    lrdmc_E_mean, lrdmc_E_std, _, _ = lrdmc.get_E(
        num_mcmc_warmup_steps=int(num_lrdmc_steps * 0.1),
        num_mcmc_bin_blocks=10,
    )

    if rank == 0:
        logger.info(f"LRDMC Energy (alat={alat}): {lrdmc_E_mean:.5f} +/- {lrdmc_E_std:.5f} Ha")
    lrdmc_results.append((alat, lrdmc_E_mean, lrdmc_E_std))


# =============================================================================
# 5. Extrapolation (Only Rank 0)
# =============================================================================
if rank == 0:
    x = np.array([r[0] ** 2 for r in lrdmc_results])
    y = np.array([r[1] for r in lrdmc_results])
    w = np.array([1.0 / r[2] ** 2 for r in lrdmc_results])

    A = np.vstack([x, np.ones(len(x))]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    c1, c0 = cov @ (A.T @ W @ y)
    # Standard error of the intercept from the weighted least-squares covariance.
    c0_err = np.sqrt(cov[1, 1])
    residuals = y - (c0 + c1 * x)
    chi2 = float(np.sum(w * residuals**2))
    dof = len(x) - 2  # data points minus fitted parameters

    logger.info("-" * 40)
    logger.info("Extrapolation Results (E vs a^2):")
    for a, E, err in lrdmc_results:
        logger.info(f"alat={a:.2f}, a^2={a**2:.4f}, E={E:.5f} +/- {err:.5f}")

    logger.info(f"Extrapolated Energy (a->0): {c0:.5f} +/- {c0_err:.5f} Ha")
    logger.info(f"Fit quality: chi^2 = {chi2:.2f} for {dof} degree(s) of freedom")
    logger.info("-" * 40)
