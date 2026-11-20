"""jQMC miscs."""

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

from ._setting import MPC_CUTOFF_HA, OPTIMIZER_DEFAULT_CG_MAX_ITER, OPTIMIZER_DEFAULT_CG_TOL

# Keys that a generated input file must not carry. The generated
# TOML doubles as the CONTINUATION TOML -- the documented journey edits
# ``restart = true`` in the very same file -- so any key that is BOTH "frozen on
# restart" AND evolves during the run would be emitted with a placeholder
# default, disagree with the checkpoint's evolved value, and make the frozen-key
# guard refuse the restart. ``E_scf`` is the only such key: every other frozen
# key still equals what the TOML said when the run finished. Omitting the key
# avoids emitting a value the user never chose and then refusing it.
GENERATED_INPUT_OMITTED_KEYS = {
    "lrdmc-bra": ("E_scf",),
}


cli_parameters = {
    "control": {
        "job_type": None,
        "mcmc_seed": 34456,
        "number_of_walkers": 4,
        "max_time": 86400,
        "restart": False,
        "restart_chk": "restart.h5",
        "hamiltonian_h5": "hamiltonian_data.h5",
        "verbosity": "low",
    },
    "control_comments": {
        "job_type": 'Specify the job type. "mcmc", "vmc", "lrdmc-bra", or "lrdmc-tau".',
        "mcmc_seed": "Random seed for MCMC",
        "number_of_walkers": "Number of walkers per MPI process",
        "max_time": "Maximum time in sec.",
        "restart": "Restart calculation. A checkpoint file should be specified.",
        "restart_chk": (
            "Restart checkpoint file. If restart is True, this file is used. "
            "For twist-averaged runs (when the input has a combined /twists group "
            "and the driver dispatches to the twist-averaged path) this is "
            "the checkpoint DIRECTORY (twist_NNN.h5 + shared_state.h5 layout)."
        ),
        "hamiltonian_h5": "Hamiltonian checkpoint file. If restart is False, this file is used.",
        "verbosity": 'Verbosity level. "low", "high", "devel", "mpi-low", "mpi-high", "mpi-devel"',
    },
    # Restart-inheritance contract. On restart the driver
    # rebuilds from the checkpoint (MCMC.load_from_hdf5), so some TOML keys are
    # silently ignored. This table makes the contract explicit and is consumed
    # by (a) _dump_with_comments -> "[frozen/mutable on restart]" labels and
    # (b) the CLI restart guard, which raises ValueError when an *explicitly
    # set* frozen key conflicts with the checkpoint. "frozen": restored from the
    # checkpoint (TOML ignored). "mutable": re-read from the TOML every run.
    # A drift test cross-checks the frozen set against the checkpoint
    # driver_config keys via TOML_TO_CFG (see jqmc_cli).
    "control_restart_policy": {
        "job_type": "mutable",
        "mcmc_seed": "frozen",
        "number_of_walkers": "frozen",
        "max_time": "mutable",
        "restart": "mutable",
        "restart_chk": "mutable",
        "hamiltonian_h5": "frozen",
        "verbosity": "mutable",
    },
    "mcmc": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 5,
        "Dt": 2.0,
        "epsilon_AS": 0.0,
        "atomic_force": False,
        "compute_stress": False,
        "use_swct": True,
        "parameter_derivatives": False,
    },
    "mcmc_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": "Number of MCMC updates between successive observable measurements.",
        "num_mcmc_warmup_steps": "Number of initial observable measurements discarded as warmup.",
        "num_mcmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
        "Dt": "Step size for the MCMC update (bohr).",
        "epsilon_AS": "Epsilon parameter for Attaccalite-Sorella regularization.",
        "atomic_force": "If true, compute atomic forces.",
        "compute_stress": (
            "If true, compute the symmetric stress tensor sigma (Ha/bohr^3) and the diagonal pressure "
            "P = -diag(sigma). Periodic systems only; ignored for molecular runs. Requires the "
            "electron-position derivatives, which the CLI enables automatically when either "
            "atomic_force or compute_stress is true. Default: false."
        ),
        "use_swct": "If true, apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is True for MCMC.",
        "parameter_derivatives": "If true, compute parameter derivatives.",
    },
    # Restart-inheritance contract for the [mcmc] section
    # (see the control_restart_policy comment above). frozen keys are restored
    # from MCMC.save_to_hdf5's driver_config; mutable keys are re-read every run
    # (num_mcmc_steps/warmup/bin_blocks feed run()/get_E()).
    "mcmc_restart_policy": {
        "num_mcmc_steps": "mutable",
        "num_mcmc_per_measurement": "frozen",
        "num_mcmc_warmup_steps": "mutable",
        "num_mcmc_bin_blocks": "mutable",
        "Dt": "frozen",
        "epsilon_AS": "frozen",
        "atomic_force": "frozen",
        "compute_stress": "frozen",
        "use_swct": "frozen",
        "parameter_derivatives": "frozen",
    },
    # VMC optimization and both LRDMC drivers also restore checkpoint physics
    # (alat, non_local_move, tau, and collection steps) while ignoring those
    # continuation-TOML values, so every key needs an explicit policy.
    "vmc_restart_policy": {
        "num_mcmc_steps": "mutable",
        "num_mcmc_per_measurement": "frozen",
        "num_mcmc_warmup_steps": "mutable",
        "num_mcmc_bin_blocks": "mutable",
        "Dt": "frozen",
        "epsilon_AS": "frozen",
        "num_opt_steps": "mutable",
        "wf_dump_freq": "mutable",
        "opt_J1_param": "mutable",
        "opt_J2_param": "mutable",
        "opt_J3_param": "mutable",
        "opt_geminal": "mutable",
        "fix_geminal_rank": "mutable",
        "opt_J3_basis_exp": "mutable",
        "opt_J3_basis_coeff": "mutable",
        "opt_lambda_basis_exp": "mutable",
        "opt_lambda_basis_coeff": "mutable",
        "opt_JDTN_u_param": "mutable",
        "opt_JDTN_chi_param": "mutable",
        "opt_JDTN_f_param": "mutable",
        "optimizer_kwargs": "mutable",
    },
    "lrdmc-bra_restart_policy": {
        "num_mcmc_steps": "mutable",
        "num_mcmc_per_measurement": "frozen",
        "alat": "frozen",
        "non_local_move": "frozen",
        "num_gfmc_warmup_steps": "mutable",
        "num_gfmc_bin_blocks": "mutable",
        "num_gfmc_collect_steps": "frozen",
        "E_scf": "frozen",
        "atomic_force": "frozen",
        "compute_stress": "frozen",
        "use_swct": "frozen",
        "epsilon_PW": "frozen",
    },
    "lrdmc-tau_restart_policy": {
        "num_mcmc_steps": "mutable",
        "tau": "frozen",
        "alat": "frozen",
        "non_local_move": "frozen",
        "num_gfmc_warmup_steps": "mutable",
        "num_gfmc_bin_blocks": "mutable",
        "num_gfmc_collect_steps": "frozen",
        "atomic_force": "frozen",
        "compute_stress": "frozen",
        "use_swct": "frozen",
        "epsilon_PW": "frozen",
    },
    "vmc": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": 40,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 5,
        "Dt": 2.0,
        "epsilon_AS": 0.0,
        "num_opt_steps": None,
        "wf_dump_freq": 1,
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_geminal": False,
        "fix_geminal_rank": 0,
        "opt_J3_basis_exp": False,
        "opt_J3_basis_coeff": False,
        "opt_lambda_basis_exp": False,
        "opt_lambda_basis_coeff": False,
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": True,
        "optimizer_kwargs": {
            "method": "sr",
            "delta": 0.15,
            "epsilon": 0.001,
            "cg_flag": True,
            "cg_max_iter": OPTIMIZER_DEFAULT_CG_MAX_ITER,
            "cg_tol": OPTIMIZER_DEFAULT_CG_TOL,
            "use_lm": True,
            "lm_subspace_dim": 0,
            "lm_cond": 0.001,
        },
    },
    "vmc_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": "Number of MCMC updates between successive observable measurements.",
        "num_mcmc_warmup_steps": "Number of initial observable measurements discarded as warmup.",
        "num_mcmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_mcmc_bin_blocks * mpi_size * number_of_walkers.",
        "Dt": "Step size for the MCMC update (bohr).",
        "epsilon_AS": "Epsilon parameter for Attaccalite-Sorella regularization.",
        "num_opt_steps": "Number of optimization steps.",
        "wf_dump_freq": "Frequency of wavefunction (i.e. hamiltonian_data) dump.",
        "opt_J1_param": "Optimize the J1 parameter",
        "opt_J2_param": "Optimize the J2 parameter.",
        "opt_J3_param": "Optimize the J3 parameters.",
        "opt_geminal": (
            "If true, optimize the geminal: the pairing matrix for AGP, the MO coefficients and the "
            "diagonal pairing weights for AGPn, the MO coefficients only for SD."
        ),
        "fix_geminal_rank": (
            "Rank control of the geminal optimization. 0 (default) means 'not set': full-rank "
            "optimization of the input geminal. A positive integer converts an AGP geminal to AGPn "
            "of that rank at the start of the optimization; AGPn / SD inputs have a fixed rank, so "
            "a mismatch is warned (AGPn) or raised (SD). Default: 0."
        ),
        "opt_J3_basis_exp": "Optimize the AO basis exponents in the three-body Jastrow factor.",
        "opt_J3_basis_coeff": "Optimize the AO basis contraction coefficients in the three-body Jastrow factor.",
        "opt_lambda_basis_exp": "Optimize the AO basis exponents in the geminal (lambda) determinant.",
        "opt_lambda_basis_coeff": "Optimize the AO basis contraction coefficients in the geminal (lambda) determinant.",
        "opt_JDTN_u_param": (
            "Optimize the DTN polynomial e-e channel (alpha_indep, 'u' block, J2 analogue). "
            "Has effect only when the wavefunction was built with a DTN preset containing 'u' "
            "(e.g. 'u_chi_f', 'u_chi', 'u'). Default: true."
        ),
        "opt_JDTN_chi_param": (
            "Optimize the DTN polynomial e-N channel (beta_indep, 'chi' block, J1 analogue). "
            "Has effect only when the wavefunction's DTN preset includes 'chi'. Default: true."
        ),
        "opt_JDTN_f_param": (
            "Optimize the DTN polynomial factorised e-e-N channel (gamma_indep, 'f' block, "
            "J3 analogue). Has effect only when the wavefunction's DTN preset includes 'f'. "
            "Default: true."
        ),
        "optimizer_kwargs": (
            "Optimizer configuration. 'method' must be 'sr' (default; stochastic reconfiguration / natural "
            "gradient -- the only supported optimizer). Keys: 'delta' (prefactor in c_i <- c_i + delta * "
            "S^{-1} f), 'epsilon' (regularization strength added to S), optional 'cg_flag'/'cg_max_iter'/'cg_tol' "
            "controlling the conjugate-gradient solver ('cg_tol' is relative: CG stops once ||r|| <= cg_tol * ||b||), "
            "and 'use_lm' (with 'lm_subspace_dim'/'lm_cond') to enable the linear method. 'lm_cond' is the smallest "
            "eigenvalue kept in the correlation matrix of the LM coordinates and must lie in the open interval (0, 1); "
            "coordinates are removed by one pivoted Cholesky pass until the kept set satisfies it."
        ),
    },
    "lrdmc-bra": {
        "num_mcmc_steps": None,
        "num_mcmc_per_measurement": None,
        "alat": 0.30,
        "non_local_move": "tmove",
        "num_gfmc_warmup_steps": 25,
        "num_gfmc_bin_blocks": 10,
        "num_gfmc_collect_steps": 5,
        "E_scf": 0.0,
        "atomic_force": False,
        "compute_stress": False,
        "use_swct": False,
        "epsilon_PW": 0.0,
    },
    "lrdmc-bra_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.",
        "num_mcmc_per_measurement": (
            "Number of GFMC projections between successive observable measurements (an integer >= 1). Required, "
            "no default: it sets the imaginary time per measurement (about num_mcmc_per_measurement * alat**2 / "
            "(3 N_e)) and depends on the system and on alat. On restart it may be omitted; the checkpoint value is kept."
        ),
        "alat": "The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.",
        "non_local_move": "The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.",
        "num_gfmc_warmup_steps": "Number of initial observable measurements discarded as warmup.",
        "num_gfmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
        "num_gfmc_collect_steps": "Number of measurement (before binning) for collecting the weights.",
        "E_scf": "The initial guess of the total energy. This is used to compute the initial energy shift in the GFMC.",
        "atomic_force": "If true, compute atomic forces.",
        "compute_stress": (
            "If true, compute the symmetric stress tensor sigma (Ha/bohr^3) and the diagonal pressure "
            "P = -diag(sigma). Supported by both lrdmc-bra (GFMC_n) and lrdmc-tau (GFMC_t). "
            "Periodic systems only; molecular systems return zeros. Default: false."
        ),
        "use_swct": "If true, apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is False for LRDMC.",
        "epsilon_PW": "Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is regularized near the nodal surface. Default is 0.0 (no regularization).",
    },
    "lrdmc-tau": {
        "num_mcmc_steps": None,
        "tau": 0.10,
        "alat": 0.30,
        "non_local_move": "tmove",
        "num_gfmc_warmup_steps": 25,
        "num_gfmc_bin_blocks": 10,
        "num_gfmc_collect_steps": 5,
        "atomic_force": False,
        "compute_stress": False,
        "use_swct": False,
        "epsilon_PW": 0.0,
    },
    "lrdmc-tau_comments": {
        "num_mcmc_steps": "Number of observable measurement steps per MPI process and walker. The global total is num_mcmc_steps * mpi_size * number_of_walkers.",
        "tau": "the imaginary time between branchings (Ha^-1).",
        "alat": "The lattice discretization parameter (i.e. grid size) used for discretized the Hamiltonian and potential. The lattice spacing is alat * a0, where a0 is the Bohr radius.",
        "non_local_move": "The treatment of the non-local term in the Effective core potential. tmove (T-move) and dltmove (Determinant locality approximation with T-move) are available.",
        "num_gfmc_warmup_steps": "Number of initial observable measurements discarded as warmup.",
        "num_gfmc_bin_blocks": "Number of blocks for binning per MPI and Walker. i.e., the total number of binned blocks is num_gfmc_bin_blocks, not num_gfmc_bin_blocks * mpi_size * number_of_walkers.",
        "num_gfmc_collect_steps": "Number of measurement (before binning) for collecting the weights.",
        "atomic_force": "If true, compute atomic forces.",
        "compute_stress": (
            "If true, compute the symmetric stress tensor sigma (Ha/bohr^3) and the diagonal pressure "
            "P = -diag(sigma). Periodic systems only; ignored for molecular runs. Requires the "
            "electron-position derivatives, which the CLI enables automatically when either "
            "atomic_force or compute_stress is true. Default: false."
        ),
        "use_swct": "If true, apply Space Warp Coordinate Transformation (SWCT) to atomic forces. Default is False for LRDMC.",
        "epsilon_PW": "Pathak-Wagner regularization parameter (bohr). When > 0, the force estimator is regularized near the nodal surface. Default is 0.0 (no regularization).",
    },
    "coulomb": {
        "mpc": False,
        "mpc_cutoff": MPC_CUTOFF_HA,
    },
    "coulomb_restart_policy": {
        "mpc": "frozen",
        "mpc_cutoff": "frozen",
    },
    "coulomb_comments": {
        "mpc": (
            "If true, accumulate the MPC (Model Periodic Coulomb) finite-size correction to the "
            "electron-electron channel. MPC is a PASSIVE observable: walkers are propagated under "
            "Ewald and <V_ee^MPC - V_ee^Ewald> is averaged over the same configurations, so the "
            "reported energy gains E_MPC = E_Ewald + <dV> while sampling, branching, forces and "
            "parameter gradients all stay on the Ewald channel unchanged. Energy only -- forces "
            "are NOT MPC-corrected. Requires a periodic system. Default: false."
        ),
        "mpc_cutoff": (
            "Reciprocal-space kinetic-energy cutoff (Ha) for the MPC kernel and density FFT grid. "
            "This is the only numerical knob of MPC -- it is a CONVERGENCE parameter, not a "
            "physical one: double it until the correction stops moving. Matches CASINO's "
            "mpc_cutoff and QMCPACK's cutoff. Ignored when mpc = false. Default: 30.0."
        ),
    },
}
