"""MPC cutoff domain: ``mpc_cutoff`` must be a positive finite real scalar (issue #802).

``mpc_cutoff`` keeps the reciprocal vectors with ``G^2/2 <= mpc_cutoff``.
A cutoff of 0 keeps only ``G = 0``, i.e. none of the ``G != 0`` long-range
model correction, and once shifted dE_MPC of the periodic H2 cell by about
78 mHa without any error. Negative and non-finite values failed only with
unrelated conversion errors from inside the grid resolver, or, with an
explicit grid, silently returned the bare minimum-image sum.

The expected outcome is the documented contract itself (the same one
``jqmc_tool``'s MPC extrapolation applies to stored cutoffs): every public
route that takes the cutoff -- ``build_mpc_data``, the MCMC / GFMC_t / GFMC_n
constructors with ``mpc=True`` and the ``[coulomb]`` TOML section of the CLI --
raises ``ValueError`` naming ``mpc_cutoff`` and the rejected value, and the
CLI writes no restart checkpoint.
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

import copy
import logging
import sys

import jax
import pytest
import toml

from jqmc import jqmc_cli
from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.mpc import build_mpc_data

jax.config.update("jax_enable_x64", True)

_REJECTION = r"mpc_cutoff must be a positive finite real scalar"
_INVALID_CUTOFFS = {"zero": 0.0, "negative": -1.0, "nan": float("nan"), "inf": float("inf")}
_EXPLICIT_GRID = (8, 8, 8)
_NO_ELECTRONS = "the cutoff is rejected before any electron configuration is generated"

_CLI_JOB_VALUES = {
    "mcmc": {
        "num_mcmc_steps": 12,
        "num_mcmc_per_measurement": 2,
        "num_mcmc_warmup_steps": 2,
        "num_mcmc_bin_blocks": 2,
    },
    "lrdmc-bra": {
        "num_mcmc_steps": GFMC_MIN_STEPS,
        "num_mcmc_per_measurement": 2,
        "alat": 0.30,
        "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
        "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
        "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
    },
    "lrdmc-tau": {
        "num_mcmc_steps": GFMC_MIN_STEPS,
        "tau": 0.05,
        "alat": 0.30,
        "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
        "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
        "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
    },
}


@pytest.fixture(scope="module")
def h2_pbc_hamiltonian():
    """Periodic all-electron H2 cell (the issue's reproducer), no Jastrow."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    return _build_ham_for_cell("H2_pbc_ae_cart.h5", False, None, "SD")


@pytest.fixture(scope="module")
def h2_pbc_hamiltonian_h5(h2_pbc_hamiltonian, tmp_path_factory):
    """The same Hamiltonian written with the production writer, for the CLI."""
    path = tmp_path_factory.mktemp("mpc_cutoff_ham") / "hamiltonian.h5"
    h2_pbc_hamiltonian.save_to_hdf5(str(path))
    return path


def _assert_names_rejected_value(excinfo, cutoff):
    assert repr(cutoff) in str(excinfo.value), f"the error must name the rejected value {cutoff!r}: {excinfo.value}"


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.gdd_electron_configuration(mode="not-required", reason=_NO_ELECTRONS)
@pytest.mark.parametrize("grid_kind", ["auto", "explicit"])
@pytest.mark.parametrize("cutoff_case", list(_INVALID_CUTOFFS))
def test_build_mpc_data_rejects_non_positive_or_non_finite_cutoff(h2_pbc_hamiltonian, cutoff_case, grid_kind):
    """``build_mpc_data`` refuses a cutoff outside ``(0, inf)``, with or without an explicit grid."""
    cutoff = _INVALID_CUTOFFS[cutoff_case]
    grid = "auto" if grid_kind == "auto" else _EXPLICIT_GRID
    with pytest.raises(ValueError, match=_REJECTION) as excinfo:
        build_mpc_data(h2_pbc_hamiltonian, cutoff_ha=cutoff, grid=grid)
    _assert_names_rejected_value(excinfo, cutoff)


@pytest.mark.light
@pytest.mark.without_chdir
@pytest.mark.gdd_electron_configuration(mode="not-required", reason=_NO_ELECTRONS)
@pytest.mark.parametrize("driver", ["mcmc", "gfmc_t", "gfmc_n"])
@pytest.mark.parametrize("cutoff_case", list(_INVALID_CUTOFFS))
def test_mpc_samplers_reject_non_positive_or_non_finite_cutoff(h2_pbc_hamiltonian, cutoff_case, driver):
    """MCMC, GFMC_t and GFMC_n with ``mpc=True`` refuse the cutoff at construction, before sampling."""
    cutoff = _INVALID_CUTOFFS[cutoff_case]
    with pytest.raises(ValueError, match=_REJECTION) as excinfo:
        if driver == "mcmc":
            MCMC(hamiltonian_data=h2_pbc_hamiltonian, mcmc_seed=1, num_walkers=1, mpc=True, mpc_cutoff=cutoff)
        elif driver == "gfmc_t":
            GFMC_t(hamiltonian_data=h2_pbc_hamiltonian, num_walkers=1, mcmc_seed=1, alat=0.30, mpc=True, mpc_cutoff=cutoff)
        else:
            GFMC_n(
                hamiltonian_data=h2_pbc_hamiltonian,
                num_walkers=1,
                mcmc_seed=1,
                alat=0.30,
                num_mcmc_per_measurement=16,
                mpc=True,
                mpc_cutoff=cutoff,
            )
    _assert_names_rejected_value(excinfo, cutoff)


@pytest.mark.light
@pytest.mark.with_chdir
@pytest.mark.gdd_electron_configuration(mode="not-required", reason=_NO_ELECTRONS)
@pytest.mark.parametrize("job_type", ["mcmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.parametrize("cutoff_case", list(_INVALID_CUTOFFS))
def test_cli_rejects_non_positive_or_non_finite_mpc_cutoff(h2_pbc_hamiltonian_h5, tmp_path, monkeypatch, cutoff_case, job_type):
    """A ``[coulomb] mpc = true`` TOML with such a cutoff fails and writes no restart checkpoint."""
    cutoff = _INVALID_CUTOFFS[cutoff_case]
    restart_chk = tmp_path / "restart.h5"
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": 13,
            "number_of_walkers": 1,
            "hamiltonian_h5": str(h2_pbc_hamiltonian_h5),
            "restart": False,
            "restart_chk": str(restart_chk),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: dict(_CLI_JOB_VALUES[job_type]),
        "coulomb": {"mpc": True, "mpc_cutoff": cutoff},
    }
    input_file = tmp_path / "input.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            with pytest.raises(ValueError, match=_REJECTION) as excinfo:
                jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    _assert_names_rejected_value(excinfo, cutoff)
    assert not restart_chk.exists(), "a rejected mpc_cutoff must not leave a restart checkpoint"
