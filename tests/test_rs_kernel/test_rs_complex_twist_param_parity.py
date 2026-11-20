"""Complex-twist Rust/XLA parameter-derivative parity outside the generated matrix.

The generated c-axis matrix (``test_rs_analytic_matrix.py``) walks the covering array of
``gdd.engine.derivative_matrix._generate_matrix``. Two production complex-twist Rust legs are
unreachable from that array, so they get dedicated nodes here that reuse the matrix module's
builders, XLA oracle and tolerance policy verbatim (no new tolerance):

* ``grad_eL_ecp_wrt_jastrow_3b_radial_complex``. The matrix yields no
  complex-pGTO twist ECP J3poly cell, so the production assembler's ECP J3 radial add
  (``_RsComplexPdCtx.pd_el_ecp_j3_basis``) is not compared there. The node below builds exactly
  the cell ``('sphe_pgto_complex', 'J3poly', 'ECP', 'twist', 'SD')`` (``lithium_k025.h5`` with a
  complex-pGTO J3) through the SAME ``_resolve`` / ``_build_ham_for_cell`` and compares the
  production ``_rs_native_c_param_grads`` J3 radial blocks against the XLA
  ``_reduce_c_param_grads`` reduction, after proving that the ECP contribution is non-zero.
* Open-shell AGPn real paired/unpaired energy parameters.
  A nonfactorized oxygen twist supplies live ECP derivatives absent from the
  closed-shell matrix rows. Same-precision primal AD independently checks the
  XLA reference before it anchors Rust parity; neither a common analytical
  projection error nor an empty unpaired block may close this evidence.
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

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import jqmc_rs

from jqmc import derivatives
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION
from jqmc.derivatives import bridge

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell, _resolve
from tests.test_rs_kernel.test_rs_analytic_matrix import (
    _BLOCK_FLAGS,
    _Z_KIN,
    ATOL_PERIODIC_EL_KINETIC_FLOOR,
    _config,
    _is_complex,
    _one_move_mesh,
    _xla_c,
)

pytestmark = pytest.mark.with_chdir


@pytest.fixture(autouse=True)
def _clear_caches():
    """Same per-test teardown as the generated matrix (unbounded bridge LRU)."""
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


def _periodic_e_l_tolerance():
    """The matrix's e_L policy: medium XLA-Rust envelope + the periodic kinetic floor."""
    atol, rtol = get_tolerance_xla_rust_consistency(_Z_KIN, level="medium")
    return max(atol, ATOL_PERIODIC_EL_KINETIC_FLOOR), rtol


# --------------------------------------------------------------------------- #
# Complex-pGTO twist ECP J3 radial e_L
# --------------------------------------------------------------------------- #
_J3_ECP_TWIST_CELL = ("sphe_pgto_complex", "J3poly", "ECP", "twist", "SD")


def _j3_basis_ecp_twist_case():
    build = _resolve(_J3_ECP_TWIST_CELL)
    assert build is not None, f"{_J3_ECP_TWIST_CELL}: no committed builder"
    stem, with_jastrow, spec, gem = build
    assert stem == "lithium_k025.h5" and spec == {"j3": "complex"}, (stem, spec)
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
    assert _is_complex(ham), "the J3 ECP twist cell must build a complex-twist geminal"
    assert bool(ham.coulomb_potential_data.ecp_flag), "the J3 ECP twist cell must carry an ECP"
    r_up, r_dn, _generation = _config(ham)
    return ham, r_up, r_dn


def _j3_basis_production(ham, r_up, r_dn, *, ecp_flag):
    """The PRODUCTION Rust J3 radial blocks, via the same assembler the MCMC driver uses."""
    from jqmc.jqmc_mcmc import _RsComplexPdCtx, _rs_native_c_param_grads

    wf = ham.wavefunction_data
    ctx = _RsComplexPdCtx(wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data)
    weights, grid_points = _one_move_mesh()
    got = _rs_native_c_param_grads(
        ctx,
        ham,
        [np.asarray(r_up)],
        [np.asarray(r_dn)],
        _BLOCK_FLAGS["J3Basis"],
        1,
        quantity="e_L",
        ecp_flag=ecp_flag,
        ecp_w=weights,
        ecp_gp=grid_points,
        rts_b=[np.eye(3)],
        complex_twist=True,
    )
    return {name: np.asarray(value) for name, value in got.items() if name.startswith("j3_") and "basis_" in name}


def _j3_basis_xla(ham, r_up, r_dn):
    from jqmc.derivatives.bridge import _reduce_c_param_grads

    flags = _BLOCK_FLAGS["J3Basis"]
    c = _xla_c(ham, r_up, r_dn, "J3Basis", "e_L")
    reduced = _reduce_c_param_grads(
        ham.wavefunction_data, jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), c), 1, flags
    )
    return {name: np.asarray(value) for name, value in reduced.items() if name.startswith("j3_") and "basis_" in name}


def _assert_j3_basis_close(got, expected):
    assert set(got) == set(expected) and got, f"J3 radial block sets differ: rust={sorted(got)} xla={sorted(expected)}"
    atol, rtol = _periodic_e_l_tolerance()
    for name in sorted(expected):
        np.testing.assert_allclose(
            got[name], expected[name], atol=atol, rtol=rtol, err_msg=f"complex twist ECP J3Basis/{name} (Rust vs XLA)"
        )


@pytest.mark.heavy
def test_rs_complex_twist_ecp_j3_basis_e_l_matches_xla(monkeypatch):
    """``grad_eL_ecp_wrt_jastrow_3b_radial_complex`` reaches and matches XLA.

    1. Non-vacuity: the free export is non-zero on this configuration, and the production
       ECP-on vs ECP-off J3 radial blocks differ by more than the named distinction floor --
       so dropping the ECP J3 radial add cannot pass unnoticed.
    2. Value: production Rust (ECP on) == XLA analytic reduction, block by block, at the
       matrix's periodic e_L tolerance.
    3. Negative control: with the export monkeypatched to zeros the same comparison fails.
    """
    ham, r_up, r_dn = _j3_basis_ecp_twist_case()
    wf = ham.wavefunction_data
    weights, grid_points = _one_move_mesh()
    ecp_exp, ecp_coeff = jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_radial_complex(
        wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, r_up, r_dn, weights, grid_points
    )
    ecp_mag = max(float(np.max(np.abs(np.asarray(ecp_exp)))), float(np.max(np.abs(np.asarray(ecp_coeff)))))
    assert ecp_mag > ATOL_NONVACUITY_DISTINCTION, f"ECP J3 radial export is ~0 ({ecp_mag}) -- the cell is vacuous"

    got = _j3_basis_production(ham, r_up, r_dn, ecp_flag=True)
    got_no_ecp = _j3_basis_production(ham, r_up, r_dn, ecp_flag=False)
    shift = max(float(np.max(np.abs(got[name] - got_no_ecp[name]))) for name in got)
    assert shift > ATOL_NONVACUITY_DISTINCTION, f"production ECP J3 radial add is ~0 ({shift})"

    expected = _j3_basis_xla(ham, r_up, r_dn)
    _assert_j3_basis_close(got, expected)

    # Negative control: the export returning zeros must turn the comparison red.
    def _zeros(*args, **kwargs):
        e, c = ecp_exp, ecp_coeff
        return np.zeros_like(np.asarray(e)), np.zeros_like(np.asarray(c))

    monkeypatch.setattr(jqmc_rs, "grad_eL_ecp_wrt_jastrow_3b_radial_complex", _zeros)
    mutated = _j3_basis_production(ham, r_up, r_dn, ecp_flag=True)
    with pytest.raises(AssertionError, match="complex twist ECP J3Basis/"):
        _assert_j3_basis_close(mutated, expected)


def _open_shell_real_agpn_case(tmp_path, parameter_rng, electron_rng):
    """Build a nonfactorized real-parameter AGPn on the committed oxygen twist."""
    from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd
    from tests.test_rs_kernel.test_periodic_lsda import _build_twist_pgto, _config as oxygen_config

    ham = _build_twist_pgto(tmp_path)
    wf = ham.wavefunction_data
    gem = build_explicit_common_rank_agpn_from_sd(wf.geminal_data)
    rng = parameter_rng
    gem = gem.replace(
        lambda_diag=jnp.asarray(0.3 + rng.random(np.asarray(gem.lambda_diag).shape)),
        lambda_unpaired=jnp.asarray(
            np.real(np.asarray(gem.lambda_unpaired)) + 0.2 * rng.normal(size=gem.lambda_unpaired.shape)
        ),
    )
    ham = ham.replace(wavefunction_data=wf.replace(geminal_data=gem))
    r_up, r_dn = oxygen_config(ham, electron_rng)
    assert (gem.num_electron_up, gem.num_electron_dn) == (4, 2)
    assert gem.lambda_unpaired.shape == (4, 2)
    assert not np.iscomplexobj(gem.lambda_diag) and not np.iscomplexobj(gem.lambda_unpaired)
    return ham, jnp.asarray(r_up), jnp.asarray(r_dn)


@pytest.mark.heavy
@pytest.mark.parametrize("block", ["paired", "unpaired"])
def test_rs_open_shell_twist_agpn_real_el_matches_xla(tmp_path, block):
    """Rust AGPn real-parameter kinetic plus ECP gradients match XLA at an oxygen twist.

    Reuses the exact committed O-lsda_k025 pGTO ccECP fixture and nonfactorized
    real parameter seed 431 / electron seed 41 of the independent AD anchor:
    open shell, four up/two down, one configuration, no cond(G) rejection,
    12-point ECP mesh and identity rotation. Paired and unpaired leaves are
    explicit cases in full and mixed; the ECP contribution must be nonzero.
    Native free kernels are checked, not the unwired AGPn complex driver
    adapter. Neither reference nor result is projected to hide an imaginary
    mismatch; both describe derivatives of Re(e_L) for real parameters.
    """
    ham, r_up, r_dn = _open_shell_real_agpn_case(tmp_path, np.random.default_rng(431), np.random.default_rng(41))
    wf = ham.wavefunction_data
    gem = wf.geminal_data
    kinetic = jqmc_rs.grad_eL_wrt_lambda_diag_complex(gem, wf.jastrow_data, np.asarray(r_up), np.asarray(r_dn))
    ecp = jqmc_rs.grad_eL_ecp_wrt_lambda_diag_complex(
        gem, wf.jastrow_data, ham.coulomb_potential_data, np.asarray(r_up), np.asarray(r_dn), *_one_move_mesh()
    )
    index = 0 if block == "paired" else 1
    actual = np.asarray(kinetic[index]) + np.asarray(ecp[index])
    assert np.max(np.abs(ecp[index])) > ATOL_NONVACUITY_DISTINCTION
    grad = _xla_c(ham, r_up, r_dn, "Det", "e_L").geminal_data
    expected = np.asarray(grad.lambda_diag if block == "paired" else grad.lambda_unpaired)
    atol, rtol = _periodic_e_l_tolerance()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=f"AGPn real e_L Rust/XLA/{block}")
