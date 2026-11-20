"""Typed-raise contract for bounded Rust guards.

* ``_gfmc_grad_ln_Psi`` on a PERIODIC geminal must raise a typed
  ``NotImplementedError``, not a raw ``pyo3_runtime.PanicException``
  (``ndarray: index out of bounds`` from ``geminal_cart_from_py`` ->
  ``orbdata_to_cart`` on the periodic leaf). Mirrors the sibling
  ``_gfmc_grad_e_L_ae`` molecular-only guard.

The pure plane-wave / B-spline complex-twist eps stress leaf
(``vmc_grad_e_L_eps_pbc_complex``) does not raise: its XLA parity is required in
``test_ungate_complex_stress_leaf`` + ``test_rs_analytic_matrix_rRe``, and the
pGTO(+)B-spline composite has direct Gamma/twist XLA parity in
``test_vmc_stress_pbc_pgto_bspline_composite``.
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
import pathlib

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jqmc_rs

from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.light
def test_production_drivers_do_not_use_optimized_away_asserts():
    """User/input validation must survive ``python -O`` in both CPU drivers."""
    offenders = []
    for rel in ("jqmc/jqmc_mcmc.py", "jqmc/jqmc_gfmc.py"):
        path = _ROOT / rel
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assert):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, "production validation disappears under python -O:\n  " + "\n  ".join(offenders)


def _config(ham, seed=1, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=scale, size=(nd, 3))
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


# --------------------------------------------------------------------------- #
# Periodic geminal -> typed NotImplementedError (not PanicException).
# --------------------------------------------------------------------------- #
@pytest.mark.light
def test_gfmc_grad_ln_psi_periodic_raises():
    """GFMC grad ln psi periodic raises."""
    ham = rs_ham("pbc_gamma_sphepgto_jnone")
    wf = ham.wavefunction_data
    ru, rd = _config(ham)
    with pytest.raises(NotImplementedError):
        jqmc_rs._gfmc_grad_ln_Psi(wf.geminal_data, wf.jastrow_data, ru, rd)


# --------------------------------------------------------------------------- #
# The complex-twist pure-PW / B-spline eps stress leaf has no raise contract: its
# XLA parity is required in ``test_ungate_complex_stress_leaf`` /
# ``test_rs_analytic_matrix_rRe`` (see the module docstring).
# --------------------------------------------------------------------------- #
