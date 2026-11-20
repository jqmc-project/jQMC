"""No-inverse determinant derivatives assemble G from the fused orbital value table.

Fused-table rule (#762): without a supplied running inverse, an analytic
determinant derivative must assemble the geminal ``G`` and its fresh inverse
from the fused value/gradient/Laplacian table
(``determinant._orbital_value_grad_lap_tables``), the table that the drift
(``kinetic._det_grad_field``), the kinetic reverse pass and the ECP mesh base
invert. The value-only ``compute_orb_api`` table routes AO values through
``ao_eval`` (float32 in mixed). #762 moved the drift and the mesh base onto the
fused table, but the nuclear force ``assemble.grad_lndet_wrt_R``, the periodic
strain derivative ``pbc_stress.grad_lndet_wrt_strain`` and the radial-basis
value cotangents ``basis._det_value_cotangents`` still inverted the value-only
table. Without an inverse, d/dR and d/dr therefore contracted two different
``G^{-1}``: the debug MCMC force oracle lost the SWCT translation cancellation
in mixed (``test_jqmc_mcmc[N_ae_ccpvdz_cart.h5-True-True-False]``).

Two detectors:

* ``test_noinv_det_position_derivatives_are_translation_invariant``: the
  determinant part of ln|Psi| is invariant under a rigid translation of all
  electrons and nuclei, so ``sum_I dlnPsi/dR_I + sum_e dlnPsi/dr_e = 0``
  exactly for any shared ``G^{-1}``. The public ``compute_grad_ln_Psi`` is
  evaluated without an inverse on a determinant-only wavefunction (no
  Jastrow, so no ``jastrow_grad_lap`` term enters). The residual is compared
  with zero under the strict ``det_grad_lap``/``ao_grad_lap`` budget (float64
  in both modes).
* ``test_noinv_det_derivatives_do_not_read_the_value_only_table``: each
  no-inverse kernel is evaluated with the geminal's ``compute_orb_api``
  replaced by a NaN table and must equal its unpatched value exactly; the
  supplied-inverse route of the same kernel must read the patched table
  (liveness control), so the probe is live.
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

import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import derivatives
from jqmc._precision import get_tolerance_min
from jqmc.derivatives.assemble import grad_lndet_wrt_R
from jqmc.derivatives.basis import _det_value_cotangents
from jqmc.derivatives.fast import geminal_pinv
from jqmc.derivatives.kinetic import _det_grad_field
from jqmc.derivatives.pbc_stress import grad_lndet_wrt_strain
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel._generated_matrix_base import _apply_gem

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

# (trexio file, geminal representation). Molecular Cartesian SD and dense AO
# AGP, molecular spherical AGPn, periodic real-Gamma ccECP (spherical pGTO)
# and the complex-twist bcc Li ccECP (its Bloch AO values are ao_bloch_eval).
_CASES = [
    ("N_ae_ccpvdz_cart.h5", "SD"),
    ("N_ae_ccpvdz_cart.h5", "AGP"),
    ("H2_ae_ccpvdz_sphe.h5", "AGPn"),
    ("lithium.h5", "SD"),
    ("lithium_k025.h5", "SD"),
]


def _det_only_hamiltonian(stem, gem):
    """The committed fixture's geminal with no Jastrow factor."""
    structure, _, _, _, geminal, coulomb = read_trexio_file(trexio_file=os.path.join(_TREXIO, stem), store_tuple=True)
    jastrow = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    wavefunction = Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal)
    hamiltonian = Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)
    return _apply_gem(hamiltonian, gem)


def _configuration(hamiltonian, seed=0):
    geminal = hamiltonian.wavefunction_data.geminal_data
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(int(geminal.num_electron_up), 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(int(geminal.num_electron_dn), 3))),
    )


@pytest.mark.parametrize("trexio_file,geminal", _CASES)
def test_noinv_det_position_derivatives_are_translation_invariant(trexio_file, geminal):
    """``sum_I dlnPsi/dR_I + sum_e dlnPsi/dr_e = 0`` on the no-inverse analytic path.

    Translating every electron and nucleus by the same vector leaves the
    determinant unchanged (periodic AOs follow their home atom, so this holds
    at Gamma and at a twist). Both axes contract one ``G^{-1}`` with the same
    AO gradients, so the identity is exact up to float64 roundoff for any
    inverse; a residual means the axes inverted two different geminal
    matrices. The reference value is exactly zero, so the strict budget acts
    through its absolute part.
    """
    hamiltonian = _det_only_hamiltonian(trexio_file, geminal)
    r_up, r_dn = _configuration(hamiltonian)
    grads = derivatives.compute_grad_ln_Psi(hamiltonian, r_up, r_dn, {"opt_r": True, "opt_R": True}, analytic=True)
    d_R = np.real(np.asarray(grads["R"]))
    d_r = np.real(np.concatenate([np.asarray(grads["r"][0]), np.asarray(grads["r"][1])], axis=0))
    assert np.all(np.isfinite(d_R)) and np.all(np.isfinite(d_r))
    atol, rtol = get_tolerance_min(("det_grad_lap", "ao_grad_lap"), "strict")
    # Liveness: the determinant drift is a live, non-negligible signal.
    assert np.max(np.abs(d_r)) > 1.0e3 * atol
    residual = d_R.sum(axis=0) + d_r.sum(axis=0)
    np.testing.assert_allclose(residual, np.zeros(3), atol=atol, rtol=rtol)


def _value_inverse(geminal, r_up, r_dn):
    """A fresh inverse of the value-only geminal (the driver's running-inverse boundary)."""
    orb_up = geminal.compute_orb_api(geminal.orb_data_up_spin, r_up)
    orb_dn = geminal.compute_orb_api(geminal.orb_data_dn_spin, r_dn)
    n_dn = geminal.num_electron_dn
    if isinstance(geminal, SD_data):
        G = jnp.hstack([orb_up[:n_dn].T @ orb_dn, orb_up[n_dn:].T])
    else:
        lam_p, lam_unp = jnp.hsplit(geminal._lambda_matrix_jnp, [geminal.orb_num_dn])
        G = jnp.hstack([orb_up.T @ lam_p @ orb_dn, orb_up.T @ lam_unp])
    return geminal_pinv(G)


def _kernel(axis):
    if axis == "R":
        return lambda gd, ru, rd, gi: grad_lndet_wrt_R(gd, ru, rd, take_real=True, geminal_inv=gi)
    if axis == "r":
        return lambda gd, ru, rd, gi: _det_grad_field(gd, ru, rd, geminal_inv=gi)
    if axis == "eps":
        return lambda gd, ru, rd, gi: grad_lndet_wrt_strain(gd, ru, rd, take_real=True, geminal_inv=gi)
    if axis == "basis":
        return lambda gd, ru, rd, gi: _det_value_cotangents(gd, ru, rd, geminal_inv=gi)
    raise ValueError(axis)


@pytest.mark.parametrize(
    "trexio_file,geminal,axis",
    [
        ("N_ae_ccpvdz_cart.h5", "SD", "R"),
        ("N_ae_ccpvdz_cart.h5", "SD", "r"),
        ("N_ae_ccpvdz_cart.h5", "SD", "basis"),
        ("N_ae_ccpvdz_cart.h5", "AGP", "R"),
        ("N_ae_ccpvdz_cart.h5", "AGP", "basis"),
        ("lithium.h5", "SD", "R"),
        ("lithium.h5", "SD", "eps"),
        ("lithium.h5", "SD", "basis"),
        ("lithium_k025.h5", "SD", "R"),
        ("lithium_k025.h5", "SD", "eps"),
    ],
)
def test_noinv_det_derivatives_do_not_read_the_value_only_table(trexio_file, geminal, axis, monkeypatch):
    """A no-inverse determinant derivative is independent of the value-only orbital table.

    The kernel runs eagerly once unpatched and once with the geminal class's
    ``compute_orb_api`` evaluator returning NaN; the no-inverse results must be
    identical. With a supplied value-table inverse the same kernel must read
    the patched table (non-finite output), which shows the patch is live.
    """
    hamiltonian = _det_only_hamiltonian(trexio_file, geminal)
    gd = hamiltonian.wavefunction_data.geminal_data
    r_up, r_dn = _configuration(hamiltonian)
    kernel = _kernel(axis)
    supplied = _value_inverse(gd, r_up, r_dn)

    def leaves(value):
        return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(value)]

    with jax.disable_jit():
        reference = leaves(kernel(gd, r_up, r_dn, None))
        original = inspect.getattr_static(gd, "compute_orb_api")  # a property returning the evaluator

        def poisoned(self):
            evaluate = original.fget(self)
            return lambda *args, **kwargs: jnp.full_like(evaluate(*args, **kwargs), jnp.nan)

        monkeypatch.setattr(type(gd), "compute_orb_api", property(poisoned))
        probed = leaves(kernel(gd, r_up, r_dn, None))
        control = leaves(kernel(gd, r_up, r_dn, supplied))
    assert all(np.all(np.isfinite(leaf)) for leaf in reference)
    assert not all(np.all(np.isfinite(leaf)) for leaf in control), "the supplied-inverse route must read the patched table"
    probed_values = np.concatenate([leaf.ravel() for leaf in probed])
    reference_values = np.concatenate([leaf.ravel() for leaf in reference])
    np.testing.assert_array_equal(probed_values, reference_values)
