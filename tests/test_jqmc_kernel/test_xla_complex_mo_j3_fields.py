"""XLA J3 orbital fields and local-energy gradients of a complex-MO three-body Jastrow on a twist (#96).

The three-body Jastrow consumes the real orbital field ``Re(C @ AO)`` (``Jastrow_three_body_data``
takes ``Re(compute_orb_api)``, i.e. ``Re(compute_MOs)`` for an MO basis). On a twist the public QE
converter's ``-j3 mo`` writes an MO basis with complex coefficients ``C``
(``jastrow_factor._phase_free_j3_orbitals`` keeps the complex occupied-MO matrix). Its AO basis can
itself be complex: a phase-free plane wave ``exp(i G.r)`` or a complex B-spline band keeps a nonzero
``Im AO``. The XLA analytic J3 field tower ``ao_deriv.j3_basis_derivs`` contracted ``C`` with the
already ``Re``-projected AO fields, which is ``Re(C) @ Re(AO)`` and drops ``-Im(C) @ Im(AO)``. Every
consumer of the tower (the e_L electron gradient ``kinetic._j3_side_r``, the J3 stress, the J3
``j_matrix`` and basis-parameter kinetic gradients) then used a wrong J3 orbital.

The cases are the committed Li ccECP k = 0.25 twist fixtures with a J3 on the phase-free complex
occupied MO of the determinant:

* ``pgto``: the ``lithium_k025.h5`` sphe-pGTO SD (``build_twist_mo_j3_ham``); its phase-free pGTO AO
  is real, so this is the control where both lowerings agree;
* ``pgto_pw``: the QE-projected pGTO(+)plane-wave composite SD, the converter's default kind;
* ``pgto_bspline``: the QE-projected pGTO(+)B-spline composite SD. The converter's B-spline block of
  ``C`` is real, so the rotated case multiplies the J3 MO coefficients (only the J3 copy; the
  determinant is untouched) by the global phase ``exp(i pi / 3)``. This rotates the J3 orbital, as
  an optimized complex MO basis would, and makes ``Im(C)`` meet the complex B-spline ``Im AO``.
  The field test also runs every kind with this rotation.

References are independent of the analytic code under test. The fields are compared with
forward-mode autodiff of ``Re(compute_MOs)`` (value, gradient, Hessian, Laplacian = trace of the
Hessian, gradient of the Laplacian from the third derivative). The analytic e_L gradients of
``assemble.compute_grad_e_L`` (``wrt="r"``, ``"eps"``, ``"c"``) and the ln Psi strain gradient of
``assemble.compute_grad_ln_Psi`` are compared with reverse-mode autodiff of the XLA
``Re compute_local_energy`` and ``Re evaluate_ln_wavefunction``. The ``j_matrix`` is a seeded random
matrix of scale 0.3 with a symmetric pair block, so the J3 share is a sizeable part of each gradient.
Full retains its mathematical AD oracles. Mixed rebuilds the analytical route independently
under full and mixed policies and uses central derivative-fidelity budgets; value API parity
and the original real/finite/shape/J3-share/nonzero checks remain separate.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import assemble
from jqmc.derivatives.ao_deriv import j3_basis_derivs
from jqmc.hamiltonians import compute_local_energy
from jqmc.jastrow_factor import Jastrow_three_body_data, _phase_free_j3_orbitals
from jqmc.jqmc_mcmc import _geminal_inv_batched
from jqmc.molecular_orbital import compute_MOs
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_rs_kernel._rs_fixtures import build_twist_mo_j3_ham, draw_well_conditioned_cfg, rs_ham
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._grad_fidelity_harness import (
    _clear_policy_model_caches,
    assert_orbital_mixed_fidelity,
    assert_parameter_mixed_fidelity,
)

jax.config.update("jax_enable_x64", True)

# The tests read only committed fixtures through absolute paths.
pytestmark = [pytest.mark.with_chdir]

#: Zones crossed by the J3 orbital field tower (AO value and derivatives, J3 orbital).
_FIELD_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap")
#: Zones crossed by the twist local energy with J1/J2/J3 and its derivatives.
_ENERGY_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "ewald", "coulomb", "jastrow_eval", "jastrow_grad_lap")
#: Scale of the seeded random J3 matrix (the converter writes zeros; an optimized J3 is nonzero).
_J3_SCALE = 0.3
#: Global phase applied to the MO coefficients of the rotated cases.
_ROTATION = np.pi / 3.0
_FIXTURES = {"pgto_pw": "pgto_pw_twist_jnone", "pgto_bspline": "pgto_bspline_twist_jnone"}
_FIELD_NAMES = ("value", "gradient", "laplacian", "hessian", "gradlap")


def _complex_mo_j3_orbital(kind, rotated):
    """Return ``(ham, orbital)``: the fixture and its phase-free complex occupied-MO J3 basis."""
    if kind == "pgto":
        ham = build_twist_mo_j3_ham()
        orbital = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    else:
        ham = rs_ham(_FIXTURES[kind])
        orbital = _phase_free_j3_orbitals(ham.wavefunction_data.geminal_data.orb_data_up_spin)
    coefficients = np.asarray(orbital.mo_coefficients)
    if rotated:
        coefficients = coefficients * np.exp(1j * _ROTATION)
        orbital = orbital.replace(mo_coefficients=coefficients)
    assert type(orbital).__name__ == "MOs_data", type(orbital).__name__
    assert np.iscomplexobj(coefficients) and np.max(np.abs(coefficients.imag)) > 1.0e-2, coefficients
    return ham, orbital


def _with_complex_mo_j3(kind, rotated):
    """Attach a complex-MO J3 with a seeded random ``j_matrix`` to the fixture of ``kind``."""
    ham, orbital = _complex_mo_j3_orbital(kind, rotated)
    n_mo = np.asarray(orbital.mo_coefficients).shape[0]
    rng = np.random.default_rng(7)
    j_matrix = _J3_SCALE * rng.standard_normal((n_mo, n_mo + 1))
    j_matrix[:, :n_mo] = 0.5 * (j_matrix[:, :n_mo] + j_matrix[:, :n_mo].T)
    wf = ham.wavefunction_data
    j3 = Jastrow_three_body_data(orb_data=orbital, j_matrix=j_matrix)
    ham = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3)))
    ham.sanity_check()
    return ham


def _autodiff_fields(orbital, r):
    """Forward-mode autodiff tower of ``Re(compute_MOs)`` in the ``j3_basis_derivs`` layout."""

    def value(x):
        return jnp.real(compute_MOs(orbital, x[None, :]))[:, 0]

    gradient = jax.jacfwd(value)
    hessian = jax.jacfwd(gradient)
    third = jax.jacfwd(hessian)
    val = np.asarray(jax.vmap(value)(r)).T  # (mo, N)
    grad = np.asarray(jax.vmap(gradient)(r)).transpose(2, 1, 0)  # (3, mo, N)
    hess = np.asarray(jax.vmap(hessian)(r)).transpose(2, 3, 1, 0)  # (3, 3, mo, N)
    lap = np.trace(hess, axis1=0, axis2=1)  # (mo, N)
    gradlap = np.einsum("nmdcc->dmn", np.asarray(jax.vmap(third)(r)))  # (3, mo, N)
    return val, grad, lap, hess, gradlap


def _field_fidelity_case(kind, rotated):
    _clear_policy_model_caches()
    ham, orbital = _complex_mo_j3_orbital(kind, rotated)
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    return orbital, jnp.asarray(np.concatenate([r_up, r_dn], axis=0))


def _analytical_field_outputs(case):
    orbital, r = case
    return {
        "fields": tuple(np.asarray(field) for field in j3_basis_derivs(orbital, r)),
        "value": np.asarray(jnp.real(compute_MOs(orbital, r))),
        "positions": np.asarray(r),
    }


def _assert_same_mode_j3_value(actual, reference, label):
    """Preserve the original strict value identity within the active policy."""
    atol, rtol = get_tolerance_min(_FIELD_ZONES, "strict")
    assert actual.shape == reference.shape
    assert not np.iscomplexobj(actual) and np.all(np.isfinite(actual))
    assert np.all(np.isfinite(reference))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _check_field_fidelity(kind, rotated):
    mixed, full = mixed_vs_full(lambda: _field_fidelity_case(kind, rotated), _analytical_field_outputs)
    assert np.array_equal(mixed["positions"], full["positions"]), "full/mixed field fixture coordinates differ"
    actual, reference = mixed["fields"], full["fields"]
    assert len(actual) == len(reference) == len(_FIELD_NAMES)
    atol, rtol = get_tolerance_min(_FIELD_ZONES, "strict")
    for name, field, ref in zip(_FIELD_NAMES, actual, reference, strict=True):
        assert field.shape == ref.shape, (name, field.shape, ref.shape)
        assert not np.iscomplexobj(field) and np.all(np.isfinite(field)), (name, field)
        assert np.all(np.isfinite(ref)), (name, ref)
        live_reference = mixed["value"] if name == "value" else ref
        assert np.any(np.abs(live_reference) > atol + rtol * np.abs(live_reference)), (name, live_reference)
        label = f"{kind} rotated={rotated} J3 {name}"
        if name == "value":
            _assert_same_mode_j3_value(field, mixed["value"], label)
        elif name in ("gradient", "gradlap"):
            for axis in range(3):
                assert_orbital_mixed_fidelity(field[axis], ref[axis], f"{label}[{axis}] mixed vs full")
        elif name == "hessian":
            for first in range(3):
                for second in range(3):
                    assert_orbital_mixed_fidelity(
                        field[first, second], ref[first, second], f"{label}[{first},{second}] mixed vs full"
                    )
        else:
            assert name == "laplacian"
            assert_orbital_mixed_fidelity(field, ref, label + " mixed vs full")


@pytest.mark.parametrize(
    ("kind", "rotated"),
    [
        pytest.param("pgto", False, marks=pytest.mark.light),
        pytest.param("pgto", True, marks=pytest.mark.light),
        pytest.param("pgto_pw", False, marks=pytest.mark.light),
        pytest.param("pgto_pw", True, marks=pytest.mark.light),
        pytest.param("pgto_bspline", False, marks=pytest.mark.light),
        pytest.param("pgto_bspline", True, marks=pytest.mark.light),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_xla_complex_mo_j3_basis_fields_match_autodiff_of_re_mo(kind, rotated):
    """Full retains the field AD oracle; mixed retains full analytical derivatives and same-mode value parity."""
    if is_mixed_precision_enabled():
        _check_field_fidelity(kind, rotated)
        return
    ham, orbital = _complex_mo_j3_orbital(kind, rotated)
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    r = jnp.asarray(np.concatenate([r_up, r_dn], axis=0))

    reference = _autodiff_fields(orbital, r)
    atol, rtol = get_tolerance_min(_FIELD_ZONES, "strict")
    actual = [np.asarray(field) for field in j3_basis_derivs(orbital, r)]
    assert len(actual) == len(reference) == len(_FIELD_NAMES)
    for name, field, ref in zip(_FIELD_NAMES, actual, reference, strict=True):
        assert field.shape == ref.shape, (name, field.shape, ref.shape)
        assert not np.iscomplexobj(field) and np.all(np.isfinite(field)), (name, field)
        assert np.all(np.isfinite(ref)), (name, ref)
        # Non-vacuous: a zero field would fail the comparison below.
        assert np.any(np.abs(ref) > atol + rtol * np.abs(ref)), (name, ref)
    np.testing.assert_allclose(
        np.concatenate([field.ravel() for field in actual]),
        np.concatenate([ref.ravel() for ref in reference]),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"j3_basis_derivs(MO) != autodiff of Re(compute_MOs) ({kind}, rotated={rotated})",
    )


def _j3_radial_leaf(orbital):
    """Return ``(child_index, leaf)`` of the GTO/pGTO leaf carrying the J3 radial parameters."""
    aos = orbital.aos_data
    if type(aos).__name__ != "AOs_composite_data":
        return None, aos
    index = next(i for i, child in enumerate(aos.children) if "gto" in type(child).__name__)
    return index, aos.children[index]


def _with_j3_params(ham, j_matrix, exponents, coefficients):
    """Replace the J3 ``j_matrix`` and the radial exponents/coefficients of its GTO/pGTO leaf."""
    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    orbital = j3.orb_data
    index, leaf = _j3_radial_leaf(orbital)
    leaf = leaf.replace(radial=leaf.radial.replace(exponents=exponents, coefficients=coefficients))
    if index is None:
        aos = leaf
    else:
        children = list(orbital.aos_data.children)
        children[index] = leaf
        aos = orbital.aos_data.replace(children=tuple(children))
    j3 = j3.replace(j_matrix=j_matrix, orb_data=orbital.replace(aos_data=aos))
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3)))


def _j3_params(j3):
    _, leaf = _j3_radial_leaf(j3.orb_data)
    return jnp.asarray(j3.j_matrix), jnp.asarray(leaf.radial.exponents), jnp.asarray(leaf.radial.coefficients)


def _re_local_energy(ham, r_up, r_dn):
    return jnp.real(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))


def _re_ln_psi(ham, r_up, r_dn):
    return jnp.real(evaluate_ln_wavefunction(ham.wavefunction_data, r_up, r_dn))


def _strained(function, ham, r_up, r_dn):
    def at(eps):
        deformation = jnp.eye(3) + eps
        return function(ham._apply_strain(deformation), r_up @ deformation.T, r_dn @ deformation.T)

    return at


def _gradient_pair(ham, r_up, r_dn, geminal_inv, quantity):
    """Return ``(analytic, autodiff)`` flat real arrays for ``quantity`` (no J3 -> autodiff only)."""
    if quantity == "e_L/r":
        reference = jax.grad(lambda u, d: _re_local_energy(ham, u, d), argnums=(0, 1))(r_up, r_dn)
        reference = np.concatenate([np.asarray(x) for x in reference], axis=0)
        if geminal_inv is None:
            return None, reference
        actual = assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="r", geminal_inv=geminal_inv)
        return np.concatenate([np.asarray(x) for x in actual], axis=0), reference
    if quantity in ("e_L/eps", "ln_Psi/eps"):
        function = _re_local_energy if quantity == "e_L/eps" else _re_ln_psi
        reference = np.asarray(jax.grad(_strained(function, ham, r_up, r_dn))(jnp.zeros((3, 3))))
        if geminal_inv is None:
            return None, reference
        if quantity == "e_L/eps":
            actual = assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="eps", geminal_inv=geminal_inv)
        else:
            actual = assemble.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="eps", geminal_inv=geminal_inv)
        return np.asarray(actual), reference
    assert quantity == "e_L/c", quantity
    j3 = ham.wavefunction_data.jastrow_data.jastrow_three_body_data
    params = _j3_params(j3)
    reference = jax.grad(
        lambda *p: _re_local_energy(_with_j3_params(ham, *p), r_up, r_dn),
        argnums=(0, 1, 2),
    )(*params)
    reference = np.concatenate([np.asarray(x).ravel() for x in reference])
    mask = {"opt_J3_param": True, "opt_J3_basis_exp": True, "opt_J3_basis_coeff": True}
    grad = assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", geminal_inv=geminal_inv, mask=mask)
    actual = np.concatenate([np.asarray(x).ravel() for x in _j3_params(grad.jastrow_data.jastrow_three_body_data)])
    return actual, reference


def _analytical_gradient_blocks(ham, r_up, r_dn, geminal_inv, quantity):
    """Production analytical derivative only, keeping heterogeneous parameter blocks separate."""
    if quantity == "ln_Psi/eps":
        return assemble.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="eps", geminal_inv=geminal_inv)
    if quantity in ("e_L/r", "e_L/eps"):
        return assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt=quantity.split("/")[1], geminal_inv=geminal_inv)
    assert quantity == "e_L/c"
    mask = {"opt_J3_param": True, "opt_J3_basis_exp": True, "opt_J3_basis_coeff": True}
    grad = assemble.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", geminal_inv=geminal_inv, mask=mask)
    j_matrix, exponents, coefficients = _j3_params(grad.jastrow_data.jastrow_three_body_data)
    return {"j_matrix": j_matrix, "j3_basis_exp": exponents, "j3_basis_coeff": coefficients}


def _gradient_fidelity_case(kind, rotated):
    _clear_policy_model_caches()
    ham = _with_complex_mo_j3(kind, rotated)
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    r_up, r_dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    gd = ham.wavefunction_data.geminal_data
    geminal_inv = jnp.asarray(_geminal_inv_batched(gd, r_up[None], r_dn[None])[1])[0]
    return ham, r_up, r_dn, geminal_inv


def _analytical_gradient_outputs(case, quantity):
    ham, r_up, r_dn, geminal_inv = case
    actual = _analytical_gradient_blocks(ham, r_up, r_dn, geminal_inv, quantity)
    no_j3 = None
    if quantity != "e_L/c":
        wf = ham.wavefunction_data
        without = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=None)))
        no_j3 = _analytical_gradient_blocks(without, r_up, r_dn, geminal_inv, quantity)
    return {"actual": actual, "no_j3": no_j3, "r_up": np.asarray(r_up), "r_dn": np.asarray(r_dn)}


def _flat_derivative_blocks(blocks):
    return np.concatenate([np.asarray(value).ravel() for value in jax.tree_util.tree_leaves(blocks)])


def _check_gradient_fidelity(kind, rotated, quantity):
    mixed, full = mixed_vs_full(
        lambda: _gradient_fidelity_case(kind, rotated), lambda case: _analytical_gradient_outputs(case, quantity)
    )
    assert np.array_equal(mixed["r_up"], full["r_up"]) and np.array_equal(mixed["r_dn"], full["r_dn"]), (
        "full/mixed gradient fixture coordinates differ"
    )
    actual, reference = mixed["actual"], full["actual"]
    for value in jax.tree_util.tree_leaves(actual):
        assert not np.iscomplexobj(value) and np.all(np.isfinite(value)), value
    ref_flat = _flat_derivative_blocks(reference)
    assert np.all(np.isfinite(ref_flat)), ref_flat
    j3_share = ref_flat if quantity == "e_L/c" else ref_flat - _flat_derivative_blocks(full["no_j3"])
    atol, rtol = get_tolerance_min(_ENERGY_ZONES, "strict")
    assert np.any(np.abs(j3_share) > atol + rtol * np.abs(ref_flat)), j3_share
    assert_parameter_mixed_fidelity(actual, reference, f"{kind} rotated={rotated} {quantity} complex-MO J3 mixed vs full")


# Every row runs one or two reverse-mode autodiff passes of the twist local energy (45-150 s on the
# CPU reference host), above the 60 s light budget.
@pytest.mark.parametrize("quantity", ["e_L/r", "e_L/eps", "ln_Psi/eps", "e_L/c"])
@pytest.mark.parametrize(
    ("kind", "rotated"),
    [
        pytest.param("pgto", False, marks=pytest.mark.heavy),
        pytest.param("pgto_pw", False, marks=pytest.mark.heavy),
        pytest.param("pgto_bspline", True, marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_xla_complex_mo_j3_local_energy_gradients_match_autodiff(kind, rotated, quantity):
    """Full retains AD; mixed analytical observable derivatives retain independently rebuilt full fidelity."""
    if is_mixed_precision_enabled():
        _check_gradient_fidelity(kind, rotated, quantity)
        return
    ham = _with_complex_mo_j3(kind, rotated)
    r_up, r_dn = draw_well_conditioned_cfg(ham, seed=1)
    r_up, r_dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    gd = ham.wavefunction_data.geminal_data
    geminal_inv = jnp.asarray(_geminal_inv_batched(gd, r_up[None], r_dn[None])[1])[0]

    actual, reference = _gradient_pair(ham, r_up, r_dn, geminal_inv, quantity)
    atol, rtol = get_tolerance_min(_ENERGY_ZONES, "strict")
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    assert not np.iscomplexobj(actual) and np.all(np.isfinite(actual)), actual
    assert np.all(np.isfinite(reference)), reference
    if quantity != "e_L/c":
        # Non-vacuous: dropping the J3 share of the reference would fail the comparison below.
        # (Every e_L/c entry compared here is a J3 parameter, so its whole reference is the J3 share.)
        wf = ham.wavefunction_data
        no_j3 = ham.replace(wavefunction_data=wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=None)))
        _, reference_no_j3 = _gradient_pair(no_j3, r_up, r_dn, None, quantity)
        j3_share = reference - reference_no_j3
    else:
        j3_share = reference
    assert np.any(np.abs(j3_share) > atol + rtol * np.abs(reference)), j3_share
    np.testing.assert_allclose(
        actual,
        reference,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        err_msg=f"XLA analytic {quantity} != autodiff with a complex-MO J3 on a twist ({kind}, rotated={rotated})",
    )
