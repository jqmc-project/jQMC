"""Physical periodic J3 basis updates, domain validation and persistence."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import compute_Jastrow_three_body
from jqmc.jqmc_optimizer import BlockStorageUpdate
from tests.test_jqmc_kernel._periodic_j3_basis_cases import (
    AUDIT_FIXTURE,
    R_DN,
    R_UP,
    gaussian_lattice_reference,
    j3_leaf,
    physical_case,
)
from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block
from tests.test_jqmc_kernel.test_periodic_j3_audit_optimization import _assert_same_basis

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


def update_case(angular, wrapper, flags):
    ham, home = physical_case(angular, wrapper)
    wf = ham.wavefunction_data
    blocks = wf.get_variational_blocks(
        state_token=("periodic-physical-basis", 0),
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=flags in ("matrix", "combined"),
        opt_geminal=False,
        opt_J3_basis_exp=flags in ("exp", "combined"),
        opt_J3_basis_coeff=flags in ("coeff", "combined"),
    )
    updates = tuple(
        BlockStorageUpdate(
            name=b.name,
            state_token=b.state_token,
            delta=np.asarray(b.values) * (-0.3 if b.name.endswith("basis_exp") else 3.0 if flags == "coeff" else 0.2),
        )
        for b in blocks
    )
    candidate = wf.apply_optimization_updates(blocks=blocks, updates=updates, expected_state_token=blocks[0].state_token)
    return ham, ham.replace(wavefunction_data=candidate), home


@pytest.mark.parametrize("angular", ["cart", "sphe"])
@pytest.mark.parametrize("wrapper", ["ao", "mo"])
@pytest.mark.parametrize("flags", ["matrix", "exp", "coeff", "combined"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Fixed off-nucleus coordinates without seed screening."
)
def test_periodic_j3_physical_updates_and_derivatives(angular, wrapper, flags, tmp_path):
    """All flag combinations retain shell/image ties, physical derivatives and exact restart data.

    A contracted s shell and a p shell on each of two real Li supercell sites
    have six physical radial primitives. Both AO/MO and Cartesian/spherical
    forms use the unchanged actual PW determinant. The reference evaluates
    normalized s/p Gaussians over an independent [-6,6]^3 lattice cube and
    explicit electron pairs; its analytic alpha/c derivatives include every
    image and angular component. HDF5 must preserve the actual physical
    parameter identity, order and image envelope exactly within each mode.
    """
    initial, updated, home = update_case(angular, wrapper, flags)
    old_leaf, leaf = j3_leaf(initial), j3_leaf(updated)
    np.testing.assert_array_equal(old_leaf.radial.exponents, home.radial.exponents, strict=True)
    np.testing.assert_array_equal(old_leaf.radial.coefficients, home.radial.coefficients, strict=True)
    assert len(leaf.radial.exponents) == 6
    expected_exp = np.asarray(home.radial.exponents) * (0.7 if flags in ("exp", "combined") else 1)
    expected_coeff = np.asarray(home.radial.coefficients) * (4.0 if flags == "coeff" else 1.2 if flags == "combined" else 1)
    if flags in ("exp", "coeff", "combined"):
        assert leaf._image_indices != old_leaf._image_indices, "This update must exercise image-envelope rebuilding."
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    np.testing.assert_allclose(leaf.radial.exponents, expected_exp, atol=atol, rtol=rtol)
    np.testing.assert_allclose(leaf.radial.coefficients, expected_coeff, atol=atol, rtol=rtol)
    assert np.isfinite(leaf.radial.exponents).all() and (leaf.radial.exponents > 0).all()
    old_j3 = initial.wavefunction_data.jastrow_data.jastrow_three_body_data
    j3 = updated.wavefunction_data.jastrow_data.jastrow_three_body_data
    actual = np.asarray(compute_Jastrow_three_body(j3, R_UP, R_DN))
    expected, derivatives = gaussian_lattice_reference(j3, angular)
    assert np.isfinite(actual) and actual != 0
    assert actual != np.asarray(compute_Jastrow_three_body(old_j3, R_UP, R_DN))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    for field, setter, reference in zip(
        ("exponents", "coefficients"),
        (j3.with_updated_ao_exponents, j3.with_updated_ao_coefficients),
        derivatives,
        strict=True,
    ):
        gradient = jax.grad(lambda x: compute_Jastrow_three_body(setter(x), R_UP, R_DN))(
            jnp.asarray(getattr(leaf.radial, field))
        )
        assert np.isfinite(gradient).all() and np.any(gradient != 0)
        np.testing.assert_allclose(gradient, reference, atol=atol, rtol=rtol)
    path = tmp_path / "physical-update.h5"
    updated.save_to_hdf5(str(path))
    restored = Hamiltonian_data.load_from_hdf5(str(path))
    _assert_same_basis(leaf, j3_leaf(restored))
    restored_j3 = restored.wavefunction_data.jastrow_data.jastrow_three_body_data
    np.testing.assert_array_equal(j3.j_matrix, restored_j3.j_matrix, strict=True)
    np.testing.assert_array_equal(actual, compute_Jastrow_three_body(restored_j3, R_UP, R_DN), strict=True)


@pytest.mark.parametrize("angular", ["cart", "sphe"])
@pytest.mark.parametrize("wrapper", ["ao", "mo"])
@pytest.mark.parametrize(
    "parameter,bad_case",
    [("exp", "zero"), ("exp", "negative"), ("exp", "nan"), ("exp", "infinity"), ("coeff", "nan"), ("coeff", "infinity")],
)
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="Fixed off-nucleus finite-state probes before and after a rejected candidate.",
)
def test_invalid_periodic_basis_candidate_is_atomic(angular, wrapper, parameter, bad_case):
    """Invalid raw arrays and optimizer candidates never mutate a finite accepted basis."""
    ham, _ = physical_case(angular, wrapper)
    wf = ham.wavefunction_data
    leaf = j3_leaf(ham)
    j3 = wf.jastrow_data.jastrow_three_body_data
    old_exp = np.array(leaf.radial.exponents, copy=True)
    old_coeff = np.array(leaf.radial.coefficients, copy=True)
    old_matrix = np.array(j3.j_matrix, copy=True)
    value = np.asarray(compute_Jastrow_three_body(j3, R_UP, R_DN))
    field = "exponents" if parameter == "exp" else "coefficients"
    values = np.array(getattr(leaf.radial, field), copy=True)
    values[0] = {"zero": 0.0, "negative": -1.0, "nan": np.nan, "infinity": np.inf}[bad_case]
    block = make_test_block(f"basis_{parameter}", values, field="real", derivative_policy="direct_real", sync_scope="world")
    with pytest.raises(ValueError, match=r"finite|positive"):
        leaf.apply_basis_block_update(block)
    blocks = wf.get_variational_blocks(
        state_token=("reject-periodic-basis", 0),
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=True,
        opt_geminal=False,
        opt_J3_basis_exp=(parameter == "exp"),
        opt_J3_basis_coeff=(parameter == "coeff"),
    )
    updates = tuple(
        BlockStorageUpdate(
            name=b.name,
            state_token=b.state_token,
            delta=(values - np.asarray(b.values)) if b.name.endswith(f"basis_{parameter}") else np.asarray(b.values) * 0.2,
        )
        for b in blocks
    )
    with pytest.raises(ValueError, match=r"finite|positive"):
        wf.apply_optimization_updates(blocks=blocks, updates=updates, expected_state_token=blocks[0].state_token)
    np.testing.assert_array_equal(j3.j_matrix, old_matrix, strict=True)
    setter = j3.with_updated_ao_exponents if parameter == "exp" else j3.with_updated_ao_coefficients
    with pytest.raises(ValueError, match=r"finite|positive"):
        setter(values)
    # Direct replacement is a low-level pytree operation. Validation must
    # still reject an invalid state before a configured J3 can be accepted.
    bad_leaf = leaf.replace(radial=leaf.radial.replace(**{field: values}))
    bad_orb = j3.orb_data.replace(aos_data=bad_leaf) if wrapper == "mo" else bad_leaf
    with pytest.raises(ValueError, match=r"finite|positive"):
        j3.replace(orb_data=bad_orb).sanity_check()
    np.testing.assert_array_equal(leaf.radial.exponents, old_exp, strict=True)
    np.testing.assert_array_equal(leaf.radial.coefficients, old_coeff, strict=True)
    np.testing.assert_array_equal(compute_Jastrow_three_body(j3, R_UP, R_DN), value, strict=True)


@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="Fixed positions probe coincident but independently owned physical radial entries.",
)
def test_periodic_basis_coincident_parameters_preserve_identity(tmp_path):
    """Equal values in two distinct physical s shells never merge their independent coordinates."""
    ham = Hamiltonian_data.load_from_hdf5(str(AUDIT_FIXTURE))
    leaf = j3_leaf(ham)
    values = np.array(leaf.radial.exponents, copy=True)
    assert values[0] != values[1]
    assert leaf.radial.coefficients[0] == leaf.radial.coefficients[1]
    values[1] = values[0]
    block = make_test_block("j3_basis_exp", values, field="real", derivative_policy="direct_real", sync_scope="world")
    wf = ham.wavefunction_data.replace(jastrow_data=ham.wavefunction_data.jastrow_data.apply_block_update(block))
    candidate = ham.replace(wavefunction_data=wf)
    assert len(j3_leaf(candidate).radial.exponents) == len(values) == 16
    np.testing.assert_array_equal(j3_leaf(candidate).radial.exponents, values, strict=True)
    path = tmp_path / "coincident.h5"
    candidate.save_to_hdf5(str(path))
    loaded = Hamiltonian_data.load_from_hdf5(str(path))
    _assert_same_basis(j3_leaf(candidate), j3_leaf(loaded))


@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="production-driver-generated",
    reason="Actual MCMC current walkers test rejection after candidate construction and before publication.",
)
def test_optimizer_rejects_nonfinite_candidate_observable(tmp_path, monkeypatch):
    """Finite candidate arrays may still overflow; reject using the actual walkers before mutation."""
    from jqmc.jqmc_mcmc import MCMC
    from jqmc.wavefunction import Wavefunction_data

    initial = Hamiltonian_data.load_from_hdf5(str(AUDIT_FIXTURE))
    driver = MCMC(
        hamiltonian_data=initial, num_walkers=2, mcmc_seed=718, num_mcmc_per_measurement=1, comput_log_WF_param_deriv=True
    )
    j3 = initial.wavefunction_data.jastrow_data.jastrow_three_body_data
    old_matrix = np.array(j3.j_matrix, copy=True)
    matrix = np.full(j3.j_matrix.shape, np.finfo(np.float64).max)
    assert np.isfinite(matrix).all()
    bad_wf = initial.wavefunction_data.replace(
        jastrow_data=initial.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=matrix))
    )
    bad_wf.sanity_check()  # Array/domain checks alone cannot detect this case.
    boundary = {}

    def candidate(*args, **kwargs):
        boundary["ham"] = driver.hamiltonian_data
        boundary["token"] = driver.wavefunction_state_token
        boundary["up"] = np.array(driver._MCMC__latest_r_up_carts, copy=True)
        boundary["dn"] = np.array(driver._MCMC__latest_r_dn_carts, copy=True)
        return bad_wf

    monkeypatch.setattr(Wavefunction_data, "apply_optimization_updates", candidate)
    with pytest.raises(ValueError, match="non-finite log magnitude"):
        driver.run_optimize(
            num_mcmc_steps=4,
            num_opt_steps=1,
            num_mcmc_bin_blocks=2,
            dump_wavefunction=False,
            opt_J3_basis_exp=True,
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
        )
    assert driver.hamiltonian_data is boundary["ham"]
    assert driver.wavefunction_state_token == boundary["token"]
    assert driver.i_opt == 0
    np.testing.assert_array_equal(driver._MCMC__latest_r_up_carts, boundary["up"], strict=True)
    np.testing.assert_array_equal(driver._MCMC__latest_r_dn_carts, boundary["dn"], strict=True)
    np.testing.assert_array_equal(
        driver.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_three_body_data.j_matrix, old_matrix, strict=True
    )
    saved = tmp_path / "rejected-candidate.h5"
    driver.hamiltonian_data.save_to_hdf5(str(saved))
    restored = Hamiltonian_data.load_from_hdf5(str(saved))
    _assert_same_basis(j3_leaf(driver.hamiltonian_data), j3_leaf(restored))


@pytest.mark.parametrize("angular", ["cart", "sphe"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="Fixed off-nucleus coordinates validate retained higher-l canonical basis identity through HDF5.",
)
def test_periodic_high_l_preserves_canonical_home_gather(angular, tmp_path):
    """d/f image expansion preserves existing canonical radial identities and normalization.

    Cartesian coefficients carry the usual component factors. Their existing
    molecular grouping is deliberately retained; this test does not claim a
    redesigned one-parameter physical Cartesian shell normalization model.
    """
    from math import factorial
    from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data, _apply_image_expansion
    from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data

    ham = Hamiltonian_data.load_from_hdf5(str(AUDIT_FIXTURE))
    powers = [(x, y, l - x - y) for l in (2, 3) for x in range(l + 1) for y in range(l + 1 - x)]
    momenta = [2] * (6 if angular == "cart" else 5) + [3] * (10 if angular == "cart" else 7)
    rank = len(momenta)
    if angular == "cart":
        coefficients = np.asarray([np.sqrt(np.prod([factorial(2 * x) / factorial(x) for x in power])) for power in powers])
    else:
        coefficients = np.ones(rank)
    fields = dict(
        structure_data=ham.structure_data,
        nucleus_index=(0,) * rank,
        num_ao=rank,
        num_ao_prim=rank,
        orbital_indices=tuple(range(rank)),
        exponents=np.asarray([1 + l / 8 for l in momenta]),
        coefficients=coefficients,
        angular_momentums=tuple(momenta),
    )
    if angular == "cart":
        home = AOs_cart_gto_data.build(
            **fields,
            polynominal_order_x=tuple(p[0] for p in powers),
            polynominal_order_y=tuple(p[1] for p in powers),
            polynominal_order_z=tuple(p[2] for p in powers),
        )
    else:
        home = AOs_sphe_gto_data.build(**fields, magnetic_quantum_numbers=tuple(range(-2, 3)) + tuple(range(-3, 4)))
    leaf = _apply_image_expansion(home, ham.structure_data)
    np.testing.assert_array_equal(leaf.radial.exponents, home.radial.exponents, strict=True)
    np.testing.assert_array_equal(leaf.radial.coefficients, home.radial.coefficients, strict=True)
    for kernel, h in enumerate(leaf._ragged_kernel_to_home):
        actual = np.asarray(leaf.ao_prim_to_shell_prim)[np.asarray(leaf.orbital_indices) == kernel]
        expected = np.asarray(home.ao_prim_to_shell_prim)[np.asarray(home.orbital_indices) == h]
        np.testing.assert_array_equal(actual, expected, strict=True)
    leaf = leaf._with_updated_basis_parameter("exponents", np.asarray(home.radial.exponents) * 0.9)
    leaf = leaf._with_updated_basis_parameter("coefficients", np.asarray(home.radial.coefficients) * 1.1)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(leaf, random_init=True, random_scale=0.02, seed=41)
    candidate = ham.replace(
        wavefunction_data=ham.wavefunction_data.replace(jastrow_data=Jastrow_data(jastrow_three_body_data=j3))
    )
    path = tmp_path / "canonical-high-l.h5"
    candidate.save_to_hdf5(str(path))
    restored = Hamiltonian_data.load_from_hdf5(str(path))
    _assert_same_basis(leaf, j3_leaf(restored))
