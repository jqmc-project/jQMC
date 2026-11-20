"""Phase-free J3 and shared twist Jastrow parameters on fixed skew-cell inputs."""

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

from dataclasses import fields, is_dataclass

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gdd.engine.fixture_observations import observe_hamiltonian
from jqmc._precision import get_tolerance_min
from jqmc._twist_average import (
    load_shared_jastrow,
    load_twist,
    run_twist_average_optimize,
    save_shared_jastrow,
    save_twist_average_input,
)
from jqmc._twist_checkpoint import load_shared_state, save_shared_state, twist_checkpoint_path
from jqmc.hamiltonians import _save_dataclass_to_hdf5
from jqmc.jastrow_factor import (
    Jastrow_DTN_data,
    Jastrow_three_body_data,
    _phase_free_j3_orbitals,
    compute_Jastrow_part,
    compute_Jastrow_three_body,
)
from jqmc.jqmc_tool import build_jastrow_three_body_for_basis
from jqmc.structure import Structure_data
from tests.canonical_wavefunction_fixtures import _mos, _orbital, build_canonical_hamiltonian

jax.config.update("jax_enable_x64", True)
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_UP = np.asarray([[0.21, 0.34, 0.17], [-0.31, 0.27, 0.43]])
_DN = np.asarray([[0.35, -0.22, 0.19], [-0.16, -0.38, 0.29]])
_SHIFTS = np.asarray([[1, 0, 0], [0, 1, -1], [2, -1, 1]])
_K = (0.25, 0.125, 0.0)
_KINDS = (
    "pgto-cartesian-complex",
    "pgto-spherical-complex",
    "plane-wave-complex",
    "bspline-complex",
    "pgto-plane-wave-complex",
)


def _structure():
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3)),
        atomic_numbers=(4,),
        element_symbols=("Be",),
        atomic_labels=("Be1",),
        vec_a=np.asarray([6.0, 0.0, 0.0]),
        vec_b=np.asarray([1.0, 7.0, 0.0]),
        vec_c=np.asarray([0.5, 1.5, 8.0]),
    )


def _leaves(orbital):
    ao = getattr(orbital, "aos_data", orbital)
    children = getattr(ao, "children", ())
    return tuple(child for item in children for child in _leaves(item)) if children else (ao,)


def _assert_phase_free(j3):
    for leaf in _leaves(j3.orb_data):
        k = getattr(leaf, "k_point", None)
        assert k is None or np.array_equal(np.asarray(k), np.zeros(3))
        phase = getattr(leaf, "_image_phase_factors", None)
        if phase is not None:
            np.testing.assert_array_equal(np.asarray(phase), np.ones_like(phase))


def _unvalidated_j3(j3, orb_data):
    """Deliberately invalid J3 that skips construction validation.

    Construction rejects a phase-bearing basis, so a test of an INDEPENDENT
    downstream guard (writer, loader, fixture observer) needs this explicit
    bypass to hand that guard the invalid payload it must reject.
    """
    bad = object.__new__(type(j3))
    for field in fields(j3):
        object.__setattr__(bad, field.name, orb_data if field.name == "orb_data" else getattr(j3, field.name))
    return bad


def _j3_value(j3, up, dn):
    return np.asarray(compute_Jastrow_three_body(j3, jnp.asarray(up), jnp.asarray(dn)))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_converter_builds_live_phase_free_j3_without_changing_determinant(kind):
    """The real converter strips J3 Bloch phase, preserving the determinant's nonzero twist.

    Five orbital families, deterministic nonzero symmetric J matrices and explicit
    two-up/two-down positions in a triclinic cell activate the periodic identity.
    Fixed lattice translations must preserve J3 at the active precision's strict
    AO/Jastrow policy; the original determinant MO coefficients and k are unchanged.
    """
    structure = _structure()
    ao = _orbital(kind, structure)
    mo = _mos(ao, 619)
    coefficients = np.asarray(mo.mo_coefficients).copy()
    j3 = build_jastrow_three_body_for_basis(ao, mo, "mo")
    j3 = j3.replace(j_matrix=np.full(j3.j_matrix.shape, 0.03, dtype=j3.j_matrix.dtype))
    j3.sanity_check()
    _assert_phase_free(j3)
    for leaf in _leaves(mo):
        np.testing.assert_array_equal(np.asarray(leaf.k_point), _K)
    np.testing.assert_array_equal(mo.mo_coefficients, coefficients)
    np.testing.assert_array_equal(j3.orb_data.mo_coefficients, coefficients)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "jastrow_eval"), "strict")
    reference = _j3_value(j3, _UP, _DN)
    assert np.all(np.isfinite(reference)) and np.any(np.abs(reference) > atol)
    for n in _SHIFTS:
        up = _UP.copy()
        up[0] += n @ np.asarray(structure.cell)
        np.testing.assert_allclose(_j3_value(j3, up, _DN), reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("kind", ["pgto-cartesian-complex", "pgto-spherical-complex"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_converter_ao_j3_is_phase_free_and_periodic(kind):
    """AO-full conversion retile uses Gamma images, not the determinant's image phases."""
    structure = _structure()
    ao = _orbital(kind, structure)
    j3 = build_jastrow_three_body_for_basis(ao, _mos(ao, 619), "ao-full")
    j3 = j3.replace(j_matrix=np.full(j3.j_matrix.shape, 0.03, dtype=j3.j_matrix.dtype))
    j3.sanity_check()
    _assert_phase_free(j3)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval"), "strict")
    reference = _j3_value(j3, _UP, _DN)
    assert np.all(np.isfinite(reference)) and np.any(np.abs(reference) > atol)
    for n in _SHIFTS:
        up = _UP.copy()
        up[0] += n @ np.asarray(structure.cell)
        np.testing.assert_allclose(_j3_value(j3, up, _DN), reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Reject invalid orbital metadata before evaluation")
def test_direct_j3_builder_rejects_nonzero_bloch_phase(kind):
    """Explicit phase-bearing J3 inputs are rejected, not silently reinterpreted as periodic."""
    with pytest.raises(ValueError, match=r"phase|[Bb]loch|[Gg]amma|k.point"):
        Jastrow_three_body_data.build(orb_data=_mos(_orbital(kind, _structure()), 619), random_init=True, seed=741)


@pytest.mark.parametrize("path", ["direct", "init", "replace"])
@pytest.mark.parametrize("fault", ["nonzero-k", "stale-image-phase"])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Reject invalid orbital metadata before evaluation")
def test_every_j3_construction_path_rejects_phase_bearing_basis(path, fault):
    """Direct construction, init_jastrow_three_body_data and replace reject a phase-bearing J3 basis.

    Bypassing sanity_check on these paths would let a Bloch-phased J3 (the
    lithium_k025 j123ao fixtures) reach the Rust home-cell wrap and the
    unwrapped XLA image sum, which silently disagree. The spherical complex
    pGTO leaf at the nonzero twist, and the same leaf at k=0 with stale image
    phases, must be rejected on every path; its phase-free shared copy is
    accepted and passes sanity_check.
    """
    orbital = _orbital("pgto-spherical-complex", _structure())
    assert np.any(np.asarray(orbital.k_point) != 0)
    if fault == "stale-image-phase":
        orbital = orbital.replace(k_point=(0.0, 0.0, 0.0))
        phases = orbital._image_phase_factors
        assert isinstance(phases, np.ndarray) and np.any(phases != 1)
    j_matrix = np.zeros((orbital._num_orb, orbital._num_orb + 1))
    valid = Jastrow_three_body_data(orb_data=_phase_free_j3_orbitals(orbital), j_matrix=j_matrix)
    valid.sanity_check()
    _assert_phase_free(valid)
    with pytest.raises(ValueError, match=r"phase|[Bb]loch|[Gg]amma|k.point"):
        if path == "direct":
            Jastrow_three_body_data(orb_data=orbital, j_matrix=j_matrix)
        elif path == "init":
            Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=orbital)
        else:
            valid.replace(orb_data=orbital)


def _shared_template(representation="AO"):
    ham = build_canonical_hamiltonian("WF-PATTERN-006")
    dtn = Jastrow_DTN_data.build(
        structure_data=ham.structure_data,
        z_eff_per_species=(4.0,),
        pseudized_per_species=(True,),
        use_u=True,
        use_chi=True,
        use_f=True,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        random_init=True,
        seed=852,
    )
    jas = ham.wavefunction_data.jastrow_data.replace(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_dtn_data=dtn,
    )
    if representation == "MO":
        mo = _mos(jas.jastrow_three_body_data.orb_data, 619)
        mo = type(mo).build(aos_data=mo.aos_data, mo_coefficients=np.asarray(mo.mo_coefficients)[:3].copy())
        assert mo.mo_coefficients.shape == (3, 4)
        assert np.any(np.asarray(mo.mo_coefficients).imag != 0.0)
        jas = jas.replace(jastrow_three_body_data=Jastrow_three_body_data.build(orb_data=mo, random_init=True, seed=741))
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jas))


def _write_twists(path, ham):
    gem = ham.wavefunction_data.geminal_data
    up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
    dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
    save_twist_average_input(
        str(path),
        ham,
        [None, _K, tuple(-x for x in _K)],
        [1.0, 2.0, 1.0],
        [up.real, up, up],
        [dn.real, dn, dn],
        lambda_matrix_per_twist=[np.asarray(gem.lambda_matrix)] * 3,
    )


def _assert_same_dataclass(actual, expected):
    """Compare all stored fields, including static metadata omitted from a JAX pytree."""
    if is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_same_dataclass(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            _assert_same_dataclass(a, e)
    elif expected is None or isinstance(expected, (str, bool, int, float)):
        assert actual == expected
    else:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _assert_same_shared(actual, expected):
    _assert_phase_free(actual.jastrow_three_body_data)
    np.testing.assert_array_equal(actual.jastrow_three_body_data.j_matrix, expected.jastrow_three_body_data.j_matrix)
    _assert_same_dataclass(actual.jastrow_three_body_data.orb_data, expected.jastrow_three_body_data.orb_data)
    a, e = actual.jastrow_dtn_data, expected.jastrow_dtn_data
    _assert_same_dataclass(a, e)
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "jastrow_eval"), "strict")
    for n in (np.zeros(3, dtype=int), *_SHIFTS):
        up = _UP.copy()
        up[0] += n @ np.asarray(e.structure_data.cell)
        np.testing.assert_allclose(
            compute_Jastrow_part(actual, jnp.asarray(up), jnp.asarray(_DN)),
            compute_Jastrow_part(expected, jnp.asarray(_UP), jnp.asarray(_DN)),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )


@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
@pytest.mark.parametrize("representation", ["AO", "MO"])
def test_gamma_and_signed_twist_inputs_keep_one_live_shared_jastrow(representation, tmp_path):
    """Gamma/+k/-k loaders retain identical nonzero J3 and DTN, with distinct determinant phases."""
    ham = _shared_template(representation)
    path = tmp_path / "twists.h5"
    _write_twists(path, ham)
    expected = ham.wavefunction_data.jastrow_data
    assert np.any(np.asarray(expected.jastrow_three_body_data.j_matrix) != 0.0)
    for field in ("alpha_indep", "beta_indep", "gamma_indep"):
        assert np.any(np.asarray(getattr(expected.jastrow_dtn_data, field)) != 0.0)
    # The combined-input loader keeps the radial basis as fp64 model storage
    # in both precision modes (issue #124, IMPL-HAM-HDF5-BASIS-FP64-STORAGE-001).
    # Declare that exact dtype, rather than introducing a numerical tolerance.
    j3 = expected.jastrow_three_body_data
    orbital = j3.orb_data
    ao = getattr(orbital, "aos_data", orbital)
    radial = ao.radial.replace(
        exponents=np.asarray(ao.radial.exponents, dtype=np.float64),
        coefficients=np.asarray(ao.radial.coefficients, dtype=np.float64),
    )
    ao = ao.replace(radial=radial)
    orbital = orbital.replace(aos_data=ao) if representation == "MO" else ao
    expected = expected.replace(jastrow_three_body_data=j3.replace(orb_data=orbital))
    first_shared = None
    for index, k in enumerate((None, _K, tuple(-x for x in _K))):
        loaded = load_twist(str(path), index)
        _assert_same_shared(loaded.wavefunction_data.jastrow_data, expected)
        if first_shared is None:
            first_shared = loaded.wavefunction_data.jastrow_data
        else:
            _assert_same_shared(loaded.wavefunction_data.jastrow_data, first_shared)
        for leaf in _leaves(loaded.wavefunction_data.geminal_data.orb_data_up_spin):
            actual_k = getattr(leaf, "k_point", None)
            np.testing.assert_array_equal(np.zeros(3) if actual_k is None else actual_k, np.zeros(3) if k is None else k)


@pytest.mark.parametrize("storage", ["shared-jastrow", "optimizer-checkpoint"])
@pytest.mark.parametrize("representation", ["AO", "MO"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_shared_jastrow_update_and_restart_keep_periodic_values(storage, representation, tmp_path):
    """A deterministic world-scoped J3/DTN update survives both real shared-state restart loaders.

    Optimizer blocks are updated without Monte Carlo sampling. The nonzero J3
    matrix and DTN coefficients remain identical; fixed cutoff values and the
    phase-free basis survive serialization, including translated out-of-cell
    evaluations that require pGTO image reconstruction.
    """
    ham = _shared_template(representation)
    wf = ham.wavefunction_data
    blocks = wf.get_variational_blocks(state_token=("shared-jastrow", 0), opt_geminal=False)
    assert {block.name for block in blocks} == {"j3_matrix", "j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"}
    assert all(block.sync_scope == "world" for block in blocks)
    updated_wf = wf.apply_block_updates(blocks, np.full(sum(block.size for block in blocks), 0.002), learning_rate=1.0)
    updated = updated_wf.jastrow_data
    updated.sanity_check()
    if representation == "MO":
        coefficients = np.asarray(updated.jastrow_three_body_data.orb_data.mo_coefficients)
        assert coefficients.dtype == np.complex128
        assert np.any(coefficients.imag != 0.0)
    assert not np.array_equal(updated.jastrow_three_body_data.j_matrix, wf.jastrow_data.jastrow_three_body_data.j_matrix)
    for field in ("alpha_indep", "beta_indep", "gamma_indep"):
        assert not np.array_equal(getattr(updated.jastrow_dtn_data, field), getattr(wf.jastrow_data.jastrow_dtn_data, field))
    for field in ("L_u", "L_chi", "L_f", "cutoff_scale"):
        np.testing.assert_array_equal(
            getattr(updated.jastrow_dtn_data, field), getattr(wf.jastrow_data.jastrow_dtn_data, field)
        )
    if storage == "shared-jastrow":
        path = tmp_path / "shared.h5"
        save_shared_jastrow(str(path), updated)
        loaded = load_shared_jastrow(str(path))
    else:
        save_shared_state(str(tmp_path), ham.replace(wavefunction_data=updated_wf))
        loaded = load_shared_state(str(tmp_path))
    _assert_same_shared(loaded, updated)


@pytest.mark.parametrize("storage", ["combined-input", "shared-jastrow", "optimizer-checkpoint"])
@pytest.mark.parametrize("boundary", ["writer", "loader"])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Validate serialized orbital metadata, not coordinates")
def test_shared_jastrow_persistence_rejects_phase_bearing_payload(storage, boundary, tmp_path):
    """Shared writers and restart loaders reject a live nonzero-k J3 payload at every storage boundary.

    Loader cases first write a valid file and replace only its serialized J3
    orbital with a deliberately invalid Bloch orbital. This tests validation of
    persisted input independently of the writer's guard, without silent migration.
    """
    ham = _shared_template()
    jas = ham.wavefunction_data.jastrow_data
    bad_j3 = _unvalidated_j3(jas.jastrow_three_body_data, _orbital("pgto-spherical-complex", ham.structure_data))
    bad_jas = jas.replace(jastrow_three_body_data=bad_j3)
    bad_ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=bad_jas))
    path = tmp_path / ("shared_state.h5" if storage == "optimizer-checkpoint" else "input.h5")

    def write(value):
        if storage == "combined-input":
            _write_twists(path, value)
        elif storage == "shared-jastrow":
            save_shared_jastrow(str(path), value.wavefunction_data.jastrow_data)
        else:
            save_shared_state(str(tmp_path), value)

    if boundary == "writer":
        with pytest.raises(ValueError, match=r"phase|[Bb]loch|[Gg]amma|k.point"):
            write(bad_ham)
        return

    write(ham)
    group_path = "shared/wavefunction_data/jastrow_data" if storage == "combined-input" else "jastrow_data"
    with h5py.File(path, "r+") as stream:
        j3_group = stream[group_path]["jastrow_three_body_data"]
        del j3_group["orb_data"]
        _save_dataclass_to_hdf5(j3_group.create_group("orb_data"), bad_j3.orb_data)
    with pytest.raises(ValueError, match=r"phase|[Bb]loch|[Gg]amma|k.point"):
        if storage == "combined-input":
            load_twist(str(path), 1)
        elif storage == "shared-jastrow":
            load_shared_jastrow(str(path))
        else:
            load_shared_state(str(tmp_path))


@pytest.mark.parametrize("fault", ["nonzero-k", "stale-image-phase"])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Observe orbital phase metadata without evaluation")
def test_fixture_observer_rejects_phase_bearing_j3_without_builder_validation(fault):
    """The fixture observer independently rejects nonzero J3 twist and stale phase factors at zero k.

    A valid canonical Hamiltonian is observed first; replacing only its J3 leaf
    bypasses constructor validation so the observer's own fail-closed guard must
    detect either defect. The determinant and untouched reference remain valid.
    """
    ham = build_canonical_hamiltonian("WF-PATTERN-006")
    reference = observe_hamiltonian(ham)
    orbital = _orbital("pgto-spherical-complex", ham.structure_data)
    if fault == "stale-image-phase":
        orbital = orbital.replace(k_point=(0.0, 0.0, 0.0))
        assert np.any(np.asarray(orbital._image_phase_factors) != 1.0)
    jas = ham.wavefunction_data.jastrow_data
    bad_jas = jas.replace(jastrow_three_body_data=_unvalidated_j3(jas.jastrow_three_body_data, orbital))
    faulty = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=bad_jas))
    with pytest.raises(AssertionError, match=r"phase|[Bb]loch"):
        observe_hamiltonian(faulty)
    assert observe_hamiltonian(ham) == reference


@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Reject a missing shared checkpoint before sampling")
def test_twist_optimizer_resume_requires_canonical_shared_state_before_driver_load(tmp_path):
    """An existing twist checkpoint cannot resume without its authoritative shared Jastrow file.

    A real phase-free Hamiltonian and COMM_SELF reach the actual restart driver.
    The empty twist file is intentionally unreadable: the missing-shared-state
    error must occur before reading it or constructing a sampler.
    """
    from pathlib import Path

    from mpi4py import MPI

    ham = _shared_template()
    Path(twist_checkpoint_path(str(tmp_path), 0)).touch()
    with pytest.raises(FileNotFoundError, match="canonical shared Jastrow"):
        run_twist_average_optimize(
            [ham],
            [1.0],
            world_comm=MPI.COMM_SELF,
            checkpoint_dir=str(tmp_path),
            resume=True,
            num_walkers=1,
            optimize_kwargs={"num_opt_steps": 1, "num_mcmc_steps": 1},
        )
