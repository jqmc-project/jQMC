"""HDF5 reload keeps the fp64 AO basis storage of the in-memory Hamiltonian (issue #124).

AO radial exponents and contraction coefficients are model storage, fp64 in
both precision modes (basis-data storage exemption in ``jqmc/_precision.py``).
A Hamiltonian reloaded from HDF5 must therefore hold exactly the stored fp64
datasets, and in mixed precision it must evaluate the same AO values,
gradients and Laplacians as the in-memory Hamiltonian it was saved from.
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

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from jqmc._checkpoint import load_hamiltonian_from_checkpoint, merge_rank_checkpoints
from jqmc._twist_average import (
    load_shared_jastrow,
    load_twist,
    run_twist_average_optimize,
    save_shared_jastrow,
    save_twist_average_input,
)
from jqmc._twist_checkpoint import load_shared_state, save_shared_state, shared_state_path
from jqmc.atomic_orbital import compute_AOs, compute_AOs_grad, compute_AOs_laplacian
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data, Jastrow_two_body_data, compute_Jastrow_three_body
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
_RADIAL_SUFFIXES = ("/radial/exponents", "/radial/coefficients")
_ROUTES = ("load_from_hdf5", "checkpoint", "load_twist")
_ROUTE_ROOT = {"load_from_hdf5": "", "checkpoint": "hamiltonian_data/", "load_twist": "shared/"}


def _build_hamiltonian(trexio_file):
    """Committed TREXIO SD with an exponential J2 and an AO J3 on the same basis."""
    structure, aos, _, _, geminal, coulomb = read_trexio_file(
        trexio_file=os.path.join(_TREXIO_DIR, trexio_file), store_tuple=True
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp"),
        jastrow_three_body_data=Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos),
    )
    wavefunction = Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal)
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)


def _save_and_reload(hamiltonian, route, path):
    """Write ``hamiltonian`` through one production persistence route and read it back."""
    if route == "load_from_hdf5":
        hamiltonian.save_to_hdf5(str(path))
        return Hamiltonian_data.load_from_hdf5(str(path))
    if route == "checkpoint":
        rank_file = path.parent / "rank0.h5"
        with h5py.File(rank_file, "w"):
            pass
        merge_rank_checkpoints(
            str(path),
            mpi_size=1,
            driver_type="MCMC",
            hamiltonian_data=hamiltonian,
            tmp_pattern=str(path.parent / "rank{rank}.h5"),
        )
        return load_hamiltonian_from_checkpoint(str(path))
    geminal = hamiltonian.wavefunction_data.geminal_data
    save_twist_average_input(
        filepath=str(path),
        template_hamiltonian=hamiltonian,
        k_points=[None],
        weights=[1.0],
        mo_coefficients_up_per_twist=[np.asarray(geminal.orb_data_up_spin.mo_coefficients)],
        mo_coefficients_dn_per_twist=[np.asarray(geminal.orb_data_dn_spin.mo_coefficients)],
    )
    return load_twist(str(path), 0)


def _field(hamiltonian, path):
    value = hamiltonian
    for component in path.split("/"):
        value = value[int(component)] if component.isdigit() else getattr(value, component)
    return np.asarray(value)


def _ao_leaves(hamiltonian):
    wavefunction = hamiltonian.wavefunction_data
    return {
        "up": wavefunction.geminal_data.orb_data_up_spin.aos_data,
        "dn": wavefunction.geminal_data.orb_data_dn_spin.aos_data,
        "j3": wavefunction.jastrow_data.jastrow_three_body_data.orb_data,
    }


def _electron_positions(structure, seed):
    """Five seeded points scattered around the nuclei (no screening)."""
    rng = np.random.default_rng(seed)
    centres = np.asarray(structure._positions_cart_np)[rng.integers(0, len(structure._positions_cart_np), size=5)]
    return centres + rng.normal(scale=0.7, size=(5, 3))


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("trexio_file", ["H_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 124, five nucleus-centred normal(scale=0.7) points; no cond(G) screen or resampling.",
)
def test_hdf5_reload_keeps_fp64_basis_and_bit_identical_aos(tmp_path, trexio_file, route):
    """Reloaded AO radial arrays equal the stored fp64 datasets and give bit-identical AOs.

    The committed Cartesian H_ae_ccpvdz_cart.h5 and spherical H2_ae_ccpvdz_sphe.h5
    SD Hamiltonians (exponential J2, AO J3 on the same basis) are written through
    Hamiltonian_data.save_to_hdf5, the merged MCMC checkpoint writer, or the
    twist-averaged input writer, and read back by the matching production loader.
    Every radial exponents/coefficients dataset, read independently with h5py,
    must be float64 and bit-equal to the in-memory TREXIO arrays, and the loaded
    leaves (SD up/dn and J3) must have the same dtype and bits as the dataset.
    compute_AOs, compute_AOs_grad and compute_AOs_laplacian of the in-memory
    and reloaded leaves must then be bit-identical at five seeded points in the
    active precision mode; no tolerance is involved. A mixed loader that rounds
    the radial arrays to float32 changes the AO values.
    """
    hamiltonian = _build_hamiltonian(trexio_file)
    path = Path(tmp_path) / "reload.h5"
    loaded = _save_and_reload(hamiltonian, route, path)
    root = _ROUTE_ROOT[route]

    with h5py.File(path, "r") as handle:
        raw = {}

        def collect(name, item):
            if isinstance(item, h5py.Dataset) and name.startswith(root) and name.endswith(_RADIAL_SUFFIXES):
                raw[name[len(root) :]] = np.asarray(item)

        handle.visititems(collect)
    assert {name.split("/")[2] for name in raw} >= {"orb_data_up_spin", "orb_data_dn_spin", "jastrow_three_body_data"}
    assert len(raw) == 6, sorted(raw)
    for name, dataset in raw.items():
        original = _field(hamiltonian, name)
        reloaded = _field(loaded, name)
        assert dataset.dtype == np.dtype(np.float64), name
        assert original.dtype == np.dtype(np.float64), name
        np.testing.assert_array_equal(dataset, original, err_msg=name, strict=True)
        np.testing.assert_array_equal(reloaded, dataset, err_msg=name, strict=True)

    r_carts = _electron_positions(hamiltonian.structure_data, seed=124)
    before, after = _ao_leaves(hamiltonian), _ao_leaves(loaded)
    for leaf in before:
        value_before = np.asarray(compute_AOs(before[leaf], r_carts))
        value_after = np.asarray(compute_AOs(after[leaf], r_carts))
        assert value_before.size > 0 and np.all(np.isfinite(value_before)), leaf
        assert np.any(value_before != 0.0), leaf
        np.testing.assert_array_equal(value_after, value_before, err_msg=f"{leaf} AO value", strict=True)
        for axis, grad_before, grad_after in zip(
            "xyz", compute_AOs_grad(before[leaf], r_carts), compute_AOs_grad(after[leaf], r_carts), strict=True
        ):
            grad_before, grad_after = np.asarray(grad_before), np.asarray(grad_after)
            assert np.all(np.isfinite(grad_before)), leaf
            np.testing.assert_array_equal(grad_after, grad_before, err_msg=f"{leaf} AO grad {axis}", strict=True)
        lap_before = np.asarray(compute_AOs_laplacian(before[leaf], r_carts))
        lap_after = np.asarray(compute_AOs_laplacian(after[leaf], r_carts))
        assert np.all(np.isfinite(lap_before)), leaf
        np.testing.assert_array_equal(lap_after, lap_before, err_msg=f"{leaf} AO laplacian", strict=True)


@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checkpoint field/dtype restore; no electron-coordinate evaluation."
)
def test_checkpoint_restore_lifts_legacy_fp32_basis_storage_to_fp64(tmp_path):
    """A checkpoint with float32 radial datasets restores them exactly as float64 storage.

    Checkpoints written while the mixed loader rounded the AO basis to float32
    carry float32 radial exponents/coefficients. The committed
    H_ae_ccpvdz_cart.h5 SD/J2/J3 Hamiltonian is written by the merged MCMC
    checkpoint writer, its six radial datasets are replaced by their float32
    rounding, and load_hamiltonian_from_checkpoint must return float64 leaves
    holding exactly those float32 values (an exact widening), as the HDF5
    loaders do, in either precision mode.
    """
    hamiltonian = _build_hamiltonian("H_ae_ccpvdz_cart.h5")
    path = Path(tmp_path) / "restart.h5"
    _save_and_reload(hamiltonian, "checkpoint", path)
    expected = {}
    with h5py.File(path, "r+") as handle:
        names = []
        handle["hamiltonian_data"].visititems(
            lambda name, item: (
                names.append(name) if isinstance(item, h5py.Dataset) and name.endswith(_RADIAL_SUFFIXES) else None
            )
        )
        for name in names:
            narrowed = np.asarray(handle["hamiltonian_data"][name]).astype(np.float32)
            del handle["hamiltonian_data"][name]
            handle["hamiltonian_data"].create_dataset(name, data=narrowed)
            expected[name] = narrowed.astype(np.float64)
    assert len(expected) == 6, sorted(expected)
    restored = load_hamiltonian_from_checkpoint(str(path))
    for name, reference in expected.items():
        np.testing.assert_array_equal(_field(restored, name), reference, err_msg=name, strict=True)


def _narrow_radial_datasets(group):
    """Replace every radial exponents/coefficients dataset under ``group`` by its float32 rounding.

    This is the pre-#133 mixed-precision layout. Returns the exact float64 widening of each
    narrowed dataset, read back independently with h5py, keyed by its path relative to ``group``.
    """
    names = []
    group.visititems(
        lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) and name.endswith(_RADIAL_SUFFIXES) else None
    )
    widened = {}
    for name in names:
        narrowed = np.asarray(group[name]).astype(np.float32)
        del group[name]
        group.create_dataset(name, data=narrowed)
        assert group[name].dtype == np.dtype(np.float32), name
        widened[name] = np.asarray(group[name]).astype(np.float64)
    return widened


def _with_radial(leaf, exponents, coefficients):
    """``leaf`` with its radial storage replaced by the given arrays (independent reference leaf)."""
    return leaf.replace(radial=leaf.radial.replace(exponents=exponents, coefficients=coefficients))


def _assert_bit_identical_aos(leaf, reference_leaf, r_carts, label):
    value_ref = np.asarray(compute_AOs(reference_leaf, r_carts))
    assert value_ref.size > 0 and np.all(np.isfinite(value_ref)) and np.any(value_ref != 0.0), label
    np.testing.assert_array_equal(np.asarray(compute_AOs(leaf, r_carts)), value_ref, err_msg=f"{label} AO value", strict=True)
    for axis, grad, grad_ref in zip(
        "xyz", compute_AOs_grad(leaf, r_carts), compute_AOs_grad(reference_leaf, r_carts), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(grad), np.asarray(grad_ref), err_msg=f"{label} AO grad {axis}", strict=True)
    np.testing.assert_array_equal(
        np.asarray(compute_AOs_laplacian(leaf, r_carts)),
        np.asarray(compute_AOs_laplacian(reference_leaf, r_carts)),
        err_msg=f"{label} AO laplacian",
        strict=True,
    )


@pytest.mark.parametrize("loader", ["load_shared_state", "load_shared_jastrow"])
@pytest.mark.parametrize("trexio_file", ["H_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 124, five nucleus-centred normal(scale=0.7) points; no cond(G) screen or resampling.",
)
def test_shared_jastrow_loaders_lift_legacy_fp32_basis_storage_to_fp64(tmp_path, trexio_file, loader):
    """A legacy float32 shared Jastrow file restores fp64 J3 AO storage holding exactly the stored values.

    A legacy shared_state.h5 can hold float32 J3 radial
    exponents/coefficients. The committed
    Cartesian H_ae_ccpvdz_cart.h5 and spherical H2_ae_ccpvdz_sphe.h5 SD
    Hamiltonians with an exponential J2 and an AO J3 are written by
    save_shared_state or save_shared_jastrow, and the J3 radial datasets are
    replaced by their float32 rounding. load_shared_state / load_shared_jastrow
    must return float64 J3 leaves bit-equal to the raw h5py datasets widened
    with astype(float64). compute_AOs, compute_AOs_grad, compute_AOs_laplacian
    and the J3 value of the loaded J3 must be bit-identical (no tolerance) to
    those of the in-memory J3 holding the widened arrays, at five seeded points
    in the active precision mode. Float32 leaves would shift the mixed
    spherical AO values by 1.2e-7.
    """
    hamiltonian = _build_hamiltonian(trexio_file)
    jastrow = hamiltonian.wavefunction_data.jastrow_data
    if loader == "load_shared_state":
        checkpoint_dir = str(Path(tmp_path) / "checkpoint")
        save_shared_state(checkpoint_dir, hamiltonian)
        path = shared_state_path(checkpoint_dir)
    else:
        path = str(Path(tmp_path) / "shared_jastrow.h5")
        save_shared_jastrow(path, jastrow)
    with h5py.File(path, "r+") as handle:
        expected = _narrow_radial_datasets(handle["jastrow_data"])
    assert sorted(expected) == [
        "jastrow_three_body_data/orb_data/radial/coefficients",
        "jastrow_three_body_data/orb_data/radial/exponents",
    ], sorted(expected)

    loaded = load_shared_state(checkpoint_dir) if loader == "load_shared_state" else load_shared_jastrow(path)
    for name, reference in expected.items():
        np.testing.assert_array_equal(_field(loaded, name), reference, err_msg=name, strict=True)

    j3 = jastrow.jastrow_three_body_data
    reference_orb = _with_radial(
        j3.orb_data,
        expected["jastrow_three_body_data/orb_data/radial/exponents"],
        expected["jastrow_three_body_data/orb_data/radial/coefficients"],
    )
    loaded_j3 = loaded.jastrow_three_body_data
    r_carts = _electron_positions(hamiltonian.structure_data, seed=124)
    _assert_bit_identical_aos(loaded_j3.orb_data, reference_orb, r_carts, "j3")
    num_up = hamiltonian.wavefunction_data.geminal_data.num_electron_up
    r_up, r_dn = r_carts[:num_up], r_carts[num_up : num_up + 2]
    j3_value = np.asarray(compute_Jastrow_three_body(loaded_j3, r_up, r_dn))
    j3_reference = np.asarray(compute_Jastrow_three_body(j3.replace(orb_data=reference_orb), r_up, r_dn))
    assert np.isfinite(j3_reference)
    np.testing.assert_array_equal(j3_value, j3_reference, err_msg="J3 value", strict=True)


@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checkpoint field/dtype restore through the resume handoff; no fixed-coordinate evaluation."
)
def test_twist_optimizer_resume_lifts_legacy_fp32_shared_jastrow_basis_storage(tmp_path, monkeypatch):
    """The twist-optimizer resume handoff restores fp64 J3 AO storage from a legacy float32 shared_state.h5.

    The committed H_ae_ccpvdz_cart.h5 SD Hamiltonian with an exponential J2
    and an AO J3 (radial arrays pre-rounded to float32-representable float64
    values) runs one SR step of run_twist_average_optimize with a checkpoint
    directory (J2 optimized; the J3 basis is not). The J3 radial datasets of
    shared_state.h5 are then replaced by their float32 rounding (the legacy
    float32 layout), and the run is resumed for one more SR
    step. The resumed Hamiltonian's J3 radial arrays must be float64 and
    bit-equal to the raw h5py datasets widened with astype(float64), and the
    shared_state.h5 rewritten after the resumed step must store float64 J3
    radial datasets with the same values.
    """
    hamiltonian = _build_hamiltonian("H_ae_ccpvdz_cart.h5")
    j3 = hamiltonian.wavefunction_data.jastrow_data.jastrow_three_body_data
    radial = j3.orb_data.radial
    j3 = j3.replace(
        orb_data=_with_radial(
            j3.orb_data,
            np.asarray(radial.exponents).astype(np.float32).astype(np.float64),
            np.asarray(radial.coefficients).astype(np.float32).astype(np.float64),
        )
    )
    hamiltonian = hamiltonian.replace(
        wavefunction_data=hamiltonian.wavefunction_data.replace(
            jastrow_data=hamiltonian.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3)
        )
    )
    optimize_kwargs = dict(
        num_mcmc_steps=10,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        opt_J1_param=False,
        opt_J2_param=True,
        opt_J3_param=False,
        opt_geminal=False,
        optimizer_kwargs={"method": "sr", "delta": 1e-2, "epsilon": 1e-3},
        num_opt_steps=1,
        wf_dump_freq=1,
    )
    # The single-twist optimizer writes hamiltonian_data_opt_step_*.h5 to the CWD.
    monkeypatch.chdir(tmp_path)
    driver = dict(weights=[1.0], mcmc_seed=12345, num_walkers=2, checkpoint_dir=str(Path(tmp_path) / "checkpoint"))
    run_twist_average_optimize(hamiltonian_list=[hamiltonian], optimize_kwargs=dict(optimize_kwargs), **driver)
    shared_path = shared_state_path(driver["checkpoint_dir"])
    with h5py.File(shared_path, "r+") as handle:
        expected = _narrow_radial_datasets(handle["jastrow_data"])
    assert len(expected) == 2, sorted(expected)

    resumed = run_twist_average_optimize(
        hamiltonian_list=[hamiltonian], optimize_kwargs=dict(optimize_kwargs), resume=True, **driver
    )
    with h5py.File(shared_path, "r") as handle:
        rewritten = {name: np.asarray(handle["jastrow_data"][name]) for name in expected}
    for name, reference in expected.items():
        np.testing.assert_array_equal(
            _field(resumed.wavefunction_data.jastrow_data, name), reference, err_msg=f"resumed {name}", strict=True
        )
        np.testing.assert_array_equal(rewritten[name], reference, err_msg=f"rewritten {name}", strict=True)


@pytest.mark.parametrize("trexio_file", ["H_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="seeded-random",
    reason="PCG64 seed 124, five nucleus-centred normal(scale=0.7) points; no cond(G) screen or resampling.",
)
def test_load_twist_lifts_legacy_fp32_basis_storage_to_fp64(tmp_path, trexio_file):
    """A combined twist input with float32 AO radial datasets loads as exactly widened fp64 storage.

    write_twist_optimized_hamiltonian copies an older float32 shared Jastrow
    verbatim into a combined dump. The committed Cartesian
    H_ae_ccpvdz_cart.h5 and spherical H2_ae_ccpvdz_sphe.h5 SD Hamiltonians
    with an exponential J2 and an AO J3 are written by the twist-averaged input
    writer, and all six /shared radial exponents/coefficients datasets (SD
    up/dn and J3) are replaced by their float32 rounding. load_twist must return
    float64 leaves bit-equal to the raw h5py datasets widened with
    astype(float64), and compute_AOs, compute_AOs_grad and compute_AOs_laplacian
    of each loaded leaf must be bit-identical (no tolerance) to those of the
    in-memory leaf holding the widened arrays, at five seeded points in the
    active precision mode. Removing the alignment from load_twist leaves the
    leaves float32 and fails this test in both modes.
    """
    hamiltonian = _build_hamiltonian(trexio_file)
    path = Path(tmp_path) / "twist.h5"
    _save_and_reload(hamiltonian, "load_twist", path)
    with h5py.File(path, "r+") as handle:
        expected = _narrow_radial_datasets(handle["shared"])
    assert len(expected) == 6, sorted(expected)

    loaded = load_twist(str(path), 0)
    for name, reference in expected.items():
        np.testing.assert_array_equal(_field(loaded, name), reference, err_msg=name, strict=True)

    leaf_prefix = {
        "up": "wavefunction_data/geminal_data/orb_data_up_spin/",
        "dn": "wavefunction_data/geminal_data/orb_data_dn_spin/",
        "j3": "wavefunction_data/jastrow_data/jastrow_three_body_data/orb_data/",
    }
    r_carts = _electron_positions(hamiltonian.structure_data, seed=124)
    in_memory, after = _ao_leaves(hamiltonian), _ao_leaves(loaded)
    for leaf, prefix in leaf_prefix.items():
        (exponents_name,) = [name for name in expected if name.startswith(prefix) and name.endswith("/radial/exponents")]
        reference_leaf = _with_radial(
            in_memory[leaf], expected[exponents_name], expected[exponents_name[: -len("exponents")] + "coefficients"]
        )
        _assert_bit_identical_aos(after[leaf], reference_leaf, r_carts, leaf)
