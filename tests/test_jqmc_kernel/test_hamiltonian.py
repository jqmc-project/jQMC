"""collections of unit tests."""

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
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

pytestmark = [pytest.mark.with_chdir]
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import AO_IMAGE_TRUNCATION_ATOL
from jqmc.atomic_orbital import (
    AOs_bspline_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    _apply_image_expansion,
)
from jqmc.determinant import (
    AGP_data,
    Geminal_data,
    SD_data,
    compute_geminal_all_elements,
)
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
from jqmc.hamiltonians import (
    Hamiltonian_data,
    _compute_local_energy_auto,
    compute_local_energy,
    compute_local_energy_fast,
    get_num_electrons,
)
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _phase_free_j3_orbitals,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    evaluate_wavefunction,
)

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


test_trexio_files = ["H2_ecp_ccpvtz_cart.h5", "H_ecp_ccpvqz.h5"]

PERIODIC_FIXTURE = "silicon.h5"
PERIODIC_FIXTURE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", PERIODIC_FIXTURE)


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def assert_dataclasses_equal(d1, d2):
    """Helper to compare two dataclasses (or pytrees) with numpy arrays.

    Numeric arrays are compared with the HDF5 round-trip tolerance from
    :py:func:`jqmc._setting.get_tol_hdf5_roundtrip` so the assertion stays
    correct across precision modes -- ``Hamiltonian_data.load_from_hdf5``
    runs ``_apply_{ao_basis,geminal,dtn}_dtype_alignment_post_load`` which
    casts loaded values to the active precision zone, so bit-exact
    ``np.testing.assert_array_equal`` would fail under
    ``--precision-mode=mixed`` even when the round-trip is correct.
    """
    import dataclasses
    from collections.abc import Mapping

    from jqmc._setting import get_tol_hdf5_roundtrip

    atol, rtol = get_tol_hdf5_roundtrip("strict")

    if dataclasses.is_dataclass(d1) and dataclasses.is_dataclass(d2):
        assert d1.__class__ == d2.__class__
        fields = dataclasses.fields(d1)
        if fields:
            for field in fields:
                val1 = getattr(d1, field.name)
                val2 = getattr(d2, field.name)
                assert_dataclasses_equal(val1, val2)
    elif isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        assert len(d1) == len(d2)
        if d1:
            for v1, v2 in zip(d1, d2, strict=True):
                assert_dataclasses_equal(v1, v2)
    elif isinstance(d1, Mapping) and isinstance(d2, Mapping):
        assert d1.keys() == d2.keys()
        if d1:
            for k in d1:
                assert_dataclasses_equal(d1[k], d2[k])
    elif isinstance(d1, (np.ndarray, jnp.ndarray, jax.Array)) or isinstance(d2, (np.ndarray, jnp.ndarray, jax.Array)):
        # Handle JAX arrays by converting to numpy
        a1 = np.asarray(d1)
        a2 = np.asarray(d2)

        # Check for string arrays (which might be bytes vs str)
        if a1.dtype.kind in ("S", "U") and a2.dtype.kind in ("S", "U"):
            np.testing.assert_array_equal(a1.astype(str), a2.astype(str))
        elif a1.dtype.kind in ("i", "u", "b") and a2.dtype.kind in ("i", "u", "b"):
            # Integer / bool fields: bit-exact comparison still valid post-load
            # (the dtype alignment hook only touches floating fields).
            np.testing.assert_array_equal(a1, a2)
        else:
            np.testing.assert_allclose(a1, a2, atol=atol, rtol=rtol)
    else:
        assert d1 == d2


class TestHamiltonianHDF5:
    @pytest.mark.light
    def test_hamiltonian_hdf5_rejects_missing_sd_provider(self, tmp_path):
        """An occupied SD requires both explicitly typed orbital providers."""
        import h5py

        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "H2_ae_ccpvdz_cart.h5",
            ),
            store_tuple=True,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=geminal_data,
            ),
        )
        h5_file = tmp_path / "legacy_sd.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        with h5py.File(h5_file, "r+") as h5:
            group = h5["wavefunction_data/geminal_data"]
            del group["orb_data_up_spin"]

        with pytest.raises(ValueError, match="requires the orb_data_up_spin orbital provider"):
            Hamiltonian_data.load_from_hdf5(str(h5_file))

    @pytest.mark.light
    @pytest.mark.parametrize(
        "corruption",
        ["lambda-matrix", "lambda-diag", "lambda-unpaired", "extra-mo-row-metadata"],
    )
    def test_hamiltonian_hdf5_rejects_noncanonical_sd_payload(self, tmp_path, corruption):
        """Occupied SD input must reject lambda payloads and extra rows."""
        import h5py

        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "H2_ae_ccpvdz_cart.h5",
            ),
            store_tuple=True,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_data),
        )
        h5_file = tmp_path / f"noncanonical_sd_{corruption}.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        with h5py.File(h5_file, "r+") as h5:
            group = h5["wavefunction_data/geminal_data"]
            assert "sd_representation" not in group.attrs
            if corruption.startswith("lambda-"):
                dataset_name = corruption.replace("-", "_")
                group.create_dataset(dataset_name, data=np.eye(geminal_data.num_occ_up))
            else:
                up_group = group["orb_data_up_spin"]
                up_group.attrs["num_mo"] = int(geminal_data.num_occ_up) + 1

        match = "must not contain lambda datasets" if corruption.startswith("lambda-") else "non-canonical up-spin MO row count"
        with pytest.raises(ValueError, match=match):
            Hamiltonian_data.load_from_hdf5(str(h5_file))

    @pytest.mark.light
    def test_hamiltonian_hdf5_loader_rejects_inconsistent_time_reversal_sd(self, tmp_path):
        """A forged tied-spin SD must be rejected at the single-file loader boundary."""
        import h5py

        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "H2_ae_ccpvdz_cart.h5",
            ),
            store_tuple=True,
        )
        assert geminal_data.time_reversal
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=geminal_data,
            ),
        )
        h5_file = tmp_path / "inconsistent_time_reversal_sd.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        with h5py.File(h5_file, "r+") as h5:
            coefficients = h5["wavefunction_data/geminal_data/orb_data_dn_spin/mo_coefficients"]
            coefficients[...] = coefficients[...] + 0.125

        with pytest.raises(ValueError, match="time_reversal=True requires down-spin"):
            Hamiltonian_data.load_from_hdf5(str(h5_file))

    @pytest.mark.light
    def test_hamiltonian_hdf5_writer_rejects_inconsistent_time_reversal_sd(self, tmp_path):
        """The single-file writer validates tied-spin SD before persistence."""
        structure_data, _, _, _, tied_sd, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "H2_ae_ccpvdz_cart.h5",
            ),
            store_tuple=True,
        )
        forged_sd = tied_sd.replace(
            orb_data_dn_spin=tied_sd.orb_data_dn_spin.replace(
                mo_coefficients=np.ascontiguousarray(np.asarray(tied_sd.orb_data_dn_spin.mo_coefficients) + 0.125)
            )
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=forged_sd,
            ),
        )

        with pytest.raises(ValueError, match="time_reversal=True requires down-spin"):
            hamiltonian_data.save_to_hdf5(str(tmp_path / "forged_time_reversal_sd.h5"))

    @pytest.mark.light
    def test_hamiltonian_hdf5_roundtrip_accepts_independent_complex_sd(self, tmp_path):
        """The loader must not impose the tied-spin relation when it is disabled."""
        from jqmc._setting import get_tol_hdf5_roundtrip

        structure_data, _, _, _, tied_sd, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                "lithium_k025.h5",
            ),
            store_tuple=True,
        )
        independent_dn = tied_sd.orb_data_dn_spin.replace(
            mo_coefficients=np.ascontiguousarray(np.asarray(tied_sd.orb_data_dn_spin.mo_coefficients) * np.exp(0.2j))
        )
        independent_sd = SD_data.build(
            num_electron_up=tied_sd.num_electron_up,
            num_electron_dn=tied_sd.num_electron_dn,
            mos_up=tied_sd.orb_data_up_spin,
            mos_dn=independent_dn,
            time_reversal=False,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(
                jastrow_data=Jastrow_data(),
                geminal_data=independent_sd,
            ),
        )
        h5_file = tmp_path / "independent_complex_sd.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))

        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))
        loaded_sd = loaded.wavefunction_data.geminal_data
        assert isinstance(loaded_sd, SD_data)
        assert loaded_sd.time_reversal is False
        atol, rtol = get_tol_hdf5_roundtrip("strict")
        np.testing.assert_allclose(
            np.asarray(loaded_sd.orb_data_up_spin.mo_coefficients),
            np.asarray(independent_sd.orb_data_up_spin.mo_coefficients),
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.asarray(loaded_sd.orb_data_dn_spin.mo_coefficients),
            np.asarray(independent_sd.orb_data_dn_spin.mo_coefficients),
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", test_trexio_files)
    @pytest.mark.parametrize("use_1b", [True, False])
    @pytest.mark.parametrize("use_2b", [True, False])
    @pytest.mark.parametrize("use_3b", [True, False])
    @pytest.mark.parametrize("use_nn", [False])
    @pytest.mark.parametrize("geminal_type", ["mo", "ao"])
    def test_hamiltonian_hdf5(self, trexio_file, use_1b, use_2b, use_3b, use_nn, geminal_type, tmp_path):
        """Test HDF5 reading/writing for Hamiltonian_data."""
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        if geminal_type == "ao":
            geminal_data = Geminal_data.convert_from_MOs_to_AOs(geminal_mo_data)
        else:
            geminal_data = geminal_mo_data

        jastrow_one_body_data = None
        if use_1b:
            # For ECPs, we need core_electrons.
            # The test files seem to be ECP files (H2_ecp..., H_ecp...).
            # Assuming 0 core electrons for Hydrogen if not specified, but let's check if we can deduce it.
            # Or just use dummy values for testing serialization.
            core_electrons = tuple([0.0] * len(structure_data.positions))
            jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0, structure_data=structure_data, core_electrons=core_electrons, jastrow_1b_type="pade"
            )

        jastrow_two_body_data = None
        if use_2b:
            jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=1.0, jastrow_2b_type="exp"
            )

        jastrow_three_body_data = None
        if use_3b:
            # We need orb_data. aos_data from read_trexio_file should work.
            # If aos_data is None or not compatible, we might need to create a dummy one.
            # But read_trexio_file should return valid aos_data.
            # However, Jastrow_three_body_data expects AOs_sphe_gto_data, AOs_cart_gto_data or MOs_data.
            # Let's assume aos_data is compatible.
            # If aos_data is None (which shouldn't happen for these files), we create a dummy.
            if aos_data is None:
                raise ValueError("aos_data is required for Jastrow_three_body_data but is None.")
            jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos_data)

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_one_body_data,
            jastrow_two_body_data=jastrow_two_body_data,
            jastrow_three_body_data=jastrow_three_body_data,
        )

        jastrow_data.sanity_check()

        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)
        wavefunction_data.sanity_check()

        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        # Save to HDF5
        h5_file = tmp_path / "test_hamiltonian.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))

        # Load from HDF5
        loaded_hamiltonian_data = Hamiltonian_data.load_from_hdf5(str(h5_file))

        # Compare
        # Note: Direct equality (==) on dataclasses with numpy arrays is ambiguous in boolean context.
        # We use a helper to compare leaves.
        assert_dataclasses_equal(hamiltonian_data, loaded_hamiltonian_data)


class TestLocalEnergyFast:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "lithium_k025.h5"])
    def test_direct_sd_local_energy_matches_identity_geminal(self, trexio_file):
        """Real/complex direct SD and its old identity oracle have equal e_L."""
        structure_data, _, _, _, sd, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trexio_example_files",
                trexio_file,
            ),
            store_tuple=True,
        )
        assert isinstance(sd, SD_data)
        assert sd.num_electron_up == sd.num_electron_dn
        n_occ = sd.num_electron_up
        legacy = AGP_data.build(
            num_electron_up=n_occ,
            num_electron_dn=n_occ,
            orb_data_up_spin=sd.orb_data_up_spin,
            orb_data_dn_spin=sd.orb_data_dn_spin,
            lambda_matrix=np.eye(n_occ, dtype=sd.orb_data_up_spin.mo_coefficients.dtype),
            time_reversal=sd.time_reversal,
        )
        wf_sd = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=sd)
        ham_sd = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wf_sd,
        )
        ham_legacy = ham_sd.replace(wavefunction_data=wf_sd.replace(geminal_data=legacy))
        rng = np.random.default_rng(4242)
        origin = np.asarray(structure_data.positions[0])
        r_up = jnp.asarray(origin + 0.7 * rng.normal(size=(n_occ, 3)))
        r_dn = jnp.asarray(origin + 0.7 * rng.normal(size=(n_occ, 3)))
        RT = jnp.eye(3, dtype=jnp.float64)

        G_sd = compute_geminal_all_elements(sd, r_up, r_dn)
        G_legacy = compute_geminal_all_elements(legacy, r_up, r_dn)
        e_sd = compute_local_energy(ham_sd, r_up, r_dn, RT)
        e_legacy = compute_local_energy(ham_legacy, r_up, r_dn, RT)
        e_sd_fast = compute_local_energy_fast(ham_sd, r_up, r_dn, RT, jnp.linalg.inv(G_sd))
        e_legacy_fast = compute_local_energy_fast(
            ham_legacy,
            r_up,
            r_dn,
            RT,
            jnp.linalg.inv(G_legacy),
        )
        atol, rtol = get_tolerance_min(
            ("ao_eval", "det_eval", "coulomb", "wf_kinetic", "local_energy"),
            "strict",
        )
        np.testing.assert_allclose(np.asarray(e_sd), np.asarray(e_legacy), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(e_sd_fast), np.asarray(e_legacy_fast), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(e_sd_fast), np.asarray(e_sd), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvtz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_compute_local_energy_fast(self, trexio_file):
        """compute_local_energy_fast must equal compute_local_energy for well-conditioned G."""
        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(geminal_data=geminal_data),
        )
        geminal_data = hamiltonian_data.wavefunction_data.geminal_data
        RT = jnp.eye(3, dtype=jnp.float64)
        rng = np.random.default_rng(42)
        first_nucleus = np.array(hamiltonian_data.structure_data.positions[0])
        n_up = geminal_data.num_electron_up
        n_dn = geminal_data.num_electron_dn

        # e_L crosses ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic/local_energy
        # zones; the achievable agreement is bounded by the weakest (fp32 in mixed).
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "local_energy"),
            "strict",
        )
        for _ in range(10):
            r_up = jnp.array(first_nucleus + rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
            r_dn = jnp.array(first_nucleus + rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)

            G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
            G_inv = jnp.linalg.inv(G)

            e_ref = float(compute_local_energy(hamiltonian_data, r_up, r_dn, RT))
            e_fast = float(compute_local_energy_fast(hamiltonian_data, r_up, r_dn, RT, G_inv))

            assert np.isfinite(e_ref), f"Reference e_L is not finite: {e_ref}"
            assert np.isfinite(e_fast), f"Fast e_L is not finite: {e_fast}"
            np.testing.assert_allclose(
                e_fast,
                e_ref,
                atol=atol,
                rtol=rtol,
                err_msg=f"compute_local_energy_fast={e_fast:.10f} != compute_local_energy={e_ref:.10f}",
            )


def _compare_grad_leaves(
    grad_ref,
    grad_test,
    label,
    atol=None,
    rtol=None,
):
    """Flatten two pytrees and compare every leaf."""
    if atol is None or rtol is None:
        _atol, _rtol = get_tolerance_min(["det_eval", "jastrow_eval"], "strict")
        if atol is None:
            atol = _atol
        if rtol is None:
            rtol = _rtol
    # The optional arguments may only restate the reviewed strict pair; verify
    # the supplied pair, then compare with the same local resolver.
    expected_tolerance = get_tolerance_min(["det_eval", "jastrow_eval"], "strict")
    assert (atol, rtol) == expected_tolerance, f"{label}: unexpected comparison budget"
    atol, rtol = get_tolerance_min(["det_eval", "jastrow_eval"], "strict")
    leaves_ref = jax.tree_util.tree_leaves(grad_ref)
    leaves_tst = jax.tree_util.tree_leaves(grad_test)
    assert len(leaves_ref) == len(leaves_tst), f"{label}: number of leaves differ ({len(leaves_ref)} vs {len(leaves_tst)})"
    assert jax.tree_util.tree_structure(grad_ref) == jax.tree_util.tree_structure(grad_test), f"{label}: pytree differs"
    leaf_paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(grad_ref)[0]]
    for i, (leaf_path, lr, lt) in enumerate(zip(leaf_paths, leaves_ref, leaves_tst)):
        lr = np.asarray(lr)
        lt = np.asarray(lt)
        assert lr.shape == lt.shape, f"{label} leaf {i}: shape {lr.shape} vs {lt.shape}"
        if lr.size == 0:
            # Molecular cells carry zero-length lattice vectors. Their exact
            # (0,) shape is the whole obligation; there is no value to compare.
            assert leaf_path.endswith((".vec_a", ".vec_b", ".vec_c")) and lr.shape == (0,), f"{label}: empty {leaf_path}"
            continue
        max_abs_diff = float(np.max(np.abs(lt - lr)))
        np.testing.assert_allclose(
            lt,
            lr,
            atol=atol,
            rtol=rtol,
            err_msg=f"{label}: gradient mismatch at leaf {i}  (max |diff|={max_abs_diff:.3e})",
        )


def _local_energy_gradient_case(trexio_file):
    """Rebuild the original TREXIO Hamiltonian and PCG64(123) positions."""
    seed = 123
    structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
        store_tuple=True,
    )
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(geminal_data=geminal_data),
    )
    RT = jnp.eye(3, dtype=jnp.float64)
    rng = np.random.default_rng(seed)
    first_nucleus = np.array(hamiltonian_data.structure_data.positions[0])
    n_up = geminal_data.num_electron_up
    n_dn = geminal_data.num_electron_dn

    r_up = jnp.array(first_nucleus + rng.standard_normal((n_up, 3)) * 0.5, dtype=jnp.float64)
    r_dn = jnp.array(first_nucleus + rng.standard_normal((n_dn, 3)) * 0.5, dtype=jnp.float64)

    return hamiltonian_data, r_up, r_dn, RT


def _assert_gradient_tree_structure(grad_ref, grad_test, label):
    """Retain the old non-AD leaf, shape and permitted-empty obligations."""
    leaves_ref = jax.tree_util.tree_leaves(grad_ref)
    leaves_tst = jax.tree_util.tree_leaves(grad_test)
    assert len(leaves_ref) == len(leaves_tst), f"{label}: number of leaves differ ({len(leaves_ref)} vs {len(leaves_tst)})"
    assert jax.tree_util.tree_structure(grad_ref) == jax.tree_util.tree_structure(grad_test), f"{label}: pytree differs"
    leaf_paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(grad_ref)[0]]
    for i, (leaf_path, lr, lt) in enumerate(zip(leaf_paths, leaves_ref, leaves_tst)):
        lr = np.asarray(lr)
        lt = np.asarray(lt)
        assert lr.shape == lt.shape, f"{label} leaf {i}: shape {lr.shape} vs {lt.shape}"
        if lr.size == 0:
            assert leaf_path.endswith((".vec_a", ".vec_b", ".vec_c")) and lr.shape == (0,), f"{label}: empty {leaf_path}"


class TestLocalEnergyGrad:
    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvtz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_grad_compute_local_energy(self, trexio_file):
        """grad(compute_local_energy, argnums=0) must match grad(_compute_local_energy_auto, argnums=0).

        Both functions compute e_L = T + V.  compute_local_energy uses the custom VJP in
        compute_grads_and_laplacian_ln_Det (and _ln_det_bwd), while _compute_local_energy_auto
        uses a fully-automatic Laplacian via JAX second-order AD.  The gradients w.r.t.
        all Hamiltonian pytree leaves (lambda_matrix, Jastrow params, positions, ...) must
        be numerically identical for a well-conditioned geminal matrix in full
        precision. Mixed rebuilds the production value/gradient in full and mixed,
        using spatial- and parameter-derivative fidelity budgets respectively.
        """
        if is_mixed_precision_enabled():

            def analytical(case):
                ham, up, dn, rotation = case
                energy = compute_local_energy(ham, up, dn, rotation)
                gradient = jax.grad(compute_local_energy, argnums=0)(ham, up, dn, rotation)
                return energy, gradient

            mixed, full = mixed_vs_full(lambda: _local_energy_gradient_case(trexio_file), analytical)
            _assert_gradient_tree_structure(full[1], mixed[1], f"mixed/full production gradient [{trexio_file}]")
            blocks = mixed_fidelity_blocks(mixed[0], full[0])
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            blocks = mixed_fidelity_blocks(mixed[1], full[1])
            atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        hamiltonian_data, r_up, r_dn, RT = _local_energy_gradient_case(trexio_file)
        seed = 123

        # Sanity: both forward values must agree.
        atol, rtol = get_tolerance("local_energy", "strict")
        e_auto = float(_compute_local_energy_auto(hamiltonian_data, r_up, r_dn, RT))
        e_custom = float(compute_local_energy(hamiltonian_data, r_up, r_dn, RT))
        np.testing.assert_allclose(
            e_custom,
            e_auto,
            atol=atol,
            rtol=rtol,
            err_msg="forward e_L mismatch",
        )

        # Gradient comparison (w.r.t. full Hamiltonian pytree, argnums=0).
        # Gradients flow through geminal and jastrow zones (float32 in mixed mode),
        # so tolerance is bounded by the weakest zone on the path.
        grad_auto = jax.grad(_compute_local_energy_auto, argnums=0)(hamiltonian_data, r_up, r_dn, RT)
        grad_custom = jax.grad(compute_local_energy, argnums=0)(hamiltonian_data, r_up, r_dn, RT)

        atol_grad, rtol_grad = get_tolerance_min(["det_eval", "jastrow_eval"], "strict")
        _compare_grad_leaves(
            grad_auto,
            grad_custom,
            label=f"grad(compute_local_energy) vs _auto [{trexio_file}, seed={seed}]",
            atol=atol_grad,
            rtol=rtol_grad,
        )


# ---------------------------------------------------------------------------
# Periodic Hamiltonian_data HDF5 round-trip tests over PW / B-spline / pGTO /
# composite AO leaves (dummy instances, no external TREXIO / QE file).
#
# Mirrors the periodic kinetic-energy dummy harness in
# ``tests/test_jqmc_kernel/test_wave_function.py`` so the persistence layer is exercised on the
# same five periodic AO leaf / composite kinds without an external fixture
# dependency. Each kind is wrapped through ``MOs_data`` so the Geminal_data
# path matches the MOs-or-AOs-base branch.
# ---------------------------------------------------------------------------

_PBC_HAM_A = 6.0  # cubic cell edge (Bohr)
_PBC_HAM_BSPLINE_NS = (10, 10, 10)
_PBC_HAM_BSPLINE_ORDER = 5
_PBC_HAM_PERIODIC_KINDS = ["pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"]


def _pbc_ham_cubic_structure(a: float = _PBC_HAM_A) -> Structure_data:
    """Charge-neutral 4-H cubic periodic cell (Z_total = 4 == n_up + n_dn).

    The post-load Ewald rebuild
    (:py:func:`jqmc.hamiltonians._apply_ewald_rebuild_post_load`) calls
    ``build_ewald_data`` with ``n_electron = n_up + n_dn``, which now
    enforces charge neutrality via
    :py:func:`jqmc.coulomb_potential._check_charge_neutrality_for_ewald`.
    Four hydrogen sites (Z=1 each) at fractional ``(0,0,0)``,
    ``(1/2, 0, 0)``, ``(0, 1/2, 0)``, ``(0, 0, 1/2)`` give a charge-neutral
    cell against the 2-up / 2-dn dummy AGP used here.
    """
    cell = np.eye(3, dtype=np.float64) * a
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5 * a, 0.0, 0.0],
            [0.0, 0.5 * a, 0.0],
            [0.0, 0.0, 0.5 * a],
        ],
        dtype=np.float64,
    )
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H", "H", "H", "H"),
        atomic_labels=("H1", "H2", "H3", "H4"),
    )


def _pbc_ham_pw_leaf(structure: Structure_data):
    """Real cos/sin plane-wave leaf with a small G-set (constant + 3 pairs)."""
    recip = structure.recip_cell
    millers = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    g_list: list[np.ndarray] = []
    is_sin: list[float] = []
    for m in millers:
        g = m @ recip
        if np.all(m == 0):
            g_list.append(g)
            is_sin.append(0.0)
        else:
            g_list.append(g)
            is_sin.append(0.0)
            g_list.append(g)
            is_sin.append(1.0)
    g_vectors = np.asarray(g_list, dtype=np.float64)
    phase_is_sin = np.asarray(is_sin, dtype=np.float64)
    return AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)


def _pbc_ham_bspline_bands(frac):
    """Two smooth periodic bands of fractional coords."""
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _pbc_ham_bspline_leaf(structure: Structure_data, order: int = _PBC_HAM_BSPLINE_ORDER):
    """Interpolating B-spline leaf reproducing two smooth periodic bands."""
    ns = _PBC_HAM_BSPLINE_NS
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)
    vals = _pbc_ham_bspline_bands(frac)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), order)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, order)


def _pbc_ham_pgto_leaf(structure: Structure_data):
    """Image-expanded periodic pGTO leaf: one s shell per atom (4 AOs over 4 H)."""
    n_atom = len(structure.atomic_numbers)
    base = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=tuple(range(n_atom)),
        num_ao=n_atom,
        num_ao_prim=n_atom,
        orbital_indices=tuple(range(n_atom)),
        exponents=np.full(n_atom, 0.6, dtype=np.float64),
        coefficients=np.ones(n_atom, dtype=np.float64),
        angular_momentums=tuple([0] * n_atom),
        magnetic_quantum_numbers=tuple([0] * n_atom),
    )
    return _apply_image_expansion(base, structure, AO_IMAGE_TRUNCATION_ATOL)


def _pbc_ham_aos_data(kind: str, structure: Structure_data):
    """Build the AO sub-tree for one periodic kind."""
    if kind == "pw":
        return _pbc_ham_pw_leaf(structure)
    if kind == "bspline":
        return _pbc_ham_bspline_leaf(structure)
    if kind == "pgto":
        return _pbc_ham_pgto_leaf(structure)
    if kind == "pgto+pw":
        return _pbc_ham_pgto_leaf(structure) + _pbc_ham_pw_leaf(structure)
    if kind == "pgto+bspline":
        return _pbc_ham_pgto_leaf(structure) + _pbc_ham_bspline_leaf(structure)
    raise ValueError(f"unknown periodic Hamiltonian kind: {kind}")


def _build_pbc_dummy_hamiltonian(kind: str, seed: int = 2025):
    """Build a minimal periodic dummy ``Hamiltonian_data`` for one AO kind.

    Single-atom hydrogen cubic cell with 2-up / 2-dn electrons (charge
    neutrality is not enforced because ``build_ewald_data`` is called with
    ``n_electron=None``; the dummy fixture is non-physical by design and
    only exists to exercise the save/load helpers).

    Builds:
      * ``Structure_data`` -- ``_pbc_ham_cubic_structure``.
      * ``AOs_*_data`` -- one of the five periodic leaves/composites
        (``_pbc_ham_aos_data``).
      * ``MOs_data`` over the AO leaf so the Geminal_data exercises the
        ``MOs_data.aos_data: AOs_base_data`` widening.
      * ``Geminal_data`` -- 2-up / 2-dn closed-shell AGP with a small
        random ``lambda_matrix``.
      * ``Jastrow_two_body_data`` (pade) -- carries the periodic
        ``structure_data`` so the persistence layer covers the
        ``_apply_jastrow_structure_backfill_post_load`` no-op branch.
      * ``Coulomb_potential_data`` -- bare-Coulomb (no ECP) + Ewald data
        rebuilt via ``build_ewald_data(structure, ..., tol=1e-10,
        n_electron=None)``.
    """
    structure = _pbc_ham_cubic_structure()
    aos_data = _pbc_ham_aos_data(kind, structure)
    aos_data.sanity_check()

    num_mo = 2
    n_up = n_dn = 2
    rng = np.random.default_rng(seed)
    mo_coefficients = rng.standard_normal((num_mo, aos_data.num_ao)).astype(np.float64)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    lambda_paired = rng.uniform(-0.2, 0.2, size=(num_mo, num_mo))
    lambda_paired = 0.5 * (lambda_paired + lambda_paired.T) + np.eye(num_mo)
    lambda_matrix = np.asarray(lambda_paired, dtype=np.float64)

    geminal_data = Geminal_data(
        num_electron_up=n_up,
        num_electron_dn=n_dn,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=lambda_matrix,
    )
    geminal_data.sanity_check()

    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0,
        jastrow_2b_type="pade",
        structure_data=structure,
    )
    jastrow_data = Jastrow_data(jastrow_two_body_data=j2)
    wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)

    coulomb_potential_data = Coulomb_potential_data(structure_data=structure)
    ewald_data = build_ewald_data(structure, coulomb_potential_data, tol=1e-10)
    coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb_with_ewald,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()
    return hamiltonian_data


def _pbc_ham_node_far_configuration(hamiltonian_data: Hamiltonian_data, kind: str, seed: int):
    """Pick electron positions where ``|Psi|`` is not vanishingly small."""
    rng = np.random.default_rng(seed)
    n_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
    n_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
    a = _PBC_HAM_A
    for _ in range(50):
        r_up = rng.uniform(0.5, a - 0.5, size=(n_up, 3)).astype(np.float64)
        r_dn = rng.uniform(0.5, a - 0.5, size=(n_dn, 3)).astype(np.float64)
        psi = evaluate_wavefunction(
            hamiltonian_data.wavefunction_data,
            jnp.asarray(r_up),
            jnp.asarray(r_dn),
        )
        if abs(float(psi)) > 1.0e-8:
            return r_up, r_dn
    raise AssertionError(f"Could not find a node-far configuration for periodic kind {kind}")


class TestHamiltonianHDF5PBC:
    @pytest.mark.light
    def test_hamiltonian_data_save_and_load_hdf5_roundtrip_pbc_silicon(self, tmp_path):
        """Silicon PBC Hamiltonian_data round-trips through HDF5 with all PBC helpers.

        Single roundtrip pins the PBC-only save/load helpers in one shot:

          * ``_detile_aos_data_for_save`` (pGTO image expansion collapsed on save)
          * ``_strip_derived_arrays_for_save`` (Ewald heavy-array strip on save)
          * ``_apply_image_expansion_post_load`` (pGTO rebuild on load)
          * ``_apply_ewald_rebuild_post_load`` (``build_ewald_data`` re-runs
            with the saved ``(eta, r_real_cut, g_max)`` triple)
          * ``_apply_jastrow_structure_backfill_post_load`` (no-op branch:
            ``Jastrow_two_body_data.structure_data`` is already populated)
          * unversioned HDF5 write + read

        No JIT compile of the heavy ECP non-local kernel runs in this path
        (pure HDF5 I/O), so silicon is fast.
        """
        if not os.path.exists(PERIODIC_FIXTURE_PATH):
            pytest.skip(f"Periodic TREXIO fixture {PERIODIC_FIXTURE} is not present.")
        structure_data, _aos_data, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=PERIODIC_FIXTURE_PATH, store_tuple=True
        )
        ewald_data = build_ewald_data(structure_data, coulomb_potential_data, tol=1e-10)
        coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="exp",
            structure_data=structure_data,
        )
        jastrow_data = Jastrow_data(jastrow_two_body_data=j2)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_with_ewald,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        h5_file = tmp_path / "test_hamiltonian_pbc.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))

        assert_dataclasses_equal(hamiltonian_data, loaded)

        loaded_ewald = loaded.coulomb_potential_data.ewald_data
        assert loaded_ewald is not None
        assert loaded_ewald.real_image_list is not None
        assert loaded_ewald.g_vectors is not None
        assert loaded_ewald.g_factors is not None

        from jqmc._setting import get_tol_hdf5_roundtrip

        hdf5_atol, hdf5_rtol = get_tol_hdf5_roundtrip("strict")
        np.testing.assert_allclose(
            float(loaded_ewald.ii_energy),
            float(hamiltonian_data.coulomb_potential_data.ewald_data.ii_energy),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )

        import h5py

        with h5py.File(str(h5_file), "r") as f:
            assert "pbc_schema_version" not in f.attrs

    @pytest.mark.light
    def test_hamiltonian_data_save_and_load_hdf5_roundtrip_pbc_lithium_k025(self, tmp_path):
        """Non-Gamma twist (lithium_k025) Hamiltonian_data round-trips through HDF5.

        Mirrors :py:func:`test_hamiltonian_data_save_and_load_hdf5_roundtrip_pbc_silicon`
        on the complex Bloch path so the round-trip exercises:

          * the complex-Bloch detile: ``_detile_aos_data_for_save`` must preserve
            the leaf's ``k_point`` field (``k`` lives on the AO leaf, not on
            ``Structure_data``) so the on-load rebuild via
            ``_apply_image_expansion_complex`` re-promotes the leaf to its
            complex Bloch variant. Without ``k_point`` propagation the leaf
            would round-trip as a real molecular GTO and the saved complex
            ``mo_coefficients`` would trip ``MOs_data.sanity_check``.
          * the BCS / TR-pair dn channel: the dn AO leaf carries ``k_point``
            equal to the negation of the up leaf's ``k_point``; this asymmetry
            must also survive the detile round-trip (each spin channel detiles
            independently).
          * ``MOs_data.sanity_check`` dtype-leaf contract (complex MOs require a
            complex Bloch AO leaf) on both spins after the load.

        Regression target: an earlier ``_detile_aos_data_for_save`` that did not
        copy ``aos_data.k_point`` into the molecular GTO leaf demoted the saved
        AO leaves to ``k_point=None``; on load the dispatcher routed them to
        the real path, leaving complex ``mo_coefficients`` paired with a real
        AO leaf -- production MCMC runs failed with hundreds of mirrored
        ``real AO leaf requires real mo_coefficients`` errors before the
        walkers started.
        """
        from jqmc.atomic_orbital import is_complex_aos

        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "lithium_k025.h5")
        if not os.path.exists(fixture):
            pytest.skip("Non-Gamma TREXIO fixture lithium_k025.h5 is not present.")
        structure_data, _aos_data, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=fixture, store_tuple=True
        )
        # Precondition: the loader built complex Bloch AO leaves on both spins,
        # and the dn leaf's k_point is the negation of the up leaf's. Any
        # regression on the loader side will trip these before save/load runs.
        _up_leaf = geminal_data.orb_data_up_spin.aos_data
        _dn_leaf = geminal_data.orb_data_dn_spin.aos_data
        assert is_complex_aos(_up_leaf), "expected lithium_k025 to load as a complex Bloch AO leaf for the up channel"
        assert is_complex_aos(_dn_leaf), "expected lithium_k025 to load as a complex Bloch AO leaf for the dn channel"
        assert _up_leaf.k_point is not None and _dn_leaf.k_point is not None
        assert tuple(float(c) for c in _dn_leaf.k_point) == tuple(-float(c) for c in _up_leaf.k_point), (
            "BCS / TR-pair convention: the dn AO leaf must carry -k of the up leaf."
        )

        ewald_data = build_ewald_data(structure_data, coulomb_potential_data, tol=1e-10)
        coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="exp",
            structure_data=structure_data,
        )
        jastrow_data = Jastrow_data(jastrow_two_body_data=j2)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_with_ewald,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        h5_file = tmp_path / "test_hamiltonian_pbc_lithium_k025.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))

        # Postcondition: the load path must re-promote both AO leaves to their
        # complex Bloch variants with the correct (+k for up, -k for dn) twist.
        # ``sanity_check`` enforces the dtype-leaf contract that detected the
        # original regression; the explicit leaf checks below pin the specific
        # field (``k_point``) that the bug was zeroing out.
        loaded.sanity_check()
        _loaded_up = loaded.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
        _loaded_dn = loaded.wavefunction_data.geminal_data.orb_data_dn_spin.aos_data
        assert is_complex_aos(_loaded_up), "round-trip demoted the up AO leaf to a real leaf (k_point lost on detile)"
        assert is_complex_aos(_loaded_dn), "round-trip demoted the dn AO leaf to a real leaf (k_point lost on detile)"
        assert tuple(float(c) for c in _loaded_up.k_point) == tuple(float(c) for c in _up_leaf.k_point)
        assert tuple(float(c) for c in _loaded_dn.k_point) == tuple(float(c) for c in _dn_leaf.k_point)

        assert_dataclasses_equal(hamiltonian_data, loaded)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_pbc_file", ["lithium.h5", "lithium_k025.h5"])
    def test_hamiltonian_data_roundtrip_pbc_j3_via_uncontracted_aos(self, tmp_path, trexio_pbc_file):
        """PBC Hamiltonian round-trip with a Jastrow_3b leaf built from
        ``aos_data._build_uncontracted_aos()`` (the same construction used by
        ``jqmc-tool trexio convert-to -j3 ao-{small,full}``).

        Complements the integration coverage in
        ``test_jqmc_tool.py::TestTrexioConvertTo::test_trexio_convert_to_pbc_j3_load_roundtrip``
        by bypassing the ``jqmc-tool`` wrapper and exercising the same chain
        (``_build_uncontracted_aos -> Jastrow_three_body_data ->
        Hamiltonian_data.save_to_hdf5 -> load_from_hdf5``) directly. If a
        future regression slips past the ``jqmc-tool`` tests (e.g. by
        reverting an internal ``Jastrow_three_body_data`` build helper) the
        failure is still caught here at the persistence boundary.

        The failure mode is an ``OverflowError`` in
        ``_apply_image_expansion_post_load`` when the J3 ``orb_data`` has
        ``_image_truncation_tol=0.0`` (a dataclass default leaking through
        ``type(self)(...)`` in ``_build_uncontracted_aos``). Source-level
        coverage:
        ``test_atomic_orbital.py::TestPGTOSemantics::test_build_uncontracted_aos_preserves_pgto_fields``.
        """
        fixture = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_pbc_file)
        if not os.path.exists(fixture):
            pytest.skip(f"PBC TREXIO fixture {trexio_pbc_file} is not present.")
        structure_data, aos_data, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=fixture, store_tuple=True
        )

        # Build the J3 leaf the same way jqmc-tool does for ``-j3 ao-full``: take
        # the unique-(orbital, exponent) basis of the loaded pgto AOs and wrap
        # it in Jastrow_three_body_data. This build path must keep
        # ``_image_truncation_tol`` / ``_n_image``.
        uncontracted = _phase_free_j3_orbitals(aos_data._build_uncontracted_aos())
        np.testing.assert_array_equal(uncontracted.k_frac_np, np.zeros(3))
        j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=uncontracted)

        ewald_data = build_ewald_data(structure_data, coulomb_potential_data, tol=1e-10)
        coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0,
            jastrow_2b_type="exp",
            structure_data=structure_data,
        )
        jastrow_data = Jastrow_data(jastrow_two_body_data=j2, jastrow_three_body_data=j3)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_with_ewald,
            wavefunction_data=wavefunction_data,
        )
        hamiltonian_data.sanity_check()

        h5_file = tmp_path / f"test_hamiltonian_pbc_j3_uncontracted_{trexio_pbc_file}"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        # The bug surfaces here: ``Hamiltonian_data.load_from_hdf5`` runs
        # ``_apply_image_expansion_post_load`` which feeds the loaded
        # ``_image_truncation_tol`` into ``_primitive_image_radius``. If the
        # value was zero'd by ``_build_uncontracted_aos``, this raises
        # ``OverflowError: cannot convert float infinity to integer``.
        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))
        loaded.sanity_check()
        loaded_j3 = loaded.wavefunction_data.jastrow_data.jastrow_three_body_data
        assert loaded_j3 is not None
        loaded_orbitals = loaded_j3.orb_data
        assert type(loaded_orbitals) is type(uncontracted)
        assert loaded_orbitals.num_ao == uncontracted.num_ao
        assert loaded_orbitals._image_truncation_tol == uncontracted._image_truncation_tol
        assert loaded_orbitals._image_truncation_tol > 0.0
        assert loaded_orbitals._n_image == uncontracted._n_image
        np.testing.assert_array_equal(loaded_orbitals.k_frac_np, np.zeros(3))

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PBC_HAM_PERIODIC_KINDS)
    def test_hamiltonian_data_save_and_load_hdf5_roundtrip_pbc_dummy(self, kind: str, tmp_path):
        """Round-trip ``Hamiltonian_data`` through HDF5 over every periodic AO kind.

        Exercises the periodic save/load helpers
        (:py:func:`_detile_aos_data_for_save`,
        :py:func:`_strip_derived_arrays_for_save`,
        :py:func:`_apply_image_expansion_post_load`,
        :py:func:`_apply_ewald_rebuild_post_load`) on a tiny dummy periodic
        Hamiltonian that uses the PW / B-spline / pGTO / pGTO+PW / pGTO+B-spline
        bases. TREXIO-independent so the same coverage is reachable without the
        silicon fixture.
        """
        hamiltonian_data = _build_pbc_dummy_hamiltonian(kind, seed=2025)
        r_up, r_dn = _pbc_ham_node_far_configuration(hamiltonian_data, kind, seed=11)
        r_up_jnp = jnp.asarray(r_up)
        r_dn_jnp = jnp.asarray(r_dn)
        psi_before = float(evaluate_wavefunction(hamiltonian_data.wavefunction_data, r_up_jnp, r_dn_jnp))
        assert np.isfinite(psi_before), f"|Psi| not finite pre-save ({kind})"

        h5_file = tmp_path / f"test_hamiltonian_pbc_{kind.replace('+', '_')}.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))

        # Schema attr present + correct version.
        import h5py

        with h5py.File(str(h5_file), "r") as f:
            assert "pbc_schema_version" not in f.attrs

        # Structure / pbc flag / canonical occupied SD rows preserved.
        from jqmc._setting import get_tol_hdf5_roundtrip

        hdf5_atol, hdf5_rtol = get_tol_hdf5_roundtrip("strict")
        assert loaded.structure_data.pbc_flag is True
        np.testing.assert_allclose(
            np.asarray(loaded.structure_data.positions),
            np.asarray(hamiltonian_data.structure_data.positions),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients),
            np.asarray(hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )
        np.testing.assert_allclose(
            np.asarray(loaded.wavefunction_data.geminal_data.orb_data_dn_spin.mo_coefficients),
            np.asarray(hamiltonian_data.wavefunction_data.geminal_data.orb_data_dn_spin.mo_coefficients),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )
        assert (
            loaded.wavefunction_data.geminal_data.num_electron_up
            == hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        )
        assert (
            loaded.wavefunction_data.geminal_data.num_electron_dn
            == hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        )

        # AO sub-tree type and num_ao preserved through detile + re-expand.
        orig_up = hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin
        load_up = loaded.wavefunction_data.geminal_data.orb_data_up_spin
        assert type(load_up) is type(orig_up), f"MOs_data type changed: {type(orig_up)} -> {type(load_up)}"
        assert load_up.aos_data.num_ao == orig_up.aos_data.num_ao
        assert type(load_up.aos_data) is type(orig_up.aos_data), (
            f"AO sub-tree type changed: {type(orig_up.aos_data)} -> {type(load_up.aos_data)}"
        )

        # Ewald rebuild populated heavy tables again.
        loaded_ewald = loaded.coulomb_potential_data.ewald_data
        assert loaded_ewald is not None
        assert loaded_ewald.real_image_list is not None
        assert loaded_ewald.g_vectors is not None
        assert loaded_ewald.g_factors is not None
        np.testing.assert_allclose(
            float(loaded_ewald.ii_energy),
            float(hamiltonian_data.coulomb_potential_data.ewald_data.ii_energy),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )

        # Wavefunction value at the same configuration must reproduce.
        psi_after = float(evaluate_wavefunction(loaded.wavefunction_data, r_up_jnp, r_dn_jnp))
        assert np.isfinite(psi_after), f"|Psi| not finite post-load ({kind})"
        atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "wf_kinetic"), "strict")
        np.testing.assert_allclose(psi_after, psi_before, atol=atol, rtol=rtol)
        jax.clear_caches()


class TestHamiltonianHDF5Compat:
    @pytest.mark.light
    def test_hamiltonian_data_load_hdf5_without_version_metadata(self, tmp_path):
        """The writer emits the current layout without a schema-version field."""
        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ecp_ccpvtz_cart.h5"
            ),
            store_tuple=True,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(geminal_data=geminal_data),
        )
        h5_file = tmp_path / "test_hamiltonian_legacy.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))

        import h5py

        with h5py.File(str(h5_file), "r+") as f:
            assert "pbc_schema_version" not in f.attrs

        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))
        assert_dataclasses_equal(hamiltonian_data, loaded)

    @pytest.mark.light
    def test_hamiltonian_data_load_hdf5_backfills_jastrow_two_body_structure_data_in_pbc(self, tmp_path):
        """``Jastrow_two_body_data.structure_data`` is backfilled from the top-level
        structure on load when (a) the top-level is PBC and (b) the loaded
        ``j2.structure_data`` is the empty default.

        Simulates an older checkpoint where ``Jastrow_two_body_data`` was
        serialised without a ``structure_data`` subgroup: constructs the
        in-memory object with the empty default, saves, and loads it. The
        post-load fixup substitutes the top-level structure (silicon here).
        The roundtrip is therefore deliberately not identity: the loaded
        ``j2.structure_data`` equals the silicon structure rather than the
        pre-save empty default.
        """
        if not os.path.exists(PERIODIC_FIXTURE_PATH):
            pytest.skip(f"Periodic TREXIO fixture {PERIODIC_FIXTURE} is not present.")
        structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
            trexio_file=PERIODIC_FIXTURE_PATH, store_tuple=True
        )
        ewald_data = build_ewald_data(structure_data, coulomb_potential_data, tol=1e-10)
        coulomb_with_ewald = coulomb_potential_data.replace(ewald_data=ewald_data)
        # No ``structure_data`` argument: simulate the empty default written by
        # checkpoints that predate the nested periodic structure.
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
        assert np.asarray(j2.structure_data.positions).size == 0
        assert j2.structure_data.pbc_flag is False
        jastrow_data = Jastrow_data(jastrow_two_body_data=j2)
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_with_ewald,
            wavefunction_data=wavefunction_data,
        )
        h5_file = tmp_path / "test_hamiltonian_j2_backfill.h5"
        hamiltonian_data.save_to_hdf5(str(h5_file))
        loaded = Hamiltonian_data.load_from_hdf5(str(h5_file))

        loaded_j2 = loaded.wavefunction_data.jastrow_data.jastrow_two_body_data
        assert loaded_j2.structure_data.pbc_flag is True
        from jqmc._setting import get_tol_hdf5_roundtrip

        hdf5_atol, hdf5_rtol = get_tol_hdf5_roundtrip("strict")
        np.testing.assert_allclose(
            np.asarray(loaded_j2.structure_data.positions),
            np.asarray(structure_data.positions),
            atol=hdf5_atol,
            rtol=hdf5_rtol,
        )


@pytest.mark.light
def test_get_num_electrons_from_h5(tmp_path):
    """get_num_electrons reads (num_electron_up, num_electron_dn) from a saved h5.

    Prerequisite of the workflow's num_mcmc auto setting (the mechanism lives in
    jqmc): build a minimal H2 (2-electron) Hamiltonian_data, round-trip it through
    save_to_hdf5 / load_from_hdf5, and confirm the electron counts read back
    match the source geminal. The workflow multiplies the total by 4 for
    ``num_mcmc_per_measurement='auto'``; that policy stays in the workflow.
    """
    trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (
        structure_data,
        _,
        _mos_data,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(trexio_file=trexio_path, store_tuple=True)

    wavefunction_data = Wavefunction_data(
        jastrow_data=Jastrow_data(
            jastrow_one_body_data=None,
            jastrow_two_body_data=None,
            jastrow_three_body_data=None,
        ),
        geminal_data=geminal_mo_data,
    )
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    h5_path = str(tmp_path / "hamiltonian_data.h5")
    hamiltonian_data.save_to_hdf5(h5_path)

    nup, ndn = get_num_electrons(h5_path)
    # H2 all-electron: 2 electrons, closed shell -> (1, 1).
    assert (nup, ndn) == (1, 1)
    # ... and must equal the source geminal's counts for any fixture.
    assert nup == int(geminal_mo_data.num_electron_up)
    assert ndn == int(geminal_mo_data.num_electron_dn)


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Issue #67 follow-up: HDF5 / checkpoint restore of a stale minimum-image box.
# ---------------------------------------------------------------------------
#: Rectangular lattice diag(6, 7, 8) Bohr written in a non-reduced basis
#: ``U @ diag(6, 7, 8)`` (``det U = 1``), as in the issue #67 reproduction.
_STALE_N_MAX_LENGTHS = np.array([6.0, 7.0, 8.0])
_STALE_N_MAX_TRANSFORM = np.array([[6, 1, -1], [-2, 1, 0], [-3, -2, 1]], dtype=np.float64)
#: Displacement from the issue report (the old ``a + b + c``-only box returns 5.768 Bohr, the minimum is 5.042 Bohr).
_STALE_N_MAX_REPORTED_DISPLACEMENT = np.array([3.7458936435198424, 39.16755006833722, -35.50971359433966])


def _stale_n_max_skewed_hamiltonian() -> Hamiltonian_data:
    """Charge-neutral 4-H periodic dummy Hamiltonian on the non-reduced basis (plane-wave AOs)."""
    lengths = _STALE_N_MAX_LENGTHS
    cell = _STALE_N_MAX_TRANSFORM @ np.diag(lengths)
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]) * lengths
    structure = Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H", "H", "H", "H"),
        atomic_labels=("H1", "H2", "H3", "H4"),
    )
    aos_data = _pbc_ham_pw_leaf(structure)
    num_mo = 2
    rng = np.random.default_rng(2067)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=rng.standard_normal((num_mo, aos_data.num_ao)))
    geminal_data = Geminal_data(
        num_electron_up=2,
        num_electron_dn=2,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=np.eye(num_mo, dtype=np.float64),
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
    )
    coulomb_potential_data = Coulomb_potential_data(structure_data=structure)
    ewald_data = build_ewald_data(structure, coulomb_potential_data, tol=1e-10)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb_potential_data.replace(ewald_data=ewald_data),
        wavefunction_data=Wavefunction_data(geminal_data=geminal_data, jastrow_data=Jastrow_data(jastrow_two_body_data=j2)),
    )
    hamiltonian_data.sanity_check()
    return hamiltonian_data


def _overwrite_structure_n_max(h5_group, stale_n_max: int) -> int:
    """Rewrite ``_n_max`` of every persisted ``Structure_data`` group (simulates a pre-#67 file)."""
    groups = []

    def _visit(_name, obj):
        if obj.attrs.get("_class_name") == "Structure_data" and "_n_max" in obj.attrs:
            groups.append(obj)

    h5_group.visititems(_visit)
    for grp in groups:
        grp.attrs["_n_max"] = np.int64(stale_n_max)
    return len(groups)


def _reachable_structures(obj, seen=None):
    """Every ``Structure_data`` reachable from a dataclass tree."""
    import dataclasses

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, Structure_data):
        return [obj]
    found = []
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            found += _reachable_structures(getattr(obj, field.name, None), seen)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found += _reachable_structures(item, seen)
    return found


@pytest.mark.light
def test_hdf5_restore_recomputes_stale_minimum_image_box_on_non_reduced_basis(tmp_path):
    """A periodic structure restored from HDF5 / a checkpoint gets a sufficient minimum-image box.

    Files may persist a ``Structure_data._n_max`` from an
    ``a + b + c``-only heuristic. On a non-reduced basis that box misses minimum
    images. Both restore layouts (``Hamiltonian_data.save_to_hdf5`` file and the
    merged-checkpoint ``hamiltonian_data`` group) are written with the stale box
    in every ``Structure_data`` group and reloaded through the public loaders.
    Every reachable restored structure must carry the value ``Structure_data.build``
    computes for the same lattice, and the minimum images obtained with the restored
    structure must be the shortest lattice-equivalent vectors. The reference is a
    brute-force 27-image search in the rectangular basis diag(6, 7, 8) of the same
    lattice after the per-axis wrap; no production helper is involved.
    """
    import h5py

    from jqmc._checkpoint import load_hamiltonian_from_checkpoint
    from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
    from jqmc.hamiltonians import _save_dataclass_to_hdf5
    from jqmc.structure import _get_min_dist_rel_R_cart_jnp, _minimum_image_displacement
    from tests.conftest import brute_force_mic

    hamiltonian_data = _stale_n_max_skewed_hamiltonian()
    built = hamiltonian_data.structure_data
    cell = np.asarray(built.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    lengths = _STALE_N_MAX_LENGTHS

    rng = np.random.default_rng(seed=67)
    displacements = np.concatenate(
        (_STALE_N_MAX_REPORTED_DISPLACEMENT[None, :], rng.uniform(-2.0, 2.0, size=(400, 3)) @ cell), axis=0
    )
    rect_cell = np.diag(lengths)
    reference = np.stack([brute_force_mic(d - np.round(d / lengths) * lengths, rect_cell, n_range=1) for d in displacements])
    reference_norm = np.linalg.norm(reference, axis=1)
    atol, rtol = ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE

    # Stale box of the pre-#67 heuristic (a + b + c body diagonal over the inradius).
    half_diag = 0.5 * np.linalg.norm(cell.sum(axis=0))
    stale_n_max = int(np.ceil(half_diag / built.cell_inradius())) + 1

    def _mic(n_max):
        return np.asarray(
            jax.vmap(lambda d: _minimum_image_displacement(d, jnp.asarray(cell), jnp.asarray(inv_cell), n_max))(
                jnp.asarray(displacements)
            ),
            dtype=np.float64,
        )

    # Non-vacuity: the stale box really returns a non-minimal image here.
    stale_norm = np.linalg.norm(_mic(stale_n_max), axis=1)
    assert np.max(stale_norm - reference_norm) > 10.0 * (atol + rtol * np.max(reference_norm))

    # Layout 1: standalone Hamiltonian file.
    ham_file = tmp_path / "stale_n_max_hamiltonian.h5"
    hamiltonian_data.save_to_hdf5(str(ham_file))
    with h5py.File(str(ham_file), "r+") as f:
        n_ham_groups = _overwrite_structure_n_max(f, stale_n_max)
    # Layout 2: merged-checkpoint group, as written by ``merge_rank_checkpoints``.
    chk_file = tmp_path / "stale_n_max_checkpoint.h5"
    with h5py.File(str(chk_file), "w") as f:
        f.create_group("_meta")
        _save_dataclass_to_hdf5(f.create_group("hamiltonian_data"), hamiltonian_data)
        n_chk_groups = _overwrite_structure_n_max(f["hamiltonian_data"], stale_n_max)
    assert n_ham_groups >= 3 and n_chk_groups >= 3, (n_ham_groups, n_chk_groups)

    for label, loaded in (
        ("load_from_hdf5", Hamiltonian_data.load_from_hdf5(str(ham_file))),
        ("load_hamiltonian_from_checkpoint", load_hamiltonian_from_checkpoint(str(chk_file))),
    ):
        structures = [s for s in _reachable_structures(loaded) if s.pbc_flag]
        assert len(structures) >= 3, label
        for s in structures:
            assert s._n_max == built._n_max, (label, s._n_max, built._n_max)
            assert type(s._n_max) is int, label

        restored = loaded.structure_data
        xla = _mic(restored._n_max)
        nearest = np.stack(
            [
                np.asarray(_get_min_dist_rel_R_cart_jnp(restored, jnp.asarray(-d), 0), dtype=np.float64)
                for d in displacements[:200]
            ]
        )
        for name, got in (("xla", xla), ("nearest_jnp", nearest)):
            count = len(got)
            np.testing.assert_allclose(
                np.linalg.norm(got, axis=1), reference_norm[:count], atol=atol, rtol=rtol, err_msg=f"{label}/{name}"
            )
            shift = (got - displacements[:count]) @ inv_cell
            np.testing.assert_allclose(shift, np.rint(shift), atol=atol, rtol=rtol, err_msg=f"{label}/{name}")
    jax.clear_caches()
