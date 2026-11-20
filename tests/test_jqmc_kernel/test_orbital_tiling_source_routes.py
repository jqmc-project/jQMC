"""Real-source Gamma/general/twist ensembles versus ordinary explicit supercells."""

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

from pathlib import Path
import gc

import jax

import jax.numpy as jnp
import numpy as np
import pytest
from click.testing import CliRunner
import jqmc_rs

from jqmc._precision import get_tolerance_min
from jqmc._setting import get_tolerance_xla_rust_consistency
from jqmc._twist_average import load_twist, load_twist_average_input, normalize_twist_weights
from jqmc.coulomb_potential import compute_coulomb_potential, icosahedron_sym_mesh_Nv12
from jqmc.derivatives import assemble, _auto
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jqmc_qe import run_qe2jqmc
from jqmc.jqmc_tool import _cli
from jqmc.jastrow_factor import Jastrow_data
from jqmc.wavefunction import compute_kinetic_energy, evaluate_ln_wavefunction
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel._orbital_tiling_reference import (
    explicit_supercell_pw,
    explicit_supercell_bspline,
    explicit_supercell_pgto,
    explicit_supercell_composite,
)
from tests.test_jqmc_kernel._generated_matrix_base import _make_dtn
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]
_ROOT = Path(__file__).parents[1]
_BACKENDS = [("qe", "pw"), ("qe", "bspline"), ("qe", "pgto"), ("qe", "pgto+pw"), ("trexio", "sphe"), ("trexio", "cart")]


def _convert(tmp_path, source, backend, mode, independent):
    output = str(tmp_path / "tiled.h5")
    twist = (0.0, 0.0, 0.0) if mode == "gamma" else (0.22, 0.13, 0.09)
    if source == "qe":
        save = "O_tiling_sectors/out/o_tiling.save" if independent else "Li_tiling_sectors/out/li_tiling.save"
        params = {"qe_save_dir": str(qe_example_files_root() / save), "kind": backend, "tiling": [2, 1, 1]}
        params.update({"twist_average": True} if mode == "average" else {"supertwist": twist})
        run_qe2jqmc(params, {"j1_parameter": 0.5, "j2_parameter": 0.7, "j3_basis_type": "ao-small"}, output)
    else:
        filename = (
            ("oxygen_tiling_sectors_" + backend + ".h5")
            if independent
            else ("lithium_tiling_sectors.h5" if backend == "sphe" else "lithium_tiling_sectors_cart.h5")
        )
        options = ["--twist-average"] if mode == "average" else ["--supertwist", " ".join(str(x) for x in twist)]
        result = CliRunner().invoke(
            _cli,
            [
                "trexio",
                "convert-to",
                str(_ROOT / "trexio_example_files" / filename),
                "-o",
                output,
                "--tiling",
                "2 1 1",
                "-j1",
                "0.5",
                "-j2",
                "0.7",
                "-j3",
                "ao-small",
                *options,
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
    if mode == "average":
        meta = load_twist_average_input(output)
        return [load_twist(output, i) for i in range(meta.n_twist)], normalize_twist_weights(meta.weights)
    return [Hamiltonian_data.load_from_hdf5(output)], np.ones(1)


def _active_jastrow(ham, family):
    wf = ham.wavefunction_data
    if family == "dtn":
        jd = Jastrow_data(jastrow_dtn_data=_make_dtn(ham.structure_data, ham.coulomb_potential_data))
    else:
        jd = wf.jastrow_data
        j3 = jd.jastrow_three_body_data
        assert j3 is not None and j3.orb_data.structure_data.natom == ham.structure_data.natom
        from jqmc.hamiltonians import _detile_aos_data_for_save

        home = _detile_aos_data_for_save(j3.orb_data)
        assert set(home.nucleus_index) == set(range(ham.structure_data.natom))
        rng = np.random.default_rng(121)
        M = rng.uniform(-0.02, 0.02, size=j3.j_matrix.shape)
        M[:, :-1] = 0.5 * (M[:, :-1] + M[:, :-1].T)
        # Independent real symmetric M includes all translation-to-translation couplings.
        assert np.all(M[:, :-1] != 0)
        jd = jd.replace(jastrow_three_body_data=j3.replace(j_matrix=M))
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jd))


def _reference(ham, backend):
    sd = ham.wavefunction_data.geminal_data
    builder = {"pw": explicit_supercell_pw, "bspline": explicit_supercell_bspline, "pgto+pw": explicit_supercell_composite}.get(
        backend, explicit_supercell_pgto
    )
    plain = SD_data.build(
        num_electron_up=sd.num_electron_up,
        num_electron_dn=sd.num_electron_dn,
        mos_up=builder(sd.orb_data_up_spin),
        mos_dn=builder(sd.orb_data_dn_spin),
        time_reversal=False,
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=plain))


def _energy_components(ham, up, dn):
    wf = ham.wavefunction_data
    return np.asarray(
        [
            compute_kinetic_energy(wf, up, dn),
            compute_coulomb_potential(ham.coulomb_potential_data, up, dn, jnp.eye(3), wavefunction_data=wf),
            compute_local_energy(ham, up, dn, jnp.eye(3)),
        ]
    )


@pytest.mark.parametrize("source,backend", _BACKENDS)
@pytest.mark.parametrize("mode", ["gamma", "general", "average"])
@pytest.mark.parametrize("family", ["j123", "dtn"])
@pytest.mark.parametrize("independent", [False, True])
def test_six_source_routes_energy_force_stress(tmp_path, source, backend, mode, family, independent, monkeypatch):
    """Actual source routes match untiled SC T/V/E, all atom/strain derivatives, AD and Rust."""
    hams, weights = _convert(tmp_path, source, backend, mode, independent)
    if mode == "average":
        watol, wrtol = get_tolerance_min(("tiling_eval",), "strict")
        np.testing.assert_allclose(weights, np.array([0.2, 0.8]), atol=watol, rtol=wrtol)

    def native_call(function, *args):
        def forbidden(*unused_args, **unused_kwargs):
            pytest.fail("native source-route observable called XLA")

        with monkeypatch.context() as context:
            for entry in (
                "jqmc.orbital_tiling.compute_tiled_orbitals",
                "jqmc.orbital_tiling.compute_tiled_orbitals_value_grad_lap",
                "jqmc.determinant.compute_tiled_orbitals",
                "jqmc.determinant.compute_tiled_orbitals_value_grad_lap",
                "jqmc.derivatives.orbital_tiling.tiled_field_vjp",
            ):
                context.setattr(entry, forbidden)
            return function(*args)

    all_actual, all_reference, all_native, all_auto = [], [], [], []
    atol, rtol = get_tolerance_min(
        (
            "tiling_eval",
            "tiling_grad",
            "tiling_lap",
            "ao_eval",
            "ao_grad_lap",
            "det_grad_lap",
            "jastrow_grad_lap",
            "coulomb",
            "local_energy",
        ),
        "medium",
    )
    natol, nrtol = get_tolerance_xla_rust_consistency(
        ("tiling_grad", "tiling_lap", "ao_eval", "ao_grad_lap", "jastrow_grad_lap", "wf_kinetic", "coulomb"), level="medium"
    )
    mesh = icosahedron_sym_mesh_Nv12
    mw, mp = np.asarray(mesh.weights, float), np.asarray(mesh.grid_points, float)
    for index, ham in enumerate(hams):
        # Each sector has different static Bloch labels. Keep only completed
        # NumPy observations across this phase boundary; otherwise a two-sector
        # AD comparison can exhaust Linux's executable VMA table in one test.
        jax.clear_caches()
        gc.collect()
        ham = _active_jastrow(ham, family)
        if backend in ("pgto", "sphe", "cart"):
            from jqmc._setting import TILING_PGTO_IMAGE_TRUNCATION_ATOL

            sd = ham.wavefunction_data.geminal_data
            for provider in (sd.orb_data_up_spin, sd.orb_data_dn_spin):
                assert all(
                    block.backend._image_truncation_tol <= TILING_PGTO_IMAGE_TRUNCATION_ATOL for block in provider.blocks
                )
        if backend == "pgto+pw":
            # Neither a zero residual nor a zero GTO component may masquerade
            # as coverage of a composite source conversion.
            for provider in (
                ham.wavefunction_data.geminal_data.orb_data_up_spin,
                ham.wavefunction_data.geminal_data.orb_data_dn_spin,
            ):
                for block in provider.blocks:
                    pgto, pw = block.backend.children
                    assert pgto.num_ao > 0 and pw.num_ao > 0
                    assert np.linalg.norm(block.coefficients[:, : pgto.num_ao]) > atol
                    assert np.linalg.norm(block.coefficients[:, pgto.num_ao :]) > atol
        plain = _reference(ham, backend)
        up, dn = draw_well_conditioned_cfg(ham, seed=31 + index, scale=0.6)
        actual, wanted = _energy_components(ham, up, dn), _energy_components(plain, up, dn)
        np.testing.assert_allclose(actual, wanted, atol=atol, rtol=rtol)
        # Finite Jastrow must participate in the represented wavefunction.
        wf = ham.wavefunction_data
        without = wf.replace(jastrow_data=Jastrow_data())
        assert abs(complex(evaluate_ln_wavefunction(wf, up, dn) - evaluate_ln_wavefunction(without, up, dn))) > atol
        gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
        inv = np.asarray(np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, up, dn))), complex)
        native_t = complex(*native_call(jqmc_rs.compute_kinetic_withj_complex, gd, jd, up, dn))
        native_e = complex(*native_call(jqmc_rs.compute_local_energy_fast_complex, ham, up, dn, mw, mp, inv))
        np.testing.assert_allclose([native_t, native_e - native_t, native_e], actual, atol=natol, rtol=nrtol)
        rows = [actual.ravel()]
        refs = [wanted.ravel()]
        natives = [np.asarray([native_t, native_e - native_t, native_e])]
        autos = [actual.ravel()]
        for coordinate in ("R", "eps"):
            for name in ("compute_grad_e_L", "compute_grad_ln_Psi"):
                # Completed observations are NumPy arrays. Release the energy
                # or previous derivative executables before this independent
                # analytical pair so executable mappings do not accumulate.
                jax.clear_caches()
                gc.collect()
                args = (up, dn, jnp.eye(3)) if name == "compute_grad_e_L" else (up, dn)
                calculated = np.asarray(getattr(assemble, name)(ham, *args, wrt=coordinate))
                reference = np.asarray(getattr(assemble, name)(plain, *args, wrt=coordinate))
                # The AD graph is a separate reference; keeping both analytical
                # graphs alive through its compilation can exhaust Linux's VMA
                # limit even while RSS remains below the worker memory budget.
                jax.clear_caches()
                gc.collect()
                autodiff = np.asarray(getattr(_auto, name)(ham, *args, wrt=coordinate))
                np.testing.assert_allclose(calculated, reference, atol=atol, rtol=rtol)
                np.testing.assert_allclose(calculated, autodiff, atol=atol, rtol=rtol)
                native = (
                    (
                        native_call(jqmc_rs.vmc_grad_e_L_pbc_complex, gd, jd, cp, up, dn, mw, mp, inv)[0]
                        if coordinate == "R"
                        else native_call(jqmc_rs.vmc_grad_e_L_eps_pbc_complex, gd, jd, cp, up, dn, mw, mp, inv)
                    )
                    if name == "compute_grad_e_L"
                    else (
                        native_call(jqmc_rs.vmc_grad_ln_psi_pbc_complex, gd, jd, up, dn, inv)[0]
                        if coordinate == "R"
                        else native_call(jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex, gd, jd, up, dn, inv)
                    )
                )
                np.testing.assert_allclose(native, calculated, atol=natol, rtol=nrtol)
                rows.append(calculated.ravel())
                refs.append(reference.ravel())
                natives.append(np.asarray(native).ravel())
                autos.append(autodiff.ravel())
        all_actual.append(np.concatenate(rows))
        all_reference.append(np.concatenate(refs))
        all_native.append(np.concatenate(natives))
        all_auto.append(np.concatenate(autos))
    # Check each sector first, then the actual nonuniform quadrature aggregate.
    np.testing.assert_allclose(
        np.asarray(weights) @ np.stack(all_actual), np.asarray(weights) @ np.stack(all_reference), atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        np.asarray(weights) @ np.stack(all_actual), np.asarray(weights) @ np.stack(all_auto), atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        np.asarray(weights) @ np.stack(all_native), np.asarray(weights) @ np.stack(all_actual), atol=natol, rtol=nrtol
    )
