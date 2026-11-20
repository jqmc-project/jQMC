"""Native LRDMC production, measurement derivatives and sector restart coverage."""

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

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jqmc_rs

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import get_tolerance_min
from jqmc._setting import get_tol_hdf5_roundtrip, get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import _auto, assemble, gfmc
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.orbital_tiling import conjugate_tiled_orbitals
from tests.test_jqmc_kernel._orbital_tiling_reference import add_pw_component
from tests.test_jqmc_kernel.test_gfmc_measurement_fd import _measurement_energy
from tests.test_jqmc_kernel.test_orbital_tiling_source_routes import _active_jastrow, _convert, _reference
from tests.test_rs_kernel._route_spy import assert_arg_reached, assert_native_ran, install_ctx_spy
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_BACKENDS = ["pw", "bspline", "pgto_cart", "pgto_sphe", "composite_cart", "composite_sphe"]
_ZONES = (
    "tiling_eval",
    "tiling_grad",
    "tiling_lap",
    "ao_eval",
    "ao_grad_lap",
    "det_grad_lap",
    "jastrow_grad_lap",
    "wf_kinetic",
    "coulomb",
    "local_energy",
)
_FIELDS = (
    "_latest_r_up_carts",
    "_latest_r_dn_carts",
    "_jax_PRNG_key_list",
    "_stored_e_L",
    "_stored_e_L2",
    "_stored_w_L",
    "_stored_force_HF",
    "_stored_force_PP",
    "_stored_E_L_force_PP",
    "_stored_stress_HF",
    "_stored_stress_PP",
    "_stored_E_L_stress_PP",
)


def _inputs(tmp_path, backend, mode, independent, family):
    source, kind = {
        "pw": ("qe", "pw"),
        "bspline": ("qe", "bspline"),
        "pgto_cart": ("trexio", "cart"),
        "pgto_sphe": ("trexio", "sphe"),
        "composite_cart": ("trexio", "cart"),
        "composite_sphe": ("qe", "pgto+pw"),
    }[backend]
    hams, weights = _convert(tmp_path, source, kind, mode, independent)
    result = []
    for ham in hams:
        if backend == "composite_cart":
            # TREXIO supplies GTOs, not PW data. Add an explicitly nonzero PW
            # component for the Cartesian composite runtime fixture only.
            sd = ham.wavefunction_data.geminal_data
            up = add_pw_component(sd.orb_data_up_spin, reverse=True)
            down = (
                None
                if not independent
                else add_pw_component(conjugate_tiled_orbitals(sd.orb_data_dn_spin, spin="up"), reverse=True)
            )
            sd = SD_data.build_tiled(up_source=up, down_source=down)
            ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=sd))
        assert ham.coulomb_potential_data.ecp_flag
        ham = _active_jastrow(ham, family)
        result.append(ham)
    if mode == "average":
        np.testing.assert_array_equal(weights, [0.2, 0.8])
    return result, weights


def _assert_payload(original, restored, independent):
    before, after = original.wavefunction_data.geminal_data, restored.wavefunction_data.geminal_data
    for spin in ("up", "dn"):
        a, b = getattr(before, f"orb_data_{spin}_spin"), getattr(after, f"orb_data_{spin}_spin")
        assert a.supertwist == b.supertwist and a.tiling_matrix == b.tiling_matrix
        for x, y in zip(a.blocks, b.blocks, strict=True):
            assert x.k_point == y.k_point and x.conjugate == y.conjugate
            assert x.source_band_indices == y.source_band_indices
            if x.coefficients is None:
                assert y.coefficients is None
            else:
                np.testing.assert_array_equal(x.coefficients, y.coefficients)
    for block in after.orb_data_dn_spin.blocks:
        shared = any(
            block.backend is up.backend and block.coefficients is up.coefficients for up in after.orb_data_up_spin.blocks
        )
        assert shared is (not independent)


def _measurements(ham, backend, driver, monkeypatch):
    """Same-input continuum measurement, all atoms/strain and AD, not a smoke check."""
    plain = _reference(ham, "pgto+pw" if backend.startswith("composite") else backend)
    up, dn = draw_well_conditioned_cfg(ham, seed=31, scale=0.6)
    RT = jnp.eye(3)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    mesh = icosahedron_sym_mesh_Nv12
    w, g = np.asarray(mesh.weights, float), np.asarray(mesh.grid_points, float)
    atol, rtol = get_tolerance_min(_ZONES, "medium")
    natol, nrtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    with monkeypatch.context() as patch:
        patch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
        energy = _measurement_energy(driver, "tmove")
        actual, expected = float(energy(ham, up, dn, RT)), float(energy(plain, up, dn, RT))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        inverse = np.asarray(np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, up, dn))), complex)
        positions = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions")(ham, up, dn, RT, 0.3, non_local_move="tmove")
        strain = getattr(gfmc, f"grad_e_L_{driver}_wrt_strain")(ham, up, dn, RT, 0.3, non_local_move="tmove")
        plain_positions = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions")(plain, up, dn, RT, 0.3, non_local_move="tmove")
        plain_strain = getattr(gfmc, f"grad_e_L_{driver}_wrt_strain")(plain, up, dn, RT, 0.3, non_local_move="tmove")
        for got, ref in zip((*positions, strain), (*plain_positions, plain_strain), strict=True):
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)
        # T-move continuum measurement has the ordinary local-energy derivative;
        # propagation S-prime membership is deliberately not differentiated.
        for coordinate, value in (("R", positions[0]), ("eps", strain)):
            # Each energy AD reference compiles a distinct graph. Release the
            # completed analytical or previous AD executables before compiling
            # the next graph so their executable mappings do not accumulate.
            jax.clear_caches()
            gc.collect()
            auto = _auto.compute_grad_e_L(ham, up, dn, RT, wrt=coordinate)
            np.testing.assert_allclose(value, np.real(auto), atol=atol, rtol=rtol)
        pulay = [assemble.compute_grad_ln_Psi(ham, up, dn, wrt=axis) for axis in ("R", "eps")]
        for axis, value in zip(("R", "eps"), pulay, strict=True):
            # Energy/analytical comparisons above are complete. On JAX 0.7.2,
            # retaining their executables through both Pulay AD compiles can
            # exhaust vm.max_map_count even while RSS is well below its limit.
            # Keep the result arrays and every comparison, release only caches.
            jax.clear_caches()
            gc.collect()
            np.testing.assert_allclose(value, _auto.compute_grad_ln_Psi(ham, up, dn, wrt=axis), atol=atol, rtol=rtol)
            np.testing.assert_allclose(value, assemble.compute_grad_ln_Psi(plain, up, dn, wrt=axis), atol=atol, rtol=rtol)

    def forbidden(*args, **kwargs):
        pytest.fail("native LRDMC measurement called an XLA tiled evaluator")

    with monkeypatch.context() as patch:
        patch.setattr("jqmc.orbital_tiling.compute_tiled_orbitals", forbidden)
        patch.setattr("jqmc.orbital_tiling.compute_tiled_orbitals_value_grad_lap", forbidden)
        patch.setattr("jqmc.derivatives.orbital_tiling.tiled_field_vjp", forbidden)
        native_e = complex(*jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, g, inverse)).real
        args = (gd, jd, cp, np.asarray(up), np.asarray(dn), 0.3, np.asarray(RT), w, g, 1, False)
        native_positions = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(*args)
        native_strain = jqmc_rs._gfmc_stress_ecp_pbc_complex(*args)
        native_pulay = (
            jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, up, dn, inverse)[0],
            jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(gd, jd, up, dn, inverse),
        )
    np.testing.assert_allclose(native_e, actual, atol=natol, rtol=nrtol)
    for got, ref in zip((*native_positions, native_strain, *native_pulay), (*positions, strain, *pulay), strict=True):
        np.testing.assert_allclose(got, np.real(ref), atol=natol, rtol=nrtol)
    native = np.concatenate(
        [np.atleast_1d(native_e), *(np.asarray(x).real.ravel() for x in (*native_positions, native_strain, *native_pulay))]
    )
    reference = np.concatenate(
        [np.atleast_1d(expected), *(np.asarray(x).real.ravel() for x in (*plain_positions, plain_strain, *pulay))]
    )
    return native, reference


@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("family", ["j123", "dtn"])
@pytest.mark.parametrize("mode", ["gamma", "general", "average"])
@pytest.mark.parametrize("independent", [False, True])
def test_native_tiled_lrdmc_observables_and_restart(driver, backend, family, mode, independent, tmp_path, monkeypatch):
    """Native full observables, exact saved-state continuation and per-sector physics."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    hams, weights = _inputs(tmp_path, backend, mode, independent, family)
    histories, resumed_histories, natives, references = [], [], [], []
    for index, ham in enumerate(hams):
        sector_dir = tmp_path / f"sector-{index}"
        sector_dir.mkdir()
        monkeypatch.chdir(sector_dir)
        cls = GFMC_n if driver == "n" else GFMC_t
        options = {"num_mcmc_per_measurement": 2} if driver == "n" else {"tau": 0.01}
        run = cls(
            hamiltonian_data=ham,
            num_walkers=2,
            mcmc_seed=812,
            num_gfmc_collect_steps=0,
            alat=0.3,
            non_local_move="tmove",
            comput_position_deriv=True,
            comput_stress=True,
            use_swct=False,
            **options,
        )
        with monkeypatch.context() as patch:
            spy = install_ctx_spy(patch)
            run.run(num_mcmc_steps=2)
            assert run._native_gfmc_chunk_route == "native"
            pattern = str(sector_dir / "rank-{rank}.h5")
            run.save_to_hdf5(pattern.format(rank=0))
            merged = str(sector_dir / "merged.h5")
            merge_rank_checkpoints(merged, mpi_size=1, driver_type=f"GFMC_{driver}", hamiltonian_data=ham, tmp_pattern=pattern)
            restored = cls.load_from_hdf5(merged)
            _assert_payload(ham, restored.hamiltonian_data, independent)
            for field in _FIELDS:
                before, after = np.asarray(getattr(run, field)), np.asarray(getattr(restored, field))
                assert before.size and np.isfinite(before).all()
                np.testing.assert_array_equal(after, before)
            run.run(num_mcmc_steps=2)
            restored.run(num_mcmc_steps=2)
            assert run._native_gfmc_chunk_route == restored._native_gfmc_chunk_route == "native"
            method = f"run_gfmc_projection_{driver}_complex_batch"
            assert_native_ran(spy, method, at_least=3)
            assert_arg_reached(spy, method, "want_force", True)
            assert_arg_reached(spy, method, "want_stress", True)
            assert_arg_reached(spy, method, "dltmove", False)
        atol, rtol = get_tol_hdf5_roundtrip("strict")
        for field in _FIELDS:
            a, b = np.asarray(getattr(run, field)), np.asarray(getattr(restored, field))
            assert a.size and b.size and np.isfinite(a).all() and np.isfinite(b).all()
            if field == "_jax_PRNG_key_list":
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
        assert run.mcmc_counter == restored.mcmc_counter == 4
        fields = [field for field in _FIELDS if field.startswith("_stored")]
        histories.append(np.concatenate([np.asarray(getattr(run, field)).ravel() for field in fields]))
        resumed_histories.append(np.concatenate([np.asarray(getattr(restored, field)).ravel() for field in fields]))
        # Release trajectory executables before the independent derivative phase.
        jax.clear_caches()
        gc.collect()
        native, reference = _measurements(ham, backend, driver, monkeypatch)
        natives.append(native)
        references.append(reference)
        jax.clear_caches()
        gc.collect()
    atol, rtol = get_tol_hdf5_roundtrip("strict")
    np.testing.assert_allclose(
        np.asarray(weights) @ np.stack(histories), np.asarray(weights) @ np.stack(resumed_histories), atol=atol, rtol=rtol
    )
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    np.testing.assert_allclose(
        np.asarray(weights) @ np.stack(natives), np.asarray(weights) @ np.stack(references), atol=atol, rtol=rtol
    )
