"""GFMC/LRDMC public-entry and representation coverage.

These nodes are REGISTERED GDD evidence, so they live outside
``test_gfmc_force_analytic.py``: that file is the derivative universe's
requirement source and is exempt from orphan registration, and the orphan gate
(correctly) refuses to let an exempt host also carry registered evidence.

Fixtures, oracles and tolerance helpers are IMPORTED from the battery rather than
re-encoded, so the two files cannot drift apart.
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

from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.derivatives import gfmc
from jqmc.hamiltonians import Hamiltonian_data
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

from .test_gfmc_force_analytic import (
    _CASES,
    _build_ham,
    _build_ham_gem,
    _e_L_n_ae_jnone,
    _e_L_t_ae_jnone,
    _extract_active_positions_from_ham,
    _gdd_assert_allclose,
    _gdd_get_tolerance_min,
    _inject_active_positions,
    _jit_grad_pos,
    _rot,
)


def _full_representation_source(gem):
    """Independently reload the same full physical non-SD geminal input."""
    directory = tempfile.TemporaryDirectory(prefix="jqmc_gfmc_representation_fidelity_")
    source = None

    def factory():
        nonlocal source
        if source is None:
            assert not is_mixed_precision_enabled()
            source = Path(directory.name) / "full_physical_source.h5"
            _build_ham_gem("H2_ae_ccpvdz_cart.h5", gem).save_to_hdf5(source)
        return Hamiltonian_data.load_from_hdf5(source)

    return factory


def _representation_forces(ham, driver):
    """Original seed-0/scale-.9 coordinates, quadrature and public position entry."""
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(0)
    up = jnp.asarray(rng.normal(scale=0.9, size=(int(gd.num_electron_up), 3)))
    down = jnp.asarray(rng.normal(scale=0.9, size=(int(gd.num_electron_dn), 3)))
    forces = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions")(ham, up, down, _rot(1), 1.0)
    return {name: np.asarray(value) for name, value in zip(("nuclear", "electron-up", "electron-down"), forces, strict=True)}


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Bind the real central homogeneous-block norm comparison to this module."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


@pytest.fixture(autouse=True)
def _release_representation_compilations():
    yield
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("stem, alat, seed, scale", _CASES)
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_fused_positions_and_strain_matches_separate_entries(driver, stem, alat, seed, scale):
    """The FUSED public entry the live driver calls must equal the
    separate position and strain entries elementwise.

    ``grad_e_L_{t,n}_wrt_positions_and_strain`` is what jqmc_gfmc actually
    invokes for force+stress runs, and it takes its own ``_dispatch_gfmc("both")``
    route. Before this node the fused entries had ZERO direct evidence: only the
    separate ``_wrt_positions`` / ``_wrt_strain`` routes were oracle-tested, so a
    defect confined to the fused assembly (wrong tuple order, a dropped stress
    term, a stale geminal inverse) reached production silently.
    """
    ham = _build_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    fused = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions_and_strain")
    separate_positions = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions")
    separate_strain = getattr(gfmc, f"grad_e_L_{driver}_wrt_strain")

    dR_f, dr_up_f, dr_dn_f, stress_f = fused(ham, r_up, r_dn, RT, alat)
    dR_s, dr_up_s, dr_dn_s = separate_positions(ham, r_up, r_dn, RT, alat)
    stress_s = separate_strain(ham, r_up, r_dn, RT, alat)

    # Same arithmetic reached by two routes of the same backend: only the fusion
    # (and hence the reduction order) differs.
    atol, rtol = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    _gdd_assert_allclose(np.asarray(dR_f), np.asarray(dR_s), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_up_f), np.asarray(dr_up_s), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(dr_dn_f), np.asarray(dr_dn_s), atol=atol, rtol=rtol)
    _gdd_assert_allclose(np.asarray(stress_f), np.asarray(stress_s), atol=atol, rtol=rtol)

    # Non-vacuity: a fused route that returned zeros would satisfy an equality
    # against a zero sibling, so require the compared quantities to be alive.
    assert float(np.max(np.abs(np.asarray(dR_f)))) > 0.0
    assert float(np.max(np.abs(np.asarray(stress_f)))) > 0.0


@pytest.mark.light
@pytest.mark.parametrize("gem", ["AGPn", "AGP_MO"])
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_force_analytic_vs_autograd_non_sd_geminal(driver, gem, request):
    """Full GFMC/LRDMC forces match AD; mixed retains full fidelity for
    the NON-SD geminal representations production actually branches on.

    Every other analytic force oracle in this file builds a plain SD geminal, so
    the AGPn / AGP_MO lambda leg of the S'-reduction (the intended MO-optimization
    representation) reached production without an autodiff comparison.
    """
    if is_mixed_precision_enabled():
        # Keep the battery's separate same-policy tolerance consistency control.
        zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb")
        battery_tolerance = _gdd_get_tolerance_min(zones, "strict")
        assert get_tolerance_min(zones, "medium") == battery_tolerance, (
            "GFMC battery tolerance drifted from its full/mixed arms"
        )

        def analytical(fresh):
            values = _representation_forces(fresh, driver)
            assert set(values) == {"nuclear", "electron-up", "electron-down"}, values.keys()
            gd = fresh.wavefunction_data.geminal_data
            shapes = {
                "nuclear": np.asarray(fresh.structure_data.positions).shape,
                "electron-up": (int(gd.num_electron_up), 3),
                "electron-down": (int(gd.num_electron_dn), 3),
            }
            for name, value in values.items():
                assert value.shape == shapes[name], (name, value.shape, shapes[name])
                assert np.all(np.isfinite(value)), (name, value)
            # Original non-vacuity uses >0, and is retained in both callbacks.
            assert float(np.max(np.abs(values["nuclear"]))) > 0.0
            return values

        mixed, full = mixed_vs_full(_full_representation_source(gem), analytical)
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for quantity, (mixed_block, full_block) in zip(sorted(mixed), blocks, strict=True):
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    stem, alat, seed, scale = "H2_ae_ccpvdz_cart.h5", 1.0, 0, 0.9
    ham = _build_ham_gem(stem, gem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=scale, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=scale, size=(nd, 3)))
    RT = _rot(seed + 1)

    active0 = _extract_active_positions_from_ham(ham)
    e_L_fn = _e_L_t_ae_jnone if driver == "t" else _e_L_n_ae_jnone

    def f_active(active, a, b):
        return e_L_fn(_inject_active_positions(ham, active), a, b, RT, alat)

    gR_oracle, gru_oracle, grd_oracle = _jit_grad_pos(f_active, active0, r_up, r_dn)

    analytic = getattr(gfmc, f"grad_e_L_{driver}_wrt_positions")
    dR, dr_up, dr_dn = analytic(ham, r_up, r_dn, RT, alat)

    battery_tolerance = _gdd_get_tolerance_min(("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "strict")
    atol, rtol = get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb"), "medium" if is_mixed_precision_enabled() else "strict"
    )
    assert (atol, rtol) == battery_tolerance, "GFMC battery tolerance drifted from its full/mixed arms"
    np.testing.assert_allclose(np.asarray(dR), np.asarray(gR_oracle), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(dr_up), np.asarray(gru_oracle), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(dr_dn), np.asarray(grd_oracle), atol=atol, rtol=rtol)
    # Non-vacuity: the compared gradients must be alive for this representation.
    assert float(np.max(np.abs(np.asarray(dR)))) > 0.0
