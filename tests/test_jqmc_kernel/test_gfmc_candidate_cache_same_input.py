"""Strict cache-leaf checks at observed production GFMC_t hop configurations."""

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
from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.jqmc_gfmc import GFMC_t
from tests._gfmc_n_same_input import leaf_reference
from tests.test_jqmc_kernel import test_gfmc_dltmove_propagation_reference as propagation
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.mark.parametrize(
    "stem,spec",
    [
        pytest.param("H2_ecp_ccpvdz_cart.h5", {"j3": "pgto"}, id="h2-ecp-j123"),
        pytest.param("QEPROJ:bspline:Li_gamma:ECP", None, id="li-gamma-bspline-j12"),
        pytest.param("Li_ae_ccpvdz_cart.h5", {"j3": "pgto"}, id="li-ae-j123"),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_gfmc_t_candidate_cache_at_actual_inputs(stem, spec, monkeypatch):
    """Actual cached kinetic/ECP orbitals equal fresh same-mode values at their own inputs.

    Molecular H2 ccECP/J123, Gamma Li ccECP B-spline/J12 and molecular Li AE/J123
    use one walker, seed 3446, four intervals, tau 0.10, alat 0.30 and rotating
    meshes. The existing observer records the actual categorical-selection call;
    every subsequent cached hop is checked, without advancing a reference walk.
    Kinetic candidate positions equal current positions plus the fixed interval
    shifts exactly. Both kinetic and ECP candidate orbital tables match fresh
    orbital evaluation under the existing strict AO/MO/determinant-ratio policy.
    Full-recomputed Hamiltonian leaves additionally check candidate weights and
    lattice energy with the existing strict-full/medium-mixed propagation policy.
    No mixed result is compared to a full-mode reference.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham = _build_ham_for_cell(stem, True, spec)
    ham.sanity_check()
    fields = propagation._T_FIELDS

    class CacheObserver(propagation.ObserveGFMCTHops):
        def _receive(self, *values):
            row = {name: np.array(value, copy=True) for name, value in zip(fields, values[: len(fields)], strict=True)}
            row["counter"] = int(values[-3])
            row["state"] = values[-2]
            row["ecp_state"] = values[-1]
            self.hops.append(row)

    monkeypatch.setattr(propagation, "_T_FIELDS", fields + ("projection_counter", "kinetic_mesh_state", "ecp_mesh_state"))
    observer = CacheObserver()
    driver = GFMC_t(
        hamiltonian_data=ham,
        num_walkers=1,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=3446,
        tau=0.10,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=False,
        non_local_move="tmove",
    )
    observer.install(monkeypatch)
    driver.run(num_mcmc_steps=4)
    jax.effects_barrier()
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_ratio"), "strict")
    value_atol, value_rtol = propagation._tolerance()
    cached = [row for row in observer.hops if row["state"] is not None]
    assert cached, "no cached production hop was observed"
    assert any(row["state"] is None for row in observer.hops), "no initial full-kernel hop was observed"
    gem = ham.wavefunction_data.geminal_data
    for row in cached:
        state = row["state"]
        assert row["counter"] > 1
        assert (row["ecp_state"] is not None) == bool(ham.coulomb_potential_data.ecp_flag)
        for cache in (state, row["ecp_state"]):
            if cache is None:
                continue
            for spin in ("up", "dn"):
                positions = getattr(cache, f"candidate_{spin}_positions")
                if cache is state:
                    reference_positions = row[f"r_{spin}_carts"][:, None, :] + np.asarray(state.shifts)[None, :, :]
                    np.testing.assert_array_equal(positions, reference_positions)
                actual = getattr(cache.det_candidate_state, f"orb_{spin}_candidates")
                reference = np.asarray(
                    gem.compute_orb_api(getattr(gem, f"orb_data_{spin}_spin"), jnp.asarray(positions).reshape(-1, 3))
                )
                assert actual.size > 0 and np.any(reference != 0)
                np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg="cached orbital")
        leaves = leaf_reference(
            ham, row["r_up_carts"], row["r_dn_carts"], row["RT"], row["ecp_RT"], 0.30, non_local_move="tmove"
        )
        np.testing.assert_allclose(row["p_list"], leaves[2], atol=value_atol, rtol=value_rtol, equal_nan=False)
        np.testing.assert_allclose(
            row["e_L"], leaves[3] + jnp.sum(leaves[2]), atol=value_atol, rtol=value_rtol, equal_nan=False
        )


#: Size of the injected wrong-leaf fault relative to the largest cached orbital
#: value, the size of a driver-level stale-cache fault (about 7e-4).
_WRONG_LEAF_FRACTION = 7.0e-4


def _wrong_leaf_advance(original):
    """Kinetic mesh advance whose up-spin orbital leaf is pulled toward its pre-advance value.

    The returned ``orb_up_candidates`` is ``new + lam * (old - new)`` with
    ``lam`` chosen so the largest displacement is ``_WRONG_LEAF_FRACTION`` of
    the largest ``|new|``: the leaf of the previous sub-step leaks into the
    one the hop consumes, scaled to that fault size. Hops that did
    not move an up electron keep ``old == new`` and are unchanged.
    """

    def advance(*args, **kwargs):
        state = kwargs["state"] if "state" in kwargs else args[1]
        new_state = original(*args, **kwargs)
        old = state.det_candidate_state.orb_up_candidates
        new = new_state.det_candidate_state.orb_up_candidates
        gap = jnp.max(jnp.abs(old - new))
        safe_gap = jnp.where(gap > 0, gap, 1)
        lam = jnp.where(gap > 0, _WRONG_LEAF_FRACTION * jnp.max(jnp.abs(new)) / safe_gap, 0)
        faulty = (new + lam.astype(new.dtype) * (old - new)).astype(new.dtype)
        return new_state.replace(det_candidate_state=new_state.det_candidate_state.replace(orb_up_candidates=faulty))

    return advance


@pytest.mark.parametrize(
    "stem,spec",
    [
        pytest.param("H2_ecp_ccpvdz_cart.h5", {"j3": "pgto"}, id="h2-ecp-j123"),
        pytest.param("Li_ae_ccpvdz_cart.h5", {"j3": "pgto"}, id="li-ae-j123"),
    ],
)
def test_gfmc_t_candidate_cache_rejects_a_7e4_wrong_leaf(stem, spec, monkeypatch):
    """Negative control: a ~7e-4 stale-leaf leak into the consumed up-spin orbital table is rejected.

    The production GFMC_t kinetic mesh advance is
    wrapped so the up-spin candidate orbital table that the next hop consumes
    is displaced toward its pre-advance (previous sub-step) value by 7e-4 of
    its largest entry -- the size of a driver-level cache defect,
    below the medium mixed trajectory pin (rtol 2e-3). The unchanged
    same-input observer above must fail on the cached-orbital comparison in
    each precision mode with its own existing strict tolerance.
    """
    import jqmc.jqmc_gfmc as gfmc_mod

    monkeypatch.setattr(
        gfmc_mod,
        "_advance_discretized_kinetic_mesh_streaming_state",
        _wrong_leaf_advance(gfmc_mod._advance_discretized_kinetic_mesh_streaming_state),
    )
    with pytest.raises(AssertionError, match="cached orbital"):
        test_gfmc_t_candidate_cache_at_actual_inputs(stem, spec, monkeypatch)
