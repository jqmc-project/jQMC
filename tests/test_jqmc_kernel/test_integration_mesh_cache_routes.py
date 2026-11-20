"""Integration gates for projection-scoped XLA and Rust mesh-cache routes.

Every projection-scoped candidate cache -- on both backends -- engages through a
branch whose two sides return the SAME numbers.  A cache that is never reached,
is gated out, or was never written for a channel is therefore invisible to every
numerical test in the tree; it shows up only as wall-clock.  The route counters
therefore complement numerical backend-parity tests with cache-engagement proof.

The analytic force/stress path (``derivatives/gfmc.py``) is DELIBERATELY not
gated here: it is a single ``jax.grad`` evaluation at one configuration, outside
the projection ``lax.while_loop`` and with no hop sequence to amortize a
candidate cache over -- there is no "previous hop" state to cache. The
``kinetic_mesh_state`` the interval loop builds lives only in that loop's
carry and is discarded on exit; even if it were threaded out, its mesh rotation
is not guaranteed to match the force step's, so a cache built for one would not
validly serve the other. The site label still records ``"force"`` (see below)
so this can be reconfirmed by inspection, but no test in this file asserts on
it. Dense DTN mesh cost that grows with system size has a separate geometry-mesh
implementation and is not evidence that this single-shot route needs a cache.

The existing route assertions are POSITIVE (``assert counts[key] >= 1`` --
"the fast route ran at least once"), which passes even when almost every
sub-step took the slow route.  The gates here are NEGATIVE: for a given
(fixture, driver) the slow route must run ZERO times.

Backends
--------
* XLA -- ``jqmc._route_probe`` counters, bumped at TRACE time from both sides of
  each Python-static ``is None`` branch.  Because they are trace-time, a jit
  cache left over from an earlier test would keep the kernel from being
  re-traced and the gate would read all-zeros -- a false green.  Every XLA test
  below therefore calls ``jax.clear_caches()`` before the run, and asserts the
  channel was OBSERVED (cached or not) before asserting it was cached.
* Rust -- ``jqmc_rs.gfmc_route_counters()`` indices 41-44, the real-path
  derivative-cache miss counters (the twins of the complex ``*_GRAD_SLOW``
  trio).

For Rust, cache eligibility is also a precision contract.  The native delta
chain is fp64.  In mixed precision it cannot reproduce XLA's fp32
``jastrow_grad_lap`` arithmetic, so affected J1/J2 and DTN channels deliberately
take a full rebuild while determinant and J3 caches remain active.

Sites
-----
The counters are trace-time, so an uncached count of 1 cannot by itself
distinguish "once per interval by design" from "every sub-step". The ``site``
label supplies that: ``projection`` (must be cached), ``substep0`` (uncached by
design -- it BUILDS the state), ``observable`` / ``force`` (measurement-point
and derivative paths).
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

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc._route_probe as route_probe
import jqmc.jastrow_factor as jastrow_factor
from jqmc._route_probe import CHANNELS, route_counters, route_counters_reset
from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import _run_gfmc

pytestmark = [pytest.mark.with_chdir]

# Short runs: the gate reads which route COMPILED, not how well it converged.
_GATE_STEPS = 3
_GATE_WALKERS = 2


def _uncached_channels(mesh: str, site: str) -> set[str]:
    """Channels that compiled the uncached route on ``mesh`` at ``site``."""
    counts = route_counters()
    return {ch for ch in CHANNELS if counts.get((ch, mesh, False, site), 0) > 0}


def _observed_channels(mesh: str, site: str) -> set[str]:
    """Channels that reported a route at all -- guards against a false green."""
    counts = route_counters()
    return {ch for ch in CHANNELS if counts.get((ch, mesh, True, site), 0) or counts.get((ch, mesh, False, site), 0)}


def _run_xla_gate(driver, pattern, **kw):
    """Run one production driver on XLA with a clean jit cache and fresh counters."""
    if pattern == "mol_ecp_single_spin_j12":
        # Keep this route-only fixture local: global rs_ham patterns also
        # parametrize unrelated MCMC transition tests.
        ham = _build_ham_for_cell("Li_ecp_ccpvdz_cart.h5", True, None, "SD")
        gd = ham.wavefunction_data.geminal_data
        assert (int(gd.num_electron_up), int(gd.num_electron_dn)) == (1, 0)
        assert ham.coulomb_potential_data.ecp_flag
        # Li's live projector distinguishes this case from zero-projector H ECP.
        # J2 has no physical pairs here; its route is checked as a cache consumer.
        r_up = jnp.asarray(ham.structure_data.positions[:1]) + jnp.asarray([0.31, 0.42, 0.53])
        nonlocal_sum = compute_ecp_non_local_parts_all_pairs(
            ham.coulomb_potential_data,
            ham.wavefunction_data,
            r_up,
            jnp.empty((0, 3)),
            jnp.eye(3),
        )[3]
        assert np.isfinite(nonlocal_sum) and nonlocal_sum != 0, "single-spin ECP fixture must have a live projector"
    else:
        ham = rs_ham(pattern)
    jax.clear_caches()
    route_counters_reset()
    _run_gfmc(driver, ham, 34356, "jax", _GATE_STEPS, num_walkers=_GATE_WALKERS, **kw)
    return ham


# --------------------------------------------------------------------------- #
# XLA: the kinetic 6N lattice
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        pytest.param("mol_ecp_cart_j12", {"det", "j1", "j2"}, id="j1-j2"),
        pytest.param("mol_ecp_cart_j123", {"det", "j1", "j2", "j3"}, id="j1-j2-j3"),
        pytest.param("mol_ecp_cart_jdtn", {"det", "j1", "j2", "dtn"}, id="j1-j2-dtn"),
        pytest.param("pbc_gamma_sphepgto_j12", {"det", "j1", "j2"}, id="pbc-gamma-j1-j2"),
        pytest.param("twist_sphepgto_j12", {"det", "j1", "j2"}, id="twist-j1-j2"),
        pytest.param("mol_ecp_single_spin_j12", {"det", "j1", "j2"}, id="single-spin-j1-j2"),
    ],
)
def test_xla_kinetic_mesh_caches_every_jastrow_channel(driver, pattern, expected):
    """Check all active cache reports and independently observe DTN's dense kernel."""
    dense_sites = []
    dense_kernel = jastrow_factor._compute_delta_J_DTN_mesh_dense

    def observe_dense_dtn(*args, **kwargs):
        dense_sites.append(route_probe._site)
        return dense_kernel(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        if "dtn" in expected:
            patcher.setattr(jastrow_factor, "_compute_delta_J_DTN_mesh_dense", observe_dense_dtn)
        _run_xla_gate(driver, pattern)

    observed = _observed_channels("kinetic", "projection")
    assert observed == expected, f"wrong kinetic channel coverage for {pattern}: {sorted(observed)}"
    uncached = _uncached_channels("kinetic", "projection")
    assert uncached == set(), (
        f"channels on the UNCACHED kinetic-mesh route during projection: {sorted(uncached)} (observed {sorted(observed)})"
    )
    if "dtn" in expected:
        assert dense_sites, "the dense DTN observer never saw a control trace outside projection"
        assert "projection" not in dense_sites, "the dense DTN kernel traced during kinetic projection"


@pytest.mark.light
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        pytest.param("mol_ecp_cart_j12", {"det", "j1", "j2"}, id="j1-j2"),
        pytest.param("mol_ecp_cart_j123", {"det", "j1", "j2", "j3"}, id="j1-j2-j3"),
        pytest.param("mol_ecp_cart_jdtn", {"det", "j1", "j2", "dtn"}, id="j1-j2-dtn"),
        pytest.param("pbc_gamma_sphepgto_j12", {"det", "j1", "j2"}, id="pbc-gamma-j1-j2"),
        pytest.param("twist_sphepgto_j12", {"det", "j1", "j2"}, id="twist-j1-j2"),
        pytest.param("mol_ecp_single_spin_j12", {"det", "j1", "j2"}, id="single-spin-j1-j2"),
    ],
)
def test_xla_ecp_mesh_caches_every_jastrow_channel(driver, pattern, expected):
    """Same complete active-channel obligation for the ECP lattice."""
    _run_xla_gate(driver, pattern)
    observed = _observed_channels("ecp", "projection")
    assert observed == expected, f"wrong ECP channel coverage for {pattern}: {sorted(observed)}"
    uncached = _uncached_channels("ecp", "projection")
    assert uncached == set(), (
        f"channels on the UNCACHED ECP-mesh route during projection: {sorted(uncached)} (observed {sorted(observed)})"
    )


@pytest.mark.light
def test_xla_force_path_evaluates_no_mesh_route_itself():
    """The analytic force path evaluates NO kinetic-mesh route of its own.

    The novar=0 measurement split has no force-side propagation-mesh
    derivative: HF force/stress differentiate the continuous measurement local
    energy, while S-prime membership remains propagation state only. Site
    "force" therefore observes NOTHING.

    If any channel reports at site "force", a propagation-only mesh derivative has leaked back into the
    measurement estimator or a new force-side mesh consumer needs an explicit
    design.
    """
    _run_xla_gate("t", "mol_ecp_cart_jdtn", pos_deriv=True)
    observed = _observed_channels("kinetic", "force")
    assert observed == set(), (
        f"the force path evaluated kinetic-mesh routes itself: {sorted(observed)} -- "
        "the novar=0 measurement derivative must not consume the propagation mesh"
    )


# --------------------------------------------------------------------------- #
# Rust: real-path derivative caches
# --------------------------------------------------------------------------- #

# Index map of the real-path derivative-cache miss counters appended to
# ``gfmc_route_counters()`` (see the PyO3 docstring in ``jqmc/rs/src/lib.rs``).
_RS_SLOW = {"det": 41, "j12": 42, "j3": 43, "dtn": 44}
_RS_DTN_TABPIP = {"builds": 45, "advances": 46, "hits": 47}
_RS_J12_TABPIP_KIN = {"builds": 48, "advances": 49, "hits": 50, "fallbacks": 51}
_RS_J12_TABPIP_ECP = {"builds": 52, "advances": 53, "hits": 54, "fallbacks": 55}


# Raw-AO molecular AGPs with Jastrow recipes of the shared ``rs_ham`` table
# (which has only Jastrow-free raw-AO AGPs, whose J1/J2 channel is the counted
# J1/J2 slow route by construction).
_RAW_AO_ENGAGE_RECIPES = {
    "mol_cart_agp_jdtn": ("N_ae_ccpvdz_cart.h5", True, {"dtn": True}, "AGP"),
    "mol_ecp_sphe_agp_j12": ("water_ccecp_ccpvqz.h5", True, None, "AGP"),
}


def _engage_ham(pattern):
    if pattern in _RAW_AO_ENGAGE_RECIPES:
        return _build_ham_for_cell(*_RAW_AO_ENGAGE_RECIPES[pattern])
    return rs_ham(pattern)


def _rs_slow_counts():
    import jqmc_rs

    counters = jqmc_rs.gfmc_route_counters()
    return {name: counters[idx] for name, idx in _RS_SLOW.items()}


@pytest.mark.light
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("mol_cart_sd_jdtn", id="molecular_gto_dtn"),
        pytest.param("pbc_gamma_sphepgto_j123", id="periodic_pgto_j3"),
        # The molecular RAW-AO geminal (AO-basis AGP) rebuilt the
        # determinant grad/lap on every sub-step (counter 41) before the
        # basis-generic derivative cache was enabled for it (all-electron
        # Cartesian and ECP spherical).
        pytest.param("mol_cart_agp_jdtn", id="molecular_raw_ao_agp"),
        pytest.param("mol_ecp_sphe_agp_j12", id="molecular_raw_ao_agp_ecp_sphe"),
    ],
)
def test_rust_real_path_derivative_caches_engage(pattern):
    """Every supported real-path derivative cache must engage in every precision mode.

    A delta update may accumulate in a different order from a full fp32 rebuild,
    but that numerical difference belongs to backend-parity tolerance. It must
    not disable the production cache or turn mixed precision into a slow route.
    """
    import jqmc_rs

    ham = _engage_ham(pattern)
    jqmc_rs.gfmc_route_counters_reset()
    _run_gfmc("n", ham, 34356, "rust", _GATE_STEPS, num_walkers=_GATE_WALKERS)
    slow = _rs_slow_counts()
    offenders = {k: v for k, v in slow.items() if v}
    assert offenders == {}, f"real-path derivative caches took the slow route: {offenders}"
    if pattern in _RAW_AO_ENGAGE_RECIPES:
        geminal = ham.wavefunction_data.geminal_data
        assert getattr(geminal.orb_data_up_spin, "mo_coefficients", None) is None, f"{pattern}: not a raw-AO geminal"
        assert jqmc_rs.gfmc_route_counters()[13] > 0, f"{pattern}: basis-generic det grad/lap cache was not built"
    if pattern == "mol_cart_sd_jdtn":
        counters = jqmc_rs.gfmc_route_counters()
        tabpip = {name: counters[idx] for name, idx in _RS_DTN_TABPIP.items()}
        assert all(tabpip.values()), f"DTN candidate table did not build/advance/serve rows: {tabpip}"
        assert counters[0] == 0, "kinetic DTN rows fell back to per-candidate geometry recomputation"


@pytest.mark.light
@pytest.mark.parametrize(
    ("pattern", "has_ecp"),
    [
        pytest.param("mol_cart_sd_j12", False, id="molecular-exp-exp"),
        pytest.param("mol_cart_sd_j12_pade", False, id="molecular-pade-pade"),
        pytest.param("mol_cart_sd_j12_p1e2", False, id="molecular-pade-exp"),
        pytest.param("mol_cart_sd_j12_e1p2", False, id="molecular-exp-pade"),
        pytest.param("pbc_gamma_sphepgto_j12", True, id="periodic-real-exp-exp"),
        pytest.param("pbc_gamma_sphepgto_j12_pade", True, id="periodic-real-pade-pade"),
        pytest.param("twist_sphepgto_j12", True, id="periodic-complex-exp-exp"),
        pytest.param("twist_sphepgto_j12_pade", True, id="periodic-complex-pade-pade"),
    ],
)
def test_rust_j12_candidate_table_covers_forms_boundaries_and_twist(pattern, has_ecp):
    """J1/J2 tabpip must build, advance and serve every production family."""
    import jqmc_rs

    ham = rs_ham(pattern)
    jqmc_rs.gfmc_route_counters_reset()
    _run_gfmc(
        "n",
        ham,
        34356,
        "rust",
        _GATE_STEPS,
        random_mesh=False,
        num_walkers=1,
    )
    counters = jqmc_rs.gfmc_route_counters()
    kinetic = {name: counters[idx] for name, idx in _RS_J12_TABPIP_KIN.items()}
    assert all(kinetic[name] > 0 for name in ("builds", "advances", "hits")), (
        f"kinetic J1/J2 table did not fully engage for {pattern}: {kinetic}"
    )
    assert kinetic["fallbacks"] == 0, f"kinetic J1/J2 fallback for {pattern}: {kinetic}"
    ecp = {name: counters[idx] for name, idx in _RS_J12_TABPIP_ECP.items()}
    if has_ecp:
        assert all(ecp[name] > 0 for name in ("builds", "advances", "hits")), (
            f"ECP J1/J2 table did not fully engage for {pattern}: {ecp}"
        )
        assert ecp["fallbacks"] == 0, f"ECP J1/J2 fallback for {pattern}: {ecp}"
    else:
        assert ecp == {"builds": 0, "advances": 0, "hits": 0, "fallbacks": 0}


# --------------------------------------------------------------------------- #
# Meta: the gate must itself be un-fakeable
# --------------------------------------------------------------------------- #


@pytest.mark.light
def test_route_probe_strict_mode_raises_on_an_uncached_route():
    """``require_mesh_cache()`` must turn an uncached route into a failure.

    Without this, a future refactor could silence the gate by making
    ``note_route`` a no-op and every test above would still pass.
    """
    from jqmc._route_probe import MeshCacheMissing, note_route, require_mesh_cache

    route_counters_reset()
    with pytest.raises(MeshCacheMissing), require_mesh_cache():
        note_route("dtn", "kinetic", cached=False)
    # ... and must NOT raise for a cached route.
    with require_mesh_cache():
        note_route("dtn", "kinetic", cached=True)


@pytest.mark.light
def test_route_probe_env_flag_is_documented():
    """The env-var ledger in ``_setting.py`` must list the strict flag."""
    from pathlib import Path

    ledger = Path(__file__).resolve().parents[2] / "jqmc" / "_setting.py"
    assert "JQMC_REQUIRE_MESH_CACHE" in ledger.read_text(), (
        "JQMC_REQUIRE_MESH_CACHE is read by jqmc/_route_probe.py but is absent "
        "from the environment-variable ledger in jqmc/_setting.py"
    )
    assert os.environ.get("JQMC_REQUIRE_MESH_CACHE") in (None, "", "0"), (
        "the gate suite must run with strict mode OFF -- it asserts on counters, "
        "and a strict raise would mask which channels were observed"
    )
