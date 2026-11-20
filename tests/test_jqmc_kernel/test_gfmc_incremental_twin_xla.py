"""Incremental-vs-full-recompute twin for the GFMC projection interval (XLA).

IMPL-GFMC-INCREMENTAL-TWIN-001, XLA leg. The production XLA projection body
(``_projection_t_core`` / GFMC_n ``_body_step_core``) consumes the carried
interval potential state; the environment switch
``JQMC_GFMC_POT_FULL_RECOMPUTE=1`` statically selects the debug/oracle
full-recompute route -- the SAME body with per-sub-step dispatch calls.

Twin contract (same backend, same seeds):

1. **Route-invariant walk** (exact): ``b_L`` / hop cdf never touch the carried
   potential, so the walkers and PRNG keys are BYTE-identical across routes.
2. **Tight-absolute agreement** of the stored per-step ``w_L`` / ``e_L``
   (``ATOL_GFMC_INCREMENTAL_TWIN``; the routes are exact modulo fp
   reordering).
3. **Liveness (anti-vacuity)**: the module-level trace witness
   ``_GFMC_POT_TRACE_WITNESS`` advances when the production body is traced
   with the incremental branch active, and the driver reports strictly
   positive per-walker ``_latest_gfmc_pot_updates`` -- the twin route reports
   zeros. A test cannot pass by silently tracing the full-recompute branch.

This suite drives the XLA ``lax.while_loop`` / ``fori_loop`` projection (the
autouse conftest fixture pins ``JQMC_FLAG_RUST_ON_CPU=0`` for non-rs-kernel
tests), i.e. the verification backend of the Rust-side twin in
``tests/test_rs_kernel/test_gfmc_incremental_twin.py``.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import (
    ATOL_GFMC_INCREMENTAL_TWIN,
    GFMC_MIN_COLLECT_STEPS,
    RTOL_GFMC_INCREMENTAL_TWIN,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._ecp_core_electrons import core_electrons_of

jax.config.update("jax_enable_x64", True)

_STEPS = 3


def _build_lithium_ham():
    """lithium.h5 (2-atom Li ccECP, PBC) + J1/J2 -- the PBC Ewald fixture."""
    structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trexio_example_files",
            "lithium.h5",
        ),
        store_tuple=True,
    )
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
    )
    ham.sanity_check()
    return ham


def _run(driver, ham, full_recompute, monkeypatch, start=None):
    """Run one driver; ``start`` optionally replaces every walker's initial ``(r_up, r_dn)``."""
    if full_recompute:
        monkeypatch.setenv("JQMC_GFMC_POT_FULL_RECOMPUTE", "1")
    else:
        monkeypatch.delenv("JQMC_GFMC_POT_FULL_RECOMPUTE", raising=False)
    common = dict(
        hamiltonian_data=ham,
        num_walkers=2,
        num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
        mcmc_seed=3446,
        alat=0.30,
        non_local_move="tmove",
        random_discretized_mesh=True,
    )
    if driver == "t":
        g = GFMC_t(**common, tau=0.05)
    else:
        g = GFMC_n(**common, num_mcmc_per_measurement=6, E_scf=-1.0)
    if start is not None:
        r_up, r_dn = start
        n_walkers = g._latest_r_up_carts.shape[0]
        dtype = g._latest_r_up_carts.dtype
        g._latest_r_up_carts = jnp.asarray(np.broadcast_to(r_up, (n_walkers,) + r_up.shape), dtype=dtype)
        g._latest_r_dn_carts = jnp.asarray(np.broadcast_to(r_dn, (n_walkers,) + r_dn.shape), dtype=dtype)
    g.run(num_mcmc_steps=_STEPS)
    return g


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_xla_incremental_twin(driver, monkeypatch):
    """XLA production (incremental) == XLA debug full-recompute twin + liveness."""
    import jqmc.jqmc_gfmc as gfmc_mod

    ham = _build_lithium_ham()

    g_twin = _run(driver, ham, full_recompute=True, monkeypatch=monkeypatch)
    twin_updates = np.asarray(getattr(g_twin, "_latest_gfmc_pot_updates", np.zeros(2)))

    witness_before = gfmc_mod._GFMC_POT_TRACE_WITNESS
    g_prod = _run(driver, ham, full_recompute=False, monkeypatch=monkeypatch)
    witness_after = gfmc_mod._GFMC_POT_TRACE_WITNESS

    # Leg 3: liveness / anti-vacuity.
    assert witness_after > witness_before, "incremental branch was never traced"
    prod_updates = np.asarray(g_prod._latest_gfmc_pot_updates)
    assert np.all(prod_updates > 0), prod_updates
    assert np.all(twin_updates == 0), twin_updates

    # Leg 1: route-invariant walk, byte for byte.
    np.testing.assert_array_equal(np.asarray(g_prod._latest_r_up_carts), np.asarray(g_twin._latest_r_up_carts))
    np.testing.assert_array_equal(np.asarray(g_prod._latest_r_dn_carts), np.asarray(g_twin._latest_r_dn_carts))
    np.testing.assert_array_equal(np.asarray(g_prod._jax_PRNG_key_list), np.asarray(g_twin._jax_PRNG_key_list))

    # Leg 2: stored per-step diagonal observables at the tight twin pin.
    np.testing.assert_allclose(
        np.asarray(g_prod._stored_w_L),
        np.asarray(g_twin._stored_w_L),
        atol=ATOL_GFMC_INCREMENTAL_TWIN,
        rtol=RTOL_GFMC_INCREMENTAL_TWIN,
        err_msg="w_L: incremental vs full-recompute twin",
    )
    np.testing.assert_allclose(
        np.real(np.asarray(g_prod._stored_e_L)),
        np.real(np.asarray(g_twin._stored_e_L)),
        atol=ATOL_GFMC_INCREMENTAL_TWIN,
        rtol=RTOL_GFMC_INCREMENTAL_TWIN,
        err_msg="e_L: incremental vs full-recompute twin",
    )


def _build_h4_pbc_ae_ham():
    """H4_pbc_ae_cart.h5 (4-atom H chain, all-electron, PBC) + J1/J2 -- the AE Ewald ``ei_disc`` fixture."""
    structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trexio_example_files",
            "H4_pbc_ae_cart.h5",
        ),
        store_tuple=True,
    )
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
    )
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
    )
    ham.sanity_check()
    return ham


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_xla_incremental_twin_ae_pbc(driver, monkeypatch):
    """Driver-level incremental vs full-recompute twin on the periodic all-electron H4 cell.

    Same contract as ``test_gfmc_xla_incremental_twin`` on a second periodic
    fixture: 4 nuclei, no ECP, so the AE Ewald ion-electron rows (the
    ``ei_disc`` floor branch) are the carried potential. Production
    (incremental, env unset) and ``JQMC_GFMC_POT_FULL_RECOMPUTE=1`` run with
    the same seed. Walkers and PRNG keys are byte-identical; stored w_L, e_L
    and e_L2 agree at ATOL/RTOL_GFMC_INCREMENTAL_TWIN; the production route
    traces the incremental branch and reports positive per-walker update
    counts while the twin reports zeros.
    """
    import jqmc.jqmc_gfmc as gfmc_mod

    ham = _build_h4_pbc_ae_ham()
    assert bool(ham.structure_data.pbc_flag), "H4_pbc_ae_cart.h5 must be periodic"
    assert not bool(ham.coulomb_potential_data.ecp_flag), "H4_pbc_ae_cart.h5 must be all-electron"
    assert len(ham.structure_data.atomic_numbers) == 4

    g_twin = _run(driver, ham, full_recompute=True, monkeypatch=monkeypatch)
    twin_updates = np.asarray(getattr(g_twin, "_latest_gfmc_pot_updates", np.zeros(2)))

    witness_before = gfmc_mod._GFMC_POT_TRACE_WITNESS
    g_prod = _run(driver, ham, full_recompute=False, monkeypatch=monkeypatch)
    witness_after = gfmc_mod._GFMC_POT_TRACE_WITNESS

    assert witness_after > witness_before, "incremental branch was never traced"
    prod_updates = np.asarray(g_prod._latest_gfmc_pot_updates)
    assert np.all(prod_updates > 0), prod_updates
    assert np.all(twin_updates == 0), twin_updates

    np.testing.assert_array_equal(np.asarray(g_prod._latest_r_up_carts), np.asarray(g_twin._latest_r_up_carts))
    np.testing.assert_array_equal(np.asarray(g_prod._latest_r_dn_carts), np.asarray(g_twin._latest_r_dn_carts))
    np.testing.assert_array_equal(np.asarray(g_prod._jax_PRNG_key_list), np.asarray(g_twin._jax_PRNG_key_list))

    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        prod_values = np.asarray(getattr(g_prod, name))
        twin_values = np.asarray(getattr(g_twin, name))
        assert prod_values.shape == twin_values.shape and prod_values.size > 0, name
        np.testing.assert_allclose(
            prod_values,
            twin_values,
            atol=ATOL_GFMC_INCREMENTAL_TWIN,
            rtol=RTOL_GFMC_INCREMENTAL_TWIN,
            equal_nan=False,
            err_msg=f"{name}: incremental vs full-recompute twin (H4 AE PBC)",
        )


# --------------------------------------------------------------------------- #
# Driver twin in which the carried ei_disc row update is consumed.
# --------------------------------------------------------------------------- #
def _floor_start(ham):
    """Same-spin pairs at +-0.1 bohr along z around protons 0 (up) and 2 (dn).

    Every electron starts inside the ``alat`` floor of a proton and across the
    same-spin exchange node from its partner, so hops of these electrons flip
    kinetic signs and the ``max(ZV, ei_disc)`` composition selects carried
    ``ei_disc`` rows of moved electrons. From the default random start of
    ``test_gfmc_xla_incremental_twin_ae_pbc`` the stored observables are
    bit-identical under both carried-row faults below (the branch is never
    selected), which is why this start is imposed.
    """
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0])
    r_up = np.stack([nuclei[0] + 0.1 * ez, nuclei[0] - 0.1 * ez])
    r_dn = np.stack([nuclei[2] + 0.1 * ez, nuclei[2] - 0.1 * ez])
    return r_up, r_dn


def _assert_driver_twin(g_prod, g_twin, label):
    """Stored diagonal observables at the twin pin (real and imaginary parts), then the exact walk."""
    for name in ("_stored_w_L", "_stored_e_L", "_stored_e_L2"):
        prod_values = np.asarray(getattr(g_prod, name))
        twin_values = np.asarray(getattr(g_twin, name))
        assert prod_values.shape == twin_values.shape and prod_values.size > 0, name
        assert np.all(np.isfinite(prod_values)) and np.all(np.isfinite(twin_values)), name
        for part, take in (("Re", np.real), ("Im", np.imag)):
            np.testing.assert_allclose(
                take(prod_values),
                take(twin_values),
                atol=ATOL_GFMC_INCREMENTAL_TWIN,
                rtol=RTOL_GFMC_INCREMENTAL_TWIN,
                equal_nan=False,
                err_msg=f"{part}({name}): incremental vs full-recompute twin ({label})",
            )
    for name in ("_latest_r_up_carts", "_latest_r_dn_carts", "_jax_PRNG_key_list"):
        np.testing.assert_array_equal(
            np.asarray(getattr(g_prod, name)), np.asarray(getattr(g_twin, name)), err_msg=f"{name} ({label})"
        )


def _run_twin_pair(driver, ham, monkeypatch, start=None, advance=None):
    """Full-recompute twin, then the production route (optionally with a replaced carried-state advance)."""
    import jqmc.jqmc_gfmc as gfmc_mod

    g_twin = _run(driver, ham, full_recompute=True, monkeypatch=monkeypatch, start=start)
    witness_before = gfmc_mod._GFMC_POT_TRACE_WITNESS
    with pytest.MonkeyPatch.context() as patch:
        if advance is not None:
            patch.setattr(gfmc_mod, "_gfmc_advance_interval_pot_state", advance)
        g_prod = _run(driver, ham, full_recompute=False, monkeypatch=monkeypatch, start=start)
    assert gfmc_mod._GFMC_POT_TRACE_WITNESS > witness_before, "incremental branch was never traced"
    assert np.all(np.asarray(g_prod._latest_gfmc_pot_updates) > 0), g_prod._latest_gfmc_pot_updates
    assert np.all(np.asarray(getattr(g_twin, "_latest_gfmc_pot_updates", np.zeros(2))) == 0)
    return g_prod, g_twin


@pytest.mark.light
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_xla_incremental_twin_ae_pbc_ei_disc_live(driver, monkeypatch):
    """Driver twin on H4 AE PBC from a start where the carried ``ei_disc`` rows are consumed.

    GFMC_t and GFMC_n production runs (incremental interval potential) and
    ``JQMC_GFMC_POT_FULL_RECOMPUTE=1`` runs with the seed, walkers and steps of
    ``_run`` start from ``_floor_start``. Stored w_L, e_L and e_L2 (real and
    imaginary parts) agree at ATOL/RTOL_GFMC_INCREMENTAL_TWIN, walkers and PRNG
    keys are byte-identical, the incremental branch is traced and reports
    positive update counts while the twin reports zeros. That the moved
    electrons' carried ``ei_disc`` rows reach the stored observables in this
    configuration is the verdict of
    ``test_gfmc_xla_ei_disc_twin_detects_carried_row_faults``.
    """
    ham = _build_h4_pbc_ae_ham()
    assert bool(ham.structure_data.pbc_flag) and not bool(ham.coulomb_potential_data.ecp_flag)
    g_prod, g_twin = _run_twin_pair(driver, ham, monkeypatch, start=_floor_start(ham))
    _assert_driver_twin(g_prod, g_twin, "H4 AE PBC, floor start")


def _faulty_advance(original, fault):
    """Carried-state advance with a fault in the moved electron's ``ei_disc`` row.

    ``row_times_1p5`` writes 1.5 x the new row; ``row_not_written`` drops the
    ``.at[gi].set`` (the row stays stale).
    """

    def advance(*args, **kwargs):
        old = args[1]
        new = original(*args, **kwargs)
        if fault == "row_times_1p5":
            ei_disc = jnp.where(new[4] != old[4], 1.5 * new[4], new[4])
        else:
            ei_disc = old[4]
        return new[:4] + (ei_disc,) + new[5:]

    return advance


@pytest.mark.light
@pytest.mark.parametrize("fault", ["row_times_1p5", "row_not_written"])
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_xla_ei_disc_twin_detects_carried_row_faults(driver, fault, monkeypatch):
    """Negative control and liveness proof for the floor-start driver twin.

    The production route runs with ``_gfmc_advance_interval_pot_state``
    replaced by a copy whose moved ``ei_disc`` row is either scaled by 1.5 or
    left unwritten; the full-recompute twin is unchanged. The stored-observable
    comparison of ``test_gfmc_xla_incremental_twin_ae_pbc_ei_disc_live`` must
    fail, which proves that in this configuration the carried ``ei_disc`` rows
    of moved electrons are consumed by both drivers.
    """
    import jqmc.jqmc_gfmc as gfmc_mod

    advance = _faulty_advance(gfmc_mod._gfmc_advance_interval_pot_state, fault)
    ham = _build_h4_pbc_ae_ham()
    g_prod, g_twin = _run_twin_pair(driver, ham, monkeypatch, start=_floor_start(ham), advance=advance)
    with pytest.raises(AssertionError, match=r"Re\(_stored_(w_L|e_L)\): incremental vs full-recompute twin"):
        _assert_driver_twin(g_prod, g_twin, "H4 AE PBC, floor start")


# --------------------------------------------------------------------------- #
# Complex twist, DTN and a skewed cell.
# --------------------------------------------------------------------------- #
def _periodic_twin_ham(fixture):
    from tests._gfmc_twin_fixtures import skewed_h4_ham
    from tests.test_rs_kernel._rs_fixtures import rs_ham

    patterns = {
        "lithium_j12": "pbc_gamma_sphepgto_j12",
        "lithium_jdtn": "pbc_gamma_sphepgto_jdtn",
        "lithium_k025_j12": "twist_sphepgto_j12",
        "lithium_k025_jdtn": "twist_sphepgto_jdtn",
    }
    if fixture == "h4_skew_k025":
        return skewed_h4_ham(twist=True)
    return rs_ham(patterns[fixture])


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param("lithium_j12", marks=pytest.mark.heavy),
        pytest.param("lithium_jdtn", marks=pytest.mark.heavy),
        pytest.param("lithium_k025_j12", marks=pytest.mark.heavy),
        pytest.param("lithium_k025_jdtn", marks=pytest.mark.heavy),
        pytest.param("h4_skew_k025", marks=pytest.mark.light),
    ],
)
@pytest.mark.parametrize("driver", ["t", "n"])
def test_gfmc_xla_incremental_twin_periodic_complex(fixture, driver, monkeypatch):
    """XLA driver twin over {lithium.h5, lithium_k025.h5} x {J1J2, JDTN} and a skewed-cell complex twist.

    Production (incremental) and ``JQMC_GFMC_POT_FULL_RECOMPUTE=1`` runs with
    the seed, walkers and steps of ``_run``. Stored w_L, e_L and e_L2 are
    compared separately in real and imaginary parts at
    ATOL/RTOL_GFMC_INCREMENTAL_TWIN (no real projection), walkers and PRNG
    keys byte-identical, trace witness and update counters as in the other
    twins. ``h4_skew_k025`` is ``H4_pbc_ae_k025_cart.h5`` on the strongly
    skewed triclinic basis (``tests/_gfmc_twin_fixtures.py``): the carried ee
    state is advanced through the general nearest-image search and the
    ``ei_disc`` rows are carried; its walkers start unwrapped, each electron
    translated by its own lattice vector (``far_start``), so electron-electron
    displacements span several cells. At k != Gamma the stored e_L must have a
    non-zero imaginary part, so the imaginary comparison is not vacuous.
    """
    from tests._gfmc_twin_fixtures import far_start

    ham = _periodic_twin_ham(fixture)
    assert bool(ham.structure_data.pbc_flag)
    start = far_start(ham) if fixture == "h4_skew_k025" else None
    g_prod, g_twin = _run_twin_pair(driver, ham, monkeypatch, start=start)
    _assert_driver_twin(g_prod, g_twin, fixture)
    if "k025" in fixture:
        assert np.max(np.abs(np.imag(np.asarray(g_twin._stored_e_L)))) > ATOL_GFMC_INCREMENTAL_TWIN, "twist e_L is real"
