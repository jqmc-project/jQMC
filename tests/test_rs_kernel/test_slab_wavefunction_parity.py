"""Rust==XLA parity for ASSEMBLED observables on a SLAB-WITH-VACUUM cell carrying a
real trial wavefunction: aggregate e_L, VMC HF+Pulay forces, HF+Pulay stress, and
the production MCMC driver route.

Why this file exists
--------------------
The Ewald slab bug (since fixed) proved that regimes never visited by the
parity fixtures can hide backend divergences. The point-charge slab parity
(``test_ewald_backend_parity.py`` energies, ``test_ewald_force_stress_backend_
parity.py`` derivatives) closed the Ewald channels -- but every parity test that
carries a WAVEFUNCTION (e_L, VMC forces/stress, the drivers) still runs
exclusively on compact bulk cells (cubic lithium / H2 / H4 boxes). On a slab the
Ewald tables are anisotropic (~400 real images, few
G-vectors), electrons roam the vacuum region far from every nucleus, and the
periodic min-image/AO-image machinery works at a high-aspect cell -- none of
which any wavefunction-bearing parity fixture exercised.

No committed trexio stem has a slab cell, so this file BUILDS one synthetically
(the ``test_wave_function.py::_periodic_geminal_setup`` idiom): an H4 layer in a
9 x 9 x 30 Bohr vacuum cell, sphe-pGTO s-shells (image-expanded), an MO-wrapped
AGP geminal, and the production J1+J2 exp Jastrow keyed to the periodic cell.
All-electron, charge-neutral (4 H, 2 up + 2 dn) so the Ewald background terms
drop out identically on both backends. Everything is in-test numpy -- no new
binary fixture (committed-fixtures rule trivially satisfied).

Scope note: physics-level correctness is validated on
the XLA path only; this file enforces exactly Rust == XLA on the slab regime,
mirroring the call shapes of the bulk parity files
(``test_grads_laplacian_far.py`` e_L, ``test_vmc_forces.py`` forces,
``test_vmc_stress.py`` stress, ``test_rust_dispatch_equivalence.py`` driver).
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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled  # noqa: E402
from jqmc._setting import (  # noqa: E402
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_FLOOR_MEDIUM,
    RTOL_FLOOR_MEDIUM,
)
from jqmc.atomic_orbital import AOs_sphe_gto_data, _apply_image_expansion  # noqa: E402
from jqmc.coulomb_potential import (  # noqa: E402
    Coulomb_potential_data,
    build_ewald_data,
    icosahedron_sym_mesh_Nv12,
)
from jqmc.determinant import Geminal_data, compute_geminal_all_elements  # noqa: E402
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy, compute_local_energy_fast  # noqa: E402
from jqmc.jastrow_factor import (  # noqa: E402
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_two_body_data,
)
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from jqmc.structure import Structure_data  # noqa: E402
from jqmc.wavefunction import Wavefunction_data, evaluate_wavefunction  # noqa: E402

# Shared oracles / helpers of the bulk parity files (single source of truth --
# no call shape or tolerance rationale re-encoded here).
from tests.test_rs_kernel.test_vmc_forces import (  # noqa: E402
    _GP,
    _W,
    _jax_ref,
    _jax_ref_pulay,
    _pbc_geminv,
)
from tests.test_rs_kernel.test_vmc_stress import _jax_stress  # noqa: E402
from tests._ecp_core_electrons import core_electrons_of

pytestmark = pytest.mark.with_chdir

# --------------------------------------------------------------------------- #
# Synthetic slab Hamiltonian (in-test, no committed binary).
# --------------------------------------------------------------------------- #

#: Slab cell: compact in-plane axes, one long vacuum axis (aspect > 3), the
#: regime of the slab Ewald fix. Layer thickness ~1.5 Bohr around z=0.
_SLAB_A = 9.0
_SLAB_C = 30.0


def _slab_structure() -> Structure_data:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.5, 0.0, 1.5],
            [0.0, 4.5, 1.5],
            [4.5, 4.5, 0.0],
        ],
        dtype=np.float64,
    )
    return Structure_data.build(
        pbc_flag=True,
        positions=positions,
        vec_a=(_SLAB_A, 0.0, 0.0),
        vec_b=(0.0, _SLAB_A, 0.0),
        vec_c=(0.0, 0.0, _SLAB_C),
        atomic_numbers=(1, 1, 1, 1),
        element_symbols=("H", "H", "H", "H"),
        atomic_labels=("H1", "H2", "H3", "H4"),
    )


def _slab_pgto_leaf(structure: Structure_data):
    """Image-expanded sphe-pGTO leaf: two s shells per H (8 AOs)."""
    n_atom = len(structure.positions)
    base = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=tuple(i for i in range(n_atom) for _ in range(2)),
        num_ao=2 * n_atom,
        num_ao_prim=2 * n_atom,
        orbital_indices=tuple(range(2 * n_atom)),
        exponents=np.tile(np.array([0.9, 0.4], dtype=np.float64), n_atom),
        coefficients=np.ones(2 * n_atom, dtype=np.float64),
        angular_momentums=(0,) * (2 * n_atom),
        magnetic_quantum_numbers=(0,) * (2 * n_atom),
    )
    return _apply_image_expansion(base, structure, AO_IMAGE_TRUNCATION_ATOL)


def _slab_ham() -> Hamiltonian_data:
    """H4-layer slab Hamiltonian: MO-wrapped AGP over sphe pGTOs + J1/J2 exp."""
    structure = _slab_structure()
    aos_data = _slab_pgto_leaf(structure)
    aos_data.sanity_check()

    num_mo = 2
    rng = np.random.default_rng(2026)
    mo_coefficients = rng.standard_normal((num_mo, aos_data.num_ao)).astype(np.float64)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    lambda_paired = rng.uniform(-0.2, 0.2, size=(num_mo, num_mo))
    lambda_paired = 0.5 * (lambda_paired + lambda_paired.T) + np.eye(num_mo)
    geminal_data = Geminal_data(
        num_electron_up=2,
        num_electron_dn=2,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=np.hstack([lambda_paired, np.zeros((num_mo, 0), dtype=np.float64)]),
    )
    geminal_data.sanity_check()

    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, tol=1e-10))

    n_atom = len(structure.positions)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure,
        core_electrons=core_electrons_of(coulomb, n_atom),
        jastrow_1b_type="exp",
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure)
    jastrow_data = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None)
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data)

    ham = Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wf)
    ham.sanity_check()
    return ham


#: Far-scatter scale: same 4.0 the bulk force/stress far rows use -- electrons
#: land in the vacuum region and in neighbouring periodic copies (a rigid
#: integer-lattice shift would be wrapped away identically by every kernel; a
#: scatter is what exposes J3/DTN raw-coordinate force errors).
_FAR_SCALE = 4.0


def _slab_cfg(ham, seed, scale=0.6):
    """Node-far electron draw near the layer atoms (or scattered when far)."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
        rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
        psi = evaluate_wavefunction(ham.wavefunction_data, jnp.asarray(ru), jnp.asarray(rd))
        if abs(float(psi)) > 1.0e-10:  # tol-ok: node-far acceptance floor, not a tolerance
            return ru, rd
    pytest.fail("could not find a node-far slab configuration; wavefunction evaluation regressed")


# --------------------------------------------------------------------------- #
# 1. Aggregate e_L (the production observable) -- near and vacuum-scattered.
# --------------------------------------------------------------------------- #

_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_ATOL_SLAB_DERIV, _RTOL_SLAB_DERIV = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium")
_ATOL_SLAB_DERIV = max(_ATOL_SLAB_DERIV, ATOL_FLOOR_MEDIUM)
_RTOL_SLAB_DERIV = max(_RTOL_SLAB_DERIV, RTOL_FLOOR_MEDIUM)

# ROOT-CAUSED floor for the VACUUM-SCATTERED rows (level "medium" instead of
# "strict"; near rows stay strict). The LEPSBAS screen
# is a deliberate Rust advantage (dynamic pruning the dense XLA path cannot do)
# and STAYS; this path is explicitly exempt from bit-level parity, and the
# medium gate below is the permanent documented contract, not a placeholder.
# The residual is NOT a truncation/regime bug:
# the reused-inverse kinetic inside the native e_L
# (``compute_grads_and_laplacian_ln_det_with_inv`` via_single ->
# ``compute_aos_single_value_grad_lap``) applies the sweep-convention LEPSBAS
# per-image screen (jqmc/rs/src/atomic_orbital.rs::compute_aos_single_value_grad_lap,
# ``min_exp*r^2 > LEPSBAS`` -> image skipped, tail ~exp(-20.72)), while XLA
# evaluates the full build-time image set. On BULK cells every AO is O(0.1-1) so
# the dropped tail stays ~1e-9 relative (existing strict e_L parity holds); at a
# VACUUM-scattered slab config the surviving AO amplitude is itself ~1e-5, so the
# same absolute tail becomes ~1e-4 relative per AO and ~1e-6 on e_L. Experiment:
# rebuilding jqmc_rs with that screen disabled collapses this
# residual to ~1e-14 on every seed; the screen saves ~29% of bulk-Li e_L time
# (5.2 -> 7.3 ms/call unscreened). A wiring/regime bug is O(0.1-1), 5+ orders
# above the medium gate, so the gate keeps its teeth.


@pytest.mark.light
@pytest.mark.parametrize("seed,scale", [(1, 0.6), (3, _FAR_SCALE), (6, _FAR_SCALE)], ids=["near", "far3", "far6"])
def test_slab_local_energy_fast_parity(seed, scale):
    """Native ``compute_local_energy_fast`` == jqmc's on the slab cell, reusing a
    sweep-consistent SM inverse, with electrons near the layer AND scattered into
    the vacuum / neighbouring copies (far rows: documented LEPSBAS floor above)."""
    ham = _slab_ham()
    gem = ham.wavefunction_data.geminal_data
    ru, rd = _slab_cfg(ham, seed, scale=scale)
    RT = np.eye(3)
    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gem, ru, rd)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, ru, rd, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, ru, rd, jnp.asarray(RT), jnp.asarray(g_inv), None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (seed, scale, native, jax_eL, abs(native - jax_eL))


# --------------------------------------------------------------------------- #
# 2. VMC HF + Pulay forces (same aggregates the production MCMC driver stores).
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("seed,scale", [(0, 0.6), (3, _FAR_SCALE), (6, _FAR_SCALE)], ids=["near", "far3", "far6"])
def test_slab_vmc_force_parity(seed, scale):
    """``vmc_grad_e_L_pbc`` / ``vmc_grad_ln_psi_pbc`` == the JAX assemble oracles
    on the slab cell (HF dR/dr and Pulay dR/dr), near and vacuum-scattered."""
    ham = _slab_ham()
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _slab_cfg(ham, seed, scale=scale)
    ginv = _pbc_geminv(ham, ru, rd, False)
    dR_j, dru_j, drd_j = _jax_ref(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc(gd, jd, cp, ru, rd, _W, _GP, ginv)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab HF dr_dn")
    pR_j, pru_j, prd_j = _jax_ref_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc(gd, jd, ru, rd, ginv)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab Pulay dr_dn")


# --------------------------------------------------------------------------- #
# 3. HF + Pulay stress (the periodic-only observable; the slab's anisotropic
#    Ewald tables feed the ee/ei/ii strain terms directly).
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("seed,scale", [(0, 0.6), (3, _FAR_SCALE)], ids=["near", "far3"])
def test_slab_vmc_stress_parity(seed, scale):
    """``vmc_grad_e_L_eps_pbc`` / ``vmc_grad_ln_psi_eps_pbc`` == the JAX strain
    oracles on the slab cell, near and vacuum-scattered."""
    ham = _slab_ham()
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _slab_cfg(ham, seed, scale=scale)
    ginv = _pbc_geminv(ham, ru, rd, False)
    RT = np.eye(3)
    hf_j, pp_j = _jax_stress(ham, ru, rd, RT, ginv)
    hf_n = np.asarray(jqmc_rs.vmc_grad_e_L_eps_pbc(gd, jd, cp, ru, rd, _W, _GP @ RT, ginv))
    pp_n = np.asarray(jqmc_rs.vmc_grad_ln_psi_eps_pbc(gd, jd, ru, rd, ginv))
    np.testing.assert_allclose(hf_n, hf_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab HF stress")
    np.testing.assert_allclose(pp_n, pp_j, atol=_ATOL_SLAB_DERIV, rtol=_RTOL_SLAB_DERIV, err_msg="slab Pulay stress")


# --------------------------------------------------------------------------- #
# 4. GFMC/LRDMC projection sub-step on the slab ham: every intermediate of one
#    discretized projection step (kinetic elements, sign flip, S', e_L, b_L),
#    XLA vs Rust. PBC has NO bitexact GFMC chain contract (periodic native-e_L
#    fp floor enters the hop cdf -- test_gfmc_chain_modes.py), so the GFMC slab
#    observable parity is pinned at the step-component level, the same harness
#    the bulk cells use (test_gfmc_projection.py).
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("seed,scale", [(1, 0.6), (3, _FAR_SCALE)], ids=["near", "far3"])
def test_slab_gfmc_step_components_parity(seed, scale):
    """Slab GFMC step components parity."""
    from tests.test_rs_kernel.test_gfmc_projection import _rs_step_eL, _xla_step

    ham = _slab_ham()
    wf = ham.wavefunction_data
    ru, rd = _slab_cfg(ham, seed, scale=scale)
    alat = 0.40
    ref = _xla_step(ham, ru, rd, alat, np.eye(3), "tmove")
    els = np.asarray(jqmc_rs._gfmc_discretized_kinetic_elements(wf.geminal_data, wf.jastrow_data, ru, rd, alat, np.eye(3)))
    # The discretized kinetic elements and
    # e_L are cancelling aggregates that ride the same reused-inverse AO kernels
    # as the e_L above (LEPSBAS floor note at the top of this section). Mixed
    # precision uses the reviewed medium backend envelope for these two rows;
    # full precision keeps the strict pair. b_L keeps strict in both modes.
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict")
    atol_b, rtol_b = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert els.shape == ref["elements"].shape
    assert np.allclose(els, ref["elements"], atol=atol, rtol=rtol), np.abs(els - ref["elements"]).max()
    rs_flip = np.any(els.reshape(-1, 6) >= 0.0, axis=1)
    assert np.array_equal(rs_flip, ref["sign_flip"]), (rs_flip, ref["sign_flip"])
    e_L, b_L = _rs_step_eL(ham, ru, rd, alat, np.eye(3), "tmove", fast=False)
    assert np.isclose(e_L, ref["e_L"], rtol=rtol, atol=atol), (e_L, ref["e_L"], abs(e_L - ref["e_L"]))
    assert np.isclose(b_L, ref["b_L"], rtol=rtol_b, atol=atol_b), (b_L, ref["b_L"], abs(b_L - ref["b_L"]))


# --------------------------------------------------------------------------- #
# 5. Production MCMC driver route on the slab ham: flag-ON (native) vs flag-OFF
#    (JAX) runs have independent RNG streams. Re-evaluate each final stored
#    energy at that route's own final coordinates; do not compare independent
#    sampled trajectories or use a statistical acceptance band.
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
def test_slab_mcmc_driver_route_parity(monkeypatch):
    """Both slab driver routes store the correct final-configuration energies.

    Native route counters must fire. Each stored energy is compared with a
    fresh XLA evaluation of the same coordinates in the same precision mode;
    the independently sampled Rust and XLA trajectories are not compared.
    Force/stress buffers additionally retain their finite storage contract.
    """
    from tests.test_rs_kernel.test_rust_dispatch_equivalence import (
        _assert_run_equivalent,
        _run,
        _run_flag_on_with_native_spy,
    )

    ham = _slab_ham()
    m_rust, counts = _run_flag_on_with_native_spy(ham, monkeypatch)
    m_jax = _run(ham, rust_on=False)
    # e_L block at "medium": the bitexact chain walks identical walkers, but the
    # native e_L kinetic carries the ROOT-CAUSED LEPSBAS single-kernel screen
    # floor documented above test_slab_local_energy_fast_parity -- measured here
    # as 2/80 samples at ~1.2e-6 abs (walkers roamed into the vacuum), identical
    # magnitude to the kernel-level far rows. Force/stress blocks use the
    # documented derivative-of-medium-e_L-floor constants (jqmc/_setting.py).
    _assert_run_equivalent(m_rust, m_jax)
    # Real force+stress runs use the fused native chunk, not Python-level
    # per-step sweep/observable methods. Require the actual production route.
    assert counts["energy_chunk"] > 0, f"native slab chunk did not execute: {counts}"
    assert counts["real"] == 0 and counts["e_L"] == 0, f"slab unexpectedly used the per-step route: {counts}"
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium")
    assert not ham.coulomb_potential_data.ecp_flag, "identity mesh is valid here only for the all-electron slab"
    for driver in (m_rust, m_jax):
        r_up = np.asarray(driver._MCMC__latest_r_up_carts)
        r_dn = np.asarray(driver._MCMC__latest_r_dn_carts)
        reference = np.asarray([compute_local_energy(ham, up, down, jnp.eye(3)) for up, down in zip(r_up, r_dn, strict=True)])
        np.testing.assert_allclose(driver.e_L[-1], reference, atol=atol, rtol=rtol)
