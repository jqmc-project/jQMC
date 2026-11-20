"""Production parity for Cartesian-GTO shells at f/g angular momentum
(l=3, l=4), Rust vs jqmc.

Coverage note: every ``cart_gto`` fixture actually wired into
``tests/test_rs_kernel`` tops out at l=2 (d shells) -- ``N_ae_ccpvdz_cart.h5``
(l_max=2, the mol_cart_sd_* rows in ``_rs_fixtures``) and
``H2_ecp_ccpvdz_cart.h5`` (l_max=1). The only l>=3 (f/g) fixture used anywhere
in the rs-kernel suite is ``water_ccecp_ccpvqz.h5`` (l_max=4), but that basis
is SPHERICAL (``mol_ecp_sphe_*``), not cartesian -- cartesian shells carry a
DIFFERENT component count per l (``(l+1)(l+2)/2`` cartesian components vs
``2l+1`` spherical) and a different real-solid-harmonic-free ordering
convention, so a spherical f/g pin does not cover the cartesian f/g code path.
The committed ``Cl2_ecp_ccpvtz_cart.h5`` (l_max=3) and
``Ti2_ecp_ccpvtz_cart.h5`` (l_max=4) fixtures DO carry cartesian f/g shells
(confirmed via ``trexio.read_basis_shell_ang_mom``) but are referenced only
under ``tests/test_jqmc_kernel`` (XLA-side: ``test_jqmc_tool.py``,
``test_gfmc_force_analytic.py``), never under ``tests/test_rs_kernel``.

The ECP non-local channel loop itself is data-driven, not l-hardcoded
(``jqmc/rs/src/coulomb_potential.rs``: ``ang_mom: &[i64]`` indexed per channel
inside the loop), so the plausible NEW code surface here is specifically the
cartesian AO/MO evaluation (component ordering/normalization) at l=3/4, not
the ECP loop structure. This module pins that surface directly: ln-det
grad/Laplacian, the assembled kinetic energy, and the aggregate
``compute_local_energy_fast`` on both fixtures (closed-shell ECP, jnone and
J1+J2, mirroring the ``mol_ecp_cart_{jnone,j12}`` naming convention).
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

import jqmc_rs

pytestmark = pytest.mark.with_chdir

import jax.numpy as jnp

from jqmc._precision import get_tolerance_min, get_tolerance_xla_rust_consistency
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.determinant import compute_geminal_all_elements, compute_grads_and_laplacian_ln_Det
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.wavefunction import compute_kinetic_energy

from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham

_ZONES_LNDET = ("ao_eval", "ao_grad_lap", "det_grad_lap")
_ZONES_KINETIC = ("wf_kinetic", "ao_eval", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")
_ZONES_EL = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap")
_MESH = icosahedron_sym_mesh_Nv12

# stem -> l_max (cartesian, confirmed via trexio.read_basis_shell_ang_mom); both
# ECP + closed-shell (Cl2: 7up/7dn, Ti2: 12up/12dn).
_CASES = [
    ("Cl2_ecp_ccpvtz_cart.h5", False, "jnone_lmax3"),
    ("Cl2_ecp_ccpvtz_cart.h5", True, "j12_lmax3"),
    ("Ti2_ecp_ccpvtz_cart.h5", False, "jnone_lmax4"),
    ("Ti2_ecp_ccpvtz_cart.h5", True, "j12_lmax4"),
]

# Backend parity must measure implementation agreement, not an accidental
# near-node amplification by a nearly singular Slater/geminal matrix.  The old
# ``scale=0.3, seed={7,8,9}`` points had condition numbers up to 1.3e5.  These
# deterministic points still exercise the same Cl Cartesian-f and Ti
# Cartesian-g production routes, while keeping cond(G) moderate (measured
# mixed-mode values: Cl2 5.9e1, Ti2 3.1e2).
_CONFIG_BY_STEM = {
    "Cl2_ecp_ccpvtz_cart.h5": (0.8, 22),
    "Ti2_ecp_ccpvtz_cart.h5": (1.0, 47),
}
_MAX_FIXTURE_CONDITION = 1.0e3


def _config(ham, stem):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    scale, seed = _CONFIG_BY_STEM[stem]
    rng = np.random.default_rng(seed)
    r_up = R[np.arange(nu) % len(R)] + rng.normal(scale=scale, size=(nu, 3))
    r_dn = R[np.arange(nd) % len(R)] + rng.normal(scale=scale, size=(nd, 3))
    return r_up, r_dn


@pytest.mark.light
@pytest.mark.parametrize("stem", tuple(_CONFIG_BY_STEM), ids=("lmax3", "lmax4"))
def test_high_angmom_parity_fixture_is_well_conditioned(stem):
    """Keep the strict backend-parity points away from determinant nodes."""
    ham = _build_ham(stem, with_jastrow=False)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, stem)
    geminal = np.asarray(compute_geminal_all_elements(gd, r_up, r_dn))
    condition = np.linalg.cond(geminal)
    assert condition < _MAX_FIXTURE_CONDITION


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, tag", _CASES, ids=[c[2] for c in _CASES])
def test_grads_laplacian_ln_det_high_angmom(stem, with_jastrow, tag):
    """Grads laplacian ln det high angmom."""
    ham = _build_ham(stem, with_jastrow)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, stem)
    gu, gdn, lu, ldn = jqmc_rs.compute_grads_and_laplacian_ln_det(gd, r_up, r_dn)
    Gu, Gdn, Lu, Ldn = (np.asarray(x, float) for x in compute_grads_and_laplacian_ln_Det(gd, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")
    assert np.allclose(gu, Gu, atol=atol, rtol=rtol), np.max(np.abs(gu - Gu))
    assert np.allclose(gdn, Gdn, atol=atol, rtol=rtol), np.max(np.abs(gdn - Gdn))
    assert np.allclose(lu, Lu, atol=atol, rtol=rtol), np.max(np.abs(lu - Lu))
    assert np.allclose(ldn, Ldn, atol=atol, rtol=rtol), np.max(np.abs(ldn - Ldn))


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, tag", _CASES, ids=[c[2] for c in _CASES])
def test_kinetic_energy_high_angmom(stem, with_jastrow, tag):
    """Kinetic energy high angmom."""
    ham = _build_ham(stem, with_jastrow)
    wf = ham.wavefunction_data
    r_up, r_dn = _config(ham, stem)
    k_rust = jqmc_rs.compute_kinetic_energy(wf.geminal_data, wf.jastrow_data, r_up, r_dn)
    k_jqmc = float(compute_kinetic_energy(wf, r_up, r_dn))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_KINETIC, level="strict")
    assert np.isclose(k_rust, k_jqmc, rtol=rtol, atol=atol), (k_rust, k_jqmc, abs(k_rust - k_jqmc))


@pytest.mark.light
@pytest.mark.parametrize("stem, with_jastrow, tag", _CASES, ids=[c[2] for c in _CASES])
def test_local_energy_fast_high_angmom(stem, with_jastrow, tag):
    """Local energy fast high angmom."""
    ham = _build_ham(stem, with_jastrow)
    gd = ham.wavefunction_data.geminal_data
    r_up, r_dn = _config(ham, stem)
    RT = np.eye(3)
    w = np.asarray(_MESH.weights, float)
    gp = np.asarray(_MESH.grid_points, float) @ RT
    g_inv = np.linalg.inv(np.asarray(compute_geminal_all_elements(gd, r_up, r_dn)))
    native = float(jqmc_rs.compute_local_energy_fast(ham, r_up, r_dn, w, gp, g_inv))
    jax_eL = float(compute_local_energy_fast(ham, r_up, r_dn, jnp.asarray(RT), jnp.asarray(g_inv), None))
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="strict")
    assert np.isclose(native, jax_eL, rtol=rtol, atol=atol), (native, jax_eL, abs(native - jax_eL))


# --------------------------------------------------------------------------- #
# Cartesian l=7 through the production MCMC chunk, against a closed form.
# --------------------------------------------------------------------------- #
# The committed cartesian fixtures stop at l=4, and the parity rows above use
# the exact (no-cutoff) kernels. The production MCMC chunk evaluates the AO
# grad/Laplacian through the LEPSBAS-cutoff kernels, whose Cartesian power
# tables are sized for x^0..x^7; nothing exercised their top entry. One electron
# bound to a bare proton in psi = x^7 exp(-a r^2) (a single l=7 Cartesian
# primitive, MO-projected onto its x^7 component) has the closed-form local
# energy
#     e_L = -(1/2) (42 / x^2 + 4 a^2 r^2 - 34 a) - 1 / r,
# from Laplacian(x^7 e^{-a r^2}) / (x^7 e^{-a r^2}) = 42/x^2 - 34a + 4a^2 r^2.
# It needs no normalisation and no second implementation.
_L7_EXPONENT = 0.7
_L7_ZONES = ("wf_kinetic", "ao_grad_lap", "det_grad_lap", "coulomb")


def _hydrogenic_cart_l7_ham():
    from jqmc.atomic_orbital import AOs_cart_gto_data
    from jqmc.coulomb_potential import Coulomb_potential_data
    from jqmc.determinant import SD_data
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.structure import Structure_data
    from jqmc.wavefunction import Wavefunction_data

    structure = Structure_data(
        pbc_flag=False,
        positions=np.zeros((1, 3)),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H",),
    )
    comps = [(nx, ny, 7 - nx - ny) for nx in range(7, -1, -1) for ny in range(7 - nx, -1, -1)]
    n = len(comps)
    aos = AOs_cart_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=(0,) * n,
        num_ao=n,
        num_ao_prim=n,
        orbital_indices=tuple(range(n)),
        exponents=np.full(n, _L7_EXPONENT),
        coefficients=np.ones(n),
        angular_momentums=(7,) * n,
        polynominal_order_x=tuple(c[0] for c in comps),
        polynominal_order_y=tuple(c[1] for c in comps),
        polynominal_order_z=tuple(c[2] for c in comps),
    )
    coefficients = np.zeros((1, n))
    coefficients[0, comps.index((7, 0, 0))] = 1.0
    mos = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=coefficients)
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=0, mos_up=mos, mos_dn=mos)
    jastrow = Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure, ecp_flag=False),
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal),
    )
    ham.sanity_check()
    return ham


@pytest.mark.light
@pytest.mark.parametrize("rust_on", [True, False], ids=("rust", "xla"))
def test_mcmc_local_energy_cart_l7_closed_form(rust_on):
    """Production MCMC e_L for a Cartesian x^7 hydrogenic orbital equals its closed form."""
    import os

    from jqmc.jqmc_mcmc import MCMC

    from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy

    ham = _hydrogenic_cart_l7_ham()
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        mcmc = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=1,
            epsilon_AS=0.0,
            num_walkers=4,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
        )
        with native_ctx_spy() as counts:
            mcmc.run(num_mcmc_steps=5)
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous
    if rust_on:
        assert_native_ran(counts, "cart l=7 MCMC")
    else:
        assert_no_native_ran(counts, "cart l=7 MCMC")

    r = np.asarray(mcmc._MCMC__latest_r_up_carts, dtype=np.float64).reshape(4, 3)
    a = _L7_EXPONENT
    r2 = np.sum(r * r, axis=1)
    expected = -0.5 * (42.0 / r[:, 0] ** 2 + 4.0 * a * a * r2 - 34.0 * a) - 1.0 / np.sqrt(r2)
    got = np.asarray(mcmc.e_L, dtype=np.float64)[-1].reshape(4)
    atol, rtol = get_tolerance_min(_L7_ZONES)
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)
