"""Native complex local energy, energy chunk and drivers on a geminal below sqrt(tiny) (issue #164).

``num_complex``'s ``a / b`` forms ``|b|^2`` unscaled, so the native complex
Gauss-Jordan inverse (``complex_det::inverse_complex``), the in-LU
eliminations and the reference complex ECP / GFMC mesh ratios
``det(G_mesh) / det(G_base)`` returned ``NaN`` / ``inf`` once a pivot or a
determinant fell below ``sqrt(f64::MIN_POSITIVE) ~ 1.5e-154`` although the
inverse and the ratios are finite. Every native complex local energy seeded
from that inverse (kinetic, fast ECP, the energy chunk of the default twist
MCMC, GFMC_n) was ``NaN`` and the chain froze.

Witnesses: the committed twist cells ``lithium_k025`` (ccECP, one pair,
``rs_ham("twist_sphepgto_jnone")``) and ``H4_pbc_ae_k025_cart`` (all-electron,
two pairs, so the elimination and the ln-det LU steps run), no Jastrow, with
both spin MO coefficient arrays multiplied by ``c = 1e-85`` (``G -> c^2 G``,
``|G| ~ 1e-173``), at a PCG64(13) uniform start in the cell.

Expected property (independent references): e_L, every determinant ratio and
``G^-1 dG`` are invariant under a common MO scale, and
``ln|det(c^2 G)| = ln|det G| + 2 n ln c``. Full precision compares the scaled
native values with the unscaled native values at the same configuration (and
chain keys), with the XLA ``compute_local_energy`` at the scaled
configuration and with numpy (``inv`` / ``slogdet``) for the inverse and the
ln-det. Mixed precision follows the mixed-vs-full rule
(``tests/_precision_fidelity.py``): the scaled case is rebuilt under each
policy, every mixed output must be finite, and the mixed chunk's acceptance
decisions must equal the full ones. The free LRDMC / ECP force / strain /
GFMC HF routes (non-AD outputs) additionally compare, within mixed precision,
the scaled outputs with the unscaled mixed outputs at the same configuration
(same-precision scale invariance under the ``det_ratio`` zone tolerance).

Not covered here (separate issues): the AS factor underflow of #131 / #134
(``w_L`` of the chunk and the frozen chain at ``epsilon_AS > 0``) -- the chunk
and the MCMC equality checks therefore run at ``epsilon_AS = 0`` and do not
compare ``w_L``; the default ``epsilon_AS`` MCMC run is checked for finite
``e_L`` only.
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

import math
import os

import numpy as np
import pytest
import jqmc_rs

from jqmc._precision import (
    get_tolerance,
    get_tolerance_min,
    get_tolerance_xla_rust_consistency,
    is_mixed_precision_enabled,
    precision_policy_snapshot,
)
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_gfmc import GFMC_n
from jqmc.jqmc_mcmc import MCMC, _NV_BASE_MESH
from jqmc._setting import Nv_default
from tests._precision_fidelity import mixed_vs_full
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.with_chdir]

_SCALE = 1.0e-85
_SQRT_TINY = math.sqrt(np.finfo(np.float64).tiny)
# Zones of the native complex local energy and of its XLA twin; local so the
# static tolerance resolver sees the exact budget.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)
_REFERENCE_ENV = "JQMC_RS_ECP_COMPLEX_REFERENCE"


def _base_ham(witness):
    if witness == "li_k025_ccecp":
        return rs_ham("twist_sphepgto_jnone")
    return _build_ham_for_cell("H4_pbc_ae_k025_cart.h5", False, None, "SD")


def _scaled(ham, scale):
    """``ham`` with both spin MO coefficient arrays multiplied by ``scale``."""
    g = ham.wavefunction_data.geminal_data
    sc = g.replace(
        orb_data_up_spin=g.orb_data_up_spin.replace(mo_coefficients=g.orb_data_up_spin.mo_coefficients * scale),
        orb_data_dn_spin=g.orb_data_dn_spin.replace(mo_coefficients=g.orb_data_dn_spin.mo_coefficients * scale),
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=sc))


def _start(ham):
    g = ham.wavefunction_data.geminal_data
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    rng = np.random.default_rng(13)
    up = rng.random((g.num_electron_up, 3)) @ cell
    dn = rng.random((g.num_electron_dn, 3)) @ cell
    return up, dn


def _mesh():
    mesh = _NV_BASE_MESH[Nv_default]
    return np.asarray(mesh.weights, dtype=np.float64), np.asarray(mesh.grid_points, dtype=np.float64)


def _geminal(ham, up, dn):
    return np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, dn), dtype=np.complex128)


def _native_parts(ham, up, dn):
    """Native complex e_L parts at one configuration, the inverse taken from numpy."""
    gem = ham.wavefunction_data.geminal_data
    w, gp = _mesh()
    inv = np.ascontiguousarray(np.linalg.inv(_geminal(ham, up, dn)))
    out = {
        "kinetic": complex(*jqmc_rs.compute_kinetic_det_complex(gem, up, dn)),
        "ln_det": jqmc_rs.compute_ln_det_complex(gem, up, dn),
        "e_L_fast": complex(*jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, gp, inv)),
    }
    if ham.coulomb_potential_data.ecp_flag:
        out["ecp_reference"] = complex(*jqmc_rs.compute_ecp_nonlocal_complex(gem, ham.coulomb_potential_data, up, dn, w, gp))
        previous = os.environ.get(_REFERENCE_ENV)
        os.environ[_REFERENCE_ENV] = "1"
        try:
            out["e_L_reference_ecp"] = complex(*jqmc_rs.compute_local_energy_fast_complex(ham, up, dn, w, gp, inv))
        finally:
            if previous is None:
                del os.environ[_REFERENCE_ENV]
            else:
                os.environ[_REFERENCE_ENV] = previous
    return out


def _assert_finite(values, label):
    for key, value in values.items():
        assert np.all(np.isfinite(np.asarray(value))), f"{label}: {key} = {value} is not finite"


def _assert_tiny_valid(ham, up, dn):
    """Nested validity: the scaled G is nonsingular, below sqrt(tiny), with a finite numpy inverse."""
    G = _geminal(ham, up, dn)
    assert 0.0 < np.abs(G).max() < _SQRT_TINY
    inv = np.linalg.inv(G)
    assert np.all(np.isfinite(inv))
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(G @ inv, np.eye(G.shape[0]), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("witness", ["li_k025_ccecp", "h4_k025_ae"])
def test_native_complex_local_energy_parts_are_finite_and_scale_invariant_for_a_tiny_geminal(witness):
    """Native kinetic, ln-det, fast and reference-ECP e_L of a geminal below sqrt(tiny) equal the unscaled values."""
    base = _base_ham(witness)
    up, dn = _start(base)
    n = base.wavefunction_data.geminal_data.num_electron_up

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: _scaled(_base_ham(witness), _SCALE),
            lambda ham: _native_parts(ham, up, dn),
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        assert set(mixed) == set(full)
        _assert_finite(mixed, "mixed")
        _assert_finite(full, "full")
        return

    scaled = _scaled(base, _SCALE)
    _assert_tiny_valid(scaled, up, dn)
    got = _native_parts(scaled, up, dn)
    ref = _native_parts(base, up, dn)
    _assert_finite(got, "scaled")
    _assert_finite(ref, "unscaled")
    assert set(got) == set(ref)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for key in got:
        expected = ref[key] + (2 * n * math.log(_SCALE) if key == "ln_det" else 0.0)
        np.testing.assert_allclose(got[key], expected, atol=atol, rtol=rtol, err_msg=key)
    # Independent references at the scaled configuration: numpy slogdet and the XLA local energy.
    np.testing.assert_allclose(got["ln_det"], np.linalg.slogdet(_geminal(scaled, up, dn))[1], atol=atol, rtol=rtol)
    xla = complex(compute_local_energy(scaled, up, dn, np.eye(3)))
    xatol, xrtol = get_tolerance_xla_rust_consistency(_ZONES)
    for key in ("e_L_fast", "e_L_reference_ecp"):
        if key in got:
            np.testing.assert_allclose(got[key], xla, atol=xatol, rtol=xrtol, err_msg=key)


def _chunk(ham, up, dn, n_steps=5):
    """Native complex energy chunk (``complex=True``, native RNG) for three walkers from one start, epsilon_AS = 0."""
    st = ham.structure_data
    cp = ham.coulomb_potential_data
    g = ham.wavefunction_data.geminal_data
    charges = np.asarray(st.atomic_numbers, dtype=np.float64)
    if cp.ecp_flag:
        charges = charges - np.asarray(cp.z_cores, dtype=np.float64)
    ctx = jqmc_rs.build_obs_ctx(
        g,
        ham.wavefunction_data.jastrow_data,
        cp,
        True,
        True,
        False,
        False,
        False,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    mesh = _NV_BASE_MESH[Nv_default]
    nw = 3
    out = ctx.run_energy_chunk_batch(
        np.asarray(st.positions, dtype=np.float64),
        charges,
        np.ascontiguousarray(np.broadcast_to(up, (nw, *up.shape))),
        np.ascontiguousarray(np.broadcast_to(dn, (nw, *dn.shape))),
        np.zeros((nw, 1, 1)),
        np.zeros((nw, 1, 1)),
        1.2,
        0.0,
        np.asarray([[777, 35], [0, 13], [29, 63]], dtype=np.uint32),
        np.zeros((nw, 0, 0)),
        np.zeros((nw, 0, 0, 0)),
        np.asarray(mesh.weights, dtype=np.float64),
        1,
        np.asarray(st.cell, dtype=np.float64),
        int(st._n_max),
        native_rng=True,
        random_mesh=True,
        base_grid_points=np.asarray(mesh.grid_points, dtype=np.float64),
        n_steps_fast=n_steps,
        complex=True,
    )
    return {
        "e_L": np.asarray(out["e_L"]),
        "accepted": np.asarray(out["accepted"]),
        "r_up": np.asarray(out["r_up"]),
        "r_dn": np.asarray(out["r_dn"]),
        "geminal_inv": np.asarray(out["geminal_inv"]),
    }


@pytest.mark.light
@pytest.mark.parametrize("witness", ["li_k025_ccecp", "h4_k025_ae"])
def test_native_complex_energy_chunk_is_finite_and_scale_invariant_for_a_tiny_geminal(witness):
    """The native complex energy chunk of a geminal below sqrt(tiny) reproduces the unscaled chain and e_L."""
    base = _base_ham(witness)
    up, dn = _start(base)

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: _scaled(_base_ham(witness), _SCALE),
            lambda ham: _chunk(ham, up, dn),
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        _assert_finite(mixed, "mixed")
        _assert_finite(full, "full")
        assert int(full["accepted"].sum()) > 0
        np.testing.assert_array_equal(mixed["accepted"], full["accepted"])
        return

    scaled = _scaled(base, _SCALE)
    _assert_tiny_valid(scaled, up, dn)
    got = _chunk(scaled, up, dn)
    ref = _chunk(base, up, dn)
    _assert_finite(got, "scaled")
    _assert_finite(ref, "unscaled")
    assert int(ref["accepted"].sum()) > 0
    np.testing.assert_array_equal(got["accepted"], ref["accepted"])
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for key in ("e_L", "r_up", "r_dn"):
        np.testing.assert_allclose(got[key], ref[key], atol=atol, rtol=rtol, err_msg=key)
    # G -> c^2 G, so G^-1 -> G^-1 / c^2.
    np.testing.assert_allclose(got["geminal_inv"] * _SCALE**2, ref["geminal_inv"], atol=atol, rtol=rtol)


def _drive(ham, driver):
    """Public driver on ``ham`` (Rust backend, all other arguments default); returns its stored observables."""
    if driver == "gfmc_n":
        gf = GFMC_n(
            hamiltonian_data=ham,
            num_walkers=2,
            num_gfmc_collect_steps=2,
            mcmc_seed=34356,
            alat=0.30,
            num_mcmc_per_measurement=6,
            E_scf=-1.0,
        )
        gf.run(num_mcmc_steps=10)
        return {"e_L": np.asarray(gf.e_L), "w_L": np.asarray(gf.bare_w_L)}
    kwargs = {"epsilon_AS": 0.0} if driver == "mcmc_eps0" else {}
    m = MCMC(hamiltonian_data=ham, mcmc_seed=34356, num_walkers=2, **kwargs)
    m.run(num_mcmc_steps=20)
    out = {"e_L": np.asarray(m.e_L)}
    if driver == "mcmc_eps0":
        out["acceptance_ratio"] = np.asarray(m.acceptance_ratio)
    return out


@pytest.mark.light
@pytest.mark.parametrize("driver", ["mcmc_default", "mcmc_eps0", "gfmc_n"])
def test_native_complex_public_drivers_have_finite_scale_invariant_e_L_for_a_tiny_geminal(driver, monkeypatch):
    """MCMC.run and GFMC_n (Rust) on the Li twist witness below sqrt(tiny) give finite, unscaled-equal e_L.

    ``mcmc_default``: default epsilon_AS (the chain itself is frozen by the
    AS-factor underflow, a separate defect), so only finite e_L is required.
    ``mcmc_eps0``: e_L and the acceptance ratio equal the unscaled run.
    ``gfmc_n``: e_L and the bare weights equal the unscaled run.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    base = _base_ham("li_k025_ccecp")

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: _scaled(_base_ham("li_k025_ccecp"), _SCALE),
            lambda ham: _drive(ham, driver),
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        _assert_finite(mixed, "mixed")
        _assert_finite(full, "full")
        if driver == "mcmc_eps0":
            # Decision check: the mixed chain accepts exactly the moves of the full chain.
            assert float(full["acceptance_ratio"]) > 0.0
            np.testing.assert_array_equal(mixed["acceptance_ratio"], full["acceptance_ratio"])
        return

    got = _drive(_scaled(base, _SCALE), driver)
    _assert_finite(got, "scaled")
    if driver == "mcmc_default":
        return
    ref = _drive(base, driver)
    _assert_finite(ref, "unscaled")
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    for key in ref:
        np.testing.assert_allclose(got[key], ref[key], atol=atol, rtol=rtol, err_msg=key)
    if driver == "mcmc_eps0":
        assert float(ref["acceptance_ratio"]) > 0.0


def _free_route(ham, up, dn, route):
    """One free native complex route on the Li twist witness, flattened to a complex/real vector."""
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cd = ham.coulomb_potential_data
    w, gp = _mesh()
    alat = 0.30
    if route.startswith("lrdmc_ecp"):
        fast = route.endswith("_fast")
        values = jqmc_rs._gfmc_lrdmc_ecp_complex(gd, jd, cd, up, dn, alat, np.eye(3), w, gp, 1, False, fast)[:4]
    elif route.startswith("lrdmc_kinetic"):
        fast = route.endswith("_fast")
        values = jqmc_rs._gfmc_lrdmc_ae_complex(gd, jd, cd, up, dn, alat, np.eye(3), fast)
    elif route == "ecp_force":
        values = jqmc_rs.grad_V_ecp_nonlocal_wrt_R_complex(gd, cd, up, dn, w, gp)
    elif route == "ecp_strain":
        values = jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_complex(gd, cd, up, dn, w, gp)
    else:
        values = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(gd, jd, cd, up, dn, alat, np.eye(3), w, gp, 1, False)
    if isinstance(values, tuple):
        return {"values": np.concatenate([np.ravel(np.asarray(v, dtype=np.float64)) for v in values])}
    return {"values": np.ravel(np.asarray(values))}


@pytest.mark.light
@pytest.mark.parametrize(
    "route",
    [
        "lrdmc_ecp_reference",
        "lrdmc_ecp_fast",
        "lrdmc_kinetic_reference",
        "lrdmc_kinetic_fast",
        "ecp_force",
        "ecp_strain",
        "gfmc_hf_force_stress",
    ],
)
def test_native_complex_gfmc_force_stress_routes_are_finite_and_scale_invariant_for_a_tiny_geminal(route):
    """Native complex LRDMC e_L / b_L (reference and fast mesh), ECP force / strain and GFMC HF force-stress below sqrt(tiny).

    The reference (non-fast) LRDMC kinetic and ECP meshes use the ratio
    ``det(G_mesh) / det(G_base)``, the fast ones seed from the native
    inverse; the ECP force / strain and the GFMC HF force use the same mesh
    determinant ratios. All are invariant under a common MO scale.

    Mixed precision: besides the finite mixed / full outputs, the scaled mixed
    outputs must equal the unscaled outputs evaluated under the same mixed policy
    at the same configuration (a non-AD same-precision invariant; the common MO
    scale acts on the policy-independent MO coefficients, so the ratios agree to
    the ``det_ratio`` zone tolerance of the active policy).
    """
    base = _base_ham("li_k025_ccecp")
    up, dn = _start(base)

    if is_mixed_precision_enabled():
        mixed, full = mixed_vs_full(
            lambda: _scaled(_base_ham("li_k025_ccecp"), _SCALE),
            lambda ham: _free_route(ham, up, dn, route),
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        _assert_finite(mixed, "mixed")
        _assert_finite(full, "full")
        # Same-precision scale invariance (issue #164): the unscaled case
        # evaluated under the restored mixed policy at the same configuration.
        mixed_ref = _free_route(_base_ham("li_k025_ccecp"), up, dn, route)
        _assert_finite(mixed_ref, "mixed unscaled")
        assert np.abs(mixed_ref["values"]).max() > 0.0
        atol, rtol = get_tolerance("det_ratio", "strict")
        np.testing.assert_allclose(mixed["values"], mixed_ref["values"], atol=atol, rtol=rtol, err_msg=f"mixed {route}")
        return

    scaled = _scaled(base, _SCALE)
    _assert_tiny_valid(scaled, up, dn)
    got = _free_route(scaled, up, dn, route)
    ref = _free_route(base, up, dn, route)
    _assert_finite(got, "scaled")
    _assert_finite(ref, "unscaled")
    assert np.abs(ref["values"]).max() > 0.0
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    np.testing.assert_allclose(got["values"], ref["values"], atol=atol, rtol=rtol)
