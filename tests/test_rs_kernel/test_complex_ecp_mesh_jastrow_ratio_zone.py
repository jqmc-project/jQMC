"""Issue #757: native complex-twist ECP mesh kernels keep their Jastrow ratio in the fp64 jastrow_ratio zone.

The precision policy (``jqmc/_precision.py``) assigns the ECP / discretized-hop
Jastrow ratio ``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes);
whole Jastrow values belong to ``jastrow_eval`` (float32 in mixed). Several
native complex-twist kernels formed that ratio from two whole
``jastrow_value_complex`` values, i.e. in ``jastrow_eval``:

* the mesh-force / mesh-strain fallbacks of ``gfmc_forces``
  (``gfmc_ecp_nonlocal_force_complex`` incl. its grouped ``MeshJasC`` path, which
  only a Cartesian determinant selects, and ``gfmc_ecp_nonlocal_strain_complex``),
  taken whenever the wavefunction carries a J3 (the one-electron Jastrow mesh covers J1/J2/DTN only). Production reaches
  them through the MCMC force / stress batch entries
  (``NativeObsCtx.grad_e_L_pbc_complex_batch`` / ``grad_e_L_eps_pbc_complex_batch``)
  and the GFMC force / stress fold;
* the free ``complex_stress`` ECP non-local force / strain kernels
  (``grad_V_ecp_nonlocal_wrt_R_withJ_complex`` /
  ``grad_V_ecp_nonlocal_wrt_strain_withJ_complex``);
* the full-recompute reference routes: ``compute_ecp_nonlocal_withj_complex``
  (the complex e_L fallback, ``compute_ecp_nonlocal_withJ_complex``), the
  reference branch of ``compute_ecp_nonlocal_mesh_complex`` and the reference
  discretized-kinetic hops of the complex LRDMC e_L
  (``_gfmc_lrdmc_ecp_complex(..., fast=False)``, the injected-rotation chain).

Fixture: ``twist_sphepgto_j123`` (lithium_k025.h5, bcc Li ccECP,
k = (1/4, 1/4, 1/4), complex sphe-pGTO single determinant, exp J1/J2 and a
complex Bloch pGTO J3), the three ``_walker_batch(seed=0)`` walkers and the three
production per-walker rotations of ``test_batch_mesh_per_walker`` on the default
Nv mesh.

Two detectors:

* ``test_twist_ecp_mesh_kernels_do_not_read_jastrow_eval``: force / stress
  outputs also legitimately cross ``jastrow_grad_lap`` (float32 in mixed), so
  their whole mixed error cannot isolate the ratio. Each route is therefore
  evaluated with the float64 policy except ``jastrow_eval``, which takes the
  active mode's dtype (float32 in a mixed run); in a full run the policy is all
  float64 (a same-case consistency check). None of these outputs consumes a whole
  Jastrow value, so actual and reference execute the same float64 arithmetic on
  the same serialized input and may differ by reduction order only
  (``ATOL_REDUCTION_ORDER`` / ``RTOL_REDUCTION_ORDER``). A route that forms the
  ratio from whole values in ``jastrow_eval`` makes the force / stress routes exceed
  the strict float64 budget of their zones (max abs 5.5e-7, rel 4.5e-5 on near-zero
  elements) and gives the reference energy routes a float32-sized error (2.1e-7 on
  the ECP energy, 4.5e-6 on ``b_L``). The reference is an independently loaded full-precision Hamiltonian
  of the same serialized input; the full MCMC force / stress must also equal XLA
  at the strict float64 budget.
* ``test_twist_ecp_reference_routes_match_full_precision_rebuild``: the ECP
  non-local energy of the reference route and the reference LRDMC branching sum
  ``b_L`` read only zones that are float64 in both modes at a twist
  (``ao_bloch_eval``, ``mo_eval``, ``det_eval``, ``det_ratio``, ``jastrow_ratio``,
  ``coulomb``), so under the actual mixed policy the native values run the same
  float64 arithmetic as the full rebuild and must equal it to reduction order.
  XLA's holomorphic ECP sum (whose mixed forward may cross float32 real-AO
  zones) is held to the strict float64 budget of those zones.
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

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER, Nv_default
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_holomorphic
from jqmc.derivatives import assemble
from tests._precision_fidelity import mixed_vs_full
from tests.test_rs_kernel import test_batch_mesh_per_walker as per_walker
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_mcmc_observables import _geminal_inv, _is_complex, _walker_batch

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_PATTERN = "twist_sphepgto_j123"
# Index of ``jqmc_rs.gfmc_route_counters()`` (append-only list in gfmc.rs): grouped complex
# mesh-force candidates served by the fallback ``MeshJasC::ratio``.
_CPLX_ECPF_GROUPED_JAS_RATIOS = 73
_ALAT = 0.3
_NN = 1
# Zones float64 in BOTH modes that these outputs cross; their strict budget is the
# float64 strict budget in a full and in a mixed run alike.
_ZONES_F64 = (
    "ao_bloch_eval",
    "mo_eval",
    "mo_grad",
    "det_eval",
    "det_grad_lap",
    "det_ratio",
    "jastrow_ratio",
    "coulomb",
    "ewald",
)
# Every zone of the reference ECP energy / LRDMC b_L at a twist (float64 in both modes).
_ZONES_REFERENCE = ("ao_bloch_eval", "mo_eval", "det_eval", "det_ratio", "jastrow_ratio", "coulomb")


def _cartesian_determinant_ham():
    """``twist_sphepgto_j123`` with ONLY the determinant converted to Cartesian AOs.

    The production ``geminal_data.to_cartesian()`` conversion; the active complex
    spherical J3, walkers and rotations are unchanged. A Cartesian determinant passes
    the ``mesh_vgs::supported`` gate, so the complex mesh force runs the GROUPED
    reduction (``ecp_nl_force_complex_grouped``), whose J3 fallback ratio is
    ``MeshJasC::ratio``; the spherical fixture never reaches it.
    """
    ham = rs_ham(_PATTERN)
    wf = ham.wavefunction_data
    return ham.replace(wavefunction_data=wf.replace(geminal_data=wf.geminal_data.to_cartesian()))


_BUILDERS = {
    "mcmc_force_stress": lambda: rs_ham(_PATTERN),
    "mcmc_force_stress_cart_grouped": _cartesian_determinant_ham,
    "gfmc_force_stress": lambda: rs_ham(_PATTERN),
    "free_force_strain": lambda: rs_ham(_PATTERN),
    "reference_energy": lambda: rs_ham(_PATTERN),
}
_MCMC_ROUTES = ("mcmc_force_stress", "mcmc_force_stress_cart_grouped")


def _case(ham):
    """Walkers, per-walker rotations and swept-style inverses of the fixture."""
    assert _is_complex(ham) and bool(ham.coulomb_potential_data.ecp_flag)
    jas = ham.wavefunction_data.jastrow_data
    assert jas.jastrow_three_body_data is not None, "the J3 leaf is what selects the mesh fallbacks"
    nw = per_walker._NW
    r_up, r_dn = _walker_batch(ham, nw=nw, seed=0)
    r_up, r_dn = np.ascontiguousarray(r_up), np.ascontiguousarray(r_dn)
    rts = per_walker._distinct_rotations(nw)
    ginv = np.stack([_geminal_inv(ham, r_up[w], r_dn[w]) for w in range(nw)]).astype(np.complex128)
    return r_up, r_dn, rts, ginv


def _native_route(ham, route, policy):
    """Native outputs of one route with the native policy ``policy``."""
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    gd, jd = wf.geminal_data, wf.jastrow_data
    r_up, r_dn, rts, ginv = _case(ham)
    nw = per_walker._NW
    jqmc_rs.configure_native_precision(policy)
    out = {"r_up": r_up, "r_dn": r_dn}
    if route in _MCMC_ROUTES:
        jqmc_rs.gfmc_route_counters_reset()
        ctx = jqmc_rs.build_obs_ctx(
            gd, jd, cp, True, True, True, want_force=True, want_gfmc=False, want_param_deriv=False, precision_policy=policy
        )
        gp_all = per_walker._GP @ rts
        g_r, g_up, g_dn = (
            np.asarray(x, float) for x in ctx.grad_e_L_pbc_complex_batch(r_up, r_dn, per_walker._W, gp_all, ginv)
        )
        out["force_R"], out["force_r"] = g_r, np.concatenate([g_up, g_dn], axis=1)
        out["stress"] = np.asarray(ctx.grad_e_L_eps_pbc_complex_batch(r_up, r_dn, per_walker._W, gp_all, ginv), float)
        grouped_ratios = int(jqmc_rs.gfmc_route_counters()[_CPLX_ECPF_GROUPED_JAS_RATIOS])
        if route == "mcmc_force_stress_cart_grouped":
            assert grouped_ratios > 0, "the Cartesian-determinant case must reach the grouped MeshJasC::ratio"
        else:
            assert grouped_ratios == 0, "the spherical fixture takes the per-candidate mesh-force path"
        return out
    rows = {}
    for w in range(nw):
        gp = per_walker._GP @ rts[w]
        if route == "gfmc_force_stress":
            f = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
                gd, jd, cp, r_up[w], r_dn[w], _ALAT, rts[w], per_walker._W, per_walker._GP, _NN
            )
            rows.setdefault("force", []).append(np.concatenate([np.ravel(np.asarray(x, float)) for x in f]))
            s = jqmc_rs._gfmc_stress_ecp_pbc_complex(
                gd, jd, cp, r_up[w], r_dn[w], _ALAT, rts[w], per_walker._W, per_walker._GP, _NN
            )
            rows.setdefault("stress", []).append(np.asarray(s, float))
        elif route == "free_force_strain":
            f = jqmc_rs.grad_V_ecp_nonlocal_wrt_R_withJ_complex(gd, jd, cp, r_up[w], r_dn[w], per_walker._W, gp)
            rows.setdefault("force", []).append(np.asarray(f, float))
            s = jqmc_rs.grad_V_ecp_nonlocal_wrt_strain_withJ_complex(gd, jd, cp, r_up[w], r_dn[w], per_walker._W, gp)
            rows.setdefault("strain", []).append(np.asarray(s, float))
        elif route == "reference_energy":
            rows.setdefault("vnl", []).append(
                complex(*jqmc_rs.compute_ecp_nonlocal_withJ_complex(gd, jd, cp, r_up[w], r_dn[w], per_walker._W, gp))
            )
            e_re, e_im, b_re, b_im, _ = jqmc_rs._gfmc_lrdmc_ecp_complex(
                gd, jd, cp, r_up[w], r_dn[w], _ALAT, rts[w], per_walker._W, per_walker._GP, _NN, False, False
            )
            rows.setdefault("lrdmc_e_L", []).append(complex(e_re, e_im))
            rows.setdefault("lrdmc_b_L", []).append(complex(b_re, b_im))
        else:
            raise AssertionError(route)
    out.update({k: np.stack(v) if np.ndim(v[0]) else np.asarray(v) for k, v in rows.items()})
    return out


def _xla_mcmc_force_stress(ham):
    """XLA dE_L/d(R, r, eps) of the same walkers and rotations (full-precision reference)."""
    r_up, r_dn, rts, ginv = _case(ham)
    f_r, f_e, s = [], [], []
    for w in range(per_walker._NW):
        args = (ham, jnp.asarray(r_up[w]), jnp.asarray(r_dn[w]), rts[w])
        gi = jnp.asarray(ginv[w])
        f_r.append(np.real(np.asarray(assemble.compute_grad_e_L(*args, wrt="R", geminal_inv=gi))))
        g_up, g_dn = assemble.compute_grad_e_L(*args, wrt="r", geminal_inv=gi)
        f_e.append(np.concatenate([np.real(np.asarray(g_up)), np.real(np.asarray(g_dn))]))
        s.append(np.real(np.asarray(assemble.compute_grad_e_L(*args, wrt="eps", geminal_inv=gi))))
    return {"force_R": np.stack(f_r), "force_r": np.stack(f_e), "stress": np.stack(s)}


@contextlib.contextmanager
def _full_python_and_restored_native():
    """Run the body under the full XLA policy; restore both backends' entry policies."""
    was_mixed = is_mixed_precision_enabled()
    native_entry = dict(jqmc_rs.native_precision_support())
    configure("full")
    try:
        yield
    finally:
        try:
            configure("mixed" if was_mixed else "full")
        finally:
            jqmc_rs.configure_native_precision(native_entry)


_ROUTES = [
    pytest.param("mcmc_force_stress", marks=pytest.mark.light, id="mcmc_force_stress"),
    pytest.param("mcmc_force_stress_cart_grouped", marks=pytest.mark.light, id="mcmc_force_stress_cart_grouped"),
    pytest.param("gfmc_force_stress", marks=pytest.mark.heavy, id="gfmc_force_stress"),
    pytest.param("free_force_strain", marks=pytest.mark.heavy, id="free_force_strain"),
    pytest.param("reference_energy", marks=pytest.mark.heavy, id="reference_energy"),
]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("route", _ROUTES)
def test_twist_ecp_mesh_kernels_do_not_read_jastrow_eval(route):
    """Native twist ECP mesh outputs with the mode's jastrow_eval dtype equal the float64 rebuild.

    ``twist_sphepgto_j123``, ``_walker_batch(seed=0)`` walkers, the production
    per-walker rotations, default Nv mesh, NN = 1, ``alat = 0.3``. The actual value
    uses the all-float64 native policy except ``jastrow_eval``, which takes the
    active mode's dtype (float32 in a mixed run); the reference uses the all-float64
    native policy on an independently loaded Hamiltonian of the same serialized full
    input (and, for the MCMC force / stress, XLA on that Hamiltonian). No output
    here consumes a whole Jastrow value (the mesh ratio is ``jastrow_ratio``, the
    drift / Pulay / strain fields ``jastrow_grad_lap``), so actual and reference run
    the same float64 arithmetic and must agree elementwise to reduction order. The
    full native MCMC force / stress must equal XLA at the strict float64 budget.
    Outputs must be finite and live. ``mcmc_force_stress_cart_grouped`` is the same
    input with only the determinant converted to Cartesian AOs, which selects the
    grouped mesh-force reduction; a native route counter must show that its fallback
    ratio ``MeshJasC::ratio`` ran (and that the spherical case does not reach it).
    """
    mode_jastrow_eval = precision_policy_snapshot()["jastrow_eval"]
    with _full_python_and_restored_native():
        full_policy = precision_policy_snapshot()
        assert set(full_policy.values()) == {"float64"}
        actual_policy = {**full_policy, "jastrow_eval": mode_jastrow_eval}
        factory = full_source_hamiltonian_factory(_BUILDERS[route])
        reference_ham = factory()
        reference = _native_route(reference_ham, route, full_policy)
        xla = _xla_mcmc_force_stress(reference_ham) if route in _MCMC_ROUTES else None
        actual = _native_route(factory(), route, actual_policy)
    np.testing.assert_array_equal(actual["r_up"], reference["r_up"])
    np.testing.assert_array_equal(actual["r_dn"], reference["r_dn"])
    atol, rtol = get_tolerance_min(_ZONES_F64, "strict")
    for key in sorted(set(reference) - {"r_up", "r_dn"}):
        got, ref = np.asarray(actual[key]), np.asarray(reference[key])
        assert got.shape == ref.shape and got.shape[0] == per_walker._NW, (key, got.shape, ref.shape)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{route} {key}: non-finite"
        assert np.max(np.abs(ref)) > 1.0, f"{route} {key}: inactive output"
        if xla is not None:
            # Independent implementation of the same full-precision quantity.
            np.testing.assert_allclose(ref, xla[key], atol=atol, rtol=rtol, err_msg=f"{route} {key}: native full vs XLA full")
        for part, f in (("Re", np.real), ("Im", np.imag)):
            np.testing.assert_allclose(
                f(got),
                f(ref),
                atol=ATOL_REDUCTION_ORDER,
                rtol=RTOL_REDUCTION_ORDER,
                err_msg=f"{route} {key} ({part}): jastrow_eval={mode_jastrow_eval}",
            )


def _reference_energies(ham):
    """Native reference-route ECP energy / LRDMC b_L and XLA's holomorphic ECP sum."""
    nw = per_walker._NW
    native = _native_route(ham, "reference_energy", precision_policy_snapshot())
    r_up, r_dn, rts, _ = _case(ham)
    natom = np.asarray(ham.structure_data.positions).shape[0]
    xla = [
        complex(
            compute_ecp_non_local_parts_nearest_neighbors_holomorphic(
                ham.coulomb_potential_data,
                ham.wavefunction_data,
                jnp.asarray(r_up[w]),
                jnp.asarray(r_dn[w]),
                jnp.asarray(rts[w]),
                NN=natom,
                Nv=Nv_default,
                flag_determinant_only=False,
            )[3]
        )
        for w in range(nw)
    ]
    return {**native, "xla_vnl": np.asarray(xla)}


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("backend", ["native", "xla"])
def test_twist_ecp_reference_routes_match_full_precision_rebuild(backend):
    """Mixed twist reference ECP energy (and native reference LRDMC b_L) equal the full rebuild.

    Same fixture, walkers and rotations as above. In mixed the actual values come
    from a Hamiltonian loaded from the serialized full input under the mixed policy
    (both backends) and the reference from an independently loaded full Hamiltonian;
    in full both come from the same case (a consistency check). Native: the ECP
    non-local energy of ``compute_ecp_nonlocal_withJ_complex`` (the complex e_L
    fallback route) and ``b_L`` of the reference LRDMC branch
    ``_gfmc_lrdmc_ecp_complex(..., fast=False)`` (reference discretized hops), held to
    reduction order since every zone they read is float64 in both modes. XLA: the
    holomorphic nearest-neighbour ECP sum at the strict float64 budget of those
    zones. The full native ECP energy must equal XLA's at that budget. Values must
    be finite and live.
    """
    factory = full_source_hamiltonian_factory(lambda: rs_ham(_PATTERN))
    if is_mixed_precision_enabled():
        result, full = mixed_vs_full(
            factory,
            _reference_energies,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
    else:
        result = full = _reference_energies(factory())
    np.testing.assert_array_equal(result["r_up"], full["r_up"])
    np.testing.assert_array_equal(result["r_dn"], full["r_dn"])
    atol, rtol = get_tolerance_min(_ZONES_REFERENCE, "strict")
    np.testing.assert_allclose(full["vnl"], full["xla_vnl"], atol=atol, rtol=rtol, err_msg="native full vs XLA full ECP energy")
    keys = ("vnl", "lrdmc_b_L") if backend == "native" else ("xla_vnl",)
    if backend == "native":
        atol, rtol = ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
    for key in keys:
        got, ref = np.asarray(result[key]), np.asarray(full[key])
        assert got.shape == ref.shape == (per_walker._NW,)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{key}: non-finite"
        assert np.max(np.abs(ref)) > 1.0, f"{key}: inactive"
        for part, f in (("Re", np.real), ("Im", np.imag)):
            np.testing.assert_allclose(f(got), f(ref), atol=atol, rtol=rtol, err_msg=f"{backend} {key} ({part})")
