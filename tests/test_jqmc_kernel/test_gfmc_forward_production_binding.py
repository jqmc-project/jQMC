"""Bind derivative-test forwards to actual GFMC measurement energy, not to another derivative.

Eight declared fixture/move rows cover molecular AE, molecular ECP tmove/dltmove,
Gamma and non-Gamma periodic ECP, AO J3 and DTN (including composed J3+DTN).
Each row runs GFMC_t and GFMC_n at zero and nonzero affine strain. This is a
representative forward-family binding, not the full derivative Cartesian matrix,
an independent physical oracle, a finite difference, or a sampler trajectory.
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

import ast
import inspect
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import Ewald_data
from jqmc.structure import Structure_data
from tests.test_jqmc_kernel import test_gfmc_force_analytic as forward

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

# The move axis is explicit, not multiplied over AE (which has no ECP branch).
_CASES = {
    "h2-ae-jnone": ("H2_ae_ccpvdz_cart.h5", "jnone", "tmove"),
    "h2-ae-j3-dtn": ("H2_ae_ccpvdz_cart.h5", "j123ao_dtn", "tmove"),
    "n2-ecp-j12-tmove": ("N2_ecp_ccpvtz_cart.h5", "j12", "tmove"),
    "n2-ecp-j12-dltmove": ("N2_ecp_ccpvtz_cart.h5", "j12", "dltmove"),
    "li-gamma-j3-tmove": ("lithium.h5", "j123ao", "tmove"),
    "li-gamma-j3-dltmove": ("lithium.h5", "j123ao", "dltmove"),
    "li-twist-j3-tmove": ("lithium_k025.h5", "j123ao", "tmove"),
    "li-twist-dtn-dltmove": ("lithium_k025.h5", "dtn", "dltmove"),
}
_SEED = 2026
_ALAT = 0.3
_ORIGIN_SHIFT = np.array([0.375, 0.125, -0.25])
_STRAINS = {
    "unstrained": np.zeros((3, 3)),
    "affine": np.array([[0.013, 0.007, -0.004], [0.002, -0.009, 0.005], [0.006, -0.003, 0.011]]),
}
_BASE_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "det_ratio", "wf_kinetic", "coulomb", "local_energy")


@cache
def _production_n():
    """Compile the unique production GFMC_n.run closure without rewriting its body.

    There is no exported module-level n callable. This is the production
    definition, executed in its production module globals, not a formula copied
    into this test. Undefined new closure dependencies fail rather than falling
    back to the t implementation. The force/stress route uses a fresh inverse
    (A_old_inv=None), with use_fast_update=False; both sides use that same route.
    """
    tree = ast.parse(inspect.getsource(jqmc_gfmc))
    (cls,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GFMC_n"]
    (run,) = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    (definition,) = [
        node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "_compute_V_elements_n"
    ]
    namespace = dict(vars(jqmc_gfmc))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), inspect.getsourcefile(jqmc_gfmc), "exec"), namespace)
    return jax.jit(namespace["_compute_V_elements_n"], static_argnums=(5, 7))


def _translated(ham):
    """Rebuild every structure at a common fixed origin, including periodic Ewald.

    Translation is a fixture convention, not an energy correction. It avoids an
    accidentally zero nuclear x sum in origin-dependent negative controls.
    """

    def translate(sd):
        options = dict(
            positions=np.asarray(sd.positions) + _ORIGIN_SHIFT,
            pbc_flag=sd.pbc_flag,
            atomic_numbers=tuple(sd.atomic_numbers),
            element_symbols=tuple(sd.element_symbols),
            atomic_labels=tuple(sd.atomic_labels),
        )
        if sd.pbc_flag:
            options.update(vec_a=sd.vec_a, vec_b=sd.vec_b, vec_c=sd.vec_c)
        return Structure_data.build(**options)

    result = jax.tree_util.tree_map(
        lambda value: translate(value) if isinstance(value, Structure_data) else value,
        ham,
        is_leaf=lambda value: isinstance(value, Structure_data),
    )
    cp = result.coulomb_potential_data
    if cp.ewald_data is not None:
        old = ham.coulomb_potential_data.ewald_data
        ewald = Ewald_data.build(
            structure_data=cp.structure_data,
            coulomb_potential_data=cp,
            eta_override=old.eta,
            r_real_cut_override=old.r_real_cut,
            g_max_override=old.g_max,
        )
        result = result.replace(coulomb_potential_data=cp.replace(ewald_data=ewald))
    return result


@cache
def _inputs(case):
    stem, kind, _move = _CASES[case]
    if stem == "lithium_k025.h5" and kind == "dtn":
        ham = forward._build_ham_twist_dtn()
    elif "dtn" in kind:
        ham = forward._build_ham_dtn(stem, kind)
    else:
        ham = forward._build_ham_ecp(stem, kind)
    ham = _translated(ham)
    rng = np.random.default_rng(_SEED)
    nuclei = np.asarray(ham.structure_data.positions)
    geminal = ham.wavefunction_data.geminal_data

    def shell(count, offset):
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.uniform(0.6, 1.2, size=(count, 1))
        return jnp.asarray(nuclei[(np.arange(count) + offset) % len(nuclei)] + radii * directions)

    r_up = shell(int(geminal.num_electron_up), 0)
    # Both spins near the same Li nucleus keep DTN u/f inside their cutoffs;
    # separate home nuclei make the ECP-sphere Jastrow ratio radial and unity.
    r_dn = shell(int(geminal.num_electron_dn), 0 if case == "li-twist-dtn-dltmove" else 1)
    return ham, r_up, r_dn, forward._rot(7)


@cache
def _reference(algorithm, ecp, strained):
    name = f"_e_L_{algorithm}_{'ecp' if ecp else 'ae_jnone'}{'_strained' if strained else ''}"
    return jax.jit(getattr(forward, name), static_argnames=("non_local_move",) if ecp else ())


@jax.jit
def _ecp_activity(ham, up, dn, rt):
    """Witness nonzero ECP and the dltmove-specific positive/SP localization term.

    Negative/FN candidates include the Jastrow ratio in both tmove and dltmove.
    Their difference is on positive/SP candidates: dltmove omits that ratio.
    """
    wavefunction = ham.wavefunction_data
    mesh_up, mesh_dn, values, _ = forward.compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data, wavefunction, up, dn, flag_determinant_only=True, RT=rt
    )
    baseline = forward.compute_Jastrow_part(wavefunction.jastrow_data, up, dn)
    mesh = jax.vmap(forward.compute_Jastrow_part, in_axes=(None, 0, 0))(wavefunction.jastrow_data, mesh_up, mesh_dn)
    ratio = jnp.exp(mesh - baseline)
    correction = jnp.sum(jnp.maximum(values, 0) * (ratio - 1))
    return jnp.any(values != 0), correction


def _values(case, algorithm, strain):
    ham, up, dn, rt = _inputs(case)
    _stem, kind, move = _CASES[case]
    eps = jnp.asarray(_STRAINS[strain])
    transform = jnp.eye(3) + eps
    actual_ham = ham if strain == "unstrained" else ham._apply_strain(transform)
    # Electrons follow the strain; the ECP quadrature rotation stays fixed,
    # matching the strained derivative-test forwards in test_gfmc_force_analytic.
    actual_up, actual_dn = (value @ transform.T for value in (up, dn))
    actual_rt = rt
    if ham.coulomb_potential_data.ecp_flag:
        nonlocal_live, localization_correction = _ecp_activity(actual_ham, actual_up, actual_dn, actual_rt)
        assert bool(nonlocal_live), f"{case}/{strain}: ECP nonlocal candidates are all zero"
        if move == "dltmove":
            assert np.isfinite(localization_correction) and float(localization_correction) != 0, (
                f"{case}/{strain}: dltmove positive/SP localization correction is zero or nonfinite"
            )
    if algorithm == "t":
        actual = jqmc_gfmc._compute_V_elements_t(actual_ham, actual_up, actual_dn, actual_rt, move, _ALAT)[2]
    else:
        actual = _production_n()(actual_ham, actual_up, actual_dn, actual_rt, actual_rt, move, _ALAT, False)[2]
    arguments = (ham, up, dn, rt, _ALAT) + ((eps,) if strain != "unstrained" else ())
    options = {"non_local_move": move} if ham.coulomb_potential_data.ecp_flag else {}
    expected = _reference(algorithm, bool(ham.coulomb_potential_data.ecp_flag), strain != "unstrained")(*arguments, **options)
    zones = _BASE_ZONES
    if kind != "jnone":
        zones += ("jastrow_eval", "jastrow_grad_lap")
    if ham.structure_data.pbc_flag:
        zones += ("ewald",)
    return np.asarray(jnp.real(actual)), np.asarray(expected), zones


@pytest.mark.parametrize("case", tuple(_CASES))
@pytest.mark.parametrize("algorithm", ("t", "n"))
@pytest.mark.parametrize("strain", tuple(_STRAINS))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_derivative_forward_matches_production_measurement(case, algorithm, strain):
    """Same-mode Re(e_L) from the real t/n measurement equals the derivative-test forward.

    Inputs are committed H2 AE Cartesian, N2 ccECP Cartesian, and Li Gamma/k025
    spherical periodic TREXIO SD fixtures, with the eight declared Jastrow/move
    rows above. Rebuild their structure leaves at origin shift
    (0.375,0.125,-0.25) Bohr. One NumPy seed-2026 shell configuration puts each
    up/down electron 0.6--1.2 Bohr from cyclic nucleus i/i+1, except the twist
    DTN row puts both spins near nucleus i so its u/f terms are inside cutoff.
    No rejection scan or condition-number screen is used. QR rotation seed 7 and alat=0.3 define
    the common mesh. Both routes recompute the same geminal inverse; no carried
    inverse, sampled trajectory, propagation energy, or FD estimate is compared.

    The nonzero strain is the explicit nine-component _STRAINS['affine']; the
    production input is ham._apply_strain(I+eps), with electrons and RT mapped
    by (I+eps).T, matching the driver's strained measurement convention. The
    other side calls the existing _e_L_t/_n strained or unstrained reproduction.
    Both use the same active precision. Compare the physical real scalar
    measurement with central strict active energy-zone atol/rtol, as the existing
    GFMC nodal/ECP energy anchors do, not the derivative matrix's mixed-medium
    gradient policy. Finiteness is required independently; NaNs cannot agree.
    ECP candidates must be nonzero; each dltmove row must have a finite nonzero
    positive/SP localization correction sum(max(V_det,0)*(J_ratio-1)) (exact
    activity witness, not a separate tolerance-relaxed energy comparison).

    Twist J3 agreement does not certify the separate lattice-periodicity defect of J3 at a twist.
    This binds selected forward families, not all derivative-matrix fixtures.
    """
    actual, expected, zones = _values(case, algorithm, strain)
    assert actual.shape == expected.shape == (), "forward-production measurement must be a scalar on both routes"
    assert np.isfinite(actual) and np.isfinite(expected), "forward-production binding has non-finite energy"
    # Keep each source-bound budget explicit without changing the active zones
    # returned by _values. Distinct names let the existing GDD resolver bind
    # every conditional comparison to its authored strict policy.
    if case == "h2-ae-jnone":
        assert zones == _BASE_ZONES
        base_atol, base_rtol = get_tolerance_min(_BASE_ZONES, "strict")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=base_atol,
            rtol=base_rtol,
            equal_nan=False,
            err_msg=f"forward-production measurement binding: {case}/{algorithm}/{strain}",
        )
    elif case.startswith("li-"):
        periodic_zones = _BASE_ZONES + ("jastrow_eval", "jastrow_grad_lap", "ewald")
        assert zones == periodic_zones
        periodic_atol, periodic_rtol = get_tolerance_min(periodic_zones, "strict")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=periodic_atol,
            rtol=periodic_rtol,
            equal_nan=False,
            err_msg=f"forward-production measurement binding: {case}/{algorithm}/{strain}",
        )
    else:
        jastrow_zones = _BASE_ZONES + ("jastrow_eval", "jastrow_grad_lap")
        assert zones == jastrow_zones
        jastrow_atol, jastrow_rtol = get_tolerance_min(jastrow_zones, "strict")
        np.testing.assert_allclose(
            actual,
            expected,
            atol=jastrow_atol,
            rtol=jastrow_rtol,
            equal_nan=False,
            err_msg=f"forward-production measurement binding: {case}/{algorithm}/{strain}",
        )
