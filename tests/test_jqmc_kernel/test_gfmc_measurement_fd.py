"""Finite-difference anchors for the GFMC measurement-energy derivatives (``e_L_t`` / ``e_L_n``).

The analytic GFMC force and stress kernels (``jqmc.derivatives.gfmc.grad_e_L_{t,n}_wrt_{positions,strain}``)
are compared with a central finite difference of the production measurement energy. The numerical
derivative and input rebuilding are independent of the analytic derivative and strained-forward helpers;
the energy implementation itself remains the production implementation, not an independent energy oracle.

* GFMC_t: the third output of the module-level ``jqmc.jqmc_gfmc._compute_V_elements_t``.
* GFMC_n: the third output of the ``_compute_V_elements_n`` closure defined inside
  ``GFMC_n.run``. It is not reachable as an attribute, so its source is taken verbatim from the
  production module and compiled in that module's namespace (``_production_V_elements_n``), then
  called with ``use_fast_update=False`` exactly like the production ``_compute_local_energy_n``.

Every finite-difference point is a physically rebuilt input. A nuclear displacement or a strain
``F = I + eps`` builds each ``Structure_data`` leaf again through ``Structure_data.build`` and
rebuilds the periodic Ewald tables through ``Ewald_data.build`` with the unchanged split and cutoffs.
The electrons move as ``r @ F.T``; the quadrature orientation ``RT`` is held fixed, so every
strained ECP quadrature point lies on the sphere of the strained electron-nucleus distance.
The strained-forward helper ``Hamiltonian_data._apply_strain`` is not used.

Stencil: fourth-order central difference, step ``_H`` (Bohr for positions, dimensionless for
strain). The truncation error is O(h^4 f^(5)). Each case also evaluates the same stencil at ``2 _H``
and requires the two results to agree within the comparison tolerance. This step-doubling check
detects unstable stencils; it is not a rigorous truncation-error bound.
The electron configuration keeps every electron 0.6 to 1.2 Bohr from its home nucleus, so the
finite-difference stencil does not cross the nuclear cusp region (see ``_shell_configuration``).

The finite-difference correctness verdict is full precision only. Both tests explicitly skip
mixed precision because rounding in its fp32 evaluation zones dominates these small stencils.
Mixed analytical correctness retains the existing same-precision family-level AD/identity
guarantees on their own configurations. They are not same-fixture replacements for these FD
cases, and these tests do not substitute a finite/nonzero smoke verdict in mixed precision.
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
import os
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jqmc_gfmc
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.coulomb_potential import Ewald_data, compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import gfmc
from jqmc.structure import Structure_data
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham_jastrow, _rot

pytestmark = [pytest.mark.with_chdir]

# case -> (TREXIO stem, non_local_move, committed source id). The AE case has no ECP move (None is
# the production representation, validated by ``gfmc._validate_non_local_move``).
_CASES = {
    "h2-ae-molecular": ("H2_ae_ccpvdz_cart.h5", None, "WF-SRC-H2-AE-CART-MOL"),
    "n2-ecp-molecular-tmove": ("N2_ecp_ccpvtz_cart.h5", "tmove", "WF-SRC-N2-ECP-CART-MOL"),
    "n2-ecp-molecular-dltmove": ("N2_ecp_ccpvtz_cart.h5", "dltmove", "WF-SRC-N2-ECP-CART-MOL"),
    "li-ecp-periodic-tmove": ("lithium.h5", "tmove", "WF-SRC-LI-ECP-SPHE-GAMMA"),
    "li-ecp-periodic-dltmove": ("lithium.h5", "dltmove", "WF-SRC-LI-ECP-SPHE-GAMMA"),
}
_CASE_IDS = list(_CASES)
# ``alat`` enters only the propagation outputs; the measurement energy does not depend on it.
_ALAT = 0.3
_H = 5.0e-4
_ANALYTIC = {
    "t": (gfmc.grad_e_L_t_wrt_positions, gfmc.grad_e_L_t_wrt_strain),
    "n": (gfmc.grad_e_L_n_wrt_positions, gfmc.grad_e_L_n_wrt_strain),
}


@cache
def _production_V_elements_n():
    """Return the verbatim ``_compute_V_elements_n`` closure of ``GFMC_n.run``.

    The nested definition references only module-level names of ``jqmc.jqmc_gfmc`` (checked when
    this was written), so it is compiled unchanged in that module's namespace.
    """
    source = inspect.getsource(jqmc_gfmc)
    tree = ast.parse(source)
    (gfmc_n,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GFMC_n"]
    (run,) = [node for node in gfmc_n.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    found = [node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "_compute_V_elements_n"]
    assert len(found) == 1, f"expected one production _compute_V_elements_n closure, found {len(found)}"
    module = ast.Module(body=[found[0]], type_ignores=[])
    namespace = dict(vars(jqmc_gfmc))
    exec(compile(module, inspect.getsourcefile(jqmc_gfmc), "exec"), namespace)
    return namespace["_compute_V_elements_n"]


@cache
def _measurement_energy(algorithm: str, non_local_move: str | None):
    """Jitted production measurement energy ``Re e_L(ham, r_up, r_dn, RT)`` of one GFMC algorithm."""
    move = non_local_move or "tmove"  # AE: the ECP branch is not taken, the move label is unused.
    if algorithm == "t":

        def energy(ham, r_up, r_dn, RT):
            return jnp.real(jqmc_gfmc._compute_V_elements_t(ham, r_up, r_dn, RT, move, _ALAT)[2])

    else:
        closure = _production_V_elements_n()

        def energy(ham, r_up, r_dn, RT):
            return jnp.real(closure(ham, r_up, r_dn, RT, RT, move, _ALAT, False)[2])

    return jax.jit(energy)


def _rebuilt(ham, F: np.ndarray, displacement: tuple[int, int, float] | None = None):
    """Rebuild every ``Structure_data`` leaf (and the Ewald tables) for ``x -> F (x + d)``."""

    def structure(sd: Structure_data) -> Structure_data:
        positions = np.array(sd.positions, dtype=np.float64)
        if displacement is not None:
            atom, axis, delta = displacement
            positions[atom, axis] += delta
        kwargs = dict(
            positions=positions @ F.T,
            pbc_flag=sd.pbc_flag,
            atomic_numbers=tuple(sd.atomic_numbers),
            element_symbols=tuple(sd.element_symbols),
            atomic_labels=tuple(sd.atomic_labels),
        )
        if sd.pbc_flag:
            kwargs.update(
                vec_a=F @ np.asarray(sd.vec_a, dtype=np.float64),
                vec_b=F @ np.asarray(sd.vec_b, dtype=np.float64),
                vec_c=F @ np.asarray(sd.vec_c, dtype=np.float64),
            )
        return Structure_data.build(**kwargs)

    def is_structure(node) -> bool:
        return isinstance(node, Structure_data)

    new = jax.tree_util.tree_map(lambda x: structure(x) if is_structure(x) else x, ham, is_leaf=is_structure)
    coulomb = new.coulomb_potential_data
    if coulomb.ewald_data is not None:
        old = ham.coulomb_potential_data.ewald_data
        ewald = Ewald_data.build(
            structure_data=coulomb.structure_data,
            coulomb_potential_data=coulomb,
            eta_override=old.eta,
            r_real_cut_override=old.r_real_cut,
            g_max_override=old.g_max,
        )
        new = new.replace(coulomb_potential_data=coulomb.replace(ewald_data=ewald))
    return new


def _shell_configuration(ham, seed: int):
    """Electrons 0.6-1.2 Bohr from a home nucleus (up: nucleus i, dn: nucleus i+1, cyclic)."""
    rng = np.random.default_rng(seed)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    geminal = ham.wavefunction_data.geminal_data

    def shell(count: int, offset: int) -> jax.Array:
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = rng.uniform(0.6, 1.2, size=(count, 1))
        home = positions[(np.arange(count) + offset) % len(positions)]
        return jnp.asarray(home + radius * direction, dtype=jnp.float64)

    return shell(int(geminal.num_electron_up), 0), shell(int(geminal.num_electron_dn), 1)


def _fourth_order(evaluate, h: float) -> float:
    return (-evaluate(2.0 * h) + 8.0 * evaluate(h) - 8.0 * evaluate(-h) + evaluate(-2.0 * h)) / (12.0 * h)


def _finite_differences(energy, ham, r_up, r_dn, RT, h: float):
    """Central FD of the measurement energy w.r.t. nuclei, electrons and strain."""
    identity = np.eye(3)
    n_atom = len(np.asarray(ham.structure_data.positions))
    fd_R = np.zeros((n_atom, 3))
    for atom in range(n_atom):
        for axis in range(3):
            fd_R[atom, axis] = _fourth_order(
                lambda d, atom=atom, axis=axis: float(energy(_rebuilt(ham, identity, (atom, axis, d)), r_up, r_dn, RT)), h
            )
    fd_r = []
    for spin in (0, 1):
        coords = (r_up, r_dn)[spin]
        grad = np.zeros(coords.shape)
        for electron in range(coords.shape[0]):
            for axis in range(3):

                def shifted(d, spin=spin, electron=electron, axis=axis):
                    moved = [r_up, r_dn]
                    moved[spin] = moved[spin].at[electron, axis].add(d)
                    return float(energy(ham, moved[0], moved[1], RT))

                grad[electron, axis] = _fourth_order(shifted, h)
        fd_r.append(grad)
    fd_eps = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):

            def strained(d, a=a, b=b):
                F = np.eye(3)
                F[a, b] += d
                return float(energy(_rebuilt(ham, F), r_up @ F.T, r_dn @ F.T, RT))

            fd_eps[a, b] = _fourth_order(strained, h)
    return fd_R, fd_r[0], fd_r[1], fd_eps


def _assert_measurement_fd(algorithm: str, case: str) -> None:
    stem, non_local_move, _source_id = _CASES[case]
    ham = _build_ham_jastrow(stem, "j12")
    r_up, r_dn = _shell_configuration(ham, seed=2026)
    RT = _rot(seed=7)
    grad_positions, grad_strain = _ANALYTIC[algorithm]
    dR, dr_up, dr_dn = (np.asarray(x) for x in grad_positions(ham, r_up, r_dn, RT, _ALAT, non_local_move=non_local_move))
    d_eps = np.asarray(grad_strain(ham, r_up, r_dn, RT, _ALAT, non_local_move=non_local_move))
    analytic = {"R": np.real(dR), "r_up": np.real(dr_up), "r_dn": np.real(dr_dn), "eps": np.real(d_eps)}
    for name, value in analytic.items():
        assert value.size and np.all(np.isfinite(value)), f"{case} {name}: analytic GFMC_{algorithm} output is not finite"
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    if non_local_move is not None:
        # The ECP non-local part must be live, or the ECP cases would not exercise it.
        _, _, v_nonlocal, _ = compute_ecp_non_local_parts_nearest_neighbors(
            ham.coulomb_potential_data, ham.wavefunction_data, r_up, r_dn, flag_determinant_only=False, RT=RT
        )
        assert not np.isclose(0.0, float(np.sum(np.asarray(v_nonlocal))), atol=atol, rtol=rtol), (
            f"{case}: ECP non-local energy is zero on this configuration"
        )
    if non_local_move == "dltmove":
        # The dltmove Jastrow-reweight term must change the measured energy, or dltmove == tmove.
        e_dlt = float(_measurement_energy(algorithm, "dltmove")(ham, r_up, r_dn, RT))
        e_t = float(_measurement_energy(algorithm, "tmove")(ham, r_up, r_dn, RT))
        assert not np.isclose(e_dlt, e_t, atol=atol, rtol=rtol), f"{case}: dltmove reweight term is not live"

    energy = _measurement_energy(algorithm, non_local_move)
    fd = dict(zip(("R", "r_up", "r_dn", "eps"), _finite_differences(energy, ham, r_up, r_dn, RT, _H), strict=True))
    fd_2h = dict(zip(("R", "r_up", "r_dn", "eps"), _finite_differences(energy, ham, r_up, r_dn, RT, 2.0 * _H), strict=True))
    for name in fd:
        assert np.all(np.isfinite(fd[name])) and np.all(np.isfinite(fd_2h[name])), f"{case} {name}: FD is not finite"
        # Step doubling checks stencil stability without claiming an error bound.
        np.testing.assert_allclose(
            fd[name],
            fd_2h[name],
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"{case} {name}: FD not converged in h (smooth regime)",
        )
        # Non-vacuity: every compared component is larger than the tolerance, so an analytic
        # component set to zero would fail the comparison below.
        vacuous = np.isclose(np.zeros_like(fd[name]), fd[name], atol=atol, rtol=rtol)
        assert not np.any(vacuous), f"{case} {name}: FD components {np.argwhere(vacuous).tolist()} are within tolerance of zero"
        np.testing.assert_allclose(
            analytic[name],
            fd[name],
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"{case} {name}: analytic GFMC_{algorithm} != FD",
        )


@pytest.mark.heavy
@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed retains existing family-level AD/identity guarantees.",
    alternative_guarantee_ref="MATH-DERIV-CONVENTIONAL-ALGORITHM-001",
)
@pytest.mark.parametrize("case", _CASE_IDS)
def test_gfmc_t_measurement_derivatives_match_fd(case):
    """Conventional GFMC measurement nuclear/electron/strain derivatives match rebuilt-input FD."""
    _assert_measurement_fd("t", case)
    jax.clear_caches()


@pytest.mark.heavy
@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed retains existing family-level AD/identity guarantees.",
    alternative_guarantee_ref="MATH-DERIV-LOAD-BALANCED-ALGORITHM-001",
)
@pytest.mark.parametrize("case", _CASE_IDS)
def test_gfmc_n_measurement_derivatives_match_fd(case):
    """Load-balanced GFMC measurement nuclear/electron/strain derivatives match rebuilt-input FD."""
    _assert_measurement_fd("n", case)
    jax.clear_caches()


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
