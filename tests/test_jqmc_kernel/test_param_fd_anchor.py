"""Finite-difference anchors of the production e_L and ln|Psi| along variational-parameter (c-axis) blocks.

Analytic side = what the optimizer consumes: the production bridge closures
``bridge.batched_grad_ln_Psi(wrt="c", mask)`` / ``bridge.batched_grad_e_L(wrt="c", mask)`` with the
configured default (analytic) route, reduced to the flat block-name dict by
``bridge._reduce_c_param_grads`` (and ``_project_real_param_grads`` for e_L, as the MCMC driver does),
flattened in ``get_variational_blocks`` order.

Finite-difference side = the production primal forward (``evaluate_ln_wavefunction`` and
``compute_local_energy`` with a fixed non-identity ECP quadrature rotation) of a wavefunction whose
parameter is actually replaced by ``Wavefunction_data.apply_block_updates`` (the optimizer's update
path).  The realized block values c(t) are read back from ``get_variational_blocks`` of the updated
wavefunction and their tangent c'(0) is taken with the same stencil; the analytic prediction
contracts the analytic block gradient with c'(0).  Owner-enforced ties (symmetric J3 / lambda) and
retractions (S-metric polar retraction of MO coefficients) are therefore honoured by construction,
and the check fails if the update path does not move the probed parameter.

Stencil: fourth-order central difference with step ``_H``; the same stencil at ``2 _H`` must agree
within the comparison tolerance. This is a step-doubling consistency check,
not a rigorous truncation-error bound.
Tolerance: ``ATOL_FD_VS_ANALYTIC_TIGHT`` / ``RTOL_FD_VS_ANALYTIC_TIGHT`` (TOL-FD-AUTODIFF-FULL-001).
Probes: per block, the two components with the largest |analytic dE_L/dc| and the one with the
largest |analytic dln|Psi|/dc| (deduplicated). At least one probed FD value per
block must be non-zero beyond the comparison tolerance (non-vacuity), for e_L
and for ln|Psi| separately across the two configurations.

Fixtures (all real, two electron configurations each, electrons 0.6-1.2 Bohr from a home nucleus):

* ``n2-ecp-sd-j123``: N2 ccECP cart GTO, SD geminal, J1+J2+J3(cart GTO) -- live non-local ECP.
* ``n2-ecp-agp-jdtn``: N2 ccECP, dense AO AGP (lambda_matrix), J1+J2+DTN (u/chi/f).
* ``li-pbc-sd-j123``: bcc Li ccECP spherical pGTO (Gamma), SD geminal, J1+J2+J3(pGTO).
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.derivatives import bridge
from jqmc.hamiltonians import compute_local_energy
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_H = 5.0e-4

# case -> (stem, with_jastrow, spec, gem, mask)
_CASES = {
    "n2-ecp-sd-j123": (
        "N2_ecp_ccpvtz_cart.h5",
        True,
        {"j3": "cart_gto"},
        "SD",
        {"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": True, "opt_geminal": True},
    ),
    "n2-ecp-agp-jdtn": (
        "N2_ecp_ccpvtz_cart.h5",
        True,
        {"dtn": True},
        "AGP",
        {
            "opt_J1_param": True,
            "opt_J2_param": True,
            "opt_geminal": True,
            "opt_JDTN_u_param": True,
            "opt_JDTN_chi_param": True,
            "opt_JDTN_f_param": True,
        },
    ),
    "li-pbc-sd-j123": (
        "lithium.h5",
        True,
        {"j3": "pgto"},
        "SD",
        {"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": True, "opt_geminal": True},
    ),
}
_MASK_KEYS = (
    "opt_J1_param",
    "opt_J2_param",
    "opt_J3_param",
    "opt_geminal",
    "opt_J3_basis_exp",
    "opt_J3_basis_coeff",
    "opt_lambda_basis_exp",
    "opt_lambda_basis_coeff",
    "opt_JDTN_u_param",
    "opt_JDTN_chi_param",
    "opt_JDTN_f_param",
)

_ln = jax.jit(evaluate_ln_wavefunction)
_eL = jax.jit(compute_local_energy)


def _full_mask(mask):
    return {key: bool(mask.get(key, False)) for key in _MASK_KEYS}


def _rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q


def _shell_configuration(ham, seed: int):
    """Each electron 0.6-1.2 Bohr from a home nucleus (round-robin over atoms, cores excluded)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    gd = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(gd.num_electron_up), int(gd.num_electron_dn)
    positions = np.asarray(ham.structure_data._positions_cart_np)
    homes = [i % len(positions) for i in range(n_up + n_dn)]
    out = []
    for home in homes:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        out.append(positions[home] + rng.uniform(0.6, 1.2) * direction)
    out = np.asarray(out)
    return out[:n_up], out[n_up:]


def _flat(flat_map, blocks):
    return np.concatenate([np.asarray(flat_map[block.name]).reshape(block.size) for block in blocks])


def _block_values(wf, mask):
    blocks = wf.get_variational_blocks(state_token=("param-fd-anchor", 0), **mask)
    return np.concatenate([np.asarray(block.values).reshape(block.size) for block in blocks])


def _stencil(values_by_offset, h):
    fm2, fm1, fp1, fp2 = (values_by_offset[k] for k in (-2, -1, 1, 2))
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)


@pytest.mark.heavy
@pytest.mark.parametrize("case_id", sorted(_CASES))
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.skip_mixed_precision(
    reason="FD / numerical quadrature comparison dominated by float32 round-off",
    alternative_guarantee_ref="MATH-DERIVAPI-CONSISTENCY-001",
)
def test_c_axis_fd_anchor(case_id):
    """Parameter e_L and ln|Psi| derivatives on N2 SD/J123, N2 AO-AGP/DTN and Li SD/J123.

    Two fixed shell configurations (seeds 2026/2027), QR rotation seed 7, and
    fourth-order FD steps 5e-4 and 1e-3 exercise each active parameter block.
    Forward values and update-path tangents use actual apply_block_updates;
    the analytic bridge contractions are compared at the central full-only
    FD tolerance, with finite and per-block non-vacuity checks.

    Mixed has no FD verdict here. MATH-DERIVAPI-CONSISTENCY-001 supplies
    public analytical-versus-AD checks for e_L and ln|Psi|, not these same
    configurations or the finite-differenced optimizer update-path tangent.
    """
    assert not is_mixed_precision_enabled(), "FD verdict is full-precision only (run with --precision-mode=full)"
    stem, with_jastrow, spec, gem, mask = _CASES[case_id]
    mask = _full_mask(mask)
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
    wf = ham.wavefunction_data
    blocks = wf.get_variational_blocks(state_token=("param-fd-anchor", 0), **mask)
    names = [block.name for block in blocks]
    assert all(np.isrealobj(np.asarray(block.values)) for block in blocks), "real fixtures only"
    offsets, start = {}, 0
    for block in blocks:
        offsets[block.name] = (start, start + block.size)
        start += block.size
    rt = _rotation(7)

    configurations = [_shell_configuration(ham, seed) for seed in (2026, 2027)]
    r_up_b = jnp.asarray(np.stack([c[0] for c in configurations]))
    r_dn_b = jnp.asarray(np.stack([c[1] for c in configurations]))
    rt_b = jnp.asarray(np.stack([rt, rt]))
    n = len(configurations)

    o_raw = bridge.batched_grad_ln_Psi(wrt="c", mask=mask)(ham, r_up_b, r_dn_b)
    o_map = bridge._reduce_c_param_grads(wf, o_raw, n, mask)
    de_raw = bridge.batched_grad_e_L(wrt="c", mask=mask)(ham, r_up_b, r_dn_b, rt_b)
    de_map = bridge._project_real_param_grads(wf, bridge._reduce_c_param_grads(wf, de_raw, n, mask), mask)
    assert set(o_map) >= set(names) and set(de_map) >= set(names), (sorted(o_map), names)

    if ham.coulomb_potential_data.ecp_flag:
        _, _, nonlocal_values, _ = compute_ecp_non_local_parts_nearest_neighbors(
            ham.coulomb_potential_data, wf, r_up_b[0], r_dn_b[0], jnp.asarray(rt)
        )
        assert np.max(np.abs(np.asarray(nonlocal_values))) > 1.0e-3, "non-local ECP term must be live"

    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    rows = []
    for w in range(n):
        o_flat = np.concatenate([np.asarray(o_map[name]).reshape(n, -1)[w] for name in names])
        de_flat = np.concatenate([np.asarray(de_map[name]).reshape(n, -1)[w] for name in names])
        assert np.all(np.isfinite(o_flat)) and np.all(np.isfinite(de_flat))
        for name in names:
            s, e = offsets[name]
            probes = list(np.argsort(np.abs(de_flat[s:e]))[::-1][:2]) + [int(np.argmax(np.abs(o_flat[s:e])))]
            for local in dict.fromkeys(int(p) for p in probes):
                k = s + local
                direction = np.zeros(start)
                direction[k] = 1.0
                ln_values, e_values, block_values = {}, {}, {}
                for step in (-4, -2, -1, 1, 2, 4):
                    new_wf = wf.apply_block_updates(blocks=blocks, thetas=step * _H * direction, learning_rate=1.0)
                    new_ham = ham.replace(wavefunction_data=new_wf)
                    block_values[step] = _block_values(new_wf, mask)
                    ln_values[step] = float(np.real(_ln(new_wf, r_up_b[w], r_dn_b[w])))
                    e_values[step] = float(np.real(_eL(new_ham, r_up_b[w], r_dn_b[w], jnp.asarray(rt))))
                # The update path c(t) (owner ties and retractions included) is differentiated with the
                # same stencil; its tangent at h and 2h must agree and must move the probed parameter.
                d = _stencil({1: block_values[1], 2: block_values[2], -1: block_values[-1], -2: block_values[-2]}, _H)
                d_2h = _stencil({1: block_values[2], 2: block_values[4], -1: block_values[-2], -2: block_values[-4]}, 2 * _H)
                np.testing.assert_allclose(d_2h, d, atol=atol, rtol=rtol, equal_nan=False)
                assert np.abs(d[k]) > 0.1, f"{name}[{local}]: update path did not move the parameter"
                predicted = {"ln_Psi": float(o_flat @ d), "e_L": float(de_flat @ d)}
                for quantity, values in (("ln_Psi", ln_values), ("e_L", e_values)):
                    fd_h = _stencil({1: values[1], 2: values[2], -1: values[-1], -2: values[-2]}, _H)
                    fd_2h = _stencil({1: values[2], 2: values[4], -1: values[-2], -2: values[-4]}, 2 * _H)
                    rows.append((case_id, w, name, local, quantity, fd_h, fd_2h, predicted[quantity]))
                    assert np.isfinite(fd_h) and np.isfinite(fd_2h) and np.isfinite(predicted[quantity])
                    label = f"{case_id} cfg{w} {name}[{local}] {quantity}"
                    np.testing.assert_allclose(fd_2h, fd_h, atol=atol, rtol=rtol, equal_nan=False, err_msg=label + " h/2h")
                    np.testing.assert_allclose(
                        predicted[quantity], fd_h, atol=atol, rtol=rtol, equal_nan=False, err_msg=label + " analytic vs FD"
                    )
    for quantity in ("ln_Psi", "e_L"):
        for name in names:
            live = [abs(r[5]) > atol + rtol * abs(r[5]) for r in rows if r[4] == quantity and r[2] == name]
            assert any(live), f"{case_id} {name} {quantity}: every probed FD value is within tolerance of zero"
    for row in rows:
        print("FDROW", *row)
    jax.clear_caches()
