"""Long-chain incremental-vs-fresh gate for the XLA determinant grad/lap streaming state.

The LRDMC projection advances ``Det_streaming_state`` once per
hop with ``_advance_grads_laplacian_ln_Det_streaming_state``; the per-electron
``grad ln|Det|`` / ``lap ln|Det|`` it carries feed the continuum kinetic term.
``_init_grads_laplacian_ln_Det_streaming_state`` is the from-scratch reference
(fresh all-electron orbital tables plus the full matrix products).

This module walks ``N_HOPS`` deterministic LRDMC-like hops (see
``tests/_det_gradlap_chain_fixtures.py``) and, after EVERY hop, compares every
field of the advanced state with a fresh initialisation at the same
configuration and with the same inverse: the orbital value/gradient/Laplacian
tables, the paired tables, the four geminal derivative tables and the four
outputs. The inverse is the exact inverse at each configuration, so the
comparison isolates the streaming tables from the Sherman-Morrison inverse
chain of the driver. Because the first electron of each multi-electron spin
channel never moves, its rows are never rewritten and any per-hop corruption
of a carried table accumulates over the whole chain.

The tables are compared under the ao_eval/det_grad_lap strict tier: the
advance evaluates the moved electron's orbitals in a different XLA program
than the fresh build, and XLA:GPU rounds the same fp32 ``ao_eval`` value
differently from program to program (issue #546). Each output is the exact contraction of its own tables with the
shared inverse, so it may differ from the fresh output by at most the image
of the actual table differences through that inverse plus the det_grad_lap
strict pair. In full precision both resolve to the det_grad_lap strict pair.

Cases cover the direct occupied-orbital SD and dense AGP, real and complex,
molecular and periodic, open shell (unpaired columns active). Full and mixed
are each judged within their own precision by these central pairs.
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

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc.determinant import (
    AGP_data,
    SD_data,
    _advance_grads_laplacian_ln_Det_streaming_state,
    _init_grads_laplacian_ln_Det_streaming_state,
    compute_geminal_all_elements,
)
from tests._det_gradlap_chain_fixtures import (
    N_HOPS,
    NEAR_SINGULAR_MIN_COND,
    SHORT_HOPS,
    build_case,
    hop_schedule,
    near_singular_schedule,
    start_configuration,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_TABLE_FIELDS = (
    "ao_up",
    "ao_dn",
    "ao_up_grads",
    "ao_dn_grads",
    "ao_up_lap",
    "ao_dn_lap",
    "paired_dn",
    "paired_dn_grads",
    "paired_dn_lap",
    "paired_up_lambda",
    "geminal_grad_up",
    "geminal_grad_dn",
    "geminal_lap_up",
    "geminal_lap_dn",
)
_OUTPUT_FIELDS = ("grad_ln_D_up", "grad_ln_D_dn", "lap_ln_D_up", "lap_ln_D_dn")
_TABLE_ZONES = ("ao_eval", "det_grad_lap")

# (case, expected geminal class, complex orbitals)
_CASES = {
    "sd_real_mol_open": (SD_data, False),
    "sd_real_pbc_open": (SD_data, False),
    "sd_complex_twist_open": (SD_data, True),
    "sd_complex_bspline_twist": (SD_data, True),
    "agp_real_mol_open": (AGP_data, False),
    "agp_mo_real_mol_open": (AGP_data, False),
    "agp_real_pbc_open": (AGP_data, False),
    "agp_complex_twist_open": (AGP_data, True),
    # Multi-electron B-spline (O-lsda, 4 up / 2 dn).
    "sd_real_bspline_open": (SD_data, False),
    "sd_complex_bspline_open": (SD_data, True),
}


@pytest.fixture
def _chain_tmp(tmp_path_factory):
    # Per case: a module-scoped directory let a later case load the
    # HDF5 an earlier case converted from the committed QE save, so its own
    # execution never reached that save and its receipt depended on test order.
    return str(tmp_path_factory.mktemp("det_gradlap_chain"))


def _inverse(geminal, r_up, r_dn):
    return jnp.linalg.inv(compute_geminal_all_elements(geminal, r_up, r_dn))


def _table_difference_image(diff, ref, g_inv, n_dn):
    """Elementwise bound of the output differences implied by the table differences ``diff``.

    The outputs are bilinear in the geminal tables and the shared inverse (no conjugation),
    so ``|d grad| <= |d table| . |G^-1|``; the Laplacian adds ``|d grad| (2 |grad| + |d grad|)``.
    """
    g_inv = np.abs(np.asarray(g_inv))
    grad_up = np.einsum("gij,ji->ig", np.abs(diff["geminal_grad_up"]), g_inv)
    grad_dn = np.einsum("ij,gji->ig", g_inv, np.abs(diff["geminal_grad_dn"]))[:n_dn]
    lap_up = np.einsum("ij,ji->i", np.abs(diff["geminal_lap_up"]), g_inv)
    lap_dn = np.einsum("ij,ji->i", g_inv, np.abs(diff["geminal_lap_dn"]))[:n_dn]
    lap_up = lap_up + np.sum(grad_up * (2.0 * np.abs(ref["grad_ln_D_up"]) + grad_up), axis=1)
    lap_dn = lap_dn + np.sum(grad_dn * (2.0 * np.abs(ref["grad_ln_D_dn"]) + grad_dn), axis=1)
    return {"grad_ln_D_up": grad_up, "grad_ln_D_dn": grad_dn, "lap_ln_D_up": lap_up, "lap_ln_D_dn": lap_dn}


def _assert_state_matches_fresh(state, fresh, g_inv, n_dn, label):
    atol, rtol = get_tolerance_min(_TABLE_ZONES, "strict")
    got = {name: np.asarray(getattr(state, name)) for name in _TABLE_FIELDS + _OUTPUT_FIELDS}
    ref = {name: np.asarray(getattr(fresh, name)) for name in _TABLE_FIELDS + _OUTPUT_FIELDS}
    for name in _TABLE_FIELDS + _OUTPUT_FIELDS:
        assert got[name].shape == ref[name].shape, f"{label}: {name} shape {got[name].shape} != {ref[name].shape}"
        assert np.all(np.isfinite(got[name])) and np.all(np.isfinite(ref[name])), f"{label}: non-finite {name}"
    for name in _TABLE_FIELDS:
        np.testing.assert_allclose(got[name], ref[name], atol=atol, rtol=rtol, err_msg=f"{label}: {name}")
    # Outputs: within the det_grad_lap strict pair plus the image of the actual table
    # differences through the shared inverse of the fresh outputs.
    s_atol, s_rtol = get_tolerance("det_grad_lap", "strict")
    image = _table_difference_image({name: got[name] - ref[name] for name in _TABLE_FIELDS}, ref, g_inv, n_dn)
    for name in _OUTPUT_FIELDS:
        excess = np.abs(got[name] - ref[name]) - (s_atol + s_rtol * np.abs(ref[name]) + image[name])
        assert np.all(excess <= 0.0), f"{label}: {name} exceeds the table-difference image by {np.max(excess):.3e}"


def _run_chain(ham, n_hops, corrupt=None, hops=None):
    """Advance the streaming state along the fixed hop schedule, checking every hop."""
    geminal = ham.wavefunction_data.geminal_data
    r_up, r_dn = start_configuration(ham)
    if hops is None:
        hops = hop_schedule(ham, r_up, r_dn, n_hops=n_hops)
    r_up = jnp.asarray(r_up)
    r_dn = jnp.asarray(r_dn)
    state = _init_grads_laplacian_ln_Det_streaming_state(geminal, r_up, r_dn, _inverse(geminal, r_up, r_dn))
    for k, (is_up, e, new) in enumerate(hops, start=1):
        if is_up:
            r_up = r_up.at[e].set(jnp.asarray(new))
        else:
            r_dn = r_dn.at[e].set(jnp.asarray(new))
        g_inv = _inverse(geminal, r_up, r_dn)
        state = _advance_grads_laplacian_ln_Det_streaming_state(
            geminal, state, jnp.bool_(is_up), jnp.int32(e), r_up, r_dn, g_inv
        )
        if corrupt is not None:
            state = corrupt(state)
        fresh = _init_grads_laplacian_ln_Det_streaming_state(geminal, r_up, r_dn, g_inv)
        _assert_state_matches_fresh(state, fresh, g_inv, int(geminal.num_electron_dn), f"hop {k}")
    return hops


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_CASES))
def test_det_gradlap_streaming_long_chain_matches_fresh(case, _chain_tmp):
    """After every one of N_HOPS hops the advanced state equals a fresh init."""
    ham = build_case(case, _chain_tmp)
    geminal = ham.wavefunction_data.geminal_data
    cls, is_complex = _CASES[case]
    assert type(geminal) is cls, f"{case}: geminal is {type(geminal).__name__}"
    assert bool(np.iscomplexobj(np.zeros((), dtype=geminal._determinant_dtype_jnp))) == is_complex
    hops = _run_chain(ham, N_HOPS)
    moved_up = {e for is_up, e, _ in hops if is_up}
    moved_dn = {e for is_up, e, _ in hops if not is_up}
    assert moved_up and moved_dn, f"{case}: the chain must move both spins"


def _frozen_row_drift(rtol):
    """A carried-table corruption that accumulates only on rows no hop rewrites.

    Each hop scales the whole carried ``geminal_lap_up`` table by ``1 + rtol / (4
    * SHORT_HOPS)``. Rows that a hop rewrites lose the corruption; the frozen up
    electron's row keeps all of it, so it stays within ``rtol / 4`` over
    ``SHORT_HOPS`` hops and exceeds ``rtol`` over ``N_HOPS``.
    """
    scale = 1.0 + rtol / (4 * SHORT_HOPS)

    def corrupt(state):
        return state.replace(geminal_lap_up=state.geminal_lap_up * scale)

    return corrupt


@pytest.mark.light
@pytest.mark.parametrize("case", ["sd_real_mol_open", "agp_real_mol_open"])
def test_det_gradlap_long_chain_rejects_accumulating_drift(case, _chain_tmp):
    """Negative control: a slow carried-table drift passes a short chain but not N_HOPS."""
    ham = build_case(case, _chain_tmp)
    _, rtol = get_tolerance_min(_TABLE_ZONES, "strict")
    corrupt = _frozen_row_drift(rtol)
    # The drift stays inside the table pair for SHORT_HOPS hops ...
    _run_chain(ham, SHORT_HOPS, corrupt=corrupt)
    # ... and the same comparison rejects it over the long chain.
    with pytest.raises(AssertionError, match="geminal_lap_up|lap_ln_D_up"):
        _run_chain(ham, N_HOPS, corrupt=corrupt)


# Cases with two or more up electrons (the excursion moves up electron 1 onto 0).
_NEAR_SINGULAR_CASES = sorted(case for case in _CASES if case != "sd_complex_bspline_twist")


@pytest.mark.light
@pytest.mark.parametrize("case", _NEAR_SINGULAR_CASES)
def test_det_gradlap_streaming_near_singular_excursions_match_fresh(case, _chain_tmp):
    """The advanced state equals a fresh init through near-coincident excursions.

    The walk of :func:`near_singular_schedule` repeatedly visits a geminal with
    ``cond(G) >= NEAR_SINGULAR_MIN_COND`` (a same-spin pair ``1e-5`` bohr apart,
    the near-coincident-pair construction) and leaves it again. Every hop compares every field
    of the advanced state with ``_init_grads_laplacian_ln_Det_streaming_state`` at
    the same configuration and with the same exact inverse, so the row/column
    advance of the moved electron is checked where the derivative tables and the
    contraction are most ill-conditioned. The pairs of ``_assert_state_matches_fresh``,
    each precision mode within itself.
    """
    ham = build_case(case, _chain_tmp)
    r_up, r_dn = start_configuration(ham)
    hops, cond_max = near_singular_schedule(ham, r_up, r_dn)
    assert cond_max >= NEAR_SINGULAR_MIN_COND, f"{case}: excursion cond(G)={cond_max:.3g} is not near-singular"
    _run_chain(ham, len(hops), hops=hops)
