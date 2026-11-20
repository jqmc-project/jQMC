"""A suppressed GFMC hop must leave every streaming cache unchanged (issue #784 B).

In a GFMC_t sub-step whose ``tau_left`` is exhausted no electron moves, yet the
streaming wrappers still advance the kinetic grad/lap state and the kinetic and
ECP candidate meshes, and the walker's later (masked) sub-steps read them for
the measured local energy. A mover named
``(moved_spin_is_up=has_up, moved_index=where(has_up, up_idx, dn_idx))`` is
``(False, 0)`` on a suppressed hop. With an empty dn channel that is global
electron ``num_up``, one past the end. The DTN candidate advance would clamp
its gathers onto the last up electron but keep the out-of-range index in the
``_dtn_candidate_delta_rows`` self mask (and the u-channel spin parity), so it
would rebuild that electron's row WITH its own pair (WF-PATTERN-008: up to 1.9e-2 in
``log_delta``, visible in the GFMC_t pure-leaf e_L).

Detectors (light, full and mixed):

* ``test_streaming_mover_names_an_existing_electron``: the wrappers' mover rule
  ``jqmc_gfmc._streaming_mover`` names an electron of a non-empty channel for
  fully polarized up-only / dn-only and two-spin shapes, on real and on
  suppressed hops.
* ``test_suppressed_hop_advance_is_a_no_op``: every leaf of the kinetic
  grad/lap state, the kinetic candidate mesh and (with an ECP) the ECP
  candidate mesh, advanced at UNCHANGED positions, equals its fresh build to
  the strict zone round-off. Both the legacy mover ``(False, 0)`` (what the
  advances can still be handed; it must be harmless) and the wrapper's
  ``_streaming_mover`` output are exercised, on the fully polarized up-only
  fixtures WF-PATTERN-003 (all-electron, J1/J2/J3, DTN u) and WF-PATTERN-008
  (ccECP, J2/J3, DTN chi+f) and the two-spin control WF-PATTERN-012 (ccECP,
  DTN u).
* ``test_suppressed_hop_jastrow_candidate_advance_is_a_no_op_dn_only``: no
  canonical fixture is dn-only, so the four Jastrow candidate-mesh advances
  (J1, J2, J3, DTN) of WF-PATTERN-003's Jastrow are run on a dn-only electron
  set (up channel empty) with both movers.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import coulomb_potential as coulomb
from jqmc import jastrow_factor as jf
from jqmc import wavefunction
from jqmc._precision import get_tolerance_min
from jqmc.determinant import compute_geminal_all_elements
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_ZONES = ("jastrow_ratio", "jastrow_grad_lap", "det_ratio", "det_grad_lap", "wf_kinetic", "coulomb")
_ALAT = 0.3
_MOVERS = ("legacy-dn0", "streaming-mover")


def _positions(ham, seed):
    """Electrons close to the nuclei so every ECP / DTN cutoff channel is live."""
    geminal = ham.wavefunction_data.geminal_data
    n_up, n_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    atoms = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = atoms[rng.integers(0, len(atoms), size=n_up + n_dn)]
    r = centers + rng.normal(scale=0.6, size=(n_up + n_dn, 3))
    return jnp.asarray(r[:n_up]), jnp.asarray(r[n_up:])


def _mover(kind, n_up, n_dn):
    """The (moved_spin_is_up, moved_index) a suppressed hop hands to the advances."""
    has_up, up_idx, dn_idx = jnp.asarray(False), jnp.asarray(0), jnp.asarray(0)
    if kind == "legacy-dn0":
        return has_up, jnp.where(has_up, up_idx, dn_idx)
    from jqmc.jqmc_gfmc import _streaming_mover

    return _streaming_mover(has_up, up_idx, dn_idx, n_up, n_dn)


def _assert_tree_unchanged(advanced, fresh, label):
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    adv_leaves, adv_def = jax.tree_util.tree_flatten(advanced)
    ref_leaves, ref_def = jax.tree_util.tree_flatten(fresh)
    assert adv_def == ref_def, label
    assert ref_leaves, label
    for k, (actual, reference) in enumerate(zip(adv_leaves, ref_leaves, strict=True)):
        actual, reference = np.asarray(actual), np.asarray(reference)
        assert actual.shape == reference.shape, (label, k)
        if not np.issubdtype(reference.dtype, np.inexact):
            np.testing.assert_array_equal(actual, reference, err_msg=f"{label} leaf {k}")
            continue
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=f"{label} leaf {k}")


@pytest.mark.parametrize(
    "n_up,n_dn,has_up,up_idx,dn_idx",
    [
        (3, 0, False, 0, 0),
        (3, 0, True, 2, 0),
        (0, 3, False, 0, 0),
        (0, 3, False, 0, 2),
        (2, 2, False, 0, 0),
        (2, 2, True, 1, 0),
        (2, 2, False, 0, 1),
    ],
)
def test_streaming_mover_names_an_existing_electron(n_up, n_dn, has_up, up_idx, dn_idx):
    """The mover always lies in a non-empty spin channel, and keeps every real hop."""
    from jqmc.jqmc_gfmc import _streaming_mover

    spin, index = _streaming_mover(jnp.asarray(has_up), jnp.asarray(up_idx), jnp.asarray(dn_idx), n_up, n_dn)
    spin, index = bool(spin), int(index)
    assert 0 <= index < (n_up if spin else n_dn)
    if n_up and n_dn:
        assert (spin, index) == (has_up, up_idx if has_up else dn_idx)


@pytest.mark.parametrize("mover", _MOVERS)
@pytest.mark.parametrize("pattern", ["WF-PATTERN-003", "WF-PATTERN-008", "WF-PATTERN-012"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_suppressed_hop_advance_is_a_no_op(pattern, mover):
    """Advancing every streaming cache at unchanged positions reproduces its fresh build."""
    ham = build_canonical_hamiltonian(pattern)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    r_up, r_dn = _positions(ham, seed=11)
    n_up, n_dn = r_up.shape[0], r_dn.shape[0]
    spin, index = _mover(mover, n_up, n_dn)
    rotation = jnp.eye(3)
    inverse = jnp.linalg.inv(compute_geminal_all_elements(wf.geminal_data, r_up, r_dn))

    kinetic = wavefunction._init_kinetic_energy_all_elements_streaming_state(wf, r_up, r_dn, inverse)
    advanced = wavefunction._advance_kinetic_energy_all_elements_streaming_state(
        wavefunction_data=wf,
        state=kinetic,
        moved_spin_is_up=spin,
        moved_index=index,
        r_up_carts_new=r_up,
        r_dn_carts_new=r_dn,
        A_new_inv=inverse,
    )
    _assert_tree_unchanged(advanced, kinetic, f"{pattern} {mover} kinetic grad/lap state")

    mesh = wavefunction._init_discretized_kinetic_mesh_streaming_state(_ALAT, wf, r_up, r_dn, rotation)
    advanced = wavefunction._advance_discretized_kinetic_mesh_streaming_state(
        wavefunction_data=wf,
        state=mesh,
        moved_spin_is_up=spin,
        moved_index=index,
        r_up_carts_old=r_up,
        r_dn_carts_old=r_dn,
        r_up_carts_new=r_up,
        r_dn_carts_new=r_dn,
    )
    _assert_tree_unchanged(advanced, mesh, f"{pattern} {mover} kinetic candidate mesh")

    if cp.ecp_flag:
        ecp = coulomb._init_ecp_candidate_mesh_streaming_state(cp, wf, r_up, r_dn, rotation)
        advanced = coulomb._advance_ecp_candidate_mesh_streaming_state(
            coulomb_potential_data=cp,
            wavefunction_data=wf,
            state=ecp,
            moved_spin_is_up=spin,
            moved_index=index,
            r_up_carts_old=r_up,
            r_dn_carts_old=r_dn,
            r_up_carts_new=r_up,
            r_dn_carts_new=r_dn,
            RT=rotation,
        )
        _assert_tree_unchanged(advanced, ecp, f"{pattern} {mover} ECP candidate mesh")


@pytest.mark.parametrize("mover", _MOVERS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_suppressed_hop_jastrow_candidate_advance_is_a_no_op_dn_only(mover):
    """Dn-only electron set (empty up channel): the Jastrow candidate advances stay no-ops."""
    ham = build_canonical_hamiltonian("WF-PATTERN-003")
    jd = ham.wavefunction_data.jastrow_data
    assert None not in (
        jd.jastrow_one_body_data,
        jd.jastrow_two_body_data,
        jd.jastrow_three_body_data,
        jd.jastrow_dtn_data,
    )
    up, dn = _positions(ham, seed=12)
    r_up, r_dn = jnp.zeros((0, 3), dtype=up.dtype), jnp.concatenate([up, dn], axis=0)
    spin, index = _mover(mover, 0, r_dn.shape[0])
    shifts = _ALAT * jnp.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=up.dtype)
    cand_up = r_up[:, None, :] + shifts[None]
    cand_dn = r_dn[:, None, :] + shifts[None]

    j1 = jf._init_j1_candidate_mesh_state(jd.jastrow_one_body_data, r_up, r_dn, cand_up, cand_dn)
    adv = jf._advance_j1_candidate_mesh_state(jd.jastrow_one_body_data, j1, spin, index, r_up, r_dn, cand_up, cand_dn)
    _assert_tree_unchanged(adv, j1, f"dn-only {mover} J1 candidate mesh")

    j2 = jf._init_j2_candidate_mesh_state(jd.jastrow_two_body_data, r_up, r_dn, cand_up, cand_dn)
    adv = jf._advance_j2_candidate_mesh_state(
        jd.jastrow_two_body_data, j2, spin, index, r_up, r_dn, r_up, r_dn, cand_up, cand_dn
    )
    _assert_tree_unchanged(adv, j2, f"dn-only {mover} J2 candidate mesh")

    j3 = jf._init_j3_candidate_mesh_state(jd.jastrow_three_body_data, cand_up, cand_dn)
    adv = jf._advance_j3_candidate_mesh_state(jd.jastrow_three_body_data, j3, spin, index, cand_up, cand_dn)
    _assert_tree_unchanged(adv, j3, f"dn-only {mover} J3 candidate mesh")

    dtn = jf._init_j_dtn_candidate_mesh_state(jd.jastrow_dtn_data, r_up, r_dn, cand_up, cand_dn)
    adv = jf._advance_j_dtn_candidate_mesh_state(
        jd.jastrow_dtn_data, dtn, spin, index, r_up, r_dn, r_up, r_dn, cand_up, cand_dn
    )
    _assert_tree_unchanged(adv, dtn, f"dn-only {mover} DTN candidate mesh")
