"""Rust complex MCMC sweep with two or more unpaired up columns: carried G / G^-1 against a rebuild.

The up-move row delta of the native complex sweep (``complex_det.rs``,
``compute_v_up_move_complex``) builds the unpaired part of the geminal row from
``lambda_c_value(a, orb_num_dn + j)`` for every unpaired column ``j``. With at most one
unpaired column the index is always ``orb_num_dn``, and every complex sweep test used a
closed-shell twist, so replacing ``orb_num_dn + j`` by ``orb_num_dn`` survived every test in
full and mixed precision (issue #74).

This module runs ``jqmc_rs._mcmc_sweep_complex`` (the ``mcmc::mcmc_sweep_complex`` core of the
production complex energy chunk) on the committed all-electron twist ``H4_pbc_ae_k025_cart.h5``
(k_frac = (0.25, 0, 0), complex Bloch Cartesian pGTO) in three geminals with two or more
unpaired up columns:

* ``sd_up2_dn0``: the committed two complex MOs as a single determinant with two up and no down
  electrons (no Jastrow);
* ``sd_up3_dn0``: the same plus one fixed complex MO row, three up and no down electrons
  (no Jastrow);
* ``agp_open``: the reviewed open-shell twist AGP of ``_twist_geminal_cases`` (three up, one down,
  two complex unpaired AO columns, J1+J2).

Several sweeps are chained, each carrying the geminal, its Sherman-Morrison inverse and the PRNG
key. After every sweep the carried geminal must equal the geminal rebuilt from scratch at the
returned electron positions (``compute_geminal_all_elements``) and the carried inverse must equal
the numpy inverse of that rebuild, in the active precision policy. The reference uses no
Sherman-Morrison chain and no Rust code. Non-vacuity: every sweep accepts moves, every up electron
moves, and at least two unpaired columns are live.
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
from jax import random

import jqmc_rs
from jqmc._precision import get_tolerance_min
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.molecular_orbital import MOs_data
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_jqmc_kernel._twist_geminal_cases import STEM, build_case

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

_DT = 1.2
_NUM_SWEEPS = 4
_NUM_MOVES = 200
_SEED = 0
_EXTRA_ROW_SEED = 11
_EXTRA_ROW_SCALE = 0.3
#: Zones crossed by the carried geminal / inverse and by the from-scratch rebuild.
_ZONES = ("ao_eval", "det_eval", "det_ratio")


def _only_up_sd(num_up):
    """Single determinant with ``num_up`` complex up MOs and no down electrons (no Jastrow)."""
    ham = _build_ham_for_cell(STEM, False, None, "SD")
    sd = ham.wavefunction_data.geminal_data
    aos = sd.orb_data_up_spin.aos_data
    occupied = np.asarray(sd.orb_data_up_spin.mo_coefficients)
    assert np.iscomplexobj(occupied)
    rng = np.random.default_rng(_EXTRA_ROW_SEED)
    extra = rng.normal(size=(num_up, aos.num_ao)) + 1j * rng.normal(size=(num_up, aos.num_ao))
    coefficients = np.vstack([occupied, _EXTRA_ROW_SCALE * extra])[:num_up]
    mos = MOs_data.build(aos_data=aos, mo_coefficients=jnp.asarray(coefficients))
    geminal = SD_data.build(num_electron_up=num_up, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))


def _hamiltonian(case):
    if case == "sd_up2_dn0":
        return _only_up_sd(2)
    if case == "sd_up3_dn0":
        return _only_up_sd(3)
    return build_case(case)


@pytest.mark.light
@pytest.mark.parametrize("case", ["sd_up2_dn0", "sd_up3_dn0", "agp_open"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_sweep_unpaired_columns_carried_state_matches_rebuild(case):
    """Carried complex G and G^-1 equal the from-scratch geminal and its inverse after every sweep."""
    ham = _hamiltonian(case)
    geminal = ham.wavefunction_data.geminal_data
    jastrow = ham.wavefunction_data.jastrow_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    assert num_up - num_dn >= 2, (case, num_up, num_dn)
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    charges = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    assert not ham.coulomb_potential_data.ecp_flag
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    n_max = int(ham.structure_data._n_max)

    rng = np.random.default_rng(_SEED)
    r_up = np.ascontiguousarray(rng.random((num_up, 3)) @ cell, dtype=np.float64)
    r_dn = np.ascontiguousarray(rng.random((num_dn, 3)) @ cell, dtype=np.float64)
    carried = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
    assert np.iscomplexobj(carried)
    carried_inv = np.linalg.inv(carried)
    key = np.asarray(random.PRNGKey(_SEED), dtype=np.uint32)
    key0, key1 = int(key[0]), int(key[1])
    moved_up = np.zeros(num_up, dtype=bool)

    atol, rtol = get_tolerance_min(_ZONES, "medium")
    for sweep in range(_NUM_SWEEPS):
        injected = rng.normal(size=_NUM_MOVES).astype(np.float64)
        result = jqmc_rs._mcmc_sweep_complex(
            geminal, jastrow, nuclei, charges, r_up, r_dn, carried, carried_inv, _DT, 0.0, key0, key1, injected, cell, n_max
        )
        assert int(result["accepted"]) > 0, (case, sweep)
        new_up = np.asarray(result["r_up"], dtype=np.float64)
        moved_up |= np.any(new_up != r_up, axis=1)
        r_up, r_dn = new_up, np.asarray(result["r_dn"], dtype=np.float64)
        carried = np.asarray(result["geminal"])
        carried_inv = np.asarray(result["geminal_inv"])
        key0, key1 = (int(x) for x in np.asarray(result["final_key"]).ravel()[:2])

        rebuilt = np.asarray(compute_geminal_all_elements(geminal, r_up, r_dn))
        rebuilt_inv = np.linalg.inv(rebuilt)
        for name, value, reference in (("geminal", carried, rebuilt), ("geminal_inv", carried_inv, rebuilt_inv)):
            assert np.all(np.isfinite(value)), (case, sweep, name)
            np.testing.assert_allclose(
                value,
                reference,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
                err_msg=f"carried {name} != from-scratch rebuild after sweep {sweep} ({case})",
            )
    assert moved_up.all(), (case, moved_up)
