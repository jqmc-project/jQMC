"""Layer-1 equivalence test: jqmc_rs Jastrow vs jqmc JAX.

Covers the autograd-free molecular Jastrow value functions J1/J2 (via from_py
of the real jqmc dataclasses) and the J3 bilinear algebra (AO matrices supplied
by jqmc). NN-J3 / DTN-NN excluded.

Also exercises the StructureData from_py marshalling round-trip.

The tests build their own synthetic Structure_data / AOs_sphe_gto_data (no h5
fixture); tolerances come from the zone-derived helper.
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

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT
from jqmc.atomic_orbital import AOs_sphe_gto_data, compute_AOs
from jqmc.jastrow_factor import (
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    compute_Jastrow_one_body,
    compute_Jastrow_three_body,
    compute_Jastrow_two_body,
)
from jqmc.structure import Structure_data

_ZONES_J12 = ("jastrow_eval",)
# The J3 bilinear reference AO matrices are produced by jqmc's own ao_eval.
_ZONES_J3 = ("jastrow_eval", "ao_eval")


def _make_structure(R_carts, atomic_number):
    n = len(R_carts)
    return Structure_data(
        pbc_flag=False,
        positions=np.asarray(R_carts, dtype=np.float64),
        atomic_numbers=tuple([atomic_number] * n),
        element_symbols=tuple(["X"] * n),
        atomic_labels=tuple(["X"] * n),
    )


def _electrons(rng, n_up, n_dn):
    r_up = (2.0 * rng.random((n_up, 3)) - 1.0).astype(np.float64)
    r_dn = (2.0 * rng.random((n_dn, 3)) - 1.0).astype(np.float64)
    return r_up, r_dn


# --- from_py round-trip -------------------------------------------------


@pytest.mark.light
def test_structure_data_from_py_roundtrip():
    """Structure data from py roundtrip."""
    R = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5], [0.0, 0.0, 2.0]], dtype=np.float64)
    sd = _make_structure(R, atomic_number=6)
    positions, atomic_numbers, pbc_flag = jqmc_rs._structure_data_from_py_roundtrip(sd)
    assert np.allclose(positions, R, rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT)
    assert atomic_numbers == [6.0, 6.0, 6.0]
    assert pbc_flag is False


# --- J1 ---------------------------------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("j1b_type", ["exp", "pade"])
@pytest.mark.parametrize("seed", [0, 3, 17])
def test_jastrow_one_body_matches_jax(j1b_type, seed):
    """Jastrow one body matches jax."""
    rng = np.random.default_rng(seed)
    n_atom = 4
    R = (rng.random((n_atom, 3)) - 0.5) * 2.0
    structure = _make_structure(R, atomic_number=6)
    core_electrons = tuple([3.0] * n_atom)  # Z_eff = 3
    data = Jastrow_one_body_data(
        jastrow_1b_param=0.8,
        jastrow_1b_type=j1b_type,
        structure_data=structure,
        core_electrons=core_electrons,
    )
    r_up, r_dn = _electrons(rng, 5, 3)
    ref = float(compute_Jastrow_one_body(data, r_up, r_dn))
    got = jqmc_rs.compute_Jastrow_one_body(data, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_J12, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)


# --- J2 ---------------------------------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("j2b_type", ["pade", "exp"])
@pytest.mark.parametrize("seed", [1, 5, 23])
def test_jastrow_two_body_matches_jax(j2b_type, seed):
    """Jastrow two body matches jax."""
    rng = np.random.default_rng(seed)
    data = Jastrow_two_body_data(jastrow_2b_param=1.3, jastrow_2b_type=j2b_type)
    r_up, r_dn = _electrons(rng, 4, 4)
    ref = float(compute_Jastrow_two_body(data, r_up, r_dn))
    got = jqmc_rs.compute_Jastrow_two_body(data, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_J12, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)


# --- J3 bilinear (AO matrices from jqmc) ------------------------------------


def _make_aos_data(R_carts):
    """Minimal s+p sphe-GTO AO basis on each atom (matches test_jastrow style)."""
    n_atom = len(R_carts)
    structure = _make_structure(R_carts, atomic_number=1)
    nucleus_index, orbital_indices, exponents, coefficients = [], [], [], []
    angular_momentums, magnetic = [], []
    ao_idx = 0
    for a in range(n_atom):
        # one s shell (l=0), one p shell (l=1, m=-1,0,1); each (l,m) is one AO.
        for l in range(2):
            for m in range(-l, l + 1):
                nucleus_index.append(a)
                angular_momentums.append(l)
                magnetic.append(m)
                orbital_indices.append(ao_idx)  # 1 primitive per AO
                exponents.append(1.0 + 0.3 * l)
                coefficients.append(1.0)
                ao_idx += 1
    return AOs_sphe_gto_data._from_full_basis(
        exponents=np.array(exponents, dtype=np.float64),
        coefficients=np.array(coefficients, dtype=np.float64),
        structure_data=structure,
        nucleus_index=tuple(nucleus_index),
        num_ao=len(angular_momentums),
        num_ao_prim=len(exponents),
        orbital_indices=tuple(orbital_indices),
        angular_momentums=tuple(angular_momentums),
        magnetic_quantum_numbers=tuple(magnetic),
    )


@pytest.mark.light
@pytest.mark.parametrize("seed", [2, 11, 29])
def test_jastrow_three_body_value_matches_jax(seed):
    """Jastrow three body value matches jax."""
    rng = np.random.default_rng(seed)
    R = (rng.random((2, 3)) - 0.5) * 3.0
    aos_data = _make_aos_data(R)
    orb_num = aos_data.num_ao
    j_matrix = (rng.random((orb_num, orb_num + 1)) - 0.5).astype(np.float64)
    j3_data = Jastrow_three_body_data(orb_data=aos_data, j_matrix=j_matrix)

    r_up, r_dn = _electrons(rng, 4, 3)
    ref = float(compute_Jastrow_three_body(j3_data, r_up, r_dn))

    # AO matrices (orb_num, n_elec) supplied by jqmc; Rust ports only the bilinear algebra.
    aos_up = np.asarray(np.real(compute_AOs(aos_data, r_up)), dtype=np.float64)
    aos_dn = np.asarray(np.real(compute_AOs(aos_data, r_dn)), dtype=np.float64)
    got = jqmc_rs.compute_Jastrow_three_body_value(j_matrix, aos_up, aos_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_J3, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)
