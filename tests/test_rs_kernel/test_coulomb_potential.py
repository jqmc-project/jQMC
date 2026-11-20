"""Layer-1 equivalence test: jqmc_rs (Rust) vs jqmc (JAX).

Covers the autograd-free / RNG-free *molecular* bare Coulomb value functions.
The JAX path is the reference; the Rust port must match it to fp64 tolerance.

The tests build their own synthetic Structure_data / Coulomb_potential_data (no
Hamiltonian fixture) and take their tolerances from the zone-derived helper.
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
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    compute_bare_coulomb_potential,
    compute_bare_coulomb_potential_el_el,
    compute_bare_coulomb_potential_el_ion_element_wise,
    compute_bare_coulomb_potential_ion_ion,
)
from jqmc.structure import Structure_data

_ZONES = ("coulomb",)


def _make_molecular_fixture(seed: int):
    """Molecular (pbc_flag=False) all-electron fixture: random LiH-like system.

    Returns (coulomb_data, structure_data, r_up, r_dn).
    """
    rng = np.random.default_rng(seed)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    structure = Structure_data.build(
        pbc_flag=False,
        positions=positions,
        atomic_numbers=(3, 1),  # Li, H
        element_symbols=("Li", "H"),
        atomic_labels=("Li1", "H1"),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    # 2 up + 2 dn electrons scattered around the two nuclei.
    r_up = (positions[rng.integers(0, 2, size=2)] + rng.normal(scale=0.8, size=(2, 3))).astype(np.float64)
    r_dn = (positions[rng.integers(0, 2, size=2)] + rng.normal(scale=0.8, size=(2, 3))).astype(np.float64)
    return coulomb, structure, r_up, r_dn


def _ref_arrays(coulomb):
    R_carts = np.asarray(coulomb.structure_data._positions_cart_jnp, dtype=np.float64)
    charges = np.asarray(coulomb._effective_charges, dtype=np.float64)
    return R_carts, charges


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_el_el_matches_jax(seed):
    """El el matches jax."""
    coulomb, _, r_up, r_dn = _make_molecular_fixture(seed)
    ref = float(compute_bare_coulomb_potential_el_el(r_up, r_dn))
    got = jqmc_rs.compute_bare_coulomb_potential_el_el(r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_ion_ion_matches_jax(seed):
    """Ion ion matches jax."""
    coulomb, _, _, _ = _make_molecular_fixture(seed)
    R_carts, charges = _ref_arrays(coulomb)
    ref = float(compute_bare_coulomb_potential_ion_ion(coulomb))
    got = jqmc_rs.compute_bare_coulomb_potential_ion_ion(R_carts, charges)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_el_ion_element_wise_matches_jax(seed):
    """El ion element wise matches jax."""
    coulomb, _, r_up, r_dn = _make_molecular_fixture(seed)
    R_carts, charges = _ref_arrays(coulomb)
    ref_up, ref_dn = compute_bare_coulomb_potential_el_ion_element_wise(coulomb, r_up, r_dn)
    got_up, got_dn = jqmc_rs.compute_bare_coulomb_potential_el_ion_element_wise(R_carts, charges, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.allclose(got_up, np.asarray(ref_up), rtol=rtol, atol=atol), (got_up, ref_up)
    assert np.allclose(got_dn, np.asarray(ref_dn), rtol=rtol, atol=atol), (got_dn, ref_dn)


@pytest.mark.light
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_composite_matches_jax(seed):
    """Composite matches jax."""
    coulomb, _, r_up, r_dn = _make_molecular_fixture(seed)
    R_carts, charges = _ref_arrays(coulomb)
    ref = float(compute_bare_coulomb_potential(coulomb, r_up, r_dn))
    got = jqmc_rs.compute_bare_coulomb_potential(R_carts, charges, r_up, r_dn)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.allclose(got, ref, rtol=rtol, atol=atol), (got, ref)
