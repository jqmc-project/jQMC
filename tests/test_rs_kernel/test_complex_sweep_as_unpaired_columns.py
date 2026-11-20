"""Rust complex MCMC sweep: the AS factor of an up-move proposal with live unpaired columns.

With the Attaccalite-Sorella guide on (``epsilon_as > 0``) the native complex sweep
(``mcmc::mcmc_sweep_complex``, the core of the production complex per-step sweep and energy chunk)
rebuilds the moved row of the proposed geminal with ``complex_det::compute_up_row_complex`` and
uses ``R_AS(G_p)`` in the Metropolis acceptance through the guide ratio
``g(eps) = (max(R_AS_p, eps) / R_AS_p) / (max(R_AS_o, eps) / R_AS_o)``. In an open-shell geminal
the trailing ``N_up - N_dn`` entries of that row are the unpaired orbitals,
``sum_a ao_new[a] * lambda[a, orb_num_dn + j]``. Every earlier complex test with ``epsilon_as > 0``
used a closed-shell geminal, and the only open-shell complex sweep test runs with
``epsilon_as = 0``, so replacing ``orb_num_dn + j`` by ``j`` survived every test in both precision
modes (issue #576).

The sweep reports only accept / reject, so this module reads the native acceptance ratio back as a
continuous number from the native decision itself. For one up-move proposal from a fixed state and
Threefry key, the injected normal ``t`` sets the step length only; the moved electron, the axis and
the Metropolis draw ``b`` are fixed by the key. The native acceptance ratio
``h(t) = |g(eps) * J_ratio * det_ratio|^2 * T_ratio`` is continuous in ``t`` (``h(0) = 1``), and the
native decision is accept iff ``b < min(1, h(t))``. Scanning ``t`` outward from zero until the native
decision changes and bisecting that bracket for a fixed number of halvings locates a native
threshold ``t*`` with ``h_native(t*) = b`` to rounding. The dense reference must give
``h_ref(t*) = b`` in the strict Rust/XLA tier. Moving an electron away from the nuclei shrinks its
geminal row, so at these thresholds the smallest row / column norm of the proposed geminal is
typically the rebuilt row itself and ``R_AS_p`` reads its unpaired entries.

The reference shares no code with the native row builder: the same Threefry draws, the proposed
geminal from its defining formula over the production AO values (``reference_geminal``),
``R_AS`` from that matrix's row / column norms and singular values (no inverse), the determinant
ratio from ``numpy.linalg.det``, the Jastrow ratio from the XLA ratio
``_compute_ratio_Jastrow_part_rank1_update`` (``jastrow_ratio`` zone, float64 in both modes; the
float32 ``jastrow_eval`` of two absolute ``compute_Jastrow_part`` values would put a reference-side
rounding of ``|J| * 2^-24`` into the comparison in mixed precision) and the Kato transition ratio
from an explicit lattice-image search.

Cases: the reviewed open-shell twist geminals ``agp_open`` and ``agpn_open`` of
``_twist_geminal_cases`` (committed all-electron ``H4_pbc_ae_k025_cart.h5``, k_frac = (0.25, 0, 0),
three up and one down electron, two complex unpaired columns, J1+J2), at seeded-random positions
near the nuclei. ``epsilon_as = 1e30`` keeps the guide fully active (``g = R_AS_o / R_AS_p``).
Non-vacuity: every case measures at least four thresholds; at least one has a guide factor
``R_AS_o / R_AS_p`` beyond 1.5 (or below 1 / 1.5), and at least one has the smallest row / column
norm of the proposed geminal at the moved row or an unpaired column, so ``R_AS_p`` depends on the
unpaired entries of the rebuilt row.
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
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.jastrow_factor import _compute_ratio_Jastrow_part_rank1_update
from tests._mcmc_kato_reference import nearest_charge_distance
from tests.test_jqmc_kernel._twist_geminal_cases import OPEN_SHELL_CASES, build_case, reference_geminal

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

_DT = 1.2
_SEED = 3
#: The guide is fully active: max(R_AS, eps) = eps for both states, so g = R_AS_o / R_AS_p.
_EPSILON_AS = 1.0e30
#: Width (bohr) of the Gaussian electron clouds around the nuclei of each start configuration.
_START_WIDTHS = (0.5, 1.0)
#: Threefry keys tried per start; only up moves are measured.
_KEYS_PER_START = 4
_MIN_THRESHOLDS = 4
#: Non-vacuity floor: at least one measured threshold has a guide factor beyond this ratio.
_MIN_GUIDE_FACTOR = 1.5
#: Step-length scan (injected normal, both signs) used to bracket the first native decision change.
_SCAN = np.geomspace(0.02, 8.0, 48)
#: Fixed bisection length; the bracket shrinks below double rounding.
_BISECTION_STEPS = 64
#: Zones crossed by the geminal, its AS factor and the acceptance ratio.
_ZONES = ("ao_eval", "det_eval", "det_ratio", "jastrow_ratio")


def _as_factor(matrix):
    """``(S * ||G^-1||_F^2)^(-3/8)`` from row / column norms and singular values (no inverse)."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert np.all(singular_values > 0.0)
    squared = np.abs(matrix) ** 2
    minimum_norm = min(np.min(squared.sum(axis=0)), np.min(squared.sum(axis=1)))
    return float((minimum_norm * np.sum(1.0 / singular_values**2)) ** (-3.0 / 8.0))


def _minimum_norm_reads_rebuilt_row(matrix, moved_row, num_paired_columns):
    """True when the smallest row / column norm is the moved row or an unpaired column."""
    squared = np.abs(matrix) ** 2
    rows, columns = squared.sum(axis=1), squared.sum(axis=0)
    if np.min(rows) <= np.min(columns):
        return int(np.argmin(rows)) == moved_row
    return int(np.argmin(columns)) >= num_paired_columns


def _kato_width(nuclei, charges, cell, r):
    z, distance = nearest_charge_distance(nuclei, charges, r, cell)
    return 1.0 / z**2 * (1.0 + z**2 * distance) / (1.0 + distance)


def _reference_proposal(wavefunction, nuclei, charges, cell, r_up, r_dn, key, normal):
    """One proposal of the native sweep replayed from dense quantities."""
    num_up, num_dn = len(r_up), len(r_dn)
    key, subkey = random.split(key)
    is_up = int(random.randint(subkey, (), 0, num_up + num_dn)) < num_up
    key, subkey = random.split(key)
    up_index = int(random.randint(subkey, (), 0, num_up))
    key, subkey = random.split(key)
    dn_index = int(random.randint(subkey, (), 0, num_dn))
    index = up_index if is_up else dn_index
    old = (r_up if is_up else r_dn)[index]
    width_old = _kato_width(nuclei, charges, cell, old)
    key, subkey = random.split(key)
    step = float(normal) * width_old * _DT
    key, subkey = random.split(key)
    axis = int(random.randint(subkey, (), 0, 3))
    new = old.copy()
    new[axis] += step
    width_new = _kato_width(nuclei, charges, cell, new)
    distance2 = float(np.sum((new - old) ** 2))
    t_ratio = (width_old / width_new) * np.exp(
        -distance2 * (1.0 / (2.0 * width_new**2 * _DT**2) - 1.0 / (2.0 * width_old**2 * _DT**2))
    )
    new_up, new_dn = r_up.copy(), r_dn.copy()
    (new_up if is_up else new_dn)[index] = new
    g_old = np.asarray(reference_geminal(wavefunction, r_up, r_dn))
    g_new = np.asarray(reference_geminal(wavefunction, new_up, new_dn))
    det_ratio = complex(np.linalg.det(g_new) / np.linalg.det(g_old))
    jastrow = wavefunction.jastrow_data
    jastrow_ratio = float(
        np.asarray(_compute_ratio_Jastrow_part_rank1_update(jastrow, r_up, r_dn, new_up[None], new_dn[None]))[0]
    )
    r_as_p, r_as_o = _as_factor(g_new), _as_factor(g_old)
    guide = (max(r_as_p, _EPSILON_AS) / r_as_p) / (max(r_as_o, _EPSILON_AS) / r_as_o)
    key, subkey = random.split(key)
    draw = float(random.uniform(subkey, (), dtype=jnp.float64))
    return {
        "is_up": is_up,
        "index": index,
        "g_new": g_new,
        "guide": guide,
        "ratio": abs(guide * jastrow_ratio * det_ratio) ** 2 * t_ratio,
        "draw": draw,
    }


@pytest.mark.light
@pytest.mark.parametrize("case", OPEN_SHELL_CASES)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_sweep_as_acceptance_threshold_unpaired_columns_matches_dense_reference(case):
    """At native accept / reject thresholds of open-shell up moves the dense acceptance ratio equals the draw."""
    ham = build_case(case)
    wavefunction = ham.wavefunction_data
    geminal = wavefunction.geminal_data
    jastrow = wavefunction.jastrow_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    assert num_dn > 0 and num_up - num_dn >= 2, (case, num_up, num_dn)
    assert not ham.coulomb_potential_data.ecp_flag
    nuclei = np.asarray(ham.structure_data.positions, dtype=np.float64)
    charges = np.asarray(ham.structure_data.atomic_numbers, dtype=np.float64)
    cell = np.asarray(ham.structure_data.cell, dtype=np.float64)
    n_max = int(ham.structure_data._n_max)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    rng = np.random.default_rng(_SEED)

    reference_ratios, draws, contexts, reads_rebuilt_row = [], [], [], 0
    for start_index, width in enumerate(_START_WIDTHS):
        sites = np.arange(num_up + num_dn) % len(nuclei)
        clouds = np.ascontiguousarray(nuclei[sites] + rng.normal(scale=width, size=(num_up + num_dn, 3)))
        r_up, r_dn = np.ascontiguousarray(clouds[:num_up]), np.ascontiguousarray(clouds[num_up:])
        g_old = np.asarray(reference_geminal(wavefunction, r_up, r_dn))
        assert np.iscomplexobj(g_old) and g_old.shape == (num_up, num_up)
        g_old_inv = np.linalg.inv(g_old)
        for key_index in range(_KEYS_PER_START):
            key = random.fold_in(random.PRNGKey(_SEED), start_index * _KEYS_PER_START + key_index)
            native_key = np.asarray(key, dtype=np.uint32)

            def native_accepts(normal, native_key=native_key, r_up=r_up, r_dn=r_dn, g_old=g_old, g_old_inv=g_old_inv):
                result = jqmc_rs._mcmc_sweep_complex(
                    geminal,
                    jastrow,
                    nuclei,
                    charges,
                    r_up,
                    r_dn,
                    g_old,
                    g_old_inv,
                    _DT,
                    _EPSILON_AS,
                    int(native_key[0]),
                    int(native_key[1]),
                    np.asarray([normal], dtype=np.float64),
                    cell,
                    n_max,
                )
                return int(result["accepted"]) == 1

            if not _reference_proposal(wavefunction, nuclei, charges, cell, r_up, r_dn, key, 0.0)["is_up"]:
                continue
            assert native_accepts(0.0), (case, width, key_index, "a zero step must be accepted")
            for sign in (1.0, -1.0):
                inside, outside = 0.0, None
                for normal in sign * _SCAN:
                    if not native_accepts(normal):
                        outside = normal
                        break
                    inside = normal
                if outside is None:
                    continue
                for _ in range(_BISECTION_STEPS):
                    middle = 0.5 * (inside + outside)
                    if native_accepts(middle):
                        inside = middle
                    else:
                        outside = middle
                p = _reference_proposal(wavefunction, nuclei, charges, cell, r_up, r_dn, key, 0.5 * (inside + outside))
                assert p["is_up"]
                reference_ratios.append(p["ratio"])
                draws.append(p["draw"])
                contexts.append((width, key_index, sign, inside, p["guide"]))
                reads_rebuilt_row += _minimum_norm_reads_rebuilt_row(p["g_new"], p["index"], num_dn)

    np.testing.assert_allclose(
        np.asarray(reference_ratios),
        np.asarray(draws),
        atol=atol,
        rtol=rtol,
        err_msg=f"dense acceptance ratio at the native accept / reject thresholds != Metropolis draw ({case}): {contexts}",
    )
    assert len(draws) >= _MIN_THRESHOLDS, (case, len(draws))
    assert reads_rebuilt_row >= 1, (case, "no measured R_AS_p depends on the rebuilt row's norm")
    # The guide must move the acceptance ratio at the measured thresholds (R_AS_p != R_AS_o).
    guides = np.asarray([context[-1] for context in contexts])
    assert np.max(np.abs(np.log(guides))) >= np.log(_MIN_GUIDE_FACTOR), (case, guides)
