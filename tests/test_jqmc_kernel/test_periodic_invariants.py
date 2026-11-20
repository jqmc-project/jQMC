"""Oracle-free periodic invariant tests for the JAX-XLA analytic kernels.

This is the JAX mirror of ``tests/test_rs_kernel/test_periodic_invariants.py`` (read
that module's docstring for the full motivation): four recent periodic far-image bugs
(J3 raw-coordinate AO eval, DTN raw-displacement force, J1/J2 kinetic-force metric
borrowing the geminal's cell, truncated AO image-tiling) were invisible to
near-nucleus fixtures, RIGID all-electron lattice shifts, and the effectively-vacuous
lithium 1up/1dn e-e channel. The rs suite closes that class for the NATIVE kernels;
this module closes it for jqmc's JAX analytic kernels INDEPENDENTLY of Rust (no
``jqmc_rs`` import anywhere here), with the same two ORACLE-FREE invariants on the
same non-rigid, non-vacuous, far-from-home-cell configs:

  1. single-electron lattice-translation invariance -- translating ONE electron by an
     integer lattice vector, leaving every other electron fixed, must leave ln|Psi|
     (``evaluate_ln_wavefunction`` -- already the MODULUS, so at a k != Gamma twist
     the Bloch phase e^{i k.T} drops out and the same equality holds; the rs file
     needed a modulus-assembly workaround, here none is needed), e_L
     (``compute_local_energy_fast`` threading the same geminal_inv the MCMC sweep
     threads -- complex at twist, compared full-complex), the HF force
     (``assemble.compute_grad_e_L`` wrt R / r) and the Pulay force
     (``assemble.compute_grad_ln_Psi`` wrt R / r) unchanged.
  2. per-config translation sum rule -- sum_I dE_L/dR_I + sum_i dE_L/dr_i == 0 and
     sum_I dln|Psi|/dR_I + sum_i dln|Psi|/dr_i == 0 on ANY config, a from-scratch
     identity (no reference run) that a wrong per-term metric or a dropped image-sum
     piece generically breaks.

Fixtures, config recipe (image-offset anchors n in {0,2,5} lattice hops + Gaussian
jitter, one forced-close antiparallel pair), the cond(G) < 1e4 well-conditioning
guard, and the shift matrix (n in {1,3,6} x 3 directions x {up0, dn0}) are copied
verbatim from the rs module -- see its docstring for the rationale of each. The one
structural difference: the rs suite parametrizes each of the 18 shifts as its own
test row (native calls are cheap); here each pattern's 18-shift sweep runs inside a
SINGLE test so the JAX jit warm-up (the dominant cost, ~30-250 s per pattern for the
assemble grads) is paid once per pattern per xdist worker, not once per row.
Likewise the 3 sum-rule seeds share one test per pattern. Measured wall (fresh
process per pattern): sweep 60-315 s, sum rule 27-100 s, so the whole file at
``-n 8`` lands in a few minutes -- kept ``light``, mirroring the rs file.

Tolerances: translation invariance reuses ``ATOL/RTOL_FLOOR_MEDIUM``
(the same constants the rs file uses) -- the JAX floors measure FAR inside it (max
abs residual 2.5e-13 across all 6 patterns x 18 shifts x 8 quantities; the JAX
analytic path is the reference implementation and satisfies the invariant at fp64
round-off). The sum rule gets NEW JAX-specific constants
(``ATOL/RTOL_FLOOR_MEDIUM``, ``ATOL/RTOL_FLOOR_MEDIUM``): the rs
constants (2e-5 / 5e-6) absorb the native periodic-kinetic fp-amplification floor,
which the JAX side does not exhibit here -- measured HF sum max 4.2e-14, Pulay sum
max 4.8e-13 (6 patterns x 3 seeds), constants = measured + ~4x margin (see
jqmc/_setting.py). NO row violated any invariant, so there are no xfails.

Absolute anchor: the invariances above cannot
see an error that is uniform in the electron positions. The last two nodes compare
e_L of a Jastrow-free single-plane-wave determinant at k != Gamma on a triclinic
cell with an independent numpy value (its determinant kinetic energy, exactly
sum_occ |G + k|^2 / 2, evaluated per precision zone, plus a numpy Ewald sum)
and show that a 1% error in the Bloch momentum breaks that comparison.
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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jqmc._precision import get_tolerance_min
from jqmc._setting import (
    ATOL_FLOOR_MEDIUM,
    RS_PERIODIC_INVARIANCE_MIN_JASTROW_TWO_BODY,
    RTOL_FLOOR_MEDIUM,
)
from jqmc.derivatives import assemble
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.jastrow_factor import Jastrow_three_body_data, compute_Jastrow_two_body
from jqmc.jqmc_mcmc import _geminal_inv_batched
from jqmc.wavefunction import evaluate_ln_wavefunction, evaluate_wavefunction
from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham as _build_ham_generic

pytestmark = [
    pytest.mark.with_chdir,
    pytest.mark.light,
]

# --------------------------------------------------------------------------- #
# Fixtures: the SAME (stem, spec) matrix as the rs mirror (see its docstring for
# why H4-pbc is the primary -- the only committed periodic fixture with >= 2
# electrons PER SPIN -- and lithium sphe-ECP the ECP/J3 complement). Built once,
# cached; every test row of a pattern reuses the SAME ham (and hence the same
# jitted programs within an xdist worker).
# --------------------------------------------------------------------------- #
_REAL_PATTERNS = ("h4_j12", "h4_jdtn", "li_j123", "li_jdtn")
# ``triclinic_agp_twist_j12``: canonical WF-PATTERN-016, a
# triclinic cell at k=(0.25, 0.125, 0) with a spin-restricted complex spherical-pGTO AGP
# and J1+J2. Its J3 on a complex Bloch orbital is removed (that J3 is not lattice
# periodic, so it is not a subject of these invariants). It is the only pattern here
# whose cell matrix is not symmetric, which is what a transposed ``inv_cell`` needs.
_TWIST_PATTERNS = ("h4_twist_j12", "li_twist_jdtn", "triclinic_agp_twist_j12")
_ALL_PATTERNS = _REAL_PATTERNS + _TWIST_PATTERNS


def _rand_j3_jax(wf, seed=7, scale=0.15):
    """Random symmetric J3 coupling (verbatim ``test_rs_kernel.test_gfmc_forces._rand_j3``,
    duplicated locally so this JAX-only suite never imports a module that imports
    ``jqmc_rs`` -- the point of this mirror is Rust-independence).
    """
    j3 = wf.jastrow_data.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=scale, size=(n, n + 1))
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    assert isinstance(j3, Jastrow_three_body_data)
    return wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm)))


_CANONICAL_016_K = np.asarray((0.25, 0.125, 0.0))


def _canonical_016_without_bloch_j3():
    """WF-PATTERN-016 with its complex-Bloch-orbital J3 removed (see ``_TWIST_PATTERNS``)."""
    from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian

    ham = build_canonical_hamiltonian("WF-PATTERN-016")
    jas = ham.wavefunction_data.jastrow_data
    assert jas.jastrow_three_body_data is not None and jas.jastrow_two_body_data is not None
    cell = np.asarray(ham.structure_data.cell, float)
    assert not np.array_equal(cell, cell.T), "WF-PATTERN-016 must keep its asymmetric triclinic cell"
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jas.replace(jastrow_three_body_data=None)))


@functools.cache
def _pattern_ham(pattern: str):
    if pattern == "h4_j12":
        return _build_ham_generic("H4_pbc_ae_cart.h5", True, None)
    if pattern == "h4_jdtn":
        return _build_ham_generic("H4_pbc_ae_cart.h5", True, {"dtn": True})
    if pattern == "li_j123":
        ham = _build_ham_generic("lithium.h5", True, {"j3": "pgto"})
        return ham.replace(wavefunction_data=_rand_j3_jax(ham.wavefunction_data))
    if pattern == "li_jdtn":
        return _build_ham_generic("lithium.h5", True, {"dtn": True})
    if pattern == "h4_twist_j12":
        return _build_ham_generic("H4_pbc_ae_k025_cart.h5", True, None)
    if pattern == "li_twist_jdtn":
        return _build_ham_generic("lithium_k025.h5", True, {"dtn": True})
    if pattern == "triclinic_agp_twist_j12":
        return _canonical_016_without_bloch_j3()
    raise KeyError(f"unknown pattern {pattern!r}")


# --------------------------------------------------------------------------- #
# Image-offset config recipe + well-conditioning guard (verbatim from the rs
# mirror; the geminal matrix / condition number is computed with the JAX kernel
# so the guard itself is Rust-free).
# --------------------------------------------------------------------------- #
_N_HOPS = (0, 2, 5)
_HOP_DIRS = ("a1", "a2", "a3")
_COND_MAX = 1.0e4
_MAX_DRAW_TRIES = 300


def _lattice_vec(ham, direction: str):
    cell = np.asarray(ham.structure_data.cell, float)
    a1, a2, a3 = cell[0], cell[1], cell[2]
    return {"a1": a1, "a2": a2, "a3": a3, "a1+a2-a3": a1 + a2 - a3}[direction]


def _image_offset_anchor(ham, rng, jitter=0.6):
    """One electron's position: near a randomly chosen nucleus IMAGE (an atom plus
    n*T lattice hops, n in ``_N_HOPS``), Gaussian-jittered -- see the rs docstring.
    """
    R = np.asarray(ham.structure_data.positions, float)
    atom = R[rng.integers(0, len(R))]
    n_i = rng.choice(_N_HOPS)
    direction = _lattice_vec(ham, rng.choice(_HOP_DIRS))
    return atom + n_i * direction + rng.normal(scale=jitter, size=3)


def _draw_cfg(ham, seed: int):
    """Per-electron independent image-offset positions, with a forced OPPOSITE-spin
    close pair (``dn[0]`` near ``up[0]``) for a non-vacuous antiparallel e-e channel.
    """
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    ru = np.array([_image_offset_anchor(ham, rng) for _ in range(nu)])
    rd = np.array([_image_offset_anchor(ham, rng) for _ in range(nd)])
    rd[0] = ru[0] + rng.normal(scale=0.8, size=3)
    return ru, rd


def _draw_well_conditioned_cfg(ham, seed: int):
    """``_draw_cfg`` filtered to reject near-node (ill-conditioned geminal) draws.
    ``compute_geminal_all_elements`` handles the complex twist geminal natively, so
    (unlike the rs mirror) no real/complex branch is needed.
    """
    for k in range(_MAX_DRAW_TRIES):
        ru, rd = _draw_cfg(ham, seed * 100_000 + k)
        G = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, jnp.asarray(ru), jnp.asarray(rd)))
        if np.linalg.cond(G) < _COND_MAX:
            return ru, rd
    raise RuntimeError(f"no well-conditioned config found for seed={seed} after {_MAX_DRAW_TRIES} tries")


def _ginv(ham, ru, rd):
    """The sweep-shaped Sherman-Morrison geminal inverse (``_geminal_inv_batched``,
    the SAME entry ``test_rs_kernel/test_vmc_forces.py``'s ``_jax_ref`` threads), so
    every gradient below consumes production's reused inverse, never a fresh LU.
    """
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return ginv[0]


# --------------------------------------------------------------------------- #
# JAX aggregate evaluators (e_L, ln|Psi|, HF force, Pulay force).
# --------------------------------------------------------------------------- #
_RT = np.eye(3)  # identity mesh rotation, as test_vmc_forces._jax_ref uses


def _eL_jax(ham, ru, rd, ginv):
    """``compute_local_energy_fast`` threading the sweep geminal_inv (the production
    e_L path). Complex at a k != Gamma twist; compared full-complex by the caller.
    """
    return complex(np.asarray(compute_local_energy_fast(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(_RT), ginv)))


def _ln_psi_jax(ham, ru, rd):
    """``evaluate_ln_wavefunction`` = Jastrow part + ln|det| -- already the MODULUS
    (real also at twist, where the Bloch phase lives outside |Psi|), and it includes
    an active J3, so unlike the rs mirror no J3-skip / value-assembly is needed.
    """
    return float(np.real(np.asarray(evaluate_ln_wavefunction(ham.wavefunction_data, jnp.asarray(ru), jnp.asarray(rd)))))


def _hf_and_pulay_force(ham, ru, rd, ginv):
    """HF force (``assemble.compute_grad_e_L`` wrt R / r) + Pulay force
    (``assemble.compute_grad_ln_Psi`` wrt R / r), threading one consistent
    ``geminal_inv`` -- exactly ``test_rs_kernel/test_vmc_forces.py``'s ``_jax_ref``
    / ``_jax_ref_pulay`` helpers.
    """
    ru_j, rd_j = jnp.asarray(ru), jnp.asarray(rd)
    hfR = np.asarray(assemble.compute_grad_e_L(ham, ru_j, rd_j, _RT, wrt="R", geminal_inv=ginv))
    hfu, hfd = assemble.compute_grad_e_L(ham, ru_j, rd_j, _RT, wrt="r", geminal_inv=ginv)
    puR = np.asarray(assemble.compute_grad_ln_Psi(ham, ru_j, rd_j, wrt="R", geminal_inv=ginv))
    puu, pud = assemble.compute_grad_ln_Psi(ham, ru_j, rd_j, wrt="r", geminal_inv=ginv)
    return (hfR, np.asarray(hfu), np.asarray(hfd)), (puR, np.asarray(puu), np.asarray(pud))


# =========================================================================== #
# Sanity: the forced-close e-e pair is genuinely non-vacuous (same guard and the
# same measured-|J2| floor constant as the rs mirror -- identical recipe, identical
# fixtures, so the measured range [0.33, 2.52] carries over).
# =========================================================================== #
@pytest.mark.parametrize("pattern", _ALL_PATTERNS)
def test_nonvacuous_two_body_sanity(pattern):
    """Nonvacuous two body sanity."""
    ham = _pattern_ham(pattern)
    ru, rd = _draw_well_conditioned_cfg(ham, 0)
    j2_data = ham.wavefunction_data.jastrow_data.jastrow_two_body_data
    assert j2_data is not None, f"{pattern}: no two-body Jastrow -- non-vacuousness is not checkable"
    j2 = abs(float(np.real(np.asarray(compute_Jastrow_two_body(j2_data, jnp.asarray(ru), jnp.asarray(rd))))))
    assert j2 > RS_PERIODIC_INVARIANCE_MIN_JASTROW_TWO_BODY, (
        f"{pattern}: forced close e-e pair is vacuous (|J2|={j2:.4g}); config recipe regressed"
    )


@pytest.mark.parametrize("trexio_file", ("H4_pbc_ae_cart.h5", "H4_pbc_ae_k025_cart.h5"))
def test_periodic_wavefunction_is_fermion_antisymmetric_under_same_spin_exchange(trexio_file):
    """Same-spin exchange gives ``Psi -> -Psi`` and preserves complex ``e_L`` in PBC."""
    ham = _build_ham_generic(trexio_file, True, None)
    ru, rd = _draw_well_conditioned_cfg(ham, seed=33)
    exchanged_up = ru.copy()
    exchanged_up[[0, 1]] = exchanged_up[[1, 0]]

    psi = complex(np.asarray(evaluate_wavefunction(ham.wavefunction_data, jnp.asarray(ru), jnp.asarray(rd))))
    psi_exchanged = complex(
        np.asarray(evaluate_wavefunction(ham.wavefunction_data, jnp.asarray(exchanged_up), jnp.asarray(rd)))
    )
    assert abs(psi) > 1.0e-12
    np.testing.assert_allclose(psi_exchanged, -psi, atol=ATOL_FLOOR_MEDIUM * abs(psi), rtol=RTOL_FLOOR_MEDIUM)

    local_energy = complex(np.asarray(compute_local_energy_fast(ham, ru, rd, _RT, _ginv(ham, ru, rd))))
    exchanged_local_energy = complex(
        np.asarray(compute_local_energy_fast(ham, exchanged_up, rd, _RT, _ginv(ham, exchanged_up, rd)))
    )
    np.testing.assert_allclose(
        exchanged_local_energy,
        local_energy,
        atol=ATOL_FLOOR_MEDIUM,
        rtol=RTOL_FLOOR_MEDIUM,
    )


# =========================================================================== #
# Test 1: single-electron lattice-translation invariance. The 18-shift matrix
# (n x direction x electron) runs INSIDE one test per pattern -- the JAX jit
# warm-up dominates and is paid once per pattern this way (see module docstring);
# the per-shift assertion set is identical to the rs mirror's per-row body.
# =========================================================================== #
_SHIFT_N = (1, 3, 6)
_SHIFT_DIRS = ("a1", "a2", "a1+a2-a3")
_SHIFT_ELECS = (("up", 0), ("dn", 0))


@pytest.mark.parametrize("pattern", _ALL_PATTERNS)
def test_single_electron_lattice_translation_invariance(pattern):
    """Single electron lattice translation invariance."""
    ham = _pattern_ham(pattern)
    ru, rd = _draw_well_conditioned_cfg(ham, 0)
    # A far-cell translation is an exact physical identity, but both sides
    # independently traverse every fp32 production zone in mixed mode.  Use
    # the centrally governed propagated-error tier for that path; full mode
    # remains at its fp64 medium tier.
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"),
        "medium",
    )

    ginv1 = _ginv(ham, ru, rd)
    eL1 = _eL_jax(ham, ru, rd, ginv1)
    lp1 = _ln_psi_jax(ham, ru, rd)
    (hfR1, hfu1, hfd1), (puR1, puu1, pud1) = _hf_and_pulay_force(ham, ru, rd, ginv1)

    for n in _SHIFT_N:
        for direction in _SHIFT_DIRS:
            for elec in _SHIFT_ELECS:
                tag = f"{pattern} n={n} dir={direction} elec={elec[0]}{elec[1]}"
                T = n * _lattice_vec(ham, direction)
                ru2, rd2 = ru.copy(), rd.copy()
                if elec[0] == "up":
                    ru2[elec[1]] = ru2[elec[1]] + T
                else:
                    rd2[elec[1]] = rd2[elec[1]] + T
                ginv2 = _ginv(ham, ru2, rd2)

                # e_L (production fast path, sweep-inverse threaded): exercises the
                # AO/kinetic/ECP-mesh far-image logic all four bugs lived in.
                # Full-complex compare (Im must be invariant too at twist).
                eL2 = _eL_jax(ham, ru2, rd2, ginv2)
                np.testing.assert_allclose(eL2, eL1, atol=atol, rtol=rtol, err_msg=f"{tag} e_L")

                # ln|Psi| modulus (J1+J2+J3+DTN Jastrow part + ln|det|; the Bloch
                # phase e^{i k.T} of a twist drops out of the modulus).
                lp2 = _ln_psi_jax(ham, ru2, rd2)
                np.testing.assert_allclose(
                    lp2,
                    lp1,
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"{tag} ln|Psi|",
                )

                # HF force and Pulay force (assemble gradients, one consistent inverse).
                (hfR2, hfu2, hfd2), (puR2, puu2, pud2) = _hf_and_pulay_force(ham, ru2, rd2, ginv2)
                for a, b, sub in ((hfR1, hfR2, "dR"), (hfu1, hfu2, "dr_up"), (hfd1, hfd2, "dr_dn")):
                    np.testing.assert_allclose(
                        b,
                        a,
                        atol=atol,
                        rtol=rtol,
                        err_msg=f"{tag} HF force {sub}",
                    )
                for a, b, sub in ((puR1, puR2, "dR"), (puu1, puu2, "dr_up"), (pud1, pud2, "dr_dn")):
                    np.testing.assert_allclose(
                        b,
                        a,
                        atol=atol,
                        rtol=rtol,
                        err_msg=f"{tag} Pulay force {sub}",
                    )


# =========================================================================== #
# Test 2: per-config translation sum rule (no shift, no MCMC -- a from-scratch
# identity on any single config). The 3 seeds share one test per pattern (jit
# amortization, as above). JAX-specific floors: the JAX assemble gradients close
# the identity at raw fp64 cancellation (HF max 4.2e-14, Pulay max 4.8e-13
# measured), far inside the native constants -- see jqmc/_setting.py.
# =========================================================================== #
_SUM_RULE_SEEDS = (0, 1, 2)


@pytest.mark.parametrize("pattern", _ALL_PATTERNS)
def test_per_config_translation_sum_rule(pattern):
    """Per config translation sum rule."""
    ham = _pattern_ham(pattern)
    for seed in _SUM_RULE_SEEDS:
        comparison_iteration = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _draw_well_conditioned_cfg(ham, seed)
        ginv = _ginv(ham, ru, rd)

        (hfR, hfu, hfd), (puR, puu, pud) = _hf_and_pulay_force(ham, ru, rd, ginv)

        hf_sum = np.real(hfR).sum(axis=0) + np.real(hfu).sum(axis=0) + np.real(hfd).sum(axis=0)
        np.testing.assert_allclose(
            hf_sum,
            np.zeros(3),
            atol=ATOL_FLOOR_MEDIUM,
            rtol=RTOL_FLOOR_MEDIUM,
            err_msg=f"{pattern} seed={seed}: sum_I dE_L/dR_I + sum_i dE_L/dr_i != 0",
        )

        pulay_sum = np.real(puR).sum(axis=0) + np.real(puu).sum(axis=0) + np.real(pud).sum(axis=0)
        np.testing.assert_allclose(
            pulay_sum,
            np.zeros(3),
            atol=ATOL_FLOOR_MEDIUM,
            rtol=RTOL_FLOOR_MEDIUM,
            err_msg=f"{pattern} seed={seed}: sum_I dln|Psi|/dR_I + sum_i dln|Psi|/dr_i != 0",
        )


# =========================================================================== #
# Many-body Bloch phase on every twist pattern, including the triclinic one.
# The modulus identities above cannot see a wrong
# per-electron phase factor: a transposed ``inv_cell`` in the pGTO wrap-restore
# phase exp(2 pi i k.floor(r inv_cell)) is locally constant in r, so |Psi|, e_L
# and the forces stay invariant under it. The full complex Psi must pick up
# exp(2 pi i k_s.n) when one electron of spin channel s moves by n.cell.
# =========================================================================== #
_BLOCH_SHIFTS = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, -1], [0, 3, 0], [2, -1, 1]])
_BLOCH_PSI_ZONES = ("ao_eval", "mo_eval", "jastrow_eval", "det_eval", "wf_eval")


def _channel_k(pattern, ham):
    """Fractional k of the up channel: the fixture request, checked against the built leaf."""
    geminal = ham.wavefunction_data.geminal_data
    orb = getattr(geminal, "orb_data_up_spin", None) or geminal.mos_up
    leaf = getattr(orb, "aos_data", orb)
    k_leaf = np.asarray(leaf.k_point, dtype=float)
    if pattern == "triclinic_agp_twist_j12":
        np.testing.assert_array_equal(k_leaf, _CANONICAL_016_K)
        return _CANONICAL_016_K
    return k_leaf


def _bloch_phase_violations(pattern):
    """Per-shift messages for every violated many-body Bloch phase (empty = pass).

    Every twist pattern is spin-restricted / time-reversal paired, so the dn
    channel carries -k. The cell is read from the Hamiltonian, the phase from k.
    """
    ham = _pattern_ham(pattern)
    ru, rd = _draw_well_conditioned_cfg(ham, 0)
    k_up = _channel_k(pattern, ham)
    cell = np.asarray(ham.structure_data.cell, float)
    atol, rtol = get_tolerance_min(_BLOCH_PSI_ZONES, "medium")
    wf = ham.wavefunction_data
    psi0 = complex(np.asarray(evaluate_wavefunction(wf, jnp.asarray(ru), jnp.asarray(rd))))
    assert np.isfinite(psi0) and abs(psi0) > 0.0
    failures = []
    for spin, k in (("up", k_up), ("dn", -k_up)):
        for n in _BLOCH_SHIFTS:
            expected = np.exp(2j * np.pi * float(k @ n))
            assert abs(expected - 1.0) > 0.5, f"{pattern}: vacuous shift n={n} for k={k}"
            ru2, rd2 = ru.copy(), rd.copy()
            (ru2 if spin == "up" else rd2)[0] += n @ cell
            ratio = complex(np.asarray(evaluate_wavefunction(wf, jnp.asarray(ru2), jnp.asarray(rd2)))) / psi0
            tag = f"{spin}0 n={tuple(int(x) for x in n)}"
            if not np.isclose(abs(ratio), 1.0, atol=atol, rtol=rtol):
                failures.append(f"{tag}: modulus |Psi(r+L)/Psi(r)|={abs(ratio):.12g}")
            unit = ratio / abs(ratio)
            if not abs(unit - expected) <= atol + rtol:
                failures.append(f"{tag}: phase {np.angle(unit):.9f} != expected {np.angle(expected):.9f}")
    return failures


@pytest.mark.parametrize("pattern", _TWIST_PATTERNS)
def test_many_body_bloch_phase_on_twist_patterns(pattern):
    """Moving one electron by a lattice vector multiplies the complex Psi by its channel Bloch phase.

    Patterns: the cubic H4 J1J2 and Li ECP J1J2DTN twists and the triclinic
    WF-PATTERN-016 AGP (J3 removed: no J3 at the twist). The configuration is the far-image
    recipe of this module. For the first up and first dn electron and three
    shifts with a nontrivial phase, |Psi(r+L)/Psi(r)| must be one and the unit
    ratio must equal exp(2 pi i k_s.n) with k_up from the fixture request and
    k_dn = -k_up. Full and mixed each use their medium wavefunction policy.
    """
    failures = _bloch_phase_violations(pattern)
    assert failures == [], f"{pattern}: " + "; ".join(failures)
    jax.clear_caches()


@pytest.mark.parametrize("pattern", ["h4_twist_j12", "triclinic_agp_twist_j12"])
def test_many_body_bloch_phase_rejects_a_transposed_inverse_cell(pattern, monkeypatch):
    """``inv_cell -> inv_cell.T`` in the pGTO wrap-restore phase is red only on the skewed cell.

    The fault replaces floor(r inv_cell) by floor(r inv_cell.T) in
    ``_ComplexBlochPGTOMixin._restore_phase``. It is the identity for a
    symmetric cell matrix (cubic H4), where the check must stay green, and must
    produce a phase (not a modulus) violation on the triclinic pattern. The
    modulus-only translation and sum-rule nodes above cannot see it.
    """
    from jqmc import atomic_orbital as ao

    def restore(self, r_carts, zone):
        inv_cell_t = ao._inv_3x3(self.structure_data._cell_jnp.astype(jnp.float64)).T
        k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
        n0 = jnp.floor(jnp.asarray(r_carts, dtype=jnp.float64) @ inv_cell_t)
        angle = jnp.asarray(2.0 * jnp.pi * (n0 @ k_frac), dtype=ao.get_dtype_jnp(zone))
        return jnp.exp(jnp.asarray(1j, dtype=ao.get_complex_dtype_jnp(zone)) * angle)

    jax.clear_caches()
    monkeypatch.setattr(ao._ComplexBlochPGTOMixin, "_restore_phase", restore)
    failures = _bloch_phase_violations(pattern)
    monkeypatch.undo()
    jax.clear_caches()
    cell = np.asarray(_pattern_ham(pattern).structure_data.cell, float)
    if np.array_equal(cell, cell.T):
        assert failures == [], f"{pattern}: fault must be invisible on a symmetric cell: {failures}"
    else:
        assert any(": phase " in f for f in failures), f"{pattern}: transposed inv_cell not detected: {failures}"
        assert not any(": modulus " in f for f in failures), f"{pattern}: unexpected modulus change: {failures}"


# =========================================================================== #
# Absolute-value anchor. Every node above compares a
# quantity with itself (shifted) or with zero, so an error that is uniform in the
# electron positions -- a wrong |k| in the Bloch phase or kinetic term, a global
# Re-projection, a wrong Coulomb constant -- passes them. Here e_L is compared
# with an independent absolute value: for a Jastrow-free Slater determinant of
# single plane waves exp(i (G + k) . r) the local kinetic energy equals
# sum_occ |G + k|^2 / 2 at every configuration in exact arithmetic, and the
# all-electron Coulomb part equals a textbook Ewald sum of the point charges
# (tests/test_jqmc_kernel/_noncubic_oracles.py). The reference kinetic
# energy is not that closed form but the determinant kinetic energy evaluated by
# numpy per precision zone (plane-wave values in ao_eval as the denominator,
# their gradients and Laplacians in ao_grad_lap, determinant algebra in det_eval,
# assembly in wf_kinetic; Ewald in the ewald zone), so mixed is judged against a
# reference in its own precision regime; in full every zone is fp64 and it equals
# the closed form to fp64 round-off. The cell is triclinic and k != Gamma.
# =========================================================================== #
_PW_CELL = np.array([[7.0, 0.0, 0.0], [1.5, 6.5, 0.0], [1.0, 2.0, 7.5]])
_PW_NUCLEI_FRAC = np.array([[0.1, 0.2, 0.3], [0.6, 0.15, 0.45], [0.35, 0.7, 0.2], [0.8, 0.55, 0.75]])
_PW_K = (0.25, -0.1, 0.4)
_PW_MILLERS_UP = np.array([[0, 0, 0], [1, 0, 0]])
_PW_MILLERS_DN = np.array([[0, 0, 0], [0, 1, -1]])
_PW_SEEDS = (0, 1, 2)
_PW_E_L_ZONES = ("ao_eval", "ao_grad_lap", "mo_eval", "det_eval", "det_grad_lap", "wf_kinetic", "coulomb", "ewald")
_PW_E_L_ZONES += ("local_energy",)


@functools.cache
def _free_electron_ham():
    from tests.test_jqmc_kernel._noncubic_oracles import build_free_electron_pw_hamiltonian

    return build_free_electron_pw_hamiltonian(_PW_CELL, _PW_NUCLEI_FRAC, _PW_K, _PW_MILLERS_UP, _PW_MILLERS_DN)


def _free_electron_config(seed):
    rng = np.random.default_rng(seed)
    # Fractional coordinates in [-1, 2): electrons sit in and outside the home cell.
    ru = rng.uniform(-1.0, 2.0, size=(len(_PW_MILLERS_UP), 3)) @ _PW_CELL
    rd = rng.uniform(-1.0, 2.0, size=(len(_PW_MILLERS_DN), 3)) @ _PW_CELL
    return ru, rd


def _free_electron_reference(ru, rd):
    from tests.test_jqmc_kernel._noncubic_oracles import free_electron_kinetic, numpy_ewald_energy

    # Both routes judged here (JAX sweep inverse, native inverse of the native geminal)
    # build the geminal inverse from ao_eval AO values: fused_values=False.
    kinetic = free_electron_kinetic(_PW_CELL, _PW_K, _PW_MILLERS_UP, _PW_MILLERS_DN, ru, rd, fused_values=False)
    x = np.concatenate([_PW_NUCLEI_FRAC @ _PW_CELL, ru, rd])
    q = np.concatenate([np.ones(len(_PW_NUCLEI_FRAC)), -np.ones(len(ru) + len(rd))])
    return kinetic, numpy_ewald_energy(_PW_CELL, x, q)


def _free_electron_mismatches():
    """Messages for every config whose e_L differs from the zone-dtype reference (empty = pass)."""
    ham = _free_electron_ham()
    atol, rtol = get_tolerance_min(_PW_E_L_ZONES, "strict")
    failures, coulombs = [], []
    for seed in _PW_SEEDS:
        ru, rd = _free_electron_config(seed)
        e_l = _eL_jax(ham, ru, rd, _ginv(ham, ru, rd))
        kinetic, coulomb = _free_electron_reference(ru, rd)
        assert np.isfinite(e_l) and np.isfinite(kinetic) and np.isfinite(coulomb)
        coulombs.append(coulomb)
        if not np.isclose(e_l, kinetic + coulomb, atol=atol, rtol=rtol, equal_nan=False):
            failures.append(
                f"seed={seed}: free-electron absolute e_L mismatch: e_L={e_l!r}, T_ref + Ewald = {kinetic!r} + {coulomb!r}"
            )
    return failures, coulombs, atol


def test_free_electron_plane_wave_local_energy_absolute_value():
    """e_L of a plane-wave determinant at k != Gamma on a triclinic cell equals its independent value.

    Four H nuclei in a triclinic cell, two up electrons in exp(i(G+k).r) with
    G in {0, b1} and two down electrons in exp(i(G-k).r) with G in {0, b2-b3},
    k=(0.25,-0.1,0.4), no Jastrow. At three seeded configurations (electrons also
    outside the home cell) the production e_L (compute_local_energy_fast with the
    sweep geminal inverse) must equal, full complex, the numpy determinant kinetic
    energy (exactly sum_occ |G +- k|^2 / 2, real, in exact arithmetic) plus the
    numpy Ewald energy of the neutral point charges, within the strict zone policy
    of the current mode. The reference evaluates each piece in the dtype of
    the production zone that computes it (plane-wave values in ao_eval as the
    determinant denominator, their derivatives in ao_grad_lap, determinant algebra
    in det_eval, Ewald in ewald), so mixed is not judged against an fp64 value;
    in full it is fp64 throughout. The Coulomb parts differ between the
    configurations by more than the tolerance, so a configuration-independent
    reference could not pass.
    """
    failures, coulombs, atol = _free_electron_mismatches()
    assert failures == [], "; ".join(failures)
    assert max(coulombs) - min(coulombs) > 1.0e3 * atol, "Coulomb reference does not vary with the configuration"
    jax.clear_caches()


def test_free_electron_absolute_value_rejects_a_scaled_bloch_momentum(monkeypatch):
    """Scaling k by 1.01 in the plane-wave Bloch factor must break the absolute anchor.

    The fault leaves every translation-invariance and sum-rule identity above intact
    (it is uniform in the electron positions) but changes the kinetic energy.
    """
    from jqmc import atomic_orbital as ao

    original = ao.AOs_plane_wave_complex_data._gk_vectors_jnp.fget

    def scaled(self):
        recip = 2.0 * jnp.pi * ao._inv_3x3(self.structure_data._cell_jnp).T
        k_cart = jnp.asarray(self.k_frac_np, dtype=jnp.float64) @ recip
        return original(self) + 0.01 * k_cart[None, :]

    jax.clear_caches()
    monkeypatch.setattr(ao.AOs_plane_wave_complex_data, "_gk_vectors_jnp", property(scaled))
    failures, _, _ = _free_electron_mismatches()
    monkeypatch.undo()
    jax.clear_caches()
    assert len(failures) == len(_PW_SEEDS), failures
    assert all("free-electron absolute e_L mismatch" in f for f in failures), failures
