"""Band folding: a supercell at Gamma equals the twist average of its primitive cell.

The twist-average tests in ``test_twist_average.py`` check the weighted BZ
combine as arithmetic on given per-twist numbers. They cannot see a per-twist
quantity that is the wrong physical object (a wrong ``k_frac -> k_cart`` map,
a missing ``2 pi``, a weight attached to the wrong twist): any set of numbers
has a valid weighted mean. This module checks the physics with an exact
identity.

System: a neutral lattice of H nuclei in a triclinic primitive cell ``P``
and a Jastrow-free Slater determinant of single plane waves (up electrons in
``exp(i (G + k) . r)``, down electrons in ``exp(i (G - k) . r)``), all
electrons, no ECP. For an ``n x 1 x 1`` supercell ``S = diag(n, 1, 1) P``,
the twists ``k = p b1 / n`` (``p = 0 .. n-1``) of ``P`` fold onto Gamma of
``S``: the orbital ``G + p b1 / n`` of ``P`` is the supercell reciprocal
vector with Miller index ``(n m1 + p, m2, m3)``. Take the supercell
determinant made of all folded orbitals and the supercell configuration made
of the primitive configuration and its copies shifted by ``a1, .., (n-1) a1``.
Then, exactly,

* the kinetic energy (configuration independent for single plane waves) is
  the sum of the primitive kinetic energies at the folded twists;
* the Ewald energy of the replicated point charges is ``n`` times the
  primitive Ewald energy (it is the same infinite periodic system);

so ``e_L(supercell) / n`` equals the twist average of the primitive ``e_L``
at the same configuration. The twist average is taken with the production
combine ``bz_mean_energy``. Two folds are checked: ``2 x 1 x 1`` with the
full-zone twists ``{0, 1/2}`` and equal weights, and ``3 x 1 x 1`` with the
irreducible twists ``{0, 1/3}`` and weights ``{1, 2}`` (the ``2/3`` orbitals of
the supercell are the time-reversed ``1/3`` orbitals).

Verdict: each precision mode against its own strict zone policy, at three
seeded configurations with electrons also outside the home cell. The
per-twist local energies differ from each other by far more than the
tolerance, so the comparison is not vacuous. Negative controls show that a
wrong supercell Miller index, an omitted twist, the total instead of the
per-cell supercell energy, permuted irreducible weights and a wrong primitive
twist each fail with the folding message.

Driver pairing (fourth control): the same 3 x 1 x 1 identity is also run
through the production driver ``run_twist_average_from_hamiltonians``, whose
per-twist MCMC is replaced by a deterministic stand-in returning the exact
production local energy of the Hamiltonian it was handed at the fixed primitive
configuration. The driver's pairing of each Hamiltonian (its k point) with its
weight, its twist-root reduction and its combine must reproduce the supercell
value. Handing the driver its Hamiltonian list permuted against the unequal
weights must fail with the folding message.
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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._twist_average import bz_mean_energy
from jqmc.hamiltonians import compute_local_energy
from jqmc.wavefunction import compute_kinetic_energy
from tests.test_jqmc_kernel._noncubic_oracles import build_free_electron_pw_hamiltonian, free_electron_kinetic

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_P = np.array([[7.0, 0.0, 0.0], [1.5, 6.5, 0.0], [1.0, 2.0, 7.5]])
_NUCLEI_FRAC = np.array([[0.1, 0.2, 0.3], [0.6, 0.15, 0.45], [0.35, 0.7, 0.2], [0.8, 0.55, 0.75]])
_MILLERS_UP = np.array([[0, 0, 0], [1, 0, 0]])
_MILLERS_DN = np.array([[0, 0, 0], [0, 1, -1]])
_SEEDS = (0, 1, 2)
_ZONES = ("ao_eval", "ao_grad_lap", "mo_eval", "det_eval", "det_grad_lap", "wf_kinetic", "coulomb", "ewald", "local_energy")

# fold -> (n, [(p, weight, add_time_reversed_partner_to_supercell), ...])
_FOLDS = {
    "2x1x1_full_zone": (2, ((0, 1.0, False), (1, 1.0, False))),
    "3x1x1_irreducible": (3, ((0, 1.0, False), (1, 2.0, True))),
}


def _supercell_millers(n, members, shift_first_up=0):
    """Supercell Miller indices of every folded primitive orbital."""
    up, dn = [], []
    for p, _w, time_reversed in members:
        up += [[n * m[0] + p, m[1], m[2]] for m in _MILLERS_UP]
        dn += [[n * m[0] - p, m[1], m[2]] for m in _MILLERS_DN]
        if time_reversed:
            up += [[-n * m[0] - p, -m[1], -m[2]] for m in _MILLERS_UP]
            dn += [[-n * m[0] + p, -m[1], -m[2]] for m in _MILLERS_DN]
    up = np.array(up)
    up[0, 1] += shift_first_up  # fault only: move one up orbital to an unoccupied G
    return up, np.array(dn)


@functools.cache
def _supercell_ham(fold, shift_first_up=0):
    n, members = _FOLDS[fold]
    cell = np.diag([float(n), 1.0, 1.0]) @ _P
    nuclei = np.concatenate([(_NUCLEI_FRAC + [t, 0.0, 0.0]) / [n, 1.0, 1.0] for t in range(n)])
    up, dn = _supercell_millers(n, members, shift_first_up)
    return build_free_electron_pw_hamiltonian(cell, nuclei, (0.0, 0.0, 0.0), up, dn)


@functools.cache
def _primitive_ham(k_first):
    return build_free_electron_pw_hamiltonian(_P, _NUCLEI_FRAC, (k_first, 0.0, 0.0), _MILLERS_UP, _MILLERS_DN)


def _config(seed, n):
    rng = np.random.default_rng(seed)
    ru = rng.uniform(-1.0, 2.0, size=(len(_MILLERS_UP), 3)) @ _P
    rd = rng.uniform(-1.0, 2.0, size=(len(_MILLERS_DN), 3)) @ _P
    rus = np.concatenate([ru + t * _P[0] for t in range(n)])
    rds = np.concatenate([rd + t * _P[0] for t in range(n)])
    return ru, rd, rus, rds


def _e_l(ham, ru, rd):
    return complex(np.asarray(compute_local_energy(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.eye(3))))


def _kinetic(ham, ru, rd):
    return complex(np.asarray(compute_kinetic_energy(ham.wavefunction_data, jnp.asarray(ru), jnp.asarray(rd))))


def _folding_mismatches(fold, fault="none"):
    """Messages for every seed whose supercell and twist-averaged energies differ (empty = pass)."""
    n, members = _FOLDS[fold]
    twists = [p / n for p, _w, _tr in members]
    weights = [w for _p, w, _tr in members]
    per_cell = float(n)
    if fault == "wrong-miller":
        supercell = _supercell_ham(fold, shift_first_up=1)
    else:
        supercell = _supercell_ham(fold)
    if fault == "omitted-twist":
        weights = [1.0] + [0.0] * (len(weights) - 1)
    elif fault == "total-not-per-cell":
        per_cell = 1.0
    elif fault == "permuted-weights":
        weights = weights[::-1]
    elif fault == "wrong-twist":
        twists = [twists[0]] + [0.5 * t for t in twists[1:]]
    primitives = [_primitive_ham(k) for k in twists]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    failures, spreads = [], []
    for seed in _SEEDS:
        ru, rd, rus, rds = _config(seed, n)
        e_super = _e_l(supercell, rus, rds)
        e_prim = [_e_l(h, ru, rd) for h in primitives]
        t_super = _kinetic(supercell, rus, rds)
        t_prim = [_kinetic(h, ru, rd) for h in primitives]
        values = [e_super, t_super, *e_prim, *t_prim]
        assert all(np.isfinite(v.real) and np.isfinite(v.imag) for v in values), values
        e_avg = bz_mean_energy([e.real for e in e_prim], [0.0] * len(e_prim), weights).E_TA
        t_avg = bz_mean_energy([t.real for t in t_prim], [0.0] * len(t_prim), weights).E_TA
        spreads.append(max(e.real for e in e_prim) - min(e.real for e in e_prim))
        checks = (
            ("e_L", e_super.real / per_cell, e_avg),
            ("kinetic", t_super.real / per_cell, t_avg),
            ("Im e_L supercell", e_super.imag, 0.0),
        )
        for label, got, want in checks:
            if not np.isclose(got, want, atol=atol, rtol=rtol, equal_nan=False):
                failures.append(
                    f"seed={seed}: twist-supercell folding mismatch ({label}): supercell/cell={got!r}, twist average={want!r}"
                )
    return failures, spreads, atol


@pytest.mark.parametrize("fold", sorted(_FOLDS))
def test_supercell_gamma_equals_twist_average_of_folded_primitive_cells(fold):
    """e_L and kinetic energy of the folded supercell per primitive cell equal the production twist average.

    The kinetic energies also equal, full complex, the independent numpy
    determinant kinetic energy of each twist (exactly sum_occ |G + k|^2 / 2 in
    exact arithmetic), so the per-twist objects are the right physical quantities
    and not only mutually consistent. That reference is evaluated per
    precision zone as compute_kinetic_energy routes it (plane-wave values from
    the fused ao_grad_lap evaluation stored in the ao_eval dtype as the
    denominator, derivatives in ao_grad_lap, determinant algebra in det_eval),
    not in fp64 for mixed.
    """
    failures, spreads, atol = _folding_mismatches(fold)
    assert failures == [], "; ".join(failures)
    assert min(spreads) > 1.0e3 * atol, f"per-twist local energies are too close to test the weights: {spreads}"
    n, members = _FOLDS[fold]
    ru, rd, _rus, _rds = _config(_SEEDS[0], n)
    atol_k, rtol_k = get_tolerance_min(_ZONES, "strict")
    for p, _w, _tr in members:
        got = _kinetic(_primitive_ham(p / n), ru, rd)
        want = free_electron_kinetic(_P, (p / n, 0.0, 0.0), _MILLERS_UP, _MILLERS_DN, ru, rd, fused_values=True)
        assert np.isclose(got, want, atol=atol_k, rtol=rtol_k, equal_nan=False), (p, got, want)
    jax.clear_caches()


@pytest.mark.parametrize(
    "fold,fault",
    [
        ("2x1x1_full_zone", "wrong-miller"),
        ("2x1x1_full_zone", "omitted-twist"),
        ("2x1x1_full_zone", "total-not-per-cell"),
        ("2x1x1_full_zone", "wrong-twist"),
        ("3x1x1_irreducible", "permuted-weights"),
    ],
)
def test_supercell_folding_rejects_wrong_folds(fold, fault):
    """Unmutated control passes; each wrong fold fails the folding identity at every seed, with the folding message."""
    control, _spreads, _atol = _folding_mismatches(fold)
    assert control == []
    failures, _spreads, _atol = _folding_mismatches(fold, fault)
    failed_seeds = sorted({int(f.split(":", 1)[0].removeprefix("seed=")) for f in failures})
    assert failed_seeds == list(_SEEDS), failures
    assert all("twist-supercell folding mismatch" in f for f in failures), failures
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# Driver pairing of per-twist k and weights (fourth control)
# --------------------------------------------------------------------------- #
_DRIVER_FOLD = "3x1x1_irreducible"


class _RecordingInterComm:
    """Twist-roots comm of the rank owning twist 1: records its Level-2 SUM operands."""

    def __init__(self):
        self.sent = []

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        self.sent.append(np.array(value, copy=True))
        return np.array(value, copy=True)


class _ReplayInterComm:
    """Twist-roots comm of the rank owning twist 0: adds the recorded twist-1 operands."""

    def __init__(self, remote):
        self._remote = remote
        self.calls = 0

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        remote = self._remote[self.calls]
        self.calls += 1
        return np.asarray(value) + remote


def _exact_local_energy_mcmc(ru, rd, seen):
    """Deterministic MCMC stand-in: E is the production e_L of its Hamiltonian at (ru, rd)."""

    class _ExactMCMC:
        def __init__(self, hamiltonian_data, **_kwargs):
            self._ham = hamiltonian_data
            seen.append(tuple(np.asarray(hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.k_point)))

        def run(self, num_mcmc_steps, max_time):
            assert num_mcmc_steps == 1
            # A completed MCMC run: the driver reads ``MCMC.mcmc_counter``.
            self.mcmc_counter = num_mcmc_steps

        def get_E(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
            return _e_l(self._ham, ru, rd).real, 0.0, None, None

    return _ExactMCMC


def _driver_twist_average(hams, weights, ru, rd):
    """Run the production twist driver on two twists through the recorded two-rank data flow."""
    from mpi4py import MPI

    import jqmc._twist_average as twist_average
    import jqmc.jqmc_mcmc as jqmc_mcmc

    recorder = _RecordingInterComm()
    replay = _ReplayInterComm(recorder.sent)
    seen = []
    result = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jqmc_mcmc, "MCMC", _exact_local_energy_mcmc(ru, rd, seen))
        for twist_index, inter in ((1, recorder), (0, replay)):
            mp.setattr(
                twist_average,
                "split_world_for_twists",
                lambda world_comm, n_twist, _t=twist_index, _c=inter: SimpleNamespace(
                    intra_comm=MPI.COMM_SELF, inter_comm=_c, twist_index=_t
                ),
            )
            result = twist_average.run_twist_average_from_hamiltonians(
                hams,
                weights,
                num_mcmc_steps=1,
                num_walkers=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=1,
                world_comm=MPI.COMM_SELF,
            )
    assert replay.calls == len(recorder.sent) > 0, "the Level-2 inter-comm SUMs of both twists must pair up"
    # The only integer Level-2 SUM is ``per_twist_stopped``; a completed twist sends zeros.
    stopped = [sent for sent in recorder.sent if sent.dtype.kind == "i"]
    assert len(stopped) == 1, "exactly one per_twist_stopped SUM"
    np.testing.assert_array_equal(stopped[0], np.zeros(len(hams), dtype=np.int64))
    return result, seen


def _driver_folding_mismatches(permute=False):
    """Messages for every seed whose supercell e_L per cell differs from the driver's twist average."""
    n, members = _FOLDS[_DRIVER_FOLD]
    twists = [p / n for p, _w, _tr in members]
    weights = np.asarray([w for _p, w, _tr in members])
    assert len(set(weights.tolist())) == len(weights), "the pairing control needs unequal weights"
    hams = [_primitive_ham(k) for k in twists]
    if permute:
        hams = hams[::-1]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    failures = []
    for seed in _SEEDS:
        ru, rd, rus, rds = _config(seed, n)
        result, seen = _driver_twist_average(hams, weights, ru, rd)
        assert sorted(seen) == sorted({(k, 0.0, 0.0) for k in twists}), seen
        got = result.energy.E_TA
        want = _e_l(_supercell_ham(_DRIVER_FOLD), rus, rds).real / n
        assert np.isfinite(got) and np.isfinite(want)
        if not np.isclose(got, want, atol=atol, rtol=rtol, equal_nan=False):
            failures.append(
                f"seed={seed}: twist-supercell folding mismatch (driver E_TA): supercell/cell={want!r}, driver={got!r}"
            )
    return failures


def test_twist_driver_pairs_each_k_point_with_its_weight_on_the_folding_identity():
    """The production twist driver reproduces the supercell value; a k/weight permutation fails.

    3 x 1 x 1 fold, primitive twists 0 and 1/3 with unequal irreducible weights 1 and 2,
    three seeded configurations. ``run_twist_average_from_hamiltonians`` runs twist 1
    (recording its twist-root SUM operands) and then twist 0 (receiving them), with a
    deterministic per-twist MCMC stand-in that returns the exact production e_L of the
    Hamiltonian the driver handed it. The driver's E_TA must equal e_L(supercell) / 3
    within the strict zone policy. The same driver input with the Hamiltonian list
    reversed against the weights (k point permuted relative to its weight) must fail
    at every seed with the folding message.
    """
    control = _driver_folding_mismatches()
    assert control == [], control
    failures = _driver_folding_mismatches(permute=True)
    failed_seeds = sorted({int(f.split(":", 1)[0].removeprefix("seed=")) for f in failures})
    assert failed_seeds == list(_SEEDS), failures
    assert all("twist-supercell folding mismatch" in f for f in failures), failures
    jax.clear_caches()
