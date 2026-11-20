"""Deterministic negative controls for the GFMC same-input audit adapter."""

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

import copy
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from tests import _gfmc_n_same_input as reference
from tests._fixed_estimator_oracle import estimator_reference, fixed_samples


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _assert_pairs(pairs):
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"), "medium")
    for name, (actual, expected) in pairs.items():
        if name == "exact":
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        else:
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=name)


def _toy_projection(monkeypatch):
    """Two prescribed synthetic hops; not a physical wavefunction witness."""
    ham = SimpleNamespace()
    up = np.zeros((1, 3))
    down = np.empty((0, 3))
    key = jax.random.PRNGKey(31)
    initial_key = np.array(key)
    hops = []
    for step in range(2):
        key, _ = jax.random.split(key)
        key, move = jax.random.split(key)
        uniform = np.asarray(jax.random.uniform(move))
        cdf = np.array([0.25, 1.0])
        selected = np.searchsorted(cdf, uniform)
        next_up = up + (-1 if selected == 0 else 1)
        hops.append(
            dict(
                i=step,
                r_up_carts=up,
                r_dn_carts=down,
                A_old_inv=np.ones((1, 1)),
                move_key=np.array(move),
                random_value=uniform,
                cdf=cdf,
                k=selected,
                p_list=np.array([-1.0, -3.0]),
                b_x=1.0,
                w_L=1.0,
                diagonal_sum_hamiltonian=3.0,
                non_diagonal_sum_hamiltonian=-4.0,
                RT=np.eye(3),
                ecp_RT=np.eye(3),
                next_up=next_up,
                next_dn=down,
            )
        )
        up = next_up
    interval = dict(
        step=0,
        inputs=(np.ones(1), np.zeros((1, 1, 3)), np.empty((1, 0, 3)), np.ones((1, 1, 1)), initial_key[None], -1.0),
        settings=(2, False, "tmove", 0.3),
        ham=ham,
        hops=hops,
        outputs=(np.ones(1), up[None], down[None], np.ones((1, 1, 1)), np.asarray(key)[None]),
    )

    def leaves(ham, up, down, *args, **kwargs):
        count = len(up)
        return (
            np.stack((up - 1, up + 1), axis=1),
            np.stack((down, down), axis=1),
            np.tile([-1.0, -3.0], (count, 1)),
            np.full(count, 3.0),
            np.zeros(count),
            np.tile([0.25, 0.75], (count, 1)),
            np.ones((count, 1, 1)),
        )

    monkeypatch.setattr(reference, "_batch_leaves", leaves)
    options = dict(
        expected_intervals=1,
        expected_walkers=1,
        expected_hops=2,
        expected_random_mesh=False,
        expected_non_local_move="tmove",
        expected_alat=0.3,
        expected_hamiltonian=ham,
    )
    return interval, options


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-hop",
        "duplicate-hop",
        "candidate",
        "cdf",
        "selection",
        "uniform",
        "key",
        "weight",
        "factor",
        "inverse",
        "handoff",
        "output",
        "hop-index",
        "mesh",
        "alat",
        "move",
        "hamiltonian",
        "interval",
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Synthetic toy hop records exercise the audit adapter only."
)
@pytest.mark.gdd_physics_object(
    mode="not-required",
    reason="Synthetic control: ham is an empty SimpleNamespace; _batch_leaves is a monkeypatched toy table.",
)
def test_gfmc_projection_reference_rejects_incomplete_or_corrupt_records(monkeypatch, mutation):
    """Every prescribed hop is required and corrupt numerical leaves or exact transitions fail.

    Parameter-set input: a one-walker/two-hop scalar-matrix synthetic protocol,
    not TREXIO, a physical wavefunction, or a spin-state coverage claim. The leaf
    backend is a fixed toy table; the key schedule and all comparison machinery
    are real. The intact record first passes, then one explicit alteration must
    raise. Existing central medium tolerances judge numeric leaves; selection,
    key and carry wiring are exact. No empirical sampling bands are used.
    """
    interval, options = _toy_projection(monkeypatch)
    _assert_pairs(reference.projection_comparisons([interval], **options))
    row = interval["hops"][0]
    if mutation == "missing-hop":
        interval["hops"].pop()
    elif mutation == "duplicate-hop":
        interval["hops"].append(copy.deepcopy(row))
    elif mutation in {"candidate", "cdf", "inverse", "handoff"}:
        field = {"candidate": "p_list", "cdf": "cdf", "inverse": "A_old_inv", "handoff": "r_up_carts"}[mutation]
        row[field] = row[field] + 0.5
    elif mutation in {"selection", "uniform", "weight", "factor", "hop-index"}:
        field = {"selection": "k", "uniform": "random_value", "weight": "w_L", "factor": "b_x", "hop-index": "i"}[mutation]
        row[field] = row[field] + 1
    elif mutation == "key":
        row["move_key"] = row["move_key"] + 1
    elif mutation == "output":
        interval["outputs"][1][0, 0, 0] += 1
    elif mutation in {"mesh", "alat", "move"}:
        settings = list(interval["settings"])
        index, value = {"mesh": (1, True), "alat": (3, 0.4), "move": (2, "dltmove")}[mutation]
        settings[index] = value
        interval["settings"] = tuple(settings)
    elif mutation == "hamiltonian":
        interval["ham"] = SimpleNamespace()
    elif mutation == "interval":
        interval["step"] = 1
    else:
        raise AssertionError("unknown negative control")
    with pytest.raises(AssertionError):
        _assert_pairs(reference.projection_comparisons([interval], **options))


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("collect,warmup,bins", [(0, 0, 4), (2, 1, 3)])
def test_gfmc_finite_sample_force_reference_matches_independent_quotient_derivative(complex_values, collect, warmup, bins):
    """The new fixed-data force arithmetic agrees with AD of a normalized energy.

    Ten explicit slices and three unequal fractions per slice retain within-slice
    covariance, real/complex energy, unequal blocks, and historical products.
    This validates the reference formula, not a production trajectory. Inputs
    are synthetic moments without electron positions, TREXIO or cond(G). Both
    modes use their current local-energy strict tolerance; an estimated standard
    error is itself compared as a deterministic number, never used as a band.
    """
    samples = fixed_samples(complex_values, np.float64)
    fractions, energy = samples.fractions, samples.energy
    mean_energy = np.sum(fractions * energy, axis=1)
    mean_square = np.sum(fractions * np.abs(energy) ** 2, axis=1)
    hf = np.sum(fractions[:, :, None] * samples.force_hf, axis=1)
    pp = np.sum(fractions[:, :, None] * samples.force_score, axis=1)
    ep = np.sum((fractions * energy)[:, :, None] * samples.force_score, axis=1)
    _, actual = reference.finite_sample_estimates(
        samples.bare_weights, mean_energy, mean_square, hf, pp, ep, collect=collect, warmup=warmup, bins=bins
    )
    expected = estimator_reference(samples, "force", collect, warmup, bins, 1.0)
    atol, rtol = get_tolerance_min(("local_energy",), "strict")
    np.testing.assert_allclose(np.asarray(actual).reshape(2, 2, 3), expected, atol=atol, rtol=rtol, equal_nan=False)


def _toy_population():
    """Prescribed two-rank/two-walker population with nonzero covariance."""
    rng = np.random.RandomState(13)
    state, zeta = rng.get_state(), rng.random()
    bare = np.array([1.0, 3.0, 2.0, 4.0])
    diagonal = np.array([5.0, 4.0, 3.0, 2.0])
    energy = np.array([2.0, -1.0, 3.0, -2.0])
    up = np.arange(12, dtype=float).reshape(4, 1, 3)
    down = np.empty((4, 0, 3))
    hf = (np.arange(12, dtype=float) / 10).reshape(4, 1, 3)
    pp = np.cos(hf)
    weights = bare / (diagonal + 1)
    norm = weights.sum()
    cdf = np.cumsum(bare / bare.sum())
    stored = {
        "_stored_w_L": np.array([[bare.mean()]]),
        "_stored_e_L": np.array([[np.dot(weights, energy) / norm]]),
        "_stored_e_L2": np.array([[np.dot(weights, energy**2) / norm]]),
        "_stored_force_HF": (weights[:, None, None] * hf).sum(axis=0)[None, None] / norm,
        "_stored_force_PP": (weights[:, None, None] * pp).sum(axis=0)[None, None] / norm,
        "_stored_E_L_force_PP": ((weights * energy)[:, None, None] * pp).sum(axis=0)[None, None] / norm,
    }
    payloads = []
    for rank in range(2):
        subset = slice(rank * 2, rank * 2 + 2)
        keys = np.asarray([jax.random.PRNGKey(1 + rank * 2 + i) for i in range(2)])
        draws = (np.arange(rank * 2, rank * 2 + 2) + zeta) / 4
        parents = np.searchsorted(cdf, draws)
        inputs = (np.ones(2), up[subset].copy(), down[subset].copy(), np.ones((2, 1, 1)), keys, -1.0)
        outputs = (bare[subset].copy(), up[subset].copy(), down[subset].copy(), np.ones((2, 1, 1)), keys.copy())
        branch = dict(
            step=0,
            zeta=zeta,
            w_L_list=bare[subset].copy(),
            V_diag_list=diagonal[subset].copy(),
            e_L_list=energy[subset].copy(),
            E_scf=-1.0,
            up=up[subset].copy(),
            down=down[subset].copy(),
        )
        fields = dict(
            copy.deepcopy(stored),
            _E_scf=-1.0,
            _epsilon_PW=0.0,
            _latest_r_up_carts=up[parents].copy(),
            _latest_r_dn_carts=down[parents].copy(),
            _jax_PRNG_key_list=keys.copy(),
        )
        payloads.append(
            dict(
                intervals=[dict(step=0, inputs=inputs, outputs=outputs)],
                branches=[branch],
                initial=(inputs[1].copy(), inputs[2].copy(), inputs[4].copy()),
                branch_rng_state=state,
                reconfigurations=[dict(step=0, cdf=cdf.copy(), draws=draws, parents=parents)],
                assembled=[(hf[subset].copy(), pp[subset].copy(), np.ones(2))],
                driver_fields=fields,
            )
        )
    return payloads


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "diagonal",
        "energy",
        "inverse",
        "rotation",
        "position",
        "grad_e_L_R",
        "grad_e_L_r_up",
        "grad_ln_Psi_dR",
        "grad_ln_Psi_r_up",
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Synthetic measurement records exercise the audit adapter only."
)
@pytest.mark.gdd_physics_object(
    mode="not-required",
    reason="Synthetic control: ham is an empty SimpleNamespace; _batch_leaves/_force_leaves are toy tables.",
)
def test_gfmc_measurement_reference_rejects_wrong_inputs_and_leaves(monkeypatch, mutation):
    """Stored-moment agreement cannot compensate for corrupted measurement inputs or derivative leaves.

    An explicit one-electron toy record uses fixed scalar-matrix and derivative
    tables, no physical wavefunction or TREXIO, no SWCT, and no conditioning
    selection. The normal observation first passes; each mutation must fail an
    exact input/inverse check or a separately compared leaf before aggregation.
    These controls validate reference wiring, not the derivative formulas.
    """
    interval, _ = _toy_projection(monkeypatch)
    interval["outputs"] = (*interval["outputs"], np.eye(3)[None], np.eye(3)[None], np.array([3.0]), np.array([-4.0]))
    raw = tuple(
        np.full((1, electrons, 3), value)
        for electrons, value in ((1, 0.125), (1, 0.25), (0, 0.25), (1, 0.5), (1, 0.2), (0, 0.2))
    )
    monkeypatch.setattr(
        reference, "_force_leaves", lambda *args, **kwargs: (raw, raw[0], raw[3], np.array([1 / np.sqrt(0.12)]))
    )
    outputs = interval["outputs"]
    branch = dict(
        step=0,
        E_scf=-1.0,
        w_L_list=outputs[0].copy(),
        up=outputs[1].copy(),
        down=outputs[2].copy(),
        latest_kinetic_RTs=outputs[5].copy(),
        latest_RTs=outputs[6].copy(),
        V_diag_list=np.array([3.0]),
        e_L_list=np.array([0.0]),
    )
    branch.update({field: value.copy() for field, value in zip(reference._GRADIENT_FIELDS, raw, strict=True)})
    branches = [branch]
    _assert_pairs(reference.measurement_comparisons([interval], branches, use_swct=False)[0])
    if mutation == "missing":
        branches.clear()
    elif mutation == "inverse":
        interval["outputs"][3][0, 0, 0] += 1
    else:
        field = {"diagonal": "V_diag_list", "energy": "e_L_list", "rotation": "latest_RTs", "position": "up"}.get(
            mutation, mutation
        )
        branch[field].flat[0] += 1
    with pytest.raises(AssertionError):
        _assert_pairs(reference.measurement_comparisons([interval], branches, use_swct=False)[0])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-rank-interval",
        "missing-branch",
        "missing-selection",
        "missing-force",
        "cdf",
        "draw",
        "parent",
        "initial",
        "carry",
        "key",
        "zeta",
        "shift",
        "bare-weight",
        "energy",
        "energy-square",
        "hf",
        "pp",
        "energy-pp",
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Synthetic population records exercise the audit adapter only."
)
def test_gfmc_population_reference_rejects_missing_or_corrupt_reductions(mutation):
    """Incomplete population records, wrong parents, carries and stored moments cannot pass.

    Explicit synthetic two-rank/two-walker inputs exercise the comparison adapter
    without running MPI or a Hamiltonian. Unequal weights, diagonal denominators,
    local energies and force scores prevent covariance from factorizing. The
    intact record must pass before each single corruption is required to fail.
    This is a deterministic guard test, not a stochastic physics guarantee.
    """
    payloads = _toy_population()
    options = dict(expected_intervals=1, expected_walkers=2, initial_energy_shift=-1.0)
    _assert_pairs(reference.population_comparisons(payloads, **options))
    p = payloads[0]
    if mutation.startswith("missing-"):
        field = {
            "missing-rank-interval": "intervals",
            "missing-branch": "branches",
            "missing-selection": "reconfigurations",
            "missing-force": "assembled",
        }[mutation]
        p[field].clear()
    elif mutation in {"cdf", "draw", "parent"}:
        field = {"cdf": "cdf", "draw": "draws", "parent": "parents"}[mutation]
        p["reconfigurations"][0][field][0] += 1
    elif mutation == "initial":
        p["initial"][0][0, 0, 0] += 1
    elif mutation in {"carry", "key"}:
        field = "_latest_r_up_carts" if mutation == "carry" else "_jax_PRNG_key_list"
        p["driver_fields"][field].flat[0] += 1
    elif mutation in {"zeta", "shift"}:
        p["branches"][0]["zeta" if mutation == "zeta" else "E_scf"] += 1
    else:
        field = {
            "bare-weight": "_stored_w_L",
            "energy": "_stored_e_L",
            "energy-square": "_stored_e_L2",
            "hf": "_stored_force_HF",
            "pp": "_stored_force_PP",
            "energy-pp": "_stored_E_L_force_PP",
        }[mutation]
        p["driver_fields"][field].flat[0] += 1
    with pytest.raises(AssertionError):
        _assert_pairs(reference.population_comparisons(payloads, **options))
