"""Exponent-tail bound of the exponent-only pGTO optimiser.

Port of trexio-tools ``tests/espresso/test_exp_tail_bounds.py``. A primitive of shell
angular momentum l and exponent alpha places the fraction
``w_l(alpha) = Q(l + 3/2, g_max^2 / (2 alpha))`` of its momentum-space norm
outside the WFC sphere; ``exp_tail_eps`` bounds ``ln(alpha) <= ln(alpha_max)``.
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

import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from jqmc._setting import (
    ATOL_BIT_EXACT,
    PGTO_EXP_TAIL_CLIP_MARGIN,
    PGTO_EXP_TAIL_CLIP_SPREAD_RATIO,
    PGTO_EXP_TAIL_QUADRATURE_EPSREL,
    PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_TIGHT,
)
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_pgto_exponent_optimizer import _one_gaussian_reference

pytestmark = [pytest.mark.with_chdir]

_QE_DIR = qe_example_files_root()
_LI_GAMMA = _QE_DIR / "Li_gamma" / "out" / "li.save"
_LI_K025 = _QE_DIR / "Li_k025" / "out" / "li_k025.save"
_LI_TWIST = _QE_DIR / "Li_k_twist" / "out" / "li_twist.save"
_O_LSDA = _QE_DIR / "O-lsda_gamma" / "out" / "o.save"


def _qe_save(name):
    """The one committed QE save a parametrized row reads, selected outside the test body.

    A test body (or one line) that names several save constants would let receipt
    collection attribute every named save to each row; each save is returned from
    its own statement here.
    """
    if name == "Li_gamma":
        return _LI_GAMMA
    if name == "Li_k025":
        return _LI_K025
    if name == "Li_k_twist":
        return _LI_TWIST
    if name == "O-lsda_gamma":
        return _O_LSDA
    raise KeyError(name)


_CLIP_WARNING_PREFIX = "exp_tail_eps bound:"

# exp_tail_eps=None must be the unbounded optimiser.  A recorded hex
# snapshot is not a valid oracle for that: the SLSQP iterates differ in the
# last bits between precision modes and between jax versions (mixed precision:
# z_opt[0] = 0x1.0fe64d7b3dfb1p-1 with jax 0.7.0/0.7.2 and 0x1.0fe64d7c92fa9p-1
# with jax 0.9.2, against 0x1.0fe64d7fd2f8bp-1 in full precision).
# The test therefore checks, environment-independently, that the
# None path hands SLSQP exactly the unbounded arguments and post-processes its
# point exactly as the unbounded optimiser does, and that every spelling of
# "disabled" agrees bit for bit with the plain call in the same process.  Seed exponents (0.25, 0.9) on
# the one-Gaussian reference (target exponent 1.7), maxiter=40.
_DISABLED_BOUND_CASES = {
    "pgto": {"maxiter": 40},
    "mixed_floor": {"g_cut": 0.8, "exponent_floor": 0.1, "maxiter": 40},
}
_RESULT_FIELDS = (
    "accepted",
    "scipy_success",
    "message",
    "n_iterations",
    "initial_spilling",
    "optimized_spilling",
    "initial_condition",
    "optimized_condition",
    "deployed_initial_spilling",
    "deployed_optimized_spilling",
    "deployed_initial_condition",
    "deployed_optimized_condition",
)


def _s_shell(exponents):
    from jqmc.upf_wrapper import PrimitiveShell

    return PrimitiveShell(0, tuple(exponents), (1.0,) * len(exponents), 0.1, 1.0)


def _clip_warnings(caplog):
    return [
        rec for rec in caplog.records if rec.levelno == logging.WARNING and rec.getMessage().startswith(_CLIP_WARNING_PREFIX)
    ]


def _spy_minimize(monkeypatch, *, overshoot_last=False):
    """Record SLSQP's arguments and returned x; optionally push one entry past the box.

    After the real SLSQP call returns, the objective, its gradient and the
    constraint are also evaluated at x0 so that two optimiser calls can be
    compared through the functions they handed to SLSQP.
    """
    from jqmc import pgto_exponent_optimizer

    real_minimize = pgto_exponent_optimizer.minimize
    calls = []

    def _wrapped(fun, x0, *args, **kwargs):
        result = real_minimize(fun, x0, *args, **kwargs)
        x = np.asarray(result.x, dtype=np.float64).copy()
        if overshoot_last:
            index = int(np.argmax(x))
            x[index] = float(kwargs["bounds"][index][1]) + 1.0e-3
            result.x = x
        x0_arr = np.asarray(x0, dtype=np.float64).copy()
        constraints = kwargs.get("constraints")
        calls.append(
            {
                "x0": x0_arr,
                "bounds": kwargs.get("bounds"),
                "x": x.copy(),
                "args": args,
                "method": kwargs.get("method"),
                "options": kwargs.get("options"),
                "constraints": constraints,
                "has_callback": callable(kwargs.get("callback")),
                "fun_x0": float(fun(x0_arr)),
                "jac_x0": np.asarray(kwargs["jac"](x0_arr), dtype=np.float64).copy(),
                "con_x0": [np.asarray(c.fun(x0_arr), dtype=np.float64).copy() for c in constraints],
                "con_jac_x0": [np.asarray(c.jac(x0_arr), dtype=np.float64).copy() for c in constraints],
            }
        )
        return result

    monkeypatch.setattr(pgto_exponent_optimizer, "minimize", _wrapped)
    return calls


def _spy_deployed(monkeypatch):
    """Record the shells passed to the deployed audit (seed first, candidate second)."""
    from jqmc import pgto_exponent_optimizer

    real_deployed = pgto_exponent_optimizer._deployed_diagnostics
    calls = []

    def _wrapped(refs, shells, **kwargs):
        calls.append(shells)
        return real_deployed(refs, shells, **kwargs)

    monkeypatch.setattr(pgto_exponent_optimizer, "_deployed_diagnostics", _wrapped)
    return calls


def _o_lsda_references():
    from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

    up = load_qe_reference(str(_O_LSDA), spin="up")
    dn = load_qe_reference(str(_O_LSDA), spin="dw")
    # Same occupied-only truncation as run_qe2jqmc's LSDA branch.
    return (
        truncate_qe_reference_bands(up, int(up.bands.occupied_band_indices().max()) + 1),
        truncate_qe_reference_bands(dn, int(dn.bands.occupied_band_indices().max()) + 1),
    )


def _upf_shells(reference, upf_path):
    from jqmc.upf_wrapper import UPFExponentInit

    labels = reference.atom_elements or reference.atom_labels
    strategy = UPFExponentInit(
        {labels[0]: str(upf_path)},
        g_target=float(np.sqrt(2.0 * reference.ecutwfc)),
        cell_min_edge=float(np.min(np.linalg.norm(reference.cell.a, axis=1))),
    )
    return [strategy.shells_for_element(label) for label in reference.atom_labels]


# --- 1-3: closed forms ------------------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("angular_l", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_pw_tail_weight_matches_momentum_space_quadrature(angular_l, alpha):
    """pw_tail_weight equals the quadrature ratio of the momentum-space radial density beyond g_max."""
    from jqmc.pgto_exponent_optimizer import pw_tail_weight

    g_max = 10.0

    def density(q):
        return q ** (2 * angular_l + 2) * np.exp(-(q**2) / (2.0 * alpha))

    tail, _ = quad(density, g_max, np.inf, epsabs=ATOL_BIT_EXACT, epsrel=PGTO_EXP_TAIL_QUADRATURE_EPSREL, limit=200)
    total, _ = quad(density, 0.0, np.inf, epsabs=ATOL_BIT_EXACT, epsrel=PGTO_EXP_TAIL_QUADRATURE_EPSREL, limit=200)
    weight = float(pw_tail_weight(angular_l, alpha, g_max))

    assert np.isfinite(weight) and weight > 0.0
    assert weight == pytest.approx(tail / total, rel=RTOL_PYTHON_FP64_LOOSE, abs=ATOL_BIT_EXACT)


@pytest.mark.light
@pytest.mark.parametrize("angular_l", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("eps", [1.0e-1, 1.0e-2, 1.0e-4])
def test_alpha_max_for_tail_is_closed_form_inverse(angular_l, eps):
    """alpha_max_for_tail inverts pw_tail_weight, and a softer exponent stays inside the bound."""
    from jqmc.pgto_exponent_optimizer import alpha_max_for_tail, pw_tail_weight

    g_max = float(np.sqrt(60.0))
    alpha_max = alpha_max_for_tail(angular_l, g_max, eps)

    assert float(pw_tail_weight(angular_l, alpha_max, g_max)) == pytest.approx(
        eps, rel=RTOL_PYTHON_FP64_LOOSE, abs=ATOL_BIT_EXACT
    )
    assert float(pw_tail_weight(angular_l, 0.5 * alpha_max, g_max)) < eps


@pytest.mark.light
@pytest.mark.parametrize("case", ["si_ecut50Ha", "li_ecut30Ha"])
def test_alpha_max_for_tail_reference_values(case):
    """alpha_max(l=0, eps=1e-2) reproduces the trexio-tools reference values."""
    from jqmc.pgto_exponent_optimizer import alpha_max_for_tail

    # g_max = sqrt(2 ecutwfc[Ha]): Si at 50 Ha and Li at 30 Ha.
    g_max, expected = {"si_ecut50Ha": (10.0, 8.8146), "li_ecut30Ha": (float(np.sqrt(60.0)), 5.2887)}[case]
    # The reference is quoted to four decimals: compare exactly at that precision.
    assert round(alpha_max_for_tail(0, g_max, 1.0e-2), 4) == expected


# --- 4, 5, 10: g_max and layout ----------------------------------------------


@pytest.mark.light
@pytest.mark.parametrize("fixture", ["Li_gamma", "Li_k025"])
def test_wfc_g_max_equals_reference_block_target_q(fixture):
    """_wfc_g_max is the largest |G+k| of the q rows the spilling objective targets."""
    from jqmc.pgto_exponent_optimizer import _reference_block, _wfc_g_max
    from jqmc.qe_wrapper import load_qe_reference

    reference = load_qe_reference(str(_qe_save(fixture)))
    block = _reference_block(
        reference,
        occ_only=True,
        occ_tol=1.0e-6,
        g_cut=None,
        fixed_mixed_overlap=False,
        chunk_rows=1024,
    )
    q = np.asarray(block.target_q_chunks).reshape(-1, 3)
    mask = np.asarray(block.target_mask_chunks).reshape(-1) > 0.0
    expected = float(np.sqrt(np.max(np.sum(q[mask] * q[mask], axis=1))))

    assert int(np.count_nonzero(mask)) == reference.wfc.millers.shape[0]
    assert _wfc_g_max([reference]) == expected
    if fixture == "Li_k025":
        # The k shift is part of the bound: |G| alone differs at this twist.
        g_only = np.asarray(reference.wfc.millers, dtype=np.float64) @ reference.cell.b
        assert float(np.sqrt(np.max(np.sum(g_only * g_only, axis=1)))) != expected


@pytest.mark.light
def test_layout_records_shell_angular_momentum_per_exponent():
    """Every unique exponent carries the angular momentum of its shell; a mismatch across equivalent atoms is rejected."""
    from jqmc.pgto_exponent_optimizer import _build_layout
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import PrimitiveShell

    reference = load_qe_reference(str(_LI_GAMMA))
    shells = _upf_shells(reference, _LI_GAMMA / "Li.ccECP.upf")
    layout = _build_layout(reference, shells)

    assert layout.uid_angular_l.shape == layout.initial_exponents.shape
    seen = set()
    for atom_shells, atom_map in zip(shells, layout.shell_param_ids, strict=True):
        for shell, uids in zip(atom_shells, atom_map, strict=True):
            for uid in uids:
                assert int(layout.uid_angular_l[uid]) == int(shell.angular_l)
                seen.add(uid)
    assert seen == set(range(layout.initial_exponents.size))
    assert {int(v) for v in layout.uid_angular_l} == {0, 1}

    mismatched = [
        [PrimitiveShell(0, (1.0,), (1.0,), 0.1, 1.0)],
        [PrimitiveShell(1, (1.0,), (1.0,), 0.1, 1.0)],
    ]
    with pytest.raises(ValueError, match="shell angular momentum"):
        _build_layout(reference, mismatched)


@pytest.mark.light
def test_g_max_is_maximum_over_lsda_channels_and_twists():
    """LSDA up/dn give the same g_max and bound; several twists give the maximum over all of them."""
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail
    from jqmc.qe_wrapper import load_qe_reference

    up, dn = _o_lsda_references()
    g_up = _wfc_g_max([up])
    assert g_up == _wfc_g_max([dn])
    assert g_up == _wfc_g_max([up, dn])
    for angular_l in (0, 1):
        assert alpha_max_for_tail(angular_l, g_up, 1.0e-2) == alpha_max_for_tail(angular_l, _wfc_g_max([up, dn]), 1.0e-2)

    # Gamma (|G|max 7.6957) and the k=0.25 twist (|G+k|max 7.7357) of the same
    # Li cell, plus the eight symmetry-equivalent twists of the 2x2x2 grid.
    twists = [load_qe_reference(str(_LI_GAMMA)), load_qe_reference(str(_LI_K025))]
    twists += [load_qe_reference(str(_LI_TWIST), ik=ik) for ik in range(1, 9)]
    per_twist = [_wfc_g_max([ref]) for ref in twists]
    assert len(set(per_twist)) > 1
    assert _wfc_g_max(twists) == max(per_twist)
    assert _wfc_g_max(list(reversed(twists))) == max(per_twist)
    assert _wfc_g_max(twists) > per_twist[0]


# --- 6, 7: bound active / inactive ------------------------------------------


@pytest.mark.light
def test_active_bound_clips_lsda_seed_and_bounds_optimised_exponents(monkeypatch, caplog):
    """O LSDA ccECP seed: s 6.88 and p 8.45 exceed alpha_max(l) at the WFC sphere; they are clipped and bounded."""
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail, optimize_pgto_exponents
    from jqmc.pw_pgto_projection import alpha_min_from_gcut, filter_shells_by_alpha_min

    up, dn = _o_lsda_references()
    g_cut = float(np.sqrt(10.0))
    floor = alpha_min_from_gcut(g_cut)
    shells = filter_shells_by_alpha_min(_upf_shells(up, _O_LSDA / "O.ccECP.upf"), floor)
    tail_fraction = 1.0e-2
    g_max = _wfc_g_max([up, dn])
    alpha_max = {angular_l: alpha_max_for_tail(angular_l, g_max, tail_fraction) for angular_l in (0, 1)}
    violating = {(shell.angular_l, e) for shell in shells[0] for e in shell.exponents if e > alpha_max[shell.angular_l]}
    assert {angular_l for angular_l, _ in violating} == {0, 1} and len(violating) == 2
    assert g_max > g_cut

    deployed_calls = _spy_deployed(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        result = optimize_pgto_exponents(
            [up, dn], shells, occ_only=True, g_cut=g_cut, exponent_floor=floor, exp_tail_eps=tail_fraction
        )

    warnings = _clip_warnings(caplog)
    assert len(warnings) == 1 and "2 initial exponent(s)" in warnings[0].getMessage()
    assert result.n_clipped == 2
    assert result.exp_tail_eps == tail_fraction
    assert result.g_max == g_max

    # The deployed baseline is the clipped seed, not the infeasible original.
    seed = deployed_calls[0]
    for original, clipped in zip(shells[0], seed[0], strict=True):
        bound = alpha_max[original.angular_l]
        expected = sorted(
            float(np.exp(np.log(bound) - PGTO_EXP_TAIL_CLIP_MARGIN)) if e > bound else e for e in original.exponents
        )
        assert clipped.exponents == tuple(expected)

    assert result.accepted
    assert set(result.alpha_report) == {0, 1}
    for angular_l, entry in result.alpha_report.items():
        assert entry["ok"] is True
        assert entry["alpha_max"] == alpha_max[angular_l]
    for shell in result.shells_per_atom[0]:
        assert max(shell.exponents) <= alpha_max[shell.angular_l] * (1.0 + RTOL_PYTHON_FP64_TIGHT)
    assert result.deployed_optimized_spilling < result.deployed_initial_spilling


@pytest.mark.light
def test_inactive_bound_leaves_seed_unchanged_and_silent(monkeypatch, caplog):
    """When every seed exponent is below alpha_max no warning fires and the SLSQP start point is the seed."""
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail, optimize_pgto_exponents

    reference = _one_gaussian_reference(1.7)
    seed = (0.25, 0.9)
    tail_fraction = 1.0e-2
    g_max = _wfc_g_max([reference])
    assert max(seed) < alpha_max_for_tail(0, g_max, tail_fraction)
    reference_run = optimize_pgto_exponents(reference, [[_s_shell(seed)]], maxiter=40)

    calls = _spy_minimize(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        result = optimize_pgto_exponents(reference, [[_s_shell(seed)]], maxiter=40, exp_tail_eps=tail_fraction)

    assert _clip_warnings(caplog) == []
    assert result.n_clipped == 0
    assert np.array_equal(calls[0]["x0"], np.log(np.asarray(seed)))
    assert calls[0]["bounds"] == [(None, float(np.log(alpha_max_for_tail(0, g_max, tail_fraction))))] * 2
    assert result.initial_spilling == reference_run.initial_spilling
    assert result.deployed_initial_spilling == reference_run.deployed_initial_spilling
    assert result.alpha_report[0]["ok"] is True


# --- 7b: clamp and rejection fallback ---------------------------------------


@pytest.mark.light
def test_slsqp_overshoot_is_clamped_to_box_upper_bound(monkeypatch):
    """A returned SLSQP point outside the box is clamped before the deployed audit and the adoption test."""
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail, optimize_pgto_exponents

    reference = _one_gaussian_reference(1.7)
    tail_fraction = 1.0e-2
    alpha_max = alpha_max_for_tail(0, _wfc_g_max([reference]), tail_fraction)
    _spy_minimize(monkeypatch, overshoot_last=True)
    deployed_calls = _spy_deployed(monkeypatch)

    result = optimize_pgto_exponents(reference, [[_s_shell((0.25, 0.9))]], maxiter=40, exp_tail_eps=tail_fraction)

    candidate = deployed_calls[1][0][0].exponents
    assert max(candidate) == float(np.exp(np.log(alpha_max)))
    assert max(candidate) <= alpha_max * (1.0 + RTOL_PYTHON_FP64_TIGHT)
    assert result.alpha_report[0]["ok"] is True
    assert result.accepted
    assert result.shells_per_atom[0][0].exponents == candidate


@pytest.mark.light
def test_rejected_point_returns_clipped_seed_per_provisional_policy(monkeypatch, caplog):
    """A rejected SLSQP point returns the clipped seed (PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT)."""
    from jqmc import pgto_exponent_optimizer
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail

    reference = _one_gaussian_reference(1.7)
    tail_fraction = 1.0e-2
    alpha_max = alpha_max_for_tail(0, _wfc_g_max([reference]), tail_fraction)
    shells = [[_s_shell((0.25, 9.0))]]
    assert 9.0 > alpha_max
    seen = []
    forced = iter(((1.0e-2, 1.0), (1.0e-3, float("inf"))))

    def _forced(refs, candidate_shells, **kwargs):
        seen.append(candidate_shells)
        return next(forced)

    monkeypatch.setattr(pgto_exponent_optimizer, "_deployed_diagnostics", _forced)
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        result = pgto_exponent_optimizer.optimize_pgto_exponents(
            reference, shells, g_cut=0.8, exponent_floor=0.1, maxiter=20, exp_tail_eps=tail_fraction
        )

    clipped = (0.25, float(np.exp(np.log(alpha_max) - PGTO_EXP_TAIL_CLIP_MARGIN)))
    assert len(_clip_warnings(caplog)) == 1
    assert result.n_clipped == 1
    assert not result.accepted
    assert seen[0][0][0].exponents == clipped
    if PGTO_EXP_TAIL_RETURN_CLIPPED_ON_REJECT:
        assert result.shells_per_atom is seen[0]
        assert result.shells_per_atom[0][0].exponents == clipped
    else:
        assert result.shells_per_atom is shells


# --- 8: exp_tail_eps=None is the unbounded optimiser -------------------------


@pytest.mark.light
@pytest.mark.parametrize("case", ["mixed_floor", "pgto"])
@pytest.mark.parametrize("spelling", ["omitted", "explicit-none", "none-with-g-max"])
def test_disabled_bound_is_bit_identical_to_pre_change_optimizer(monkeypatch, case, spelling):
    """exp_tail_eps=None hands SLSQP the unbounded arguments, keeps its point and matches the plain call bit for bit."""
    from scipy.optimize import NonlinearConstraint

    from jqmc.pgto_exponent_optimizer import optimize_pgto_exponents

    kwargs = _DISABLED_BOUND_CASES[case]
    reference = _one_gaussian_reference(1.7)
    seed = (0.25, 0.9)
    extra = {"omitted": {}, "explicit-none": {"exp_tail_eps": None}, "none-with-g-max": {"exp_tail_eps": None, "g_max": 1.0}}

    def _run(call_kwargs):
        shells = [[_s_shell(seed)]]
        with monkeypatch.context() as patch:
            calls = _spy_minimize(patch)
            deployed = _spy_deployed(patch)
            result = optimize_pgto_exponents(reference, shells, **kwargs, **call_kwargs)
        return shells, calls, deployed, result

    shells, calls, deployed, result = _run(extra[spelling])
    _, plain_calls, _, plain = _run({})

    # Pre-change SLSQP arguments: the unclipped seed, the floor-only (or no)
    # box, one log-condition inequality, and the unchanged options.
    assert len(calls) == 1
    call = calls[0]
    assert np.array_equal(call["x0"], np.log(np.asarray(seed, dtype=np.float64)))
    floor = kwargs.get("exponent_floor")
    assert call["bounds"] == (None if floor is None else [(float(np.log(floor)), None)] * len(seed))
    assert call["args"] == ()
    assert call["method"] == "SLSQP"
    assert call["options"] == {"ftol": 1.0e-6, "maxiter": kwargs["maxiter"], "disp": False}
    assert call["has_callback"]
    assert len(call["constraints"]) == 1
    constraint = call["constraints"][0]
    assert isinstance(constraint, NonlinearConstraint)
    assert constraint.lb == 0.0 and constraint.ub == np.inf

    # Pre-change post-processing: the SLSQP point is used unclamped, the
    # deployed baseline is the caller's own seed object, and no bound
    # diagnostics are produced.
    assert deployed[0] is shells
    assert deployed[1][0][0].exponents == tuple(sorted(float(np.exp(v)) for v in call["x"]))
    assert result.accepted
    assert result.shells_per_atom is deployed[1]
    assert result.exp_tail_eps is None and result.g_max is None
    assert result.alpha_report == {} and result.n_clipped == 0

    # Every spelling of "disabled" is the plain call, bit for bit, in-process.
    plain_call = plain_calls[0]
    assert np.array_equal(call["x"], plain_call["x"])
    assert call["fun_x0"] == plain_call["fun_x0"]
    assert np.array_equal(call["jac_x0"], plain_call["jac_x0"])
    assert all(np.array_equal(a, b) for a, b in zip(call["con_x0"], plain_call["con_x0"], strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(call["con_jac_x0"], plain_call["con_jac_x0"], strict=True))
    for field in _RESULT_FIELDS:
        assert getattr(result, field) == getattr(plain, field), field
    assert result.shells_per_atom[0][0].exponents == plain.shells_per_atom[0][0].exponents


# --- 9: infeasible box and argument validation -------------------------------


@pytest.mark.light
def test_floor_above_tail_ceiling_raises():
    """exponent_floor >= alpha_max(l, eps) leaves an empty box and must raise before any optimisation."""
    from jqmc.pgto_exponent_optimizer import PGTOExponentBoundsError, _wfc_g_max, alpha_max_for_tail, optimize_pgto_exponents

    reference = _one_gaussian_reference(1.7)
    alpha_max = alpha_max_for_tail(0, _wfc_g_max([reference]), 1.0e-2)
    shells = [[_s_shell((0.25, 0.9))]]

    with pytest.raises(PGTOExponentBoundsError, match="exponent box is infeasible"):
        optimize_pgto_exponents(reference, shells, exponent_floor=1.5 * alpha_max, exp_tail_eps=1.0e-2)
    with pytest.raises(PGTOExponentBoundsError, match="exponent box is infeasible"):
        optimize_pgto_exponents(reference, shells, exponent_floor=alpha_max, exp_tail_eps=1.0e-2)
    for bad_eps in (0.0, 1.0, -1.0e-2, 2.0):
        with pytest.raises(ValueError, match="exp_tail_eps must satisfy") as error:
            optimize_pgto_exponents(reference, shells, exp_tail_eps=bad_eps)
        assert type(error.value) is ValueError
    for bad_g_max in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="g_max must be finite and positive") as error:
            optimize_pgto_exponents(reference, shells, exp_tail_eps=1.0e-2, g_max=bad_g_max)
        assert type(error.value) is ValueError


# --- entry-point wiring (user default) -----------------


class _StopAtOptimizer(Exception):
    pass


def _abort_at_optimizer(monkeypatch):
    """Record the kwargs of the first optimize_pgto_exponents call and abort the conversion there."""
    from jqmc import pgto_exponent_optimizer

    calls = []

    def _recording(references, shells_per_atom, **kwargs):
        calls.append(kwargs)
        raise _StopAtOptimizer

    monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", _recording)
    return calls


@pytest.mark.light
@pytest.mark.parametrize("setting", ["absent", "none", "zero", "explicit"])
@pytest.mark.parametrize("twist_average", [False, True])
def test_run_qe2jqmc_resolves_and_forwards_exp_tail_eps(monkeypatch, tmp_path, setting, twist_average):
    """run_qe2jqmc forwards the user default when the key is absent, None/0 disable, a value passes through."""
    from jqmc._setting import PGTO_EXP_TAIL_EPS_USER_DEFAULT
    from jqmc.jqmc_qe import run_qe2jqmc

    qe_params = {
        "qe_save_dir": str(_qe_save("Li_k_twist" if twist_average else "O-lsda_gamma")),
        "kind": "pgto+pw",
        "upf_paths": [],
        "ecutwfc": 10.0,
        "alpha_min": None,
        "twist_average": twist_average,
    }
    values = {"none": None, "zero": 0.0, "explicit": 5.0e-2}
    if setting != "absent":
        qe_params["exp_tail_eps"] = values[setting]
    expected = {"absent": PGTO_EXP_TAIL_EPS_USER_DEFAULT, "none": None, "zero": None, "explicit": 5.0e-2}[setting]
    calls = _abort_at_optimizer(monkeypatch)

    with pytest.raises(_StopAtOptimizer):
        run_qe2jqmc(qe_params, {"j1_parameter": None, "j2_parameter": 1.0}, str(tmp_path / "ham.h5"))

    assert len(calls) == 1
    assert calls[0]["exp_tail_eps"] == expected


@pytest.mark.light
@pytest.mark.parametrize("bad_eps", [-1.0e-2, 1.0, 2.0])
def test_run_qe2jqmc_rejects_out_of_range_exp_tail_eps(tmp_path, bad_eps):
    """A nonzero exp_tail_eps outside (0, 1) is rejected at the entry point."""
    from jqmc.jqmc_qe import run_qe2jqmc

    qe_params = {"qe_save_dir": str(_O_LSDA), "kind": "pgto+pw", "upf_paths": [], "ecutwfc": 10.0, "exp_tail_eps": bad_eps}
    with pytest.raises(ValueError, match="exp_tail_eps must satisfy"):
        run_qe2jqmc(qe_params, {"j1_parameter": None, "j2_parameter": 1.0}, str(tmp_path / "ham.h5"))


@pytest.mark.light
def test_run_qe2jqmc_default_deploys_bounded_basis_on_fixed_lsda_cutoff(monkeypatch, tmp_path):
    """With the default exp_tail_eps the fixed 10 Ry O LSDA conversion projects onto a basis inside the bound."""
    from jqmc import pgto_exponent_optimizer, pw_pgto_projection
    from jqmc._setting import PGTO_EXP_TAIL_EPS_USER_DEFAULT
    from jqmc.jqmc_qe import run_qe2jqmc

    original_optimizer = pgto_exponent_optimizer.optimize_pgto_exponents
    original_projector = pw_pgto_projection.project_pw_to_mixed
    results = []
    projected = []

    def _recording_optimizer(*args, **kwargs):
        results.append(original_optimizer(*args, **kwargs))
        return results[-1]

    def _recording_projector(reference, shells, *args, **kwargs):
        projected.append(shells)
        return original_projector(reference, shells, *args, **kwargs)

    monkeypatch.setattr(pgto_exponent_optimizer, "optimize_pgto_exponents", _recording_optimizer)
    monkeypatch.setattr(pw_pgto_projection, "project_pw_to_mixed", _recording_projector)
    out = tmp_path / "ham_default_bound.h5"
    run_qe2jqmc(
        {"qe_save_dir": str(_O_LSDA), "kind": "pgto+pw", "upf_paths": [], "ecutwfc": 10.0, "alpha_min": None},
        {"j1_parameter": None, "j2_parameter": 1.0},
        str(out),
    )

    assert out.is_file()
    assert len(results) == 1
    result = results[0]
    assert result.exp_tail_eps == PGTO_EXP_TAIL_EPS_USER_DEFAULT
    assert projected and all(shells is result.shells_per_atom for shells in projected)
    if PGTO_EXP_TAIL_EPS_USER_DEFAULT is not None:
        assert result.n_clipped == 2
        for shell in result.shells_per_atom[0]:
            bound = result.alpha_report[shell.angular_l]["alpha_max"]
            assert max(shell.exponents) <= bound * (1.0 + RTOL_PYTHON_FP64_TIGHT)


@pytest.mark.light
@pytest.mark.parametrize("cli_value", ["default", "0", "0.05"])
def test_jqmc_tool_qe_exp_tail_eps_option(monkeypatch, cli_value):
    """jqmc-tool qe convert-to passes --exp-tail-eps (default PGTO_EXP_TAIL_EPS_USER_DEFAULT) into qe_params."""
    from typer.testing import CliRunner

    from jqmc import jqmc_qe
    from jqmc._setting import PGTO_EXP_TAIL_EPS_USER_DEFAULT
    from jqmc.jqmc_tool import qe_app

    seen = []
    monkeypatch.setattr(jqmc_qe, "run_qe2jqmc", lambda qe_params, jastrow_params, hamiltonian_h5: seen.append(qe_params))
    argv = ["/nonexistent.save", "-k", "pgto+pw"]  # single-command app: no subcommand token
    if cli_value != "default":
        argv += ["--exp-tail-eps", cli_value]
    result = CliRunner().invoke(qe_app, argv)

    assert result.exit_code == 0, result.output
    expected = {"default": PGTO_EXP_TAIL_EPS_USER_DEFAULT, "0": 0.0, "0.05": 0.05}[cli_value]
    assert seen[0]["exp_tail_eps"] == expected


@pytest.mark.light
def test_qe_workflow_exp_tail_eps_command():
    """QE_Workflow omits --exp-tail-eps at the tool default and forwards other values (None as 0)."""
    from jqmc._setting import PGTO_EXP_TAIL_EPS_USER_DEFAULT
    from jqmc_workflow.qe_workflow import QE_Workflow

    default = QE_Workflow(qe_save_dir="/x", kind="pgto+pw")
    assert default.exp_tail_eps == PGTO_EXP_TAIL_EPS_USER_DEFAULT
    assert "--exp-tail-eps" not in default._build_command()
    assert default.configure()["exp_tail_eps"] == PGTO_EXP_TAIL_EPS_USER_DEFAULT

    for value, flag in ((1.0e-3, "0.001"), (0.0, "0.0"), (None, "0")):
        if value == PGTO_EXP_TAIL_EPS_USER_DEFAULT:
            continue
        cmd = QE_Workflow(qe_save_dir="/x", kind="pgto+pw", exp_tail_eps=value)._build_command()
        assert cmd[cmd.index("--exp-tail-eps") + 1] == flag


# --- coincident clipped seeds are spread below the bound ---


def _li_fixed_cutoff_case():
    from jqmc.pgto_exponent_optimizer import _wfc_g_max, alpha_max_for_tail
    from jqmc.pw_pgto_projection import alpha_min_from_gcut, filter_shells_by_alpha_min
    from jqmc.qe_wrapper import load_qe_reference, truncate_qe_reference_bands

    full = load_qe_reference(str(_LI_GAMMA))
    reference = truncate_qe_reference_bands(full, int(full.bands.occupied_band_indices().max()) + 1)
    g_cut = float(np.sqrt(10.0))
    floor = alpha_min_from_gcut(g_cut)
    shells = filter_shells_by_alpha_min(_upf_shells(reference, _LI_GAMMA / "Li.ccECP.upf"), floor)
    alpha_max = {angular_l: alpha_max_for_tail(angular_l, _wfc_g_max([reference]), 1.0e-2) for angular_l in (0, 1)}
    return reference, shells, g_cut, floor, alpha_max


@pytest.mark.light
def test_coincident_clipped_li_seed_is_spread_on_the_geometric_ladder():
    """Li ccECP s 190.08 and 5.938 both exceed alpha_max(0): they land on distinct ladder rungs, deterministically."""
    from jqmc.pgto_exponent_optimizer import _build_layout, _clip_seed_log_exponents

    reference, shells, _, floor, alpha_max = _li_fixed_cutoff_case()
    layout = _build_layout(reference, shells)
    z0 = np.log(layout.initial_exponents)
    z_ub = np.log(np.asarray([alpha_max[int(l)] for l in layout.uid_angular_l]))
    s_uids = layout.shell_param_ids[0][0]
    s_seed = np.exp(z0[list(s_uids)])
    assert int(np.count_nonzero(s_seed > alpha_max[0])) == 2

    z, violations = _clip_seed_log_exponents(z0, z_ub, layout, float(np.log(floor)))
    z_again, _ = _clip_seed_log_exponents(z0.copy(), z_ub, layout, float(np.log(floor)))

    assert np.array_equal(z, z_again)
    assert sorted(int(i) for i in violations) == sorted(uid for uid in s_uids if z0[uid] > z_ub[uid])
    top = float(np.log(alpha_max[0])) - PGTO_EXP_TAIL_CLIP_MARGIN
    step = float(np.log(PGTO_EXP_TAIL_CLIP_SPREAD_RATIO))
    by_seed = sorted(violations, key=lambda uid: -z0[uid])
    kept_s = [float(z0[uid]) for uid in s_uids if uid not in set(int(i) for i in violations)]
    # The top rung alpha_max = 5.22 lies within half a step of the unclipped s
    # exponent 3.866 and is skipped; the second and third rungs (2.61, 1.31)
    # are clear of every unclipped exponent (0.589, 3.866).
    assert min(abs(top - other) for other in kept_s) < 0.5 * step
    for rung in (1, 2):
        assert min(abs(top - rung * step - other) for other in kept_s) >= 0.5 * step
    assert z[by_seed[0]] == top - step
    assert z[by_seed[1]] == top - 2.0 * step
    for uids in layout.shell_param_ids[0]:
        shell_exponents = np.exp(z[list(uids)])
        assert len(set(shell_exponents.tolist())) == len(uids)
        assert np.all(z[list(uids)] <= z_ub[list(uids)])
        assert np.all(z[list(uids)] > np.log(floor))
    unclipped = [uid for uid in range(z0.size) if uid not in set(int(i) for i in violations)]
    assert np.array_equal(z[unclipped], z0[unclipped])
    # Equivalent atoms share the element-tied exponents.
    assert layout.shell_param_ids[0] == layout.shell_param_ids[1]


@pytest.mark.light
def test_spread_li_seed_is_nonsingular_within_the_condition_constraint(monkeypatch):
    """The spread Li seed passed to SLSQP and the deployed audit is well conditioned (the collapsed seed had cond ~6e16)."""
    from jqmc.pgto_exponent_optimizer import optimize_pgto_exponents

    reference, shells, g_cut, floor, alpha_max = _li_fixed_cutoff_case()
    deployed_calls = _spy_deployed(monkeypatch)
    svd_rcond = 1.0e-8

    result = optimize_pgto_exponents(
        [reference], shells, occ_only=True, svd_rcond=svd_rcond, g_cut=g_cut, exponent_floor=floor, exp_tail_eps=1.0e-2
    )

    assert result.n_clipped == 2
    assert np.isfinite(result.deployed_initial_condition)
    assert result.deployed_initial_condition < 1.0 / svd_rcond
    assert result.initial_condition < 1.0 / svd_rcond
    seed = deployed_calls[0]
    for atom_shells in seed:
        for shell in atom_shells:
            assert len(set(shell.exponents)) == len(shell.exponents)
            assert max(shell.exponents) <= alpha_max[shell.angular_l] * (1.0 + RTOL_PYTHON_FP64_TIGHT)


@pytest.mark.light
def test_spread_rung_next_to_an_unclipped_exponent_is_skipped():
    """A ladder rung within half a step of an unclipped exponent of the shell is skipped."""
    from jqmc.pgto_exponent_optimizer import _build_layout, _clip_seed_log_exponents, _wfc_g_max, alpha_max_for_tail

    reference = _one_gaussian_reference(1.7)
    alpha_max = alpha_max_for_tail(0, _wfc_g_max([reference]), 1.0e-2)
    step = float(np.log(PGTO_EXP_TAIL_CLIP_SPREAD_RATIO))
    top = float(np.log(alpha_max)) - PGTO_EXP_TAIL_CLIP_MARGIN
    unclipped = float(np.exp(top - step))  # sits exactly on the second rung
    layout = _build_layout(reference, [[_s_shell((unclipped, 9.0, 12.0))]])
    z0 = np.log(layout.initial_exponents)
    z_ub = np.full(z0.shape, float(np.log(alpha_max)))

    z, violations = _clip_seed_log_exponents(z0, z_ub, layout, None)

    assert sorted(int(i) for i in violations) == [1, 2]
    assert z[2] == top
    assert z[1] == top - 2.0 * step
    assert z[0] == z0[0]


@pytest.mark.light
def test_spreading_into_the_exponent_floor_raises():
    """When the ladder needed for coincident clipped seeds reaches exponent_floor, the optimiser refuses."""
    from jqmc.pgto_exponent_optimizer import PGTOExponentBoundsError, _wfc_g_max, alpha_max_for_tail, optimize_pgto_exponents

    reference = _one_gaussian_reference(1.7)
    alpha_max = alpha_max_for_tail(0, _wfc_g_max([reference]), 1.0e-2)
    # Second rung alpha_max / r sits below the floor, which is itself below alpha_max.
    floor = 0.75 * alpha_max
    assert alpha_max / PGTO_EXP_TAIL_CLIP_SPREAD_RATIO < floor < alpha_max
    shells = [[_s_shell((0.9 * alpha_max, 9.0, 12.0))]]

    with pytest.raises(PGTOExponentBoundsError, match="reaches the exponent floor"):
        optimize_pgto_exponents(reference, shells, exponent_floor=floor, exp_tail_eps=1.0e-2)


@pytest.mark.light
def test_manual_cutoff_propagates_infeasible_exponent_bounds(tmp_path):
    """A numeric cutoff retains the caller-visible ValueError contract."""
    from jqmc.jqmc_qe import run_qe2jqmc
    from jqmc.pgto_exponent_optimizer import PGTOExponentBoundsError

    output = tmp_path / "manual.h5"
    with pytest.raises(PGTOExponentBoundsError, match="reaches the exponent floor") as error:
        run_qe2jqmc(
            {"qe_save_dir": str(_LI_GAMMA), "kind": "pgto+pw", "upf_paths": [], "ecutwfc": 40.0},
            {},
            str(output),
        )
    assert isinstance(error.value, ValueError)
    assert not output.exists()
