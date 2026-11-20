"""Analytic MATRIX MIRROR: the Rust ``jqmc_rs`` c-axis parameter-gradient kernels ==
jqmc's XLA analytic derivatives, walked over the SAME generated fixture matrix the XLA
analytic-vs-``_auto`` route-fidelity matrix uses.

Where ``tests/test_jqmc_kernel/_generated_matrix_base.py`` asserts, per
generated cell, that the jqmc XLA *analytic* c-axis gradient equals its ``_auto`` autograd
twin, THIS module asserts, per cell x active parameter block x quantity, that the Rust
kernel equals that same XLA *analytic* gradient (extracted from the public
``jqmc.derivatives.compute_grad_{ln_Psi,e_L}(..., analytic=True)['c']`` pytree). Chaining
the two gives Rust == XLA-analytic == ``_auto`` -- the Rust-side leg of the
"all-grads-analytic" verification.

Cell universe + builders are IMPORTED (never re-implemented) from the XLA matrix module:
``_generate_matrix`` (the deterministic pairwise+3-way covering array over
AO x Jastrow x ECP x BC x geminal), ``_resolve`` / ``_build_ham_for_cell`` (the committed
trexio + QE-projection fixture builders). So every ham here flows the EXACT committed
fixture the XLA matrix walks -- no bench h5.

Scope of the Rust c-axis kernels (the ported ``test_param_grad.py`` /
``test_param_grad_el.py`` demonstrate the exact call signatures this module reuses). Each
(cell, block, quantity) is asserted where a Rust c-axis kernel serves that path. A missing
kernel first evaluates the XLA oracle and then fails through ``rust_port_gap``; it is never
skipped, xfailed, allowlisted, or treated as completion. Notable missing boundaries are:

  * ``Det`` O_k rides the periodic/twist-aware det-value gradient
    (``grad_ln_det_wrt_{mo_coeff,lambda}`` real, ``grad_ln_det_{mo,lambda}_complex`` twist),
    so it extends to molecular + real-pGTO/composite/plane-wave/B-spline (real) +
    complex-pGTO / composite (twist). The pure plane-wave leaf ingests through the
    ``OrbData`` pure-PW path (empty GTO block + PW child). The complex ``plane_wave_complex``
    / ``bspline_complex`` leaves have no ``_num_ao_kernel`` det kernel -> fails.
  * ``Det`` e_L uses the with-Jastrow, periodic-aware kinetic c-axis kernel
    (``grad_eL_wrt_{mo_coeff,lambda}`` -- verified periodic-aware for real pGTO) plus the
    ECP non-local c-axis add (``grad_eL_ecp_wrt_*``, molecular). Twist AE uses the complex
    kinetic kernel (``grad_T_{mo,lambda}_complex``), which is Jastrow-FREE, so it is
    asserted only on jnone twist cells. ECP-on-periodic / twist-withJ Det e_L is not a
    single ported kernel -> fails.
  * ``J1``/``J2``/``J3`` O_k are pure Jastrow-VALUE gradients (min-image aware in the value
    eval) -> asserted on every cell. Their e_L (kinetic) reduction is asserted on every
    MOLECULAR and REAL-GAMMA cell: ``dT/dc`` needs only the (basis-independent) Jastrow
    field ``f_c`` plus the determinant drift ``gD``, and ``gD`` rides the basis-agnostic
    ``orb_value_grad_lap``, which ingests pGTO, pure plane-wave, B-spline and composite
    leaves alike. At a TWIST the complex kinetic reduction is native-complex-pGTO only.
  * ``DTN`` O_k is asserted molecular-only (the Rust ``grad_ln_Psi_dtn_params`` uses
    molecular distances -- no periodic min-image -- so periodic/twist diverge: a port gap).
    ``DTN`` e_L rides the same drift-plus-field reduction as J1/J2, so it is asserted on
    molecular AE and on every REAL-GAMMA leaf (pGTO / plane-wave / B-spline / composite);
    at a twist it is native-complex-pGTO only.
  * ``Basis`` (geminal radial exp/coeff) O_k is molecular only (the Rust
    ``grad_ln_det_wrt_radial`` panics on the empty periodic-pGTO radial and has no complex
    kernel). ``Basis`` e_L uses the jnone-only ``grad_eL_wrt_radial`` -> molecular AE jnone.
  * ``AGPn`` geminal ``Det``: the XLA ``opt_geminal`` c-axis lands on the natural-orbital
    leaves (``lambda_diag`` real, ``lambda_unpaired``). The Rust ``*_lambda_diag*`` kernels
    map the dense ``lambda_matrix`` gradient back to those leaves (paired-block diagonal +
    unpaired columns -- the chain rule for
    ``lambda_matrix = hstack([diag(lambda_diag), lambda_unpaired])``), so AGPn rides the
    SAME det-value / kinetic support map as SD (its orbitals are MOs). All committed AGPn
    fixtures are closed-shell (``lambda_unpaired`` empty).

A cell/block whose Rust vs XLA compare diverges numerically FAILS per element; it is never
converted into a skip, which would turn any regression in the matrix into a green skip.
The comparison is unconditional, and every allowlist mapping must remain empty.

Tolerance comes from the dedicated XLA--Rust consistency policy. It has no
per-cell strict/medium level; pytest's full/mixed precision mode is the only
selector.
Bridge + jit-assemble caches are cleared per test (the inner-jit LRU is unbounded; walking
dozens of hams in one worker exhausts LLVM mmap).
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

import ast
import pathlib

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import jqmc_rs
from jqmc._precision import precision_policy_snapshot

from jqmc import derivatives
from jqmc.derivatives import bridge
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_PERIODIC_EL_KINETIC_FLOOR

from tests.test_jqmc_kernel._generated_matrix_base import (
    _build_ham_for_cell,
    _cell_id,
    _generate_matrix,
    _resolve,
    _split_cell,
)
from tests.test_rs_kernel._rs_fixtures import draw_well_conditioned_cfg
from tests.test_rs_kernel._gdd_rust_adapters import record_rust_matrix_fragment

# with_chdir: the autouse ``_isolate_cwd`` chdirs into a per-test tmp (xdist safety);
# committed fixtures load by absolute path so cwd is irrelevant. Every param row ALSO
# carries exactly one of light/heavy (attached per-cell below) -- the conftest enforces both.
pytestmark = pytest.mark.with_chdir


# --------------------------------------------------------------------------- #
# Active-block enumeration (token-only; no ham built at collection time). Mirrors the
# builder logic in the XLA matrix's ``_resolve``: the committed molecular builder couples
# J1+J2 whenever a Jastrow is attached; ``J3poly`` adds an active J3; ``DTN`` adds DTN. A
# radial ``Basis`` block exists iff the AO family carries GTO/pGTO radial parameters.
# --------------------------------------------------------------------------- #
# A radial ``Basis`` block exists iff the AO family carries GTO/pGTO radial parameters.
# ``composite`` (pgto(+)pw) DOES: its GTO child holds the exponents/coefficients, exactly
# what ``determinant._gto_leaves_of`` / ``_concat_gto_radial`` enumerate. It is
# in the population so the composite Basis cells are WALKED (real-Gamma composite is served
# by the Rust radial kernels; the twist composite still skips as a complex-radial
# port gap). Pure ``plane_wave`` / ``bspline`` leaves carry no GTO radial and stay excluded.
_RADIAL_AO = frozenset(
    {"cart_gto", "sphe_gto", "cart_pgto", "sphe_pgto", "cart_pgto_complex", "sphe_pgto_complex", "composite"}
)


def _active_blocks(cell) -> list[str]:
    ao, jas, _ecp, _bc, _gem = _split_cell(cell)
    blocks = ["Det"]
    if ao in _RADIAL_AO:
        blocks.append("Basis")
    if jas in ("J1", "J2", "J3poly", "DTN"):
        blocks += ["J1", "J2"]
    if jas == "J3poly":
        blocks.append("J3")
        # P09/P10 (simple GTO/pGTO J3) and P14/P15 (the GTO child of a
        # composite J3) are first-class analytical parameter leaves on main.
        # Keep them as a separate owner: ``J3`` covers only ``j3_matrix`` and
        # cannot be counted as evidence for the radial exponent/coeff leaves.
        if ao in _RADIAL_AO:
            blocks.append("J3Basis")
    if jas == "DTN":
        blocks.append("DTN")
    return blocks


_QUANTITIES = ("ln_Psi", "e_L")

# Block -> the single-block XLA optimization flag(s) that turn on exactly that c-axis block.
_BLOCK_FLAGS = {
    "Det": {"opt_geminal": True},
    "Basis": {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True},
    "J1": {"opt_J1_param": True},
    "J2": {"opt_J2_param": True},
    "J3": {"opt_J3_param": True},
    "J3Basis": {"opt_J3_basis_exp": True, "opt_J3_basis_coeff": True},
    "DTN": {"opt_JDTN_u_param": True, "opt_JDTN_chi_param": True, "opt_JDTN_f_param": True},
}

# (ao, bc) pairs whose det-VALUE c-axis kernel is served for an SD geminal (real
# ``grad_ln_det_wrt_mo_coeff`` / complex ``grad_ln_det_mo_complex``). The pure
# plane-wave leaf (real Gamma) ingests through the ``OrbData`` pure-PW path (empty
# GTO block + PW child), so its ``mo_coefficients`` O_k rides ``grad_ln_det_wrt_mo_coeff``
# just like the composite. The twist composite is served by the complex det kernel
# (verified). Still no det path: complex ``plane_wave_complex`` / ``bspline_complex``
# (their AO leaves expose no ``_num_ao_kernel`` for the complex geminal orb eval).
_DET_OK_SD = frozenset(
    {
        ("cart_gto", "mol"),
        ("sphe_gto", "mol"),
        ("cart_pgto", "pbc_gamma"),
        ("sphe_pgto", "pbc_gamma"),
        ("bspline", "pbc_gamma"),
        ("composite", "pbc_gamma"),
        ("plane_wave", "pbc_gamma"),
        ("composite", "twist"),
        ("cart_pgto_complex", "twist"),
        ("sphe_pgto_complex", "twist"),
        # The complex ``plane_wave_complex`` / ``bspline_complex`` twist leaves ingest
        # through ``ComplexOrb::from_py`` (pure-PW: empty GTO block + PW child;
        # pure-B-spline: ``AosBsplineComplexData`` child), so ``grad_ln_det_mo_complex``
        # (via ``ComplexOrb::ao``/``value``, whose ``ao_full`` owns the bspline/PW branch)
        # produces the mo_coefficients O_k with no per-leaf det kernel needed. Measured
        # Rust-vs-XLA over every committed twist cell: plane_wave |diff| <= 9.9e-12
        # (ref ~24), bspline <= 1.9e-12 (ref 1.0) -- fp floor, not a wiring gap.
        ("plane_wave_complex", "twist"),
        ("bspline_complex", "twist"),
    }
)
# Twist AE kinetic (jnone) Det e_L families.
_DET_OK_CPLX = frozenset({"cart_pgto_complex", "sphe_pgto_complex"})

# Complex-twist Jastrow / DTN kinetic dE_L/dc leaves. The reduction needs only the
# (basis-free) Jastrow field ``f_c`` plus the determinant drift ``gD``, and ``gD`` rides
# ``ComplexOrb::value_grad_lap``, which owns the whole complex leaf taxonomy (native
# complex pGTO, pure ``plane_wave_complex``, pure ``bspline_complex``, and the
# pgto(+){pw,bspline} composite). Measured across every committed SD / AGPn twist cell:
# max|Rust - XLA| <= 1.5e-12 (bspline / plane_wave / composite), i.e. fp floor.
#
# The DENSE-lambda AGP twist: a genuine complex-twist ``AGP_data`` carries a
# ``lambda_matrix`` with a LARGE imaginary part (its only committed cell
# ``composite__J2__AE__twist__AGP``: max|Im lambda| = 6.43 vs max|Re lambda| = 56.5). A
# geminal built from the Re-ONLY ``ComplexGeminal.lambda`` field drops Im(lambda), so ``gD``
# is wrong and dE_L/dc diverges O(1e-2) relative. ``ComplexGeminal`` therefore also stores
# the FULL complex ``lambda_c``, and every holomorphic geminal-construction seam
# (``geminal_w`` / ``complex_gem_forward`` / ``complex_geminal_reverse`` /
# ``grads_and_laplacian_ln_det_complex*`` / the complex stress base G) threads it, mirroring
# the det-VALUE O_k ``_clam`` variant. A real-pairing twist has Im == 0 so ``lambda_c`` is
# byte-identical to a ``Re(lambda)+0i`` cast; the dense-``lambda`` AGP twist carries
# Im(lambda) into ``gD``, dE_L/dc and the r/R/eps kinetic chains. ``Re`` is taken LAST by
# every consumer.
_JAS_KIN_TWIST_OK_AO = frozenset(
    {"cart_pgto_complex", "sphe_pgto_complex", "plane_wave_complex", "bspline_complex", "composite"}
)


def _jas_kin_twist_okay(ao: str, gk: str) -> bool:
    return ao in _JAS_KIN_TWIST_OK_AO


# (No J3 support set: the Rust j_matrix c-axis kernels serve EVERY ``_J3_KIND`` --
# single-leaf GTO/pGTO and MO-wrapped composite / plane-wave / B-spline, real and complex.
# The J3 block below therefore has no leaf gate; see its comment for why it is basis-free.)


def _det_okay(ao: str, bc: str, gk: str, gem_token: str) -> bool:
    """Whether the Rust det-value c-axis kernel serves this representation and AO/BC leaf."""
    if gk == "AGP_data":
        if gem_token == "AGP_MO":
            # The lambda derivative is basis-agnostic below orbital evaluation.  An MO-AGP
            # therefore has the same orbital-ingestion reach as the SD fixture from which its
            # explicit MOs were built, while retaining AGP's dense lambda_matrix parameter leaf.
            return (ao, bc) in _DET_OK_SD
        # dense AO-AGP `grad_ln_det_wrt_lambda` (`param_grad::grad_ln_det_lambda`) is
        # basis-agnostic BELOW the orbital eval: it consumes `OrbData::compute(r)` and then
        # does pure geminal matrix algebra (G = A*lambda_p*B, G^-1, pull back to lambda).
        # `OrbData` ingests the real-Gamma composite (pgto child + PW child, concatenated in
        # `gto_first` child order) exactly as it does for the SD `grad_ln_det_wrt_mo_coeff`
        # leg, which `_DET_OK_SD` already serves at ("composite", "pbc_gamma"). Measured on
        # ``composite__J1__ECP__pbc_gamma__AGP``: max|Rust - XLA| = 8.45e-11 against a
        # max|XLA| of 4.24e-02, i.e. 3 orders inside the medium ``det_eval`` atol (1e-7).
        # The residual is UNIFORM across the pGTO rows (0-9, ~1e-11) and the PW rows
        # (10-42, ~1e-10) -- so it is ordinary G^-1-scale fp accumulation, NOT a dropped or
        # LEPSBAS-screened AO block (that failure mode zeroes one child's rows and leaves
        # the other's clean; measured row profile shows neither).
        # The complex-twist dense AO-AGP composite leaf rides ``grad_ln_det_lambda_complex``
        # (holomorphic ``dln|det|/dlambda`` over the ``ComplexGeminal``, ``Re`` last); its
        # ``ComplexOrb`` ingests the composite (pgto child + complex-PW/B-spline child) the
        # same way the real-Gamma AGP composite does. Measured on
        # ``composite__J2__AE__twist__AGP``: max|Rust - XLA| = 5.2e-17 (ref 2.17e-2), bit level.
        return (
            (bc == "mol" and ao in ("cart_gto", "sphe_gto"))
            or (ao, bc) == ("composite", "pbc_gamma")
            or (ao, bc) == ("composite", "twist")
        )
    return (ao, bc) in _DET_OK_SD


# Tolerance zones (mirrors the ported per-kernel rs tests).
_Z_DET = ("ao_eval", "det_eval")
_Z_KIN = ("ao_eval", "mo_eval", "mo_grad", "mo_lap", "det_grad_lap", "wf_kinetic", "jastrow_grad_lap")
_Z_JAS = ("jastrow_eval",)


# Periodic native e_L kinetic-derivative fp-amplification floor (ABSOLUTE).
#
# The AGPn Det e_L c-axis (dE_L/d lambda_diag) is a STRUCTURAL ZERO: the XLA-analytic
# reference is exactly 0.0 on every committed AGPn fixture (molecular, real-Gamma pGTO,
# plane-wave, B-spline) -- the local kinetic energy is invariant under the natural-orbital
# lambda_diag scaling, so its lambda_diag gradient vanishes identically. The pGTO periodic
# path reaches that zero to ~3e-17, but the PW / B-spline periodic kinetic derivative forms
# the large-|G|^2 Laplacian cancellation EXPLICITLY and, amplified by the 1/Psi geminal-
# inverse contraction, lands at ~1e-8 .. 1.2e-7 -- platform-dependent, straddling the 1e-7
# medium ``wf_kinetic`` atol (3.9e-8 on one machine; another machine's fp rounding
# reaches 1.19e-7 on the same fixed seed). This is the SAME uncloseable periodic-kinetic fp
# floor the native periodic e_L carries vs the XLA forward graph ("up to ~2e-7" by a
# term-by-term decomposition), NOT a correctness gap: the value agrees with zero
# to 7-8 significant digits. Periodic e_L comparisons therefore take it as an absolute floor.
# It sits 2 orders below ATOL_FLOOR_MEDIUM (2e-5) and ~5 below the ~3.5e-2-per-config signal
# of every real periodic bug this suite catches (a dropped image / wrong metric is an O(1)
# relative error, governed by rtol against the O(1)-valued non-AGPn e_L cells), so this floor
# is inert wherever the reference is non-zero and bug detection is unaffected.
def _cell_params():
    params = []
    for cell, heavy in _generate_matrix():
        mark = pytest.mark.heavy if heavy else pytest.mark.light
        cid = _cell_id(cell)
        for block in _active_blocks(cell):
            for q in _QUANTITIES:
                params.append(pytest.param(tuple(cell), block, q, marks=mark, id=f"{cid}__{block}__{q}"))
    return params


_PARAMS = _cell_params()


# --------------------------------------------------------------------------- #
# Config + comparison helpers (identical style to the ported test_param_grad*.py).
# --------------------------------------------------------------------------- #
def _config(ham, seed: int = 1):
    """A deterministic XLA-conditioned config, selected before backend parity."""
    return draw_well_conditioned_cfg(ham, seed=seed, scale=1.0, return_metadata=True)


def _ao_leaf(orb):
    return orb.aos_data if type(orb).__name__ == "MOs_data" else orb


def _is_complex(ham) -> bool:
    # A composite wrapper (``AOs_composite_data``) carries no ``complex`` in its own class
    # name -- its complexity lives in the children (complex-pGTO + complex-PW at a twist), so
    # recurse into them; a flat leaf reports through its own class name.
    leaf = _ao_leaf(ham.wavefunction_data.geminal_data.orb_data_up_spin)
    if hasattr(leaf, "children"):
        return any("complex" in type(c).__name__ for c in leaf.children)
    return "complex" in type(leaf).__name__


def _gem_kind(ham) -> str:
    return type(ham.wavefunction_data.geminal_data).__name__  # SD_data / AGP_data / AGPn_data


# (cell-id, block, quantity) -> one-line reason, for a Rust-vs-XLA compare that RUNS but
# genuinely DIVERGES.  Entries here (and ONLY here) are permitted to fail the
# per-element compare; everything else asserts.  The dict is EMPTY and
# ``test_parameter_derivative_matrix_has_no_green_escape_hatch`` requires it to stay so.
#
# A divergence whose "measured mechanism" looks like an unavoidable fp floor still needs
# a term-by-term decomposition before it is frozen: for example, raw (un-home-cell-wrapped)
# electron coordinates fed to the AO stack while the contracted geminal is wrapped, or a
# runtime ``min_exp*r^2 > LEPSBAS`` screen that XLA does not apply (jqmc truncates the image
# SET once at build time and then evaluates every survivor exactly), each produce
# periodic-Gamma ECP (Det, e_L) differences of O(1e-6) that read as a cancellation floor
# but are formula errors.
_DIVERGENCE_ALLOWLIST: dict[tuple[str, str, str], str] = {}

# Staleness bookkeeping for the freeze above.
#
# A fixed cell's entry becomes inert, and an inert freeze is the dead-guard failure mode
# (the same as a stale fallback grep or a stale signature allowlist), so staleness is
# checked.
#
# ``_ALLOWLIST_WALKED`` = keys this worker actually compared with a freeze in force;
# ``_ALLOWLIST_FIRED``  = those whose comparison really did breach tolerance.
# WALKED minus FIRED = the freeze is no longer earning its place. Both are per-worker,
# which is what makes the check work under xdist without cross-process aggregation: a key
# is walked and fires (or not) inside one worker, so the verdict is complete locally.
_ALLOWLIST_WALKED: set[tuple[str, str, str]] = set()
_ALLOWLIST_FIRED: set[tuple[str, str, str]] = set()


# --------------------------------------------------------------------------- #
# XLA-analytic ORACLE gaps.
#
# When the XLA-analytic ORACLE itself raises ``NotImplementedError``, ``xla()`` records
# the gap and FAILS the cell -- never a skip, which would let a regression of the XLA
# analytic c-axis into raising (an "assemble gate silently drops a block" failure) remove
# the Rust-vs-XLA leg of the parity chain silently. This is the same policy as the curated
# harness ``tests/test_jqmc_kernel/test_analytic_grad_matches_autograd_curated.py``
# (``_record_gap`` + ``pytest.fail``).
#
# No matrix cell reaches this branch, so the allowlist below is EMPTY -- it is a trap for
# a future regression, not a carve-out. An entry may be added ONLY with a measured,
# one-line mechanism (same discipline as ``_DIVERGENCE_ALLOWLIST``).
# --------------------------------------------------------------------------- #
_XLA_ORACLE_GAP_ALLOWLIST: dict[tuple[str, str, str], str] = {}


@pytest.mark.light
def test_parameter_derivative_matrix_has_no_green_escape_hatch():
    """The production parity matrix may neither skip/xfail cells nor freeze gaps."""
    tree = ast.parse(pathlib.Path(__file__).read_text())
    escapes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr in {"skip", "xfail"}
    ]
    assert not escapes, f"parameter-derivative parity cells can become green at lines {escapes}"
    assert not _DIVERGENCE_ALLOWLIST, "numerical parity gaps must fail, not be allowlisted"
    assert not _XLA_ORACLE_GAP_ALLOWLIST, "XLA oracle gaps must fail, not be allowlisted"


# GAP inventory, mirroring the curated harness: every (cell, block, quantity) whose
# XLA-analytic oracle raised, reported once at session teardown so a full walk names the
# remaining analytic work in one place instead of one failure line at a time.
_XLA_ORACLE_GAPS: list[tuple[str, str, str, str]] = []


def _record_xla_gap(key: tuple[str, str, str], why: str) -> None:
    _XLA_ORACLE_GAPS.append((*key, why))


@pytest.fixture(scope="session", autouse=True)
def _divergence_allowlist_staleness_gate():
    """FAIL the session if a frozen divergence was walked but never actually diverged.

    The freeze exists to tolerate a MEASURED discrepancy. Once the underlying defect is
    fixed the entry stops doing anything -- and, left alone, stays there forever asserting
    a divergence that no longer exists.

    Per-worker by construction (a key is walked and fires, or not, inside one worker), so
    this needs no cross-process aggregation under xdist. Only keys this worker actually
    walked are judged; a key another worker owns is simply not in this worker's WALKED set.
    """
    yield
    stale = sorted(_ALLOWLIST_WALKED - _ALLOWLIST_FIRED)
    if stale:
        detail = "\n  ".join(
            f"{cid} {blk}/{qty} :: {_DIVERGENCE_ALLOWLIST.get((cid, blk, qty), '?')}" for cid, blk, qty in stale
        )
        pytest.fail(
            "STALE _DIVERGENCE_ALLOWLIST entries -- these were compared and did NOT diverge, "
            "so the freeze is no longer earning its place. Delete them (the defect is fixed):\n  " + detail
        )


@pytest.fixture(scope="session", autouse=True)
def _xla_gap_inventory_report():
    yield
    if _XLA_ORACLE_GAPS:
        lines = [
            "",
            "=" * 78,
            "XLA-analytic c-axis ORACLE gap inventory -- (cell, block, quantity) raising",
            "NotImplementedError from the jqmc analytic reference (this xdist worker):",
            "-" * 78,
        ]
        for cid, blk, qty, why in sorted(_XLA_ORACLE_GAPS):
            lines.append(f"  GAP: {cid:44s} {blk:6s} {qty:7s} :: {why}")
        lines.append("=" * 78)
        print("\n".join(lines))


# The (cell, block, quantity) under test, set once at the top of the test body so the
# many ``_assert_close`` call sites below do not each have to re-thread it.
_CURRENT: dict = {}


def _allowlisted() -> str | None:
    return _DIVERGENCE_ALLOWLIST.get((_CURRENT.get("cell_id"), _CURRENT.get("block"), _CURRENT.get("quantity")))


def _assert_dtn_liveness(d):
    """Guard against the fixture's use_u/use_chi/use_f flags going all-False,
    which would make every ``if d.use_*:`` branch below a silent no-op and the
    whole test vacuously pass with zero blocks actually compared (same pattern
    as ``_assert_dtn_liveness`` in test_param_grad*.py).
    """
    assert d.use_u or d.use_chi or d.use_f, (
        "DTN fixture has use_u/use_chi/use_f all False -- no block was compared, this test would otherwise pass vacuously"
    )


def _assert_close(pairs, zones, *, expect_nonempty: bool = True):
    """``pairs`` = list of (rust, xla, label): assert Rust == XLA-analytic PER ELEMENT.

    A shape mismatch or a tolerance breach FAILS (never a skip, which would report a
    green SKIP for a regression anywhere in the ~600-cell c-axis matrix); a genuinely
    known divergence must be frozen in ``_DIVERGENCE_ALLOWLIST`` with a reason (empty,
    see above).

    The criterion is per element: a ``d <= atol or d <= rtol*max|b|`` GLOBAL max norm
    would let one large component's magnitude set the relative budget for every other,
    masking a small component's large relative error. ``assert_allclose`` applies
    ``atol + rtol*|b|`` PER ELEMENT -- the suite standard.
    """
    # A c-axis result is a propagated aggregate, not a leaf-kernel value: the
    # fp32 AO/Jastrow leaves flow through a geminal inverse and kinetic reverse.
    # Their libm/GEMM reduction order changes across supported JAX versions and
    # is legitimately amplified even for the XLA-only conditioned walkers.
    # Keep leaf and precision-contract parity strict; use the one static mixed
    # aggregate envelope here.  In full mode ``medium`` resolves to the exact
    # same fp64 pair as strict.
    atol, rtol = get_tolerance_xla_rust_consistency(zones, level="medium")
    # Periodic native e_L kinetic derivatives carry the 1/Psi fp-amplification floor
    # (see ATOL_PERIODIC_EL_KINETIC_FLOOR): take it as an ABSOLUTE floor so a structural
    # zero's PW / B-spline Laplacian-cancellation noise (~1.2e-7 on one CPU) is not judged
    # against the tighter fp64 medium ``wf_kinetic`` atol (1e-7). Molecular e_L (single image,
    # no 1/Psi amplification) and the det-value ``ln_Psi`` c-axis (no kinetic term) keep the
    # unmodified zone atol; the floor is inert against every non-zero reference (rtol-bound).
    if _CURRENT.get("quantity") == "e_L" and _CURRENT.get("bc") not in (None, "mol"):
        atol = max(atol, ATOL_PERIODIC_EL_KINETIC_FLOOR)
    # A caller that reached here to compare something but handed an EMPTY list
    # (e.g. a DTN block whose ``if d.use_*:`` guards were all False) would otherwise
    # loop over nothing and pass vacuously. Every real call site expects at least one
    # comparison; the lone opt-out is ``expect_nonempty=False``.
    if expect_nonempty and not pairs:
        raise AssertionError(
            f"_assert_close reached with an EMPTY pairs list for "
            f"{(_CURRENT.get('cell_id'), _CURRENT.get('block'), _CURRENT.get('quantity'))} -- "
            "no comparison was made, so this cell would pass vacuously. If empty is genuinely "
            "expected here, pass expect_nonempty=False; otherwise the block-building guards dropped every pair."
        )
    for rust, xla, label in pairs:
        a = np.asarray(rust)
        b = np.asarray(xla)
        assert a.shape == b.shape, f"{label}: Rust shape {a.shape} vs XLA shape {b.shape}"
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=f"Rust vs XLA-analytic c-axis: {label}")
    record_rust_matrix_fragment(
        cell=_CURRENT["cell"],
        quantity=_CURRENT["quantity"],
        axis="c",
        component_id=_CURRENT["block"],
        pytest_node_id="tests/test_rs_kernel/test_rs_analytic_matrix.py::test_rs_analytic_matrix",
        ham=_CURRENT["ham"],
        r_up=_CURRENT["r_up"],
        r_dn=_CURRENT["r_dn"],
        compared_pairs=tuple((rust, xla) for rust, xla, _label in pairs),
        zones=tuple(zones),
        level="medium",
        atol=atol,
        rtol=rtol,
        generation_attempts=_CURRENT["generation_attempts"],
    )


# --------------------------------------------------------------------------- #
# XLA-analytic oracle: the c-axis gradient leaf(s) for one block/quantity.
# --------------------------------------------------------------------------- #
def _xla_c(ham, r_up, r_dn, block, quantity):
    flags = _BLOCK_FLAGS[block]
    if quantity == "ln_Psi":
        wf = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
    else:
        wf = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
    return wf


def _radial_of(orb):
    # Composite (pgto(+)pw): the radial DOF lives on the GTO child (child order,
    # matching ``determinant._concat_gto_radial``). A bare leaf carries ``radial`` itself.
    leaf = _ao_leaf(orb)
    if hasattr(leaf, "children"):
        for c in leaf.children:
            if hasattr(c, "radial"):
                return c.radial
    return leaf.radial


# --------------------------------------------------------------------------- #
# ECP non-local mesh, matching the global production quadrature default.
# --------------------------------------------------------------------------- #
def _one_move_mesh():
    from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

    w = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    gp = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)  # RT == eye
    return w, gp


# --------------------------------------------------------------------------- #
# The single per-(cell, block, quantity) route-fidelity assertion.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clear_caches():
    """Per-test teardown: drop the UNBOUNDED bridge inner-jit LRU + the public-entry
    ``_jitted_assemble`` LRU + the XLA compiled programs, so walking dozens of distinct
    hams in one worker does not exhaust LLVM mmap (measured).
    """
    yield
    bridge.cache_clear()
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


@pytest.mark.parametrize("cell, block, quantity", _PARAMS)
def test_rs_analytic_matrix(cell, block, quantity):
    _CURRENT.clear()
    _CURRENT.update(cell=tuple(cell), cell_id=_cell_id(cell), block=block, quantity=quantity)
    build = _resolve(cell)
    assert build is not None, f"cell {_cell_id(cell)} has no committed builder (all matrix cells should)"
    stem, with_jastrow, spec, gem = build
    ham = _build_ham_for_cell(stem, with_jastrow, spec, gem)
    r_up, r_dn, generation = _config(ham)
    _CURRENT.update(ham=ham, r_up=r_up, r_dn=r_dn, generation_attempts=generation["attempts"])

    ao, _jas, _ecp_tok, bc, gem_tok = _split_cell(cell)
    _CURRENT["bc"] = bc  # consumed by _assert_close for the periodic e_L kinetic atol floor
    cx = _is_complex(ham)
    ecp = bool(getattr(ham.coulomb_potential_data, "ecp_flag", False))
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    gk = _gem_kind(ham)
    # XLA-analytic oracle, evaluated LAZILY (and once): only the asserting branches touch it,
    # so a skipped (documented-gap) path never pays the XLA compile.
    #
    # A ``NotImplementedError`` from the XLA-analytic reference is recorded in the GAP
    # inventory and FAILED, matching the curated harness
    # (``test_analytic_grad_matches_autograd_curated.py``) -- an analytic reference that
    # cannot produce the c-axis is a regression, never a green cell. Only a
    # statically-frozen, measured entry in ``_XLA_ORACLE_GAP_ALLOWLIST`` may skip; that dict
    # is empty (no cell reaches this branch).
    _cache: dict = {}

    def xla():
        if "v" not in _cache:
            try:
                _cache["v"] = _xla_c(ham, r_up, r_dn, block, quantity)
            except NotImplementedError as e:
                why = str(e).splitlines()[0]
                key = (_CURRENT["cell_id"], block, quantity)
                _record_xla_gap(key, why)
                pytest.fail(
                    f"XLA-ORACLE GAP {key[0]} {block}/{quantity}: the jqmc XLA-analytic c-axis "
                    f"raised NotImplementedError -- {why}. This is the parity chain's reference "
                    "leg; without it the Rust kernel is compared against nothing. Implement it; "
                    "skip/xfail/allowlist escapes are forbidden."
                )
        return _cache["v"]

    def rust_port_gap(message: str):
        """Evaluate the XLA reference before reporting a missing Rust comparison leg."""
        xla()
        pytest.fail(f"Rust port gap after XLA oracle evaluation: {message}")

    # ============================ Det ============================
    if block == "Det":
        if gk == "AGPn_data":
            # AGPn ``opt_geminal`` c-axis lands on the natural-orbital leaves
            # (``lambda_diag`` real, ``lambda_unpaired``). The Rust det/kinetic lambda
            # kernels expose the dense ``lambda_matrix`` gradient; the ``*_lambda_diag*``
            # kernels map it back to those leaves (paired-block DIAGONAL + unpaired
            # columns -- the exact chain rule for
            # ``lambda_matrix = hstack([diag(lambda_diag), lambda_unpaired])``). AGPn
            # rides the SAME det-value support map as SD (its orbitals are MOs). All the
            # committed AGPn fixtures are closed-shell, so ``lambda_unpaired`` is empty.
            lu_active = np.asarray(gd.lambda_unpaired).size > 0

            def _agpn_close(d, u, tag, zones):
                pr = [(d, xla().geminal_data.lambda_diag, f"AGPn/{tag}/diag")]
                if lu_active:
                    pr.append((u, xla().geminal_data.lambda_unpaired, f"AGPn/{tag}/unpaired"))
                return _assert_close(pr, zones)

            if quantity == "ln_Psi":
                # The natural-orbital det-value c-axis needs only the ORBITAL VALUES
                # (``dln|det|/d lambda_diag[k] = Re(orb_up[k] . M_p . orb_dn[k])`` --
                # ``geminal.dln_det_dlambda_bare``), never the ``mo_coefficients``
                # gradient the SD leg rides. So it does NOT inherit the SD det-value
                # support map: every committed AGPn leaf -- incl. the complex
                # ``plane_wave_complex`` / ``bspline_complex`` twists, which the
                # ``ComplexOrb`` pure-PW / pure-B-spline ingestion serves with a 0-AO
                # GTO placeholder -- flows the same two kernels. An unported leaf
                # raises out of Rust and the cell fails, which is the honest signal.
                d, u = (
                    jqmc_rs.grad_ln_det_lambda_diag_complex(gd, r_up, r_dn)
                    if cx
                    else jqmc_rs.grad_ln_det_wrt_lambda_diag(gd, r_up, r_dn)
                )
                return _agpn_close(d, u, "lnPsi", _Z_DET)

            # AGPn Det e_L: same reachability as the SD lambda kinetic c-axis, mapped to
            # the lambda_diag leaf, plus the natural-orbital ECP non-local c-axis add
            # (``grad_eL_ecp_wrt_lambda_diag*`` -- the streamed weighted-mesh O_k reduce
            # that ``one_move_mesh.fast_supported_c`` forces for a ``lambda_diag`` geminal).
            def _ecp_add(d, u, real: bool):
                fn = jqmc_rs.grad_eL_ecp_wrt_lambda_diag if real else jqmc_rs.grad_eL_ecp_wrt_lambda_diag_complex
                ed, eu = fn(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                return np.asarray(d) + np.asarray(ed), np.asarray(u) + np.asarray(eu)

            if bc == "mol" or (
                bc == "pbc_gamma" and not cx and ao in ("cart_pgto", "sphe_pgto", "composite", "plane_wave", "bspline")
            ):
                # grad_t_lambda rides the periodic-aware orbital value/grad/lap eval
                # (image-folded pGTO, PW cos/sin, or B-spline) -> lambda_diag reduction.
                d, u = jqmc_rs.grad_eL_wrt_lambda_diag(gd, jd, r_up, r_dn)
                if ecp:
                    d, u = _ecp_add(d, u, True)
                return _agpn_close(d, u, "eL", _Z_KIN)
            if bc == "twist" and cx:
                # Complex-twist kinetic c-axis WITH Jastrow: ``J`` is real and
                # lambda-independent, so it enters only through the drift cross term
                # ``-(gD + gJ) . d gD/dlambda``; the chain stays HOLOMORPHIC and ``Re``
                # is taken LAST, on the reduced natural-orbital leaves. jnone reduces to
                # the Jastrow-free ``grad_T_lambda_diag_complex``.
                d, u = jqmc_rs.grad_eL_wrt_lambda_diag_complex(gd, jd, r_up, r_dn)
                if ecp:
                    d, u = _ecp_add(d, u, False)
                return _agpn_close(d, u, "eL", _Z_KIN)
            rust_port_gap(
                f"AGPn Det e_L on (bc={bc}, ao={ao!r}, withJ={jd.jastrow_one_body_data is not None}): "
                "not a single ported natural-orbital kinetic c-axis kernel"
            )
        if quantity == "ln_Psi":
            if not _det_okay(ao, bc, gk, gem_tok):
                rust_port_gap(
                    f"Det O_k: no ported Rust det-value c-axis kernel for (ao={ao!r}, bc={bc!r}, gem={gk}) "
                    "(plane-wave / bspline_complex lack _num_ao_kernel; twist/AGP composite ingestion unported)"
                )
            if gk == "SD_data":
                gu, gdn = (
                    jqmc_rs.grad_ln_det_mo_complex(gd, r_up, r_dn) if cx else jqmc_rs.grad_ln_det_wrt_mo_coeff(gd, r_up, r_dn)
                )
                pairs = [
                    (gu, xla().geminal_data.orb_data_up_spin.mo_coefficients, "Det/lnPsi/mo_up"),
                    (gdn, xla().geminal_data.orb_data_dn_spin.mo_coefficients, "Det/lnPsi/mo_dn"),
                ]
            else:  # AGP_data (dense lambda in AO or MO representation)
                rz = (
                    jqmc_rs.grad_ln_det_lambda_complex(gd, r_up, r_dn) if cx else jqmc_rs.grad_ln_det_wrt_lambda(gd, r_up, r_dn)
                )
                pairs = [(rz, xla().geminal_data.lambda_matrix, "Det/lnPsi/lambda")]
            return _assert_close(pairs, _Z_DET)

        # Det e_L
        if bc == "mol":
            # withJ + ECP-aware molecular kinetic c-axis (grad_eL_wrt_*), plus ECP non-local add.
            if gk == "SD_data":
                au, ad = jqmc_rs.grad_eL_wrt_mo_coeff(gd, jd, r_up, r_dn)
                au, ad = np.asarray(au), np.asarray(ad)
                if ecp:
                    eu, ed = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                    au = au + np.asarray(eu)
                    ad = ad + np.asarray(ed)
                pairs = [
                    (au, xla().geminal_data.orb_data_up_spin.mo_coefficients, "Det/eL/mo_up"),
                    (ad, xla().geminal_data.orb_data_dn_spin.mo_coefficients, "Det/eL/mo_dn"),
                ]
            else:  # AGP_data
                rz = np.asarray(jqmc_rs.grad_eL_wrt_lambda(gd, jd, r_up, r_dn))
                if ecp:
                    rz = rz + np.asarray(
                        jqmc_rs.grad_eL_ecp_wrt_lambda(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                    )
                pairs = [(rz, xla().geminal_data.lambda_matrix, "Det/eL/lambda")]
            return _assert_close(pairs, _Z_KIN)

        if bc == "pbc_gamma" and not cx and ao in ("cart_pgto", "sphe_pgto", "composite", "plane_wave", "bspline"):
            # grad_eL_wrt_{mo,lambda} is periodic-aware for the Det block on real pGTO
            # AND on the composite / pure plane-wave / B-SPLINE leaves (grad_t_mo contracts
            # the FULL AO stack via aos_stack_value_grad_lap, whose B-spline branch returns
            # the spline value/grad/lap stack, and the kinetic reverse rides
            # `orb_value_grad_lap`, whose B-spline branch has served the already-ungated
            # AGPn lambda_diag periodic path); the ECP non-local add rides grad_ln_det_mo
            # (basis-agnostic) on the min-image mesh.
            #
            # B-spline note (measured, not a bare floor): every committed periodic B-spline
            # fixture has a SQUARE mo_coefficients with num_mo == num_electron (4x4 / 1x1),
            # so the occupied space is the whole orbital space and e_L is EXACTLY invariant
            # under GL(C) -> dE_L/dC is a STRUCTURAL ZERO. Both legs reach it as noise of
            # the same size (measured max|XLA| = 3.9e-8..1.7e-7 vs max|Rust| = 1.0e-8..1.5e-7),
            # and the residual tracks the fixture's GEMINAL CONDITIONING, not the leaf: the
            # 4x4 B-spline cells have cond(G) = 1.89e5 and land at 6.6e-8..2.15e-7, while
            # `bspline__J1__ECP__pbc_gamma__SD` -- the SAME Rust code path on a 1x1 geminal,
            # cond(G) = 1.0 -- agrees to 6.6e-17, i.e. bit level. A wiring/convention error
            # would be O(1) relative and would NOT vanish with the conditioning, so this is
            # the documented 1/Psi kinetic fp amplification (ATOL_PERIODIC_EL_KINETIC_FLOOR),
            # not a port gap.
            if gk == "SD_data":
                au, ad = jqmc_rs.grad_eL_wrt_mo_coeff(gd, jd, r_up, r_dn)
                au, ad = np.asarray(au), np.asarray(ad)
                if ecp:
                    eu, ed = jqmc_rs.grad_eL_ecp_wrt_mo_coeff(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                    au = au + np.asarray(eu)
                    ad = ad + np.asarray(ed)
                pairs = [
                    (au, xla().geminal_data.orb_data_up_spin.mo_coefficients, "Det/eL/mo_up"),
                    (ad, xla().geminal_data.orb_data_dn_spin.mo_coefficients, "Det/eL/mo_dn"),
                ]
            else:
                rz = np.asarray(jqmc_rs.grad_eL_wrt_lambda(gd, jd, r_up, r_dn))
                if ecp:
                    rz = rz + np.asarray(
                        jqmc_rs.grad_eL_ecp_wrt_lambda(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                    )
                pairs = [(rz, xla().geminal_data.lambda_matrix, "Det/eL/lambda")]
            return _assert_close(pairs, _Z_KIN)

        if bc == "twist" and cx:
            # Complex-twist SD Det e_L (AGPn was handled by its own branch above). The
            # kinetic dE_L/dMO rides the complex MO kinetic c-axis: jnone
            # ``grad_T_mo_complex`` or, with a Jastrow, ``grad_T_mo_withJ_complex`` (whose
            # drift ``gj`` uses the Re-projected COMPLEX J3 leaf -- ``twist_gj`` +
            # ``from_py_without_three_body`` -- never an eager real ``OrbData``, which would
            # panic on a complex composite/PW/B-spline J3 leaf). Both ride
            # ``ComplexOrb::value_grad_lap`` -> ``compute_aos_vgl_tuple``, which owns the full
            # complex leaf taxonomy (native pGTO, pure plane_wave_complex, pure
            # bspline_complex, and the pgto(+){pw,bspline} composite). For an ECP twist the
            # non-local c-axis adds ``grad_eL_ecp_wrt_mo_coeff_complex`` (complex mesh, weight
            # ``Re(det_amp*jr)`` projected as a PRODUCT). The dense-lambda AGP twist kinetic
            # c-axis is unported (no ``grad_T_lambda_withJ_complex``) and fails. Measured
            # Rust-vs-XLA over every committed SD twist Det e_L cell: |diff| <= 4.4e-8 against
            # references O(1e0 .. 1e3) (rel <= 1.7e-9); the square-MO bspline cells are
            # structural zeros at ~1e-9 (ref ~1e-9), inside ATOL_PERIODIC_EL_KINETIC_FLOOR.
            if not _jas_kin_twist_okay(ao, gk):
                rust_port_gap(
                    f"Det e_L on twist (ao={ao!r}, gem={gk}, ecp={ecp}, "
                    f"withJ={jd.jastrow_one_body_data is not None}): the complex twist kinetic "
                    "lambda/MO c-axis serves this leaf only via _JAS_KIN_TWIST_OK_AO"
                )
            if gk == "AGP_data":
                # Dense-lambda AO-AGP twist Det e_L rides the HOLOMORPHIC complex dT/dlambda
                # (``grad_T_lambda{,_withJ}_complex``), threaded with the FULL complex
                # ``lambda_c`` so Im(lambda) reaches the drift; ``Re`` is taken last (the SR/LM
                # complex-DOF convention), and the un-projected complex derivative is compared
                # against the complex ``lambda_matrix`` DOF -- the kinetic twin of the det-VALUE
                # ``grad_ln_det_lambda_complex`` O_k that ``_det_okay`` already serves here.
                if jd.jastrow_one_body_data is None and jd.jastrow_two_body_data is None:
                    rz = np.asarray(jqmc_rs.grad_T_lambda_complex(gd, r_up, r_dn))
                else:
                    rz = np.asarray(jqmc_rs.grad_T_lambda_withJ_complex(gd, jd, r_up, r_dn))
                if ecp:
                    rz = rz + np.asarray(
                        jqmc_rs.grad_eL_ecp_wrt_lambda_complex(
                            gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                        )
                    )
                pairs = [(rz, xla().geminal_data.lambda_matrix, "Det/eL/lambda")]
                if gem_tok == "AGP_MO":
                    # The fixture must prove the driver-facing assembler as well as the
                    # free kernel.  This catches a kernel that exists but is still rejected
                    # by `_rs_pd_complex_gap` or absent from `_RsComplexPdCtx`.
                    from jqmc.derivatives.bridge import _reduce_c_param_grads
                    from jqmc.jqmc_mcmc import _RsComplexPdCtx, _rs_native_c_param_grads

                    flags = _BLOCK_FLAGS["Det"]
                    expected = _reduce_c_param_grads(
                        ham.wavefunction_data,
                        jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), xla()),
                        1,
                        flags,
                    )
                    weights, grid_points = _one_move_mesh()
                    production = _rs_native_c_param_grads(
                        _RsComplexPdCtx(gd, jd, ham.coulomb_potential_data),
                        ham,
                        [np.asarray(r_up)],
                        [np.asarray(r_dn)],
                        flags,
                        1,
                        quantity="e_L",
                        ecp_flag=ecp,
                        ecp_w=weights,
                        ecp_gp=grid_points,
                        rts_b=[np.eye(3)],
                        complex_twist=True,
                    )
                    assert set(production) == {"lambda_matrix"}
                    pairs.append(
                        (
                            production["lambda_matrix"],
                            expected["lambda_matrix"],
                            "Det/eL/lambda/production-assembler",
                        )
                    )
                return _assert_close(pairs, _Z_KIN)
            if jd.jastrow_one_body_data is None:
                au, ad = jqmc_rs.grad_T_mo_complex(gd, r_up, r_dn)
            else:
                au, ad = jqmc_rs.grad_T_mo_withJ_complex(gd, jd, r_up, r_dn)
            au, ad = np.asarray(au), np.asarray(ad)
            if ecp:
                eu, ed = jqmc_rs.grad_eL_ecp_wrt_mo_coeff_complex(
                    gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                )
                au, ad = au + np.asarray(eu), ad + np.asarray(ed)
            pairs = [
                (au, xla().geminal_data.orb_data_up_spin.mo_coefficients, "Det/eL/mo_up"),
                (ad, xla().geminal_data.orb_data_dn_spin.mo_coefficients, "Det/eL/mo_dn"),
            ]
            return _assert_close(pairs, _Z_KIN)

        rust_port_gap(
            f"Det e_L on (bc={bc}, ecp={ecp}, withJ={jd.jastrow_one_body_data is not None}, ao={ao!r}): "
            "not a single ported Rust c-axis kernel (periodic/twist kinetic + ECP-nl c-axis assembly)"
        )

    # ============================ Basis (geminal radial exp/coeff) ============================
    if block == "Basis":
        # Rust grad_{ln_det,eL}_wrt_radial serves cart/sphe-GTO molecular (radial
        # sensitivity is angular-kind agnostic -- same (2l+3)/(4Z) normalization power),
        # the real-pGTO periodic leaf (image-folded eval, home-cell wrapping so the
        # per-image radial attribution matches XLA), AND the real-Gamma composite
        # (pgto(+)pw): its GTO child's radial is served child-wise, the pulled cotangent is
        # sliced to the GTO home rows. The COMPLEX-TWIST (Bloch) family --
        # sphe/cart-pgto_complex and the pgto(+)pw composite at a twist -- is served by the
        # complex twins ``grad_{ln_det,eL,eL_ecp}_wrt_radial_complex``: the image phase
        # and the wrap-restore phase are radial-independent and factor through, so the same
        # real per-primitive sensitivity carries, with the whole chain kept HOLOMORPHIC and
        # ``Re`` taken last. Still unported: PURE plane_wave / bspline (no GTO radial at all).
        if ao in ("plane_wave", "bspline") or "pw" in ao:
            rust_port_gap(
                f"Basis c-axis on ao={ao!r}: pure plane-wave / B-spline leaves carry no GTO/pGTO "
                "radial params (Rust grad_*_wrt_radial raises NotImplementedError) -- structurally n/a"
            )
        if quantity == "ln_Psi":
            rz = (  # jastrow-independent
                jqmc_rs.grad_ln_det_wrt_radial_complex(gd, r_up, r_dn) if cx else jqmc_rs.grad_ln_det_wrt_radial(gd, r_up, r_dn)
            )
            zones = _Z_DET
        else:
            # e_L = dT_det/d(radial) (with-Jastrow kinetic: the `-Sum gJ.gD` cross term
            # rides the orbital-GRADIENT cotangent, jnone when ``jastrow_data`` is None)
            # + dV_ecp_nl/d(radial) (mesh-O_k det).
            rz = (
                jqmc_rs.grad_eL_wrt_radial_complex(gd, r_up, r_dn, jd) if cx else jqmc_rs.grad_eL_wrt_radial(gd, r_up, r_dn, jd)
            )
            if ecp:
                ecp_radial = jqmc_rs.grad_eL_ecp_wrt_radial_complex if cx else jqmc_rs.grad_eL_ecp_wrt_radial
                erz = ecp_radial(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                rz = {
                    s: (np.asarray(rz[s][0]) + np.asarray(erz[s][0]), np.asarray(rz[s][1]) + np.asarray(erz[s][1]))
                    for s in ("orb_data_up_spin", "orb_data_dn_spin")
                }
            zones = _Z_KIN
        pairs = []
        for side in ("orb_data_up_spin", "orb_data_dn_spin"):
            aZ, ac = rz[side]
            rad = _radial_of(getattr(xla().geminal_data, side))
            pairs.append((aZ, rad.exponents, f"Basis/{side}/exp"))
            pairs.append((ac, rad.coefficients, f"Basis/{side}/coeff"))
        return _assert_close(pairs, zones)

    # ============================ J3 orbital radial exp/coeff ============================
    if block == "J3Basis":
        # This goes through the same production block assembler the MCMC driver
        # uses, rather than comparing an isolated Rust helper.  That is required
        # for composite J3: main publishes child-qualified names
        # (``j3_child{i}_basis_{exp,coeff}``) and a numerically correct kernel is
        # still unusable if the production route drops or misnames those blocks.
        from jqmc.derivatives.bridge import _reduce_c_param_grads
        from jqmc.jqmc_mcmc import _RsComplexPdCtx, _rs_native_c_param_grads

        flags = _BLOCK_FLAGS[block]
        expected_all = _reduce_c_param_grads(
            ham.wavefunction_data,
            jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), xla()),
            1,
            flags,
        )
        expected = {name: value for name, value in expected_all.items() if name.startswith("j3_") and "basis_" in name}
        declared = {
            parameter.name
            for parameter in ham.wavefunction_data.get_variational_blocks(
                state_token=("rust-analytic-matrix", 0),
                **flags,
            )
            if parameter.name.startswith("j3_") and "basis_" in parameter.name
        }
        assert declared, f"{_CURRENT['cell_id']}: J3Basis row has no declared upstream P09/P10/P14/P15 block"
        assert set(expected) == declared, (
            f"{_CURRENT['cell_id']}: XLA reduction did not publish every declared J3 radial block: "
            f"declared={sorted(declared)}, reduced={sorted(expected)}"
        )

        if cx:
            ctx = _RsComplexPdCtx(gd, jd, ham.coulomb_potential_data)
        else:
            ctx = jqmc_rs.build_obs_ctx(
                gd,
                jd,
                ham.coulomb_potential_data,
                False,
                bool(ham.structure_data.pbc_flag),
                False,
                False,
                False,
                True,
                precision_policy=precision_policy_snapshot(),
            )
        weights, grid_points = _one_move_mesh()
        got = _rs_native_c_param_grads(
            ctx,
            ham,
            [np.asarray(r_up)],
            [np.asarray(r_dn)],
            flags,
            1,
            quantity=quantity,
            ecp_flag=ecp,
            ecp_w=weights,
            ecp_gp=grid_points,
            rts_b=[np.eye(3)],
            complex_twist=cx,
        )
        got = {name: value for name, value in got.items() if name.startswith("j3_") and "basis_" in name}
        assert set(got) == declared, (
            f"{_CURRENT['cell_id']}: Rust production J3 radial block set differs from main: "
            f"declared={sorted(declared)}, rust={sorted(got)}"
        )
        return _assert_close(
            [(got[name], expected[name], f"J3Basis/{name}") for name in sorted(declared)],
            _Z_KIN if quantity == "e_L" else _Z_JAS,
        )

    # ============================ J1 / J2 (Jastrow one/two-body param) ============================
    if block in ("J1", "J2"):
        if quantity == "ln_Psi":
            if block == "J1":
                rz = jqmc_rs.grad_ln_Psi_jastrow_1b_param(jd.jastrow_one_body_data, r_up, r_dn)
                leaf = xla().jastrow_data.jastrow_one_body_data.jastrow_1b_param
            else:
                rz = jqmc_rs.grad_ln_Psi_jastrow_2b_param(jd.jastrow_two_body_data, r_up, r_dn)
                leaf = xla().jastrow_data.jastrow_two_body_data.jastrow_2b_param
            return _assert_close([(rz, leaf, f"{block}/lnPsi")], _Z_JAS)
        # e_L = dT/dc (periodic-aware kinetic reduction -- molecular Euclidean or
        # Williamson smooth periodic distance) + dV_ecp_nl/dc (mesh-O_k, molecular).
        if bc == "twist":
            # Complex-twist Jastrow-kinetic dE_L/dc = Re(dT/dc) (+ complex-mesh
            # dV_ecp_nl/dc for ECP). Served on the native complex sphe/cart-pGTO leaves;
            # plane-wave / bspline / composite twist have no complex kinetic reduction.
            if not _jas_kin_twist_okay(ao, gk):
                rust_port_gap(
                    f"{block} e_L on twist (ao={ao!r}, gem={gk}): the Rust complex Jastrow-kinetic "
                    "serves every SD/AGPn complex leaf (native pGTO, plane_wave_complex, "
                    "bspline_complex, composite); the dense-lambda AGP twist drift is unported"
                )
            if block == "J1":
                rz = jqmc_rs.grad_eL_wrt_jastrow_1b_param_complex(gd, jd, r_up, r_dn)
                if ecp:
                    rz = rz + jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param_complex(
                        gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                    )
                leaf = xla().jastrow_data.jastrow_one_body_data.jastrow_1b_param
            else:
                rz = jqmc_rs.grad_eL_wrt_jastrow_2b_param_complex(gd, jd, r_up, r_dn)
                if ecp:
                    rz = rz + jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param_complex(
                        gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                    )
                leaf = xla().jastrow_data.jastrow_two_body_data.jastrow_2b_param
            return _assert_close([(rz, leaf, f"{block}/eL")], _Z_KIN)
        # Real-Gamma periodic: EVERY real leaf is served. `grad_eL_wrt_jastrow_{1b,2b}_param`
        # reduce `dT/dc = -1/2 Sum grad^2 f_c - Sum (gJ + gD) . grad f_c` (param_grad_el.rs
        # `grad_t_j1` / `grad_t_j2`). `f_c = dJ/dc` is a pure electron-distance field --
        # basis-INDEPENDENT -- so the only leaf-dependent input is the determinant drift
        # `gD = grad ln|det G|`, which rides `det_grad_field` -> `gem_forward` ->
        # `orb_value_grad_lap`. That evaluator ingests the pure plane-wave leaf (0-row GTO
        # placeholder + PW child), the B-spline leaf, and the composite (gto + PW/spline
        # child concatenated in `gto_first` order) -- the SAME ingestion the Det e_L branch
        # above already exercises at pbc_gamma. Measured Rust-vs-XLA over all 11 periodic
        # non-pGTO cells x {J1, J2}: max|diff| = 2.97e-10 (plane-wave), 1.76e-11 (B-spline),
        # 1.10e-12 (composite), against references of O(1e-1 .. 1e1) -- 3+ orders inside the
        # medium atol and ~7 orders inside the rtol budget.
        if block == "J1":
            rz = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_1b_param(gd, jd, r_up, r_dn))
            if ecp:
                rz = rz + np.asarray(
                    jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                )
            leaf = xla().jastrow_data.jastrow_one_body_data.jastrow_1b_param
        else:
            rz = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_2b_param(gd, jd, r_up, r_dn))
            if ecp:
                rz = rz + np.asarray(
                    jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                )
            leaf = xla().jastrow_data.jastrow_two_body_data.jastrow_2b_param
        return _assert_close([(rz, leaf, f"{block}/eL")], _Z_KIN)

    # ============================ J3 (three-body j_matrix) ============================
    if block == "J3":
        # EVERY J3 leaf the matrix builds is served: the
        # single-leaf GTO/pGTO leaves AND the MO-wrapped composite / plane-wave / B-spline
        # ones, real-Gamma and complex-twist alike. ``J3 = Sum_{i<j} A M A + j1col . A``
        # depends on the basis ONLY through the ORBITAL values ``A``, so the c-axis reduction
        # is basis-free once the leaf's own value/grad/laplacian evaluator supplies them
        # (Rust ``param_grad::j3_jmatrix_from_orb_values``, fed by ``OrbData::compute`` /
        # ``ComplexOrb::value{,_grad_lap}``, which own the whole leaf taxonomy including the
        # MO projection). The complex leaves route through the ``_complex`` twins, which take
        # ``Re`` per element AFTER the MO contraction (``Re(C)Re(AO) != Re(C*AO)``), matching
        # jqmc's ``map(jnp.real, compute_orb_vgl(orb_data, r))``.
        if quantity == "ln_Psi":
            rz = (
                jqmc_rs.grad_ln_Psi_jastrow_3b_jmatrix_complex(jd.jastrow_three_body_data, r_up, r_dn)
                if cx
                else jqmc_rs.grad_ln_Psi_jastrow_3b_jmatrix(jd.jastrow_three_body_data, r_up, r_dn)
            )
            return _assert_close([(rz, xla().jastrow_data.jastrow_three_body_data.j_matrix, "J3/lnPsi")], _Z_JAS)
        # e_L = dT/dc (periodic J3 kinetic rides the image-folded orbital eval) +
        # dV_ecp_nl/dc (mesh-O_k; complex mesh at a twist).
        if bc == "twist":
            # Complex-twist J3-kinetic dE_L/dc = Re(dT/dj_matrix) (A = Re(orbital)), plus the
            # complex-mesh dV_ecp_nl/dj_matrix for an ECP twist. The mesh weight
            # ``Re(det_amp * jr)`` is projected as a PRODUCT, never factor-by-factor.
            rz = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix_complex(gd, jd, r_up, r_dn))
            if ecp:
                rz = rz + np.asarray(
                    jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix_complex(
                        gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                    )
                )
            return _assert_close([(rz, xla().jastrow_data.jastrow_three_body_data.j_matrix, "J3/eL")], _Z_KIN)
        rz = np.asarray(jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix(gd, jd, r_up, r_dn))
        if ecp:
            rz = rz + np.asarray(
                jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
            )
        return _assert_close([(rz, xla().jastrow_data.jastrow_three_body_data.j_matrix, "J3/eL")], _Z_KIN)

    # ============================ DTN (u/chi/f polynomial params) ============================
    if block == "DTN":
        d = jd.jastrow_dtn_data
        if quantity == "ln_Psi":
            # grad_ln_Psi_dtn_params uses the min-image distance (matching the DTN VALUE)
            # so it is periodic-correct for pbc_gamma AND twist (DTN is a REAL distance
            # Jastrow -> its O_k = dJ_DTN/dc is real and basis/twist-independent).
            a, b, g = jqmc_rs.grad_ln_Psi_dtn_params(d, r_up, r_dn)
            zones = _Z_JAS
        elif bc == "twist":
            # Complex-twist DTN-kinetic dE_L/dc = Re(dT/dc) (+ complex-mesh dV_ecp_nl/dc
            # for ECP). Served on the native complex sphe/cart-pGTO leaves.
            if not _jas_kin_twist_okay(ao, gk):
                rust_port_gap(
                    f"DTN e_L on twist (ao={ao!r}, gem={gk}): the Rust complex DTN-kinetic serves "
                    "every SD/AGPn complex leaf (native pGTO, plane_wave_complex, bspline_complex, "
                    "composite); the dense-lambda AGP twist drift is unported"
                )
            a, b, g = (
                None if x is None else np.asarray(x) for x in jqmc_rs.grad_eL_dtn_kinetic_params_complex(gd, jd, r_up, r_dn)
            )
            if ecp:
                ea, eb, eg = jqmc_rs.grad_eL_ecp_dtn_params_complex(
                    gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh()
                )
                a = a if ea is None else a + np.asarray(ea)
                b = b if eb is None else b + np.asarray(eb)
                g = g if eg is None else g + np.asarray(eg)
            zones = _Z_KIN
        else:
            # e_L = dT/dc (periodic-aware u/chi/f kinetic reduction on the min-image
            # displacement) + dV_ecp_nl/dc (mesh-O_k, molecular). Served on EVERY real leaf
            # at real Gamma, for the same reason as J1/J2 above: the DTN u/chi/f fields are
            # min-image electron-distance polynomials (basis-independent), and the only
            # leaf-dependent input -- the determinant drift gD -- rides the basis-agnostic
            # `orb_value_grad_lap`, which serves plane-wave / B-spline / composite. Measured
            # over the 4 periodic non-pGTO DTN cells: max|diff| = 6.52e-08 (plane-wave gamma,
            # against a reference of 6.90e+02, i.e. 9.5e-11 relative vs a 1e-5 rtol) and
            # 6.15e-10 (B-spline).
            a, b, g = (None if x is None else np.asarray(x) for x in jqmc_rs.grad_eL_dtn_kinetic_params(gd, jd, r_up, r_dn))
            if ecp:
                ea, eb, eg = jqmc_rs.grad_eL_ecp_dtn_params(gd, jd, ham.coulomb_potential_data, r_up, r_dn, *_one_move_mesh())
                a = a if ea is None else a + np.asarray(ea)
                b = b if eb is None else b + np.asarray(eb)
                g = g if eg is None else g + np.asarray(eg)
            zones = _Z_KIN
        xd = xla().jastrow_data.jastrow_dtn_data
        # The 7th DTN ``if d.use_*:`` site. Assert at least one leg is live so a
        # fixture that zeroed all three flags fails loudly instead of comparing nothing;
        # ``_assert_close(expect_nonempty=True)`` below is the backstop for the same hole.
        _assert_dtn_liveness(d)
        pairs = []
        if d.use_u:
            pairs.append((a, xd.alpha_indep, f"DTN/{quantity}/alpha"))
        if d.use_chi:
            pairs.append((b, xd.beta_indep, f"DTN/{quantity}/beta"))
        if d.use_f:
            pairs.append((g, xd.gamma_indep, f"DTN/{quantity}/gamma"))
        return _assert_close(pairs, zones)

    raise AssertionError(f"unhandled block {block!r}")


@pytest.mark.light
def test_divergence_allowlist_is_covered_by_the_matrix():
    """Every frozen divergence must name a (cell, block, quantity) the matrix ACTUALLY
    walks -- otherwise a typo in the key silently disarms the freeze AND the assertion.

    (Staleness -- an allowlisted entry that has since started passing -- is caught by the
    walk itself: ``_assert_close`` only consults the allowlist after a compare has already
    failed, so a fixed cell simply goes green and its entry becomes inert; the coverage
    check here is what keeps the KEY meaningful. With the allowlist empty this is a
    guard on future additions.)
    """
    walked = {(_cell_id(p.values[0]), p.values[1], p.values[2]) for p in _PARAMS}
    unknown = sorted(k for k in _DIVERGENCE_ALLOWLIST if k not in walked)
    assert not unknown, f"_DIVERGENCE_ALLOWLIST keys not present in the matrix: {unknown}"
    # The XLA-oracle-gap allowlist gets the same key-integrity guard, so a typo
    # there cannot silently disarm the fail-on-unimplemented-oracle trap either.
    unknown_gap = sorted(k for k in _XLA_ORACLE_GAP_ALLOWLIST if k not in walked)
    assert not unknown_gap, f"_XLA_ORACLE_GAP_ALLOWLIST keys not present in the matrix: {unknown_gap}"
