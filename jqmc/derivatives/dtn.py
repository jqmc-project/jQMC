"""DTN polynomial Jastrow O_k: d J_DTN / d (alpha/beta/gamma indep params).

J_DTN = J_u + J_chi + J_f is LINEAR in the inflated coefficients (full = M @ indep),
so the O_k is closed-form feature-sums chained through M (no AD). Cutoff phi(r) =
max(L-r,0)^C. Channels: u(e-e), chi(e-N), f(e-e-N). Molecular and
minimum-image periodic geometries use the same feature algebra.

Pure-``jnp`` and JIT-FRIENDLY: every structural Python for-loop (over electrons /
atoms / electron-PAIRS) is folded into a single vectorized op -- broadcast over the
electron/atom/pair AXIS with a species ``.at[sp].add`` scatter, or a ``jax.vmap`` of
the scalar 3-body body for the f-channel (``grad_T_dtn_wrt_param`` and the kinetic
``_f_triple_grads``). The kernels therefore emit a SMALL HLO (no giant unroll) and
compile under ``jax.jit``. Static shape/index data
(``species_index``, ``triu_indices``, spin labels, order counts) stay concrete numpy;
only the position-derived distances/features are traced ``jnp``.
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

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.determinant import AGP_data, AGPn_data, SD_data
from jqmc.jastrow_factor import (
    Jastrow_DTN_data,
    Jastrow_data,
    _dtn_cutoff_policy_full_coefficients,
    _dtn_effective_cutoffs,
    _dtn_inflate_alpha,
    _dtn_inflate_beta,
    _dtn_inflate_gamma,
    _dtn_minimum_image_plan,
    _dtn_minimum_image,
)

GeminalData: TypeAlias = AGP_data | AGPn_data | SD_data
RadialDerivatives: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
RadialThirdDerivatives: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
KineticDerivativeFields: TypeAlias = tuple[jax.Array, jax.Array, jax.Array]
TripleDisplacementGradients: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
OneMoveMeshResult: TypeAlias = tuple[jax.Array, jax.Array]


def _md(diff: jax.Array, d: Jastrow_DTN_data) -> jax.Array:
    """Minimum-image displacement (PBC) or the raw displacement (molecular). Matches the DTN
    forward jastrow_factor._dtn_scalar_dist_with_derivs: the shared ``_dtn_minimum_image`` (wrap
    plus the static ``mic_shell_half_widths`` shell, exact below the WS inscribed radius on any
    cell, issue #68); the distance / grad (u_min/r) / Laplacian (2/r) then take the SAME
    plain-Euclidean forms as the molecular branch on u_min (the CASINO hard cutoff pins a unique
    image -> image index locally constant -> d u_min/d r = I)."""
    sd = d.structure_data
    if not getattr(sd, "pbc_flag", False):
        return diff
    from jqmc.structure import _inv_3x3

    cell = jnp.asarray(sd._cell_jnp, jnp.float64)
    inv = _inv_3x3(cell)
    return _dtn_minimum_image(diff, cell, inv, _dtn_minimum_image_plan(d))


def _dists(r_a: jax.Array, r_b: jax.Array, d: Jastrow_DTN_data) -> jax.Array:
    return jnp.linalg.norm(_md(r_a[:, None, :] - r_b[None, :, :], d), axis=-1)


def _phi(L: jax.Array, r: jax.Array, C: int) -> RadialDerivatives:
    """cutoff phi=max(L-r,0)^C and phi', phi'' (0 for r>=L). Elementwise in r (scalar or array)."""
    s = jnp.maximum(L - r, 0.0)
    m = jnp.where(r < L, 1.0, 0.0)
    return s**C * m, -C * s ** (C - 1) * m, C * (C - 1) * s ** (C - 2) * m


def _radial(L: jax.Array, r: jax.Array, Nk: int, C: int) -> RadialDerivatives:
    """h_l = phi(r) r^l and h_l', h_l'' for l=0..Nk (e-N radial with cutoff). r scalar."""
    phi, dphi, ddphi = _phi(L, r, C)
    k = jnp.arange(Nk + 1)
    rk = r**k
    rk1 = jnp.where(k >= 1, k * r ** jnp.maximum(k - 1, 0), 0.0)
    rk2 = jnp.where(k >= 2, (k * (k - 1)) * r ** jnp.maximum(k - 2, 0), 0.0)
    return phi * rk, dphi * rk + phi * rk1, ddphi * rk + 2 * dphi * rk1 + phi * rk2


def _power(r: jax.Array, Nn: int) -> RadialDerivatives:
    """p_n = r^n and p_n', p_n'' for n=0..Nn (e-e factor, no cutoff). r scalar."""
    n = jnp.arange(Nn + 1)
    p = r**n
    pp = jnp.where(n >= 1, n * r ** jnp.maximum(n - 1, 0), 0.0)
    ppp = jnp.where(n >= 2, (n * (n - 1)) * r ** jnp.maximum(n - 2, 0), 0.0)
    return p, pp, ppp


# These radial helpers broadcast over arbitrary electron, atom, or pair axes.
# Keeping those axes in one array avoids Python loops in the compiled kernels.
def _poly3_b(r: jax.Array, coef: jax.Array) -> RadialThirdDerivatives:
    """g, g', g'', g''' of g=sum_k coef[..,k] r^k for r of shape ``S``; ``coef`` broadcasts
    to ``S + (Nk+1,)``. Batched twin of ``_poly3`` (``(coef*rk).sum(-1)`` == ``coef @ rk``)."""
    Nk = coef.shape[-1] - 1
    k = jnp.arange(Nk + 1)
    rk = r[..., None] ** k
    rk1 = jnp.where(k >= 1, k * r[..., None] ** jnp.maximum(k - 1, 0), 0.0)
    rk2 = jnp.where(k >= 2, (k * (k - 1)) * r[..., None] ** jnp.maximum(k - 2, 0), 0.0)
    rk3 = jnp.where(k >= 3, (k * (k - 1) * (k - 2)) * r[..., None] ** jnp.maximum(k - 3, 0), 0.0)
    return (coef * rk).sum(-1), (coef * rk1).sum(-1), (coef * rk2).sum(-1), (coef * rk3).sum(-1)


def _h3_b(L: jax.Array, coef: jax.Array, r: jax.Array, C: int) -> RadialThirdDerivatives:
    """Return radial cutoff-polynomial values and three derivatives.

    ``r`` has arbitrary leading shape ``S``. ``L`` broadcasts to ``S`` and
    ``coef`` broadcasts to ``S + (Nk+1,)``.
    """
    p, p1, p2, p3 = _phi3(L, r, C)
    g, g1, g2, g3 = _poly3_b(r, coef)
    return p * g, p1 * g + p * g1, p2 * g + 2 * p1 * g1 + p * g2, p3 * g + 3 * p2 * g1 + 3 * p1 * g2 + p * g3


def _policy_h3_b(
    L: jax.Array,
    coef: jax.Array,
    coef_dot: jax.Array,
    r: jax.Array,
    C: int,
) -> RadialThirdDerivatives:
    """``d/dln(ell_V) [(L-r)^C P(r)]`` and three r derivatives."""
    same_c = _h3_b(L, coef_dot, r, C)
    cutoff_tangent = _h3_b(L, coef, r, C - 1)
    scale = C * L
    return tuple(a + scale * b for a, b in zip(same_c, cutoff_tangent))


def _policy_P3_b(L: jax.Array, r: jax.Array, Nmax: int, C: int) -> RadialThirdDerivatives:
    """Cutoff-length tangent of ``_P3_b`` (the polynomial powers are fixed)."""
    return tuple(C * L[..., None] * x for x in _P3_b(L, r, Nmax, C - 1))


def _P3_b(L: jax.Array, r: jax.Array, Nmax: int, C: int) -> RadialThirdDerivatives:
    """Batched twin of ``_P3``: P_k=phi(r) r^k and P',P'',P''' for k=0..Nmax over r of shape
    ``S`` (leading axis). ``L`` broadcasts to ``S``. Returns each as ``S + (Nmax+1,)``."""
    p, p1, p2, p3 = _phi3(L, r, C)
    k = jnp.arange(Nmax + 1)
    rk = r[..., None] ** k
    rk1 = jnp.where(k >= 1, k * r[..., None] ** jnp.maximum(k - 1, 0), 0.0)
    rk2 = jnp.where(k >= 2, (k * (k - 1)) * r[..., None] ** jnp.maximum(k - 2, 0), 0.0)
    rk3 = jnp.where(k >= 3, (k * (k - 1) * (k - 2)) * r[..., None] ** jnp.maximum(k - 3, 0), 0.0)
    pe, p1e, p2e, p3e = p[..., None], p1[..., None], p2[..., None], p3[..., None]
    P = pe * rk
    P1 = p1e * rk + pe * rk1
    P2 = p2e * rk + 2 * p1e * rk1 + pe * rk2
    P3 = p3e * rk + 3 * p2e * rk1 + 3 * p1e * rk2 + pe * rk3
    return P, P1, P2, P3


def _Cn3_b(r: jax.Array, Nmax: int) -> RadialThirdDerivatives:
    """Batched twin of ``_Cn3``: C_n=r^n and C',C'',C''' for n=0..Nmax over r of shape ``S``
    (no cutoff, f-channel e-e factor). Returns each as ``S + (Nmax+1,)``."""
    n = jnp.arange(Nmax + 1)
    C0 = r[..., None] ** n
    C1 = jnp.where(n >= 1, n * r[..., None] ** jnp.maximum(n - 1, 0), 0.0)
    C2 = jnp.where(n >= 2, (n * (n - 1)) * r[..., None] ** jnp.maximum(n - 2, 0), 0.0)
    C3 = jnp.where(n >= 3, (n * (n - 1) * (n - 2)) * r[..., None] ** jnp.maximum(n - 3, 0), 0.0)
    return C0, C1, C2, C3


def _compile_stable(fun: Callable) -> Callable:
    """Cached ``jax.jit`` boundary for a kernel that contains a pair-chunk ``lax.scan``.

    Each such kernel builds its scan body as a closure on every call.  Called
    EAGERLY (``assemble.compute_grad_*`` outside ``jax.jit``, as the test oracles
    and GFMC/VMC reference helpers do), an eager ``lax.scan`` re-traces that new
    closure and compiles and loads a NEW XLA executable on every call; the
    executables stay alive in JAX's caches, so the per-process memory-mapping
    count grows without bound (issue #130: 5-9 compiles and ~1.2k mappings per
    ``compute_grad_e_L`` call, which exhausted ``vm.max_map_count`` in CI).
    One module-level ``jax.jit`` per kernel keys the compiled program on the
    argument pytree structure (the static DTN fields live in the treedef) and
    the shapes, so a repeated eager call reuses it (0 new compiles).  Inside an
    outer ``jax.jit`` / ``vmap`` / AD the boundary is an inlined nested call, so
    the production (already jitted) programs are unchanged.  The precision
    policy is read at trace time; ``jqmc._precision.configure`` clears the JAX
    caches when it changes the policy, so no stale-dtype program is reused.
    """
    return jax.jit(fun)


def _pair_chunks(
    body: Callable[[object, object], tuple[object, None]],
    carry0: object,
    xs_of: Callable[[np.ndarray, np.ndarray, jax.Array], object],
    ia: np.ndarray,
    ib: np.ndarray,
) -> object:
    """Chunked ``lax.scan`` inputs ``xs_of(iaC, ibC, mask)`` for a pair-chunk ``body``.

    The chunk width is the shared measured-budget width (see the streaming header of
    ``jqmc._setting``): ``body`` evaluated on a one-pair chunk is closure-converted and its
    workspace measured, against the PER-WALKER ``JASTROW_DTN_PAIR_BUDGET_BYTES``, and the
    width never exceeds the number of pairs, so the pair list is padded at most to a whole
    number of chunks of that width (a single chunk needs no padding). ``xs_of`` builds the
    scan inputs from the ``(n_chunk, chunk)`` index arrays and mask of :func:`_chunk_pairs`.
    """
    from jqmc._setting import JASTROW_DTN_PAIR_BUDGET_BYTES, measured_closure_stream_batch

    n_pair = int(ia.shape[0])
    ia1, ib1, m1 = _chunk_pairs(ia[:1], ib[:1], 1)
    one = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[0, 0], xs_of(ia1, ib1, m1))

    def one_pair(x: object) -> object:
        return body(carry0, jax.tree_util.tree_map(lambda a: a[None], x))[0]

    width = measured_closure_stream_batch(one_pair, (one,), n_pair, JASTROW_DTN_PAIR_BUDGET_BYTES)
    iaC, ibC, mask = _chunk_pairs(ia, ib, min(width, n_pair))
    return xs_of(iaC, ibC, mask)


def _chunk_pairs(
    ia: np.ndarray,
    ib: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    """Pad the (static, numpy) pair-index arrays to a whole number of ``chunk``-sized blocks and
    reshape to ``(n_chunk, chunk)``.  Padding repeats the LAST real pair (so padded displacements stay
    non-degenerate -- no 0/r) and returns a matching ``(n_chunk, chunk)`` float mask (1 real, 0 pad) so
    the padded pairs contribute nothing to the reduced accumulators."""
    n = int(ia.shape[0])
    n_chunk = (n + chunk - 1) // chunk
    pad = n_chunk * chunk - n
    if pad:
        ia = np.concatenate([ia, np.full(pad, ia[-1])])
        ib = np.concatenate([ib, np.full(pad, ib[-1])])
        m = np.concatenate([np.ones(n), np.zeros(pad)])
    else:
        m = np.ones(n)
    return ia.reshape(n_chunk, chunk), ib.reshape(n_chunk, chunk), jnp.asarray(m.reshape(n_chunk, chunk), jnp.float64)


@_compile_stable
def grad_dtn_wrt_param(d: Jastrow_DTN_data, r_up: jax.Array, r_dn: jax.Array) -> dict[str, jax.Array]:
    """{alpha_indep, beta_indep, gamma_indep} grads of d J_DTN / d indep params."""
    C = int(d.cutoff_smoothness)
    sd = d.structure_data
    R = jnp.asarray(sd.positions, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([r_up, r_dn], 0)
    out = {}

    if d.use_u:  # alpha (u, e-e)
        Lu = _dtn_effective_cutoffs(d, jnp.float64)[0]
        Nu = int(d.n_order_u)
        kk = jnp.arange(Nu + 1)

        def chan_feat(rr: jax.Array) -> jax.Array:
            phi = jnp.maximum(Lu - rr, 0.0) ** C
            return (phi[:, None] * rr[:, None] ** kk[None, :]).sum(0)

        def same_pairs(r_s: jax.Array) -> jax.Array:
            i, j = np.triu_indices(len(r_s), 1)
            return jnp.linalg.norm(_md(r_s[i] - r_s[j], d), axis=-1)

        same = jnp.concatenate([same_pairs(r_up), same_pairs(r_dn)]) if (len(r_up) + len(r_dn)) else jnp.asarray([])
        opp = _dists(r_up, r_dn, d).ravel() if (len(r_up) and len(r_dn)) else jnp.asarray([])
        feat = jnp.stack(
            [chan_feat(same) if same.size else jnp.zeros(Nu + 1), chan_feat(opp) if opp.size else jnp.zeros(Nu + 1)], 0
        )  # (2, Nu+1)
        out["alpha_indep"] = jnp.einsum("sk,ski->si", feat, jnp.asarray(d.M_alpha))

    if d.use_chi:  # beta (chi, e-N)
        Nc = int(d.n_order_chi)
        kk = jnp.arange(Nc + 1)
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1]
        n_sp = int(np.asarray(d.M_beta).shape[0])
        d_eI = _dists(all_e, R, d)  # (Ne, n_atom)
        # Fold the per-atom loop: build phi/r over the (Ne, n_atom) grid, sum the
        # per-electron feature (sum over e), then species-scatter over atoms.
        L_at = Lchi[sp]  # (n_atom,)
        phi = jnp.maximum(L_at[None, :] - d_eI, 0.0) ** C  # (Ne, n_atom)
        rk = d_eI[:, :, None] ** kk[None, None, :]  # (Ne, n_atom, Nc+1)
        feat_at = (phi[:, :, None] * rk).sum(0)  # (n_atom, Nc+1)
        feat = jnp.zeros((n_sp, Nc + 1)).at[sp].add(feat_at)
        out["beta_indep"] = jnp.einsum("sk,ski->si", feat, jnp.asarray(d.M_beta))

    if d.use_f:  # gamma (f, e-e-N)
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2]
        i_idx, j_idx = np.triu_indices(len(all_e), 1)
        kl = jnp.arange(NeN + 1)
        kn = jnp.arange(Nee + 1)
        Mg = jnp.asarray(d.M_gamma)
        n_atom = R.shape[0]
        L_at = Lf[sp]  # (n_atom,)
        n_pair = int(i_idx.shape[0])
        feat = jnp.zeros(Mg.shape[:4])
        # Adjoint-order: accumulate the reduced per-atom feature feat[I,l,m,n] over PAIR CHUNKS in a
        # ``lax.scan`` (rolled -> ONE chunk's working set is live, reused across steps).  The chunk width
        # is the measured budget width (``_pair_chunks``); padded pairs are masked to zero.  Replaces the field-first
        # einsum("pI,pIl,pIm,pn->Ilmn") whose (p,I,l,m,n) intermediate scaled as n_pair*n_atom*order^3.
        if n_pair:

            def _feat_body(
                carry: jax.Array,
                xs: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, None]:
                ra_c, rb_c, m = xs  # (chunk, 3), (chunk, 3), (chunk,)
                da = jnp.linalg.norm(_md(ra_c[:, None, :] - R[None, :, :], d), axis=-1)  # (chunk, n_atom)
                db = jnp.linalg.norm(_md(rb_c[:, None, :] - R[None, :, :], d), axis=-1)
                rab = jnp.linalg.norm(_md(ra_c - rb_c, d), axis=-1)  # (chunk,)
                ca = jnp.maximum(L_at[None, :] - da, 0.0) ** C  # (chunk, n_atom)
                cb = jnp.maximum(L_at[None, :] - db, 0.0) ** C
                pa = da[:, :, None] ** kl[None, None, :]  # (chunk, n_atom, NeN+1)
                pb = db[:, :, None] ** kl[None, None, :]
                powab = rab[:, None] ** kn[None, :]  # (chunk, Nee+1)
                return carry + jnp.einsum("pI,pIl,pIm,pn->Ilmn", ca * cb * m[:, None], pa, pb, powab), None

            feat0 = jnp.zeros((n_atom,) + Mg.shape[1:4])
            xs = _pair_chunks(_feat_body, feat0, lambda iaC, ibC, mask: (all_e[iaC], all_e[ibC], mask), i_idx, j_idx)
            feat_at, _ = jax.lax.scan(_feat_body, feat0, xs)
            feat = jnp.zeros(Mg.shape[:4]).at[sp].add(feat_at)
        out["gamma_indep"] = jnp.einsum("slmn,slmni->si", feat, Mg)

    return out


@_compile_stable
def dtn_cutoff_policy_fields(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Analytic field ``O_L=dJ_DTN/dln(ell_V)`` and its electron derivatives.

    Returns ``(O_L, grad_up, grad_dn, lap_up, lap_dn)``.  The full
    coefficients follow the minimum-norm tangent of the cutoff-dependent
    cusp/null constraint manifold; the cutoff polynomial itself follows
    ``dL/dln(ell_V)=L``.  Molecular data return exact zeros.
    """
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate((ru, rd), axis=0)
    Ne = int(all_e.shape[0])
    n_up = int(ru.shape[0])
    if not d.structure_data.pbc_flag:
        return (
            jnp.asarray(0.0, jnp.float64),
            jnp.zeros_like(ru),
            jnp.zeros_like(rd),
            jnp.zeros((n_up,)),
            jnp.zeros((Ne - n_up,)),
        )

    sd = d.structure_data
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    alpha, beta, gamma = _inflate_alpha(d), _inflate_beta(d), _inflate_gamma(d)
    alpha_dot, beta_dot, gamma_dot = _dtn_cutoff_policy_full_coefficients(d, jnp.float64)
    grad = jnp.zeros((Ne, 3), jnp.float64)
    lap = jnp.zeros((Ne,), jnp.float64)
    value = jnp.asarray(0.0, jnp.float64)

    if d.use_u and Ne > 1:
        ia, ib = np.triu_indices(Ne, 1)
        w = _md(all_e[ia] - all_e[ib], d)
        r = jnp.linalg.norm(w, axis=-1)
        r_safe = jnp.where(r > 0.0, r, 1.0)
        unit = w / r_safe[:, None]
        spin = np.asarray([0] * n_up + [1] * (Ne - n_up), np.int32)
        channel = np.where(spin[ia] == spin[ib], 0, 1)
        h0, h1, h2, _ = _policy_h3_b(
            _dtn_effective_cutoffs(d, jnp.float64)[0], alpha[channel], alpha_dot[channel], r, int(d.cutoff_smoothness)
        )
        pair_grad = h1[:, None] * unit
        pair_lap = h2 + 2.0 * h1 / r_safe
        value = value + jnp.sum(h0)
        grad = grad.at[ia].add(pair_grad).at[ib].add(-pair_grad)
        lap = lap.at[ia].add(pair_lap).at[ib].add(pair_lap)

    if d.use_chi and Ne > 0:
        w = _md(all_e[:, None, :] - R[None, :, :], d)
        r = jnp.linalg.norm(w, axis=-1)
        r_safe = jnp.where(r > 0.0, r, 1.0)
        unit = w / r_safe[..., None]
        L = _dtn_effective_cutoffs(d, jnp.float64)[1][sp]
        h0, h1, h2, _ = _policy_h3_b(L[None, :], beta[sp][None, :, :], beta_dot[sp][None, :, :], r, int(d.cutoff_smoothness))
        value = value + jnp.sum(h0)
        grad = grad + jnp.sum(h1[..., None] * unit, axis=1)
        lap = lap + jnp.sum(h2 + 2.0 * h1 / r_safe, axis=1)

    if d.use_f and Ne > 1:
        C = int(d.cutoff_smoothness)
        N_en, N_ee = int(d.n_order_eN), int(d.n_order_ee)
        ia, ib = np.triu_indices(Ne, 1)
        L = _dtn_effective_cutoffs(d, jnp.float64)[2][sp]
        gamma_at, gamma_dot_at = gamma[sp], gamma_dot[sp]

        def pair_fields(ra: jax.Array, rb: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            x, y, z = _md(ra[None, :] - R, d), _md(rb[None, :] - R, d), _md(ra - rb, d)
            rx, ry, rz = jnp.linalg.norm(x, axis=-1), jnp.linalg.norm(y, axis=-1), jnp.linalg.norm(z)
            rxs, rys, rzs = jnp.where(rx > 0.0, rx, 1.0), jnp.where(ry > 0.0, ry, 1.0), jnp.where(rz > 0.0, rz, 1.0)
            ux, uy, uz = x / rxs[:, None], y / rys[:, None], z / rzs
            A, B = _P3_b(L, rx, N_en, C), _P3_b(L, ry, N_en, C)
            Ad, Bd = _policy_P3_b(L, rx, N_en, C), _policy_P3_b(L, ry, N_en, C)
            N = _Cn3_b(rz, N_ee)

            def t(a: int, b: int, c: int) -> jax.Array:
                return (
                    jnp.einsum("Ilmn,Il,Im,n->I", gamma_dot_at, A[a], B[b], N[c])
                    + jnp.einsum("Ilmn,Il,Im,n->I", gamma_at, Ad[a], B[b], N[c])
                    + jnp.einsum("Ilmn,Il,Im,n->I", gamma_at, A[a], Bd[b], N[c])
                )

            F = t(0, 0, 0)
            Fx, Fy, Fz = t(1, 0, 0), t(0, 1, 0), t(0, 0, 1)
            Fxx, Fyy, Fzz = t(2, 0, 0), t(0, 2, 0), t(0, 0, 2)
            Fxz, Fyz = t(1, 0, 1), t(0, 1, 1)
            gi = jnp.sum(Fx[:, None] * ux + Fz[:, None] * uz[None, :], axis=0)
            gj = jnp.sum(Fy[:, None] * uy - Fz[:, None] * uz[None, :], axis=0)
            li = jnp.sum(Fxx + Fzz + 2.0 * Fxz * jnp.einsum("Ic,c->I", ux, uz) + 2.0 * Fx / rxs + 2.0 * Fz / rzs)
            lj = jnp.sum(Fyy + Fzz - 2.0 * Fyz * jnp.einsum("Ic,c->I", uy, uz) + 2.0 * Fy / rys + 2.0 * Fz / rzs)
            return jnp.sum(F), gi, li, gj, lj

        n_pair = int(ia.shape[0])
        if n_pair:

            def pair_chunk(
                carry: tuple[jax.Array, jax.Array, jax.Array],
                xs: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
                val_c, grad_c, lap_c = carry
                ia_c, ib_c, valid = xs
                vals, gi, li, gj, lj = jax.vmap(pair_fields)(all_e[ia_c], all_e[ib_c])
                valid_f = valid.astype(jnp.float64)
                val_c = val_c + jnp.sum(vals * valid_f)
                grad_c = grad_c.at[ia_c].add(gi * valid_f[:, None]).at[ib_c].add(gj * valid_f[:, None])
                lap_c = lap_c.at[ia_c].add(li * valid_f).at[ib_c].add(lj * valid_f)
                return (val_c, grad_c, lap_c), None

            xs = _pair_chunks(pair_chunk, (value, grad, lap), lambda iaC, ibC, mask: (iaC, ibC, mask), ia, ib)
            (value, grad, lap), _ = jax.lax.scan(pair_chunk, (value, grad, lap), xs)

    return value, grad[:n_up], grad[n_up:], lap[:n_up], lap[n_up:]


def dtn_cutoff_policy_value(d: Jastrow_DTN_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """Scalar ``dJ_DTN/dln(ell_V)`` used by the Pulay stress."""
    return dtn_cutoff_policy_fields(d, r_up, r_dn)[0]


def _dtn_cutoff_policy_one_electron_value(
    d: Jastrow_DTN_data,
    all_e: jax.Array,
    n_up: int,
    src: jax.Array,
    electron_index: jax.Array,
) -> jax.Array:
    """Policy-field terms containing one moved electron at ``src``.

    This is the rank-one counterpart of :func:`dtn_cutoff_policy_value` used
    by the LRDMC/ECP weighted geometry mesh.  Subtracting its old-position
    value gives the exact policy delta without a full O(Ne^2*Natom) rebuild.
    """
    if not d.structure_data.pbc_flag:
        return jnp.asarray(0.0, jnp.float64)
    sd = d.structure_data
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    Ne = int(all_e.shape[0])
    ids = jnp.arange(Ne, dtype=jnp.int32)
    mask = ids != electron_index
    spin = ids < int(n_up)
    alpha, beta, gamma = _inflate_alpha(d), _inflate_beta(d), _inflate_gamma(d)
    alpha_dot, beta_dot, gamma_dot = _dtn_cutoff_policy_full_coefficients(d, jnp.float64)
    out = jnp.asarray(0.0, jnp.float64)
    C = int(d.cutoff_smoothness)

    if d.use_u:
        r = jnp.linalg.norm(_md(src[None, :] - all_e, d), axis=-1)
        channel = (spin[electron_index] != spin).astype(jnp.int32)
        h0 = _policy_h3_b(_dtn_effective_cutoffs(d, jnp.float64)[0], alpha[channel], alpha_dot[channel], r, C)[0]
        out = out + jnp.sum(jnp.where(mask, h0, 0.0))
    if d.use_chi:
        r = jnp.linalg.norm(_md(src[None, :] - R, d), axis=-1)
        L = _dtn_effective_cutoffs(d, jnp.float64)[1][sp]
        out = out + jnp.sum(_policy_h3_b(L, beta[sp], beta_dot[sp], r, C)[0])
    if d.use_f:
        moved_is_a = electron_index < ids
        a = jnp.where(moved_is_a[:, None], src[None, :], all_e)
        b = jnp.where(moved_is_a[:, None], all_e, src[None, :])
        rx = jnp.linalg.norm(_md(a[None, :, :] - R[:, None, :], d), axis=-1)
        ry = jnp.linalg.norm(_md(b[None, :, :] - R[:, None, :], d), axis=-1)
        rz = jnp.linalg.norm(_md(a - b, d), axis=-1)
        L = _dtn_effective_cutoffs(d, jnp.float64)[2][sp]
        A, B = _P3_b(L[:, None], rx, int(d.n_order_eN), C)[0], _P3_b(L[:, None], ry, int(d.n_order_eN), C)[0]
        Ad = _policy_P3_b(L[:, None], rx, int(d.n_order_eN), C)[0]
        Bd = _policy_P3_b(L[:, None], ry, int(d.n_order_eN), C)[0]
        N = _Cn3_b(rz, int(d.n_order_ee))[0]
        g, gd = gamma[sp], gamma_dot[sp]
        vals = (
            jnp.einsum("Ilmn,Ijl,Ijm,jn->Ij", gd, A, B, N)
            + jnp.einsum("Ilmn,Ijl,Ijm,jn->Ij", g, Ad, B, N)
            + jnp.einsum("Ilmn,Ijl,Ijm,jn->Ij", g, A, Bd, N)
        )
        out = out + jnp.sum(vals * mask[None, :])
    return out


# --------------------------------------------------------------------------- #
# DTN kinetic  dT/dc = -1/2 sum_e lap f_c - sum_e (gJ+gD)_e . grad f_c   (f_c = O_k field)
# Terms: alpha/beta (u/chi) and gamma (f, 3-body).
# --------------------------------------------------------------------------- #
def _cotangents(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _kinetic_det_grad_field

    ru, rd = jnp.asarray(r_up), jnp.asarray(r_dn)
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, ru, rd)
    # Reuse the running geminal inverse so MPI execution does not lower a fresh
    # SVD custom call. geminal_inv=None retains the non-fast SVD-pinv fallback.
    gD_up, gD_dn = _kinetic_det_grad_field(gd, ru, rd, geminal_inv=geminal_inv)
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], 0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], 0)
    return gJ + gD  # (Ne,3)


@_compile_stable
def grad_T_dtn_wrt_param(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """dT/d(alpha/beta/gamma indep) for DTN (u + chi + f channels)."""
    C = int(d.cutoff_smoothness)
    GD = _cotangents(d, jastrow_data, gd, jnp.asarray(r_up), jnp.asarray(r_dn), geminal_inv=geminal_inv)  # (Ne,3); jnp args
    r_up = jnp.asarray(r_up, jnp.float64)
    r_dn = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([r_up, r_dn], 0)
    n_up = len(r_up)
    Ne = len(all_e)
    spin = np.array([0] * n_up + [1] * (Ne - n_up))
    sd = d.structure_data
    R = jnp.asarray(sd.positions, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    out = {}

    if d.use_u:  # alpha (u, e-e) -- fold the electron/channel loops into an (Ne, Ne) pair grid.
        Lu = _dtn_effective_cutoffs(d, jnp.float64)[0]
        Nu = int(d.n_order_u)
        M = jnp.asarray(d.M_alpha)
        eye = np.eye(Ne, dtype=bool)
        diff = _md(all_e[:, None, :] - all_e[None, :, :], d)  # (Ne, Ne, 3) = r_e - r_f
        r = jnp.linalg.norm(diff, axis=-1)  # (Ne, Ne); diagonal 0
        r_safe = jnp.where(eye, 1.0, r)  # avoid the 0-division on the (masked) diagonal
        u = diff / r_safe[..., None]  # (Ne, Ne, 3)
        phi, dphi, ddphi = _phi(Lu, r_safe, C)  # (Ne, Ne)
        kk = jnp.arange(Nu + 1)
        p = r_safe[..., None] ** kk  # (Ne, Ne, Nu+1)
        pp = jnp.where(kk >= 1, kk * r_safe[..., None] ** jnp.maximum(kk - 1, 0), 0.0)
        ppp = jnp.where(kk >= 2, (kk * (kk - 1)) * r_safe[..., None] ** jnp.maximum(kk - 2, 0), 0.0)
        u_dot_GD = jnp.einsum("efd,ed->ef", u, GD)  # u[e,f].GD[e]  (Ne, Ne)
        same = (spin[:, None] == spin[None, :]) & ~eye  # channel-0 (same-spin) ordered pairs
        opp = spin[:, None] != spin[None, :]  # channel-1 (opposite-spin) ordered pairs
        chans = []
        for s in (0, 1):
            g, gp, gpp = p @ M[s], pp @ M[s], ppp @ M[s]  # (Ne, Ne, n_indep)
            hp = dphi[..., None] * g + phi[..., None] * gp
            hpp = ddphi[..., None] * g + 2 * dphi[..., None] * gp + phi[..., None] * gpp
            total = -0.5 * (hpp + 2 * hp / r_safe[..., None]) - hp * u_dot_GD[..., None]  # (Ne, Ne, n_indep)
            mask = same if s == 0 else opp
            chans.append((total * mask[..., None]).sum((0, 1)))
        out["alpha_indep"] = jnp.stack(chans, 0)  # (2, n_indep)

    if d.use_chi:  # beta (chi, e-N) -- fold the electron/atom loops into an (Ne, n_atom) grid.
        Nc = int(d.n_order_chi)
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1]
        M = jnp.asarray(d.M_beta)
        diff = _md(all_e[:, None, :] - R[None, :, :], d)  # (Ne, n_atom, 3)
        r = jnp.linalg.norm(diff, axis=-1)  # (Ne, n_atom)
        u = diff / r[..., None]
        L_at = Lchi[sp]  # (n_atom,)
        phi, dphi, ddphi = _phi(L_at[None, :], r, C)  # (Ne, n_atom)
        kk = jnp.arange(Nc + 1)
        pw = r[..., None] ** kk  # (Ne, n_atom, Nc+1)
        pw1 = jnp.where(kk >= 1, kk * r[..., None] ** jnp.maximum(kk - 1, 0), 0.0)
        pw2 = jnp.where(kk >= 2, (kk * (kk - 1)) * r[..., None] ** jnp.maximum(kk - 2, 0), 0.0)
        M_at = M[sp]  # (n_atom, Nc+1, n_indep)
        g = jnp.einsum("eIk,Iki->eIi", pw, M_at)  # (Ne, n_atom, n_indep)
        gp = jnp.einsum("eIk,Iki->eIi", pw1, M_at)
        gpp = jnp.einsum("eIk,Iki->eIi", pw2, M_at)
        hp = dphi[..., None] * g + phi[..., None] * gp  # (Ne, n_atom, n_indep)
        hpp = ddphi[..., None] * g + 2 * dphi[..., None] * gp + phi[..., None] * gpp
        u_dot_GD = jnp.einsum("eId,ed->eI", u, GD)  # u[e,I].GD[e]  (Ne, n_atom)
        total = -0.5 * (hpp + 2 * hp / r[..., None]) - hp * u_dot_GD[..., None]  # (Ne, n_atom, n_indep)
        per_atom = total.sum(0)  # (n_atom, n_indep)
        # Complex-Bloch (twist): GD (geminal drift) is complex128 while the DTN feature fields are real,
        # so per_atom is complex; carry that dtype into the species-scatter zeros (a float64 base would
        # FutureWarning + silently drop Im). Real systems: result_type(f64, f64) = f64 -> bit-identical.
        out["beta_indep"] = jnp.zeros((M.shape[0], M.shape[2]), dtype=jnp.result_type(M.dtype, GD.dtype)).at[sp].add(per_atom)

    if d.use_f:  # gamma (f, e-e-N) -- vmap the DERIVED scalar 3-body body over the (pair, atom) grid.
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2]
        Mg = jnp.asarray(d.M_gamma)  # (n_sp, NeN+1, NeN+1, Nee+1, n_indep)
        ia, ib = np.triu_indices(Ne, 1)

        def _body(
            Mg_s: jax.Array,
            L: jax.Array,
            ra: jax.Array,
            rb: jax.Array,
            RI: jax.Array,
            GDa: jax.Array,
            GDb: jax.Array,
        ) -> jax.Array:
            d_aI = _md(ra - RI, d)
            d_bI = _md(rb - RI, d)
            d_ab = _md(ra - rb, d)
            r_aI = jnp.linalg.norm(d_aI)
            r_bI = jnp.linalg.norm(d_bI)
            r_ab = jnp.linalg.norm(d_ab)
            u_aI, u_bI, u_ab = d_aI / r_aI, d_bI / r_bI, d_ab / r_ab
            ha, hap, happ = _radial(L, r_aI, NeN, C)  # (NeN+1,)
            hb, hbp, hbpp = _radial(L, r_bI, NeN, C)
            pn, pnp, pnpp = _power(r_ab, Nee)  # (Nee+1,)
            lap_a_r = happ + 2 * hap / r_aI  # (NeN+1,)
            lap_b_r = hbpp + 2 * hbp / r_bI
            lap_ab = pnpp + 2 * pnp / r_ab  # (Nee+1,)
            cab = u_aI @ u_ab
            cba = u_bI @ (-u_ab)
            # ln-tensors over (l,n) for a-side, (m,n) for b-side
            Ta = jnp.outer(lap_a_r, pn) + 2 * cab * jnp.outer(hap, pnp) + jnp.outer(ha, lap_ab)  # (NeN+1,Nee+1)
            Tb = jnp.outer(lap_b_r, pn) + 2 * cba * jnp.outer(hbp, pnp) + jnp.outer(hb, lap_ab)
            GAa, GAab = GDa @ u_aI, GDa @ u_ab
            GBb, GBba = GDb @ u_bI, GDb @ (-u_ab)
            Ga = jnp.outer(hap * GAa, pn) + jnp.outer(ha, pnp * GAab)  # (NeN+1,Nee+1)
            Gb = jnp.outer(hbp * GBb, pn) + jnp.outer(hb, pnp * GBba)
            # K[l,m,n] = -1/2(lap_a_B + lap_b_B) - GD_a.grad_a B - GD_b.grad_b B
            K = (
                -0.5 * (Ta[:, None, :] * hb[None, :, None] + ha[:, None, None] * Tb[None, :, :])
                - Ga[:, None, :] * hb[None, :, None]
                - ha[:, None, None] * Gb[None, :, :]
            )  # (NeN+1,NeN+1,Nee+1)
            return jnp.einsum("lmn,lmni->i", K, Mg_s)

        n_atom = R.shape[0]
        Mg_at = Mg[sp]  # (n_atom, NeN+1, NeN+1, Nee+1, n_indep)
        L_at = Lf[sp]  # (n_atom,)
        n_pair = int(ia.shape[0])
        over_pairs = jax.vmap(_body, in_axes=(None, None, 0, 0, None, 0, 0))  # over the chunk's pairs
        over_grid = jax.vmap(over_pairs, in_axes=(0, 0, None, None, 0, None, None))  # over atoms
        # Complex-Bloch (twist): the K-tensor carries GD (complex128) via GAa/GBb, so ``contrib`` and the
        # scan carry are complex; a float64 carry -> lax.scan "carry type float64 vs complex128" TypeError
        # at trace time. Real systems: result_type(f64, f64) = f64 -> bit-identical (unchanged carry).
        per_atom = jnp.zeros((n_atom, Mg.shape[4]), dtype=jnp.result_type(Mg.dtype, GD.dtype))
        # Adjoint-order: reduce the (atom, pair, order^3, n_indep) grid over PAIR CHUNKS in a rolled
        # ``lax.scan`` (only one chunk's vmap working set is live).  Each chunk is a
        # vmap-over-(atom, chunk-pairs), fully vectorized; padded pairs masked to zero.  Chunk width:
        # the measured budget width (``_pair_chunks``), which includes the n_indep axis.
        if n_pair:
            # The reduced grid (K[order^3] contracted with Mg_s[order^3, n_indep]) is a per-atom matmul,
            # so the working set is the K order-field (n_atom x order^3), NOT n_atom x order^3 x n_indep
            # (XLA does the matmul without materializing that product, so the working set is 2-4 GB, not
            # the ~400 GB a full outer product would need).
            def _dc_body(
                carry: jax.Array,
                xs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, None]:
                ra_c, rb_c, gda_c, gdb_c, m = xs  # (chunk,3) x4, (chunk,)
                contrib = over_grid(Mg_at, L_at, ra_c, rb_c, R, gda_c, gdb_c)  # (n_atom, chunk, n_indep)
                return carry + (contrib * m[None, :, None]).sum(1), None

            xs = _pair_chunks(
                _dc_body, per_atom, lambda iaC, ibC, mask: (all_e[iaC], all_e[ibC], GD[iaC], GD[ibC], mask), ia, ib
            )
            per_atom, _ = jax.lax.scan(_dc_body, per_atom, xs)
        out["gamma_indep"] = (
            jnp.zeros((Mg.shape[0], Mg.shape[4]), dtype=jnp.result_type(Mg.dtype, GD.dtype)).at[sp].add(per_atom)
        )

    return out


# =========================================================================== #
# DTN NUCLEAR FORCE (d/dR) + STRESS (d/d eps) for the local energy and ln|Psi|.
#
# These kernels share the same displacement and polynomial conventions:
#   * min-image distance via ``_md`` everywhere (molecular AND real-Gamma /
#     complex-twist PBC -- the CASINO hard cutoff pins a unique lattice image so
#     the min-image displacement u_min transforms affinely u_min -> u_min (I+eps)^T,
#     i.e. d r/d eps_cd = u_min,c u_min,d / r exactly -> the STRESS is the
#     DISPLACEMENT virial sum_w (dQ/dw)_c w_min,d, uniform for molecular & PBC);
#   * pure ``jnp`` + the traced-ham-safe geometry twins (_cell_jnp via ``_md`` and
#     ``_positions_cart_jnp`` for R), so the functions compile under the production
#     jit(vmap) bridge with the hamiltonian as a traced arg.
#
# All four force/stress quantities reduce to a single set of per-displacement
# gradients (radial derivatives chained through the inflated full coefficients):
#   - ln|Psi| (J_DTN value) needs h'(r);  forces -> translation identity, stress
#     -> (w (x) w)/r.
#   - kinetic T_DTN = sum_e [-1/2 lap_e - cot_e . grad_e] J  (cot = gJ + gD fixed)
#     needs h''' (third radial derivative); per displacement w the gradient dT/dw
#     gives dr (dT/dw), dR (-sum dT/dw, translation identity) and stress
#     (sum_w (dT/dw) (x) w_min).
# =========================================================================== #
def _inflate_alpha(d: Jastrow_DTN_data) -> jax.Array:
    """alpha_full[s, k] (2, Nu+1) in fp64 (``jastrow_factor._dtn_inflate_alpha``, incl. the comoving
    first-order transport when ``cutoff_scale != 1``)."""
    return _dtn_inflate_alpha(d, jnp.float64)


def _inflate_beta(d: Jastrow_DTN_data) -> jax.Array:
    """beta_full[I, k] (n_sp, Nc+1) in fp64."""
    return _dtn_inflate_beta(d, jnp.float64)


def _inflate_gamma(d: Jastrow_DTN_data) -> jax.Array:
    """gamma_full[I, l, m, n] (n_sp, NeN+1, NeN+1, Nee+1) in fp64 (homogeneous, no bias)."""
    return _dtn_inflate_gamma(d, jnp.float64)


def _phi3(L: jax.Array, r: jax.Array, C: int) -> RadialThirdDerivatives:
    """cutoff phi=max(L-r,0)^C and phi',phi'',phi''' at scalar r (0 for r>=L). The safe
    base ``ss`` avoids 0**(C-3) overflow on the masked r>=L branch."""
    s = jnp.maximum(L - r, 0.0)
    m = jnp.where(r < L, 1.0, 0.0)
    ss = jnp.where(r < L, s, 1.0)
    return ss**C * m, -C * ss ** (C - 1) * m, C * (C - 1) * ss ** (C - 2) * m, -C * (C - 1) * (C - 2) * ss ** (C - 3) * m


def _P3(L: jax.Array, r: jax.Array, Nmax: int, C: int) -> RadialThirdDerivatives:
    """P_k=phi(r) r^k and P',P'',P''' for k=0..Nmax at scalar r (f-channel e-N radial)."""
    p, p1, p2, p3 = _phi3(L, r, C)
    k = jnp.arange(Nmax + 1)
    rk = r**k
    rk1 = jnp.where(k >= 1, k * r ** jnp.maximum(k - 1, 0), 0.0)
    rk2 = jnp.where(k >= 2, (k * (k - 1)) * r ** jnp.maximum(k - 2, 0), 0.0)
    rk3 = jnp.where(k >= 3, (k * (k - 1) * (k - 2)) * r ** jnp.maximum(k - 3, 0), 0.0)
    P = p * rk
    P1 = p1 * rk + p * rk1
    P2 = p2 * rk + 2 * p1 * rk1 + p * rk2
    P3 = p3 * rk + 3 * p2 * rk1 + 3 * p1 * rk2 + p * rk3
    return P, P1, P2, P3


def _Cn3(r: jax.Array, Nmax: int) -> RadialThirdDerivatives:
    """C_n=r^n and C',C'',C''' for n=0..Nmax at scalar r (f-channel e-e factor, no cutoff)."""
    n = jnp.arange(Nmax + 1)
    C0 = r**n
    C1 = jnp.where(n >= 1, n * r ** jnp.maximum(n - 1, 0), 0.0)
    C2 = jnp.where(n >= 2, (n * (n - 1)) * r ** jnp.maximum(n - 2, 0), 0.0)
    C3 = jnp.where(n >= 3, (n * (n - 1) * (n - 2)) * r ** jnp.maximum(n - 3, 0), 0.0)
    return C0, C1, C2, C3


def _dtn_cotangent(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: GeminalData,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """cot = gJ + gD (Ne, 3): the FULL Jastrow drift (all channels incl. DTN) plus the
    determinant drift. Re() for complex-Bloch twist (no-op for real systems)."""
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_part

    from .kinetic import _kinetic_det_grad_field

    ru, rd = jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64)
    gJ_up, gJ_dn, _, _ = compute_grads_and_laplacian_Jastrow_part(jastrow_data, ru, rd)
    gD_up, gD_dn = _kinetic_det_grad_field(gd, ru, rd, geminal_inv=geminal_inv)
    gJ = jnp.concatenate([jnp.asarray(gJ_up), jnp.asarray(gJ_dn)], 0)
    gD = jnp.concatenate([jnp.asarray(gD_up), jnp.asarray(gD_dn)], 0)
    return jnp.real(gJ + gD)


# --------------------------------------------------------------------------- #
# ln|Psi| side:  d J_DTN / dR  and  d J_DTN / d eps
# --------------------------------------------------------------------------- #
@_compile_stable
def grad_dtn_wrt_R(d: Jastrow_DTN_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """d J_DTN / dR  (n_atom, 3). R enters only via the e-N distances (chi, f channels;
    u is e-e). Translation identity d r_eI/dR_I = -u_eI (min-image)."""
    C = int(d.cutoff_smoothness)
    sd = d.structure_data
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    all_e = jnp.concatenate([jnp.asarray(r_up, jnp.float64), jnp.asarray(r_dn, jnp.float64)], 0)
    n_atom = R.shape[0]
    out = jnp.zeros((n_atom, 3))

    if d.use_chi:  # fold the atom loop into an (n_atom, Ne) grid.
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1]
        beta = _inflate_beta(d)  # (n_sp, Nc+1)
        diff = _md(all_e[None, :, :] - R[:, None, :], d)  # (n_atom, Ne, 3) = r_e - R_I (min-image)
        r = jnp.linalg.norm(diff, axis=-1)  # (n_atom, Ne)
        u = diff / r[..., None]
        L_at = Lchi[sp]  # (n_atom,)
        phi, d1, _, _ = _phi3(L_at[:, None], r, C)  # (n_atom, Ne)
        beta_at = beta[sp]  # (n_atom, Nc+1)
        kk = jnp.arange(beta.shape[1])
        rk = r[..., None] ** kk  # (n_atom, Ne, Nc+1)
        rk1 = jnp.where(kk >= 1, kk * r[..., None] ** jnp.maximum(kk - 1, 0), 0.0)
        G = jnp.einsum("aek,ak->ae", rk, beta_at)  # (n_atom, Ne)
        Gp = jnp.einsum("aek,ak->ae", rk1, beta_at)
        coef = d1 * G + phi * Gp  # h_chi'(r) per (atom, electron)
        out = out - (coef[..., None] * u).sum(1)  # sum over electrons -> (n_atom, 3)

    if d.use_f:  # reduce the (n_atom, n_pair, order^3) grid over PAIR CHUNKS (adjoint-order).
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2]
        gamma = _inflate_gamma(d)  # (n_sp, NeN+1, NeN+1, Nee+1)
        ia, ib = np.triu_indices(all_e.shape[0], 1)
        Mg_at = gamma[sp]  # (n_atom, NeN+1, NeN+1, Nee+1)
        L_at = Lf[sp]  # (n_atom,)
        n_atom = R.shape[0]
        n_pair = int(ia.shape[0])
        if n_pair:  # rolled lax.scan over PAIR CHUNKS; padded pairs masked to zero

            def _R_body(
                carry: jax.Array,
                xs: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, None]:
                ra_c, rb_c, m = xs  # (chunk, 3), (chunk, 3), (chunk,)
                x = _md(ra_c[None, :, :] - R[:, None, :], d)  # (n_atom, chunk, 3)
                y = _md(rb_c[None, :, :] - R[:, None, :], d)
                r_x = jnp.linalg.norm(x, axis=-1)  # (n_atom, chunk)
                r_y = jnp.linalg.norm(y, axis=-1)
                r_z = jnp.linalg.norm(_md(ra_c - rb_c, d), axis=-1)  # (chunk,)
                A, A1, _, _ = _P3_b(L_at[:, None], r_x, NeN, C)  # (n_atom, chunk, NeN+1)
                B, B1, _, _ = _P3_b(L_at[:, None], r_y, NeN, C)
                N0, _, _, _ = _Cn3_b(r_z, Nee)  # (chunk, Nee+1)
                coefA = jnp.einsum("Ilmn,Ipl,Ipm,pn->Ip", Mg_at, A1, B, N0) * m[None, :]  # (n_atom, chunk)
                coefB = jnp.einsum("Ilmn,Ipl,Ipm,pn->Ip", Mg_at, A, B1, N0) * m[None, :]
                return carry - (coefA[..., None] * x / r_x[..., None] + coefB[..., None] * y / r_y[..., None]).sum(1), None

            xs = _pair_chunks(_R_body, jnp.zeros((n_atom, 3)), lambda iaC, ibC, mask: (all_e[iaC], all_e[ibC], mask), ia, ib)
            out_f, _ = jax.lax.scan(_R_body, jnp.zeros((n_atom, 3)), xs)
            out = out + out_f
    return out


@_compile_stable
def grad_dtn_wrt_strain(d: Jastrow_DTN_data, r_up: jax.Array, r_dn: jax.Array) -> jax.Array:
    """d J_DTN / d eps (3, 3). Affine strain F=I+eps deforms r, R, cell AND the lattice
    image: the min-image displacement w transforms w -> w F^T, so d r/d eps_cd =
    w_c w_d / r and d J_DTN/d eps = sum_terms (dJ/dr) (w (x) w)/r. Uniform molecular & PBC."""
    C = int(d.cutoff_smoothness)
    sd = d.structure_data
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([ru, rd], 0)
    n_up = ru.shape[0]
    Ne = all_e.shape[0]
    spin = np.array([0] * n_up + [1] * (Ne - n_up))
    out = jnp.zeros((3, 3))

    if d.use_u:  # fold the pair loop; displacement virial sum_p h'(w (x) w)/r.
        Lu = _dtn_effective_cutoffs(d, jnp.float64)[0]
        alpha = _inflate_alpha(d)  # (2, Nu+1)
        ia, ib = np.triu_indices(Ne, 1)
        w = _md(all_e[ia] - all_e[ib], d)  # (n_pair, 3)
        r = jnp.linalg.norm(w, axis=-1)  # (n_pair,)
        ch = np.where(spin[ia] == spin[ib], 0, 1)  # (n_pair,) static channel index
        _, h1, _, _ = _h3_b(Lu, alpha[ch], r, C)  # (n_pair,)
        out = out + jnp.einsum("p,pc,pd->cd", h1 / r, w, w)

    if d.use_chi:  # fold the electron/atom loops into an (Ne, n_atom) grid.
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1]
        beta = _inflate_beta(d)
        w = _md(all_e[:, None, :] - R[None, :, :], d)  # (Ne, n_atom, 3)
        r = jnp.linalg.norm(w, axis=-1)  # (Ne, n_atom)
        L_at = Lchi[sp]  # (n_atom,)
        beta_at = beta[sp]  # (n_atom, Nc+1)
        _, h1, _, _ = _h3_b(L_at[None, :], beta_at[None, :, :], r, C)  # (Ne, n_atom)
        out = out + jnp.einsum("eI,eIc,eId->cd", h1 / r, w, w)

    if d.use_f:  # reduce the (n_atom, n_pair, order^3) grid over PAIR CHUNKS (adjoint-order).
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2]
        gamma = _inflate_gamma(d)
        ia, ib = np.triu_indices(Ne, 1)
        Mg_at = gamma[sp]  # (n_atom, NeN+1, NeN+1, Nee+1)
        L_at = Lf[sp]  # (n_atom,)
        n_atom = R.shape[0]
        n_pair = int(ia.shape[0])
        if n_pair:  # rolled lax.scan over PAIR CHUNKS; displacement virial; padded pairs masked to zero

            def _strain_body(
                st_c: jax.Array,
                xs: tuple[jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, None]:
                ra_c, rb_c, m = xs  # (chunk, 3), (chunk, 3), (chunk,)
                x = _md(ra_c[None, :, :] - R[:, None, :], d)  # (n_atom, chunk, 3)
                y = _md(rb_c[None, :, :] - R[:, None, :], d)
                z = _md(ra_c - rb_c, d)  # (chunk, 3); R-independent
                r_x = jnp.linalg.norm(x, axis=-1)  # (n_atom, chunk)
                r_y = jnp.linalg.norm(y, axis=-1)
                r_z = jnp.linalg.norm(z, axis=-1)  # (chunk,)
                A, A1, _, _ = _P3_b(L_at[:, None], r_x, NeN, C)  # (n_atom, chunk, NeN+1)
                B, B1, _, _ = _P3_b(L_at[:, None], r_y, NeN, C)
                N0, N1, _, _ = _Cn3_b(r_z, Nee)  # (chunk, Nee+1)
                coefA = jnp.einsum("Ilmn,Ipl,Ipm,pn->Ip", Mg_at, A1, B, N0) * m[None, :]  # (n_atom, chunk)
                coefB = jnp.einsum("Ilmn,Ipl,Ipm,pn->Ip", Mg_at, A, B1, N0) * m[None, :]
                coefN = jnp.einsum("Ilmn,Ipl,Ipm,pn->Ip", Mg_at, A, B, N1) * m[None, :]
                st_c = (
                    st_c
                    + jnp.einsum("Ip,Ipc,Ipd->cd", coefA / r_x, x, x)
                    + jnp.einsum("Ip,Ipc,Ipd->cd", coefB / r_y, y, y)
                    + jnp.einsum("Ip,pc,pd->cd", coefN / r_z[None, :], z, z)
                )
                return st_c, None

            xs = _pair_chunks(_strain_body, jnp.zeros((3, 3)), lambda iaC, ibC, mask: (all_e[iaC], all_e[ibC], mask), ia, ib)
            st_f, _ = jax.lax.scan(_strain_body, jnp.zeros((3, 3)), xs)
            out = out + st_f
    if sd.pbc_flag:
        policy = dtn_cutoff_policy_value(d, ru, rd)
        out = out + (policy / 3.0) * jnp.eye(3, dtype=out.dtype)
    return out


# --------------------------------------------------------------------------- #
# kinetic side:  the DTN contribution to d T / d{r, R, eps}. All three come from
# the same per-displacement gradients dT/dw (cot = gJ + gD held fixed):
#   dr[e]      = sum_{w containing e} (+/-) dT/dw       (chain rule)
#   dR[I]      = -sum_{w at atom I} dT/dw               (translation identity)
#   stress     = sum_w (dT/dw) (x) w_min                (displacement virial)
# --------------------------------------------------------------------------- #
@_compile_stable
def _dtn_kinetic_rRstrain(
    d: Jastrow_DTN_data,
    cot: jax.Array,
    r_up: jax.Array,
    r_dn: jax.Array,
    wts: jax.Array | None = None,
) -> KineticDerivativeFields:
    """Return (dr (Ne,3), dR (n_atom,3), stress (3,3)) for the DTN kinetic contribution.

    ``wts`` (optional per-electron weights, shape (Ne,)): when ``None`` every electron
    contributes with weight 1 -- the unweighted ``d T_DTN/dX`` used by the VMC force/stress.
    The GFMC S'-reduction passes the S' membership ``w_k`` so the result is the WEIGHTED DTN
    kinetic reverse ``d(sum_k w_k T_DTN,k)/dX`` (every per-electron kinetic seed -- the
    ``-1/2 lap`` term and the ``cot`` cotangent -- scaled by ``w_k``).  Because the u/chi/f
    channels build from displacements, ``st == affine_virial(dr, dR)`` holds for molecular AND
    PBC min-image (so the molecular GFMC stress can take the affine virial of the weighted
    position gradients, while the periodic path adds ``st`` directly)."""
    C = int(d.cutoff_smoothness)
    sd = d.structure_data
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([ru, rd], 0)
    n_up = ru.shape[0]
    Ne = all_e.shape[0]
    n_atom = R.shape[0]
    spin = np.array([0] * n_up + [1] * (Ne - n_up))
    # Complex-Bloch (twist): ``cot`` is the COMPLEX determinant drift ``gD + gJ`` (holomorphic in the
    # Bloch orbitals).  The u/chi/f feature fields below are REAL functions of the min-image displacements,
    # so this kinetic reverse is LINEAR in ``cot`` with real coefficients; hence Re-projecting the
    # ASSEMBLED (dr, dR, st) at the boundary equals taking ``Re(cot)`` up front (no re-projection trap --
    # there is no second complex factor).  ``jnp.real`` is a no-op at real Gamma (byte-identical).
    cot = jnp.real(jnp.asarray(cot)).astype(jnp.float64)
    wv = jnp.ones((Ne,), jnp.float64) if wts is None else jnp.asarray(wts, jnp.float64)
    dr = jnp.zeros((Ne, 3))
    dR = jnp.zeros((n_atom, 3))
    st = jnp.zeros((3, 3))

    # ---- u-channel (e-e): pair gradient g = dS_pair/dw, S_pair = -1/2(w_a+w_b) Lp
    #      - h'((w_a cot_a - w_b cot_b).u); the shared radial Laplacian carries the pair weight. ----
    if d.use_u:  # fold the pair loop into an (n_pair,) grid.
        Lu = _dtn_effective_cutoffs(d, jnp.float64)[0]
        alpha = _inflate_alpha(d)
        ia, ib = np.triu_indices(Ne, 1)
        w = _md(all_e[ia] - all_e[ib], d)  # (n_pair, 3)
        r = jnp.linalg.norm(w, axis=-1)  # (n_pair,)
        u = w / r[:, None]
        ch = np.where(spin[ia] == spin[ib], 0, 1)  # (n_pair,) static channel index
        _, h1, h2, h3 = _h3_b(Lu, alpha[ch], r, C)  # (n_pair,)
        Lp1 = h3 + 2.0 * h2 / r - 2.0 * h1 / r**2  # (h''+2h'/r)'
        wa, wb = wv[ia], wv[ib]  # (n_pair,)
        s = wa[:, None] * cot[ia] - wb[:, None] * cot[ib]  # (n_pair, 3)
        su = (s * u).sum(-1)  # (n_pair,)
        g = (-0.5 * (wa + wb) * Lp1)[:, None] * u - (h2 * su)[:, None] * u - (h1 / r)[:, None] * (s - su[:, None] * u)
        dr = dr.at[ia].add(g).at[ib].add(-g)
        st = st + jnp.einsum("pc,pd->cd", g, w)

    # ---- chi-channel (e-N): single-electron -> the whole per-atom gradient scaled by w_e ----
    if d.use_chi:  # fold the electron/atom loops into an (Ne, n_atom) grid.
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1]
        beta = _inflate_beta(d)
        w = _md(all_e[:, None, :] - R[None, :, :], d)  # (Ne, n_atom, 3)
        r = jnp.linalg.norm(w, axis=-1)  # (Ne, n_atom)
        u = w / r[..., None]
        L_at = Lchi[sp]  # (n_atom,)
        beta_at = beta[sp]  # (n_atom, Nc+1)
        _, h1, h2, h3 = _h3_b(L_at[None, :], beta_at[None, :, :], r, C)  # (Ne, n_atom)
        Lp1 = h3 + 2.0 * h2 / r - 2.0 * h1 / r**2
        cu = jnp.einsum("eId,ed->eI", u, cot)  # cot[e].u[e,I]  (Ne, n_atom)
        g = wv[:, None, None] * (
            -0.5 * Lp1[..., None] * u - (h2 * cu)[..., None] * u - (h1 / r)[..., None] * (cot[:, None, :] - cu[..., None] * u)
        )  # (Ne, n_atom, 3)
        dr = dr + g.sum(1)  # dr[e] += sum_I g[e,I]
        dR = dR - g.sum(0)  # dR[I] += -sum_e g[e,I]
        st = st + jnp.einsum("eIc,eId->cd", g, w)

    # ---- f-channel (e-e-N): three per-displacement gradients (dSdx, dSdy, dSdz); the a-side lap /
    #      cot pieces carry w_a, the b-side pieces w_b (threaded into _f_triple_grads). ----
    if d.use_f:  # vmap _f_triple_grads over (atom, pair) in PAIR CHUNKS (adjoint-order).
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2]
        gamma = _inflate_gamma(d)
        ia, ib = np.triu_indices(Ne, 1)
        Mg_at = gamma[sp]  # (n_atom, NeN+1, NeN+1, Nee+1)
        L_at = Lf[sp]  # (n_atom,)
        n_pair = int(ia.shape[0])

        def _one(
            Mg: jax.Array,
            L: jax.Array,
            ra: jax.Array,
            rb: jax.Array,
            RI: jax.Array,
            cot_a: jax.Array,
            cot_b: jax.Array,
            wa: jax.Array,
            wb: jax.Array,
        ) -> TripleDisplacementGradients:
            return _f_triple_grads(Mg, L, C, NeN, Nee, ra, rb, RI, cot_a, cot_b, d, wa, wb)

        over_pairs = jax.vmap(_one, in_axes=(None, None, 0, 0, None, 0, 0, 0, 0))  # over the chunk's pairs
        over_grid = jax.vmap(over_pairs, in_axes=(0, 0, None, None, 0, None, None, None, None))  # over atoms
        # A field-first vmap over (atom, pair) creates an
        # (n_atom, n_pair, order-fields) intermediate that reaches 2.9 GB at 128 atoms.
        # Chunk the pair axis in a rolled lax.scan so only one chunk's
        # vmap working set is live); each chunk is fully vectorized; padded pairs masked to zero.
        if n_pair:

            def _kin_body(
                carry: KineticDerivativeFields,
                xs: tuple[
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
            ) -> tuple[KineticDerivativeFields, None]:
                dr_c, dR_c, st_c = carry
                ra_c, rb_c, cota_c, cotb_c, wa_c, wb_c, ic_j, jc_j, m = xs  # (chunk, ...) leaves
                gx, gy, gz, x, y, z = over_grid(Mg_at, L_at, ra_c, rb_c, R, cota_c, cotb_c, wa_c, wb_c)  # (n_atom, chunk, 3)
                mm = m[None, :, None]  # (1, chunk, 1) -- zero out padded pairs
                gx, gy, gz = gx * mm, gy * mm, gz * mm
                dr_c = dr_c.at[ic_j].add((gx + gz).sum(0)).at[jc_j].add((gy - gz).sum(0))  # sum atoms, scatter
                dR_c = dR_c - (gx + gy).sum(1)  # translation identity
                st_c = (
                    st_c
                    + jnp.einsum("Ipc,Ipd->cd", gx, x)
                    + jnp.einsum("Ipc,Ipd->cd", gy, y)
                    + jnp.einsum("Ipc,Ipd->cd", gz, z)
                )
                return (dr_c, dR_c, st_c), None

            def _kin_xs(iaC: np.ndarray, ibC: np.ndarray, mask: jax.Array) -> tuple[jax.Array, ...]:
                iaC_j, ibC_j = jnp.asarray(iaC), jnp.asarray(ibC)
                return (all_e[iaC], all_e[ibC], cot[iaC], cot[ibC], wv[iaC], wv[ibC], iaC_j, ibC_j, mask)

            xs = _pair_chunks(_kin_body, (dr, dR, st), _kin_xs, ia, ib)
            (dr, dR, st), _ = jax.lax.scan(_kin_body, (dr, dR, st), xs)
    if sd.pbc_flag:
        _, policy_gu, policy_gd, policy_lu, policy_ld = dtn_cutoff_policy_fields(d, ru, rd)
        policy_g = jnp.concatenate((policy_gu, policy_gd), axis=0)
        policy_l = jnp.concatenate((policy_lu, policy_ld), axis=0)
        policy_D = -0.5 * jnp.sum(wv * policy_l) - jnp.sum(wv[:, None] * cot * policy_g)
        st = st + (policy_D / 3.0) * jnp.eye(3, dtype=st.dtype)
    return dr, dR, st


def _f_triple_grads(
    Mg: jax.Array,
    L: jax.Array,
    C: int,
    NeN: int,
    Nee: int,
    ra: jax.Array,
    rb: jax.Array,
    RI: jax.Array,
    cot_a: jax.Array,
    cot_b: jax.Array,
    d: Jastrow_DTN_data,
    wa: jax.Array | float = 1.0,
    wb: jax.Array | float = 1.0,
) -> TripleDisplacementGradients:
    """Per-displacement gradients (dS/dx, dS/dy, dS/dz) of one f-channel triple's kinetic
    contribution S = -1/2(w_a lap_a + w_b lap_b) - w_a cot_a.grad_a - w_b cot_b.grad_b, with
    x=r_a-R_I, y=r_b-R_I, z=r_a-r_b (min-image displacements; cot held fixed).  ``w_a``/``w_b``
    (default 1) are the per-electron S' weights on the a-/b-electron kinetic seeds.  Returns the
    three gradients and the three min-image displacements."""
    x = _md(ra - RI, d)
    y = _md(rb - RI, d)
    z = _md(ra - rb, d)
    r_x, r_y, r_z = jnp.linalg.norm(x), jnp.linalg.norm(y), jnp.linalg.norm(z)
    ux, uy, uz = x / r_x, y / r_y, z / r_z
    A, A1, A2, A3 = _P3(L, r_x, NeN, C)
    B, B1, B2, B3 = _P3(L, r_y, NeN, C)
    N0, N1, N2, N3 = _Cn3(r_z, Nee)

    def G(i: jax.Array, j: jax.Array, n: jax.Array) -> jax.Array:
        return jnp.einsum("lmn,l,m,n->", Mg, i, j, n)

    LAx = A2 + 2.0 * A1 / r_x
    LAx1 = A3 + 2.0 * A2 / r_x - 2.0 * A1 / r_x**2
    LBy = B2 + 2.0 * B1 / r_y
    LBy1 = B3 + 2.0 * B2 / r_y - 2.0 * B1 / r_y**2
    LNz = N2 + 2.0 * N1 / r_z
    LNz1 = N3 + 2.0 * N2 / r_z - 2.0 * N1 / r_z**2

    coefA = G(A1, B, N0)
    coefB = G(A, B1, N0)
    coefN = G(A, B, N1)
    coefAN = G(A1, B, N1)
    coefBN = G(A, B1, N1)
    cosI = ux @ uz
    cosJ = uy @ uz
    sa_x = cot_a @ ux
    sa_z = cot_a @ uz
    sb_y = cot_b @ uy
    sb_z = cot_b @ uz

    # ---- d/dx ----
    dphi_aa = G(A2, B, N0) * sa_x * ux + coefA * (cot_a - sa_x * ux) / r_x + coefAN * sa_z * ux
    dphi_ba = G(A1, B1, N0) * sb_y * ux - coefAN * sb_z * ux
    dlap_a = (G(LAx1, B, N0) + 2.0 * cosI * G(A2, B, N1) + G(A1, B, LNz)) * ux + 2.0 * coefAN * (uz - cosI * ux) / r_x
    dlap_b = (G(A1, LBy, N0) - 2.0 * cosJ * G(A1, B1, N1) + G(A1, B, LNz)) * ux
    dSdx = -0.5 * (wa * dlap_a + wb * dlap_b) - wa * dphi_aa - wb * dphi_ba

    # ---- d/dy ----
    dphi_aa = G(A1, B1, N0) * sa_x * uy + coefBN * sa_z * uy
    dphi_ba = G(A, B2, N0) * sb_y * uy + coefB * (cot_b - sb_y * uy) / r_y - coefBN * sb_z * uy
    dlap_a = (G(LAx, B1, N0) + 2.0 * cosI * G(A1, B1, N1) + G(A, B1, LNz)) * uy
    dlap_b = (G(A, LBy1, N0) - 2.0 * cosJ * G(A, B2, N1) + G(A, B1, LNz)) * uy - 2.0 * coefBN * (uz - cosJ * uy) / r_y
    dSdy = -0.5 * (wa * dlap_a + wb * dlap_b) - wa * dphi_aa - wb * dphi_ba

    # ---- d/dz ----  (coefA=G(A',B,N0) and coefB=G(A,B',N0) depend on r_z via N0=r_z^n)
    dphi_aa = coefAN * sa_x * uz + G(A, B, N2) * sa_z * uz + coefN * (cot_a - sa_z * uz) / r_z
    dphi_ba = coefBN * sb_y * uz - (G(A, B, N2) * sb_z * uz + coefN * (cot_b - sb_z * uz) / r_z)
    dlap_a = (G(LAx, B, N1) + 2.0 * cosI * G(A1, B, N2) + G(A, B, LNz1)) * uz + 2.0 * coefAN * (ux - cosI * uz) / r_z
    dlap_b = (G(A, LBy, N1) - 2.0 * cosJ * G(A, B1, N2) + G(A, B, LNz1)) * uz - 2.0 * coefBN * (uy - cosJ * uz) / r_z
    dSdz = -0.5 * (wa * dlap_a + wb * dlap_b) - wa * dphi_aa - wb * dphi_ba

    return dSdx, dSdy, dSdz, x, y, z


def grad_T_dtn_wrt_r(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """DTN contribution to d T / d r per electron (Ne, 3). cot = gJ + gD computed here."""
    cot = _dtn_cotangent(d, jastrow_data, gd, r_up, r_dn, geminal_inv=geminal_inv)
    dr, _, _ = _dtn_kinetic_rRstrain(d, cot, r_up, r_dn)
    return dr


def grad_T_dtn_wrt_R(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """DTN contribution to the kinetic nuclear force d T / d R (n_atom, 3)."""
    cot = _dtn_cotangent(d, jastrow_data, gd, r_up, r_dn, geminal_inv=geminal_inv)
    _, dR, _ = _dtn_kinetic_rRstrain(d, cot, r_up, r_dn)
    return dR


def grad_T_dtn_wrt_strain(
    d: Jastrow_DTN_data,
    jastrow_data: Jastrow_data,
    gd: AGP_data | AGPn_data | SD_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    geminal_inv: jax.Array | None = None,
) -> jax.Array:
    """DTN contribution to the kinetic stress d T / d eps (3, 3); displacement virial
    (correct for molecular AND PBC min-image)."""
    cot = _dtn_cotangent(d, jastrow_data, gd, r_up, r_dn, geminal_inv=geminal_inv)
    _, _, st = _dtn_kinetic_rRstrain(d, cot, r_up, r_dn)
    return st


# --------------------------------------------------------------------------- #
# Fused DTN drift-mesh kernel: the full per-config
# drift field O[c, j] = grad_j J_DTN(config c) for one-electron-moved configs,
# batched over the generic config axis -- replaces the 768-serial
# `_dtn_drift_probe` stream in one_move_mesh.dtn_mesh.
# --------------------------------------------------------------------------- #
def _dtn_mesh_stream(
    one_config_fn: Callable[[jax.Array, jax.Array], OneMoveMeshResult],
    pts: jax.Array,
    e_idx: jax.Array,
) -> OneMoveMeshResult:
    """Stream a one-move DTN mesh kernel over its config axis.

    ``_mesh_map`` with the DTN-owned PER-WALKER budget
    ``JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES`` instead of the generic mesh one:
    the chunk width is the measured workspace of ``one_config_fn`` against that
    budget (see ``jqmc._setting.measured_closure_stream_batch``).
    """
    from jqmc._setting import JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES

    from .mesh_stream import _mesh_map

    return _mesh_map(one_config_fn, (pts, e_idx), budget_bytes=JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES)


def dtn_drift_mesh_dense(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    pts: jax.Array,
    e_idx: jax.Array,
) -> jax.Array:
    """Rank-one DTN drift field over one-electron-moved mesh configs.

    Config ``c`` moves the GLOBAL electron ``m = e_idx[c]`` from its base
    position ``q = r_m`` to ``pts[c]``; every other electron stays at base.
    Exact decomposition:

    * moved row:   ``O[c, m] = grad_p J(config)`` -- fresh single-electron
      gradient (u vs all partners, chi vs all atoms, f moved-leg over atoms
      x all partners);
    * unmoved row: ``O[c, j] = gJ_base[j] + delta(c, j)`` where ``delta``
      holds ONLY the pair terms involving ``m`` (u pair-gradient difference
      at ``pts`` vs at ``q``; f partner-leg difference; chi has no pair
      structure -> zero).

    ``gJ_base`` is the dense production kernel evaluated once. The f e-e directions use the
    per-lane minimum-image vector ``w_pe`` already computed for the radii. A
    raw-coordinate factorization such as
    ``p*sum(c/r) - sum((c/r)*e_nbr)`` would drop the image shift and be wrong
    at PBC whenever an f-active pair straddles a cell boundary.

    Densely every electron is a partner of every atom, so no per-atom
    membership structure is required. This is not merely
    ``max_ne = N_e``: a per-atom neighbor-list form carries three costs that
    exist ONLY because each atom has a DIFFERENT partner list, and the dense
    form has none of them:

    * with an atom-dependent ``e_nbr`` the e-e leg ``w_pe`` must be built as
      ``(K, N_atom, max_ne, 3)``. Dense partners are the SAME for every atom,
      so ``w_pe`` is ``(K, N_e, 3)`` -- an ``N_atom``-fold memory saving (at
      128 atoms that is 300 MB -> 2.4 MB per walker).
    * a slot -> global scatter needs a ``(N_atom, max_ne, N_e)`` one-hot
      ``member`` matrix and a 3-factor einsum; densely ``member`` IS the
      identity, so the scatter collapses to ``"kIj,Ijc->kjc"`` with no
      one-hot.
    * there is no capacity, hence no overflow and no NaN poison.

    The surviving ``(K, N_atom, N_e)`` coefficient tensors are inherent to the
    three-body f term.

    Args:
        d: ``Jastrow_DTN_data``.
        r_up: base up-electron positions.
        r_dn: base dn-electron positions.
        pts: ``(K, 3)`` displaced positions (any one-move mesh -- the LRDMC
            kinetic lattice or the ECP quadrature; the caller owns the mesh).
        e_idx: ``(K,)`` GLOBAL moved-electron index per config.

    Returns:
        ``(dJ, O)``: the ``(K,)`` one-move VALUE deltas
        ``J(config) - J(base)`` and the ``(K, Ne, 3)`` drift field (both
        fp64; the value shares the drift's coefficient tensors, so no
        separate per-config value stream is needed).
    """
    from jqmc.jastrow_factor import compute_grads_and_laplacian_Jastrow_DTN

    sd = d.structure_data
    C = int(d.cutoff_smoothness)
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([ru, rd], 0)
    n_up = ru.shape[0]
    Ne = all_e.shape[0]
    pts = jnp.asarray(pts, jnp.float64)
    e_idx = jnp.asarray(e_idx, jnp.int32)
    spin = jnp.arange(Ne) < n_up
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)

    # Base drift ONCE (dense production kernel == the streamed oracle's base).
    g_up, g_dn, _, _ = compute_grads_and_laplacian_Jastrow_DTN(d, ru, rd)
    gJ_base = jnp.concatenate([jnp.asarray(g_up, jnp.float64), jnp.asarray(g_dn, jnp.float64)], 0)  # (Ne, 3)

    # --- hoisted, CONFIG-INDEPENDENT setup -------------------------------
    # Everything below the chunk boundary is built ONCE per call. Chunking the
    # configs without this hoist re-pays the base field and the j-side legs per
    # chunk -- MEASURED at 3.2x on this kernel (the naive form).
    Lu = _dtn_effective_cutoffs(d, jnp.float64)[0] if d.use_u else None
    alpha = _inflate_alpha(d) if d.use_u else None  # (2, Nu+1)
    Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1][sp] if d.use_chi else None  # (Na,)
    beta = _inflate_beta(d)[sp] if d.use_chi else None  # (Na, Nchi+1)
    if d.use_f:
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2][sp]  # (Na,)
        Mg = _inflate_gamma(d)[sp]  # (Na, NeN+1, NeN+1, Nee+1)
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
        # j-side legs: config-INDEPENDENT, and indexed by the GLOBAL
        # electron j, not by a per-atom slot.
        x_jI = _md(all_e[None, :, :] - R[:, None, :], d)  # (Na, Ne, 3)
        r_jI = jnp.linalg.norm(x_jI, axis=-1)
        r_jI_safe = jnp.where(r_jI > 0.0, r_jI, 1.0)
        Pj, P1j, _, _ = _P3_b(Lf[:, None], r_jI, NeN, C)  # (Na, Ne, NeN+1)
        uhat_jI = x_jI / r_jI_safe[..., None]  # (Na, Ne, 3)

    def _one(pt: jax.Array, ei: jax.Array) -> OneMoveMeshResult:
        """ONE config: ``(dJ_c (), O_c (Ne, 3))``.

        The (K, Na, Ne) coefficient tensors of the batched form become
        (Na, Ne) here, so the streamed peak is O(chunk) not O(K).
        """
        q_c = all_e[ei]
        notself_c = (jnp.arange(Ne) != ei).astype(jnp.float64)  # (Ne,)
        dJ_c = jnp.zeros((), jnp.float64)
        row_c = jnp.zeros((3,), jnp.float64)
        delta_c = jnp.zeros((Ne, 3), jnp.float64)

        # --- u / chi: exact rank-one closed forms (no membership involved) --
        if d.use_u:
            ch = (spin[ei] != spin).astype(jnp.int32)  # (Ne,)
            a_sel = alpha[ch]  # (Ne, Nu+1)
            for src, sign_row in ((pt, 1.0), (q_c, -1.0)):
                w = _md(src[None, :] - all_e, d)  # (Ne, 3)
                r = jnp.linalg.norm(w, axis=-1)
                r_safe = jnp.where(r > 0.0, r, 1.0)
                h0, h1, _, _ = _h3_b(Lu, a_sel, r, C)  # (Ne,)
                coef = notself_c * h1 / r_safe
                dJ_c = dJ_c + sign_row * jnp.sum(notself_c * h0)
                if sign_row > 0:
                    row_c = row_c + jnp.einsum("j,jc->c", coef, w)
                    delta_c = delta_c - coef[:, None] * w
                else:
                    delta_c = delta_c + coef[:, None] * w

        if d.use_chi:
            wI = _md(pt[None, :] - R, d)  # (Na, 3)
            rI = jnp.linalg.norm(wI, axis=-1)
            rI_safe = jnp.where(rI > 0.0, rI, 1.0)
            h0, h1, _, _ = _h3_b(Lchi, beta, rI, C)  # (Na,)
            row_c = row_c + jnp.einsum("I,Ic->c", h1 / rI_safe, wI)
            rIq = jnp.linalg.norm(_md(q_c[None, :] - R, d), axis=-1)
            h0q, _, _, _ = _h3_b(Lchi, beta, rIq, C)
            dJ_c = dJ_c + jnp.sum(h0 - h0q)

        if d.use_f:
            excl = notself_c[None, :]  # (1, Ne), broadcasts over atoms

            def _f_terms(
                src: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                wI_ = _md(src[None, :] - R, d)  # (Na, 3)
                rI_ = jnp.linalg.norm(wI_, axis=-1)
                rI_s = jnp.where(rI_ > 0.0, rI_, 1.0)
                Pp, P1p, _, _ = _P3_b(Lf, rI_, NeN, C)  # (Na, NeN+1)
                # e-e leg: ATOM-INDEPENDENT densely (the partner set is the
                # same for every atom). Still the per-lane MINIMUM-IMAGE
                # vector required by the f-channel pair contraction.
                w_pe = _md(src[None, :] - all_e, d)  # (Ne, 3)
                r_pe = jnp.linalg.norm(w_pe, axis=-1)  # (Ne,)
                r_pe_s = jnp.where(r_pe > 0.0, r_pe, 1.0)
                N0, N1, _, _ = _Cn3_b(r_pe, Nee)  # (Ne, Nee+1)
                A = jnp.einsum("Ilmn,Il->Imn", Mg, Pp)
                A1 = jnp.einsum("Ilmn,Il->Imn", Mg, P1p)
                coef_val = jnp.einsum("Imn,Ijm,jn->Ij", A, Pj, N0) * excl
                coef_row = jnp.einsum("Imn,Ijm,jn->Ij", A1, Pj, N0) * excl
                coef_jeN = jnp.einsum("Imn,Ijm,jn->Ij", A, P1j, N0) * excl
                coef_ee = jnp.einsum("Imn,Ijm,jn->Ij", A, Pj, N1) * excl
                cee_r = coef_ee / r_pe_s[None, :]  # (Na, Ne)
                rc = jnp.einsum("Ij,Ic->c", coef_row, wI_ / rI_s[:, None])
                rc = rc + jnp.einsum("Ij,jc->c", cee_r, w_pe)
                return rc, coef_jeN, cee_r, w_pe, jnp.sum(coef_val)

            row_p, jeN_p, cee_p, wpe_p, val_p = _f_terms(pt)
            row_q, jeN_q, cee_q, wpe_q, val_q = _f_terms(q_c)
            row_c = row_c + row_p
            dJ_c = dJ_c + val_p - val_q
            # j-row scatter: densely the slot -> global `member` one-hot is the
            # identity, so the scatter is a plain contraction over the atoms.
            d_jeN = jnp.einsum("Ij,Ijc->jc", jeN_p - jeN_q, uhat_jI)
            d_ee_p = -jnp.einsum("Ij,jc->jc", cee_p, wpe_p)
            d_ee_q = -jnp.einsum("Ij,jc->jc", cee_q, wpe_q)
            delta_c = delta_c + d_jeN + d_ee_p - d_ee_q

        O_c = gJ_base + delta_c * notself_c[:, None]
        return dJ_c, O_c.at[ei].set(row_c)

    return _dtn_mesh_stream(_one, pts, e_idx)


def dtn_param_mesh_weighted(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    pts: jax.Array,
    e_idx: jax.Array,
    weights: jax.Array,
) -> dict[str, jax.Array]:
    """Weighted one-electron-move DTN parameter gradients over a mesh.

    Returns, for the active channels, ``sum_c weights[c] * (grad_dtn_wrt_param(config_c) -
    grad_dtn_wrt_param(base))`` with the same keys (``alpha_indep`` / ``beta_indep`` /
    ``gamma_indep``). Config ``c`` moves the GLOBAL electron ``e_idx[c]`` to ``pts[c]``.

    ``J_DTN`` is linear in its parameters and its features are sums over e-e pairs (u),
    e-N legs (chi) and e-e-N triples (f), so the config-minus-base feature difference holds
    only the terms of the moved electron: its pairs, its legs and the triples with one of its
    pairs, evaluated at ``pts[c]`` minus at its base position. Those differences are weighted
    and summed over the mesh, then chained through ``M_alpha`` / ``M_beta`` / ``M_gamma``
    once, exactly as :func:`grad_dtn_wrt_param` does for the full sums. The f triples keep the
    dense kernel's a < b orientation (the moved electron is the a leg iff ``ei < partner``).
    Cost ``O(N_e N_atom)`` per config instead of the dense ``O(N_e^2 N_atom)``; the dense
    per-config kernel stays the parity oracle.
    """
    from jqmc._setting import JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES

    from .mesh_stream import _mesh_wsum

    C = int(d.cutoff_smoothness)
    R = jnp.asarray(d.structure_data.positions, jnp.float64)
    sp = np.asarray(d.species_index, np.int32)
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([ru, rd], 0)
    Ne = int(all_e.shape[0])
    n_up = int(ru.shape[0])
    electron_ids = jnp.arange(Ne, dtype=jnp.int32)
    spin_up = electron_ids < n_up
    pts = jnp.asarray(pts, jnp.float64)
    e_idx = jnp.asarray(e_idx, jnp.int32)
    cutoffs = _dtn_effective_cutoffs(d, jnp.float64)

    if d.use_u:
        Lu = cutoffs[0]
        ku = jnp.arange(int(d.n_order_u) + 1)
    if d.use_chi:
        Lchi_at = cutoffs[1][sp]  # (Na,)
        kc = jnp.arange(int(d.n_order_chi) + 1)
    if d.use_f:
        Lf_at = cutoffs[2][sp]  # (Na,)
        kl = jnp.arange(int(d.n_order_eN) + 1)
        kn = jnp.arange(int(d.n_order_ee) + 1)
        # Partner e-N legs phi(r_jI) r_jI^l do not depend on the moved electron's position.
        d_pI = jnp.linalg.norm(_md(all_e[None, :, :] - R[:, None, :], d), axis=-1)  # (Na, Ne)
        P_pI = (jnp.maximum(Lf_at[:, None] - d_pI, 0.0) ** C)[..., None] * d_pI[..., None] ** kl  # (Na, Ne, NeN+1)

    def _terms(src: jax.Array, ei: jax.Array) -> dict[str, jax.Array]:
        """Features of the DTN terms containing electron ``ei`` placed at ``src``."""
        notself = (electron_ids != ei).astype(jnp.float64)
        out = {}
        if d.use_u:
            rr = jnp.linalg.norm(_md(src[None, :] - all_e, d), axis=-1)  # (Ne,)
            h = (notself * jnp.maximum(Lu - rr, 0.0) ** C)[:, None] * rr[:, None] ** ku  # (Ne, Nu+1)
            opposite = (spin_up[ei] != spin_up).astype(jnp.float64)
            out["u"] = jnp.stack([((1.0 - opposite)[:, None] * h).sum(0), (opposite[:, None] * h).sum(0)], 0)
        if d.use_chi:
            dI = jnp.linalg.norm(_md(src[None, :] - R, d), axis=-1)  # (Na,)
            out["chi"] = (jnp.maximum(Lchi_at - dI, 0.0) ** C)[:, None] * dI[:, None] ** kc  # (Na, Nchi+1)
        if d.use_f:
            dI = jnp.linalg.norm(_md(src[None, :] - R, d), axis=-1)  # (Na,)
            P_m = (jnp.maximum(Lf_at - dI, 0.0) ** C)[:, None] * dI[:, None] ** kl  # (Na, NeN+1)
            moved_is_a = ei < electron_ids
            z = _md(jnp.where(moved_is_a[:, None], src[None, :] - all_e, all_e - src[None, :]), d)
            N = jnp.linalg.norm(z, axis=-1)[:, None] ** kn  # (Ne, Nee+1)
            a_mask = (moved_is_a.astype(jnp.float64) * notself)[None, :, None]
            b_mask = ((~moved_is_a).astype(jnp.float64) * notself)[None, :, None]
            out["f"] = jnp.einsum("Il,Ijm,jn->Ilmn", P_m, P_pI * a_mask, N) + jnp.einsum(
                "Ijl,Im,jn->Ilmn", P_pI * b_mask, P_m, N
            )  # (Na, NeN+1, NeN+1, Nee+1)
        return out

    # Base-position terms depend only on the moved electron: once per electron.
    old_terms = jax.vmap(_terms)(all_e, electron_ids)

    def _one(pt: jax.Array, ei: jax.Array) -> dict[str, jax.Array]:
        new = _terms(pt, ei)
        return {k: new[k] - old_terms[k][ei] for k in new}

    wsum = _mesh_wsum(_one, jnp.asarray(weights), (pts, e_idx), budget_bytes=JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES)

    out = {}
    if d.use_u:
        out["alpha_indep"] = jnp.einsum("sk,ski->si", wsum["u"], jnp.asarray(d.M_alpha))
    if d.use_chi:
        n_sp = int(np.asarray(d.M_beta).shape[0])
        feat = jnp.zeros((n_sp, wsum["chi"].shape[-1]), wsum["chi"].dtype).at[sp].add(wsum["chi"])
        out["beta_indep"] = jnp.einsum("sk,ski->si", feat, jnp.asarray(d.M_beta))
    if d.use_f:
        Mg = jnp.asarray(d.M_gamma)
        feat = jnp.zeros(Mg.shape[:4], wsum["f"].dtype).at[sp].add(wsum["f"])
        out["gamma_indep"] = jnp.einsum("slmn,slmni->si", feat, Mg)
    return out


def dtn_geometry_mesh_weighted(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    pts: jax.Array,
    e_idx: jax.Array,
    weights: jax.Array,
    *,
    compute_force: bool = True,
    compute_strain: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Weighted one-electron-move DTN nuclear/strain deltas.

    For config ``c`` the global electron ``e_idx[c]`` moves from its base
    position to ``pts[c]``.  Only ``chi`` legs touching that electron and
    ``u``/``f`` pairs containing it can change, hence

    ``sum_c weights[c] * (geometry_DTN(config_c) - geometry_DTN(base))``

    is evaluated in ``O(K * (N_atom + N_e*N_atom))`` work rather than calling
    the dense ``O(N_e**2*N_atom)`` geometry kernels for every config.  The
    candidate axis is reduced inside :func:`mesh_stream._mesh_wsum`, so no
    ``(K,N_atom,3)`` force stack is materialised.  This is the JAX analogue of
    Rust ``grad_j_dtn_wrt_r_atom_move_delta`` and
    ``dtn_value_strain_move_delta``.

    The returned arrays keep the weight-promoted dtype (complex at a twist).
    ``compute_force`` / ``compute_strain`` are Python-static route selectors;
    an unrequested output is a shape-correct zero and its channel algebra is
    not lowered.
    """
    from jqmc._setting import JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES

    from .mesh_stream import _mesh_wsum

    one = _dtn_geometry_move_delta(d, r_up, r_dn, compute_force=compute_force, compute_strain=compute_strain)
    pts = jnp.asarray(pts, jnp.float64)
    e_idx = jnp.asarray(e_idx, jnp.int32)
    return _mesh_wsum(one, jnp.asarray(weights), (pts, e_idx), budget_bytes=JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES)


def dtn_force_mesh_delta(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    pts: jax.Array,
    e_idx: jax.Array,
) -> jax.Array:
    """Per-config one-electron-move DTN Pulay-force deltas ``(K, N_atom, 3)``.

    Row ``c`` is ``grad_dtn_wrt_R(config_c) - grad_dtn_wrt_R(base)``, where
    config ``c`` moves the GLOBAL electron ``e_idx[c]`` to ``pts[c]``.  Only
    the chi leg of the moved electron and the f triples containing it change,
    so each row costs ``O(N_e * N_atom)`` instead of the dense
    ``O(N_e**2 * N_atom)`` kernel per config.  The terms are the
    ones :func:`dtn_geometry_mesh_weighted` reduces with weights; this entry
    keeps the per-config stack for callers that contract it later
    (``one_move_mesh.mesh_force``).  The dense per-config kernel
    (``one_move_mesh._dtn_forceR_mesh``) remains the parity oracle.
    """
    from jqmc._setting import JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES

    from .mesh_stream import _mesh_map

    one = _dtn_geometry_move_delta(d, r_up, r_dn, compute_force=True, compute_strain=False)
    pts = jnp.asarray(pts, jnp.float64)
    e_idx = jnp.asarray(e_idx, jnp.int32)
    return _mesh_map(lambda pt, ei: one(pt, ei)[0], (pts, e_idx), budget_bytes=JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES)


def _dtn_geometry_move_delta(
    d: Jastrow_DTN_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    *,
    compute_force: bool,
    compute_strain: bool,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """One-config DTN geometry delta ``(pt, ei) -> (dR (N_atom, 3), strain (3, 3))``.

    The delta is ``terms(pt) - terms(r_ei)`` over the DTN terms that contain the moved
    electron ``ei``; shared by :func:`dtn_geometry_mesh_weighted` and
    :func:`dtn_force_mesh_delta`.
    """
    sd = d.structure_data
    C = int(d.cutoff_smoothness)
    ru = jnp.asarray(r_up, jnp.float64)
    rd = jnp.asarray(r_dn, jnp.float64)
    all_e = jnp.concatenate([ru, rd], 0)
    Ne = int(all_e.shape[0])
    n_up = int(ru.shape[0])
    R = jnp.asarray(sd._positions_cart_jnp, jnp.float64)
    n_atom = int(R.shape[0])
    sp = np.asarray(d.species_index, np.int32)
    electron_ids = jnp.arange(Ne, dtype=jnp.int32)
    spin = electron_ids < n_up

    Lu = _dtn_effective_cutoffs(d, jnp.float64)[0] if d.use_u and compute_strain else None
    alpha = _inflate_alpha(d) if d.use_u and compute_strain else None
    if d.use_chi and (compute_force or compute_strain):
        Lchi = _dtn_effective_cutoffs(d, jnp.float64)[1][sp]
        beta = _inflate_beta(d)[sp]
    if d.use_f and (compute_force or compute_strain):
        Lf = _dtn_effective_cutoffs(d, jnp.float64)[2][sp]
        Mg = _inflate_gamma(d)[sp]
        NeN = int(d.n_order_eN)
        Nee = int(d.n_order_ee)
    f_on = bool(d.use_f and (compute_force or compute_strain))
    if f_on:
        # Partner e-N legs do not depend on the moved electron's position: build them once.
        w_pI = _md(all_e[None, :, :] - R[:, None, :], d)  # (Na, Ne, 3)
        r_pI = jnp.linalg.norm(w_pI, axis=-1)
        r_pI_safe = jnp.where(r_pI > 0.0, r_pI, 1.0)
        P_pI, P1_pI, _, _ = _P3_b(Lf[:, None], r_pI, NeN, C)  # (Na, Ne, NeN+1)
        u_pI = w_pI / r_pI_safe[..., None]

    def _f_terms(src: jax.Array, ei: jax.Array, notself: jax.Array) -> tuple[jax.Array, jax.Array]:
        """f-channel nuclear and strain terms of the triples containing ``ei`` at ``src``.

        Triple ``(I, a, b)`` with ``a < b`` contributes ``Mg[I] . A(x) B(y) N(z)`` where ``x``/``y``
        are the a/b e-N legs and ``z = a - b``.  The moved electron supplies the a leg iff
        ``ei < partner``.  The moved leg is contracted into ``Mg`` first and the partner legs are
        the hoisted ``P_pI`` / ``P1_pI``: ``O(N_atom N_e NeN Nee)`` per position instead of the
        ``O(N_atom N_e NeN^2 Nee)`` of evaluating every triple's coefficients from scratch
        (issue #656; the force-only form of issue #558).
        """
        moved_is_a = ei < electron_ids  # moved electron supplies the a leg iff ei < partner
        w_m = _md(src[None, :] - R, d)  # (Na, 3)
        r_m = jnp.linalg.norm(w_m, axis=-1)
        r_m_safe = jnp.where(r_m > 0.0, r_m, 1.0)
        P_m, P1_m, _, _ = _P3_b(Lf, r_m, NeN, C)  # (Na, NeN+1)
        z = _md(jnp.where(moved_is_a[:, None], src[None, :] - all_e, all_e - src[None, :]), d)
        N0, N1, _, _ = _Cn3_b(jnp.linalg.norm(z, axis=-1), Nee)  # (Ne, Nee+1)
        # moved = a leg: coefA (moved leg) and coefB (partner leg)
        c_mov_a = jnp.einsum("Imn,Ijm,jn->Ij", jnp.einsum("Ilmn,Il->Imn", Mg, P1_m), P_pI, N0)
        c_par_a = jnp.einsum("Imn,Ijm,jn->Ij", jnp.einsum("Ilmn,Il->Imn", Mg, P_m), P1_pI, N0)
        # moved = b leg: coefA (partner leg) and coefB (moved leg)
        c_par_b = jnp.einsum("Iln,Ijl,jn->Ij", jnp.einsum("Ilmn,Im->Iln", Mg, P_m), P1_pI, N0)
        c_mov_b = jnp.einsum("Iln,Ijl,jn->Ij", jnp.einsum("Ilmn,Im->Iln", Mg, P1_m), P_pI, N0)
        mask = notself[None, :]
        c_mov = jnp.where(moved_is_a[None, :], c_mov_a, c_mov_b) * mask
        c_par = jnp.where(moved_is_a[None, :], c_par_a, c_par_b) * mask
        f_dR = jnp.zeros((n_atom, 3), jnp.float64)
        f_strain = jnp.zeros((3, 3), jnp.float64)
        if compute_force:
            f_dR = -(jnp.sum(c_mov, axis=1)[:, None] * (w_m / r_m_safe[:, None]) + jnp.einsum("Ij,Ijc->Ic", c_par, u_pI))
        if compute_strain:
            # e-e leg coefficient: Mg . A B N1 with the moved leg contracted first.
            c_N_a = jnp.einsum("Imn,Ijm,jn->Ij", jnp.einsum("Ilmn,Il->Imn", Mg, P_m), P_pI, N1)
            c_N_b = jnp.einsum("Iln,Ijl,jn->Ij", jnp.einsum("Ilmn,Im->Iln", Mg, P_m), P_pI, N1)
            c_N = jnp.where(moved_is_a[None, :], c_N_a, c_N_b) * mask
            r_z = jnp.linalg.norm(z, axis=-1)
            r_z_safe = jnp.where(r_z > 0.0, r_z, 1.0)
            f_strain = (
                jnp.einsum("I,Ic,Id->cd", jnp.sum(c_mov, axis=1) / r_m_safe, w_m, w_m)
                + jnp.einsum("Ij,Ijc,Ijd->cd", c_par / r_pI_safe, w_pI, w_pI)
                + jnp.einsum("j,jc,jd->cd", jnp.sum(c_N, axis=0) / r_z_safe, z, z)
            )
        return f_dR, f_strain

    def _at_position(src: jax.Array, ei: jax.Array) -> tuple[jax.Array, jax.Array]:
        """DTN geometry terms containing moved electron ``ei`` at ``src``."""
        notself = (electron_ids != ei).astype(jnp.float64)
        dR = jnp.zeros((n_atom, 3), jnp.float64)
        strain = jnp.zeros((3, 3), jnp.float64)

        if d.use_u and compute_strain:
            w = _md(src[None, :] - all_e, d)
            r = jnp.linalg.norm(w, axis=-1)
            r_safe = jnp.where(r > 0.0, r, 1.0)
            channel = (spin[ei] != spin).astype(jnp.int32)
            _, h1, _, _ = _h3_b(Lu, alpha[channel], r, C)
            coef = notself * h1 / r_safe
            strain = strain + jnp.einsum("j,jc,jd->cd", coef, w, w)

        if d.use_chi and (compute_force or compute_strain):
            wI = _md(src[None, :] - R, d)
            rI = jnp.linalg.norm(wI, axis=-1)
            rI_safe = jnp.where(rI > 0.0, rI, 1.0)
            _, h1, _, _ = _h3_b(Lchi, beta, rI, C)
            if compute_force:
                dR = dR - (h1 / rI_safe)[:, None] * wI
            if compute_strain:
                strain = strain + jnp.einsum("I,Ic,Id->cd", h1 / rI_safe, wI, wI)

        if f_on:
            f_dR, f_strain = _f_terms(src, ei, notself)
            dR = dR + f_dR
            strain = strain + f_strain
        if compute_strain and sd.pbc_flag:
            policy = _dtn_cutoff_policy_one_electron_value(d, all_e, n_up, src, ei)
            strain = strain + (policy / 3.0) * jnp.eye(3, dtype=strain.dtype)
        return dR, strain

    # The base-position terms depend only on the moved electron: evaluate them once per
    # electron, not once per mesh config (N_v configs share each electron).
    old_R_all, old_strain_all = jax.vmap(_at_position)(all_e, electron_ids)

    def _one(pt: jax.Array, ei: jax.Array) -> tuple[jax.Array, jax.Array]:
        new_R, new_strain = _at_position(pt, ei)
        return new_R - old_R_all[ei], new_strain - old_strain_all[ei]

    # Geometry carries the same dominant (N_atom,N_e) f coefficient grids as the drift mesh;
    # the callers stream ``_one`` against the DTN-owned per-walker budget
    # (``JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES``) through ``_mesh_wsum`` / ``_mesh_map``.
    return _one
