"""Bounded mesh-configuration streaming for the derivative kernels.

These helpers ``vmap`` a per-config wavefunction re-evaluation over a leading
mesh axis in fixed-size chunks (``vmap`` WITHIN a chunk, ``lax.scan`` ACROSS
chunks), so live memory is ``O(chunk)`` not ``O(mesh)``.  They are the shared,
estimator- and potential-agnostic streaming primitive used by:

  * ``ecp_force`` / ``one_move_mesh`` -- the ECP non-local quadrature mesh (the
    ``Ne x Nv`` config set),
  * ``derivatives.gfmc`` -- the LRDMC/GFMC LATTICE-DISCRETISATION S' mesh (the
    ``6 * Ne`` finite-difference kinetic hops ``+-alat`` along each axis).  This
    mesh exists for EVERY LRDMC system, all-electron INCLUDED -- it has nothing
    to do with pseudopotentials,
  * ``assemble`` -- the weighted O_k mesh reduction (``_mesh_wsum``).

Nothing here is ECP-specific; the module depends only on ``jax`` and the
streaming helpers of ``jqmc._setting``.

The chunk is NOT a tuning knob: it is the shared measured-budget width of
``jqmc._setting.measured_closure_stream_batch`` -- the per-config callee is
closure-converted, its workspace measured by XLA's compile-time memory analysis
and the selected width verified by compiling it -- against a PER-WALKER budget
(``DERIVATIVE_MESH_BUDGET_BYTES`` by default; the DTN meshes pass
``JASTROW_DTN_ONE_MOVE_MESH_BUDGET_BYTES``). These kernels run inside the
drivers' walker ``vmap``; the walker axis itself is bounded separately by
``WALKER_STREAM_BUDGET_BYTES``. A measurement failure raises (no fallback
width). Everything stays inside the jit boundary (scan under jit). Most
PBC/DTN/molecular real ECP meshes take the one_move_mesh rank-1 fast path (no
chunking); this streamed fallback governs complex-twist / non-(pade/exp)
Jastrow / GFMC S' meshes.
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

import jax
import jax.numpy as jnp

from jqmc._setting import DERIVATIVE_MESH_BUDGET_BYTES, budget_map, measured_closure_stream_batch


def _mesh_width(g: Callable[[object], object], xs: object, n: int, budget_bytes: int) -> int:
    """Measured mesh chunk width (``> n`` means one plain ``vmap``)."""
    one = jax.tree_util.tree_map(lambda a: a[0], xs)
    return measured_closure_stream_batch(g, (one,), n, budget_bytes, min_batch=1)


def _mesh_map(
    f: Callable[..., object],
    xs: object,
    budget_bytes: int = DERIVATIVE_MESH_BUDGET_BYTES,
) -> object:
    """``vmap`` ``f`` over the leading (mesh-config) axis of pytree ``xs`` in bounded
    chunks -- streaming, live memory ``O(chunk)`` not ``O(mesh)``.  ``xs`` is a tuple of
    per-config arrays; ``f`` receives one config's slices (``f(*slice)``).  Output stack
    is bit-identical to ``vmap(lambda x: f(*x))(xs)`` (per-config math is independent of
    the batch grouping; downstream reductions run over the full stack unchanged).

    Implemented by ``jqmc._setting.budget_map`` (an explicit ``lax.scan`` of ``vmap`` over
    padded chunks, which tolerates the zero-size leaves of O_k parameter-gradient pytrees).
    """
    return budget_map(f, xs, budget_bytes)


def _mesh_wsum(
    f: Callable[..., object],
    w: jax.Array,
    xs: object,
    budget_bytes: int = DERIVATIVE_MESH_BUDGET_BYTES,
) -> object:
    """Weighted mesh reduction ``sum_g w_g * f(*xs_g)`` ACCUMULATED in the scan carry.

    For mesh maps whose PER-CONFIG output is itself large (the O_k parameter-gradient
    pytree: lambda/MO/J3 blocks are ~N_AO^2 per config), even the chunk-streamed STACK
    is mesh-sized (960 configs x 13 MB leaves = the water e_L/c 138 GB OOM class).  The
    bounded form never stacks: live memory = the running sum + one chunk of outputs.
    ``w`` is the (n,) per-config weight (complex allowed); ``f`` returns a pytree.
    Padded tail configs get weight 0 (their f-value is a finite copy of config 0, so
    ``0 * f == 0`` exactly).  Same per-config math as ``einsum('g,g...', w, vmap(f))``;
    only the reduction grouping differs (fp-benign chunk-wise reassociation).
    """
    g = lambda x: f(*x)  # noqa: E731
    tm = jax.tree_util.tree_map
    leaves = jax.tree_util.tree_leaves(xs)
    n = int(leaves[0].shape[0])
    b = _mesh_width(g, xs, n, budget_bytes)
    w = jnp.asarray(w)
    if n <= b:
        ys = jax.vmap(g)(xs)
        return tm(lambda y: jnp.einsum("g,g...->...", w, y), ys)
    n_pad = (-n) % b
    n_ch = (n + n_pad) // b

    def _pad(a: jax.Array, zero: bool) -> jax.Array:
        if n_pad == 0:
            return a
        tail = jnp.zeros_like(a[:1]) if zero else a[:1]
        return jnp.concatenate([a, jnp.broadcast_to(tail, (n_pad,) + a.shape[1:])], axis=0)

    xs_r = tm(lambda a: _pad(a, False).reshape((n_ch, b) + a.shape[1:]), xs)
    w_r = _pad(w, True).reshape(n_ch, b)

    def _step(carry: object, ch: tuple[jax.Array, object]) -> tuple[object, None]:
        chw, chx = ch
        ys = jax.vmap(g)(chx)
        return tm(lambda c, y: c + jnp.einsum("g,g...->...", chw, y), carry, ys), None

    # carry template: the output structure of one config, zeroed, in the w-promoted dtype.
    y0 = jax.eval_shape(g, tm(lambda a: a[0], xs))
    carry0 = tm(lambda y: jnp.zeros(y.shape, dtype=jnp.result_type(y.dtype, w.dtype)), y0)
    out, _ = jax.lax.scan(_step, carry0, (w_r, xs_r))
    return out
