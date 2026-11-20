"""Route-engagement probe for the projection-scoped mesh caches.

Every projection-scoped cache on the XLA path engages through a Python-static
``if <state> is None`` branch.  A missing cache is therefore *invisible*: the
uncached branch returns the same numbers, only slower.  Numerical parity alone
therefore cannot distinguish an engaged cache from a wiring omission.  The
analytic force/stress path is deliberately uncached because it performs a
single evaluation rather than a projection sequence.

This module turns the branch choice into an observable, and -- under
``require_mesh_cache()`` -- into a loud failure.  It is the XLA twin of the
Rust ``gfmc_route_counters`` / ``gfmc_route_counters_reset`` pair
(``jqmc/rs/src/lib.rs``), which the Rust route-parity tests also assert on.

Trace-time semantics (important for anyone writing a gate test)
--------------------------------------------------------------
The ``is None`` branches are resolved while ``jax.jit`` *traces*, not while the
compiled graph *runs*.  A counter therefore records which route was COMPILED,
once per traced signature -- not once per call.  A test that asserts on the
counters must clear the jit caches first (``jax.clear_caches()``).  Without
that reset, a signature compiled by an earlier test is not re-traced, so no
counter moves and an assertion can produce a false green.

Channels are the four Jastrow legs plus the determinant; meshes are the two
candidate lattices that carry a projection-scoped state (the 6N LRDMC kinetic
lattice and the ECP quadrature lattice).
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

import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "CHANNELS",
    "MESHES",
    "SITES",
    "MeshCacheMissing",
    "note_route",
    "route_counters",
    "route_counters_reset",
    "route_site",
    "require_mesh_cache",
    "strict_mesh_cache_enabled",
]

# Channels that can carry a projection-scoped candidate cache.
CHANNELS = ("det", "j1", "j2", "j3", "dtn")
# Candidate lattices that carry such a state.
MESHES = ("kinetic", "ecp")
# WHICH caller took the route. Counters are trace-time, so a raw uncached count
# cannot distinguish "once per interval by design" from "every sub-step": both
# read as 1. The site label restores that distinction, and is what lets a gate
# say "the FORCE path must not be uncached" without also condemning the
# deliberately-uncached substep-0 warm-up.
#   projection -- the per-sub-step projection kernels (must be cached)
#   substep0   -- GFMC_t's first sub-step of an interval, uncached BY DESIGN
#                 (it is what builds the state the rest of the interval reuses)
#   observable -- the measurement-point e_L evaluation
#   force      -- the analytic force/stress derivative path
#   mcmc       -- MCMC single-move (no mesh; reported for completeness)
SITES = ("projection", "substep0", "observable", "force", "mcmc", "unlabelled")

_ENV_STRICT = "JQMC_REQUIRE_MESH_CACHE"

_counters: Counter[tuple[str, str, bool, str]] = Counter()
# Nesting depth of ``require_mesh_cache()``; the env var acts as depth 1.
_strict_depth = 0
# Label applied by the innermost enclosing ``route_site()`` block.
_site = "unlabelled"


class MeshCacheMissing(RuntimeError):
    """A projection-scoped mesh cache was absent where one was required.

    Raised at TRACE time, so the traceback points at the kernel that compiled
    the uncached route -- not at a later numerical symptom (there is none).
    """


def strict_mesh_cache_enabled() -> bool:
    """True when an uncached mesh route must raise instead of being counted."""
    if _strict_depth > 0:
        return True
    return os.environ.get(_ENV_STRICT, "") not in ("", "0", "false", "False")


def note_route(channel: str, mesh: str, cached: bool, site: str | None = None) -> None:
    """Record which route a mesh kernel compiled for ``channel`` on ``mesh``.

    Call this from BOTH sides of the ``is None`` branch, so a green gate means
    "the cached branch was compiled", never "the kernel was not reached".
    A channel with no cache implementation at all calls it with
    ``cached=False`` unconditionally -- that is a gap, and the gate should say
    so rather than stay silent.

    Args:
        channel: One of :data:`CHANNELS`.
        mesh: One of :data:`MESHES`.
        cached: Whether the projection-scoped candidate cache was consumed.
        site: One of :data:`SITES` -- which caller took this route. ``None``
            (the default) takes the ambient label from the enclosing
            :func:`route_site` block, or ``"unlabelled"`` outside one, which a
            gate should treat as un-auditable rather than as passing.

    Raises:
        MeshCacheMissing: If ``cached`` is False under strict mode.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown route channel {channel!r} (expected one of {CHANNELS})")
    if mesh not in MESHES:
        raise ValueError(f"unknown route mesh {mesh!r} (expected one of {MESHES})")
    site = _site if site is None else site
    if site not in SITES:
        raise ValueError(f"unknown route site {site!r} (expected one of {SITES})")
    _counters[(channel, mesh, cached, site)] += 1
    if not cached and strict_mesh_cache_enabled():
        raise MeshCacheMissing(
            f"{channel} on the {mesh} mesh compiled the UNCACHED route at site {site!r} "
            f"while {_ENV_STRICT} / require_mesh_cache() was active"
        )


def route_counters() -> dict[tuple[str, str, bool, str], int]:
    """Snapshot of the trace-time route counters.

    Keys are ``(channel, mesh, cached, site)``.  Absent keys mean the kernel was
    never traced -- which is NOT the same as "the cached route ran"; see the
    module docstring on clearing jit caches.
    """
    return dict(_counters)


def route_counters_reset() -> None:
    """Zero the counters (start of a measurement window)."""
    _counters.clear()


@contextmanager
def route_site(name: str) -> Iterator[None]:
    """Label every route reported inside this block with the caller ``name``.

    The mesh entry points are ``@jit``-decorated, so the site cannot ride in as
    an argument (a string parameter would have to be declared static on every
    one of them). A context established around the CALL works instead, because
    ``note_route`` fires while that call traces.

    Nests; the innermost label wins. Leaves the label unchanged for anything
    traced outside a block, which is what ``"unlabelled"`` reports.
    """
    global _site
    prev = _site
    if name not in SITES:
        raise ValueError(f"unknown route site {name!r} (expected one of {SITES})")
    _site = name
    try:
        yield
    finally:
        _site = prev


@contextmanager
def require_mesh_cache() -> Iterator[None]:
    """Make any uncached mesh route raise :class:`MeshCacheMissing`.

    Nests, and composes with the ``JQMC_REQUIRE_MESH_CACHE`` env var (which
    behaves as a permanently-entered outermost context).
    """
    global _strict_depth
    _strict_depth += 1
    try:
        yield
    finally:
        _strict_depth -= 1
