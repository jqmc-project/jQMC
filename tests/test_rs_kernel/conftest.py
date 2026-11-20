"""Shared helpers for the Rust CPU-kernel equivalence suite.

These tests assert that the Rust analytic kernels (`import jqmc_rs`) reproduce
jqmc's XLA analytic derivatives / value functions in both full and mixed
precision on COMMITTED fixtures
(`tests/trexio_example_files`, `tests/qe_example_files`) -- NO external bench h5.
The parent `tests/conftest.py`
supplies the `light`/`heavy` markers, `--skip-heavy`, `with_chdir`, and the
`jax.distributed` session init; this conftest only adds the fixture loaders.

`import jqmc_rs` is UNCONDITIONAL in every module here: if the wheel is not
built the modules fail collection (intentional -- a missing Rust build is a hard
error in this suite, never a silent skip).
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

import collections
import contextlib
import os

import jax

from tests._test_data_cache import qe_example_files_root

jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    """Configure precision before collection and cap the Rust Rayon pool.

    The root per-test fixture configures precision during test setup, which is
    too late for the few parity modules that resolve a tolerance at import
    time. Configure the CLI-selected mode here as well so those module-level
    values do not silently bake in the default full-precision tier during a
    mixed run. The per-test fixture remains responsible for resetting state
    after tests that deliberately mutate the precision map.

    The XLA=Rust parity suite runs ``num_walkers=2`` fixtures, so with >=2 rayon
    threads it exercises the multi-thread walker-batch path -- the verification of
    the parallel route (there are no thread-specific permanent contract tests). A shared
    machine is oversubscribed under ``-n``, so the session is capped at 4 rayon threads (still >=2).

    This runs at session start, well BEFORE the first ``jqmc_rs`` parallel call --
    the hard requirement for ``configure_threads`` (``rayon::build_global`` succeeds
    once per process). Under xdist it runs once per worker process, each of which
    owns its own rayon pool. ``setdefault`` lets a developer override the width via
    the ``JQMC_RS_NUM_THREADS`` env (also making xdist workers inherit it) while
    defaulting the session to 4. The single ``jqmc_rs threads: N`` log line is emitted
    from ``pytest_report_header`` (controller only), which prints in the session
    header block; ``write_line`` here in ``pytest_configure`` runs before the
    reporter session starts and would be swallowed.
    """
    import jqmc_rs
    from jqmc._precision import configure, precision_policy_snapshot

    configure(config.getoption("--precision-mode"))
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    n = int(os.environ.setdefault("JQMC_RS_NUM_THREADS", "4"))
    jqmc_rs.configure_threads(n)


def pytest_report_header(config):
    """One-line rayon-width banner for the rs test session."""
    import jqmc_rs

    return f"jqmc_rs threads: {jqmc_rs.current_num_threads()}"


# --------------------------------------------------------------------------- #
# Native-execution spy
# --------------------------------------------------------------------------- #
# Several "flag ON vs flag OFF" tests built their ON side purely by DELETING
# ``JQMC_FLAG_RUST_ON_CPU`` and relying on the default being True, with nothing
# asserting that native code actually ran. If the default flips, or a new
# capability gate silently sends the family back to JAX, both sides execute the
# SAME XLA code and every assertion in the test passes vacuously -- the test
# turns into a tautology without going red.
#
# The remedy already in use by ``test_rust_dispatch_equivalence`` is a spy on the
# native observable context. This is the same instrument, generalised: the counter
# is keyed by ctx METHOD NAME (rather than a hand-maintained family map), so it
# serves the MCMC observable methods (``run_energy_chunk`` / ``mcmc_sweep`` /
# ``e_L`` / ``grad_*``) and the GFMC interval methods
# (``run_gfmc_projection_t`` / ``_n``) with one implementation, and a newly added
# native entry point is counted automatically instead of being invisible.


# NaN-agreement guard: this directory installs no autouse ``assert_allclose``
# wrapper of its own; it relies on the suite-wide guard in
# ``tests/_nan_guard.py`` (installed from ``tests/conftest.py`` before any test
# module is imported), which also covers module-level
# ``from numpy.testing import assert_allclose`` bindings, ``assert_array_equal`` and
# the decimal comparators, and restricts explicit ``equal_nan=True`` to a reviewed
# allow-list.


class _CountingCtx:
    """Proxy around a native obs-ctx that counts calls per method name."""

    def __init__(self, inner, counts):
        self._inner = inner
        self._counts = counts

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _counted(*args, **kwargs):
            self._counts[name] += 1
            return attr(*args, **kwargs)

        return _counted


@contextlib.contextmanager
def native_ctx_spy():
    """Count every native obs-ctx method call made inside the block.

    Yields a ``collections.Counter`` keyed by ctx method name. ``sum(counts.values())``
    is the total number of native kernel invocations, so ``== 0`` positively proves the
    run stayed on XLA and ``> 0`` positively proves it did not -- which is exactly the
    claim a flag-ON/flag-OFF pair needs.

    Not a fixture: it is a context manager so a test can wrap ONE side of an A/B pair
    (the ON run) without the patch leaking into the reference run.
    """
    import jqmc_rs

    counts: collections.Counter = collections.Counter()
    orig_build = jqmc_rs.build_obs_ctx

    def _spy_build(*args, **kwargs):
        return _CountingCtx(orig_build(*args, **kwargs), counts)

    jqmc_rs.build_obs_ctx = _spy_build
    try:
        yield counts
    finally:
        jqmc_rs.build_obs_ctx = orig_build


def assert_native_ran(counts, context):
    """The flag-ON side ACTUALLY executed native kernels."""
    assert sum(counts.values()) > 0, (
        f"{context}: the flag-ON side executed ZERO native obs-ctx calls -- the ON and OFF "
        "sides both ran the JAX path, so every downstream assertion in this test is "
        "vacuous. Either the rust-on default changed or a capability gate silently sent "
        f"this family back to XLA. counts={dict(counts)}"
    )


def assert_no_native_ran(counts, context):
    """The flag-OFF / jax-mode side executed NO native kernel (the A/B pair is real)."""
    assert sum(counts.values()) == 0, (
        f"{context}: the flag-OFF side executed native obs-ctx calls, so it is not an "
        f"independent XLA reference. counts={dict(counts)}"
    )


# Committed-fixture roots (never bench/).
_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


def trexio_path(name: str) -> str:
    """Absolute path to a committed trexio fixture (e.g. ``H2_ae_ccpvdz_cart.h5``)."""
    return os.path.abspath(os.path.join(_TREXIO, name))


def qe_dir(name: str) -> str:
    """Absolute path to a committed QE-projection fixture dir (e.g. ``Li_gamma``)."""
    # Resolved per call: this conftest can be imported before tests/conftest.py
    # builds the session mirror.
    return os.path.abspath(os.path.join(qe_example_files_root(), name))
